import time

import numpy as np
import torch
import zmq
from torch import nn

from models.baseline.GA3C_CADRL.agent import Agent
from scenario.CollisionAvoidance_base import observation_to_dict
import requests
import json


# Initialize ZeroMQ context and socket
context = zmq.Context()
socket = context.socket(zmq.REQ)  # REQ for request mode
socket.connect("tcp://127.0.0.1:5555")  # Connect to the server

def get_prediction(features):
    socket.send_string(json.dumps(features))  # Send request to server
    message = socket.recv()  # Wait for the reply
    return json.loads(message)['predictions']  # Return the predictions


# NH-OCRA
class GA3CPolicy(nn.Module):
    def __init__(self, vmas_env, dynamics={}, object_vert_inflate_radius=0.15):
        super(GA3CPolicy, self).__init__()  # for the wrapper
        self.env = vmas_env
        self.dt = vmas_env.scenario.dt
        # neighbor_dist = Config.SENSING_HORIZON
        # max_neighbors = Config.MAX_NUM_AGENTS_IN_ENVIRONMENT
        self.n_agents = self.env.n_agents
        self.num_env = self.env.batch_dim

        self.agent_radius = self.env.world.agents[0].shape.radius

        self.has_fixed_speed = False
        self.heading_noise = False

        self.dynamics = dynamics
        self.max_delta_heading = self.dynamics.get("omega_limit")
        self.max_speed = self.dynamics.get("v_limit")
        self.is_init = False
        self.object_vert_inflate_radius = object_vert_inflate_radius

        debug = 0

    def interpolate_vertices(self, vertices, max_distance):
        """
        Interpolates points between consecutive vertices so that the distance
        between any two points is no greater than max_distance.

        Parameters:
        vertices (numpy.ndarray): Array of shape (N, 2) representing the vertices.
        max_distance (float): The maximum allowed distance between points.

        Returns:
        numpy.ndarray: Array of interpolated vertices.
        """

        def interpolate_between_points(p1, p2, max_distance):
            distance = np.linalg.norm(p2 - p1)
            if distance <= max_distance:
                return [p1, p2]

            num_points = int(np.ceil(distance / max_distance))
            return [p1 + (p2 - p1) * i / num_points for i in range(num_points + 1)]

        interpolated_vertices = []
        for i in range(len(vertices) - 1):
            p1, p2 = vertices[i], vertices[i + 1]
            interpolated_segment = interpolate_between_points(p1, p2, max_distance)
            interpolated_vertices.extend(interpolated_segment[:-1])  # Exclude last point to avoid duplicates

        # Add the final point
        interpolated_vertices.append(vertices[-1])

        return np.array(interpolated_vertices)

    def init_env(self):
        state_dim = 2
        self.obstacles_info = np.empty((self.num_env, self.n_agents, state_dim))
        self.object_map = {}
        # Init simulation
        for i in range(self.num_env):
            objects = []
            for obstacle in self.env.world.landmarks:
                if not obstacle.collide:
                    continue
                vert = np.array(obstacle.shape.get_geometry().v)  # list of tuples
                # vert_loop = np.append(vert, [vert[0]], axis=0)
                vert_loop = self.interpolate_vertices(vert, self.object_vert_inflate_radius * 2)
                # translate and rotate the obstacle
                translate = obstacle.state.pos[i].cpu().numpy()  # x, y
                rotate = obstacle.state.rot[i].item()  # radians
                vert_loop = vert_loop @ np.array([[np.cos(rotate), -np.sin(rotate)], [np.sin(rotate), np.cos(rotate)]])
                vert_loop += translate

                # back to list of tuples
                vert_tuple = list(map(tuple, vert_loop))
                for x, y in vert_tuple:
                    # a = Agent(x, y, 0, 0, radius=self.object_vert_inflate_radius, pref_speed=0, id=object_id)
                    o = [float(x),
                         float(y),
                         self.object_vert_inflate_radius,
                         0.,
                         0.]  # we need to send it so that the server can understand it
                    objects.append(o)
            self.object_map[i] = objects
        self.is_init = True

    def forward(self, obs: torch.tensor):
        if not self.is_init:
            self.init_env()

        obs = obs.cpu().numpy()
        obs_dict = observation_to_dict(obs, False, None, None, True, num_agents=self.n_agents)

        # TODO: Should we use the env info or the obs_dict?
        # Take agents info from obs_dict as they are noisy and the env info for the objects
        actions = np.zeros((self.num_env, self.n_agents, 2))
        for world_id in range(self.num_env):
            for agent_id in range(self.n_agents):
                # make own state
                state = obs_dict["state"][world_id, agent_id, :6]  # x,y, rot, vel_x, vel_y, ang_vel
                goal = obs_dict["global_path"][world_id, agent_id, :2]  # goal_x, goal_y
                this_agent = [float(state[0]),
                              float(state[1]),
                              float(goal[0]),
                              float(goal[1]),
                              self.max_speed,
                              self.agent_radius,
                              float(state[2]),
                              float(state[3]),
                              float(state[4])]  # x, y, goal_x, goal_y, pref_speed, radius, heading_angle, v_x, v_y
                # make other agents state:
                other_agents_list = []
                if "other_agents" in obs_dict:
                    other_agents = obs_dict["other_agents"][world_id * self.n_agents + agent_id]  # x,y, vel_x, vel_y
                    for other_agent in other_agents:
                        other_agents_list.append(
                            [float(other_agent[0]),
                             float(other_agent[1]),
                             self.agent_radius,
                             float(other_agent[2]),
                             float(other_agent[3])])  # x, y, radius, v_x, v_y
                # add objects
                objects = self.object_map[world_id]
                other_agents_list.extend(objects)
                state_to_send = {
                    "agent_state": this_agent,
                    "other_agents": other_agents_list
                }
                predictions = get_prediction(state_to_send)
                if predictions is not None:
                    actions[world_id, agent_id] = np.array(predictions)
                else:
                    print("Error in getting predictions")
                    exit(1)
        # actions = np.clip(actions, -1, 1)
        return torch.tensor(actions, dtype=torch.float32, device=self.env.device)


# keep angle between [-pi, pi]
def wrap(angle):
    while angle >= np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle
