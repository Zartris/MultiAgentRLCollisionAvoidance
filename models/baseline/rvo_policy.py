import numpy as np
import rvo2
import torch
from torch import nn

from scenario.CollisionAvoidance_base import observation_to_dict

### RVO AGENTS
RVO_TIME_HORIZON = 5.
RVO_COLLAB_COEFF = 0.5
RVO_ANTI_COLLAB_T = 1.0


# NH-OCRA
class RVOPolicy(nn.Module):
    def __init__(self, vmas_env, dt, max_neighbors, neighbor_dist, time_horizon=RVO_TIME_HORIZON,
                 dynamics={}):
        super(RVOPolicy, self).__init__()  # for the wrapper
        self.env = vmas_env
        self.dt = dt
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

        # TODO share this parameter with environment
        horizon = time_horizon  # NOTE: bjorn used 1.0 in training for corl19
        # Initialize RVO simulator
        self.sim = {}
        self.rvo_agents = {}
        for i in range(self.num_env):
            self.sim[i] = rvo2.PyRVOSimulator(timeStep=self.dt, neighborDist=neighbor_dist,
                                              maxNeighbors=max_neighbors, timeHorizon=time_horizon,
                                              timeHorizonObst=time_horizon, radius=self.agent_radius,
                                              maxSpeed=self.max_speed)
            self.rvo_agents[i] = [None] * self.n_agents

        self.is_init = False

        self.use_non_coop_policy = True

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

    def init(self):
        state_dim = 2
        self.pos_agents = np.empty((self.num_env, self.n_agents, state_dim))
        self.rot_agents = np.empty((self.num_env, self.n_agents))
        self.vel_agents = np.empty((self.num_env, self.n_agents, state_dim))
        self.goal_agents = np.empty((self.num_env, self.n_agents, state_dim))
        self.pref_vel_agents = np.empty((self.num_env, self.n_agents, state_dim))
        self.pref_speed_agents = np.empty((self.num_env, self.n_agents))

        # Init simulation
        for i in range(self.num_env):
            for a in range(self.n_agents):
                self.rvo_agents[i][a] = self.sim[i].addAgent((0, 0))

            # print("obstacle positions:")
            for obstacle in self.env.world.landmarks:
                if not obstacle.collide:
                    continue

                # !!!IMPORTANT!!!: THESE HAVE TO BE COUNTERCLOCKWISE
                vert = np.array(obstacle.shape.get_geometry().v)  # list of tuples
                if self.is_clockwise(vert):
                    vert = vert[::-1]

                # translate and rotate the obstacle
                translate = obstacle.state.pos[i].cpu().numpy()  # x, y
                rotate = obstacle.state.rot[i].item()  # radians
                vert = vert @ np.array([[np.cos(rotate), -np.sin(rotate)], [np.sin(rotate), np.cos(rotate)]])
                vert += translate

                # back to list of tuples
                vert_tuple = list(map(tuple, vert))
                self.sim[i].addObstacle(vert_tuple)
            self.sim[i].processObstacles()

        self.is_init = True

    @staticmethod
    def is_clockwise(polygon):
        """
        Check if the polygon vertices are ordered in a clockwise direction.

        Args:
        - polygon: List of (x, y) tuples representing the polygon vertices.

        Returns:
        - True if the vertices are in clockwise order, False if counterclockwise.
        """
        sum = 0
        n = len(polygon)

        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]
            sum += (x2 - x1) * (y2 + y1)

        return sum > 0

    def update_world(self, goals):
        for i, agent in enumerate(self.env.world.agents):
            for world in range(self.num_env):
                self.pos_agents[world, i, :] = agent.state.pos[world].cpu().numpy()
                self.rot_agents[world, i] = agent.state.rot[world].item()
                self.vel_agents[world, i, :] = agent.state.vel[world].cpu().numpy()

                # adding noise:
                self.pos_agents[world, i, :] += np.random.uniform(-0.1, 0.1, agent.state.pos[world].shape)
                self.vel_agents[world, i, :] += np.random.uniform(-0.1, 0.1, agent.state.vel[world].shape)

                self.goal_agents[world, i, :] = goals[world, i].cpu().numpy()
                self.pref_speed_agents[world, i] = self.max_speed * (1 - agent.on_goal[world].int().cpu().numpy())
                # Calculate preferred velocity
                # Assumes non RVO agents are acting like RVO agents
                self.pref_vel_agents[world, i, :] = self.goal_agents[world, i, :] - self.pos_agents[world, i, :]
                self.pref_vel_agents[world, i, :] = self.pref_speed_agents[world, i] / np.linalg.norm(
                    self.pref_vel_agents[world, i, :]) * self.pref_vel_agents[world, i, :]

                # Set agent positions and velocities in RVO simulator
                self.sim[world].setAgentMaxSpeed(self.rvo_agents[world][i], self.pref_speed_agents[world, i])
                self.sim[world].setAgentRadius(self.rvo_agents[world][i], 1.5 * self.agent_radius)
                self.sim[world].setAgentPosition(self.rvo_agents[world][i], tuple(self.pos_agents[world, i, :]))
                self.sim[world].setAgentVelocity(self.rvo_agents[world][i], tuple(self.vel_agents[world, i, :]))
                self.sim[world].setAgentPrefVelocity(self.rvo_agents[world][i],
                                                     tuple(self.pref_vel_agents[world, i, :]))
                self.sim[world].setAgentCollabCoeff(self.rvo_agents[world][i], RVO_COLLAB_COEFF)

    def dump(self, obs):
        obs_dict = observation_to_dict(obs, False, None, None, True, num_agents=self.n_agents)
        other_agents = obs_dict["other_agents"]
        ego_agent = obs_dict["state"]

        agent_state = obs_dict["state"]
        agent_pos = agent_state[..., :2]
        agent_rot = agent_state[..., 2:3]
        agent_vel = agent_state[..., 3:5]
        agent_vel = torch.linalg.vector_norm(agent_vel, dim=-1).unsqueeze(-1)
        agent_ang_vel = agent_state[..., 5:6]

        global_path = obs_dict["global_path"]
        goal_global = global_path[
                      ..., :2
                      ]  # x,y, dist, angle_to_point_diff, angle_diff_to_next

        # Convert to local coordinates and then to polar coordinates
        global_position = other_agents[..., :2]
        velocities = other_agents[..., 2:]  # velocities are already relative

    def forward(self, obs: torch.tensor):
        if not self.is_init:
            self.init()  # setting obstacles
        obs_dict = observation_to_dict(obs, False, None, None, True, num_agents=self.n_agents)
        global_path = obs_dict["global_path"]
        goal_global = global_path[
                      ..., :2
                      ]  # x,y, dist, angle_to_point_diff, angle_diff_to_next

        self.update_world(goal_global)
        actions = np.zeros((self.num_env, self.n_agents, 2))
        for i in range(self.num_env):
            # Set ego agent's collaborativity
            # if RVO_COLLAB_COEFF < 0:
            #     # agent is anti-collaborative ==> every X seconds, it chooses btwn non-coop and adversarial,
            #     # where the PMF of which policy to run is defined by abs(collab_coeff)\in(0,1].

            #     # if a certain freq, randomly select btwn use non coop policy vs. rvo
            #     if round(agents[agent_index].t % RVO_ANTI_COLLAB_T, 3) < self.dt or \
            #             round(RVO_ANTI_COLLAB_T - self.rvo_agents[agent_index].t % RVO_ANTI_COLLAB_T, 3) < self.dt:
            #         self.use_non_coop_policy = np.random.choice([True, False], p=[1 - abs(RVO_COLLAB_COEFF),
            #                                                                         abs(RVO_COLLAB_COEFF)])
            #     if self.use_non_coop_policy:
            #         self.sim[i].setAgentCollabCoeff(self.rvo_agents[agent_index], 0.0)
            #     else:
            #         self.sim[i].setAgentCollabCoeff(self.rvo_agents[agent_index], RVO_COLLAB_COEFF)
            # else:
            #     self.sim[i].setAgentCollabCoeff(self.rvo_agents[i][agent_index], RVO_COLLAB_COEFF)
            self.sim[i].doStep()
            for agent_id in range(self.n_agents):
                # Calculate desired change of heading
                new_rvo_pos = self.sim[i].getAgentPosition(self.rvo_agents[i][agent_id])[:]
                deltaPos = new_rvo_pos - self.pos_agents[i, agent_id, :]
                p1 = deltaPos
                p2 = np.array([1, 0])  # Angle zero is parallel to x-axis
                ang1 = np.arctan2(*p1[::-1])
                ang2 = np.arctan2(*p2[::-1])
                new_heading_global_frame = (ang1 - ang2) % (2 * np.pi)
                delta_heading = wrap(new_heading_global_frame - self.rot_agents[i, agent_id])

                # Calculate desired speed
                pref_speed = 1 / self.dt * np.linalg.norm(deltaPos)

                # Limit the turning rate: stop and turn in place if exceeds
                if abs(delta_heading) > self.max_delta_heading:
                    delta_heading = np.sign(delta_heading) * self.max_delta_heading
                    pref_speed = 0.

                # Ignore speed
                if self.has_fixed_speed:
                    pref_speed = self.max_speed

                # Add noise
                if self.heading_noise:
                    delta_heading = delta_heading + np.random.normal(0, 0.5)

                action = np.array([pref_speed, delta_heading])
                actions[i, agent_id] = action
        return torch.tensor(actions, dtype=torch.float32, device=obs.device)


# keep angle between [-pi, pi]
def wrap(angle):
    while angle >= np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle
