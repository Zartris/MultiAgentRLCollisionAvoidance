import math
import typing
from typing import List, Optional

import numpy as np
import torch as th
from vmas.simulator.core import World, Landmark, Box, Sphere
from vmas.simulator.utils import Color, ScenarioUtils

from scenario.CollisionAvoidance_base import CollisionAvoidance, point_to_goal, nget
from scenario.GlobalPlanner.global_planner import AStarPlanner

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom
    from typing import List

scenario_type = "room"


class ScenarioRoomSpawner:
    @staticmethod
    def make_objects(
            world: World,
            world_radius: float,
            spawn_size_p: float,
            num_obstacles: int,
            obstacle_size_p: float,
            agent_radius: float,
            buffer: float,
    ):
        # We are splitting this into 3 spawn, obstacle zone, goal zone
        ar_buf = agent_radius + buffer + 0.25
        spawn_size = spawn_size_p * world_radius * 2
        obstacle_zone_radius = world_radius - spawn_size
        spawn_zones = {
            0: [
                -world_radius + ar_buf,
                world_radius - ar_buf,  # start zone
                -world_radius + ar_buf,
                -world_radius + spawn_size - ar_buf,
            ],  # x,y
            1: [
                -world_radius + ar_buf,
                world_radius - ar_buf,  # goal zone
                world_radius - spawn_size + ar_buf,
                world_radius - ar_buf,
            ],  # x,y
            2: [
                -world_radius + 1.5,
                world_radius - 1.5,  # obstacle zone
                -obstacle_zone_radius,
                obstacle_zone_radius,
            ],
        }
        # corridor walls:
        room_walls: List[Landmark] = []
        for i in range(4):
            if i < 2:  # boxes for making the room
                shape = Box(world_radius * 2, 0.5)
            else:
                shape = Box(0.5, world_radius * 2)

            obstacle = Landmark(
                name=f"room_wall_{i}_{scenario_type}",
                collide=True,
                shape=shape,
                color=Color.BLACK.value,
                collision_filter=lambda e: e.movable,
            )
            world.add_landmark(obstacle)
            room_walls.append(obstacle)

        num_obstacles = 10
        obstacle_size = obstacle_size_p * world_radius * 2
        obstacles: List[Landmark] = []
        for i in range(num_obstacles):
            shape = Box(obstacle_size, 0.2)
            # shape = Sphere(obstacle_size / 4)
            obstacle = Landmark(
                name=f"obstacle_wall_{i}_{scenario_type}",
                collide=True,
                shape=shape,
                color=Color.BLACK.value,
                collision_filter=lambda e: e.movable,
            )
            world.add_landmark(obstacle)
            obstacles.append(obstacle)
        return room_walls, obstacles, spawn_zones

    @staticmethod
    def spawn(
            env_index,
            world: World,
            room_walls: List[Landmark],
            obstacle_list: List[Landmark],
            spawn_zones,
            world_radius: float,
            obstacle_size_p: float,
            agent_radius: float,
            buffer: float,
            spawn_offset: Optional[th.Tensor] = None,
    ):
        batch_dim = world.batch_dim if env_index is None else 1
        if spawn_offset is None:
            spawn_offset = th.zeros(2, device=world.device)
        spawn_offset = spawn_offset.to(world.device)
        room_len = world_radius * 2

        # top and bottom
        y = room_len / 2
        x = 0
        room_walls[0].set_pos(
            th.tensor([x, y], device=world.device) + spawn_offset, env_index
        )
        room_walls[1].set_pos(
            th.tensor([-x, -y], device=world.device) + spawn_offset, env_index
        )

        # plug ends
        x = room_len / 2
        y = 0
        room_walls[2].set_pos(
            th.tensor([x, y], device=world.device) + spawn_offset, env_index
        )
        room_walls[3].set_pos(
            th.tensor([-x, -y], device=world.device) + spawn_offset, env_index
        )

        # Batched per-zone placement (see scenario/spawn_utils.py). Agents, goals
        # and obstacles each get their own wall-free zone (0/1/2) with only
        # within-group spacing, so they need no cross-avoidance and the boundary
        # clamp keeps them off the walls. Replaces vmas rejection sampling.
        from scenario.spawn_utils import build_required_dist, relax_positions

        agents = [a for a in world.agents if a.name.endswith(scenario_type)]
        if agents:
            d = 2 * agent_radius + buffer
            req = build_required_dist([len(agents)], {(0, 0): d}, world.device)
            ax0, ax1, ay0, ay1 = spawn_zones[0]
            gx0, gx1, gy0, gy1 = spawn_zones[1]
            agent_pos, _ = relax_positions(
                req, (ax0, ax1), (ay0, ay1), batch_dim, world.device
            )
            goal_pos, _ = relax_positions(
                req, (gx0, gx1), (gy0, gy1), batch_dim, world.device
            )
            for k, agent in enumerate(agents):
                agent.set_pos(agent_pos[:, k] + spawn_offset, env_index)
                agent.goal.set_pos(goal_pos[:, k] + spawn_offset, env_index)
                new_rot = point_to_goal(agent)
                agent.set_rot(
                    new_rot if env_index is None else new_rot[env_index], env_index
                )

        obs_size = obstacle_size_p * world_radius * 2
        if obstacle_list:
            obs_req = build_required_dist(
                [len(obstacle_list)], {(0, 0): obs_size}, world.device
            )
            ox0, ox1, oy0, oy1 = spawn_zones[2]
            obs_pos, _ = relax_positions(
                obs_req, (ox0, ox1), (oy0, oy1), batch_dim, world.device
            )
            for k, obstacle in enumerate(obstacle_list):
                obstacle.set_pos(obs_pos[:, k] + spawn_offset, env_index)


class CollisionAvoidanceRoom(CollisionAvoidance):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def make_world(self, batch_dim: int, device: th.device, **kwargs) -> World:
        """
        This function needs to be implemented when creating a scenario
        In this function the user should instantiate the world and insert agents and landmarks in it

        Args:
        :param batch_dim: the number of environments to step parallely
        :param device: the torch device to use
        :param kwargs: named arguments passed during environment creation
        :return world: returns the instantiated world which is automatically set in 'self.world'
        """
        world = super().make_world(batch_dim, device, **kwargs)
        self.buffer = 0.3
        self.corridor_length = self.world_radius * 2
        self.spawn_size_p = self.config.get("spawn_size_p", 0.2)
        self.obstacle_size_p = self.config.get("obstacle_size_p", 0.15)
        self.num_obstacles = self.config.get("num_obstacles", 0)
        self.room_walls, self.obstacles, self.spawn_zones = (
            ScenarioRoomSpawner.make_objects(
                world,
                self.world_radius,
                self.spawn_size_p,
                self.num_obstacles,
                self.obstacle_size_p,
                self.agent_radius,
                self.buffer,
            )
        )
        self.static_objects = self.room_walls + self.obstacles
        self.gp = None
        self.inflate_radius = self.agent_radius + 0.16
        if nget(self.config, ("use_global_path",), False):
            self.gp = AStarPlanner(
                self.world_size,
                10,
                self.inflate_radius,
                self.batch_dim,
                self.num_agents,
                device,
            )

        return world

    def spawn_room(self, env_index: int):
        ScenarioRoomSpawner.spawn(
            env_index,
            self.world,
            self.room_walls,
            self.obstacles,
            self.spawn_zones,
            self.world_radius,
            self.obstacle_size_p,
            self.agent_radius,
            self.buffer,
        )

    def reset_world_at(self, env_index: int = None):
        if self.batch_dim == 1:
            env_index = [0]
        elif env_index is None:
            env_index = th.arange(0, self.batch_dim).int().tolist()
        elif isinstance(env_index, int):
            env_index = [env_index]

        for index in env_index:
            for i in range(10):
                try:
                    self.spawn_room(index)
                    super().reset_world_at(index)
                    if self.gp is not None:
                        self.gp.reset(self.world, self.static_objects, index)
                    break
                except Exception as e:
                    print(e)
                    print(
                        f"Something went wrong reseting env {index}, retrying... ({10 - i} attempts left)"
                    )
                    if i == 9:
                        raise e

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        geoms = super().extra_render(env_index)
        return geoms


def generate_points_on_circle(N, radius=1.0) -> (list, list, list):
    start_points = []
    goal_points = []
    orientations = []
    angle_step = 2 * math.pi / N

    for i in range(N):
        angle_start = i * angle_step
        angle_goal = (i * angle_step + math.pi) % (2 * math.pi)
        start_x = radius * math.cos(angle_start)
        start_y = radius * math.sin(angle_start)
        goal_x = radius * math.cos(angle_goal)
        goal_y = radius * math.sin(angle_goal)
        start_points.append((start_x, start_y))
        goal_points.append((goal_x, goal_y))
        orientations.append(normalize_angle(angle_start + math.pi))  # Add orientation

    return start_points, goal_points, orientations


def normalize_angle(angle):
    """
    Normalize an angle in radians to be within the range [-pi, pi].

    :param angle: Angle in radians to be normalized.
    :return: Normalized angle in radians.
    """
    normalized_angle = angle % (2 * np.pi)  # Normalize angle to be within [0, 2*pi]

    if normalized_angle > np.pi:
        normalized_angle -= 2 * np.pi  # Shift to be within [-pi, pi]

    return normalized_angle
