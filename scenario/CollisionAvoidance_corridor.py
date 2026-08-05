import math
import typing
from typing import List, Optional

import numpy as np
import torch as th
from vmas.simulator.core import World, Landmark, Sphere, Box
from vmas.simulator.utils import Color, ScenarioUtils

from scenario.CollisionAvoidance_base import CollisionAvoidance, point_to_goal

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom
    from typing import List

scenario_type = "corridor"


class ScenarioCorridorSpawner:
    @staticmethod
    def make_objects(
        world: World,
        world_radius: float,
        corridor_width_p: float,
        agent_radius: float,
        buffer: float,
    ):
        # We are splitting the world in 3 zones (0.25, 0.5, 0.25) and placing the corridor in the middle zone
        world_len = world_radius * 2
        corridor_radius = corridor_width_p * world_len / 2
        ar_buf = agent_radius + buffer + +0.25
        spawn_zones = {
            0: [
                -world_radius + ar_buf,
                world_radius - ar_buf,
                (world_radius - world_len / 4) + ar_buf,
                world_radius - ar_buf,
            ],  # x,y
            1: [
                -world_radius + ar_buf,
                world_radius - ar_buf,
                -world_radius + ar_buf,
                (-world_radius + world_len / 4) - ar_buf,
            ],  # x,y
        }

        # corridor walls:
        corridor_walls: List[Landmark] = []
        for i in range(9):  # width is up and down, length is left and right
            if i < 2:  # boxes for making the corridor
                shape = Box(
                    world_len / 2 - corridor_radius,
                    world_len / 2,
                )
            elif i < 4:
                # top and bottom walls
                shape = Box(world_len, 0.5)
            elif i < 8:
                # 4 corners
                shape = Box(0.5, world_len / 4)
            else:
                # small middle box
                shape = Box(corridor_radius, corridor_radius * 2)

            obstacle = Landmark(
                name=f"corridor_wall_{i}_{scenario_type}",
                collide=True,
                shape=shape,
                color=Color.BLACK.value,
                collision_filter=lambda e: e.movable,
            )
            world.add_landmark(obstacle)
            corridor_walls.append(obstacle)

        return corridor_walls, spawn_zones
    @staticmethod
    def spawn(
        env_index,
        world: World,
        corridor_walls: List[Landmark],
        agent_radius: float,
        world_radius: float,
        corridor_width_p: float,
        spawn_zones,
        buffer: float,
        spawn_offset: Optional[th.Tensor] = None,
    ):
        batch_dim = world.batch_dim if env_index is None else 1
        if spawn_offset is None:
            spawn_offset = th.zeros(2, device=world.device)
        spawn_offset = spawn_offset.to(world.device)
        world_len = world_radius * 2
        corridor_radius = corridor_width_p * world_len / 2

        # Two side walls
        x = world_radius / 2 + corridor_radius / 2
        corridor_walls[0].set_pos(
            th.tensor([x, 0], device=world.device) + spawn_offset, env_index
        )
        corridor_walls[1].set_pos(
            th.tensor([-x, 0], device=world.device) + spawn_offset, env_index
        )
        # Top and bottom
        y = world_radius
        corridor_walls[2].set_pos(
            th.tensor([0, y], device=world.device) + spawn_offset, env_index
        )
        corridor_walls[3].set_pos(
            th.tensor([0, -y], device=world.device) + spawn_offset, env_index
        )

        # plugging holes
        x = world_radius
        y = world_radius * 0.75
        corridor_walls[4].set_pos(
            th.tensor([x, y], device=world.device) + spawn_offset, env_index
        )
        corridor_walls[5].set_pos(
            th.tensor([-x, y], device=world.device) + spawn_offset, env_index
        )
        corridor_walls[6].set_pos(
            th.tensor([-x, -y], device=world.device) + spawn_offset, env_index
        )
        corridor_walls[7].set_pos(
            th.tensor([x, -y], device=world.device) + spawn_offset, env_index
        )

        # obstacle
        x = corridor_radius / 1.5
        corridor_walls[8].set_pos(
            th.tensor([-x, 0], device=world.device) + spawn_offset, env_index
        )

        # Batched per-zone placement (see scenario/spawn_utils.py). Agents go in
        # spawn_zones[0], goals in spawn_zones[1]; both zones are wall-free, so the
        # boundary clamp keeps everything off the walls — no fixed-wall handling
        # needed. Replaces vmas rejection sampling, which can stall in a tight zone.
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
            for i, agent in enumerate(agents):
                agent.set_pos(agent_pos[:, i] + spawn_offset, env_index)
                agent.goal.set_pos(goal_pos[:, i] + spawn_offset, env_index)
                new_rot = point_to_goal(agent)
                agent.set_rot(
                    new_rot if env_index is None else new_rot[env_index], env_index
                )


class CollisionAvoidanceCorridor(CollisionAvoidance):
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
        self.corridor_width_p = 0.2

        self.corridor_walls, self.spawn_zones = ScenarioCorridorSpawner.make_objects(
            world,
            self.world_radius,
            self.corridor_width_p,
            self.agent_radius,
            self.buffer,
        )

        self.static_objects = self.corridor_walls

        return world

    def spawn_corridor(self, env_index: int):
        ScenarioCorridorSpawner.spawn(
            env_index,
            self.world,
            self.corridor_walls,
            self.agent_radius,
            self.world_radius,
            self.corridor_width_p,
            self.spawn_zones,
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
                    self.spawn_corridor(index)
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
