import math
import typing
from typing import List, Optional

import numpy as np
import torch as th
from vmas.simulator.core import World, Landmark, Box
from vmas.simulator.utils import Color, ScenarioUtils

from scenario.CollisionAvoidance_base import CollisionAvoidance, point_to_goal, nget
from scenario.CollisionAvoidance_circle import scenario_type
from scenario.GlobalPlanner.global_planner import AStarPlanner

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom
    from typing import List

scenario_type = "hallway"


class ScenarioHallwaySpawner:
    @staticmethod
    def make_objects(
            world: World,
            world_radius: float,
            hall_width_p: float,
            agent_radius: float,
            buffer: float,
    ):
        assert (
                0 < hall_width_p < 1
        ), "hallway width should be less than 1 and greater than 0"
        corridor_radius = hall_width_p * world_radius
        world_length = world_radius * 2
        ar_buffer = agent_radius + buffer + 0.25
        spawn_zones = {
            0: [
                -world_radius + ar_buffer,
                -(world_radius / 2),
                -corridor_radius + ar_buffer,
                corridor_radius - ar_buffer,
            ],  # x,y (start)
            1: [
                world_radius / 2,
                world_radius - ar_buffer,
                -corridor_radius + ar_buffer,
                corridor_radius - ar_buffer,
            ],  # x,y (goal)
        }

        # corridor walls:
        corridor_walls: List[Landmark] = []
        for i in range(4):
            if i < 2:  # boxes for making the corridor
                shape = Box(world_length, 0.5)
            else:
                shape = Box(0.5, corridor_radius * 2)

            obstacle = Landmark(
                name=f"hallway_wall_{i}_{scenario_type}",
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
            hallway_walls: List[Landmark],
            agent_radius: float,
            world_radius: float,
            hall_width_p: float,
            spawn_zones,
            buffer: float,
            spawn_offset: Optional[th.Tensor] = None,
    ):
        batch_dim = world.batch_dim if env_index is None else 1
        if spawn_offset is None:
            spawn_offset = th.zeros(2, device=world.device)
        spawn_offset = spawn_offset.to(world.device)
        corridor_radius = hall_width_p * world_radius
        # top and bottom
        y = corridor_radius
        x = 0
        hallway_walls[0].set_pos(
            th.tensor([x, y], device=world.device) + spawn_offset, env_index
        )
        hallway_walls[1].set_pos(
            th.tensor([-x, -y], device=world.device) + spawn_offset, env_index
        )

        # plug ends
        x = world_radius
        y = 0
        hallway_walls[2].set_pos(
            th.tensor([x, y], device=world.device) + spawn_offset, env_index
        )
        hallway_walls[3].set_pos(
            th.tensor([-x, -y], device=world.device) + spawn_offset, env_index
        )

        # Batched per-zone placement (see scenario/spawn_utils.py). Hallway is
        # cross-traffic: even-index agents start in spawn_zones[0], odd in
        # spawn_zones[1], and each goal is the agent's start shifted straight to
        # the opposite side (deterministic). Zones are wall-free so the boundary
        # clamp keeps starts off the walls. Replaces vmas rejection sampling.
        from scenario.spawn_utils import build_required_dist, relax_positions

        agents = [a for a in world.agents if a.name.endswith(scenario_type)]
        shift = world_radius + world_radius / 2 - (agent_radius + buffer + 0.25)
        right = th.tensor([shift, 0.0], device=world.device)
        left = th.tensor([-shift, 0.0], device=world.device)
        d = 2 * agent_radius + buffer

        def _place(group, zone):
            if not group:
                return None
            req = build_required_dist([len(group)], {(0, 0): d}, world.device)
            x0, x1, y0, y1 = zone
            pos, _ = relax_positions(req, (x0, x1), (y0, y1), batch_dim, world.device)
            return pos

        for parity, side in ((0, right), (1, left)):
            group = agents[parity::2]
            pos = _place(group, spawn_zones[parity])
            for k, agent in enumerate(group):
                agent.set_pos(pos[:, k] + spawn_offset, env_index)
                agent.goal.set_pos(pos[:, k] + side + spawn_offset, env_index)
                new_rot = point_to_goal(agent)
                agent.set_rot(
                    new_rot if env_index is None else new_rot[env_index], env_index
                )


class CollisionAvoidanceHallway(CollisionAvoidance):
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
        self.hall_width_p = self.config.get("hall_width_p", 0.25)
        self.corridor_walls, self.spawn_zones = ScenarioHallwaySpawner.make_objects(
            world,
            self.world_radius,
            self.hall_width_p,
            self.agent_radius,
            self.buffer,
        )
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
        self.static_objects = self.corridor_walls

        return world

    def spawn_hallway(self, env_index: int):
        ScenarioHallwaySpawner.spawn(
            env_index,
            self.world,
            self.corridor_walls,
            self.agent_radius,
            self.world_radius,
            self.hall_width_p,
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
                    self.spawn_hallway(index)
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

    def print_env_info(self):
        super().print_env_info()
        print("\tHallway info:")
        print("\t\tHall width percentage:", self.hall_width_p, "%")
        print("\t\tHallway width:", self.hall_width_p * self.world_radius * 2, "m")
        print("\t\tHallway length:", self.world_radius * 2, "m")
        print("\t\tHallway buffer:", self.buffer)
        print("\t\tinflate radius:", self.inflate_radius)

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        geoms = super().extra_render(env_index)
        return geoms
