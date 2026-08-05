import math
import typing
from typing import List, Optional

import numpy as np
import torch as th
from vmas.simulator.core import World, Landmark, Box
from vmas.simulator.utils import Color, ScenarioUtils

import scenario
from scenario.CollisionAvoidance_base import CollisionAvoidance, point_to_goal
from scenario.GlobalPlanner.global_planner import AStarPlanner

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom
    from typing import List

scenario_type = "doorway"


class ScenarioDoorwaySpawner:
    @staticmethod
    def make_objects(
            world: World,
            world_radius: float,
            corridor_width_p: float,
            doorway_len_p: float,
            agent_radius: float,
            buffer: float,
    ):
        world_len = world_radius * 2
        corridor_radius = world_len * corridor_width_p / 2
        doorway_len = world_len * doorway_len_p
        ar_buf = agent_radius + buffer + 0.25
        spawn_zones = {
            0: [
                -world_radius + ar_buf,
                world_radius - ar_buf,
                -corridor_radius * 2 + ar_buf,
                -ar_buf,
            ],  # x,y (start)
            1: [
                -world_radius + ar_buf,
                world_radius - ar_buf,
                ar_buf
                + corridor_radius,  # we do not want points to spawn too close to the exit
                2 * corridor_radius,
            ],  # (goal)
        }

        # corridor walls:
        corridor_walls: List[Landmark] = []
        for i in range(5):
            if i < 2:  # boxes for making the corridor
                shape = Box(world_len / 2 - doorway_len / 2, 0.5)
            elif i < 3:
                shape = Box(world_len, 0.5)
            else:
                shape = Box(0.5, corridor_radius * 2)

            obstacle = Landmark(
                name=f"doorway_wall_{i}_{scenario_type}",
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
            world_radius: float,
            agent_radius: float,
            corridor_width_p: float,
            doorway_len_p: float,
            spawn_zones,
            buffer: float,
            spawn_offset: Optional[th.Tensor] = None,
    ):
        batch_dim = world.batch_dim if env_index is None else 1
        if spawn_offset is None:
            spawn_offset = th.zeros(2, device=world.device)
        spawn_offset = spawn_offset.to(world.device)
        world_len = world_radius * 2
        corridor_radius = world_len * corridor_width_p / 2
        doorway_len = world_len * doorway_len_p
        # two top
        y = 0
        x = world_len / 4 + doorway_len / 2
        corridor_walls[0].set_pos(
            th.tensor([x, y], device=world.device) + spawn_offset, env_index
        )
        corridor_walls[1].set_pos(
            th.tensor([-x, y], device=world.device) + spawn_offset, env_index
        )

        y = -corridor_radius * 2
        x = 0
        # bottom
        corridor_walls[2].set_pos(
            th.tensor([x, y], device=world.device) + spawn_offset, env_index
        )

        # plug ends
        x = world_len / 2
        y = -corridor_radius
        corridor_walls[3].set_pos(
            th.tensor([x, y], device=world.device) + spawn_offset, env_index
        )
        corridor_walls[4].set_pos(
            th.tensor([-x, y], device=world.device) + spawn_offset, env_index
        )

        # Batched per-zone placement (see scenario/spawn_utils.py). Agents in
        # spawn_zones[0], goals in spawn_zones[1]; both wall-free, so the boundary
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
            for i, agent in enumerate(agents):
                agent.set_pos(agent_pos[:, i] + spawn_offset, env_index)
                agent.goal.set_pos(goal_pos[:, i] + spawn_offset, env_index)
                new_rot = point_to_goal(agent)
                agent.set_rot(
                    new_rot if env_index is None else new_rot[env_index], env_index
                )


class CollisionAvoidanceDoorway(CollisionAvoidance):
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
        self.buffer = 0.5
        self.inflate_radius = self.agent_radius + 0.2
        if self.config.get("use_global_path", False):
            self.gp = AStarPlanner(
                self.world_size,
                10,
                self.inflate_radius,
                self.batch_dim,
                self.num_agents,
                self.device,
            )
        self.corridor_length = self.world_radius * 2
        self.corridor_width_p = self.config.get("corridor_width_p", 0.4)
        self.doorway_len_p = self.config.get("doorway_len_p", self.agent_radius * 4 / self.corridor_length)
        self.doorway_walls, self.spawn_zones = ScenarioDoorwaySpawner.make_objects(
            world,
            self.world_radius,
            self.corridor_width_p,
            self.doorway_len_p,
            agent_radius=self.agent_radius,
            buffer=self.buffer,
        )

        self.static_objects = self.doorway_walls
        #
        return world

    def spawn_doorway(self, env_index: int):
        ScenarioDoorwaySpawner.spawn(
            env_index,
            self.world,
            self.doorway_walls,
            self.world_radius,
            self.agent_radius,
            self.corridor_width_p,
            self.doorway_len_p,
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
                    self.spawn_doorway(index)
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
        print("\tDoorway info:")
        print("\t\tCorridor width percentage:", self.corridor_width_p, "%")
        print("\t\tCorridor width:", self.corridor_width_p * self.world_radius * 2, "m")
        print("\t\tCorridor length:", self.world_radius * 2, "m")
        print("\t\tDoorway length percentage:", self.doorway_len_p, "%")
        print("\t\tDoorway length:", self.doorway_len_p * self.world_radius * 2, "m")
        print("\t\tDoorway buffer:", self.buffer)
        print("\t\tinflate radius:", self.inflate_radius)

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        geoms = super().extra_render(env_index)
        return geoms
