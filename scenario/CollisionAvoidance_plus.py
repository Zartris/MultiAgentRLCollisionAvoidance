import typing
from typing import List, Optional

import torch as th
from vmas.simulator.core import World, Landmark, Sphere, Box
from vmas.simulator.utils import Color, ScenarioUtils

import scenario
from scenario.CollisionAvoidance_base import CollisionAvoidance, point_to_goal

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom
    from typing import List

scenario_type = "plus"


class ScenarioPlusSpawner:
    @staticmethod
    def make_objects(
        world: World,
        world_radius: float,
        corridor_width_p: float,
        agent_radius: float,
        buffer: float,
    ):
        assert (
            0 < corridor_width_p < 1
        ), "corridor_width should be between 0 and 1 (percentage of world_radius)"
        world_len = world_radius * 2
        corridor_width_p = corridor_width_p * world_len
        corridor_radius = corridor_width_p / 2
        zone_radius = corridor_radius - (agent_radius + buffer)
        spawn_zones = {
            0: [
                -zone_radius,
                zone_radius,
                world_radius - 2 * zone_radius,
                world_radius - (agent_radius + buffer),
            ],  # x,y
            1: [
                -world_radius + (agent_radius + buffer),
                -world_radius + 2 * zone_radius,
                -zone_radius,
                zone_radius,
            ],  # x,y
            2: [
                -zone_radius,
                zone_radius,
                -world_radius + (agent_radius + buffer),
                -world_radius + 2 * zone_radius,
            ],  # x,y
            3: [
                world_radius - 2 * zone_radius,
                world_radius - (agent_radius + buffer),
                -zone_radius,
                zone_radius,
            ],
        }

        box_size = world_radius - corridor_radius
        plus_walls: List[Landmark] = []
        for i in range(8):
            if i < 4:
                shape = Box(box_size, box_size)
            elif i < 6:
                # to plug the ends
                shape = Box(corridor_radius * 2 + 1, 0.5)
            else:
                shape = Box(0.5, corridor_radius * 2 + 1)
            obstacle = Landmark(
                name=f"plus_wall_{i}_{scenario_type}",
                collide=True,
                shape=shape,
                color=Color.BLACK.value,
                collision_filter=lambda e: e.movable,
            )
            world.add_landmark(obstacle)
            plus_walls.append(obstacle)

        return plus_walls, spawn_zones

    @staticmethod
    def spawn(
        env_index,
        world: World,
        world_radius: float,
        plus_walls: List[Landmark],
        corridor_width_p: float,
        agent_radius: float,
        spawn_zones: float,
        buffer: float,
        spawn_offset: Optional[th.Tensor] = None,
    ):
        batch_dim = world.batch_dim if env_index is None else 1
        if spawn_offset is None:
            spawn_offset = th.zeros(2, device=world.device)
        spawn_offset = spawn_offset.to(world.device)
        world_width = world_radius * 2
        corridor_radius = (world_width * corridor_width_p) / 2
        box_size = world_radius - corridor_radius
        x = y = box_size / 2 + corridor_radius
        plus_walls[0].set_pos(
            th.tensor([x, y], device=world.device) + spawn_offset, env_index
        )
        plus_walls[1].set_pos(
            th.tensor([x, -y], device=world.device) + spawn_offset, env_index
        )
        plus_walls[2].set_pos(
            th.tensor([-x, -y], device=world.device) + spawn_offset, env_index
        )
        plus_walls[3].set_pos(
            th.tensor([-x, y], device=world.device) + spawn_offset, env_index
        )

        x = y = world_radius + 0.20
        plus_walls[4].set_pos(
            th.tensor([0, y], device=world.device) + spawn_offset, env_index
        )
        plus_walls[5].set_pos(
            th.tensor([0, -y], device=world.device) + spawn_offset, env_index
        )
        plus_walls[6].set_pos(
            th.tensor([x, 0], device=world.device) + spawn_offset, env_index
        )
        plus_walls[7].set_pos(
            th.tensor([-x, 0], device=world.device) + spawn_offset, env_index
        )

        occupied_positions = th.empty((batch_dim, 0, 2), device=world.device)
        occupied_positions_goal = occupied_positions.clone()
        i = 0
        for agent in world.agents:
            if not agent.name.endswith(scenario_type):
                continue
            min_x, max_x, min_y, max_y = spawn_zones[i % 4]
            position = ScenarioUtils.find_random_pos_for_entity(
                occupied_positions=occupied_positions,
                env_index=env_index,
                world=world,
                min_dist_between_entities=2 * agent_radius + buffer,
                x_bounds=(min_x, max_x),
                y_bounds=(min_y, max_y),
            )
            agent.set_pos(position.squeeze(1) + spawn_offset, env_index)
            occupied_positions = th.cat([occupied_positions, position], dim=1)
            min_x, max_x, min_y, max_y = spawn_zones[(i + 2) % 4]
            g_pos = ScenarioUtils.find_random_pos_for_entity(
                occupied_positions=occupied_positions_goal,
                env_index=env_index,
                world=world,
                min_dist_between_entities=2 * agent_radius + buffer,
                x_bounds=(min_x, max_x),
                y_bounds=(min_y, max_y),
            )
            occupied_positions_goal = th.cat([occupied_positions_goal, g_pos], dim=1)
            agent.goal.set_pos(g_pos.squeeze(1) + spawn_offset, env_index)
            new_rot = point_to_goal(agent)
            agent.set_rot(
                new_rot if env_index is None else new_rot[env_index], env_index
            )
            i += 1


class CollisionAvoidancePlus(CollisionAvoidance):
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
        self.buffer = 0.2
        self.corridor_width_p = 0.2

        # Enviroment:
        self.plus_walls, self.spawn_zones = ScenarioPlusSpawner.make_objects(
            world,
            self.world_radius,
            self.corridor_width_p,
            self.agent_radius,
            self.buffer,
        )
        self.static_objects = self.plus_walls
        return world

    def spawn_plus(self, env_index: int):
        ScenarioPlusSpawner.spawn(
            env_index,
            self.world,
            self.world_radius,
            self.plus_walls,
            self.corridor_width_p,
            self.agent_radius,
            self.spawn_zones,
            self.buffer,
        )

    def reset_world_at(self, env_index: int = None):
        self.spawn_plus(env_index)
        super().reset_world_at(env_index)
        if self.gp is not None:
            self.gp.reset(self.world, self.static_objects, env_index)

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        geoms = super().extra_render(env_index)
        return geoms
