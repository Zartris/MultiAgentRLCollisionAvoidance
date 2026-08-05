import math
import typing
from typing import List, Optional

import numpy as np
import torch as th
from vmas.simulator.core import World, Landmark, Box
from vmas.simulator.utils import Color, ScenarioUtils

import scenario
from scenario.CollisionAvoidance_base import CollisionAvoidance, point_to_goal, rad_to_vector, is_on_goal
from scenario.GlobalPlanner.global_planner import AStarPlanner

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom
    from typing import List

scenario_type = "localMinima"


class ScenarioLocalMinimaSpawner:
    @staticmethod
    def make_objects(
            world: World,
            world_radius: float,
    ):
        world_len = world_radius * 2
        object_len = world_len / 3
        object_width = world_len / 3
        # corridor walls:
        corridor_walls: List[Landmark] = []

        middle = Landmark(
            name=f"middle_wall_{0}_{scenario_type}",
            collide=True,
            shape=Box(object_width, 0.5),
            color=Color.BLACK.value,
            collision_filter=lambda e: e.movable,
        )
        world.add_landmark(middle)
        corridor_walls.append(middle)

        left_wall = Landmark(
            name=f"left_wall_{1}_{scenario_type}",
            collide=True,
            shape=Box(0.5, object_len),
            color=Color.BLACK.value,
            collision_filter=lambda e: e.movable,
        )
        world.add_landmark(left_wall)
        corridor_walls.append(left_wall)

        right_wall = Landmark(
            name=f"right_wall_{2}_{scenario_type}",
            collide=True,
            shape=Box(0.5, object_len),
            color=Color.BLACK.value,
            collision_filter=lambda e: e.movable,
        )
        world.add_landmark(right_wall)
        corridor_walls.append(right_wall)

        return corridor_walls

    @staticmethod
    def spawn(
            env_index,
            world: World,
            corridor_walls: List[Landmark],
            world_radius: float,
            agent_radius: float,
            buffer: float,
            spawn_offset: Optional[th.Tensor] = None,
    ):
        batch_dim = world.batch_dim if env_index is None else 1
        if spawn_offset is None:
            spawn_offset = th.zeros(2, device=world.device)
        spawn_offset.to(world.device)
        world_len = world_radius * 2
        object_len = world_len / 3
        object_width = world_len / 3

        # place middle wall
        x = 0
        y = object_len / 2
        corridor_walls[0].set_pos(
            th.tensor([x, y], device=world.device) + spawn_offset, env_index
        )

        # place left wall
        x = -object_width / 2
        y = 0
        corridor_walls[1].set_pos(
            th.tensor([x, y], device=world.device) + spawn_offset, env_index
        )

        # place right wall
        x = object_width / 2
        corridor_walls[2].set_pos(
            th.tensor([x, y], device=world.device) + spawn_offset, env_index
        )

        assert len(world.agents) == 1, "This scenario only supports one agent"
        agent = world.agents[0]
        agent.set_pos(th.tensor([0, 0], device=world.device) + spawn_offset, env_index)
        agent.goal.set_pos(th.tensor([0, object_len], device=world.device) + spawn_offset, env_index)
        new_rot = point_to_goal(agent)
        agent.set_rot(
            new_rot if env_index is None else new_rot[env_index], env_index
        )


class CollisionAvoidanceLocalMinima(CollisionAvoidance):
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
        self.inflate_radius = self.agent_radius + 0.16
        if self.config.get("use_global_path", False):
            self.gp = AStarPlanner(
                self.world_size,
                10,
                self.inflate_radius,
                self.batch_dim,
                self.num_agents,
                self.device,
            )



        self.walls  = ScenarioLocalMinimaSpawner.make_objects(
            world,
            self.world_radius,
        )

        self.static_objects = self.walls
        #
        return world

    def spawn(self, env_index: int):
        ScenarioLocalMinimaSpawner.spawn(
            env_index,
            self.world,
            self.walls,
            self.world_radius,
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
                    self.spawn(index)
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
        from vmas.simulator import rendering

        geoms: List[Geom] = []

        # ego agent
        obs_agent = self.world.agents[0]

        # draw orientation
        for agent in self.world.agents:
            color = Color.BLACK.value
            orientation_vector = (
                    agent.state.pos[env_index]
                    + rad_to_vector(agent.state.rot[env_index]).squeeze()
                    * agent.shape.radius
                    * 2
            ).tolist()
            line = rendering.Line(
                tuple(agent.state.pos[env_index].tolist()),
                tuple(orientation_vector),
                width=4,
            )
            line.set_color(*color)
            geoms.append(line)

            if is_on_goal(agent)[env_index].bool():
                circle = rendering.make_circle(agent.shape.radius * 1.5, filled=True)
                xform = rendering.Transform()
                circle.add_attr(xform)
                xform.set_translation(*agent.goal.state.pos[env_index])
                circle.set_color(*Color.GREEN.value)
                geoms.append(circle)

            if agent.has_collided[env_index].bool():
                circle = rendering.make_circle(agent.shape.radius * 1.5, filled=True)
                xform = rendering.Transform()
                circle.add_attr(xform)
                xform.set_translation(*agent.state.pos[env_index])
                circle.set_color(*Color.RED.value)
                geoms.append(circle)
                # draw global path
        for obs_agent in self.obs_agents:
            if self.gp is not None:
                path = obs_agent.global_path[env_index].squeeze(0)
                # path = self.gp.get_path(0).squeeze(0)
                # render_path = path.unique(dim=0)
                prev_x, prev_y = path[0]
                for x, y in path:
                    if self.draw_gp_as_circles:  # Draw circles:
                        circle = rendering.make_circle(0.1, filled=True)
                        xform = rendering.Transform()
                        circle.add_attr(xform)
                        xform.set_translation(x, y)
                        circle.set_color(*Color.BLUE.value)
                        geoms.append(circle)
                    else:  # Draw lines
                        line = rendering.Line(
                            (prev_x, prev_y),
                            (x, y),
                            width=2,
                        )
                        line.set_color(*Color.BLUE.value)
                        geoms.append(line)
                        prev_x, prev_y = x, y
                if self.draw_lookahead:
                    lookahead_point, lookahead_direction = self.get_gp_target_pos(obs_agent)
                    circle = rendering.make_circle(0.2, filled=False)
                    xform = rendering.Transform()
                    circle.add_attr(xform)
                    xform.set_translation(*lookahead_point[env_index])
                    circle.set_color(*Color.RED.value)
                    geoms.append(circle)
                    # render direction
                    line = rendering.Line(
                        tuple(lookahead_point[env_index].tolist()),
                        tuple(
                            (
                                    lookahead_point[env_index] + lookahead_direction[env_index]
                            ).tolist()
                        ),
                        4,
                    )
                    line.set_color(*Color.BLACK.value)
                    geoms.append(line)

        # draw personal space
        for agent in self.world.agents:
            if agent.personal_space_penalty[env_index] != 0:
                circle = rendering.make_circle(self.personal_space_dist + agent.shape.radius, filled=False)
                xform = rendering.Transform()
                circle.add_attr(xform)
                xform.set_translation(*agent.state.pos[env_index])
                circle.set_color(*Color.RED.value)
                geoms.append(circle)

        return geoms

