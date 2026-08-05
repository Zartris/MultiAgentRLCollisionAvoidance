import math
import typing
from typing import List, Optional

import numpy as np
import torch as th
from vmas.simulator.core import World, Landmark, Box, Sphere
from vmas.simulator.utils import Color, ScenarioUtils

import scenario
from scenario.CollisionAvoidance_base import CollisionAvoidance, point_to_goal, rad_to_vector, is_on_goal
from scenario.GlobalPlanner.global_planner import AStarPlanner

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom
    from typing import List

scenario_type = "GPFocus"


class ScenarioGPFocusSpawner:
    @staticmethod
    def make_objects(
            world: World,
            world_radius: float,
    ):
        # world_len = world_radius * 2
        # corridor walls:
        obstacles: List[Landmark] = []

        obstacle = Landmark(
            name=f"Obstacle_{0}_{scenario_type}",
            collide=True,
            shape=Sphere(1.),
            color=Color.BLACK.value,
            collision_filter=lambda e: e.movable,
        )
        world.add_landmark(obstacle)
        obstacles.append(obstacle)

        return obstacles

    @staticmethod
    def spawn(
            env_index,
            world: World,
            obstacles: List[Landmark],
            world_radius: float,
            spawn_offset: Optional[th.Tensor] = None,
    ):
        assert len(world.agents) == 2, "This scenario only supports two agent"

        batch_dim = world.batch_dim if env_index is None else 1
        if spawn_offset is None:
            spawn_offset = th.zeros(2, device=world.device)
        spawn_offset.to(world.device)

        spawn_point = world_radius - world_radius * 0.1

        # place first obstacle
        x = 0
        y = 1.2
        obstacles[0].set_pos(
            th.tensor([x, y], device=world.device) + spawn_offset, env_index
        )

        agent = world.agents[0]
        agent.set_pos(th.tensor([spawn_point, 0], device=world.device) + spawn_offset, env_index)
        agent.goal.set_pos(th.tensor([-spawn_point, 0], device=world.device) + spawn_offset, env_index)
        new_rot = point_to_goal(agent)
        agent.set_rot(
            new_rot if env_index is None else new_rot[env_index], env_index
        )

        agent = world.agents[1]
        agent.set_pos(th.tensor([0, -0.6], device=world.device) + spawn_offset, env_index)
        agent.goal.set_pos(th.tensor([0, -0.6], device=world.device) + spawn_offset, env_index)
        new_rot = point_to_goal(agent)
        agent.set_rot(
            new_rot if env_index is None else new_rot[env_index], env_index
        )


class CollisionAvoidanceGPFocus(CollisionAvoidance):
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
        self.inflate_radius = 0.1
        if self.config.get("use_global_path", False):
            self.gp = AStarPlanner(
                self.world_size,
                10,
                self.inflate_radius,
                self.batch_dim,
                self.num_agents,
                self.device,
            )

        self.walls = ScenarioGPFocusSpawner.make_objects(
            world,
            self.world_radius,
        )

        self.static_objects = self.walls
        #
        return world

    def spawn(self, env_index: int):
        ScenarioGPFocusSpawner.spawn(
            env_index,
            self.world,
            self.walls,
            self.world_radius
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
