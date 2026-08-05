import math
import typing
from typing import List, Optional

import numpy as np
import torch as th
from vmas.simulator.core import World, Landmark, Sphere
from vmas.simulator.utils import Color, ScenarioUtils

from scenario.CollisionAvoidance_base import CollisionAvoidance

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom
    from typing import List
scenario_type = "circle"


class ScenarioCircleSpawner:
    @staticmethod
    def make_objects(
            world: World,
            num_obstacles: int,
            obstacle_size: float,
    ):
        random_obstacles: List[Landmark] = []
        for i in range(num_obstacles):
            obstacle = Landmark(
                name=f"obstacle_{i}_{scenario_type}",
                collide=True,
                shape=Sphere(radius=obstacle_size / 2),
                color=Color.BLACK.value,
                collision_filter=lambda e: e.movable and e.name.endswith(scenario_type),
            )
            world.add_landmark(obstacle)
            random_obstacles.append(obstacle)
        return random_obstacles

    @staticmethod
    def spawn(
            env_index,
            num_agents: int,
            spawn_circle_radius: float,
            world: World,
            random_obstacles: List[Landmark],
            obstacle_size: float,
            agent_radius: float,
            world_radius: float,
            buffer: float,
            spawn_mode: str = "random",
            spawn_offset: Optional[th.Tensor] = None,
    ):
        batch_dim = world.batch_dim if env_index is None else 1
        if spawn_offset is None:
            spawn_offset = th.zeros(2, device=world.device)
        spawn_offset = spawn_offset.to(world.device)
        start_points, goal_points, orientation = generate_points_on_circle(
            num_agents, spawn_circle_radius, agent_radius, mode=spawn_mode, batch_dim=batch_dim
        )
        i = 0
        for agent in world.agents:
            if not agent.name.endswith(scenario_type):
                continue
            agent.set_pos(
                th.tensor(
                    np.array([*start_points[i]]),
                    dtype=th.float32,
                    device=world.device,
                ).permute(1, 0)
                + spawn_offset,
                batch_index=env_index,
            )
            agent.set_rot(
                th.tensor(
                    np.array([orientation[i]]),
                    dtype=th.float32,
                    device=world.device,
                ).permute(1, 0),
                batch_index=env_index,
            )

            agent.goal.set_pos(
                th.tensor(
                    np.array([*goal_points[i]]),
                    dtype=th.float32,
                    device=world.device,
                ).permute(1, 0)
                + spawn_offset,
                batch_index=env_index,
            )
            i += 1
        occupied_positions = (
            th.tensor(np.array(goal_points + start_points), device=world.device).permute(2, 0, 1)
        )

        min_dist = obstacle_size / 2 + agent_radius + buffer
        for obstacle in random_obstacles:
            position = ScenarioUtils.find_random_pos_for_entity(
                occupied_positions=occupied_positions,
                env_index=env_index,
                world=world,
                min_dist_between_entities=min_dist,
                x_bounds=(-world_radius, world_radius),
                y_bounds=(-world_radius, world_radius),
            )
            obstacle.set_pos(
                position.squeeze(1) + spawn_offset,
                batch_index=env_index,
            )
            occupied_positions = th.cat([occupied_positions, position], dim=1)


class CollisionAvoidanceCircle(CollisionAvoidance):
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
        # Enviroment:
        self.random_obstacles: List[Landmark] = ScenarioCircleSpawner.make_objects(
            world, self.num_obstacles, self.obstacle_size
        )
        self.spawn_circle_radius = self.world_size / 2
        self.spawn_mode = self.config.get("spawn_mode", "random")
        return world

    def spawn_circle(self, env_index):
        ScenarioCircleSpawner.spawn(
            env_index,
            self.num_agents,
            self.spawn_circle_radius,
            self.world,
            self.random_obstacles,
            self.obstacle_size,
            self.agent_radius,
            self.world_radius,
            self.buffer,
            self.spawn_mode,
        )

    def reset_world_at(self, env_index: int = None):
        self.spawn_circle(env_index)
        super().reset_world_at(env_index)
        if self.gp is not None:
            self.gp.reset(self.world, self.random_obstacles, env_index, scenario_type)

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms = super().extra_render(env_index)
        # Center circle
        color = Color.BLACK.value
        circle = rendering.make_circle(self.spawn_circle_radius, filled=False)
        xform = rendering.Transform()
        circle.add_attr(xform)
        xform.set_translation(0, 0)
        circle.set_color(*color)
        geoms.append(circle)
        return geoms


# def generate_points_on_circle(N, radius=1.0) -> (list, list, list):
#     start_points = []
#     goal_points = []
#     orientations = []
#     angle_step = 2 * math.pi / N

#     for i in range(N):
#         angle_start = i * angle_step
#         angle_goal = (i * angle_step + math.pi) % (2 * math.pi)
#         start_x = radius * math.cos(angle_start)
#         start_y = radius * math.sin(angle_start)
#         goal_x = radius * math.cos(angle_goal)
#         goal_y = radius * math.sin(angle_goal)
#         start_points.append((start_x, start_y))
#         goal_points.append((goal_x, goal_y))
#         orientations.append(normalize_angle(angle_start + math.pi))  # Add orientation

#     return start_points, goal_points, orientations


def normalize_angle(angle):
    """
    Normalize an angle in radians to be within the range [-pi, pi].

    :param angle: np array of angle in radians to be normalized.
    :return: Normalized angle in radians.
    """
    normalized_angle = angle % (2 * np.pi)  # Normalize angle to be within [0, 2*pi]
    normalized_angle = np.where(normalized_angle > np.pi, normalized_angle - 2 * np.pi,
                                normalized_angle)  # Shift to be within [-pi, pi]
    return normalized_angle


def generate_points_on_circle(
        N, radius=1.0, agent_radius=0.1, mode="equally_spaced", batch_dim=1
) -> (List, List, List):
    start_points = []
    goal_points = []
    orientations = []

    if mode == "equally_spaced":
        angle_step = 2 * np.pi / N

        for i in range(N):
            angle_start = np.array([i * angle_step]).repeat(batch_dim)
            angle_goal = np.array([(i * angle_step + np.pi) % (2 * np.pi)]).repeat(batch_dim)
            start_x = radius * np.cos(angle_start)
            start_y = radius * np.sin(angle_start)
            goal_x = radius * np.cos(angle_goal)
            goal_y = radius * np.sin(angle_goal)
            start_points.append((start_x, start_y))
            goal_points.append((goal_x, goal_y))
            orientations.append(
                normalize_angle(angle_start + math.pi)
            )  # Add orientation


    elif mode == "random":
        while len(start_points) < N:
            angle_start = np.random.uniform(0, 2 * math.pi, size=batch_dim)
            start_x = radius * np.cos(angle_start)
            start_y = radius * np.sin(angle_start)

            # Check if the new point is at least agent_radius away from all other points
            if len(start_points) == 0:
                too_close = False
            else:
                too_close = np.any(
                    np.hypot(start_x - np.array(start_points)[:, 0], start_y - np.array(start_points)[:, 1])
                    < agent_radius)
            # too_close = any(
            #     np.hypot(start_x - x, start_y - y) < agent_radius
            #     for x, y in start_points
            # )

            if not too_close:
                angle_goal = (angle_start + np.pi) % (2 * np.pi)
                goal_x = radius * np.cos(angle_goal)
                goal_y = radius * np.sin(angle_goal)
                start_points.append((start_x, start_y))
                goal_points.append((goal_x, goal_y))
                orientations.append(
                    normalize_angle(angle_start + np.pi)
                )  # Add orientation

    return start_points, goal_points, orientations
