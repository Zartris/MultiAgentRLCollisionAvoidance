import typing
from typing import List, Optional

import torch as th
from vmas.simulator.core import World, Landmark, Sphere
from vmas.simulator.utils import Color, ScenarioUtils

from scenario.CollisionAvoidance_base import CollisionAvoidance, point_to_goal, nget
from scenario.GlobalPlanner.global_planner import AStarPlanner

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom
    from typing import List

scenario_type = "random"


class ScenarioRandomSpawner:
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
            world: World,
            random_obstacles: List[Landmark],
            obstacle_size: float,
            agent_radius: float,
            world_radius: float,
            buffer: float,
            spawn_offset: Optional[th.Tensor] = None,
    ):
        # Batched, retry-free placement (see scenario/spawn_utils.py). Replaces
        # vmas rejection sampling, which stalls when packing many agents/goals.
        # Obstacles, agents and goals are placed in one relaxation with a per-pair
        # required-distance matrix:
        #   obstacle-obstacle: obstacle_size
        #   agent-{obstacle,agent}: obstacle_size/2 + agent_radius + buffer
        #   goal-{obstacle,goal}:   obstacle_size/2 + 2*agent_radius + buffer
        #   agent-goal: 0 (independent, matching the original two-chain logic)
        from scenario.spawn_utils import build_required_dist, relax_positions

        device = world.device
        batch_dim = world.batch_dim if env_index is None else 1
        if spawn_offset is None:
            spawn_offset = th.zeros(2, device=device)
        spawn_offset = spawn_offset.to(device)

        agents = [a for a in world.agents if a.name.endswith(scenario_type)]
        n_obs, n_ag = len(random_obstacles), len(agents)
        if n_obs == 0 and n_ag == 0:
            return

        d_obs = obstacle_size
        d_agent = obstacle_size / 2 + agent_radius + buffer
        d_goal = obstacle_size / 2 + agent_radius * 2 + buffer
        required = build_required_dist(
            [n_obs, n_ag, n_ag],
            {(0, 0): d_obs, (0, 1): d_agent, (1, 1): d_agent,
             (0, 2): d_goal, (2, 2): d_goal},
            device,
        )
        bound = world_radius - obstacle_size / 2
        pos, _ = relax_positions(
            required, (-bound, bound), (-bound, bound), batch_dim, device
        )

        obs_pos = pos[:, :n_obs]
        agent_pos = pos[:, n_obs:n_obs + n_ag]
        goal_pos = pos[:, n_obs + n_ag:]

        for i, obstacle in enumerate(random_obstacles):
            obstacle.set_pos(obs_pos[:, i] + spawn_offset, batch_index=env_index)

        for i, agent in enumerate(agents):
            agent.set_pos(agent_pos[:, i] + spawn_offset, batch_index=env_index)
            agent.goal.set_pos(goal_pos[:, i] + spawn_offset, batch_index=env_index)
            new_rot = point_to_goal(agent)
            agent.set_rot(
                new_rot if env_index is None else new_rot[env_index], env_index
            )


class CollisionAvoidanceRandom(CollisionAvoidance):
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
        self.buffer = 0.4
        # Enviroment:
        self.random_obstacles: List[Landmark] = ScenarioRandomSpawner.make_objects(
            world=world,
            num_obstacles=self.num_obstacles,
            obstacle_size=self.obstacle_size,
        )
        self.static_objects = self.random_obstacles

        self.gp = None
        if nget(self.config, ("use_global_path",), False):
            self.inflate_radius = self.agent_radius + 0.16
            self.gp = AStarPlanner(
                self.world_size,
                10,
                self.inflate_radius,
                self.batch_dim,
                self.num_agents,
                device,
            )

        return world

    def spawn_random(self, env_index):
        ScenarioRandomSpawner.spawn(
            env_index=env_index,
            world=self.world,
            random_obstacles=self.random_obstacles,
            obstacle_size=self.obstacle_size,
            agent_radius=self.agent_radius,
            world_radius=self.world_radius,
            buffer=self.buffer,
        )

    def reset_world_at(self, env_index: int = None):
        # Bounded retry. The unbounded recursive retry that used to live here
        # turned any systematic error (API drift, config mismatch, broken global
        # planner, ...) into a RecursionError ~1000 frames deep instead of
        # surfacing the real exception. See vmas 1.5.2 `Viewer.clear()` removal
        # incident (cleanup/drop-vmas-fork PR) — that crash took three diagnostic
        # iterations to find because of this swallow-and-retry loop. Matches the
        # bounded-retry style used by the other scenarios (hallway / corridor /
        # room / plus / doorway / gp_focus / local_minima).
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                self.spawn_random(env_index)
                super().reset_world_at(env_index)
                if self.gp is not None:
                    self.gp.reset(self.world, self.static_objects, env_index)
                return
            except Exception as e:
                print(e)
                print(
                    f"Something went wrong reseting env {env_index}, retrying... "
                    f"({max_attempts - attempt - 1} attempts left)"
                )
                if attempt == max_attempts - 1:
                    raise

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        geoms = super().extra_render(env_index)
        return geoms
