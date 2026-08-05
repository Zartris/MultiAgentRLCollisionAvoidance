import os
import sys
import torch as th
from tqdm import tqdm

from vmas.simulator.core import World, Landmark, Sphere, Entity, Agent
from vmas.simulator.dynamics.diff_drive import DiffDrive


# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from scenario.CollisionAvoidance_circle import ScenarioCircleSpawner
from scenario.CollisionAvoidance_doorway import ScenarioDoorwaySpawner
from scenario.CollisionAvoidance_hallway import ScenarioHallwaySpawner
from scenario.CollisionAvoidance_plus import ScenarioPlusSpawner
from scenario.CollisionAvoidance_random import ScenarioRandomSpawner
from scenario.CollisionAvoidance_room import ScenarioRoomSpawner
from scenario.GlobalPlanner.global_planner import AStarPlanner
from scenario.diff_drive import MyDiffDrive
from scenario.CollisionAvoidance_corridor import ScenarioCorridorSpawner
from scenario.CollisionAvoidance_base import CollisionAvoidance, angle_to_goal, nget

import typing

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom
    from typing import Dict, List, Optional

# these are to not delete the imports that are used in the code
MyDiffDrive = MyDiffDrive
DiffDrive = DiffDrive

scenarios = {
    "random": ScenarioRandomSpawner,  # 7 in paper
    "circle": ScenarioCircleSpawner,  # 4 in paper
    # "plus": ScenarioPlusSpawner,  # 1 in paper
    "doorway": ScenarioDoorwaySpawner,  # 3 in paper
    # "corridor": ScenarioCorridorSpawner,  # 2 in paper
    "hallway": ScenarioHallwaySpawner,  # 5 in paper
    # "room": ScenarioRoomSpawner,  # 6 in paper
}


class CollisionAvoidanceMultiEnv(CollisionAvoidance):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # compute number of agents:
        self.num_agents = 0
        for scenario in scenarios.keys():
            # print(scenario, nget(self.config, (scenario, "num_agents"), 0))
            self.num_agents += nget(self.config, (scenario, "num_agents"), 0)
        # self.verbose = True

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
        self.device = device
        self.batch_dim = int(batch_dim)

        self.world_size = 37
        self.world_radius = self.world_size / 2
        self.gp = None
        self.inflate_radius = self.agent_radius + 0.16
        self.buffer = 0.3
        if nget(self.config, ("use_global_path",), False):
            self.gp = AStarPlanner(
                self.world_size,
                10,
                self.inflate_radius,
                self.batch_dim,
                self.num_agents,
                device,
            )

        world = World(
            batch_dim,
            device,
            # x_semidim=self.world_size / 2,
            # y_semidim=self.world_size / 2,
            # drag=0,
            dt=self.dt,
            linear_friction=0,
            substeps=2,
            collision_force=1000,
        )

        self.pos_rew = th.zeros(batch_dim, device=device)
        self.final_rew = self.pos_rew.clone()
        self.all_goal_reached = th.zeros(batch_dim, device=device).bool()

        # parent make_world initializes draw_states; this override skips super()
        # and pre_step would AttributeError when draw_trajectory=True.
        self.draw_states = {b: [[] for _ in range(self.num_agents)] for b in range(self.batch_dim)}

        self.scenario_agents = {}
        for scenario_name in scenarios.keys():
            self.scenario_agents[scenario_name] = []

        self.make_random_env(world, grid_x=-0.5, grid_y=0.5)
        self.make_circle_env(world, grid_x=0.5, grid_y=0.5)
        # self.make_plus_env(world)
        self.make_doorway_env(world, grid_x=-0.5, grid_y=-0.5)
        # self.make_corridor_env(world)
        self.make_hallway_env(world, grid_x=0.5, grid_y=-0.5)
        # self.make_room_env(world)

        # Since we are reseting some agent and not the world we need a way stop someone else to get the wrong values after a reset
        self.compute_new_state_values = True
        self.agent_info_dict = {}
        self.agent_reward_dict = {}
        self.agent_done_dict = {}

        # self.num_agents = len(world.agents)
        self.static_objects = (
            self.random_obstacles
            + self.circle_obstacles
            # + self.plus_walls
            + self.doorway_walls
            # + self.corridor_walls
            + self.hallway_walls
            # + self.room_walls
            # + self.room_obstacles
        )
        return world

    def make_random_env(self, world: World, grid_x=-1, grid_y=1):
        self.v_print("making random env")
        # Make random env:
        self.random_scenario = "random"
        self.random_world_size = nget(self.config, ("random", "world_size"), 10)
        self.random_num_agents = nget(self.config, ("random", "num_agents"))
        self.random_num_obstacles = nget(self.config, ("random", "num_obstacles"), 10)
        self.random_obstacle_size = nget(self.config, ("random", "obstacle_size"), 2)
        self.random_offset = th.tensor(
            [
                grid_x * (self.random_world_size + 2),
                grid_y * (self.random_world_size + 2),
            ],
            device=self.device,
        )
        self.random_obstacles = ScenarioRandomSpawner.make_objects(
            world, self.random_num_obstacles, self.random_obstacle_size
        )
        for i in range(self.random_num_agents):
            agent = self.make_agent(i, self.random_scenario, world)
            self.scenario_agents[self.random_scenario].append(agent)

    def make_circle_env(self, world: World, grid_x=0, grid_y=1):
        self.v_print("making circle env")
        # Make circle env:
        self.circle_scenario = "circle"
        self.circle_world_size = nget(self.config, ("circle", "world_size"), 7)
        self.circle_num_agents = nget(self.config, ("circle", "num_agents"))
        self.circle_num_obstacles = nget(self.config, ("circle", "num_obstacles"), 0)
        self.circle_obstacle_size = nget(self.config, ("circle", "obstacle_size"), 2)
        self.circle_offset = th.tensor(
            [
                grid_x * (self.circle_world_size + 2),
                grid_y * (self.circle_world_size + 2),
            ],
            device=self.device,
        )
        self.circle_obstacles = ScenarioCircleSpawner.make_objects(
            world, self.circle_num_obstacles, obstacle_size=self.circle_obstacle_size
        )
        self.circle_spawn_mode = nget(self.config, ("circle", "spawn_mode"), "random")
        for i in range(self.circle_num_agents):
            agent = self.make_agent(i, self.circle_scenario, world)
            self.scenario_agents[self.circle_scenario].append(agent)

    def make_plus_env(self, world: World, grid_x=1, grid_y=1):
        self.v_print("making plus env")
        # Make plus env:
        self.plus_scenario = "plus"
        self.plus_world_size = nget(self.config, ("plus", "world_size"), 10)
        self.plus_num_agents = nget(self.config, ("plus", "num_agents"))
        self.plus_num_obstacles = nget(self.config, ("plus", "num_obstacles"), 0)
        self.plus_corridor_width_p = nget(
            self.config,
            ("plus", "corridor_width_p"),
            0.2,  # 0.15
        )
        self.plus_offset = th.tensor(
            [grid_x * (self.plus_world_size + 2), grid_y * (self.plus_world_size + 2)],
            device=self.device,
        )
        self.plus_walls, self.plus_spawn_zones = ScenarioPlusSpawner.make_objects(
            world,
            self.plus_world_size / 2,
            self.plus_corridor_width_p,
            self.agent_radius,
            self.buffer,
        )
        for i in range(self.plus_num_agents):
            agent = self.make_agent(i, self.plus_scenario, world)
            self.scenario_agents[self.plus_scenario].append(agent)

    def make_doorway_env(self, world: World, grid_x=-1, grid_y=0):
        self.v_print("making doorway env")
        # Make doorway env:
        self.doorway_scenario = "doorway"
        self.doorway_world_size = nget(self.config, ("doorway", "world_size"), 7)
        self.doorway_num_agents = nget(self.config, ("doorway", "num_agents"))
        self.doorway_num_obstacles = nget(self.config, ("doorway", "num_obstacles"), 0)
        self.doorway_corridor_width_p = nget(
            self.config, ("doorway", "corridor_width_p"), 0.4
        )
        self.doorway_doorway_len_p = nget(
            self.config, ("doorway", "doorway_len_p"), self.agent_radius * 4 / self.doorway_world_size
        )
        self.doorway_offset = th.tensor(
            [
                grid_x * (self.doorway_world_size + 2),
                grid_y * (self.doorway_world_size + 2),
            ],
            device=self.device,
        )
        self.doorway_walls, self.doorway_spawn_zones = (
            ScenarioDoorwaySpawner.make_objects(
                world,
                self.doorway_world_size / 2,
                self.doorway_corridor_width_p,
                self.doorway_doorway_len_p,
                self.agent_radius,
                self.buffer,
            )
        )

        for i in range(self.doorway_num_agents):
            agent = self.make_agent(i, self.doorway_scenario, world)
            self.scenario_agents[self.doorway_scenario].append(agent)

    def make_corridor_env(self, world: World, grid_x=0, grid_y=0):
        self.v_print("making corridor env")
        # Make corridor env:
        self.corridor_scenario = "corridor"
        self.corridor_world_size = nget(self.config, ("corridor", "world_size"), 8)
        self.corridor_num_agents = nget(self.config, ("corridor", "num_agents"))
        self.corridor_num_obstacles = nget(
            self.config, ("corridor", "num_obstacles"), 0
        )
        self.corridor_width_p = nget(
            self.config, ("corridor", "width_p"), default=0.3
        )  # 0.2
        self.corridor_offset = th.tensor(
            [
                grid_x * (self.corridor_world_size + 2),
                grid_y * (self.corridor_world_size + 2),
            ],
            device=self.device,
        )
        self.corridor_walls, self.corridor_spawn_zones = (
            ScenarioCorridorSpawner.make_objects(
                world,
                self.corridor_world_size / 2,
                self.corridor_width_p,
                self.agent_radius,
                self.buffer,
            )
        )
        for i in range(self.corridor_num_agents):
            agent = self.make_agent(i, self.corridor_scenario, world)
            self.scenario_agents[self.corridor_scenario].append(agent)

    def make_hallway_env(self, world: World, grid_x=1, grid_y=0):
        self.v_print("making hallway env")
        # Make hallway env:
        self.hallway_scenario = "hallway"
        self.hallway_world_size = nget(self.config, ("hallway", "world_size"), 7)
        self.hallway_num_agents = nget(self.config, ("hallway", "num_agents"))
        self.hallway_width_p = nget(self.config, ("hallway", "width_p"), 0.25)  # 0.3

        self.hallway_offset = th.tensor(
            [
                grid_x * (self.hallway_world_size + 2),
                grid_y * (self.hallway_world_size + 2),
            ],
            device=self.device,
        )
        self.hallway_walls, self.hallway_spawn_zones = (
            ScenarioHallwaySpawner.make_objects(
                world,
                self.hallway_world_size / 2,
                self.hallway_width_p,
                self.agent_radius,
                self.buffer,
            )
        )

        for i in range(self.hallway_num_agents):
            agent = self.make_agent(i, self.hallway_scenario, world)
            self.scenario_agents[self.hallway_scenario].append(agent)

    def make_room_env(self, world: World, grid_x=0, grid_y=-1):
        self.v_print("making room env")
        # Make room env:
        self.room_scenario = "room"
        self.room_world_size = 12
        self.room_num_agents = nget(self.config, ("room", "num_agents"))
        self.room_spawn_size_p = nget(self.config, ("room", "spawn_size_p"), 0.2)
        self.room_obstacle_size_p = nget(self.config, ("room", "obstacle_size_p"), 0.15)
        self.room_num_obstacles = nget(self.config, ("room", "num_obstacles"), 10)
        self.room_offset = th.tensor(
            [grid_x * (self.room_world_size + 2), grid_y * (self.room_world_size + 2)],
            device=self.device,
        )
        self.room_walls, self.room_obstacles, self.room_spawn_zones = (
            ScenarioRoomSpawner.make_objects(
                world,
                self.room_world_size / 2,
                self.room_spawn_size_p,
                self.room_num_obstacles,
                self.room_obstacle_size_p,
                self.agent_radius,
                self.buffer,
            )
        )

        for i in range(self.room_num_agents):
            agent = self.make_agent(i, self.room_scenario, world)
            self.scenario_agents[self.room_scenario].append(agent)

    def spawn_random(self, env_index: int, reset_agents: bool = True):
        self.v_print("resetting random")
        ScenarioRandomSpawner.spawn(
            env_index,
            self.world,
            self.random_obstacles,
            self.random_obstacle_size,
            self.agent_radius,
            self.random_world_size / 2,
            self.buffer,
            self.random_offset,
        )
        if reset_agents:
            self.reset_agent_at(env_index, self.random_scenario)
            self.gp.reset(
                self.world, self.static_objects, env_index, self.random_scenario
            )

    def spawn_circle(self, env_index: int, reset_agents: bool = True):
        self.v_print("resetting circle")
        ScenarioCircleSpawner.spawn(
            env_index,
            self.circle_num_agents,
            self.circle_world_size / 2,
            self.world,
            self.circle_obstacles,
            self.circle_obstacle_size,
            self.agent_radius,
            self.circle_world_size / 2,
            self.buffer,
            self.circle_spawn_mode,
            self.circle_offset,
        )
        if reset_agents:
            self.reset_agent_at(env_index, self.circle_scenario)
            self.gp.reset(
                self.world, self.static_objects, env_index, self.circle_scenario
            )

    def spawn_plus(self, env_index: int, reset_agents: bool = True):
        self.v_print("resetting plus")
        ScenarioPlusSpawner.spawn(
            env_index,
            self.world,
            self.plus_world_size / 2,
            self.plus_walls,
            self.plus_corridor_width_p,
            self.agent_radius,
            self.plus_spawn_zones,
            self.buffer,
            self.plus_offset,
        )
        if reset_agents:
            self.reset_agent_at(env_index, self.plus_scenario)
            self.gp.reset(
                self.world, self.static_objects, env_index, self.plus_scenario
            )

    def spawn_doorway(self, env_index: int, reset_agents: bool = True):
        self.v_print("resetting doorway")
        ScenarioDoorwaySpawner.spawn(
            env_index,
            self.world,
            self.doorway_walls,
            self.doorway_world_size / 2,
            self.agent_radius,
            self.doorway_corridor_width_p,
            self.doorway_doorway_len_p,
            self.doorway_spawn_zones,
            self.buffer,
            self.doorway_offset,
        )
        if reset_agents:
            self.reset_agent_at(env_index, self.doorway_scenario)
            self.gp.reset(
                self.world, self.static_objects, env_index, self.doorway_scenario
            )

    def spawn_corridor(self, env_index: int, reset_agents: bool = True):
        self.v_print("resetting corridor")
        ScenarioCorridorSpawner.spawn(
            env_index,
            self.world,
            self.corridor_walls,
            self.agent_radius,
            self.corridor_world_size / 2,
            self.corridor_width_p,
            self.corridor_spawn_zones,
            self.buffer,
            self.corridor_offset,
        )
        if reset_agents:
            self.reset_agent_at(env_index, self.corridor_scenario)
            self.gp.reset(
                self.world, self.static_objects, env_index, self.corridor_scenario
            )

    def spawn_hallway(self, env_index: int, reset_agents: bool = True):
        self.v_print("resetting hallway")
        ScenarioHallwaySpawner.spawn(
            env_index,
            self.world,
            self.hallway_walls,
            self.agent_radius,
            self.hallway_world_size / 2,
            self.hallway_width_p,
            self.hallway_spawn_zones,
            self.buffer,
            self.hallway_offset,
        )
        if reset_agents:
            self.reset_agent_at(env_index, self.hallway_scenario)
            self.gp.reset(
                self.world, self.static_objects, env_index, self.hallway_scenario
            )

    def spawn_room(self, env_index: int, reset_agents: bool = True):
        self.v_print("resetting room")
        ScenarioRoomSpawner.spawn(
            env_index,
            self.world,
            self.room_walls,
            self.room_obstacles,
            self.room_spawn_zones,
            self.room_world_size / 2,
            self.room_obstacle_size_p,
            self.agent_radius,
            self.buffer,
            self.room_offset,
        )
        if reset_agents:
            self.reset_agent_at(env_index, self.room_scenario)
            self.gp.reset(
                self.world, self.static_objects, env_index, self.room_scenario
            )

    def reset_world_at(self, env_index: int = None):
        if self.batch_dim == 1:
            env_index = [0]
        elif env_index is None:
            env_index = th.arange(0, self.batch_dim).int().tolist()
        elif isinstance(env_index, int):
            env_index = [env_index]


        for index in tqdm(env_index, desc="Resetting envs",
                          position=0, leave=True):
            for i in range(10):
                try:
                    self.try_reset_world(index)
                    break
                except Exception as e:
                    print(e)
                    print(
                        f"Something went wrong reseting env {index}, retrying... ({10-i} attempts left)"
                    )
                    if i == 9:
                        raise e

    def reset_agent_at(self, env_index: int, scenario_type: str):
        env_to_change = env_index
        if env_to_change is None:
            # set it to all environments
            env_to_change = th.arange(0, self.batch_dim).int()

        for agent in self.world.agents:
            if not agent.name.endswith(scenario_type):
                continue
            agent.pos_shaping[env_to_change] = (
                th.linalg.vector_norm(
                    agent.state.pos[env_to_change]
                    - agent.goal.state.pos[env_to_change],
                    dim=-1,
                )
                * self.pos_shaping_factor
            )
            agent.rot_shaping[env_to_change] = (
                angle_to_goal(agent)[env_to_change] * self.rot_shaping_factor
            )
            agent.has_collided[env_to_change] = 0
            agent.is_terminated[env_to_change] = 0
            agent.on_goal[env_to_change] = 0
            if self.config.get("use_lidar", False):
                measurements = agent.sensors[0].measure().unsqueeze(1)
                hist = measurements.repeat(1, self.lidar_history_len, 1)
                agent.lidar_history[env_to_change] = hist[env_to_change]

    def reset_scenario(
        self, env_index: int, scenario_type: str, reset_agents: bool = True
    ):
        if scenario_type == "random":
            self.spawn_random(env_index, reset_agents)
        elif scenario_type == "circle":
            self.spawn_circle(env_index, reset_agents)
        elif scenario_type == "plus":
            self.spawn_plus(env_index, reset_agents)
        elif scenario_type == "doorway":
            self.spawn_doorway(env_index, reset_agents)
        elif scenario_type == "corridor":
            self.spawn_corridor(env_index, reset_agents)
        elif scenario_type == "hallway":
            self.spawn_hallway(env_index, reset_agents)
        elif scenario_type == "room":
            self.spawn_room(env_index, reset_agents)

    def try_reset_world(self, index: int):
        self.v_print("resetting world at", index)
        for scenario_name in scenarios.keys():
            self.reset_scenario(index, scenario_name, reset_agents=False)
        super().reset_world_at(index)
        if self.gp is not None:
            self.v_print("Do path planning:")
            self.gp.reset(self.world, self.static_objects, index)
        self.v_print("Done resetting the different scenarios: ", index)

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        geoms = super().extra_render(env_index)
        return geoms

    def v_print(self, *args):
        if self.verbose:
            print(*args)
