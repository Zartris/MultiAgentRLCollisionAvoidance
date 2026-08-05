import time
from typing import List, Optional
import cv2
import numpy as np
import torch as th
from torch import Tensor
from numba import njit, prange
from vmas.simulator.rendering import Viewer
from colorama import Fore, Style
from vmas.simulator.core import Entity

from scenario.GlobalPlanner.DStarLite import DStarLite
from scenario.GlobalPlanner.batched_global_path_planner import (
    BatchAStar,
)


class AStarPlanner:
    def __init__(
            self,
            world_size: float,
            map_scale: float,
            inflate_radius: int,
            batch_dim: int,
            num_agents: int,
            device: th.device,
    ) -> None:
        self.world_size = world_size + 4
        self.map_scale = map_scale
        self.inflate_radius = inflate_radius * self.map_scale
        self.batch_dim = batch_dim
        self.num_agents = num_agents
        self.device = device

        self.map_size = self.world_size * self.map_scale
        assert int(self.world_size) == self.world_size, "world size must be an integer"
        self.map_size = int(self.map_size)
        self.map_renderer = Viewer(  # used for rendering map to a numpy array
            self.map_size,
            self.map_size,
            display=None,
            visible=False,  # do we need visible=True?
        )
        offset = Tensor([self.world_size / 2])
        # for global path
        self.map_renderer.set_bounds(
            left=-offset, right=offset, bottom=-offset, top=offset
        )
        self.max_path_len = 100
        self.print_time = False

    def reset(
            self,
            world,
            list_of_static_objects: List[Entity],
            env_index: Optional[int] = None,
            scenario_type: Optional[str] = None,
    ):
        start_time = time.perf_counter()
        env_to_change = env_index
        if env_to_change is None:
            # set it to all environments
            env_to_change = th.arange(0, self.batch_dim).int().tolist()

        elif isinstance(env_to_change, int):
            env_to_change = [env_to_change]
        for index in env_to_change:
            # we need a convas as it is not easy to draw directly on the tensors
            # Upstream vmas 1.5.2's Viewer.render() already does the window clear
            # (glClearColor + window.clear + switch_to + dispatch_events) at its start
            # and resets onetime_geoms at its end, so the old explicit `clear()` call
            # the fork added is no longer needed and was dropped when we migrated.
            canvas = np.zeros((self.map_size, self.map_size))
            for o in list_of_static_objects:
                if not o.movable and not o.rotatable:
                    self.map_renderer.add_onetime_list(o.render(index))
            global_static_map_np: np.ndarray = self.map_renderer.render(True)
            global_static_map_np = np.flipud(global_static_map_np)
            canvas = cv2.cvtColor(global_static_map_np, cv2.COLOR_RGB2GRAY, canvas)
            # inflate the map to make sure the agents can pass through
            self.compute_paths(world, canvas, index, scenario_type)
        if self.print_time:
            time_to_reset = time.perf_counter() - start_time
            print(
                Fore.GREEN,
                f"Time to reset {len(env_to_change)} envs: ",
                time_to_reset,
                "seconds, mean time: ",
                time_to_reset / len(env_to_change),
                "seconds",
                Style.RESET_ALL,
            )

    def convert_to_map_coordinates(self, pos):
        return ((pos + self.world_size / 2) * self.map_scale).to(th.int)

    def convert_to_sim_coordinates(self, pos):
        return pos / self.map_scale - self.world_size / 2.0

    def compute_paths(self, world, discrete_map, env_index: int, scenario_type=None):
        if scenario_type is None or scenario_type == "circle":
            circle_agents = [
                agent for agent in world.agents if agent.name.endswith("circle")
            ]
            # handle circle differently as it needs to be going through the center
            self.interpolate_circle_path(circle_agents, env_index)

        if scenario_type == "circle":
            return

        rest = [agent for agent in world.agents if not agent.name.endswith("circle")]
        agents = (
            rest
            if scenario_type is None
            else [agent for agent in world.agents if agent.name.endswith(scenario_type)]
        )  # allow to reset agents only for specific scenario

        a_map_coord = []
        g_map_coord = []
        for _, agent in enumerate(agents):
            a_map_coord.append(
                self.convert_to_map_coordinates(agent.state.pos[env_index])
            )
            g_map_coord.append(
                self.convert_to_map_coordinates(agent.goal.state.pos[env_index])
            )
        import time as _time

        from train.utils.profiling import enabled as _prof_on
        from train.utils.profiling import profile_phase, record

        with profile_phase("planner: coords -> cpu/numpy"):
            a_map_coord = th.stack(tensors=a_map_coord).to("cpu").numpy()
            g_map_coord = th.stack(g_map_coord).to("cpu").numpy()

        _t0 = _time.perf_counter()
        with profile_phase("planner: BatchAStar (numba)"):
            paths, visisted = BatchAStar(
                discrete_map,
                a_map_coord,
                g_map_coord,
                inflate_radius=self.inflate_radius,
                heuristic_type="euclidean",
                verbose=False,
                draw=False,
            ).searching(print_time=False)
        if _prof_on():
            record("planner_call_ms", (_time.perf_counter() - _t0) * 1000.0)
            record("planner_batch_size", float(len(a_map_coord)))
            _gm = getattr(discrete_map, "shape", (0, 0))
            record("planner_grid_cells", float(_gm[-1] * _gm[-2]) if len(_gm) >= 2 else 0.0)
        if paths is None:
            debug = 0
        # we need to scale path down to normal size again:
        for path_scaled, agent in zip(paths, agents):
            # Fill in the path where start is 0 and goal is 1
            # values = th.linspace(0, 1, path_scaled.shape[0], device=world.device)
            # scale path down to normal size again:
            path = self.convert_to_sim_coordinates(path_scaled)
            # interpolate path to have fixed length
            # i_path = interpolate_points(path, self.max_path_len)
            # i_path = th.tensor(i_path, device=self.world.device)
            sim_path = th.tensor(path, device=world.device)
            # repeat the last position to fill the rest of the path
            sim_path = sim_path[::3]
            assert (
                    sim_path.shape[0] < self.max_path_len
            ), f"Path is too long ({sim_path.shape[0]}), if this is continuing, adjust max_path_len={self.max_path_len}!"
            path_dist = th.linalg.norm(sim_path[1:] - sim_path[:-1], dim=1).sum()
            agent.path_dist[env_index] = path_dist
            sim_path = th.cat(
                [
                    sim_path,
                    agent.goal.state.pos[env_index]
                    .unsqueeze(0)
                    .repeat(self.max_path_len - sim_path.shape[0], 1),
                ]
            )
            # only take every third point
            if not hasattr(agent, "global_path"):
                agent.global_path = th.zeros(
                    (self.batch_dim, self.max_path_len, 2), device=agent.device
                )
            agent.global_path[env_index] = sim_path

    # def get_path(self, agent_id):
    #     return self.paths[:, agent_id]
    def interpolate_circle_path(self, circle_agents, env_index):
        target_dist_between_points = 0.4
        for agent in circle_agents:
            start = agent.state.pos[env_index]
            end = agent.goal.state.pos[env_index]
            dist = th.linalg.vector_norm(end - start)
            n_points = int(dist / target_dist_between_points)
            i_path = interpolate_two_points(start, end, n_points)
            agent.path_dist[env_index] = dist
            sim_path = th.cat(
                [
                    i_path,
                    agent.goal.state.pos[env_index]
                    .unsqueeze(0)
                    .repeat(self.max_path_len - i_path.shape[0], 1),
                ]
            )
            # only take every third point
            if not hasattr(agent, "global_path"):
                agent.global_path = th.zeros(
                    (self.batch_dim, self.max_path_len, 2), device=agent.device
                )
            agent.global_path[env_index] = sim_path


class DStarLitePlanner:
    def __init__(
            self,
            world_size: float,
            map_scale: float,
            inflate_radius: int,
            batch_dim: int,
            num_agents: int,
            device: th.device,
    ) -> None:
        self.world_size = world_size + 4
        self.map_scale = map_scale
        self.inflate_radius = int(inflate_radius * self.map_scale)
        self.batch_dim = batch_dim
        self.num_agents = num_agents
        self.device = device

        self.map_size = int(self.world_size * self.map_scale)
        self.map_renderer = Viewer(
            self.map_size,
            self.map_size,
            display=None,
            visible=False,
        )
        offset = Tensor([self.world_size / 2])
        self.map_renderer.set_bounds(
            left=-offset, right=offset, bottom=-offset, top=offset
        )
        self.max_path_len = 100
        self.print_time = False
        self.dstar_lite_planners = {}  # Dictionary to store D* Lite planners for each agent
        self.grid_maps = {}  # Dictionary to store grid maps for each environment

    def reset(
            self,
            world,
            list_of_static_objects: List[Entity],
            env_index: Optional[int] = None,
            scenario_type: Optional[str] = None,
    ):
        start_time = time.perf_counter()
        env_to_change = env_index
        if env_to_change is None:
            env_to_change = th.arange(0, self.batch_dim).int().tolist()
        elif isinstance(env_to_change, int):
            env_to_change = [env_to_change]
        for index in env_to_change:
            # See note at the other call site (~line 68) — Viewer.render() in upstream
            # vmas handles the window clear on its own, so the explicit clear() is
            # dropped.
            canvas = np.zeros((self.map_size, self.map_size))
            for o in list_of_static_objects:
                if not o.movable and not o.rotatable:
                    self.map_renderer.add_onetime_list(o.render(index))
            global_static_map_np: np.ndarray = self.map_renderer.render(True)
            global_static_map_np = np.flipud(global_static_map_np)
            canvas = cv2.cvtColor(global_static_map_np, cv2.COLOR_RGB2GRAY, canvas)
            self.grid_maps[index] = self.inflate_map(canvas, self.inflate_radius)
            self.initialize_dstar_lite(world, self.grid_maps[index], index, scenario_type)
        if self.print_time:
            time_to_reset = time.perf_counter() - start_time
            print(
                Fore.GREEN,
                f"Time to reset {len(env_to_change)} envs: ",
                time_to_reset,
                "seconds, mean time: ",
                time_to_reset / len(env_to_change),
                "seconds",
                Style.RESET_ALL,
            )

    def convert_to_map_coordinates(self, pos):
        return ((pos + self.world_size / 2) * self.map_scale).to(th.int)

    def convert_to_sim_coordinates(self, pos):
        return pos / self.map_scale - self.world_size / 2.0

    def initialize_dstar_lite(self, world, discrete_map, env_index: int, scenario_type=None):
        rest = [agent for agent in world.agents]
        agents = rest

        a_map_coord = []
        g_map_coord = []
        for _, agent in enumerate(agents):
            a_map_coord.append(
                self.convert_to_map_coordinates(agent.state.pos[env_index])
            )
            g_map_coord.append(
                self.convert_to_map_coordinates(agent.goal.state.pos[env_index])
            )
        a_map_coord = th.stack(tensors=a_map_coord).to("cpu").numpy()
        g_map_coord = th.stack(g_map_coord).to("cpu").numpy()

        # Initialize D* Lite planners for each agent
        for agent, a_coord, g_coord in zip(agents, a_map_coord, g_map_coord):
            for i in range(world.scenario.num_agents):
                dstar_lite_planner = DStarLite(
                    grid_map=discrete_map,
                    s_start=a_coord,
                    s_goal=g_coord,
                    heuristic_type="euclidean"
                )
                if agent.name not in self.dstar_lite_planners:
                    self.dstar_lite_planners[agent.name] = {}

                self.dstar_lite_planners[agent.name][env_index] = dstar_lite_planner

                path_scaled = dstar_lite_planner.plan(tuple(a_coord))
                path = self.convert_to_sim_coordinates(path_scaled)
                sim_path = th.tensor(path, device=world.device)
                sim_path = sim_path[::3]
                assert (
                        sim_path.shape[0] < self.max_path_len
                ), f"Path is too long ({sim_path.shape[0]}), if this is continuing, adjust max_path_len={self.max_path_len}!"
                path_dist = th.linalg.norm(sim_path[1:] - sim_path[:-1], dim=1).sum()
                agent.path_dist[env_index] = path_dist
                sim_path = th.cat(
                    [
                        sim_path,
                        agent.goal.state.pos[env_index]
                        .unsqueeze(0)
                        .repeat(self.max_path_len - sim_path.shape[0], 1),
                    ]
                )
                if not hasattr(agent, "global_path"):
                    agent.global_path = th.zeros(
                        (self.batch_dim, self.max_path_len, 2), device=agent.device
                    )
                agent.global_path[env_index] = sim_path

    def update_paths(self, world, env_index: int):
        for agent in world.agents:
            dstar_lite_planner = self.dstar_lite_planners.get(agent.name).get(env_index)
            if dstar_lite_planner:
                current_position = self.convert_to_map_coordinates(agent.state.pos[env_index])
                new_path_scaled = dstar_lite_planner.replan(tuple(current_position))
                if new_path_scaled is None:
                    print(f"No valid path found for agent {agent.name}, keeping the old path.")
                    continue
                new_path = self.convert_to_sim_coordinates(np.array(new_path_scaled))
                sim_path = th.tensor(new_path, device=world.device)
                sim_path = sim_path[::3]
                assert (
                        sim_path.shape[0] < self.max_path_len
                ), f"Path is too long ({sim_path.shape[0]}), if this is continuing, adjust max_path_len={self.max_path_len}!"
                path_dist = th.linalg.norm(sim_path[1:] - sim_path[:-1], dim=1).sum()
                agent.path_dist[env_index] = path_dist
                sim_path = th.cat(
                    [
                        sim_path,
                        agent.goal.state.pos[env_index]
                        .unsqueeze(0)
                        .repeat(self.max_path_len - sim_path.shape[0], 1),
                    ]
                )
                agent.global_path[env_index] = sim_path

    def update_map_with_observations(self, observations, env_index: int, agent_radius: int):
        for obs in observations:
            map_coord = self.convert_to_map_coordinates(th.tensor(obs))
            current_gridmap = self.inflate_map_with_observation(self.grid_maps[env_index], map_coord,
                                                                self.inflate_radius + agent_radius)



            for agent_name, dstar_lite_planner in self.dstar_lite_planners.items():
                dstar_lite_planner.grid_map = self.grid_maps[env_index]

    def inflate_map(self, grid_map, radius):
        return circular_dilation(grid_map, radius)

    def inflate_map_with_observation(self, grid_map, observation, radius):
        return circular_dilation_with_observation(grid_map, observation, radius)


def interpolate_two_points(point1, point2, num_points):
    """
    Interpolates between two 2D points to generate a specified number of points.

    Parameters:
    point1 (torch.Tensor): The first point, a tensor of shape (2,).
    point2 (torch.Tensor): The second point, a tensor of shape (2,).
    num_points (int): The number of points to interpolate, including the start and end points.

    Returns:
    torch.Tensor: A tensor of shape (num_points, 2) containing the interpolated points.
    """
    weights = th.linspace(0, 1, steps=num_points, device=point1.device).view(-1, 1)
    interpolated_points = (1 - weights) * point1 + weights * point2
    return interpolated_points


@njit()
def interpolate_points(path: np.ndarray, N: int):
    interpolated_path = np.zeros((N, 2), dtype=float)
    total_distance = norm_2d(path[1:] - path[:-1]).sum()
    desired_distance = total_distance / (N - 1)
    interpolated_path[0] = path[0]
    start_point = path[0]
    point_index = 1
    for i in range(len(path) - 1):
        end_point = path[i + 1]
        distance = nb_norm(end_point - start_point)
        if distance == 0:
            continue
        num_interpolated_points = int(distance / desired_distance)
        step = end_point - start_point
        step_direction = step / nb_norm(step)
        for j in range(1, num_interpolated_points + 1):
            start_point = start_point + step_direction * desired_distance
            interpolated_path[point_index] = start_point
            point_index += 1
    interpolated_path[point_index: len(interpolated_path)] = path[-1]
    return interpolated_path


@njit()
def norm_2d(vector):
    return np.sqrt(vector[:, 0] ** 2 + vector[:, 1] ** 2)


@njit()
def nb_norm(vector):
    return np.sqrt(vector[0] ** 2 + vector[1] ** 2)


@njit()
def circular_dilation(grid_map, radius):
    output_map = np.copy(grid_map)
    rows, cols = grid_map.shape
    r_squared = radius ** 2

    for x in prange(rows):
        for y in range(cols):
            if grid_map[x, y] < 255:
                min_x = max(0, x - radius)
                max_x = min(rows, x + radius + 1)
                min_y = max(0, y - radius)
                max_y = min(cols, y + radius + 1)
                for i in range(min_x, max_x):
                    for j in range(min_y, max_y):
                        if (x - i) ** 2 + (y - j) ** 2 <= r_squared:
                            output_map[i, j] = 255
    return output_map


@njit()
def circular_dilation_with_observation(grid_map, observation, radius):
    output_map = np.copy(grid_map)
    rows, cols = grid_map.shape
    r_squared = radius ** 2
    x, y = observation

    min_x = max(0, x - radius)
    max_x = min(rows, x + radius + 1)
    min_y = max(0, y - radius)
    max_y = min(cols, y + radius + 1)
    for i in range(min_x, max_x):
        for j in range(min_y, max_y):
            if (x - i) ** 2 + (y - j) ** 2 <= r_squared:
                output_map[i, j] = 255
    return output_map
