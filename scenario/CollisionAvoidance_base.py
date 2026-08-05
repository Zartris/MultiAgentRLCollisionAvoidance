import copy
import math
import typing
from typing import Dict, List, Optional

import numpy as np
import torch as th
from colorhash import ColorHash
from torch import Tensor
from vmas.simulator.core import World, Landmark, Sphere, Entity, Agent
from vmas.simulator.dynamics.diff_drive import DiffDrive
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color

from scenario.GlobalPlanner.global_planner import AStarPlanner
from scenario.diff_drive import MyDiffDrive

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom
    from typing import Dict, List, Optional

# these are to not delete the imports that are used in the code
MyDiffDrive = MyDiffDrive
DiffDrive = DiffDrive

scenario_name_to_int = {
    "random": 0,
    "circle": 1,
    "doorway": 2,
    "hallway": 3,
    "room": 4,
    "plus": 5,
    "corridor": 6,
    "localMinima": 7,
    "GPFocus": 8,
}
int_to_scenario_name = {v: k for k, v in scenario_name_to_int.items()}


def nget(d, keys, default=None):
    """
    Retrieve a value from a nested dictionary using a tuple of keys,
    returning a default value if any key is not found.
    """
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return default
    return d


# https://arxiv.org/pdf/1709.10082


class CollisionAvoidance(BaseScenario):
    def __init__(self, **kwargs):
        super().__init__()
        self.reward_weights = None
        self.dt = None
        # Note: The input is not decided yet, but currently we have a position controller
        # however the interactive mode is giving us x, y values, so we need to convert them before giving it to the controller
        self.interactive_debug = kwargs.get("interactive_debug", False)
        self.viewer_zoom = kwargs.get("min_zoom", 4.6)
        self.viewer_size = kwargs.get("viewer_size", (1600, 800))
        self.config = copy.deepcopy(kwargs.get("config", {}))
        self.viewer_zoom = self.config.get("min_zoom", self.viewer_zoom)
        self.viewer_size = self.config.get("viewer_size", self.viewer_size)
        self.dt = self.config.get("dt")
        self.verbose = self.config.get("verbose", False)
        self._scenario_type = self.config.get("scenario_type", "random")
        self.deactivated_entities: List[Entity] = []
        self.num_agents = self.config.get("num_agents", 5)
        self.num_obstacles = self.config.get("num_obstacles", 5)
        self.obstacle_size = self.config.get("obstacle_size", 1.0)
        self.agent_radius = self.config.get(
            "agent_radius", 0.2
        )  # m # from https://arxiv.org/pdf/1910.09441
        self.buffer = 0.3  # makes sure we dont spawn in walls and such
        self.agent_mass = self.config.get("agent_mass", 1)  # kg
        self.v_limit = self.config.get("v_limit", 1.0)  # m/s
        self.omega_limit = self.config.get(
            "omega_limit", 1.0
        )  # rad/s # half a turn per second
        self.accel_limit = self.config.get("accel_limit", 10)  # m/s^2
        self.force_limit = self.accel_limit * self.agent_mass  # N
        self.use_shared_rew = self.config.get("use_shared_rew", False)
        self.pos_shaping_factor = self.config.get("pos_shaping_factor", 1)
        self.rot_shaping_factor = self.config.get("rot_shaping_factor", 0)
        self.cooperative_factor = self.config.get("cooperative_factor", 0.5)
        self.final_reward = self.config.get("final_reward", 15)
        self.agent_collision_penalty = self.config.get("collision_penalty", -15)
        self.time_penalty = self.config.get("time_penalty", 0)
        self.personal_space_penalty = self.config.get("personal_space_penalty", -0.1)
        self.personal_space_dist = self.config.get("personal_space_distance", 0.5)

        self.obs_agents = []

        self.min_distance_between_entities = self.agent_radius * 2 + 0.05
        self.min_collision_distance = 0.05
        self.observation_range = self.config.get("lidar_range", 5)
        self.lidar_noise = self.config.get("lidar_noise", 0.035)
        self.lidar_history_len = self.config.get("lidar_history_len", 4)
        self.other_agent_noise = self.config.get("other_agent_noise", 0.1)

        # drawing:
        self.draw_lookahead = self.config.get("draw_lookahead", True)
        self.draw_gp_as_circles = self.config.get("draw_gp_as_circles", True)
        self.gp_circle_size = self.config.get("gp_circle_size", 0.05)
        self.draw_all_gp = self.config.get("draw_all_gp", True)
        self.draw_gp_target_index = self.config.get("draw_gp_target_index", -1)  # -1 means none

        self.draw_info_text = self.config.get("draw_info_text", True)
        self.render_lidar = self.config.get("render_lidar", True)
        self.draw_action_forces = self.config.get("draw_action_forces", False)
        self.draw_personal_space = self.config.get("draw_personal_space", True)
        self.draw_trajectory = self.config.get("draw_trajectory", True)

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
        self.batch_dim = batch_dim
        self.world_size = self.config.get("world_size", 15)  # m
        self.world_radius = self.world_size / 2
        self.gp = None
        if self.config.get("use_global_path", False) and kwargs.get("make_gp", True):
            self.gp = AStarPlanner(
                self.world_size,
                10,
                self.agent_radius + 0.2,
                self.batch_dim,
                self.num_agents,
                self.device,
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
        for i in range(self.num_agents):
            agent = self.make_agent(i, nget(self.config, ["scenario_type"]), world)
        self.pos_rew = th.zeros(batch_dim, device=device)
        self.final_rew = self.pos_rew.clone()
        self.all_goal_reached = th.zeros(batch_dim, device=device).bool()
        self.draw_states = {}
        for b in range(self.batch_dim):
            agent_states = []
            for i in range(self.num_agents):
                agent_states.append([])
            self.draw_states[b] = agent_states
        return world

    def print_env_info(self):
        print("Environment info:")
        print("\tScenario type:", self._scenario_type)
        print("\tWorld size:", self.world_size, "meter")
        print("\tWorld radius:", self.world_radius, "meter")
        print("\tdt:", self.dt, "seconds")
        print("\tNumber of agents:", self.num_agents)
        print("\tNumber of obstacles:", self.num_obstacles)
        print("\tRandom_obstacle size:", self.obstacle_size, "meter")
        print("\tAgent info:")
        print("\t\tAgent radius:", self.agent_radius, "meter")
        print("\t\tAgent mass:", self.agent_mass, "kg")
        print("\t\tVelocity limit:", self.v_limit, "m/s")
        print("\t\tAngular velocity limit:", self.omega_limit, "rad/s")
        print("\t\tAcceleration limit:", self.accel_limit, "m/s²")
        print("\t\tForce limit:", self.force_limit)
        print("\t\tMin collision distance:", self.min_collision_distance)
        print("\tReward info:")
        print("\t\tPosition shaping factor:", self.pos_shaping_factor)
        print("\t\tRotation shaping factor:", self.rot_shaping_factor)
        print("\t\tCooperative factor:", self.cooperative_factor)
        print("\t\tFinal reward:", self.final_reward)
        print("\t\tAgent collision penalty:", self.agent_collision_penalty)
        print("\t\tTime penalty:", self.time_penalty)
        print("\t\tPersonal space penalty:", self.personal_space_penalty)
        print("\t\tPersonal space distance:", self.personal_space_dist)
        print("\tObservation info:")
        print("\t\tLidar range:", self.observation_range)
        print("\t\tLidar noise:", self.lidar_noise)
        print("\t\tLidar history length:", self.lidar_history_len)
        print("\t\tOther agent noise:", self.other_agent_noise)

    def make_agent(self, index: int, scenario_name: str, world: World):
        def entity_filter_targets(e: Entity) -> bool:
            return not e.name.startswith("goal") and e.name.endswith(scenario_name)

        rgb = ColorHash(index).rgb
        rgb = brighten_color(rgb, 20)
        color = (rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)


        goal = Landmark(
            name=f"goal_{index}_{scenario_name}",
            collide=False,
            shape=Sphere(radius=self.agent_radius * 0.5),
            color=color,
        )

        sensors = (
            [
                Lidar(
                    world=world,
                    angle_start=self.config.get("lidar_angle_start", 0),
                    angle_end=self.config.get("lidar_angle_end", 2 * math.pi),
                    n_rays=nget(self.config, ("num_lidar_rays",), 120),
                    max_range=self.observation_range,
                    entity_filter=entity_filter_targets,
                    render=index == 0 and self.render_lidar,
                    render_color=Color.GRAY,
                    alpha=0.25,
                )
            ]
            if nget(self.config, ("use_lidar",), False)
            else None
        )
        agent = Agent(
            name=f"agent_{index}_{scenario_name}",
            sensors=sensors,
            shape=Sphere(radius=self.agent_radius),
            color=color,
            mass=self.agent_mass,
            render_action=self.draw_action_forces,
            u_range=[
                self.v_limit,
                self.omega_limit,
            ],  # action limits (vel, angular vel)
            u_multiplier=[1, 1],
            v_range=self.v_limit,
            f_range=self.force_limit,
            dynamics=MyDiffDrive(world, integration="rk4"),
            collision_filter=lambda e: e.name.endswith(scenario_name),
        )
        agent.goal = goal
        agent.scenario_name = th.zeros(self.batch_dim, dtype=th.int8, device=self.device) + scenario_name_to_int[
            scenario_name]

        if index == self.draw_gp_target_index or self.draw_all_gp:
            self.obs_agents.append(agent)

        self.set_agent_variables(agent)
        world.add_agent(agent)
        world.add_landmark(goal)
        return agent

    def set_agent_variables(self, agent):
        agent.pos_rew = th.zeros(self.batch_dim, device=self.device)
        agent.rot_rew = th.zeros(self.batch_dim, device=self.device)
        agent.final_rew = th.zeros(self.batch_dim, device=self.device)
        agent.agent_collision_rew = th.zeros(self.batch_dim, device=self.device)
        agent.obstacle_collision_rew = th.zeros(self.batch_dim, device=self.device)
        agent.cooperative_rew = th.zeros(self.batch_dim, device=self.device)
        agent.personal_space_penalty = th.zeros(self.batch_dim, device=self.device)
        agent.close_agents = th.zeros(self.batch_dim, device=self.device)
        agent.has_collided = th.zeros(self.batch_dim, dtype=bool, device=self.device)
        agent.on_goal = th.zeros(self.batch_dim, dtype=bool, device=self.device)
        agent.is_terminated = th.zeros(self.batch_dim, dtype=bool, device=self.device)

        agent.path_dist = th.zeros(self.batch_dim, device=self.device)
        # we want to know when an agent is done and we are just padding the rest of the batch
        agent.is_padding = th.zeros(self.batch_dim, dtype=bool, device=self.device)

        agent.pos_shaping = th.zeros(self.batch_dim, device=self.device)
        agent.rot_shaping = th.zeros(self.batch_dim, device=self.device)
        agent.lidar_history = th.zeros(
            self.batch_dim,
            self.lidar_history_len,
            self.config.get("num_lidar_rays", 100),
            device=self.device,
        )
        # agent.lidar_vel_history = th.zeros(
        #     self.batch_dim,
        #     self.lidar_history_len,
        #     self.config.get("num_lidar_rays", 100),
        #     2,  # vel_x, vel_y (global)
        #     device=self.device,
        # )

        agent.state_history = th.zeros(
            self.batch_dim,
            self.lidar_history_len,
            6,
            device=self.device,
        )  # x, y, rot, vel_x, vel_y, ang_vel

    def reset_world_at(self, env_index: int = None):
        env_to_change = env_index
        if env_to_change is None:
            # set it to all environments
            env_to_change = th.arange(0, self.batch_dim).int()

        # Clear the per-episode trajectory trail for the envs being reset. It is
        # appended every pre_step but was never cleared, so it accumulated agent
        # positions across ALL episodes forever -> eval videos drew a huge
        # cross-episode trail (agent appears to "start mid-path") AND it leaked
        # memory every step during training (unbounded growth of GPU tensors).
        if self.draw_trajectory:
            _reset_envs = (
                env_to_change.tolist() if hasattr(env_to_change, "tolist")
                else [env_to_change] if isinstance(env_to_change, int)
                else list(env_to_change)
            )
            for _b in _reset_envs:
                row = self.draw_states.get(int(_b))
                if row is not None:
                    for _i in range(len(row)):
                        row[_i] = []

        self.all_goal_reached[env_to_change] = False
        for agent in self.world.agents:
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
            agent.is_padding[env_to_change] = 0
            if self.config.get("use_lidar", False):
                measurements = agent.sensors[0].measure()
                if measurements.dim() == 2:
                    measurements = measurements.unsqueeze(1)
                dist_hist = measurements.repeat(1, self.lidar_history_len, 1)
                agent.lidar_history[env_to_change] = dist_hist[env_to_change]
                # vel_hist = th.zeros_like(agent.lidar_vel_history)
                # agent.lidar_vel_history[env_to_change] = vel_hist[env_to_change]

            agent_pos = agent.state.pos
            agent_rot = agent.state.rot
            agent_vel = agent.state.vel
            agent_ang_vel = agent.state.ang_vel
            agent_state = (
                th.cat([agent_pos, agent_rot, agent_vel, agent_ang_vel], dim=-1)
                .unsqueeze(1)
                .repeat(1, self.lidar_history_len, 1)
            )

            agent.state_history[env_to_change] = agent_state[env_to_change]

    def post_reset(self, env_index):
        ...

    def _get_collidable_masks(self, agents):
        """Static [A,A] masks of which agent pairs interact, from the real
        entity collision filters (computed once, cached by agent count).
          ps_gate  = collides(i,j) OR  collides(j,i)   (personal-space gate)
          col_gate = collides(i,j) AND collides(j,i)   (== world.collides for
                     agent-agent: both are movable spheres, and the broadphase
                     check is redundant with the per-env min-distance test below)
        """
        n = len(agents)
        if getattr(self, "_coll_mask_n", None) == n:
            return self._ps_gate, self._col_gate
        dev = agents[0].state.pos.device
        ab = th.zeros(n, n, dtype=th.bool, device=dev)
        for i, a in enumerate(agents):
            for j, b in enumerate(agents):
                if i != j:
                    ab[i, j] = bool(a.collides(b))
        eye = th.eye(n, dtype=th.bool, device=dev)
        self._ps_gate = (ab | ab.t()) & ~eye
        self._col_gate = (ab & ab.t()) & ~eye
        self._coll_mask_n = n
        return self._ps_gate, self._col_gate

    def _agent_collision_batched(self, agents):
        n = len(agents)
        dev = agents[0].state.pos.device
        ps_gate, col_gate = self._get_collidable_masks(agents)
        pos = th.stack([a.state.pos for a in agents], dim=0)            # [A,B,2]
        radii = th.tensor([a.shape.radius for a in agents], device=dev)  # [A]
        center = th.linalg.vector_norm(pos.unsqueeze(1) - pos.unsqueeze(0), dim=-1)
        dist = center - radii.view(n, 1, 1) - radii.view(1, n, 1)       # surface [A,A,B]

        psd = self.personal_space_dist
        ps = dist <= psd
        penalty = ((psd - dist) / psd) * self.personal_space_penalty * ps.int()
        masked = th.where(ps_gate.unsqueeze(-1), penalty,
                          th.full_like(penalty, float("inf")))
        psp = th.minimum(masked.min(dim=1).values, th.zeros((), device=dev))  # [A,B]

        # world.collides() also requires the pair to actually overlap (surface
        # distance <= 0, circumscribed radii) in at least one env -- a batch-wide
        # broadphase gate. A pair that is merely close (<= min_collision_distance)
        # but never overlapping in any env is NOT counted as a collision.
        overlap_any = (dist <= 0.0).any(dim=-1)                                 # [A,A]
        col = (col_gate & overlap_any).unsqueeze(-1) & (dist <= self.min_collision_distance)
        acr = col.to(pos.dtype).sum(dim=1) * self.agent_collision_penalty       # [A,B]
        hc_any = col.any(dim=1)                                                 # [A,B]
        for i, a in enumerate(agents):
            a.personal_space_penalty = psp[i]
            a.agent_collision_rew = a.agent_collision_rew + acr[i]
            a.has_collided[hc_any[i]] = 1

    def check_collision_penalty(self):
        agents = self.world.agents
        if self.agent_collision_penalty != 0:
            # agent <-> agent (vectorized, see _agent_collision_batched)
            self._agent_collision_batched(agents)
            # agent <-> landmark (kept as a loop: handles box/line walls via vmas)
            for a in agents:
                for l in self.world.landmarks:
                    if self.world.collides(a, l):
                        distance = self.world.get_distance(a, l)
                        has_collided = distance <= self.min_collision_distance
                        a.obstacle_collision_rew[has_collided] += (
                            self.agent_collision_penalty
                        )
                        a.has_collided[has_collided] = 1

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.final_rew[:] = 0

            # Batched per-agent shaping: sets on_goal/pos_rew/rot_rew/angle_diff for
            # every agent at once.
            self._compute_agent_rewards_batched()

            for a in self.world.agents:
                # a.is_terminated[:] = 0
                a.agent_collision_rew[:] = 0
                a.obstacle_collision_rew[:] = 0
                a.cooperative_rew[:] = 0
                a.close_agents[:] = 0
                a.personal_space_penalty[:] = 0

                self.pos_rew += a.pos_rew
                a.final_rew = a.on_goal * self.final_reward

                # (
                #     2
                #     - th.linalg.vector_norm(
                #         a.state.vel, dim=-1
                #     )  # we want it to stand still!

                # )

            self.all_goal_reached = th.all(
                th.stack([a.on_goal for a in self.world.agents], dim=-1),
                dim=-1,
            )

            self.final_rew[self.all_goal_reached] = self.final_reward
            self.check_collision_penalty()
            # if self.cooperative_factor > 0:
            #     self.compute_cooperative_reward()

        agent.is_terminated = agent.is_terminated.bool() + self.is_terminated(agent)
        collision_pen = agent.agent_collision_rew + agent.obstacle_collision_rew
        return (
                agent.pos_rew * (1 - agent.on_goal.int())  # only given
                + agent.final_rew
                + collision_pen
                + agent.personal_space_penalty
                + self.time_penalty
            # + agent.rot_rew
        ) * (1 - agent.is_padding.int())  # Do not give reward to padding agents

    def _compute_agent_rewards_batched(self):
        """Per-agent goal-shaping, batched over the agent dim.

        Sets on_goal / distance_to_goal / angle_diff / rot_rew / pos_rew on each
        agent. The shaping is per-agent independent. The gp target case is still
        looped (small, per-agent).
        """
        agents = self.world.agents
        device = agents[0].state.pos.device
        pos = th.stack([a.state.pos for a in agents], dim=0)            # [A,B,2]
        rot = th.stack([a.state.rot for a in agents], dim=0)            # [A,B,1]
        goal = th.stack([a.goal.state.pos for a in agents], dim=0)      # [A,B,2]
        hist = th.stack([a.state_history[:, -1, :] for a in agents], dim=0)  # [A,B,F]
        hist_pos = hist[..., :2]
        hist_rot = hist[..., 2:3]
        radii = th.tensor([a.shape.radius for a in agents], device=device).view(-1, 1)

        distance_to_goal = th.linalg.vector_norm(pos - goal, dim=-1)    # [A,B]
        on_goal = distance_to_goal < radii                             # [A,B]

        if self.config.get("target_point", "goal") == "gp":
            target = th.stack([self.get_gp_target_pos(a)[0] for a in agents], dim=0)
        else:
            target = goal

        angle_diff_ = angle_to_point_rew(pos, rot, target)             # [A,B]
        prev_angle_diff = angle_to_point_rew(hist_pos, hist_rot, target)
        not_on_goal = 1 - on_goal.int()
        rot_rew = (prev_angle_diff - angle_diff_) * self.rot_shaping_factor * not_on_goal
        dist_to_target = th.linalg.vector_norm(pos - target, dim=-1)
        prev_dist_to_target = th.linalg.vector_norm(hist_pos - target, dim=-1)
        pos_rew = (prev_dist_to_target - dist_to_target) * self.pos_shaping_factor * not_on_goal

        for i, a in enumerate(agents):
            a.distance_to_goal = distance_to_goal[i]
            a.on_goal = on_goal[i]
            a.angle_diff = angle_diff_[i]
            a.rot_rew = rot_rew[i]
            a.pos_rew = pos_rew[i]

    def distance(self, pos1, pos2):
        return th.linalg.vector_norm(pos1 - pos2, dim=-1)

    def compute_cooperative_reward(self):
        threshold_distance = self.config.get(
            "cooperative_dist", 1
        )  # Define a threshold for being "close"

        for i in range(self.num_agents):
            agent_i = self.world.agents[i]
            agent_i_terminated = self.is_terminated(agent_i).int()
            i_scenario = agent_i.name.split("_")[-1]
            for j in range(i + 1, self.num_agents):
                agent_j = self.world.agents[j]
                if not agent_j.name.endswith(i_scenario):
                    continue
                agent_j_terminated = self.is_terminated(agent_j).int()
                not_terminated = (1 - agent_i_terminated) * (1 - agent_j_terminated)
                dist = self.distance(agent_i.state.pos, agent_j.state.pos)
                agents_close = dist < threshold_distance
                agent_i.cooperative_rew[agents_close] += (
                        agent_j.pos_rew[agents_close]
                        * self.cooperative_factor
                        * not_terminated[agents_close]
                )
                agent_i.close_agents[agents_close] += not_terminated[agents_close]

                agent_j.cooperative_rew[agents_close] += (
                        agent_i.pos_rew[agents_close]
                        * self.cooperative_factor
                        * not_terminated[agents_close]
                )
                agent_j.close_agents[agents_close] += not_terminated[agents_close]
            agent_i.cooperative_rew[agent_i.close_agents > 0] /= agent_i.close_agents[
                agent_i.close_agents > 0
                ]

    def get_gp_target_pos(self, agent: Agent):
        path = agent.global_path
        agent_pos = agent.state.pos.unsqueeze(1)
        closest_point_index = th.argmin(
            th.linalg.vector_norm(path - agent_pos, dim=-1), dim=-1
        )
        lookahead_point_index = th.clip(
            closest_point_index + self.config.get("gp_lookahead", 3),
            max=path.shape[1] - 2,
        )  # we want to look ahead
        lookahead_point = path[th.arange(self.batch_dim), lookahead_point_index]
        lookahead_point_next_index = th.clip(
            lookahead_point_index + 1, max=path.shape[1] - 1
        )
        lookahead_point_next = path[
            th.arange(self.batch_dim), lookahead_point_next_index
        ]
        lookahead_direction = angle_to_point(lookahead_point, lookahead_point_next)

        # The path is padded by REPEATING the goal point (see compute_paths), so once the
        # lookahead window enters that plateau, lookahead_point == lookahead_point_next and
        # angle_to_point gives a degenerate (~zero) direction. BUGFIX: the old guard only
        # caught the very last index; detect ANY collapsed pair (identical consecutive
        # points) so the whole goal-padded tail falls back to heading straight at the goal.
        no_next_point = (lookahead_point_index + 1 >= path.shape[1] - 1) | (
            th.linalg.vector_norm(lookahead_point_next - lookahead_point, dim=-1) < 1e-6
        )
        if th.any(no_next_point):
            # At the path end there is no "next" point to define a direction, so head
            # straight for the goal. angle_to_point returns a UNIT DIRECTION VECTOR
            # [dx, dy] matching lookahead_direction's shape. The previous code assigned
            # angle_to_goal()'s scalar *difference* angle, which broadcast into both
            # components ([a, a]) and made angle_diff() see a constant 45deg heading.
            goal_dir = angle_to_point(agent.state.pos, agent.goal.state.pos)
            lookahead_direction[no_next_point] = goal_dir[no_next_point]

        return lookahead_point, lookahead_direction

    def get_gp_observation(self, agent: Agent):
        lookahead_point, lookahead_direction = self.get_gp_target_pos(agent)
        agent_dir = th.cat([th.cos(agent.state.rot), th.sin(agent.state.rot)], dim=-1)
        lookahead_diff = angle_diff(agent_dir, lookahead_direction)
        lookahead_dist = th.linalg.vector_norm(
            lookahead_point - agent.state.pos, dim=-1
        )
        lookahead_angle_to_point = angle_to_point(agent.state.pos, lookahead_point)
        lookahead_angle_to_point_diff = angle_diff(agent_dir, lookahead_angle_to_point)
        return (
            lookahead_point,
            lookahead_dist,
            lookahead_angle_to_point_diff,
            lookahead_diff,
        )

    def observation(self, agent: Agent):
        # vmas calls observation(agent) once per agent per step in world-agent
        # order. The per-agent assembly (especially the O(agents^2) other-agents
        # block) is CPU-dispatch-bound, so compute every agent's observation once
        # (batched over the agent dim) on the first agent and serve the rest from
        # cache. See _compute_all_observations.
        if agent is self.world.agents[0]:
            self._obs_cache = self._compute_all_observations()
        return self._obs_cache[self.world.agents.index(agent)]

    def _compute_all_observations(self):
        agents = self.world.agents
        n = len(agents)
        device = agents[0].state.pos.device

        # --- own-state block: [goal, pos, rot, vel, ang_vel] per agent ---
        goal = th.stack([a.goal.state.pos for a in agents], dim=0)      # [A,B,2]
        pos = th.stack([a.state.pos for a in agents], dim=0)            # [A,B,2]
        rot = th.stack([a.state.rot for a in agents], dim=0)            # [A,B,1]
        vel = th.stack([a.state.vel for a in agents], dim=0)            # [A,B,2]
        ang_vel = th.stack([a.state.ang_vel for a in agents], dim=0)    # [A,B,1]
        state = th.cat([goal, pos, rot, vel, ang_vel], dim=-1)         # [A,B,8]
        b = pos.shape[1]

        blocks = [state]

        # --- global-path block (per-agent gp logic unchanged) ---
        if self.gp is not None:
            pp = []
            for a in agents:
                lp, ld, lap, lf = self.get_gp_observation(a)
                pp.append(
                    th.cat([lp, ld.unsqueeze(-1), lap.unsqueeze(-1), lf.unsqueeze(-1)],
                           dim=-1)
                )
            blocks.append(th.stack(pp, dim=0))                         # [A,B,5]

        # --- other-agents block (leave-one-out, per-pair gaussian noise) ---
        pos_noise = th.randn(n, n, b, 2, device=device) * self.other_agent_noise
        vel_noise = th.randn(n, n, b, 2, device=device) * self.other_agent_noise
        feat = th.cat(
            [pos.unsqueeze(0).expand(n, n, b, 2) + pos_noise,
             vel.unsqueeze(0).expand(n, n, b, 2) + vel_noise],
            dim=-1,
        )                                                              # [obs,other,B,4]
        ar = th.arange(n, device=device)
        keep = th.stack([th.cat([ar[:i], ar[i + 1:]]) for i in range(n)])  # [A,A-1]
        others = feat[ar.unsqueeze(1), keep]                          # [A,A-1,B,4]
        others = others.permute(0, 2, 1, 3).reshape(n, b, (n - 1) * 4)
        blocks.append(others)

        # --- lidar block ---
        lidar_hist = th.stack([a.lidar_history for a in agents], dim=0)  # [A,B,hist,rays]
        max_range = agents[0].sensors[0]._max_range
        blocks.append(th.flatten(max_range - lidar_hist, 2, 3))        # [A,B,hist*rays]

        return th.cat(blocks, dim=-1)                                 # [A,B,obs_dim]

    def done(self):
        is_terminated = (
            th.stack([self.is_terminated(agent) for agent in self.world.agents], dim=-1)
            .all(-1)
            .bool()
        )
        return is_terminated

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        rel_goal = agent.state.pos - agent.goal.state.pos
        info = {
            "pos_rew": agent.pos_rew,
            "rot_rew": agent.rot_rew,
            "final_rew": agent.final_rew * (1 - agent.is_padding.int()),
            "agent_collision_rew": agent.agent_collision_rew
                                   * (1 - agent.is_padding.int()),
            "obstacle_collision_rew": agent.obstacle_collision_rew
                                      * (1 - agent.is_padding.int()),
            "personal_space_rew": agent.personal_space_penalty
                                  * (1 - agent.is_padding.int()),
            "ang_vel": agent.state.ang_vel,
            "vel": th.linalg.vector_norm(agent.state.vel, dim=-1),
            "vel_x": agent.state.vel[..., 0],
            "vel_y": agent.state.vel[..., 1],
            "rot": agent.state.rot,
            "goal_dist": th.linalg.vector_norm(rel_goal, dim=-1),
            "goal_angle": angle_to_goal(agent),
            # "terminated": agent.is_terminated,
            "terminated": self.is_terminated(agent),
            "is_padding": agent.is_padding,
            "on_goal": agent.on_goal,
            "has_collided": agent.has_collided,
            "path_dist": agent.path_dist,
            "scenario_name": agent.scenario_name,
        }
        if self.gp is not None:
            (
                lookahead_point,
                lookahead_dist,
                lookahead_angle_to_point_diff,
                lookahead_diff,
            ) = self.get_gp_observation(agent)
            info["lookahead_point"] = lookahead_point
            info["lookahead_dist"] = lookahead_dist
            info["lookahead_angle_to_point_diff"] = lookahead_angle_to_point_diff
            # info["lookahead_dir_diff"] = lookahead_diff
        return info

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms: List[Geom] = []
        left_panel_text = {}
        right_panel_text = {}

        # ego agent
        obs_agent = self.world.agents[0]
        if self.draw_info_text:
            info = self.info(obs_agent)
            left_panel_text = add_text_to_panel(
                left_panel_text, info, exclude_list=["vel_x", "vel_y"], env_index=env_index
            )
            left_panel_text["u_cmd"] = (
                    obs_agent.action.u[env_index, 0]
                    * (1 - obs_agent.is_terminated[env_index].int())
            ).cpu()
            left_panel_text["w_cmd"] = obs_agent.action.u[env_index, 1] * (
                    1 - obs_agent.is_terminated[env_index].int()
            )

            draw_left_and_right_panel(
                left_panel_text, right_panel_text, geoms, self.viewer_size
            )
        # draw orientation
        for i, agent in enumerate(self.world.agents):
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

            if self.draw_trajectory:
                agent_hist = self.draw_states[env_index][i]
                # One polyline through the history points instead of a circle per
                # point. The per-point make_circle loop was the eval render
                # bottleneck: it grows every step (~steps * agents circles/frame,
                # ~30 verts each). A polyline is 1 geom / N verts -> ~100x cheaper
                # and shows the same trail.
                if len(agent_hist) > 1:
                    pts = [tuple(s.tolist()) for s in agent_hist]
                    traj = rendering.make_polyline(pts)
                    traj.set_color(*agent.color)
                    geoms.append(traj)

        # draw global path
        for obs_agent in self.obs_agents:
            if self.gp is not None:
                path = obs_agent.global_path[env_index].squeeze(0)
                # One polyline through the path instead of a circle/line per point
                # (the per-point make_circle loop was a render-cost hotspot with
                # many agents). Same path shown, ~30x fewer vertices, 1 geom.
                pts = [(float(x), float(y)) for x, y in path]
                if len(pts) > 1:
                    gp_line = rendering.make_polyline(pts)
                    gp_line.set_color(*obs_agent.color)
                    geoms.append(gp_line)
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
        if self.draw_personal_space:
            for agent in self.world.agents:
                if agent.personal_space_penalty[env_index] != 0:
                    circle = rendering.make_circle(self.personal_space_dist + agent.shape.radius, filled=False)
                    xform = rendering.Transform()
                    circle.add_attr(xform)
                    xform.set_translation(*agent.state.pos[env_index])
                    circle.set_color(*Color.RED.value)
                    geoms.append(circle)

        return geoms

    def is_terminated(self, agent: Agent):
        return agent.on_goal.bool() + agent.has_collided.bool()  # + = or, *  = and

    def pre_step(self):
        # Roll every agent's state_history (drop oldest frame, append the current
        # [pos, rot, vel, ang_vel]) and apply the delayed is_padding update, batched
        # over the agent dim. Fully-padding agents are skipped (history kept), using
        # the pre-update is_padding.
        agents = self.world.agents
        if self.draw_trajectory:
            for i, agent in enumerate(agents):
                for b in range(self.batch_dim):
                    if agent.is_padding[b]:
                        continue
                    self.draw_states[b][i].append(agent.state.pos[b])

        update = [not bool(a.is_padding.all()) for a in agents]
        for a, u in zip(agents, update):
            if u:
                a.is_padding = a.is_terminated  # delayed

        pos = th.stack([a.state.pos for a in agents], dim=0)        # [A,B,2]
        rot = th.stack([a.state.rot for a in agents], dim=0)        # [A,B,1]
        vel = th.stack([a.state.vel for a in agents], dim=0)        # [A,B,2]
        ang_vel = th.stack([a.state.ang_vel for a in agents], dim=0)  # [A,B,1]
        new_state = th.cat([pos, rot, vel, ang_vel], dim=-1)        # [A,B,6]
        hist = th.stack([a.state_history for a in agents], dim=0)   # [A,B,H,6]
        rolled = th.cat([hist[:, :, 1:, :], new_state.unsqueeze(2)], dim=2)
        for i, (a, u) in enumerate(zip(agents, update)):
            if u:
                # clone: reset_world_at writes state_history in place, which would
                # otherwise alias the shared batched buffer across agents.
                a.state_history = rolled[i].clone()

    def post_step(self):
        # measure data
        if self.config.get("use_lidar", False):
            for agent in self.world.agents:
                # if agent.is_padding.all():
                #     continue
                lidar_dist = agent.sensors[0].measure()
                lidar_noise = (
                        (th.rand_like(lidar_dist) - 0.5) * 2 * self.lidar_noise * lidar_dist
                )
                lidar_dist = th.clamp(
                    lidar_dist + lidar_noise, 0, self.observation_range
                )
                agent.lidar_history = th.cat(
                    [agent.lidar_history[:, 1:, :], lidar_dist.unsqueeze(1)], dim=1
                )


def normalize_observation(
        obs: Tensor, v_limit, omega_limit, lidar_obs_range, lidar_hist=3, lidar_rays=120
) -> Tensor:
    goal_dist = obs[
        ..., 0, None
    ]  # not normalized acording to https://www.sciencedirect.com/science/article/pii/S0736584523000467#sec4.2.3
    n_goal_angle = obs[..., 1, None] / omega_limit
    n_vel = obs[..., 2, None] / v_limit
    n_ang_vel = obs[..., 3, None] / omega_limit
    dist_to_path = obs[..., 4, None]  # dist to path
    angle_to_path = obs[..., 5, None] / omega_limit
    angle_to_lookahead = obs[..., 6, None] / omega_limit

    num_feat = 7 + lidar_hist * lidar_rays
    n_lidar_dist = obs[..., 7:num_feat] / lidar_obs_range
    # n_lidar_vel = obs[..., num_feat:] / v_limit
    return th.cat(
        [
            goal_dist,
            n_goal_angle,
            n_vel,
            n_ang_vel,
            dist_to_path,
            angle_to_path,
            angle_to_lookahead,
            n_lidar_dist,
        ],
        dim=-1,
    )


def to_polar(target_pose, agent_pose, agent_rot, epsilon=1e-8):
    target_dist = th.linalg.vector_norm(agent_pose - target_pose, dim=-1).unsqueeze(-1)
    target_dir = angle_to_point(agent_pose, target_pose, epsilon=epsilon)
    agent_dir = th.cat([th.cos(agent_rot), th.sin(agent_rot)], dim=-1)
    angle_difference = angle_diff(agent_dir, target_dir).unsqueeze(-1)
    return th.cat([target_dist, angle_difference], dim=-1)


def observation_to_dict(
        obs: Tensor, use_lidar, lidar_hist_len, num_rays, use_global_path, num_agents
) -> Dict[str, Tensor]:
    # obs is a flat per-agent feature vector. Layout (contiguous):
    #   [ goal_xy (2)
    #   | agent_state (6) = pos_xy, rot, vel_xy, ang_vel
    #   | global_path (5, if use_global_path) = gp_pt_xy, gp_dist, angle_to_path, angle_to_lookahead
    #   | other_agents (4*(num_agents-1), if num_agents > 1) = [dx, dy, vx, vy] per peer, ego excluded
    #   | lidar_dist (lidar_hist_len * num_rays, if use_lidar) = time-major flattened ]
    # Shape of obs here is (..., F) where ... is however many leading dims (batch, worlds,
    # agents, time, collapsed ones, etc.). The helper indexes along the last axis and keeps
    # leading dims as-is. Keep the slice lengths in sync with the reverse packing done by the
    # scenario's `observation(agent)` in CollisionAvoidance_base.observation.
    #
    # Concrete example for W=10 worlds, N=12 agents, H=3 lidar-history frames, R=120 rays,
    # use_global_path=True (matches the LidarSingleStep defaults):
    #     F = 2 + 6 + 5 + 4*(N-1) + H*R = 2 + 6 + 5 + 44 + 360 = 417
    #     callers typically arrive with obs.shape = (10, 12, 417). After the model's
    #     flatten(0, -2) they become (120, 417), and that is what this function then
    #     carves up.
    obs_dict: dict[str, Tensor] = {
        "goal": obs[..., :2],  # x, y
        "state": obs[..., 2:8],  # x,y, rot, vel_x, vel_y, ang_vel
    }
    current_index = 8
    if use_global_path:
        obs_dict["global_path"] = obs[..., current_index: current_index + 5]
        current_index += 5  # dist to path, angle to path, angle to lookahead direction

    if num_agents > 1:
        other_agents = obs[
                       ..., current_index: current_index + 4 * (num_agents - 1)
                       ]  # x, y, vel_x, vel_y, goal_x, goal_y (goal is only used for RVO)
        # Split the peer block into a 2D (peer, feat) tail:
        #   flat (..., 4*(num_agents-1))  ->  (..., num_agents-1, 4)
        # reshape(-1, num_agents-1, 4) collapses ALL leading dims into one big batch axis.
        # This assumes the caller has pre-flattened (worlds, agents) into a single batch — see
        # the model forwards which do obs.flatten(0, len(batch)-1) before calling prepare_obs.
        other_agents = other_agents.reshape(-1, num_agents - 1, 4)  # batch, num_agents, 4
        obs_dict["other_agents"] = other_agents
        current_index += 4 * (num_agents - 1)

    if use_lidar:
        flat_lidar_dist = obs[
                          ..., current_index: current_index + lidar_hist_len * num_rays
                          ]
        # Unflatten the time-major lidar block only on the last axis so leading dims stay intact:
        #   (..., lidar_hist_len * num_rays)  ->  (..., lidar_hist_len, num_rays)
        # Unlike the other_agents reshape above, this preserves the original batch shape.
        lidar_dist = flat_lidar_dist.reshape(
            *flat_lidar_dist.shape[:-1], lidar_hist_len, num_rays
        )
        obs_dict["lidar_dist"] = lidar_dist
        current_index += lidar_hist_len * num_rays

        # flat_lidar_vel = obs[
        #                  ..., current_index: current_index + lidar_hist_len * num_rays * 2
        #                  ]
        # # lidar_vel = flat_lidar_vel.reshape(
        # #     *flat_lidar_vel.shape[:-1], lidar_hist_len, num_rays, 2
        # # )
        # # obs_dict["lidar_vel"] = lidar_vel
        # current_index += lidar_hist_len * num_rays * 2  # vel_x, vel_y

    return obs_dict


def add_text_to_panel(
        panel: dict,
        text_dict: dict,
        exclude_list: list = [],
        env_index: Optional[int] = None,
):
    # remove excluded keys from text_dict
    for key in exclude_list:
        text_dict.pop(key, None)
    if env_index is not None:
        for key, value in text_dict.items():
            # if more than one value make it a list:
            if isinstance(value, th.Tensor) and value.shape[-1] > 1:
                value = value[env_index].tolist()
            else:
                value = value[env_index].item()
            text_dict[f"{key}"] = value
    return {**panel, **text_dict}


def draw_left_and_right_panel(
        left_panel_text: Dict, right_panel_text: Dict, geoms: list, viewer_size: tuple
) -> list:
    from vmas.simulator import rendering

    screen_index = 0
    for key, value in left_panel_text.items():
        if isinstance(value, float):
            value = round(value, 4)
        elif isinstance(value, list):
            value = [round(v, 4) for v in value]
        geoms.append(
            rendering.TextLine(
                f"{key}: {value}", y=(len(left_panel_text) - screen_index) * 40
            )
        )
        screen_index += 1

    screen_index = 0
    for key, value in right_panel_text.items():
        if isinstance(value, float):
            value = round(value, 4)
        elif isinstance(value, list):
            value = [round(v, 4) for v in value]
        geoms.append(
            rendering.TextLine(
                f"{key}: {value:.4f}",
                x=viewer_size[0] * 0.8,
                y=(len(right_panel_text) - screen_index) * 40,
            )
        )
        screen_index += 1
    return geoms


def point_to_goal(agent, epsilon=1e-8):
    goal_dir = agent.goal.state.pos - agent.state.pos
    goal_dir = goal_dir / (
            th.linalg.vector_norm(goal_dir, dim=-1).unsqueeze(-1) + epsilon
    )

    # Calculate the angle using atan2 for a signed angle difference
    goal_dir = th.atan2(goal_dir[:, 1], goal_dir[:, 0])
    return goal_dir.unsqueeze(1)


def angle_to_goal(agent, epsilon=1e-8):
    goal_dir = angle_to_point(agent.state.pos, agent.goal.state.pos, epsilon=epsilon)
    # Compute the direction vector from the agent's orientation
    # Compute the direction vector from the agent's orientation
    agent_dir = th.cat([th.cos(agent.state.rot), th.sin(agent.state.rot)], dim=-1)
    angle_difference = angle_diff(agent_dir, goal_dir)
    return angle_difference


def angle_to_point(source, target, return_rads=False, epsilon=1e-8):
    # Ensure source and target are in the same shape
    source = source.unsqueeze(0) if source.dim() == 1 else source
    target = target.unsqueeze(0) if target.dim() == 1 else target

    point = target - source

    # Normalize the point to get the direction vector
    norm = th.linalg.vector_norm(point, dim=-1, keepdim=True)
    angle_vec = point / (norm + epsilon)

    if return_rads:
        # Calculate the angle in radians
        angle = th.atan2(angle_vec[:, 1], angle_vec[:, 0])
        # Normalize the angle to be within the range [-pi, pi]
        angle = (angle + th.pi) % (2 * th.pi) - th.pi
        return angle
    return angle_vec


def angle_diff(current_angle, target_angle):
    # Calculate the angle using atan2 for a signed angle difference
    angle_difference = th.atan2(target_angle[..., 1], target_angle[..., 0]) - th.atan2(
        current_angle[..., 1], current_angle[..., 0]
    )
    # Normalize the angle to be within the range [-pi, pi]
    angle_difference = (angle_difference + th.pi) % (2 * th.pi) - th.pi
    return angle_difference


def angle_to_point_rew(source_pos, source_dir, target_pos, epsilon=1e-8):
    goal_dir = target_pos - source_pos
    goal_dir = goal_dir / (
            th.linalg.vector_norm(goal_dir, dim=-1).unsqueeze(-1) + epsilon
    )
    agent_dir = th.cat([th.cos(source_dir), th.sin(source_dir)], dim=-1)
    angle_difference = th.acos(
        th.sum(goal_dir * agent_dir, dim=-1).clamp(-1 + epsilon, 1 - epsilon)
    )
    return angle_difference


def is_on_goal(agent: Agent):
    agent.distance_to_goal = th.linalg.vector_norm(
        agent.state.pos - agent.goal.state.pos,
        dim=-1,
    )
    return agent.distance_to_goal < agent.shape.radius


def rad_to_vector(theta: Tensor) -> Tensor:
    x = th.cos(theta)
    y = th.sin(theta)
    magnitude = th.sqrt(x ** 2 + y ** 2)
    normalized_x = x / magnitude
    normalized_y = y / magnitude
    normalized_vector = th.stack((normalized_x, normalized_y), dim=-1)
    return normalized_vector


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


def brighten_color(color, percentage):
    """
    Brightens an RGB color by a given percentage.

    Parameters:
    color (tuple): A tuple representing the RGB color (R, G, B), with values between 0 and 255.
    percentage (float): The percentage by which to brighten the color. Positive values brighten, negative values darken.

    Returns:
    tuple: A tuple representing the brightened RGB color.
    """
    # Ensure the percentage is between 0 and 100
    if percentage < 0:
        raise ValueError("Percentage must be positive to brighten the color.")

    # Calculate the brightened color
    brightened_color = tuple(
        min(int(c + (c * percentage / 100)), 255) for c in color
    )

    return brightened_color
