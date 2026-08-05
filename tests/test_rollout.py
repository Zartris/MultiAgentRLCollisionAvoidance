"""End-to-end rollout: env.reset -> policy -> env.step -> critic. Requires xvfb."""
import pytest
import torch
from tensordict import TensorDict

pytest.importorskip("vmas")
try:
    from torchrl.envs.libs.vmas import VmasEnv
    from scenario.CollisionAvoidance_random import CollisionAvoidanceRandom
    from models.model_loader import make_network
except Exception as e:  # pragma: no cover
    pytest.skip(f"env dependencies unavailable: {e}", allow_module_level=True)


def _cfg(n_agents=3):
    cfg = {
        "model": "oursGraph",
        "scenario_type": "random",
        "num_agents": n_agents,
        "num_obstacles": 3,
        "obstacle_size": 0.5,
        "world_size": 15,
        "dt": 0.1,
        "use_global_path_obs": True,
        "set_gp_as_goal": False,
        "gp_lookahead": 5,
        "target_point": "gp",
        "use_lidar": True,
        "num_lidar_rays": 60,
        "lidar_angle_start": 0.0,
        "lidar_angle_end": 6.283185307,
        "lidar_range": 3.5,
        "lidar_history_len": 3,
        "lidar_noise": 0.0,
        "omega_limit": 1.0,
        "v_limit": 1.0,
        "use_polar_coordinates": True,
        "cooperative_dist": 2.0,
        "dist_type": "IndependentNormal",
        "pos_shaping_factor": 2.5,
        "time_penalty": 0.0,
        "final_reward": 10.0,
        "collision_penalty": -10.0,
        "personal_space_penalty": -0.1,
        "personal_space_distance": 0.3,
        "cooperative_factor": 0,
        "other_agent_noise": 0.0,
        "draw_trajectory": False,
        "ppo": {"max_steps": 20, "std_max_start": 0.5},
    }
    # LidarSingleStep.main() derives this key right before passing cfg to the scenario.
    # The scenario reads `use_global_path` (not `use_global_path_obs`) to decide whether
    # to build the A* planner, which in turn determines the global_path slot in obs.
    cfg["use_global_path"] = cfg.get("use_global_path_obs", False) or cfg.get("set_gp_as_goal", False)
    return cfg


def test_end_to_end_single_step():
    cfg = _cfg(n_agents=3)
    scenario = CollisionAvoidanceRandom(config=cfg)
    # num_envs is intentionally NOT equal to the per-agent action dim (2). When the
    # two happen to match, `env.action_spec.shape[-1]` returning the outer batch
    # (torchrl 0.7+ behavior) silently coincides with the correct action dim, and
    # the model_loader's n_agent_outputs ends up right by accident. A mismatched
    # pair forces the distinction to surface.
    num_envs = 3
    env = VmasEnv(
        scenario=scenario,
        num_envs=num_envs,
        continuous_actions=True,
        max_steps=20,
        device="cpu",
        n_agents=cfg["num_agents"],
        clamp_actions=True,
    )

    _, policy, critic = make_network(env, cfg, None, False, "cpu")

    td = env.reset()
    # policy writes an action into the same td
    td = policy(td)
    action = td.get(("agents", "action"))
    assert action.shape == (num_envs, cfg["num_agents"], 2)
    # critic writes a scalar value per agent
    td = critic(td)
    assert td.get(("agents", "state_value")).shape == (num_envs, cfg["num_agents"], 1)

    # env.step consumes the action
    td_next = env.step(td)
    assert ("next", "agents", "observation") in td_next.keys(include_nested=True)
    env.close()


def test_collector_rollout_multi_step():
    """End-to-end regression guard: run the actual torchrl collector loop for a few
    frames. The previous test_end_to_end_single_step hit env._step once directly, so
    it missed a 0.11 upgrade bug where `env.action_spec.shape[-1]` returns the outer
    batch dim instead of the leaf action dim — the collector path is what surfaces
    this because it feeds its own tensordicts back through the env.
    """
    from torchrl.collectors import SyncDataCollector

    cfg = _cfg(n_agents=3)
    scenario = CollisionAvoidanceRandom(config=cfg)
    num_envs = 3
    env = VmasEnv(
        scenario=scenario,
        num_envs=num_envs,
        continuous_actions=True,
        max_steps=20,
        device="cpu",
        n_agents=cfg["num_agents"],
        clamp_actions=True,
    )
    _, policy, _ = make_network(env, cfg, None, False, "cpu")

    frames = 6
    collector = SyncDataCollector(
        env, policy, frames_per_batch=frames, total_frames=frames, device="cpu"
    )
    for batch in collector:
        action = batch.get(("agents", "action"))
        # (num_envs, time, n_agents, action_dim)
        assert action.shape == (num_envs, frames // num_envs, cfg["num_agents"], 2), (
            f"collector produced action with shape {action.shape}, expected last dim 2"
        )
        break
    env.close()
