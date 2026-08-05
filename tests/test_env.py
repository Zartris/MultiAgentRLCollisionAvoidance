"""Environment construction + reset for each scenario type.

These tests require a display (xvfb) and the vmas thirdparty fork. Run via:
    PYTHONPATH=/repo MPLCONFIGDIR=/tmp/mpl xvfb-run -s "-screen 0 1400x900x24" pytest tests/test_env.py
"""
import os
import pytest

# Only try to import vmas / torchrl.envs once; if they fail (no GL, etc) skip the file.
vmas = pytest.importorskip("vmas")
try:
    from torchrl.envs.libs.vmas import VmasEnv
    from scenario.CollisionAvoidance_random import CollisionAvoidanceRandom
    from scenario.CollisionAvoidance_doorway import CollisionAvoidanceDoorway
    from scenario.CollisionAvoidance_hallway import CollisionAvoidanceHallway
    from scenario.CollisionAvoidance_room import CollisionAvoidanceRoom
except Exception as e:  # pragma: no cover
    pytest.skip(f"env dependencies unavailable: {e}", allow_module_level=True)


def _base_cfg(n_agents=4):
    # Minimal config that satisfies CollisionAvoidance.__init__ and spawn helpers.
    cfg = {
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
        "draw_trajectory": False,  # avoids draw_states init bugs in multi scenarios
        "ppo": {"max_steps": 20},
    }
    # LidarSingleStep.main() derives this before creating the scenario; mirror it here.
    cfg["use_global_path"] = cfg.get("use_global_path_obs", False) or cfg.get("set_gp_as_goal", False)
    return cfg


@pytest.mark.parametrize("scenario_cls,extra", [
    (CollisionAvoidanceRandom, {}),
    (CollisionAvoidanceDoorway, {}),
    (CollisionAvoidanceHallway, {}),
    (CollisionAvoidanceRoom, {"num_obstacles": 5}),
])
def test_scenario_env_resets(scenario_cls, extra):
    cfg = _base_cfg()
    cfg.update(extra)
    scenario = scenario_cls(config=cfg)
    env = VmasEnv(
        scenario=scenario,
        num_envs=2,
        continuous_actions=True,
        max_steps=20,
        device="cpu",
        n_agents=cfg["num_agents"],
        clamp_actions=True,
    )
    td = env.reset()
    # observation key exists and has the right first two axes (worlds, agents).
    obs = td.get(("agents", "observation"))
    assert obs.shape[0] == 2
    assert obs.shape[1] == cfg["num_agents"]
    env.close()


def test_env_step_progresses_step_counter():
    from torchrl.envs import StepCounter, TransformedEnv, Compose
    cfg = _base_cfg(n_agents=3)
    scenario = CollisionAvoidanceRandom(config=cfg)
    env = VmasEnv(
        scenario=scenario,
        num_envs=2,
        continuous_actions=True,
        max_steps=20,
        device="cpu",
        n_agents=cfg["num_agents"],
        clamp_actions=True,
    )
    env = TransformedEnv(env, Compose(StepCounter()))
    td = env.reset()
    td = env.rand_step(td)
    assert ("next", "step_count") in td.keys(include_nested=True)
    env.close()
