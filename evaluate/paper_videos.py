"""
This script is used to make videos of the trained models. It is not the parameters used for training, nor for evaluation.
These parameters are to showcase the trained models in the paper in smaller environments and focusing on the problem areas.
"""
import argparse
import json
import math
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np
# Torch
import torch as th
# Tensordict modules
from torch import multiprocessing
# Data collection
# Env
from torchrl.envs import RewardSum, TransformedEnv, StepCounter, Compose
from torchrl.envs.libs.vmas import VmasEnv

from scenario.PaperScenarioes.CollisionAvoidance_gp_focus import CollisionAvoidanceGPFocus

# Loss

# Utils
th.manual_seed(0)

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from models.baseline.rvo_policy import RVO_COLLAB_COEFF, RVO_TIME_HORIZON
from models.model_loader import make_network
from scenario.PaperScenarioes.CollisionAvoidance_local_minima import CollisionAvoidanceLocalMinima
from scenario.CollisionAvoidance_room import CollisionAvoidanceRoom
from scenario.CollisionAvoidance_circle import CollisionAvoidanceCircle
from scenario.CollisionAvoidance_corridor import CollisionAvoidanceCorridor
from scenario.CollisionAvoidance_hallway import CollisionAvoidanceHallway
from scenario.CollisionAvoidance_random import CollisionAvoidanceRandom
from scenario.CollisionAvoidance_doorway import CollisionAvoidanceDoorway
from scenario.CollisionAvoidance_multi import CollisionAvoidanceMultiEnv

from train.utils.common import evaluate

extra_folder = "Random"
prefix = "800x600_img"
# Set to the absolute path of your checkpoint file before running.
load_model = None  # e.g. "train/results/2024-08-03/oursGraph_20h-16m-46s/checkpoints/checkpoint_45.pth"
baseline_model = None  # e.g. "train/results/2024-07-24/baseline_07h-03m-34s/checkpoints/baseline_net.pth"
checkpoints = {
    # "RVO": "RVO",
    # baseline_model: "baseline",
    load_model: "oursGraph",
    # "GA3C-CARL": "GA3CPolicy"
}

test_full_runs = "results/2024-07-10/baseline_06h-55m-59s/checkpoints_combined/"
# test_full_runs = "results/2024-07-07/20h-49m-45s/checkpoints"

valid_scenario_types = {
    # "multi": CollisionAvoidanceMultiEnv,  # all environments
    "random": CollisionAvoidanceRandom,  # 7 in paper
    "circle": CollisionAvoidanceCircle,  # 4 in paper
    # "plus": CollisionAvoidancePlus,  # 1 in paper
    "doorway": CollisionAvoidanceDoorway,  # 3 in paper
    # "corridor": CollisionAvoidanceCorridor,  # 2 in paper
    "hallway": CollisionAvoidanceHallway,  # 5 in paper
    # "room": CollisionAvoidanceRoom,  # 6 in paper
    # "localMinima": CollisionAvoidanceLocalMinima,
    # "GPFocus": CollisionAvoidanceGPFocus,
}

# Full registry of selectable scenarios for the --scenario CLI flag (each has a
# matching scenario_config entry below).
ALL_SCENARIOS = {
    "random": CollisionAvoidanceRandom,
    "circle": CollisionAvoidanceCircle,
    "corridor": CollisionAvoidanceCorridor,
    "hallway": CollisionAvoidanceHallway,
    "doorway": CollisionAvoidanceDoorway,
    "room": CollisionAvoidanceRoom,
    "localMinima": CollisionAvoidanceLocalMinima,
    "GPFocus": CollisionAvoidanceGPFocus,
}
lookahead = 5
set_gp_as_goal = True
goal_point = "gp"  # "gp" or "goal"
multi_config = {
    "random": {
        "num_agents": 40,  # 20
        "num_obstacles": 8,  # 6
        "obstacle_size": 1,  # 1
        "world_size": 15,
    },
    "circle": {
        "num_agents": 40,  # 15
        "spawn_mode": "equally_spaced",
        "world_size": 15
    },  # "equally_spaced" or "random"
    "plus": {"num_agents": 4},  # 4
    "doorway": {
        "num_agents": 15,
        "world_size": 15
    },  # 5
    "corridor": {"num_agents": 5},  # 5
    "hallway": {
        "num_agents": 16,
        "world_size": 15
    },  # 8
    "room": {"num_agents": 13, "num_obstacles": 10},
}

scenario_config = {
    "random": {
        "num_agents": [40],  # [10, 20, 40],
        "num_eval_envs": 10,
        "num_obstacles": 8,
        "obstacle_size": 2.0,
        "world_size": 15,
        "min_zoom": 3.,
        "eval_render_envs": list(range(10))
    },
    "circle": {
        "num_agents": [40],  # [10, 20, 40],
        "num_eval_envs": 10,
        "world_size": 15,
        "spawn_mode": "equally_spaced",
        "num_obstacles": 0,
        "min_zoom": 3.,
        "eval_render_envs": list(range(4))
    },
    "plus": {
        "num_agents": 4,
        "world_size": 10,
        "min_zoom": 2.6
    },
    "doorway": {
        "num_agents": [15],  # [5, 10, 15],
        "world_size": 15,
        "num_eval_envs": 10,
        "min_zoom": 3.,
        "eval_render_envs": list(range(4))
    },
    "corridor": {
        "num_agents": 5,
        "world_size": 10,
        "min_zoom": 3.
    },
    "hallway": {
        "num_agents": [12],  # [8, 12, 16],
        "world_size": 15,  # 15
        "num_eval_envs": 10, # 10
        "min_zoom": 3.,
        "eval_render_envs": list(range(5)),
        "hall_width_p": 0.26,
    },
    # "hallway": {"num_agents": [10, 20, 30], "world_size": 15, "min_zoom": 3.6},
    "room": {
        "num_agents": [8, 12, 25],
        "world_size": 15,
        "num_eval_envs": 10,
        "gp_lookahead": 5,
        "min_zoom": 3.,
    },
    "multi": multi_config,  # drawing multiple scenarios in one environment (needed for training)

    "localMinima": {  # Used for showcasing the local minima problem without GP
        "num_agents": [1],
        "world_size": 10,
        "min_zoom": 2.0,
        "num_eval_envs": 1,
        "eval_render_envs": [0],
        "max_steps": 600
    },
    "GPFocus": {  # Used for showcasing the increased focus on the temporary goal can lead to local minima
        "num_agents": [2],
        "world_size": 10,
        "min_zoom": 2.0,
        "num_eval_envs": 1,
        "eval_render_envs": [0],
        "max_steps": 600
    },
}

baseline_config = {
    "use_global_path_obs": False,
    "set_gp_as_goal": set_gp_as_goal,  #
    "gp_lookahead": lookahead,
    "target_point": goal_point,  # "gp" or "goal"
    # Observations
    "use_lidar": True,
    "num_lidar_rays": 512,
    "lidar_angle_start": np.deg2rad(-90),  # 0,  # rads
    "lidar_angle_end": np.deg2rad(90),  # math.pi * 2,  # rads
    # 360 but we are merge 3 lines (mean) on a real robot  # https://emanual.robotis.com/docs/en/platform/turtlebot3/appendix_lds_01/
    "lidar_range": 4,  # meter
    "lidar_history_len": 3,
    "lidar_noise": 0.035,  # 0.035,  # Distance Precision(500mm ~ 3,500mm) = ±3.5%
    # rewards
    "collision_penalty": -15,
    "final_reward": 15,
    "personal_space_penalty": 0,
    "pos_shaping_factor": 2.5,
    "value_loss_factor": 20.0,  # to make the learning faster than the policy
}

GA3CPolicy_config = {
    "object_vert_inflate_radius": 0.15,  # 0.05 for doorway or it will be horrible, but else set it to 0.15
    "omega_limit": 6.,  # the model is trained for this limit, so we are allowing it to go to the limit
    "render_lidar": False,
    "use_global_path_obs": False,
    "set_gp_as_goal": set_gp_as_goal,
    "gp_lookahead": lookahead,
    "target_point": goal_point,  # "gp" or "goal"
    "device": "cpu",
}

RVO_config = {
    "render_lidar": False,
    "use_global_path_obs": False,
    "set_gp_as_goal": set_gp_as_goal,
    "gp_lookahead": lookahead,
    "target_point": goal_point,  # "gp" or "goal"
    "device": "cpu",
}

config = {
    "model": "oursGraph",  # "baseline", "oursD", "oursGraph", RVO
    # baseline, oursD (ours with distance), oursDV (ours with distance and velocity), oursGraph (ours with GNN and distance)
    "fine_tune_from": load_model,
    "max_steps": int(600),  # 256 * 2 = 512 steps * 0.1 = 51.2s, 600 * 2 = 1200 steps * 0.1 = 120s
    # Env
    "scenario_type": "multi",
    "storing_device": "cpu",
    "device": "cuda:0",
    "dt": 0.1,
    "num_agents": 10,
    "num_obstacles": 10,
    "use_global_path_obs": True,
    "set_gp_as_goal": False,
    "gp_lookahead": lookahead,
    "target_point": "gp",  # "gp" or "goal"
    # Observations
    "use_lidar": True,
    "num_lidar_rays": 120,
    "lidar_angle_start": 0,  # rads
    "lidar_angle_end": math.pi * 2,  # rads
    # 360 but we are merge 3 lines (mean) on a real robot  # https://emanual.robotis.com/docs/en/platform/turtlebot3/appendix_lds_01/
    "lidar_range": 3.5,  # meter
    "lidar_history_len": 3,
    "lidar_noise": 0.035,  # 0.035,  # Distance Precision(500mm ~ 3,500mm) = ±3.5%
    "other_agent_noise": 0.1,  # 0.1,  # meter
    "omega_limit": 1.,
    "v_limit": 1.,
    "use_polar_coordinates": True,
    "cooperative_dist": 2.0,  # meter
    "dist_type": "IndependentNormal",  # IndependentNormal, TanhNormal, beta
    # rewards:
    "pos_shaping_factor": 2.5,
    "time_penalty": -0.00,
    "final_reward": 15,
    "collision_penalty": -25,
    "cooperative_factor": 0,
    "personal_space_penalty": -1.,
    "personal_space_distance": 0.3,
    # Dist need to be more than a collision can occure (two agents running straight into eachother).
    # logging
    "log": False,
    "save_video_to_disk": True,
    # Eval
    "eval_seed": 0,
    "eval_every": 5,
    "num_eval_envs": 10,
    "eval_render_envs": list(range(5)),
    "eval_render_to_screen": False,
    "ppo": {
        "std_max_start": 0.5,
        "std_max_end": 0.5,
    },  # not used but needed to initiate models.

    # Drawing
    "draw_lookahead": False,
    "draw_gp_as_circles": True,
    "gp_circle_size": 0.05,
    "draw_all_gp": False,
    "draw_gp_target_index": -1,  # -1 means none
    "draw_info_text": False,
    "render_lidar": False,
    "draw_action_forces": False,
    "draw_personal_space": False,

    #
    "viewer_size": (1600, 1200)
}


def make_env(cfg: dict, max_steps):
    scenario_type = cfg.get("scenario_type", "random")
    if scenario_type == "multi":
        cfg.update(multi_config)

    scenario = valid_scenario_types[scenario_type](config=cfg)
    if scenario_type == "multi":
        cfg["num_agents"] = scenario.num_agents

    vmas_device = (
        th.device(0)
        if th.cuda.is_available() and not multiprocessing.get_start_method() == "fork"
        else th.device("cpu")
    )
    vmas_device = th.device(config.get("device", vmas_device))
    env = VmasEnv(
        scenario=scenario,
        num_envs=int(cfg.get("num_eval_envs", 1)),
        continuous_actions=True,  # VMAS supports both continuous and discrete actions
        max_steps=int(max_steps),
        device=vmas_device,
        # Scenario kwargs
        n_agents=cfg["num_agents"],
        # These are custom kwargs that change for each VMAS scenario, see the VMAS repo to know more.
        clamp_actions=True,
    )

    env = TransformedEnv(
        env,
        Compose(
            RewardSum(
                in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]
            ),
            StepCounter(),
        ),
    )
    return env


def main(args=None):
    global checkpoints, valid_scenario_types, load_model
    save_video = True
    output_root = None
    if args is not None:
        from train.config import load_config_file, merge_overrides

        file_cfg = load_config_file(args.config) if args.config else {}

        def pick(flag_val, key, default):
            # Precedence: explicit flag > config file > built-in default.
            if flag_val is not None:
                return flag_val
            if file_cfg.get(key) is not None:
                return file_cfg[key]
            return default

        if file_cfg.get("config"):
            config.update(merge_overrides(config, file_cfg["config"]))
        for _scen, _ov in (file_cfg.get("scenario_config") or {}).items():
            scenario_config[_scen] = merge_overrides(scenario_config.get(_scen, {}), _ov)

        model_type = pick(args.model_type, "model_type", "oursGraph")
        checkpoint = pick(args.checkpoint, "checkpoint", load_model)
        scenarios = pick(args.scenario, "scenarios", None)
        num_agents = pick(args.num_agents, "num_agents", None)
        max_steps = pick(args.max_steps, "max_steps", None)
        num_eval_envs = pick(args.num_eval_envs, "num_eval_envs", None)
        device = pick(args.device, "device", None)
        seed = pick(args.seed, "seed", None)
        save_video = pick(args.video, "save_video", True)
        output_root = pick(args.output_dir, "output_dir", None)

        learned = {"oursGraph", "oursD", "oursDV", "baseline"}
        if model_type in learned and not checkpoint:
            sys.exit(
                f"error: a checkpoint is required for model-type '{model_type}'. "
                "Pass --checkpoint or set 'checkpoint:' in --config "
                "(e.g. models/checkpoints/ours/OurGraphModel.pth)."
            )
        unknown = [s for s in (scenarios or []) if s not in ALL_SCENARIOS]
        if unknown:
            sys.exit(
                f"error: unknown scenario(s) {unknown}; "
                f"choose from {list(ALL_SCENARIOS)}."
            )

        config["model"] = model_type
        load_model = checkpoint
        checkpoints = {load_model: model_type}
        if scenarios:
            valid_scenario_types = {s: ALL_SCENARIOS[s] for s in scenarios}
        if max_steps is not None:
            config["max_steps"] = max_steps
        if num_eval_envs is not None:
            config["num_eval_envs"] = num_eval_envs
        if device is not None:
            config["device"] = device
        if seed is not None:
            config["eval_seed"] = seed
        if num_agents is not None:
            for _scen in valid_scenario_types:
                scenario_config.setdefault(_scen, {})
                scenario_config[_scen]["num_agents"] = num_agents

        if args.dry_run:
            print("Resolved paper-video plan:")
            if args.config:
                print(f"  config file: {args.config}")
            print(f"  model_type : {model_type}")
            print(f"  checkpoint : {load_model}")
            print(f"  scenarios  : {list(valid_scenario_types)}")
            print(f"  max_steps  : {config.get('max_steps')}")
            print(f"  device     : {config.get('device')}")
            print(f"  seed       : {config.get('eval_seed')}")
            print(f"  save_video : {save_video}")
            print(f"  output_dir : {output_root or os.getcwd()}")
            for _scen in valid_scenario_types:
                print(
                    f"    {_scen}: num_agents="
                    f"{scenario_config.get(_scen, {}).get('num_agents')}"
                )
            return

    np.random.seed(config.get("eval_seed", 0))
    # th.autograd.set_detect_anomaly(False)
    th.backends.cudnn.deterministic = True
    # th.backends.cudnn.benchmark = True
    th.manual_seed(config.get("eval_seed", 0))
    if th.cuda.is_available():
        th.cuda.manual_seed(config.get("eval_seed", 0))
        th.cuda.manual_seed_all(config.get("eval_seed", 0))

    # sort but so it is 0, 1, 2, 3, 4,...10, 11... instead of 0, 1, 10, 11, 2, 3, 4
    result_dict = {}
    for checkpoint, model_type in checkpoints.items():
        print(f"Model type: {model_type}")
        print(f"Checkpoint: {checkpoint}")
        cfg = config.copy()
        cfg["model"] = model_type
        for scen_name, _ in valid_scenario_types.items():
            result_dict[scen_name] = {}
            ###### SETUP SCENARIO ######
            scen_config = cfg.copy()
            scen_config.update(scenario_config[scen_name])
            scen_config["scenario_type"] = scen_name
            num_agents_list = scen_config.get("num_agents")
            max_steps = scen_config.get("max_steps", 200)  # Episode steps before done
            if isinstance(num_agents_list, int):
                num_agents_list = [num_agents_list]
            for num_agents in num_agents_list:
                result_dict[scen_name][num_agents] = {}
                scen_config["num_agents"] = num_agents
                if scen_name == "multi":
                    eval_env = make_env(scen_config, max_steps)
                    scen_config["num_agents"] = eval_env.scenario.num_agents
                    del eval_env

                result_dict[scen_name][num_agents][checkpoint] = {}
                print("")
                if scen_config.get("model") == "RVO":  # no run name
                    run_name = "RVO"
                    checkpoint_name = prefix + f"RVO_{RVO_COLLAB_COEFF}_{RVO_TIME_HORIZON}"
                    scen_config.update(RVO_config)
                elif scen_config.get("model") == "GA3CPolicy":
                    run_name = "GA3C_CADRL"
                    checkpoint_name = prefix + (checkpoint or "GA3C_CADRL")
                    scen_config.update(GA3CPolicy_config)
                elif scen_config.get("model") == "baseline":
                    run_name = "baseline"
                    checkpoint_name = prefix + checkpoint
                    scen_config.update(baseline_config)
                else:  # oursGraph
                    run_name = (
                            Path(checkpoint).parent.parent.parent.stem
                            + "_"
                            + Path(checkpoint).parent.parent.stem
                    )
                    checkpoint_name = prefix + Path(checkpoint).stem

                save_dir = Path(output_root) if output_root else Path(os.getcwd())

                video_folder = Path(save_dir, "videos")
                if extra_folder is not None:
                    video_folder = Path(video_folder, extra_folder)

                if not video_folder.exists():
                    video_folder.mkdir(parents=True, exist_ok=True)

                scen_config["use_global_path"] = (
                        scen_config.get("use_global_path_obs", False)
                        or scen_config.get("set_gp_as_goal", False)
                        or scen_config.get("target_point") == "gp"
                )

                device = (
                    th.device(0) if th.cuda.is_available() else th.device("cpu")
                )
                device = th.device(scen_config.get("device", device))
                eval_env = make_env(scen_config, max_steps)
                current_step, policy, _ = make_network(eval_env, scen_config, checkpoint, None, device)

                print(f"env: '{scen_name}' is create and ready for evaluating")
                eval_env.scenario.print_env_info()
                start = time.perf_counter()
                log, eval_str = evaluate(
                    eval_env,
                    policy,
                    max_steps,
                    current_step,
                    None,
                    save_video=save_video,
                    save_folder=str(video_folder),
                    save_name=scen_name + f"_{prefix}{model_type}.mp4",
                    eval_seed=cfg.get("eval_seed", 0),
                    eval_render_to_screen=False,
                    render_env_index=scen_config.get("eval_render_envs"),
                )
                result_dict[scen_name][num_agents][checkpoint][cfg.get("eval_seed")] = log
                print(eval_str, "time: ", time.perf_counter() - start)
                # write to local
                del policy, current_step, eval_env


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Render showcase videos of a trained policy (paper scenarios)."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a YAML/JSON config (see configs/paper_videos.yaml). "
             "Precedence: script defaults < config file < explicit flags.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to the model .pth. Required for learned models "
             "(oursGraph/oursD/oursDV/baseline) unless set in --config. "
             "e.g. models/checkpoints/ours/OurGraphModel.pth.",
    )
    parser.add_argument(
        "--model-type", type=str, default=None,
        choices=["oursGraph", "oursD", "oursDV", "baseline", "RVO", "GA3CPolicy"],
        help="Policy type to render (default: oursGraph).",
    )
    parser.add_argument(
        "--scenario", nargs="+", default=None, choices=list(ALL_SCENARIOS),
        help="Scenario(s) to render (default: random circle doorway hallway).",
    )
    parser.add_argument(
        "--num-agents", type=int, nargs="+", default=None,
        help="Override the per-scenario agent count(s).",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Override the episode length in steps.",
    )
    parser.add_argument(
        "--num-eval-envs", type=int, default=None,
        help="Number of parallel environments.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device, e.g. cuda:0 or cpu (default: auto).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random/eval seed.")
    parser.add_argument(
        "--video", action=argparse.BooleanOptionalAction, default=None,
        help="Render/save videos (use --no-video for a quick sanity check).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for the rendered videos (default: current directory).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the resolved plan and exit.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args())
