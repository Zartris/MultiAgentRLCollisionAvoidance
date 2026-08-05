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


# Loss

# Utils
th.manual_seed(0)

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from models.baseline.rvo_policy import RVO_COLLAB_COEFF, RVO_TIME_HORIZON
from models.model_loader import make_network
from scenario.PaperScenarioes.CollisionAvoidance_local_minima import (
    CollisionAvoidanceLocalMinima,
)
from scenario.CollisionAvoidance_room import CollisionAvoidanceRoom
from scenario.CollisionAvoidance_circle import CollisionAvoidanceCircle
from scenario.CollisionAvoidance_corridor import CollisionAvoidanceCorridor
from scenario.CollisionAvoidance_hallway import CollisionAvoidanceHallway
from scenario.CollisionAvoidance_random import CollisionAvoidanceRandom
from scenario.CollisionAvoidance_doorway import CollisionAvoidanceDoorway
from scenario.CollisionAvoidance_multi import CollisionAvoidanceMultiEnv
from scenario.PaperScenarioes.CollisionAvoidance_gp_focus import (
    CollisionAvoidanceGPFocus,
)

from train.utils.common import evaluate

hw = 0.26
prefix = f"hw{str(hw).replace('.', '_')}"
# Set to the absolute path of your checkpoint file before running.
load_model = None  # e.g. "models/checkpoints/ours/OurGraphModel.pth"
baseline_model = None  # e.g. "train/results/2024-07-24/baseline_07h-03m-34s/checkpoints/baseline_net.pth"
checkpoints = {
    load_model: "oursGraph",
    # "RVO_v2": "RVO",
    # baseline_model: "baseline",
    # "GA3C-CARL": "GA3CPolicy",
}

test_full_runs = "results/2024-07-10/baseline_06h-55m-59s/checkpoints_combined/"
# test_full_runs = "results/2024-07-07/20h-49m-45s/checkpoints"

valid_scenario_types = {
    # "multi": CollisionAvoidanceMultiEnv,  # all eval environments (random, circle, door, hallway)
    "random": CollisionAvoidanceRandom,  # 7 in paper
    # "circle": CollisionAvoidanceCircle,  # 4 in paper
    # "plus": CollisionAvoidancePlus,  # 1 in paper
    "doorway": CollisionAvoidanceDoorway,  # 3 in paper
    # "corridor": CollisionAvoidanceCorridor,  # 2 in paper
    "hallway": CollisionAvoidanceHallway,  # 5 in paper
    "room": CollisionAvoidanceRoom,  # 6 in paper
}

# Full registry of selectable scenarios for the --scenario CLI flag.
ALL_SCENARIOS = {
    "random": CollisionAvoidanceRandom,
    "circle": CollisionAvoidanceCircle,
    "corridor": CollisionAvoidanceCorridor,
    "hallway": CollisionAvoidanceHallway,
    "doorway": CollisionAvoidanceDoorway,
    "room": CollisionAvoidanceRoom,
    "multi": CollisionAvoidanceMultiEnv,
}

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
        "world_size": 15,
    },  # "equally_spaced" or "random"
    "plus": {"num_agents": 4},  # 4
    "doorway": {"num_agents": 15, "world_size": 15},  # 5
    "corridor": {"num_agents": 5},  # 5
    "hallway": {"num_agents": 16, "world_size": 15},  # 8
    "room": {"num_agents": 13, "num_obstacles": 10},
}

scenario_config = {
    "random": {
        "num_agents": [10, 20, 40],
        "num_obstacles": 8,
        "obstacle_size": 1,
        "world_size": 15,
        "min_zoom": 3.6,
        "eval_render_envs": list(range(1)),
    },
    "circle": {
        "num_agents": [10, 20, 40],
        "num_eval_envs": 1,
        "world_size": 15,
        "spawn_mode": "equally_spaced",
        "num_obstacles": 0,
        "min_zoom": 3.6,
    },
    "plus": {"num_agents": 4, "world_size": 10, "min_zoom": 3.6},
    "doorway": {
        "num_agents": [5, 10, 15],
        "world_size": 15,
        "num_eval_envs": 10,
        "min_zoom": 3.6,
        "eval_render_envs": list(range(1)),
    },
    "corridor": {"num_agents": 5, "world_size": 10, "min_zoom": 3.6},
    "hallway": {
        "num_agents": [8, 12, 16],  # [8, 12, 16],
        "world_size": 15,  # 15
        "num_eval_envs": 10,
        "min_zoom": 3.6,
        "eval_render_envs": list(range(1)),
        "hall_width_p": hw,
    },
    # "hallway": {"num_agents": [10, 20, 30], "world_size": 15, "min_zoom": 3.6},
    "room": {
        "num_agents": [8, 12, 25],
        "world_size": 15,
        "num_eval_envs": 10,
        "gp_lookahead": 5,
        "min_zoom": 3.6,
        "eval_render_envs": list(range(1)),
    },
    "multi": multi_config,  # drawing multiple scenarios in one environment (needed for training)
}

baseline_config = {
    "use_global_path_obs": False,
    "set_gp_as_goal": True,  #
    "gp_lookahead": 5,
    "target_point": "gp",  # "gp" or "goal"
    # Observations
    "use_lidar": True,
    "num_lidar_rays": 512,
    "lidar_angle_start": np.deg2rad(-90),  # 0,  # rads
    "lidar_angle_end": np.deg2rad(90),  # math.pi * 2,  # rads
    # 360 but we are merge 3 lines (mean) on a real robot  # https://emanual.robotis.com/docs/en/platform/turtlebot3/appendix_lds_01/
    "lidar_range": 4,  # meter
    "lidar_history_len": 3,
    "lidar_noise": 0.035,  # Distance Precision(500mm ~ 3,500mm) = ±3.5%
    # rewards
    "collision_penalty": -15,
    "final_reward": 15,
    "personal_space_penalty": 0,
    "pos_shaping_factor": 2.5,
    "value_loss_factor": 20.0,  # to make the learning faster than the policy
    "device": "cuda:0",
}

GA3CPolicy_config = {
    "object_vert_inflate_radius": 0.15,  # 0.05 for doorway or it will be horrible, but else set it to 0.15
    "omega_limit": 6.0,  # the model is trained for this limit, so we are allowing it to go to the limit
    "render_lidar": False,
    "use_global_path_obs": False,
    "set_gp_as_goal": True,
    "gp_lookahead": 5,
    "target_point": "gp",  # "gp" or "goal"
    "device": "cpu",
}

RVO_config = {
    "render_lidar": False,
    "use_global_path_obs": False,
    "set_gp_as_goal": True,
    "gp_lookahead": 5,
    "target_point": "gp",  # "gp" or "goal"
}

config = {
    "model": "oursGraph",  # "baseline", "oursD", "oursGraph", RVO
    # baseline, oursD (ours with distance), oursDV (ours with distance and velocity), oursGraph (ours with GNN and distance)
    "fine_tune_from": load_model,
    "max_steps": int(
        600 * 2
    ),  # 256 * 2 = 512 steps * 0.1 = 51.2s, 600 * 2 = 1200 steps * 0.1 = 120s
    # Env
    "scenario_type": "multi",
    "storing_device": "cpu",
    "device": "cuda:0",
    "dt": 0.1,
    "num_agents": 10,
    "num_obstacles": 10,
    "use_global_path_obs": True,
    "set_gp_as_goal": False,
    "gp_lookahead": 5,
    "target_point": "gp",  # "gp" or "goal"
    # Observations
    "use_lidar": True,
    "num_lidar_rays": 120,
    "lidar_angle_start": 0,  # rads
    "lidar_angle_end": math.pi * 2,  # rads
    # 360 but we are merge 3 lines (mean) on a real robot  # https://emanual.robotis.com/docs/en/platform/turtlebot3/appendix_lds_01/
    "lidar_range": 3.5,  # meter
    "lidar_history_len": 3,
    "lidar_noise": 0.035,  # Distance Precision(500mm ~ 3,500mm) = ±3.5%
    "other_agent_noise": 0.1,  # meter
    "omega_limit": 1.0,
    "v_limit": 1.0,
    "use_polar_coordinates": True,
    "cooperative_dist": 2.0,  # meter
    "dist_type": "IndependentNormal",  # IndependentNormal, TanhNormal, beta
    # rewards:
    "pos_shaping_factor": 2.5,
    "time_penalty": -0.00,
    "final_reward": 15,
    "collision_penalty": -25,
    "cooperative_factor": 0,
    "personal_space_penalty": -1.0,
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
    "draw_lookahead": True,
    "draw_gp_as_circles": True,
    "gp_circle_size": 0.05,
    "draw_all_gp": False,
    "draw_gp_target_index": 0,  # -1 means none
    "draw_info_text": False,
    "render_lidar": True,
    "draw_action_forces": False,
    "draw_personal_space": False,
    #
    # "viewer_size": (800, 600)
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

        # Deep overrides into the base config / per-scenario config.
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
            print("Resolved eval plan:")
            if args.config:
                print(f"  config file  : {args.config}")
            print(f"  model_type   : {model_type}")
            print(f"  checkpoint   : {load_model}")
            print(f"  scenarios    : {list(valid_scenario_types)}")
            print(f"  max_steps    : {config.get('max_steps')}")
            print(f"  num_eval_envs: {config.get('num_eval_envs')}")
            print(f"  device       : {config.get('device')}")
            print(f"  seed         : {config.get('eval_seed')}")
            print(f"  save_video   : {save_video}")
            for _scen in valid_scenario_types:
                print(
                    f"    {_scen}: num_agents="
                    f"{scenario_config.get(_scen, {}).get('num_agents')}"
                )
            return

    eval_results_root = (
        output_root if output_root else os.path.join(os.getcwd(), "results", "eval")
    )
    np.random.seed(config.get("eval_seed", 0))
    th.autograd.set_detect_anomaly(False)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = True
    th.manual_seed(config.get("eval_seed", 0))
    if th.cuda.is_available():
        th.cuda.manual_seed(config.get("eval_seed", 0))
        th.cuda.manual_seed_all(config.get("eval_seed", 0))

    # checkpoint = load_model if config.get("model") != "baseline" else baseline_model

    # sort but so it is 0, 1, 2, 3, 4,...10, 11... instead of 0, 1, 10, 11, 2, 3, 4
    result_dict = {}
    for checkpoint, model_type in checkpoints.items():
        print(f"Model type: {model_type}")
        print(f"Checkpoint: {checkpoint}")
        cfg = config.copy()
        if model_type == "baseline":
            cfg.update(baseline_config)

        cfg["use_global_path"] = cfg.get("use_global_path_obs", False) or cfg.get(
            "set_gp_as_goal", False
        )
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
                if scen_config.get("model") == "RVO":  # baseline, no checkpoint
                    run_name = "RVO"
                    checkpoint_name = (
                        prefix + f"RVO_{RVO_COLLAB_COEFF}_{RVO_TIME_HORIZON}"
                    )
                    scen_config.update(RVO_config)
                elif scen_config.get("model") == "GA3CPolicy":
                    run_name = "GA3C_CADRL"
                    checkpoint_name = prefix + (checkpoint or "GA3C_CADRL")
                    scen_config.update(GA3CPolicy_config)
                else:
                    run_name = (
                        Path(checkpoint).parent.parent.parent.stem
                        + "_"
                        + Path(checkpoint).parent.parent.stem
                    )
                    checkpoint_name = prefix + Path(checkpoint).stem

                save_dir = Path(
                    eval_results_root,
                    scen_config.get("model"),
                    run_name,
                    checkpoint_name,
                    scen_name + f"_{scen_config.get('num_agents')}",
                )
                video_folder = Path(save_dir, "videos")
                if not video_folder.exists():
                    video_folder.mkdir(parents=True, exist_ok=True)
                elif any(video_folder.iterdir()):
                    print(
                        f"The test: {run_name} '{scen_name}_{scen_config.get('num_agents')} already exists, delete or rename if you want this to run again."
                    )
                    continue

                device = th.device(scen_config.get("device", "cuda:0"))
                eval_env = make_env(scen_config, max_steps)
                current_step, policy, _ = make_network(
                    eval_env, scen_config, checkpoint, None, device
                )

                print(f"env: '{scen_name}' is create and ready for evaluating")

                eval_env.scenario.print_env_info()
                start = time.perf_counter()
                log, eval_str, _ = evaluate(
                    eval_env,
                    policy,
                    max_steps,
                    current_step,
                    None,
                    save_video=save_video,
                    save_folder=str(video_folder),
                    save_name=scen_name + f"_{checkpoint_name}.mp4",
                    eval_seed=cfg.get("eval_seed", 0),
                    eval_render_to_screen=False,
                    render_env_index=scen_config.get("eval_render_envs")
                    if scen_name != "circle"
                    else [0],
                )
                result_dict[scen_name][num_agents][checkpoint][cfg.get("eval_seed")] = (
                    log
                )
                print(eval_str, "time: ", time.perf_counter() - start)
                # write to local
                del policy, current_step, eval_env

                # result_dict[scen_name][num_agents][checkpoint][config.get("eval_seed")]
                # save dict as json file
                result_file = Path(save_dir, f"{scen_name}_results.json")

                with open(result_file, "w") as f:
                    json.dump(result_dict[scen_name][num_agents], f, indent=4)
        combined_result_file = Path(
            eval_results_root,
            cfg.get("model"),
            run_name,
            checkpoint_name,
            f"combined_results.json",
        )
        with open(combined_result_file, "w") as f:
            json.dump(result_dict, f, indent=4)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained policy across collision-avoidance scenarios."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a YAML/JSON eval config (see configs/eval.yaml). "
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
        help="Policy type to evaluate (default: oursGraph).",
    )
    parser.add_argument(
        "--scenario", nargs="+", default=None, choices=list(ALL_SCENARIOS),
        help="Scenario(s) to evaluate (default: random doorway hallway room).",
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
        help="Number of parallel evaluation environments.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device, e.g. cuda:0 or cpu (default: auto).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Evaluation seed.")
    parser.add_argument(
        "--video", action=argparse.BooleanOptionalAction, default=None,
        help="Render/save evaluation videos (use --no-video to disable).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for results + videos (default: ./results/eval).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the resolved evaluation plan and exit.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args())
