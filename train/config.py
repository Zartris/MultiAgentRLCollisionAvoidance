"""Training-config defaults + loader.

Split out from LidarSingleStep.py so agents (and CLIs) can build config dicts
without needing to mutate module-level globals in the training script.

Public API:

  build_default_config() -> dict
      Returns a fresh copy of the canonical default training config. Equivalent
      to what the module-level `config`, `ppo_config`, `multi_config`,
      `multi_config_eval`, `baseline_config` used to produce, collapsed into a
      single nested dict.

  merge_overrides(base: dict, overrides: dict) -> dict
      Deep-merge `overrides` into `base`. For nested dicts, keys are merged;
      for every other type the override replaces the base. Returns the merged
      result — does NOT mutate inputs.

  load_config_file(path) -> dict
      Read a YAML (or JSON if .json extension) file and return a dict ready
      for `merge_overrides(build_default_config(), ...)`.

  resolve_config(path_or_none, overrides=None) -> dict
      Shortcut: defaults -> optional YAML file -> optional dict overrides.
      This is what CLIs should use.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np


def _baseline_config() -> dict:
    return {
        "use_global_path_obs": False,
        "set_gp_as_goal": False,
        "gp_lookahead": 5,
        "target_point": "goal",
        "use_lidar": True,
        "num_lidar_rays": 512,
        "lidar_angle_start": float(np.deg2rad(-90)),
        "lidar_angle_end": float(np.deg2rad(90)),
        "lidar_range": 4,
        "lidar_history_len": 3,
        "lidar_noise": 0.035,
        "collision_penalty": -15,
        "final_reward": 15,
        "personal_space_penalty": 0,
        "pos_shaping_factor": 2.5,
        "value_loss_factor": 20.0,
    }


def _multi_config() -> dict:
    return {
        "random": {"num_agents": 20, "num_obstacles": 6, "obstacle_size": 1, "world_size": 10},
        "circle": {"num_agents": 15, "spawn_mode": "equally_spaced", "world_size": 10},
        "plus": {"num_agents": 4},
        "doorway": {"num_agents": 5, "world_size": 10},
        "corridor": {"num_agents": 5},
        "hallway": {"num_agents": 8, "world_size": 10},
        "room": {"num_agents": 8, "num_obstacles": 10, "world_size": 10},
    }


def _multi_config_eval() -> dict:
    mc = _multi_config()
    return {
        "random": mc["random"],
        "circle": {"num_agents": 40, "spawn_mode": "equally_spaced", "world_size": 15},
        "plus": mc["plus"],
        "doorway": mc["doorway"],
        "corridor": mc["corridor"],
        "hallway": mc["hallway"],
        "room": mc["room"],
        "ppo": {"max_steps": 256 * 2},
    }


def _ppo_config(num_worlds: int, max_steps: int) -> dict:
    return {
        "frames_per_batch": int(max_steps * num_worlds),
        "n_iters": 2000,
        "max_steps": max_steps,
        "num_epochs": 10,
        "minibatch_size": 50,
        # collect_chunks: how many memory-chunks per-iter collection is split into
        # (separate from minibatch_size). Raise minibatch_size for faster training
        # WITHOUT changing this, so collection stays fast.
        "collect_chunks": 50,
        "gae_batch_size": int(max_steps),
        "lr_start": 2e-5,
        "lr_end": 2e-5,
        # GIANT (Table I) uses an asymmetric learning rate: actor 2e-5, critic
        # 4e-4. Adam neutralises value_loss_factor (it normalises per-param), so
        # the critic rate must come from its own optimizer param group, not the
        # loss weight. Set to None to train both at lr_start.
        "critic_lr": 4e-4,
        "max_grad_norm": -1,
        "clip_epsilon": 0.2,
        "gamma": 0.99,
        "lmbda": 0.95,
        "entropy_eps_start": 1e-4,
        "entropy_eps_end": 1e-4,
        "std_max_start": 0.5,
        # Floor for the annealed std ceiling. Kept at ~0.05 (not ~0) so the policy
        # retains some stochasticity late in each level — avoids a fully-greedy,
        # exploration-dead policy (eval uses the MEAN action, so deployment is still
        # effectively deterministic). Paired with the per-level reset of this schedule.
        "std_max_end": 0.05,
        # Per-level exploration anneal length (iters from each level's start, since the
        # schedule resets per curriculum level). 250 (not 100) so the HARD level (L3,
        # 40-agent circle) keeps substantial exploration long enough to DISCOVER the
        # yield/deviate behavior before annealing down to refine — lets a clean
        # from-scratch curriculum run crack L3 in one go without a manual re-warm.
        # Easy levels (L0-L2) master in 20-40 iters, well before this matters.
        "std_max_decay_steps": 250,
        "value_loss_factor": 20.0,
        "policy_loss_factor": 1.0,
        # Default = clipped PPO. This uses the repo's CUSTOM ClipPPOLoss
        # (train/utils/PPOLoss.py), which masks padded/terminated agents (is_padding)
        # in both the critic and policy loss. DO NOT switch back to "kl_pen_ppo" until
        # the masking bug below is fixed: the kl_pen path uses stock torchrl
        # KLPENPPOLoss, which does NOT mask is_padding, so dead/terminated agents
        # corrupt the value targets (critic trained toward 0 on their still-valid obs)
        # and inject a spurious policy objective — degrades multi-agent (L1+) learning.
        # See [[kl-pen-missing-padding-mask]] in agent memory.
        "loss_module": "clip",
        "kl_target": 1e-3,
        "kl_init_div": 1.0,
        "kl_increment": 2.0,
        "kl_decrement": 0.5,
    }


def build_default_config() -> dict:
    """Return a fresh dict with the canonical training defaults.

    Safe to mutate — the caller owns the result. No module-level state.
    """
    num_worlds = 150
    max_steps = 400

    return {
        # Checkpoint + logging
        "load_model": None,
        "load_only_model": False,
        # Fine-tuning: when True, train only the params NOT matching finetune_freeze
        # (with a fresh optimizer). Used to retrain the corrected `emb` attention on
        # top of a model trained with the dead attention, keeping the lidar encoders
        # fixed. Enabled in one shot by the --finetune CLI flag.
        "finetune": False,
        "finetune_freeze": ["lidar_static_encoder", "lidar_dynamic_encoder"],
        "log": True,
        "logging_backend": "wandb",           # "wandb" / "file" / "none"
        "save_video_to_disk": True,

        # Training loop shape
        "num_worlds": num_worlds,
        "base_seed": 0,
        "max_runtime_hours": 24,              # cost guard; nanny enforces

        # Perf knobs (no model change). tf32 is ~fp32-accurate (verified rel-diff
        # ~5e-5) and default ON: ~-11% training step + speeds the collection policy
        # forward. cudnn_benchmark is OFF by default: the multi scenario has variable
        # conv batch sizes (per-scenario agent counts), so benchmark caches a workspace
        # per shape and OOMs a 12GB GPU over a full run. cudnn_deterministic kept ON
        # (original behaviour). amp_dtype enables autocast for the PPO fwd/bwd: "bf16"
        # measured ~-22% but reduces gradient precision, so OFF until a convergence run
        # validates it. Set to "bf16"/"fp16".
        "perf": {"tf32": True, "cudnn_benchmark": False, "cudnn_deterministic": True, "fused_adam": True, "amp_dtype": "bf16"},

        # Env / scenario
        "model": "oursGraph",
        # Peer-attention in AgentGraphNet: "none" (legacy sum-pool / no-op gate,
        # loads old checkpoints exactly), "raw" (attention over peers from raw
        # features) or "emb" (attention over peers from the encoded embedding).
        # Default is "emb" (the corrected attention) for fresh training; use "none"
        # only when loading an old (pre-attention) checkpoint that must match exactly.
        "gnn_attention": "emb",
        "fine_tune_from": None,
        "scenario_type": "multi",
        "storing_device": "cpu",
        "dt": 0.1,
        "num_agents": 10,
        "num_obstacles": 10,
        "use_global_path_obs": True,
        "set_gp_as_goal": False,
        "gp_lookahead": 5,
        "target_point": "gp",

        # Observations
        "use_lidar": True,
        "num_lidar_rays": 120,
        "lidar_angle_start": 0,
        "lidar_angle_end": float(math.pi * 2),
        "lidar_range": 3.5,
        "lidar_history_len": 3,
        "lidar_noise": 0.035,
        "omega_limit": 1,
        "v_limit": 1,
        "use_polar_coordinates": True,
        "cooperative_dist": 2.0,
        "dist_type": "IndependentNormal",

        # Rewards
        "pos_shaping_factor": 2.5,
        # Reward shape from the bug-fixed run (the publishable model). With the
        # truncation-as-termination + freeze bugs fixed, the agent froze under the
        # old collision==goal ratio, so: time_penalty -0.03 (a real per-step cost so
        # standing still isn't free) and collision_penalty -10 (< final_reward 25, so
        # a goal-run is +EV). NOTE: differs from the paper's Table I (r_goal 15,
        # r_collision -25) — reconcile the paper or revisit now that bugs are fixed.
        "time_penalty": -0.03,
        "final_reward": 25,
        "collision_penalty": -10,
        "cooperative_factor": 0,
        "personal_space_penalty": -0.1,
        "personal_space_distance": 0.3,

        # Eval
        "eval_seed": 0,
        "eval_every": 10,
        # Eval env is heavier per-world than training (multi eval `circle` has 40
        # agents vs 15) and is held in GPU memory alongside the training env. On a
        # 12GB GPU, no-freeze full training fits with 16 (measured ~7.7GB); 32 OOMs
        # at the iter boundary. Raise on a bigger GPU.
        "num_eval_envs": 16,
        "eval_render_to_screen": False,
        # Which eval env indices to render into videos. Each extra index is a full
        # extra render pass per frame (~6min/video per env at real eval scale), so
        # default to 1. With scenario_type=multi, indices map to different scenarios.
        "render_env_index": [0],
        # Metrics run every eval_every (no render); videos render only every
        # video_every. Keep video_every a multiple of eval_every.
        "video_every": 20,

        # Nested config blocks
        "baseline_config": _baseline_config(),
        "multi_config": _multi_config(),
        "multi_config_eval": _multi_config_eval(),
        "ppo": _ppo_config(num_worlds, max_steps),
    }


def merge_overrides(base: Mapping, overrides: Mapping) -> dict:
    """Deep-merge overrides into base. Returns a new dict; does not mutate inputs.

    For any key present in both `base` and `overrides`:
      - If both values are dicts, merge recursively.
      - Otherwise the overrides value wins.

    Keys in `overrides` that are not in `base` are still added (agents may
    legitimately add new knobs). A warning is printed in that case since it is
    also a common source of typos.
    """
    result = copy.deepcopy(dict(base))
    for key, override_val in overrides.items():
        if key not in result:
            print(f"[config] new key introduced by override: {key!r}")
            result[key] = copy.deepcopy(override_val)
            continue
        base_val = result[key]
        if isinstance(base_val, dict) and isinstance(override_val, Mapping):
            result[key] = merge_overrides(base_val, override_val)
        else:
            result[key] = copy.deepcopy(override_val)
    return result


def load_config_file(path) -> dict:
    """Read a YAML or JSON config file into a dict.

    YAML support uses PyYAML if available. A clear error is raised if it is not
    installed and a .yaml / .yml file is requested.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file does not exist: {p}")
    text = p.read_text()
    ext = p.suffix.lower()
    if ext == ".json":
        return json.loads(text)
    if ext in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load YAML configs. "
                "Install with `pip install pyyaml`."
            ) from exc
        return yaml.safe_load(text) or {}
    raise ValueError(f"Unsupported config file extension: {ext!r}. Use .yaml, .yml, or .json.")


def resolve_config(
    config_path: Optional[str] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Compose the effective config: defaults -> YAML file -> dict overrides.

    Typical CLI use::

        cfg = resolve_config(args.config, overrides={"num_worlds": args.worlds})
        run_training(cfg)

    Order of precedence (later wins): defaults < file < programmatic overrides.
    """
    cfg = build_default_config()
    if config_path is not None:
        cfg = merge_overrides(cfg, load_config_file(config_path))
    if overrides:
        cfg = merge_overrides(cfg, overrides)
    return cfg
