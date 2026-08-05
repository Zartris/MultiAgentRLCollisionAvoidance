"""Multi-agent RL training entry point.

The module-level config that used to live here (`config`, `ppo_config`,
`multi_config`, `multi_config_eval`, `baseline_config`, `_num_worlds`,
`_max_steps`, `load_model`, `load_only_model`) has moved to
``train.config.build_default_config``. Use ``resolve_config()`` there to
build a fresh config dict, then call ``run_training(cfg)`` below.

CLI:

    python train/LidarSingleStep.py                     # defaults
    python train/LidarSingleStep.py --config my.yaml    # YAML override
    python train/LidarSingleStep.py --config my.yaml --resume <ckpt.pth>
"""
from __future__ import annotations

import argparse
import math
import multiprocessing
import os
import sys
import tempfile
from typing import Optional

import numpy as np
import torch as th
from colorama import Fore, Style
from torch import multiprocessing
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage, LazyMemmapStorage
from torchrl.envs import RewardSum, TransformedEnv, StepCounter, Compose
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs
from torchrl.modules import TanhNormal, IndependentNormal
from torchrl.objectives import ValueEstimators, KLPENPPOLoss

# Add the parent directory to sys.path so the scenario / models modules resolve.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from models.model_loader import make_network  # noqa: E402
from scenario.CollisionAvoidance_circle import CollisionAvoidanceCircle  # noqa: E402
from scenario.CollisionAvoidance_corridor import CollisionAvoidanceCorridor  # noqa: E402
from scenario.CollisionAvoidance_doorway import CollisionAvoidanceDoorway  # noqa: E402
from scenario.CollisionAvoidance_hallway import CollisionAvoidanceHallway  # noqa: E402
from scenario.CollisionAvoidance_plus import CollisionAvoidancePlus  # noqa: E402
from scenario.CollisionAvoidance_random import CollisionAvoidanceRandom  # noqa: E402
from scenario.CollisionAvoidance_room import CollisionAvoidanceRoom  # noqa: E402
from scenario.CollisionAvoidance_multi import CollisionAvoidanceMultiEnv  # noqa: E402

from train.config import resolve_config  # noqa: E402
from train.utils.common import PPOTrainer, make_logger  # noqa: E402
from train.utils.schedulers import EntropyDecay, LRDecay, STDDecay  # noqa: E402
from train.utils.PPOLoss import ClipPPOLoss  # noqa: E402


# Class registry — this stays module-level because it's code, not config.
valid_scenario_types = {
    "random": CollisionAvoidanceRandom,
    "circle": CollisionAvoidanceCircle,
    "plus": CollisionAvoidancePlus,
    "doorway": CollisionAvoidanceDoorway,
    "corridor": CollisionAvoidanceCorridor,
    "hallway": CollisionAvoidanceHallway,
    "room": CollisionAvoidanceRoom,
    "multi": CollisionAvoidanceMultiEnv,
}


def make_env(cfg: dict, num_env: Optional[int] = None, check_env: bool = False,
             eval: bool = False):
    """Build a VmasEnv wrapped with RewardSum + StepCounter transforms.

    `cfg` is expected to contain at least: scenario_type, num_agents, ppo (with
    max_steps), and — when scenario_type == "multi" — `multi_config` /
    `multi_config_eval` nested dicts.
    """
    scenario_type = cfg.get("scenario_type", "random")
    if scenario_type == "multi":
        if eval:
            cfg.update(cfg.get("multi_config_eval", {}))
        else:
            cfg.update(cfg.get("multi_config", {}))

    scenario = valid_scenario_types[scenario_type](config=cfg)
    if scenario_type == "multi":
        cfg["num_agents"] = scenario.num_agents

    max_steps = cfg["ppo"]["max_steps"]
    if num_env is None:
        frames_per_batch = cfg["ppo"].get("frames_per_batch")
        num_env = frames_per_batch // max_steps

    vmas_device = (
        th.device(0)
        if th.cuda.is_available() and not multiprocessing.get_start_method() == "fork"
        else th.device("cpu")
    )
    vmas_device = th.device(cfg.get("device", vmas_device))

    env = VmasEnv(
        scenario=scenario,
        num_envs=int(num_env),
        continuous_actions=True,
        # max_steps is handled by the StepCounter transform below, NOT here.
        # VmasEnv folds its horizon into `done`, which TorchRL then copies into
        # `terminated` (vmas lib: terminated = done.clone()), with no `truncated`
        # key. GAE zeroes the value bootstrap on `terminated`, so a timeout would
        # be treated as a true terminal (value-to-go = 0) — the truncation-as-
        # termination bug that corrupts the critic on every timeout episode.
        # Leaving VMAS unbounded and truncating via StepCounter keeps
        # `terminated = goal|collision` (real termination) and emits a distinct
        # `truncated` key, so GAE bootstraps correctly at the horizon.
        max_steps=None,
        device=vmas_device,
        n_agents=cfg["num_agents"],
        clamp_actions=True,
    )
    print("action_spec:", env.full_action_spec)
    print("reward_spec:", env.full_reward_spec)
    print("done_spec:", env.full_done_spec)
    print("observation_spec:", env.observation_spec)
    print("action_keys:", env.action_keys)
    print("reward_keys:", env.reward_keys)
    print("done_keys:", env.done_keys)
    env = TransformedEnv(
        env,
        Compose(
            RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
            # Owns episode truncation now (see VmasEnv max_steps=None above): emits a
            # `truncated` key at the horizon distinct from `terminated`, so GAE bootstraps.
            StepCounter(max_steps=max_steps),
        ),
    )
    if check_env:
        print("checking env specs")
        check_env_specs(env)
    return env


def run_training(cfg: dict) -> None:
    """Run a full training pass with the given resolved config.

    `cfg` is the single source of truth. Build one via ``train.config.resolve_config``
    and hand it in; no module-level globals are consulted.
    """
    # Seed torch & numpy from base_seed for reproducibility.
    base_seed = cfg.get("base_seed", 0)
    np.random.seed(base_seed)
    th.manual_seed(base_seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(base_seed)
        th.cuda.manual_seed_all(base_seed)

    # Perf knobs (no model change). The lidar-encoder conv dominates training, so:
    #  - cudnn_deterministic=False lets cudnn pick faster (non-deterministic) conv
    #    algos; combined with cudnn.benchmark autotune. Off => bit-reproducible but
    #    slower; quality is identical either way (training is already stochastic).
    #  - tf32 enables TF32 matmul (Ada tensor cores) — ~fp32 accuracy, faster matmul.
    # Both default ON for speed and also accelerate the collection policy forward.
    perf = cfg.get("perf", {}) or {}
    th.backends.cudnn.deterministic = bool(perf.get("cudnn_deterministic", False))
    th.backends.cudnn.benchmark = bool(perf.get("cudnn_benchmark", True))
    if perf.get("tf32", True):
        th.backends.cuda.matmul.allow_tf32 = True
        th.backends.cudnn.allow_tf32 = True
        th.set_float32_matmul_precision("high")

    # The scenario file looks for cfg["use_global_path"] — derive it from the
    # two related flags so callers only need to set the obvious ones.
    cfg["use_global_path"] = cfg.get("use_global_path_obs", False) or cfg.get(
        "set_gp_as_goal", False
    )

    # Baseline model uses a different reward / observation config; merge it in.
    if cfg.get("model", "baseline") == "baseline":
        cfg.update(cfg.get("baseline_config", {}))

    ppo_config = cfg["ppo"]
    load_model = cfg.get("load_model")
    load_only_model = cfg.get("load_only_model", False)
    num_worlds = cfg.get("num_worlds", 150)

    # Logger / W&B handshake.
    logger = (
        make_logger(load_model, load_only_model, cfg)
        if cfg.get("log", True)
        else None
    )
    if logger is None:
        for _ in range(5):
            print(Fore.RED + "!!!!!!!!!!!!!!!! NO LOGGING !!!!!!!!!!!!!!!!" + Style.RESET_ALL)

    # Devices.
    is_fork = multiprocessing.get_start_method() == "fork"
    device = th.device(0) if th.cuda.is_available() and not is_fork else th.device("cpu")
    device = th.device(cfg.get("device", device))
    vmas_device = device

    # Sampling config pulled from ppo block.
    frames_per_batch = ppo_config["frames_per_batch"]
    n_iters = ppo_config["n_iters"]
    total_frames = frames_per_batch * n_iters
    num_epochs = ppo_config["num_epochs"]
    minibatch_size = ppo_config["minibatch_size"]
    lr = ppo_config["lr_start"]
    max_grad_norm = ppo_config["max_grad_norm"]
    clip_epsilon = ppo_config["clip_epsilon"]
    gamma = ppo_config["gamma"]
    lmbda = ppo_config["lmbda"]
    entropy_eps = ppo_config["entropy_eps_start"]
    max_steps = ppo_config["max_steps"]

    # --- adaptive curriculum (optional): start at level 0 by applying its cfg
    # overrides before the first env build. The policy is built once from level 0
    # and transfers across levels (GNN + agent-padding are agent-count-agnostic).
    from train.config import merge_overrides as _merge_overrides
    curriculum = None
    _curr_cfg = cfg.get("curriculum") or {}
    if _curr_cfg.get("enabled"):
        from train.utils.curriculum import CurriculumManager
        curriculum = CurriculumManager(_curr_cfg)
        # Resume: restore the curriculum level/progress from the checkpoint BEFORE the
        # env is built, so the env (num_worlds, scenario, agent count) is created at the
        # RESUMED level rather than start_level. Backward compatible — old checkpoints
        # have no "curriculum" key, so this is a no-op and start_level is used.
        if load_model is not None and not load_only_model:
            try:
                _ck_peek = th.load(load_model, map_location="cpu")
                _cstate = _ck_peek.get("curriculum") if isinstance(_ck_peek, dict) else None
                if _cstate:
                    curriculum.load_state_dict(_cstate)
                    print(f"[curriculum] resumed at level {curriculum.level_idx} "
                          f"({curriculum.name}) from checkpoint "
                          f"(iters_at_level={curriculum._iters_at_level})")
            except Exception as _e:
                print(f"[curriculum] WARNING: could not read curriculum state from "
                      f"checkpoint ({_e}); starting at level {curriculum.level_idx}")
        _merged = _merge_overrides(cfg, curriculum.level_overrides())
        cfg.clear(); cfg.update(_merged)
        # Level 0 may set its own num_worlds -> recompute the sampling sizes so the
        # initial env/collector/buffer are built at level 0's scale.
        num_worlds = int(cfg.get("num_worlds", num_worlds))
        frames_per_batch = int(cfg["ppo"].get("max_steps", max_steps)) * num_worlds
        total_frames = frames_per_batch * n_iters
        print(f"[curriculum] starting at level {curriculum.level_idx} ({curriculum.name}); "
              f"num_worlds={num_worlds}, frames_per_batch={frames_per_batch}")

    # Build envs. make_env mutates the cfg dict (merges multi_config etc.) —
    # use copies for eval so training cfg stays intact.
    print("\nMake eval env:")
    eval_cfg = dict(cfg)  # shallow copy is fine; nested dicts re-used
    eval_env = make_env(eval_cfg, cfg.get("num_eval_envs"), check_env=False, eval=True)
    print("\nMake train env:")
    env = make_env(cfg, num_env=num_worlds)

    if cfg.get("scenario_type", "random") != "multi":
        rollout = eval_env.rollout(
            1,
            callback=lambda env, _: env.render(),
            auto_cast_to_device=True,
            break_when_any_done=False,
        )
        print("rollout of three steps:", rollout)
        print("Shape of the rollout TensorDict:", rollout.batch_size)

    # Build nets.
    current_step, policy, critic = make_network(env, cfg, load_model, load_only_model, device)
    eval_env.set_seed(0)
    print("Running value:", critic(eval_env.reset()))

    storing_device = cfg.get("storing_device", device)
    # collect_chunks = how many memory-chunks the per-iter collection is split into
    # (each chunk is collected + GAE'd separately to bound memory). This is SEPARATE
    # from minibatch_size (the training minibatch). They used to be the same value,
    # which meant raising minibatch_size shrank the collector chunk -> many tiny
    # collector calls -> slow collection. Default to minibatch_size for back-compat.
    collect_chunks = ppo_config.get("collect_chunks", minibatch_size)
    collector = SyncDataCollector(
        env,
        policy,
        device=vmas_device,
        storing_device=storing_device,
        frames_per_batch=frames_per_batch // collect_chunks,
        total_frames=total_frames,
    )

    # Replay buffer. When storing_device is a CUDA device, keep the per-iter batch
    # resident in GPU memory (LazyTensorStorage) so sampling is a pure index op and
    # there is NO host<->device transfer per training step — measured ~25% of a
    # (lidar-frozen) finetune step was sample-from-memmap + h2d. On CPU we keep the
    # disk-backed LazyMemmapStorage (`existsok=True` reuses the scratch dir across
    # restarts; without it torchrl 0.11 crashes on the second run).
    if str(storing_device).startswith("cuda"):
        storage = LazyTensorStorage(frames_per_batch, device=storing_device)
    else:
        storage = LazyMemmapStorage(
            frames_per_batch,
            # Namespace by PID so concurrent runs don't share/corrupt one scratch dir.
            scratch_dir=os.path.join(tempfile.gettempdir(), f"giant_memmap_{os.getpid()}"),
            device=storing_device,
            existsok=True,
        )
    replay_buffer = ReplayBuffer(
        storage=storage,  # holds the frames_per_batch collected each iteration
        sampler=SamplerWithoutReplacement(),
        batch_size=minibatch_size,
    )

    # Curriculum level-switch callback: apply the new level's cfg overrides and
    # rebuild env / eval_env / collector / replay_buffer (fresh data) while the
    # policy, critic, optimizer and schedulers carry over unchanged. A level may
    # set its own `num_worlds` (e.g. many more cheap small worlds at L0/L1), so we
    # recompute frames_per_batch = max_steps * num_worlds and resize the collector +
    # buffer accordingly; the new frames_per_batch is returned so the trainer's
    # per-iter minibatch count tracks it.
    def _build_storage(fpb):
        if str(storing_device).startswith("cuda"):
            return LazyTensorStorage(fpb, device=storing_device)
        return LazyMemmapStorage(
            fpb,
            # Namespace by PID so concurrent runs don't share/corrupt one scratch dir.
            scratch_dir=os.path.join(tempfile.gettempdir(), f"giant_memmap_{os.getpid()}"),
            device=storing_device, existsok=True,
        )

    def on_level_switch(overrides):
        merged = _merge_overrides(cfg, overrides)
        cfg.clear(); cfg.update(merged)
        nw = int(cfg.get("num_worlds", num_worlds))
        ms = int(cfg["ppo"].get("max_steps", max_steps))
        new_fpb = ms * nw
        new_total = new_fpb * n_iters
        new_eval = make_env(dict(cfg), cfg.get("num_eval_envs"), eval=True)
        new_env = make_env(cfg, num_env=nw)
        new_collector = SyncDataCollector(
            new_env, policy, device=vmas_device, storing_device=storing_device,
            frames_per_batch=new_fpb // collect_chunks, total_frames=new_total,
        )
        new_buffer = ReplayBuffer(
            storage=_build_storage(new_fpb), sampler=SamplerWithoutReplacement(),
            batch_size=minibatch_size,
        )
        return new_env, new_eval, new_collector, new_buffer, new_fpb

    # Loss module. Default is "clip" (custom padding-masked ClipPPOLoss). The
    # "kl_pen_ppo" path uses stock torchrl KLPENPPOLoss which does NOT mask
    # padded/terminated agents, corrupting value targets in multi-agent settings
    # (see config.py loss_module note). Guard so a swap-back is loud and deliberate.
    if cfg.get("loss_module", "clip") == "kl_pen_ppo" or ppo_config.get("loss_module") == "kl_pen_ppo":
        print(
            "\n[loss] !!! WARNING: loss_module='kl_pen_ppo' uses stock KLPENPPOLoss "
            "which does NOT mask is_padding. Dead/terminated agents corrupt the critic "
            "value targets and policy objective (degrades L1+ multi-agent learning). "
            "FIX the padding masking before relying on this; prefer loss_module='clip'.\n"
        )
        loss_module = KLPENPPOLoss(
            actor_network=policy,
            critic_network=critic,
            dtarg=ppo_config.get("kl_target", 0.01),
            beta=ppo_config.get("kl_init_div", 1.0),
            increment=ppo_config.get("kl_increment", 2.0),
            decrement=ppo_config.get("kl_decrement", 0.5),
            samples_mc_kl=1,
            samples_mc_entropy=1,
            entropy_coeff=entropy_eps,  # torchrl >=0.11 renamed entropy_coef -> entropy_coeff
            normalize_advantage=False,
        )
    else:
        loss_module = ClipPPOLoss(
            actor_network=policy,
            critic_network=critic,
            clip_epsilon=clip_epsilon,
            entropy_coef=entropy_eps,
            normalize_advantage=False,
        )
    loss_module.set_keys(
        reward=env.reward_key,
        action=env.action_key,
        sample_log_prob=("agents", "sample_log_prob"),
        value=("agents", "state_value"),
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )
    loss_module.make_value_estimator(ValueEstimators.GAE, gamma=gamma, lmbda=lmbda)
    GAE = loss_module.value_estimator

    if cfg.get("finetune"):
        from train.utils.finetune import freeze_by_substring

        freeze_subs = cfg.get(
            "finetune_freeze", ["lidar_static_encoder", "lidar_dynamic_encoder"]
        )
        n_frozen, _ = freeze_by_substring(loss_module, freeze_subs)
        print(f"[finetune] froze {n_frozen} param tensors matching {freeze_subs}")

    # Optimizer over the trainable params. GIANT (Table I) uses an asymmetric
    # learning rate: actor at lr_start (2e-5), critic at critic_lr (4e-4). Adam is
    # ~invariant to a constant gradient scale and actor/critic params are disjoint,
    # so value_loss_factor does NOT set the critic's step size — the separate
    # critic_lr param group does. critic_lr=None (or == lr) -> single lr.
    from train.utils.finetune import split_actor_critic_params

    actor_params, critic_params = split_actor_critic_params(loss_module)
    critic_lr = ppo_config.get("critic_lr")
    # fused Adam fuses the per-parameter step into a single CUDA kernel (same math).
    # On GPU this measurably cuts the optimizer cost, which is a non-trivial share of
    # a step once the backward shrinks (e.g. lidar-frozen finetune). CUDA-only.
    _all_params = actor_params + critic_params
    _fused = (bool(_all_params) and _all_params[0].is_cuda
              and (cfg.get("perf", {}) or {}).get("fused_adam", True))
    if critic_lr is not None and critic_lr != lr:
        optim = th.optim.Adam(
            [{"params": actor_params, "lr": lr},
             {"params": critic_params, "lr": critic_lr}],
            fused=_fused,
        )
        print(f"[optim] asymmetric LR (GIANT): actor={len(actor_params)} @ {lr}, "
              f"critic={len(critic_params)} @ {critic_lr} (fused={_fused})")
    else:
        optim = th.optim.Adam(_all_params, lr, fused=_fused)
        print(f"[optim] single LR {lr}: {len(_all_params)} trainable tensors (fused={_fused})")

    current_step = -1
    if load_model is not None:
        if not load_only_model:
            checkpoint = th.load(load_model)
            # save_checkpoint defaults to the verbose schema (optimizer_state_dict);
            # accept the legacy short key ("optimizer") too.
            optim_state = checkpoint.get("optimizer_state_dict")
            if optim_state is None:
                optim_state = checkpoint.get("optimizer")
            if optim_state is not None:
                optim.load_state_dict(optim_state)
            current_step = checkpoint.get("step", current_step)
        print(f"Loading checkpoint {load_model} completed, (only model = {load_only_model})")

    # Schedulers.
    schedulers = {}
    if entropy_eps != ppo_config.get("entropy_eps_end") and entropy_eps != 0:
        schedulers["entropy coef"] = EntropyDecay(
            device, entropy_eps, ppo_config.get("entropy_eps_end", 1e-5), 100, 0, loss_module,
        )
    if lr != ppo_config.get("lr_end"):
        schedulers["lr"] = LRDecay(optim, lr, ppo_config.get("lr_end"), 100, 0)

    std_max_start = ppo_config.get("std_max_start", -1)
    if (
        std_max_start != -1
        and std_max_start != ppo_config.get("std_max_end", -1)
        and cfg.get("model", "baseline") != "baseline"
    ):
        schedulers["std_max"] = STDDecay(
            policy, std_max_start, ppo_config.get("std_max_end", -1),
            ppo_config.get("std_max_decay_steps", 100), 0,
        )

    if logger is not None and (load_only_model or load_model is None):
        if hasattr(logger, "log_hparams"):
            logger.log_hparams(cfg)

    PPOTrainer(
        GAE,
        collector,
        None,
        "stage_one",
        critic,
        env,
        eval_env,
        frames_per_batch,
        loss_module,
        schedulers,
        max_grad_norm,
        max_steps,
        minibatch_size,
        n_iters,
        num_epochs,
        optim,
        policy,
        replay_buffer,
        ppo_config,
        current_step,
        logger,
        cfg.get("save_video_to_disk", False),
        cfg.get("eval_every", 10),
        cfg.get("eval_render_to_screen", False),
        storing_device,
        device,
        cfg.get("eval_seed", 0),
        render_env_index=cfg.get("render_env_index"),
        video_every=cfg.get("video_every", cfg.get("eval_every", 10)),
        collect_chunks=collect_chunks,
        amp_dtype=(cfg.get("perf", {}) or {}).get("amp_dtype"),
        curriculum=curriculum,
        on_level_switch=on_level_switch,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML / JSON config overrides. Applied on top of defaults.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint .pth to resume from (overrides cfg['load_model']).",
    )
    parser.add_argument(
        "--finetune",
        type=str,
        default=None,
        metavar="CKPT",
        help="Path to a checkpoint .pth to fine-tune: loads its weights (fresh "
        "optimizer) and freezes cfg['finetune_freeze'] (default: the lidar "
        "encoders). Pair with gnn_attention: emb to retrain the corrected attention.",
    )
    parser.add_argument(
        "--logging-backend",
        type=str,
        choices=("wandb", "file", "none"),
        default=None,
        help="Override cfg['logging_backend']. Useful for running without wandb.",
    )
    parser.add_argument(
        "--n-iters",
        type=int,
        default=None,
        help="Override ppo.n_iters (useful for smoke runs).",
    )
    parser.add_argument(
        "--num-worlds",
        type=int,
        default=None,
        help="Override cfg['num_worlds'] (also rescales ppo.frames_per_batch).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the effective config, then exit.",
    )
    return parser.parse_args()


def _apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    from train.config import merge_overrides

    overrides = {}
    if args.resume is not None:
        overrides["load_model"] = args.resume
    if args.finetune is not None:
        # Fine-tune: load weights only (fresh optimizer) + enable the freeze.
        overrides["load_model"] = args.finetune
        overrides["load_only_model"] = True
        overrides["finetune"] = True
    if args.logging_backend is not None:
        overrides["logging_backend"] = args.logging_backend
        if args.logging_backend == "none":
            overrides["log"] = False
    if args.n_iters is not None:
        overrides["ppo"] = {"n_iters": args.n_iters}
    if args.num_worlds is not None:
        # Rescale frames_per_batch to stay consistent.
        max_steps = cfg["ppo"]["max_steps"]
        overrides["num_worlds"] = args.num_worlds
        overrides["ppo"] = overrides.get("ppo", {})
        overrides["ppo"]["frames_per_batch"] = args.num_worlds * max_steps
    return merge_overrides(cfg, overrides) if overrides else cfg


if __name__ == "__main__":
    args = _parse_args()

    # Pretty tensor repr (kept from the original entry point).
    normal_repr = th.Tensor.__repr__
    th.Tensor.__repr__ = lambda self: f"{self.shape}_{normal_repr(self)}"

    cfg = resolve_config(args.config)
    cfg = _apply_cli_overrides(cfg, args)

    if args.dry_run:
        import json

        def _default(o):
            if isinstance(o, th.device):
                return str(o)
            return str(o)

        print(json.dumps(cfg, indent=2, default=_default, sort_keys=True))
        sys.exit(0)

    run_training(cfg)
