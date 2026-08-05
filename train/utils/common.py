import copy
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

# Env
import cv2
import numpy as np
import torch as th
from tensordict import LazyStackedTensorDict
from torchrl._utils import (
    _ends_with,
    _replace_last,
)
from torchrl.collectors import SyncDataCollector
from torchrl.envs.common import _get_sync_func
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.envs.utils import (
    _make_compatible_policy,
    _terminated_or_truncated,
)
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.advantages import GAE
from torchrl.record.loggers import Logger
from tqdm import tqdm
from vmas.simulator.environment import Environment

from scenario.CollisionAvoidance_base import int_to_scenario_name
from train.utils.logger import MyWandbLogger

frames_to_video = {}


def render_for_debugging(env, policy, n_steps=3):
    _ = env.rollout(
        n_steps,
        policy=policy,
        callback=lambda env, _: env.render(),
        auto_cast_to_device=True,
        break_when_any_done=False,
    )


def add_to_log(logs, key, value):
    # TensorDict / dict values: flatten nested entries with slash-joined keys.
    # torchrl 0.11's loss_module returns nested TensorDicts for some objectives
    # (e.g. loss_objective carries mean/std children). An unflattened dict slips
    # past the .item() path and crashes the downstream `th.tensor(list)`
    # aggregation with "Could not infer dtype of dict". Flattening on insert
    # keeps every logged key a flat list of Python scalars.
    if hasattr(value, "items"):
        for subkey, subval in value.items():
            add_to_log(logs, f"{key}/{subkey}", subval)
        return logs
    if key not in logs:
        logs[key] = []
    if isinstance(value, th.Tensor):
        # Some torchrl 0.11 objectives (e.g. KLPENPPOLoss) emit per-sample
        # diagnostics in their output dict; these are non-scalar and would crash
        # .item(). They are logging-only (the backprop loss is already reduced),
        # so summarise with the mean.
        value = value.mean().item() if value.numel() != 1 else value.item()
    logs[key].append(value)
    return logs


def _flatten_scenario_logs(logs):
    """Flatten one level of scenario-name nesting from `extract_logging`.

    `extract_logging` writes `logs[<scenario>][<metric>] = [values]`; downstream
    consumers (logger loop + progress-bar metrics) want flat keys. Suffix the
    scenario onto the metric so "train reward" from scenario "random" becomes
    "train reward (random)", which also reads well in wandb plots.
    """
    flat = {}
    for key, value in logs.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat[f"{sub_key} ({key})"] = sub_value
        else:
            flat[key] = value
    return flat


def _pool_metric_across_scenarios(flat_logs, metric):
    """Gather every value of `metric` across all scenario-suffixed variants.

    Returns a flat list, empty if the metric is absent. `th.tensor([]).mean()`
    gives NaN, which is the right behavior for "nothing logged yet".
    """
    pooled = []
    for key, value in flat_logs.items():
        # BUGFIX: startswith(f"{metric} (") also matches the std variant, e.g. metric
        # "eval reward" was pooling "eval reward (std)" INTO the mean, contaminating the
        # reported reward and the curriculum demote gate. Exclude derived-stat variants.
        if "(std)" in key:
            continue
        if key == metric or key.startswith(f"{metric} ("):
            pooled.extend(value)
    return pooled


def _env_max_steps(env):
    """Resolve the episode horizon robustly.

    Truncation was moved off VmasEnv (max_steps=None, so GAE bootstraps on
    timeout instead of treating it as termination) onto the StepCounter
    transform. That leaves ``env.max_steps`` as None, so fall back to reading
    ``max_steps`` from the StepCounter inside the (Compose) transform stack.
    """
    ms = getattr(env, "max_steps", None)
    if ms is not None:
        return ms
    tr = getattr(env, "transform", None)
    subs = getattr(tr, "transforms", [tr]) if tr is not None else []
    for t in subs:
        m = getattr(t, "max_steps", None)
        if m is not None:
            return m
    return ms


def extract_scenario_logging(
    tensordict_data_current, tensordict_data_next, env, prefix="eval", logs: Dict = None
):
    done = tensordict_data_next.get("done").bool()
    padding = tensordict_data_next.get(("info", "is_padding")).bool()
    not_padding = ~padding
    not_done = ~done
    true_done = done * not_padding
    true_not_done = ~true_done
    # Reward
    add_to_log(
        logs,
        f"{prefix} reward",
        tensordict_data_next.get("episode_reward")[true_done].mean().item(),
    )

    add_to_log(
        logs,
        f"{prefix} reward (std)",
        tensordict_data_next.get("episode_reward")[true_done].std().item(),
    )

    # reward distribution (
    info = tensordict_data_next.get("info")
    for name in list(iter(info.keys())):
        if "_rew" in name:
            plot_name = f"{prefix} " + name.replace("rew", "reward").replace("_", " ")
            add_to_log(logs, plot_name, info[name].sum(1).mean().item())

    # num of collisions
    num_world = info["agent_collision_rew"].shape[0]
    num_agents = info["agent_collision_rew"].shape[2]
    collisions = info["has_collided"][true_done].sum() / num_world
    add_to_log(
        logs,
        f"{prefix} collisions",
        collisions.item(),
    )  # collisions per world
    add_to_log(
        logs,
        f"{prefix} num agents",
        num_agents,
    )
    collisions_rate = (collisions / num_agents).item()
    add_to_log(
        logs,
        f"{prefix} collisions rate",
        collisions_rate,
    )  # collisions per world

    finished = info["on_goal"][true_done].sum() / num_world
    success_rate = (finished / num_agents).item()
    add_to_log(
        logs,
        f"{prefix} success rate",
        success_rate,
    )  # collisions per world

    add_to_log(logs, f"{prefix} stuck rate", 1 - (collisions_rate + success_rate))

    # compute extra time.
    # In the paper they describe the extra time as following:
    # the difference between the average travel time of all robots and the lower bound of the robots’
    # travel time. The latter is computed as the average travel time when going straight towards the goal
    # at the maximum speed without checking for any collisions.
    # However as we have the global path, we take the average time to reach the goal following the path (without other agents)
    # vs the avg time running our policy.
    single_travel_time = info["path_dist"][
        true_done
    ]  # in meters with a vel of 1 m/s gives the same as time
    steps = th.argwhere(true_done)[:, 1] + 1  # each agent steps
    collisions = info["has_collided"][true_done].bool()
    steps[collisions] = _env_max_steps(env)
    policy_travel_time = steps * env.scenario.dt  # 0.1 is the time step
    add_to_log(
        logs,
        f"{prefix} extra time",
        (policy_travel_time.mean().item() - single_travel_time.mean().item()),
    )
    add_to_log(
        logs,
        f"{prefix} extra time (std)",
        (policy_travel_time - single_travel_time).std().item(),
    )

    # scale:
    if tensordict_data_current.get("scale", None) is not None:
        add_to_log(
            logs,
            f"{prefix} policy std",
            tensordict_data_current.get("scale").mean().item(),
        )
    vel = tensordict_data_current.get(("info", "vel"))
    # If we collide, we will count it as 0 velocity for the rest of the episode
    not_on_goal = ~info["on_goal"].bool()
    not_on_goal = th.cumprod(not_on_goal, dim=1).bool()
    add_to_log(logs, f"{prefix} vel mean", vel[not_on_goal].mean().item())
    add_to_log(logs, f"{prefix} vel (std)", vel[not_on_goal].std().item())
    return logs


def extract_logging(tensordict_data, env, prefix="eval", logs: Dict = None):
    if logs is None:
        logs = {}
    scenarios = tensordict_data.get((
        "next",
        "agents",
        "info",
        "scenario_name",
    )).unique()
    td_shape = tensordict_data.shape
    scenario_names_seen = []
    for scenario in scenarios:
        scenario_name = int_to_scenario_name[scenario.item()]
        scenario_names_seen.append(scenario_name)
        scenario_mask = (
            tensordict_data.get(("next", "agents", "info", "scenario_name")) == scenario
        )
        scenario_mask = scenario_mask.squeeze(-1)
        current = tensordict_data.get("agents")[scenario_mask].view(
            td_shape[0], td_shape[1], -1
        )
        next = tensordict_data.get(("next", "agents"))[scenario_mask].view(
            td_shape[0], td_shape[1], -1
        )
        if scenario_name not in logs:
            logs[scenario_name] = {}
        logs[scenario_name] = extract_scenario_logging(
            current, next, env, prefix, logs[scenario_name]
        )

    # Also write flat top-level mirrors of each per-scenario metric, extending
    # lists across scenarios. PPOTrainer's pbar description (common.py:907-908)
    # reads `loss_dict["train reward"]` directly — without this mirror, the
    # lookup raises KeyError because extract_scenario_logging only ever writes
    # the nested `logs[scenario_name][key]` form.
    _mirror_scenarios_to_flat(logs, scenario_names_seen)
    return logs


def _mirror_scenarios_to_flat(logs: Dict, scenario_names: list) -> None:
    inner_keys = set()
    for name in scenario_names:
        scoped = logs.get(name)
        if isinstance(scoped, dict):
            inner_keys.update(scoped.keys())
    for key in inner_keys:
        flat_values = logs.setdefault(key, []) if isinstance(logs.get(key), list) else []
        if not isinstance(logs.get(key), list):
            logs[key] = flat_values
        for name in scenario_names:
            v = logs.get(name, {}).get(key)
            if v is None:
                continue
            if isinstance(v, list):
                flat_values.extend(v)
            else:
                flat_values.append(v)
    # done = tensordict_data.get(("next", "agents", "done")).bool()
    # padding = tensordict_data.get(("next", "agents", "info", "is_padding")).bool()
    # not_padding = ~padding
    # not_done = ~done
    # true_done = done * not_padding
    # true_not_done = ~true_done
    # # Reward
    # add_to_log(
    #     logs,
    #     f"{prefix} reward",
    #     tensordict_data.get(("next", "agents", "episode_reward"))[true_done]
    #     .mean()
    #     .item(),
    # )
    #
    # add_to_log(
    #     logs,
    #     f"{prefix} reward (std)",
    #     tensordict_data.get(("next", "agents", "episode_reward"))[true_done]
    #     .std()
    #     .item(),
    # )
    #
    # # reward distribution (
    # info = tensordict_data.get(("next", "agents", "info"))
    # for name in list(iter(info.keys())):
    #     if "_rew" in name:
    #         plot_name = f"{prefix} " + name.replace("rew", "reward").replace("_", " ")
    #         add_to_log(logs, plot_name, info[name].sum(1).mean().item())
    #
    # # num of collisions
    # num_world = info["agent_collision_rew"].shape[0]
    # num_agents = info["agent_collision_rew"].shape[2]
    # collisions = info["has_collided"][true_done].sum() / num_world
    # add_to_log(
    #     logs,
    #     f"{prefix} collisions",
    #     collisions.item(),
    # )  # collisions per world
    #
    # collisions_rate = (collisions / num_agents).item()
    # add_to_log(
    #     logs,
    #     f"{prefix} collisions rate",
    #     collisions_rate,
    # )  # collisions per world
    #
    # finished = info["on_goal"][true_done].sum() / num_world
    # success_rate = (finished / num_agents).item()
    # add_to_log(
    #     logs,
    #     f"{prefix} success rate",
    #     success_rate,
    # )  # collisions per world
    #
    # add_to_log(
    #     logs,
    #     f"{prefix} stuck rate",
    #     1 - (collisions_rate + success_rate)
    # )
    # # Makespan
    # if tensordict_data.get("step_count", None) is not None:
    #     env_steps = tensordict_data["step_count"]  # env steps
    #     add_to_log(logs, f"{prefix} step_count", env_steps.max().item())
    #
    # # compute extra time.
    # # In the paper they describe the extra time as following:
    # # the difference between the average travel time of all robots and the lower bound of the robots’
    # # travel time. The latter is computed as the average travel time when going straight towards the goal
    # # at the maximum speed without checking for any collisions.
    # # However as we have the global path, we take the average time to reach the goal following the path (without other agents)
    # # vs the avg time running our policy.
    # single_travel_time = info["path_dist"][
    #     true_done
    # ]  # in meters with a vel of 1 m/s gives the same as time
    # steps = th.argwhere(true_done)[:, 1] + 1  # each agent steps
    # collisions = info["has_collided"][true_done].bool()
    # steps[collisions] = env.max_steps
    # policy_travel_time = steps * env.scenario.dt  # 0.1 is the time step
    # add_to_log(
    #     logs,
    #     f"{prefix} extra time",
    #     (policy_travel_time.mean().item() - single_travel_time.mean().item()),
    # )
    # add_to_log(
    #     logs,
    #     f"{prefix} extra time (std)",
    #     (policy_travel_time - single_travel_time).std().item(),
    # )
    #
    # # scale:
    # if tensordict_data.get(("agents", "scale"), None) is not None:
    #     add_to_log(
    #         logs,
    #         f"{prefix} policy std",
    #         tensordict_data.get(("agents", "scale")).mean().item(),
    #     )
    # vel = tensordict_data.get(("agents", "info", "vel"))
    # # If we collide, we will count it as 0 velocity for the rest of the episode
    # not_on_goal = ~info["on_goal"].bool()
    # not_on_goal = th.cumprod(not_on_goal, dim=1).bool()
    # add_to_log(logs, f"{prefix} vel mean", vel[not_on_goal].mean().item())
    # add_to_log(logs, f"{prefix} vel (std)", vel[not_on_goal].std().item())
    # return logs


def post_process_rollout(tensordict_data):
    n_agents = tensordict_data.get(("next", "agents", "info", "terminated")).shape[2]
    agent_dones = (
        tensordict_data.get(("next", "done")).unsqueeze(-1).repeat(1, 1, n_agents, 1)
    )
    agent_terminated = (
        tensordict_data.get(("next", "agents", "info", "terminated")).int().bool()
    )
    agent_dones = agent_dones + agent_terminated  # + is for or and * is for and

    tensordict_data.set(("next", "agents", "done"), agent_dones)
    tensordict_data.set(("next", "agents", "terminated"), agent_terminated)
    return tensordict_data


def eval_rollout_no_reset(
    eval_env: VmasEnv,
    auto_cast_to_device,
    max_steps,
    policy,
    return_contiguous: bool = True,
    set_truncated: bool = False,
    callback=None,
):
    tensordict = eval_env.reset()
    env_device = eval_env.device

    if policy is not None:
        policy = _make_compatible_policy(
            policy, eval_env.observation_spec, env=eval_env, fast_wrap=True
        )
        if auto_cast_to_device:
            try:
                policy_device = next(policy.parameters()).device
            except (StopIteration, AttributeError):
                policy_device = None
        else:
            policy_device = None
    else:
        policy = eval_env.rand_action
        policy_device = None

    if auto_cast_to_device:
        sync_func = _get_sync_func(policy_device, env_device)
    tensordicts = []
    # Assuming max_steps is defined
    for i in tqdm(
        range(max_steps), desc="Processing", unit="step", position=0, leave=True
    ):
        if auto_cast_to_device:
            if policy_device is not None:
                tensordict = tensordict.to(policy_device, non_blocking=True)
                sync_func()
            else:
                tensordict.clear_device_()
        tensordict = policy(tensordict)
        # Guard: a degenerate / under-trained policy can emit NaN or inf actions,
        # which trip vmas' ``assert not action.isnan().any()`` and crash eval.
        # Replace any non-finite action with 0 (a neutral action) so a long run
        # is never killed by a transient bad policy output during evaluation.
        _akey = eval_env.action_key
        _act = tensordict.get(_akey, None)
        if _act is not None and not th.isfinite(_act).all():
            tensordict.set(
                _akey, th.nan_to_num(_act, nan=0.0, posinf=0.0, neginf=0.0)
            )
        if auto_cast_to_device:
            if env_device is not None:
                tensordict = tensordict.to(env_device, non_blocking=True)
                sync_func()
            else:
                tensordict.clear_device_()

        tensordict = eval_env.step(tensordict)
        td_append = tensordict.copy()
        tensordicts.append(td_append)

        # We moved this callback before the termination check
        if callback is not None:
            callback(eval_env, tensordict)

        if i == max_steps - 1:
            # we don't truncate as one could potentially continue the run
            break

        if tensordict.get(("next", "done")).all():
            break

        tensordict = eval_env._step_mdp(tensordict)

        # done and truncated are in done_keys
        # We read if any key is done.
        _ = _terminated_or_truncated(
            tensordict,
            full_done_spec=eval_env.output_spec["full_done_spec"],
            key=None,
        )

    batch_size = eval_env.batch_size if tensordict is None else tensordict.batch_size
    if return_contiguous:
        out_td = th.stack(tensordicts, len(batch_size))
    else:
        out_td = LazyStackedTensorDict.lazy_stack(tensordicts, len(batch_size))
    if set_truncated:
        found_truncated = False
        for key in eval_env.done_keys:
            if _ends_with(key, "truncated"):
                val = out_td.get(("next", key))
                val[(slice(None),) * (out_td.ndim - 1) + (-1,)] = True
                out_td.set(("next", key), val)
                out_td.set(("next", _replace_last(key, "done")), val)
                found_truncated = True
        if not found_truncated:
            raise RuntimeError(
                "set_truncated was set to True but no truncated key could be found. "
                "Make sure a 'truncated' entry was set in the environment "
                "full_done_keys using `env.add_truncated_keys()`."
            )

    out_td.refine_names(..., "time")
    return out_td


def evaluate(
    eval_env: VmasEnv,
    policy,
    n_steps,
    current_step,
    logger: Optional[Logger],
    save_video=False,
    save_folder=None,
    save_name=None,
    eval_seed=0,
    eval_render_to_screen: bool = False,
    render_env_index=None,
):
    global frames_to_video

    logs = {}
    # We evaluate the policy once every 10 batches of data.
    # Evaluation is rather simple: execute the policy without exploration
    # (take the expected value of the action distribution) for a given
    # number of steps (1000, which is our ``env`` horizon).
    # The ``rollout`` method of the ``env`` can take a policy as argument:
    # it will then execute this policy at each step.
    with set_exploration_type(ExplorationType.MEAN), th.no_grad():
        _eval_ms = _env_max_steps(eval_env)
        if n_steps < _eval_ms:
            print(
                f"Warning: n_steps ({n_steps}) < eval_env.max_steps ({_eval_ms})."
                f"Setting n_steps to eval_env.max_steps."
            )
            n_steps = _eval_ms

        eval_env.set_seed(eval_seed)
        # eval_env.reset() # the rollout handles this
        # execute a rollout with the trained policy
        # eval_rollout = eval_env.rollout(
        #     eval_env.max_steps,
        #     policy=policy,
        #     callback=lambda env, _: save_frames(env, eval_render_to_screen)
        #     if save_video
        #     else None,
        #     auto_cast_to_device=True,
        #     break_when_any_done=True,
        #     auto_reset=True,  # reset the env at the beginning of the rollout
        # )
        print(f"\tCollecting evaluation data ({n_steps} steps)... ", end="")
        # Rendering is the eval bottleneck: save_frames() rasterizes the scene AND
        # writes a PNG to disk EVERY step (CPU-bound, GPU idle ~20 min). The eval
        # METRICS come entirely from the rollout tensordict below, not the frames —
        # frames are only needed to assemble the video. So only render when we are
        # actually saving a video; otherwise skip it and the eval is GPU-bound/fast.
        render_callback = None
        if save_video:
            render_callback = lambda env, _: save_frames(
                env, eval_render_to_screen, render_env_index
            )
        eval_rollout = eval_rollout_no_reset(
            eval_env,
            auto_cast_to_device=True,
            max_steps=n_steps,
            policy=policy,
            callback=render_callback,
        )
        if save_video:
            for _ in range(10):
                save_frames(
                    eval_env, eval_render_to_screen, render_env_index, force_render=True
                )
        print("done")

        eval_rollout = post_process_rollout(eval_rollout)
        logs = extract_logging(eval_rollout, eval_env, prefix="eval")

        eval_rew = 0
        eval_collision_rate = 0
        for scenario_name, scenario_logs in logs.items():
            # extract_logging also writes flat top-level mirror keys (e.g.
            # logs["eval reward"] = [...]) for the progress bar; those values are
            # lists, not per-scenario dicts. Only aggregate the real scenarios.
            if not isinstance(scenario_logs, dict):
                continue
            print("=====================================================")
            eval_rew += th.tensor(scenario_logs["eval reward"]).mean().item()
            eval_collision_rate += (
                th.tensor(scenario_logs["eval collisions rate"]).mean().item()
            )
            print(f"Eval Results {scenario_name}:")
            print(
                f"\tReward: {th.tensor(scenario_logs['eval reward']).mean().item():4.4f}"
            )
            print(
                f"\tCollision rate: {th.tensor(scenario_logs['eval collisions rate']).mean().item():4.4f}"
            )
            print(
                f"\tCollisions: {th.tensor(scenario_logs['eval collisions']).item() * eval_env.scenario.batch_dim}"
            )
            print(
                f"\tnum total agents: {(scenario_logs['eval num agents'][0] * eval_env.scenario.batch_dim)}"
            )
            print(
                f"\tSuccess rate: {th.tensor(scenario_logs['eval success rate']).mean().item():4.4f}"
            )
            print(f"\tStuck rate: {th.tensor(scenario_logs['eval stuck rate']).item()}")
            print(
                f"\tVel mean: {th.tensor(scenario_logs['eval vel mean']).mean().item():4.4f}"
            )
            if "eval policy std" in scenario_logs:
                print(
                    f"\tPolicy std: {th.tensor(scenario_logs['eval policy std']).mean().item():4.4f}"
                )
            print(
                f"\tExtra time: {th.tensor(scenario_logs['eval extra time']).mean().item():4.4f}"
            )
            if "eval step_count" in scenario_logs:
                print(
                    f"\tStep count: {th.tensor(scenario_logs['eval step_count']).mean().item():4.4f}"
                )
        print("=====================================================")
        eval_str = (
            f"eval - mean reward: {eval_rew: 4.4f}, "
            f"eval collision rate: {eval_collision_rate:4.4f}"
        )
        print("Before delete eval rollout")
        del eval_rollout
        print("After delete eval rollout")
    print("out")

    if save_video:
        print(f"\tSaving video to {Path(save_folder, save_name)}... ", end="")
        save_as_gif(
            eval_env, policy, n_steps, save_folder, save_name, do_rollout=False, dt=0.1
        )
        if logger is not None:
            fps = 1 / eval_env.scenario.dt
            # Log every rendered env (render_env_index can request several), so
            # wandb shows one video per scenario instead of only env 0.
            for env_idx, video in sorted(frames_to_video.items()):
                if not video:
                    continue
                frames = (
                    th.tensor(np.stack(video, axis=0)).unsqueeze(0).permute(0, 1, 4, 2, 3)
                )
                logger.log_video(
                    name=f"eval_video_{env_idx}",
                    video=frames,
                    fps=fps,
                    format="mp4",
                    step=current_step,
                )
        print("done")
    if logger is not None:
        # Same scenario-nesting `extract_logging` produces for training logs
        # applies to eval logs — flatten before iterating so `th.tensor(value)`
        # doesn't see a dict and crash with "Could not infer dtype of dict".
        for name, value in _flatten_scenario_logs(logs).items():
            mean_val = th.tensor(value).float().mean().item()
            logger.log_scalar(name, mean_val, step=current_step)

    frames_to_video = {}  # Reset the frames_to_video list no matter what

    # Aggregate eval metrics (mean across scenarios) for the curriculum manager.
    _flat = _flatten_scenario_logs(logs)
    def _mean_metric(metric):
        vals = _pool_metric_across_scenarios(_flat, metric)
        return float(sum(vals) / len(vals)) if vals else float("nan")
    eval_metrics = {
        "reward": _mean_metric("eval reward"),
        "collision_rate": _mean_metric("eval collisions rate"),
        "success_rate": _mean_metric("eval success rate"),
        "stuck_rate": _mean_metric("eval stuck rate"),
    }
    return logs, eval_str, eval_metrics


def save_as_gif(
    env,
    policy,
    n_steps,
    save_folder,
    save_name,
    do_rollout=True,
    dt=0.1,
    eval_render_to_screen: bool = False,
):
    global frames_to_video
    if do_rollout:
        _ms = _env_max_steps(env)
        if n_steps < _ms:
            n_steps = _ms
        with set_exploration_type(ExplorationType.MEAN), th.no_grad():
            data = env.rollout(
                n_steps,
                policy=policy,
                callback=lambda env, _: save_frames(env, eval_render_to_screen),
                auto_cast_to_device=True,
                break_when_any_done=False,
            )
            del data
    for env_index, frames in frames_to_video.items():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        FPS = 1 / dt  # 1 / dt
        ei_save_name = save_name.replace(".mp4", f"_{env_index}.mp4")
        save_loc = Path(save_folder, ei_save_name)
        if not save_loc.parent.exists():
            os.makedirs(save_loc.parent)

        h, w = frames[0].shape[:2]
        video = cv2.VideoWriter(
            filename=f"{str(save_loc)}",
            fourcc=fourcc,
            fps=FPS,
            frameSize=(w, h),
            isColor=True,
        )
        for i, frame in enumerate(frames):
            # Performace monitor – in the same line we will print % of completeness and current file
            # print(f"Done: {round(i * 100 / len(frames_to_video), 1)}%", end="\r")
            # for writing, we need to convert this image to numpy array and specify colour scheme
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        # after all images are stored in video file we need to write proper file end marks
        video.release()


def save_frames(
    env: Environment,
    eval_render_to_screen: bool = False,
    indexs_to_render: Optional[list[int]] = None,
    force_render=False,
):
    global frames_to_video
    if indexs_to_render is None:
        indexs_to_render = [0]

    dones = env.scenario.done()
    # Only render envs that actually exist: eval_render_envs defaults to [0..4],
    # but num_eval_envs may be smaller, which would index past the batch.
    num_envs = len(dones)
    indexs_to_render = [i for i in indexs_to_render if i < num_envs]

    if len(frames_to_video) == 0:
        for i in indexs_to_render:
            frames_to_video[i] = []
    for i in indexs_to_render:
        if dones[i] and not force_render:
            continue
        frame = env.render(
            mode="rgb_array", visualize_when_rgb=eval_render_to_screen, env_index=i
        )
        frames_to_video[i].append(frame)
        # NOTE: no per-step PNG dump here — the mp4/gif is assembled from the
        # in-memory `frames_to_video` (see save_as_gif), so writing a PNG every
        # step was redundant disk I/O on the eval hot path.


def save_checkpoint(
    policy, critic, optimizer, step, checkpoint_dir="results/checkpoints",
    legacy_format=False, curriculum_state=None,
):
    """Write a training checkpoint.

    By default uses the verbose key schema (``policy_state_dict`` etc.), which is what
    the loader in ``models/model_loader.py`` now expects. Pass ``legacy_format=True`` to
    emit the old short-key schema (``policy``, ``critic``, ``optimizer``) for tooling
    that still reads that format. The loader accepts both either way.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pth")
    if legacy_format:
        payload = {
            "policy": policy.state_dict(),
            "critic": critic.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        }
    else:
        payload = {
            "policy_state_dict": policy.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
        }
    if curriculum_state is not None:
        payload["curriculum"] = curriculum_state
    th.save(payload, checkpoint_path)


def compute_advantage_and_target(
    tensordict_data,
    loss_module,
    gae: GAE,
    gae_batch_size,
    storage_device,
    model_device,
    verbose=False,
):
    if verbose:
        print("\tComputing GAE: ", end="")
    with th.no_grad():
        mbatch = gae_batch_size
        num_minibatches = tensordict_data.shape[0] // mbatch
        for i in range(num_minibatches):
            start = i * mbatch
            end = (i + 1) * mbatch
            minibatch = tensordict_data[start:end]
            # Manually compute value and target_value
            critic = loss_module.value_estimator.value_network
            with loss_module.critic_network_params.to_module(critic):
                obs = minibatch.select(
                    *critic.in_keys, loss_module.tensor_keys.value, strict=False
                )
                value = critic(obs)
                minibatch.set(
                    loss_module.tensor_keys.value,
                    value.get(loss_module.tensor_keys.value),
                )

            with loss_module.target_critic_network_params.to_module(critic):
                obs = minibatch.get("next").select(
                    *critic.in_keys, loss_module.tensor_keys.value, strict=False
                )
                value = critic(obs)
                minibatch.set(
                    ("next", loss_module.tensor_keys.value),
                    value.get(loss_module.tensor_keys.value),
                )
            minibatch = minibatch.to(model_device)
            d = gae(minibatch)
            d = d.to(storage_device)
            tensordict_data[start:end] = d
            if verbose:
                print(".", end="")

        # Process the remaining data if it doesn't fit perfectly into the minibatches
        if tensordict_data.shape[0] % mbatch != 0:
            start = num_minibatches * mbatch
            end = tensordict_data.shape[0]
            minibatch = tensordict_data[start:end]
            # Manually compute value and target_value
            critic = loss_module.value_estimator.value_network
            with loss_module.critic_network_params.to_module(critic):
                obs = minibatch.select(
                    *critic.in_keys, loss_module.tensor_keys.value, strict=False
                )
                value = critic(obs)
                minibatch.set(
                    loss_module.tensor_keys.value,
                    value.get(loss_module.tensor_keys.value),
                )

            with loss_module.target_critic_network_params.to_module(critic):
                obs = minibatch.get("next").select(
                    *critic.in_keys, loss_module.tensor_keys.value, strict=False
                )
                value = critic(obs)
                minibatch.set(
                    ("next", loss_module.tensor_keys.value),
                    value.get(loss_module.tensor_keys.value),
                )

            # Compute GAE and add it to the minibatch
            d = gae(
                minibatch,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )
            d = d.to(storage_device)  # .to() returns a copy; must reassign
            tensordict_data[start:end] = d
    if verbose:
        print("done", end="\n")
    return tensordict_data


def get_data(
    collector: SyncDataCollector,
    log=None,  # log is a dict
):
    from train.utils.profiling import profile_phase

    with profile_phase("collector.next (env step + policy + resets)"):
        tensordict_data = collector.next()
    # The collector returns None once it has produced its total_frames budget.
    # Surface that as "no more data" so the training loop can stop cleanly instead
    # of crashing in post_process_rollout(None).
    if tensordict_data is None:
        return None, log
    with profile_phase("post_process_rollout"):
        tensordict_data = post_process_rollout(tensordict_data)
    with profile_phase("extract_logging (train)"):
        log = extract_logging(tensordict_data, collector.env, prefix="train", logs=log)
    return tensordict_data, log


def PPOTrainer(
    gae,
    collector: SyncDataCollector,
    collectors: list[SyncDataCollector],
    stage: str,
    critic,
    env,
    eval_env,
    frames_per_batch,  # how many frames we collect before training
    loss_module: ClipPPOLoss,
    schedulers: Dict,
    max_grad_norm,
    max_steps,
    minibatch_size,
    n_iters,
    num_epochs,
    optim,
    policy,
    replay_buffer,
    ppo_config,
    current_step=-1,
    logger=None,
    save_video_to_disk=False,
    eval_every=10,
    eval_render_to_screen=False,
    storage_device="cpu",
    model_device="cpu",
    eval_seed=0,
    base_seed=0,
    render_env_index=None,
    video_every=50,
    collect_chunks=None,
    amp_dtype=None,
    curriculum=None,
    on_level_switch=None,
):
    # train
    value_loss_factor = ppo_config["value_loss_factor"]
    policy_loss_factor = ppo_config["policy_loss_factor"]
    gae_batch_size = ppo_config["gae_batch_size"]
    pbar = tqdm(total=n_iters, initial=current_step + 1, desc="episode_reward_mean = 0")
    # vmap does not allow for dynamic batching (graphs of different sizes) and gae relies on vmap, so this is a workaround.
    # todo:: change this when they update torch to allow for dynamic batching.
    gae_dummy = copy.deepcopy(gae)
    gae_dummy.value_network = None  #
    # Per-level exploration reset: the std_max (and entropy) schedule anneals RELATIVE
    # to the start of the current curriculum level, so exploration re-warms each level
    # (each level is a fresh, harder problem). LR (training-progress) stays on the
    # global step. `level_start_step` marks where the current level began; it resets on
    # every level switch below. On resume it starts at the loaded step (re-warm there).
    _PER_LEVEL_SCHED = {"std_max", "entropy coef"}
    # On resume the current level may be partway done — reconstruct the per-level origin
    # from the restored iters_at_level so exploration continues mid-level instead of
    # re-warming from scratch. Fresh start: iters_at_level=0 -> origin = first iter.
    _resumed_iters_at_level = curriculum._iters_at_level if curriculum is not None else 0
    level_start_step = current_step + 1 - _resumed_iters_at_level
    # Terminate on the explicit n_iters budget rather than relying on Ctrl-C.
    # `current_step` starts at -1 (see default), so the first iter bumps to 0 and the
    # loop runs for iterations 0 .. n_iters-1 inclusive.
    while current_step + 1 < n_iters:
        current_step += 1
        if curriculum is not None:
            curriculum.note_iter()

        # Seed per-iter but mix in base_seed so different runs actually get different
        # random streams. The old version reseeded from just `current_step`, which made
        # iter N of run A and iter N of run B byte-identical — the opposite of what you
        # want across experiments.
        seed = base_seed * 10_000 + current_step
        th.manual_seed(seed)
        np.random.seed(seed)
        env.seed(seed)

        for name, scheduler in schedulers.items():
            # Exploration schedulers anneal off the per-level step (re-warm each level);
            # others (e.g. lr) stay on the global step. The plot X-axis stays GLOBAL
            # (step=current_step) so the timeline is continuous and aligns with reward/
            # eval; only the VALUE is computed from the per-level step -> a sawtooth that
            # resets up at each level switch.
            _sched_step = (
                current_step - level_start_step
                if name in _PER_LEVEL_SCHED
                else current_step
            )
            scheduler.step(_sched_step)
            if logger is not None:
                logger.log_scalar(
                    name, scheduler.get_value(_sched_step), step=current_step
                )
        loss_dict = {}
        # Outer collect loop. The collector yields `frames_per_batch // minibatch_size`
        # frames per .next() call; looping `minibatch_size` times gathers exactly
        # `frames_per_batch` total frames per training iteration. The chunking exists
        # for MEMORY — the default config (150 worlds × 400 steps × 10 agents × 417
        # features × 4 bytes) would allocate >20 GB in a single-call collector, which
        # OOMs a 30 GB host.
        #
        # On torchrl 0.6 there was ALSO a per-frame GAE chunk inside
        # compute_advantage_and_target (gae_batch_size=1) because GAE ran under
        # torch.vmap and vmap tripped on our GNN's variable peer-graph sizes.
        # torchrl 0.11's GAE(deactivate_vmap=True) removes that constraint, so each
        # outer-loop iteration now calls gae() ONCE on the whole collector chunk
        # instead of once per frame. Roughly a 1000× reduction in Python-level
        # overhead per iter, at no memory cost.
        # Number of memory-chunks to split this iter's collection into. Decoupled
        # from minibatch_size (the training minibatch) so raising minibatch_size for
        # faster training does NOT shrink the collector chunk / slow collection.
        num_collect_chunks = collect_chunks if collect_chunks is not None else minibatch_size
        collector_exhausted = False
        # Gated op-level profile of collection (GIANT_COLLECT_PROF=1): shows whether
        # collector.next time is env-step (vmas physics / lidar raycast), policy
        # forward (conv/GNN), or planner resets.
        _cp = bool(os.environ.get("GIANT_COLLECT_PROF"))
        _cp_prof, _cp_n = None, 0
        if _cp:
            from torch.profiler import profile as _cpf, ProfilerActivity as _cPA
        for _ in tqdm(
            range(num_collect_chunks),
            desc="Collecting data",
            unit="batch",
            position=0,
            leave=True,
        ):
            if _cp:
                _cp_n += 1
                if _cp_n == 3 and _cp_prof is None:
                    # start profiling after 2 warmup chunks (no schedule -> events kept)
                    _cp_prof = _cpf(activities=[_cPA.CPU, _cPA.CUDA])
                    _cp_prof.start()
                elif _cp_prof is not None and _cp_n >= 9:
                    _cp_prof.stop()
                    print("\n===== COLLECTION OP-LEVEL CUDA BREAKDOWN (env step + policy) =====")
                    print(_cp_prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))
                    print("\n--- by self_cpu_time_total ---")
                    print(_cp_prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25))
                    raise SystemExit(0)
            tensordict_data, loss_dict = get_data(collector, loss_dict)
            if tensordict_data is None:
                # Collector hit its total_frames budget mid-iteration; this
                # (incomplete) iteration's data is dropped and training stops.
                collector_exhausted = True
                break
            ################## COMPUTE GAE  ##################
            from train.utils.profiling import profile_phase
            with profile_phase("compute_gae"):
                tensordict_data = compute_advantage_and_target(
                    tensordict_data,
                    loss_module,
                    gae_dummy,
                    gae_batch_size,
                    storage_device,
                    model_device,
                )
            add_to_log(
                loss_dict,
                "advantage mean",
                tensordict_data.get("advantage").mean().item(),
            )
            add_to_log(
                loss_dict,
                "value_target mean",
                tensordict_data.get("value_target").mean().item(),
            )
            data_view = tensordict_data.reshape(
                -1
            )  # Flatten the batch size (num_world x time) to shuffle data
            # Ensure contiguous storage before handing off to the replay buffer.
            # `reshape` on a non-contiguous tensor returns a view that some buffer
            # backends silently re-allocate and others reject with a shape error.
            data_view = data_view.contiguous()
            replay_buffer.extend(data_view)  # it overwrites old data

        if collector_exhausted:
            # No full batch this iteration -> all configured frames are consumed.
            # Stop cleanly; prior complete iterations are already trained + saved.
            break

        start_time = time.perf_counter()
        print("training... ")
        # Gated per-step training breakdown (cuda-synced). GIANT_TORCH_PROF=1 only.
        # Reveals where the PPO-update time goes: sample/h2d/forward/backward/log/optim.
        _tp = bool(os.environ.get("GIANT_TORCH_PROF"))
        _tp_acc = {"sample": 0.0, "h2d": 0.0, "forward": 0.0, "backward": 0.0, "log": 0.0, "optim": 0.0}
        _tp_n, _tp_warmup, _tp_active = 0, 5, 40
        # bf16/fp16 autocast for the PPO forward/backward. From the `amp_dtype` arg
        # (config perf.amp_dtype); GIANT_AMP env overrides for ad-hoc testing.
        _amp_key = os.environ.get("GIANT_AMP") or (amp_dtype or "")
        _amp_dtype = {"bf16": th.bfloat16, "fp16": th.float16}.get(_amp_key)
        _verify_prec = bool(os.environ.get("GIANT_VERIFY_PREC"))
        # Gated op-level profile (GIANT_OP_PROF=1): aten-op CUDA breakdown of fwd+bwd.
        _op = bool(os.environ.get("GIANT_OP_PROF"))
        _op_prof, _op_n = None, 0
        if _op:
            from torch.profiler import profile as _pf, ProfilerActivity as _PA, schedule as _sch
            _op_prof = _pf(activities=[_PA.CPU, _PA.CUDA],
                           schedule=_sch(wait=3, warmup=3, active=15, repeat=1))
            _op_prof.start()
        for i in range(num_epochs):
            # We need to expand the done and terminated to match the reward shape (this is expected by the value estimator)

            # print(f"\tEpoch {i}:", end="")
            ################## TRAINING ##################
            for _ in tqdm(
                range(frames_per_batch // minibatch_size),
                desc=f"Training epoch {i + 1}",
                unit="batch",
                position=0,
                leave=True,
            ):
                if _tp: th.cuda.synchronize(); _t = [time.perf_counter()]
                subdata = replay_buffer.sample()
                if _tp: th.cuda.synchronize(); _t.append(time.perf_counter())
                subdata = subdata.to(model_device)
                if _tp: th.cuda.synchronize(); _t.append(time.perf_counter())

                if _verify_prec:
                    # One-shot: same minibatch through the SAME weights in fp32 vs
                    # tf32 vs bf16; report relative diff to prove no behavior change.
                    import torch.nn.functional as _F
                    with th.no_grad():
                        sd = subdata.clone()
                        th.backends.cuda.matmul.allow_tf32 = False
                        o32 = loss_module(sd.clone())
                        ref = {k: float(o32[k]) for k in ("loss_objective", "loss_critic", "loss_entropy") if k in o32.keys()}
                        th.backends.cuda.matmul.allow_tf32 = True
                        otf = loss_module(sd.clone())
                        with th.autocast("cuda", dtype=th.bfloat16):
                            obf = loss_module(sd.clone())
                        print("\n===== PRECISION EQUIVALENCE (same weights+batch) =====")
                        for k in ref:
                            r = ref[k] if abs(ref[k]) > 1e-9 else 1.0
                            print(f"  {k:16s} fp32={ref[k]:+.6e}  tf32 rel={abs(float(otf[k])-ref[k])/abs(r):.2e}  bf16 rel={abs(float(obf[k])-ref[k])/abs(r):.2e}")
                    raise SystemExit(0)

                if _amp_dtype is not None:
                    with th.autocast("cuda", dtype=_amp_dtype):
                        loss_vals = loss_module(subdata)
                else:
                    loss_vals = loss_module(subdata)

                loss_value = (
                    loss_vals["loss_objective"] * policy_loss_factor
                    + loss_vals["loss_critic"] * value_loss_factor
                    + loss_vals["loss_entropy"]
                )
                if _tp: th.cuda.synchronize(); _t.append(time.perf_counter())

                loss_value.backward()
                if _tp: th.cuda.synchronize(); _t.append(time.perf_counter())

                loss_vals_dict = dict(loss_vals)
                # `clip_grad_norm_` with max_norm=inf just computes the norm (no clip).
                # This keeps the log path on-device; the old manual branch did a
                # per-parameter `.item()` which triggers a CUDA sync per tensor per
                # gradient step — measurable overhead on small minibatches.
                effective_max_norm = float("inf") if max_grad_norm <= 0 else max_grad_norm
                grad_norm = th.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), max_norm=effective_max_norm
                )
                loss_vals_dict["grad_norm"] = grad_norm

                loss_vals_dict["loss_value_(combined)"] = loss_value
                for key, value in loss_vals_dict.items():
                    # add_to_log handles scalar / tensor / nested TensorDict.
                    add_to_log(loss_dict, key, value)

                if _tp: th.cuda.synchronize(); _t.append(time.perf_counter())
                optim.step()
                optim.zero_grad()
                subdata = subdata.to(storage_device)
                del loss_vals_dict
                if _op_prof is not None:
                    _op_prof.step()
                    _op_n += 1
                    if _op_n >= 21:
                        _op_prof.stop()
                        print("\n===== OP-LEVEL CUDA BREAKDOWN (fwd+bwd+optim, real config) =====")
                        print(_op_prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))
                        raise SystemExit(0)
                if _tp:
                    th.cuda.synchronize(); _t.append(time.perf_counter())
                    _tp_n += 1
                    if _tp_n > _tp_warmup:
                        _tp_acc["sample"] += _t[1] - _t[0]
                        _tp_acc["h2d"] += _t[2] - _t[1]
                        _tp_acc["forward"] += _t[3] - _t[2]
                        _tp_acc["backward"] += _t[4] - _t[3]
                        _tp_acc["log"] += _t[5] - _t[4]
                        _tp_acc["optim"] += _t[6] - _t[5]
                    if _tp_n >= _tp_warmup + _tp_active:
                        _tot = sum(_tp_acc.values())
                        print("\n===== TRAINING STEP BREAKDOWN (real config, cuda-synced, "
                              f"avg of {_tp_active} steps, minibatch={minibatch_size}) =====")
                        for _k, _v in _tp_acc.items():
                            print(f"  {_k:9s} {_v / _tp_active * 1000:8.2f} ms/step  ({_v / _tot * 100:5.1f}%)")
                        print(f"  {'TOTAL':9s} {_tot / _tp_active * 1000:8.2f} ms/step")
                        raise SystemExit(0)
                # print(".", end="")
            # print("done")
        collector.update_policy_weights_()
        # Flatten `extract_logging`'s scenario-name nesting (logs["random"]
        # ["train reward"]) into slash/paren-suffixed flat keys. Downstream
        # code then sees plain lists everywhere.
        flat_loss_dict = _flatten_scenario_logs(loss_dict)
        if logger is not None:
            for name, value in flat_loss_dict.items():
                plot_name = name.replace("_", " ").replace("objective", "actor")
                mean_value = th.tensor(value).float().mean().item()
                logger.log_scalar(plot_name, mean_value, step=current_step)
            from train.utils.profiling import profile_phase
            with profile_phase("checkpoint.save"):
                logger.save_checkpoint(
                    policy, critic, optim, current_step,
                    curriculum_state=(curriculum.state_dict() if curriculum is not None else None),
                )

        from train.utils.profiling import record as _record
        _record("ppo_update_s", time.perf_counter() - start_time)
        print(f"done: ({(time.perf_counter() - start_time):.1f}s)")
        eval_str = ""
        if current_step % eval_every == 0:
            start_time = time.perf_counter()
            # Metrics every eval_every (cheap, no render). Render the videos only
            # every video_every steps — the per-step render scales with the number
            # of rendered envs, so 4-scenario videos are ~4x the eval cost; we don't
            # want that on every metrics eval.
            do_video = save_video_to_disk and (current_step % video_every == 0)
            print("Evaluating... ", end="")
            from train.utils.profiling import profile_phase
            with profile_phase("eval (full)"):
                _, eval_str, _eval_metrics = evaluate(
                    eval_env,
                    policy,
                    n_steps=_env_max_steps(eval_env),
                    current_step=current_step,
                    logger=logger,
                    save_video=do_video,
                    save_folder=str(Path(logger.save_dir, "video"))
                    if logger
                    else str(Path(os.getcwd(), "results", "video")),
                    save_name=f"eval_{current_step}.mp4",
                    eval_render_to_screen=eval_render_to_screen,
                    eval_seed=eval_seed,
                    render_env_index=render_env_index,
                )
            print(f"done: ({(time.perf_counter() - start_time):.1f}s)")

            # --- adaptive curriculum: advance/demote level on eval metrics ---
            if curriculum is not None and on_level_switch is not None:
                action = curriculum.update(_eval_metrics)
                if action is not None:
                    print(f"[curriculum] {action.upper()} -> level {curriculum.level_idx} "
                          f"({curriculum.name}) at iter {current_step}; metrics={_eval_metrics}")
                    new_objs = on_level_switch(curriculum.level_overrides())
                    if new_objs is not None:
                        # frames_per_batch may change if the level sets its own
                        # num_worlds -> update it so the per-iter minibatch count
                        # (frames_per_batch // minibatch_size) tracks the new size.
                        env, eval_env, collector, replay_buffer, frames_per_batch = new_objs
                    # Re-warm exploration for the new level: the std_max/entropy schedule
                    # restarts from its start value (anneals over the next decay_steps).
                    # Next iter is current_step+1, so per-level step there = 0.
                    level_start_step = current_step + 1
                    if logger is not None:
                        logger.log_scalar("curriculum level", curriculum.level_idx, step=current_step)
        # Pool across every scenario that produced this metric — with
        # scenario_type="multi" we have one entry per sub-scenario
        # ("train reward (random)", "train reward (circle)", …) and want
        # a single progress-bar summary.
        episode_reward_mean = th.tensor(
            _pool_metric_across_scenarios(flat_loss_dict, "train reward")
        ).float().mean().item()
        vel = th.tensor(
            _pool_metric_across_scenarios(flat_loss_dict, "train vel mean")
        ).float().mean().item()
        pbar.set_description(
            f"train - episode_reward_mean = {episode_reward_mean:.4f}, vel mean = {vel:.2f}, {eval_str}",
            refresh=True,
        )
        print(
            f"train - episode_reward_mean = {episode_reward_mean:.4f}, vel mean = {vel:.2f}, {eval_str}"
        )
        pbar.update()
        print("\n")


def make_logger(load_model, load_only_model, config):
    """Build a logger backend based on ``config["logging_backend"]``.

    Backends:
      - ``"wandb"`` (default for backward compat): MyWandbLogger. Requires a
        valid W&B API key in the environment.
      - ``"file"``: FileLogger. Writes scalars to JSONL + checkpoints to disk.
        No network, no auth.
      - ``"none"``: NoOpLogger. Swallows everything. Useful for benchmarks.

    The checkpoint-resume path (deriving exp_name / save_dir / id from the
    checkpoint directory) is wandb-specific; for file/none backends the
    derived values are ignored.
    """
    backend = (config or {}).get("logging_backend", "wandb")

    if backend == "none":
        from train.utils.loggers import NoOpLogger
        return NoOpLogger()

    if backend == "file":
        from train.utils.loggers import FileLogger
        # Mirror the wandb backend's resume-from-checkpoint logic where possible
        # — a prior run's save_dir lives at checkpoint.parent.parent.
        exp_name = None
        save_dir = None
        if load_model is not None and not load_only_model:
            ckpt_path = Path(load_model)
            save_dir = ckpt_path.parent.parent
            exp_name = save_dir.name
        return FileLogger(exp_name=exp_name, save_dir=save_dir, config=config)

    # Default: wandb
    logging_kwargs = {
        "exp_name": None,
        "save_dir": None,
        "id": None,
        "project_name": "giant",
        "offline": False,
        "config": config,
    }
    if load_model is not None and not load_only_model:
        exp_name, save_dir, id = MyWandbLogger.get_log_variables_from_checkpoint(
            load_model
        )
        logging_kwargs["exp_name"] = exp_name
        logging_kwargs["save_dir"] = save_dir
        logging_kwargs["id"] = id

    return MyWandbLogger(**logging_kwargs)
