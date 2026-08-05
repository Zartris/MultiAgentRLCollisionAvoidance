"""Logger backends for training runs.

Three backends live here with a common interface so that PPOTrainer doesn't
care where metrics end up:

  - ``WandbLoggerBackend`` -> re-exports the existing MyWandbLogger (wandb
    cloud, needs an API key).
  - ``FileLogger`` -> writes scalars to JSONL under ``save_dir/scalars.jsonl``
    and checkpoints to ``save_dir/checkpoints/``. No network, no auth.
  - ``NoOpLogger`` -> swallows everything. Useful for benchmarks, CI, and
    smoke tests where logging output is noise.

Use ``make_logger(config, ...)`` in train.utils.common to pick one; the key
``config["logging_backend"]`` selects ``"wandb"`` (default for backward compat),
``"file"``, or ``"none"``.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import torch as th


def _default_exp_name(config: Optional[dict]) -> str:
    time_formatted = datetime.now().strftime("%Hh-%Mm-%Ss")
    date_formatted = datetime.now().strftime("%Y-%m-%d")
    model_type = (config or {}).get("model", "run")
    return f"{date_formatted}/{model_type}_{time_formatted}"


def _default_save_dir(exp_name: str) -> Path:
    return Path(os.getcwd(), "results", exp_name)


class NoOpLogger:
    """Swallows every log call. Interface-compatible with MyWandbLogger.

    Instantiate with ``NoOpLogger()``; no side effects. ``save_dir`` points at
    a throwaway path so any consumer doing ``Path(logger.save_dir, "video")``
    still returns a valid Path.
    """

    def __init__(self, save_dir: Optional[Union[str, Path]] = None):
        self.save_dir = Path(save_dir) if save_dir else Path("/tmp/malp_noop")

    def log_scalar(self, name: str, value: Any, step: Optional[int] = None) -> None:  # noqa: D401
        return None

    def log_image(self, name: str, image: Any, **kwargs) -> None:  # noqa: D401
        return None

    def log_video(self, name: str, video: Any, **kwargs) -> None:  # noqa: D401
        return None

    def save_checkpoint(
        self, policy, critic, optimizer, step: int, legacy_format: bool = False,
        curriculum_state: dict = None,
    ) -> None:
        return None

    def save_network_summary(self, policy, critic, env) -> None:
        return None


class FileLogger:
    """File-only logger. Writes scalars to JSONL and checkpoints to disk.

    Layout::

        <save_dir>/
            scalars.jsonl     # one JSON object per log_scalar call
            checkpoints/
                checkpoint_<step>.pth
            networks/         # written by save_network_summary
                actor.txt
                critic.txt

    Images and videos are skipped (with a one-time warning). This keeps the
    logger dependency-free — no pillow/opencv required at log time.
    """

    _image_warning_printed = False
    _video_warning_printed = False

    def __init__(
        self,
        exp_name: Optional[str] = None,
        save_dir: Optional[Union[Path, str]] = None,
        config: Optional[dict] = None,
    ):
        if exp_name is None:
            exp_name = _default_exp_name(config)
        self.exp_name = exp_name
        if save_dir is None:
            save_dir = _default_save_dir(exp_name)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._scalars_path = self.save_dir / "scalars.jsonl"
        # Append mode so resuming a run keeps history; callers wanting a fresh
        # file should delete save_dir between runs.
        self._scalars_fh = open(self._scalars_path, "a")

    def __del__(self):
        fh = getattr(self, "_scalars_fh", None)
        if fh is not None and not fh.closed:
            try:
                fh.close()
            except Exception:
                pass

    def log_scalar(self, name: str, value: Any, step: Optional[int] = None) -> None:
        if isinstance(value, th.Tensor):
            value = value.detach().cpu().item() if value.numel() == 1 else value.tolist()
        record = {"name": name, "value": value, "step": step,
                  "t": datetime.now().isoformat(timespec="seconds")}
        self._scalars_fh.write(json.dumps(record) + "\n")
        self._scalars_fh.flush()

    def log_image(self, name: str, image: Any, **kwargs) -> None:
        if not FileLogger._image_warning_printed:
            print("FileLogger: log_image is a no-op (images are not persisted).")
            FileLogger._image_warning_printed = True

    def log_video(self, name: str, video: Any, **kwargs) -> None:
        if not FileLogger._video_warning_printed:
            print("FileLogger: log_video is a no-op (videos are not persisted).")
            FileLogger._video_warning_printed = True

    def save_checkpoint(
        self, policy, critic, optimizer, step: int, legacy_format: bool = False,
        curriculum_state: dict = None,
    ) -> None:
        ckpt_dir = self.save_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"checkpoint_{step}.pth"
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
        th.save(payload, str(ckpt_path))

    def save_network_summary(self, policy, critic, env) -> None:
        from torch_geometric.nn import summary

        net_summary_dir = self.save_dir / "networks"
        net_summary_dir.mkdir(parents=True, exist_ok=True)
        actor_table = summary(policy, env.reset())
        (net_summary_dir / "actor.txt").write_text(str(actor_table))
        critic_table = summary(critic, env.reset())
        (net_summary_dir / "critic.txt").write_text(str(critic_table))
