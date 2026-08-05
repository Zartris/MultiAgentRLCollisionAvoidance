"""Tests for the pluggable logger backends + the `extract_logging` flat-key fix."""
from __future__ import annotations

import json
import os
import sys

import pytest
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


# ---------- NoOpLogger --------------------------------------------------------
def test_noop_logger_accepts_every_interface_call():
    """Every call on NoOpLogger returns None and raises nothing."""
    from train.utils.loggers import NoOpLogger

    logger = NoOpLogger()
    # Interface-compat surface that PPOTrainer uses.
    assert logger.log_scalar("foo", 1.0, step=0) is None
    assert logger.log_scalar("bar", torch.tensor(1.0), step=1) is None
    assert logger.log_image("img", torch.zeros(3, 4, 4)) is None
    # save_dir must be Path-like so downstream `Path(logger.save_dir, "video")`
    # callers don't explode.
    from pathlib import Path as _Path
    assert isinstance(logger.save_dir, _Path)


# ---------- FileLogger --------------------------------------------------------
def test_file_logger_writes_scalars_jsonl(tmp_path):
    from train.utils.loggers import FileLogger

    logger = FileLogger(exp_name="unit-test", save_dir=tmp_path)
    logger.log_scalar("loss", 0.42, step=0)
    logger.log_scalar("loss", 0.37, step=1)
    logger.log_scalar("reward", torch.tensor(1.5), step=1)

    scalars_file = tmp_path / "scalars.jsonl"
    assert scalars_file.exists()
    lines = [json.loads(line) for line in scalars_file.read_text().splitlines()]
    assert [(r["name"], r["step"]) for r in lines] == [
        ("loss", 0, ),
        ("loss", 1),
        ("reward", 1),
    ][:3] or [(r["name"], r["step"]) for r in lines] == [
        ("loss", 0),
        ("loss", 1),
        ("reward", 1),
    ]
    # Tensor values get converted to plain numbers.
    assert lines[2]["value"] == 1.5


def test_file_logger_save_checkpoint_round_trip(tmp_path):
    from train.utils.loggers import FileLogger
    from torch import nn

    logger = FileLogger(exp_name="ckpt-test", save_dir=tmp_path)
    policy = nn.Linear(4, 2)
    critic = nn.Linear(4, 1)
    opt = torch.optim.Adam(list(policy.parameters()) + list(critic.parameters()))

    logger.save_checkpoint(policy, critic, opt, step=5)
    ckpt = torch.load(tmp_path / "checkpoints" / "checkpoint_5.pth")
    # Default is the verbose schema, matching common.save_checkpoint.
    assert "policy_state_dict" in ckpt
    assert "critic_state_dict" in ckpt


def test_file_logger_legacy_format_opt_in(tmp_path):
    from train.utils.loggers import FileLogger
    from torch import nn

    logger = FileLogger(exp_name="ckpt-legacy", save_dir=tmp_path)
    policy = nn.Linear(4, 2)
    critic = nn.Linear(4, 1)
    opt = torch.optim.Adam(list(policy.parameters()) + list(critic.parameters()))

    logger.save_checkpoint(policy, critic, opt, step=3, legacy_format=True)
    ckpt = torch.load(tmp_path / "checkpoints" / "checkpoint_3.pth")
    assert set(ckpt.keys()) == {"policy", "critic", "optimizer", "step"}


# ---------- make_logger dispatch ---------------------------------------------
def test_make_logger_backend_none(tmp_path):
    from train.utils.common import make_logger

    cfg = {"logging_backend": "none", "model": "oursGraph"}
    logger = make_logger(load_model=None, load_only_model=False, config=cfg)
    # NoOpLogger's defining attribute is that it ignores every call.
    logger.log_scalar("x", 1.0, step=0)
    logger.save_checkpoint(None, None, None, step=0)  # None args are fine — it's a no-op


def test_make_logger_backend_file(tmp_path, monkeypatch):
    from train.utils.common import make_logger
    from train.utils.loggers import FileLogger

    # Point FileLogger at tmp_path so we don't write into the real results/ dir.
    monkeypatch.chdir(tmp_path)
    cfg = {"logging_backend": "file", "model": "oursGraph"}
    logger = make_logger(load_model=None, load_only_model=False, config=cfg)
    assert isinstance(logger, FileLogger)
    logger.log_scalar("y", 2.0, step=0)
    # The scalars file now exists inside logger.save_dir.
    assert (logger.save_dir / "scalars.jsonl").exists()


# ---------- extract_logging flat-key mirror -----------------------------------
def test_extract_logging_mirrors_scenario_keys_flat(monkeypatch):
    """Regression guard for common.py:907 reading `loss_dict["train reward"]`.

    `extract_logging` previously only wrote `logs[scenario_name][key]` — the flat
    lookup at line 907 would `KeyError` every time. The new flat mirror step
    should append the per-scenario values into a top-level list under the same
    key name, so the lookup succeeds.
    """
    from train.utils import common as cm

    # Avoid calling the real extract_scenario_logging (needs a full rollout td).
    # Replace it with a stub that writes two known keys into the inner dict.
    def _stub_inner(current, nxt, env, prefix, logs):
        logs[f"{prefix} reward"] = [0.1, 0.2]
        logs[f"{prefix} vel mean"] = [0.5]
        return logs

    monkeypatch.setattr(cm, "extract_scenario_logging", _stub_inner)

    # Fake tensordict that returns two scenario ids when you ask for the
    # scenario_name key, and whose 'agents' / 'next.agents' slices can be
    # `.view`ed to match the real shape contract. We only need the iteration to
    # visit two scenarios and call the stub twice.
    class _FakeTD:
        shape = (2, 3)

        def __init__(self):
            # (W=2, T=3, A=2, 1) scenario_name tensor: first scenario "random"=0,
            # second "circle"=1 — need to hit int_to_scenario_name for both.
            import torch
            self._name = torch.tensor([[[[0], [1]]]]).expand(2, 3, 2, 1)
            self._agents = torch.zeros(2, 3, 2, 4)
            self._next_agents = torch.zeros(2, 3, 2, 4)

        def get(self, key):
            import torch
            if key == ("next", "agents", "info", "scenario_name"):
                return self._name
            if key == "agents":
                return self._agents
            if key == ("next", "agents"):
                return self._next_agents
            raise KeyError(key)

    td = _FakeTD()
    logs = cm.extract_logging(td, env=None, prefix="train")

    # Nested keys exist per scenario.
    assert "random" in logs
    assert "circle" in logs
    assert logs["random"]["train reward"] == [0.1, 0.2]

    # And the flat mirror now also has the keys — both scenarios' values
    # concatenated into one list.
    assert "train reward" in logs, "flat mirror must expose scenario keys at top level"
    assert sorted(logs["train reward"]) == [0.1, 0.1, 0.2, 0.2]
    assert sorted(logs["train vel mean"]) == [0.5, 0.5]
