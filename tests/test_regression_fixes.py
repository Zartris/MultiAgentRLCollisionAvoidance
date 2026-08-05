"""Regression guards for previously fixed bugs.

Every test here calls the real function / imports the real module and asserts an
externally-visible behavior. Tests that only re-implemented the 1-3 lines of the fix
and asserted the re-implementation ("tautology tests") were deliberately not kept —
they don't catch regressions in the real code.
"""
import io
import sys

import pytest
import torch
from torch import nn


# ---------- A8: typing.Optional import -------------------------------------------
def test_collision_avoidance_base_imports_without_gitpython():
    # Exercising the module import is enough — if `from git import Optional` came back,
    # this would raise `ImportError: cannot import name 'Optional' from 'git'` on any
    # machine without GitPython.
    import scenario.CollisionAvoidance_base as m
    from typing import Optional
    # Make sure the name resolves to typing.Optional, not a git re-export.
    assert m.Optional is Optional


# ---------- A3: profile import is guarded ---------------------------------------
def test_model_module_imports_when_line_profiler_pycharm_missing(monkeypatch):
    """Simulate a container where `line_profiler_pycharm` isn't installed."""
    # Force a fresh import under a patched finder that rejects the profiler.
    import importlib

    # Drop cached modules so the reimport actually runs the top-level try/except.
    for mod in list(sys.modules):
        if mod.startswith("models.MultiAgentLidarModel") or mod == "line_profiler_pycharm":
            monkeypatch.delitem(sys.modules, mod, raising=False)

    # Install a meta-finder that blocks the profiler import.
    class _BlockProfiler:
        def find_spec(self, name, path=None, target=None):
            if name == "line_profiler_pycharm":
                raise ImportError("simulated missing dep")
            return None

    monkeypatch.setattr(sys, "meta_path", [_BlockProfiler(), *sys.meta_path])
    mod = importlib.import_module("models.MultiAgentLidarModel")
    # profile should be a no-op function that returns its arg unchanged.
    assert callable(mod.profile)

    def _sample(x):
        return x + 1

    assert mod.profile(_sample)(2) == 3


# ---------- A1: save_checkpoint emits both schemas ------------------------------
def test_save_checkpoint_legacy_format_round_trip(tmp_path):
    from train.utils.common import save_checkpoint

    policy = nn.Linear(4, 2)
    critic = nn.Linear(4, 1)
    opt = torch.optim.Adam(list(policy.parameters()) + list(critic.parameters()))
    save_checkpoint(policy, critic, opt, step=3, checkpoint_dir=str(tmp_path), legacy_format=True)
    ckpt = torch.load(tmp_path / "checkpoint_3.pth")
    # Legacy opt-in should produce short-key payload.
    assert set(ckpt.keys()) == {"policy", "critic", "optimizer", "step"}


def test_save_checkpoint_default_uses_verbose_schema(tmp_path):
    from train.utils.common import save_checkpoint

    policy = nn.Linear(4, 2)
    critic = nn.Linear(4, 1)
    opt = torch.optim.Adam(list(policy.parameters()) + list(critic.parameters()))
    save_checkpoint(policy, critic, opt, step=5, checkpoint_dir=str(tmp_path))
    ckpt = torch.load(tmp_path / "checkpoint_5.pth")
    assert "policy_state_dict" in ckpt
    assert "critic_state_dict" in ckpt
    assert "policy" not in ckpt


# ---------- A4: STDDecay accepts inner module ------------------------------------
def test_std_decay_accepts_inner_module_directly():
    from train.utils.schedulers import STDDecay

    class _Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(2, 2)
            self._std_max = 1.0

        def set_std_max(self, v):
            self._std_max = v

    inner = _Toy()
    sched = STDDecay(inner, start=1.0, end=0.1, decay_steps=10)
    sched.step(5)
    # Value halfway through the decay is linearly interpolated.
    assert abs(inner._std_max - 0.55) < 1e-6


def test_std_decay_rejects_non_module_with_clear_error():
    from train.utils.schedulers import STDDecay

    with pytest.raises(AttributeError, match="set_std_max"):
        STDDecay(object(), start=1.0, end=0.1, decay_steps=10)


# ---------- B1: logger no longer calls input() ---------------------------------
def test_logger_get_id_from_experience_does_not_call_input(monkeypatch):
    # Swap stdin so input() would immediately EOF and raise, then make sure we
    # actually hit the fallback path.
    monkeypatch.setattr("sys.stdin", io.StringIO(""))

    class _FakeRun:
        def __init__(self, name, id):
            self.name = name
            self.id = id

    class _FakeApi:
        def runs(self, project):
            return [_FakeRun("other-run", "abc")]

    fake_wandb = type("_Fake", (), {"Api": lambda self: _FakeApi()})()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    from train.utils.logger import MyWandbLogger

    result = MyWandbLogger.get_id_from_experience("missing-run", project_name="p")
    assert result is None
