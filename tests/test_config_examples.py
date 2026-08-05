"""Smoke coverage for every YAML under configs/.

Ensures that each example file can be loaded + merged with the defaults without
producing an obviously-broken config. This is the "configs/examples won't drift"
guard called out in plan.md §Decisions.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_REPO = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# Eval configs (configs/eval.yaml, configs/paper_videos.yaml) carry a different
# schema (checkpoint/scenarios/...) and are consumed by the eval scripts, not by
# train.config.resolve_config. Identify them by their top-level keys so the two
# families get validated against the right expectations.
_EVAL_CONFIG_KEYS = ("checkpoint:", "scenarios:", "model_type:")


def _looks_like_eval_config(path):
    for line in path.read_text().splitlines():
        if line.startswith(_EVAL_CONFIG_KEYS):
            return True
    return False


def _all_config_yamls():
    configs_dir = _REPO / "configs"
    if not configs_dir.exists():
        return []
    return sorted(configs_dir.rglob("*.yaml")) + sorted(configs_dir.rglob("*.yml"))


TRAINING_CONFIGS = [p for p in _all_config_yamls() if not _looks_like_eval_config(p)]
EVAL_CONFIGS = [p for p in _all_config_yamls() if _looks_like_eval_config(p)]


@pytest.mark.skipif(not TRAINING_CONFIGS, reason="no training configs under configs/")
@pytest.mark.parametrize(
    "config_path", TRAINING_CONFIGS, ids=lambda p: p.relative_to(_REPO).as_posix()
)
def test_config_yaml_resolves(config_path):
    pytest.importorskip("yaml")
    from train.config import resolve_config

    cfg = resolve_config(str(config_path))
    # After merging with defaults, every full config needs these keys.
    assert "ppo" in cfg
    assert cfg["ppo"]["n_iters"] > 0
    assert cfg["ppo"]["max_steps"] > 0
    assert cfg["num_worlds"] > 0
    assert cfg["logging_backend"] in {"wandb", "file", "none"}
    # And frames_per_batch must be sane (num_worlds * max_steps).
    assert cfg["ppo"]["frames_per_batch"] > 0


@pytest.mark.skipif(not EVAL_CONFIGS, reason="no eval configs under configs/")
@pytest.mark.parametrize(
    "config_path", EVAL_CONFIGS, ids=lambda p: p.relative_to(_REPO).as_posix()
)
def test_eval_config_yaml_loads(config_path):
    """Eval configs must load and expose the keys the eval CLIs read."""
    pytest.importorskip("yaml")
    from train.config import load_config_file

    cfg = load_config_file(str(config_path))
    assert isinstance(cfg, dict)
    assert cfg.get("model_type") in {
        "oursGraph", "oursD", "oursDV", "baseline", "RVO", "GA3CPolicy"
    }
    assert isinstance(cfg.get("scenarios"), list) and cfg["scenarios"]
    # Optional deep-override blocks, when present, must be mappings.
    for key in ("config", "scenario_config"):
        if cfg.get(key) is not None:
            assert isinstance(cfg[key], dict)
