"""Unit tests for train.config and the CLI override plumbing in LidarSingleStep."""
from __future__ import annotations

import json
import os
import sys

import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


# ---------- build_default_config ----------------------------------------------
def test_default_config_has_expected_top_level_keys():
    from train.config import build_default_config

    cfg = build_default_config()
    # Fields that the training loop directly indexes.
    required = {
        "model", "scenario_type", "num_agents", "num_worlds",
        "load_model", "load_only_model", "log", "logging_backend",
        "base_seed", "max_runtime_hours",
        "ppo", "baseline_config", "multi_config", "multi_config_eval",
    }
    missing = required - set(cfg.keys())
    assert not missing, f"default config is missing required keys: {missing}"


def test_default_config_ppo_block_has_valid_shape():
    from train.config import build_default_config

    cfg = build_default_config()
    ppo = cfg["ppo"]
    assert ppo["frames_per_batch"] == ppo["max_steps"] * cfg["num_worlds"]
    # deactivate_vmap mode — gae_batch_size should match the per-chunk size.
    assert ppo["gae_batch_size"] == ppo["max_steps"]
    assert ppo["n_iters"] > 0
    assert ppo["num_epochs"] > 0


def test_default_config_is_fresh_copy():
    """Callers own the result; mutating it must not affect subsequent calls."""
    from train.config import build_default_config

    a = build_default_config()
    a["num_worlds"] = 999
    a["ppo"]["n_iters"] = 7
    b = build_default_config()
    assert b["num_worlds"] != 999
    assert b["ppo"]["n_iters"] != 7


# ---------- merge_overrides ---------------------------------------------------
def test_merge_overrides_scalar_replaces():
    from train.config import merge_overrides

    out = merge_overrides({"a": 1, "b": 2}, {"a": 99})
    assert out == {"a": 99, "b": 2}


def test_merge_overrides_nested_dict_merges():
    from train.config import merge_overrides

    base = {"a": {"x": 1, "y": 2}, "b": 3}
    override = {"a": {"y": 20, "z": 30}}
    out = merge_overrides(base, override)
    assert out == {"a": {"x": 1, "y": 20, "z": 30}, "b": 3}


def test_merge_overrides_does_not_mutate_inputs():
    from train.config import merge_overrides

    base = {"a": {"x": 1}}
    override = {"a": {"y": 2}}
    _ = merge_overrides(base, override)
    assert base == {"a": {"x": 1}}
    assert override == {"a": {"y": 2}}


def test_merge_overrides_dict_replacing_scalar():
    """If the base stored a scalar and the override is a dict, the override wins."""
    from train.config import merge_overrides

    out = merge_overrides({"a": 5}, {"a": {"nested": True}})
    assert out == {"a": {"nested": True}}


def test_merge_overrides_warns_on_new_keys(capsys):
    from train.config import merge_overrides

    out = merge_overrides({"a": 1}, {"new_knob": 42})
    assert out["new_knob"] == 42
    captured = capsys.readouterr()
    assert "new key" in captured.out


# ---------- load_config_file --------------------------------------------------
def test_load_yaml_config(tmp_path):
    pytest.importorskip("yaml")
    from train.config import load_config_file

    p = tmp_path / "test.yaml"
    p.write_text(
        "num_worlds: 32\n"
        "ppo:\n"
        "  n_iters: 5\n"
        "  lr_start: 0.0001\n"
    )
    cfg = load_config_file(p)
    assert cfg == {"num_worlds": 32, "ppo": {"n_iters": 5, "lr_start": 0.0001}}


def test_load_json_config(tmp_path):
    from train.config import load_config_file

    p = tmp_path / "test.json"
    p.write_text(json.dumps({"num_worlds": 8, "ppo": {"n_iters": 2}}))
    cfg = load_config_file(p)
    assert cfg == {"num_worlds": 8, "ppo": {"n_iters": 2}}


def test_load_config_file_missing_raises(tmp_path):
    from train.config import load_config_file

    with pytest.raises(FileNotFoundError):
        load_config_file(tmp_path / "does-not-exist.yaml")


def test_load_config_file_rejects_unknown_extension(tmp_path):
    from train.config import load_config_file

    p = tmp_path / "x.txt"
    p.write_text("irrelevant")
    with pytest.raises(ValueError, match="Unsupported"):
        load_config_file(p)


# ---------- resolve_config ----------------------------------------------------
def test_resolve_config_defaults_only():
    from train.config import build_default_config, resolve_config

    cfg = resolve_config(None)
    assert cfg == build_default_config()


def test_resolve_config_file_overrides_defaults(tmp_path):
    pytest.importorskip("yaml")
    from train.config import resolve_config

    p = tmp_path / "o.yaml"
    p.write_text("num_worlds: 64\nppo:\n  n_iters: 11\n")
    cfg = resolve_config(p)
    assert cfg["num_worlds"] == 64
    assert cfg["ppo"]["n_iters"] == 11
    # A key not in the override keeps its default value.
    assert cfg["ppo"]["lr_start"] == 2e-5


def test_resolve_config_programmatic_overrides_win(tmp_path):
    pytest.importorskip("yaml")
    from train.config import resolve_config

    p = tmp_path / "o.yaml"
    p.write_text("num_worlds: 64\n")
    cfg = resolve_config(p, overrides={"num_worlds": 128})
    assert cfg["num_worlds"] == 128
