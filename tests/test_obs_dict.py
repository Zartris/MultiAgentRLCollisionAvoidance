"""Shape + layout contract for scenario.CollisionAvoidance_base.observation_to_dict.

The refactor target here is the per-agent observation vector produced by the scenario
and consumed by the model. If either side changes independently, the layout breaks in
silent, hard-to-debug ways. These tests pin the layout as-is.
"""
import torch
from scenario.CollisionAvoidance_base import observation_to_dict


def test_obs_dict_full_config(small_obs_config):
    cfg = small_obs_config
    B, N, H, R = cfg["n_worlds"] * cfg["n_agents"], cfg["n_agents"], cfg["lidar_history_len"], cfg["num_lidar_rays"]
    F = 2 + 6 + 5 + 4 * (N - 1) + H * R
    obs = torch.zeros(B, F)
    d = observation_to_dict(obs, True, H, R, use_global_path=True, num_agents=N)

    assert d["goal"].shape == (B, 2)
    assert d["state"].shape == (B, 6)
    assert d["global_path"].shape == (B, 5)
    # other_agents gets unflattened to (B, N-1, 4) — this is the reshape the doc-string
    # calls out as "collapses ALL leading dims". Here B is already flat so the shape is
    # (B, N-1, 4) exactly.
    assert d["other_agents"].shape == (B, N - 1, 4)
    assert d["lidar_dist"].shape == (B, H, R)


def test_obs_dict_no_global_path(small_obs_config):
    cfg = small_obs_config
    B, N, H, R = 5, cfg["n_agents"], cfg["lidar_history_len"], cfg["num_lidar_rays"]
    F = 2 + 6 + 4 * (N - 1) + H * R  # no global_path
    obs = torch.zeros(B, F)
    d = observation_to_dict(obs, True, H, R, use_global_path=False, num_agents=N)

    assert "global_path" not in d
    assert d["goal"].shape == (B, 2)
    assert d["state"].shape == (B, 6)
    assert d["other_agents"].shape == (B, N - 1, 4)
    assert d["lidar_dist"].shape == (B, H, R)


def test_obs_dict_single_agent_no_peers():
    # With num_agents == 1 the 'other_agents' block disappears entirely.
    H, R, N = 3, 60, 1
    F = 2 + 6 + H * R
    obs = torch.zeros(4, F)
    d = observation_to_dict(obs, True, H, R, use_global_path=False, num_agents=N)
    assert "other_agents" not in d
    assert d["lidar_dist"].shape == (4, H, R)


def test_obs_dict_no_lidar():
    # use_lidar=False drops the trailing lidar block.
    N = 3
    F = 2 + 6 + 4 * (N - 1)
    obs = torch.zeros(2, F)
    d = observation_to_dict(obs, False, 3, 60, use_global_path=False, num_agents=N)
    assert "lidar_dist" not in d


def test_obs_dict_preserves_leading_dims(small_obs_config):
    # The reshapes in observation_to_dict should operate only on the last axis. This is
    # critical because callers sometimes pass (worlds, agents, F) directly, and collapsing
    # those dims would be catastrophic.
    cfg = small_obs_config
    N, H, R = cfg["n_agents"], cfg["lidar_history_len"], cfg["num_lidar_rays"]
    F = 2 + 6 + 5 + 4 * (N - 1) + H * R
    # 4D input: (worlds, time, agents, F)
    obs = torch.zeros(3, 4, N, F)
    d = observation_to_dict(obs, True, H, R, use_global_path=True, num_agents=N)
    # goal/state/global_path just slice the last axis -> leading dims preserved
    assert d["goal"].shape == (3, 4, N, 2)
    assert d["state"].shape == (3, 4, N, 6)
    assert d["global_path"].shape == (3, 4, N, 5)
    # lidar_dist unflattens the last axis only
    assert d["lidar_dist"].shape == (3, 4, N, H, R)
    # other_agents is the outlier: reshape(-1, N-1, 4) collapses the leading dims.
    # The total element count must still match, though.
    assert d["other_agents"].shape == (3 * 4 * N, N - 1, 4)


def test_obs_dict_slices_do_not_overlap(small_obs_config):
    # A regression check: if someone bumps one field's width but forgets to advance
    # current_index, later fields will quietly read the wrong bytes. Here we write a
    # distinct marker into each region of the flat obs and verify each key reads its
    # own region back.
    cfg = small_obs_config
    N, H, R = cfg["n_agents"], cfg["lidar_history_len"], cfg["num_lidar_rays"]
    F = 2 + 6 + 5 + 4 * (N - 1) + H * R
    obs = torch.arange(F, dtype=torch.float32).unsqueeze(0)  # (1, F) monotonic

    d = observation_to_dict(obs, True, H, R, use_global_path=True, num_agents=N)
    # goal is the first 2 values -> 0, 1
    assert torch.equal(d["goal"][0], torch.tensor([0., 1.]))
    # state is next 6 values -> 2..7
    assert torch.equal(d["state"][0], torch.arange(2, 8, dtype=torch.float32))
    # global_path follows at index 8..12
    assert torch.equal(d["global_path"][0], torch.arange(8, 13, dtype=torch.float32))
    # lidar_dist is the last block and ends at F
    assert d["lidar_dist"][0, -1, -1].item() == float(F - 1)
