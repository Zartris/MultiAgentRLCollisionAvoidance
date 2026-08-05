"""prepare_obs shape contract on the LocalNavigationNetBase hierarchy.

This is the single biggest "batch abuse" site in the codebase: every net forward()
flattens (worlds, agents) into one batch dim B before calling prepare_obs, and the
GNN branch depends on B being flat. If a refactor ever delivers non-flat inputs to
prepare_obs, these tests fail.
"""
import torch

from models.MultiAgentLidarModel import LocalNavigationGraphNetDist


def _make_net(cfg, dynamics, device="cpu", use_global_path_obs=True):
    return LocalNavigationGraphNetDist(
        lidar_history_len=cfg["lidar_history_len"],
        lidar_num_rays=cfg["num_lidar_rays"],
        # state_input_dim=4 is the "pre-GP" state (goal_polar(2) + vel_mag(1) + ang_vel(1)).
        # The net adds +3 for the GP tail internally when use_global_path_obs=True.
        state_input_dim=4,
        conv_channels=[32, 32],
        kernel_sizes=[5, 2],
        gnn_emb_size=16,
        output_dim=2,
        dynamics=dynamics,
        use_global_path_obs=use_global_path_obs,
        set_gp_as_goal=False,
        device=device,
    )


import pytest


@pytest.fixture
def _obs_cfg(request):
    """Resolver for parametrize over the two conftest fixtures."""
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize("_obs_cfg", ["small_obs_config", "small_obs_config_no_gp"],
                         indirect=True, ids=["gp", "no_gp"])
def test_prepare_obs_flat_input(_obs_cfg, small_dynamics):
    """Shape contract for prepare_obs across the GP on/off branches.

    state_data dim depends on the config: 7 with GP (goal_polar(2) + vel_mag +
    ang_vel + GP tail(3)), 4 without. The state_input_dim passed to the net is the
    pre-GP width (4); the net internally handles the +3 when use_global_path_obs=True.
    """
    cfg = _obs_cfg
    net = _make_net(cfg, small_dynamics, use_global_path_obs=cfg["use_global_path"])
    B, N = cfg["n_worlds"] * cfg["n_agents"], cfg["n_agents"]
    H, R = cfg["lidar_history_len"], cfg["num_lidar_rays"]
    F = 2 + 6 + (5 if cfg["use_global_path"] else 0) + 4 * (N - 1) + H * R

    torch.manual_seed(0)
    obs = torch.randn(B, F) * 0.3
    # Bias peers so some survive the GNN distance+nonzero-vel mask.
    peer_start = 2 + 6 + (5 if cfg["use_global_path"] else 0)
    for p in range(N - 1):
        obs[..., peer_start + 4 * p + 0] = 0.1  # dx
        obs[..., peer_start + 4 * p + 1] = 0.1  # dy
        obs[..., peer_start + 4 * p + 2] = 0.5  # vx
        obs[..., peer_start + 4 * p + 3] = 0.5  # vy

    lidar, state, graph = net.prepare_obs(obs, num_agents=N)
    assert lidar.shape == (B, H, R)
    assert state.shape == (B, 7 if cfg["use_global_path"] else 4)
    # Dense peer representation: (peers [B, N-1, 4], mask [B, N-1]).
    assert graph is not None
    peers, mask = graph
    assert peers.shape == (B, N - 1, 4)
    assert mask.shape == (B, N - 1)
    assert mask.any()  # peers were biased close + moving, so some qualify


def test_prepare_obs_no_peers_qualify_gives_empty_mask(small_obs_config, small_dynamics):
    # All peers at zero velocity -> mask filters them all out. The dense GNN then
    # sums to zero, exactly like the old "graph is None -> zero buffer" behavior.
    cfg = small_obs_config
    net = _make_net(cfg, small_dynamics)
    B, N = cfg["n_worlds"] * cfg["n_agents"], cfg["n_agents"]
    H, R = cfg["lidar_history_len"], cfg["num_lidar_rays"]
    F = 2 + 6 + 5 + 4 * (N - 1) + H * R
    obs = torch.zeros(B, F)  # all zeros -> peer vel is zero
    lidar, state, graph = net.prepare_obs(obs, num_agents=N)
    peers, mask = graph
    assert not mask.any()


def test_prepare_obs_graph_populates_all_peers_when_all_qualify(small_dynamics):
    # Put peers close + moving, expect exactly B * (N-1) peer entries in the graph.
    dyn = dict(small_dynamics)
    dyn["lidar_obs_range"] = 100.0  # make the distance filter trivially pass
    cfg = {"n_worlds": 2, "n_agents": 3, "lidar_history_len": 3, "num_lidar_rays": 60}
    net = _make_net(cfg, dyn)
    B, N = cfg["n_worlds"] * cfg["n_agents"], cfg["n_agents"]
    H, R = cfg["lidar_history_len"], cfg["num_lidar_rays"]
    F = 2 + 6 + 5 + 4 * (N - 1) + H * R

    obs = torch.zeros(B, F)
    # peer block starts at 2 + 6 + 5 = 13
    peer_start = 13
    for p in range(N - 1):
        obs[..., peer_start + 4 * p + 0] = 0.1   # dx
        obs[..., peer_start + 4 * p + 1] = 0.1   # dy
        obs[..., peer_start + 4 * p + 2] = 1.0   # vx
        obs[..., peer_start + 4 * p + 3] = 1.0   # vy

    lidar, state, graph = net.prepare_obs(obs, num_agents=N)
    assert graph is not None
    peers, mask = graph
    # Every ego keeps every peer -> mask is all True, shape (B, N-1).
    assert mask.shape == (B, N - 1)
    assert mask.all()


def test_prepare_obs_normalization_applies_to_state(small_dynamics):
    # With v_limit>1, the normalized vel component should be divided by v_limit.
    dyn = dict(small_dynamics)
    dyn["v_limit"] = 4.0
    cfg = {"n_worlds": 1, "n_agents": 2, "lidar_history_len": 3, "num_lidar_rays": 60}
    net = _make_net(cfg, dyn)
    B, N = cfg["n_worlds"] * cfg["n_agents"], cfg["n_agents"]
    H, R = cfg["lidar_history_len"], cfg["num_lidar_rays"]
    F = 2 + 6 + 5 + 4 * (N - 1) + H * R

    obs = torch.zeros(B, F)
    # Put agent vel_x = 4.0 (state is at index 2..8, vel at 5..7)
    obs[..., 5] = 4.0  # vel_x
    obs[..., 6] = 0.0  # vel_y
    obs[..., 7] = 2.0  # ang_vel with omega_limit=1 -> stays 2.0
    # Pass nonzero lidar so normalize runs
    obs[..., 13 + 4 * (N - 1):] = 1.0

    _, state, _ = net.prepare_obs(obs, num_agents=N)
    # state_data layout is [goal_dist, goal_angle, vel_mag/v_limit, ang_vel/omega_limit, ...]
    # vel magnitude 4.0 / v_limit 4.0 = 1.0
    assert torch.allclose(state[..., 2], torch.ones(B), atol=1e-5)
    # ang_vel 2.0 / omega_limit 1.0 = 2.0
    assert torch.allclose(state[..., 3], torch.full((B,), 2.0), atol=1e-5)




def test_goal_polar_independent_of_velocity(small_obs_config, small_dynamics):
    # Regression: to_polar() was called with the agent velocity as its `epsilon`
    # argument, so the goal direction depended on velocity. goal_polar (the first
    # two state_data features) must be invariant to the agent's velocity.
    cfg = small_obs_config
    net = _make_net(cfg, small_dynamics)
    B, N = cfg["n_worlds"] * cfg["n_agents"], cfg["n_agents"]
    H, R = cfg["lidar_history_len"], cfg["num_lidar_rays"]
    F = 2 + 6 + 5 + 4 * (N - 1) + H * R
    torch.manual_seed(0)
    obs = torch.randn(B, F) * 0.3
    obs2 = obs.clone()
    obs2[..., 5:7] = obs[..., 5:7] + 3.0  # change ONLY velocity (state vel_xy at idx 5:7)
    _, sd1, _ = net.prepare_obs(obs, num_agents=N)
    _, sd2, _ = net.prepare_obs(obs2, num_agents=N)
    assert torch.allclose(sd1[..., 0:2], sd2[..., 0:2], atol=1e-6), "goal_polar leaks velocity"
    assert not torch.allclose(sd1[..., 2], sd2[..., 2], atol=1e-6), "vel magnitude should change"
