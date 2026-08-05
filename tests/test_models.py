"""Forward-pass shape contracts for each LocalNavigation* net.

These are the single-agent nets wrapped by MultiAgentLocalNavNet. They're responsible
for the flatten/unflatten pattern that lets the multi-agent wrapper treat (worlds,
agents) as a single batch axis.
"""
import pytest
import torch

from models.MultiAgentLidarModel import (
    LocalNavigationGraphNetDist,
    LocalNavigationNetDist,
    LocalNavigationNetDistV2,
)


def _cfg():
    return {
        "lidar_history_len": 3,
        "num_lidar_rays": 120,
        # state_input_dim = 4 (goal_polar(2) + vel_mag(1) + ang_vel(1)). The MLP
        # internally adds +3 for the global-path tail when use_global_path_obs=True.
        "state_input_dim": 4,
        "conv_channels": [32, 32],
        "kernel_sizes": [5, 2],
        "dynamics": {
            "omega_limit": 1.0,
            "v_limit": 1.0,
            "lidar_obs_range": 3.5,
            "agent_radius": 0.2,
        },
    }


def _feat_dim(cfg, n_agents, use_gp=True):
    return (
        2 + 6
        + (5 if use_gp else 0)
        + 4 * (n_agents - 1)
        + cfg["lidar_history_len"] * cfg["num_lidar_rays"]
    )


@pytest.mark.parametrize("use_gp", [True, False], ids=["gp", "no_gp"])
def test_local_nav_graph_dist_critic_output_shape(use_gp):
    # output_dim=1 triggers critic mode (value function). Covers both GP on/off so
    # a refactor that mis-handles combined_dim += 3 is caught on either branch.
    cfg = _cfg()
    net = LocalNavigationGraphNetDist(
        lidar_history_len=cfg["lidar_history_len"],
        lidar_num_rays=cfg["num_lidar_rays"],
        state_input_dim=cfg["state_input_dim"],
        conv_channels=cfg["conv_channels"],
        kernel_sizes=cfg["kernel_sizes"],
        gnn_emb_size=16,
        output_dim=1,
        dynamics=cfg["dynamics"],
        use_global_path_obs=use_gp,
    )
    W, N = 4, 3
    F = _feat_dim(cfg, N, use_gp=use_gp)
    obs = torch.randn(W, N, F) * 0.1
    out = net(obs)
    # Critic: (worlds, agents, 1). Unflatten must preserve the leading dims.
    assert out.shape == (W, N, 1)


@pytest.mark.parametrize("use_gp", [True, False], ids=["gp", "no_gp"])
def test_local_nav_graph_dist_actor_output_shape(use_gp):
    # output_dim=2 (actions) -> actor mode. Output = [v, w, scale_v, scale_w].
    # Parametrized over the GP branch for the same reason as the critic test above.
    cfg = _cfg()
    net = LocalNavigationGraphNetDist(
        lidar_history_len=cfg["lidar_history_len"],
        lidar_num_rays=cfg["num_lidar_rays"],
        state_input_dim=cfg["state_input_dim"],
        conv_channels=cfg["conv_channels"],
        kernel_sizes=cfg["kernel_sizes"],
        gnn_emb_size=16,
        output_dim=2,
        dynamics=cfg["dynamics"],
        use_global_path_obs=use_gp,
    )
    W, N = 4, 3
    F = _feat_dim(cfg, N, use_gp=use_gp)
    obs = torch.randn(W, N, F) * 0.1
    out = net(obs)
    assert out.shape == (W, N, 4)
    # linear action is passed through sigmoid -> [0, 1]
    assert (out[..., 0] >= 0).all() and (out[..., 0] <= 1).all()
    # angular action is through tanh -> [-1, 1]
    assert (out[..., 1] >= -1).all() and (out[..., 1] <= 1).all()
    # scales are softplus -> positive
    assert (out[..., 2:] > 0).all()


def test_local_nav_dist_actor_output_shape():
    # LocalNavigationNetDist (the older head without a separate scale_net) encodes
    # [mean, scale] jointly in its output_dim slots, so we pass 2*n_actions = 4 here.
    cfg = _cfg()
    net = LocalNavigationNetDist(
        lidar_history_len=cfg["lidar_history_len"],
        lidar_num_rays=cfg["num_lidar_rays"],
        state_input_dim=cfg["state_input_dim"],
        conv_channels=cfg["conv_channels"],
        kernel_sizes=cfg["kernel_sizes"],
        output_dim=4,
        dynamics=cfg["dynamics"],
        use_global_path_obs=True,
    )
    W, N = 3, 4
    F = _feat_dim(cfg, N)
    obs = torch.randn(W, N, F) * 0.1
    out = net(obs)
    # Output layout after split: linear(1) + angular(1) + scale(out//2 = 2) = 4
    assert out.shape == (W, N, 4)


def test_local_nav_dist_v2_actor_output_shape():
    cfg = _cfg()
    net = LocalNavigationNetDistV2(
        lidar_history_len=cfg["lidar_history_len"],
        lidar_num_rays=cfg["num_lidar_rays"],
        state_input_dim=cfg["state_input_dim"],
        conv_channels=cfg["conv_channels"],
        kernel_sizes=cfg["kernel_sizes"],
        output_dim=2,
        dynamics=cfg["dynamics"],
        use_global_path_obs=True,
    )
    W, N = 2, 5
    F = _feat_dim(cfg, N)
    obs = torch.randn(W, N, F) * 0.1
    out = net(obs)
    assert out.shape == (W, N, 4)


def test_flatten_unflatten_symmetry():
    # The forward's flatten-then-unflatten pattern must be a round trip on the
    # leading dims. Passing a 4D input (simulating time axis) should produce a 4D
    # output with the first three dims unchanged.
    cfg = _cfg()
    net = LocalNavigationGraphNetDist(
        lidar_history_len=cfg["lidar_history_len"],
        lidar_num_rays=cfg["num_lidar_rays"],
        state_input_dim=cfg["state_input_dim"],
        conv_channels=cfg["conv_channels"],
        kernel_sizes=cfg["kernel_sizes"],
        gnn_emb_size=16,
        output_dim=2,
        dynamics=cfg["dynamics"],
        use_global_path_obs=True,
    )
    W, T, N = 2, 3, 4
    F = _feat_dim(cfg, N)
    obs = torch.randn(W, T, N, F) * 0.1
    out = net(obs)
    assert out.shape == (W, T, N, 4)


def test_forward_is_deterministic_under_fixed_seed():
    # Shape-wise deterministic: running forward twice should not change output shape
    # even when peer mask changes between runs (no NaN/shape collapse).
    cfg = _cfg()
    net = LocalNavigationGraphNetDist(
        lidar_history_len=cfg["lidar_history_len"],
        lidar_num_rays=cfg["num_lidar_rays"],
        state_input_dim=cfg["state_input_dim"],
        conv_channels=cfg["conv_channels"],
        kernel_sizes=cfg["kernel_sizes"],
        gnn_emb_size=16,
        output_dim=2,
        dynamics=cfg["dynamics"],
        use_global_path_obs=True,
    ).eval()
    W, N = 2, 3
    F = _feat_dim(cfg, N)

    torch.manual_seed(0)
    obs = torch.randn(W, N, F) * 0.1
    with torch.no_grad():
        out_a = net(obs)
        out_b = net(obs)
    assert torch.allclose(out_a, out_b)


def test_forward_handles_all_quiet_peers():
    # Peers all stationary (zero velocity) -> GNN branch should fall back to zero-filled
    # embeddings. We intentionally leave agent_pos / goal nonzero so to_polar does not
    # divide by zero (the real env never gives us a perfectly co-located pair).
    cfg = _cfg()
    net = LocalNavigationGraphNetDist(
        lidar_history_len=cfg["lidar_history_len"],
        lidar_num_rays=cfg["num_lidar_rays"],
        state_input_dim=cfg["state_input_dim"],
        conv_channels=cfg["conv_channels"],
        kernel_sizes=cfg["kernel_sizes"],
        gnn_emb_size=16,
        output_dim=2,
        dynamics=cfg["dynamics"],
        use_global_path_obs=True,
    )
    W, N = 2, 3
    F = _feat_dim(cfg, N)
    obs = torch.randn(W, N, F) * 0.05
    # Zero the peer velocity slots so every peer is filtered out by the nonzero-vel mask.
    peer_start = 2 + 6 + 5
    for p in range(N - 1):
        obs[..., peer_start + 4 * p + 2] = 0.0  # vx
        obs[..., peer_start + 4 * p + 3] = 0.0  # vy
    out = net(obs)
    assert out.shape == (W, N, 4)
    assert torch.isfinite(out).all()


