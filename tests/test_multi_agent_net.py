"""MultiAgentLocalNavNet: shared vs non-shared parameter modes.

This wrapper sits between torchrl and our LocalNavigation* nets. It decides whether
all agents share one policy (fast path: share_params=True) or each agent gets its own
vmapped copy. The shape contract on the outside stays the same.
"""
import pytest
import torch

from models.MultiAgentLidarModel import MultiAgentLocalNavNet


def _make(n_agents, share_params, n_outputs, base="oursGraph", use_global_path_obs=True):
    return MultiAgentLocalNavNet(
        base_net=base,
        lidar_history_len=3,
        lidar_input_dim=120,
        # 4 = goal_polar(2) + vel_mag(1) + ang_vel(1); the +3 GP block is added inside the net.
        state_input_dim=4,
        conv_channels=[32, 32],
        kernel_sizes=[5, 2],
        gnn_emb_size=16,
        n_agent_outputs=n_outputs,
        n_agents=n_agents,
        share_params=share_params,
        use_global_path_obs=use_global_path_obs,
        set_gp_as_goal=False,
        dynamics={
            "omega_limit": 1.0,
            "v_limit": 1.0,
            "lidar_obs_range": 3.5,
            "agent_radius": 0.2,
        },
    )


def _feat(n_agents, use_gp=True):
    return 2 + 6 + (5 if use_gp else 0) + 4 * (n_agents - 1) + 3 * 120


@pytest.mark.parametrize("use_gp", [True, False], ids=["gp", "no_gp"])
def test_multi_agent_shared_actor_forward_shape(use_gp):
    N = 4
    net = _make(n_agents=N, share_params=True, n_outputs=2, use_global_path_obs=use_gp)
    W = 3
    obs = torch.randn(W, N, _feat(N, use_gp=use_gp)) * 0.1
    out = net(obs)
    # actor output is [v, w, scale_v, scale_w] => last dim 4
    assert out.shape == (W, N, 4)


@pytest.mark.parametrize("use_gp", [True, False], ids=["gp", "no_gp"])
def test_multi_agent_shared_critic_forward_shape(use_gp):
    N = 4
    net = _make(n_agents=N, share_params=True, n_outputs=1, use_global_path_obs=use_gp)
    W = 3
    obs = torch.randn(W, N, _feat(N, use_gp=use_gp)) * 0.1
    out = net(obs)
    # critic output is scalar per agent
    assert out.shape == (W, N, 1)


def test_multi_agent_shared_accepts_4d_input():
    # torchrl rollouts can deliver (worlds, time, agents, F). The wrapper should
    # still round-trip through the single-agent net's flatten/unflatten.
    N = 3
    net = _make(n_agents=N, share_params=True, n_outputs=2)
    W, T = 2, 4
    obs = torch.randn(W, T, N, _feat(N)) * 0.1
    out = net(obs)
    assert out.shape == (W, T, N, 4)


@pytest.mark.parametrize("base,n_outputs", [("oursGraph", 2), ("oursD", 4)])
def test_multi_agent_build_each_base_net(base, n_outputs):
    # Each model family wrapped by MultiAgentLocalNavNet must construct without error.
    # oursGraph has separate action/scale heads (output_dim = n_actions = 2).
    # oursD encodes [mean, scale] jointly so output_dim = 2*n_actions = 4.
    # Both produce a (W, N, 4) output tensor via the internal split in the net.
    net = _make(n_agents=3, share_params=True, n_outputs=n_outputs, base=base)
    W, N = 2, 3
    obs = torch.randn(W, N, _feat(N)) * 0.1
    out = net(obs)
    assert out.shape == (W, N, 4)


