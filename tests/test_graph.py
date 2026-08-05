"""GNN plumbing: convert_to_incremental + AgentGraphNet forward shapes.

Both are in models/MultiAgentLidarModel.py. They are critical to the "batch abuse"
trick — if either changes behavior, the scatter-back to (B, gnn_emb_size) in
LocalNavigationGraphNetDist.forward silently misaligns.
"""
import torch

from models.MultiAgentLidarModel import AgentGraphNet, convert_to_incremental


def test_convert_to_incremental_preserves_groups():
    # Unique values get mapped to contiguous ints starting at 0, preserving order.
    x = torch.tensor([7, 7, 3, 3, 9, 9, 9])
    out = convert_to_incremental(x)
    # 3 -> 0, 7 -> 1, 9 -> 2 (sorted order)
    expected = torch.tensor([1, 1, 0, 0, 2, 2, 2])
    assert torch.equal(out, expected)


def test_convert_to_incremental_monotonic_sorted_input():
    x = torch.tensor([0, 0, 1, 2, 2, 4])
    out = convert_to_incremental(x)
    # 0->0, 1->1, 2->2, 4->3
    expected = torch.tensor([0, 0, 1, 2, 2, 3])
    assert torch.equal(out, expected)


def test_convert_to_incremental_empty():
    x = torch.tensor([], dtype=torch.long)
    out = convert_to_incremental(x)
    assert out.numel() == 0


def test_agent_graph_net_forward_shape():
    # Two graphs:
    #   graph 0: 3 nodes
    #   graph 1: 2 nodes
    # Each node has 4 features (dist, angle, vx, vy).
    net = AgentGraphNet(in_channels=4, out_channels=8)
    x = torch.randn(5, 4)
    batch = torch.tensor([0, 0, 0, 1, 1])
    out = net(x, batch)
    # AgentGraphNet uses global_add_pool -> one row per graph
    assert out.shape == (2, 8)


def test_agent_graph_net_single_graph():
    net = AgentGraphNet(in_channels=4, out_channels=6)
    x = torch.randn(4, 4)
    batch = torch.zeros(4, dtype=torch.long)
    out = net(x, batch)
    assert out.shape == (1, 6)


def test_agent_graph_net_permutation_invariant():
    # Graph nets should be node-order invariant when using sum pooling.
    torch.manual_seed(0)
    net = AgentGraphNet(in_channels=4, out_channels=4).eval()
    x = torch.randn(6, 4)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    with torch.no_grad():
        a = net(x, batch)
    # Permute nodes within each graph and expect the same sum-pooled output.
    perm = torch.tensor([2, 0, 1, 5, 4, 3])
    with torch.no_grad():
        b = net(x[perm], batch[perm])
    assert torch.allclose(a, b, atol=1e-5)
