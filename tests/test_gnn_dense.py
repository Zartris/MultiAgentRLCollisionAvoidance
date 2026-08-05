"""AgentGraphNet.forward_dense must equal the sparse build + global_add_pool path.

The GNN peer aggregation was a sparse torch_geometric graph (argwhere/unique ->
device sync every forward). forward_dense replaces it with a masked sum, which is
mathematically identical (global_add_pool is a sum; per-node encoders have no
cross-node interaction). This test pins that equivalence. CPU, fast.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch as th

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

torch_geometric = pytest.importorskip("torch_geometric")

from models.MultiAgentLidarModel import (  # noqa: E402
    AgentGraphNet,
    convert_to_incremental,
)
from torch_geometric.nn import global_add_pool  # noqa: E402


def _sparse_reference(net, X, mask):
    """Replicate LocalNavigationGraphNetDist.forward's sparse path -> [B, out]."""
    b, _, _ = X.shape
    out_ch = net.node_encoder[0].out_features
    buf = th.zeros(b, out_ch)
    x = X[mask]
    if x.shape[0] == 0:
        return buf
    graph_batch = th.argwhere(mask)[:, 0]
    batch_inc = convert_to_incremental(graph_batch)
    global_bat = graph_batch.unique()
    buf[global_bat] = net(x, batch_inc)
    return buf


@pytest.mark.parametrize("seed", range(6))
def test_dense_equals_sparse(seed):
    th.manual_seed(seed)
    in_ch, out_ch = 4, 8
    net = AgentGraphNet(in_ch, out_ch).eval()
    b, p = 7, 5
    X = th.randn(b, p, in_ch)
    mask = th.rand(b, p) > 0.4
    with th.no_grad():
        dense = net.forward_dense(X, mask)
        sparse = _sparse_reference(net, X, mask)
    assert th.allclose(dense, sparse, atol=1e-5), \
        f"max|d|={(dense - sparse).abs().max().item()}"


def test_no_peers_gives_zero():
    th.manual_seed(0)
    net = AgentGraphNet(4, 8).eval()
    X = th.randn(3, 5, 4)
    mask = th.zeros(3, 5, dtype=th.bool)  # nobody qualifies
    with th.no_grad():
        dense = net.forward_dense(X, mask)
    assert th.count_nonzero(dense) == 0


def test_some_egos_empty():
    # mix: ego 0 has peers, ego 1 has none, ego 2 has peers
    th.manual_seed(1)
    net = AgentGraphNet(4, 8).eval()
    X = th.randn(3, 4, 4)
    mask = th.tensor([[True, False, True, False],
                      [False, False, False, False],
                      [True, True, False, False]])
    with th.no_grad():
        dense = net.forward_dense(X, mask)
        sparse = _sparse_reference(net, X, mask)
    assert th.allclose(dense, sparse, atol=1e-5)
    assert th.count_nonzero(dense[1]) == 0


# ---- working peer-attention modes (raw / emb) ---------------------------------
@pytest.mark.parametrize("mode", ["raw", "emb"])
def test_working_attention_is_live_and_masks(mode):
    th.manual_seed(0)
    net = AgentGraphNet(4, 8, attention_mode=mode).eval()
    X = th.randn(3, 4, 4)
    mask = th.tensor([[True, True, True, False],     # 3 peers
                      [True, False, False, False],    # exactly 1 peer -> weight 1
                      [False, False, False, False]])  # 0 peers -> zero
    with th.no_grad():
        out = net.forward_dense(X, mask)
        node_emb = net.node_encoder(X)
    # ego with no qualifying peers -> zero embedding
    assert th.count_nonzero(out[2]) == 0
    # ego with exactly one peer -> softmax weight is 1 -> output is that peer's emb
    assert th.allclose(out[1], node_emb[1, 0], atol=1e-5)
    # attention is actually live: perturbing its weights changes the multi-peer ego
    attn_mod = net.node_attention_emb if mode == "emb" else net.node_attention
    before = out[0].clone()
    attn_mod.lin.weight.data.add_(5.0)
    attn_mod.lin.bias.data.add_(2.0)
    with th.no_grad():
        after = net.forward_dense(X, mask)[0]
    assert not th.allclose(before, after, atol=1e-4), "attention had no effect"


def test_none_mode_attention_is_dead():
    # Contrast: in the legacy "none" mode the gate is a no-op, so perturbing the
    # attention weights must NOT change the output.
    th.manual_seed(0)
    net = AgentGraphNet(4, 8, attention_mode="none").eval()
    X = th.randn(3, 4, 4)
    mask = th.rand(3, 4) > 0.3
    with th.no_grad():
        before = net.forward_dense(X, mask)
    net.node_attention.lin.weight.data.add_(10.0)
    net.node_attention.lin.bias.data.add_(5.0)
    with th.no_grad():
        after = net.forward_dense(X, mask)
    assert th.allclose(before, after), "none-mode gate should be inert"
