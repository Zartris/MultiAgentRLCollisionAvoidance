"""Fine-tune wiring: loading an old (``none``-attention) checkpoint into the new
``emb`` architecture, and freezing only the lidar encoders.

These pin the two non-obvious behaviours the finetune path relies on:

1. The policy/critic store weights as functional ``TensorDictParams``. Loading a
   state-dict that lacks the new ``node_attention_emb`` keys must copy every
   matching tensor and leave the new attention layer at its random init (it does
   NOT raise, and it does NOT zero the new layer).
2. ``freeze_by_substring`` must freeze exactly the two lidar encoders (in every
   net it is given) and leave the GNN — including the new ``node_attention_emb``
   — and all heads trainable, with the optimizer seeing only the trainable set.

All CPU-only; no env / vmas / torchrl loss module needed.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from models.MultiAgentLidarModel import MultiAgentLocalNavNet  # noqa: E402
from train.utils.finetune import freeze_by_substring  # noqa: E402

# Same dims as model_loader.make_network / OurGraphModel.pth (rays=120, hist=3,
# gnn_emb=64, state_input=4, gp on).
_DYN = {"omega_limit": 1.0, "v_limit": 1.0, "lidar_obs_range": 3.0,
        "lidar_hist": 3, "lidar_rays": 120}
_LIDAR = ("lidar_static_encoder", "lidar_dynamic_encoder")


def _build(attention, seed):
    torch.manual_seed(seed)
    return MultiAgentLocalNavNet(
        base_net="oursGraph", lidar_history_len=3, lidar_input_dim=120,
        state_input_dim=4, conv_channels=[32, 32], kernel_sizes=[5, 2],
        gnn_emb_size=64, n_agent_outputs=2, n_agents=4, share_params=True,
        use_global_path_obs=True, set_gp_as_goal=False, dynamics=_DYN,
        device="cpu", gnn_attention=attention,
    )


# ---------- checkpoint load: none -> emb ------------------------------------
def test_load_none_into_emb_copies_matched_and_keeps_new_attention_at_init():
    src = _build("none", seed=1)          # legacy arch (no node_attention_emb)
    dst = _build("emb", seed=2)           # new arch (has node_attention_emb)
    emb_key = "params.agent_gnn.node_attention_emb.lin.weight"
    emb_init = dst.state_dict()[emb_key].clone()

    # Lenient TensorDictParams load: must not raise even though src lacks emb keys.
    dst.load_state_dict(src.state_dict(), strict=False)

    s, d = src.state_dict(), dst.state_dict()
    # every shared (tensor) key copied across from the checkpoint ...
    shared = [k for k in s if isinstance(s[k], torch.Tensor)]
    assert any("lidar_static_encoder" in k for k in shared)  # sanity: real keys present
    for k in shared:
        assert torch.equal(d[k], s[k]), f"{k} was not loaded from the checkpoint"
    # ... and the brand-new attention layer is untouched (still at its init).
    assert torch.equal(d[emb_key], emb_init), "node_attention_emb should stay at init"


def test_strict_true_load_is_lenient_for_superset_arch():
    # Documents the relied-on behaviour: loading the smaller (none) state dict into
    # the larger (emb) module does not raise, even with strict=True, because the
    # weights live in a TensorDictParams. (If a torch/torchrl upgrade changes this,
    # the finetune load path must switch to strict=False explicitly.)
    src, dst = _build("none", 1), _build("emb", 2)
    dst.load_state_dict(src.state_dict(), strict=True)  # must not raise


# ---------- freeze_by_substring ---------------------------------------------
def test_freeze_targets_only_lidar_encoders():
    net = _build("emb", seed=0)
    n_frozen, trainable = freeze_by_substring(net, _LIDAR)
    assert n_frozen == 12, f"expected 12 lidar params frozen, got {n_frozen}"

    frozen_names, train_names = [], []
    for name, p in net.named_parameters():
        (frozen_names if not p.requires_grad else train_names).append(name)

    # every frozen param is a lidar encoder; every lidar encoder param is frozen
    assert all(any(s in n for s in _LIDAR) for n in frozen_names)
    assert not any(any(s in n for s in _LIDAR) for n in train_names)
    # the new attention + the GNN + the heads must remain trainable
    for must_train in ("node_attention_emb", "node_encoder", "MLP",
                       "action_net", "scale_net"):
        assert any(must_train in n for n in train_names), f"{must_train} got frozen"


def test_optimizer_param_set_excludes_frozen():
    net = _build("emb", seed=0)
    _, trainable = freeze_by_substring(net, _LIDAR)
    opt = torch.optim.Adam(trainable, lr=1e-3)
    opt_ids = {id(p) for grp in opt.param_groups for p in grp["params"]}

    for name, p in net.named_parameters():
        if any(s in name for s in _LIDAR):
            assert id(p) not in opt_ids, f"frozen {name} leaked into the optimizer"
        else:
            assert id(p) in opt_ids, f"trainable {name} missing from the optimizer"


def test_freeze_raises_when_nothing_matches():
    net = _build("emb", seed=0)
    with pytest.raises(ValueError):
        freeze_by_substring(net, ["this_substring_matches_nothing"])


# ---------- split_actor_critic_params (asymmetric LR) ------------------------
def test_split_actor_critic_params():
    import torch.nn as nn
    from train.utils.finetune import split_actor_critic_params

    class LossLike(nn.Module):  # mimics torchrl's *_network_params naming
        def __init__(self):
            super().__init__()
            self.actor_network_params = nn.Linear(2, 2)
            self.critic_network_params = nn.Linear(2, 3)

    m = LossLike()
    actor, critic = split_actor_critic_params(m)
    assert len(actor) == 2 and len(critic) == 2          # weight + bias each side
    assert all("critic" not in n for n, p in m.named_parameters()
               if any(p is q for q in actor))

    # frozen params are excluded from both groups
    for p in m.actor_network_params.parameters():
        p.requires_grad_(False)
    actor2, critic2 = split_actor_critic_params(m)
    assert len(actor2) == 0 and len(critic2) == 2
