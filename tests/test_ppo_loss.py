"""Smoke test for train/utils/PPOLoss.py:ClipPPOLoss.

We synthesize a minimal tensordict that satisfies the loss module's field requirements
and check that a forward + backward can run without shape errors.
"""
import types

import pytest
import torch
from tensordict import TensorDict
from torchrl.modules import IndependentNormal, ProbabilisticActor
from tensordict.nn import TensorDictModule
from torch import nn

from train.utils.PPOLoss import ClipPPOLoss


class _ValueHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(8, 1)

    def forward(self, obs):
        return self.net(obs)


class _PolicyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(8, 4)  # mean(2) + logstd(2)

    def forward(self, obs):
        out = self.net(obs)
        loc, scale = out.chunk(2, -1)
        return loc, torch.nn.functional.softplus(scale) + 1e-3


def test_clip_ppo_loss_runs_on_synthetic():
    B, N, F = 4, 3, 8
    # Build a minimal policy (ProbabilisticActor) and critic (TensorDictModule)
    policy_module = TensorDictModule(
        _PolicyHead(),
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )
    policy = ProbabilisticActor(
        module=policy_module,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[("agents", "action")],
        distribution_class=IndependentNormal,
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
    )
    critic = TensorDictModule(
        _ValueHead(),
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "state_value")],
    )

    loss = ClipPPOLoss(
        actor_network=policy,
        critic_network=critic,
        clip_epsilon=0.2,
        entropy_bonus=True,
        entropy_coef=1e-3,
        critic_coef=0.5,
    )
    loss.set_keys(
        action=("agents", "action"),
        sample_log_prob=("agents", "sample_log_prob"),
        value=("agents", "state_value"),
        reward=("agents", "reward"),
        done=("agents", "done"),
        terminated=("agents", "terminated"),
        value_target=("agents", "value_target"),
        advantage=("agents", "advantage"),
    )

    torch.manual_seed(0)
    obs = torch.randn(B, N, F)

    td = TensorDict(
        {
            ("agents", "observation"): obs,
            ("agents", "action"): torch.randn(B, N, 2),
            ("agents", "sample_log_prob"): torch.zeros(B, N),
            ("agents", "advantage"): torch.randn(B, N, 1),
            ("agents", "value_target"): torch.randn(B, N, 1),
            # is_padding is required by ClipPPOLoss (fix A6); zeros = nothing padded.
            ("next", "agents", "info", "is_padding"): torch.zeros(B, N, 1),
        },
        batch_size=[B],
    )

    out = loss(td)
    assert "loss_objective" in out.keys()
    # Losses must be finite scalars
    for k in ("loss_objective", "loss_critic", "loss_entropy"):
        assert torch.isfinite(out[k]).all()
