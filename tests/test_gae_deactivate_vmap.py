"""Proof that torchrl 0.11's GAE(deactivate_vmap=True) handles a multi-frame batch.

Before torchrl 0.11 (and before the work we contributed upstream), running GAE over
a rollout with per-step variable graph topology crashed under torch.vmap. The fix on
our side was to chunk GAE calls down to 1-frame batches (see
train/utils/common.py:compute_advantage_and_target, and the C1 comment in PPOTrainer).

torchrl 0.11 added a `deactivate_vmap=True` kwarg that skips the vmap path entirely.
This test verifies we can call GAE on a real multi-frame tensordict without vmap
erroring, which is the precondition for removing our chunking workaround.
"""
import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.objectives.value import GAE


class _TinyValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, obs):
        return self.fc(obs)


def test_gae_deactivate_vmap_handles_multi_frame_batch():
    value_net = TensorDictModule(
        _TinyValueNet(),
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "state_value")],
    )

    # B = frames per iteration in our trainer; use a batch large enough that the old
    # gae_batch_size=1 chunking would process it as many separate calls.
    B, N, F = 64, 3, 4

    gae = GAE(
        gamma=0.99,
        lmbda=0.95,
        value_network=value_net,
        deactivate_vmap=True,
    )
    gae.set_keys(
        advantage=("agents", "advantage"),
        value_target=("agents", "value_target"),
        value=("agents", "state_value"),
        reward=("agents", "reward"),
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )

    torch.manual_seed(0)
    obs = torch.randn(B, N, F)
    td = TensorDict(
        {
            ("agents", "observation"): obs,
            # Per-agent reward/done/terminated at each frame.
            ("agents", "reward"): torch.randn(B, N, 1),
            ("agents", "done"): torch.zeros(B, N, 1, dtype=torch.bool),
            ("agents", "terminated"): torch.zeros(B, N, 1, dtype=torch.bool),
            ("next", ("agents", "observation")): torch.randn(B, N, F),
            ("next", ("agents", "reward")): torch.randn(B, N, 1),
            ("next", ("agents", "done")): torch.zeros(B, N, 1, dtype=torch.bool),
            ("next", ("agents", "terminated")): torch.zeros(B, N, 1, dtype=torch.bool),
        },
        batch_size=[B],
    )

    out = gae(td)

    adv = out.get(("agents", "advantage"))
    vtarg = out.get(("agents", "value_target"))
    assert adv.shape == (B, N, 1)
    assert vtarg.shape == (B, N, 1)
    assert torch.isfinite(adv).all()
    assert torch.isfinite(vtarg).all()


def test_gae_make_value_estimator_accepts_deactivate_vmap():
    """Our PPOTrainer builds GAE via loss_module.make_value_estimator(...). Confirm the
    kwarg forwards through that path (otherwise LidarSingleStep.py's call site wouldn't
    actually reach the deactivated-vmap mode).
    """
    from torchrl.modules import ProbabilisticActor, IndependentNormal
    from torchrl.objectives import ClipPPOLoss, ValueEstimators

    obs_key = ("agents", "observation")
    action_key = ("agents", "action")

    class _PolicyHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(4, 4)

        def forward(self, obs):
            loc, scale = self.net(obs).chunk(2, -1)
            return loc, torch.nn.functional.softplus(scale) + 1e-3

    policy_module = TensorDictModule(
        _PolicyHead(),
        in_keys=[obs_key],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )
    policy = ProbabilisticActor(
        module=policy_module,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[action_key],
        distribution_class=IndependentNormal,
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
    )
    critic = TensorDictModule(
        _TinyValueNet(),
        in_keys=[obs_key],
        out_keys=[("agents", "state_value")],
    )

    loss = ClipPPOLoss(actor_network=policy, critic_network=critic, clip_epsilon=0.2)
    # The loss needs to know the nested value key before wiring the value estimator.
    loss.set_keys(value=("agents", "state_value"))
    # If make_value_estimator ignored the kwarg, the estimator would silently use vmap
    # and our chunking workaround would still be load-bearing. Verify the flag sticks.
    loss.make_value_estimator(ValueEstimators.GAE, gamma=0.99, lmbda=0.95, deactivate_vmap=True)
    assert loss.value_estimator.deactivate_vmap is True
