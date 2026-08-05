"""Fine-tuning helpers.

A fine-tune run loads a previously trained checkpoint and continues training
only a subset of the network — typically because the architecture grew a new,
randomly-initialised module (here: the corrected ``emb`` peer-attention layer
``node_attention_emb``) and we want to keep the already-learned feature
extractors fixed while the new piece (and the layers that consume it) adapt.

The freeze is expressed by *substring match on the parameter's qualified name*
rather than by holding module references, because the policy/critic store their
weights as functional ``TensorDictParams`` (see ``MultiAgentNetBase._make_params``)
and the torchrl loss module wraps them again — so the only stable handle we have
across all those layers is the dotted parameter name, which always contains the
original submodule name (e.g. ``...params.lidar_static_encoder.fc.weight``).
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

from torch import nn


def freeze_by_substring(
    module: nn.Module, substrings: Iterable[str]
) -> Tuple[int, List[nn.Parameter]]:
    """Freeze every parameter whose qualified name contains any of ``substrings``.

    Sets ``requires_grad=False`` on the matching parameters and returns
    ``(n_frozen, trainable)`` where ``n_frozen`` is the number of frozen
    parameter tensors and ``trainable`` is the list of the still-trainable
    parameters (suitable to hand straight to an optimizer).

    Raises ``ValueError`` if nothing matched — a finetune that silently freezes
    zero parameters (because the naming drifted) would train the whole net while
    pretending to fine-tune, so we fail loudly instead.
    """
    subs = list(substrings)
    n_frozen = 0
    for name, p in module.named_parameters():
        if any(s in name for s in subs):
            p.requires_grad_(False)
            n_frozen += 1
    if n_frozen == 0:
        raise ValueError(
            f"freeze_by_substring matched no parameters for {subs}; "
            "check the names against module.named_parameters()"
        )
    trainable = [p for p in module.parameters() if p.requires_grad]
    return n_frozen, trainable


def split_actor_critic_params(module):
    """Partition the *trainable* parameters of a torchrl loss module into
    ``(actor_params, critic_params)`` by whether ``"critic"`` appears in the
    qualified name.

    torchrl loss modules register weights under ``actor_network_params.*`` and
    ``critic_network_params.*`` (plus a detached ``target_critic_network_params``
    which has ``requires_grad=False``), so the substring cleanly separates the
    two. Frozen params (e.g. the finetune lidar encoders) and the target critic
    are skipped. Use to build an optimizer with an asymmetric actor/critic LR.
    """
    actor, critic = [], []
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        (critic if "critic" in name else actor).append(p)
    return actor, critic
