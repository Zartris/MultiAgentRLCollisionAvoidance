"""Batched, retry-free entity placement via GPU repulsion relaxation.

vmas's ``ScenarioUtils.find_random_pos_for_entity`` places entities by rejection
sampling (pick a random point, reject if within ``min_dist`` of any placed
entity, retry). Retry count explodes as the world fills up, so dense scenarios
(20 agents + 20 goals + obstacles in a 10x10 world) stall.

This replaces it with **sample-then-relax**: sample every entity's position
uniformly at once (independently per world), then run a few vectorized
"push apart" iterations driven by a per-pair required-distance matrix. Because
the required distance is per-pair, heterogeneous sizes (big obstacles vs small
agents) each keep their own spacing — no single grid sized to the biggest
entity, and no retries. Random per-world initialisation gives a different layout
in every batch element.

Entities that should ignore each other (agents vs their goals) simply get a
required distance of 0 in the matrix.
"""
from __future__ import annotations

from typing import Optional

import torch as th
from torch import Tensor


def relax_positions(
    required_dist: Tensor,            # [N, N] per-pair min distance (0 on diagonal)
    x_bounds: tuple,
    y_bounds: tuple,
    batch_dim: int,
    device: th.device,
    fixed_positions: Optional[Tensor] = None,   # [B, F, 2] or [F, 2] immovable (walls)
    fixed_required: Optional[Tensor] = None,    # [N, F] required dist movable->fixed
    iters: int = 80,
    step: float = 0.5,
    tol: float = 1e-3,
    generator: Optional[th.Generator] = None,
) -> tuple[Tensor, Tensor]:
    """Return ``(positions [batch, N, 2], max_residual_overlap [batch])``.

    ``positions`` are uniformly initialised per world and relaxed so that every
    movable pair (i, j) ends up >= ``required_dist[i, j]`` apart.

    ``fixed_positions`` are immovable entities (e.g. walls, manually-placed
    obstacles): they push movable entities away but are NEVER moved. Long walls
    should be passed as several points sampled along their length, each with the
    appropriate ``fixed_required`` clearance. ``max_residual_overlap`` reports
    the worst remaining violation per world so the caller can spot an infeasible
    density.
    """
    n = required_dist.shape[0]
    if n == 0:  # nothing to place (e.g. a scenario with zero obstacles)
        return (
            th.empty((batch_dim, 0, 2), device=device),
            th.zeros(batch_dim, device=device),
        )
    x_lo, x_hi = float(x_bounds[0]), float(x_bounds[1])
    y_lo, y_hi = float(y_bounds[0]), float(y_bounds[1])
    lo = th.tensor([x_lo, y_lo], device=device)
    span = th.tensor([x_hi - x_lo, y_hi - y_lo], device=device)

    pos = lo + span * th.rand(
        (batch_dim, n, 2), device=device, generator=generator
    )
    R = required_dist.to(device).unsqueeze(0)  # [1, N, N]
    eye_mask = th.eye(n, dtype=th.bool, device=device).unsqueeze(0)  # [1, N, N]

    # Cap each per-iteration move at half the smallest required gap. Without this
    # an entity overlapping many neighbours at once can receive a push far larger
    # than the world, get flung to the boundary, clamp into a corner with another
    # entity, and stick there. Half a gap per step can't overshoot a neighbour.
    gaps = [float(required_dist[required_dist > 0].min())] if (required_dist > 0).any() else []
    if fixed_required is not None and (fixed_required > 0).any():
        gaps.append(float(fixed_required[fixed_required > 0].min()))
    max_move = 0.5 * min(gaps) if gaps else 0.5

    fixed = None
    Rf = None
    if fixed_positions is not None and fixed_positions.shape[-2] > 0:
        fixed = fixed_positions.to(device)
        if fixed.dim() == 2:
            fixed = fixed.unsqueeze(0).expand(batch_dim, -1, -1)
        Rf = fixed_required.to(device).unsqueeze(0)  # [1, N, F]

    overlap = None
    for _ in range(iters):
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)        # [B, N, N, 2]
        dist = diff.norm(dim=-1)                          # [B, N, N]
        overlap = (R - dist).clamp_min(0.0)              # how much too close
        overlap.diagonal(dim1=1, dim2=2).zero_()         # ignore self-pairs
        direction = diff / dist.clamp_min(1e-6).unsqueeze(-1)
        # Coincident entities (dist ~ 0, e.g. two clamped into the same corner)
        # have an undefined separation direction (diff/dist -> 0) and would stay
        # stuck forever. Give those pairs a random kick so they can separate.
        coincident = (dist < 1e-5) & ~eye_mask
        if bool(coincident.any()):
            kick = th.randn_like(direction)
            kick = kick / kick.norm(dim=-1, keepdim=True).clamp_min(1e-6)  # unit
            direction = th.where(coincident.unsqueeze(-1), kick, direction)
        push = (overlap.unsqueeze(-1) * direction).sum(dim=2) * step  # [B, N, 2]

        worst = float(overlap.max())
        if fixed is not None:
            fdiff = pos.unsqueeze(2) - fixed.unsqueeze(1)            # [B, N, F, 2]
            fdist = fdiff.norm(dim=-1)                               # [B, N, F]
            foverlap = (Rf - fdist).clamp_min(0.0)                   # [B, N, F]
            fdir = fdiff / fdist.clamp_min(1e-6).unsqueeze(-1)
            # AVERAGE (not sum) over the fixed points a movable overlaps: a wall
            # passed as many collinear points would otherwise sum into a huge
            # push and collapse the layout. The average gives the net "away from
            # the wall" direction with a bounded magnitude. The fixed point never
            # moves (it is an input, not part of `pos`).
            n_over = (foverlap > 0).sum(dim=2, keepdim=True).clamp_min(1)  # [B,N,1]
            fpush = (foverlap.unsqueeze(-1) * fdir).sum(dim=2) / n_over    # [B,N,2]
            push = push + fpush * step
            worst = max(worst, float(foverlap.max()))

        if worst < tol:
            break
        # Cap the per-iteration move (see max_move above) so a many-neighbour
        # overlap can't overshoot and fling entities into a corner.
        pnorm = push.norm(dim=-1, keepdim=True)
        push = push * (max_move / pnorm.clamp_min(1e-9)).clamp_max(1.0)
        pos = pos + push
        # Hard boundary clamp: agents must NEVER leave the world, or the global
        # planner (grid A*) gets a start outside the valid map and fails.
        pos = th.maximum(pos, lo)
        pos = th.minimum(pos, lo + span)

    # Residual on the FINAL positions (the in-loop overlap is one step stale).
    final_dist = (pos.unsqueeze(2) - pos.unsqueeze(1)).norm(dim=-1)
    final_ov = (R - final_dist).clamp_min(0.0)
    final_ov.diagonal(dim1=1, dim2=2).zero_()
    max_resid = final_ov.amax(dim=(1, 2))
    if fixed is not None:
        ffd = (pos.unsqueeze(2) - fixed.unsqueeze(1)).norm(dim=-1)
        max_resid = th.maximum(max_resid, (Rf - ffd).clamp_min(0.0).amax(dim=(1, 2)))
    return pos, max_resid


def build_required_dist(
    group_sizes: list[int],
    pair_dist: dict,
    device: th.device,
) -> Tensor:
    """Build the [N, N] required-distance matrix from per-group-pair distances.

    ``group_sizes[k]`` = number of entities in group k (concatenated in order).
    ``pair_dist[(a, b)]`` = required distance between groups a and b (symmetric;
    missing pairs default to 0, i.e. "ignore each other"). Diagonal is 0.
    """
    n = sum(group_sizes)
    R = th.zeros((n, n), device=device)
    # group id per entity index
    gid = th.cat([
        th.full((sz,), k, dtype=th.long) for k, sz in enumerate(group_sizes)
    ])
    for i_a in range(len(group_sizes)):
        for i_b in range(len(group_sizes)):
            d = pair_dist.get((i_a, i_b), pair_dist.get((i_b, i_a), 0.0))
            if d <= 0:
                continue
            mask = (gid.unsqueeze(0) == i_a) & (gid.unsqueeze(1) == i_b)
            R[mask.to(device)] = d
    R.fill_diagonal_(0.0)
    return R
