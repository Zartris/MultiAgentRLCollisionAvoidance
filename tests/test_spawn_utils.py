"""Tests for the batched relaxation spawner (scenario/spawn_utils.py).

These are the guarantees the scenarios rely on when they replaced vmas rejection
sampling: every placed pair respects its required min distance, nothing leaves
the bounds (or the grid planner fails), layouts differ per world, and entity
types that should ignore each other do. All run on CPU so they are fast and need
no GPU/vmas.
"""
from __future__ import annotations

import os
import sys

import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch as th  # noqa: E402

from scenario.spawn_utils import build_required_dist, relax_positions  # noqa: E402

DEV = th.device("cpu")


def _pairwise(p):  # correct per-pair distance: [B, N, N]
    return (p.unsqueeze(2) - p.unsqueeze(1)).norm(dim=-1)


# ---------- build_required_dist ----------------------------------------------
def test_build_required_dist_blocks_and_symmetry():
    R = build_required_dist([2, 3], {(0, 0): 1.0, (0, 1): 0.5, (1, 1): 0.7}, DEV)
    assert R.shape == (5, 5)
    assert th.allclose(R.diagonal(), th.zeros(5))            # no self-distance
    assert R[0, 1] == 1.0 and R[1, 0] == 1.0                 # group0-group0
    assert R[0, 2] == 0.5 and R[2, 0] == 0.5                 # group0-group1 symmetric
    assert R[3, 4] == 0.7                                     # group1-group1


def test_build_required_dist_missing_pair_is_zero():
    # group 1 vs group 2 not specified -> 0 (ignore each other)
    R = build_required_dist([1, 1, 1], {(0, 0): 1.0}, DEV)
    assert R[1, 2] == 0.0 and R[2, 1] == 0.0


# ---------- relax_positions: core guarantees ---------------------------------
def _obstacles_agents_goals(seed, batch=8, bound=4.5):
    th.manual_seed(seed)
    R = build_required_dist(
        [6, 20, 20],
        {(0, 0): 1.0, (0, 1): 0.95, (1, 1): 0.95, (0, 2): 1.0, (2, 2): 1.0},
        DEV,
    )
    pos, resid = relax_positions(R, (-bound, bound), (-bound, bound), batch, DEV, iters=200)
    return R, pos, resid


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 7])
def test_min_dist_respected(seed):
    R, pos, _ = _obstacles_agents_goals(seed)
    dist = _pairwise(pos)
    n = pos.shape[1]
    # ignore self-pairs; check every required pair is satisfied (small tolerance)
    big = dist + th.eye(n).unsqueeze(0) * 1e3
    violation = (R.unsqueeze(0) - big).clamp_min(0.0).max()
    assert float(violation) < 0.02, f"min-dist violated by {float(violation)}"


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 7])
def test_positions_in_bounds(seed):
    bound = 4.5
    _, pos, _ = _obstacles_agents_goals(seed, bound=bound)
    assert bool((pos.abs() <= bound + 1e-4).all()), "an entity left the bounds"


def test_per_world_diversity():
    _, pos, _ = _obstacles_agents_goals(seed=0, batch=16)
    # the same entity should land in different places across worlds
    assert float(pos[:, 0, 0].std()) > 0.1


def test_agent_goal_independent():
    # agents (group1) and goals (group2) have no required distance -> a goal may
    # sit arbitrarily close to an agent. Confirm we do NOT force them apart.
    th.manual_seed(0)
    R = build_required_dist([0, 10, 10], {(1, 1): 0.95, (2, 2): 0.95}, DEV)
    pos, _ = relax_positions(R, (-4.5, 4.5), (-4.5, 4.5), 8, DEV, iters=150)
    agents, goals = pos[:, :10], pos[:, 10:]
    cross = (agents.unsqueeze(2) - goals.unsqueeze(1)).norm(dim=-1)
    assert float(cross.min()) < 0.95  # at least one agent-goal pair is closer than 0.95


def test_dense_case_converges():
    # 40 agents + 40 goals + obstacles (eval density) should still resolve.
    th.manual_seed(0)
    R = build_required_dist(
        [6, 40, 40],
        {(0, 0): 1.0, (0, 1): 0.95, (1, 1): 0.95, (0, 2): 1.0, (2, 2): 1.0},
        DEV,
    )
    _, resid = relax_positions(R, (-4.5, 4.5), (-4.5, 4.5), 8, DEV, iters=300)
    assert float(resid.max()) < 0.05


# ---------- relax_positions: fixed obstacles (walls) -------------------------
def test_fixed_obstacles_avoided_and_unmoved():
    th.manual_seed(0)
    n_ag, clear, d = 20, 0.6, 0.95
    R = build_required_dist([n_ag], {(0, 0): d}, DEV)
    wy = th.linspace(-4, 4, 17)
    wall = th.stack([th.zeros_like(wy), wy], dim=-1)          # vertical wall of points
    wall_in = wall.clone()
    Rf = th.full((n_ag, wall.shape[0]), clear)
    pos, _ = relax_positions(
        R, (-4.5, 4.5), (-4.5, 4.5), 8, DEV,
        fixed_positions=wall, fixed_required=Rf, iters=400,
    )
    # agents keep their own spacing
    dd = _pairwise(pos) + th.eye(n_ag).unsqueeze(0) * 9
    assert float(dd.min()) > d - 0.05
    # agents clear the wall
    aw = (pos.unsqueeze(2) - wall.unsqueeze(0).unsqueeze(0)).norm(dim=-1)
    assert float(aw.min()) > clear - 0.05
    # the wall itself was never modified (it is an input, not part of the output)
    assert th.equal(wall, wall_in)


# ---------- robustness: the property scenarios depend on --------------------
def test_no_overlaps_across_many_seeds():
    bound = 4.5
    fails = 0
    for seed in range(25):
        R, pos, _ = _obstacles_agents_goals(seed, batch=12, bound=bound)
        dist = _pairwise(pos)
        n = pos.shape[1]
        big = dist + th.eye(n).unsqueeze(0) * 1e3
        if float((R.unsqueeze(0) - big).clamp_min(0.0).max()) > 0.05:
            fails += 1
    assert fails == 0, f"{fails}/25 seeds produced an overlap"


# ============================================================================
# HARD / CORNER CASES — these should trip if the spawner regresses.
# ============================================================================
def _no_nan(t):
    return not bool(th.isnan(t).any() or th.isinf(t).any())


def test_zero_entities_no_crash():
    R = build_required_dist([0], {}, DEV)
    assert R.shape == (0, 0)
    pos, resid = relax_positions(R, (-4.0, 4.0), (-4.0, 4.0), 4, DEV, iters=10)
    assert pos.shape == (4, 0, 2)


def test_single_entity_in_bounds():
    R = build_required_dist([1], {(0, 0): 1.0}, DEV)
    pos, resid = relax_positions(R, (-2.0, 2.0), (-3.0, 3.0), 8, DEV, iters=20)
    assert pos.shape == (8, 1, 2)
    assert bool((pos[..., 0].abs() <= 2.0 + 1e-4).all())
    assert bool((pos[..., 1].abs() <= 3.0 + 1e-4).all())
    assert float(resid.max()) == 0.0  # nothing to violate


def test_determinism_same_seed():
    R = build_required_dist([4, 8], {(0, 0): 1.0, (0, 1): 0.7, (1, 1): 0.6}, DEV)
    th.manual_seed(42)
    a, _ = relax_positions(R, (-4.0, 4.0), (-4.0, 4.0), 6, DEV, iters=100)
    th.manual_seed(42)
    b, _ = relax_positions(R, (-4.0, 4.0), (-4.0, 4.0), 6, DEV, iters=100)
    assert th.equal(a, b)


@pytest.mark.parametrize("seed", range(6))
def test_never_nan(seed):
    th.manual_seed(seed)
    R = build_required_dist([5, 15, 15], {(0, 0): 1.0, (0, 1): 0.9, (1, 1): 0.9,
                                          (0, 2): 1.0, (2, 2): 1.0}, DEV)
    pos, resid = relax_positions(R, (-4.5, 4.5), (-4.5, 4.5), 8, DEV, iters=200)
    assert _no_nan(pos) and _no_nan(resid)


def test_infeasible_density_no_hang_no_silent_overlap():
    # 60 agents needing 1.0 apart can't fit in a 2x2 box. The spawner must
    # return (not hang / not NaN) AND report a large residual so the caller can
    # tell the density is infeasible rather than getting silent overlaps.
    th.manual_seed(0)
    R = build_required_dist([60], {(0, 0): 1.0}, DEV)
    pos, resid = relax_positions(R, (-1.0, 1.0), (-1.0, 1.0), 4, DEV, iters=100)
    assert _no_nan(pos)
    assert bool((pos.abs() <= 1.0 + 1e-4).all())   # still clamped in bounds
    assert float(resid.max()) > 0.1                # flagged as infeasible


def test_thin_asymmetric_bounds():
    # corridor-like: a wide, thin zone. Agents must line up and stay inside.
    th.manual_seed(0)
    R = build_required_dist([8], {(0, 0): 0.6}, DEV)
    pos, _ = relax_positions(R, (-4.0, 4.0), (-0.4, 0.4), 8, DEV, iters=200)
    assert bool((pos[..., 0].abs() <= 4.0 + 1e-4).all())
    assert bool((pos[..., 1].abs() <= 0.4 + 1e-4).all())


def test_heterogeneous_sizes_each_keep_own_spacing():
    # big obstacles (need 2.0) + small agents (need 0.5); cross-pair needs 1.2.
    th.manual_seed(0)
    R = build_required_dist([4, 16], {(0, 0): 2.0, (0, 1): 1.2, (1, 1): 0.5}, DEV)
    pos, _ = relax_positions(R, (-6.0, 6.0), (-6.0, 6.0), 8, DEV, iters=300)
    d = _pairwise(pos)
    n = pos.shape[1]
    big = d + th.eye(n).unsqueeze(0) * 1e3
    assert float((R.unsqueeze(0) - big).clamp_min(0).max()) < 0.05


def test_fixed_obstacle_inside_zone_is_avoided():
    # THE corridor corner case: an obstacle sits inside the placement zone.
    # Movables must route around it (and never sit on it) while staying in bounds.
    th.manual_seed(0)
    n_ag, clear = 12, 1.0
    R = build_required_dist([n_ag], {(0, 0): 0.7}, DEV)
    obstacle = th.tensor([[0.0, 0.0]])                 # one fixed point at centre
    Rf = th.full((n_ag, 1), clear)
    pos, _ = relax_positions(R, (-3.0, 3.0), (-3.0, 3.0), 8, DEV,
                             fixed_positions=obstacle, fixed_required=Rf, iters=400)
    to_obs = (pos - obstacle.view(1, 1, 2)).norm(dim=-1)
    assert float(to_obs.min()) > clear - 0.05
    assert bool((pos.abs() <= 3.0 + 1e-4).all())


def test_fixed_clearance_impossible_no_crash():
    # clearance larger than the whole zone -> impossible; must not crash/NaN and
    # must stay in bounds (resid will be high, that's fine).
    th.manual_seed(0)
    R = build_required_dist([4], {(0, 0): 0.5}, DEV)
    obstacle = th.tensor([[0.0, 0.0]])
    Rf = th.full((4, 1), 100.0)                         # absurd clearance
    pos, resid = relax_positions(R, (-1.0, 1.0), (-1.0, 1.0), 4, DEV,
                                 fixed_positions=obstacle, fixed_required=Rf, iters=50)
    assert _no_nan(pos)
    assert bool((pos.abs() <= 1.0 + 1e-4).all())


def test_two_entities_just_fit_separate():
    # exactly enough room for two at distance 1.0 in [-0.6,0.6] (diag ~1.7).
    th.manual_seed(0)
    R = build_required_dist([2], {(0, 0): 1.0}, DEV)
    pos, _ = relax_positions(R, (-0.6, 0.6), (-0.6, 0.6), 16, DEV, iters=300)
    d = (pos[:, 0] - pos[:, 1]).norm(dim=-1)
    assert float(d.min()) > 1.0 - 0.05
    assert bool((pos.abs() <= 0.6 + 1e-4).all())


def test_fixed_wall_robust_many_seeds():
    n_ag, clear, d = 16, 0.6, 0.9
    R = build_required_dist([n_ag], {(0, 0): d}, DEV)
    wy = th.linspace(-4, 4, 17)
    wall = th.stack([th.zeros_like(wy), wy], dim=-1)
    Rf = th.full((n_ag, wall.shape[0]), clear)
    fails = 0
    for seed in range(20):
        th.manual_seed(seed)
        pos, _ = relax_positions(R, (-4.5, 4.5), (-4.5, 4.5), 8, DEV,
                                 fixed_positions=wall, fixed_required=Rf, iters=400)
        dd = _pairwise(pos) + th.eye(n_ag).unsqueeze(0) * 9
        aw = (pos.unsqueeze(2) - wall.unsqueeze(0).unsqueeze(0)).norm(dim=-1)
        if float(dd.min()) < d - 0.05 or float(aw.min()) < clear - 0.05:
            fails += 1
    assert fails == 0, f"{fails}/20 wall seeds failed"
