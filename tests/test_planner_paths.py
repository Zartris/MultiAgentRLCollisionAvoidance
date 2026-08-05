"""Validate the batched A* planner on real captured inputs (golden set).

tests/data/planner_golden.npz holds real BatchAStar inputs (grid, starts, goals,
inflate radius) captured from random + corridor runs, including calls with real
obstacles. These tests assert the planner returns valid paths -- the property the
training loop depends on -- so the planner optimisation (compiled neighbour
expansion) and the snap-to-free fix can't silently regress. CPU/numba, ~seconds.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from scenario.GlobalPlanner.batched_global_path_planner import BatchAStar  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data", "planner_golden.npz")


def _load_calls():
    d = np.load(DATA)
    n = int(d["n"])
    return [
        dict(grid=d[f"grid_{i}"], s=d[f"s_{i}"], g=d[f"g_{i}"],
             infl=float(d[f"infl_{i}"]), tag=str(d[f"tag_{i}"]))
        for i in range(n)
    ]


CALLS = _load_calls()


def _solve(c):
    ba = BatchAStar(
        c["grid"].copy(), c["s"].astype(np.float64), c["g"].astype(np.float64),
        inflate_radius=c["infl"], heuristic_type="euclidean", verbose=False, draw=False,
    )
    inflated = ba.inflated_map.copy()          # 0 = free, 255 = obstacle (what A* navigates)
    paths, _ = ba.searching()
    return ba, inflated, paths


def test_golden_data_present():
    assert len(CALLS) > 0
    assert any(int((c["grid"] < 255).sum()) > 0 for c in CALLS), "expected calls with obstacles"


@pytest.mark.parametrize("i", range(len(CALLS)))
def test_golden_paths_are_valid(i):
    c = CALLS[i]
    ba, inflated, paths = _solve(c)
    batch = c["s"].shape[0]
    assert paths is not None, f"{c['tag']}[{i}] planner returned None"
    assert len(paths) == batch

    for k, p in enumerate(paths):
        p = np.asarray(p).astype(int)
        assert len(p) >= 1
        # endpoints connect the (snapped) start and goal
        assert np.array_equal(p[0], ba.s_start[k]), "path does not start at the start cell"
        assert np.array_equal(p[-1], ba.s_goal[k]), "path does not end at the goal cell"
        # contiguous: each step moves to an 8-adjacent cell
        if len(p) > 1:
            assert np.abs(np.diff(p, axis=0)).max() <= 1, "path has a non-adjacent jump"
        # never routes through an obstacle cell
        for x, y in p:
            assert inflated[y, x] == 0, f"{c['tag']}[{i}] path cell ({x},{y}) is in an obstacle"
