"""Hard/corner-case tests for the heap-based batched A* (astar_search_heap_njit).

Unlike the legacy code path, the heap search is well-behaved on synthetic maps, so
we can pin its edge cases directly: reachability, no-path detection, optimal cost,
obstacle avoidance, path contiguity, start==goal, and mixed batches. CPU/numba.

Map encoding (input grid): 255 = free, <255 = obstacle. The planner inflates
obstacles and forces the border to occupied, so the search stays interior.
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

SQ2 = 1.4142135623730951


def _solve(grid, s, g, infl=1):
    ba = BatchAStar(
        grid.astype(np.float64), np.array(s, np.float64), np.array(g, np.float64),
        inflate_radius=infl, heuristic_type="euclidean", verbose=False, draw=False,
    )
    inflated = ba.inflated_map.copy()
    paths, _ = ba.searching()
    return ba, inflated, paths


def _free(h=40, w=40):
    return np.full((h, w), 255, dtype=np.uint8)


def _cost(p):
    p = np.asarray(p)
    c = 0.0
    for k in range(1, len(p)):
        dx = abs(int(p[k, 0] - p[k - 1, 0]))
        dy = abs(int(p[k, 1] - p[k - 1, 1]))
        c += min(dx, dy) * SQ2 + (max(dx, dy) - min(dx, dy))
    return c


def _assert_valid(p, start, goal, inflated):
    p = np.asarray(p).astype(int)
    assert np.array_equal(p[0], start), "path does not start at the start"
    assert np.array_equal(p[-1], goal), "path does not end at the goal"
    if len(p) > 1:
        assert np.abs(np.diff(p, axis=0)).max() <= 1, "non-adjacent jump"
    for x, y in p:
        assert inflated[y, x] == 0, f"path cell ({x},{y}) in obstacle"


def test_free_map_optimal_diagonal():
    ba, inflated, paths = _solve(_free(), [[5, 5]], [[35, 35]])
    assert paths is not None
    _assert_valid(paths[0], ba.s_start[0], ba.s_goal[0], inflated)
    # optimal octile distance is 30 diagonal steps
    assert abs(_cost(paths[0]) - 30 * SQ2) < 1e-6


def test_start_equals_goal():
    # extract_path always appends the parent, so start==goal yields a degenerate
    # path sitting on the start cell (same as the legacy planner).
    ba, inflated, paths = _solve(_free(), [[10, 10]], [[10, 10]])
    assert paths is not None
    p = np.asarray(paths[0]).astype(int)
    assert all(np.array_equal(cell, ba.s_start[0]) for cell in p)


def test_wall_with_gap_routes_through():
    grid = _free()
    grid[:, 20] = 0       # vertical wall
    grid[0:5, 20] = 255   # leave a gap near the top
    ba, inflated, paths = _solve(grid, [[5, 10]], [[35, 10]], infl=0)
    assert paths is not None
    _assert_valid(paths[0], ba.s_start[0], ba.s_goal[0], inflated)
    # must detour through the gap -> strictly longer than the straight-line octile
    straight = _cost([ba.s_start[0], ba.s_goal[0]])
    assert _cost(paths[0]) > straight


def test_disconnected_returns_none():
    grid = np.full((30, 30), 255, dtype=np.uint8)
    grid[15, :] = 0       # full horizontal wall, no gap
    _, _, paths = _solve(grid, [[5, 5]], [[5, 25]], infl=1)
    assert paths is None


def test_unreachable_agent_fails_whole_batch():
    # one reachable, one walled-off: the planner returns None for the batch
    grid = np.full((30, 30), 255, dtype=np.uint8)
    grid[15, :] = 0
    _, _, paths = _solve(grid, [[5, 5], [5, 5]], [[20, 5], [5, 25]], infl=1)
    assert paths is None


def test_all_reachable_batch_ok():
    ba, inflated, paths = _solve(_free(), [[5, 5], [6, 30]], [[35, 35], [30, 6]])
    assert paths is not None and len(paths) == 2
    for k in range(2):
        _assert_valid(paths[k], ba.s_start[k], ba.s_goal[k], inflated)


def test_obstacle_block_is_avoided():
    grid = _free(50, 50)
    grid[20:30, 20:30] = 0   # obstacle block
    ba, inflated, paths = _solve(grid, [[10, 25]], [[40, 25]], infl=1)
    assert paths is not None
    _assert_valid(paths[0], ba.s_start[0], ba.s_goal[0], inflated)


@pytest.mark.parametrize("seed", range(5))
def test_random_obstacles_paths_valid(seed):
    rng = np.random.RandomState(seed)
    grid = _free(60, 60)
    # scatter obstacle cells, keep a clear-ish band
    mask = rng.rand(60, 60) < 0.08
    grid[mask] = 0
    grid[5, 5] = 255
    grid[54, 54] = 255
    ba, inflated, paths = _solve(grid, [[5, 5]], [[54, 54]], infl=1)
    if paths is None:
        pytest.skip("randomly disconnected")
    _assert_valid(paths[0], ba.s_start[0], ba.s_goal[0], inflated)
