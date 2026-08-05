"""Tests for the planner's snap-to-free-cell robustness (batched A*).

Grid discretization + obstacle inflation can map a start/goal that is
geometrically clear into an occupied cell. The planner used to abort the whole
batch (return None) on that, causing reset-retry stalls (corridor). It now snaps
such endpoints to the nearest free cell. These tests pin the snap helper and its
corner cases so it can't silently regress. CPU-only (numba), fast.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from scenario.GlobalPlanner.batched_global_path_planner import (  # noqa: E402
    BatchAStar,
    snap_to_free,
)


# ---------- snap_to_free -----------------------------------------------------
def test_snap_free_point_unchanged():
    grid = np.zeros((20, 20), dtype=np.uint8)
    pts = np.array([[5, 5], [10, 12]], dtype=np.int64)
    out = snap_to_free(pts.copy(), grid)
    assert np.array_equal(out, pts)


def test_snap_point_in_obstacle_moves_to_free():
    grid = np.zeros((20, 20), dtype=np.uint8)
    grid[8:12, 8:12] = 1                       # obstacle block
    pts = np.array([[9, 9]], dtype=np.int64)   # inside the block (x=9,y=9)
    out = snap_to_free(pts.copy(), grid)
    x, y = int(out[0, 0]), int(out[0, 1])
    assert grid[y, x] == 0, "snapped cell is still occupied"


def test_snap_is_to_a_near_cell():
    grid = np.zeros((30, 30), dtype=np.uint8)
    grid[10:20, 10:20] = 1
    pts = np.array([[10, 15]], dtype=np.int64)   # on the left edge of the block
    out = snap_to_free(pts.copy(), grid)
    # nearest free is just outside the left edge (x=9) -> Chebyshev distance 1
    moved = np.abs(out[0] - pts[0]).max()
    assert moved <= 2, f"snapped too far ({moved})"
    assert grid[int(out[0, 1]), int(out[0, 0])] == 0


def test_snap_fully_blocked_no_crash():
    grid = np.ones((10, 10), dtype=np.uint8)     # nowhere free
    pts = np.array([[5, 5]], dtype=np.int64)
    out = snap_to_free(pts.copy(), grid)          # must return, not hang/crash
    assert out.shape == pts.shape


def test_snap_multiple_mixed():
    grid = np.zeros((40, 40), dtype=np.uint8)
    grid[20:25, 20:25] = 1
    pts = np.array([[2, 2], [22, 22], [30, 30]], dtype=np.int64)  # free, blocked, free
    out = snap_to_free(pts.copy(), grid)
    for k in range(pts.shape[0]):
        assert grid[int(out[k, 1]), int(out[k, 0])] == 0
    assert np.array_equal(out[0], pts[0]) and np.array_equal(out[2], pts[2])


def test_snap_out_of_bounds_input_no_crash():
    # A start/goal can map slightly outside the grid (agent at the world edge).
    # snap_to_free must clamp + snap rather than do an out-of-bounds grid read.
    grid = np.zeros((20, 20), dtype=np.uint8)
    grid[5:15, 5:15] = 1  # obstacle block; border is free
    pts = np.array([[25, 25], [-3, 10], [7, 7], [19, 30]], dtype=np.int64)
    out = snap_to_free(pts.copy(), grid)
    assert out.shape == pts.shape
    for x, y in out:
        assert 0 <= int(x) < 20 and 0 <= int(y) < 20
        assert grid[int(y), int(x)] == 0  # all snapped to free, no crash
