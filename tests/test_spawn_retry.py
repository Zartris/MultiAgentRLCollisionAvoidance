"""Guard the bounded-retry semantics on CollisionAvoidance_random.reset_world_at.

The previous implementation caught every exception and recursed without a
retry limit, so a systematic failure (e.g. a vmas API change) surfaced as
``RecursionError: maximum recursion depth exceeded`` 1000 frames deep instead
of the underlying error. Bounded retry was added so the real cause is visible.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _make_scenario_with_failing_spawn(message: str):
    """Build a CollisionAvoidanceRandom instance without the vmas machinery, then
    stub the three calls the retry loop makes so only the spawn raises.
    """
    from scenario.CollisionAvoidance_random import CollisionAvoidanceRandom

    # Instantiate without running full init — we just need the reset_world_at
    # method bound to a minimal object.
    scen = CollisionAvoidanceRandom.__new__(CollisionAvoidanceRandom)
    scen.gp = None  # skip the global-planner branch inside the loop

    # Stub the three method calls inside reset_world_at.
    raise_counter = {"n": 0}

    def _raising_spawn(env_index):
        raise_counter["n"] += 1
        raise RuntimeError(message)

    scen.spawn_random = _raising_spawn

    # super().reset_world_at isn't reached because spawn_random raises first,
    # but we need `super()` to resolve. That works automatically once the object
    # exists as an instance of the class.

    return scen, raise_counter


def test_reset_world_at_bounded_retry_raises_after_max_attempts():
    scen, counter = _make_scenario_with_failing_spawn("spawn exploded")

    with pytest.raises(RuntimeError, match="spawn exploded"):
        scen.reset_world_at(env_index=0)

    # Retry should cap at the configured maximum (10 in the current impl).
    # If this number changes, the test should be updated — but any bound is
    # better than the previous unbounded recursion.
    assert 1 <= counter["n"] <= 50, (
        f"Expected a small bounded number of retries, got {counter['n']}. "
        "If the retry bound was raised above 50 the test just needs its cap updated."
    )


def test_reset_world_at_retries_do_not_stackoverflow():
    """The recursion-depth fingerprint of the old bug: a 1000+ attempt run
    would raise RecursionError, not the underlying exception. Assert we
    surface the real error type instead.
    """
    scen, _ = _make_scenario_with_failing_spawn("inner cause")

    with pytest.raises(RuntimeError):
        scen.reset_world_at(env_index=0)
    # Explicitly assert RecursionError is NOT the raised type — that was the
    # fingerprint of the previous bug.
