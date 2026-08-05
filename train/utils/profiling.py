"""Lightweight, opt-in wall-clock phase profiler.

Enabled only when env var ``GIANT_PROFILE`` is truthy, so it is a no-op in normal
runs. Times are accumulated per named phase at the *call boundary*, which is the
right granularity here because the hot planner is numba ``@njit`` (opaque to
cProfile/py-spy) — we can still measure how much wall time is spent inside it.

GPU work is async, so each phase boundary calls ``torch.cuda.synchronize()``
(unless ``GIANT_PROFILE_CUDA_SYNC=0``) to attribute GPU time honestly.

Usage::

    from train.utils.profiling import profile_phase, report_profile
    with profile_phase("collector.next"):
        td = collector.next()
    ...
    report_profile()   # prints the breakdown
"""
from __future__ import annotations

import atexit
import contextlib
import os
import time
from collections import defaultdict

_ENABLED = os.environ.get("GIANT_PROFILE", "").lower() not in ("", "0", "false", "no")
_SYNC = os.environ.get("GIANT_PROFILE_CUDA_SYNC", "1").lower() not in ("0", "false", "no")

_times: dict[str, float] = defaultdict(float)
_counts: dict[str, int] = defaultdict(int)
_reported = False


def enabled() -> bool:
    return _ENABLED


def _sync() -> None:
    if not _SYNC:
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


@contextlib.contextmanager
def profile_phase(name: str):
    if not _ENABLED:
        yield
        return
    _sync()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _sync()
        _times[name] += time.perf_counter() - t0
        _counts[name] += 1


def add_time(name: str, seconds: float, calls: int = 1) -> None:
    if not _ENABLED:
        return
    _times[name] += seconds
    _counts[name] += calls


_samples: dict[str, list] = defaultdict(list)


def record(name: str, value: float) -> None:
    """Record an individual sample for distribution/percentile reporting."""
    if _ENABLED:
        _samples[name].append(value)


def _percentiles(vals):
    s = sorted(vals)
    n = len(s)

    def pct(p):
        return s[min(n - 1, int(p / 100 * n))]

    return pct


def report_profile() -> None:
    global _reported
    if not _ENABLED or _reported or not _times:
        return
    _reported = True
    width = max((len(k) for k in _times), default=10)
    print("\n===== GIANT PROFILE (wall seconds, cuda-synced) =====")
    for name, t in sorted(_times.items(), key=lambda kv: -kv[1]):
        n = _counts[name]
        print(f"  {name:<{width}}  {t:9.2f}s   {n:>6} calls   {t / max(n, 1) * 1e3:8.1f} ms/call")
    print("=" * 52)
    for name, vals in _samples.items():
        if not vals:
            continue
        pct = _percentiles(vals)
        print(
            f"  dist[{name}] n={len(vals)}  "
            f"min={min(vals):.2f} p50={pct(50):.2f} p90={pct(90):.2f} "
            f"p99={pct(99):.2f} max={max(vals):.2f}  sum={sum(vals):.1f}"
        )
    print("=" * 52 + "\n")


# Print whatever we have at process exit, even if the run errors out.
atexit.register(report_profile)
