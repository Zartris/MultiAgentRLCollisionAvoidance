"""Adaptive curriculum for multi-scenario training.

A *curriculum* is an ordered list of *levels*. Each level is a set of config
overrides (scenario_type + sizing, or a full multi_config) the env is rebuilt
with. Training starts at level 0 and:

  - ADVANCES to the next level when ALL of `advance.conditions` hold for
    `advance.patience` consecutive evals, and
  - DEMOTES one level when ALL of `demote.conditions` hold for `demote.patience`
    consecutive evals.

A condition is ``{metric, op, threshold}`` where metric is one of the eval
metrics (success_rate / collision_rate / reward / stuck_rate) and op is one of
>= <= > < ==. Example advance gate: success_rate >= 0.7 AND collision_rate <= 0.05.

`min_iters_per_level` prevents switching before a level has had a fair chance.
A switch rebuilds the env; the policy/optimizer transfer (the GNN + agent-padding
are agent-count-agnostic).

Backward-compatible single-metric form is still accepted:
    advance: {metric: success_rate, threshold: 0.7, patience: 2}
is treated as one condition (op defaults to >= for advance, < for demote).

Pure decision logic — no torch/env dependency, unit-testable.
"""
from __future__ import annotations

from typing import Dict, List, Optional

_OPS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    "==": lambda a, b: a == b,
}


def _parse_conditions(spec: dict, default_op: str) -> List[dict]:
    """Build a list of {metric, op, threshold} from a gate spec.

    Accepts either the multi-condition form (`conditions: [...]`) or the legacy
    single-metric form (`metric`/`threshold`). Returns [] if nothing usable.
    """
    spec = spec or {}
    if spec.get("conditions"):
        out = []
        for c in spec["conditions"]:
            out.append({
                "metric": c["metric"],
                "op": c.get("op", default_op),
                "threshold": float(c["threshold"]),
            })
        return out
    if spec.get("metric") is not None and spec.get("threshold") is not None:
        return [{"metric": spec["metric"], "op": spec.get("op", default_op),
                 "threshold": float(spec["threshold"])}]
    return []


def _conditions_hold(conditions: List[dict], metrics: Dict[str, float]) -> bool:
    """True iff every condition is satisfiable and holds (AND)."""
    if not conditions:
        return False
    for c in conditions:
        val = metrics.get(c["metric"])
        if val is None:
            return False
        if not _OPS[c["op"]](val, c["threshold"]):
            return False
    return True


class CurriculumManager:
    def __init__(self, curriculum_cfg: dict):
        self.cfg = curriculum_cfg or {}
        self.levels: List[dict] = list(self.cfg.get("levels", []))
        if not self.levels:
            raise ValueError("curriculum.levels is empty — define at least one level")

        adv = self.cfg.get("advance", {}) or {}
        dem = self.cfg.get("demote", {}) or {}
        self.adv_conditions = _parse_conditions(adv, default_op=">=")
        self.adv_patience: int = int(adv.get("patience", 2))
        self.dem_conditions = _parse_conditions(dem, default_op="<")
        self.dem_patience: int = int(dem.get("patience", 1))
        self.min_iters_per_level: int = int(self.cfg.get("min_iters_per_level", 0))

        self.level_idx: int = int(self.cfg.get("start_level", 0))
        self.level_idx = max(0, min(self.level_idx, len(self.levels) - 1))
        self._adv_streak = 0
        self._dem_streak = 0
        self._iters_at_level = 0

    @property
    def current(self) -> dict:
        return self.levels[self.level_idx]

    @property
    def name(self) -> str:
        return self.current.get("name", f"level_{self.level_idx}")

    def level_overrides(self) -> dict:
        """cfg overrides for the current level (everything but 'name')."""
        return {k: v for k, v in self.current.items() if k != "name"}

    def describe_gates(self) -> str:
        def fmt(conds):
            return " AND ".join(f"{c['metric']} {c['op']} {c['threshold']}" for c in conds) or "(none)"
        return (f"advance: [{fmt(self.adv_conditions)}] x{self.adv_patience} consec; "
                f"demote: [{fmt(self.dem_conditions)}] x{self.dem_patience}; "
                f"min_iters_per_level={self.min_iters_per_level}")

    def note_iter(self):
        self._iters_at_level += 1

    def state_dict(self) -> dict:
        """Serializable curriculum progress for checkpoint/resume."""
        return {
            "level_idx": self.level_idx,
            "adv_streak": self._adv_streak,
            "dem_streak": self._dem_streak,
            "iters_at_level": self._iters_at_level,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore progress saved by state_dict(). Clamps level to a valid index so
        a checkpoint from a different curriculum length still loads safely."""
        if not state:
            return
        self.level_idx = max(0, min(int(state.get("level_idx", self.level_idx)),
                                    len(self.levels) - 1))
        self._adv_streak = int(state.get("adv_streak", 0))
        self._dem_streak = int(state.get("dem_streak", 0))
        self._iters_at_level = int(state.get("iters_at_level", 0))

    def update(self, eval_metrics: Dict[str, float]) -> Optional[str]:
        """Feed latest eval metrics; return 'advance' / 'demote' / None."""
        if self._iters_at_level < self.min_iters_per_level:
            return None

        # demote first (safety)
        if self.level_idx > 0 and self.dem_conditions:
            if _conditions_hold(self.dem_conditions, eval_metrics):
                self._dem_streak += 1
                if self._dem_streak >= self.dem_patience:
                    self._switch(self.level_idx - 1)
                    return "demote"
            else:
                self._dem_streak = 0

        # advance
        if self.level_idx < len(self.levels) - 1 and self.adv_conditions:
            if _conditions_hold(self.adv_conditions, eval_metrics):
                self._adv_streak += 1
                if self._adv_streak >= self.adv_patience:
                    self._switch(self.level_idx + 1)
                    return "advance"
            else:
                self._adv_streak = 0
        return None

    def _switch(self, new_idx: int):
        self.level_idx = max(0, min(new_idx, len(self.levels) - 1))
        self._adv_streak = 0
        self._dem_streak = 0
        self._iters_at_level = 0
