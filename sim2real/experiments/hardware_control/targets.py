"""What each test commands at an outer (20 Hz) update. Shared by sim and hardware.

``TargetSource.update(k, elapsed, T_current)`` is called once per outer index k (in
order, possibly skipping indices on hardware overruns). It returns a new absolute TCP
target, or None to keep holding the previous one. Targets are never regenerated between
outer updates.

  single     index 0 commands the one plan target; it is then held.
  scheduled  index k commands plan target k (fixed timestamps k * period, independent of
             progress); after the sequence the last target is held.
  relative   each update steps from the system's own current TCP toward the fixed goal,
             bounded by the step limit; the final step lands exactly on the goal. Once
             the measured TCP is within tolerance no new targets are made.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from geometry import bounded_step, pose_error, pose_from_json


class TargetSource:
    def __init__(self, plan: dict):
        self.plan = plan
        self.test = plan["test"]
        self.period = float(plan["outer_period_s"])
        self.targets = [pose_from_json(t["pose"]) for t in plan.get("targets", [])]
        self.reached_at: Optional[float] = None
        if self.test == "relative":
            rel = plan["relative"]
            self.goal = pose_from_json(rel["goal"])
            self.rel = rel
        elif self.test not in ("single", "scheduled"):
            raise ValueError(f"unknown test {self.test!r}")

    def update(self, k: int, elapsed: float, T_current) -> tuple[Optional[np.ndarray], dict]:
        if self.test == "single":
            return (self.targets[0], {"plan_index": 0}) if k == 0 else (None, {})
        if self.test == "scheduled":
            if k < len(self.targets):
                return self.targets[k], {"plan_index": k}
            return None, {}
        if self.reached_at is not None:
            return None, {}
        dp, dr = pose_error(T_current, self.goal)
        info = {"goal_position_error_m": float(np.linalg.norm(dp)), "goal_rotation_error_rad": float(np.linalg.norm(dr))}
        if (info["goal_position_error_m"] <= self.rel["position_tolerance_m"]
                and info["goal_rotation_error_rad"] <= self.rel["orientation_tolerance_rad"]):
            self.reached_at = elapsed
            info["reached"] = True
            return None, info
        step = bounded_step(T_current, self.goal, self.rel["max_translation_step_m"],
                            self.rel["max_rotation_step_rad"], self.rel["step_limit"])
        info.update(step_translation_m=step.translation.tolist(), step_rotvec_rad=step.rotvec.tolist(),
                    translation_limited=step.translation_limited, rotation_limited=step.rotation_limited,
                    final_step=step.final)
        return step.T_target, info

    def done(self, elapsed: float) -> Optional[str]:
        """Stop reason once the trial is complete, else None."""
        p = self.plan
        if self.test == "single":
            return "completed" if elapsed >= p["single"]["observe_s"] else None
        if self.test == "scheduled":
            end = len(self.targets) * self.period + p["scheduled"]["post_observe_s"]
            return "completed" if elapsed >= end else None
        if self.reached_at is not None and elapsed >= self.reached_at + self.rel["post_goal_observe_s"]:
            return "goal_reached"
        if elapsed >= self.rel["timeout_s"] and self.reached_at is None:
            return "timeout"
        return None
