"""Fixed-rate follow controller: smoothed FP target -> rate-limited tool goals -> arm.

Runs in its own thread so the arm is commanded at a steady rate regardless of FP speed,
and so a stalled FP loop simply stops target updates (the controller then holds).
"""

import threading
import time
import traceback
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .grasp_goals import (ENCLOSE, HOVER, ManipulationConfig, check_workspace, joint_margin_problem, step_toward,
                    tool_goal)

HOLD, KILLED = "hold", "killed"


class TargetFilter:
    """EMA of FP object positions with jump rejection and staleness (monotonic seconds)."""

    def __init__(self, cfg: ManipulationConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.p: Optional[np.ndarray] = None
        self.t_ok: Optional[float] = None
        self.rejected = 0

    def update(self, p: np.ndarray, t: float) -> None:
        p = np.asarray(p, dtype=np.float64)
        if not np.all(np.isfinite(p)):
            return
        if self.p is not None and np.linalg.norm(p - self.p) > self.cfg.max_jump_m:
            self.rejected += 1
            return
        self.p = p if self.p is None else (1 - self.cfg.target_ema) * self.p + self.cfg.target_ema * p
        self.t_ok = t

    def current(self, now: float) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """(position, None) if usable, else (position or None, reason to hold)."""
        if self.p is None:
            return None, "no target yet"
        if now - self.t_ok > self.cfg.stale_s:
            why = "FP jumped" if self.rejected else "no FP update"
            return self.p, f"target lost ({why} for {now - self.t_ok:.1f}s); reselect if it persists"
        return self.p, None


@dataclass(frozen=True)
class Snapshot:
    mode: str
    status: str
    at_goal: bool
    target: Optional[np.ndarray]
    goal: Optional[np.ndarray]
    commanded: Optional[np.ndarray]
    measured: Optional[np.ndarray]
    joints: Optional[np.ndarray]  # measured arm joint positions (rad)


class FollowController:
    def __init__(self, arm, cfg: ManipulationConfig):
        self.arm, self.cfg = arm, cfg
        self.dt = 1.0 / cfg.control_hz
        # Re-entrant: the SIGINT handler calls kill() on the main thread, which may already hold it.
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._filter = TargetFilter(cfg)
        self._mode = HOLD
        self._status = "guide the arm by hand to a ready pose, then click an object"
        self._active: Optional[np.ndarray] = None
        self._goal = self._commanded = self._measured = self._joints = None
        self._at_goal = False
        self.step_from_measured = False
        self._thread = threading.Thread(target=self._run, name="follow-controller", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    # ----- called from the tracking loop / UI --------------------------------

    def set_target(self, p_base_object: np.ndarray) -> None:
        with self._lock:
            self._filter.update(p_base_object, time.monotonic())

    def new_target(self) -> None:
        """Forget the old object's track before a newly selected one starts feeding set_target."""
        with self._lock:
            self._filter.reset()
            self._active = self._goal = None

    def toggle_step_source(self) -> bool:
        """Switch between stepping from the last command and from the measured pose; True = measured."""
        with self._lock:
            self.step_from_measured = not self.step_from_measured
            return self.step_from_measured

    def request(self, mode: str) -> Optional[str]:
        """Switch mode; returns a refusal message if the arm cannot start moving."""
        with self._lock:
            if self._mode == KILLED:
                return "killed"
            if mode == ENCLOSE and self._mode != HOVER:
                self._status = "enclose only from hover"
                return self._status
            if self._mode == HOLD and mode != HOLD:
                refusal = self.arm.engage(self.cfg.min_start_joint_margin_deg)
                if refusal is not None:
                    self._status = refusal
                    return refusal
                self._commanded = None  # restart from the measured pose
            self._mode = mode
            self._at_goal = False
            self._status = {HOLD: "holding", HOVER: "moving to hover", ENCLOSE: "enclosing"}[mode]
            return None

    def kill(self, reason: str) -> None:
        with self._lock:
            self._mode, self._status = KILLED, f"KILLED ({reason}): gravity compensation"
        self.arm.kill(reason)

    @property
    def mode(self) -> str:
        return self._mode

    def snapshot(self) -> Snapshot:
        with self._lock:
            return Snapshot(self._mode, self._status, self._at_goal,
                            None if self._filter.p is None else self._filter.p.copy(),
                            self._goal, self._commanded, self._measured, self._joints)

    # ----- control thread ----------------------------------------------------

    def _run(self) -> None:
        next_tick = time.monotonic()
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as error:
                traceback.print_exc()
                self.kill(f"controller error: {error}")
            next_tick += self.dt
            delay = next_tick - time.monotonic()
            if delay < 0:
                next_tick, delay = time.monotonic(), 0.0
            self._stop.wait(delay)

    def _tick(self) -> None:
        state = self.arm.measured_state()
        measured = None if state is None else state.T_base_ee
        with self._lock:
            self._measured = measured
            self._joints = None if state is None else state.arm_positions
            if self._mode in (HOLD, KILLED):
                self._at_goal = False
                return
            near_limit = joint_margin_problem(self._joints, self.arm.joint_limits, self.cfg.min_joint_margin_deg)
            if near_limit is not None:
                self._at_goal = False
                self._status = f"holding: {near_limit}"
                return
            p, hold_reason = self._filter.current(time.monotonic())
            if hold_reason is not None:
                self._at_goal = False
                self._status = f"holding: {hold_reason}"
                return
            if self._active is None or np.linalg.norm(p - self._active) > self.cfg.deadband_m:
                self._active = p.copy()
            goal = tool_goal(self._active, HOVER if self._mode == HOVER else ENCLOSE, self.cfg)
            self._goal = goal
            bad = check_workspace(goal, self.cfg)
            if bad is not None:
                self._at_goal = False
                self._status = f"holding: goal outside workspace ({bad})"
                return
            if self._commanded is None:
                self._commanded = goal if measured is None else measured
            elif measured is not None:
                error = float(np.linalg.norm(measured[:3, 3] - self._commanded[:3, 3]))
                if error > self.cfg.max_tracking_error_m:
                    raise RuntimeError(f"tool is {error * 100:.1f} cm from its command (collision or stuck?)")
            start = measured if self.step_from_measured and measured is not None else self._commanded
            command, reached = step_toward(start, goal, self.dt, self.cfg)
            changed = not np.allclose(command, self._commanded, atol=1e-6)
            mode = self._mode
        if changed:
            self.arm.send_tool_pose(command, self.dt)
        with self._lock:
            if self._mode != mode:
                return
            self._commanded = command
            self._at_goal = reached
            remaining = float(np.linalg.norm(goal[:3, 3] - command[:3, 3]))
            label = "hover" if mode == HOVER else "enclose"
            self._status = f"at {label} goal, following" if reached else f"to {label}: {remaining * 100:.1f} cm left"
