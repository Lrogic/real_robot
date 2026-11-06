"""PolicyArm driven like the sim: the policy's TCP goal is held and the training IK tracks it.

send_tcp() only replaces the goal (process_actions). A thread then runs the sim's physics
substep at inner_hz: one get_robot_output(), one ik.DifferentialIK solve from the measured
arm joints toward the held goal (apply_actions), and set_arm_positions(q_target, goal_time,
non-blocking). goal_time 0 makes the driver apply the target immediately. The loop starts with
the first goal and stops for the ready move, a stop (hold_at) and kill; between goals and
during control holds it keeps tracking the last goal, as the driver does with a Cartesian
goal. A failure in the loop kills the arm (gravity compensation).

Deadlines are period apart with no catch-up (a tick more than a period late re-anchors).
The GIL switch interval is lowered while the loop runs so the tracking thread cannot hold it
for the default 5 ms.
"""

from __future__ import annotations

import sys
import threading
import time
import traceback
from typing import Optional

import numpy as np

from .arm import PolicyArm
from .ik import DifferentialIK


class SimIkPolicyArm(PolicyArm):
    MODE_NOTE = (PolicyArm.MODE_NOTE + "; TCP goals tracked by the training IK in a joint-position loop")

    def __init__(self, identity: dict, gripper_limit_margin_m: float, ik: DifferentialIK, inner_hz: float,
                 goal_time_s: float):
        super().__init__(identity, gripper_limit_margin_m)
        self.ik, self.inner_hz, self.inner_goal_time_s = ik, float(inner_hz), float(goal_time_s)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._stats_lock = threading.Lock()
        self._stats = self._new_stats()
        self.loop_error: Optional[str] = None

    def __enter__(self) -> "SimIkPolicyArm":
        super().__enter__()
        self.info.update(inner_hz=self.inner_hz, inner_goal_time_s=self.inner_goal_time_s)
        return self

    # ----- PolicyArm interface ----------------------------------------------------------

    def move_to_ready(self, joints, gripper: float, move_time_s: float) -> None:
        self._stop_loop()
        super().move_to_ready(joints, gripper, move_time_s)

    def hold_at(self, joints, goal_time_s: float) -> None:
        self._stop_loop()
        super().hold_at(joints, goal_time_s)

    def send_tcp(self, T_base_tcp: np.ndarray, goal_time_s: float) -> None:
        """Hold a new TCP goal; the first one starts the loop. goal_time_s is unused (inner_goal_time_s)."""
        with self.lock:
            self._check()
            if not self.position_mode:
                raise RuntimeError("the arm is not in position mode (move to the ready pose first)")
            self.ik.set_goal(T_base_tcp)
            if self._thread is None:
                self._start_loop()

    def kill(self, reason: str) -> bool:
        self._stop_loop()
        return super().kill(reason)

    def drain_stats(self) -> Optional[dict]:
        """Inner-loop statistics since the previous call; None while it has not ticked."""
        with self._stats_lock:
            s, self._stats = self._stats, self._new_stats()
        n = s["ticks"]
        if n == 0:
            return None
        span = s["last_at"] - s["first_at"]
        return {"ticks": n, "rate_hz": (n - 1) / span if n > 1 and span > 0 else None,
                "late_ticks": s["late"], "reanchored": s["reanchored"], "max_late_s": s["max_late"],
                "reused_state": s["reused"], "max_dq_clamped": s["clamped"], "max_solve_s": s["max_solve"],
                "max_send_s": s["max_send"], "last_q_target": s["q_target"], "last_dq": s["dq"]}

    # ----- loop ---------------------------------------------------------------------------

    @staticmethod
    def _new_stats() -> dict:
        return {"ticks": 0, "first_at": None, "last_at": None, "late": 0, "reanchored": 0, "max_late": 0.0,
                "reused": 0, "clamped": 0, "max_solve": 0.0, "max_send": 0.0, "q_target": None, "dq": None}

    def _start_loop(self) -> None:
        self._stop.clear()
        self._switch_interval = sys.getswitchinterval()
        sys.setswitchinterval(min(self._switch_interval, 0.25 / self.inner_hz))
        self._thread = threading.Thread(target=self._run, name="sim-ik-loop", daemon=True)
        self._thread.start()

    def _stop_loop(self) -> None:
        thread = self._thread
        if thread is None:
            return
        self._stop.set()
        if thread is not threading.current_thread():
            thread.join()
        self._thread = None
        sys.setswitchinterval(self._switch_interval)

    def _run(self) -> None:
        period = 1.0 / self.inner_hz
        max_dq = self.ik.params.max_dq
        due, last_id = time.monotonic(), None
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                late = now - due
                with self.lock:
                    if self._stop.is_set():
                        break
                    state = self.read()
                    t0 = time.perf_counter()
                    q_target, dq = self.ik.solve(state.arm_q)
                    t1 = time.perf_counter()
                    self.driver.set_arm_positions([float(v) for v in q_target], self.inner_goal_time_s, False)
                    t2 = time.perf_counter()
                reanchor = late > period
                with self._stats_lock:
                    s = self._stats
                    s["ticks"] += 1
                    s["first_at"] = s["first_at"] or now
                    s["last_at"] = now
                    s["late"] += late > 0.5 * period
                    s["reanchored"] += reanchor
                    s["max_late"] = max(s["max_late"], late)
                    s["reused"] += state.header_id == last_id
                    s["clamped"] += bool(np.any(np.abs(dq) >= max_dq - 1e-9))
                    s["max_solve"] = max(s["max_solve"], t1 - t0)
                    s["max_send"] = max(s["max_send"], t2 - t1)
                    s["q_target"], s["dq"] = q_target.tolist(), dq.tolist()
                last_id = state.header_id
                due = (now if reanchor else due) + period
                self._stop.wait(max(0.0, due - time.monotonic()))
        except Exception as error:
            traceback.print_exc()
            self.loop_error = repr(error)
            self.kill(f"sim IK loop: {error!r}")
