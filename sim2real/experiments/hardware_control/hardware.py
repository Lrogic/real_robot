"""Hardware trials: driver access, safety checks, the two control backends and the timed loop.

Driver access reuses ``manipulation.policy.arm.PolicyArm`` (the policy deployment's arm):
connect in gravity compensation, ``move_to_ready`` (arm and gripper to position mode, joint
move), ``hold_at`` (joint-space stop at a measured pose) and ``kill`` (arm and gripper to
gravity compensation, reconnecting with clear_error if the controller faulted). Every
driver call goes through its lock.

The trial loop is single-threaded. Each tick (deadline-scheduled, no catch-up) reads the
latest robot output once; a header id not seen before is a fresh state sample and is
logged. At each 20 Hz outer index the test's ``TargetSource`` gives a new absolute TCP
target or None (hold). Backends:

  trossen_cartesian      set_cartesian_positions(target, space, goal_time, blocking=False),
                         only when there is a new target. The loop polls state at poll_hz.
  external_training_ik   every tick (ik_hz): the training ClippedDifferentialIKAction solves
                         from the latest measured joints toward the held target
                         (ik_adapter.TrainingIK), then set_arm_positions(all six,
                         goal_time, blocking=False). Ticks whose state message was already
                         used are logged as reusing a sample.

Stops: a safety violation holds at the measured joints (hold_at, joint-space,
stop_goal_time_s); a driver/controller error or Ctrl+C kills to gravity compensation.
"""

from __future__ import annotations

import math
import traceback
from dataclasses import dataclass
from typing import Optional

import numpy as np

from geometry import make_T, pose_json, quat_wxyz_to_matrix, rotation_angle
from common.poses import rotvec_to_matrix
from targets import TargetSource
from timing import Deadlines, StateTracker, interval_stats

SPACES = ("joint", "cartesian")


def driver_interpolation(goal_time_s: float) -> str:
    """Interpolation the installed driver (trossen_arm 1.11.0 docstrings) picks for a goal time."""
    if goal_time_s < 0:
        raise ValueError("goal_time must be >= 0")
    if goal_time_s > 0.2:
        return "quintic"
    if goal_time_s > 0.001:
        return "linear"
    return "immediate"


@dataclass(frozen=True)
class StateSample:
    host_before: float          # monotonic, just before get_robot_output()
    host_after: float           # monotonic, just after it returned
    header_id: int
    header_ts_us: int
    q: np.ndarray               # arm joints, rad
    qd: np.ndarray              # rad/s
    gripper_q: float            # m
    driver_tcp: np.ndarray      # driver tool pose: m, angle-axis rad


class SafetyStop(Exception):
    pass


class HardwareRig:
    """PolicyArm plus the raw calls this experiment needs."""

    def __init__(self, identity: dict, gripper_limit_margin_m: float):
        from manipulation.policy.arm import PolicyArm

        self.arm = PolicyArm(identity, gripper_limit_margin_m)

    def __enter__(self) -> "HardwareRig":
        self.arm.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        self.arm.__exit__(*exc)

    @property
    def driver(self):
        return self.arm.driver

    def read(self, clock) -> StateSample:
        with self.arm.lock:
            t0 = clock.now()
            out = self.arm.driver.get_robot_output()
            t1 = clock.now()
        return StateSample(t0, t1, int(out.header.id), int(out.header.timestamp),
                           np.array(out.joint.arm.positions, dtype=np.float64),
                           np.array(out.joint.arm.velocities, dtype=np.float64),
                           float(out.joint.gripper.position),
                           np.array(out.cartesian.positions, dtype=np.float64))

    def send_cartesian(self, T, space: str, goal_time_s: float) -> None:
        import trossen_arm
        from manipulation.policy.actions import driver_pose6

        with self.arm.lock:
            self.arm._check()
            if not self.arm.position_mode:
                raise RuntimeError("arm is not in position mode")
            self.arm.driver.set_cartesian_positions(driver_pose6(T), getattr(trossen_arm.InterpolationSpace, space),
                                                    goal_time_s, False)

    def send_joints(self, q, goal_time_s: float) -> None:
        with self.arm.lock:
            self.arm._check()
            if not self.arm.position_mode:
                raise RuntimeError("arm is not in position mode")
            self.arm.driver.set_arm_positions([float(v) for v in q], goal_time_s, False)

    def move_to_ready(self, q, gripper: float, move_time_s: float) -> None:
        self.arm.move_to_ready(q, gripper, move_time_s)

    def stop_at(self, q, goal_time_s: float) -> None:
        self.arm.hold_at(q, goal_time_s)

    def kill(self, reason: str) -> None:
        self.arm.kill(reason)

    @property
    def killed(self) -> bool:
        return self.arm.killed

    def error_information(self) -> Optional[str]:
        try:
            with self.arm.lock:
                return str(self.arm.driver.get_error_information())
        except Exception as exc:  # the driver may be unusable at this point
            return f"unavailable: {exc!r}"

    def arm_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        with self.arm.lock:
            limits = self.arm.driver.get_joint_limits()[:-1]
        return (np.array([l.position_min for l in limits]), np.array([l.position_max for l in limits]))


def _pid(p) -> dict:
    return {"kp": float(p.kp), "ki": float(p.ki), "kd": float(p.kd), "imax": float(p.imax)}


def _link(link) -> dict:
    return {"mass_kg": float(link.mass), "inertia": [float(v) for v in link.inertia],
            "origin_xyz_m": [float(v) for v in link.origin_xyz], "origin_rpy_rad": [float(v) for v in link.origin_rpy]}


def hardware_configuration(driver) -> dict:
    """Everything readable about the controller configuration. Read-only."""
    from common.hardware import end_effector_record

    record, failed = {}, {}

    def read(name, fn):
        try:
            record[name] = fn()
        except Exception as exc:
            failed[name] = repr(exc)

    read("driver_version", driver.get_driver_version)
    read("controller_version", driver.get_controller_version)
    read("num_joints", driver.get_num_joints)
    read("modes", lambda: [m.name for m in driver.get_modes()])
    read("joint_limits", lambda: [{k: float(getattr(l, k)) for k in (
        "position_min", "position_max", "position_tolerance", "velocity_max", "velocity_tolerance",
        "effort_max", "effort_tolerance")} for l in driver.get_joint_limits()])
    read("motor_parameters", lambda: [{mode.name: {"position": _pid(p.position), "velocity": _pid(p.velocity)}
                                       for mode, p in joint.items()} for joint in driver.get_motor_parameters()])
    read("joint_characteristics", lambda: [{k: float(getattr(c, k)) for k in (
        "effort_correction", "friction_constant_term", "friction_coulomb_coef", "friction_transition_velocity",
        "friction_viscous_coef", "position_offset")} for c in driver.get_joint_characteristics()])
    read("end_effector", lambda: {**end_effector_record(driver.get_end_effector()),
                                  "palm": _link(driver.get_end_effector().palm),
                                  "finger_left": _link(driver.get_end_effector().finger_left),
                                  "finger_right": _link(driver.get_end_effector().finger_right)})
    read("algorithm_singularity_threshold", lambda: float(driver.get_algorithm_parameter().singularity_threshold))
    read("factory_reset_flag", driver.get_factory_reset_flag)
    record["unreadable"] = failed
    record["note"] = ("Read only; nothing is written. Motor PID gains are the controller's cascaded "
                      "position/velocity loops in its own units; they are not MuJoCo kp/kv.")
    return record


class Safety:
    def __init__(self, sc: dict, T_ref, joint_min, joint_max):
        self.sc = sc
        self.T_ref = np.asarray(T_ref, dtype=np.float64)
        self.lo = np.asarray(joint_min) + sc["joint_limit_margin_rad"]
        self.hi = np.asarray(joint_max) - sc["joint_limit_margin_rad"]
        self.box_lo = np.asarray(sc["workspace_min_m"], dtype=np.float64)
        self.box_hi = np.asarray(sc["workspace_max_m"], dtype=np.float64)
        self.slack = float(sc["measured_workspace_slack_m"])

    def target(self, T) -> Optional[str]:
        p = np.asarray(T)[:3, 3]
        if np.any(p < self.box_lo) or np.any(p > self.box_hi):
            return f"target TCP {np.round(p, 4).tolist()} m outside the workspace box"
        off = float(np.linalg.norm(p - self.T_ref[:3, 3]))
        if off > self.sc["max_target_offset_m"]:
            return f"target {off * 1000:.1f} mm from the reference TCP > {self.sc['max_target_offset_m'] * 1000:.0f} mm"
        ang = rotation_angle(self.T_ref[:3, :3], np.asarray(T)[:3, :3])
        if ang > self.sc["max_target_rotation_rad"]:
            return f"target rotated {ang:.3f} rad from the reference > {self.sc['max_target_rotation_rad']} rad"
        return None

    def joint_command(self, q_target, q_meas) -> Optional[str]:
        q_target = np.asarray(q_target)
        if not np.all(np.isfinite(q_target)):
            return f"non-finite joint target {q_target.tolist()}"
        if np.any(q_target < self.lo) or np.any(q_target > self.hi):
            return f"joint target {np.round(q_target, 4).tolist()} outside the driver limits minus margin"
        step = float(np.max(np.abs(q_target - np.asarray(q_meas))))
        if step > self.sc["max_joint_step_rad"]:
            return f"joint target {step:.3f} rad from the measured joints > {self.sc['max_joint_step_rad']} rad"
        return None

    def state(self, q, qd, T_fk, T_held) -> Optional[str]:
        if np.any(np.asarray(q) < self.lo) or np.any(np.asarray(q) > self.hi):
            return f"measured joints {np.round(q, 4).tolist()} within the limit margin"
        speed = float(np.max(np.abs(qd)))
        if speed > self.sc["max_joint_speed_rad_s"]:
            return f"joint speed {speed:.2f} rad/s > {self.sc['max_joint_speed_rad_s']}"
        p = np.asarray(T_fk)[:3, 3]
        if np.any(p < self.box_lo - self.slack) or np.any(p > self.box_hi + self.slack):
            return f"measured TCP {np.round(p, 4).tolist()} m outside the workspace box + slack"
        if T_held is not None:
            err = float(np.linalg.norm(p - np.asarray(T_held)[:3, 3]))
            if err > self.sc["max_tracking_error_m"]:
                return f"TCP {err * 1000:.1f} mm from the held target"
            ang = rotation_angle(np.asarray(T_held)[:3, :3], np.asarray(T_fk)[:3, :3])
            if ang > self.sc["max_tracking_error_rad"]:
                return f"TCP {ang:.3f} rad from the held target"
        return None


class CartesianBackend:
    name = "trossen_cartesian"

    def __init__(self, io, space: str, goal_time_s: float, poll_hz: float):
        if space not in SPACES:
            raise ValueError(f"interpolation space must be one of {SPACES}, got {space!r}")
        self.interpolation = driver_interpolation(goal_time_s)
        self.io, self.space, self.goal_time_s, self.loop_hz = io, space, float(goal_time_s), float(poll_hz)
        self.sends: list = []

    def condition(self) -> dict:
        return {"backend": self.name, "interpolation_space": self.space, "goal_time_s": self.goal_time_s,
                "driver_interpolation": self.interpolation, "blocking": False, "loop_hz": self.loop_hz,
                "feedforward": "driver defaults (none passed)", "num_trajectory_check_samples": "default (0)",
                "tag": f"cart_{self.space}_gt{self.goal_time_s:g}"}

    def on_target(self, T, clock, onset: float) -> dict:
        t0 = clock.now()
        self.io.send_cartesian(T, self.space, self.goal_time_s)
        t1 = clock.now()
        rec = {"call_t_s": t0 - onset, "return_t_s": t1 - onset}
        self.sends.append(rec)
        return rec

    def on_tick(self, sample, fresh: bool, clock, onset: float) -> None:
        return None

    def summary(self) -> dict:
        return {"commands_sent": len(self.sends),
                "send_call_duration_s": interval_stats(s["return_t_s"] - s["call_t_s"] for s in self.sends)}


class ExternalIKBackend:
    name = "external_training_ik"

    def __init__(self, io, ik, safety: Safety, goal_time_s: float, ik_hz: float):
        self.io, self.ik, self.safety = io, ik, safety
        self.goal_time_s, self.loop_hz = float(goal_time_s), float(ik_hz)
        self.interpolation = driver_interpolation(goal_time_s)
        self.rows: list = []
        self._last_used_id: Optional[int] = None
        self.target_updates = 0

    def condition(self) -> dict:
        return {"backend": self.name, "ik_hz": self.loop_hz, "goal_time_s": self.goal_time_s,
                "driver_interpolation": self.interpolation, "blocking": False,
                "command": "set_arm_positions(all six joint targets)",
                "tag": f"ik_{self.loop_hz:g}hz_gt{self.goal_time_s:g}"}

    def on_target(self, T, clock, onset: float) -> dict:
        self.ik.set_target(T)
        self.target_updates += 1
        return {"ik_target_set_t_s": clock.now() - onset}

    def on_tick(self, sample: StateSample, fresh: bool, clock, onset: float) -> None:
        reused = sample.header_id == self._last_used_id
        self._last_used_id = sample.header_id
        q_target, info = self.ik.solve(sample.q, sample.qd)
        reason = self.safety.joint_command(q_target, sample.q)
        if reason:
            raise SafetyStop(reason)
        t0 = clock.now()
        self.io.send_joints(q_target, self.goal_time_s)
        t1 = clock.now()
        self.rows.append((t0 - onset, t1 - t0, sample.header_id, reused, sample.q.copy(), q_target,
                          info["compute_s"], info["max_dq_clamped"]))

    def summary(self) -> dict:
        n = len(self.rows)
        return {"ik_updates": n, "commands_sent": n, "target_updates": self.target_updates,
                "reused_state_updates": int(sum(r[3] for r in self.rows)),
                "max_dq_clamped_updates": int(sum(r[7] for r in self.rows)),
                "ik_compute_s": interval_stats(r[6] for r in self.rows),
                "send_call_duration_s": interval_stats(r[1] for r in self.rows)}

    def log(self) -> dict:
        r = self.rows
        return {"send_t_s": [x[0] for x in r], "send_duration_s": [x[1] for x in r],
                "state_header_id": [x[2] for x in r], "state_reused": [x[3] for x in r],
                "arm_q_measured_rad": [x[4] for x in r], "arm_q_target_rad": [x[5] for x in r],
                "ik_compute_s": [x[6] for x in r], "max_dq_clamped": [x[7] for x in r]}


def _rate(times) -> Optional[float]:
    times = list(times)
    if len(times) < 2 or times[-1] <= times[0]:
        return None
    return (len(times) - 1) / (times[-1] - times[0])


def run_trial(io, backend, plan: dict, safety: Safety, fk, clock, *, stale_s: float, max_trial_s: float,
              stop_goal_time_s: float) -> dict:
    """One measured trial from the settled ready pose. Returns the trial record (not saved)."""
    source = TargetSource(plan)
    period = float(plan["outer_period_s"])
    tracker = StateTracker()
    fresh_rows, ticks, outer, errors = [], [], [], []
    stop, stop_action, interrupted = None, None, False

    first = io.read(clock)
    tracker.add(first.header_id, first.header_ts_us, first.host_after)
    latest, T_latest = first, fk.tcp(first.q)
    T_ref = make_T(np.asarray(plan["reference"]["tcp"]["rotation_matrix"]), plan["reference"]["tcp"]["position_m"])
    initial = {"arm_q_rad": first.q, "arm_qd_rad_s": first.qd, "gripper_q_m": first.gripper_q,
               "header_id": first.header_id, "tcp_fk": pose_json(T_latest), "driver_tcp": first.driver_tcp,
               "tcp_fk_offset_from_reference_m": T_latest[:3, 3] - T_ref[:3, 3],
               "tcp_fk_rotation_from_reference_rad": rotation_angle(T_ref[:3, :3], T_latest[:3, :3])}
    reason = safety.state(first.q, first.qd, T_latest, None)
    if reason:
        return {"aborted_before_onset": reason, "initial_state": initial, "stop_reason": f"safety: {reason}"}

    onset = clock.now()

    def keep(s: StateSample, T):
        fresh_rows.append((s.host_after - onset, s.host_before - onset, s.header_id, s.header_ts_us, s.q, s.qd,
                           s.gripper_q, s.driver_tcp, T[:3, 3].copy(), pose_json(T)["quaternion_wxyz"]))

    keep(first, T_latest)
    sched = Deadlines(1.0 / backend.loop_hz, onset)
    due, last_k, held = onset, -1, None
    try:
        while True:
            start = clock.now()
            elapsed = start - onset
            sample = io.read(clock)
            fresh = tracker.add(sample.header_id, sample.header_ts_us, sample.host_after)
            if fresh:
                latest, T_latest = sample, fk.tcp(sample.q)
                keep(sample, T_latest)
            ticks.append((sched.index, due - onset, elapsed, sample.header_id, fresh))
            stop = source.done(elapsed)
            if stop:
                break
            if elapsed > max_trial_s:
                raise SafetyStop(f"trial exceeded max_trial_s = {max_trial_s} s")
            k = int(math.floor(elapsed / period + 1e-9))
            if k > last_k:
                T_new, info = source.update(k, elapsed, T_latest)
                rec = {"index": k, "scheduled_t_s": k * period, "actual_t_s": elapsed,
                       "skipped_before": k - last_k - 1, "state_header_id": latest.header_id,
                       "state_fresh_this_tick": fresh, "new_target": T_new is not None,
                       "target": pose_json(T_new) if T_new is not None else None,
                       "current_tcp": pose_json(T_latest), "info": info}
                outer.append(rec)
                last_k = k
                if T_new is not None:
                    reason = safety.target(T_new)
                    if reason:
                        raise SafetyStop(reason)
                    held = T_new
                    rec["command"] = backend.on_target(T_new, clock, onset)
            backend.on_tick(sample, fresh, clock, onset)
            if fresh:
                reason = safety.state(sample.q, sample.qd, T_latest, held)
                if reason:
                    raise SafetyStop(reason)
            if clock.now() - tracker.last_fresh_host > stale_s:
                raise SafetyStop(f"no new state message for {stale_s * 1000:.0f} ms")
            _, due, _ = sched.advance(clock.now())
            clock.sleep_until(due)
    except SafetyStop as exc:
        stop = f"safety: {exc}"
        try:
            io.stop_at(latest.q, stop_goal_time_s)
            stop_action = f"hold_at measured joints over {stop_goal_time_s} s"
        except Exception as inner:
            errors.append({"during": "stop_at", "error": repr(inner), "traceback": traceback.format_exc()})
            io.kill("stop_at failed")
            stop_action = "killed (gravity compensation)"
    except KeyboardInterrupt:
        stop, interrupted = "keyboard_interrupt", True
        io.kill("keyboard interrupt")
        stop_action = "killed (gravity compensation)"
    except Exception as exc:
        stop = f"error: {exc!r}"
        errors.append({"during": "trial", "error": repr(exc), "traceback": traceback.format_exc(),
                       "controller_error_information": io.error_information()})
        io.kill(f"error: {exc!r}")
        stop_action = "killed (gravity compensation)"
    end = clock.now() - onset

    lateness = [t[2] - t[1] for t in ticks]
    tick_period = 1.0 / backend.loop_hz
    outer_late = [o["actual_t_s"] - o["scheduled_t_s"] for o in outer]
    drv_res = [float(np.linalg.norm(r[8] - r[7][:3])) for r in fresh_rows]
    drv_ang = [rotation_angle(rotvec_to_matrix(r[7][3:]), quat_wxyz_to_matrix(r[9])) for r in fresh_rows]
    timing = {
        "duration_s": end,
        "loop": {"requested_hz": backend.loop_hz, "ticks": len(ticks), "achieved_hz": len(ticks) / end if end > 0 else None,
                 "lateness_s": interval_stats(lateness),
                 "late_ticks_over_half_period": int(sum(l > 0.5 * tick_period for l in lateness)),
                 "skipped_slots": sched.skipped_total},
        "outer": {"requested_hz": 1.0 / period, "updates": len(outer), "achieved_hz": _rate(o["actual_t_s"] for o in outer),
                  "lateness_s": interval_stats(outer_late),
                  "skipped_indices": int(sum(o["skipped_before"] for o in outer)),
                  "new_targets": int(sum(o["new_target"] for o in outer))},
        "state": tracker.summary(end),
        "backend": backend.summary(),
    }
    timing["backend"]["achieved_command_hz"] = (
        _rate(r[0] for r in backend.rows) if hasattr(backend, "rows") else _rate(s["call_t_s"] for s in backend.sends))
    return {
        "initial_state": initial,
        "onset": {"definition": "monotonic host time of the first loop tick; the first target is sent "
                                "(Cartesian) or set and solved (external IK) in that tick",
                  "first_outer_command": outer[0].get("command") if outer else None},
        "samples": {
            "t_s": [r[0] for r in fresh_rows], "read_call_t_s": [r[1] for r in fresh_rows],
            "header_id": [r[2] for r in fresh_rows], "header_timestamp_us": [r[3] for r in fresh_rows],
            "arm_q_rad": [r[4] for r in fresh_rows], "arm_qd_rad_s": [r[5] for r in fresh_rows],
            "gripper_q_m": [r[6] for r in fresh_rows], "driver_tcp_xyz_rotvec": [r[7] for r in fresh_rows],
            "tcp_position_m": [r[8] for r in fresh_rows], "tcp_quat_wxyz": [r[9] for r in fresh_rows],
            "note": "fresh state messages only; t_s is the host time the read returned (first seen), "
                    "relative to onset; tcp is training-model FK of arm_q",
        },
        "ticks": {"index": [t[0] for t in ticks], "due_t_s": [t[1] for t in ticks], "start_t_s": [t[2] for t in ticks],
                  "header_id": [t[3] for t in ticks], "fresh": [t[4] for t in ticks]},
        "outer_updates": outer,
        "commands": backend.log() if hasattr(backend, "log") else {"cartesian_sends": backend.sends},
        "timing_summary": timing,
        "driver_tcp_vs_training_fk": {"position_m": interval_stats(drv_res), "rotation_rad": interval_stats(drv_ang)},
        "errors": errors, "stop_reason": stop, "stop_action": stop_action, "interrupted": interrupted,
    }


def reset_to_ready(io, ready_q, gripper: float, rc: dict, clock, confirm) -> dict:
    """Conservative joint move to the ready pose, then a settle check. Not part of a trial."""
    s = io.read(clock)
    dist = float(np.max(np.abs(s.q - np.asarray(ready_q))))
    if dist > rc["max_auto_start_distance_rad"]:
        confirm(f"The arm is {dist:.2f} rad (max joint) from the ready pose; "
                f"more than {rc['max_auto_start_distance_rad']} rad.")
    move_time = max(rc["min_move_time_s"], dist / rc["max_reset_speed_rad_s"])
    io.move_to_ready(ready_q, gripper, move_time)
    t_start = clock.now()
    clock.sleep_until(t_start + move_time)
    # Motion is judged by position drift over the hold window: single driver velocity samples
    # are noisy (~0.02 rad/s) on a stationary arm.
    window, record = [], None
    while True:
        now = clock.now()
        s = io.read(clock)
        err = float(np.max(np.abs(s.q - np.asarray(ready_q))))
        window = [w for w in window if now - w[0] <= rc["settle_hold_s"]] if err <= rc["joint_tolerance_rad"] else []
        window.append((now, s.q.copy(), err))
        span = now - window[0][0]
        drift = float(np.max(np.abs(s.q - window[0][1])) / span) if span > 0 else float("inf")
        record = {"start_distance_rad": dist, "move_time_s": move_time, "max_joint_error_rad": err,
                  "drift_speed_rad_s": drift, "sample_joint_speed_rad_s": float(np.max(np.abs(s.qd))),
                  "arm_q_rad": s.q, "gripper_q_m": s.gripper_q, "elapsed_s": now - t_start}
        if (err <= rc["joint_tolerance_rad"] and span >= 0.95 * rc["settle_hold_s"]
                and drift <= rc["velocity_tolerance_rad_s"]):
            record["settled"] = True
            return record
        if now - t_start > move_time + rc["settle_timeout_s"]:
            record["settled"] = False
            return record
        clock.sleep_until(now + 0.01)
