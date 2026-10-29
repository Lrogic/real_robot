"""Hooks that run the jenga policy inside pipeline.tracking_loop.track_live (--manipulation policy).

Every tracking-loop iteration (after FoundationPose and SAM recovery) updates the pose trust.
At each policy tick one robot-state read and the tracked object's original-mesh pose in the
base frame, T_base_object = T_base_camera @ center_pose @ to_origin, build the observation;
the actor turns it into a TCP goal (actions.ee_delta_target), and the controller decides
whether the driver gets it (controller.py has the states). control.mode in the config, or
--policy-control, picks shadow / step / slow / full.

In the tracking window:
    s        move to the ready pose and arm an episode (shadow: arm it in place);
             while commanding: stop at the current pose and end the episode
    r        start sending commands; after a tracking hold, resume once the pose is trusted
    n        step mode: send one policy step
    k        KILL: arm and gripper to gravity compensation until restart
    q / Esc  quit (the arm goes to gravity compensation, then idle after the prompt)
"""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from calibration.session import find_calibration
from calibration.transforms import inv_T, load_T_base_camera

from ..common.drawing import draw_point_marker, draw_tool, fmt_vec
from . import controller as ctl
from . import observations as obs
from .actions import EeTarget, ee_delta_target, in_workspace
from .actor import PolicyActor
from .arm import DryRunPolicyArm, PolicyArm, ready_error
from .jenga.config import JengaPolicyConfig
from .jenga.observation_manager import ACTIVE, IDLE, JengaObservationManager
from .policy_log import LOG_FILE, PolicyLog
from .robot_state import resolve_robot_identity
from .symmetry import BOX_SYMMETRY_NAMES
from .trust import Trust, TrustTracker

TCP_COLOR, GOAL_COLOR, MESSAGE_COLOR = (0, 255, 0), (255, 0, 255), (255, 220, 0)
TARGET_COLOR = (0, 255, 255)
TRUST_COLORS = {Trust.VERIFIED: (0, 255, 0), Trust.UNVERIFIED_OK: (235, 235, 235), Trust.UNTRUSTED: (255, 60, 60)}
STATE_COLORS = {ctl.RUNNING: (0, 255, 0), ctl.HOLD_STALE: (255, 220, 0), ctl.HOLD_TRACKING: (255, 60, 60),
                ctl.KILLED: (255, 60, 60)}
AXIS_COLORS = ((255, 0, 0), (0, 255, 0), (0, 0, 255))
STRIP_LINE = 18
PRINTED_EVENTS = {"episode_armed", "episode_start", "episode_end", "episode_cancelled", "reregistration",
                  "recovery_disabled", "episode_length_exceeded", "robot_error", "tracking_resumed",
                  "control", "command_error", "kill"}


class TickSchedule:
    """Deadlines period apart. A late tick keeps the phase; one more than a period late re-anchors."""

    def __init__(self, hz: float):
        self.period = 1.0 / hz
        self.next_due: Optional[float] = None

    def due(self, now: float):
        """(due, overrun) for a loop iteration at monotonic time now."""
        if self.next_due is None:
            self.next_due = now
        if now < self.next_due:
            return False, False
        overrun = now - self.next_due > self.period
        self.next_due = now + self.period if overrun else self.next_due + self.period
        return True, overrun


def _file_digest(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _pose7(T) -> Optional[list]:
    return None if T is None else obs.pose7(T).tolist()


class PolicySession:
    def __init__(self, cfg: JengaPolicyConfig, T_base_camera: np.ndarray, arm, actor: PolicyActor, recovery,
                 log: PolicyLog, save_path: Path, stats_every_s: float):
        self.cfg = cfg
        self.X, self.X_inv = T_base_camera, inv_T(T_base_camera)
        self.arm, self.actor, self.recovery, self.log = arm, actor, recovery, log
        self.save_path = save_path
        self.stats_every_s = stats_every_s
        self.manager = JengaObservationManager(cfg)
        self.trust = TrustTracker()
        self.control = ctl.Controller.from_config(cfg)
        self.schedule = TickSchedule(1.0 / self.control.tick_period_s)
        self.keys = {"episode": ord(cfg.episode_key), "ready": ord(cfg.control.ready_key),
                     "kill": ord(cfg.control.kill_key), "step": ord(cfg.control.step_key)}
        self.message = ""
        self._index: Optional[int] = None
        self._rec_index: Optional[int] = None
        self._object_name: Optional[str] = None
        self._K = None
        self._result = None
        self._robot = None
        self._target: Optional[EeTarget] = None
        self._raw_action: Optional[np.ndarray] = None
        self._decision: Optional[ctl.Decision] = None
        self._ready_error = None
        self._last_robot_error: Optional[str] = None
        self._robot_errors = 0
        self._last_update_at: Optional[float] = None
        self._last_captured_at: Optional[float] = None
        self._frame_gap: Optional[float] = None
        self._last_print_at = time.monotonic()
        self._closed = False

    # ----- lifecycle --------------------------------------------------------------------

    def start(self, info: dict) -> None:
        self.log.event("start", **info)
        print(__doc__)
        c = self.cfg.control
        print(f"Policy {self.actor.info.get('experiment')} iteration {self.actor.info.get('iteration')}; "
              f"control mode {c.mode.upper()}, policy ticks at {1.0 / self.control.tick_period_s:g} Hz, "
              f"goals over {1000 * self.control.goal_time_s:.0f} ms. Logging: "
              f"{self.log.path if self.log.enabled else 'off (--manipulation-logging)'}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.manager.phase != IDLE:
            self._emit(self.manager.end_episode("tracking stopped"))
        print(f"[policy overall] {self.log.stats_line()}")
        self.log.event("stop", ticks=self.log.ticks, overall_rate_hz=self.log.overall_rate(),
                       overruns=self.log.overruns, control_state=self.control.state)
        self.log.close()

    # ----- hooks called by track_live ---------------------------------------------------

    def on_mouse(self, *_args) -> None:
        pass

    def update(self, tracker, K: np.ndarray, timing) -> None:
        now = time.monotonic()
        update_interval = None if self._last_update_at is None else now - self._last_update_at
        self._last_update_at = now
        if timing.captured_at:
            if self._last_captured_at is not None:
                self._frame_gap = timing.captured_at - self._last_captured_at
            self._last_captured_at = timing.captured_at
        self._K = K
        if self._index is None:
            self._resolve_object(tracker)
        tracked = tracker.objects[self._index]
        T_raw = None if tracked.center_pose is None else self.X @ tracked.center_pose @ tracked.to_origin

        trust, events = self.trust.observe(self.recovery, self._rec_index, timing.frame_number)
        for event in events:
            self._emit(event, frame_number=timing.frame_number)
        if trust is Trust.VERIFIED and T_raw is not None:
            self.manager.note_verified(T_raw, timing.frame_number)

        due, overrun = self.schedule.due(now)
        if not due:
            return
        tick_at = time.monotonic()
        frame_age = tick_at - timing.captured_at if timing.captured_at else None
        robot, robot_read_s = None, None
        try:
            robot = self.arm.read()
            robot_read_s = None if robot is None else robot.read_at - tick_at
        except Exception as error:  # a failed read must not end the tracking loop
            self._robot_error(error, timing.frame_number)
        self._check_ready(robot)
        result = self.manager.tick(robot, T_raw, trust, timing.frame_number)
        for event in result.events:
            self._emit(event, frame_number=timing.frame_number)
        if result.phase == ACTIVE and self.control.state == ctl.AWAIT_GO and not self.message:
            self.message = f"episode active: '{self.cfg.control.ready_key}' starts the policy"

        target, raw, action_error = None, None, None
        if result.vector is not None and robot is not None:
            raw = self.actor(result.vector)
            try:
                target = self._target_for(robot, raw)
            except ValueError as error:
                action_error = str(error)
        elif result.phase == ACTIVE:
            action_error = "no robot state" if robot is None else "no observation (object pose missing)"
        decision = self.control.tick(ctl.Gate(
            trust=trust, frame_age_s=frame_age, frame_gap_s=self._frame_gap, action_error=action_error,
            in_workspace=None if target is None else in_workspace(
                target.T_base_target[:3, 3], self.cfg.workspace_min, self.cfg.workspace_max)))
        sent, gripper_sent, send_error = False, False, None
        if decision.send:
            try:
                self.arm.send_tcp(target.T_base_target, self.control.goal_time_s)
                gripper_sent = self.arm.set_gripper(target.gripper, self.cfg.control.gripper_goal_time_s)
                sent = True
            except Exception as error:
                send_error = repr(error)
                self.control.report_send_error(send_error)
                self._emit({"event": "command_error", "error": send_error}, frame_number=timing.frame_number)
        self._emit_control(timing.frame_number)
        build_s = time.monotonic() - tick_at
        self._result, self._robot, self._target, self._raw_action, self._decision = \
            result, robot, target, raw, decision

        choice = result.choice
        record = {
            "frame_number": timing.frame_number, "camera_timestamp_ms": timing.camera_timestamp_ms,
            "update_interval_s": update_interval, "frame_gap_s": self._frame_gap, "robot_read_s": robot_read_s,
            "phase": result.phase, "episode": self.manager.episode if result.phase == ACTIVE else None,
            "step": result.step, "object": self._object_name,
            "trust": result.trust, "last_sam_match_frame": self.trust.last_match_frame,
            "last_trusted_frame": self.manager.last_trusted_frame,
            "object_pose_raw": _pose7(result.T_raw), "object_pose_corrected": _pose7(result.T_corrected),
            "T_base_object_raw": None if result.T_raw is None else result.T_raw.tolist(),
            "symmetry_index": None if choice is None else choice.index,
            "symmetry": None if choice is None else BOX_SYMMETRY_NAMES[choice.index],
            "symmetry_angles_deg": None if choice is None else np.degrees(choice.angles_rad).tolist(),
            "symmetry_reference": result.reference,
            "goal": None if result.goal is None else result.goal.tolist(),
            "terms": {name: value.tolist() for name, value in result.terms.items()},
            "obs": None if result.vector is None else result.vector.tolist(),
            "robot": None if robot is None else robot.to_json(),
            "control_mode": self.control.mode, "control_state": self.control.state,
            "action_raw": None if raw is None else raw.tolist(),
            "action_clipped": None if target is None else target.clipped.tolist(),
            "target_tcp": _pose7(None if target is None else target.T_base_target),
            "gripper_target": None if target is None else target.gripper,
            "goal_time_s": self.control.goal_time_s, "sent": sent, "gripper_sent": gripper_sent,
            "hold_reason": decision.reason, "send_error": send_error,
        }
        self.log.tick(record, tick_at, build_s, frame_age, overrun)
        if tick_at - self._last_print_at >= self.stats_every_s:
            print(f"[policy recent] {self.log.stats_line()} | {result.phase}, trust {result.trust.value}, "
                  f"control {self.control.state}")
            self._last_print_at = tick_at

    def on_key(self, key: int) -> bool:
        """Handle a key; return True to quit."""
        if key in (ord("q"), 27):
            return True
        if key == self.keys["kill"]:
            self._kill("kill key")
        elif key == self.keys["episode"]:
            self._episode_key()
        elif key == self.keys["ready"]:
            problem = self.control.press_go(self.manager.phase == ACTIVE, self.trust.state)
            self.message = problem or ""
        elif key == self.keys["step"]:
            self.message = self.control.press_step() or ""
        self._emit_control()
        return False

    def draw(self, rgb: np.ndarray) -> np.ndarray:
        """Markers on the image only; the text is in draw_strip, below the image."""
        vis = rgb.copy()
        K, result, m = self._K, self._result, self.manager
        if K is not None:
            if self._robot is not None:
                draw_tool(vis, self.X_inv, self._robot.T_base_tcp, TCP_COLOR, K)
            if self._target is not None:
                draw_tool(vis, self.X_inv, self._target.T_base_target, TARGET_COLOR, K)
            if result is not None and result.T_corrected is not None:
                self._draw_axes(vis, result.T_corrected, K)
            if m.phase == ACTIVE and m.goal is not None:
                draw_point_marker(vis, self.X_inv, m.goal, GOAL_COLOR, K, from_base=m.initial_pose[:3, 3])
        return vis

    def draw_strip(self, width: int) -> np.ndarray:
        """Policy status as a strip under the image (tracking_loop stacks it below recovery's)."""
        result, m, c, target = self._result, self.manager, self.control, self._target
        rate = self.log.recent_rate()
        step = f" step {m.step}/{self.cfg.episode_steps}" if m.phase == ACTIVE else ""
        dry = " DRY RUN" if self.arm.dry_run else ""
        control = c.state
        if self._decision is not None and self._decision.reason and c.state in ctl.CONTROL_STATES:
            control += f" ({self._decision.reason})"
        if c.state == ctl.MOVING and self._ready_error is not None:
            control += (f" (joint err {np.degrees(self._ready_error[0]):.1f} deg, "
                        f"gripper {1000 * self._ready_error[1]:.1f} mm)")
        lines = [(f"POLICY {c.mode.upper()}{dry} | {control} | ep {m.episode} {m.phase}{step} | "
                  f"{'-' if rate is None else f'{rate:.1f}'}/{1.0 / c.tick_period_s:g} Hz",
                  STATE_COLORS.get(c.state, (235, 235, 235)))]
        if result is not None:
            sym = "-" if result.choice is None else BOX_SYMMETRY_NAMES[result.choice.index]
            line = f"{self._object_name} trust {result.trust.value} | symmetry {sym}"
            if "gripper_width" in result.terms:
                line += f" | gripper closure {result.terms['gripper_width'][0]:.2f}"
            lines.append((line, TRUST_COLORS[result.trust]))
            if "tcp_to_obj_pos" in result.terms:
                line = f"tcp to obj {fmt_vec(result.terms['tcp_to_obj_pos'])} m"
                if "obj_to_goal_pos" in result.terms:
                    line += f" | obj to goal {fmt_vec(result.terms['obj_to_goal_pos'])} m"
                lines.append((line, TCP_COLOR))
        if target is not None and self._raw_action is not None:
            lines.append((f"dp {fmt_vec(1000 * target.delta_pos)} mm | dr {fmt_vec(np.degrees(target.delta_rotvec))} "
                          f"deg | {'OPEN' if target.gripper_open else 'CLOSE'} {self._raw_action[6]:+.2f}",
                          TARGET_COLOR))
        if self.message:
            lines.append((self.message, MESSAGE_COLOR))
        k = self.cfg.control
        lines.append((f"{self.cfg.episode_key} ready/stop  {k.ready_key} go  {k.step_key} step  {k.kill_key} KILL  "
                      "q quit | TCP green, target cyan, goal magenta", (160, 160, 160)))
        strip = np.full((8 + STRIP_LINE * len(lines), width, 3), 20, dtype=np.uint8)
        for i, (text, color) in enumerate(lines):
            cv2.putText(strip, text, (8, 18 + STRIP_LINE * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                        cv2.LINE_AA)
        return strip

    # ----- control ----------------------------------------------------------------------

    def _target_for(self, robot, raw: np.ndarray) -> EeTarget:
        cfg = self.cfg
        return ee_delta_target(robot.T_base_tcp, raw, pos_scale=cfg.ee_pos_scale, rot_scale=cfg.ee_rot_scale,
                               gripper_open=cfg.gripper_open, gripper_closed=cfg.gripper_closed,
                               action_scale=cfg.control.action_scale)

    def _episode_key(self) -> None:
        do = self.control.press_episode()
        if do == ctl.DO_READY:
            if self.manager.phase != IDLE:
                self._emit(self.manager.end_episode("ready move"))
            try:
                self.arm.move_to_ready(self.cfg.ready_joint_pos, self.cfg.ready_gripper,
                                       self.cfg.control.ready_move_time_s)
                self.message = "moving to the ready pose"
            except Exception as error:
                self._emit({"event": "command_error", "error": repr(error), "during": "ready move"})
                self._kill(f"ready move failed: {error!r}")
        elif do == ctl.DO_ARM:
            self._emit(self.manager.arm_episode())
            self.message = "episode armed (shadow): waiting for a trusted object pose"
        elif do == ctl.DO_STOP:
            self._stop_at_current_pose()
            if self.manager.phase != IDLE:
                self._emit(self.manager.end_episode("key"))
            self.message = f"stopped; '{self.cfg.episode_key}' moves to the ready pose for a new episode"
        elif do == ctl.DO_NOTHING and self.control.state == ctl.KILLED:
            self.message = "arm was killed; restart to command it again"

    def _check_ready(self, robot) -> None:
        if self.control.state != ctl.MOVING:
            self._ready_error = None
            return
        if self.arm.dry_run:
            reached = True
        elif robot is None:
            return
        else:
            self._ready_error = ready_error(robot, self.cfg.ready_joint_pos, self.cfg.ready_gripper)
            reached = (self._ready_error[0] <= self.cfg.control.ready_tolerance_rad
                       and self._ready_error[1] <= self.cfg.control.ready_gripper_tolerance)
        if reached:
            self.control.ready_reached()
            self._emit(self.manager.arm_episode())
            self.message = "at the ready pose: waiting for a trusted object pose"
            self._ready_error = None

    def _stop_at_current_pose(self) -> None:
        if self.control.shadow:
            return
        try:
            robot = self.arm.read()
            if robot is not None:
                self.arm.hold_at(robot.arm_q, self.cfg.control.stop_goal_time_s)
        except Exception as error:
            self._emit({"event": "command_error", "error": repr(error), "during": "stop"})
            self._kill(f"stop failed: {error!r}")

    def _robot_error(self, error: Exception, frame_number: int) -> None:
        """A failed read usually means the controller faulted (and went idle): kill, which clears the
        error and restores gravity compensation. Repeats of the same error are logged, not printed."""
        text = repr(error)
        repeat = text == self._last_robot_error
        self._last_robot_error = text
        self._robot_errors += 1
        if repeat:
            self.log.event("robot_error", error=text, frame_number=frame_number, repeat=self._robot_errors)
            return
        self._emit({"event": "robot_error", "error": text}, frame_number=frame_number)
        lines = str(error).splitlines()
        self._kill(f"driver error: {lines[0][:120] if lines else type(error).__name__}")

    def _kill(self, reason: str) -> None:
        if self.control.state == ctl.KILLED:
            return
        self.arm.kill(reason)
        self.control.kill(reason)
        self._emit({"event": "kill", "reason": reason})
        if self.manager.phase != IDLE:
            self._emit(self.manager.end_episode("killed"))
        self.message = "KILLED: gravity compensation; restart to command the arm again"

    def _emit_control(self, frame_number: Optional[int] = None) -> None:
        for event in self.control.drain_events():
            extra = {} if frame_number is None else {"frame_number": frame_number}
            self._emit(event, **extra)

    # ----- helpers ----------------------------------------------------------------------

    def _draw_axes(self, vis, T_base_obj, K, length=0.03) -> None:
        from pipeline.drawing import project_point

        T_cam = self.X_inv @ T_base_obj
        origin = project_point(T_cam[:3, 3], K)
        if origin is None:
            return
        for axis, color in enumerate(AXIS_COLORS):
            tip = project_point(T_cam[:3, 3] + T_cam[:3, axis] * length, K)
            if tip is not None:
                cv2.line(vis, origin, tip, color, 2, cv2.LINE_AA)

    def _resolve_object(self, tracker) -> None:
        """The tracked object to observe: config tracked_name, else the one whose mesh is the scene's."""
        from pipeline.run_folder import load_run

        meshes = {o.name: o.mesh_path for o in load_run(self.save_path).objects}
        names = [t.name for t in tracker.objects]
        want = _file_digest(self.cfg.mesh_path)
        if self.cfg.tracked_name is not None:
            if self.cfg.tracked_name not in names:
                raise ValueError(f"object.tracked_name {self.cfg.tracked_name!r} is not tracked; tracked: {names}")
            name = self.cfg.tracked_name
        else:
            matches = [n for n in names if meshes.get(n) is not None and _file_digest(meshes[n]) == want]
            if len(matches) != 1:
                raise ValueError(f"{len(matches)} tracked objects use the scene mesh {self.cfg.mesh_path} "
                                 f"(tracked: { {n: str(meshes.get(n)) for n in names} }); "
                                 "set object.tracked_name in the policy config")
            name = matches[0]
        same_mesh = meshes.get(name) is not None and _file_digest(meshes[name]) == want
        if not same_mesh:
            print(f"WARNING: {name} tracks {meshes.get(name)}, not the training mesh {self.cfg.mesh_path}; "
                  "its mesh frame may differ from the simulated object's")
        self._index = names.index(name)
        rec_names = [s.name for s in self.recovery.states]
        self._rec_index = rec_names.index(name)
        self._object_name = name
        self.log.event("object", name=name, mesh=str(meshes.get(name)), same_mesh_as_training=same_mesh,
                       to_origin=tracker.objects[self._index].to_origin.tolist())
        print(f"Policy observations use {name} ({meshes.get(name)})")

    def _emit(self, event: dict, **extra) -> None:
        data = {k: v for k, v in event.items() if k != "event"}
        data.update(extra)
        if event["event"] == "episode_start":
            self.message = ""
        self.log.event(event["event"], **data)
        if event["event"] in PRINTED_EVENTS:
            brief = {k: v for k, v in data.items() if not k.startswith("T_") and k != "symmetry_angles_deg"}
            print(f"[policy] {event['event']} {json.dumps(brief, default=str)}")


@contextmanager
def policy_session(args, recovery):
    """Load the config, policy and calibration, connect the arm (unless dry run), yield a started PolicySession."""
    if recovery is None:
        raise ValueError("--manipulation policy requires --recovery")
    cfg = JengaPolicyConfig.load(args.manipulation_config).with_mode(getattr(args, "policy_control", None))
    if cfg.checkpoint is None:
        raise ValueError(f"{cfg.path}: policy.checkpoint is not set")
    actor = PolicyActor.load(cfg.checkpoint, cfg)
    actor(actor.obs_mean)  # first numpy call is slow; keep it off the first tick
    print(f"Policy {cfg.checkpoint} (iteration {actor.info.get('iteration')}) matches the scene and task.")
    cal_path = find_calibration(args.calibration or cfg.calibration)
    cal = json.loads(cal_path.read_text())
    X = load_T_base_camera(cal_path)
    rms = cal.get("results", {}).get(cal.get("method"), {}).get("residuals", {}).get("corner_rms_mm")
    print(f"Calibration {cal_path} ({cal.get('method')}, corner RMS {rms} mm); camera at "
          f"{np.round(X[:3, 3], 3)} m in the base frame. The camera arm must be at its calibrated parked pose.")
    if args.manipulation_dry_run:
        arm = DryRunPolicyArm()
    else:
        arm = PolicyArm(resolve_robot_identity(cfg.robot, cal, args.calibration_config),
                        cfg.control.gripper_limit_margin_m)
    log = PolicyLog(args.save_path / LOG_FILE if args.manipulation_logging else None)
    try:
        with arm:
            session = PolicySession(cfg, X, arm, actor, recovery, log, args.save_path, args.stats_every)
            session.start({"config": cfg.describe(), "calibration": str(cal_path), "T_base_camera": X.tolist(),
                           "robot": arm.info, "dry_run": bool(args.manipulation_dry_run),
                           "policy": actor.info, "obs_layout": [list(t) for t in obs.ACTOR_TERMS]})
            try:
                yield session
            finally:
                session.close()
    finally:
        log.close()
