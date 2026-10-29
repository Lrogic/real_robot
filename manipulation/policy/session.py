"""Hooks that add the policy observation pipeline to pipeline.tracking_loop.track_live (--manipulation policy).

Stage 1: observations only. The arm is in gravity compensation and is never commanded; guide
it by hand. Every tracking-loop iteration (after FoundationPose and SAM recovery) updates the
pose trust; at the policy's control rate the observation is built from one robot-state read
and the tracked object's original-mesh pose in the base frame,
T_base_object = T_base_camera @ center_pose @ to_origin.

In the tracking window:
    s        arm an episode (starts on the next tick with a trusted pose); press again to end it
    q / Esc  quit
"""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from calibration.session import find_calibration
from calibration.transforms import inv_T, load_T_base_camera

from ..common.drawing import draw_label, draw_point_marker, draw_tool, fmt_vec
from . import observations as obs
from .jenga.config import JengaPolicyConfig
from .jenga.observation_manager import ACTIVE, ARMED, IDLE, JengaObservationManager
from .policy_log import LOG_FILE, PolicyLog
from .robot_state import RobotStateReader, resolve_robot_identity
from .symmetry import BOX_SYMMETRY_NAMES
from .trust import Trust, TrustTracker

TCP_COLOR, GOAL_COLOR, MESSAGE_COLOR = (0, 255, 0), (255, 0, 255), (255, 220, 0)
TRUST_COLORS = {Trust.VERIFIED: (0, 255, 0), Trust.UNVERIFIED_OK: (235, 235, 235), Trust.UNTRUSTED: (255, 60, 60)}
AXIS_COLORS = ((255, 0, 0), (0, 255, 0), (0, 0, 255))
PRINTED_EVENTS = {"episode_armed", "episode_start", "episode_end", "episode_cancelled", "reregistration",
                  "recovery_disabled", "episode_length_exceeded", "robot_error", "tracking_resumed"}


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
    def __init__(self, cfg: JengaPolicyConfig, T_base_camera: np.ndarray, reader, recovery, log: PolicyLog,
                 save_path: Path, stats_every_s: float):
        self.cfg = cfg
        self.X, self.X_inv = T_base_camera, inv_T(T_base_camera)
        self.reader, self.recovery, self.log = reader, recovery, log
        self.save_path = save_path
        self.stats_every_s = stats_every_s
        self.manager = JengaObservationManager(cfg)
        self.trust = TrustTracker()
        self.schedule = TickSchedule(cfg.control_hz)
        self.key = ord(cfg.episode_key)
        self.message = ""
        self._index: Optional[int] = None
        self._rec_index: Optional[int] = None
        self._object_name: Optional[str] = None
        self._K = None
        self._result = None
        self._robot = None
        self._last_update_at: Optional[float] = None
        self._last_print_at = time.monotonic()
        self._closed = False

    # ----- lifecycle --------------------------------------------------------------------

    def start(self, info: dict) -> None:
        self.log.event("start", **info)
        print(__doc__)
        print(f"Policy observations at {self.cfg.control_hz:g} Hz; key '{self.cfg.episode_key}' arms/ends an "
              f"episode. Logging: {self.log.path if self.log.enabled else 'off (--manipulation-logging)'}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.manager.phase != IDLE:
            self._emit(self.manager.end_episode("tracking stopped"))
        print(f"[policy overall] {self.log.stats_line()}")
        self.log.event("stop", ticks=self.log.ticks, overall_rate_hz=self.log.overall_rate(),
                       overruns=self.log.overruns)
        self.log.close()

    # ----- hooks called by track_live ---------------------------------------------------

    def on_mouse(self, *_args) -> None:
        pass

    def update(self, tracker, K: np.ndarray, timing) -> None:
        now = time.monotonic()
        update_interval = None if self._last_update_at is None else now - self._last_update_at
        self._last_update_at = now
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
        if self.reader is not None:
            try:
                robot = self.reader.read()
                robot_read_s = robot.read_at - tick_at
            except Exception as error:  # a failed read must not end the tracking loop
                self._emit({"event": "robot_error", "error": repr(error)}, frame_number=timing.frame_number)
        result = self.manager.tick(robot, T_raw, trust, timing.frame_number)
        build_s = time.monotonic() - tick_at
        for event in result.events:
            self._emit(event, frame_number=timing.frame_number)
        if result.phase == ACTIVE:
            self.message = ""
        self._result, self._robot = result, robot

        choice = result.choice
        record = {
            "frame_number": timing.frame_number, "camera_timestamp_ms": timing.camera_timestamp_ms,
            "update_interval_s": update_interval, "robot_read_s": robot_read_s,
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
        }
        self.log.tick(record, tick_at, build_s, frame_age, overrun)
        if tick_at - self._last_print_at >= self.stats_every_s:
            print(f"[policy recent] {self.log.stats_line()} | {result.phase}, trust {result.trust.value}")
            self._last_print_at = tick_at

    def on_key(self, key: int) -> bool:
        """Handle a key; return True to quit."""
        if key in (ord("q"), 27):
            return True
        if key != self.key:
            return False
        phase = self.manager.phase
        if phase == IDLE:
            self._emit(self.manager.arm_episode())
            self.message = "episode armed: waiting for a trusted object pose"
        elif phase == ARMED:
            self._emit(self.manager.end_episode("key before start"))
            self.message = "episode cancelled"
        else:
            self._emit(self.manager.end_episode("key"))
            self.message = f"episode ended; '{self.cfg.episode_key}' starts a new one"
        return False

    def draw(self, rgb: np.ndarray) -> np.ndarray:
        vis = rgb.copy()
        K, result, m = self._K, self._result, self.manager
        if K is not None:
            if self._robot is not None:
                draw_tool(vis, self.X_inv, self._robot.T_base_tcp, TCP_COLOR, K)
            if result is not None and result.T_corrected is not None:
                self._draw_axes(vis, result.T_corrected, K)
            if m.phase == ACTIVE and m.goal is not None:
                draw_point_marker(vis, self.X_inv, m.goal, GOAL_COLOR, K, label="goal",
                                  from_base=m.initial_pose[:3, 3])

        lines = []
        step = "" if m.phase != ACTIVE else f"  step {m.step}/{self.cfg.episode_steps}"
        dry = "  DRY RUN (no arm)" if self.reader is None else ""
        lines.append((f"POLICY OBS {m.phase.upper()}  episode {m.episode}{step}{dry}", (255, 255, 255)))
        if result is not None:
            sym = "-"
            if result.choice is not None:
                k = result.choice.index
                sym = (f"{BOX_SYMMETRY_NAMES[k]} ({np.degrees(result.choice.angles_rad[k]):.0f} deg from "
                       f"{result.reference})")
            lines.append((f"{self._object_name}: trust {result.trust.value} | symmetry {sym}",
                          TRUST_COLORS[result.trust]))
            rate = self.log.recent_rate()
            lines.append((f"rate {'-' if rate is None else f'{rate:.1f}'} Hz (target {self.cfg.control_hz:g})",
                          (235, 235, 235)))
            if result.goal is not None and "obj_to_goal_pos" in result.terms:
                lines.append((f"goal base {fmt_vec(result.goal)} m | obj to goal "
                              f"{fmt_vec(result.terms['obj_to_goal_pos'])} m", GOAL_COLOR))
            if "tcp_to_obj_pos" in result.terms:
                lines.append((f"tcp to obj {fmt_vec(result.terms['tcp_to_obj_pos'])} m | gripper closure "
                              f"{result.terms['gripper_width'][0]:.2f}", TCP_COLOR))
        if self.message:
            lines.append((self.message, MESSAGE_COLOR))
        y = 24
        for text, color in lines:
            draw_label(vis, text, (10, y), color)
            y += 24
        draw_label(vis, f"{self.cfg.episode_key} arm/end episode | q quit | TCP green, goal magenta",
                   (10, vis.shape[0] - 12), (200, 200, 200), 0.45)
        return vis

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
    """Load the config and calibration, connect the arm (unless dry run), yield a started PolicySession."""
    if recovery is None:
        raise ValueError("--manipulation policy requires --recovery")
    cfg = JengaPolicyConfig.load(args.manipulation_config)
    cal_path = find_calibration(args.calibration or cfg.calibration)
    cal = json.loads(cal_path.read_text())
    X = load_T_base_camera(cal_path)
    rms = cal.get("results", {}).get(cal.get("method"), {}).get("residuals", {}).get("corner_rms_mm")
    print(f"Calibration {cal_path} ({cal.get('method')}, corner RMS {rms} mm); camera at "
          f"{np.round(X[:3, 3], 3)} m in the base frame. The camera arm must be at its calibrated parked pose.")
    reader = None
    if not args.manipulation_dry_run:
        reader = RobotStateReader(resolve_robot_identity(cfg.robot, cal, args.calibration_config))
    log = PolicyLog(args.save_path / LOG_FILE if args.manipulation_logging else None)
    try:
        with reader if reader is not None else nullcontext():
            session = PolicySession(cfg, X, reader, recovery, log, args.save_path, args.stats_every)
            session.start({"config": cfg.describe(), "calibration": str(cal_path), "T_base_camera": X.tolist(),
                           "robot": None if reader is None else reader.info,
                           "dry_run": bool(args.manipulation_dry_run),
                           "obs_layout": [list(t) for t in obs.ACTOR_TERMS]})
            try:
                yield session
            finally:
                session.close()
    finally:
        log.close()
