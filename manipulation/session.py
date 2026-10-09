"""Hooks that add the manipulation test to track_objects.track_live (--manipulation).

The arm starts in gravity compensation: guide it by hand to a ready pose (elbow bent, gripper
level, well above the table). Confirming locks it there in position mode, unless a joint is
near its limit (e.g. the folded rest pose), in which case confirm is refused.

In the tracking window:
    left click   select a tracked object (highlighted, asks to confirm)
    y / Enter    confirm: the arm moves slowly to the hover pose and follows the object
    n            cancel the pending selection
    g            enclose: move in until the object sits between the open fingers (from hover)
    b            back off to hover
    h            hold (stop following; the arm holds its pose)
    s            switch stepping from the last command (default) to the measured pose and back
    Space        KILL: gravity compensation, no further commands (also Enter in the terminal, Ctrl+C)
    q / Esc      kill and quit
"""

import json
import select
import signal
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import calibration
from calibration.transforms import inv_T, load_T_base_camera, transform_points
from fp_capture.tracking import _project

from .controller import HOLD, KILLED, FollowController
from .grasp import ENCLOSE, HOVER, ManipulationConfig, joint_limit_margins

CALIBRATION_SESSIONS = Path(calibration.__file__).resolve().parent / "data" / "sessions"
LOG_FILE = "manipulation_log.jsonl"
PICK_TOLERANCE_PX = 25
PENDING_COLOR, SELECTED_COLOR = (255, 220, 0), (0, 220, 255)
GOAL_COLOR, COMMAND_COLOR, MEASURED_COLOR = (255, 0, 255), (255, 255, 255), (0, 255, 0)
KILL_COLOR = (255, 40, 40)
FONT = cv2.FONT_HERSHEY_SIMPLEX
HELP = "click select | y confirm | n cancel | g enclose | b back | h hold | s step source | SPACE KILL | q quit"


def find_calibration(path: Optional[Path]) -> Path:
    if path is not None:
        return path
    found = sorted(CALIBRATION_SESSIONS.glob("*/T_base_camera.json"))
    if not found:
        raise FileNotFoundError(f"No T_base_camera.json under {CALIBRATION_SESSIONS}; pass --calibration")
    return found[-1]


def robot_config(calibration_json: dict, cfg: ManipulationConfig) -> dict:
    """ip / model / end_effector from config.yaml, falling back to the calibration session's robot."""
    session_robot = calibration_json.get("session_meta", {}).get("robot", {})
    robot = {}
    for key in ("ip", "model", "end_effector"):
        robot[key] = cfg.robot.get(key) or session_robot.get(key)
        if robot[key] is None:
            raise ValueError(f"robot.{key} is not in the calibration metadata; set it in config.yaml")
    if robot["end_effector"] != session_robot.get("end_effector"):
        print(f"WARNING: end effector {robot['end_effector']} differs from the calibration's "
              f"{session_robot.get('end_effector')}")
    return robot


def _box_hull(obj, K: np.ndarray) -> Optional[np.ndarray]:
    if obj.center_pose is None:
        return None
    lo, hi = obj.bbox
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    cam = transform_points(obj.center_pose, corners)
    if np.any(cam[:, 2] <= 0):
        return None
    uv = (cam @ K.T)[:, :2] / cam[:, 2:3]
    return cv2.convexHull(np.round(uv).astype(np.int32))


def _text(img, text, org, color=(235, 235, 235), scale=0.55, thickness=1):
    (w, h), base = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x - 4, y - h - 4), (x + w + 4, y + base + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def _fmt(p) -> str:
    return "-" if p is None else "(" + ", ".join(f"{v:+.3f}" for v in p) + ")"


class ManipulationSession:
    def __init__(self, arm, T_base_camera: np.ndarray, cfg: ManipulationConfig, log_path: Path):
        self.arm, self.cfg = arm, cfg
        self.X, self.X_inv = T_base_camera, inv_T(T_base_camera)
        self.controller = FollowController(arm, cfg)
        self.pending: Optional[int] = None
        self.selected: Optional[int] = None
        self.message = ""
        self._objects, self._K = [], None
        self._log = open(log_path, "a")
        self._stdin_stop = threading.Event()
        self._stdin_thread = threading.Thread(target=self._watch_stdin, name="stdin-kill", daemon=True)
        self._old_sigint = None
        self._closed = False

    def start(self) -> None:
        self._write_log({"event": "start", "T_base_camera": self.X.tolist(), "config": self.cfg.__dict__,
                         "arm": self.arm.info})
        self.controller.start()
        self._stdin_thread.start()
        self._old_sigint = signal.signal(signal.SIGINT, self._on_sigint)
        print(__doc__)
        print("KILL SWITCH: Space in the window, Enter in this terminal, or Ctrl+C.")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.controller.kill("tracking stopped")
        self.controller.stop()
        self._stdin_stop.set()
        self._stdin_thread.join(timeout=1.0)
        if self._old_sigint is not None:
            signal.signal(signal.SIGINT, self._old_sigint)
        self._write_log({"event": "stop"})
        self._log.close()

    # ----- kill paths outside the window ------------------------------------

    def _on_sigint(self, _sig, _frame):
        self.controller.kill("Ctrl+C")
        raise KeyboardInterrupt

    def _watch_stdin(self) -> None:
        while not self._stdin_stop.is_set():
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.2)
            except (OSError, ValueError):
                return
            if not ready:
                continue
            if sys.stdin.readline() == "":
                return
            self.controller.kill("terminal Enter")

    # ----- hooks called by track_live ---------------------------------------

    def on_mouse(self, event, x, y, _flags, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or self._K is None:
            return
        best = None
        for index, obj in enumerate(self._objects):
            hull = _box_hull(obj, self._K)
            if hull is None:
                continue
            distance = cv2.pointPolygonTest(hull, (float(x), float(y)), True)
            if best is None or distance > best[0]:
                best = (distance, index)
        if best is None or best[0] < -PICK_TOLERANCE_PX:
            self.message = "no tracked object there"
        elif self.controller.mode == KILLED:
            self.message = "killed: restart to move the arm again"
        else:
            self.pending, self.message = best[1], ""

    def update(self, tracker, K: np.ndarray, timing) -> None:
        self._objects, self._K = tracker.objects, K
        if self.selected is None:
            return
        obj = self._objects[self.selected]
        if obj.center_pose is None:
            return
        T_base_object = self.X @ obj.center_pose
        self.controller.set_target(T_base_object[:3, 3])
        snap = self.controller.snapshot()
        self._write_log({"frame_number": timing.frame_number, "object": obj.name, "mode": snap.mode,
                         "status": snap.status, "step_from_measured": self.controller.step_from_measured, "T_camera_object": obj.center_pose.tolist(),
                         "T_base_object": T_base_object.tolist(),
                         "target_filtered": None if snap.target is None else snap.target.tolist(),
                         "goal": None if snap.goal is None else snap.goal.tolist(),
                         "commanded": None if snap.commanded is None else snap.commanded.tolist(),
                         "T_base_ee_measured": None if snap.measured is None else snap.measured.tolist(),
                         "arm_positions_rad": None if snap.joints is None else snap.joints.tolist()})

    def on_key(self, key: int) -> bool:
        """Handle a key; return True to quit."""
        mode = self.controller.mode
        if key == 32:
            self.controller.kill("Space")
        elif key in (ord("q"), 27):
            self.controller.kill("quit")
            return True
        elif key in (ord("y"), 13, 10) and self.pending is not None:
            if mode == ENCLOSE and self.pending != self.selected:
                self.message = "back off to hover (b) before switching objects"
            else:
                previous = self.selected
                self.selected = self.pending
                self.controller.new_target()
                refusal = self.controller.request(HOVER)
                if refusal is not None:
                    self.selected, self.message = previous, refusal
                else:
                    self.pending, self.message = None, ""
                    self._write_log({"event": "confirm", "object": self._objects[self.selected].name})
        elif key == ord("n"):
            self.pending = None
        elif key == ord("g") and self.selected is not None:
            self.controller.request(ENCLOSE)
        elif key == ord("b") and mode == ENCLOSE:
            self.controller.request(HOVER)
        elif key == ord("h"):
            self.controller.request(HOLD)
        elif key == ord("s"):
            source = "measured" if self.controller.toggle_step_source() else "command"
            print(f"Stepping from the {source} pose")
            self._write_log({"event": "step_source", "source": source})
        return False

    def draw(self, rgb: np.ndarray) -> np.ndarray:
        vis = rgb.copy()
        K = self._K
        snap = self.controller.snapshot()
        for index, color, thickness in ((self.selected, SELECTED_COLOR, 3), (self.pending, PENDING_COLOR, 4)):
            if index is None or K is None:
                continue
            hull = _box_hull(self._objects[index], K)
            if hull is not None:
                cv2.polylines(vis, [hull], True, color, thickness, cv2.LINE_AA)
        if K is not None:
            for T, color in ((snap.goal, GOAL_COLOR), (snap.commanded, COMMAND_COLOR), (snap.measured, MEASURED_COLOR)):
                self._draw_tool(vis, T, color, K)

        name = "-" if self.selected is None else self._objects[self.selected].name
        source = "MEASURED" if self.controller.step_from_measured else "COMMAND"
        lines = [(f"MANIPULATION {snap.mode.upper()}  object {name}  step from {source}", (255, 255, 255)),
                 (snap.status, KILL_COLOR if snap.mode == KILLED else (235, 235, 235)),
                 (f"target base {_fmt(snap.target)} m", (235, 235, 235))]
        if snap.goal is not None and snap.measured is not None:
            gap = np.linalg.norm(snap.goal[:3, 3] - snap.measured[:3, 3]) * 100
            lines.append((f"tool to goal {gap:.1f} cm  (goal magenta, cmd white, measured green)", (235, 235, 235)))
        if snap.joints is not None and self.arm.joint_limits is not None and np.all(np.isfinite(snap.joints)):
            margins = np.degrees(joint_limit_margins(snap.joints, self.arm.joint_limits))
            j = int(np.argmin(margins))
            lines.append((f"closest to limit: joint {j + 1} ({margins[j]:.0f} deg)",
                          PENDING_COLOR if margins[j] < 10 else (235, 235, 235)))
        if self.message:
            lines.append((self.message, PENDING_COLOR))
        y = 24
        for text, color in lines:
            _text(vis, text, (10, y), color)
            y += 24
        if self.pending is not None:
            _text(vis, f"Move arm to {self._objects[self.pending].name}?  y = confirm, n = cancel",
                  (10, vis.shape[0] - 44), PENDING_COLOR, 0.7, 2)
        if snap.mode == KILLED:
            _text(vis, "KILLED - gravity compensation. Guide the arm to park, q to quit.",
                  (10, vis.shape[0] // 2), KILL_COLOR, 0.75, 2)
        _text(vis, HELP, (10, vis.shape[0] - 12), (200, 200, 200), 0.45)
        return vis

    def _draw_tool(self, vis, T_base_tool, color, K) -> None:
        if T_base_tool is None:
            return
        T_cam_tool = self.X_inv @ T_base_tool
        origin = _project(T_cam_tool[:3, 3], K)
        tip = _project(T_cam_tool[:3, 3] - T_cam_tool[:3, 0] * 0.05, K)
        if origin is None:
            return
        cv2.circle(vis, origin, 6, color, 2, cv2.LINE_AA)
        if tip is not None:
            cv2.line(vis, origin, tip, color, 2, cv2.LINE_AA)

    def _write_log(self, record: dict) -> None:
        if not self._log.closed:
            self._log.write(json.dumps({"t": time.time(), **record}) + "\n")


@contextmanager
def manipulation_session(args):
    """Connect the arm (or the dry-run stand-in) and yield a started ManipulationSession."""
    from .arm import DryRunArm, ManipulatorArm

    cfg = ManipulationConfig.load(args.manipulation_config)
    cal_path = find_calibration(args.calibration)
    cal = json.loads(cal_path.read_text())
    X = load_T_base_camera(cal_path)
    rms = cal.get("results", {}).get(cal.get("method"), {}).get("residuals", {}).get("corner_rms_mm")
    print(f"Calibration {cal_path} ({cal.get('method')}, corner RMS {rms} mm); camera at "
          f"{np.round(X[:3, 3], 3)} m in the base frame. The camera arm must be at its calibrated parked pose.")
    if args.manipulation_dry_run:
        arm = DryRunArm()
    else:
        arm = ManipulatorArm(robot_config(cal, cfg), cfg.gripper_open_margin_m, cfg.gripper_open_time_s)
    with arm:
        session = ManipulationSession(arm, X, cfg, args.save_path / LOG_FILE)
        session.start()
        try:
            yield session
        finally:
            session.close()
