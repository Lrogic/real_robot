"""Camera (follower) arm setup, run before anything else: --camera parked | move.

parked  The follower moves to the parked calibration's recorded joints and holds them in
        position mode. T_base_camera is that calibration's.
move    The follower goes to gravity compensation and a live camera window opens. Guide it by
        hand, then press Enter or space to lock it in position mode where it is (q/Esc aborts).
        T_base_camera = T_base_fbase @ F @ T_fee_camera from the two-arm calibration, with F the
        follower's driver Cartesian pose once locked. Tracking a saved run (--track alone) moves
        the follower back to the joints that run's setup locked.

The follower stays held until the run exits; run.sh runs setup and tracking in one process so
the hold never lapses between them. On exit a move-mode follower first goes back to the parked
joints: disconnecting sets idle, and idle drops an extended arm (docs/camera_calibration.md, 3).
Paths and timings are in pipeline/camera_arm.yaml (--camera-config).
"""

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from calibration.board import load_config
from calibration.reset_follower import session_follower_joints
from calibration.session import read_json
from calibration.transforms import inv_T, load_T_base_camera, rotation_angle_deg

CAMERA_MODES = ("parked", "move")
DEFAULT_CAMERA_ARM_CONFIG = Path(__file__).resolve().parents[1] / "camera_arm.yaml"
GUIDE_WINDOW = "camera arm: guide, Enter/space locks, q/Esc aborts"
LOCK_KEYS = (13, 10, 32)


@dataclass(frozen=True)
class CameraArmConfig:
    path: Path
    parked_calibration: Path
    move_calibration: Path
    lock_settle_s: float
    move_time_s: float

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "CameraArmConfig":
        path = Path(path or DEFAULT_CAMERA_ARM_CONFIG).resolve()
        raw = yaml.safe_load(path.read_text()) or {}
        unknown = set(raw) - {"parked", "move", "move_time_s"}
        if unknown:
            raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
        try:
            parked, move = raw["parked"], raw["move"]
            cfg = cls(path=path,
                      parked_calibration=(path.parent / parked["calibration"]).resolve(),
                      move_calibration=(path.parent / move["calibration"]).resolve(),
                      lock_settle_s=float(move.get("lock_settle_s", 0.5)),
                      move_time_s=float(raw.get("move_time_s", 2.0)))
        except (KeyError, TypeError) as error:
            raise ValueError(f"{path}: needs parked.calibration and move.calibration ({error})") from error
        if cfg.move_time_s <= 0 or cfg.lock_settle_s < 0:
            raise ValueError(f"{path}: move_time_s must be > 0 and move.lock_settle_s >= 0")
        return cfg


@dataclass
class CameraPose:
    """Where the camera is for this run. Saved to the run folder as camera_pose.json."""
    mode: str
    T_base_camera: np.ndarray
    calibration: Path                      # the calibration file the pose comes from
    parked_calibration: Path               # its session_meta names the manipulator (ip/model/EE)
    follower_joints_rad: Optional[list] = None
    T_fbase_fee: Optional[np.ndarray] = None  # move only: the follower's locked Cartesian pose
    extra: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"mode": self.mode, "T_base_camera": self.T_base_camera.tolist(),
                "calibration": str(self.calibration), "follower_joints_rad": self.follower_joints_rad,
                "T_fbase_fee": None if self.T_fbase_fee is None else self.T_fbase_fee.tolist(),
                "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **self.extra}

    def manipulator_calibration_json(self) -> dict:
        """The parked T_base_camera.json, whose session_meta.robot is the manipulator either way."""
        return json.loads(self.parked_calibration.read_text())


def parked_calibration(path: Path):
    """(T_base_camera, the follower joints it was measured at) from a parked T_base_camera.json."""
    session = path.parent / "session.json"
    if not session.is_file():
        raise FileNotFoundError(f"{path}: no session.json beside it for the parked follower joints")
    return load_T_base_camera(path), session_follower_joints(read_json(session))


def move_calibration(path: Path):
    """(T_base_fbase, T_fee_camera, follower identity) from a two-arm T_follower_camera.json."""
    data = json.loads(Path(path).read_text())
    try:
        Z = np.asarray(data["T_base_fbase"], dtype=np.float64)
        X = np.asarray(data["T_fee_camera"], dtype=np.float64)
    except KeyError as error:
        raise ValueError(f"{path} is not a two-arm calibration (no {error})") from error
    return Z, X, data.get("follower") or {}


def pose_difference(Ta: np.ndarray, Tb: np.ndarray) -> str:
    d = inv_T(Ta) @ Tb
    return f"{1000 * np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]):.1f} mm, {rotation_angle_deg(d[:3, :3]):.2f} deg"


class _Follower:
    """The follower driver, held in position mode from connect to close; never idled while extended."""

    def __init__(self, follower_cfg: dict):
        from calibration.robot import FollowerArm

        self.arm = FollowerArm(follower_cfg, hold_mode="position")

    def __enter__(self) -> "_Follower":
        self.arm.__enter__()  # connects and locks where it is
        return self

    def move_to(self, joints, move_time_s: float) -> float:
        """Joint-space move in position mode; returns the max joint error (rad) after it."""
        import trossen_arm

        driver = self.arm.driver
        driver.set_arm_modes(trossen_arm.Mode.position)
        driver.set_arm_positions([float(q) for q in joints], move_time_s, True)
        self.arm.locked = True
        reached = np.asarray(driver.get_arm_positions(), dtype=np.float64)
        return float(np.abs(reached - np.asarray(joints, dtype=np.float64)).max())

    def free(self) -> None:
        self.arm.free()

    def lock(self) -> None:
        self.arm.lock()

    def read_state(self):
        return self.arm.read_state()

    def close(self) -> None:
        if self.arm.driver is not None:
            self.arm.driver.cleanup()
            self.arm.driver = None


def guide_follower(follower: _Follower, settings) -> bool:
    """Gravity compensation and a live camera feed until Enter/space (lock, True) or q/Esc (lock, False)."""
    from ..camera import RealSenseStream

    follower.free()
    print("Follower in gravity compensation: guide the camera, then press Enter or space in the window to lock it.")
    try:
        with RealSenseStream(settings) as stream:
            cv2.namedWindow(GUIDE_WINDOW, cv2.WINDOW_NORMAL)
            while True:
                rgb, _, _ = stream.read()
                view = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                for i, text in enumerate(("Follower FREE (gravity compensation): guide the camera",
                                          "Enter/space: lock here    q/Esc: abort")):
                    cv2.putText(view, text, (10, 25 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
                cv2.imshow(GUIDE_WINDOW, view)
                key = cv2.waitKey(1) & 0xFF
                if key in LOCK_KEYS:
                    return True
                if key in (ord("q"), 27) or cv2.getWindowProperty(GUIDE_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    return False
    finally:
        follower.lock()  # never leave the follower free, whatever ended the loop
        cv2.destroyWindow(GUIDE_WINDOW)
        cv2.waitKey(1)


def _locked_F(follower: _Follower, settle_s: float):
    time.sleep(settle_s)
    state = follower.read_state()
    return state.T_base_ee, state.arm_positions.tolist()


@contextmanager
def held_camera(args, settings, saved: Optional[dict] = None):
    """Set up the camera arm for args.camera and yield its CameraPose; the follower is held until exit.

    saved: the run folder's camera_pose.json when tracking a run set up earlier (move: return there).
    """
    cfg = CameraArmConfig.load(getattr(args, "camera_config", None))
    follower_cfg = load_config(getattr(args, "calibration_config", None))["follower"]
    T_parked, parked_joints = parked_calibration(cfg.parked_calibration)
    if args.camera == "move":
        Z, X, identity = move_calibration(cfg.move_calibration)
        for key in ("model", "end_effector"):
            if identity.get(key) and identity[key] != follower_cfg[key]:
                raise ValueError(f"follower {key} {follower_cfg[key]!r} differs from the two-arm calibration's "
                                 f"{identity[key]!r} ({cfg.move_calibration}); F would be read in another frame")
    print(f"Camera arm ({args.camera}): follower {follower_cfg['ip']}, config {cfg.path}")

    follower = _Follower(follower_cfg).__enter__()
    try:
        if args.camera == "parked":
            err = follower.move_to(parked_joints, cfg.move_time_s)
            print(f"Follower parked at {cfg.parked_calibration.parent.name} joints "
                  f"(max error {np.degrees(err):.2f} deg) and held in position mode.")
            pose = CameraPose("parked", T_parked, cfg.parked_calibration, cfg.parked_calibration,
                              list(parked_joints))
            if saved is not None and saved.get("calibration") != str(cfg.parked_calibration):
                print(f"WARNING: the run was set up with the parked calibration {saved.get('calibration')}, "
                      f"now {cfg.parked_calibration}")
        else:
            if saved is None:
                if not guide_follower(follower, settings):
                    raise SystemExit("Camera arm setup aborted")
            else:
                err = follower.move_to(saved["follower_joints_rad"], cfg.move_time_s)
                print(f"Follower back at the run's locked joints (max error {np.degrees(err):.2f} deg).")
            F, joints = _locked_F(follower, cfg.lock_settle_s)
            pose = CameraPose("move", Z @ F @ X, cfg.move_calibration, cfg.parked_calibration, joints, F,
                              {"lock_settle_s": cfg.lock_settle_s})
            print(f"Follower locked in position mode; camera {pose_difference(T_parked, pose.T_base_camera)} "
                  "from the parked calibration's camera pose.")
            if saved is not None:
                print(f"Camera pose vs the run's setup: "
                      f"{pose_difference(np.asarray(saved['T_base_camera']), pose.T_base_camera)}")
        print(f"T_base_camera from {pose.calibration}: camera at {np.round(pose.T_base_camera[:3, 3], 3)} m.")
        yield pose
    finally:
        try:
            if args.camera == "move":
                print("Returning the follower to the parked joints before it goes idle...")
                try:
                    follower.move_to(parked_joints, cfg.move_time_s)
                except Exception as error:
                    print(f"Could not park the follower ({error}).")
                    try:
                        input("The follower will go idle where it is. SUPPORT IT, then press Enter... ")
                    except (EOFError, KeyboardInterrupt):
                        pass
        finally:
            follower.close()
