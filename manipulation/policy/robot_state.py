"""Measured state of the manipulator, read from the Trossen driver without commanding it.

The arm and gripper are put in gravity compensation (external effort 0), so they can be
guided by hand while the observations are watched. Each RobotState comes from one
get_robot_output() call: joints, gripper and the driver's tool pose share a timestamp.
The driver tool pose is the training TCP site (sim2real/experiments/TCP_alignment).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .observations import make_T, rotvec_to_matrix

ROBOT_KEYS = ("ip", "model", "end_effector")


@dataclass(frozen=True)
class RobotState:
    read_at: float              # time.monotonic() right after the driver call returned
    header_id: int
    header_timestamp_us: int
    arm_q: np.ndarray           # rad, joint_0..joint_5
    arm_qd: np.ndarray          # rad/s
    gripper_q: float            # m, driver gripper joint (= each sim carriage joint)
    gripper_qd: float           # m/s
    tcp_xyz_rotvec: np.ndarray  # driver tool pose in the base frame: m, then angle-axis rad

    @property
    def T_base_tcp(self) -> np.ndarray:
        return make_T(rotvec_to_matrix(self.tcp_xyz_rotvec[3:]), self.tcp_xyz_rotvec[:3])

    def to_json(self) -> dict:
        return {"header_id": self.header_id, "header_timestamp_us": self.header_timestamp_us,
                "arm_q_rad": self.arm_q.tolist(), "arm_qd_rad_s": self.arm_qd.tolist(),
                "gripper_q_m": self.gripper_q, "gripper_qd_m_s": self.gripper_qd,
                "tcp_xyz_m_rotvec_rad": self.tcp_xyz_rotvec.tolist()}


def resolve_robot_identity(override: dict, calibration_json: dict,
                           calibration_config: Optional[Path] = None) -> dict:
    """ip / model / end_effector: the policy config, else the calibration session's robot,
    else vision/calibration/config.yaml."""
    session_robot = calibration_json.get("session_meta", {}).get("robot", {}) or {}
    fallback = {}
    if any(not (override.get(k) or session_robot.get(k)) for k in ROBOT_KEYS):
        from calibration.board import load_config

        fallback = load_config(calibration_config).get("robot", {}) or {}
    robot = {}
    for key in ROBOT_KEYS:
        robot[key] = override.get(key) or session_robot.get(key) or fallback.get(key)
        if not robot[key]:
            raise ValueError(f"robot.{key} is not set in the policy config, the calibration session, "
                             "or the calibration config")
    if session_robot.get("end_effector") and robot["end_effector"] != session_robot["end_effector"]:
        print(f"WARNING: end effector {robot['end_effector']} differs from the calibration's "
              f"{session_robot['end_effector']}")
    return robot


class RobotStateReader:
    """Context manager: connect, gravity compensation on entry; support prompt and idle on exit."""

    def __init__(self, identity: dict):
        self.identity = dict(identity)
        self.driver = None
        self.info: dict = {}

    def __enter__(self) -> "RobotStateReader":
        import trossen_arm

        c = self.identity
        self.driver = trossen_arm.TrossenArmDriver()
        self.driver.configure(getattr(trossen_arm.Model, c["model"]),
                              getattr(trossen_arm.StandardEndEffector, c["end_effector"]), c["ip"], False)
        try:
            n_arm = self.driver.get_num_joints() - 1
            if n_arm != 6:
                raise RuntimeError(f"expected 6 arm joints, the driver reports {n_arm}")
            self.driver.set_arm_modes(trossen_arm.Mode.external_effort)
            self.driver.set_arm_external_efforts([0.0] * n_arm, 0.0, False)
            self.driver.set_gripper_mode(trossen_arm.Mode.external_effort)
            self.driver.set_gripper_external_effort(0.0, 0.0, False)
        except Exception:
            self.driver.cleanup()
            self.driver = None
            raise
        self.info = {**c, "driver_version": self.driver.get_driver_version(),
                     "controller_version": self.driver.get_controller_version(),
                     "mode": "gravity compensation: arm and gripper external_effort 0, never commanded"}
        print(f"Manipulator {c['ip']}: gravity compensation on (arm and gripper). It is never commanded; "
              "guide it by hand.")
        return self

    def read(self) -> RobotState:
        out = self.driver.get_robot_output()
        read_at = time.monotonic()
        return RobotState(read_at=read_at, header_id=int(out.header.id),
                          header_timestamp_us=int(out.header.timestamp),
                          arm_q=np.array(out.joint.arm.positions, dtype=np.float64),
                          arm_qd=np.array(out.joint.arm.velocities, dtype=np.float64),
                          gripper_q=float(out.joint.gripper.position),
                          gripper_qd=float(out.joint.gripper.velocity),
                          tcp_xyz_rotvec=np.array(out.cartesian.positions, dtype=np.float64))

    def __exit__(self, *_exc) -> None:
        if self.driver is None:
            return
        try:
            input("\nThe manipulator will go limp (idle). SUPPORT THE ARM, then press Enter... ")
        except (EOFError, KeyboardInterrupt):
            pass
        self.driver.cleanup()
        self.driver = None
