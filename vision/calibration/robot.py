"""Manipulator in gravity compensation with the gripper held closed; measured state only.

Same driver setup as trossen_arm/demos/python/gravity_compensation_gripper_closed.py:
arm joints in external_effort mode with zero effort (gravity compensation), gripper in
external_effort mode with a constant closing effort. No position/Cartesian commands are sent.
"""

import time
from dataclasses import dataclass

import numpy as np
import trossen_arm

from .transforms import T_from_rvec


@dataclass
class RobotState:
    host_time_s: float
    header_id: int
    header_timestamp_us: int
    arm_positions: np.ndarray     # rad
    arm_velocities: np.ndarray    # rad/s
    gripper_position: float       # m
    cartesian: np.ndarray         # [x, y, z, rx, ry, rz] of the EE in the base frame (m, angle-axis rad)

    @property
    def T_base_ee(self) -> np.ndarray:
        return T_from_rvec(self.cartesian[3:], self.cartesian[:3])

    def to_json(self) -> dict:
        return {"host_time_s": self.host_time_s, "header_id": self.header_id,
                "header_timestamp_us": self.header_timestamp_us,
                "arm_positions_rad": self.arm_positions.tolist(),
                "arm_velocities_rad_s": self.arm_velocities.tolist(),
                "gripper_position_m": self.gripper_position,
                "cartesian_xyz_m_rotvec_rad": self.cartesian.tolist(),
                "T_base_ee": self.T_base_ee.tolist()}


class GravityCompManipulator:
    def __init__(self, robot_cfg: dict):
        self.cfg = robot_cfg
        self.driver = None
        self.info = {}

    def connect(self, clear_error: bool = False) -> None:
        c = self.cfg
        self.driver = trossen_arm.TrossenArmDriver()
        self.driver.configure(getattr(trossen_arm.Model, c["model"]),
                              getattr(trossen_arm.StandardEndEffector, c["end_effector"]),
                              c["ip"], clear_error)

    def __enter__(self) -> "GravityCompManipulator":
        c = self.cfg
        self.connect()
        n_arm = self.driver.get_num_joints() - 1
        self.enable_gravity_compensation(c["gripper_close_effort"])
        self.info = {"ip": c["ip"], "model": c["model"], "end_effector": c["end_effector"],
                     "gripper_close_effort_n": c["gripper_close_effort"], "num_arm_joints": n_arm,
                     "driver_version": self.driver.get_driver_version(),
                     "controller_version": self.driver.get_controller_version(),
                     "mode": "gravity compensation (arm external_effort = 0), gripper external_effort closed",
                     "T_base_ee_source": "driver get_robot_output().cartesian.positions (FK of measured joints)"}
        print(f"Manipulator {c['ip']}: gravity compensation on, gripper held closed. Guide it by hand.")
        return self

    def enable_gravity_compensation(self, gripper_effort: float = 0.0) -> None:
        """Arm joints in external_effort mode with zero effort; gripper held with gripper_effort (N)."""
        n_arm = self.driver.get_num_joints() - 1
        self.driver.set_arm_modes(trossen_arm.Mode.external_effort)
        self.driver.set_arm_external_efforts([0.0] * n_arm, 0.0, False)
        self.driver.set_gripper_mode(trossen_arm.Mode.external_effort)
        self.driver.set_gripper_external_effort(gripper_effort, 0.0, False)

    def __exit__(self, *_exc) -> None:
        if self.driver is None:
            return
        try:
            input("\nThe manipulator will go limp (idle). SUPPORT THE ARM, then press Enter... ")
        except (EOFError, KeyboardInterrupt):
            pass
        self.driver.cleanup()
        self.driver = None

    def read_state(self) -> RobotState:
        out = self.driver.get_robot_output()
        return RobotState(host_time_s=time.time(), header_id=int(out.header.id),
                          header_timestamp_us=int(out.header.timestamp),
                          arm_positions=np.array(out.joint.arm.positions, dtype=np.float64),
                          arm_velocities=np.array(out.joint.arm.velocities, dtype=np.float64),
                          gripper_position=float(out.joint.gripper.position),
                          cartesian=np.array(out.cartesian.positions, dtype=np.float64))

    def wait_still(self, window_s: float, vel_thresh: float, range_thresh: float, rate_hz: float = 100.0):
        """Sample for window_s; still if every joint speed and excursion stays below threshold."""
        states = []
        end = time.time() + window_s
        while time.time() < end:
            states.append(self.read_state())
            time.sleep(1.0 / rate_hz)
        motion = joint_motion(states)
        still = motion["max_joint_drift_rad_s"] < vel_thresh and motion["max_joint_range_rad"] < range_thresh
        return still, states, {**motion, "window_s": window_s, "num_states": len(states)}


def joint_motion(states) -> dict:
    """Stillness from measured positions over a window.

    The driver's per-sample joint velocities are too noisy for this (peaks of ~0.1 rad/s while
    positions stay within ~1 mrad), so motion is judged from the position range and the
    least-squares drift rate of each joint; the raw peak velocity is reported for reference only.
    """
    q = np.array([s.arm_positions for s in states])
    t = np.array([s.host_time_s for s in states])
    drift = np.zeros(q.shape[1])
    if len(states) >= 2 and np.ptp(t) > 0:
        tc = t - t.mean()
        drift = (tc @ (q - q.mean(axis=0))) / (tc @ tc)
    return {"max_joint_range_rad": float(np.ptp(q, axis=0).max()),
            "max_joint_drift_rad_s": float(np.abs(drift).max()),
            "max_raw_joint_vel_rad_s": float(np.abs([s.arm_velocities for s in states]).max())}
