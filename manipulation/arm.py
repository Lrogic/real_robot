"""Manipulator for the follow test: position control of the tool, kill switch to gravity compensation.

Reuses calibration.robot.GravityCompManipulator for the driver connection, measured state,
gravity compensation, and the "support the arm" prompt before going idle at exit.
"""

import threading
import time
import traceback
from typing import Optional

import numpy as np
import trossen_arm

from calibration.robot import GravityCompManipulator, RobotState
from calibration.transforms import rvec_from_R

from .grasp import joint_margin_problem, pose_to_driver


class ManipulatorArm(GravityCompManipulator):
    """Starts in gravity compensation (guide it by hand); engage() locks it in position mode.

    kill() switches back to gravity compensation for good.
    """

    def __init__(self, robot_cfg: dict, gripper_open_margin_m: float, gripper_open_time_s: float,
                 gripper_close_effort: float):
        super().__init__({**robot_cfg, "gripper_close_effort": gripper_close_effort})
        self.gripper_open_margin_m = gripper_open_margin_m
        self.gripper_open_time_s = gripper_open_time_s
        self.gripper_open_m: Optional[float] = None
        self.lock = threading.RLock()
        self.killed = False
        self.kill_reason = ""
        self.engaged = False

    def __enter__(self) -> "ManipulatorArm":
        c = self.cfg
        self.connect()
        try:
            self.driver.set_arm_modes(trossen_arm.Mode.external_effort)
            self.driver.set_arm_external_efforts([0.0] * (self.driver.get_num_joints() - 1), 0.0, False)
            limits = self.driver.get_joint_limits()
            gripper_open = limits[-1].position_max - self.gripper_open_margin_m
            self.gripper_open_m = gripper_open
            self.driver.set_gripper_mode(trossen_arm.Mode.position)
            self.driver.set_gripper_position(gripper_open, self.gripper_open_time_s, False)
        except BaseException:
            self.kill("startup failed")
            self.__exit__()
            raise
        self.joint_limits = np.array([[lim.position_min, lim.position_max] for lim in limits[:-1]])
        self.info = {"ip": c["ip"], "model": c["model"], "end_effector": c["end_effector"],
                     "t_flange_tool": list(self.driver.get_end_effector().t_flange_tool),
                     "arm_joint_limits_rad": self.joint_limits.tolist(),
                     "gripper_open_m": gripper_open,
                     "driver_version": self.driver.get_driver_version(),
                     "controller_version": self.driver.get_controller_version()}
        print(f"Manipulator {c['ip']}: GRAVITY COMPENSATION, opening the gripper to {gripper_open * 1000:.1f} mm.\n"
              "Guide the arm by hand to a ready pose (elbow bent, gripper level, well above the table).\n"
              "It locks in place when you confirm; nothing is commanded before that.")
        return self

    def engage(self, min_joint_margin_deg: float) -> Optional[str]:
        """Lock the current pose in position mode; return a refusal message instead if unsafe."""
        with self.lock:
            if self.killed or self.driver is None:
                return "arm was killed; restart to move it again"
            if self.engaged:
                return None
            joints = np.array(self.driver.get_arm_positions())
            problem = joint_margin_problem(joints, self.joint_limits, min_joint_margin_deg)
            if problem is not None:
                return f"not starting: {problem}. Guide the arm to a less folded pose."
            self.driver.set_arm_modes(trossen_arm.Mode.position)
            self.driver.set_arm_positions(list(joints), 0.5, True)
            self.engaged = True
            print(f"Manipulator locked at joints {np.degrees(joints).round(1)} deg")
            return None

    def __exit__(self, *_exc) -> None:
        self.kill("exit")
        super().__exit__()

    def measured_state(self) -> Optional[RobotState]:
        with self.lock:
            if self.driver is None:
                return None
            return self.read_state()

    def open_gripper(self) -> None:
        with self.lock:
            if self.killed or self.driver is None or self.gripper_open_m is None:
                return
            self.driver.set_gripper_mode(trossen_arm.Mode.position)
            self.driver.set_gripper_position(self.gripper_open_m, self.gripper_open_time_s, False)

    def close_gripper(self) -> None:
        with self.lock:
            if self.killed or self.driver is None:
                return
            effort = float(self.cfg["gripper_close_effort"])
            self.driver.set_gripper_mode(trossen_arm.Mode.external_effort)
            self.driver.set_gripper_external_effort(effort, 0.0, False)

    def send_tool_pose(self, T_base_tool: np.ndarray, goal_time: float) -> None:
        """Non-blocking Cartesian goal (driver IK); a no-op once killed."""
        with self.lock:
            if self.killed or not self.engaged or self.driver is None:
                return
            self.driver.set_cartesian_positions(pose_to_driver(T_base_tool), trossen_arm.InterpolationSpace.cartesian,
                                                goal_time, False)

    def kill(self, reason: str) -> None:
        """Stop all control: arm and gripper to gravity compensation (never idle, never dropped)."""
        with self.lock:
            if self.killed or self.driver is None:
                return
            self.killed, self.kill_reason = True, reason
            try:
                self.enable_gravity_compensation(0.0)
            except Exception:
                # The controller may have faulted (and braked to idle); clear the error and retry.
                traceback.print_exc()
                try:
                    self.driver.cleanup()
                    self.connect(clear_error=True)
                    self.enable_gravity_compensation(0.0)
                except Exception:
                    traceback.print_exc()
                    print("!!! Could not enable gravity compensation; the arm is probably braked (idle).")
                    return
            print(f"\n!!! KILLED ({reason}): manipulator in gravity compensation. "
                  "Guide it to a safe parked pose by hand.")


class DryRunArm:
    """Stand-in with the ManipulatorArm interface: records goals, never touches hardware."""

    def __init__(self):
        self.lock = threading.RLock()
        self.killed = False
        self.kill_reason = ""
        self.last_T: Optional[np.ndarray] = None
        self.joint_limits = None
        self.info = {"dry_run": True}
        self.engaged = False

    def engage(self, _min_joint_margin_deg: float) -> Optional[str]:
        if self.killed:
            return "arm was killed; restart to move it again"
        self.engaged = True
        return None

    def __enter__(self) -> "DryRunArm":
        print("Manipulation DRY RUN: goals are drawn and logged, no arm is connected.")
        return self

    def __exit__(self, *_exc) -> None:
        pass

    def measured_state(self) -> Optional[RobotState]:
        """The last command, reported as if tracked perfectly; joints are unknown (NaN)."""
        if self.last_T is None:
            return None
        cartesian = np.concatenate([self.last_T[:3, 3], rvec_from_R(self.last_T[:3, :3])])
        return RobotState(host_time_s=time.time(), header_id=0, header_timestamp_us=0,
                          arm_positions=np.full(6, np.nan), arm_velocities=np.full(6, np.nan),
                          gripper_position=float("nan"), cartesian=cartesian)

    def send_tool_pose(self, T_base_tool: np.ndarray, goal_time: float) -> None:
        if not self.killed:
            self.last_T = T_base_tool.copy()

    def open_gripper(self) -> None:
        print("[dry run] gripper open")

    def close_gripper(self) -> None:
        print("[dry run] gripper close")

    def kill(self, reason: str) -> None:
        if not self.killed:
            self.killed, self.kill_reason = True, reason
            print(f"\n!!! KILLED ({reason}) [dry run]")
