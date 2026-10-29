"""The manipulator under policy control: ready move, Cartesian goals, gripper, kill switch.

PolicyArm starts like RobotStateReader, in gravity compensation, and is not commanded until
move_to_ready() switches the arm and gripper to position mode. send_tcp() then sends
non-blocking Cartesian goals that the driver solves with its own IK; between goals the
driver holds the last one. kill() puts the arm and gripper back in gravity compensation for
the rest of the session. Every driver call holds one lock.
"""

from __future__ import annotations

import threading
import traceback
from typing import Optional, Tuple

import numpy as np

from .actions import driver_pose6
from .robot_state import RobotState, RobotStateReader


def ready_error(robot: RobotState, joints, gripper: float) -> Tuple[float, float]:
    """(max arm joint error rad, gripper error m) from the ready pose."""
    return (float(np.max(np.abs(robot.arm_q - np.asarray(joints, dtype=np.float64)))),
            float(abs(robot.gripper_q - gripper)))


class PolicyArm(RobotStateReader):
    MODE_NOTE = "gravity compensation until the ready move, then position mode; kill returns to gravity compensation"
    STARTUP_NOTE = "Nothing is commanded until the episode key moves it to the ready pose."
    dry_run = False

    def __init__(self, identity: dict, gripper_limit_margin_m: float):
        super().__init__(identity)
        self.gripper_limit_margin_m = gripper_limit_margin_m
        self.lock = threading.RLock()
        self.killed = False
        self.kill_reason = ""
        self.position_mode = False
        self.gripper_target: Optional[float] = None
        self.gripper_range: Optional[Tuple[float, float]] = None

    def __enter__(self) -> "PolicyArm":
        super().__enter__()
        limit = self.driver.get_joint_limits()[-1]
        self.gripper_range = (limit.position_min + self.gripper_limit_margin_m,
                              limit.position_max - self.gripper_limit_margin_m)
        self.info["gripper_command_range_m"] = list(self.gripper_range)
        return self

    def _gripper_goal(self, target: float) -> float:
        """The target kept gripper_limit_margin_m inside the driver's limits (overshoot past them faults
        the controller to idle)."""
        lo, hi = self.gripper_range
        return float(min(max(target, lo), hi))

    def _check(self) -> None:
        if self.killed:
            raise RuntimeError(f"arm was killed ({self.kill_reason}); restart to command it again")
        if self.driver is None:
            raise RuntimeError("arm is not connected")

    def read(self) -> RobotState:
        with self.lock:
            return super().read()

    def move_to_ready(self, joints, gripper: float, move_time_s: float) -> None:
        import trossen_arm

        with self.lock:
            self._check()
            self.driver.set_arm_modes(trossen_arm.Mode.position)
            self.driver.set_gripper_mode(trossen_arm.Mode.position)
            self.position_mode = True
            self.driver.set_arm_positions([float(q) for q in joints], move_time_s, False)
            self.driver.set_gripper_position(self._gripper_goal(gripper), move_time_s, False)
            self.gripper_target = float(gripper)

    def hold_at(self, joints, goal_time_s: float) -> None:
        """Stop at the given (measured) arm joints; a no-op before the ready move."""
        with self.lock:
            self._check()
            if self.position_mode:
                self.driver.set_arm_positions([float(q) for q in joints], goal_time_s, False)

    def send_tcp(self, T_base_tcp: np.ndarray, goal_time_s: float) -> None:
        """Non-blocking Cartesian goal; raises what the driver raises (e.g. no IK solution)."""
        import trossen_arm

        with self.lock:
            self._check()
            if not self.position_mode:
                raise RuntimeError("the arm is not in position mode (move to the ready pose first)")
            self.driver.set_cartesian_positions(driver_pose6(T_base_tcp), trossen_arm.InterpolationSpace.cartesian,
                                                goal_time_s, False)

    def set_gripper(self, target: float, goal_time_s: float) -> bool:
        """Send a gripper goal if it differs from the last one; True if sent."""
        with self.lock:
            self._check()
            if not self.position_mode:
                raise RuntimeError("the gripper is not in position mode (move to the ready pose first)")
            if self.gripper_target is not None and abs(target - self.gripper_target) < 1e-9:
                return False
            self.driver.set_gripper_position(self._gripper_goal(target), goal_time_s, False)
            self.gripper_target = float(target)
            return True

    def kill(self, reason: str) -> bool:
        """Arm and gripper to gravity compensation for good; True on the first call."""
        with self.lock:
            if self.killed or self.driver is None:
                return False
            self.killed, self.kill_reason = True, reason
            self.position_mode = False
            try:
                self._gravity_compensation()
                # A faulted controller can accept the mode calls; the read raises its error.
                self.driver.get_robot_output()
            except Exception:
                # The controller may have faulted (and braked to idle); clear the error and retry.
                traceback.print_exc()
                try:
                    self.driver.cleanup()
                    self._connect(clear_error=True)
                    self._gravity_compensation()
                except Exception:
                    traceback.print_exc()
                    print("!!! Could not enable gravity compensation; the arm is probably braked (idle).")
                    return True
            print(f"\n!!! KILLED ({reason}): manipulator in gravity compensation. "
                  "Guide it to a safe parked pose by hand.")
            return True

    def __exit__(self, *exc) -> None:
        self.kill("exit")
        super().__exit__(*exc)


class DryRunPolicyArm:
    """Stand-in with the PolicyArm interface: no arm, so no robot state; commands are recorded."""

    dry_run = True

    def __init__(self):
        self.lock = threading.RLock()
        self.killed = False
        self.kill_reason = ""
        self.position_mode = False
        self.gripper_target: Optional[float] = None
        self.last_T: Optional[np.ndarray] = None
        self.info = {"dry_run": True}

    def __enter__(self) -> "DryRunPolicyArm":
        print("Policy DRY RUN: no arm is connected, so there is no robot state and no policy action.")
        return self

    def __exit__(self, *_exc) -> None:
        pass

    def read(self) -> None:
        return None

    def move_to_ready(self, _joints, gripper: float, _move_time_s: float) -> None:
        if not self.killed:
            self.position_mode, self.gripper_target = True, float(gripper)

    def hold_at(self, _joints, _goal_time_s: float) -> None:
        pass

    def send_tcp(self, T_base_tcp: np.ndarray, _goal_time_s: float) -> None:
        if not self.killed:
            self.last_T = np.array(T_base_tcp, dtype=np.float64)

    def set_gripper(self, target: float, _goal_time_s: float) -> bool:
        if self.killed or (self.gripper_target is not None and abs(target - self.gripper_target) < 1e-9):
            return False
        self.gripper_target = float(target)
        return True

    def kill(self, reason: str) -> bool:
        if self.killed:
            return False
        self.killed, self.kill_reason = True, reason
        print(f"\n!!! KILLED ({reason}) [dry run]")
        return True
