"""Driver connection and gravity compensation shared by hardware experiments."""

from __future__ import annotations


def robot_identity(calibration: dict) -> dict:
    robot = calibration.get("robot") or {}
    identity = {}
    for key in ("ip", "model", "end_effector"):
        value = robot.get(key)
        if not value:
            raise ValueError(
                f"robot.{key} is empty in the calibration config; set it before measuring"
            )
        identity[key] = value
    return identity


def end_effector_record(ee) -> dict:
    return {
        "offset_finger_left_m": float(ee.offset_finger_left),
        "offset_finger_right_m": float(ee.offset_finger_right),
        "pitch_circle_radius_m": float(ee.pitch_circle_radius),
        "t_flange_tool_m_rotvec_rad": [float(v) for v in ee.t_flange_tool],
        "finger_offset_meaning": (
            "offset_finger_left/right are the palm-centre to carriage-centre "
            "offsets with the fingers at the closed configuration, in metres. "
            "They are not the gripper joint coordinate."
        ),
    }


def hold_arm_with_gravity_compensation(driver) -> int:
    """Arm joints at zero external effort. Returns the number of arm joints.

    The gripper mode is left unchanged.
    """
    import trossen_arm

    n_arm = driver.get_num_joints() - 1
    driver.set_arm_modes(trossen_arm.Mode.external_effort)
    driver.set_arm_external_efforts([0.0] * n_arm, 0.0, False)
    return n_arm


def release_to_idle(driver) -> None:
    """Drop the arm to idle. The operator has to be supporting it."""
    if driver is None:
        return
    try:
        input("\nThe manipulator will go idle. SUPPORT THE ARM, then press Enter... ")
    except (EOFError, KeyboardInterrupt):
        pass
    driver.cleanup()
