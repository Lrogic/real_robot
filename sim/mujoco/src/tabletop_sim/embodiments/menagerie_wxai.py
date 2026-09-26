"""Trossen WidowX AI from MuJoCo Menagerie (system-identified follower arm).

The follower's D405 camera mount and camera are stripped to get the base arm.
"""

from tabletop_sim import ASSETS_DIR
from tabletop_sim.embodiments.base import EmbodimentCfg

ARM_JOINTS = tuple(f"joint_{i}" for i in range(6))

MENAGERIE_WXAI = EmbodimentCfg(
  name="menagerie_wxai",
  xml_path=ASSETS_DIR / "robots" / "trossen_wxai" / "wxai_follower.xml",
  strip_bodies=("camera_mount_d405",),
  arm_joint_names=ARM_JOINTS,
  gripper_joint_names=("left_carriage_joint", "right_carriage_joint"),
  gripper_actuator_joint="right_carriage_joint",
  tcp_site="ee_gripper_link",
  finger_bodies=("gripper_left", "gripper_right"),
  finger_outward_axes=((0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
  ready_joint_pos={
    "joint_0": 0.0,
    "joint_1": 1.38,
    "joint_2": 1.04,
    "joint_3": -1.26,
    "joint_4": 0.0,
    "joint_5": 0.0,
    "left_carriage_joint": 0.026,
    "right_carriage_joint": 0.026,
  },
  integrator="euler",
  impratio=10.0,
  cone="elliptic",
)
