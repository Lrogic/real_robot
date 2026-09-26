"""Trossen WidowX AI base arm from TrossenRobotics/trossen_arm_mujoco."""

from tabletop_sim import ASSETS_DIR
from tabletop_sim.embodiments.base import EmbodimentCfg
from tabletop_sim.embodiments.menagerie_wxai import ARM_JOINTS, MENAGERIE_WXAI

TROSSEN_WXAI_BASE = EmbodimentCfg(
  name="trossen_wxai_base",
  xml_path=ASSETS_DIR / "robots" / "trossen_wxai_base" / "wxai" / "wxai_base.xml",
  arm_joint_names=ARM_JOINTS,
  gripper_joint_names=("left_carriage_joint", "right_carriage_joint"),
  gripper_actuator_joint="left_carriage_joint",
  tcp_site="ee_site",
  # Finger collision geoms live directly on the carriage bodies in this model.
  finger_bodies=("carriage_left", "carriage_right"),
  finger_outward_axes=((0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
  ready_joint_pos=dict(MENAGERIE_WXAI.ready_joint_pos),
  integrator="implicitfast",
)
