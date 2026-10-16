"""Trossen WidowX AI base arm from TrossenRobotics/trossen_arm_mujoco."""

from tabletop_sim import ASSETS_DIR
from tabletop_sim.embodiments.base import EmbodimentCfg, StiffnessDampingTargets
from tabletop_sim.embodiments.menagerie_wxai import ARM_JOINTS, GRIPPER_JOINTS, MENAGERIE_WXAI

# Position actuators with kp and kv; the joints carry no damping.
ACTUATORS = (*ARM_JOINTS, "left_gripper")

TROSSEN_WXAI_BASE = EmbodimentCfg(
  name="trossen_wxai_base",
  xml_path=ASSETS_DIR / "robots" / "trossen_wxai_base" / "wxai" / "wxai_base.xml",
  arm_joint_names=ARM_JOINTS,
  gripper_joint_names=GRIPPER_JOINTS,
  gripper_actuator_joint="left_carriage_joint",
  tcp_site="ee_site",
  # Finger collision geoms live directly on the carriage bodies in this model.
  finger_bodies=("carriage_left", "carriage_right"),
  finger_outward_axes=((0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
  ready_joint_pos=dict(MENAGERIE_WXAI.ready_joint_pos),
  stiffness_damping=StiffnessDampingTargets(
    stiffness_actuators=ACTUATORS,
    damping_source="actuator",
    damping_actuators=ACTUATORS,
  ),
  integrator="implicitfast",
)
