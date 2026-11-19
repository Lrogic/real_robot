"""Trossen WidowX AI base arm from TrossenRobotics/trossen_arm_mujoco."""

from dataclasses import replace

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

# Same geometry as the base arm, with position-mode gains and effort limits
# matched to the measured controller.
TROSSEN_WXAI_BASE_REAL = replace(
  TROSSEN_WXAI_BASE,
  name="trossen_wxai_base_real",
  xml_path=ASSETS_DIR / "robots" / "trossen_wxai_base" / "wxai" / "wxai_base_real_robot.xml",
)

# The real-robot model with the arm joints' dry friction fitted to single-joint
# step tests on the arm (sim2real/experiments/joint_step).
TROSSEN_WXAI_BASE_REAL_FRICTION = replace(
  TROSSEN_WXAI_BASE_REAL,
  name="trossen_wxai_base_real_friction",
  xml_path=ASSETS_DIR
  / "robots"
  / "trossen_wxai_base"
  / "wxai"
  / "wxai_base_real_robot_friction.xml",
)

# The friction model with the gripper matched to the real one
# (sim2real/experiments/gripper_force): the gripper closes on an object at the driver's
# 83.3 N effort limit, the finger pads and the carriage mimic are stiffened so that squeeze
# neither sinks the fingers into the object nor pulls them apart, and elliptic friction
# cones with impratio 10 (as the Menagerie WXAI, ALOHA and Robotiq models use) stop a held
# object creeping out of the grip. Pair it with task action.gripper_goal_time_s 0.5, the
# real control.gripper_goal_time_s; without it the gripper slams shut in ~20 ms.
TROSSEN_WXAI_BASE_REAL_FRICTION_GRIP = replace(
  TROSSEN_WXAI_BASE_REAL_FRICTION,
  name="trossen_wxai_base_real_friction_grip",
  xml_path=ASSETS_DIR
  / "robots"
  / "trossen_wxai_base"
  / "wxai"
  / "wxai_base_real_robot_friction_grip.xml",
  impratio=10.0,
  cone="elliptic",
)
