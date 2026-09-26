"""Robot embodiment description and conversion to an mjlab entity."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import mujoco

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

@dataclass(frozen=True)
class IkDefaults:
  damping: float = 0.05
  max_dq: float = 0.5
  position_weight: float = 1.0
  orientation_weight: float = 1.0
  joint_limit_weight: float = 1.0
  posture_weight: float = 0.0


@dataclass(frozen=True)
class EmbodimentCfg:
  """Everything the task needs to know about a robot arm + parallel gripper.

  Gripper joint positions follow the convention 0 = closed and ``gripper_open``
  = fully open for both carriage joints.
  """

  name: str
  xml_path: Path
  arm_joint_names: tuple[str, ...]
  gripper_joint_names: tuple[str, str]
  """(left, right) carriage joints. Observations list them in this order."""
  gripper_actuator_joint: str
  """Joint driven by the single gripper actuator; the other follows by equality."""
  tcp_site: str
  finger_bodies: tuple[str, str]
  """(left, right) bodies whose contacts with the object define a grasp."""
  finger_outward_axes: tuple[tuple[float, float, float], tuple[float, float, float]]
  """Per finger, the local-frame axis the object's contact force on that finger
  points along during a grasp (away from the gripper centre)."""
  ready_joint_pos: dict[str, float]
  gripper_open: float = 0.044
  gripper_closed: float = 0.0
  strip_bodies: tuple[str, ...] = ()
  gravity_compensation: bool = True
  """Cancel link gravity through the actuators (bounded by their force limits),
  as the real arm's controller does. Without it the low-gain position servos
  sag by more than one delta-EE action step."""
  integrator: Literal["euler", "implicitfast"] = "implicitfast"
  impratio: float = 1.0
  cone: Literal["pyramidal", "elliptic"] = "pyramidal"
  ik: IkDefaults = field(default_factory=IkDefaults)

  @property
  def obs_joint_names(self) -> tuple[str, ...]:
    return (*self.arm_joint_names, *self.gripper_joint_names)


def load_spec(emb: EmbodimentCfg) -> mujoco.MjSpec:
  """Load the robot XML, drop stripped bodies and any keyframes."""
  spec = mujoco.MjSpec.from_file(str(emb.xml_path))  # mjlab builds its own init_state keyframe and uses only the first key. Keys
  # must go before any body deletion, otherwise MuJoCo resurrects them on compile.
  for key in list(spec.keys):
    spec.delete(key)
  for name in emb.strip_bodies:
    body = spec.body(name)
    if body is None:
      raise ValueError(f"{emb.name}: body '{name}' to strip not found")
    spec.delete(body)
  if emb.gravity_compensation:
    for body in spec.bodies[1:]:
      body.gravcomp = 1.0
    for joint in spec.joints:
      joint.actgravcomp = True
  # Solver options are set on the scene (see EmbodimentCfg); MjSpec.attach would
  # drop them anyway.
  spec.option.impratio = 1.0
  spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
  return spec


def build_entity_cfg(emb: EmbodimentCfg) -> EntityCfg:
  return EntityCfg(
    spec_fn=lambda: load_spec(emb),
    init_state=EntityCfg.InitialStateCfg(
      joint_pos=dict(emb.ready_joint_pos),
      joint_vel={".*": 0.0},
    ),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        XmlActuatorCfg(
          target_names_expr=(*emb.arm_joint_names, emb.gripper_actuator_joint)
        ),
      ),
    ),
  )
