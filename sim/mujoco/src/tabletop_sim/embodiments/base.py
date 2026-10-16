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
class StiffnessDampingTargets:
  """Model parameters scaled by stiffness/damping domain randomization.

  kp is scaled on the position actuators in ``stiffness_actuators``. Damping is
  scaled on the ``dof_damping`` of ``damping_joints`` when ``damping_source`` is
  ``"joint"``, or on the kv of the position actuators in ``damping_actuators``
  when it is ``"actuator"``. The other damping mechanism is never touched.
  """

  stiffness_actuators: tuple[str, ...]
  damping_source: Literal["joint", "actuator"]
  damping_joints: tuple[str, ...] = ()
  damping_actuators: tuple[str, ...] = ()

  def __post_init__(self) -> None:
    if not self.stiffness_actuators:
      raise ValueError("stiffness_actuators must not be empty")
    if self.damping_source == "joint":
      if not self.damping_joints or self.damping_actuators:
        raise ValueError(
          "damping_source='joint' needs damping_joints and no damping_actuators"
        )
    elif self.damping_source == "actuator":
      if not self.damping_actuators or self.damping_joints:
        raise ValueError(
          "damping_source='actuator' needs damping_actuators and no damping_joints"
        )
    else:
      raise ValueError(
        f"damping_source must be 'joint' or 'actuator', got {self.damping_source!r}"
      )


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
  stiffness_damping: StiffnessDampingTargets
  """Where kp and damping live in this model, for their randomization."""
  finger_collision_bodies: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())
  """(left, right) bodies that move with each finger and whose collidable geoms
  are that finger's object contact surfaces (explicit friction pairs). Empty
  means the finger's ``finger_bodies`` entry alone."""
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
  soft_joint_pos_limit_factor: float = 0.85
  """Fraction of each joint's model range that counts as the soft limit, used by
  the joint-limit penalty and the entity's soft limits."""

  @property
  def obs_joint_names(self) -> tuple[str, ...]:
    return (*self.arm_joint_names, *self.gripper_joint_names)

  @property
  def finger_contact_bodies(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
    left, right = self.finger_collision_bodies
    return (left or (self.finger_bodies[0],), right or (self.finger_bodies[1],))


def finger_collision_geoms(
  spec: mujoco.MjSpec, emb: EmbodimentCfg
) -> tuple[tuple[mujoco.MjsGeom, ...], tuple[mujoco.MjsGeom, ...]]:
  """(left, right) collidable geoms (contype or conaffinity set) on each
  finger's ``finger_contact_bodies``. Unnamed ones are named
  ``<body>_collision_<k>``, ``k`` counting the body's collidable geoms in
  order, so the same XML always yields the same names."""
  taken = {g.name for g in spec.geoms if g.name}
  out: list[tuple[mujoco.MjsGeom, ...]] = []
  for bodies in emb.finger_contact_bodies:
    geoms: list[mujoco.MjsGeom] = []
    for body_name in bodies:
      body = spec.body(body_name)
      if body is None:
        raise ValueError(f"{emb.name}: finger body '{body_name}' not found")
      collidable = [g for g in body.geoms if g.contype or g.conaffinity]
      for k, g in enumerate(collidable):
        if not g.name:
          name = f"{body_name}_collision_{k}"
          if name in taken:
            raise ValueError(f"{emb.name}: geom name '{name}' already in use")
          g.name = name
          taken.add(name)
      geoms.extend(collidable)
    if not geoms:
      raise ValueError(f"{emb.name}: finger bodies {bodies} have no collidable geoms")
    out.append(tuple(geoms))
  return out[0], out[1]


def load_spec(emb: EmbodimentCfg, name_finger_geoms: bool = False) -> mujoco.MjSpec:
  """Load the robot XML, drop stripped bodies and any keyframes. With
  ``name_finger_geoms``, also name the finger collision geoms (see
  :func:`finger_collision_geoms`) so contact pairs can reference them."""
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
  if name_finger_geoms:
    finger_collision_geoms(spec, emb)
  return spec


def build_entity_cfg(emb: EmbodimentCfg, name_finger_geoms: bool = False) -> EntityCfg:
  return EntityCfg(
    spec_fn=lambda: load_spec(emb, name_finger_geoms),
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
      soft_joint_pos_limit_factor=emb.soft_joint_pos_limit_factor,
    ),
  )
