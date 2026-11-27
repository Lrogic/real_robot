import json
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from tabletop_sim.embodiments import EMBODIMENTS, StiffnessDampingTargets, get_embodiment
from tabletop_sim.embodiments.base import load_spec


@pytest.mark.parametrize("name", sorted(EMBODIMENTS))
def test_embodiment_compiles(name):
  emb = get_embodiment(name)
  model = load_spec(emb).compile()
  assert model.nkey == 0
  for joint in emb.obs_joint_names:
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint) >= 0
  assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, emb.tcp_site) >= 0
  for body in emb.finger_bodies:
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body) >= 0
  for joint in emb.ready_joint_pos:
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint) >= 0
  if emb.gravity_compensation:
    assert (model.body_gravcomp[1:] == 1.0).all()


@pytest.mark.parametrize("name", sorted(EMBODIMENTS))
def test_stiffness_damping_targets_match_model(name):
  emb = get_embodiment(name)
  t = emb.stiffness_damping
  model = load_spec(emb).compile()

  def actuator(n):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    assert i >= 0, n
    return i

  def dof(n):
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
    assert j >= 0, n
    return model.jnt_dofadr[j]

  # kp only on position servos: gain = kp, bias = [0, -kp, -kv].
  for n in t.stiffness_actuators:
    i = actuator(n)
    assert model.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_FIXED
    assert model.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_AFFINE
    kp = model.actuator_gainprm[i, 0]
    assert kp > 0 and model.actuator_biasprm[i, 0] == 0 and model.actuator_biasprm[i, 1] == -kp
  # Every actuator is a stiffness target.
  assert sorted(actuator(n) for n in t.stiffness_actuators) == list(range(model.nu))
  arm_and_gripper = {dof(n) for n in emb.obs_joint_names}
  if t.damping_source == "joint":
    # Both carriages, including the one only coupled by the equality.
    assert set(emb.gripper_joint_names) <= set(t.damping_joints)
    assert {dof(n) for n in t.damping_joints} == arm_and_gripper
    assert all(model.dof_damping[dof(n)] > 0 for n in t.damping_joints)
    # The system-identified follower is kp-only. The real-robot copy also has
    # actuator kv; domain randomization still scales the joint damping.
    if name == "menagerie_wxai":
      assert (model.actuator_biasprm[:, 2] == 0).all()
  else:
    assert t.damping_source == "actuator"
    assert sorted(actuator(n) for n in t.damping_actuators) == list(range(model.nu))
    assert all(model.actuator_biasprm[actuator(n), 2] < 0 for n in t.damping_actuators)
    assert all(model.dof_damping[d] == 0 for d in arm_and_gripper)


def test_embodiment_damping_sources():
  assert get_embodiment("menagerie_wxai").stiffness_damping.damping_source == "joint"
  assert get_embodiment("trossen_wxai_base").stiffness_damping.damping_source == "actuator"
  real = get_embodiment("trossen_wxai_base_real")
  assert real.stiffness_damping.damping_source == "actuator"
  assert real.xml_path.name == "wxai_base_real_robot.xml"
  friction = get_embodiment("trossen_wxai_base_real_friction")
  assert friction.xml_path.name == "wxai_base_real_robot_friction.xml"
  assert replace(friction, name=real.name, xml_path=real.xml_path) == real
  follower = get_embodiment("menagerie_wxai_real")
  assert follower.stiffness_damping.damping_source == "joint"
  assert follower.xml_path.name == "wxai_follower_real_robot.xml"
  assert follower.strip_bodies == ("camera_mount_d405",)


@pytest.mark.parametrize(
  "kwargs",
  [
    dict(stiffness_actuators=(), damping_source="joint", damping_joints=("a",)),
    dict(stiffness_actuators=("a",), damping_source="joint"),
    dict(stiffness_actuators=("a",), damping_source="actuator"),
    dict(stiffness_actuators=("a",), damping_source="joint", damping_joints=("a",),
         damping_actuators=("a",)),
    dict(stiffness_actuators=("a",), damping_source="auto", damping_joints=("a",)),
  ],
)
def test_stiffness_damping_targets_validated(kwargs):
  with pytest.raises(ValueError):
    StiffnessDampingTargets(**kwargs)


def test_menagerie_camera_mount_removed():
  emb = get_embodiment("menagerie_wxai")
  model = load_spec(emb).compile()
  assert model.ncam == 0
  assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "camera_mount_d405") == -1


def test_unknown_embodiment():
  with pytest.raises(KeyError):
    get_embodiment("ur5")


def test_friction_model_changes_only_the_arm_joint_friction():
  base = mujoco.MjModel.from_xml_path(str(get_embodiment("trossen_wxai_base_real").xml_path))
  fric = mujoco.MjModel.from_xml_path(str(get_embodiment("trossen_wxai_base_real_friction").xml_path))
  arm = [fric.jnt_dofadr[mujoco.mj_name2id(fric, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{i}")] for i in range(6)]
  assert fric.dof_frictionloss[arm].tolist() == pytest.approx([0.64, 1.85, 1.92, 0.10, 0.10, 0.16])
  rest = [i for i in range(fric.nv) if i not in arm]
  assert fric.dof_frictionloss[rest].tolist() == base.dof_frictionloss[rest].tolist()
  for name in ("dof_armature", "dof_damping", "actuator_gainprm", "actuator_biasprm", "jnt_range", "body_mass"):
    assert (getattr(fric, name) == getattr(base, name)).all(), name


def _model(name):
  return mujoco.MjModel.from_xml_path(str(get_embodiment(name).xml_path))


def _finger_geoms(model):
  bodies = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in ("carriage_left", "carriage_right")}
  return [
    g for g in range(model.ngeom)
    if model.geom_bodyid[g] in bodies and (model.geom_contype[g] or model.geom_conaffinity[g])
  ]


def test_grip_embodiment_is_the_friction_one_with_elliptic_cones():
  friction = get_embodiment("trossen_wxai_base_real_friction")
  grip = get_embodiment("trossen_wxai_base_real_friction_grip")
  assert grip.xml_path.name == "wxai_base_real_robot_friction_grip.xml"
  assert grip.cone == "elliptic" and grip.impratio == 10.0
  assert friction.cone == "pyramidal" and friction.impratio == 1.0
  same = replace(grip, name=friction.name, xml_path=friction.xml_path, cone=friction.cone, impratio=friction.impratio)
  assert same == friction


def test_grip_model_changes_only_the_gripper():
  fric, grip = _model("trossen_wxai_base_real_friction"), _model("trossen_wxai_base_real_friction_grip")
  # Gripper actuator: kp 1e4, kv 85, clamped at the real gripper's 100 N / 1.2 effort limit.
  a = mujoco.mj_name2id(grip, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_gripper")
  assert grip.actuator_gainprm[a, 0] == 10000.0
  assert grip.actuator_biasprm[a, :3].tolist() == [0.0, -10000.0, -85.0]
  assert grip.actuator_forcerange[a].tolist() == [-83.3, 83.3]
  arm = [i for i in range(grip.nu) if i != a]
  for f in ("actuator_gainprm", "actuator_biasprm", "actuator_forcerange", "actuator_ctrlrange"):
    assert (getattr(grip, f)[arm] == getattr(fric, f)[arm]).all(), f
  carriages = [mujoco.mj_name2id(grip, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_carriage_joint") for s in ("left", "right")]
  assert grip.jnt_actfrcrange[carriages].tolist() == [[-83.3, 83.3]] * 2
  others = [j for j in range(grip.njnt) if j not in carriages]
  assert (grip.jnt_actfrcrange[others] == fric.jnt_actfrcrange[others]).all()
  # Finger pads stiffer, with priority so their contacts use these; every other geom unchanged.
  fingers = _finger_geoms(grip)
  assert fingers == _finger_geoms(fric) and len(fingers) == 6
  assert (grip.geom_priority[fingers] == 1).all()
  assert grip.geom_solref[fingers].tolist() == [[0.01, 1.0]] * 6
  assert np.allclose(grip.geom_solimp[fingers, :3], [0.95, 0.99, 0.001])
  rest = [g for g in range(grip.ngeom) if g not in fingers]
  for f in ("geom_priority", "geom_solref", "geom_solimp"):
    assert (getattr(grip, f)[rest] == getattr(fric, f)[rest]).all(), f
  for f in ("geom_friction", "geom_condim", "geom_size", "geom_pos", "geom_quat"):
    assert (getattr(grip, f) == getattr(fric, f)).all(), f
  # The carriage mimic, stiffened.
  assert grip.neq == fric.neq == 1
  assert grip.eq_solref.tolist() == [[0.005, 1.0]]
  assert np.allclose(grip.eq_solimp[0, :3], [0.95, 0.99, 0.001])
  assert (grip.eq_data == fric.eq_data).all()
  # Everything else, including the fitted arm joint friction, as the friction model.
  for f in ("dof_frictionloss", "dof_armature", "dof_damping", "jnt_range", "body_mass", "body_inertia"):
    assert (getattr(grip, f) == getattr(fric, f)).all(), f


def test_grip_scene_is_the_friction_scene():
  scenes = Path(__file__).resolve().parents[1] / "configs" / "grasp" / "scenes"
  grip = json.loads((scenes / "tabletop_jenga_wxai_friction_grip.json").read_text())
  fric = json.loads((scenes / "tabletop_jenga_wxai_friction.json").read_text())
  assert grip["robot"].pop("embodiment") == "trossen_wxai_base_real_friction_grip"
  assert fric["robot"].pop("embodiment") == "trossen_wxai_base_real_friction"
  assert grip == fric
