import mujoco
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
