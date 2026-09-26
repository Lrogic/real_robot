import mujoco
import pytest

from tabletop_sim.embodiments import EMBODIMENTS, get_embodiment
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


def test_menagerie_camera_mount_removed():
  emb = get_embodiment("menagerie_wxai")
  model = load_spec(emb).compile()
  assert model.ncam == 0
  assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "camera_mount_d405") == -1


def test_unknown_embodiment():
  with pytest.raises(KeyError):
    get_embodiment("ur5")
