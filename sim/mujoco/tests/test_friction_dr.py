"""Explicit object contact pairs and curriculum-scaled pair friction."""

import itertools
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.scene import Scene
from tabletop_sim.embodiments import get_embodiment
from tabletop_sim.embodiments.base import finger_collision_geoms, load_spec
from tabletop_sim.scene import ConfigError, FrictionRandomizationConfig, load_scene_config, load_task_config
from tabletop_sim.scene.builder import HULL_REF_PREFIX, PairGeom, mixed_solver_params
from tabletop_sim.tasks.grasp import mdp
from tabletop_sim.tasks.grasp.env_cfg import CONFIGS_DIR, DEFAULT_SCENE, make_grasp_env_cfg
from tabletop_sim.tasks.grasp.mdp import curriculum, task_state

JENGA_SCENE = CONFIGS_DIR / "scenes" / "tabletop_jenga_wxai.json"
FRICTION_TASK = CONFIGS_DIR / "tasks" / "grasp_curriculum_size_variants_friction_wxai.json"
EVENT = "randomize_pair_friction"
EMBODIMENT_NAMES = ["menagerie_wxai", "trossen_wxai_base"]
GRIPPER = (1.0, 0.005, 0.0005)
TABLE = (1.0, 0.005, 0.0001)
DEVICE = "cuda:0"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
needs_jenga = pytest.mark.skipif(
  not load_scene_config(JENGA_SCENE).manipulated.asset_path.exists(), reason="jenga mesh missing"
)


def _write(path: Path, data: dict) -> Path:
  path.write_text(json.dumps(data))
  return path


def _task(tmp_path: Path, variants: bool = True, **friction) -> Path:
  data = json.loads(FRICTION_TASK.read_text())
  dr = data["domain_randomization"]
  dr["object_size_variants"]["enabled"] = variants
  dr["object_size_variants"]["scales"] = [0.9, 1.0, 1.1]
  dr["friction"].update(friction)
  # No promotions of its own; tests set the stage directly.
  data["curriculum"]["min_completed_episodes"] = 10**9
  return _write(tmp_path / "task.json", data)


def _jenga_scene(tmp_path: Path, embodiment: str = "menagerie_wxai", table_last: bool = False) -> Path:
  data = json.loads(JENGA_SCENE.read_text())
  data["robot"]["embodiment"] = embodiment
  for obj in data["objects"]:
    for key in ("asset_path", "texture_path"):
      if key in obj:
        obj[key] = str((JENGA_SCENE.parent / obj[key]).resolve())
  if table_last:
    data["objects"] = data["objects"][::-1]
  return _write(tmp_path / "scene.json", data)


def _compiled(scene: Path, task: Path, num_envs: int = 2):
  cfg = make_grasp_env_cfg(scene, task, None, num_envs)
  return cfg, Scene(cfg.scene, "cpu").compile()


def _pair_ids(m: mujoco.MjModel, group: str) -> list[int]:
  return [i for i in range(m.npair) if m.pair(i).name.split("/")[-1].startswith(f"{group}_")]


# --- Configuration -----------------------------------------------------------


def test_defaults_disabled():
  c = FrictionRandomizationConfig()
  assert not c.enabled
  assert c.gripper_object == GRIPPER and c.table_object == TABLE
  for r in (c.sliding_scale_range, c.torsional_scale_range, c.rolling_scale_range):
    assert r == (0.7, 1.3)
  for task in ("grasp_standard_wxai", "grasp_curriculum_wxai", "grasp_curriculum_size_variants_wxai"):
    assert not load_task_config(CONFIGS_DIR / "tasks" / f"{task}.json").domain_randomization.friction.enabled
  assert load_task_config(FRICTION_TASK).domain_randomization.friction.enabled


def test_config_loads(tmp_path):
  path = _task(
    tmp_path,
    gripper_object=[0.8, 0.01, 0.001],
    sliding_scale_range=[0.5, 1.5],
    torsional_scale_range=[1, 1],
    rolling_scale_range=[0.9, 2],
  )
  c = load_task_config(path).domain_randomization.friction
  assert c.gripper_object == (0.8, 0.01, 0.001) and c.table_object == TABLE
  assert c.sliding_scale_range == (0.5, 1.5)
  assert c.torsional_scale_range == (1.0, 1.0) and c.rolling_scale_range == (0.9, 2.0)


@pytest.mark.parametrize(
  ("override", "match"),
  [
    ({"enabled": 1}, "must be a bool"),
    ({"gripper_object": [1.0, 0.005, 0.0]}, "positive coefficients"),
    ({"table_object": [-1.0, 0.005, 0.0001]}, "positive coefficients"),
    ({"table_object": [1.0, 0.005]}, "list of 3"),
    ({"sliding_scale_range": [1.3, 0.7]}, "lower <= upper"),
    ({"torsional_scale_range": [0.0, 1.3]}, "lower <= upper"),
    ({"rolling_scale_range": [-0.5, 1.3]}, "lower <= upper"),
    ({"rolling_scale_range": [0.7]}, "list of 2"),
    ({"sliding_scale_range": [0.7, True]}, "list of 2"),
    ({"unknown": 1}, "unknown keys"),
  ],
)
def test_config_rejects(tmp_path, override, match):
  with pytest.raises(ConfigError, match=match):
    load_task_config(_task(tmp_path, **override))


def test_needs_support_object(tmp_path):
  # The default cube scene has no support_surface.
  with pytest.raises(ConfigError, match="support_surface"):
    make_grasp_env_cfg(DEFAULT_SCENE, _task(tmp_path, variants=False))


@needs_jenga
def test_disabled_adds_nothing(tmp_path):
  scene = _jenga_scene(tmp_path)
  task = _task(tmp_path, variants=False, enabled=False)
  cfg, m = _compiled(scene, task)
  assert EVENT not in cfg.events
  assert list(cfg.scene.entities) == ["robot", "ground", "table", "jenga"]
  assert m.npair == 0
  # Robot XML geoms keep their (empty) names, so geom friction mixing applies.
  assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "robot/gripper_left_collision_0") == -1
  # Same model as without the friction block at all.
  data = json.loads(task.read_text())
  del data["domain_randomization"]["friction"]
  _, ref = _compiled(scene, _write(tmp_path / "ref.json", data))
  assert m.ngeom == ref.ngeom and m.nbody == ref.nbody
  for f in ("geom_friction", "geom_condim", "geom_solref", "geom_solimp", "geom_margin", "geom_gap"):
    np.testing.assert_array_equal(getattr(m, f), getattr(ref, f))


@needs_jenga
def test_event_wiring(tmp_path):
  cfg, _ = _compiled(_jenga_scene(tmp_path, table_last=True), _task(tmp_path))
  term = cfg.events[EVENT]
  assert term.func is mdp.RandomizePairFriction and term.mode == "reset"
  assert mdp.RandomizePairFriction.model_fields == ("pair_friction",)
  p = term.params
  assert p["gripper_cfg"].name == p["table_cfg"].name == "jenga"
  assert p["gripper_friction"] == GRIPPER and p["table_friction"] == TABLE
  assert p["sliding_scale_range"] == (0.7, 1.3)
  # Attached after the robot and table its pairs reference.
  assert list(cfg.scene.entities)[-1] == "jenga"


# --- Finger geoms ------------------------------------------------------------


@pytest.mark.parametrize("name", EMBODIMENT_NAMES)
def test_finger_collision_geoms(name):
  emb = get_embodiment(name)
  assert not [g.name for g in load_spec(emb).geoms if "_collision_" in g.name]
  spec = load_spec(emb, name_finger_geoms=True)
  model = spec.compile()
  geoms = finger_collision_geoms(spec, emb)
  names = [[g.name for g in side] for side in geoms]
  again = load_spec(emb, name_finger_geoms=True)
  assert names == [[g.name for g in side] for side in finger_collision_geoms(again, emb)]
  for side, bodies in zip(names, emb.finger_contact_bodies, strict=True):
    expected = []
    for body in bodies:
      bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
      expected += [
        model.geom(g).name for g in range(model.ngeom)
        if model.geom_bodyid[g] == bid and (model.geom_contype[g] or model.geom_conaffinity[g])
      ]
    assert side == expected and len(set(side)) == len(side) > 0
  if name == "menagerie_wxai":
    assert "gripper_left_pad" in names[0] and "carriage_left_collision_0" in names[0]
    assert len(names[0]) == len(names[1]) == 7
  else:
    assert names == [[f"carriage_{s}_collision_{k}" for k in range(3)] for s in ("left", "right")]


# --- Solver parameter mixing --------------------------------------------------


_GEOM_PARAMS = [
  dict(priority=0, solmix=1.0, solref=(0.01, 1.0), solimp=(0.9, 0.95, 0.001, 0.5, 2), margin=0.0, gap=0.0),
  dict(priority=0, solmix=3.0, solref=(0.02, 0.8), solimp=(0.8, 0.9, 0.002, 0.4, 3), margin=0.002, gap=0.001),
  dict(priority=1, solmix=1.0, solref=(0.03, 1.2), solimp=(0.7, 0.99, 0.003, 0.5, 2), margin=0.001, gap=0.0),
  dict(priority=0, solmix=0.0, solref=(-1000.0, -50.0), solimp=(0.85, 0.9, 0.001, 0.5, 2), margin=0.0, gap=0.0005),
]


def _engines_agree(a: int, b: int) -> bool:
  """With different priorities and a direct (negative) solref, MuJoCo C copies
  the higher-priority solref while MuJoCo Warp, which pairs follow, still takes
  the elementwise min."""
  pa, pb = _GEOM_PARAMS[a], _GEOM_PARAMS[b]
  direct = pa["solref"][0] <= 0 or pb["solref"][0] <= 0
  return not (pa["priority"] != pb["priority"] and direct)


@pytest.mark.parametrize(
  ("a", "b"),
  [ab for ab in itertools.combinations(range(len(_GEOM_PARAMS)), 2) if _engines_agree(*ab)],
)
def test_pair_solver_params_match_dynamic_contact(a, b):
  """Pairs carry exactly the solref/solimp/margin/gap MuJoCo gives the same
  geoms' dynamic contact."""

  def spec(with_pair: bool):
    s = mujoco.MjSpec()
    free = s.worldbody.add_body(pos=[0, 0, 0.099])
    free.add_freejoint()
    geoms = []
    for i, body in ((a, s.worldbody), (b, free)):
      p = _GEOM_PARAMS[i]
      geoms.append(body.add_geom(
        name=f"g{i}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.05],
        priority=p["priority"], solmix=p["solmix"], solref=list(p["solref"]),
        solimp=list(p["solimp"]), margin=p["margin"], gap=p["gap"],
      ))
    if with_pair:
      g1, g2 = (PairGeom.from_spec(g) for g in geoms)
      s.add_pair(geomname1=g1.name, geomname2=g2.name, **mixed_solver_params(g1, g2))
    return s

  contacts = []
  for with_pair in (False, True):
    m = spec(with_pair).compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert d.ncon == 1
    c = d.contact[0]
    contacts.append((c.solref.copy(), c.solimp.copy(), c.includemargin, m.npair))
  (ref_solref, ref_solimp, ref_margin, _), (solref, solimp, margin, npair) = contacts
  assert npair == 1
  np.testing.assert_allclose(solref, ref_solref, rtol=1e-12)
  np.testing.assert_allclose(solimp, ref_solimp, rtol=1e-12)
  assert margin == pytest.approx(ref_margin, abs=1e-15)


# --- Compiled pairs -----------------------------------------------------------


def _geom_attr(m, g) -> PairGeom:
  return PairGeom(
    name=m.geom(g).name, contype=int(m.geom_contype[g]), conaffinity=int(m.geom_conaffinity[g]),
    priority=int(m.geom_priority[g]), solmix=float(m.geom_solmix[g]),
    solref=tuple(m.geom_solref[g]), solimp=tuple(m.geom_solimp[g]),
    margin=float(m.geom_margin[g]), gap=float(m.geom_gap[g]),
  )


@needs_jenga
@pytest.mark.parametrize("embodiment", EMBODIMENT_NAMES)
@pytest.mark.parametrize("variants", [False, True])
@pytest.mark.parametrize("table_last", [False, True])
def test_compiled_pairs(tmp_path, embodiment, variants, table_last):
  _, m = _compiled(_jenga_scene(tmp_path, embodiment, table_last), _task(tmp_path, variants))
  emb = get_embodiment(embodiment)
  body = lambda g: m.body(m.geom_bodyid[g]).name  # noqa: E731
  obj_collision = [
    g for g in range(m.ngeom)
    if body(g) == "jenga/jenga" and m.geom_contype[g] and not m.geom(g).name.startswith(f"jenga/{HULL_REF_PREFIX}")
  ]
  assert len(obj_collision) == 1
  obj = obj_collision[0]
  assert ("/mjlab/pad/" in m.geom(obj).name) == variants
  finger_bodies = {f"robot/{b}" for side in emb.finger_contact_bodies for b in side}
  fingers = {g for g in range(m.ngeom) if body(g) in finger_bodies and m.geom_contype[g]}
  tables = {g for g in range(m.ngeom) if body(g) == "table/table"}

  seen = set()
  for group, partners, friction in (("gripper_object", fingers, GRIPPER), ("table_object", tables, TABLE)):
    ids = _pair_ids(m, group)
    got = set()
    for i in ids:
      g1, g2 = int(m.pair_geom1[i]), int(m.pair_geom2[i])
      other = g1 if g2 == obj else g2
      assert obj in (g1, g2) and other in partners
      got.add(other)
      key = frozenset((g1, g2))
      assert key not in seen
      seen.add(key)
      s, t, r = friction
      assert m.pair_dim[i] == 6
      assert m.pair_friction[i].tolist() == [s, s, t, r, r]
      expected = mixed_solver_params(_geom_attr(m, g1), _geom_attr(m, g2))
      np.testing.assert_allclose(m.pair_solref[i], expected["solref"], rtol=1e-12)
      np.testing.assert_allclose(m.pair_solimp[i], expected["solimp"], rtol=1e-12)
      assert m.pair_margin[i] == expected["margin"] and m.pair_gap[i] == expected["gap"]
      np.testing.assert_array_equal(m.pair_solreffriction[i], [0.0, 0.0])
    assert got == partners
  assert m.npair == len(seen) == len(fingers) + len(tables)


# --- Simulation ---------------------------------------------------------------


NUM_ENVS = 8


def _make_env(scene, task, num_envs=NUM_ENVS, seed=0) -> ManagerBasedRlEnv:
  cfg = make_grasp_env_cfg(scene, task, None, num_envs)
  cfg.seed = seed
  return ManagerBasedRlEnv(cfg, device=DEVICE)


def _set_difficulty(env, d):
  """Difficulty the next resetting episodes start with."""
  state = curriculum.get_curriculum_state(env)
  state.stage = round(d / state.difficulty_increment)
  assert state.difficulty == pytest.approx(d)


class _FrictionView:
  """Live pair friction as multipliers of the compiled values, grouped by pair
  name independently of the event term."""

  def __init__(self, env):
    self.env = env
    m = env.sim.mj_model
    self.gripper = torch.tensor(_pair_ids(m, "gripper_object"), device=DEVICE)
    self.table = torch.tensor(_pair_ids(m, "table_object"), device=DEVICE)
    self.default = env.sim.get_default_field("pair_friction")

  @property
  def value(self):
    return self.env.sim.model.pair_friction

  def scale(self):
    return self.value / self.default

  def components(self, group):
    """(num_envs, 3) sliding / torsional / rolling multipliers, after checking
    every pair of the group shares them and they are isotropic."""
    s = self.scale()[:, getattr(self, group)]
    torch.testing.assert_close(s, s[:, :1].expand_as(s), rtol=1e-6, atol=0)
    assert torch.equal(self.value[:, :, 1], self.value[:, :, 0])
    assert torch.equal(self.value[:, :, 4], self.value[:, :, 3])
    return s[:, 0, [0, 2, 3]]

  def assert_baseline(self, ids=slice(None)):
    assert torch.equal(self.value[ids], self.default.expand_as(self.value)[ids])


@pytest.fixture(scope="module")
def friction_env(tmp_path_factory):
  tmp = tmp_path_factory.mktemp("friction")
  env = _make_env(_jenga_scene(tmp), _task(tmp))
  yield env
  env.close()


@needs_cuda
@needs_jenga
def test_disabled_env_has_no_pairs(tmp_path):
  env = _make_env(_jenga_scene(tmp_path), _task(tmp_path, enabled=False), num_envs=2)
  try:
    assert EVENT not in env.event_manager.active_terms["reset"]
    assert "pair_friction" not in env.event_manager.domain_randomization_fields
    assert env.sim.mj_model.npair == 0
  finally:
    env.close()


@needs_cuda
@needs_jenga
def test_registered_and_baselines_at_zero(friction_env):
  env = friction_env
  assert EVENT in env.event_manager.active_terms["reset"]
  assert "pair_friction" in env.event_manager.domain_randomization_fields
  f = _FrictionView(env)
  assert f.value.shape == (NUM_ENVS, 15, 5)
  m = env.sim.mj_model
  for ids, (s, t, r) in ((f.gripper, GRIPPER), (f.table, TABLE)):
    for i in ids.tolist():
      assert m.pair_friction[i].tolist() == [s, s, t, r, r]
  _set_difficulty(env, 0.0)
  for _ in range(3):
    env.reset()
    f.assert_baseline()


@needs_cuda
@needs_jenga
def test_full_difficulty_bounds_and_independence(friction_env):
  env = friction_env
  f = _FrictionView(env)
  _set_difficulty(env, 1.0)
  for _ in range(5):
    env.reset()
    grip, table = f.components("gripper"), f.components("table")
    for s in (grip, table):
      assert (s >= 0.7 * (1 - 1e-6)).all() and (s <= 1.3 * (1 + 1e-6)).all()
      assert (s - 1).abs().max() > 0.01
      assert torch.unique(s).numel() == s.numel()  # per env, per component
    assert not torch.allclose(grip, table)

  # Held for the episode.
  before = f.value.clone()
  for _ in range(5):
    env.step(torch.zeros(NUM_ENVS, 7, device=DEVICE))
  assert torch.equal(f.value, before)

  _set_difficulty(env, 0.0)
  env.reset()
  f.assert_baseline()


@needs_cuda
@needs_jenga
def test_log_uniform_samples(tmp_path):
  env = _make_env(_jenga_scene(tmp_path), _task(tmp_path, variants=False), num_envs=256)
  try:
    f = _FrictionView(env)
    _set_difficulty(env, 1.0)
    samples = []
    for _ in range(8):
      env.reset()
      samples.append(torch.cat((f.components("gripper"), f.components("table")), dim=1))
    logs = torch.log(torch.cat(samples)).double().cpu()  # (2048, 6)
    lo, hi = math.log(0.7), math.log(1.3)
    u = ((logs - lo) / (hi - lo)).sort(dim=0).values
    ecdf = torch.arange(1, u.shape[0] + 1, dtype=torch.double)[:, None] / u.shape[0]
    assert (u > -1e-6).all() and (u < 1 + 1e-6).all()
    assert (ecdf - u).abs().max() < 0.05  # KS distance to uniform in log space
    corr = torch.corrcoef(logs.T)
    assert (corr - torch.eye(6, dtype=torch.double)).abs().max() < 0.1
  finally:
    env.close()


@needs_cuda
@needs_jenga
def test_subset_and_mixed_stage_resets(tmp_path):
  # Degenerate ranges make every multiplier deterministic.
  env = _make_env(
    _jenga_scene(tmp_path),
    _task(
      tmp_path,
      sliding_scale_range=[0.5, 0.5],
      torsional_scale_range=[2.0, 2.0],
      rolling_scale_range=[1.5, 1.5],
    ),
    num_envs=4,
  )
  try:
    f = _FrictionView(env)

    def expect(ids, d):
      want = torch.tensor([1 - 0.5 * d, 1 + d, 1 + 0.5 * d], device=DEVICE)
      for group in ("gripper", "table"):
        s = f.components(group)[ids]
        torch.testing.assert_close(s, want.expand_as(s), rtol=1e-6, atol=0)

    env.reset()
    expect(slice(None), 0.0)

    # The curriculum assigns env 1's difficulty before the event reads it.
    _set_difficulty(env, 1.0)
    env.reset(env_ids=torch.tensor([1], device=DEVICE))
    expect([1], 1.0)
    expect([0, 2, 3], 0.0)
    f.assert_baseline([0, 2, 3])

    # Repeated resets rescale the compiled values, not the current ones.
    for _ in range(3):
      env.reset(env_ids=torch.tensor([1], device=DEVICE))
      expect([1], 1.0)
    _set_difficulty(env, 0.5)
    env.reset(env_ids=torch.tensor([3], device=DEVICE))
    expect([3], 0.5)
    expect([1], 1.0)
    expect([0, 2], 0.0)

    # One batch whose envs started at different stages: each stage gets its
    # own range.
    state = curriculum.get_curriculum_state(env)
    for env_id, d in ((0, 0.25), (1, 1.0), (2, 0.25), (3, 0.0)):
      _set_difficulty(env, d)
      state.episode_stage[env_id] = state.stage
      state.episode_difficulty[env_id] = state.difficulty
    term = env.event_manager.get_term_cfg(EVENT)
    term.func(env, torch.arange(4, device=DEVICE), **term.params)
    expect([0, 2], 0.25)
    expect([1], 1.0)
    expect([3], 0.0)
    f.assert_baseline([3])
  finally:
    env.close()


def _scripted_grasp(env, steps=120):
  obj = env.scene["jenga"]
  goal = task_state.goal(env, "grasp")
  tcp = env.reward_manager.get_term_cfg("reach").params["tcp_cfg"]
  for i in range(steps):
    o, t = obj.data.root_link_pos_w, task_state.tcp_pose_w(env, tcp)[0]
    if i < 30:
      target, grip = o + torch.tensor([0, 0, 0.06], device=DEVICE), 1.0
    elif i < 50:
      target, grip = o.clone(), 1.0
    elif i < 65:
      target, grip = o.clone(), -1.0
    else:
      target, grip = goal.target_pos.clone(), -1.0
    action = torch.zeros(env.num_envs, 7, device=DEVICE)
    action[:, :3] = ((target - t) / 0.01).clamp(-1, 1)
    action[:, 6] = grip
    env.step(action)


def _object_contacts(env):
  """Per object contact: (world, partner geom name, geom pair key, dim,
  friction, solref, solimp, includemargin)."""
  d, m = env.sim.wp_data, env.sim.mj_model
  n = int(d.nacon.numpy()[0])
  c = d.contact
  geom, world, dim = c.geom.numpy()[:n], c.worldid.numpy()[:n], c.dim.numpy()[:n]
  friction, solref = c.friction.numpy()[:n], c.solref.numpy()[:n]
  solimp, margin = c.solimp.numpy()[:n], c.includemargin.numpy()[:n]
  out = []
  for k in range(n):
    names = [m.geom(int(g)).name for g in geom[k]]
    if not any(nm.startswith("jenga/") for nm in names):
      continue
    partner = next(nm for nm in names if not nm.startswith("jenga/"))
    out.append(dict(
      world=int(world[k]), partner=partner, key=tuple(sorted(int(g) for g in geom[k])),
      dim=int(dim[k]), friction=friction[k], solref=solref[k], solimp=solimp[k], margin=margin[k],
    ))
  return out


@needs_cuda
@needs_jenga
@pytest.mark.parametrize(("embodiment", "variants"), [("menagerie_wxai", True), ("trossen_wxai_base", False)])
def test_contacts_use_pair_coefficients(tmp_path, embodiment, variants):
  scene = _jenga_scene(tmp_path, embodiment)
  (tmp_path / "on").mkdir()
  (tmp_path / "off").mkdir()
  on = _make_env(scene, _task(tmp_path / "on", variants), num_envs=4)
  off = _make_env(scene, _task(tmp_path / "off", variants, enabled=False), num_envs=4)
  envs = (on, off)
  try:
    f = _FrictionView(on)
    m = on.sim.mj_model
    pair_of = {
      tuple(sorted((int(m.pair_geom1[i]), int(m.pair_geom2[i])))): i for i in range(m.npair)
    }
    _set_difficulty(on, 1.0)
    seen = {"table": 0, "finger": 0}
    reference: dict[tuple[int, ...], tuple] = {}
    for env in (off, on):
      env.reset()
      for phase in ("rest", "grasp"):
        if phase == "rest":
          for _ in range(20):
            env.step(torch.zeros(env.num_envs, 7, device=DEVICE))
        else:
          _scripted_grasp(env)
        assert int(env.sim.wp_data.nefc.numpy().max()) < env.sim.wp_data.njmax
        for c in _object_contacts(env):
          params = (c["solref"], c["solimp"], c["margin"])
          if env is off:
            reference.setdefault(c["key"], params)
            continue
          i = pair_of[c["key"]]  # every object contact comes from a pair
          assert c["dim"] == 6
          np.testing.assert_allclose(c["friction"], f.value[c["world"], i].cpu().numpy(), rtol=1e-6)
          if c["key"] in reference:
            for got, want in zip(params, reference[c["key"]], strict=True):
              np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-9)
          seen["table" if c["partner"].startswith("table/") else "finger"] += 1
    assert seen["table"] > 0 and seen["finger"] > 0
    # Contacts saw randomized, not baseline, coefficients.
    assert (f.components("gripper") - 1).abs().max() > 0.01
  finally:
    for env in envs:
      env.close()
