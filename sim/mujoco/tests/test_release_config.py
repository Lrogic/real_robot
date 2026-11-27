"""Release task config: schema, validation, and the dataset metadata check."""

import copy
import json
from pathlib import Path

import pytest

from tabletop_sim import CONFIGS_DIR
from tabletop_sim.scene import ConfigError, load_scene_config
from tabletop_sim.scene.release_config import ReleaseTaskConfig, load_release_task_config
from tabletop_sim.tasks.release import starts

RELEASE = CONFIGS_DIR / "release"
TASKS = sorted((RELEASE / "tasks").glob("*.json"))
TOWER_TASK = RELEASE / "tasks" / "release_tower_wxai.json"
TOWER_SCENE = RELEASE / "scenes" / "tabletop_jenga_tower_wxai.json"
BRIDGE_TASK = RELEASE / "tasks" / "release_bridge_wxai.json"
BRIDGE_SCENE = RELEASE / "scenes" / "tabletop_jenga_bridge_wxai.json"


def _load(tmp_path: Path, data: dict) -> ReleaseTaskConfig:
  path = tmp_path / "task.json"
  path.write_text(json.dumps(data))
  return load_release_task_config(path)


def _tower() -> dict:
  return json.loads(TOWER_TASK.read_text())


def _keys(d: dict, prefix: str = "") -> set[str]:
  out = set()
  for k, v in d.items():
    if isinstance(v, dict) and k != "objects":
      out |= _keys(v, f"{prefix}{k}.")
    else:
      out.add(prefix + k)
  return out


@pytest.mark.parametrize("task_path", TASKS, ids=lambda p: p.name)
def test_checked_in_tasks_list_every_option(task_path):
  """Every release task file writes out every option (as grasp_wxai.json does)."""
  from dataclasses import asdict

  written = json.loads(task_path.read_text())
  cfg = asdict(load_release_task_config(task_path))
  cfg["relative_placement"]["objects"] = dict(cfg["relative_placement"]["objects"])
  # The shared size-variant config keeps a clearance field that release rejects.
  del cfg["domain_randomization"]["object_size_variants"]["clearance"]
  assert _keys(written) == _keys(cfg)


@pytest.mark.parametrize(
  ("task_path", "scene_path"), [(TOWER_TASK, TOWER_SCENE), (BRIDGE_TASK, BRIDGE_SCENE)]
)
def test_checked_in_tasks_match_their_scenes(task_path, scene_path):
  load_release_task_config(task_path).validate_scene(load_scene_config(scene_path))


def test_dataset_path_resolves_against_the_task_file():
  task = load_release_task_config(TOWER_TASK)
  assert task.start.dataset == (TOWER_TASK.parent / "../../../data/release_starts/tower_wxai.pt").resolve()


@pytest.mark.parametrize(
  ("edit", "message"),
  [
    (lambda d: d.update(bogus=1), "unknown keys"),
    (lambda d: d["domain_randomization"].update(object_reset_radius=0.05), "do not apply to release"),
    (lambda d: d["domain_randomization"]["object_size_variants"].update(clearance=0.0), "clearance"),
    (lambda d: d.update(w_robot_collision=1.0), "nonpositive"),
    (lambda d: d.update(w_place=-1.0), "nonnegative"),
    (lambda d: d["relative_placement"].update(goal_symmetry="cylinder"), "goal_symmetry"),
    (lambda d: d["relative_placement"].pop("goal_pose"), "goal_pose is required"),
    (lambda d: d["relative_placement"].update(objects={}), "at least one"),
    (lambda d: d["start"].update(min_height_above_goal=0.05), "min_height_above_goal"),
    (lambda d: d["start"].update(settle_steps_range=[5, 2]), "settle_steps_range"),
    (lambda d: d["domain_randomization"]["friction"].update(support_object=[0.5, 0.0, 0.1]), "support_object"),
  ],
)
def test_invalid_options_raise(tmp_path, edit, message):
  data = _tower()
  edit(data)
  with pytest.raises(ConfigError, match=message):
    _load(tmp_path, data)


def test_scene_validation(tmp_path):
  scene = load_scene_config(TOWER_SCENE)
  data = _tower()
  data["relative_placement"]["objects"] = {"jenga": [0, 0, 0.0075, 1, 0, 0, 0]}
  with pytest.raises(ConfigError, match="must not be the manipulated object"):
    _load(tmp_path, data).validate_scene(scene)
  data["relative_placement"]["objects"] = {"nope": [0, 0, 0.0075, 1, 0, 0, 0]}
  with pytest.raises(ConfigError, match="no such scene object"):
    _load(tmp_path, data).validate_scene(scene)


def test_release_dr_matches_grasp_dr_for_shared_options():
  task = load_release_task_config(TOWER_TASK)
  dr = task.domain_randomization.as_grasp()
  assert dr.friction.enabled == task.domain_randomization.friction.enabled
  assert dr.object_reset_radius == 0.0


def _meta(task: ReleaseTaskConfig, scene_name: str) -> dict:
  sizes = task.domain_randomization.object_size_variants
  return {
    **json.loads(json.dumps(starts.generation_key(task))),
    "format_version": starts.FORMAT_VERSION,
    "scene_name": scene_name,
    "variant_scales": list(sizes.scales) if sizes.enabled else [1.0],
  }


def test_meta_check_accepts_matching_dataset_and_rejects_changes(tmp_path):
  task = load_release_task_config(TOWER_TASK)
  meta = _meta(task, "wxai_tabletop_jenga_tower")
  starts.check_meta(task, meta, "wxai_tabletop_jenga_tower")

  changed = _tower()
  changed["relative_placement"]["anchor_radius"] = 0.08
  with pytest.raises(ConfigError, match="regenerate"):
    starts.check_meta(_load(tmp_path, changed), meta, "wxai_tabletop_jenga_tower")
  changed = _tower()
  changed["start"]["ball_radius"] = 0.05
  with pytest.raises(ConfigError, match="regenerate"):
    starts.check_meta(_load(tmp_path, changed), meta, "wxai_tabletop_jenga_tower")
  with pytest.raises(ConfigError, match="scene"):
    starts.check_meta(task, meta, "other_scene")
  missing = copy.deepcopy(meta)
  missing["variant_scales"] = meta["variant_scales"][:-1]
  with pytest.raises(ConfigError, match="size scales"):
    starts.check_meta(task, missing, "wxai_tabletop_jenga_tower")


def test_missing_dataset_says_how_to_make_it(tmp_path):
  with pytest.raises(ConfigError, match="generate_release_starts"):
    starts.read_meta(tmp_path / "none.pt")


@pytest.mark.parametrize("scene_path", [TOWER_SCENE, BRIDGE_SCENE], ids=lambda p: p.name)
def test_release_scenes_use_stiff_contacts(scene_path):
  """Release stacks blocks: with MuJoCo's default solref (20 ms) a dropped block
  pushed the base block up to 16 mm into the table and it bounced back."""
  for obj in load_scene_config(scene_path).objects:
    assert obj.solref == (0.004, 1.0), obj.name


def test_every_scene_uses_the_same_stiff_contacts():
  """One physics for every task: the grasp policy is unchanged by it (same hold
  score, success and lift with default and stiff contacts)."""
  for path in sorted(CONFIGS_DIR.glob("*/scenes/*.json")):
    for obj in load_scene_config(path).objects:
      assert obj.solref == (0.004, 1.0) and obj.solimp is None, f"{path.name}: {obj.name}"


@pytest.mark.parametrize(
  ("value", "message"),
  [({"solref": [0.004]}, "list of 2"), ({"solref": [0.004, -1.0]}, "timeconst"),
   ({"solimp": [0.99, 0.9, 0.001, 0.5, 2.0]}, "dmin <= dmax")],
)
def test_object_contact_params_are_validated(tmp_path, value, message):
  data = json.loads(TOWER_SCENE.read_text())
  data["objects"][0].update(value)
  path = TOWER_SCENE.parent / "_tmp_invalid_scene.json"
  try:
    path.write_text(json.dumps(data))
    with pytest.raises(ConfigError, match=message):
      load_scene_config(path)
  finally:
    path.unlink(missing_ok=True)


def test_object_solref_reaches_the_compiled_geom():
  from tabletop_sim.scene import object_spec

  obj = next(o for o in load_scene_config(TOWER_SCENE).objects if o.name == "base_block")
  model = object_spec(obj).compile()
  geom = next(g for g in range(model.ngeom) if model.geom_contype[g])
  assert tuple(model.geom_solref[geom]) == pytest.approx((0.004, 1.0))
