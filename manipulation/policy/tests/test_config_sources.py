"""jenga/config.yaml takes the scene and task from sim.scene / sim.task, or, when one is null,
from the copy train.py saved in the checkpoint's run folder (params/scene.json, task.json).

  /opt/conda-envs/wxai_mjlab/bin/python -m pytest real_robot/manipulation/policy/tests/test_config_sources.py
"""

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from manipulation.policy.jenga.config import DEFAULT_CONFIG, JengaPolicyConfig  # noqa: E402

CONFIG_DIR = Path(DEFAULT_CONFIG).resolve().parent
SCENE = (CONFIG_DIR / "../../../sim/mujoco/configs/scenes/tabletop_jenga_wxai.json").resolve()
TASK = (CONFIG_DIR / "../../../sim/mujoco/configs/tasks/grasp_wxai.json").resolve()


def _run(tmp_path: Path, scene: bool = True, task: bool = True, source: bool = True) -> Path:
    """A run folder as train.py leaves it: a checkpoint and params/ with a 10 Hz task. Returns the checkpoint."""
    params = tmp_path / "run" / "params"
    params.mkdir(parents=True)
    if scene:
        (params / "scene.json").write_bytes(SCENE.read_bytes())      # relative asset paths kept
    if task:
        saved = json.loads(TASK.read_text())
        saved["timing"].update(control_hz=10.0, episode_steps=150)
        (params / "task.json").write_text(json.dumps(saved))
    if source:
        (params / "source.json").write_text(json.dumps({"scene": str(SCENE), "task": str(TASK)}))
    checkpoint = tmp_path / "run" / "model_1.pt"
    checkpoint.touch()
    return checkpoint


def _config(tmp_path: Path, scene, task, checkpoint) -> Path:
    raw = yaml.safe_load(Path(DEFAULT_CONFIG).read_text())
    raw["sim"] = {"scene": scene and str(scene), "task": task and str(task)}
    raw["policy"] = {"checkpoint": checkpoint and str(checkpoint)}
    raw["sim_ik"]["model"] = str((CONFIG_DIR / raw["sim_ik"]["model"]).resolve())
    if (raw.get("calibration") or {}).get("T_base_camera"):
        raw["calibration"]["T_base_camera"] = str((CONFIG_DIR / raw["calibration"]["T_base_camera"]).resolve())
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_null_scene_and_task_come_from_the_checkpoint_run(tmp_path):
    checkpoint = _run(tmp_path)
    cfg = JengaPolicyConfig.load(_config(tmp_path, None, None, checkpoint))
    params = checkpoint.parent / "params"
    assert (cfg.scene_path, cfg.task_path) == (params / "scene.json", params / "task.json")
    assert (cfg.control_hz, cfg.episode_steps) == (10.0, 150)
    # The saved scene's relative mesh path resolves as it did for the original scene file.
    explicit = JengaPolicyConfig.load(_config(tmp_path, SCENE, TASK, checkpoint))
    assert cfg.mesh_path == explicit.mesh_path and cfg.mesh_path.is_file()
    assert cfg.embodiment == explicit.embodiment


def test_set_scene_and_task_take_priority_over_the_saved_copies(tmp_path):
    checkpoint = _run(tmp_path)
    cfg = JengaPolicyConfig.load(_config(tmp_path, SCENE, TASK, checkpoint))
    assert (cfg.scene_path, cfg.task_path) == (SCENE, TASK)
    assert (cfg.control_hz, cfg.episode_steps) == (20.0, 300)


def test_scene_and_task_fall_back_independently(tmp_path):
    checkpoint = _run(tmp_path)
    params = checkpoint.parent / "params"
    cfg = JengaPolicyConfig.load(_config(tmp_path, SCENE, None, checkpoint))
    assert (cfg.scene_path, cfg.task_path, cfg.control_hz) == (SCENE, params / "task.json", 10.0)
    cfg = JengaPolicyConfig.load(_config(tmp_path, None, TASK, checkpoint))
    assert (cfg.scene_path, cfg.task_path, cfg.control_hz) == (params / "scene.json", TASK, 20.0)


@pytest.mark.parametrize("missing", ["scene", "task"])
def test_null_without_a_saved_copy_is_an_error(tmp_path, missing):
    checkpoint = _run(tmp_path, **{missing: False})
    with pytest.raises(ValueError, match=f"sim.{missing} is null and the checkpoint's run has no saved {missing}"):
        JengaPolicyConfig.load(_config(tmp_path, None, None, checkpoint))
    # Setting the missing one is enough.
    given = {"scene": SCENE if missing == "scene" else None, "task": TASK if missing == "task" else None}
    JengaPolicyConfig.load(_config(tmp_path, given["scene"], given["task"], checkpoint))


def test_null_without_a_checkpoint_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="sim.scene is null and policy.checkpoint is not set"):
        JengaPolicyConfig.load(_config(tmp_path, None, None, None))


def test_saved_scene_needs_its_source_record(tmp_path):
    checkpoint = _run(tmp_path, source=False)
    with pytest.raises(ValueError, match="no source.json beside it"):
        JengaPolicyConfig.load(_config(tmp_path, None, None, checkpoint))
