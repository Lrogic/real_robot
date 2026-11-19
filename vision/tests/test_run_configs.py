"""run.sh's per-invocation config snapshot (pipeline/run_configs.py), from parsed run.sh options.

    cd real_robot/vision && python -m unittest tests.test_run_configs -v
"""

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

VISION_DIR = Path(__file__).resolve().parents[1]
for path in (VISION_DIR, VISION_DIR.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from manipulation.policy.jenga.config import JengaPolicyConfig  # noqa: E402
from pipeline.pipeline_args import parse_pipeline_args, pipeline_parser  # noqa: E402
from pipeline.run_configs import snapshot_run_configs  # noqa: E402


def _parse(argv):
    with redirect_stdout(io.StringIO()):
        return parse_pipeline_args(pipeline_parser(""), argv)


def _snapshot(args, stage, argv, now=None):
    with redirect_stdout(io.StringIO()):
        return snapshot_run_configs(args, stage, [stage, *argv], now=now)


class RunConfigsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.run = Path(self.directory.name) / "run1"
        self.run.mkdir()

    def tearDown(self):
        self.directory.cleanup()

    def test_camera_and_recovery(self):
        argv = ["--save-path", str(self.run), "--camera", "parked", "--recovery", "--sam-fps", "1"]
        out = _snapshot(_parse(argv), "track", argv)
        self.assertEqual(out.parent, self.run / "configs")
        record = json.loads((out / "run.json").read_text())
        self.assertEqual(record["stage"], "track")
        self.assertEqual(record["argv"], ["track", *argv])
        self.assertEqual(record["options"]["camera"], "parked")
        self.assertEqual(record["recovery"]["sam_fps"], 1.0)
        for name in ("camera_arm/camera_arm.yaml", "camera_arm/calibration_config.yaml",
                     "camera_arm/parked/T_base_camera.json", "camera_arm/parked/session.json",
                     "camera_arm/move/T_follower_camera.json", "recovery/recovery.yaml"):
            self.assertTrue((out / name).is_file(), name)
        self.assertFalse((out / "manipulation").exists())

    def test_policy(self):
        cfg = JengaPolicyConfig.load()
        if cfg.checkpoint is None or not cfg.checkpoint.is_file():
            self.skipTest(f"checkpoint {cfg.checkpoint} is missing")
        argv = ["--save-path", str(self.run), "--camera", "move", "--recovery", "--manipulation", "policy",
                "--policy-control", "shadow"]
        out = _snapshot(_parse(argv), "track", argv) / "manipulation"
        policy = json.loads((out / "policy.json").read_text())
        self.assertEqual(policy["control"]["mode"], "shadow")
        self.assertEqual(policy["checkpoint"], str(cfg.checkpoint))
        self.assertEqual((out / "checkpoint" / cfg.checkpoint.name).read_bytes(), cfg.checkpoint.read_bytes())
        self.assertEqual(json.loads((out / "task.json").read_text()), json.loads(cfg.task_path.read_text()))
        for name in ("config.yaml", "scene.json"):
            self.assertTrue((out / name).is_file(), name)
        if (cfg.checkpoint.parent / "params").is_dir():
            self.assertTrue((out / "checkpoint" / "params" / "env.yaml").is_file())

    def test_each_invocation_keeps_its_own_folder(self):
        argv = ["--save-path", str(self.run), "--camera", "parked"]
        args, now = _parse(argv), datetime(2026, 11, 19, 12, 0, 0)
        first = _snapshot(args, "track", argv, now)
        second = _snapshot(args, "track", argv, now)
        self.assertNotEqual(first, second)
        self.assertEqual(len(list((self.run / "configs").iterdir())), 2)


if __name__ == "__main__":
    unittest.main()
