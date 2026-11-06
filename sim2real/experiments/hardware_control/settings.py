"""Experiment settings (config.yaml) and the import paths shared by every script.

Importing this module puts ``real_robot`` (for ``manipulation.policy``), the experiments
directory (for ``common``) and this directory on ``sys.path``. It needs only numpy and yaml,
so the read-only diagnostic also runs in the fp_robot environment.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
REAL_ROBOT = HERE.parents[2]
DEFAULT_CONFIG = HERE / "config.yaml"
RESULTS_DIR = HERE / "results"

for _path in (REAL_ROBOT, EXPERIMENTS, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

_PATH_KEYS = (("training", "scene"), ("training", "task"), ("hardware", "calibration_config"))


def load_settings(path: Path | None = None) -> dict:
    """config.yaml as a dict with its relative paths resolved against the file."""
    path = Path(path or DEFAULT_CONFIG).resolve()
    settings = yaml.safe_load(path.read_text())
    for section, key in _PATH_KEYS:
        value = settings[section][key]
        settings[section][key] = str((path.parent / value).resolve())
    settings["config_file"] = str(path)
    return settings


def snapshot(settings: dict) -> dict:
    return copy.deepcopy(settings)


def robot_identity_from_calibration(settings: dict) -> dict:
    """ip / model / end_effector from the calibration config's robot section.

    Read with yaml directly: calibration.board needs OpenCV, which the mjlab
    environments do not have.
    """
    from common.hardware import robot_identity

    calibration = yaml.safe_load(Path(settings["hardware"]["calibration_config"]).read_text())
    return robot_identity(calibration)
