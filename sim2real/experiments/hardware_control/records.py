"""Trial result files: one JSON per trial, written after the trial (never inside the loop)."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from settings import RESULTS_DIR
from common.io import write_json

RESULT_KIND = "hardware_control_trial"
UNITS = {
    "time": "s since command onset (monotonic host clock on hardware, simulated time in sim)",
    "joint_position": "rad", "joint_velocity": "rad/s", "gripper_position": "m (one carriage)",
    "tcp_position": "m, robot base frame", "tcp_orientation": "quaternion wxyz (w >= 0), robot base frame",
    "driver_tcp": "m then angle-axis rad, driver base frame (driver's own FK)",
    "header_timestamp": "us since driver configuration (controller clock)",
}


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def environment_record() -> dict:
    record = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("numpy", "mujoco", "mjlab", "torch", "trossen_arm"):
        module = sys.modules.get(name)
        if module is not None:
            record[name] = getattr(module, "__version__", "unknown")
    return record


def result_path(source: str, plan_name: str, tag: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return RESULTS_DIR / source / f"{plan_name}__{tag}__{stamp}.json"


def save_result(result: dict, path: Path) -> Path:
    write_json(path, to_jsonable(result))
    return path


def load_result(path: Path) -> dict:
    result = json.loads(Path(path).read_text())
    if result.get("kind") != RESULT_KIND:
        raise ValueError(f"{path} is not a {RESULT_KIND} file")
    return result
