"""Snapshot of every config a run.sh invocation used, written into the run folder.

    <run>/configs/<YYYYmmdd_HHMMSS>_<stage>/
        run.json                  command line, resolved options, interpreter, git state
        git.diff                  uncommitted changes to tracked files (only when there are any)
        camera_arm/               camera_arm.yaml, the calibration config (follower ip/model), and the
                                  parked and two-arm calibration files it names, each with its session.json
        recovery/recovery.yaml    with --recovery (CLI overrides are in run.json "recovery")
        manipulation/             with --manipulation: its config.yaml; for policy also
                                  policy.json (the resolved config, --policy-control applied),
                                  scene.json, task.json, and checkpoint/ (the .pt and its run's params/)

Every invocation gets its own folder, so a run folder tracked several times keeps one per track.
The folder's timestamp matches the policy_log.jsonl start event of that invocation.
"""

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .pipeline_args import resolved_recovery_settings
from .recovery.recovery_config import DEFAULT_RECOVERY_CONFIG
from .robot_arm.camera_arm import CameraArmConfig

CONFIGS_DIR = "configs"
RUN_FILE = "run.json"
DIFF_FILE = "git.diff"
REPO_DIR = Path(__file__).resolve().parents[2]


def _git(repo: Path, *cmd: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", "-C", str(repo), *cmd], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout if out.returncode == 0 else None


def git_state(repo: Path) -> Optional[dict]:
    """Commit, branch and dirty files of repo, or None if it is not a git checkout."""
    commit = _git(repo, "rev-parse", "HEAD")
    if commit is None:
        return None
    status = _git(repo, "status", "--porcelain") or ""
    return {"path": str(repo), "commit": commit.strip(),
            "branch": (_git(repo, "rev-parse", "--abbrev-ref", "HEAD") or "").strip(),
            "dirty": [line for line in status.splitlines() if line.strip()]}


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def _copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(src)


def _copy_calibration(src: Path, dst_dir: Path, sources: dict, key: str) -> None:
    """A calibration result file and the session.json beside it (parked joints, robot identity)."""
    sources[key] = _copy(src, dst_dir / src.name)
    session = src.parent / "session.json"
    if session.is_file():
        sources[f"{key}_session"] = _copy(session, dst_dir / "session.json")


def _snapshot_camera_arm(args, out: Path, sources: dict) -> None:
    from calibration.board import DEFAULT_CONFIG as DEFAULT_CALIBRATION_CONFIG

    cfg = CameraArmConfig.load(args.camera_config)
    sources["camera_arm"] = _copy(cfg.path, out / "camera_arm.yaml")
    calibration_config = Path(args.calibration_config or DEFAULT_CALIBRATION_CONFIG).resolve()
    sources["calibration_config"] = _copy(calibration_config, out / "calibration_config.yaml")
    _copy_calibration(cfg.parked_calibration, out / "parked", sources, "parked_calibration")
    _copy_calibration(cfg.move_calibration, out / "move", sources, "move_calibration")


def _snapshot_policy(args, out: Path, sources: dict) -> None:
    from manipulation.policy.jenga.config import JengaPolicyConfig

    cfg = JengaPolicyConfig.load(args.manipulation_config).with_mode(args.policy_control)
    sources["manipulation_config"] = _copy(cfg.path, out / "config.yaml")
    (out / "policy.json").write_text(json.dumps(cfg.describe(), indent=2) + "\n")
    sources["scene"] = _copy(cfg.scene_path, out / "scene.json")
    sources["task"] = _copy(cfg.task_path, out / "task.json")
    if cfg.checkpoint is not None:
        sources["checkpoint"] = _copy(cfg.checkpoint, out / "checkpoint" / cfg.checkpoint.name)
        params = cfg.checkpoint.parent / "params"
        if params.is_dir():
            shutil.copytree(params, out / "checkpoint" / "params")
            sources["checkpoint_params"] = str(params)


def _snapshot_basic(args, out: Path, sources: dict) -> None:
    from manipulation.basic_manipulation.grasp_goals import DEFAULT_CONFIG

    sources["manipulation_config"] = _copy(Path(args.manipulation_config or DEFAULT_CONFIG).resolve(),
                                           out / "config.yaml")


def snapshot_run_configs(args, stage: str, argv, now: Optional[datetime] = None) -> Path:
    """Write <save-path>/configs/<time>_<stage>/ for this invocation and return it."""
    now = now or datetime.now()
    out = Path(args.save_path) / CONFIGS_DIR / f"{now:%Y%m%d_%H%M%S}_{stage}"
    suffix = 1
    while out.exists():
        suffix += 1
        out = out.with_name(f"{now:%Y%m%d_%H%M%S}_{stage}_{suffix}")
    out.mkdir(parents=True)

    sources = {}
    _snapshot_camera_arm(args, out / "camera_arm", sources)
    if args.recovery:
        sources["recovery"] = _copy(Path(args.recovery_config or DEFAULT_RECOVERY_CONFIG).resolve(),
                                    out / "recovery" / "recovery.yaml")
    if args.manipulation == "policy":
        _snapshot_policy(args, out / "manipulation", sources)
    elif args.manipulation == "basic":
        _snapshot_basic(args, out / "manipulation", sources)

    repo = git_state(REPO_DIR)
    if repo is not None and repo["dirty"]:
        diff = _git(REPO_DIR, "diff", "HEAD")
        if diff:
            (out / DIFF_FILE).write_text(diff)
    options = {k: v for k, v in vars(args).items() if k != "recovery_cfg"}
    record = {
        "time": now.isoformat(timespec="seconds"),
        "stage": stage,
        "argv": list(argv),
        "cwd": os.getcwd(),
        "options": options,
        "recovery": resolved_recovery_settings(args) if args.recovery_cfg is not None else None,
        "sources": sources,
        "python": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "git": {"real_robot": repo, "foundationpose": git_state(Path(args.foundationpose_dir))},
    }
    (out / RUN_FILE).write_text(json.dumps(_jsonable(record), indent=2) + "\n")
    print(f"Configs saved to {out}")
    return out
