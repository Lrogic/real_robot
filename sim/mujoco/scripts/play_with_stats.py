"""Roll out a policy like ``play.py``, logging arm/gripper joint |qvel|.

  python scripts/play_with_stats.py --checkpoint-file logs/rsl_rl/.../model_299.pt
  python scripts/play_with_stats.py --agent random --log-interval 0.25

Remaining flags go to mjlab's PlayConfig (``--viewer viser``, ``--video`` ...).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import mjlab
import mjlab.viewer as viewer_mod
import tyro
from mjlab.scripts import play as play_mod
from mjlab.scripts.play import PlayConfig, run_play
from tabletop_sim.embodiments import get_embodiment
from tabletop_sim.scene import ACTION_SPACES, load_scene_config
from tabletop_sim.tasks.grasp import register_play_task
from tabletop_sim.tasks.grasp.env_cfg import DEFAULT_SCENE, DEFAULT_TASK

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass
class _JointVelStatsLogger:
  arm_joint_names: tuple[str, ...]
  gripper_joint_names: tuple[str, ...]
  log_interval_s: float
  env_index: int

  _arm_ids: tuple[int, ...] | None = None
  _gripper_ids: tuple[int, ...] | None = None
  _last_log_time: float = 0.0
  _peak_arm: float = 0.0
  _peak_grip: float = 0.0
  _peak_arm_joint: str = ""
  _peak_grip_joint: str = ""

  def on_reset(self) -> None:
    self._peak_arm = 0.0
    self._peak_grip = 0.0
    self._peak_arm_joint = ""
    self._peak_grip_joint = ""

  def _resolve_ids(self, env: ManagerBasedRlEnv) -> None:
    if self._arm_ids is not None:
      return
    robot = env.scene["robot"]
    arm_ids, _ = robot.find_joints(self.arm_joint_names, preserve_order=True)
    grip_ids, _ = robot.find_joints(self.gripper_joint_names, preserve_order=True)
    self._arm_ids = tuple(int(i) for i in arm_ids)
    self._gripper_ids = tuple(int(i) for i in grip_ids)

  def on_step(self, env: ManagerBasedRlEnv, step_count: int) -> None:
    self._resolve_ids(env)
    assert self._arm_ids is not None and self._gripper_ids is not None

    qd = env.scene["robot"].data.joint_vel[self.env_index]
    arm_abs = qd[list(self._arm_ids)].abs()
    grip_abs = qd[list(self._gripper_ids)].abs()

    arm_max = float(arm_abs.max().item())
    grip_max = float(grip_abs.max().item())
    if arm_max >= self._peak_arm:
      self._peak_arm = arm_max
      self._peak_arm_joint = self.arm_joint_names[int(arm_abs.argmax().item())]
    if grip_max >= self._peak_grip:
      self._peak_grip = grip_max
      self._peak_grip_joint = self.gripper_joint_names[int(grip_abs.argmax().item())]

    now = time.perf_counter()
    if now - self._last_log_time < self.log_interval_s:
      return
    self._last_log_time = now

    arm_sorted = sorted(
      zip(self.arm_joint_names, arm_abs.tolist()), key=lambda x: x[1], reverse=True
    )
    grip_sorted = sorted(
      zip(self.gripper_joint_names, grip_abs.tolist()), key=lambda x: x[1], reverse=True
    )
    arm_line = ", ".join(f"{name}={speed:.4f}" for name, speed in arm_sorted)
    grip_line = ", ".join(f"{name}={speed:.4f}" for name, speed in grip_sorted)

    print(
      f"[joint_vel] step={step_count} env={self.env_index} "
      f"arm_max={arm_max:.4f} grip_max={grip_max:.4f} "
      f"session_peak_arm={self._peak_arm:.4f}({self._peak_arm_joint}) "
      f"session_peak_grip={self._peak_grip:.4f}({self._peak_grip_joint})"
    )
    print(f"  arm |qvel| sorted: {arm_line}")
    print(f"  grip |qvel| sorted: {grip_line}")


_LOGGER: _JointVelStatsLogger | None = None
_OrigNativeViewer = viewer_mod.NativeMujocoViewer
_OrigViserViewer = viewer_mod.ViserPlayViewer
# Viewer-only spacing (m) between envs drawn around the selected one.
_VISUAL_SPACING = 1.5


def _grid_xy(i: int, n: int, spacing: float) -> np.ndarray:
  """(x, y) of env ``i`` on the same grid mjlab uses for env origins."""
  rows = math.ceil(n / math.isqrt(n))
  cols = math.ceil(n / rows)
  ii, jj = divmod(i, cols)
  return np.array(
    [-(ii - (rows - 1) / 2) * spacing, (jj - (cols - 1) / 2) * spacing]
  )


class _SpreadNativeViewer(_OrigNativeViewer):
  """Native viewer that draws the other envs offset from the selected one.

  mjlab draws every env at its physical origin. This task has no per-env
  origin offset, so all envs overlap. Only the drawn geoms are shifted here;
  the simulation state is unchanged.
  """

  def _render_other_env_geoms(self, viewer, sim, sim_data) -> None:
    if self.vd is None:
      return
    assert self.mjm is not None and self.vopt is not None and self.pert is not None
    n = self.env.unwrapped.num_envs
    ref = _grid_xy(self.env_idx, n, _VISUAL_SPACING)
    for i in range(n):
      if i == self.env_idx:
        continue
      self._sync_env_state_to_mjdata(self.vd, sim_data, i)
      self._sync_model_fields(sim, i)
      mujoco.mj_forward(self.mjm, self.vd)
      dx, dy = _grid_xy(i, n, _VISUAL_SPACING) - ref
      self.vd.geom_xpos[:] += np.array([dx, dy, 0.0])
      mujoco.mjv_addGeoms(
        self.mjm, self.vd, self.vopt, self.pert, self.catmask, viewer.user_scn
      )
    self._sync_model_fields(sim, self.env_idx)


class _StatsNativeViewer(_SpreadNativeViewer):
  def reset_environment(self) -> None:
    if _LOGGER is not None:
      _LOGGER.on_reset()
    super().reset_environment()

  def _execute_step(self) -> bool:
    ok = super()._execute_step()
    if ok and _LOGGER is not None:
      _LOGGER.on_step(self.env.unwrapped, self._step_count)
    return ok


class _StatsViserViewer(_OrigViserViewer):
  def reset_environment(self) -> None:
    if _LOGGER is not None:
      _LOGGER.on_reset()
    super().reset_environment()

  def _execute_step(self) -> bool:
    ok = super()._execute_step()
    if ok and _LOGGER is not None:
      _LOGGER.on_step(self.env.unwrapped, self._step_count)
    return ok


def _install_stats_viewers() -> None:
  viewer_mod.NativeMujocoViewer = _StatsNativeViewer
  viewer_mod.ViserPlayViewer = _StatsViserViewer
  play_mod.NativeMujocoViewer = _StatsNativeViewer
  play_mod.ViserPlayViewer = _StatsViserViewer


def main() -> None:
  global _LOGGER, _VISUAL_SPACING

  parser = argparse.ArgumentParser(
    description=__doc__, add_help=False, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument("--scene", default=str(DEFAULT_SCENE), help="scene JSON")
  parser.add_argument("--task", default=str(DEFAULT_TASK), help="task JSON")
  parser.add_argument("--action-space", choices=ACTION_SPACES, default=None)
  parser.add_argument(
    "--log-interval",
    type=float,
    default=0.5,
    help="seconds between |qvel| log lines (default: 0.5)",
  )
  parser.add_argument(
    "--log-env",
    type=int,
    default=0,
    help="which parallel env index to log (default: 0)",
  )
  parser.add_argument(
    "--visual-spacing",
    type=float,
    default=1.5,
    help="native viewer only: metres between envs drawn around the selected one "
    "(does not change the simulation; default: 1.5)",
  )
  ours, rest = parser.parse_known_args()
  if "-h" in rest or "--help" in rest:
    parser.print_help()
    print("\nmjlab PlayConfig flags:\n")

  _VISUAL_SPACING = ours.visual_spacing
  scene_cfg = load_scene_config(ours.scene)
  emb = get_embodiment(scene_cfg.robot.embodiment)
  _LOGGER = _JointVelStatsLogger(
    arm_joint_names=emb.arm_joint_names,
    gripper_joint_names=emb.gripper_joint_names,
    log_interval_s=ours.log_interval,
    env_index=ours.log_env,
  )
  _install_stats_viewers()

  task_id = register_play_task(ours.scene, ours.task, ours.action_space)
  args = tyro.cli(PlayConfig, args=rest, prog=sys.argv[0], config=mjlab.TYRO_FLAGS)
  print(
    f"[joint_vel] logging arm ({len(emb.arm_joint_names)} joints) and "
    f"gripper ({len(emb.gripper_joint_names)} joints); interval={ours.log_interval}s"
  )
  run_play(task_id, args)


if __name__ == "__main__":
  main()
