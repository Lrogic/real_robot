"""Release goal: the clean goal pose and the relative objects' reset poses
(``ReleaseGoalCommand``), and the actor's noisy copy of the goal
(``PerceivedGoal``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from tabletop_sim.tasks.grasp.mdp import curriculum
from tabletop_sim.tasks.grasp.mdp.perception import GaussianPoseNoise, clipped_gaussian, perturb_pose
from tabletop_sim.tasks.release.mdp import events, placement

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

_NEVER = (1.0e9, 1.0e9)


class _CaptureOnReset(CommandTerm):
  """Resets only flag an env; the next ``_update_command`` (after the
  post-reset forward pass, before observations) captures it. Same pattern as
  grasp's ``GraspGoalCommand``."""

  def __init__(self, cfg: CommandTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._needs_capture = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._needs_capture[env_ids] = True

  def _pending(self, env_ids: torch.Tensor | None) -> torch.Tensor:
    pending = self._needs_capture.clone()
    if env_ids is not None:
      scope = torch.zeros_like(pending)
      scope[env_ids] = True
      pending &= scope
    return pending.nonzero(as_tuple=False).squeeze(-1)

  def _update_metrics(self) -> None:
    pass


@dataclass(kw_only=True)
class ReleaseGoalCommandCfg(CommandTermCfg):
  support_names: tuple[str, ...]
  symmetry: str
  resampling_time_range: tuple[float, float] = _NEVER

  def build(self, env: ManagerBasedRlEnv) -> ReleaseGoalCommand:
    return ReleaseGoalCommand(self, env)


class ReleaseGoalCommand(_CaptureOnReset):
  """Per env: the clean goal pose (world) of the restored start row, and the
  relative objects' poses right after the reset."""

  cfg: ReleaseGoalCommandCfg

  def __init__(self, cfg: ReleaseGoalCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    n, s, dev = self.num_envs, len(cfg.support_names), self.device
    self.goal_pos = torch.zeros(n, 3, device=dev)
    self.goal_quat = torch.zeros(n, 4, device=dev)
    self.goal_quat[:, 0] = 1.0
    self.support_pos0 = torch.zeros(n, s, 3, device=dev)
    self.support_quat0 = torch.zeros(n, s, 4, device=dev)
    self.support_quat0[..., 0] = 1.0

  @property
  def command(self) -> torch.Tensor:
    return torch.cat((self.goal_pos, self.goal_quat), dim=-1)

  def closest_goal_quat(self, object_quat: torch.Tensor) -> torch.Tensor:
    return placement.closest_symmetric_quat(self.goal_quat, object_quat, self.cfg.symmetry)

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    ids = self._pending(env_ids)
    if len(ids) == 0:
      return
    starts, rows = events.current_rows(self._env)
    goal = starts.tensors["goal_pose"][rows[ids]]
    self.goal_pos[ids] = goal[:, :3] + self._env.scene.env_origins[ids]
    self.goal_quat[ids] = goal[:, 3:]
    for k, name in enumerate(self.cfg.support_names):
      data = self._env.scene[name].data
      self.support_pos0[ids, k] = data.root_link_pos_w[ids]
      self.support_quat0[ids, k] = data.root_link_quat_w[ids]
    self._needs_capture[ids] = False

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    for i in visualizer.get_env_indices(self.num_envs):
      visualizer.add_sphere(
        center=self.goal_pos[i].cpu().numpy(),
        radius=0.006,
        color=(0.1, 0.9, 0.2, 0.6),
        label=f"release_goal_{i}",
      )


@dataclass(kw_only=True)
class PerceivedGoalCfg(CommandTermCfg):
  goal_command_name: str
  noise: GaussianPoseNoise = field(default_factory=GaussianPoseNoise)
  robot_name: str = "robot"
  resampling_time_range: tuple[float, float] = _NEVER

  def build(self, env: ManagerBasedRlEnv) -> PerceivedGoal:
    return PerceivedGoal(self, env)


class PerceivedGoal(_CaptureOnReset):
  """The actor's goal: the clean goal plus one base-frame pose error drawn at
  reset (scaled by the episode difficulty, clipped) and kept for the episode.
  Must be registered after its goal command so it reads this reset's goal."""

  cfg: PerceivedGoalCfg

  def __init__(self, cfg: PerceivedGoalCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    n, dev = self.num_envs, self.device
    self.pos_w = torch.zeros(n, 3, device=dev)
    self.quat_w = torch.zeros(n, 4, device=dev)
    self.quat_w[:, 0] = 1.0
    self._t_std = torch.tensor(cfg.noise.translation_std, dtype=torch.float, device=dev)
    self._r_std = torch.tensor(cfg.noise.rotation_std, dtype=torch.float, device=dev)

  @property
  def command(self) -> torch.Tensor:
    return torch.cat((self.pos_w, self.quat_w), dim=-1)

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    ids = self._pending(env_ids)
    if len(ids) == 0:
      return
    goal: ReleaseGoalCommand = self._env.command_manager.get_term(self.cfg.goal_command_name)  # type: ignore[assignment]
    pos, quat = goal.goal_pos[ids], goal.goal_quat[ids]
    if self.cfg.noise.enabled:
      d = curriculum.episode_difficulty(self._env, ids)[:, None]
      z = torch.randn(len(ids), 6, device=self.device)
      dp = clipped_gaussian(z[:, :3], d * self._t_std, self.cfg.noise.translation_clip_sigma)
      dr = clipped_gaussian(z[:, 3:], d * self._r_std, self.cfg.noise.rotation_clip_sigma)
      base_quat = self._env.scene[self.cfg.robot_name].data.root_link_quat_w[ids]
      pos, quat = perturb_pose(pos, quat, base_quat, (dp,), (dr,))
    self.pos_w[ids], self.quat_w[ids] = pos, quat
    self._needs_capture[ids] = False
