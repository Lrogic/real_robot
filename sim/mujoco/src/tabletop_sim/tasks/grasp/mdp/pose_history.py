"""True pose of a free object a fixed number of physics substeps ago.

mjlab's observation delay counts control steps; a perception latency shorter
than one control step needs the pose from inside the decimation loop. This
sensor is updated after every physics substep and keeps the last
``delay_substeps + 1`` poses per env.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.sensor.sensor import Sensor, SensorCfg

if TYPE_CHECKING:
  import mujoco
  import mujoco_warp as mjwarp

  from mjlab.entity import Entity


@dataclass
class ObjectPoseHistoryCfg(SensorCfg):
  object_name: str = ""
  """Entity with a free joint whose root pose is recorded."""
  delay_substeps: int = 1

  def build(self) -> ObjectPoseHistory:
    if self.delay_substeps < 1:
      raise ValueError(f"delay_substeps must be >= 1, got {self.delay_substeps}")
    return ObjectPoseHistory(self)


class ObjectPoseHistory(Sensor[torch.Tensor]):
  """``data``: world pose ``[x, y, z, qw, qx, qy, qz]`` of the object
  ``delay_substeps`` substeps before the latest one, shape ``(num_envs, 7)``.

  Poses are read from the free joint's ``qpos``, which is current after every
  substep (``xpos`` lags it by one until the next ``forward()``).

  A reset marks an env; its next substep fills the env's whole history with the
  pose after that substep, so the old episode never leaks into the new one.
  Until then the env's ``data`` is not meaningful.
  """

  def __init__(self, cfg: ObjectPoseHistoryCfg):
    super().__init__()
    self.cfg = cfg
    self._entity: Entity | None = None

  def edit_spec(self, scene_spec: mujoco.MjSpec, entities: dict[str, Entity]) -> None:
    del scene_spec
    self._entity = entities[self.cfg.object_name]

  def initialize(
    self, mj_model: mujoco.MjModel, model: mjwarp.Model, data: mjwarp.Data, device: str
  ) -> None:
    del mj_model, model
    assert self._entity is not None
    q_adr = self._entity.indexing.free_joint_q_adr
    if len(q_adr) != 7:
      raise ValueError(f"'{self.cfg.object_name}' has no free joint to record")
    self._q_adr = q_adr.to(device=device, dtype=torch.long)
    self._data = data
    num_envs = data.qpos.shape[0]
    self._buffer = torch.zeros(num_envs, self.cfg.delay_substeps + 1, 7, device=device)
    self._head = 0
    self._refill = torch.ones(num_envs, dtype=torch.bool, device=device)

  def _compute_data(self) -> torch.Tensor:
    oldest = (self._head + 1) % self._buffer.shape[1]
    return self._buffer[:, oldest]

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    self._refill[slice(None) if env_ids is None else env_ids] = True

  def update(self, dt: float) -> None:
    super().update(dt)
    pose = self._data.qpos[:, self._q_adr]
    self._head = (self._head + 1) % self._buffer.shape[1]
    self._buffer[:, self._head] = pose
    if self._refill.any():
      self._buffer[self._refill] = pose[self._refill, None]
      self._refill[:] = False
