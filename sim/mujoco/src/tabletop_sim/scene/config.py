"""Scene and task configuration files (JSON) and their validation.

Scene JSON layout::

  {
    "scene_name": str,
    "robot": {"embodiment": str, "action_space": str, "base_pose": [x,y,z,qw,qx,qy,qz]},
    "camera": {"render_eye": [x,y,z], "render_target": [x,y,z]},     # optional
    "ground_altitude": float | null,                                  # optional
    "manipulated_object": str,
    "objects": [ObjectConfig, ...]
  }

Poses are ``[x, y, z, qw, qx, qy, qz]`` in the world frame. Each object's
``initial_poses`` lists candidate reset poses; one is sampled uniformly per
environment at every reset (a single entry gives a fixed reset).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

Pose = tuple[float, float, float, float, float, float, float]
Primitive = Literal["box", "sphere", "cylinder", "capsule", "mesh"]
PRIMITIVES = ("box", "sphere", "cylinder", "capsule", "mesh")

ACTION_SPACES = ("ee_delta_pose_binary_gripper", "delta_joint_abs_gripper")
RESERVED_ENTITY_NAMES = ("robot", "ground", "terrain")


class ConfigError(ValueError):
  pass


def _check_keys(where: str, data: dict[str, Any], cls: type, extra: tuple[str, ...] = ()) -> None:
  allowed = {f.name for f in fields(cls)} | set(extra)
  unknown = set(data) - allowed
  if unknown:
    raise ConfigError(f"{where}: unknown keys {sorted(unknown)}; allowed {sorted(allowed)}")


def _vec(where: str, value: Any, n: int) -> tuple[float, ...]:
  if not isinstance(value, (list, tuple)) or len(value) != n:
    raise ConfigError(f"{where}: expected a list of {n} numbers, got {value!r}")
  out = tuple(float(v) for v in value)
  if not all(math.isfinite(v) for v in out):
    raise ConfigError(f"{where}: values must be finite, got {value!r}")
  return out


def _pose(where: str, value: Any) -> Pose:
  pose = _vec(where, value, 7)
  norm = math.sqrt(sum(q * q for q in pose[3:]))
  if abs(norm - 1.0) > 1e-3:
    raise ConfigError(f"{where}: quaternion must be unit length (|q|={norm:.4f})")
  q = tuple(v / norm for v in pose[3:])
  return (*pose[:3], *q)  # type: ignore[return-value]


@dataclass(frozen=True)
class RobotConfig:
  embodiment: str = "menagerie_wxai"
  action_space: str = "ee_delta_pose_binary_gripper"
  base_pose: Pose = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> RobotConfig:
    _check_keys("robot", data, cls)
    action_space = data.get("action_space", cls.action_space)
    if action_space not in ACTION_SPACES:
      raise ConfigError(f"robot.action_space must be one of {ACTION_SPACES}, got {action_space!r}")
    return cls(
      embodiment=str(data.get("embodiment", cls.embodiment)),
      action_space=action_space,
      base_pose=_pose("robot.base_pose", data.get("base_pose", cls.base_pose)),
    )


@dataclass(frozen=True)
class CameraConfig:
  render_eye: tuple[float, float, float] = (0.45, 0.5, 0.5)
  render_target: tuple[float, float, float] = (-0.2, 0.0, 0.2)

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> CameraConfig:
    _check_keys("camera", data, cls)
    return cls(
      render_eye=_vec("camera.render_eye", data.get("render_eye", cls.render_eye), 3),  # type: ignore[arg-type]
      render_target=_vec(
        "camera.render_target", data.get("render_target", cls.render_target), 3
      ),  # type: ignore[arg-type]
    )


@dataclass(frozen=True)
class ObjectConfig:
  """One scene object.

  ``size`` depends on the primitive: box = full extents [x, y, z];
  sphere = [radius]; cylinder / capsule = [radius, full height]. Meshes use
  ``asset_path`` (visual) and optional ``collision_asset_path`` (collision is the
  convex hull MuJoCo computes for mesh geoms) with ``scale``.
  """

  name: str
  primitive: Primitive
  initial_poses: tuple[Pose, ...]
  dynamic: bool = True
  size: tuple[float, ...] = ()
  asset_path: Path | None = None
  collision_asset_path: Path | None = None
  scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
  density: float = 1000.0
  mass: float | None = None
  friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
  color: tuple[float, float, float, float] = (0.6, 0.6, 0.6, 1.0)

  @classmethod
  def from_dict(cls, data: dict[str, Any], base_dir: Path) -> ObjectConfig:
    name = data.get("name")
    where = f"objects[{name!r}]"
    if not isinstance(name, str) or not name:
      raise ConfigError(f"{where}: 'name' must be a non-empty string")
    if name in RESERVED_ENTITY_NAMES:
      raise ConfigError(f"{where}: name is reserved ({RESERVED_ENTITY_NAMES})")
    if "trajectory" in data:
      raise ConfigError(
        f"{where}: 'trajectory' was replaced by 'initial_poses' (list of reset poses)"
      )
    _check_keys(where, data, cls)

    primitive = data.get("primitive", "box")
    if primitive not in PRIMITIVES:
      raise ConfigError(f"{where}: primitive must be one of {PRIMITIVES}")

    poses = data.get("initial_poses")
    if not isinstance(poses, list) or not poses:
      raise ConfigError(f"{where}: 'initial_poses' must be a non-empty list of poses")
    initial_poses = tuple(
      _pose(f"{where}.initial_poses[{i}]", p) for i, p in enumerate(poses)
    )

    size_dims = {"box": 3, "sphere": 1, "cylinder": 2, "capsule": 2}
    size: tuple[float, ...] = ()
    asset_path = collision_asset_path = None
    if primitive == "mesh":
      if "asset_path" not in data:
        raise ConfigError(f"{where}: mesh objects need 'asset_path'")
      asset_path = (base_dir / data["asset_path"]).resolve()
      collision_asset_path = (
        (base_dir / data["collision_asset_path"]).resolve()
        if data.get("collision_asset_path")
        else asset_path
      )
      for p in (asset_path, collision_asset_path):
        if not p.exists():
          raise ConfigError(f"{where}: mesh file not found: {p}")
    else:
      size = _vec(f"{where}.size", data.get("size"), size_dims[primitive])
      if any(s <= 0 for s in size):
        raise ConfigError(f"{where}: size entries must be positive")

    scale_raw = data.get("scale", 1.0)
    scale = (
      (float(scale_raw),) * 3
      if isinstance(scale_raw, (int, float))
      else _vec(f"{where}.scale", scale_raw, 3)
    )
    friction_raw = data.get("friction", cls.friction)
    friction = (
      (float(friction_raw), cls.friction[1], cls.friction[2])
      if isinstance(friction_raw, (int, float))
      else _vec(f"{where}.friction", friction_raw, 3)
    )
    mass = data.get("mass")
    return cls(
      name=name,
      primitive=primitive,
      initial_poses=initial_poses,
      dynamic=bool(data.get("dynamic", True)),
      size=size,
      asset_path=asset_path,
      collision_asset_path=collision_asset_path,
      scale=scale,  # type: ignore[arg-type]
      density=float(data.get("density", cls.density)),
      mass=None if mass is None else float(mass),
      friction=friction,  # type: ignore[arg-type]
      color=_vec(f"{where}.color", data.get("color", cls.color), 4),  # type: ignore[arg-type]
    )


@dataclass(frozen=True)
class SceneConfig:
  scene_name: str
  robot: RobotConfig
  manipulated_object: str
  objects: tuple[ObjectConfig, ...]
  camera: CameraConfig = field(default_factory=CameraConfig)
  ground_altitude: float | None = None

  @property
  def manipulated(self) -> ObjectConfig:
    return next(o for o in self.objects if o.name == self.manipulated_object)

  @classmethod
  def from_dict(cls, data: dict[str, Any], base_dir: Path) -> SceneConfig:
    _check_keys("scene", data, cls)
    objects = tuple(ObjectConfig.from_dict(o, base_dir) for o in data.get("objects", []))
    if not objects:
      raise ConfigError("scene: 'objects' must be a non-empty list")
    names = [o.name for o in objects]
    if len(set(names)) != len(names):
      raise ConfigError(f"scene: object names must be unique, got {names}")
    manipulated = data.get("manipulated_object", names[0])
    if manipulated not in names:
      raise ConfigError(f"scene: manipulated_object {manipulated!r} not in objects {names}")
    if not next(o for o in objects if o.name == manipulated).dynamic:
      raise ConfigError(f"scene: manipulated_object {manipulated!r} must be dynamic")
    ground = data.get("ground_altitude")
    return cls(
      scene_name=str(data.get("scene_name", "scene")),
      robot=RobotConfig.from_dict(data.get("robot", {})),
      manipulated_object=manipulated,
      objects=objects,
      camera=CameraConfig.from_dict(data.get("camera", {})),
      ground_altitude=None if ground is None else float(ground),
    )


@dataclass(frozen=True)
class TimingConfig:
  sim_dt: float = 0.002
  control_hz: float = 20.0
  episode_steps: int = 300

  @property
  def decimation(self) -> int:
    exact = 1.0 / (self.control_hz * self.sim_dt)
    rounded = round(exact)
    if rounded < 1 or abs(exact - rounded) > 1e-6:
      raise ConfigError(
        f"timing: control_hz={self.control_hz} must divide 1/sim_dt={1 / self.sim_dt:g}"
      )
    return rounded

  @property
  def step_dt(self) -> float:
    return self.sim_dt * self.decimation


@dataclass(frozen=True)
class ActionConfig:
  ee_pos_scale: float = 0.01
  """Metres of TCP translation per step at |action| = 1."""
  ee_rot_scale: float = 0.05
  """Radians of TCP rotation per step at |action| = 1."""
  joint_delta_scale: float = 0.1
  """Radians of arm joint target change per step at |action| = 1."""


@dataclass(frozen=True)
class TaskConfig:
  """Reward, success and observation parameters of the grasp-lift-hold task."""

  lift_threshold: float = 0.01
  stable_hold_duration_sec: float = 1.0
  stable_linear_vel_tol: float = 0.05
  stable_angular_vel_tol: float = 0.35
  arm_joint_vel_tol: float = 0.5
  gripper_joint_vel_tol: float = 0.05
  reach_scale: float = 5.0
  reach_saturation_dist: float = 0.05
  w_reach: float = 0.5
  w_grasp: float = 1.0
  target_lift_height: float = 0.03
  lift_target_scale: float = 30.0
  w_lift_target: float = 1.0
  w_hold: float = 1.0
  w_success: float = 5.0
  w_xy: float = 0.0
  w_ori: float = 0.0
  xy_tolerance: float = 0.05
  orientation_tolerance: float = 0.3
  reward_normalization: float = 10.0
  include_is_grasped_obs: bool = True
  use_minimal_obs: bool = False
  grasp_min_force: float = 0.2
  grasp_max_angle_deg: float = 110.0
  timing: TimingConfig = field(default_factory=TimingConfig)
  action: ActionConfig = field(default_factory=ActionConfig)
  domain_randomization: dict[str, Any] = field(default_factory=dict)
  """Reserved: domain randomization terms (not implemented yet)."""
  curriculum: dict[str, Any] = field(default_factory=dict)
  """Reserved: curriculum terms (not implemented yet)."""

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> TaskConfig:
    _check_keys("task", data, cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
      if f.name not in data:
        continue
      value = data[f.name]
      if f.name == "timing":
        _check_keys("task.timing", value, TimingConfig)
        kwargs[f.name] = TimingConfig(
          sim_dt=float(value.get("sim_dt", TimingConfig.sim_dt)),
          control_hz=float(value.get("control_hz", TimingConfig.control_hz)),
          episode_steps=int(value.get("episode_steps", TimingConfig.episode_steps)),
        )
      elif f.name == "action":
        _check_keys("task.action", value, ActionConfig)
        kwargs[f.name] = ActionConfig(**{k: float(v) for k, v in value.items()})
      elif f.name in ("domain_randomization", "curriculum"):
        if not isinstance(value, dict):
          raise ConfigError(f"task.{f.name} must be an object")
        kwargs[f.name] = dict(value)
      elif f.type in ("bool",):
        kwargs[f.name] = bool(value)
      else:
        kwargs[f.name] = float(value)
    cfg = cls(**kwargs)
    if cfg.domain_randomization or cfg.curriculum:
      raise ConfigError(
        "task: domain_randomization / curriculum are reserved but not implemented yet"
      )
    if cfg.reward_normalization <= 0:
      raise ConfigError("task.reward_normalization must be positive")
    _ = cfg.timing.decimation  # validates timing
    return cfg


def _read_json(path: Path) -> dict[str, Any]:
  with open(path, encoding="utf-8") as f:
    data = json.load(f)
  if not isinstance(data, dict):
    raise ConfigError(f"{path}: top level must be a JSON object")
  return data


def load_scene_config(path: str | Path) -> SceneConfig:
  path = Path(path).resolve()
  return SceneConfig.from_dict(_read_json(path), base_dir=path.parent)


def load_task_config(path: str | Path) -> TaskConfig:
  return TaskConfig.from_dict(_read_json(Path(path)))
