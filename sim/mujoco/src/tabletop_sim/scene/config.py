"""Scene and task configuration files (JSON) and their validation.

Scene JSON layout::

  {
    "scene_name": str,
    "robot": {"embodiment": str, "action_space": str, "base_pose": [x,y,z,qw,qx,qy,qz]},
    "camera": {"render_eye": [x,y,z], "render_target": [x,y,z]},     # optional
    "ground_altitude": float | null,                                  # optional
    "support_surface": {"object": str} | {"height": float},          # optional
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

ACTION_SPACES = (
  "ee_delta_pose_binary_gripper",
  "delta_joint_abs_gripper",
  "ee_delta_pose_delta_gripper",
  "delta_joint_delta_gripper",
)
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
  convex hull MuJoCo computes for mesh geoms) with ``scale``. Optional
  ``collision_size`` replaces that hull with a box of those full extents, in the
  mesh frame. Optional ``texture_path`` is a diffuse image for the visual mesh.
  When it is set it replaces ``color``; otherwise the geom uses ``color``.
  """

  name: str
  primitive: Primitive
  initial_poses: tuple[Pose, ...]
  dynamic: bool = True
  size: tuple[float, ...] = ()
  asset_path: Path | None = None
  collision_asset_path: Path | None = None
  texture_path: Path | None = None
  collision_size: tuple[float, ...] = ()
  """Full box extents [x, y, z] used instead of the mesh convex hull."""
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
    asset_path = collision_asset_path = texture_path = None
    collision_size: tuple[float, ...] = ()
    raw_texture = data.get("texture_path")
    raw_collision_size = data.get("collision_size")
    if raw_texture is not None and primitive != "mesh":
      raise ConfigError(f"{where}: 'texture_path' is only valid for mesh objects")
    if raw_collision_size is not None and primitive != "mesh":
      raise ConfigError(f"{where}: 'collision_size' is only valid for mesh objects")
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
      if raw_texture is not None:
        if not isinstance(raw_texture, str) or not raw_texture:
          raise ConfigError(f"{where}: 'texture_path' must be a non-empty string")
        texture_path = (base_dir / raw_texture).resolve()
        if not texture_path.is_file():
          raise ConfigError(f"{where}: texture file not found: {texture_path}")
      if raw_collision_size is not None:
        collision_size = _vec(f"{where}.collision_size", raw_collision_size, 3)
        if any(s <= 0 for s in collision_size):
          raise ConfigError(f"{where}: collision_size entries must be positive")
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
      texture_path=texture_path,
      collision_size=collision_size,
      scale=scale,  # type: ignore[arg-type]
      density=float(data.get("density", cls.density)),
      mass=None if mass is None else float(mass),
      friction=friction,  # type: ignore[arg-type]
      color=_vec(f"{where}.color", data.get("color", cls.color), 4),  # type: ignore[arg-type]
    )


@dataclass(frozen=True)
class SupportSurfaceConfig:
  """The horizontal plane the manipulated object rests on at reset: the top face
  of a static box ``object``, or an explicit world-frame ``height`` (relative to
  each env origin). Exactly one is set."""

  object: str | None = None
  height: float | None = None

  @classmethod
  def from_dict(cls, data: Any) -> SupportSurfaceConfig:
    where = "scene.support_surface"
    if not isinstance(data, dict):
      raise ConfigError(f"{where} must be an object")
    _check_keys(where, data, cls)
    if len(data) != 1:
      raise ConfigError(f"{where} needs exactly one of 'object' or 'height', got {sorted(data)}")
    if "object" in data:
      if not isinstance(data["object"], str) or not data["object"]:
        raise ConfigError(f"{where}.object must be a non-empty string")
      return cls(object=data["object"])
    height = data["height"]
    if isinstance(height, bool) or not isinstance(height, (int, float)) or not math.isfinite(height):
      raise ConfigError(f"{where}.height must be a finite number, got {height!r}")
    return cls(height=float(height))


def _box_top_height(obj: ObjectConfig) -> float:
  """World z of the upward-facing face of a static, horizontally placed box."""
  where = f"scene.support_surface: object {obj.name!r}"
  if obj.primitive != "box" or obj.dynamic or len(obj.initial_poses) != 1:
    raise ConfigError(f"{where} must be a static box with exactly one initial pose")
  pose = obj.initial_poses[0]
  w, x, y, z = pose[3:]
  # Bottom row of the rotation matrix: world z of each box axis.
  up = (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y))
  axis = max(range(3), key=lambda i: abs(up[i]))
  if abs(abs(up[axis]) - 1.0) > 1e-6:
    raise ConfigError(f"{where} has no horizontal face (one box axis must be vertical)")
  return pose[2] + obj.size[axis] * obj.scale[axis] / 2


@dataclass(frozen=True)
class SceneConfig:
  scene_name: str
  robot: RobotConfig
  manipulated_object: str
  objects: tuple[ObjectConfig, ...]
  camera: CameraConfig = field(default_factory=CameraConfig)
  ground_altitude: float | None = None
  support_surface: SupportSurfaceConfig | None = None

  @property
  def manipulated(self) -> ObjectConfig:
    return next(o for o in self.objects if o.name == self.manipulated_object)

  def support_height(self) -> float:
    """z of the support plane relative to each env origin."""
    s = self.support_surface
    if s is None:
      raise ConfigError("scene: 'support_surface' is not set")
    if s.height is not None:
      return s.height
    obj = next((o for o in self.objects if o.name == s.object), None)
    if obj is None:
      raise ConfigError(f"scene.support_surface: object {s.object!r} is not in objects")
    if obj.name == self.manipulated_object:
      raise ConfigError("scene.support_surface: object must not be the manipulated object")
    return _box_top_height(obj)

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
    support = data.get("support_surface")
    scene = cls(
      scene_name=str(data.get("scene_name", "scene")),
      robot=RobotConfig.from_dict(data.get("robot", {})),
      manipulated_object=manipulated,
      objects=objects,
      camera=CameraConfig.from_dict(data.get("camera", {})),
      ground_altitude=None if ground is None else float(ground),
      support_surface=None if support is None else SupportSurfaceConfig.from_dict(support),
    )
    if scene.support_surface is not None:
      scene.support_height()  # validates the selected object
    return scene


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
  gripper_delta_scale: float = 0.01
  """Metres of carriage target change per step at |action| = 1 (delta gripper
  only). Each finger moves this much, so the gap changes by twice as much."""


DEFAULT_SIZE_SCALES = tuple(round(0.90 + 0.01 * i, 2) for i in range(21))


@dataclass(frozen=True)
class ObjectSizeVariantsConfig:
  """Fixed per-env size variants of the manipulated (mesh) object.

  Each env is assigned one isotropic factor from ``scales``, applied on top of
  the object's configured ``scale``, once at simulation start (seeded by the
  env seed); it never changes on reset. At constant density mass scales by s^3
  and inertia by s^5; an explicit ``mass`` is the s = 1 mass. Every reset
  re-seats the object so its lowest collision point is ``clearance`` above the
  scene's ``support_surface``. Independent of the curriculum difficulty.
  """

  enabled: bool = False
  scales: tuple[float, ...] = DEFAULT_SIZE_SCALES
  clearance: float = 0.0
  """Gap (m) between the lowest collision point and the support surface at reset."""

  def __post_init__(self) -> None:
    where = "domain_randomization.object_size_variants"
    if not isinstance(self.enabled, bool):
      raise ConfigError(f"{where}.enabled must be a bool, got {self.enabled!r}")
    if not self.scales or not all(math.isfinite(s) and s > 0 for s in self.scales):
      raise ConfigError(f"{where}.scales must be a non-empty list of positive numbers")
    if len(set(self.scales)) != len(self.scales):
      raise ConfigError(f"{where}.scales must be unique, got {list(self.scales)}")
    if not (math.isfinite(self.clearance) and self.clearance >= 0):
      raise ConfigError(f"{where}.clearance must be finite and nonnegative, got {self.clearance}")

  @classmethod
  def from_dict(cls, data: Any) -> ObjectSizeVariantsConfig:
    where = "task.domain_randomization.object_size_variants"
    if not isinstance(data, dict):
      raise ConfigError(f"{where} must be an object")
    _check_keys(where, data, cls)
    kwargs: dict[str, Any] = {}
    if "enabled" in data:
      kwargs["enabled"] = data["enabled"]
    if "scales" in data:
      value = data["scales"]
      if not isinstance(value, list) or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) for v in value
      ):
        raise ConfigError(f"{where}.scales must be a list of numbers, got {value!r}")
      kwargs["scales"] = tuple(float(v) for v in value)
    if "clearance" in data:
      value = data["clearance"]
      if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where}.clearance must be a number, got {value!r}")
      kwargs["clearance"] = float(value)
    return cls(**kwargs)


@dataclass(frozen=True)
class DomainRandomizationConfig:
  object_reset_radius: float = 0.05
  """Radius (m) of the disk around each object's default position in which the
  manipulated object is placed uniformly at every reset."""
  object_reset_yaw_range_deg: float = 20.0
  """Half-width (deg) of the uniform yaw range about the world z-axis applied to
  the manipulated object at every reset: yaw ~ U[-r, r]."""
  stiffness_damping_enabled: bool = False
  """Scale the embodiment's actuator kp and its damping (joint damping or
  actuator kv, see ``StiffnessDampingTargets``) at every reset."""
  stiffness_scale_range: tuple[float, float] = (0.8, 1.2)
  """kp multiplier range [lower, upper] at episode difficulty 1. At difficulty d
  each bound is 1 + d * (bound - 1), so d = 0 gives the nominal gains."""
  damping_scale_range: tuple[float, float] = (0.8, 1.2)
  """Damping multiplier range at difficulty 1, interpolated the same way."""
  object_size_variants: ObjectSizeVariantsConfig = field(default_factory=ObjectSizeVariantsConfig)

  def __post_init__(self) -> None:
    where = "domain_randomization"
    if not isinstance(self.stiffness_damping_enabled, bool):
      raise ConfigError(
        f"{where}.stiffness_damping_enabled must be a bool, "
        f"got {self.stiffness_damping_enabled!r}"
      )
    for name in ("stiffness_scale_range", "damping_scale_range"):
      lower, upper = getattr(self, name)
      if not (math.isfinite(lower) and math.isfinite(upper) and 0 < lower <= upper):
        raise ConfigError(
          f"{where}.{name} must be finite with 0 < lower <= upper, got [{lower}, {upper}]"
        )
    if not (math.isfinite(self.object_reset_radius) and self.object_reset_radius >= 0):
      raise ConfigError(
        "domain_randomization.object_reset_radius must be finite and nonnegative, "
        f"got {self.object_reset_radius}"
      )
    yaw_range = self.object_reset_yaw_range_deg
    if not (math.isfinite(yaw_range) and yaw_range >= 0):
      raise ConfigError(
        "domain_randomization.object_reset_yaw_range_deg must be finite and nonnegative, "
        f"got {self.object_reset_yaw_range_deg}"
      )

  @classmethod
  def from_dict(cls, data: Any) -> DomainRandomizationConfig:
    where = "task.domain_randomization"
    if not isinstance(data, dict):
      raise ConfigError(f"{where} must be an object")
    _check_keys(where, data, cls)
    kwargs: dict[str, Any] = {}
    for name in ("object_reset_radius", "object_reset_yaw_range_deg"):
      if name in data:
        kwargs[name] = float(data[name])
    if "stiffness_damping_enabled" in data:
      kwargs["stiffness_damping_enabled"] = data["stiffness_damping_enabled"]
    for name in ("stiffness_scale_range", "damping_scale_range"):
      if name in data:
        value = data[name]
        if not isinstance(value, list) or any(
          isinstance(v, bool) or not isinstance(v, (int, float)) for v in value
        ):
          raise ConfigError(f"{where}.{name} must be a list of 2 numbers, got {value!r}")
        kwargs[name] = _vec(f"{where}.{name}", value, 2)
    if "object_size_variants" in data:
      kwargs["object_size_variants"] = ObjectSizeVariantsConfig.from_dict(
        data["object_size_variants"]
      )
    return cls(**kwargs)


@dataclass(frozen=True)
class CurriculumConfig:
  """Performance-based difficulty curriculum. A global difficulty d in
  [initial_difficulty, max_difficulty] steps up when the mean completed-episode
  score at the current stage beats ``performance_threshold``; each episode keeps
  the d it started with."""

  enabled: bool = False
  metric_name: str = "grasped_lifted_frac"
  """Episode metric (``reduce="sum"``) whose per-episode total is the score."""
  initial_difficulty: float = 0.0
  max_difficulty: float = 1.0
  difficulty_increment: float = 0.05
  """Difficulty added per promotion (capped at ``max_difficulty``)."""
  performance_threshold: float = 0.70
  """Promote when the window's mean score is strictly greater than this."""
  min_completed_episodes: int = 5000
  """Eligible completed episodes needed before the window is evaluated."""

  def __post_init__(self) -> None:
    where = "task.curriculum"
    if not isinstance(self.enabled, bool):
      raise ConfigError(f"{where}.enabled must be a bool, got {self.enabled!r}")
    if not isinstance(self.metric_name, str) or not self.metric_name:
      raise ConfigError(f"{where}.metric_name must be a non-empty string")
    for name in ("initial_difficulty", "max_difficulty", "performance_threshold"):
      value = getattr(self, name)
      if not (math.isfinite(value) and 0.0 <= value <= 1.0):
        raise ConfigError(f"{where}.{name} must be in [0, 1], got {value}")
    if self.initial_difficulty > self.max_difficulty:
      raise ConfigError(
        f"{where}.initial_difficulty ({self.initial_difficulty}) must not exceed "
        f"max_difficulty ({self.max_difficulty})"
      )
    if not (math.isfinite(self.difficulty_increment) and self.difficulty_increment > 0):
      raise ConfigError(
        f"{where}.difficulty_increment must be positive, got {self.difficulty_increment}"
      )
    count = self.min_completed_episodes
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
      raise ConfigError(f"{where}.min_completed_episodes must be an integer >= 1, got {count!r}")

  @classmethod
  def from_dict(cls, data: Any) -> CurriculumConfig:
    if not isinstance(data, dict):
      raise ConfigError("task.curriculum must be an object")
    _check_keys("task.curriculum", data, cls)
    kwargs: dict[str, Any] = {}
    for name in ("enabled", "metric_name"):
      if name in data:
        kwargs[name] = data[name]
    for name in ("initial_difficulty", "max_difficulty", "difficulty_increment",
                 "performance_threshold"):
      if name in data:
        value = data[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
          raise ConfigError(f"task.curriculum.{name} must be a number, got {value!r}")
        kwargs[name] = float(value)
    if "min_completed_episodes" in data:
      value = data["min_completed_episodes"]
      if isinstance(value, float) and value.is_integer():
        value = int(value)
      kwargs["min_completed_episodes"] = value
    return cls(**kwargs)


@dataclass(frozen=True)
class TaskConfig:
  """Reward, metric and observation parameters of the grasp-lift-hold task."""

  lift_threshold: float = 0.01
  """Object-to-target distance (m) within which the object counts as lifted."""
  reach_scale: float = 5.0
  w_reach: float = 0.5
  w_grasp: float = 1.0
  target_lift_height: float = 0.03
  lift_target_scale: float = 30.0
  w_lift_target: float = 1.0
  w_hold: float = 1.0
  arm_speed_threshold: float = 0.5
  """Per-joint arm speed (rad/s) above which the arm speed penalty applies.
  A tuning value, not a hardware limit."""
  gripper_speed_threshold: float = 0.5
  """Per-joint gripper carriage speed (m/s) above which the gripper speed
  penalty applies. A tuning value, not a hardware limit."""
  w_arm_speed: float = -0.1
  """Arm speed penalty weight (nonpositive)."""
  w_gripper_speed: float = 0.0
  """Gripper speed penalty weight (nonpositive)."""
  w_joint_pos_limits: float = -10.0
  """Weight (nonpositive) of the penalty on arm joint positions outside their
  soft limits (the embodiment's ``soft_joint_pos_limit_factor`` of each model
  range). The gripper is not penalized."""
  reward_normalization: float = 10.0
  include_is_grasped_obs: bool = True
  """Adds ``is_grasped`` to the critic observations (never the actor)."""
  grasp_min_force: float = 0.2
  grasp_max_angle_deg: float = 110.0
  timing: TimingConfig = field(default_factory=TimingConfig)
  action: ActionConfig = field(default_factory=ActionConfig)
  domain_randomization: DomainRandomizationConfig = field(default_factory=DomainRandomizationConfig)
  curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)

  def __post_init__(self) -> None:
    for name in ("arm_speed_threshold", "gripper_speed_threshold"):
      value = getattr(self, name)
      if not (math.isfinite(value) and value >= 0):
        raise ConfigError(f"task.{name} must be finite and nonnegative, got {value}")
    for name in ("w_arm_speed", "w_gripper_speed", "w_joint_pos_limits"):
      value = getattr(self, name)
      if not (math.isfinite(value) and value <= 0):
        raise ConfigError(f"task.{name} must be finite and nonpositive, got {value}")

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
      elif f.name == "domain_randomization":
        kwargs[f.name] = DomainRandomizationConfig.from_dict(value)
      elif f.name == "curriculum":
        kwargs[f.name] = CurriculumConfig.from_dict(value)
      elif f.type in ("bool",):
        kwargs[f.name] = bool(value)
      else:
        kwargs[f.name] = float(value)
    cfg = cls(**kwargs)
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
