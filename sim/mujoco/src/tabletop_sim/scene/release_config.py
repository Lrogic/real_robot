"""Release task configuration (JSON) and its validation.

The release task starts with the manipulated object grasped and near a goal
pose that is defined relative to other scene objects (the "relative objects",
e.g. two standing jenga blocks for a bridge). Its layout lives in the anchor
frame: origin on the support surface, z up, x/y turned by the anchor yaw. At
every start state the anchor is sampled in a disk with a yaw range, and each
relative object and the goal are placed at ``T_world_anchor @ pose_in_anchor``.

Release task JSON layout (see ``configs/release/tasks/release_bridge_wxai.json``,
which lists every option)::

  {
    <shared with the grasp task: speed / joint-limit / gripper-switch weights,
     reward_normalization, include_is_grasped_obs, grasp_min_force,
     grasp_max_angle_deg, timing, action, curriculum>,
    <release rewards and terminations: w_place ... topple_displacement_m>,
    "domain_randomization": {<grasp's, minus object_reset_radius /
                              object_reset_yaw_range_deg / size-variant
                              clearance, plus friction.support_object>},
    "goal_pose_noise": {<PoseNoiseSourceConfig>},
    "relative_placement": {"objects": {name: pose}, "goal_pose": pose,
                           "goal_symmetry": "box" | "none", "anchor_center": [x, y],
                           "anchor_radius": r, "anchor_yaw_deg": y0,
                           "anchor_yaw_range_deg": dy},
    "start": {"dataset": path, "ball_radius": r, "min_height_above_goal": h,
              "tcp_rotation_noise_std_deg": s, "transport_clearance": c,
              "settle_steps_range": [lo, hi]}
  }

Poses are ``[x, y, z, qw, qx, qy, qz]``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal

from tabletop_sim.scene.config import (
  ActionConfig,
  ConfigError,
  CurriculumConfig,
  DomainRandomizationConfig,
  FrictionCoefficients,
  FrictionRandomizationConfig,
  JointFrictionConfig,
  ObjectPoseNoiseConfig,
  ObjectSizeVariantsConfig,
  Pose,
  PoseNoiseSourceConfig,
  SceneConfig,
  TimingConfig,
  _check_keys,
  _pose,
  _read_json,
  _vec,
)

GOAL_SYMMETRIES = ("box", "none")


def _number(where: str, value: Any) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
    raise ConfigError(f"{where} must be a finite number, got {value!r}")
  return float(value)


@dataclass(frozen=True)
class RelativePlacementConfig:
  """The relative objects' layout and the goal pose, both in the anchor frame,
  and where the anchor is sampled."""

  objects: tuple[tuple[str, Pose], ...]
  """(scene object name, pose in the anchor frame) of every relative object."""
  goal_pose: Pose
  """Goal pose of the manipulated object in the anchor frame."""
  goal_symmetry: Literal["box", "none"] = "box"
  """``box``: the four orientations that differ by 180 deg about the object's own
  x, y or z axis are the same goal. ``none``: only the exact orientation."""
  anchor_center: tuple[float, float] = (-0.30, 0.15)
  """World xy (relative to each env origin) of the disk the anchor origin is drawn from."""
  anchor_radius: float = 0.05
  """Radius (m) of that disk; the anchor origin is uniform over its area."""
  anchor_yaw_deg: float = 0.0
  anchor_yaw_range_deg: float = 30.0
  """Anchor yaw ~ U[anchor_yaw_deg - range, anchor_yaw_deg + range] (deg), about world z."""

  @property
  def object_names(self) -> tuple[str, ...]:
    return tuple(name for name, _ in self.objects)

  def __post_init__(self) -> None:
    where = "task.relative_placement"
    if not self.objects:
      raise ConfigError(f"{where}.objects must name at least one object")
    if len(set(self.object_names)) != len(self.object_names):
      raise ConfigError(f"{where}.objects has duplicate names")
    if self.goal_symmetry not in GOAL_SYMMETRIES:
      raise ConfigError(f"{where}.goal_symmetry must be one of {GOAL_SYMMETRIES}")
    if not (math.isfinite(self.anchor_radius) and self.anchor_radius >= 0):
      raise ConfigError(f"{where}.anchor_radius must be finite and nonnegative")
    if not (math.isfinite(self.anchor_yaw_range_deg) and self.anchor_yaw_range_deg >= 0):
      raise ConfigError(f"{where}.anchor_yaw_range_deg must be finite and nonnegative")

  @classmethod
  def from_dict(cls, data: Any) -> RelativePlacementConfig:
    where = "task.relative_placement"
    if not isinstance(data, dict):
      raise ConfigError(f"{where} must be an object")
    _check_keys(where, data, cls)
    for key in ("objects", "goal_pose"):
      if key not in data:
        raise ConfigError(f"{where}.{key} is required")
    objects = data["objects"]
    if not isinstance(objects, dict):
      raise ConfigError(f"{where}.objects must map object names to poses")
    kwargs: dict[str, Any] = {
      "objects": tuple(
        (str(name), _pose(f"{where}.objects[{name!r}]", pose)) for name, pose in objects.items()
      ),
      "goal_pose": _pose(f"{where}.goal_pose", data["goal_pose"]),
    }
    if "goal_symmetry" in data:
      kwargs["goal_symmetry"] = data["goal_symmetry"]
    if "anchor_center" in data:
      kwargs["anchor_center"] = _vec(f"{where}.anchor_center", data["anchor_center"], 2)
    for name in ("anchor_radius", "anchor_yaw_deg", "anchor_yaw_range_deg"):
      if name in data:
        kwargs[name] = _number(f"{where}.{name}", data[name])
    return cls(**kwargs)


@dataclass(frozen=True)
class StartConfig:
  """The start-state dataset and how its states were produced (read by
  ``scripts/generate_release_starts.py``; the env checks the dataset was made
  with these values)."""

  dataset: Path | None = None
  """Start-state ``.pt`` file, relative to the task JSON; its ``.meta.json``
  sits beside it. None: the env cannot be built until one is generated."""
  ball_radius: float = 0.04
  """The arrival offset from the goal centre (world axes) is uniform in this ball..."""
  min_height_above_goal: float = 0.01
  """...restricted to offsets at least this far above the goal (a spherical cap)."""
  tcp_rotation_noise_std_deg: float = 3.0
  """After arriving, the TCP turns by roll / pitch / yaw ~ N(0, std) about its own
  axes, each clipped at 2 std. The object turns with it."""
  transport_clearance: float = 0.05
  """Gap (m) between the lowest finger / object point and the top of the
  highest relative object while crossing over to the goal."""
  settle_steps_range: tuple[int, int] = (0, 10)
  """Control steps waited after the last waypoint, uniform in [lo, hi], before
  the state is checked and recorded."""

  def __post_init__(self) -> None:
    where = "task.start"
    if not (math.isfinite(self.ball_radius) and self.ball_radius > 0):
      raise ConfigError(f"{where}.ball_radius must be positive")
    h = self.min_height_above_goal
    if not (math.isfinite(h) and 0 <= h < self.ball_radius):
      raise ConfigError(f"{where}.min_height_above_goal must be in [0, ball_radius)")
    for name in ("tcp_rotation_noise_std_deg", "transport_clearance"):
      value = getattr(self, name)
      if not (math.isfinite(value) and value >= 0):
        raise ConfigError(f"{where}.{name} must be finite and nonnegative, got {value}")
    lo, hi = self.settle_steps_range
    if not (0 <= lo <= hi):
      raise ConfigError(f"{where}.settle_steps_range must satisfy 0 <= lo <= hi")

  def generation_params(self) -> dict[str, Any]:
    """Everything except the dataset path: what a dataset must have been made with."""
    out = asdict(self)
    out.pop("dataset")
    out["settle_steps_range"] = list(self.settle_steps_range)
    return out

  @classmethod
  def from_dict(cls, data: Any, base_dir: Path) -> StartConfig:
    where = "task.start"
    if not isinstance(data, dict):
      raise ConfigError(f"{where} must be an object")
    _check_keys(where, data, cls)
    kwargs: dict[str, Any] = {}
    if data.get("dataset") is not None:
      if not isinstance(data["dataset"], str) or not data["dataset"]:
        raise ConfigError(f"{where}.dataset must be a non-empty string or null")
      kwargs["dataset"] = (base_dir / data["dataset"]).resolve()
    for name in (
      "ball_radius",
      "min_height_above_goal",
      "tcp_rotation_noise_std_deg",
      "transport_clearance",
    ):
      if name in data:
        kwargs[name] = _number(f"{where}.{name}", data[name])
    if "settle_steps_range" in data:
      value = data["settle_steps_range"]
      if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) for v in value)
      ):
        raise ConfigError(f"{where}.settle_steps_range must be two integers, got {value!r}")
      kwargs["settle_steps_range"] = (value[0], value[1])
    return cls(**kwargs)


@dataclass(frozen=True)
class ReleaseFrictionConfig(FrictionRandomizationConfig):
  """Grasp's contact-pair friction randomization plus a third interface:
  manipulated object vs. every relative object."""

  support_object: FrictionCoefficients = (0.5, 0.005, 0.0001)

  def __post_init__(self) -> None:
    super().__post_init__()
    value = self.support_object
    if len(value) != 3 or not all(math.isfinite(v) and v > 0 for v in value):
      raise ConfigError(
        "domain_randomization.friction.support_object must be 3 finite positive coefficients"
      )

  @classmethod
  def from_dict(cls, data: Any) -> ReleaseFrictionConfig:
    if not isinstance(data, dict):
      raise ConfigError("task.domain_randomization.friction must be an object")
    rest = {k: v for k, v in data.items() if k != "support_object"}
    base = super().from_dict(rest)
    if "support_object" not in data:
      return base  # type: ignore[return-value]
    value = _vec("task.domain_randomization.friction.support_object", data["support_object"], 3)
    return replace(base, support_object=value)  # type: ignore[arg-type]


_RELEASE_DR_DROPPED = ("object_reset_radius", "object_reset_yaw_range_deg")


@dataclass(frozen=True)
class ReleaseDomainRandomizationConfig:
  """Grasp's domain randomization without the reset placement (the start
  comes from the dataset) and with block-support friction."""

  stiffness_damping_enabled: bool = False
  stiffness_scale_range: tuple[float, float] = (0.8, 1.2)
  damping_scale_range: tuple[float, float] = (0.8, 1.2)
  object_size_variants: ObjectSizeVariantsConfig = field(default_factory=ObjectSizeVariantsConfig)
  """``enabled`` and ``scales`` only; ``scales`` must be the dataset's."""
  friction: ReleaseFrictionConfig = field(default_factory=ReleaseFrictionConfig)
  object_pose_noise: ObjectPoseNoiseConfig = field(default_factory=ObjectPoseNoiseConfig)
  joint_friction: JointFrictionConfig = field(default_factory=JointFrictionConfig)
  object_pose_delay_s: float = 0.0
  action_hold_probability: float = 0.0

  def as_grasp(self) -> DomainRandomizationConfig:
    """The equivalent grasp config (for code shared with the grasp env)."""
    kwargs = {f.name: getattr(self, f.name) for f in fields(self)}
    return DomainRandomizationConfig(**kwargs, object_reset_radius=0.0, object_reset_yaw_range_deg=0.0)

  @classmethod
  def from_dict(cls, data: Any) -> ReleaseDomainRandomizationConfig:
    where = "task.domain_randomization"
    if not isinstance(data, dict):
      raise ConfigError(f"{where} must be an object")
    dropped = sorted(set(data) & set(_RELEASE_DR_DROPPED))
    if dropped:
      raise ConfigError(
        f"{where}: {dropped} do not apply to release (the start state comes from the dataset)"
      )
    _check_keys(where, data, cls)
    sizes = data.get("object_size_variants")
    if isinstance(sizes, dict) and "clearance" in sizes:
      raise ConfigError(
        f"{where}.object_size_variants.clearance does not apply to release "
        "(the object starts in the hand)"
      )
    shared = {k: v for k, v in data.items() if k != "friction"}
    grasp = DomainRandomizationConfig.from_dict(shared)  # validates the shared options
    kwargs = {f.name: getattr(grasp, f.name) for f in fields(cls) if f.name != "friction"}
    if "friction" in data:
      kwargs["friction"] = ReleaseFrictionConfig.from_dict(data["friction"])
    return cls(**kwargs)


def _default_goal_noise() -> PoseNoiseSourceConfig:
  return PoseNoiseSourceConfig(
    translation_std_m=(0.003, 0.003, 0.003), rotation_std_deg=(2.0, 2.0, 2.0)
  )


def _default_curriculum() -> CurriculumConfig:
  return CurriculumConfig(metric_name="released_frac")


@dataclass(frozen=True)
class ReleaseTaskConfig:
  """Reward, termination, observation and start-state parameters of the release task."""

  relative_placement: RelativePlacementConfig
  start: StartConfig = field(default_factory=StartConfig)

  # Shared with the grasp task (same meaning as in TaskConfig).
  arm_speed_threshold: float = 0.5
  gripper_speed_threshold: float = 0.5
  w_arm_speed: float = -0.1
  w_gripper_speed: float = 0.0
  w_joint_pos_limits: float = -10.0
  w_gripper_switch: float = 0.0
  reward_normalization: float = 10.0
  include_is_grasped_obs: bool = True
  grasp_min_force: float = 0.2
  grasp_max_angle_deg: float = 110.0

  # Release rewards.
  w_place: float = 1.0
  """Weight of ``place = 1 - tanh(place_pos_scale * d_pos + place_rot_scale * d_rot)``."""
  place_pos_scale: float = 30.0
  """1/m."""
  place_rot_scale: float = 2.0
  """1/rad. d_rot is the angle to the closest symmetric goal orientation."""
  w_released: float = 2.0
  """Weight of ``released``: within the tolerances, no finger touches the
  object, the gripper is open, and the object is still."""
  release_pos_tol: float = 0.01
  release_rot_tol_deg: float = 10.0
  release_max_closure: float = 0.5
  """Gripper closure (0 open, 1 closed) at or below which the gripper counts as open."""
  w_retreat: float = 1.0
  """Weight of ``retreat = released * clamp(tcp_z - object_z, 0, retreat_height) / retreat_height``."""
  retreat_height: float = 0.05
  w_robot_collision: float = -1.0
  """Weight (nonpositive) of the fraction of the control step's substeps in which
  a moving robot part touches the support surface or a relative object (the
  ``touched_relative`` metric logs the relative objects alone)."""
  w_support_disturb: float = -1.0
  """Weight (nonpositive) of the mean over relative objects of
  ``tanh(disturb_pos_scale * displacement + disturb_rot_scale * rotation)``
  from their reset poses."""
  disturb_pos_scale: float = 50.0
  disturb_rot_scale: float = 5.0
  w_terminated: float = -5.0
  """Weight (nonpositive) on the step an episode terminates (drop or topple)."""
  w_release_moving: float = -5.0
  """Weight (nonpositive) on the control step whose gripper command switches from
  closed to open, times how much the object is moving then (1 - stillness, the
  form ``released`` uses): opening on a still object costs about nothing. 0
  disables it, together with the ``moving_release_frac`` metric."""
  release_scale_by_stillness: bool = False
  """Scale ``released`` and ``retreat`` by how still the object was when the gripper was
  last commanded open: (1 - tanh(speed_scale * speed)) * (1 - tanh(spin_scale * spin)),
  recorded at that step and kept until the next opening or the episode ends. A release at
  rest keeps the full reward; a hurried one keeps little of it. Adds the
  ``release_stillness`` metric (the recorded value; 1 when the gripper never opened)."""
  release_stillness_speed_scale: float = 11.0
  """1/(m/s). 11 puts the score's midpoint at 0.05 m/s, where a release toward the edge of
  the 25 mm support tops starts to cost sideways margin."""
  release_stillness_spin_scale: float = 0.5
  """1/(rad/s). Spin barely affects whether a released bridge stays on the supports, so it
  is scored leniently (midpoint about 1.1 rad/s)."""

  # Terminations.
  drop_below_goal_m: float = 0.03
  """The object counts as dropped when it is not grasped and its centre is this
  far below the goal centre."""
  topple_tilt_deg: float = 30.0
  topple_displacement_m: float = 0.03
  """A relative object counts as toppled past this tilt or displacement from its reset pose."""

  goal_pose_noise: PoseNoiseSourceConfig = field(default_factory=_default_goal_noise)
  """Error of the actor's goal: one draw at reset, kept for the episode, scaled
  by the episode difficulty. Off unless enabled."""

  timing: TimingConfig = field(default_factory=lambda: TimingConfig(episode_steps=150))
  action: ActionConfig = field(default_factory=ActionConfig)
  domain_randomization: ReleaseDomainRandomizationConfig = field(
    default_factory=ReleaseDomainRandomizationConfig
  )
  curriculum: CurriculumConfig = field(default_factory=_default_curriculum)

  @property
  def object_pose_delay_substeps(self) -> int:
    delay = self.domain_randomization.object_pose_delay_s
    exact = delay / self.timing.sim_dt
    if abs(exact - round(exact)) > 1e-6:
      raise ConfigError(
        f"task.domain_randomization.object_pose_delay_s={delay} must be a multiple of "
        f"timing.sim_dt={self.timing.sim_dt}"
      )
    return round(exact)

  @property
  def command_delay_substeps(self) -> int:
    delay = self.action.command_delay_s
    exact = delay / self.timing.sim_dt
    if abs(exact - round(exact)) > 1e-6:
      raise ConfigError(
        f"task.action.command_delay_s={delay} must be a multiple of timing.sim_dt={self.timing.sim_dt}"
      )
    return round(exact)

  def __post_init__(self) -> None:
    self.goal_pose_noise.validate("task.goal_pose_noise")
    for name in (
      "w_arm_speed",
      "w_gripper_speed",
      "w_joint_pos_limits",
      "w_gripper_switch",
      "w_robot_collision",
      "w_support_disturb",
      "w_terminated",
      "w_release_moving",
    ):
      value = getattr(self, name)
      if not (math.isfinite(value) and value <= 0):
        raise ConfigError(f"task.{name} must be finite and nonpositive, got {value}")
    for name in (
      "arm_speed_threshold",
      "gripper_speed_threshold",
      "w_place",
      "w_released",
      "w_retreat",
      "place_pos_scale",
      "place_rot_scale",
      "release_stillness_speed_scale",
      "release_stillness_spin_scale",
      "disturb_pos_scale",
      "disturb_rot_scale",
    ):
      value = getattr(self, name)
      if not (math.isfinite(value) and value >= 0):
        raise ConfigError(f"task.{name} must be finite and nonnegative, got {value}")
    for name in (
      "release_pos_tol",
      "release_rot_tol_deg",
      "retreat_height",
      "drop_below_goal_m",
      "topple_tilt_deg",
      "topple_displacement_m",
      "reward_normalization",
    ):
      value = getattr(self, name)
      if not (math.isfinite(value) and value > 0):
        raise ConfigError(f"task.{name} must be finite and positive, got {value}")
    if not 0.0 <= self.release_max_closure <= 1.0:
      raise ConfigError("task.release_max_closure must be in [0, 1]")
    _ = self.timing.decimation
    _ = self.object_pose_delay_substeps
    _ = self.command_delay_substeps

  def validate_scene(self, scene: SceneConfig) -> None:
    """The relative objects exist, are dynamic, and are neither the manipulated
    object nor the support surface."""
    names = {o.name: o for o in scene.objects}
    support = scene.support_surface.object if scene.support_surface else None
    for name in self.relative_placement.object_names:
      where = f"task.relative_placement.objects[{name!r}]"
      if name not in names:
        raise ConfigError(f"{where}: no such scene object (have {sorted(names)})")
      if name in (scene.manipulated_object, support):
        raise ConfigError(f"{where}: must not be the manipulated object or the support surface")
      if not names[name].dynamic:
        raise ConfigError(f"{where}: must be dynamic")
    if scene.support_surface is None:
      raise ConfigError("release needs the scene's 'support_surface' (the anchor sits on it)")

  @classmethod
  def from_dict(cls, data: dict[str, Any], base_dir: Path) -> ReleaseTaskConfig:
    _check_keys("task", data, cls)
    if "relative_placement" not in data:
      raise ConfigError("task.relative_placement is required")
    kwargs: dict[str, Any] = {
      "relative_placement": RelativePlacementConfig.from_dict(data["relative_placement"])
    }
    for f in fields(cls):
      if f.name not in data or f.name == "relative_placement":
        continue
      value = data[f.name]
      if f.name == "start":
        kwargs[f.name] = StartConfig.from_dict(value, base_dir)
      elif f.name == "timing":
        _check_keys("task.timing", value, TimingConfig)
        kwargs[f.name] = TimingConfig(
          sim_dt=float(value.get("sim_dt", TimingConfig.sim_dt)),
          control_hz=float(value.get("control_hz", TimingConfig.control_hz)),
          episode_steps=int(value.get("episode_steps", 150)),
        )
      elif f.name == "action":
        _check_keys("task.action", value, ActionConfig)
        kwargs[f.name] = ActionConfig(
          **{k: v if k == "ee_goal_interpolation" else float(v) for k, v in value.items()}
        )
      elif f.name == "domain_randomization":
        kwargs[f.name] = ReleaseDomainRandomizationConfig.from_dict(value)
      elif f.name == "curriculum":
        kwargs[f.name] = CurriculumConfig.from_dict({"metric_name": "released_frac", **value})
      elif f.name == "goal_pose_noise":
        kwargs[f.name] = PoseNoiseSourceConfig.from_dict(
          value, "task.goal_pose_noise", _default_goal_noise()
        )
      elif f.name in ("include_is_grasped_obs", "release_scale_by_stillness"):
        kwargs[f.name] = bool(value)
      else:
        kwargs[f.name] = _number(f"task.{f.name}", value)
    return cls(**kwargs)


def load_release_task_config(path: str | Path) -> ReleaseTaskConfig:
  path = Path(path).resolve()
  return ReleaseTaskConfig.from_dict(_read_json(path), base_dir=path.parent)
