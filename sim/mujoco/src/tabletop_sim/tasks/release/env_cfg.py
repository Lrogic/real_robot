"""Build the mjlab environment config for the release task from a scene JSON
and a release task JSON. Shares grasp's building blocks (actions, entities,
sensors, friction pairs, pose noise, viewer, curriculum) without changing
them."""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import joint_pos_limits
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.envs.mdp.rewards import is_terminated
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, SensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from tabletop_sim import CONFIGS_DIR
from tabletop_sim.embodiments import EmbodimentCfg, build_entity_cfg, get_embodiment
from tabletop_sim.embodiments.base import load_spec
from tabletop_sim.scene import SceneConfig, load_scene_config, scene_entity_cfgs
from tabletop_sim.scene.builder import ContactPairGroup, scene_pair_geoms
from tabletop_sim.scene.release_config import ReleaseTaskConfig, load_release_task_config
from tabletop_sim.tasks.grasp import env_cfg as grasp_cfg
from tabletop_sim.tasks.grasp import mdp as grasp_mdp
from tabletop_sim.tasks.release import mdp, starts

COMMAND_NAME = "release_goal"
GOAL_PERCEPTION_NAME = "goal_perception"
PERCEPTION_NAME = grasp_cfg.PERCEPTION_NAME
ACTION_HOLD_NAME = grasp_cfg.ACTION_HOLD_NAME
GRIPPER_ACTION_NAME = grasp_cfg.GRIPPER_ACTION_NAME
POSE_HISTORY_SENSOR = grasp_cfg.POSE_HISTORY_SENSOR
FINGER_SENSORS = grasp_cfg.FINGER_SENSORS
SUPPORT_PAIRS = "support_object"
DEFAULT_SCENE = CONFIGS_DIR / "release" / "scenes" / "tabletop_jenga_tower_wxai.json"
DEFAULT_TASK = CONFIGS_DIR / "release" / "tasks" / "release_tower_wxai.json"


def contact_sensor_name(target: str) -> str:
  return f"robot_{target}_contact"


def robot_contact_sensor(emb: EmbodimentCfg, target: str, decimation: int) -> ContactSensorCfg:
  """Contacts between the robot's moving bodies (fingers included, never the
  fixed base) and ``target``, one force sample per substep of a control step.
  Grasp's table-contact sensor for an arbitrary object."""
  robot_spec = load_spec(emb)
  first_moving_body = robot_spec.joint(emb.arm_joint_names[0]).parent.name
  return ContactSensorCfg(
    name=contact_sensor_name(target),
    primary=ContactMatch(mode="subtree", pattern=first_moving_body, entity="robot"),
    secondary=ContactMatch(mode="body", pattern=target, entity=target),
    fields=("found", "force"),
    reduce="netforce",
    history_length=decimation,
  )


def collision_targets(scene: SceneConfig, task: ReleaseTaskConfig) -> tuple[str, ...]:
  """What the robot is penalized for touching: the support surface and the relative objects."""
  support = scene.support_surface
  surface = (support.object,) if support is not None and support.object is not None else ()
  return (*surface, *task.relative_placement.object_names)


def contact_targets(scene: SceneConfig, task: ReleaseTaskConfig) -> tuple[str, ...]:
  """Every object with a robot contact sensor (the collision targets)."""
  return collision_targets(scene, task)


def _support_pair_group(scene: SceneConfig, task: ReleaseTaskConfig) -> ContactPairGroup:
  names = task.relative_placement.object_names
  partners = tuple(g for o in scene.objects if o.name in names for g in scene_pair_geoms(o))
  return ContactPairGroup(SUPPORT_PAIRS, partners, task.domain_randomization.friction.support_object)


def _grasp_params(ent: grasp_cfg._Entities, task: ReleaseTaskConfig) -> dict:
  """Grasp's ``_Entities.grasp_params`` for a release task (same keys and values)."""
  return {
    "sensor_names": FINGER_SENSORS,
    "finger_cfg": ent.fingers(),
    "finger_outward_axes": ent.emb.finger_outward_axes,
    "min_force": task.grasp_min_force,
    "max_angle_deg": task.grasp_max_angle_deg,
  }


def _observations(task: ReleaseTaskConfig, ent: grasp_cfg._Entities) -> dict[str, ObservationGroupCfg]:
  g_obs, r_obs = grasp_mdp.observations, mdp.observations
  supports = task.relative_placement.object_names
  symmetry = task.relative_placement.goal_symmetry

  def terms(critic: bool) -> dict[str, ObservationTermCfg]:
    joints = lambda: {"arm_cfg": ent.arm(), "gripper_cfg": ent.gripper()}  # noqa: E731
    t = {
      "qpos": ObservationTermCfg(func=g_obs.joint_pos, params=joints()),
      "qvel": ObservationTermCfg(func=g_obs.joint_vel, params=joints()),
      "ee_pose": ObservationTermCfg(func=g_obs.ee_pose, params={"tcp_cfg": ent.tcp()}),
      "gripper_width": ObservationTermCfg(
        func=g_obs.gripper_width,
        params={
          "gripper_cfg": ent.gripper(),
          "open_pos": ent.emb.gripper_open,
          "closed_pos": ent.emb.gripper_closed,
        },
      ),
    }
    if critic:
      t |= {
        "object_pose": ObservationTermCfg(func=g_obs.object_pose, params={"object_cfg": ent.object()}),
        "tcp_to_obj_pose": ObservationTermCfg(
          func=r_obs.tcp_to_obj_pose, params={"tcp_cfg": ent.tcp(), "object_cfg": ent.object()}
        ),
        "obj_to_goal_pose": ObservationTermCfg(
          func=r_obs.obj_to_goal_pose,
          params={"command_name": COMMAND_NAME, "object_cfg": ent.object()},
        ),
        "support_poses": ObservationTermCfg(
          func=r_obs.support_poses, params={"support_names": supports}
        ),
      }
      if task.include_is_grasped_obs:
        t["is_grasped"] = ObservationTermCfg(func=g_obs.is_grasped, params=_grasp_params(ent, task))
    else:
      perceived = {"perception_name": PERCEPTION_NAME}
      t |= {
        "object_pose": ObservationTermCfg(func=g_obs.perceived_object_pose, params=perceived),
        "tcp_to_obj_pose": ObservationTermCfg(
          func=r_obs.perceived_tcp_to_obj_pose, params={"tcp_cfg": ent.tcp(), **perceived}
        ),
        "obj_to_goal_pose": ObservationTermCfg(
          func=r_obs.perceived_obj_to_goal_pose,
          params={**perceived, "goal_perception_name": GOAL_PERCEPTION_NAME, "symmetry": symmetry},
        ),
      }
    return t

  return {
    "actor": ObservationGroupCfg(terms(critic=False), enable_corruption=False),
    "critic": ObservationGroupCfg(terms(critic=True), enable_corruption=False),
  }


def _released_params(task: ReleaseTaskConfig, ent: grasp_cfg._Entities, scaled: bool = True) -> dict:
  """Params of ``released``. ``scaled`` adds the stillness-at-release factor when the task
  enables it (rewards); metrics pass ``scaled=False`` to count releases as they are."""
  extra = (
    {"stillness_action_name": GRIPPER_ACTION_NAME} if scaled and task.release_scale_by_stillness else {}
  )
  return {
    **extra,
    "command_name": COMMAND_NAME,
    "object_cfg": ent.object(),
    "pos_tol": task.release_pos_tol,
    "rot_tol_deg": task.release_rot_tol_deg,
    "sensor_names": FINGER_SENSORS,
    "gripper_cfg": ent.gripper(),
    "open_pos": ent.emb.gripper_open,
    "closed_pos": ent.emb.gripper_closed,
    "max_closure": task.release_max_closure,
  }


def _rewards(task: ReleaseTaskConfig, ent: grasp_cfg._Entities, collision: tuple[str, ...]) -> dict:
  r, g = mdp.rewards, grasp_mdp.rewards
  n = task.reward_normalization
  supports = task.relative_placement.object_names
  rewards = {
    "place": RewardTermCfg(
      func=r.place,
      weight=task.w_place / n,
      params={
        "command_name": COMMAND_NAME,
        "object_cfg": ent.object(),
        "pos_scale": task.place_pos_scale,
        "rot_scale": task.place_rot_scale,
      },
    ),
    "released": RewardTermCfg(func=r.released, weight=task.w_released / n, params=_released_params(task, ent)),
    "retreat": RewardTermCfg(
      func=r.retreat,
      weight=task.w_retreat / n,
      params={"tcp_cfg": ent.tcp(), "retreat_height": task.retreat_height, **_released_params(task, ent)},
    ),
    "robot_collision": RewardTermCfg(
      func=r.robot_collision,
      weight=task.w_robot_collision / n,
      params={"sensor_names": tuple(contact_sensor_name(c) for c in collision)},
    ),
    "support_disturb": RewardTermCfg(
      func=r.support_disturb,
      weight=task.w_support_disturb / n,
      params={
        "command_name": COMMAND_NAME,
        "support_names": supports,
        "pos_scale": task.disturb_pos_scale,
        "rot_scale": task.disturb_rot_scale,
      },
    ),
    "terminated": RewardTermCfg(func=is_terminated, weight=task.w_terminated / n),
    "arm_speed": RewardTermCfg(
      func=g.joint_speed_excess,
      weight=task.w_arm_speed / n,
      params={"asset_cfg": ent.arm(), "speed_threshold": task.arm_speed_threshold},
    ),
    "gripper_speed": RewardTermCfg(
      func=g.joint_speed_excess,
      weight=task.w_gripper_speed / n,
      params={"asset_cfg": ent.gripper(), "speed_threshold": task.gripper_speed_threshold},
    ),
    "joint_pos_limits": RewardTermCfg(
      func=joint_pos_limits, weight=task.w_joint_pos_limits / n, params={"asset_cfg": ent.arm()}
    ),
  }
  if task.w_release_moving != 0:
    rewards["release_moving"] = RewardTermCfg(
      func=r.release_moving,
      weight=task.w_release_moving / n,
      params={"action_name": GRIPPER_ACTION_NAME, "object_cfg": ent.object()},
    )
  if task.w_gripper_switch != 0:
    rewards["gripper_switch"] = RewardTermCfg(
      func=g.gripper_switch,
      weight=task.w_gripper_switch / n,
      params={"action_name": GRIPPER_ACTION_NAME},
    )
  return rewards


def _terminations(task: ReleaseTaskConfig, ent: grasp_cfg._Entities) -> dict[str, TerminationTermCfg]:
  t = mdp.terminations
  return {
    **grasp_cfg._terminations(),
    "object_dropped": TerminationTermCfg(
      func=t.object_dropped,
      params={
        "command_name": COMMAND_NAME,
        "object_cfg": ent.object(),
        "drop_below_goal_m": task.drop_below_goal_m,
        **_grasp_params(ent, task),
      },
    ),
    "support_toppled": TerminationTermCfg(
      func=t.support_toppled,
      params={
        "command_name": COMMAND_NAME,
        "support_names": task.relative_placement.object_names,
        "tilt_deg": task.topple_tilt_deg,
        "displacement_m": task.topple_displacement_m,
      },
    ),
  }


def _metrics(task: ReleaseTaskConfig, ent: grasp_cfg._Entities, collision: tuple[str, ...]) -> dict:
  m = mdp.metrics
  goal = lambda: {"command_name": COMMAND_NAME, "object_cfg": ent.object()}  # noqa: E731
  tol = {"pos_tol": task.release_pos_tol, "rot_tol_deg": task.release_rot_tol_deg}
  terms = _terminations(task, ent)
  metrics = {
    # Curriculum default: released steps / max episode length.
    "released_frac": MetricsTermCfg(
      func=m.released_fraction, params=_released_params(task, ent, scaled=False), reduce="sum"
    ),
    "placed_frac": MetricsTermCfg(func=m.placed_fraction, params={**goal(), **tol}, reduce="sum"),
    "final_pos_err": MetricsTermCfg(func=m.position_error, params=goal(), reduce="last"),
    "final_rot_err_deg": MetricsTermCfg(func=m.rotation_error_deg, params=goal(), reduce="last"),
    "grasped_frac": MetricsTermCfg(
      func=grasp_mdp.metrics.grasped_fraction, params=_grasp_params(ent, task), reduce="sum"
    ),
    "dropped": MetricsTermCfg(
      func=m.object_dropped_metric, params=terms["object_dropped"].params, reduce="max"
    ),
    "toppled": MetricsTermCfg(
      func=m.support_toppled_metric, params=terms["support_toppled"].params, reduce="max"
    ),
    # 1 for an episode in which a robot part touched the table or a relative object ...
    "collided": MetricsTermCfg(
      func=m.collided,
      params={"sensor_names": tuple(contact_sensor_name(c) for c in collision)},
      reduce="max",
    ),
    # ... and the relative objects alone.
    "touched_relative": MetricsTermCfg(
      func=m.collided,
      params={
        "sensor_names": tuple(contact_sensor_name(c) for c in task.relative_placement.object_names)
      },
      reduce="max",
    ),
  }
  if task.release_scale_by_stillness:
    metrics["release_stillness"] = MetricsTermCfg(
      func=m.release_stillness, params={"action_name": GRIPPER_ACTION_NAME}, reduce="last"
    )
  if task.w_release_moving != 0:
    # Fraction of episodes with an opening while the object was clearly moving.
    metrics["moving_release_frac"] = MetricsTermCfg(
      func=m.moving_release,
      params={"action_name": GRIPPER_ACTION_NAME, "object_cfg": ent.object()},
      reduce="max",
    )
  if task.w_gripper_switch != 0:
    metrics["gripper_switches"] = MetricsTermCfg(
      func=grasp_mdp.rewards.gripper_switch, params={"action_name": GRIPPER_ACTION_NAME}, reduce="sum"
    )
  return metrics


def make_release_env_cfg_from_configs(
  scene: SceneConfig,
  task: ReleaseTaskConfig,
  action_space: str | None = None,
  num_envs: int = 1,
  check_dataset: bool = True,
) -> ManagerBasedRlEnvCfg:
  task.validate_scene(scene)
  dataset = task.start.dataset
  if dataset is None:
    raise ValueError("task.start.dataset is not set; generate one with scripts/generate_release_starts.py")
  if check_dataset:
    starts.check_meta(task, starts.read_meta(dataset), scene.scene_name)
  emb = get_embodiment(scene.robot.embodiment)
  action_space = action_space or scene.robot.action_space
  timing = task.timing
  obj = scene.manipulated
  supports = task.relative_placement.object_names
  dr = task.domain_randomization
  sizes = dr.object_size_variants
  if sizes.enabled and obj.primitive != "mesh":
    raise ValueError("domain_randomization.object_size_variants needs a mesh manipulated object")

  events: dict[str, EventTermCfg] = {
    "reset_scene_to_default": EventTermCfg(func=reset_scene_to_default, mode="reset"),
    "restore_start": EventTermCfg(
      func=mdp.RestoreReleaseStart,
      mode="reset",
      params={"dataset": str(dataset), "object_name": obj.name, "support_names": supports},
    ),
  }
  if dr.stiffness_damping_enabled:
    targets = emb.stiffness_damping
    events["randomize_stiffness_damping"] = EventTermCfg(
      func=grasp_mdp.STIFFNESS_DAMPING_EVENTS[targets.damping_source],
      mode="reset",
      params={
        "entity_name": "robot",
        "stiffness_actuators": targets.stiffness_actuators,
        "damping_targets": (
          targets.damping_joints if targets.damping_source == "joint" else targets.damping_actuators
        ),
        "stiffness_scale_range": dr.stiffness_scale_range,
        "damping_scale_range": dr.damping_scale_range,
      },
    )
  if dr.joint_friction.enabled:
    events["randomize_joint_friction"] = EventTermCfg(
      func=grasp_mdp.RandomizeJointFriction,
      mode="reset",
      params={
        "entity_name": "robot",
        "joint_names": emb.arm_joint_names,
        "scale_range": dr.joint_friction.scale_range,
      },
    )
  friction = dr.friction
  pair_groups = grasp_cfg._friction_pair_groups(scene, emb, friction)
  if pair_groups:
    gripper_pairs, table_pairs = pair_groups
    support_pairs = _support_pair_group(scene, task)
    pair_groups = (*pair_groups, support_pairs)
    ranges = {
      "sliding_scale_range": friction.sliding_scale_range,
      "torsional_scale_range": friction.torsional_scale_range,
      "rolling_scale_range": friction.rolling_scale_range,
    }
    events["randomize_pair_friction"] = EventTermCfg(
      func=grasp_mdp.RandomizePairFriction,
      mode="reset",
      params={
        "gripper_cfg": SceneEntityCfg(obj.name, pair_names=(gripper_pairs.pair_name_pattern,)),
        "table_cfg": SceneEntityCfg(obj.name, pair_names=(table_pairs.pair_name_pattern,)),
        "gripper_friction": friction.gripper_object,
        "table_friction": friction.table_object,
        **ranges,
      },
    )
    events["randomize_support_pair_friction"] = EventTermCfg(
      func=mdp.RandomizeSupportPairFriction,
      mode="reset",
      params={
        "support_cfg": SceneEntityCfg(obj.name, pair_names=(support_pairs.pair_name_pattern,)),
        **ranges,
      },
    )

  ent = grasp_cfg._Entities(emb, obj.name)
  collision = collision_targets(scene, task)
  metrics = _metrics(task, ent, collision)

  sensors: tuple[SensorCfg, ...] = grasp_cfg._finger_sensors(emb, obj.name)
  sensors += tuple(
    robot_contact_sensor(emb, c, timing.decimation) for c in contact_targets(scene, task)
  )
  delay_substeps = task.object_pose_delay_substeps
  if delay_substeps > 0:
    sensors += (
      grasp_mdp.ObjectPoseHistoryCfg(
        name=POSE_HISTORY_SENSOR, object_name=obj.name, delay_substeps=delay_substeps
      ),
    )

  # Order matters: the perceived goal reads this reset's clean goal.
  commands: dict[str, CommandTermCfg] = {
    COMMAND_NAME: mdp.ReleaseGoalCommandCfg(
      support_names=supports, symmetry=task.relative_placement.goal_symmetry, debug_vis=True
    ),
    GOAL_PERCEPTION_NAME: mdp.PerceivedGoalCfg(
      goal_command_name=COMMAND_NAME, noise=grasp_cfg._pose_noise(task.goal_pose_noise)
    ),
    PERCEPTION_NAME: grasp_mdp.PerceivedObjectPoseCfg(
      object_name=obj.name,
      target_lift_height=0.0,  # Its lift goal is unused; release reads PerceivedGoal.
      persistent_bias=grasp_cfg._pose_noise(dr.object_pose_noise.persistent_bias),
      per_step=grasp_cfg._pose_noise(dr.object_pose_noise.per_step),
      history_sensor_name=POSE_HISTORY_SENSOR if delay_substeps > 0 else None,
    ),
  }
  if dr.action_hold_probability > 0:
    commands[ACTION_HOLD_NAME] = grasp_mdp.ActionHoldCfg(probability=dr.action_hold_probability)

  actions = grasp_cfg._actions(emb, action_space, task)  # type: ignore[arg-type]
  stillness = (
    {
      "stillness_object": obj.name,
      "stillness_speed_scale": task.release_stillness_speed_scale,
      "stillness_spin_scale": task.release_stillness_spin_scale,
    }
    if task.release_scale_by_stillness
    else {}
  )
  actions[GRIPPER_ACTION_NAME] = mdp.HeldGripperActionCfg.from_grasp(actions[GRIPPER_ACTION_NAME], **stillness)  # type: ignore[arg-type]

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      num_envs=num_envs,
      entities=scene_entity_cfgs(scene, build_entity_cfg(emb, name_finger_geoms=friction.enabled), sizes, pair_groups),
      sensors=sensors,
    ),
    observations=_observations(task, ent),
    actions=actions,
    commands=commands,
    events=events,
    rewards=_rewards(task, ent, collision),
    terminations=_terminations(task, ent),
    curriculum={**grasp_cfg._curriculum(task, metrics), **grasp_cfg._termination_counts()},  # type: ignore[arg-type]
    metrics=metrics,
    viewer=grasp_cfg._viewer(scene.camera),
    sim=SimulationCfg(
      nconmax=grasp_cfg.NCONMAX_PER_OBJECT * len(scene.objects) + 32,
      njmax=grasp_cfg.NJMAX_PER_OBJECT * len(scene.objects) + 128,
      mujoco=MujocoCfg(timestep=timing.sim_dt, integrator=emb.integrator, impratio=emb.impratio, cone=emb.cone),
    ),
    decimation=timing.decimation,
    episode_length_s=timing.episode_steps * timing.step_dt - 1e-6,
    scale_rewards_by_dt=False,
  )


def make_release_env_cfg(
  scene_path: str | Path = DEFAULT_SCENE,
  task_path: str | Path = DEFAULT_TASK,
  action_space: str | None = None,
  num_envs: int = 1,
) -> ManagerBasedRlEnvCfg:
  return make_release_env_cfg_from_configs(
    load_scene_config(scene_path), load_release_task_config(task_path), action_space, num_envs
  )
