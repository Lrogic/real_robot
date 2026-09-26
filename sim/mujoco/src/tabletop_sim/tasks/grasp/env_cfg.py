"""Build the mjlab environment config for the grasp-lift-hold task from a scene
JSON and a task JSON."""

from __future__ import annotations

import math
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import time_out
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig
from tabletop_sim import CONFIGS_DIR
from tabletop_sim.embodiments import EmbodimentCfg, build_entity_cfg, get_embodiment
from tabletop_sim.scene import (
  ACTION_SPACES,
  CameraConfig,
  SceneConfig,
  TaskConfig,
  load_scene_config,
  load_task_config,
  scene_entity_cfgs,
)
from tabletop_sim.tasks.grasp import mdp

COMMAND_NAME = "grasp"
FINGER_SENSORS = ("left_finger_object_contact", "right_finger_object_contact")
DEFAULT_SCENE = CONFIGS_DIR / "scenes" / "tabletop_cube_wxai.json"
DEFAULT_TASK = CONFIGS_DIR / "tasks" / "grasp_standard_wxai.json"
# Per-world contact / constraint budget. Elliptic cones use several constraint
# rows per contact, and a gripped box can touch the fingers and table at once.
NCONMAX_PER_OBJECT = 32
NJMAX_PER_OBJECT = 192


def _actions(emb: EmbodimentCfg, action_space: str, task: TaskConfig) -> dict[str, ActionTermCfg]:
  gripper_mode = "binary" if action_space == "ee_delta_pose_binary_gripper" else "absolute"
  gripper = mdp.GripperActionCfg(
    entity_name="robot",
    joint_name=emb.gripper_actuator_joint,
    open_pos=emb.gripper_open,
    closed_pos=emb.gripper_closed,
    mode=gripper_mode,
  )
  if action_space == "ee_delta_pose_binary_gripper":
    arm: ActionTermCfg = mdp.ClippedDifferentialIKActionCfg(
      entity_name="robot",
      actuator_names=emb.arm_joint_names,
      frame_type="site",
      frame_name=emb.tcp_site,
      use_relative_mode=True,
      delta_pos_scale=task.action.ee_pos_scale,
      delta_ori_scale=task.action.ee_rot_scale,
      damping=emb.ik.damping,
      max_dq=emb.ik.max_dq,
      position_weight=emb.ik.position_weight,
      orientation_weight=emb.ik.orientation_weight,
      joint_limit_weight=emb.ik.joint_limit_weight,
      posture_weight=emb.ik.posture_weight,
    )
  elif action_space == "delta_joint_abs_gripper":
    arm = mdp.DeltaJointPositionActionCfg(
      entity_name="robot",
      joint_names=emb.arm_joint_names,
      scale=task.action.joint_delta_scale,
    )
  else:
    raise ValueError(f"action_space must be one of {ACTION_SPACES}, got {action_space!r}")
  return {"arm": arm, "gripper": gripper}


def _observations(task: TaskConfig) -> dict[str, ObservationGroupCfg]:
  obs = mdp.observations
  p = {"command_name": COMMAND_NAME}
  terms = {
    "qpos": ObservationTermCfg(func=obs.joint_pos, params=p),
    "qvel": ObservationTermCfg(func=obs.joint_vel, params=p),
    "ee_pose": ObservationTermCfg(func=obs.ee_pose, params=p),
    "gripper_width": ObservationTermCfg(func=obs.gripper_width, params=p),
  }
  if task.use_minimal_obs:
    terms["tcp_to_goal_pos"] = ObservationTermCfg(func=obs.tcp_to_goal_pos, params=p)
    terms["obj_initial_pose"] = ObservationTermCfg(func=obs.obj_initial_pose, params=p)
  else:
    terms["object_pose"] = ObservationTermCfg(func=obs.object_pose, params=p)
    terms["tcp_to_obj_pos"] = ObservationTermCfg(func=obs.tcp_to_obj_pos, params=p)
    terms["obj_to_goal_pos"] = ObservationTermCfg(func=obs.obj_to_goal_pos, params=p)
    if task.include_is_grasped_obs:
      terms["is_grasped"] = ObservationTermCfg(func=obs.is_grasped, params=p)
  # Noise (and asymmetric critic terms) go here once domain randomization lands.
  return {
    "actor": ObservationGroupCfg(dict(terms), enable_corruption=False),
    "critic": ObservationGroupCfg(dict(terms), enable_corruption=False),
  }


def _rewards(task: TaskConfig) -> dict[str, RewardTermCfg]:
  r = mdp.rewards
  n = task.reward_normalization
  p = {"command_name": COMMAND_NAME}
  terms = {
    "reach": RewardTermCfg(
      func=r.reach,
      weight=task.w_reach / n,
      params={**p, "scale": task.reach_scale, "saturation_dist": task.reach_saturation_dist},
    ),
    "grasp": RewardTermCfg(func=r.grasp, weight=task.w_grasp / n, params=p),
    "lift_target": RewardTermCfg(
      func=r.lift_target,
      weight=task.w_lift_target / n,
      params={**p, "scale": task.lift_target_scale},
    ),
    "stable_hold": RewardTermCfg(func=r.stable_hold, weight=task.w_hold / n, params=p),
    "success": RewardTermCfg(func=r.success, weight=task.w_success / n, params=p),
  }
  if task.w_xy != 0.0:
    terms["xy_displacement"] = RewardTermCfg(
      func=r.xy_displacement_penalty,
      weight=-task.w_xy / n,
      params={**p, "tolerance": task.xy_tolerance},
    )
  if task.w_ori != 0.0:
    terms["orientation"] = RewardTermCfg(
      func=r.orientation_penalty,
      weight=-task.w_ori / n,
      params={**p, "tolerance": task.orientation_tolerance},
    )
  return terms


def _finger_sensors(emb: EmbodimentCfg, object_name: str) -> tuple[ContactSensorCfg, ...]:
  return tuple(
    ContactSensorCfg(
      name=sensor_name,
      primary=ContactMatch(mode="body", pattern=body, entity="robot"),
      secondary=ContactMatch(mode="body", pattern=object_name, entity=object_name),
      fields=("found", "force"),
      reduce="netforce",
    )
    for sensor_name, body in zip(FINGER_SENSORS, emb.finger_bodies, strict=True)
  )


def _viewer(camera: CameraConfig) -> ViewerConfig:
  eye, target = camera.render_eye, camera.render_target
  fwd = [t - e for e, t in zip(eye, target, strict=True)]
  dist = math.sqrt(sum(f * f for f in fwd))
  return ViewerConfig(
    origin_type=ViewerConfig.OriginType.WORLD,
    lookat=tuple(target),
    distance=dist,
    azimuth=math.degrees(math.atan2(fwd[1], fwd[0])),
    elevation=math.degrees(math.asin(fwd[2] / dist)),
    max_extra_envs=0,
  )


def make_grasp_env_cfg_from_configs(
  scene: SceneConfig,
  task: TaskConfig,
  action_space: str | None = None,
  num_envs: int = 1,
) -> ManagerBasedRlEnvCfg:
  emb = get_embodiment(scene.robot.embodiment)
  action_space = action_space or scene.robot.action_space
  timing = task.timing
  obj = scene.manipulated

  events: dict[str, EventTermCfg] = {
    "reset_scene_to_default": EventTermCfg(func=reset_scene_to_default, mode="reset"),
  }
  for o in scene.objects:
    if len(o.initial_poses) > 1:
      events[f"reset_{o.name}_pose"] = EventTermCfg(
        func=mdp.reset_object_from_pose_list,
        mode="reset",
        params={"entity_name": o.name, "poses": o.initial_poses},
      )

  tracker = mdp.GraspTrackerCfg(
    object_name=obj.name,
    tcp_site=emb.tcp_site,
    arm_joint_names=emb.arm_joint_names,
    gripper_joint_names=emb.gripper_joint_names,
    finger_sensor_names=FINGER_SENSORS,
    finger_bodies=emb.finger_bodies,
    finger_outward_axes=emb.finger_outward_axes,
    gripper_open=emb.gripper_open,
    gripper_closed=emb.gripper_closed,
    hold_steps=math.ceil(task.stable_hold_duration_sec * timing.control_hz - 1e-9),
    lift_threshold=task.lift_threshold,
    target_lift_height=task.target_lift_height,
    stable_linear_vel_tol=task.stable_linear_vel_tol,
    stable_angular_vel_tol=task.stable_angular_vel_tol,
    arm_joint_vel_tol=task.arm_joint_vel_tol,
    gripper_joint_vel_tol=task.gripper_joint_vel_tol,
    grasp_min_force=task.grasp_min_force,
    grasp_max_angle_deg=task.grasp_max_angle_deg,
    debug_vis=True,
  )

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      num_envs=num_envs,
      entities=scene_entity_cfgs(scene, build_entity_cfg(emb)),
      sensors=_finger_sensors(emb, obj.name),
    ),
    observations=_observations(task),
    actions=_actions(emb, action_space, task),
    commands={COMMAND_NAME: tracker},
    events=events,
    rewards=_rewards(task),
    terminations={
      "time_out": TerminationTermCfg(func=time_out, time_out=True),
      "success": TerminationTermCfg(
        func=mdp.terminations.task_success, params={"command_name": COMMAND_NAME}
      ),
    },
    viewer=_viewer(scene.camera),
    sim=SimulationCfg(
      nconmax=NCONMAX_PER_OBJECT * len(scene.objects) + 32,
      njmax=NJMAX_PER_OBJECT * len(scene.objects) + 128,
      mujoco=MujocoCfg(
        timestep=timing.sim_dt,
        integrator=emb.integrator,
        impratio=emb.impratio,
        cone=emb.cone,
      ),
    ),
    decimation=timing.decimation,
    # Slightly under N steps so float rounding cannot add an extra step.
    episode_length_s=timing.episode_steps * timing.step_dt - 1e-6,
    scale_rewards_by_dt=False,
  )


def make_grasp_env_cfg(
  scene_path: str | Path = DEFAULT_SCENE,
  task_path: str | Path = DEFAULT_TASK,
  action_space: str | None = None,
  num_envs: int = 1,
) -> ManagerBasedRlEnvCfg:
  return make_grasp_env_cfg_from_configs(
    load_scene_config(scene_path), load_task_config(task_path), action_space, num_envs
  )
