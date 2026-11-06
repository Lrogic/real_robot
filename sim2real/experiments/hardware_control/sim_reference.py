"""Simulated reference trials with the training robot, physics and arm action.

The env is built by tabletop_sim's ``make_grasp_env_cfg_from_configs`` from the deployed
policy's scene and task, so the robot model, actuators, gravity compensation, timestep
(2 ms), decimation (25) and ``ClippedDifferentialIKAction`` settings are the training ones.
Randomization is off: stiffness/damping, friction and object-size randomization, object
pose noise and the curriculum are disabled, and only the reset-to-default event is kept.

Each trial reproduces the inner loop of ``ManagerBasedRlEnv.step`` without rewards,
terminations or auto-reset: per physics substep the arm action's ``apply_actions`` (one DLS
step) runs, then ``write_data_to_sim``, ``sim.step``, ``scene.update``; ``sim.forward`` runs
after every 25 substeps. At each 20 Hz outer update the test's absolute TCP target is
written into the action's desired pose (``ik_adapter.set_held_target``) and held until the
next update. The gripper carriage is held at the ready value (0.026 m) by a joint target.

Before onset: reset to the ready joints with zero velocity, hold them with joint-position
targets for ``sim.settle_s``, and check the settle.

Launch from the wxai_mjlab environment:

  python sim_reference.py results/plans/single_tra_z+.json
  python sim_reference.py results/plans/*.json --device cpu
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from settings import load_settings, snapshot
from geometry import make_T, pose_json
from ik_adapter import action_config, held_target, set_held_target, world_from_base
from plans import load_plan
from records import RESULT_KIND, UNITS, environment_record, result_path, save_result
from targets import TargetSource


def task_without_randomization(task):
    dr = task.domain_randomization
    noise = dr.object_pose_noise
    dr = replace(
        dr,
        stiffness_damping_enabled=False,
        friction=replace(dr.friction, enabled=False),
        object_size_variants=replace(dr.object_size_variants, enabled=False),
        object_pose_noise=replace(noise, persistent_bias=replace(noise.persistent_bias, enabled=False),
                                  per_step=replace(noise.per_step, enabled=False)),
    )
    return replace(task, domain_randomization=dr, curriculum=replace(task.curriculum, enabled=False))


def build_training_env(cfg: dict, device: str):
    from mjlab.envs import ManagerBasedRlEnv
    from tabletop_sim.scene import load_scene_config, load_task_config
    from tabletop_sim.tasks.grasp.env_cfg import make_grasp_env_cfg_from_configs
    from tabletop_sim.tasks.grasp.mdp.actions import ClippedDifferentialIKAction

    scene = load_scene_config(cfg["training"]["scene"])
    task = task_without_randomization(load_task_config(cfg["training"]["task"]))
    env_cfg = make_grasp_env_cfg_from_configs(scene, task, num_envs=1)
    env_cfg.events = {k: v for k, v in env_cfg.events.items() if k == "reset_scene_to_default"}
    env = ManagerBasedRlEnv(env_cfg, device=device)
    env.reset(seed=0)
    term = env.action_manager.get_term("arm")
    if type(term) is not ClippedDifferentialIKAction:
        raise RuntimeError(f"arm action is {type(term).__name__}, expected ClippedDifferentialIKAction "
                           f"(scene action_space {scene.robot.action_space!r})")
    return env


def training_controller_record(env) -> dict:
    """What the sim actually runs, read back from the built env."""
    import mujoco

    m = env.sim.mj_model
    term = env.action_manager.get_term("arm")
    act = []
    for i in range(m.nu):
        act.append({
            "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i),
            "kp": float(m.actuator_gainprm[i, 0]), "bias": m.actuator_biasprm[i, :3].tolist(),
            "forcerange": m.actuator_forcerange[i].tolist(), "ctrlrange": m.actuator_ctrlrange[i].tolist(),
        })
    def named(obj, count):
        names = [mujoco.mj_id2name(m, obj, i) or "" for i in range(count)]
        return [(i, n) for i, n in enumerate(names) if n.startswith("robot/")]

    bodies, joints = named(mujoco.mjtObj.mjOBJ_BODY, m.nbody), named(mujoco.mjtObj.mjOBJ_JOINT, m.njnt)
    dofs = [int(m.jnt_dofadr[i]) for i, _ in joints]
    return {
        "physics_dt_s": float(m.opt.timestep),
        "decimation": int(env.cfg.decimation),
        "outer_dt_s": float(env.step_dt),
        "ik_calls_per_outer_step": int(env.cfg.decimation),
        "integrator": int(m.opt.integrator), "impratio": float(m.opt.impratio), "cone": int(m.opt.cone),
        "gravity": m.opt.gravity.tolist(),
        "robot_body_gravcomp": {n: float(m.body_gravcomp[i]) for i, n in bodies},
        "joint_actgravcomp": {n: bool(m.jnt_actgravcomp[i]) for i, n in joints},
        "joint_armature": {n: float(m.dof_armature[d]) for (_, n), d in zip(joints, dofs)},
        "joint_frictionloss": {n: float(m.dof_frictionloss[d]) for (_, n), d in zip(joints, dofs)},
        "joint_damping": {n: float(m.dof_damping[d]) for (_, n), d in zip(joints, dofs)},
        "actuators": act,
        "arm_action": action_config(term),
        "randomization": "disabled (stiffness/damping, friction, size variants, pose noise, curriculum); "
                         "only reset_scene_to_default is kept",
        "ik_timing": "apply_actions every physics substep (DLS step from the current q); target "
                     "written only at outer updates; sim.forward after each 25 substeps",
    }


class SimReference:
    def __init__(self, env, fk):
        self.env = env
        self.fk = fk
        self.robot = env.scene["robot"]
        self.term = env.action_manager.get_term("arm")
        self.arm_ids = self.term._joint_ids
        ids, _ = self.robot.find_joints((fk.emb.gripper_actuator_joint,), preserve_order=True)
        self.gripper_ids = torch.tensor(ids, device=env.device, dtype=torch.long)
        self.dt = float(env.physics_dt)
        self.decimation = int(env.cfg.decimation)
        self.substep_hook = None

    def tcp_base(self) -> np.ndarray:
        """Training TCP in the base frame from the sim's current derived site pose."""
        d = self.env.sim.data
        p = d.site_xpos[0, self.term._frame_id].detach().cpu().numpy().astype(np.float64)
        R = d.site_xmat[0, self.term._frame_id].detach().cpu().numpy().astype(np.float64).reshape(3, 3)
        T_wb = world_from_base(self.env)
        return np.linalg.inv(T_wb) @ make_T(R, p)

    def _physics_substep(self, arm_target=None) -> None:
        if arm_target is None:
            self.term.apply_actions()
        else:
            self.robot.set_joint_position_target(arm_target, joint_ids=self.arm_ids)
        self.robot.set_joint_position_target(self._gripper_target, joint_ids=self.gripper_ids)
        self.env.scene.write_data_to_sim()
        self.env.sim.step()
        self.env.scene.update(dt=self.dt)

    def reset_and_settle(self, plan: dict, sim_cfg: dict) -> dict:
        ref = plan["reference"]
        ready = np.asarray(ref["arm_joints_rad"])
        if not np.allclose(ready, self.fk.ready_arm, atol=1e-9):
            raise ValueError("plan ready joints differ from the training embodiment's ready pose")
        self.env.reset(seed=0)
        self._gripper_target = torch.full((1, 1), float(ref["gripper_m"]), device=self.env.device)
        ready_t = torch.as_tensor(ready, dtype=torch.float32, device=self.env.device).unsqueeze(0)
        n = int(round(sim_cfg["settle_s"] / self.dt))
        for i in range(n):
            self._physics_substep(arm_target=ready_t)
            if (i + 1) % self.decimation == 0:
                self.env.sim.forward()
        self.env.sim.forward()
        q = self.robot.data.joint_pos[0, self.arm_ids].cpu().numpy().astype(np.float64)
        qd = self.robot.data.joint_vel[0, self.arm_ids].cpu().numpy().astype(np.float64)
        settle = {"settle_s": n * self.dt, "max_joint_error_rad": float(np.max(np.abs(q - ready))),
                  "max_joint_speed_rad_s": float(np.max(np.abs(qd)))}
        settle["settled"] = (settle["max_joint_error_rad"] <= sim_cfg["settle_joint_tolerance_rad"]
                             and settle["max_joint_speed_rad_s"] <= sim_cfg["settle_velocity_tolerance_rad_s"])
        return settle

    def _object_poses(self) -> dict:
        return {name: self.env.scene[name].data.root_link_pose_w[0, :3].cpu().numpy().copy()
                for name in self.env.scene.entities if name not in ("robot", "ground")
                and not self.env.scene[name].is_fixed_base}

    def run(self, plan: dict, cfg: dict) -> dict:
        settle = self.reset_and_settle(plan, cfg["sim"])
        if not settle["settled"]:
            raise RuntimeError(f"sim did not settle at the ready pose: {settle}")
        T_wb = world_from_base(self.env)
        source = TargetSource(plan)
        T_ref = make_T(np.asarray(plan["reference"]["tcp"]["rotation_matrix"]), plan["reference"]["tcp"]["position_m"])
        objects_before = self._object_poses()

        q0 = self.robot.data.joint_pos[0, self.arm_ids].clone()
        qs, qds, qts, gs = [q0], [self.robot.data.joint_vel[0, self.arm_ids].clone()], [], []
        gs.append(self.robot.data.joint_pos[0, self.gripper_ids].clone())
        outer = []
        substep, stop_reason = 0, None
        wall0 = time.perf_counter()
        while True:
            if substep % self.decimation == 0:
                k = substep // self.decimation
                elapsed = substep * self.dt
                stop_reason = source.done(elapsed)
                if stop_reason:
                    break
                T_cur = self.tcp_base()
                T_new, info = source.update(k, elapsed, T_cur)
                if T_new is not None:
                    set_held_target(self.term, T_wb @ T_new)
                outer.append({"index": k, "scheduled_t_s": elapsed, "actual_t_s": elapsed, "skipped_before": 0,
                              "new_target": T_new is not None, "target": pose_json(T_new) if T_new is not None else None,
                              "current_tcp": pose_json(T_cur), "info": info})
            self._physics_substep()
            if self.substep_hook is not None:
                self.substep_hook(substep, held_target(self.term))
            substep += 1
            qs.append(self.robot.data.joint_pos[0, self.arm_ids].clone())
            qds.append(self.robot.data.joint_vel[0, self.arm_ids].clone())
            qts.append(self.robot.data.joint_pos_target[0, self.arm_ids].clone())
            gs.append(self.robot.data.joint_pos[0, self.gripper_ids].clone())
            if substep % self.decimation == 0:
                self.env.sim.forward()
        wall = time.perf_counter() - wall0

        q = torch.stack(qs).cpu().numpy().astype(np.float64)
        qd = torch.stack(qds).cpu().numpy().astype(np.float64)
        q_target = torch.stack(qts).cpu().numpy().astype(np.float64)
        t = np.arange(len(q)) * self.dt
        tcp = [self.fk.tcp(qi) for qi in q]
        moved = {name: float(np.linalg.norm(p - objects_before[name])) for name, p in self._object_poses().items()}
        T_init = tcp[0]
        return {
            "kind": RESULT_KIND, "source": "sim", "backend": "sim_training_ik",
            "condition": {"tag": "sim", "ik_hz": 1.0 / self.dt, "outer_hz": 1.0 / (self.dt * self.decimation)},
            "plan": plan, "plan_sha256": plan["sha256"], "plan_name": plan["name"], "test": plan["test"],
            "settings": snapshot(cfg), "units": UNITS, "environment": environment_record(),
            "training_controller": training_controller_record(self.env),
            "tcp_model": self.fk.describe(),
            "settle": settle,
            "initial_state": {"arm_q_rad": q[0], "arm_qd_rad_s": qd[0], "tcp": pose_json(T_init),
                              "tcp_offset_from_reference_m": (T_init[:3, 3] - T_ref[:3, 3]).tolist()},
            "onset": {"definition": "simulated time 0 = first outer update; its target is set before "
                                    "the first physics substep"},
            "samples": {
                "t_s": t, "arm_q_rad": q, "arm_qd_rad_s": qd,
                "gripper_q_m": torch.stack(gs).cpu().numpy().reshape(-1),
                "tcp_position_m": np.array([T[:3, 3] for T in tcp]),
                "tcp_quat_wxyz": np.array([pose_json(T)["quaternion_wxyz"] for T in tcp]),
                "note": "sample i is the state after i physics substeps; tcp is training-model FK of arm_q",
            },
            "joint_targets": {"t_s": t[1:] - self.dt, "arm_q_target_rad": q_target,
                              "note": "set by apply_actions at the start of each substep (time = substep start)"},
            "outer_updates": outer,
            "timing_summary": {"physics_substeps": substep, "outer_updates": len(outer),
                               "simulated_s": substep * self.dt, "wall_s": wall},
            "object_displacement_m": moved,
            "errors": [], "stop_reason": stop_reason,
        }


def main() -> None:
    from fk import TrainingFK

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plans", type=Path, nargs="+")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", default=None, help="default: sim.device in config.yaml")
    args = parser.parse_args()
    cfg = load_settings(args.config)
    plans = [load_plan(p) for p in args.plans]
    fk = TrainingFK(cfg["training"]["scene"])
    env = build_training_env(cfg, args.device or cfg["sim"]["device"])
    sim = SimReference(env, fk)
    for plan in plans:
        result = sim.run(plan, cfg)
        path = save_result(result, result_path("sim", plan["name"], "sim"))
        moved = max(result["object_displacement_m"].values(), default=0.0)
        final = np.asarray(result["samples"]["tcp_position_m"][-1]) - np.asarray(result["samples"]["tcp_position_m"][0])
        print(f"{plan['name']}: {result['stop_reason']}, {result['timing_summary']['simulated_s']:.2f} s, "
              f"TCP moved {np.round(final * 1000, 2).tolist()} mm"
              + (f", WARNING object moved {moved * 1000:.2f} mm" if moved > 1e-4 else "") + f"\n  {path}")


if __name__ == "__main__":
    main()
