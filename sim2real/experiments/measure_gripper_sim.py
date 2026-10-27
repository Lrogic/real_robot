"""Measure the simulated inside-tip gap at the training gripper settings.

Loads the embodiment and robot XML used by ``scripts/train.py`` (the scene's
embodiment; the default scene is the Menagerie WidowX). The gripper command is
the position target written to the single gripper position actuator, which is
the same target ``GripperAction`` applies. The other carriage follows through
the model's equality. Arm actuators hold the training ready pose.

Fully open is the upper limit of the gripper position actuator, the same role
as the driver ``position_max`` sent to ``set_gripper_position``. Fully closed
stays at the embodiment's closed position. Half open is their midpoint. The
metre values are this model's carriage coordinate, not the hardware driver's.

The reported gap is the distance between the inside surfaces of the two
fingertips. Those points are taken from the visual finger meshes: vertices in
the distal 1 mm whose position along the finger's outward axis is within
0.05 mm of the inward-most tip vertex. The ``gripper_*_pad`` boxes are the
model's flat contact pads on that same surface; their inner-face locations are
saved beside the mesh points. The gap is that distance, not an assumed sum of
the carriage coordinates.

Launch from the wxai_mjlab environment, from ``real_robot/sim/mujoco`` or any
directory (the package is installed in that environment):

  conda activate wxai_mjlab
  python real_robot/sim2real/experiments/measure_gripper_sim.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
from tabletop_sim.embodiments import get_embodiment
from tabletop_sim.embodiments.base import EmbodimentCfg, load_spec
from tabletop_sim.scene import load_scene_config, load_task_config
from tabletop_sim.tasks.grasp.env_cfg import DEFAULT_SCENE, DEFAULT_TASK

EXPERIMENT = "gripper_inside_tip_gap"
SETTINGS = ("fully_open", "half_open", "fully_closed")
TIP_BAND_M = 0.001
FACE_BAND_M = 5e-5
PAD_AGREEMENT_M = 5e-4
VELOCITY_TOL_M_S = 1e-4
DWELL_S = 0.25
TIMEOUT_S = 5.0
RESULTS_DIR = Path(__file__).resolve().parent / "results"
INTEGRATORS = {
    "euler": mujoco.mjtIntegrator.mjINT_EULER,
    "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
}
CONES = {
    "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
    "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC,
}


def _carriage_limits(model: mujoco.MjModel, emb: EmbodimentCfg, actuator_id: int) -> dict:
    """Joint range, training soft limit, and position-actuator command limit.

    The joint range is MuJoCo ``jnt_range``. The soft limit is mjlab's inset of
    that range, ``mean ± 0.5 * span * soft_joint_pos_limit_factor``. The
    controller limit is the position actuator ``ctrlrange``, which clips the
    commanded carriage position. Only the actuated carriage has one.
    """
    factor = emb.soft_joint_pos_limit_factor
    limits = {}
    for name in emb.gripper_joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lower, upper = (float(v) for v in model.jnt_range[joint_id])
        center = 0.5 * (lower + upper)
        half_span = 0.5 * (upper - lower) * factor
        entry = {
            "joint_range_m": [lower, upper],
            "soft_joint_limit_m": [center - half_span, center + half_span],
            "controller_joint_limit_m": None,
        }
        if name == emb.gripper_actuator_joint:
            entry["controller_joint_limit_m"] = [float(v) for v in model.actuator_ctrlrange[actuator_id]]
        limits[name] = entry
    return {
        "unit": "m",
        "soft_joint_pos_limit_factor": factor,
        "joint_range": "MuJoCo jnt_range, the joint's position limits in the model.",
        "soft_joint_limit": (
            "mjlab soft limit: the center of jnt_range plus or minus half the "
            "span times soft_joint_pos_limit_factor. Used by the joint-limit "
            "penalty. The gripper is not penalized."
        ),
        "controller_joint_limit": (
            "Position-actuator ctrlrange, which clips the gripper command. "
            "The carriage without an actuator has none; the equality makes it "
            "follow the actuated carriage."
        ),
        "carriages": limits,
    }


def _commands(model: mujoco.MjModel, emb: EmbodimentCfg, actuator_id: int) -> dict[str, float]:
    """Fully open is the position-actuator upper limit, matching a driver position_max command."""
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, emb.gripper_actuator_joint)
    open_ = min(float(model.jnt_range[joint_id, 1]), float(model.actuator_ctrlrange[actuator_id, 1]))
    closed = float(emb.gripper_closed)
    if open_ <= closed:
        raise ValueError(f"{emb.name}: gripper open {open_} m is not above closed {closed} m")
    return {
        "fully_closed": closed,
        "half_open": 0.5 * (closed + open_),
        "fully_open": open_,
    }


def _apply_training_solver(model: mujoco.MjModel, emb: EmbodimentCfg, timestep: float) -> None:
    """``load_spec`` clears the solver; training puts the embodiment values back."""
    model.opt.timestep = timestep
    model.opt.integrator = INTEGRATORS[emb.integrator]
    model.opt.impratio = emb.impratio
    model.opt.cone = CONES[emb.cone]


def _joint_addresses(model: mujoco.MjModel, name: str) -> tuple[int, int]:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise ValueError(f"joint {name!r} not in the model")
    return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])


def _visual_vertices(model: mujoco.MjModel, body_name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"body {body_name!r} not in the model")
    clouds = []
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] != body_id:
            continue
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]:
            continue
        mesh_id = model.geom_dataid[geom_id]
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        vertices = model.mesh_vert[start : start + count]
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[geom_id])
        clouds.append(vertices @ rotation.reshape(3, 3).T + model.geom_pos[geom_id])
    if not clouds:
        raise ValueError(f"{body_name} has no visual mesh")
    return np.concatenate(clouds)


def _inside_tip_local(vertices: np.ndarray, outward: np.ndarray) -> np.ndarray:
    """Body-frame point on the inward face of the distal fingertip."""
    distal = vertices[vertices[:, 0] >= vertices[:, 0].max() - TIP_BAND_M]
    if len(distal) == 0:
        raise ValueError("fingertip band is empty")
    opening = distal @ outward
    inward = float(opening.min())
    face = distal[opening <= inward + FACE_BAND_M]
    return face.mean(axis=0)


def _pad_inner_tip(model: mujoco.MjModel, geom_name: str, outward: np.ndarray) -> np.ndarray:
    """Distal midpoint of the pad box face that looks into the jaw."""
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if geom_id < 0:
        raise ValueError(f"pad geom {geom_name!r} not in the model")
    half = model.geom_size[geom_id, :3]
    corners = np.array([[sx, sy, sz] for sx in (-half[0], half[0]) for sy in (-half[1], half[1]) for sz in (-half[2], half[2])])
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[geom_id])
    corners = corners @ rotation.reshape(3, 3).T + model.geom_pos[geom_id]
    opening = corners @ outward
    inner = corners[np.isclose(opening, opening.min())]
    distal = float(inner[:, 0].max())
    edge = inner[np.isclose(inner[:, 0], distal)]
    return edge.mean(axis=0)


def _measurement_points(model: mujoco.MjModel, emb: EmbodimentCfg) -> dict:
    points = {}
    for side, body, outward in zip(
        ("left", "right"), emb.finger_bodies, emb.finger_outward_axes, strict=True
    ):
        axis = np.asarray(outward, dtype=float)
        mesh_point = _inside_tip_local(_visual_vertices(model, body), axis)
        pad_point = _pad_inner_tip(model, f"gripper_{side}_pad", axis)
        along_mesh = float(mesh_point @ axis)
        along_pad = float(pad_point @ axis)
        if abs(along_mesh - along_pad) > PAD_AGREEMENT_M:
            raise RuntimeError(
                f"{body}: visual inside tip and gripper_{side}_pad disagree by "
                f"{(along_mesh - along_pad) * 1000:.2f} mm along the opening axis"
            )
        points[side] = {
            "body": body,
            "body_id": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body),
            "outward_axis_body": axis.tolist(),
            "mesh_point_m": mesh_point,
            "pad_inner_distal_point_m": pad_point,
        }
    return points


def _world_point(data: mujoco.MjData, body_id: int, local: np.ndarray) -> np.ndarray:
    return data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ local


def _set_ready(model: mujoco.MjModel, data: mujoco.MjData, emb: EmbodimentCfg) -> None:
    for name, position in emb.ready_joint_pos.items():
        address, _ = _joint_addresses(model, name)
        data.qpos[address] = position
    for actuator_id in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        if name in emb.ready_joint_pos and name != emb.gripper_actuator_joint:
            data.ctrl[actuator_id] = emb.ready_joint_pos[name]
    mujoco.mj_forward(model, data)


def _gripper_actuator(model: mujoco.MjModel, emb: EmbodimentCfg) -> int:
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, emb.gripper_actuator_joint)
    if actuator_id < 0:
        raise ValueError(f"gripper actuator {emb.gripper_actuator_joint!r} not in the model")
    return int(actuator_id)


def _settle(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    addresses: list[tuple[int, int]],
    timeout_s: float,
) -> bool:
    """True once both carriages stay slower than the velocity tolerance for the dwell.

    The position actuator does not have to land on the command. Friction in the
    carriage joint leaves a fraction of a millimetre, and that actual position
    is what the sample records.
    """
    stable_since = None
    deadline = data.time + timeout_s
    while data.time < deadline:
        mujoco.mj_step(model, data)
        velocities = np.array([data.qvel[v] for _, v in addresses])
        if np.all(np.abs(velocities) <= VELOCITY_TOL_M_S):
            if stable_since is None:
                stable_since = data.time
            if data.time - stable_since >= DWELL_S:
                return True
        else:
            stable_since = None
    return False


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def run(args: argparse.Namespace) -> None:
    scene = load_scene_config(args.scene)
    task = load_task_config(args.task)
    emb = get_embodiment(scene.robot.embodiment)
    model = load_spec(emb).compile()
    _apply_training_solver(model, emb, task.timing.sim_dt)
    data = mujoco.MjData(model)
    points = _measurement_points(model, emb)
    addresses = [_joint_addresses(model, name) for name in emb.gripper_joint_names]
    actuator_id = _gripper_actuator(model, emb)
    commands = _commands(model, emb, actuator_id)
    carriage_limits = _carriage_limits(model, emb, actuator_id)
    _set_ready(model, data, emb)

    output = args.output or (
        RESULTS_DIR / f"gripper_gap_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    payload = {
        "experiment": EXPERIMENT,
        "source": "simulation",
        "coordinate_units": {
            "commanded_gripper_position": "m",
            "actual_gripper_position": "m",
            "carriage_position": "m",
            "inside_tip_gap": "mm",
        },
        "configuration": {
            "scene": str(args.scene),
            "task": str(args.task),
            "embodiment": emb.name,
            "xml_path": str(emb.xml_path),
            "gripper_joint_names": list(emb.gripper_joint_names),
            "gripper_actuator_joint": emb.gripper_actuator_joint,
            "gripper_closed_m": emb.gripper_closed,
            "carriage_limits": carriage_limits,
            "commands_m": commands,
            "command_definition": (
                "fully_open is the minimum of the actuated carriage joint upper "
                "limit and the gripper position-actuator ctrlrange upper limit. "
                "fully_closed is the embodiment closed position. half_open is "
                "their midpoint. These metres are the simulated carriage coordinate, "
                "not the hardware driver coordinate."
            ),
            "sim_dt_s": task.timing.sim_dt,
            "integrator": emb.integrator,
            "impratio": emb.impratio,
            "cone": emb.cone,
            "settle_timeout_s": args.timeout_s,
            "inside_tip_measurement": {
                "gap": (
                    "Euclidean distance between the left and right inside-tip points "
                    "after the carriage bodies move. This is the separation a caliper "
                    "would read between those surfaces."
                ),
                "mesh_selection": (
                    "On each finger visual mesh, vertices within 1 mm of the maximum "
                    "body-frame x (the tip). Of those, the ones within 0.05 mm of the "
                    "minimum projection onto the embodiment finger_outward_axis. Their "
                    "mean is the inside-tip point."
                ),
                "pad_check": (
                    "gripper_left_pad and gripper_right_pad are the flat inner contact "
                    "boxes. The distal midpoint of each box face that faces into the "
                    "jaw is compared with the mesh point along finger_outward_axis. "
                    "The saved gap uses the mesh points."
                ),
                "fingers": {
                    side: {
                        "body": item["body"],
                        "outward_axis_body": item["outward_axis_body"],
                        "mesh_point_m": item["mesh_point_m"].tolist(),
                        "pad_inner_distal_point_m": item["pad_inner_distal_point_m"].tolist(),
                    }
                    for side, item in points.items()
                },
            },
        },
        "samples": [],
    }

    print(
        f"Embodiment {emb.name}\n"
        f"Model {emb.xml_path}\n"
        f"Actuated joint {emb.gripper_actuator_joint}; "
        f"carriages {emb.gripper_joint_names[0]}, {emb.gripper_joint_names[1]}\n"
        f"Commands (m): open {commands['fully_open']:.6f}, "
        f"half {commands['half_open']:.6f}, closed {commands['fully_closed']:.6f}"
    )
    for name, limits in carriage_limits["carriages"].items():
        joint_range = limits["joint_range_m"]
        soft = limits["soft_joint_limit_m"]
        controller = limits["controller_joint_limit_m"]
        controller_text = "none" if controller is None else f"[{controller[0]:.6f}, {controller[1]:.6f}]"
        print(
            f"{name}: joint range [{joint_range[0]:.6f}, {joint_range[1]:.6f}] m, "
            f"soft limit [{soft[0]:.6f}, {soft[1]:.6f}] m, "
            f"controller limit {controller_text} m"
        )
    for name in SETTINGS:
        command = commands[name]
        data.ctrl[actuator_id] = command
        settled = _settle(model, data, addresses, args.timeout_s)
        left = float(data.qpos[addresses[0][0]])
        right = float(data.qpos[addresses[1][0]])
        left_world = _world_point(data, points["left"]["body_id"], points["left"]["mesh_point_m"])
        right_world = _world_point(data, points["right"]["body_id"], points["right"]["mesh_point_m"])
        gap_mm = float(np.linalg.norm(left_world - right_world) * 1000.0)
        sample = {
            "setting": name,
            "commanded_gripper_position": command,
            "actual_gripper_position": None,
            "actual_left_carriage_position": left,
            "actual_right_carriage_position": right,
            "inside_tip_gap_mm": gap_mm,
            "inside_tip_gap_minus_carriage_sum_mm": gap_mm - (left + right) * 1000.0,
            "left_inside_tip_world_m": left_world.tolist(),
            "right_inside_tip_world_m": right_world.tolist(),
            "inside_tip_gap_source": "simulated_visual_mesh_inside_fingertips",
            "settled": settled,
            "timestamp": datetime.now().astimezone().isoformat(),
        }
        payload["samples"].append(sample)
        _write(output, payload)
        print(
            f"{name}: command {command:.6f} m, "
            f"left {left:.6f} m, right {right:.6f} m, "
            f"inside-tip gap {gap_mm:.3f} mm, settled {settled}"
        )
        if not settled:
            print(f"  did not settle within {args.timeout_s:.1f} s")

    residuals = [sample["inside_tip_gap_minus_carriage_sum_mm"] for sample in payload["samples"]]
    spread = max(residuals) - min(residuals)
    print(
        "Inside-tip gap minus the sum of the carriage positions (mm): "
        + ", ".join(f"{value:.3f}" for value in residuals)
    )
    if spread > 0.2:
        print(
            "Those residuals are not constant, so the inside-tip gap is not one "
            "affine function of the two carriage positions across these settings."
        )
    else:
        print(
            "The residual is constant to 0.2 mm: the fingertip surfaces sit at a "
            "fixed offset from the carriage coordinates. This is not a hardware mapping."
        )
    print(f"Saved {output}")
    print(
        "No hardware-to-simulation mapping was fit. Compare these gaps with the "
        "hardware file at matching measured gaps."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--task", type=Path, default=DEFAULT_TASK)
    parser.add_argument("--timeout-s", type=float, default=TIMEOUT_S)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.timeout_s <= 0:
        raise SystemExit("--timeout-s must be positive")
    run(args)


if __name__ == "__main__":
    main()
