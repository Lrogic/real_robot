"""Compare the training TCP site with hardware tool poses by forward kinematics.

Reads a file from ``capture_tcp_hardware.py``. For each sample the arm joints
are set to the measured positions and ``mj_kinematics`` updates the site pose.
The state is not integrated, so servo lag and simulated sag stay out of the
residual.

Both poses are expressed in the robot base: the driver base frame on the
hardware side, and the body that carries the first arm joint on the simulation
side. The position residual is the simulated TCP minus the hardware tool, in
that base frame. The orientation residual is the angle of the rotation that
takes the hardware tool frame onto the simulated site frame.

The scene file selects the embodiment. The comparison loads that robot the way
training does (``load_spec``) and does not apply the scene's base pose; the
reported poses are already relative to the robot base.

Launch from the wxai_mjlab environment:

  conda activate wxai_mjlab
  python real_robot/sim2real/experiments/TCP_alignment/compare_tcp_fk.py HARDWARE.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
from tabletop_sim.tasks.grasp.env_cfg import DEFAULT_SCENE

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))

from common.io import write_json  # noqa: E402
from common.poses import pose_record, rotation_angle_deg, rotvec_to_matrix  # noqa: E402
from common.sim import joint_addresses, load_training_robot  # noqa: E402

EXPERIMENT = "tcp_alignment"
WRIST_SPAN_NOTE_DEG = 90.0
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _body_name(model: mujoco.MjModel, body_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    return "world" if name is None else name


def _base_body_id(model: mujoco.MjModel, first_arm_joint: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, first_arm_joint)
    if joint_id < 0:
        raise ValueError(f"joint {first_arm_joint!r} not in the model")
    child = int(model.jnt_bodyid[joint_id])
    return int(model.body_parentid[child])


def _joint_moves_body(model: mujoco.MjModel, joint_name: str, body_id: int) -> bool:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"joint {joint_name!r} not in the model")
    child = int(model.jnt_bodyid[joint_id])
    body = body_id
    while body != 0:
        if body == child:
            return True
        body = int(model.body_parentid[body])
    return False


def _site_pose_in_body(
    data: mujoco.MjData, site_id: int, body_id: int
) -> tuple[np.ndarray, np.ndarray]:
    base_rotation = np.array(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    base_position = np.array(data.xpos[body_id], dtype=np.float64)
    site_rotation = np.array(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
    site_position = np.array(data.site_xpos[site_id], dtype=np.float64)
    position = base_rotation.T @ (site_position - base_position)
    rotation = base_rotation.T @ site_rotation
    return position, rotation


def _apply_arm(
    model: mujoco.MjModel, data: mujoco.MjData, joint_names: tuple[str, ...], positions: list[float]
) -> tuple[dict[str, float], list[dict]]:
    if len(positions) != len(joint_names):
        raise ValueError(
            f"hardware has {len(positions)} arm joints and the model has {len(joint_names)}: "
            + ", ".join(joint_names)
        )
    data.qvel[:] = 0.0
    written = {}
    outside = []
    for name, position in zip(joint_names, positions):
        address, _ = joint_addresses(model, name)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lower, upper = (float(v) for v in model.jnt_range[joint_id])
        if position < lower or position > upper:
            outside.append(
                {
                    "joint": name,
                    "position_rad": float(position),
                    "model_range_rad": [lower, upper],
                }
            )
        data.qpos[address] = position
        written[name] = float(position)
    mujoco.mj_kinematics(model, data)
    return written, outside


def _load_hardware(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("experiment") != EXPERIMENT or payload.get("source") != "hardware":
        raise SystemExit(
            f"{path} is not a {EXPERIMENT} hardware capture "
            f"(experiment={payload.get('experiment')!r}, source={payload.get('source')!r})"
        )
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise SystemExit(f"{path} has no samples")
    return payload


def _wrist_span_deg(samples: list[dict]) -> float | None:
    if len(samples) < 2:
        return None
    wrist = [float(sample["arm_joint_positions_rad"][-1]) for sample in samples]
    return float(np.degrees(max(wrist) - min(wrist)))


def _summary(samples: list[dict]) -> dict:
    magnitudes = [sample["position_residual_magnitude_mm"] for sample in samples]
    angles = [sample["orientation_residual_deg"] for sample in samples]
    vectors = np.array([sample["position_residual_mm"] for sample in samples])
    return {
        "num_samples": len(samples),
        "position_residual_magnitude_mm": {
            "min": float(min(magnitudes)),
            "mean": float(np.mean(magnitudes)),
            "max": float(max(magnitudes)),
        },
        "position_residual_mean_mm": vectors.mean(axis=0).tolist(),
        "orientation_residual_deg": {
            "min": float(min(angles)),
            "mean": float(np.mean(angles)),
            "max": float(max(angles)),
        },
    }


def run(args: argparse.Namespace) -> None:
    hardware = _load_hardware(args.hardware)
    model, emb = load_training_robot(args.scene)
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, emb.tcp_site)
    if site_id < 0:
        raise SystemExit(f"TCP site {emb.tcp_site!r} is not in the model")
    site_id = int(site_id)
    base_id = _base_body_id(model, emb.arm_joint_names[0])
    site_body_id = int(model.site_bodyid[site_id])
    gripper_that_moves_site = [
        name for name in emb.gripper_joint_names if _joint_moves_body(model, name, site_body_id)
    ]
    if gripper_that_moves_site:
        raise SystemExit(
            f"TCP site {emb.tcp_site!r} moves with {gripper_that_moves_site}. "
            "The hardware file has one gripper position and this model has two "
            "carriage joints, so this comparison does not apply that reading."
        )
    mujoco.mj_kinematics(model, data)
    carriage_positions = {}
    for name in emb.gripper_joint_names:
        address, _ = joint_addresses(model, name)
        carriage_positions[name] = float(data.qpos[address])

    output = args.output or (
        RESULTS_DIR / f"tcp_fk_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    payload = {
        "experiment": EXPERIMENT,
        "source": "simulation_forward_kinematics",
        "hardware_file": str(args.hardware),
        "coordinate_units": {
            "position": "m",
            "position_residual": "mm",
            "orientation": "rotvec_rad",
            "orientation_residual": "deg",
        },
        "configuration": {
            "scene": str(args.scene),
            "embodiment": emb.name,
            "xml_path": str(emb.xml_path),
            "mujoco_version": mujoco.__version__,
            "tcp_site": emb.tcp_site,
            "tcp_site_body": _body_name(model, site_body_id),
            "tcp_site_local_pos_m": [float(v) for v in model.site_pos[site_id]],
            "base_body": _body_name(model, base_id),
            "arm_joint_names": list(emb.arm_joint_names),
            "kinematics": (
                "Measured arm positions are written into qpos and mj_kinematics "
                "updates the site. mj_step is not called."
            ),
            "frames": (
                "Hardware Cartesian positions are the driver end-effector frame in "
                "the driver base frame. The simulated pose is the training TCP site "
                "in the body that carries the first arm joint. The scene base pose "
                "is not applied."
            ),
            "position_residual": (
                "Simulated TCP position minus the hardware tool position, in the "
                "base frame, in millimetres."
            ),
            "orientation_residual": (
                "Angle, in degrees, of the rotation that takes the hardware tool "
                "frame onto the simulated TCP frame."
            ),
            "gripper": (
                "Carriage joints stay at the compiled default. The TCP site does "
                "not move with them. Each sample still keeps the hardware gripper reading."
            ),
            "carriage_positions_m": carriage_positions,
            "hardware_configuration": hardware.get("configuration"),
        },
        "samples": [],
    }

    print(
        f"Embodiment {emb.name}\n"
        f"Model {emb.xml_path}\n"
        f"TCP site {emb.tcp_site} on {_body_name(model, site_body_id)}, "
        f"local position {payload['configuration']['tcp_site_local_pos_m']} m\n"
        f"Base body {_body_name(model, base_id)}\n"
        f"Hardware {args.hardware} ({len(hardware['samples'])} samples)"
    )
    for reading in hardware["samples"]:
        cartesian = reading["cartesian_positions_m_rotvec_rad"]
        if len(cartesian) != 6:
            raise SystemExit(
                f"sample {reading.get('sample')}: Cartesian pose has {len(cartesian)} values, expected 6"
            )
        hardware_position = np.asarray(cartesian[:3], dtype=np.float64)
        hardware_rotation = rotvec_to_matrix(cartesian[3:])
        written, outside = _apply_arm(
            model, data, emb.arm_joint_names, reading["arm_joint_positions_rad"]
        )
        sim_position, sim_rotation = _site_pose_in_body(data, site_id, base_id)
        residual_mm = (sim_position - hardware_position) * 1000.0
        magnitude_mm = float(np.linalg.norm(residual_mm))
        angle_deg = rotation_angle_deg(hardware_rotation.T @ sim_rotation)
        sample = {
            "sample": reading.get("sample"),
            "hardware": reading,
            "simulated_arm_joint_positions_rad": written,
            "arm_joints_outside_model_range": outside,
            "hardware_tool_pose_base": pose_record(hardware_position, hardware_rotation),
            "simulated_tcp_pose_base": pose_record(sim_position, sim_rotation),
            "position_residual_mm": residual_mm.tolist(),
            "position_residual_magnitude_mm": magnitude_mm,
            "orientation_residual_deg": angle_deg,
        }
        payload["samples"].append(sample)
        write_json(output, payload)
        vector = ", ".join(f"{value:.2f}" for value in residual_mm)
        print(
            f"sample {sample['sample']}: position residual {magnitude_mm:.2f} mm "
            f"[{vector}], orientation {angle_deg:.3f} deg"
        )
        for item in outside:
            lower, upper = item["model_range_rad"]
            print(
                f"  {item['joint']} at {item['position_rad']:.4f} rad is outside "
                f"the model range [{lower:.4f}, {upper:.4f}] rad; kinematics used the measured value"
            )

    payload["summary"] = _summary(payload["samples"])
    span = _wrist_span_deg(hardware["samples"])
    if span is not None:
        payload["summary"]["wrist_joint_span_deg"] = span
    write_json(output, payload)
    summary = payload["summary"]
    position = summary["position_residual_magnitude_mm"]
    orientation = summary["orientation_residual_deg"]
    mean_vector = ", ".join(f"{value:.2f}" for value in summary["position_residual_mean_mm"])
    print(
        f"\nPosition residual magnitude (mm): "
        f"min {position['min']:.2f}, mean {position['mean']:.2f}, max {position['max']:.2f}"
    )
    print(f"Mean position residual (mm, base frame): [{mean_vector}]")
    print(
        f"Orientation residual (deg): "
        f"min {orientation['min']:.3f}, mean {orientation['mean']:.3f}, max {orientation['max']:.3f}"
    )
    if span is None:
        print("One wrist orientation cannot show a TCP offset that depends on wrist angle.")
    else:
        print(f"Wrist joint (last arm joint) spanned {span:.1f} deg across these samples.")
        if span < WRIST_SPAN_NOTE_DEG:
            print(
                f"That span is under {WRIST_SPAN_NOTE_DEG:.0f} deg. "
                "A TCP offset error can stay small until the wrist turns further."
            )
    print(f"Saved {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hardware", type=Path, help="JSON from capture_tcp_hardware.py")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not args.hardware.is_file():
        raise SystemExit(f"hardware file not found: {args.hardware}")
    if not args.scene.is_file():
        raise SystemExit(f"scene file not found: {args.scene}")
    run(args)


if __name__ == "__main__":
    main()
