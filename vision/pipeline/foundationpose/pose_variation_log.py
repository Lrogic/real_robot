"""Summarize FoundationPose pose scatter on a static object.

Poses are the tracked oriented-bounds frame in the camera optical frame
(+X right, +Y down, +Z forward). Position is that origin in metres.
Orientation is roll, pitch, yaw in degrees: intrinsic XYZ Euler angles of the
object-to-camera rotation, the same decomposition as
scipy.spatial.transform.Rotation.as_euler("xyz").
"""

from pathlib import Path

import numpy as np

_PERCENTILES = (5, 25, 50, 75, 95)
_LOG_INTERVAL_S = 1.0


class PoseVariationLog:
    """One 4x4 camera-frame pose per object per processed frame."""

    def __init__(self):
        self.samples = []  # list of (t_s, {name: (4, 4) ndarray})

    def record(self, t_s: float, tracker) -> None:
        poses = {}
        for tracked in tracker.objects:
            if tracked.center_pose is None:
                continue
            poses[tracked.name] = np.array(tracked.center_pose, dtype=np.float64, copy=True)
        if poses:
            self.samples.append((float(t_s), poses))

    def write(self, path: Path) -> None:
        path.write_text(format_log(self.samples))
        print(f"Saved pose variation for {len(self.samples)} frames to {path}")


def format_log(samples) -> str:
    lines = [
        "summary stats:",
        "Pose is the tracked oriented-bounds frame in the camera optical frame "
        "(+X right, +Y down, +Z forward).",
        "Position is that origin in metres.",
        "Orientation is roll, pitch, yaw in degrees: intrinsic XYZ Euler of the "
        "object-to-camera rotation (scipy Rotation.as_euler('xyz')).",
        "Standard deviation is the population std (ddof=0) over every processed frame.",
        "Angular deviation is the angle between each frame's rotation and the mean rotation.",
        f"frames: {len(samples)}",
        "",
    ]
    names = []
    for _, poses in samples:
        for name in poses:
            if name not in names:
                names.append(name)
    if not names:
        lines.append("(no poses recorded)")
        lines.append("")
        lines.append("every 1s logging:")
        lines.append("(no poses recorded)")
        lines.append("")
        return "\n".join(lines)

    for name in names:
        series = [(t_s, poses[name]) for t_s, poses in samples if name in poses]
        lines.extend(_summary_lines(name, series))
        lines.append("")

    lines.append("every 1s logging:")
    lines.append("t_s is seconds since tracking started. One row about every 1s.")
    lines.append("")
    for name in names:
        series = [(t_s, poses[name]) for t_s, poses in samples if name in poses]
        lines.extend(_timeseries_lines(name, series))
        lines.append("")
    return "\n".join(lines)


def _summary_lines(name, series):
    poses = np.stack([pose for _, pose in series])
    position = poses[:, :3, 3]
    rpy = np.degrees(_unwrap_rpy(poses[:, :3, :3]))
    angular = _angular_deviation_deg(poses[:, :3, :3])
    lines = [name]
    lines.append(f"  frames: {len(series)}")
    lines.append("  mean position (m): " + _vec(position.mean(axis=0)))
    lines.append("  mean orientation (deg roll pitch yaw): " + _vec(rpy.mean(axis=0)))
    lines.append("  std position (m): " + _vec(position.std(axis=0)))
    lines.append("  std orientation (deg roll pitch yaw): " + _vec(rpy.std(axis=0)))
    lines.append("  percentiles position (m):")
    lines.extend(_percentile_block(("x", "y", "z"), position))
    lines.append("  percentiles orientation (deg):")
    lines.extend(_percentile_block(("roll", "pitch", "yaw"), rpy))
    lines.append("  angular deviation from mean rotation (deg):")
    lines.append(f"    mean: {angular.mean():.4f}")
    lines.append(f"    std: {angular.std():.4f}")
    pct = np.percentile(angular, _PERCENTILES)
    lines.append("    " + "  ".join(f"p{p}: {v:.4f}" for p, v in zip(_PERCENTILES, pct)))
    return lines


def _timeseries_lines(name, series):
    lines = [name, "  t_s        x_m        y_m        z_m      roll_deg   pitch_deg  yaw_deg"]
    next_t = series[0][0]
    for t_s, pose in series:
        if t_s + 1e-9 < next_t:
            continue
        rpy = np.degrees(_rotation_to_rpy(pose[:3, :3]))
        xyz = pose[:3, 3]
        lines.append(
            f"  {t_s:8.3f}  {xyz[0]:10.6f}  {xyz[1]:10.6f}  {xyz[2]:10.6f}"
            f"  {rpy[0]:10.4f}  {rpy[1]:10.4f}  {rpy[2]:10.4f}"
        )
        next_t = t_s + _LOG_INTERVAL_S
    return lines


def _percentile_block(labels, values):
    lines = []
    for index, label in enumerate(labels):
        pct = np.percentile(values[:, index], _PERCENTILES)
        parts = "  ".join(f"p{p}: {v:.6f}" for p, v in zip(_PERCENTILES, pct))
        lines.append(f"    {label}: {parts}")
    return lines


def _vec(values) -> str:
    return " ".join(f"{v:.6f}" for v in values)


def _rotation_to_rpy(rotation: np.ndarray) -> np.ndarray:
    """Intrinsic XYZ Euler (roll, pitch, yaw) in radians."""
    pitch = np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0))
    if abs(rotation[2, 0]) < 0.999999:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        # Gimbal lock: yaw is set to 0 and roll absorbs the remaining rotation.
        roll = np.arctan2(-rotation[0, 1], rotation[1, 1])
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float64)


def _unwrap_rpy(rotations: np.ndarray) -> np.ndarray:
    rpy = np.stack([_rotation_to_rpy(rotation) for rotation in rotations])
    return np.unwrap(rpy, axis=0)


def _angular_deviation_deg(rotations: np.ndarray) -> np.ndarray:
    mean = _mean_rotation(rotations)
    cosine = np.clip((np.trace(mean.T @ rotations, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _mean_rotation(rotations: np.ndarray) -> np.ndarray:
    quaternions = np.stack([_rotation_to_quat(rotation) for rotation in rotations])
    signs = np.where(quaternions @ quaternions[0] < 0.0, -1.0, 1.0)
    mean = (quaternions * signs[:, None]).mean(axis=0)
    mean /= np.linalg.norm(mean)
    return _quat_to_rotation(mean)


def _rotation_to_quat(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        return np.array([
            0.25 * scale,
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
        ])
    if rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        return np.array([
            (rotation[2, 1] - rotation[1, 2]) / scale,
            0.25 * scale,
            (rotation[0, 1] + rotation[1, 0]) / scale,
            (rotation[0, 2] + rotation[2, 0]) / scale,
        ])
    if rotation[1, 1] > rotation[2, 2]:
        scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        return np.array([
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[0, 1] + rotation[1, 0]) / scale,
            0.25 * scale,
            (rotation[1, 2] + rotation[2, 1]) / scale,
        ])
    scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
    return np.array([
        (rotation[1, 0] - rotation[0, 1]) / scale,
        (rotation[0, 2] + rotation[2, 0]) / scale,
        (rotation[1, 2] + rotation[2, 1]) / scale,
        0.25 * scale,
    ])


def _quat_to_rotation(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
