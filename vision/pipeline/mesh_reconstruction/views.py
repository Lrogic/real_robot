"""Camera poses for the mesh views, planned around the object's masked depth points (base frame).

The front view looks down at the centroid from the side the setup camera was on, at the distance
where the points' projected bounding box fills the configured fraction of the frame; the side
views rotate it about the vertical through the centroid. Cameras are OpenCV (x right, y down,
z forward) with no roll: x stays horizontal.
"""

import numpy as np

from calibration.transforms import make_T, transform_points


def masked_points(depth, K, mask, T_base_camera) -> np.ndarray:
    """(N, 3) base-frame points of the mask's pixels that have depth, without outliers.

    Edge and background pixels inside the mask carry depth well behind the object (up to 1 m on
    mesh_test3); from a side view they spread along the setup camera's rays and blow up the
    projected box, so points over 3x the median distance from the median are dropped.
    """
    v, u = np.nonzero(mask & (depth > 0))
    z = depth[v, u]
    pts = transform_points(T_base_camera, np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1))
    r = np.linalg.norm(pts - np.median(pts, 0), axis=1)
    return pts[r <= 3 * np.median(r)]


def look_at(eye, target) -> np.ndarray:
    z = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross([0.0, 0.0, -1.0], z)  # horizontal: no roll
    x /= np.linalg.norm(x)
    return make_T(np.stack([x, np.cross(z, x), z], 1), eye)


def project_box(points, T_base_camera, K, shape, margin_px=10):
    """Pixel bounding box (x0, y0, x1, y1) of the points seen from T_base_camera, clipped to the frame."""
    cam = transform_points(np.linalg.inv(T_base_camera), points)
    uv = cam[:, :2] / cam[:, 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    (x0, y0), (x1, y1) = uv.min(0) - margin_px, uv.max(0) + margin_px
    return (max(x0, 0), max(y0, 0), min(x1, shape[1] - 1), min(y1, shape[0] - 1))


def box_fraction(box, shape) -> float:
    """Larger side of a bounding box as a fraction of that frame dimension."""
    x0, y0, x1, y1 = box
    return max((x1 - x0) / shape[1], (y1 - y0) / shape[0])


def mask_box(mask):
    ys, xs = np.nonzero(mask)
    return (xs.min(), ys.min(), xs.max(), ys.max())


def plan(points, T_setup, K, shape, cfg, distance=None):
    """(pose(angle_deg, distance) -> T_base_camera, front distance) around the points' centroid.

    Angle 0 is the front view, on the setup camera's side; the side views are +/- side_angle_deg.
    distance=None sizes the front view from the points' projection (cfg bbox_fraction).
    """
    centre = np.median(points, 0)
    horizontal = T_setup[:3, 3] - centre
    horizontal[2] = 0
    horizontal /= np.linalg.norm(horizontal)
    down = np.radians(cfg["look_down_deg"])
    direction = np.cos(down) * horizontal + [0, 0, np.sin(down)]  # from the centroid to the camera

    def pose(angle_deg, d):
        c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
        return look_at(centre + d * np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ direction, centre)

    if distance is None:  # the projected size scales roughly as 1 / distance; refine a few times
        distance = 0.3
        for _ in range(5):
            distance *= box_fraction(project_box(points, pose(0, distance), K, shape, 0), shape) / cfg["bbox_fraction"]
    return pose, float(np.clip(distance, cfg["min_distance_m"], cfg["max_distance_m"]))


def transit(pose, distance, a0, a1, cfg):
    """Waypoints from the view at angle a0 to the one at a1: back out by retreat_m, follow the arc
    at that radius in steps of at most arc_step_deg, and come in again (each step is a joint-space
    move, so short arcs keep it from cutting toward the object)."""
    steps = max(1, int(np.ceil(abs(a1 - a0) / cfg["arc_step_deg"])))
    far = distance + cfg["retreat_m"]
    return [pose(a, far) for a in np.linspace(a0, a1, steps + 1)] + [pose(a1, distance)]
