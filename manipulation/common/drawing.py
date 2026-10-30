"""Overlays for the tracking window: labels, tool poses and base-frame points projected with K."""

from typing import Optional

import cv2
import numpy as np

from calibration.transforms import transform_points
from pipeline.drawing import project_point

FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR = (235, 235, 235)


def draw_label(img, text, org, color=TEXT_COLOR, scale=0.55, thickness=1) -> None:
    """Text on a black box; org is the text baseline start."""
    (w, h), base = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x - 4, y - h - 4), (x + w + 4, y + base + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def draw_tool(img, T_cam_base, T_base_tool, color, K, axis_len=0.05) -> None:
    """Circle at the tool origin and a line back along its -x (toward the wrist)."""
    if T_base_tool is None:
        return
    T_cam_tool = T_cam_base @ T_base_tool
    origin = project_point(T_cam_tool[:3, 3], K)
    tip = project_point(T_cam_tool[:3, 3] - T_cam_tool[:3, 0] * axis_len, K)
    if origin is None:
        return
    cv2.circle(img, origin, 6, color, 2, cv2.LINE_AA)
    if tip is not None:
        cv2.line(img, origin, tip, color, 2, cv2.LINE_AA)


def draw_point_marker(img, T_cam_base, p_base, color, K, label: Optional[str] = None,
                      from_base=None, size=8) -> None:
    """Circle and cross at a base-frame point, optionally labelled and linked from another point."""
    if p_base is None:
        return
    p_cam = transform_points(T_cam_base, np.asarray(p_base, dtype=np.float64).reshape(1, 3))[0]
    uv = project_point(p_cam, K)
    if uv is None:
        return
    if from_base is not None:
        q_cam = transform_points(T_cam_base, np.asarray(from_base, dtype=np.float64).reshape(1, 3))[0]
        start = project_point(q_cam, K)
        if start is not None:
            cv2.line(img, start, uv, color, 1, cv2.LINE_AA)
            cv2.circle(img, start, 3, color, -1, cv2.LINE_AA)
    x, y = uv
    cv2.circle(img, uv, size, color, 2, cv2.LINE_AA)
    cv2.line(img, (x - size, y), (x + size, y), color, 1, cv2.LINE_AA)
    cv2.line(img, (x, y - size), (x, y + size), color, 1, cv2.LINE_AA)
    if label:
        draw_label(img, label, (x + size + 6, y - size), color, 0.45)


def box_hull(obj, K: np.ndarray) -> Optional[np.ndarray]:
    """Image convex hull of a tracked object's oriented bounding box, or None if not in front."""
    if obj.center_pose is None:
        return None
    lo, hi = obj.bbox
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    cam = transform_points(obj.center_pose, corners)
    if np.any(cam[:, 2] <= 0):
        return None
    uv = (cam @ K.T)[:, :2] / cam[:, 2:3]
    return cv2.convexHull(np.round(uv).astype(np.int32))


def fmt_vec(p) -> str:
    return "-" if p is None else "(" + ", ".join(f"{v:+.3f}" for v in p) + ")"
