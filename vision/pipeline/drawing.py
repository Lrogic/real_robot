"""Image overlays shared by the interactive windows and the saved outputs."""

from typing import List, Optional

import cv2
import numpy as np

MASK_COLOR = (30, 144, 255)  # RGB
POSITIVE_COLOR = (0, 220, 0)
NEGATIVE_COLOR = (230, 30, 30)
BOX_COLOR = (255, 200, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE_HEIGHT = 22


def project_point(point_cam: np.ndarray, K: np.ndarray):
    """Pixel (u, v) of a camera-frame point, or None if it is not in front of the camera."""
    if point_cam[2] <= 0:
        return None
    u, v = (K @ point_cam)[:2] / point_cam[2]
    return int(round(u)), int(round(v))


def draw_text_strip(width: int, lines: List[str], height: Optional[int] = None) -> np.ndarray:
    """Dark RGB strip with one line of text per entry."""
    strip = np.full((height or 12 + LINE_HEIGHT * len(lines), width, 3), 32, dtype=np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(strip, line, (10, 24 + LINE_HEIGHT * i), FONT, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    return strip


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=MASK_COLOR,
                 alpha: float = 0.45) -> np.ndarray:
    """Return a copy of rgb with the mask tinted and outlined."""
    out = rgb.copy()
    if mask is None or not mask.any():
        return out
    tint = np.empty_like(rgb)
    tint[:] = color
    out[mask] = (out[mask] * (1 - alpha) + tint[mask] * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, (255, 255, 255), 2)
    return out


def draw_prompts(rgb: np.ndarray, points, labels, box) -> np.ndarray:
    """Draw clicks (green positive, red negative) and the box onto a copy of rgb."""
    out = rgb.copy()
    if box is not None:
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        cv2.rectangle(out, (x0, y0), (x1, y1), BOX_COLOR, 2)
    for (x, y), label in zip(points, labels):
        center = (int(round(x)), int(round(y)))
        cv2.circle(out, center, 6, POSITIVE_COLOR if label else NEGATIVE_COLOR, -1)
        cv2.circle(out, center, 6, (255, 255, 255), 1)
    return out
