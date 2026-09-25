"""Image overlays shared by the interactive window and the saved outputs."""

import cv2
import numpy as np

MASK_COLOR = (30, 144, 255)  # RGB
POSITIVE_COLOR = (0, 220, 0)
NEGATIVE_COLOR = (230, 30, 30)
BOX_COLOR = (255, 200, 0)


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
