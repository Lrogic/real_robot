"""OpenCV display helpers (same camera frame size for SAM2 and SAM3)."""

import cv2
import numpy as np


def snap_window_to_frame(window: str, rgb: np.ndarray) -> None:
    """Resize the window client area to match the image (avoids a small tile in a large window)."""
    h, w = rgb.shape[:2]
    cv2.resizeWindow(window, w, h)


def show_rgb(window: str, rgb: np.ndarray, *, snap: bool = True) -> None:
    cv2.imshow(window, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if snap:
        snap_window_to_frame(window, rgb)
