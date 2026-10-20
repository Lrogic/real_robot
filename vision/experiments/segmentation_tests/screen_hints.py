"""On-screen hint bars for setup and live tracking status."""

from typing import Sequence

import cv2
import numpy as np

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_SCALE = 0.52
_THICK = 1
_PAD = 10
_LINE = 22
_TITLE_EXTRA = 6
_TITLE_RGB = (255, 220, 120)
_TEXT_RGB = (245, 245, 245)
_WARN_RGB = (255, 180, 100)


def draw_setup_banner(rgb: np.ndarray, title: str, lines: Sequence[str]) -> np.ndarray:
    lines = [line for line in lines if line]
    if not title and not lines:
        return rgb

    h, w = rgb.shape[:2]
    title_size = cv2.getTextSize(title, _FONT, _SCALE + 0.06, _THICK + 1)[0] if title else (0, 0)
    body_sizes = [cv2.getTextSize(line, _FONT, _SCALE, _THICK)[0] for line in lines]
    bar_h = _PAD * 2 + title_size[1] + _TITLE_EXTRA + len(lines) * _LINE
    bar_h = min(bar_h, h // 2)

    out = rgb.copy()
    bar = out[:bar_h, :]
    tint = np.empty_like(bar)
    tint[:] = (28, 28, 28)
    cv2.addWeighted(tint, 0.88, bar, 0.12, 0, bar)
    out[:bar_h, :] = bar

    y = _PAD + title_size[1]
    if title:
        cv2.putText(out, title, (_PAD, y), _FONT, _SCALE + 0.06, _TITLE_RGB, _THICK + 1, cv2.LINE_AA)
        y += _TITLE_EXTRA + _LINE // 2
    for line, size in zip(lines, body_sizes):
        y += size[1] + (_LINE - size[1]) // 2
        cv2.putText(out, line, (_PAD, y), _FONT, _SCALE, _TEXT_RGB, _THICK, cv2.LINE_AA)
        y += (_LINE - size[1]) // 2
    return out


def draw_status_bar_bottom(rgb: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    """Compact readout at the bottom (mask / scores during tracking)."""
    lines = [line for line in lines if line]
    if not lines:
        return rgb

    h, w = rgb.shape[:2]
    sizes = [cv2.getTextSize(line, _FONT, _SCALE, _THICK)[0] for line in lines]
    bar_h = _PAD * 2 + len(lines) * _LINE
    bar_h = min(bar_h, h // 3)
    y0 = h - bar_h

    out = rgb.copy()
    bar = out[y0:h, :]
    tint = np.empty_like(bar)
    tint[:] = (28, 28, 28)
    cv2.addWeighted(tint, 0.88, bar, 0.12, 0, bar)
    out[y0:h, :] = bar

    y = y0 + _PAD + sizes[0][1]
    for line, size in zip(lines, sizes):
        color = _WARN_RGB if " |  no mask" in line else _TEXT_RGB
        cv2.putText(out, line, (_PAD, y), _FONT, _SCALE, color, _THICK, cv2.LINE_AA)
        y += _LINE
    return out
