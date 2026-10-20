"""Frozen-frame object selection with a live mask preview (same conventions as fp_capture/ui.py).

    left click   positive point (object)
    right click  negative point (background)
    left drag    box (replaces any previous box)
Keys: Enter confirm, u undo, r reset, q/Esc cancel.
"""

from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from fp_capture.drawing import BOX_COLOR, draw_prompts, overlay_mask
from fp_capture.ui import DRAG_THRESHOLD_PX, FONT, Prompts

PreviewFn = Callable[[Sequence, Sequence, Optional[Tuple]], Optional[Tuple[np.ndarray, float]]]
LINE_HEIGHT = 22


def draw_text_strip(width: int, lines: List[str], height: Optional[int] = None) -> np.ndarray:
    """Dark RGB strip with one line of text per entry."""
    strip = np.full((height or 12 + LINE_HEIGHT * len(lines), width, 3), 32, dtype=np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(strip, line, (10, 24 + LINE_HEIGHT * i), FONT, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    return strip


class InitMaskUI:
    def __init__(self, rgb: np.ndarray, preview: PreviewFn, title: str, window: str):
        self.rgb = rgb
        self.preview = preview
        self.title = title
        self.window = window
        self.height, self.width = rgb.shape[:2]
        self.prompts = Prompts()
        self.mask: Optional[np.ndarray] = None
        self.score: Optional[float] = None
        self.status = ""
        self._dirty = False
        self._done = False
        self._confirmed = False
        self._drag_start = self._drag_end = None

    def run(self) -> Optional[np.ndarray]:
        """Return the confirmed HxW bool mask, or None if the user cancelled."""
        cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window, self._on_mouse)
        try:
            while not self._done:
                if self._dirty:
                    self._update_mask()
                cv2.imshow(self.window, cv2.cvtColor(self._render(), cv2.COLOR_RGB2BGR))
                self._on_key(cv2.waitKey(20) & 0xFF)
                if cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            cv2.setMouseCallback(self.window, lambda *_: None)
        return self.mask if self._confirmed else None

    def _update_mask(self) -> None:
        self._dirty = False
        result = self.preview(self.prompts.points, self.prompts.labels, self.prompts.box)
        self.mask, self.score = (None, None) if result is None else result
        if self.mask is not None and not self.mask.any():
            self.mask = None
        self.status = ""

    def _on_key(self, key: int) -> None:
        if key in (13, 10):
            if self.mask is None:
                self.status = "No valid mask yet: add clicks on the object"
            else:
                self._confirmed = self._done = True
        elif key == ord("u"):
            self.prompts.undo()
            self._dirty = True
        elif key == ord("r"):
            self.prompts.clear()
            self._dirty = True
        elif key in (ord("q"), 27):
            self._done = True

    def _on_mouse(self, event, x, y, _flags, _param) -> None:
        in_image = 0 <= x < self.width and 0 <= y < self.height
        clamped = (min(max(x, 0), self.width - 1), min(max(y, 0), self.height - 1))
        if event == cv2.EVENT_LBUTTONDOWN and in_image:
            self._drag_start, self._drag_end = (x, y), None
        elif event == cv2.EVENT_MOUSEMOVE and self._drag_start is not None:
            self._drag_end = clamped
        elif event == cv2.EVENT_LBUTTONUP and self._drag_start is not None:
            (x0, y0), (x1, y1) = self._drag_start, clamped
            if max(abs(x1 - x0), abs(y1 - y0)) >= DRAG_THRESHOLD_PX:
                self.prompts.set_box((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
            else:
                self.prompts.add_point((x0, y0), 1)
            self._drag_start = self._drag_end = None
            self._dirty = True
        elif event == cv2.EVENT_RBUTTONDOWN and in_image:
            self.prompts.add_point((x, y), 0)
            self._dirty = True

    def _render(self) -> np.ndarray:
        image = overlay_mask(self.rgb, self.mask)
        image = draw_prompts(image, self.prompts.points, self.prompts.labels, self.prompts.box)
        if self._drag_start is not None and self._drag_end is not None:
            cv2.rectangle(image, self._drag_start, self._drag_end, BOX_COLOR, 1)
        if self.mask is not None:
            state = f"mask {int(self.mask.sum())} px, preview presence {self.score:.3f}"
        else:
            state = "no mask" if not self.prompts.empty else "add clicks on the object"
        lines = [self.title,
                 "L click: object  R click: background  L drag: box",
                 "Enter: confirm  u: undo  r: reset  q/Esc: quit",
                 self.status or state]
        return np.vstack([image, draw_text_strip(self.width, lines)])
