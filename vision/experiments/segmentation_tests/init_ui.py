"""Click prompts on a single RGB frame (live capture, not a video editor)."""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from fp_capture.drawing import draw_prompts, overlay_mask

from experiments.segmentation_tests.screen_hints import draw_setup_banner
from experiments.segmentation_tests.ui_display import show_rgb, snap_window_to_frame

WINDOW = "SAM live tracking"


@dataclass
class PromptState:
    points: List[Tuple[float, float]] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)

    def clear(self) -> None:
        self.points.clear()
        self.labels.clear()

    @property
    def empty(self) -> bool:
        return not self.points


PredictFn = Callable[[List, List, bool], object]


class InitPromptUI:
    def __init__(self, predict: PredictFn):
        self.predict = predict
        self.rgb: Optional[np.ndarray] = None
        self.prompts = PromptState()
        self.mask: Optional[np.ndarray] = None
        self._dirty = False
        self._done = False
        self._confirmed = False
        self._cancelled = False

    def set_frame(self, rgb: np.ndarray) -> None:
        self.rgb = rgb.copy()
        self.prompts.clear()
        self.mask = None
        self._dirty = True

    def run(self) -> bool:
        if self.rgb is None:
            raise RuntimeError("No frame set for initialization")
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        snap_window_to_frame(WINDOW, self.rgb)
        cv2.setMouseCallback(WINDOW, self._on_mouse)
        self._done = False
        self._confirmed = False
        self._cancelled = False
        try:
            while not self._done:
                if self._dirty:
                    self._update_mask()
                image = overlay_mask(self.rgb, self.mask)
                image = draw_prompts(image, self.prompts.points, self.prompts.labels, None)
                ready = self.mask is not None and not self.prompts.empty
                hint_lines = [
                    "Left click: object    Right click: background",
                    "r: clear clicks",
                    "Enter: start tracking" + ("  (ready)" if ready else "  (add clicks first)"),
                    "q / Esc: quit",
                ]
                if self.prompts.empty:
                    hint_lines.insert(0, "Add at least one click on the object")
                elif self.mask is None or not self.mask.any():
                    hint_lines.insert(0, "no mask — adjust clicks")
                else:
                    hint_lines.insert(0, f"mask area {int(self.mask.sum())}")
                image = draw_setup_banner(image, "Object selection", hint_lines)
                show_rgb(WINDOW, image, snap=False)
                self._on_key(cv2.waitKey(20) & 0xFF)
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    self._cancelled = True
                    self._done = True
        finally:
            cv2.setMouseCallback(WINDOW, lambda *args: None)
        return self._confirmed and not self._cancelled

    def _update_mask(self) -> None:
        self._dirty = False
        if self.prompts.empty:
            self.mask = None
            return
        out = self.predict(self.prompts.points, self.prompts.labels, False)
        self.mask = out.mask

    def _on_key(self, key: int) -> None:
        if key in (ord("q"), 27):
            self._cancelled = True
            self._done = True
        elif key in (13, 10):
            if not self.prompts.empty and self.mask is not None:
                self._confirmed = True
                self._done = True
        elif key == ord("r"):
            self.prompts.clear()
            self._dirty = True

    def _on_mouse(self, event, x, y, _flags, _param) -> None:
        if self.rgb is None:
            return
        h, w = self.rgb.shape[:2]
        if x < 0 or y < 0 or x >= w or y >= h:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.prompts.points.append((float(x), float(y)))
            self.prompts.labels.append(1)
            self._dirty = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.prompts.points.append((float(x), float(y)))
            self.prompts.labels.append(0)
            self._dirty = True
