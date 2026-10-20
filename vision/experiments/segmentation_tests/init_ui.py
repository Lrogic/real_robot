"""Click-based mask initialization on a frozen camera frame."""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from fp_capture.drawing import draw_prompts, overlay_mask

WINDOW = "SAM live tracking"
FONT = cv2.FONT_HERSHEY_SIMPLEX


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
    """Freeze-frame point prompts with live mask preview."""

    def __init__(self, predict: PredictFn):
        self.predict = predict
        self.rgb: Optional[np.ndarray] = None
        self.prompts = PromptState()
        self.mask: Optional[np.ndarray] = None
        self.score_line = ""
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
        """Return True if user confirmed with Enter, False if cancelled."""
        if self.rgb is None:
            raise RuntimeError("No frame set for initialization")
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, self._on_mouse)
        self._done = False
        self._confirmed = False
        self._cancelled = False
        try:
            while not self._done:
                if self._dirty:
                    self._update_mask()
                vis = self._render()
                cv2.imshow(WINDOW, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
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
            self.score_line = ""
            return
        out = self.predict(self.prompts.points, self.prompts.labels, False)
        self.mask = out.mask
        self.score_line = ""
        if out.presence:
            parts = [f"{k}={v:.3f}" for k, v in out.presence.items()]
            self.score_line = ", ".join(parts)

    def _render(self) -> np.ndarray:
        assert self.rgb is not None
        image = overlay_mask(self.rgb, self.mask)
        image = draw_prompts(image, self.prompts.points, self.prompts.labels, None)
        h, w = image.shape[:2]
        lines = [
            "Init: left=positive, right=negative, r=clear, Enter=track, q=quit",
            "Keep the scene still while clicking.",
        ]
        if self.score_line:
            lines.append(self.score_line)
        if self.mask is None and not self.prompts.empty:
            lines.append("(no mask)")
        y = 24
        for line in lines:
            cv2.putText(image, line, (8, y), FONT, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, line, (8, y), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            y += 22
        return image

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
