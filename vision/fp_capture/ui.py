"""OpenCV window for click/box segmentation, mesh selection, and multi-object saving.

Image area (segment screen):
    left click   positive point (object)
    right click  negative point (background)
    left drag    box (replaces any previous box)
Keys: s save, u undo, r reset, q/Esc quit (Esc goes back on the mesh screen).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .drawing import BOX_COLOR, draw_prompts, overlay_mask

WINDOW = "SAM3 object masks"
PANEL_WIDTH = 260
MARGIN = 14
BUTTON_HEIGHT = 34
BUTTON_GAP = 8
DRAG_THRESHOLD_PX = 6
FONT = cv2.FONT_HERSHEY_SIMPLEX

PredictFn = Callable[[Sequence, Sequence, Optional[Tuple]], Optional[Tuple[np.ndarray, float]]]
SaveFn = Callable[[np.ndarray, Optional[Path]], Path]


@dataclass
class Button:
    label: str
    action: str
    enabled: bool = True
    primary: bool = False
    rect: Tuple[int, int, int, int] = (0, 0, 0, 0)

    def contains(self, x: int, y: int) -> bool:
        x0, y0, x1, y1 = self.rect
        return x0 <= x < x1 and y0 <= y < y1


@dataclass
class Prompts:
    points: List[Tuple[float, float]] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)
    box: Optional[Tuple[float, float, float, float]] = None
    _history: list = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.points and self.box is None

    def add_point(self, point, label: int) -> None:
        self.points.append(point)
        self.labels.append(label)
        self._history.append(("point", None))

    def set_box(self, box) -> None:
        self._history.append(("box", self.box))
        self.box = box

    def undo(self) -> None:
        if not self._history:
            return
        kind, previous_box = self._history.pop()
        if kind == "point":
            self.points.pop()
            self.labels.pop()
        else:
            self.box = previous_box

    def clear(self) -> None:
        self.points, self.labels, self.box, self._history = [], [], None, []


class SegmentationApp:
    def __init__(self, rgb: np.ndarray, predict: PredictFn, save: SaveFn,
                 mesh_dirs: Sequence[Path]):
        self.rgb = rgb
        self.predict = predict
        self.save = save
        self.mesh_dirs = list(mesh_dirs)
        self.height, self.width = rgb.shape[:2]

        self.screen = "segment"
        self.prompts = Prompts()
        self.mask: Optional[np.ndarray] = None
        self.score: Optional[float] = None
        self.saved_masks: List[np.ndarray] = []
        self.status = ""
        self.done = False

        self._dirty = False
        self._buttons: List[Button] = []
        self._pending_action: Optional[str] = None
        self._drag_start: Optional[Tuple[int, int]] = None
        self._drag_end: Optional[Tuple[int, int]] = None

    def run(self) -> int:
        """Show the window until the user finishes or quits; return objects saved."""
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self._on_mouse)
        try:
            while not self.done:
                if self._dirty:
                    self._update_mask()
                cv2.imshow(WINDOW, cv2.cvtColor(self._render(), cv2.COLOR_RGB2BGR))
                self._on_key(cv2.waitKey(20) & 0xFF)
                if self._pending_action is not None:
                    action, self._pending_action = self._pending_action, None
                    self._dispatch(action)
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            cv2.destroyAllWindows()
        return len(self.saved_masks)

    # ----- screens -----------------------------------------------------------

    def _screen_buttons(self) -> Tuple[str, List[Button], List[str]]:
        """Return (title, buttons, help lines) for the current screen."""
        obj = f"obj{len(self.saved_masks) + 1}"
        if self.screen == "segment":
            return (f"Segment {obj}", [
                Button("Save  [s]", "save", enabled=self.mask is not None, primary=True),
                Button("Undo  [u]", "undo", enabled=not self.prompts.empty),
                Button("Reset  [r]", "reset", enabled=not self.prompts.empty),
                Button("Quit  [q]", "quit"),
            ], ["Left click: object", "Right click: background", "Left drag: box"])
        if self.screen == "mesh":
            buttons = [Button(d.name, f"mesh:{i}") for i, d in enumerate(self.mesh_dirs)]
            buttons.append(Button("No mesh", "no_mesh", primary=True))
            buttons.append(Button("Back  [Esc]", "back"))
            return (f"Mesh for {obj}", buttons, [])
        return (f"Saved obj{len(self.saved_masks)}", [
            Button("Add another  [a]", "add", primary=True),
            Button("Finish  [f]", "finish"),
        ], [])

    def _dispatch(self, action: str) -> None:
        if self.screen == "segment":
            if action == "save" and self.mask is not None:
                self.screen, self.status = "mesh", ""
            elif action == "undo":
                self.prompts.undo()
                self._dirty = True
            elif action == "reset":
                self.prompts.clear()
                self._dirty = True
            elif action == "quit":
                self.done = True
        elif self.screen == "mesh":
            if action == "back":
                self.screen = "segment"
            elif action == "no_mesh" or action.startswith("mesh:"):
                mesh_dir = (None if action == "no_mesh"
                            else self.mesh_dirs[int(action.split(":")[1])])
                self._save(mesh_dir)
        elif action == "add":
            self.prompts.clear()
            self.mask, self.score, self.status = None, None, ""
            self.screen = "segment"
        elif action == "finish":
            self.done = True

    def _save(self, mesh_dir: Optional[Path]) -> None:
        try:
            obj_dir = self.save(self.mask, mesh_dir)
        except (ValueError, OSError) as error:
            self.status = str(error)
            self.screen = "segment"
            print(f"Save failed: {error}")
            return
        self.saved_masks.append(self.mask)
        self.status = f"-> {obj_dir}"
        self.screen = "next"

    # ----- input -------------------------------------------------------------

    def _on_key(self, key: int) -> None:
        keymap = {
            "segment": {ord("s"): "save", ord("u"): "undo", ord("r"): "reset",
                        ord("q"): "quit", 27: "quit"},
            "mesh": {27: "back"},
            "next": {ord("a"): "add", ord("f"): "finish", ord("q"): "finish", 27: "finish"},
        }[self.screen]
        if key in keymap:
            self._pending_action = keymap[key]

    def _on_mouse(self, event, x, y, flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and x >= self.width:
            for button in self._buttons:
                if button.enabled and button.contains(x, y):
                    self._pending_action = button.action
            return
        if self.screen != "segment":
            return
        in_image = x < self.width and y < self.height
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

    def _update_mask(self) -> None:
        self._dirty = False
        result = self.predict(self.prompts.points, self.prompts.labels, self.prompts.box)
        self.mask, self.score = (None, None) if result is None else result
        if self.mask is not None and not self.mask.any():
            self.mask = None

    # ----- rendering ---------------------------------------------------------

    def _render(self) -> np.ndarray:
        title, buttons, help_lines = self._screen_buttons()
        panel_height = max(self.height, MARGIN * 2 + 40
                           + len(buttons) * (BUTTON_HEIGHT + BUTTON_GAP)
                           + (len(help_lines) + 3) * 22)
        canvas = np.zeros((panel_height, self.width + PANEL_WIDTH, 3), dtype=np.uint8)
        canvas[:self.height, :self.width] = self._render_image()
        canvas[:, self.width:] = (32, 32, 32)

        x0 = self.width + MARGIN
        y = MARGIN + 20
        cv2.putText(canvas, title, (x0, y), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y += 20
        for button in buttons:
            button.rect = (x0, y, self.width + PANEL_WIDTH - MARGIN, y + BUTTON_HEIGHT)
            self._draw_button(canvas, button)
            y += BUTTON_HEIGHT + BUTTON_GAP
        self._buttons = buttons

        y += 10
        lines = list(help_lines)
        if self.screen == "segment" and self.score is not None:
            lines.append(f"Mask score: {self.score:.3f}")
        if self.status:
            lines.append(self.status)
        for line in lines:
            y += 22
            cv2.putText(canvas, _fit(line, PANEL_WIDTH - 2 * MARGIN, 0.5), (x0, y),
                        FONT, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
        return canvas

    def _render_image(self) -> np.ndarray:
        image = self.rgb
        for saved in self.saved_masks:
            image = overlay_mask(image, saved, color=(150, 150, 150), alpha=0.3)
        image = overlay_mask(image, self.mask)
        if self.screen == "segment":
            image = draw_prompts(image, self.prompts.points, self.prompts.labels,
                                 self.prompts.box)
            if self._drag_start is not None and self._drag_end is not None:
                cv2.rectangle(image, self._drag_start, self._drag_end, BOX_COLOR, 1)
        return image

    @staticmethod
    def _draw_button(canvas: np.ndarray, button: Button) -> None:
        x0, y0, x1, y1 = button.rect
        if not button.enabled:
            fill, text = (48, 48, 48), (110, 110, 110)
        elif button.primary:
            fill, text = (40, 110, 200), (255, 255, 255)
        else:
            fill, text = (75, 75, 75), (240, 240, 240)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), fill, -1)
        label = _fit(button.label, x1 - x0 - 20, 0.55)
        (_, text_h), _ = cv2.getTextSize(label, FONT, 0.55, 1)
        cv2.putText(canvas, label, (x0 + 10, y0 + (BUTTON_HEIGHT + text_h) // 2),
                    FONT, 0.55, text, 1, cv2.LINE_AA)


def _fit(text: str, max_width: int, scale: float) -> str:
    """Truncate text with an ellipsis so it fits within max_width pixels."""
    if cv2.getTextSize(text, FONT, scale, 1)[0][0] <= max_width:
        return text
    while text and cv2.getTextSize(text + "...", FONT, scale, 1)[0][0] > max_width:
        text = text[:-1]
    return text + "..."
