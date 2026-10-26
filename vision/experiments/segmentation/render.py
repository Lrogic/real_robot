"""Annotated frames and side-by-side comparison panels (never inside the timed interval)."""

from typing import List, Optional

import numpy as np

from pipeline.drawing import draw_text_strip, overlay_mask

HEADER_LINES = 2


def annotate(rgb: np.ndarray, mask: Optional[np.ndarray], model: str,
             presence: Optional[float], status: str) -> np.ndarray:
    """Mask over its input frame, with a header strip above so the image keeps its aspect ratio."""
    score = "presence -" if presence is None else f"presence {presence:.3f}"
    header = draw_text_strip(rgb.shape[1], [f"{model} | {score}", status])
    return np.vstack([header, overlay_mask(rgb, mask)])


def side_by_side(panels: List[np.ndarray]) -> np.ndarray:
    heights = {p.shape[0] for p in panels}
    if len(heights) != 1:
        raise ValueError(f"Panels have different heights: {sorted(heights)}")
    separator = np.full((panels[0].shape[0], 4, 3), 255, dtype=np.uint8)
    out = [panels[0]]
    for panel in panels[1:]:
        out += [separator, panel]
    return np.hstack(out)
