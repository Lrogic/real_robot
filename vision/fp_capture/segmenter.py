"""SAM3 point/box segmentation of a single image (the SAM 1 task API)."""

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import PIL.Image
import torch

from .cli import DEFAULT_SAM3_CHECKPOINT

Point = Tuple[float, float]
Box = Tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels


class Sam3Segmenter:
    def __init__(self, checkpoint: Path = DEFAULT_SAM3_CHECKPOINT):
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        if not checkpoint.is_file():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint}")
        if not torch.cuda.is_available():
            raise RuntimeError("SAM3 requires a CUDA GPU")
        print(f"Loading SAM3 from {checkpoint}...")
        self._model = build_sam3_image_model(
            checkpoint_path=str(checkpoint), load_from_HF=False,
            enable_inst_interactivity=True,
        )
        self._processor = Sam3Processor(self._model)
        self._state = None

    @torch.inference_mode()
    def set_image(self, rgb: np.ndarray) -> None:
        # Sam3Processor reads HxW from PIL images; raw HxWxC arrays are misread.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self._state = self._processor.set_image(PIL.Image.fromarray(rgb))
        # The first prediction is several seconds slower; pay that cost before the UI opens.
        height, width = rgb.shape[:2]
        self.predict([(width / 2, height / 2)], [1], None)

    @torch.inference_mode()
    def predict(self, points: Sequence[Point], labels: Sequence[int],
                box: Optional[Box]) -> Optional[Tuple[np.ndarray, float]]:
        """Return (HxW bool mask, score) for the best mask, or None without prompts."""
        if self._state is None:
            raise RuntimeError("Call set_image before predict")
        if not points and box is None:
            return None
        # A lone click is ambiguous, so ask for several candidates and keep the best.
        multimask = box is None and len(points) == 1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            masks, scores, _ = self._model.predict_inst(
                self._state,
                point_coords=np.asarray(points, dtype=np.float32) if points else None,
                point_labels=np.asarray(labels, dtype=np.int32) if points else None,
                box=np.asarray(box, dtype=np.float32)[None] if box is not None else None,
                multimask_output=multimask,
            )
        best = int(np.argmax(scores))
        return masks[best] > 0, float(scores[best])
