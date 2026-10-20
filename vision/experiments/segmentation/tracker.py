"""One stateful HF SAM video-tracking session per trial, shared by live and recorded modes.

Every request (preview, init, track) runs the same path: the HF processor resizes and
normalizes the RGB frame on the GPU, the model runs one streamed frame under bfloat16
autocast, and the processor upsamples the mask logits to the input resolution, which are
thresholded at 0 and copied to the host together with the object-score logit.

Bounded storage: the HF streaming session keeps every input frame (~12 MiB each at
1024x1024 float32) and every frame's outputs (~3 MiB) forever, although the model only
reads the conditioning (init) frame, the last num_maskmem-1 frames' memories, and the last
max_object_pointers_in_encoder-1 frames' object pointers. After each step, entries the
model can no longer read are deleted; native memory selection is unchanged
(validate.py checks pruned and unpruned sessions give identical outputs).
"""

import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from .models import AUTOCAST_DTYPE, SESSION_DTYPE, LoadedModel, load_model

OBJ_ID = 1

Point = Tuple[float, float]
Box = Tuple[float, float, float, float]


@dataclass(frozen=True)
class TrackResult:
    frame_idx: int        # index in the tracking session; 0 is the init frame
    mask: np.ndarray      # HxW bool at the input resolution
    presence_logit: float  # native object_score_logits
    elapsed_s: float      # host RGB in -> host mask and score out (see timing in README)

    @property
    def presence(self) -> float:
        """sigmoid(object_score_logits): object-presence score, not a calibrated mask accuracy."""
        return float(1.0 / (1.0 + np.exp(-self.presence_logit)))

    @property
    def area(self) -> int:
        return int(self.mask.sum())


class HfVideoTracker:
    def __init__(self, model_key: str, device: str = "cuda", prune: bool = True,
                 loaded: Optional[LoadedModel] = None):
        self.loaded = loaded or load_model(model_key, device)
        self.key = self.loaded.spec.key
        self._model = self.loaded.model
        self._processor = self.loaded.processor
        self._device = self.loaded.device
        self._prune_enabled = prune
        # Step t+1 reads non-conditioning outputs back to t+1-history; older ones are dead.
        self.history = max(self._model.num_maskmem - 1,
                           self._model.config.max_object_pointers_in_encoder - 1)
        self._session = None
        self._size: Optional[Tuple[int, int]] = None
        self._next_idx = 0

    # ----- preview (disposable sessions, never the trial) -------------------

    def preview(self, rgb: np.ndarray, points: Sequence[Point], labels: Sequence[int],
                box: Optional[Box]) -> Optional[Tuple[np.ndarray, float]]:
        """Mask and presence score from clicks/box on one frame, in a throwaway session."""
        if not points and box is None:
            return None
        session = self._new_session()

        def add_prompt(original_size):
            self._processor.add_inputs_to_inference_session(
                session, frame_idx=0, obj_ids=OBJ_ID,
                input_points=[[[list(p) for p in points]]] if points else None,
                input_labels=[[list(labels)]] if points else None,
                input_boxes=[[list(box)]] if box is not None else None,
                original_size=original_size, clear_old_inputs=True)

        mask, logit = self._forward(session, rgb, 0, add_prompt)
        return mask, float(1.0 / (1.0 + np.exp(-logit)))

    def warmup(self, rgb: np.ndarray, steps: int) -> float:
        """Run init + `steps` tracking requests on a disposable session; return seconds.

        Uses a synthetic centered-box mask so warm-up never depends on, or touches, the
        trial's prompt or temporal memory. The trial always starts from a fresh session.
        """
        h, w = rgb.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        mask[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = True
        started = time.perf_counter()
        self.start(rgb, mask)
        for _ in range(steps):
            self.step(rgb)
        self.close()
        torch.cuda.synchronize()
        return time.perf_counter() - started

    # ----- trial --------------------------------------------------------------

    def start(self, rgb: np.ndarray, mask: np.ndarray) -> TrackResult:
        """Begin a trial: fresh session, binary mask as the frame-0 prompt."""
        mask = np.asarray(mask)
        if mask.dtype != bool or mask.shape != rgb.shape[:2]:
            raise ValueError(f"Init mask must be a bool {rgb.shape[:2]} array, got {mask.dtype} {mask.shape}")
        if not mask.any():
            raise ValueError("Init mask is empty")
        self.close()
        self._session = self._new_session()
        self._size = rgb.shape[:2]
        session = self._session

        def add_prompt(_original_size):
            self._processor.add_inputs_to_inference_session(
                session, frame_idx=0, obj_ids=OBJ_ID, input_masks=mask)

        result = self._timed(lambda: self._forward(session, rgb, 0, add_prompt), 0)
        self._next_idx = 1
        return result

    def step(self, rgb: np.ndarray) -> TrackResult:
        """Track the trial's object on the next streamed frame."""
        if self._session is None:
            raise RuntimeError("Call start() before step()")
        if rgb.shape[:2] != self._size:
            raise ValueError(f"Frame size {rgb.shape[:2]} differs from the init frame {self._size}")
        idx = self._next_idx

        def run():
            out = self._forward(self._session, rgb, idx, None)
            if self._prune_enabled:
                self._prune(idx)
            return out

        result = self._timed(run, idx)
        self._next_idx += 1
        return result

    def close(self) -> None:
        self._session = None
        self._size = None
        self._next_idx = 0

    def storage_counts(self) -> Tuple[int, int]:
        """(stored input frames, stored non-conditioning outputs) in the trial session."""
        if self._session is None:
            return 0, 0
        frames = len(self._session.processed_frames or {})
        outputs = sum(len(o["non_cond_frame_outputs"]) for o in self._session.output_dict_per_obj.values())
        return frames, outputs

    # ----- internals ----------------------------------------------------------

    def _new_session(self):
        return self._processor.init_video_session(inference_device=self._device, dtype=SESSION_DTYPE)

    def _timed(self, fn, idx: int) -> TrackResult:
        # Nothing queued from earlier requests may leak into this measurement.
        torch.cuda.synchronize()
        started = time.perf_counter()
        mask, logit = fn()
        # mask/logit are already host copies (blocking); the sync is a belt-and-braces stop.
        torch.cuda.synchronize()
        return TrackResult(idx, mask, logit, time.perf_counter() - started)

    @torch.inference_mode()
    def _forward(self, session, rgb: np.ndarray, frame_idx: int, add_prompt) -> Tuple[np.ndarray, float]:
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 uint8 RGB, got {rgb.dtype} {rgb.shape}")
        with torch.autocast("cuda", dtype=AUTOCAST_DTYPE):
            inputs = self._processor(images=rgb, device=self._device, return_tensors="pt")
            if add_prompt is not None:
                add_prompt(inputs.original_sizes[0])
            # Explicit frame_idx: pruned frames would otherwise shift HF's len()-based numbering.
            out = self._model(inference_session=session, frame_idx=frame_idx,
                              frame=inputs.pixel_values[0])
            if list(out.object_ids) != [OBJ_ID]:
                raise RuntimeError(f"Session tracks objects {out.object_ids}, expected only [{OBJ_ID}]")
            masks = self._processor.post_process_masks(
                [out.pred_masks], original_sizes=inputs.original_sizes, binarize=True)[0]
        mask = masks[0, 0].cpu().numpy()
        logit = float(out.object_score_logits.float().reshape(-1)[0].item())
        return mask, logit

    def _prune(self, idx: int) -> None:
        session = self._session
        for k in [k for k in session.processed_frames if k < idx]:
            del session.processed_frames[k]
        oldest_needed = idx + 1 - self.history
        for obj_idx, outputs in session.output_dict_per_obj.items():
            non_cond = outputs["non_cond_frame_outputs"]
            tracked = session.frames_tracked_per_obj[obj_idx]
            for store in (non_cond, tracked):
                for k in [k for k in store if k < oldest_needed]:
                    del store[k]
