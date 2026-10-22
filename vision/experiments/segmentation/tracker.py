"""One stateful HF SAM video-tracking session per trial, shared by live and recorded modes.

Every request (preview, init, track) runs the same path: the HF processor resizes and
normalizes the RGB frame on the GPU, the model runs one streamed frame under bfloat16
autocast, and the processor upsamples the mask logits to the input resolution, which are
thresholded at 0 and copied to the host together with the object-score logits.

A session tracks one object (start/step, OBJ_ID) or several (start_objects/step_objects,
caller-chosen IDs). With several objects the image is encoded once per frame and each
object keeps its own prompt, memories, and object pointers; HF applies no cross-object
constraint, so objects do not influence each other's masks.

Bounded storage: the HF streaming session keeps every input frame (~12 MiB each at
1024x1024 float32) and every frame's outputs (~3 MiB per object) forever, although the
model only reads the conditioning (init) frame, the last num_maskmem-1 frames' memories,
and the last max_object_pointers_in_encoder-1 frames' object pointers. After each step,
entries the model can no longer read are deleted for every object; native memory selection
is unchanged (validate.py checks pruned and unpruned sessions give identical outputs, for
one object and for several).
"""

import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from .models import AUTOCAST_DTYPE, HF_CACHE, SESSION_DTYPE, LoadedModel, load_model

OBJ_ID = 1

Point = Tuple[float, float]
Box = Tuple[float, float, float, float]


def _sigmoid(logit):
    return 1.0 / (1.0 + np.exp(-logit))


@dataclass(frozen=True)
class TrackResult:
    frame_idx: int        # index in the tracking session; 0 is the init frame
    mask: np.ndarray      # HxW bool at the input resolution
    presence_logit: float  # native object_score_logits
    elapsed_s: float      # host RGB in -> host mask and score out (see timing in README)

    @property
    def presence(self) -> float:
        """sigmoid(object_score_logits): object-presence score, not a calibrated mask accuracy."""
        return float(_sigmoid(self.presence_logit))

    @property
    def area(self) -> int:
        return int(self.mask.sum())


@dataclass(frozen=True)
class MultiTrackResult:
    frame_idx: int
    obj_ids: Tuple[int, ...]      # row order of masks and presence_logits
    masks: np.ndarray             # NxHxW bool
    presence_logits: np.ndarray   # N float
    elapsed_s: float

    def index(self, obj_id: int) -> int:
        return self.obj_ids.index(obj_id)

    def mask(self, obj_id: int) -> np.ndarray:
        return self.masks[self.index(obj_id)]

    def presence(self, obj_id: int) -> float:
        return float(_sigmoid(self.presence_logits[self.index(obj_id)]))


class HfVideoTracker:
    def __init__(self, model_key: str, device: str = "cuda", prune: bool = True,
                 loaded: Optional[LoadedModel] = None, sync_timing: bool = True,
                 cache_dir=HF_CACHE):
        """sync_timing: bracket each request with torch.cuda.synchronize() (experiments).

        Without it, elapsed_s still ends at completed work, because the mask and logit
        copies to the host block; it then relies on the caller having no GPU work in flight.
        """
        self.loaded = loaded or load_model(model_key, device, cache_dir)
        self.key = self.loaded.spec.key
        self._model = self.loaded.model
        self._processor = self.loaded.processor
        self._device = self.loaded.device
        self._prune_enabled = prune
        self._sync_timing = sync_timing
        # Step t+1 reads non-conditioning outputs back to t+1-history; older ones are dead.
        self.history = max(self._model.num_maskmem - 1,
                           self._model.config.max_object_pointers_in_encoder - 1)
        self._session = None
        self._size: Optional[Tuple[int, int]] = None
        self._obj_ids: Tuple[int, ...] = ()
        self._next_idx = 0

    @property
    def obj_ids(self) -> Tuple[int, ...]:
        return self._obj_ids

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

        masks, logits = self._forward(session, rgb, 0, add_prompt, (OBJ_ID,))
        return masks[0], float(_sigmoid(logits[0]))

    def warmup(self, rgb: np.ndarray, steps: int, num_objects: int = 1) -> float:
        """Run init + `steps` tracking requests on a disposable session; return seconds.

        Uses synthetic centered-box masks (one per object) so warm-up never depends on, or
        touches, the trial's prompt or temporal memory. The trial always starts from a fresh session.
        """
        h, w = rgb.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        mask[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = True
        started = time.perf_counter()
        self.start_objects(rgb, {obj_id: mask for obj_id in range(1, num_objects + 1)})
        for _ in range(steps):
            self.step_objects(rgb)
        self.close()
        torch.cuda.synchronize()
        return time.perf_counter() - started

    # ----- trial --------------------------------------------------------------

    def start(self, rgb: np.ndarray, mask: np.ndarray) -> TrackResult:
        """Begin a one-object trial: fresh session, binary mask as the frame-0 prompt."""
        return self._single(self.start_objects(rgb, {OBJ_ID: mask}))

    def step(self, rgb: np.ndarray) -> TrackResult:
        """Track the one-object trial on the next streamed frame."""
        if self._obj_ids != (OBJ_ID,):
            raise RuntimeError(f"step() needs a one-object trial; this one tracks {self._obj_ids}")
        return self._single(self.step_objects(rgb))

    def start_objects(self, rgb: np.ndarray, masks: Dict[int, np.ndarray]) -> MultiTrackResult:
        """Begin a trial: fresh session, one binary frame-0 mask prompt per object ID."""
        if not masks:
            raise ValueError("No init masks")
        for obj_id, mask in masks.items():
            mask = np.asarray(mask)
            if mask.dtype != bool or mask.shape != rgb.shape[:2]:
                raise ValueError(f"Init mask {obj_id} must be a bool {rgb.shape[:2]} array, "
                                 f"got {mask.dtype} {mask.shape}")
            if not mask.any():
                raise ValueError(f"Init mask {obj_id} is empty")
        self.close()
        self._session = self._new_session()
        self._size = rgb.shape[:2]
        obj_ids = tuple(int(i) for i in masks)
        session = self._session

        def add_prompt(_original_size):
            self._processor.add_inputs_to_inference_session(
                session, frame_idx=0, obj_ids=list(obj_ids),
                input_masks=[np.asarray(masks[i]) for i in obj_ids])

        result = self._timed(lambda: self._forward(session, rgb, 0, add_prompt, obj_ids), 0, obj_ids)
        self._obj_ids = obj_ids
        self._next_idx = 1
        return result

    def step_objects(self, rgb: np.ndarray) -> MultiTrackResult:
        """Track every object of the trial on the next streamed frame."""
        if self._session is None:
            raise RuntimeError("Call start() or start_objects() first")
        if rgb.shape[:2] != self._size:
            raise ValueError(f"Frame size {rgb.shape[:2]} differs from the init frame {self._size}")
        idx = self._next_idx

        def run():
            out = self._forward(self._session, rgb, idx, None, self._obj_ids)
            if self._prune_enabled:
                self._prune(idx)
            return out

        result = self._timed(run, idx, self._obj_ids)
        self._next_idx += 1
        return result

    def close(self) -> None:
        self._session = None
        self._size = None
        self._obj_ids = ()
        self._next_idx = 0

    def storage_counts(self) -> Tuple[int, int]:
        """(stored input frames, stored non-conditioning outputs summed over objects)."""
        if self._session is None:
            return 0, 0
        frames = len(self._session.processed_frames or {})
        outputs = sum(len(o["non_cond_frame_outputs"]) for o in self._session.output_dict_per_obj.values())
        return frames, outputs

    # ----- internals ----------------------------------------------------------

    def _new_session(self):
        return self._processor.init_video_session(inference_device=self._device, dtype=SESSION_DTYPE)

    @staticmethod
    def _single(result: MultiTrackResult) -> TrackResult:
        return TrackResult(result.frame_idx, result.masks[0], float(result.presence_logits[0]),
                           result.elapsed_s)

    def _timed(self, fn, idx: int, obj_ids: Tuple[int, ...]) -> MultiTrackResult:
        if self._sync_timing:
            # Nothing queued from earlier requests may leak into this measurement.
            torch.cuda.synchronize()
        started = time.perf_counter()
        masks, logits = fn()
        if self._sync_timing:
            # masks/logits are already host copies (blocking); the sync is a belt-and-braces stop.
            torch.cuda.synchronize()
        return MultiTrackResult(idx, obj_ids, masks, logits, time.perf_counter() - started)

    @torch.inference_mode()
    def _forward(self, session, rgb: np.ndarray, frame_idx: int, add_prompt,
                 obj_ids: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 uint8 RGB, got {rgb.dtype} {rgb.shape}")
        with torch.autocast("cuda", dtype=AUTOCAST_DTYPE):
            inputs = self._processor(images=rgb, device=self._device, return_tensors="pt")
            if add_prompt is not None:
                add_prompt(inputs.original_sizes[0])
            # Explicit frame_idx: pruned frames would otherwise shift HF's len()-based numbering.
            out = self._model(inference_session=session, frame_idx=frame_idx,
                              frame=inputs.pixel_values[0])
            if tuple(out.object_ids) != tuple(obj_ids):
                raise RuntimeError(f"Session tracks objects {out.object_ids}, expected {list(obj_ids)}")
            masks = self._processor.post_process_masks(
                [out.pred_masks], original_sizes=inputs.original_sizes, binarize=True)[0]
        host_masks = masks[:, 0].cpu().numpy()
        logits = out.object_score_logits.float().reshape(-1).cpu().numpy().astype(np.float64)
        if host_masks.shape[0] != len(obj_ids) or logits.shape[0] != len(obj_ids):
            raise RuntimeError(f"Expected {len(obj_ids)} masks and scores, got "
                               f"{host_masks.shape[0]} and {logits.shape[0]}")
        return host_masks, logits

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
