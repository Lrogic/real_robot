"""SAM-based FoundationPose recovery, run inside the track stage's sequential loop (pipeline/tracking_loop.py).

Each loop iteration: read the newest RGB-D frameset, FoundationPose tracks every object,
then, at most sam_fps times per second, SAM tracks every object on that same RGB image.
Each object's SAM mask is compared with its FP mesh silhouette rendered at that frame's
pose. After failure_limit consecutive low-IoU observations the object is re-registered,
synchronously, from the triggering SAM mask and that frame's RGB-D and K. Neither tracker
runs during registration, and the SAM session is never reset or re-prompted.

Accepted limitations:
- Silhouettes use FoundationPose's pinhole projection with K. Without --undistort the RGB
  is RealSense's distorted image, so silhouettes are off by the lens distortion
  (fractions of a pixel near the centre, more at the borders), as for FP itself.
- Registration and SAM requests delay FP; frames captured meanwhile are skipped. If the
  object moves while it is being registered, the registered pose is stale when tracking
  resumes, and FP tracks on from there.
- No depth consistency check, and no compensation for gripper occlusion.
"""

import json
import statistics
import time
import traceback
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np

from ..drawing import draw_text_strip, overlay_mask
from ..foundationpose.multi_object_fp_tracker import MultiObjectTracker, RegistrationError
from ..run_folder import SavedObject, sam_object_id
from .recovery_config import RecoveryConfig
from .recovery_state import ObjectRecoveryState, RateSchedule, Verdict

SILHOUETTE_COLOR = (0, 255, 255)  # RGB cyan
TIMING_SAMPLES = 100_000  # per-request timings kept for the exit summary (bounded)
MIN_REGISTRATION_DEPTH_PX = 4  # FoundationPose.register's own minimum


@dataclass(frozen=True)
class LiveFrame:
    """One camera frameset; everything computed in an iteration refers to exactly this."""
    rgb: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    frame_number: int
    camera_timestamp_ms: float
    captured_at: float  # time.monotonic() when the frameset was ready in host memory


@contextmanager
def cpu_default_tensors():
    """Run HF code with torch's CPU default tensor type.

    FoundationPose's refiner sets torch.cuda.FloatTensor as the process-wide default on
    every call; HF model loading rejects it, and the HF tracker was validated with the CPU
    default.
    """
    import torch

    fp_default = torch.empty(0).is_cuda
    if fp_default:
        torch.set_default_tensor_type(torch.FloatTensor)
    try:
        yield
    finally:
        if fp_default:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)


def check_registration_inputs(mask: np.ndarray, depth: np.ndarray) -> Optional[str]:
    """Reason the mask/depth pair cannot be registered, or None if it can."""
    if mask.dtype != bool:
        return f"mask is {mask.dtype}, expected bool"
    if mask.shape != depth.shape:
        return f"mask {mask.shape} does not match depth {depth.shape}"
    valid = int(np.count_nonzero(mask & (depth >= 0.001)))
    if valid < MIN_REGISTRATION_DEPTH_PX:
        return f"only {valid} valid depth pixels inside the mask"
    return None


def _ms_stats(values: Sequence[float]) -> str:
    if not values:
        return "none"
    return (f"mean {statistics.fmean(values):.1f} ms, median {statistics.median(values):.1f} ms "
            f"(n={len(values)})")


class SamRecovery:
    def __init__(self, cfg: RecoveryConfig, tracker: MultiObjectTracker,
                 objects: Sequence[SavedObject], est_refine_iter: int, log_path: Path,
                 settings: dict, sam=None):
        """sam: an HfVideoTracker-compatible tracker to use instead of loading cfg.sam_model."""
        self.cfg = cfg
        self.tracker = tracker
        self.est_refine_iter = est_refine_iter
        self.states: List[ObjectRecoveryState] = []
        for tracked, saved in zip(tracker.objects, objects):
            if tracked.name != saved.name:
                raise ValueError(f"Tracker object {tracked.name} does not match saved object {saved.name}")
            self.states.append(ObjectRecoveryState(tracked.name, sam_object_id(saved.name),
                                                   cfg.iou_threshold, cfg.failure_limit))
        ids = [s.sam_id for s in self.states]
        if len(set(ids)) != len(ids):
            raise ValueError(f"SAM object IDs are not unique: {ids}")
        self._log = open(log_path, "a")
        self._event("start", settings=settings,
                    objects={s.name: {"sam_id": s.sam_id, "mesh": str(o.mesh_path)}
                             for s, o in zip(self.states, objects)})
        self.init_timing: Dict[str, float] = {}
        if sam is None:
            from ..segmentation.sam_model_loader import load_model
            from ..segmentation.sam_video_tracker import HfVideoTracker

            print(f"Loading SAM model {cfg.sam_model} from {cfg.hf_cache}...")
            # Loading too: from_pretrained refuses to run with a CUDA default device.
            with cpu_default_tensors():
                loaded = load_model(cfg.sam_model, cache_dir=cfg.hf_cache)
            sam = HfVideoTracker(cfg.sam_model, loaded=loaded, sync_timing=False)
            self.init_timing["sam_load_s"] = loaded.load_s
        self.sam = sam
        self.model_info = sam.loaded.describe() if hasattr(sam, "loaded") else {"key": cfg.sam_model}
        self.schedule: Optional[RateSchedule] = None
        self.disabled_reason: Optional[str] = None
        self.sam_ms = deque(maxlen=TIMING_SAMPLES)
        self.compare_ms = deque(maxlen=TIMING_SAMPLES)
        self.registrations: deque = deque(maxlen=TIMING_SAMPLES)
        self._recent_sam = 0
        self._last_sam_frame: Optional[int] = None
        self._shown: Optional[dict] = None  # masks of the newest SAM observation, for its own frame
        self._registering: set = set()

    # ----- initialization -----------------------------------------------------

    def initialize(self, rgb: np.ndarray, objects: Sequence[SavedObject], frame_label: str) -> None:
        """Warm up, then start the SAM session from the saved frame and the saved masks.

        These are the same RGB image and masks FoundationPose was registered with.
        """
        from ..segmentation.memory_usage import reset_peak_memory

        masks = {}
        for state, saved in zip(self.states, objects):
            if saved.mask.dtype != bool or saved.mask.shape != rgb.shape[:2]:
                raise ValueError(f"{saved.name}: saved mask must be bool {rgb.shape[:2]}")
            masks[state.sam_id] = saved.mask
        with cpu_default_tensors():
            self.init_timing["sam_warmup_s"] = self.sam.warmup(rgb, self.cfg.warmup_steps, len(masks))
            result = self.sam.start_objects(rgb, masks)
        self.init_timing["sam_init_ms"] = result.elapsed_s * 1000
        reset_peak_memory()
        print(f"SAM {self.cfg.sam_model} initialized on {frame_label} with "
              + ", ".join(f"{s.name} (SAM id {s.sam_id})" for s in self.states)
              + f"; warm-up {self.init_timing['sam_warmup_s']:.1f} s, "
                f"init {self.init_timing['sam_init_ms']:.0f} ms")
        self._event("sam_initialized", frame=frame_label, model=self.model_info, **self.init_timing,
                    init_presence={s.name: result.presence(s.sam_id) for s in self.states})

    def start_loop(self) -> None:
        self.schedule = RateSchedule(self.cfg.sam_fps, time.monotonic())

    # ----- per iteration ------------------------------------------------------

    def after_fp(self, frame: LiveFrame, display: Callable[[np.ndarray], None]) -> bool:
        """Run a scheduled SAM comparison on the frame FP just tracked; True if SAM ran."""
        if self.disabled_reason is not None or not self.schedule.due(time.monotonic()):
            return False
        try:
            with cpu_default_tensors():
                result = self.sam.step_objects(frame.rgb)
        except Exception as error:
            # An exception is not an empty mask; the session may be inconsistent, so stop using it.
            traceback.print_exc()
            self.disabled_reason = f"SAM failed at cam frame {frame.frame_number}: {error!r}"
            print(f"[recovery] {self.disabled_reason}; recovery disabled, FP tracking continues")
            self._event("sam_error", frame_number=frame.frame_number, error=repr(error))
            return False
        if result.masks.dtype != bool or result.masks.shape[1:] != frame.depth.shape:
            raise RuntimeError(f"SAM masks {result.masks.dtype} {result.masks.shape} do not match "
                               f"the frame {frame.depth.shape}")
        self.sam_ms.append(result.elapsed_s * 1000)
        self._recent_sam += 1

        compare_started = time.perf_counter()
        sam_masks = {s.sam_id: result.mask(s.sam_id) for s in self.states}
        nonempty = [i for i, s in enumerate(self.states) if sam_masks[s.sam_id].any()]
        height, width = frame.depth.shape
        rendered = self.tracker.render_silhouettes(nonempty, frame.K, height, width)
        silhouettes = {i: rendered[row] for row, i in enumerate(nonempty)}
        now = time.monotonic()
        verdicts = []
        for i, state in enumerate(self.states):
            verdicts.append(state.observe(sam_masks[state.sam_id], silhouettes.get(i),
                                          result.presence(state.sam_id), frame.frame_number, now))
        compare_ms = (time.perf_counter() - compare_started) * 1000
        self.compare_ms.append(compare_ms)

        skipped = (frame.frame_number - self._last_sam_frame - 1) if self._last_sam_frame is not None else None
        self._last_sam_frame = frame.frame_number
        self._shown = {"frame_number": frame.frame_number, "sam": sam_masks, "silhouettes": silhouettes}
        self._print_observation(frame, result, compare_ms, verdicts, skipped)
        self._event("sam_observation", frame_number=frame.frame_number,
                    camera_timestamp_ms=frame.camera_timestamp_ms, sam_frame_idx=result.frame_idx,
                    camera_frames_since_last_sam=skipped, sam_ms=result.elapsed_s * 1000,
                    compare_ms=compare_ms,
                    objects={s.name: {"verdict": v.value, "iou": s.last_iou, "presence": s.last_presence,
                                      "sam_area": int(np.count_nonzero(sam_masks[s.sam_id])),
                                      "fp_area": (int(np.count_nonzero(silhouettes[i]))
                                                  if i in silhouettes else None),
                                      "mismatch_count": s.mismatch_count}
                             for i, (s, v) in enumerate(zip(self.states, verdicts))})

        triggered = [i for i, v in enumerate(verdicts) if v is Verdict.TRIGGER]
        if triggered:
            self._registering = {self.states[i].name for i in triggered}
            display(self.draw(self.tracker.draw(frame.rgb, frame.K), frame))
            for i in triggered:
                self._reregister(i, sam_masks[self.states[i].sam_id], frame)
            self._registering = set()
        return True

    def _reregister(self, index: int, mask: np.ndarray, frame: LiveFrame) -> None:
        state = self.states[index]
        problem = check_registration_inputs(mask, frame.depth)
        started = time.perf_counter()
        if problem is None:
            try:
                self.tracker.register_one(index, frame.rgb, frame.depth, frame.K, mask, self.est_refine_iter)
            except RegistrationError as error:
                problem = str(error)
        elapsed_ms = (time.perf_counter() - started) * 1000
        state.registered(problem is None, frame.frame_number)
        self.registrations.append({"name": state.name, "frame_number": frame.frame_number,
                                   "ok": problem is None, "ms": elapsed_ms})
        if problem is None:
            print(f"[recovery] re-registered {state.name} on cam frame {frame.frame_number} "
                  f"from its SAM mask ({elapsed_ms:.0f} ms)")
        else:
            print(f"[recovery] re-registration of {state.name} on cam frame {frame.frame_number} "
                  f"FAILED ({problem}); keeping its previous FP state")
        self._event("registration", name=state.name, frame_number=frame.frame_number,
                    camera_timestamp_ms=frame.camera_timestamp_ms, ok=problem is None,
                    error=problem, ms=elapsed_ms, mask_area=int(np.count_nonzero(mask)))

    # ----- display ------------------------------------------------------------

    def draw(self, vis: np.ndarray, frame: LiveFrame) -> np.ndarray:
        """Overlay SAM masks/FP silhouettes on the frame they were computed for; add a status strip."""
        shown = self._shown
        on_this_frame = shown is not None and shown["frame_number"] == frame.frame_number
        if on_this_frame:
            for i, (state, tracked) in enumerate(zip(self.states, self.tracker.objects)):
                vis = overlay_mask(vis, shown["sam"][state.sam_id], color=tracked.color, alpha=0.35)
                if i in shown["silhouettes"]:
                    contours, _ = cv2.findContours(shown["silhouettes"][i].astype(np.uint8),
                                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(vis, contours, -1, SILHOUETTE_COLOR, 1)
        header = f"{self.cfg.sam_model} <={self.cfg.sam_fps:g} Hz | frame {frame.frame_number} | "
        if self.disabled_reason is not None:
            header += "RECOVERY DISABLED (SAM error)"
        elif on_this_frame:
            header += "SAM fill + FP outline (cyan): this frame"
        else:
            header += "no SAM here; masks only on own frame"
        lines = [header]
        for state in self.states:
            lines.append(self._status_line(state, frame))
        return np.vstack([vis, draw_text_strip(vis.shape[1], lines)])

    def _status_line(self, state: ObjectRecoveryState, frame: LiveFrame) -> str:
        line = f"{state.name}: "
        if state.name in self._registering:
            return line + "REGISTERING from SAM mask..."
        if state.last_frame_number is None:
            return line + "waiting for first SAM comparison"
        age = (f"@{state.last_frame_number} ({frame.frame_number - state.last_frame_number} fr, "
               f"{time.monotonic() - state.last_observed_at:.1f}s)")
        presence = "-" if state.last_presence is None else f"{state.last_presence:.2f}"
        if state.sam_empty:
            return f"{line}Object lost - SAM mask empty | pres {presence} | {age}"
        regs = f" | rereg {state.registrations}" + (f"/{state.failed_registrations} failed"
                                                    if state.failed_registrations else "")
        return (f"{line}IoU {state.last_iou:.2f} | miss {state.mismatch_count}/{state.failure_limit} | "
                f"pres {presence} | {age}{regs}")

    # ----- logging ------------------------------------------------------------

    def _print_observation(self, frame, result, compare_ms, verdicts, skipped) -> None:
        parts = []
        for state, verdict in zip(self.states, verdicts):
            if verdict is Verdict.SAM_EMPTY:
                parts.append(f"{state.name} EMPTY (presence {state.last_presence:.2f})")
            else:
                parts.append(f"{state.name} IoU {state.last_iou:.2f} {verdict.value} "
                             f"{state.mismatch_count}/{state.failure_limit} (presence {state.last_presence:.2f})")
        gap = "" if skipped is None else f" | {skipped} cam frames since last SAM"
        print(f"[sam] cam frame {frame.frame_number} proc {result.frame_idx} | sam {result.elapsed_s * 1000:.1f} ms"
              f" | silhouette+IoU {compare_ms:.1f} ms{gap} | " + " | ".join(parts), flush=True)

    def report(self, window_s: float) -> None:
        """Periodic SAM rate, memory, and session storage (no device synchronization)."""
        from ..segmentation.memory_usage import sample_memory

        memory = sample_memory()
        frames, outputs = self.sam.storage_counts()
        rate = self._recent_sam / max(window_s, 1e-6)
        print(f"[recovery recent] SAM {self._recent_sam} requests ({rate:.2f} Hz) | {memory.line()} | "
              f"SAM session stores {frames} frames / {outputs} outputs")
        self._event("memory", cuda_allocated=memory.cuda_allocated, cuda_reserved=memory.cuda_reserved,
                    cuda_peak_allocated=memory.cuda_peak_allocated, cpu_rss=memory.cpu_rss,
                    sam_rate_hz=rate, sam_stored_frames=frames, sam_stored_outputs=outputs)
        self._recent_sam = 0

    def summary(self, fp_seconds: Sequence[float], elapsed_s: float) -> None:
        from ..segmentation.memory_usage import sample_memory

        fp_ms = [1000 * s for s in fp_seconds]
        sam_ms, compare_ms = list(self.sam_ms), list(self.compare_ms)
        reg_ms = [r["ms"] for r in self.registrations]
        print("\n=== FoundationPose + SAM recovery ===")
        print(f"  FP tracking (all objects, per frame): {_ms_stats(fp_ms)}")
        print(f"  SAM tracking (all objects, per request): {_ms_stats(sam_ms)}, "
              f"{len(sam_ms) / max(elapsed_s, 1e-6):.2f} requests/s")
        print(f"  Silhouette render + IoU: {_ms_stats(compare_ms)}")
        ok = sum(r["ok"] for r in self.registrations)
        print(f"  Re-registrations: {ok} ok, {len(self.registrations) - ok} failed; {_ms_stats(reg_ms)}")
        for r in self.registrations:
            print(f"    {r['name']} cam frame {r['frame_number']}: {'ok' if r['ok'] else 'FAILED'} {r['ms']:.0f} ms")
        print("  Initialization (excluded above): " + ", ".join(
            f"{k} {v:.2f}" for k, v in self.init_timing.items()))
        for state in self.states:
            print(f"  {state.name} (SAM id {state.sam_id}): {state.observations} observations, "
                  f"{state.empty_observations} empty, {state.mismatches} mismatches, "
                  f"{state.registrations} re-registrations ({state.failed_registrations} failed)")
        memory = sample_memory()
        print(f"  Memory at exit: {memory.line()} (peak since SAM init)")
        if self.disabled_reason:
            print(f"  Recovery was disabled: {self.disabled_reason}")
        self._event("summary", fp_ms_mean=statistics.fmean(fp_ms) if fp_ms else None,
                    fp_ms_median=statistics.median(fp_ms) if fp_ms else None,
                    sam_ms_mean=statistics.fmean(sam_ms) if sam_ms else None,
                    sam_ms_median=statistics.median(sam_ms) if sam_ms else None,
                    compare_ms_median=statistics.median(compare_ms) if compare_ms else None,
                    registrations=list(self.registrations), init=self.init_timing,
                    disabled_reason=self.disabled_reason, elapsed_s=elapsed_s)

    def close(self) -> None:
        self.sam.close()
        self._log.close()

    def _event(self, kind: str, **data) -> None:
        self._log.write(json.dumps({"event": kind, "time": time.time(), **data}, default=str) + "\n")
        self._log.flush()
