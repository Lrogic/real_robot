"""One model on one recorded video, meant to run in its own (spawned) process.

Outputs in out_dir: frames.csv (one row per request), summary.json, and predictions.npz
(source/processed indices, presence logits, bit-packed host masks) for later rendering.
"""

import os
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

PREDICTIONS_FILE = "predictions.npz"
SUMMARY_FILE = "summary.json"
FRAMES_FILE = "frames.csv"


def run_model_on_video(model_key: str, video: Path, mask: np.ndarray, tracking_fps: Optional[float],
                       out_dir: Path, warmup_steps: int) -> None:
    from pipeline.segmentation.memory_usage import reset_peak_memory
    from pipeline.segmentation.sam_video_tracker import HfVideoTracker

    from .run_log import (MEMORY_DEFINITION, TIMING_DEFINITION, RunLog,
                          software_versions, write_json)
    from .video_io import FrameSelector, array_sha256, iter_frames, video_info

    out_dir.mkdir(parents=True, exist_ok=False)
    process_started = time.time()
    tracker = HfVideoTracker(model_key)
    info = video_info(video)
    selector = FrameSelector(info.fps, tracking_fps)
    frames = iter_frames(video)
    _, frame0 = next(frames)
    if frame0.shape[:2] != mask.shape:
        raise ValueError(f"Mask {mask.shape} does not match the first frame {frame0.shape[:2]}")

    warmup_s = tracker.warmup(frame0, warmup_steps)
    reset_peak_memory()
    log = RunLog(model_key)
    results = [tracker.start(frame0, mask)]
    sources = [0]
    log.add("init", results[0], tracker, source_idx=0)

    packed, logits, processed = [np.packbits(results[0].mask)], [results[0].presence_logit], [0]
    decoded = 1
    log.loop_started = time.perf_counter()
    for source_idx, rgb in frames:
        decoded += 1
        if not selector.selected(source_idx):
            continue
        result = tracker.step(rgb)
        log.add("track", result, tracker, source_idx=source_idx)
        sources.append(source_idx)
        processed.append(result.frame_idx)
        logits.append(result.presence_logit)
        packed.append(np.packbits(result.mask))
    log.loop_ended = time.perf_counter()
    tracker.close()

    np.savez(out_dir / PREDICTIONS_FILE, source_idx=np.asarray(sources),
             processed_idx=np.asarray(processed), presence_logit=np.asarray(logits),
             masks_packed=np.stack(packed), mask_shape=np.asarray(mask.shape))
    log.write_frames_csv(out_dir / FRAMES_FILE)
    summary = log.summary()
    summary.update({
        "model_info": tracker.loaded.describe(),
        "video": str(video),
        "decoded_frames": decoded,
        "frame_selection": selector.describe(),
        "init_mask_sha256": array_sha256(mask),
        "warmup": {"steps": warmup_steps, "seconds": warmup_s,
                   "policy": "disposable session on source frame 0 with a synthetic centered-box mask; "
                             "init + `steps` tracking requests, all excluded; the trial uses a fresh session"},
        "peak_reset_point": "after model load, warm-up; immediately before the trial's init request",
        "timing_definition": TIMING_DEFINITION,
        "memory_definition": MEMORY_DEFINITION,
        "software": software_versions(),
        "pid": os.getpid(),
        "process_started": process_started,
        "process_finished": time.time(),
    })
    write_json(out_dir / SUMMARY_FILE, summary)
    log.print_report(summary)


def worker_entry(*args) -> None:
    try:
        run_model_on_video(*args)
    except BaseException:
        traceback.print_exc()
        raise SystemExit(1)
