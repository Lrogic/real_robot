"""Timing and GPU memory helpers for live tracking experiments."""

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch


@dataclass
class FrameStats:
    frame_number: int
    video_frame_idx: int
    mask_area: int
    skipped_camera_frames: int
    wait_s: float
    prepare_s: float
    infer_s: float
    loop_s: float
    warmup: bool = False


@dataclass
class RunMetrics:
    warmup_frames: int = 5
    samples: List[FrameStats] = field(default_factory=list)
    peak_gpu_bytes: int = 0

    def record(self, sample: FrameStats) -> None:
        self.samples.append(sample)
        if torch.cuda.is_available():
            self.peak_gpu_bytes = max(self.peak_gpu_bytes, torch.cuda.max_memory_allocated())

    def _steady(self) -> List[FrameStats]:
        return [s for s in self.samples if not s.warmup][self.warmup_frames :]

    def report(self, label: str, window_s: float, camera_fps: int) -> None:
        steady = self._steady()
        if not steady:
            return
        infer_ms = 1000 * np.mean([s.infer_s for s in steady])
        loop_ms = 1000 * np.mean([s.loop_s for s in steady])
        wait_ms = 1000 * np.mean([s.wait_s for s in steady])
        prep_ms = 1000 * np.mean([s.prepare_s for s in steady])
        processed_fps = len(steady) / max(window_s, 1e-6)
        skipped = sum(s.skipped_camera_frames for s in steady)
        peak_gb = self.peak_gpu_bytes / (1024 ** 3) if self.peak_gpu_bytes else 0.0
        print(
            f"[{label}] {len(steady)} steady frames in {window_s:.1f}s: "
            f"{processed_fps:.1f} processed fps (camera configured {camera_fps} fps), "
            f"{skipped} camera frame numbers skipped"
        )
        print(
            f"  Mean: infer {infer_ms:.1f} ms, loop {loop_ms:.1f} ms, "
            f"camera wait {wait_ms:.1f} ms, prepare {prep_ms:.1f} ms, "
            f"peak GPU alloc {peak_gb:.2f} GiB"
        )


class CudaTimer:
    """CUDA event timer; falls back to perf_counter on CPU."""

    def __init__(self):
        self._use_cuda = torch.cuda.is_available()
        if self._use_cuda:
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
        self._t0 = 0.0

    def start(self) -> None:
        if self._use_cuda:
            self._start.record()
        else:
            self._t0 = time.perf_counter()

    def stop(self) -> float:
        if self._use_cuda:
            self._end.record()
            torch.cuda.synchronize()
            return self._start.elapsed_time(self._end) / 1000.0
        return time.perf_counter() - self._t0
