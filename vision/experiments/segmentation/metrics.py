"""Per-request timing/memory records, terminal lines, and the machine-readable summary."""

import csv
import json
import platform
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import torch

MIB = 2 ** 20

TIMING_DEFINITION = (
    "tracking_ms: time.perf_counter() from an RGB uint8 frame in host memory (after "
    "torch.cuda.synchronize() so no earlier GPU work is pending) through HF preprocessing "
    "(upload, resize, normalize on the GPU), one streamed model forward, mask upsampling and "
    "thresholding, session pruning, and the blocking host copy of the mask and object-score "
    "logit, followed by torch.cuda.synchronize(). Excludes camera waiting, video decoding, "
    "prompting, initialization, warm-up, UI rendering, logging, and video writing."
)
MEMORY_DEFINITION = (
    "cuda_allocated/reserved: current torch.cuda.memory_allocated/memory_reserved after the "
    "request. cuda_peak_allocated: torch.cuda.max_memory_allocated high-water mark since the "
    "reset immediately before the trial's init request (after model load, warm-up, and "
    "previews). cpu_rss: current VmRSS of the tracking process from /proc/self/status."
)


def cpu_rss_bytes() -> int:
    with open("/proc/self/status") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS missing from /proc/self/status")


@dataclass(frozen=True)
class MemorySample:
    cuda_allocated: int
    cuda_reserved: int
    cuda_peak_allocated: int
    cpu_rss: int

    def line(self) -> str:
        return (f"cuda alloc {self.cuda_allocated / MIB:.0f} reserved {self.cuda_reserved / MIB:.0f} "
                f"peak {self.cuda_peak_allocated / MIB:.0f} MiB | rss {self.cpu_rss / MIB:.0f} MiB")


def sample_memory() -> MemorySample:
    return MemorySample(torch.cuda.memory_allocated(), torch.cuda.memory_reserved(),
                        torch.cuda.max_memory_allocated(), cpu_rss_bytes())


def reset_peak_memory() -> None:
    torch.cuda.reset_peak_memory_stats()


@dataclass
class FrameRecord:
    model: str
    processed_idx: int          # session frame index (0 = init)
    source_idx: Optional[int]   # video frame index, or RealSense frame number when live
    kind: str                   # "init" or "track"
    tracking_ms: float
    presence: float
    presence_logit: float
    mask_area: int
    cuda_allocated_mib: float
    cuda_reserved_mib: float
    cuda_peak_allocated_mib: float
    cpu_rss_mib: float
    stored_frames: int
    stored_outputs: int
    camera_timestamp_ms: Optional[float] = None
    skipped_camera_frames: Optional[int] = None


class RunLog:
    """Collects one model run's records, prints a line per request, and summarizes."""

    def __init__(self, model: str):
        self.model = model
        self.records: List[FrameRecord] = []
        self.loop_started: Optional[float] = None
        self.loop_ended: Optional[float] = None

    def add(self, kind: str, result, tracker, source_idx=None, camera_timestamp_ms=None,
            skipped=None) -> FrameRecord:
        memory = sample_memory()
        frames, outputs = tracker.storage_counts()
        record = FrameRecord(
            model=self.model, processed_idx=result.frame_idx, source_idx=source_idx, kind=kind,
            tracking_ms=result.elapsed_s * 1000, presence=result.presence,
            presence_logit=result.presence_logit, mask_area=result.area,
            cuda_allocated_mib=memory.cuda_allocated / MIB,
            cuda_reserved_mib=memory.cuda_reserved / MIB,
            cuda_peak_allocated_mib=memory.cuda_peak_allocated / MIB,
            cpu_rss_mib=memory.cpu_rss / MIB, stored_frames=frames, stored_outputs=outputs,
            camera_timestamp_ms=camera_timestamp_ms, skipped_camera_frames=skipped)
        self.records.append(record)
        src = "-" if source_idx is None else source_idx
        skip = "" if skipped is None else f" | skipped {skipped}"
        print(f"[{self.model}] {kind} proc {result.frame_idx} src {src} | {record.tracking_ms:6.1f} ms | "
              f"presence {result.presence:.3f} | area {result.area} px | {memory.line()} | "
              f"stored {frames}f/{outputs}o{skip}", flush=True)
        return record

    @property
    def tracked(self) -> List[FrameRecord]:
        return [r for r in self.records if r.kind == "track"]

    def write_frames_csv(self, path: Path) -> None:
        fields = list(FrameRecord.__dataclass_fields__)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for record in self.records:
                writer.writerow(asdict(record))

    def summary(self) -> dict:
        tracked = self.tracked
        ms = [r.tracking_ms for r in tracked]
        init = [r for r in self.records if r.kind == "init"]
        wall = (self.loop_ended - self.loop_started
                if self.loop_started is not None and self.loop_ended is not None else None)
        skipped = [r.skipped_camera_frames for r in tracked if r.skipped_camera_frames is not None]
        return {
            "model": self.model,
            "init_ms": init[0].tracking_ms if init else None,
            "timed_requests": len(ms),
            "mean_ms": statistics.fmean(ms) if ms else None,
            "median_ms": statistics.median(ms) if ms else None,
            "p90_ms": _percentile(ms, 90),
            "max_ms": max(ms) if ms else None,
            "tracking_loop_wall_s": wall,
            "processed_fps": len(ms) / wall if wall else None,
            # The first request's gap spans warm-up and prompting after the frozen init frame.
            "camera_frames_between_init_and_first_track": skipped[0] if skipped else None,
            "skipped_camera_frames_total": sum(skipped[1:]) if skipped else None,
            "empty_mask_requests": sum(1 for r in tracked if r.mask_area == 0),
            "peak_cuda_allocated_mib": max((r.cuda_peak_allocated_mib for r in self.records), default=None),
            "max_cuda_reserved_mib": max((r.cuda_reserved_mib for r in self.records), default=None),
            "first_track_memory": _memory_of(tracked[0]) if tracked else None,
            "last_track_memory": _memory_of(tracked[-1]) if tracked else None,
            "max_stored_frames": max((r.stored_frames for r in tracked), default=None),
            "max_stored_outputs": max((r.stored_outputs for r in tracked), default=None),
        }

    def print_report(self, summary: Optional[dict] = None) -> None:
        s = summary or self.summary()
        print(f"\n=== {self.model} ===")
        if not s["timed_requests"]:
            print("  no timed tracking requests")
            return
        print(f"  timed tracking requests: {s['timed_requests']}")
        print(f"  mean / median tracking:  {s['mean_ms']:.1f} / {s['median_ms']:.1f} ms "
              f"(p90 {s['p90_ms']:.1f}, max {s['max_ms']:.1f})")
        if s["processed_fps"] is not None:
            print(f"  processed FPS:           {s['processed_fps']:.2f} "
                  f"({s['timed_requests']} requests in {s['tracking_loop_wall_s']:.1f} s wall)")
        if s["skipped_camera_frames_total"] is not None:
            print(f"  skipped camera frames:   {s['skipped_camera_frames_total']} while tracking "
                  f"(+{s['camera_frames_between_init_and_first_track']} between the init frame and the first request)")
        print(f"  init request:            {s['init_ms']:.1f} ms (excluded from the statistics)")
        print(f"  empty-mask requests:     {s['empty_mask_requests']}")
        first, last = s["first_track_memory"], s["last_track_memory"]
        print(f"  cuda allocated first/last: {first['cuda_allocated_mib']:.0f} / {last['cuda_allocated_mib']:.0f} MiB, "
              f"peak {s['peak_cuda_allocated_mib']:.0f} MiB, reserved max {s['max_cuda_reserved_mib']:.0f} MiB")
        print(f"  cpu rss first/last:      {first['cpu_rss_mib']:.0f} / {last['cpu_rss_mib']:.0f} MiB")


def _memory_of(record: FrameRecord) -> dict:
    return {"processed_idx": record.processed_idx,
            "cuda_allocated_mib": record.cuda_allocated_mib,
            "cuda_reserved_mib": record.cuda_reserved_mib,
            "cuda_peak_allocated_mib": record.cuda_peak_allocated_mib,
            "cpu_rss_mib": record.cpu_rss_mib}


def _percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q / 100 * (len(ordered) - 1))))]


def software_versions() -> dict:
    import cv2
    import numpy
    import transformers

    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "transformers": transformers.__version__,
        "numpy": numpy.__version__,
        "opencv": cv2.__version__,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        versions["gpu"] = props.name
        versions["gpu_capability"] = f"{props.major}.{props.minor}"
        versions["gpu_memory_mib"] = props.total_memory / MIB
    if "pyrealsense2" in sys.modules:
        versions["pyrealsense2"] = getattr(sys.modules["pyrealsense2"], "__version__", "unknown")
    return versions


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")
