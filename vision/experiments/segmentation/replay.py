"""Track one object through a recorded video with HF SAM trackers, one streamed frame at a time.

    python -m experiments.segmentation.replay --video rec/video.mkv --model sam2-small
    python -m experiments.segmentation.replay --video rec/video.mkv --model all --tracking-fps 10

The init mask is <video>.init_mask.png; if missing it is created interactively on the first
decoded frame with --mask-model, then reused unchanged by every model. With --model all the
models run sequentially, each in its own process, and the panel video is built afterwards.
"""

import argparse
import json
import multiprocessing
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .init_mask import InitMaskError, create_init_mask, load_init_mask, mask_paths
from .metrics import MEMORY_DEFINITION, TIMING_DEFINITION, write_json
from .models import MODELS
from .render import annotate, side_by_side
from .video_io import ANNOTATED_FOURCC, FrameSelector, iter_frames, open_writer, video_info
from .worker import PREDICTIONS_FILE, SUMMARY_FILE, worker_entry

RUNS_DIR = Path(__file__).resolve().parent / "runs"
PLAYBACK_POLICY = (
    "Output video plays at the source FPS with one output frame per source frame. Each panel "
    "shows the newest prediction whose source index is <= the displayed frame; frames that "
    "were not processed are labelled 'held from src N' instead of 'inferred'.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", required=True, choices=[*MODELS, "all"])
    parser.add_argument("--tracking-fps", type=float, default=None,
                        help="Deterministic subsampling rate (default: every source frame)")
    parser.add_argument("--mask-model", choices=list(MODELS), default="sam3",
                        help="Model used only to create a missing init mask (default: sam3)")
    parser.add_argument("--recreate-mask", action="store_true",
                        help="Replace an existing (e.g. mismatched) init mask interactively")
    parser.add_argument("--warmup-steps", type=int, default=5,
                        help="Tracking requests on a disposable warm-up session (default: 5)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help=f"Run folder (default: {RUNS_DIR}/replay_<video>_<time>)")
    parser.add_argument("--no-video", action="store_true", help="Skip writing the annotated video")
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error(f"--video {args.video} does not exist")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be >= 0")
    return args


def ensure_init_mask(video: Path, mask_model: str, recreate: bool, ctx) -> tuple:
    png, _ = mask_paths(video)
    if recreate or not png.is_file():
        print(f"Creating init mask for {video} with {mask_model}...")
        # Own process, so this process never touches CUDA and every tracker starts clean.
        process = ctx.Process(target=create_init_mask, args=(video, mask_model))
        process.start()
        process.join()
        if process.exitcode != 0:
            raise SystemExit(f"Init mask creation failed (exit code {process.exitcode})")
    try:
        return load_init_mask(video)
    except FileNotFoundError:
        raise SystemExit("No init mask was saved; nothing to track")
    except InitMaskError as error:
        raise SystemExit(str(error))


def load_predictions(path: Path) -> dict:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def render_video(video: Path, fps: float, predictions: dict, out_path: Path) -> None:
    writer = None
    cursor = {model: 0 for model in predictions}
    try:
        for idx, rgb in iter_frames(video):
            panels = []
            for model, pred in predictions.items():
                sources = pred["source_idx"]
                j = cursor[model]
                while j + 1 < len(sources) and sources[j + 1] <= idx:
                    j += 1
                cursor[model] = j
                shape = tuple(pred["mask_shape"])
                mask = np.unpackbits(pred["masks_packed"][j], count=shape[0] * shape[1]).reshape(shape).astype(bool)
                presence = float(1 / (1 + np.exp(-pred["presence_logit"][j])))
                if idx == 0:
                    status = "src 0 | init (mask prompt; presence set from mask)"
                elif sources[j] == idx:
                    status = f"src {idx} | inferred (proc {pred['processed_idx'][j]})"
                else:
                    status = f"src {idx} | held from src {sources[j]}"
                panels.append(annotate(rgb, mask, model, presence, status))
            frame = side_by_side(panels)
            if writer is None:
                writer = open_writer(out_path, ANNOTATED_FOURCC, fps, (frame.shape[1], frame.shape[0]))
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        if writer is not None:
            writer.release()
    print(f"Wrote {out_path}")


def main() -> int:
    args = parse_args()
    ctx = multiprocessing.get_context("spawn")
    info = video_info(args.video)
    try:
        selector = FrameSelector(info.fps, args.tracking_fps)
    except ValueError as error:
        raise SystemExit(str(error))
    mask, mask_meta = ensure_init_mask(args.video, args.mask_model, args.recreate_mask, ctx)

    models = list(MODELS) if args.model == "all" else [args.model]
    out_dir = args.output_dir or RUNS_DIR / f"replay_{args.video.stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=False)
    print(f"Run folder {out_dir}; models {models} (sequential, one process each)")

    timeline = []
    for key in models:
        started = time.time()
        process = ctx.Process(target=worker_entry,
                              args=(key, args.video, mask, args.tracking_fps, out_dir / key, args.warmup_steps))
        process.start()
        process.join()
        timeline.append({"model": key, "pid": process.pid, "exitcode": process.exitcode,
                         "launched": started, "joined": time.time()})
        if process.exitcode != 0:
            print(f"{key} failed (exit code {process.exitcode}); continuing with the remaining models")

    finished = [t["model"] for t in timeline if t["exitcode"] == 0]
    summaries = {key: json.loads((out_dir / key / SUMMARY_FILE).read_text()) for key in finished}
    predictions = {key: load_predictions(out_dir / key / PREDICTIONS_FILE) for key in finished}
    reference = next(iter(predictions.values()))["source_idx"] if predictions else None
    same_frames = all(np.array_equal(p["source_idx"], reference) for p in predictions.values())
    sequential = all(a["joined"] <= b["launched"] for a, b in zip(timeline, timeline[1:]))
    if not same_frames:
        print("ERROR: models processed different source frames; refusing to build a comparison panel")

    video_path = None
    if predictions and same_frames and not args.no_video:
        name = "comparison.mp4" if args.model == "all" else f"annotated_{models[0]}.mp4"
        video_path = out_dir / name
        render_video(args.video, info.fps, predictions, video_path)

    write_json(out_dir / "summary.json", {
        "mode": "recorded-video",
        "video": str(args.video.resolve()),
        "video_info": vars(info),
        "init_mask": {"path": str(mask_paths(args.video)[0]), "meta": mask_meta},
        "models": models,
        "frame_selection": selector.describe(),
        "processed_source_frames": int(len(reference)) if reference is not None else 0,
        "identical_frame_indices_across_models": same_frames,
        "sequential_workers": sequential,
        "worker_timeline": timeline,
        "playback_policy": PLAYBACK_POLICY,
        "output_video": str(video_path) if video_path else None,
        "timing_definition": TIMING_DEFINITION,
        "memory_definition": MEMORY_DEFINITION,
        "per_model": summaries,
    })
    print(f"\nSummary: {out_dir / 'summary.json'}")
    for key, summary in summaries.items():
        print(f"  {key:10s} requests {summary['timed_requests']:5d}  mean {summary['mean_ms']:.1f} ms  "
              f"median {summary['median_ms']:.1f} ms  processed {summary['processed_fps']:.1f} FPS  "
              f"peak cuda {summary['peak_cuda_allocated_mib']:.0f} MiB")
    return 0 if len(finished) == len(models) and same_frames else 1


if __name__ == "__main__":
    sys.exit(main())
