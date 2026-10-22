"""Camera-free checks on a synthetic video (GPU and cached checkpoints required).

    python -m experiments.segmentation.validate [--frames 300] [--sustained-frames 3000]

1. Pruned vs unpruned sessions give identical masks and presence logits (each model).
2. A stale/mismatched init mask is rejected.
3. replay --model all (the real CLI) on a video whose target leaves the view and returns:
   sequential workers, identical frame indices, tracking continues through empty masks,
   bounded session storage. Repeated with --tracking-fps 10.
4. Optional sustained single-model replay for memory behaviour over a long run.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .init_mask import InitMaskError, load_init_mask, mask_paths, save_init_mask
from .models import MODELS
from .video_io import SOURCE_FOURCC, first_frame, iter_frames, open_writer

RUNS_DIR = Path(__file__).resolve().parent / "runs"
W, H, FPS, RADIUS = 640, 480, 30, 50
VISION_DIR = Path(__file__).resolve().parents[2]
SETTLED_IDX = 20


def target_center(i: int):
    """Target path, period 300 frames: right across, fully out of view (~frames 105-180), back."""
    t = i % 300
    if t < 120:
        x = 150 + 5 * t            # 150 -> 745, leaves the image (x - r > 640) at t ~ 108
    elif t < 180:
        x = 745
    else:
        x = 745 - 5 * (t - 180)    # back in view from t ~ 191, to 145
    return int(x), int(240 + 60 * np.sin(i / 25))


DISTRACTOR = ((60, 340), (160, 440))


def distractor_mask() -> np.ndarray:
    mask = np.zeros((H, W), np.uint8)
    cv2.rectangle(mask, *DISTRACTOR, 1, -1)
    return mask > 0


def synthetic_frame(i: int, background: np.ndarray):
    frame = background.copy()
    cv2.rectangle(frame, *DISTRACTOR, (40, 80, 220), -1)                  # static distractor
    mask = np.zeros((H, W), np.uint8)
    cv2.circle(mask, target_center(i), RADIUS, 1, -1)
    stripes = np.zeros_like(frame)
    stripes[:] = (230, 60, 40)
    stripes[(np.arange(W)[None, :] // 10 % 2 == 0).repeat(H, 0)] = (250, 210, 40)
    frame[mask > 0] = stripes[mask > 0]
    return frame, mask > 0


def write_video(path: Path, frames: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    background = cv2.GaussianBlur(rng.integers(0, 255, (H, W, 3), dtype=np.uint8), (0, 0), 6)
    background = cv2.normalize(background, None, 60, 190, cv2.NORM_MINMAX)
    writer = open_writer(path, SOURCE_FOURCC, FPS, (W, H))
    for i in range(frames):
        rgb, mask = synthetic_frame(i, background)
        if i == 0:
            first_mask = mask
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    writer.release()
    save_init_mask(path, first_mask, first_frame(path), {"key": "synthetic-ground-truth"})
    return first_mask


def check_pruning(video: Path, mask: np.ndarray, frames: int) -> dict:
    import torch
    from .models import load_model
    from .tracker import HfVideoTracker

    results = {}
    for key in MODELS:
        loaded = load_model(key)
        runs = {}
        for name, prune in (("unpruned_a", False), ("unpruned_b", False), ("pruned", True)):
            tracker = HfVideoTracker(key, prune=prune, loaded=loaded)
            masks, logits, stored = [], [], []
            for idx, rgb in iter_frames(video):
                if idx >= frames:
                    break
                result = tracker.start(rgb, mask) if idx == 0 else tracker.step(rgb)
                masks.append(result.mask)
                logits.append(result.presence_logit)
                stored.append(tracker.storage_counts())
            tracker.close()
            runs[name] = (np.stack(masks), np.asarray(logits), stored)
        control_equal = (np.array_equal(runs["unpruned_a"][0], runs["unpruned_b"][0])
                         and np.array_equal(runs["unpruned_a"][1], runs["unpruned_b"][1]))
        a, p = runs["unpruned_a"], runs["pruned"]
        results[key] = {
            "frames": frames,
            "unpruned_runs_identical": control_equal,
            "pruned_masks_identical": bool(np.array_equal(a[0], p[0])),
            "pruned_logits_identical": bool(np.array_equal(a[1], p[1])),
            "max_abs_logit_diff": float(np.abs(a[1] - p[1]).max()),
            "mask_pixels_differing": int((a[0] != p[0]).sum()),
            "unpruned_final_storage": a[2][-1],
            "pruned_final_storage": p[2][-1],
        }
        print(f"[pruning] {key}: {results[key]}", flush=True)
        results[f"{key}_multi_object"] = check_pruning_multi(loaded, video, mask, frames, a[0])
        print(f"[pruning] {key} multi-object: {results[f'{key}_multi_object']}", flush=True)
        del loaded
        torch.cuda.empty_cache()
    return results


def check_pruning_multi(loaded, video: Path, target_mask: np.ndarray, frames: int,
                        single_target_masks: np.ndarray) -> dict:
    """Two objects (moving target id 3, static distractor id 7) in one shared session."""
    from .tracker import HfVideoTracker

    init = {3: target_mask, 7: distractor_mask()}
    runs = {}
    for name, prune in (("unpruned_a", False), ("unpruned_b", False), ("pruned", True)):
        tracker = HfVideoTracker(loaded.spec.key, prune=prune, loaded=loaded)
        masks, logits, stored, ids = [], [], [], set()
        for idx, rgb in iter_frames(video):
            if idx >= frames:
                break
            result = tracker.start_objects(rgb, init) if idx == 0 else tracker.step_objects(rgb)
            ids.add(result.obj_ids)
            masks.append(result.masks)
            logits.append(result.presence_logits)
            stored.append(tracker.storage_counts())
        tracker.close()
        runs[name] = (np.stack(masks), np.stack(logits), stored, ids)
    a, b, p = runs["unpruned_a"], runs["unpruned_b"], runs["pruned"]
    target = a[0][:, 0]
    union = np.logical_or(target, single_target_masks).sum(axis=(1, 2))
    inter = np.logical_and(target, single_target_masks).sum(axis=(1, 2))
    iou_vs_single = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    return {
        "frames": frames,
        "object_ids_stable": a[3] == p[3] == {(3, 7)},
        "unpruned_runs_identical": bool(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])),
        "pruned_masks_identical": bool(np.array_equal(a[0], p[0])),
        "pruned_logits_identical": bool(np.array_equal(a[1], p[1])),
        "max_abs_logit_diff": float(np.abs(a[1] - p[1]).max()),
        "unpruned_final_storage": a[2][-1],
        "pruned_final_storage": p[2][-1],
        # Informational: the shared session batches memory encoding, so the target's masks need
        # not be bit-identical to its one-object session, only close.
        "target_min_iou_vs_single_object_session": float(iou_vs_single.min()),
    }


def check_mismatch(tmp: Path, good_video: Path) -> dict:
    other = tmp / "other.mkv"
    write_video(other, 5, seed=1)
    for src, dst in zip(mask_paths(good_video), mask_paths(other)):
        shutil.copy(src, dst)
    try:
        load_init_mask(other)
        rejected, message = False, ""
    except InitMaskError as error:
        rejected, message = True, str(error).splitlines()[0]
    load_init_mask(good_video)  # the matching mask still loads
    print(f"[mismatch] rejected={rejected}: {message}")
    return {"rejected": rejected, "message": message}


def run_replay(video: Path, out_dir: Path, model: str, extra=()) -> dict:
    cmd = [sys.executable, "-m", "experiments.segmentation.replay", "--video", str(video),
           "--model", model, "--output-dir", str(out_dir), *extra]
    print("[replay] " + " ".join(cmd), flush=True)
    log_path = out_dir.with_suffix(".log")
    with open(log_path, "w") as log:
        code = subprocess.run(cmd, cwd=VISION_DIR, stdout=log, stderr=subprocess.STDOUT).returncode
    summary = json.loads((out_dir / "summary.json").read_text())
    return {"exit_code": code, "log": str(log_path), "dir": out_dir, "summary": summary}


def analyse_replay(run: dict, empty_expected: bool) -> dict:
    s = run["summary"]
    timeline = s["worker_timeline"]
    out = {"exit_code": run["exit_code"],
           "sequential_workers": s["sequential_workers"],
           "distinct_worker_pids": len({t["pid"] for t in timeline}) == len(timeline),
           "identical_frame_indices": s["identical_frame_indices_across_models"],
           "processed_source_frames": s["processed_source_frames"],
           "output_video": s["output_video"], "models": {}}
    for key, m in s["per_model"].items():
        rows = list(_read_csv(run["dir"] / key / "frames.csv"))
        track = [r for r in rows if r["kind"] == "track"]
        empty = [int(r["source_idx"]) for r in track if int(r["mask_area"]) == 0]
        after_first_empty = [r for r in track if empty and int(r["source_idx"]) > empty[0]]
        recovered = [int(r["source_idx"]) for r in after_first_empty if int(r["mask_area"]) > 0]
        # Memory once the 15-frame memory window is full, to separate fill-up from growth.
        settled = [r for r in track if int(r["processed_idx"]) >= SETTLED_IDX]
        alloc = [float(r["cuda_allocated_mib"]) for r in settled]
        rss = [float(r["cpu_rss_mib"]) for r in settled]
        out["models"][key] = {
            "timed_requests": m["timed_requests"], "mean_ms": m["mean_ms"], "median_ms": m["median_ms"],
            "processed_fps": m["processed_fps"], "init_ms": m["init_ms"],
            "warmup": m["warmup"], "empty_mask_requests": m["empty_mask_requests"],
            "first_empty_src": empty[0] if empty else None, "last_empty_src": empty[-1] if empty else None,
            "requests_after_first_empty": len(after_first_empty),
            "every_selected_frame_tracked": m["timed_requests"] == s["processed_source_frames"] - 1,
            "first_nonempty_after_empty_src": recovered[0] if recovered else None,
            f"settled_from_proc_{SETTLED_IDX}": {
                "cuda_allocated_mib_first_last_min_max": [alloc[0], alloc[-1], min(alloc), max(alloc)] if alloc else None,
                "cpu_rss_mib_first_last": [rss[0], rss[-1]] if rss else None,
                "cuda_allocated_slope_mib_per_1000_requests":
                    1000 * float(np.polyfit(np.arange(len(alloc)), alloc, 1)[0]) if len(alloc) > 1 else None,
                "cpu_rss_slope_mib_per_1000_requests":
                    1000 * float(np.polyfit(np.arange(len(rss)), rss, 1)[0]) if len(rss) > 1 else None,
            },
            "min_presence_while_empty": min((float(r["presence"]) for r in track if int(r["mask_area"]) == 0), default=None),
            "max_stored_frames": m["max_stored_frames"], "max_stored_outputs": m["max_stored_outputs"],
            "first_track_memory": m["first_track_memory"], "last_track_memory": m["last_track_memory"],
            "peak_cuda_allocated_mib": m["peak_cuda_allocated_mib"],
            "init_mask_sha256": m["init_mask_sha256"],
        }
    shas = {m["init_mask_sha256"] for m in out["models"].values()}
    out["same_init_mask_for_all_models"] = len(shas) == 1
    if empty_expected:
        out["tracking_continued_through_empty_masks"] = all(
            m["empty_mask_requests"] > 0 and m["requests_after_first_empty"] > 0
            and m["every_selected_frame_tracked"] for m in out["models"].values())
    return out


def _read_csv(path: Path):
    import csv
    with open(path) as f:
        yield from csv.DictReader(f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, default=300, help="Synthetic video length (default: 300)")
    parser.add_argument("--pruning-frames", type=int, default=80)
    parser.add_argument("--sustained-frames", type=int, default=0,
                        help="If > 0, also replay a video this long with --sustained-model")
    parser.add_argument("--sustained-model", choices=[*MODELS, "all"], default="sam3")
    parser.add_argument("--skip-pruning", action="store_true")
    args = parser.parse_args()

    out = RUNS_DIR / f"validation_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    video = out / "synthetic.mkv"
    mask = write_video(video, args.frames)
    report = {"video": str(video), "frames": args.frames}
    if not args.skip_pruning:
        report["pruning"] = check_pruning(video, mask, args.pruning_frames)
    report["mismatch"] = check_mismatch(out, video)
    report["replay_all"] = analyse_replay(run_replay(video, out / "replay_all", "all"), True)
    report["replay_all_10fps"] = analyse_replay(
        run_replay(video, out / "replay_all_10fps", "all", ["--tracking-fps", "10"]), True)
    if args.sustained_frames > 0:
        long_video = out / "sustained.mkv"
        write_video(long_video, args.sustained_frames)
        report["sustained"] = analyse_replay(
            run_replay(long_video, out / "replay_sustained", args.sustained_model, ["--no-video"]), True)
    (out / "validation.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))
    print(f"\nValidation report: {out / 'validation.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
