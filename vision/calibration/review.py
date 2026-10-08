"""Review saved samples offline (no hardware needed).

    python -m calibration.review --session data/sessions/NAME            # interactive viewer
    python -m calibration.review --session data/sessions/NAME --list     # table only
    python -m calibration.review --session data/sessions/NAME --redetect [--write]

Viewer keys:  n / SPACE next   p previous   x reject (move to rejected/)
              u restore (when viewing rejected)   t toggle accepted/rejected   q quit
--redetect re-runs detection on the raw images with the current config.yaml thresholds
(using the intrinsics saved with each sample); --write stores the new detection in
sample.json (the previous one is kept under detection_history).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from .board import BoardConfig, board_metadata, load_config, make_board, make_detector, same_board
from .detection import Detection, detect, draw_detection
from .diversity import analyze, summary_lines
from .session import Session, load_image, load_sample, sample_E_C, write_json

WINDOW = "review samples"


def sample_K_dist(sample):
    return np.asarray(sample["intrinsics"]["K"]), np.asarray(sample["intrinsics"]["dist"])


def detection_from_json(d: dict) -> Detection:
    det = Detection(np.asarray(d["charuco_corners_px"]).reshape(-1, 2), np.asarray(d["charuco_ids"], np.int32),
                    np.asarray(d["marker_corners_px"]).reshape(-1, 4, 2), np.asarray(d["marker_ids"], np.int32),
                    metrics=d["metrics"], accepted=d["accepted"], reasons=d["reject_reasons"])
    if d["rvec_camera_board"] is not None:
        det.rvec, det.tvec = np.asarray(d["rvec_camera_board"]), np.asarray(d["tvec_camera_board_m"])
    return det


def redetect(sample, config) -> Detection:
    cfg = BoardConfig.from_dict(config["board"])
    board = make_board(cfg)
    K, dist = sample_K_dist(sample)
    return detect(load_image(sample), make_detector(board), board, cfg, K, dist, config["quality"])


def row(sample, det_json=None) -> str:
    d = det_json or sample["detection"]
    m = d["metrics"]
    rb = sample["robot"]
    return (f"#{sample['id']:>3}  corners {m.get('num_corners', 0):>2}  rms {m.get('reproj_rms_px', float('nan')):5.2f}px  "
            f"area {m.get('hull_fraction', 0) * 100:4.1f}%  sharp {m.get('sharpness', 0):6.0f}  "
            f"dist {m.get('distance_m', float('nan')):.2f}m  tilt {m.get('tilt_deg', float('nan')):4.0f}  "
            f"inplane {m.get('inplane_deg', float('nan')):+5.0f}  dq {rb['joint_delta_rad']:.4f}  "
            f"{'ok' if d['accepted'] else 'REJECT ' + '; '.join(d['reject_reasons'])}")


def print_table(session: Session, config):
    samples = [load_sample(d) for d in session.sample_dirs()]
    print(f"{session.root}: {len(samples)} samples, {len(session.sample_dirs(rejected=True))} rejected")
    for s in samples:
        print("  " + row(s))
    if samples:
        Es, Cs = zip(*(sample_E_C(s) for s in samples))
        res = analyze(list(Es), list(Cs), [s["detection"]["metrics"] for s in samples], config["diversity"],
                      ids=[s["id"] for s in samples])
        for line in summary_lines(res):
            print("  " + line)
        for w in res["warnings"]:
            print("  ! " + w)


def run_redetect(session: Session, config, write: bool):
    cfg = BoardConfig.from_dict(config["board"])
    if not same_board(session.meta["board"], board_metadata(cfg)):
        raise SystemExit("config.yaml board differs from the session's board definition")
    for d in session.sample_dirs():
        s = load_sample(d)
        det = redetect(s, config)
        new = det.to_json()
        old_T, new_T = s["detection"]["T_camera_board"], new["T_camera_board"]
        dt = float("nan") if old_T is None or new_T is None else \
            1000 * np.linalg.norm(np.asarray(old_T)[:3, 3] - np.asarray(new_T)[:3, 3])
        print(f"  {row(s, new)}  |dt| vs saved {dt:.2f} mm")
        if write:
            s.setdefault("detection_history", []).append(s["detection"])
            s["detection"] = new
            s["redetected_with_opencv"] = cv2.__version__
            write_json(d / "sample.json", {k: v for k, v in s.items() if k != "_dir"})
    if write:
        print("Updated sample.json files. Samples that are now rejected are excluded by calibrate.")


def viewer(session: Session):
    show_rejected = False
    idx = 0
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    while True:
        dirs = session.sample_dirs(rejected=show_rejected)
        if not dirs:
            print("no " + ("rejected" if show_rejected else "accepted") + " samples")
            if show_rejected:
                show_rejected = False
                continue
            break
        idx = int(np.clip(idx, 0, len(dirs) - 1))
        s = load_sample(dirs[idx])
        K, dist = sample_K_dist(s)
        vis = draw_detection(load_image(s), detection_from_json(s["detection"]), K, dist)
        label = f"{'REJECTED ' if show_rejected else ''}{idx + 1}/{len(dirs)}  " + row(s)
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 34), (30, 30, 30), -1)
        cv2.putText(vis, label[:150], (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW, vis)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord("n"), ord(" ")):
            idx += 1
        elif key == ord("p"):
            idx -= 1
        elif key == ord("t"):
            show_rejected, idx = not show_rejected, 0
        elif key == ord("x") and not show_rejected:
            print(f"rejected #{s['id']} -> {session.reject(dirs[idx])}")
        elif key == ord("u") and show_rejected:
            print(f"restored #{s['id']} -> {session.restore(dirs[idx])}")
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--list", action="store_true")
    p.add_argument("--redetect", action="store_true")
    p.add_argument("--write", action="store_true")
    args = p.parse_args()

    config = load_config(args.config)
    session = Session(args.session)
    if args.redetect:
        run_redetect(session, config, args.write)
    elif args.list:
        print_table(session, config)
    else:
        print_table(session, config)
        viewer(session)


if __name__ == "__main__":
    main()
