"""Check that an existing ChArUco print matches config.yaml before collecting data.

    python -m calibration.verify_board                 # live D405 preview; SPACE to check, q to quit
    python -m calibration.verify_board --image shot.png

Hold the whole board in view, flat and well lit. Checks:
  * marker IDs are exactly the configured set (no foreign IDs) under the configured dictionary
  * every ChArUco corner interpolates (layout / legacy pattern / dictionary consistent)
  * marker-to-square size ratio measured in the image matches marker_m / square_m
  * operator ruler measurements of the square size and overall pattern size
Writes data/board_verification.json, which collect.py requires.
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .board import DEFAULT_DATA_DIR, BoardConfig, board_metadata, load_config, make_board, make_detector, \
    make_dictionary, same_board
from .detection import Detection, detect_raw, draw_detection, quiet_opencv

SURVEY_DICTS = ["DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000", "DICT_5X5_50",
                "DICT_6X6_50", "DICT_ARUCO_ORIGINAL", "DICT_APRILTAG_36h11"]
WINDOW = "verify board (SPACE check, q quit)"
MIN_MARKER_PX_FOR_RATIO = 40


def survey_dictionaries(gray) -> dict:
    out = {}
    for name in SURVEY_DICTS:
        det = cv2.aruco.ArucoDetector(make_dictionary(name), cv2.aruco.DetectorParameters())
        _, ids, _ = det.detectMarkers(gray)
        out[name] = [] if ids is None else sorted(int(i) for i in np.ravel(ids))
    return out


def marker_square_ratio(cfg: BoardConfig, board, corners, ids, marker_corners, K, dist):
    """Marker side / square side measured on the board plane via a corner homography."""
    def undistort(p):
        p = p.reshape(-1, 1, 2).astype(np.float64)
        return p.reshape(-1, 2) if K is None else cv2.undistortPoints(p, K, dist, P=K).reshape(-1, 2)

    obj = np.asarray(board.getChessboardCorners(), dtype=np.float64)[ids, :2]
    H, _ = cv2.findHomography(undistort(corners), obj)
    if H is None:
        return None, 0.0
    sides = []
    for quad in marker_corners:
        q = cv2.perspectiveTransform(undistort(quad).reshape(-1, 1, 2), H).reshape(4, 2)
        sides += [np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)]
    side_px = float(np.mean([np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for quad in marker_corners for i in range(4)]))
    return float(np.mean(sides) / cfg.square_m), side_px


def evaluate(image_bgr, cfg: BoardConfig, K=None, dist=None) -> dict:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    board = make_board(cfg)
    cc, ci, mc, mi = detect_raw(gray, make_detector(board))
    alt_cfg = replace(cfg, legacy_pattern=not cfg.legacy_pattern)
    _, alt_ci, _, _ = detect_raw(gray, make_detector(make_board(alt_cfg)))

    expected = set(cfg.marker_ids.tolist())
    survey = survey_dictionaries(gray)
    seen = set(survey[cfg.dictionary])
    foreign = sorted(seen - expected)
    ratio, marker_px = marker_square_ratio(cfg, board, cc, ci, mc, K, dist) if len(ci) >= 4 and len(mc) \
        else (None, 0.0)
    expected_ratio = cfg.marker_m / cfg.square_m
    # A pixel of edge blur on a small marker biases the ratio by several percent.
    ratio_reliable = marker_px >= MIN_MARKER_PX_FOR_RATIO

    checks = {
        "all_markers_found": sorted(set(mi.tolist())) == sorted(expected),
        "no_foreign_ids": not foreign,
        "all_corners_found": len(ci) == cfg.num_corners,
        "legacy_setting_best": len(ci) >= len(alt_ci),
        "marker_ratio_ok": ratio is not None and ratio_reliable and abs(ratio / expected_ratio - 1) < 0.05,
    }
    return {
        "checks": checks,
        "detected_marker_ids": sorted(mi.tolist()),
        "missing_marker_ids": sorted(expected - set(mi.tolist())),
        "foreign_marker_ids": foreign,
        "num_corners": int(len(ci)),
        "num_corners_with_legacy_flipped": int(len(alt_ci)),
        "marker_square_ratio": ratio,
        "expected_marker_square_ratio": expected_ratio,
        "marker_side_px": marker_px,
        "marker_ratio_reliable": ratio_reliable,
        "dictionary_survey": survey,
        "_vis": draw_detection(image_bgr, Detection(cc, ci, mc, mi), K, dist),
    }


def print_report(r: dict, cfg: BoardConfig):
    print("\n--- automatic checks ---")
    for k, v in r["checks"].items():
        print(f"  [{'OK' if v else 'FAIL'}] {k}")
    print(f"  markers {r['detected_marker_ids']} missing {r['missing_marker_ids']} foreign {r['foreign_marker_ids']}")
    print(f"  corners {r['num_corners']}/{cfg.num_corners} (legacy flipped: {r['num_corners_with_legacy_flipped']})")
    if r["marker_square_ratio"] is not None:
        print(f"  marker/square ratio {r['marker_square_ratio']:.3f} (expected {r['expected_marker_square_ratio']:.3f}),"
              f" markers {r['marker_side_px']:.0f} px wide")
        if not r["marker_ratio_reliable"]:
            print(f"  ratio not trusted below {MIN_MARKER_PX_FOR_RATIO} px per marker: hold the board closer "
                  "so it fills much of the image")
    hits = {k: len(v) for k, v in r["dictionary_survey"].items() if v}
    print(f"  markers found per dictionary: {hits}")
    if cfg.dictionary.startswith("DICT_4X4"):
        print("  (DICT_4X4_50/100/250/1000 share their first 50 markers, so IDs 0-16 look identical)")


def ask_mm(prompt, given):
    if given is not None:
        return given
    while True:
        s = input(prompt).strip()
        try:
            return float(s)
        except ValueError:
            print("enter a number in mm")


def measurement_checks(cfg: BoardConfig, args) -> dict:
    w_exp, h_exp = (v * 1000 for v in cfg.pattern_size_m)
    print(f"\n--- ruler checks (expected square {cfg.square_m * 1000:g} mm, pattern {w_exp:g} x {h_exp:g} mm "
          "excluding margins and tab) ---")
    print("Tip: measure across all squares and divide, rather than one square.")
    sq = ask_mm("Measured square side [mm]: ", args.square_mm)
    w = ask_mm(f"Measured pattern width ({cfg.squares_x} squares) [mm]: ", args.width_mm)
    h = ask_mm(f"Measured pattern height ({cfg.squares_y} squares) [mm]: ", args.height_mm)
    tol = args.tolerance
    res = {"square_mm": sq, "width_mm": w, "height_mm": h, "tolerance": tol,
           "square_ok": abs(sq / (cfg.square_m * 1000) - 1) <= tol,
           "width_ok": abs(w / w_exp - 1) <= tol, "height_ok": abs(h / h_exp - 1) <= tol}
    for k in ("square_ok", "width_ok", "height_ok"):
        print(f"  [{'OK' if res[k] else 'FAIL'}] {k}")
    return res


def previous_ruler(path: Path, board_meta: dict):
    """Ruler measurements from an earlier verification of the same board definition, if any."""
    if not path.exists():
        return None
    with open(path) as f:
        prev = json.load(f)
    ruler = prev.get("ruler") or {}
    if not same_board(prev.get("board", {}), board_meta):
        return None
    if any(ruler.get(k) is None for k in ("square_mm", "width_mm", "height_mm")):
        return None
    return ruler


def live_capture(cam_cfg: dict, cfg: BoardConfig):
    from .camera import ColorStream
    with ColorStream(cam_cfg) as stream:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        board = make_board(cfg)
        detector = make_detector(board)
        while True:
            f = stream.read()
            gray = cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY)
            cc, ci, mc, mi = detect_raw(gray, detector)
            vis = draw_detection(f.image, Detection(cc, ci, mc, mi), stream.K, stream.dist)
            cv2.putText(vis, f"markers {len(mi)}/{cfg.num_markers} corners {len(ci)}/{cfg.num_corners}  "
                        "SPACE: check  q: quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(WINDOW, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                cv2.destroyWindow(WINDOW)
                return f.image, stream.K, stream.dist, stream.info
            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                return None, None, None, None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--image", type=Path, help="verify from a saved image instead of the live camera")
    p.add_argument("--out", type=Path, default=DEFAULT_DATA_DIR / "board_verification.json")
    p.add_argument("--square-mm", type=float)
    p.add_argument("--width-mm", type=float)
    p.add_argument("--height-mm", type=float)
    p.add_argument("--remeasure", action="store_true",
                   help="ask for ruler measurements even if a previous verification of this board has them")
    p.add_argument("--tolerance", type=float, default=0.01, help="relative tolerance for ruler checks")
    args = p.parse_args()
    quiet_opencv()

    config = load_config(args.config)
    cfg = BoardConfig.from_dict(config["board"])
    meta = board_metadata(cfg)
    print(f"Configured board: {cfg.squares_x}x{cfg.squares_y} squares, square {cfg.square_m * 1000:g} mm, "
          f"marker {cfg.marker_m * 1000:g} mm, {cfg.dictionary}, IDs {cfg.marker_ids[0]}-{cfg.marker_ids[-1]}, "
          f"pattern {meta['pattern_size_mm'][0]:g} x {meta['pattern_size_mm'][1]:g} mm, legacy={cfg.legacy_pattern}")

    if args.image:
        image, K, dist, cam_info = cv2.imread(str(args.image)), None, None, None
        if image is None:
            raise SystemExit(f"Cannot read {args.image}")
    else:
        image, K, dist, cam_info = live_capture(config["camera"], cfg)
        if image is None:
            raise SystemExit("Aborted")

    r = evaluate(image, cfg, K, dist)
    vis = r.pop("_vis")
    print_report(r, cfg)
    previous = previous_ruler(args.out, meta) if not args.remeasure else None
    if previous and all(getattr(args, k) is None for k in ("square_mm", "width_mm", "height_mm")):
        args.square_mm, args.width_mm, args.height_mm = (previous[k] for k in ("square_mm", "width_mm", "height_mm"))
        print(f"\nReusing ruler measurements from {args.out} (pass --remeasure to enter new ones)")
    ruler = measurement_checks(cfg, args)
    passed = all(r["checks"].values()) and all(v for k, v in ruler.items() if k.endswith("_ok"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    stem = args.out.with_suffix("")
    cv2.imwrite(f"{stem}_image.png", image)
    cv2.imwrite(f"{stem}_overlay.png", vis)
    with open(args.out, "w") as f:
        json.dump({"passed": passed, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "board": meta,
                   "automatic": r, "ruler": ruler, "camera": cam_info,
                   "source_image": str(args.image) if args.image else f"{stem}_image.png"}, f, indent=2)
    print(f"\n{'PASSED' if passed else 'FAILED'}: wrote {args.out}")
    if not passed:
        print("Fix config.yaml (dictionary, IDs, legacy_pattern, sizes) or reprint with generate_board.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
