"""Render the configured ChArUco board at exact physical scale for printing.

    python -m calibration.generate_board --out data/board.png [--dpi 600]

Print at 100% / "actual size" (no fit-to-page), then measure: the pattern must be
exactly squares_x*square_m by squares_y*square_m (140 x 100 mm by default).
A blank tab is left on one side for the gripper jaws, plus a 100 mm scale bar.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from .board import DEFAULT_DATA_DIR, BoardConfig, board_metadata, load_config, make_board

MM_PER_INCH = 25.4


def render(cfg: BoardConfig, dpi: int, margin_mm: float, tab_mm: float) -> np.ndarray:
    px_per_mm = dpi / MM_PER_INCH
    square_px = cfg.square_m * 1000 * px_per_mm
    if abs(square_px - round(square_px)) > 1e-6:
        raise ValueError(f"square of {cfg.square_m * 1000} mm is {square_px:.3f} px at {dpi} dpi; "
                         "choose a dpi giving an integer pixel count (e.g. 254, 508, 1016)")
    square_px = int(round(square_px))
    pattern = make_board(cfg).generateImage((cfg.squares_x * square_px, cfg.squares_y * square_px),
                                            marginSize=0, borderBits=1)
    margin = int(round(margin_mm * px_per_mm))
    tab = int(round(tab_mm * px_per_mm))
    page = cv2.copyMakeBorder(pattern, margin, margin + tab, margin, margin, cv2.BORDER_CONSTANT, value=255)

    text_y = margin + pattern.shape[0] + int(4 * px_per_mm)
    label = (f"{cfg.squares_x}x{cfg.squares_y} sq {cfg.square_m * 1000:g}mm marker {cfg.marker_m * 1000:g}mm "
             f"{cfg.dictionary} ids {cfg.first_marker_id}-{cfg.marker_ids[-1]}"
             f"{' legacy' if cfg.legacy_pattern else ''}")
    scale = 0.1 * px_per_mm  # Hershey glyphs are ~22 px tall at scale 1, so text is ~2 mm
    cv2.putText(page, label, (margin, text_y), cv2.FONT_HERSHEY_SIMPLEX, scale, 0, max(1, int(scale * 2)))
    bar_y = text_y + int(3 * px_per_mm)
    bar_len = int(round(100 * px_per_mm))
    cv2.line(page, (margin, bar_y), (margin + bar_len, bar_y), 0, max(1, int(0.3 * px_per_mm)))
    cv2.putText(page, "100 mm", (margin + bar_len + int(2 * px_per_mm), bar_y + int(1 * px_per_mm)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, 0, max(1, int(scale * 2)))
    cv2.putText(page, "gripper tab (unprinted area below)", (margin, bar_y + int(6 * px_per_mm)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, 128, max(1, int(scale * 2)))
    return page


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--out", type=Path, default=DEFAULT_DATA_DIR / "board.png")
    p.add_argument("--dpi", type=int, default=1016)
    p.add_argument("--margin-mm", type=float, default=10.0)
    p.add_argument("--tab-mm", type=float, default=30.0)
    args = p.parse_args()

    cfg = BoardConfig.from_dict(load_config(args.config)["board"])
    page = render(cfg, args.dpi, args.margin_mm, args.tab_mm)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        Image.fromarray(page).save(args.out, dpi=(args.dpi, args.dpi))
    except ImportError:
        cv2.imwrite(str(args.out), page)
        print("PIL not available: PNG has no DPI tag; set the print size manually")
    meta = board_metadata(cfg)
    w_mm, h_mm = meta["pattern_size_mm"]
    print(f"Wrote {args.out} ({page.shape[1]}x{page.shape[0]} px at {args.dpi} dpi)")
    print(f"Pattern {w_mm} x {h_mm} mm, {meta['num_markers']} markers, {meta['num_charuco_corners']} corners, "
          f"OpenCV {meta['opencv_version']}")
    print("Print at 100% scale, glue flat to cardboard, then run verify_board.")


if __name__ == "__main__":
    main()
