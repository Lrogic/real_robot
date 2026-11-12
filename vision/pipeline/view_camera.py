"""Live D405 colour feed and nothing else. Environment: vision. Run from vision/.

    /opt/conda-envs/vision/bin/python -m pipeline.view_camera            # q or Esc quits
    /opt/conda-envs/vision/bin/python -m pipeline.view_camera --depth    # also print the depth at the crosshair

Uses the tracking pipeline's stream (640x480 @ 30 fps, depth aligned to colour). The camera
can be open in one process only, so stop the tracking pipeline first.
"""

import argparse

import cv2
import numpy as np

from .camera import CameraSettings, RealSenseStream

WINDOW = "camera (q quits)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--depth", action="store_true",
                        help="draw a crosshair at the image centre with the depth image's value there")
    parser.add_argument("--undistort", action="store_true", help="undistort the frames as --undistort tracking does")
    args = parser.parse_args()

    with RealSenseStream(CameraSettings(undistort=args.undistort)) as stream:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        while True:
            rgb, depth, _ = stream.read()
            view = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if args.depth:
                h, w = depth.shape
                patch = depth[h // 2 - 2:h // 2 + 3, w // 2 - 2:w // 2 + 3]
                valid = patch[patch > 0]
                text = f"{np.median(valid):.3f} m" if len(valid) else "no depth"
                cv2.drawMarker(view, (w // 2, h // 2), (0, 255, 0), cv2.MARKER_CROSS, 16, 1)
                cv2.putText(view, text, (w // 2 + 12, h // 2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
