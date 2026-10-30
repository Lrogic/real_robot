"""Video writing keeps every displayed frame, including ones whose status strip changed size.

    cd real_robot/vision && python -m unittest tests.test_tracking_loop -v
"""

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

VISION_DIR = Path(__file__).resolve().parents[1]
if str(VISION_DIR) not in sys.path:
    sys.path.insert(0, str(VISION_DIR))

from pipeline.tracking_loop import write_video  # noqa: E402


class WriteVideoTests(unittest.TestCase):
    def test_variable_height_frames_are_all_stored(self):
        short = np.full((40, 64, 3), 10, dtype=np.uint8)
        tall = np.full((51, 64, 3), 200, dtype=np.uint8)
        frames = [short, tall, short]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracking.mp4"
            write_video(frames, path, elapsed_seconds=1.0)
            capture = cv2.VideoCapture(str(path))
            try:
                self.assertTrue(capture.isOpened())
                stored = []
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    stored.append(frame)
            finally:
                capture.release()
        self.assertEqual(len(stored), 3)
        self.assertTrue(all(frame.shape == (52, 64, 3) for frame in stored))
        # mp4v is lossy; check the short frames kept their image and gained padding,
        # and the tall frame was kept rather than dropped.
        self.assertLess(abs(float(stored[0][:32].mean()) - 10), 4)
        self.assertLess(abs(float(stored[0][46:].mean()) - 20), 8)
        self.assertGreater(float(stored[1][:48].mean()), 150)
        self.assertLess(abs(float(stored[2][:32].mean()) - 10), 4)


if __name__ == "__main__":
    unittest.main()
