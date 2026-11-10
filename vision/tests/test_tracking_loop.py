"""Video writing keeps every displayed frame, including ones whose status strip changed size;
the hooks run in order around recovery.

    cd real_robot/vision && python -m unittest tests.test_tracking_loop -v
"""

import io
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

VISION_DIR = Path(__file__).resolve().parents[1]
if str(VISION_DIR) not in sys.path:
    sys.path.insert(0, str(VISION_DIR))

from pipeline import tracking_loop  # noqa: E402
from pipeline.tracking_loop import track_live, write_video  # noqa: E402


class _Tracker:
    def __init__(self, calls):
        self.calls = calls

    def track(self, *_args):
        self.calls.append("fp")
        return {"obj1": 0.01}

    def draw(self, rgb, _K):
        return rgb


class _Stream:
    K = np.eye(3)

    def read(self):
        timing = SimpleNamespace(frame_number=1, camera_timestamp_ms=0.0, wait_s=0.0, prepare_s=0.0,
                                 captured_at=time.monotonic())
        return np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4), np.float32), timing


class _Recovery:
    def __init__(self, calls):
        self.calls = calls

    def start_loop(self):
        pass

    def after_fp(self, _frame, _display):
        self.calls.append("recovery")

    def draw(self, vis, _frame):
        return vis

    def report(self, *_args):
        pass

    def summary(self, *_args):
        pass


class _Hooks:
    def __init__(self, calls):
        self.calls = calls

    def on_mouse(self, *_args):
        pass

    def after_recovery(self, *_args):
        self.calls.append("after_recovery")

    def draw(self, vis):
        return vis

    def on_key(self, _key):
        return True  # quit after one iteration

    def close(self):
        pass


class _PolicyHooks(_Hooks):
    def before_recovery(self, *_args):
        self.calls.append("before_recovery")


class LoopOrderTests(unittest.TestCase):
    def _run(self, hooks_class):
        calls = []
        args = SimpleNamespace(save_video=False, log_variation=False, num_seconds=None, track_refine_iter=1,
                               stats_every=1e9, fps=60, save_path=None)
        with mock.patch.object(tracking_loop, "cv2") as fake_cv2, redirect_stdout(io.StringIO()):
            fake_cv2.waitKey.return_value = 0xFF
            track_live(args, _Tracker(calls), _Stream(), hooks=hooks_class(calls), recovery=_Recovery(calls))
        return calls

    def test_before_recovery_runs_between_foundationpose_and_recovery(self):
        self.assertEqual(self._run(_PolicyHooks), ["fp", "before_recovery", "recovery", "after_recovery"])

    def test_hooks_without_before_recovery_run_only_after_recovery(self):
        self.assertEqual(self._run(_Hooks), ["fp", "recovery", "after_recovery"])


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
