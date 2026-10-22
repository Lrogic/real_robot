"""Recovery logic, CLI resolution, and frame association, without GPU, camera, or weights.

    cd real_robot/vision && python -m unittest tests.test_recovery -v
"""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

VISION_DIR = Path(__file__).resolve().parents[1]
if str(VISION_DIR) not in sys.path:
    sys.path.insert(0, str(VISION_DIR))

from experiments.segmentation.tracker import MultiTrackResult  # noqa: E402
from fp_capture.cli import DEFAULT_SAM3_CHECKPOINT, parse_pipeline_args, pipeline_parser  # noqa: E402
from fp_capture.recovery import LiveFrame, SamRecovery  # noqa: E402
from fp_capture.recovery_config import load_recovery_config  # noqa: E402
from fp_capture.recovery_state import ObjectRecoveryState, RateSchedule, Verdict, mask_iou  # noqa: E402
from fp_capture.run_folder import SavedObject, sam_object_id  # noqa: E402
from fp_capture.tracking import RegistrationError  # noqa: E402

H, W = 48, 64


def parse(*argv):
    with contextlib.redirect_stdout(io.StringIO()):
        return parse_pipeline_args(pipeline_parser("test"), ["--save-path", "/tmp/x", *argv])


def parse_error(*argv) -> str:
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
        try:
            parse(*argv)
        except SystemExit:
            return err.getvalue()
    raise AssertionError(f"{argv} parsed without error")


def box(x0, y0, x1, y1):
    mask = np.zeros((H, W), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


class CliTest(unittest.TestCase):
    def test_without_recovery_defaults_unchanged(self):
        args = parse()
        self.assertEqual((args.width, args.height, args.fps), (640, 480, 30))
        self.assertFalse(args.undistort)
        self.assertEqual(args.checkpoint, DEFAULT_SAM3_CHECKPOINT)
        self.assertEqual(args.fp_mode, "sequential")
        self.assertEqual((args.est_refine_iter, args.track_refine_iter), (5, 2))
        self.assertIsNone(args.recovery_cfg)

    def test_recovery_defaults(self):
        args = parse("--recovery")
        cfg = args.recovery_cfg
        self.assertEqual((args.width, args.height, args.fps), (640, 480, 60))
        self.assertFalse(args.undistort)
        self.assertEqual(args.fp_mode, "sequential")
        self.assertEqual((cfg.sam_model, cfg.sam_fps, cfg.iou_threshold, cfg.failure_limit),
                         ("sam3", 2.0, 0.4, 3))

    def test_recovery_cli_overrides(self):
        args = parse("--recovery", "--fps", "30", "--sam-model", "sam2-base-plus", "--sam-fps", "5",
                     "--iou-threshold", "0.6", "--failure-limit", "2", "--sam-hf-cache", "/tmp/hf",
                     "--undistort", "--fp-mode", "batched-render")
        cfg = args.recovery_cfg
        self.assertEqual(args.fps, 30)
        self.assertTrue(args.undistort)
        self.assertEqual(args.fp_mode, "batched-render")
        self.assertEqual((cfg.sam_model, cfg.sam_fps, cfg.iou_threshold, cfg.failure_limit, str(cfg.hf_cache)),
                         ("sam2-base-plus", 5.0, 0.6, 2, "/tmp/hf"))

    def test_recovery_config_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
            f.write("camera: {width: 848, height: 480, fps: 90}\n"
                    "sam: {model: sam2-small, tracking_fps: 4, hf_cache: /tmp/hf, warmup_steps: 0}\n"
                    "recovery: {iou_threshold: 0.5, failure_limit: 5}\n")
            f.flush()
            args = parse("--recovery", "--recovery-config", f.name, "--failure-limit", "4")
        self.assertEqual((args.width, args.fps), (848, 90))
        self.assertEqual((args.recovery_cfg.sam_model, args.recovery_cfg.failure_limit), ("sam2-small", 4))

    def test_recovery_only_flags_need_recovery(self):
        for flags in (["--sam-model", "sam3"], ["--sam-fps", "2"], ["--iou-threshold", "0.5"],
                      ["--failure-limit", "2"], ["--recovery-config", "x.yaml"]):
            self.assertIn("requires --recovery", parse_error(*flags))

    def test_setup_checkpoint_not_hf_model(self):
        self.assertIn("--checkpoint", parse_error("--recovery", "--checkpoint", "/tmp/sam3.pt"))
        self.assertEqual(parse("--checkpoint", "/tmp/sam3.pt").checkpoint, Path("/tmp/sam3.pt"))

    def test_invalid_values(self):
        self.assertIn("exceed camera fps", parse_error("--recovery", "--fps", "30", "--sam-fps", "31"))
        self.assertIn("IoU", parse_error("--recovery", "--iou-threshold", "1.5"))
        self.assertIn("failure limit", parse_error("--recovery", "--failure-limit", "0"))
        self.assertIn("invalid choice", parse_error("--recovery", "--sam-model", "sam9"))

    def test_shared_forwarded_arguments(self):
        # run.sh forwards one flat list (setup, tracking, manipulation flags) to every stage.
        args = parse("--recovery", "--meshes-dir", "/tmp", "--num-seconds", "10", "--save-video",
                     "--fp-mode", "batched", "--reset", "--manipulation", "--manipulation-dry-run")
        self.assertTrue(args.recovery and args.reset and args.manipulation)
        self.assertEqual(args.fp_mode, "batched")

    def test_bundled_config_loads(self):
        cfg = load_recovery_config(None)
        self.assertEqual((cfg.camera_width, cfg.camera_height, cfg.camera_fps), (640, 480, 60))


class IouTest(unittest.TestCase):
    def test_values(self):
        a, b = box(0, 0, 10, 10), box(5, 0, 15, 10)
        self.assertAlmostEqual(mask_iou(a, b), 50 / 150)
        self.assertEqual(mask_iou(a, a), 1.0)
        self.assertEqual(mask_iou(a, np.zeros_like(a)), 0.0)  # nonempty SAM vs empty FP

    def test_rejects_non_bool_or_shape(self):
        with self.assertRaises(ValueError):
            mask_iou(box(0, 0, 2, 2).astype(np.uint8) * 255, box(0, 0, 2, 2))
        with self.assertRaises(ValueError):
            mask_iou(box(0, 0, 2, 2), np.zeros((H + 1, W), dtype=bool))


class StateTest(unittest.TestCase):
    def setUp(self):
        self.state = ObjectRecoveryState("obj1", 1, iou_threshold=0.4, failure_limit=3)
        self.sam = box(0, 0, 10, 10)
        self.bad = box(30, 30, 40, 40)

    def observe(self, sam, fp):
        return self.state.observe(sam, fp, 0.9, 0, 0.0)

    def test_trigger_after_consecutive_mismatches(self):
        self.assertEqual(self.observe(self.sam, self.bad), Verdict.MISMATCH)
        self.assertEqual(self.observe(self.sam, self.bad), Verdict.MISMATCH)
        self.assertEqual(self.observe(self.sam, self.bad), Verdict.TRIGGER)
        self.assertEqual(self.state.mismatch_count, 3)

    def test_good_iou_resets(self):
        self.observe(self.sam, self.bad)
        self.observe(self.sam, self.bad)
        self.assertEqual(self.observe(self.sam, self.sam), Verdict.MATCH)
        self.assertEqual(self.state.mismatch_count, 0)
        self.assertEqual(self.observe(self.sam, self.bad), Verdict.MISMATCH)

    def test_threshold_is_inclusive(self):
        fp = box(0, 0, 10, 10)
        fp[:, :] = False
        fp[0:10, 0:4] = True  # IoU 40/100
        self.assertEqual(self.observe(self.sam, fp), Verdict.MATCH)

    def test_empty_sam_resets_and_skips(self):
        self.observe(self.sam, self.bad)
        self.observe(self.sam, self.bad)
        self.assertEqual(self.state.observe(np.zeros((H, W), bool), None, 0.01, 0, 0.0), Verdict.SAM_EMPTY)
        self.assertTrue(self.state.sam_empty)
        self.assertIsNone(self.state.last_iou)
        self.assertEqual(self.state.mismatch_count, 0)
        self.assertEqual(self.observe(self.sam, self.bad), Verdict.MISMATCH)
        self.assertFalse(self.state.sam_empty)

    def test_nonempty_sam_needs_silhouette(self):
        with self.assertRaises(ValueError):
            self.observe(self.sam, None)


class ScheduleTest(unittest.TestCase):
    def test_rate_and_no_catch_up(self):
        s = RateSchedule(2.0, start=0.0)
        self.assertTrue(s.due(0.0))
        self.assertFalse(s.due(0.4))
        self.assertTrue(s.due(0.5))
        # A 2 s stall: one request, then a full period before the next.
        self.assertTrue(s.due(2.5))
        self.assertFalse(s.due(2.6))
        self.assertFalse(s.due(2.99))
        self.assertTrue(s.due(3.0))

    def test_never_exceeds_rate(self):
        s = RateSchedule(2.0, start=0.0)
        times = np.arange(0, 10, 0.013)
        fired = [t for t in times if s.due(t)]
        self.assertLessEqual(len(fired), 20)
        self.assertTrue(all(b - a >= 0.5 - 1e-9 for a, b in zip(fired, fired[1:])))


class FakeTracked:
    def __init__(self, name):
        self.name = name
        self.color = (0, 255, 0)


class FakeFpTracker:
    """Records calls; silhouettes come from `self.silhouette[name]`."""

    def __init__(self, names):
        self.objects = [FakeTracked(n) for n in names]
        self.silhouette = {}
        self.render_calls, self.register_calls = [], []
        self.fail_register = set()
        self.events = None

    def render_silhouettes(self, indices, K, height, width):
        self.render_calls.append((list(indices), K, (height, width)))
        if self.events is not None:
            self.events.append(("render", list(indices)))
        return np.stack([self.silhouette[self.objects[i].name] for i in indices]) if indices \
            else np.zeros((0, height, width), bool)

    def register_one(self, index, rgb, depth, K, mask, iterations):
        self.register_calls.append({"name": self.objects[index].name, "rgb": rgb, "depth": depth,
                                    "K": K, "mask": mask, "iterations": iterations})
        if self.events is not None:
            self.events.append(("register", self.objects[index].name))
        if self.objects[index].name in self.fail_register:
            raise RegistrationError("forced")

    def draw(self, rgb, K):
        return rgb.copy()


class FakeSam:
    """step_objects returns the next scripted {sam_id: mask}; records the RGB it was given."""

    def __init__(self, obj_ids, script):
        self.obj_ids = tuple(obj_ids)
        self.script = list(script)
        self.seen_rgb = []
        self.idx = 0
        self.resets = 0
        self.events = None

    def step_objects(self, rgb):
        self.seen_rgb.append(rgb)
        if self.events is not None:
            self.events.append(("sam", None))
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        self.idx += 1
        masks = np.stack([step[i] for i in self.obj_ids])
        return MultiTrackResult(self.idx, self.obj_ids, masks, np.full(len(self.obj_ids), 3.0), 0.01)

    def start_objects(self, *_):
        self.resets += 1

    def close(self):
        self.resets += 1

    def storage_counts(self):
        return 1, 15


class Clock:
    def __init__(self):
        self.t = 0.0


class RecoveryLoopTest(unittest.TestCase):
    """SamRecovery.after_fp with fake FP/SAM trackers and a controlled monotonic clock."""

    def make(self, script, names=("obj1", "obj3"), limit=3, sam_fps=2.0):
        from fp_capture import recovery as recovery_module

        self.clock = Clock()
        self._real_monotonic = recovery_module.time.monotonic
        recovery_module.time.monotonic = lambda: self.clock.t
        self.addCleanup(setattr, recovery_module.time, "monotonic", self._real_monotonic)
        cfg = load_recovery_config(None, sam_fps=sam_fps, failure_limit=limit)
        self.fp = FakeFpTracker(names)
        objects = [SavedObject(n, box(0, 0, 5, 5), Path(f"/m/{n}.obj")) for n in names]
        self.sam = FakeSam([sam_object_id(n) for n in names], script)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        with contextlib.redirect_stdout(io.StringIO()):
            rec = SamRecovery(cfg, self.fp, objects, 7, Path(self.tmp.name) / "log.jsonl", {}, sam=self.sam)
        self.addCleanup(rec._log.close)
        rec.start_loop()
        self.displayed = []
        return rec

    def frame(self, number, depth_value=0.5):
        rgb = np.full((H, W, 3), number % 255, np.uint8)
        depth = np.full((H, W), depth_value, np.float32)
        return LiveFrame(rgb, depth, np.eye(3) * number, number, number * 16.7, self.clock.t)

    def step(self, rec, frame, dt=0.5):
        with contextlib.redirect_stdout(io.StringIO()):
            ran = rec.after_fp(frame, self.displayed.append)
            rec.draw(frame.rgb, frame)
        self.clock.t += dt
        return ran

    def test_object_ids_follow_folder_names(self):
        rec = self.make([])
        self.assertEqual([(s.name, s.sam_id) for s in rec.states], [("obj1", 1), ("obj3", 3)])

    def test_same_frame_association_and_trigger(self):
        good, bad, sam = box(0, 0, 10, 10), box(30, 30, 40, 40), box(0, 0, 10, 10)
        rec = self.make([{1: sam, 3: sam}] * 3)
        self.fp.silhouette = {"obj1": bad, "obj3": good}
        frames = [self.frame(n) for n in (100, 131, 162)]
        for f in frames:
            self.assertTrue(self.step(rec, f))
        # SAM saw exactly each frame's RGB; silhouettes used that frame's K and size.
        self.assertEqual([id(r) for r in self.sam.seen_rgb], [id(f.rgb) for f in frames])
        self.assertEqual([c[1][0, 0] for c in self.fp.render_calls], [100, 131, 162])
        self.assertEqual(self.fp.render_calls[0][2], (H, W))
        # obj1 triggered on the third observation, registered from that frame and mask only.
        self.assertEqual(len(self.fp.register_calls), 1)
        call = self.fp.register_calls[0]
        self.assertEqual(call["name"], "obj1")
        self.assertIs(call["rgb"], frames[2].rgb)
        self.assertIs(call["depth"], frames[2].depth)
        self.assertIs(call["K"], frames[2].K)
        self.assertTrue(np.array_equal(call["mask"], sam) and call["mask"].dtype == bool)
        self.assertEqual(call["iterations"], 7)
        self.assertEqual(rec.states[0].mismatch_count, 0)
        self.assertEqual(rec.states[0].registrations, 1)
        self.assertEqual(rec.states[1].mismatch_count, 0)
        self.assertEqual(self.sam.resets, 0)  # session survives re-registration
        self.assertEqual(len(self.displayed), 1)  # "registering" frame shown before blocking

    def test_fp_only_frames_do_not_count(self):
        sam, bad = box(0, 0, 10, 10), box(30, 30, 40, 40)
        rec = self.make([{1: sam}] * 2, names=("obj1",))
        self.fp.silhouette = {"obj1": bad}
        self.assertTrue(self.step(rec, self.frame(1), dt=0.1))
        for n in range(2, 6):  # not due yet: cached masks are not re-evaluated
            self.assertFalse(self.step(rec, self.frame(n), dt=0.1))
        self.assertEqual(rec.states[0].mismatch_count, 1)
        self.assertEqual(rec.states[0].observations, 1)
        self.assertTrue(self.step(rec, self.frame(6)))
        self.assertEqual(rec.states[0].mismatch_count, 2)
        self.assertEqual(len(self.fp.render_calls), 2)

    def test_empty_mask_resets_and_skips_render(self):
        sam, bad, empty = box(0, 0, 10, 10), box(30, 30, 40, 40), np.zeros((H, W), bool)
        rec = self.make([{1: sam}, {1: sam}, {1: empty}, {1: sam}, {1: sam}], names=("obj1",))
        self.fp.silhouette = {"obj1": bad}
        for n in range(5):
            self.step(rec, self.frame(n))
        self.assertEqual(self.fp.render_calls[2][0], [])  # nothing rendered for the empty observation
        self.assertEqual(self.fp.register_calls, [])
        self.assertEqual(rec.states[0].mismatch_count, 2)
        self.assertEqual(rec.states[0].empty_observations, 1)

    def test_sam_exception_is_not_an_empty_mask(self):
        sam, bad = box(0, 0, 10, 10), box(30, 30, 40, 40)
        rec = self.make([{1: sam}, {1: sam}, RuntimeError("boom"), {1: sam}], names=("obj1",))
        self.fp.silhouette = {"obj1": bad}
        with contextlib.redirect_stderr(io.StringIO()):
            for n in range(4):
                self.step(rec, self.frame(n))
        self.assertIsNotNone(rec.disabled_reason)
        self.assertEqual(rec.states[0].mismatch_count, 2)  # neither reset nor incremented
        self.assertEqual(rec.states[0].empty_observations, 0)
        self.assertEqual(len(self.sam.seen_rgb), 3)  # no further SAM requests once disabled

    def test_multiple_triggers_register_sequentially(self):
        sam1, sam3, bad = box(0, 0, 10, 10), box(20, 20, 30, 30), box(40, 40, 50, 50)
        rec = self.make([{1: sam1, 3: sam3}], limit=1)
        self.fp.silhouette = {"obj1": bad, "obj3": bad}
        order = []
        self.fp.events = self.sam.events = order
        f = self.frame(9)
        self.step(rec, f)
        self.assertEqual([c["name"] for c in self.fp.register_calls], ["obj1", "obj3"])
        self.assertTrue(np.array_equal(self.fp.register_calls[0]["mask"], sam1))
        self.assertTrue(np.array_equal(self.fp.register_calls[1]["mask"], sam3))
        self.assertTrue(all(c["depth"] is f.depth for c in self.fp.register_calls))
        # No SAM (or render) request between or during the registrations.
        self.assertEqual(order, [("sam", None), ("render", [0, 1]), ("register", "obj1"), ("register", "obj3")])

    def test_failed_registration_reported_and_counter_cleared(self):
        sam, bad = box(0, 0, 10, 10), box(30, 30, 40, 40)
        rec = self.make([{1: sam}], names=("obj1",), limit=1)
        self.fp.silhouette = {"obj1": bad}
        self.fp.fail_register = {"obj1"}
        self.step(rec, self.frame(1))
        self.assertEqual((rec.states[0].registrations, rec.states[0].failed_registrations), (0, 1))
        self.assertEqual(rec.states[0].mismatch_count, 0)
        self.assertFalse(rec.registrations[0]["ok"])

    def test_unusable_depth_skips_registration(self):
        sam, bad = box(0, 0, 10, 10), box(30, 30, 40, 40)
        rec = self.make([{1: sam}], names=("obj1",), limit=1)
        self.fp.silhouette = {"obj1": bad}
        self.step(rec, self.frame(1, depth_value=0.0))
        self.assertEqual(self.fp.register_calls, [])
        self.assertEqual(rec.states[0].failed_registrations, 1)

    def test_objects_independent(self):
        sam, bad, good = box(0, 0, 10, 10), box(30, 30, 40, 40), box(0, 0, 10, 10)
        rec = self.make([{1: sam, 3: sam}] * 2)
        self.fp.silhouette = {"obj1": bad, "obj3": good}
        self.step(rec, self.frame(1))
        self.fp.silhouette = {"obj1": good, "obj3": bad}
        self.step(rec, self.frame(2))
        self.assertEqual([s.mismatch_count for s in rec.states], [0, 1])


if __name__ == "__main__":
    unittest.main()
