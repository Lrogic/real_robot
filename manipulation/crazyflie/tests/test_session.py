"""Crazyflie session logic with a fake tracker and link (no radio, camera or GPU).

    cd /home/luka/research/real_robot && /opt/conda-envs/foundationpose/bin/python -m unittest manipulation.crazyflie.tests.test_session -v
"""

import dataclasses
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REAL_ROBOT = Path(__file__).resolve().parents[3]
for path in (REAL_ROBOT, REAL_ROBOT / "vision"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from manipulation.crazyflie.session import (CrazyflieConfig, CrazyflieSession, closest_copy,  # noqa: E402
                                            mesh_yaw_deg, wrap_deg)
from manipulation.policy.trust import Trust  # noqa: E402


class FakeLink:
    lost = None
    vbat = 4.1

    def __init__(self):
        self.sent, self.resets = [], 0
        self.yaw, self.initial_yaw_rad = 0.0, 0.0

    def attitude(self):
        return np.array([0.0, 0.0, self.yaw]), 0.01

    def set_initial_yaw(self, yaw_rad):
        self.initial_yaw_rad = yaw_rad

    def send(self, p):
        self.sent.append(np.array(p))

    def estimate(self):
        return np.array([0.1, 0.2, 0.3]), 0.01

    def status(self):
        return {"var": (1e-5, 1e-5, 1e-5), "vbat": 4.1, "supervisor": 526, "age": 0.05}

    def reset_estimator(self):
        self.resets += 1
        self.yaw = float(np.degrees(self.initial_yaw_rad))


def translation(x, y, z):
    T = np.eye(4)
    T[:3, 3] = (x, y, z)
    return T


def rot_z90():
    T = np.eye(4)
    T[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    return T


class SessionFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = dataclasses.replace(CrazyflieConfig.load(), tracked_name="obj1", reset_after_sends=3)
        self.X = translation(0.5, 0.0, 0.2) @ rot_z90()
        self.obj = SimpleNamespace(name="obj1", center_pose=translation(0.0, 0.0, 0.4),
                                   to_origin=translation(0.01, 0.0, 0.0))
        self.tracker = SimpleNamespace(objects=[self.obj])
        self.link = FakeLink()

    def tearDown(self):
        self.tmp.cleanup()

    def session(self, cfg=None, recovery=None):
        s = CrazyflieSession(self.link, cfg or self.cfg, self.X, recovery, Path(self.tmp.name))
        s.start()
        return s

    def tick(self, s, frame=1, age=0.02):
        import time
        timing = SimpleNamespace(frame_number=frame, captured_at=time.monotonic() - age)
        s.before_recovery(self.tracker, np.eye(3), timing)
        s.after_recovery(self.tracker, np.eye(3), timing)


class SessionTest(SessionFixture):
    def test_sends_base_frame_position(self):
        s = self.session()
        self.tick(s)
        expected = (self.X @ self.obj.center_pose @ self.obj.to_origin)[:3, 3]
        np.testing.assert_allclose(self.link.sent[0], expected)
        np.testing.assert_allclose(expected, [0.5, 0.01, 0.6])

    def test_offset_is_in_the_mesh_frame(self):
        self.obj.center_pose = translation(0, 0, 0.4) @ rot_z90()
        s = self.session(dataclasses.replace(self.cfg, origin_offset_m=(0.0, 0.0, 0.005)))
        self.tick(s)
        T = self.X @ self.obj.center_pose @ self.obj.to_origin
        np.testing.assert_allclose(self.link.sent[0], T[:3, 3] + T[:3, 2] * 0.005)

    def test_holds(self):
        s = self.session()
        self.tick(s, age=1.0)
        self.assertEqual(self.link.sent, [])
        self.assertIn("old", s._status)
        s.on_key(ord("p"))
        self.tick(s)
        self.assertEqual((self.link.sent, s._status), ([], "holding: paused"))
        s.on_key(ord("p"))
        self.obj.center_pose = None
        self.tick(s)
        self.assertEqual((self.link.sent, s._status), ([], "holding: no pose"))
        s.close()
        records = [json.loads(line) for line in (Path(self.tmp.name) / "crazyflie_log.jsonl").read_text().splitlines()]
        self.assertEqual(sum(1 for r in records if "hold_reason" in r and r["hold_reason"]), 3)

    def test_untrusted_pose_is_held_with_recovery(self):
        recovery = SimpleNamespace(states=[SimpleNamespace(name="obj1")], disabled_reason=None)
        s = self.session(recovery=recovery)
        s.after_recovery = lambda *a: None  # trust is set by hand below
        self.tick(s)
        self.assertEqual(self.link.sent, [])
        s.trust.state = Trust.VERIFIED
        self.tick(s, frame=2)
        self.assertEqual(len(self.link.sent), 1)
        s.trust.state = Trust.UNTRUSTED
        self.tick(s, frame=3)
        self.assertEqual(len(self.link.sent), 1)
        s2 = self.session(dataclasses.replace(self.cfg, send_untrusted=True), recovery=recovery)
        s2.after_recovery = lambda *a: None
        self.tick(s2)
        self.assertEqual(len(self.link.sent), 2)

    def test_auto_reset_once_and_key_reset(self):
        s = self.session()
        for frame in range(6):
            self.tick(s, frame=frame)
        self.assertEqual(self.link.resets, 1)
        s.on_key(ord("e"))
        self.assertEqual(self.link.resets, 2)
        self.assertTrue(s.on_key(ord("q")))

    def test_draw(self):
        s = self.session()
        self.tick(s)
        s._K = np.array([[600.0, 0, 320], [0, 600, 240], [0, 0, 1]])
        out = s.draw(np.zeros((480, 640, 3), np.uint8))
        self.assertEqual(out.shape, (480, 640, 3))


def rot_z(deg):
    T = np.eye(4)
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return T


class HeadingTest(SessionFixture):
    def set_fp_yaw(self, mesh_deg):
        """Mesh rotated mesh_deg about the camera z; X adds 90 deg, so base heading is mesh_deg + 90."""
        self.obj.center_pose = translation(0.0, 0.0, 0.4) @ rot_z(mesh_deg)

    def test_helpers(self):
        self.assertAlmostEqual(closest_copy(30.0, 118.0), 120.0)
        self.assertAlmostEqual(closest_copy(210.0, 118.0), 120.0)
        self.assertAlmostEqual(closest_copy(-170.0, 175.0), -170.0)
        self.assertAlmostEqual(closest_copy(100.0, -170.0), -170.0)
        self.assertAlmostEqual(wrap_deg(190.0), -170.0)
        self.assertAlmostEqual(mesh_yaw_deg(rot_z(30)[:3, :3]), 30.0)
        self.assertIsNone(mesh_yaw_deg(np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)))

    def test_proposal_turn_and_confirm(self):
        s = self.session()
        self.set_fp_yaw(10.0)
        self.tick(s)
        self.assertAlmostEqual(s._heading, 100.0)
        s.on_key(ord("h"))
        self.assertAlmostEqual(s._heading, -170.0)
        self.set_fp_yaw(10.0 + 90.0 + 2.0)  # FoundationPose flips a copy and the drone turns 2 deg
        self.tick(s, frame=2)
        self.assertAlmostEqual(s._heading, -168.0)
        resets = self.link.resets
        s.on_key(ord("c"))
        self.assertTrue(s.heading_confirmed)
        self.assertAlmostEqual(self.link.initial_yaw_rad, np.radians(-168.0))
        self.assertEqual(self.link.resets, resets + 1)
        self.set_fp_yaw(10.0 + 180.0 + 2.0 + 1.0)  # another flip, 1 deg of real turn
        self.link.yaw = -167.5
        self.tick(s, frame=3)
        self.assertAlmostEqual(s._heading, -167.0)
        self.assertAlmostEqual(s._heading_error, 0.5)
        self.assertIsNone(s.heading_refusal())

    def test_refusals_and_reset_uses_heading(self):
        s = self.session()
        self.set_fp_yaw(0.0)
        self.tick(s)
        self.assertIn("confirm", s.heading_refusal())
        s.on_key(ord("c"))
        self.link.yaw = 90.0 + 30.0  # drone thinks it turned 30 deg; FoundationPose disagrees
        self.tick(s, frame=2)
        self.assertIn("deg from the drone", s.heading_refusal())
        self.link.yaw = 91.0
        self.tick(s, frame=3)
        s.on_key(ord("e"))
        self.assertAlmostEqual(self.link.initial_yaw_rad, np.radians(90.0))
        s.on_key(ord("h"))
        self.assertFalse(s.heading_confirmed)
        self.assertIn("confirm", s.heading_refusal())

    def test_takeoff_needs_confirmed_front(self):
        s = CrazyflieSession(self.link, self.cfg, self.X, None, Path(self.tmp.name), fly=True)
        self.set_fp_yaw(0.0)
        self.tick(s)
        s.on_key(ord("t"))
        self.assertIn("confirm the drone's front", s.message)
        self.assertEqual(s.flight.state, "grounded")
        s._log.close()

    def test_log_has_heading(self):
        s = self.session()
        self.set_fp_yaw(0.0)
        self.tick(s)
        s.close()
        rows = [json.loads(line) for line in (Path(self.tmp.name) / "crazyflie_log.jsonl").read_text().splitlines()]
        frame = [r for r in rows if "frame_number" in r and "p_base" in r][-1]
        self.assertAlmostEqual(frame["fp_yaw_raw_deg"], 90.0)
        self.assertEqual(len(frame["q_base_object"]), 4)
        self.assertEqual(frame["drone_rpy_deg"], [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
