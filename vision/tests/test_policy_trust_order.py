"""The policy tick runs before recovery on its frame and uses the trust left by the previous
frame's after_recovery(), which takes that frame's own SAM verdict and verified pose.

    cd real_robot/vision && python -m unittest tests.test_policy_trust_order -v
"""

import io
import json
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np

VISION_DIR = Path(__file__).resolve().parents[1]
for path in (VISION_DIR, VISION_DIR.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from manipulation.policy.arm import DryRunPolicyArm  # noqa: E402
from manipulation.policy.jenga.config import JengaPolicyConfig  # noqa: E402
from manipulation.policy.policy_log import PolicyLog  # noqa: E402
from manipulation.policy.session import PolicySession  # noqa: E402
from pipeline.recovery.recovery_state import ObjectRecoveryState, Verdict  # noqa: E402


class TrustOrderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.log_path = Path(self.directory.name) / "policy_log.jsonl"
        self.state = ObjectRecoveryState(name="obj1", sam_id=1, iou_threshold=0.4, failure_limit=3)
        recovery = SimpleNamespace(disabled_reason=None, states=[self.state])
        cfg = JengaPolicyConfig.load().with_mode("shadow")
        self.session = PolicySession(cfg, np.eye(4), DryRunPolicyArm(), SimpleNamespace(info={}), recovery,
                                     PolicyLog(self.log_path), Path(self.directory.name), stats_every_s=1e9)
        # The object is resolved from a run folder in the real loop.
        self.session._index, self.session._rec_index, self.session._object_name = 0, 0, "obj1"
        self.tracker = SimpleNamespace(objects=[SimpleNamespace(center_pose=np.eye(4), to_origin=np.eye(4))])

    def tearDown(self):
        self.session.log.close()
        self.directory.cleanup()

    def _frame(self, frame_number, verdict=None):
        """One loop iteration: tick, then recovery (a SAM verdict on this frame if given), then
        after_recovery."""
        timing = SimpleNamespace(frame_number=frame_number, captured_at=time.monotonic(), camera_timestamp_ms=0.0)
        self.session.schedule.next_due = None  # every frame ticks
        with redirect_stdout(io.StringIO()):
            self.session.before_recovery(self.tracker, np.eye(3), timing)
            if verdict is not None:
                self.state.last_frame_number, self.state.last_verdict = frame_number, verdict
            self.session.after_recovery(self.tracker, np.eye(3), timing)

    def _logged_trust(self):
        self.session.log.close()
        lines = [json.loads(line) for line in self.log_path.read_text().splitlines()]
        return [(line["frame_number"], line["trust"]) for line in lines if "tick" in line]

    def test_tick_uses_the_previous_frames_trust(self):
        self._frame(10, Verdict.MATCH)
        self._frame(11)
        self._frame(12, Verdict.MISMATCH)  # this frame's tick acts before its verdict
        self._frame(13)
        self.assertEqual(self._logged_trust(), [(10, "untrusted"), (11, "verified"), (12, "unverified_ok"),
                                                (13, "untrusted")])

    def test_update_takes_the_verified_frames_pose(self):
        self._frame(10, Verdict.MATCH)
        self.assertEqual(self.session.manager.last_trusted_frame, 10)
        self._frame(11)
        self.assertEqual(self.session.manager.last_trusted_frame, 10)

    def test_reregistration_makes_the_next_tick_untrusted(self):
        self._frame(10, Verdict.MATCH)
        self._frame(11)
        self.state.registered(True, 12)
        self._frame(12)
        self._frame(13)
        self.assertEqual(self._logged_trust()[-2:], [(12, "unverified_ok"), (13, "untrusted")])


if __name__ == "__main__":
    unittest.main()
