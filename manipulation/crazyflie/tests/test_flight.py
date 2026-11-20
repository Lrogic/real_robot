"""Flight state machine with a fake link and a hand-stepped clock (no radio).

    cd /home/luka/research/real_robot && /opt/conda-envs/foundationpose/bin/python -m unittest manipulation.crazyflie.tests.test_flight -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np

REAL_ROBOT = Path(__file__).resolve().parents[3]
if str(REAL_ROBOT) not in sys.path:
    sys.path.insert(0, str(REAL_ROBOT))

from manipulation.crazyflie.flight import (ARMING, GROUNDED, HOVER, KILLED, LANDING,  # noqa: E402
                                           TAKEOFF, FlightConfig, FlightController, supervisor_problem)

CFG = FlightConfig(hover_height_m=0.25, takeoff_time_s=2.0, land_time_s=2.0, bounds_xy_m=0.2,
                   max_height_m=0.45, pose_timeout_s=0.3, max_estimate_error_m=0.04,
                   max_position_variance_m2=4e-4, min_battery_v=3.8, min_send_hz=15, watchdog_period_s=0.1,
                   max_heading_error_deg=10)
GROUND = np.array([0.4, 0.0, 0.01])


class FakeLink:
    def __init__(self):
        self.p = GROUND.copy()
        self.vbat, self.var = 4.1, (1e-5, 1e-5, 1e-5)
        self.supervisor = 526  # armed, auto-arm, can fly, trajectory finished (drone22 on the ground)
        self.calls = []

    def estimate(self):
        return self.p.copy(), 0.01

    def status(self):
        return {"var": self.var, "vbat": self.vbat, "supervisor": self.supervisor, "age": 0.05}

    def arm(self, on):
        self.calls.append(("arm", on))

    def takeoff(self, z, t):
        self.calls.append(("takeoff", round(z, 4), t))

    def land(self, z, t):
        self.calls.append(("land", round(z, 4), t))

    def stop_motors(self):
        self.calls.append(("stop",))

    def emergency_stop(self):
        self.calls.append(("estop",))

    def feed_watchdog(self):
        self.calls.append(("watchdog",))

    def names(self):
        return [c[0] for c in self.calls if c[0] != "watchdog"]


class FlightFixture(unittest.TestCase):
    def setUp(self):
        self.link = FakeLink()
        self.fc = FlightController(self.link, CFG)
        self.now = 100.0

    def run_for(self, seconds, send=True, dt=0.02):
        """Advance the clock, sending the link's true position as the tracked one."""
        end = self.now + seconds
        while self.now < end - 1e-9:
            self.now += dt
            if send:
                self.fc.note_sent(self.link.p, self.now)
            self.fc.tick(self.now)

    def fly_to_hover(self):
        self.run_for(1.0)
        self.assertIsNone(self.fc.request_takeoff(self.now))
        self.assertEqual(self.fc.state, ARMING)
        self.run_for(0.4)
        self.assertEqual(self.fc.state, TAKEOFF)
        self.link.p = GROUND + [0, 0, 0.25]
        self.run_for(2.0)
        self.assertEqual(self.fc.state, HOVER)


class FlightTest(FlightFixture):
    def test_full_flight(self):
        self.fly_to_hover()
        self.assertEqual(self.link.calls[0], ("arm", True))
        self.assertIn(("takeoff", round(GROUND[2] + 0.25, 4), 2.0), self.link.calls)
        self.fc.request_land("test", self.now)
        self.assertEqual(self.fc.state, LANDING)
        self.assertIn(("land", round(GROUND[2], 4), 2.0), self.link.calls)
        self.link.p = GROUND.copy()
        self.run_for(2.6)
        self.assertEqual(self.fc.state, GROUNDED)
        self.assertEqual(self.link.names(), ["arm", "takeoff", "land", "stop", "arm"])

    def test_watchdog_fed_through_flight_and_after(self):
        self.fly_to_hover()
        fed = sum(1 for c in self.link.calls if c[0] == "watchdog")
        self.assertGreaterEqual(fed, 20)  # about 10 Hz for 2.4 s
        self.fc.request_land("test", self.now)
        self.run_for(3.0)
        before = sum(1 for c in self.link.calls if c[0] == "watchdog")
        self.run_for(1.0)
        self.assertGreater(sum(1 for c in self.link.calls if c[0] == "watchdog"), before)

    def test_preflight_refusals(self):
        self.assertIn("Hz", self.fc.request_takeoff(self.now))
        self.run_for(1.0)
        self.link.vbat = 3.6
        self.assertIn("battery", self.fc.request_takeoff(self.now))
        self.link.vbat = 4.1
        self.link.var = (1e-2, 1e-5, 1e-5)
        self.assertIn("variance", self.fc.request_takeoff(self.now))
        self.link.var = (1e-5,) * 3
        self.fc.note_sent(GROUND + [0.1, 0, 0], self.now)
        self.assertIn("mm from the tracked", self.fc.request_takeoff(self.now))
        self.assertEqual(self.fc.state, GROUNDED)
        self.assertEqual(self.link.calls, [])

    def test_pose_loss_lands(self):
        self.fly_to_hover()
        self.run_for(0.4, send=False)
        self.assertEqual(self.fc.state, LANDING)
        self.assertIn("last position sent", self.fc.status)

    def test_pose_loss_while_arming_cancels(self):
        self.run_for(1.0)
        self.fc.request_takeoff(self.now)
        self.run_for(0.31, send=False, dt=0.31)
        self.assertEqual(self.fc.state, GROUNDED)
        self.assertEqual(self.link.names(), ["arm", "arm"])

    def test_estimate_disagreement_lands(self):
        self.fly_to_hover()
        self.fc.note_sent(self.link.p + [0.05, 0, 0], self.now)
        self.now += 0.01
        self.fc.tick(self.now)
        self.assertEqual(self.fc.state, LANDING)

    def test_out_of_bounds_kills(self):
        self.fly_to_hover()
        self.link.p = GROUND + [0.25, 0, 0.25]
        self.run_for(0.02, send=False)
        self.assertEqual(self.fc.state, KILLED)
        self.assertIn("estop", self.link.names())
        self.assertIsNotNone(self.fc.request_takeoff(self.now))

    def test_too_high_kills(self):
        self.fly_to_hover()
        self.link.p = GROUND + [0, 0, 0.5]
        self.run_for(0.02)
        self.assertEqual(self.fc.state, KILLED)

    def test_kill_stops_watchdog_feeding(self):
        self.fly_to_hover()
        self.fc.kill("test", self.now)
        before = len(self.link.calls)
        self.run_for(0.5)
        self.assertEqual(len(self.link.calls), before)

    def test_close_lands_when_airborne(self):
        self.fly_to_hover()
        self.fc.close()  # thread not started: land is sent; the wait times out and kills
        self.assertIn("land", self.link.names())
        self.assertEqual(self.fc.state, KILLED)


class SupervisorTest(FlightFixture):
    """The drone's supervisor state: locked drones, arming, lift-off."""

    def test_decode(self):
        self.assertIsNone(supervisor_problem({"supervisor": 526}))
        self.assertEqual(supervisor_problem({"supervisor": 580}), "locked")  # drone23/24
        self.assertEqual(supervisor_problem({"supervisor": (1 << 6) | (1 << 7)}), "locked, crashed")
        self.assertIsNone(supervisor_problem({}))

    def test_locked_refuses_takeoff(self):
        self.link.supervisor = 580
        self.run_for(1.0)
        self.assertIn("locked", self.fc.request_takeoff(self.now))
        self.assertEqual(self.link.calls, [])

    def test_waits_for_can_fly_then_times_out(self):
        self.run_for(1.0)
        self.assertIsNone(self.fc.request_takeoff(self.now))
        self.link.supervisor = 516  # disarmed: no can-fly bit
        self.run_for(1.0)
        self.assertEqual(self.fc.state, ARMING)
        self.run_for(0.6)
        self.assertEqual(self.fc.state, GROUNDED)
        self.assertIn("can-fly", self.fc.status)
        self.assertEqual(self.link.names(), ["arm", "arm"])

    def test_arms_late_then_takes_off(self):
        self.run_for(1.0)
        self.fc.request_takeoff(self.now)
        self.link.supervisor = 516
        self.run_for(0.6)
        self.link.supervisor = 526
        self.run_for(0.04)
        self.assertEqual(self.fc.state, TAKEOFF)

    def test_no_liftoff_lands(self):
        self.run_for(1.0)
        self.fc.request_takeoff(self.now)
        self.run_for(0.4)
        self.assertEqual(self.fc.state, TAKEOFF)
        self.link.p = GROUND + [0, 0, 0.01]  # 1 cm of the 25 cm climb
        self.run_for(2.1)
        self.assertEqual(self.fc.state, LANDING)
        self.assertIn("did not lift off", self.fc.status)

    def test_lock_in_flight_kills(self):
        self.fly_to_hover()
        self.link.supervisor = 580
        self.run_for(0.02)
        self.assertEqual(self.fc.state, KILLED)
        self.assertIn("locked", self.fc.status)


if __name__ == "__main__":
    unittest.main()
