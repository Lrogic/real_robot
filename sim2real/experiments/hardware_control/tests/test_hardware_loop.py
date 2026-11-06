"""The hardware trial loop against a simulated clock and a fake driver (no arm, no mjlab).

Checks: Cartesian targets sent once / per schedule index, targets held between outer
updates, external-IK updates every tick with reused-sample flags, no catch-up after an
overrun, stale-state and Ctrl+C stops.
"""

import dataclasses

import numpy as np
import pytest

from geometry import displaced, make_T, pose_from_json
from hardware import CartesianBackend, ExternalIKBackend, Safety, StateSample, reset_to_ready, run_trial
from plans import make_plan
from settings import load_settings

T_REF = make_T(np.diag([1.0, -1.0, -1.0]), [0.2, 0.0, 0.1])


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def now(self):
        return self.t

    def sleep_until(self, deadline):
        self.t = max(self.t, deadline)


class FakeFK:
    """TCP = reference translated by q[:3] (m); enough for loop bookkeeping."""

    def tcp(self, q):
        return displaced(T_REF, np.asarray(q)[:3])


class FakeIO:
    def __init__(self, clock, state_hz=1000.0, read_s=50e-6, send_s=100e-6):
        self.clock, self.state_hz, self.read_s, self.send_s = clock, state_hz, read_s, send_s
        self.q = np.zeros(6)
        self.cartesian = []
        self.joints = []
        self.stopped = None
        self.killed = None
        self.freeze_after = None
        self.slow_read_at = None
        self.interrupt_at = None

    def read(self, clock):
        t0 = clock.now()
        if self.interrupt_at is not None and t0 >= self.interrupt_at:
            raise KeyboardInterrupt
        clock.t += self.read_s
        if self.slow_read_at is not None and t0 >= self.slow_read_at:
            clock.t += 0.17
            self.slow_read_at = None
        t = t0 if self.freeze_after is None else min(t0, self.freeze_after)
        hid = int(t * self.state_hz)
        return StateSample(t0, clock.now(), hid, int(hid * 1e6 / self.state_hz), self.q.copy(), np.zeros(6), 0.026,
                           np.zeros(6))

    def send_cartesian(self, T, space, goal_time):
        self.clock.t += self.send_s
        self.cartesian.append((self.clock.t, np.array(T), space, goal_time))
        self.q[:3] = np.asarray(T)[:3, 3] - T_REF[:3, 3]

    def send_joints(self, q, goal_time):
        self.clock.t += self.send_s
        self.joints.append((self.clock.t, np.array(q), goal_time))
        self.q = np.array(q)

    def stop_at(self, q, goal_time):
        self.stopped = (np.array(q), goal_time)

    def kill(self, reason):
        self.killed = reason

    def error_information(self):
        return "none"


class FakeIK:
    def __init__(self, clock):
        self.clock = clock
        self.targets = []
        self.solves = []

    def set_target(self, T):
        self.targets.append((self.clock.t, np.array(T)))

    def solve(self, q, qd=None):
        self.clock.t += 200e-6
        T = self.targets[-1][1]
        self.solves.append((self.clock.t, T.copy()))
        q_target = np.array(q, dtype=float)
        q_target[:3] = T[:3, 3] - T_REF[:3, 3]
        return q_target, {"compute_s": 200e-6, "max_dq_clamped": False}


class PlanFK:
    def __init__(self):
        self.ready_arm = np.zeros(6)
        self.ready_gripper = 0.026

    def tcp(self, q):
        return FakeFK().tcp(q)

    def describe(self):
        return {"fake": True}


@pytest.fixture(scope="module")
def cfg():
    return load_settings()


def plan_for(cfg, test, **kw):
    return make_plan(PlanFK(), cfg, test, kw.pop("kind", "translation"), kw.pop("axis", "x"), kw.pop("sign", 1))


def safety(cfg):
    sc = dict(cfg["hardware"]["safety"], workspace_min_m=[-1, -1, -1], workspace_max_m=[1, 1, 1])
    return Safety(sc, T_REF, np.full(6, -3.0), np.full(6, 3.0))


def trial(cfg, io, backend, plan, clock, stale_s=0.05):
    return run_trial(io, backend, plan, safety(cfg), FakeFK(), clock, stale_s=stale_s, max_trial_s=15.0,
                     stop_goal_time_s=0.2)


def test_cartesian_single_move_sends_the_target_once(cfg):
    clock = FakeClock()
    io = FakeIO(clock)
    plan = plan_for(cfg, "single")
    out = trial(cfg, io, CartesianBackend(io, "joint", 0.05, 1000.0), plan, clock)
    assert out["stop_reason"] == "completed"
    assert len(io.cartesian) == 1
    assert np.allclose(io.cartesian[0][1], pose_from_json(plan["targets"][0]["pose"]))
    assert io.cartesian[0][2:] == ("joint", 0.05)
    outer = out["outer_updates"]
    assert outer[0]["new_target"] and not any(o["new_target"] for o in outer[1:])
    assert len(outer) == pytest.approx(cfg["single"]["observe_s"] * 20, abs=1)
    lateness = [o["actual_t_s"] - o["scheduled_t_s"] for o in outer]
    assert max(lateness) < 0.0015
    assert out["timing_summary"]["loop"]["skipped_slots"] == 0


def test_cartesian_scheduled_sends_every_index_in_order(cfg):
    clock = FakeClock()
    io = FakeIO(clock)
    plan = plan_for(cfg, "scheduled", kind="rotation", axis="z", sign=-1)
    out = trial(cfg, io, CartesianBackend(io, "cartesian", 0.5, 1000.0), plan, clock)
    sent = [s[1] for s in io.cartesian]
    assert len(sent) == len(plan["targets"])
    for T, t in zip(sent, plan["targets"]):
        assert np.allclose(T, pose_from_json(t["pose"]))
    times = np.array([o["actual_t_s"] for o in out["outer_updates"] if o["new_target"]])
    assert np.allclose(np.diff(times), 0.05, atol=0.0015)
    assert out["timing_summary"]["outer"]["skipped_indices"] == 0


def test_external_ik_holds_target_between_outer_updates_and_flags_reused_states(cfg):
    clock = FakeClock()
    io = FakeIO(clock, state_hz=250.0)
    ik = FakeIK(clock)
    plan = plan_for(cfg, "scheduled")
    backend = ExternalIKBackend(io, ik, safety(cfg), 0.0, 500.0)
    out = trial(cfg, io, backend, plan, clock)
    assert len(ik.targets) == len(plan["targets"])
    assert len(io.joints) == len(ik.solves) == out["timing_summary"]["backend"]["ik_updates"]
    # every solve uses the most recent target set at an outer update
    set_times = np.array([t for t, _ in ik.targets])
    for t_solve, T in ik.solves:
        latest = np.flatnonzero(set_times <= t_solve)[-1]
        assert np.array_equal(T, ik.targets[latest][1])
    per_update = len(ik.solves) / len(out["outer_updates"])
    assert per_update == pytest.approx(25, abs=1.5)
    reused = out["commands"]["state_reused"]
    assert 0.4 < np.mean(reused) < 0.6          # 500 Hz ticks on a 250 Hz state stream
    assert out["timing_summary"]["backend"]["reused_state_updates"] == sum(reused)
    assert out["timing_summary"]["loop"]["achieved_hz"] == pytest.approx(500, rel=0.02)


def test_overrun_skips_outer_indices_without_catch_up(cfg):
    clock = FakeClock()
    io = FakeIO(clock)
    io.slow_read_at = clock.t + 0.101                   # one read takes 170 ms at about t = 0.1 s
    plan = plan_for(cfg, "scheduled")
    out = trial(cfg, io, CartesianBackend(io, "joint", 0.1, 1000.0), plan, clock)
    indices = [o["index"] for o in out["outer_updates"]]
    assert indices == sorted(set(indices))
    assert out["timing_summary"]["outer"]["skipped_indices"] >= 2
    assert out["timing_summary"]["loop"]["skipped_slots"] > 100
    sent = [o for o in out["outer_updates"] if o["new_target"]]
    for o in sent:                                       # no burst: each send is inside its own slot
        assert o["scheduled_t_s"] - 1e-9 <= o["actual_t_s"] < o["scheduled_t_s"] + plan["outer_period_s"]
    sent_idx = [o["index"] for o in sent]
    for k, (_, T, *_) in zip(sent_idx, io.cartesian):
        assert np.allclose(T, pose_from_json(plan["targets"][k]["pose"]))


def test_stale_state_stops_and_holds(cfg):
    clock = FakeClock()
    io = FakeIO(clock)
    io.freeze_after = clock.t + 0.3
    out = trial(cfg, io, CartesianBackend(io, "joint", 0.05, 1000.0), plan_for(cfg, "single"), clock)
    assert out["stop_reason"].startswith("safety: no new state")
    assert io.stopped is not None and io.killed is None


def test_keyboard_interrupt_kills(cfg):
    clock = FakeClock()
    io = FakeIO(clock)
    io.interrupt_at = clock.t + 0.2
    out = trial(cfg, io, CartesianBackend(io, "joint", 0.05, 1000.0), plan_for(cfg, "single"), clock)
    assert out["interrupted"] and out["stop_reason"] == "keyboard_interrupt"
    assert io.killed is not None


def test_joint_command_check_rejects_large_steps(cfg):
    clock = FakeClock()
    io = FakeIO(clock)

    class JumpIK(FakeIK):
        def solve(self, q, qd=None):
            return np.asarray(q) + 0.5, {"compute_s": 0.0, "max_dq_clamped": False}

    backend = ExternalIKBackend(io, JumpIK(clock), safety(cfg), 0.0, 500.0)
    out = trial(cfg, io, backend, plan_for(cfg, "single"), clock)
    assert out["stop_reason"].startswith("safety: joint target")
    assert io.joints == []


def test_unsupported_conditions_are_rejected(cfg):
    io = FakeIO(FakeClock())
    with pytest.raises(ValueError):
        CartesianBackend(io, "task", 0.05, 1000.0)
    with pytest.raises(ValueError):
        CartesianBackend(io, "joint", -0.1, 1000.0)


class ResetIO(FakeIO):
    """Ends the reset with a static offset; reported velocity is noise, position may drift."""

    def __init__(self, clock, offset, drift_rad_s=0.0, velocity_noise=0.03):
        super().__init__(clock)
        self.offset, self.drift, self.noise = np.asarray(offset), drift_rad_s, velocity_noise
        self.rng = np.random.default_rng(0)

    def move_to_ready(self, q, gripper, move_time):
        self.q_ready, self.t_arrive = np.asarray(q, dtype=float), self.clock.t + move_time

    def read(self, clock):
        if hasattr(self, "q_ready"):
            self.q = self.q_ready + self.offset + self.drift * max(0.0, clock.t - self.t_arrive)
        return dataclasses.replace(super().read(clock), qd=self.rng.uniform(-self.noise, self.noise, 6))


def test_reset_settles_despite_noisy_velocity_samples(cfg):
    rc = cfg["hardware"]["reset"]
    clock = FakeClock()
    io = ResetIO(clock, offset=[0.0, -0.011, -0.013, 0.003, 0.0, 0.0])
    rec = reset_to_ready(io, np.zeros(6), 0.026, rc, clock, confirm=lambda msg: None)
    assert rec["settled"]
    assert rec["drift_speed_rad_s"] < 1e-9


def test_reset_rejects_a_drifting_arm(cfg):
    rc = cfg["hardware"]["reset"]
    clock = FakeClock()
    drift = np.zeros(6)
    drift[1] = 1.5 * rc["velocity_tolerance_rad_s"]
    io = ResetIO(clock, offset=np.zeros(6), drift_rad_s=drift, velocity_noise=0.0)
    rec = reset_to_ready(io, np.zeros(6), 0.026, rc, clock, confirm=lambda msg: None)
    assert not rec["settled"]
