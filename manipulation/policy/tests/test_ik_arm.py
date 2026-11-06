"""ik_arm.SimIkPolicyArm's inner loop against a fake driver whose joints follow the commanded
targets (first order, tau 30 ms): rate, goal time, convergence to the held goal, stop and kill.

Needs mujoco and trossen_arm (wxai_mjlab or wxai_mjlab_hw):

  /opt/conda-envs/wxai_mjlab/bin/python -m pytest real_robot/manipulation/policy/tests/test_ik_arm.py
"""

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("trossen_arm")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from manipulation.policy.actions import ee_delta_target  # noqa: E402
from manipulation.policy.ik import DifferentialIK  # noqa: E402
from manipulation.policy.ik_arm import SimIkPolicyArm  # noqa: E402
from manipulation.policy.jenga.config import JengaPolicyConfig  # noqa: E402

TAU_S = 0.03


class FakeDriver:
    def __init__(self, q0):
        self.lock = threading.Lock()
        self.q, self.target, self.t = np.array(q0, dtype=np.float64), np.array(q0, dtype=np.float64), time.monotonic()
        self.id = 0
        self.sends, self.goal_times, self.fail_after = [], set(), None

    def _advance(self):
        now = time.monotonic()
        self.q += (self.target - self.q) * (1 - np.exp(-(now - self.t) / TAU_S))
        self.t = now

    def get_robot_output(self):
        with self.lock:
            self._advance()
            self.id += 1
            q = self.q.tolist()
        return SimpleNamespace(header=SimpleNamespace(id=self.id, timestamp=0),
                               joint=SimpleNamespace(arm=SimpleNamespace(positions=q, velocities=[0.0] * 6),
                                                     gripper=SimpleNamespace(position=0.026, velocity=0.0)),
                               cartesian=SimpleNamespace(positions=[0.0] * 6))

    def set_arm_positions(self, q, goal_time, blocking):
        if self.fail_after is not None and len(self.sends) >= self.fail_after:
            raise RuntimeError("controller fault")
        with self.lock:
            self._advance()
            self.target = np.array(q)
        self.sends.append(time.monotonic())
        self.goal_times.add((goal_time, blocking))

    def get_num_joints(self):
        return 7

    def __getattr__(self, name):  # mode and effort calls of hold / kill
        return lambda *a, **k: None


@pytest.fixture()
def setup():
    cfg = JengaPolicyConfig.load()
    k = cfg.sim_ik
    ik = DifferentialIK(k.model, k.site, k.params, cfg.ready_joint_pos)
    arm = SimIkPolicyArm({"ip": "fake"}, 0.002, ik, k.inner_hz, k.goal_time_s)
    arm.driver = FakeDriver(cfg.ready_joint_pos)
    arm.position_mode, arm.gripper_range = True, (0.002, 0.042)
    yield cfg, ik, arm
    arm._stop_loop()


def _goal(cfg, ik, action):
    return ee_delta_target(ik.fk(cfg.ready_joint_pos), [*action, 1.0], pos_scale=cfg.ee_pos_scale,
                           rot_scale=cfg.ee_rot_scale, gripper_open=0.04, gripper_closed=0.0).T_base_target


def test_tracks_held_goal_at_inner_rate(setup):
    cfg, ik, arm = setup
    T_goal = _goal(cfg, ik, [1.0, -1.0, -1.0, 0.0, 0.0, 1.0])
    arm.send_tcp(T_goal, 0.05)
    time.sleep(0.6)
    stats = arm.drain_stats()
    T = ik.fk(arm.driver.q)
    assert arm.driver.goal_times == {(cfg.sim_ik.goal_time_s, False)}
    arm.hold_at(arm.driver.q, 0.2)
    sends = list(arm.driver.sends[:-1])  # the last is hold_at's stop
    time.sleep(0.05)
    assert len(arm.driver.sends) == len(sends) + 1  # hold_at stops the loop
    rate = (len(sends) - 1) / (sends[-1] - sends[0])
    print(f"inner loop {rate:.0f} Hz; stats {stats}")
    assert rate == pytest.approx(cfg.sim_ik.inner_hz, rel=0.1)
    assert np.linalg.norm(T[:3, 3] - T_goal[:3, 3]) < 1e-4
    assert np.allclose(T[:3, :3], T_goal[:3, :3], atol=1e-3)


def test_new_goal_replaces_held_one(setup):
    cfg, ik, arm = setup
    arm.send_tcp(_goal(cfg, ik, [1.0, 0, 0, 0, 0, 0]), 0.05)
    time.sleep(0.05)
    T_goal = _goal(cfg, ik, [0, 1.0, 0, 0, 0, 0])
    arm.send_tcp(T_goal, 0.05)
    time.sleep(0.5)
    assert np.linalg.norm(ik.fk(arm.driver.q)[:3, 3] - T_goal[:3, 3]) < 1e-4


def test_driver_error_kills(setup):
    cfg, ik, arm = setup
    arm.driver.fail_after = 20
    arm.send_tcp(_goal(cfg, ik, [1.0, 0, 0, 0, 0, 0]), 0.05)
    time.sleep(0.2)
    assert arm.killed and "controller fault" in arm.kill_reason and arm._thread is None
    with pytest.raises(RuntimeError):
        arm.send_tcp(_goal(cfg, ik, [0, 0, 0, 0, 0, 0]), 0.05)
