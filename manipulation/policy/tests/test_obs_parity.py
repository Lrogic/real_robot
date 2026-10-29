"""The real-robot observation functions reproduce the simulator's actor observation.

Builds the jenga training env (pose noise off, so the actor sees the true object pose),
drives it with random actions, and rebuilds every env's actor vector from raw sim state
(joint positions and velocities, TCP site pose, object body pose, actor goal) with
manipulation.policy.observations. Also checks the policy config against the embodiment.

Needs the wxai_mjlab env and a CUDA device:

  /opt/conda-envs/wxai_mjlab/bin/python -m pytest real_robot/manipulation/policy/tests/test_obs_parity.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from manipulation.policy import observations as obs  # noqa: E402
from manipulation.policy.jenga.config import JengaPolicyConfig  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEVICE = "cuda:0"
NUM_ENVS = 4
STEPS = 80


def _T(pos, quat_wxyz) -> np.ndarray:
    return obs.make_T(obs.quat_wxyz_to_matrix(quat_wxyz), pos)


@pytest.fixture(scope="module")
def cfg() -> JengaPolicyConfig:
    return JengaPolicyConfig.load()


def test_config_matches_embodiment(cfg):
    from tabletop_sim.embodiments import get_embodiment

    emb = get_embodiment(cfg.embodiment)
    assert cfg.gripper_open == emb.gripper_open
    assert cfg.gripper_closed == emb.gripper_closed
    assert emb.arm_joint_names == tuple(f"joint_{i}" for i in range(6))
    assert emb.gripper_joint_names == ("left_carriage_joint", "right_carriage_joint")


def test_actor_observation_parity(cfg, tmp_path):
    from mjlab.envs import ManagerBasedRlEnv
    from tabletop_sim.tasks.grasp.env_cfg import PERCEPTION_NAME, make_grasp_env_cfg
    from tabletop_sim.tasks.grasp.mdp import task_state

    task = json.loads(cfg.task_path.read_text())
    for source in task["domain_randomization"]["object_pose_noise"].values():
        source["enabled"] = False
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task))

    env_cfg = make_grasp_env_cfg(cfg.scene_path, task_path, None, NUM_ENVS)
    env_cfg.seed = 0
    env = ManagerBasedRlEnv(env_cfg, device=DEVICE)
    try:
        assert list(env.observation_manager.active_terms["actor"]) == [n for n, _ in obs.ACTOR_TERMS]
        arm_cfg = env.reward_manager.get_term_cfg("arm_speed").params["asset_cfg"]
        gripper_cfg = env.reward_manager.get_term_cfg("gripper_speed").params["asset_cfg"]
        tcp_cfg = env.reward_manager.get_term_cfg("reach").params["tcp_cfg"]
        object_cfg = env.reward_manager.get_term_cfg("reach").params["object_cfg"]
        robot = env.scene["robot"]
        gen = torch.Generator(device=DEVICE).manual_seed(0)
        actor, _ = env.reset()
        actor = actor["actor"]
        worst = np.zeros(obs.ACTOR_DIM)
        for step in range(STEPS + 1):
            assert actor.shape == (NUM_ENVS, obs.ACTOR_DIM)
            q = robot.data.joint_pos.cpu().numpy()
            qd = robot.data.joint_vel.cpu().numpy()
            root = robot.data.root_link_pose_w.cpu().numpy()
            tcp_pos, tcp_quat = (t.cpu().numpy() for t in task_state.tcp_pose_w(env, tcp_cfg))
            obj_pos, obj_quat = (t.cpu().numpy() for t in task_state.object_pose_w(env, object_cfg))
            goal_w = task_state.perception(env, PERCEPTION_NAME).actor_target_pos.cpu().numpy()
            expected = actor.cpu().numpy()
            for i in range(NUM_ENVS):
                T_base_world = np.linalg.inv(_T(root[i, :3], root[i, 3:7]))
                goal_b = T_base_world[:3, :3] @ goal_w[i] + T_base_world[:3, 3]
                terms = obs.actor_terms(
                    arm_q=q[i, arm_cfg.joint_ids], arm_qd=qd[i, arm_cfg.joint_ids],
                    carriage_q=q[i, gripper_cfg.joint_ids], carriage_qd=qd[i, gripper_cfg.joint_ids],
                    T_base_tcp=T_base_world @ _T(tcp_pos[i], tcp_quat[i]),
                    T_base_obj=T_base_world @ _T(obj_pos[i], obj_quat[i]),
                    goal_pos=goal_b, gripper_open=cfg.gripper_open, gripper_closed=cfg.gripper_closed)
                ours = obs.build_actor_obs(terms, dtype=np.float64)
                worst = np.maximum(worst, np.abs(ours - expected[i]))
            if step < STEPS:
                action = torch.rand(NUM_ENVS, env.action_manager.total_action_dim, generator=gen,
                                    device=DEVICE) * 2 - 1
                actor = env.step(action)[0]["actor"]
        print("max |real-side - sim| per term:",
              {name: float(worst[s:s + n].max()) for (name, n), s in
               zip(obs.ACTOR_TERMS, np.cumsum([0] + [n for _, n in obs.ACTOR_TERMS]))})
        assert worst.max() < 1e-5, worst
    finally:
        env.close()
