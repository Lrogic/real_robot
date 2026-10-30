"""The numpy PolicyActor matches rsl_rl's MLPModel on the configured checkpoint, and the run check
rejects a run that does not match the deployment.

  /opt/conda-envs/wxai_mjlab/bin/python -m pytest real_robot/manipulation/policy/tests/test_actor.py
"""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from manipulation.policy.actor import ACTION_DIM, PolicyActor, check_run, load_run_yaml  # noqa: E402
from manipulation.policy.jenga.config import JengaPolicyConfig  # noqa: E402
from manipulation.policy.observations import ACTOR_DIM  # noqa: E402

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def cfg() -> JengaPolicyConfig:
    cfg = JengaPolicyConfig.load()
    if cfg.checkpoint is None or not cfg.checkpoint.is_file():
        pytest.skip(f"policy checkpoint {cfg.checkpoint} not found")
    return cfg


@pytest.fixture(scope="module")
def run(cfg):
    params = cfg.checkpoint.parent / "params"
    return load_run_yaml(params / "env.yaml"), load_run_yaml(params / "agent.yaml")


def test_numpy_actor_matches_rsl_rl(cfg, run):
    mlp_model = pytest.importorskip("rsl_rl.models.mlp_model")
    from tensordict import TensorDict

    _, agent = run
    actor = PolicyActor.load(cfg.checkpoint, cfg)
    a = agent["actor"]
    obs = TensorDict({"actor": torch.zeros(1, ACTOR_DIM)}, batch_size=[1])
    model = mlp_model.MLPModel(obs, {"actor": ["actor"]}, "actor", ACTION_DIM, hidden_dims=tuple(a["hidden_dims"]),
                               activation=a["activation"], obs_normalization=a["obs_normalization"],
                               distribution_cfg=a["distribution_cfg"]).double()
    state = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"]
    model.load_state_dict({k: v.double() if v.is_floating_point() else v for k, v in state.items()})
    model.eval()

    rng = np.random.default_rng(0)
    x = actor.obs_mean + rng.normal(size=(64, ACTOR_DIM)) * 3.0 * actor.obs_std
    with torch.no_grad():
        expected = model(TensorDict({"actor": torch.tensor(x)}, batch_size=[len(x)])).numpy()
    ours = np.stack([actor(row) for row in x])
    assert np.abs(ours - expected).max() < 1e-9


def test_run_matches_config(cfg, run):
    env, agent = run
    assert check_run(env, agent, cfg) == []


@pytest.mark.parametrize("edit, fragment", [
    (lambda env, agent: env["actions"]["arm"].__setitem__("delta_pos_scale", 0.02), "delta_pos_scale"),
    (lambda env, agent: env["actions"]["gripper"].__setitem__("mode", "delta"), "gripper action mode"),
    (lambda env, agent: env["observations"]["actor"]["terms"].pop("qvel"), "observation terms"),
    (lambda env, agent: env.__setitem__("decimation", 10), "control step"),
    (lambda env, agent: agent["actor"].__setitem__("activation", "tanh"), "activation"),
])
def test_run_mismatch_is_rejected(cfg, run, edit, fragment):
    env, agent = copy.deepcopy(run)
    edit(env, agent)
    problems = check_run(env, agent, cfg)
    assert any(fragment in p for p in problems), problems
