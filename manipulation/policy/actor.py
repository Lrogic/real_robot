"""The trained actor: rsl_rl MLPModel weights and observation normalizer, evaluated in numpy.

PolicyActor.load reads model_*.pt and the run's params/env.yaml and params/agent.yaml, and
refuses to build if the run does not match the deployment: the actor observation terms and
order (observations.ACTOR_TERMS), the action terms and scales, the gripper positions, the
control rate, and the network type. The deterministic action is the MLP output (the mean of
the Gaussian), as mjlab play uses it; the env clips it later (actions.ee_delta_target).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml

from .observations import ACTOR_DIM, ACTOR_TERMS

ACTION_DIM = 7  # ee delta pose (6) + binary gripper (1)
# rsl_rl EmpiricalNormalization's default eps; it is not stored in the checkpoint.
NORMALIZER_EPS = 1e-2
ARM_JOINTS = [f"joint_{i}" for i in range(6)]


class _RunYamlLoader(yaml.SafeLoader):
    """Safe loader that reads !!python/* tagged nodes as plain lists, dicts and strings."""


def _construct_tagged(loader, _suffix, node):
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return loader.construct_scalar(node)


_RunYamlLoader.add_multi_constructor("tag:yaml.org,2002:python/", _construct_tagged)


def load_run_yaml(path: Path) -> dict:
    return yaml.load(Path(path).read_text(), Loader=_RunYamlLoader)


def _close(a, b, tol=1e-9) -> bool:
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


def check_run(env: dict, agent: dict, cfg) -> List[str]:
    """Every mismatch between a training run's saved configs and the deployment config."""
    problems = []
    actor = agent.get("actor", {})
    if actor.get("class_name") != "MLPModel":
        problems.append(f"agent actor.class_name is {actor.get('class_name')!r}, expected 'MLPModel'")
    if actor.get("activation") != "elu":
        problems.append(f"agent actor.activation is {actor.get('activation')!r}, expected 'elu'")
    if actor.get("rnn_type") is not None or actor.get("cnn_cfg") is not None:
        problems.append("agent actor has an RNN or CNN; only a plain MLP is supported")
    if actor.get("obs_normalization") is not True:
        problems.append("agent actor.obs_normalization is not true")
    if list(agent.get("obs_groups", {}).get("actor", [])) != ["actor"]:
        problems.append(f"agent obs_groups.actor is {agent.get('obs_groups', {}).get('actor')!r}, expected ['actor']")

    group = env.get("observations", {}).get("actor", {})
    terms = group.get("terms", {})
    names = [n for n, _ in ACTOR_TERMS]
    if list(terms) != names:
        problems.append(f"env actor observation terms are {list(terms)}, expected {names}")
    if group.get("history_length") not in (None, 0) or group.get("enable_corruption"):
        problems.append("env actor observation group has history or corruption enabled")
    for name, term in terms.items():
        if term.get("history_length") not in (None, 0) or term.get("clip") is not None \
                or term.get("scale") is not None or term.get("delay_max_lag") not in (None, 0):
            problems.append(f"env actor term {name} has history, clip, scale or delay")

    actions = env.get("actions", {})
    if list(actions) != ["arm", "gripper"]:
        problems.append(f"env action terms are {list(actions)}, expected ['arm', 'gripper']")
    arm, gripper = actions.get("arm", {}), actions.get("gripper", {})
    if list(arm.get("actuator_names", [])) != ARM_JOINTS:
        problems.append(f"env arm action actuators are {arm.get('actuator_names')}, expected {ARM_JOINTS}")
    if arm.get("frame_name") != "ee_site" or arm.get("use_relative_mode") is not True:
        problems.append("env arm action is not a relative delta of ee_site (the driver tool frame)")
    if not _close(arm.get("delta_pos_scale"), cfg.ee_pos_scale):
        problems.append(f"env arm delta_pos_scale {arm.get('delta_pos_scale')} != task ee_pos_scale {cfg.ee_pos_scale}")
    if not _close(arm.get("delta_ori_scale"), cfg.ee_rot_scale):
        problems.append(f"env arm delta_ori_scale {arm.get('delta_ori_scale')} != task ee_rot_scale {cfg.ee_rot_scale}")
    if arm.get("clip") is not None or gripper.get("clip") is not None:
        problems.append("env action terms have a clip other than the [-1, 1] clamp")
    if gripper.get("mode") != "binary":
        problems.append(f"env gripper action mode is {gripper.get('mode')!r}, expected 'binary'")
    if not _close(gripper.get("open_pos"), cfg.gripper_open) or not _close(gripper.get("closed_pos"), cfg.gripper_closed):
        problems.append(f"env gripper open/closed {gripper.get('open_pos')}/{gripper.get('closed_pos')} != "
                        f"config {cfg.gripper_open}/{cfg.gripper_closed}")

    try:
        step_dt = float(env["decimation"]) * float(env["sim"]["mujoco"]["timestep"])
    except (KeyError, TypeError):
        step_dt = None
    if step_dt is None or abs(step_dt - 1.0 / cfg.control_hz) > 1e-9:
        problems.append(f"env control step {step_dt} s != 1 / task control_hz ({1.0 / cfg.control_hz} s)")
    return problems


def _elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.0, x, np.expm1(np.minimum(x, 0.0)))


@dataclass
class PolicyActor:
    path: Path
    layers: List[Tuple[np.ndarray, np.ndarray]]  # (W, b) per Linear; ELU between them
    obs_mean: np.ndarray
    obs_std: np.ndarray
    info: dict = field(default_factory=dict)

    @classmethod
    def load(cls, checkpoint: Path, cfg) -> "PolicyActor":
        import torch

        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"policy checkpoint {checkpoint} not found (policy.checkpoint)")
        params = checkpoint.parent / "params"
        env, agent = load_run_yaml(params / "env.yaml"), load_run_yaml(params / "agent.yaml")
        problems = check_run(env, agent, cfg)

        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        actor = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in state["actor_state_dict"].items()
                 if hasattr(v, "detach")}
        layers = []
        index = 0
        while f"mlp.{index}.weight" in actor:
            layers.append((actor[f"mlp.{index}.weight"], actor[f"mlp.{index}.bias"]))
            index += 2  # Linear, activation, Linear, ...
        if not layers:
            raise ValueError(f"{checkpoint}: no mlp.* weights in actor_state_dict")
        hidden = [W.shape[0] for W, _ in layers[:-1]]
        if layers[0][0].shape[1] != ACTOR_DIM:
            problems.append(f"actor input size {layers[0][0].shape[1]} != {ACTOR_DIM} observations")
        if layers[-1][0].shape[0] != ACTION_DIM:
            problems.append(f"actor output size {layers[-1][0].shape[0]} != {ACTION_DIM} actions")
        if hidden != list(agent.get("actor", {}).get("hidden_dims", [])):
            problems.append(f"actor weights have hidden sizes {hidden}, agent.yaml says "
                            f"{agent.get('actor', {}).get('hidden_dims')}")
        if "obs_normalizer._mean" not in actor:
            problems.append("actor_state_dict has no obs_normalizer")
        if problems:
            raise ValueError(f"policy run {checkpoint.parent} does not match the deployment config:\n  - "
                             + "\n  - ".join(problems))
        mean = actor["obs_normalizer._mean"].reshape(ACTOR_DIM)
        std = actor["obs_normalizer._std"].reshape(ACTOR_DIM)
        info = {"checkpoint": str(checkpoint), "iteration": state.get("iter"), "hidden_dims": hidden,
                "experiment": agent.get("experiment_name"),
                "normalizer_count": float(actor.get("obs_normalizer.count", np.nan))}
        return cls(path=checkpoint, layers=layers, obs_mean=mean, obs_std=std, info=info)

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        return (np.asarray(obs, dtype=np.float64).reshape(ACTOR_DIM) - self.obs_mean) / (self.obs_std + NORMALIZER_EPS)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Raw (unclipped) action for one actor observation vector."""
        x = self.normalize(obs)
        for W, b in self.layers[:-1]:
            x = _elu(W @ x + b)
        W, b = self.layers[-1]
        return W @ x + b
