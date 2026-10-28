"""Training-model loading shared by simulation experiments.

The model is the embodiment XML compiled the way ``scripts/train.py`` compiles
it (``load_spec``), so stripped bodies and the training TCP site match training.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
from tabletop_sim.embodiments import get_embodiment
from tabletop_sim.embodiments.base import EmbodimentCfg, load_spec
from tabletop_sim.scene import load_scene_config

INTEGRATORS = {
    "euler": mujoco.mjtIntegrator.mjINT_EULER,
    "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
}
CONES = {
    "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
    "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC,
}


def apply_training_solver(model: mujoco.MjModel, emb: EmbodimentCfg, timestep: float) -> None:
    """``load_spec`` clears the solver; training puts the embodiment values back."""
    model.opt.timestep = timestep
    model.opt.integrator = INTEGRATORS[emb.integrator]
    model.opt.impratio = emb.impratio
    model.opt.cone = CONES[emb.cone]


def joint_addresses(model: mujoco.MjModel, name: str) -> tuple[int, int]:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise ValueError(f"joint {name!r} not in the model")
    return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])


def load_training_robot(scene_path: Path) -> tuple[mujoco.MjModel, EmbodimentCfg]:
    """Robot model for the scene's embodiment. The scene base pose is not applied."""
    scene = load_scene_config(scene_path)
    emb = get_embodiment(scene.robot.embodiment)
    model = load_spec(emb).compile()
    return model, emb
