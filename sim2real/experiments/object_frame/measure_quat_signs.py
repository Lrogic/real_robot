"""Temp check (wxai_mjlab, CUDA): how often the observation quaternions have
w < 0, and how often they flip sign between consecutive steps, in random-action
rollouts of the training config. Columns follow the actor layout used by the
sim tests: ee quat 19:23, object quat 27:31."""

import sys
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from tabletop_sim.tasks.grasp.env_cfg import CONFIGS_DIR, make_grasp_env_cfg

SCENE = CONFIGS_DIR / "scenes" / "tabletop_jenga_wxai.json"
TASK = CONFIGS_DIR / "tasks" / "grasp_curriculum_size_variants_friction_wxai.json"
EE_Q, OBJ_Q = slice(19, 23), slice(27, 31)
steps = int(sys.argv[1]) if len(sys.argv) > 1 else 300

cfg = make_grasp_env_cfg(SCENE, TASK, None, 256)
cfg.seed = 0
env = ManagerBasedRlEnv(cfg, device="cuda:0")
try:
    obs, _ = env.reset()
    stats = {k: {"w_neg": 0, "flips": 0, "n": 0, "min_abs_w": 1.0} for k in ("ee", "obj_actor", "obj_critic")}
    prev = None
    for _ in range(steps):
        cur = {"ee": obs["actor"][:, EE_Q], "obj_actor": obs["actor"][:, OBJ_Q],
               "obj_critic": obs["critic"][:, OBJ_Q]}
        for k, q in cur.items():
            s = stats[k]
            s["w_neg"] += int((q[:, 0] < 0).sum()); s["n"] += q.shape[0]
            s["min_abs_w"] = min(s["min_abs_w"], float(q[:, 0].abs().min()))
            if prev is not None:
                # Same rotation in consecutive steps but opposite sign: a representation flip.
                s["flips"] += int(((q * prev[k]).sum(-1) < -0.5).sum())
        prev = {k: v.clone() for k, v in cur.items()}
        act = torch.rand(env.num_envs, env.action_manager.total_action_dim, device="cuda:0") * 2 - 1
        obs, *_ = env.step(act)
    for k, s in stats.items():
        print(f"{k:10s} w<0 in {s['w_neg']}/{s['n']} samples, sign flips between steps {s['flips']}, "
              f"min |w| {s['min_abs_w']:.3f}")
finally:
    env.close()
