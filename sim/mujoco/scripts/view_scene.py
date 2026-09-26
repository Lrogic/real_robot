"""Check a scene config: print a model summary, then either run the task env
with a zero / random agent, or open the plain MuJoCo viewer on the reset state.

  python scripts/view_scene.py [--scene JSON] [--task JSON] [--agent zero|random]
                               [--viewer auto|native|viser] [--static] [--no-viewer]
"""

from __future__ import annotations

import argparse

import mujoco
from mjlab.scene import Scene, SceneCfg
from mjlab.scripts.play import PlayConfig, run_play
from tabletop_sim.embodiments import build_entity_cfg, get_embodiment
from tabletop_sim.scene import ACTION_SPACES, load_scene_config, scene_entity_cfgs
from tabletop_sim.tasks.grasp import register_play_task
from tabletop_sim.tasks.grasp.env_cfg import DEFAULT_SCENE, DEFAULT_TASK


def main() -> None:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument("--scene", default=str(DEFAULT_SCENE), help="scene JSON")
  parser.add_argument("--task", default=str(DEFAULT_TASK), help="task JSON")
  parser.add_argument("--action-space", choices=ACTION_SPACES, default=None)
  parser.add_argument("--agent", choices=("zero", "random"), default="zero")
  parser.add_argument("--num-envs", type=int, default=1)
  parser.add_argument("--viewer", choices=("auto", "native", "viser"), default="auto")
  parser.add_argument("--static", action="store_true", help="plain MuJoCo viewer, no env")
  parser.add_argument("--no-viewer", action="store_true", help="only print the summary")
  args = parser.parse_args()

  scene_cfg = load_scene_config(args.scene)
  emb = get_embodiment(scene_cfg.robot.embodiment)
  entities = scene_entity_cfgs(scene_cfg, build_entity_cfg(emb))
  model = Scene(SceneCfg(num_envs=1, entities=entities), device="cpu").compile()

  print(f"scene '{scene_cfg.scene_name}' with embodiment '{emb.name}'")
  print(f"  entities: {list(entities)}")
  print(f"  nbody={model.nbody} njnt={model.njnt} ngeom={model.ngeom} nu={model.nu} ncam={model.ncam}")
  for obj in scene_cfg.objects:
    body = model.body(f"{obj.name}/{obj.name}")
    print(f"  {obj.name}: mass={float(body.mass[0]):.4f} kg, {len(obj.initial_poses)} initial pose(s)")

  if args.no_viewer:
    return
  if args.static:
    import mujoco.viewer

    data = mujoco.MjData(model)
    if model.nkey > 0:
      mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    mujoco.viewer.launch(model, data)
    return

  task_id = register_play_task(args.scene, args.task, args.action_space)
  run_play(task_id, PlayConfig(agent=args.agent, num_envs=args.num_envs, viewer=args.viewer))


if __name__ == "__main__":
  main()
