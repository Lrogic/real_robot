# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

This repo is sim-to-real work for a Trossen WidowX AI (WXAI) arm. A grasp/lift/hold PPO policy is trained in MuJoCo (mjlab 1.6.0 with MuJoCo Warp) and deployed on the real arm. On the real arm, the object pose comes from FoundationPose tracking a RealSense D405. The camera is mounted on a second, parked "follower" arm. The current target object is a jenga block, and the current work is closing the sim-vs-real gap in arm control.

## Workspace outside this repo

The repo lives at `/home/luka/research/real_robot`. Code reaches sibling directories of `/home/luka/research` by relative path:

- `FoundationPose/` is a checkout imported at runtime by `vision/pipeline/foundationpose/multi_object_fp_tracker.py` (`--foundationpose-dir`).
- `assets/meshes/` holds the object meshes. They are used by the setup stage's `--meshes-dir` and by the jenga scene JSON's `asset_path`.
- `trossen_arm/` is the driver source (Python package `trossen_arm`).
- HF SAM weights load offline from `/bigdata/luka/models/hf`.
- `sim/mujoco/logs/rsl_rl/` (git-ignored) holds the training runs. `manipulation/policy/jenga/config.yaml` points at a checkpoint there.

## Conda environments

No single environment runs everything. Each script's docstring or `run.sh` names the env it needs. Interpreters are at `/opt/conda-envs/<env>/bin/python`.

| Env | Key packages | Used for |
|---|---|---|
| `wxai_mjlab` | mjlab, mujoco, torch, pytest | Sim training/play, sim tests, policy parity tests, sim-side experiments (`sim/mujoco/environment.yml`) |
| `wxai_mjlab_hw` | `wxai_mjlab` + `trossen-arm` | `sim2real/experiments/hardware_control`, `test_ik_arm.py` |
| `vision` | FoundationPose, HF SAM, pyrealsense2, trossen_arm, mujoco. **No mjlab, no pytest.** | `vision/run.sh` (camera pipeline and on-robot policy), vision unittests |
| `fp_robot` | trossen_arm, pyrealsense2, OpenCV 5 | Calibration capture, hardware-side sim2real experiments |
| `sam3` | OpenCV 4.x | Offline calibration (`review`/`calibrate` need `calibrateHandEye`, which OpenCV 5 dropped) |

## Commands

### Simulation (run from `sim/mujoco`, env `wxai_mjlab`)

```bash
python scripts/train.py --scene configs/scenes/tabletop_jenga_wxai.json \
    --task configs/tasks/grasp_curriculum_size_variants_friction_wxai.json \
    --agent.experiment-name NAME [--agent.max-iterations N] [--agent.logger wandb]
python scripts/play.py --scene ... --task ... --checkpoint-file logs/rsl_rl/<exp>/<run>/model_N.pt
python scripts/view_scene.py --scene ... --task ... --agent zero|random [--static]
python -m pytest tests                                  # tests that build the env skip without CUDA
python -m pytest tests/test_reward_terms.py -k NAME     # single test
```

Each script parses its own flags first. The remaining flags go to mjlab's `TrainConfig`/`PlayConfig` (tyro). Logs are written to `logs/rsl_rl/<experiment_name>/` under the cwd. The default scene and task are the cube and `grasp_standard_wxai.json`.

### Vision pipeline and on-robot policy (`vision/run.sh`, env `vision`)

```bash
cd vision
./run.sh --setup-track --save-path foundationpose_runs/run1 --recovery [--reset]   # mask objects on one frame, then track live
./run.sh --track --save-path foundationpose_runs/run1 --recovery --manipulation policy \
    --policy-control shadow [--manipulation-logging]
```

`run.sh` activates the env, sets `PYTHONPATH=real_robot:real_robot/vision`, and forwards every flag to `pipeline.setup_stage` and/or `pipeline.track_stage`. Its header comment is the flag reference. `--manipulation policy` requires `--recovery`, because pose trust comes from the SAM verdicts. Run folders (`vision/foundationpose_runs/`) are git-ignored. Their layout is in `pipeline/run_folder.py`. With `--manipulation-logging`, the run folder also gets `policy_log.jsonl`, which `sim2real/experiments/smooth_control_test` analyzes.

### Calibration (`vision/calibration/run.sh`)

This is a ChArUco eye-to-hand calibration of the D405 against the manipulator base. `verify`, `record-follower` and `collect` touch hardware and run in `fp_robot`. `review`, `calibrate` and `generate` run offline in `sam3`. The result is `data/sessions/<name>/T_base_camera.json`. The pipeline and the policy use the newest session that has one, unless `--calibration` is passed. Only `sessions/20261009_combined/session.json` is tracked in git.

### Other tests

```bash
# manipulation/policy: pytest. The tests add real_robot/ to sys.path themselves.
cd /home/luka/research && /opt/conda-envs/wxai_mjlab/bin/python -m pytest real_robot/manipulation/policy/tests
#   test_ik_arm.py skips without trossen_arm (use wxai_mjlab_hw). test_obs_parity.py needs CUDA.
#   test_actor.py skips if the checkpoint in jenga/config.yaml is missing.

# vision: unittest, since the vision env has no pytest. Run from vision/.
cd vision && /opt/conda-envs/vision/bin/python -m unittest tests.test_recovery tests.test_tracking_loop -v
#   GPU + FoundationPose checks on a saved run folder: python -m tests.gpu_checks.check_recovery --save-path RUN

# hardware_control experiment: no hardware needed. Run from its directory.
cd sim2real/experiments/hardware_control && /opt/conda-envs/wxai_mjlab/bin/python -m pytest tests -q
```

The repo has no lint or format config. Don't run `ruff format` on `sim/mujoco`: it would reindent the 2-space code.

## Architecture

### Real-robot loop

Everything runs in one process: `pipeline.track_stage` → `pipeline.tracking_loop.track_live`. Each loop iteration does the following:

1. Read the newest RealSense frameset.
2. FoundationPose tracks every object (`multi_object_fp_tracker`: one estimator per object, shared networks; `--fp-mode` picks sequential or batched).
3. Optionally, SAM recovery (`pipeline/recovery/`) runs at `sam_fps`. It compares each SAM mask with the FP mesh silhouette and re-registers an object after `failure_limit` consecutive mismatches.
4. Manipulation hooks (`pipeline/robot_arm/manipulation_hooks.py`: `on_mouse`, `update`, `on_key`, `draw`, `close`) plug in `manipulation/basic_manipulation` (click-to-follow test) or `manipulation/policy`.

`manipulation/policy/session.py` runs the policy at the training control rate (20 Hz). Each tick goes through these modules:

- `trust.py`: pose trust from the SAM verdicts.
- `jenga/observation_manager.py`: episode lifecycle, the box-symmetry choice against the training reset orientation, and the lift goal.
- `observations.py`, then `actor.py` (the rsl_rl MLP and normalizer, evaluated in numpy).
- `actions.py`: the absolute TCP goal.
- `controller.py`: the state machine and command gate (shadow/step/slow/full, holds on stale frames or untrusted poses).
- The arm. `control.ik_solver` picks how the goal reaches it:
  - `driver`: `arm.py` sends the goal to the driver's Cartesian IK.
  - `sim`: `ik_arm.py` re-solves the training IK from the measured joints at `sim_ik.inner_hz` and sends joint positions.

The object pose in the base frame is `T_base_camera @ center_pose @ to_origin`.

### Sim ↔ real parity (the key invariant)

The deployment re-implements the trained env's interface in plain numpy/mujoco instead of importing mjlab, because the `vision` env has no mjlab. Parity tests check each copy against the real mjlab code:

| Real side (`manipulation/policy/`) | Mirrors (`sim/mujoco/src/tabletop_sim/`) | Checked by |
|---|---|---|
| `observations.py` | Actor group of `tasks/grasp/env_cfg.py::_observations` and `mdp/observations.py` | `test_obs_parity.py` |
| `actions.py` | `ClippedDifferentialIKAction` (relative mode) and binary `GripperAction` in `mdp/actions.py` | `test_actions.py` |
| `ik.py` | mjlab `DifferentialIKAction.apply_actions`: one damped least-squares solve per 2 ms substep, 25 per policy step | `test_ik_parity.py` |
| `actor.py` | rsl_rl `MLPModel`. It refuses to load if the run's `params/env.yaml`/`agent.yaml` disagree with the deployment. | `test_actor.py` |
| `jenga/config.yaml` `embodiment`/`sim_ik` | `EmbodimentCfg`/`IkDefaults` of `trossen_wxai_base_real` | `test_obs_parity.py`, `test_ik_parity.py` |

A change to sim observations, actions, action scales, control rate, embodiment or IK settings needs the matching change on the real side, and the parity tests rerun in `wxai_mjlab`. `jenga/config.py` derives control_hz, action scales, lift height and reset pose from the same scene and task JSONs the policy was trained on.

### Simulation package (`sim/mujoco/src/tabletop_sim`)

- Environments are built from config files: a **scene JSON** plus a **task JSON**, passed to `tasks/grasp/env_cfg.py::make_grasp_env_cfg`, which returns an mjlab `ManagerBasedRlEnvCfg`.
  - The scene JSON sets the embodiment, action space, base pose, objects and support surface. Its schema is in the docstring of `scene/config.py`.
  - The task JSON sets timing, action scales, rewards, domain randomization, curriculum, pose noise and size variants.
  - Unknown keys raise `ConfigError`.
- Importing `tabletop_sim.tasks` registers one mjlab task per action space (`TASK_IDS` in `tasks/grasp/__init__.py`). `play.py` registers `WXAI-Grasp-Play` from any scene/task pair.
- Embodiments (`embodiments/registry.py`) are the menagerie and Trossen WXAI models. Each has a `_real` variant whose `*_real_robot.xml` carries measured position-mode gains. The jenga scene uses `trossen_wxai_base_real`.

### sim2real experiments (`sim2real/experiments`)

These are standalone scripts. Each docstring says which env to launch from. `common/` holds shared driver, pose, prompt and IO helpers. `hardware_control/README.md` describes in detail what the training controller actually does: IK once per substep with no trajectory, the PD actuator response, and the `joint_3` soft limit at the ready pose. Read it before changing arm control. `hardware_control/results/` and the `object_frame` renders are git-ignored. The other experiments commit their result JSONs.

## Conventions

- Quaternions are `(w, x, y, z)` with `w >= 0`. Pose JSON is `[x, y, z, qw, qx, qy, qz]`.
- Poses are in the robot base frame unless the name says otherwise. The table top is base z = 0.
- Transforms are named `T_<to>_<from>` (for example, `T_base_camera`). Units are m, rad and s.
- `sim/mujoco` uses 2-space indentation (mjlab style). Everything else uses 4 spaces.
- Relative paths inside YAML configs resolve against the YAML file's own directory.
- The manipulator is at `192.168.1.2` and the follower (camera) arm at `192.168.1.3` (`vision/calibration/config.yaml`).
- Hardware experiments are dry runs unless `--execute` is passed. The policy session starts in gravity compensation and moves only on the `s`/`r` keys. But `jenga/config.yaml` sets `control.mode: full`, so pass `--policy-control shadow` or `--manipulation-dry-run` to never command the arm. Run anything that commands an arm only when the user asks for it and is at the arm.
