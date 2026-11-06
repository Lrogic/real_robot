# Hardware control vs the training controller

Small experiments that put the same 20 Hz TCP-target sequences through

- **sim**: the mjlab training env and its `ClippedDifferentialIKAction` (the controller the policy was trained with),
- **hardware, Cartesian backend**: the Trossen driver's `set_cartesian_positions` (position mode, nonblocking),
- **hardware, external-IK backend**: the *training* IK action evaluated on measured hardware state, sending all six joints with `set_arm_positions`,

and compare the resulting TCP motion offline. Nothing here runs the policy or the vision pipeline. No command moves the arm unless you pass `--execute`.

## Environment

| Command | Environment |
|---|---|
| `plans.py`, `sim_reference.py`, `compare.py`, tests | `wxai_mjlab` (or `wxai_mjlab_hw`) |
| `run_hardware.py --backend external_ik`, `diagnose.py --with-ik` | `wxai_mjlab_hw` |
| `run_hardware.py --backend cartesian`, `diagnose.py` | `wxai_mjlab_hw` or `fp_robot` |

`wxai_mjlab_hw` is a clone of `wxai_mjlab` with `trossen-arm==1.11.0` added. `fp_robot` has no mjlab, so external IK cannot run there. Run everything from this directory. The sim needs CUDA by default (`--device cpu` works but is about 5x slower).

## Files

| File | Purpose |
|---|---|
| `config.yaml` | Every experiment setting (motion sizes, timing, conditions, safety, reset). |
| `plans.py` | Generates and saves the absolute target plans from the training-model ready TCP. Plans carry a sha256 that every result records. |
| `targets.py` | Per-test target logic, shared by sim and hardware (single, scheduled, relative-from-current). |
| `geometry.py` | Pose helpers. Rotations compose in the base frame: `R_target = exp(r) R`, as in mjlab `apply_delta_pose`. Also `bounded_step`. |
| `fk.py` | Plain-MuJoCo FK and Jacobian of the training robot model (`ee_site`, robot base frame). |
| `ik_adapter.py` | Drives the training IK action with an absolute target on given joint state; `self_check` proves it gives the action's own joint targets. |
| `sim_reference.py` | The training env (domain randomization off) replaying `env.step`'s inner loop with 20 Hz target updates; logs every 2 ms substep. |
| `hardware.py` | Driver wrapper (reuses `manipulation/policy/arm.PolicyArm`), safety checks, the two backends, the single-threaded trial loop, reset-to-ready. |
| `timing.py` | Monotonic deadline scheduling without catch-up, state-message freshness tracking, interval stats. |
| `run_hardware.py` | Hardware trial CLI. Dry run by default. |
| `diagnose.py` | Read-only state-rate diagnostic and IK compute timing. |
| `compare.py` | Offline sim-vs-hardware comparison: plots, metrics JSON/CSV. |
| `records.py`, `settings.py` | Result files, units, config loading. |
| `tests/` | pytest suite (see Validation). |

Results go to `results/{plans,sim,hardware,diagnostics,compare}/`. `results/synthetic/` holds the offline check described under Validation; those files are **not** hardware data.

## What the training controller actually does

Read from mjlab 1.6.0 and confirmed in sim:

- `process_actions` runs once per 50 ms policy step. It turns the clipped action into an absolute desired pose: `p + a·0.01 m` and `exp(a·0.05 rad) R`, both in the base frame, both relative to the TCP *at that moment*.
- `apply_actions` runs on every 2 ms physics substep (25 per policy step). Each call is one damped-least-squares solve (damping 0.05, position and orientation weights 1, joint-limit rows on the 0.85 soft limits, posture weight 0). Each joint increment is clamped to ±0.5 rad, and the joint target is set to `q + dq`. There is no dt, no gain on the step, and no integration. The only clipping is the action's [-1, 1] clamp and `max_dq`.
- The joint-position actuators are PD (kp/kv 128/8 for joints 0–2, 80/1, 40/1, 40/1), with gravity compensation on every link.
- After `mj_step`, the site pose and Jacobian come from the *previous* substep's qpos until the next `forward`. Only the first substep after `process_actions` sees fresh kinematics. `sim_reference.py` reproduces this exactly.

Consequences visible in the sim reference:

- **A 10 mm step reaches its joint target in one solve.** The joint target jumps almost to the final solution on the first substep. The motion that follows is the PD actuators' response, not a trajectory. Because the wrist (kp 40, light) responds faster than the shoulder/elbow, a 10 mm +z single move swings the TCP about **8 mm in −x** before it comes back, reaching 10.0 mm on axis after about 0.2 s.
- **The `joint_3` soft limit restricts upward motion at the ready pose.** The ready pose has `joint_3 = −1.26` and the soft limit is −1.335. With the tool kept pointing down, the training IK can raise the TCP only about 20 mm. `relative_tra_z+` (40 mm goal) and `relative_rot_y+` (same wrist direction) therefore time out in sim with about 20 mm residual. The other 10 relative trials reach their goal. This is the training controller's real behavior and it is not hidden. The external-IK backend will show the same limit on hardware; the Cartesian backend (driver limits ±π/2) will not.

## Tests and timing definitions

All poses are the TCP in the **robot base frame**. Lengths are in m, angles in rad, and times in s (units are recorded in every result).

- **Command onset (t = 0)** is the first 20 Hz deadline. The outer update `k` is due at `k · 50 ms`. The plotted time axis is time since onset, with no latency shift.
- **single**: one absolute target, `reference ⊕ step`, held for `single.observe_s`. The Cartesian backend sends it once. External IK re-solves toward it at every IK tick.
- **scheduled**: a precomputed absolute sequence, sent every 50 ms: 4 steps out, 4 back, 10 holds. Every target is sent even when it repeats.
- **relative**: every 50 ms, a bounded step from the *measured* TCP toward a goal 0.04 m or 0.20 rad away. `step_limit: per_axis` clips each base-frame component, like the policy action. `euclidean` limits the norm and the angle. For single-axis goals the two are identical. When the remaining error is within the limit the step is exactly the goal. The trial stops at `goal_reached` (within 1 mm and 0.01 rad, followed by 1 s of observation) or at `timeout` (4 s). Each intermediate target is logged in `outer_updates` with its step info.

**Scheduling.** Deadlines come from `time.monotonic()`. A late tick runs once for the current slot, and slots that were missed entirely are skipped and counted; there are no bursts. The results record the achieved outer and loop rates, lateness, and skipped indices.

**Freshness.** A state sample is fresh when `header.id` changes. Repeats and id gaps are counted. If no new message arrives for 50 ms the trial stops.

## Safety

Workspace box, target offset and rotation from the reference, a measured-pose box with slack, tracking error, joint speed, joint-limit margins, and `max_joint_step` for external-IK commands. A stale state, a driver error, or `max_trial_s` stops the trial with a joint-space hold at the measured pose (goal_time 0.2). Ctrl+C or an unexpected exception switches the arm to gravity compensation (`PolicyArm.kill`).

Each trial starts with a separate, slow, blocking reset to the ready joints, followed by a settling check. A reset longer than 1 rad needs a typed confirmation. Unsupported conditions are rejected before anything connects:

- unknown space or goal_time,
- Cartesian options passed to external IK,
- IK rates at or below 20 Hz or above 1000 Hz,
- IK rates whose measured p95 solve time exceeds half the period,
- plans outside the safety box.

Gains, limits, and persistent driver configuration are never written. The driver configuration is read and saved with every hardware result.

## First commands

```bash
conda activate wxai_mjlab_hw
cd real_robot/sim2real/experiments/hardware_control

# 0. Tests (no hardware).
python -m pytest tests -q

# 1. Plans and sim reference (already generated once; rerun after config changes).
python plans.py --suite basic
python sim_reference.py results/plans/*.json

# 2. Read-only timing diagnostic: hardware state rate + training IK timing.
#    The arm goes to gravity compensation; support it.
python diagnose.py --with-ik

# 3. Dry run, then one Cartesian trial.
python run_hardware.py results/plans/single_tra_z-.json --backend cartesian --goal-time 0.2 --space joint
python run_hardware.py results/plans/single_tra_z-.json --backend cartesian --goal-time 0.2 --space joint --execute

# 4. The goal_time x space sweep (10 conditions, prompts per trial; 'a' runs the rest).
python run_hardware.py results/plans/scheduled_tra_x+.json --backend cartesian --sweep --execute

# 5. External IK at 500 Hz (IK self-check and timing run first, also in the dry run).
python run_hardware.py results/plans/single_tra_z-.json --backend external_ik --execute
python run_hardware.py results/plans/relative_tra_x+.json --backend external_ik --ik-hz 500 250 100 --execute

# 6. Compare (newest sim result of the same plan is picked automatically).
python compare.py results/hardware/single_tra_z-__*.json
```

`compare.py` writes a four-panel PNG to `results/compare/`:

1. displacement along the commanded axis (with the commanded targets),
2. off-axis error,
3. hardware − sim position and orientation discrepancy,
4. TCP speed.

It also writes JSON and CSV with these metrics:

- displacement after 50 ms
- overshoot
- settling time
- peak joint speed
- peak TCP speed
- final goal error (relative test)
- mean and maximum discrepancy over the overlapping interval only
- achieved outer rate
- driver-TCP-vs-FK residual

For scheduled plans it also checks that every hardware condition received the identical target sequence.

## Known mismatches and limitations

- **TCP model.** The suite uses the training model's `ee_site` (0.156 m from the flange). The earlier TCP_alignment work used the menagerie model (0.156062 m, which equals the driver's `t_flange_tool`). Hardware results log the driver's own TCP next to the training-model FK of the measured joints, together with their residual. Comparisons use the training FK on both sides.
- **Gripper.** Held at 0.026 m in position mode in both sim and hardware. The policy commands it binarily (0.04/0.0), which is out of scope here. There is no payload.
- **Nominal sim.** Domain randomization is off: nominal friction, masses, and actuator gains. Driver PID gains are not MuJoCo kp/kv, so equal gains are not expected to give equal responses.
- **Kinematics freshness.** On hardware each external-IK solve uses the freshly measured state and fresh FK (`sim.forward()` before `apply_actions`). The training substeps 2–25 use one-substep-old kinematics. At 500 Hz the effect is small but nonzero.
- **IK rate.** The training IK solve takes 0.71 ms median and 0.82 ms p95 on the RTX 5090, and 3.4 ms / 3.7 ms on the CPU. 500 Hz is sustainable only on the GPU, and the CPU supports up to roughly 130 Hz under the half-period rule. Each solve includes the state write and readback.
- **Driver replacement semantics.** What `set_cartesian_positions` / `set_arm_positions` do when a new goal arrives during an unfinished interpolation (restart from the current setpoint or from the measured state) is not documented in the stub. The Cartesian sweep with overlapping goal_times is designed to measure it; check the hardware data before interpreting it.
- **Send cost.** The diagnostic measures read cost only; send cost is measured only in real trials (`send_call_duration_s` in the backend summary, per-call durations under `commands`).

## Validation done (no hardware)

- `python -m pytest tests` passes all 31 tests: target math and conventions, the scheduler, the hardware loop against a fake driver and clock (sends, holds, reuse flags, overrun without catch-up, stale-state stop, Ctrl+C, joint-step rejection, unsupported conditions), and, in the mjlab env, preserved training config, adapter = action (1.2e-7 rad), no physics stepping in `solve`, held targets between updates, and a relative goal reached.
- Full sim suite (36 plans) on the GPU.
- `diagnose.py --ik-only` on the GPU and the CPU.
- `run_hardware.py` dry runs for both backends, plus the rejection paths.
- `compare.py` on synthetic "hardware" results. These were produced by the real trial loop and the training IK at 500 Hz, driving a joint plant with 10 ms delay and τ = 40 ms on a simulated clock (`results/synthetic/`).
