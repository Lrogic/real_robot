# Sim-to-real arm control: findings (November 2026)

Written 2026-11-12. This records what was tried between 2026-11-10 and 2026-11-12 to make the
jenga grasp policy move smoothly at full speed on the real WidowX AI arm, what was measured,
and what was kept. The companion notes [block_height_offset.md](block_height_offset.md) and
[arm_joint_offsets.md](arm_joint_offsets.md) cover the block-height error found along the way: the
depth camera's part and the arm's part.
[Holding a position](#holding-a-position-the-friction-deadband) was added on 2026-11-16, and
[The gripper's grip](#the-grippers-grip-squeeze-force-and-contact-softness) on 2026-11-18.

## Bottom line

- **The original policy is still the best one.** It is the run
  `grasp_jenga_DR_adjusted_gripper_width_with_grav_comp_real_robot_parameters_higher_gain_adjusted_quat/2026-10-28_14-55-03`,
  trained on what is now `sim/mujoco/configs/tasks/grasp_wxai.json`.
- **Its smoothest real setting needs no retraining:** `control.ik_solver: sim`,
  `sim_ik.goal_time_s: 0.05` (a joint goal time), `sim_ik.interpolate_goal: false`. It lifted
  4 of 5. `manipulation/policy/jenga/config.yaml` is set to this.
- **Every sim addition trained since then gave a worse policy.** The options still exist and
  are off in the task file.
- **The arm's big joints have far more dry friction than the model.** That, a 6 ms command
  delay, and how the driver treats a goal time were measured in
  `sim2real/experiments/joint_step`.

## Terms used below

These are easy to mix up, so each is defined once here.

**Policy tick / control rate.** The policy outputs one action per tick: a small move of the
gripper tip (TCP) relative to where it is now, at most 10 mm and 0.05 rad, plus open or close.
Training runs at 20 Hz, so a tick is 50 ms. The 10 Hz runs use 100 ms ticks.

**IK solver** (`control.ik_solver`). How the TCP goal becomes joint motion on the real arm.

- `driver`: the TCP goal is sent to the Trossen driver, which runs its own IK
  (`set_cartesian_positions`).
- `sim`: the host re-runs the training IK from the measured joints every 2 ms (500 Hz,
  `sim_ik.inner_hz`) and sends six joint positions (`set_arm_positions`). This is what the
  simulator does: one IK solve per 2 ms physics step, 25 per tick.

**Goal time.** The time the driver is told to take to reach a command. There are three, and
they are separate settings:

| Name | What it applies to | Setting | Used when |
|---|---|---|---|
| Joint goal time | joint position commands | `sim_ik.goal_time_s` | `ik_solver: sim` |
| Cartesian goal time | TCP pose commands | `control.driver_goal_time_s` | `ik_solver: driver` |
| Gripper goal time | the open/close command | `control.gripper_goal_time_s`; in sim `action.gripper_goal_time_s` | always |

The driver applies a goal time of 0.001 s or less immediately, interpolates linearly up to
0.2 s, and uses a quintic above that.

**Re-sent versus send-once.** With `ik_solver: sim` a new joint command goes out every 2 ms.
With a 50 ms joint goal time the driver therefore never finishes a move: each command restarts
it. "Send-once" means a command is sent a single time and its move runs to the end. The policy
loop always re-sends; send-once was only tried in the joint-step experiment.

**IK goal interpolation** (`sim_ik.interpolate_goal`; in sim `action.ee_goal_interpolation`).
Works on the TCP goal, before the IK. Off: every IK solve in a tick aims at the tick's final
goal. On: the solves aim at a subgoal that slides in a straight line from the TCP's pose at
the start of the tick to the goal, arriving at the end of the tick. It is not a driver goal
time, and the joint commands it produces are still sent with whatever joint goal time is set.

**Control modes** (`control.mode`). `full` runs at the training rate. `slow` stretches each
tick by `slow_time_dilation` (4). `shadow` computes everything and never commands the arm.

## Policies trained

All on the jenga scene. "Edits" are relative to `configs/tasks/grasp_wxai.json`; see
[The task file](#the-task-file) for the exact keys.

| Name here | Run folder under `sim/mujoco/logs/rsl_rl/` | Edits to the task | Outcome |
|---|---|---|---|
| Original (v1) | `grasp_jenga_..._higher_gain_adjusted_quat/2026-10-28_14-55-03` | none | About 100% on the real arm in slow mode. Works at full speed with the settings above. |
| Latency (v2) | `collision-delay-hold/2026-11-10_13-48-20` | pose delay, held actions, table-contact penalty | Real-arm outcome not recorded here. |
| Interpolation | `smooth_control-collision-delay-hold/2026-11-10_16-05-11` | v2 + IK goal interpolation | Gripper open/close flutter appears in sim. |
| Gripper (v3) | `gripper_modelling-smooth_control-collision-delay-hold/2026-11-10_17-00-53` | interpolation run + gripper goal time, switch penalty | Fails on the real arm: grips high, then gets stuck sliding down. Brittle in sim. |
| 10 Hz | `10HZ-gripper-interpolation-collision-hold-delay/2026-11-11_16-04-26` | v3 at 10 Hz | Real runs logged, not analysed here. |
| Joint model | `joint_friction_model-command_delay-goal_time_model/2026-11-11_17-21-17` | v2 + gripper + joint friction, command delay, joint goal time model | Learned to knock the block over, then pick it up. |
| Joint model, 10 Hz | `joint_friction_model-command_delay-no_joint_goal-interpolation-10Hz-.../2026-11-11_18-09-48` | 10 Hz, interpolation, gripper goal time, command delay, joint friction | Real run logged, not analysed here. |

Sim comparison of v1 and v3 (1024 envs, full randomization):

- Both grip near the block centre and slide down it. v1 slides more: 14.9 mm against 7.7 mm.
- v1 run in v3's environment held the block in 64% of episodes. v3 in v1's environment held
  it in 20%, with about 97 gripper open/close switches per episode. v3 only works in the
  environment it was trained in.
- The gripper flutter starts with interpolation. Every interpolation policy closes the
  gripper on step 0; none of the others do.

## Real-arm runs of the original policy at full speed

All with `ik_solver: sim`, 20 Hz, logs in `vision/foundationpose_runs/`. The millimetre
figures are TCP travel per 50 ms tick along each axis during the approach, as computed from
`policy_log.jsonl` at the time. The policy asks for up to 10 mm per tick.

| Run folder | Joint goal time | IK goal interpolation | x | y | z | Result |
|---|---|---|---|---|---|---|
| `old_policy-sim-full-50ms_goal_time` | 50 ms | off | 9.3 | 5.5 | 2.2 | Smooth; 4 of 5 lifted |
| `old_policy-sim-full-0ms_goal_time` | 0 | off | 14.1 | 5.2 | 1.7 | Jerky; 2 of 3 full episodes lifted |
| `old_policy-sim-full-0ms_goal_time-interpolate` | 0 | on | 7.7 | 2.7 | 0.3 | Clunkier and less successful; vertical motion stalls |

Earlier results, before these logs: slow mode (4x) works about 100% of the time, and full
speed with the driver's own IK hit the table.

### Why vertical motion is weak

The IK asks for the full 10 mm and no joint limit is involved. The joints respond unequally:
the wrist pitch joint completes about 100% of its step in a tick and the shoulder about 45%.
For a vertical move their contributions to height nearly cancel, so the TCP barely moves in z
until the shoulder catches up. This happens in sim too. On the real arm the elbow adds a dead
zone (below).

## What the driver does

Read from the driver API and the Trossen docs.

- **Position mode is a cascade.** A position loop produces a desired velocity; a
  feed-forward velocity is added; a velocity loop produces effort; a feed-forward effort is
  added.
- **The gains match the sim model.** Position gains 16/16/16/80/40/40 and velocity gains
  8/8/8/1/1/1 (joints 0 to 5), no integral or derivative. The model's single position
  actuator uses their product, 128/128/128/80/40/40, with the same velocity gains.
- **Friction compensation is by velocity sign.** `friction_transition_velocity` is 0 on every
  joint, so at standstill the friction term flips with the sign of the velocity noise and
  averages to about zero (a few tenths of a Nm either way; read 2026-11-16).
- **The shoulder holds 3.4 Nm at the ready pose and 7.4 Nm at the reach pose.**
- **The compensation effort is fed forward in position mode too,** and at standstill it is the
  sim model's gravity torque. See [Holding a position](#holding-a-position-the-friction-deadband).
- **Joint characteristics read on 2026-11-16** (`get_joint_characteristics`, manipulator):
  `effort_correction` 1.20 / 1.12 / 1.17 / 1.10 / 1.11 / 1.20, `position_offset` 0 on every
  joint, and small friction terms (constant -0.28 to 0.24 Nm, Coulomb 0 to 0.14 of the gravity
  effort). The Trossen docs say `effort_correction` is tuned against links being pulled down by
  gravity; they do not say whether it also scales the friction term.

## The joint-step experiment

`sim2real/experiments/joint_step`. One joint is commanded at a time while the other five are
told to hold. The loop reads the arm and sends `set_arm_positions` every 2 ms, as the policy's
sim IK loop does. The same commands are then replayed in a MuJoCo model of the arm.

**Poses.** `ready` is the policy's start pose (TCP 0.214, 0, 0.085 m). `reach` has the arm
stretched toward the block (TCP 0.36, 0, 0.08 m).

**Step sizes.** 5, 10, 20 and 40 mrad, both directions.

**Every goal time in this experiment is a joint goal time.** The Cartesian goal time was not
tested.

**Command styles:**

| Name in the code | What is sent |
|---|---|
| `step` | The target jumps once by the step size and stays. Re-sent every 2 ms. |
| `ramp` | The target slides to the new value over 50 ms, then holds. Re-sent every 2 ms. This is what IK goal interpolation produces for one joint. Run at joint goal time 0 only. |
| `relative` | Every 50 ms the target becomes "the joint's measured position plus the step size", for up to 8 ticks. Re-sent every 2 ms. This is the policy's own pattern, since each action is relative to the current pose. |
| `step_once`, `relative_once` | The same targets, but each new target is sent a single time, with a 50 ms joint goal time, so the driver's move finishes. |

**Measure.** The fraction of the commanded step that the joint covers in 50 ms (`step`,
`ramp`), or per 50 ms tick averaged over the ticks (`relative`). 1.0 means the joint gets all
the way there within one policy tick.

`summarize.py` in the experiment folder prints the tables below.

### Hardware: how each joint responds

Fraction of the step covered in 50 ms, at the ready pose / at the reach pose, averaged over
step sizes and directions. Commands re-sent every 2 ms.

| Command style | Joint goal time | Base | Shoulder | Elbow | Wrist pitch | Wrist 2 | Wrist 3 |
|---|---|---|---|---|---|---|---|
| `step` | 0 | 0.32 / 0.24 | 0.16 / 0.09 | 0.12 / 0.11 | 0.92 / 1.04 | 0.76 / 0.76 | 0.76 / 0.77 |
| `step` | 50 ms | 0.38 / 0.32 | 0.23 / 0.09 | 0.18 / 0.18 | 0.45 / 0.55 | 0.56 / 0.57 | 0.58 / 0.59 |
| `relative` | 0 | 0.36 / 0.31 | 0.21 / 0.09 | 0.14 / 0.14 | 1.02 / 1.04 | 0.76 / 0.76 | 0.69 / 0.69 |
| `relative` | 50 ms | 0.41 / 0.36 | 0.26 / 0.12 | 0.17 / 0.17 | 0.58 / 0.59 | 0.54 / 0.53 | 0.50 / 0.50 |
| `ramp` | 0 | 0.17 / 0.15 | 0.08 / 0.02 | 0.04 / 0.03 | 0.39 / 0.46 | 0.35 / 0.36 | 0.50 / 0.47 |

The same `step` test at joint goal time 0, by step size (ready / reach):

| Step size | Base | Shoulder | Elbow | Wrist pitch | Wrist 2 | Wrist 3 |
|---|---|---|---|---|---|---|
| 5 mrad | 0.15 / 0.11 | 0.04 / 0.00 | 0.00 / 0.00 | 0.88 / 1.03 | 0.53 / 0.53 | 0.61 / 0.65 |
| 10 mrad | 0.36 / 0.29 | 0.11 / 0.08 | 0.02 / 0.00 | 0.88 / 1.07 | 0.74 / 0.78 | 0.78 / 0.78 |
| 20 mrad | 0.40 / 0.29 | 0.21 / 0.08 | 0.15 / 0.16 | 0.93 / 1.02 | 0.88 / 0.87 | 0.81 / 0.83 |
| 40 mrad | 0.37 / 0.29 | 0.30 / 0.21 | 0.30 / 0.29 | 0.97 / 1.03 | 0.87 / 0.87 | 0.83 / 0.82 |

What this shows:

- **The three big joints are slow and the three wrist joints are fast.** In one tick the
  base covers about a third of a step, the shoulder and elbow a fifth or less, the wrist
  joints most of it.
- **The elbow ignores small steps.** It does not move for 5 or 10 mrad steps. Its dry
  friction is larger than the torque those steps produce.
- **The shoulder depends on the pose.** It is about half as responsive at the reach pose,
  where it carries twice the load. The base is 15 to 25% slower there. The elbow and the two
  wrist rotations are the same in both poses; the wrist pitch is slightly faster at reach.
- **A ramp covers much less ground per tick than a step,** which is why interpolation slows
  the arm.
- **Every joint has about 8 ms of dead time** between a command and the first motion, at
  joint goal time 0 as well as 50 ms.

### Fitting a model to it

`fit.py` fits per-joint parameters of the MuJoCo model to the hardware `step` and `relative`
trials at joint goal time 0, with the gains fixed at the driver's. Results are in
`results/fit_fitted.json`.

| | Base | Shoulder | Elbow | Wrist pitch | Wrist 2 | Wrist 3 |
|---|---|---|---|---|---|---|
| Dry friction, Nm (the model had 0.1 everywhere) | 0.64 | 1.85 | 1.92 | 0.10 | 0.09 | 0.16 |
| Command delay, 2 ms ticks | 2 | 2 | 5 | 3 | 2 | 2 |

- **Delay:** 3 ticks (6 ms) was used for all joints. It is present at every goal time; it was
  fitted on the goal-time-0 trials. It is measured from the host, so it includes the command
  reaching the motors and the position report coming back, and those cannot be separated.
- **Actuator lag:** fitted as roughly zero.
- **Armature and damping:** the fits were inconsistent between joints and were not used.
- **The friction model has one value per joint,** so it does not capture the shoulder's
  dependence on pose.

**How the driver treats a 50 ms joint goal time that is re-sent every 2 ms.** Three models
were scored against the hardware. The best on every joint: on each re-sent command the
driver's setpoint closes (2 ms / goal time) of the remaining gap, and the velocity of that
move (gap / goal time) is fed forward to the velocity loop. So the setpoint lags behind the
target and never finishes its move. At joint goal time 0 there is no feed-forward.

**Send-once with a 50 ms joint goal time overshoots.** Motion is smoother and faster (0.5 to
0.85 of a step per tick), but the joint settles past its target: at 2.25 times the step on the
three big joints, 1.25 times on the wrist pitch and 1.5 times on the other two wrist joints
(measured 2.27 to 2.46, 1.25 to 1.28, 1.50 to 1.53). Those factors equal
1 + 1 / (goal time x position gain), which is what a fed-forward velocity that persists after
the move would give. That explanation is inferred from the data; the driver is a prebuilt
library and its firmware source was not read. A send-once test with the feed-forward velocity
explicitly set to zero would check it and has not been run.

### Does the fitted model match the real arm?

Mean difference between hardware and sim in the fraction of a step covered per 50 ms, over
all six joints, both poses, all step sizes and directions. 0.19 means sim and real disagree by
19% of the commanded step on average.

- **Training model:** the model the original policy trained on: matched gains, 0.1 Nm
  friction on every joint, no delay. For the 50 ms rows its setpoint lags with no
  feed-forward.
- **Fitted model:** the same with the friction above, a 6 ms command delay, and the
  lag-plus-feed-forward treatment of the joint goal time.

| Command style | Joint goal time | Training model | Fitted model |
|---|---|---|---|
| `relative` (the policy's pattern) | 0 | 0.19 | 0.05 |
| `relative` | 50 ms | 0.115 | 0.03 |
| `step` | 0 | 0.14 | 0.10 |
| `step` | 50 ms | 0.21 | 0.11 |
| `ramp` (what IK goal interpolation sends) | 0 | 0.09 | 0.11 |
| `relative_once` (send-once) | 50 ms | not run | 0.05 |
| `step_once` (send-once) | 50 ms | not run | 0.15 |

- **Where it helps:** friction is a property of the arm, not of a control mode, and it
  improves the shoulder and elbow in every test. The gain is largest for the policy's own
  pattern.
- **Where it does not:** the ramp case is no better overall. The elbow improves (0.11 to
  0.03) but the base and wrist joints get worse. Do not expect the fitted model to describe a
  run with IK goal interpolation.
- **The goal-time treatment only covers a 50 ms joint goal time re-sent every 2 ms.** It says
  nothing about other goal times or about the Cartesian goal time.
- **It is not validated on the task.** These are single-joint tests at two poses. In a full
  rollout of the original policy in the fitted sim, the elbow moved 0.30 of its step per tick
  against 0.13 on the real arm, and only 19% of episodes held the block. The one policy
  trained with all three effects learned the knock-down behaviour.

## Holding a position: the friction deadband

Measured 2026-11-16 with `sim2real/sag_test/hold_z_sweep.py`
(`results/hold_z_20261116_134852.json`). The arm held the tool straight down at z = 100 mm
and stepped along +x from 0.25 m to 0.43 m and back in 45 mm steps. The joint targets came
from IK on the MuJoCo model and were sent with `set_arm_positions` (2 s goal time, so the
driver's quintic). Each stop was read after a 1 s settle, averaged over 0.5 s. The same run
measured the arm's height error ([arm_joint_offsets.md](arm_joint_offsets.md), section 2).

**The joints stop short of their targets in the direction they were moving.** Measured minus
target, in degrees:

| Reach | Shoulder, out | Elbow, out | Shoulder, back | Elbow, back |
|---|---|---|---|---|
| 0.250 m | -0.37 | -0.61 | +0.53 | +0.84 |
| 0.295 m | -0.62 | -0.74 | +0.63 | +0.64 |
| 0.340 m | -0.69 | -0.74 | +0.80 | +0.82 |
| 0.385 m | -0.58 | -0.71 | +0.80 | +0.64 |

Going out both angles increase, coming back both decrease, so the sign flip means the error
opposes the motion. Gravity would give the same sign on both legs; this is friction.

**Why they stay there.** Position mode has no integral term (above), so a held joint is pushed
by 128 Nm/rad x its error and nothing else. Once that is less than the joint's dry friction,
it stops for good. With the joint-step friction (1.85 Nm shoulder, 1.92 Nm elbow) the band is
about +-0.85 deg; the stops were 0.4 to 0.8 deg short. A joint that is not commanded to move
stays wherever it last stopped: the base read -0.23 deg for the whole run, its position after
the arm was guided by hand. The fingertip barely moved during the readings (at most 0.2 mm in
0.5 s), so nothing was creeping toward the target.

**What it does to the fingertip** (from the model's Jacobian at each stop; it adds up to the
logged pose errors to a few tenths of a millimetre):

- **x, along the motion: about 3 mm short,** on both legs. Only the elbow moves the tip in x
  at these poses (about 4.5 mm per degree).
- **z: within about +-2 mm, with no fixed sign.** A shoulder that stops short leaves the tip
  1.5 to 4 mm high, an elbow that stops short leaves it 2.5 to 3.5 mm low, and the two mostly
  cancel.
- **y: -1 to -2 mm,** growing with reach: the base's -0.23 deg.

The joint readings see all of this, so it shows in the policy's TCP and joint observations.

**The compensation is fed forward in position mode.** The driver reports the effort, the
compensation effort and the external effort (effort minus compensation). At every stop the
external effort had the sign of 128 Nm/rad x the error and 0.82 to 0.92 of its size, so the
motor torque is the compensation plus the position loop's correction. The remaining 10 to 18%
is not explained; joint velocity was not logged, so the velocity term was assumed zero.

**The compensation is the sim model's gravity.** At standstill the driver's compensation
effort matched the gravity torque of `wxai_base_real_robot.xml` (MuJoCo, at the logged joints)
to 0.03 Nm on the shoulder and elbow at every stop: shoulder -4.7 to -9.6 Nm with reach. The
elbow's stayed at 5.4 Nm because holding the tool straight down at a fixed height kept the
forearm's tilt almost constant (4 to 17 deg); the model agrees.

**In the sim.** The base model's 0.1 Nm friction gives a band of about +-0.05 deg. The fitted
model (`wxai_base_real_robot_friction.xml`) has the measured friction and so the real band
width. The original policy was trained on the base model.

## The gripper's grip: squeeze force and contact softness

Measured 2026-11-18. On the real arm a held block does not slip down in the gripper, and
pushing down on a gripped block resting on the table does not slide the fingers down it. In
the training model both happen. Scripted tests in the Warp training env
(`sim2real/experiments/gripper_force/run_sim.py`,
`results/gripper_sim_20261118_185255.json`; medians over 32 envs, every block size variant):

| Model | Squeeze | Block slip, 4 s hold | Fingers slide when pushing 2 / 5 / 10 mm per tick for 1.5 s |
|---|---|---|---|
| `trossen_wxai_base_real` (original policy) | 4.0 N | 7.4 mm | 6.4 / 13.1 / 17.0 mm |
| `trossen_wxai_base_real_friction` | 4.0 N | 7.0 mm | 3.4 / 7.0 / 11.5 mm |
| `trossen_wxai_base_real_friction_grip` | 83.3 N | 0.07 mm | 0.1 / -0.1 / -0.7 mm |

**The real gripper squeezes at its effort limit.** `sim2real/experiments/gripper_force/run_hardware.py`
(`results/gripper_force_20261118_181829.json`) closed the gripper on a jenga block held across
its 25 mm face, at targets of 2, 6 and 10 mm (half the jaw gap). Every close reported the same
-83.49 N from about 20 ms after contact to the end of the 3 s log, even at the 10 mm target
where only 0.9 to 1.5 mm of position error was left. That is the limit 100 N /
`effort_correction` 1.2 = 83.3 N (the WXAI spec: the motor effort is the effort in N times
`effort_correction`, and the motor is capped at 100); the reported effort comes in steps of
about 0.47 N. Closing on nothing, the gripper reached its target and reported -7.7 N there, a
no-load baseline; the squeeze on the block is the 83 N (about 76 N if the baseline is an offset
in the reading). The 83 N is the driver's estimate from motor current; nothing measured the
force at the fingers.

The gripper's position-mode gains (position 20, velocity 1) are per motor radian. Through the
8.75 mm pinion radius they are a stiffness of 20 x 1 / r^2 = 2.6e5 N/m and a damping of
1.3e4 N s/m; the no-block close (32.6 um short of target at 7.7 N, against 8.5 N predicted)
agrees. The sim gripper was kp 339 N/m: 3.8 N on the block, about 22 times less.

**Why the sim slipped.** Two things, and raising friction fixes neither: sliding friction
0.5, 1.0 and 1.5 gave the same slip.

- **Friction creep.** MuJoCo friction is a soft constraint, so a steady sideways load gives a
  steady slip velocity even well inside the friction cone. With pyramidal cones (mjlab's
  default) friction is as soft as the contact. Elliptic cones give friction its own
  constraint rows, and `impratio` makes them that many times stiffer. Hold-test creep on CPU:
  6.7 mm (pyramidal, impratio 1), 0.27 mm (elliptic, 10), 0.03 mm (elliptic, 100).
- **The weak squeeze.** With 3.8 N a downward push slides the fingers or twists the block in
  the grip.

**Raising the squeeze alone breaks the grasp.** At 83 N with default contacts (solref 0.02,
solimp 0.9 to 0.95) the fingers sink about 9 mm into the 22.5 mm block, the actuator never
reaches its clamp, and the soft carriage mimic lets the fingers drift 3.6 mm apart; with the
pyramidal cone the block was dropped. So the finger contacts and the mimic are stiffened too.

**The grip model** (`wxai_base_real_robot_friction_grip.xml`, embodiment
`trossen_wxai_base_real_friction_grip`, scene `tabletop_jenga_wxai_friction_grip.json`) is the
joint-friction model with:

| Change | Value | Source |
|---|---|---|
| Gripper actuator | kp 1e4, kv 85, force range +-83.3 N | the measured limit; kp reaches it 8.3 mm past contact, which the closed target 0 always is on the block |
| Finger collision boxes | priority 1, solref 0.01 1, solimp 0.95 0.99 0.001 | solref of the Menagerie WXAI and ALOHA grippers, solimp of the Menagerie Robotiq 2F-85 pads; priority makes the pads' values win over the object's instead of averaging |
| Carriage mimic equality | solref 0.005 1, solimp 0.95 0.99 0.001 | the Menagerie Panda and Robotiq finger equalities |
| Solver | elliptic cone, impratio 10 | the Menagerie WXAI, ALOHA and Robotiq models |

With these, at 83 N the fingers sink 0.75 mm and sit 0.44 mm apart. The stiffer mimic only
centres the block (2.5 mm finger offset without it); it does not change the slip.

- **kp is not the real 2.6e5.** That also ran stable and gave the same hold and push results,
  because the clamp keeps the force low while the gripper moves; 1e4 was kept as the milder
  value. The real damping cannot be copied: the driver damps against its planned velocity, a
  position actuator against the absolute one, and 1.3e4 N s/m oscillated at the 2 ms step.
- **Use it with `action.gripper_goal_time_s: 0.5`** (the real `control.gripper_goal_time_s`).
  Without it the 83 N gripper shuts in about 20 ms and hits the block at up to about 340 N
  (CPU test); with it the fingers reach the block about 0.3 s after the command, where the
  real gripper reached its limit 0.30 to 0.42 s after it.
- **Cost:** about 10% fewer env steps per second (90k against 100k at 4096 envs).
- **Stiffness randomization** scales the gripper kp by 0.8 to 1.2 too; 8000 N/m still reaches
  the clamp on the smallest block variant.
- **Not tested:** whether a policy trained on it behaves better on the real arm. The push
  tests rotate the gripper 10 to 55 degrees because the script holds no orientation; the
  block turns 1 to 4 degrees in the grip.

## What is in the simulator now

All off in `configs/tasks/grasp_wxai.json`.

| Option (task JSON key) | What it models | Value used in runs |
|---|---|---|
| `domain_randomization.object_pose_delay_s` | age of the block pose the policy sees | 0.01 |
| `domain_randomization.action_hold_probability` | ticks that keep the previous goal | 0.1 |
| `w_table_contact` | penalty for the robot touching the table | -1.0 |
| `action.ee_goal_interpolation` | IK goal interpolation | true |
| `action.gripper_goal_time_s` | gripper goal time | 0.5 |
| `w_gripper_switch` | penalty per open/close change | -0.2 |
| `action.command_delay_s` | delay of the IK joint targets | 0.006 |
| `domain_randomization.joint_friction` | scales the arm joints' dry friction at reset | enabled, 0.6 to 1.4 |

The fitted friction values live in the robot model
`wxai_base_real_robot_friction.xml` (embodiment `trossen_wxai_base_real_friction`), selected by
the scene `configs/scenes/tabletop_jenga_wxai_friction.json`. The randomization option scales
whatever friction the scene's model has, so it only means something with that scene.

The grip model `wxai_base_real_robot_friction_grip.xml` (embodiment
`trossen_wxai_base_real_friction_grip`, scene `tabletop_jenga_wxai_friction_grip.json`) is that
model with the gripper changes in [The gripper's grip](#the-grippers-grip-squeeze-force-and-contact-softness).

**Removed on 2026-11-12:** `action.joint_goal_time_s`, the sim model of the re-sent 50 ms
joint goal time. The only policy trained with it learned the knock-down behaviour, and the
real arm runs the original policy well with a 50 ms joint goal time that training never
modelled. The joint-step experiment keeps its own copy of the model (`sim_model.py`).
Checkpoints trained with that option (`joint_friction_model-command_delay-goal_time_model`)
can no longer be rebuilt in sim.

Things to know when changing the control rate to 10 Hz:

- The arm's speed halves unless `ee_pos_scale` and `ee_rot_scale` are doubled.
- `action_hold_probability` is per tick, so held time doubles.
- Rewards are scaled by the tick length, so a per-event penalty such as `w_gripper_switch`
  doubles per event; halve it to keep the same cost.
- `episode_steps` must be halved (150) to keep the episode length in seconds.

## The task file

`sim/mujoco/configs/tasks/grasp_wxai.json` is the only task file. It lists every option, and
as written it is the original policy's task. Every other configuration is this file with a
few values changed. A training run saves the task it used as `params/task.json` in its run
folder (runs from 2026-11-10 16:22 on).

To deploy a checkpoint on the real arm, `manipulation/policy/jenga/config.yaml` needs the
scene and task the checkpoint was trained on. For each of `sim.scene` and `sim.task`:

- **Set to a path:** that file is used.
- **Set to null:** the copy saved in the checkpoint's run folder is used (`params/scene.json`,
  `params/task.json`). Startup fails with a clear error if the run has no copy.

So for any run from 2026-11-10 16:22 on, set both to null and only change
`policy.checkpoint`. The original policy's run predates the copies and needs both set, which
is how the file is checked in. The real side reads only a few values from the task (rates,
action scales, lift height), so a saved task with an option that has since been removed, such
as `joint_goal_time_s`, still deploys; it only fails if you try to rebuild that sim
environment from it.

Edits that reproduce the past configurations:

| Configuration | Changes from `grasp_wxai.json` |
|---|---|
| Original policy | none; scene `tabletop_jenga_wxai.json` |
| Latency (v2) | `w_table_contact` -1.0; `object_pose_delay_s` 0.01; `action_hold_probability` 0.1 |
| Interpolation | latency + `ee_goal_interpolation` true |
| Gripper (v3) | interpolation + `gripper_goal_time_s` 0.5; `w_gripper_switch` -0.2 |
| 10 Hz | gripper + `control_hz` 10.0; `episode_steps` 150 |
| Joint model, 10 Hz (2026-11-11 18:09 run) | `control_hz` 10.0; `episode_steps` 150; `gripper_goal_time_s` 0.5; `ee_goal_interpolation` true; `command_delay_s` 0.006; `joint_friction.enabled` true; scene `tabletop_jenga_wxai_friction.json` |
| No curriculum, no randomization (used by tests) | `lift_threshold` 0.01; `curriculum.enabled` false; `stiffness_damping_enabled` false; `object_size_variants.enabled` false; `friction.enabled` false; both `object_pose_noise` sources disabled |

The sim tests build their variants the same way, in `sim/mujoco/tests/task_files.py`.

## Open items

- **More full-speed episodes of the original policy** with the 50 ms joint goal time. 4 of 5
  is a small sample.
- **The depth sensor reads short,** which makes the policy grip about 7 mm high and reach
  about 14 mm far. See [block_height_offset.md](block_height_offset.md). Fixing that may
  matter more for grasp success than anything in this note. (Fixed by a tare calibration on
  2026-11-13.)
- **Send-once** would need a guard on the real arm that re-sends the last target when no new
  goal arrives, and a matching sim option. Not built.
- **A larger action scale at 10 Hz** to restore arm speed. Not tried.
- **Velocity feed-forward is real-only.** With a goal time above 0.001 s the driver feeds the
  planned joint velocity into its velocity loop, so the velocity term damps the difference from
  the plan. The MuJoCo position actuator damps the absolute velocity, which makes a joint
  moving at speed v lag by v / 16 rad (128 x lag = 8 x v). With goal time 0 the real arm has no
  feed-forward and behaves like the sim. How much this matters for the policy's moves has not
  been checked.
- **Inertia feed-forward.** The driver's compensation is "inverse dynamics", which at speed
  includes the torque to accelerate the links; the docs do not say whether it uses the planned
  or the measured acceleration. Logging the compensation effort through a fast joint move would
  show it (a rise before the joint starts moving means planned). The sim compensates gravity
  only.
