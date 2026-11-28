# Release training

What was tried to train the release skill: placing a grasped jenga block at a goal defined
relative to other blocks, opening the gripper and backing away. Covers the bridge goal's reachability
problem, the tower goal used to test the pipeline, the flat-block grasp that makes the bridge
possible, the real-arm test of that grasp, how the perception noise in training compares with
FoundationPose, the stillness-scaled release reward, and where the wide bridge's anchor was placed
and why the start-state transport loses about half its attempts. Results are from 2026-11-26 and
2026-11-27.

Code: `sim/mujoco/src/tabletop_sim/tasks/release/`, task schema in `scene/release_config.py`,
start states from `scripts/generate_release_starts.py`. Configs: `configs/release/{scenes,tasks}`.

## How the release task works

- **Relative goals.** The goal is given in an *anchor frame* together with the *relative objects*
  (e.g. two standing support blocks). The anchor frame has its origin on the table, z up, and x/y
  turned by the anchor yaw. Every start state samples the anchor in a disk (`anchor_center`,
  `anchor_radius`) with a yaw range (`anchor_yaw_deg` ± `anchor_yaw_range_deg`), and the supports
  and goal follow it: `T_world_X = T_world_anchor @ X_in_anchor`. With `goal_symmetry: box`, the
  four orientations that differ by 180 deg about the block's own axes count as the same goal.
- **Start states are a saved dataset.** The release policy starts with the block already held
  near the goal. `generate_release_starts.py` produces those states by running the trained grasp
  policy, then a scripted IK transport:
  1. lift straight up to a safe height, turning the block toward the goal orientation (the
     height is the support tops + `transport_clearance` + how far the fingers and block reach
     below the TCP *at the goal orientation*; the supports are still parked during the lift);
  2. move the supports from their parking spot to the sampled anchor;
  3. cross to above the target;
  4. descend to the goal plus a random offset in a ball above it;
  5. turn the TCP by a small random roll / pitch / yaw;
  6. settle, check, and save.

  A state is kept only if the block is still held and inside the ball, nothing touched the table
  or a support, and the supports did not move. Reset restores a random kept state (arm, gripper,
  block, supports, goal). The env refuses a dataset whose `.meta.json` was made with different
  placement or start settings, scene or size scales. Datasets live in `sim/mujoco/data/release_starts/`
  (git-ignored).
- **Rewards.** `place` (closeness to the goal), `released` (within tolerance, no finger contact,
  gripper open, block still), `retreat` (`released` times how far the TCP has risen), and
  penalties for robot contact with the table or supports, disturbing the supports, ending the
  episode (block dropped or support toppled), arm speed and joint limits, and `release_moving`.
  With `release_scale_by_stillness`, `released` and `retreat` are also multiplied by the block's
  stillness when the gripper was last commanded open (see "Stable release").
- **Logging.** `Episode_Termination/<term>` is the number of envs that term ended per control
  step, averaged over the iteration's steps, so one NaN reset in a 32-step iteration shows as
  0.0312. `Curriculum/termination_counts/<term>` (grasp and release) is the running total since the
  start of training. `Episode_Metrics/released_frac` counts releases without the stillness factor
  and drives the curriculum. `Episode_Metrics/release_stillness` (only with
  `release_scale_by_stillness`) is the stillness recorded at the episode's last opening, and 1 for
  an episode that never opened, so it reads near 1 until `released_frac` is clearly above 0.
- **Generator counts.** The generator's `attempts` counts every lift started, including the
  attempts still in progress when it stops (about one per env), and leaves out grasp timeouts. The
  kept share of *finished* attempts is `kept / (kept + rejections other than grasp_timeout)`. The
  rates in this document use that.

## The bridge goal could not be reached with the upright grasp

The first goal was a bridge: two standing jenga blocks with the held block laid across them.

The grasp policy at the time (`joint_friction-no_slip_interpolation`) picks the block up
standing upright. It grips the middle of the block with the long axis along the tool's
approach direction: the TCP is within 3 mm of the block centre, the long axis is about along the
approach, and the tool is about 30 deg from vertical. To lay that block flat, the tool has to point
horizontally about 8 cm above the table.

A CPU IK study (damped least squares, many random restarts, checked by solving the grasp policy's
own hold pose exactly) gave these residual errors:

| Tool orientation | z = 0.08 m | z = 0.12 m | z = 0.16 m |
|---|---|---|---|
| Horizontal, soft limits (0.85 of range) | 86–167 mm at every reach 0.25–0.45 m | 24–190 mm | 0–93 mm |
| Horizontal, full joint range | 21–166 mm | 0 at reach 0.40–0.45 m | 0 at reach ≥ 0.35 m |
| Straight down, soft limits | 0 at reach 0.25–0.40 m | 5–62 mm | 50–105 mm |
| Straight down, full range | 0 everywhere tested | 0 up to 0.40 m | 0 up to 0.35 m |
| 45 deg down, soft limits | 0 everywhere tested | 0 | 0 |

A horizontal tool at bridge height is out of reach even with the full joint range. The real arm
has no soft limits: one configured range per joint, matching the model's except `joint_0` (±180 deg
on the arm, ±175 deg in the model). Commands are clipped to it, and the controller faults past it
plus a tolerance. The 0.85 soft limit is the sim's (mjlab `soft_joint_pos_limit_factor`, used by
the IK and the `joint_pos_limits` penalty).

The scripted transport showed the same thing in sim. With the bridge first laid out sideways
(tool horizontal along ±y), the IK settled on the soft limits 6–29 cm short. Turning the bridge
radially, so the gripper would reach forward, and preferring the symmetric goal whose approach
points away from the base, did not fix it.

## The tower goal, to test the pipeline

To test the whole pipeline with the existing grasp, the goal was changed to a **tower**: the held
upright block stood on the centre of a block lying flat (`release_tower_wxai.json`). That keeps
the grasp orientation, so the tool stays about 30 deg from vertical.

**Run 1 (`release_tower`, MuJoCo's default soft contact).** This plateaued at about 32% of steps
released, with about 45% of episodes ending in a drop. The video showed the base block sinking
into the table and springing back. Measured base-block sink below its resting height:

| Case | Default contact, median / worst | `solref` [0.004, 1], median / worst |
|---|---|---|
| Base block alone | 0.11 mm, steady | 0.08 mm |
| Block dropped from the open gripper | 5.8 / 16.1 mm, bouncing for > 1 s | 0.17 / 0.39 mm |
| Trained policy placing | 3.6 / 10.5 mm | 0.18 / 1.4 mm |

MuJoCo's default `solref` (0.02 s, 1) lets an object sink by roughly impact speed × 20 ms, about
1 cm at 0.5 m/s. Scene objects now take optional `solref` / `solimp`, and **every scene, grasp and
release, uses `solref: [0.004, 1.0]`**, the stiffest MuJoCo recommends at a 2 ms step. The
fingers have contact priority 1 with their own tuned settings, so the grip is unchanged. Checked
on the grasp policy, one episode in each of 2048 envs:

| Difficulty | Contact | Hold score | Holding at end | Lift |
|---|---|---|---|---|
| 1.0 | default | 0.724 | 94.0% | 30.8 mm |
| 1.0 | stiff | 0.724 | 94.3% | 30.8 mm |
| 0.0 | default | 0.798 | 100% | 33.2 mm |
| 0.0 | stiff | 0.798 | 100% | 33.3 mm |

**Run 2 (`release_tower_stiff`, stiff contact).**
- **Curriculum:** reached full difficulty by iteration 300.
- **End of training (with exploration noise):** 62% of steps released, 9.1% of episodes dropped,
  final error 1.2 cm and 10 deg, 11 NaN resets in about 1.5 M episodes.
- **Deterministic check on 2048 start states it never trained on:** 81% released and standing
  at the end, 4.7% dropped. It plateaued from about iteration 500 (`model_500`: 84.6% / 4.6%).
- **The remaining drops** all happen as the gripper opens, at about step 12–35 of 150, with the
  block still moving at about 0.12 m/s and spinning at about 1.5 rad/s. Falls and successes look
  the same at that moment.

**Dataset size.** On 2000 start states it never saw, `model_500` scored the same as on its
training rows (84.6% vs 84.0% released, 4.6% vs 5.1% dropped). Distance from an unseen start to its
nearest training start, in a 6-D space of anchor xy, anchor yaw (5 cm per rad) and arrival offset:

| Rows | Median | p90 | Max |
|---|---|---|---|
| 1,000 | 14 mm | 18 mm | 25 mm |
| 5,000 | 10 mm | 13 mm | 20 mm |
| 50,000 | 7 mm | 9 mm | 13 mm |

So 5,000–20,000 rows are enough. The bridge datasets use 20,000.

## A flat-block grasp, to make the bridge reachable

Picked from above with the tool pointing down, a block lying flat stays flat in the hand. Tool-down
poses at bridge height are reachable (table above).

- **Grasp scene:** `tabletop_jenga_flat_wxai_friction_grip_base8mm.json`. This is the `base8mm`
  scene with the block lying on its broad face (75 × 25 mm down, 15 mm tall) and the long axis along
  world y.
- **Grasp task:** `grasp_wxai_grip_interp_table_contact.json`. This is `grasp_wxai_grip_interp.json`
  with `w_table_contact: -1`. The TCP is at the fingertips and the pads cover their last 14.6 mm,
  so gripping a 15 mm-tall block puts the fingertips at or just above the table.
- **Run:** `grasp_jenga_flat/2026-11-27_09-58-52`. In sim it holds the block at the end of 94% of
  episodes at full difficulty and 100% at difficulty 0. Its commanded goal goes about 5 mm *into*
  the table (median low point base z = −12.9 mm, table at −8.1 mm) and the TCP stops at about
  −3.3 mm. The real `safety.workspace_min` z was therefore lowered from −0.003 to −0.020.

**Bridge on the flat grasp.** The bridge scene was rebuilt on the flat-block scene.
- **First generation:** only 8% of attempts were kept. Lifts timed out because, with the tool
  pointing down, `joint_3`'s soft limit stops a lift to about 11–12 cm at 0.35 m out.
- **Fix:** `transport_clearance` 3 → 1 cm and `anchor_center` (−0.30, 0.15) → (−0.35, 0.10). That
  kept 20%, then 28% on the full set.
- **No placement bias:** kept anchors are uniform in yaw and over the disk (radius quartiles 25.6 /
  35.9 / 43.1 mm against 25.0 / 35.4 / 43.3 for a uniform disk). The rejection depends on how the
  block sits in the hand.

**First bridge run (`release_bridge_flat`, stopped at iteration 1352).** Diagnosis of `model_1000`,
one deterministic episode in each of 2048 envs: 12.1% dropped, 1.6% toppled a support. None of the
drops slipped out of the closed gripper. Block state at the last opening:

| | Kept (1770) | Dropped (248) |
|---|---|---|
| Offset across the bridge, median | 2.1 mm | 5.9 mm |
| More than 5 mm across | 11% | **67%** |
| Offset along the bridge, median | 4.1 mm | 3.8 mm |
| Height above goal, median | 5.9 mm | 7.0 mm |
| Speed / spin, median | 0.094 m/s / 1.37 rad/s | 0.104 m/s / 1.38 rad/s |
| Step of opening, median | 7 | 7 |

- **Sideways misplacement is the main cause.** In that layout the supports were 15 mm thick across
  the bridge, so a bridge centre more than about 5–7 mm off tips sideways.
- **The policy lets go almost at once,** about 0.35 s in, while the block is still moving.
- **The training pose noise is 10 mm per step,** larger than the margin. See the noise section.

This led to **`release_bridge_wide_wxai.json`**:
- **Supports:** at anchor (0, ∓22.5 mm), each standing with its **25 mm side across the bridge**.
- **Bridge:** long axis along anchor y, covering the full 25 × 15 mm top of each support with
  7.5 mm overhanging past each outer edge. There is a 30 mm gap for the fingers.
- **Anchor:** centre base (0.25, −0.13) m, radius 50 mm, yaw 0 ± 30 deg. See "Anchor placement
  for the wide bridge".
- **Status:** start states generated (20,000 rows, below). Training started 2026-11-27 as
  `release_bridge_wide/2026-11-27_16-52-50` (1500 iterations).

## Flat block on the real arm

Runs `vision/foundationpose_runs/flat_jenga_real_test_logging` and `..._3`, both with
`--camera move`. Run 1 had 19 attempts. The other 19 "episodes" were returns to the ready pose
without pressing go.

| | Real (run 1) | Sim, difficulty 1 |
|---|---|---|
| Block lifted more than 2 cm | 63% | 94% |
| Block centre at start (base z) | −2.1 mm | −0.6 mm |
| Lowest actual TCP | −5.6 mm | −3.3 mm |
| Lowest commanded goal | −14.9 mm | −13.0 mm |
| Raw actions beyond ±1 | 31% | 20% |

There were no workspace or tracking holds.

**Placement.** The block sat on average at base (357, −31) mm, about 31 mm to the robot's right
(−y) of the training centre (365, 0). Five attempts started outside the 5 cm training disk.

**Failures follow the block's yaw** (training: long axis at 90 ± 20 deg):

| Long-axis yaw at start | Run 1 | Run 3 (block near the centre) |
|---|---|---|
| 100–117 deg | 4 never lifted, 3 partial | all 6 failed (lift 7–9 mm, block flipped) |
| 60–97 deg | all 12 successes | 3 of 4 lifted 25–27 mm |

Run 3's failing yaws are inside the training range.

**What goes wrong: the real arm descends faster than the sim arm.** Free air, TCP 40–90 mm above
the table, per control step:

| | Real | Sim, difficulty 0 | Sim, difficulty 1 |
|---|---|---|---|
| Commanded wrist yaw / descent | 2.86 deg / 10 mm (both at the limit) | same | same |
| Achieved yaw | 1.54 deg | 1.31 deg | 1.36 deg |
| Achieved descent | 3.45 mm | 2.57 mm | 2.20 mm |
| Wrist turn per mm of descent | **0.45 deg/mm** | 0.51 deg/mm | 0.62 deg/mm |

The policy turns and descends at full speed at once. On the real arm the fingers reach the block
with less of the turn done, so the far finger lands on the block's edge and flips it. The larger
the turn, the worse it gets.

**Replay.** The real commands were replayed through the training controller with randomization off:

| Free-air step, median | Descent | Yaw | Shoulder `joint_1` | Wrist `joint_5` |
|---|---|---|---|---|
| Real arm | 3.45 mm | 1.53 deg | 1.07 deg | 1.62 deg |
| Sim, same actions, 20 Hz | 2.58 mm | 1.70 deg | 0.79 deg | 1.75 deg |
| Sim, same actions, real tick times | 2.90 mm | 1.85 deg | 0.90 deg | 1.88 deg |

- **Loop timing is not the cause.** The real loop ran at a mean of 17.5 Hz (ticks 31–91 ms p10–p90,
  inner IK about 445 Hz). Longer ticks move the arm further per step but slightly *less* per second.
  The turn per mm barely changes (0.66 → 0.64 deg/mm).
- **The arm's own response is the cause.** Under the same actions and timing, the real shoulder
  moves about 19% more and the wrist turns about 14% less.
- **Up and down differ.** Real vs sim, full-speed vertical steps 10–90 mm above the table: down
  1.65× (438 steps), up 0.84× (19 steps, during the lift). MuJoCo's dry friction is symmetric, so
  this points to something gravity- or direction-dependent, although the driver's gravity
  compensation matched the model at standstill (`sim2real_control.md`). Not resolved. The
  `hardware_control` `relative_tra_z±` / `relative_rot_z±` reruns on the current arm would isolate it.
- **The gripper finger speeds** (`qvel[6]`, `qvel[7]`) were the most out-of-range observations
  (6–8 std at p95).

**Why domain randomization did not cover it.**
- **Randomization did produce arms like this.** Stiffness and damping ×0.8–1.2 and joint friction
  ×0.7–1.3 per joint. 13% of sim episodes at full difficulty turn as little per mm as the real arm.
- **A misaligned approach rarely flips the block in sim.** The block tilted past 45 deg before the
  first stable hold in only 2% of episodes, and the policy still held it at the end in 56–65% of
  those. It reopened and retried in about 75% of episodes and held at the end in 96%.
- **So care isn't rewarded.** A clumsy approach usually just nudges the block and costs a few steps
  of hold reward, which full-speed descent more than makes up. On the real arm the same mistake
  flips the block and ends the attempt.

Two reasons a misaligned approach is cheap in sim: the sim arm arrives better aligned (0.6 against
0.45 deg/mm), and when it does hit an edge the sim block flips less often than the real one
appears to.

**Not implemented:**
- a penalty (and termination) for tilting or moving the block before it is grasped, so retries
  are no longer free;
- reset yaw ±45 deg instead of ±20;
- wider arm randomization toward a faster shoulder and elbow and a slower wrist, plus
  `action_hold_probability` for uneven step timing;
- a residual encoder bias (mjlab `dr.encoder_bias`). As built it would not reach our observations
  or the IK, so it would need the actor's `qpos` / `ee_pose` computed at the biased angles.

## Per-step and persistent pose noise

The actor sees the object pose with two errors (`domain_randomization.object_pose_noise`), both
scaled by the curriculum difficulty:
- `persistent_bias`: drawn once per episode;
- `per_step`: drawn every control step.

The release goal gets one draw per episode (`goal_pose_noise`). The grasp tasks and the first
release tasks use per-step 10 mm and 5 deg std (clipped at 2 std) and a persistent bias of 0.5 mm
and 0.1 deg. That is **far stricter than FoundationPose is now**, and the split is the wrong way
round:

| | Measured | Training (old) |
|---|---|---|
| Per-step position jitter, flat block at rest | std x 0.8, y 0.2, z 0.6 mm (p95 of 3-D deviation 2.0 mm) | 10 mm |
| Per-step rotation jitter, flat block at rest | median 0.6 deg, p95 1.6 deg, no symmetry flips | 5 deg |
| Jitter in the hand, lifted | 0.75–1.0 mm per axis (p95 2.4 mm) | same model |
| Persistent, height | −1.7 mm (std 0.6 mm across attempts) | 0.5 mm |
| Persistent, tilt | 1.8 deg median (max 2.9) | 0.1 deg |
| Persistent, sideways along the squeeze axis | +3.3 mm (std 1.1 mm across 21 lifts) | 0.5 mm |
| Persistent, other horizontal | not measurable; calibration about 5.5 mm rms (up to 9–10) in `move` mode, about 1 mm `parked` (`camera_calibration.md`) | 0.5 mm |

How these were measured:
- **Jitter at rest:** policy running, TCP more than 50 mm above the table, gripper not yet closed,
  so the block is not moving.
- **Persistent height and tilt:** against a block lying flat on the table (centre 7.5 mm up,
  15 mm axis vertical).
- **Sideways bias:** a squeezed block is centred between the fingers in sim to within 0.09 mm,
  but the tracked block sits about 3 mm off-centre on the real arm. That mixes tracking,
  calibration and arm kinematics.

The upright block tracks worse on the same setup:

| Upright-block runs (after the 2026-11-13 depth tare) | Position std x / y / z | Rotation median / p95 |
|---|---|---|
| `GOOD_RUN!` (parked) | 2.06 / 0.48 / 1.01 mm | 2.4 / 5.7 deg |
| `GOOD_RUN_move` | 0.53 / 0.25 / 0.35 mm | 1.2 / 3.3 deg |
| `GOOD_RUN_test2` (move) | 0.29 / 0.16 / 0.23 mm | 0.8 / 1.9 deg |
| `GOOD_RUN_test3` (move) | 2.54 / 1.15 / 5.60 mm | 1.3 / 4.4 deg |
| Three flat-block runs (move) | 0.59–0.91 / 0.15–0.26 / 0.48–0.65 mm | 0.65–0.75 / 1.5–1.6 deg |

- `GOOD_RUN_test3`'s 5.6 mm height spread is mostly one attempt with a 17 mm tracking jump (29
  re-registrations and 80 SAM mismatches in that run).
- Older upright runs (Oct–Nov, before the tare and recalibration): median 1.95 / 0.44 / 0.92 mm and
  3.2 deg (p95 about 8 deg), with a few runs at 5–12 mm.

So 10 mm and 5 deg was a cautious setting for the old upright block. For the flat block it is
about 10× the real jitter, while the persistent error is underestimated. The first bridge run had
to hold ±5 mm sideways through 10 mm of per-step noise.

`release_bridge_wide_wxai.json` uses:
- **per step:** 1.5 mm and 1.5 deg;
- **persistent bias:** 3 mm in x and y, 2 mm in z, 2 deg;
- **goal:** 3 mm / 3 mm / 2 mm and 2 deg.

The `--log-variation` run that set the original values was not found on disk, including in the
trash. Measuring the flat block at rest with `./run.sh --setup-track ... --num-seconds 60
--log-variation` (once `parked`, once `move`) would pin the jitter down.

## Anchor placement for the wide bridge

The anchor disk was first centred at base (0.309, −0.127) m, 2 cm right of where a real support
stood in the shadow run `vision/foundationpose_runs/anchor_check` (support at base (309, −107) mm,
target block at (356, +18) mm). It moved closer to the base for two reasons:
- **Reach.** With the tool pointing down, `joint_3`'s soft limit (0.85 of its range) limits the
  lift and the descent from about 0.33 m out. The first disk reached 0.384 m.
- **Real-arm clearance.** On the real arm the supports already stand while the arm grasps (in sim
  they are parked until the lift ends). The measured arm joints of the 43 real flat-block grasps
  (`flat_jenga_real_test_logging*`, every second tick, turned about the base so the block is at base
  y = +12 mm) were replayed in the bridge scene with supports at the disk centre and edges, and the
  ticks where a robot geom overlaps a support were counted.

Each candidate got a 300-row trial generation. Rates are kept / finished attempts; the
overlap columns are at anchor yaw −30 / +30 deg.

| Centre (base, m) | Radius | Distance | Kept, max hang | Kept, goal hang | Lift timeouts (goal hang) | Overlap at centre | Overlap at edge toward the grasp |
|---|---|---|---|---|---|---|---|
| (0.309, −0.127) | 50 mm | 0.334 m | 43% | 46% | 248 | 1 / 0 | 65 / 51 |
| (0.265, −0.15) | 50 mm | 0.305 m | 45% | 49% | 221 | 0 / 0 | 5 / 6 |
| (0.23, −0.15) | 50 mm | 0.275 m | 39% | 41% | 270 | 0 / 0 | 5 / 1 |
| (0.23, −0.15) | 30 mm | 0.275 m | 41% | 45% | 241 | – | – |
| (0.23, −0.18) | 50 mm | 0.292 m | 37% | – | – | 0 / 0 | 0 / 0 |
| (0.20, −0.18) | 50 mm | 0.269 m | 30% | – | – | 0 / 0 | 0 / 0 |
| (0.25, −0.10) | 50 mm | 0.269 m | – | 55% | 183 | 6 / 6 | 164 / 87 (3 cm in: 86 / 25) |
| **(0.25, −0.13)** | 50 mm | 0.282 m | – | **53%** | 190 | 0 / 0 | 18 / 10 (3 cm in: 6 / 6) |
| (0.25, −0.15) | 50 mm | 0.292 m | – | 48% | 229 | 0 / 0 | 6 / 7 |

"Max hang" and "goal hang" are the generator before and after the transport-height change below.

- **(0.25, −0.13) was chosen.** It keeps nearly the most attempts, its far edge (0.332 m) stays
  about where `joint_3` starts to limit, and the real grasps touch a support only at the disk edge
  nearest the grasp.
- **(0.25, −0.10) keeps the most in sim but is unusable on the real arm.** It lies on the line
  from the base to the block, so the arm passes over it on the way to the grasp.
- **Anchors nearer the base's x axis had fewer lift timeouts,** although the lift goes straight up
  above the grasp before the supports appear. Probably the anchor's direction from the base
  changes which symmetric copy of the goal is chosen, and with it the wrist turn the lift asks for.
  Not checked; 300-row counts vary by about ±20.
- **On the real arm** the support must stand about 6 cm closer to the base and 2 cm further right
  than in `anchor_check`.

**The full dataset** (`bridge_wide_wxai.pt`, 20,001 rows at 2048 envs, 504 s) kept 53% of finished
attempts. Rejections: `timeout_lift` 11,587, `timeout_across` 3,658, `timeout_noise` 1,237,
`joint_limits` 471, `outside_ball` 467, `timeout_down` 376, `collision_in_transport` 158,
`lost_grasp` 11, and 1,290 grasp timeouts.

### Why the lift times out

The lift is the main loss (31% of finished attempts in the full dataset). An instrumented trial
logged every lift:
- **The turn finishes, the rise does not.** Timed-out lifts end with about 2 deg of rotation error
  but 28 mm of position error (median): about 20 mm short in height and about 20 mm back toward the
  base. `joint_3` sits near its lower soft limit (median 0.05 of its range).
- **The transport height was set by the grasp's tilt.** `hang` was the larger of how far the
  fingers and block reach below the TCP at the grasp orientation and at the goal orientation. The
  grasp ends with the tool tilted about 23 deg, and the 75 mm block's end then hangs 19.5 mm below
  the TCP, against 5.3 mm at the goal orientation. Nothing can be hit at the grasp tilt during the
  lift (the supports are parked) and the turn completes long before the rise, so the generator now
  uses the goal orientation only. That lowered the transport height from about 97 to 90 mm and the
  rise from about 69 to 62 mm. Kept rates rose 3–4 points, and lift timeouts fell about 20%.
- **There is no fixed height limit; the outcome depends on the path.** Before the change, lifts
  that arrived rose 69 mm (median) and timed-out ones 72 mm. After it, 62 and 64 mm: the cut-off
  moved down with the target. A timed-out lift aims at the same kind of pose as one that arrives.
  The turn (rotation action scale 0.05 rad per step, 1 rad/s) completes near the bottom of the lift
  (0.2 m/s), `joint_3` then sits near its limit, and the per-substep IK stalls. Turn size, tool
  tilt and lean, reach, block size and how the block sits in the hand do not predict it on their
  own (AUC 0.4–0.6; the rise and `joint_3`'s starting position reach 0.7).
- **Next fix (not done):** change the lift's order: rise first and turn at the top, turn more
  slowly, or lift diagonally toward the target.

`collision_in_transport` means a moving robot link or the block touched the table or a support
during across, down, noise or settle (the supports are placed when the lift ends). It is about 0.4%
of finished attempts; which phase and body was not checked.

## Stable release

`release_moving` (`w_release_moving`, −5 in the release tasks) is a penalty on the step the gripper
command switches to open, times (1 − stillness) of the block. Stillness is
`(1 − tanh(5·speed))·(1 − tanh(2·spin))`; `released` uses the same measure. It does not change
behaviour: per episode about −0.9, against +13.5 for `released`, +8.9 for `place` and +6.2 for
`retreat`. `moving_release_frac` stayed at about 1.0 throughout the first bridge run. Stillness now
enters the reward through `released` (every step, after the release) and `release_moving` (on the
opening step only). So a block that settles at the goal after a moving release still earns almost
the full release reward.

Two options were considered: raise `w_release_moving` to about −20 to −30, or gate `released`
and `retreat` on how the release happened. The second is implemented.

**Stillness-scaled release (`release_scale_by_stillness`).**
- `HeldGripperAction` (`tasks/release/mdp/actions.py`) records the block's stillness on each step
  whose gripper command switches to open: `(1 − tanh(s_v·speed))·(1 − tanh(s_w·spin))`, measured
  when the command is processed. It is 1 at reset, and every later opening overwrites it, so a
  regrasp and a calmer second release replace a hurried first one.
- `released`, and `retreat` through it, are multiplied by the recorded value. `released_frac`
  (the curriculum metric) stays unscaled. The `release_stillness` metric logs the recorded value.
- Options: `release_stillness_speed_scale` (s_v, default 11 per m/s) and
  `release_stillness_spin_scale` (s_w, default 0.5 per rad/s). On only in
  `release_bridge_wide_wxai.json`, which also sets `w_release_moving` to 0.
- Tests: `test_release_env.py` (still and moving releases, the overwrite, scaling off) and
  `test_release_config.py`.

**How the scales were set,** from two sim measurements on the wide bridge:
- **Floor:** during a deliberate hold at the goal the block moves under 1 mm/s, which scores about
  0.99.
- **Ceiling:** a bridge put down still stays on the supports out to about 11 mm of sideways offset.
  Releasing at 0.05 m/s toward the edge costs about 2 mm of that, 0.1 m/s about 5 mm. Spin hardly
  matters.

So s_v = 11 puts the score's midpoint at 0.05 m/s (0.1 m/s scores about 0.2), and s_w = 0.5 is
lenient (midpoint about 1.1 rad/s). The first bridge run's typical release (0.1 m/s, 1.4 rad/s)
would keep about 0.08 of the release reward.

## Open items

- Evaluate `release_bridge_wide/2026-11-27_16-52-50`.
- Start-state lift: change the lift's order so `joint_3` is not pinned (see "Why the lift times
  out"). Make the generator report kept / finished attempts.
- Simplify the gripper "opened" check to `target == open_pos` for binary grippers, in
  `HeldGripperAction` and in `gripper_opened` (`rewards.py`).
- Real side: in `anchor_check` the support was masked first, so `tracked_name: obj1` in
  `manipulation/policy/jenga/config.yaml` tracked the support. Set it to `obj2`, or mask the block
  first, for real runs.
- Measure FoundationPose jitter with `--log-variation`, `parked` and `move`.
- Rerun `hardware_control` vertical and wrist-rotation moves on the current arm, to explain the
  faster real descent and the up/down asymmetry.
- Grasp robustness: the block-disturbance penalty, wider yaw and wider arm randomization.
- The real-robot release skill (`manipulation/policy`), including tracking the supports, is not
  started.
