# The arm's height error and the shoulder and elbow offsets (November 2026)

Started 2026-11-12 as part of [block_height_offset.md](block_height_offset.md) and moved here on
2026-11-26. The manipulator reports its fingertip higher than it is, by an amount that grows with
reach and height: about 26 mm per metre of reach with the tool straight down. This note collects
every measurement of that error, what it turned out to be, and how to correct it.

## Bottom line

- **It is a constant error in two joint angles, not gravity bending.** The shoulder's true angle is
  its reading + 1.71 deg and the elbow's is its reading - 0.55 deg. With those two corrections the
  arm's height error is a constant at every reach, height, tool pitch and direction measured, to
  the ruler's resolution (about 1 mm). "Sag" was the working name; the data rules out sag in the
  sense of bending under load (section 3).
- **Four independent checks agree.** The ruler map (section 3); a refit of the October parked camera
  calibration, 70 board poses in 3D (section 4); a refit of the November follower calibration
  (section 4); and the two arms' measured placement on the table (section 5).
- **The follower arm has the same error.** Its reach error matches the manipulator's
  ([camera_calibration.md](camera_calibration.md), section 6), and the follower calibration fits best
  and gives a physically level, correctly spaced follower base only with the offsets on both arms.
- **After correction the table is flat at about -8.6 mm** in the arm's frame: the base frame's
  z = 0 is about 8.6 mm above the table (the mounting plate and the base's own geometry).
- **Status (2026-11-26): applied and checked.** Both arms' `position_offset` is set, the camera
  calibrations are re-solved to match, and the table now reads -8.1 mm at every reach, with the
  camera agreeing with the fingertip on a block's top to 0.1 mm parked and within 1.7 mm in move
  mode (section 9). The scene `tabletop_jenga_wxai_friction_grip_base8mm.json` puts the sim's
  table at the same -8.1 mm (section 7).

## Terms

- **Reported height:** the tool point's z from the driver's FK of the measured joints (equal to the
  MuJoCo model's FK to 0.01 mm).
- **Measured height:** the tool point's height above the table, read with a ruler.
- **err = reported - measured.** Positive means the arm thinks the tip is higher than it is.
- **Offset o:** true joint angle = reading + o. The driver's `position_offset` is the negative of
  this (its convention: motor position = reported position + `position_offset`).
- **Radial:** horizontal, along the line from the base axis out to the tip. Straight ahead it is +x;
  at yaw -90 deg it is -y.

## 1. Hand-guided table touches (2026-11-12 and 13)

Shadow-mode runs with the arm in gravity compensation, the fingertip guided onto the table by hand.

`foundationpose_runs/offset_check` (2026-11-12):

| Where | TCP height at contact |
|---|---|
| (0.297, -0.211) | 3.9 mm |
| (0.286, -0.298) | 6.3 mm |
| (0.373, -0.089) | 5.4 mm |
| (0.464, -0.193) | 10.5 mm |

`foundationpose_runs/offset_check_undistort_test3`, spread over 90 deg of direction:

| Direction from straight ahead | Reach | TCP height at contact |
|---|---|---|
| -17 deg | 0.306 m | 2.1 mm |
| -27 deg | 0.471 m | 8.7 mm |
| -77 deg | 0.495 m | 9.3 mm |
| -95 deg | 0.468 m | 7.7 mm |
| -106 deg | 0.500 m | 9.1 mm |
| -108 deg | 0.482 m | 7.6 mm |

`foundationpose_runs/tare_check` (2026-11-13):

| Where | Reach | Direction | TCP height |
|---|---|---|---|
| In front of the base | 0.138 m | -8 deg | -1.7 mm |
| 90 deg clockwise | 0.130 m | -76 deg | -2.6 mm |
| Beside the block (1) | 0.460 m | -20 deg | +8.2 mm |
| Beside the block (2) | 0.485 m | -21 deg | +9.1 mm |

- **Reach, not direction.** A tilted table would read high on one side and low on the other; touches
  90 deg apart at the same reach agree to about 1 mm.
- **Straight lines in reach:** 36 to 46 mm/m (first two sessions, no touches inside 0.3 m) and
  32 mm/m minus 6.4 mm (the third, with touches near the base). The line puts the table at 7.0 mm at
  the block's reach (0.42 m); the block-top touch at 81.5 mm then gives a 74.5 mm block, against the
  real 75 mm.

## 2. Under position control (`sim2real/sag_test/hold_z_sweep.py`, 2026-11-16)

The arm held the tool straight down at a commanded z of 100 mm and stepped along +x from 0.25 to
0.43 m and back, with joint targets from IK on the MuJoCo model. At each stop the tool point's
height was read with a ruler between the fingertips (gripper open 10 mm).

Run 2 (`sim2real/sag_test/results/hold_z_20261116_134852.json`), err in mm:

| Reach | Going out | Coming back |
|---|---|---|
| 0.250 m | 2.8 | 2.6 |
| 0.295 m | 3.7 | 3.8 |
| 0.340 m | 4.6 | 5.1 |
| 0.385 m | 5.6 | 6.7 |
| 0.430 m | 7.3 | |

A line through all nine: **+25.7 mm per metre of reach, residual 0.3 mm rms.** Run 1
(`results/hold_z_20261116_133146.json`, the driver's Cartesian IK, outward only) gave +29 mm/m.

- **The error is there with the motors holding**, about as large as in gravity compensation.
- **It does not depend on the direction of travel.** The two legs agree to about 1 mm, although the
  joints stopped on opposite sides of their targets (the friction deadband,
  [sim2real_control.md](sim2real_control.md#holding-a-position-the-friction-deadband)). The joint
  readings see that shortfall, so it moves the reading and the tip together and cancels from err.
- **Limits:** one direction, one height, one tool orientation.

## 3. The map over direction, reach, height and tool pitch (`sim2real/sag_test/sag_map.py`, 2026-11-25)

To tell the candidate causes apart, `sag_map.py` held the arm in position mode at 61 stops: base
directions 0, -30, -60 and -90 deg (the clear side of the table) and, in each direction, the tool
straight down at z 0.10 m for reaches 0.15 to 0.40 m, pitched 45 deg at z 0.05 m for 0.30 to
0.50 m, and level (pointing away from the base) at z 0.12 m for 0.45 and 0.50 m and at z 0.30 m for
0.30 to 0.50 m. The level rows load the elbow differently at the same reach (3.6 to 6.6 Nm), which
the dry run's design check showed was needed to separate elbow effects from a line in reach. A
reference pose (yaw -45 deg, straight down, reach 0.30 m, z 0.10 m) was visited five times. The tool
point's height was read with a ruler to the whole millimetre: the bottom of the fingertips straight
down, the centre of the tip's end at 45 deg, the middle of the tip's height when level. Results:
`sim2real/sag_test/results/sag_map_20261125_173202.json` (60 stops measured, one skipped).

`analyze_sag.py` fits these models to err and scores each on held-out stops (one base direction,
one tool pitch, one height or one stop left out):

| Model | What it assumes |
|---|---|
| reach | err = a + b x reach |
| reach+tilt | plus a plane in x and y: a table or base tilted against the arm |
| compliance | each joint bends after its encoder in proportion to its gravity torque |
| zero offsets | each joint's true angle is its reading plus a constant |

The gravity torques come from MuJoCo (`wxai_base_real_robot.xml`, link masses and centres of mass,
at the measured joints). They match the driver's own compensation effort to 0.04 Nm at every stop.
Only the shoulder, elbow and wrist pitch were fitted: the base, wrist yaw and wrist roll carry no
gravity torque in these poses and do not move the tool point's height (0 mm per degree).

Fits, the first reference visit left out (err in mm rms):

| Model | Fit | Held out: direction | pitch | height | stop |
|---|---|---|---|---|---|
| reach | 1.29 | 1.33 | 1.65 | 1.59 | 1.33 |
| reach+tilt | 1.29 | 1.84 | 1.72 | 1.67 | 1.40 |
| compliance, shoulder | 2.77 | 2.81 | 4.51 | 4.35 | 2.87 |
| compliance, shoulder + elbow | 1.65 | 1.70 | 3.09 | 3.00 | 1.73 |
| compliance, + wrist pitch | 1.06 | 1.13 | 4.90 | 2.74 | 1.13 |
| zero offsets, shoulder + elbow | **1.05** | **1.10** | **1.25** | **1.22** | **1.10** |

Without the 45 deg stops (48 stops): reach line 1.15 mm, shoulder + elbow offsets 0.78 mm (0.86 mm
leaving out a direction, against 1.20).

- **The 45 deg stops were the hardest to read.** At 0.30 m they agree with the rest; at 0.40 to
  0.50 m they scatter 2 to 4 mm about every model. `analyze_sag.py --exclude-pitch 45` leaves them
  out.
- **The same pose gives the same error.** The reference pose read 2.0, 2.1, 2.0 and 2.0 mm on visits
  two to five; the first visit read 0.8 mm and is left out.
- **Section 2 is reproduced.** The straight-down stops alone give 25.1 mm/m over all four directions
  (26.4 straight ahead), against 25.7.
- **No tilt.** The fitted plane is 0.1 deg, and adding a plane to any model makes every held-out
  error worse. With the best model, a 1.5 to 2 mm drift from yaw 0 to -90 deg remains in two stop
  groups (straight down at 0.10 m: +0.64 to -0.72 mm; level at 0.30 m: +0.96 to -1.00 mm) but not
  the third (level at 0.12 m: -0.16 to +0.44 mm), so it is not a plane. It is at the ruler's
  resolution and unexplained.
- **Not a function of reach alone.** At a fixed reach the error grows with height, about 2 mm between
  z 0.10 and 0.30 m (level at 0.50 m: 11.3 mm at z 0.30 m, 9.7 mm at z 0.13 m).
- **Not gravity bending.** In those two level poses the shoulder holds less torque at z 0.30 m (6.1
  against 7.1 Nm) and the error is larger. Bending would also grow roughly with reach squared; the
  data is close to a line. Compliance with the wrist fits the measured stops but fails on a left-out
  pitch or height: the wrist load changes only with pitch, so it acts as a per-pitch constant.
- **Constant offsets fit to the ruler's resolution:** shoulder +1.71 deg, elbow -0.55 deg (48 stops;
  +1.74 and -0.52 deg with all). The wrist pitch fits -0.2 deg and adds nothing. The constant term,
  **-8.6 mm**, is where the table sits in the corrected frame (section 7).
- **A shoulder offset and a line in reach look alike in height.** A shoulder offset moves the tip
  vertically by about its horizontal distance from the shoulder axis, which is nearly reach. The
  elbow offset is what carries the height dependence.

What the offsets predict radially (not measurable with a ruler), for example at the block's usual
position (0.42 m, z 0.04 m, straight down):

| | Radial | z |
|---|---|---|
| Shoulder +1.71 deg | -1.9 mm | -11.9 mm |
| Elbow -0.55 deg | -2.7 mm | -2.4 mm |
| Total | **-4.6 mm** (toward the base) | -14.3 mm |

An angle error moves the tip at right angles to the line from that joint's axis to the tip. The
shoulder axis is 103.5 mm above the base frame and 20 mm in front of it, so a tip below it moves
partly back toward the base and a tip above it partly outward (+5.9 mm radially from the shoulder
alone, level at 0.50 m and z 0.30 m).

## 4. The camera calibrations, refitted with joint offsets (2026-11-25)

`sim2real/sag_test/refit_calibration_offsets.py` (results in
`sim2real/sag_test/results/refit_calibration_offsets.json`) refits two existing calibration
sessions with offsets on the manipulator's joints as extra unknowns. The sessions are only read.

- **Parked** (`20261009_combined`, 70 board poses, camera on the parked follower):
  `T_base_camera @ C_i = FK(q_i + o) @ T_ee_board`.
- **Moving** (`20261113_130351_dual`, 83 board poses at 16 follower poses):
  `Z @ F(q_follower + o_f) @ X_model @ C_i = FK(q_i + o) @ T_ee_board`, with the follower model's
  camera mount `X_model`, as the calibration in use.

Each minimises the board corners' reprojection error (robust loss), like the calibration's own
refinement, with the camera (or follower base) and board mount re-solved at the same time. The
offsets start at 0 with no prior, so they move from 0 only if they beat the best rigid camera
placement. Every variant is scored on held-out images: 10-fold for the parked session, one follower
pose left out at a time for the moving one.

**Why this tests more than height.** The camera measures the board's pose in 3D, so an error in x
shows as disagreement in x. But the solve also places the camera freely, and any error that one
rigid camera placement can mimic (a constant shift or rotation of the whole workspace, or much of a
slowly varying one) is absorbed and cannot be seen. Only the part that varies from pose to pose is
tested. The calibration poses vary widely in reach and board angle, so much of the offsets' pattern
is testable.

Offsets fitted as unknowns (true = reading + o, 1 sigma):

| Session | Shoulder | Elbow | Wrist pitch | Wrist yaw (joint 4) |
|---|---|---|---|---|
| Sag map (ruler, z only) | +1.71 | -0.55 | | |
| Parked, shoulder only | +1.71 +- 0.15 | | | |
| Parked, shoulder + elbow | +1.31 +- 0.15 | -0.44 +- 0.04 | | |
| Parked, joints 1 to 5 | +1.27 +- 0.16 | -0.36 +- 0.06 | -0.14 +- 0.07 | +0.76 +- 0.11 |
| Moving, shoulder + elbow | +1.54 +- 0.29 | +0.15 +- 0.08 | | |
| Moving, joints 1 to 5 | +1.47 +- 0.34 | +0.18 +- 0.08 | +0.02 +- 0.18 | +0.45 +- 0.25 |

The last joint (wrist roll) cannot be determined: a constant roll of the gripper is the same as
rotating the board's mount, which is solved freely, so any roll offset fits equally well.

Fit quality, board-corner disagreement (mm rms; the same measure the calibration scripts report,
which is per image and includes each image's own pose noise; the in-use parked SHAH solve scores
2.65 on it):

| Session | Variant | Fit | Held out |
|---|---|---|---|
| Parked | no offsets | 2.47 | 2.54 |
| Parked | sag-map offsets on the manipulator, fixed | 2.16 | **2.23** |
| Parked | shoulder + elbow fitted | 2.19 | 2.26 |
| Moving | no offsets | 4.54 | 5.09 |
| Moving | sag-map offsets on the manipulator, fixed | 4.59 | 5.09 |
| Moving | sag-map offsets on both arms, fixed | 4.26 | **4.82** |

- **The parked session finds the same shoulder and elbow offsets** from different data in 3D, a
  month earlier, and the sag-map values fit it as well as its own best fit.
- **The moving session needs the offsets on both arms** to improve. With them on the manipulator
  only, the follower's own uncorrected error remains.
- **The no-offset refits reproduce the calibrations in use:** the moving one exactly; the parked one
  within 1 mm at the block (2 to 3 mm at 0.30 m to the side), since it refines on pixels where the
  file in use came from OpenCV's linear SHAH solve.
- **A hint of a wrist-yaw offset** (joint 4, +0.76 deg parked, +0.45 deg moving), small and not
  applied.

The re-solved transforms with the sag-map offsets fixed:

| | In use | Offsets on the manipulator | Offsets on both arms |
|---|---|---|---|
| Parked `T_base_camera` translation | (806.1, 10.2, 243.3) mm | (807.8, 7.3, 214.0) mm | (follower offsets do not enter) |
| Moving `T_base_fbase` translation | (982.1, 23.9, 37.0) mm | (976.1, 23.5, 2.5) mm | (970.2, 22.3, 6.1) mm |
| Follower base tilt against the manipulator's | 4.8 deg | 3.0 deg | 1.1 deg |

**What the change does to tracked objects.** The physical camera does not move; its estimated pose
does, because the calibration makes the camera agree with the arm. With the corrected arm, a block
at 0.42 m reads about 5.3 mm closer to the base and 15 mm lower than with the calibration in use (the
x shift matches the -4.6 mm the offsets predict, section 3; this is the first evidence for the
radial prediction). In move mode, the follower's offsets mostly change the estimated follower base
(the follower base is solved, so it takes up whatever the follower's FK change does not vary between
follower poses), and add about 2 mm more at the block.

**Parked and moving calibrations still disagree by 11.6 mm** at the parked pose without offsets and
11.9 to 13.3 mm with them. That gap is set by the follower's gear play
([camera_calibration.md](camera_calibration.md)), which offsets do not fix.

## 5. The two bases' placement (2026-11-26)

The moving calibration places the follower's base relative to the manipulator's. Checked against the
table:

| | In use | Offsets on the manipulator | Offsets on both arms | Measured |
|---|---|---|---|---|
| Distance between base centres | 982.4 mm | 976.4 mm | 970.4 mm | about 970 mm (tape) |
| Tilt between the bases | 4.8 deg | 3.0 deg | 1.1 deg | none visible (spirit level) |
| Height difference | 37.0 mm | 2.5 mm | 6.1 mm | about the same height |

The calibration in use had absorbed both arms' uncorrected errors into a tilted, raised follower
base ([camera_calibration.md](camera_calibration.md), section 3, explains the 4.8 deg as the two
arms' reach errors adding). With the offsets on both arms the solve finds the bases' real layout.

**The 22 mm sideways offset is not explained.** Every variant puts the follower's base 22 to 24 mm to
the manipulator's left (bearing +1.3 deg), facing straight back at the manipulator (within 0.16 deg).
The bases are believed to be aligned. Two arrangements produce exactly this, and the calibration
cannot tell them apart:

- **The manipulator's x axis is turned 1.3 deg to its right** and the follower sits on the true line.
  The x axis is where joint_0 reads 0, so this is either a joint_0 zero error or the manipulator's
  base physically rotated on the table; to the arm both are the same. The follower then appears at
  (969.8, +22.0) mm, facing -178.7 deg, as found.
- **The follower is 22 mm to the left and turned 1.3 deg** to face the manipulator.

Ruled out: the follower 22 mm left but facing parallel (it would point 1.3 deg off the manipulator's
base), and a follower joint_0 zero error alone (it would turn the follower without shifting it).
To decide: stretch a string between the base centres, guide the manipulator along it and read
joint_0. About +1.3 deg is the manipulator's x axis; about 0 is the follower's placement. Then the
base housing against the string separates a joint_0 zero error (housing square) from a rotated base.

It does not matter for the policy: the camera is calibrated against the manipulator, so the block
and the gripper are reported in the same frame, turned or not, and a turn about vertical changes no
height.

## 6. What the offsets are

The data shows a constant angle error under the loads of normal operation. What causes it is not
settled:

- **Not an encoder disagreeing with the folded pose.** Trossen's homing script
  (`trossen_arm/docs/_downloads/trossen_arm_homing.zip`) has jigs only for the base and wrist rotate;
  the shoulder and elbow are zeroed wherever they sit folded. They read 0.03 to 0.21 deg folded, as
  homing defines. Homing would therefore move them by a few tenths of a degree, not 1.7.
- **Most likely: the model's 0 deg is not the real folded pose.** In the model, at shoulder = 0 the
  upper arm lies exactly level (elbow axis 244 mm behind the shoulder axis, both at 103.5 mm). The
  fit says the real folded upper arm is tilted 1.71 deg, elbow end up (about 7 mm over 244 mm), for
  example resting on a bumper. **Check:** an inclinometer on a surface of the folded upper arm that
  runs parallel to the line between the two axes. The elbow's -0.55 deg would show on the forearm as
  2.26 deg nose-down against the model (11.5 deg instead of 13.8 deg between the elbow and wrist
  axes), which needs a forearm surface of known angle.
- **Gear play may contribute.** The encoders read the motor side of each gearbox; the link can turn
  within the teeth's gap without the reading changing. Gravity loads the shoulder (-0.5 to -8.8 Nm)
  and elbow (+3.6 to +6.6 Nm) the same way at every pose measured, so the link always rests at the
  same end of the gap, which acts as a constant offset. The joints are zeroed folded, under a
  different load. The measured play, 0.5 to 1 deg over the whole chain
  ([camera_calibration.md](camera_calibration.md)), is less than the shoulder's 1.7 deg alone, so
  play is at most part of it. To measure it: hold an extended pose in position mode, lift the gripper
  gently and let go; at 0.4 m reach, 1 deg of shoulder play moves the tip about 7 mm while the
  reading stays put.
- **Link lengths** slightly different from the model could add to the same pattern; height alone
  cannot separate them from angle offsets, though the height correction is the same.

Whatever the cause, the correction holds while gravity loads the shoulder and elbow the usual way.
It would not if something pushed the arm up hard enough to move the gears across their gap.

**Homing is not the fix and is not free.** The homing script warns that it factory-resets the
controller (joint characteristics such as `effort_correction` and friction are lost; back them up
with the driver's `save_configs_to_file`), and its "set home" cannot be undone except by setting
`position_offset` to the recorded difference.

## 7. After correction: the table at -8.6 mm

The constant term of the offset fit is -8.6 mm: with the corrected FK, reported minus measured
height is the same at every stop. The ruler measures from the table and the arm from its base
frame's z = 0, so the table is 8.6 mm below z = 0 (the mounting plate, and where the model puts the
base origin). Today, uncorrected, the table reads from about -2 mm near the base to +9 mm at 0.5 m.

Consequences once applied:

- A standing block's centre reads about 30.7 mm (section 9), against 37.5 mm in a sim with the
  table at base z = 0, and 44.7 mm before the correction.
- The policy sees absolute heights, so the sim has to match. The scene
  `sim/mujoco/configs/scenes/tabletop_jenga_wxai_friction_grip_base8mm.json` is
  `tabletop_jenga_wxai_friction_grip.json` with the robot's `base_pose` z at +0.0081: the base 8.1 mm
  above the table, so in the base frame (all the policy's observations) the table is at -8.1 mm and
  a standing block's centre at 29.4 mm, as on the real arm. Resets, the lift goal and rewards are in
  world coordinates and unchanged; `jenga/config.py` reads the base pose from the scene, so a
  checkpoint trained on it deploys with matching reset and lift heights. Checkpoints trained with
  the table at 0 see the real table 8.1 mm lower than in training. The value is the table touches'
  -8.1 mm (section 9) rather than the ruler's -8.6 mm, fixed (not randomized).
- `workspace_min` z in `manipulation/policy/jenga/config.yaml` is now -3 mm (was 5 mm), 5 mm above
  the table as before.

## 8. Applying the correction

1. `python -m calibration.joint_offsets --apply --execute` (from `vision/`, both arms folded at
   rest). It backs up each arm's full configuration (`save_configs_to_file`, to
   `vision/calibration/data/arm_configs/`) and sets `position_offset`: joint 1 (shoulder)
   -0.0298 rad, joint 2 (elbow) +0.0097 rad, others unchanged. It is stored in the arm's EEPROM
   (driver 1.9 and later); `--restore --execute` sets joints 1 and 2 back to 0. Both also move each
   changed joint's position limits with its readings, so every physical limit stays where it was
   (otherwise the elbow could not fold its last 0.55 deg, which clips the parked follower pose and
   costs about 5.6 mm at the block, and the shoulder could be driven 1.71 deg further into its rest). Everything that reads
   the driver then uses the corrected angles: its FK and Cartesian IK, the policy's joint
   observations and the sim IK.
2. `vision/pipeline/camera_arm.yaml` points at calibrations re-solved with the same offsets, made by
   `python -m calibration.resolve_joint_offsets` (no new capture; section 4):
   - `20261113_130351_dual_model_all_joint_offsets_both_arms` (move mode): `calibrate_dual.py` with
     `--manipulator-offsets-deg` and `--follower-offsets-deg`, otherwise as the one in use. Follower
     base (970.2, 22.3, 6.1) mm, 1.12 deg; 4.88 mm at a left-out follower pose (in use 5.45).
   - `20261009_combined_joint_offsets_both_arms` (parked): least squares with the manipulator's
     offsets; the follower's parked joints converted (new = old + offset), reachable because the
     limits move with the offsets.
   Both folders have a README; the originals are untouched, and the old yaml lines are commented
   out beside the new ones.
3. Run a fresh `--setup-track`; do not reuse run folders set up before the change.
4. Check: a fingertip touch on a tracked block's top should agree with the camera to about 1 mm, and
   table touches should read about -8.6 mm at every reach.

Cautions:

- Joint angles stored before the change mean a different physical pose after it (new = old + o):
  the follower's parked joints in the parked session, and `camera_pose.json` in older run folders.
  The parked follower's converted elbow target, -0.41 deg, is below the old 0 deg limit, which is
  why the limits are moved too.
- Commanding the manipulator to all-zero joints would fold the shoulder 1.7 deg further than before,
  into its rest.
- The camera's tracked coordinates move by about 15 mm (section 4); do not mix runs from before and
  after.

## 9. Checks after applying (2026-11-26)

Both arms' offsets set with `joint_offsets.py --apply` (each arm's configuration before and after in
`vision/calibration/data/arm_configs/`; exactly the two offsets and the two joints' limits changed).
Then tracking with the policy in shadow mode and logging, the manipulator in gravity compensation,
touches hand-guided and held still.

**Parked camera** (`foundationpose_runs/joint_offsets_check`). Ten table touches:

| Reach | Direction | Table |
|---|---|---|
| 0.233 m | -103 deg | -8.64 mm |
| 0.262 m | -31 deg | -8.10 / -8.39 mm |
| 0.366 m | -12 deg | -7.73 mm |
| 0.392 m | -75 deg | -7.88 mm |
| 0.404 m | -102 deg | -8.65 mm |
| 0.407 m | -75 deg | -8.43 mm |
| 0.411 m | +2 deg | -7.74 mm |
| 0.420 m | -25 deg | -7.85 mm |
| 0.455 m | -45 deg | -7.85 mm |

- **Mean -8.13 mm, std 0.37 mm**, 2.2 mm per metre of reach (32 to 46 mm/m before, section 1). The
  ruler map's constant was -8.6 mm.
- About 0.9 mm lower toward -100 deg than toward 0 deg: the direction drift of section 3.
- A standing block at (0.482, -0.046) m: camera top 68.22 mm (centre 30.75 mm, 1161 untouched
  ticks), fingertip on the top 68.31 mm: **0.1 mm apart.** The touches imply a 76 mm block; jenga
  blocks vary by about a millimetre.

**Moving camera** (`foundationpose_runs/joint_offsets_check_move`), the follower guided to a new pose
and locked, the block touched on top at three positions:

| Block (x, y) | Camera top | Fingertip | Camera - arm |
|---|---|---|---|
| (486, -49) mm | 67.68 mm | 67.89 mm | -0.2 mm |
| (404, -122) mm | 68.28 mm | 66.64 mm | +1.6 mm |
| (376, -24) mm | 65.80 mm | 66.91 mm | -1.1 mm |

1.2 mm rms in height. The touches landed within 0.7 to 4.2 mm of the tracked centre sideways, as
close as a hand aims at a 15 x 25 mm face. One follower placement only: its gear play settles
differently at each placement (base yaw and the wrist joints carry little gravity load), and play
that turns the camera mostly moves objects sideways, which a top touch cannot see.

## Open questions

- **Radial and sideways errors directly.** A board lying on the table and a marker on the gripper,
  both in the same image, give the gripper's position relative to the table in 3D, independent of
  the camera's calibration and pose (the camera can move between images; the board must not).
- **The 1.5 to 2 mm drift between directions** (section 3).
- **The 22 mm sideways offset between the bases** (section 5).
- **A wrist-yaw offset** of +0.5 to +0.8 deg (section 4).

## Tools

| Tool | Purpose |
|---|---|
| `sim2real/sag_test/hold_z_sweep.py` | Holds a commanded height along +x in position mode, ruler at each stop (section 2) |
| `sim2real/sag_test/sag_map.py` | The grid over direction, reach, height and tool pitch (section 3); its dry run checks that the grid separates the models |
| `sim2real/sag_test/analyze_sag.py` | Fits reach, tilt, compliance and joint-offset models with held-out errors; `--exclude-pitch 45` |
| `sim2real/sag_test/refit_calibration_offsets.py` | Refits the parked and moving calibrations with joint offsets (section 4); read only |
| `vision/calibration/joint_offsets.py` | The offset values; shows, applies (with a configuration backup) or restores them on the arms |
| `vision/calibration/resolve_joint_offsets.py` | Writes the re-solved calibrations into new session folders (section 8) |
| `vision/calibration/calibrate_dual.py --manipulator-offsets-deg/--follower-offsets-deg` | The two-arm solve for arms with corrected joints |
| `sim2real/experiments/follower_reach/capture_touches.py` | Hand-guided touches with live joints (the follower's reach error) |
| `./run.sh --track ... --manipulation policy --policy-control shadow --manipulation-logging` | Logs TCP and tracked block together for hand-guided touches (section 1) |
