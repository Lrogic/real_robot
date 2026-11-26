# Camera calibration: parked and moving follower (November 2026)

Written 2026-11-13; the camera mount and the fitted transforms were added on 2026-11-16. The
D405 sits on the follower (camera) arm. This note covers how its pose in the manipulator base
frame is calibrated, how accurate that is, and what happened when we tried to let the follower
move. Read `docs/block_height_offset.md` first for the depth error and
`docs/arm_joint_offsets.md` for the arm's own height error; this note refers to both.

## Short version

| Setup | Error at objects ~0.4 m from the camera | Status |
|---|---|---|
| Follower parked (`run.sh --camera parked`), October eye-to-hand calibration | about 1 mm (0.7 mm at the block top) | In use |
| Follower moved anywhere, joint-based two-arm calibration (camera mount from the follower model) | about 5.5 mm rms, up to 9 mm, at poses the fit has not seen; 10 mm at the parked pose | `run.sh --camera move`: guided by hand and locked once per run |
| Follower moved anywhere, camera located from a marker in the scene | about 1 mm (estimate) | Not built |

The moving calibration is limited by gear play in the follower's joints: the camera can sit
0.5 to 1 deg away from where the joint readings put it, which is several millimetres at the
objects. No calibration of the joint readings can remove that, because it changes every time
the arm is moved or pushed. We decided to live with about 10 mm when the follower moves, and to
keep using the parked calibration when it does not.

## Frames and convention

Column vectors, `T_destination_source`, metres (the repository convention).

    T_base_camera(q) = Z @ F(q) @ X

| Symbol | Meaning |
|---|---|
| `F(q) = T_fbase_fee` | follower tool pose from its joints: the driver's Cartesian pose for the configured end effector (`wxai_v0_base`) |
| `X = T_fee_camera` | the camera on the follower's tool frame (the mount) |
| `Z = T_base_fbase` | the follower base in the manipulator base frame |
| `E = T_base_ee`, `Y = T_ee_board` | manipulator tool pose, and the ChArUco board in its gripper (calibration only) |

The parked calibration measures `T_base_camera` at one follower pose directly and needs neither
`X` nor `Z`.

## 1. The parked calibration (in use)

`vision/calibration/run.sh collect` and `calibrate`, session `data/sessions/20261009_combined`.
The follower is parked (`--reset` drives it to the recorded joints, then leaves it in idle), the
manipulator holds the ChArUco board through 70 poses, and OpenCV's hand-eye solvers find
`T_base_camera` (SHAH was chosen).

- **Per-sample scatter is large:** 5.8 px reprojection rms, about 0.6 deg in the implied board
  mount. That is the manipulator's own error (section 4) on each board pose. With the camera fixed,
  70 samples average it out.
- **Accuracy where it is used:** after the depth fix, a fingertip touch on the block top agreed
  with the tracked block to 0.7 mm (`block_height_offset.md`, section 8).
- **Repeatability of the parked pose:** idle, then held in position mode, then idle again, the
  camera returns to within 0.03 mm (`check_follower_hold`, section 3). `--reset` approaches the
  pose the same way every time, so the gear play settles the same way.
- **Lens distortion** is handled: the camera's factory coefficients are passed to OpenCV and checked
  against librealsense's projection at every start (`calibration/camera.py`, within 0.1 px). The
  intrinsics themselves are factory values; refining them in the two-arm fit changed nothing
  (section 4).

## 2. Depth (tare, 2026-11-13)

The D405 depth read about 3.8% short at the block's distance. A RealSense tare calibration
(`python -m calibration.tare`) fixed it to under 1 mm at 0.25 and 0.42 m. Details and numbers are
in `block_height_offset.md`, sections 7 and 8. The hand-eye calibration never used depth and did not
change.

## 3. Idle versus position mode on the follower

`idle` in the Trossen driver is a motor brake, not motors off: a velocity loop with an integral
term and no position target. It holds the arm wherever it is when it engages. `position` mode
drives each joint to a target angle.

`python -m calibration.check_follower_hold` (parked pose, board still on the table):

| Change | Objects at the board move |
|---|---|
| idle to position mode at the same joints | 0.45 mm mean (camera 0.15 deg, readings jumped by up to 0.2 deg) |
| idle, held, idle again | 0.03 mm |

So the October anchor is valid for a follower held in position mode, and the parked pose is very
repeatable. Commanding the recorded parked joints from the resting joints (0.3 deg away) did not
move the arm at all: that is inside the joint friction deadband.

**Idle drops a loaded arm.** Because idle starts with no holding torque and builds it only as the
arm moves, switching an arm from gravity compensation into idle at an extended pose lets it fall
until the brake catches up. Both arms did this. The folded parked pose carries almost no load, which
is why the follower does not drop there.

## 4. The moving-follower (two-arm) calibration

### Method

`run.sh collect-dual` records samples in stations. At each station the follower is locked at a
camera pose and the manipulator holds the board through several poses. Each sample stores both
arms' joints before and after the image, their Cartesian poses, the station, the capture order,
the image, intrinsics and detection, so a session could be replayed later. `run.sh calibrate-dual`
solves `X`, `Z` and `Y` together:

1. Start point: Shah's linear `AX = ZB` solve per station (camera pose per station and board mount),
   then across stations (`X`, `Z`). Plain numpy; OpenCV 5 lacks the hand-eye functions and the
   `sam3` env lacks scipy.
2. Refinement: all three transforms by least squares on the corner reprojection error of every
   sample (scipy, soft-L1 loss). This is more accurate than a closed-form solver and is what sets the
   result.

On synthetic data it recovers the true transforms to 0.02 deg and 0.2 mm
(`vision/tests/test_dual_solver.py` and an end-to-end synthetic session).

Metrics it prints:

- **reprojection (px):** predicted against detected board corners.
- **corner disagreement (mm):** each board corner placed through the manipulator (`E Y`) and through
  the camera (`Z F X C`, with `C` from that image alone). This is the closest number to a tracked
  object's error. A single image's board pose adds up to about 1 mm of its own.
- **held out:** every 5th sample left out of the fit.
- **leave one station out:** refit without each follower pose and test on it. This is the error at
  a follower pose the calibration has not seen, the number that matters for a moving follower.

### Result (session `20261113_130351_dual`, both arms in position mode)

83 samples in 16 stations, follower reach 0.25 to 0.60 m.

| | Reprojection | Corner disagreement |
|---|---|---|
| Fit (67 samples) | 8.7 px rms | 4.6 mm rms |
| Held out (16) | 9.3 px | 4.6 mm |
| Leave one station out | 12.4 px | 6.4 mm rms, worst station 11.9 mm |
| October parked pose, predicted vs the October calibration | | objects shift 10.1 mm mean (max 13.7); 1.5 deg, 3.3 mm |

A good fit would be well under 1 px.

### The camera mount: from the model, not solved (2026-11-16)

`calibrate-dual --mount model` takes `X` from the follower's MuJoCo model
(`wxai_follower.xml`, the `wrist_cam` camera in the `ee_gripper_link` site, which matches the
driver's tool frame to 0.000 deg and 0.00 mm) and solves only `Z` and `Y`. Both fits on all 83
samples (`--holdout-every 0`):

| | Mount solved | Mount from the model |
|---|---|---|
| Fit: reprojection, corners | 8.8 px, 4.6 mm rms | 9.1 px, 4.5 mm rms |
| Leave one station out | 6.0 mm rms | **5.5 mm rms** (worst station 9.2 mm) |
| October parked pose vs the October calibration | 1.4 deg; objects 9.1 mm mean | 1.7 deg; objects 10.2 mm mean |

- **The solved mount differs from the model by 0.9 deg and 5 mm** (4.6 mm of it along the tool z
  axis). `Z` and `Y` move with it (0.26 deg / 3.2 mm and 0.9 deg / 3.1 mm), so part of that
  difference is the fit trading parameters against the play in the data, not a real mount error.
- **The final camera pose barely changes.** Over the 16 stations `T_base_camera` differs by 0.85
  deg and 2.6 mm at the camera; objects 0.3 to 0.5 m in front move 2.4 mm on average (4.8 mm at
  most), much of the rotation being about the optical axis.
- **The model mount predicts unseen poses slightly better** with six fewer parameters. The
  difference is well inside the 6 to 10 mm set by the play.

`pipeline/camera_arm.yaml` uses the model mount on all samples:
`data/sessions/20261113_130351_dual_model_all/T_follower_camera.json` (session.json copied, `samples`
a link to the original session's).

### What the solved transforms are, and why the follower base is tilted

The values in use (model mount, all samples). Angles are roll, pitch, yaw (`R = Rz Ry Rx`):

| Transform | Translation (mm) | Rotation (deg) |
|---|---|---|
| `Z = T_base_fbase`, follower base in the manipulator base | (982.1, 23.9, 37.0) | (0.65, **4.76**, -178.8) |
| `Y = T_ee_board`, board in the manipulator's gripper | (22.4, 86.3, -69.2) | (91.1, -1.6, -92.0) |
| `X = T_fee_camera`, camera on the follower's tool (model) | (-97.4, 9.0, 54.3) | (-110.0, 0.0, -90.0) |

The follower base is 0.98 m away, facing the manipulator to within 1.2 deg. The camera looks along
the fingers, pitched 20 deg toward the tool's -z (down at the parked pose), from about 10 cm behind
and 5 cm above the tool point. The 4.8 deg pitch of `Z` is mostly the two arms' reach errors:

- **Each arm reads as if its base were tilted.** The manipulator reads a level table as rising
  about 32 mm per metre of reach (`arm_joint_offsets.md`, sections 1 and 2), the follower about
  31 mm/m (section 6 below). Those readings are what an arm without that error would give if its
  base were tilted forward and down by about 1.8 deg. The bases are not tilted; the error (a
  constant shoulder and elbow angle error on both arms, `arm_joint_offsets.md`) produces the same
  readings. Re-solved with the offsets on both arms, `Z` comes out 1.1 deg from level, 970 mm away
  and 6 mm higher, which matches the bases' measured layout (`arm_joint_offsets.md`, section 5).
- **`Z` relates those two "as-if" bases, not the real ones.** The calibration only sees each
  arm's readings, so it finds the transform that makes them agree.
- **Facing each other, the two tilts add.** A table point moved toward the follower gets further
  from the manipulator (its reading rises) and closer to the follower (its reading falls), so the
  difference between the readings changes by 32 + 31 = 63 mm/m: a 3.6 deg tilt, with the follower's
  forward axis pointing down in the manipulator's frame. The fit found 4.8 deg that way; the rest is
  a real tilt or a reach error that is not exactly linear. Arms facing the same way would cancel.
- **The height is mostly the same effect.** At the follower's base, 0.98 m away, the manipulator's
  frame reads about 31 mm high, so `Z`'s 37 mm is not the mounting height difference.
- **Why it still works.** `Z` contains "undo the follower's error, apply the manipulator's", so
  `Z @ F(q) @ X` puts the camera in the manipulator's reported frame at any follower pose, as the
  parked calibration does. That holds only as far as each arm's error is one fixed tilt. It grows
  with reach in every direction around the base (a cone) and follows the gravity load, so a single
  `Z` matches it best along the line between the bases. The leftover is probably a few tenths of a
  degree, under the play.

The parked calibration absorbs the manipulator's error the same way: `T_base_camera` is tilted by
about 1.8 deg against the camera's true pose, so the camera reports positions as the arm does. That
is why a fingertip touch and the tracked block agree to 0.7 mm although the table is not at z = 0 in
the arm's coordinates.

### What was ruled out

| Suspect | Test | Result |
|---|---|---|
| Solver or model | Synthetic data | Exact |
| Follower or manipulator joint zero offsets | Offsets on joints 1 to 4 of either or both arms, MuJoCo FK (identical to the driver's) | Fit 8.8 to 6.1 px; unseen follower poses unchanged (about 5 mm) |
| Gravity compliance of the follower | Deflection proportional to MuJoCo gravity torque | No gain |
| Camera intrinsics | Refit fx, fy, cx, cy, k1, k2 | 8.8 to 8.6 px |
| Board sagging in the gripper | Board-mount error against gravity's direction in the board frame | Explains under 10% |

Fitting a free camera pose per station (only the board mount shared) still left 3.4 px, so the
manipulator side alone carries about 1 deg and 2.5 mm of scatter per sample, with no pattern in
joint angle, gravity or time. The October calibration shows the same (5.8 px per sample). Per
station, the follower's real camera pose differed from the best rigid chain by 0.2 to 2 deg.

### The cause: gear play (`python -m calibration.check_follower_play`)

The encoders read the motor side of each gearbox. The gaps between gear teeth let the output turn a
little with no change in the reading, and the output stays wherever it was last pushed within that
gap. With a board still in view, the follower was held and pushed gently up, down, left and right:

| Mode | Hidden rotation after release (camera rotation the joints did not register) | Object shift after release |
|---|---|---|
| position | 0.59 to 1.14 deg, mean 0.95 | 2.9 to 5.3 mm |
| idle | 0.04 to 0.69 deg, mean 0.40 | 0.8 to 2.8 mm |

- After a gentle push the camera stays 0.5 to 1 deg away from where it was while the readings
  change by 0.1 to 0.25 deg. That matches the calibration scatter.
- In position mode the arm did not return after a push: it stays in the friction deadband.
- Idle hid less (the motor yields and turns with the output, so more of the motion reaches the
  encoder), but one test with four hand pushes per mode, position first, is not a precise comparison.
  Idle was not usable anyway, because it drops a loaded arm (section 3).
- The test was at the parked pose, where gravity does not press the gears to one side. Load-bearing
  joints are more consistent in extended poses; the wrist yaw and roll joints are not.

Approaching a pose from the same direction every time makes the play settle the same way, which is
why the parked pose repeats to 0.03 mm. Hand-placed poses do not get that.

### Why the parked calibration does not suffer the same way

The manipulator has similar play. In the parked calibration it puts random error on each board
pose, but the camera is fixed and 70 samples average it down. With a moving follower the play is in
the camera's own pose, each placement gets one random error that nothing averages, and the camera's
tilt is multiplied by the 0.4 m to the objects.

More stations, or moving the follower after every capture, would give more independent play
errors and so a better estimate of `X` and `Z`. It would not help at run time: each new placement
still lands with its own 0.5 to 1 deg that the joints do not register.

## 5. Options for later

1. **Locate the camera from the scene.** Fix a ChArUco board or ArUco markers on the table, measure
   them once in the manipulator frame (from the parked calibration), and after each follower move
   compute `T_base_camera` from the marker in view. About 0.1 deg and 0.3 mm, independent of the
   follower's joints. A marker only has to be visible when the camera re-localizes after a move.
   The manipulator's own play remains, as it does today.
2. **Fixed approach direction.** Scripted moves that always finish the same way, so the play
   settles the same way, for a few saved follower poses (as `--camera parked` does for the parked
   pose).
   Each saved pose would still need its own check.
3. **Use the joint-based result as is** (about 6 to 10 mm) where that is good enough. This is
   `run.sh --camera move`.

Not tried: idle for both arms (drops). A joint-offset model was tried on 2026-11-25
(`arm_joint_offsets.md`, section 4): the shoulder and elbow offsets on both arms improve the
leave-one-station-out error only from 5.09 to 4.82 mm, since the play dominates, but they give
the follower base its real layout.

## 6. The follower's reach error

With the follower in gravity compensation and its carriage touching the top of the upright block at
three reaches (`sim2real/experiments/follower_reach/capture_touches.py`, result
`results/touches_follower_20261113_113913.json`), the reading rose about 31 mm per metre of reach.
That is close to the manipulator's 32 mm/m (`arm_joint_offsets.md`, section 1), so both arms show
the same pattern. Three touches only; the third, at another direction, was 2 mm off the line.
`results/workspace_follower.json` holds 18 hand-placed workspace poses with images: reach 0.25 to
0.67 m, base angle -29 to +50 deg, camera height 0.10 to 0.38 m. The manipulator's error stays the
same under position control and in both directions of travel (`arm_joint_offsets.md`, section 2);
the follower was not measured that way.

## Tools

All in `vision/calibration/` unless noted. Hardware tools need the tracking pipeline stopped.

| Tool | Env | Purpose |
|---|---|---|
| `run.sh collect` / `calibrate` | fp_robot / sam3 | Parked eye-to-hand calibration |
| `run.sh collect-dual` | fp_robot | Two-arm collection in stations (`--follower-hold`, `--manipulator-hold`, both position by default) |
| `run.sh calibrate-dual` | fp_robot | Solves `X`, `Z`, `Y` (or `Z`, `Y` with `--mount model`); writes `<session>/T_follower_camera.json` with the per-station and leave-one-station-out errors |
| `dual_solver.py` | | The solve (start point and refinement); tests in `vision/tests/test_dual_solver.py` |
| `reset_follower.py` | vision | Drive the follower to the session's parked joints, then idle |
| `check_follower_hold.py` | vision | Camera pose idle against held, from a still board |
| `check_follower_play.py` | vision | Push test: camera motion the joint readings miss |
| `tare.py`, `check_depth.py` | vision | Depth calibration and check |
| `sim2real/experiments/follower_reach/capture_touches.py` | fp_robot | Hand-guided touches or poses with live joints, optional camera window (`--camera`) |

Data: `data/sessions/*_dual/`, `data/follower_hold/`, `data/follower_play/`, `data/tare/` (all
git-ignored).
