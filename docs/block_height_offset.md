# Why the tracked block reads 13 mm too high (November 2026)

Written 2026-11-12. On the real arm the tracked jenga block's centre reads about 51 mm above
the table. In simulation a standing block's centre is at 37.5 mm (the block is 75 mm tall and
the table is at height 0). This note explains where the 13 to 14 mm difference comes from,
what it does to the policy, and what can be done about it.

**Update 2026-11-13: the depth part is fixed.** A RealSense tare calibration, written to the
camera's flash, cut the depth error at the block's distance from -15.8 mm to -0.9 mm. See
[7. The tare calibration](#7-the-tare-calibration-2026-11-13). Runs tracked before that date
place the block about 6.5 mm higher and 14 mm further from the robot than runs after it. On a
tracked block the camera now agrees with a fingertip touch on the block top to 0.7 mm, and the
standing block's centre reads 44.7 mm ([8. After the tare](#8-the-tracked-block-after-the-tare-foundationpose_runstare_check)).
The arm's part (about 5.5 to 7 mm at the block's reach) is unchanged.

## Short version

The 13 to 14 mm is two unrelated errors that happen to add up.

| Source | Size at the usual block position | Does it hurt the grasp? |
|---|---|---|
| The D405 depth sensor reads short, which lifts everything the camera sees | 6.5 to 8 mm in height (and about 14 mm further from the robot). **Since the 2026-11-13 tare: about 0.4 mm.** | Yes. The gripper ends up that much too high on the block, and reaches too far. |
| The arm reports its fingertip higher than it is, by an amount that grows with reach, so in the arm's own coordinates the table there is at 5.5 mm, not 0 | 5.5 mm | Mostly no. It applies equally to the block and to the gripper. |

FoundationPose, lens distortion and the hand-eye calibration were each checked and are not
the cause.

## The key idea: everything is in the arm's coordinates

The policy sees the block's position in the robot base frame and commands the gripper in the
same frame. But nothing in the system measures that frame directly:

- **The arm's position** is computed from its joint angles through a rigid model of the arm.
  Anything the joint sensors do not see (links bending, play in the joints, a joint whose zero
  is slightly off) is missing from it.
- **The camera's position** comes from the hand-eye calibration. That calibration was made by
  mounting a ChArUco board on the gripper, moving the arm through 70 poses, and solving for
  the camera placement that makes "where the camera sees the board" agree with "where the arm
  says the board is". So the camera reports positions as the arm would report them, including
  the arm's own errors.

So "height 0" in this system means "where the arm reads 0", and that is not the table.

## A worked example

Take a block standing at the usual position, about 0.43 m in front of the robot.

1. **Physically,** the block's centre is 37.5 mm above the table.
2. **The arm reads the table there at 5.5 mm.** Touching the table with the fingertip at that
   reach gives a TCP height of about 5.5 mm, not 0.
3. **So in the arm's coordinates the block centre is at 5.5 + 37.5 = 43 mm,** and its top is
   at about 80.5 mm. A direct fingertip touch on the top read 82.2 mm.
4. **A perfect camera, calibrated against this arm, would report the centre at about 43 mm.**
   That is correct in the only frame the system has.
5. **The depth sensor reads about 15 mm short at this distance.** The camera looks down at
   the block from in front, so pulling a point 15 mm toward the camera raises it by about
   6.5 mm and moves it about 14 mm further from the robot.
6. **The tracker therefore reports the centre at about 51 mm.**

What the policy then does: it positions the gripper relative to the block it sees, at 51 mm.
The real block is at 43 mm in those coordinates. So the gripper is about 7 to 8 mm higher on
the block than the policy intends. The 5.5 mm from step 2 does not add to that, because it is
in both the block's true position and the gripper's.

A common confusion: the two effects add when asking "why does the reading say 51 and not
37.5?" (13.5 mm = 5.5 + 8). They do not add when asking "how far off is the grasp?" (only the
camera's 7 to 8 mm).

## The measurements

### 1. The offset is in every run

The tracked block centre at the start of each of 47 episodes in nine logged runs
(`vision/foundationpose_runs/*/policy_log.jsonl`, all with calibration `20261009_combined`):

| Block state | Episodes | Expected centre | Tracked |
|---|---|---|---|
| Standing on end | 38 | 37.5 mm | 46 to 60 mm, mean 51.3 |
| Lying on its wide face | 8 | 7.5 mm | 15 to 21 mm |

Because the lying block also reads high, the error is not along the block's long axis and is
not a problem with the block model.

### 2. Fingertip touches (`foundationpose_runs/offset_check`)

Done in shadow mode: the arm is in gravity compensation, hand-guided, and the log records the
TCP position from the joints alongside the tracked block.

| Touch | Arm TCP height | Camera |
|---|---|---|
| Table at (0.297, -0.211) | 3.9 mm | |
| Table at (0.286, -0.298) | 6.3 mm | |
| Table at (0.373, -0.089) | 5.4 mm | |
| Table at (0.464, -0.193) | 10.5 mm | |
| Top of the standing block at (0.422, 0.000) | 82.2 mm | 88.7 mm (tracked centre 51.2 + 37.5) |

So the camera places the block top 6.5 mm above where the arm touched it, and the arm reads
the table at 4 to 10 mm.

### 3. The raw depth image shows the same error

Projecting the saved depth image into the base frame with the calibration, with no pose
estimation involved, puts the table at 12 to 12.6 mm near the block where the arm found 5.5 mm.
FoundationPose's block top (88.7 mm) lies inside the range of the raw block pixels (86 to
91 mm). So the tracker faithfully reports what the depth image says.

### 4. Lens distortion is not the cause (`foundationpose_runs/offset_check_undistort`)

Tracking treats the image as an ideal pinhole image by default. Rerunning with `--undistort`
left the tracked centre at 51.0 mm (51.2 before) and the depth-derived table at 11 to 13 mm.

### 5. Depth against the calibration board (`vision/calibration/check_depth.py`)

The script compares the depth image with the board's distance computed from the colour image
alone, which is the measurement the hand-eye calibration is built from.

| Board distance from camera | Depth minus colour | As % of depth | Error / distance squared |
|---|---|---|---|
| 0.253 m | -4.6 mm | -1.9% | 72 mm/m^2 |
| 0.314 m | -7.5 mm | -2.4% | 76 mm/m^2 |
| 0.420 m | -15.6 mm | -3.7% | 88 mm/m^2 |
| 0.522 m | -19.9 mm | -3.8% | 73 mm/m^2 |

- The camera is about 0.42 m from the block (tape measure: about 44 cm).
- At 0.42 m the error moves points by +13.9 mm in x, +1.9 mm in y and +6.4 mm in z in the
  base frame. That z shift matches the 6.5 mm from the block-top touch.
- The error grows roughly with distance squared. That is what a stereo camera gives when its
  internal calibration has drifted by a constant pixel shift. A single scale factor would not
  correct it.
- One formula fits all four points to within 2.2 mm:
  `corrected = depth / (1 - 0.076 * depth)`, depth in metres. A straight line fits about as
  well over this range, so four points do not prove the square law.
- With the depth error removed, camera and arm agree at the block top: 88.7 - 6.4 = 82.3 mm
  against the touched 82.2 mm. So the hand-eye calibration is not the problem.

### 6. The arm's side (`foundationpose_runs/offset_check_undistort_test3`)

More table touches, spread over 80 degrees of direction around the robot:

| Direction from straight ahead | Reach | TCP height at contact |
|---|---|---|
| -17 deg | 0.306 m | 2.1 mm |
| -27 deg | 0.471 m | 8.7 mm |
| -77 deg | 0.495 m | 9.3 mm |
| -95 deg | 0.468 m | 7.7 mm |
| -106 deg | 0.500 m | 9.1 mm |
| -108 deg | 0.482 m | 7.6 mm |

- **The reading depends on reach, not direction.** A table tilted relative to the base would
  read higher on one side and lower on the other; a tilted-plane fit to the first session
  predicted -3 to -7 mm for the sideways touches, and they read +8 to +9 mm.
- **Both sessions fit a straight line in reach:** height = 36 to 46 mm per metre of reach,
  minus 9 to 12.6 mm. That is about 4 mm of extra height for every 10 cm of reach.
- **Extrapolated to the base, the table is 9 to 13 mm below the base frame.** That is the
  sign expected from the mounting plate the base sits on. Nobody touched the table there, and
  the plate has not been measured. **Section 8 has touches at 13 to 14 cm from the base,
  which read only -2 mm, so this extrapolation overstates the error near the base.**
- **If the table really is flat and 9 to 13 mm below the base frame,** the arm's reported
  fingertip height is about 15 to 18 mm too high at the block position. The 5.5 mm reading is
  where the plate (negative) and the reach effect (positive) net out. With the section 8
  touches the same reasoning gives about 6 mm for the plate and about 13 mm too high at the
  block, still assuming a straight line in reach.

The cause on the arm side is not known. Candidates:

- **Droop:** links bending and play in the joints under the arm's weight, which the joint
  sensors do not see. The sign matches (the real tip is lower than computed).
- **A joint zero error:** a shoulder zero off by about 2.3 degrees would give this pattern.
  That is large, and the shoulder and elbow read within 0.005 rad of zero when the arm is at
  rest, so this is less likely if that rest pose is the mechanical zero. The driver has a
  per-joint `position_offset` for homing error, and the Trossen docs describe a homing
  procedure with jigs.

These touches were hand-guided in gravity compensation. Under position control the motors
hold the arm more stiffly, so droop could be smaller during a policy run. That has not been
measured.

One limit on "the camera inherits the arm's error": a calibration is one rigid placement of
the camera, so it can only absorb the arm's error as a flat tilt plus a shift. The arm's error
grows with reach in every direction, which is a cone, not a tilt. The two agree near the block
(the evidence above) and may drift apart elsewhere on the table.

### 7. The tare calibration (2026-11-13)

Run with `python -m calibration.tare` (from `vision/`, env `vision`). The board was at 0.417 m,
tilted 9 deg, centred. Ground truth along the depth axis was 417.3 mm. The colour and depth
origins agree to 0.2 mm on the D405, so the colour image's board pose gives the depth camera's
distance directly. The firmware-mode tare ran without host assistance. Its own health check
went from -3.58% before to -0.04% after. The new table differs from the old one in one float of
the left camera's rectification rotation (plus the checksum). It was written to flash and the
camera was rebooted. Then `check_depth` gave:

| Board distance | Before tare | After tare | Shift in base z after |
|---|---|---|---|
| 0.25 m | -4.6 mm | +0.4 mm | -0.3 mm |
| 0.42 m (block distance, tare pose) | -15.8 mm | -0.9 mm | +0.4 mm |
| 0.52 m | -19.9 mm | -3.8 mm | +1.3 mm |

- A little of the distance-dependent error remains beyond the tare distance (-3.8 mm along the
  view at 0.52 m, 1.3 mm of height). A tare at about 0.5 m would move the zero there. It is not
  needed while objects stay near 0.42 m from the camera.
- **The D405 (firmware 5.17) ignores a calibration table set in RAM.** After
  `set_calibration_table`, depth was unchanged to 0.1 mm whether the table was set before a
  stream started or while one ran. It only took effect after `write_calibration` and a reboot
  (`hardware_reset` or replugging). So a tare result cannot be checked before it is written.
  `calibration.tare` writes, reboots, checks, and then offers to write the saved table back.
- The old table is in `vision/calibration/data/tare/20261113_105800/calibration_before.bin`
  (git-ignored). Restore it with
  `python -m calibration.tare --restore calibration/data/tare/20261113_105800/calibration_before.bin`.
- The hand-eye calibration is unaffected. The board's colour-image pose and reprojection error
  were the same before and after.

### 8. The tracked block after the tare (`foundationpose_runs/tare_check`)

Run on 2026-11-13 after the tare, with the camera arm parked by `--reset`. The policy ran in
shadow mode with logging, the arm was in gravity compensation, and the touches were
hand-guided. The block stood upright (long axis within 1 deg of vertical) at about
(0.420, 0.007) m.

| | Before tare (section 2) | After tare |
|---|---|---|
| Tracked centre z, block still | 51.2 mm | 44.7 mm (std 0.6 mm over 177 ticks; 44.9 after the touches) |
| Tracked top (centre + 37.5) | 88.7 mm | 82.2 mm |
| Fingertip touch on the top | 82.2 mm | 81.5 mm (at x 0.420, y 0.005) |
| Camera minus arm at the top | +6.5 mm | +0.7 mm |

So the camera and the arm now agree at the block, and the grasp is no longer shifted by the
depth sensor. The block's x cannot be compared with older runs (it was placed by hand), so the
expected 14 mm move toward the robot was not checked.

Table touches in the same run:

| Where | Reach | Direction from straight ahead | TCP height |
|---|---|---|---|
| In front of the base | 0.138 m | -8 deg | -1.7 mm |
| 90 deg clockwise | 0.130 m | -76 deg | -2.6 mm |
| Beside the block (1) | 0.460 m | -20 deg | +8.2 mm |
| Beside the block (2) | 0.485 m | -21 deg | +9.1 mm |

- **Reach, not direction, again.** The two touches near the base are 68 deg apart and differ
  by 0.9 mm.
- **A straight line through these four touches:** height = 32 mm per metre of reach, minus
  6.4 mm. It is shallower than the section 6 fits (36 to 46 mm/m, minus 9 to 12.6 mm), which
  had no touches closer than 0.3 m. Two clusters of reach cannot show whether the true curve
  is a line.
- **The arm checks itself.** The line puts the table at 7.0 mm at the block's reach (0.420 m).
  The top touch at 81.5 mm then gives a block height of 74.5 mm, against the real 75 mm.
- **What remains of the original 13.5 mm** is the arm's frame: the block's centre reads
  44.7 mm against 37.5 mm in sim, because the table at that reach reads about 7 mm. It moves
  the gripper and the block together.

## What to do about it

### The depth sensor (this is the part that hurts the grasp)

Done on 2026-11-13 with option 1 (section 7) and checked on a tracked block (step 3,
section 8). The steps are kept for a camera swap or recalibration.

1. **RealSense tare calibration.** This is the sensor's own routine for absolute distance
   error: give it the true distance to a flat target and it corrects the stereo calibration
   in the camera. Note that the similarly named on-chip calibration targets depth noise, not
   absolute accuracy.
   - Do not move the camera or the follower arm; the hand-eye calibration is only valid at
     the parked pose.
   - Prop the ChArUco board centred and roughly face-on (a tilt under about 20 deg is fine) at
     the objects' distance from the camera, about 0.42 m. One pose is enough.
   - Run `python -m calibration.tare` from `vision/` (env `vision`). It saves the current
     table, measures the ground truth with the `check_depth` code, runs the tare, writes the
     result and reboots the camera, checks the depth again wherever the board is put, and
     writes the old table back unless told to keep the new one.
   - Within 1 to 2 mm at 0.25, 0.42 and 0.52 m means it worked. If it is off by a similar few
     millimetres at every distance, suspect the colour/depth origin assumption (0.2 mm on this
     camera).
2. **Or correct the depth in software.** Apply `depth / (1 - 0.076 * depth)` in
   `vision/pipeline/camera.py` before the depth reaches FoundationPose. At the block's
   distance this leaves about 2 mm along the view, about 1 mm in height. Not implemented, and
   not needed after the tare. Do not apply it on top of the tare.
3. **Check afterwards.** With the block at its usual position the tracked centre should drop
   from about 51 mm to about 43 to 44 mm, and it should sit about 14 mm closer to the robot.
4. **The hand-eye calibration does not need redoing.** It never used depth.

### The arm

This part does not shift the grasp, but it matters for anything tied to absolute height.

- **Workspace floor.** `workspace_min` z in `manipulation/policy/jenga/config.yaml` is 5 mm.
  In the arm's coordinates the real table is at about 5.5 to 7 mm at the block position, so
  that floor is about at the table surface there. Near the base the table reads about -2 mm,
  so the floor is about 7 mm above the table there.
- **Reading logs.** A logged TCP height is not a clearance. Subtract the table reading at
  that reach: roughly 0.032 x reach(m) x 1000 - 6.4 mm (section 8; the section 6 sessions gave
  0.036 to 0.046 and -9 to -12.6, with no touches inside 0.3 m).
- **The policy sees absolute heights.** Its observations include the gripper and block
  heights in the base frame. In sim the table is always at 0 and a standing block's centre at
  37.5 mm; on the real arm, after the tare, the same block reads 44.7 mm (section 8). Whether
  the policy relies on absolute height near the table has not been tested.

Options, none done:

- **Measure it properly.** Measure the mounting plate. Touch the table under position control
  (command the tip down slowly until contact) to see how much of the effect remains when the
  motors are holding. Touch at more reaches, including close to the base.
- **Check the joint zeros** with the homing jigs if a zero error is suspected.
- **Model it.** Either correct the real side (subtract the reach-dependent height from the
  TCP and block heights before the policy sees them, so the table is at 0 as in training), or
  make training tolerate it (randomize the table height or an offset on the height
  observations by 10 mm or so).
- **Leave it.** It mostly cancels for the grasp. Fix the depth first and see what remains.

## Tools

| Tool | Purpose |
|---|---|
| `vision/calibration/check_depth.py` | Depth image against the board's colour-image distance |
| `vision/calibration/tare.py` | Tare calibration of the D405 against the board, with backup and restore |
| `vision/pipeline/view_camera.py` | Plain live camera feed; `--depth` shows the depth at the image centre |
| `./run.sh --track ... --manipulation policy --policy-control shadow --manipulation-logging` | Logs TCP and tracked block together for hand-guided touches |
