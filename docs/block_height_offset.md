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
The arm's part (about 5.5 to 7 mm at the block's reach) is unchanged; it is a separate problem,
covered in [arm_joint_offsets.md](arm_joint_offsets.md) since 2026-11-26.

## Short version

The 13 to 14 mm is two unrelated errors that happen to add up.

| Source | Size at the usual block position | Does it hurt the grasp? |
|---|---|---|
| The D405 depth sensor reads short, which lifts everything the camera sees | 6.5 to 8 mm in height (and about 14 mm further from the robot). **Since the 2026-11-13 tare: about 0.4 mm.** | Yes. The gripper ends up that much too high on the block, and reaches too far. |
| The arm reports its fingertip higher than it is, by an amount that grows with reach, so in the arm's own coordinates the table there is at 5.5 mm, not 0. It is a constant shoulder and elbow angle error ([arm_joint_offsets.md](arm_joint_offsets.md)) | 5.5 mm | Mostly no. It applies equally to the block and to the gripper. |

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

### 6. The arm's side

Moved to [arm_joint_offsets.md](arm_joint_offsets.md), section 1. Table touches over 90 deg of
direction read higher with reach, not with direction. The cause turned out to be a constant error
in the shoulder and elbow angles, which the camera calibration partly absorbs: a calibration is one
rigid placement of the camera, so it absorbs the arm's error near the block but not everywhere.

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

Table touches in the same run (two near the base, two beside the block) put the table at about
7.0 mm in the arm's coordinates at the block's reach; they are listed in
[arm_joint_offsets.md](arm_joint_offsets.md), section 1. With that, the top touch at 81.5 mm gives a
74.5 mm block, against the real 75 mm.

- **What remains of the original 13.5 mm** is the arm's frame: the block's centre reads
  44.7 mm against 37.5 mm in sim, because the table at that reach reads about 7 mm. It moves
  the gripper and the block together.

### 9 and 10. The arm under position control, and over direction, height and tool pitch

Moved to [arm_joint_offsets.md](arm_joint_offsets.md), sections 2 and 3.

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

See [arm_joint_offsets.md](arm_joint_offsets.md). It does not shift the grasp, since the camera is
calibrated against the arm, but it sets where the table and block sit in the arm's frame, which the
policy sees.

## Tools

| Tool | Purpose |
|---|---|
| `vision/calibration/check_depth.py` | Depth image against the board's colour-image distance |
| `vision/calibration/tare.py` | Tare calibration of the D405 against the board, with backup and restore |
| `vision/pipeline/view_camera.py` | Plain live camera feed; `--depth` shows the depth at the image centre |
| `./run.sh --track ... --manipulation policy --policy-control shadow --manipulation-logging` | Logs TCP and tracked block together for hand-guided touches |

The arm's measurement tools are listed in [arm_joint_offsets.md](arm_joint_offsets.md).
