# Why the tracked block reads 13 mm too high (November 2026)

Written 2026-11-12. On the real arm the tracked jenga block's centre reads about 51 mm above
the table. In simulation a standing block's centre is at 37.5 mm (the block is 75 mm tall and
the table is at height 0). This note explains where the 13 to 14 mm difference comes from,
what it does to the policy, and what can be done about it.

## Short version

The 13 to 14 mm is two unrelated errors that happen to add up.

| Source | Size at the usual block position | Does it hurt the grasp? |
|---|---|---|
| The D405 depth sensor reads short, which lifts everything the camera sees | 6.5 to 8 mm in height (and about 14 mm further from the robot) | Yes. The gripper ends up that much too high on the block, and reaches too far. |
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
  the plate has not been measured.
- **If the table really is flat and 9 to 13 mm below the base frame,** the arm's reported
  fingertip height is about 15 to 18 mm too high at the block position. The 5.5 mm reading is
  where the plate (negative) and the reach effect (positive) net out.

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

## What to do about it

### The depth sensor (this is the part that hurts the grasp)

1. **RealSense tare calibration.** This is the sensor's own routine for absolute distance
   error: give it the true distance to a flat target and it corrects the stereo calibration
   in the camera. Note that the similarly named on-chip calibration targets depth noise, not
   absolute accuracy.
   - Do not move the camera or the follower arm; the hand-eye calibration is only valid at
     the parked pose.
   - Prop the ChArUco board face-on and centred, 0.30 to 0.40 m away. Run
     `python -m calibration.check_depth` and take the "board plane on the optical axis" value
     as the ground truth in millimetres.
   - Run Tare Calibration in the RealSense Viewer (not installed on the lab machine;
     `librealsense2-utils`) or through `pyrealsense2`'s `run_tare_calibration`, and write the
     result to the camera. The same menu can restore the factory calibration.
   - Rerun `check_depth` at 0.25, 0.42 and 0.52 m. Within 1 to 2 mm at all three means it
     worked.
   - Not yet tried on this camera. It assumes the D405's colour and depth images share an
     origin; if the result is off by a few millimetres at every distance, suspect that.
2. **Or correct the depth in software.** Apply `depth / (1 - 0.076 * depth)` in
   `vision/pipeline/camera.py` before the depth reaches FoundationPose. At the block's
   distance this leaves about 2 mm along the view, about 1 mm in height. Not implemented.
3. **Check afterwards.** With the block at its usual position the tracked centre should drop
   from about 51 mm to about 43 to 44 mm, and it should sit about 14 mm closer to the robot.
4. **The hand-eye calibration does not need redoing.** It never used depth.

### The arm

This part does not shift the grasp, but it matters for anything tied to absolute height.

- **Workspace floor.** `workspace_min` z in `manipulation/policy/jenga/config.yaml` is 5 mm.
  In the arm's coordinates the real table is at about 5.5 mm at the block position, so that
  floor is at the table surface there and above it closer in.
- **Reading logs.** A logged TCP height is not a clearance. Subtract the table reading at
  that reach: roughly 0.04 x reach(m) x 1000 - 11 mm.
- **The policy sees absolute heights.** Its observations include the gripper and block
  heights in the base frame. In sim the table is always at 0 and a standing block's centre at
  37.5 mm; on the real arm the same block is at 43 mm even with perfect depth. Whether the
  policy relies on absolute height near the table has not been tested.

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
| `vision/pipeline/view_camera.py` | Plain live camera feed; `--depth` shows the depth at the image centre |
| `./run.sh --track ... --manipulation policy --policy-control shadow --manipulation-logging` | Logs TCP and tracked block together for hand-guided touches |
