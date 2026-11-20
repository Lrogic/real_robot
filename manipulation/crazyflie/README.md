h# Crazyflie external position from FoundationPose

`--manipulation crazyflie` streams the tracked Crazyflie's position to the drone over a Crazyradio, using cflib's `send_extpos`. The position is in the manipulator base frame (z up), from the camera arm setup's `T_base_camera`. The drone's Kalman filter fuses it with the IMU. Attitude is not sent: the Crazyflie is nearly 4-fold symmetric, so FoundationPose can settle on a yaw that is 90 or 180 deg off. The drone's own estimate is logged back and drawn next to the tracked point. No setpoints are sent, so the motors never spin. `session.py` has the details and the keys. Settings are in `config.yaml`.

This runs in the `foundationpose` conda env, which has the `vision` env's packages plus `cflib`. `vision/run.sh` picks that env for this mode; set `VISION_ENV` to override it.

## Radio access (once per machine)

Without a udev rule, cflib gets `Access denied` on the Crazyradio:

```bash
sudo groupadd -f plugdev && sudo usermod -aG plugdev $USER
cat <<'EOF' | sudo tee /etc/udev/rules.d/99-bitcraze.rules > /dev/null
# Crazyradio
SUBSYSTEM=="usb", ATTRS{idVendor}=="1915", ATTRS{idProduct}=="7777", MODE="0664", GROUP="plugdev"
# Crazyradio bootloader
SUBSYSTEM=="usb", ATTRS{idVendor}=="1915", ATTRS{idProduct}=="0101", MODE="0664", GROUP="plugdev"
# Crazyflie over USB
SUBSYSTEM=="usb", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="5740", MODE="0664", GROUP="plugdev"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Log out and back in for the group change, then replug the radio. Find the drone's URI and put it in `config.yaml`:

```bash
cd /home/luka/research/real_robot && /opt/conda-envs/foundationpose/bin/python -m manipulation.crazyflie.scan
```

## Running

```bash
cd vision
# mask the Crazyflie on one frame with the follower guided to a view of it, then track
./run.sh --setup-track --save-path foundationpose_runs/drone1 --camera move --recovery --manipulation crazyflie
# later runs reuse the masks; the follower returns to the setup's joints
./run.sh --track --save-path foundationpose_runs/drone1 --camera move --recovery --manipulation crazyflie
# no radio: draw and log only
./run.sh --track --save-path foundationpose_runs/drone1 --camera move --recovery --manipulation crazyflie --manipulation-dry-run
```

In the setup's mask window, give the masked drone the `crazyflie` mesh (from `assets/meshes/`). With `--recovery`, positions are held back until SAM confirms the track and after every mismatch or re-registration. Without it, every tracked pose is sent, including one from a lost track. Each frame goes to `<run>/crazyflie_log.jsonl`: the sent position, why a position was held back, the trust and the drone's estimate.

To check it, move the drone by hand. The magenta `CF est` marker should follow the green `FP` marker, and the `diff` readout should stay at a few mm. The drone's estimate is close to the sent position by construction, so this checks the link and the estimator setup. It does not check the camera calibration (`--camera move` is good to about 6 to 10 mm, `docs/camera_calibration.md`).

## The drone's front

Only position is sent, so the drone's Kalman filter keeps the heading it was last reset to. A wrong heading turns every position correction the wrong way; in `drone6` the drone faced about 135 deg from the assumed heading and flew off along a curve while tracking was still good. FoundationPose gives the heading only up to a 90 deg turn, because the drone looks nearly the same turned 90 deg. So the front is confirmed by hand. A cyan arrow shows the proposed front: `h` turns it 90 deg until it points at the M1/M4 end (opposite the blue LEDs), and `c` confirms. That sets `kalman.initialYaw` to FoundationPose's heading for that front and resets the filter. Afterwards the window draws FoundationPose's heading (green, the 90 deg copy closest to the drone's) and the drone's own (magenta). The log records both on every frame (`heading_deg`, `drone_rpy_deg`, `heading_error_deg`, plus FoundationPose's raw `fp_yaw_raw_deg` and `q_base_object`). The heading itself is not sent.

## Hover test (`--crazyflie-fly`)

`--crazyflie-fly` (needs `--recovery`) adds a hover: `t` arms and takes off to `flight.hover_height_m` above the takeoff point, `l` lands, Space (or Enter in the terminal, or Ctrl+C) kills the motors. The drone flies on its own Kalman estimate, and `flight.py` only commands takeoff, hover in place and landing. It refuses takeoff unless positions arrive at 15 Hz or more, the estimate agrees with the tracked position to 4 cm with under 2 cm std, the battery reads at least `flight.min_battery_v`, the drone's supervisor does not report it locked, crashed or tumbled, and the front is confirmed with FoundationPose's heading within 10 deg of the drone's. After arming it waits up to 1.5 s for the drone to report it can fly before sending the takeoff. If the drone has not risen half the hover height by the end of the takeoff time, it lands (it did not lift off). In the air it lands when no position has been sent for 0.3 s or the estimate and the tracked position disagree by more than 4 cm. It kills the motors when either leaves a 20 cm radius around the takeoff point or rises above 45 cm, or when the drone reports it is locked, crashed or tumbled. The limits are in `config.yaml` under `flight`.

The firmware's emergency-stop watchdog stops the motors if the process hangs or the radio drops. It is fed from the first arming until the session exits, so it runs out about a second after every flying session ends, and the drone then locks itself until it reboots. A kill locks it too. A `--crazyflie-fly` session therefore checks the drone's supervisor when it connects, and if the drone is locked it power-cycles the drone's main processor over the radio and reconnects, which takes about 12 s. `python -m manipulation.crazyflie.reboot` (from `real_robot/`) does the power cycle by hand.

```bash
sg plugdev -c "./run.sh --track --save-path foundationpose_runs/drone1 --camera move --recovery --manipulation crazyflie --crazyflie-fly"
```

Leave the follower's view of the drone wide enough that the drone stays in frame at hover height. Spinning propellers change the drone's silhouette, which can make SAM report a mismatch, and the drone then lands.

Tests (no radio, camera or GPU):

```bash
cd /home/luka/research/real_robot && /opt/conda-envs/foundationpose/bin/python -m unittest manipulation.crazyflie.tests.test_session manipulation.crazyflie.tests.test_flight -v
```
