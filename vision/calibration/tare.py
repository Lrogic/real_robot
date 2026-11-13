"""RealSense tare calibration of the D405 against the ChArUco board, with a before/after depth check.

    cd vision && /opt/conda-envs/vision/bin/python -m calibration.tare
    cd vision && /opt/conda-envs/vision/bin/python -m calibration.tare --restore data/tare/<run>/calibration_before.bin

Camera only; no arm is touched. Stop the tracking pipeline first (one process per camera). Do not
move the camera or the follower arm: T_base_camera is only valid at the parked pose.

Why
    The D405 depth reads short by an amount that grows with distance squared (4.6 mm at 0.25 m,
    15.6 mm at 0.42 m, 19.9 mm at 0.52 m; docs/block_height_offset.md, calibration/check_depth.py).
    That is the signature of a constant disparity offset in the stereo calibration, which is what
    the camera's own tare calibration corrects from one flat target at a known distance.

What it does (interactive; nothing is written to the camera without typing "write")
    1. Saves the camera's current calibration table to data/tare/<timestamp>/calibration_before.bin.
    2. Measures the ground truth with check_depth: the board plane's distance along the depth
       camera's optical axis, from the colour image's board pose (PnP on the ChArUco corners).
       The plane is moved from the colour frame into the depth frame with the camera's own
       extrinsics; on the D405 the colour image comes from the left imager, so they are nearly
       identical, and the script prints the difference.
    3. Runs tare with that distance (firmware mode at 256x144@90, falling back to host-assisted
       mode at 1280x720@30 if the firmware refuses).
    4. Writes the new table to flash if you type "write", and reboots the camera
       (hardware_reset). The D405 (firmware 5.17) ignores a table set in RAM with
       set_calibration_table: depth was unchanged to 0.1 mm whether it was set before a stream
       started or while one ran. It only uses the flash table, loaded at boot, so there is no
       way to check a tare result before writing it.
    5. Re-runs check_depth wherever you put the board, as many times as you like.
    6. Keeps the new table, or writes the saved one back and reboots again.
    Everything, including the new table and both health numbers, goes to data/tare/<timestamp>/.

Board placement for step 2
    Flat, still, covering the middle of the image (tare averages a centred region), roughly facing
    the camera (a tilt under about 20 deg is fine), at about the block's distance (0.40-0.42 m),
    good light (the D405 has no projector, the board texture is what the stereo matches on).

Reading the result
    Tare's health numbers are (before, after) fractional depth errors at the target, so after
    should be near 0. The real test is step 5: within 1-2 mm at 0.25, 0.42 and 0.52 m. If it is
    off by a similar few mm at every distance afterwards, suspect the colour/depth origin
    assumption in step 2; use the software correction from the doc instead and restore.

First result (2026-11-13, this D405, board at 0.417 m, data/tare/20261113_105800): health
-3.58% before, -0.04% after. Depth minus board plane went from -4.6 / -15.8 / -19.9 mm to
+0.4 / -0.9 / -3.8 mm at 0.25 / 0.42 / 0.52 m. Written to flash.
"""

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pyrealsense2 as rs

from .board import DEFAULT_DATA_DIR
from .check_depth import DEFAULT_CALIBRATION, load_T_base_camera, measure, report

TARE_DIR = DEFAULT_DATA_DIR / "tare"
ACCURACY = {"very_high": 0, "high": 1, "medium": 2, "low": 3}


def find_device():
    devices = list(rs.context().query_devices())
    if len(devices) != 1:
        raise SystemExit(f"Expected exactly one RealSense device, found {len(devices)}. Is the pipeline still running?")
    dev = devices[0]
    info = {k: dev.get_info(getattr(rs.camera_info, k)) for k in
            ("name", "serial_number", "firmware_version", "usb_type_descriptor", "product_line", "product_id")
            if dev.supports(getattr(rs.camera_info, k))}
    return dev, info


def depth_to_color(dev):
    """(R, t) with p_color = R @ p_depth + t, metres, from the device's stream profiles."""
    def profile(stream, fmt):
        for sensor in dev.query_sensors():
            for p in sensor.get_stream_profiles():
                if p.stream_type() == stream and p.format() == fmt:
                    return p
        raise RuntimeError(f"no {stream} {fmt} profile")
    ext = profile(rs.stream.depth, rs.format.z16).get_extrinsics_to(profile(rs.stream.color, rs.format.bgr8))
    return np.array(ext.rotation).reshape(3, 3).T, np.array(ext.translation)   # librealsense stores column-major


def ground_truth(rows, R, t):
    """Median board-plane distance along the depth camera's optical axis, mm, plus the colour-frame value."""
    gt_depth, gt_color = [], []
    for r in rows:
        n, d = np.array(r["plane_normal"]), r["plane_d_m"]
        gt_color.append(d / n[2])
        n_d, d_d = R.T @ n, d - n @ t                     # plane n.p_c = d with p_c = R p_d + t
        gt_depth.append(d_d / n_d[2])
    return float(np.median(gt_depth) * 1e3), float(np.median(gt_color) * 1e3), float(np.std(gt_depth) * 1e3)


def check(T_base_camera, frames, label, log):
    print(f"\n--- depth check: {label}")
    rows, tried, _ = measure(frames, T_base_camera)
    ok = report(rows, tried, T_base_camera)
    if ok:
        med = lambda k: float(np.median([r[k] for r in rows]))
        log["checks"].append({"label": label, "distance_m": med("board_distance_m"), "tilt_deg": med("tilt_deg"),
                              "z_diff_mm": med("z_diff_mm"), "z_diff_percent": med("z_diff_percent"),
                              "frames": len(rows)})
    return rows if ok else None


def progress(p):
    print(f"\r  tare progress {p:5.1f}%", end="", flush=True)


def start_depth(serial, w, h, fps):
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
    if not cfg.can_resolve(rs.pipeline_wrapper(pipe)):
        raise RuntimeError(f"depth {w}x{h}@{fps} is not available")
    profile = pipe.start(cfg)
    sensor = profile.get_device().first_depth_sensor()
    saved = {}
    for opt, value in ((rs.option.emitter_enabled, 0), (rs.option.thermal_compensation, 0)):
        if sensor.supports(opt):
            saved[opt] = sensor.get_option(opt)
            sensor.set_option(opt, value)
    for _ in range(30):                                   # let auto exposure settle
        pipe.wait_for_frames()
    return pipe, profile.get_device().as_auto_calibrated_device(), sensor, saved


def stop_depth(pipe, sensor, saved):
    for opt, value in saved.items():
        try:
            sensor.set_option(opt, value)
        except RuntimeError:
            pass
    pipe.stop()


def run_tare(serial, gt_mm, params, mode, timeout_s):
    """Returns (new_table, (health_before, health_after), mode_used)."""
    if mode in ("auto", "firmware"):
        pipe, adev, sensor, saved = start_depth(serial, 256, 144, 90)
        try:
            print("Running tare in firmware mode (256x144@90) ...")
            table, health = adev.run_tare_calibration(gt_mm, json.dumps(params), progress, int(timeout_s * 1000))
            print()
            return list(table), tuple(health), "firmware"
        except RuntimeError as e:
            print(f"\n  firmware mode failed: {e}")
            if mode == "firmware":
                raise
        finally:
            stop_depth(pipe, sensor, saved)
    pipe, adev, sensor, saved = start_depth(serial, 1280, 720, 30)
    try:
        print("Running tare in host-assisted mode (1280x720@30) ...")
        adev.run_tare_calibration(gt_mm, json.dumps({**params, "host assistance": 1}), progress, 5000)
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            frame = pipe.wait_for_frames().get_depth_frame()
            table, health = adev.process_calibration_frame(frame, progress, 5000)
            if len(table):
                print()
                return list(table), tuple(health), "host"
        raise RuntimeError("host-assisted tare timed out")
    finally:
        stop_depth(pipe, sensor, saved)


def write_and_reboot(table, serial, wait_s=30.0):
    """Write `table` to flash, hardware-reset the camera and wait for it to come back."""
    dev, _ = find_device()
    adev = dev.as_auto_calibrated_device()
    adev.set_calibration_table(list(table))
    adev.write_calibration()
    print("Written to flash. Rebooting the camera ...")
    dev.hardware_reset()
    time.sleep(3.0)
    end = time.monotonic() + wait_s
    while time.monotonic() < end:
        devices = rs.context().query_devices()
        if any(d.get_info(rs.camera_info.serial_number) == serial for d in devices):
            time.sleep(2.0)                               # let it finish enumerating
            print("Camera is back.")
            return
        time.sleep(0.5)
    raise SystemExit(f"Camera {serial} did not come back within {wait_s:.0f} s; unplug and replug it.")


def restore(path):
    _, info = find_device()
    table = Path(path).read_bytes()
    print(f"Writing {path} ({len(table)} bytes) to {info.get('name')} {info.get('serial_number')} flash.")
    if input('Type "write" to confirm: ').strip() != "write":
        print("Nothing written.")
        return
    write_and_reboot(table, info["serial_number"])
    print("Restored.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, default=30, help="board frames per depth check")
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION), help="T_base_camera for base-frame shifts")
    parser.add_argument("--mode", choices=("auto", "firmware", "host"), default="auto")
    parser.add_argument("--accuracy", choices=tuple(ACCURACY), default="medium", help="tare accuracy (firmware default: medium)")
    parser.add_argument("--timeout", type=float, default=60.0, help="tare timeout, s")
    parser.add_argument("--restore", default=None, help="write this saved calibration table back to flash and exit")
    args = parser.parse_args()

    if args.restore:
        restore(args.restore)
        return

    dev, info = find_device()
    print("Camera:", ", ".join(f"{k} {v}" for k, v in info.items()))
    adev = dev.as_auto_calibrated_device()
    out = TARE_DIR / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True)
    before = list(adev.get_calibration_table())
    (out / "calibration_before.bin").write_bytes(bytes(before))
    print(f"Saved the current calibration table ({len(before)} bytes) to {out / 'calibration_before.bin'}")

    R, t = depth_to_color(dev)
    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    print(f"Depth to colour extrinsics: translation {np.round(t * 1e3, 2).tolist()} mm, rotation {angle:.3f} deg")

    T_base_camera = load_T_base_camera(args.calibration)
    log = {"camera": info, "time": out.name, "extrinsics_depth_to_color": {"R": R.tolist(), "t_m": t.tolist()},
           "checks": [], "written": False}

    input("\nBoard at the tare pose (flat, still, centred, ~0.4 m)? Press Enter to measure the ground truth.")
    rows = check(T_base_camera, args.frames, "before tare, tare pose", log)
    if rows is None:
        raise SystemExit("No board pose; fix the board placement and rerun.")
    gt_mm, gt_color_mm, gt_std = ground_truth(rows, R, t)
    print(f"\nGround truth along the depth axis: {gt_mm:.1f} mm (colour axis {gt_color_mm:.1f} mm, "
          f"frame-to-frame std {gt_std:.2f} mm). The firmware takes whole millimetres: {round(gt_mm)} mm.")
    params = {"accuracy": ACCURACY[args.accuracy]}
    log.update(ground_truth_mm=gt_mm, ground_truth_color_mm=gt_color_mm, tare_params=params)
    if input("Run tare now? [y/N] ").strip().lower() != "y":
        (out / "log.json").write_text(json.dumps(log, indent=2))
        print(f"Stopped before tare. Log in {out}")
        return

    table, health, mode = run_tare(info["serial_number"], float(round(gt_mm)), params, args.mode, args.timeout)
    (out / "calibration_tare.bin").write_bytes(bytes(table))
    log.update(tare_mode=mode, health_before=health[0], health_after=health[1])
    print(f"Tare done ({mode} mode). Health before {health[0] * 100:+.2f}%, after {health[1] * 100:+.2f}% "
          f"(depth error at the target; implied before {(health[0] + 1) * round(gt_mm):.1f} mm)")
    (out / "log.json").write_text(json.dumps(log, indent=2))
    print("This camera only uses the table in flash, so checking the result means writing it and rebooting.\n"
          f"The old table is saved in {out / 'calibration_before.bin'} and you will be offered to put it back.")
    if input('Type "write" to write the new table and reboot the camera; anything else stops here: ').strip() != "write":
        print(f"Nothing written. Log in {out}")
        return
    write_and_reboot(table, info["serial_number"])
    log["written"] = True
    (out / "log.json").write_text(json.dumps(log, indent=2))

    check(T_base_camera, args.frames, "after tare, tare pose", log)
    while True:
        label = input('\nMove the board and type a label to check again (e.g. "0.25 m"), or Enter to finish: ').strip()
        if not label:
            break
        check(T_base_camera, args.frames, f"after tare, {label}", log)

    print("\nSummary (depth minus colour board plane):")
    for c in log["checks"]:
        print(f"  {c['label']:<32} {c['distance_m']:.3f} m  {c['z_diff_mm']:+6.1f} mm  ({c['z_diff_percent']:+.2f}%)")
    if input('\nType "keep" to keep the new table; anything else writes the old one back: ').strip() == "keep":
        print(f"Kept. To undo later: python -m calibration.tare --restore {out / 'calibration_before.bin'}")
    else:
        write_and_reboot(before, info["serial_number"])
        log["written"] = False
        log["restored"] = True
        print("Old table restored.")
    (out / "log.json").write_text(json.dumps(log, indent=2))
    print(f"Log in {out}")


if __name__ == "__main__":
    main()
