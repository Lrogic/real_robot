"""The shoulder and elbow angle offsets of the WXAI arms, and setting them on the drivers.

Both arms read their shoulder and elbow angles off by a constant (docs/arm_joint_offsets.md):

    true angle = reading + OFFSETS_DEG[joint]

The driver corrects this itself through each joint's position_offset, which it adds to commands and
subtracts from feedback (motor position = reported position + position_offset). So the driver value
is minus the offset above. It is stored in the arm's EEPROM and stays until changed.

The joint limits are in the same reported coordinates, so changing position_offset moves every
physical limit by the offset: the elbow could no longer fold its last 0.55 deg (the parked follower
pose would be clipped, about 5.6 mm of error at the block), and the shoulder could be driven 1.71 deg
further into its rest. So --apply and --restore shift each changed joint's position_min and
position_max by the same amount as its readings (old position_offset - new), keeping every
physical limit where it was.

Camera calibrations solved with these offsets (calibration/resolve_joint_offsets.py) are valid only
while the arms report corrected joints, and the calibrations solved without them only while they do
not. Switch both together: vision/pipeline/camera_arm.yaml.

    python -m calibration.joint_offsets                               # show both arms, change nothing
    python -m calibration.joint_offsets --apply --execute             # back up, then set the offsets
    python -m calibration.joint_offsets --restore --execute           # set joints 1 and 2 back to 0
    ... --arm manipulator | follower | both (default both)

Run from vision/ (env vision or fp_robot). Each arm must be folded at rest and not in use by another
program: setting position_offset is refused while a joint is in position mode, and disconnecting
leaves the arm idle. Before changing anything, --execute saves the arm's full configuration with the
driver's save_configs_to_file to calibration/data/arm_configs/<time>_<apply|restore>_<arm>.yaml and
the old offsets to a JSON beside it. Afterwards it saves the configuration again (..._after.yaml) and
reports every value that changed; anything outside position_offset and the position limits is
flagged.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from .board import load_config

OFFSETS_DEG = {1: 1.71, 2: -0.55}      # true = reading + offset; sag map and calibration refits, 2026-11-25
BACKUP_DIR = Path(__file__).resolve().parent / "data" / "arm_configs"


def offsets_rad(num_arm_joints: int = 6) -> np.ndarray:
    o = np.zeros(num_arm_joints)
    for j, deg in OFFSETS_DEG.items():
        o[j] = np.radians(deg)
    return o


def corrected_joints(q_reading) -> np.ndarray:
    """Joint angles as an arm with the offsets applied reports the same physical pose (new = old + offset)."""
    q = np.asarray(q_reading, dtype=float)
    return q + offsets_rad(len(q))


def driver_position_offsets(current: list[float]) -> list[float]:
    """The driver's position_offset vector (all joints, gripper last) with joints 1 and 2 corrected."""
    out = list(current)
    for j, deg in OFFSETS_DEG.items():
        out[j] = -float(np.radians(deg))
    return out


def arms(which: str) -> list[tuple[str, dict]]:
    cfg = load_config()
    out = []
    if which in ("manipulator", "both"):
        out.append(("manipulator", cfg["robot"]))
    if which in ("follower", "both"):
        out.append(("follower", cfg["follower"]))
    return out


def connect(arm_cfg: dict):
    import trossen_arm

    driver = trossen_arm.TrossenArmDriver()
    driver.configure(getattr(trossen_arm.Model, arm_cfg["model"]),
                     getattr(trossen_arm.StandardEndEffector, arm_cfg["end_effector"]), arm_cfg["ip"], False)
    return driver


def describe(name: str, driver) -> dict:
    offsets = list(driver.get_position_offsets())
    q = list(driver.get_all_positions())
    modes = [m.name for m in driver.get_modes()]
    limits = [(float(l.position_min), float(l.position_max)) for l in driver.get_joint_limits()]
    print(f"{name}: modes {modes}")
    print("  position_offset (rad): " + " ".join(f"{v:+.5f}" for v in offsets))
    print("  joints now (deg):      " + " ".join(f"{np.degrees(v):+8.3f}" for v in q[:-1])
          + f"   gripper {q[-1] * 1000:.2f} mm")
    print("  limits joints 1, 2 (deg): " + "  ".join(f"[{np.degrees(lo):+.2f}, {np.degrees(hi):+.2f}]"
                                                    for lo, hi in limits[1:3]))
    return {"position_offsets": offsets, "positions": q, "modes": modes, "position_limits": limits}


def shift_limits(driver, shift: list[float]) -> None:
    """Move each joint's position_min/max by shift (rad), the change in its readings."""
    limits = driver.get_joint_limits()
    for j, d in enumerate(shift):
        if d != 0.0:
            limits[j].position_min = float(limits[j].position_min + d)
            limits[j].position_max = float(limits[j].position_max + d)
    driver.set_joint_limits(limits)


def config_differences(a, b, path="") -> list[str]:
    """Paths where two loaded configuration files differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            out += config_differences(a.get(k), b.get(k), f"{path}.{k}" if path else str(k))
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in config_differences(x, y, f"{path}[{i}]")]
    if isinstance(a, float) and isinstance(b, float):
        return [] if abs(a - b) < 1e-9 else [path]
    return [] if a == b else [path]


EXPECTED_CHANGES = ("position_offset", "position_min", "position_max")


def change(name: str, arm_cfg: dict, restore: bool, execute: bool) -> None:
    import trossen_arm

    driver = connect(arm_cfg)
    try:
        before = describe(name, driver)
        target = list(before["position_offsets"])
        if restore:
            for j in OFFSETS_DEG:
                target[j] = 0.0
        else:
            target = driver_position_offsets(target)
        print("  target position_offset: " + " ".join(f"{v:+.5f}" for v in target))
        if np.allclose(target, before["position_offsets"], atol=1e-7):
            print("  already set; nothing to do")
            return
        if not execute:
            print("  dry run: pass --execute to back up and write it")
            return
        if any(m == trossen_arm.Mode.position.name for m in before["modes"]):
            raise SystemExit(f"{name}: a joint is in position mode; stop the program holding it first")
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S") + ("_restore" if restore else "_apply")
        yaml_path = BACKUP_DIR / f"{stamp}_{name}.yaml"
        driver.save_configs_to_file(str(yaml_path))
        (BACKUP_DIR / f"{stamp}_{name}.json").write_text(json.dumps(
            {"arm": name, "ip": arm_cfg["ip"], "saved": stamp, "before": before, "target_position_offsets": target,
             "offsets_deg_true_minus_reading": {str(k): v for k, v in OFFSETS_DEG.items()},
             "restore": restore, "full_configuration": str(yaml_path)}, indent=2))
        print(f"  backed up the configuration to {yaml_path}")
        # Readings change by -(new - old) position_offset; move the limits with them.
        shift = [float(a - b) for a, b in zip(before["position_offsets"], target)]
        shift_limits(driver, shift)
        try:
            driver.set_position_offsets(target)
        except Exception:
            shift_limits(driver, [-d for d in shift])      # leave the arm as it was
            raise
        after = describe(name, driver)
        lim_moved = np.degrees(np.array(after["position_limits"]) - np.array(before["position_limits"]))
        print("  limit change (deg), joints 0-5 min: " + " ".join(f"{v:+.2f}" for v in lim_moved[:-1, 0]))
        moved = np.degrees(np.array(after["positions"][:-1]) - np.array(before["positions"][:-1]))
        print("  reading change (deg):  " + " ".join(f"{v:+8.3f}" for v in moved)
              + "   (the arm did not move; only its reported angles)")
        if not np.allclose(after["position_offsets"], target, atol=1e-6):
            raise SystemExit(f"{name}: read back {after['position_offsets']}, expected {target}")
        after_path = BACKUP_DIR / f"{stamp}_{name}_after.yaml"
        driver.save_configs_to_file(str(after_path))
        diffs = config_differences(yaml.safe_load(yaml_path.read_text()), yaml.safe_load(after_path.read_text()))
        other = [d for d in diffs if not any(k in d for k in EXPECTED_CHANGES)]
        print(f"  configuration before/after: {len(diffs)} values changed"
              + (": " + ", ".join(diffs) if diffs else ""))
        if other:
            print(f"  WARNING: {name} changed outside position_offset and the position limits: {other}. "
                  f"Compare {yaml_path.name} with {after_path.name}.")
        else:
            print("  nothing else in the configuration changed")
    finally:
        try:
            input(f"\n{name} will be left idle. Make sure it is folded at rest, then press Enter... ")
        except (EOFError, KeyboardInterrupt):
            pass
        driver.cleanup()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = p.add_mutually_exclusive_group()
    action.add_argument("--apply", action="store_true", help="set the offsets")
    action.add_argument("--restore", action="store_true", help="set joints 1 and 2 back to 0")
    p.add_argument("--arm", choices=("manipulator", "follower", "both"), default="both")
    p.add_argument("--execute", action="store_true", help="write to the arm (default: show only)")
    args = p.parse_args()
    print(f"Offsets (true = reading + offset): " + ", ".join(f"joint_{j} {d:+.2f} deg" for j, d in OFFSETS_DEG.items())
          + f"; driver position_offset = " + ", ".join(f"joint_{j} {-np.radians(d):+.5f} rad"
                                                       for j, d in OFFSETS_DEG.items()))
    for name, arm_cfg in arms(args.arm):
        if args.apply or args.restore:
            change(name, arm_cfg, restore=args.restore, execute=args.execute)
        else:
            driver = connect(arm_cfg)
            try:
                if any(m != "idle" for m in describe(name, driver)["modes"]):
                    input(f"\n{name} is not idle and will be left idle. Support it, then press Enter... ")
            finally:
                driver.cleanup()


if __name__ == "__main__":
    main()
