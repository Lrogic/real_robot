"""List the Crazyflies the Crazyradio can reach (env foundationpose, from real_robot/).

    python -m manipulation.crazyflie.scan [--address E7E7E7E7E7]

Put the printed URI in manipulation/crazyflie/config.yaml.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--address", default="E7E7E7E7E7", help="Radio address to scan (hex, default E7E7E7E7E7)")
    args = parser.parse_args()

    import cflib.crtp

    cflib.crtp.init_drivers()
    found = cflib.crtp.scan_interfaces(int(args.address, 16))
    if not found:
        print("No Crazyflie found. Is it on, is the Crazyradio plugged in, and can this user open it "
              "(udev rule, see manipulation/crazyflie/README.md)?")
    for uri, description in found:
        print(uri, description)


if __name__ == "__main__":
    main()
