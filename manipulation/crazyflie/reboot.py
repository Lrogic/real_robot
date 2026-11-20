"""Power-cycle the Crazyflie's main processor over the radio (env foundationpose, from real_robot/).

    python -m manipulation.crazyflie.reboot [--uri radio://0/80/2M/E7E7E7E7E7]

This clears the supervisor's lock (after a kill, or the emergency-stop watchdog running out when a
session ended). The radio chip stays on, so the drone comes back without touching it. session.py
runs this in its own process: cflib's PowerSwitch opens the Crazyradio directly, and a later link
in the same process then hangs.
"""

import argparse

from .session import CrazyflieConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--uri", default=None, help="Default: uri in manipulation/crazyflie/config.yaml")
    args = parser.parse_args()

    import cflib.crtp
    from cflib.utils.power_switch import PowerSwitch

    uri = args.uri or CrazyflieConfig.load().uri
    cflib.crtp.init_drivers()
    switch = PowerSwitch(uri)
    try:
        switch.stm_power_cycle()
    finally:
        switch.close()
    print(f"Power-cycled the Crazyflie at {uri}")


if __name__ == "__main__":
    main()
