#!/usr/bin/env bash
# ChArUco eye-to-hand calibration of the parked D405 against the manipulator base.
#
#   ./run.sh verify                                   # check the existing print (live camera)
#   ./run.sh collect [--session data/sessions/NAME] [--camera-arm-joints j1,..,j6]
#   ./run.sh review    --session data/sessions/NAME [--list | --redetect [--write]]
#   ./run.sh calibrate --session data/sessions/NAME [--method auto] [--exclude 3,7]
#   ./run.sh generate  [--out data/board.png]        # only for a new print
#
# verify/collect need trossen_arm + pyrealsense2 (fp_robot env). review/calibrate/generate
# run offline in the sam3 env, whose OpenCV 4.x still has calibrateHandEye and
# calibrateRobotWorldHandEye (OpenCV 5.0 in fp_robot does not).
# Override with HW_ENV=... / OFFLINE_ENV=...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HW_ENV="${HW_ENV:-fp_robot}"
OFFLINE_ENV="${OFFLINE_ENV:-sam3}"

usage() {
    sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 2
}

[[ $# -ge 1 ]] || usage
cmd="$1"
shift
case "${cmd}" in
    verify)    env="${HW_ENV}";      module=verify_board ;;
    collect)   env="${HW_ENV}";      module=collect ;;
    review)    env="${OFFLINE_ENV}"; module=review ;;
    calibrate) env="${OFFLINE_ENV}"; module=calibrate ;;
    generate)  env="${OFFLINE_ENV}"; module=generate_board ;;
    *) usage ;;
esac

CONDA_BASE="$(conda info --base 2>/dev/null || echo /opt/anaconda3)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
set +u
conda activate "${env}"
set -u
# Relative paths (e.g. --session data/sessions/NAME) resolve against this directory.
cd "${SCRIPT_DIR}"
echo "[${env}] python -m calibration.${module} $*"
PYTHONPATH="${SCRIPT_DIR}/..${PYTHONPATH:+:${PYTHONPATH}}" exec python -m "calibration.${module}" "$@"
