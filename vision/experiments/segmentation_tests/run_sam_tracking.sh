#!/usr/bin/env bash
# Run live SAM tracking with the conda env that matches --model.
#
#   ./experiments/segmentation_tests/run_sam_tracking.sh --model sam21-small
#   ./experiments/segmentation_tests/run_sam_tracking.sh --model sam31 --undistort
#
# From real_robot/vision (or any cwd); requires --model except with --help.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISION_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRACK_PY="${SCRIPT_DIR}/test_sam_tracking.py"

model=""
forward=()
want_help=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            [[ $# -ge 2 ]] || { echo "run_sam_tracking.sh: --model requires a value" >&2; exit 2; }
            model="$2"
            forward+=(--model "$2")
            shift 2
            ;;
        -h|--help)
            want_help=1
            forward+=("$1")
            shift
            ;;
        *)
            forward+=("$1")
            shift
            ;;
    esac
done

env_for_model() {
    case "$1" in
        sam21-small|sam21-base-plus|sam21-large) echo sam21 ;;
        sam31) echo sam3 ;;
        *)
            echo "run_sam_tracking.sh: unknown --model '$1' (expected sam21-small, sam21-base-plus, sam21-large, or sam31)" >&2
            return 1
            ;;
    esac
}

if [[ -z "${model}" ]]; then
    if [[ "${want_help}" -eq 1 ]]; then
        env="sam21"
    else
        echo "run_sam_tracking.sh: --model is required" >&2
        echo "Example: ${VISION_DIR}/experiments/segmentation_tests/run_sam_tracking.sh --model sam31" >&2
        exit 2
    fi
else
    env="$(env_for_model "${model}")"
fi

CONDA_BASE="$(conda info --base 2>/dev/null || echo /opt/anaconda3)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

set +u
conda activate "${env}"
set -u

cd "${VISION_DIR}"
echo "[${env}] $(command -v python) ${TRACK_PY} ${forward[*]}"
exec python "${TRACK_PY}" "${forward[@]}"
