#!/usr/bin/env bash
# Segment objects with SAM3 and/or track them with FoundationPose.
#
#   ./run.sh --setup       --save-path foundationpose_runs/run1
#   ./run.sh --setup-track --save-path foundationpose_runs/run1 [--num-seconds 30] [--save-video]
#   ./run.sh --track       --save-path foundationpose_runs/run1   # reuse an existing run
#   ./run.sh --track       --save-path foundationpose_runs/run1 --fp-mode batched
#
# --fp-mode sequential (default) runs FoundationPose's track_one per object;
# --fp-mode batched refines every object in one network pass per iteration.
#
# Exactly one mode flag is required; every other argument is forwarded to both
# stages (see: python segment_objects.py --help).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_ENV="sam3"
TRACK_ENV="fp_robot"

usage() {
    sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 2
}

mode=""
forward=()
for arg in "$@"; do
    case "${arg}" in
        --setup|--setup-track|--track)
            [[ -z "${mode}" ]] || { echo "Only one mode flag allowed" >&2; usage; }
            mode="${arg#--}"
            ;;
        *) forward+=("${arg}") ;;
    esac
done
[[ -n "${mode}" ]] || usage

CONDA_BASE="$(conda info --base 2>/dev/null || echo /opt/anaconda3)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

run_stage() {
    local env="$1" script="$2"
    # Conda activation hooks (e.g. fp_robot's cuda-nvcc) read unset variables.
    set +u
    conda activate "${env}"
    set -u
    echo "[${env}] $(command -v python) ${script}"
    python "${SCRIPT_DIR}/${script}" "${forward[@]}"
    set +u
    conda deactivate
    set -u
}

if [[ "${mode}" == "setup" || "${mode}" == "setup-track" ]]; then
    run_stage "${SETUP_ENV}" segment_objects.py
fi
if [[ "${mode}" == "track" || "${mode}" == "setup-track" ]]; then
    run_stage "${TRACK_ENV}" track_objects.py
fi
