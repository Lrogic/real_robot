#!/usr/bin/env bash
# Segment objects with SAM3 and/or track them with FoundationPose.
#
#   ./run.sh --setup       --save-path foundationpose_runs/run1
#   ./run.sh --setup-track --save-path foundationpose_runs/run1 [--num-seconds 30] [--save-video]
#   ./run.sh --track       --save-path foundationpose_runs/run1   # reuse an existing run
#   ./run.sh --track       --save-path foundationpose_runs/run1 --fp-mode batched
#   ./run.sh --setup-track --save-path foundationpose_runs/run1 --reset   # park camera arm, then mask + track
#   ./run.sh --track       --save-path foundationpose_runs/run1 --manipulation [--manipulation-dry-run]
#   ./run.sh --track       --save-path foundationpose_runs/run1 --log-variation
#   ./run.sh --setup-track --save-path foundationpose_runs/run1 --recovery [--sam-model sam2-small]
#   ./run.sh --track       --save-path foundationpose_runs/run1 --recovery [--sam-fps 2 --iou-threshold 0.4 --failure-limit 3]
#
# --recovery runs both stages in the vision env: setup masks come from the HF SAM model, and
# tracking re-registers an object after consecutive SAM-vs-FP silhouette mismatches. Defaults
# (640x480 @ 60 fps, sam3, 2 Hz SAM) are in fp_capture/recovery.yaml (--recovery-config).
# --fp-mode sequential (default) runs FoundationPose's track_one per object;
# --fp-mode batched refines every object in one network pass per iteration;
# --fp-mode batched-render also renders every object's crop in one call.
# --log-variation keeps every tracked pose and, when the window closes or
# --num-seconds expires, writes save-path/pose_variation.txt (mean, std,
# percentiles, and a pose about once per second).
#
# Exactly one mode flag is required; every other argument is forwarded to both
# stages (see: python segment_objects.py --help).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_ENV="sam3"
TRACK_ENV="fp_robot"
RESET_ENV="fp_robot"
# FoundationPose and the HF SAM tracker in one process: fp_robot plus transformers.
RECOVERY_ENV="vision"

usage() {
    sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
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
for arg in "${forward[@]}"; do
    if [[ "${arg}" == "--recovery" ]]; then
        SETUP_ENV="${RECOVERY_ENV}"
        TRACK_ENV="${RECOVERY_ENV}"
    fi
done

CONDA_BASE="$(conda info --base 2>/dev/null || echo /opt/anaconda3)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

forward_has_reset() {
    for arg in "$@"; do
        [[ "${arg}" == "--reset" ]] && return 0
    done
    return 1
}

run_stage() {
    local env="$1" script="$2"
    # Conda activation hooks (e.g. fp_robot's cuda-nvcc) read unset variables.
    set +u
    conda activate "${env}"
    set -u
    echo "[${env}] $(command -v python) ${script}"
    # real_robot/ for the manipulation package (track_objects.py --manipulation).
    PYTHONPATH="${SCRIPT_DIR}/..${PYTHONPATH:+:${PYTHONPATH}}" python "${SCRIPT_DIR}/${script}" "${forward[@]}"
    set +u
    conda deactivate
    set -u
}

if [[ "${mode}" == "setup" || "${mode}" == "setup-track" ]]; then
    # Follower reset needs trossen_arm (fp_robot); SAM3 setup runs in sam3 (vision with --recovery).
    if forward_has_reset "${forward[@]}"; then
        run_stage "${RESET_ENV}" pipeline_follower_reset.py
        export REAL_ROBOT_FOLLOWER_RESET_DONE=1
    fi
    run_stage "${SETUP_ENV}" segment_objects.py
    unset REAL_ROBOT_FOLLOWER_RESET_DONE 2>/dev/null || true
fi
if [[ "${mode}" == "track" || "${mode}" == "setup-track" ]]; then
    run_stage "${TRACK_ENV}" track_objects.py
fi
