#!/usr/bin/env bash
# Segment objects with an HF SAM tracker and/or track them with FoundationPose.
#
#   ./run.sh --setup       --save-path foundationpose_runs/run1
#   ./run.sh --setup-track --save-path foundationpose_runs/run1 [--num-seconds 30] [--save-video]
#   ./run.sh --track       --save-path foundationpose_runs/run1   # reuse an existing run
#   ./run.sh --track       --save-path foundationpose_runs/run1 --fp-mode batched
#   ./run.sh --setup-track --save-path foundationpose_runs/run1 --reset   # park camera arm, then mask + track
#   ./run.sh --track       --save-path foundationpose_runs/run1 --manipulation basic [--manipulation-dry-run]
#   ./run.sh --track       --save-path foundationpose_runs/run1 --recovery --manipulation policy [--manipulation-logging] [--manipulation-dry-run]
#   ./run.sh --track       --save-path foundationpose_runs/run1 --log-variation
#   ./run.sh --setup-track --save-path foundationpose_runs/run1 --recovery [--sam-model sam2-small]
#   ./run.sh --track       --save-path foundationpose_runs/run1 --recovery [--sam-fps 2 --iou-threshold 0.4 --failure-limit 3]
#
# Setup masks come from the HF sam3 tracker; with --recovery from the recovery config's model,
# and tracking re-registers an object after consecutive SAM-vs-FP silhouette mismatches.
# Recovery defaults (640x480 @ 60 fps, 2 Hz SAM) are in pipeline/recovery/recovery.yaml (--recovery-config).
# --fp-mode sequential (default) runs FoundationPose's track_one per object;
# --fp-mode batched refines every object in one network pass per iteration;
# --fp-mode batched-render also renders every object's crop in one call.
# --manipulation basic adds the click-to-follow arm test (manipulation/basic_manipulation).
# --manipulation policy builds the sim policy's observations from the arm (gravity compensation,
# never commanded) and the tracked object at the policy rate; s starts/ends an episode
# (manipulation/policy). --manipulation-logging writes save-path/policy_log.jsonl.
# --log-variation keeps every tracked pose and, when the window closes or
# --num-seconds expires, writes save-path/pose_variation.txt (mean, std,
# percentiles, and a pose about once per second).
#
# Exactly one mode flag is required; every other argument is forwarded to both
# stages: pipeline/setup_stage.py and pipeline/track_stage.py (see: python -m pipeline.setup_stage --help).
# Both stages run in the vision env.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV="vision"

usage() {
    sed -n '2,31p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
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
# Conda activation hooks (e.g. cuda-nvcc) read unset variables.
set +u
conda activate "${ENV}"
set -u
# real_robot/ for manipulation, real_robot/vision for pipeline and calibration.
export PYTHONPATH="${SCRIPT_DIR}/..:${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

run_stage() {
    local module="$1"
    echo "[${ENV}] $(command -v python) -m ${module}"
    python -m "${module}" "${forward[@]}"
}

if [[ "${mode}" == "setup" || "${mode}" == "setup-track" ]]; then
    run_stage pipeline.setup_stage
fi
if [[ "${mode}" == "track" || "${mode}" == "setup-track" ]]; then
    run_stage pipeline.track_stage
fi
