#!/usr/bin/env bash
# Segment objects with an HF SAM tracker and/or track them with FoundationPose.
#
#   ./run.sh --setup       --save-path foundationpose_runs/run1 --camera parked
#   ./run.sh --setup-track --save-path foundationpose_runs/run1 --camera parked [--num-seconds 30] [--save-video]
#   ./run.sh --setup-track --save-path foundationpose_runs/run1 --camera move --recovery
#   ./run.sh --track       --save-path foundationpose_runs/run1 --camera parked   # reuse an existing run
#   ./run.sh --track       --save-path foundationpose_runs/run1 --camera parked --fp-mode batched
#   ./run.sh --track       --save-path foundationpose_runs/run1 --camera parked --manipulation basic [--manipulation-dry-run]
#   ./run.sh --track       --save-path foundationpose_runs/run1 --camera parked --recovery --manipulation policy [--manipulation-logging] [--manipulation-dry-run] [--policy-control shadow|step|slow|full]
#   ./run.sh --track       --save-path foundationpose_runs/run1 --camera parked --log-variation
#   ./run.sh --setup-track --save-path foundationpose_runs/run1 --camera parked --recovery [--sam-model sam2-small]
#   ./run.sh --track       --save-path foundationpose_runs/run1 --camera parked --recovery [--sam-fps 2 --iou-threshold 0.4 --failure-limit 3]
#
# --camera (required) sets up the follower (camera) arm before anything else and holds it in
# position mode until the run ends (pipeline/robot_arm/camera_arm.py; calibration files in
# pipeline/camera_arm.yaml, --camera-config):
#   parked  move the follower to the parked calibration's joints; T_base_camera is that calibration's.
#   move    gravity compensation and a live camera window: guide the follower, Enter/space locks it.
#           T_base_camera comes from the two-arm calibration and the locked follower pose. At exit
#           the follower returns to the parked joints before it goes idle.
# --track must use the run's setup mode (camera_pose.json); move returns to the setup's joints.
# Setup masks come from the HF sam3 tracker; with --recovery from the recovery config's model,
# and tracking re-registers an object after consecutive SAM-vs-FP silhouette mismatches.
# Recovery defaults (640x480 @ 60 fps, 2 Hz SAM) are in pipeline/recovery/recovery.yaml (--recovery-config).
# --fp-mode sequential (default) runs FoundationPose's track_one per object;
# --fp-mode batched refines every object in one network pass per iteration;
# --fp-mode batched-render also renders every object's crop in one call.
# --manipulation basic adds the click-to-follow arm test (manipulation/basic_manipulation).
# --manipulation policy runs the sim policy on the arm and the tracked object (manipulation/policy):
# s moves to the ready pose and arms an episode (or stops), r sends commands, n steps (step mode),
# k kills to gravity compensation. --policy-control overrides the config's control.mode (shadow
# never commands the arm). --manipulation-logging writes save-path/policy_log.jsonl.
# --log-variation keeps every tracked pose and, when the window closes or
# --num-seconds expires, writes save-path/pose_variation.txt (mean, std,
# percentiles, and a pose about once per second).
#
# Exactly one mode flag is required; every other argument is forwarded to pipeline/run_stages.py,
# which runs the setup and/or track stage in one process, so the follower stays held between
# them (options: python -m pipeline.setup_stage --help). Runs in the vision env.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV="vision"

usage() {
    sed -n '2,43p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
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

echo "[${ENV}] $(command -v python) -m pipeline.run_stages ${mode}"
python -m pipeline.run_stages "${mode}" "${forward[@]}"
