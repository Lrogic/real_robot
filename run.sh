#!/usr/bin/env bash
# Activate the sam3 conda env, capture a frame, and segment objects for FoundationPose.
# Arguments are forwarded to segment_objects.py; --save-path is required, e.g.
#   ./run.sh --save-path foundationpose_runs/run1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="sam3"

CONDA_BASE="$(conda info --base 2>/dev/null || echo /opt/anaconda3)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
echo "Using $(command -v python) ($(python --version 2>&1))"

python "${SCRIPT_DIR}/segment_objects.py" "$@"
