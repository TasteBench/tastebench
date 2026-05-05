#!/bin/bash
# ============================================================================
# AUTO-SIZING TASTE_GNN GRID SEARCH SUBMISSION SCRIPT
# ============================================================================
# Wrapper that determines the number of grid configs and submits the array
# job with the right size. Cluster-specific paths come from environment
# variables; defaults match `submit_grid.sh`.
#
# USAGE:
#   bash molecular/scripts/submit_grid_auto.sh <grid_yaml_basename>
#
# OPTIONAL ENVIRONMENT (defaults shown):
#   CODE_DIR         = $HOME/sustainable_protein_foundation_model
#   CONDA_ENV_NAME   = sustainable_protein_foundation_model
#   CONDA_SH         = $HOME/miniconda3/etc/profile.d/conda.sh
#   SLURM_PARTITION  passed as --partition=  to sbatch if set
#   SLURM_ACCOUNT    passed as --account=    to sbatch if set
# ============================================================================

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <grid_yaml_basename>"
    echo "Example: $0 dmpnn_grid.yaml"
    exit 1
fi

CONFIG_ARG="$1"

if [[ "$CONFIG_ARG" == *"/"* ]]; then
    GRID_PATH="$CONFIG_ARG"
else
    GRID_PATH="molecular/configs/${CONFIG_ARG}"
fi

if [ ! -f "$GRID_PATH" ]; then
    echo "ERROR: grid file not found: $GRID_PATH" >&2
    exit 1
fi

CODE_DIR="${CODE_DIR:-${HOME}/sustainable_protein_foundation_model}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-sustainable_protein_foundation_model}"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"

echo "Grid file: ${GRID_PATH}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"
cd "${CODE_DIR}"

N=$(python - <<PY
from molecular.src.train.grid_search import expand_grid
from pathlib import Path
print(len(expand_grid(Path("${GRID_PATH}"))))
PY
)

ARRAY_MAX=$((N - 1))
RUN_ID=$(basename "$GRID_PATH" .yaml)
RUN_NAME="${RUN_ID}_$(date +%Y%m%d_%H%M%S)"

echo "Submitting SLURM array job, size 0-${ARRAY_MAX} (run_name=${RUN_NAME})"

EXTRA_SBATCH=()
[ -n "${SLURM_PARTITION:-}" ] && EXTRA_SBATCH+=(--partition="${SLURM_PARTITION}")
[ -n "${SLURM_ACCOUNT:-}" ]   && EXTRA_SBATCH+=(--account="${SLURM_ACCOUNT}")

sbatch --array=0-${ARRAY_MAX} \
    "${EXTRA_SBATCH[@]}" \
    --export=ALL,GRID_PATH="${CODE_DIR}/${GRID_PATH}",RUN_NAME="${RUN_NAME}" \
    "${CODE_DIR}/molecular/scripts/submit_grid.sh"
