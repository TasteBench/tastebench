#!/bin/bash
#SBATCH --job-name=taste_gnn_grid
#SBATCH --output=%x.%A_%a.out
#SBATCH --error=%x.%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
# NOTE: --partition and --account are cluster-specific. Pass them at submit
# time, e.g. `sbatch --partition=$YOUR_PARTITION --account=$YOUR_ACCOUNT ...`,
# or via the SLURM_PARTITION / SLURM_ACCOUNT env vars in submit_grid_auto.sh.

# ============================================================================
# TASTE_GNN GRID SEARCH ARRAY SCRIPT
# ============================================================================
# Each array task trains one config from the expanded grid. Cluster-specific
# paths come from environment variables; defaults assume a $HOME-based layout.
# Override before submission as needed.
#
# RECOMMENDED USAGE (auto-sizes array based on grid):
#   bash molecular/scripts/submit_grid_auto.sh <grid_yaml_basename>
#
# MANUAL USAGE:
#   sbatch --array=0-N --partition=$YOUR_PARTITION --account=$YOUR_ACCOUNT \
#       --export=ALL,GRID_PATH=/path/to/grid.yaml,RUN_NAME=my_run \
#       molecular/scripts/submit_grid.sh
#
# REQUIRED:
#   GRID_PATH : Absolute path to the grid YAML config
#   RUN_NAME  : Human-readable run name (used for output directory)
#
# OPTIONAL (defaults shown):
#   CODE_DIR        = $HOME/sustainable_protein_foundation_model
#   SCRATCH_DIR     = $HOME/scratch/sustainable_protein_data
#   CONDA_ENV_NAME  = sustainable_protein_foundation_model
#   CONDA_SH        = $HOME/miniconda3/etc/profile.d/conda.sh
# ============================================================================

set -e
set -u

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "ERROR: This script must be run as a SLURM array job."
    echo "RECOMMENDED: bash molecular/scripts/submit_grid_auto.sh <grid_yaml_basename>"
    exit 1
fi

if [ -z "${GRID_PATH:-}" ] || [ -z "${RUN_NAME:-}" ]; then
    echo "ERROR: GRID_PATH and RUN_NAME must be set via --export"
    echo "  GRID_PATH: ${GRID_PATH:-NOT SET}"
    echo "  RUN_NAME:  ${RUN_NAME:-NOT SET}"
    exit 1
fi

CODE_DIR="${CODE_DIR:-${HOME}/sustainable_protein_foundation_model}"
SCRATCH_DIR="${SCRATCH_DIR:-${HOME}/scratch/sustainable_protein_data}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-sustainable_protein_foundation_model}"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
OUTPUT_DIR="${SCRATCH_DIR}/results/taste_gnn/${RUN_NAME}"
LOGS_DIR="${SCRATCH_DIR}/logs"

mkdir -p "${OUTPUT_DIR}" "${LOGS_DIR}"

echo "============================================================================"
echo "TASTE_GNN GRID TASK ${SLURM_ARRAY_TASK_ID} / $((${SLURM_ARRAY_TASK_COUNT} - 1))"
echo "============================================================================"
echo "Grid:       ${GRID_PATH}"
echo "Run name:   ${RUN_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Array ID:   ${SLURM_ARRAY_TASK_ID}"
echo "Job ID:     ${SLURM_ARRAY_JOB_ID}"
echo "Node:       ${SLURMD_NODENAME:-unknown}"
echo "============================================================================"

source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"
echo "Conda environment activated: ${CONDA_ENV_NAME}"

cd "${CODE_DIR}"
echo "cwd: ${CODE_DIR}"

if [ ! -f "${GRID_PATH}" ]; then
    echo "ERROR: Grid file not found: ${GRID_PATH}"
    exit 1
fi

RUN_OUT=$(python - <<PY
import sys, yaml
from pathlib import Path
from molecular.src.train.grid_search import expand_grid

cfgs = expand_grid(Path("${GRID_PATH}"))
cfg  = cfgs[${SLURM_ARRAY_TASK_ID}]
out  = Path("${OUTPUT_DIR}") / cfg["run_name"]
out.mkdir(parents=True, exist_ok=True)
(out / "config.in.yaml").write_text(yaml.safe_dump(cfg))
print(str(out), file=sys.stdout)
print(f"Selected config: {cfg['run_name']}", file=sys.stderr)
PY
)

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to select/write config for task ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

echo "Config written to: ${RUN_OUT}/config.in.yaml"

python -m molecular.src.train.train_dmpnn \
    --config "${RUN_OUT}/config.in.yaml" \
    --output_dir "${RUN_OUT}"

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID} completed (results: ${RUN_OUT})"
else
    echo "Task ${SLURM_ARRAY_TASK_ID} FAILED with exit code ${EXIT_CODE}"
    exit ${EXIT_CODE}
fi

# Submit aggregation only after the final array task finishes.
if [ "${SLURM_ARRAY_TASK_ID}" -eq "$((${SLURM_ARRAY_TASK_COUNT} - 1))" ]; then
    echo "Final task complete - submitting aggregation job."
    AGG_JOB=$(sbatch \
        --dependency=afterok:${SLURM_ARRAY_JOB_ID} \
        --export=ALL,OUTPUT_DIR="${OUTPUT_DIR}",GRID_PATH="${GRID_PATH}",RUN_NAME="${RUN_NAME}" \
        "${CODE_DIR}/molecular/scripts/aggregate_grid.sh" | awk '{print $NF}')
    echo "Aggregation job submitted: ${AGG_JOB}"
fi
