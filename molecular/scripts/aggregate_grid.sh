#!/bin/bash
#SBATCH --job-name=taste_gnn_agg
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
# NOTE: set --partition / --account at submit time per cluster.

# ============================================================================
# TASTE_GNN GRID SEARCH AGGREGATION SCRIPT
# ============================================================================
# Collects per-config results, writes grid_summary.csv, and logs the best
# hyperparameter configuration. Auto-submitted by submit_grid.sh after all
# array tasks complete.
#
# REQUIRED:
#   OUTPUT_DIR : Directory containing one subdirectory per grid config
#   GRID_PATH  : Path to the original grid YAML (logging only)
#   RUN_NAME   : Name of the run (logging only)
#
# OPTIONAL (defaults shown):
#   CODE_DIR        = $HOME/sustainable_protein_foundation_model
#   CONDA_ENV_NAME  = sustainable_protein_foundation_model
#   CONDA_SH        = $HOME/miniconda3/etc/profile.d/conda.sh
# ============================================================================

set -e
set -u

if [ -z "${OUTPUT_DIR:-}" ]; then
    echo "ERROR: OUTPUT_DIR not set"
    exit 1
fi

CODE_DIR="${CODE_DIR:-${HOME}/sustainable_protein_foundation_model}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-sustainable_protein_foundation_model}"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"

echo "============================================================================"
echo "TASTE_GNN GRID AGGREGATION (job ${SLURM_JOB_ID})"
echo "============================================================================"
echo "Output dir : ${OUTPUT_DIR}"
echo "Grid       : ${GRID_PATH:-unknown}"
echo "Run name   : ${RUN_NAME:-unknown}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"
cd "${CODE_DIR}"

if [ ! -d "${OUTPUT_DIR}" ]; then
    echo "ERROR: Output directory not found: ${OUTPUT_DIR}"
    exit 1
fi

RUN_COUNT=$(ls -d "${OUTPUT_DIR}"/*/ 2>/dev/null | wc -l | tr -d ' ')
echo "Found ${RUN_COUNT} run directories in ${OUTPUT_DIR}"

if [ "${RUN_COUNT}" -eq 0 ]; then
    echo "ERROR: No run directories found in ${OUTPUT_DIR}"
    exit 1
fi

python -m molecular.src.train.grid_search \
    --action aggregate \
    --results_dir "${OUTPUT_DIR}"

echo "Results saved to: ${OUTPUT_DIR}/grid_summary.csv"
