#!/usr/bin/env bash
# Re-render every paper table and figure from the artifacts shipped in
# this directory: committed OOF prediction CSVs, parquet predictions,
# GNN grid outputs, and human-baseline analysis. No external data is
# downloaded; no models are trained.
#
# Inputs:
#   food_similarity/results/oof_predictions/
#   molecular/results/grid/, molecular/results/fart_augmented_test/
#   human_baseline/results/
#
# Runtime is dominated by BCa bootstrap (n=10,000 resamples). The
# slowest single render is render_table_ablation_features.py, which
# sweeps CIs over 105 (model, feature-subset) pairs.
#
# Each area has its own verify.sh; this top-level script chains them.
#
# Usage:  bash verify_paper.sh
#         bash verify_paper.sh --check-diff   # also git-diff outputs

set -euo pipefail

NEURIPS_DIR="$(cd "$(dirname "$0")" && pwd)"

CHECK_DIFF=0
[[ "${1:-}" == "--check-diff" ]] && CHECK_DIFF=1

echo "========================================================"
echo "  TasteBench paper verification"
echo "  Re-rendering all tables from committed OOFs."
echo "  No downloads, no training."
echo "========================================================"

bash "${NEURIPS_DIR}/food_similarity/verify.sh"
bash "${NEURIPS_DIR}/molecular/verify.sh"
bash "${NEURIPS_DIR}/human_baseline/verify.sh"

echo ""
echo "[$(date +%H:%M:%S)] All renders complete."
echo ""
echo "Outputs:"
echo "  ${NEURIPS_DIR}/paper/human_baseline/{human_baseline_table.tex, group_size_curve.pdf}"
echo "  ${NEURIPS_DIR}/paper/model_results_tables/*.tex"
echo "  ${NEURIPS_DIR}/paper/molecular_prediction/*.tex"

if [[ "${CHECK_DIFF}" == "1" ]]; then
  echo ""
  echo "[$(date +%H:%M:%S)] Diffing against committed outputs..."
  if git -C "${NEURIPS_DIR}" diff --stat -- paper/ | grep -q .; then
    echo "Some outputs differ from committed copies; see git diff for details."
    exit 1
  else
    echo "All rendered outputs are byte-identical to the committed copies."
  fi
fi

echo ""
echo "Done."
