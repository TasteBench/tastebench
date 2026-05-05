#!/usr/bin/env bash
# Re-render every food-similarity table from the committed OOF
# prediction CSVs in results/oof_predictions/. No training, no
# external data.
#
# Runtime: ~25–30 minutes total. The long pole is
# render_table_ablation_features.py (~25 min: 105 BCa bootstrap calls
# parallelised across cores; each call uses the full Python-loop
# bootstrap and takes ~2 min). The other six renders together finish
# in ~5 min.
#
# Outputs (all written to paper/model_results_tables/):
#   table_results.tex                (combined ranking + recall metrics)
#   table_per_category.tex
#   table_per_category_nnls.tex
#   table_per_model_nnls.tex
#   table_ablation_features.tex
#   table_ablation_llm.tex
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python}"

run() {
  echo ""
  echo "[$(date +%H:%M:%S)] food_similarity: $*"
  ( cd "${HERE}" && "${PY}" "$@" ) > /tmp/verify_food_similarity.log 2>&1 \
    || { echo "FAILED — last 30 lines of log:"; tail -30 /tmp/verify_food_similarity.log; exit 1; }
  tail -3 /tmp/verify_food_similarity.log | sed 's/^/  /'
}

run scripts/render_table_results.py
run scripts/render_table_per_category.py
run scripts/render_table_per_category_nnls.py
run scripts/render_table_per_model_nnls.py
run scripts/render_table_ablation_features.py
run scripts/render_table_ablation_llm.py
