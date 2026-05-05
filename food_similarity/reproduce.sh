#!/usr/bin/env bash
# End-to-end reproduction for the food_similarity section.
#
# Regenerates the food_similarity OOF predictions and tables from raw
# inputs (caches + NECTAR data). Assumes prerequisites are in place:
#   - shared/data/caches/*.pkl, *.csv (see README.md → Caches)
#   - results/oof_predictions/*.csv   (already committed)
#   - data/{consolidated_datasets,food_atlas,foodb_2020_04_07_csv,
#           product_images,taste_like}/  (gated NECTAR + public)
#
# This is the heavyweight train-from-scratch path (~2-3h on 8 cores).
# For a quick reviewer-side render of only the paper tables/figures
# from committed OOFs, use verify_paper.sh instead.
#
# Pinned versions live in environment.yml; git SHA at run-time is
# recorded by Phase 0 below.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
NEURIPS_DIR="$(cd "${HERE}/.." && pwd)"
MOL="${NEURIPS_DIR}/molecular"
PY="${PY:-python}"

echo "== Phase 0: record git SHA + environment ==============="
git -C "${NEURIPS_DIR}" rev-parse HEAD 2>/dev/null \
  | tee "${HERE}/results/repro_git_sha.txt" || true

echo ""
echo "== Phase 1: verify input caches ========================"
(cd "${NEURIPS_DIR}" && \
  sha256sum -c "${HERE}/results/input_cache_sha256.txt" --ignore-missing) \
  && echo "input caches match manifest" \
  || { echo "WARNING: input cache drift detected; numbers may differ"; }

echo ""
echo "== Phase 2: archived BCa CI sanity check ==============="
"${PY}" "${HERE}/scripts/sanity_check_archived_cis.py"

echo ""
echo "== Phase 3: product_features.pkl regeneration =========="
"${PY}" "${HERE}/prepare_data.py"

echo ""
echo "== Phase 4: GNN compound encoder comparison ============"
"${PY}" "${HERE}/train/run_taste_gnn_nectar.py"

echo ""
echo "== Phase 5: regenerate v4.0 supervised + distance OOFs ="
"${PY}" -m train.regenerate_supervised_oofs
"${PY}" -m train.regenerate_distance_oofs

echo ""
echo "== Phase 6: per-model nested NNLS ensembles ============"
"${PY}" -m train.regenerate_per_model_nnls
"${PY}" -m train.regenerate_bt_aggregations

echo ""
echo "== Phase 7: BCa CIs for paper-table cells (n_jobs=8) ==="
"${PY}" "${HERE}/scripts/compute_cis_parallel.py" \
  --discover --n-jobs 8 \
  --out "${HERE}/results/cis_final_tables.csv"

echo ""
echo "== Phase 8: LLM CI regeneration ========================"
"${PY}" "${HERE}/scripts/regenerate_llm_cis.py"

echo ""
echo "== Phase 9: render food_similarity paper tables ========"
bash "${HERE}/verify.sh"

echo ""
echo "== Phase 10: render molecular paper tables ============="
bash "${MOL}/verify.sh"

echo ""
echo "Done."
echo "  food_similarity tables:  ${HERE}/results/tables/"
echo "  food_similarity figures: ${HERE}/results/figures/"
echo "  molecular tables:        ${MOL}/results/tables/tex/"
echo ""
echo "Run human_baseline/human_panelist_baseline.py separately for human-baseline numbers."
