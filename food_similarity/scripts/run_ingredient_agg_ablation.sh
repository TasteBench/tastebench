#!/usr/bin/env bash
# Ingredient-aggregation ablation: run the food-similarity supervised
# pipeline once per ingredient-aggregation variant, save per-variant
# OOFs, and render a comparison table.
#
# Five variants are tested (matching the options implemented in
# prepare_data.py's aggregate_compounds()):
#
#   mean, max, top3_by_conc, weighted_average, log_weighted_average
#
# log_weighted_average is the canonical default; the run order here
# places it LAST so the canonical results/oof_predictions/ files end
# up in their canonical state after the script finishes.
#
# For each variant we capture:
#   - bradley_terry_SNCTI_bench.csv         (supervised BT alone)
#   - nested_bt_gemini_nnls.csv             (BT + Gemini NNLS ensemble)
# and copy them to per-variant paths:
#   - bradley_terry_SNCTI_<variant>.csv
#   - nested_bt_gemini_nnls_<variant>.csv
#
# Requires the gated NECTAR data (see data/GATED.md). Total runtime
# is roughly 30 min per variant × 5 variants = ~2.5 hours on 8 cores.
#
# Usage:  bash food_similarity/scripts/run_ingredient_agg_ablation.sh
#
# Idempotent — re-running won't regenerate variants whose per-variant
# OOFs are already on disk. Pass --force to wipe and rerun.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
FOOD_SIM_DIR="$(cd "${HERE}/.." && pwd)"
PY="${PY:-python}"

VARIANTS=(mean max top3_by_conc weighted_average log_weighted_average)

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
    FORCE=1
fi

OOF_DIR="${FOOD_SIM_DIR}/results/oof_predictions"
mkdir -p "${OOF_DIR}"

cd "${FOOD_SIM_DIR}"

for variant in "${VARIANTS[@]}"; do
    bt_target="${OOF_DIR}/bradley_terry_SNCTI_${variant}.csv"
    nnls_target="${OOF_DIR}/nested_bt_gemini_nnls_${variant}.csv"

    if [[ "${FORCE}" -eq 0 ]] && [[ -f "${bt_target}" ]] && [[ -f "${nnls_target}" ]]; then
        echo "=== [${variant}] both OOFs exist, skipping ==="
        continue
    fi

    echo "================================================="
    echo "[$(date +%H:%M:%S)]  Running ingredient_agg=${variant}"
    echo "================================================="

    # 1. Rebuild product_features.pkl with this variant
    INGREDIENT_AGG="${variant}" "${PY}" prepare_data.py

    # 2. Retrain supervised models on the new compound features.
    #    BT is the only one we need, but the regenerate script
    #    runs all 5 supervised + 4 distance OOFs in one parallel
    #    call (~10 min). Cheaper than refactoring.
    INGREDIENT_AGG="${variant}" "${PY}" -m train.regenerate_supervised_oofs

    # 3. Train the nested BT + Gemini NNLS ensemble. (~25 min: nested LOOCV.)
    INGREDIENT_AGG="${variant}" "${PY}" -m train.regenerate_per_model_nnls

    # 4. Snapshot the variant-specific OOFs.
    cp "${OOF_DIR}/bradley_terry_SNCTI_bench.csv" "${bt_target}"
    cp "${OOF_DIR}/nested_bt_gemini_nnls.csv"     "${nnls_target}"

    echo "[$(date +%H:%M:%S)]  ${variant} done; saved to ${bt_target##*/} + ${nnls_target##*/}"
    echo ""
done

echo "================================================="
echo "All 5 variants complete."
echo "Per-variant OOFs in ${OOF_DIR}/"
echo "Render the ablation table next:"
echo "  python scripts/render_table_ingredient_agg.py"
echo "================================================="
