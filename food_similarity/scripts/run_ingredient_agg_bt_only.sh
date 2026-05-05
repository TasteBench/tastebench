#!/usr/bin/env bash
# Faster path through the ingredient-aggregation ablation: BT-only
# (skips the ~25-min nested NNLS per variant).
#
# For each of the 5 INGREDIENT_AGG variants, runs:
#   1. prepare_data.py    (~5 min — rebuilds product_features.pkl)
#   2. regenerate_supervised_oofs.py  (~10 min — retrains BT/HBT/KSVM/ridge/LGBM)
#
# Then snapshots bradley_terry_SNCTI_bench.csv to a per-variant path.
# Total runtime ~75 min on 8 cores.
#
# Usage:  bash food_similarity/scripts/run_ingredient_agg_bt_only.sh
#
# Idempotent: skips variants whose per-variant BT OOF is already on disk.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
FOOD_SIM_DIR="$(cd "${HERE}/.." && pwd)"
PY="${PY:-python}"

VARIANTS=(mean max top3_by_conc weighted_average log_weighted_average)

OOF_DIR="${FOOD_SIM_DIR}/results/oof_predictions"
mkdir -p "${OOF_DIR}"

cd "${FOOD_SIM_DIR}"

for variant in "${VARIANTS[@]}"; do
    target="${OOF_DIR}/bradley_terry_SNCTI_${variant}.csv"
    if [[ -f "${target}" ]]; then
        echo "=== [${variant}] BT OOF exists, skipping ==="
        continue
    fi

    echo "================================================="
    echo "[$(date +%H:%M:%S)]  ingredient_agg=${variant}"
    echo "================================================="

    INGREDIENT_AGG="${variant}" "${PY}" prepare_data.py
    INGREDIENT_AGG="${variant}" "${PY}" -m train.regenerate_supervised_oofs

    cp "${OOF_DIR}/bradley_terry_SNCTI_bench.csv" "${target}"
    echo "[$(date +%H:%M:%S)]  ${variant} BT done → $(basename ${target})"

    # Quick point-estimate check (no CI) for immediate feedback
    "${PY}" -c "
import pandas as pd, sys
sys.path.insert(0, '.')
from evaluation.metrics import compute_all_metrics
df = pd.read_csv('${target}').dropna(subset=['predicted_score','true_score'])
m = compute_all_metrics(df)
print(f'    pw_acc={m[\"pairwise_accuracy\"]:.4f}  '
      f'spearman={m[\"spearman\"]:.3f}  '
      f'R@1={m[\"recall_at_1\"]:.3f}')
"
    echo ""
done

echo "================================================="
echo "All 5 variants done. BT-only point estimates:"
echo ""
for variant in "${VARIANTS[@]}"; do
    target="${OOF_DIR}/bradley_terry_SNCTI_${variant}.csv"
    "${PY}" -c "
import pandas as pd, sys
sys.path.insert(0, '.')
from evaluation.metrics import compute_all_metrics
df = pd.read_csv('${target}').dropna(subset=['predicted_score','true_score'])
m = compute_all_metrics(df)
print(f'  ${variant:<22}  pw_acc={m[\"pairwise_accuracy\"]:.4f}  '
      f'rho={m[\"spearman\"]:.3f}  R@1={m[\"recall_at_1\"]:.3f}')
"
done
echo "================================================="
