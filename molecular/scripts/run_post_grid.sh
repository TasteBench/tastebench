#!/usr/bin/env bash
# Run the post-grid pipeline: select val-best -> evaluate FART test ->
# regenerate GNN embedding cache -> run NECTAR encoder transfer ->
# render the 2 surviving tables -> tectonic compile-check.
#
# Assumes the 12-config D-MPNN grid has finished and that
# molecular/results/grid/grid_summary.csv exists.
#
# Usage:
#   bash molecular/scripts/run_post_grid.sh
set -euo pipefail

cd "$(dirname "$0")/../../.."
echo "[$(date)] cwd=$(pwd)"

if [[ ! -f molecular/results/grid/grid_summary.csv ]]; then
    echo "ERROR: grid_summary.csv not found -- the grid hasn't finished yet."
    exit 1
fi

echo
echo "=== 1/3: select val-best, evaluate FART test, build cache, NECTAR transfer"
python -m molecular.src.train.select_best_and_evaluate \
    --results_dir molecular/results/grid

echo
echo "=== 2/3: render the 3 molecular-prediction tables (test-set bootstrap ~1 min)"
for r in render_table_molecular_prediction \
         render_table_molecular_per_class \
         render_table_gnn_per_model ; do
    echo
    echo "  >>> $r"
    python "food_similarity/scripts/${r}.py"
done

echo
echo "=== 3/3: tectonic compile-check"
python food_similarity/scripts/compile_check_tables.py

echo
echo "[$(date)] done"
echo "Tables:        molecular/results/tables/tex/*.tex"
echo "CSV sidecars:  molecular/results/tables/tex/*.csv"
echo "Grid summary:  molecular/results/grid/grid_summary.csv"
