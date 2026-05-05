#!/usr/bin/env bash
# Re-render every molecular-area table from the committed parquet
# predictions in results/grid/ and results/fart_augmented_test/.
# No training, no external data.
#
# Outputs (all written to paper/molecular_prediction/):
#   table_molecular_prediction.tex
#   table_gnn_per_model.tex
#   table_molecular_per_class.tex
#   table_gnn_grid.tex
#
# CSV sidecars (parallel data versions of each table) land in
# molecular/results/tables_csv/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python}"

run() {
  echo ""
  echo "[$(date +%H:%M:%S)] molecular: $*"
  ( cd "${HERE}" && "${PY}" "$@" ) > /tmp/verify_molecular.log 2>&1 \
    || { echo "FAILED — last 30 lines of log:"; tail -30 /tmp/verify_molecular.log; exit 1; }
  tail -3 /tmp/verify_molecular.log | sed 's/^/  /'
}

run scripts/render_table_molecular_prediction.py
run scripts/render_table_molecular_per_class.py
run scripts/render_table_gnn_grid.py
run scripts/render_table_gnn_per_model.py
