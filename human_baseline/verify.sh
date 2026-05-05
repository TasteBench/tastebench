#!/usr/bin/env bash
# Re-render the human-baseline figure from the committed analysis
# artifacts in results/ (group_size_curve.csv, split_half_reliability.json,
# summary.json). No NECTAR data required.
#
# Outputs (written to paper/human_baseline/):
#   group_size_curve.pdf
#
# The accompanying .tex table (paper/human_baseline/human_baseline_table.tex)
# is committed as a static artifact; regenerating it requires running
# human_panelist_baseline.py with the gated NECTAR data.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python}"

echo ""
echo "[$(date +%H:%M:%S)] human_baseline: plot_human_baseline.py"
( cd "${HERE}" && "${PY}" plot_human_baseline.py ) > /tmp/verify_human_baseline.log 2>&1 \
  || { echo "FAILED — last 30 lines of log:"; tail -30 /tmp/verify_human_baseline.log; exit 1; }
tail -3 /tmp/verify_human_baseline.log | sed 's/^/  /'
