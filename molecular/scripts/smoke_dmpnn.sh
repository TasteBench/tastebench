#!/bin/bash
# 1-epoch smoke test on a 200-row subset of the real splits.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON="${PYTHON:-/opt/homebrew/Caskroom/miniforge/base/bin/python3}"

mkdir -p /tmp/taste_gnn_smoke

head -201 molecular/data/splits/fart_train.csv > /tmp/taste_gnn_smoke/train.csv
head -51  molecular/data/splits/fart_val.csv   > /tmp/taste_gnn_smoke/val.csv

cat > /tmp/taste_gnn_smoke/smoke.yaml <<'EOF'
run_name: smoke
seed: 42
data:
  train_csv: /tmp/taste_gnn_smoke/train.csv
  val_csv:   /tmp/taste_gnn_smoke/val.csv
  test_csv:  /tmp/taste_gnn_smoke/val.csv
  smiles_column: "Canonicalized SMILES"
  label_column:  "Canonicalized Taste"
  label_order: ["sweet", "bitter", "sour", "umami", "undefined"]
model:
  type: dmpnn
  hidden_dim: 64
  depth: 2
  dropout: 0.0
  n_classes: 5
training:
  batch_size: 32
  max_epochs: 1
  patience: 1
  init_lr: 1.0e-3
  device: auto
  class_weighting: inverse_frequency
wandb:
  mode: disabled
EOF

$PYTHON -m molecular.src.train.train_dmpnn \
    --config /tmp/taste_gnn_smoke/smoke.yaml \
    --output_dir /tmp/taste_gnn_smoke/run

echo "Smoke test OK. val_macro_f1:"
$PYTHON -c "import json; print(json.load(open('/tmp/taste_gnn_smoke/run/val_metrics.json'))['val_macro_f1'])"
