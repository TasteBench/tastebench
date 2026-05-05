# Molecular taste classification

This area covers the molecular taste classification task: given a
SMILES string, predict its taste label among
{sweet, bitter, sour, umami, undefined}. The task uses the FART
dataset (Zimmermann et al. 2024), with the original train/val/test
splits (10,517 / 2,254 / 2,254 molecules). The penultimate-layer
embeddings of our val-best D-MPNN checkpoint are also evaluated as a
compound encoder for the food-similarity task in
`food_similarity/results/oof_predictions/*_tastegnn.csv`.

## Outputs

Rendered tables land in
[`../paper/molecular_prediction/`](../paper/molecular_prediction/);
CSV sidecars are written alongside under `results/tables_csv/`. See
the top-level [README](../README.md) for the artifact-to-renderer
mapping.

## Verifying numbers

```bash
bash molecular/verify.sh
```

Re-renders every molecular-area table from the committed parquet
predictions in `results/grid/<best>/fart_test_eval/predictions.parquet`
and `results/fart_augmented_test/predictions.parquet`. No model
training, no external downloads.

## Full retrain (GNN training)

```bash
bash scripts/submit_grid.sh dmpnn_grid
python -m molecular.src.train.select_best_and_evaluate
bash scripts/aggregate_grid.sh
```

The D-MPNN trains on CPU; no GPU is required.

Then run `verify.sh` to re-render tables.

## Layout

```
molecular/
├── data/splits/                FART train/val/test SMILES splits
├── src/
│   ├── data/                   FartDataset, scaffold_split.py, download.py
│   ├── models/                 D-MPNN architecture
│   ├── train/                  train_dmpnn.py, grid_search.py, select_best_and_evaluate.py
│   ├── eval/                   evaluate.py, metrics.py (per-class + macro CIs)
│   └── embed/                  predict_embeddings.py, generate_cache.py
├── configs/                    dmpnn_{base,grid,scaffold}.yaml
├── scripts/                    Render scripts (table_*) + training drivers (submit_grid*.sh)
├── tests/
└── results/
    ├── grid/                   D-MPNN grid-search outputs (one folder per run)
    │   └── best -> run_*/      Symlink to val-best run
    ├── fart_augmented_test/    FART checkpoint test eval (predictions.parquet)
    └── tables_csv/             CSV sidecars (one per paper table); .tex versions land in ../paper/molecular_prediction/
```
