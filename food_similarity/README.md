# Food-similarity ranking

This area covers the food-level pairwise sensory ranking task: given two
plant-based products in the same category (e.g. two bacons), predict
which one was rated as more similar to the animal-based reference. The
task is evaluated leave-one-product-out (LOOCV) over 215 NECTAR
products in 24 categories.

All food-similarity tables are produced from this folder. The tables
cite four families of models — supervised (Bradley–Terry, hierarchical
BT, kernel RankSVM, ridge, LightGBM), unsupervised distance baselines
(cosine, L2 over multimodal embeddings), zero-shot LLMs (Gemini 3.1
Pro, Qwen 3.5 397B-A17B), and the NNLS ensemble.

Rendered `.tex` outputs land in
[`../paper/model_results_tables/`](../paper/model_results_tables/).
See the top-level [README](../README.md) for the artifact-to-renderer
mapping.

## Verifying numbers (no NECTAR required)

```bash
bash food_similarity/verify.sh
```

Re-renders every food-similarity table from the committed OOF
prediction CSVs in `results/oof_predictions/`. Runtime is dominated by
`render_table_ablation_features.py`, which sweeps BCa CIs over 105
(model, feature-subset) pairs using the full Python-loop bootstrap.
No data downloads, no model training. Pass `--check-diff` to also
`git diff` outputs against committed copies and assert byte-identical
reproduction.

## Full retrain (heavyweight; needs gated NECTAR data)

```bash
bash food_similarity/reproduce.sh
```

This regenerates `prepare_data.py`'s `product_features.pkl` from raw
NECTAR ingredient/nutrition CSVs, then retrains every supervised
baseline LOOCV, every NNLS ensemble (nested LOOCV), recomputes BCa
CIs, and re-renders the tables.

## Layout

```
food_similarity/
├── prepare_data.py             Build product_features.pkl from NECTAR
├── data/                       LOOCV loaders + cached features
├── models/                     BT, ridge, LightGBM, NNLS, … (one file per model)
├── train/                      Training entry points (LOOCV runners, NNLS meta-learner,
│                               LLM integration into ensemble)
├── evaluation/                 Metrics + BCa bootstrap (compute_bca_pw_acc, …)
├── zero_shot_baselines/        LLM and distance-predictor baseline pipelines
│   ├── configs/{llm,cosine_dist,l2_dist}/*.yaml
│   ├── lib/                    Shared embedding + preprocessor utilities
│   ├── run.py                  Single entry point
│   └── results/                LLM logs + distance-pred outputs (committed)
├── scripts/                    Render scripts for paper tables/figures
└── results/
    ├── oof_predictions/        Per-model LOOCV predictions (committed; canonical input
    │                           for all render scripts)
    └── llm_bootstrap_cis.csv   Precomputed BCa CIs for all 14 LLM modality combos
```

Rendered `.tex` tables land in
[`../paper/model_results_tables/`](../paper/model_results_tables/),
the single location consumed by `\input{...}` from the paper source.
