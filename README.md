# TasteBench (NeurIPS 2026)

Code, splits, and rendering scripts accompanying the TasteBench paper
submission to the NeurIPS 2026 Evaluations and Datasets Track.

> All commands below are run from inside this directory (the
> `tastebench/` folder after unzipping). `cd` into it before running
> anything.

## Reviewer reproduction guide

Every numerical claim is reproducible at four tiers, each requiring
strictly more than the last. Tier 0 has zero prerequisites and verifies
all rendered tables and figures from committed artifacts; tiers 1–3 add
the dependencies needed to regenerate those artifacts from scratch.

**This submission ships all model artifacts needed for inference**:
the 12 trained GNN checkpoints (one per hyperparameter-grid configuration)
at `molecular/results/grid/run_*/ckpt.pt`, all leave-one-out prediction
CSVs from the food-similarity baselines (including the paid LLM
predictions) at `food_similarity/results/oof_predictions/`, the
human-baseline analysis artifacts at `human_baseline/results/`, and the
rendered LaTeX tables and figures at `paper/`. Tier 0 verification is
self-contained — no downloads, no GPU, no API calls required.

### Tier 0 — Verify rendered numbers (no downloads, no API spend)

```bash
bash verify_paper.sh
```

Re-renders every table and the human-baseline figure from the committed
OOF prediction CSVs in `food_similarity/results/oof_predictions/`, the
parquet predictions in `molecular/results/`, and the human-baseline
analysis artifacts in `human_baseline/results/`. Runtime is dominated by
the BCa bootstrap (10,000 resamples), with `render_table_ablation_features.py`
as the long pole. Pass `--check-diff` to also `git diff` outputs against
committed copies and assert byte-identical reproduction.

### Tier 1 — Retrain non-LLM baselines from raw NECTAR (needs NECTAR)

Requires NECTAR access. NeurIPS reviewers can access the data via the
private access link provided in the paper; other researchers go
through the Google Form linked in the Datasets section of the paper. The bundle's internal layout already mirrors the
project-root-relative paths the pipeline expects:

```bash
# 1. Drop the gated bundle into the right place:
bash data/unpack_nectar_bundle.sh <path-to-downloaded-bundle>

# 2. Fetch public datasets and run the pipeline:
bash data/download_public.sh                       # FoodAtlas v4.0 + FooDB
cd food_similarity && bash reproduce.sh
```

See [`data/GATED.md`](data/GATED.md) for the full file-by-file
breakdown, NDA terms, and the four post-unpack paths.

Regenerates `product_features.pkl`, retrains every supervised LOOCV
baseline (Bradley–Terry, Hierarchical BT, Ridge, Kernel RankSVM,
LightGBM), refits the nested NNLS meta-learner, recomputes BCa CIs, and
re-renders the food-similarity tables. Distance baselines (cosine, L2
MMRF) are included in this run.

The human-baseline analysis is a separate script that runs on the raw
NECTAR ratings:

```bash
cd human_baseline
python human_panelist_baseline.py
python plot_human_baseline.py
```

### Tier 2 — Retrain the GNN from scratch

The 12-configuration hyperparameter grid:

```bash
bash molecular/scripts/submit_grid.sh dmpnn_grid
python -m molecular.src.train.select_best_and_evaluate
bash molecular/scripts/aggregate_grid.sh
```

The D-MPNN trains on CPU; no GPU is required.

**All 12 trained GNN checkpoints are shipped in this submission** at
`molecular/results/grid/run_*/ckpt.pt` (~1.2 MB each, ~14.4 MB total),
together with the per-run `config.yaml` and `val_metrics.json`. This
tier is only needed if you want to verify training from the raw FART
splits; inference and embedding extraction with the committed
checkpoints run on CPU.

### Tier 3 — Regenerate LLM baselines (needs OpenRouter API; ~$300)

Gemini 3.1 Pro and Qwen 3.5 397B-A17B predictions are generated via the
OpenRouter API. The committed OOFs in
`food_similarity/results/oof_predictions/llm_*.csv` are the source of
the LLM numbers in every table. Regenerating them from scratch cost
approximately $300 in OpenRouter API spend across both models and the
LLM modality ablation grid (current pricing as of submission).

```bash
export OPENROUTER_API_KEY=...
cd food_similarity/zero_shot_baselines
python run.py --config configs/llm/gemini_3_1_pro.yaml
python run.py --config configs/llm/qwen3_5_397b.yaml
```

### Summary

| Component | Tier | Needs NECTAR | Notes |
|---|---|---|---|
| Re-render every table & figure | 0 | no | runs from committed OOFs |
| Distance baselines (MMRF cos / L2) | 1 | yes | |
| Supervised pairwise baselines (BT, HBT, Ridge, KSVM, LGBM) | 1 | yes | |
| NNLS meta-learner ensembles | 1 | yes | |
| Human-panelist baseline analysis | 1 | yes | |
| GNN inference / embedding extraction | 1 | yes | uses shipped ckpts |
| GNN training from scratch | 2 | no | runs on CPU; no GPU required |
| LLM baselines (Gemini, Qwen) regeneration | 3 | yes | ~$300 OpenRouter API spend |

## Layout

The repo is split into five areas, one per paper section, plus shared
infrastructure. Each area has a top-level `README.md` describing its
inputs, outputs, and verify path.

```
tastebench/
├── verify_paper.sh           Top-level umbrella: re-renders every table & figure
│                              from committed OOFs (no NECTAR, no training)
├── make_submission.sh        Builds the supplementary submission zip
│
├── food_similarity/          Food-level pairwise ranking (results + ablations)
│   ├── README.md
│   ├── verify.sh             Re-render this area's tables/figures only
│   ├── reproduce.sh          End-to-end retrain (heavyweight; needs NECTAR)
│   ├── prepare_data.py       Build product_features.pkl from raw NECTAR
│   ├── data/                 LOOCV loaders + cached features
│   ├── models/               BT, hierarchical-BT, ridge, kernel-RankSVM, LightGBM, NNLS
│   ├── train/                LOOCV runners + nested NNLS meta-learner + LLM integration
│   ├── evaluation/           Metrics + BCa bootstrap
│   ├── zero_shot_baselines/  LLM (Gemini, Qwen) + distance (cosine, L2) baselines
│   ├── scripts/              Render scripts for this area's tables/figures
│   └── results/              oof_predictions/, tables/, figures/, metrics/
│
├── molecular/                Molecular taste classification (results + ablations)
│   ├── README.md
│   ├── verify.sh
│   ├── data/splits/          FART train/val/test SMILES splits
│   ├── src/                  D-MPNN data, models, train, eval, embed
│   ├── configs/              dmpnn_{base,grid,scaffold}.yaml
│   ├── scripts/              Render scripts + training shell drivers
│   └── results/              grid/, fart_augmented_test/, tables/{tex,csv}/
│
├── kaggle_tastebench/        Public Kaggle competition for the food-similarity task
│   ├── README.md
│   ├── generate_data/        Obfuscated NECTAR + Taste Like construction; competition CSVs
│   └── predict/              LLM → Kaggle-format submission converter (added in phase D)
│
├── kaggle_molecular_taste/   Public Kaggle competition for the molecular task
│   ├── README.md
│   ├── generate_kaggle_datasets.py
│   └── dataset/              train.csv, val.csv, test.csv, sample_submission.csv (solution.csv gated)
│
├── human_baseline/           Untrained human-panelist baseline on the food-similarity task
│   ├── README.md
│   ├── verify.sh             Re-render figures from committed CSVs
│   ├── human_panelist_baseline.py   → results/{summary,split_half}.json, group_size_curve.csv,
│   │                                  human_baseline_table.tex (needs NECTAR)
│   ├── plot_human_baseline.py       → paper/human_baseline/group_size_curve.pdf
│   └── results/
│
├── paper/          Rendered table & figure outputs consumed by the paper
│   ├── human_baseline/       human_baseline_table.tex, group_size_curve.pdf
│   ├── model_results_tables/ table_results, table_per_category, table_ablation_*, …
│   └── molecular_prediction/ table_molecular_prediction, table_gnn_*, …
│
├── shared/                   Cross-area code + cached embeddings
│   ├── compound_mapping/     FoodAtlas → SMILES + FooDB concentrations
│   ├── scripts/              prepare_smiles_cache.py, prepare_fart_embeddings.py, …
│   └── data/                 Manually-curated labels + cached embeddings (gitignored, hosted)
│
└── data/                     Inputs (see "Data: public vs. gated" below)
    ├── GATED.md              Gating policy + Google Form access flow
    ├── download_public.sh    Fetches FoodAtlas + FooDB
    ├── taste_like/           Taste Like CSVs (gated — see data/GATED.md)
    ├── food_atlas/v4.0/      FoodAtlas v4.0 (82 MB — download)
    ├── foodb_2020_04_07_csv/ FooDB CSV release (953 MB — download)
    ├── consolidated_datasets/  NECTAR CSVs (gated — request via Form)
    └── product_images/       NECTAR product photos (gated — request via Form)
```

## Artifact-to-renderer mapping

| Artifact | Producer |
|---|---|
| Combined food-similarity results (Pw.Acc + ρ + R@k) | `food_similarity/scripts/render_table_results.py` |
| Molecular taste classification | `molecular/scripts/render_table_molecular_prediction.py` |
| GNN per-model transfer | `molecular/scripts/render_table_gnn_per_model.py` |
| Per-category model comparison | `food_similarity/scripts/render_table_per_category.py` |
| Per-category NNLS metrics | `food_similarity/scripts/render_table_per_category_nnls.py` |
| Per-model NNLS swap | `food_similarity/scripts/render_table_per_model_nnls.py` |
| Feature-subset ablation | `food_similarity/scripts/render_table_ablation_features.py` |
| LLM input ablation | `food_similarity/scripts/render_table_ablation_llm.py` |
| GNN hyperparameter grid | `molecular/scripts/render_table_gnn_grid.py` |
| Molecular per-class breakdown | `molecular/scripts/render_table_molecular_per_class.py` |
| Human-panelist baseline | `human_baseline/human_panelist_baseline.py` |
| Group-size curve figure | `human_baseline/plot_human_baseline.py` |

## Data: public vs. gated

NECTAR ratings, ingredient/nutrition CSVs, product images, and the
Taste Like CPG product directory are not redistributed with this
artifact; see `data/GATED.md` for the access procedure.

**Public, fetched by `data/download_public.sh`:**

| Dataset | Source | Target path |
|---|---|---|
| FoodAtlas v4.0 | https://www.foodatlas.ai/food-composition-downloads | `data/food_atlas/v4.0/` |
| FooDB 2020-04-07 | https://foodb.ca/downloads | `data/foodb_2020_04_07_csv/` |

**Public, vendored in repo:**

| Dataset | Where |
|---|---|
| FART train/val/test splits | `molecular/data/splits/` (~1 MB) |
| Kaggle competition obfuscated dataset | `kaggle_tastebench/generate_data/dataset/` |

**Gated — see [`data/GATED.md`](data/GATED.md):** NECTAR sensory + ingredient
+ nutrition CSVs, NECTAR product images, the derivative label files
(`nectar_product_labels.csv`, `product_labels_manually_cleaned.csv`), and the
Taste Like CPG product directory CSVs. After receiving access (private Kaggle
Dataset for reviewers, Google Form for other researchers), you'll receive a
bundle that unpacks into the existing `data/` paths and the pipeline
reproduces all results byte-identically.

Without the gated bundle, the Tier 0 path (`bash verify_paper.sh`)
still re-renders every table from the committed OOF predictions and
parquet predictions in this repository. See `data/GATED.md`.

## Caches

The `shared/data/caches/` folder holds derived embeddings consumed by
`food_similarity/prepare_data.py` and `molecular/`. They are large and excluded
from git via `.gitignore`. To regenerate (or download):

```bash
# Regenerate from scratch (slow; needs FART weights + GNN training):
python shared/scripts/prepare_caches/prepare_smiles_cache.py
python shared/scripts/prepare_caches/resolve_chebi_smiles.py
python shared/scripts/prepare_caches/prepare_fart_embeddings.py
python -m molecular.src.embed.generate_cache              # molecular GNN embeddings
python shared/scripts/generate_sensory_descriptions.py    # text augmentation
```

A SHA-256 manifest of every cache file is at
`food_similarity/results/input_cache_sha256.txt`. `food_similarity/reproduce.sh`
verifies caches against this manifest before running.

## Pre-computed predictions

`food_similarity/results/oof_predictions/` ships every leave-one-out prediction
CSV the paper depends on, including the `llm_*` predictions sourced from
`food_similarity/zero_shot_baselines/results/llm/` (regenerating these costs real $$ via
the OpenRouter API and so they are committed deliberately).

## Anonymization

The public Kaggle competition uses renumbered product codes; the
back-mapping (`kaggle_tastebench/generate_data/dataset/product_code_map.csv`)
is gitignored and not redistributed. See `data/GATED.md` and the paper
for the privacy framing.
