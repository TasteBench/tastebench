# Gated data

Three inputs are not redistributable in the public release of this submission:

1. **NECTAR sensory + ingredient + nutrition CSVs** — `data/consolidated_datasets/nectar_consolidated_*.csv`, plus the derivative label files `shared/data/nectar_product_labels.csv` and `food_similarity/zero_shot_baselines/data/product_labels_manually_cleaned.csv`. These contain participating brand identities and are NDA-protected by NECTAR.
2. **NECTAR product images** — `data/product_images/cropped/`. The packaging-cropped photos are NECTAR-owned and fall under the same NDA.
3. **Taste Like CPG product directory** — `data/taste_like/*.csv`. NECTAR-acquired (November 2024). Public availability of this data via the original taste-like.com website is uncertain post-acquisition; we ship it gated alongside NECTAR's own data so that the cross-reference deanonymization attack against the public Kaggle competition (matching public Taste Like features against the obfuscated bundle to identify NECTAR products by elimination) is not made trivial by the supplementary repo.

These files are deliberately excluded from version control (see `.gitignore`) and are not part of the submitted artifact.

## What is *not* gated

- The **TasteBench Kaggle competition** dataset (`tastebench-challenge-2026`), which is a deliberately obfuscated, zero-shot release: NECTAR products are interleaved with samples from the Taste Like CPG product directory, all product codes are renumbered, and labels are withheld. See `kaggle_tastebench/generate_data/generate_kaggle_datasets.ipynb` for the construction.
- **All rendered tables and analysis artifacts**: out-of-fold prediction CSVs (`food_similarity/results/oof_predictions/`), human-baseline tables and figures (`human_baseline/results/`), CI tables, and rendered LaTeX (`paper/`). These are sufficient to verify every numerical claim and re-render every table and figure without access to the gated data.
- **Public datasets** the pipeline depends on: FART splits (vendored), FooDB (fetched via `data/download_public.sh`).

## How to request access

Two paths are available:

**1. NeurIPS 2026 reviewers — private Kaggle Dataset.** The four NECTAR CSVs plus the cropped product images are hosted as a private Kaggle Dataset; the private access link is provided in the paper for reviewer verification. The Croissant metadata for this bundle ships at `croissant/nectar.json`.

**2. All other researchers — Google Form.** Request access via the Google Form linked in the paper's Datasets section. Approval grants the same bundle.

## Downloading the bundle

1. Open the Kaggle Dataset URL in your browser. (Reviewers: the URL is provided in the paper. Other researchers: complete the Google Form first to receive the URL.)
2. Click **Download** on the Kaggle Dataset page. Depending on your browser and Kaggle's delivery path, you will end up with one of:
   - A folder named `archive/` directly (some setups auto-extract on download), or
   - A zip file `archive.zip`, which you'll need to extract: `unzip archive.zip -d archive`.

   In either case, the resulting `archive/` folder contains four NECTAR CSVs plus image and Taste Like contents. The image and Taste Like contents may appear as `images.zip` and `taste_like.zip` (Kaggle's native bundle layout) or as already-extracted `images/` and `taste_like/` directories — the unpack script in the next step accepts both forms.

## Unpacking into the pipeline

The Kaggle Dataset uses the flat layout convention: four CSVs at the top level plus `images.zip` (cropped product photos) and `taste_like.zip` (Taste Like CPG directory). The recommended path is the one-command helper script.

From the project root (the `tastebench/` directory after unzipping the supplementary):

```bash
bash data/unpack_nectar_bundle.sh ~/Downloads/archive
```

The script copies each CSV to its target subdirectory, places the image and Taste Like contents (extracting `images.zip`/`taste_like.zip` or copying the already-extracted directories), and verifies every expected file is present. If you would rather copy by hand (e.g., to debug a failure), the file-to-folder mapping is:

| File in the downloaded bundle | Target folder (relative to project root) |
|---|---|
| `nectar_consolidated_ingredients_nutrition.csv` | `data/consolidated_datasets/` |
| `nectar_consolidated_sensory_rating.csv` | `data/consolidated_datasets/` |
| `nectar_product_labels.csv` | `shared/data/` |
| `product_labels_manually_cleaned.csv` | `food_similarity/zero_shot_baselines/data/` |
| `images.zip` (extract contents) | `data/product_images/` (creates `cropped/<year>/<category>/<product_code>/<view>.jpg`) |
| `taste_like.zip` (extract contents) | `data/taste_like/` (5 CSVs from the Taste Like CPG product directory) |

After unpacking, the artifacts sit at:

```
data/consolidated_datasets/nectar_consolidated_ingredients_nutrition.csv
data/consolidated_datasets/nectar_consolidated_sensory_rating.csv
data/product_images/cropped/<year>/<category>/<product_code>/<view>.jpg
shared/data/nectar_product_labels.csv
food_similarity/zero_shot_baselines/data/product_labels_manually_cleaned.csv
data/taste_like/*.csv
```

These are the paths `food_similarity/prepare_data.py`, `human_baseline/human_panelist_baseline.py`, and the LLM zero-shot baselines read from. With the bundle in place, the Tier 1 retraining path (`bash food_similarity/reproduce.sh`) reproduces all rendered artifacts byte-identically (subject to BCa CI bootstrap seed = 42 and joblib worker scheduling).

The Tier 0 verification path (`bash verify_paper.sh`) does NOT require this bundle — it works from the committed out-of-fold prediction CSVs.

## Reviewer guidance

Reviewers can verify all rendered numbers from the committed artifacts alone:

- `food_similarity/results/oof_predictions/` contains every leave-one-out prediction the food-similarity tables depend on (including the LLM predictions, which cost real money to regenerate via the OpenRouter API).
- `food_similarity/scripts/render_*.py` and `molecular/scripts/render_*.py` re-render every table from those OOFs and the molecular parquet predictions.
- `human_baseline/results/` contains the human-baseline summary, group-size curve data, per-category comparisons, and split-half reliability JSON used to render the human-baseline table and the group-size curve figure.
- `molecular/results/` contains the GNN grid-search outputs (12 trained checkpoints + per-run metrics) used for the molecular tables.

The entire numerical content of every rendered table is committed and rerunnable from public inputs. The gated NECTAR raw ratings are needed only to regenerate the OOFs from scratch.
