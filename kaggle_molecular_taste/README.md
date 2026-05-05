# Kaggle Molecular Taste Classification

This folder generates the data for a public Kaggle competition where participants predict the taste class (sweet, bitter, sour, umami, undefined) of a small molecule from its SMILES string. The competition is a faithful redistribution of the **FartDB** corpus (Zimmermann et al. 2024), pinned to `github.com/fart-lab/fart` at commit `bde90e6562ce5d248e76af791fab29ffc9ae901b` and released under MIT.

The first half of this README is the **generation pipeline** for the maintainer. The second half is **ready-to-paste text for the Kaggle competition page**.

---

## Generation pipeline

Run from the project root:

```bash
python kaggle_molecular_taste/generate_kaggle_datasets.py
```

The script verifies the source FART splits against pinned SHA-256 hashes, writes the data CSVs to `dataset/`, and runs output sanity checks. Deterministic — running twice produces byte-identical outputs.

The `dataset/` folder is **committed to git**, so what's on disk is what gets uploaded to Kaggle without a regenerate step. Re-run the script only when upstream FartDB changes (the source-hash check enforces this).

### What's in `dataset/`

| File | Purpose |
|---|---|
| `train.csv` / `val.csv` / `test.csv` | 10,517 / 2,254 / 2,254 molecules; `taste` withheld in test |
| `sample_submission.csv` | All-`sweet` baseline |
| `solution.csv` | Test labels with `Usage = Public`; **never upload as competition data** |

Upload everything in `dataset/` to Kaggle **except** `solution.csv` (use Kaggle's separate solution-upload step for that). The MIT license is declared on the Kaggle competition page itself — no LICENSE file ships in the data zip.

### GNN baseline submission

A pre-generated GNN submission `predict/gnn_submission.csv` is
committed and ready to upload to the leaderboard. It maps the val-best
D-MPNN's predictions on the FART test split into the (id, taste)
Kaggle format. Regenerate via:

```bash
python kaggle_molecular_taste/predict/gnn_to_kaggle_submission.py \
    --out gnn_submission.csv
```

The script auto-locates the val-best run under
`molecular/results/grid/best/fart_test_eval/predictions.parquet` (or
the first run with a predictions.parquet if the `best` symlink is
missing) and joins by SMILES to the Kaggle test ids.

### Source

Local splits are pulled from `github.com/fart-lab/fart` at commit `bde90e6562ce5d248e76af791fab29ffc9ae901b` (MIT-licensed). This is the data the FART model was trained and evaluated on (Zimmermann et al. 2024). A separately-versioned Parquet distribution exists at <https://huggingface.co/datasets/FartLabs/FartDB> but may differ from the GitHub commit and is not what this redistribution mirrors. See [`../data/PROVENANCE.md`](../data/PROVENANCE.md) for hashes and per-class counts.

---

## Kaggle competition page text

Paste-ready files for each Kaggle tab. Each file is the literal text to paste — no instructional preamble.

| Kaggle tab / field | Source |
|---|---|
| **Title** | `Molecular Taste Classification` |
| **Subtitle** | `Predict whether a molecule tastes sweet, bitter, sour, umami, or undefined from its SMILES string.` |
| **Citation / Author / Sponsor** field (single-line, used by Kaggle's auto-BibTeX) | `TasteBench Organizers` |
| **Overview tab** (Description, Dataset Description, Evaluation, Submission File, Citation, Timeline, Prizes) | [`OVERVIEW.md`](OVERVIEW.md) |
| **Data tab** (Dataset Description) | [`OVERVIEW.md`](OVERVIEW.md) — paste the **Dataset Description** section |
| **Rules tab** | [`RULES.md`](RULES.md) |
| **Custom metric notebook** (Kaggle scoring) | [`kaggle_metric.py`](kaggle_metric.py) — paste into a Kaggle metric notebook and attach to the competition |

> **Important:** The Citation / Author / Sponsor field on Kaggle's competition setup is a single-line field that goes into the `author = {...}` of an auto-generated BibTeX. Paste **only** the sponsor name (`TasteBench Organizers`), not the full plain-text citation block. The full FartDB plus competition citations live in OVERVIEW.md's `## Citation` section, which gets pasted into the Description body and appears as visible text on the Overview tab.
