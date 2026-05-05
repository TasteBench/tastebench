## Overview

Predict the taste of a small molecule from its chemical structure. Each molecule is one of: **sweet, bitter, sour, umami,** or **undefined.** This is the FartDB benchmark (Zimmermann et al. 2024), released on Kaggle as a fully public, transparent leaderboard.

## Description

You get **15,025 small molecules** as SMILES strings, each labeled with one of five taste classes. The split is:

- **Train:** 10,517 labeled molecules
- **Validation:** 2,254 labeled molecules
- **Test:** 2,254 unlabeled molecules — predict these

The dataset is **heavily imbalanced.** Sweet molecules outnumber umami by 100-to-1: there are only 53 umami molecules in train and 6 in test. A model that always predicts `sweet` already gets ~65% accuracy. Beating that means correctly identifying the minority classes — especially umami.

### A fully public benchmark

This is **not** a hidden-label challenge. The test labels are openly available in the upstream FartDB GitHub repository (<https://github.com/fart-lab/fart>) at the same commit this competition uses. That's intentional — the goal is direct comparison with published results.

The Kaggle leaderboard is **100% Public.** Every test molecule counts toward your visible score; there is no private-leaderboard reshuffle. See the Rules tab for the honor-system policy on test-label lookups.

## Dataset Description

### Files

- **train.csv** — 10,517 labeled training molecules
- **val.csv** — 2,254 labeled validation molecules. Recommended for model selection or early stopping.
- **test.csv** — 2,254 unlabeled test molecules. Predict the `taste` of each.
- **sample_submission.csv** — a submission file in the correct format. The baseline predicts `sweet` for every row.

### Columns

- `id` — integer molecule identifier, contiguous 0..15024 and globally unique across all three files. Use this as the join key in your submission.
- `canonicalized_smiles` — the molecule as an RDKit-canonicalized SMILES string. This is the model input.
- `taste` — the class label, one of `sweet`, `bitter`, `sour`, `umami`, or `undefined`. **Withheld from `test.csv`** — predict this.

### Class distribution

| Class      | Train  | Val   | Test  | Total |
|------------|-------:|------:|------:|------:|
| sweet      |  6,612 | 1,499 | 1,473 | 9,584 |
| undefined  |  1,479 |   294 |   304 | 2,077 |
| bitter     |  1,241 |   221 |   233 | 1,695 |
| sour       |  1,132 |   237 |   238 | 1,607 |
| umami      |     53 |     3 |     6 |    62 |
| **Total**  | 10,517 | 2,254 | 2,254 | 15,025 |

The test split is identical (same molecules, same row order) to the FartDB test split.

## Evaluation

**Primary metric: macro-F1** — the average per-class F1 across the five taste classes (each class weighted equally).

Plain accuracy is not useful on this dataset: predicting `sweet` for everything already scores 0.65. Macro-F1 forces the model to do well on minority classes too.

We also report **balanced accuracy** (average per-class recall) for transparency, but the Kaggle leaderboard ranks by macro-F1 only.

*Note on umami:* there are only 6 umami molecules in the test set, so umami F1 is noisy — one mistake swings it by ~0.15. Macro-F1 averages across all five classes, which dampens this but doesn't eliminate it.

### Submission File

Submit a CSV with one prediction per test molecule. Two columns: `id` and `taste`.

```
id,taste
12771,sweet
12772,bitter
12773,undefined
...
```

Requirements:

- Header row exactly `id,taste`
- Exactly 2,254 prediction rows (plus the header), one per test molecule
- Every test `id` appears exactly once (any order)
- `taste` is one of `sweet`, `bitter`, `sour`, `umami`, `undefined` (case-sensitive)

Submissions that don't meet these are rejected before scoring.

## Acknowledgments

Thanks to the FartDB authors — Yoel Zimmermann, Leif Sieben, Henrik Seng, Philipp Pestlin, and Franz Görlich — for releasing the dataset under MIT.

## Citation

Cite the **FartDB paper** in any work based on this competition's data. This is required by the upstream MIT license and credits the team who built the dataset:

> Zimmermann, Y., Sieben, L., Seng, H., Pestlin, P., & Görlich, F. (2024). A Chemical Language Model for Molecular Taste Prediction. https://doi.org/10.26434/chemrxiv-2024-d6n15-v2, Preprint. ChemRxiv.

If you also want to cite the competition itself (e.g., to reference your leaderboard standing), use:

> TasteBench Organizers. Molecular Taste Classification. https://kaggle.com/competitions/molecular-taste-classification-2026, Unpublished. Kaggle.

**Source:** <https://github.com/fart-lab/fart> at commit `bde90e6562ce5d248e76af791fab29ffc9ae901b`. Released under the MIT License.

## Timeline

See the **Timeline** section above for start and submission deadlines.

## Prizes

No cash prizes. This is an open academic benchmark — recognition is the public leaderboard standing and the option to publish a write-up.
