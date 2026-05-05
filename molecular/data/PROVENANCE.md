# Data Provenance

## FART split CSVs

Source: https://github.com/fart-lab/fart
Path in repo: `dataset/splits/fart_{train,val,test}.csv`

Pinned git SHA: `bde90e6562ce5d248e76af791fab29ffc9ae901b`

Expected file row counts (including 1-row header):
- fart_train.csv: 10,518 lines (10,517 rows)
- fart_val.csv:   2,255 lines (2,254 rows)
- fart_test.csv:  2,255 lines (2,254 rows)

Expected SHA-256 hashes (populated after first successful download):
- fart_train.csv: 35020569fb47d10d8c981d65c712cb73dc7b23512911d35d2b161665584248a3
- fart_val.csv:   b8ee024f5b73324468341475e713d25d340e42aeffcb3a8b64487d8184c2c657
- fart_test.csv:  74096827602864a8aff09234ae68a0637b13f3749b57d662b45fd7863185f4f1

Expected class distribution per split:

| Class     | train  | val   | test  |
|-----------|-------:|------:|------:|
| sweet     |  6,612 | 1,499 | 1,473 |
| bitter    |  1,241 |   221 |   233 |
| sour      |  1,132 |   237 |   238 |
| undefined |  1,479 |   294 |   304 |
| umami     |     53 |     3 |     6 |

## Kaggle redistribution

These splits are also redistributed via a public Kaggle competition (Molecular Taste Classification) under FartDB's MIT license. The Kaggle-formatted CSVs are produced by `kaggle_molecular_taste/generate_kaggle_datasets.py`, which:

- Reassigns ids to a contiguous globally-unique range 0..15024 (because the upstream FART CSVs use inconsistent id schemes that collide between splits).
- Drops the `Standardized SMILES` and `Original Labels` columns. The latter would leak the target via trivial keyword matching (~0.88 test accuracy).
- Withholds `taste` from the test split.

The canonical source for the data we redistribute is `github.com/fart-lab/fart` at commit `bde90e6562ce5d248e76af791fab29ffc9ae901b`. A related distribution exists at <https://huggingface.co/datasets/FartLabs/FartDB> in Parquet format with its own version history; that distribution may have evolved since the pinned GitHub commit and is **not** what this redistribution mirrors.
