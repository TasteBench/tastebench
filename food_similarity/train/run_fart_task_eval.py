"""FART-task test accuracy for the two compound encoders.

Linear probe: sklearn LogisticRegression (multinomial, L2, C=1.0) trained on
FART train split embeddings, evaluated on FART test split.

Same probe for both encoders, so the comparison isolates the quality of the
representation. FART-task is 5-class taste classification (sweet, bitter,
sour, umami, undefined) over 2,254 held-out SMILES.

Reports test accuracy + BCa-bootstrap 95% CI.
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

TOP_DIR = Path(__file__).resolve().parents[2]
SHARED = TOP_DIR / "shared"
SPLITS = TOP_DIR / "molecular/data/splits"
SEED = 42

SMILES_COL = "Canonicalized SMILES"
LABEL_COL = "Canonicalized Taste"


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(SPLITS / name)[[SMILES_COL, LABEL_COL]].dropna()


def probe(train_df: pd.DataFrame, test_df: pd.DataFrame, embeddings: dict) -> tuple:
    """Linear probe: LogisticRegression(multinomial) on embedding → labels."""
    Xtr, ytr, Xte, yte = [], [], [], []
    for _, r in train_df.iterrows():
        emb = embeddings.get(r[SMILES_COL])
        if emb is None: continue
        Xtr.append(emb); ytr.append(r[LABEL_COL])
    for _, r in test_df.iterrows():
        emb = embeddings.get(r[SMILES_COL])
        if emb is None: continue
        Xte.append(emb); yte.append(r[LABEL_COL])
    Xtr = np.asarray(Xtr); ytr = np.asarray(ytr)
    Xte = np.asarray(Xte); yte = np.asarray(yte)
    print(f"  n_train={len(Xtr)}  n_test={len(Xte)}  dim={Xtr.shape[1]}")

    clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=1.0,
                             max_iter=2000, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = (pred == yte).mean()
    # BCa bootstrap over test SMILES for CI
    from scipy.stats import norm
    rng = np.random.default_rng(SEED)
    boot = np.empty(10_000)
    idx = np.arange(len(yte))
    for b in range(10_000):
        samp = rng.choice(idx, size=len(idx), replace=True)
        boot[b] = (pred[samp] == yte[samp]).mean()
    # Jackknife for BCa
    jack = np.empty(len(yte))
    for i in range(len(yte)):
        mask = np.ones(len(yte), dtype=bool); mask[i] = False
        jack[i] = (pred[mask] == yte[mask]).mean()
    z0 = norm.ppf((boot < acc).mean() + 0.5 * (boot == acc).mean())
    jm = jack.mean()
    num = ((jm - jack) ** 3).sum()
    den = 6.0 * (((jm - jack) ** 2).sum() ** 1.5)
    a = num / den if den != 0 else 0.0
    def q(z):
        p = norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
        return np.quantile(boot, np.clip(p, 0.0, 1.0))
    lo, hi = q(norm.ppf(0.025)), q(norm.ppf(0.975))
    return acc, lo, hi


def main():
    train = load_split("fart_train.csv")
    test = load_split("fart_test.csv")
    print(f"FART train: {len(train)}  test: {len(test)}")

    print("\nFART [CLS] (768-dim):")
    with open(SHARED / "data/caches/fart_compound_embeddings.pkl", "rb") as f:
        fart_emb = pickle.load(f)
    t0 = time.time()
    fart_acc, fart_lo, fart_hi = probe(train, test, fart_emb)
    print(f"  acc={fart_acc:.4f} [{fart_lo:.4f}, {fart_hi:.4f}]  ({time.time()-t0:.0f}s)")

    print("\ntaste_gnn (300-dim):")
    with open(SHARED / "data/caches/taste_gnn_best_compound_embeddings.pkl", "rb") as f:
        tg_emb = pickle.load(f)
    t0 = time.time()
    tg_acc, tg_lo, tg_hi = probe(train, test, tg_emb)
    print(f"  acc={tg_acc:.4f} [{tg_lo:.4f}, {tg_hi:.4f}]  ({time.time()-t0:.0f}s)")

    # Save CSV
    pd.DataFrame([
        {"embedding": "FART [CLS]", "acc": fart_acc, "lo": fart_lo, "hi": fart_hi,
         "n_train": len(train), "n_test": len(test), "dim": 768},
        {"embedding": "taste_gnn",   "acc": tg_acc,   "lo": tg_lo,   "hi": tg_hi,
         "n_train": len(train), "n_test": len(test), "dim": 300},
    ]).to_csv(TOP_DIR / "food_similarity/results/fart_task_linear_probe.csv",
              index=False, float_format="%.6f")
    print(f"\nWrote food_similarity/results/fart_task_linear_probe.csv")


if __name__ == "__main__":
    main()
