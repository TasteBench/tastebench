"""NECTAR pw-acc for table_taste_gnn.tex: L2 Bradley-Terry + Gemini nested NNLS.

Three rows:
  FART [CLS]  — C = fart_compound_embeddings (default)
  taste_gnn   — C = taste_gnn_best_compound_embeddings (300-dim D-MPNN output)
  no C        — C block dropped (SNTI features)

Same methodology as the archived ensemble row (L2 BT + Gemini NNLS via
nested LOOCV), with only the compound encoder swapped between rows.

FART row reuses the archived nested_bt_gemini_nnls.csv (0.6818) —
this is exactly the "FART as C" config computed once and cached.
"""
import pickle
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import scipy.linalg
from scipy.optimize import nnls
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings("ignore")

TOP_DIR = Path(__file__).resolve().parents[2]
SUP = TOP_DIR / "food_similarity"
SHARED = TOP_DIR / "shared"
sys.path.insert(0, str(SUP))
sys.path.insert(0, str(SHARED))

from data.loocv import load_product_features
from evaluation.metrics import compute_all_metrics
from prepare_data import _extract_compound_generic, load_nectar_products
from train.paper_table import impute_missing_images_inplace

SEED = 42


def _rank_per_cat(x, cats_arr):
    out = np.full_like(x, np.nan, dtype=np.float64)
    for c in np.unique(cats_arr):
        m = cats_arr == c
        if m.sum() > 0:
            valid = ~np.isnan(x[m])
            vec = np.full(m.sum(), np.nan)
            if valid.sum() > 0:
                vec[valid] = rankdata(x[m][valid]) / valid.sum()
            out[m] = vec
    return out


def nested_l2_bt_nnls(pf: dict, letters: str, compound_dim: int | None = None) -> tuple:
    """Nested leak-free L2-BT + Gemini NNLS for arbitrary S-prefixed subsets.

    compound_dim defaults to autodetection from the first non-None compound
    vector in pf (only used as the fallback zero-vector size for products that
    are missing compound features).
    """
    analog_keys = sorted(k for k, p in pf.items() if p.get("is_analog", True))
    k_idx = {k: i for i, k in enumerate(analog_keys)}
    N = len(analog_keys)

    if compound_dim is None:
        for k in analog_keys:
            v = pf[k].get("compound")
            if v is not None:
                compound_dim = int(v.shape[0]); break
        if compound_dim is None:
            compound_dim = 300

    BLOCK_SPECS = {"S": ("category_subset", 4), "N": ("nutrition", 6),
                   "C": ("compound", compound_dim), "T": ("text", 1024),
                   "I": ("image", 1024)}

    RAW = {}
    for l in letters:
        key_name, dim = BLOCK_SPECS[l]
        rows = []
        for k in analog_keys:
            v = pf[k].get(key_name)
            rows.append(v.astype(np.float32) if v is not None else np.zeros(dim, dtype=np.float32))
        RAW[l] = np.stack(rows)

    PAIRS = pd.read_csv(SUP / "data/pairs.csv")
    pair_records = []
    for _, r in PAIRS.iterrows():
        k1 = (r["category"], int(r["product_code_1"]))
        k2 = (r["category"], int(r["product_code_2"]))
        higher = (r["category"], int(r["higher_rated_product"]))
        if k1 not in k_idx or k2 not in k_idx:
            continue
        pair_records.append((k_idx[k1], k_idx[k2], 1 if higher == k1 else 0))
    pair_records = np.array(pair_records, dtype=np.int32)

    gem = pd.read_csv(SUP / "results/oof_predictions/llm_gemini_3_1_pro_preview_ingredients_image.csv").dropna(subset=["predicted_score"])
    gem_map = {(r["category"], int(r["product_code"])): r["predicted_score"] for _, r in gem.iterrows()}
    gem_oof = np.array([gem_map.get(k, np.nan) for k in analog_keys])
    y_all = np.array([pf[k].get("mean_similarity") for k in analog_keys], dtype=np.float64)

    def _pca_fit_transform(Xtr, Xall, var_target=0.95):
        """PCA-95% via scipy.linalg.svd with the gesvd LAPACK driver.

        Replaces sklearn.PCA(svd_solver='full') which uses LAPACK gesdd by
        default; gesdd is faster but occasionally raises 'SVD did not
        converge' on ill-conditioned blocks (the taste_gnn compound block
        triggers this in inner-fold matrices). gesvd is slower but always
        converges, giving fully reproducible nested LOOCV across runs.
        """
        mu = Xtr.mean(axis=0)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            _U, S, Vt = scipy.linalg.svd(
                Xtr - mu, full_matrices=False, lapack_driver="gesvd",
            )
            sq = S ** 2
            total = sq.sum()
            var_ratio = sq / total if total > 0 else sq
            n = int(np.searchsorted(np.cumsum(var_ratio), var_target) + 1)
            n = min(n, len(S))
            return (Xall - mu) @ Vt[:n].T

    def build_X_fold(train_mask):
        blocks = []
        for l in letters:
            Xraw = RAW[l]
            if Xraw.shape[1] <= 10:
                blocks.append(Xraw); continue
            Xtr = Xraw[train_mask]
            blocks.append(_pca_fit_transform(Xtr, Xraw, var_target=0.95))
        Xcat = np.hstack(blocks)
        return StandardScaler().fit(Xcat[train_mask]).transform(Xcat).astype(np.float32)

    def l2_bt_fit(Xd, y):
        m = LogisticRegression(penalty="l2", solver="liblinear", C=1.0,
                               random_state=SEED, max_iter=1000, n_jobs=1).fit(Xd, y)
        return m.coef_.ravel()

    def fit_excl(exclude):
        keep = np.array([i1 not in exclude and i2 not in exclude for i1, i2, _ in pair_records])
        tm = np.ones(N, dtype=bool)
        for idx in exclude:
            tm[idx] = False
        X = build_X_fold(tm)
        pr = pair_records[keep]
        Xd = X[pr[:, 0]] - X[pr[:, 1]]
        y = pr[:, 2].astype(np.int32)
        if len(np.unique(y)) < 2:
            return None, X
        return l2_bt_fit(Xd, y), X

    def outer_fold(i):
        beta, X = fit_excl(frozenset([i]))
        return np.nan if beta is None else float(beta @ X[i])

    def inner_pair(ij):
        i, j = ij
        beta, X = fit_excl(frozenset([i, j]))
        if beta is None:
            return (i, j, np.nan, np.nan)
        return (i, j, float(beta @ X[i]), float(beta @ X[j]))

    print(f"  outer LOOCV N={N} (letters={letters}, C-dim={compound_dim})")
    t0 = time.time()
    bt_outer = np.array(Parallel(n_jobs=-1)(delayed(outer_fold)(i) for i in range(N)))
    print(f"  outer done in {time.time()-t0:.0f}s")

    t1 = time.time()
    pairs = list(combinations(range(N), 2))
    print(f"  inner nested ({len(pairs)} pairs)")
    results = Parallel(n_jobs=-1)(delayed(inner_pair)(ij) for ij in pairs)
    bt_inner = np.full((N, N), np.nan)
    for i, j, si, sj in results:
        bt_inner[i, j] = sj
        bt_inner[j, i] = si
    print(f"  inner done in {time.time()-t1:.0f}s")

    cats_arr = np.array([k[0] for k in analog_keys])
    bt_outer_rk = _rank_per_cat(bt_outer, cats_arr)
    bt_inner_rk = np.full_like(bt_inner, np.nan, dtype=np.float64)
    for i in range(N):
        bt_inner_rk[i] = _rank_per_cat(bt_inner[i], cats_arr)
    gem_rk = _rank_per_cat(gem_oof, cats_arr)

    oof = np.zeros(N)
    for i in range(N):
        mask = np.ones(N, dtype=bool); mask[i] = False
        valid = mask & ~np.isnan(bt_inner_rk[i]) & ~np.isnan(gem_rk) & ~np.isnan(y_all)
        if valid.sum() < 2:
            oof[i] = np.nan; continue
        X_tr = np.column_stack([bt_inner_rk[i, valid], gem_rk[valid]])
        w, _ = nnls(X_tr, y_all[valid])
        oof[i] = np.array([bt_outer_rk[i], gem_rk[i]]) @ w

    df = pd.DataFrame({
        "category": [k[0] for k in analog_keys],
        "product_code": [k[1] for k in analog_keys],
        "true_score": y_all, "predicted_score": oof,
    }).dropna(subset=["predicted_score", "true_score"])
    return df, compute_all_metrics(df)["pairwise_accuracy"]


def rebuild_pf_with_taste_gnn() -> dict:
    """Swap compound embedding cache to taste_gnn_best, recompute C block."""
    pf = load_product_features()
    impute_missing_images_inplace(pf)
    products_df = load_nectar_products()
    tg_cache = SHARED / "data/caches/taste_gnn_best_compound_embeddings.pkl"
    new_C = _extract_compound_generic(
        products_df, tg_cache, "taste_gnn",
        product_agg="top3", ingredient_agg="weighted_average",
    )
    for k in pf:
        pf[k]["compound"] = new_C.get(k)
    return pf


def main():
    OOF = SUP / "results/oof_predictions"

    # Row 1: FART [CLS]
    archive = OOF / "nested_bt_gemini_nnls.csv"
    if archive.exists():
        df = pd.read_csv(archive).dropna(subset=["predicted_score", "true_score"])
        fart_acc = compute_all_metrics(df)["pairwise_accuracy"]
        print(f"FART [CLS] (reused archived L2+SNCTI+Gem NNLS): {fart_acc:.4f}")
    else:
        pf = load_product_features()
        impute_missing_images_inplace(pf)
        df, fart_acc = nested_l2_bt_nnls(pf, "SNCTI", compound_dim=768)
        df.to_csv(archive, index=False)
        print(f"FART [CLS] (fresh L2+SNCTI+Gem NNLS): {fart_acc:.4f}")

    # Row 2: taste_gnn C
    print("\nSwapping C → taste_gnn_best_compound_embeddings...")
    pf_tg = rebuild_pf_with_taste_gnn()
    df, tg_acc = nested_l2_bt_nnls(pf_tg, "SNCTI", compound_dim=300)
    out = OOF / "nested_bt_l2_SNCTI_tastegnn_gemini_nnls.csv"
    df.to_csv(out, index=False)
    print(f"taste_gnn: {tg_acc:.4f}  → {out.name}")

    # Row 3: no C
    print("\nDropping C block entirely (letters=SNTI)...")
    pf_noC = load_product_features()
    impute_missing_images_inplace(pf_noC)
    df, noC_acc = nested_l2_bt_nnls(pf_noC, "SNTI", compound_dim=768)
    out = OOF / "nested_bt_l2_SNTI_gemini_nnls.csv"
    df.to_csv(out, index=False)
    print(f"no C: {noC_acc:.4f}  → {out.name}")

    print("\n" + "=" * 50)
    print("table_taste_gnn.tex NECTAR pw-acc column:")
    print("=" * 50)
    print(f"  FART [CLS]:    {fart_acc:.4f}")
    print(f"  taste_gnn:     {tg_acc:.4f}  ({tg_acc - fart_acc:+.4f} vs FART)")
    print(f"  no C:          {noC_acc:.4f}  ({noC_acc - fart_acc:+.4f} vs FART)")


if __name__ == "__main__":
    main()
