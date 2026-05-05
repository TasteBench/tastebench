"""Generate the four taste_gnn-side OOFs needed for the encoder-transfer table.

Reuses the existing `run_taste_gnn_nectar.py` L2-BT pipeline (so the
taste_gnn NNLS row stays byte-identical) but adds three more aggregations:

  nested_bt_l2_SNCTI_tastegnn_gemini_rank.csv   — within-category rank avg
  nested_bt_l2_SNCTI_tastegnn_gemini_mean.csv   — (bt_outer + gemini) / 2
  dist_pred_cosine_NCI_tastegnn.csv             — MMRF cosine on NCI (C=taste_gnn)
  dist_pred_l2_NCI_tastegnn.csv                 — MMRF L2     on NCI (C=taste_gnn)

The rank/mean aggregations only need `bt_outer` (LOOCV, n fits, not n^2),
so this is fast.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings("ignore")

TOP_DIR = Path(__file__).resolve().parents[2]
SUP = TOP_DIR / "food_similarity"
sys.path.insert(0, str(SUP))

from data.loocv import get_analog_keys
from evaluation.metrics import compute_all_metrics
from models.distance_predictor import DistancePredictor
from train.paper_table import impute_missing_images_inplace
from train.run_taste_gnn_nectar import rebuild_pf_with_taste_gnn

SEED = 42
OOF = SUP / "results" / "oof_predictions"
GEM_FILE = OOF / "llm_gemini_3_1_pro_preview_ingredients_image.csv"


def compute_bt_outer_tastegnn(pf: dict) -> tuple:
    """Outer LOOCV with L2-BT on SNCTI (C=taste_gnn). Returns (bt_outer, analog_keys, y_all, gem_oof)."""
    analog_keys = sorted(k for k, p in pf.items() if p.get("is_analog", True))
    k_idx = {k: i for i, k in enumerate(analog_keys)}
    N = len(analog_keys)

    compound_dim = 300
    for k in analog_keys:
        v = pf[k].get("compound")
        if v is not None:
            compound_dim = int(v.shape[0]); break

    SPECS = {"S": ("category_subset", 4), "N": ("nutrition", 6),
             "C": ("compound", compound_dim), "T": ("text", 1024), "I": ("image", 1024)}
    letters = "SNCTI"
    RAW = {}
    for l in letters:
        key_name, dim = SPECS[l]
        rows = [v.get(key_name).astype(np.float32) if v.get(key_name) is not None
                else np.zeros(dim, dtype=np.float32)
                for v in [pf[k] for k in analog_keys]]
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

    gem = pd.read_csv(GEM_FILE).dropna(subset=["predicted_score"])
    gem_map = {(r["category"], int(r["product_code"])): r["predicted_score"] for _, r in gem.iterrows()}
    gem_oof = np.array([gem_map.get(k, np.nan) for k in analog_keys])
    y_all = np.array([pf[k].get("mean_similarity") for k in analog_keys], dtype=np.float64)

    def build_X_fold(train_mask):
        blocks = []
        for l in letters:
            Xraw = RAW[l]
            if Xraw.shape[1] <= 10:
                blocks.append(Xraw); continue
            Xtr = Xraw[train_mask]
            p = PCA(n_components=min(0.95, min(Xtr.shape)-1), svd_solver="full",
                    random_state=SEED).fit(Xtr)
            blocks.append(p.transform(Xraw))
        Xcat = np.hstack(blocks)
        return StandardScaler().fit(Xcat[train_mask]).transform(Xcat).astype(np.float32)

    def l2_bt_fit(Xd, y):
        m = LogisticRegression(penalty="l2", solver="liblinear", C=1.0,
                               random_state=SEED, max_iter=1000, n_jobs=1).fit(Xd, y)
        return m.coef_.ravel()

    def outer_fold(i):
        keep = np.array([i1 != i and i2 != i for i1, i2, _ in pair_records])
        tm = np.ones(N, dtype=bool); tm[i] = False
        X = build_X_fold(tm)
        pr = pair_records[keep]
        Xd = X[pr[:, 0]] - X[pr[:, 1]]
        y = pr[:, 2].astype(np.int32)
        if len(np.unique(y)) < 2:
            return np.nan
        beta = l2_bt_fit(Xd, y)
        return float(beta @ X[i])

    t0 = time.time()
    bt_outer = np.array(Parallel(n_jobs=-1)(delayed(outer_fold)(i) for i in range(N)))
    print(f"  outer LOOCV done in {time.time()-t0:.0f}s  (N={N})")
    return bt_outer, analog_keys, y_all, gem_oof


def save_oof(analog_keys, y_all, preds, fname: str):
    df = pd.DataFrame({
        "category":     [k[0] for k in analog_keys],
        "product_code": [k[1] for k in analog_keys],
        "true_score":   y_all,
        "predicted_score": preds,
    }).dropna(subset=["predicted_score", "true_score"])
    path = OOF / fname
    df.to_csv(path, index=False)
    m = compute_all_metrics(df)
    print(f"    {fname:<55}  pw={m['pairwise_accuracy']:.4f}")
    return m["pairwise_accuracy"]


def run_mean(bt_outer, gem_oof, analog_keys, y_all) -> float:
    preds = 0.5 * (bt_outer + gem_oof)
    return save_oof(analog_keys, y_all, preds, "nested_bt_l2_SNCTI_tastegnn_gemini_mean.csv")


def run_rank(bt_outer, gem_oof, analog_keys, y_all) -> float:
    cats = np.array([k[0] for k in analog_keys])
    df = pd.DataFrame({"cat": cats, "bt": bt_outer, "gem": gem_oof})
    df["r_bt"]  = df.groupby("cat")["bt"].rank(pct=True)
    df["r_gem"] = df.groupby("cat")["gem"].rank(pct=True)
    preds = 0.5 * (df["r_bt"].values + df["r_gem"].values)
    return save_oof(analog_keys, y_all, preds, "nested_bt_l2_SNCTI_tastegnn_gemini_rank.csv")


def run_mmrf(pf: dict) -> None:
    """MMRF cosine + L2 on NCI with C=taste_gnn."""
    analog_keys = sorted(get_analog_keys(pf))
    for metric, tag in [("cosine", "cosine"), ("euclidean", "l2")]:
        model = DistancePredictor(
            feature_types=["category_nutrition", "compound", "image"],
            product_features=pf,
            distance_metric=metric,
            missing_feature_strategy="skip",
        )
        model.fit()
        scores = model.get_all_scores()
        rows = []
        for key in analog_keys:
            p = pf[key]
            rows.append({
                "category":     p["category"],
                "product_code": p["product_code"],
                "true_score":   p["mean_similarity"],
                "predicted_score": scores.get(key, np.nan),
            })
        df = pd.DataFrame(rows)
        fname = f"dist_pred_{tag}_NCI_tastegnn.csv"
        df.to_csv(OOF / fname, index=False)
        n_valid = df["predicted_score"].notna().sum()
        m = compute_all_metrics(df.dropna(subset=["predicted_score", "true_score"]))
        print(f"    {fname:<55}  pw={m['pairwise_accuracy']:.4f}  ({n_valid}/{len(df)})")


def run_supervised_loocv(pf: dict) -> None:
    """Run Ridge / Bradley-Terry / Hierarchical BT / Kernel RankSVM / LightGBM
    LOOCV on SNCTI features with C=taste_gnn (current shared cache), writing
    *_SNCTI_tastegnn.csv. Mirrors paper_table.py's _run_supervised but writes
    the _tastegnn suffix instead of _bench, force-overwriting any existing
    OOF so embedding swaps actually take effect.
    """
    import train.run_loocv as loocv_module
    from train.run_loocv import run_single

    saved = (loocv_module._SKIP_BOOTSTRAP, loocv_module._KNN_IMPUTE,
             loocv_module._PCA_VARIANCE)
    loocv_module._SKIP_BOOTSTRAP = True
    loocv_module._KNN_IMPUTE = 0           # images already imputed
    loocv_module._PCA_VARIANCE = 0.95
    try:
        for model_name in ("ridge", "bradley_terry", "hierarchical_bt",
                           "kernel_ranksvm", "lightgbm_reg"):
            out = SUP / "results/oof_predictions" / f"{model_name}_SNCTI_tastegnn.csv"
            if out.exists():
                out.unlink()           # force regeneration with new cache
            print(f"  {model_name} SNCTI -> {out.name}")
            try:
                run_single(model_name, "SNCTI", pf, suffix="_tastegnn")
            except Exception as e:
                print(f"    FAILED {model_name}: {e}")
                import traceback; traceback.print_exc()
    finally:
        (loocv_module._SKIP_BOOTSTRAP, loocv_module._KNN_IMPUTE,
         loocv_module._PCA_VARIANCE) = saved


def run_nnls_full_nested(pf: dict) -> None:
    """Run the full O(n^2) nested L2-BT + Gemini NNLS pipeline with
    C=taste_gnn, writing nested_bt_l2_SNCTI_tastegnn_gemini_nnls.csv.
    Reuses run_taste_gnn_nectar.nested_l2_bt_nnls so the methodology
    matches the archived NNLS row exactly."""
    from train.run_taste_gnn_nectar import nested_l2_bt_nnls
    out = SUP / "results/oof_predictions" / "nested_bt_l2_SNCTI_tastegnn_gemini_nnls.csv"
    if out.exists():
        out.unlink()
    df, acc = nested_l2_bt_nnls(pf, "SNCTI")  # compound_dim auto-detected
    df.to_csv(out, index=False)
    print(f"  NNLS nested -> {out.name}  pw={acc:.4f}")


def main():
    print("Swapping C -> taste_gnn_best_compound_embeddings...")
    pf_tg = rebuild_pf_with_taste_gnn()

    print("\n[1/4] MMRF (cosine, L2) with taste_gnn C block")
    run_mmrf(pf_tg)

    print("\n[2/4] 5 supervised models (Ridge/BT/HBT/KSVM/LightGBM) on SNCTI")
    run_supervised_loocv(pf_tg)

    print("\n[3/4] Outer L2-BT LOOCV + rank-avg / mean aggregations")
    bt_outer, analog_keys, y_all, gem_oof = compute_bt_outer_tastegnn(pf_tg)
    run_mean(bt_outer, gem_oof, analog_keys, y_all)
    run_rank(bt_outer, gem_oof, analog_keys, y_all)

    print("\n[4/4] Nested L2-BT + Gemini NNLS (full O(n^2) pipeline)")
    run_nnls_full_nested(pf_tg)


if __name__ == "__main__":
    main()
