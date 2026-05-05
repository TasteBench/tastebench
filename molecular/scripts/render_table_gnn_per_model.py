"""Compound-encoder swap (FART [CLS] vs. taste_gnn) per model.

Rows mirror table_results.tex (minus the two LLM rows -- which take
ingredient text + image, not compound embeddings -- and Random):

  Unsupervised: MMRF (cosine), MMRF (L2)
  Supervised -- linear:      Ridge
  Supervised -- pairwise:    BT, Hierarchical BT, Kernel RankSVM
  Supervised -- nonlinear:   LightGBM
  Ensemble (BT + Gemini):    NNLS, Rank average, Mean

Both encoders are single-checkpoint single-pass: FART [CLS] is the 768-dim
output of the FartLabs/FART_Augmented HuggingFace checkpoint; taste_gnn is
the 300-dim penultimate-layer output of the val-best D-MPNN checkpoint
(same checkpoint as Table~\\ref{tab:molecular-prediction}). Per-modality
PCA@95% is applied per LOOCV fold to handle dimensional asymmetry between
modalities.

Each row reports pairwise accuracy [95\\% BCa CI] under C = FART [CLS]
vs. C = taste\\_gnn, with \\Delta = taste\\_gnn - FART (bold if \\geq .02).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

MOLECULAR_DIR = Path(__file__).resolve().parent.parent
FOOD_SIM_DIR = MOLECULAR_DIR.parent / "food_similarity"
sys.path.insert(0, str(FOOD_SIM_DIR))

# Use the Python-loop BCa bootstrap so the FART column here reproduces the
# Pw. Acc. column of Table~\ref{tab:results} byte-identically. The vectorized
# compute_bca_pw_acc is faster but produces +/-0.001 third-decimal CI drift.
from evaluation.bootstrap import compute_bca_cis
from evaluation.metrics import compute_all_metrics

OOF = FOOD_SIM_DIR / "results" / "oof_predictions"
PAPER_DIR  = MOLECULAR_DIR.parent / "paper" / "molecular_prediction"
CSV_DIR    = MOLECULAR_DIR / "results" / "tables_csv"
OUT_TEX = PAPER_DIR / "table_gnn_per_model.tex"
OUT_CSV = CSV_DIR / "table_gnn_per_model.csv"
CACHE   = MOLECULAR_DIR / "results" / "cis_gnn_per_model.csv"

# (group_label, model_display, fart_csv, tastegnn_csv)
GROUPS = [
    (r"\textit{Unsupervised}", [
        ("MMRF (cosine)",    "dist_pred_cosine_NCI.csv",        "dist_pred_cosine_NCI_tastegnn.csv"),
        ("MMRF (L2)",        "dist_pred_l2_NCI.csv",            "dist_pred_l2_NCI_tastegnn.csv"),
    ]),
    (r"\textit{Supervised -- linear}", [
        ("Ridge",            "ridge_SNCTI_bench.csv",           "ridge_SNCTI_tastegnn.csv"),
    ]),
    (r"\textit{Supervised -- pairwise}", [
        ("Bradley--Terry",   "bradley_terry_SNCTI_bench.csv",   "bradley_terry_SNCTI_tastegnn.csv"),
        ("Hierarchical BT",  "hierarchical_bt_SNCTI_bench.csv", "hierarchical_bt_SNCTI_tastegnn.csv"),
        ("Kernel RankSVM",   "kernel_ranksvm_SNCTI_bench.csv",  "kernel_ranksvm_SNCTI_tastegnn.csv"),
    ]),
    (r"\textit{Supervised -- nonlinear}", [
        ("LightGBM",         "lightgbm_reg_SNCTI_bench.csv",    "lightgbm_reg_SNCTI_tastegnn.csv"),
    ]),
    (r"\textit{Ensemble (BT $+$ Gemini)}", [
        ("NNLS",             "nested_bt_gemini_nnls.csv",       "nested_bt_l2_SNCTI_tastegnn_gemini_nnls.csv"),
        ("Rank average",     "nested_bt_gemini_rank.csv",       "nested_bt_l2_SNCTI_tastegnn_gemini_rank.csv"),
        ("Mean",             "nested_bt_gemini_mean.csv",       "nested_bt_l2_SNCTI_tastegnn_gemini_mean.csv"),
    ]),
]


def _short(v):
    if np.isnan(v): return "--"
    s = f"{v:.3f}"
    return s[1:] if s.startswith("0") else ("$-" + s[2:] + "$" if s.startswith("-0") else s)


def _overlaps(lo_a, hi_a, lo_b, hi_b):
    """Two CIs overlap iff max(lo_a, lo_b) <= min(hi_a, hi_b)."""
    if any(np.isnan(v) for v in (lo_a, hi_a, lo_b, hi_b)):
        return False
    return max(lo_a, lo_b) <= min(hi_a, hi_b)


def fmt_ci(p, lo, hi, bold: bool = False, overlap: bool = False):
    """Inline ``X.XXX [lo,hi]``. ``bold`` wraps the point estimate; ``overlap``
    appends ``$^\\dag$`` to mark cells whose CI overlaps the column leader's CI."""
    if np.isnan(p): return "--"
    pt = _short(p)
    if bold:
        pt = f"\\textbf{{{pt}}}"
    elif overlap:
        pt = f"{pt}$^\\dag$"
    return f"{pt} [{_short(lo)},{_short(hi)}]"


def _delta(d):
    if np.isnan(d): return "--"
    sign = "$+$" if d > 0 else "$-$"
    return f"{sign}.{abs(int(round(d * 1000))):03d}"


def _bca_pw_acc(oof_csv: str) -> tuple[float, float, float]:
    """Pairwise-accuracy point estimate + BCa CI via the Python-loop bootstrap.

    Returns (point, ci_lo, ci_hi). Uses ``compute_bca_cis`` from
    ``evaluation/bootstrap.py`` -- the same bootstrap that produces
    ``table_results.tex``. Slightly slower than the vectorized version
    but byte-identical to the Pw. Acc. column of Table~\\ref{tab:results}.
    """
    df = pd.read_csv(OOF / oof_csv).dropna(subset=["predicted_score", "true_score"])
    point = compute_all_metrics(df)["pairwise_accuracy"]
    cis = compute_bca_cis(df, n_bootstrap=10_000, seed=42)
    lo, hi = cis["pairwise_accuracy"]
    return point, lo, hi


def _load_cache(todo) -> dict[str, tuple[float, float, float]] | None:
    if not CACHE.exists():
        return None
    df = pd.read_csv(CACHE)
    by_csv = {r.oof_csv: (float(r.point), float(r.ci_lo), float(r.ci_hi))
              for r in df.itertuples(index=False)}
    needed = {t[2] for t in todo}
    if not needed.issubset(by_csv.keys()):
        return None
    return by_csv


def _save_cache(by_csv: dict) -> None:
    rows = [(k, p, lo, hi) for k, (p, lo, hi) in sorted(by_csv.items())]
    df = pd.DataFrame(rows, columns=["oof_csv", "point", "ci_lo", "ci_hi"])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index=False, float_format="%.6f")
    print(f"Wrote cache → {CACHE.relative_to(MOLECULAR_DIR.parent)}")


def main() -> int:
    # Flatten all OOF csvs we need to bootstrap.
    todo: list[tuple[str, str, str]] = []  # (group_lbl, label, csv_name)
    for group_lbl, model_rows in GROUPS:
        for label, f_fart, f_tg in model_rows:
            todo.append((group_lbl, label, f_fart))
            todo.append((group_lbl, label, f_tg))

    by_csv = _load_cache(todo)
    if by_csv is not None:
        print(f"Loaded {len(by_csv)} cached BCa CIs from "
              f"{CACHE.relative_to(MOLECULAR_DIR.parent)}; skipping bootstrap.")
    else:
        print(f"Bootstrapping {len(todo)} OOF files in parallel "
              f"(10,000 resamples, seed 42, Python-loop bootstrap). "
              f"~5 min on 8 cores; cached for instant re-renders.")
        cis = Parallel(n_jobs=-1, backend="loky", verbose=5)(
            delayed(_bca_pw_acc)(t[2]) for t in todo
        )
        by_csv = {t[2]: cis[i] for i, t in enumerate(todo)}
        _save_cache(by_csv)

    all_rows = []
    for group_lbl, model_rows in GROUPS:
        print(f"\n  {group_lbl}")
        group_rows = []
        for label, f_fart, f_tg in model_rows:
            p_f, lo_f, hi_f = by_csv[f_fart]
            p_t, lo_t, hi_t = by_csv[f_tg]
            group_rows.append((label, (p_f, lo_f, hi_f), (p_t, lo_t, hi_t)))
            print(f"    {label:<18}  FART={p_f:.4f}  GNN={p_t:.4f}  "
                  f"Δ={p_t - p_f:+.4f}")
        all_rows.append((group_lbl, group_rows))


    # Per-column max + leader CI for overlap markers. Compared at 3-decimal
    # precision so display ties are all bolded.
    fart_points = [p_f for _, rows in all_rows
                   for _, (p_f, _, _), _ in rows]
    gnn_points  = [p_t for _, rows in all_rows
                   for _, _, (p_t, _, _) in rows]
    fart_max = round(max(fart_points), 3)
    gnn_max  = round(max(gnn_points), 3)
    leader_fart_ci = next(((lo, hi) for _, rows in all_rows
                           for _, (p, lo, hi), _ in rows
                           if round(p, 3) == fart_max), (np.nan, np.nan))
    leader_gnn_ci  = next(((lo, hi) for _, rows in all_rows
                           for _, _, (p, lo, hi) in rows
                           if round(p, 3) == gnn_max),  (np.nan, np.nan))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{0.95}",
        r"\caption{Pairwise accuracy on 215 NECTAR products (935 within-category pairs; LOOCV) when "
        r"the compound block is swapped from FART to GNN embeddings, "
        r"for every non-LLM model in Table~\ref{tab:results} (otherwise "
        r"identical setup). 95\% BCa CIs (10{,}000 resamples). $\Delta$ $=$ GNN $-$ FART. "
        r"\textbf{Bold} = best in column; $^\dag$ = CI overlaps the leader "
        r"(no significant difference at 95\%).}",
        r"\label{tab:gnn-per-model}",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r"Model & C $=$ FART & C $=$ GNN & $\Delta$ \\",
        r"\midrule",
    ]
    first = True
    for group_lbl, group_rows in all_rows:
        if not first:
            lines.append(r"\midrule")
        first = False
        lines.append(rf"\multicolumn{{4}}{{@{{}}l}}{{{group_lbl}}} \\")
        for label, (p_f, lo_f, hi_f), (p_t, lo_t, hi_t) in group_rows:
            d = p_t - p_f
            f_best = round(p_f, 3) == fart_max
            t_best = round(p_t, 3) == gnn_max
            f_ovl = (not f_best) and _overlaps(lo_f, hi_f, *leader_fart_ci)
            t_ovl = (not t_best) and _overlaps(lo_t, hi_t, *leader_gnn_ci)
            lines.append(
                f"\\quad {label} & "
                f"{fmt_ci(p_f, lo_f, hi_f, bold=f_best, overlap=f_ovl)} & "
                f"{fmt_ci(p_t, lo_t, hi_t, bold=t_best, overlap=t_ovl)} & "
                f"{_delta(d)} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(lines))
    print(f"\nWrote {OUT_TEX.relative_to(MOLECULAR_DIR.parent)}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "model",
                    "pw_acc_fart", "ci_lo_fart", "ci_hi_fart",
                    "pw_acc_tastegnn", "ci_lo_tastegnn", "ci_hi_tastegnn",
                    "delta"])
        for group_lbl, group_rows in all_rows:
            group_clean = (group_lbl.replace(r"\textit{", "").replace("}", "")
                                    .replace("--", "-").replace("$+$", "+"))
            for label, (p_f, lo_f, hi_f), (p_t, lo_t, hi_t) in group_rows:
                model_clean = label.replace("--", "-").replace(r"\_", "_")
                w.writerow([group_clean, model_clean,
                            f"{p_f:.4f}", f"{lo_f:.4f}", f"{hi_f:.4f}",
                            f"{p_t:.4f}", f"{lo_t:.4f}", f"{hi_t:.4f}",
                            f"{p_t - p_f:+.4f}"])
    print(f"Wrote {OUT_CSV.relative_to(MOLECULAR_DIR.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
