"""Render table_molecular_prediction.tex.

Models-as-rows layout: 5 model rows (3 baselines from the FART paper +
FART + our GNN) x 5 metric columns (Accuracy, Precision, Recall, F1,
AUROC). Tree-based baseline numbers are reported in Zimmermann et
al. 2025 (Table 1) and are reproduced here with citation; FART and GNN
are computed from predictions.parquet with 95% BCa CIs (10,000 resamples,
seed 42).

All rows are single-checkpoint single-pass: no test-time augmentation
for FART (the ``FART augmented + confidence'' TTA variant from
Zimmermann et al. is not included here for comparability) and no
multi-seed ensembling for the GNN. The same val-best GNN checkpoint
provides the 300-dim penultimate embeddings used in
Table~\\ref{tab:gnn-per-model}.

Reads predictions.parquet (y_true + per-class probs) written by
molecular.src.eval.evaluate.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TOP_DIR = Path(__file__).resolve().parents[2]
MOLECULAR_DIR = TOP_DIR / "molecular"
MOLECULAR_RESULTS = MOLECULAR_DIR / "results"
PAPER_DIR  = MOLECULAR_DIR.parent / "paper" / "molecular_prediction"
CSV_DIR    = MOLECULAR_RESULTS / "tables_csv"
OUT_TEX = PAPER_DIR / "table_molecular_prediction.tex"
OUT_CSV = CSV_DIR / "table_molecular_prediction.csv"
CACHE   = MOLECULAR_RESULTS / "cis_molecular_prediction.csv"

sys.path.insert(0, str(TOP_DIR))
from molecular.src.eval.metrics import (
    bootstrap_classification_cis,
    mcnemar_accuracy,
)
from molecular.src.data.dataset import LABEL_ORDER

N_BOOT = 10_000
SEED = 42

METRICS = [
    ("Accuracy",  "accuracy"),
    ("Precision", "precision"),
    ("Recall",    "recall"),
    ("F1",        "f1"),
    ("AUROC",     "auroc"),
]

# Baselines reported in Zimmermann et al. 2025 (FART paper, Table 1).
# Order: (label, accuracy, precision, recall, f1, auroc).
# We reproduce these point estimates with citation; the FART paper does
# not report CIs for these rows.
BASELINES = [
    ("XGBoost (fp)",      0.8988, 0.8169, 0.7400, 0.7661, 0.8520),
    ("XGBoost (fp+desc)", 0.8962, 0.8842, 0.7402, 0.7779, 0.8513),
    ("Balanced RF (fp)",  0.7972, 0.5845, 0.7322, 0.6014, 0.8391),
]


def _find_grid_best_eval() -> str:
    grid = MOLECULAR_RESULTS / "grid"
    best_link = grid / "best"
    if best_link.exists():
        target = best_link.resolve() if best_link.is_symlink() else best_link
        cand = target / "fart_test_eval"
        if (cand / "predictions.parquet").exists():
            return str(cand.relative_to(MOLECULAR_RESULTS))
    for cand in sorted(grid.glob("run_*/fart_test_eval/predictions.parquet")):
        return str(cand.parent.relative_to(MOLECULAR_RESULTS))
    raise SystemExit(
        "Missing fart_test_eval/predictions.parquet under molecular/results/grid/. "
        "Run select_best_and_evaluate.py first."
    )


COLUMNS = [
    ("FART", "fart_augmented_test"),
    ("GNN",  _find_grid_best_eval()),
]


def _short(v: float) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    s = f"{v:.3f}"
    return s[1:] if s.startswith("0") else s


def _delta_cell(d: float) -> str:
    if np.isnan(d):
        return "--"
    sign = "$+$" if d >= 0 else "$-$"
    return f"{sign}.{abs(int(round(d * 1000))):03d}"


def _overlaps(lo_a: float, hi_a: float,
              lo_b: float, hi_b: float) -> bool:
    """Two CIs overlap iff max(lo_a, lo_b) <= min(hi_a, hi_b)."""
    if any(np.isnan(v) for v in (lo_a, hi_a, lo_b, hi_b)):
        return False
    return max(lo_a, lo_b) <= min(hi_a, hi_b)


def _ci_cell(point: float, lo: float, hi: float,
             bold: bool = False, overlap: bool = False) -> str:
    """Inline ``X.XXX [X.XXX, X.XXX]`` matching table_gnn_per_model.tex.

    ``bold`` wraps the point estimate for the per-column max; ``overlap``
    appends ``$^\\dag$`` to mark cells whose CI overlaps the leader's CI.
    """
    pt = _short(point)
    if bold:
        pt = f"\\textbf{{{pt}}}"
    elif overlap:
        pt = f"{pt}$^\\dag$"
    if np.isnan(lo) or np.isnan(hi):
        return pt
    return f"{pt} [{_short(lo)},{_short(hi)}]"


def _baseline_cell(point: float, bold: bool) -> str:
    """Plain point-only cell for citation baselines (no CIs published)."""
    s = _short(point)
    return f"\\textbf{{{s}}}" if bold else s


def _stacked_ci_cell(point: float, lo: float, hi: float) -> str:
    """Two-line cell with point above and CI below (in scriptsize), matching
    table_molecular_per_class.tex. Keeps column width narrow enough to fit
    a 6-column table in a single NeurIPS column."""
    if np.isnan(lo) or np.isnan(hi):
        return _short(point)
    return (f"\\makecell{{{_short(point)} \\\\ "
            f"{{\\scriptsize [{_short(lo)},{_short(hi)}]}}}}")


def _load_cis_cache(label: str) -> dict | None:
    """Load cached BCa CIs for one column ('FART' or 'GNN'). Returns
    a dict mapping metric → {'point', 'ci_lo', 'ci_hi'}, or None if
    the cache doesn't have all 5 metrics for this label."""
    if not CACHE.exists():
        return None
    df = pd.read_csv(CACHE)
    sub = df[df["label"] == label]
    if sub.empty:
        return None
    out = {r.metric: {"point": float(r.point), "ci_lo": float(r.ci_lo), "ci_hi": float(r.ci_hi)}
           for r in sub.itertuples(index=False)}
    needed = {m for _, m in METRICS}
    if not needed.issubset(out.keys()):
        return None
    return out


def _save_cis_cache(by_label: dict[str, dict]) -> None:
    rows = []
    for label, cis_dict in sorted(by_label.items()):
        for metric, v in cis_dict.items():
            rows.append((label, metric, v["point"], v["ci_lo"], v["ci_hi"]))
    df = pd.DataFrame(rows, columns=["label", "metric", "point", "ci_lo", "ci_hi"])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index=False, float_format="%.6f")
    print(f"Wrote cache → {CACHE.relative_to(MOLECULAR_DIR.parent)}")


def load_run(run_dir: str, label: str | None = None) -> dict:
    """Bootstrap CIs from predictions.parquet for one column, plus raw
    correctness aligned by SMILES (for the paired McNemar's test).

    If ``label`` is provided and a cache is available, loads BCa CIs
    from the cache; the parquet is still read for the cheap
    ``correct``/``smiles`` arrays McNemar's needs.
    """
    pred_path = MOLECULAR_RESULTS / run_dir / "predictions.parquet"
    df = pd.read_parquet(pred_path).sort_values("smiles").reset_index(drop=True)
    prob_cols = [f"prob_{name}" for name in LABEL_ORDER]
    y_true = df["y_true"].to_numpy()
    probs = df[prob_cols].to_numpy()

    cis = None
    if label is not None:
        cis = _load_cis_cache(label)
    if cis is None:
        cis = bootstrap_classification_cis(y_true, probs, n_bootstrap=N_BOOT,
                                           seed=SEED, n_classes=len(LABEL_ORDER))

    correct = (df["y_pred"].to_numpy() == y_true)
    return {"n": int(len(df)), "smiles": df["smiles"].to_numpy(),
            "correct": correct, **cis}


def main() -> int:
    print(f"Bootstrapping ({N_BOOT} resamples, seed={SEED}). "
          f"~30s/column on first run; cached to "
          f"{CACHE.relative_to(MOLECULAR_DIR.parent)} for re-renders.")
    columns: list[tuple[str, dict]] = []
    for label, spec in COLUMNS:
        d = load_run(spec, label=label)
        columns.append((label, d))
        msg = "  ".join(
            f"{m}={d[m]['point']:.3f} [{d[m]['ci_lo']:.3f}, {d[m]['ci_hi']:.3f}]"
            for _, m in METRICS
        )
        print(f"  {label:<32}  {msg}  n={d['n']}")

    # Save cache (no-op if it already covered all labels)
    by_label = {label: {m: d[m] for _, m in METRICS} for label, d in columns}
    if not (CACHE.exists() and all(_load_cis_cache(label) for label, _ in columns)):
        _save_cis_cache(by_label)

    # McNemar's on paired (FART correct?, taste_gnn correct?) outcomes.
    smi_a, c_a = columns[0][1]["smiles"], columns[0][1]["correct"]
    smi_b, c_b = columns[1][1]["smiles"], columns[1][1]["correct"]
    if not np.array_equal(smi_a, smi_b):
        raise SystemExit("SMILES mismatch between FART and taste_gnn predictions.parquet")
    mcn = mcnemar_accuracy(c_a, c_b)
    if mcn["p_value"] < 1e-3:
        p_str = "$<$ 0.001"
    elif mcn["p_value"] < 1e-2:
        p_str = f"$=$ {mcn['p_value']:.3f}"
    else:
        p_str = f"$=$ {mcn['p_value']:.2f}"
    print(f"\nMcNemar's: b={mcn['b']} (FART correct only), c={mcn['c']} "
          f"(taste_gnn correct only), chi2={mcn['chi2']:.2f}, p={mcn['p_value']:.4g}")

    fart_label, fart_d = columns[0]
    gnn_label,  gnn_d  = columns[1]

    # Per-column max across all 5 rows (3 baselines + FART + GNN). Compared
    # at 3-decimal precision so display ties are all bolded.
    col_max: dict[str, float] = {}
    leader_ci: dict[str, tuple[float, float]] = {}
    for _, m in METRICS:
        baseline_vals = []
        for label, acc, prec, rec, f1, auroc in BASELINES:
            val = {"accuracy": acc, "precision": prec,
                   "recall": rec, "f1": f1, "auroc": auroc}[m]
            baseline_vals.append((label, val))
        all_pts = ([p for _, p in baseline_vals]
                   + [fart_d[m]["point"], gnn_d[m]["point"]])
        col_max[m] = round(max(all_pts), 3)
        # Leader CI for overlap testing: prefer FART/GNN if they tie the
        # max (baselines don't carry CIs).
        for d in (fart_d, gnn_d):
            if round(d[m]["point"], 3) == col_max[m]:
                leader_ci[m] = (d[m]["ci_lo"], d[m]["ci_hi"])
                break

    def _row_baseline_cells(label: str, vals: dict[str, float]) -> list[str]:
        cells = []
        for _, m in METRICS:
            cells.append(_baseline_cell(
                vals[m], bold=round(vals[m], 3) == col_max[m]
            ))
        return cells

    def _row_ours_cells(d: dict) -> list[str]:
        cells = []
        for _, m in METRICS:
            p, lo, hi = d[m]["point"], d[m]["ci_lo"], d[m]["ci_hi"]
            is_best = round(p, 3) == col_max[m]
            ovl = (not is_best) and m in leader_ci \
                and _overlaps(lo, hi, *leader_ci[m])
            cells.append(_ci_cell(p, lo, hi, bold=is_best, overlap=ovl))
        return cells

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{0.95}",
        r"\caption{Taste classification on the FART test split "
        r"(2{,}254 molecules). Macro-averaged Precision, Recall, F1; "
        r"AUROC is macro one-vs-rest. The tree-based baselines and the "
        r"FART checkpoint are from~\citet{zimmermann2025chemical}; we "
        r"re-evaluate FART to add 95\% BCa CIs (10{,}000 resamples) and "
        r"train the GNN on the same task. The ``FART augmented + "
        r"confidence'' test-time augmentation variant is excluded for "
        r"single-checkpoint comparability. fp $=$ Morgan fingerprints; "
        r"desc $=$ molecular descriptors. McNemar's test on the FART "
        rf"vs GNN accuracy gap: $\chi^2$ $=$ {mcn['chi2']:.2f}, $p$ {p_str}. "
        r"\textbf{Bold} = best in column; $^\dag$ = CI overlaps the leader "
        r"(no significant difference at 95\%). "
        r"Per-class breakdown in "
        r"Table~\ref{tab:molecular-prediction-per-class}.}",
        r"\label{tab:molecular-prediction}",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"Model & Accuracy & Precision & Recall & F1 & AUROC \\",
        r"\midrule",
        r"\multicolumn{6}{@{}l}{\textit{Reported by~\citet{zimmermann2025chemical}}} \\",
    ]
    for label, acc, prec, rec, f1, auroc in BASELINES:
        vals = {"accuracy": acc, "precision": prec,
                "recall": rec, "f1": f1, "auroc": auroc}
        cells = _row_baseline_cells(label, vals)
        lines.append(f"\\quad {label} & " + " & ".join(cells) + r" \\")
    # FART belongs in the Zimmermann group (their checkpoint), but with our
    # CIs. Use inline ``X.XXX [lo, hi]`` cells so the table fits 1-column
    # NeurIPS textwidth without resizebox.
    fart_cells = _row_ours_cells(fart_d)
    lines.append(
        f"\\quad {fart_label} & " + " & ".join(fart_cells) + r" \\"
    )
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{@{}l}{\textit{This work}} \\")
    gnn_cells = _row_ours_cells(gnn_d)
    lines.append(
        f"\\quad {gnn_label} & " + " & ".join(gnn_cells) + r" \\"
    )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(lines))
    print(f"\nWrote {OUT_TEX.relative_to(MOLECULAR_DIR.parent)}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "source",
                    "accuracy", "accuracy_ci_lo", "accuracy_ci_hi",
                    "precision", "precision_ci_lo", "precision_ci_hi",
                    "recall", "recall_ci_lo", "recall_ci_hi",
                    "f1", "f1_ci_lo", "f1_ci_hi",
                    "auroc", "auroc_ci_lo", "auroc_ci_hi"])
        for label, acc, prec, rec, f1, auroc in BASELINES:
            w.writerow([label, "Zimmermann et al. 2025",
                        f"{acc:.4f}", "", "",
                        f"{prec:.4f}", "", "",
                        f"{rec:.4f}", "", "",
                        f"{f1:.4f}", "", "",
                        f"{auroc:.4f}", "", ""])
        for label, d in [(fart_label, fart_d), (gnn_label, gnn_d)]:
            row = [label, "this work"]
            for _, m in METRICS:
                row += [f"{d[m]['point']:.4f}",
                        f"{d[m]['ci_lo']:.4f}",
                        f"{d[m]['ci_hi']:.4f}"]
            w.writerow(row)
    print(f"Wrote {OUT_CSV.relative_to(MOLECULAR_DIR.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
