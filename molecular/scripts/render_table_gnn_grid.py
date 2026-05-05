"""Render table_gnn_grid.tex.

A 12-row table showing val macro-F1 for each configuration in the GNN
hyperparameter grid (depth x dropout x class weighting). The selected
configuration (highest val macro-F1) is bolded; this is the
checkpoint used in Tables~\\ref{tab:molecular-prediction} and
\\ref{tab:gnn-per-model}.

Reads from molecular/results/grid/grid_summary.csv,
written by select_best_and_evaluate.py during grid search.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

TOP_DIR = Path(__file__).resolve().parents[2]
MOLECULAR_DIR = TOP_DIR / "molecular"
GRID_CSV = MOLECULAR_DIR / "results" / "grid" / "grid_summary.csv"
PAPER_DIR  = MOLECULAR_DIR.parent / "paper" / "molecular_prediction"
CSV_DIR    = MOLECULAR_DIR / "results" / "tables_csv"
OUT_TEX = PAPER_DIR / "table_gnn_grid.tex"
OUT_CSV = CSV_DIR / "table_gnn_grid.csv"


CW_DISPLAY = {
    "none": "none",
    "sqrt_inverse_frequency": "sqrt-inverse-frequency",
    "inverse_frequency": "inverse-frequency",
}


def _short(v: float) -> str:
    s = f"{v:.3f}"
    return s[1:] if s.startswith("0") else s


def main() -> int:
    df = pd.read_csv(GRID_CSV)
    if len(df) != 12:
        print(f"WARNING: expected 12 grid rows, found {len(df)}", file=sys.stderr)

    df = df.sort_values("val_macro_f1", ascending=False).reset_index(drop=True)
    best_idx = 0  # already sorted descending

    lines = [
        r"\begin{center}",
        r"\captionof{table}{Validation macro-F1 for all 12 configurations in the "
        r"GNN hyperparameter grid (depth $\times$ dropout $\times$ "
        r"class weighting). The selected configuration (\textbf{bold}) is "
        r"the highest val macro-F1; this checkpoint is used in "
        r"Tables~\ref{tab:molecular-prediction} and "
        r"\ref{tab:gnn-per-model}. Selection committed before evaluating "
        r"on the FART test set or any NECTAR data.}",
        r"\label{tab:gnn-grid}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{0.95}",
        r"\begin{tabular}{@{}cccc@{}}",
        r"\toprule",
        r"Depth & Dropout & Class weighting & Val macro-F1 \\",
        r"\midrule",
    ]
    for i, row in df.iterrows():
        depth = int(row["depth"])
        dropout = row["dropout"]
        cw = CW_DISPLAY[row["class_weighting"]]
        f1 = row["val_macro_f1"]
        cells = [str(depth), f"{dropout:g}", cw, _short(f1)]
        if i == best_idx:
            cells = [f"\\textbf{{{c}}}" for c in cells]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}", ""]

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(lines))
    print(f"Wrote {OUT_TEX.relative_to(MOLECULAR_DIR.parent)}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["depth", "dropout", "class_weighting", "val_macro_f1", "selected"])
        for i, row in df.iterrows():
            w.writerow([
                int(row["depth"]), row["dropout"],
                row["class_weighting"], f"{row['val_macro_f1']:.4f}",
                "yes" if i == best_idx else "no",
            ])
    print(f"Wrote {OUT_CSV.relative_to(MOLECULAR_DIR.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
