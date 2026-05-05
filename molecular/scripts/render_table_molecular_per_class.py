"""Render table_molecular_per_class.tex.

Per-class P / R / F1 / AUROC + support for FART Augmented and taste\\_gnn
on the FART test split, with 95% BCa bootstrap CIs (10,000 resamples,
seed 42) inline next to each point estimate. Classes as columns, metrics
as rows. The wide CIs on the umami column (n=6) make the macro-CI width
in Table~\\ref{tab:molecular-prediction} self-explanatory.

Reads predictions.parquet for both runs.
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
OUT_TEX = PAPER_DIR / "table_molecular_per_class.tex"
OUT_CSV = CSV_DIR / "table_molecular_per_class.csv"
CACHE   = MOLECULAR_RESULTS / "cis_molecular_per_class.csv"

sys.path.insert(0, str(TOP_DIR))
from molecular.src.eval.metrics import bootstrap_per_class_cis
from molecular.src.data.dataset import LABEL_ORDER

N_BOOT = 10_000
SEED = 42


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
    raise SystemExit("Missing fart_test_eval/predictions.parquet under taste_gnn/results/grid/.")


ROWS = [
    ("FART", "fart_augmented_test"),
    ("GNN",       _find_grid_best_eval()),
]


def _short(v: float) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    s = f"{v:.3f}"
    return s[1:] if s.startswith("0") else s


def _overlaps(lo_a: float, hi_a: float,
              lo_b: float, hi_b: float) -> bool:
    """Two CIs overlap iff max(lo_a, lo_b) <= min(hi_a, hi_b)."""
    if any(np.isnan(v) for v in (lo_a, hi_a, lo_b, hi_b)):
        return False
    return max(lo_a, lo_b) <= min(hi_a, hi_b)


def _ci_cell(point: float, lo: float, hi: float,
             bold: bool = False, overlap: bool = False) -> str:
    """Stacked cell: point on top, CI in \\scriptsize below. Lets the
    6-column table fit a single NeurIPS column at \\footnotesize without
    \\resizebox (which would scale text inconsistently across appendix
    tables). Uses built-in \\shortstack — no extra package.

    ``overlap=True`` appends ``$^\\dag$`` to mark cells whose CI overlaps
    the per-(class, metric) leader's CI."""
    p = _short(point)
    if bold:
        p = rf"\textbf{{{p}}}"
    elif overlap:
        p = p + r"$^\dag$"
    if np.isnan(lo) or np.isnan(hi):
        return p
    return (f"\\shortstack{{{p} \\\\ "
            f"{{\\scriptsize [{_short(lo)},{_short(hi)}]}}}}")


def _load_cache_for(label: str) -> list[dict] | None:
    """Reconstruct per-class CIs for one row from cache, or None if missing."""
    if not CACHE.exists():
        return None
    df = pd.read_csv(CACHE)
    sub = df[df["label"] == label]
    if sub.empty:
        return None
    by_class: dict[int, dict[str, dict]] = {}
    for r in sub.itertuples(index=False):
        by_class.setdefault(int(r.class_idx), {})[r.metric] = {
            "point": float(r.point), "ci_lo": float(r.ci_lo), "ci_hi": float(r.ci_hi)
        }
    n_classes = len(LABEL_ORDER)
    needed_metrics = {"precision", "recall", "f1", "auroc", "support"}
    if set(by_class.keys()) != set(range(n_classes)):
        return None
    for ci in range(n_classes):
        if not needed_metrics.issubset(by_class[ci].keys()):
            return None
    return [by_class[ci] for ci in range(n_classes)]


def _save_cache(rows: list[tuple[str, list[dict]]]) -> None:
    """Cache per-class metrics. The 'support' entry is a count and only
    has a 'point' field; treat ci_lo/ci_hi as nan in that case."""
    out = []
    for label, per_class in rows:
        for ci, cls_metrics in enumerate(per_class):
            for metric, v in cls_metrics.items():
                lo = v.get("ci_lo", float("nan"))
                hi = v.get("ci_hi", float("nan"))
                out.append((label, ci, metric, v["point"], lo, hi))
    df = pd.DataFrame(out, columns=["label", "class_idx", "metric", "point", "ci_lo", "ci_hi"])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index=False, float_format="%.6f")
    print(f"Wrote cache → {CACHE.relative_to(MOLECULAR_DIR.parent)}")


def load_run(run_dir: str, label: str | None = None) -> list[dict]:
    if label is not None:
        cached = _load_cache_for(label)
        if cached is not None:
            return cached
    pred_path = MOLECULAR_RESULTS / run_dir / "predictions.parquet"
    df = pd.read_parquet(pred_path)
    prob_cols = [f"prob_{name}" for name in LABEL_ORDER]
    y_true = df["y_true"].to_numpy()
    probs = df[prob_cols].to_numpy()
    return bootstrap_per_class_cis(y_true, probs, n_bootstrap=N_BOOT,
                                   seed=SEED, n_classes=len(LABEL_ORDER))


def main() -> int:
    print(f"Bootstrapping per-class CIs ({N_BOOT} resamples, seed={SEED}). "
          f"~1 min/row on first run; cached to "
          f"{CACHE.relative_to(MOLECULAR_DIR.parent)} for re-renders.")
    rows: list[tuple[str, list[dict]]] = []
    for label, spec in ROWS:
        per_class = load_run(spec, label=label)
        rows.append((label, per_class))
        print(f"  {label}")
        for cname, pc in zip(LABEL_ORDER, per_class):
            print(f"    {cname:<10}  n={int(pc['support']['point']):<5} "
                  f"P={pc['precision']['point']:.3f} [{pc['precision']['ci_lo']:.3f}, {pc['precision']['ci_hi']:.3f}]  "
                  f"F1={pc['f1']['point']:.3f} [{pc['f1']['ci_lo']:.3f}, {pc['f1']['ci_hi']:.3f}]")

    if not (CACHE.exists() and all(_load_cache_for(label) for label, _ in rows)):
        _save_cache(rows)

    classes = LABEL_ORDER
    support_by_class = {c: int(rows[0][1][i]["support"]["point"])
                        for i, c in enumerate(classes)}

    lines = [
        r"\begin{center}",
        r"\captionof{table}{Per-class breakdown for Table~\ref{tab:molecular-prediction}: "
        r"Precision, Recall, F1 and one-vs-rest AUROC of FART and "
        r"GNN on each FART test class, with 95\% BCa CIs (10{,}000 resamples; "
        r"point on top, CI in brackets below). Support row gives the number "
        r"of test molecules per class. Umami CIs span much of $[0, 1]$ "
        r"because $n=6$. \textbf{Bold} = winner of FART vs GNN per "
        r"(class, metric); $^\dag$ = CI overlaps the leader (no significant "
        r"difference at 95\%).}",
        r"\label{tab:molecular-prediction-per-class}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{tabular}{@{}l" + "c" * len(classes) + r"@{}}",
        r"\toprule",
        r" & " + " & ".join(c.title() for c in classes) + r" \\",
        r"\midrule",
        r"Support & " + " & ".join(str(support_by_class[c]) for c in classes) + r" \\",
        r"\midrule",
    ]

    metric_specs = [
        ("Precision", "precision"),
        ("Recall",    "recall"),
        ("F1",        "f1"),
        ("AUROC",     "auroc"),
    ]
    # Per-(class, metric) winner across the two model rows. Compared at
    # 3-decimal precision so display ties are all bolded.
    n_classes = len(classes)
    winners: dict[tuple[int, str], set[int]] = {}
    leader_ci: dict[tuple[int, str], tuple[float, float]] = {}
    for _, metric_key in metric_specs:
        for ci in range(n_classes):
            pts = [round(rows[ri][1][ci][metric_key]["point"], 3)
                   for ri in range(len(rows))]
            mx = max(pts)
            winners[(ci, metric_key)] = {ri for ri, p in enumerate(pts)
                                          if p == mx}
            leader_ri = next(iter(winners[(ci, metric_key)]))
            m = rows[leader_ri][1][ci][metric_key]
            leader_ci[(ci, metric_key)] = (m["ci_lo"], m["ci_hi"])

    for ri, (label, per_class) in enumerate(rows):
        if ri > 0:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{1 + len(classes)}}}{{@{{}}l}}{{\textit{{{label}}}}} \\")
        for metric_label, metric_key in metric_specs:
            cells = []
            for ci, _ in enumerate(classes):
                m = per_class[ci][metric_key]
                is_best = ri in winners[(ci, metric_key)]
                ovl = (not is_best) and _overlaps(
                    m["ci_lo"], m["ci_hi"],
                    *leader_ci[(ci, metric_key)],
                )
                cells.append(_ci_cell(m["point"], m["ci_lo"], m["ci_hi"],
                                      bold=is_best, overlap=ovl))
            lines.append(rf"\quad {metric_label} & " + " & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}", ""]

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(lines))
    print(f"\nWrote {OUT_TEX.relative_to(MOLECULAR_DIR.parent)}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "class", "support",
                    "precision_point", "precision_ci_lo", "precision_ci_hi",
                    "recall_point",    "recall_ci_lo",    "recall_ci_hi",
                    "f1_point",        "f1_ci_lo",        "f1_ci_hi",
                    "auroc_point",     "auroc_ci_lo",     "auroc_ci_hi"])
        for label, per_class in rows:
            clean_label = label.replace(r"\_", "_")
            for ci, cname in enumerate(classes):
                pc = per_class[ci]
                row = [clean_label, cname, int(pc["support"]["point"])]
                for m in ("precision", "recall", "f1", "auroc"):
                    row += [f"{pc[m]['point']:.4f}",
                            f"{pc[m]['ci_lo']:.4f}" if not np.isnan(pc[m]['ci_lo']) else "",
                            f"{pc[m]['ci_hi']:.4f}" if not np.isnan(pc[m]['ci_hi']) else ""]
                w.writerow(row)
    print(f"Wrote {OUT_CSV.relative_to(MOLECULAR_DIR.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
