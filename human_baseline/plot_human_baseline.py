"""Publication figure for human panelist baseline.

Reads pre-computed analysis results from results/ and generates the
group-size curve PDF.

Usage:
    cd human_baseline
    python plot_human_baseline.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from human_panelist_baseline import _interpolate_k_star

# --- Paths ---
BASELINES_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASELINES_DIR / "results"
PAPER_DIR = BASELINES_DIR.parent / "paper" / "human_baseline"

# NeurIPS formatting
plt.rcParams.update({
    "font.size": 8,
    "font.family": "serif",
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save_fig(fig, stem: Path) -> None:
    """Save figure as PDF (the paper artifact)."""
    fig.savefig(stem.with_suffix(".pdf"))
    print(f"  Saved: {stem.with_suffix('.pdf')}")


def load_results():
    """Load pre-computed inputs needed for the group-size curve."""
    group_curve = pd.read_csv(RESULTS_DIR / "group_size_curve.csv")
    with open(RESULTS_DIR / "split_half_reliability.json") as f:
        split_half = json.load(f)
    with open(RESULTS_DIR / "summary.json") as f:
        summary = json.load(f)
    return group_curve, split_half, summary


def plot_group_size_curve(group_df, summary, split_half):
    """Group-size k vs pairwise accuracy.

    Two model reference lines (within-block and all-pairs) are drawn so
    the apples-to-apples (within-block) comparison with the human curve
    is visually distinct from the all-pairs reference value. k* is
    interpolated from where the human curve crosses the within-block
    model line.
    """
    # Figsize tuned for the side-by-side combined figure+table float
    # documented in human_baseline/README.md. Uses ~58% of the column
    # paired with the table at ~40%; ~1.82:1 aspect keeps the legend
    # and curve readable when scaled down.
    fig, ax = plt.subplots(figsize=(4.0, 2.2))

    model_wb  = summary.get("best_model_within_block_accuracy", summary["best_model_pairwise_accuracy"])
    model_all = summary.get("best_model_all_pairs_accuracy", model_wb)
    split_half_acc = split_half["pairwise_accuracy"]

    # Compute percentiles per k
    stats = group_df.groupby("k")["pairwise_accuracy"].agg(
        median="median",
        lo=lambda x: np.percentile(x, 5),
        hi=lambda x: np.percentile(x, 95),
    ).reset_index()

    ax.fill_between(stats["k"], stats["lo"], stats["hi"], alpha=0.2, color="#4C72B0")
    ax.plot(stats["k"], stats["median"], color="#4C72B0", linewidth=1.5, label="Human panel (median)")

    # Reference lines — two model scopes for explicit comparison
    ax.axhline(model_wb,  color="#C44E52", linestyle="--", linewidth=1.1,
               label=f"Best model, within-block ({model_wb:.3f})")
    if abs(model_all - model_wb) > 0.001:
        ax.axhline(model_all, color="#DD8452", linestyle="--", linewidth=1.0,
                   label=f"Best model, all pairs ({model_all:.3f})")
    ax.axhline(split_half_acc, color="#55A868", linestyle=":", linewidth=1,
               label=f"Split-half reliability ({split_half_acc:.3f})")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)

    # Compute k* from the within-block model accuracy (apples-to-apples)
    k_medians = group_df.groupby("k")["pairwise_accuracy"].median()
    k_star = _interpolate_k_star(k_medians, model_wb)
    if k_star is not None and not np.isnan(k_star):
        ax.axvline(k_star, color="#C44E52", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.annotate(
            f"k* = {k_star:.0f}",
            xy=(k_star, model_wb),
            xytext=(k_star + 3, model_wb - 0.03),
            fontsize=6,
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
        )

    ax.set_xlabel("Number of Panelists (k)")
    ax.set_ylabel("Pairwise Ranking Accuracy")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=6)
    ax.set_ylim(0.45, 1.02)

    fig.tight_layout()
    out_path = PAPER_DIR / "group_size_curve.pdf"
    _save_fig(fig, out_path.with_suffix(""))
    plt.close(fig)


if __name__ == "__main__":
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    group_curve, split_half, summary = load_results()
    print("Generating figure...")
    plot_group_size_curve(group_curve, summary, split_half)
    print("Done.")
