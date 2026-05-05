# Human-panelist baseline

This area characterizes the difficulty of the food-similarity task by
asking: how do groups of untrained omnivore panelists perform when
their individual ratings are aggregated into a panel mean? It produces
the human-baseline table fragment and the group-size curve figure
(panel mean accuracy vs. group size).

The analysis runs over NECTAR's individual-panelist similarity ratings
(BIBD-design consumer panel). The pipeline computes individual
LOO accuracy, panel-mean accuracy at each group size, split-half
reliability, and a per-category breakdown.

## Outputs

Rendered artifacts land in [`../paper/human_baseline/`](../paper/human_baseline/):

- `human_baseline_table.tex` — produced by `human_panelist_baseline.py` (needs NECTAR)
- `group_size_curve.pdf` — produced by `plot_human_baseline.py`

The `.tex` is a *fragment* (just the `\begin{tabular}...\end{tabular}` plus
the dagger footnote) — not a self-contained `\begin{table}` environment.
This lets you compose it with `group_size_curve.pdf` into a single
`figure` float with a shared caption (saves vertical space), or wrap
it in your own `\begin{table}` for a standalone presentation.

### Combined figure + table layout (recommended for tight page budgets)

```latex
\begin{figure}[t]
\centering
\begin{minipage}{0.58\columnwidth}
  \includegraphics[width=\linewidth]{human_baseline/group_size_curve.pdf}
\end{minipage}\hfill
\begin{minipage}{0.40\columnwidth}
  \scriptsize
  \input{human_baseline/human_baseline_table}
\end{minipage}
\caption{Human-panelist baseline on the within-category ranking task.
The panel-mean curve (left) crosses the within-block best-model line
at $k^*$; discrete panel sizes (right) compare the model against
panels of fixed size. (See the paper for the exact $k^*$, comparison
verdicts, and split-half reliability $\rho$.)}
\label{fig:human-baseline}
\end{figure}
```

`group_size_curve.pdf` is rendered at matplotlib `figsize=(4.0, 2.2)`
specifically tuned for the ~58% of column it occupies in this layout.

### Standalone table (alternative)

If you'd rather keep the table separate from the figure:

```latex
\begin{table}[t]
\centering\small
\caption{Human-panelist baseline for pairwise ranking accuracy.
(See the paper for the precise $k^*$, the bootstrap test verdict, and
split-half ranking reliability of the panel mean.)}
\label{tab:human-baseline}
\input{human_baseline/human_baseline_table}
\end{table}
```

The committed `results/group_size_curve.csv` (and the per-category and
split-half summary files) are sufficient to regenerate the figure
without NECTAR access. The `.tex` table is committed as a static
artifact; regenerating it requires NECTAR.

## Verifying the figure (no NECTAR required)

```bash
bash human_baseline/verify.sh
```

Reads pre-computed analysis CSVs from `results/` and re-renders the
figures.

## Full reanalysis (needs gated NECTAR data)

```bash
cd human_baseline
python human_panelist_baseline.py    # re-runs the panel analysis
python plot_human_baseline.py        # re-renders figures
```

The first command writes both the .tex table and the CSV inputs the
plot script consumes; running them in sequence regenerates everything
end-to-end.

## Layout

```
human_baseline/
├── human_panelist_baseline.py     Panel-mean analysis + table writer (uses NECTAR)
├── plot_human_baseline.py         Figure rendering (uses committed CSVs)
└── results/                      Analysis artifacts (committed; canonical input
                                   for plot_human_baseline.py)
    ├── summary.json               Pairwise accuracy / Spearman / R@1
    ├── group_size_curve.csv       Panel mean accuracy vs. group size
    ├── split_half_reliability.json
    ├── per_category_comparison.csv
    ├── individual_loo_accuracy.csv
    ├── inter_rater_reliability.csv
    └── pair_difficulty.csv
```

Rendered outputs (`human_baseline_table.tex`, `group_size_curve.pdf`)
land in [`../paper/human_baseline/`](../paper/human_baseline/), the
single location consumed by `\input{...}` from the paper source.
