"""Compile-check all final tables + figures.

Wraps each table fragment in a minimal NeurIPS-style standalone document
and compiles with tectonic. Reports pass/fail + first error per artifact.

Tables live in two parallel directories so the molecular-prediction work
(taste_gnn / FART encoder comparison, grid search, probes) can be browsed
on its own:

  results/tables/                       -- main NECTAR ranking tables
  molecular/results/tables/                  -- compound-encoder comparison tables

Figures correspondingly:

  results/figures/                       -- main pipeline figures
  molecular/results/figures/                 -- compound-encoder comparison figures
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

FOOD_SIM_DIR = Path(__file__).resolve().parent.parent
PAPER_DIR = FOOD_SIM_DIR.parent / "paper"
TABLES_DIR = PAPER_DIR / "model_results_tables"
MP_TABLES_DIR = PAPER_DIR / "molecular_prediction"
HB_DIR = PAPER_DIR / "human_baseline"

# Each entry: (filename, dir-this-table-lives-in)
TABLES = [
    ("table_results.tex",                       TABLES_DIR),
    ("table_ablation_features.tex",             TABLES_DIR),
    ("table_ablation_llm.tex",                  TABLES_DIR),
    ("table_per_category.tex",                  TABLES_DIR),
    ("table_per_category_nnls.tex",             TABLES_DIR),
    ("table_per_model_nnls.tex",                TABLES_DIR),
    ("table_molecular_prediction.tex",          MP_TABLES_DIR),
    ("table_molecular_per_class.tex",           MP_TABLES_DIR),
    ("table_gnn_per_model.tex",                 MP_TABLES_DIR),
    ("table_gnn_grid.tex",                      MP_TABLES_DIR),
]

FIGURES = [
    ("group_size_curve.pdf",     HB_DIR),
]

WRAPPER = r"""\documentclass[11pt]{article}
\usepackage[margin=0.75in]{geometry}
\usepackage{booktabs}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{float}
\usepackage{array}
\usepackage{makecell}
\usepackage{url}
%% Stub citation commands so tables that \citet{...} compile standalone.
%% The paper preamble loads natbib; this wrapper isn't the paper.
\providecommand{\citet}[1]{[#1]}
\providecommand{\citep}[1]{[#1]}
\begin{document}
\input{%s}
\end{document}
"""


def compile_one(tex_path: Path) -> tuple:
    with tempfile.TemporaryDirectory() as td:
        wrapper = Path(td) / "main.tex"
        wrapper.write_text(WRAPPER % tex_path.resolve())
        result = subprocess.run(
            ["tectonic", "-X", "compile", "-o", td, str(wrapper)],
            capture_output=True, text=True, timeout=60,
        )
        ok = result.returncode == 0
        pdf = Path(td) / "main.pdf"
        pdf_size = pdf.stat().st_size if pdf.exists() else 0
        return ok, pdf_size, result.stderr


def check_pdf(path: Path) -> tuple:
    if not path.exists():
        return False, 0
    head = path.open("rb").read(4)
    if head[:4] != b"%PDF":
        return False, path.stat().st_size
    return True, path.stat().st_size


def main() -> int:
    failures = []
    for name, dir_ in TABLES:
        path = dir_ / name
        rel = path.relative_to(FOOD_SIM_DIR.parent)
        if not path.exists():
            print(f"  MISSING  {rel}")
            failures.append(name); continue
        ok, size, err = compile_one(path)
        marker = "OK " if ok else "FAIL"
        print(f"  [{marker}]  {name:<40}  {size:>8,} bytes pdf  ({rel.parent})")
        if not ok:
            for line in err.split("\n"):
                if "error" in line.lower() or "! " in line:
                    print(f"          ↳ {line.strip()[:140]}"); break
            failures.append(name)

    for name, dir_ in FIGURES:
        path = dir_ / name
        rel = path.relative_to(FOOD_SIM_DIR.parent)
        ok, size = check_pdf(path)
        marker = "OK " if ok else "FAIL"
        if not ok and size == 0:
            marker = "MISSING"
            print(f"  {marker}  {rel}"); failures.append(name); continue
        print(f"  [{marker}]  {name:<40}  {size:>8,} bytes pdf  ({rel.parent})")
        if not ok: failures.append(name)

    total = len(TABLES) + len(FIGURES)
    if failures:
        print(f"\n{len(failures)} artifact(s) failed: {failures}")
        return 1
    print(f"\nAll {total} artifacts compile/validate cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
