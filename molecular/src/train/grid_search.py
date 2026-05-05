"""Expand a grid YAML to N configs and run them (local mode)."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

logger = logging.getLogger(__name__)


def set_nested(d: dict, dotted_key: str, value: Any) -> None:
    """Set d[a][b][c] = value for dotted_key 'a.b.c'. Creates missing dicts."""
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def expand_grid(grid_path: Path) -> list[dict]:
    grid = yaml.safe_load(grid_path.read_text())
    base_path = Path(grid["base_config"])
    base = yaml.safe_load(base_path.read_text())

    sweep = grid["sweep"]
    keys = list(sweep.keys())
    values_per_key: list[list[Any]] = [sweep[k] for k in keys]

    configs: list[dict] = []
    for combo in itertools.product(*values_per_key):
        cfg = copy.deepcopy(base)
        for k, v in zip(keys, combo):
            set_nested(cfg, k, v)
        cfg["run_name"] = _hash_config(cfg)
        configs.append(cfg)
    return configs


def _hash_config(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return "run_" + hashlib.sha256(blob).hexdigest()[:10]


def run_local(configs: list[dict], results_dir: Path) -> None:
    for i, cfg in enumerate(configs):
        out = results_dir / cfg["run_name"]
        if (out / "val_metrics.json").exists():
            logger.info("[%d/%d] skip (already done): %s", i + 1, len(configs), cfg["run_name"])
            continue
        logger.info("[%d/%d] training: %s", i + 1, len(configs), cfg["run_name"])
        out.mkdir(parents=True, exist_ok=True)
        cfg_path = out / "config.in.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))
        subprocess.run(
            [sys.executable, "-m", "molecular.src.train.train_dmpnn",
             "--config", str(cfg_path), "--output_dir", str(out)],
            check=True,
        )


def aggregate(results_dir: Path) -> Path:
    rows = []
    for run_dir in sorted(results_dir.glob("run_*")):
        mpath = run_dir / "val_metrics.json"
        cpath = run_dir / "config.yaml"
        if not (mpath.exists() and cpath.exists()):
            continue
        m = json.loads(mpath.read_text())
        cfg = yaml.safe_load(cpath.read_text())
        rows.append({
            "run_name":        cfg["run_name"],
            "hidden_dim":      cfg["model"]["hidden_dim"],
            "depth":           cfg["model"]["depth"],
            "dropout":         cfg["model"]["dropout"],
            "init_lr":         cfg["training"]["init_lr"],
            "class_weighting": cfg["training"].get("class_weighting",
                                                   "inverse_frequency"),
            "val_macro_f1":    m["val_macro_f1"],
        })
    import pandas as pd
    df = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    out_csv = results_dir / "grid_summary.csv"
    df.to_csv(out_csv, index=False)
    if len(df):
        best_name = df.iloc[0]["run_name"]
        best_link = results_dir / "best"
        if best_link.is_symlink() or best_link.exists():
            best_link.unlink()
        best_link.symlink_to(best_name)
    return out_csv


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=False, type=Path,
                    help="Grid YAML path (required for --action run, not needed for aggregate)")
    ap.add_argument("--submit", choices=["local", "slurm", "list"], default="local")
    ap.add_argument("--results_dir", required=True, type=Path)
    ap.add_argument("--action", choices=["run", "aggregate"], default="run")
    args = ap.parse_args()

    if args.action == "aggregate":
        out = aggregate(args.results_dir)
        logger.info("summary: %s", out)
        return

    if args.grid is None:
        ap.error("--grid is required for --action run")
    configs = expand_grid(args.grid)
    logger.info("Grid expanded to %d configs", len(configs))

    if args.submit == "list":
        for c in configs:
            print(c["run_name"])
        return

    if args.submit == "local":
        run_local(configs, args.results_dir)
        aggregate(args.results_dir)
        return

    # SLURM path — see Task 8
    raise NotImplementedError("SLURM submission implemented in Task 8 (submit_grid_auto.sh).")


if __name__ == "__main__":
    main()
