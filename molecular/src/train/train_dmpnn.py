"""Single-config Chemprop D-MPNN training CLI.

Loads a YAML config, trains, saves val-best checkpoint + metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score

from chemprop.data import build_dataloader
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from molecular.src.data.dataset import FartDataset
from molecular.src.models.dmpnn import (
    build_model,
    featurize_smiles,
    predict_proba,
    save_checkpoint,
    _safe_batch_size,
)

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def pick_device(desired: str) -> str:
    if desired != "auto":
        return desired
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    seed_everything(config["seed"])
    device = pick_device(config["training"].get("device", "auto"))
    logger.info("Using device: %s", device)

    # Data
    train_ds = FartDataset(Path(config["data"]["train_csv"]))
    val_ds   = FartDataset(Path(config["data"]["val_csv"]))

    train_mol = featurize_smiles(train_ds.smiles, train_ds.labels)
    val_mol   = featurize_smiles(val_ds.smiles,   val_ds.labels)

    batch_size = config["training"]["batch_size"]
    train_dl = build_dataloader(
        train_mol,
        batch_size=_safe_batch_size(len(train_ds), batch_size),
        num_workers=0,
        shuffle=True,
    )
    val_dl = build_dataloader(
        val_mol,
        batch_size=_safe_batch_size(len(val_ds), batch_size),
        num_workers=0,
        shuffle=False,
    )

    # Model
    cw_strategy = config["training"].get("class_weighting", "inverse_frequency")
    logger.info("class_weighting strategy: %s", cw_strategy)
    model = build_model(
        hidden_dim=config["model"]["hidden_dim"],
        depth=config["model"]["depth"],
        dropout=config["model"]["dropout"],
        n_classes=config["model"]["n_classes"],
        init_lr=config["training"]["init_lr"],
        class_weights=train_ds.class_weights(cw_strategy),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="ckpt-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    early_cb = EarlyStopping(
        monitor="val_loss", patience=config["training"]["patience"], mode="min", verbose=True
    )
    trainer = Trainer(
        max_epochs=config["training"]["max_epochs"],
        callbacks=[ckpt_cb, early_cb],
        accelerator="auto",
        devices=1,
        logger=False,
        enable_progress_bar=True,
        deterministic=True,
    )
    trainer.fit(model, train_dl, val_dl)

    # Load best weights back
    if ckpt_cb.best_model_path:
        best = torch.load(ckpt_cb.best_model_path, map_location=device, weights_only=False)
        state = best.get("state_dict", best)
        model.load_state_dict(state)

    # Final eval on val
    val_probs = predict_proba(model, val_ds.smiles, batch_size=batch_size, device=device)
    val_preds = val_probs.argmax(axis=-1)
    val_labels = np.array(val_ds.labels)
    val_macro_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)

    metrics = {
        "val_macro_f1": float(val_macro_f1),
        "best_ckpt_path": ckpt_cb.best_model_path,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
    }

    save_checkpoint(model, output_dir / "ckpt.pt", config)
    (output_dir / "val_metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config))

    logger.info("val_macro_f1=%.4f", val_macro_f1)
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    args = ap.parse_args()
    with args.config.open() as f:
        config = yaml.safe_load(f)
    train(config, args.output_dir)


if __name__ == "__main__":
    main()
