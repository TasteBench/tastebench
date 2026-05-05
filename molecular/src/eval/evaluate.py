"""Evaluate a trained D-MPNN or the FART Augmented baseline on a CSV test split.

Produces under ``output_dir``:
  metrics.json, per_class_metrics.csv, predictions.parquet,
  confusion_matrix.png, roc_curves.png, reliability_diagram.png.

Two model types
---------------
dmpnn_ckpt      -- our D-MPNN, requires ``--ckpt path/to/ckpt.pt``.
fart_augmented  -- the FartLabs/FART_Augmented HuggingFace checkpoint
                   (RoBERTa sequence classifier over SMILES). This is
                   the row reported in Table~\\ref{tab:molecular-prediction}.

Both paths produce a single-checkpoint, single-pass prediction (no TTA),
so the molecular-task comparison in the paper stays apples-to-apples.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve

from molecular.src.data.dataset import FartDataset, LABEL_ORDER
from molecular.src.eval.metrics import compute_metrics

logger = logging.getLogger(__name__)

# FartLabs/FART_Augmented = the paper's final SMILES-augmented checkpoint
# (Moon et al., 2024). Published 2024-12-04, ~37 HF downloads at time of writing.
FART_AUGMENTED_HF_ID = "FartLabs/FART_Augmented"


def _plot_confusion(conf: np.ndarray, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        conf, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER, ax=ax,
    )
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def _plot_roc(y_true: np.ndarray, probs: np.ndarray, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, name in enumerate(LABEL_ORDER):
        if not (y_true == i).any():
            continue
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), probs[:, i])
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("Per-class ROC (OvR)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def _plot_reliability(y_true: np.ndarray, probs: np.ndarray, out_png: Path, n_bins: int = 10) -> None:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        xs.append(conf[m].mean())
        ys.append(correct[m].mean())
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
    ax.plot(xs, ys, "o-", label="observed")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy"); ax.set_title("Reliability diagram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def _pick_device(desired: str) -> str:
    import torch
    if desired != "auto":
        return desired
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _write_outputs(
    y_true: np.ndarray,
    probs: np.ndarray,
    smiles: list[str],
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    m = compute_metrics(y_true, probs)

    (output_dir / "metrics.json").write_text(json.dumps(m, indent=2))
    pd.DataFrame(m["per_class"]).to_csv(output_dir / "per_class_metrics.csv", index=False)

    y_pred = probs.argmax(axis=1)
    preds_df = pd.DataFrame({"smiles": smiles, "y_true": y_true, "y_pred": y_pred})
    for i, name in enumerate(LABEL_ORDER):
        preds_df[f"prob_{name}"] = probs[:, i]
    preds_df.to_parquet(output_dir / "predictions.parquet", index=False)

    _plot_confusion(np.array(m["confusion_matrix"]), output_dir / "confusion_matrix.png")
    _plot_roc(y_true, probs, output_dir / "roc_curves.png")
    _plot_reliability(y_true, probs, output_dir / "reliability_diagram.png")

    logger.info("accuracy=%.4f macro_f1=%.4f ece=%.4f", m["accuracy"], m["macro_f1"], m["ece"])
    return m


def run_from_ckpt(ckpt_path: Path, test_csv: Path, output_dir: Path,
                  device: str = "auto") -> dict:
    """Evaluate a trained D-MPNN checkpoint on test_csv and write all outputs."""
    from molecular.src.models.dmpnn import load_checkpoint, predict_proba

    device = _pick_device(device)
    model, cfg = load_checkpoint(ckpt_path, device=device)
    ds = FartDataset(test_csv)
    batch_size = cfg.get("training", {}).get("batch_size", 50)
    probs = predict_proba(model, ds.smiles, batch_size=batch_size, device=device)
    return _write_outputs(np.array(ds.labels), probs, ds.smiles, output_dir)


def run_from_fart_augmented(test_csv: Path, output_dir: Path,
                            device: str = "auto", batch_size: int = 32) -> dict:
    """Evaluate FartLabs/FART_Augmented on test_csv and write all outputs.

    The FART model is a RoBERTa sequence classifier whose ``id2label`` is
    {0:'Bitter', 1:'Sour', 2:'Sweet', 3:'Umami', 4:'Undefined'}; we reorder
    the probability columns so that column ``i`` corresponds to
    ``LABEL_ORDER[i]`` before computing metrics.
    """
    import torch
    import torch.nn.functional as F
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as exc:
        raise RuntimeError(
            "transformers library not found. pip install transformers."
        ) from exc

    torch_device = torch.device(_pick_device(device))
    logger.info("Loading FART model from HuggingFace: %s", FART_AUGMENTED_HF_ID)
    tokenizer = AutoTokenizer.from_pretrained(FART_AUGMENTED_HF_ID)
    hf_model = AutoModelForSequenceClassification.from_pretrained(FART_AUGMENTED_HF_ID)
    hf_model.eval()
    hf_model.to(torch_device)

    fart_id2label = {int(k): v.lower() for k, v in hf_model.config.id2label.items()}
    try:
        reindex = np.array([LABEL_ORDER.index(fart_id2label[j]) for j in range(len(LABEL_ORDER))])
    except (KeyError, ValueError) as exc:
        raise RuntimeError(
            f"FART model label not in LABEL_ORDER {LABEL_ORDER}: {exc}"
        ) from exc
    logger.info("FART label reindex (fart_col -> our_col): %s", list(reindex))

    ds = FartDataset(test_csv)
    smiles_list = ds.smiles
    y_true = np.array(ds.labels)

    all_probs: list[np.ndarray] = []
    logger.info("Inference on %d molecules (batch_size=%d)", len(smiles_list), batch_size)
    for start in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[start: start + batch_size]
        inputs = tokenizer(batch_smiles, return_tensors="pt", padding=True,
                           truncation=True, max_length=512)
        inputs = {k: v.to(torch_device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = hf_model(**inputs).logits  # (B, 5) in FART order
        probs_fart = F.softmax(logits, dim=-1).cpu().numpy()
        probs_ours = np.empty_like(probs_fart)
        for fart_col, our_col in enumerate(reindex):
            probs_ours[:, our_col] = probs_fart[:, fart_col]
        all_probs.append(probs_ours)

    probs = np.concatenate(all_probs, axis=0)
    return _write_outputs(y_true, probs, smiles_list, output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(
        description="Evaluate a taste classifier on a FART CSV test split."
    )
    ap.add_argument(
        "--model_type",
        choices=["dmpnn_ckpt", "fart_augmented"],
        default="dmpnn_ckpt",
        help="Which model to evaluate. 'dmpnn_ckpt' requires --ckpt; "
             "'fart_augmented' loads FartLabs/FART_Augmented from HuggingFace.",
    )
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="Path to ckpt.pt (required when --model_type dmpnn_ckpt).")
    ap.add_argument("--test_csv",   type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.model_type == "dmpnn_ckpt":
        if args.ckpt is None:
            ap.error("--ckpt is required when --model_type is dmpnn_ckpt")
        run_from_ckpt(args.ckpt, args.test_csv, args.output_dir, args.device)
    else:
        run_from_fart_augmented(args.test_csv, args.output_dir, args.device)


if __name__ == "__main__":
    main()
