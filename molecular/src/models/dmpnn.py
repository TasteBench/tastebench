"""Chemprop v2 D-MPNN wrapper with penultimate-layer embedding extraction.

API verified against chemprop==2.0.4.

Adaptation notes vs. original spec:
- ``model.predictor(z)`` returns shape (N, 1, 5) and is **already softmax-normalised**
  (chemprop's ``MulticlassClassificationFFN.forward`` ends with ``.softmax(-1)``),
  so ``predict_proba`` returns its output directly without a redundant softmax.
  ``model.predictor.ffn(z)`` returns shape (N, 5) raw logits if logits are needed.
- ``model.predictor.ffn`` is a chemprop.nn.ffn.MLP (a torch.nn.Sequential subclass).
  With n_layers=1 it has 2 blocks: the input projection (index 0) and the
  activation+dropout+final linear (index 1). The penultimate walk takes blocks
  [0:final_idx] which gives the post-input-linear representation before the
  final projection -- this is the (N, hidden_dim) vector used as the
  POM-methodology embedding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from chemprop import featurizers, nn as cnn
from chemprop.models import MPNN
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
try:
    from chemprop.nn.loss import CrossEntropyLoss as ChempropCELoss   # <=2.0.x
except ImportError:
    from chemprop.nn.metrics import CrossEntropyLoss as ChempropCELoss  # >=2.1.0


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


LABEL_ORDER = ("sweet", "bitter", "sour", "umami", "undefined")
N_CLASSES = len(LABEL_ORDER)


class _WeightedMulticlassCELoss(ChempropCELoss):
    """Chemprop-compatible CrossEntropyLoss with per-class weights."""

    def __init__(self, class_weights: torch.Tensor) -> None:
        super().__init__()
        # Register as a buffer so it moves with .to(device)
        self.register_buffer("class_weights", class_weights)

    def _calc_unreduced_loss(self, preds: torch.Tensor, targets: torch.Tensor, *args) -> torch.Tensor:
        # preds: (B, 1, n_classes) -> transpose to (B, n_classes, 1) for F.cross_entropy
        preds = preds.transpose(1, 2)
        targets = targets.long()
        # F.cross_entropy with weight expects class_weights on same device
        return F.cross_entropy(preds, targets, weight=self.class_weights, reduction="none")


def build_model(
    hidden_dim: int,
    depth: int,
    dropout: float,
    n_classes: int = N_CLASSES,
    init_lr: float = 1e-4,
    class_weights: Optional[np.ndarray] = None,
) -> MPNN:
    """Construct a Chemprop MPNN for multiclass classification."""
    mp = cnn.BondMessagePassing(d_h=hidden_dim, depth=depth, dropout=dropout)
    agg = cnn.MeanAggregation()
    predictor = cnn.MulticlassClassificationFFN(
        n_classes=n_classes,
        input_dim=hidden_dim,
        hidden_dim=hidden_dim,
        n_layers=1,
        dropout=dropout,
    )
    if class_weights is not None:
        w = torch.tensor(class_weights, dtype=torch.float32)
        predictor.criterion = _WeightedMulticlassCELoss(w)

    model = MPNN(
        message_passing=mp,
        agg=agg,
        predictor=predictor,
        batch_norm=True,
        init_lr=init_lr,
        max_lr=init_lr * 10,
        final_lr=init_lr,
    )
    return model


def featurize_smiles(smiles: list[str], labels: Optional[list[int]] = None) -> MoleculeDataset:
    """Build a Chemprop MoleculeDataset from SMILES + optional int labels."""
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    if labels is None:
        labels = [0] * len(smiles)
    pts = [
        MoleculeDatapoint.from_smi(smi, y=np.array([float(lbl)]))
        for smi, lbl in zip(smiles, labels)
    ]
    return MoleculeDataset(pts, feat)


def _safe_batch_size(n: int, batch_size: int) -> int:
    """Clamp batch_size so build_dataloader never drops the last molecule.

    chemprop's build_dataloader sets drop_last=True when
    ``len(dataset) % batch_size == 1``.  We avoid that by shrinking
    batch_size until the condition no longer holds.
    """
    bs = min(batch_size, n)
    while n > 0 and n % bs == 1:
        bs = max(1, bs - 1)
    return bs


def predict_proba(
    model: MPNN,
    smiles: list[str],
    batch_size: int = 64,
    device: str = "cpu",
) -> np.ndarray:
    """Return shape (N, n_classes) probability distributions (rows sum to 1).

    Chemprop's ``MulticlassClassificationFFN.forward`` already applies
    ``.softmax(-1)`` to its output, so we return ``model.predictor(z)``
    directly (squeezing the n_tasks=1 axis). Applying softmax a second time
    would squash confident predictions toward uniform (this used to be a bug
    that inflated ECE without affecting argmax-based metrics).
    """
    device = resolve_device(device)
    ds = featurize_smiles(smiles)
    dl = build_dataloader(ds, batch_size=_safe_batch_size(len(smiles), batch_size), num_workers=0, shuffle=False)
    model.eval()
    model.to(device)
    outs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in dl:
            bmg, *_ = batch
            bmg.to(device)  # in-place; BatchMolGraph.to() returns None
            z = model.fingerprint(bmg)
            probs = model.predictor(z)  # (N, 1, n_classes), already softmax
            probs = probs.squeeze(1)    # (N, n_classes)
            outs.append(probs.cpu().numpy())
    return np.concatenate(outs, axis=0)


def predict_embedding(
    model: MPNN,
    smiles: list[str],
    extraction_point: str = "penultimate",
    batch_size: int = 64,
    device: str = "cpu",
) -> np.ndarray:
    """Extract per-molecule embeddings.

    Parameters
    ----------
    extraction_point:
        'encoder'     — model.fingerprint(bmg), the post-aggregation encoder
                        output before any FFN layers.  Shape (N, hidden_dim).
        'penultimate' — output of predictor.ffn up to (but not including) the
                        final projection block.  Shape (N, hidden_dim).
                        This is the POM-methodology representation.
    """
    device = resolve_device(device)
    ds = featurize_smiles(smiles)
    dl = build_dataloader(ds, batch_size=_safe_batch_size(len(smiles), batch_size), num_workers=0, shuffle=False)
    model.eval()
    model.to(device)
    outs: list[np.ndarray] = []

    if extraction_point == "encoder":
        with torch.no_grad():
            for batch in dl:
                bmg, *_ = batch
                bmg.to(device)  # in-place; BatchMolGraph.to() returns None
                z = model.fingerprint(bmg)
                outs.append(z.cpu().numpy())
        return np.concatenate(outs, axis=0)

    if extraction_point != "penultimate":
        raise ValueError(f"unknown extraction_point: {extraction_point!r}")

    # Walk predictor FFN layers up to (but not including) the final block.
    # With n_layers=1, model.predictor.ffn has 2 blocks:
    #   [0] input linear, [1] act+dropout+final linear.
    # final_idx = 1 so we run only block 0, giving the (N, hidden_dim) vector.
    ffn = model.predictor.ffn
    final_idx = len(ffn) - 1
    with torch.no_grad():
        for batch in dl:
            bmg, *_ = batch
            bmg.to(device)  # in-place; BatchMolGraph.to() returns None
            z = model.fingerprint(bmg)
            x = z
            for layer in list(ffn)[:final_idx]:
                x = layer(x)
            outs.append(x.cpu().numpy())
    return np.concatenate(outs, axis=0)


def save_checkpoint(model: MPNN, dst: Path, config: dict) -> None:
    """Save model state dict and config to a .pt file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config}, dst)


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[MPNN, dict]:
    """Load a model saved with save_checkpoint, strict-matching the state_dict.

    If the checkpoint was saved with class-weighted CE (``_WeightedMulticlassCELoss``),
    the state_dict carries a ``predictor.criterion.class_weights`` buffer; we
    detect that here and rebuild the model with matching class weights so
    ``strict=True`` succeeds. Strict matching guarantees no parameter is
    silently dropped at load time.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    state = ckpt["state_dict"]

    cw_buf = state.get("predictor.criterion.class_weights")
    class_weights = cw_buf.detach().cpu().numpy() if cw_buf is not None else None

    model = build_model(
        hidden_dim=cfg["model"]["hidden_dim"],
        depth=cfg["model"]["depth"],
        dropout=cfg["model"]["dropout"],
        n_classes=cfg["model"].get("n_classes", N_CLASSES),
        init_lr=cfg["training"]["init_lr"],
        class_weights=class_weights,
    )
    model.load_state_dict(state, strict=True)
    return model, cfg
