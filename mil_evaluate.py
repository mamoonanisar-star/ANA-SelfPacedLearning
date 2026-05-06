"""
mil_evaluate.py

Phase 4 – High-Rigor Evaluation for the MIL Pipeline.

Metrics computed
----------------
* F1-Macro        : Unweighted average F1 across all 8 classes.
* mAP             : Mean Average Precision (ranking-aware, threshold-free).
* Per-class F1    : F1, precision, recall for every ICAP class.
* Confusion matrix: Per-class binary confusion matrices.
* Threshold tuning: Searches the threshold that maximises F1-Macro on the
                    validation set (useful when class balance is extreme).

Usage
-----
# Evaluate a saved checkpoint
python mil_evaluate.py \\
    --ann_file ./data/features_index_english.csv \\
    --tokens_root ./data/clip_tokens/ \\
    --checkpoint ./checkpoints/mil_best.pt \\
    --split test \\
    --tune_threshold
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
    )
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

from mil_dataset import ANAFeatureDataset, ICAP_CLASSES
from mil_model import ANAMILModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 8,
    threshold: float = 0.5,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Run inference over ``loader`` and compute evaluation metrics.

    Args:
        model       : ANAMILModel (or any model returning ``(logits, ...)``)
                      in eval mode.
        loader      : DataLoader yielding ``(tokens, labels, indices)``.
        device      : Torch device.
        num_classes : Number of output classes.
        threshold   : Decision threshold for converting probabilities to
                      binary predictions (default 0.5).
        class_names : Optional list of class names for logging.

    Returns:
        Dictionary with keys:
          f1_macro, mAP, f1_per_class, precision_per_class, recall_per_class,
          all_probs (np.ndarray), all_labels (np.ndarray).
    """
    if not _SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for evaluation. "
            "Install with: pip install scikit-learn"
        )

    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            tokens, labels, _ = batch
            tokens = tokens.to(device, non_blocking=True)
            logits, _ = model(tokens)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    all_probs  = np.concatenate(all_probs,  axis=0)  # [N, C]
    all_labels = np.concatenate(all_labels, axis=0)  # [N, C]

    preds = (all_probs >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Global metrics
    # ------------------------------------------------------------------
    f1_macro = f1_score(all_labels, preds, average="macro", zero_division=0)

    # mAP: average of per-class AP; handles empty classes gracefully
    if all_labels.max() > 0:
        mAP = average_precision_score(all_labels, all_probs, average="macro")
    else:
        mAP = 0.0
        logger.warning("All labels are zero; mAP set to 0.")

    # ------------------------------------------------------------------
    # Per-class metrics
    # ------------------------------------------------------------------
    f1_per   = f1_score(all_labels, preds, average=None, zero_division=0)
    prec_per = precision_score(all_labels, preds, average=None, zero_division=0)
    rec_per  = recall_score(all_labels, preds, average=None, zero_division=0)

    names = class_names or [f"Class_{c}" for c in range(num_classes)]
    _log_per_class(names, f1_per, prec_per, rec_per)

    return {
        "f1_macro":          float(f1_macro),
        "mAP":               float(mAP),
        "f1_per_class":      f1_per.tolist(),
        "precision_per_class": prec_per.tolist(),
        "recall_per_class":  rec_per.tolist(),
        "all_probs":         all_probs,
        "all_labels":        all_labels,
    }


def _log_per_class(
    names: List[str],
    f1: np.ndarray,
    prec: np.ndarray,
    rec: np.ndarray,
) -> None:
    logger.info("%-20s  %6s  %6s  %6s", "Class", "Prec", "Rec", "F1")
    logger.info("-" * 46)
    for name, p, r, f in zip(names, prec, rec, f1):
        logger.info("%-20s  %6.3f  %6.3f  %6.3f", name, p, r, f)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def compute_confusion_matrices(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    threshold: float = 0.5,
    class_names: Optional[List[str]] = None,
) -> List[np.ndarray]:
    """
    Compute binary confusion matrix for each class.

    Returns:
        List of [2, 2] NumPy arrays (one per class).
        Rows = actual (neg, pos), Columns = predicted (neg, pos).
    """
    preds = (all_probs >= threshold).astype(int)
    num_classes = all_labels.shape[1]
    names = class_names or [f"Class_{c}" for c in range(num_classes)]
    matrices = []

    for c, name in enumerate(names):
        tp = int(((preds[:, c] == 1) & (all_labels[:, c] == 1)).sum())
        fp = int(((preds[:, c] == 1) & (all_labels[:, c] == 0)).sum())
        tn = int(((preds[:, c] == 0) & (all_labels[:, c] == 0)).sum())
        fn = int(((preds[:, c] == 0) & (all_labels[:, c] == 1)).sum())
        cm = np.array([[tn, fp], [fn, tp]])
        matrices.append(cm)
        logger.info(
            "Confusion [%s]: TN=%d FP=%d FN=%d TP=%d",
            name, tn, fp, fn, tp,
        )

    return matrices


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def tune_threshold(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    candidates: Optional[np.ndarray] = None,
) -> float:
    """
    Find the decision threshold that maximises F1-Macro.

    Evaluates each candidate threshold on the provided predictions and labels.
    Intended to be run on the **validation** set; apply the tuned threshold
    on the test set.

    Args:
        all_labels  : [N, C] binary ground-truth.
        all_probs   : [N, C] predicted probabilities.
        candidates  : Array of thresholds to search (default: 0.1–0.9 in
                      steps of 0.05).

    Returns:
        best_threshold (float) that maximises F1-Macro.
    """
    if candidates is None:
        candidates = np.arange(0.10, 0.91, 0.05)

    best_t  = 0.5
    best_f1 = -1.0

    for t in candidates:
        preds = (all_probs >= t).astype(int)
        f1 = f1_score(all_labels, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t  = float(t)

    logger.info(
        "Threshold tuning: best threshold=%.2f → F1-Macro=%.4f",
        best_t, best_f1,
    )
    return best_t


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a saved ANAMILModel checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ann_file",    required=True,
                   help="Path to CSV annotation file.")
    p.add_argument("--tokens_root", required=True,
                   help="Root directory of .npy token files.")
    p.add_argument("--checkpoint",  required=True,
                   help="Path to saved model checkpoint (.pt file).")
    p.add_argument("--split",       default="test",
                   choices=["train", "val", "test"],
                   help="Dataset split to evaluate.")
    p.add_argument("--batch_size",  default=32,    type=int)
    p.add_argument("--num_workers", default=4,     type=int)
    p.add_argument("--token_dim",   default=768,   type=int)
    p.add_argument("--num_tokens",  default=257,   type=int)
    p.add_argument("--num_classes", default=8,     type=int)
    p.add_argument("--proj_dim",    default=512,   type=int)
    p.add_argument("--mid_dim",     default=1024,  type=int)
    p.add_argument("--threshold",   default=0.5,   type=float)
    p.add_argument("--tune_threshold", action="store_true",
                   help="Search for the optimal threshold on the val set first.")
    p.add_argument("--device", default=None,
                   help="'cuda' or 'cpu' (default: auto-detect).")
    return p


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _build_parser().parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint!r}")

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = ANAMILModel(
        token_dim=args.token_dim,
        num_classes=args.num_classes,
        proj_dim=args.proj_dim,
        mid_dim=args.mid_dim,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    logger.info("Loaded checkpoint from %s", args.checkpoint)

    threshold = args.threshold

    # Optional: tune threshold on validation set
    if args.tune_threshold:
        val_ds = ANAFeatureDataset(
            args.tokens_root, args.ann_file, "val",
            num_tokens=args.num_tokens, token_dim=args.token_dim,
            num_classes=args.num_classes,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers,
        )
        val_metrics = evaluate(
            model, val_loader, device,
            num_classes=args.num_classes, threshold=0.5,
        )
        threshold = tune_threshold(
            val_metrics["all_labels"], val_metrics["all_probs"]
        )

    # Evaluate on the target split
    ds = ANAFeatureDataset(
        args.tokens_root, args.ann_file, args.split,
        num_tokens=args.num_tokens, token_dim=args.token_dim,
        num_classes=args.num_classes,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )
    metrics = evaluate(
        model, loader, device,
        num_classes=args.num_classes, threshold=threshold,
        class_names=ICAP_CLASSES,
    )
    compute_confusion_matrices(
        metrics["all_labels"], metrics["all_probs"],
        threshold=threshold, class_names=ICAP_CLASSES,
    )

    logger.info(
        "\n=== %s Results ===\n  F1-Macro : %.4f\n  mAP      : %.4f\n"
        "  Threshold: %.2f",
        args.split.upper(),
        metrics["f1_macro"],
        metrics["mAP"],
        threshold,
    )
