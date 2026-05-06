"""
mil_train.py

Phase 3 – Sprint Training Loop for the MIL Pipeline.

Training strategy
-----------------
* Optimizer   : AdamW  (lr=1e-4, weight_decay=0.05)
* Loss        : BCEWithLogitsLoss with per-class pos_weight (handles class
                imbalance automatically).
* Grad clip   : max-norm 1.0 to prevent gradient explosions.
* Scheduler   : ReduceLROnPlateau (patience=5, factor=0.5).
* Early stop  : Triggered when F1-Macro on the validation set fails to
                improve for ``patience`` consecutive epochs.
* Checkpoints : Best checkpoint saved to disk based on F1-Macro.

Usage
-----
python mil_train.py \\
    --ann_file ./data/features_index_english.csv \\
    --tokens_root ./data/clip_tokens/ \\
    --epochs 50 --batch_size 16 --lr 1e-4 \\
    --weight_decay 0.05 --patience 10 \\
    --checkpoint_dir ./checkpoints/

Hyperparameter guidance
-----------------------
lr=1e-4      Fine-tuning range for AdamW on frozen or near-frozen features.
             Increase to 3e-4 if convergence is slow.
weight_decay=0.05  Standard AdamW decoupled weight decay; reduce to 0.01
             if under-fitting.
batch_size=16  Works on 12 GB GPU.  Increase to 32+ if memory allows.
patience=10  Recommended with ReduceLROnPlateau; set lower (5-7) if training
             is expensive.
Early stopping on F1-Macro is preferred for multi-label classification with
class imbalance.  If all classes have similar frequency, mAP is equally valid.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from mil_dataset import ANAFeatureDataset, ICAP_CLASSES
from mil_model import ANAMILModel
from mil_evaluate import evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mil_train")


# ---------------------------------------------------------------------------
# Early stopping helper
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Stops training when a monitored metric fails to improve.

    Args:
        patience    (int): Epochs without improvement before stopping.
        min_delta   (float): Minimum change to count as improvement.
        mode        (str): 'max' (higher is better, e.g. F1) or
                           'min' (lower is better, e.g. loss).
        checkpoint_path (str | None): If provided, saves best model here.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "max",
        checkpoint_path: Optional[str] = None,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.checkpoint_path = checkpoint_path
        self.best_score: Optional[float] = None
        self.counter: int = 0
        self.stop: bool = False

    def step(self, score: float, model: nn.Module) -> bool:
        """
        Call after each epoch.

        Returns:
            True  → training should stop.
            False → continue training.
        """
        improved = (
            self.best_score is None
            or (self.mode == "max" and score > self.best_score + self.min_delta)
            or (self.mode == "min" and score < self.best_score - self.min_delta)
        )
        if improved:
            self.best_score = score
            self.counter = 0
            if self.checkpoint_path:
                torch.save(model.state_dict(), self.checkpoint_path)
                logger.info("  ✓ Saved best checkpoint → %s", self.checkpoint_path)
        else:
            self.counter += 1
            logger.info(
                "  EarlyStopping: no improvement for %d / %d epochs.",
                self.counter,
                self.patience,
            )
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    ann_file: str,
    tokens_root: str,
    checkpoint_dir: str = "./checkpoints/",
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 0.05,
    patience: int = 10,
    num_workers: int = 4,
    token_dim: int = 768,
    num_tokens: int = 257,
    num_classes: int = 8,
    proj_dim: int = 512,
    mid_dim: int = 1024,
    attn_dropout: float = 0.1,
    proj_dropout: float = 0.2,
    grad_clip: float = 1.0,
    threshold: float = 0.5,
    device: Optional[str] = None,
) -> ANAMILModel:
    """
    Full training loop for ANAMILModel.

    Args:
        ann_file       : Path to CSV annotation file.
        tokens_root    : Root directory of .npy token files.
        checkpoint_dir : Directory for saving best checkpoint.
        epochs         : Maximum training epochs.
        batch_size     : Samples per mini-batch.
        lr             : Learning rate for AdamW.
        weight_decay   : Weight decay for AdamW.
        patience       : Early-stopping patience (epochs).
        num_workers    : DataLoader worker processes.
        token_dim      : Token embedding dimension (768 for ViT-L/14).
        num_tokens     : Tokens per sample (257 = 1 CLS + 256 patches).
        num_classes    : Number of output classes (8 ICAP classes).
        proj_dim       : Projection head output dimension.
        mid_dim        : Projection head intermediate dimension.
        attn_dropout   : Dropout in ClassWiseMaxPoolingHead.
        proj_dropout   : Dropout in MedicalProjectionHead.
        grad_clip      : Max gradient norm for clipping.
        threshold      : Decision threshold for F1 computation.
        device         : 'cuda', 'cpu', or None (auto-detect).

    Returns:
        Trained ANAMILModel.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, "mil_best.pt")

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    logger.info("Using device: %s", dev)

    # ------------------------------------------------------------------
    # Datasets and DataLoaders
    # ------------------------------------------------------------------
    train_ds = ANAFeatureDataset(
        tokens_root, ann_file, "train",
        num_tokens=num_tokens, token_dim=token_dim, num_classes=num_classes,
    )
    val_ds = ANAFeatureDataset(
        tokens_root, ann_file, "val",
        num_tokens=num_tokens, token_dim=token_dim, num_classes=num_classes,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(dev.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(dev.type == "cuda"),
    )

    # ------------------------------------------------------------------
    # Class imbalance → pos_weight
    # ------------------------------------------------------------------
    pos_weight = train_ds.compute_pos_weight().to(dev)
    logger.info(
        "Class counts (train): %s", train_ds.class_counts.tolist()
    )
    logger.info(
        "pos_weight: %s", [f"{w:.2f}" for w in pos_weight.cpu().tolist()]
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = ANAMILModel(
        token_dim=token_dim,
        num_classes=num_classes,
        proj_dim=proj_dim,
        mid_dim=mid_dim,
        attn_dropout=attn_dropout,
        proj_dropout=proj_dropout,
    ).to(dev)
    logger.info("Model parameters: {:,}".format(sum(p.numel() for p in model.parameters())))

    # ------------------------------------------------------------------
    # Loss, optimiser, scheduler
    # ------------------------------------------------------------------
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    early_stopper = EarlyStopping(
        patience=patience, mode="max", checkpoint_path=ckpt_path
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    logger.info(
        "Starting training: %d epochs | batch=%d | lr=%.1e | wd=%.3f",
        epochs, batch_size, lr, weight_decay,
    )

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0

        for tokens, labels, _ in train_loader:
            tokens = tokens.to(dev, non_blocking=True)
            labels = labels.to(dev, non_blocking=True)

            optimizer.zero_grad()
            logits, _ = model(tokens)
            loss = criterion(logits, labels)
            loss.backward()

            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        metrics = evaluate(
            model, val_loader, dev,
            num_classes=num_classes, threshold=threshold,
            class_names=ICAP_CLASSES,
        )
        f1_macro = metrics["f1_macro"]
        mAP = metrics["mAP"]

        scheduler.step(f1_macro)

        elapsed = time.time() - t0
        logger.info(
            "Epoch %3d/%d | loss=%.4f | val F1-Macro=%.4f | val mAP=%.4f | "
            "lr=%.2e | %.1fs",
            epoch, epochs, avg_loss, f1_macro, mAP,
            optimizer.param_groups[0]["lr"], elapsed,
        )

        if early_stopper.step(f1_macro, model):
            logger.info("Early stopping triggered at epoch %d.", epoch)
            break

    # ------------------------------------------------------------------
    # Load best checkpoint
    # ------------------------------------------------------------------
    if os.path.isfile(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=dev))
        logger.info("Loaded best model from %s (F1-Macro=%.4f).", ckpt_path, early_stopper.best_score)

    return model


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train ANAMILModel on CLIP token embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ann_file",       required=True,
                   help="Path to CSV annotation file.")
    p.add_argument("--tokens_root",    required=True,
                   help="Root directory of pre-computed .npy token files.")
    p.add_argument("--checkpoint_dir", default="./checkpoints/",
                   help="Directory for saving best checkpoint.")
    p.add_argument("--epochs",         default=50,   type=int)
    p.add_argument("--batch_size",     default=16,   type=int)
    p.add_argument("--lr",             default=1e-4, type=float)
    p.add_argument("--weight_decay",   default=0.05, type=float)
    p.add_argument("--patience",       default=10,   type=int)
    p.add_argument("--num_workers",    default=4,    type=int)
    p.add_argument("--token_dim",      default=768,  type=int)
    p.add_argument("--num_tokens",     default=257,  type=int)
    p.add_argument("--num_classes",    default=8,    type=int)
    p.add_argument("--proj_dim",       default=512,  type=int)
    p.add_argument("--mid_dim",        default=1024, type=int)
    p.add_argument("--attn_dropout",   default=0.1,  type=float)
    p.add_argument("--proj_dropout",   default=0.2,  type=float)
    p.add_argument("--grad_clip",      default=1.0,  type=float)
    p.add_argument("--threshold",      default=0.5,  type=float)
    p.add_argument("--device",         default=None,
                   help="'cuda' or 'cpu' (default: auto-detect).")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    train(
        ann_file=args.ann_file,
        tokens_root=args.tokens_root,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        num_workers=args.num_workers,
        token_dim=args.token_dim,
        num_tokens=args.num_tokens,
        num_classes=args.num_classes,
        proj_dim=args.proj_dim,
        mid_dim=args.mid_dim,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
        grad_clip=args.grad_clip,
        threshold=args.threshold,
        device=args.device,
    )
