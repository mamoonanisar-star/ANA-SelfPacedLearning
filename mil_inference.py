"""
mil_inference.py

Phase 5 – Demo Inference with Clinical Context.

Loads a saved ANAMILModel checkpoint, selects a random sample from the test
split (or a user-specified file path), and prints a clinical report showing:
  * Top predicted ICAP patterns with probabilities
  * Attention heatmap summary (which patches were most relevant per class)
  * Decision threshold and confidence guidance

Usage
-----
# Random test sample
python mil_inference.py \\
    --ann_file    ./data/features_index_english.csv \\
    --tokens_root ./data/clip_tokens/ \\
    --checkpoint  ./checkpoints/mil_best.pt

# Specific .npy file
python mil_inference.py \\
    --ann_file    ./data/features_index_english.csv \\
    --tokens_root ./data/clip_tokens/ \\
    --checkpoint  ./checkpoints/mil_best.pt \\
    --npy_file    ./data/clip_tokens/patient01/001.npy

# Top-k predictions only
python mil_inference.py \\
    --ann_file    ./data/features_index_english.csv \\
    --tokens_root ./data/clip_tokens/ \\
    --checkpoint  ./checkpoints/mil_best.pt \\
    --top_k 3
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch

from mil_dataset import ANAFeatureDataset, ICAP_CLASSES
from mil_model import ANAMILModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ICAP clinical descriptions
# ---------------------------------------------------------------------------

_ICAP_DESCRIPTIONS = {
    "ICAP_AC":  "Anti-Centromere Pattern – discrete speckles on mitotic chromosomes",
    "ICAP_CF":  "Coarse Fluorescent – large irregular nuclear speckles",
    "ICAP_DFS": "Dense Fine Speckled – fine uniform nuclear speckles",
    "ICAP_DC":  "Diffuse Cytoplasmic – cytoplasmic staining pattern",
    "ICAP_HS":  "Homogeneous Speckled – uniform nuclear staining",
    "ICAP_NMH": "Nuclear Membrane / Homogeneous – rim-like nuclear envelope staining",
    "ICAP_NUC": "Nucleolar – staining of nucleolar regions",
    "ICAP_OT":  "Other – non-specific or mixed pattern",
}


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------

def predict_single(
    tokens: torch.Tensor,
    model: ANAMILModel,
    device: torch.device,
    threshold: float = 0.5,
    class_names: Optional[List[str]] = None,
) -> Tuple[List[Tuple[str, float]], np.ndarray]:
    """
    Run inference on a single sample.

    Args:
        tokens     : Float tensor [num_tokens, token_dim].
        model      : Trained ANAMILModel in eval mode.
        device     : Torch device.
        threshold  : Decision threshold.
        class_names: List of class names.

    Returns:
        predictions : List of (class_name, probability) tuples for predicted
                      classes (probability >= threshold), sorted descending.
        attn_map    : NumPy array [num_classes, num_patches] – attention weights.
    """
    names = class_names or ICAP_CLASSES
    model.eval()

    with torch.no_grad():
        tokens_batch = tokens.unsqueeze(0).to(device)  # [1, num_tokens, token_dim]
        logits, attn_weights = model(tokens_batch)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()   # [num_classes]
        attn  = attn_weights.squeeze(0).cpu().numpy()            # [num_classes, num_patches]

    predictions = [
        (names[c], float(probs[c]))
        for c in range(len(names))
        if probs[c] >= threshold
    ]
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions, attn


def top_k_predictions(
    tokens: torch.Tensor,
    model: ANAMILModel,
    device: torch.device,
    k: int = 3,
    class_names: Optional[List[str]] = None,
) -> List[Tuple[str, float]]:
    """Return the top-k predicted classes regardless of threshold."""
    names = class_names or ICAP_CLASSES
    model.eval()

    with torch.no_grad():
        tokens_batch = tokens.unsqueeze(0).to(device)
        logits, _ = model(tokens_batch)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    ranked = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    return [(names[c], float(p)) for c, p in ranked[:k]]


# ---------------------------------------------------------------------------
# Clinical report printer
# ---------------------------------------------------------------------------

def print_clinical_report(
    sample_id: str,
    predictions: List[Tuple[str, float]],
    attn_map: np.ndarray,
    all_probs: np.ndarray,
    threshold: float,
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Print a human-readable clinical inference report.

    Args:
        sample_id   : Identifier string for the sample.
        predictions : List of (class_name, prob) for positive predictions.
        attn_map    : [num_classes, num_patches] attention weights.
        all_probs   : [num_classes] probabilities for all classes.
        threshold   : Decision threshold used.
        class_names : List of class names.
    """
    names = class_names or ICAP_CLASSES
    sep = "=" * 64

    print(f"\n{sep}")
    print(f"  ANA ICAP Pattern Inference Report")
    print(f"  Sample: {sample_id}")
    print(f"  Decision threshold: {threshold:.2f}")
    print(sep)

    if not predictions:
        print("  ⚠  No pattern predicted above threshold.")
        print(f"     Top probability: {all_probs.max():.3f} "
              f"({names[int(all_probs.argmax())]})")
    else:
        print(f"\n  Predicted ICAP Pattern(s) [{len(predictions)}]:\n")
        for name, prob in predictions:
            bar_len = int(prob * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            desc = _ICAP_DESCRIPTIONS.get(name, "")
            print(f"  {'✓':2s} {name:<12s}  {bar}  {prob:.3f}")
            if desc:
                print(f"     {desc}")
            # Most attended patches for this class
            c_idx = names.index(name) if name in names else -1
            if 0 <= c_idx < len(attn_map):
                top_patch_idxs = np.argsort(attn_map[c_idx])[::-1][:3] + 1  # +1: patch indices start at 1
                print(f"     Most relevant patches (1-indexed): {top_patch_idxs.tolist()}")
            print()

    print("\n  All class probabilities:")
    for i, (name, prob) in enumerate(zip(names, all_probs)):
        marker = "→" if prob >= threshold else " "
        print(f"  {marker} {name:<12s}  {prob:.3f}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Demo inference with clinical context output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ann_file",    required=True,
                   help="Path to CSV annotation file.")
    p.add_argument("--tokens_root", required=True,
                   help="Root directory of .npy token files.")
    p.add_argument("--checkpoint",  required=True,
                   help="Path to saved model checkpoint (.pt file).")
    p.add_argument("--npy_file",    default=None,
                   help="Specific .npy file to run inference on. "
                        "If omitted, a random test sample is used.")
    p.add_argument("--split",       default="test",
                   choices=["train", "val", "test"],
                   help="Split to sample from (when --npy_file is not given).")
    p.add_argument("--token_dim",   default=768,   type=int)
    p.add_argument("--num_tokens",  default=257,   type=int)
    p.add_argument("--num_classes", default=8,     type=int)
    p.add_argument("--proj_dim",    default=512,   type=int)
    p.add_argument("--mid_dim",     default=1024,  type=int)
    p.add_argument("--threshold",   default=0.5,   type=float)
    p.add_argument("--top_k",       default=None,  type=int,
                   help="Show top-k predictions regardless of threshold.")
    p.add_argument("--seed",        default=None,  type=int,
                   help="Random seed for reproducible sample selection.")
    p.add_argument("--device",      default=None,
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

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model = ANAMILModel(
        token_dim=args.token_dim,
        num_classes=args.num_classes,
        proj_dim=args.proj_dim,
        mid_dim=args.mid_dim,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    logger.info("Loaded checkpoint from %s", args.checkpoint)

    # ------------------------------------------------------------------
    # Load tokens
    # ------------------------------------------------------------------
    if args.npy_file:
        if not os.path.isfile(args.npy_file):
            raise FileNotFoundError(f"Token file not found: {args.npy_file!r}")
        arr = np.load(args.npy_file, allow_pickle=False)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        tokens = torch.from_numpy(arr.astype(np.float32))
        sample_id = os.path.basename(args.npy_file)
    else:
        if args.seed is not None:
            random.seed(args.seed)

        ds = ANAFeatureDataset(
            args.tokens_root, args.ann_file, args.split,
            num_tokens=args.num_tokens, token_dim=args.token_dim,
            num_classes=args.num_classes,
        )
        if len(ds) == 0:
            raise RuntimeError(f"No samples found in split={args.split!r}.")

        idx = random.randint(0, len(ds) - 1)
        tokens, label, _ = ds[idx]
        stem = ds.examples[idx][2]
        sample_id = f"{args.split}[{idx}] — {stem}"
        logger.info(
            "Selected random sample %d/%d: %s | GT labels: %s",
            idx, len(ds) - 1, stem,
            [ICAP_CLASSES[i] for i, v in enumerate(label.numpy()) if v > 0.5],
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    if args.top_k:
        top_preds = top_k_predictions(
            tokens, model, device, k=args.top_k, class_names=ICAP_CLASSES
        )
        print(f"\nTop-{args.top_k} predictions for {sample_id}:")
        for rank, (name, prob) in enumerate(top_preds, 1):
            print(f"  {rank}. {name:<12s}  {prob:.3f}  — {_ICAP_DESCRIPTIONS.get(name, '')}")
    else:
        predictions, attn_map = predict_single(
            tokens, model, device,
            threshold=args.threshold, class_names=ICAP_CLASSES,
        )
        # Collect all probabilities for the report
        with torch.no_grad():
            logits, _ = model(tokens.unsqueeze(0).to(device))
            all_probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        print_clinical_report(
            sample_id=sample_id,
            predictions=predictions,
            attn_map=attn_map,
            all_probs=all_probs,
            threshold=args.threshold,
            class_names=ICAP_CLASSES,
        )
