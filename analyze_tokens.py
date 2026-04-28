"""
analyze_tokens.py

Visualization and analysis of learned token attention / patch suppression
for ANA token-based classification.

Capabilities:
  1. Patch suppression map   – which patches get suppressed (low SPL weight)
  2. Token importance heatmap – which of the 257 tokens drive each decision
  3. CLS vs spatial token comparison
  4. Quantitative suppression metrics
  5. Benchmark comparison table: Token vs ResNet-50

Usage
-----
# Visualize attention & suppression for a single image group
python analyze_tokens.py \\
    --model_path ./token_run/best_f1ma.pt \\
    --tokens_root ./data/clip_tokens/ \\
    --annFile all_single_small_82240_19400_19330.csv \\
    --split val \\
    --out_dir ./analysis_output/

# Print suppression statistics only (no plots)
python analyze_tokens.py \\
    --model_path ./token_run/best_f1ma.pt \\
    --tokens_root ./data/clip_tokens/ \\
    --annFile all_single_small_82240_19400_19330.csv \\
    --stats_only
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Optional visualisation dependencies
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    PLT_AVAILABLE = True
except ImportError:
    PLT_AVAILABLE = False

try:
    import seaborn as sns
    SNS_AVAILABLE = True
except ImportError:
    SNS_AVAILABLE = False

from config import NUM_CATEGORIES, GPU_IN_USE
from models_with_tokens import ANATokenClassifier
from data_clip_tokens import ANATokenDataset

ANA_CLASS_NAMES = [
    "Homogeneous",
    "Speckled",
    "Nucleolar",
    "Centromere",
    "Nuclear Membrane",
    "Cytoplasmic",
    "Golgi",
    "Negative",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(model_path: str, num_tokens: int, token_dim: int, device) -> ANATokenClassifier:
    checkpoint = torch.load(model_path, map_location=device)
    # Infer architecture from checkpoint keys
    hidden_dim = 256
    model = ANATokenClassifier(
        token_dim=token_dim,
        num_tokens=num_tokens,
        num_classes=NUM_CATEGORIES,
        hidden_dim=hidden_dim,
    )
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    return model


def _group_patches_by_image(dataset: ANATokenDataset) -> dict:
    """
    Return mapping: base_image_name → list of dataset indices.

    The base name is derived by stripping a trailing numeric suffix separated
    by an underscore from the patch stem (e.g. "img_0003" → "img").
    If no such suffix is found the full stem is used as the key, so all
    patches with the same stem are grouped together.

    Expected naming convention: ``<base_image>_<patch_index>`` where
    ``<patch_index>`` is purely numeric.  Other naming patterns fall back
    to using the full stem as the key (each patch in its own group).
    """
    groups: dict = {}
    for idx, (_, _, stem) in enumerate(dataset.examples):
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            base = parts[0]
        else:
            base = stem
        groups.setdefault(base, []).append(idx)
    return groups


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

def compute_patch_suppression(
    dataset: ANATokenDataset,
    spl_weights: torch.Tensor,
    out_dir: str,
    top_n: int = 5,
):
    """
    Compute and visualise which patches are most suppressed (lowest SPL weight).

    Args:
        dataset    : ANATokenDataset instance.
        spl_weights: 1-D float tensor of length len(dataset).
        out_dir    : Directory to save plots.
        top_n      : Number of most/least suppressed patches to report.
    """
    os.makedirs(out_dir, exist_ok=True)
    w = spl_weights.float().cpu().numpy()

    print("\n=== Patch Suppression Statistics ===")
    print(f"  Total patches : {len(w)}")
    print(f"  Mean weight   : {w.mean():.4f}")
    print(f"  Std  weight   : {w.std():.4f}")
    print(f"  Min  weight   : {w.min():.4f}")
    print(f"  Max  weight   : {w.max():.4f}")
    print(f"  Suppressed (<0.1): {(w < 0.1).sum()} ({100*(w<0.1).mean():.1f}%)")

    if not PLT_AVAILABLE:
        warnings.warn("matplotlib not available – skipping plots.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(w, bins=50, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("SPL Weight")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of Patch SPL Weights")

    # Group by image
    groups = _group_patches_by_image(dataset)
    group_means = {k: float(w[v].mean()) for k, v in groups.items()}
    sorted_groups = sorted(group_means.items(), key=lambda x: x[1])
    names = [x[0][-20:] for x in sorted_groups[:20]]
    vals  = [x[1] for x in sorted_groups[:20]]
    axes[1].barh(range(len(names)), vals, color="coral")
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels(names, fontsize=6)
    axes[1].set_xlabel("Mean SPL Weight")
    axes[1].set_title("20 Most Suppressed Image Groups")

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "patch_suppression.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir}/patch_suppression.png")


def compute_token_importance(
    model: ANATokenClassifier,
    dataset: ANATokenDataset,
    device,
    out_dir: str,
    max_samples: int = 500,
):
    """
    Aggregate attention weights over a subset of the dataset to understand
    which of the 257 tokens the model relies on most.

    Produces:
      - Mean attention per token position (bar chart)
      - CLS token attention vs spatial tokens comparison
      - Spatial token map (if 16×16 layout: 256 spatial + 1 CLS = 257)
    """
    os.makedirs(out_dir, exist_ok=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=2
    )

    all_attn = []
    all_conf = []
    count = 0

    with torch.no_grad():
        for tokens, _, _, _ in tqdm(loader, desc="Computing token importance"):
            if count >= max_samples:
                break
            if GPU_IN_USE:
                tokens = tokens.to(device)
            _, _, confidence, attn_w = model(tokens)
            all_attn.append(attn_w.cpu().numpy())
            all_conf.append(confidence.cpu().numpy())
            count += len(tokens)

    all_attn = np.vstack(all_attn)[:max_samples]   # [N, num_tokens]
    all_conf = np.concatenate(all_conf)[:max_samples]

    mean_attn = all_attn.mean(axis=0)               # [num_tokens]
    num_tokens = mean_attn.shape[0]

    print("\n=== Token Importance ===")
    print(f"  CLS token (idx 0) mean attention : {mean_attn[0]:.6f}")
    spatial_mean = mean_attn[1:].mean()
    print(f"  Spatial tokens (1:{num_tokens}) mean : {spatial_mean:.6f}")
    print(f"  Top-5 token indices: {np.argsort(mean_attn)[::-1][:5].tolist()}")
    print(f"  Mean confidence: {all_conf.mean():.4f} ± {all_conf.std():.4f}")

    if not PLT_AVAILABLE:
        warnings.warn("matplotlib not available – skipping plots.")
        return

    # ---- Plot 1: mean attention per token position ----
    fig, axes = plt.subplots(1, 3 if num_tokens == 257 else 2, figsize=(16, 4))

    axes[0].bar(range(num_tokens), mean_attn, color="teal", width=1.0)
    axes[0].axvline(x=0.5, color="red", linestyle="--", label="CLS boundary")
    axes[0].set_xlabel("Token Index")
    axes[0].set_ylabel("Mean Attention Weight")
    axes[0].set_title("Token Importance (all tokens)")
    axes[0].legend()

    axes[1].bar(["CLS (0)", f"Spatial\n(1-{num_tokens-1})"],
                [mean_attn[0], mean_attn[1:].mean()],
                color=["crimson", "steelblue"])
    axes[1].set_title("CLS vs Spatial Tokens")
    axes[1].set_ylabel("Mean Attention")

    # ---- Plot 2: spatial map (16×16 grid) ----
    if num_tokens == 257:
        spatial = mean_attn[1:].reshape(16, 16)
        im = axes[2].imshow(spatial, cmap="hot", interpolation="nearest")
        axes[2].set_title("Spatial Token Attention Map (16×16)")
        plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "token_importance.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir}/token_importance.png")

    # ---- Confidence distribution ----
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(all_conf, bins=40, color="mediumpurple", edgecolor="white")
    ax2.set_xlabel("Region Confidence")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of Per-Patch Region Confidence")
    plt.tight_layout()
    fig2.savefig(os.path.join(out_dir, "confidence_distribution.png"), dpi=150)
    plt.close(fig2)
    print(f"  Saved: {out_dir}/confidence_distribution.png")

    # Save raw attention stats
    stats = {
        "cls_mean_attn": float(mean_attn[0]),
        "spatial_mean_attn": float(mean_attn[1:].mean()),
        "top5_token_indices": np.argsort(mean_attn)[::-1][:5].tolist(),
        "mean_confidence": float(all_conf.mean()),
        "std_confidence": float(all_conf.std()),
    }
    with open(os.path.join(out_dir, "token_stats.json"), "w") as f:
        json.dump(stats, f, indent=4)
    print(f"  Saved: {out_dir}/token_stats.json")


def benchmark_comparison(results_files: dict, out_dir: str):
    """
    Build a comparison table from results.json files of different runs.

    Args:
        results_files : mapping backbone_name → path to results.json
        out_dir       : where to save comparison CSV / PNG
    """
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for backbone, path in results_files.items():
        if not os.path.isfile(path):
            warnings.warn(f"Results file not found: {path}")
            continue
        try:
            with open(path) as f:
                records = json.load(f)
            if not isinstance(records, list):
                records = [records]
        except json.JSONDecodeError as exc:
            warnings.warn(f"Could not parse results file {path}: {exc}")
            continue
        # Take the best mAP record
        best = max(records, key=lambda r: r.get("val_results", {}).get("mAP", 0))
        vr = best.get("val_results", {})
        rows.append({
            "backbone": backbone,
            "epoch": best.get("epoch"),
            "acc": vr.get("acc_total"),
            "f1_micro": vr.get("f1_mi"),
            "f1_macro": vr.get("f1_ma"),
            "mAP": vr.get("mAP"),
        })

    if not rows:
        print("No valid results found for comparison.")
        return

    df = pd.DataFrame(rows).set_index("backbone")
    print("\n=== Benchmark Comparison ===")
    print(df.to_string())

    csv_path = os.path.join(out_dir, "benchmark_comparison.csv")
    df.to_csv(csv_path)
    print(f"Saved: {csv_path}")

    if PLT_AVAILABLE:
        metrics = ["acc", "f1_micro", "f1_macro", "mAP"]
        fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4))
        for ax, metric in zip(axes, metrics):
            vals = df[metric].dropna()
            colors = ["steelblue" if "resnet" in idx.lower() else "darkorange"
                      for idx in vals.index]
            ax.bar(vals.index, vals.values, color=colors)
            ax.set_title(metric)
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=30)
        plt.suptitle("CLIP Token vs ResNet Benchmark Comparison", fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "benchmark_comparison.png"), dpi=150)
        plt.close(fig)
        print(f"Saved: {out_dir}/benchmark_comparison.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze token attention and patch suppression.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path",   required=False, default=None,
                        help="Path to saved ANATokenClassifier .pt checkpoint")
    parser.add_argument("--tokens_root",  default="./data/clip_tokens/")
    parser.add_argument("--annFile",      default="all_single_small_82240_19400_19330.csv")
    parser.add_argument("--split",        default="val", choices=["train", "val", "test"])
    parser.add_argument("--num_tokens",   default=257,   type=int)
    parser.add_argument("--token_dim",    default=768,   type=int)
    parser.add_argument("--out_dir",      default="./analysis_output/")
    parser.add_argument("--max_samples",  default=500,   type=int,
                        help="Max patches to use for token importance analysis")
    parser.add_argument("--stats_only",   action="store_true",
                        help="Only print statistics; do not save plots")
    # Benchmark comparison
    parser.add_argument("--compare",      nargs="*", default=None,
                        metavar="NAME:PATH",
                        help="Compare results files: resnet50:results.json token:token_results.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.compare:
        files = {}
        for item in args.compare:
            parts = item.split(":", 1)
            if len(parts) == 2:
                files[parts[0]] = parts[1]
        benchmark_comparison(files, args.out_dir)
    elif args.model_path:
        model = _load_model(args.model_path, args.num_tokens, args.token_dim, device)
        dataset = ANATokenDataset(
            tokens_root=args.tokens_root,
            annFile=args.annFile,
            split=args.split,
            num_tokens=args.num_tokens,
            token_dim=args.token_dim,
        )
        print(f"Loaded {len(dataset)} patches from '{args.split}' split.")

        if not args.stats_only:
            # For suppression map we need SPL weights – here we use confidence as proxy
            loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
            confs = []
            with torch.no_grad():
                for tokens, _, _, _ in tqdm(loader, desc="Confidence scores"):
                    if GPU_IN_USE:
                        tokens = tokens.to(device)
                    _, _, conf, _ = model(tokens)
                    confs.append(conf.cpu())
            confs = torch.cat(confs)
            compute_patch_suppression(dataset, confs, args.out_dir)

        compute_token_importance(model, dataset, device, args.out_dir, args.max_samples)
    else:
        print("Provide --model_path for analysis or --compare NAME:PATH ... for benchmarking.")
