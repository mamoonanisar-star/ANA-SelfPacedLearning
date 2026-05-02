"""
extract_clip_tokens.py

Utility to extract CLIP token embeddings from ANA patch images and save
them as .npy files for use with ANATokenDataset / main_with_tokens.py.

    Each output file contains an array of shape [num_tokens, token_dim], e.g.:
  - ViT-B/16 : [197, 768]
  - ViT-L/14 : [257, 1024]
  - ViT-B/32 : [50,  512]

Usage
-----
    # Basic – extract tokens for all splits from a CSV
python extract_clip_tokens.py \\
    --annFile  all_single_small_82240_19400_19330.csv \\
    --img_root ./data/images/ \\
    --out_dir  ./data/clip_tokens/ \\
    --model    ViT-L/14

    # Only process the training split
python extract_clip_tokens.py \\
    --annFile  all_single_small_82240_19400_19330.csv \\
    --img_root ./data/images/ \\
    --out_dir  ./data/clip_tokens/ \\
    --split    train
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

try:
    import clip  # openai/clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

try:
    import open_clip  # open-clip-torch (alternative)
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _load_clip_model(model_name: str, device: str):
    """Load a CLIP model; try openai/clip then open_clip as fallback."""
    if CLIP_AVAILABLE:
        model, preprocess = clip.load(model_name, device=device)
        return model, preprocess, "openai"
    if OPEN_CLIP_AVAILABLE:
        # Map common model names
        name_map = {
            "ViT-L/14": ("ViT-L-14", "openai"),
            "ViT-B/16": ("ViT-B-16", "openai"),
            "ViT-B/32": ("ViT-B-32", "openai"),
        }
        ocname, pretrained = name_map.get(model_name, ("ViT-L-14", "openai"))
        model, _, preprocess = open_clip.create_model_and_transforms(ocname, pretrained=pretrained)
        model = model.to(device)
        return model, preprocess, "open_clip"
    raise ImportError(
        "Neither 'clip' nor 'open_clip' is installed. "
        "Install with: pip install git+https://github.com/openai/CLIP.git  "
        "or: pip install open-clip-torch"
    )


def _encode_image_tokens(model, image_tensor: torch.Tensor, backend: str) -> np.ndarray:
    """
    Extract the full token sequence (not just the pooled embedding) from CLIP.

    For openai/clip (ViT):
        model.visual.transformer returns a sequence [num_tokens, B, dim].
        We hook the output of the final transformer block before the projection.

    Returns:
        tokens : np.ndarray [num_tokens, token_dim]  (float32)
    """
    tokens_out = {}

    if backend == "openai":
        # Register a forward hook on the last transformer block
        last_block = model.visual.transformer.resblocks[-1]

        def hook_fn(module, inp, out):
            # out shape: [num_tokens, B, dim]
            tokens_out["raw"] = out.detach().permute(1, 0, 2)  # [B, num_tokens, dim]

        handle = last_block.register_forward_hook(hook_fn)
        with torch.no_grad():
            _ = model.encode_image(image_tensor)
        handle.remove()

        arr = tokens_out["raw"][0].cpu().float().numpy()  # [num_tokens, dim]
        return arr

    elif backend == "open_clip":
        visual = model.visual
        last_block = visual.transformer.resblocks[-1]

        def hook_fn(module, inp, out):
            tokens_out["raw"] = out.detach().permute(1, 0, 2)

        handle = last_block.register_forward_hook(hook_fn)
        with torch.no_grad():
            _ = model.encode_image(image_tensor)
        handle.remove()

        arr = tokens_out["raw"][0].cpu().float().numpy()
        return arr

    else:
        raise ValueError(f"Unknown backend: {backend}")


# ---------------------------------------------------------------------------
# Main extraction logic
# ---------------------------------------------------------------------------

def extract_tokens(
    annFile: str,
    img_root: str,
    out_dir: str,
    model_name: str = "ViT-L/14",
    split=None,
    batch_size: int = 1,
    device_str=None,
):
    """
    Extract CLIP token embeddings for all images in the annotation CSV and
    save each as a .npy file in out_dir, mirroring any subdirectory structure
    found in the CSV `path` column.

    Args:
        annFile    : Path to the CSV annotation file.
        img_root   : Root directory for images (prepended to CSV 'path' column).
        out_dir    : Output directory for .npy token files.
        model_name : CLIP model name (e.g. 'ViT-L/14').
        split      : If given, restrict to this CSV split ('train'/'val'/'test').
        batch_size : Batch size for GPU extraction (default: 1 for safety).
        device_str : 'cuda' / 'cpu' (auto-detected when None).
    """
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    print(f"Loading CLIP model: {model_name} on {device}")
    model, preprocess, backend = _load_clip_model(model_name, device_str)
    model.eval()

    df = pd.read_csv(annFile)
    if split is not None:
        df = df[df["Split"] == split].reset_index(drop=True)
    print(f"Processing {len(df)} images (split={split or 'all'})")

    os.makedirs(out_dir, exist_ok=True)
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting tokens"):
        img_rel_path = row["path"]
        img_path = os.path.join(img_root, img_rel_path)

        stem = os.path.splitext(os.path.basename(img_rel_path))[0]
        rel_dir = os.path.dirname(img_rel_path)
        out_subdir = os.path.join(out_dir, rel_dir) if rel_dir else out_dir
        os.makedirs(out_subdir, exist_ok=True)
        out_path = os.path.join(out_subdir, f"{stem}.npy")

        if os.path.isfile(out_path):
            continue  # already extracted

        if not os.path.isfile(img_path):
            warnings.warn(f"Image not found, skipping: {img_path}")
            skipped += 1
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            img_tensor = preprocess(image).unsqueeze(0).to(device)
            tokens = _encode_image_tokens(model, img_tensor, backend)
            np.save(out_path, tokens)
        except Exception as exc:
            warnings.warn(f"Failed to process {img_path}: {exc}")
            skipped += 1

    print(f"Done. {len(df) - skipped} token files saved to {out_dir}. Skipped: {skipped}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Extract CLIP token embeddings from ANA patch images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--annFile", required=True, help="CSV annotation file path")
    parser.add_argument("--img_root", default="./data/images/", help="Root image directory")
    parser.add_argument("--out_dir", default="./data/clip_tokens/", help="Output directory for .npy files")
    parser.add_argument("--model", default="ViT-L/14", help="CLIP model name (e.g. ViT-L/14, ViT-B/16)")
    parser.add_argument("--split", default=None, choices=["train", "val", "test", None],
                        help="Restrict to a specific CSV split (default: all)")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size (1 is safe for all GPUs)")
    parser.add_argument("--device", default=None, help="Compute device: 'cuda' or 'cpu'")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    extract_tokens(
        annFile=args.annFile,
        img_root=args.img_root,
        out_dir=args.out_dir,
        model_name=args.model,
        split=args.split,
        batch_size=args.batch_size,
        device_str=args.device,
    )
