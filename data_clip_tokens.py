"""
data_clip_tokens.py

Dataset class for loading pre-computed CLIP token embeddings (.npy files)
for ANA multi-label classification.

Expected token file shape: [num_tokens, token_dim]  e.g. [257, 768]

Supported directory layouts
-----------------------------
Flat (image_patch_*.npy):
    tokens_root/
    ├── 0001_patch_0.npy
    ├── 0001_patch_1.npy
    └── ...

Hierarchical (image/patch_*.npy):
    tokens_root/
    ├── 0001/
    │   ├── patch_0.npy
    │   └── patch_1.npy
    └── ...

The class auto-detects the layout and falls back gracefully (zeros + warning)
when a token file is missing or corrupted.
"""

import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data


class ANATokenDataset(data.Dataset):
    """
    Dataset that returns pre-computed CLIP token tensors instead of raw images.

    Each row in the CSV annotation file corresponds to one patch.  The dataset
    resolves the token file path from the patch filename, loads it, and
    returns (tokens, label, index, filename).

    Args:
        tokens_root   (str): Root directory containing .npy token files.
        annFile       (str): Path to the CSV annotation file.
        split         (str): 'train', 'val', or 'test'.
        num_tokens    (int): Expected first dimension of token array (e.g. 257).
        token_dim     (int): Expected second dimension of token array (e.g. 768).
        num_classes   (int): Number of output classes (default: 8).
    """

    def __init__(
        self,
        tokens_root: str,
        annFile: str,
        split: str,
        num_tokens: int = 257,
        token_dim: int = 768,
        num_classes: int = 8,
    ):
        self.tokens_root = tokens_root
        self.split = split
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.num_classes = num_classes

        info = pd.read_csv(annFile)
        info = info[info["Split"] == split].reset_index(drop=True)
        self.examples = self._build_examples(info)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_examples(self, info: pd.DataFrame):
        examples = []
        for row_idx, row in info.iterrows():
            patch_name = row["path"]                          # e.g. "0001.jpg" or "sub/0001.jpg"
            stem = os.path.splitext(os.path.basename(patch_name))[0]  # "0001"
            try:
                raw_target = row["TARGET"]
                if pd.isna(raw_target):
                    warnings.warn(
                        f"Row {row_idx}: TARGET is NaN; treating as class 0.",
                        stacklevel=2,
                    )
                    labels = [0]
                else:
                    labels = [int(i) for i in str(raw_target).split()]
            except (ValueError, TypeError) as exc:
                warnings.warn(
                    f"Row {row_idx}: could not parse TARGET={row['TARGET']!r}: {exc}. "
                    "Treating as class 0.",
                    stacklevel=2,
                )
                labels = [0]
            token_path = self._resolve_token_path(stem, patch_name)
            examples.append((token_path, labels, stem))
        return examples

    def _resolve_token_path(self, stem: str, original_path: str) -> str:
        """
        Try flat layout first, then hierarchical, then fallback sentinel.

        Flat     : tokens_root/<stem>.npy
        Hierarchical : tokens_root/<parent_dir>/<stem>.npy
        """
        # Flat layout
        flat = os.path.join(self.tokens_root, f"{stem}.npy")
        if os.path.isfile(flat):
            return flat

        # Hierarchical layout: use parent directory from CSV path
        parent = os.path.dirname(original_path)
        if parent:
            hier = os.path.join(self.tokens_root, parent, f"{stem}.npy")
            if os.path.isfile(hier):
                return hier

        # Return flat path as sentinel; __getitem__ will warn and use zeros
        return flat

    def _load_tokens(self, path: str) -> torch.Tensor:
        """Load token .npy file with error handling; return zeros on failure."""
        try:
            arr = np.load(path, allow_pickle=False)
            # Accept [num_tokens, token_dim] or [1, num_tokens, token_dim]
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.shape != (self.num_tokens, self.token_dim):
                warnings.warn(
                    f"Token file {path} has shape {arr.shape}, "
                    f"expected ({self.num_tokens}, {self.token_dim}). "
                    "Padding/truncating to expected shape.",
                    stacklevel=2,
                )
                arr = self._reshape_tokens(arr)
            return torch.from_numpy(arr.astype(np.float32))
        except FileNotFoundError:
            warnings.warn(
                f"Token file not found: {path}. Using zero tensor.",
                stacklevel=2,
            )
        except Exception as exc:
            warnings.warn(
                f"Failed to load token file {path}: {exc}. Using zero tensor.",
                stacklevel=2,
            )
        return torch.zeros(self.num_tokens, self.token_dim, dtype=torch.float32)

    def _reshape_tokens(self, arr: np.ndarray) -> np.ndarray:
        """
        Pad or truncate a token array to the expected (num_tokens, token_dim) shape.

        Handles three common cases:
          - 2-D array [T, D]: copy the overlapping region into the output.
          - 1-D array [D]:    treat as a single token, place at row 0.
          - anything else:    return zeros.
        """
        out = np.zeros((self.num_tokens, self.token_dim), dtype=np.float32)
        if arr.ndim == 2:
            t = min(arr.shape[0], self.num_tokens)
            d = min(arr.shape[1], self.token_dim)
            out[:t, :d] = arr[:t, :d]
        elif arr.ndim == 1:
            d = min(arr.shape[0], self.token_dim)
            out[0, :d] = arr[:d]
        # Other ndim values → return zeros (no assignment)
        return out

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        """
        Returns:
            tokens   : Float tensor [num_tokens, token_dim]
            label    : Float tensor [num_classes]  (multi-hot)
            index    : int
            filename : str – patch stem used to look up the token file
        """
        token_path, real_labels, stem = self.examples[index]
        tokens = self._load_tokens(token_path)

        label = torch.zeros(self.num_classes, dtype=torch.float32)
        for i in real_labels:
            if 0 <= i < self.num_classes:
                label[i] = 1.0

        return tokens, label, index, stem


# ---------------------------------------------------------------------------
# Convenience factory functions (mirror data_weighted_filename.py API)
# ---------------------------------------------------------------------------

def get_traindata(
    tokens_root: str,
    annFile: str,
    split: str = "train",
    num_tokens: int = 257,
    token_dim: int = 768,
    num_classes: int = 8,
) -> ANATokenDataset:
    """Return training split of ANATokenDataset."""
    return ANATokenDataset(tokens_root, annFile, split, num_tokens, token_dim, num_classes)


def get_testdata(
    tokens_root: str,
    annFile: str,
    split: str = "val",
    num_tokens: int = 257,
    token_dim: int = 768,
    num_classes: int = 8,
) -> ANATokenDataset:
    """Return validation/test split of ANATokenDataset."""
    return ANATokenDataset(tokens_root, annFile, split, num_tokens, token_dim, num_classes)
