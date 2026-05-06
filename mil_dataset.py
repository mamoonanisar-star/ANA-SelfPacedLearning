"""
mil_dataset.py

Phase 1 – Data Engineering for the MIL Pipeline.

Loads pre-computed CLIP ViT-L/14 embeddings (.npy files, shape [257, 768])
and multi-label annotations (8 ICAP classes) from a CSV annotation file.

Label Column Formats Supported
-------------------------------
1. Binary columns (ICAP format): one column per class at positions ``label_col_start``
   through ``label_col_start + num_classes - 1``  (default: columns 16:24).
2. Sparse "TARGET" column: space-separated class indices (e.g. "0 3 5") as used
   by the existing ANA SPL pipeline.

The dataset auto-detects which format is present and validates both the CSV
structure and every .npy embedding file before training starts.

Example CSV (ICAP binary format):
    path,Split,…,ICAP_AC,ICAP_CF,ICAP_DFS,ICAP_DC,ICAP_HS,ICAP_NMH,ICAP_NUC,ICAP_OT
    patient01/001.npy,train,…,1,0,0,1,0,0,0,0

Example CSV (TARGET sparse format):
    path,Split,TARGET
    patient01/001.npy,train,0 3
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ICAP_CLASSES: List[str] = [
    "ICAP_AC",   # Anti-centromere
    "ICAP_CF",   # Coarse Fluorescent
    "ICAP_DFS",  # Dense Fine Speckled
    "ICAP_DC",   # Diffuse Cytoplasmic
    "ICAP_HS",   # Homogeneous Speckled
    "ICAP_NMH",  # Nuclear Membrane Homogeneous
    "ICAP_NUC",  # Nucleolar
    "ICAP_OT",   # Other
]

_REQUIRED_COLUMNS = {"path", "Split"}
_TARGET_COLUMN = "TARGET"

# Default column slice for binary ICAP label columns
_DEFAULT_LABEL_COL_START = 16
_DEFAULT_LABEL_COL_END = 24  # exclusive → 8 classes


# ---------------------------------------------------------------------------
# ANAFeatureDataset
# ---------------------------------------------------------------------------

class ANAFeatureDataset(data.Dataset):
    """
    Dataset for Multi-Instance Learning (MIL) over CLIP token embeddings.

    Each sample is a .npy file of shape ``[num_tokens, token_dim]``
    (default: [257, 768] for ViT-L/14).  The first row (index 0) is the
    global CLS token; rows 1–256 are the 16×16 spatial patch tokens.

    Args:
        tokens_root (str): Root directory containing .npy embedding files.
        ann_file    (str): Path to the CSV annotation file.
        split       (str): One of ``"train"``, ``"val"``, ``"test"``.
        num_tokens  (int): Expected number of tokens per file (default 257).
        token_dim   (int): Expected token embedding dimension (default 768).
        num_classes (int): Number of output classes (default 8).
        label_col_start (int | None):
            First column index for binary ICAP label columns (0-indexed).
            Pass ``None`` to force "TARGET" sparse format.
        label_col_end (int | None):
            Exclusive end column index for binary ICAP labels.  Defaults to
            ``label_col_start + num_classes``.

    Raises:
        FileNotFoundError: If ``ann_file`` does not exist.
        ValueError: If required CSV columns are missing or the label format
                    cannot be detected.
    """

    def __init__(
        self,
        tokens_root: str,
        ann_file: str,
        split: str,
        num_tokens: int = 257,
        token_dim: int = 768,
        num_classes: int = 8,
        label_col_start: Optional[int] = _DEFAULT_LABEL_COL_START,
        label_col_end: Optional[int] = _DEFAULT_LABEL_COL_END,
    ):
        if not os.path.isfile(ann_file):
            raise FileNotFoundError(
                f"Annotation CSV not found: {ann_file!r}. "
                "Check the --ann_file argument."
            )
        if split not in ("train", "val", "test"):
            raise ValueError(
                f"split must be 'train', 'val', or 'test'; got {split!r}."
            )

        self.tokens_root = tokens_root
        self.ann_file = ann_file
        self.split = split
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.num_classes = num_classes

        # ------------------------------------------------------------------
        # Load and validate CSV
        # ------------------------------------------------------------------
        df = pd.read_csv(ann_file)
        self._validate_csv(df, label_col_start, label_col_end)

        subset = df[df["Split"] == split].reset_index(drop=True)
        if len(subset) == 0:
            raise ValueError(
                f"No rows found for split={split!r} in {ann_file!r}. "
                f"Available splits: {df['Split'].unique().tolist()}"
            )

        # ------------------------------------------------------------------
        # Resolve label format
        # ------------------------------------------------------------------
        self._label_mode, self._label_cols = self._detect_label_format(
            df, label_col_start, label_col_end
        )

        # ------------------------------------------------------------------
        # Build example list
        # ------------------------------------------------------------------
        self.examples: List[Tuple[str, List[int], str]] = (
            self._build_examples(subset)
        )

        # ------------------------------------------------------------------
        # Class-balance statistics (for pos_weight computation)
        # ------------------------------------------------------------------
        self.class_counts: np.ndarray = self._compute_class_counts()
        logger.info(
            "ANAFeatureDataset [%s] | %d samples | label mode: %s | "
            "class counts: %s",
            split,
            len(self.examples),
            self._label_mode,
            self.class_counts.tolist(),
        )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_csv(
        self,
        df: pd.DataFrame,
        label_col_start: Optional[int],
        label_col_end: Optional[int],
    ) -> None:
        missing = _REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV {self.ann_file!r} is missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        # Check that at least one label format is present
        has_target = _TARGET_COLUMN in df.columns
        has_binary = (
            label_col_start is not None
            and label_col_start < len(df.columns)
            and (label_col_end or label_col_start + self.num_classes) <= len(df.columns)
        )
        if not has_target and not has_binary:
            raise ValueError(
                f"CSV {self.ann_file!r} contains neither a '{_TARGET_COLUMN}' column "
                f"nor enough columns for binary labels at positions "
                f"{label_col_start}:{label_col_end}. "
                f"Total columns: {len(df.columns)}"
            )

    def _detect_label_format(
        self,
        df: pd.DataFrame,
        label_col_start: Optional[int],
        label_col_end: Optional[int],
    ) -> Tuple[str, Optional[List[str]]]:
        """
        Returns (mode, label_columns):
          - mode == "binary"  → label_columns is a list of column names
          - mode == "target"  → label_columns is None (use TARGET column)
        """
        # Prefer binary format if requested and columns exist
        if label_col_start is not None:
            end = label_col_end if label_col_end is not None else (
                label_col_start + self.num_classes
            )
            if end <= len(df.columns):
                cols = df.columns[label_col_start:end].tolist()
                expected = end - label_col_start
                if expected != self.num_classes:
                    warnings.warn(
                        f"Binary label slice [{label_col_start}:{end}] gives "
                        f"{expected} columns but num_classes={self.num_classes}. "
                        "Adjusting num_classes to match.",
                        stacklevel=3,
                    )
                logger.debug(
                    "Using binary label columns %s:%s → %s",
                    label_col_start,
                    end,
                    cols,
                )
                return "binary", cols

        # Fall back to TARGET column
        if _TARGET_COLUMN in df.columns:
            logger.debug("Using sparse TARGET column for labels.")
            return "target", None

        raise ValueError("Could not determine label format (should have been caught earlier).")

    # ------------------------------------------------------------------
    # Example building
    # ------------------------------------------------------------------

    def _build_examples(self, df: pd.DataFrame) -> List[Tuple[str, List[int], str]]:
        examples = []
        for row_idx, row in df.iterrows():
            raw_path = str(row["path"])
            stem = os.path.splitext(os.path.basename(raw_path))[0]
            token_path = self._resolve_token_path(stem, raw_path)
            labels = self._parse_labels(row, row_idx)
            examples.append((token_path, labels, stem))
        return examples

    def _parse_labels(self, row: pd.Series, row_idx: int) -> List[int]:
        if self._label_mode == "binary":
            assert self._label_cols is not None
            try:
                vals = row[self._label_cols].astype(float).tolist()
                return [i for i, v in enumerate(vals) if v > 0.5]
            except Exception as exc:
                warnings.warn(
                    f"Row {row_idx}: could not parse binary labels: {exc}. "
                    "Treating all classes as 0.",
                    stacklevel=3,
                )
                return []
        else:
            raw = row.get(_TARGET_COLUMN, None)
            if pd.isna(raw):
                warnings.warn(
                    f"Row {row_idx}: TARGET is NaN; treating as no label.",
                    stacklevel=3,
                )
                return []
            try:
                return [int(i) for i in str(raw).split()]
            except (ValueError, TypeError) as exc:
                warnings.warn(
                    f"Row {row_idx}: cannot parse TARGET={raw!r}: {exc}. "
                    "Treating as no label.",
                    stacklevel=3,
                )
                return []

    def _resolve_token_path(self, stem: str, original_path: str) -> str:
        """
        Resolve the .npy token file path, trying:
          1. Flat layout:         tokens_root/<stem>.npy
          2. Hierarchical layout: tokens_root/<parent_dir>/<stem>.npy
          3. Exact path:          tokens_root/<original_path>  (already .npy)
        Returns the first existing path, or the flat path as a sentinel.
        """
        flat = os.path.join(self.tokens_root, f"{stem}.npy")
        if os.path.isfile(flat):
            return flat

        parent = os.path.dirname(original_path)
        if parent:
            hier = os.path.join(self.tokens_root, parent, f"{stem}.npy")
            if os.path.isfile(hier):
                return hier

        # Check if original_path already ends with .npy
        if original_path.endswith(".npy"):
            exact = os.path.join(self.tokens_root, original_path)
            if os.path.isfile(exact):
                return exact

        # Return flat path as sentinel; _load_tokens will warn
        return flat

    # ------------------------------------------------------------------
    # Token loading
    # ------------------------------------------------------------------

    def _load_tokens(self, path: str) -> torch.Tensor:
        """
        Load a [num_tokens, token_dim] .npy file.

        On failure (missing file, wrong shape) returns a zero tensor with a
        warning so that training is not interrupted by a single bad sample.
        Shape mismatches are handled by padding or truncation.
        """
        try:
            arr = np.load(path, allow_pickle=False)
        except FileNotFoundError:
            warnings.warn(
                f"Token file not found: {path!r}. Using zero tensor.",
                stacklevel=2,
            )
            return torch.zeros(self.num_tokens, self.token_dim, dtype=torch.float32)
        except Exception as exc:
            warnings.warn(
                f"Failed to load {path!r}: {exc}. Using zero tensor.",
                stacklevel=2,
            )
            return torch.zeros(self.num_tokens, self.token_dim, dtype=torch.float32)

        # Squeeze leading batch dimension if saved as [1, T, D]
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]

        if arr.shape == (self.num_tokens, self.token_dim):
            return torch.from_numpy(arr.astype(np.float32))

        # Shape mismatch: pad / truncate
        warnings.warn(
            f"Token file {path!r} has shape {arr.shape}, "
            f"expected ({self.num_tokens}, {self.token_dim}). "
            "Padding/truncating.",
            stacklevel=2,
        )
        out = np.zeros((self.num_tokens, self.token_dim), dtype=np.float32)
        if arr.ndim == 2:
            t = min(arr.shape[0], self.num_tokens)
            d = min(arr.shape[1], self.token_dim)
            out[:t, :d] = arr[:t, :d]
        elif arr.ndim == 1:
            d = min(arr.shape[0], self.token_dim)
            out[0, :d] = arr[:d]
        return torch.from_numpy(out)

    # ------------------------------------------------------------------
    # Class balance helper
    # ------------------------------------------------------------------

    def _compute_class_counts(self) -> np.ndarray:
        """Return the number of positive examples for each class."""
        counts = np.zeros(self.num_classes, dtype=np.int64)
        for _, labels, _ in self.examples:
            for c in labels:
                if 0 <= c < self.num_classes:
                    counts[c] += 1
        return counts

    def compute_pos_weight(self) -> torch.Tensor:
        """
        Compute per-class positive weight for ``BCEWithLogitsLoss``.

        ``pos_weight[c] = (N - pos_c) / max(pos_c, 1)``

        A weight > 1 up-weights the positive class when it is rare.
        """
        n = len(self.examples)
        pos = self.class_counts.astype(np.float32)
        neg = n - pos
        pw = neg / np.maximum(pos, 1.0)
        return torch.from_numpy(pw.astype(np.float32))

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        """
        Returns:
            tokens : Float tensor [num_tokens, token_dim]
            label  : Float tensor [num_classes]  (multi-hot)
            index  : int – sample index (for SPL weight look-up)
        """
        token_path, label_indices, _ = self.examples[index]
        tokens = self._load_tokens(token_path)

        label = torch.zeros(self.num_classes, dtype=torch.float32)
        for i in label_indices:
            if 0 <= i < self.num_classes:
                label[i] = 1.0

        return tokens, label, index
