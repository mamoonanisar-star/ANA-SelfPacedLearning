"""
src/data.py

Improved ANAFeatureDataset for the ANA MIL pipeline.

Key improvements over the original implementation
--------------------------------------------------
* Flexible label-column specification: named list (``label_cols``) *or*
  index range (``label_col_start``/``label_col_end``).
* Full CSV validation – checks required columns and split-column values.
* Embedding shape validation – logs a warning and pads/truncates when the
  .npy array differs from the expected ``[num_tokens, token_dim]`` shape.
* Configurable missing-file handling: ``'zeros'`` | ``'skip'`` | ``'error'``.
* Data-statistics logging at construction time (class distribution, missing
  file count).
* ``get_pos_weights()`` – per-class positive weights for
  ``BCEWithLogitsLoss``.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

from .config import DataConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ANAFeatureDataset(data.Dataset):
    """
    Dataset for pre-computed CLIP ViT-L/14 feature embeddings.

    Each sample returns ``(embedding, label, index, stem)`` where

    * ``embedding`` – float32 tensor ``[num_tokens, token_dim]``
    * ``label``     – float32 tensor ``[num_classes]``  (multi-hot binary)
    * ``index``     – integer position in this dataset
    * ``stem``      – file-stem string used to look up the .npy file

    Supported directory layouts
    ---------------------------
    Flat layout::

        embeddings_root/
        ├── 0001.npy
        └── ...

    Hierarchical layout::

        embeddings_root/
        ├── patient01/
        │   ├── 0001.npy
        │   └── ...
        └── ...

    The class auto-detects the layout at example-building time.

    Args:
        cfg     : :class:`~src.config.DataConfig` with all data settings.
        split   : One of ``'train'``, ``'val'``, ``'test'``.
        verbose : Log data statistics at construction time.
    """

    def __init__(
        self,
        cfg: DataConfig,
        split: str,
        verbose: bool = True,
    ) -> None:
        self.cfg = cfg
        self.split = split
        self.verbose = verbose

        # ---- validate split argument ----
        if split not in cfg.valid_splits:
            raise ValueError(
                f"split={split!r} not in valid_splits={cfg.valid_splits}"
            )

        # ---- load and validate CSV ----
        df = self._load_csv(cfg.csv_path)
        df = self._validate_csv(df)

        # ---- filter to requested split ----
        df_split = df[df[cfg.split_col] == split].reset_index(drop=True)
        if len(df_split) == 0:
            raise ValueError(
                f"No samples found for split={split!r} in {cfg.csv_path}"
            )

        # ---- resolve label columns ----
        self.label_col_names: List[str] = self._resolve_label_cols(df_split)
        self.num_classes: int = len(self.label_col_names)

        if verbose:
            logger.info(
                "Split=%s: %d rows, %d classes (%s)",
                split,
                len(df_split),
                self.num_classes,
                ", ".join(self.label_col_names),
            )

        # ---- build examples ----
        self.examples: List[Dict]
        self.skipped: int
        self.examples, self.skipped = self._build_examples(df_split)

        if verbose:
            self._log_statistics()

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------

    def _load_csv(self, path: str) -> pd.DataFrame:
        """Load the annotation CSV with basic validation."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Annotation CSV not found: {path}")
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise RuntimeError(f"Failed to read CSV {path}: {exc}") from exc
        if len(df) == 0:
            raise ValueError(f"CSV file is empty: {path}")
        return df

    def _validate_csv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate that required columns are present and split values are known."""
        cfg = self.cfg
        required = [cfg.path_col, cfg.split_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"CSV missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        found_splits = set(df[cfg.split_col].dropna().unique())
        unknown = found_splits - set(cfg.valid_splits)
        if unknown:
            logger.warning(
                "CSV split column contains unexpected values: %s "
                "(expected one of: %s)",
                unknown,
                cfg.valid_splits,
            )
        return df

    def _resolve_label_cols(self, df: pd.DataFrame) -> List[str]:
        """
        Determine the label columns.

        Priority:
          1. ``cfg.label_cols`` – explicit list of column names.
          2. ``cfg.label_col_start`` / ``cfg.label_col_end`` – index range.
        """
        cfg = self.cfg
        if cfg.label_cols is not None:
            missing = [c for c in cfg.label_cols if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Specified label_cols not found in CSV: {missing}. "
                    f"Available: {list(df.columns)}"
                )
            return list(cfg.label_cols)

        # Index-range fallback
        ncols = len(df.columns)
        start = cfg.label_col_start
        end = min(cfg.label_col_end, ncols)
        if start >= ncols:
            raise ValueError(
                f"label_col_start={start} >= number of columns ({ncols})"
            )
        label_cols = list(df.columns[start:end])
        if not label_cols:
            raise ValueError(
                f"No label columns found in range [{start}:{end}]. "
                f"CSV has {ncols} columns."
            )
        return label_cols

    # ------------------------------------------------------------------
    # Example building
    # ------------------------------------------------------------------

    def _build_examples(
        self, df: pd.DataFrame
    ) -> Tuple[List[Dict], int]:
        """
        Build the list of example dicts from the filtered DataFrame.

        Returns
        -------
        examples : list of dicts with keys ``'path'``, ``'npy_path'``,
                   ``'labels'``, ``'stem'``.
        skipped  : number of rows excluded (only non-zero when
                   ``missing_file_strategy == 'skip'``).
        """
        examples: List[Dict] = []
        skipped = 0
        cfg = self.cfg

        for idx, row in df.iterrows():
            patch_path = str(row[cfg.path_col])
            stem = os.path.splitext(os.path.basename(patch_path))[0]

            npy_path = self._resolve_npy_path(stem, patch_path)

            if not os.path.isfile(npy_path):
                if cfg.missing_file_strategy == "error":
                    raise FileNotFoundError(
                        f"Embedding file not found: {npy_path}"
                    )
                if cfg.missing_file_strategy == "skip":
                    logger.debug("Skipping missing embedding: %s", npy_path)
                    skipped += 1
                    continue
                # 'zeros' – defer to __getitem__
                logger.debug(
                    "Missing embedding (will use zeros): %s", npy_path
                )

            labels = self._parse_labels(row, idx)
            examples.append(
                {
                    "path": patch_path,
                    "npy_path": npy_path,
                    "labels": labels,
                    "stem": stem,
                }
            )

        if skipped > 0:
            logger.info(
                "Excluded %d examples with missing .npy files (strategy='skip')",
                skipped,
            )

        return examples, skipped

    def _resolve_npy_path(self, stem: str, original_path: str) -> str:
        """
        Resolve the .npy file path, trying flat layout first then hierarchical.

        Flat layout     : ``embeddings_root/<stem>.npy``
        Hierarchical    : ``embeddings_root/<parent_dir>/<stem>.npy``
        """
        cfg = self.cfg

        # Flat
        flat = os.path.join(cfg.embeddings_root, f"{stem}.npy")
        if os.path.isfile(flat):
            return flat

        # Hierarchical
        parent = os.path.dirname(original_path)
        if parent:
            hier = os.path.join(cfg.embeddings_root, parent, f"{stem}.npy")
            if os.path.isfile(hier):
                return hier

        # Return flat path as sentinel; _load_embedding will handle it
        return flat

    def _parse_labels(self, row: pd.Series, row_idx: object) -> np.ndarray:
        """Parse the multi-label binary vector from a DataFrame row."""
        try:
            labels = row[self.label_col_names].values.astype(np.float32)
        except Exception as exc:
            logger.warning(
                "Row %s: failed to parse labels (%s). Using zeros.", row_idx, exc
            )
            labels = np.zeros(self.num_classes, dtype=np.float32)
        return np.clip(labels, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Embedding loading
    # ------------------------------------------------------------------

    def _load_embedding(self, npy_path: str) -> torch.Tensor:
        """
        Load a .npy embedding file and validate its shape.

        Returns
        -------
        Float tensor ``[num_tokens, token_dim]``.
        A zero tensor is returned when the file is missing or corrupted
        (unless ``missing_file_strategy == 'error'`` was requested, which
        is already handled in ``_build_examples``).
        """
        cfg = self.cfg
        expected = (cfg.expected_num_tokens, cfg.expected_token_dim)

        if not os.path.isfile(npy_path):
            logger.warning(
                "Embedding not found at runtime: %s. Returning zeros.", npy_path
            )
            return torch.zeros(expected, dtype=torch.float32)

        try:
            arr = np.load(npy_path, allow_pickle=False)
        except Exception as exc:
            logger.warning(
                "Failed to load embedding %s: %s. Returning zeros.", npy_path, exc
            )
            return torch.zeros(expected, dtype=torch.float32)

        # Strip leading batch dimension if present
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]

        # Validate shape
        if arr.shape != expected:
            logger.warning(
                "Embedding %s has shape %s, expected %s. "
                "Padding / truncating to expected shape.",
                npy_path,
                arr.shape,
                expected,
            )
            arr = self._reshape_embedding(arr, expected)

        return torch.from_numpy(arr.astype(np.float32))

    @staticmethod
    def _reshape_embedding(
        arr: np.ndarray,
        expected_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Pad or truncate *arr* to *expected_shape*."""
        out = np.zeros(expected_shape, dtype=np.float32)
        if arr.ndim == 2:
            t = min(arr.shape[0], expected_shape[0])
            d = min(arr.shape[1], expected_shape[1])
            out[:t, :d] = arr[:t, :d]
        elif arr.ndim == 1:
            d = min(arr.shape[0], expected_shape[1])
            out[0, :d] = arr[:d]
        # Any other ndim → return the all-zero array
        return out

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _log_statistics(self) -> None:
        """Log per-class positive rates and overall sample counts."""
        if not self.examples:
            logger.warning("No examples loaded for split=%s", self.split)
            return

        all_labels = np.stack([ex["labels"] for ex in self.examples])
        counts = all_labels.sum(axis=0)
        total = len(self.examples)

        logger.info("=== Data Statistics [split=%s] ===", self.split)
        logger.info(
            "  Samples: %d  |  Skipped: %d", total, self.skipped
        )
        for i, (name, count) in enumerate(
            zip(self.label_col_names, counts)
        ):
            pct = 100.0 * count / total if total > 0 else 0.0
            logger.info(
                "  Class %d (%s): %d / %d  (%.1f%%)",
                i, name, int(count), total, pct,
            )

    def get_pos_weights(self) -> torch.Tensor:
        """
        Compute per-class positive weights for ``BCEWithLogitsLoss``.

        ``pos_weight[c] = (num_negative[c] / num_positive[c])``

        Classes with no positive samples receive a weight of 1.0 to avoid
        division-by-zero and NaN propagation.

        Returns
        -------
        Float tensor ``[num_classes]``.
        """
        all_labels = np.stack([ex["labels"] for ex in self.examples])
        pos = all_labels.sum(axis=0)
        neg = len(self.examples) - pos
        # Guard against zero-positive classes
        pos_safe = np.where(pos > 0, pos, 1.0)
        weights = neg / pos_safe
        return torch.from_numpy(weights.astype(np.float32))

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        """
        Returns
        -------
        embedding : Float tensor ``[num_tokens, token_dim]``
        label     : Float tensor ``[num_classes]``
        index     : int
        stem      : str – file-stem identifier
        """
        ex = self.examples[index]
        embedding = self._load_embedding(ex["npy_path"])
        label = torch.from_numpy(ex["labels"])
        return embedding, label, index, ex["stem"]


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def get_traindata(
    cfg: DataConfig, verbose: bool = True
) -> ANAFeatureDataset:
    """Return the training split of :class:`ANAFeatureDataset`."""
    return ANAFeatureDataset(cfg, split="train", verbose=verbose)


def get_valdata(
    cfg: DataConfig, verbose: bool = True
) -> ANAFeatureDataset:
    """Return the validation split of :class:`ANAFeatureDataset`."""
    return ANAFeatureDataset(cfg, split="val", verbose=verbose)


def get_testdata(
    cfg: DataConfig, verbose: bool = True
) -> ANAFeatureDataset:
    """Return the test split of :class:`ANAFeatureDataset`."""
    return ANAFeatureDataset(cfg, split="test", verbose=verbose)
