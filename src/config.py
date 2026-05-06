"""
src/config.py

Centralised configuration for the ANA MIL pipeline.

All hyper-parameters, paths and class definitions live here.
YAML or JSON config files can be used to override any default via
``load_config(path)``.

Example JSON override
---------------------
{
  "data": {
    "csv_path": "/data/features_index_english.csv",
    "embeddings_root": "/data/anadata/",
    "label_cols": ["AC-1", "AC-2", "AC-3", "AC-4", "AC-5", "AC-6", "AC-7", "AC-8"]
  },
  "training": {
    "epochs": 100,
    "learning_rate": 5e-4,
    "loss_type": "bce_weighted"
  }
}
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Data loading and pre-processing configuration."""

    # ---- Paths ----
    # Path to the CSV annotation file (e.g. features_index_english.csv)
    csv_path: str = "features_index_english.csv"
    # Root directory that contains the .npy embedding files
    embeddings_root: str = "./data/anadata/"

    # ---- CSV column names ----
    # Column that holds the relative path to the patch image / embedding
    path_col: str = "path"
    # Column that identifies train / val / test split membership
    split_col: str = "Split"

    # ---- Label columns ----
    # Option 1 – explicit list of column names (takes priority when set)
    label_cols: Optional[List[str]] = None
    # Option 2 – fallback: index range [label_col_start, label_col_end)
    #   Default [16:24] matches the 8 ICAP class columns in
    #   features_index_english.csv; adjust if your CSV differs.
    label_col_start: int = 16
    label_col_end: int = 24

    # ---- Embedding validation ----
    # Expected shape of each .npy file: [num_tokens, token_dim]
    # ViT-L/14 produces 257 tokens × 768 dim (project default)
    expected_num_tokens: int = 257
    expected_token_dim: int = 768

    # ---- Missing file handling ----
    # 'zeros'  – return a zero tensor and log a warning (default)
    # 'skip'   – exclude the sample from the dataset
    # 'error'  – raise FileNotFoundError immediately
    missing_file_strategy: str = "zeros"

    # ---- Class count ----
    # Number of ANA fluorescence pattern classes (ICAP categories)
    num_classes: int = 8

    # ---- MIL bag grouping ----
    # Column to group patches into bags; None means each row is its own bag
    bag_key_col: Optional[str] = None

    # ---- Valid split labels ----
    valid_splits: List[str] = field(
        default_factory=lambda: ["train", "val", "test"]
    )


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    # ---- Token / embedding dimensions (mirror DataConfig) ----
    token_dim: int = 768
    num_tokens: int = 257
    num_classes: int = 8

    # ---- MIL pooling ----
    # Number of top-scoring instances pooled per class
    top_k: int = 10
    # Hidden dimension for feature projector and pooling head
    hidden_dim: int = 256
    # Dropout probability
    dropout: float = 0.1

    # ---- Weight initialisation ----
    # 'xavier_uniform' | 'kaiming_normal' | 'default' (no explicit init)
    init_strategy: str = "xavier_uniform"


@dataclass
class TrainingConfig:
    """Training hyper-parameters."""

    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    # Adam betas
    momentum: tuple = (0.9, 0.999)

    # ---- Loss ----
    # 'bce'          – plain BCEWithLogitsLoss (no pos_weight)
    # 'bce_weighted' – BCEWithLogitsLoss with pos_weight from class distribution
    # 'self_paced'   – DynamicSelfPacedLoss (integrates SPL curriculum)
    loss_type: str = "bce_weighted"
    # Whether to compute and apply pos_weight from training set distribution
    use_pos_weight: bool = True

    # ---- Self-paced learning ----
    spl_lambda_init: float = 0.5   # initial pace threshold
    spl_lambda_max: float = 1.0    # final pace threshold
    spl_warmup_epochs: int = 5     # epochs to ramp λ from init to max

    # ---- Gradient accumulation ----
    # Effective batch size = batch_size × grad_accum_steps
    grad_accum_steps: int = 1

    # ---- Reproducibility ----
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True

    # ---- Checkpointing ----
    save_dir: str = "./outputs/mil_pipeline/"
    # Metric used to select the best checkpoint
    # 'mAP' | 'f1_mi' | 'f1_ma' | 'acc'
    save_best_metric: str = "mAP"

    # ---- Early stopping ----
    patience: int = 10


@dataclass
class EvaluationConfig:
    """Evaluation and threshold-tuning configuration."""

    # ---- Threshold tuning ----
    tune_thresholds: bool = True
    # Metric optimised per class during threshold search
    # 'f1' | 'precision' | 'recall'
    threshold_metric: str = "f1"
    # Candidate thresholds to search
    threshold_search_values: List[float] = field(
        default_factory=lambda: [i / 20 for i in range(1, 20)]  # 0.05 … 0.95
    )
    default_threshold: float = 0.5

    # ---- Output artefacts ----
    save_confusion_matrix: bool = True
    save_roc_curves: bool = True
    save_pr_curves: bool = True
    # 'json' | 'html' | 'both'
    report_format: str = "json"


@dataclass
class MILConfig:
    """Top-level MIL pipeline configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # ---- Logging ----
    log_level: str = "INFO"
    log_file: Optional[str] = None


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: Union[str, Path, None] = None) -> MILConfig:
    """
    Load configuration, optionally overriding defaults from a JSON or YAML file.

    Args:
        path: Path to a ``.json`` or ``.yaml`` / ``.yml`` config file.
              If *None*, the function returns the default :class:`MILConfig`.

    Returns:
        :class:`MILConfig` with all settings resolved.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file extension is unsupported.
        ImportError: If PyYAML is required but not installed.
    """
    cfg = MILConfig()
    if path is None:
        return cfg

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path) as f:
            overrides = json.load(f)
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for YAML config files: pip install pyyaml"
            ) from exc
        with open(path) as f:
            overrides = yaml.safe_load(f)
    else:
        raise ValueError(
            f"Unsupported config format: {suffix!r}. Use .json or .yaml"
        )

    _apply_overrides(cfg, overrides)
    logger.info("Loaded config from: %s", path)
    return cfg


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_overrides(cfg: MILConfig, overrides: dict) -> None:
    """Recursively apply a dict of overrides to a :class:`MILConfig`."""
    sub_map = {
        "data": cfg.data,
        "model": cfg.model,
        "training": cfg.training,
        "evaluation": cfg.evaluation,
    }
    for key, val in overrides.items():
        if key in sub_map and isinstance(val, dict):
            _apply_overrides_to_dataclass(sub_map[key], val)
        elif hasattr(cfg, key):
            setattr(cfg, key, val)
        else:
            logger.warning("Unknown top-level config key: %r", key)


def _apply_overrides_to_dataclass(obj: object, overrides: dict) -> None:
    """Apply a flat dict of overrides to a nested dataclass."""
    for key, val in overrides.items():
        if hasattr(obj, key):
            setattr(obj, key, val)
        else:
            logger.warning(
                "Unknown config key %r in %s", key, type(obj).__name__
            )
