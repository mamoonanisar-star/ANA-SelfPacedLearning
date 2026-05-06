"""
src/model.py

Refactored MIL model architecture for ANA fluorescence pattern
classification using CLIP ViT-L/14 embeddings.

Architecture overview
---------------------

    CLIP tokens [B, T, D]
          │
    MILFeatureExtractor
      (CLS token + spatial mean → concat → MLP → LayerNorm)
          │
    instance features [B, hidden_dim]
          │
    ClassWiseMaxPoolingHead
      (vectorised top-k per class – NO Python loops)
          │
    bag logits [B, num_classes]

Key improvements over the original monolithic version
------------------------------------------------------
* ``ClassWiseMaxPoolingHead.forward_bags()`` is **fully vectorised** via
  ``torch.topk()``, replacing the O(B × C × K) nested Python loop with a
  single CUDA-parallel operation.
* Explicit weight-initialisation strategies (Xavier uniform by default).
* ``DynamicSelfPacedLoss`` is implemented and properly integrated into the
  training loop (previously defined but never used).
* All public classes carry type-annotated signatures.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

class MILFeatureExtractor(nn.Module):
    """
    Projects a CLIP token sequence to a single instance feature vector.

    Architecture::

        tokens [B, T, D]
          → CLS token [B, D]  +  spatial mean [B, D]
          → concat            [B, 2D]
          → Linear + GELU + Dropout + Linear
          → LayerNorm         [B, hidden_dim]

    Args:
        token_dim  : Input token embedding dimension (e.g. 768).
        hidden_dim : Output feature dimension.
        dropout    : Dropout probability.
    """

    def __init__(
        self,
        token_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim

        self.proj = nn.Sequential(
            nn.Linear(token_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens : Float tensor ``[B, T, D]``

        Returns:
            features : Float tensor ``[B, hidden_dim]``
        """
        cls_feat = tokens[:, 0, :]               # [B, D]  – CLS token
        mean_feat = tokens[:, 1:, :].mean(dim=1) # [B, D]  – spatial mean
        combined = torch.cat([cls_feat, mean_feat], dim=-1)  # [B, 2D]
        return self.norm(self.proj(combined))     # [B, hidden_dim]


# ---------------------------------------------------------------------------
# Vectorised class-wise max pooling head
# ---------------------------------------------------------------------------

class ClassWiseMaxPoolingHead(nn.Module):
    """
    Vectorised class-wise top-k mean pooling for Multi-Instance Learning.

    For each class *c*, the head selects the *k* instances with the highest
    class-*c* score and averages them to produce the bag-level logit for
    class *c*.

    **Before (nested Python loops – O(B × C × K) iterations)**::

        for b in range(batch_size):
            for c in range(num_classes):
                for k in range(top_k):
                    ...  # Python overhead per element

    **After (single vectorised call – O(1) PyTorch / CUDA ops)**::

        inst_scores = score_proj(instance_features)   # [B, N, C]
        scores_t    = inst_scores.permute(0, 2, 1)    # [B, C, N]
        top_vals, _ = torch.topk(scores_t, k, dim=-1) # [B, C, k]
        bag_logits  = top_vals.mean(dim=-1)            # [B, C]

    Args:
        hidden_dim  : Instance feature dimension (must match
                      :class:`MILFeatureExtractor` output).
        num_classes : Number of output classes.
        top_k       : Number of top instances pooled per class.
        dropout     : Dropout on instance features before projection.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_classes: int = 8,
        top_k: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.top_k = top_k

        self.drop = nn.Dropout(dropout)
        # Per-instance score projection: [*, hidden_dim] → [*, num_classes]
        self.score_proj = nn.Linear(hidden_dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.score_proj.weight)
        nn.init.zeros_(self.score_proj.bias)

    def forward(
        self,
        instance_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-instance forward (each row is treated independently).

        Args:
            instance_features : ``[B, hidden_dim]``

        Returns:
            logits      : ``[B, num_classes]``
            inst_scores : ``[B, num_classes]``  (same as logits here)
        """
        scores = self.score_proj(self.drop(instance_features))  # [B, C]
        return scores, scores

    def forward_bags(
        self,
        bag_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorised class-wise top-k pooling over bags of instances.

        This method replaces the original O(B × C × K) nested-loop
        implementation with a single call to ``torch.topk()``.

        Args:
            bag_features : ``[B, N, hidden_dim]``
                           *B* bags, each containing *N* instances.

        Returns:
            bag_logits  : ``[B, num_classes]`` – bag-level class scores.
            inst_scores : ``[B, N, num_classes]`` – per-instance scores.
        """
        B, N, D = bag_features.shape
        k = min(self.top_k, N)

        # Score every instance for every class
        flat_scores = self.score_proj(
            self.drop(bag_features.view(B * N, D))
        )                                              # [B*N, C]
        inst_scores = flat_scores.view(B, N, self.num_classes)  # [B, N, C]

        # Transpose so topk operates over the instance dimension
        scores_t = inst_scores.permute(0, 2, 1)       # [B, C, N]

        # ── Vectorised top-k (replaces all nested Python loops) ──────────
        top_vals, _ = torch.topk(
            scores_t, k=k, dim=-1, largest=True, sorted=False
        )                                              # [B, C, k]

        # Bag logit = mean of top-k scores per class
        bag_logits = top_vals.mean(dim=-1)             # [B, C]

        return bag_logits, inst_scores


# ---------------------------------------------------------------------------
# Dynamic self-paced loss (previously defined but never used in training)
# ---------------------------------------------------------------------------

class DynamicSelfPacedLoss(nn.Module):
    """
    Dynamic self-paced learning loss.

    Samples whose per-sample loss is below the current pace threshold λ are
    included in the gradient update; the rest are masked out.  λ increases
    linearly from ``lambda_init`` to ``lambda_max`` over ``warmup_epochs``.

    References
    ----------
    Kumar et al. "Self-Paced Learning for Latent Variable Models."
    *NeurIPS 2010*.

    Jiang et al. "Self-Paced Curriculum Learning." *AAAI 2015*.

    Args:
        lambda_init    : Initial pace threshold (lower → fewer samples).
        lambda_max     : Final pace threshold (1.0 ≈ all samples).
        warmup_epochs  : Epochs to ramp λ from *lambda_init* to *lambda_max*.
        pos_weight     : Per-class positive weights for the inner BCE loss.
    """

    def __init__(
        self,
        lambda_init: float = 0.5,
        lambda_max: float = 1.0,
        warmup_epochs: int = 5,
        pos_weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.lambda_init = lambda_init
        self.lambda_max = lambda_max
        self.warmup_epochs = warmup_epochs
        self.current_lambda = lambda_init

        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight, reduction="none"
        )

    def update_lambda(self, epoch: int) -> None:
        """Advance the pace threshold to match *epoch*."""
        if self.warmup_epochs > 0:
            progress = min(epoch / self.warmup_epochs, 1.0)
            self.current_lambda = (
                self.lambda_init
                + (self.lambda_max - self.lambda_init) * progress
            )
        else:
            self.current_lambda = self.lambda_max

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            logits  : ``[B, C]`` – raw model outputs.
            targets : ``[B, C]`` – binary labels.

        Returns:
            loss           : Scalar weighted loss.
            sample_weights : ``[B]`` – 0 / 1 inclusion mask.
        """
        # Per-sample mean loss
        per_sample = self.bce(logits, targets).mean(dim=-1)  # [B]

        # Binary self-paced gate: include samples with loss ≤ λ
        sample_weights = (per_sample <= self.current_lambda).float().detach()

        weighted_sum = (per_sample * sample_weights).sum()
        n_selected = sample_weights.sum().clamp(min=1.0)
        loss = weighted_sum / n_selected

        return loss, sample_weights


# ---------------------------------------------------------------------------
# Full MIL classifier
# ---------------------------------------------------------------------------

class MILClassifier(nn.Module):
    """
    End-to-end MIL classifier for ANA multi-label fluorescence
    pattern classification.

    Architecture Diagram::

        ┌──────────────────────────────────────────────────┐
        │  Input: CLIP tokens  [B, T, D]                   │
        │    T = num_tokens (e.g. 257 = 1 CLS + 256 patch) │
        │    D = token_dim  (e.g. 768)                     │
        └───────────────────────┬──────────────────────────┘
                                │
                    MILFeatureExtractor
              (CLS + mean-pool → proj → LayerNorm)
                                │
                        [B, hidden_dim]
                                │
                ClassWiseMaxPoolingHead
             (vectorised torch.topk, no Python loops)
                                │
                        [B, num_classes]   ← bag logits

    Args:
        cfg : :class:`~src.config.ModelConfig`
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.extractor = MILFeatureExtractor(
            token_dim=cfg.token_dim,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
        )
        self.pooling_head = ClassWiseMaxPoolingHead(
            hidden_dim=cfg.hidden_dim,
            num_classes=cfg.num_classes,
            top_k=cfg.top_k,
            dropout=cfg.dropout,
        )

        self._apply_init(cfg.init_strategy)

    def _apply_init(self, strategy: str) -> None:
        """Apply a weight-initialisation strategy to all linear layers."""
        if strategy == "default":
            return
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if strategy == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight)
                elif strategy == "kaiming_normal":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-instance (or flat-batch) forward pass.

        Args:
            tokens : ``[B, T, D]`` – CLIP token embeddings.

        Returns:
            logits : ``[B, num_classes]``
            probs  : ``[B, num_classes]``  (sigmoid of logits)
        """
        features = self.extractor(tokens)          # [B, hidden_dim]
        logits, _ = self.pooling_head(features)    # [B, num_classes]
        probs = torch.sigmoid(logits)
        return logits, probs

    def forward_bags(
        self, bag_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        MIL forward pass over bags of instances.

        Args:
            bag_tokens : ``[B, N, T, D]`` – *B* bags of *N* instances each.

        Returns:
            bag_logits  : ``[B, num_classes]``
            bag_probs   : ``[B, num_classes]``
            inst_scores : ``[B, N, num_classes]``
        """
        B, N, T, D = bag_tokens.shape
        features = self.extractor(bag_tokens.view(B * N, T, D))  # [B*N, H]
        bag_features = features.view(B, N, -1)                   # [B, N, H]

        bag_logits, inst_scores = self.pooling_head.forward_bags(bag_features)
        bag_probs = torch.sigmoid(bag_logits)
        return bag_logits, bag_probs, inst_scores
