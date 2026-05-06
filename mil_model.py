"""
mil_model.py

Phase 2 – Token-Aware MIL Architecture.

Components
----------
MedicalProjectionHead    : 768 → 1024 → 512 projection with ReLU, Dropout,
                           LayerNorm; shared for CLS and patch tokens.
ClassWiseMaxPoolingHead  : Learns class-specific attention over patch tokens
                           so each class "finds" its most relevant patches.
ANAMILModel              : Full model combining CLS anchor + class-wise
                           pooled patches → multi-label logits.

Tensor conventions
------------------
  tokens      : [B, num_tokens, token_dim]
  CLS token   : tokens[:, 0, :]   – global image representation
  patch tokens: tokens[:, 1:, :]  – 16×16 spatial patch representations

Design notes
------------
* BCEWithLogitsLoss (caller-side) is more numerically stable than BCE + sigmoid,
  so ANAMILModel returns raw logits (not probabilities).
* ClassWiseMaxPoolingHead uses soft (learned, softmax-normalised) attention
  rather than hard top-k pooling; this is differentiable end-to-end while
  still concentrating weight on the most relevant patches.
* The final per-class decision fuses the global CLS feature with the
  class-specific patch feature via a shared binary linear head applied
  independently to each class slot.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# MedicalProjectionHead
# ---------------------------------------------------------------------------

class MedicalProjectionHead(nn.Module):
    """
    Two-layer projection network: in_dim → mid_dim → out_dim.

    Default pathway: 768 → 1024 → 512 (ReLU + Dropout after each linear,
    LayerNorm on the output).

    The projection is shared between the CLS token and the patch tokens, so
    both live in the same semantic space before pooling.

    Args:
        in_dim  (int): Input feature dimension (768 for CLIP ViT-L/14).
        mid_dim (int): Intermediate hidden dimension (default 1024).
        out_dim (int): Output projection dimension (default 512).
        dropout (float): Dropout probability (default 0.2).
    """

    def __init__(
        self,
        in_dim: int = 768,
        mid_dim: int = 1024,
        out_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., in_dim]  (any number of leading dimensions)
        Returns:
            [..., out_dim]
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# ClassWiseMaxPoolingHead
# ---------------------------------------------------------------------------

class ClassWiseMaxPoolingHead(nn.Module):
    """
    Class-specific soft attention pooling over patch tokens.

    For each class *c*, learns a separate attention network that scores every
    patch token.  The class-specific feature is a softmax-weighted sum of the
    projected patch representations.

    Implementation
    --------------
    A single linear layer maps each patch feature to ``num_classes`` scores
    (one per class), which is equivalent to having ``num_classes`` independent
    dot-product attention queries — but computed in a single matrix multiply
    for efficiency.

    Args:
        proj_dim    (int): Projected patch feature dimension (output of
                           MedicalProjectionHead, default 512).
        num_classes (int): Number of output classes (default 8).
        dropout     (float): Dropout on attention weights (default 0.1).
    """

    def __init__(
        self,
        proj_dim: int = 512,
        num_classes: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Maps each patch embedding to a score for every class
        self.class_attention = nn.Linear(proj_dim, num_classes, bias=True)
        self.attn_drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.class_attention.weight)
        nn.init.zeros_(self.class_attention.bias)

    def forward(self, patch_features: torch.Tensor):
        """
        Args:
            patch_features: Float tensor [B, num_patches, proj_dim]
        Returns:
            class_features : Float tensor [B, num_classes, proj_dim]
                             class_features[:, c, :] is the soft-attended
                             representation for class *c*.
            attn_weights   : Float tensor [B, num_classes, num_patches]
                             Interpretable: high weight → patch is relevant
                             for that class.
        """
        # [B, num_patches, num_classes] – one score per (patch, class) pair
        attn_scores = self.class_attention(patch_features)
        # [B, num_classes, num_patches] – transpose for bmm
        attn_scores = attn_scores.transpose(1, 2)
        # Normalise over patches so weights sum to 1 per class
        attn_weights = F.softmax(attn_scores, dim=-1)      # [B, C, P]
        attn_weights = self.attn_drop(attn_weights)
        # Weighted sum of patches → one feature vector per class
        # [B, C, P] × [B, P, D] → [B, C, D]
        class_features = torch.bmm(attn_weights, patch_features)
        return class_features, attn_weights


# ---------------------------------------------------------------------------
# ANAMILModel
# ---------------------------------------------------------------------------

class ANAMILModel(nn.Module):
    """
    Multi-Instance Learning model for ANA fluorescence pattern classification.

    Architecture
    ------------
    1. CLS token   → MedicalProjectionHead → [B, proj_dim]          (global)
    2. Patch tokens→ MedicalProjectionHead (shared) →               (local)
                     [B, num_patches, proj_dim]
    3. Patch features → ClassWiseMaxPoolingHead →                   (class-specific)
                        [B, num_classes, proj_dim]
    4. For each class *c*:
           [CLS_feat ; class_c_feat]  ∈ R^{2*proj_dim}
       →   shared binary linear head  → logit_c
    5. Output: logits [B, num_classes]

    The CLS feature acts as a global anchor that anchors each class decision
    in the full-image context, while the class-wise pooled feature provides
    locally-discriminative information for that specific class.

    Args:
        token_dim   (int): Input token dimension (768 for ViT-L/14 @ 768-d).
        num_classes (int): Number of output classes (default 8).
        proj_dim    (int): Projected feature dimension (default 512).
        mid_dim     (int): Intermediate dimension in MedicalProjectionHead
                           (default 1024).
        attn_dropout(float): Dropout inside ClassWiseMaxPoolingHead (default 0.1).
        proj_dropout(float): Dropout inside MedicalProjectionHead (default 0.2).
    """

    def __init__(
        self,
        token_dim: int = 768,
        num_classes: int = 8,
        proj_dim: int = 512,
        mid_dim: int = 1024,
        attn_dropout: float = 0.1,
        proj_dropout: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.proj_dim = proj_dim

        # Shared projection for CLS and patch tokens
        self.proj = MedicalProjectionHead(token_dim, mid_dim, proj_dim, proj_dropout)

        # Class-specific attention pooling over projected patches
        self.class_pool = ClassWiseMaxPoolingHead(proj_dim, num_classes, attn_dropout)

        # Shared binary classifier applied independently to each class slot:
        #   input  [B*C, 2*proj_dim]  →  output [B*C, 1]
        self.classifier = nn.Linear(proj_dim * 2, 1)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, tokens: torch.Tensor):
        """
        Args:
            tokens: Float tensor [B, num_tokens, token_dim]
                    tokens[:, 0, :] must be the CLS token.
                    tokens[:, 1:, :] are the patch tokens (256 for ViT-L/14).

        Returns:
            logits     : Float tensor [B, num_classes]  – raw class logits.
                         Pass to BCEWithLogitsLoss (NOT to sigmoid first).
            attn_weights: Float tensor [B, num_classes, num_patches]
                          Class-specific patch attention (interpretable).
        """
        # -----------------------------------------------------------------
        # 1. Split CLS token from patch tokens
        # -----------------------------------------------------------------
        cls_tokens   = tokens[:, 0, :]    # [B, token_dim]
        patch_tokens = tokens[:, 1:, :]   # [B, num_patches, token_dim]

        # -----------------------------------------------------------------
        # 2. Project into shared feature space
        # -----------------------------------------------------------------
        cls_feat    = self.proj(cls_tokens)    # [B, proj_dim]
        # self.proj supports arbitrary leading dims (nn.Linear acts on last dim)
        patch_feats = self.proj(patch_tokens)  # [B, num_patches, proj_dim]

        # -----------------------------------------------------------------
        # 3. Class-wise pooling over patches
        # -----------------------------------------------------------------
        class_feats, attn_weights = self.class_pool(patch_feats)
        # class_feats : [B, num_classes, proj_dim]
        # attn_weights: [B, num_classes, num_patches]

        # -----------------------------------------------------------------
        # 4. Fuse CLS global feature with class-specific patch feature
        # -----------------------------------------------------------------
        # Expand CLS to match class dimension
        B, C, D = class_feats.shape
        cls_expanded = cls_feat.unsqueeze(1).expand(B, C, D)  # [B, C, proj_dim]
        combined = torch.cat([cls_expanded, class_feats], dim=-1)  # [B, C, 2*proj_dim]

        # Apply shared binary head to each class slot independently
        combined_flat = combined.view(B * C, 2 * D)          # [B*C, 2*proj_dim]
        logits_flat   = self.classifier(combined_flat)        # [B*C, 1]
        logits        = logits_flat.view(B, C)                # [B, num_classes]

        return logits, attn_weights

    def predict_proba(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities [B, num_classes] (no grad)."""
        with torch.no_grad():
            logits, _ = self.forward(tokens)
            return torch.sigmoid(logits)
