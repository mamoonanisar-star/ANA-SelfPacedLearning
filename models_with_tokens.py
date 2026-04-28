"""
models_with_tokens.py

Token-aware architecture for ANA multi-label classification.

Core components:
  - TokenAttentionPooling : Learns to weight CLIP token sequence → 1 vector
  - RegionConfidenceEstimator : Scores each patch's relevance (0-1)
  - ANATokenClassifier : End-to-end model for [B, num_tokens, token_dim] input

Token counts depend on the CLIP model used, e.g.:
  - ViT-L/14: 257 tokens (1 CLS + 16×16 spatial) × 1024 dim
  - ViT-B/16: 197 tokens (1 CLS + 14×14 spatial) × 768 dim
  - ViT-B/32: 50  tokens (1 CLS + 7×7 spatial)   × 512 dim
Both num_tokens and token_dim are fully configurable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenAttentionPooling(nn.Module):
    """
    Weighted aggregation of CLIP token sequence → single feature vector.

    The CLS token (index 0) acts as a query; the remaining patch tokens
    act as keys/values.  A learned projection produces per-token attention
    weights so the network can focus on diagnostically relevant tokens.

    Args:
        token_dim   (int): Embedding dimension of each token (e.g. 768).
        hidden_dim  (int): Dimension of the attention projection layer.
        dropout     (float): Dropout probability on attention weights.
    """

    def __init__(self, token_dim: int = 768, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.token_dim = token_dim
        # Query from CLS token
        self.query_proj = nn.Linear(token_dim, hidden_dim)
        # Key from all tokens
        self.key_proj = nn.Linear(token_dim, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.scale = hidden_dim ** -0.5

    def forward(self, tokens: torch.Tensor):
        """
        Args:
            tokens: Float tensor [B, num_tokens, token_dim]
                    tokens[:, 0, :] is expected to be the CLS token.
        Returns:
            pooled : Float tensor [B, token_dim]  – attended representation
            attn_w : Float tensor [B, num_tokens] – attention weights (interpretable)
        """
        cls = tokens[:, 0:1, :]           # [B, 1, token_dim]
        q = self.query_proj(cls)          # [B, 1, hidden_dim]
        k = self.key_proj(tokens)         # [B, num_tokens, hidden_dim]
        # Scaled dot-product attention
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale  # [B, 1, num_tokens]
        attn_w = F.softmax(scores, dim=-1)                      # [B, 1, num_tokens]
        attn_w = self.attn_drop(attn_w)
        pooled = torch.bmm(attn_w, tokens).squeeze(1)           # [B, token_dim]
        return pooled, attn_w.squeeze(1)                        # [B, token_dim], [B, num_tokens]


class RegionConfidenceEstimator(nn.Module):
    """
    Estimates a scalar confidence score (0-1) for each patch, indicating
    how much information the patch contributes to the classification.

    A high score means the patch contains diagnostically relevant structure;
    a low score suggests the patch is background/noise and should be
    down-weighted by the self-paced learning loss.

    Architecture:
        tokens → attention pooling → 2-layer MLP → sigmoid confidence

    Args:
        token_dim  (int): Token embedding dimension.
        hidden_dim (int): MLP hidden dimension.
        attn_hidden(int): Attention pooling hidden dimension.
        dropout    (float): Dropout on the MLP.
    """

    def __init__(
        self,
        token_dim: int = 768,
        hidden_dim: int = 128,
        attn_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn_pool = TokenAttentionPooling(token_dim, attn_hidden, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens: torch.Tensor):
        """
        Args:
            tokens: Float tensor [B, num_tokens, token_dim]
        Returns:
            confidence : Float tensor [B]        – patch confidence (0-1)
            attn_weights: Float tensor [B, num_tokens] – token importance
        """
        pooled, attn_w = self.attn_pool(tokens)   # [B, token_dim], [B, num_tokens]
        confidence = torch.sigmoid(self.mlp(pooled)).squeeze(-1)  # [B]
        return confidence, attn_w


class ANATokenClassifier(nn.Module):
    """
    End-to-end token-based classifier for ANA multi-label classification.

    Takes pre-computed CLIP token embeddings ([B, num_tokens, token_dim]) and
    predicts multi-label class probabilities, while simultaneously estimating
    per-patch region confidence for self-paced learning.

    Architecture:
        tokens → TokenAttentionPooling → LayerNorm → dropout → Linear(num_classes)
        tokens → RegionConfidenceEstimator → confidence score

    Args:
        token_dim   (int): CLIP token embedding dimension (e.g. 768 for ViT-L/14).
        num_tokens  (int): Number of tokens per patch (e.g. 257 = 1 CLS + 256 spatial).
        num_classes (int): Number of output classes (8 for ANA dataset).
        hidden_dim  (int): Hidden dimension for attention / confidence MLP.
        dropout     (float): Dropout probability.
    """

    def __init__(
        self,
        token_dim: int = 768,
        num_tokens: int = 257,
        num_classes: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.num_tokens = num_tokens
        self.num_classes = num_classes

        self.attn_pool = TokenAttentionPooling(token_dim, hidden_dim, dropout)
        self.norm = nn.LayerNorm(token_dim)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(token_dim, num_classes)

        self.confidence_head = RegionConfidenceEstimator(
            token_dim=token_dim,
            hidden_dim=hidden_dim // 2,
            attn_hidden=hidden_dim,
            dropout=dropout,
        )

    def forward(self, tokens: torch.Tensor):
        """
        Args:
            tokens: Float tensor [B, num_tokens, token_dim]

        Returns:
            logits      : Float tensor [B, num_classes]  – raw class scores
            probs       : Float tensor [B, num_classes]  – sigmoid probabilities
            confidence  : Float tensor [B]               – patch confidence (0-1)
            attn_weights: Float tensor [B, num_tokens]   – token attention weights
        """
        pooled, attn_w = self.attn_pool(tokens)     # [B, token_dim]
        pooled = self.norm(pooled)
        pooled = self.drop(pooled)
        logits = self.classifier(pooled)             # [B, num_classes]
        probs = torch.sigmoid(logits)                # [B, num_classes]
        confidence, _ = self.confidence_head(tokens) # [B]
        return logits, probs, confidence, attn_w

    def get_features(self, tokens: torch.Tensor):
        """Return pooled feature representation without classification head."""
        pooled, attn_w = self.attn_pool(tokens)
        return self.norm(pooled), attn_w


class TokenWeightedBCELoss(nn.Module):
    """
    Self-paced weighted BCE loss that incorporates both:
      1. Instance weights  (learned by the SPL framework, as in the baseline)
      2. Region confidence (from RegionConfidenceEstimator)

    Final per-sample weight = relu(instance_weight) * (1 + alpha * confidence)

    Args:
        granularity (str): 'sample' or 'label' — mirrors the baseline option.
        alpha       (float): Contribution of confidence term (0 disables it).
    """

    def __init__(self, granularity: str = "label", alpha: float = 0.5):
        super().__init__()
        self.granularity = granularity
        self.alpha = alpha

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        confidence: torch.Tensor,
    ):
        """
        Args:
            input      : [B, C] – model output probabilities
            target     : [B, C] – binary ground-truth labels
            weight     : [B] or [B, C] – SPL instance / label weights
            confidence : [B]   – per-patch confidence from RegionConfidenceEstimator
        Returns:
            loss : scalar
        """
        bce = F.binary_cross_entropy(input, target, reduction="none")  # [B, C]
        relu_w = F.relu(weight.nan_to_num())
        conf_factor = 1.0 + self.alpha * confidence.detach()           # [B]

        if self.granularity == "sample":
            combined = relu_w * conf_factor                             # [B]
            bce = bce * combined[:, None]
        else:
            combined = relu_w * conf_factor[:, None]                   # [B, C]
            bce[target == 1] = bce[target == 1] * combined[target == 1]

        return bce.sum()
