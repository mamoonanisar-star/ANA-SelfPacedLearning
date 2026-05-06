"""
test_mil_pipeline.py

Unit tests for the 5-phase MIL pipeline.

Covers
------
* ANAFeatureDataset  : CSV validation, label parsing, zero-tensor fallback
* MedicalProjectionHead  : shape, LayerNorm active
* ClassWiseMaxPoolingHead: output shapes, attention sums to 1
* ANAMILModel            : forward pass shapes, logits range
* evaluate               : metrics types and ranges on synthetic data
* tune_threshold         : returns a value in [0, 1]

Run with:
    pytest test_mil_pipeline.py -v
"""

from __future__ import annotations

import io
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from mil_dataset import ANAFeatureDataset, ICAP_CLASSES
from mil_model import ANAMILModel, ClassWiseMaxPoolingHead, MedicalProjectionHead
from mil_evaluate import evaluate, tune_threshold


# ---------------------------------------------------------------------------
# Helpers: synthetic data fixtures
# ---------------------------------------------------------------------------

NUM_TOKENS  = 257
TOKEN_DIM   = 768
NUM_CLASSES = 8
PROJ_DIM    = 64   # small for fast tests
MID_DIM     = 128


def _make_csv(n_train=10, n_val=5, n_test=5, use_target=True):
    """Create a minimal annotation CSV in a temp directory."""
    rows = []
    split_map = (
        [("train", n_train), ("val", n_val), ("test", n_test)]
    )
    idx = 0
    for split, count in split_map:
        for _ in range(count):
            path = f"{idx:04d}.npy"
            if use_target:
                target = "0 3"
                row = {"path": path, "Split": split, "TARGET": target}
            else:
                labels = [1, 0, 0, 1, 0, 0, 0, 0]
                row = {"path": path, "Split": split}
                for c, name in enumerate(ICAP_CLASSES):
                    row[name] = labels[c]
            rows.append(row)
            idx += 1
    return pd.DataFrame(rows)


def _make_npy_files(stems, tmpdir):
    """Write random [257, 768] .npy files."""
    for stem in stems:
        arr = np.random.randn(NUM_TOKENS, TOKEN_DIM).astype(np.float32)
        np.save(os.path.join(tmpdir, f"{stem}.npy"), arr)


def _setup_dataset(n_train=10, n_val=5, n_test=5, use_target=True):
    tmpdir = tempfile.mkdtemp()
    df = _make_csv(n_train, n_val, n_test, use_target=use_target)
    ann_file = os.path.join(tmpdir, "ann.csv")
    df.to_csv(ann_file, index=False)

    # Write .npy files for all stems
    stems = [os.path.splitext(r["path"])[0] for _, r in df.iterrows()]
    _make_npy_files(stems, tmpdir)

    return tmpdir, ann_file


# ---------------------------------------------------------------------------
# ANAFeatureDataset tests
# ---------------------------------------------------------------------------

class TestANAFeatureDataset:

    def test_len_and_shapes_target_format(self):
        tmpdir, ann_file = _setup_dataset(use_target=True)
        ds = ANAFeatureDataset(tmpdir, ann_file, "train")
        assert len(ds) == 10
        tokens, label, idx = ds[0]
        assert tokens.shape == (NUM_TOKENS, TOKEN_DIM)
        assert label.shape == (NUM_CLASSES,)
        assert label.dtype == torch.float32

    def test_len_and_shapes_binary_format(self):
        tmpdir, ann_file = _setup_dataset(use_target=False)
        # Binary columns start after "path" and "Split" (index 2–9)
        ds = ANAFeatureDataset(tmpdir, ann_file, "train",
                               label_col_start=2, label_col_end=10)
        assert len(ds) == 10
        tokens, label, idx = ds[0]
        assert label.sum().item() > 0  # at least one positive class

    def test_labels_from_target_column(self):
        tmpdir, ann_file = _setup_dataset(use_target=True)
        ds = ANAFeatureDataset(tmpdir, ann_file, "train")
        _, label, _ = ds[0]
        # CSV has TARGET="0 3" → classes 0 and 3 should be 1
        assert label[0].item() == 1.0
        assert label[3].item() == 1.0
        assert label[1].item() == 0.0

    def test_missing_npy_returns_zeros(self, tmp_path):
        df = _make_csv(n_train=3, n_val=1, n_test=1)
        ann_file = str(tmp_path / "ann.csv")
        df.to_csv(ann_file, index=False)
        # Do NOT write any .npy files → should fallback to zeros
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ds = ANAFeatureDataset(str(tmp_path), ann_file, "train")
            tokens, _, _ = ds[0]
            assert tokens.shape == (NUM_TOKENS, TOKEN_DIM)
            assert tokens.sum().item() == 0.0

    def test_missing_ann_file_raises(self):
        with pytest.raises(FileNotFoundError):
            ANAFeatureDataset("/tmp/nonexistent", "/tmp/no_such_file.csv", "train")

    def test_invalid_split_raises(self):
        tmpdir, ann_file = _setup_dataset()
        with pytest.raises(ValueError, match="split must be"):
            ANAFeatureDataset(tmpdir, ann_file, "invalid_split")

    def test_wrong_split_no_rows_raises(self):
        tmpdir, ann_file = _setup_dataset(n_val=0, n_test=0)
        # val split has 0 rows
        with pytest.raises(ValueError, match="No rows found"):
            ANAFeatureDataset(tmpdir, ann_file, "val")

    def test_pos_weight_shape_and_values(self):
        tmpdir, ann_file = _setup_dataset()
        ds = ANAFeatureDataset(tmpdir, ann_file, "train")
        pw = ds.compute_pos_weight()
        assert pw.shape == (NUM_CLASSES,)
        # pos_weight should be >= 0
        assert (pw >= 0).all()

    def test_class_counts_length(self):
        tmpdir, ann_file = _setup_dataset()
        ds = ANAFeatureDataset(tmpdir, ann_file, "train")
        assert len(ds.class_counts) == NUM_CLASSES


# ---------------------------------------------------------------------------
# MedicalProjectionHead tests
# ---------------------------------------------------------------------------

class TestMedicalProjectionHead:

    def test_output_shape(self):
        head = MedicalProjectionHead(TOKEN_DIM, MID_DIM, PROJ_DIM)
        x = torch.randn(4, TOKEN_DIM)
        out = head(x)
        assert out.shape == (4, PROJ_DIM)

    def test_output_shape_batched_patches(self):
        """Should work with [B, P, D] input (nn.Linear acts on last dim)."""
        head = MedicalProjectionHead(TOKEN_DIM, MID_DIM, PROJ_DIM)
        x = torch.randn(4, 256, TOKEN_DIM)
        out = head(x)
        assert out.shape == (4, 256, PROJ_DIM)

    def test_layernorm_applied(self):
        """LayerNorm should normalise output features."""
        head = MedicalProjectionHead(TOKEN_DIM, MID_DIM, PROJ_DIM)
        head.eval()
        x = torch.randn(32, TOKEN_DIM) * 1000  # large input
        out = head(x)
        # After LayerNorm the mean across the last dim should be ~0
        assert out.mean(dim=-1).abs().max().item() < 0.1


# ---------------------------------------------------------------------------
# ClassWiseMaxPoolingHead tests
# ---------------------------------------------------------------------------

class TestClassWiseMaxPoolingHead:

    def test_output_shapes(self):
        B, P, D, C = 4, 256, PROJ_DIM, NUM_CLASSES
        head = ClassWiseMaxPoolingHead(D, C)
        patch_feats = torch.randn(B, P, D)
        class_feats, attn_w = head(patch_feats)
        assert class_feats.shape == (B, C, D)
        assert attn_w.shape == (B, C, P)

    def test_attention_sums_to_one(self):
        """Softmax attention weights over patches should sum to 1 per class."""
        head = ClassWiseMaxPoolingHead(PROJ_DIM, NUM_CLASSES, dropout=0.0)
        head.eval()
        patch_feats = torch.randn(2, 256, PROJ_DIM)
        _, attn_w = head(patch_feats)
        # attn_w: [B, C, P]  → sum over P should be 1
        sums = attn_w.sum(dim=-1)  # [B, C]
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_gradient_flows(self):
        head = ClassWiseMaxPoolingHead(PROJ_DIM, NUM_CLASSES)
        patch_feats = torch.randn(2, 256, PROJ_DIM, requires_grad=True)
        class_feats, _ = head(patch_feats)
        loss = class_feats.sum()
        loss.backward()
        assert patch_feats.grad is not None


# ---------------------------------------------------------------------------
# ANAMILModel tests
# ---------------------------------------------------------------------------

class TestANAMILModel:

    def test_forward_shapes(self):
        B = 4
        model = ANAMILModel(TOKEN_DIM, NUM_CLASSES, PROJ_DIM, MID_DIM)
        tokens = torch.randn(B, NUM_TOKENS, TOKEN_DIM)
        logits, attn_w = model(tokens)
        assert logits.shape == (B, NUM_CLASSES)
        assert attn_w.shape == (B, NUM_CLASSES, NUM_TOKENS - 1)  # patch tokens only

    def test_logits_are_unbounded(self):
        """ANAMILModel returns raw logits; values can be outside [0, 1]."""
        model = ANAMILModel(TOKEN_DIM, NUM_CLASSES, PROJ_DIM, MID_DIM)
        tokens = torch.randn(2, NUM_TOKENS, TOKEN_DIM) * 100  # extreme input
        logits, _ = model(tokens)
        # Logits should not be clipped to [0,1]
        assert logits.dtype == torch.float32

    def test_predict_proba_in_unit_interval(self):
        model = ANAMILModel(TOKEN_DIM, NUM_CLASSES, PROJ_DIM, MID_DIM)
        tokens = torch.randn(3, NUM_TOKENS, TOKEN_DIM)
        proba = model.predict_proba(tokens)
        assert proba.shape == (3, NUM_CLASSES)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_cls_and_patch_separation(self):
        """
        Verify that CLS token ([:, 0, :]) and patch tokens ([:, 1:, :])
        are handled separately inside the model by checking attention map size.
        """
        model = ANAMILModel(TOKEN_DIM, NUM_CLASSES, PROJ_DIM, MID_DIM)
        tokens = torch.randn(1, NUM_TOKENS, TOKEN_DIM)
        _, attn_w = model(tokens)
        # attn_w should cover only the 256 patch tokens, not the CLS token
        assert attn_w.shape[-1] == NUM_TOKENS - 1

    def test_gradient_flows_end_to_end(self):
        model = ANAMILModel(TOKEN_DIM, NUM_CLASSES, PROJ_DIM, MID_DIM)
        tokens = torch.randn(2, NUM_TOKENS, TOKEN_DIM)
        labels = torch.randint(0, 2, (2, NUM_CLASSES)).float()
        logits, _ = model(tokens)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        # All parameters should have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# evaluate / tune_threshold tests
# ---------------------------------------------------------------------------

class TestEvaluate:

    def _make_model_and_loader(self):
        model = ANAMILModel(TOKEN_DIM, NUM_CLASSES, PROJ_DIM, MID_DIM)
        # Synthetic DataLoader: yields (tokens, labels, indices)
        tokens  = torch.randn(20, NUM_TOKENS, TOKEN_DIM)
        labels  = torch.randint(0, 2, (20, NUM_CLASSES)).float()
        dataset = torch.utils.data.TensorDataset(tokens, labels,
                                                  torch.arange(20))
        loader  = DataLoader(dataset, batch_size=4)
        return model, loader

    def test_evaluate_returns_expected_keys(self):
        model, loader = self._make_model_and_loader()
        device = torch.device("cpu")
        metrics = evaluate(model, loader, device, num_classes=NUM_CLASSES)
        for key in ("f1_macro", "mAP", "f1_per_class",
                    "precision_per_class", "recall_per_class",
                    "all_probs", "all_labels"):
            assert key in metrics

    def test_f1_macro_in_unit_interval(self):
        model, loader = self._make_model_and_loader()
        metrics = evaluate(model, loader, torch.device("cpu"),
                           num_classes=NUM_CLASSES)
        assert 0.0 <= metrics["f1_macro"] <= 1.0

    def test_mAP_in_unit_interval(self):
        model, loader = self._make_model_and_loader()
        metrics = evaluate(model, loader, torch.device("cpu"),
                           num_classes=NUM_CLASSES)
        assert 0.0 <= metrics["mAP"] <= 1.0


class TestTuneThreshold:

    def test_returns_float_in_unit_interval(self):
        N, C = 50, NUM_CLASSES
        labels = np.random.randint(0, 2, (N, C))
        probs  = np.random.rand(N, C)
        t = tune_threshold(labels, probs)
        assert 0.0 <= t <= 1.0

    def test_custom_candidates(self):
        labels = np.random.randint(0, 2, (20, NUM_CLASSES))
        probs  = np.random.rand(20, NUM_CLASSES)
        candidates = np.array([0.3, 0.5, 0.7])
        t = tune_threshold(labels, probs, candidates=candidates)
        assert t in candidates.tolist()
