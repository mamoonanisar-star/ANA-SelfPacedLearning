# Methodology: Token-Level Region Suppression for ANA Image Classification

## Research Question

> Can foundation model token embeddings ([257 × 768] CLIP tokens) improve the
> Instance Sampler's ability to suppress low-confidence/irrelevant sub-regions
> in multi-patch ANA image analysis compared to the ResNet-50 baseline?

---

## 1. Background and Motivation

Antinuclear Antibody (ANA) images are large high-resolution fluorescence
microscopy slides.  A single slide is too large to be fed directly into a
convolutional or transformer network, so the **Instance Sampler** (implemented
as `get_small()` in `utils.py`) divides each slide into ≈20 non-overlapping
448 × 448 sub-regions, each later resized to 224 × 224 for the classifier.

The original Self-Paced Learning (SPL) framework learns scalar importance
weights for each patch using `WeightedBCELoss`, suppressing noisy or
irrelevant patches over training epochs.  However, the baseline uses
**ResNet-50** features — a convolutional global-pooling representation that
discards all spatial structure within a patch.

**Our hypothesis**: CLIP's Vision Transformer produces **N token embeddings**
per patch (1 CLS token + H × W spatial tokens; e.g. 1 + 16 × 16 = 257 for
ViT-L/14, or 1 + 14 × 14 = 197 for ViT-B/16).  These token embeddings encode
fine-grained spatial structure and support *interpretable* region-level
analysis, which should allow the SPL framework to make better suppression
decisions.

---

## 2. Token-Aware Architecture

### 2.1 TokenAttentionPooling

Aggregates the 257-token sequence into a single representation using
CLS-attended soft pooling:

```
q = W_q(CLS)                         [B, 1, H]
k = W_k(tokens)                      [B, T, H]
a = softmax(q · kᵀ / √H)             [B, 1, T]
z = a · tokens                        [B, token_dim]
```

The attention weights **a** over 257 positions form an **interpretable importance
map** — position 0 is the CLS token and positions 1–256 correspond to a 16 × 16
spatial grid within the patch.

### 2.2 RegionConfidenceEstimator

A lightweight MLP on top of the attention-pooled representation produces a
scalar **confidence score** ∈ (0, 1) per patch:

```
confidence = σ( MLP( TokenAttentionPool(tokens) ) )
```

Patches with low confidence (background, out-of-focus regions) receive
reduced effective weight in the loss.

### 2.3 ANATokenClassifier

End-to-end model:

```
Input:  [B, 257, 768]  (pre-computed CLIP tokens per patch)
  ↓ TokenAttentionPooling
  ↓ LayerNorm → Dropout
  ↓ Linear(token_dim → 8)
Output: logits [B, 8] + sigmoid probs [B, 8]
        confidence [B]
        attention weights [B, 257]
```

---

## 3. Token-Weighted Self-Paced Loss

The baseline `WeightedBCELoss` applies SPL instance weights *w_i* to each
sample.  We extend this to a **TokenWeightedBCELoss** that incorporates the
region confidence:

```
effective_weight = relu(w_i) × (1 + α × confidence_i)
```

where α is a hyperparameter (default: 0.5) controlling the contribution of
the confidence signal relative to the learned SPL weights.

When α = 0 the loss reduces exactly to the baseline `WeightedBCELoss`.

For **label-granularity** weights (as in the best baseline configuration):

```
loss = Σ_{i,c : target_{i,c}=1}  BCE(p_{i,c}, y_{i,c}) × effective_weight_{i,c}
```

---

## 4. Data Pipeline

### 4.1 Token File Convention

Pre-computed CLIP tokens are stored as **NumPy arrays** (`.npy`) with shape
`[num_tokens, token_dim]`, e.g. `[257, 768]` for ViT-L/14.

Supported layouts:

| Layout | Path pattern | Example |
|--------|-------------|---------|
| Flat | `tokens_root/<stem>.npy` | `./data/clip_tokens/0001.npy` |
| Hierarchical | `tokens_root/<parent>/<stem>.npy` | `./data/clip_tokens/patient01/0001.npy` |

The `ANATokenDataset` class auto-detects the layout and falls back to a
zero tensor with a warning when a file is missing.

### 4.2 Token Extraction (`extract_clip_tokens.py`)

Hooks the **last transformer block** of the CLIP visual encoder to capture
the full token sequence (not just the pooled CLS output).  This gives richer
spatial information than `model.encode_image()`.

Compatible with both `openai/clip` and `open-clip-torch`.

---

## 5. Training Protocol

| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| LR (model) | 1e-3 |
| LR (weights) | 1e-3 |
| Weight init | `iw-sample` |
| LR update | `ulr-adaptive` |
| Granularity | `label` |
| Weighted sampling | ✓ |
| Training labels | `pseudo` |
| Random seed | 42 |
| Early stopping patience | 5 epochs |

The SPL weight initialisation (`iw-sample`) groups patches from the same
parent image and initialises their weights proportional to the per-label
class frequency within that image, exactly matching the baseline.

---

## 6. Evaluation

Metrics computed on the validation split at each epoch, and on the held-out
test split when a new best is achieved:

| Metric | Description |
|--------|-------------|
| Accuracy | Exact-match multi-label accuracy |
| F1-micro | Micro-averaged F1 across all classes |
| F1-macro | Macro-averaged F1 across all classes |
| mAP | Mean Average Precision (primary metric) |

Additional suppression metrics produced by `analyze_tokens.py`:

- **Suppression rate**: fraction of patches with effective weight < 0.1
- **Confidence distribution**: histogram of per-patch confidence scores
- **Token importance map**: mean attention weight per token position
- **CLS vs spatial ratio**: compares CLS vs spatial token contributions

---

## 7. Benchmark Comparison Protocol

To compare CLIP token embeddings against ResNet-50 and ResNet-101 baselines
on identical data splits and SPL settings:

```bash
# 1. ResNet-50 baseline
python main.py --saveModel --lr 1e-3 --weight_lr 1e-3 \
    --initWeight iw-sample --updateLR ulr-adaptive \
    --granularity label --sampling --trainingLabel pseudo

# 2. Token-based CLIP
python main_with_tokens.py \
    --use_tokens \
    --tokens_root ./data/clip_tokens/ \
    --num_tokens 257 --token_dim 768 \
    --saveModel --lr 1e-3 --weight_lr 1e-3 \
    --initWeight iw-sample --updateLR ulr-adaptive \
    --granularity label --sampling --trainingLabel pseudo

# 3. Generate comparison table
python analyze_tokens.py \
    --compare resnet50:./resnet50_run/results.json \
              clip_token:./token_run/results.json \
    --out_dir ./comparison/
```

---

## 8. Expected Deliverables

| File | Purpose |
|------|---------|
| `models_with_tokens.py` | Token classifier + region confidence estimator |
| `data_clip_tokens.py` | CLIP token dataset loader |
| `main_with_tokens.py` | Unified training script (image or token mode) |
| `analyze_tokens.py` | Visualisation and suppression metrics |
| `extract_clip_tokens.py` | Utility to extract tokens from images |
| `config.py` (updated) | Token-specific hyperparameters |
| `METHODOLOGY.md` | This document |

---

## 9. Interpretability Insights

The 257-dimensional token attention weights provide interpretable evidence
for the model's suppression decisions:

- **CLS token (index 0)**: captures global patch-level semantics; a high CLS
  attention weight indicates the patch is globally salient.
- **Spatial tokens (1–256)**: correspond to a 16 × 16 non-overlapping patch
  grid within the input patch; high attention on a specific spatial token
  localises the discriminative region within the patch.
- **Region Confidence**: patches with fine-nuclear-detail patterns should
  receive high confidence, while background or out-of-focus patches should
  receive low confidence, verifiable visually via the suppression map.

---

## 10. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Missing token file | Zero tensor fallback + warning in `ANATokenDataset` |
| Shape mismatch | Automatic pad/truncate in `_reshape_tokens` |
| Memory: large token tensors | Smaller default batch sizes for token mode |
| No improvement over baseline | Interpretability still a valid contribution; ablate α |
| CLIP library unavailable | `open-clip-torch` fallback in `extract_clip_tokens.py` |
