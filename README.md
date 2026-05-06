# Self-Paced Learning for Images of Antinuclear Antibodies

Official PyTorch implementation of **"Self-Paced Learning for Images of Antinuclear Antibodies"** published in **IEEE Transactions on Medical Imaging (TMI)**.

This repository contains the code for multi-label classification of Antinuclear Antibodies (ANA) images using self-paced learning with adaptive sample weighting and pseudo-label training.



## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

- Place image files in `./data/images/` directory
- Prepare CSV annotation file with the following columns:
  - `path`: Image path
  - `TARGET`: Labels (space-separated numbers, e.g., "0 3 5")
  - `Split`: Dataset split (train/val/test)

### 3. Update Configuration

Edit `config.py` or `config_single.py` to update data paths:

```python
DIR_TRAIN_IMAGES = './data/images/'
DIR_TEST_IMAGES = './data/images/'
PATH_TRAIN_ANNFILE = 'your_annotations.csv'
PATH_TEST_ANNFILE = 'your_annotations.csv'
```

### 4. Train Model

**Full training command (recommended):**

```bash
python main.py --saveModel --lr 1e-3 --weight_lr 1e-3 --initWeight iw-sample --updateLR ulr-adaptive --granularity label --sampling --trainingLabel pseudo
```

**Simple training command:**

```bash
python main.py --saveModel --lr 1e-3
```

**Single-label training:**

```bash
python main_single.py --saveModel --lr 1e-3
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--lr` | Model learning rate | 5e-3 |
| `--weight_lr` | Sample weight learning rate | 5e-3 |
| `--epoch` | Number of epochs | 200 |
| `--trainBatchSize` | Training batch size | 32 |
| `--saveModel` | Save model checkpoints | False |
| `--initWeight` | Weight initialization (iw-ones/iw-data/iw-sample) | iw-ones |
| `--updateLR` | Learning rate update strategy (ulr-ones/ulr-adaptive) | ulr-ones |
| `--trainingLabel` | Training label type (real/pseudo) | real |
| `--granularity` | Weight granularity (label/sample) | label |
| `--sampling` | Enable weighted sampling | False |

## Features

- Multi-label classification
- Adaptive sample weighting
- Pseudo-label training
- Weighted random sampling
- Early stopping mechanism
- Multiple evaluation metrics (Accuracy, F1-score, mAP)

## Output Files

Training generates the following files:
- `{model_name}/output.txt` - Training log
- `{model_name}/{epoch}_f1mi.pt` - Best micro-F1 model
- `{model_name}/{epoch}_f1ma.pt` - Best macro-F1 model
- `{model_name}/{epoch}_acc.pt` - Best accuracy model
- `{model_name}/{epoch}_mAP.pt` - Best mAP model
- `{model_name}/results.json` - Evaluation results

## Requirements

- Python 3.7+
- PyTorch 1.7+
- CUDA (recommended for GPU acceleration)

See `requirements.txt` for full dependencies.

## Notes

- Supports standard image formats (jpg, png, etc.)
- Ensure paths in config files point to your data
- GPU will be used automatically if available
- CSV file must contain `path`, `TARGET`, `Split` columns


## License

This code is released for academic research use only.

---

## MIL Pipeline — CLIP ViT-L/14 Embeddings

A validated, production-ready **Multi-Instance Learning (MIL)** pipeline for
ANA fluorescence pattern classification using pre-computed CLIP ViT-L/14
embeddings (`[257, 768]` per sample: 1 CLS token + 256 patch tokens).

### New modules

| File | Description |
|------|-------------|
| `mil_dataset.py` | `ANAFeatureDataset` — CSV validation, multi-label loading, class-imbalance stats, zero-tensor fallback |
| `mil_model.py` | `MedicalProjectionHead` · `ClassWiseMaxPoolingHead` · `ANAMILModel` |
| `mil_train.py` | AdamW training loop with BCEWithLogitsLoss, pos_weight, grad clipping, early stopping |
| `mil_evaluate.py` | F1-Macro, mAP, per-class metrics, confusion matrix, threshold tuning |
| `mil_inference.py` | Demo inference with clinical report |
| `test_mil_pipeline.py` | 25 unit tests (run with `pytest test_mil_pipeline.py -v`) |

### Architecture overview

```
tokens [B, 257, 768]
  ├─ CLS token  [:, 0, :]  ──► MedicalProjectionHead (768→1024→512) ──► cls_feat  [B, 512]
  └─ Patch toks [:, 1:, :] ──► MedicalProjectionHead (shared)       ──► patch_feats [B, 256, 512]
                                        │
                               ClassWiseMaxPoolingHead
                               (per-class softmax attention)
                                        │
                               class_feats [B, 8, 512]
                                        │
  For each class c:  [cls_feat ; class_c_feat] → Linear(1024, 1) → logit_c
```

### Quickstart

**Step 1 — Extract CLIP tokens** (if not already done)

```bash
python extract_clip_tokens.py \
    --annFile ./data/features_index_english.csv \
    --img_root ./data/images/ \
    --out_dir  ./data/clip_tokens/ \
    --model    ViT-L/14
```

**Step 2 — Train**

```bash
python mil_train.py \
    --ann_file    ./data/features_index_english.csv \
    --tokens_root ./data/clip_tokens/ \
    --epochs 50 --batch_size 16 --lr 1e-4 \
    --weight_decay 0.05 --patience 10 \
    --checkpoint_dir ./checkpoints/
```

**Step 3 — Evaluate** (with optional threshold tuning)

```bash
python mil_evaluate.py \
    --ann_file    ./data/features_index_english.csv \
    --tokens_root ./data/clip_tokens/ \
    --checkpoint  ./checkpoints/mil_best.pt \
    --split test \
    --tune_threshold
```

**Step 4 — Demo inference**

```bash
python mil_inference.py \
    --ann_file    ./data/features_index_english.csv \
    --tokens_root ./data/clip_tokens/ \
    --checkpoint  ./checkpoints/mil_best.pt
```

### CSV label format

Two formats are supported:

| Format | Description | Columns |
|--------|-------------|---------|
| **TARGET** (sparse) | Space-separated class indices | `path`, `Split`, `TARGET` |
| **Binary** (ICAP)   | One column per class (0/1)   | `path`, `Split`, `ICAP_AC`, `ICAP_CF`, … (8 columns) |

For binary format, pass `--label_col_start <N>` where `N` is the zero-based
column index of the first ICAP label column (default: 16).  The dataset
validates that `N` through `N+8` fall within the CSV.

### Design decisions & hyperparameter guidance

| Decision | Rationale |
|----------|-----------|
| **BCEWithLogitsLoss** (not BCE) | Numerically stable log-sum-exp formulation. |
| **pos_weight** per class | Computed from training set; corrects class imbalance automatically. |
| **Early stopping on F1-Macro** | Treats all 8 classes equally; robust to class imbalance. |
| **lr = 1e-4** | AdamW fine-tuning range for CLIP-derived features. Increase to 3e-4 if convergence is slow. |
| **weight_decay = 0.05** | Standard decoupled AdamW value; lower to 0.01 if under-fitting. |
| **batch_size = 16** | Safe for 12 GB GPU; increase to 32+ if memory allows. |
| **grad_clip = 1.0** | Prevents gradient explosions with LayerNorm + attention. |
| **Threshold = 0.5 → tunable** | `mil_evaluate.py --tune_threshold` searches 0.10–0.90 on the val set. |

