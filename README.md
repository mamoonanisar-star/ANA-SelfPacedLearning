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

### 5. Token-based training (CLIP embeddings)

**Step 1: Extract CLIP tokens**

```bash
python extract_clip_tokens.py \
    --annFile  all_single_small_82240_19400_19330.csv \
    --img_root ./data/images/ \
    --out_dir  ./data/clip_tokens/ \
    --model    ViT-L/14
```

This writes one `.npy` file per patch and mirrors any subdirectory structure
from the CSV `path` column (e.g., `sub/0001.jpg` → `./data/clip_tokens/sub/0001.npy`).

**Step 2: Configure token shapes**

Update `config.py` to match your CLIP export:

```python
TOKENS_ROOT = "./data/clip_tokens/"
NUM_TOKENS = 257
TOKEN_DIM = 768
TOKEN_HIDDEN_DIM = 256
TOKEN_DROPOUT = 0.1
CONF_ALPHA = 0.5
```

**Step 3: Train with tokens**

```bash
python main_with_tokens.py \
    --use_tokens \
    --tokens_root ./data/clip_tokens/ \
    --num_tokens 257 \
    --token_dim 768 \
    --token_hidden 256 \
    --token_dropout 0.1 \
    --conf_alpha 0.5 \
    --saveModel \
    --lr 1e-3
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

**Token-specific parameters (main_with_tokens.py):**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--use_tokens` | Use pre-computed CLIP token files | False |
| `--tokens_root` | Root directory of `.npy` token files | `./data/clip_tokens/` |
| `--num_tokens` | Tokens per patch | 257 |
| `--token_dim` | Token embedding dimension | 768 |
| `--token_hidden` | Hidden dimension for attention/confidence head | 256 |
| `--token_dropout` | Dropout for token attention/confidence head | 0.1 |
| `--conf_alpha` | Region confidence weight in loss | 0.5 |

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

