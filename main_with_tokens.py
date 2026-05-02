"""
main_with_tokens.py

Self-paced learning training script with optional CLIP token mode.

When --use_tokens is set, loads pre-computed [num_tokens x token_dim] .npy
files instead of raw images, and runs the ANATokenClassifier.

When --use_tokens is NOT set, falls back to the standard ResNet-50 pipeline
(identical to main.py) so both modes can be benchmarked from one script.

Usage
-----
# Token mode
python main_with_tokens.py \\
    --use_tokens \\
    --tokens_root ./data/clip_tokens/ \\
    --num_tokens 257 --token_dim 768 \\
    --saveModel \\
    --lr 1e-3 --weight_lr 1e-3 \\
    --initWeight iw-sample --updateLR ulr-adaptive \\
    --granularity label --sampling --trainingLabel pseudo

# ResNet-50 baseline (same results as main.py)
python main_with_tokens.py --saveModel --lr 1e-3
"""

import os
import sys
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm

from config import (
    EPOCH, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, MOMENTUM,
    GPU_IN_USE, NUM_CATEGORIES, N,
    PATH_TRAIN_ANNFILE, PATH_TEST_ANNFILE,
    DIR_TRAIN_IMAGES, DIR_TEST_IMAGES,
    of1, cf1, best_mi, best_ma, best_acc, best_mAP,
    TOKENS_ROOT, NUM_TOKENS, TOKEN_DIM, TOKEN_HIDDEN_DIM, TOKEN_DROPOUT, CONF_ALPHA,
)
from utils import (
    set_random_seed, Logger, inference, test_small, test, save,
    get_train_dataloader, save_results_to_json,
)
from data_weighted_filename import get_traindata, get_testdata
from models_with_tokens import ANATokenClassifier, TokenWeightedBCELoss

set_random_seed(42, True)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="ANA Self-Paced Learning – with optional CLIP token mode",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
# Shared args (same as main.py)
parser.add_argument("--lr",             default=LEARNING_RATE, type=float)
parser.add_argument("--epoch",          default=EPOCH,         type=int)
parser.add_argument("--trainBatchSize", default=BATCH_SIZE,    type=int)
parser.add_argument("--testBatchSize",  default=BATCH_SIZE,    type=int)
parser.add_argument("--weightDecay",    default=WEIGHT_DECAY,  type=float)
parser.add_argument("--pathModelParams",default=None,          type=str)
parser.add_argument("--saveModel",      action="store_true")
parser.add_argument("--loadModel",      action="store_true")
parser.add_argument("--weight_lr",      default=LEARNING_RATE, type=float)
parser.add_argument("--initWeight",     default="iw-ones",     type=str)
parser.add_argument("--updateLR",       default="ulr-ones",    type=str)
parser.add_argument("--trainingLabel",  default="real",        type=str)
parser.add_argument("--sampling",       action="store_true")
parser.add_argument("--granularity",    default="label",       type=str)
# Token-specific args
parser.add_argument("--use_tokens",     action="store_true",
                    help="Use pre-computed CLIP token .npy files instead of images")
parser.add_argument("--tokens_root",    default=TOKENS_ROOT, type=str,
                    help="Root directory containing .npy token files")
parser.add_argument("--num_tokens",     default=NUM_TOKENS,     type=int,
                    help="Number of tokens per patch (e.g. 257 for CLIP ViT-L/14)")
parser.add_argument("--token_dim",      default=TOKEN_DIM,      type=int,
                    help="Token embedding dimension (e.g. 768)")
parser.add_argument("--token_hidden",   default=TOKEN_HIDDEN_DIM, type=int,
                    help="Hidden dimension for token attention / confidence head")
parser.add_argument("--token_dropout",  default=TOKEN_DROPOUT,  type=float,
                    help="Dropout probability for token attention / confidence head")
parser.add_argument("--conf_alpha",     default=CONF_ALPHA,     type=float,
                    help="Weight of region confidence in the loss (0 = SPL only)")
parser.add_argument("--num_patches",    default=20,            type=int,
                    help="Expected patches per large image (informational only)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------
mode_tag = "token" if args.use_tokens else "resnet50"
if args.pathModelParams is None:
    PATH_MODEL_PARAMS = (
        f"{mode_tag}_"
        + "_".join([
            str(args.lr), "lr",
            str(args.weight_lr), "wlr",
            args.initWeight, args.updateLR,
            args.trainingLabel, str(args.sampling), args.granularity,
        ])
    )
else:
    PATH_MODEL_PARAMS = args.pathModelParams

if args.saveModel:
    os.makedirs(PATH_MODEL_PARAMS, exist_ok=True)

sys.stdout = Logger(f"{PATH_MODEL_PARAMS}/output.txt")
print(f"Mode: {'TOKEN (CLIP)' if args.use_tokens else 'IMAGE (ResNet-50)'}")
print(f"Output dir: {PATH_MODEL_PARAMS}")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

if args.use_tokens:
    from data_clip_tokens import (
        get_traindata as get_token_traindata,
        get_testdata  as get_token_testdata,
    )
    train_dataset = get_token_traindata(
        tokens_root=args.tokens_root,
        annFile=PATH_TRAIN_ANNFILE,
        split="train",
        num_tokens=args.num_tokens,
        token_dim=args.token_dim,
        num_classes=NUM_CATEGORIES,
    )
    val_dataset = get_token_testdata(
        tokens_root=args.tokens_root,
        annFile=PATH_TEST_ANNFILE,
        split="val",
        num_tokens=args.num_tokens,
        token_dim=args.token_dim,
        num_classes=NUM_CATEGORIES,
    )
    print(f"Token dataset: train={len(train_dataset)}, val={len(val_dataset)}")
else:
    normalize = transforms.Normalize(
        mean=[0.005, 0.190, 0.006],
        std=[0.008, 0.102, 0.008],
    )
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((N, N)),
        transforms.ToTensor(),
        normalize,
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((N, N)),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = get_traindata(
        root=DIR_TRAIN_IMAGES,
        annFile=PATH_TRAIN_ANNFILE,
        transform=train_transforms,
        split="train",
    )
    val_dataset = get_testdata(
        root=DIR_TEST_IMAGES,
        annFile=PATH_TEST_ANNFILE,
        transform=val_transforms,
        split="val",
    )
    print(f"Image dataset: train={len(train_dataset)}, val={len(val_dataset)}")

val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=args.testBatchSize,
    shuffle=False,
    num_workers=4,
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

if args.use_tokens:
    model = ANATokenClassifier(
        token_dim=args.token_dim,
        num_tokens=args.num_tokens,
        num_classes=NUM_CATEGORIES,
        hidden_dim=args.token_hidden,
        dropout=args.token_dropout,
    )
    print(
        f"ANATokenClassifier | tokens={args.num_tokens} × dim={args.token_dim} "
        f"→ classes={NUM_CATEGORIES}"
    )
else:
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, NUM_CATEGORIES)
    print("ResNet-50 baseline")

if GPU_IN_USE:
    model.to(device)

print("Model Preparation : Finished")

# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

if args.use_tokens:
    loss_function = TokenWeightedBCELoss(
        granularity=args.granularity,
        alpha=args.conf_alpha,
    )
else:
    # Identical to the WeightedBCELoss in main.py
    class WeightedBCELoss(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, input, target, weight):
            bce = F.binary_cross_entropy(input, target, reduction="none")
            if args.granularity == "sample":
                bce *= F.relu(weight.nan_to_num())[:, None]
            else:
                bce[target == 1] *= F.relu(weight[target == 1].nan_to_num())
            return bce.sum()
    loss_function = WeightedBCELoss()

# ---------------------------------------------------------------------------
# Weight initialisation (mirrors main.py exactly)
# ---------------------------------------------------------------------------

t_start = time.time()
train_loader_init = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=args.trainBatchSize,
    num_workers=4,
)

print("Initialize weights")
labels = torch.zeros((len(train_dataset), NUM_CATEGORIES), dtype=torch.float64)
label_filenames = np.array([""] * len(train_dataset)).astype("U256")
weight_idx: list = []
label_types: list = []

for _, (_, label, idx, filenames) in tqdm(
    enumerate(train_loader_init),
    total=len(train_loader_init),
    desc="Processing Train Data",
):
    labels[idx] = label
    label_filenames[idx] = filenames

    for i, l1 in enumerate(label):
        break_flag = False
        for j, l2 in enumerate(label_types):
            if (l1 - l2).sum() == 0:
                break_flag = True
                weight_idx[j].append(idx[i])
                break
        if not break_flag:
            label_types.append(l1)
            weight_idx.append([idx[i]])

print("label_types", torch.stack(label_types), torch.stack(label_types).sum(dim=1).max())

targets = labels.clone()

if args.updateLR == "ulr-ones":
    learning_rates = torch.ones(len(label_types))
else:
    learning_rates = (
        torch.log(torch.stack(label_types).sum(dim=1))
        / torch.log(torch.stack(label_types).sum(dim=1).max())
    ).cuda()

print("learning_rates", learning_rates)
weight_idx_flatten = torch.tensor([ww for w in weight_idx for ww in w])
group_idx = torch.argsort(weight_idx_flatten)

# --- Weight init strategies (iw-ones / iw-data / iw-sample) ---
def _slice(tensor, i):
    start = sum(map(len, weight_idx[:i]))
    return tensor[start : start + len(weight_idx[i])]

if args.initWeight == "iw-ones":
    if args.granularity == "sample":
        weights = [_slice(torch.ones(len(train_dataset)), i) for i in range(len(weight_idx))]
    else:
        weights = [_slice(labels[weight_idx_flatten], i) for i in range(len(weight_idx))]

elif args.initWeight == "iw-data":
    if args.granularity == "sample":
        iw = F.softmax(torch.tensor(list(map(len, weight_idx))) / len(train_dataset))
        weights = [torch.ones(len(weight_idx[i])) * iw[i] for i in range(len(weight_idx))]
    else:
        iw = labels.sum(dim=0)
        iw /= iw.sum()
        iw = labels * iw
        iw[labels == 0] = float("-inf")
        iw = F.softmax(iw)
        weights = [_slice((labels * iw)[weight_idx_flatten], i) for i in range(len(weight_idx))]

elif args.initWeight == "iw-sample":
    iw = torch.zeros((len(train_dataset), NUM_CATEGORIES), dtype=torch.float64)
    for filename in np.unique(label_filenames):
        iw_sub = labels[label_filenames == filename].sum(dim=0)
        iw_sub /= iw_sub.sum()
        iw[label_filenames == filename] = iw_sub
    weights = [_slice(iw[weight_idx_flatten], i) for i in range(len(weight_idx))]

else:
    print("No valid initialisation method for weights")
    sys.exit(1)

print("Initialization Done", time.time() - t_start)

# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------

weight_lr = args.weight_lr
weight_params = []
for group_i, (p, l) in enumerate(zip(weights, learning_rates)):
    p.requires_grad = True

    # Capture group_i in closure so each hook reports the correct group index
    def _make_grad_hook(gi):
        def _hook(grad):
            nan_count = torch.isnan(grad).sum().item()
            if nan_count > 0:
                print(f"  [WARNING] NaN gradient in weight group {gi}: {nan_count} NaN values")
        return _hook

    p.register_hook(_make_grad_hook(group_i))
    weight_params.append({"params": p, "lr": float(l) * weight_lr})

optimizer = optim.Adam(
    [{"params": model.parameters()}] + weight_params,
    lr=args.lr,
    weight_decay=args.weightDecay,
    betas=MOMENTUM,
)

# ---------------------------------------------------------------------------
# Token-aware training step
# ---------------------------------------------------------------------------

def train_token_epoch(optimizer, train_loader, weights, model, group_idx, loss_fn, targets_store):
    """One training epoch for the token-based classifier."""
    model.train()
    train_loss = 0.0
    print(f"len dataloader {len(train_loader)}")

    for _, (tokens, label, index, filenames) in tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc="Training Progress",
    ):
        if label.sum() == 0:
            continue

        # Gather SPL weights for this batch
        weight = []
        for i in index:
            offset = 0
            for w in weights:
                if group_idx[i] < offset + len(w):
                    weight.append(w[group_idx[i] - offset])
                    break
                offset += len(w)
        weight = torch.stack(weight)

        if GPU_IN_USE:
            tokens = tokens.to(device)
            label  = label.to(device)
            weight = weight.to(device)

        target = label.float()
        if args.trainingLabel != "real":
            if args.granularity == "sample":
                target = label * F.relu(weight.detach().nan_to_num())[:, None]
            else:
                target = label * F.relu(weight.detach().nan_to_num())
            target = (target - target.min(dim=1, keepdim=True)[0]) / (
                target.max(dim=1, keepdim=True)[0] - target.min(dim=1, keepdim=True)[0]
            )
            target[torch.isnan(target)] = 0
            if args.trainingLabel == "pseudo":
                target = target.detach().bernoulli()
            targets_store[index] = target.detach().cpu()
            target = target.float()

        optimizer.zero_grad()
        _, probs, confidence, _ = model(tokens)
        loss = loss_fn(probs, target, weight, confidence)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f"train done | loss={train_loss:.4f}")
    # Return 6 placeholder values matching the signature of utils.train for
    # compatibility with any callers that expect (cp, cr, cf1, op, or, of1).
    return 1, 1, 1, 1, 1, 1


def test_small_token(val_loader, model, device):
    """Validation loop for token-based classifier; mirrors utils.test_small."""
    import sklearn.metrics as sm
    print(f"val: {len(val_loader)}")
    model.eval()

    all_probs, all_labels, all_preds = [], [], []

    for _, (tokens, target, _, _) in tqdm(
        enumerate(val_loader),
        total=len(val_loader),
        desc="Validation Progress",
    ):
        if target.sum() == 0:
            continue
        if GPU_IN_USE:
            tokens, target = tokens.to(device), target.to(device)

        with torch.no_grad():
            _, probs, _, _ = model(tokens)

        probs_np  = probs.detach().cpu().numpy()
        preds_np  = (probs_np > 0.5).astype(float)
        target_np = target.detach().cpu().numpy()

        all_probs.append(probs_np)
        all_preds.append(preds_np)
        all_labels.append(target_np)

    all_probs  = np.vstack(all_probs)
    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    mAP      = sm.average_precision_score(all_labels, all_probs)
    acc_tot  = sm.accuracy_score(all_labels, all_preds)
    f1_mi    = sm.f1_score(all_labels, all_preds, average="micro")
    f1_ma    = sm.f1_score(all_labels, all_preds, average="macro")

    print(
        f"Val | acc={acc_tot:.4f}  f1_mi={f1_mi:.4f}  "
        f"f1_ma={f1_ma:.4f}  mAP={mAP:.4f}"
    )
    return acc_tot, f1_mi, f1_ma, mAP


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

num_early_stop = 5
count_early_stop = 0

for current_epoch in range(1, args.epoch + 1):
    print(f"\n===> epoch: {current_epoch}/{args.epoch}")

    # Build weighted dataloader
    with torch.no_grad():
        combined_w = torch.cat(weights)[group_idx]
        if args.granularity == "sample":
            sw = F.relu(combined_w.nan_to_num())
        else:
            sw = (F.relu(combined_w.nan_to_num()) * labels).max(dim=1)[0]

        sampler = None
        if args.sampling:
            sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.trainBatchSize,
            num_workers=4,
            sampler=sampler,
        )

    # --- one training epoch ---
    if args.use_tokens:
        train_token_epoch(
            optimizer, train_loader, weights, model,
            group_idx, loss_function, targets,
        )
    else:
        from utils import train as train_image
        train_image(
            optimizer, train_loader, weights, args, model,
            group_idx, loss_function, targets, device,
        )

    # --- adaptive LR update (mirrors main.py) ---
    with torch.no_grad():
        if args.updateLR == "ulr-adaptive":
            weight_idx = []
            label_types = []
            for i, l1 in enumerate((targets > 0).float()):
                break_flag = False
                for j, l2 in enumerate(label_types):
                    if (l1 - l2).sum() == 0:
                        break_flag = True
                        weight_idx[j].append(i)
                        break
                if not break_flag:
                    label_types.append(l1)
                    weight_idx.append([i])

            print("label_types", torch.stack(label_types),
                  torch.stack(label_types).sum(dim=1).max())
            weight_idx_flatten = torch.tensor([ww for w in weight_idx for ww in w])
            weights = [
                _slice(torch.cat(weights)[group_idx][weight_idx_flatten], i)
                for i in range(len(weight_idx))
            ]
            learning_rates = (
                torch.log(torch.stack(label_types).sum(dim=1))
                / torch.log(torch.stack(label_types).sum(dim=1).max())
            ).cuda()
            learning_rates[torch.isinf(learning_rates)] = 0
            learning_rates[torch.isnan(learning_rates)] = 0
            group_idx = torch.argsort(weight_idx_flatten)

            weight_params = []
            for grp_i, (p, l) in enumerate(zip(weights, learning_rates)):
                p.requires_grad = True
                weight_params.append({"params": p, "lr": float(l) * weight_lr})
            optimizer = optim.Adam(
                [{"params": model.parameters()}] + weight_params,
                lr=args.lr,
                weight_decay=args.weightDecay,
                betas=MOMENTUM,
            )

        # --- validation ---
        if args.use_tokens:
            acc_total, f1_mi, f1_ma, mAP = test_small_token(val_loader, model, device)
        else:
            acc_total, f1_mi, f1_ma, mAP = test_small(val_loader, model, device)

    # --- checkpoint & early stopping ---
    improved = False
    if f1_ma > of1 and f1_mi > cf1:
        of1, cf1 = f1_ma, f1_mi
    if f1_mi > best_mi:
        if args.saveModel:
            save(f"{PATH_MODEL_PARAMS}/{current_epoch}_f1mi.pt", model)
        best_mi = f1_mi
        improved = True
    if f1_ma > best_ma:
        if args.saveModel:
            save(f"{PATH_MODEL_PARAMS}/{current_epoch}_f1ma.pt", model)
        best_ma = f1_ma
        improved = True
    if acc_total > best_acc:
        if args.saveModel:
            save(f"{PATH_MODEL_PARAMS}/{current_epoch}_acc.pt", model)
        best_acc = acc_total
        improved = True
    if mAP > best_mAP:
        if args.saveModel:
            save(f"{PATH_MODEL_PARAMS}/{current_epoch}_mAP.pt", model)
        best_mAP = mAP
        improved = True

    if improved:
        count_early_stop = 0
    else:
        count_early_stop += 1
    if count_early_stop >= num_early_stop:
        print("Early stopping triggered.")
        break

    val_results = {
        "acc_total": acc_total,
        "f1_mi": f1_mi,
        "f1_ma": f1_ma,
        "mAP": mAP,
    }

    print(f"===> BEST PERFORMANCE (OF1/CF1): {of1:.3f} / {cf1:.3f}")
    print(f"===> BEST PERFORMANCE (mi/ma): {best_mi:.3f} / {best_ma:.3f}")
    print(f"===> BEST PERFORMANCE (acc): {best_acc:.3f}")
    print(f"===> BEST PERFORMANCE (mAP): {best_mAP:.3f}")

    if count_early_stop == 0:
        if not args.use_tokens:
            with torch.no_grad():
                acc_0, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7, acc_total, f1_mi, f1_ma, mAP = test(model, args, device)
                test_results = {
                    "acc_total": acc_total,
                    "f1_mi": f1_mi,
                    "f1_ma": f1_ma,
                    "mAP": mAP,
                }
                save_results_to_json(
                    current_epoch, val_results, test_results,
                    f"{PATH_MODEL_PARAMS}/results.json",
                )

    t_use = int((time.time() - t_start) / 60)
    print(f"Use {t_use}/min")

print(PATH_MODEL_PARAMS)
