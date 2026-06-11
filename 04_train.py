"""
PHASE 4: The Training Loop — Putting It All Together
=====================================================
This is the heart of every deep learning project.
All the pieces from previous phases come together here:
  Phase 1 → tensors
  Phase 2 → data pipeline (DataLoader)
  Phase 3 → model, loss, optimiser

Training strategy (two stages):
  Stage 1 — Head only (5 epochs)
    Backbone is frozen. Only our classifier head trains.
    Fast, cheap. Gets us to ~50-60% accuracy quickly.

  Stage 2 — Fine-tuning (10 epochs)
    Unfreeze the last 3 blocks of EfficientNet.
    Train everything at a much smaller LR.
    Pushes accuracy to 75-80%+.

Checkpointing: we save the best model (by val accuracy) throughout.
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from torchvision.datasets import Food101
from torchvision.models import EfficientNet_B2_Weights
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import json
import time

# ── Local import ──────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from models.food_classifier import FoodClassifier   # we'll create this module


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — change these to experiment
# ─────────────────────────────────────────────────────────────────────────────

CFG = {
    "data_dir"        : "data",
    "checkpoint_dir"  : "checkpoints",
    "results_dir"     : "results",
    "num_classes"     : 101,
    "batch_size"      : 32,
    "num_workers"     : 2,

    # Stage 1: train head only
    "stage1_epochs"   : 5,
    "stage1_lr"       : 3e-4,

    # Stage 2: fine-tune backbone
    "stage2_epochs"   : 10,
    "stage2_lr"       : 5e-5,      # much smaller — backbone weights are fragile

    "weight_decay"    : 1e-4,
    "label_smoothing" : 0.1,
    "seed"            : 42,
}

# Reproducibility — set seeds so results are consistent across runs
torch.manual_seed(CFG["seed"])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}\n")

Path(CFG["checkpoint_dir"]).mkdir(exist_ok=True)
Path(CFG["results_dir"]).mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transforms = T.Compose([
    T.Resize((256, 256)),
    T.RandomCrop(224),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

print("Loading datasets...")
train_dataset = Food101(CFG["data_dir"], split="train", transform=train_transforms, download=False)
val_dataset   = Food101(CFG["data_dir"], split="test",  transform=val_transforms,  download=False)

train_loader = DataLoader(train_dataset, batch_size=CFG["batch_size"],
                          shuffle=True,  num_workers=CFG["num_workers"], pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=CFG["batch_size"],
                          shuffle=False, num_workers=CFG["num_workers"], pin_memory=True)

print(f"Train: {len(train_dataset):,} images | Val: {len(val_dataset):,} images\n")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

model = FoodClassifier(num_classes=CFG["num_classes"], freeze_backbone=True)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"])


# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimiser, device):
    """
    One full pass over the training set.
    Returns: average loss, top-1 accuracy
    """
    model.train()   # IMPORTANT: enables dropout, batch norm in training mode
    total_loss, correct, total = 0.0, 0, 0

    # tqdm wraps any iterable and draws a live progress bar
    pbar = tqdm(loader, desc="  train", leave=False, unit="batch")

    for images, labels in pbar:
        # 1. Move data to the same device as the model
        images, labels = images.to(device), labels.to(device)

        # 2. Zero gradients — if you forget this, gradients accumulate across
        #    batches and your training will be wrong. Classic beginner bug.
        optimiser.zero_grad()

        # 3. Forward pass
        logits = model(images)              # [batch, 101] raw scores

        # 4. Compute loss
        loss = criterion(logits, labels)

        # 5. Backward pass — compute gradients for all trainable params
        loss.backward()

        # 6. Gradient clipping — if gradients explode (very large values),
        #    this caps them. Prevents training instability.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 7. Optimiser step — apply gradients (gradient descent)
        optimiser.step()

        # ── Metrics ──
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)        # pick class with highest score
        correct += (preds == labels).sum().item()
        total += images.size(0)

        pbar.set_postfix(loss=f"{loss.item():.3f}",
                         acc=f"{100*correct/total:.1f}%")

    return total_loss / total, correct / total


@torch.no_grad()    # decorator: disables gradient tracking for the whole function
def evaluate(model, loader, criterion, device):
    """
    Evaluate on validation set. No gradient computation — faster + less memory.
    Returns: average loss, top-1 accuracy
    """
    model.eval()    # disables dropout, uses running stats for batch norm
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  val  ", leave=False, unit="batch"):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def run_stage(stage_num, epochs, lr, unfreeze_blocks=None):
    """Run a training stage and return history."""

    print(f"\n{'='*55}")
    print(f"  STAGE {stage_num}  |  {epochs} epochs  |  lr={lr}")
    print(f"{'='*55}")

    if unfreeze_blocks is not None:
        model.unfreeze_backbone(unfreeze_last_n_blocks=unfreeze_blocks)

    optimiser = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=CFG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=epochs, eta_min=lr * 0.01
    )

    history = []
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimiser, DEVICE)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        elapsed = time.time() - t0
        current_lr = optimiser.param_groups[0]["lr"]

        print(f"  train loss={train_loss:.4f}  acc={100*train_acc:.2f}%"
              f"  |  val loss={val_loss:.4f}  acc={100*val_acc:.2f}%"
              f"  |  lr={current_lr:.2e}  [{elapsed:.0f}s]")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = Path(CFG["checkpoint_dir"]) / f"best_stage{stage_num}.pt"
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "val_acc"    : val_acc,
                "val_loss"   : val_loss,
                "cfg"        : CFG,
            }, ckpt_path)
            print(f"  ✓ New best! Saved → {ckpt_path}  (val_acc={100*val_acc:.2f}%)")

        history.append({
            "epoch": epoch, "stage": stage_num,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss,     "val_acc": val_acc,
        })

    print(f"\nStage {stage_num} complete. Best val acc: {100*best_val_acc:.2f}%")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────────────────────

all_history = []

# Stage 1: train head only — backbone frozen
all_history += run_stage(
    stage_num=1,
    epochs=CFG["stage1_epochs"],
    lr=CFG["stage1_lr"],
)

# Stage 2: unfreeze last 3 backbone blocks + fine-tune
all_history += run_stage(
    stage_num=2,
    epochs=CFG["stage2_epochs"],
    lr=CFG["stage2_lr"],
    unfreeze_blocks=3,
)

# Save full training history as JSON (used for plotting in the next phase)
history_path = Path(CFG["results_dir"]) / "training_history.json"
with open(history_path, "w") as f:
    json.dump(all_history, f, indent=2)
print(f"\nTraining history saved → {history_path}")

print("\n✅ Training complete! Next: Grad-CAM visualisation.")
