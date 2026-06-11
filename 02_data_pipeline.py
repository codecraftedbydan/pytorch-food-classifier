"""
PHASE 2: Datasets & DataLoaders — Feeding Real Data to a Model
==============================================================
A model is only as good as its data pipeline.
This file covers:
  - torchvision.datasets  : pre-built loaders for famous datasets
  - transforms            : how we pre-process images before feeding them in
  - DataLoader            : the engine that batches + shuffles data efficiently
  - exploring Food-101    : the dataset we'll train our classifier on

Food-101: 101 food categories, 1000 images per class = 101,000 images total.
Split: 75,750 train / 25,250 test (provided by the dataset authors).
Source: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
"""

import torch
import torchvision
import torchvision.transforms.v2 as T
from torchvision.datasets import Food101
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

# ── Where to store the downloaded data ───────────────────────────────────────
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. TRANSFORMS — preprocessing images before they enter the model
# ─────────────────────────────────────────────────────────────────────────────
#
# Raw images come in all shapes and sizes. Neural networks need:
#   - A fixed size (we'll use 224x224 — the ImageNet standard)
#   - Pixel values in a known range (normalised to mean=0, std=1 per channel)
#
# For TRAINING we also add random augmentations — these artificially increase
# dataset diversity and stop the model from memorising specific photos.
#
# For VALIDATION/TEST we only resize + normalise — no randomness, we want
# deterministic evaluation.

# ImageNet normalisation constants — used because our backbone was pretrained
# on ImageNet. These values are the mean and std of all ImageNet pixels.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transforms = T.Compose([
    T.Resize((256, 256)),               # resize shortest side to 256
    T.RandomCrop(224),                  # randomly crop a 224x224 patch
    T.RandomHorizontalFlip(p=0.5),      # 50% chance of flipping horizontally
    T.ColorJitter(brightness=0.3,       # randomly tweak brightness/contrast
                  contrast=0.3,
                  saturation=0.2),
    T.ToImage(),                        # convert PIL → tensor-friendly format
    T.ToDtype(torch.float32, scale=True),  # pixels → float32 in [0, 1]
    T.Normalize(mean=IMAGENET_MEAN,     # normalise to ImageNet stats
                std=IMAGENET_STD),
])

val_transforms = T.Compose([
    T.Resize((224, 224)),               # just resize, no randomness
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATASETS — download + load Food-101
# ─────────────────────────────────────────────────────────────────────────────
print("Downloading Food-101 dataset (this will take a few minutes the first time)...")
print("~500MB download — saved to ./data and reused on future runs.\n")

train_dataset = Food101(
    root=DATA_DIR,
    split="train",          # 75,750 images
    transform=train_transforms,
    download=True,
)

val_dataset = Food101(
    root=DATA_DIR,
    split="test",           # 25,250 images (used as validation here)
    transform=val_transforms,
    download=False,         # already downloaded above
)

print(f"Train samples : {len(train_dataset):,}")
print(f"Val samples   : {len(val_dataset):,}")
print(f"Num classes   : {len(train_dataset.classes)}")
print(f"First 10 classes: {train_dataset.classes[:10]}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATALOADERS — batching, shuffling, parallel loading
# ─────────────────────────────────────────────────────────────────────────────
#
# You never feed images one-by-one to a model — that's too slow.
# A DataLoader groups them into "batches" and handles:
#   - shuffle=True  : randomise order each epoch (prevents the model learning
#                     the order of the data instead of the content)
#   - num_workers   : how many CPU threads to use for loading images in parallel
#   - pin_memory    : speeds up GPU transfer (safe to set True even on CPU)

BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,          # never shuffle validation — keeps results reproducible
    num_workers=2,
    pin_memory=True,
)

print(f"Batch size     : {BATCH_SIZE}")
print(f"Train batches  : {len(train_loader):,}  ({len(train_dataset):,} images / {BATCH_SIZE})")
print(f"Val batches    : {len(val_loader):,}  ({len(val_dataset):,} images / {BATCH_SIZE})\n")


# ─────────────────────────────────────────────────────────────────────────────
# 4. INSPECT A BATCH — what the model actually sees
# ─────────────────────────────────────────────────────────────────────────────

# Grab one batch from the training loader
images, labels = next(iter(train_loader))

print("── One batch from the DataLoader ──")
print(f"images tensor shape : {images.shape}")
#  [32, 3, 224, 224]
#   ^   ^   ^    ^
#   |   |   |    └─ width  (pixels)
#   |   |   └────── height (pixels)
#   |   └────────── channels: R, G, B
#   └────────────── batch size: 32 images at once

print(f"labels tensor shape : {labels.shape}")   # [32] — one class index per image
print(f"labels dtype        : {labels.dtype}")   # int64 — required by CrossEntropyLoss
print(f"pixel value range   : [{images.min():.2f}, {images.max():.2f}]")
print(f"                      (normalised — not [0,1] anymore)\n")

# Show the class names for this batch
batch_class_names = [train_dataset.classes[l.item()] for l in labels[:8]]
print(f"First 8 labels: {batch_class_names}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 5. VISUALISE — plot a grid of images from the batch
# ─────────────────────────────────────────────────────────────────────────────

def denormalise(tensor):
    """Reverse the ImageNet normalisation so we can display the image."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)

fig = plt.figure(figsize=(14, 6))
fig.suptitle("Sample images from Food-101 (with training augmentations applied)",
             fontsize=13, fontweight="bold")

for i in range(16):
    ax = fig.add_subplot(2, 8, i + 1)
    img = denormalise(images[i]).permute(1, 2, 0).numpy()  # [C,H,W] → [H,W,C]
    ax.imshow(img)
    ax.set_title(train_dataset.classes[labels[i].item()].replace("_", " "),
                 fontsize=7)
    ax.axis("off")

plt.tight_layout()
output_path = Path("results") / "phase2_sample_images.png"
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=120, bbox_inches="tight")
print(f"Saved visualisation → {output_path}")
plt.show()


print("\n✅ Phase 2 complete! You now have a production-quality data pipeline.")
print("   Next: building the model itself using transfer learning.")
