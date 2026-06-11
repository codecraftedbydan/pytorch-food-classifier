"""
PHASE 3: Transfer Learning — Building the Classifier
=====================================================
Training a deep neural network from scratch needs millions of images and
days of GPU time. We don't need that.

Transfer Learning: take a model already trained on ImageNet (1.2M images,
1000 classes) and repurpose it for our task (Food-101, 101 classes).

The pretrained model has already learned:
  - Early layers : edges, corners, textures
  - Middle layers: shapes, patterns, parts
  - Late layers  : high-level concepts ("furry thing", "round food")

We keep all of that. We just replace the final classification head
with one that outputs 101 classes instead of 1000.

This technique is why modern AI is so powerful — you stand on the
shoulders of giants rather than starting from zero.

Architecture: EfficientNet-B2
  - Released by Google Brain (2019)
  - One of the best accuracy-per-parameter models ever
  - Pretrained on ImageNet, top-1 accuracy ~80%
"""

import ssl
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B2_Weights
from pathlib import Path

# Python 3.13 on Windows has an SSL cert issue with PyTorch's model hub.
# This bypasses verification for the weights download only.
ssl._create_default_https_context = ssl._create_unverified_context


NUM_CLASSES = 101   # Food-101


# ─────────────────────────────────────────────────────────────────────────────
# 1. UNDERSTANDING A NEURAL NETWORK MODULE
# ─────────────────────────────────────────────────────────────────────────────
#
# Every model in PyTorch is a class that inherits from nn.Module.
# You must implement:
#   __init__  : define the layers
#   forward   : define how data flows through them
#
# Let's build a tiny one from scratch first so you understand the pattern.

class TinyNet(nn.Module):
    """
    A minimal neural network for intuition.
    Input : a flat vector of 10 numbers
    Output: probabilities over 3 classes
    """
    def __init__(self):
        super().__init__()
        # nn.Linear(in, out) is a fully-connected layer — every input connects
        # to every output. It's just a matrix multiply + bias: y = Wx + b
        self.fc1 = nn.Linear(10, 64)   # 10 inputs  → 64 hidden neurons
        self.fc2 = nn.Linear(64, 32)   # 64 hidden  → 32 hidden neurons
        self.fc3 = nn.Linear(32, 3)    # 32 hidden  → 3 class scores
        self.relu = nn.ReLU()          # activation: max(0, x) — adds non-linearity

    def forward(self, x):
        # x shape: [batch_size, 10]
        x = self.relu(self.fc1(x))     # [batch, 10]  → [batch, 64]
        x = self.relu(self.fc2(x))     # [batch, 64]  → [batch, 32]
        x = self.fc3(x)                # [batch, 32]  → [batch, 3]
        return x                       # raw scores ("logits") — no softmax here
                                       # CrossEntropyLoss applies softmax internally

# Try it
tiny = TinyNet()
fake_input = torch.randn(4, 10)       # batch of 4 samples, 10 features each
output = tiny(fake_input)
print("── TinyNet sanity check ──")
print(f"Input  shape: {fake_input.shape}")
print(f"Output shape: {output.shape}   ← [4 samples, 3 class scores each]")
print(f"Params: {sum(p.numel() for p in tiny.parameters()):,}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 2. WHY RELU? — the non-linearity that makes deep networks work
# ─────────────────────────────────────────────────────────────────────────────
#
# Without an activation function, stacking linear layers is useless:
#   W2 @ (W1 @ x) == (W2 @ W1) @ x  — just one linear layer in disguise.
#
# ReLU(x) = max(0, x) — zeroes out negatives, keeps positives.
# This simple operation creates the non-linearity that lets networks learn
# complex patterns like "this is a pizza and not a hot dog".


# ─────────────────────────────────────────────────────────────────────────────
# 3. OUR REAL MODEL — EfficientNet-B2 with custom head
# ─────────────────────────────────────────────────────────────────────────────

class FoodClassifier(nn.Module):
    """
    EfficientNet-B2 pretrained backbone + custom classification head.

    Strategy:
      - FREEZE the backbone initially (don't update its weights)
      - Train only our new head for a few epochs (fast, cheap)
      - Then UNFREEZE the backbone and fine-tune everything together
        at a very small learning rate (this is called "fine-tuning")
    """

    def __init__(self, num_classes: int = NUM_CLASSES, freeze_backbone: bool = True):
        super().__init__()

        # Load pretrained EfficientNet-B2
        # weights=DEFAULT gives us the best available pretrained weights
        backbone = models.efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)

        # The backbone has two parts:
        #   backbone.features   : all the convolutional layers (the "feature extractor")
        #   backbone.classifier : the final head that outputs 1000 ImageNet classes
        # We keep features, replace classifier.

        self.features   = backbone.features       # the pretrained gold
        self.avgpool    = backbone.avgpool         # global average pooling

        # Find out how many features the backbone outputs
        # EfficientNet-B2 outputs 1408 features before the classifier
        in_features = backbone.classifier[1].in_features

        # Our new classification head — a few layers with dropout for regularisation
        # Dropout randomly zeroes neurons during training → prevents overfitting
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),    # final output: 101 food class scores
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        """Freeze all backbone weights — only our head will train."""
        for param in self.features.parameters():
            param.requires_grad = False
        print("Backbone frozen. Only the classifier head will train.")

    def unfreeze_backbone(self, unfreeze_last_n_blocks: int = 3):
        """
        Unfreeze the last N blocks of the backbone for fine-tuning.
        We don't unfreeze everything — early layers detect basic edges/textures
        that are universal, no need to retrain them.
        """
        # First freeze everything
        for param in self.features.parameters():
            param.requires_grad = False

        # Then selectively unfreeze the last N blocks
        blocks = list(self.features.children())
        for block in blocks[-unfreeze_last_n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Unfrozen last {unfreeze_last_n_blocks} blocks. "
              f"Trainable params: {trainable:,}")

    def forward(self, x):
        # x shape: [batch, 3, 224, 224]
        x = self.features(x)           # [batch, 1408, 7, 7]  — feature maps
        x = self.avgpool(x)            # [batch, 1408, 1, 1]  — pool spatially
        x = torch.flatten(x, 1)        # [batch, 1408]        — flatten
        x = self.classifier(x)         # [batch, 101]         — class scores
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 4. INSPECT THE MODEL
# ─────────────────────────────────────────────────────────────────────────────

print("── FoodClassifier ──")
model = FoodClassifier(num_classes=NUM_CLASSES, freeze_backbone=True)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params    = total_params - trainable_params

print(f"Total parameters     : {total_params:,}")
print(f"Trainable (our head) : {trainable_params:,}  ({100*trainable_params/total_params:.1f}%)")
print(f"Frozen (backbone)    : {frozen_params:,}  ({100*frozen_params/total_params:.1f}%)")

# Run a fake batch through to confirm shapes work end to end
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

fake_images = torch.randn(4, 3, 224, 224).to(device)   # fake batch of 4 images
with torch.no_grad():
    logits = model(fake_images)

print(f"\nForward pass: {fake_images.shape} → {logits.shape}")
print(f"  ← 4 images, each gets 101 class scores (logits)")

# Convert logits → probabilities using softmax
probs = torch.softmax(logits, dim=1)
print(f"Max probability in each prediction: {probs.max(dim=1).values.tolist()}")
print("  ← Untrained model is basically guessing (~1/101 ≈ 1% per class)")


# ─────────────────────────────────────────────────────────────────────────────
# 5. LOSS FUNCTION AND OPTIMISER
# ─────────────────────────────────────────────────────────────────────────────

# CrossEntropyLoss: the standard loss for multi-class classification
# It combines LogSoftmax + NegativeLogLikelihood in one numerically stable op
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
#  label_smoothing=0.1 : instead of target=[0,0,1,0,...], use [0.001,0.001,0.91,...]
#  This stops the model from being overconfident and improves generalisation.

# AdamW: a very reliable optimiser — Adam with proper weight decay
# We give it only the trainable params (no point optimising frozen ones)
optimiser = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=3e-4,        # learning rate — 3e-4 is the "Karpathy constant", works well
    weight_decay=1e-4,
)

# Learning rate scheduler — reduces LR when val loss plateaus
# This gives the optimiser a chance to settle into a minimum more precisely
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimiser,
    T_max=10,       # number of epochs for one cosine cycle
    eta_min=1e-6,
)

print("\n── Training setup ──")
print(f"Loss     : CrossEntropyLoss (label_smoothing=0.1)")
print(f"Optimiser: AdamW  (lr=3e-4, weight_decay=1e-4)")
print(f"Scheduler: CosineAnnealingLR")

# Save model architecture for use in the training script
CHECKPOINTS = Path("checkpoints")
CHECKPOINTS.mkdir(exist_ok=True)
torch.save(model.state_dict(), CHECKPOINTS / "model_init.pt")
print(f"\nSaved initial weights → checkpoints/model_init.pt")


print("\n✅ Phase 3 complete! Model is built and ready.")
print("   Next: the training loop — putting it all together.")
