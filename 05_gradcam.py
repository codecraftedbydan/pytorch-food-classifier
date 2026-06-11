"""
PHASE 5: Grad-CAM — Making the Model Explain Itself
====================================================
Grad-CAM (Gradient-weighted Class Activation Mapping) answers the question:
  "Which part of this image made you say it's a pizza?"

It produces a heatmap overlaid on the original image, highlighting the
regions the model focused on. This is called **model interpretability** —
one of the hottest areas in modern AI (safety, trust, debugging).

How it works:
  1. Pick a convolutional layer near the end of the backbone
     (late layers = high-level semantics, not just edges)
  2. Do a forward pass and record the feature maps at that layer
  3. Do a backward pass and record the gradients flowing back into that layer
  4. Average the gradients to get a weight per feature map channel
  5. Weighted-sum the feature maps → one heatmap
  6. ReLU + resize to image size → overlay on original image

The result tells you: "these pixels caused this prediction."

This is what separates a toy project from an impressive one.
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from models.food_classifier import FoodClassifier

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 101
CHECKPOINT  = Path("checkpoints") / "best_stage2.pt"   # use best_stage1.pt if stage2 not done yet
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# 1. GRAD-CAM IMPLEMENTATION
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM for any model with a named convolutional target layer.

    Usage:
        cam = GradCAM(model, target_layer=model.features[-1])
        heatmap = cam(image_tensor, class_idx=None)  # None = predicted class
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._features    = None    # will store the forward feature maps
        self._gradients   = None    # will store the backward gradients

        # Register hooks — these are callbacks PyTorch fires automatically
        # during forward/backward passes
        self._fwd_hook = target_layer.register_forward_hook(self._save_features)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_features(self, module, input, output):
        """Called automatically during forward pass."""
        self._features = output.detach()    # shape: [1, C, H, W]

    def _save_gradients(self, module, grad_input, grad_output):
        """Called automatically during backward pass."""
        self._gradients = grad_output[0].detach()   # shape: [1, C, H, W]

    def __call__(self, image: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.

        Args:
            image     : preprocessed image tensor [1, 3, 224, 224]
            class_idx : target class (None = use predicted class)
        Returns:
            heatmap   : numpy array [224, 224] in range [0, 1]
        """
        self.model.eval()
        image = image.to(DEVICE).requires_grad_(False)

        # Forward pass — hooks fire and save feature maps
        logits = self.model(image)                      # [1, 101]
        probs  = torch.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()     # use predicted class

        # Backward pass on the score for the target class
        # We zero the model gradients, then backprop the single class score
        self.model.zero_grad()
        logits[0, class_idx].backward()                 # hooks fire, save gradients

        # ── Compute the heatmap ──────────────────────────────────────────────

        # gradients shape: [1, C, h, w]
        # features  shape: [1, C, h, w]
        gradients = self._gradients[0]      # [C, h, w]
        features  = self._features[0]       # [C, h, w]

        # Global average pool the gradients over spatial dimensions → one weight per channel
        weights = gradients.mean(dim=(1, 2))    # [C]

        # Weighted combination of feature maps
        cam = (weights[:, None, None] * features).sum(dim=0)   # [h, w]

        # ReLU: only keep positive activations (negative = evidence against the class)
        cam = F.relu(cam)

        # Normalise to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input image size
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),      # [1, 1, h, w]
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()               # [224, 224]

        return cam, class_idx, probs[0, class_idx].item()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: Path) -> FoodClassifier:
    model = FoodClassifier(num_classes=NUM_CLASSES, freeze_backbone=False)
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded checkpoint: {checkpoint_path}  "
              f"(val_acc={100*ckpt['val_acc']:.2f}%)")
    else:
        print(f"WARNING: No checkpoint at {checkpoint_path}. Using untrained weights.")
        print("Run 04_train.py first, or use best_stage1.pt if stage 1 is done.")
    return model.to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# 3. IMAGE PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def denormalise(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised tensor back to a displayable [H, W, 3] numpy array."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img  = (tensor.cpu() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 4. VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def overlay_heatmap(image_np: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.45) -> np.ndarray:
    """Blend a Grad-CAM heatmap onto the original image."""
    colormap  = cm.get_cmap("jet")
    heatmap_c = colormap(heatmap)[..., :3]   # [H, W, 3] RGB
    overlay   = (1 - alpha) * image_np + alpha * heatmap_c
    return np.clip(overlay, 0, 1)


def visualise_gradcam(model, image_paths: list[Path], class_names: list[str]):
    """
    Run Grad-CAM on a list of images and save a single comparison figure.
    Each image shows: original | Grad-CAM overlay | heatmap only
    """
    # Hook into the last convolutional block of EfficientNet-B2
    # features[-1] is the final MBConv block — highest-level feature maps
    target_layer = model.features[-1]
    gradcam      = GradCAM(model, target_layer)

    n = len(image_paths)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[None, :]    # ensure 2D indexing

    fig.suptitle("Grad-CAM: What the model looks at", fontsize=14, fontweight="bold")

    for row, img_path in enumerate(image_paths):
        # Load + preprocess
        pil_img   = Image.open(img_path).convert("RGB")
        tensor    = preprocess(pil_img).unsqueeze(0)    # [1, 3, 224, 224]
        image_np  = denormalise(tensor[0])              # [224, 224, 3]

        # Run Grad-CAM
        heatmap, pred_idx, confidence = gradcam(tensor)
        pred_name = class_names[pred_idx].replace("_", " ")
        overlay   = overlay_heatmap(image_np, heatmap)

        # Plot
        axes[row, 0].imshow(image_np)
        axes[row, 0].set_title("Original", fontsize=10)

        axes[row, 1].imshow(overlay)
        axes[row, 1].set_title(
            f'Pred: "{pred_name}"\nConfidence: {100*confidence:.1f}%',
            fontsize=10, color="darkgreen"
        )

        axes[row, 2].imshow(image_np)
        axes[row, 2].imshow(heatmap, cmap="jet", alpha=0.5)
        axes[row, 2].set_title("Heatmap only", fontsize=10)

        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    out_path = RESULTS_DIR / "gradcam_results.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.show()

    gradcam.remove_hooks()


# ─────────────────────────────────────────────────────────────────────────────
# 5. RUN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from torchvision.datasets import Food101

    # Load class names from the dataset (no images needed, just metadata)
    ds          = Food101("data", split="test", download=False)
    class_names = ds.classes

    model = load_model(CHECKPOINT)

    # Grab a few sample images from the val set to visualise
    # We pick one image per class from the first N classes
    sample_images = []
    target_classes = ["pizza", "hamburger", "sushi", "ice_cream", "ramen"]

    import random
    random.seed(42)
    for cls_name in target_classes:
        cls_idx = class_names.index(cls_name)
        # Find an image for this class
        candidates = [
            Path("data/food-101/images") / cls_name / f
            for f in (Path("data/food-101/images") / cls_name).iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg")
        ] if (Path("data/food-101/images") / cls_name).exists() else []

        if candidates:
            sample_images.append(random.choice(candidates))

    if not sample_images:
        print("\nNo images found yet — Food-101 download may still be in progress.")
        print("Run this script again once 04_train.py has completed.\n")
    else:
        print(f"\nRunning Grad-CAM on {len(sample_images)} images...\n")
        visualise_gradcam(model, sample_images, class_names)
        print("\n✅ Phase 5 complete! Check results/gradcam_results.png")
