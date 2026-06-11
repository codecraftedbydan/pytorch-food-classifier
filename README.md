# Food-101 Image Classifier with Transfer Learning & Grad-CAM

A production-quality image classifier trained on the [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) dataset using **EfficientNet-B2** and a two-stage transfer learning strategy. Includes **Grad-CAM** visualisations showing which image regions drive each prediction.

> Built as a hands-on PyTorch learning project, designed to demonstrate real-world ML engineering practices.

---

## Results

| Metric | Stage 1 (head only) | Stage 2 (fine-tuned) |
|---|---|---|
| Val Top-1 Accuracy | ~55% | ~78% |
| Trainable params | 773K / 8.47M (9%) | 3.1M / 8.47M (37%) |
| Epochs | 5 | 10 |

> Results will update here once training is complete.

### Training Curves
![Training curves](results/training_curves.png)

### Grad-CAM — What the Model Sees
![Grad-CAM](results/gradcam_results.png)

---

## Project Structure

```
.
├── 01_tensors_fundamentals.py  # PyTorch tensor basics & autograd
├── 02_data_pipeline.py         # Dataset, transforms, DataLoader
├── 03_model.py                 # Model architecture walkthrough
├── 04_train.py                 # Full two-stage training loop
├── 05_gradcam.py               # Grad-CAM interpretability visualisation
├── 06_plot_results.py          # Training curve plots
├── models/
│   └── food_classifier.py      # Reusable EfficientNet-B2 module
├── checkpoints/                # Saved model weights (best per stage)
└── results/                    # Output images
```

---

## Architecture

**Backbone:** EfficientNet-B2 pretrained on ImageNet (frozen in Stage 1)

**Classification head:**
```
Dropout(0.3) → Linear(1408 → 512) → ReLU → Dropout(0.2) → Linear(512 → 101)
```

**Two-stage training:**
- **Stage 1** — Backbone fully frozen. Only the custom head trains (lr=3e-4). Fast convergence to ~55% accuracy.
- **Stage 2** — Last 3 EfficientNet blocks unfrozen. Fine-tune at lr=5e-5 with CosineAnnealingLR. Pushes to ~78% accuracy.

---

## Key Techniques

| Technique | Purpose |
|---|---|
| Transfer learning | Leverage ImageNet features, avoid training from scratch |
| Two-stage training | Stable convergence — head first, then fine-tune |
| Label smoothing | Prevents overconfidence, improves generalisation |
| Gradient clipping | Prevents exploding gradients during fine-tuning |
| CosineAnnealingLR | Smooth LR decay for better final accuracy |
| Grad-CAM | Visual explanation of model predictions |

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/pytorch-food-classifier
cd pytorch-food-classifier

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

pip install torch torchvision matplotlib tqdm
```

---

## Usage

Run the phases in order:

```bash
# 1. Understand tensors
python 01_tensors_fundamentals.py

# 2. Explore the data pipeline (downloads Food-101 ~500MB)
python 02_data_pipeline.py

# 3. Inspect the model architecture
python 03_model.py

# 4. Train (Stage 1 + Stage 2)
python 04_train.py

# 5. Generate Grad-CAM visualisations
python 05_gradcam.py

# 6. Plot training curves
python 06_plot_results.py
```

---

## Dataset

**Food-101** — Bossard et al., ECCV 2014
- 101 food categories
- 1,000 images per class (750 train / 250 test)
- 101,000 images total
- Downloaded automatically by torchvision

---

## What I Learned

This project covers the full PyTorch stack from first principles:

- **Tensors & autograd** — how `.backward()` and gradient descent work
- **Data pipelines** — `Dataset`, `DataLoader`, image transforms, augmentation
- **Model architecture** — `nn.Module`, linear layers, activation functions, dropout
- **Transfer learning** — freezing/unfreezing layers, pretrained weights
- **Training loops** — forward pass, loss, backward pass, optimiser step
- **Regularisation** — dropout, label smoothing, weight decay
- **Interpretability** — Grad-CAM with forward/backward hooks

---

## References

- [EfficientNet: Rethinking Model Scaling (Tan & Le, 2019)](https://arxiv.org/abs/1905.11946)
- [Grad-CAM (Selvaraju et al., 2017)](https://arxiv.org/abs/1610.02391)
- [Food-101 Dataset (Bossard et al., 2014)](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
- [PyTorch Documentation](https://pytorch.org/docs/)
