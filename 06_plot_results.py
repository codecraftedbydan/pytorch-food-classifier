"""
PHASE 6: Results Visualisation
===============================
Reads training_history.json (saved by 04_train.py) and produces
publication-quality training curves for the README / GitHub.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

RESULTS_DIR  = Path("results")
HISTORY_FILE = RESULTS_DIR / "training_history.json"

if not HISTORY_FILE.exists():
    print("No training history yet — run 04_train.py first.")
    raise SystemExit

with open(HISTORY_FILE) as f:
    history = json.load(f)

epochs      = [h["epoch"] + (5 if h["stage"] == 2 else 0) for h in history]
train_loss  = [h["train_loss"] for h in history]
val_loss    = [h["val_loss"]   for h in history]
train_acc   = [h["train_acc"] * 100 for h in history]
val_acc     = [h["val_acc"]   * 100 for h in history]

stage1_end  = sum(1 for h in history if h["stage"] == 1)

fig = plt.figure(figsize=(14, 5))
fig.suptitle("EfficientNet-B2 Fine-tuned on Food-101", fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)

# ── Loss ──
ax1 = fig.add_subplot(gs[0])
ax1.plot(epochs, train_loss, "o-", label="Train loss", color="#2196F3", linewidth=2)
ax1.plot(epochs, val_loss,   "o-", label="Val loss",   color="#FF5722", linewidth=2)
ax1.axvline(stage1_end + 0.5, color="grey", linestyle="--", linewidth=1.2,
            label="Fine-tuning starts")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cross-entropy loss")
ax1.set_title("Loss")
ax1.legend()
ax1.grid(alpha=0.3)

# ── Accuracy ──
ax2 = fig.add_subplot(gs[1])
ax2.plot(epochs, train_acc, "o-", label="Train acc", color="#2196F3", linewidth=2)
ax2.plot(epochs, val_acc,   "o-", label="Val acc",   color="#FF5722", linewidth=2)
ax2.axvline(stage1_end + 0.5, color="grey", linestyle="--", linewidth=1.2,
            label="Fine-tuning starts")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Top-1 accuracy (%)")
ax2.set_title("Accuracy")
ax2.legend()
ax2.grid(alpha=0.3)

# Annotate best val acc
best_val  = max(val_acc)
best_ep   = epochs[val_acc.index(best_val)]
ax2.annotate(f"Best: {best_val:.1f}%",
             xy=(best_ep, best_val),
             xytext=(best_ep - 1.5, best_val - 5),
             arrowprops=dict(arrowstyle="->", color="black"),
             fontsize=10, color="darkred", fontweight="bold")

plt.tight_layout()
out = RESULTS_DIR / "training_curves.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.show()

print(f"\nFinal results:")
print(f"  Best val accuracy : {best_val:.2f}%")
print(f"  Final train loss  : {train_loss[-1]:.4f}")
print(f"  Final val loss    : {val_loss[-1]:.4f}")
