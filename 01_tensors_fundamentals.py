"""
PHASE 1: PyTorch Tensors — The Foundation of Everything
========================================================
A tensor is the core data structure in PyTorch.
Think of it like a numpy array — but with superpowers:
  - It can live on a GPU (for fast computation)
  - It tracks gradients (so PyTorch knows how to train models)

Run this file and read the output alongside the code.
"""

import torch
import numpy as np

print("=" * 55)
print("1. CREATING TENSORS")
print("=" * 55)

# A scalar — a single number (0D tensor)
scalar = torch.tensor(3.14)
print(f"Scalar:       {scalar}  | shape: {scalar.shape}  | ndim: {scalar.ndim}")

# A vector — a 1D list of numbers
vector = torch.tensor([1.0, 2.0, 3.0, 4.0])
print(f"Vector:       {vector} | shape: {vector.shape} | ndim: {vector.ndim}")

# A matrix — 2D (rows x cols)
matrix = torch.tensor([[1, 2, 3],
                        [4, 5, 6]])
print(f"Matrix:\n{matrix}\nshape: {matrix.shape} | ndim: {matrix.ndim}")

# A 3D tensor — like a colour image (channels x height x width)
image_like = torch.zeros(3, 64, 64)  # 3 channels (RGB), 64x64 pixels
print(f"\n3D tensor (fake image): shape {image_like.shape}")


print("\n" + "=" * 55)
print("2. TENSOR OPERATIONS")
print("=" * 55)

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([10.0, 20.0, 30.0])

print(f"a + b  = {a + b}")         # element-wise addition
print(f"a * b  = {a * b}")         # element-wise multiplication
print(f"a @ b  = {a @ b}")         # dot product (used constantly in neural nets)
print(f"a.mean() = {a.mean():.4f}")
print(f"a.sum()  = {a.sum():.4f}")


print("\n" + "=" * 55)
print("3. SHAPES AND RESHAPING (critical for debugging neural nets)")
print("=" * 55)

x = torch.arange(12, dtype=torch.float32)
print(f"Original shape: {x.shape}  -> {x}")

x_2d = x.reshape(3, 4)
print(f"Reshaped (3,4):\n{x_2d}")

# unsqueeze adds a dimension — you'll see this constantly
x_batched = x_2d.unsqueeze(0)   # adds a batch dimension at position 0
print(f"After unsqueeze(0): {x_batched.shape}")  # [1, 3, 4]

x_squeezed = x_batched.squeeze(0)
print(f"After squeeze(0):   {x_squeezed.shape}") # back to [3, 4]


print("\n" + "=" * 55)
print("4. DTYPES — the type of data stored in a tensor")
print("=" * 55)

# Neural networks use float32 by default
floats  = torch.tensor([1.0, 2.0, 3.0])        # float32
ints    = torch.tensor([1, 2, 3])               # int64
bools   = torch.tensor([True, False, True])     # bool

print(f"floats dtype: {floats.dtype}")
print(f"ints   dtype: {ints.dtype}")
print(f"bools  dtype: {bools.dtype}")

# You'll often need to cast — e.g., labels must be long (int64) for loss functions
labels = torch.tensor([0, 1, 2]).long()
print(f"labels dtype: {labels.dtype}  <- needed for CrossEntropyLoss")


print("\n" + "=" * 55)
print("5. GPU / DEVICE — how PyTorch uses your hardware")
print("=" * 55)

# This is one of PyTorch's superpowers: move tensors to GPU trivially
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

t = torch.tensor([1.0, 2.0, 3.0]).to(device)
print(f"Tensor lives on: {t.device}")
# In your training loop you'll always do: model.to(device), data.to(device)


print("\n" + "=" * 55)
print("6. AUTOGRAD — how PyTorch learns (the magic behind backprop)")
print("=" * 55)

# requires_grad=True tells PyTorch to track operations for differentiation
w = torch.tensor(2.0, requires_grad=True)   # a "weight" parameter
x = torch.tensor(3.0)                        # an input

# Forward pass: compute a prediction and a loss
y_pred = w * x          # our "model": y = w * x
loss   = (y_pred - 5) ** 2  # Mean Squared Error vs target=5

print(f"w={w.item()}, x={x.item()}")
print(f"y_pred = {y_pred.item()}")
print(f"loss   = {loss.item()}")

# Backward pass: compute how to nudge w to reduce loss
loss.backward()
print(f"d(loss)/dw = {w.grad.item():.4f}  <- gradient: how much loss changes per unit of w")
print("(PyTorch computed this automatically via backpropagation)")

# --- gradient descent step (this is exactly what optimizers do!) ---
learning_rate = 0.1
with torch.no_grad():               # don't track this operation
    w -= learning_rate * w.grad     # move w slightly in the downhill direction

print(f"\nAfter one gradient descent step: w = {w.item():.4f}")
print("This is ALL a training step is — done millions of times.")


print("\n" + "=" * 55)
print("7. INTEROPERABILITY WITH NUMPY")
print("=" * 55)

np_array = np.array([1.0, 2.0, 3.0])
pt_tensor = torch.from_numpy(np_array)      # numpy -> torch (shares memory!)
back_to_np = pt_tensor.numpy()              # torch -> numpy

print(f"numpy  -> torch: {pt_tensor}")
print(f"torch  -> numpy: {back_to_np}")


print("\n✅ Phase 1 complete! You now understand the fundamental building block.")
print("   Next: loading real data with Datasets and DataLoaders.")
