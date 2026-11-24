"""Step 3: same data, same MSE loss, same training loop -- only the model changes.

Linear (step 2): nn.Linear(2, 1)
MLP    (step 3): nn.Linear(2, 16) -> nn.ReLU() -> nn.Linear(16, 1)

Run: open in VS Code, click Run (top right).
"""

import numpy as np
import torch
import torch.nn as nn

# ---- same data as step 2 (same seed -> same 400 points) ----
rng = np.random.default_rng(0)
n = 200
r0 = rng.uniform(0.0, 1.0, n)
t0 = rng.uniform(0.0, 2 * np.pi, n)
x0 = np.stack([r0 * np.cos(t0), r0 * np.sin(t0)], axis=1)

r1 = rng.uniform(1.4, 2.0, n)
t1 = rng.uniform(0.0, 2 * np.pi, n)
x1 = np.stack([r1 * np.cos(t1), r1 * np.sin(t1)], axis=1)

X = np.concatenate([x0, x1], axis=0)
y = np.concatenate([np.zeros(n), np.ones(n)])

X = torch.tensor(X, dtype=torch.float32)                  # (400, 2)
Y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)   # (400, 1)

# ---- the only change: 2 -> 16 -> 1 instead of 2 -> 1 ----
torch.manual_seed(0)
model = nn.Sequential(
    nn.Linear(2, 16),    # weight (16,2), bias (16,)   = 32 + 16 = 48 params
    nn.ReLU(),           # max(0, z), no params
    nn.Linear(16, 1),    # weight (1,16), bias (1,)    = 16 + 1  = 17 params
)
loss_fn = nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=0.1)

n_params = sum(p.numel() for p in model.parameters())
print("total trainable params:", n_params)

for step in range(3000):
    opt.zero_grad()
    score = model(X)             # (400, 1)
    loss = loss_fn(score, Y)
    loss.backward()
    opt.step()

# ---- evaluate ----
pred = (score > 0.5).float()
correct = (pred == Y).float()
print("loss      =", round(loss.item(), 6))
print("accuracy  = %.4f" % correct.mean().item())
print()
print("             correct / total")
print("class0 inner: %3d / 200   acc %.3f" % (correct[:200].sum().item(), correct[:200].mean().item()))
print("class1 outer: %3d / 200   acc %.3f" % (correct[200:].sum().item(), correct[200:].mean().item()))
print()
print("score range: [%.2f, %.2f]" % (score.min().item(), score.max().item()))
