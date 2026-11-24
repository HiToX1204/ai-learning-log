"""Step 2: how good can a STRAIGHT LINE be on this data?

Model: one nn.Linear(2, 1) -> outputs a score, classify as 1 if score > 0.5.
Trains with MSE (same loss as the XOR experiment), then reports accuracy.

Run: open in VS Code, click Run (top right).
"""

import numpy as np
import torch
import torch.nn as nn

# ---- rebuild the same data (same seed -> same points) ----
rng = np.random.default_rng(0)
n = 200
r0 = rng.uniform(0.0, 1.0, n)
t0 = rng.uniform(0.0, 2 * np.pi, n)
x0 = np.stack([r0 * np.cos(t0), r0 * np.sin(t0)], axis=1)

r1 = rng.uniform(1.4, 2.0, n)
t1 = rng.uniform(0.0, 2 * np.pi, n)
x1 = np.stack([r1 * np.cos(t1), r1 * np.sin(t1)], axis=1)

X = np.concatenate([x0, x1], axis=0)                 # (400, 2)
y = np.concatenate([np.zeros(n), np.ones(n)])        # (400,)

X = torch.tensor(X, dtype=torch.float32)                       # (400, 2)
Y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)        # (400, 1)

# ---- the model: ONE straight line ----
torch.manual_seed(0)
model = nn.Linear(2, 1)          # weight (1,2), bias (1,)
loss_fn = nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=0.1)

for step in range(2000):
    opt.zero_grad()
    score = model(X)             # (400, 1)  a real number per point
    loss = loss_fn(score, Y)     # (400, 1) vs (400, 1)
    loss.backward()
    opt.step()

# ---- evaluate ----
pred = (score > 0.5).float()                 # (400, 1) of 0.0/1.0
correct = (pred == Y).float()                # (400, 1) of 0.0/1.0
acc = correct.mean()

print("loss      =", round(loss.item(), 4))
print("accuracy  = %.4f" % acc.item())
print()
print("             correct / total")
print("class0 inner: %3d / 200   acc %.3f" % (correct[:200].sum().item(), correct[:200].mean().item()))
print("class1 outer: %3d / 200   acc %.3f" % (correct[200:].sum().item(), correct[200:].mean().item()))
print()
print("learned line: score = %.3f*x1 + %.3f*x2 + %.3f" % (
    model.weight[0, 0].item(), model.weight[0, 1].item(), model.bias[0].item()))
