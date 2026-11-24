"""Step 4: draw what each model actually learned.

Left : linear  nn.Linear(2, 1)
Right: MLP     nn.Linear(2,16) -> ReLU -> nn.Linear(16,1)
Colour = the model's score. Black line = the decision boundary (score = 0.5).

Run: open in VS Code, click Run (top right).
Saves 07-mlp-classify/boundary_compare.png
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ---------------- data ----------------
rng = np.random.default_rng(0)
n = 200
r0 = rng.uniform(0.0, 1.0, n)
t0 = rng.uniform(0.0, 2 * np.pi, n)
x0 = np.stack([r0 * np.cos(t0), r0 * np.sin(t0)], axis=1)
r1 = rng.uniform(1.4, 2.0, n)
t1 = rng.uniform(0.0, 2 * np.pi, n)
x1 = np.stack([r1 * np.cos(t1), r1 * np.sin(t1)], axis=1)

Xnp = np.concatenate([x0, x1], axis=0)
ynp = np.concatenate([np.zeros(n), np.ones(n)])
X = torch.tensor(Xnp, dtype=torch.float32)
Y = torch.tensor(ynp, dtype=torch.float32).reshape(-1, 1)

# ---------------- two models ----------------
torch.manual_seed(0)
lin = nn.Linear(2, 1)

torch.manual_seed(0)
mlp = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))

loss_fn = nn.MSELoss()

def train(model, steps=3000, lr=0.1):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(X), Y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        s = model(X)
        acc = ((s > 0.5).float() == Y).float().mean().item()
        return loss.item(), acc

loss_lin, acc_lin = train(lin)
loss_mlp, acc_mlp = train(mlp)
print("linear: loss %.4f  acc %.4f  params %d" % (loss_lin, acc_lin, sum(p.numel() for p in lin.parameters())))
print("MLP   : loss %.4f  acc %.4f  params %d" % (loss_mlp, acc_mlp, sum(p.numel() for p in mlp.parameters())))

# ---------------- score field over the plane ----------------
g = np.linspace(-2.3, 2.3, 300)
gx, gy = np.meshgrid(g, g)
G = torch.tensor(np.stack([gx.ravel(), gy.ravel()], axis=1), dtype=torch.float32)

with torch.no_grad():
    z_lin = lin(G).numpy().reshape(gx.shape)
    z_mlp = mlp(G).numpy().reshape(gx.shape)

# ---------------- plot ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, Z, title, acc in ((axes[0], z_lin, "linear  nn.Linear(2,1)", acc_lin),
                          (axes[1], z_mlp, "MLP  2->16->1 with ReLU", acc_mlp)):
    cf = ax.contourf(gx, gy, Z, levels=20, cmap="RdBu", alpha=0.75)
    ax.contour(gx, gy, Z, levels=[0.5], colors="k", linewidths=2.5)
    ax.scatter(x0[:, 0], x0[:, 1], s=8, c="tab:blue", edgecolors="none", label="class0 inner")
    ax.scatter(x1[:, 0], x1[:, 1], s=8, c="tab:orange", edgecolors="none", label="class1 outer")
    ax.set_title("%s\naccuracy = %.4f" % (title, acc))
    ax.set_xlabel("x1")
    ax.set_aspect("equal")
    fig.colorbar(cf, ax=ax, shrink=0.85)

plt.tight_layout()
plt.savefig("07-mlp-classify/boundary_compare.png", dpi=130)
print("saved 07-mlp-classify/boundary_compare.png")
