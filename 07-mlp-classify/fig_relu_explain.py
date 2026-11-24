"""What ReLU actually does, in one picture.

Left : the function itself, y = max(0, z). One fold at z = 0.
Right: what ONE neuron of the trained MLP looks like on the plane.
       Its pre-activation  z = w1*x1 + w2*x2 + b  is a tilted plane,
       and ReLU flattens the half where z < 0 to zero.
       The boundary of that flat half is a straight line -- the fold.

Run: open in VS Code, click Run (top right).
Saves 07-mlp-classify/relu_explain.png
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ---------- train the same MLP as step 3/4 ----------
rng = np.random.default_rng(0)
n = 200
r0 = rng.uniform(0.0, 1.0, n); t0 = rng.uniform(0.0, 2 * np.pi, n)
x0 = np.stack([r0 * np.cos(t0), r0 * np.sin(t0)], axis=1)
r1 = rng.uniform(1.4, 2.0, n); t1 = rng.uniform(0.0, 2 * np.pi, n)
x1 = np.stack([r1 * np.cos(t1), r1 * np.sin(t1)], axis=1)
Xnp = np.concatenate([x0, x1], axis=0)
ynp = np.concatenate([np.zeros(n), np.ones(n)])

X = torch.tensor(Xnp, dtype=torch.float32)
Y = torch.tensor(ynp, dtype=torch.float32).reshape(-1, 1)

torch.manual_seed(0)
mlp = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
opt = torch.optim.SGD(mlp.parameters(), lr=0.1)
loss_fn = nn.MSELoss()
for _ in range(3000):
    opt.zero_grad()
    loss = loss_fn(mlp(X), Y)
    loss.backward()
    opt.step()

W = mlp[0].weight.detach().numpy()     # (16, 2)
B = mlp[0].bias.detach().numpy()       # (16,)
j = 0                                   # look at neuron 0
w1, w2, b = W[j, 0], W[j, 1], B[j]
print("neuron 0:  z = %.3f*x1 + %.3f*x2 + %.3f" % (w1, w2, b))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

# ---- left: the 1D function ----
ax = axes[0]
z = np.linspace(-3, 3, 400)
ax.axvspan(-3, 0, color="gray", alpha=0.2)
ax.plot(z, np.maximum(0, z), lw=3, color="tab:red")
ax.axhline(0, color="k", lw=0.6)
ax.axvline(0, color="k", lw=0.6)
ax.plot([0], [0], "ko", ms=8)
ax.annotate("fold here\nz = 0", (0, 0), textcoords="offset points", xytext=(18, 40), fontsize=11)
ax.text(-2.6, 2.4, "z < 0  ->  all outputs are 0\n(neuron switched OFF)", fontsize=10)
ax.text(0.6, 0.6, "z > 0  ->  pass through\n(neuron switched ON)", fontsize=10)
ax.set_title("ReLU:  y = max(0, z)")
ax.set_xlabel("z  (the neuron's weighted sum)")
ax.set_ylabel("y  (what the neuron passes on)")
ax.set_xlim(-3, 3); ax.set_ylim(-0.5, 3)
ax.grid(alpha=0.3)

# ---- right: one neuron on the plane ----
ax = axes[1]
g = np.linspace(-2.3, 2.3, 300)
gx, gy = np.meshgrid(g, g)
zp = w1 * gx + w2 * gy + b
cf = ax.contourf(gx, gy, zp, levels=20, cmap="RdBu", alpha=0.7)
ax.contour(gx, gy, zp, levels=[0], colors="k", linewidths=2)
ax.contourf(gx, gy, zp, levels=[-1e6, 0], colors="gray", alpha=0.45)
ax.scatter(x0[:, 0], x0[:, 1], s=6, c="tab:blue", edgecolors="none")
ax.scatter(x1[:, 0], x1[:, 1], s=6, c="tab:orange", edgecolors="none")
ax.set_title("one neuron on the plane:  z = %.2f*x1 + %.2f*x2 + %.2f\n"
             "grey half = switched OFF (output 0), black line = the fold" % (w1, w2, b))
ax.set_xlabel("x1")
ax.set_aspect("equal")
fig.colorbar(cf, ax=ax, shrink=0.85, label="z before ReLU")

plt.tight_layout()
plt.savefig("07-mlp-classify/relu_explain.png", dpi=130)
print("saved 07-mlp-classify/relu_explain.png")
print("fraction of plane switched OFF:", round((zp < 0).mean(), 3))
