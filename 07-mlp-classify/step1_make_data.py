"""Step 1: make 2D data that a straight line cannot separate.

Class 0: points inside the circle  r < 1.0
Class 1: points in the ring       1.4 < r < 2.0

Run: open in VS Code, click Run (top right).
Output: prints counts, saves 07-mlp-classify/data_2d.png
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)      # generator; seed 0 -> same points every run
n = 200                             # points per class

# ---- class 0: inside the circle ----
r0 = rng.uniform(0.0, 1.0, n)       # radius   in [0, 1)
t0 = rng.uniform(0.0, 2 * np.pi, n) # angle    in [0, 2pi)
x0 = np.stack([r0 * np.cos(t0), r0 * np.sin(t0)], axis=1)   # (200, 2)

# ---- class 1: inside the ring ----
r1 = rng.uniform(1.4, 2.0, n)
t1 = rng.uniform(0.0, 2 * np.pi, n)
x1 = np.stack([r1 * np.cos(t1), r1 * np.sin(t1)], axis=1)   # (200, 2)

X = np.concatenate([x0, x1], axis=0)          # (400, 2)
y = np.concatenate([np.zeros(n), np.ones(n)]) # (400,)   0 for inner, 1 for outer

print("class 0 points:", int((y == 0).sum()))
print("class 1 points:", int((y == 1).sum()))
print("X shape:", X.shape, "  y shape:", y.shape)
print("class 0 radius: min %.2f  max %.2f" % (r0.min(), r0.max()))
print("class 1 radius: min %.2f  max %.2f" % (r1.min(), r1.max()))

# ---- plot ----
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(X[y == 0, 0], X[y == 0, 1], s=12, label="class 0 (r < 1)")
ax.scatter(X[y == 1, 0], X[y == 1, 1], s=12, label="class 1 (1.4 < r < 2)")
ax.set_aspect("equal")          # same scale on both axes, otherwise circles look like ovals
ax.grid(alpha=0.3)
ax.legend()
ax.set_title("2D data: inner circle vs outer ring")
plt.tight_layout()
plt.savefig("07-mlp-classify/data_2d.png", dpi=120)
print("saved 07-mlp-classify/data_2d.png")
