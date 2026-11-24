"""Geometry of why a straight line cannot separate a disk from a ring.

Left  : the linear score 2*x1 + 2*x2 - 3 drawn as a colour field.
        Its equal-score lines are PARALLEL STRAIGHT LINES.
Right : the ideal answer -- "how far from the origin" -- whose equal-value
        lines are CONCENTRIC CIRCLES.

The two pictures cannot be made to match: that is the whole problem.

Run: open in VS Code, click Run (top right).
Saves 07-mlp-classify/linear_vs_ideal.png
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
n = 200
r0 = rng.uniform(0.0, 1.0, n)
t0 = rng.uniform(0.0, 2 * np.pi, n)
x0 = np.stack([r0 * np.cos(t0), r0 * np.sin(t0)], axis=1)

r1 = rng.uniform(1.4, 2.0, n)
t1 = rng.uniform(0.0, 2 * np.pi, n)
x1 = np.stack([r1 * np.cos(t1), r1 * np.sin(t1)], axis=1)

X = np.concatenate([x0, x1], axis=0)

# ---- grid over the plane ----
g = np.linspace(-2.3, 2.3, 300)
gx, gy = np.meshgrid(g, g)

score = 2 * gx + 2 * gy - 3          # the linear model's score
radius = np.sqrt(gx ** 2 + gy ** 2)  # what actually separates the classes

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# ================= left: linear score =================
ax = axes[0]
cf = ax.contourf(gx, gy, score, levels=20, cmap="RdBu", alpha=0.75)
ax.contour(gx, gy, score, levels=[0.5], colors="k", linewidths=2)
ax.scatter(x0[:, 0], x0[:, 1], s=8, c="tab:blue", edgecolors="none", label="class0 inner")
ax.scatter(x1[:, 0], x1[:, 1], s=8, c="tab:orange", edgecolors="none", label="class1 outer")

# the four points from the explanation
pts = [(2.0, 0.0, "1"), (-2.0, 0.0, "-7"), (0.7, 0.0, "-1.6"), (-0.7, 0.0, "-4.4")]
for px, py, s in pts:
    ax.plot(px, py, "ko", ms=7)
    ax.annotate(s, (px, py), textcoords="offset points", xytext=(8, 8),
                fontsize=11, fontweight="bold")
ax.set_title("linear score $2x_1+2x_2-3$\nequal-score lines are PARALLEL LINES")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_aspect("equal")
ax.legend(loc="lower left", fontsize=8)
fig.colorbar(cf, ax=ax, shrink=0.85)

# ================= right: radius (the ideal) =================
ax = axes[1]
cf2 = ax.contourf(gx, gy, radius, levels=20, cmap="RdBu", alpha=0.75)
ax.contour(gx, gy, radius, levels=[1.0, 1.4], colors="k", linewidths=2)
ax.scatter(x0[:, 0], x0[:, 1], s=8, c="tab:blue", edgecolors="none")
ax.scatter(x1[:, 0], x1[:, 1], s=8, c="tab:orange", edgecolors="none")
ax.set_title("the answer: distance from origin\nequal-value lines are CONCENTRIC CIRCLES")
ax.set_xlabel("x1")
ax.set_aspect("equal")
fig.colorbar(cf2, ax=ax, shrink=0.85)

plt.tight_layout()
plt.savefig("07-mlp-classify/linear_vs_ideal.png", dpi=130)
print("saved 07-mlp-classify/linear_vs_ideal.png")
print("score at (2,0)  =", 2 * 2.0 + 2 * 0.0 - 3)
print("score at (-2,0) =", 2 * (-2.0) + 2 * 0.0 - 3)
print("score at (0.7,0)  =", round(2 * 0.7 - 3, 2))
print("score at (-0.7,0) =", round(2 * (-0.7) - 3, 2))
