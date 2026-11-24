"""M2 check: the learning-rate ceiling eta < 2 / lambda_max, seen numerically.

A = [[2,1],[1,2]] has eigenvalues 3 and 1.
  eigenvector (1, 1) -> lambda = 3   (the dangerous direction)
  eigenvector (1,-1) -> lambda = 1

On an eigen-direction, one gradient step multiplies the length by
    r = 1 - eta * lambda
so convergence needs |r| < 1 for every direction, i.e. eta < 2/lambda_max = 2/3.

Run: open in VS Code, click Run (top right).
"""

import numpy as np

A = np.array([[2.0, 1.0],
              [1.0, 2.0]])

# start exactly along the (1,1) direction, which has lambda = 3
x0 = np.array([1.0, 1.0])

print("=" * 62)
print("Part 1: start x0 = (1,1)  ->  only the lambda=3 direction")
print("  step   x                      |x|")
print("-" * 62)

for eta in (0.5, 2.0 / 3.0, 0.7):
    r = 1.0 - eta * 3.0
    print(f"eta = {eta:.4f}   predicted r = 1 - eta*3 = {r:+.4f}")
    x = x0.copy()
    for step in range(6):
        print(f"  {step:>2}   ({x[0]:+9.4f}, {x[1]:+9.4f})   {np.linalg.norm(x):9.4f}")
        x = x - eta * (A @ x)
    print("-" * 62)

# ---- Part 2: two directions at once, and the predicted formula ----
# x0 = (2,0) = 1*(1,1) + 1*(1,-1)
print("Part 2: start x0 = (2,0)  ->  both directions")
eta = 0.5
r1 = 1.0 - eta * 3.0     # for (1,1)
r2 = 1.0 - eta * 1.0     # for (1,-1)
a, b = 1.0, 1.0

x = np.array([2.0, 0.0])
print(f"  eta = {eta}, predicted factors: r1 = {r1:+.4f}, r2 = {r2:+.4f}")
print("  step   actual x              |x|_actual  |x|_predicted   diff")
worst = 0.0
for step in range(8):
    pred = a * (r1 ** step) * np.array([1.0, 1.0]) + b * (r2 ** step) * np.array([1.0, -1.0])
    d = np.abs(x - pred).max()
    worst = max(worst, d)
    print(f"  {step:>2}   ({x[0]:+9.5f}, {x[1]:+9.5f})  "
          f"{np.linalg.norm(x):9.5f}   {np.linalg.norm(pred):9.5f}   {d:.2e}")
    x = x - eta * (A @ x)

print("-" * 62)
print(f"  worst diff between loop and formula: {worst:.2e}")
print(f"  eta ceiling 2/lambda_max = 2/3 = {2.0/3.0:.4f}")
