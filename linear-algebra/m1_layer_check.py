"""M1 check: hand-written triple loop  vs  one line x @ W.T + b

How to run: open this file in VS Code, click the Run button (top right).
Both results must be identical; the printed diff must be 0.0.
"""

import numpy as np

# ============================================================
# Part 1: the numbers you did by hand (2 inputs, 2 neurons)
#   by hand: without b -> (-3, 3)   with b -> (-2, 1)
# ============================================================
W = np.array([[2.0, -1.0],
              [0.0,  1.0]])
b = np.array([1.0, -2.0])
x = np.array([0.0, 3.0])

print("=" * 52)
print("Part 1: one sample")
print("  x shape", x.shape, "  W shape", W.shape, "  b shape", b.shape)

# hand-written: outer loop over neurons, inner loop over inputs
manual = []
for j in range(W.shape[0]):          # j = which neuron  (a row of W)
    total = 0.0
    for i in range(W.shape[1]):      # i = which input   (a column of W)
        total += W[j, i] * x[i]
    manual.append(total + b[j])      # then add this neuron's own bias

# one line: transpose so the middle dims line up, then do it all at once
fast = x @ W.T + b

print("  manual loop   :", [float(v) for v in manual])
print("  x @ W.T + b   :", [float(v) for v in fast])
print("  diff          :", np.abs(np.array(manual) - fast).max())


# ============================================================
# Part 2: the table you drew by hand (2 samples, 3 neurons, b = 0)
#   by hand:  sample1 -> (1, 5, 8)
#             sample2 -> (1, 1, 2)
# ============================================================
W3 = np.array([[2.0, -1.0],
               [0.0,  1.0],
               [1.0,  1.0]])
b3 = np.array([0.0, 0.0, 0.0])
X = np.array([[3.0, 5.0],
              [1.0, 1.0]])

print("=" * 52)
print("Part 2: two samples at once")
print("  X shape", X.shape, "  W3 shape", W3.shape)

manual2 = np.zeros((X.shape[0], W3.shape[0]))
for n in range(X.shape[0]):          # n = which sample  (a row of X)
    for j in range(W3.shape[0]):     # j = which neuron  (a row of W3)
        total = b3[j]
        for i in range(W3.shape[1]): # i = which input
            total += W3[j, i] * X[n, i]
        manual2[n, j] = total

fast2 = X @ W3.T + b3

print("  manual 3 loops:")
print("   ", manual2.tolist())
print("  X @ W3.T + b3 :")
print("   ", fast2.tolist())
print("  diff          :", np.abs(manual2 - fast2).max())
print("  result shape  :", fast2.shape, "= (samples 2, neurons 3)")
