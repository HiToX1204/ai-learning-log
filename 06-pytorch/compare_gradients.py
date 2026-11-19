import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '05-micrograd'))
from micrograd_v1 import Value
import torch
torch.set_default_dtype(torch.float64)     # 放在 import torch 之后
# ── 1. micrograd 版 ──────────────────────────
mx1, mx2 = Value(2.0), Value(0.0)
mw1, mw2 = Value(-3.0), Value(1.0)
mb = Value(6.8813735870195432)

mn = mx1 * mw1 + mx2 * mw2 + mb
mo = mn.tanh()
mo.backward()

# ── 2. PyTorch 版 ────────────────────────────
x1 = torch.tensor(2.0, requires_grad=True)
x2 = torch.tensor(0.0, requires_grad=True)
w1 = torch.tensor(-3.0, requires_grad=True)
w2 = torch.tensor(1.0, requires_grad=True)
b  = torch.tensor(6.8813735870195432, requires_grad=True)

n = b+ w1 * x1 + w2 * x2 
o = torch.tanh(n)
o.backward()

# ── 3. 对照 ──────────────────────────────────
print(f"o:  micrograd={mo.data:.6f}   pytorch={o.item():.6f}")
print(f"{'':4} {'micrograd':>14} {'pytorch':>14} {'差值':>12}")
for name, mg, pt in [("w1", mw1, w1), ("b", mb, b), ("x1", mx1, x1), ("w2", mw2, w2)]:
    print(f"{name:4} {mg.grad:>14.8f} {pt.grad.item():>14.8f} {abs(mg.grad - pt.grad.item()):>12.2e}")