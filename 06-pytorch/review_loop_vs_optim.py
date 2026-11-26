"""Step 8: the training loop == the 5 steps of one round, with tools.

Two identical models (same seed), same data, same 2000 rounds:
  A) manual : for p in model.parameters(): p -= lr * p.grad ; p.grad.zero_()
  B) optim  : opt.step() ; opt.zero_grad()
If diff == 0.0, the optimizer is nothing but that manual loop.

Run: open in VS Code, click Run (top right).
"""

import torch
import torch.nn as nn


def make_data():
    x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = torch.tensor([[0.], [1.], [1.], [0.]])
    return x, y


def make_model():
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(2, 4),
        nn.Tanh(),
        nn.Linear(4, 1),
    )


x, y = make_data()
loss_fn = nn.MSELoss()
LR = 0.1
STEPS = 2000

# ---------------- A: manual update ----------------
model_a = make_model()
for _ in range(STEPS):
    pred = model_a(x)                 # ① 前向
    loss = loss_fn(pred, y)           # ② 差距
    loss.backward()                   # ③ 反向
    with torch.no_grad():             # ④ 更新（手写版：自己遍历参数）
        for p in model_a.parameters():
            p -= LR * p.grad
            p.grad.zero_()            # ⑤ 清零

# ---------------- B: optimizer ----------------
model_b = make_model()
opt = torch.optim.SGD(model_b.parameters(), lr=LR)
for _ in range(STEPS):
    opt.zero_grad()                   # ⑤ 清零（提前到开头，等价）
    pred = model_b(x)                 # ① 前向
    loss = loss_fn(pred, y)           # ② 差距
    loss.backward()                   # ③ 反向
    opt.step()                        # ④ 更新

# ---------------- compare ----------------
print("参数个数:", sum(p.numel() for p in model_a.parameters()))
worst = 0.0
for (na, pa), (nb, pb) in zip(model_a.named_parameters(), model_b.named_parameters()):
    d = (pa - pb).abs().max().item()
    worst = max(worst, d)
    print("  %-10s 手写版 vs 优化器版  最大差 = %.2e" % (na, d))

print()
print("loss (手写版)   = %.8f" % loss_fn(model_a(x), y).item())
print("loss (优化器版) = %.8f" % loss_fn(model_b(x), y).item())
print("参数最大差      = %.2e" % worst)
