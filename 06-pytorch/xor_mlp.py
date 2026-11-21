import torch

torch.manual_seed(0)

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = torch.tensor([[0.], [1.], [1.], [0.]])

# 隐藏层：2 个输入 -> 4 个神经元
W1 = (torch.randn(2, 4) * 1.0).requires_grad_(True)
b1 = torch.zeros(4, requires_grad=True)

# 输出层：4 个神经元 -> 1 个输出
W2 = (torch.randn(4, 1) * 1.0).requires_grad_(True)
b2 = torch.zeros(1, requires_grad=True)

lr = 0.1
for step in range(5000):
    h = torch.tanh(X @ W1 + b1)     # 隐藏层：4 个"弯折"
    pred = h @ W2 + b2              # 输出层：线性组合
    loss = ((pred - Y) ** 2).mean()

    loss.backward()
    with torch.no_grad():
        for p in (W1, b1, W2, b2):
            p -= lr * p.grad
            p.grad.zero_()

print("loss =", round(loss.item(), 6))
print("pred =", [round(v, 3) for v in pred.flatten().tolist()])
print("Y    =", Y.flatten().tolist())
# 线性模型（用于右图对照）
w_lin = torch.zeros(2, 1, requires_grad=True)
b_lin = torch.zeros(1, requires_grad=True)

for _ in range(2000):
    p = X @ w_lin + b_lin
    l = ((p - Y) ** 2).mean()
    l.backward()
    with torch.no_grad():
        w_lin -= 0.5 * w_lin.grad
        b_lin -= 0.5 * b_lin.grad
    w_lin.grad.zero_()
    b_lin.grad.zero_()

print("线性模型 loss =", round(l.item(), 6))
import numpy as np
import matplotlib.pyplot as plt

# 在平面上铺一层点，让模型预测每个点
xx, yy = np.meshgrid(
    np.linspace(-0.5, 1.5, 200),
    np.linspace(-0.5, 1.5, 200),
)
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

with torch.no_grad():
    h_grid = torch.tanh(grid @ W1 + b1)
    zz = (h_grid @ W2 + b2).numpy().reshape(xx.shape)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# 左：神经网络
axes[0].contourf(xx, yy, zz, levels=20, cmap="RdBu")
axes[0].scatter(X[:, 0], X[:, 1], c=Y.flatten(), edgecolors="k", s=120)
axes[0].set_title("MLP: loss=0.0")

# 右：线性模型
with torch.no_grad():
    zz_lin = (torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32) @ w_lin + b_lin).numpy().reshape(xx.shape)
axes[1].contourf(xx, yy, zz_lin, levels=20, cmap="RdBu")
axes[1].scatter(X[:, 0], X[:, 1], c=Y.flatten(), edgecolors="k", s=120)
axes[1].set_title("Linear: loss=0.25")

plt.tight_layout()
plt.savefig("06-pytorch/xor_decision.png", dpi=120)
print("saved")