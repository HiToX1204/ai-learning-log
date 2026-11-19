import torch

x = torch.tensor([0., 1., 2., 3., 4.])
y = torch.tensor([1., 3., 5., 7., 9.])

w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

pred = w * x + b
loss = ((pred - y) ** 2).mean()

print("pred =", pred)
print("loss =", loss.item())
loss.backward()
print("w.grad =", w.grad.item())
print("b.grad =", b.grad.item())
for lr in [0.001, 0.01, 0.1, 1.0]:
    w2 = torch.tensor(0.0, requires_grad=True)
    b2 = torch.tensor(0.0, requires_grad=True)
    for _ in range(20):
        p = w2 * x + b2
        l = ((p - y) ** 2).mean()
        l.backward()
        with torch.no_grad():
            w2 -= lr * w2.grad
            b2 -= lr * b2.grad
        w2.grad.zero_()
        b2.grad.zero_()
    print(f"lr={lr:<6} w={w2.item():>8.4f} b={b2.item():>8.4f}")
    w3 = torch.tensor(0.0, requires_grad=True)
b3 = torch.tensor(0.0, requires_grad=True)

losses = []
lr = 0.1

for step in range(200):
    p = w3 * x + b3
    l = ((p - y) ** 2).mean()
    l.backward()
    with torch.no_grad():
        w3 -= lr * w3.grad
        b3 -= lr * b3.grad
    w3.grad.zero_()
    b3.grad.zero_()
    losses.append(l.item())

print("w, b =", round(w3.item(), 4), round(b3.item(), 4))
print("前 5:", [round(v, 3) for v in losses[:5]])
print("后 5:", [round(v, 6) for v in losses[-5:]])
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# 左图：loss 曲线
axes[0].plot(losses)
axes[0].set_xlabel("step")
axes[0].set_ylabel("loss")
axes[0].set_title("Loss curve")
axes[0].set_yscale("log")          # ← 关键：对数纵轴

# 右图：数据和拟合直线
axes[1].scatter(x.numpy(), y.numpy(), label="data")
xs = torch.linspace(0, 4, 50)
with torch.no_grad():
    axes[1].plot(xs.numpy(), (w3 * xs + b3).numpy(), "r-", label="fit")
axes[1].set_title(f"y = {w3.item():.2f}x + {b3.item():.2f}")
axes[1].legend()

plt.tight_layout()
plt.savefig("06-pytorch/loss_curve.png", dpi=120)
print("图已保存到 06-pytorch/loss_curve.png")