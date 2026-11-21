import torch

torch.manual_seed(0)

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = torch.tensor([[0.], [1.], [1.], [0.]])

w = torch.zeros(2, 1, requires_grad=True)     # 两个权重
b = torch.zeros(1, requires_grad=True)

lr = 0.5
for step in range(2000):
    pred = X @ w + b                 # 线性模型：一个平面
    loss = ((pred - Y) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad.zero_()
    b.grad.zero_()

print("w =", w.flatten().tolist(), " b =", b.item())
print("loss =", round(loss.item(), 6))
print("pred =", [round(v, 3) for v in pred.flatten().tolist()])
print("Y    =", Y.flatten().tolist())