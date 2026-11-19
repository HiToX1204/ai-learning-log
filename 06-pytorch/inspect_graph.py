import torch
torch.set_default_dtype(torch.float64)

x1 = torch.tensor(2.0, requires_grad=True)
x2 = torch.tensor(0.0, requires_grad=True)
w1 = torch.tensor(-3.0, requires_grad=True)
w2 = torch.tensor(1.0, requires_grad=True)
b  = torch.tensor(6.8813735870195432, requires_grad=True)

n = w1 * x1 + w2 * x2 + b
o = torch.tanh(n)

print("o.grad_fn      =", o.grad_fn)
print("n.grad_fn      =", n.grad_fn)
print("x1.grad_fn     =", x1.grad_fn)
print("x1.is_leaf     =", x1.is_leaf)
print("o.is_leaf      =", o.is_leaf)
def trace(fn, depth=0):
    if fn is None:
        print("  " * depth + "(叶子)")
        return
    print("  " * depth + fn.__class__.__name__)
    for next_fn, _ in fn.next_functions:
        trace(next_fn, depth + 1)

print("\n--- 计算图 ---")
trace(o.grad_fn)
for fn, _ in o.grad_fn.next_functions:
    print(fn.__class__.__name__, "→ 输出值:", fn.variable.data)