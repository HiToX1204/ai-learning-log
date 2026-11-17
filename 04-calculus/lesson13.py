def f(x):
    return x ** 2

def num_diff(f, x, h=1e-4):
    return (f(x + h) - f(x - h)) / (2 * h)

for lr in [0.01, 0.5, 1.1]:
    x = 10.0
    for i in range(20):
        x = x - lr * num_diff(f, x)
    print(f"lr={lr:4}  x={x:12.4f}  f(x)={f(x):14.4f}")
def g(x):
    return 3*x**2 - 4*x + 5

print(g(2))
print(num_diff(g, 2))