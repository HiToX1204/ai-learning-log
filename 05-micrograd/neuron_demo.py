from micrograd_v1 import Value

x1 = Value(2.0)
x2 = Value(0.0)
w1 = Value(-3.0)
w2 = Value(1.0)
b  = Value(6.8813735870195432)

# 你的任务：用 + 和 * 拼出下面这个式子，最后过 tanh
n = x1*w1 + x2*w2 + b
o = n.tanh()

o.backward()
print(o)
print("x1.grad =", x1.grad, " w1.grad =", w1.grad)
print("x2.grad =", x2.grad, " w2.grad =", w2.grad)
print("b.grad  =", b.grad)