from micrograd_v1 import Value

x1, x2 = Value(1.0), Value(2.0)

# 两个隐藏神经元
w11, w12, b1 = Value(0.5), Value(-1.0), Value(0.1)
w21, w22, b2 = Value(0.3), Value(0.8), Value(-0.2)

n1 = x1*w11 + x2*w12 + b1
n2 = x1*w21 + x2*w22 + b2

a1, a2 = n1.tanh(), n2.tanh()

# 输出层
out = a1*Value(0.7) + a2*Value(-0.4)
out.backward()

print("out =", out)
print("w11.grad =", w11.grad, " w12.grad =", w12.grad)
print("x1.grad  =", x1.grad,  " x2.grad  =", x2.grad)