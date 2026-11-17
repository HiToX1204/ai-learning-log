import numpy as np

# a = np.arange(12)
# print("a:", a.shape, a)

# b = a.reshape(3, 4)
# print("b:", b.shape)
# print(b)

# c = a.reshape(4, 3)
# print("c:", c.shape)
# print(c)
# a = np.arange(12)
# b = a.reshape(3, 4)

# b[0, 0] = 99

# print("a =", a)
# print("b =", b)
# print("共享内存?", np.shares_memory(a, b))
# b = np.arange(12).reshape(3, 4)
# t = b.T

# print("b:"); print(b)
# print("t:"); print(t)
# print("t.shape =", t.shape)
# print("共享内存?", np.shares_memory(b, t))
x = np.arange(24).reshape(2, 3, 4)
print("x.shape =", x.shape)

y = x.transpose(2, 0, 1)
print("y.shape =", y.shape)
print("共享内存?", np.shares_memory(x, y))
x = np.arange(24).reshape(2, 3, 4)
y = x.transpose(2, 0, 1)

print(y[1, 0, 2], x[0, 2, 1])    # 应该相等