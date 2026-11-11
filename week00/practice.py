import numpy as np

a = np.arange(20).reshape(4, 5)
print(a.size)
print(a.shape)
print(a.sum(axis=0))
print(a.mean(axis=1))
print((a > 10).sum())
print((a > 10).mean())
print(a.argmax())
print(a.reshape(5, 4).sum(axis=1))
print(a.T.shape)