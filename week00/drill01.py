# NumPy 题单 01：创建 / 索引 / 切片
# 用法：每题先在心里（或纸上）写答案，填到 "# 预测:" 后面，再运行对照。
# 不要先跑再看答案——那样练的是眼熟，不是记忆。

import numpy as np

a = np.arange(10)

# # Q1
# print(a)
# # 预测:【

# # Q2
# print(a[2:8:2])
# # 预测:

# # Q3
# print(a[::3])
# # 预测:

# # Q4
# print(a[-3:])
# # 预测:

# # Q5
# print(a[::-1])
# # 预测:

# # Q6
# print(a[2:5:-1])
# # 预测:            ← 这题有坑，注意起点终点方向

b = np.arange(12).reshape(3, 4)

# # Q7
# print(b[1])
# # 预测:

# # Q8
# print(b[:, 1])
# # 预测:

# # Q9
# print(b[0:2, 1:3])
# # 预测:

# Q10
print([b > 5])
# 预测:

# Q11
print(b[[0, 2]])
# 预测:

# Q12
print(b[:, [0, 3]])
# 预测:

# Q13
print(b.sum(axis=0))
# 预测:

# Q14
print(b.sum(axis=1))
# 预测:

# Q15
c = b[0]          # 注意：这是视图还是复制？
c[0] = 99
print(b[0, 0])
# 预测:

# Q16
print(np.eye(3))
# 预测:

# Q17
print(np.full((2, 2), 7).dtype)
# 预测:

# Q18
print(np.arange(0, 1, 0.3))
# 预测:
