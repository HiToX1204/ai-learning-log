# NumPy 题单 02：运算 / dtype / copy-view / 迭代 / where·sort·筛选
# 用法：先写预测，再跑。只记"预测与实际不一致"的题。

import numpy as np

a = np.array([1, 2, 3])

# Q1 逐元素运算
print(a * 2)
# 预测:

# Q2 标量广播（后面会正式学）
print(a + 10)
# 预测:

# Q3 两个数组逐元素相乘
print(a * np.array([1, 0, 2]))
# 预测:

# Q4 平方
print(a ** 2)
# 预测:

# Q5 比较返回什么
print(a > 1)
# 预测:

# Q6 除法：类型会变吗
print(a / 2)
# 预测:

# Q7 溢出（有坑）
u = np.array([1, 2, 3], dtype=np.uint8)
print(u * 100)
# 预测:

# Q8 dtype 推断：整数
print(np.array([1, 2, 3]).dtype)
# 预测:

# Q9 dtype 推断：混合
print(np.array([1.0, 2]).dtype)
# 预测:

# Q10 dtype 推断：布尔
print(np.array([True, False]).dtype)
# 预测:

# Q11 dtype 推断：字符串
print(np.array(['apple', 'banana']).dtype)
# 预测:

# Q12 类型转换
print(a.astype(np.float32).dtype)
# 预测:

# ── copy / view ──────────────────────────────
arr = np.array([10, 20, 30, 40, 50])

# Q13 切片是视图还是复制？
s = arr[1:4]
s[0] = 99
print(arr)
# 预测:

# Q14 .copy() 之后改
c = arr.copy()
c[0] = -1
print(arr)
# 预测:

# Q15 .view() 之后改
v = arr.view()
v[0] = -1
print(arr)
# 预测:

# Q16 切片的 base 是不是原数组
s2 = arr[1:3]
print(s2.base is arr)
# 预测:

# ── 迭代 ─────────────────────────────────────
m = np.arange(6).reshape(2, 3)

# Q17 迭代二维数组，每次拿到什么
for row in m:
    print(row)
# 预测:

# ── where / sort / 筛选 ───────────────────────
b = np.array([5, 2, 8, 2, 9])

# Q18 where 找等值
print(np.where(b == 2))
# 预测:

# Q19 where 找偶数
print(np.where(b % 2 == 0))
# 预测:

# Q20 二维排序：按行还是按列
print(np.sort([[3, 1], [2, 0]]))
# 预测:

# Q21 综合：筛出及格的分数
scores = np.array([45, 82, 60, 59, 91, 73])
print(scores[scores >= 60])
print(np.where(scores >= 60)[0])
print(np.sort(scores))
# 预测:

# Q22 反例：list 没有这些行为
lst = [1, 2, 3]
print(lst * 2)
# 预测:
