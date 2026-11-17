import numpy as np
import time

n = 1_000_000
r = np.random.rand(n)          # 100 万个随机数

t0 = time.perf_counter()
s1 = 0.0
for v in r:
    s1 += v
t1 = time.perf_counter()

t2 = time.perf_counter()
s2 = r.sum()
t3 = time.perf_counter()

print("循环:", round(t1 - t0, 4), "秒, 结果", s1)
print("向量化:", round(t3 - t2, 4), "秒, 结果", s2)
print("相差", round((t1 - t0) / (t3 - t2), 1), "倍")