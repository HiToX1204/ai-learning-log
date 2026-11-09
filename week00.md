\# Week 0 · 预备

\- 现在的环境：Windows，git 有，conda 没有，python 是坏的（WindowsApps 占位符）

\- 今天要做完：装 conda、建环境、跑通 import numpy

\- 我的 Python 水平：能做一点题，祝明天ics考试顺利

\- 我的 C++ 水平：能读懂 OpenCV 项目，写起来慢

\- 两周后的验收：一个别人能 clone 跑起来的仓库


## Week 0 · NumPy 与阈值

- 掩码索引 `arr[mask]` 会把结果压成一维，`.mean()` 的分母是选中个数
- 阈值扫描是阶梯函数：每级台阶 = 一块目标的面积
- Otsu 只适合"双峰、前景背景比例接近"的图；黑底占 85% 时它会选 0
- 固定阈值失效的两个原因：光照变化 + 前景多层灰度
