# ai-learning-log

记录我从零开始补 AI 基础的过程：每天的练习代码、学习笔记、以及阶段性产出。

## 这份仓库是什么

一个 AI 本科生的学习轨迹，不是教程仓库。所有代码都是我自己写的（AI 只用来解释概念和 review），每天一次 commit，失败和卡住的地方也记在笔记里。

目标：把数组运算、表格数据处理、微积分直觉、自动微分、PyTorch、Transformer 这一条链走通，最终产出三个可复现的小项目。

## 目录

| 路径 | 内容 |
|---|---|
| `01-array-basics/` | NumPy：形状、索引、切片、布尔掩码、统计 |
| `02-array-advanced/` | 视图与复制、广播、向量化、聚合与 axis |
| `03-pandas/` | 表格数据：筛选、groupby、CSV 读写、缺失值 |
| `04-calculus/` | 导数、数值求导、梯度下降、学习率 |
| `05-micrograd/` | 手写自动微分与反向传播（进行中） |
| `06-pytorch/` | PyTorch 对照、训练闭环、loss 曲线 |
| `topics/` | **专题笔记**：一个主题一篇，可独立阅读 |
| `notes/` | 每日详细笔记（公式、数据、踩过的坑） |
| `logs/` | 月度日志与阶段进度 |
| `plan/` | 学习计划与 X Program 课题门槛分析 |

## 专题笔记（`topics/`）

| 文件 | 内容 |
|---|---|
| [`topics/神经网络的理解.md`](topics/神经网络的理解.md) | 两层解释（小白 + 专业）：数据/损失/梯度/反向传播/训练闭环；术语对应表、易错点、自检题 |
| [`topics/非线性的必要性.md`](topics/非线性的必要性.md) | 以本人提出的三个观点为主线：分段线性逼近、封闭性、泰勒类比、ReLU 与深度、激活函数选择 |

## 环境与运行

```bash
# 环境：Python 3.14 + 单一 venv（uv 管理）
uv venv .venv --python 3.14
uv pip install numpy opencv-python pandas torch

# 运行任一练习
uv run python 01-array-basics/lesson02.py
```

- 只有一个环境：`.venv`（3.14），所有脚本共用
- 不需要激活虚拟环境，`uv run` 会自动找到 `.venv`；VS Code 里直接点 ▶ 也可以
- PowerShell 侧禁用了 venv 自动激活（系统禁止运行未签名的 `.ps1`），不影响运行

## 进度

- **阶段 A 完成**（2026-11-16）：NumPy 形状与索引 · 视图/广播/向量化 · pandas · 导数与梯度下降
- **阶段 B 进行中**：手写 autograd（micrograd）
- 后续：ML 概念与评估 → PyTorch 与 CNN → Transformer

详细进度见 [`logs/PROGRESS.md`](logs/PROGRESS.md)，每日记录见 [`notes/`](notes/)。

## 一些可复现的小结论

- 向量化 vs Python 循环（100 万个数求和）：**0.0006s vs 0.0516s，约 86 倍**
- 固定阈值在光照渐变下失效：检出像素 **1922 → 0**；改用 Canny 边缘判据后 **360 → 360**
- 梯度下降学习率：`lr=0.5` 一步到 0，`lr=1.1` 二十步发散到 383
