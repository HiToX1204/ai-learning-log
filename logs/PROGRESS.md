# 阶段进度（自用）

> 计划：`plan/学习计划-15周.md` ｜ 门槛依据：`plan/XProgram门槛分析.md`
> 目标：2027 年 2–3 月达到 X Program 课题表中 90% AI 强相关课题的显式门槛

## 我在哪

| 项 | 状态 |
|---|---|
| 当前阶段 | **B — micrograd（手写 autograd 与反向传播）** |
| 起点 | 2026-11-06 |
| 阶段 A 完成 | 2026-11-16 |

## 五个阶段

| 阶段 | 内容 | 状态 |
|---|---|---|
| A | NumPy 收尾 · pandas · 导数与梯度直觉 | ✅ 11-16 |
| B | micrograd：手写 autograd 与反向传播 | 🚧 核心完成（11-17） |
| C | ML 概念 · 过拟合 · 评估指标 | ⬜ |
| C+ | 视频：Karpathy 第 1 讲（已看前 1 小时） | ⬜ 余 1:10 后段 |
| D | PyTorch 训练循环 · CNN | ⬜ |
| E | Transformer · mini-GPT · 复现报告 | ⬜ |
| F | 三个仓库打包 · 简历 | ⬜ |

### 阶段 B 明细

| 内容 | 状态 |
|---|---|
| `Value` 类骨架 + `__add__` / `__mul__` | ✅ 11-17 |
| `_backward`（加法 / 乘法）+ 闭包 | ✅ 11-17 |
| 数值求导对拍（误差 1e-9） | ✅ 11-17 |
| `backward()` + 后序拓扑排序 | ✅ 11-17 |
| 更多运算：`__sub__` `__pow__` `__neg__` `__radd__` | ⬜ |
| `tanh` 节点 + 一个神经元 | ⬜ |
| 小 MLP + 与 PyTorch 对照 | ⬜ |

## 产出

| 阶段 | 仓库/文件 | 状态 |
|---|---|---|
| A | `01-array-basics/` `02-array-advanced/` `03-pandas/` `04-calculus/` | ✅ |
| A | `plan/XProgram门槛分析.md`（288 课题门槛统计） | ✅ |
| B | `05-micrograd/` | 🚧 |
| C | `06-ml-basics/` | ⬜ |
| D | `07-pytorch/` | ⬜ |
| E | `08-transformer/` | ⬜ |
