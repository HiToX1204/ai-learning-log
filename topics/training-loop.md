# 训练闭环：从 loss 到收敛

> 来源：2026-11-19 第一个训练脚本（线性回归）
> 数据：$x = [0,1,2,3,4]$，$y = [1,3,5,7,9]$（即 $y = 2x+1$）

---

## 一、训练的三要素

| 要素 | 是什么                  | 代码                     |
| ---- | ----------------------- | ------------------------ |
| 数据 | $(x_i, y_i)$          | `X`, `Y`             |
| 模型 | 参数化函数 $f_\theta$ | `pred = w*x + b`       |
| 损失 | 预测有多错              | `((pred-y)**2).mean()` |

**训练 = 反复调整 $\theta$，让损失变小。**

---

## 二、梯度的手工推导

$$
\mathcal{L} = \frac{1}{n}\sum_{i=1}^{n}\left(\hat{y}_i - y_i\right)^2,
\qquad \hat{y}_i = w x_i + b
$$

对 $w$：

$$
\frac{\partial \mathcal{L}}{\partial w}
= \frac{2}{n}\sum_i\left(\hat{y}_i - y_i\right)\frac{\partial \hat{y}_i}{\partial w}
= \frac{2}{n}\sum_i\left(\hat{y}_i - y_i\right)x_i
$$

对 $b$（因为 $\partial \hat{y}_i/\partial b = 1$）：

$$
\frac{\partial \mathcal{L}}{\partial b}
= \frac{2}{n}\sum_i\left(\hat{y}_i - y_i\right)
$$

初始 $\hat{y}=0$，$\hat{y}_i - y_i = -y_i$：

$$
\frac{\partial \mathcal{L}}{\partial w} = -2\,\overline{yx} = -2\times14 = -28,
\qquad
\frac{\partial \mathcal{L}}{\partial b} = -2\,\overline{y} = -2\times5 = -10
$$

**实测完全一致。**

---

## 三、学习率：步长，不是精度旋钮

$$
\theta \leftarrow \theta - \eta\,\nabla_\theta\mathcal{L}
$$

### 四种取值实测（20 步，目标 $w=2, b=1$）

| $\eta$ | $w$                  | $b$                  | 现象           |
| -------- | ---------------------- | ---------------------- | -------------- |
| 0.001    | 0.494                  | 0.177                  | 太慢，几乎没动 |
| 0.01     | 1.965                  | 0.723                  | 接近但不够     |
| 0.1      | 2.027                  | 0.922                  | 最好           |
| 1.0      | $-1.55\times10^{21}$ | $-5.45\times10^{21}$ | **发散** |

### 为什么 $\eta=1.0$ 会爆炸

线性回归的更新是线性动力系统：

$$
\theta_{k+1} - \theta^\* = \left(I - \eta H\right)\left(\theta_k - \theta^\*\right),
\qquad H = \frac{2}{n}X^{\top}X
$$

本数据：

$$
H = \begin{bmatrix} 12 & 4 \\ 4 & 2\end{bmatrix},
\quad \lambda_{\max}\approx13.40,\ \lambda_{\min}\approx0.597
$$

稳定性条件：

$$
0 < \eta < \frac{2}{\lambda_{\max}} \approx 0.149
$$

- $\eta=0.1 < 0.149$ → 收敛 ✅
- $\eta=1.0$ 远超阈值 → 增长因子 $|1-13.4|=12.4$，$12.4^{20}\approx10^{22}$ ✅ 与实测同量级

**直观版**：步长 > 到谷底的距离 → 冲过头；每次冲得更远 → 越弹越高。

（更通俗的一维版：$f=x^2$ 时 $x_{k+1}=(1-2\eta)x_k$，收敛要求 $0<\eta<1$。）

---

## 四、收敛速率与 loss 曲线

参数误差按因子 $1-\eta\lambda$ 衰减，而损失是误差的平方，所以损失的衰减因子是**平方**：

$$
r_{\text{loss}} = \left(1-\eta\lambda_{\text{slow}}\right)^2
$$

本数据：$\left(1-0.1\times0.597\right)^2 = 0.94^2\approx 0.884$

**从图上反推**：step 10 约为 $10^{-2}$，step 200 约为 $10^{-12}$，跨 10 个数量级 / 190 步：

$$
r \approx 10^{-10/190}\approx 0.886
$$

**实测 0.886 vs 理论 0.884** —— 双层闭环（梯度对拍 + 收敛速率）。

### 对数纵轴为什么必要

$\mathcal{L}_k\approx\mathcal{L}_0 r^k$ 取对数：

$$
\log\mathcal{L}_k = \log\mathcal{L}_0 + k\log r
$$

**是直线**。线性纵轴下后 190 步全挤在 0 附近，看不出结构。

---

## 五、末尾的抖动：浮点地板

200 步时 loss 约 $10^{-12}$，误差约 $10^{-6}$，接近 `float64` 的机器精度（$\sim2\times10^{-16}$ 相对误差）。**曲线末端不平滑不是算法问题，是数值极限。**

这也解释了"$\eta$ 太小会停滞"：更新量 $\eta g$ 可能小于参数的 ulp。

---

## 六、"用极小 $\eta$ 迭代到极高精度"可行吗

**纯优化意义上可行，但机器学习里不是最优：**

| 问题     | 说明                                                                                                              |
| -------- | ----------------------------------------------------------------------------------------------------------------- |
| 步数成本 | 线性收敛，$\eta$ 小 10 倍 → 步数多 10 倍（到 $10^{-6}$：$\eta=0.1$ 需 226 步，$\eta=0.001$ 需 23000 步） |
| 浮点停滞 | $\eta g < \text{ulp}(\theta)$ 时参数不再变化                                                                    |
| 目标错位 | 训练 loss 不是目标；真实数据有噪声，最优 loss $\neq 0$                                                          |
| 过拟合   | 压训练 loss 会拟合噪声，泛化变差                                                                                  |

**更好的做法**：

1. 学习率衰减（先大后小）
2. 自适应优化器（Adam/RMSprop 按参数维度缩放步长）
3. 验证集早停（不是"训练 loss 足够小"）
4. 凸问题用闭式解：$\theta^\* = (X^{\top}X)^{-1}X^{\top}y$（零迭代）

**结论：小学习率是"保险"，不是"最优"。**

---

## 七、标准训练循环模板

```python
for step in range(n_steps):
    pred = model(x)                 # 前向
    loss = loss_fn(pred, y)         # 算损失
    loss.backward()                 # 反向（累积梯度）
    with torch.no_grad():           # 更新参数，不建图
        for p in params:
            p -= lr * p.grad
            p.grad.zero_()          # 必须清零（AccumulateGrad）
```

**四个动作：前向 → 损失 → 反向 → 更新（含清零）。**
