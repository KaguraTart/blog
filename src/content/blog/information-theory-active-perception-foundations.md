---
title: '信息论视角的主动感知：Fisher Information 与 Cramér-Rao 下界'
tags: ['UAV', '主动感知', '信息论', 'Fisher Information', 'SLAM', 'Cramér-Rao']
category: Tech
pubDate: '2026-04-27'
---

# 信息论视角的主动感知：Fisher Information 与 Cramér-Rao 下界

> **UAV 感知规划系列 · 第X篇**
> 聚焦：信息论基础、主动感知框架、Fisher Information 的计算与在 SLAM 中的应用

---

## 1. 什么是主动感知？

传统感知是**被动**的：机器人接收传感器数据，更新环境模型。

**主动感知**则更进一步：机器人**主动选择"看哪里"**，以最大化任务价值。

```
被动感知：
传感器 → 数据 → 地图更新（机器人不动）

主动感知：
当前地图 → 信息价值评估 → 最优下一视角选择 → 移动 → 传感器 → 地图更新
                ↑
           核心问题：如何量化"信息价值"？
```

对于 UAV 来说，主动感知尤为关键：
- **能量约束**：飞行消耗能量，不能随意乱飞
- **视野广阔**：在空中移动时视野剧变，选择最优路径至关重要
- **三维空间**：建筑、山体、树木都需要从多个角度观测才能完整建模

---

## 2. 信息论的数学基础

### 2.1 Fisher Information（费舍尔信息）

给定一个概率模型 $p(x|\theta)$，其中 $\theta$ 是待估计的参数，**Fisher Information** 衡量的是观测数据 $X$ 携带的关于 $\theta$ 的信息量：

$$
I(\theta) = \mathbb{E}_X \left[ \left( \frac{\partial}{\partial \theta} \log p(X|\theta) \right)^2 \right] = - \mathbb{E}_X \left[ \frac{\partial^2}{\partial \theta^2} \log p(X|\theta) \right]
$$

**直观理解：**
- 如果 $\log p(x|\theta)$ 在 $\theta$ 附近变化**很陡**，说明数据对 $\theta$ 很敏感 → Fisher Information **大**
- 如果 $\log p(x|\theta)$ 在 $\theta$ 附近变化**很平**，数据对 $\theta$ 不敏感 → Fisher Information **小**

**标量 vs 矩阵形式：**

- 标量：$I(\theta)$（一维参数）
- 矩阵：**Fisher Information Matrix（FIM）** $I(\boldsymbol{\theta})$（多维参数）

$$
[I(\boldsymbol{\theta})]_{ij} = \mathbb{E} \left[ \frac{\partial \log p(X|\boldsymbol{\theta})}{\partial \theta_i} \cdot \frac{\partial \log p(X|\boldsymbol{\theta})}{\partial \theta_j} \right]
$$

FIM 是参数空间中的黎曼度量张量，决定了**你能把参数估计得多准**。

---

### 2.2 Cramér-Rao 下界（CRLB）

Cramér-Rao 下界是 Fisher Information 的核心应用：**给出了无偏估计器方差的最优下界**。

$$
\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}
$$

**物理意义：** 不管你用什么估计方法（只要是无偏的），估计精度都不可能超过 $1/I(\theta)$。

**在 SLAM 中的意义：**
- 机器人位姿 $\mathbf{x}$ 的协方差下界由 FIM 决定
- $[\text{Cov}(\mathbf{x})]^{-1} \preceq I(\mathbf{x})$
- FIM 的逆越小 → 估计越准确

---

### 2.3 Mutual Information（互信息）

互信息衡量两个随机变量之间的统计依赖性：

$$
I(X; Y) = \int \int p(x,y) \log \frac{p(x,y)}{p(x)p(y)} \, dx \, dy = H(X) - H(X|Y)
$$

**在主动感知中的意义：**
- $X$ = 未来的传感器观测
- $Y$ = 当前地图的不确定状态

最大化 $I(X; Y)$ = 选择让**未来观测最能减少当前地图不确定性**的视角。

这就是主动感知中"**信息增益（Information Gain）**"的信息论定义。

---

## 3. 主动感知框架

### 3.1 核心问题：Next-Best-View（NBV）

主动感知的核心问题是 **NBV 规划**：给定当前已观测区域，下一步应该移动到哪里才能最有效地减少不确定性？

**NBV 问题的数学形式：**

$$
a^* = \arg\max_{a \in \mathcal{A}} \quad \mathbb{E}_{z \sim p(z|x, a)} \left[ \log \det I(\theta_{new}(x, z)) \right] - \log \det I(\theta_{old}(x))
$$

即：选择动作 $a$，使得执行后 FIM 的行列式（衡量总体不确定性的标量）最大化。

---

### 3.2 主动感知系统的三大组件

**He et al. (ACC 2024)** 在 *Active Perception using Neural Radiance Fields* 中提出了主动感知系统的三组件框架：

```
┌─────────────────────────────────────────────────────────┐
│                   Active Perception System              │
│                                                         │
│  Component 1: 状态估计 & 地图表示                        │
│  (State Estimation & Map Representation)                 │
│  → 当前已观测区域的完整表示（几何 + 语义）               │
│                                                         │
│  Component 2: 未来观测合成                               │
│  (Generative Model of Future Observations)              │
│  → 给定候选动作，生成未来会看到的图像/传感器数据         │
│                                                         │
│  Component 3: 信息驱动的规划                              │
│  (Information-Driven Planning)                          │
│  → 在候选轨迹上计算互信息，选择最优                     │
└─────────────────────────────────────────────────────────┘
```

**为什么需要 Component 2（生成模型）？**
- 你不能真的飞过去试每个位置（成本太高）
- 你需要一个模型来"想象"飞到每个候选位置会看到什么
- **NeRF / 3DGS 是完美的生成模型**（已经在你的 blog 里写过！）

---

## 4. Fisher Information 在 SLAM 中的应用

### 4.1 SLAM 中的 FIM

在 visual SLAM 中，机器人需要同时估计：
- **位姿** $\mathbf{x}_k$（相机在哪里）
- **地图点** $\mathbf{m}_i$（空间中的 3D 点在哪里）

观测模型：$z_{k,i} = h(\mathbf{x}_k, \mathbf{m}_i) + \mathbf{n}$

- $h(\cdot)$ 是投影函数（3D → 2D 图像坐标）
- $\mathbf{n} \sim \mathcal{N}(0, \Sigma)$ 是测量噪声

**观测的 Fisher Information：**
$$
I(\mathbf{x}_k, \mathbf{m}_i) = \frac{\partial h^\top}{\partial [\mathbf{x}_k, \mathbf{m}_i]} \Sigma^{-1} \frac{\partial h}{\partial [\mathbf{x}_k, \mathbf{m}_i]}
$$

**关键洞察：**
- 观测同一个 3D 点，**不同视角**产生不同的 Fisher Information
- 观测深度越深（离得越远），信息量越小
- 观测基线越大（视角变化越大），信息量越大

**这就是为什么 UAV 需要主动选择视角！**

---

### 4.2 经典论文解读

#### **FIT-SLAM（arXiv, January 2024）**

**论文：** *FIT-SLAM -- Fisher Information and Traversability estimation-based Active SLAM for exploration in 3D environments*
**作者：** Suchetan Saravanan, Corentin Chauffaut, Caroline Chanel, Damien Vivet
**来源：** arXiv:2401.07504 | IROS 2024（投稿）

**核心贡献：**
- 将 **Fisher Information** 显式引入 **Active SLAM** 的目标函数
- 同时考虑**可通行性（Traversability）**——不只是"看得清楚"，还要"飞得到"
- 针对 **3D 环境**（非平面），适合 UAV 在复杂城市峡谷中的探索

**方法：**

```
传统 SLAM：
  minimize: Σ ||z - h(x,m)||²（重投影误差）

FIT-SLAM：
  minimize: Σ ||z - h(x,m)||² - λ · log det I(x,m)（重投影误差 - 信息增益）
```

**关键创新：**
1. **信息-可通行性联合优化**：信息增益大的位置如果飞不到，也没用
2. **3D-aware FIM 计算**：考虑 UAV 的完整六自由度运动
3. **地形可达性估计**：地形坡度、障碍物密度作为约束

---

#### **Active View Planning for Visual SLAM: Continuous Information Modeling（arXiv, 2022/2023）**

**论文：** *Active View Planning for Visual SLAM in Outdoor Environments Based on Continuous Information Modeling*
**作者：** Zhihao Wang, Haoyao Chen, Shiwu Zhang, Yunjiang Lou
**来源：** arXiv:2211.xxxxx | ICRA/IROS 2023

**核心贡献：**
- 提出**连续信息建模**替代离散信息网格
- 在连续空间而非离散候选点集上优化下一视角
- 使用 **Gaussian Process（GP）** 建模空间不确定性

**关键洞察：**

传统方法把空间离散化为候选点 → 信息增益只在这有限点集上评估

连续方法：用 GP 表示"任意位置的信息量"，然后**在连续空间直接优化**

$$
\mu(a) = \text{GP 预测的 action } a \text{ 处的信息量} \\
\sigma(a) = \text{GP 的预测不确定度} \\
\text{Acquisition function: } a^* = \arg\max_a \, \mu(a) + \beta \sigma(a)
$$

**在 UAV 上的优势：**
- UAV 的运动空间是连续的，不应该被强制离散化
- 可以优化完整的 6-DoF 轨迹，而非仅选择离散航点

---

## 5. 主动感知的信息增益计算

### 5.1 基于 Fisher Information 的信息增益

**信息增益（Information Gain）** = 动作前后的 FIM 变化：

$$
\Delta I(a) = \det I(\theta_{after}) - \det I(\theta_{before})
$$

但实际计算时不需要真的重建，只需：
1. 预测新视角下的观测
2. 计算新增观测的 FIM
3. 用 **Schur complement** 高效更新总 FIM

### 5.2 Mutual Information 的蒙特卡洛估计

互信息 $I(X; Y)$ 通常无法解析计算，需要用蒙特卡洛方法：

$$
\hat{I}(X; Y) = \frac{1}{N} \sum_{i=1}^N \log \frac{p(x_i|y_i)}{p(x_i)}
$$

在主动感知中：
- 从当前地图的不确定分布中**采样**多个可能的地图版本
- 对每个候选动作，计算**平均互信息**
- 选择互信息最大的动作

---

## 6. 信息论 vs 其他准则

| 准则 | 优点 | 缺点 |
|------|------|------|
| **Fisher Information** | 理论最优、紧下界 | 计算复杂、需概率模型 |
| **Mutual Information** | 直观、测量简单 | 估计方差大 |
| **Entropy（熵）** | 直观 | 无法处理连续分布 |
| **Distance-based（距离）** | 简单快速 | 不考虑遮挡/外观 |
| **Coverage-based（覆盖率）** | 简单 | 不考虑信息密度 |

**最佳实践：** 组合多个准则
- **安全性**：基于 distance 的碰撞检查
- **效率**：基于 entropy 的覆盖率
- **精度**：基于 FIM 的位姿精度

---

## 7. 与你已有工作的连接

你在 blog 中已经写了：
- **NeRF/3DGS + UAV**：环境表示（正是主动感知的 Component 1！）
- **语义 SLAM**：带语义的地图（语义 FIM > 几何 FIM）
- **数字孪生**：实时更新的环境模型

**这意味着：**
你已有主动感知框架的 **地图表示层**，再加 **信息增益评估层** 就能搭一个完整的主动感知系统！

**自然延伸：**
```
你已有的 NeRF/3DGS 地图
    ↓ + FIT-SLAM 的 FIM 计算方法
    ↓ + GP-based continuous NBV 优化
= 你的主动感知 UAV 系统
```

---

## 📚 参考文献

1. He et al. *Active Perception using Neural Radiance Fields*. ACC 2024. arXiv:2310.09892.
2. Saravanan et al. *FIT-SLAM -- Fisher Information and Traversability estimation-based Active SLAM for exploration in 3D environments*. arXiv:2401.07504, January 2024.
3. Wang et al. *Active View Planning for Visual SLAM in Outdoor Environments Based on Continuous Information Modeling*. arXiv:2211.xxxxx, 2022/2023.
4. Lee et al. *SO-NeRF: Active View Planning for NeRF using Surrogate Objectives*. arXiv:2312.xxxxx, December 2023.
5. Pan et al. *How Many Views Are Needed to Reconstruct an Unknown Object Using NeRF?* ICRA/IROS 2024.
6. Marza et al. *AutoNeRF: Training Implicit Scene Representations with Autonomous Agents*. ICRA 2024.
7. Chen et al. *Active Human Pose Estimation via an Autonomous UAV Agent*. IROS 2024.
