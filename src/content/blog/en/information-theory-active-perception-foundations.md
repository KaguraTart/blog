---
title: "Active perception from an information theory perspective: Fisher Information and Cramér-Rao lower bounds"
description: "Explain the information theory foundation of active sensing from first principles: Fisher Information, Cramér-Rao lower bound, mutual information, and its application in SLAM work such as FIT-SLAM and Continuous Info Modeling."
tags: ["UAV", "active perception", "information theory", "Fisher Information", "SLAM", "Cramér-Rao"]
category: "Tech"
pubDate: 2026-04-27
---

# Active perception from an information theory perspective: Fisher Information and Cramér-Rao lower bound

> **UAV Perception Planning Series · Part X**
> Focus: Basics of information theory, active sensing framework, calculation of Fisher Information and application in SLAM

---

## 1. What is active perception?

Traditional perception is **passive**: the robot receives sensor data and updates a model of the environment.

**Active perception** goes one step further: the robot **actively chooses "where to look"** to maximize the value of the task.

```
被动感知：
传感器 → 数据 → 地图更新（机器人不动）

主动感知：
当前地图 → 信息价值评估 → 最优下一视角选择 → 移动 → 传感器 → 地图更新
                ↑
           核心问题：如何量化"信息价值"？
```

For UAVs, active sensing is particularly critical:
- **Energy Constraint**: Flying consumes energy and cannot fly randomly.
- **Wide field of view**: When moving in the air, the field of view changes drastically, and it is crucial to choose the optimal path.
- **Three-dimensional space**: Buildings, mountains, and trees all need to be observed from multiple angles for complete modeling.

---

## 2. Mathematical foundation of information theory

### 2.1 Fisher Information

Given a probability model $p(x|\theta)$, where $\theta$ is the parameter to be estimated, **Fisher Information** measures the amount of information about $\theta$ carried by the observation data $X$:

$$
I(\theta) = \mathbb{E}_X \left[ \left( \frac{\partial}{\partial \theta} \log p(X|\theta) \right)^2 \right] = - \mathbb{E}_X \left[ \frac{\partial^2}{\partial \theta^2} \log p(X|\theta) \right]
$$

**Intuitive understanding:**
- If $\log p(x|\theta)$ changes **very steeply** near $\theta$, it means that the data is very sensitive to $\theta$ → Fisher Information **large**
- If $\log p(x|\theta)$ changes **flat** around $\theta$, the data is not sensitive to $\theta$ → Fisher Information **small**

**Scalar vs matrix form:**- Scalar: $I(\theta)$ (one-dimensional parameter)
- Matrix: **Fisher Information Matrix (FIM)** $I(\boldsymbol{\theta})$ (multidimensional parameters)

$$
[I(\boldsymbol{\theta})]_{ij} = \mathbb{E} \left[ \frac{\partial \log p(X|\boldsymbol{\theta})}{\partial \theta_i} \cdot \frac{\partial \log p(X|\boldsymbol{\theta})}{\partial \theta_j} \right]
$$

FIM is the Riemannian metric tensor in parameter space, which determines how accurately you can estimate parameters.

---

### 2.2 Cramér-Rao Lower Bound (CRLB)

The Cramér-Rao lower bound is a core application of Fisher Information: **gives an optimal lower bound on the variance of an unbiased estimator**.

$$
\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}
$$

**Physical meaning:** No matter what estimation method you use (as long as it is unbiased), the estimation accuracy cannot exceed $1/I(\theta)$.

**Meaning in SLAM:**
- The lower bound of the covariance of the robot pose $\mathbf{x}$ is determined by FIM
- $[\text{Cov}(\mathbf{x})]^{-1} \preceq I(\mathbf{x})$
- The smaller the inverse of FIM → the more accurate the estimate

---

### 2.3 Mutual Information

Mutual information measures the statistical dependence between two random variables:

$$
I(X; Y) = \int \int p(x,y) \log \frac{p(x,y)}{p(x)p(y)} \, dx \, dy = H(X) - H(X|Y)
$$

**Meaning in active perception:**
- $X$ = future sensor observations
- $Y$ = the uncertain state of the current map

Maximizing $I(X; Y)$ = choosing the perspective where future observations will best reduce uncertainty in the current map.This is the information theory definition of "**Information Gain**" in active perception.

---

## 3. Active sensing framework

### 3.1 Core issue: Next-Best-View (NBV)

The core problem of active sensing is **NBV planning**: given the currently observed area, where should we move next to most effectively reduce uncertainty?

**Mathematical form of the NBV problem:**

$$
a^* = \arg\max_{a \in \mathcal{A}} \quad \mathbb{E}_{z \sim p(z|x, a)} \left[ \log \det I(\theta_{new}(x, z)) \right] - \log \det I(\theta_{old}(x))
$$

That is: Choose action $a$ such that the determinant of the FIM (a scalar measure of the overall uncertainty) after execution is maximized.

---

### 3.2 Three major components of active sensing system

**Information theory active perception framework** proposes three components of an active perception system:

```
┌─────────────────────────────────────────────────────────┐
│                   Active Perception System              │
│                                                         │
│  Component 1: 状态估计 & 地图表示                        │
│  (State Estimation & Map Representation)               │
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

**Why do you need Component 2 (generated model)? **
- You can't really fly out and try every location (too expensive)
- You need a model to "imagine" what you would see by flying to each candidate location
- **NeRF/3DGS are perfect generative models** (already written about it in your blog!)

---

## 4. Application of Fisher Information in SLAM

### 4.1 FIM in SLAM

In visual SLAM, the robot needs to simultaneously estimate:
- **Pose** $\mathbf{x}_k$ (where is the camera)
- **Map Point** $\mathbf{m}_i$ (where is the 3D point in space)

Observation model: $z_{k,i} = h(\mathbf{x}_k, \mathbf{m}_i) + \mathbf{n}$

- $h(\cdot)$ is the projection function (3D → 2D image coordinates)
- $\mathbf{n} \sim \mathcal{N}(0, \Sigma)$ is the measurement noise**Observed Fisher Information:**
$$
I(\mathbf{x}_k, \mathbf{m}_i) = \frac{\partial h^\top}{\partial [\mathbf{x}_k, \mathbf{m}_i]} \Sigma^{-1} \frac{\partial h}{\partial [\mathbf{x}_k, \mathbf{m}_i]}
$$

**Key Insights:**
- Observing the same 3D point, **different perspectives** produce different Fisher Information
- The deeper the observation depth (the further away), the smaller the amount of information
- The larger the observation baseline (the greater the change in viewing angle), the greater the amount of information

**This is why UAVs need to actively choose their perspective! **

---

### 4.2 Interpretation of classic papers

#### **FIT-SLAM (arXiv, January 2024)**

**Paper:** *FIT-SLAM -- Fisher Information and Traversability estimation-based Active SLAM for exploration in 3D environments*
**Author:** Suchetan Saravanan, Corentin Chauffaut, Caroline Chanel, Damien Vivet
**Source:** arXiv:2401.09322, January 2024

**Core contribution:**
- Explicitly introduce **Fisher Information** into the objective function of **Active SLAM**
- Also consider **Traversability** - not just "see clearly", but also "fly"
- Targeted at **3D environment** (non-planar), suitable for UAV exploration in complex urban canyons

**Note:** This paper was published on arXiv (it was submitted to IEEE ICARA 2024). No clear publication record has been found at the top conference. The arXiv version should be noted when citing.

---#### **Active View Planning for Visual SLAM: Continuous Information Modeling (arXiv, 2022/2023)**

**Paper:** *Active View Planning for Visual SLAM in Outdoor Environments Based on Continuous Information Modeling*
**Author:** Zhihao Wang, Haoyao Chen, Shiwu Zhang, Yunjiang Lou
**Source:** arXiv:2211.xxxxx, 2022

**Core contribution:**
- Proposed **continuous information modeling** to replace discrete information grids
- Optimize the next view on a continuous space rather than a discrete set of candidate points
- Model spatial uncertainty using **Gaussian Process (GP)**

**Key Insights:**

Traditional methods discretize space into candidate points → information gain is only evaluated on this limited set of points

Continuous method: Use GP to represent "the amount of information at any position", and then **directly optimize in the continuous space**

$$
\mu(a) = \text{GP predicted action} a \text{The amount of information at} \\
\sigma(a) = \text{Prediction uncertainty of GP} \\
\text{Acquisition function: } a^* = \arg\max_a \, \mu(a) + \beta \sigma(a)
$$

**Advantages over UAV:**
- The motion space of UAV is continuous and should not be forced to discretize
- Ability to optimize complete 6-DoF trajectories rather than just discrete waypoint selections

---

## 5. Information gain calculation for active sensing

### 5.1 Information gain based on Fisher Information

**Information Gain** = FIM change before and after the action:

$$
\Delta I(a) = \det I(\theta_{after}) - \det I(\theta_{before})
$$But the actual calculation does not require actual reconstruction, just:
1. Predicting observations from a new perspective
2. Calculate FIM of newly added observations
3. Use **Schur complement** to efficiently update the total FIM

### 5.2 Monte Carlo estimation of Mutual Information

Mutual information $I(X; Y)$ usually cannot be calculated analytically and requires the use of Monte Carlo methods:

$$
\hat{I}(X; Y) = \frac{1}{N} \sum_{i=1}^N \log \frac{p(x_i|y_i)}{p(x_i)}
$$

In active perception:
- Sample multiple possible map versions from an uncertain distribution of the current map
- For each candidate action, calculate the **average mutual information**
- Select the action with the largest mutual information

---

## 6. Information theory vs other principles

| Guidelines | Advantages | Disadvantages |
|------|------|------|
| **Fisher Information** | Theoretical optimal, tight lower bound | Complex calculation, requires probabilistic model |
| **Mutual Information** | Intuitive and simple to measure | Large estimation variance |
| **Entropy** | Intuitive | Cannot handle continuous distributions |
| **Distance-based** | Simple and fast | Does not consider occlusion/appearance |
| **Coverage-based** | Simple | Does not consider information density |

**Best Practice:** Combining Multiple Criteria
- **SAFETY**: distance based collision checking
- **Efficiency**: entropy-based coverage
- **Accuracy**: FIM-based pose accuracy

---

## 7. Connections to your existing work

You already wrote in your blog:
- **NeRF/3DGS + UAV**: Environment representation (exactly Component 1 of active sensing!)
- **Semantic SLAM**: Maps with semantics (Semantic FIM > Geometric FIM)
- **Digital Twin**: a real-time updated environment model

**This means:**
You already have the **map representation layer** of the active sensing framework, and by adding the **information gain evaluation layer**, you can build a complete active sensing system!

**Natural extension:**
```
你已有的 NeRF/3DGS 地图
    ↓ + FIT-SLAM 的 FIM 计算方法
    ↓ + GP-based continuous NBV 优化
= 你的主动感知 UAV 系统
```

---

## 📚 References1. Saravanan et al. *FIT-SLAM -- Fisher Information and Traversability estimation-based Active SLAM for exploration in 3D environments*. arXiv:2401.09322, January 2024.
2. Wang et al. *Active View Planning for Visual SLAM in Outdoor Environments Based on Continuous Information Modeling*. arXiv, 2022.
3. Chen et al. *ActiveGAMER: Active Gaussian Mapping through Efficient Rendering*. arXiv:2501.06897, January 2025.
4. Lee et al. *SO-NeRF: Active View Planning for NeRF using Surrogate Objectives*. arXiv:2312.XXXXX, December 2023.
5. He et al. *Active Perception using Neural Radiance Fields*. arXiv:2310.09892, October 2023.
6. Marza et al. *AutoNeRF: Training Implicit Scene Representations with Autonomous Agents*. arXiv, 2024.
7. Chaplot et al. *Learning Visual Exploration for Long-Range Navigation*. NeurIPS, 2020.