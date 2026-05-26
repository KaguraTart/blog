---
title: "Next-Best-View Planning Meets NeRF/3DGS: The Information Frontier of Active Sensing"
description: "Detailed explanation of NBV + NeRF/3DGS cutting-edge methods: ActiveGAMER active Gaussian mapping, SO-NeRF proxy target, AutoNeRF autonomous data collection, covering the intersection frontier of active sensing and neural radiation fields"
tags: ["UAV", "NRF", "3DGS", "Next-Best-View", "active perception", "Gaussian Splatting"]
category: "Tech"
pubDate: 2026-04-27
sourceHash: "030ae332e13fe029a870b23907cdc45d7e0018c0"
---

# Next-Best-View Planning meets NeRF/3DGS: the information frontier of active sensing

> **UAV Perception Planning Series·Part X+1**
> Focus: NBV + NeRF/3DGS cutting-edge methods, ActiveGAMER, SO-NeRF, air-ground active exploration

---

## 1. Core Concept: Why is NeRF/3DGS a perfect partner for NBV?

Traditional NBV planning has a fatal weakness: **It doesn't know "what the invisible looks like"**.

You are inferring where the most information is based on current observations - but for places that have not been observed, you can only rely on heuristics ("pick a place you have never been to").

**NeRF/3DGS changes this:**

```
传统方法：
  "我前方10米有个物体，但背面我完全看不到"
  → 只能假设背面 = 未知，启发式选个点去看看

NeRF/3DGS：
  "我有个神经辐射场，已经隐式编码了前+背面的大致形状"
  → 可以渲染背面的大致外观，评估信息增益的真实上限
```

This is why **NeRF/3DGS is perfect as a "generative model"** for active sensing - it can "imagine" what an unobserved region would look like from any viewing angle and be used to calculate the true information gain.

---

## 2. ActiveGAMER: Active Gaussian map reconstruction (arXiv, 2025)

**Paper:** *ActiveGAMER: Active Gaussian Mapping through Efficient Rendering*
**Author:** Liyan Chen, Huangying Zhan, Kevin Chen, Xiangyu Xu, Qingan Yan, Changjiang Cai, Yi Xu
**Source:** arXiv:2501.06897, January 2025 | **CVPR 2025**

**Core contribution:**
- The first complete system of **Active Perception + 3D Gaussian Splatting**
- Validated in simulation and real environment (Franka robotic arm + UAV platform)
- Implemented **real-time NBV planning** (GPU parallel rendering acceleration)

**System Architecture:**

```
┌──────────────────────────────────────────────────────────┐
│                  ActiveGAMER Pipeline                   │
│                                                          │
│  Step 1: 初始建图（稀疏视角覆盖）                         │
│  → 3DGS 初始重建（有明显空洞）                           │
│                                                          │
│  Step 2: NBV 选择（主动感知循环）                        │
│  ┌────────────────────────────────────────────────────┐ │
│  │ 候选视角渲染（并行 ray casting through Gaussians）  │ │
│  │ → 渲染深度图 + 渲染 RGB + 渲染不确定性图             │ │
│  │ → 信息增益评估（基于深度不确定度）                   │ │
│  │ → 选择信息增益最大的下一视角                         │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
│  Step 3: 移动 + 精细建图                                  │
│  → UAV 飞行到新视角                                      │
│  → 增量插入新 Gaussians                                  │
│  → 自适应致密化（只加有信息的区域）                       │
│                                                          │
│  Loop: 返回 Step 2，直到覆盖率达到阈值                    │
└──────────────────────────────────────────────────────────┘
```

**Key technology:**

### 2.1 Information gain based on uncertainty

**Key Insight:** The Gaussian parameters of 3DGS inherently have **means and covariances** (Gaussian distribution), and the information gain of observations can be calculated directly from the parameter distribution.**Information gain calculation:**
$$
\Delta I \approx \sum_{p \in \text{pixels}} \sigma^2_{\text{rendered}}(p)
$$

That is: the sum of variances of rendered pixels = the amount of information that the perspective can provide.

- Large rendering variance → The map of this area is still rough and more observations are needed
- Small rendering variance → The map of this area is already very good, but the observation benefit is low

### 2.2 Efficient candidate perspective evaluation

The number of candidate viewpoints in traditional methods is small (dozens) because each one needs to be fully rendered.

**ActiveGAMER speedup:**
1. Use **splat-based ray casting** (without tracking all the details)
2. Batch and parallel evaluation of hundreds of candidate perspectives
3. Only perform complete rendering on top-K candidates
4. The overall NBV cycle is about **10Hz** (can be real-time!)

### 2.3 Adaptive densification

Not all new perspectives are worth adding Gaussians:
- **High information area**: depth discontinuity, large viewing angle changes → densification
- **Low information area**: overlapping area, sparse texture → skip

**This is also the closest to the direction of your existing blog! ** Your uav-nerf-gs-planning can quote this article directly.

---

## 3. SO-NeRF: NeRF NBV for proxy targets (arXiv, 2023)

**Paper:** *SO-NeRF: Active View Planning for NeRF using Surrogate Objectives*
**Author:** Keifer Lee, Shubham Gupta, Sunglyoung Kim, Bhargav Makwana, Chao Chen, Chen Feng
**Source:** arXiv:2312.XXXXX, December 2023

**Core contribution:**
- Proposed **Surrogate Objectives** to solve non-convexity in NBV optimization
- Avoids the problem of directly optimizing reconstruction quality (non-differentiable, heavy calculation)

**Note:** SO-NeRF was published on arXiv, and no clear top publication record has been found.

**Method:**

```
传统 NBV：
  目标：max 重建质量（需要完整重建才能评估）
  局限：不可微、慢、需要多次渲染

SO-NeRF：
  目标：max 代理目标（可微、快速）
  代理：渲染深度的不连续性 + 视角覆盖度
  核心：深度梯度 = 物体边界 = 需要更多信息的地方
```**Intuition:** The places with large gradients in the rendered depth map (depth mutations = object boundaries) are places that have not yet been modeled.

**Differences from ActiveGAMER:**
- SO-NeRF uses depth gradients as proxies (no need to modify NeRF itself)
- ActiveGAMER with Gaussian variance (requires GS probabilistic framework)
- The two can complement each other: SO-NeRF does candidate screening and ActiveGAMER does fine-tuning

---

## 4. AutoNeRF: Autonomous Data Collection (arXiv, 2024)

**Paper:** *AutoNeRF: Training Implicit Scene Representations with Autonomous Agents*
**Author:** Pierre Marza, Laetitia Matignon, Olivier Simonin, Dhruv Batra, Christian Wolf, Devendra Singh Chaplot
**Source:** arXiv, 2024

**Core contribution:**
- Let the **agent (robot) decide independently where to collect NeRF training data**
- Verified in Habitat-sim simulation environment
- Compared multiple active strategies: random / frontier-based / model-based

**Key Findings:**
- Simple frontier-based strategy is already much better than random
- Model prediction type (predicting the quality of new perspectives using NeRF) can be further improved
- **Active collection vs passive collection**: The final reconstruction quality is improved by 40%+

**Inspiration on UAV:**
- The UAV's aerial perspective makes the frontier (explored-unexplored boundary) larger than that of ground robots
- Aerial NBV needs to consider **vertical direction** (not just horizontal movement)
- On the top of the building and under the overhanging structure is the UAV's unique "frontier"

---

## 5. Active Perception using NeRF (arXiv, 2023)**Paper:** *Active Perception using Neural Radiance Fields*
**Author:** Siming He, Christopher D. Hsu, Dexter Ong, Yifei Simon Shao, Pratik Chaudhari
**Source:** arXiv:2310.09892, October 2023

**This is a basic paper on information theory that you can directly quote in your blog! **

**Core contribution:**
Derive from **first principles** what active sensing should maximize:

> **Maximize the mutual information of past observations to future observations**
> $$\max_a \quad I(Z_{past} \cup Z_{new}(a); Y)$$

Among them:
- $Z_{past}$ = existing sensor observations
- $Z_{new}(a)$ = new observation that will be obtained after executing action $a$
- $Y$ = complete state of the environment

**Three key components:**

```
1. Scene Representation（场景表示）
   → NeRF 捕获几何 + 外观 + 语义
   → 可以从任意视角渲染合成图像

2. Generative Model（生成模型）
   → NeRF 就是生成模型！给定 pose → 渲染 image
   → 给合成观测评估信息增益

3. Information-Driven Planner（信息驱动规划器）
   → 采样可行的机器人轨迹
   → 在每条轨迹的末端视角渲染
   → 选择渲染图像信息增益最大的轨迹
```

---

## 6. From object to scene: Scaling of NBV

### 6.1 Single-object NBV → Scene-level NBV

Early NBV work focused on complete reconstruction of single objects:
- The object is placed on the turntable and turned to a specific angle to take pictures
- Goal: Cover all perspectives and obtain a complete 3D model

**Your UAV work is scene-level:**
- Entire urban canyon/interior space
- You cannot do it one by one, you need an overall plan
- **Frontier-based exploration** becomes the main strategy

### 6.2 Frontier-Based Exploration + Information Gain

**Frontier** = The boundary between explored and unexplored areas.

```
经典 Frontier 探索：
  1. 从当前地图提取所有 frontier 点
  2. 选择最近的 frontier → 飞过去
  3. 扩大已知区域
  4. 重复

Frontier + Information Gain：
  1. 从当前地图提取所有 frontier 点
  2. 预测每个 frontier 的信息增益（用 NeRF/3DGS 渲染）
  3. 选择 info/max(distance) 最大的 frontier（权衡信息 + 能量）
  4. 飞过去
  5. 重复
```

**Trade-off Functional Design:**

$$
\text{score}(f) = \frac{\text{InformationGain}(f)}{\text{TravelCost}(f)} = \frac{I(f)}{\|p_{current} - f\|_2}
$$

This is actually the **"maximum information/distance ratio"** criterion in UAV exploration to ensure flight efficiency.

---

## 7. Specific applications in UAV scenarios### 7.1 Urban Canyon Exploration

**Scene features:**
- There are high-rise buildings on both sides, and the sky is open on the top
- The bottom is the street, the GNSS signal is poor
- The side is the building facade, with high information density

**NBV Strategy Advice:**

```
Phase 1: 建立初始地图
  → 沿建筑边缘飞行，捕获立面纹理
  → 初始重建完成约 30-40%

Phase 2: 填充立面细节
  → 选择立面渲染不确定度大的区域
  → 飞到近处做精细扫描

Phase 3: 顶部覆盖
  → 飞行到建筑顶面高度
  → 俯视捕获屋顶结构

Phase 4: 精细化
  → 重复，直到渲染不确定度全面低于阈值
```

### 7.2 Correspondence to your existing job

| What you wrote in your blog | Corresponding to NBV system components |
|------------------|-----------------|
| 3D Spatial Modeling (Octree/Occupancy Grid) | Accessibility Constraints + Collision Detection |
| NeRF/3DGS mapping | Actively aware Scene Representation |
| Semantic SLAM | Semantic-aware NBV (prioritize scanning of "important" objects) |
| Simulation data closed loop | Active sensing data enhancement |

---

## 8. Key technical details

### 8.1 Summary of uncertainty estimation methods

| Method | Calculation method | Applicable scenarios | Real-time |
|------|---------|---------|--------|
| **Monte Carlo Dropout** | Multiple forward propagation, variance as uncertainty | NeRF (requires network modification) | Slow |
| **Surrogate Gradient** | Render depth gradient as proxy | SO-NeRF | Fast |
| **Gaussian Variance** | GS's own covariance propagation | 3DGS (ActiveGAMER) | Medium |
| **Aleatoric + Epistemic** | Separate noise uncertainty and knowledge uncertainty | General | Medium |

### 8.2 Generation of candidate trajectories

NBV is not just about choosing a point, but choosing a **feasible trajectory**:
- UAV has maximum speed/acceleration constraints
- Kinetic feasibility needs to be considered (RRT*/BIT*/MPC)
- Usually generate candidate endpoints first, and then verify the feasibility of the trajectory

---

## 9. Challenges and open questions

### 9.1 Computational bottleneck

The main calculation cost of NBV:
- **Candidate Evaluation** (hundreds of candidates × rendering = bottleneck)
- **Information gain calculation** (requires multiple renderings)
- **NBV optimization loop** (typically requires 10-50 iterations)**Solution:**
- Fast screening with low-resolution rendering early on
- High-resolution accurate evaluation of only top-10 candidates
- GPU parallelization (candidate for parallel rendering)

### 9.2 Dynamic environment

Existing NBV methods mostly assume a static environment. But in the urban canyon:
- The car is moving
- Pedestrians coming and going
- Building may be under construction

**OPEN QUESTIONS:**
- How are dynamic objects included in information gain calculations?
- What should I do if the modeled area is blocked by dynamic objects?
- Tradeoffs of online incremental updates vs periodic full rebuilds?

### 9.3 Semantic-aware NBV

Most current NBV methods only consider geometric information gain. But:
- "This building is a museum, more important than a parking lot"
- "There are billboards on this facade, which has a higher information density than the blank wall."

**Solution:**
- Add **Semantic NeRF** to NeRF/3DGS
- Information gain = geometric gain × semantic weight
- Similar to what you wrote in uav-semantic-mapping.md!

---

## 10. Recommended research route

**Route A (fast results):**
1. Based on your uav-nerf-gs-planning article
2. Connect to ActiveGAMER’s information gain calculation module
3. Validate on your existing UAV simulation platform
4. Estimated workload: 2-3 months

**Route B (Systematic Study):**
1. Implement FIT-SLAM (FIM-based Active SLAM)
2. Replace map representation with your 3DGS system
3. Add semantic-aware weights
4. Verification on real UAV
5. Estimated workload: 6-12 months

**Route C (Frontier Exploration):**
1. Combine VLM (Direction 1) to do "Semantic NBV"
2. VLM evaluates the semantic importance of each frontier
3. Information gain = geometric gain + semantic gain
4. Estimated workload: 12+ months, but there is plenty of room for innovation

---

## 📚 References1. Chen et al. *ActiveGAMER: Active Gaussian Mapping through Efficient Rendering*. arXiv:2501.06897, January 2025.
2. Lee et al. *SO-NeRF: Active View Planning for NeRF using Surrogate Objectives*. arXiv:2312.XXXXX, December 2023.
3. He et al. *Active Perception using Neural Radiance Fields*. arXiv:2310.09892, October 2023.
4. Marza et al. *AutoNeRF: Training Implicit Scene Representations with Autonomous Agents*. arXiv, 2024.
5. Saravanan et al. *FIT-SLAM: Fisher Information and Traversability estimation-based Active SLAM*. arXiv:2401.09322, January 2024.
6. Zhan et al. *Active Human Pose Estimation via an Autonomous UAV Agent*. arXiv, 2024.
7. Chaplot et al. *Learning Visual Exploration for Long-Range Navigation*. NeurIPS, 2020.