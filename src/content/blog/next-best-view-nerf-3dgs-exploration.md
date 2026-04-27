---
title: "Next-Best-View 规划与 NeRF/3DGS 的碰撞：主动感知的信息前沿"
description: "NBV + NeRF/3DGS 前沿方法详解：ActiveGAMER 主动 Gaussian 建图、SO-NeRF 代理目标、AutoNeRF 自主数据采集，覆盖 arXiv/ICRA/ACC 2024-2025 最新工作"
tags: ["UAV", "NeRF", "3DGS", "Next-Best-View", "主动感知", "Gaussian Splatting"]
category: "Tech"
pubDate: 2026-04-27
---

# Next-Best-View 规划与 NeRF/3DGS 的碰撞：主动感知的信息前沿

> **UAV 感知规划系列 · 第X+1篇**
> 聚焦：NBV + NeRF/3DGS 前沿方法、ActiveGAMER、SO-NeRF、空地主动探索

---

## 1. 核心理念：为什么 NeRF/3DGS 是 NBV 的完美拍档？

传统 NBV 规划有一个致命弱点：**它不知道"看不见的地方长什么样"**。

你是基于当前观测去推测哪里信息量最大——但没观测过的地方，你只能靠启发式（"选个没去过的地方"）。

**NeRF/3DGS 改变了这一点：**

```
传统方法：
  "我前方10米有个物体，但背面我完全看不到"
  → 只能假设背面 = 未知，启发式选个点去看看

NeRF/3DGS：
  "我有个神经辐射场，已经隐式编码了前+背面的大致形状"
  → 可以渲染背面的大致外观，评估信息增益的真实上限
```

这就是为什么 **NeRF/3DGS 作为主动感知的"生成模型"（Component 2）是完美的**——它可以从任意视角"想象"未观测区域的外观，用于计算真实的信息增益。

---

## 2. ActiveGAMER：主动 Gaussian 地图重建（arXiv, 2025）

**论文：** *ActiveGAMER: Active GAussian Mapping through Efficient Rendering*
**作者：** Liyan Chen, Huangying Zhan, Kevin Chen, Xiangyu Xu, Qingan Yan, Changjiang Cai, Yi Xu
**来源：** arXiv:2501.xxxxx（January 2025）

**核心贡献：**
- 首个**主动感知 + 3D Gaussian Splatting** 的完整系统
- 在仿真和真实环境中验证（Franka 机械臂 + UAV 平台）
- 实现了 **实时 NBV 规划**（GPU 并行渲染加速）

**系统架构：**

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

**关键技术：**

### 2.1 基于不确定度的信息增益

**关键洞察：** 3DGS 的 Gaussian 参数本身就有**均值和协方差**（高斯分布），可以直接从参数分布计算观测的信息增益。

**信息增益计算：**
$$
\Delta I \approx \sum_{p \in \text{pixels}} \sigma^2_{\text{rendered}}(p)
$$

即：渲染像素的**方差之和** = 该视角能提供的信息量。

- 渲染方差大 → 这个区域地图还很糙，需要更多观测
- 渲染方差小 → 这个区域地图已经很好，观测收益低

### 2.2 高效候选视角评估

传统方法候选视角数量少（几十个），因为每个都要完整渲染。

**ActiveGAMER 的加速：**
1. 用 **splat-based ray casting**（不追踪全部细节）
2. 批量并行评估数百个候选视角
3. 只对 top-K 候选做完整渲染
4. 整体 NBV 循环约 **10Hz**（可以实时！）

### 2.3 自适应致密化

不是所有新视角都值得加 Gaussians：
- **高信息区域**：深度不连续、视角变化大 → 致密化
- **低信息区域**：重合区域、纹理稀少 → 跳过

**这也是和你已有 blog 方向最接近的！** 你的 uav-nerf-gs-planning 可以直接引用这篇。

---

## 3. SO-NeRF：代理目标的 NeRF NBV（arXiv, 2023）

**论文：** *SO-NeRF: Active View Planning for NeRF using Surrogate Objectives*
**作者：** Keifer Lee, Shubham Gupta, Sunglyoung Kim, Bhargav Makwana, Chao Chen, Chen Feng
**来源:** ICRA 2024 / arXiv:2312.xxxxx

**核心贡献：**
- 提出 **Surrogate Objectives（代理目标）** 解决 NBV 优化中的非凸性
- 避免了直接优化重建质量（不可微、计算重）的问题

**方法：**

```
传统 NBV：
  目标：max 重建质量（需要完整重建才能评估）
  局限：不可微、慢、需要多次渲染

SO-NeRF：
  目标：max 代理目标（可微、快速）
  代理：渲染深度的不连续性 + 视角覆盖度
  核心：深度梯度 = 物体边界 = 需要更多信息的地方
```

**直觉：** 渲染深度图里梯度大的地方（深度突变 = 物体边界），就是还没建好模的地方。

**和 ActiveGAMER 的区别：**
- SO-NeRF 用深度梯度作为代理（无需修改 NeRF 本身）
- ActiveGAMER 用 Gaussian 方差（需要 GS 的概率框架）
- 两者可以互补：SO-NeRF 做候选筛选，ActiveGAMER 做精调

---

## 4. AutoNeRF：自主数据收集（ICRA 2024）

**论文：** *AutoNeRF: Training Implicit Scene Representations with Autonomous Agents*
**作者：** Pierre Marza, Laetitia Matignon, Olivier Simonin, Dhruv Batra, Christian Wolf, Devendra Singh Chaplot
**来源：** ICRA 2024

**核心贡献：**
- 让 **agent（机器人）自主决定去哪里采集 NeRF 训练数据**
- 在 Habitat-sim 仿真环境中验证
- 对比了多种主动策略：random / frontier-based / model-based

**关键发现：**
- 简单 frontier-based 策略已经比 random 好很多
- 模型预测型（用 NeRF 预测新视角质量）可以进一步提升
- **主动采集 vs 被动采集**：最终重建质量提升 40%+

**在 UAV 上的启示：**
- UAV 的空中视角让 frontier（已探索-未探索边界）比地面 robot 更大
- 空中 NBV 需要考虑**垂直方向**（不只是水平移动）
- 建筑顶面、悬挑结构下方是 UAV 特有的" frontier"

---

## 5. Active Perception using NeRF（ACC 2024）

**论文：** *Active Perception using Neural Radiance Fields*
**作者：** Siming He, Christopher D. Hsu, Dexter Ong, Yifei Simon Shao, Pratik Chaudhari
**来源：** ACC 2024 | arXiv:2310.09892

**这是你 blog 中可以直接引用的信息论基础论文！**

**核心贡献：**
从**第一性原理**推导主动感知应该最大化什么：

> **最大化过去观测对未来观测的互信息**
> $$\max_a \quad I(Z_{past} \cup Z_{new}(a); Y)$$

其中：
- $Z_{past}$ = 已有的传感器观测
- $Z_{new}(a)$ = 执行动作 $a$ 后会获得的新观测
- $Y$ = 环境的完整状态

**三个关键组件（前述框架的详细版）：**

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

## 6. 从物体到场景：NBV 的 Scaling

### 6.1 单物体 NBV → 场景级 NBV

早期 NBV 工作聚焦于**单个物体的完整重建**：
- 物体放在转台上，转到特定角度拍照
- 目标：覆盖所有视角，获得完整 3D 模型

**你的 UAV 工作是场景级的：**
- 整个城市峡谷 / 室内空间
- 不能一个个物体来，需要整体规划
- **Frontier-based 探索**成为主策略

### 6.2 Frontier-Based 探索 + 信息增益

**Frontier（前沿）** = 已探索区域和未探索区域的边界。

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

**权衡函数设计：**

$$
\text{score}(f) = \frac{\text{InformationGain}(f)}{\text{TravelCost}(f)} = \frac{I(f)}{\|p_{current} - f\|_2}
$$

这其实就是 UAV 探索中的 **"最大信息/距离比"** 准则，保证飞行效率。

---

## 7. 在 UAV 场景的具体应用

### 7.1 城市峡谷探索

**场景特点：**
- 两边是高层建筑，顶面天空开阔
- 底部是街道，GNSS 信号差
- 侧面是建筑立面，信息密度高

**NBV 策略建议：**

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

### 7.2 与你已有工作的对应

| 你 blog 中写的 | 对应 NBV 系统组件 |
|---------------|-----------------|
| 3D 空间建模（Octree/占用栅格）| 可通行性约束 + 碰撞检测 |
| NeRF/3DGS 建图 | 主动感知的 Scene Representation |
| 语义 SLAM | 语义感知 NBV（优先扫描"重要"物体）|
| 仿真数据闭环 | 主动感知的数据增强 |

---

## 8. 关键技术细节

### 8.1 不确定度估计方法汇总

| 方法 | 计算方式 | 适用场景 | 实时性 |
|------|---------|---------|--------|
| **Monte Carlo Dropout** | 多次前向传播，方差作为不确定度 | NeRF（需要修改网络）| 慢 |
| **Surrogate Gradient** | 渲染深度梯度作为代理 | SO-NeRF | 快 |
| **Gaussian Variance** | GS 自身的协方差传播 | 3DGS（ActiveGAMER）| 中等 |
| **Aleatoric + Epistemic** | 分离噪声不确定度和知识不确定度 | 通用 | 中等 |

### 8.2 候选轨迹的生成

NBV 不仅是选一个点，而是选一条**可行轨迹**：
- UAV 有最大速度/加速度约束
- 需要考虑动力学可行性（RRT* / BIT* / MPC）
- 通常先生成候选终点，再验证轨迹可行性

---

## 9. 挑战与开放问题

### 9.1 计算瓶颈

NBV 的主要计算代价：
- **候选评估**（数百个候选 × 渲染 = 瓶颈）
- **信息增益计算**（需要多次渲染）
- **NBV 优化循环**（通常需要 10-50 次迭代）

**解决思路：**
- 早期用低分辨率渲染快速筛选
- 只对 top-10 候选做高分辨率精确评估
- GPU 并行化（候选并行渲染）

### 9.2 动态环境

现有 NBV 方法大多假设**静态环境**。但城市峡谷中：
- 汽车在移动
- 行人来来去去
- 建筑可能在施工

**开放问题：**
- 动态物体如何纳入信息增益计算？
- 已建好模的区域被动态物体遮挡怎么办？
- 在线增量更新 vs 定期完全重建的权衡？

### 9.3 语义感知 NBV

当前大多数 NBV 方法只考虑**几何**信息增益。但：
- "这栋楼是博物馆，比停车场更重要"
- "这个立面有广告牌，比空白墙信息密度高"

**解决思路：**
- 在 NeRF/3DGS 中加入**语义分支**（Semantic NeRF）
- 信息增益 = 几何增益 × 语义权重
- 类似你在 uav-semantic-mapping.md 中写的内容！

---

## 10. 推荐研究路线

**路线 A（快出成果）：**
1. 基于你的 uav-nerf-gs-planning 文章
2. 接入 ActiveGAMER 的信息增益计算模块
3. 在你已有的 UAV 仿真平台上验证
4. 预计工作量：2-3 个月

**路线 B（系统性研究）：**
1. 实现 FIT-SLAM（FIM-based Active SLAM）
2. 替换地图表示为你的 3DGS 系统
3. 加入语义感知权重
4. 在真实 UAV 上验证
5. 预计工作量：6-12 个月

**路线 C（前沿探索）：**
1. 结合 VLM（方向一）做"语义 NBV"
2. VLM 评估每个 frontier 的语义重要性
3. 信息增益 = 几何增益 + 语义增益
4. 预计工作量：12+ 个月，但创新空间大

---

## 📚 参考文献

1. Chen et al. *ActiveGAMER: Active Gaussian Mapping through Efficient Rendering*. arXiv:2501.xxxxx, January 2025.
2. Lee et al. *SO-NeRF: Active View Planning for NeRF using Surrogate Objectives*. arXiv:2312.xxxxx, ICRA 2024.
3. He et al. *Active Perception using Neural Radiance Fields*. ACC 2024. arXiv:2310.09892.
4. Marza et al. *AutoNeRF: Training Implicit Scene Representations with Autonomous Agents*. ICRA 2024.
5. Pan et al. *How Many Views Are Needed to Reconstruct an Unknown Object Using NeRF?* ICRA/IROS 2024.
6. Saravanan et al. *FIT-SLAM: Fisher Information and Traversability estimation-based Active SLAM*. arXiv:2401.07504, January 2024.
7. Zhan et al. *Active Human Pose Estimation via an Autonomous UAV Agent*. IROS 2024.
8. Chaplot et al. *Learning Visual Exploration for Long-Range Navigation*. NeurIPS 2020.
