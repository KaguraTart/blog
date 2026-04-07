---
title: "无人机路径规划冲突消解算法综述"
description: "深入解析多无人机系统中的冲突识别与消解算法，涵盖几何法、优化法、多机协同和学习式方法，从经典算法到前沿进展系统梳理"
pubDate: 2026-04-07
tags: ["无人机", "路径规划", "冲突消解", "多智能体", "算法综述"]
category: Tech
---

# 无人机路径规划冲突消解算法综述

> 随着无人机（UAV）从单机作业向集群协同演进，多架无人机在同一空域内执行任务时，路径冲突成为不可避免的核心问题。**冲突消解（Conflict Resolution）** 是指在保证飞行安全的前提下，通过调整各无人机的轨迹或决策，使冲突状态消失并继续完成任务。本文系统梳理冲突识别与消解的主流算法框架，从几何方法到深度强化学习，探讨各路技术的核心思想、优缺点与演进脉络。

---

## 1. 冲突的定义与分类

### 1.1 什么是路径冲突

在多无人机系统中，**冲突（Conflict）** 指两架或多架无人机在时空维度上同时占据相同空间位置（或小于安全隔离距离）的状态。形式化地：

$$
\exists \, i, j, \, i \neq j, \quad \| \mathbf{p}_i(t) - \mathbf{p}_j(t) \| < d_{safe}
$$

其中 $\mathbf{p}_i(t)$ 为第 $i$ 架无人机的位置，$d_{safe}$ 为安全隔离距离（通常取 5–50m，视任务场景而定）。

### 1.2 冲突分类

| 类型 | 描述 | 典型场景 |
|------|------|---------|
| **空间冲突** | 轨迹在空间上相交 | 交叉航线、对向飞行 |
| **时空冲突** | 轨迹在时间维度上重叠 | 先后进入同一空域 |
| **速度冲突** | 相对速度超过安全阈值 | 追及场景 |
| **高度冲突** | 垂直方向上的冲突 | 升降交汇 |
| **动态冲突** | 移动障碍物（其他飞行器）引发的冲突 | 空中会遇 |

### 1.3 冲突的度量指标

- **冲突时间（Time-to-Conflict）**：预测冲突发生前的剩余时间
- **冲突概率（Conflict Probability）**：考虑不确定性的冲突风险评估
- **最小间隔距离（Minimum Separation Distance）**：轨迹间的最接近距离
- **冲突解除时间（Resolution Time）**：消解动作生效所需时间

---

## 2. 冲突识别算法

冲突识别是冲突消解的前置步骤，其核心是**冲突预测**——在冲突实际发生之前判断其是否会发生。

### 2.1 几何预测法

最直观的方法是基于几何计算的空间检测：

```python
import numpy as np

def detect_conflict_2D(traj_i, traj_j, safe_radius=5.0):
    """
    检测两条轨迹是否发生空间冲突
    
    traj_i, traj_j: shape (N, 3) 的轨迹数组，每行为 (x, y, z)
    safe_radius: 安全隔离距离 (m)
    返回: (是否冲突, 最小间隔距离, 冲突时间点索引)
    """
    min_dist = float('inf')
    conflict_time = -1
    
    for t in range(len(traj_i)):
        dist = np.linalg.norm(traj_i[t] - traj_j[t])
        if dist < min_dist:
            min_dist = dist
            if dist < safe_radius:
                conflict_time = t
    
    is_conflict = min_dist < safe_radius
    return is_conflict, min_dist, conflict_time
```

### 2.2 速度障碍法（Velocity Obstacle）

**Velocity Obstacle（VO）** 是机器人领域最经典的冲突检测与预测方法，由 Fioretti & Fraichard（1999）引入 UAV 领域。

核心思想：在速度空间中构造"禁止区域"，若无人机当前速度矢量落在此区域内，则必定发生冲突。

$$
VO_{ij} = \{ \mathbf{v} \mid \lambda(\mathbf{p}_j - \mathbf{p}_i, \mathbf{v} - \mathbf{v}_j) \cap D_{ij} \neq \varnothing \}
$$

其中 $D_{ij}$ 是以 $\mathbf{p}_j - \mathbf{p}_i$ 为轴、半径为安全距离的圆柱体，$\lambda$ 是半射线。

```python
def velocity_obstacle(p_i, v_i, p_j, v_j, r_safe=5.0):
    """
    计算第 i 架 UAV 的速度障碍区域
    p_i, p_j: 位置向量
    v_i, v_j: 速度向量
    r_safe: 安全半径
    """
    rel_pos = p_j - p_i
    rel_vel = v_i - v_j
    dist = np.linalg.norm(rel_pos)
    
    if dist == 0:
        return None
    
    # 相对位置的夹角（安全圆柱的视角）
    theta = np.arcsin(r_safe / dist)
    
    # 障碍扇区的两条边向量
    dir_pos = rel_pos / dist
    perp_dir = np.array([-dir_pos[1], dir_pos[0]])
    
    # 两条边界速度向量
    v_left  = v_j + np.linalg.norm(v_j) * (np.cos(theta) * dir_pos + np.sin(theta) * perp_dir)
    v_right = v_j + np.linalg.norm(v_j) * (np.cos(theta) * dir_pos - np.sin(theta) * perp_dir)
    
    return v_left, v_right  # VO 的两条边界

def is_in_vo(v_i, v_left, v_right, v_j):
    """判断速度 v_i 是否落在 VO 区域内"""
    # 转换到相对坐标系
    rel_v = v_i - v_j
    rel_left  = v_left - v_j
    rel_right = v_right - v_j
    
    # 检查 rel_v 是否在 rel_left 和 rel_right 之间
    cross_left  = np.cross(rel_left,  rel_v)
    cross_right = np.cross(rel_right, rel_v)
    
    return np.sign(cross_left) == np.sign(cross_right)
```

### 2.3 不确定性感知冲突检测

在实际系统中，位置信息往往存在 GPS 误差、传感器噪声等不确定性。**概率冲突检测** 将概率分布引入冲突判断：

$$
P_{conflict} = \int\int \mathbb{1}(\| \mathbf{p}_i - \mathbf{p}_j \| < d_{safe}) \cdot f_i(\mathbf{p}_i) \cdot f_j(\mathbf{p}_j) \, d\mathbf{p}_i \, d\mathbf{p}_j
$$

其中 $f_i, f_j$ 为位置的概率密度函数（通常假设为高斯分布）。当 $P_{conflict} > P_{threshold}$ 时触发冲突告警。

常用方法包括：
- **Monte Carlo 采样**：对概率分布大量采样后统计冲突比例
- **线性化方法（Linear Validation Tool, LVT）**：对高斯分布假设下的概率冲突进行解析近似
- **随机可达集（Stochastic Reachable Set）**：基于随机控制理论的集合表示

---

## 3. 冲突消解算法

### 3.1 几何法

#### 3.1.1 速率障碍法（Rate Obstacle / VO 修正）

基于 VO 的消解策略：找到一条能避开 VO 区域的目标速度 $\mathbf{v}_{new}$：

```python
def vo_resolution(p_i, v_i, p_j, v_j, v_max=20.0, r_safe=5.0):
    """
    基于 Velocity Obstacle 的冲突消解
    返回满足速度约束且避开 VO 的新速度
    """
    vo = velocity_obstacle(p_i, v_i, p_j, v_j, r_safe)
    if vo is None:
        return v_i
    
    v_left, v_right = vo
    
    # 所有候选速度（速度空间中均匀采样）
    best_v = v_i
    min_dist_to_vo = float('inf')
    
    for speed in np.linspace(0, v_max, 20):
        for angle in np.linspace(0, 2*np.pi, 36):
            v_candidate = speed * np.array([np.cos(angle), np.sin(angle)])
            
            # 跳过落在 VO 内的速度
            if is_in_vo(v_candidate, v_left, v_right, v_j):
                continue
            
            # 选择最接近原始速度方向且最"远离"VO 的速度
            dist_to_original = np.linalg.norm(v_candidate - v_i)
            # 到 VO 边界的距离
            dist_to_vo = min(
                np.linalg.norm(v_candidate - v_left),
                np.linalg.norm(v_candidate - v_right)
            )
            
            # 优化目标：尽量接近原速度，同时远离 VO
            score = dist_to_vo - 0.5 * dist_to_original
            if dist_to_vo > min_dist_to_vo and dist_to_vo > 1.0:
                min_dist_to_vo = dist_to_vo
                best_v = v_candidate
    
    return best_v
```

#### 3.1.2 人工势场法（Artificial Potential Field）

将无人机视为在"势场"中运动的带电粒子：
- **目标点**产生吸引力
- **障碍物/其他无人机**产生排斥力

$$
\mathbf{F}_{total} = \mathbf{F}_{att} + \sum_j \mathbf{F}_{rep,j}
$$

其中：
$$
\mathbf{F}_{att} = k_{att} \cdot (\mathbf{p}_{goal} - \mathbf{p}_i)
$$
$$
\mathbf{F}_{rep,j} = k_{rep} \cdot \frac{\mathbf{p}_i - \mathbf{p}_j}{\| \mathbf{p}_i - \mathbf{p}_j \|^3} \cdot (\| \mathbf{p}_i - \mathbf{p}_j \| - d_{safe})
$$

**优点**：计算速度快，适合实时控制
**缺点**：易陷入局部极小值（两个无人机相互"推不开"时会发生震荡）

**改进方向**：
- 势场合形（Potential Field Shaping）：调整势场形状避免局部极小
- 多虚拟势场：引入虚拟障碍物引导路径绕过陷阱区域
- 混合方法：与 A* 或 RRT* 结合，用势场做局部微调

#### 3.1.3 Voronoi 图法

利用 Voronoi 图将空间划分为多个区域，每个无人机在其 Voronoi 单元内飞行，从而保证与其他无人机的距离永远大于其到 Voronoi 边界的距离：

1. 实时构建当前时刻的 Voronoi 图
2. 每架无人机在其 Voronoi 单元的 Farthest Point（最优点）附近选择航点
3. 沿航点方向运动，若检测到冲突则切换到新的 Voronoi 路径

```python
from scipy.spatial import Voronoi
import numpy as np

def voronoi_resolution(positions, v_i, p_i, v_max=20.0):
    """
    基于 Voronoi 图的多机冲突消解
    positions: 所有无人机位置 (N, 2)
    """
    vor = Voronoi(positions)
    
    # 找到当前无人机 i 的 Voronoi 单元
    region_idx = vor.point_region[np.where(vor.point_region == vor.point_region[0])[0][0]]
    # 简化的 Voronoi 路径选择：取 Voronoi 顶点的方向
    vertices = [vor.vertices[v] for v in vor.regions[region_idx] if v >= 0]
    
    if not vertices:
        return v_i
    
    # 选择最接近原始速度方向的安全顶点
    best_dir = v_i / np.linalg.norm(v_i)
    best_vertex = None
    max_projection = -float('inf')
    
    for v in vertices:
        direction = v - p_i
        if np.linalg.norm(direction) < 0.01:
            continue
        direction = direction / np.linalg.norm(direction)
        projection = np.dot(direction, best_dir)
        
        if projection > max_projection:
            max_projection = projection
            best_vertex = v
    
    if best_vertex is None:
        return v_i
    
    return np.clip(best_vertex - p_i, -v_max, v_max)
```

### 3.2 优化方法

#### 3.2.1 混合整数线性规划（MILP）

MILP 是将多无人机轨迹规划形式化为数学优化问题的经典框架，由 Schouwenaars 等人（2001）开创性地引入 UAV 领域。

**核心思想**：将连续轨迹用分段多项式或固定航点序列表示，冲突约束和安全约束用线性不等式表达，引入二元整数变量表示航段的切换逻辑：

$$
\min \quad \sum_i \sum_k \| \mathbf{v}_{i,k} - \mathbf{v}_{i,k-1} \|^2 + \lambda \sum_i \sum_k \| \mathbf{v}_{i,k} - \mathbf{v}_{i,k}^{pref} \|^2
$$

**约束条件**：
- 运动学约束：$\| \mathbf{v}_{i,k} \| \leq v_{max}$，$\| \mathbf{a}_{i,k} \| \leq a_{max}$
- 冲突避免约束：
  - 若 $\| \mathbf{p}_{i,k} - \mathbf{p}_{j,k} \| < d_{safe}$，则对应的二元变量 $\delta_{ijk} = 1$
  - 引入 OR 约束：$\sum_j \delta_{ijk} \leq 0$（强制所有 $\delta$ 必须为 0，即无冲突）
- 任务完成约束：$\| \mathbf{p}_{i,K} - \mathbf{p}_{goal,i} \| < \varepsilon$

```python
# 概念性 MILP 冲突约束（伪代码）
"""
minimize: sum_i sum_k (v_i,k - v_pref_i,k)^2

subject to:
    for each UAV i, segment k:
        p_i,k+1 = p_i,k + v_i,k * dt          # 运动学
        norm(v_i,k) <= v_max                   # 速度限幅
        norm(a_i,k) <= a_max                   # 加速度限幅
        
    for each UAV pair (i,j), segment k:
        norm(p_i,k - p_j,k)^2 >= d_safe^2 OR delta_ik = 0
        M * delta_ik >= norm(p_i,k - p_j,k)^2 - d_safe^2
        sum_j delta_jk <= 0                     # 所有 delta 必须为 0
"""
```

**优点**：全局最优解，保证硬约束满足
**缺点**：MILP 求解器（CPLEX、Gurobi）计算复杂度随无人机数量指数增长，难以实时求解超过 5–10 架无人机的场景

#### 3.2.2 动态窗口法（Dynamic Window Approach, DWA）

DWA 从机器人运动规划中借鉴而来，在速度空间 $(v, \omega)$ 中采样，评估每个候选速度的：
1. **轨迹朝向目标性**
2. **碰撞安全性**（通过模拟短时轨迹判断）
3. **速度可达性**

```python
def dwa_resolution(p_i, v_i, v_goal, obstacles,
                   v_max=3.0, v_min=0.0,
                   a_max=2.0, dt=0.1, predict_time=2.0,
                   safe_radius=1.5):
    """
    Dynamic Window Approach 用于 UAV 冲突消解
    """
    # 1. 构建动态窗口（当前可达速度集）
   Vw = []
    for v in np.arange(max(0, v_i[0] - a_max*dt), min(v_max, v_i[0] + a_max*dt), 0.1):
        for w in np.arange(v_i[1] - a_max*dt, v_i[1] + a_max*dt, 0.1):
            Vw.append((v, w))
    
    best_score = -float('inf')
    best_v = v_i
    
    for (v, w) in Vw:
        # 2. 预测轨迹
        traj = []
        p_pred = p_i.copy()
        v_pred = np.array([v, w])
        for t in np.arange(0, predict_time, dt):
            traj.append(p_pred.copy())
            p_pred = p_pred + v_pred * dt
        
        # 3. 碰撞检测
        collision = False
        for p_obs in obstacles:
            for p_t in traj:
                if np.linalg.norm(p_t - p_obs) < safe_radius:
                    collision = True
                    break
            if collision:
                break
        
        if collision:
            continue
        
        # 4. 评分函数
        score_heading = np.linalg.norm(p_pred - v_goal)  # 越小越好
        score_velocity = v  # 越大越好（偏好高速）
        score_clearance = min([np.linalg.norm(p_t - p_obs)
                               for p_obs in obstacles for p_t in traj])
        
        total_score = (
            2.0 * (1.0 / (score_heading + 1e-6)) +
            1.0 * score_velocity +
            0.5 * score_clearance
        )
        
        if total_score > best_score:
            best_score = total_score
            best_v = np.array([v, w])
    
    return best_v
```

#### 3.2.3 分布式模型预测控制（DMPC）

DMPC 是大规模多无人机集群的主流方法，每架无人机：
1. 基于本地信息和邻机通信构建局部预测模型
2. 在有限时域内求解局部优化问题
3. 仅执行第一步控制，然后滚动重优化

**核心一致性约束**：
$$
\sum_{j \in \mathcal{N}_i} a_{ij} (\mathbf{x}_i[k+k_p|k] - \mathbf{x}_j[k+k_p|k]) = 0, \quad \forall k_p \in \{1, \dots, N_p\}
$$

其中 $\mathcal{N}_i$ 为无人机 $i$ 的邻机集合，$a_{ij}$ 为邻接矩阵权重。

DMPC 的关键优势在于**可扩展性**：每架无人机只需与邻机通信，计算量不随全局无人机数量指数增长。

### 3.3 多机协同方法

#### 3.3.1 基于图论的一致性算法

将多无人机系统建模为图 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$：
- 节点 $v_i \in \mathcal{V}$ 代表无人机
- 边 $e_{ij} \in \mathcal{E}$ 代表通信链路

**共识协议（Consensus Protocol）**：
$$
\dot{\mathbf{x}}_i = \sum_{j \in \mathcal{N}_i} a_{ij} (\mathbf{x}_j - \mathbf{x}_i)
$$

应用于冲突消解时，将各无人机的**优先级**或**代价函数**作为状态变量，通过共识收敛后选择消解动作。

常用拓扑结构对比：
- **最近邻拓扑**：$\mathcal{O}(N)$ 通信量，但收敛慢
- **全连接拓扑**：收敛快，但 $\mathcal{O}(N^2)$ 通信量
- ** Metropolis-Hastings 加权**：平衡收敛速度与通信开销

#### 3.3.2 市场拍卖法（Market-Based）

仿生算法思想：将冲突区域视为"资源"，各无人机通过拍卖竞争该资源的使用权：
1. **投标（Bid）**：每架无人机计算自身任务的紧迫度和代价
2. **竞标（Auction）**：最高出价者获得通行权，其他无人机等待或绕行
3. **结算（Allocation）**：更新资源分配表，重复直到无冲突

```python
import heapq

def auction_based_resolution(uavs, conflict_zone, max_iterations=20):
    """
    基于拍卖的多机冲突消解
    uavs: List[UAV] - 无人机列表
    conflict_zone: 冲突区域中心及半径
    """
    allocation = {}  # zone_id -> winner_uav_id
    unallocated = list(uavs)
    
    for iteration in range(max_iterations):
        if not unallocated:
            break
        
        # 每次拍卖冲突区域使用权
        bids = []
        for uav in unallocated:
            urgency = uav.task_deadline - time.now()
            cost = uav.compute_detour_cost(conflict_zone)
            bid = urgency / (cost + 1e-6)
            bids.append((bid, uav))
        
        # 最高出价者胜出
        bids.sort(reverse=True)
        winner_bid, winner = bids[0]
        
        allocation[conflict_zone.id] = winner.id
        unallocated.remove(winner)
        
        # 对非胜出者计算绕行路径
        for uav in unallocated:
            uav.compute_alternative_path(conflict_zone)
    
    return allocation
```

#### 3.3.3 博弈论方法

将冲突消解建模为**非合作博弈（Non-cooperative Game）**：
- 每个无人机是**参与者（Player）**
- 各无人机的轨迹是**策略（Strategy）**
- 最小化自身冲突风险和飞行代价是**收益函数（Utility）**

**纳什均衡（Nash Equilibrium）** 是一组策略组合，在该组合中，没有任何一个参与者能通过单方面改变策略来获得更好的收益：

$$
\forall i, \quad \mathbf{s}_i^* \in \arg\min_{\mathbf{s}_i \in \mathcal{S}_i} 
\mathcal{J}_i(\mathbf{s}_i^*, \mathbf{s}_{-i}^*)
$$

**相关均衡（Correlated Equilibrium）** 比纳什均衡更易于分布式求解，在 UAV 集群中更具实用性。

### 3.4 学习式方法

#### 3.4.1 强化学习（RL）

近年来，**深度强化学习（DRL）** 在 UAV 集群冲突消解中取得了显著进展。典型框架：

- **状态空间 $\mathcal{S}$**：所有 UAV 的位置、速度、目标点、障碍物
- **动作空间 $\mathcal{A}$**：速度变化量 $(\Delta v_x, \Delta v_y, \Delta v_z)$ 或航向角变化
- **奖励函数 $\mathcal{R}$**：
  - 碰撞：$r_{collision} = -100$
  - 接近目标：$r_{progress} = +10 \cdot \Delta dist$
  - 保持安全距离：$r_{safety} = +5$（当 $\|p_i - p_j\| > d_{safe}$）
  - 能量消耗：$r_{energy} = -0.1 \cdot \|\Delta v\|^2$

**MADDPG（Multi-Agent DDPG）** 是最常用的多机 RL 框架之一：

```python
# MADDPG 核心思路（伪代码）
class MADDPGAgent:
    def __init__(self, state_dim, action_dim, lr_actor=1e-4, lr_critic=1e-3):
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim * n_agents, action_dim * n_agents)
        self.target_actor = copy_network(self.actor)
        self.target_critic = copy_network(self.critic)
        self.replay_buffer = ReplayBuffer(capacity=1e6)
    
    def select_action(self, state, noise=0.1):
        action = self.actor.forward(state)
        action += noise * np.random.randn(action.shape)
        return np.clip(action, -1, 1)
    
    def update(self, batch):
        # 从全局视角更新 Critic（这是 MADDPG 的关键创新）
        states, actions, rewards, next_states = batch
        
        # 目标网络更新
        target_actions = [self.target_actor.forward(ns) for ns in next_states]
        target_Q = self.target_critic.forward(
            torch.cat(states), torch.cat(target_actions))
        
        # 均值聚集（Mean Aggregation）：所有智能体的目标 Q 值取平均
        target_Q = sum(target_Q) / n_agents
        
        # 策略更新
        ...
```

#### 3.4.2 注意力机制（Attention）

**Graph Attention Network（GAT）** 被用于建模 UAV 之间的相对重要性关系：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        self.W = nn.Linear(in_features, out_features)
        self.a = nn.Linear(2 * out_features, 1)
        
        self.leaky_relu = nn.LeakyReLU(alpha)
    
    def forward(self, h, adj):
        """
        h: (N, D_in) - N 个节点的特征
        adj: (N, N) - 邻接矩阵
        """
        Wh = self.W(h)  # (N, D_out)
        N = Wh.size(0)
        
        # 计算注意力系数
        a_input = torch.cat([
            Wh.repeat(1, N).view(N, N, -1),
            Wh.repeat(N, 1).view(N, N, -1)
        ], dim=2)  # (N, N, 2*D_out)
        
        e = self.leaky_relu(self.a(a_input).squeeze(2))  # (N, N)
        
        # Mask 掉非邻接节点
        e = e.masked_fill(adj == 0, -1e9)
        attention = F.softmax(e, dim=1)  # (N, N)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # 特征聚集
        h_out = torch.matmul(attention, Wh)  # (N, D_out)
        return F.elu(h_out)
```

通过 GAT，无人机可以自适应地学习哪些邻机对自身决策影响最大，从而实现**软协调**——不必与所有无人机通信，只需关注注意力权重高的邻机。

#### 3.4.3 模仿学习（Imitation Learning）

利用专家轨迹（DMPC 或几何方法的求解结果）训练策略网络：

$$
\mathcal{L} = -\mathbb{E}_{(s,a) \sim d_{\pi^*}}[\log \pi_\theta(a \mid s)]
$$

**DAgger（Dataset Aggregation）** 方法可以循环迭代地收集专家标注数据，解决分布偏移问题。

---

## 4. 算法对比与选型指南

| 算法类别 | 实时性 | 可扩展性 | 最优性 | 对不确定性的处理 | 典型应用场景 |
|---------|--------|---------|--------|----------------|------------|
| **VO / 几何法** | ⭐⭐⭐⭐⭐ | ⭐⭐ | 局部最优 | ❌ | 2–5 架，追及冲突 |
| **势场法** | ⭐⭐⭐⭐⭐ | ⭐⭐ | 局部最优 | ❌ | 实时避障，动态障碍 |
| **Voronoi** | ⭐⭐⭐ | ⭐⭐⭐ | 局部最优 | ❌ | 稀疏集群路径规划 |
| **MILP** | ⭐ | ⭐⭐ | **全局最优** | ⚠️ 可扩展 | ≤10 架离线规划 |
| **DMPC** | ⭐⭐⭐ | ⭐⭐⭐⭐ | 次优 | ⚠️ 内置 | 10–50 架集群 |
| **图论/拍卖** | ⭐⭐⭐ | ⭐⭐⭐⭐ | 次优 | ❌ | 大规模集群任务分配 |
| **博弈论** | ⭐⭐ | ⭐⭐⭐ | 相关均衡 | ⚠️ 可扩展 | 竞争场景 |
| **DRL** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 策略最优 | ✅ 内置 | 50+ 架集群，端到端 |
| **GAT+RL** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 策略最优 | ✅ 内置 | 超大规模异构集群 |

**选型建议**：
- **5 架以下**：优先 VO 或 DMPC，计算快且保证解的质量
- **5–50 架**：DMPC + 图一致性协议，或 MADDPG
- **50 架以上**：GAT + Attention + RL，端到端策略是最现实的选择
- **有竞争关系的多方**：博弈论框架（纳什均衡/相关均衡）
- **不确定性显著**（GPS 误差、风扰动）：概率冲突检测 + robust MPC

---

## 5. 前沿进展与趋势

### 5.1 端到端强化学习

Google DeepMind 2023 年发表的 [SMARTS](https://github.com/hijkzzz/SMARTS) 平台和 [AlphaPilot](https://www.microsoft.com/en-us/research/project/alpha-pilot/) 项目推动了端到端 RL 在 UAV 集群中的应用，展示了从感知到决策的单一策略网络可以直接处理原始传感器数据。

### 5.2 分布式学习（Federated Learning）

在保护各无人机数据隐私的前提下，通过联邦学习聚合多机经验：
1. 各无人机本地训练策略
2. 只上传梯度而非原始数据到中央服务器
3. 聚合更新后下发新策略

解决了 UAV 集群中数据分散、标签获取困难的问题。

### 5.3 不确定性Robust MPC

近年来，** tube-based MPC** 和 **scenario-based MPC** 将不确定性建模为有界扰动或概率场景，在优化问题中显式约束鲁棒性：

$$
\forall \omega \in \mathbb{W}: \quad \mathbf{x}[k+1] = A\mathbf{x}[k] + B\mathbf{u}[k] + E\omega[k]
$$

通过预计算"不变集"来保证在最坏情况下仍然满足安全约束。

### 5.4 多目标冲突消解

实际任务中，冲突消解还需要同时考虑：
- **任务完成率**：不能一味绕行导致任务超时
- **能量消耗**：电量有限的 UAV 需要最小化额外飞行距离
- **通信延迟**：分布式系统中信息滞后可能导致误判
- **公平性**：不能让某几架 UAV 总是退让（饥饿问题）

**Pareto 最优前沿**搜索是解决多目标冲突消解的核心工具。

---

## 6. 总结

无人机路径规划冲突消解是一个横跨**几何计算、优化理论、分布式系统与机器学习**的交叉问题。从最早的几何速率障碍法，到 MILP 全局优化，再到分布式 MPC 和深度强化学习，算法演进的核心驱动力始终是：

> **如何在更短的时间内、为更多的无人机、在更强的不确定性下，找到更安全的轨迹。**

未来的趋势将是**混合架构**：用学习式方法做快速局部决策，用优化方法做全局轨迹验证，用通信协议保证多机协同的一致性。三者结合，才能真正实现安全、高效、可扩展的无人机集群自主飞行。

---

**参考文献**（按时间排序）：

1. Schouwenaars, T., et al. (2001). *Mixed integer programming for multi-vehicle path planning.* European Control Conference.
2. Fioretti, F., & Fraichard, T. (2001). *Multi-robot formation for cooperative surveillance.* ICRA.
3. van den Berg, J., Lin, M., & Manocha, D. (2008). *Reciprocal velocity obstacles for real-time multi-agent navigation.* ICRA.
4. Richards, A., & How, J. P. (2002). *Aircraft trajectory planning with collision avoidance using mixed integer linear programming.* AIAA GNC.
5. Alonso-Ayuso, A., et al. (2014). *An optimization-based negotiation approach for air traffic deconflict resolution.* Transportation Research.
6. Foerster, J., et al. (2018). *Counterfactual multi-agent policy gradients.* AAAI.
7. Zhou, M., et al. (2019). *A survey on path planning for UAVs in cluttered environments.* IEEE Transactions on Intelligent Transportation Systems.
8. Liu, X., et al. (2021). *Multi-UAV coordination and control: A survey.* IEEE Transactions on Vehicular Technology.
9. Li, Z., et al. (2023). *Graph attention network for multi-robot coordination.* ICRA.
10. Wu, J., et al. (2024). *Distributed model predictive control for UAV swarm conflict resolution: A review.* Aerospace Science and Technology.
