---
title: "A review of conflict resolution algorithms for UAV path planning"
description: "In-depth analysis of conflict identification and resolution algorithms in multi-UAV systems, covering geometric methods, optimization methods, multi-machine collaboration and learning methods, from classic algorithms to cutting-edge progress system review"
pubDate: 2026-04-07T11:12:59+08:00
tags: ["drone", "path planning", "conflict resolution", "multi-agent", "Algorithm overview"]
category: Tech
sourceHash: "bdea72e467b5ee1ca0825e4536706f6e89e09f1b"
---

# Overview of UAV path planning conflict resolution algorithm

> As unmanned aerial vehicles (UAVs) evolve from single-machine operation to cluster collaboration, path conflicts have become an inevitable core issue when multiple UAVs perform tasks in the same airspace. **Conflict Resolution** refers to adjusting the trajectory or decision-making of each drone to eliminate the conflict state and continue to complete the mission while ensuring flight safety. This article systematically sorts out the mainstream algorithm frameworks for conflict identification and resolution, from geometric methods to deep reinforcement learning, and explores the core ideas, advantages, disadvantages, and evolution of each technology.

---

## 1. Definition and classification of conflicts

### 1.1 What is a path conflict?

In a multi-UAV system, **Conflict** refers to a state in which two or more UAVs occupy the same spatial position (or less than a safe isolation distance) in the space-time dimension at the same time. Formally:

$$
\exists \, i, j, \, i \neq j, \quad \| \mathbf{p}_i(t) - \mathbf{p}_j(t) \| < d_{safe}
$$

Among them, $\mathbf{p}_i(t)$ is the position of the $i$-th drone, and $d_{safe}$ is the safe isolation distance (usually 5–50m, depending on the mission scenario).

### 1.2 Conflict classification

| Type | Description | Typical Scenario |
|------|------|---------|
| **Space Conflict** | Trajectories intersect in space | Crossing routes, opposing flights |
| **Time and Space Conflict** | Trajectories overlap in the time dimension | Enter the same airspace one after another |
| **Speed Conflict** | Relative speed exceeds safety threshold | Catch-up scenario |
| **High Conflict** | Conflict in the vertical direction | Lifting intersection |
| **Dynamic Conflicts** | Conflicts caused by moving obstacles (other aircraft) | Aerial encounters |

### 1.3 Metrics of conflict

- **Time-to-Conflict**: Predict the remaining time before conflict occurs
- **Conflict Probability**: Conflict risk assessment taking into account uncertainty
- **Minimum Separation Distance**: The closest distance between trajectories
- **Resolution Time**: the time required for the resolution action to take effect

---

## 2. Conflict identification algorithmConflict identification is a preliminary step to conflict resolution, and its core is **conflict prediction** - determining whether a conflict will occur before it actually occurs.

### 2.1 Geometric prediction method

The most intuitive method is spatial detection based on geometric calculations:

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

### 2.2 Velocity Obstacle

**Velocity Obstacle (VO)** is the most classic conflict detection and prediction method in the robotics field. It was introduced into the UAV field by Fioretti & Fraichard (1999).

Core idea: Construct a "forbidden area" in the speed space. If the current speed vector of the drone falls within this area, a conflict will definitely occur.

$$
VO_{ij} = \{ \mathbf{v} \mid \lambda(\mathbf{p}_j - \mathbf{p}_i, \mathbf{v} - \mathbf{v}_j) \cap D_{ij} \neq \varnothing \}
$$

Where $D_{ij}$ is a cylinder with $\mathbf{p}_j - \mathbf{p}_i$ as the axis and the radius as the safe distance, and $\lambda$ is the half-ray.

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

### 2.3 Uncertainty-aware conflict detection

In actual systems, location information often contains uncertainties such as GPS errors and sensor noise. **Probabilistic conflict detection** Introduces probability distribution into conflict judgment:

$$
P_{conflict} = \int\int \mathbb{1}(\| \mathbf{p}_i - \mathbf{p}_j \| < d_{safe}) \cdot f_i(\mathbf{p}_i) \cdot f_j(\mathbf{p}_j) \, d\mathbf{p}_i \, d\mathbf{p}_j
$$

where $f_i, f_j$ is the probability density function of the location (usually assumed to be a Gaussian distribution). A conflict alarm is triggered when $P_{conflict} > P_{threshold}$.Common methods include:
- **Monte Carlo Sampling**: Statistics of conflict ratios after sampling a large number of probability distributions
- **Linear Validation Tool (LVT)**: Analytical approximation of probability conflicts under the assumption of Gaussian distribution
- **Stochastic Reachable Set**: Set representation based on stochastic control theory

---

## 3. Conflict resolution algorithm

### 3.1 Geometric method

#### 3.1.1 Rate Obstacle method (Rate Obstacle / VO correction)

VO-based elimination strategy: find a target speed $\mathbf{v}_{new}$ that can avoid the VO area:

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

#### 3.1.2 Artificial Potential Field method (Artificial Potential Field)

Think of drones as charged particles moving in a "potential field":
- **Target Point** generates attraction
- **obstacles/other drones** generate repulsive force

$$
\mathbf{F}_{total} = \mathbf{F}_{att} + \sum_j \mathbf{F}_{rep,j}
$$

Among them:
$$
\mathbf{F}_{att} = k_{att} \cdot (\mathbf{p}_{goal} - \mathbf{p}_i)
$$
$$
\mathbf{F}_{rep,j} = k_{rep} \cdot \frac{\mathbf{p}_i - \mathbf{p}_j}{\| \mathbf{p}_i - \mathbf{p}_j \|^3} \cdot (\| \mathbf{p}_i - \mathbf{p}_j \| - d_{safe})
$$

**Advantages**: Fast calculation, suitable for real-time control
**Disadvantages**: Easy to fall into local minima (two drones will oscillate when they "cannot push" each other)

**Improvement directions**:
- Potential Field Shaping: Adjust the shape of the potential field to avoid local minima
- Multiple virtual potential fields: introduce virtual obstacles to guide paths around trap areas
- Hybrid method: combined with A* or RRT*, using potential field for local fine-tuning

#### 3.1.3 Voronoi diagram methodThe Voronoi diagram is used to divide the space into multiple areas, and each drone flies within its Voronoi unit, thereby ensuring that the distance to other drones is always greater than its distance to the Voronoi boundary:

1. Construct the Voronoi diagram of the current moment in real time
2. Each drone selects a waypoint near the Farthest Point (optimal point) of its Voronoi unit
3. Move in the direction of the waypoint, and switch to the new Voronoi path if a conflict is detected.

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

### 3.2 Optimization method

#### 3.2.1 Mixed Integer Linear Programming (MILP)

MILP is a classic framework that formalizes multi-UAV trajectory planning as a mathematical optimization problem and was pioneered in the UAV field by Schouwenaars et al. (2001).

**Core idea**: The continuous trajectory is represented by piecewise polynomials or fixed waypoint sequences, conflict constraints and safety constraints are expressed by linear inequalities, and binary integer variables are introduced to represent the switching logic of the flight segments:

$$
\min \quad \sum_i \sum_k \| \mathbf{v}_{i,k} - \mathbf{v}_{i,k-1} \|^2 + \lambda \sum_i \sum_k \| \mathbf{v}_{i,k} - \mathbf{v}_{i,k}^{pref} \|^2
$$

**Constraints**:
- Kinematic constraints: $\| \mathbf{v}_{i,k} \| \leq v_{max}$, $\| \mathbf{a}_{i,k} \| \leq a_{max}$
- Conflict avoidance constraints:
  - If $\| \mathbf{p}_{i,k} - \mathbf{p}_{j,k} \| < d_{safe}$, then the corresponding binary variable $\delta_{ijk} = 1$
  - Introduce OR constraint: $\sum_j \delta_{ijk} \leq 0$ (forces all $\delta$ to be 0, that is, no conflict)
- Task completion constraints: $\| \mathbf{p}_{i,K} - \mathbf{p}_{goal,i} \| < \varepsilon$

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
```**Advantages**: Global optimal solution, ensuring that hard constraints are satisfied
**Disadvantages**: The computational complexity of MILP solvers (CPLEX, Gurobi) increases exponentially with the number of drones, making it difficult to solve scenarios with more than 5–10 drones in real time

#### 3.2.2 Dynamic Window Approach (DWA)

DWA, borrowed from robot motion planning, samples the velocity space $(v, \omega)$ and evaluates each candidate velocity:
1. **Trajectory towards target**
2. **Collision safety** (judged by simulating short-term trajectories)
3. **Speed Reachability**

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

#### 3.2.3 Distributed Model Predictive Control (DMPC)

DMPC is a mainstream method for large-scale multi-UAV swarms, where each UAV:
1. Build a local prediction model based on local information and neighbor communication
2. Solve local optimization problems in finite time domain
3. Only perform the first step of control, and then perform rolling re-optimization

**Core Consistency Constraints**:
$$
\sum_{j \in \mathcal{N}_i} a_{ij} (\mathbf{x}_i[k+k_p|k] - \mathbf{x}_j[k+k_p|k]) = 0, \quad \forall k_p \in \{1, \dots, N_p\}
$$

Where $\mathcal{N}_i$ is the set of neighbors of UAV $i$, and $a_{ij}$ is the adjacency matrix weight.

The key advantage of DMPC is **scalability**: each drone only needs to communicate with its neighbors, and the amount of calculation does not increase exponentially with the number of global drones.

### 3.3 Multi-machine collaboration method

#### 3.3.1 Consistency algorithm based on graph theory

Model the multi-UAV system as the graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:
- Node $v_i \in \mathcal{V}$ represents the drone
- Edge $e_{ij} \in \mathcal{E}$ represents the communication link

**Consensus Protocol**:
$$
\dot{\mathbf{x}}_i = \sum_{j \in \mathcal{N}_i} a_{ij} (\mathbf{x}_j - \mathbf{x}_i)
$$When applied to conflict resolution, the **priority** or **cost function** of each drone is used as a state variable, and the resolution action is selected after consensus convergence.

Comparison of common topologies:
- **Nearest neighbor topology**: $\mathcal{O}(N)$ traffic, but slow convergence
- **Fully connected topology**: fast convergence, but $\mathcal{O}(N^2)$ traffic
- **Metropolis-Hastings weighting**: balances convergence speed and communication overhead

#### 3.3.2 Market-Based

Bionic algorithm idea: Treat the conflict area as a "resource", and each drone competes for the right to use the resource through auction:
1. **Bid**: Each drone calculates the urgency and cost of its own mission
2. **Auction**: The highest bidder gets the right of way, other drones wait or go around
3. **Settlement (Allocation)**: Update the resource allocation table, repeat until there is no conflict

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

#### 3.3.3 Game theory method

Model conflict resolution as a **Non-cooperative Game**:
- Each drone is a **Player**
- The trajectory of each drone is **Strategy**
- Minimizing one's own conflict risk and flight cost is **Benefit Function (Utility)**

**Nash Equilibrium** is a set of strategy combinations in which no player can obtain better returns by unilaterally changing his strategy:

$$
\forall i, \quad \mathbf{s}_i^* \in \arg\min_{\mathbf{s}_i \in \mathcal{S}_i}
\mathcal{J}_i(\mathbf{s}_i^*, \mathbf{s}_{-i}^*)
$$

**Correlated Equilibrium** is easier to solve distributedly than Nash equilibrium and is more practical in UAV clusters.

### 3.4 Learning method

#### 3.4.1 Reinforcement Learning (RL)

In recent years, **Deep Reinforcement Learning (DRL)** has made significant progress in UAV cluster conflict resolution. Typical framework:- **State space $\mathcal{S}$**: positions, velocities, target points, obstacles of all UAVs
- **Action space $\mathcal{A}$**: Speed change $(\Delta v_x, \Delta v_y, \Delta v_z)$ or heading angle change
- **Reward function $\mathcal{R}$**:
  - Collision: $r_{collision} = -100$
  - Close to target: $r_{progress} = +10 \cdot \Delta dist$
  - Maintain a safe distance: $r_{safety} = +5$ (when $\|p_i - p_j\| > d_{safe}$)
  - Energy consumption: $r_{energy} = -0.1 \cdot \|\Delta v\|^2$

**MADDPG (Multi-Agent DDPG)** is one of the most commonly used multi-machine RL frameworks:

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

#### 3.4.2 Attention mechanism (Attention)

**Graph Attention Network (GAT)** is used to model the relative importance relationship between UAVs:

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

Through GAT, drones can adaptively learn which neighbor aircraft have the greatest impact on their own decisions, thereby achieving **soft coordination**—it does not have to communicate with all drones, but only needs to pay attention to neighbor aircraft with high attention weights.

#### 3.4.3 Imitation Learning

Train the policy network using expert trajectories (solutions from DMPC or geometric methods):

$$
\mathcal{L} = -\mathbb{E}_{(s,a) \sim d_{\pi^*}}[\log \pi_\theta(a \mid s)]
$$

The **DAgger (Dataset Aggregation)** method can iteratively collect expert annotated data to solve the distribution shift problem.

---

## 4. Algorithm comparison and selection guide| Algorithm category | Real-time | Scalability | Optimality | Handling of uncertainty | Typical application scenarios |
|---------|--------|---------|--------|----------------|---------------------|
| **VO / Geometry** | ⭐⭐⭐⭐⭐ | ⭐⭐ | Local Optimal | ❌ | 2–5 frames, catch-up conflicts |
| **Potential field method** | ⭐⭐⭐⭐⭐ | ⭐⭐ | Local optimum | ❌ | Real-time obstacle avoidance, dynamic obstacles |
| **Voronoi** | ⭐⭐⭐ | ⭐⭐⭐ | Local Optimal | ❌ | Sparse Cluster Path Planning |
| **MILP** | ⭐ | ⭐⭐ | **Global Optimum** | ⚠️ Scalable | ≤10 racks offline planning |
| **DMPC** | ⭐⭐⭐ | ⭐⭐⭐⭐ | Sub-optimal | ⚠️ Built-in | 10–50 rack cluster |
| **Graph theory/auction** | ⭐⭐⭐ | ⭐⭐⭐⭐ | Sub-optimal | ❌ | Large-scale cluster task allocation |
| **Game Theory** | ⭐⭐ | ⭐⭐⭐ | Relevant Equilibrium | ⚠️ Scalable | Competitive Scenarios |
| **DRL** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Optimal strategy | ✅ Built-in | 50+ clusters, end-to-end |
| **GAT+RL** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Optimal strategy | ✅ Built-in | Ultra-large-scale heterogeneous cluster |

**Selection suggestions**:
- **5 or less**: Prioritize VO or DMPC, the calculation is fast and the quality of the solution is guaranteed
- **5–50**: DMPC + Graph Coherence Protocol, or MADDPG
- **More than 50 racks**: GAT + Attention + RL, end-to-end strategy is the most realistic choice
- **Competing parties**: Game theory framework (Nash equilibrium/correlated equilibrium)
- **Significant uncertainty** (GPS errors, wind disturbance): probabilistic conflict detection + robust MPC

---

## 5. Frontier progress and trends

### 5.1 End-to-end reinforcement learningThe [SMARTS](https://github.com/hijkzzz/SMARTS) platform and the [AlphaPilot](https://www.microsoft.com/en-us/research/project/alpha-pilot/) project published by Google DeepMind in 2023 promoted the application of end-to-end RL in UAV clusters, demonstrating that a single policy network from perception to decision-making can directly process raw sensor data.

### 5.2 Federated Learning

On the premise of protecting the privacy of each drone's data, the experience of multiple drones is aggregated through federated learning:
1. Local training strategy for each UAV
2. Only upload gradients instead of raw data to the central server
3. Issue new strategies after aggregating updates

It solves the problems of data dispersion and difficulty in label acquisition in UAV clusters.

### 5.3 Uncertain Robust MPC

In recent years, **tube-based MPC** and **scenario-based MPC** model uncertainty as bounded perturbations or probabilistic scenarios to explicitly constrain robustness in optimization problems:

$$
\forall \omega \in \mathbb{W}: \quad \mathbf{x}[k+1] = A\mathbf{x}[k] + B\mathbf{u}[k] + E\omega[k]
$$

Precompute "invariant sets" to ensure that safety constraints are still met in the worst case.

### 5.4 Multi-objective conflict resolution

In actual tasks, conflict resolution also needs to consider:
- **Task Completion Rate**: Don’t just go around and cause the task to time out.
- **Energy Consumption**: UAVs with limited power need to minimize additional flight distance
- **Communication Delay**: Information lag in distributed systems may lead to misjudgments
- **Fairness**: Certain UAVs cannot always give in (hungry problem)

**Pareto optimal frontier** search is the core tool for solving multi-objective conflict resolution.

---

## 6. Summary

Conflict resolution in UAV path planning is a cross-cutting problem spanning geometric calculation, optimization theory, distributed systems and machine learning. From the earliest geometric rate barrier method, to MILP global optimization, to distributed MPC and deep reinforcement learning, the core driving force for algorithm evolution has always been:

> **How to find safer trajectories for more drones in a shorter time and under greater uncertainty. **The future trend will be **hybrid architecture**: using learning methods to make fast local decisions, using optimization methods to verify global trajectories, and using communication protocols to ensure the consistency of multi-machine collaboration. The combination of the three can truly realize safe, efficient, and scalable autonomous flight of UAV swarms.

---

**References** (sorted by time):1. van den Berg, J., Lin, M., & Manocha, D. (2008). *Reciprocal velocity obstacles for real-time multi-agent navigation.* IEEE International Conference on Robotics and Automation (ICRA).
2. Richards, A., & How, J. P. (2002). *Aircraft trajectory planning with collision avoidance using mixed integer linear programming.* AIAA Guidance, Navigation, and Control Conference (GNC).
3. Alonso-Mora, J., et al. (2018). *Optimization-based collision avoidance for multi-vehicle systems.* IEEE Transactions on Robotics (TRO).
4. Everett, M., et al. (2021). *Collision avoidance in dense traffic with deep reinforcement learning.* IEEE International Conference on Robotics and Automation (ICRA).
5. Zhou, M., et al. (2019). *A survey on path planning for UAVs in cluttered environments.* IEEE Transactions on Intelligent Transportation Systems (T-ITS).
6. Lowe, R., et al. (2017). *Multi-agent actor-critic for mixed cooperative-competitive environments (MADDPG).* Conference on Neural Information Processing Systems (NeurIPS).
7. Foerster, J., et al. (2018). *Counterfactual multi-agent policy gradients (COMA).* AAAI Conference on Artificial Intelligence.
8. Rashid, T., et al. (2018). *QMIX: Monotonic value function factorisation for deep multi-agent reinforcement learning.* International Conference on Machine Learning (ICML).
9. Veličković, P., et al. (2018). *Graph attention networks.* International Conference on Learning Representations (ICLR).
10. Yan, C., et al. (2025). *Multi-Agent Reinforcement Learning With Spatial-Temporal Attention for Flocking With Collision Avoidance of a Scalable Fixed-Wing UAV Fleet.* IEEE Transactions on Intelligent Transportation Systems (T-ITS).
11. Huo, D., et al. (2023). *Collision-Free Model Predictive Trajectory Tracking Control for UAVs in Obstacle Environment.* IEEE Transactions on Aerospace and Electronic Systems (TAES).
12. Fan, T., et al. (2020). *Didistributed Multi-Robot Collision Avoidance via Deep Reinforcement Learning for Navigation in Complex Scenarios.* The International Journal of Robotics Research (IJRR).
13. Jiang, C., et al. (2024). *Distributed Sampling-Based Model Predictive Control via Belief Propagation for Multi-Robot Formation Navigation.* IEEE Robotics and Automation Letters (RA-L).
14. Goeckner, A., et al. (2024). *Graph Neural Network-based Multi-Agent Reinforcement Learning for Resilient Distributed Coordination of Multi-Robot Systems.* IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS).