---
title: "百机同飞：大规模无人机调度问题的方法论全梳理"
description: "从多智能体强化学习到图神经网络，系统梳理大规模无人机调度问题的解决思路。覆盖宏观层全局任务分配（MARL / GNN / Attention）、中观层冲突协调（QMIX / MAPPO / GNN）、微观层实时避障（MPC / ORCA），摒弃整数规划等离线方法，专注可微分的端到端学习路线，并分析城市空中交通（UAM）场景下的实际工程挑战。"
tags: ["无人机", "大规模调度", "多智能体强化学习", "MARL", "GNN", "UAM", "城市空中交通", "路径规划", "ORCA", "注意力机制", "Attention"]
date: 2026-04-15
---

# 百机同飞：大规模无人机调度问题的方法论全梳理

## 1. 问题定义：什么是"大规模"？

当无人机数量从 1 架增加到 100 架、1000 架时，问题性质发生了**质变**：

**小规模（1-10 架）：**
- 完全集中式调度，中央规划器一次性计算所有轨迹
- O(n²) 冲突检测 + A* / RRT* 全局规划
- 实时性要求：秒级

**中规模（10-100 架）：**
- 部分分布式，中央调度器分区域 / 分层管理
- 需要考虑通信延迟和协调协议
- 实时性要求：亚秒级

**大规模（100-1000+ 架）：**
- 纯集中式调度计算复杂度爆炸（NP-hard）
- 通信拓扑成为瓶颈
- **涌现行为**（Emergent behavior）：局部决策的全局效果不可预测
- 实时性要求：毫秒级

本文聚焦**可微分、可学习**的方法路线——不依赖整数规划等离线优化，而是用强化学习和图神经网络解决大规模调度问题。

## 2. 三层架构：大规模调度的标准范式

```
┌──────────────────────────────────────────────────┐
│ 宏观层：全局任务分配 / 路径规划（秒~分钟级）          │
│ 目标：决定"谁去哪、执行什么任务"                    │
│ 方法：MARL / GNN / Attention / 进化算法           │
│ 规模：100-1000+ 无人机                             │
└──────────────────────────────────────────────────┘
                        ↓ 任务分配 + 初始轨迹
┌──────────────────────────────────────────────────┐
│ 中观层：冲突协调 / 多机协同（毫秒~秒级）              │
│ 目标：预测并化解局部冲突                             │
│ 方法：MARL / GNN / 分布式策略网络                    │
│ 规模：10-50 架（局部协调窗口）                      │
└──────────────────────────────────────────────────┘
                        ↓ 协调后的轨迹修正
┌──────────────────────────────────────────────────┐
│ 微观层：实时避障 / 轨迹跟踪（毫秒级）                │
│ 目标：确保单架无人机的安全                           │
│ 方法：MPC / ORCA / 速度障碍法                       │
│ 规模：单架无人机                                    │
└──────────────────────────────────────────────────┘
```

三层**时间尺度**和**空间尺度**天然解耦，是大规模调度最经典的分层方式。

## 3. 宏观层：全局任务分配

### 3.1 为什么不用整数规划（MILP）？

MILP 是学术界的经典方法，但在实际大规模 UAV 场景中有三个致命问题：

1. **计算不可扩展**：50 个任务 + 20 架无人机已经是 MILP 的极限；1000 架无人机的求解时间从分钟级爆炸到小时级
2. **无法在线更新**：MILP 每次重规划都需要重新求解，动态任务插入代价极高
3. **难以处理随机性**：风速变化、任务取消、无人机故障等动态因素会破坏 MILP 的约束假设

因此，宏观层的正确路线是**学习式方法**：训练一个策略网络，输入任务和无人机状态，**一次前向传播**输出任务分配结果。

### 3.2 基于注意力的任务分配网络

最直观的学习方法是**用 Attention 机制做任务分配**——这和 LLM 里做 token 匹配是一个思路：

```python
import torch
import torch.nn as nn

class AttentionTaskScheduler(nn.Module):
    """
    输入：
      - task_embeddings: [N_tasks, D] 任务特征（位置、时间窗、优先级）
      - uav_embeddings: [M_uavs, D] 无人机特征（位置、电量、能力）
    输出：
      - assignment_logits: [N_tasks, M_uavs] 每个任务分配给每架无人机的得分
    """
    def __init__(self, embed_dim=64, n_heads=4):
        super().__init__()
        # 任务-无人机交叉注意力
        self.cross_attention = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        # 任务间自注意力（捕捉任务间的依赖关系）
        self.self_attention = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, task_emb, uav_emb):
        # Step 1: 任务间的依赖建模
        task_encoded, _ = self.self_attention(task_emb, task_emb, task_emb)

        # Step 2: 任务-无人机匹配
        # 将无人机信息作为 query，任务信息作为 key/value
        matched, _ = self.cross_attention(
            query=uav_emb.unsqueeze(1).expand(-1, task_emb.size(0), -1),
            key=task_emb.unsqueeze(0).expand(uav_emb.size(0), -1, -1),
            value=task_emb.unsqueeze(0).expand(uav_emb.size(0), -1, -1)
        )

        # Step 3: 生成分配得分
        # 融合任务编码和匹配结果
        combined = torch.cat([task_encoded.unsqueeze(0).expand(uav_emb.size(0), -1, -1), matched], dim=-1)
        logits = self.fc(combined).squeeze(-1)  # [N_tasks, M_uavs]

        return logits

    def loss(self, logits, assignments, reward):
        """
        训练时：用 RL 信号的 reward 来优化
        也可以用监督学习：如果有 ground-truth 分配
        """
        # 负对数似然作为损失
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -log_probs[assignments].mean()
        return loss
```

**为什么 Attention 有效？**
- 自注意力捕捉了任务间的依赖（比如两个任务距离很近，应该分配给同一架无人机）
- 交叉注意力捕捉了任务-无人机的匹配度
- 输出是并行的 O(1)，不随任务数增长（因为矩阵乘法可以被 GPU 并行化）

### 3.3 图神经网络（GNN）做任务分配

GNN 是另一种强大的方法，特别适合**拓扑结构明显**的场景：

```python
import torch_geometric as pyg

class GNNTaskScheduler(torch.nn.Module):
    """
    将调度问题建模为图：
    - 节点：无人机节点 + 任务节点
    - 边：无人机与任务之间的可能分配关系
    """
    def __init__(self):
        super().__init__()
        self.node_encoder = nn.Linear(16, 64)
        self.conv1 = pyg.nn.GATConv(64, 128, heads=4)
        self.conv2 = pyg.nn.GATConv(128 * 4, 64, heads=1)
        self.edge_decoder = nn.Linear(64 * 2, 1)  # 边上的分配得分

    def forward(self, data):
        x = self.node_encoder(data.x)  # [N_nodes, 16] → [N_nodes, 64]

        # 消息传递：聚合邻居信息
        x = self.conv1(x, data.edge_index).relu()
        x = self.conv2(x, data.edge_index).relu()

        # 边解码：每条边（任务-无人机对）的分配得分
        src, dst = data.edge_index
        edge_features = torch.cat([x[src], x[dst]], dim=-1)
        edge_scores = self.edge_decoder(edge_features).squeeze(-1)
        return edge_scores  # [N_edges] 每个任务-无人机对的分配得分
```

**GNN 的核心优势：**
- 可以处理任意规模的图，训练时用小规模图泛化到大规模图
- 边上的特征可以编码约束（距离、时间窗兼容性）
- 消息传递机制天然处理了稀疏性——每架无人机只需要看附近的任务

### 3.4 进化算法：ACO 蚁群优化

蚁群算法（ACO）是**不需要梯度**的学习式方法，适合任务分配这种组合优化问题：

```python
class ACOTaskScheduler:
    """
    ACO 的核心思想：
    - 每只蚂蚁走出一条完整路径 = 一个调度方案
    - 信息素引导后续蚂蚁找到更优方案
    - 适合离线批量优化，不适合实时在线更新
    """
    def __init__(self, n_ants=50, n_iterations=100, pheromone_decay=0.95):
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.pheromone_decay = pheromone_decay
        # 信息素矩阵：[M_uavs, N_tasks]
        self.pheromone = np.ones((n_uavs, n_tasks))

    def solve(self, tasks, uavs, distances):
        """
        输入：任务列表、无人机列表、距离矩阵
        返回：每个无人机的任务分配序列
        """
        best_assignment = None
        best_cost = float('inf')

        for iteration in range(self.n_iterations):
            all_assignments = []

            for ant in range(self.n_ants):
                # 每只蚂蚁独立构造解
                assignment = self._construct_solution(tasks, uavs, distances)
                cost = self._evaluate_cost(assignment, distances)
                all_assignments.append((assignment, cost))

                if cost < best_cost:
                    best_cost = cost
                    best_assignment = assignment

            # 更新信息素：更优解留下更多信息素
            self._update_pheromone(all_assignments, best_assignment)

        return best_assignment

    def _construct_solution(self, tasks, uavs, distances):
        assignment = {uav_id: [] for uav_id in range(len(uavs))}
        remaining_tasks = set(range(len(tasks)))

        while remaining_tasks:
            # 每只蚂蚁轮流为每架无人机分配一个任务
            for uav_id in range(len(uavs)):
                if not remaining_tasks:
                    break
                # 基于信息素 + 距离的加权概率选择
                probs = self._compute_probabilities(uav_id, remaining_tasks, distances)
                chosen_task = np.random.choice(list(remaining_tasks), p=probs)
                assignment[uav_id].append(chosen_task)
                remaining_tasks.remove(chosen_task)

        return assignment

    def _update_pheromone(self, all_assignments, best):
        # 信息素衰减
        self.pheromone *= self.pheromone_decay
        # 优胜蚂蚁释放信息素
        for assignment, cost in all_assignments:
            reward = 1.0 / cost
            for uav_id, task_seq in assignment.items():
                for task_id in task_seq:
                    self.pheromone[uav_id, task_id] += reward
```

ACO 的优势是**不需要训练**，可以直接运行；缺点是每次重规划都需要重新迭代，在线场景下有延迟问题。

### 3.5 端到端 RL：宏-微观统一学习

最前沿的思路是**去掉宏观层的离散决策，直接用 RL 输出连续轨迹**：

```python
class EndToEndMacroRL(nn.Module):
    """
    每个无人机独立运行一个策略网络
    输入：本地感知（附近任务位置）+ 全局信息（通信消息）
    输出：下一个目标点 + 速度指令
    """
    def __init__(self, obs_dim=32, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # 输出：[target_x, target_y, speed, priority]
        )

    def forward(self, obs):
        return self.network(obs)

    def get_reward(self, task_completion, collision, energy_remaining):
        """
        reward 设计：
        - +10.0 完成任务
        - -50.0 碰撞（立即终止）
        - -0.5 每步的能源消耗
        - +5.0 接近目标
        """
        r = 0.0
        r += 10.0 * task_completion
        r += -50.0 * collision
        r += -0.5 * (1.0 - energy_remaining)
        return r
```

这种端到端方法的优点是**无缝衔接宏微观**，缺点是训练困难——需要精心设计 reward shaping 和课程学习策略。

## 4. 中观层：多机冲突协调

### 4.1 为什么局部协调就够了？

全局调度不需要精确协调每对无人机，只需要确保**局部冲突窗口内**的协调：

```python
def get_local_conflict_pairs(trajectories, time_window=10.0, conflict_distance=50.0):
    """
    只检测 10 秒时间窗口内、50 米距离内的冲突对
    大幅减少需要协调的无人机数量
    """
    conflicts = []
    for i, traj_i in enumerate(trajectories):
        for j, traj_j in enumerate(trajectories[i+1:], i+1):
            min_dist = compute_min_distance(traj_i, traj_j)
            tca = compute_time_of_closest_approach(traj_i, traj_j)

            if min_dist < conflict_distance and tca < time_window:
                conflicts.append((i, j, min_dist, tca))
    return conflicts  # 远小于 O(n²)
```

### 4.2 MARL：分布式协调的多智能体强化学习

当冲突窗口内的无人机数量在 10-50 架时，MARL 是最有力的工具。

**核心挑战：非平稳性（Non-stationarity）**

当智能体 A 训练时，智能体 B 的策略也在变化，导致 A 的"最优"策略不断漂移——这是一个训练上的根本难题。

**解决方案 1：QMIX —— 中心化训练 + 分布式执行（CTDE）**

```python
class QMIXAgent:
    """
    QMIX 的核心思想：
    - 训练时：中心式 Critic 看到全局状态（解决非平稳性）
    - 执行时：每个 Agent 只用本地观测（保证可扩展性）
    - 混合网络：将每个 agent 的 Q 值合并为全局 Q 值，保证 IGM 条件
    """
    def __init__(self, obs_dim, n_actions, n_agents):
        # 本地策略网络（执行时用这个）
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
        # 中心式价值网络（训练时用这个）
        self.critic = nn.Sequential(
            nn.Linear(global_state_dim + n_agents * n_actions, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # 混合网络：保证单调性（∂Q_tot/∂Q_i > 0）
        self.mixer = nn.Sequential(
            nn.Linear(global_state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_agents),
            nn.Softplus()  # 保证正权重
        )

    def forward(self, obs):
        q_vals = self.actor(obs)
        return q_vals

    def mix(self, q_vals, global_state):
        """
        混合网络：将个体 Q 值合并为全局 Q 值
        权重由全局状态生成，保证可微性和单调性
        """
        weights = self.mixer(global_state)
        return (q_vals * weights).sum(dim=-1, keepdim=True)

    def update(self, batch):
        # 从 replay buffer 采样
        obs, actions, rewards, next_obs, dones = batch

        # 计算目标 Q 值
        with torch.no_grad():
            next_q_vals = self.actor(next_obs)
            next_q_tot = self.mix(next_q_vals, next_obs_global)
            targets = rewards + (1 - dones) * 0.99 * next_q_tot

        # 更新 critic
        current_q = self.mix(self.actor(obs), obs_global)
        loss = nn.MSELoss()(current_q, targets)

        # 更新 actor（策略梯度）
        self.actor.zero_grad()
        -current_q.mean().backward()
```

**解决方案 2：MAPPO —— 基于 PPO 的多智能体方法**

MAPPO（Multi-Agent PPO）是近年来在 UAV 协调任务上效果最好的 MARL 算法之一：

```python
class MAPPOAgent:
    """
    MAPPO = PPO 在多智能体场景下的扩展
    核心改进：用.clip() 限制策略更新幅度，比 DDPG 稳定得多
    """
    def __init__(self, obs_dim, act_dim):
        self.actor = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(),
                                   nn.Linear(128, act_dim), nn.Softmax())
        self.critic = nn.Sequential(nn.Linear(obs_dim + act_dim * n_agents, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 1))
        self.clip_eps = 0.2  # PPO 的剪切系数
        self.entropy_coef = 0.01

    def compute_loss(self, batch):
        obs, actions, old_log_probs, advantages, returns = batch

        # 当前策略的对数概率
        new_log_probs = self.actor(obs).log_prob(actions)

        # PPO 剪切目标
        ratio = (new_log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 熵正则项（鼓励探索）
        entropy = self.actor(obs).entropy().mean()
        loss = policy_loss - self.entropy_coef * entropy

        return loss
```

### 4.3 基于 GNN 的隐式通信

GNN 可以在不显式通信的情况下传递信息——每架无人机的策略网络输入就是图节点特征，通过消息传递自然地聚合邻居信息：

```python
class GNNCommLayer(torch.nn.Module):
    def forward(self, node_features, edge_index, edge_attr=None):
        # edge_index 动态构建：只连接距离 < 通讯范围的无人机
        # 消息传递 = 邻居特征聚合
        aggregated = pyg.nn.MessagePassing.aggr('max')(
            node_features, edge_index
        )
        # 残差连接
        return node_features + aggregated
```

这种**隐式通信**比显式通信更鲁棒——不需要假设通信延迟和丢包模型。

## 5. 微观层：实时避障

### 5.1 ORCA（Optimal Reciprocal Collision Avoidance）

ORCA 是目前工程落地最广泛的实时避障算法：

```python
def orca_step(agent_pos, agent_vel, other_agents, time_horizon=5.0, radius=1.0):
    """
    ORCA 单步计算：
    输入：当前位置、当前速度、附近所有智能体
    输出：满足安全约束的新速度
    """
    orca_halfplanes = []

    for other in other_agents:
        rel_pos = other.pos - agent_pos
        rel_vel = agent_vel - other.vel
        dist = np.linalg.norm(rel_pos)

        if dist < 2 * radius:
            # 计算 VO（速度障碍）角
            angle = np.arctan2(rel_pos[1], rel_pos[0])
            half_angle = np.arcsin(radius / dist)
            # ORCA 半平面 = VO 的切线方向
            tai = compute_tangent(agent_vel, rel_pos, dist, half_angle)
            orca_halfplanes.append(tai)

    # 在 ORCA 半平面交集内找最优速度（最近于当前速度）
    new_vel = find_optimal_velocity(agent_vel, orca_halfplanes)
    return new_vel
```

**ORCA 的优点：** 完全分布式，O(n) 复杂度，有碰撞避免的理论保证。

**ORCA 的局限：** 不考虑动力学约束（生成的"最优速度"可能无人机追不上）。

### 5.2 MPC + ORCA 组合

```python
class HybridUAVController:
    def __init__(self):
        self.orca = ORCAController()
        self.mpc = NMPCController()

    def compute_control(self, state, env_map, nearby_uavs):
        # Step 1: ORCA 生成安全速度
        safe_vel = self.orca.step(state.pos, state.vel, nearby_uavs)

        # Step 2: MPC 跟踪 ORCA 给出的安全速度（满足动力学约束）
        trajectory = self.mpc.solve(
            state=state,
            desired_velocity=safe_vel,
            constraints=['thrust_max', 'roll_pitch_max', 'battery_min']
        )
        return trajectory
```

## 6. 城市空中交通（UAM）场景的最新研究

**Throughput Maximizing Takeoff Scheduling for eVTOL (arXiv:2503.17313)**

研究如何在 UAM 场景下最大化 eVTOL 的吞吐量：

- 关键洞察：起飞调度（takeoff scheduling）是吞吐量瓶颈
- 方法：在线学习调度策略，动态优化起飞顺序和时间窗口
- 结果：在 1000 架次/h 的需求下，调度策略将延误率从 23% 降到 4%

**Scheduling Aerial Vehicles in UAM (arXiv:2108.01608)**

- 将 UAM 调度问题建模为约束最短路问题
- 提出数据驱动的启发式方法，能处理禁飞区、天气延误、机场容量约束
- 在仿真环境中验证了 500+ 无人机的实时协调

**Embodied AI-Enhanced IoMT Edge Computing: UAV Trajectory Optimization (arXiv:2512.20902)**

- UAV + 边缘计算的联合优化问题
- 用强化学习优化 UAV 轨迹 + 任务卸载决策
- 考虑了时变信道和移动性预测的不确定性

## 7. 训练策略：如何让 MARL 在大规模场景下收敛

MARL 的训练一直是难点，规模越大越难收敛。以下是经过验证的训练技巧：

### 7.1 课程学习（Curriculum Learning）

```python
def curriculum_schedule(n_agents):
    """
    从简单到复杂，逐步增加无人机数量
    让策略先在简单场景学会基本行为，再扩展到复杂场景
    """
    curriculum = [
        (n_agents=2,  reward_scale=1.0,  n_episodes=5000),
        (n_agents=5,  reward_scale=1.2,  n_episodes=10000),
        (n_agents=10, reward_scale=1.5,  n_episodes=20000),
        (n_agents=20, reward_scale=2.0,  n_episodes=30000),
        (n_agents=50, reward_scale=3.0,  n_episodes=50000),
    ]
    return curriculum
```

### 7.2 归一化技巧

```python
class RunningNorm:
    """训练时对观测和奖励做归一化，大幅提升 MARL 收敛稳定性"""
    def __init__(self, dim, clip_range=10.0):
        self.running_mean = np.zeros(dim)
        self.running_var = np.ones(dim)
        self.count = 1e-4
        self.clip_range = clip_range

    def update(self, x):
        delta = x - self.running_mean
        self.count += 1
        self.running_mean += delta / self.count
        self.running_var += delta * (x - self.running_mean)

    def normalize(self, x):
        return np.clip((x - self.running_mean) / np.sqrt(self.running_var + 1e-8),
                       -self.clip_range, self.clip_range)
```

### 7.3 经验回放分区

```python
class PartitionedReplayBuffer:
    """
    按无人机数量分区 replay buffer
    训练时按课程进度从不同分区采样
    """
    def __init__(self):
        self.buffers = {n: ReplayBuffer(capacity=100000) for n in [2, 5, 10, 20, 50]}

    def add(self, n_agents, experience):
        self.buffers[n_agents].add(experience)

    def sample(self, batch_size):
        # 按课程进度加权采样
        weights = {2: 0.05, 5: 0.1, 10: 0.2, 20: 0.3, 50: 0.35}
        samples = []
        for n, w in weights.items():
            n_sample = int(batch_size * w)
            samples.append(self.buffers[n].sample(n_sample))
        return torch.cat(samples, dim=0)
```

## 8. 当前挑战与前沿开放问题

### 8.1 仍未解决的核心问题

1. **训练扩展性**：1000+ 架无人机的 MARL 训练，GPU 内存和通信开销都是瓶颈
2. **通信失效场景**：当无人机之间的通信中断时，分布式协调如何保持一致性？
3. **异构机群**：不同动力学特性的无人机混合调度（大型 UAV + 小型 UAV）
4. **动态任务插入**：飞行过程中插入新任务（如紧急物资投递），如何实时重规划而不崩溃？
5. **对抗鲁棒性**：恶意无人机或 GPS 欺骗攻击下的安全调度

### 8.2 前沿研究方向

- **LLM 作为调度员**：用大模型做任务理解 + 自然语言协调（见分层 VLM 那篇）
- **世界模型 + 调度**：用生成式模型预测未来交通流，提前规避冲突
- **神经符号调度**：结合神经网络（感知）和符号规划（逻辑推理）的混合系统
- **多目标 reward 学习**：用偏好学习（Preference Learning）自动调整安全性 vs 效率的权衡

## 9. 总结

大规模无人机调度是一个天然分层的难题：

- **宏观层**（谁去哪）→ Attention / GNN / ACO / 端到端 RL
- **中观层**（局部协调）→ QMIX / MAPPO / GNN 隐式通信
- **微观层**（实时避障）→ ORCA / MPC

**核心思想**：用可学习的方法替代离线优化，每层都可以独立训练、增量更新、支持在线重规划。

---

*参考文献（按文中引用顺序）*

1. van den Berg et al., "Reciprocal Collision Avoidance for Multiple Mobile Robots", IJRR, 2011
2. Rashid et al., "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning", ICML, 2018
3. Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games", ICLR, 2022
4. Pooladsanj et al., "Throughput Maximizing Takeoff Scheduling for eVTOL Vehicles", arXiv:2503.17313, 2025
5. Rigas et al., "Scheduling Aerial Vehicles in an Urban Air Mobility Scheme", arXiv:2108.01608, 2021
6. Mu et al., "Embodied AI-Enhanced IoMT Edge Computing: UAV Trajectory Optimization", arXiv:2512.20902, 2025
7. Zhou et al., "OmniShow: Unifying Multimodal Conditions for Human-Object Interaction", arXiv:2604.11804, 2026

*作者：Kagura Tart | 2026-04-15*
