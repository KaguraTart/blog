---
title: "Hundreds of machines fly together: A comprehensive review of the methodology for large-scale drone dispatching problems"
description: "From multi-agent reinforcement learning to graph neural networks, we systematically sort out solutions to large-scale drone dispatch problems. Covering macro-level global task allocation (MARL/GNN/Attention), meso-level conflict coordination (QMIX/MAPPO/GNN), and micro-level real-time obstacle avoidance (MPC/ORCA), abandoning offline methods such as integer planning, focusing on differentiable end-to-end learning routes, and analyzing actual engineering challenges in urban air traffic (UAM) scenarios."
tags: ["drone", "large-scale scheduling", "Multi-agent reinforcement learning", "MARL", "GNN", "UAM", "urban air mobility", "path planning", "ORCA", "attention mechanism", "Attention"]
pubDate: 2026-04-15
---

# Hundreds of machines fly together: A complete review of the methodology for large-scale drone dispatching problems

## 1. Problem definition: What is "large scale"?

When the number of drones increases from 1 to 100 or 1,000, the nature of the problem changes qualitatively:

**Small scale (1-10 units):**
- Fully centralized scheduling, central planner calculates all trajectories at once
- O(n²) conflict detection + A* / RRT* global planning
- Real-time requirements: second level

**Medium scale (10-100 aircraft):**
- Partially distributed, regional/hierarchical management by central scheduler
- Communication delays and coordination protocols need to be considered
- Real-time requirements: sub-second level

**Large scale (100-1000+ racks):**
- The computational complexity of pure centralized scheduling explodes (NP-hard)
- Communication topology becomes a bottleneck
- **Emergent behavior**: The global effects of local decisions are unpredictable
- Real-time requirements: millisecond level

This article focuses on the **differentiable and learnable** method route - which does not rely on offline optimization such as integer programming, but uses reinforcement learning and graph neural networks to solve large-scale scheduling problems.

## 2. Three-tier architecture: standard paradigm for large-scale scheduling

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

The three-layer **time scale** and **spatial scale** are naturally decoupled and are the most classic hierarchical method for large-scale scheduling.

## 3. Macro level: global task allocation

### 3.1 Why not use integer programming (MILP)?

MILP is a classic method in academia, but it has three fatal problems in actual large-scale UAV scenarios:

1. **Computation is not scalable**: 50 tasks + 20 drones is already the limit of MILP; the solution time for 1000 drones explodes from minutes to hours.
2. **Unable to update online**: MILP needs to be re-solved every time it is re-planned, and dynamic task insertion is extremely expensive.
3. **Difficult to deal with randomness**: Dynamic factors such as wind speed changes, mission cancellations, drone failures, etc. will destroy the constraint assumptions of MILP

Therefore, the correct route at the macro level is the **learning method**: train a policy network, input the task and drone status, and **one forward propagation** outputs the task assignment results.

### 3.2 Attention-based task allocation network

The most intuitive learning method is to **use the Attention mechanism for task allocation** - this is the same idea as token matching in LLM:

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
```**Why does Attention work? **
- Self-attention captures the dependencies between tasks (for example, two tasks are very close and should be assigned to the same drone)
- Cross-attention captures task-drone fit
- Output is O(1) parallel and does not scale with the number of tasks (because matrix multiplication can be parallelized by the GPU)

### 3.3 Graph Neural Network (GNN) for task allocation

GNN is another powerful method, especially suitable for scenarios with obvious topology:

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

**Core advantages of GNN:**
- Can handle graphs of any size, and use small-scale graphs to generalize to large-scale graphs during training
- Features on edges can encode constraints (distance, time window compatibility)
- The messaging mechanism handles sparsity naturally - each drone only needs to look at nearby tasks

### 3.4 Evolutionary Algorithm: ACO Ant Colony Optimization

Ant colony algorithm (ACO) is a learning method that does not require gradients and is suitable for combinatorial optimization problems such as task allocation:

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

The advantage of ACO is that it does not require training and can be run directly; the disadvantage is that it needs to be re-iterated every time it is re-planned, and there is a delay problem in online scenarios.

### 3.5 End-to-end RL: macro-micro unified learning

The most cutting-edge idea is to remove the discrete decision-making at the macro level and directly use RL to output continuous trajectories**:

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

The advantage of this end-to-end approach is that it seamlessly connects macro and micro. The disadvantage is that training is difficult - it requires careful design of reward shaping and course learning strategies.

## 4. Meso layer: multi-machine conflict coordination

### 4.1 Why is local coordination sufficient?

Global scheduling does not require precise coordination of each pair of drones, it only needs to ensure coordination within the local conflict window:

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

### 4.2 MARL: Distributed coordinated multi-agent reinforcement learning

MARL is the most powerful tool when the number of drones within the conflict window is 10-50.

**Core Challenge: Non-stationarity**

When agent A is trained, the strategy of agent B is also changing, causing A's "optimal" strategy to continuously drift - this is a fundamental problem in training.

**Solution 1: QMIX - Centralized Training + Distributed Execution (CTDE)**

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
```**Solution 2: MAPPO – PPO-based multi-agent approach**

MAPPO (Multi-Agent PPO) is one of the best MARL algorithms for UAV coordination tasks in recent years:

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

### 4.3 Implicit communication based on GNN

GNNs can pass information without explicit communication - the policy network input for each drone is the graph node feature, naturally aggregating neighbor information through message passing:

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

This kind of implicit communication is more robust than explicit communication - no assumptions about communication delay and packet loss models are required.

## 5. Micro layer: real-time obstacle avoidance

### 5.1 ORCA (Optimal Reciprocal Collision Avoidance)

ORCA is currently the most widely implemented real-time obstacle avoidance algorithm:

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

**ORCA's advantages:** Fully distributed, O(n) complexity, theoretical guarantee of collision avoidance.

**ORCA's limitations:** Dynamic constraints are not considered (the generated "optimal speed" may not be caught by the drone).

### 5.2 MPC + ORCA combination

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

## 6. Latest research on urban air mobility (UAM) scenarios

**Throughput Maximizing Takeoff Scheduling for eVTOL (arXiv:2503.17313)**

Investigate how to maximize eVTOL throughput in UAM scenarios:

- Key insight: Takeoff scheduling is the throughput bottleneck
- Method: Online learning scheduling strategy, dynamic optimization of take-off sequence and time window
- Result: At a demand of 1000 sorties/h, the scheduling strategy reduces the delay rate from 23% to 4%

**Scheduling Aerial Vehicles in UAM (arXiv:2108.01608)**

- Model the UAM scheduling problem as a constrained shortest path problem
- Propose a data-driven heuristic method that can handle no-fly zones, weather delays, and airport capacity constraints
- Verified real-time coordination of 500+ drones in a simulation environment**Embodied AI-Enhanced IoMT Edge Computing: UAV Trajectory Optimization (arXiv:2512.20902)**

- Joint optimization problem of UAV + edge computing
-Using reinforcement learning to optimize UAV trajectories + task offloading decisions
- Accounts for time-varying channel and mobility prediction uncertainties

## 7. Training strategy: How to make MARL converge in large-scale scenarios

The training of MARL has always been difficult. The larger the scale, the more difficult it is to converge. Here are proven training techniques:

### 7.1 Curriculum Learning

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

### 7.2 Normalization techniques

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

### 7.3 Experience replay partition

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

## 8. Current challenges and frontier open issues

### 8.1 Core issues that remain unresolved

1. **Training scalability**: For MARL training of 1000+ drones, GPU memory and communication overhead are bottlenecks
2. **Communication Failure Scenario**: When communication between drones is interrupted, how does distributed coordination maintain consistency?
3. **Heterogeneous fleet**: Mixed scheduling of UAVs with different dynamic characteristics (large UAV + small UAV)
4. **Dynamic task insertion**: Insert new tasks (such as emergency material delivery) during flight. How to re-plan in real time without crashing?
5. **Adversarial Robustness**: Safe Scheduling under Malicious Drone or GPS Spoofing Attacks

### 8.2 Frontier Research Directions

- **LLM as a scheduler**: using large models for task understanding + natural language coordination (see the article on hierarchical VLM)
- **World Model + Scheduling**: Use generative models to predict future traffic flows and avoid conflicts in advance
- **Neural Symbolic Scheduling**: a hybrid system combining neural networks (perception) and symbolic planning (logical reasoning)
- **Multi-objective reward learning**: Use preference learning to automatically adjust the security vs. efficiency trade-off

## 9. Summary

Large-scale drone scheduling is a naturally layered problem:- **Macro layer** (who goes where) → Attention / GNN / ACO / End-to-end RL
- **Mesolayer** (local coordination) → QMIX / MAPPO / GNN implicit communication
- **Micro Layer** (real-time obstacle avoidance) → ORCA/MPC

**Core idea**: Use learnable methods to replace offline optimization. Each layer can be independently trained, incrementally updated, and supports online re-planning.

---

*References (in order of citation in the text)*

1. van den Berg et al., "Reciprocal Collision Avoidance for Multiple Mobile Robots", IJRR, 2011
2. Rashid et al., "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning", ICML, 2018
3. Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games", ICLR, 2022
4. Pooladsanj et al., "Throughput Maximizing Takeoff Scheduling for eVTOL Vehicles", arXiv:2503.17313, 2025
5. Rigas et al., "Scheduling Aerial Vehicles in an Urban Air Mobility Scheme", arXiv:2108.01608, 2021
6. Mu et al., "Embodied AI-Enhanced IoMT Edge Computing: UAV Trajectory Optimization", arXiv:2512.20902, 2025
7. Zhou et al., "OmniShow: Unifying Multimodal Conditions for Human-Object Interaction", arXiv:2604.11804, 2026*Author: Kagura Tart | 2026-04-15*