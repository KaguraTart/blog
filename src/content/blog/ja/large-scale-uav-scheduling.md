---
title: "数百台のマシンが一緒に飛行: 大規模なドローン派遣の問題に対する方法論の包括的なレビュー"
description: "マルチエージェント強化学習からグラフニューラルネットワークまで、大規模ドローン派遣問題の解決策を体系的に整理します。マクロレベルのグローバルタスク割り当て（MARL/GNN/Attendance）、メソレベルの衝突調整（QMIX/MAPPO/GNN）、ミクロレベルのリアルタイム障害物回避（MPC/ORCA）をカバーし、整数計画などのオフライン手法を放棄し、微分可能なエンドツーエンドの学習ルートに焦点を当て、都市航空交通（UAM）シナリオにおける実際のエンジニアリング課題を分析します。"
tags: ["ドローン", "大規模なスケジューリング", "マルチエージェント強化学習", "マール", "GNN", "UAM", "都市部のエアモビリティ", "パスの計画", "オルカ", "注意メカニズム", "注意"]
pubDate: 2026-04-15
sourceHash: "9fbd426769d12070bb2cc1eb5cfd88e0dd839003"
---

# 数百台のマシンが一緒に飛行: 大規模ドローン派遣の問題に対する方法論の完全な見直し

## 1. 問題定義: 「大規模」とは何ですか?

ドローンの数が 1 台から 100 台、または 1,000 台に増加すると、問題の性質は質的に変化します。

**小規模 (1 ～ 10 ユニット):**
- 完全に集中化されたスケジューリング、中央プランナーがすべての軌道を一度に計算します
- O(n²) 件の競合検出 + A* / RRT* のグローバル プランニング
- リアルタイム要件: 第 2 レベル

**中規模 (10 ～ 100 機):**
- 中央スケジューラによる部分分散、地域/階層管理
- 通信遅延と調整プロトコルを考慮する必要がある
- リアルタイム要件: 1 秒未満のレベル

**大規模 (100 ～ 1000 ラック以上):**
- 純粋な集中スケジューリングの計算の複雑さが爆発する (NP ハード)
- 通信トポロジがボトルネックになる
- **緊急の行動**: ローカルな決定が世界に及ぼす影響は予測不可能です
- リアルタイム要件: ミリ秒レベル

この記事では、**微分可能で学習可能な**手法ルートに焦点を当てます。これは、整数計画法などのオフライン最適化に依存せず、強化学習とグラフ ニューラル ネットワークを使用して大規模なスケジューリング問題を解決します。

## 2. 3 層アーキテクチャ: 大規模なスケジューリングの標準パラダイム

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

3 層の **時間スケール** と **空間スケール** は自然に分離されており、大規模なスケジューリングのための最も古典的な階層的方法です。

## 3. マクロレベル: グローバルタスク割り当て

### 3.1 整数計画法 (MILP) を使用しないのはなぜですか?

MILP は学術界では古典的な手法ですが、実際の大規模 UAV シナリオでは 3 つの致命的な問題があります。

1. **計算はスケーラブルではありません**: 50 タスク + 20 ドローンはすでに MILP の制限です。 1,000 台のドローンの解決時間は、数分から数時間に爆発的に増加します。
2. **オンラインで更新できません**: MILP は再計画されるたびに再解決する必要があり、動的なタスクの挿入は非常にコストがかかります。
3. **ランダム性への対処が難しい**: 風速の変化、ミッションのキャンセル、ドローンの故障などの動的要因により、MILP の制約条件が破壊されます。

したがって、マクロレベルでの正しいルートは**学習方法**です。つまり、ポリシーネットワークを学習させ、タスクとドローンのステータスを入力し、**1回の順伝播**でタスクの割り当て結果を出力します。

### 3.2 アテンションベースのタスク割り当てネットワーク

最も直観的な学習方法は、**タスク割り当てにアテンション メカニズムを使用する**です。これは、LLM のトークン マッチングと同じ考え方です。

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
```**なぜ注意が機能するのでしょうか? **
- Self-attention はタスク間の依存関係を把握します (たとえば、2 つのタスクは非常に近いため、同じドローンに割り当てる必要があります)
- クロスアテンションがタスクとドローンの適合性を捉える
- 出力は O(1) 並列であり、タスクの数に応じてスケールされません (行列の乗算は GPU によって並列化できるため)

### 3.3 タスク割り当てのためのグラフ ニューラル ネットワーク (GNN)

GNN も強力な方法であり、トポロジが明らかなシナリオに特に適しています。

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

**GNN の主な利点:**
- あらゆるサイズのグラフを処理でき、トレーニング中に小規模なグラフを使用して大規模なグラフに一般化できます。
- エッジ上のフィーチャは制約 (距離、時間ウィンドウの互換性) をエンコードできます。
- メッセージング メカニズムは疎性を自然に処理します - 各ドローンは近くのタスクのみを確認する必要があります

### 3.4 進化的アルゴリズム: ACO アリ コロニーの最適化

Ant コロニー アルゴリズム (ACO) は、勾配を必要としない学習方法であり、タスク割り当てなどの組み合わせ最適化問題に適しています。

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

ACO の利点は、トレーニングが必要なく、直接実行できることです。欠点は、再計画するたびに繰り返す必要があり、オンライン シナリオでは遅延の問題が発生することです。

### 3.5 エンドツーエンドの RL: マクロとミクロの統合学習

最も最先端のアイデアは、マクロ レベルでの個別の意思決定を削除し、RL を直接使用して連続的な軌跡を出力することです**。

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

このエンドツーエンドのアプローチの利点は、マクロとミクロをシームレスに接続できることです。欠点は、トレーニングが難しいことです。報酬の形成とコース学習戦略を慎重に設計する必要があります。

## 4. Meso レイヤー: マルチマシンの競合調整

### 4.1 なぜローカル調整だけで十分なのでしょうか?

グローバル スケジューリングでは、ドローンの各ペアを正確に調整する必要はありません。必要なのは、ローカルな紛争ウィンドウ内での調整を確保することだけです。

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

### 4.2 MARL: 分散協調マルチエージェント強化学習

MARL は、競合ウィンドウ内のドローンの数が 10 ～ 50 機の場合に最も強力なツールです。

**中心的な課題: 非定常性**

エージェント A がトレーニングされると、エージェント B の戦略も変化し、A の「最適な」戦略が常に変動します。これはトレーニングにおける根本的な問題です。

**解決策 1: QMIX - 集中トレーニング + 分散実行 (CTDE)**

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
```**ソリューション 2: MAPPO – PPO ベースのマルチエージェント アプローチ**

MAPPO (Multi-Agent PPO) は、近年の UAV 調整タスクに最適な MARL アルゴリズムの 1 つです。

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

### 4.3 GNN に基づく暗黙的な通信

GNN は明示的な通信なしで情報を渡すことができます。各ドローンのポリシー ネットワーク入力はグラフ ノード機能であり、メッセージ パッシングを通じて近隣情報を自然に集約します。

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

この種の暗黙的な通信は、明示的な通信よりも堅牢です。通信遅延やパケット損失モデルについての仮定は必要ありません。

## 5. マイクロレイヤー: リアルタイムの障害物回避

### 5.1 ORCA (最適な相互衝突回避)

ORCA は、現在最も広く実装されているリアルタイム障害物回避アルゴリズムです。

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

**ORCA の利点:** 完全分散、O(n) の複雑さ、衝突回避の理論的保証。

**ORCA の制限:** 動的制約は考慮されません (生成された「最適な速度」はドローンによって捕捉されない可能性があります)。

### 5.2 MPC + ORCA の組み合わせ

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

## 6. 都市航空モビリティ (UAM) シナリオに関する最新の研究

**eVTOL のスループットを最大化する離陸スケジュール (arXiv:2503.17313)**

UAM シナリオで eVTOL スループットを最大化する方法を調査します。

- 重要な洞察: 離陸スケジュールがスループットのボトルネックである
- 方法: オンライン学習のスケジュール戦略、離陸シーケンスと時間枠の動的最適化
- 結果: 1000 出撃/時間の要求で、スケジューリング戦略により遅延率が 23% から 4% に減少しました。

**UAM での航空機のスケジュール (arXiv:2108.01608)**

- UAM スケジューリング問題を制約付き最短パス問題としてモデル化する
- 飛行禁止区域、天候による遅延、空港の収容力の制約に対処できるデータ主導のヒューリスティック手法を提案する
- シミュレーション環境で500台以上のドローンのリアルタイム調整を検証**組み込み型 AI 強化 IoMT エッジ コンピューティング: UAV 軌道最適化 (arXiv:2512.20902)**

- UAV + エッジコンピューティングの統合最適化問題
- 強化学習を使用して UAV 軌道 + タスクオフロードの決定を最適化
- 時間とともに変化するチャネルとモビリティ予測の不確実性を考慮

## 7. トレーニング戦略: 大規模シナリオで MARL を収束させる方法

MARLの訓練は常に困難でした。規模が大きくなるほど、収束するのは難しくなります。実証済みのトレーニングテクニックは次のとおりです。

### 7.1 カリキュラム学習

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

### 7.2 正規化手法

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

### 7.3 エクスペリエンス再生パーティション

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

## 8. 現在の課題と未解決の未解決問題

### 8.1 未解決のままの中核問題

1. **トレーニングのスケーラビリティ**: 1,000 機以上のドローンの MARL トレーニングでは、GPU メモリと通信オーバーヘッドがボトルネックになります
2. **通信障害のシナリオ**: ドローン間の通信が中断された場合、分散調整はどのようにして一貫性を維持しますか?
3. **異種フリート**: 異なる動的特性を持つ UAV の混合スケジューリング (大型 UAV + 小型 UAV)
4. **動的タスク挿入**: 飛行中に新しいタスク (緊急物資配達など) を挿入します。クラッシュせずにリアルタイムで再計画するにはどうすればよいでしょうか?
5. **敵対的な堅牢性**: 悪意のあるドローンまたは GPS スプーフィング攻撃下での安全なスケジューリング

### 8.2 フロンティア研究の方向性

- **スケジューラとしての LLM**: タスク理解 + 自然言語調整のための大規模モデルの使用 (階層型 VLM に関する記事を参照)
- **ワールド モデル + スケジューリング**: 生成モデルを使用して将来の交通の流れを予測し、事前に競合を回避します。
- **ニューラル シンボリック スケジューリング**: ニューラル ネットワーク (知覚) とシンボリック プランニング (論理的推論) を組み合わせたハイブリッド システム
- **多目的報酬学習**: 優先学習を使用して、セキュリティと効率のトレードオフを自動的に調整します。

## 9. まとめ

大規模なドローンのスケジュール設定は、当然のことながら階層化された問題です。- **マクロ層** (誰がどこに行くか) → アテンション / GNN / ACO / エンドツーエンド RL
- **Mesolayer** (ローカル調整) → QMIX / MAPPO / GNN 暗黙的通信
- **マイクロレイヤー** (リアルタイム障害物回避) → ORCA/MPC

**中心的なアイデア**: 学習可能な方法を使用して、オフラインの最適化を置き換えます。各レイヤーは個別にトレーニング、段階的に更新でき、オンラインの再計画をサポートします。

---

＊参考文献（本文中の引用順）＊

1. van den Berg et al.、「複数の移動ロボットの相互衝突回避」、IJRR、2011
2. Rashid et al.、「QMIX: 深層マルチエージェント強化学習のための単調値関数因数分解」、ICML、2018
3. Yu 他、「協力型マルチエージェント ゲームにおける PPO の驚くべき効果」、ICLR、2022
4. Pooladsanj et al.、「eVTOL 車両の離陸スケジュールを最大化するスループット」、arXiv:2503.17313、2025
5. Rigas et al.、「都市航空モビリティスキームにおける航空機のスケジューリング」、arXiv:2108.01608、2021
6. Mu et al.、「Embodied AI-Enhanced IoMT Edge Computing: UAV Trajectory Optimization」、arXiv:2512.20902、2025
7. Zhou et al.、「OmniShow: Unifying Multimodal Conditions for Human-Object Interaction」、arXiv:2604.11804、2026※著者：神楽タルト | 2026-04-15*