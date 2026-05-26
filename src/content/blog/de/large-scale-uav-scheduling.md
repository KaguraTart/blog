---
title: "Hunderte von Maschinen fliegen zusammen: Ein umfassender Überblick über die Methodik für groß angelegte Drohnen-Dispatch-Probleme"
description: "Von Multi-Agent Reinforcement Learning bis hin zu grafischen neuronalen Netzen finden wir systematisch Lösungen für groß angelegte Drohnen-Versandprobleme. Behandelt die globale Aufgabenzuweisung auf Makroebene (MARL/GNN/Attention), die Konfliktkoordination auf Mesoebene (QMIX/MAPPO/GNN) und die Echtzeit-Hindernisvermeidung auf Mikroebene (MPC/ORCA), verzichtet auf Offline-Methoden wie Ganzzahlplanung, konzentriert sich auf differenzierbare End-to-End-Lernrouten und analysiert tatsächliche technische Herausforderungen in Szenarien des städtischen Luftverkehrs (UAM)."
tags: ["Drohne", "Großplanung", "Verstärkungslernen mit mehreren Agenten", "MERGEL", "GNN", "UAM", "urbane Luftmobilität", "Wegplanung", "ORCA", "Aufmerksamkeitsmechanismus", "Aufmerksamkeit"]
pubDate: 2026-04-15
sourceHash: "9fbd426769d12070bb2cc1eb5cfd88e0dd839003"
---

# Hunderte von Maschinen fliegen zusammen: Eine vollständige Überprüfung der Methodik für groß angelegte Drohnen-Dispatch-Probleme

## 1. Problemdefinition: Was ist „großer Maßstab“?

Wenn die Anzahl der Drohnen von 1 auf 100 oder 1.000 steigt, ändert sich die Art des Problems qualitativ:

**Kleiner Maßstab (1-10 Einheiten):**
- Vollständig zentralisierte Planung, zentraler Planer berechnet alle Trajektorien auf einmal
- O(n²)-Konflikterkennung + A* / RRT* globale Planung
- Echtzeitanforderungen: zweite Ebene

**Mittlerer Maßstab (10–100 Flugzeuge):**
- Teilweise verteilte, regionale/hierarchische Verwaltung durch zentralen Scheduler
- Kommunikationsverzögerungen und Koordinierungsprotokolle müssen berücksichtigt werden
- Echtzeitanforderungen: Subsekundenebene

**Großer Maßstab (100-1000+ Racks):**
- Die Rechenkomplexität der rein zentralisierten Planung explodiert (NP-schwer)
- Die Kommunikationstopologie wird zum Engpass
- **Emergentes Verhalten**: Die globalen Auswirkungen lokaler Entscheidungen sind unvorhersehbar
- Echtzeitanforderungen: Millisekundenebene

Dieser Artikel konzentriert sich auf die **differenzierbare und lernbare** Methodenroute, die nicht auf Offline-Optimierung wie Ganzzahlprogrammierung beruht, sondern Verstärkungslernen und grafische neuronale Netze verwendet, um groß angelegte Planungsprobleme zu lösen.

## 2. Dreistufige Architektur: Standardparadigma für die Planung im großen Maßstab

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

Die dreischichtige **Zeitskala** und **räumliche Skala** sind auf natürliche Weise entkoppelt und stellen die klassischste hierarchische Methode für die Planung im großen Maßstab dar.

## 3. Makroebene: globale Aufgabenverteilung

### 3.1 Warum nicht Integer-Programmierung (MILP) verwenden?

MILP ist eine klassische Methode in der Wissenschaft, weist jedoch in tatsächlichen groß angelegten UAV-Szenarien drei schwerwiegende Probleme auf:

1. **Berechnung ist nicht skalierbar**: 50 Aufgaben + 20 Drohnen sind bereits die Grenze von MILP; Die Lösungszeit für 1000 Drohnen explodiert von Minuten auf Stunden.
2. **Online-Aktualisierung nicht möglich**: MILP muss bei jeder Neuplanung neu gelöst werden, und das dynamische Einfügen von Aufgaben ist extrem teuer.
3. **Schwierig mit Zufälligkeit umzugehen**: Dynamische Faktoren wie Windgeschwindigkeitsänderungen, Missionsabbrüche, Drohnenausfälle usw. zerstören die Einschränkungsannahmen von MILP

Daher ist der richtige Weg auf Makroebene die **Lernmethode**: Trainieren Sie ein Richtliniennetzwerk, geben Sie die Aufgabe und den Drohnenstatus ein und **eine Vorwärtsausbreitung** gibt die Ergebnisse der Aufgabenzuweisung aus.

### 3.2 Aufmerksamkeitsbasiertes Aufgabenverteilungsnetzwerk

Die intuitivste Lernmethode besteht darin, **den Aufmerksamkeitsmechanismus für die Aufgabenzuweisung zu verwenden** – dies ist die gleiche Idee wie beim Token-Matching in LLM:

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
```**Warum funktioniert Aufmerksamkeit? **
- Selbstaufmerksamkeit erfasst die Abhängigkeiten zwischen Aufgaben (z. B. liegen zwei Aufgaben sehr nahe beieinander und sollten derselben Drohne zugewiesen werden)
- Cross-Attention erfasst die Task-Drone-Passform
- Die Ausgabe erfolgt O(1)-parallel und skaliert nicht mit der Anzahl der Aufgaben (da die Matrixmultiplikation von der GPU parallelisiert werden kann)

### 3.3 Graph Neural Network (GNN) zur Aufgabenzuweisung

GNN ist eine weitere leistungsstarke Methode, die sich besonders für Szenarien mit offensichtlicher Topologie eignet:

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

**Hauptvorteile von GNN:**
- Kann Diagramme jeder Größe verarbeiten und während des Trainings kleine Diagramme verwenden, um sie auf Diagramme im großen Maßstab zu verallgemeinern
- Features an Kanten können Einschränkungen kodieren (Entfernung, Zeitfensterkompatibilität)
- Der Nachrichtenmechanismus geht mit Sparsity auf natürliche Weise um – jede Drohne muss sich nur die Aufgaben in der Nähe ansehen

### 3.4 Evolutionärer Algorithmus: ACO-Ameisenkolonie-Optimierung

Der Ameisenkolonie-Algorithmus (ACO) ist eine Lernmethode, die keine Gradienten erfordert und sich für kombinatorische Optimierungsprobleme wie die Aufgabenzuweisung eignet:

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

Der Vorteil von ACO besteht darin, dass keine Schulung erforderlich ist und direkt ausgeführt werden kann. Der Nachteil besteht darin, dass es bei jeder Neuplanung wiederholt werden muss und in Online-Szenarien ein Verzögerungsproblem auftritt.

### 3.5 End-to-End-RL: einheitliches Makro-Mikro-Lernen

Die innovativste Idee besteht darin, die diskrete Entscheidungsfindung auf Makroebene zu eliminieren und RL direkt zur Ausgabe kontinuierlicher Trajektorien zu verwenden**:

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

Der Vorteil dieses End-to-End-Ansatzes besteht darin, dass Makro und Mikro nahtlos miteinander verbunden werden. Der Nachteil besteht darin, dass das Training schwierig ist – es erfordert eine sorgfältige Gestaltung der Belohnungsgestaltung und Kurslernstrategien.

## 4. Mesoschicht: Konfliktkoordination mit mehreren Maschinen

### 4.1 Warum ist eine lokale Koordination ausreichend?

Die globale Planung erfordert keine genaue Koordination jedes Drohnenpaares, sie muss lediglich die Koordination innerhalb des lokalen Konfliktfensters sicherstellen:

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

### 4.2 MARL: Verteiltes koordiniertes Lernen zur Verstärkung mehrerer Agenten

MARL ist das leistungsstärkste Werkzeug, wenn die Anzahl der Drohnen im Konfliktfenster 10–50 beträgt.

**Kernherausforderung: Nichtstationarität**

Wenn Agent A trainiert wird, ändert sich auch die Strategie von Agent B, was dazu führt, dass die „optimale“ Strategie von A kontinuierlich abweicht – dies ist ein grundlegendes Problem beim Training.

**Lösung 1: QMIX – Zentralisiertes Training + verteilte Ausführung (CTDE)**

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
```**Lösung 2: MAPPO – PPO-basierter Multi-Agent-Ansatz**

MAPPO (Multi-Agent PPO) ist einer der besten MARL-Algorithmen für UAV-Koordinationsaufgaben der letzten Jahre:

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

### 4.3 Implizite Kommunikation basierend auf GNN

GNNs können Informationen ohne explizite Kommunikation weitergeben – die Richtliniennetzwerkeingabe für jede Drohne ist die Graphknotenfunktion, die auf natürliche Weise Nachbarinformationen durch Nachrichtenübermittlung aggregiert:

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

Diese Art der impliziten Kommunikation ist robuster als die explizite Kommunikation – es sind keine Annahmen über Kommunikationsverzögerungs- und Paketverlustmodelle erforderlich.

## 5. Mikroschicht: Hindernisvermeidung in Echtzeit

### 5.1 ORCA (Optimale reziproke Kollisionsvermeidung)

ORCA ist derzeit der am weitesten verbreitete Echtzeit-Algorithmus zur Hindernisvermeidung:

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

**ORCAs Vorteile:** Vollständig verteilt, O(n)-Komplexität, theoretische Garantie der Kollisionsvermeidung.

**ORCAs Einschränkungen:** Dynamische Einschränkungen werden nicht berücksichtigt (die erzeugte „optimale Geschwindigkeit“ wird möglicherweise nicht von der Drohne erfasst).

### 5.2 MPC + ORCA-Kombination

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

## 6. Neueste Forschung zu Szenarien der urbanen Luftmobilität (UAM).

**Durchsatzmaximierung der Startplanung für eVTOL (arXiv:2503.17313)**

Untersuchen Sie, wie Sie den eVTOL-Durchsatz in UAM-Szenarien maximieren können:

- Wichtige Erkenntnis: Die Startplanung ist der Engpass beim Durchsatz
- Methode: Online-Lernplanungsstrategie, dynamische Optimierung der Startsequenz und des Zeitfensters
- Ergebnis: Bei einem Bedarf von 1000 Einsätzen/h reduziert die Planungsstrategie die Verzögerungsrate von 23 % auf 4 %

**Planung von Luftfahrzeugen in UAM (arXiv:2108.01608)**

- Modellieren Sie das UAM-Planungsproblem als eingeschränktes Kürzeste-Wege-Problem
- Schlagen Sie eine datengesteuerte heuristische Methode vor, die Flugverbotszonen, Wetterverzögerungen und Flughafenkapazitätsbeschränkungen bewältigen kann
- Verifizierte Echtzeitkoordination von über 500 Drohnen in einer Simulationsumgebung**Embodied AI-Enhanced IoMT Edge Computing: UAV-Trajektorienoptimierung (arXiv:2512.20902)**

- Gemeinsames Optimierungsproblem von UAV + Edge Computing
-Verwendung von Reinforcement Learning zur Optimierung von UAV-Flugbahnen und Entscheidungen zur Aufgabenverlagerung
- Berücksichtigt zeitlich variierende Kanal- und Mobilitätsvorhersageunsicherheiten

## 7. Trainingsstrategie: Wie man MARL in groß angelegten Szenarien konvergieren lässt

Die Ausbildung von MARL war schon immer schwierig. Je größer der Maßstab, desto schwieriger ist die Konvergenz. Hier sind bewährte Trainingstechniken:

### 7.1 Lehrplan-Lernen

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

### 7.2 Normalisierungstechniken

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

### 7.3 Erleben Sie die Wiedergabepartition

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

## 8. Aktuelle Herausforderungen und offene Grenzfragen

### 8.1 Kernprobleme, die weiterhin ungelöst sind

1. **Trainingsskalierbarkeit**: Beim MARL-Training von über 1000 Drohnen sind GPU-Speicher und Kommunikationsaufwand Engpässe
2. **Kommunikationsfehlerszenario**: Wie sorgt die verteilte Koordination für Konsistenz, wenn die Kommunikation zwischen Drohnen unterbrochen wird?
3. **Heterogene Flotte**: Gemischte Planung von UAVs mit unterschiedlichen dynamischen Eigenschaften (großes UAV + kleines UAV)
4. **Dynamisches Einfügen von Aufgaben**: Fügen Sie während des Fluges neue Aufgaben (z. B. Lieferung von Notfallmaterial) ein. Wie kann man in Echtzeit neu planen, ohne abzustürzen?
5. **Gegnerische Robustheit**: Sichere Planung bei böswilligen Drohnen- oder GPS-Spoofing-Angriffen

### 8.2 Richtungen der Grenzforschung

- **LLM als Planer**: Verwendung großer Modelle für Aufgabenverständnis + Koordination in natürlicher Sprache (siehe Artikel über hierarchisches VLM)
- **Weltmodell + Terminplanung**: Nutzen Sie generative Modelle, um zukünftige Verkehrsströme vorherzusagen und Konflikte im Voraus zu vermeiden
- **Neural Symbolic Scheduling**: ein Hybridsystem, das neuronale Netze (Wahrnehmung) und symbolische Planung (logisches Denken) kombiniert.
- **Mehrobjektives Belohnungslernen**: Nutzen Sie Präferenzlernen, um den Kompromiss zwischen Sicherheit und Effizienz automatisch anzupassen

## 9. Zusammenfassung

Die Planung groß angelegter Drohnen ist ein natürlich vielschichtiges Problem:- **Makroebene** (wer geht wohin) → Achtung / GNN / ACO / End-to-End RL
- **Mesolayer** (lokale Koordination) → QMIX / MAPPO / GNN implizite Kommunikation
- **Micro Layer** (Echtzeit-Hindernisvermeidung) → ORCA/MPC

**Kernidee**: Erlernbare Methoden nutzen, um die Offline-Optimierung zu ersetzen. Jede Schicht kann unabhängig trainiert, schrittweise aktualisiert werden und unterstützt die Online-Neuplanung.

---

*Referenzen (in der Reihenfolge der Zitierung im Text)*

1. van den Berg et al., „Reciprocal Collision Vermeidung für mehrere mobile Roboter“, IJRR, 2011
2. Rashid et al., „QMIX: Monotonic Value Function Factorization for Deep Multi-Agent Reinforcement Learning“, ICML, 2018
3. Yu et al., „The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games“, ICLR, 2022
4. Pooladsanj et al., „Throughput Maximizing Takeoff Scheduling for eVTOL Vehicles“, arXiv:2503.17313, 2025
5. Rigas et al., „Scheduling Aerial Vehicles in an Urban Air Mobility Scheme“, arXiv:2108.01608, 2021
6. Mu et al., „Embodied AI-Enhanced IoMT Edge Computing: UAV Trajectory Optimization“, arXiv:2512.20902, 2025
7. Zhou et al., „OmniShow: Unifying Multimodal Conditions for Human-Object Interaction“, arXiv:2604.11804, 2026*Autor: Kagura Tart | 15.04.2026*