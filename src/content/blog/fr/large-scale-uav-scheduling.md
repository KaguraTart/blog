---
title: "Des centaines de machines volent ensemble : un examen complet de la méthodologie pour les problèmes de répartition de drones à grande échelle"
description: "De l’apprentissage par renforcement multi-agents aux réseaux de neurones graphiques, nous recherchons systématiquement des solutions aux problèmes de répartition de drones à grande échelle. Couvrant l'allocation globale des tâches au niveau macro (MARL/GNN/Attention), la coordination des conflits au niveau méso (QMIX/MAPPO/GNN) et l'évitement d'obstacles en temps réel au niveau micro (MPC/ORCA), en abandonnant les méthodes hors ligne telles que la planification entière, en se concentrant sur des itinéraires d'apprentissage différenciables de bout en bout et en analysant les défis d'ingénierie réels dans les scénarios de trafic aérien urbain (UAM)."
tags: ["drone", "planification à grande échelle", "Apprentissage par renforcement multi-agents", "MARNE", "GNN", "UAM", "mobilité aérienne urbaine", "planification du chemin", "ORQUE", "mécanisme d'attention", "Attention"]
pubDate: 2026-04-15
---

# Des centaines d'engins volent ensemble : Une revue complète de la méthodologie pour les problèmes de dispatching de drones à grande échelle

## 1. Définition du problème : qu'est-ce que la « grande échelle » ?

Lorsque le nombre de drones passe de 1 à 100 ou 1 000, la nature du problème change qualitativement :

**Petite échelle (1 à 10 unités) :**
- Planification entièrement centralisée, le planificateur central calcule toutes les trajectoires à la fois
- Détection de conflits O(n²) + Planification globale A* / RRT*
- Exigences temps réel : deuxième niveau

**Échelle moyenne (10 à 100 avions) :**
- Gestion partiellement distribuée, régionale/hiérarchique par ordonnanceur central
- Les délais de communication et les protocoles de coordination doivent être pris en compte
- Exigences en temps réel : niveau inférieur à la seconde

**Grande échelle (100 à 1 000+ racks) :**
- La complexité informatique de la planification centralisée pure explose (NP-hard)
- La topologie de communication devient un goulot d'étranglement
- **Comportements émergents** : les effets globaux des décisions locales sont imprévisibles
- Exigences en temps réel : niveau milliseconde

Cet article se concentre sur la méthode **différentiable et apprenable** - qui ne repose pas sur une optimisation hors ligne telle que la programmation entière, mais utilise l'apprentissage par renforcement et les réseaux neuronaux graphiques pour résoudre des problèmes de planification à grande échelle.

## 2. Architecture à trois niveaux : paradigme standard pour la planification à grande échelle

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

L'**échelle de temps** et l'**échelle spatiale** à trois niveaux sont naturellement découplées et constituent la méthode hiérarchique la plus classique pour la planification à grande échelle.

## 3. Niveau macro : allocation globale des tâches

### 3.1 Pourquoi ne pas utiliser la programmation en nombres entiers (MILP) ?

MILP est une méthode classique dans le monde universitaire, mais elle présente trois problèmes fatals dans les scénarios réels de drones à grande échelle :

1. **Le calcul n'est pas évolutif** : 50 tâches + 20 drones, c'est déjà la limite du MILP ; le temps de solution pour 1000 drones explose de quelques minutes à quelques heures.
2. **Impossible de mettre à jour en ligne** : MILP doit être résolu à chaque fois qu'il est replanifié, et l'insertion dynamique de tâches est extrêmement coûteuse.
3. **Difficile de gérer le hasard** : des facteurs dynamiques tels que les changements de vitesse du vent, les annulations de missions, les pannes de drones, etc. détruiront les hypothèses de contraintes du MILP

Par conséquent, l'itinéraire correct au niveau macro est la **méthode d'apprentissage** : former un réseau de politiques, saisir l'état de la tâche et du drone, et **une propagation vers l'avant** génère les résultats de l'attribution des tâches.

### 3.2 Réseau d'allocation de tâches basé sur l'attention

La méthode d'apprentissage la plus intuitive consiste à **utiliser le mécanisme Attention pour l'allocation des tâches** - c'est la même idée que la correspondance de jetons dans LLM :

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
```**Pourquoi Attention fonctionne-t-il ? **
- L'attention personnelle capture les dépendances entre les tâches (par exemple, deux tâches sont très proches et doivent être assignées au même drone)
- L'attention croisée capture l'adéquation tâche-drone
- La sortie est O(1) parallèle et n'évolue pas avec le nombre de tâches (car la multiplication matricielle peut être parallélisée par le GPU)

### 3.3 Réseau neuronal graphique (GNN) pour l'attribution des tâches

GNN est une autre méthode puissante, particulièrement adaptée aux scénarios avec une topologie évidente :

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

**Principaux avantages de GNN :**
- Peut gérer des graphiques de n'importe quelle taille et utiliser des graphiques à petite échelle pour généraliser à des graphiques à grande échelle pendant la formation
- Les fonctionnalités sur les bords peuvent coder des contraintes (distance, compatibilité des fenêtres temporelles)
- Le mécanisme de messagerie gère naturellement la parcimonie - chaque drone n'a besoin que de regarder les tâches à proximité

### 3.4 Algorithme évolutif : optimisation des colonies de fourmis ACO

L'algorithme de colonie de fourmis (ACO) est une méthode d'apprentissage qui ne nécessite pas de gradients et convient aux problèmes d'optimisation combinatoire tels que l'allocation de tâches :

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

L’avantage d’ACO est qu’il ne nécessite pas de formation et peut être exécuté directement ; l'inconvénient est qu'il faut le réitérer à chaque fois qu'il est replanifié, et il existe un problème de retard dans les scénarios en ligne.

### 3.5 RL de bout en bout : apprentissage macro-micro unifié

L'idée la plus avant-gardiste est de supprimer la prise de décision discrète au niveau macro et d'utiliser directement RL pour générer des trajectoires continues** :

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

L’avantage de cette approche de bout en bout est qu’elle connecte de manière transparente macro et micro. L’inconvénient est que la formation est difficile – elle nécessite une conception minutieuse des stratégies de formation des récompenses et d’apprentissage des cours.

## 4. Couche méso : coordination des conflits multi-machines

### 4.1 Pourquoi la coordination locale est-elle suffisante ?

La planification globale ne nécessite pas une coordination précise de chaque paire de drones, elle doit seulement assurer la coordination au sein de la fenêtre de conflit locale :

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

### 4.2 MARL : apprentissage par renforcement multi-agents coordonné et distribué

MARL est l'outil le plus puissant lorsque le nombre de drones dans la fenêtre de conflit est compris entre 10 et 50.

**Défi principal : non-stationnarité**

Lorsque l’agent A est formé, la stratégie de l’agent B change également, ce qui entraîne une dérive continue de la stratégie « optimale » de A – c’est un problème fondamental en formation.

**Solution 1 : QMIX - Formation centralisée + exécution distribuée (CTDE)**

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
```**Solution 2 : MAPPO – Approche multi-agents basée sur PPO**

MAPPO (Multi-Agent PPO) est l'un des meilleurs algorithmes MARL pour les tâches de coordination des drones de ces dernières années :

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

### 4.3 Communication implicite basée sur GNN

Les GNN peuvent transmettre des informations sans communication explicite - l'entrée réseau politique pour chaque drone est la fonctionnalité du nœud graphique, agrégeant naturellement les informations des voisins via la transmission de messages :

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

Ce type de communication implicite est plus robuste que la communication explicite : aucune hypothèse sur les modèles de délai de communication et de perte de paquets n'est requise.

## 5. Micro couche : évitement d'obstacles en temps réel

### 5.1 ORCA (évitement optimal des collisions réciproques)

ORCA est actuellement l’algorithme d’évitement d’obstacles en temps réel le plus largement implémenté :

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

**Avantages ORCA :** Entièrement distribué, complexité O(n), garantie théorique d'évitement des collisions.

**Limites d'ORCA :** Les contraintes dynamiques ne sont pas prises en compte (la "vitesse optimale" générée peut ne pas être captée par le drone).

### 5.2 Combinaison MPC + ORCA

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

## 6. Dernières recherches sur les scénarios de mobilité aérienne urbaine (UAM)

**Débit maximisant la planification du décollage pour eVTOL (arXiv:2503.17313)**

Découvrez comment maximiser le débit eVTOL dans les scénarios UAM :

- Aperçu clé : la planification du décollage est le goulot d'étranglement du débit
- Méthode : Stratégie de planification d'apprentissage en ligne, optimisation dynamique de la séquence de décollage et de la fenêtre horaire
- Résultat : A une demande de 1000 sorties/h, la stratégie de planification réduit le taux de retard de 23% à 4%

** Planification des véhicules aériens dans UAM (arXiv: 2108.01608) **

- Modéliser le problème d'ordonnancement UAM comme un problème de plus court chemin contraint
- Proposer une méthode heuristique basée sur les données qui peut gérer les zones d'exclusion aérienne, les retards météorologiques et les contraintes de capacité aéroportuaire
- Coordination vérifiée en temps réel de plus de 500 drones dans un environnement de simulation**Computing de périphérie IoMT amélioré par l'IA : optimisation de la trajectoire des drones (arXiv :2512.20902)**

- Problème d'optimisation conjoint UAV + edge computing
-Utiliser l'apprentissage par renforcement pour optimiser les trajectoires des drones + les décisions de déchargement des tâches
- Tient compte des incertitudes de prédiction de canal et de mobilité variant dans le temps

## 7. Stratégie de formation : Comment faire converger MARL dans des scénarios à grande échelle

La formation de MARL a toujours été difficile. Plus l’échelle est grande, plus il est difficile de converger. Voici des techniques de formation éprouvées :

### 7.1 Apprentissage du programme

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

### 7.2 Techniques de normalisation

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

### 7.3 Expérience de partition de relecture

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

## 8. Défis actuels et questions ouvertes

### 8.1 Problèmes fondamentaux qui restent non résolus

1. **Évolutivité de la formation** : pour la formation MARL de plus de 1 000 drones, la mémoire GPU et la surcharge de communication sont des goulots d'étranglement
2. **Scénario d'échec de communication** : lorsque la communication entre les drones est interrompue, comment la coordination distribuée maintient-elle la cohérence ?
3. **Flotte hétérogène** : Programmation mixte de drones avec des caractéristiques dynamiques différentes (gros drone + petit drone)
4. **Insertion dynamique de tâches** : insérez de nouvelles tâches (telles que la livraison de matériel d'urgence) pendant le vol. Comment replanifier en temps réel sans planter ?
5. **Robustesse contradictoire** : planification sécurisée en cas d'attaques malveillantes de drones ou d'usurpation d'identité GPS

### 8.2 Orientations de la recherche exploratoire

- **LLM comme planificateur** : utilisation de grands modèles pour la compréhension des tâches + coordination du langage naturel (voir l'article sur le VLM hiérarchique)
- **World Model + Scheduling** : utilisez des modèles génératifs pour prédire les flux de trafic futurs et éviter les conflits à l'avance
- **Neural Symbolic Scheduling** : un système hybride combinant réseaux de neurones (perception) et planification symbolique (raisonnement logique)
- **Apprentissage de récompense multi-objectif** : utilisez l'apprentissage des préférences pour ajuster automatiquement le compromis sécurité/efficacité

## 9. Résumé

La planification des drones à grande échelle est un problème naturellement complexe :- **Couche macro** (qui va où) → Attention / GNN / ACO / RL de bout en bout
- **Mesolayer** (coordination locale) → communication implicite QMIX / MAPPO / GNN
- **Micro Layer** (évitement d'obstacles en temps réel) → ORCA/MPC

**Idée principale** : utilisez des méthodes apprenables pour remplacer l'optimisation hors ligne. Chaque couche peut être formée indépendamment, mise à jour progressivement et prend en charge la replanification en ligne.

---

*Références (par ordre de citation dans le texte)*

1. van den Berg et al., « Évitement des collisions réciproques pour plusieurs robots mobiles », IJRR, 2011
2. Rashid et al., « QMIX : Factorisation de fonction de valeur monotonique pour un apprentissage par renforcement multi-agents profond », ICML, 2018
3. Yu et al., « L'efficacité surprenante du PPO dans les jeux multi-agents coopératifs », ICLR, 2022
4. Pooladsanj et al., « Planification de décollage maximisant le débit pour les véhicules eVTOL », arXiv : 2503.17313, 2025
5. Rigas et al., « Planification des véhicules aériens dans un programme de mobilité aérienne urbaine », arXiv :2108.01608, 2021
6. Mu et al., « Computing de bord IoMT amélioré par l'IA : optimisation de la trajectoire des drones », arXiv : 2512.20902, 2025
7. Zhou et al., « OmniShow : Unifier les conditions multimodales pour l'interaction homme-objet », arXiv :2604.11804, 2026*Auteur : Tarte Kagura | 2026-04-15*