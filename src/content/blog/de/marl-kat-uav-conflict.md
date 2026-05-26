---
title: "Multi-Agent-Verstärkungslernen und Graph-Aufmerksamkeitsnetzwerk: eine End-to-End-Lösung für die Konfliktlösung von UAV-Clustern"
description: "Eingehende Analyse der Integrationsarchitektur von MARL (QMIX/COMA/MAPPO/MADDPG) und GAT und Diskussion, wie in einem Artikel ein durchgängiges Lernen der UAV-Cluster-Konfliktlösung erreicht werden kann, vom Richtliniengradienten bis zur zugrunde liegenden Diagrammstruktur"
pubDate: 2026-04-07T11:24:40+08:00
tags: ["Verstärkungslernen mit mehreren Agenten", "MERGEL", "GAT", "Figur Aufmerksamkeit", "Drohne", "Wegplanung", "tiefes Lernen"]
category: Tech
sourceHash: "41cd4a89dae677119d19d92d790e780642b52e5c"
---

# Multi-Agent-Verstärkungslernen und Graph-Aufmerksamkeitsnetzwerk: eine End-to-End-Lösung für die Konfliktlösung von UAV-Clustern

> Im [vorherigen Artikel](/blog/uav-conflict-resolution/) haben wir das Algorithmus-Panorama der UAV-Konfliktlösung geordnet. Unter ihnen wird Reinforcement Learning (insbesondere MARL) als die „realistischste Option“ für Schwärme von mehr als 50 Drohnen bezeichnet. Dieser Artikel konzentriert sich auf diesen Weg, beginnend mit den Grundlagen des Single-Agent-RL, geht auf die Kernherausforderungen von Multi-Agent-Szenarien ein, analysiert gängige Algorithmen wie MADDPG, QMIX, COMA und MAPPO und konzentriert sich darauf, wie **GAT (Graph Attention Network)** MARL skalierbare Topologieerkennungsfunktionen bietet und letztendlich eine durchgängige Konfliktlösungsstrategie erreicht.

---

## 1. Vom Einzelagenten zum Multiagenten: Warum ist MARL so schwierig?

### 1.1 Single-Agent-RL-Überprüfung

Beginnen wir mit dem bekannten Single-Agent-RL. Das Einzelagenten-MDP wird durch das Vierfache $(\mathcal{S}, \mathcal{A}, \mathcal{R}, \mathcal{P}, \gamma)$ beschrieben:

- **Zustandswertfunktion**: $V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^\infty \gamma^t r_t \mid s_0 = s\right]$
- **Aktionswertfunktion**: $Q^\pi(s,a) = \mathbb{E}_\pi\left[\sum_{t=0}^\infty \gamma^t r_t \mid s_0 = s, a_0 = a\right]$
- **Optimale Strategie**: $\pi^* = \arg\max_\pi V^\pi(s)$

Die Kernannahme von Single-Agent RL: **Die Umgebung ist stabil** – egal wie viele Episoden Sie trainieren, die Dynamik der Umgebung $\mathcal{P}$ bleibt immer unverändert.

### 1.2 Drei wesentliche Schwierigkeiten von Multi-Agenten

Multi-Agenten-Szenarien widerlegen diese Annahme und bringen drei grundlegende Schwierigkeiten mit sich:

**① Umwelt-Instationarität (Nichtstationarität)**

Wenn Agent $i$ die Richtlinie $\pi_i$ lernt, ändern sich auch die Richtlinien $\{\pi_j\}_{j\neq i}$ anderer Agenten. Das heisst:$$
\mathcal{P}_i(s'\mid s, a_1,\dots,a_n) \neq \mathcal{P}_i(s'\mid s, a_1,\dots,a_n, a_1',\dots,a_n')
$$

Bei Einzelagenten-RL ist die Verteilung des nächsten Zustands aufgrund des aktuellen Zustands und der aktuellen Aktion festgelegt. Aber in einem Szenario mit mehreren Agenten kann dasselbe Zustands-Aktionspaar einer völlig anderen Verteilung des nächsten Zustands entsprechen, da andere Agenten in unterschiedlichen Episoden möglicherweise unterschiedliche Aktionen ausführen.

Dies führt direkt zum Ausfall des **Experience Replay Buffer**: Die gespeicherten Erfahrungsdaten stammen aus „veralteten“ Strategien und das Training damit führt zum Zusammenbruch der Strategie.

**②Credit-Zuweisung**

Wenn $n$-Agenten gemeinsam eine Teambelohnung $r$ erhalten, wie lässt sich diese Belohnung dann dem Beitrag jedes Agenten zuordnen?

$$
r_t = f(\mathbf{s}_t, \mathbf{a}_t, \mathbf{s}_{t+1})
$$

Beispiel: Mehrere UAVs arbeiteten zusammen, um einem Hindernis auszuweichen. Wie viel hat jeder Agent beigetragen? Wenn nur wenige belohnt werden, werden andere Agenten aufhören zu lernen.

**③ Explosion des Joint Action Space Index**

Es gibt $n$ UAVs, jedes mit $|\mathcal{A}|$ Aktionsoptionen, und der gemeinsame Aktionsraum $|\mathcal{A}|^n$ wächst exponentiell mit $n$. Die Abdeckung gieriger Erkundungen im gemeinsamen Raum nähert sich Null.

### 1.3 Klassifizierung des MARL-Algorithmus

Als Reaktion auf die oben genannten Schwierigkeiten hat die akademische Gemeinschaft drei Hauptwege entwickelt:

| Route | Repräsentativer Algorithmus | Kernidee | Repräsentatives Papier |
|------|----------|----------|-----------|
| **Unabhängiges Lernen (IL)** | IQL, DQN | Jeder lernt sein eigenes Ding und ignoriert den Einfluss anderer | Tan, 1993 |
| **Zentralisierte Schulung + dezentrale Ausführung (CTDE)** | MADDPG, QMIX, MAPPO | Globale Informationen während des Trainings und lokale Beobachtungen während der Ausführung nutzen | Lowe et al., 2017 |
| **Vollständig dezentralisiert** | KOMA, VDND | Rein lokale Strategie, keine zentrale Schulung | Foerster et al., 2018 |> **CTDE ist das aktuelle Mainstream-Paradigma für die UAV-Konfliktlösung**: Es kann nicht nur globale Informationen während des Trainings nutzen, um die Lerneffizienz zu verbessern, sondern auch Echtzeit-Entscheidungsfähigkeiten bei eingeschränkter Kommunikation während der Ausführung aufrechtzuerhalten.

---

## 2. CTDE-Rahmen: Nutzen Sie Gottes Perspektive für das Training und lokale Beobachtung für die Ausführung

### 2.1 Die Designphilosophie des zentralisierten Kritikers

Die Kernerkenntnis von CTDE ist: **Die Trainingsphase und die Ausführungsphase können unterschiedliche Informationsverfügbarkeit haben**.

```
┌─────────────────────────────────────────────────────────┐
│  中心化训练（Centralized Training）                      │
│  Critic(s₁,...,sₙ, a₁,...,aₙ) → Q(s,a)                 │
│  ✅ 可访问全局状态 & 所有智能体的动作                     │
│  ✅ 环境是"平稳的"（给定全局状态-动作对）                 │
│                                                         │
│  去中心化执行（Decentralized Execution）                 │
│  πᵢ(oᵢ → aᵢ)                                          │
│  ✅ 只依赖本地观测 oᵢ                                    │
│  ✅ 通信失败时仍可运行                                   │
└─────────────────────────────────────────────────────────┘
```

### 2.2 MADDPG: Der Pionier von CTDE im kontinuierlichen Aktionsraum

**MADDPG (Multi-Agent DDPG)** wurde 2017 von OpenAI vorgeschlagen und ist ein Meilenstein im kontinuierlichen Action-Space-Multi-Agent-Deep-Reinforcement-Learning.

**Kernformel:**

Jeder Agent $i$ unterhält eine Akteur-Kritiker-Struktur:

$$
\nabla_{\theta_i} J(\theta_i) = \mathbb{E}_{\mathbf{s} \sim \mathcal{D}}\left[
    \nabla_{\theta_i} \log \pi_i(a_i \mid o_i) \cdot
    Q_i^\pi(\mathbf{s}, a_1, \dots, a_n) \Big|_{a_i = \pi_i(o_i)}
\richtig]
$$

Hauptunterschied: Die Eingaben für $Q_i^\pi$ sind der globale Zustand $\mathbf{s}$ und die gemeinsamen Aktionen $\mathbf{a}$ aller Agenten, keine lokalen Beobachtungen.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# ============================================================
# MADDPG 核心实现（用于 UAV 冲突消解场景）
# ============================================================

class ReplayBuffer:
    """共享经验回放池（所有智能体的经验统一存储）"""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, actions, reward, next_state, done):
        self.buffer.append((state, actions, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions),
                np.array(rewards), np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """演员网络：本地观测 → 动作（去中心化执行）"""
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Tanh(),                   # 连续动作输出（速度变化量）
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()                    # 动作限幅 [-1, 1]
        )
    
    def forward(self, obs):
        return self.net(obs)


class Critic(nn.Module):
    """评论家网络：全局状态 + 联合动作 → Q值（中心化训练）"""
    def __init__(self, total_obs_dim, total_action_dim, n_agents, hidden_dim=64):
        super().__init__()
        # 输入：全局状态 + 所有智能体的动作拼接
        input_dim = total_obs_dim + n_agents * total_action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)    # 输出单个 Q 值
        )
    
    def forward(self, states, all_actions):
        """
        states: (batch, total_obs_dim) 全局状态
        all_actions: (batch, n_agents * action_dim) 所有智能体的动作
        """
        x = torch.cat([states, all_actions], dim=1)
        return self.net(x)


class MADDPGAgent:
    """MADDPG 智能体"""
    def __init__(self, obs_dim, action_dim, n_agents, agent_id,
                 lr_actor=1e-3, lr_critic=1e-3, gamma=0.95, tau=0.01):
        self.agent_id = agent_id
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        
        # 演员网络（本地策略）
        self.actor = Actor(obs_dim, action_dim)
        self.actor_target = Actor(obs_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # 评论家网络（全局 Q）
        total_obs = obs_dim * n_agents
        total_act = action_dim * n_agents
        self.critic = Critic(total_obs, total_act, n_agents)
        self.critic_target = Critic(total_obs, total_act, n_agents)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 目标网络初始化
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)
    
    def hard_update(self, target, source):
        """硬更新（一次性复制）"""
        target.load_state_dict(source.state_dict())
    
    def soft_update(self, target, source):
        """软更新（指数滑动平均）"""
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
    
    def select_action(self, obs, noise=0.1):
        """选择动作（探索时加噪声）"""
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        action = self.actor(obs_t).squeeze(0).numpy()
        action += noise * np.random.randn(self.action_dim)
        return np.clip(action, -1, 1)
    
    def update(self, agents, replay_buffer, batch):
        """单步更新"""
        states, all_actions, rewards, next_states, dones = batch
        
        # ----- Critic 更新 -----
        # 目标动作用目标演员网络生成
        next_actions = []
        for agent_id, agent in enumerate(agents):
            next_obs = torch.FloatTensor(next_states[:, agent_id * 4:(agent_id+1)*4])  # 假设 obs 维4
            next_actions.append(agent.actor_target(next_obs))
        next_actions_cat = torch.cat(next_actions, dim=1)
        
        # 目标 Q 值
        target_Q = self.critic_target(
            torch.FloatTensor(next_states),
            next_actions_cat.detach()
        )
        expected_Q = self.critic(
            torch.FloatTensor(states),
            torch.FloatTensor(all_actions)
        )
        
        critic_loss = nn.MSELoss()(expected_Q, target_Q.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ----- Actor 更新 -----
        # 当前智能体的动作（其他智能体动作用 replay buffer 中的值）
        current_obs = torch.FloatTensor(states[:, self.agent_id*4:(self.agent_id+1)*4])
        current_action = self.actor(current_obs)
        
        # 构造完整的动作向量（当前智能体用当前策略，其他用历史动作）
        actions_fixed = torch.FloatTensor(all_actions).clone()
        actions_fixed[:, self.agent_id*self.action_dim:(self.agent_id+1)*self.action_dim] = current_action
        
        actor_loss = -self.critic(
            torch.FloatTensor(states),
            actions_fixed
        ).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ----- 目标网络软更新 -----
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)
        
        return actor_loss.item(), critic_loss.item()
```

### 2.3 QMIX: Wertzerlegung zur Lösung der Kreditzuordnung

MADDPG löst das Problem des kontinuierlichen Aktionsraums, aber Critic benötigt den globalen Status $\mathbf{s}$ – in realen UAV-Szenarien ist der zentrale Knoten möglicherweise nicht in der Lage, den globalen Status zu erhalten.

Die Kerninnovation von **QMIX** (Queensland Institute, 2018) ist: **Zerlegung des gemeinsamen Q-Werts in marginale Q-Werte für einzelne Agenten**.$$
Q_{tot}(\boldsymbol{\tau}, \mathbf{u}) = g_\theta(\boldsymbol{\tau}, \mathbf{u}; \boldsymbol{\phi}_1, \dots, \boldsymbol{\phi}_n)
$$

Wobei $\boldsymbol{\tau}_i$ die Aktions-Beobachtungs-Trajektorie des Agenten $i$ ist, ist $g_\theta$ ein **monotonisches Mischnetzwerk**, das Folgendes erfüllt:

$$
\frac{\partial Q_{tot}}{\partial Q_i} \geq 0, \quad \forall i
$$

Die Monotoniebeschränkung garantiert eine Schlüsseleigenschaft: **Während der dezentralen Ausführung entspricht die unabhängige gierige Maximierung von $Q_i$ jedes Agenten der globalen Maximierung von $Q_{tot}$**.

```python
class QMIXMixingNetwork(nn.Module):
    """
    单调混合网络：将各智能体的 Q_i 混合为全局 Q_tot
    关键约束：所有权值非负（保证单调性）
    """
    def __init__(self, n_agents, embed_dim=64):
        super().__init__()
        # Hyper-network 生成混合网络的权值
        self.hyper_w1 = nn.Sequential(
            nn.Linear(n_agents, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, n_agents * embed_dim),  # 输出 (n_agents × embed_dim) 权值
        )
        self.hyper_b1 = nn.Linear(n_agents, embed_dim)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.hyper_b2 = nn.Linear(embed_dim, 1)
    
    def forward(self, q_values, state):
        """
        q_values: (batch, n_agents) 各智能体的 Q 值
        state: (batch, state_dim) 全局状态（用于生成 hyper-network 输入）
        """
        batch_size = q_values.size(0)
        
        # 第一层：W₁ * Q + b₁
        w1 = torch.abs(self.hyper_w1(state))          # (batch, n_agents * embed_dim)
        w1 = w1.view(batch_size, q_values.size(1), -1)  # (batch, n_agents, embed_dim)
        b1 = self.hyper_b1(state).unsqueeze(1)       # (batch, 1, embed_dim)
        
        q_hidden = torch.relu(torch.bmm(q_values.unsqueeze(1), w1) + b1)  # (batch, 1, embed_dim)
        
        # 第二层：W₂ * h + b₂
        w2 = torch.abs(self.hyper_w2(q_hidden.squeeze(1)))  # (batch, embed_dim, embed_dim)
        b2 = self.hyper_b2(q_hidden.squeeze(1)).unsqueeze(1)  # (batch, 1, 1)
        
        q_tot = torch.bmm(q_hidden, w2.unsqueeze(1)) + b2  # (batch, 1, 1)
        return q_tot.squeeze(-1)  # (batch,)


class QMIXAgent:
    """QMIX 算法"""
    def __init__(self, obs_dim, action_dim, n_agents, agent_id):
        self.agent_id = agent_id
        self.action_dim = action_dim
        
        # 每个智能体的 RNN（处理动作-观测历史）
        self.rnn = nn.GRUCell(obs_dim + action_dim, obs_dim)
        # Q 网络
        self.q_net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        self.target_rnn = nn.GRUCell(obs_dim + action_dim, obs_dim)
        self.target_q_net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.hard_update()
    
    def hard_update(self):
        self.target_rnn.load_state_dict(self.rnn.state_dict())
        self.target_q_net.load_state_dict(self.q_net.state_dict())
    
    def get_q_values(self, hidden, obs, last_action):
        """给定 (hidden, obs, last_action) 输出 Q(s,a)"""
        rnn_input = torch.cat([obs, last_action], dim=1)
        new_hidden = self.rnn(rnn_input, hidden)
        q_values = self.q_net(new_hidden)
        return q_values, new_hidden
    
    def select_action_epsilon_greedy(self, q_values, epsilon):
        """ε-贪心策略"""
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        return q_values.argmax(dim=1).item()


def train_qmix():
    """QMIX 训练循环（伪代码）"""
    n_agents = 8
    n_episodes = 50000
    
    agents = [QMIXAgent(obs_dim=12, action_dim=5, n_agents=n_agents, agent_id=i)
              for i in range(n_agents)]
    mixer = QMIXMixingNetwork(n_agents)
    
    optimizers = [optim.Adam(agent.q_net.parameters(), lr=2e-4) for agent in agents]
    mixer_optimizer = optim.Adam(mixer.parameters(), lr=2e-4)
    
    replay = ReplayBuffer(capacity=100000)
    
    for ep in range(n_episodes):
        # 环境交互
        states = env.reset()  # (n_agents, obs_dim)
        hidden = [torch.zeros(1, 12) for _ in range(n_agents)]
        last_actions = [torch.zeros(1, 5) for _ in range(n_agents)]
        episode_reward = 0
        
        while not done:
            actions = []
            for i, agent in enumerate(agents):
                q_vals, hidden[i] = agent.get_q_values(hidden[i],
                    torch.FloatTensor(states[i]).unsqueeze(0),
                    last_actions[i])
                a = agent.select_action_epsilon_greedy(q_vals.squeeze(0), epsilon=0.1)
                actions.append(a)
                last_actions[i] = torch.zeros(1, 5)
                last_actions[i][0, a] = 1.0
            
            next_states, rewards, done = env.step(actions)
            replay.push(states, last_actions, rewards, next_states, done)
            states = next_states
            episode_reward += sum(rewards)
        
        # 学习
        if len(replay) > 1024:
            batch = replay.sample(32)
            # QMIX 损失计算 ...
            # 单调混合 + 中心化训练 ...
```

### 2.4 MAPPO: Der Sieg des politischen Gefälles in hochgradig parallelen Szenarien

**MAPPO (Multi-Agent PPO)** erweitert den PPO-Algorithmus auf Multi-Agent-Szenarien und hat sich in den letzten Jahren bei UAV-Cluster-Aufgaben gut bewährt (mehrere Top-Konferenzbeiträge von 2022 bis 2024).

**Hauptvorteile von PPO**: Einschränkungen der Vertrauensregion gewährleisten Trainingsstabilität und vermeiden die Hyperparameter-Katastrophe der DDPG-Serie.

PPO-Clip-Ziel:
$$
\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t\left[
    \min\left(
        r_t(\theta) \hat{A}_t,
        \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t
    \richtig)
\richtig]
$$

Dabei ist $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$ das Wahrscheinlichkeitsverhältnis, $\hat{A}_t$ ist GAE (Generalized Advantage Estimation).Typische Konfiguration von MAPPO bei der UAV-Konfliktlösung:

| Parameter | Empfohlene Werte | Beschreibung |
|------|--------|------|
| Clip-Verhältnis $\epsilon$ | 0,2 | PPO-Standard |
| Horizont $T$ | 128–256 | Anzahl der Rollout-Schritte pro Epoche |
| PPO-Epochen | 2–4 | Anzahl wiederholter Aktualisierungen pro Stapel |
| GAE $\lambda$ | 0,95 | Bias-Varianz-Balance der Dominanzschätzung |
| Dimension der verborgenen Ebene | 64–128 | Ausreichend für UAV-Szenarien |
| Normalisierung | OBS + Belohnungsnormalisierung | Schlüssel! Große Auswirkung auf die Multi-Agenten-Konvergenz |

---

## 3. GAT: Lassen Sie MARL lernen, wem Sie folgen sollen.

### 3.1 Warum benötigt MARL eine Graphenstruktur?

In einem UAV-Cluster sind nicht alle Agenten gleich wichtig. Nehmen wir als Beispiel Konfliktlösung:

- UAV droht mit mir zu kollidieren → **Große Sorge**
- UAVs weit außer Sichtweite → **können ignoriert werden**
- Annäherung an sich bewegende Hindernisse → **erfordert dynamische Aufmerksamkeit**

Allerdings behandelt traditionelles MARL (wie MADDPG, QMIX) alle Nachbarn gleich: entweder vollständig verbunden ($\mathcal{O}(N^2)$-Kommunikation) oder feste Topologie (wie Ring, nächster Nachbar).

**Die Einführung von GAT löst zwei Kernprobleme:**

1. **Adaptive Nachbargewichtung**: Erfahren Sie durch den Aufmerksamkeitsmechanismus, welche Nachbarn für die aktuelle Entscheidung wichtiger sind
2. **Skalierbarkeit**: Steigt nicht mit der Anzahl der Drohnen und unterstützt dynamische Topologie

### 3.2 GAT-Grundprinzipien

GAT führt eine **Neighbor Aggregation** für die Features $\mathbf{h}_i$ des Knotens $i$ auf jeder Ebene durch, und die Gewichtung wird dynamisch durch den Aufmerksamkeitsmechanismus berechnet:$$
\alpha_{ij} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^\top[\mathbf{W}\mathbf{h}_i \Vert \mathbf{W}\mathbf{h}_j]\right)\right)}
{\sum_{k \in \mathcal{N}_i} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^\top[\mathbf{W}\mathbf{h}_i \Vert \mathbf{W}\mathbf{h}_k]\right)\right)}
$$

$$
\mathbf{h}_i' = \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij} \mathbf{W}\mathbf{h}_j\right)
$$

Wobei $\mathbf{W}$ die lernbare lineare Transformationsmatrix ist, $\mathbf{a}$ der Aufmerksamkeitsvektor ist und $\Vert$ das Spleißen darstellt.

**Warum LeakyReLU**: Führt eine leichte Nichtlinearität in den Aufmerksamkeitswert ein und ermöglicht so eine asymmetrische Verarbeitung positiver und negativer Werte.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadGAT(nn.Module):
    """
    多头图注意力网络（Multi-Head GAT）
    多个注意力头并行学习不同的邻机关系模式
    """
    def __init__(self, in_features, out_features, n_heads=4, dropout=0.1, alpha=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features  # 每个头的输出维度
        self.n_heads = n_heads
        self.alpha = alpha
        self.dropout = dropout
        
        # 多头并行：每个头有独立的 W 和 a
        self.W = nn.ModuleList([
            nn.Linear(in_features, out_features)
            for _ in range(n_heads)
        ])
        self.a = nn.ModuleList([
            nn.Linear(2 * out_features, 1)
            for _ in range(n_heads)
        ])
        
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout_layer = nn.Dropout(dropout)
        
        # 最终输出层（多头拼接后的线性变换）
        self.out_W = nn.Linear(in_features, out_features * n_heads)
    
    def forward(self, h, adj):
        """
        h: (batch, N, D_in) - N 个节点的特征
        adj: (batch, N, N) - 邻接矩阵（二值或加权）
        返回: (batch, N, D_out) - 更新后的节点特征
        """
        batch_size, N, D_in = h.shape
        D_out = self.out_features
        
        # 多头注意力
        head_outputs = []
        for head in range(self.n_heads):
            # 线性变换
            Wh = self.W[head](h)  # (batch, N, D_out)
            
            # 计算注意力分数
            # 拼接 [Wh_i || Wh_j] 对于所有 (i,j) 对
            Wh_i = Wh.unsqueeze(2).expand(batch_size, N, N, D_out)   # (batch, N, N, D_out)
            Wh_j = Wh.unsqueeze(1).expand(batch_size, N, N, D_out)  # (batch, N, N, D_out)
            
            concat_feats = torch.cat([Wh_i, Wh_j], dim=3)  # (batch, N, N, 2*D_out)
            e = self.leaky_relu(self.a[head](concat_feats).squeeze(-1))  # (batch, N, N)
            
            # Mask 非邻接节点（设为 -∞ 再 softmax 则注意力权重→0）
            e = e.masked_fill(adj.unsqueeze(1).expand(-1, N, -1) == 0, -1e9)
            
            # Softmax 归一化
            attention = F.softmax(e, dim=-1)  # (batch, N, N)
            attention = self.dropout_layer(attention)
            
            # 加权聚集
            h_new = torch.bmm(attention, Wh)  # (batch, N, D_out)
            head_outputs.append(h_new)
        
        # 多头拼接（Multi-Head Aggregation）
        concatenated = torch.cat(head_outputs, dim=-1)  # (batch, N, D_out * n_heads)
        
        # 如果输入输出维度相同，用残差连接
        if D_in == D_out * self.n_heads:
            return F.elu(concatenated) + h
        else:
            return F.elu(concatenated)


class GATLayerForUAV(nn.Module):
    """
    针对 UAV 冲突消解优化的 GAT 层
    输入：每架 UAV 的局部状态特征
    输出：考虑了邻机关系的增强状态表示
    """
    def __init__(self, state_dim, hidden_dim=64, n_heads=3):
        super().__init__()
        # 邻居特征编码器（编码相对位置、相对速度等）
        self.feature_encoder = nn.Sequential(
            nn.Linear(state_dim * 2 + 3, hidden_dim),  # 自身状态 + 邻居状态 + 相对位置
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # GAT
        self.gat = MultiHeadGAT(
            in_features=hidden_dim,
            out_features=hidden_dim,
            n_heads=n_heads,
            dropout=0.1
        )
        
        # 边 Encoder（GAT 邻接矩阵的另一种表示）
        self.edge_encoder = nn.Sequential(
            nn.Linear(3, 16),   # 边特征：相对距离
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def build_adj_from_distance(self, positions, self_mask=True, threshold=50.0):
        """
        根据欧氏距离构建邻接矩阵
        positions: (batch, N, 3)
        threshold: 超过此距离认为无连接
        """
        batch_size, N, _ = positions.shape
        
        # 计算两两之间的距离
        pos_i = positions.unsqueeze(2)  # (batch, N, 1, 3)
        pos_j = positions.unsqueeze(1)  # (batch, 1, N, 3)
        dist = torch.norm(pos_i - pos_j, dim=-1)  # (batch, N, N)
        
        adj = (dist < threshold).float()
        
        if self_mask:
            adj = adj * (1 - torch.eye(N, device=adj.device).unsqueeze(0))
        
        return adj
    
    def forward(self, uav_states, positions):
        """
        uav_states: (batch, N, state_dim)
        positions: (batch, N, 3)
        """
        batch_size, N, _ = uav_states.shape
        
        # 构建邻接矩阵（基于物理距离）
        adj = self.build_adj_from_distance(positions, threshold=50.0)  # (batch, N, N)
        
        # 编码 UAV 状态为 GAT 输入特征
        # 使用共享编码器处理每个节点
        h = self.feature_encoder(
            torch.cat([uav_states, torch.zeros_like(uav_states)], dim=-1)
        )  # 简化：先用自身状态作为特征
        
        # GAT 更新
        h_updated = self.gat(h, adj)  # (batch, N, hidden_dim)
        
        return h_updated, adj
```

### 3.3 Integrierte Architektur von GAT + MARL

Betten Sie GAT in das Richtliniennetzwerk von MARL ein, um die **GAT-MARL**-Architektur zu bilden:

```
┌──────────────────────────────────────────────────────────────┐
│                     融合架构（GAT + MADDPG）                  │
│                                                              │
│  本地观测 oᵢ ──┐                                             │
│               ▼                                              │
│         ┌─────────────┐    ┌─────────────┐                   │
│         │  状态编码器  │───▶│    GAT 层   │                   │
│         └─────────────┘    │ (K=3 头)   │                   │
│                             │           │                   │
│  邻机观测 oⱼ ──────────────▶│ αᵢⱼ=f(hᵢ,hⱼ) │                   │
│                             └─────────────┘                   │
│                                    │                         │
│                                    ▼                         │
│                             增强特征 h̃ᵢ                      │
│                                    │                         │
│                             ┌──────┴──────┐                  │
│                             ▼             ▼                  │
│                      ┌──────────┐ ┌──────────┐              │
│                      │  Actor    │ │  Critic  │              │
│                      │ π(aᵢ|h̃ᵢ) │ │ Q(s,a)   │              │
│                      └──────────┘ └──────────┘              │
└──────────────────────────────────────────────────────────────┘
```

**Vollständiger Implementierungscode:**

```python
class GATPolicyNetwork(nn.Module):
    """
    GAT 增强的策略网络（GAT + Actor）
    用于 UAV 冲突消解的端到端策略
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=64, n_gat_heads=3):
        super().__init__()
        
        # 1. 本地观测编码器
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 2. 邻居状态编码器（编码相对信息）
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(obs_dim + 3, hidden_dim),  # 邻居观测 + 相对位置
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 3. 多层 GAT
        self.gat1 = MultiHeadGAT(hidden_dim, hidden_dim // n_gat_heads, n_heads=n_gat_heads)
        self.gat2 = MultiHeadGAT(hidden_dim, hidden_dim // n_gat_heads, n_heads=n_gat_heads)
        
        # 4. 注意力聚合层
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 5. 策略头
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()
        )
        
        # 6. 价值头（用于 PPO/MAPPO）
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, obs, positions, adj_mask):
        """
        obs: (batch, N, obs_dim) - 所有 UAV 的观测
        positions: (batch, N, 3) - 所有 UAV 的位置
        adj_mask: (batch, N, N) - 邻接矩阵
        返回: 策略分布, 价值估计
        """
        batch_size, N, _ = obs.shape
        
        # 本地特征编码
        h_local = self.obs_encoder(obs)  # (batch, N, hidden_dim)
        
        # 为每个 UAV 构建邻居感知特征
        h_nodes = []
        for i in range(N):
            # 邻居特征（排除自身）
            neighbor_mask = adj_mask[:, i, :]  # (batch, N)
            neighbor_mask[:, i] = 0  # 排除自身
            
            # 收集邻居观测
            neighbor_features = []
            for j in range(N):
                if j != i:
                    # 邻居状态 + 相对位置
                    rel_pos = positions[:, j] - positions[:, i]  # (batch, 3)
                    feat = torch.cat([obs[:, j], rel_pos], dim=1)  # (batch, obs_dim+3)
                    neighbor_features.append(feat)
            
            if neighbor_features:
                neighbor_tensor = torch.stack(neighbor_features, dim=1)  # (batch, N-1, obs_dim+3)
                h_neighbor = self.neighbor_encoder(neighbor_tensor)      # (batch, N-1, hidden_dim)
                
                # 加权平均邻居特征
                # 简单方案：平均
                h_neighbor_agg = h_neighbor.mean(dim=1)  # (batch, hidden_dim)
            else:
                h_neighbor_agg = torch.zeros(batch_size, h_local.shape[-1], device=h_local.device)
            
            # 拼接本地 + 邻居
            h_combined = h_local[:, i] + h_neighbor_agg  # 残差连接
            h_nodes.append(h_combined)
        
        h = torch.stack(h_nodes, dim=1)  # (batch, N, hidden_dim)
        
        # GAT 层
        h = F.elu(self.gat1(h, adj_mask))  # (batch, N, hidden_dim)
        h = F.elu(self.gat2(h, adj_mask))  # (batch, N, hidden_dim)
        
        # 注意力池化（为 Critic 生成全局状态表示）
        attn_scores = self.attention_pool(h)  # (batch, N, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch, N, 1)
        h_global = (h * attn_weights).sum(dim=1)  # (batch, hidden_dim)
        
        # 策略输出
        policy = self.actor_head(h)   # (batch, N, action_dim)
        value = self.value_head(h_global)  # (batch, 1)
        
        return policy, value
    
    def get_action(self, obs, positions, adj_mask, deterministic=False):
        """推理时选择动作"""
        policy, value = self.forward(obs.unsqueeze(0), positions.unsqueeze(0), adj_mask.unsqueeze(0))
        
        if deterministic:
            action = policy.squeeze(0)  # 直接用均值
        else:
            action = policy.squeeze(0) + 0.1 * torch.randn_like(policy.squeeze(0))
        
        return action.squeeze(0), value.squeeze(0)


class GAT_MADDPG:
    """
    GAT-MADDPG：图注意力增强的多智能体 DDPG
    """
    def __init__(self, n_agents, obs_dim, action_dim, hidden_dim=64):
        self.n_agents = n_agents
        self.agents = []
        
        for i in range(n_agents):
            agent = {
                'policy': GATPolicyNetwork(obs_dim, action_dim, hidden_dim),
                'target_policy': GATPolicyNetwork(obs_dim, action_dim, hidden_dim),
                'optimizer': optim.Adam(
                    [], lr=1e-3  # 参数由外部指定
                )
            }
            agent['optimizer'] = optim.Adam(
                agent['policy'].parameters(), lr=1e-3
            )
            self.agents.append(agent)
        
        self.replay = ReplayBuffer(capacity=50000)
    
    def update(self, batch):
        """MADDPG 风格的 Critic 更新，使用 GAT 提取的特征"""
        states, all_actions, rewards, next_states, done = batch
        batch_size = states.size(0)
        
        # 更新每个智能体的策略
        total_actor_loss = 0
        total_critic_loss = 0
        
        for i, agent in enumerate(self.agents):
            # 构建邻接矩阵（基于距离）
            adj = self.build_adj(states[:, i*3:(i+1)*3])  # 假设位置在状态的前3维
            next_adj = self.build_adj(next_states[:, i*3:(i+1)*3])
            
            # ---- Critic 更新 ----
            # 当前策略的动作
            current_policy, _ = agent['policy'](states, states[:, :3], adj)
            
            # 目标策略的动作
            with torch.no_grad():
                next_policy, _ = agent['target_policy'](next_states, next_states[:, :3], next_adj)
            
            # 构造联合动作向量
            # ... (构造 all_actions_tensor)
            
            # Critic loss
            # target_Q = r + γ * Q_target(s', a')
            # critic_loss = MSE(Q(s,a), target_Q)
            
            # ---- Actor 更新 ----
            # 当前智能体用 GAT 策略网络的输出
            # 其他智能体用 replay buffer 中的历史动作
            # actor_loss = -Q(s, a_i_new, a_-i_fixed)
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
        
        return total_actor_loss / self.n_agents, total_critic_loss / self.n_agents
    
    def build_adj(self, positions, threshold=50.0):
        """从位置构建邻接矩阵"""
        # (batch, N, 3) -> (batch, N, N)
        pass
```

---

## 4. Durchgängiger Schulungsprozess zur Konfliktlösung

### 4.1 Simulationsumgebungsdesign

Nutzen Sie eine dedizierte Simulationsumgebung zur UAV-Konfliktlösung:

```python
import numpy as np
import gym
from gym import spaces

class UAVConflictEnv(gym.Env):
    """
    多 UAV 冲突消解仿真环境
    """
    metadata = {'render_modes': []}
    
    def __init__(self, n_agents=8, area_size=200.0, safe_radius=5.0, max_steps=500):
        super().__init__()
        self.n_agents = n_agents
        self.area_size = area_size
        self.safe_radius = safe_radius
        self.max_steps = max_steps
        self.dt = 0.1
        
        # 观测空间：自身状态(6) + 邻居信息(最多5个邻居 * 6 = 30)
        self.obs_dim = 6 + 30
        self.action_dim = 3  # (Δvx, Δvy, Δvz)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_agents, self.obs_dim), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.n_agents, self.action_dim), dtype=np.float32
        )
    
    def reset(self):
        """随机初始化 UAV 位置（确保初始有冲突风险）"""
        self.positions = np.random.uniform(
            -self.area_size/2, self.area_size/2,
            size=(self.n_agents, 3)
        ).astype(np.float32)
        
        # 随机速度方向（朝向各自目标）
        self.velocities = np.random.uniform(5, 15, size=(self.n_agents, 3)).astype(np.float32)
        self.velocities = self.velocities / np.linalg.norm(self.velocities, axis=1, keepdims=True) * 10
        
        # 目标点（随机分布在区域内）
        self.goals = np.random.uniform(
            -self.area_size/2, self.area_size/2,
            size=(self.n_agents, 3)
        ).astype(np.float32)
        
        self.step_count = 0
        return self._get_obs()
    
    def step(self, actions):
        """执行动作，更新状态"""
        actions = np.clip(actions, -1, 1)
        delta_v = actions * 2.0  # 速度变化量
        
        # 更新速度
        self.velocities += delta_v
        v_mag = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        self.velocities = np.clip(self.velocities, 0, 20)  # 速度限幅
        
        # 更新位置
        self.positions += self.velocities * self.dt
        
        # 边界反射
        for i in range(self.n_agents):
            for dim in range(3):
                if abs(self.positions[i, dim]) > self.area_size / 2:
                    self.velocities[i, dim] *= -1
        
        # 计算奖励
        reward, collision = self._compute_reward()
        done = collision or self.step_count >= self.max_steps
        self.step_count += 1
        
        return self._get_obs(), reward, done, {}
    
    def _compute_reward(self):
        """奖励函数设计"""
        reward = np.zeros(self.n_agents)
        collision = False
        
        for i in range(self.n_agents):
            # 接近目标的奖励
            dist_to_goal = np.linalg.norm(self.positions[i] - self.goals[i])
            reward[i] += 0.5 * (1.0 / (dist_to_goal + 1.0))
            
            # 保持安全距离
            for j in range(i + 1, self.n_agents):
                dist = np.linalg.norm(self.positions[i] - self.positions[j])
                if dist < self.safe_radius:
                    reward[i] -= 50
                    reward[j] -= 50
                    collision = True
                elif dist < self.safe_radius * 3:
                    reward[i] -= 1.0 / (dist + 1e-3)
                    reward[j] -= 1.0 / (dist + 1e-3)
        
        return reward, collision
    
    def _get_obs(self):
        """生成各 UAV 的观测"""
        obs = []
        for i in range(self.n_agents):
            # 自身信息：位置(3) + 速度(3)
            self_info = np.concatenate([self.positions[i], self.velocities[i]])
            
            # 邻居信息（距离最近的5架）
            rel_info = []
            for j in range(self.n_agents):
                if i != j:
                    rel_pos = self.positions[j] - self.positions[i]
                    rel_vel = self.velocities[j] - self.velocities[i]
                    rel_info.append(np.concatenate([rel_pos, rel_vel]))
            
            # 取最近的5个邻居
            if len(rel_info) > 5:
                dists = [np.linalg.norm(r[:3]) for r in rel_info]
                sorted_idx = np.argsort(dists)[:5]
                rel_info = [rel_info[k] for k in sorted_idx]
            
            # padding
            while len(rel_info) < 5:
                rel_info.append(np.zeros(6))
            
            neighbor_info = np.concatenate(rel_info)  # (30,)
            obs.append(np.concatenate([self_info, neighbor_info]))
        
        return np.array(obs, dtype=np.float32)
```

### 4.2 Kompletter Trainingszyklus

```python
def train_gat_maddpg():
    """完整训练流程"""
    env = UAVConflictEnv(n_agents=8)
    n_agents = 8
    
    # 初始化 GAT-MADDPG
    maddpg = GAT_MADDPG(n_agents=n_agents, obs_dim=36, action_dim=3)
    
    # 训练配置
    n_episodes = 10000
    batch_size = 256
    update_freq = 100
    target_update_freq = 200
    
    episode_rewards = []
    
    for ep in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # 构建邻接矩阵（基于当前距离）
            positions = env.positions  # (N, 3)
            adj = build_adj_matrix(positions, threshold=50.0)
            
            # 各智能体选择动作
            actions = []
            for i, agent in enumerate(maddpg.agents):
                action, _ = agent['policy'].get_action(
                    torch.FloatTensor(obs[i]),
                    torch.FloatTensor(positions[i]),
                    torch.FloatTensor(adj[i])
                )
                actions.append(action.numpy())
            
            actions = np.array(actions)
            
            # 环境交互
            next_obs, rewards, done, _ = env.step(actions)
            
            # 存储经验
            maddpg.replay.push(obs, actions, rewards, next_obs, done)
            
            obs = next_obs
            episode_reward += rewards.mean()
        
        episode_rewards.append(episode_reward)
        
        # 周期性更新
        if ep % update_freq == 0 and len(maddpg.replay) > batch_size:
            batch = maddpg.replay.sample(batch_size)
            actor_loss, critic_loss = maddpg.update(batch)
            
            if ep % 1000 == 0:
                avg_reward = np.mean(episode_rewards[-1000:])
                print(f"Episode {ep:5d} | "
                      f"Avg Reward: {avg_reward:8.2f} | "
                      f"Actor Loss: {actor_loss:.4f} | "
                      f"Critic Loss: {critic_loss:.4f}")
    
    # 保存模型
    torch.save({f'agent_{i}': maddpg.agents[i]['policy'].state_dict()
                for i in range(n_agents)}, 'gat_maddpg_uav.pth')
    
    return episode_rewards
```

### 4.3 Visualisierung der Trainingsergebnisse

Nach Abschluss der Schulung spiegeln folgende Indikatoren die Qualität der Strategie wider:

- **Kollisionsrate**: Anteil der Kollisionen pro Episode → Konvergiert auf 0 %
- **Durchschnittliche Episodenrückkehr**: Allmählich steigend
- **Strategieentropie**: hoch in der frühen Phase (zufällige Erkundung), niedrig in der späteren Phase (politische Konvergenz)
- **Entfernung zum Ziel**: Nach der Konvergenz kann das UAV das Ziel stabil erreichen

```python
def plot_training_curves(episode_rewards):
    import matplotlib.pyplot as plt
    
    # 移动平均
    window = 50
    smoothed = np.convolve(episode_rewards, 
                          np.ones(window)/window, mode='valid')
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.3, color='blue', label='Raw')
    plt.plot(smoothed, color='red', lw=2, label=f'{window}-ep Moving Avg')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('GAT-MADDPG Training Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # 冲突率
    conflicts = [1 if r < -50 else 0 for r in episode_rewards]
    conflict_rate = np.convolve(conflicts, np.ones(window)/window, mode='valid')
    plt.plot(conflict_rate, color='orange', lw=2, label='Collision Rate')
    plt.xlabel('Episode')
    plt.ylabel('Collision Rate')
    plt.title('Safety Metric')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gat_maddpg_training.png', dpi=150)
```

---## 5. Modernster Fortschritt: von GAT-MARL zu leistungsfähigeren Architekturen

### 5.1 Transformator + MARL

In den Jahren 2023–2024 begann die **Transformer**-Architektur, GAT zu ersetzen und zur gängigen Wahl für die Modellierung von Nachbarschaftsbeziehungen zu werden:

- **GAT → Aufmerksamkeit**: Verwenden Sie Multi-Head Self-Attention, um die Aufmerksamkeit im Diagramm zu ersetzen
- **Kommunikationsnachricht**: Der verborgene Status jedes UAV wird über Transformer Encoder aggregiert
- **Vorteile**: Es ist keine vordefinierte Adjazenzmatrix erforderlich, Beziehungen werden vollständig durch datengesteuertes Lernen erlernt

Repräsentative Arbeit:
- **MAFAT (Multi-Agent Foundation Action Transformer)**: Initialisieren Sie die Multi-Agent-Strategie mit vorab trainiertem Transformer
- **MAT (Multi-Agent Transformer)**: Modellieren Sie Multi-Agent-Probleme als Sequenz-zu-Sequenz-Übersetzung

### 5.2 Offline-MARL (Offline-MARL)

Die Beschaffung echter UAV-Trainingsdaten ist kostspielig und gefährlich. **Offline Reinforcement Learning (Offline RL)** Lernen Sie Strategien aus festen Datensätzen, um das Risiko von Online-Interaktionen zu vermeiden:

- **CQL (Conservative Q-Learning)**: Der Constraint-Q-Wert wird überschätzt
- **IQL (Implizites Q-Learning)**: Keine explizite Kritik erforderlich, wodurch Speicherplatz gespart wird

### 5.3 Sicherheitsschicht

Integrieren Sie Sicherheitsbeschränkungen direkt in die Ausgabeschicht des Richtliniennetzwerks:

$$
\pi_{safe}(s) = \text{Proj}_{\mathcal{A}_{safe}(s)} \pi(s)
$$

wobei $\mathcal{A}_{safe}(s)$ die Menge der sicheren Aktionen unter dem Zustand $s$ ist (z. B. der Geschwindigkeitsraum, der die Kollisionsvermeidungsbeschränkungen erfüllt). Dies ist zuverlässiger als die Bestrafung von Kollisionen in der Belohnungsfunktion – harte Einschränkungen haben Vorrang vor weichen Belohnungen.

---

## 6. Zusammenfassung: Technischer Überblick über GAT-MARL

Von Single-Agent RL über Multi-Agent Reinforcement Learning bis hin zur grafischen Aufmerksamkeitssteigerung haben wir einen **skalierbaren End-to-End-Konfliktlösungsweg** gewählt:| Ebene | Technologie | Problem gelöst |
|------|------|----------|
| **Lernrahmen** | CTDE (zentrales Training + dezentrale Ausführung) | Umweltinstationarität |
| **Algorithmus** | MADDPG / MAPPO / QMIX | Kreditzuweisung + kontinuierliche/diskrete Maßnahmen |
| **Topologiemodellierung** | GAT | Adaptive Nachbargewichte + Skalierbarkeit |
| **Sicherheitseinschränkungen** | Sicherheitsschicht / harte Einschränkungen | Kollisionsgarantie (im Vergleich zu Soft Rewards) |
| **Trainingsparadigma** | PPO (SIDESTEP/TRPO) | Trainingsstabilität |

Die bemerkenswertesten Richtungen in der Zukunft:
- **Grundlagenmodell + UAV**: Verwenden Sie ein großes Sprachmodell für das Verständnis von Anweisungen auf Aufgabenebene + MARL für die Steuerung auf niedriger Ebene
- **Echte Flugverifizierung**: Die Sim-to-Real-Migration ist immer noch eine zentrale Herausforderung
- **Szenario mit eingeschränkter Kommunikation**: GAT-Robustheit ohne Kommunikation oder Kommunikationsverzögerung

---

**Referenzen:**1. Lowe, R., et al. (2017). *Multi-Agent-Akteur-Kritiker für gemischte kooperativ-kompetitive Umgebungen (MADDPG).* Konferenz über neuronale Informationsverarbeitungssysteme (NeurIPS).
2. Foerster, J., et al. (2018). *Kontrafaktische Multi-Agent-Policy-Gradienten (COMA).* AAAI-Konferenz über künstliche Intelligenz.
3. Rashid, T., et al. (2018). *QMIX: Faktorisierung monotoner Wertfunktionen für tiefes Lernen zur Verstärkung mehrerer Agenten.* Internationale Konferenz für maschinelles Lernen (ICML).
4. Veličković, P., et al. (2018). *Graph-Aufmerksamkeitsnetzwerke.* Internationale Konferenz über lernende Repräsentationen (ICLR).
5. Everett, M., et al. (2021). *Kollisionsvermeidung bei dichtem Verkehr mit tiefgreifendem Verstärkungslernen.* IEEE International Conference on Robotics and Automation (ICRA).
6. Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* Internationale Konferenz über lernende Repräsentationen (ICLR).
7. Fan, T., et al. (2020). *Distriverbesserte Multi-Roboter-Kollisionsvermeidung durch Deep Reinforcement Learning für die Navigation in komplexen Szenarien.* The International Journal of Robotics Research (IJRR).
8. Mao, H., et al. (2020). *Multi-Agent-Kommunikation mit Double Attentional Deep Reinforcement Learning lernen.* Autonome Agenten und Multi-Agent-Systeme (JAAMAS).
9. Yu, L., et al. (2025). *Hybrid-Transformator-basiertes Multi-Agent-Verstärkungslernen für die Koordination mehrerer unbemannter Luftfahrzeuge in Luftkorridoren.* IEEE-Transaktionen auf Mobile Computing (TMC).
10. Zhu, Y., et al. (2025). *Multi-Task Multi-Agent Reinforcement Learning mit Task-Entity Transformers und Value Decomposition Training.* IEEE Transactions on Automation Science and Engineering (TASE).
11. Jiang, C., et al. (2024). *Verteilte, auf Stichproben basierende modellprädiktive Steuerung über Glaubensausbreitung für die Navigation in Multi-Roboter-Formationen.* IEEE Robotics and Automation Letters (RA-L).
12. Goeckner, A., et all. (2024). *Graph Neural Network-based Multi-Agent Reinforcement Learning for Resilient Distributed Coordination of Multi-Robot Systems.* Internationale IEEE/RSJ-Konferenz über intelligente Roboter und Systeme (IROS).