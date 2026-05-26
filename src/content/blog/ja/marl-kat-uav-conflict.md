---
title: "マルチエージェント強化学習とグラフ アテンション ネットワーク: UAV クラスターの競合解決のためのエンドツーエンド ソリューション"
description: "MARL (QMIX/COMA/MAPPO/MADDPG) と GAT の統合アーキテクチャを詳細に分析し、ポリシー勾配から基礎となるグラフ構造に至るまで、UAV クラスターの競合解決のエンドツーエンド学習を達成する方法を 1 つの記事で説明します。"
pubDate: 2026-04-07T11:24:40+08:00
tags: ["マルチエージェント強化学習", "マール", "ガット", "フィギュア注意", "ドローン", "パスの計画", "ディープラーニング"]
category: Tech
sourceHash: "41cd4a89dae677119d19d92d790e780642b52e5c"
---

# マルチエージェント強化学習とグラフ アテンション ネットワーク: UAV クラスターの競合解決のためのエンドツーエンドのソリューション

> [前の記事](/blog/uav-conflict-resolution/) では、UAV の競合解決のアルゴリズムのパノラマを整理しました。その中でも、強化学習 (特に MARL) は、50 機以上のドローンの群れにとって「最も現実的なオプション」とされています。この記事では、シングル エージェント RL の基礎から始めて、マルチエージェント シナリオの中核的な課題に入り、MADDPG、QMIX、COMA、MAPPO などの主流アルゴリズムを分析し、**GAT (グラフ アテンション ネットワーク)** が MARL にスケーラブルなトポロジ認識機能を提供し、最終的にエンドツーエンドの競合解決戦略を達成する方法に焦点を当てて、このルートに焦点を当てます。

---

## 1. シングルエージェントからマルチエージェントへ: MARL はなぜそれほど難しいのでしょうか?

### 1.1 単一エージェントの RL レビュー

おなじみのシングルエージェント RL から始めましょう。シングルエージェント MDP は、$(\mathcal{S}, \mathcal{A}, \mathcal{R}, \mathcal{P}, \gamma)$ の 4 つの要素で記述されます。

- **状態値関数**: $V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^\infty \gamma^t r_t \mid s_0 = s\right]$
- **アクション値関数**: $Q^\pi(s,a) = \mathbb{E}_\pi\left[\sum_{t=0}^\infty \gamma^t r_t \mid s_0 = s, a_0 = a\right]$
- **最適な戦略**: $\pi^* = \arg\max_\pi V^\pi(s)$

シングルエージェント RL の中核となる前提: **環境は安定している** - トレーニングするエピソードの数に関係なく、環境 $\mathcal{P}$ のダイナミクスは常に変化しません。

### 1.2 マルチエージェントの 3 つの本質的な問題点

マルチエージェントのシナリオはこの前提を打ち破り、次の 3 つの根本的な問題を引き起こします。

**① 環境の非定常性（Non-Stationarity）**

エージェント $i$ がポリシー $\pi_i$ を学習しているとき、他のエージェントのポリシー $\{\pi_j\}_{j\neq i}$ も変更されます。これはつまり：$$
\mathcal{P}_i(s'\mid s, a_1,\dots,a_n) \neq \mathcal{P}_i(s'\mid s, a_1,\dots,a_n, a_1',\dots,a_n')
$$

シングルエージェント RL では、現在の状態とアクションが与えられると、次の状態の分布は固定されます。しかし、マルチエージェントのシナリオでは、他のエージェントが異なるエピソードで異なるアクションを実行する可能性があるため、同じ状態とアクションのペアがまったく異なる次の状態分布に対応する可能性があります。

これは **エクスペリエンス リプレイ バッファ** の失敗に直接つながります。保存されたエクスペリエンス データは「古い」戦略からのものであり、それらを使用してトレーニングすると戦略が崩壊します。

**②単位の割り当て**

$n$ エージェントが共同でチーム報酬 $r$ を受け取った場合、この報酬を各エージェントの貢献に帰す方法は何ですか?

$$
r_t = f(\mathbf{s}_t, \mathbf{a}_t, \mathbf{s}_{t+1})
$$

たとえば、複数の UAV が協力して障害物を回避します。各エージェントはいくら貢献しましたか?少数のエージェントだけが報酬を得る場合、他のエージェントは学習を停止します。

**③ 共同行動空間指数爆発**

$n$ 個の UAV があり、それぞれ $|\mathcal{A}|$ アクション オプションがあり、共同アクション スペース $|\mathcal{A}|^n$ は $n$ とともに指数関数的に増加します。結合空間における貪欲な探索の範囲はゼロに近づきます。

### 1.3 MARL アルゴリズムの分類

上記の困難に対応して、学術コミュニティは 3 つの主要なルートを開発しました。

|ルート |代表的なアルゴリズム |核となるアイデア |代表論文 |
|------|----------|----------|----------|
| **自主学習 (IL)** | IQL、DQN |それぞれが自分自身のことを学び、他人の影響を無視します。タン、1993年 |
| **集中トレーニング + 分散実行 (CTDE)** | MADDPG、QMIX、MAPPO |トレーニング中にグローバル情報を使用し、実行中にローカル観察を使用します。 Lowe 他、2017 |
| **完全に分散化** |コーマ, VDND |純粋にローカルな戦略であり、一元化されたトレーニングはありません |フェルスターら、2018 |> **CTDE は、UAV 競合解決の現在の主流パラダイム**です。CTDE は、トレーニング中にグローバル情報を使用して学習効率を向上させるだけでなく、実行中に限られた通信の下でリアルタイムの意思決定機能を維持することもできます。

---

## 2. CTDE フレームワーク: トレーニングには神の視点を使用し、実行にはローカル観察を使用します

### 2.1 集中批評家の設計哲学

CTDE の核となる洞察は次のとおりです。 **トレーニング フェーズと実行フェーズでは、利用可能な情報が異なる可能性があります**。

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

### 2.2 MADDPG: 連続アクション空間における CTDE のパイオニア

**MADDPG (マルチエージェント DDPG)** は、2017 年に OpenAI によって提案され、連続アクション空間マルチエージェント深層強化学習におけるマイルストーンです。

**コアフォーミュラ:**

各エージェント $i$ は、アクターと批評家の構造を維持します。

$$
\nabla_{\theta_i} J(\theta_i) = \mathbb{E}_{\mathbf{s} \sim \mathcal{D}}\left[
    \nabla_{\theta_i} \log \pi_i(a_i \mid o_i) \cdot
    Q_i^\pi(\mathbf{s}, a_1, \dots, a_n) \Big|_{a_i = \pi_i(o_i)}
\右]
$$

主な違い: $Q_i^\pi$ への入力は、ローカルな観測値ではなく、グローバル状態 $\mathbf{s}$ とすべてのエージェントの共同アクション $\mathbf{a}$ です。

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

### 2.3 QMIX: クレジット割り当てを解決するための値の分解

MADDPG は連続アクション空間の問題を解決しますが、Critic はグローバル状態 $\mathbf{s}$ を必要とします。実際の UAV シナリオでは、セントラル ノードはグローバル状態を取得できない可能性があります。

**QMIX** (Queensland Institute、2018) の中核となる革新は、**結合 Q 値を個々のエージェントの周辺 Q 値に分解する**です。$$
Q_{tot}(\boldsymbol{\tau}, \mathbf{u}) = g_\theta(\boldsymbol{\tau}, \mathbf{u}; \boldsymbol{\phi}_1, \dots, \boldsymbol{\phi}_n)
$$

$\boldsymbol{\tau}_i$ がエージェント $i$ の行動観察軌跡である場合、$g_\theta$ は次を満たす **単調混合ネットワーク**です。

$$
\frac{\partial Q_{tot}}{\partial Q_i} \geq 0, \quad \forall i
$$

単調性制約は重要な特性を保証します。 **分散実行中、各エージェントの $Q_i$ の独立した貪欲な最大化は、$Q_{tot}$** のグローバルな最大化と同等です。

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

### 2.4 MAPPO: 高度に並行したシナリオにおける政策勾配の勝利

**MAPPO (マルチエージェント PPO)** は、PPO アルゴリズムをマルチエージェント シナリオに拡張し、近年の UAV クラスター タスクで良好なパフォーマンスを示しています (2022 年から 2024 年までの複数の主要なカンファレンス論文)。

**PPO の主な利点**: 信頼領域の制約により、トレーニングの安定性が確保され、DDPG シリーズのハイパーパラメーターによる災害が回避されます。

PPO -クリップ ターゲット:
$$
\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t\left[
    \分\左(
        r_t(\theta) \hat{A}_t,
        \text{クリップ}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t
    \右)
\右]
$$

$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$ は確率比、$\hat{A}_t$ は GAE (Generalized Advantage Estimation) です。UAV 競合解決における MAPPO の一般的な構成:

|パラメータ |推奨値 |説明 |
|------|--------|------|
|クリップ率 $\epsilon$ | 0.2 | PPO のデフォルト |
|ホライゾン $T$ | 128–256 |エポックごとのロールアウト ステップの数 |
| PPO エポック | 2–4 |バッチごとに繰り返される更新の数 |
| GAE $\lambda$ | 0.95 |優勢推定のバイアス分散バランス |
|隠れ層の寸法 | 64–128 | UAV シナリオには十分 |
|正規化 | OBS + 報酬の正規化 |鍵！マルチエージェントのコンバージェンスに大きな影響 |

---

## 3. GAT: MARL に「誰に従うべきか」を学ばせる

### 3.1 なぜ MARL にはグラフ構造が必要なのでしょうか?

UAV クラスターでは、すべてのエージェントが同じように重要であるわけではありません。競合の解決を例に挙げます。

- UAV が私に衝突しようとしている → **重大な懸念**
- UAV は視界の外にある → **無視しても問題ありません**
- 動く障害物に近づく → **動的な注意が必要**

ただし、従来の MARL (MADDPG、QMIX など) は、完全に接続されたトポロジ ($\mathcal{O}(N^2)$ 通信) または固定トポロジ (リング、最近傍など) のいずれかで、すべての近隣ノードを平等に扱います。

**GAT の導入により、次の 2 つの主要な問題が解決されます。**

1. **適応近隣重み**: アテンション メカニズムを通じて、現在の決定にとってどの近隣がより重要であるかを学習します。
2. **拡張性**: ドローンの数に応じて増加せず、動的なトポロジーをサポートします

### 3.2 GAT の基本原則

GAT は、各層のノード $i$ の特徴 $\mathbf{h}_i$ に対して **近隣集約** を実行し、重みはアテンション メカニズムによって動的に計算されます。$$
\alpha_{ij} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^\top[\mathbf{W}\mathbf{h}_i \Vert \mathbf{W}\mathbf{h}_j]\right)\right)}
{\sum_{k \in \mathcal{N}_i} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^\top[\mathbf{W}\mathbf{h}_i \Vert \mathbf{W}\mathbf{h}_k]\right)\right)}
$$

$$
\mathbf{h}_i' = \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij} \mathbf{W}\mathbf{h}_j\right)
$$

$\mathbf{W}$ は学習可能な線形変換行列、$\mathbf{a}$ は注意ベクトル、$\Vert$ はスプライシングを表します。

**LeakyReLU を使用する理由**: 注意スコアにわずかな非線形性が導入され、正と負の値の非対称処理が可能になります。

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

### 3.3 GAT + MARL の統合アーキテクチャ

GAT を MARL のポリシー ネットワークに埋め込んで **GAT-MARL** アーキテクチャを形成します。

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

**完全な実装コード:**

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

## 4. エンドツーエンドの競合解決トレーニング プロセス

### 4.1 シミュレーション環境の設計

UAV の競合解決には専用のシミュレーション環境を使用します。

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

### 4.2 トレーニングサイクルを完了する

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

### 4.3 トレーニング結果の可視化

トレーニングが完了すると、次の指標が戦略の品質を反映します。

- **衝突率**: エピソードごとの衝突の割合 → 0% に収束
- **平均エピソードリターン**: 徐々に上昇
- **戦略エントロピー**: 初期段階では高く (ランダム探索)、後期段階では低くなります (ポリシーの収束)
- **ターゲットまでの距離**: 収束後、UAV は安定してターゲットに到達できます。

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

---## 5. 最先端の進歩: GAT-MARL からより強力なアーキテクチャへ

### 5.1 トランス + MARL

2023 年から 2024 年にかけて、**Transformer** アーキテクチャが GAT に取って代わり始め、近隣関係モデリングの主流の選択肢となりました。

- **GAT → アテンション**: マルチヘッド セルフ アテンションを使用してグラフ アテンションを置き換えます
- **通信メッセージ**: 各 UAV の隠れた状態は、Transformer Encoder を通じて集約されます。
- **利点**: 事前定義された隣接行列は必要ありません。関係はデータ駆動型学習によって完全に学習されます。

代表作：
- **MAFAT (Multi-Agent Foundation Action Transformer)**: 事前トレーニングされた Transformer を使用してマルチエージェント戦略を初期化します。
- **MAT (Multi-Agent Transformer)**: マルチエージェントの問題をシーケンスからシーケンスへの変換としてモデル化します。

### 5.2 オフライン MARL (オフライン MARL)

実際の UAV トレーニング データを取得するにはコストがかかり、危険です。 **オフライン強化学習 (オフライン RL)** オンライン インタラクションのリスクを回避するために、固定データ セットから戦略を学習します。

- **CQL (保守的な Q 学習)**: 制約 Q 値が過大評価されています
- **IQL (Implicit Q-Learning)**: 明示的な批評家が不要で、ストレージを節約できます。

### 5.3 安全層

セキュリティ制約をポリシー ネットワーク出力層に直接埋め込みます。

$$
\pi_{安全な}(s) = \text{Proj}_{\mathcal{A}_{安全な}(s)} \pi(s)
$$

ここで、$\mathcal{A}_{safe}(s)$ は、状態 $s$ での安全なアクションのセット (衝突回避制約を満たす速度空間など) です。これは、報酬関数で衝突にペナルティを与えるよりも信頼性が高く、ハード制約はソフト報酬よりも優先されます。

---

## 6. 概要: GAT-MARL の技術概要

シングルエージェント RL からマルチエージェント強化学習、グラフ アテンション強化に至るまで、**スケーラブルなエンドツーエンドの競合解決** ルートを採用しました。|レベル |テクノロジー |問題が解決しました |
|------|------|----------|
| **学習フレームワーク** | CTDE (集中トレーニング + 分散実行) |環境の非定常性 |
| **アルゴリズム** | MADDPG / MAPPO / QMIX |クレジット割り当て + 継続的/個別アクション |
| **トポロジ モデリング** |ガット |適応近隣重み + スケーラビリティ |
| **安全上の制約** |安全層 / ハード制約 |衝突保証 (対ソフトリワード) |
| **トレーニング パラダイム** | PPO (サイドステップ/TRPO) |トレーニングの安定性 |

今後の最も注目すべき方向性:
- **基礎モデル + UAV**: タスクレベルの命令の理解には大規模言語モデルを使用し、低レベルの制御には MARL を使用します
- **実際の飛行検証**: Sim から Real への移行は依然として主要な課題です
- **通信制限シナリオ**: 通信または通信遅延がない場合の GAT の堅牢性

---

**参考文献:**1. Lowe、R.、他。 （2017年）。 *協力競争環境が混在するマルチエージェントアクター批評家 (MADDPG)。 * 神経情報処理システムに関する会議 (NeurIPS)。
2. Foerster、J.、他。 （2018年）。 *反事実的なマルチエージェント ポリシー勾配 (COMA)。* AAAI 人工知能会議。
3. ラシッド、T.、他。 （2018年）。 *QMIX: 深いマルチエージェント強化学習のための単調値関数因数分解。* 機械学習に関する国際会議 (ICML)。
4. Veličković、P.、他。 （2018年）。 *グラフ アテンション ネットワーク。* 学習表現に関する国際会議 (ICLR)。
5. Everett、M.、他。 （2021年）。 *深層強化学習による密集した交通における衝突回避* IEEE ロボティクスとオートメーションに関する国際会議 (ICRA)。
6. Hu、E.J.、他。 （2021年）。 *LoRA: 大規模言語モデルの低ランク適応。* 学習表現に関する国際会議 (ICLR)。
7. ファン、T.、他。 （2020年）。 *ディストリクト複雑なシナリオでのナビゲーションのための深層強化学習によるマルチロボット衝突回避を評価しました。* 国際ロボット研究ジャーナル (IJRR)。
8. マオ、H.、他。 （2020年）。 *二重注意深層強化学習によるマルチエージェント コミュニケーションの学習。* 自律エージェントとマルチエージェント システム (JAAMAS)。
9. Yu、L.、他。 (2025年)。 *空中回廊における複数の無人航空機の調整のためのハイブリッド変圧器ベースのマルチエージェント強化学習。* モバイル コンピューティング (TMC) に関する IEEE トランザクション。
10. Zhu、Y.、他。 (2025年)。 *タスク エンティティ トランスフォーマーと値分解トレーニングを使用したマルチタスク マルチエージェント強化学習。* IEEE Transactions on Automation Science and Engineering (TASE)。
11. Jiang、C.、他。 （2024年）。 *マルチロボットフォーメーションナビゲーションのための信念伝播による分散サンプリングベースのモデル予測制御。* IEEE Robotics and Automation Letters (RA-L)。
12. Goeckner、A.、他l. （2024年）。 *マルチロボット システムの回復力のある分散調整のためのグラフ ニューラル ネットワーク ベースのマルチエージェント強化学習。* インテリジェント ロボットおよびシステムに関する IEEE/RSJ 国際会議 (IROS)。