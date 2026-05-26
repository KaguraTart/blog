---
title: "UAV経路競合シミュレーション環境の構築：紙面での実践からコード実装まで"
description: "Gym/Gazebo/AirSim マルチフレームワークの完全な例を使用して、主流のシミュレーション プラットフォームの比較、ダイナミクス モデリング、状態空間設計、競合定義、報酬関数の構築、ベンチマーク テストのシナリオをカバーする、UAV 複数航空機の衝突シナリオ シミュレーション環境の構築方法を系統的にレビューします。"
pubDate: 2026-04-07T11:34:43+08:00
tags: ["ドローン", "シミュレーション環境", "マルチエージェント", "パスの計画", "ジム", "ガゼボ", "エアシム", "強化学習"]
category: Tech
sourceHash: "ff9e37b397ab58192e278968dc5a92dd4877ea60"
---

# UAV パス競合シミュレーション環境構築：紙面での実践からコード実装まで

> 最初の 2 つの記事では、[UAV 競合解決アルゴリズムのパノラマ](/blog/uav-conflict-resolution/) と [MARL+GAT エンドツーエンド ソリューション](/blog/marl-kat-uav-conflict/) をそれぞれ整理しました。この記事では、次のような低レベルの質問に答えます。**シミュレーション環境で現実的で再現可能な UAV 衝突シナリオを構築するにはどうすればよいですか? ** これは、アルゴリズム評価の信頼性と実験の比較可能性を直接決定します。

---

## 1. シミュレーション環境の構築がなぜそれほど重要なのでしょうか?

シミュレーション環境は、アルゴリズム理論と飛行測定をつなぐ架け橋です。適切に設計されたシミュレーション環境は、次の 3 つの側面を満たす必要があります。

|寸法 |意味 |よくある質問 |
|------|------|-----------|
| **信頼性** |シミュレーションと現実のギャップ |ダイナミクスの過度の単純化とセンサー モデルの欠如 |
| **再現性** |同じシードの下で一貫した結果 |制御されていないランダムな初期化、浮動小数点エラーの蓄積 |
| **スケーラビリティ** |大規模クラスターのサポート |単一マシンのシミュレーションは並列化できず、通信トポロジがありません。

この記事では、信頼性と拡張性の両方を考慮した **MARL トレーニング シナリオ** に基づくシミュレーション環境の構築に焦点を当てています。結局のところ、50 機以上のドローンで実際の飛行制御をトレーニングするのは非現実的です。

---

## 2. 主流のシミュレーション プラットフォームの比較

### 2.1 プラットフォームの概要|プラットフォーム |基礎となるエンジン |物理的な信頼性 |マルチマシンのサポート | MARL の互換性 |論文引用 |
|------|----------|-----------|----------|------------|-----------|
| **AirSim** (マイクロソフト) |アンリアル エンジン | ⭐⭐⭐⭐⭐ | ✅ | ⚠️ラッピングが必要です |高 |
| **フライトゴーグル** (MIT) |団結 | ⭐⭐⭐⭐ | ✅ | ⚠️ラッピングが必要です |高 |
| **RotorS** (チューリッヒ工科大学) |ガゼボ/ROS | ⭐⭐⭐⭐ | ✅ | ✅ PyMARL 互換 |中 |
| **モールス** (LAAS-CNRS) |ブレンダー ゲーム エンジン | ⭐⭐⭐ | ✅ | ✅ ネイティブ |中 |
| **ウェブボット** |自社開発 | ⭐⭐⭐⭐ | ✅ | ✅ |低い |
| **カスタム 2D/3D** |該当なし | ⭐⭐ | ✅✅✅ | ✅✅✅ | - |

**核となる結論:**
- **アカデミック ベンチマーク**: RotorS + ROS/Gazebo が主流の選択肢です (ETH Zurich によって作成され、PX4 と互換性があります)
- **エンドツーエンドの RL 研究**: ほとんどの論文では **カスタム ジム環境** (完全にカスタマイズされており、報酬関数を簡単に変更できます) が使用されています。
- **Sim-to-Real ターゲット**: AirSim/FlightGoggles (高忠実度ビジョン + センサー ノイズ)

### 2.2 AirSim: ゼロから 100 までの拡張体験

Microsoft AI for Earth チームは、AirSim 上で最大 **100 台の固定翼 UAV** の群れ調整をトレーニングしました。

> シャー、S.、他。 （2018年）。 *AirSim: 自動運転車向けの高忠実度の視覚的および物理的シミュレーション。* フィールドおよびサービス ロボティクス。AirSim の主な利点:
- **センサー シミュレーション**: カメラ、LiDAR、GPS、IMU の完全なシミュレーション、ノイズを注入可能
- **気象システム**: 風速、雲、光がセンサーに与える影響
- **API の統合**: C++ / Python / ROS インターフェイスをサポート

```python
import airsim
import numpy as np

class AirSimUAVEnv:
    """
    AirSim 多 UAV 冲突环境封装
    """
    def __init__(self, n_agents=8, drone_names=None):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        
        if drone_names is None:
            self.drone_names = [f"Drone_{i}" for i in range(n_agents)]
        else:
            self.drone_names = drone_names
        
        self.n_agents = len(self.drone_names)
        
        # 为每架 UAV 解锁并设置为被动模式（由 RL 控制）
        for name in self.drone_names:
            self.client.enableApiControl(True, name)
            self.client.armDisarm(True, name)
    
    def reset(self):
        """随机初始化各 UAV 位置和目标"""
        self.goals = {}
        for name in self.drone_names:
            # 随机起点
            start_pos = np.random.uniform(-50, 50, size=3)
            start_pos[2] = np.random.uniform(10, 30)  # 高度 10-30m
            
            # 随机终点
            goal_pos = np.random.uniform(-50, 50, size=3)
            goal_pos[2] = np.random.uniform(10, 30)
            
            self.client.simSetVehiclePose(
                airsim.Vector3R(*start_pos),
                airsim.ToQuaternionr(0, 0, 0),
                vehicle_name=name
            )
            
            self.goals[name] = goal_pos
        
        return self._get_obs()
    
    def _get_obs(self):
        """获取各 UAV 的本地观测"""
        obs = []
        for name in self.drone_names:
            state = self.client.getMultirotorState(name)
            pos = state.kinematics_estimated.position
            vel = state.kinematics_estimated.linear_velocity
            
            obs.append(np.array([pos.x_val, pos.y_val, pos.z_val,
                                vel.x_val, vel.y_val, vel.z_val]))
        return np.array(obs)
    
    def step(self, actions):
        """执行动作：vx, vy, vz 速度指令"""
        for i, name in enumerate(self.drone_names):
            self.client.moveByVelocityAsync(
                actions[i, 0], actions[i, 1], actions[i, 2],
                duration=0.1, vehicle_name=name
            )
        
        # 等待执行完成（AirSim 为异步 API）
        self.client.simPause(True)
       airsim.multirotorclient()
        self.client.simContinueForTime(0.1)
        self.client.simPause(False)
        
        obs = self._get_obs()
        reward, done = self._compute_reward()
        
        return obs, reward, done, {}
    
    def _compute_reward(self):
        """奖励函数"""
        reward = np.zeros(self.n_agents)
        done = np.zeros(self.n_agents, dtype=bool)
        
        for i, name in enumerate(self.drone_names):
            state = self.client.getMultirotorState(name)
            pos = np.array([state.kinematics_estimated.position.x_val,
                           state.kinematics_estimated.position.y_val,
                           state.kinematics_estimated.position.z_val])
            
            dist_to_goal = np.linalg.norm(pos - self.goals[name])
            reward[i] += 1.0 / (dist_to_goal + 1.0)
            
            # 冲突惩罚
            for j, other_name in enumerate(self.drone_names):
                if i >= j:
                    continue
                other_state = self.client.getMultirotorState(other_name)
                other_pos = np.array([other_state.kinematics_estimated.position.x_val,
                                      other_state.kinematics_estimated.position.y_val,
                                      other_state.kinematics_estimated.position.z_val])
                
                dist = np.linalg.norm(pos - other_pos)
                if dist < 5.0:  # 5m 安全距离
                    reward[i] -= 100
                    done[i] = True
        
        return reward, done.any()
```

### 2.3 Gazebo / RotorS: ROS エコシステムの標準的な選択肢

チューリッヒ工科大学の **RotorS** は、ROS/Gazebo エコシステムの中で最も完全なマルチ UAV シミュレーション パッケージです。

> フェーン、P.、他。 （2020年）。 *ETHZ フライング マシン アリーナ データセット。* ETH チューリッヒ。

RotorS は以下をサポートします。
- **mav_msgs** インターフェース (スピン + ピッチ + ロール + スラスト → モーター PWM)
- **PX4 SITL/HITL**: ループ内の完全な飛行制御ファームウェア
- **RViz 視覚化**: クラスターのステータスをリアルタイムで観察
- **Gazebo 物理エンジン**: ODE / Bullet / Simbody オプション

```bash
# 安装 RotorS（Ubuntu 20.04 + ROS Noetic）
sudo apt-get install ros-noetic-rotors-description ros-noetic-rotors-gazebo
source /opt/ros/noetic/setup.bash

# 启动 8 架 UAV 集群场景
roslaunch rotors_gazebo mav_tecs_controller.launch \
    mav_name:=hummingbird \
    world_name:=forest \
    n_uavs:=8
```

### 2.4 カスタムジム環境: 学術的な RL 研究の主流の選択肢

MARL UAV 論文の大部分 (MADDPG/QMIX/MAPPO のオリジナルの実装を含む) は、次の 3 つの理由から完全にカスタムの Gym スタイル環境を使用しています。

1. **トレーニング速度**: グラフィックス レンダリングなし、Python 直接ループ、1 秒あたり数万のインタラクションを完了可能
2. **制御可能な報酬関数**: C++ コードを変更せずに設計を迅速に反復します。
3. **再現性**: 固定ランダムシードを使用すると、結果は完全に再現可能です。

```python
import gym
from gym import spaces
import numpy as np

class UAVConflictGym(gym.Env):
    """
    标准 Gym 风格 UAV 冲突环境
    兼容 Stable-Baselines3 / rllib / PyMARL
    """
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        n_agents=8,
        area_size=200.0,     # 仿真区域边长 (m)
        safe_radius=5.0,     # 安全隔离距离 (m)
        max_speed=20.0,      # 最大速度 (m/s)
        dt=0.1,              # 仿真步长 (s)
        max_steps=500,       # 最大 episode 步数
        seed=None
    ):
        super().__init__()
        
        self.n_agents = n_agents
        self.area_size = area_size
        self.safe_radius = safe_radius
        self.max_speed = max_speed
        self.dt = dt
        self.max_steps = max_steps
        
        # 观测空间：(位置3 + 速度3 + 目标3 + 自身id编码 + 邻居信息)
        self.obs_dim = 3 + 3 + 3 + n_agents + 3 * (n_agents - 1)
        self.action_dim = 3  # Δvx, Δvy, Δvz
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1, high=1,
            shape=(self.action_dim,), dtype=np.float32
        )
        
        # 多智能体环境的 each-agent 空间
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_agents, self.obs_dim), dtype=np.float32
        )
        
        if seed is not None:
            self.seed(seed)
    
    def seed(self, seed):
        np.random.seed(seed)
    
    def reset(self):
        """初始化 UAV 状态"""
        # 位置：均匀分布在区域内
        self.positions = np.random.uniform(
            -self.area_size / 2, self.area_size / 2,
            size=(self.n_agents, 3)
        ).astype(np.float32)
        
        # 速度：随机方向，5-15 m/s
        speeds = np.random.uniform(5, 15, size=(self.n_agents,))
        directions = np.random.randn(self.n_agents, 3)
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        self.velocities = speeds[:, None] * directions
        
        # 目标点
        self.goals = np.random.uniform(
            -self.area_size / 2, self.area_size / 2,
            size=(self.n_agents, 3)
        ).astype(np.float32)
        
        # 确保起点和终点不重叠（避免初始冲突过于密集）
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                dist = np.linalg.norm(self.positions[i] - self.positions[j])
                if dist < self.safe_radius * 2:
                    # 重新生成
                    self.positions[j] = np.random.uniform(
                        -self.area_size / 2, self.area_size / 2, size=3
                    ).astype(np.float32)
        
        self.step_count = 0
        self.prev_distances = self._compute_distances()
        
        return self._get_obs()
    
    def step(self, actions):
        """
        actions: (n_agents, 3) 速度增量
        """
        actions = np.clip(actions, -1, 1) * self.max_speed * self.dt
        
        # 更新速度
        new_velocities = self.velocities + actions
        speed = np.linalg.norm(new_velocities, axis=1, keepdims=True)
        new_velocities = np.clip(new_velocities, -self.max_speed, self.max_speed)
        new_velocities = new_velocities * np.clip(speed / self.max_speed, 0, 1)
        new_velocities = new_velocities / (np.linalg.norm(new_velocities, axis=1, keepdims=True) + 1e-8) * np.clip(speed, 0, self.max_speed)
        self.velocities = new_velocities
        
        # 更新位置
        self.positions += self.velocities * self.dt
        
        # 边界处理：软边界反射
        for i in range(self.n_agents):
            for dim in range(3):
                half = self.area_size / 2
                if abs(self.positions[i, dim]) > half:
                    self.positions[i, dim] = np.clip(
                        self.positions[i, dim], -half, half
                    )
                    self.velocities[i, dim] *= -0.8  # 反射 + 能量损失
        
        # 奖励计算
        reward, info = self._compute_reward()
        
        # 冲突检测
        collision = self._check_collision()
        done = collision or self.step_count >= self.max_steps
        
        self.step_count += 1
        
        return self._get_obs(), reward, np.full(self.n_agents, done), info
    
    def _compute_distances(self):
        """计算所有 UAV 对之间的距离矩阵"""
        dist_matrix = np.zeros((self.n_agents, self.n_agents))
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                d = np.linalg.norm(self.positions[i] - self.positions[j])
                dist_matrix[i, j] = dist_matrix[j, i] = d
        return dist_matrix
    
    def _compute_reward(self):
        """分项奖励函数"""
        reward = np.zeros(self.n_agents)
        info = {'collision': False}
        
        # 1. 任务进度奖励（到目标距离减少）
        dist_to_goal = np.linalg.norm(self.positions - self.goals, axis=1)
        prev_dist = self.prev_distances.copy()
        np.fill_diagonal(prev_dist, np.inf)
        
        # 进度：到目标距离减少得正奖励
        for i in range(self.n_agents):
            progress = self.prev_distances[i, i] - dist_to_goal[i]
            reward[i] += 10.0 * progress  # 缩放系数
            # 到达目标
            if dist_to_goal[i] < 2.0:
                reward[i] += 50.0
        
        # 2. 安全奖励（维持安全距离）
        collision_any = False
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                d = np.linalg.norm(self.positions[i] - self.positions[j])
                
                if d < self.safe_radius:
                    # 碰撞！
                    reward[i] -= 200
                    reward[j] -= 200
                    collision_any = True
                elif d < self.safe_radius * 2:
                    # 接近告警
                    close_factor = (self.safe_radius * 2 - d) / self.safe_radius
                    reward[i] -= 5.0 * close_factor
                    reward[j] -= 5.0 * close_factor
        
        info['collision'] = collision_any
        
        # 3. 能量惩罚（避免无谓机动）
        action_penalty = -0.01 * np.sum(np.square(self.velocities - self.prev_velocities))
        reward += action_penalty
        
        # 4. 速度维持奖励（偏好稳定飞行）
        for i in range(self.n_agents):
            speed = np.linalg.norm(self.velocities[i])
            if self.max_speed * 0.3 < speed < self.max_speed * 0.8:
                reward[i] += 0.5
        
        self.prev_distances = self._compute_distances()
        
        return reward, info
    
    def _check_collision(self):
        """检测是否有碰撞"""
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if np.linalg.norm(self.positions[i] - self.positions[j]) < self.safe_radius:
                    return True
        return False
    
    def _get_obs(self):
        """构建各 UAV 的观测向量"""
        obs = []
        for i in range(self.n_agents):
            # 自身状态：位置 + 速度
            self_state = np.concatenate([self.positions[i], self.velocities[i]])
            
            # 目标信息：方向 + 距离
            direction_to_goal = self.goals[i] - self.positions[i]
            dist_to_goal = np.linalg.norm(direction_to_goal)
            direction_to_goal_norm = direction_to_goal / (dist_to_goal + 1e-6)
            goal_info = np.concatenate([direction_to_goal_norm, np.array([dist_to_goal])])
            
            # 自身 ID 编码（one-hot）
            id_encoding = np.zeros(self.n_agents)
            id_encoding[i] = 1.0
            
            # 邻居信息（相对位置 + 相对速度）
            neighbor_info = []
            for j in range(self.n_agents):
                if i != j:
                    rel_pos = self.positions[j] - self.positions[i]
                    rel_vel = self.velocities[j] - self.velocities[i]
                    neighbor_info.extend(rel_pos.tolist() + rel_vel.tolist())
            
            neighbor_info = np.array(neighbor_info, dtype=np.float32)
            
            # 拼接
            full_obs = np.concatenate([
                self_state,       # 6
                goal_info,        # 4
                id_encoding,      # n_agents
                neighbor_info      # (n_agents-1)*6
            ])
            
            obs.append(full_obs)
        
        return np.array(obs, dtype=np.float32)
    
    def render(self, mode='human'):
        """简单 2D 可视化"""
        import matplotlib.pyplot as plt
        
        plt.clf()
        plt.xlim(-self.area_size/2, self.area_size/2)
        plt.ylim(-self.area_size/2, self.area_size/2)
        
        # 绘制 UAV 和目标
        for i in range(self.n_agents):
            # 位置
            plt.scatter(self.positions[i, 0], self.positions[i, 1],
                       color='blue', s=100, zorder=5)
            # 速度向量
            plt.arrow(self.positions[i, 0], self.positions[i, 1],
                     self.velocities[i, 0], self.velocities[i, 1],
                     head_width=2, color='blue', alpha=0.5)
            # 目标点
            plt.scatter(self.goals[i, 0], self.goals[i, 1],
                       color='green', s=50, marker='x')
            # 连线
            plt.plot([self.positions[i, 0], self.goals[i, 0]],
                    [self.positions[i, 1], self.goals[i, 1]],
                    'g--', alpha=0.2)
        
        plt.pause(0.01)
        plt.ioff()
```

---

## 3. 環境設計の核となる要素

### 3.1 動的モデル: 粒子モデルから 6 自由度まで

**レベル 1 — 点質量**
位置と速度の更新のみを考慮する最も単純化されたバージョンは、アルゴリズムの検証と高速トレーニングに使用されます。

$$
\dot{\mathbf{p}} = \mathbf{v}、\quad \dot{\mathbf{v}} = \mathbf{a}
$$$$
\mathbf{a}_{new} = \text{clip}(\mathbf{a}_{old} + \mathbf{u}, \mathbf{v}_{min}, \mathbf{v}_{max})
$$

**レベル 2 — 単一のインテグレータ**
アクチュエータが瞬時に反応すると仮定して、速度を直接制御します。

$$
\mathbf{v}_i[k+1] = \mathbf{u}_i[k]
$$

**レベル 3 — ダブル インテグレータ**
加速限界を考慮すると、実際の航空機に近くなります。

$$
\mathbf{v}_i[k+1] = \mathbf{v}_i[k] + \mathbf{a}_i[k] \Delta t
$$
$$
\mathbf{p}_i[k+1] = \mathbf{p}_i[k] + \mathbf{v}_i[k] \Delta t + \frac{1}{2}\mathbf{a}_i[k]\Delta t^2
$$

$$
\|\mathbf{a}_i[k]\| \leq a_{max}, \quad \|\mathbf{v}_i[k]\| \leq v_{最大}
$$

**レベル 4 — 6-DOF クアッドローター**
姿勢と推力の制約を考慮した完全な非線形モデル:

$$
\dot{\boldsymbol{\eta}} = \mathbf{R}(\boldsymbol{\Theta})\boldsymbol{\omega}
$$
$$
m\ddot{\mathbf{p}} = -mg\mathbf{e}_3 + \mathbf{R}(\boldsymbol{\Theta})\mathbf{T}_i\mathbf{e}_3 - \mathbf{D}(\dot{\mathbf{p}})
$$

$\boldsymbol{\eta}$ はオイラー角、$\mathbf{R}$ は回転行列、$\mathbf{T}_i$ は $i$ 番目の軸推力です。|モデル |計算量｜信頼性 |適用ステージ |
|------|--------|----------|----------|
|質点 | ⭐ | ⭐ |アルゴリズムのクイック検証 |
|単一のインテグレータ | ⭐⭐ | ⭐⭐ | RL トレーニング (主流の選択) |
|双方向インテグレータ | ⭐⭐ | ⭐⭐⭐ |実際の飛行に近い |
| 6自由度 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Sim-to-Real 事前検証 |

### 3.2 状態空間設計

適切な状態空間には、意思決定に必要なすべての情報が含まれており、制御可能な次元が必要です。

```python
def build_observation(agent_id, all_positions, all_velocities, goals, adj_matrix):
    """
    推荐的状态空间组成（维度可控）
    """
    i = agent_id
    n = len(all_positions)
    
    # ✅ 必须包含：
    # 1. 自身状态
    self_state = np.concatenate([all_positions[i], all_velocities[i]])  # 6维
    
    # 2. 目标信息
    direction = goals[i] - all_positions[i]
    dist = np.linalg.norm(direction) + 1e-6
    goal_info = np.concatenate([direction / dist, [dist]])  # 4维
    
    # 3. K近邻信息（节省维度，不用全连接）
    neighbor_k = 3  # 只看最近3架
    rel_states = []
    for j in range(n):
        if i != j:
            rel_states.append((
                np.linalg.norm(all_positions[i] - all_positions[j]),
                j
            ))
    rel_states.sort(key=lambda x: x[0])
    
    neighbor_info = []
    for dist_j, j in rel_states[:neighbor_k]:
        rel_pos = all_positions[j] - all_positions[i]
        rel_vel = all_velocities[j] - all_velocities[i]
        neighbor_info.extend([dist_j] + list(rel_pos) + list(rel_vel))  # 7维/邻居
    
    # padding
    while len(neighbor_info) < neighbor_k * 7:
        neighbor_info.extend([0.0] * 7)
    
    return np.concatenate([self_state, goal_info, neighbor_info])
```

### 3.3 アクションスペースの設計

|アクションの種類 |寸法 |例 |利点 |デメリット |
|-------|------|------|------|------|
| **直接速度設定** | 2-3 | $[v_x, v_y]$ |シンプル、RL フレンドリー | $\|\mathbf{v}\| を確保するには後処理が必要です\leq v_{max}$ |
| **速度増加** | 2-3 | $[\Delta v_x, \Delta v_y]$ |インクリメンタル制御がよりスムーズ |速度制限を超える可能性があります |
| **ヘディング角度 + 速度** | 2 | $[\psi, v]$ |直感に従って |極座標変換により特異点が導入される |
| **目標位置** | 2-3 | $[x_{ターゲット}, y_{ターゲット}]$ |人間の指示に最も近い |パス追跡バックエンドが必要 |
| **推力ベクトル** | 4 | $[T_1,T_2,T_3,T_4]$ |最低レベルのコントロール |高次元、RLは習得が難しい |

**主流の選択肢 (MARL 論文)**: 速度増分 $(\Delta v_x, \Delta v_y, \Delta v_z)$、$\tanh$ アクティベーション + クリップ制限あり。

### 3.4 競合の定義と測定

**決定論的競合 (最も一般的に使用される)**:

$$
c_{ij} = \begin{件}
1 & \text{if } \| \mathbf{p}_i - \mathbf{p}_j \| < d_{安全} \\
0 & \text{そうでない場合}
\end{件}
$$**確率的矛盾 (不確実性を考慮)**:

$$
P_{競合} = P\left( \| \mathbf{p}_i - \mathbf{p}_j \| < d_{safe} \right) = \int_{\| \mathbf{d} \| < d_{safe}} f_{\Delta \mathbf{p}}(\mathbf{d}) \, d\mathbf{d}
$$

ここで、$f_{\Delta \mathbf{p}}$ は相対位置の同時確率密度 (通常はガウスであると想定されます) です。

**衝突までの時間 (TTC)**: 衝突が発生するまでの残り時間を予測します。 TTC が小さいほど、紛争の緊急性は高くなります。

$$
TTC = \min_{t > 0} \{ t \mid \| \mathbf{p}_i(t) - \mathbf{p}_j(t) \| = d_{安全} \}
$$

### 3.5 報酬関数の設計原則

報酬関数は RL 環境設計の核心であり、学習された戦略的行動を直接決定します。

#### 原則 1: サブアイテムの報酬 > 単一の報酬

```python
def compute_reward(self, positions, velocities, goals, prev_distances):
    """
    分项奖励设计（推荐）
    """
    r_total = np.zeros(self.n_agents)
    
    # 1. 任务奖励（正向引导）
    dist_to_goal = np.linalg.norm(positions - goals, axis=1)
    progress = prev_distances_to_goal - dist_to_goal
    r_task = 5.0 * progress
    
    # 2. 安全奖励（负向惩罚）
    r_safe = np.zeros(self.n_agents)
    for i in range(self.n_agents):
        for j in range(i+1, self.n_agents):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < self.safe_radius:
                r_safe[i] -= 100
                r_safe[j] -= 100
            elif d < self.safe_radius * 3:  # 告警区
                penalty = (self.safe_radius * 3 - d) / (self.safe_radius * 2)
                r_safe[i] -= 10 * penalty
                r_safe[j] -= 10 * penalty
    
    # 3. 效率奖励（偏好直线/短路径）
    r_efficiency = np.zeros(self.n_agents)
    for i in range(self.n_agents):
        ideal_dist = np.linalg.norm(goals[i] - self.init_positions[i])
        actual_dist = dist_to_goal[i]
        r_efficiency[i] = 1.0 - (actual_dist / (ideal_dist + 1e-6))
    
    # 4. 能量惩罚（避免无谓机动）
    r_energy = -0.01 * np.sum(np.square(velocities), axis=1)
    
    r_total = r_task + r_safe + r_efficiency + r_energy
    
    return r_total
```

#### 原則 2: 報酬が少ない + シェーピング

純粋にまばらな報酬 (衝突のみ -100、ゴール到達で +100) を高次元の状態空間でトレーニングすることはほぼ不可能です。 **密なボーナス + 疎なボーナス** がベスト プラクティスです。

- **ベースレイヤー**: 密な距離の進行報酬 (各ステップでのフィードバック付き)
- **スパースレイヤー**: 衝突ペナルティ + リーチ報酬 (キーイベント)

#### 原則 3: 公平性の制約

「ヒーロー UAV」現象を回避します。特定の UAV は常に道を譲り、他の UAV はまったく学習しません。

$$
\hat{r}_i = r_i - \lambda \cdot \text{Var}(\{r_j\}_{j \in \mathcal{N}_i})
$$

または、各エピソードの終了時に、各 UAV の「譲歩の数」をカウントすることができ、より多くの譲歩を行った UAV は追加の報酬を受け取ります。

---

## 4. ベンチマーク シナリオ

### 4.1 論文内の古典的なシーン**シナリオ 1 — 交差点**
同じ空域を反対方向から横断する 2 つの UAV チーム:

```
队1: (0, -100) → (0, +100)    [N 架，纵向]
队2: (-100, 0) → (+100, 0)   [M 架，横向]
```

古典的な競合シナリオ。競合の特定と意思決定のタイミングをテストします。

**シナリオ 2 — 循環追跡**
複数の UAV が同じ円形の軌道に沿って時計回りに飛行し、後続の航空機が先頭の航空機に追いつきます。

$$
x_i = R\cos(\theta_i)、\quad y_i = R\sin(\theta_i)
$$

$$
\theta_i = \theta_0 + \omega t + i \cdot \Delta\theta
$$

速度調整と間隔維持をテストするために使用されます。

**シナリオ 3 — 収束**
複数の UAV が異なる方向から同じ目標点に向かって飛行します。

$$
\mathbf{p}_i(0) = R \cdot [\cos(\phi_i), \sin(\phi_i)], \quad \phi_i = \frac{2\pi i}{N}
$$

最も厳しいシナリオでは、大規模なクラスターの消化機能をテストします。

**シナリオ 4 — 動的障害物回避**
静止または移動する障害物が飛行エリア全体に点在しています。

```python
class DynamicObstacleScenario:
    def __init__(self, n_obstacles=10, n_uavs=8):
        self.obstacles = []
        for _ in range(n_obstacles):
            self.obstacles.append({
                'position': np.random.uniform(-100, 100, size=3),
                'velocity': np.random.uniform(-5, 5, size=3),  # 移动障碍
                'radius': np.random.uniform(3, 10)             # 半径
            })
    
    def update(self, dt):
        for obs in self.obstacles:
            obs['position'] += obs['velocity'] * dt
            # 边界反弹
            for dim in range(3):
                if abs(obs['position'][dim]) > 100:
                    obs['velocity'][dim] *= -1
```

### 4.2 難易度の評価

|難易度 | UAV の数 |紛争密度 |動的障害物 |不確実性 |
|------|--------|----------|----------|---------|
| **簡単** | 2–4 |低い (予測可能) |なし |なし |
| **中** | 5–12 |中 (調整が必要) |オプション | GPS ノイズ |
| **ハード** | 20～50 |高 (配布する必要があります) |はい | GPS+風の乱れ |
| **極端な** | 100+ |非常に高い |複雑な |完全な不確実性 |

### 4.3 評価指標体系

「競合率」だけを指標として使用するのではなく、完全な評価システムを確立する必要があります。

```python
def evaluate_policy(env, policy, n_episodes=100):
    """
    多维度评估指标
    """
    metrics = {
        'collision_rate': [],       # 碰撞率
        'success_rate': [],          # 任务完成率（所有UAV到达目标）
        'avg_episode_length': [],    # 平均 episode 长度
        'avg_dist_to_goal': [],      # 最终到目标平均距离
        'avg_jerk': [],               # 平均加加速度（平滑度）
        'communications': [],         # 通信次数（分布式场景）
        'fairness_index': [],         # Jain公平性指数
    }
    
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        step_count = 0
        total_reward = 0
        
        while not done and step_count < env.max_steps:
            actions = policy.select_action(obs)
            obs, rewards, done, _ = env.step(actions)
            total_reward += rewards.mean()
            step_count += 1
        
        # 计算指标
        metrics['collision_rate'].append(1 if env._check_collision() else 0)
        metrics['success_rate'].append(1 if all(
            np.linalg.norm(env.positions[i] - env.goals[i]) < 2.0
            for i in range(env.n_agents)
        ) else 0)
        metrics['avg_episode_length'].append(step_count)
        
        # Jain公平性指数（衡量各UAV任务分配公平程度）
        rewards_per_uav = [env.get_cumulative_reward(i) for i in range(env.n_agents)]
        n = len(rewards_per_uav)
        if sum(rewards_per_uav) > 0:
            fairness = sum(rewards_per_uav)**2 / (n * sum(r**2 for r in rewards_per_uav))
        else:
            fairness = 1.0
        metrics['fairness_index'].append(fairness)
    
    # 汇总
    summary = {k: np.mean(v) for k, v in metrics.items()}
    return summary
```

---

## 5. PyMARL フレームワークでの環境登録**PyMARL** (サウサンプトン大学) は、MARL 研究のための標準化されたフレームワークです。カスタム環境を登録します。

```python
# pysocialwatcher/utils/registry.py

REGISTRY = {}

from pysocialwatcher.agents.rnn_agent import RNNAgent
from pysocialwatcher.runners import runners

# 注册自定义环境
REGISTRY['uav_conflict'] = {
    'env_class': UAVConflictGym,
    'state_dim': 37,          # obs_dim
    'n_actions': 3,           # action_dim
    'n_agents': 8,            # UAV 数量
    'episode_limit': 500,     # 最大步数
}

# 运行示例
# python3 -m pysocialwatcher.main config=baselines/maddpg/uav_conflict.yaml
```

```yaml
# baselines/maddpg/uav_conflict.yaml
env:
  name: uav_conflict
  n_agents: 8
  area_size: 200
  safe_radius: 5.0
  max_speed: 20.0
  seed: 42

algorithm:
  name: MADDPG
  actor_lr: 1.0e-3
  critic_lr: 1.0e-3
  gamma: 0.95
  tau: 0.01
  hidden_dim: 64

training:
  n_episodes: 100000
  batch_size: 256
  replay_buffer: 100000
  target_update_freq: 200
```

---

## 6. リアル物理エンジンへのアクセス: シミュレーションから実測まで

### 6.1 PX4 SITL アクセス

PX4 は、ジム環境のダイナミクス モデルを直接置き換えることができる完全なソフトウェアインザループ シミュレーション (SITL) を提供します。

```bash
# 启动 PX4 SITL + ROS
cd /path/to/PX4-Autopilot
make px4_sitl_default gazebo

# ROS 主题订阅
rostopic echo /drone_0/mavros/local_position/pose

# Python 控制接口
from pymavlink import mavutil

def set_velocity(master, vx, vy, vz):
    master.mav.set_position_target_local_ned_send(
        0, 0, 0,  # time_boot_ms, target_system, target_component
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        0x0C00,   # type_mask: 仅速度
        0, 0, 0,  # x, y, z 位置（被忽略）
        vx, vy, vz,  # 速度 m/s
        0, 0, 0,  # 加速度
        0, 0      # 偏航, yaw_rate
    )
```

### 6.2 Sim と Real のギャップの一般的な原因

動的モデルが完全に一貫していても、シミュレーションと実際の移行の間には依然として大きなギャップがあります。

|ギャップの原因 |シミュレーション |現実 |影響 |
|-------|------|------|------|
| **モーターの応答遅延** |瞬時 | 20～50ミリ秒 |高速操縦時の軌道誤差 |
| **GPS ノイズ** |理想 | 1 ～ 5m の誤差 |位置推定オフセット |
| **風の乱れ** |オプションのモデリング |連続ランダム |横方向のドリフト |
| **物理パラメータ** |公称値 |温度・電力による変化 |推力係数の変化 |
| **通信遅延** | 0ミリ秒 | 5～100ミリ秒 |協調的な意思決定の失敗 |

**ドメインのランダム化** は、Sim と Real のギャップを解決するための主流の方法です。

```python
class DomainRandomizedEnv(UAVConflictGym):
    """域随机化：训练时随机化物理参数"""
    
    def reset(self):
        # 随机化物理参数
        self.mass = np.random.uniform(0.8, 1.2)    # 质量 ±20%
        self.thrust_coeff = np.random.uniform(0.9, 1.1)  # 推力系数
        self.drag_coeff = np.random.uniform(0.01, 0.05)   # 阻力系数
        self.gps_noise = np.random.uniform(0.5, 3.0)     # GPS噪声 m
        self.wind = np.random.uniform(-3, 3, size=3)      # 风速 m/s
        
        return super().reset()
```

---

## 7. 概要: 環境構築チェックリスト

認定された UAV 競合シミュレーション環境は、以下を満たしている必要があります。

- ✅ **力学リアリズム**: 速度/加速度の制約を考慮して、少なくとも双方向積分器を使用します。
- ✅ **明確に定義された競合**: 安全な距離 $d_{safe}$ には物理的な根拠があります (通常は 5 ～ 10 メートル)
- ✅ **報酬機能は説明可能**: サブ報酬、まばらなボーナス、報酬ハッキングの回避
- ✅ **制御されたランダム性**: 固定シードの下での結果は完全に再現可能
- ✅ **完全評価指標**: 衝突率 + タスク完了率 + 公平性 + スムーズさ
- ✅ **ベースラインシナリオを再現可能**: ルートの交差/追いつき/収束のシナリオが明確に記述されています
- ✅ **インターフェイスの標準化**: Gym PettingZoo 互換または PyMARL 登録形式この記事のコード例は、Gym カスタム環境から AirSim/Gazebo プラットフォームに至るテクノロジー スタック全体をカバーしており、研究目標に応じて自由に組み合わせることができます。

---

**参考文献:**1. Shah、S.、他。 （2018年）。 *AirSim: 自動運転車向けの高忠実度の視覚的および物理的シミュレーション。* フィールド アンド サービス ロボティクス (FSR)、Springer。
2. Zhou、M.、他。 （2019年）。 *乱雑な環境における UAV の経路計画に関する調査。* 高度道路交通システム (T-ITS) に関する IEEE トランザクション。
3. Everett、M.、他。 （2021年）。 *深層強化学習による密集した交通における衝突回避* IEEE ロボティクスとオートメーションに関する国際会議 (ICRA)。
4. アロンソ・モーラ、J.、他。 （2018年）。 *複数車両システム向けの最適化ベースの衝突回避。* IEEE Transactions on Robotics (TRO)。
5. van den Berg, J.、Lin, M.、および Manocha, D. (2008)。 *リアルタイム マルチエージェント ナビゲーションのための相互速度障害物。* IEEE ロボティクスとオートメーションに関する国際会議 (ICRA)。
6. リチャーズ、A.、ハウ、JP (2002)。 *混合整数線形計画法を使用した衝突回避を伴う航空機の軌道計画* AIAA ガイダンce、ナビゲーションおよびコントロール会議 (GNC)。
7. ファン、T.、他。 （2020年）。 *複雑なシナリオでのナビゲーションのための深層強化学習による分散型マルチロボット衝突回避* 国際ロボット研究ジャーナル (IJRR)。
8. ヤン、C.、他。 (2025年)。 *スケーラブルな固定翼 UAV 艦隊の衝突回避を伴う群集のための時空間的注意を伴うマルチエージェント強化学習。* 高度道路交通システム (T-ITS) に関する IEEE トランザクション。
9. Yu、L.、他。 (2025年)。 *空中回廊における複数の無人航空機の調整のためのハイブリッド変圧器ベースのマルチエージェント強化学習。* モバイル コンピューティング (TMC) に関する IEEE トランザクション。
10. Huo、D.、他。 （2023年）。 *障害物環境における UAV の無衝突モデル予測軌道追跡制御。* 航空宇宙および電子システムに関する IEEE トランザクション (TAES)。
11. Jiang、C.、他。 （2024年）。 *分散サンプリングベースのモデル予測制御 via マルチロボットフォーメーションナビゲーションのための信念伝播。* IEEE Robotics and Automation Letters (RA-L)。