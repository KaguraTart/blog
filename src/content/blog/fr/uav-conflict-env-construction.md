---
title: "Construction d'un environnement de simulation de conflits de trajectoire de drone : de la pratique papier à l'implémentation du code"
description: "Examiner systématiquement la méthode de construction de l'environnement de simulation de scénarios de conflit multi-avions pour drones, couvrant la comparaison des plates-formes de simulation traditionnelles, la modélisation dynamique, la conception de l'espace d'état, la définition des conflits, la construction de fonctions de récompense et les scénarios de test de référence, avec des exemples complets de multi-framework Gym/Gazebo/AirSim."
pubDate: 2026-04-07T11:34:43+08:00
tags: ["drone", "environnement de simulation", "multi-agent", "planification du chemin", "Salle de sport", "Belvédère", "AirSim", "apprentissage par renforcement"]
category: Tech
---

# Construction d'un environnement de simulation de conflits de trajectoire de drone : de la pratique papier à l'implémentation du code

> Dans les deux premiers articles, nous avons trié respectivement [Panorama de l'algorithme de résolution des conflits UAV](/blog/uav-conflict-resolution/) et [MARL+GAT end-to-end solution](/blog/marl-kat-uav-conflict/). Cet article répondra à une question de niveau inférieur : **Comment construire un scénario de conflit de drones réaliste et reproductible dans un environnement de simulation ? ** Cela détermine directement la crédibilité de l'évaluation de l'algorithme et la comparabilité de l'expérience.

---

## 1. Pourquoi la construction d'un environnement de simulation est-elle si critique ?

L’environnement de simulation constitue un pont reliant la théorie des algorithmes et la mesure du vol. Un environnement de simulation bien conçu doit répondre à trois dimensions :

| Dimensions | Signification | FAQ |
|------|------|--------------|
| **Authenticité** | L'écart entre simulation et réalité | Simplification excessive de la dynamique et manque de modèles de capteurs |
| **Reproductibilité** | Des résultats cohérents sous la même graine | Initialisation aléatoire incontrôlée, accumulation d'erreurs en virgule flottante |
| **Évolutivité** | Prise en charge de clusters à grande échelle | La simulation sur une seule machine ne peut pas être parallélisée et la topologie de communication est manquante |

Cet article se concentre sur la construction de l'environnement de simulation dans le cadre du **scénario de formation MARL**, en tenant compte à la fois de l'authenticité et de l'évolutivité - après tout, il est irréaliste d'entraîner un véritable contrôle de vol sur plus de 50 drones.

---

## 2. Comparaison des plateformes de simulation grand public

### 2.1 Présentation de la plateforme| Plateforme | Moteur sous-jacent | Authenticité physique | Prise en charge multi-machines | Compatibilité MARL | Citations papier |
|------|----------|-----------|--------------|------------|---------------|
| **AirSim** (Microsoft) | Moteur irréel | ⭐⭐⭐⭐⭐ | ✅ | ⚠️ Nécessite un emballage | Élevé |
| **FlightGoggles** (MIT) | Unité | ⭐⭐⭐⭐ | ✅ | ⚠️ Nécessite un emballage | Élevé |
| **RotorS** (ETH Zurich) | Tonnelle/ROS | ⭐⭐⭐⭐ | ✅ | ✅ Compatible PyMARL | Moyen |
| **Morse** (LAAS-CNRS) | Moteur de jeu Blender | ⭐⭐⭐ | ✅ | ✅Autochtone | Moyen |
| **Webots** | Auto-développé | ⭐⭐⭐⭐ | ✅ | ✅ | Faible |
| **Personnalisé 2D/3D** | N/A | ⭐⭐ | ✅✅✅ | ✅✅✅ | - |

**Conclusion principale :**
- **Academic Benchmarks** : RotorS + ROS/Gazebo est le choix grand public (produit par l'ETH Zurich, compatible avec PX4)
- **Recherche RL de bout en bout** : la plupart des articles utilisent **Environnement Custom Gym** (entièrement personnalisé, fonction de récompense facile à modifier)
- **Sim-to-Real Target** : AirSim/FlightGoggles (vision haute fidélité + bruit du capteur)

### 2.2 AirSim : faire évoluer l'expérience de zéro à cent

L'équipe Microsoft AI for Earth a formé la coordination en essaim de jusqu'à **100 drones à voilure fixe** sur AirSim :

> Shah, S., et coll. (2018). *AirSim : simulation visuelle et physique haute fidélité pour les véhicules autonomes.* Robotique de terrain et de service.Principaux avantages d'AirSim :
- **Simulation de capteur** : simulation complète de caméra, LiDAR, GPS, IMU, du bruit peut être injecté
- **Système météo** : impact de la vitesse du vent, des nuages et de la lumière sur le capteur
- **Unification API** : prise en charge de l'interface C++ / Python / ROS

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

### 2.3 Gazebo / RotorS : le choix standard dans l'écosystème ROS

Le **RotorS** de l'ETH Zurich est le package de simulation multi-UAV le plus complet de l'écosystème ROS/Gazebo :

> Foehn, P., et al. (2020). *Ensemble de données de l'ETHZ Flying Machine Arena.* ETH Zurich.

RotorS prend en charge :
- Interface **mav_msgs** (spin + tangage + roulis + poussée → moteur PWM)
- **PX4 SITL/HITL** : firmware complet des commandes de vol dans la boucle
- **Visualisation RViz** : Observez l'état du cluster en temps réel
- **Gazebo Physics Engine** : ODE / Bullet / Simbody en option

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

### 2.4 Environnement de gym personnalisé : le choix courant pour la recherche universitaire en RL

La grande majorité des articles MARL UAV (y compris les implémentations originales de MADDPG/QMIX/MAPPO) utilisent un environnement de style Gym entièrement personnalisé pour trois raisons :

1. **Vitesse d'entraînement** : pas de rendu graphique, boucle directe Python, des dizaines de milliers d'interactions peuvent être effectuées par seconde
2. **Fonction de récompense contrôlable** : itérez rapidement la conception sans modifier le code C++
3. **Reproductibilité** : Avec une graine aléatoire fixe, les résultats sont entièrement reproductibles.

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

## 3. Éléments fondamentaux de la conception environnementale

### 3.1 Modèle dynamique : du modèle particulaire à six degrés de liberté

**Niveau 1 — Masse ponctuelle**
La version la plus simplifiée, qui ne prend en compte que les mises à jour de position et de vitesse, est utilisée pour la vérification des algorithmes et l'entraînement rapide :

$$
\dot{\mathbf{p}} = \mathbf{v}, \quad \dot{\mathbf{v}} = \mathbf{a}
$$$$
\mathbf{a}_{nouveau} = \text{clip}(\mathbf{a}_{old} + \mathbf{u}, \mathbf{v}_{min}, \mathbf{v}_{max})
$$

**Niveau 2 — Intégrateur unique**
Contrôler directement la vitesse, en supposant que l'actionneur réponde instantanément :

$$
\mathbf{v}_i[k+1] = \mathbf{u}_i[k]
$$

**Niveau 3 — Double Intégrateur**
Compte tenu de la limite d'accélération, il est plus proche de l'avion réel :

$$
\mathbf{v}_i[k+1] = \mathbf{v}_i[k] + \mathbf{a}_i[k] \Delta t
$$
$$
\mathbf{p}_i[k+1] = \mathbf{p}_i[k] + \mathbf{v}_i[k] \Delta t + \frac{1}{2}\mathbf{a}_i[k]\Delta t^2
$$

$$
\|\mathbf{a}_i[k]\| \leq a_{max}, \quad \|\mathbf{v}_i[k]\| \leq v_{max}
$$

**Niveau 4 — Quadrotor 6-DOF**
Modèle non linéaire complet, prenant en compte les contraintes d'attitude et de poussée :

$$
\dot{\boldsymbol{\eta}} = \mathbf{R}(\boldsymbol{\Theta})\boldsymbol{\omega}
$$
$$
m\ddot{\mathbf{p}} = -mg\mathbf{e}_3 + \mathbf{R}(\boldsymbol{\Theta})\mathbf{T}_i\mathbf{e}_3 - \mathbf{D}(\dot{\mathbf{p}})
$$

Où $\boldsymbol{\eta}$ est l'angle d'Euler, $\mathbf{R}$ est la matrice de rotation, $\mathbf{T}_i$ est la poussée du $i$-ième axe.| Modèle | Montant du calcul | Authenticité | Étape applicable |
|------|--------|--------|---------|
| Point de masse | ⭐ | ⭐ | Vérification rapide de l'algorithme |
| Intégrateur unique | ⭐⭐ | ⭐⭐ | Formation RL (choix grand public) |
| Intégrateur bidirectionnel | ⭐⭐ | ⭐⭐⭐ | Un vol proche du vrai |
| 6-DOF | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Pré-vérification Sim-à-Réel |

### 3.2 Conception de l'espace d'état

Un bon espace d'état doit contenir toutes les informations nécessaires à la prise de décision et avoir des dimensions contrôlables :

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

### 3.3 Conception de l'espace d'action

| Types d'actions | Dimensions | Exemples | Avantages | Inconvénients |
|--------------|------|------|------|------|
| **Réglage direct de la vitesse** | 2-3 | $[v_x, v_y]$ | Simple, convivial pour RL | Post-traitement requis pour garantir que $\|\mathbf{v}\| \leq v_{max}$ |
| **Incrément de vitesse** | 2-3 | $[\Delta v_x, \Delta v_y]$ | Le contrôle incrémental est plus fluide | Peut dépasser les contraintes de vitesse |
| **Angle de cap + vitesse** | 2 | $[\psi,v]$ | Conformément à l'intuition | La transformation des coordonnées polaires introduit la singularité |
| **Position cible** | 2-3 | $[x_{cible}, y_{cible}]$ | Le plus proche des instructions humaines | Nécessite un backend de suivi de chemin |
| **Vecteur de poussée** | 4 | $[T_1,T_2,T_3,T_4]$ | Le contrôle de niveau le plus bas | Haute dimension, RL est difficile à apprendre |

**Choix grand public (article MARL)** : incrément de vitesse $(\Delta v_x, \Delta v_y, \Delta v_z)$, avec activation $\tanh$ + limitation de clip.

### 3.4 Définition et mesure des conflits

**Conflit déterministe (le plus couramment utilisé)** :

$$
c_{ij} = \begin{cases}
1 & \text{if } \| \mathbf{p}_i - \mathbf{p}_j \| < d_{coffre-fort} \\
0 & \text{sinon}
\fin{cas}
$$**Conflit probabiliste (compte tenu de l'incertitude)** :

$$
P_{conflit} = P\left( \| \mathbf{p}_i - \mathbf{p}_j \| < d_{safe} \right) = \int_{\| \mathbf{d} \| < d_{safe}} f_{\Delta \mathbf{p}}(\mathbf{d}) \, d\mathbf{d}
$$

où $f_{\Delta \mathbf{p}}$ est la densité de probabilité conjointe des positions relatives (généralement supposée être gaussienne).

**Time-to-Conflict (TTC)** : prédisez le temps restant avant qu'une collision ne se produise. Plus le TTC est petit, plus le conflit est urgent :

$$
TTC = \min_{t > 0} \{ t \mid \| \mathbf{p}_i(t) - \mathbf{p}_j(t) \| = d_{coffre-fort} \}
$$

### 3.5 Principes de conception des fonctions de récompense

La fonction de récompense est l’âme de la conception de l’environnement RL et détermine directement le comportement stratégique appris.

#### Principe 1 : Récompenses de sous-éléments > Récompense unique

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

#### Principe 2 : Récompense clairsemée + mise en forme

Les récompenses purement rares (-100 uniquement pour les collisions et +100 pour atteindre l'objectif) sont presque impossibles à entraîner dans des espaces d'états de grande dimension. **Bonus dense + bonus clairsemé** est la meilleure pratique :

- **Base Layer** : récompenses de progression à distance dense (avec retour d'information à chaque étape)
- **Couche clairsemée** : pénalité de collision + récompense de portée (événement clé)

#### Principe 3 : Contraintes d'équité

Evitez le phénomène des « drones héros » : certains drones cèdent toujours et d'autres n'apprennent pas du tout :

$$
\hat{r}_i = r_i - \lambda \cdot \text{Var}(\{r_j\}_{j \in \mathcal{N}_i})
$$

Ou à la fin de chaque épisode, le « nombre de concessions » de chaque drone peut être compté, et le drone avec le plus de concessions recevra des récompenses supplémentaires.

---

## 4. Scénarios de référence

### 4.1 Scènes classiques du journal**Scénario 1 — Traversée**
Deux équipes de drones traversant le même espace aérien dans des directions opposées :

```
队1: (0, -100) → (0, +100)    [N 架，纵向]
队2: (-100, 0) → (+100, 0)   [M 架，横向]
```

Scénarios de conflit classiques, testant l'identification des conflits et la prise de décision temporelle.

**Scénario 2 — Poursuite circulaire**
Plusieurs drones volent dans le sens des aiguilles d'une montre le long de la même piste circulaire, l'avion suivant rattrapant celui de tête :

$$
x_i = R\cos(\theta_i), \quad y_i = R\sin(\theta_i)
$$

$$
\theta_i = \theta_0 + \omega t + i \cdot \Delta\theta
$$

Utilisé pour tester la coordination de la vitesse et le maintien de l'espacement.

**Scénario 3 — Convergence**
Plusieurs drones volent vers le même point cible depuis des directions différentes :

$$
\mathbf{p}_i(0) = R \cdot [\cos(\phi_i), \sin(\phi_i)], \quad \phi_i = \frac{2\pi i}{N}
$$

Le scénario le plus grave consiste à tester les capacités de digestion de clusters à grande échelle.

**Scénario 4 — Évitement dynamique d'obstacles**
Des obstacles fixes ou mobiles sont dispersés dans toute la zone de vol :

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

### 4.2 Cote de difficulté

| Difficulté | Nombre de drones | Densité des conflits | Obstacles dynamiques | Incertitude |
|------|---------|---------|---------|---------|
| **Facile** | 2–4 | Faible (prévisible) | Aucun | Aucun |
| **Moyen** | 5-12 | Moyen (nécessite une coordination) | Facultatif | Bruit GPS |
| **Dur** | 20-50 | Élevé (doit être distribué) | Oui | GPS+Perturbation du vent |
| **Extrême** | 100+ | Extrêmement élevé | Complexe | Incertitude totale |

### 4.3 Système d'indicateurs d'évaluation

Au lieu d’utiliser simplement le « taux de conflits » comme indicateur, un système d’évaluation complet doit être mis en place :

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

## 5. Enregistrement de l'environnement sous le framework PyMARL**PyMARL** (Université de Southampton) est un cadre standardisé pour la recherche MARL. Enregistrez un environnement personnalisé :

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

## 6. Accès au moteur physique réel : de la simulation à la mesure réelle

### 6.1 Accès SITL PX4

PX4 fournit une simulation logicielle dans la boucle (SITL) complète qui peut remplacer directement le modèle dynamique de l'environnement du gymnase :

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

### 6.2 Sources courantes d'écarts entre le Sim et le réel

Même si le modèle dynamique est tout à fait cohérent, il existe encore un écart important entre la simulation et la migration réelle :

| Source de l'écart | Simulations | Réalité | Impact |
|---------|------|------|------|
| **Délai de réponse du moteur** | Instantané | 20 à 50 ms | Erreur de trajectoire lors de manœuvres à grande vitesse |
| **Bruit GPS** | Idéal | Erreur 1 à 5 m | Compensation d'estimation de position |
| **Perturbation du vent** | Modélisation facultative | Aléatoire continu | Dérive latérale |
| **Paramètres physiques** | Valeur nominale | Changements avec température/électricité | Modifications du coefficient de poussée |
| **Délai de communication** | 0 ms | 5 à 100 ms | Échec de la prise de décision collaborative |

La **randomisation de domaine** est la méthode courante pour résoudre l'écart entre la simulation et la réalité :

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

## 7. Résumé : liste de contrôle de création d'environnement

Un environnement de simulation de conflits de drones qualifié doit répondre :

- ✅ **Réalisme dynamique** : utiliser au minimum un intégrateur bidirectionnel, prenant en compte les contraintes de vitesse/accélération
- ✅ **Conflit bien défini** : la distance de sécurité $d_{safe}$ a une base physique (généralement 5 à 10 m)
- ✅ **La fonction de récompense peut être expliquée** : sous-récompense, bonus clairsemé, évitez le piratage des récompenses
- ✅ **Caractère aléatoire contrôlé** : les résultats sous une graine fixe peuvent être entièrement reproduits
- ✅ **Indicateurs d'évaluation complets** : taux de collision + taux d'achèvement des tâches + équité + fluidité
- ✅ **Les scénarios de référence peuvent être reproduits** : Les scénarios de croisement / rattrapage / convergence sont clairement décrits
- ✅ **Standardisation de l'interface** : compatible Gym PettingZoo ou format d'inscription PyMARLLes exemples de code contenus dans cet article couvrent l'ensemble de la pile technologique, de l'environnement personnalisé Gym à la plateforme AirSim/Gazebo, et peuvent être librement combinés en fonction des objectifs de recherche.

---

**Références :**1. Shah, S., et al. (2018). *AirSim : simulation visuelle et physique haute fidélité pour les véhicules autonomes.* Field and Service Robotics (FSR), Springer.
2. Zhou, M., et al. (2019). *Une enquête sur la planification de trajectoire pour les drones dans des environnements encombrés.* Transactions IEEE sur les systèmes de transport intelligents (T-ITS).
3. Everett, M. et coll. (2021). *Évitement des collisions dans un trafic dense grâce à un apprentissage par renforcement profond.* Conférence internationale IEEE sur la robotique et l'automatisation (ICRA).
4. Alonso-Mora, J. et al. (2018). *Évitement des collisions basé sur l'optimisation pour les systèmes multi-véhicules.* Transactions IEEE sur la robotique (TRO).
5. van den Berg, J., Lin, M. et Manocha, D. (2008). *Obstacles à vitesse réciproque pour la navigation multi-agents en temps réel.* Conférence internationale IEEE sur la robotique et l'automatisation (ICRA).
6. Richards, A. et How, JP (2002). *Planification de trajectoire d'avion avec évitement de collision à l'aide d'une programmation linéaire mixte en nombres entiers.* AIAA Guidance, conférence de navigation et de contrôle (GNC).
7. Fan, T. et coll. (2020). *Évitement distribué des collisions multi-robots via un apprentissage par renforcement profond pour la navigation dans des scénarios complexes.* The International Journal of Robotics Research (IJRR).
8. Yan, C. et coll. (2025). *Apprentissage par renforcement multi-agents avec attention spatio-temporelle pour le flocage avec prévention des collisions d'une flotte évolutive de drones à voilure fixe.* Transactions IEEE sur les systèmes de transport intelligents (T-ITS).
9. Yu, L. et coll. (2025). *Apprentissage par renforcement multi-agents basé sur un transformateur hybride pour la coordination de plusieurs véhicules aériens non pilotés dans les couloirs aériens.* Transactions IEEE sur l'informatique mobile (TMC).
10. Huo, D. et coll. (2023). *Contrôle prédictif de suivi de trajectoire de modèle sans collision pour les drones dans un environnement d'obstacles.* Transactions IEEE sur les systèmes aérospatiaux et électroniques (TAES).
11. Jiang, C. et coll. (2024). *Contrôle prédictif de modèle basé sur l'échantillonnage distribué via Propagation des croyances pour la navigation en formation multi-robots.* Lettres de robotique et d'automatisation IEEE (RA-L).