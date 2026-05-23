---
title: "Aufbau einer UAV-Pfadkonflikt-Simulationsumgebung: von der Papierpraxis bis zur Code-Implementierung"
description: "Überprüfen Sie systematisch die Konstruktionsmethode der Simulationsumgebung für UAV-Konfliktszenarien mit mehreren Flugzeugen, einschließlich Vergleich der gängigen Simulationsplattformen, Dynamikmodellierung, Zustandsraumdesign, Konfliktdefinition, Konstruktion von Belohnungsfunktionen und Benchmark-Testszenarien, mit vollständigen Beispielen des Multi-Frameworks Gym/Gazebo/AirSim"
pubDate: 2026-04-07T11:34:43+08:00
tags: ["Drohne", "Simulationsumgebung", "Multi-Agent", "Wegplanung", "Fitnessstudio", "Pavillon", "AirSim", "Verstärkungslernen"]
category: Tech
---

# Aufbau einer UAV-Pfadkonflikt-Simulationsumgebung: von der Papierpraxis bis zur Code-Implementierung

> In den ersten beiden Artikeln haben wir [Panorama des UAV-Konfliktlösungsalgorithmus](/blog/uav-conflict-resolution/) bzw. [MARL+GAT-End-to-End-Lösung](/blog/marl-kat-uav-conflict/) herausgefunden. In diesem Artikel wird eine untergeordnete Frage beantwortet: **Wie erstellt man ein realistisches und reproduzierbares UAV-Konfliktszenario in einer Simulationsumgebung? ** Dies bestimmt direkt die Glaubwürdigkeit der Algorithmusbewertung und die Vergleichbarkeit des Experiments.

---

## 1. Warum ist der Aufbau einer Simulationsumgebung so wichtig?

Die Simulationsumgebung ist eine Brücke zwischen Algorithmentheorie und Flugmessung. Eine gut gestaltete Simulationsumgebung muss drei Dimensionen erfüllen:

| Abmessungen | Bedeutung | FAQ |
|------|------|---------|
| **Authentizität** | Die Kluft zwischen Simulation und Realität | Zu starke Vereinfachung der Dynamik und fehlende Sensormodelle |
| **Reproduzierbarkeit** | Konsistente Ergebnisse unter demselben Startwert | Unkontrollierte zufällige Initialisierung, Anhäufung von Gleitkommafehlern |
| **Skalierbarkeit** | Unterstützung großer Cluster | Die Einzelmaschinensimulation kann nicht parallelisiert werden und die Kommunikationstopologie fehlt |

Dieser Artikel konzentriert sich auf den Aufbau der Simulationsumgebung im Rahmen des **MARL-Trainingsszenarios** und berücksichtigt dabei sowohl Authentizität als auch Skalierbarkeit – schließlich ist es unrealistisch, echte Flugsteuerung auf mehr als 50 Drohnen zu trainieren.

---

## 2. Vergleich gängiger Simulationsplattformen

### 2.1 Plattformübersicht| Plattform | Zugrunde liegender Motor | Physische Authentizität | Unterstützung mehrerer Maschinen | MARL-Kompatibilität | Papierzitate |
|------|----------|-----------|----------|------------|-----------|
| **AirSim** (Microsoft) | Unreal Engine | ⭐⭐⭐⭐⭐ | ✅ | ⚠️ Erfordert Verpackung | Hoch |
| **FlightGoggles** (MIT) | Einheit | ⭐⭐⭐⭐ | ✅ | ⚠️ Erfordert Verpackung | Hoch |
| **RotorS** (ETH Zürich) | Pavillon/ROS | ⭐⭐⭐⭐ | ✅ | ✅ PyMARL-kompatibel | Mittel |
| **Morse** (LAAS-CNRS) | Blender Game Engine | ⭐⭐⭐ | ✅ | ✅ Einheimisch | Mittel |
| **Webbots** | Selbstentwickelt | ⭐⭐⭐⭐ | ✅ | ✅ | Niedrig |
| **Benutzerdefiniert 2D/3D** | N/A | ⭐⭐ | ✅✅✅ | ✅✅✅ | - |

**Kernschlussfolgerung:**
- **Akademische Benchmarks**: RotorS + ROS/Gazebo ist die Mainstream-Wahl (hergestellt von der ETH Zürich, kompatibel mit PX4)
- **Umfassende RL-Forschung**: Die meisten Artikel verwenden **Benutzerdefinierte Fitnessstudio-Umgebung** (vollständig angepasst, die Belohnungsfunktion lässt sich leicht ändern)
- **Sim-to-Real Target**: AirSim/FlightGoggles (High-Fidelity Vision + Sensorrauschen)

### 2.2 AirSim: Skalierung der Erfahrung von Null auf Hundert

Das Microsoft AI for Earth-Team trainierte die Schwarmkoordination von bis zu **100 Starrflügel-UAV** auf AirSim:

> Shah, S., et al. (2018). *AirSim: Hochpräzise visuelle und physikalische Simulation für autonome Fahrzeuge.* Feld- und Servicerobotik.Hauptvorteile von AirSim:
- **Sensorsimulation**: vollständige Simulation von Kamera, LiDAR, GPS, IMU, Rauschen kann eingespeist werden
- **Wettersystem**: Der Einfluss von Windgeschwindigkeit, Wolken und Licht auf den Sensor
- **API-Vereinheitlichung**: Unterstützt C++ / Python / ROS-Schnittstelle

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

### 2.3 Gazebo / RotorS: die Standardwahl im ROS-Ökosystem

**RotorS** der ETH Zürich ist das umfassendste Multi-UAV-Simulationspaket im ROS/Gazebo-Ökosystem:

> Foehn, P., et al. (2020). *ETHZ Flying Machine Arena Datensatz.* ETH Zürich.

RotorS unterstützt:
- **mav_msgs**-Schnittstelle (Spin + Pitch + Roll + Schub → Motor-PWM)
- **PX4 SITL/HITL**: komplette Flugsteuerungs-Firmware in der Schleife
- **RViz-Visualisierung**: Beobachten Sie den Clusterstatus in Echtzeit
- **Gazebo Physics Engine**: ODE / Bullet / Simbody optional

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

### 2.4 Benutzerdefinierte Fitnessstudio-Umgebung: die gängige Wahl für die akademische RL-Forschung

Die überwiegende Mehrheit der MARL-UAV-Papiere (einschließlich der ursprünglichen Implementierungen von MADDPG/QMIX/MAPPO) verwendet aus drei Gründen eine vollständig benutzerdefinierte Gym-ähnliche Umgebung:

1. **Trainingsgeschwindigkeit**: Kein Grafik-Rendering, Python-Direktschleife, Zehntausende Interaktionen können pro Sekunde abgeschlossen werden
2. **Steuerbare Belohnungsfunktion**: Iterieren Sie das Design schnell, ohne den C++-Code zu ändern
3. **Reproduzierbarkeit**: Mit einem festen Zufallsstartwert sind die Ergebnisse vollständig reproduzierbar.

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

## 3. Kernelemente des Umweltdesigns

### 3.1 Dynamisches Modell: vom Teilchenmodell zu sechs Freiheitsgraden

**Stufe 1 – Punktmasse**
Die einfachste Version, die nur Positions- und Geschwindigkeitsaktualisierungen berücksichtigt, wird zur Algorithmusüberprüfung und zum schnellen Training verwendet:

$$
\dot{\mathbf{p}} = \mathbf{v}, \quad \dot{\mathbf{v}} = \mathbf{a}
$$$$
\mathbf{a}_{new} = \text{clip}(\mathbf{a}_{old} + \mathbf{u}, \mathbf{v}_{min}, \mathbf{v}_{max})
$$

**Stufe 2 – Einzelintegrator**
Geschwindigkeit direkt steuern, vorausgesetzt, der Aktuator reagiert sofort:

$$
\mathbf{v}_i[k+1] = \mathbf{u}_i[k]
$$

**Stufe 3 – Doppelintegrator**
In Anbetracht der Beschleunigungsgrenze kommt es dem echten Flugzeug näher:

$$
\mathbf{v}_i[k+1] = \mathbf{v}_i[k] + \mathbf{a}_i[k] \Updelta t
$$
$$
\mathbf{p}_i[k+1] = \mathbf{p}_i[k] + \mathbf{v}_i[k] \Updelta t + \frac{1}{2}\mathbf{a}_i[k]\Updelta t^2
$$

$$
\|\mathbf{a}_i[k]\| \leq a_{max}, \quad \|\mathbf{v}_i[k]\| \leq v_{max}
$$

**Stufe 4 – 6-DOF-Quadrotor**
Vollständiges nichtlineares Modell unter Berücksichtigung von Lage- und Schubbeschränkungen:

$$
\dot{\boldsymbol{\eta}} = \mathbf{R}(\boldsymbol{\Theta})\boldsymbol{\omega}
$$
$$
m\ddot{\mathbf{p}} = -mg\mathbf{e}_3 + \mathbf{R}(\boldsymbol{\Theta})\mathbf{T}_i\mathbf{e}_3 - \mathbf{D}(\dot{\mathbf{p}})
$$

Dabei ist $\boldsymbol{\eta}$ der Euler-Winkel, $\mathbf{R}$ die Rotationsmatrix und $\mathbf{T}_i$ der Schub der $i$-ten Achse.| Modell | Berechnungsbetrag | Authentizität | Anwendbare Stufe |
|------|--------|--------|---------|
| Massenpunkt | ⭐ | ⭐ | Algorithmus-Schnellüberprüfung |
| Einzelintegrator | ⭐⭐ | ⭐⭐ | RL-Training (Mainstream-Wahl) |
| Zwei-Wege-Integrator | ⭐⭐ | ⭐⭐⭐ | Nahe am echten Flug |
| 6-DOF | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Sim-to-Real-Vorverifizierung |

### 3.2 Zustandsraumdesign

Ein guter Zustandsraum sollte alle für die Entscheidungsfindung erforderlichen Informationen enthalten und über kontrollierbare Dimensionen verfügen:

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

### 3.3 Aktionsraumgestaltung

| Aktionstypen | Abmessungen | Beispiele | Vorteile | Nachteile |
|---------|------|------|------|------|
| **Direkte Geschwindigkeitseinstellung** | 2-3 | $[v_x, v_y]$ | Einfach, RL-freundlich | Nachbearbeitung erforderlich, um $\|\mathbf{v}\| sicherzustellen \leq v_{max}$ |
| **Geschwindigkeitserhöhung** | 2-3 | $[\Delta v_x, \Delta v_y]$ | Die inkrementelle Steuerung ist reibungsloser | Kann Geschwindigkeitsbeschränkungen überschreiten |
| **Kurswinkel + Geschwindigkeit** | 2 | $[\psi, v]$ | Im Einklang mit der Intuition | Polarkoordinatentransformation führt Singularität ein |
| **Zielposition** | 2-3 | $[x_{Ziel}, y_{Ziel}]$ | Am nächsten an menschlichen Anweisungen | Erfordert Pfadverfolgungs-Backend |
| **Schubvektor** | 4 | $[T_1,T_2,T_3,T_4]$ | Die unterste Steuerungsebene | Hohe Dimension, RL ist schwer zu erlernen |

**Mainstream-Wahl (MARL-Papier)**: Geschwindigkeitsinkrement $(\Delta v_x, \Delta v_y, \Delta v_z)$, mit $\tanh$-Aktivierung + Clip-Begrenzung.

### 3.4 Konfliktdefinition und -messung

**Deterministischer Konflikt (am häufigsten verwendet)**:

$$
c_{ij} = \begin{cases}
1 & \text{if } \| \mathbf{p}_i - \mathbf{p}_j \| < d_{safe} \\
0 & \text{andernfalls}
\end{Fälle}
$$**Probabilistischer Konflikt (unter Berücksichtigung der Unsicherheit)**:

$$
P_{Konflikt} = P\left( \| \mathbf{p}_i - \mathbf{p}_j \| < d_{safe} \right) = \int_{\| \mathbf{d} \| < d_{safe}} f_{\Delta \mathbf{p}}(\mathbf{d}) \, d\mathbf{d}
$$

wobei $f_{\Delta \mathbf{p}}$ die gemeinsame Wahrscheinlichkeitsdichte relativer Positionen ist (normalerweise als Gaußsche Dichte angenommen).

**Time-to-Conflict (TTC)**: Sagen Sie die verbleibende Zeit bis zu einer Kollision voraus. Je kleiner der TTC, desto dringlicher ist der Konflikt:

$$
TTC = \min_{t > 0} \{ t \mid \| \mathbf{p}_i(t) - \mathbf{p}_j(t) \| = d_{sicher} \}
$$

### 3.5 Gestaltungsprinzipien der Belohnungsfunktion

Die Belohnungsfunktion ist die Seele des RL-Umgebungsdesigns und bestimmt direkt das erlernte strategische Verhalten.

#### Prinzip 1: Unterpunkt Belohnungen > Einzelbelohnung

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

#### Prinzip 2: Sparsame Belohnung + Gestaltung

Rein spärliche Belohnungen (-100 nur für Kollisionen und +100 für das Erreichen des Ziels) sind in hochdimensionalen Zustandsräumen kaum zu trainieren. **Dichter Bonus + Sparse-Bonus** ist die beste Vorgehensweise:

- **Basisschicht**: Dichte Belohnungen für den Distanzfortschritt (mit Feedback bei jedem Schritt)
- **Sparse Layer**: Kollisionsstrafe + Reichweitenbelohnung (Schlüsselereignis)

#### Prinzip 3: Fairnessbeschränkungen

Vermeiden Sie das „Helden-UAV“-Phänomen – bestimmte UAVs geben immer nach und andere UAVs lernen überhaupt nicht:

$$
\hat{r}_i = r_i - \lambda \cdot \text{Var}(\{r_j\}_{j \in \mathcal{N}_i})
$$

Oder am Ende jeder Episode kann die „Anzahl der Konzessionen“ jedes UAV gezählt werden, und das UAV mit mehr Konzessionen erhält zusätzliche Vergütungsprämien.

---

## 4. Benchmark-Szenarien

### 4.1 Klassische Szenen in der Zeitung**Szenario 1 – Kreuzung**
Zwei UAV-Teams durchqueren denselben Luftraum aus entgegengesetzten Richtungen:

```
队1: (0, -100) → (0, +100)    [N 架，纵向]
队2: (-100, 0) → (+100, 0)   [M 架，横向]
```

Klassische Konfliktszenarien, Prüfung der Konflikterkennung und Timing-Entscheidungsfindung.

**Szenario 2 – Zirkuläre Verfolgung**
Mehrere UAVs fliegen im Uhrzeigersinn entlang derselben Kreisbahn, wobei das folgende Flugzeug das führende einholt:

$$
x_i = R\cos(\theta_i), \quad y_i = R\sin(\theta_i)
$$

$$
\theta_i = \theta_0 + \omega t + i \cdot \Delta\theta
$$

Wird zum Testen der Geschwindigkeitskoordination und der Abstandshaltung verwendet.

**Szenario 3 – Konvergierung**
Mehrere UAVs fliegen aus verschiedenen Richtungen auf denselben Zielpunkt zu:

$$
\mathbf{p}_i(0) = R \cdot [\cos(\phi_i), \sin(\phi_i)], \quad \phi_i = \frac{2\pi i}{N}
$$

Das schwerwiegendste Szenario besteht darin, die Fähigkeit zur Clusterverdauung in großem Maßstab zu testen.

**Szenario 4 – Dynamische Hindernisvermeidung**
Im gesamten Fluggebiet sind stationäre oder bewegliche Hindernisse verstreut:

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

### 4.2 Schwierigkeitsgrad

| Schwierigkeit | Anzahl der UAVs | Konfliktdichte | Dynamische Hindernisse | Unsicherheit |
|------|---------|---------|---------|---------|
| **Einfach** | 2–4 | Niedrig (vorhersehbar) | Keine | Keine |
| **Mittel** | 5–12 | Mittel (Koordination erforderlich) | Optional | GPS-Rauschen |
| **Schwer** | 20–50 | Hoch (muss verteilt werden) | Ja | GPS+Windstörung |
| **Extrem** | 100+ | Extrem hoch | Komplex | Totale Unsicherheit |

### 4.3 Bewertungsindikatorensystem

Anstatt nur die „Konfliktrate“ als Indikator zu verwenden, muss ein vollständiges Bewertungssystem etabliert werden:

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

## 5. Umgebungsregistrierung unter dem PyMARL-Framework**PyMARL** (University of Southampton) ist ein standardisiertes Framework für die MARL-Forschung. Registrieren Sie eine benutzerdefinierte Umgebung:

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

## 6. Zugriff auf echte Physik-Engines: von der Simulation bis zur tatsächlichen Messung

### 6.1 PX4 SITL-Zugriff

PX4 bietet eine vollständige Software-in-the-Loop-Simulation (SITL), die das Dynamikmodell der Gym-Umgebung direkt ersetzen kann:

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

### 6.2 Häufige Ursachen für Sim-to-Real-Lücken

Auch wenn das dynamische Modell völlig konsistent ist, besteht immer noch eine erhebliche Lücke zwischen Simulation und realer Migration:

| Quelle der Lücke | Simulation | Realität | Auswirkungen |
|---------|------|------|------|
| **Motorische Reaktionsverzögerung** | Momentan | 20–50 ms | Flugbahnfehler beim Manövrieren mit hoher Geschwindigkeit |
| **GPS-Rauschen** | Ideal | 1–5 m Fehler | Positionsschätzungs-Offset |
| **Windstörung** | Optionale Modellierung | Kontinuierlich zufällig | Seitliche Drift |
| **Physikalische Parameter** | Nominalwert | Änderungen mit Temperatur/Elektrizität | Änderungen des Schubkoeffizienten |
| **Kommunikationsverzögerung** | 0ms | 5–100 ms | Versagen bei der kollaborativen Entscheidungsfindung |

**Domänen-Randomisierung** ist die gängige Methode zur Schließung der Sim-to-Real-Lücke:

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

## 7. Zusammenfassung: Checkliste für den Umgebungsaufbau

Eine qualifizierte UAV-Konfliktsimulationsumgebung sollte Folgendes erfüllen:

- ✅ **Dynamik-Realismus**: Verwenden Sie mindestens einen Zwei-Wege-Integrator unter Berücksichtigung von Geschwindigkeits-/Beschleunigungsbeschränkungen
- ✅ **Gut definierter Konflikt**: Der Sicherheitsabstand $d_{safe}$ hat eine physikalische Grundlage (normalerweise 5–10 m)
- ✅ **Belohnungsfunktion kann erklärt werden**: Unterbelohnung, spärlicher Bonus, Vermeidung von Belohnungs-Hacking
- ✅ **Kontrollierte Zufälligkeit**: Ergebnisse unter einem festen Startwert können vollständig reproduziert werden
- ✅ **Vollständige Bewertungsindikatoren**: Kollisionsrate + Aufgabenerledigungsrate + Fairness + Glätte
- ✅ **Basisszenarien können reproduziert werden**: Kreuzungsrouten/Aufhol-/Konvergenzszenarien sind klar beschrieben
- ✅ **Schnittstellenstandardisierung**: Gym PettingZoo-kompatibel oder PyMARL-RegistrierungsformatDie Codebeispiele in diesem Artikel decken den gesamten Technologie-Stack von der benutzerdefinierten Gym-Umgebung bis zur AirSim/Gazebo-Plattform ab und können je nach Forschungszielen frei kombiniert werden.

---

**Referenzen:**1. Shah, S., et al. (2018). *AirSim: Hochpräzise visuelle und physikalische Simulation für autonome Fahrzeuge.* Field and Service Robotics (FSR), Springer.
2. Zhou, M., et al. (2019). *Eine Umfrage zur Pfadplanung für UAVs in überfüllten Umgebungen.* IEEE-Transaktionen zu intelligenten Transportsystemen (T-ITS).
3. Everett, M., et al. (2021). *Kollisionsvermeidung bei dichtem Verkehr mit tiefgreifendem Verstärkungslernen.* IEEE International Conference on Robotics and Automation (ICRA).
4. Alonso-Mora, J., et al. (2018). *Optimierungsbasierte Kollisionsvermeidung für Mehrfahrzeugsysteme.* IEEE Transactions on Robotics (TRO).
5. van den Berg, J., Lin, M. & Manocha, D. (2008). *Reziproke Geschwindigkeitshindernisse für Echtzeit-Multiagentennavigation.* IEEE International Conference on Robotics and Automation (ICRA).
6. Richards, A. & How, J. P. (2002). *Flugzeugflugbahnplanung mit Kollisionsvermeidung unter Verwendung gemischter ganzzahliger linearer Programmierung.* AIAA Guidance, Navigations- und Kontrollkonferenz (GNC).
7. Fan, T., et al. (2020). *Verteilte Multi-Roboter-Kollisionsvermeidung durch Deep Reinforcement Learning für die Navigation in komplexen Szenarien.* The International Journal of Robotics Research (IJRR).
8. Yan, C., et al. (2025). *Multi-Agent-Verstärkungslernen mit räumlich-zeitlicher Aufmerksamkeit für die Beflockung mit Kollisionsvermeidung einer skalierbaren Starrflügel-UAV-Flotte.* IEEE-Transaktionen auf intelligenten Transportsystemen (T-ITS).
9. Yu, L., et al. (2025). *Hybrid-Transformator-basiertes Multi-Agent-Verstärkungslernen für die Koordination mehrerer unbemannter Luftfahrzeuge in Luftkorridoren.* IEEE-Transaktionen auf Mobile Computing (TMC).
10. Huo, D., et al. (2023). *Kollisionsfreie modellprädiktive Flugbahnverfolgungssteuerung für UAVs in Hindernisumgebungen.* IEEE-Transaktionen zu Luft- und Raumfahrt- und elektronischen Systemen (TAES).
11. Jiang, C., et al. (2024). *Distributed Sampling-Based Model Predictive Control via Belief Propagation for Multi-Robot Formation Navigation.* IEEE Robotics and Automation Letters (RA-L).