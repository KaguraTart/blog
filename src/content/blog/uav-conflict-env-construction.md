---
title: "UAV 路径冲突仿真环境构建：从论文实践到代码实现"
description: "系统梳理 UAV 多机冲突场景仿真环境的构建方法，涵盖主流仿真平台对比、动力学建模、状态空间设计、冲突定义、奖励函数构造与基准测试场景，附 Gym/Gazebo/AirSim 多框架完整示例"
pubDate: 2026-04-07
tags: ["无人机", "仿真环境", "多智能体", "路径规划", "Gym", "Gazebo", "AirSim", "强化学习"]
category: Tech
---

# UAV 路径冲突仿真环境构建：从论文实践到代码实现

> 在前两篇文章中，我们分别梳理了 [UAV 冲突消解算法全景图](/blog/uav-conflict-resolution/) 与 [MARL+GAT 端到端方案](/blog/marl-kat-uav-conflict/)。本文将回答一个更底层的问题：**如何在仿真环境中构建一个真实、可复现的 UAV 冲突场景？** 这直接决定了算法评估的可信度和实验的可对比性。

---

## 1. 为什么仿真环境构建如此关键？

仿真环境是连接算法理论与飞行实测的桥梁。一个设计良好的仿真环境需要满足三个维度：

| 维度 | 含义 | 常见问题 |
|------|------|---------|
| **真实性** | 仿真与真实的差距 | 动力学简化过度、传感器模型缺失 |
| **可复现性** | 相同种子下结果一致 | 随机初始化未控制、浮点误差累积 |
| **可扩展性** | 支持大规模集群 | 单机仿真无法并行、通信拓扑缺失 |

本文重点关注 **MARL 训练场景**下的仿真环境构建，兼顾真实性与可扩展性——毕竟在 50+ 架无人机上训练真实飞控不现实。

---

## 2. 主流仿真平台对比

### 2.1 平台总览

| 平台 | 底层引擎 | 物理真实性 | 多机支持 | MARL 兼容性 | 论文引用量 |
|------|---------|-----------|---------|------------|-----------|
| **AirSim** (Microsoft) | Unreal Engine | ⭐⭐⭐⭐⭐ | ✅ | ⚠️ 需 Wrapping | 高 |
| **FlightGoggles** (MIT) | Unity | ⭐⭐⭐⭐ | ✅ | ⚠️ 需 Wrapping | 高 |
| **RotorS** (ETH Zurich) | Gazebo/ROS | ⭐⭐⭐⭐ | ✅ | ✅ PyMARL 兼容 | 中 |
| **Morse** (LAAS-CNRS) | Blender Game Engine | ⭐⭐⭐ | ✅ | ✅ 原生 | 中 |
| **Webots** | 自研 | ⭐⭐⭐⭐ | ✅ | ✅ | 低 |
| **Custom 2D/3D** | N/A | ⭐⭐ | ✅✅✅ | ✅✅✅ | - |

**核心结论：**
- **学术基准测试**：RotorS + ROS/Gazebo 是主流选择（ETH Zurich 出品，兼容 PX4）
- **端到端 RL 研究**：大多数论文使用 **Custom Gym 环境**（完全自定义，方便改奖励函数）
- **Sim-to-Real 目标**：AirSim / FlightGoggles（高保真视觉 + 传感器噪声）

### 2.2 AirSim：从零到一百架的缩放经验

Microsoft AI for Earth 团队在 AirSim 上训练了最多 **100 架固定翼 UAV** 的集群协同：

> Shah, S., et al. (2018). *AirSim: High-Fidelity Visual and Physical Simulation for Autonomous Vehicles.* Field and Service Robotics.

AirSim 的关键优势：
- **传感器仿真**：摄像头、LiDAR、GPS、IMU 完整模拟，可注入噪声
- **天气系统**：风速、云层、光照对传感器的影响
- **API 统一**：支持 C++ / Python / ROS 接口

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

### 2.3 Gazebo / RotorS：ROS 生态下的标准选择

ETH Zurich 的 **RotorS** 是 ROS/Gazebo 生态下最完整的多 UAV 仿真包：

> Foehn, P., et al. (2020). *ETHZ Flying Machine Arena Dataset.* ETH Zurich.

RotorS 支持：
- **mav_msgs** 接口（spin + pitch + roll + thrust → motor PWM）
- **PX4 SITL / HITL**：完整飞控固件在环
- **RViz 可视化**：实时观察集群状态
- **Gazebo 物理引擎**：ODE / Bullet / Simbody 可选

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

### 2.4 Custom Gym 环境：学术 RL 研究的主流选择

绝大多数 MARL UAV 论文（包括 MADDPG / QMIX / MAPPO 的原始实现）使用完全自定义的 **Gym 风格环境**，原因有三：

1. **训练速度**：无图形渲染，Python 直接循环，每秒可完成数万步交互
2. **奖励函数可控**：快速迭代设计，不用改 C++ 代码
3. **可复现性**：固定随机种子下，结果完全可复现

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

## 3. 环境设计的核心要素

### 3.1 动力学模型：从质点模型到六自由度

**Level 1 — 质点模型（Point Mass）**
最简化，仅考虑位置和速度更新，用于算法验证和快速训练：

$$
\dot{\mathbf{p}} = \mathbf{v}, \quad \dot{\mathbf{v}} = \mathbf{a}
$$

$$
\mathbf{a}_{new} = \text{clip}(\mathbf{a}_{old} + \mathbf{u}, \mathbf{v}_{min}, \mathbf{v}_{max})
$$

**Level 2 — 单积分器（Single Integrator）**
直接控制速度，假设执行器响应瞬时：

$$
\mathbf{v}_i[k+1] = \mathbf{u}_i[k]
$$

**Level 3 — 双向积分器（Double Integrator）**
考虑加速度限制，更接近真实飞行器：

$$
\mathbf{v}_i[k+1] = \mathbf{v}_i[k] + \mathbf{a}_i[k] \Delta t
$$
$$
\mathbf{p}_i[k+1] = \mathbf{p}_i[k] + \mathbf{v}_i[k] \Delta t + \frac{1}{2}\mathbf{a}_i[k]\Delta t^2
$$

$$
\|\mathbf{a}_i[k]\| \leq a_{max}, \quad \|\mathbf{v}_i[k]\| \leq v_{max}
$$

**Level 4 — 四旋翼六自由度（6-DOF Quadrotor）**
完整非线性模型，考虑姿态和推力约束：

$$
\dot{\boldsymbol{\eta}} = \mathbf{R}(\boldsymbol{\Theta})\boldsymbol{\omega}
$$
$$
m\ddot{\mathbf{p}} = -mg\mathbf{e}_3 + \mathbf{R}(\boldsymbol{\Theta})\mathbf{T}_i\mathbf{e}_3 - \mathbf{D}(\dot{\mathbf{p}})
$$

其中 $\boldsymbol{\eta}$ 为欧拉角，$\mathbf{R}$ 为旋转矩阵，$\mathbf{T}_i$ 为第 $i$ 轴推力。

| 模型 | 计算量 | 真实性 | 适用阶段 |
|------|--------|--------|---------|
| 质点 | ⭐ | ⭐ | 算法快速验证 |
| 单积分器 | ⭐⭐ | ⭐⭐ | RL 训练（主流选择） |
| 双向积分器 | ⭐⭐ | ⭐⭐⭐ | 接近真实飞行 |
| 6-DOF | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Sim-to-Real 前验证 |

### 3.2 状态空间设计

一个好的状态空间应该**包含所有决策所需信息，且维度可控**：

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

### 3.3 动作空间设计

| 动作类型 | 维度 | 示例 | 优点 | 缺点 |
|---------|------|------|------|------|
| **速度直接设定** | 2-3 | $[v_x, v_y]$ | 简单，RL 友好 | 需后处理确保 $\|\mathbf{v}\| \leq v_{max}$ |
| **速度增量** | 2-3 | $[\Delta v_x, \Delta v_y]$ | 增量控制更平滑 | 可能超出速度约束 |
| **航向角+速度** | 2 | $[\psi, v]$ | 符合直觉 | 极坐标变换引入奇异性 |
| **目标位置** | 2-3 | $[x_{target}, y_{target}]$ | 最接近人类指令 | 需路径跟踪后端 |
| **推力向量** | 4 | $[T_1,T_2,T_3,T_4]$ | 最底层控制 | 维度高，RL 难学 |

**主流选择（MARL 论文）**：速度增量 $(\Delta v_x, \Delta v_y, \Delta v_z)$，配合 $\tanh$ 激活 + clip 限幅。

### 3.4 冲突定义与度量

**确定性冲突（最常用）**：

$$
c_{ij} = \begin{cases}
1 & \text{if } \| \mathbf{p}_i - \mathbf{p}_j \| < d_{safe} \\
0 & \text{otherwise}
\end{cases}
$$

**概率冲突（考虑不确定性）**：

$$
P_{conflict} = P\left( \| \mathbf{p}_i - \mathbf{p}_j \| < d_{safe} \right) = \int_{\| \mathbf{d} \| < d_{safe}} f_{\Delta \mathbf{p}}(\mathbf{d}) \, d\mathbf{d}
$$

其中 $f_{\Delta \mathbf{p}}$ 是相对位置的联合概率密度（通常假设为高斯）。

**Time-to-Conflict（TTC）**：预测碰撞发生前的剩余时间，TTC 越小表示冲突越紧急：

$$
TTC = \min_{t > 0} \{ t \mid \| \mathbf{p}_i(t) - \mathbf{p}_j(t) \| = d_{safe} \}
$$

### 3.5 奖励函数设计原则

奖励函数是 RL 环境设计的**灵魂**，直接决定学习到的策略行为。

#### 原则一：分项奖励 > 单一奖励

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

#### 原则二：稀疏奖励 + shaping

纯稀疏奖励（只有碰撞时给-100，到达目标给+100）在高维状态空间几乎无法训练。**稠密奖励 + sparse bonus** 是最佳实践：

- **基础层**：稠密的距离进度奖励（每步都有反馈）
- **稀疏层**：碰撞惩罚 + 到达奖励（关键事件）

#### 原则三：公平性约束

避免"英雄 UAV"现象——某几架 UAV 总是在让路，其他 UAV 完全不学：

$$
\hat{r}_i = r_i - \lambda \cdot \text{Var}(\{r_j\}_{j \in \mathcal{N}_i})
$$

或者在每个 episode 结束时统计各 UAV 的"让步次数"，让步多的 UAV 额外获得补偿奖励。

---

## 4. 基准测试场景（Benchmark Scenarios）

### 4.1 论文中的经典场景

**Scenario 1 — 交叉航线（Crossing）**
两队 UAV 从相对方向穿越同一空域：

```
队1: (0, -100) → (0, +100)    [N 架，纵向]
队2: (-100, 0) → (+100, 0)   [M 架，横向]
```

经典冲突场景，测试冲突识别和时序决策。

**Scenario 2 — 追及场景（Circular Pursuit）**
多架 UAV 沿同一圆形轨道顺时针飞行，后机追前机：

$$
x_i = R\cos(\theta_i), \quad y_i = R\sin(\theta_i)
$$

$$
\theta_i = \theta_0 + \omega t + i \cdot \Delta\theta
$$

用于测试速度协调和间距保持。

**Scenario 3 — 汇聚场景（Converging）**
多架 UAV 从不同方向向同一目标点飞行：

$$
\mathbf{p}_i(0) = R \cdot [\cos(\phi_i), \sin(\phi_i)], \quad \phi_i = \frac{2\pi i}{N}
$$

最严苛场景，测试大规模集群消解能力。

**Scenario 4 — 动态障碍规避**
静止或移动的障碍物散布在飞行区域中：

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

### 4.2 难度分级

| 难度 | UAV数量 | 冲突密度 | 动态障碍 | 不确定性 |
|------|---------|---------|---------|---------|
| **Easy** | 2–4 | 低（可预判） | 无 | 无 |
| **Medium** | 5–12 | 中（需协同） | 可选 | GPS噪声 |
| **Hard** | 20–50 | 高（必须分布式） | 有 | GPS+风扰动 |
| **Extreme** | 100+ | 极高 | 复杂 | 全不确定性 |

### 4.3 评估指标体系

不应只用"冲突率"一个指标，要建立完整的评估体系：

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

## 5. PyMARL 框架下的环境注册

**PyMARL**（Southampton大学）是 MARL 研究的标准化框架。注册自定义环境：

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

## 6. 真实物理引擎接入：从仿真到实测

### 6.1 PX4 SITL 接入

PX4 提供完整的软件在环仿真（SITL），可以直接替换 Gym 环境的动力学模型：

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

### 6.2 Sim-to-Real 差距的常见来源

即使动力学模型完全一致，仿真到真实迁移仍有显著差距：

| 差距来源 | 仿真 | 真实 | 影响 |
|---------|------|------|------|
| **电机响应延迟** | 瞬时 | 20–50ms | 高速机动时轨迹误差 |
| **GPS 噪声** | 理想 | 1–5m 误差 | 位置估计偏移 |
| **风扰动** | 可选建模 | 持续随机 | 侧向漂移 |
| **物理参数** | 标称值 | 随温度/电量变化 | 推力系数变化 |
| **通信延迟** | 0ms | 5–100ms | 协同决策失效 |

**Domain Randomization**（域随机化）是解决 Sim-to-Real 差距的主流方法：

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

## 7. 总结：环境构建检查清单

一个合格的 UAV 冲突仿真环境应该满足：

- ✅ **动力学真实性**：至少使用双向积分器，考虑速度/加速度约束
- ✅ **冲突定义明确**：安全距离 $d_{safe}$ 有物理依据（通常 5–10m）
- ✅ **奖励函数可解释**：分项奖励，稀疏 bonus，避免 reward hacking
- ✅ **随机性受控**：固定 seed 下的结果可完全复现
- ✅ **评估指标完整**：碰撞率 + 任务完成率 + 公平性 + 平滑度
- ✅ **基准场景可复现**：交叉航线 / 追及 / 汇聚场景描述清晰
- ✅ **接口标准化**：Gym PettingZoo 兼容或 PyMARL 注册格式

本文的代码示例覆盖了从 Gym 自定义环境到 AirSim/Gazebo 平台的全技术栈，可以根据研究目标自由组合。

---

**参考文献：**

1. Shah, S., et al. (2018). *AirSim: High-Fidelity Visual and Physical Simulation for Autonomous Vehicles.* Field and Service Robotics (FSR), Springer.
2. Foehn, P., et al. (2021). *AlphaPilot: Autonomous Drone Racing.* Technical Report / Competition Paper, Microsoft AI for Earth & University of Zurich.
3. Zhou, M., et al. (2019). *A Survey on Path Planning for UAVs in Cluttered Environments.* IEEE Transactions on Intelligent Transportation Systems (T-ITS), 20(10), 3834–3848.
4. Schranz, M., et al. (2020). *Swarm Robotic Behaviors and Current Applications.* Frontiers in Robotics and AI, 7, 36.
5. Everett, M., et al. (2021). *Collision Avoidance in Dense Traffic with Deep Reinforcement Learning.* IEEE International Conference on Robotics and Automation (ICRA).
6. Alonso-Mora, J., et al. (2018). *Optimization-Based Collision Avoidance for Multi-Vehicle Systems.* IEEE Transactions on Robotics (TRO), 34(4), 837–856.
7. Li, S., et al. (2021). *Cooperative Path Planning for Multi-UAV with Velocity Constraints.* IEEE Transactions on Aerospace and Electronic Systems, 57(2), 1310–1321.
