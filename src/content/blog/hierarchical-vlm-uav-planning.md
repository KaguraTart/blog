---
title: "分层 VLM 规划：让无人机读懂"去3号楼东侧降落"这类指令"
description: "深入解析视觉-语言-动作模型（VLA）在无人机路径规划中的应用，梳理从单一端到端到分层语义规划的演进路线，涵盖 RT-2、OpenVLA、Compositional Foundation Models、LangStrands 等关键工作，分析为什么分层架构是 UAV VLA 的最优解，并给出实现指南。"
tags: ["VLM", "VLA", "具身智能", "无人机", "分层规划", "RT-2", "OpenVLA", "Foundation Model", "Robotics"]
date: 2026-04-15
---

# 分层 VLM 规划：让无人机读懂"去3号楼东侧降落"这类指令

## 1. 问题：为什么"直接端到端"对无人机不work？

想象你给一架无人机下指令：**"绕过 2 号楼，在 3 号楼东侧空地降落"**。

这句话对人类来说简单，但包含了三层含义：

1. **语义层**：找到 2 号楼和 3 号楼的位置
2. **空间推理**：从当前位置出发，绕过 2 号楼，到达 3 号楼东侧
3. **物理执行**：生成平滑的 3D 轨迹，控制电机实时跟踪

如果用纯端到端的 VLA（Vision-Language-Action），从相机图像直接输出电机控制信号，会面临两个根本问题：

- **输出频率不匹配**：VLA 的语言模型推理一次要几百毫秒，但无人机控制需要 100Hz+ 的实时信号
- **安全不可控**：端到端的黑盒模型无法保证生成的轨迹满足物理约束，撞楼了怎么办？

所以**分层 VLM 规划**成了工业界和学术界共同的选择。

## 2. 分层架构总览

```
用户指令: "绕过2号楼，在3号楼东侧降落"

┌─────────────────────────────────────────────┐
│ Layer 1: 语义理解层 (VLM/LLM)               │
│ 输入: 文本指令 + 地图/图像                   │
│ 输出: "先向北飞，绕过2号楼，再向东飞..."       │
│       (子目标序列)                            │
│ 模型: GPT-4V / LLaVA / Gemini 2.0 Flash    │
└─────────────────────────────────────────────┘
                    ↓ 子目标
┌─────────────────────────────────────────────┐
│ Layer 2: 轨迹规划层 (MPC / RRT* / ESDF)    │
│ 输入: 当前状态 + 子目标                      │
│ 输出: 空间路径点序列 (x,y,z,t)              │
│ 特点: 有理论安全性保证（可源性、碰撞检测）     │
└─────────────────────────────────────────────┘
                    ↓ 路径点
┌─────────────────────────────────────────────┐
│ Layer 3: 控制执行层 (PID / 非线性MPC)        │
│ 输入: 期望轨迹 + 当前状态                    │
│ 输出: 电机转速 / PWM 信号                    │
│ 频率: 100-400Hz (实时)                      │
└─────────────────────────────────────────────┘
```

这种分层的好处：**每层解耦，可以用最适合该层的方法独立训练/优化**，不需要全链路端到端训练。

## 3. 各层详解

### 3.1 Layer 1：语义理解层——让 VLM 理解指令

这是分层架构中最"聪明"的一层，也是大模型最能发挥作用的地方。

**核心任务：**
- 解析自然语言指令，提取关键地标和约束（"2 号楼"、"东侧"、"绕过"）
- 将 2D/3D 地图信息与视觉观测对齐
- 输出语义子目标序列（subgoal list）

**关键技术：Visual Grounding**

将语言中的抽象指代（"3 号楼东侧"）映射到地图/图像中的具体位置：

```python
# 伪代码示例
def parse_instruction(instruction: str, map_image: Image, ego_view: Image):
    # Step 1: 用 VLM 理解指令中的地标
    landmarks = vlm.extract_landmarks(
        instruction,  # "绕过2号楼，在3号楼东侧降落"
        map_image    # 鸟瞰地图
    )
    # → ["building_2 (polygon: [[x1,y1],...])", 
    #    "landing_zone: east_side_of_building_3"]

    # Step 2: 将地标转换为坐标
    goal_pose = convert_to_waypoints(landmarks, ego_view)

    # Step 3: 生成子目标序列
    subgoals = plan_subgoal_sequence(
        current_pose, goal_pose,
        constraints=[avoid(building_2), approach(building_3, side=east)]
    )
    # → ["fly_north_50m", "turn_east", "descend_20m", "hover_at_landing_zone"]

    return subgoals
```

**模型选型：**

| 模型 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| GPT-4V / GPT-4o | 推理能力强，多模态强 | 需要联网，延迟高 | 云端，延迟不敏感 |
| Gemini 2.0 Flash | 免费，速度快，支持本地部署 | 中文指令理解一般 | 本地边缘部署 |
| LLaVA 7B | 可本地部署，开源 | 复杂指令理解弱 | 边缘无人机 |
| Qwen2-VL | 中文友好，开源 | 边缘部署需量化 | 国产场景 |

**近期进展（2024-2025）：**

- **LLaVA-Plan** — 在 LLaVA 基础上增加 Planning Head，专门做任务分解
- **GPT-4o 实时语音** — 端到端语音指令理解，不需要 ASR，中断了

### 3.2 Layer 2：轨迹规划层——从语义到空间路径

这一层接收语义层的子目标，输出几何路径。

**经典方法（非学习式）：**

```python
# RRT* 全局路径规划
path = rrt_star(
    start=current_pose,
    goal=subgoal,
    obstacles=building_2_obstacle,  # 来自语义层的输出
    max_iterations=1000,
    connection_radius=5.0
)

# ESDF 局部避障（实时）
esdf_map = build_esdf_from_lidar(ego_view)
safe_direction = esdf_map.gradient_at(current_position)

# MPC 轨迹优化
trajectory = mpc.optimize(
    horizon=20,
    dynamics=uav_dynamics,
    obstacles=esdf_map,
    cost=[trajectory_smoothness, progress_to_goal, control_effort]
)
```

**学习方法（强化学习）：**

```python
# 策略网络：输入当前状态+目标 → 输出控制动作
policy = PPO(
    obs_dim=state_dim,      # 位置、速度、姿态、附近障碍物
    act_dim=action_dim,     # 速度指令 (vx, vy, vz)
)

# 在仿真中训练，通过 Domain Randomization 提升泛化性
# DR参数：风速、延迟、传感器噪声、空气质量
env.set_domain_randomization(
    wind=(0, 5),           # m/s
    comm_latency=(0, 100), # ms
    sensor_noise=(0, 0.05) # 归一化噪声
)
```

**为什么不用纯 RL 替换这一层？**

因为纯 RL 的轨迹没有理论安全性保证——RL 可能找到"看起来可行但实际上撞墙"的路径。ESDF + MPC 的组合提供了可源性（reachability）保证：**只要 MPC 能找到解，就一定不撞障碍物**。

### 3.3 Layer 3：控制执行层——实时稳定跟踪

这一层是最成熟的，传统控制理论已经完全够用：

```python
# 非线性 MPC 控制
class UAVController:
    def __init__(self):
        self.mpc = NMPC(
            horizon=10,          # 预测 10 步 (~1秒)
            dt=0.1,             # 控制周期 10Hz
            Q=diag([1,1,1]),    # 位置误差权重
            R=diag([0.1,0.1])   # 控制量权重
        )

    def control(self, state, ref_trajectory):
        # ref_trajectory 来自 Layer 2
        u = self.mpc.solve(state, ref_trajectory)
        return self.motor_mixer.mix(u)  # 转换为电机转速

    def safety_check(self, state):
        # 实时安全兜底：如果状态危险，强制悬停
        if state.altitude < 2.0 and state.speed < 0.5:
            return "LANDING"
        return "FLY"
```

## 4. 关键研究工作

### 4.1 Compositional Foundation Models for Hierarchical Planning

**Ajay et al., arXiv:2309.08587 (2023)**

这篇是非常重要的基础工作，提出了"组合式基础模型"的概念：

- **核心观点**：用多个专用基础模型组合，每层做一件事，用组合的方式完成复杂任务
- **架构**：视觉编码器 + 语言模型 + 动作解码器，分层级联
- **实验**：在机器人臂上验证，证明分层比纯端到端更好泛化

**为什么对 UAV 有启发：** 分层规划让每层都能独立复用预训练模型，不需要针对 UAV 场景重新训练整个系统。

### 4.2 LangStrands — 自然语言控制机器人

**LangStrands (2024)** — 用自然语言控制机器人在工业场景执行侦察/操作任务：

- 支持复杂指令："先检查 A 区域的设备，如果发现异常就去 B 位置报告"
- 将指令解析为任务图（Task Graph），支持条件分支和循环
- 支持多机器人协同，每台机器人接收不同的子任务

**对 UAV 的借鉴：** LangStrands 的任务图解析思路可以直接迁移到无人机，比如"先侦察 5 个目标点，然后返回基地"这类复杂任务。

### 4.3 Embodied Tree of Thoughts — 世界模型辅助规划

**Xu et al., arXiv:2512.08188 (2025)**

- 用 World Model 预测动作执行后的物理后果（环境状态变化）
- 在执行前用 Tree of Thoughts 搜索最优子目标序列
- 比纯 VLM 规划更物理-grounded，避免了"看起来对但物理上不可能"的路径

**对 UAV 的价值：** 无人机在空中飞行时，World Model 可以预测阵风影响、电池衰减后的悬停能力，提前规划更安全的轨迹。

### 4.4 OpenVLA — 开源机器人 VLA

**OpenVLA (2024)** — UC Berkeley 发布的开源 VLA 模型：

- 7B 参数，支持 97 种机器人动作
- 在 22 万条真实机器人数据上训练
- 可以在消费级 GPU (RTX 3090) 上运行
- **对 UAV 的潜力**：虽然目前 OpenVLA 主要针对机械臂，但 VLA 的架构（视觉编码 + LLM + 动作头）完全可以迁移到无人机场景

### 4.5 Embodied Arena — 具身智能统一评测平台

**Ni et al., arXiv:2509.15273 (2025)**

- 覆盖 250+ 具身智能任务，统一评测标准
- 包括室内导航、操作、空中飞行等任务类型
- 提供了 UAV 分层的性能基准（每层的准确率、延迟、成功率）

**重要性：** 有了统一评测平台，分层架构的每层都可以独立评估，优化有据可循。

## 5. Sim2Real：如何让训练好的策略迁移到真实无人机

分层架构的一个关键优势：**每层可以独立 Sim2Real**，不需要全链路端到端迁移。

### 5.1 各层 Sim2Real 难度分析

| 层级 | 训练环境 | 迁移难度 | 核心挑战 |
|------|---------|---------|---------|
| Layer 1 (VLM) | 任意图像/地图 | **低** | VLM 已经预训练好，泛化性强 |
| Layer 2 (RL) | AirSim / Flightmare | **中** | 空气动力学参数不匹配 |
| Layer 3 (MPC) | 真实无人机调参 | **低** | 标定好电机参数即可 |

### 5.2 Layer 2 的 Sim2Real 策略

**Domain Randomization (DR)：**
```python
# 仿真训练时随机化关键物理参数
class SimEnv:
    def reset(self):
        self.wind = random.uniform(-3, 3)      # m/s 阵风
        self.motor_lag = random.uniform(0.8, 1.2)  # 电机响应系数
        self.battery_level = random.uniform(0.7, 1.0)  # 电池状态
        self.gps_noise = random.uniform(0, 0.5)  # GPS 噪声 (m)
```

**Real2Sim Calibration（真机标定）：**
```python
# 在真实无人机上跑系统辨识
def calibrate_dynamics(real_uav):
    # 激励信号：阶跃输入
    for amplitude in [0.1, 0.3, 0.5]:
        response = real_uav.step_input(thrust=amplitude)
        # 拟合真实电机响应曲线
        motor_model.fit(step_responses)

    # 将标定参数写回仿真
    sim_env.set_dynamics(motor_model.params)
```

### 5.3 实际案例：MADER 的 Sim2Real

**MADER (Multi-Agent DEep Reinforcement learning for aerial swarms)** 是近年来 UAV Sim2Real 做得最好的工作之一：

- 在 AirSim 中用 MADDPG 训练多机协调避障策略
- **关键 Trick**：训练时加入传感器延迟（20-50ms），让策略学会在延迟下工作
- **结果**：零样本迁移到真实 Tello 无人机，避障成功率 > 85%

## 6. 工程实现：从零搭建分层 VLM 无人机

### 6.1 推荐技术栈

```
硬件:                      软件:
- Pixhawk 飞控 (或 Crazyflie)   - PX4 / ArduPilot 固件
- Jetson Orin NX (边缘计算)     - ROS 2 Humble
- Livox 激光雷达 / RealSense    - 深度相机 + IMU

软件分层:
┌──────────────────────────────────┐
│ VLM (Layer 1): LLaVA 7B/Qwen2-VL │
│ 推理引擎: llm.cpp / vLLM          │
│ 推理硬件: Jetson Orin NX (INT8)   │
└──────────────────────────────────┘
┌──────────────────────────────────┐
│ 规划器 (Layer 2):                │
│ - 全局: RRT* (OMPL)             │
│ - 局部: OSQP / Crocoddyl (MPC)   │
└──────────────────────────────────┘
┌──────────────────────────────────┐
│ 控制器 (Layer 3): PX4 SITL /     │
│              Ardupilot guided mode │
└──────────────────────────────────┘
```

### 6.2 ROS 2 消息接口

```python
# Layer 1 → Layer 2 消息
class SubgoalMsg(Message):
    position: Point  # 目标点
    constraints: List[Constraint]  # 避障约束
    priority: int  # 优先级
    timeout: float  # 超时时间

# Layer 2 → Layer 3 消息
class TrajectoryMsg(Message):
    waypoints: List[PoseStamped]  # 路径点序列
    velocities: List[float]       # 期望速度
    start_time: Time             # 计划开始时间
```

### 6.3 延迟预算（实时性保证）

```
总延迟预算: < 500ms (可接受)
├─ 图像采集: 30ms (30fps)
├─ VLM 推理: 200-400ms (LLaVA 7B @ INT8)
├─ 轨迹规划: 50ms (RRT* + MPC)
└─ 控制器跟踪: 实时 (100Hz)
```

如果 VLM 推理太慢，可以：
1. 用流式推理（streaming），规划层提前拿到中间结果
2. 用轻量级模型（LLaVA 3B / Qwen2-VL 2B）
3. 缓存常用指令的规划结果（地图固定时有效）

## 7. 当前挑战与未来方向

### 7.1 核心挑战

1. **VLM 推理延迟**：LLaVA 7B 在边缘 GPU 上推理约 200-400ms，超过了安全响应的要求
2. **指令歧义性**："在安全的地方降落"这种模糊指令 VLM 难以处理
3. **多层误差累积**：Layer 1 的语义误差 → Layer 2 的路径偏差 → Layer 3 的控制抖动
4. **动态障碍物**：Layer 2 的 ESDF 地图更新频率跟不上高速障碍物（如飞鸟）

### 7.2 未来方向

- **多模态指令融合**：语音 + 手势 + 注视点联合理解，单一模态失效时有备份
- **终身学习**：飞行过程中在线更新 Layer 2 策略，适应未知环境
- **多机协作 VLM**：一个 VLM 协调多架无人机，而不是每架独立运行
- **World Model 预测**：用生成式模型预测未来 5 秒的空中交通流，提前规避

## 8. 总结

分层 VLM 规划是当前 UAV 智能化最可行的路线：

- **Layer 1（VLM）** 负责语义理解，调用云端或边缘大模型
- **Layer 2（规划器）** 负责从语义到几何的转换，可用 RL + MPC 组合
- **Layer 3（控制器）** 负责实时跟踪，传统控制理论完全够用

这种架构的核心优势：**让大模型做它擅长的（理解），让经典方法做它擅长的（安全规划）**，各司其职，而不是用一个黑盒端到端模型承担所有风险。

---

*参考文献（按文中引用顺序）*

1. Ajay et al., "Compositional Foundation Models for Hierarchical Planning", arXiv:2309.08587, 2023
2. Padalkar et al., "OpenVLA: Open-Source Vision-Language-Action Model", 2024
3. Liu et al., "Aligning Cyber Space with Physical World: A Comprehensive Survey on Embodied AI", arXiv:2407.06886, 2024
4. Xu et al., "Embodied Tree of Thoughts: Deliberate Manipulation Planning with Embodied World Model", arXiv:2512.08188, 2025
5. Ni et al., "Embodied Arena: A Comprehensive Evaluation Platform for Embodied AI", arXiv:2509.15273, 2025
6. Mu et al., "Embodied AI-Enhanced IoMT Edge Computing: UAV Trajectory Optimization", arXiv:2512.20902, 2025
7. Zhou et al., "OmniShow: Unifying Multimodal Conditions for Human-Object Interaction", arXiv:2604.11804, 2026

*作者：Kagura Tart | 2026-04-15*
