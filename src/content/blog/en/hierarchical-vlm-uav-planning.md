---
title: "Hierarchical VLM planning: Let the drone understand instructions such as \"land on the east side of Building 3\""
description: "In-depth analysis of the application of vision-language-action model (VLA) in UAV path planning, combing the evolution route from single end-to-end to hierarchical semantic planning, covering key work such as RT-2, OpenVLA, Compositional Foundation Models, LangStrands, etc., analyzing why hierarchical architecture is the optimal solution for UAV VLA, and giving implementation guidelines."
tags: ["VLM", "VLA", "embodied intelligence", "drone", "hierarchical planning", "RT-2", "OpenVLA", "Foundation Model", "Robotics"]
pubDate: 2026-04-15
---

# Hierarchical VLM planning: Let the drone understand instructions such as "land on the east side of Building 3"

## 1. Question: Why doesn't "direct end-to-end" work for drones?

Imagine you give a command to a drone: **"Go around Building 2 and land in the open space on the east side of Building 3"**.

This sentence is simple for humans, but it contains three levels of meaning:

1. **Semantic Layer**: Find the locations of Building 2 and Building 3
2. **Spatial Reasoning**: Starting from the current location, bypass Building 2 and reach the east side of Building 3
3. **Physical Execution**: Generate smooth 3D trajectories and control motors for real-time tracking

If you use pure end-to-end VLA (Vision-Language-Action) to directly output motor control signals from camera images, you will face two fundamental problems:

- **Output frequency mismatch**: VLA's language model inference takes hundreds of milliseconds at a time, but drone control requires 100Hz+ real-time signals
- **Uncontrollable security**: The end-to-end black box model cannot guarantee that the generated trajectory meets the physical constraints. What should I do if I hit a building?

Therefore, **hierarchical VLM planning** has become a common choice for industry and academia.

## 2. Overview of layered architecture

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

The benefits of this layering: **Each layer is decoupled and can be trained/optimized independently using the method most suitable for that layer**, without the need for full-link end-to-end training.

## 3. Detailed explanation of each layer

### 3.1 Layer 1: Semantic understanding layer - let VLM understand the instructions

This is the "smartest" layer in the layered architecture, and where large models are most useful.

**Core mission:**
- Parse natural language instructions to extract key landmarks and constraints ("Building 2", "East Side", "Bypass")
- Align 2D/3D map information with visual observations
- Output semantic subgoal sequence (subgoal list)

**Key Technology: Visual Grounding**

Map an abstract reference in the language ("East side of Building 3") to a concrete location in the map/image:

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

**Model selection:**| Model | Advantages | Disadvantages | Applicable scenarios |
|------|------|------|----------|
| GPT-4V / GPT-4o | Strong reasoning ability, strong multi-modality | Internet connection required, high latency | Cloud, latency insensitive |
| Gemini 2.0 Flash | Free, fast, supports local deployment | General understanding of Chinese instructions | Local edge deployment |
| LLaVA 7B | Locally deployable, open source | Weak in understanding complex instructions | Edge drone |
| Qwen2-VL | Chinese friendly, open source | Edge deployment needs to be quantified | Domestic scenarios |

**Recent Progress (2024-2025):**

- **LLaVA-Plan** — Adds Planning Head based on LLaVA, specializing in task decomposition
- **GPT-4o Live Voice** — end-to-end voice command understanding, no ASR required, no interruptions

### 3.2 Layer 2: Trajectory planning layer—from semantics to spatial paths

This layer receives the sub-goals of the semantic layer and outputs geometric paths.

**Classic method (non-learning style):**

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

**Learning method (reinforcement learning):**

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

**Why not replace this layer with pure RL? **

Because pure RL trajectories have no theoretical safety guarantees - RL may find paths that "look feasible but actually hit a wall". The combination of ESDF + MPC provides a reachability guarantee: **As long as MPC can find a solution, it will not hit obstacles**.

### 3.3 Layer 3: Control execution layer - real-time stable tracking

This layer is the most mature, and traditional control theory is completely sufficient:

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

## 4. Key Research Work

### 4.1 Compositional Foundation Models for Hierarchical Planning

**Ajay et al., arXiv:2309.08587 (2023)**

This article is very important basic work and proposes the concept of "combined basic model":- **Core idea**: Use multiple dedicated basic models to combine, each layer does one thing, and use combination to complete complex tasks
- **Architecture**: visual encoder + language model + action decoder, hierarchical cascading
- **Experiment**: Verified on a robot arm, proving that layering generalizes better than pure end-to-end

**Why it’s inspiring for UAV:** Hierarchical planning allows each layer to independently reuse pre-trained models, without the need to retrain the entire system for UAV scenarios.

### 4.2 LangStrands — Natural language controlled robot

**LangStrands (2024)** — Use natural language to control robots to perform reconnaissance/operation tasks in industrial scenarios:

-Supports complex instructions: "Check the equipment in area A first, and if any abnormality is found, report to location B"
- Parse instructions into Task Graph, supporting conditional branches and loops
- Supports multi-robot collaboration, each robot receives different subtasks

**References from UAVs:** LangStrands' mission map analysis ideas can be directly transferred to UAVs, such as complex tasks such as "first reconnaissance of 5 target points, and then return to base".

### 4.3 Embodied Tree of Thoughts — World Model Assisted Planning

**Xu et al., arXiv:2512.08188 (2025)**

- Use World Model to predict the physical consequences (environmental state changes) after action execution
- Use Tree of Thoughts to search for the optimal sub-goal sequence before execution
- More physics-grounded than pure VLM planning, avoiding "looks right but is physically impossible" paths

**Value to UAVs:** When the UAV is flying in the air, World Model can predict the impact of gusts, the hovering ability after battery decay, and plan a safer trajectory in advance.

### 4.4 OpenVLA — Open Source Robot VLA

**OpenVLA (2024)** — Open source VLA model released by UC Berkeley:

- 7B parameters, supporting 97 types of robot actions
- Trained on 220,000 real robot data
- Can run on consumer GPU (RTX 3090)
- **Potential for UAV**: Although currently OpenVLA is mainly targeted at robotic arms, the architecture of VLA (visual coding + LLM + action head) can be completely migrated to UAV scenarios### 4.5 Embodied Arena — Embodied Intelligence Unified Evaluation Platform

**Ni et al., arXiv:2509.15273 (2025)**

- Covers 250+ embodied intelligence tasks and unified evaluation standards
- Including indoor navigation, operation, aerial flight and other task types
- Provides performance benchmarks for UAV layering (accuracy, latency, success rate for each layer)

**Importance:** With a unified evaluation platform, each layer of the layered architecture can be independently evaluated, and optimization can be based on evidence.

## 5. Sim2Real: How to transfer trained strategies to real drones

A key advantage of the layered architecture: **Each layer can be Sim2Real** independently, eliminating the need for full-link end-to-end migration.

### 5.1 Sim2Real difficulty analysis of each layer

| Level | Training environment | Migration difficulty | Core challenges |
|------|---------|---------|---------|
| Layer 1 (VLM) | Any image/map | **Low** | VLM has been pre-trained and has strong generalization |
| Layer 2 (RL) | AirSim / Flightmare | **Medium** | Aerodynamic parameters mismatch |
| Layer 3 (MPC) | Real drone parameter adjustment | **Low** | Just calibrate the motor parameters |

### 5.2 Sim2Real Strategy for Layer 2

**Domain Randomization (DR):**
```python
# 仿真训练时随机化关键物理参数
class SimEnv:
    def reset(self):
        self.wind = random.uniform(-3, 3)      # m/s 阵风
        self.motor_lag = random.uniform(0.8, 1.2)  # 电机响应系数
        self.battery_level = random.uniform(0.7, 1.0)  # 电池状态
        self.gps_noise = random.uniform(0, 0.5)  # GPS 噪声 (m)
```

**Real2Sim Calibration (real machine calibration): **
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

### 5.3 Practical case: MADER’s Sim2Real

**MADER (Multi-Agent DEep Reinforcement learning for aerial swarms)** is one of the best works done by UAV Sim2Real in recent years:

- Use MADDPG in AirSim to train multi-machine coordinated obstacle avoidance strategies
- **Key Trick**: Add sensor delay (20-50ms) during training to let the strategy learn to work under delay
- **Results**: Zero sample migration to real Tello drone, obstacle avoidance success rate > 85%

## 6. Engineering implementation: building a hierarchical VLM drone from scratch### 6.1 Recommended technology stack

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

### 6.2 ROS 2 message interface

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

### 6.3 Delay budget (real-time guarantee)

```
总延迟预算: < 500ms (可接受)
├─ 图像采集: 30ms (30fps)
├─ VLM 推理: 200-400ms (LLaVA 7B @ INT8)
├─ 轨迹规划: 50ms (RRT* + MPC)
└─ 控制器跟踪: 实时 (100Hz)
```

If VLM inference is too slow, you can:
1. Using streaming reasoning, the planning layer can get the intermediate results in advance
2. Use lightweight model (LLaVA 3B / Qwen2-VL 2B)
3. Cache the planning results of commonly used instructions (valid when the map is fixed)

## 7. Current challenges and future directions

### 7.1 Core Challenges

1. **VLM inference latency**: LLaVA 7B infers about 200-400ms on edge GPU, exceeding the requirements for security response
2. **Command ambiguity**: The ambiguous command "land in a safe place" is difficult for VLM to handle
3. **Multi-layer error accumulation**: Semantic error of Layer 1 → Path deviation of Layer 2 → Control jitter of Layer 3
4. **Dynamic Obstacles**: Layer 2’s ESDF map update frequency cannot keep up with high-speed obstacles (such as flying birds)

### 7.2 Future Directions

- **Multi-modal command fusion**: Joint understanding of voice + gesture + gaze point, backup when a single modality fails
- **Lifelong Learning**: Update Layer 2 strategies online during flight to adapt to unknown environments
- **Multi-drone cooperative VLM**: One VLM coordinates multiple drones instead of each operating independently
- **World Model Prediction**: Use a generative model to predict air traffic flow in the next 5 seconds and avoid it in advance

## 8. Summary

Hierarchical VLM planning is currently the most feasible route for UAV intelligence:

- **Layer 1 (VLM)** is responsible for semantic understanding and calling cloud or edge large models
- **Layer 2 (Planner)** is responsible for the conversion from semantics to geometry and can be combined with RL + MPC
- **Layer 3 (Controller)** is responsible for real-time tracking, and traditional control theory is fully sufficient.

The core advantage of this architecture: **Let the big model do what it is good at (understanding), let the classic method do what it is good at (security planning)**, each performing their own duties, instead of using a black box end-to-end model to bear all risks.

---

*References (in order of citation in the text)*1. Ajay et al., "Compositional Foundation Models for Hierarchical Planning", arXiv:2309.08587, 2023
2. Padalkar et al., "OpenVLA: Open-Source Vision-Language-Action Model", 2024
3. Liu et al., "Aligning Cyber Space with Physical World: A Comprehensive Survey on Embodied AI", arXiv:2407.06886, 2024
4. Xu et al., "Embodied Tree of Thoughts: Deliberate Manipulation Planning with Embodied World Model", arXiv:2512.08188, 2025
5. Ni et al., "Embodied Arena: A Comprehensive Evaluation Platform for Embodied AI", arXiv:2509.15273, 2025
6. Mu et al., "Embodied AI-Enhanced IoMT Edge Computing: UAV Trajectory Optimization", arXiv:2512.20902, 2025
7. Zhou et al., "OmniShow: Unifying Multimodal Conditions for Human-Object Interaction", arXiv:2604.11804, 2026

*Author: Kagura Tart | 2026-04-15*