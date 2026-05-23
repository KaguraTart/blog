---
title: "LLM-Guided UAV mission planning: the frontier from inference to execution"
description: "In-depth analysis of the three major paradigms of LLM for UAV mission planning: LLM as Planner, LLM+PDDL symbol planning, and LLM+RAG, covering cutting-edge work such as VoxPoser, ActiveGAMER, and dual-process architecture."
tags: ["UAV", "LLM", "mission planning", "PDDL", "embodied intelligence", "end-to-end"]
category: "Tech"
pubDate: 2026-04-27
---

# LLM-Guided UAV Mission Planning: The Frontier from Inference to Execution

> **UAV Intelligent Series · Chapter X+1**
> Spotlight: LLM as mission planner, symbolic planning integration, real-time inference architecture

---

## 1. Why is LLM suitable for UAV mission planning?

The challenge of UAV mission planning lies in **open world uncertainty**:

```
传统规划（基于模型）：
输入：精确目标状态 + 精确环境模型
输出：最优动作序列
局限：模型不准就崩溃，无法处理语言目标

LLM 规划（基于知识）：
输入：自然语言指令 + 视觉观测 + 世界知识
输出：可执行动作序列
优势：泛化性强、零样本理解新任务
```

Advantages of LLM:
- **World Knowledge**: Pre-training contains rich physical knowledge ("Water flows", "Cars are faster than people")
- **Zero-shot inference**: No need to train separately for each task
- **Multi-step planning**: Decompose complex tasks into sub-goal chains (Chain-of-Thought)

---

## 2. LLM’s paradigm for task planning

### 2.1 Paradigm 1: LLM as Planner (directly output actions)

**Representative work:**

**ReAct (Reasoning + Acting)**
- Core idea: LLM alternates "reasoning" and "action"
- Each step: `obs → think → action → next_obs`
- Applicable to: Scenarios with observable status and clear environmental feedback
- Adaptation on UAV: requires fast action→obs loop

**SayCan (PaLM-SayCan, 2022)**
- Combine LLM's "capability description" with physical "feasibility"
- The robot says "what it can do", and the LLM decides "what it should do"
- **Enlightenment:** UAV can filter infeasible actions based on its own status (power, flight restrictions)

---

### 2.2 Paradigm 2: LLM + PDDL symbol planning

**PDDL (Planning Domain Definition Language)** is a classic robot task planning language that models tasks as discrete symbolic problems.

**Core idea:**
```
VLM 感知 → PDDL problem 生成 → 经典规划器 → UAV 动作序列
```

**Advantages:**
- Planning results can be explained and verified
- Mathematical proof to ensure task completion
- Suitable for safety-critical scenarios (urban airspace flights)

**Challenge:**
- PDDL modeling itself is a bottleneck (requires domain experts)
- The continuous dynamics of UAVs are not fully compatible with the discrete assumptions of PDDL
- **Solution idea:** PDDL handles high-level task decomposition, MPC handles low-level trajectory execution

---### 2.3 Paradigm 3: LLM + RAG (retrieval enhanced generation)

**GenerativeMPC (arXiv, 2026)**

**Paper:** *GenerativeMPC: VLM-RAG-guided Whole-Body MPC with Virtual Impedance for Bimanual Mobile Manipulation*
**Author:** Marcelino Julio Fernando et al.
**Source:** arXiv, April 2026

**Core idea:**
```
VLM 感知当前场景 → 检索相关操作知识库 → RAG 生成操作建议 → MPC 执行
```

**Key technology:**
1. **Knowledge retrieval**: Retrieve examples most relevant to the current scenario from the operational knowledge base (including robot control experience data)
2. **Virtual Impedance**: Generate compliance control parameters to avoid rigid collisions
3. **RAG filtering**: Ensure that LLM output is physically executable

**Adaptation on UAV:**
- Search building codes (height restrictions, no-fly zones)
- Retrieve historical mission experience (flight parameters under similar weather conditions)
- Retrieve safety protocols (minimum obstacle avoidance distance, emergency procedures)

---

## 3. Real-time reasoning architecture

### 3.1 Dual-process architecture (arXiv, 2026)

**Paper:** *A Dual-Process Architecture for Real-Time VLM-Based Indoor Navigation*
**Author:** Joonhee Lee, Hyunseung Shin, Jeonggil Ko
**Source:** arXiv:2601.19401, January 2026

**Core Design:**

```
┌─────────────────────────────────────────────┐
│           System Architecture               │
│                                             │
│  Process 1 (Slow): VLM Reasoning Thread     │
│  ┌─────────────────────────────────────┐   │
│  │ VLM: "What should I do next?"       │   │
│  │ Frequency: ~0.2-1 Hz                 │   │
│  │ Output: Navigation goal / decision  │   │
│  └─────────────────────────────────────┘   │
│              ↓ goal                        │
│  Process 2 (Fast): Control Execution Thread│
│  ┌─────────────────────────────────────┐   │
│  │ MPC: Track trajectory to goal        │   │
│  │ Frequency: ~100 Hz                   │   │
│  │ Output: Motor control signals        │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

**Design principles:**
- **Quick Process** (MPC): millisecond-level response, processing real-time obstacle avoidance
- **Slow Process** (VLM): Second-level reasoning, processing high-level decisions
- **Decoupling critical**: VLM is not on the critical path and does not affect the control frequency

---

### 3.2 Hierarchical planning framework

**High level (LLM/VLM, second level): **
```
任务理解 → 子目标分解 → 全局路径规划 → 授权低层执行
```

**Middle layer (differentiable optimization, 100ms level): **
```
RRT*/MPC → 局部路径重规划 → 平滑轨迹生成
```**Low layer (PID/MPC, millisecond level): **
```
姿态控制 → 电机分配 → 执行
```

---

## 4. Key algorithm depth

### 4.1 VoxPoser: LLM synthetic 3D value map

**Paper:** *VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models*
**Author:** Wenlong Huang, Chen Wang, Ruohan Zhang, Yunzhu Li, Jiajun Wu, Li Fei-Fei
**Source:** arXiv:2307.05973, July 2023

**Core contribution:**
- LLM output **3D spatial heat map** (composable 3D value map)
- Heat map encoding "where to go" and "what to avoid"
- Directly used as reward function for trajectory optimization

**Extension on UAV:**
- VLM output 3D occupancy heat map
- Heat map driven MPC cost function
- VoxPoser for UAV = "3D spatial affordance from language"

**Note:** VoxPoser was published on arXiv. No clear conference publication records have been found so far.

---

### 4.2 CoNVO (Conditional Neural Value Optimization)

Combine LLM planning with value iteration:
- LLM provides **prior preferences** (which actions are more reasonable)
- Value iteration provides **optimality guarantee**
- More robust than pure LLM planning and more flexible than pure planning

---

## 5. World model assisted planning

### 5.1 Why World Model?

The knowledge of the LLM is static, but the UAV environment is dynamic:
- The wind will change
- Obstacles will move
- GNSS signals can drift

The World Model allows UAVs to **predict the future**: 
```
当前状态 + 动作 → 世界模型 → 预测未来状态序列
LLM 在预测的未来状态序列上做规划（Plan over imagined futures）
```

### 5.2 Paper Representative**Dreamer Series** (Daniel Hafner, Jürg Widmer, etc.)
- Based on RSSM dynamic model
- Do reinforcement learning on imagined future
- Verified on robots (robot arms, unmanned vehicles)

**VMP (Video Motion Planning)**
- Use video generation models for motion planning
- Generate future frames → extract motion vectors → control UAV

---

## 6. Security and Authentication

### 6.1 Why security is key

When UAVs fly in cities, poor decision-making may cause **human casualties**. There is a fundamental contradiction between the probabilistic output of LLM and the deterministic guarantees required by aviation safety.

### 6.2 Security Framework

**CBF（Control Barrier Functions）：**
- ASMA introduces CBF to UAV VLN
- Ensure that the unsafe state is never reachable

**Formal Verification：**
- Use TLA+ / NuSMV for state machine verification
- LLM planning results are executed after model verification

**Shielding:**
- Bottom layer protector (Shield): monitors LLM output and intercepts unsafe actions
- Upper-level LLM: Focus on task completion and do not consider security details
- **Autonomous driving-like "Guardian Angel" architecture**

---

## 7. Frontier hot spots and future directions

### 7.1 End-to-end VLA (Vision-Language-Action)

**Latest trend:** Skip the hierarchical design of "sensing → planning → control" and output **action token** directly from VLM.

Representative work:
- **RT-2** (Google Robotics): Directly fine-tune the output action of VLM
- **π₀** (Physical Intelligence): VLA for humanoid robots
- **UAV version** (emerging): similar ideas applied to drones

**Challenge:**
- Continuity of action space vs discreteness of language
- Difficulty in security verification (end-to-end black box)
- Data scarcity (requires large-scale robot teleoperation data)

### 7.2 Multi-machine collaborative LLM planning

**SysNav (arXiv, March 2026)****Paper:** *SysNav: Multi-Level Systematic Cooperation Enables Real-World, Cross-Embodiment Object Navigation*
**Author:** Haokun Zhu et al.
**Source:** arXiv:2603.xxxxx, March 2026

**Core contribution:**
- Multi-agent collaborative navigation across different robot platforms
- LLM does high-level coordination (who goes to which area)
- Distributed perception fusion (each agent shares vision)

### 7.3 Physical Intelligence × UAV

- **Foundation Models for Manipulation** → **Foundation Models for Flight**
- A dedicated "UAV brain" pre-training model may appear in the future
- Similar to LLaVA but specializing in 3D spatial reasoning + flight dynamics

---

## 8. Summary and suggestions

| Dimensions | Current Best | Future Directions |
|------|---------|---------|
| Planning paradigm | Dual-process architecture (real-time feasible) | End-to-end VLA (long-term goal) |
| World knowledge | RAG (reliable but slow) | World model (fast but requires training) |
| Security | CBF + Shielding | Formal verification (fully guaranteed) |
| Edge deployment | 4-bit LLaVA (barely real-time) | Special purpose chips (NPU/TPU) |

**Advice for you:**
1. **The fastest route to results**: Dual-process architecture + LLaVA-7B + UAV platform
2. **The most room for innovation**: VLM + security verification framework (almost no one is doing it currently)
3. **Long-term layout**: Collect your own UAV control data and train a dedicated VLA model

---

## 📚 References1. Lee et al. *A Dual-Process Architecture for Real-Time VLM-Based Indoor Navigation*. arXiv:2601.19401, 2026.
2. Fernando et al. *GenerativeMPC: VLM-RAG-guided Whole-Body MPC with Virtual Impedance*. arXiv, 2026.
3. Huang et al. *VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models*. arXiv:2307.05973, 2023.
4. Brohan et al. *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*. arXiv, 2023.
5. Zhu et al. *SysNav: Multi-Level Systematic Cooperation Enables Real-World, Cross-Embodiment Object Navigation*. arXiv, 2026.
6. Ahn et al. *Do As I Can and Not As I Say: Grounding Language in Robotic Affordances*. arXiv, 2022.