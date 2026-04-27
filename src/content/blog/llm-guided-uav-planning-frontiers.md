---
title: "LLM-Guided UAV 任务规划：从推理到执行的前沿"
description: "深入解析 LLM 做 UAV 任务规划的三大范式：LLM as Planner、LLM+PDDL 符号规划、LLM+RAG，覆盖 VoxPoser、ActiveGAMER、双进程架构等前沿工作"
tags: ["UAV", "LLM", "任务规划", "PDDL", "具身智能", "端到端"]
category: "Tech"
pubDate: 2026-04-27
---

# LLM-Guided UAV 任务规划：从推理到执行的前沿

> **UAV 智能化系列 · 第X+1篇**
> 聚焦：LLM 作为任务规划器、符号规划集成、实时推理架构

---

## 1. 为什么 LLM 适合 UAV 任务规划？

UAV 任务规划的挑战在于**开放世界的不确定性**：

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

LLM 的优势：
- **世界知识**：预训练蕴含丰富的物理常识（"水会流动"、"汽车比人快"）
- **零样本推理**：无需针对每个任务单独训练
- **多步骤规划**：将复杂任务分解为子目标链（Chain-of-Thought）

---

## 2. LLM 做任务规划的范式

### 2.1 范式一：LLM as Planner（直接输出动作）

**代表工作：**

**ReAct（Reasoning + Acting）**
- 核心思想：LLM 交替进行"推理"和"动作"
- 每步：`obs → think → action → next_obs`
- 适用：状态可观测、环境反馈明确的场景
- 在 UAV 上的适配：需要快速的 action→obs 循环

**SayCan（PaLM-SayCan, 2022）**
- 将 LLM 的"能力描述"与物理"可行性"结合
- 机器人说自己"能做什么"，LLM 决定"应该做什么"
- **启示：** UAV 可以结合自身状态（电量、飞行限制）过滤不可行动作

---

### 2.2 范式二：LLM + PDDL 符号规划

**PDDL（Planning Domain Definition Language）** 是经典的机器人任务规划语言，将任务建模为离散符号问题。

**核心思想：**
```
VLM 感知 → PDDL problem 生成 → 经典规划器 → UAV 动作序列
```

**优势：**
- 规划结果可解释、可验证
- 保证任务完成的数学证明
- 适合安全关键场景（城市空域飞行）

**挑战：**
- PDDL 建模本身是瓶颈（需要领域专家）
- UAV 的连续动态特性与 PDDL 离散假设不完全兼容
- **解决思路：** PDDL 处理高层任务分解，MPC 处理低层轨迹执行

---

### 2.3 范式三：LLM + RAG（检索增强生成）

**GenerativeMPC（arXiv, 2026）**

**论文：** *GenerativeMPC: VLM-RAG-guided Whole-Body MPC with Virtual Impedance for Bimanual Mobile Manipulation*
**作者：** Marcelino Julio Fernando et al.
**来源：** arXiv, April 2026

**核心思想：**
```
VLM 感知当前场景 → 检索相关操作知识库 → RAG 生成操作建议 → MPC 执行
```

**关键技术：**
1. **知识检索**：从操作知识库（包含机器人操控经验数据）检索与当前场景最相关的示例
2. **Virtual Impedance**：生成柔顺控制参数，避免刚性碰撞
3. **RAG 过滤**：确保 LLM 输出在物理上可执行

**在 UAV 上的适配：**
- 检索建筑规范（高度限制、禁飞区）
- 检索历史任务经验（相似天气条件下的飞行参数）
- 检索安全协议（最小避障距离、应急程序）

---

## 3. 实时推理架构

### 3.1 双进程架构（arXiv, 2026）

**论文：** *A Dual-Process Architecture for Real-Time VLM-Based Indoor Navigation*
**作者：** Joonhee Lee, Hyunseung Shin, Jeonggil Ko
**来源：** arXiv:2601.19401, January 2026

**核心设计：**

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

**设计原则：**
- **快进程**（MPC）：毫秒级反应，处理实时障碍躲避
- **慢进程**（VLM）：秒级推理，处理高层决策
- **解耦关键**：VLM 不在关键路径上，不影响控制频率

---

### 3.2 分层规划框架

**高层（LLM/VLM，秒级）：**
```
任务理解 → 子目标分解 → 全局路径规划 → 授权低层执行
```

**中层（可微优化，100ms级）：**
```
RRT*/MPC → 局部路径重规划 → 平滑轨迹生成
```

**低层（PID/MPC，毫秒级）：**
```
姿态控制 → 电机分配 → 执行
```

---

## 4. 关键算法深度

### 4.1 VoxPoser：LLM 合成 3D 价值图

**论文：** *VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models*
**作者：** Wenlong Huang, Chen Wang, Ruohan Zhang, Yunzhu Li, Jiajun Wu, Li Fei-Fei
**来源：** arXiv:2307.05973, July 2023

**核心贡献：**
- LLM 输出**3D 空间热力图**（composable 3D value map）
- 热力图编码"应该去哪里"、"应该避开什么"
- 直接用作轨迹优化的奖励函数

**在 UAV 上的延伸：**
- VLM 输出 3D occupancy 热力图
- 热力图驱动 MPC 代价函数
- VoxPoser for UAV = "3D spatial affordance from language"

**注意：** VoxPoser 发表于 arXiv，目前未找到明确的会议发表记录。

---

### 4.2 CoNVO（Conditional Neural Value Optimization）

将 LLM 规划与价值迭代结合：
- LLM 提供**先验偏好**（哪些动作更合理）
- 价值迭代提供**最优性保证**
- 比纯 LLM 规划更鲁棒，比纯规划更灵活

---

## 5. 世界模型辅助规划

### 5.1 Why World Model?

LLM 的知识是**静态**的，但 UAV 环境是**动态**的：
- 风会变
- 障碍物会移动
- GNSS 信号会漂移

世界模型（World Model）让 UAV 能够**预测未来**：
```
当前状态 + 动作 → 世界模型 → 预测未来状态序列
LLM 在预测的未来状态序列上做规划（Plan over imagined futures）
```

### 5.2 论文代表

**Dreamer系列**（Daniel Hafner, Jürg Widmer, etc.）
- 基于 RSSM 动态模型
- 在 imagined future 上做强化学习
- 已在机器人上验证（机器人手臂、无人车）

**VMP（Video Motion Planning）**
- 用视频生成模型做运动规划
- 生成未来帧 → 提取运动向量 → 控制 UAV

---

## 6. 安全与验证

### 6.1 为什么安全是关键

UAV 在城市飞行时，决策失误可能造成**人员伤亡**。LLM 的概率性输出与航空安全要求的确定性保证之间存在根本矛盾。

### 6.2 安全框架

**CBF（Control Barrier Functions）：**
- ASMA 将 CBF 引入 UAV VLN
- 保证 unsafe 状态永不可达

**Formal Verification：**
- 使用 TLA+ / NuSMV 做状态机验证
- LLM 规划结果经过模型检验后才执行

**Shielding：**
- 底层保护器（Shield）：监控 LLM 输出，拦截不安全动作
- 上层 LLM：专注任务完成，不考虑安全细节
- **类似自动驾驶的" Guardian Angel"架构**

---

## 7. 前沿热点与未来方向

### 7.1 端到端 VLA（Vision-Language-Action）

**最新趋势：** 跳过"感知→规划→控制"的分层设计，直接从 VLM 输出 **action token**。

代表工作：
- **RT-2**（Google Robotics）：将 VLM 直接微调输出动作
- **π₀**（Physical Intelligence）：面向人形机器人的 VLA
- **UAV 版本**（正在涌现）：类似思想应用到无人机

**挑战：**
- 动作空间的连续性 vs 语言的离散性
- 安全验证困难（端到端黑箱）
- 数据稀缺（需要大规模 robot teleoperation 数据）

### 7.2 多机协同 LLM 规划

**SysNav（arXiv, March 2026）**

**论文：** *SysNav: Multi-Level Systematic Cooperation Enables Real-World, Cross-Embodiment Object Navigation*
**作者：** Haokun Zhu et al.
**来源：** arXiv:2603.xxxxx, March 2026

**核心贡献：**
- 多 agent 协同导航，跨不同 robot 平台
- LLM 做高层协调（谁去哪个区域）
- 分布式感知融合（各 agent 共享视野）

### 7.3 Physical Intelligence × UAV

- **Foundation Models for Manipulation** → **Foundation Models for Flight**
- 未来可能出现专用的" UAV 大脑"预训练模型
- 类似 LLaVA 但专精 3D 空间推理 + 飞行动力学

---

## 8. 总结与建议

| 维度 | 当前最佳 | 未来方向 |
|------|---------|---------|
| 规划范式 | 双进程架构（实时可行） | 端到端 VLA（长期目标）|
| 世界知识 | RAG（可靠但慢）| 世界模型（快速但需训练）|
| 安全 | CBF + Shielding | 形式化验证（完全保证）|
| 边缘部署 | 4-bit LLaVA（勉强实时）| 专用芯片（NPU/TPU）|

**给你的建议：**
1. **最快出成果路线**：双进程架构 + LLaVA-7B + 无人机平台
2. **最有创新空间**：VLM + 安全验证框架（目前几乎没人做）
3. **长期布局**：收集你自己的 UAV 操控数据，训练专用 VLA 模型

---

## 📚 参考文献

1. Lee et al. *A Dual-Process Architecture for Real-Time VLM-Based Indoor Navigation*. arXiv:2601.19401, 2026.
2. Fernando et al. *GenerativeMPC: VLM-RAG-guided Whole-Body MPC with Virtual Impedance*. arXiv, 2026.
3. Huang et al. *VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models*. arXiv:2307.05973, 2023.
4. Brohan et al. *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*. arXiv, 2023.
5. Zhu et al. *SysNav: Multi-Level Systematic Cooperation Enables Real-World, Cross-Embodiment Object Navigation*. arXiv, 2026.
6. Ahn et al. *Do As I Can and Not As I Say: Grounding Language in Robotic Affordances*. arXiv, 2022.