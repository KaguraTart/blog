---
title: "Paper G 规划 v1：低空交通云脑中的 LLM Agent 与模型微调路线"
description: "规划如何训练或微调 LLM，使其成为低空交通云脑中的可验证 Agent，并形成 AAAI/IJCAI 首篇会议论文、后续交通期刊与通用 embodied agent 转型路线。"
pubDate: 2026-05-20
updatedDate: 2026-05-23
tags: ["Paper G", "低空交通云脑", "LLM Agent", "模型微调", "Tool Use", "AAAI", "IJCAI", "UAV", "AGI"]
category: Tech
---

# Paper G 规划 v1：低空交通云脑中的 LLM Agent 与模型微调路线

> 总体判断：这条路线不应先做“低空交通聊天大模型”，而应先做 **低空交通云脑中的可验证 LLM Agent**。  
> 首篇优先冲 AAAI / IJCAI：把 LLM 放在“任务理解、工具调用、规划修复、验证闭环、调度解释”的位置，而不是直接承诺训练一个大规模 foundation model。

---

## 1. 总体判断：为什么先做 Agent 云脑而不是直接训大模型

如果直接写“微调一个低空交通 LLM”，会议审稿人很可能会问三个问题：

1. **模型贡献在哪里？**  
   LoRA / SFT / DPO 本身已经是标准训练流程 [14] [15] [16]，仅把数据换成低空交通语料，难以支撑 AAAI / IJCAI 主会。

2. **为什么 LLM 比现有调度/规划模型更必要？**  
   低空交通运行涉及调度、路径规划、风险评估、形式化验证和仿真反馈。LLM 的优势不是替代这些模型，而是把复杂任务分解成可调用工具链。

3. **怎么保证安全？**  
   低空交通云脑属于安全关键系统。LLM 直接输出控制动作会有幻觉和不可验证风险。首篇论文必须把 verifier、simulator 和 risk estimator 放进闭环。

因此，Paper G 的首篇不建议叫 “LowAltitudeGPT”。更好的首篇是：

> **CloudBrain-Agent: Tool-Augmented and Verification-Guided LLM Agents for Low-Altitude Traffic Operation**

它的核心贡献不是“模型变聪明了”，而是：

- 构造低空交通云脑的 agentic decision pipeline；
- 让 LLM 学会调用低空交通工具；
- 用验证器和仿真器纠错；
- 输出可执行、可解释、可审计的调度/规划决策。

这与 TrafficGPT 的思想接近：TrafficGPT 已经指出 LLM 本身难以处理交通数值数据和仿真交互，因此需要和 traffic foundation models 结合 [1]。Paper G 的差异在于：我们把对象从地面交通扩展到低空交通，并进一步加入 UAV 状态、空域约束、形式化验证和安全闭环。

从更宽的交通智能综述看，LLM 已经被讨论为 ITS 中的语义接口、推理模块和交通决策辅助组件 [2] [3]；UrbanGPT 与 UniST 则说明城市时空预测正在向 spatio-temporal foundation model 过渡 [4] [5]。Paper G 不直接重复这些方向，而是把“城市时空智能 + UAV 运行工具 + 可验证 agent”组合成低空交通云脑。

### 1.1 2026-05-22 写作校准：G1 是 AI agent 论文，期刊扩展才需要完整交通系统叙事

Paper G 容易被写成“低空交通大模型故事”。这条路线要分清两种评价标准：

| 阶段 | 目标 | 主要审稿逻辑 | 不能犯的错误 |
|------|------|--------------|--------------|
| G1 AAAI/IJCAI | 可验证 LLM Agent 方法 | tool use、planning、verification、benchmark、reproducibility | 为了交通叙事牺牲方法清晰度，或把 agent 写成平台展示 |
| G2 T-ITS/T-IV | 低空交通领域 LLM 微调 | 领域数据、部署可复现、交通决策辅助能力 | 只做通用 LoRA/SFT，没有交通工具链和安全指标 |
| G3 AAMAS/T-ITS | 多 agent 云脑协同 | 多角色协作、通信、冲突处理、人机协同 | 多 agent 只是多个 prompt，没有系统状态和责任边界 |
| 期刊扩展版 | 交通系统运行意义 | 安全、效率、容量、延误、资源利用、管理启示 | 只报 accuracy/tool-call success，不回答交通问题 |

所以 G1 的主线仍然是强 AI 方法：typed IR、tool-use、verifier repair、stateful evaluation。
但所有低空交通相关指标要从一开始保留，方便后续扩展到 T-ITS：

- 安全：LoWC/NMAC proxy、no-fly-zone violation、battery reserve violation。
- 效率：delay、extra distance、energy、throughput、runtime。
- 运行管理：safe refusal rate、human confirmation rate、ambiguous-task handling。
- 鲁棒性：通信缺失、天气扰动、非合作 UAV、unseen city/topology。
- 系统启示：什么条件下 LLM agent 需要退出给 deterministic solver 或 human supervisor。

### 1.2 2026-05-23 整理：G 路线的先后顺序

Paper G 是 umbrella roadmap，真正近期要完成的是 **G1 CloudBrain-Agent**。当前最快、最可投稿的路线不是先训练垂类大模型，而是用通用强模型 + typed IR + 工具链 + verifier + simulator feedback 形成可复现闭环。垂类模型训练放到 G2，用 G1 产生的 tool-call traces、repair traces 和 failure cases 做数据。

| 阶段 | 是否训练模型 | 推荐模型/部署 | 目标 |
|------|--------------|---------------|------|
| G1 now | 不作为主贡献训练 | 本地 vLLM 跑 Qwen / DeepSeek，API 模型做 teacher / upper bound | 证明 agent 工具调用、验证修复和低空任务 benchmark 有效 |
| G2 next | LoRA / SFT / DPO | 用 G1 traces 微调 Qwen / Llama / DeepSeek 系列 | 形成 LowAltitudeGPT domain cognitive module |
| G3 later | 可选多 agent 轨迹蒸馏 | 多角色 agent + shared memory + verifier | 研究空域监控、调度、风险、应急、人机协同 |
| G4 long-term | 多模态 / world model / VLA | 取决于数据与算力 | 向 embodied traffic intelligence 迁移 |

部署策略建议如下：

- **主实验用本地开源模型**：可复现、可控成本、方便报告 latency 和硬件条件；建议用 vLLM / llama.cpp 作为推理服务。
- **API 模型做 teacher 和上限**：用于生成高质量初始样本、难例修复示范和 upper-bound baseline；论文中要把 API 结果和本地模型结果分开报告。
- **MCP 先做接口风格，不先做产品化**：第一版先实现 Python tool registry 和 JSON schema；等工具稳定后再封装成 MCP-compatible server，避免把工程复杂度压到论文主线。
- **垂类模型训练不抢 G1 主线**：G1 的贡献是 agent 架构和验证闭环；G2 才把运行轨迹蒸馏进本地模型。

这个顺序能最快形成可投稿闭环：先让系统跑起来、评测起来、能解释失败，再决定哪些能力值得微调进模型。

---

## 2. 低空交通云脑的系统定义

本文中的“低空交通云脑”不是泛泛的智能平台，而是一个面向城市低空 UAV 运行的 **cognitive operation layer**：

```text
Human / operator instruction
  -> CloudBrain LLM Agent
  -> LowAltitudeIR
  -> traffic tools / UAV tools / verifier / simulator
  -> safe decision proposal
  -> human approval or autonomous execution
```

### 2.1 输入

低空交通云脑接收多源状态：

| 输入 | 示例 |
|------|------|
| 自然语言任务 | “优先处理医院附近的应急配送，避开学校和禁飞区。” |
| UAV 状态 | 位置、电量、载重、任务状态、通信状态 |
| 空域状态 | corridor 容量、禁飞区、临时管制、天气、风场 |
| 交通需求 | 配送订单、巡检任务、应急事件、乘客/货物优先级 |
| 场景状态 | Paper F 的 safety-critical scenarios、事故场景、coverage holes |
| 形式约束 | LTL/STL 安全规则、时间窗、最低高度、最小间隔 |

### 2.2 输出

云脑不直接输出“飞行动作”，而输出可审计的中间决策：

| 输出 | 示例 |
|------|------|
| LowAltitudeIR | 结构化任务、实体、约束、工具调用计划 |
| 工具调用序列 | 查询空域、调用调度器、调用路径规划器、运行验证器 |
| 调度建议 | 哪架 UAV 执行哪个任务，是否触发地面 fallback |
| 安全诊断 | 哪条约束可能违反，是否需要人工确认 |
| 解释文本 | 用自然语言解释为什么这样调度 |

### 2.3 云脑不是端到端控制器

低空交通云脑的边界要写清楚：

- LLM 做语义理解、任务分解、工具选择、解释和修复。
- 调度器做 fleet assignment 和资源优化，对应 Paper B。
- 验证器做 LTL/STL 安全检查，对应 Paper E。
- 场景模拟器和 risk generator 提供压力测试，对应 Paper F。
- 轨迹控制器仍由传统规划/MPC/安全控制模块执行。

这能避免审稿人质疑“LLM 控制 UAV 不安全”。

---

## 3. 研究路线总览：从领域 LLM 到通用 embodied agent

Paper G 可以分成 4 个阶段。

| 阶段 | 论文 | 目标 | 关键问题 |
|------|------|------|----------|
| G1 | CloudBrain-Agent | AAAI / IJCAI | LLM 如何在低空交通云脑中可靠调用工具并通过验证闭环修复 |
| G2 | LowAltitudeGPT | T-ITS / T-IV | 如何微调本地开源 LLM 成为低空交通决策认知模块 |
| G3 | Multi-Agent Cloud Brain | AAMAS / IJCAI / T-ITS | 多个专职 agent 如何协同管理低空交通 |
| G4 | World-Model / VLA Extension | 长期路线 | 如何从领域 agent 转向 embodied general intelligence |

推荐顺序是 **G1 -> G2 -> G3 -> G4**。

G1 先解决“系统能不能跑起来、能不能安全闭环、能不能发会议”。G2 再把 agent 轨迹蒸馏成领域模型。G3 走多智能体协作。G4 才谈 AGI 转型，不在第一篇中夸大。

---

## 4. Paper G1：CloudBrain-Agent，面向 AAAI/IJCAI 的首篇会议论文

### 4.1 题目

**CloudBrain-Agent: Tool-Augmented and Verification-Guided LLM Agents for Low-Altitude Traffic Operation**

### 4.2 目标会议

首投：AAAI / IJCAI。  
备选：AAMAS、ICRA/IROS workshop、T-ITS fast journal extension。

AAAI-26 Main Technical Track 鼓励跨 AI 技术方向和交通等重要应用领域的工作，主文限制为 7 页技术内容并要求 reproducibility checklist [34]。IJCAI-ECAI 2026 的 AI and Robotics special track 明确关注 robot agents、generative AI、robot control、structured modeling、reasoning 和如何执行/避免行动后果 [35]。因此 G1 应写成 AI agent / planning / tool-use / verification 论文，而不是系统工程展示。

### 4.3 核心问题

G1 要回答：

> 给定低空交通运行任务，如何让 LLM agent 可靠地理解任务、选择工具、调用调度/规划/验证模块，并在反例反馈下修复错误，从而输出安全、可执行、可解释的云脑决策？

### 4.4 方法

提出 **CloudBrain-Agent**，包含五个模块：

| 模块 | 作用 |
|------|------|
| LowAltitudeIR parser | 将自然语言任务和系统状态转成结构化表示 |
| Tool planner | 规划工具调用序列 |
| Tool executor | 调用调度器、路径规划器、验证器、仿真器、风险评估器 |
| Verifier feedback loop | 把失败工具调用、不可满足约束、STL robustness 失败转成修复反馈 |
| Safety memory | 保存已知危险场景、失败案例、人工决策和规则约束 |

CloudBrain-Agent 的行为形式：

```text
Observe -> Think -> Select Tool -> Execute -> Verify -> Repair -> Decide
```

这继承 ReAct 的 reasoning-action loop [6]，但加入两个低空交通特有机制：

1. **工具调用必须类型安全**：每个工具输入输出都经过 `LowAltitudeIR` schema 检查。
2. **决策必须通过 verifier**：任何调度或路径建议都要经过安全验证或仿真压力测试。

### 4.5 LowAltitudeIR

LowAltitudeIR 是 G1 的关键公共接口：

```json
{
  "intent": "emergency_delivery",
  "entities": ["uav_12", "hospital_zone", "landing_pad_A"],
  "constraints": {
    "avoid": ["school_zone", "temporary_no_fly_zone"],
    "deadline_sec": 600,
    "min_obstacle_distance_m": 10,
    "altitude_range_m": [30, 120]
  },
  "tool_plan": [
    "query_airspace",
    "assign_uav",
    "plan_route",
    "verify_stl",
    "simulate_scenario"
  ],
  "fallback": "ground_vehicle_transfer_if_unreachable"
}
```

LowAltitudeIR 要兼容三条既有论文线：

- Paper B：任务队列、UAV 分配、vertiport / charging / corridor 资源。
- Paper E：TaskIR、LTL/STL、验证纠错；相关可引用基础包括 Lang2LTL、LTLCodeGen 和 ConformalNL2LTL [20] [21] [22]。
- Paper F：场景生成、coverage holes、危险场景压力测试。

### 4.6 工具集合

G1 的工具不需要一开始全做真实系统，可以先做可复现实验工具：

| Tool | 输入 | 输出 |
|------|------|------|
| `query_airspace` | 区域、时间、任务类型 | corridor、禁飞区、天气、容量 |
| `assign_uav` | 任务、UAV 状态、优先级 | UAV-task assignment |
| `plan_route` | 起点、终点、约束 | 路径或 `UNREACHABLE` |
| `verify_ltl_stl` | 任务规格、轨迹 | pass / fail / counterexample |
| `simulate_scenario` | scenario seed、策略 | success、collision、delay、risk |
| `risk_assess` | 任务和场景 | 风险等级、主要约束 |
| `explain_decision` | 决策轨迹 | 人类可读解释 |

### 4.7 Baselines

| Baseline | 说明 |
|----------|------|
| Direct LLM decision | LLM 直接给出调度/路径建议 |
| Prompt-only ReAct | ReAct 风格工具调用，但无类型约束和 verifier [6] |
| Toolformer / ToolLLM-style tool-use | 学会调用工具，但不做低空安全验证 [7] [8] |
| TrafficGPT-style orchestration | LLM 调用交通模型，但无 UAV 约束和形式验证 [1] |
| LLM+P / classical planner | LLM 转换问题，外部 planner 求解 [10] |
| VERA-UAV only | 只做语言到规格验证，不做云脑多工具调度 |
| CloudBrain-Agent full | LowAltitudeIR + tool-use + verifier + simulator feedback |

PlanBench 及后续关于 LLM planning 能力的批判性研究表明，单纯让 LLM 口头规划并不可靠，必须引入外部规划器、约束检查和可复现实验任务 [11] [12]。同时，AerialVLN 和 realistic UAV-VLN 工作可作为低空视觉语言导航的对标来源 [23] [24]；DriveLM、LMDrive、DriveVLM 和 LaMPilot 则可作为自动驾驶 VLM/LLM benchmark 与闭环决策范式的横向参照 [25] [26] [27] [28]。

### 4.8 评价指标

| 指标 | 含义 |
|------|------|
| Task success rate | 云脑任务完成比例 |
| Tool-call accuracy | 工具选择和参数是否正确 |
| Executable decision rate | 输出能否被调度器/规划器执行 |
| Safety violation rate | 是否违反禁飞区、距离、高度、deadline |
| Hallucination rate | 是否引用不存在实体、工具或状态 |
| Repair success rate | 验证失败后能否修复 |
| Simulator stress pass rate | 在 Paper F 危险场景中通过比例 |
| Latency | 单任务决策时间 |
| Generalization | 未见城市/未见任务/未见工具组合上的表现 |

### 4.9 预期创新点

1. 提出面向低空交通云脑的 `LowAltitudeIR` 与 typed tool-use agent 架构。
2. 将调度、路径规划、形式验证和场景仿真统一到 LLM agent 决策闭环。
3. 提出 verification-guided repair，使 LLM 不再只靠 prompt retry。
4. 构建低空交通云脑 benchmark，覆盖任务分解、工具调用、调度、验证和解释。

---

## 5. Paper G2：LowAltitudeGPT，低空交通领域 LLM 微调

### 5.1 题目

**LowAltitudeGPT: Instruction Tuning LLMs for Low-Altitude Traffic Decision Support**

### 5.2 目标

G2 才是模型微调论文。目标是把 G1 中的 agent 运行轨迹、人工规则、仿真反馈和验证修复数据蒸馏进本地开源模型，让模型成为低空交通云脑的 domain cognitive module。

候选投稿：T-ITS、IEEE T-IV、Applied Intelligence、Knowledge-Based Systems。T-ITS 更适合强调智能交通系统、交通运行和安全决策，T-IV 更适合强调智能车辆/无人系统模型与评测 [36] [37]。若模型训练和评测足够强，也可做 AAAI / IJCAI workshop 或主会扩展。

### 5.3 训练路线

推荐三阶段：

| 阶段 | 方法 | 数据 |
|------|------|------|
| SFT | LoRA / QLoRA 微调 [14] [15] | 低空交通问答、NL-to-IR、tool-call traces、应急解释 |
| Preference tuning | DPO / preference optimization [16] | 安全决策优于危险决策、可执行工具序列优于幻觉工具序列 |
| Verifiable RL | 基于验证器和仿真器的规则奖励 | 成功任务、低风险、低延迟、无 hallucination、通过 STL 验证 |

DeepSeek-R1 说明推理能力可以通过强化学习激励出来 [19]，但 G2 不应从零训练 reasoning model。更现实路线是：用 Qwen / DeepSeek / Llama 系开源模型作为 base，采用 LoRA/QLoRA 做参数高效微调，再用 verifier reward 做小规模对齐。

### 5.4 数据构建

数据不要只做聊天问答，而应分为 7 类：

| 数据类型 | 示例 |
|----------|------|
| Domain QA | “低空 corridor 容量不足时如何处理应急任务？” |
| NL-to-LowAltitudeIR | 自然语言任务到结构化 IR |
| Tool-call trace | 正确工具调用序列和参数 |
| Verification repair | 失败反例到修复后的 IR |
| Scheduling explanation | 调度结果解释 |
| Emergency response | 高速/城市应急场景处置 |
| Safety refusal | 不安全或信息不足时拒答/澄清 |

数据来源：

- 程序生成：Paper B/F 场景生成器产生任务、地图、状态和工具结果。
- 验证生成：Paper E 的 LTL/STL 失败样本和修复样本。
- 人工校对：抽样校正高风险样本，保证引用实体、约束和工具参数真实。
- Self-Instruct 扩展：用 self-instruct 思路扩充任务模板，但必须经过规则过滤和人工抽检 [17]。

### 5.5 模型选择

第一版建议：

- `Qwen2.5-7B/14B`：中文/英文、代码和工具调用能力较好 [18]。
- `DeepSeek-R1-Distill-Qwen-14B`：适合推理和验证修复 [19]。
- `Llama-3.1-8B`：英文基线和开源生态对比。

不建议第一阶段训练 70B 以上模型。论文重点不是模型规模，而是 **domain tool-use alignment** 和 **verification feedback training**。

### 5.6 评价指标

| 指标 | 含义 |
|------|------|
| IR exact match / field F1 | LowAltitudeIR 结构化输出质量 |
| Tool-call success | 工具名称、顺序、参数正确率 |
| Verified decision rate | 输出通过验证器比例 |
| Safety refusal accuracy | 对不安全/信息不足任务是否拒答或澄清 |
| Repair ability | 看到 counterexample 后的修复成功率 |
| Local deployment latency | 本地推理延迟和显存占用 |
| Cross-city generalization | 未见城市/场景的泛化 |

---

## 6. Paper G3：Multi-Agent Cloud Brain，多智能体协同云脑

### 6.1 题目

**Multi-Agent Cloud Brain for Cooperative Low-Altitude UAV Traffic Management**

### 6.2 目标

G3 从单个 agent 扩展到多 agent 协同。候选投稿：AAMAS、IJCAI、AAAI、T-ITS。

AAMAS 主会关注 autonomous agents and multiagent systems [38]，这与低空交通云脑中的多角色协作非常贴合。

### 6.3 Agent 分工

| Agent | 职责 |
|-------|------|
| Airspace Monitor | 监控 corridor、禁飞区、天气和容量 |
| Fleet Scheduler | 负责任务队列和 UAV 分配 |
| Safety Verifier | 负责 LTL/STL、风险和反例 |
| Scenario Tester | 调用 Paper F 场景生成器做 stress test |
| Emergency Coordinator | 负责应急响应和地面联动 |
| Human Interface Agent | 负责解释、澄清和人工确认 |

### 6.4 关键研究问题

1. 多 agent 是否比单 agent 更可靠？
2. 共享 memory 会不会传播错误？
3. 当两个 agent 冲突时，谁有最终决策权？
4. verifier 是否能作为仲裁器？
5. 多 agent 带来的延迟是否可接受？

### 6.5 创新点

G3 的创新不是“多个 GPT 互相聊天”，而是：

- 专职 agent 与低空交通工具绑定；
- 共享状态以 `LowAltitudeIR` 和 event log 表示；
- 安全仲裁由 verifier 和 simulator 完成；
- multi-agent disagreement 可转化为不确定性和人工介入信号。

---

## 7. Paper G4：World-Model/VLA 扩展，面向通用 AGI 能力迁移

### 7.1 总体定位

G4 是长期路线，不应在前两篇中夸大。建议表述为：

> **towards general embodied traffic intelligence**

而不是“实现 AGI”。

Voyager 的 open-ended embodied agent 和 SayCan 的语言到机器人 affordance grounding 说明，LLM 要走向 embodied intelligence，关键不是会聊天，而是能在环境反馈、技能库和行动约束中持续改进 [9] [13]。低空交通云脑可以把这种思路放进更安全、更可评估的交通运行域。

### 7.2 为什么这是 AGI 方向的合理入口

低空交通云脑天然包含通用 embodied intelligence 所需的若干能力：

- 空间理解：城市 3D 空间、障碍物、空域层级。
- 时间推理：任务队列、deadline、动态天气、交通事件演化。
- 工具使用：调度器、规划器、验证器、仿真器。
- 行动后果：错误决策会导致延误、风险或安全违规。
- 多智能体协作：UAV、地面车辆、人类操作员、监管规则。

PaLM-E、RT-2 和 OpenVLA 已经展示了从语言/视觉预训练迁移到 embodied action 的趋势 [29] [30] [31]。但低空交通云脑不应一开始做端到端 VLA，而应先用 agent + tools + verifier 建立安全认知架构。

### 7.3 长期技术路线

| 阶段 | 能力 | 技术 |
|------|------|------|
| G1 | 工具调用和验证闭环 | LLM agent + LowAltitudeIR |
| G2 | 领域模型 | SFT / LoRA / DPO / verifier reward |
| G3 | 多 agent 协同 | shared memory + verifier arbitration |
| G4 | 世界模型 | spatio-temporal prediction + simulator feedback |
| G5 | VLA / embodied policy | 多模态输入到行动建议，但仍由 safety layer 执行 |

AGI 转型的关键词应是：**generalization, continual learning, embodied reasoning, self-evaluation, tool creation**。不要写成“我们训练了一个 AGI 模型”。

---

## 8. 数据构建与训练方案

### 8.1 数据总表

| 数据集 | 来源 | 用途 |
|--------|------|------|
| LowAltitude-Instruction | 人工模板 + LLM 生成 + 人工抽检 | 自然语言任务理解 |
| LowAltitudeIR-Gold | 规则生成 + 人工校正 | IR 训练和评测 |
| ToolTrace-Bench | G1 agent 运行轨迹 | 工具调用 SFT |
| VerifyRepair-Bench | Paper E 反例修复 | 验证纠错训练 |
| ScenarioStress-Bench | Paper F 场景生成 | 危险场景泛化 |
| FleetOps-Bench | Paper B 调度仿真 | 任务队列和资源调度 |
| EmergencyOps-Bench | 高速/城市应急合成案例 | 应急决策 |

仿真层建议先采用轻量自建 simulator 保证可控变量，再用 AirSim 和 Flightmare 做视觉、动力学和闭环飞行补充验证 [32] [33]。这样 G1/G2 不依赖重型仿真器即可复现，后续又能自然扩展到更真实的 UAV 场景。

### 8.2 训练样本格式

建议统一为 JSONL：

```json
{
  "instruction": "优先处理医院附近应急配送，避开学校和临时禁飞区。",
  "state": {
    "uavs": "...",
    "airspace": "...",
    "tasks": "..."
  },
  "target_ir": {
    "intent": "emergency_delivery",
    "constraints": ["avoid_school", "avoid_no_fly_zone"]
  },
  "tool_trace": [
    {"tool": "query_airspace", "args": {"region": "hospital_zone"}},
    {"tool": "assign_uav", "args": {"priority": "emergency"}},
    {"tool": "verify_ltl_stl", "args": {"spec": "..."}}
  ],
  "verifier_feedback": "pass",
  "final_answer": "建议派遣 uav_12，经 corridor_B 绕开学校区域。"
}
```

### 8.3 训练阶段

1. **Prompt + RAG baseline**  
   不训练，先验证任务定义和工具 schema。

2. **SFT / LoRA**  
   训练模型输出 LowAltitudeIR 和 tool call traces。

3. **DPO / preference tuning**  
   偏好安全、可执行、少幻觉、低延迟的决策。

4. **Verifier reward alignment**  
   用验证器和仿真器结果作为规则奖励，强化修复能力。

5. **Distillation**  
   将强模型或多 agent 轨迹蒸馏到本地 7B/14B 模型。

---

## 9. 实验设计、baselines 与评价指标

### 9.1 G1 主实验

| 实验 | 目的 |
|------|------|
| Tool-use success | 测试工具选择和参数填写 |
| Verified planning | 测试调度/路径是否通过验证 |
| Repair loop | 测试反例反馈能否提升成功率 |
| Scenario stress test | 用 Paper F 危险场景测试鲁棒性 |
| Generalization | 测试未见城市、未见任务和未见工具组合 |

### 9.2 G2 微调实验

| 实验 | 目的 |
|------|------|
| Base vs LoRA vs QLoRA | 验证微调收益 |
| SFT vs DPO | 验证偏好对齐收益 |
| With / without verifier feedback | 验证安全反馈价值 |
| 7B vs 14B vs reasoning model | 验证本地部署成本/性能 tradeoff |
| Cross-scenario transfer | 验证从合成场景到应急场景迁移 |

### 9.3 Baselines

| Baseline | 说明 |
|----------|------|
| GPT/Qwen direct answer | 直接回答，无工具 |
| ReAct prompting | 推理-行动提示 [6] |
| Toolformer-style API calling | 工具调用但不做安全闭环 [7] |
| ToolLLM-style trained tool user | 开源工具调用训练基线 [8] |
| TrafficGPT-style traffic orchestration | LLM + traffic models [1] |
| LLM+P | LLM + external planner [10] |
| CloudBrain-Agent full | 本文方法 |

### 9.4 指标

| 指标 | 目标 |
|------|------|
| Task success | 云脑任务完成率 |
| Tool-call accuracy | 工具调用正确率 |
| IR field F1 | LowAltitudeIR 字段级准确率 |
| Hallucination rate | 不存在工具/实体/规则的比例 |
| Safety violation rate | 违反安全规则比例 |
| Repair success | 反例修复成功率 |
| Latency | 决策延迟 |
| Human trust score | 人类评审解释质量 |
| Generalization score | 未见场景泛化 |

---

## 10. 推荐投稿路径

### 10.1 首篇会议路线

**G1 首投 AAAI / IJCAI。**

论文类型：AI agent + planning + verification + transportation application。

核心贡献压成三条：

1. LowAltitudeIR 与低空交通 tool-use agent 架构。
2. Verification-guided repair loop。
3. Low-altitude cloud brain benchmark and evaluation protocol。

### 10.2 后续期刊路线

| 论文 | 投稿 |
|------|------|
| G2 LowAltitudeGPT | T-ITS / T-IV / Applied Intelligence |
| G3 Multi-Agent Cloud Brain | AAMAS -> T-ITS extension |
| G4 World-Model/VLA | ICRA / IROS / T-RO / long-term AGI-oriented venue |

### 10.3 不建议的路线

- 不建议第一篇就训练大模型。
- 不建议写“AGI 云脑”作为主标题。
- 不建议让 LLM 直接输出 UAV 控制动作。
- 不建议只做聊天问答数据集。
- 不建议忽略 verifier，否则安全关键场景说服力不足。

---

## 11. 参考文献

[1] Siyao Zhang, Daocheng Fu, Wenzhe Liang, Zhao Zhang, Bin Yu, Pinlong Cai, and Baozhen Yao. “TrafficGPT: Viewing, Processing and Interacting with Traffic Foundation Models.” *Transport Policy*, 150:95-105, 2024. DOI: 10.1016/j.tranpol.2024.03.006. URL: <https://www.sciencedirect.com/science/article/pii/S0967070X24000726>

[2] Sebastian Wandelt, Changhong Zheng, Shuang Wang, Yucheng Liu, and Xiaoqian Sun. “Large Language Models for Intelligent Transportation: A Review of the State of the Art and Challenges.” *Applied Sciences*, 14(17):7455, 2024. DOI: 10.3390/app14177455. URL: <https://www.mdpi.com/2076-3417/14/17/7455>

[3] Doaa Mahmud, Hadeel Hajmohamed, Shamma Almentheri, Shamma Alqaydi, Lameya Aldhaheri, Ruhul Amin Khalil, and Nasir Saeed. “Integrating LLMs With ITS: Recent Advances, Potentials, Challenges, and Future Directions.” *IEEE Transactions on Intelligent Transportation Systems*, 26(5):5674-5709, 2025. DOI: 10.1109/TITS.2025.3528116. URL: <https://ieeexplore.ieee.org/document/10851302>

[4] Zhonghang Li, Lianghao Xia, Jiabin Tang, Yong Xu, Lei Shi, Long Xia, Dawei Yin, and Chao Huang. “UrbanGPT: Spatio-Temporal Large Language Models.” arXiv:2403.00813, 2024. URL: <https://arxiv.org/abs/2403.00813>

[5] Yuan Yuan, Jingtao Ding, Jie Feng, Depeng Jin, and Yong Li. “UniST: A Prompt-Empowered Universal Model for Urban Spatio-Temporal Prediction.” *Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)*, 2024. DOI: 10.1145/3637528.3671662. URL: <https://arxiv.org/abs/2402.11838>

[6] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. “ReAct: Synergizing Reasoning and Acting in Language Models.” *International Conference on Learning Representations (ICLR)*, 2023. URL: <https://openreview.net/forum?id=WE_vluYUL-X>

[7] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. “Toolformer: Language Models Can Teach Themselves to Use Tools.” *Advances in Neural Information Processing Systems 36 (NeurIPS)*, 2023. URL: <https://proceedings.neurips.cc/paper_files/paper/2023/hash/d842425e4bf79ba039352da0f658a906-Abstract-Conference.html>

[8] Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu, and Maosong Sun. “ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs.” *International Conference on Learning Representations (ICLR)*, 2024. URL: <https://openreview.net/forum?id=dHng2O0Jjr>

[9] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. “Voyager: An Open-Ended Embodied Agent with Large Language Models.” arXiv:2305.16291, 2023. URL: <https://arxiv.org/abs/2305.16291>

[10] Bo Liu, Yuqian Jiang, Xiaohan Zhang, Qiang Liu, Shiqi Zhang, Joydeep Biswas, and Peter Stone. “LLM+P: Empowering Large Language Models with Optimal Planning Proficiency.” arXiv:2304.11477, 2023. URL: <https://arxiv.org/abs/2304.11477>

[11] Karthik Valmeekam, Matthew Marquez, Alberto Olmo, Sarath Sreedharan, and Subbarao Kambhampati. “PlanBench: An Extensible Benchmark for Evaluating Large Language Models on Planning and Reasoning about Change.” *Advances in Neural Information Processing Systems 36 (NeurIPS) Datasets and Benchmarks Track*, 2023. URL: <https://openreview.net/forum?id=YXogl4uQUO>

[12] Karthik Valmeekam, Alberto Olmo, Sarath Sreedharan, and Subbarao Kambhampati. “On the Planning Abilities of Large Language Models: A Critical Investigation.” *Advances in Neural Information Processing Systems 36 (NeurIPS)*, 2023. URL: <https://arxiv.org/abs/2305.15771>

[13] Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Daniel Ho, et al. “Do As I Can, Not As I Say: Grounding Language in Robotic Affordances.” *Conference on Robot Learning (CoRL)*, PMLR 205, 2022. URL: <https://proceedings.mlr.press/v205/ahn23a.html>

[14] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. “LoRA: Low-Rank Adaptation of Large Language Models.” *International Conference on Learning Representations (ICLR)*, 2022. URL: <https://openreview.net/forum?id=nZeVKeeFYf9>

[15] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. “QLoRA: Efficient Finetuning of Quantized LLMs.” *Advances in Neural Information Processing Systems 36 (NeurIPS)*, 2023. URL: <https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html>

[16] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. “Direct Preference Optimization: Your Language Model is Secretly a Reward Model.” *Advances in Neural Information Processing Systems 36 (NeurIPS)*, 2023. URL: <https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html>

[17] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. “Self-Instruct: Aligning Language Models with Self-Generated Instructions.” *Annual Meeting of the Association for Computational Linguistics (ACL)*, 2023. URL: <https://aclanthology.org/2023.acl-long.754/>

[18] Qwen Team. “Qwen2.5 Technical Report.” arXiv:2412.15115, 2024. URL: <https://arxiv.org/abs/2412.15115>

[19] DeepSeek-AI. “DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.” arXiv:2501.12948, 2025. URL: <https://arxiv.org/abs/2501.12948>

[20] Bo Liu, Yuqian Jiang, Xiaohan Zhang, Qiang Liu, Shiqi Zhang, Joydeep Biswas, and Peter Stone. “Lang2LTL: Translating Natural Language Commands to Temporal Specification with Large Language Models.” *Conference on Robot Learning (CoRL)*, PMLR 229, 2023. URL: <https://proceedings.mlr.press/v229/liu23d.html>

[21] Behrad Rabiei, Mahesh Kumar A. R., Zhirui Dai, Surya L. S. R. Pilla, Qiyue Dong, and Nikolay Atanasov. “LTLCodeGen: Code Generation of Syntactically Correct Temporal Logic for Robot Task Planning.” arXiv:2503.07902, 2025. URL: <https://arxiv.org/abs/2503.07902>

[22] Jun Wang, David Smith Sundarsingh, Jyotirmoy V. Deshmukh, and Yiannis Kantaros. “ConformalNL2LTL: Translating Natural Language Instructions into Temporal Logic Formulas with Conformal Correctness Guarantees.” arXiv:2504.21022, 2025. URL: <https://arxiv.org/abs/2504.21022>

[23] Shubo Liu, Hongsheng Zhang, Yuankai Qi, Peng Wang, Yanning Zhang, and Qi Wu. “AerialVLN: Vision-and-Language Navigation for UAVs.” *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2023, pp. 15384-15394. URL: <https://openaccess.thecvf.com/content/ICCV2023/html/Liu_AerialVLN_Vision-and-Language_Navigation_for_UAVs_ICCV_2023_paper.html>

[24] Xiangyu Wang, Donglin Yang, Ziqin Wang, Hohin Kwan, Jinyu Chen, Wenjun Wu, Hongsheng Li, Yue Liao, and Si Liu. “Towards Realistic UAV Vision-Language Navigation: Platform, Benchmark, and Methodology.” *International Conference on Learning Representations (ICLR)*, 2025. URL: <https://openreview.net/forum?id=rUvCIvI4eB>

[25] Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Jens Beisswenger, Ping Luo, Andreas Geiger, and Hongyang Li. “DriveLM: Driving with Graph Visual Question Answering.” arXiv:2312.14150, 2023. URL: <https://arxiv.org/abs/2312.14150>

[26] Hao Shao, Yuxuan Hu, Letian Wang, Steven L. Waslander, Yu Liu, and Hongsheng Li. “LMDrive: Closed-Loop End-to-End Driving with Large Language Models.” *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2024. URL: <https://arxiv.org/abs/2312.07488>

[27] Xiaoyu Tian, Junru Gu, Bailin Li, Yicheng Liu, Yang Wang, Zhiyong Zhao, Kun Zhan, Peng Jia, Xianpeng Lang, and Hang Zhao. “DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models.” arXiv:2402.12289, 2024. URL: <https://arxiv.org/abs/2402.12289>

[28] Yunsheng Ma, Can Cui, Xu Cao, Wenqian Ye, Peiran Liu, Juanwu Lu, Amr Abdelraouf, Rohit Gupta, Kyungtae Han, Aniket Bera, James M. Rehg, and Ziran Wang. “LaMPilot: An Open Benchmark Dataset for Autonomous Driving with Language Model Programs.” *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2024, pp. 15141-15151. URL: <https://openaccess.thecvf.com/content/CVPR2024/html/Ma_LaMPilot_An_Open_Benchmark_Dataset_for_Autonomous_Driving_with_Language_CVPR_2024_paper.html>

[29] Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, and Pete Florence. “PaLM-E: An Embodied Multimodal Language Model.” *International Conference on Machine Learning (ICML)*, PMLR 202, 2023. URL: <https://proceedings.mlr.press/v202/driess23a.html>

[30] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, and others. “RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control.” arXiv:2307.15818, 2023. URL: <https://arxiv.org/abs/2307.15818>

[31] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. “OpenVLA: An Open-Source Vision-Language-Action Model.” arXiv:2406.09246, 2024. URL: <https://arxiv.org/abs/2406.09246>

[32] Shital Shah, Debadeepta Dey, Chris Lovett, and Ashish Kapoor. “AirSim: High-Fidelity Visual and Physical Simulation for Autonomous Vehicles.” *Field and Service Robotics*, Springer Proceedings in Advanced Robotics, 2017; arXiv:1705.05065. URL: <https://arxiv.org/abs/1705.05065>

[33] Yunlong Song, Selim Naji, Elia Kaufmann, Antonio Loquercio, and Davide Scaramuzza. “Flightmare: A Flexible Quadrotor Simulator.” *Conference on Robot Learning (CoRL)*, PMLR 155, 2021. URL: <https://proceedings.mlr.press/v155/song21a.html>

[34] AAAI. “AAAI-26 Main Technical Track: Call for Papers.” URL: <https://aaai.org/conference/aaai/aaai-26/main-technical-track-call/>

[35] IJCAI-ECAI 2026. “Call for Papers — AI and Robotics Special Track.” URL: <https://2026.ijcai.org/ijcai-ecai-2026-call-for-papers-ai-and-robotics/>

[36] IEEE Intelligent Transportation Systems Society. “IEEE Transactions on Intelligent Transportation Systems (T-ITS): Scope.” URL: <https://ieee-itss.org/pub/t-its/>

[37] IEEE Intelligent Transportation Systems Society. “IEEE Transactions on Intelligent Vehicles.” URL: <https://ieee-itss.org/pub/t-iv/>

[38] AAMAS 2026. “Call for Papers — Main Track.” URL: <https://cyprusconferences.org/aamas2026/call-for-papers-main-track/>

---

## 附录：12 个月推进计划

### 第 1-2 个月：冻结 G1 问题和接口

- 冻结 CloudBrain-Agent 题目、摘要和三条贡献。
- 定义 LowAltitudeIR v0.1。
- 定义工具 API：airspace、scheduler、planner、verifier、simulator、risk。
- 搭建 100-200 个小规模任务样本验证 pipeline。

### 第 3-4 个月：构建 CloudBrain-Bench

- 生成 1000+ 低空交通任务。
- 覆盖正常调度、应急配送、禁飞区避让、充电瓶颈、corridor 拥堵和不可满足任务。
- 标注 gold LowAltitudeIR、gold tool trace、expected decision。

### 第 5-6 个月：实现 G1 baselines

- Direct LLM。
- ReAct prompting。
- Tool-use without verifier。
- TrafficGPT-style orchestration。
- LLM+P。
- VERA-UAV only。

### 第 7-8 个月：实现 CloudBrain-Agent full

- 加入 typed tool schema。
- 加入 verifier feedback。
- 加入 simulator stress test。
- 加入 safety memory 和 repair loop。

### 第 9-10 个月：主实验

- 跑 task success、tool-call accuracy、safety violation、repair success、latency。
- 跑未见城市、未见任务和危险场景泛化。
- 做消融：no IR、no verifier、no simulator、no memory、no repair。

### 第 11 个月：G2 微调预实验

- 收集 G1 tool traces。
- LoRA 微调 Qwen / DeepSeek。
- 对比 base vs SFT vs DPO。
- 判断是否足以形成 G2。

### 第 12 个月：AAAI/IJCAI 初稿

- 写 G1 会议论文。
- 附录放 LowAltitudeIR schema、工具定义、数据生成规则。
- 确保 reproducibility checklist、代码、数据和实验种子准备完整。
