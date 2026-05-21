---
title: "Paper G1 完整论文方案 v1：面向低空交通云脑的可验证 LLM Agent"
description: "完整规划 CloudBrain-Agent 第一篇会议论文的研究问题、投稿定位、算法设计、数据构建、模型选择、本地部署、实验方案、评价指标、预期结论、图表设计、风险控制与执行计划。"
pubDate: 2026-05-20
tags: ["Paper G1", "CloudBrain-Agent", "低空交通云脑", "LLM Agent", "MCP", "Tool Use", "AAAI", "IJCAI", "UAV", "形式化验证"]
category: Tech
---

# Paper G1 完整论文方案 v1：面向低空交通云脑的可验证 LLM Agent

> 核心判断：第一篇论文不要写成“微调一个低空交通大模型”，而要写成 **可验证、可复现、可部署的低空交通 LLM Agent 方法论文**。  
> 推荐题目：**CloudBrain-Agent: Tool-Augmented and Verification-Guided LLM Agents for Low-Altitude Traffic Operation**。

---

## 1. 论文定位与投稿判断

### 1.1 一句话定位

本文研究低空交通云脑中的大模型智能体：给定自然语言任务、城市低空空域状态、UAV fleet 状态和安全约束，LLM agent 如何通过结构化中间表示、工具调用、形式化验证和仿真反馈，生成安全、可执行、可解释的低空交通运行决策。

### 1.2 推荐投稿

首选：**AAAI / IJCAI 主会**。  
备选：AAMAS、IROS/ICRA workshop、T-ITS 后续扩展。

按照 2026-05-20 的时间点，具体届次需要对齐下一轮 AAAI/IJCAI CFP；本文仍按 AAAI/IJCAI 主会风格设计，因为 AAAI 强调 AI 方法、应用领域和 reproducibility，IJCAI-ECAI AI and Robotics track 明确关注 robot agents、generative AI、reasoning、structured modeling 和行动后果 [1] [2]。

### 1.3 为什么这篇比“低空交通大模型微调”更适合先做

直接微调一个 LowAltitudeGPT 会遇到三个审稿风险：

1. LoRA、QLoRA、DPO 已经是成熟训练范式，仅换领域数据不够构成主会贡献 [3] [4] [5]。
2. 低空交通是安全关键系统，LLM 直接输出控制动作很难说服审稿人。
3. 真实低空交通运行数据稀缺，第一篇如果把贡献压在“大模型训练”上，会被问数据规模、训练预算和模型新意。

因此第一篇应该聚焦 **Agent + Tools + Verifier + Simulator Feedback**。大模型不是最终控制器，而是任务理解、工具编排、反例修复和解释层。这个设定与 ReAct、ToolLLM、LLM+P 等 agent/tool-use/planning 工作自然衔接 [6] [7] [8]，也能接住 TrafficGPT 对交通 foundation model 与 LLM 交互的讨论 [9]。

---

## 2. 摘要草案

城市低空交通运行需要在动态任务、有限空域资源、UAV 状态约束和安全规则之间进行实时决策。大语言模型具备自然语言理解和复杂任务分解能力，但直接用于 UAV 调度和路径规划会产生幻觉、不可执行计划和安全违规。本文提出 **CloudBrain-Agent**，一个面向低空交通云脑的工具增强与验证引导 LLM agent 框架。CloudBrain-Agent 将自然语言任务和系统状态解析为类型化 `LowAltitudeIR`，调用空域查询、UAV 分配、路径规划、LTL/STL 验证、场景仿真和风险评估工具，并利用 verifier counterexample 与 simulation feedback 迭代修复决策。我们构建 **CloudBrain-Bench**，覆盖应急配送、巡检、禁飞区避让、corridor 拥堵、充电瓶颈、多模式 fallback 和不可满足任务。实验将比较 direct LLM、prompt-only ReAct、tool-use without verification、LLM+P、TrafficGPT-style orchestration 与 CloudBrain-Agent full。预注册预期是：CloudBrain-Agent 在 task success、executable decision rate、safety violation rate、hallucination rate 和 repair success 上显著优于 prompt-only 与 tool-only baselines，同时保持可接受的本地部署延迟。

---

## 3. 研究问题与核心假设

### 3.1 研究问题

**RQ1：** LLM agent 能否在低空交通任务中稳定生成类型正确、工具可执行的决策链？

**RQ2：** 形式化验证和仿真反馈是否能显著降低 LLM 的不可执行计划、安全违规和幻觉？

**RQ3：** 相比直接微调垂类模型，通用 LLM + typed IR + MCP/tools + verifier 的方案是否能更快形成可复现、可部署、可扩展的研究系统？

**RQ4：** 本地开源模型是否能在 teacher API 生成的数据和规则反馈下接近闭源强模型表现，并支撑后续 LowAltitudeGPT 论文？

### 3.2 核心假设

H1：typed `LowAltitudeIR` 能显著提升结构化输出质量和 tool-call accuracy。  
H2：verification-guided repair 能显著提升 executable decision rate，并降低 safety violation rate。  
H3：simulator feedback 对未见危险场景泛化最关键。  
H4：第一阶段无需训练垂类 foundation model；通用模型 + agent 工具层 + verifier 后处理足以完成 G1 论文。  
H5：本地 Qwen3 / DeepSeek-R1-Distill 模型通过 vLLM 部署后，可作为可复现主实验模型；GPT-5.2 这类 API 模型作为 teacher 和性能上限 [10] [11] [12]。

---

## 4. 论文贡献设计

建议最终论文贡献写成三条，避免散：

1. **CloudBrain-Agent framework**  
   提出面向低空交通云脑的 typed tool-use LLM agent，将自然语言任务、城市空域状态、UAV fleet 状态和安全约束统一到 `LowAltitudeIR`。

2. **Verification-guided repair for low-altitude traffic operation**  
   将 LTL/STL verifier、route planner 和 simulator 的失败反馈转化为结构化反例，驱动 LLM 修复工具调用、任务约束和路径/调度建议。

3. **CloudBrain-Bench and evaluation protocol**  
   构建低空交通云脑 benchmark，覆盖 tool-call accuracy、executable decision、safety violation、repair success、generalization、latency 和 human trust 等指标。

不建议把贡献写成“我们训练了一个低空交通大模型”。微调可以作为实验扩展或下一篇 G2。

### 4.1 二轮调研后的论文定位矩阵

联网调研后，G1 的最佳切入点应更明确地放在 **domain-grounded agent evaluation + safety verification**，而不是泛泛的 LLM 应用。AgentBench 证明 LLM agent 需要在交互环境中评测 reasoning 和 decision-making [34]；BFCL 说明 function calling 需要检查函数选择、参数、并行调用和 relevance detection [35]；$\tau$-bench 进一步强调多轮交互、API、领域 policy 和一致性指标 `pass^k` [36]；ToolSandbox 则指出 state dependency、canonicalization 和 insufficient information 是工具型 agent 的关键难点 [37]。

这些工作给 G1 的启发是：CloudBrain-Bench 不能只评测“有没有输出 JSON”，而要评测 agent 在低空交通工具链中的 **状态更新、规则遵守、工具依赖、失败修复和多轮一致性**。

| 已有方向 | 代表工作 | 局限 | G1 的差异 |
|----------|----------|------|-----------|
| 通用 agent benchmark | AgentBench、$\tau$-bench、ToolSandbox [34] [36] [37] | 不包含低空交通安全约束和 UAV 工具链 | 面向 UTM/UAV 的 domain tools、policy、verifier |
| function calling benchmark | BFCL [35] | 关注函数调用正确性，不关心物理可执行和安全 | 工具调用后必须经过 planner/verifier/simulator |
| LLM + traffic | TrafficGPT、ITS LLM survey [9] [13] [14] | 多聚焦地面交通或交通模型交互 | 扩展到低空空域、UAV fleet 和形式化安全 |
| NL-to-LTL / robot task spec | Lang2LTL、LTLCodeGen、ConformalNL2LTL [21] [22] [23] | 主要解决规格生成 | 把规格验证放进完整云脑决策闭环 |
| UTM/UAM simulation | NASA TCL4、CORUS-XUAM、AAM-Gym [38] [39] [40] | 通常不研究 LLM agent 工具编排 | 用 UTM/UAM 概念和场景支撑 CloudBrain-Bench |

---

## 5. 相关工作框架

### 5.1 LLM for transportation

TrafficGPT 说明 LLM 可以作为交通 foundation models 的交互和处理入口，但也指出交通数值数据、仿真和模型交互不能只靠纯文本生成 [9]。近期 ITS 综述进一步将 LLM 放在交通语义接口、决策辅助和多源数据理解的位置 [13] [14]。UrbanGPT 与 UniST 则代表城市时空 foundation model 方向，适合支撑城市状态理解，但它们不是低空 UAV 运行工具链 [15] [16]。

### 5.2 LLM agents and tool use

ReAct 将 reasoning trace 和 action 交织起来，是本文 agent loop 的基础 [6]。Toolformer 与 ToolLLM 证明 LLM 可以学习 API/tool 使用，但它们不解决低空交通安全验证和任务可执行性问题 [7] [17]。MCP 和 OpenAI Agents SDK 提供了更标准的工具连接方式，有利于把 scheduler、planner、verifier 和 simulator 做成可替换工具 [18] [19]。

二轮调研后，相关工作还应加入 agent 评测体系：AgentBench 是多环境 LLM-as-agent benchmark [34]；BFCL 专门评估 function calling 和 relevance detection [35]；$\tau$-bench 用多轮 user-agent-tool 交互和 `pass^k` 评估可靠性 [36]；ToolSandbox 强调工具执行状态、隐式依赖和信息不足场景 [37]。G1 的评测协议应吸收这些思想，但把环境换成低空交通云脑。

### 5.3 LLM planning and formal verification

LLM+P 和 PlanBench 表明 LLM 单独做 planning 并不可靠，需要与外部规划器、形式表示和评测协议结合 [8] [20]。Lang2LTL、LTLCodeGen 和 ConformalNL2LTL 说明自然语言到 temporal logic 的翻译正在发展，但它们主要关注规格生成，不完整覆盖低空交通云脑中的调度、路径、仿真和风险闭环 [21] [22] [23]。Spot 与 RTAMT 可以分别作为 LTL/STL 验证工具 [24] [25]。

### 5.4 UAV, UTM, and simulation data

FAA UTM 将低空无人机交通管理定义为支持 flight planning、authorization、surveillance 和 conflict management 的协同生态 [26]。FAA UAS Facility Maps 提供受控空域中 Part 107 操作可快速审批的高度参考，适合做空域规则 proxy [27]。OSM/Overpass、NYC TLC OD 数据、SUMO、AirSim 和 Flightmare 可以共同支撑 synthetic-to-real benchmark [28] [29] [30] [31] [32]。

为增强低空交通可信度，G1 应进一步引用 NASA TCL4 Nevada flight tests：该测试包含 BVLOS、urban canyon、weather front、concert emergency response 和 CNS issue 场景，适合作为场景 taxonomy 与 human-system 信息质量讨论的来源 [38]。欧洲 CORUS-XUAM 则提供 U-space/UAM operational concept、U3/U4 service models、ATM-U-space coordination、vertiport guidance 和 human-in-the-loop 证据 [39]。AAM-Gym 可作为 advanced air mobility AI testbed 的仿真对照，尤其是 corridor separation assurance [40]。

---

## 6. Problem Formulation

### 6.1 系统状态

在离散决策时刻 $t$，低空交通云脑接收系统状态：

$$
S_t = \langle \mathcal{U}_t, \mathcal{R}_t, \mathcal{A}_t, \mathcal{M}, \mathcal{C}_t, \mathcal{H}_t \rangle
$$

其中：

- $\mathcal{U}_t$：UAV 集合，每架 UAV 有位置、电量、载重、速度、任务状态。
- $\mathcal{R}_t$：任务集合，包括配送、巡检、应急响应、返航、充电。
- $\mathcal{A}_t$：空域状态，包括 corridor、禁飞区、高度层、天气、容量。
- $\mathcal{M}$：城市地图，包括 OSM 路网、POI、建筑、功能区。
- $\mathcal{C}_t$：安全和运行约束，包括 LTL/STL、deadline、distance、energy。
- $\mathcal{H}_t$：历史事件、失败案例、human feedback 和 verifier feedback。

自然语言指令记为 $q_t$。目标是生成可执行决策：

$$
\pi_t = \langle z_t, a_{1:k}, y_t, e_t \rangle
$$

其中 $z_t$ 是 `LowAltitudeIR`，$a_{1:k}$ 是工具调用序列，$y_t$ 是调度/路径/风险决策，$e_t$ 是解释。

### 6.2 安全可执行目标

一个决策 $\pi_t$ 被认为成功，当且仅当同时满足：

1. **Schema validity**：$z_t$ 满足 `LowAltitudeIR` 类型约束。
2. **Tool executability**：所有工具调用参数合法且返回非错误结果。
3. **Planning feasibility**：调度和路径规划可执行。
4. **Temporal safety**：LTL/STL 规格通过验证。
5. **Simulation robustness**：在指定 scenario seeds 中不触发碰撞、禁飞区入侵或 deadline 违约。
6. **Human interpretability**：解释不包含不存在实体、工具或规则。

形式上：

$$
\text{Success}(\pi_t) =
\mathbb{1}[
V_\text{schema}(z_t)
\land V_\text{tool}(a_{1:k})
\land V_\text{plan}(y_t)
\land V_\text{logic}(y_t)
\land V_\text{sim}(y_t)
]
$$

### 6.3 本文不做的事情

- 不让 LLM 直接输出低层控制量。
- 不宣称真实空域可直接部署。
- 不把合成数据伪装成真实运营数据。
- 不从零训练低空交通 foundation model。

---

## 7. 方法：CloudBrain-Agent

### 7.1 总体架构

```text
User instruction + System state
  -> Context builder / RAG
  -> LLM planner
  -> LowAltitudeIR
  -> Tool router
  -> Scheduler / Route planner / Verifier / Simulator / Risk assessor
  -> Counterexample & robustness feedback
  -> Repair agent
  -> Final verified decision + explanation
```

### 7.2 LowAltitudeIR

`LowAltitudeIR` 是论文的关键。它要比普通 JSON 输出更严格，必须能连接工具和验证器。

```json
{
  "task_id": "task_00042",
  "intent": "emergency_delivery",
  "priority": "high",
  "entities": {
    "origin": "hospital_A",
    "destination": "accident_site_3",
    "candidate_uavs": ["uav_03", "uav_07"]
  },
  "constraints": {
    "deadline_sec": 600,
    "avoid_zones": ["school_zone_2", "nfz_temp_1"],
    "altitude_min_m": 30,
    "altitude_max_m": 120,
    "min_separation_m": 10,
    "battery_reserve_ratio": 0.2
  },
  "tool_plan": [
    {"tool": "query_airspace", "args": {"region": "downtown"}},
    {"tool": "assign_uav", "args": {"objective": "min_delay_safe"}},
    {"tool": "plan_route", "args": {"planner": "astar_3d"}},
    {"tool": "verify_ltl_stl", "args": {"logic": ["avoid_nfz", "meet_deadline"]}},
    {"tool": "simulate_scenario", "args": {"stress_level": "medium"}}
  ],
  "fallback_policy": "ground_transfer_or_human_confirm"
}
```

字段级约束：

| 字段 | 类型 | 约束 |
|------|------|------|
| `intent` | enum | delivery / patrol / inspection / emergency / return / charge |
| `priority` | enum | low / normal / high / critical |
| `entities` | object | 必须引用地图或 UAV 状态中存在的实体 |
| `constraints` | object | 必须能翻译成 planner/verifier 输入 |
| `tool_plan` | list | 工具名必须来自 registry，参数必须符合 schema |
| `fallback_policy` | enum | unreachable、unsafe、timeout 时触发 |

### 7.2.1 LowAltitudeIR v0.1 详细字段规范

第一版不要把 IR 设计得过大，而要保证每个字段都能被工具消费、被指标评估、被错误分析归因。建议将 IR 分成 9 个顶层字段：

| 顶层字段 | 必填 | 类型 | 说明 | 失败会影响的指标 |
|----------|------|------|------|------------------|
| `task_id` | 是 | string | 数据集中唯一任务 ID | traceability |
| `intent` | 是 | enum | 任务意图：delivery、inspection、patrol、emergency、return、charge、monitoring | IR field F1 |
| `priority` | 是 | enum | low、normal、high、critical | policy compliance |
| `entities` | 是 | object | origin、destination、candidate_uavs、sensitive_zones、handoff_points | hallucination rate |
| `constraints` | 是 | object | 时间、高度、距离、电量、禁飞区、容量、天气风险 | safety violation rate |
| `tool_plan` | 是 | list | 工具调用 DAG 的线性化计划 | tool-call accuracy |
| `verification_specs` | 是 | object | LTL/STL 规格和可解释自然语言规则 | verified decision rate |
| `fallback_policy` | 是 | enum | ground_transfer、wait、human_confirm、safe_refusal | safe refusal accuracy |
| `explanation_plan` | 否 | object | 解释需要引用的工具结果和约束 | human trust score |

实体字段建议具体到：

| 字段 | 示例 | 检查方式 |
|------|------|----------|
| `origin` | `hospital_A` | 必须存在于 `city_state.entities` |
| `destination` | `accident_site_3` | 必须存在于 task 或 map |
| `candidate_uavs` | `["uav_03", "uav_07"]` | 必须存在于 `uav_state` 且 status 可用 |
| `avoid_zones` | `["school_zone_2", "nfz_temp_1"]` | 必须存在于 airspace/map |
| `handoff_points` | `["metro_station_4"]` | multimodal fallback 时必填 |

约束字段建议具体到：

| 字段 | 单位 | 默认 | 说明 |
|------|------|------|------|
| `deadline_sec` | second | null | 无 deadline 时为空 |
| `altitude_min_m` | meter | 30 | 最低飞行高度 |
| `altitude_max_m` | meter | 120 | 最大高度，受空域 proxy 约束 |
| `min_separation_m` | meter | 10 | 与障碍物/UAV/sensitive zone 的最小距离 |
| `battery_reserve_ratio` | ratio | 0.2 | 到达后最低剩余电量比例 |
| `max_risk_level` | enum | medium | low、medium、high |
| `corridor_capacity_required` | int | 1 | 占用 corridor 的最小容量 |

### 7.2.2 LowAltitudeIR 校验顺序

IR 校验要分层，方便做错误分析：

1. **JSON validity**：是否能解析为 JSON。
2. **Schema validity**：字段类型、enum、必填项是否正确。
3. **Entity grounding**：所有实体是否存在于当前状态。
4. **Constraint grounding**：约束是否能转成 planner/verifier 参数。
5. **Tool dependency**：工具输入是否依赖前序工具输出。
6. **Policy compatibility**：priority、fallback、human confirm 是否符合规则。

每一层失败都要写入 `error_type`，例如：

```json
{
  "valid": false,
  "stage": "entity_grounding",
  "error_type": "nonexistent_destination",
  "field": "entities.destination",
  "value": "hospital_X",
  "allowed_entities": ["hospital_A", "hospital_B", "accident_site_3"]
}
```

### 7.3 工具注册表

第一版工具应该做成 Python 函数，之后包装成 MCP server。MCP 的优势是标准化 tools/context 接口，后续可以让不同模型和 agent runtime 复用同一组工具 [18] [19]。

| Tool | 必做 | 输入 | 输出 | 失败类型 |
|------|------|------|------|----------|
| `query_city_state` | 是 | region, time | POI, buildings, ground graph | unknown_region |
| `query_airspace` | 是 | region, altitude, time | corridor, NFZ, ceiling | restricted_airspace |
| `assign_uav` | 是 | task, UAV states | selected UAV / none | no_available_uav |
| `plan_route` | 是 | start, goal, constraints | path / unreachable | no_path |
| `verify_ltl_stl` | 是 | path, temporal specs | pass/fail/counterexample | spec_violation |
| `simulate_scenario` | 是 | decision, scenario seed | success/risk/collision | sim_failure |
| `risk_assess` | 是 | decision, state | risk score, reasons | high_risk |
| `explain_decision` | 可选 | decision trace | explanation | hallucinated_explanation |

### 7.3.1 工具 API 合约

所有工具统一返回：

```json
{
  "ok": true,
  "tool": "plan_route",
  "request_id": "tool_00042_03",
  "result": {},
  "warnings": [],
  "error": null,
  "provenance": {
    "input_hash": "sha256:...",
    "data_sources": ["city_osm", "airspace_rules"],
    "timestamp": "2026-05-20T12:00:00Z"
  }
}
```

失败时：

```json
{
  "ok": false,
  "tool": "plan_route",
  "request_id": "tool_00042_03",
  "result": null,
  "warnings": [],
  "error": {
    "type": "no_path",
    "message": "No feasible path avoiding nfz_temp_1 within altitude range.",
    "recoverable": true,
    "suggested_actions": ["relax_deadline", "choose_ground_transfer", "human_confirm"]
  },
  "provenance": {
    "input_hash": "sha256:...",
    "data_sources": ["city_grid", "airspace_rules"]
  }
}
```

具体工具输入输出：

| Tool | 关键输入字段 | 关键输出字段 | 可恢复失败 | 不可恢复失败 |
|------|--------------|--------------|------------|--------------|
| `query_city_state` | `region_id`, `bbox`, `time` | `pois`, `buildings`, `roads`, `sensitive_zones` | `partial_map` | `unknown_region` |
| `query_airspace` | `bbox`, `altitude_range`, `time` | `corridors`, `nfz`, `capacity`, `ceiling` | `capacity_low` | `restricted_airspace` |
| `assign_uav` | `task`, `uav_states`, `objective` | `uav_id`, `assignment_score`, `reason` | `low_battery_candidates` | `no_available_uav` |
| `plan_route` | `start`, `goal`, `avoid`, `altitude_range` | `waypoints`, `length_m`, `eta_sec`, `energy_est` | `deadline_risk` | `no_path` |
| `verify_ltl_stl` | `trajectory`, `specs` | `pass`, `violations`, `robustness`, `counterexample` | `negative_robustness` | `invalid_spec` |
| `simulate_scenario` | `decision`, `scenario_seed`, `stress_level` | `success`, `events`, `min_distance`, `delay`, `risk` | `near_miss` | `collision` |
| `risk_assess` | `decision`, `weather`, `traffic`, `history` | `risk_score`, `risk_level`, `top_reasons` | `medium_risk` | `high_risk_no_override` |

### 7.3.2 工具依赖 DAG

工具调用不是任意序列，应满足依赖：

```text
query_city_state
  -> query_airspace
      -> assign_uav
          -> plan_route
              -> verify_ltl_stl
                  -> simulate_scenario
                      -> risk_assess
                          -> explain_decision
```

允许跳过的情况：

- `simulate_scenario` 可在 dev-mini 关闭，但主实验必须开启。
- `risk_assess` 可合并到 `simulate_scenario`，但论文指标仍单独报告。
- `explain_decision` 不影响 task success，但影响 human trust 和 hallucination。

### 7.3.3 最小实现版本

第一版每个工具都可以先 deterministic：

| Tool | 最小算法 | 复杂版本 |
|------|----------|----------|
| `query_city_state` | 从 JSON/GeoJSON 读取实体 | OSM/Overture 动态查询 |
| `query_airspace` | 规则模板 + 多边形相交 | UTM/U-space 服务模拟 |
| `assign_uav` | greedy min ETA with battery filter | MILP / Lyapunov scheduler |
| `plan_route` | 3D A* grid | RRT* / MPC-lite |
| `verify_ltl_stl` | 手写规则 + RTAMT/Spot | 完整 temporal logic monitor |
| `simulate_scenario` | discrete-time kinematics | AirSim/Flightmare |
| `risk_assess` | weighted rule score | learned risk model |

### 7.4 Verification-guided repair

CloudBrain-Agent 的关键不是一次性生成，而是修复闭环：

```text
for i in 1..K:
  z_i = LLM(q, S, feedback_{i-1})
  if not schema_valid(z_i):
      feedback_i = schema_error(z_i)
      continue
  trace_i = execute_tools(z_i)
  verdict_i = verify_and_simulate(trace_i)
  if verdict_i.pass:
      return decision_i
  feedback_i = compress_counterexample(verdict_i)
return safe_refusal_or_human_confirm
```

反例反馈要结构化，不能只是“失败了”。例如：

```json
{
  "failure_type": "stl_robustness_negative",
  "violated_constraint": "always distance_to_school_zone > 30m",
  "counterexample_time_sec": 142,
  "offending_segment": ["p17", "p18", "p19"],
  "suggested_repair": "increase detour radius or choose corridor_C"
}
```

### 7.5 Safety memory

Safety memory 记录三类信息：

1. **Known unsafe patterns**：例如低电量 + 高风速 + deadline 紧张。
2. **Repair cases**：失败 IR、反例、成功修复 IR。
3. **Human interventions**：人工确认、拒绝、改派。

第一篇不需要做复杂长期记忆，只需实现 retrieval：给定当前任务，检索相似失败案例作为 few-shot repair context。

### 7.6 算法伪代码

论文主文建议放一个简化算法，附录放完整版本。

```text
Algorithm 1: CloudBrain-Agent
Input:
  q: natural-language instruction
  S: low-altitude traffic state
  T: typed tool registry
  R: rule and memory retriever
  K: maximum repair iterations

1: C <- BuildContext(q, S, R)
2: feedback <- null
3: for k = 0 ... K do
4:     z <- LLM_Generate_IR(q, C, feedback)
5:     schema_report <- ValidateIR(z, S, T)
6:     if schema_report fails then
7:         feedback <- Compress(schema_report)
8:         continue
9:     trace <- ExecuteToolPlan(z.tool_plan, T)
10:    if trace has unrecoverable tool error then
11:        return SafeRefusal(trace.error)
12:    verdict <- VerifyAndSimulate(z, trace)
13:    if verdict.pass then
14:        explanation <- ExplainDecision(z, trace, verdict)
15:        return VerifiedDecision(z, trace, verdict, explanation)
16:    feedback <- CompressCounterexample(verdict)
17: return HumanConfirmOrSafeRefusal(feedback)
```

### 7.7 复杂度与运行时间预期

设地图网格大小为 $G = X \times Y \times Z$，候选 UAV 数为 $|\mathcal{U}|$，工具调用轮数为 $K$。

| 模块 | 主要复杂度 | 优化方式 |
|------|------------|----------|
| IR generation | $O(K \cdot C_\text{LLM})$ | 缓存 prompt、短 feedback、低温度 |
| UAV assignment | $O(|\mathcal{U}|)$ greedy | 预过滤不可用 UAV |
| 3D A* | $O(G \log G)$ | corridor mask、分层网格、heuristic |
| STL monitoring | $O(T \cdot |\Phi|)$ | 向量化 trajectory check |
| Simulation | $O(T \cdot N_\text{agents})$ | 批量 seeds、早停 |
| Retrieval | $O(\log M)$ approximate | FAISS/Qdrant |

第一篇不需要追求极限实时性，但要报告端到端 latency。建议目标：

- dev-mini：单任务 5-20 秒；
- 本地 14B：单任务 10-40 秒；
- API upper bound：单任务 5-30 秒；
- 批量评测：异步并发，但每条样本记录独立 latency。

---

## 8. 数据来源与 CloudBrain-Bench 构建

### 8.1 数据构成

第一篇主数据集建议叫 **CloudBrain-Bench**。

| 数据层 | 来源 | 是否主实验依赖 | 作用 |
|--------|------|----------------|------|
| Synthetic city grid | 程序生成 | 是 | 可控、可复现、可扩展 |
| OSM city context | OSM / Overpass | 是 | POI、道路、建筑、功能区命名 |
| Overture Maps context | Overture Places / Buildings / Transportation | 可选增强 | 高质量 POI、建筑、道路拓扑和稳定实体 ID |
| Real airspace grids | FAA UAS Facility Map polygons + UAS Data Dictionary | 是 | 实际 UASFM geometry、ceiling、airspace/airport/LAANC 字段 |
| OD demand proxy | NYC TLC / Chicago taxi 可选 | 可选 | 生成需求热区和高峰任务 |
| Ground traffic | SUMO | 可选增强 | 地面 fallback travel time |
| Aviation weather | NOAA Aviation Weather Data API METAR + Open-Meteo | 可选增强 | 真实航空天气、风速、能见度、降水和天气风险 |
| Real UAV flight telemetry | DJI Matrice 100 package-delivery flight dataset | 可选校准 | 位置、电流、电压、风、速度、载荷、高度下的能耗/ETA 校准 |
| UTM flight-test context | NASA TCL4 reports | 可选增强 | 城市 canyon、BVLOS、weather front、emergency response 场景 taxonomy |
| UAV dynamics | 自建轻量 simulator | 是 | 路径、能耗、碰撞、延迟 |
| Visual simulator | AirSim / Flightmare | 可选补充 | 后续视觉/动力学扩展 |

OSM/Overpass 适合查询城市要素 [28]；Overture Maps 通过 GeoParquet 提供 places、buildings 和 transportation layers，可补足 POI、建筑和道路拓扑 [41]。空域层不应只写成抽象 proxy：FAA UAS Facility Maps 官方页面提供面向 data providers 的 UASFM 数据入口，数据字典明确了 geometry、center latitude/longitude、`CEILING`、airspace class、airport identifiers 和 LAANC readiness 等字段 [27] [43]。天气层可用 NOAA Aviation Weather Data API 拉取 METAR 等航空天气观测，再用 Open-Meteo 补历史/网格天气特征 [42] [44]。真实 UAV 动力学层可以使用 Scientific Data 发布的 DJI Matrice 100 小包裹配送飞行数据；该数据包含数百次飞行中位置、能量使用、风、载荷、高度和速度变化，可用于校准能耗与 ETA，而不是凭空指定 battery model [45]。NYC TLC 与 SUMO 仍只作为 demand 和 ground fallback proxy [29] [30]；AirSim 和 Flightmare 作为闭环仿真补充 [31] [32]。

### 8.1.1 真实数据可行性判断

二次检索后的结论不是“没有真实数据”，而是 **真实数据存在于不同层，缺少公开的完整低空商用运营闭环**。

| 数据问题 | 2026-05-21 可公开获得程度 | 可用于 G1 的方式 | 不能宣称的内容 |
|----------|---------------------------|------------------|----------------|
| 城市地图/POI/建筑/道路 | 高 | OSM/Overture 真实城市上下文 | 不等于真实 drone corridor |
| UAS 空域高度网格 | 高 | FAA UASFM polygon、ceiling、airspace/LAANC 字段 | UASFM 不等于飞行授权 |
| 航空气象 | 高 | NOAA METAR、Open-Meteo 风雨特征 | 机场气象不等于街区级低空风场 |
| UAV 真实飞行能耗/位置 | 中 | DJI M100 delivery telemetry 校准能耗与 ETA | 不等于百架真实运营调度 |
| UTM 试飞场景与人机信息流 | 中 | NASA TCL4 场景 taxonomy 和 UTM 信息需求 | 报告不等于公开 raw UTM fleet trajectories |
| 商业 delivery 订单流/轨迹日志 | 低 | 仅用 FAA 运营背景和未来合作动机 | 不能伪造 Zipline/Wing/Flytrex 订单轨迹 |
| Remote ID 大规模公开实时轨迹 | 低 | 不作为主数据源 | 不能把 Remote ID 当现成公开 fleet dataset |

FAA Part 135 页面说明美国已经有包裹配送无人机运营审批路线和已批准运营主体，因此研究问题并非纯假想 [46]。但公开运营订单流、空域冲突记录和 commercial flight logs 通常不随审批页面发布。Remote ID 也不应当作现成开源轨迹库：GAO 在 2024 年仍建议 FAA 识别提供实时、联网 drone location/status 数据的路径 [47]。因此 G1 的强表述应是：

> We build a real-context and real-flight-calibrated low-altitude agent benchmark, while leaving fully real operational fleet logs to future operator collaboration.

### 8.1.2 数据分层策略

CloudBrain-Bench 建议拆成三个可信度层级：

| 层级 | 名称 | 数据组成 | 在论文中的角色 |
|------|------|----------|----------------|
| L1 | `Synthetic-Controlled` | 程序城市、程序 airspace、程序任务 | 可控主对比、消融、统计稳定性 |
| L2 | `Real-Context` | OSM/Overture + FAA UASFM + NOAA/Open-Meteo + 程序任务 | 主实验优先层，证明真实上下文 grounding |
| L3 | `Real-Flight-Calibrated` | L2 + DJI M100 飞行能耗/ETA 参数校准 | 校准分析和真实飞行敏感性验证 |

不建议把 L3 写成“real operational benchmark”。更稳的拆法是：

- **任务与 gold trace**：仍由 deterministic generator、planner、verifier 产生，保证 SAT/UNSAT 真值。
- **城市/空域/天气上下文**：尽可能真实，验证 agent 是否 grounding 到真实实体和 real airspace field。
- **能耗/ETA model**：用真实飞行数据拟合或分桶校准，验证安全判断不会建立在随意能耗参数上。

### 8.1.3 真实数据获取 recipe

为了让第一篇论文可复现，建议把数据获取写成固定 pipeline：

| 步骤 | 输入 | 操作 | 输出 |
|------|------|------|------|
| 1 | 城市 bounding box | 用 Overpass 查询 hospital、school、park、police、fire_station、building、road | `city_osm.geojson` |
| 2 | 同一 bbox | 用 Overture Places/Buildings 补充 POI 与建筑 footprint，保留稳定 entity id | `city_overture.parquet` |
| 3 | FAA UASFM data download / bbox | 读取 UASFM polygon、`CEILING`、airport/airspace/LAANC 字段 | `uasfm_cells.geojson` |
| 4 | nearest ICAO stations + time window | 查询 NOAA METAR JSON，抽取 wind、visibility、precip/weather tokens | `aviation_weather.parquet` |
| 5 | 经纬度和时间段 | 查询 Open-Meteo 历史/预测天气作为非机场补充 | `weather_grid.parquet` |
| 6 | Scientific Data DJI M100 files | 解析 position、voltage、current、wind、payload、altitude、speed | `uav_flight_calibration.parquet` |
| 7 | 选定城市和日期 | 抽样 NYC TLC / Chicago taxi OD 形成 demand heatmap | `od_proxy.parquet` |
| 8 | OSM road graph | 导入 SUMO，估计 ground fallback travel time | `ground_time_matrix.parquet` |
| 9 | UTM/UASFM/CORUS/NASA TCL4 | 人工整理 rule templates 和 scenario taxonomy | `airspace_rules.yaml` |
| 10 | real context + calibrated UAV params | 程序生成 UAV tasks、NFZ、corridor、charging、weather risk | `cloudbrain_samples.jsonl` |
| 11 | samples | planner/verifier/simulator 自动标注 SAT/UNSAT、gold trace、counterexample | `cloudbrain_gold.jsonl` |

主实验最低依赖步骤 1、3、9、10、11；步骤 4-8 提供真实天气、能耗校准和 ground fallback。每次抓取需要保存原始文件快照、数据字段版本和下载日期，防止 FAA/NOAA/地图数据后续变更导致不可复现。

### 8.1.4 真实数据如何映射到 benchmark

| 真实字段 | 映射到 CloudBrain | 使用方式 |
|----------|-------------------|----------|
| OSM `amenity=hospital/school/fire_station` | `origin`, `destination`, `sensitive_zones` | 指令实体 grounding |
| Overture building footprint | obstacle polygons | route planner/simulator |
| UASFM `SHAPE`, `CEILING` | altitude cap cells | `query_airspace` 工具返回 |
| UASFM airport/airspace fields | airspace provenance | 解释与 policy 字段 |
| NOAA METAR wind/visibility/weather | weather risk | `risk_assess`、stress scenarios |
| M100 position/speed/altitude | route/ETA calibration bins | ETA distribution |
| M100 current/voltage/payload/wind | energy model calibration | battery reserve check |

### 8.1.5 真实飞行校准任务

用 DJI M100 数据只做能支撑的事情：

1. 按 payload、cruise speed、altitude、wind 分桶。
2. 从电压电流积分得到 flight energy 或能耗 proxy。
3. 拟合 `energy_per_meter`、`eta_multiplier` 或 conservative quantile lookup。
4. 将 synthetic planner 的 route length 映射成 energy estimate 和 battery reserve verdict。
5. 在附录报告 calibrated 与 uncalibrated energy model 下 safety decision 是否变化。

推荐第一版使用保守分位数，而不是复杂黑盒能耗网络：

$$
E_\text{route} = L_\text{route} \cdot q_{0.9}(e \mid v, h, p, w)
$$

其中 $e$ 表示单位距离能耗，$v$ 表示速度，$h$ 表示高度，$p$ 表示载荷，$w$ 表示风条件。这样能把真实飞行数据接进 safety check，又不会把 G1 变成能耗建模论文。

### 8.1.6 真实数据 split 设计

| Split | 数据层 | 作用 |
|-------|--------|------|
| `test_synthetic_controlled` | L1 | 主消融、可控难度 |
| `test_real_context_city_a` | L2 | 真实城市/空域/天气上下文 |
| `test_real_context_city_b` | L2 | unseen city generalization |
| `test_real_weather_stress` | L2 | METAR/Open-Meteo 风雨风险 |
| `test_energy_calibrated` | L3 | 真实 flight calibration 后的 battery/ETA safety |

论文主表可以同时给 L1 和 L2；L3 建议作为校准分析表或 appendix。若 L2 效果稳定，摘要可以写 “real-context benchmark”。若 L3 也稳定，再写 “real-flight-calibrated evaluation”。

### 8.1.7 真实数据获取代码草图

UASFM 与 METAR loader 不要混在 agent tools 里，先做离线数据准备：

```python
def load_uasfm_cells(path: Path, bbox: BoundingBox) -> gpd.GeoDataFrame:
    cells = gpd.read_file(path)
    cells = cells.to_crs("EPSG:4326")
    clipped = cells[cells.geometry.intersects(bbox.to_polygon())].copy()
    keep = ["CEILING", "UNIT", "GLOBAL_ID", "APT1_ICAO", "AIRSPACE_1", "geometry"]
    return clipped[keep]


def fetch_metar_snapshot(station_ids: list[str], hours: int) -> pd.DataFrame:
    response = requests.get(
        "https://aviationweather.gov/api/data/metar",
        params={"ids": ",".join(station_ids), "format": "json", "hours": hours},
        timeout=30,
    )
    response.raise_for_status()
    rows = response.json()
    return normalize_metar_rows(rows)
```

真实飞行校准 loader：

```python
def build_energy_calibration(flights: Iterable[FlightLog]) -> EnergyCalibrationTable:
    rows = []
    for flight in flights:
        energy_j = integrate_power(flight.voltage_v, flight.current_a, flight.time_sec)
        path_length_m = trajectory_length(flight.position_xyz)
        rows.append(
            {
                "payload_g": flight.payload_g,
                "cruise_speed_mps": flight.programmed_speed_mps,
                "altitude_m": flight.programmed_altitude_m,
                "wind_bin": wind_bin(flight.wind_speed_mps),
                "energy_per_meter_j": energy_j / max(path_length_m, 1.0),
            }
        )
    return EnergyCalibrationTable.from_rows(rows, quantile=0.9)
```

### 8.1.8 城市与场景参数

建议第一版固定 4 类城市布局：

| 城市类型 | 网格大小 | POI 特征 | 低空风险 | 用途 |
|----------|----------|----------|----------|------|
| `grid_city` | 50 x 50 x 6 | 规则路网、均匀 POI | 低 | sanity check |
| `downtown_city` | 80 x 80 x 8 | 高建筑密度、医院/学校密集 | 高 | 主实验 |
| `suburban_city` | 100 x 100 x 5 | POI 稀疏、距离长 | 中 | battery/deadline |
| `mixed_city` | 120 x 120 x 10 | 商业区、居民区、交通枢纽混合 | 高 | unseen generalization |

空间尺度：

| 参数 | 默认 | 范围 |
|------|------|------|
| cell size | 10 m | 5-20 m |
| altitude layers | 6 | 3-12 |
| max altitude | 120 m | 60-150 m |
| corridor width | 20 m | 10-40 m |
| no-fly zones | 3-12 per map | 0-20 |
| sensitive zones | 5-30 per map | 0-50 |
| charging pads | 3-10 per map | 1-20 |
| UAV count | 10 / 30 / 50 | 5-100 |

任务参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| deadline tightness | medium | loose / medium / tight / impossible |
| priority distribution | 60/25/10/5 | normal/high/critical/low 可调 |
| battery distribution | beta-like | 制造低电量边界情况 |
| weather risk | none/low/medium/high | stress split 中提高 |
| demand burst | 1x / 2x / 4x | 测 corridor 和 scheduler |

### 8.1.9 规则模板

第一版只做 8 类规则，够写论文且能复现：

| 规则 ID | 自然语言 | LTL/STL/程序检查 |
|---------|----------|------------------|
| R1 | 不进入临时禁飞区 | `G not_in_nfz` |
| R2 | 始终保持最低安全距离 | STL robustness: `dist_to_obstacle > d_min` |
| R3 | 高度保持在允许区间 | `G altitude_min <= z <= altitude_max` |
| R4 | deadline 前到达 | `F[0, deadline] at_goal` |
| R5 | 返航/到达后保留电量 | program check |
| R6 | corridor 容量不超限 | capacity monitor |
| R7 | critical 任务优先，但不能覆盖 safety | policy check |
| R8 | 信息不足或 UNSAT 时触发 safe refusal/human confirm | refusal check |

### 8.1.10 数据质量控制

CloudBrain-Bench 必须避免“LLM 生成垃圾标签”。建议每条样本记录四类质量字段：

| 字段 | 说明 |
|------|------|
| `generation_seed` | 复现实验的随机种子 |
| `source_provenance` | OSM/Overture/规则模板/程序生成来源 |
| `label_verifier` | SAT/UNSAT 标签来自哪个 checker |
| `human_review_status` | unchecked / sampled_pass / sampled_fail / corrected |

抽样检查策略：

- 每个 scenario type 至少抽查 30 条；
- 每个 failure mode 至少抽查 20 条；
- stress 和 UNSAT 样本抽查比例提高到 15%-20%；
- 人工只改自然语言和解释，不手改 planner/verifier 标签，避免引入主观标签。

### 8.2 样本格式

每条样本包括：

```json
{
  "sample_id": "cb_000001",
  "data_tier": "real_context",
  "city_seed": 12,
  "scenario_type": "emergency_delivery_with_nfz",
  "instruction": "请优先派一架无人机把急救包送到 accident_site_3，避开学校和临时禁飞区，10 分钟内到达。",
  "source_provenance": {
    "map_sources": ["osm", "overture"],
    "airspace_sources": ["faa_uasfm"],
    "weather_sources": ["noaa_metar", "open_meteo"],
    "task_source": "deterministic_generator"
  },
  "real_context": {
    "city_id": "pittsburgh_bbox_01",
    "uasfm_snapshot": "faa_uasfm_2026_05",
    "weather_snapshot": "metar_kpit_2026_05_20T12Z"
  },
  "energy_calibration_version": "dji_m100_q90_v0",
  "state": {
    "uavs": "...",
    "airspace": "...",
    "map": "...",
    "tasks": "..."
  },
  "gold_ir": "...",
  "gold_tool_trace": "...",
  "gold_decision": "...",
  "logic_specs": ["G not_in_nfz", "F[0,600] arrive_destination"],
  "label": "SAT",
  "failure_modes": []
}
```

### 8.3 场景类型

| 场景 | 比例 | 难点 |
|------|------|------|
| Normal delivery | 15% | 普通调度和路径规划 |
| Emergency delivery | 15% | priority、deadline、risk tradeoff |
| Patrol / inspection | 10% | 多 waypoint temporal constraints |
| No-fly zone avoidance | 15% | LTL/STL 安全约束 |
| Corridor congestion | 10% | 空域容量和延迟 |
| Charging bottleneck | 10% | 电量约束和 fallback |
| Weather / wind risk | 10% | 风险评估和拒绝 |
| Multimodal fallback | 10% | UAV-ground transfer |
| UNSAT / ambiguous tasks | 5% | 安全拒绝和澄清 |

### 8.4 数据规模

第一版可行规模：

| Split | 样本数 | 用途 |
|-------|--------|------|
| Dev-mini | 200 | 快速调试 pipeline |
| Train-like | 3000 | few-shot、RAG、repair memory，不用于主训练 |
| Validation | 1000 | prompt/model selection |
| Test-seen-city | 1000 | 主测试 |
| Test-unseen-city | 1000 | 泛化 |
| Test-stress | 1000 | 危险场景压力测试 |

合计约 7200 条样本，足够支撑第一篇 benchmark/method 论文。后续 G2 微调再扩到 50k-100k tool traces。

### 8.5 Gold label 生成

gold 不应全部由 LLM 生成。推荐流程：

1. 程序生成城市、任务、UAV 状态和规则。
2. 规则模板生成 gold `LowAltitudeIR`。
3. 调用 deterministic tools 得到 gold tool trace。
4. planner/verifier/simulator 决定 SAT/UNSAT。
5. LLM teacher 只负责自然语言 paraphrase 和少量解释文本。
6. 抽样 5%-10% 人工检查，重点查高风险和 UNSAT 样本。

### 8.6 数据文件组织

建议最终开源或内部复现实验用以下结构：

```text
data/cloudbrain_bench/
  README.md
  schemas/
    low_altitude_ir.schema.json
    tool_result.schema.json
    sample.schema.json
  raw/
    osm/
    overture/
    uasfm/
    aviation_weather/
    weather/
    uav_flight_calibration/
    od_proxy/
  processed/
    city_states/
    airspace_rules/
    uasfm_cells/
    weather_risk_tables/
    energy_calibration_tables/
    uav_states/
  splits/
    dev_mini.jsonl
    train_like.jsonl
    validation.jsonl
    test_seen_city.jsonl
    test_unseen_city.jsonl
    test_stress.jsonl
    test_real_context_city_a.jsonl
    test_real_context_city_b.jsonl
    test_real_weather_stress.jsonl
    test_energy_calibrated.jsonl
    test_unsat.jsonl
  gold/
    gold_ir.jsonl
    gold_tool_traces.jsonl
    gold_verdicts.jsonl
  metadata/
    split_stats.csv
    scenario_taxonomy.yaml
    data_sources.yaml
```

### 8.7 数据统计必须报告

论文 Table 2 至少报告：

| 统计项 | 必须报告 |
|--------|----------|
| 样本总数 | total / per split |
| 场景类型分布 | 9 类 scenario type |
| SAT/UNSAT 比例 | overall + per scenario |
| 城市数量 | seen / unseen |
| 数据层级 | L1/L2/L3 样本数与比例 |
| 真实上下文字段覆盖 | OSM/Overture/UASFM/NOAA/Open-Meteo snapshot coverage |
| UAV 数量分布 | min / median / max |
| 约束数量 | 每条任务平均约束数 |
| 工具调用长度 | gold trace 平均长度 |
| 失败类型分布 | no_path / nfz / battery / deadline / ambiguity |
| 人工抽检比例 | pass / corrected |

---

## 9. 模型选择与部署方案

### 9.1 第一篇不建议先训垂类大模型

G1 的主线是 agent 方法与验证闭环。垂类模型训练放在后续 G2。G1 可以包含轻量 SFT 预实验，但不应让它成为论文成败关键。

### 9.2 推荐模型矩阵

| 角色 | 模型 | 用法 | 是否必须 |
|------|------|------|----------|
| Teacher / upper bound | GPT-5.2 或同级 API | 生成数据、做强 baseline、错误分析 | 是 |
| Local main model | Qwen3-14B / Qwen3-32B | 主实验可复现 agent | 是 |
| Local reasoning model | DeepSeek-R1-Distill-Qwen-14B/32B | repair、counterexample reasoning | 是 |
| Small latency model | Qwen3-8B | 低延迟 ablation | 可选 |
| Embedding | Qwen3-Embedding / BGE-M3 | RAG 和 safety memory retrieval | 是 |

GPT-5.2 官方定位适合 coding 和 agentic tasks，可作为强 teacher 和闭源上限 [10]。Qwen3 技术报告强调 reasoning、instruction following、agent 和 multilingual 能力，适合作为本地开源主模型 [11]。DeepSeek-R1 提供蒸馏到 Qwen/Llama 的 14B/32B 等 reasoning 模型，适合做反例修复 [12]。

### 9.3 本地还是 API

推荐 **混合架构**：

| 阶段 | API | 本地 |
|------|-----|------|
| 第 1-2 周 | 快速验证 prompt、schema、工具设计 | 同步部署 Qwen3-14B |
| 第 3-5 周 | teacher 生成 paraphrase 和困难样本 | 主跑 dev/validation |
| 第 6-8 周 | 做 upper-bound baseline | 主实验和可复现结果 |
| 投稿前 | 少量错误分析 | 所有核心实验本地可复现 |

论文主表建议以本地模型为主，API 模型作为 upper bound。这样既有强效果，又能避免审稿人质疑不可复现。

### 9.4 快速处理实现

部署建议：

```text
vLLM server
  -> OpenAI-compatible endpoint
  -> Agent runtime
  -> Tool registry / MCP servers
  -> verifier / simulator
```

vLLM 提供 OpenAI-compatible server，可让本地 Qwen/DeepSeek 与 API 模型共用一套调用接口 [33]。

### 9.5 Prompt 与推理配置

为保证可复现，所有模型都要固定推理参数：

| 用途 | temperature | top_p | max tokens | repair K | 说明 |
|------|-------------|-------|------------|----------|------|
| Direct LLM | 0.2 | 0.9 | 2048 | 0 | 直接输出决策 |
| JSON-only | 0.0 | 1.0 | 2048 | 0 | 结构化输出，减少随机性 |
| ReAct | 0.2 | 0.9 | 4096 | 0 | 允许 reasoning/action |
| CloudBrain no repair | 0.0 | 1.0 | 4096 | 0 | 单次 IR + tools |
| CloudBrain full | 0.0 first, 0.2 repair | 1.0 | 4096 | 3 | 修复轮可稍微放开 |

建议 prompt 拆成四段：

1. **System role**：你是低空交通云脑 agent，不直接输出控制量。
2. **IR schema**：给出 `LowAltitudeIR` JSON schema 和 enum。
3. **Tool registry**：列出可用工具、输入输出和失败类型。
4. **Current task/state**：当前自然语言任务、UAV 状态、地图、空域规则、历史反馈。

输出格式必须固定：

```json
{
  "low_altitude_ir": {},
  "rationale_summary": "one paragraph only",
  "uncertainty": {
    "needs_human_confirmation": false,
    "missing_information": []
  }
}
```

不要让模型输出完整 chain-of-thought；论文和系统只保存简短 rationale summary、工具轨迹和 verifier feedback。

### 9.6 API 与本地成本记录

每次实验保存：

| 字段 | 说明 |
|------|------|
| `model_name` | API 或本地模型名 |
| `endpoint_type` | api / local_vllm |
| `prompt_tokens` | 输入 token |
| `completion_tokens` | 输出 token |
| `wall_time_sec` | 端到端时间 |
| `llm_time_sec` | LLM 调用时间 |
| `tool_time_sec` | 工具执行时间 |
| `repair_rounds` | 修复轮数 |
| `estimated_cost_usd` | API 估算成本，本地可填 0 或 GPU-hour |

这能支撑 Table 5 的部署分析。

---

## 10. Baselines

### 10.1 主 baseline

| Baseline | 描述 | 要回答的问题 |
|----------|------|--------------|
| Direct LLM | 模型直接输出决策文本 | LLM 裸跑有多不可靠 |
| JSON-only LLM | 只要求输出 JSON IR，无工具执行 | typed output 是否足够 |
| ReAct prompting | ReAct 风格工具调用，无 schema/verifier | reasoning-action loop 是否足够 |
| Tool-use only | 有工具调用，无 verification repair | 工具是否足够 |
| BFCL-style function calling | 只评估函数名和参数正确，不执行物理验证 | function calling 成功是否等于云脑成功 |
| Tau-bench-style policy agent | 有工具和政策规则，但无 UAV planner/verifier | domain policy following 是否足够 |
| ToolSandbox-style stateful tool agent | 有状态工具执行和信息不足处理 | stateful tool execution 对低空任务的贡献 |
| LLM+P style | LLM 转换为 planning problem，planner 求解 | 外部 planner 能解决多少 |
| TrafficGPT-style | LLM 调用交通工具，无 UAV formal safety | 交通 LLM orchestration baseline |
| CloudBrain-Agent w/o simulator | 去掉仿真压力测试 | simulator feedback 贡献 |
| CloudBrain-Agent w/o repair | 失败即停止 | repair loop 贡献 |
| CloudBrain-Agent full | 完整方法 | 本文主方法 |

### 10.2 模型 baseline

| Model | 设置 |
|-------|------|
| GPT-5.2 | API upper bound |
| Qwen3-14B | local main |
| Qwen3-32B | local stronger |
| DeepSeek-R1-Distill-Qwen-14B | local repair reasoning |
| Qwen3-8B | small local |

### 10.3 Baseline 实现细则

为避免 baseline 被审稿人认为不公平，每个 baseline 都要明确输入权限：

| Baseline | 可见自然语言 | 可见状态 | 可调用工具 | 可见 verifier feedback | 可修复 |
|----------|--------------|----------|------------|------------------------|--------|
| Direct LLM | 是 | 摘要状态 | 否 | 否 | 否 |
| JSON-only | 是 | 完整状态 | 否 | 否 | 否 |
| ReAct | 是 | 完整状态 | 是 | 工具错误，不含 counterexample | 否 |
| Tool-use only | 是 | 完整状态 | 是 | 工具错误 | 否 |
| LLM+P style | 是 | 完整状态 | planner | planner result | 否 |
| CloudBrain w/o verifier | 是 | 完整状态 | 是 | 否 | 否 |
| CloudBrain w/o simulator | 是 | 完整状态 | 是 | verifier only | 是 |
| CloudBrain full | 是 | 完整状态 | 是 | verifier + simulator | 是 |

公平性原则：

- 所有方法使用同一底座模型；
- 所有方法使用同一测试 split；
- 所有方法有相同最大 token budget；
- ReAct 和 CloudBrain 的最大工具调用次数相同；
- 只有 CloudBrain full 使用结构化 counterexample，因为这是本文贡献。

---

## 11. 实验设计

### 11.1 Experiment 1：主结果

问题：CloudBrain-Agent full 是否优于 direct LLM、ReAct、tool-use only 和 LLM+P？

数据：Test-seen-city、Test-unseen-city、Test-stress。

指标：

- Task success rate
- Executable decision rate
- Safety violation rate
- Tool-call accuracy
- Hallucination rate
- Repair success rate
- Latency

### 11.2 Experiment 2：消融实验

| Ablation | 删除内容 | 预期影响 |
|----------|----------|----------|
| no typed IR | 自由文本工具调用 | tool-call accuracy 下降 |
| no verifier | 不做 LTL/STL 检查 | safety violation 上升 |
| no simulator | 不做场景压力测试 | stress pass 下降 |
| no repair | 验证失败后不迭代 | executable rate 下降 |
| no memory | 不检索历史失败案例 | repair success 下降 |
| no RAG | 不检索规则/地图上下文 | hallucination 上升 |

### 11.3 Experiment 3：反例修复分析

统计 verifier/simulator 失败后的修复路径：

- 第 1 次修复成功率
- 第 2 次修复成功率
- 第 3 次修复成功率
- 修复后新增违规率
- 最常见失败类型：NFZ、deadline、battery、corridor、entity hallucination

### 11.4 Experiment 4：模型与部署分析

比较 API 与本地模型：

| 模型 | 指标 |
|------|------|
| GPT-5.2 | 上限效果、成本、延迟 |
| Qwen3-14B | 本地可复现主结果 |
| Qwen3-32B | 本地强模型 |
| DeepSeek-R1-Distill-Qwen-14B | repair 专项能力 |
| Qwen3-8B | 低延迟 tradeoff |

### 11.5 Experiment 5：泛化

泛化维度：

- unseen city layout
- unseen POI names
- unseen no-fly zone shape
- unseen tool combination
- unseen emergency scenario
- higher UAV density
- higher demand shock

### 11.6 Experiment 6：安全拒绝与人机协同

测试模型是否能在 UNSAT 或信息不足时拒绝执行或请求人工确认。

样例：

- deadline impossible
- all UAV battery insufficient
- destination inside NFZ
- missing destination
- conflict between priority and safety rule

### 11.7 Experiment 7：Agent 可靠性与多轮一致性

参考 $\tau$-bench 的 `pass^k` 思路，对同一任务重复运行 $k$ 次，评估 agent 是否稳定完成任务 [36]。低空交通任务里，一次成功但多次随机失败不够安全，因此建议报告：

| 指标 | 含义 |
|------|------|
| `pass@1` | 单次运行成功率 |
| `pass^3` | 同一任务连续 3 次都成功的比例 |
| `pass^5` | 同一任务连续 5 次都成功的比例 |
| policy compliance | 是否遵守空域/安全/人工确认规则 |
| state consistency | 多轮工具调用后内部状态是否与工具返回一致 |
| insufficient-information handling | 信息不足时是否澄清/拒绝，而不是幻觉补全 |

这部分会让 G1 不只是“交通应用”，而是对通用 agent 可靠性有可迁移贡献。

### 11.8 Experiment 8：任务难度分层

为了避免主结果被简单样本掩盖，按难度分层报告：

| 难度 | 定义 | 样本特征 |
|------|------|----------|
| Easy | 单任务、无 NFZ、deadline 宽松 | 普通 delivery |
| Medium | 1-2 个安全约束、正常电量 | NFZ 或 battery 单因素 |
| Hard | 多约束、deadline 紧、corridor 拥堵 | emergency + NFZ + charging |
| Extreme | 高风险或接近 UNSAT | stress split |
| UNSAT | 无可行安全方案 | safe refusal / human confirm |

主表报告 overall，附表报告 per difficulty。预期 CloudBrain 的优势在 Hard/Extreme/UNSAT 上最大。

### 11.9 Experiment 9：错误归因

每个失败样本要自动归因到首个失败阶段：

| 阶段 | 错误类型 |
|------|----------|
| IR | invalid JSON、schema missing、wrong enum |
| Grounding | nonexistent entity、wrong zone、wrong UAV |
| Tool | wrong tool、wrong order、invalid arguments |
| Planning | no path、wrong UAV、battery infeasible |
| Verification | NFZ、altitude、distance、deadline、capacity |
| Simulation | collision、near miss、weather risk、delay |
| Policy | unsafe override、missing human confirm、wrong refusal |
| Explanation | hallucinated reason、unsupported claim |

错误分析图建议用 stacked bar：不同 baseline 的失败阶段分布。这样能清楚说明 CloudBrain 到底修了什么。

---

## 12. 评价指标定义

### 12.1 结构化输出指标

**IR exact match**：

$$
\text{IR-EM} = \frac{1}{N}\sum_i \mathbb{1}[z_i = z_i^\*]
$$

**IR field F1**：对 intent、entities、constraints、tool plan 等字段分别计算 precision、recall、F1。

### 12.2 工具调用指标

**Tool-call accuracy**：

$$
\text{TCA} = \frac{\#\text{correct tool calls}}{\#\text{all tool calls}}
$$

正确要求：

- 工具名正确；
- 参数 schema 正确；
- 参数引用的实体存在；
- 调用顺序满足依赖。

**Tool dependency success**：

$$
\text{TDS} = \frac{\#\text{tool chains satisfying all data dependencies}}{\#\text{tool chains}}
$$

它衡量 agent 是否先查询空域/城市状态，再规划和验证，而不是凭空调用下游工具。

### 12.3 可执行性指标

**Executable decision rate**：

$$
\text{EDR} = \frac{\#\text{planner executable decisions}}{N}
$$

**Task success rate**：

$$
\text{TSR} = \frac{\#\text{fully verified and simulated successful tasks}}{N}
$$

### 12.4 安全指标

**Safety violation rate**：

$$
\text{SVR} = \frac{\#\text{safety violated tasks}}{N}
$$

违规类型包括：

- no-fly zone intrusion；
- altitude violation；
- min separation violation；
- battery reserve violation；
- deadline violation；
- unsafe fallback；
- hallucinated permission。

### 12.5 幻觉指标

**Hallucination rate**：

$$
\text{HR} = \frac{\#\text{outputs containing nonexistent entity/tool/rule}}{N}
$$

### 12.6 修复指标

**Repair success rate**：

$$
\text{RSR} = \frac{\#\text{failed first attempts repaired within K iterations}}{\#\text{failed first attempts}}
$$

建议 $K=3$，并报告每轮 marginal gain。

**Consistency success**：

$$
\text{pass}^k = \frac{\#\text{tasks successful in all } k \text{ repeated runs}}{N}
$$

该指标比 `pass@1` 更适合安全关键 agent，因为低空交通云脑需要稳定遵守规则，而不是偶尔成功 [36]。

### 12.7 统计检验

每个实验至少 3 个随机种子。主结果报告：

- mean ± standard error；
- paired bootstrap 95% confidence interval；
- McNemar test 或 bootstrap test 比较 success/failure 类指标；
- 对 latency 报告 median、p90、p95。

### 12.8 主结果表模板

论文 Table 3 可以直接按这个格式填：

| Method | Model | TSR ↑ | EDR ↑ | SVR ↓ | HR ↓ | TCA ↑ | RSR ↑ | pass^3 ↑ | p95 Latency ↓ |
|--------|-------|-------|-------|-------|------|-------|-------|----------|---------------|
| Direct LLM | Qwen3-14B | - | - | - | - | N/A | N/A | - | - |
| JSON-only | Qwen3-14B | - | - | - | - | N/A | N/A | - | - |
| ReAct | Qwen3-14B | - | - | - | - | - | N/A | - | - |
| Tool-use only | Qwen3-14B | - | - | - | - | - | N/A | - | - |
| LLM+P | Qwen3-14B | - | - | - | - | - | N/A | - | - |
| CloudBrain w/o repair | Qwen3-14B | - | - | - | - | - | N/A | - | - |
| CloudBrain full | Qwen3-14B | - | - | - | - | - | - | - | - |

### 12.9 消融表模板

| Variant | TSR ↑ | EDR ↑ | SVR ↓ | TCA ↑ | RSR ↑ | 主要解释 |
|---------|-------|-------|-------|-------|-------|----------|
| Full | - | - | - | - | - | 完整方法 |
| no typed IR | - | - | - | - | - | 测结构化接口 |
| no verifier | - | - | - | - | - | 测形式验证 |
| no simulator | - | - | - | - | - | 测压力反馈 |
| no repair | - | - | - | - | N/A | 测反例修复 |
| no memory | - | - | - | - | - | 测失败案例检索 |
| no RAG | - | - | - | - | - | 测规则/地图上下文 |

### 12.10 最小成功门槛

进入论文写作前，建议至少达到这些门槛：

| 指标 | 最小门槛 | 理由 |
|------|----------|------|
| CloudBrain full TSR | 比 ReAct 高 10 个百分点以上 | 方法主收益 |
| SVR | 比 Direct LLM 低 30% 以上 | 安全关键价值 |
| TCA | 超过 85% | 工具调用可靠 |
| RSR | 超过 40% | 反例修复有实效 |
| pass^3 | 明显高于 tool-use only | 多轮稳定性 |
| p95 latency | 本地 14B 小于 60 秒 | 可部署叙事 |

---

## 13. 预期实验结论

以下是预注册预期，不是实验结果：

1. 预期 CloudBrain-Agent full 在 task success、executable decision rate 和 safety violation rate 上优于 direct LLM、ReAct 和 tool-use only。
2. 预期 typed `LowAltitudeIR` 主要提升 tool-call accuracy、IR field F1 和 hallucination rate。
3. 预期 verifier feedback 主要提升 executable decision rate 和 repair success rate。
4. 预期 simulator feedback 在 stress scenarios 上最关键，尤其是 corridor congestion、wind risk、NFZ edge cases。
5. 预期本地 Qwen3-14B/32B 可作为可复现主模型，但 GPT-5.2 仍是 upper-bound。
6. 预期 DeepSeek-R1-Distill-Qwen 在 counterexample repair 上优于普通 instruct 模型。

---

## 14. 图表计划

| ID | 类型 | 内容 | 优先级 |
|----|------|------|--------|
| Fig. 1 | 架构图 | CloudBrain-Agent 从 instruction 到 verified decision 的闭环 | 高 |
| Fig. 2 | 数据生成流程图 | OSM/FAA/OD/SUMO/simulator 到 CloudBrain-Bench | 高 |
| Fig. 3 | 主结果柱状图 | TSR、EDR、SVR、HR 比较 | 高 |
| Fig. 4 | 修复曲线 | repair iteration 1-3 的成功率提升 | 高 |
| Fig. 5 | 泛化热力图 | seen/unseen city、stress、UNSAT 上表现 | 中 |
| Fig. 6 | Agent consistency 曲线 | `pass@1`、`pass^3`、`pass^5` 和 state consistency | 中 |
| Table 1 | 相关工作对比 | LLM traffic、tool-use、planning、formal verification、本文 | 高 |
| Table 2 | 数据集统计 | 场景类型、SAT/UNSAT、城市、任务数 | 高 |
| Table 3 | Baseline 主结果 | 全指标比较 | 高 |
| Table 4 | Ablation | 删除组件后的性能变化 | 高 |
| Table 5 | 模型部署 | API vs local 的效果、延迟、成本 | 中 |
| Table 6 | 数据来源可复现性 | 每类数据的 URL、许可证、是否主实验依赖、fallback | 中 |

---

## 15. 论文结构规划

按 AAAI/IJCAI 7-8 页主文压缩：

### Abstract

150-200 words。突出低空交通安全关键、LLM 不可靠、CloudBrain-Agent、benchmark、核心结果。

### 1 Introduction

内容：

- 低空交通云脑背景；
- LLM 直接决策的风险；
- 工具调用和验证闭环的必要性；
- 本文三条贡献；
- Fig. 1 hero figure。

### 2 Related Work

三段：

1. LLM for transportation and spatio-temporal intelligence；
2. LLM agents, tool use, and planning；
3. Formal verification and UAV/UTM simulation。

### 3 Problem Setup

定义状态、任务、`LowAltitudeIR`、工具、成功条件和安全约束。

### 4 Method

介绍 CloudBrain-Agent：

- context builder；
- LowAltitudeIR parser；
- tool router；
- verifier/simulator；
- repair loop；
- safety memory。

### 5 CloudBrain-Bench

介绍数据来源、生成流程、场景类型、splits、gold label 和可复现性。

### 6 Experiments

主结果、消融、修复分析、泛化、模型部署。

### 7 Conclusion

总结贡献，诚实写限制：synthetic benchmark、真实空域部署尚未验证、human-in-the-loop 仍需要。

---

## 16. 实现路线

### 16.1 最小可行系统

第一个月只做这些：

```text
cloudbrain/
  ir/schema.py
  tools/city.py
  tools/airspace.py
  tools/scheduler.py
  tools/planner.py
  tools/verifier.py
  tools/simulator.py
  agent/runner.py
  data/generator.py
  eval/metrics.py
```

### 16.2 推荐技术栈

| 模块 | 技术 |
|------|------|
| Agent runtime | Python + Pydantic + LiteLLM/OpenAI client |
| Local model | vLLM OpenAI-compatible server |
| Tool protocol | Python functions first, MCP wrapper second |
| IR validation | Pydantic JSON Schema |
| Planner | 3D A* first, RRT* optional |
| Verifier | Spot for LTL, RTAMT for STL |
| Simulator | lightweight grid/corridor simulator |
| RAG | Qdrant/FAISS + Qwen3-Embedding/BGE-M3 |
| Storage | JSONL + Parquet + DuckDB |
| Evaluation | pandas + scipy + bootstrap |

### 16.2.1 代码模块接口

建议每个模块暴露最小接口：

```python
class LowAltitudeIR(BaseModel):
    task_id: str
    intent: Literal["delivery", "inspection", "patrol", "emergency", "return", "charge", "monitoring"]
    priority: Literal["low", "normal", "high", "critical"]
    entities: dict
    constraints: dict
    tool_plan: list[dict]
    verification_specs: dict
    fallback_policy: str

class ToolResult(BaseModel):
    ok: bool
    tool: str
    request_id: str
    result: dict | None
    warnings: list[str]
    error: dict | None
    provenance: dict
```

Runner 主函数：

```python
def run_agent(sample: dict, model: str, config: AgentConfig) -> AgentTrace:
    ...
```

评测主函数：

```python
def evaluate_trace(sample: dict, trace: AgentTrace) -> dict:
    return {
        "task_success": ...,
        "executable_decision": ...,
        "safety_violation": ...,
        "tool_call_accuracy": ...,
        "hallucination": ...,
        "repair_success": ...,
        "latency_sec": ...,
    }
```

### 16.2.2 实验命令设计

建议未来实现后能用这些命令复现：

```bash
python -m cloudbrain.data.generate --config configs/data/dev_mini.yaml
python -m cloudbrain.eval.run --split dev_mini --method direct_llm --model qwen3-14b
python -m cloudbrain.eval.run --split dev_mini --method react --model qwen3-14b
python -m cloudbrain.eval.run --split dev_mini --method cloudbrain_full --model qwen3-14b
python -m cloudbrain.eval.aggregate --runs runs/dev_mini --out results/dev_mini.csv
python -m cloudbrain.figures.make_all --results results/main.csv --out figures/
```

### 16.2.3 配置文件模板

```yaml
experiment:
  name: cloudbrain_main_qwen3_14b
  seed: 42
  split: test_seen_city
  max_repair_rounds: 3

model:
  provider: local_vllm
  name: qwen3-14b
  temperature: 0.0
  top_p: 1.0
  max_tokens: 4096

tools:
  enable_city: true
  enable_airspace: true
  enable_scheduler: true
  enable_planner: true
  enable_verifier: true
  enable_simulator: true
  enable_risk: true

evaluation:
  bootstrap_samples: 1000
  report_pass_k: [1, 3, 5]
  latency_percentiles: [50, 90, 95]
```

### 16.3 10 周执行计划

| 周 | 目标 | 交付物 |
|----|------|--------|
| 1 | 冻结 problem formulation 和 IR schema | `LowAltitudeIR v0.1` |
| 2 | 实现 city/airspace/UAV/task generator | 200 dev samples |
| 3 | 实现 planner/verifier/simulator | deterministic gold labels |
| 4 | 实现 agent runner 和 direct/ReAct baselines | dev-mini results |
| 5 | 扩展 CloudBrain-Bench 到 3000+ | validation split |
| 6 | 跑本地 Qwen3-14B 和 GPT-5.2 upper bound | main baseline table draft |
| 7 | 实现 repair loop、memory、ablation | ablation results |
| 8 | 跑 unseen/stress/UNSAT | generalization figures |
| 9 | 统计检验、错误分析、图表 | camera-ready figures draft |
| 10 | 写 AAAI/IJCAI 初稿 | full paper draft |

### 16.4 每周验收标准

| 周 | 必须能跑的命令 | 验收标准 |
|----|----------------|----------|
| 1 | schema validation script | 20 条手写 IR 全部校验正确 |
| 2 | data generator | 200 条样本生成，split stats 无空字段 |
| 3 | tool unit tests | planner/verifier/simulator 至少 30 个单测 |
| 4 | dev-mini baseline | direct/ReAct/CloudBrain no repair 跑通 |
| 5 | validation split | 3000+ 样本，gold label 生成完成 |
| 6 | model matrix | Qwen3-14B 和 GPT upper bound 都有结果 |
| 7 | ablation | no IR/no verifier/no repair/no simulator 可运行 |
| 8 | stress/UNSAT | stress 与 safe refusal 指标可计算 |
| 9 | figures | 6 张图和 6 张表自动生成草稿 |
| 10 | paper draft | 主文完整，附录有 schema 和数据说明 |

### 16.5 推荐代码目录 v1

第一版代码库建议保持小而清晰，先服务论文实验，不要一开始做成大平台。

```text
cloudbrain-agent/
  pyproject.toml
  README.md
  configs/
    data/
      dev_mini.yaml
      main_bench.yaml
    experiments/
      direct_llm.yaml
      react.yaml
      cloudbrain_full.yaml
      ablation_no_verifier.yaml
    models/
      local_qwen3_14b.yaml
      api_gpt52.yaml
  data/
    cloudbrain_bench/
      schemas/
      splits/
      gold/
      metadata/
  src/
    cloudbrain/
      __init__.py
      ir/
        schema.py
        validators.py
        errors.py
      state/
        city_state.py
        airspace_state.py
        uav_state.py
        task_state.py
      tools/
        base.py
        registry.py
        city.py
        airspace.py
        scheduler.py
        planner.py
        verifier.py
        simulator.py
        risk.py
      agent/
        prompts.py
        llm_client.py
        runner.py
        repair.py
        memory.py
        traces.py
      data/
        generator.py
        osm_loader.py
        overture_loader.py
        weather_loader.py
        split.py
        quality.py
      eval/
        run.py
        metrics.py
        aggregate.py
        bootstrap.py
        error_analysis.py
      figures/
        main_results.py
        ablations.py
        repair_curve.py
      utils/
        io.py
        geometry.py
        hashing.py
        timing.py
  tests/
    test_ir_schema.py
    test_tool_registry.py
    test_planner.py
    test_verifier.py
    test_metrics.py
```

### 16.6 Pydantic schema 代码细节

`LowAltitudeIR` 要用强类型约束，尽量把错误挡在 LLM 输出之后、工具执行之前。

```python
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class Intent(str, Enum):
    delivery = "delivery"
    inspection = "inspection"
    patrol = "patrol"
    emergency = "emergency"
    return_home = "return"
    charge = "charge"
    monitoring = "monitoring"


class Priority(str, Enum):
    low = "low"
    normal = "normal"
    high = "high"
    critical = "critical"


class EntityRefs(BaseModel):
    origin: str | None = None
    destination: str | None = None
    candidate_uavs: list[str] = Field(default_factory=list)
    avoid_zones: list[str] = Field(default_factory=list)
    sensitive_zones: list[str] = Field(default_factory=list)
    handoff_points: list[str] = Field(default_factory=list)


class OperationConstraints(BaseModel):
    deadline_sec: int | None = Field(default=None, ge=1)
    altitude_min_m: float = Field(default=30.0, ge=0)
    altitude_max_m: float = Field(default=120.0, ge=0)
    min_separation_m: float = Field(default=10.0, ge=0)
    battery_reserve_ratio: float = Field(default=0.2, ge=0, le=1)
    max_risk_level: Literal["low", "medium", "high"] = "medium"
    corridor_capacity_required: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def check_altitude_range(self) -> "OperationConstraints":
        if self.altitude_min_m >= self.altitude_max_m:
            raise ValueError("altitude_min_m must be lower than altitude_max_m")
        return self


class ToolCallSpec(BaseModel):
    tool: str
    args: dict = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)


class VerificationSpecs(BaseModel):
    ltl: list[str] = Field(default_factory=list)
    stl: list[str] = Field(default_factory=list)
    program_rules: list[str] = Field(default_factory=list)


class LowAltitudeIR(BaseModel):
    task_id: str
    intent: Intent
    priority: Priority
    entities: EntityRefs
    constraints: OperationConstraints
    tool_plan: list[ToolCallSpec]
    verification_specs: VerificationSpecs
    fallback_policy: Literal[
        "ground_transfer",
        "wait",
        "human_confirm",
        "safe_refusal",
        "ground_transfer_or_human_confirm",
    ]
    explanation_plan: dict = Field(default_factory=dict)

    @field_validator("tool_plan")
    @classmethod
    def check_nonempty_tool_plan(cls, value: list[ToolCallSpec]) -> list[ToolCallSpec]:
        if not value:
            raise ValueError("tool_plan must contain at least one tool call")
        return value
```

实体 grounding 不要写在 Pydantic 里，而是单独做，因为它依赖当前地图和 UAV 状态。

```python
def validate_entity_grounding(ir: LowAltitudeIR, state: SystemState) -> ValidationReport:
    errors: list[ValidationErrorItem] = []

    known_entities = state.known_entity_ids()
    known_uavs = state.known_uav_ids()

    for field_name in ["origin", "destination"]:
        value = getattr(ir.entities, field_name)
        if value is not None and value not in known_entities:
            errors.append(
                ValidationErrorItem(
                    stage="entity_grounding",
                    field=f"entities.{field_name}",
                    value=value,
                    error_type="unknown_entity",
                )
            )

    for uav_id in ir.entities.candidate_uavs:
        if uav_id not in known_uavs:
            errors.append(
                ValidationErrorItem(
                    stage="entity_grounding",
                    field="entities.candidate_uavs",
                    value=uav_id,
                    error_type="unknown_uav",
                )
            )

    return ValidationReport(valid=not errors, errors=errors)
```

### 16.7 ToolRegistry 代码细节

所有工具都实现同一个接口，方便替换 deterministic / learned / external MCP 版本。

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter
from typing import Any


class ToolErrorType(str, Enum):
    unknown_region = "unknown_region"
    restricted_airspace = "restricted_airspace"
    no_available_uav = "no_available_uav"
    no_path = "no_path"
    spec_violation = "spec_violation"
    sim_failure = "sim_failure"
    high_risk = "high_risk"
    invalid_arguments = "invalid_arguments"


class ToolError(BaseModel):
    type: ToolErrorType | str
    message: str
    recoverable: bool = True
    suggested_actions: list[str] = Field(default_factory=list)


class ToolResult(BaseModel):
    ok: bool
    tool: str
    request_id: str
    result: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)
    error: ToolError | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    latency_sec: float = 0.0


class BaseTool(ABC):
    name: str

    @abstractmethod
    def run(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        ...


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def execute(self, call: ToolCallSpec, context: ToolContext) -> ToolResult:
        start = perf_counter()
        if call.tool not in self._tools:
            return ToolResult(
                ok=False,
                tool=call.tool,
                request_id=context.next_request_id(call.tool),
                error=ToolError(
                    type="unknown_tool",
                    message=f"Tool {call.tool} is not registered.",
                    recoverable=True,
                    suggested_actions=["choose a registered tool"],
                ),
            )
        result = self._tools[call.tool].run(call.args, context)
        result.latency_sec = perf_counter() - start
        return result
```

### 16.8 Planner 与 simulator 最小代码设计

第一版 3D A* 只需要支持 grid、NFZ mask、altitude range、battery/length 估计。

```python
def plan_route_astar(
    grid: Grid3D,
    start: GridNode,
    goal: GridNode,
    avoid_mask: set[GridNode],
    altitude_min_layer: int,
    altitude_max_layer: int,
) -> RoutePlan:
    open_set = PriorityQueue()
    open_set.put((0.0, start))
    came_from: dict[GridNode, GridNode] = {}
    g_score = {start: 0.0}

    while not open_set.empty():
        _, current = open_set.get()
        if current == goal:
            return reconstruct_route(came_from, current)

        for nxt in grid.neighbors_26(current):
            if nxt in avoid_mask:
                continue
            if not altitude_min_layer <= nxt.z <= altitude_max_layer:
                continue
            tentative = g_score[current] + grid.edge_cost(current, nxt)
            if tentative < g_score.get(nxt, float("inf")):
                came_from[nxt] = current
                g_score[nxt] = tentative
                priority = tentative + euclidean_distance(nxt, goal)
                open_set.put((priority, nxt))

    return RoutePlan(ok=False, failure_type="no_path")
```

轻量 simulator 用离散时间推进即可：

```python
def simulate_route(
    route: RoutePlan,
    scenario: ScenarioState,
    dt_sec: float = 1.0,
) -> SimulationResult:
    events: list[SimEvent] = []
    min_distance = float("inf")
    elapsed = 0.0

    for segment in route.segments:
        for pose in interpolate_segment(segment, dt_sec):
            elapsed += dt_sec
            distance = scenario.min_distance_to_obstacles(pose)
            min_distance = min(min_distance, distance)

            if scenario.inside_no_fly_zone(pose):
                events.append(SimEvent(time=elapsed, type="nfz_intrusion", pose=pose))
            if distance < scenario.min_separation_m:
                events.append(SimEvent(time=elapsed, type="separation_violation", pose=pose))
            if scenario.weather_risk_at(pose, elapsed) == "high":
                events.append(SimEvent(time=elapsed, type="weather_risk", pose=pose))

    return SimulationResult(
        success=not any(e.is_terminal for e in events),
        events=events,
        min_distance_m=min_distance,
        elapsed_sec=elapsed,
    )
```

### 16.9 Verifier 最小代码设计

第一版可以把 LTL/STL 分成两层：常见规则用程序 checker 保证稳定，复杂表达式再交给 Spot/RTAMT。

```python
def verify_common_rules(
    trajectory: Trajectory,
    specs: VerificationSpecs,
    scenario: ScenarioState,
) -> VerificationResult:
    violations: list[Violation] = []

    if "G not_in_nfz" in specs.ltl:
        for t, pose in trajectory.iter_poses():
            if scenario.inside_no_fly_zone(pose):
                violations.append(
                    Violation(
                        rule="G not_in_nfz",
                        time_sec=t,
                        failure_type="nfz_intrusion",
                        recoverable=True,
                    )
                )

    for stl_spec in specs.stl:
        if stl_spec.startswith("distance_to_obstacle"):
            robustness = min(
                scenario.distance_to_nearest_obstacle(pose) - scenario.min_separation_m
                for _, pose in trajectory.iter_poses()
            )
            if robustness < 0:
                violations.append(
                    Violation(
                        rule=stl_spec,
                        time_sec=trajectory.time_of_min_distance(scenario),
                        failure_type="negative_robustness",
                        robustness=robustness,
                        recoverable=True,
                    )
                )

    return VerificationResult(pass_=not violations, violations=violations)
```

Counterexample 压缩要短，不要把全轨迹塞回 prompt：

```python
def compress_counterexample(verdict: VerificationResult) -> dict:
    first = next(iter(verdict.violations))
    return {
        "failure_type": first.failure_type,
        "violated_rule": first.rule,
        "time_sec": first.time_sec,
        "robustness": first.robustness,
        "suggested_repair": suggest_repair(first),
    }
```

### 16.10 Agent runner 代码细节

`run_agent` 需要完整保存 trace，方便复现实验和错误分析。

```python
def run_agent(sample: Sample, model: ChatModel, tools: ToolRegistry, cfg: AgentConfig) -> AgentTrace:
    trace = AgentTrace(sample_id=sample.sample_id, method=cfg.method, model=model.name)
    context = build_context(sample, cfg)
    feedback: dict | None = None

    for repair_round in range(cfg.max_repair_rounds + 1):
        llm_output = model.generate(
            messages=build_messages(sample.instruction, context, feedback),
            temperature=cfg.temperature_for_round(repair_round),
            max_tokens=cfg.max_tokens,
        )
        trace.add_llm_call(llm_output, repair_round=repair_round)

        parse_report = parse_low_altitude_ir(llm_output.text)
        if not parse_report.ok:
            feedback = {"stage": "parse", "errors": parse_report.errors}
            trace.add_validation_failure(feedback)
            continue

        ir = parse_report.ir
        validation = validate_ir_all(ir, sample.state, tools)
        if not validation.valid:
            feedback = {"stage": "validation", "errors": validation.to_prompt_feedback()}
            trace.add_validation_failure(feedback)
            continue

        tool_trace = execute_tool_plan(ir, tools, sample.state)
        trace.add_tool_trace(tool_trace)

        if tool_trace.has_unrecoverable_error:
            trace.final_status = "safe_refusal"
            trace.final_reason = tool_trace.first_unrecoverable_error.type
            return trace

        verdict = verify_and_simulate(ir, tool_trace, sample.state)
        trace.add_verdict(verdict)

        if verdict.pass_:
            trace.final_status = "success"
            trace.final_decision = build_final_decision(ir, tool_trace, verdict)
            return trace

        feedback = compress_counterexample(verdict)

    trace.final_status = "human_confirm_or_safe_refusal"
    trace.final_reason = "max_repair_rounds_exceeded"
    return trace
```

Trace JSONL 每行建议保存：

```json
{
  "sample_id": "cb_000001",
  "method": "cloudbrain_full",
  "model": "qwen3-14b",
  "final_status": "success",
  "repair_rounds": 1,
  "llm_calls": [],
  "tool_calls": [],
  "validation_errors": [],
  "verifier_verdicts": [],
  "latency": {
    "total_sec": 18.4,
    "llm_sec": 13.2,
    "tool_sec": 4.1
  }
}
```

### 16.11 Evaluation 代码细节

Metric 要从 trace 和 gold 自动计算，避免人工主观判断。

```python
def compute_tool_call_accuracy(gold: list[ToolCallSpec], pred: list[ToolCallRecord]) -> float:
    if not gold:
        return 1.0 if not pred else 0.0
    matched = 0
    for gold_call, pred_call in zip(gold, pred):
        if gold_call.tool != pred_call.tool:
            continue
        if not args_compatible(gold_call.args, pred_call.args):
            continue
        matched += 1
    return matched / max(len(gold), len(pred), 1)


def compute_safety_violation(trace: AgentTrace) -> bool:
    if trace.final_status not in {"success", "safe_refusal"}:
        return True
    for verdict in trace.verifier_verdicts:
        if any(v.is_safety_critical for v in verdict.violations):
            return True
    for event in trace.sim_events:
        if event.type in {"collision", "nfz_intrusion", "separation_violation"}:
            return True
    return False


def evaluate_trace(sample: Sample, trace: AgentTrace) -> MetricRow:
    return MetricRow(
        sample_id=sample.sample_id,
        method=trace.method,
        model=trace.model,
        task_success=trace.final_status == "success",
        executable_decision=trace.has_executable_route(),
        safety_violation=compute_safety_violation(trace),
        hallucination=trace.has_unknown_entity_or_tool(),
        tool_call_accuracy=compute_tool_call_accuracy(sample.gold_tool_trace, trace.tool_calls),
        repair_success=trace.first_attempt_failed_and_later_succeeded(),
        latency_sec=trace.latency.total_sec,
    )
```

`pass^k` 计算方式：

```python
def compute_pass_k(rows: list[MetricRow], k: int) -> float:
    by_sample = group_by(rows, key=lambda row: row.sample_id)
    success_count = 0
    for sample_id, sample_rows in by_sample.items():
        repeated = sorted(sample_rows, key=lambda row: row.repeat_id)[:k]
        if len(repeated) == k and all(row.task_success for row in repeated):
            success_count += 1
    return success_count / len(by_sample)
```

### 16.12 单元测试计划

第一阶段测试要覆盖“不会写错论文实验”的基础问题：

| 测试文件 | 测试内容 |
|----------|----------|
| `test_ir_schema.py` | enum、必填字段、高度范围、电量比例 |
| `test_entity_grounding.py` | 不存在 UAV/POI/NFZ 能被抓到 |
| `test_tool_registry.py` | 未注册工具、重复注册、错误返回格式 |
| `test_planner.py` | 简单可达、NFZ 阻挡、无路可达 |
| `test_verifier.py` | NFZ、deadline、distance robustness |
| `test_simulator.py` | collision、near miss、weather risk |
| `test_agent_runner.py` | schema fail -> repair、verifier fail -> repair、unrecoverable -> refusal |
| `test_metrics.py` | TSR、SVR、TCA、RSR、pass^k 计算 |

建议最小测试样例：

```python
def test_invalid_altitude_range_is_rejected() -> None:
    with pytest.raises(ValueError):
        OperationConstraints(altitude_min_m=120, altitude_max_m=30)


def test_unknown_uav_is_entity_grounding_error(simple_state: SystemState) -> None:
    ir = make_valid_ir()
    ir.entities.candidate_uavs = ["uav_missing"]
    report = validate_entity_grounding(ir, simple_state)
    assert not report.valid
    assert next(iter(report.errors)).error_type == "unknown_uav"


def test_repair_success_metric() -> None:
    trace = make_trace(statuses=["validation_failed", "verifier_failed", "success"])
    assert trace.first_attempt_failed_and_later_succeeded()
```

### 16.13 第一版实现优先级

不要所有模块同时做。建议按“论文最小证据链”排序：

| 优先级 | 模块 | 为什么先做 |
|--------|------|------------|
| P0 | `LowAltitudeIR` schema + validators | 没有它无法证明 typed IR 贡献 |
| P0 | deterministic tools + trace logging | 所有实验都依赖 |
| P0 | 3D A* + basic verifier | 支撑 executable/safety 指标 |
| P0 | direct/ReAct/CloudBrain baseline runner | 形成第一张主表 |
| P1 | simulator stress seeds | 支撑 safety-critical 叙事 |
| P1 | repair loop + counterexample compression | 本文核心方法 |
| P1 | metrics + aggregation | 防止结果不可复现 |
| P2 | OSM/Overture/Open-Meteo loaders | 增强真实性 |
| P2 | MCP wrapper | 增强工程叙事，但不阻塞论文 |
| P3 | AirSim/Flightmare | 后续扩展，不阻塞 G1 |

### 16.14 MCP wrapper 代码细节

第一版工具可以先用 Python registry 跑通，之后再把同一批工具包装成 MCP server。这样做的好处是：论文实验不被 MCP 工程细节卡住，但系统叙事可以自然连接“云脑工具生态”。

MCP server 的工具命名建议与 Python registry 保持一致：

| MCP tool | Python tool | 说明 |
|----------|-------------|------|
| `cloudbrain.query_city_state` | `query_city_state` | 城市实体和地图状态 |
| `cloudbrain.query_airspace` | `query_airspace` | corridor、NFZ、高度、容量 |
| `cloudbrain.assign_uav` | `assign_uav` | UAV-task assignment |
| `cloudbrain.plan_route` | `plan_route` | 3D route planner |
| `cloudbrain.verify_ltl_stl` | `verify_ltl_stl` | safety verifier |
| `cloudbrain.simulate_scenario` | `simulate_scenario` | stress simulator |
| `cloudbrain.risk_assess` | `risk_assess` | risk scoring |

MCP wrapper 伪代码：

```python
from mcp.server.fastmcp import FastMCP

from cloudbrain.tools.registry import build_default_registry
from cloudbrain.tools.base import ToolContext


mcp = FastMCP("cloudbrain-tools")
registry = build_default_registry()


@mcp.tool()
def query_airspace(region: str, altitude_min_m: float, altitude_max_m: float, time_sec: int) -> dict:
    context = ToolContext.from_active_scenario()
    result = registry.execute_by_name(
        "query_airspace",
        {
            "region": region,
            "altitude_range": [altitude_min_m, altitude_max_m],
            "time_sec": time_sec,
        },
        context,
    )
    return result.model_dump()


@mcp.tool()
def plan_route(start: str, goal: str, avoid_zones: list[str], altitude_min_m: float, altitude_max_m: float) -> dict:
    context = ToolContext.from_active_scenario()
    result = registry.execute_by_name(
        "plan_route",
        {
            "start": start,
            "goal": goal,
            "avoid_zones": avoid_zones,
            "altitude_range": [altitude_min_m, altitude_max_m],
        },
        context,
    )
    return result.model_dump()


if __name__ == "__main__":
    mcp.run()
```

MCP 工程约束：

- MCP tool 返回值仍使用 `ToolResult` schema，避免两套协议。
- MCP server 不直接读写实验结果，只执行工具；trace 由 agent runner 保存。
- 若 MCP 调用失败，agent runner 要能 fallback 到 Python registry，保证实验不中断。
- 论文实验主结果建议使用 Python registry，MCP wrapper 放系统演示或附录。

### 16.15 数据生成器代码细节

数据生成器要 deterministic，核心输入是 seed 和 config。

```python
def generate_sample(seed: int, cfg: DataGenConfig) -> Sample:
    rng = np.random.default_rng(seed)
    context = load_context_bundle(cfg.context, rng)
    city = generate_city_layout(rng, cfg.city, context.map_snapshot)
    airspace = generate_airspace(city, rng, cfg.airspace, context.uasfm_snapshot)
    uavs = generate_uav_fleet(city, rng, cfg.fleet)
    task = generate_task(city, airspace, uavs, rng, cfg.task, context.weather_snapshot)

    gold_ir = build_gold_ir(task, city, airspace, uavs, cfg.rules)
    tool_context = ToolContext(
        city=city,
        airspace=airspace,
        uavs=uavs,
        weather=context.weather_snapshot,
        energy_calibration=context.energy_calibration,
    )
    gold_trace = execute_gold_tool_trace(gold_ir, tool_context)
    verdict = verify_and_simulate(gold_ir, gold_trace, tool_context)

    instruction = paraphrase_instruction(task, gold_ir, rng, cfg.language)

    return Sample(
        sample_id=f"cb_{seed:08d}",
        data_tier=context.data_tier,
        generation_seed=seed,
        city_id=context.city_id,
        scenario_type=task.scenario_type,
        instruction=instruction,
        source_provenance=context.provenance,
        real_context=context.real_context_metadata(),
        energy_calibration_version=context.energy_calibration.version,
        state=SystemState(city=city, airspace=airspace, uavs=uavs, tasks=[task]),
        gold_ir=gold_ir,
        gold_tool_trace=gold_trace,
        gold_verdict=verdict,
        label="SAT" if verdict.pass_ else "UNSAT",
        failure_modes=verdict.failure_modes,
    )
```

Split 生成要保证没有信息泄漏：

```python
def assign_split(sample: Sample, cfg: SplitConfig) -> str:
    if sample.data_tier == "real_flight_calibrated":
        return "test_energy_calibrated"
    if sample.data_tier == "real_context" and sample.city_id in cfg.real_context_holdout_city_ids:
        return "test_real_context_city_b"
    if sample.data_tier == "real_context" and sample.has_weather_stress:
        return "test_real_weather_stress"
    if sample.data_tier == "real_context":
        return "test_real_context_city_a"
    if sample.city_id in cfg.unseen_city_ids:
        return "test_unseen_city"
    if sample.scenario_type in cfg.stress_scenario_types:
        return "test_stress"
    if sample.label == "UNSAT":
        return "test_unsat"
    bucket = stable_hash(sample.sample_id) % 100
    if bucket < 10:
        return "validation"
    if bucket < 20:
        return "test_seen_city"
    return "train_like"
```

生成器必须输出 split stats：

```json
{
  "split": "test_stress",
  "num_samples": 1000,
  "sat_rate": 0.74,
  "scenario_counts": {
    "emergency_delivery_with_nfz": 210,
    "corridor_congestion": 180
  },
  "avg_tool_trace_len": 5.8,
  "avg_constraints_per_task": 4.2
}
```

### 16.16 vLLM 与本地模型启动方案

本地模型建议通过 vLLM 暴露 OpenAI-compatible endpoint。这样 `llm_client.py` 只维护一个接口。

示例启动命令：

```bash
vllm serve Qwen/Qwen3-14B \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen3-14b \
  --tensor-parallel-size 1 \
  --max-model-len 32768
```

DeepSeek repair specialist：

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --host 0.0.0.0 \
  --port 8001 \
  --served-model-name deepseek-r1-distill-qwen-14b \
  --tensor-parallel-size 1 \
  --max-model-len 32768
```

统一客户端伪代码：

```python
class ChatModel:
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def generate(self, messages: list[dict], temperature: float, max_tokens: int) -> LLMOutput:
        start = perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency = perf_counter() - start
        first_choice = next(iter(response.choices))
        content = first_choice.message.content or ""
        usage = response.usage
        return LLMOutput(text=content, latency_sec=latency, usage=usage.model_dump() if usage else {})
```

运行记录必须写入 `model_manifest.json`：

```json
{
  "model": "qwen3-14b",
  "provider": "local_vllm",
  "base_url": "http://localhost:8000/v1",
  "temperature": 0.0,
  "top_p": 1.0,
  "max_tokens": 4096,
  "system_fingerprint": "local",
  "prompt_version": "cloudbrain_v0.3"
}
```

### 16.17 缓存、日志与复现

为了控制 API 成本和实验时间，所有 LLM 调用、工具调用、verifier 结果都要缓存。

缓存 key：

```python
def cache_key(prefix: str, payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"
```

推荐缓存目录：

```text
runs/
  20260520_cloudbrain_main/
    config.yaml
    model_manifest.json
    samples.jsonl
    traces.jsonl
    metrics.jsonl
    aggregate.csv
    cache/
      llm/
      tools/
      verifier/
    logs/
      run.log
      errors.log
```

每条 trace 必须包含：

- `sample_id`
- `method`
- `model`
- `prompt_version`
- `config_hash`
- `random_seed`
- `repair_rounds`
- `final_status`
- `metric_row`
- `all_tool_results`
- `all_verifier_results`

### 16.18 CI 与质量门禁

即使第一阶段只有规划和实验代码，也要设质量门禁：

```yaml
checks:
  formatting:
    - ruff format --check src tests
  lint:
    - ruff check src tests
  typing:
    - mypy src
  unit_tests:
    - pytest tests -q
  smoke:
    - python -m cloudbrain.data.generate --config configs/data/dev_mini.yaml --limit 20
    - python -m cloudbrain.eval.run --split dev_mini --method cloudbrain_full --limit 5 --mock-llm
```

Smoke test 只用 mock LLM，保证 CI 不依赖 GPU 或 API key。Mock LLM 返回固定 IR，用于验证工具链、metrics 和 trace logging。

### 16.19 实验矩阵配置

最终主实验最少跑这个矩阵：

| Split | Method | Model | Repeat |
|-------|--------|-------|--------|
| validation | direct_llm / react / cloudbrain_full | qwen3-14b | 1 |
| test_seen_city | all main baselines | qwen3-14b | 3 |
| test_unseen_city | all main baselines | qwen3-14b | 3 |
| test_stress | all main baselines | qwen3-14b | 3 |
| test_unsat | direct_llm / react / cloudbrain_full | qwen3-14b | 3 |
| test_seen_city | cloudbrain_full | qwen3-8b / qwen3-32b / deepseek-repair / GPT upper bound | 3 |

自动生成实验任务：

```python
def build_experiment_matrix(cfg: MatrixConfig) -> list[ExperimentJob]:
    jobs = []
    for split in cfg.splits:
        for method in cfg.methods_for_split(split):
            for model in cfg.models_for_method(method):
                for repeat_id in range(cfg.repeats):
                    jobs.append(
                        ExperimentJob(
                            split=split,
                            method=method,
                            model=model,
                            repeat_id=repeat_id,
                            seed=stable_seed(split, method, model, repeat_id),
                        )
                    )
    return jobs
```

### 16.20 论文附录应包含的工程材料

为提高可复现性，G1 附录至少放：

| 附录 | 内容 |
|------|------|
| A | Full `LowAltitudeIR` JSON schema |
| B | Tool registry schema and error taxonomy |
| C | Data generation config and scenario taxonomy |
| D | Prompt templates for each baseline |
| E | Full metric definitions and bootstrap procedure |
| F | Extra ablation and per-scenario results |
| G | Failure case visualizations |
| H | Compute budget, model versions, cache policy |

---

## 17. 风险与备选方案

### 17.1 风险：效果不明显

备选：加大 stress/UNSAT 比例，让 verifier repair 的价值更突出；同时报告不同任务难度下的收益。

### 17.2 风险：本地模型太弱

备选：主实验用 Qwen3-32B，Qwen3-14B 做可部署版本；GPT-5.2 只做 upper bound。也可将 DeepSeek-R1-Distill 用作 repair-only specialist。

### 17.3 风险：数据被认为太合成

备选：把主实验拆成 `Synthetic-Controlled`、`Real-Context`、`Real-Flight-Calibrated` 三层，分别报告程序真值、真实城市/空域/天气 grounding、真实飞行能耗校准。论文表述明确写成 benchmark + method，不把 UASFM、METAR 或 DJI M100 数据夸大成真实商用 fleet operations。

### 17.4 风险：MCP 实现拖慢进度

备选：第一版工具先用 Python function registry，投稿时把接口描述成 MCP-compatible。真正 MCP server 放开源附录或后续版本。

### 17.5 风险：形式化验证过重

备选：LTL 先做离散事件约束，STL 先做距离、高度、deadline、battery 四类连续约束。不要一开始覆盖所有低空法规。

### 17.6 风险：Agent benchmark 贡献不够通用

备选：把 CloudBrain-Bench 的任务拆成通用 agent 评测维度：tool selection、argument grounding、state dependency、policy compliance、counterexample repair、`pass^k` consistency。这样即使审稿人不熟低空交通，也能理解它对安全关键 agent evaluation 的贡献。

---

## 18. 参考文献

[1] AAAI. “AAAI-26 Main Technical Track: Call for Papers.” URL: <https://aaai.org/conference/aaai/aaai-26/main-technical-track-call/>

[2] IJCAI-ECAI 2026. “Call for Papers — AI and Robotics Special Track.” URL: <https://2026.ijcai.org/ijcai-ecai-2026-call-for-papers-ai-and-robotics/>

[3] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. “LoRA: Low-Rank Adaptation of Large Language Models.” *International Conference on Learning Representations (ICLR)*, 2022. URL: <https://openreview.net/forum?id=nZeVKeeFYf9>

[4] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. “QLoRA: Efficient Finetuning of Quantized LLMs.” *Advances in Neural Information Processing Systems 36 (NeurIPS)*, 2023. URL: <https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html>

[5] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. “Direct Preference Optimization: Your Language Model is Secretly a Reward Model.” *Advances in Neural Information Processing Systems 36 (NeurIPS)*, 2023. URL: <https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html>

[6] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. “ReAct: Synergizing Reasoning and Acting in Language Models.” *International Conference on Learning Representations (ICLR)*, 2023. URL: <https://openreview.net/forum?id=WE_vluYUL-X>

[7] Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu, and Maosong Sun. “ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs.” *International Conference on Learning Representations (ICLR)*, 2024. URL: <https://openreview.net/forum?id=dHng2O0Jjr>

[8] Bo Liu, Yuqian Jiang, Xiaohan Zhang, Qiang Liu, Shiqi Zhang, Joydeep Biswas, and Peter Stone. “LLM+P: Empowering Large Language Models with Optimal Planning Proficiency.” arXiv:2304.11477, 2023. URL: <https://arxiv.org/abs/2304.11477>

[9] Siyao Zhang, Daocheng Fu, Wenzhe Liang, Zhao Zhang, Bin Yu, Pinlong Cai, and Baozhen Yao. “TrafficGPT: Viewing, Processing and Interacting with Traffic Foundation Models.” *Transport Policy*, 150:95-105, 2024. DOI: 10.1016/j.tranpol.2024.03.006. URL: <https://www.sciencedirect.com/science/article/pii/S0967070X24000726>

[10] OpenAI. “GPT-5.2 Model.” *OpenAI API Documentation*, 2025. URL: <https://platform.openai.com/docs/models/gpt-5.2>

[11] Qwen Team. “Qwen3 Technical Report.” arXiv:2505.09388, 2025. URL: <https://arxiv.org/abs/2505.09388>

[12] DeepSeek-AI. “DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.” arXiv:2501.12948, 2025. URL: <https://arxiv.org/abs/2501.12948>

[13] Sebastian Wandelt, Changhong Zheng, Shuang Wang, Yucheng Liu, and Xiaoqian Sun. “Large Language Models for Intelligent Transportation: A Review of the State of the Art and Challenges.” *Applied Sciences*, 14(17):7455, 2024. DOI: 10.3390/app14177455. URL: <https://www.mdpi.com/2076-3417/14/17/7455>

[14] Doaa Mahmud, Hadeel Hajmohamed, Shamma Almentheri, Shamma Alqaydi, Lameya Aldhaheri, Ruhul Amin Khalil, and Nasir Saeed. “Integrating LLMs With ITS: Recent Advances, Potentials, Challenges, and Future Directions.” *IEEE Transactions on Intelligent Transportation Systems*, 26(5):5674-5709, 2025. DOI: 10.1109/TITS.2025.3528116. URL: <https://ieeexplore.ieee.org/document/10851302>

[15] Zhonghang Li, Lianghao Xia, Jiabin Tang, Yong Xu, Lei Shi, Long Xia, Dawei Yin, and Chao Huang. “UrbanGPT: Spatio-Temporal Large Language Models.” arXiv:2403.00813, 2024. URL: <https://arxiv.org/abs/2403.00813>

[16] Yuan Yuan, Jingtao Ding, Jie Feng, Depeng Jin, and Yong Li. “UniST: A Prompt-Empowered Universal Model for Urban Spatio-Temporal Prediction.” *Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)*, 2024. DOI: 10.1145/3637528.3671662. URL: <https://arxiv.org/abs/2402.11838>

[17] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. “Toolformer: Language Models Can Teach Themselves to Use Tools.” *Advances in Neural Information Processing Systems 36 (NeurIPS)*, 2023. URL: <https://proceedings.neurips.cc/paper_files/paper/2023/hash/d842425e4bf79ba039352da0f658a906-Abstract-Conference.html>

[18] OpenAI. “Model Context Protocol (MCP) — OpenAI Agents SDK.” URL: <https://openai.github.io/openai-agents-js/guides/mcp/>

[19] OpenAI. “Tools — OpenAI Agents SDK.” URL: <https://openai.github.io/openai-agents-js/guides/tools/>

[20] Karthik Valmeekam, Matthew Marquez, Alberto Olmo, Sarath Sreedharan, and Subbarao Kambhampati. “PlanBench: An Extensible Benchmark for Evaluating Large Language Models on Planning and Reasoning about Change.” *Advances in Neural Information Processing Systems 36 (NeurIPS) Datasets and Benchmarks Track*, 2023. URL: <https://openreview.net/forum?id=YXogl4uQUO>

[21] Bo Liu, Yuqian Jiang, Xiaohan Zhang, Qiang Liu, Shiqi Zhang, Joydeep Biswas, and Peter Stone. “Lang2LTL: Translating Natural Language Commands to Temporal Specification with Large Language Models.” *Conference on Robot Learning (CoRL)*, PMLR 229, 2023. URL: <https://proceedings.mlr.press/v229/liu23d.html>

[22] Behrad Rabiei, Mahesh Kumar A. R., Zhirui Dai, Surya L. S. R. Pilla, Qiyue Dong, and Nikolay Atanasov. “LTLCodeGen: Code Generation of Syntactically Correct Temporal Logic for Robot Task Planning.” arXiv:2503.07902, 2025. URL: <https://arxiv.org/abs/2503.07902>

[23] Jun Wang, David Smith Sundarsingh, Jyotirmoy V. Deshmukh, and Yiannis Kantaros. “ConformalNL2LTL: Translating Natural Language Instructions into Temporal Logic Formulas with Conformal Correctness Guarantees.” arXiv:2504.21022, 2025. URL: <https://arxiv.org/abs/2504.21022>

[24] Alexandre Duret-Lutz, Alexandre Lewkowicz, Amaury Fauchille, Thibault Michaud, Etienne Renault, and Laurent Xu. “Spot 2.0: A Framework for LTL and ω-Automata Manipulation.” *International Symposium on Automated Technology for Verification and Analysis (ATVA)*, 2016. URL: <https://spot.lre.epita.fr/>

[25] Bardh Hoxha, Houssam Abbas, and Georgios Fainekos. “RTAMT: Online Robustness Monitors from STL.” arXiv:2005.11827, 2020. URL: <https://arxiv.org/abs/2005.11827>

[26] Federal Aviation Administration. “Unmanned Aircraft System Traffic Management (UTM).” URL: <https://www.faa.gov/uas/advanced_operations/traffic_management>

[27] Federal Aviation Administration. “UAS Facility Maps.” URL: <https://www.faa.gov/uas/commercial_operators/uas_facility_maps>

[28] OpenStreetMap / Overpass API. “OpenStreetMap and the Overpass API.” URL: <https://dev.overpass-api.de/overpass-doc/en/preface/preface.html>

[29] New York City Taxi and Limousine Commission. “TLC Trip Record Data.” URL: <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page>

[30] Eclipse SUMO. “SUMO Documentation.” URL: <https://sumo.dlr.de/docs/index.html>

[31] Shital Shah, Debadeepta Dey, Chris Lovett, and Ashish Kapoor. “AirSim: High-Fidelity Visual and Physical Simulation for Autonomous Vehicles.” *Field and Service Robotics*, Springer Proceedings in Advanced Robotics, 2017; arXiv:1705.05065. URL: <https://arxiv.org/abs/1705.05065>

[32] Yunlong Song, Selim Naji, Elia Kaufmann, Antonio Loquercio, and Davide Scaramuzza. “Flightmare: A Flexible Quadrotor Simulator.” *Conference on Robot Learning (CoRL)*, PMLR 155, 2021. URL: <https://proceedings.mlr.press/v155/song21a.html>

[33] vLLM Team. “OpenAI-Compatible Server.” *vLLM Documentation*. URL: <https://docs.vllm.ai/serving/openai_compatible_server.html>

[34] Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, and others. “AgentBench: Evaluating LLMs as Agents.” *International Conference on Learning Representations (ICLR)*, 2024. URL: <https://openreview.net/forum?id=zAdUB0aCTQ>

[35] Ivan Ortega, Fanjia Yan, Huanzhi Mao, Charlie Cheng-Jie Ji, Tianjun Zhang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. “Berkeley Function-Calling Leaderboard.” UC Berkeley Sky Computing Lab project page, 2024/2025. URL: <https://sky.cs.berkeley.edu/project/berkeley-function-calling-leaderboard/>

[36] Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. “$\tau$-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.” arXiv:2406.12045, 2024. URL: <https://arxiv.org/abs/2406.12045>

[37] Jiarui Lu, Thomas Holleis, Yizhe Zhang, Bernhard Aumayer, Feng Nan, Felix Bai, Shuang Ma, Shen Ma, Mengyu Li, Guoli Yin, Zirui Wang, and Ruoming Pang. “ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities.” arXiv:2408.04682, revised 2025. URL: <https://arxiv.org/abs/2408.04682>

[38] Lynne Martin, Cynthia Wolter, Kimberly Jobe, Mariah Manzano, Stefan Bladin, Michele Cencetti, Lauren Claudatos, Joey Mercer, and Jeffrey Homola. “TCL4 UTM (UAS Traffic Management) Nevada 2019 Flight Tests, Airspace Operations Laboratory (AOL) Report.” NASA Technical Memorandum NASA/TM-2020-220516, 2020. URL: <https://ntrs.nasa.gov/citations/20205003361>

[39] EUROCONTROL. “CORUS-XUAM: Concept of Operations for European UTM Systems — Extension for Urban Air Mobility.” Project page, 2023. URL: <https://www.eurocontrol.int/project/corus-xuam>

[40] Marc Brittain, Luis E. Alvarez, Kara Breeden, and Ian Jessen. “AAM-Gym: Artificial Intelligence Testbed for Advanced Air Mobility.” *IEEE/AIAA Digital Avionics Systems Conference (DASC)*, 2022; arXiv:2206.04513. URL: <https://arxiv.org/abs/2206.04513>

[41] Overture Maps Foundation. “Overture Maps Documentation: Places, Buildings, and Transportation Data.” URL: <https://docs.overturemaps.org/>

[42] Open-Meteo. “Weather Forecast API and Historical Forecast API Documentation.” URL: <https://open-meteo.com/en/docs>

[43] Federal Aviation Administration. “UAS Data Delivery System Data Dictionary.” PDF, 2022. URL: <https://www.faa.gov/sites/faa.gov/files/2022-08/UAS_Data_Delivery_System_Data_Dictionary.pdf>

[44] Aviation Weather Center. “Data API.” National Oceanic and Atmospheric Administration, documentation page. URL: <https://aviationweather.gov/data/api/>

[45] Thiago A. Rodrigues, Jay Patrikar, Arnav Choudhry, Jacob Feldgoise, Vaibhav Arcot, Aradhana Gahlaut, Sophia Lau, Brady Moon, Bastian Wagner, H. Scott Matthews, Sebastian Scherer, and Constantine Samaras. “In-flight positional and energy use data set of a DJI Matrice 100 quadcopter for small package delivery.” *Scientific Data*, 8:155, 2021. DOI: 10.1038/s41597-021-00930-x. URL: <https://www.nature.com/articles/s41597-021-00930-x>

[46] Federal Aviation Administration. “Package Delivery by Drone (Part 135).” URL: <https://www.faa.gov/uas/advanced_operations/package_delivery_drone>

[47] U.S. Government Accountability Office. “Drones: Actions Needed to Better Support Remote Identification in the National Airspace.” GAO-24-106158, 2024. URL: <https://www.gao.gov/products/gao-24-106158>

---

## 附录：本次执行计划

### A. 立即要做

1. 建立 `LowAltitudeIR v0.1` schema。
2. 实现 6 个 deterministic tools：city、airspace、scheduler、planner、verifier、simulator。
3. 生成 200 条 dev-mini 样本。
4. 跑 direct LLM、JSON-only、ReAct、CloudBrain-Agent without repair 四个 baseline。

### B. 第一轮实验通过标准

如果 dev-mini 上满足以下条件，就进入完整 benchmark：

- CloudBrain-Agent full 的 tool-call accuracy 超过 ReAct baseline；
- safety violation rate 低于 direct LLM；
- repair loop 能修复至少一部分 verifier 失败；
- 每条任务平均运行时间不超过可接受阈值，例如本地 14B 模型小于 30 秒。

### C. 下一篇衔接

G1 产生的 tool traces、repair traces、failure cases 和 human review 数据，将直接成为 G2 LowAltitudeGPT 的 SFT/DPO 数据。也就是说，G1 不只是论文，还是垂类模型训练数据工厂。
