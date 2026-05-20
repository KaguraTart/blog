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

---

## 5. 相关工作框架

### 5.1 LLM for transportation

TrafficGPT 说明 LLM 可以作为交通 foundation models 的交互和处理入口，但也指出交通数值数据、仿真和模型交互不能只靠纯文本生成 [9]。近期 ITS 综述进一步将 LLM 放在交通语义接口、决策辅助和多源数据理解的位置 [13] [14]。UrbanGPT 与 UniST 则代表城市时空 foundation model 方向，适合支撑城市状态理解，但它们不是低空 UAV 运行工具链 [15] [16]。

### 5.2 LLM agents and tool use

ReAct 将 reasoning trace 和 action 交织起来，是本文 agent loop 的基础 [6]。Toolformer 与 ToolLLM 证明 LLM 可以学习 API/tool 使用，但它们不解决低空交通安全验证和任务可执行性问题 [7] [17]。MCP 和 OpenAI Agents SDK 提供了更标准的工具连接方式，有利于把 scheduler、planner、verifier 和 simulator 做成可替换工具 [18] [19]。

### 5.3 LLM planning and formal verification

LLM+P 和 PlanBench 表明 LLM 单独做 planning 并不可靠，需要与外部规划器、形式表示和评测协议结合 [8] [20]。Lang2LTL、LTLCodeGen 和 ConformalNL2LTL 说明自然语言到 temporal logic 的翻译正在发展，但它们主要关注规格生成，不完整覆盖低空交通云脑中的调度、路径、仿真和风险闭环 [21] [22] [23]。Spot 与 RTAMT 可以分别作为 LTL/STL 验证工具 [24] [25]。

### 5.4 UAV, UTM, and simulation data

FAA UTM 将低空无人机交通管理定义为支持 flight planning、authorization、surveillance 和 conflict management 的协同生态 [26]。FAA UAS Facility Maps 提供受控空域中 Part 107 操作可快速审批的高度参考，适合做空域规则 proxy [27]。OSM/Overpass、NYC TLC OD 数据、SUMO、AirSim 和 Flightmare 可以共同支撑 synthetic-to-real benchmark [28] [29] [30] [31] [32]。

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

---

## 8. 数据来源与 CloudBrain-Bench 构建

### 8.1 数据构成

第一篇主数据集建议叫 **CloudBrain-Bench**。

| 数据层 | 来源 | 是否主实验依赖 | 作用 |
|--------|------|----------------|------|
| Synthetic city grid | 程序生成 | 是 | 可控、可复现、可扩展 |
| OSM city context | OSM / Overpass | 是 | POI、道路、建筑、功能区命名 |
| Airspace rule proxy | FAA UASFM / UTM 文档 | 是 | 高度、禁飞、授权、UTM 术语 |
| OD demand proxy | NYC TLC / Chicago taxi 可选 | 可选 | 生成需求热区和高峰任务 |
| Ground traffic | SUMO | 可选增强 | 地面 fallback travel time |
| UAV dynamics | 自建轻量 simulator | 是 | 路径、能耗、碰撞、延迟 |
| Visual simulator | AirSim / Flightmare | 可选补充 | 后续视觉/动力学扩展 |

OSM/Overpass 适合查询城市要素 [28]；FAA UTM/UASFM 可作为低空运行和高度约束 proxy [26] [27]；NYC TLC 提供 OD 与时间分布 proxy [29]；SUMO 是开源微观、多模式交通仿真工具 [30]；AirSim 和 Flightmare 可作为 UAV 仿真补充 [31] [32]。

### 8.2 样本格式

每条样本包括：

```json
{
  "sample_id": "cb_000001",
  "city_seed": 12,
  "scenario_type": "emergency_delivery_with_nfz",
  "instruction": "请优先派一架无人机把急救包送到 accident_site_3，避开学校和临时禁飞区，10 分钟内到达。",
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

---

## 10. Baselines

### 10.1 主 baseline

| Baseline | 描述 | 要回答的问题 |
|----------|------|--------------|
| Direct LLM | 模型直接输出决策文本 | LLM 裸跑有多不可靠 |
| JSON-only LLM | 只要求输出 JSON IR，无工具执行 | typed output 是否足够 |
| ReAct prompting | ReAct 风格工具调用，无 schema/verifier | reasoning-action loop 是否足够 |
| Tool-use only | 有工具调用，无 verification repair | 工具是否足够 |
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

### 12.7 统计检验

每个实验至少 3 个随机种子。主结果报告：

- mean ± standard error；
- paired bootstrap 95% confidence interval；
- McNemar test 或 bootstrap test 比较 success/failure 类指标；
- 对 latency 报告 median、p90、p95。

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
| Table 1 | 相关工作对比 | LLM traffic、tool-use、planning、formal verification、本文 | 高 |
| Table 2 | 数据集统计 | 场景类型、SAT/UNSAT、城市、任务数 | 高 |
| Table 3 | Baseline 主结果 | 全指标比较 | 高 |
| Table 4 | Ablation | 删除组件后的性能变化 | 高 |
| Table 5 | 模型部署 | API vs local 的效果、延迟、成本 | 中 |

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

---

## 17. 风险与备选方案

### 17.1 风险：效果不明显

备选：加大 stress/UNSAT 比例，让 verifier repair 的价值更突出；同时报告不同任务难度下的收益。

### 17.2 风险：本地模型太弱

备选：主实验用 Qwen3-32B，Qwen3-14B 做可部署版本；GPT-5.2 只做 upper bound。也可将 DeepSeek-R1-Distill 用作 repair-only specialist。

### 17.3 风险：数据被认为太合成

备选：加强 OSM/FAA/SUMO 的真实增强，并把论文定位为 benchmark + method。明确 synthetic-to-real 而非真实运营数据。

### 17.4 风险：MCP 实现拖慢进度

备选：第一版工具先用 Python function registry，投稿时把接口描述成 MCP-compatible。真正 MCP server 放开源附录或后续版本。

### 17.5 风险：形式化验证过重

备选：LTL 先做离散事件约束，STL 先做距离、高度、deadline、battery 四类连续约束。不要一开始覆盖所有低空法规。

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
