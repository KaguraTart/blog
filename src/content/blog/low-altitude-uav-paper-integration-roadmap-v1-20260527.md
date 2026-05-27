---
title: "低空规划论文矩阵 v2：三篇在做论文、后续具身低空与大模型路线"
description: "以无冲突路径规划、百架 UAV 三层调度、信息论驱动 3DGS 主动感知规划三篇在做论文为核心，重新规划后续具身低空、低空云脑、垂类大模型微调、推理加速与软硬件协同论文路线。"
pubDate: 2026-05-27
updatedDate: 2026-05-28
tags: ["低空规划", "UAV", "论文规划", "Zotero", "T-ITS", "TR-C", "T-RO", "AAAI", "ICRA", "3DGS", "MARL", "Embodied AI", "VLA", "LLM", "推理加速"]
category: Tech
---

# 低空规划论文矩阵 v2：三篇在做论文、后续具身低空与大模型路线

> 本文把目前已经写过的低空 UAV 论文方向重新整合成一个 **paper portfolio**。  
> 目标不是再散开写很多想法，而是明确：哪些文章已经有雏形，哪些可以继续做成顶刊/定会论文，每篇论文需要什么文献支撑、实验资产和投稿定位。

---

## 0. 2026-05-28 纠偏结论

当前重点需要改：不是“同时规划 7-10 篇论文”，而是先承认 **三篇文章已经在做**，后续论文必须从这三篇的资产自然长出来。

已在做的三篇是：

| 状态 | 文章 | 角色 | 不能偏离的主线 |
|---|---|---|---|
| 已在做 | Paper A：无冲突路径规划 / PPO-MAPPO / 低空冲突消解 | 战术安全层 | 高密度低空走廊、非合作 UAV、通信/定位退化、安全-效率折中 |
| 已在做 | Paper B：百架 UAV 三层分层调度 | 系统运营层 | 百架级 fleet、queue stability、vertiport/charging/corridor bottleneck、多模态调度 |
| 已在做 | Paper C：信息论驱动 UAV 3DGS 主动感知 | 环境认知层 | 3DGS / Fisher information / NBV / safe reconstruction / planning-aware mapping |

后续论文不要另起完全无关方向。正确路线是：

1. **先把 A/B/C 做成三篇能投稿的主力论文。**
2. **后续 Paper D/F/G/H/I 只作为 A/B/C 的外延**：场景覆盖支撑 A/C，低空云脑串联 A/B/C，具身低空把 C 的感知和 A 的控制接成闭环，模型微调和推理加速服务云脑落地。
3. **通用 AGI 方向不能写成空泛宣称**。更稳的表达是 “towards general embodied low-altitude intelligence”：从领域 Agent、工具调用、仿真反馈、VLA/VLN、世界模型、端侧推理开始，逐步靠近通用具身智能。
4. **第一阶段不建议从零训练垂类 foundation model**。先用普通大模型 + Agent + Skills/MCP + RAG + verifier + simulator 后处理形成可复现实验闭环；等工具调用轨迹、失败样本、仿真反馈足够多，再做 LoRA/SFT/DPO/GRPO 微调。

这意味着后续论文应该分成两层：

| 层次 | 论文目标 | 是否近期启动 |
|---|---|---|
| A/B/C 主力层 | 已在做，必须优先完成实验闭环 | 立即 |
| D 场景覆盖层 | 给 A/C 提供 benchmark、failure taxonomy 和 safety-critical data | 近期 |
| G 云脑 Agent 层 | 把 A/B/C/D/E 变成工具，做可验证低空交通云脑 | 中期 |
| H 具身低空层 | 做 UAV VLN/VLA/world model，连接通用 embodied AI | 中期偏后 |
| I 模型训练层 | 训练 LowAltitudeGPT / tool-use / LowAltitudeIR / simulation feedback | 等数据稳定后 |
| J 推理加速层 | vLLM/TensorRT-LLM/量化/端云协同/硬件部署 | 等 agent workload 稳定后 |

---

## 1. 当前已有文章与主线定位

目前已经形成内容基础的核心文章有三条：

| 编号 | 已有内容 | 当前定位 | 推荐主投 | 核心判断 |
|---|---|---|---|---|
| Paper A | 无冲突路径规划 / PPO / MAPPO / 多 UAV 冲突消解 | 低空航路网中的鲁棒冲突消解 | IEEE T-ITS / IEEE T-RO / ICRA-IROS | 不能只写 PPO，要写成“非合作 UAV、通信退化、高密度走廊下的安全-效率折中” |
| Paper B | 百架次 UAV 三层调度 | 城市低空物流/应急系统运营调度 | TR-C first, T-ITS backup | 这是交通系统论文，重点是 capacity、delay、queue stability、vertiport/charging/corridor bottleneck |
| Paper C | 信息论驱动的 UAV 3DGS 主动感知规划 | 主动感知 + 低空数字孪生 + 规划闭环 | T-RO / T-ITS / ICRA-IROS | 如果投交通期刊，要证明主动感知提升巡检、应急、避障或运行控制指标 |

这三篇已经能形成一个很稳定的低空规划三角：

```text
Paper A：战术安全
  多 UAV conflict resolution / no-conflict planning / PPO-MAPPO / CBF / RMADER

Paper B：系统运营
  hundred-UAV scheduling / queue stability / Lyapunov / multimodal logistics

Paper C：环境认知
  3DGS active perception / Fisher information / NBV / safe reconstruction
```

后续新增论文最好围绕这三角扩展，不要另起完全无关的方向。

---

## 2. 总体投稿判断

低空规划方向可以分成三类论文，不同类别的评审标准不同：

| 类型 | 代表论文 | 审稿关注 | 推荐 venues |
|---|---|---|---|
| 交通系统论文 | Paper B、应急资源调配、低空路网规划 | 真实交通问题、系统指标、数据/仿真可信度、政策或运营启示 | TR-C、T-ITS |
| 机器人规划论文 | Paper A、Paper C、数字孪生规划 | 算法新意、实时性、安全性、硬件/仿真验证 | T-RO、RA-L+ICRA/IROS、T-ITS |
| AI 方法论文 | VERA-UAV、CloudBrain-Agent、场景加速生成 | benchmark 难度、理论/验证机制、模型泛化、可复现性 | AAAI、IJCAI、NeurIPS/ICLR workshop、T-ITS extension |

TR-C 的官方定位强调 transportation systems and emerging technologies，且 intellectual core 在 transportation side [1]；T-ITS 覆盖 sensing、communications、controls、planning、design、implementation 等现代交通系统技术 [2]。因此：

- **Paper B / 应急资源调配 / 低空路网规划**：优先按 TR-C 的 transportation system operation 逻辑写。
- **Paper A / Paper C**：可以投 T-RO 或 ICRA/IROS；若转 T-ITS，需要补交通系统指标。
- **Paper E/G 类 LLM-Agent**：首篇更适合 AAAI/IJCAI，期刊版再扩展到 T-ITS。

---

## 3. 论文矩阵：3 篇已在做 + 后续外延路线

本节的读法要改：Paper A/B/C 是已经在做的三篇主力，不是“后续新增方向”。Paper D/E/G/H/I/J 是可写外延，但启动顺序必须服从 A/B/C 的实验资产成熟度。

### 3.1 Paper A：低空航路网鲁棒无冲突规划

**建议题目：** Robust Conflict-Free UAV Corridor Planning under Non-Cooperative Traffic and Communication Degradation

**对应已有文章：** 无冲突路径规划、PPO/MAPPO、UAV conflict resolution、UAV conflict env construction。

**核心问题：** 城市低空航路网中，多 UAV 在局部观测、通信延迟、定位误差、非合作飞行器插入条件下，如何保持 separation safety，同时控制延误、额外距离和吞吐损失。

**方法路线：**

- strategic layer：基于航路网的初始路径与时隙分配；
- tactical layer：MAPPO/PPO 输出速度、高度或横向偏移动作；
- safety shield：CBF-QP / ORCA / RMADER-style trajectory check；
- fallback layer：通信退化时切换 conservative priority rule；
- evaluation：训练 30/50 架，测试 100/200 架，覆盖 cooperative、non-cooperative、communication-loss、high-density corridor 四类场景。

**关键参考：**

MAPPO/PPO 的多智能体稳定训练可由 Yu et al. [3] 支撑；MAT 与 FACMAC 提供更强 MARL baseline [4,5]；HAPPO/HATRPO 给出 trust-region multi-agent policy optimization 参考 [6]。机器人侧，EGO-Swarm、MADER、RMADER、RACER、PANTHER 和 GCOPTER 分别支撑 decentralized swarm planning、trajectory sharing under delay、collaborative exploration、perception-aware planning 和 multicopter trajectory optimization [7-12]。

**创新点建议：**

1. 把 “PPO 无冲突路径规划” 从单纯 RL 任务升级为低空交通 corridor safety control。
2. 引入通信退化与非合作 UAV，形成 T-ITS 更关心的实际运行边界。
3. 采用 learning policy + formal/safety shield，避免纯 RL 安全性不足。
4. 指标交通化：LoWC、NMAC、conflict count、average delay、extra distance、throughput、runtime。

### 3.2 Paper B：百架 UAV 三层分层调度

**建议题目：** H-LyraUAV: Queue-Stable Hierarchical Scheduling for Hundred-Scale Low-Altitude UAV Logistics

**对应已有文章：** Paper B 三层调度规划。

**核心问题：** 动态需求、有限 vertiport/charging/corridor 容量、多模态转运约束下，百架级 UAV fleet 如何稳定、高效、安全地运行。

**方法路线：**

- macro layer：demand queue、fleet repositioning、mode choice；
- meso layer：vertiport、charging pad、corridor slot scheduling；
- micro layer：energy/safety/conflict-aware trajectory feasibility；
- theory：Lyapunov drift-plus-penalty 保证 queue stability 和 cost-backlog tradeoff；
- data：synthetic city grid + OSM/POI/NYC TLC/Chicago taxi/SUMO 增强。

**关键参考：**

TR-C 低空 UAV delivery traffic management 已直接讨论 low-altitude urban space 的资源分配和冲突消解 [13]；passenger-centric UAM、公平性与运营效率研究支撑 service quality framing [14]；charging-station delivery network、capacity-constrained UAM scheduling、安全学习调度支撑 infrastructure capacity 和 safe online scheduling [15-17]；truck-drone / UAV-UGV 多模态配送支撑 multimodal extension [18,19]。

**创新点建议：**

1. 百架级在线三层调度闭环，而不是离线 routing/network design。
2. queue stability 成为理论主线，学习模块只做预测或价值估计。
3. 同时评估 delay、throughput、backlog、charging utilization、vertiport bottleneck、corridor congestion。
4. 交通系统结论能回答：什么时候需要限流，哪里是瓶颈，UAV-only 何时不如 multimodal fallback。

### 3.3 Paper C：FIM-3DGS UAV 主动感知规划

**建议题目：** FIM-3DGS: Fisher-Information-Driven Active Perception Planning for Safe UAV Reconstruction

**对应已有文章：** Paper C、Next-Best-View 与 NeRF/3DGS、信息论主动感知。

**核心问题：** 在有限飞行时间、能量和安全约束下，UAV 如何主动选择视点，使 3DGS 地图更快收敛并服务低空规划任务。

**方法路线：**

- scene representation：incremental 3D Gaussian Splatting；
- information metric：对 Gaussian 参数或渲染 Jacobian 构建 Fisher Information / expected information gain；
- planner：NBV candidate generation + safe corridor / CBF constraint；
- task coupling：重建质量不仅报 PSNR/SSIM，还报 obstacle recall、planning collision rate、inspection coverage；
- baselines：ActiveNeRF、FisherRF、GS-Planner、HGS-Planner、POp-GS、frontier exploration。

**关键参考：**

3DGS 原文给出实时显式 radiance field 表示 [20]；ActiveNeRF 是神经渲染主动感知早期代表 [21]；FisherRF 直接支撑 Fisher information active view selection，并已有 3DGS backend 70 fps 结果 [22]；GS-Planner、HGS-Planner、POp-GS 和 NVF 支撑 2024-2025 的 3DGS/NBV 竞争线 [23-26]。

**创新点建议：**

1. 从 “3DGS NBV” 升级为 “服务 UAV 安全规划的 active perception”。
2. 用 Fisher 信息连接 CRB / reconstruction uncertainty / planning safety。
3. 从视觉指标扩展到交通/机器人任务指标：路径可行率、障碍召回率、应急巡检覆盖率。
4. 在 MatrixCity / AirSim / 自建城市低空 cell 上做跨场景泛化。

### 3.4 Paper D：低空安全关键场景覆盖与加速测试

**建议题目：** Coverage-Guided Accelerated Testing for Safety-Critical Low-Altitude UAV Navigation

**对应已有文章：** Paper F 场景覆盖、危险场景生成、7600 万次探索日志。

**核心问题：** 低空 UAV 避障/规划算法的测试场景空间如何定义、如何度量覆盖、如何高效发现危险但有效的 failure 场景。

**方法路线：**

- scenario grammar：局部 50m x 50m x 50m cell，障碍物组合、动态障碍、风扰、目标点、起终点；
- coverage metric：geometry coverage、semantic coverage、dynamics coverage、risk coverage、failure-mode coverage；
- accelerated testing：从 coverage holes 和 failure likelihood 主动采样；
- invalid filtering：过滤不真实、不安全无效、不可执行任务；
- cross-planner evaluation：A*/RRT*/MPC/ORCA/MAPPO/CBF-shielded planner。

**关键参考：**

Shuo Feng 的 NADE 和 testing scenario library generation 是加速测试与安全关键场景库的核心参考 [27-29]；SafeBench 提供 benchmark 平台与安全评估 protocol 参考 [30]。

**创新点建议：**

1. 从自动驾驶 scenario engineering 迁移到低空 UAV 3D 场景空间。
2. 把 coverage、criticality、feasibility 三个目标同时建模。
3. 用 7600 万次探索日志证明覆盖空间和 failure taxonomy。
4. 让结果能回答：哪些障碍组合最危险，哪些 planner 泛化最差，覆盖度提升是否真的减少未知风险。

### 3.5 Paper E：验证纠错式 UAV 语言规划

**建议题目：** VERA-UAV: Verification-and-Repair Language Planning for Low-Altitude UAV Tasks

**对应已有文章：** Paper E。

**核心问题：** LLM 能把自然语言任务转换成 UAV 可执行任务规格，但容易产生不可执行、语义错配或违反安全约束的计划。需要 typed IR、LTL/STL、验证器和反例反馈闭环。

**方法路线：**

- NL instruction -> typed TaskIR；
- TaskIR -> LTL/STL；
- Spot / RTAMT 验证；
- counterexample / robustness feedback；
- local LLM iterative repair；
- final trajectory verification。

**关键参考：**

Lang2LTL、NL2LTL、LTLCodeGen、ConformalNL2LTL 分别支撑 NL-to-LTL grounding、系统演示、code-generation-style temporal logic generation 和 conformal correctness guarantee [31-34]。

**创新点建议：**

1. 不是单纯 NL2LTL，而是 UAV 轨迹可执行闭环。
2. typed TaskIR 降低语言歧义，提高可解释性。
3. 反例反馈和 STL robustness feedback 让 repair 有具体方向。
4. AAAI/IJCAI 版本聚焦 AI planning / verification；T-ITS 扩展再接低空交通运行场景。

### 3.6 Paper G：低空交通云脑 LLM Agent

**建议题目：** CloudBrain-Agent: Tool-Augmented LLM Agents for Low-Altitude Traffic Operation

**对应已有文章：** Paper G / G1。

**核心问题：** 低空交通云脑不能只是聊天模型，而应是能调用调度器、路径规划器、验证器、仿真器和风险评估器的可验证 agent。

**方法路线：**

- LLM 负责任务理解、工具选择、状态摘要和解释；
- tools 包括 Paper A conflict resolver、Paper B scheduler、Paper C active mapper、Paper D scenario tester、Paper E verifier；
- LowAltitudeIR 作为统一中间表示；
- 技术路线优先普通大模型 + agent + skills + MCP/tool-use，后续再做领域 LoRA/SFT；
- 部署上第一阶段调用 API 形成 benchmark，第二阶段本地 Qwen/DeepSeek 系模型做复现和成本控制。

**关键参考：**

UrbanGPT、UniST、TrafficGPT 说明交通/城市时空任务已经开始向 foundation model 和 agent framework 靠近 [35-37]；DriveLM 虽是自动驾驶，但其 Graph VQA 任务形式可借鉴到低空交通 cloud brain 的多步推理 [38]。

**创新点建议：**

1. 低空交通云脑不是“垂类聊天模型”，而是 tool-augmented verifiable agent。
2. 用统一 IR 把调度、规划、感知、验证、场景测试串起来。
3. 先做 agent benchmark，再决定是否微调垂类模型，降低第一篇风险。
4. 评价指标包括 tool-call accuracy、task success、safety violation、repair success、latency、human auditability。

### 3.7 Paper H：城市低空 ODD 与语义功能区规划

**建议题目：** ODD2Route: Semantic Operational-Design-Domain Modeling for Low-Altitude UAV Route Planning

**这是还可以补写的一篇新方向。**

**核心问题：** 城市整体场景如何映射到局部低空航路规划？不同功能区、建筑密度、道路结构、人群活动、禁飞区和应急设施分布，如何决定低空航路的风险、容量和服务策略？

**方法路线：**

- city-level ODD：OSM road/building/POI/land-use + population/demand proxy；
- local test cell：从 city ODD 采样局部 3D obstacle/traffic scenario；
- route risk model：建筑峡谷、学校医院、交通枢纽、高速路段、禁飞区；
- planning output：risk-aware corridor、altitude layer、emergency landing site、charging/vertiport candidates；
- evaluation：跨城市泛化，比较 naive shortest path、risk-aware A*、multi-objective MILP、learning-based route recommender。

**文献支撑：**

这一篇可以从 Paper B 的 TR-C/UAM 文献 [13-19]、Paper D 的 scenario coverage 文献 [27-30] 和 Paper C 的 3D/digital twin 文献 [20-26] 共同支撑。它的难点不在算法复杂，而在 city-level ODD 到 local scenario / route risk 的定义要可信。

**创新点建议：**

1. 把“城市整体场景”和“局部障碍组合”建立可计算映射。
2. 用 ODD coverage 解释场景覆盖，而不是随机生成场景。
3. 为 TR-C/T-ITS 提供城市低空规划、航路设计和测试场景库之间的桥梁。

### 3.8 Paper I：具身低空智能与 Aerial VLA/VLN

**建议题目：** Embodied Low-Altitude Intelligence: Vision-Language-Action Planning for UAVs in Urban Airspace

**这是新调研后最值得保留的中长期方向。**

当前具身智能主线已经从 “LLM 说话” 走向 “VLM/VLA 直接连接感知、语言和动作”。RT-2 明确提出 vision-language-action model，把视觉、语言和机器人动作放进同一个模型范式 [44]；OpenVLA 和 Octo 说明开源 VLA / generalist robot policy 可以用大规模机器人轨迹预训练，再用少量目标域数据微调 [42,43]。UAV 方向也已经开始出现直接相关工作：SINGER 用 Gaussian Splatting 生成语言嵌入飞行仿真数据，训练 onboard drone VLN policy，并做硬件实验 [39]；FlightGPT 用 SFT + GRPO 做 UAV VLN，在 CityNav 上验证泛化和可解释推理 [40]；UAV-VLN 把自然语言、视觉感知和可行航迹规划接在一起 [41]。

**我们的可写 gap：**

现有 aerial VLN/VLA 多数聚焦 “给一个语言目标，让无人机飞到目标附近”。这还不是低空交通云脑需要的能力。低空场景要求模型同时理解：

- 城市低空 ODD、空域结构、禁飞区和风险区；
- 多 UAV 交通状态、走廊容量、避碰规则；
- 应急任务、巡检任务、物流任务的优先级；
- 视觉地图不完整、定位误差、通信退化和非合作目标；
- 输出必须可验证、可控制、可降级，而不是端到端黑箱动作。

**建议方法：**

```text
multimodal observation
  = UAV RGB/depth/semantic map/3DGS local map
  + low-altitude traffic state
  + natural-language mission
  + city ODD metadata

LLM/VLM/VLA policy
  -> LowAltitudeIR
  -> skill selection
  -> waypoint / velocity / route command
  -> verifier + safety shield
  -> simulator or hardware feedback
```

**推荐先做的版本：**

不要一开始训练端到端 AerialVLA。先做一个 **hybrid embodied agent**：

- 高层用 Qwen/DeepSeek/API 模型做任务理解和工具调用；
- 中层调用 Paper A 的 conflict resolver、Paper B 的 scheduler、Paper C 的 active mapper、Paper E 的 verifier；
- 低层用传统 controller / MPC / CBF shield 保证实时安全；
- 训练数据来自仿真轨迹、专家 planner、失败修复日志和人工标注任务。

**可投目标：**

- ICRA/IROS/T-RO：强调具身导航、硬件闭环、sim-to-real。
- AAAI/IJCAI：强调 agent planning、tool-use、verification feedback。
- T-ITS：强调低空交通运行、应急、冲突消解和系统指标。

### 3.9 Paper J：LowAltitudeGPT 训练与微调路线

**建议题目：** LowAltitudeGPT: Tool-Use and Simulation-Feedback Tuning for Low-Altitude Traffic Intelligence

**核心判断：**

现在不应该先从零训练“低空交通大模型”。这会有三个问题：

1. 数据量不够，难以支撑 foundation model 级别贡献；
2. 评审会问模型贡献是否超过普通大模型 + RAG + tool-use；
3. 训练成本高，但不一定比 agent/verifier/simulator 闭环更有论文价值。

更可行的路线是 **普通大模型 + Agent + Skills/MCP + RAG + verifier + simulator** 先跑通，然后把运行日志沉淀成可训练数据。MCP 本质上是给 LLM 暴露工具和上下文的标准接口，适合把调度器、规划器、验证器、仿真器、数据库和文献库统一接入 [47]。

低空经济大模型综述也把低空系统拆成设施网络、信息网络、航路网络和服务网络，并强调大模型需要和边缘计算、6G/ISAC、可信分布式智能结合 [50]。这说明我们的论文不能只写“训一个聊天模型”，而要写成模型、工具、网络、运行控制和系统评价的闭环。

**模型选择建议：**

| 阶段 | 推荐模型 | 原因 |
|---|---|---|
| 方案探索 / 数据生成 / teacher | 高能力 API 模型 | 先快速生成任务、tool trace、反例解释和评测样本，不把 API 作为最终可复现依赖 |
| 本地可复现实验 | Qwen3-8B / Qwen3-14B / Qwen3-32B | Qwen3 官方支持本地运行、部署、量化和训练流程，中文、工具调用和工程生态较好 [45] |
| 推理/数学/约束解释 | DeepSeek-R1-Distill-Qwen-14B / 32B | DeepSeek-R1 系列强调 RL 激励推理能力，distill 版本可本地部署，并基于 Qwen/Llama 开源模型微调 [46] |
| 多模态低空感知 | Qwen-VL / Qwen3-VL / 其他开源 VLM | 用于图片、视频帧、地图、航迹图、3DGS render 的语义理解 |
| 边缘端小模型 | Qwen3-4B / 8B 量化版、SLM | 用于端侧状态摘要、异常检测、低延迟 fallback |

**训练数据设计：**

| 数据类型 | 来源 | 训练目标 |
|---|---|---|
| NL mission -> LowAltitudeIR | 人工模板 + API teacher + 真实任务改写 | 任务解析和结构化表示 |
| tool-use trace | Paper A/B/C/D/E 工具调用日志 | 学会何时调用调度、规划、验证、仿真 |
| verifier counterexample | Spot/RTAMT/CBF/simulator 反馈 | 学会修复不可执行或危险计划 |
| simulation rollout | SUMO/AirSim/自研低空仿真 | 学会从结果解释系统瓶颈 |
| failure case | 碰撞、LoWC、超时、队列爆炸、能耗不足 | 学会风险诊断和应急降级 |
| human audit data | 人工选择更合理方案 | DPO/偏好优化 |

**训练阶段：**

1. **RAG + prompt baseline**：不微调，只用文献库、法规、系统说明和工具 schema。
2. **LoRA/QLoRA SFT**：训练 NL-to-IR、tool-call、结果解释、反例修复。
3. **DPO/IPO**：用人工偏好或 verifier 打分偏好，优化 “安全、可执行、简洁、可解释”。
4. **GRPO/RL-style tuning**：用仿真奖励训练任务成功率、低 violation、低 latency 和格式合规，FlightGPT 的 SFT + GRPO 路线可作为 UAV VLN 参考 [40]。
5. **distillation**：把 API teacher / 32B 模型能力蒸馏到 8B/4B，用于本地和边缘部署。

**评价指标：**

- task success；
- LowAltitudeIR exact/semantic match；
- tool-call precision/recall；
- executable plan rate；
- safety violation rate；
- repair success rate；
- hallucination rate；
- latency / token cost；
- cross-city / cross-task generalization；
- human audit pass rate。

### 3.10 Paper K：低空云脑推理加速与软硬件协同

**建议题目：** Edge-Cloud Co-Optimized Inference for Low-Altitude Traffic Cloud-Brain Agents

**为什么这条能写：**

如果后续要往软件硬件都做，推理加速不能只是工程优化。它需要被写成 **低空交通系统约束下的实时智能服务问题**：云端有大模型和全局状态，边缘端有低延迟和隐私/通信约束，无人机端有功耗、算力、散热和实时控制限制。General-Purpose Aerial Intelligent Agents 已经给出硬件-软件 co-design 方向的直接信号：14B 模型 onboard 运行约 5-6 tokens/sec，峰值功耗约 220W，并采用慢速 LLM 规划 + 快速反应控制的双向认知架构 [51]。

**系统架构：**

```text
cloud brain
  - full LLM / VLM
  - global scheduler
  - long-horizon planner
  - batch simulation evaluator

edge station / vertiport
  - quantized 8B/14B model
  - local RAG cache
  - route/conflict verifier
  - streaming state summarizer

onboard UAV
  - tiny policy / controller
  - VIO / obstacle avoidance
  - emergency fallback
  - compressed semantic state uplink
```

**加速技术路线：**

- 服务端：vLLM / PagedAttention / continuous batching / prefix cache。PagedAttention 的核心价值是降低 KV cache 浪费，并提高 batch serving 吞吐 [48]。
- NVIDIA GPU 生产部署：TensorRT-LLM，用 TensorRT engines、Python/C++ runtime 和 GPU 优化执行 LLM 推理 [49]。
- 端侧/边缘：AWQ/GPTQ/GGUF INT4/INT8、KV cache 压缩、speculative decoding、small model router。
- 工具调用优化：缓存 tool schema、缓存静态 RAG 检索结果、把高频工具调用编译成 deterministic skill。
- 算子方向：attention kernel、paged KV cache、prefill/decode 分离、batch scheduler、MoE expert routing、vision encoder caching。

**可写论文点：**

1. **系统论文**：低空云脑 agent workload 的 latency/cost/energy profiling。
2. **算法-系统论文**：根据任务风险动态选择 API / cloud 32B / edge 14B / onboard 4B。
3. **算子/推理论文**：针对低空交通多 agent、多工具、长上下文、流式状态更新的 KV cache 与 batching 优化。
4. **硬件协同论文**：Jetson Orin / RTX workstation / cloud GPU 三层部署，评估 tokens/sec、end-to-end latency、energy per decision、safety fallback rate。

**推荐 venue：**

- 偏交通系统：T-ITS / IEEE IoT Journal。
- 偏边缘智能：IEEE TMC / IEEE Internet of Things Journal / ACM TECS。
- 偏机器人系统：IROS / ICRA system paper。
- 偏算子和系统：MLSys / SC workshop / DAC/DATE workshop 起步，不建议一开始直接冲系统顶会主会。

---

## 4. 推荐优先级

| 优先级 | 文章 | 近期动作 | 原因 |
|---|---|---|---|
| P0-Active | Paper B | 冻结 problem formulation、队列模型、实验 benchmark | 最像 TR-C 系统论文，和低空经济/应急最贴合 |
| P0-Active | Paper A | 把 PPO/MAPPO 改写成鲁棒低空冲突消解论文 | 已有算法基础，但需要交通指标和强 baseline |
| P0-Active | Paper C | 收敛到 Fisher + 3DGS + safe planning，不再扩太多 | 算法新意较强，能投机器人/AI/ITS |
| P1-Support | Paper D | 复用 7600 万次探索日志，做 coverage-guided testing | 给 A/C 提供安全关键场景、failure taxonomy 和 benchmark |
| P1-Bridge | Paper G | 先做工具接口和 CloudBrain-Agent benchmark | 把 A/B/C/D/E 串成低空云脑，而不是空泛聊天模型 |
| P2-Embodied | Paper I | 做 aerial VLN/VLA 小规模 pilot：仿真数据、专家轨迹、端到端/混合策略对比 | 这是通向 embodied AGI 的主线，但需要 A/C 的感知与安全工具先稳定 |
| P2-Model | Paper J | 沉淀 LowAltitudeIR、tool trace、verifier feedback，再做 LoRA/SFT/GRPO | 先有数据闭环，再微调垂类模型 |
| P3-System | Paper K | 等 CloudBrain-Agent workload 固定后做 vLLM/TensorRT/量化/端云协同 | 软件硬件方向可写，但要有真实 workload 才像论文 |
| P3-Planning | Paper H | 作为 TR-C/T-ITS 后续扩展 | 需要真实城市数据 pipeline 和 ODD 定义成熟 |

**执行顺序建议：**

1. 不改变当前主战场：A/B/C 继续推进。
2. 先补 Paper D，因为它直接增强 A/C 的实验可信度，也能生成后续模型训练数据。
3. 再做 Paper G，把 A/B/C/D/E 包装成工具化云脑。
4. Paper I/J/K 不急于开大工程；先做小 pilot 和数据 schema。真正开题前必须回答：数据从哪里来、评价指标是什么、能否比普通大模型 + 工具调用更强。

---

## 4.1 文献支撑矩阵

为避免文献堆砌，当前 51 条参考文献按论文方向闭合使用：

| 方向 | 文献组 | 用法 |
|---|---|---|
| 投稿与交通系统定位 | [1,2] | 判断 TR-C / T-ITS 的 framing 差异 |
| Paper A：多智能体冲突消解 | [3-12] | PPO/MAPPO、MAT/FACMAC/HAPPO 与 EGO-Swarm/MADER/RMADER/RACER/PANTHER/GCOPTER baseline |
| Paper B：百架 UAV 调度 | [13-19] | 低空配送资源分配、UAM scheduling、safe learning、truck-drone/UAV-UGV 多模态配送 |
| Paper C：3DGS 主动感知 | [20-26] | 3DGS、ActiveNeRF、FisherRF、GS-Planner、HGS-Planner、POp-GS、NVF |
| Paper D：安全关键场景覆盖 | [27-30] | Shuo Feng 加速测试、scenario library、SafeBench |
| Paper E：语言规划与验证 | [31-34] | Lang2LTL、NL2LTL、LTLCodeGen、ConformalNL2LTL |
| Paper G：低空云脑 Agent | [35-38,47,50,51] | UrbanGPT/UniST/TrafficGPT/DriveLM、MCP、低空经济大模型综述、aerial intelligent agent |
| Paper I：具身低空 / Aerial VLA | [39-44] | SINGER、FlightGPT、UAV-VLN、OpenVLA、Octo、RT-2 |
| Paper J：模型训练与微调 | [40,45,46,47,50] | SFT/GRPO 参考、Qwen3、DeepSeek-R1、MCP/tool-use、低空大模型系统定位 |
| Paper K：推理加速与软硬件协同 | [45,48,49,51] | Qwen3 部署生态、vLLM/PagedAttention、TensorRT-LLM、onboard 14B aerial agent 硬件约束 |

---

## 5. Zotero 整理状态

目标 Zotero collection 名称：

```text
低空规划论文参考
```

当前已完成两层整理：

| 项目 | 状态 |
|---|---|
| Zotero collection | 已存在，collection key 为 `FVHS3SKY`，本地 treeViewID 为 `C17` |
| Zotero 本地选择链接 | `zotero://select/library/collections/FVHS3SKY` |
| 已导入文献 | 51 条 top-level items |
| item type 分布 | `journalArticle` 17 条，`conferencePaper` 11 条，`document/preprint/webpage` 23 条 |
| 本地备份 BibTeX | `zotero/low-altitude-planning-references-20260527.bib`；增量：`zotero/low-altitude-planning-references-update-20260528.bib` |

导入方式采用 Zotero 本地 connector server，而不是直接写 `zotero.sqlite`。具体流程是：

1. 用 `pandoc` 检查 BibTeX 可解析为 CSL JSON。
2. 通过 Zotero 本地 `/connector/import` 导入 `zotero/low-altitude-planning-references-20260527.bib`。
3. 通过 `/connector/updateSession` 把导入 session 的目标 collection 更新为 `C17 / 低空规划论文参考`。
4. 用 Zotero local API 与只读 SQLite 双重验证 collection 中有 51 条 top-level 文献。

后续如果继续补文献，建议仍然先更新本地 BibTeX，再通过同样的 connector import/updateSession 流程导入 Zotero。不要直接修改 SQLite。

---

## 6. 后续执行计划

### 6.1 第 1 周：冻结三篇在做论文

- 明确 Paper A/B/C 是当前 active pipeline，不再把后续论文写成同等优先级。
- Paper A：冻结 conflict scenario、action space、baseline 和交通化指标。
- Paper B：冻结 queue model、Lyapunov objective、synthetic benchmark 和 TR-C framing。
- Paper C：冻结 FIM/3DGS/NBV 的理论接口和 planning-aware metrics。
- 已完成 Zotero collection 初始导入与 2026-05-28 增量导入；下一步补 PDF、摘要备注和每篇文章的优先级标签。

### 6.2 第 2-3 周：补文献矩阵与后续路线查新

- 每篇主力文章至少整理 25 篇高相关文献。
- 每篇文章形成 `related work matrix`：problem、method、data、metric、gap、our angle。
- 对 Paper A/B/C 分别标出 “必须复现 baseline” 和 “只作为 related work” 的论文。
- 对 Paper I/J/K 单独做查新：
  - Paper I：aerial VLN、AerialVLA、SINGER、FlightGPT、OpenVLA、Octo、RT-2；
  - Paper J：Qwen3、DeepSeek-R1、tool-use tuning、MCP、RAG、simulation-feedback training；
  - Paper K：vLLM、TensorRT-LLM、量化、KV cache、edge-cloud deployment。

### 6.3 第 4-8 周：先推进 Paper B/A/C 三条实验线

- Paper B：synthetic UAM queueing benchmark + FCFS/greedy/MILP/backpressure/MARL baseline。
- Paper A：corridor conflict simulation + ORCA/CBF/RMADER/MAPPO baseline。
- Paper C：3DGS NBV pipeline + FisherRF/ActiveNeRF/GS-Planner/POp-GS baseline。
- Paper D：只做轻量 pilot，把 7600 万次探索日志整理成 coverage/failure taxonomy，不抢 A/B/C 资源。

### 6.4 第 9-12 周：构建低空云脑最小闭环

- 把 A/B/C/D/E 暴露成工具接口：scheduler、conflict resolver、active mapper、scenario tester、verifier。
- 定义 `LowAltitudeIR`，统一任务、空域、UAV、资源、风险和工具调用结果。
- 先用 API teacher + 本地 Qwen/DeepSeek 做 CloudBrain-Agent baseline，不急于微调。
- 采集 tool trace、失败修复、仿真 rollout，作为 Paper J 的训练数据。

### 6.5 第 13-20 周：决定投稿与模型路线

- 如果 Paper B 的 queue stability 和百架级结果最稳：先投 TR-C。
- 如果 Paper A 的 conflict safety 和泛化最强：先投 T-ITS/T-RO。
- 如果 Paper C 的 Fisher + 3DGS 理论和视觉结果最强：先投 T-RO/ICRA/IROS。
- 如果 Paper D 的 coverage/failure discovery 数据最好：先投 T-ITS。
- 如果 CloudBrain-Agent 已经能稳定调用 A/B/C/D/E 工具：启动 AAAI/IJCAI 版本。
- 如果已经积累 5k-20k 条高质量 tool trace / verifier feedback / simulation rollout：启动 LowAltitudeGPT LoRA/SFT。
- 如果 agent workload 固定且 latency 成为瓶颈：启动 Paper K 的 vLLM/TensorRT/边缘量化实验。

---

## 7. 参考文献

[1] Elsevier. *Transportation Research Part C: Emerging Technologies: Aims and Scope.* URL: <https://www.sciencedirect.com/journal/transportation-research-part-c-emerging-technologies>

[2] IEEE Intelligent Transportation Systems Society. *IEEE Transactions on Intelligent Transportation Systems: Scope.* URL: <https://ieee-itss.org/pub/t-its/>

[3] Chao Yu, Akash Velu, Eugene Vinitsky, Yu Wang, Alexandre M. Bayen, and Yi Wu. “The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games.” *Advances in Neural Information Processing Systems*, 2022. URL: <https://arxiv.org/abs/2103.01955>

[4] Muning Wen, Jakub Grudzien Kuba, Runji Lin, Weinan Zhang, Ying Wen, Jun Wang, and Yaodong Yang. “Multi-Agent Reinforcement Learning is a Sequence Modeling Problem.” *NeurIPS*, 2022. URL: <https://proceedings.neurips.cc/paper_files/paper/2022/hash/69413f87e5a34897cd010ca698097d0a-Abstract-Conference.html>

[5] Bei Peng, Tabish Rashid, Christian Schroeder de Witt, Pierre-Alexandre Kamienny, Philip Torr, Wendelin Boehmer, and Shimon Whiteson. “FACMAC: Factored Multi-Agent Centralised Policy Gradients.” *NeurIPS*, 2021. URL: <https://proceedings.neurips.cc/paper/2021/hash/65b9eea6e1cc6bb9f0cd2a47751a186f-Abstract.html>

[6] Jakub Grudzien Kuba, Ruiqing Chen, Muning Wen, Ying Wen, Fudan Sun, Jun Wang, and Yaodong Yang. “Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning.” arXiv:2109.11251, 2021. URL: <https://arxiv.org/abs/2109.11251>

[7] Boyu Zhou, Xin Zhou, Jun Zhang, Fei Gao, and Shaojie Shen. “EGO-Swarm: A Fully Autonomous and Decentralized Quadrotor Swarm System in Cluttered Environments.” *ICRA*, 2021. DOI: 10.1109/ICRA48506.2021.9561902. URL: <https://arxiv.org/abs/2011.04183>

[8] Jesus Tordesillas, Brett T. Lopez, and Jonathan P. How. “MADER: Trajectory Planner in Multiagent and Dynamic Environments.” *IEEE Transactions on Robotics*, 38(1):463-476, 2022. URL: <https://arxiv.org/abs/2010.11061>

[9] Kota Kondo, Reinaldo Figueroa, Juan Rached, Jesus Tordesillas, Parker C. Lusk, and Jonathan P. How. “Robust MADER: Decentralized Multiagent Trajectory Planner Robust to Communication Delay in Dynamic Environments.” arXiv:2303.06222, 2023. URL: <https://arxiv.org/abs/2303.06222>

[10] Boyu Zhou, Hao Xu, and Shaojie Shen. “RACER: Rapid Collaborative Exploration With a Decentralized Multi-UAV System.” *IEEE Transactions on Robotics*, 2023. DOI: 10.1109/TRO.2023.3236945. URL: <https://arxiv.org/abs/2209.08533>

[11] Jesus Tordesillas and Jonathan P. How. “PANTHER: Perception-Aware Trajectory Planner in Dynamic Environments.” *IEEE Access*, 10:22662-22677, 2022. DOI: 10.1109/ACCESS.2022.3154037. URL: <https://arxiv.org/abs/2103.06372>

[12] Zhepei Wang, Xin Zhou, Chao Xu, and Fei Gao. “Geometrically Constrained Trajectory Optimization for Multicopters.” *IEEE Transactions on Robotics*, 38(5):3259-3278, 2022. DOI: 10.1109/TRO.2022.3160022. URL: <https://arxiv.org/abs/2103.00190>

[13] Ang Li, Mark Hansen, and Bo Zou. “Traffic Management and Resource Allocation for UAV-Based Parcel Delivery in Low-Altitude Urban Space.” *Transportation Research Part C: Emerging Technologies*, 143:103808, 2022. DOI: 10.1016/j.trc.2022.103808. URL: <https://doi.org/10.1016/j.trc.2022.103808>

[14] Mehdi Bennaceur, Rémi Delmas, and Youssef Hamadi. “Passenger-Centric Urban Air Mobility: Fairness Trade-Offs and Operational Efficiency.” *Transportation Research Part C: Emerging Technologies*, 136:103519, 2022. DOI: 10.1016/j.trc.2021.103519. URL: <https://doi.org/10.1016/j.trc.2021.103519>

[15] Roberto Pinto and Alexandra Lagorio. “Point-to-Point Drone-Based Delivery Network Design with Intermediate Charging Stations.” *Transportation Research Part C: Emerging Technologies*, 135:103506, 2022. DOI: 10.1016/j.trc.2021.103506. URL: <https://doi.org/10.1016/j.trc.2021.103506>

[16] Qinshuang Wei, Gustav Nilsson, and Samuel Coogan. “Capacity-Constrained Urban Air Mobility Scheduling.” arXiv:2107.02900, 2021. URL: <https://arxiv.org/abs/2107.02900>

[17] Surya Murthy, Natasha A. Neogi, and Suda Bharadwaj. “Scheduling for Urban Air Mobility Using Safe Learning.” arXiv:2209.15457, NASA NTRS, 2022. URL: <https://arxiv.org/abs/2209.15457>

[18] Jiahao Xing, Tong Guo, and Lu Tong. “Reliable Truck-Drone Routing with Dynamic Synchronization: A High-Dimensional Network Programming Approach.” *Transportation Research Part C: Emerging Technologies*, 165:104698, 2024. DOI: 10.1016/j.trc.2024.104698. URL: <https://doi.org/10.1016/j.trc.2024.104698>

[19] Bolong Zhou, Wei Zeng, and Hai Yang. “Multi-Trip UAV-UGV Delivery Network Design with Release Times.” *Transportation Research Part C: Emerging Technologies*, 181:105389, 2025. DOI: 10.1016/j.trc.2025.105389. URL: <https://doi.org/10.1016/j.trc.2025.105389>

[20] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. “3D Gaussian Splatting for Real-Time Radiance Field Rendering.” *ACM Transactions on Graphics / SIGGRAPH*, 42(4), 2023. DOI: 10.1145/3592433. URL: <https://arxiv.org/abs/2308.04079>

[21] Xuran Pan, Zihang Lai, Shiji Song, and Gao Huang. “ActiveNeRF: Learning Where to See with Uncertainty Estimation.” *ECCV*, 2022. URL: <https://arxiv.org/abs/2209.08546>

[22] Wen Jiang, Boshu Lei, and Kostas Daniilidis. “FisherRF: Active View Selection and Mapping with Radiance Fields Using Fisher Information.” *ECCV*, 2024. DOI: 10.1007/978-3-031-72624-8_24. URL: <https://eccv.ecva.net/virtual/2024/oral/1226>

[23] Rui Jin, Yuman Gao, Yingjian Wang, Haojian Lu, and Fei Gao. “GS-Planner: A Gaussian-Splatting-Based Planning Framework for Active High-Fidelity Reconstruction.” arXiv:2405.10142, 2024. URL: <https://arxiv.org/abs/2405.10142>

[24] Zijun Xu, Rui Jin, Ke Wu, Yi Zhao, Zhiwei Zhang, Jieru Zhao, Fei Gao, Zhongxue Gan, and Wenchao Ding. “HGS-Planner: Hierarchical Planning Framework for Active Scene Reconstruction Using 3D Gaussian Splatting.” arXiv:2409.17624, 2024. URL: <https://arxiv.org/abs/2409.17624>

[25] Joey Wilson, Marcelino Almeida, Sachit Mahajan, Martin Labrie, Maani Ghaffari, Omid Ghasemalizadeh, Min Sun, Cheng-Hao Kuo, and Arnab Sen. “POp-GS: Next Best View in 3D-Gaussian Splatting with P-Optimality.” *CVPR*, 2025. URL: <https://cvpr.thecvf.com/virtual/2025/poster/34708>

[26] Shangjie Xue, Jesse Dill, Pranay Mathur, Frank Dellaert, Panagiotis Tsiotras, and Danfei Xu. “Neural Visibility Field for Uncertainty-Driven Active Mapping.” *CVPR*, 2024. URL: <https://arxiv.org/abs/2406.06948>

[27] Shuo Feng, Xintao Yan, Haowei Sun, Yiheng Feng, and Henry X. Liu. “Intelligent Driving Intelligence Test for Autonomous Vehicles with Naturalistic and Adversarial Environment.” *Nature Communications*, 12:748, 2021. DOI: 10.1038/s41467-021-21007-8. URL: <https://www.nature.com/articles/s41467-021-21007-8>

[28] Shuo Feng, Yiheng Feng, Chao Yu, Yi Zhang, and Henry X. Liu. “Testing Scenario Library Generation for Connected and Automated Vehicles, Part I: Methodology.” *IEEE Transactions on Intelligent Transportation Systems*, 2021. DOI: 10.1109/TITS.2020.2972211. URL: <https://doi.org/10.1109/TITS.2020.2972211>

[29] Shuo Feng, Yiheng Feng, Haowei Sun, Yi Zhang, and Henry X. Liu. “Testing Scenario Library Generation for Connected and Automated Vehicles, Part II: Case Studies.” *IEEE Transactions on Intelligent Transportation Systems*, 2021. DOI: 10.1109/TITS.2020.2988309. URL: <https://doi.org/10.1109/TITS.2020.2988309>

[30] Chejian Xu, Wenhao Ding, Baiming Li, Xinqing Chen, Ding Zhao, Bo Li, Jiajun Liu, and Hang Zhao. “SafeBench: A Benchmarking Platform for Safety Evaluation of Autonomous Vehicles.” *NeurIPS Datasets and Benchmarks*, 2022. URL: <https://proceedings.neurips.cc/paper_files/paper/2022/hash/a48ad12d588c597f4725a8b84af647b5-Abstract-Datasets_and_Benchmarks.html>

[31] Jason Liu, Ankit Shah, Eric Rosen, and Stefanie Tellex. “Lang2LTL: Translating Natural Language Commands to Temporal Robot Task Specification.” *PMLR / CoRL*, 229, 2023. URL: <https://proceedings.mlr.press/v229/liu23d.html>

[32] Francesco Fuggitti and Tathagata Chakraborti. “NL2LTL: A Python Package for Converting Natural Language Instructions to Linear Temporal Logic Formulas.” *AAAI Demonstration*, 37(13):16428-16430, 2023. DOI: 10.1609/aaai.v37i13.27068. URL: <https://ojs.aaai.org/index.php/AAAI/article/view/27068>

[33] Behrad Rabiei and Mahesh A. Kumar. “LTLCodeGen: Code Generation of Syntactically Correct Temporal Logic for Robot Task Planning.” arXiv:2503.07902, 2025. URL: <https://arxiv.org/abs/2503.07902>

[34] Jun Wang, David Smith Sundarsingh, Jyotirmoy V. Deshmukh, and Yiannis Kantaros. “ConformalNL2LTL: Translating Natural Language Instructions into Temporal Logic Formulas with Conformal Correctness Guarantees.” arXiv:2504.21022, 2025. URL: <https://arxiv.org/abs/2504.21022>

[35] Zhonghang Li, Lianghao Xia, Jiabin Tang, Yong Xu, Lei Shi, Long Xia, Dawei Yin, and Chao Huang. “UrbanGPT: Spatio-Temporal Large Language Models.” arXiv:2403.00813, 2024. URL: <https://arxiv.org/abs/2403.00813>

[36] Yuan Yuan, Jingtao Ding, Jie Feng, Depeng Jin, and Yong Li. “UniST: A Prompt-Empowered Universal Model for Urban Spatio-Temporal Prediction.” *KDD*, 2024. DOI: 10.1145/3637528.3671662. URL: <https://dblp.org/rec/conf/kdd/0032D0J024>

[37] Jinhui Ouyang, Yijie Zhu, Xiang Yuan, and Di Wu. “TrafficGPT: Towards Multi-Scale Traffic Analysis and Generation with Spatial-Temporal Agent Framework.” arXiv:2405.05985, 2024. URL: <https://arxiv.org/abs/2405.05985>

[38] Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Jens Beisswenger, Ping Luo, Andreas Geiger, and Hongyang Li. “DriveLM: Driving with Graph Visual Question Answering.” *ECCV*, 2024. URL: <https://github.com/OpenDriveLab/DriveLM>

[39] Maximilian Adang, JunEn Low, Ola Shorinwa, and Mac Schwager. “SINGER: An Onboard Generalist Vision-Language Navigation Policy for Drones.” arXiv:2509.18610, 2025. URL: <https://arxiv.org/abs/2509.18610>

[40] Hengxing Cai, Jinhan Dong, Jingjun Tan, Jingcheng Deng, Sihang Li, Zhifeng Gao, Haidong Wang, Zicheng Su, Agachai Sumalee, and Renxin Zhong. “FlightGPT: Towards Generalizable and Interpretable UAV Vision-and-Language Navigation with Vision-Language Models.” *EMNLP*, 2025. DOI: 10.18653/v1/2025.emnlp-main.338. URL: <https://aclanthology.org/2025.emnlp-main.338/>

[41] Pranav Saxena, Nishant Raghuvanshi, and Neena Goveas. “UAV-VLN: End-to-End Vision Language guided Navigation for UAVs.” arXiv:2504.21432, 2025. URL: <https://arxiv.org/abs/2504.21432>

[42] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. “OpenVLA: An Open-Source Vision-Language-Action Model.” arXiv:2406.09246, 2024. URL: <https://arxiv.org/abs/2406.09246>

[43] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, Jianlan Luo, You Liang Tan, Lawrence Yunliang Chen, Pannag Sanketi, Quan Vuong, Ted Xiao, Dorsa Sadigh, Chelsea Finn, and Sergey Levine. “Octo: An Open-Source Generalist Robot Policy.” arXiv:2405.12213, 2024. URL: <https://arxiv.org/abs/2405.12213>

[44] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. “RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control.” arXiv:2307.15818, 2023. URL: <https://arxiv.org/abs/2307.15818>

[45] Qwen Team. “Qwen3 Technical Report.” arXiv:2505.09388, 2025; QwenLM/Qwen3 official repository. URL: <https://arxiv.org/abs/2505.09388>; <https://github.com/QwenLM/Qwen3>

[46] DeepSeek-AI. “DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.” arXiv:2501.12948, 2025. URL: <https://arxiv.org/abs/2501.12948>; model card: <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B>

[47] OpenAI. “Model Context Protocol (MCP): OpenAI Agents SDK.” Official documentation, 2026. URL: <https://openai.github.io/openai-agents-js/guides/mcp/>

[48] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. “Efficient Memory Management for Large Language Model Serving with PagedAttention.” arXiv:2309.06180, 2023. URL: <https://arxiv.org/abs/2309.06180>

[49] NVIDIA. “NVIDIA TensorRT-LLM.” Official documentation, 2026. URL: <https://docs.nvidia.com/tensorrt-llm/index.html>

[50] Jinpeng Hu, Wei Wang, Yuxiao Liu, and Jing Zhang. “Large Model in Low-Altitude Economy: Applications and Challenges.” *Big Data and Cognitive Computing*, 10(1):33, 2026. DOI: 10.3390/bdcc10010033. URL: <https://www.mdpi.com/2504-2289/10/1/33>

[51] Ji Zhao and Xiao Lin. “General-Purpose Aerial Intelligent Agents Empowered by Large Language Models.” arXiv:2503.08302, 2025. URL: <https://arxiv.org/abs/2503.08302>
