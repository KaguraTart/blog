---
title: "Paper B 规划 v1：面向 TR-C 的百架 UAV 三层分层调度"
description: "调研 Paper B 是否更适合 TR Part C，并规划背景、相关方法、问题定义、算法路线、实验数据、预期结论、创新点与推进计划。"
pubDate: 2026-05-19
tags: ["Paper B", "TR-C", "T-ITS", "UAV", "UAM", "分层调度", "排队论", "Lyapunov", "多模态运输"]
category: Tech
---

# Paper B 规划 v1：面向 TR-C 的百架 UAV 三层分层调度

> 结论先定：**Paper B 更适合主投 Transportation Research Part C: Emerging Technologies，IEEE T-ITS 作为备选或改投方向。**  
> 核心原因不是 TR-C “更好发”，而是 Paper B 的问题本质是 transportation system operation：在动态需求、有限 vertiport / charging / corridor 容量、多模态转运约束下，让百架级 UAV fleet 稳定、高效、安全地服务城市物流/应急任务。

---

## 1. 背景与投稿判断

Paper B 关注的问题可以概括为：

> 在城市低空经济场景中，如何调度 100 架级 UAV fleet，使其在动态订单、有限起降点、有限充电资源、低空走廊容量约束和地面运输协同条件下，长期保持任务队列稳定，并最小化延误、能耗、空域拥堵和运营成本？

这不是单机路径规划，也不是单纯多智能体避碰。真正的研究对象是 **transportation service system**：

- 需求侧：订单随机到达，存在 deadline、优先级、起终点和货物/应急类型差异。
- 供给侧：UAV 电量、载重、当前位置、维护状态和可用空域随时间变化。
- 基础设施侧：vertiport、charging pad、低空 corridor、交接点和地面车辆容量有限。
- 系统侧：需要同时优化 throughput、delay、queue backlog、resource utilization、energy 和 safety。

因此，主投 TR-C 更合理。TR-C 官方 scope 明确强调 emerging technologies 对 transportation system planning、design、operation、control 和 maintenance 的影响，并说明期刊的 intellectual core 在 transportation side，而不是单项技术本身；同时欢迎 operations research、control systems、complex networks、computer science 和 AI 的融合方法，并特别关注 multimodal / intermodal transportation、on-demand transport、ITS、logistics、aviation、resource management 和开放数据集 [1]。这些关键词几乎正好覆盖 Paper B。

T-ITS 也可以作为备选。T-ITS scope 覆盖现代交通系统中的 sensing、communications、controls、planning、design、implementation、AI、formal methods、multi-agent systems 和 multimodal transportation [2]。但 T-ITS 更容易要求“智能交通系统技术实现”的味道，例如通信、感知、控制、部署架构或智能系统闭环。如果 Paper B 最终强调 Lyapunov-regulated online scheduling、GNN/MARL 控制和实时系统实现，可以转向 T-ITS；如果强调运力、队列稳定、基础设施瓶颈和多模态物流系统价值，则更应投 TR-C。

**当前建议：TR-C first, T-ITS backup。**

---

## 2. 当前方法谱系

Paper B 需要把相关方法放进 transportation engineering 的谱系中，而不是只沿着 UAV/MARL 讲。

| 方法线 | 代表方法 | 对 Paper B 的启发 | 局限 |
|--------|----------|-------------------|------|
| 传统 OR | MILP、time-space network、network design、ALNS、rolling horizon | 适合表达容量、时间窗、路径、同步和基础设施约束 | 大规模在线调度难以实时求解 |
| UAM / UTM | vertiport scheduling、capacity-constrained scheduling、conflict detection and resolution | 提供起降容量、空域冲突和 corridor 管理视角 | 多数是单层、单模式或中小规模 |
| 多模态物流 | truck-drone、UAV-UGV、ground-air transfer、ridesharing-UAM | 证明 UAV 必须嵌入城市交通系统，而不是孤立飞行 | 多为离线 routing / network design，缺少在线 queue stability |
| 学习式调度 | MARL、GNN、safe learning、demand prediction | 可扩展到百架级，适合动态需求和高维状态 | 缺少可解释稳定性保证，审稿人会质疑安全性 |
| 排队论与 Lyapunov | open queueing network、backpressure、drift-plus-penalty | 能证明 backlog stability 和 cost-delay tradeoff | 需要与实际 UAV 能量、容量、路径约束结合 |

已有 TR-C 论文已经覆盖了很多“单点能力”：低空 UAV parcel traffic management 与资源分配 [3]，UAM passenger-centric fairness 与运营效率 [4]，中继充电站网络设计 [5]，truck-drone 可靠同步 routing [6]，UAV-UGV multi-trip delivery network design [7]，以及 UAM ridesharing 动态调度 [8]。Paper B 的机会在于把这些能力收敛成一个 **百架级在线分层调度系统**。

---

## 3. 现在的论文与可引用文献

### 3.1 Venue 与 framing 文献

| 编号 | 文献/来源 | 核心信息 | 对 Paper B 的定位作用 |
|------|-----------|----------|------------------------|
| [1] | TR-C official aims & scope | transportation-side intellectual core；关注 operation、control、multimodal、logistics、aviation、open datasets | 支持主投 TR-C |
| [2] | IEEE T-ITS official scope | ITS sensing、communications、controls、planning、AI、multi-agent systems | 支持 T-ITS 备选 |
| [15] | Machine Learning for UAV-Aided ITS, T-ITS 2024 | UAV 可服务 ITS 的 traffic monitoring、emergency response、infrastructure inspection | 支持 T-ITS 备选 framing |
| [18] | 4D trajectory planning for UAV teams, T-ITS 2024 | UAV teams 在 ITS/T-ITS 中已有发表先例 | 说明 T-ITS 可投，但需更偏智能系统 |

### 3.2 TR-C / UAM / UAV 调度论文

| 编号 | 文献 | 方法 | 对 Paper B 的启发 |
|------|------|------|-------------------|
| [3] | Li, Hansen & Zou, TR-C 2022 | 低空 UAV parcel delivery 的 traffic management、path conflict、resource allocation、VCG mechanism | 直接说明低空空域资源分配是 TR-C 合法题目 |
| [4] | Bennaceur, Delmas & Hamadi, TR-C 2022 | passenger-centric UAM，公平性与运营效率 | 支持服务质量、公平性、乘客/货物指标 |
| [5] | Pinto & Lagorio, TR-C 2022 | drone network design with intermediate charging stations | 支持充电基础设施进入 formulation |
| [6] | Xing, Guo & Tong, TR-C 2024 | truck-drone reliable routing with dynamic synchronization | 支持多模态同步与不确定 travel time |
| [7] | Zhou, Zeng & Yang, TR-C 2025 | UAV-UGV multi-trip delivery network design with release times | 支持 UAV + UGV 多级配送网络 |
| [8] | Li, Zhang, Xiao & Li, TR-C 2025 | UAM ridesharing dynamic scheduling and multimodal mobility-on-demand | 支持 air-ground multimodal service architecture |
| [9] | Wei, Nilsson & Coogan, arXiv 2021 | capacity-constrained UAM scheduling with uncertain travel time and limited landing capacity | 支持 capacity-constrained scheduling formulation |
| [10] | Murthy et al., EPTCS/arXiv 2022 | safe learning for UAM scheduling with hard/soft deadlines | 支持 safe online scheduling baseline |
| [11] | NASA vertiport FCFS scheduling, 2020 | vertiport capacity and throughput under FCFS | 作为 FCFS 和 queueing capacity baseline |
| [16] | Liu, Liu & Huang, arXiv 2024 | real-time UAV delivery scheduling management middleware | 支持真实执行系统与 UAV/AGV/ground staff 协同 |

### 3.3 排队论、Lyapunov 与系统稳定性基础

| 编号 | 文献 | 核心贡献 | 对 Paper B 的作用 |
|------|------|----------|-------------------|
| [12] | Grippa et al., Autonomous Robots 2019 | drone delivery job assignment and dimensioning；使用 queuing theory 分析稳定性与 workload policy | 支持 UAV delivery queueing model |
| [13] | Neely, 2010 | stochastic network optimization and Lyapunov drift-plus-penalty | 支撑 $O(1/V)$ cost 与 $O(V)$ backlog tradeoff |
| [14] | Tassiulas & Ephremides, IEEE TAC 1992 | constrained queueing systems and throughput-optimal scheduling | 支撑 backpressure / stability 理论传统 |
| [17] | Vertiport placement with vehicle sizing and queuing, 2023 | open-network queueing for vertiport infrastructure and service rates | 支持 vertiport queue / charging queue 建模 |

**文献判断：** 现有文献已经充分证明“UAV/UAM + transportation systems + scheduling + multimodal + queueing”是 TR-C/T-ITS 的正当话题。Paper B 不能再写成“百架无人机 MARL 调度算法”，而要写成“低空交通系统运营控制与稳定性保障”。

---

## 4. 现在的问题

现有工作留下的空白主要有四个。

1. **缺少百架级在线三层调度闭环。**  
   TR-C 已有低空空域资源分配、truck-drone routing、UAM ridesharing 和 UAV-UGV network design [3,6,7,8]，但这些工作大多处理 routing、network design、resource allocation 或 ridesharing 的某一层，缺少从宏观 demand queue 到中观 corridor/vertiport resource 再到微观 UAV energy/safety 的统一在线框架。

2. **缺少 queue stability / service guarantee。**  
   学习式调度和启发式方法可以提升经验性能，但如果无法说明高峰需求下队列是否稳定，TR-C 审稿人会质疑系统可运营性。Neely 的 Lyapunov optimization 和 Tassiulas-Ephremides 的 constrained queueing scheduling 提供理论基础 [13,14]，但还没有被系统性用于百架级低空 UAV 多模态调度。

3. **缺少多模态转运视角下的 UAV fleet control。**  
   Truck-drone、UAV-UGV、ridesharing-UAM 论文已经证明 ground-air integration 是主流方向 [6,7,8]，但现有研究多是离线路线/网络设计。Paper B 应该把 ground mode 当作 online fallback 和 capacity buffer：当低空 corridor 或 charging queue 拥堵时，任务可转交 UGV/truck/ground courier。

4. **缺少实验 benchmark。**  
   TR-C scope 特别强调 open science 和 large-scale datasets [1]。如果 Paper B 只做内部仿真，不释放 synthetic benchmark schema、OD demand、corridor capacity 和 reproducible seeds，会削弱说服力。

---

## 5. 我们的方法：H-LyraUAV

方法名暂定：

**H-LyraUAV: Hierarchical Lyapunov-Regulated UAV Scheduling for Multimodal Urban Logistics**

其中 H 表示 hierarchical，Lyra 表示 Lyapunov-regulated routing and assignment。

### 5.1 三层架构

```text
Dynamic urban logistics / emergency demand
        ↓
Macro layer: regional demand queues + fleet repositioning
        ↓
Meso layer: corridor / vertiport / charging resource scheduling
        ↓
Micro layer: UAV energy, safety separation, local conflict avoidance
        ↓
Multimodal execution: UAV-only / ground-only / UAV-ground mixed mode
```

| 层级 | 时间尺度 | 决策 | 核心状态 | 输出 |
|------|----------|------|----------|------|
| 宏观层 | 1-5 min | 任务分区、UAV repositioning、mode split | 区域需求队列、电量分布、OD demand forecast | 区域级 dispatch target |
| 中观层 | 5-30 s | vertiport slot、corridor route、charging slot | 起降排队、走廊拥堵、charging queue | 可执行 schedule |
| 微观层 | 0.1-5 s | 速度、高度、局部避让、应急返航 | 近邻 UAV、障碍、剩余电量 | 安全轨迹修正 |

### 5.2 核心机制

H-LyraUAV 的关键不是“用一个大模型端到端调度”，而是将学习模块限制在预测和打分层，把稳定性建立在 Lyapunov queue control 上：

- **Queueing model**：需求、vertiport、charging 和 corridor 都用真实或虚拟队列表示。
- **Lyapunov drift-plus-penalty**：每个时间窗选择 assignment / mode / route / charging，使 queue drift 与运营成本的加权和最小。
- **Learning-assisted prediction**：GNN / temporal model 预测未来 OD demand、服务时间、corridor risk 和 ground travel time，但不作为稳定性证明来源。
- **Multimodal fallback**：当 UAV-only 导致 queue explosion 或 deadline risk 上升时，系统自动启用 UGV/truck/ground courier 或 mixed mode。

---

## 6. Problem Formulation

### 6.1 集合与状态

令 UAV 集合为 $\mathcal{U}$，动态任务集合为 $\mathcal{R}(t)$，vertiport 集合为 $\mathcal{V}$，低空 corridor 集合为 $\mathcal{E}$，地面运输模式集合为 $\mathcal{G}$。

每架 UAV $u\in\mathcal{U}$ 在时间 $t$ 的状态为：

$$
s_u(t)=(l_u(t), b_u(t), a_u(t), \kappa_u(t)),
$$

其中 $l_u(t)$ 是位置，$b_u(t)$ 是电量，$a_u(t)$ 是可用状态，$\kappa_u(t)$ 是载重/任务能力。

每个任务 $r\in\mathcal{R}(t)$ 包含：

$$
r=(o_r,d_r,\omega_r,\delta_r,\pi_r,\eta_r),
$$

其中 $o_r,d_r$ 是起终点，$\omega_r$ 是货物/乘客/应急类型，$\delta_r$ 是 deadline，$\pi_r$ 是优先级，$\eta_r$ 是可接受的运输模式集合。

### 6.2 队列定义

系统维护以下真实和虚拟队列：

| 队列 | 含义 |
|------|------|
| $Q_i(t)$ | 区域 $i$ 的未服务需求队列 |
| $B_v(t)$ | vertiport $v$ 的起降/等待队列 |
| $C_v(t)$ | vertiport $v$ 的 charging queue |
| $Z_e(t)$ | corridor $e$ 的拥堵/安全间隔 virtual queue |
| $D_i(t)$ | 区域 $i$ 的 deadline violation virtual queue |

例如区域需求队列可写为：

$$
Q_i(t+1)=\max[Q_i(t)-\mu_i(t),0]+A_i(t),
$$

其中 $A_i(t)$ 是新到达需求，$\mu_i(t)$ 是时间窗内完成服务的需求数。

### 6.3 决策变量

每个调度周期需要决定：

| 决策 | 符号 | 含义 |
|------|------|------|
| assignment | $x_{u,r}(t)$ | UAV $u$ 是否服务任务 $r$ |
| mode choice | $m_r(t)$ | UAV-only、ground-only 或 mixed mode |
| departure time | $s_u(t)$ | 起飞/发车/转运时间 |
| route / corridor | $p_u(t)$ | 选择低空 corridor 或地面路径 |
| charging decision | $c_u(t)$ | 是否充电、在哪个 vertiport 充电 |

### 6.4 优化目标

长期目标是最小化平均系统成本：

$$
\min_{\pi}
\limsup_{T\to\infty}
\frac{1}{T}\sum_{t=0}^{T-1}
\mathbb{E}\left[
\alpha_1 W(t)+
\alpha_2 E(t)+
\alpha_3 O(t)+
\alpha_4 S(t)+
\alpha_5 M(t)
\right],
$$

其中 $W(t)$ 是延误，$E(t)$ 是能耗，$O(t)$ 是运营成本，$S(t)$ 是安全/拥堵惩罚，$M(t)$ 是多模态转运惩罚。

### 6.5 约束

约束包括：

- queue stability：所有真实队列和关键 virtual queue 必须强稳定。
- battery：$b_u(t)$ 不低于安全返航阈值。
- payload：任务货物重量不能超过 UAV 或地面载具能力。
- time window：高优先级任务必须满足 deadline 或进入 deadline virtual queue。
- vertiport capacity：每个 vertiport 的 pad / parking / charging capacity 有上限。
- corridor separation：同一 corridor 内 UAV 的时间间隔和空间间隔满足安全要求。
- multimodal transfer feasibility：UAV 与 UGV/truck/ground courier 的交接时间、地点和容量可行。

### 6.6 理论目标

定义 Lyapunov function：

$$
L(\Theta(t)) =
\frac{1}{2}\left(
\sum_i Q_i(t)^2+
\sum_v B_v(t)^2+
\sum_v C_v(t)^2+
\sum_e Z_e(t)^2+
\sum_i D_i(t)^2
\right).
$$

每个时间窗求解 drift-plus-penalty：

$$
\Delta(\Theta(t)) + V\cdot \mathbb{E}[Cost(t)\mid \Theta(t)].
$$

预期理论结论：

- 若到达率位于系统 capacity region 的内部，H-LyraUAV 可保持队列稳定。
- 与最优 stationary randomized policy 相比，长期平均 cost 达到 $O(1/V)$ 近似。
- 平均 backlog 为 $O(V)$，形成可解释的 cost-delay tradeoff [13,14]。

---

## 7. 实验数据来源

### 7.1 主实验：程序生成 benchmark

主实验不依赖真实 UAV 飞行数据，而是构建可复现 synthetic UAM queueing benchmark：

- 城市地图：`50x50` 到 `200x200` grid，包含 building、no-fly zones、corridors、vertiports、charging pads。
- 需求流：Poisson / non-homogeneous Poisson / bursty demand，支持 morning peak、evening peak、shock demand。
- 任务类型：parcel delivery、medical delivery、inspection、emergency supply。
- UAV fleet：20 / 50 / 100 / 200 架，异构电池、载重、速度、充电时间。
- 基础设施：5 / 10 / 20 vertiports，不同 pad、parking、charging capacity。
- 多模态模式：UAV-only、ground-only、UAV-ground mixed mode。

### 7.2 真实增强数据

为增强 TR-C 说服力，实验可以用公开交通数据做 demand proxy 和 ground mode travel time：

| 数据源 | 用途 |
|--------|------|
| OpenStreetMap | 路网、POI、建筑密度、候选 vertiport / transfer point |
| NYC TLC Taxi Trip Data | OD demand proxy、时段 demand profile |
| Chicago Taxi Trips / Divvy / public mobility data | 跨城市泛化需求 proxy |
| SUMO | 地面车辆 travel time、拥堵、ground fallback 代价 |
| AirSim 或轻量 UAV simulator | 微观安全、飞行时间和能耗补充验证 |

AAAI 这类会议可以只做 synthetic benchmark；但 TR-C 需要 case study 质感，所以建议至少做一个城市案例：San Francisco、New York 或 Chicago。Li et al. 的低空 UAV parcel traffic management 使用 San Francisco case study [3]，可以作为对齐对象。

---

## 8. 实验设计与对比

### 8.1 Baselines

| Baseline | 描述 | 目的 |
|----------|------|------|
| FCFS vertiport scheduling | 按到达顺序分配起降资源 [11] | 传统运营基线 |
| Greedy nearest UAV | 最近可用 UAV 接最近任务 | 简单在线 dispatch |
| MILP rolling horizon | 小规模滚动优化 | 小规模 upper bound |
| ALNS / heuristic dispatch | 参考 TR-C 多模态 routing 文献 [7,8] | 强 OR heuristic |
| Queue-only backpressure | 只基于队列差调度 | 理论稳定 baseline |
| MARL / GNN dispatch | 学习式分配，无 Lyapunov virtual queues | 学习式 baseline |
| H-LyraUAV full | 三层分层 + Lyapunov + learning prediction + multimodal fallback | 主方法 |

### 8.2 Metrics

| 指标 | 含义 |
|------|------|
| Average delay | 平均任务完成延误 |
| 95th percentile delay | 长尾服务质量 |
| Deadline violation rate | 超时任务比例 |
| Throughput | 单位时间完成任务数 |
| Queue backlog | 需求、vertiport、charging、corridor 队列长度 |
| Queue stability | backlog 是否随时间有界 |
| Vertiport utilization | 起降/停靠资源利用率 |
| Charging utilization | 充电资源利用率 |
| Airspace conflict rate | corridor 安全间隔冲突比例 |
| Energy per delivery | 每单能耗 |
| Ground-UAV transfer success | 多模态交接成功率 |
| Runtime | 单步调度时间 |

### 8.3 Ablation

| 消融 | 目的 |
|------|------|
| no Lyapunov virtual queues | 验证稳定性组件贡献 |
| no multimodal fallback | 验证地面模式作为 capacity buffer 的价值 |
| no hierarchical decomposition | 验证三层结构的可扩展性 |
| no demand prediction | 验证学习预测模块贡献 |
| no charging queue modeling | 验证充电瓶颈是否必须显式建模 |
| 20 / 50 / 100 / 200 UAV scaling | 验证百架级扩展性 |

### 8.4 场景设计

至少跑四类 demand scenarios：

1. **Low demand**：系统轻载，验证 H-LyraUAV 不牺牲效率。
2. **Peak demand**：需求接近 capacity region，验证 queue stability。
3. **Shock demand**：突发应急订单，验证 deadline virtual queue 和 multimodal fallback。
4. **Infrastructure bottleneck**：charging pads 或 vertiport pads 故意减少，验证资源瓶颈识别能力。

---

## 9. 预期成功与创新点

### 9.1 预期成功

本节为预注册预期，不写成真实实验结果。

1. **高峰需求下保持 queue stability。**  
   预期 H-LyraUAV 在 peak demand 中保持 demand queue、vertiport queue 和 charging queue 有界，而 greedy / MARL-only 更容易出现 backlog accumulation。

2. **降低延误与 deadline violation。**  
   相比 FCFS 和 greedy，预期 H-LyraUAV 降低 average delay、95th percentile delay 和 deadline violation rate。

3. **提升实时性和可扩展性。**  
   相比 MILP rolling horizon，预期 H-LyraUAV 在 100 / 200 UAV 场景下能保持秒级或亚秒级在线决策。

4. **保留理论解释。**  
   相比 MARL/GNN-only，H-LyraUAV 的优势不只是经验分数，而是可以解释 cost-delay tradeoff 与稳定性边界。

5. **展示多模态 fallback 的系统价值。**  
   预期在 charging bottleneck 或 corridor congestion 场景中，UAV-ground mixed mode 降低 missed deadlines 和 queue backlog。

### 9.2 创新点

1. **TR-C framing 的低空交通系统调度论文。**  
   Paper B 不把 UAV 当作孤立机器人，而把百架 UAV fleet 当作城市交通服务系统的一部分。

2. **百架级三层 queue-stable multimodal scheduling 框架。**  
   将宏观任务队列、中观基础设施资源和微观安全/能量约束统一在一个在线框架中。

3. **Lyapunov-regulated learning-assisted dispatch。**  
   学习模块用于预测需求和代价，Lyapunov 模块提供稳定性与 cost-delay tradeoff。

4. **多模态运输容量缓冲机制。**  
   将 UGV/truck/ground courier 作为 airspace / charging bottleneck 下的 fallback，而不是附属 baseline。

5. **开放 synthetic UAM queueing benchmark。**  
   对齐 TR-C 对开放数据、可复现 benchmark 和 transferability 的偏好 [1]。

---

## 10. 参考文献

[1] Elsevier. “Transportation Research Part C: Emerging Technologies: Aims & Scope.” URL: <https://www.sciencedirect.com/journal/transportation-research-part-c-emerging-technologies>

[2] IEEE Intelligent Transportation Systems Society. “IEEE Transactions on Intelligent Transportation Systems (T-ITS): Scope.” URL: <https://ieee-itss.org/pub/t-its/>

[3] Ang Li, Mark Hansen, and Bo Zou. “Traffic management and resource allocation for UAV-based parcel delivery in low-altitude urban space.” *Transportation Research Part C: Emerging Technologies*, 143:103808, 2022. DOI: 10.1016/j.trc.2022.103808. URL: <https://www.sciencedirect.com/science/article/pii/S0968090X22002339>

[4] Mehdi Bennaceur, Rémi Delmas, and Youssef Hamadi. “Passenger-centric Urban Air Mobility: Fairness trade-offs and operational efficiency.” *Transportation Research Part C: Emerging Technologies*, 136:103519, 2022. DOI: 10.1016/j.trc.2021.103519. URL: <https://www.sciencedirect.com/science/article/pii/S0968090X21005015>

[5] Roberto Pinto and Alexandra Lagorio. “Point-to-point drone-based delivery network design with intermediate charging stations.” *Transportation Research Part C: Emerging Technologies*, 135:103506, 2022. DOI: 10.1016/j.trc.2021.103506. URL: <https://doi.org/10.1016/j.trc.2021.103506>

[6] Jiahao Xing, Tong Guo, and Lu (Carol) Tong. “Reliable truck-drone routing with dynamic synchronization: A high-dimensional network programming approach.” *Transportation Research Part C: Emerging Technologies*, 165:104698, 2024. DOI: 10.1016/j.trc.2024.104698. URL: <https://www.sciencedirect.com/science/article/pii/S0968090X24002195>

[7] Bolong Zhou, Wenjia Zeng, and Hai Yang. “Multi-trip UAV-UGV delivery network design with release times.” *Transportation Research Part C: Emerging Technologies*, 181:105389, 2025. DOI: 10.1016/j.trc.2025.105389. URL: <https://doi.org/10.1016/j.trc.2025.105389>

[8] Shanghan Li, Tengfei Zhang, Yiyong Xiao, and Daqing Li. “On-demand ridesharing based on dynamic scheduling in urban air mobility.” *Transportation Research Part C: Emerging Technologies*, 175:105111, 2025. DOI: 10.1016/j.trc.2025.105111. URL: <https://www.sciencedirect.com/science/article/pii/S0968090X25001159>

[9] Qinshuang Wei, Gustav Nilsson, and Samuel Coogan. “Capacity-Constrained Urban Air Mobility Scheduling.” arXiv:2107.02900, 2021. URL: <https://arxiv.org/abs/2107.02900>

[10] Surya Murthy, Natasha A. Neogi, and Suda Bharadwaj. “Scheduling for Urban Air Mobility using Safe Learning.” *Electronic Proceedings in Theoretical Computer Science*, 371:86-102, 2022; arXiv:2209.15457. DOI: 10.4204/EPTCS.371.7. URL: <https://arxiv.org/abs/2209.15457>

[11] Nelson M. Guerreiro, George E. Hagen, Jeffrey M. Maddalon, and Ricky W. Butler. “Capacity and Throughput of Urban Air Mobility Vertiports with a First-Come, First-Served Vertiport Scheduling Algorithm.” NASA Technical Reports Server, AIAA Aviation 2020 Forum, 2020. URL: <https://ntrs.nasa.gov/citations/20205001421>

[12] Pasquale Grippa, Doris A. Behrens, Friederike Wall, and Christian Bettstetter. “Drone delivery systems: job assignment and dimensioning.” *Autonomous Robots*, 43:261-274, 2019. DOI: 10.1007/s10514-018-9768-8. URL: <https://link.springer.com/article/10.1007/s10514-018-9768-8>

[13] Michael J. Neely. *Stochastic Network Optimization with Application to Communication and Queueing Systems.* Synthesis Lectures on Communication Networks, Morgan & Claypool Publishers, 2010. DOI: 10.2200/S00271ED1V01Y201006CNT007. URL: <https://doi.org/10.2200/S00271ED1V01Y201006CNT007>

[14] Leandros Tassiulas and Anthony Ephremides. “Stability Properties of Constrained Queueing Systems and Scheduling Policies for Maximum Throughput in Multihop Radio Networks.” *IEEE Transactions on Automatic Control*, 37(12):1936-1948, 1992. DOI: 10.1109/9.182479. URL: <https://doi.org/10.1109/9.182479>

[15] Akbar Telikani, Arupa Sarkar, Bo Du, and Jun Shen. “Machine Learning for UAV-Aided ITS: A Review With Comparative Study.” *IEEE Transactions on Intelligent Transportation Systems*, 25(11):15388-15406, 2024. DOI: 10.1109/TITS.2024.3422039. URL: <https://ieeexplore.ieee.org/document/10622103/>

[16] Han Liu, Tian Liu, and Kai Huang. “A Real-Time System for Scheduling and Managing UAV Delivery in Urban Areas.” arXiv:2412.11590, 2024. URL: <https://arxiv.org/abs/2412.11590>

[17] Jose Escribano Macias, Carl Khalife, Joseph Slim, and Panagiotis Angeloudis. “An integrated vertiport placement model considering vehicle sizing and queuing: A case study in London.” *Journal of Air Transport Management*, 113:102486, 2023. DOI: 10.1016/j.jairtraman.2023.102486. URL: <https://www.sciencedirect.com/science/article/pii/S0969699723001291>

[18] Blanca Lopez Palomino, Javier Muñoz Mendi, Fernando Quevedo Vallejo, Concepción Alicia Monje Micharet, Luis Santiago Garrido Bullon, and Luis Enrique Moreno Lorente. “4D Trajectory Planning Based on Fast Marching Square for UAV Teams.” *IEEE Transactions on Intelligent Transportation Systems*, 25(6):5703-5717, 2024. DOI: 10.1109/TITS.2023.3336008. URL: <https://doi.org/10.1109/TITS.2023.3336008>

---

## 附录：执行计划

### 第 1 周：冻结 paper positioning 与 problem formulation

- 明确主投 TR-C、备选 T-ITS。
- 冻结题目、摘要初稿、核心问题和三层架构图。
- 完成 problem formulation 的集合、队列、决策、目标和约束定义。

### 第 2-3 周：补 25+ 篇文献与相关工作矩阵

- 扩展 TR-C / T-ITS / UAM / UAV delivery / queueing / Lyapunov 文献。
- 输出 related work matrix：每篇论文的 problem、method、scale、mode、limitation。
- 明确 Paper B 相比每类工作的差异。

### 第 4-6 周：实现 synthetic UAM queueing benchmark

- 实现地图、vertiport、corridor、charging pad、OD demand generator。
- 支持 20 / 50 / 100 / 200 UAV 和 low / medium / peak / shock demand。
- 输出 manifest、seed、scenario config，保证可复现。

### 第 7-9 周：实现 baselines

- FCFS vertiport scheduling。
- Greedy nearest UAV。
- MILP rolling horizon 小规模上界。
- ALNS / heuristic dispatch。
- Queue-only backpressure。
- MARL / GNN dispatch without Lyapunov。

### 第 10-12 周：实现 H-LyraUAV 与消融

- 实现宏观 queue-aware assignment。
- 实现中观 corridor / vertiport / charging scheduling。
- 实现微观 energy / safety constraint interface。
- 实现 no-Lyapunov、no-multimodal、no-hierarchy、no-demand-prediction、no-charging-queue ablations。

### 第 13-15 周：跑主实验

- 跑 20 / 50 / 100 / 200 UAV scalability。
- 跑 peak / shock / bottleneck scenarios。
- 输出主表：delay、deadline violation、throughput、queue backlog、resource utilization、runtime。
- 输出关键图：queue trajectory、cost-delay tradeoff、scalability curve、multimodal fallback contribution。

### 第 16 周：写 TR-C 初稿

- Introduction 以 transportation operation problem 开场。
- Method 中放三层架构、Lyapunov theorem 和 algorithm。
- Experiments 强调 system performance、resource utilization、open benchmark。
- Discussion 写低空经济、vertiport capacity、charging infrastructure 和 multimodal logistics implications。

### T-ITS 改投策略

若 TR-C framing 不够强，或实验结果显示算法/控制贡献强于系统运输洞察，则保留 T-ITS 版本：

- Abstract 更强调 intelligent transportation system、online control、AI-assisted scheduling。
- Introduction 增加 sensing / communication / real-time implementation。
- Experiments 增加 runtime、communication delay、distributed execution 和 controller robustness。
- Discussion 减少政策/运营含义，增加智能系统部署和 ITS integration。
