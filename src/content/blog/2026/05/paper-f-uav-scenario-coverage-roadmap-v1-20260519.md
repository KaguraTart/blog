---
title: "Paper F 论文群规划 v1：UAV 安全关键场景生成、覆盖与应急应用"
description: "规划 UAV 安全关键场景生成、场景覆盖、城市-局部场景关联与高速应急救援资源调配方向的多篇论文路线。"
pubDate: 2026-05-19
tags: ["Paper F", "UAV", "场景生成", "场景覆盖", "Safety-Critical", "加速测试", "应急救援", "TR-C", "T-ITS"]
category: Tech
---

# Paper F 论文群规划 v1：UAV 安全关键场景生成、覆盖与应急应用

> 总体判断：除了 Paper B 的百架级系统调度、Paper C 的 3DGS/FIM 主动感知、Paper E 的 LLM/LTL 语言规划之外，还可以单独开一条 **UAV safety-critical scenario engineering** 论文线。  
> 这条线的核心不是再做一个避障算法，而是回答：**如何系统生成、覆盖、筛选和复用 UAV 安全关键场景，让后续训练、测试、应急调度和论文实验都有可信的场景底座。**

---

## 1. 总体判断：还能写什么方向

当前已有几条论文线分别偏向不同问题：

| 论文线 | 核心对象 | 已经覆盖 | 不应重复 |
|--------|----------|----------|----------|
| Paper B | 百架 UAV fleet | 三层分层调度、排队论、Lyapunov、多模态运输 | 不再单独写大规模 fleet 调度 |
| Paper C | UAV 主动感知 | 3DGS、Fisher information、next-best-view | 不再主打建图/视角选择 |
| Paper E | 语言到规划 | LLM、TaskIR、LTL/STL、形式化验证 | 不再主打自然语言任务规划 |
| Paper F | 场景工程 | 场景生成、覆盖度、危险场景、应急应用 | 新方向 |

Paper F 的价值在于它可以成为前面几篇论文的 **实验基础设施**：

- Paper B 需要城市需求和应急场景。
- Paper C 需要可控的局部 3D 场景和遮挡/视角覆盖。
- Paper E 需要任务语义、地图实体和安全约束。
- Paper F 可以提供统一的 scenario grammar、coverage metric、criticality score 和 benchmark。

你提到的 “FENG SHOU 丰硕” 方向，建议规范对应为 **Shuo Feng** 的自动驾驶 accelerated testing / testing scenario library generation 工作。其核心思想是：安全关键事件在自然数据里极其稀少，不能只靠普通随机测试，而要用数据方法构造更危险但仍合理的场景，从而加速测试和安全验证 [1] [2] [3]。这个思想非常适合迁移到 UAV obstacle avoidance、低空通道飞行和高速应急巡检。

---

## 2. 论文群层次设计

Paper F 建议规划为 4 篇递进论文：

| 层次 | 论文 | 一句话定位 | 优先级 |
|------|------|------------|--------|
| F1 | CovUAV-Bench | 做 UAV 安全关键场景覆盖 benchmark | 最高 |
| F2 | Coverage-Guided Accelerated Testing | 做覆盖引导的危险场景加速生成算法 | 最高 |
| F3 | City2Local-UAV | 做城市整体 ODD 到局部障碍组合的层次场景生成 | 中高 |
| F4 | Scenario-Aware Emergency Response | 做山东高速无人机-地面资源协同应急调配 | 中高 |

推荐先做 **F1 + F2**。F1 提供数据集、指标和问题定义，F2 提供算法贡献。F3 和 F4 可以作为扩展：F3 将 benchmark 变成城市级系统，F4 将场景工程落到真实交通应急业务。

---

## 3. 共同背景：为什么场景覆盖是 UAV 安全研究的底座

UAV 安全研究常见的弱点是：算法做得很漂亮，但实验场景过于随意。一个避障算法在 20 个手工场景里成功，并不能说明它覆盖了城市低空运行中的长尾风险。

自动驾驶领域已经形成清晰共识：真实道路中的 crash / near-crash 是稀有事件，直接靠自然测试会极低效。因此，Shuo Feng 等提出 naturalistic and adversarial environment，用自然分布保持真实性，用对抗分布提高危险事件出现概率，从而加速智能驾驶测试 [1]。他们进一步提出 testing scenario library generation，将 ODD 下的测试场景库定义为一组具有代表性和关键性的场景，并用 criticality 兼顾 exposure frequency 与 maneuver challenge [2] [3]。Ding 等的 safety-critical scenario generation 综述也将该领域归纳为 data-driven、adversarial 和 knowledge-based 三类方法，并指出 fidelity、efficiency、diversity、transferability 和 controllability 是核心挑战 [13]。

UAV 场景更需要这套思想，原因有四个：

1. **三维空间更高维。**  
   UAV 不只是平面车道，还涉及高度、障碍物体积、风场、电量、传感器视野和飞行动力学。

2. **危险事件更难收集。**  
   真实撞楼、撞线、进入禁飞区、穿越桥梁或高速事故现场的样本很少，不能依赖真实事故数据训练。

3. **普通随机生成浪费算力。**  
   大量随机场景要么太简单，要么物理不可行，要么危险但不可避免，对训练和评测都低效。

4. **场景覆盖没有统一度量。**  
   现有 UAV 论文常报告 success rate / collision rate，但很少说明测试集覆盖了哪些障碍物组合、局部几何、任务难度和 ODD 边界。

因此，Paper F 的共同科学问题是：

> 如何构建一个既真实、可控、可复现，又能高效覆盖安全关键长尾风险的 UAV 场景生成与评估体系？

---

## 4. Paper F1：UAV 安全关键场景覆盖 Benchmark

### 4.1 论文题目

**CovUAV-Bench: A Coverage-Oriented Benchmark for Safety-Critical UAV Navigation Scenarios**

### 4.2 背景

SafeBench 已经在自动驾驶中提供统一的 safety-critical benchmark，整合了多类场景模板、场景生成算法和评价指标 [5]。Scenic 也证明了用概率程序表达场景分布、硬约束和软约束是可行路线 [4]。UAV 仿真环境生成已有初步工作，例如 Nakama 等提出过自动化 UAV simulation environment generator [10]，FADS 也说明 temporal-logic safety specification 可以进入 autonomous drone safety pipeline [11]。但在 UAV 领域，仍缺少一个面向 3D obstacle avoidance、低空 corridor、城市局部空间和应急任务的 coverage-oriented benchmark。

F1 的目标不是提出最强 planner，而是定义 **UAV 场景空间如何被系统覆盖**。

### 4.3 方法

构建一个 50m x 50m x 50m 的基础测试空间，先从局部场景做起，再扩展到城市 block：

- **场景对象**：建筑块、杆塔、电线、树木、桥梁、临时障碍、动态 UAV、地面车辆、人员区域。
- **空间结构**：open space、narrow passage、urban canyon、under-bridge、landing zone、highway shoulder、accident zone。
- **环境扰动**：风、能见度、传感器噪声、GPS 偏移、通信延迟。
- **任务类型**：point-to-point navigation、inspection pass、emergency hover、landing、return-to-home。
- **可执行格式**：统一保存为 `scenario.json` 加 simulator adapter，后续可转 AirSim、Flightmare、PyBullet 或自建轻量仿真。

场景覆盖度定义为：

$$
Coverage(S)=
\sum_{k=1}^{K} w_k \cdot
\frac{|B_k(S)|}{|B_k(\Omega)|},
$$

其中 $\Omega$ 是目标 ODD 的离散化场景空间，$B_k(S)$ 是样本集 $S$ 在第 $k$ 类属性维度上覆盖到的 bin，$w_k$ 是维度权重。

已有的 **7600 万次探索** 可以写成“已有探索日志资产”，用于统计：

- 哪些场景组合被频繁探索。
- 哪些组合仍是 coverage holes。
- 哪些组合触发 collision / near-miss / timeout。
- 哪些组合属于无效训练样本。

注意：7600 万次探索只写成“可用实验基础”，不能写成已验证结论。

### 4.4 Baselines

| Baseline | 用途 |
|----------|------|
| Random scenario sampling | 最基础覆盖基线 |
| Grid sampling | 参数空间均匀离散 |
| Latin hypercube sampling | 更高效的参数覆盖 |
| Scenic-style constrained sampling | 约束场景生成基线 [4] |
| SafeBench-style template suite | 模板化安全场景基线 [5] |

### 4.5 创新点

1. 提出 UAV 场景覆盖 taxonomy：ODD、障碍物组合、动态扰动、任务类型、风险等级。
2. 给出 coverage-oriented benchmark，而不是只给若干手工地图。
3. 将 exploration log 转化为 coverage holes 和 critical scenario seed。
4. 为后续 Paper B/C/E 提供统一场景接口。

### 4.6 如何评价

| 指标 | 含义 |
|------|------|
| Parameter coverage | 参数 bin 覆盖比例 |
| Pairwise / t-wise coverage | 多属性组合覆盖 |
| Critical scenario density | 单位测试预算发现的 near-miss / collision 数 |
| Invalid scenario rate | 物理不可行或任务无意义场景比例 |
| Planner ranking stability | 不同随机种子下算法排名是否稳定 |
| Replay reproducibility | 同一 seed 是否可复现同一结果 |

### 4.7 推荐投稿

- 主线：T-ITS / IEEE ITSC / IROS benchmark-oriented paper。
- 备选：RA-L + ICRA，如果 benchmark 同时有高质量开源工具和真实 UAV 小规模验证。

---

## 5. Paper F2：覆盖引导的危险场景加速生成

### 5.1 论文题目

**Coverage-Guided Accelerated Testing for Safety-Critical UAV Obstacle Avoidance**

### 5.2 背景

自动驾驶 accelerated testing 的核心不是“制造必撞场景”，而是提高测试对 safety-critical event 的采样效率，同时保持场景真实性和可行动性 [1] [2] [3]。如果生成的场景对任何 planner 都不可行，那么它不能帮助区分算法能力；如果生成的场景过于安全，又无法暴露系统弱点。

UAV 避障训练也存在同样问题：

- 随机生成的大量场景没有安全压力。
- 对抗生成容易产生不可合理规避的障碍布局。
- 手工 curriculum 覆盖有限，无法解释是否覆盖长尾风险。
- RL 训练会在大量无效场景上浪费预算。

### 5.3 方法

提出 **CGAT-UAV: Coverage-Guided Accelerated Testing for UAVs**。

算法由四个模块组成：

1. **Scenario encoder**  
   将场景编码为结构化向量：障碍物数量、最小通道宽度、目标方向、动态障碍速度、风强度、传感器噪声、电量裕度等。

2. **Coverage memory**  
   维护已探索场景的 coverage bins、失败类型和 planner 表现。

3. **Criticality score**  
   参考 Feng 的 criticality 思路，将危险程度和暴露频率结合 [2]：

   $$
   Crit(s)=P_{\text{exposure}}(s)\cdot R_{\text{challenge}}(s)\cdot F_{\text{feasible}}(s).
   $$

   其中 $F_{\text{feasible}}(s)$ 用于惩罚不可避免碰撞和物理不合理场景。

4. **Adaptive generator**  
   在 coverage holes 和 high-criticality regions 中生成新场景，可用 Bayesian optimization、CMA-ES、RL editing 或 cross-entropy method 实现。

### 5.4 Baselines

| Baseline | 对比目的 |
|----------|----------|
| Random generation | 测试加速倍率 |
| Grid / Latin hypercube sampling | 覆盖效率 |
| Bayesian optimization | 黑盒危险搜索 |
| CMA-ES | 连续参数危险搜索 |
| RL adversarial scenario generation | 学习式危险生成 |
| Scenic constrained generation | 规则与约束生成 [4] |
| FREA-style feasibility-guided generation | 对比“合理对抗性”思想 [12] |

### 5.5 创新点

1. 将 accelerated testing 从自动驾驶迁移到 UAV 3D obstacle avoidance。
2. 同时优化 **coverage、criticality、feasibility**，避免只追求碰撞率。
3. 提出 coverage-guided curriculum，用危险但可解的场景训练 planner。
4. 给出测试加速倍率：达到同等置信区间所需仿真次数显著减少。

### 5.6 如何评价

| 指标 | 含义 |
|------|------|
| Acceleration factor | 相比随机测试达到同等失败发现率所需测试次数减少倍数 |
| Failure discovery rate | 单位预算发现 collision / near-miss / timeout 的比例 |
| Feasible criticality | 危险且存在可行避障策略的比例 |
| Naturalness score | 场景是否符合 ODD 先验 |
| Coverage gain per 1k tests | 每 1000 次测试新增覆盖 |
| Training efficiency | 用生成场景训练后，planner 在 held-out test 上的提升 |

### 5.7 推荐投稿

- 主线：AAAI / ICRA / IROS。
- 备选：T-ITS，如果更强调交通安全测试和 benchmark。

---

## 6. Paper F3：城市整体场景到局部障碍组合的层次生成

### 6.1 论文题目

**City2Local-UAV: Hierarchical Scenario Generation from Urban ODDs to Local Obstacle Compositions**

### 6.2 背景

F1 和 F2 解决局部 3D 测试空间，但真实城市低空飞行不是孤立盒子。一个局部场景为什么会出现，取决于城市整体结构：道路等级、建筑密度、功能区、桥梁、服务区、立交、医院、学校、禁飞区域和应急点位。

ASAM OpenODD / OpenSCENARIO 提供了从 ODD、当前运行域到可执行场景描述的标准化思路 [6] [7]。UAV 领域可以借鉴这个抽象层级，但需要加入三维障碍、空域约束和低空任务语义。

### 6.3 方法

提出城市到局部的三层生成 pipeline：

```text
City-level ODD
  -> district / road / highway segment selection
  -> local 50m x 50m x 50m UAV test cell
  -> concrete obstacle composition
  -> simulator executable scenario
```

具体模块：

- **City ODD parser**：从 OSM、道路等级、建筑轮廓、POI、服务区、桥梁和高速出入口抽取城市/高速语义。
- **Local cell sampler**：选择典型局部单元，例如高楼峡谷、服务区、立交桥、收费站、高速肩部、事故瓶颈区。
- **Obstacle grammar**：用规则生成局部障碍组合，例如楼体 + 电线 + 树木 + 临停车辆 + 人员禁入区。
- **Coverage controller**：监控不同城市功能区与局部组合的覆盖。

### 6.4 Baselines

| Baseline | 对比目的 |
|----------|----------|
| Pure random local generation | 不考虑城市上下文 |
| OSM-to-map direct conversion | 只转地图，不控制场景覆盖 |
| CARLA / OSM digital twin generation | 地面自动驾驶式数字孪生基线 [14] |
| Manual scenario templates | 人工规则模板 |
| CityEngine / procedural city generation | 程序城市生成基线 |

### 6.5 创新点

1. 将城市 ODD 与 UAV 局部安全测试单元关联。
2. 提出 “global city semantics -> local obstacle composition” 的层次场景生成。
3. 让场景覆盖从局部参数扩展到城市功能区覆盖。
4. 支持真实城市 case study，例如济南、青岛、山东高速重点枢纽。

### 6.6 如何评价

| 指标 | 含义 |
|------|------|
| ODD coverage | 城市功能区、道路等级、建筑密度覆盖 |
| Local composition diversity | 局部障碍组合多样性 |
| Realism score | 与 OSM / POI / 建筑统计的一致性 |
| Transferability | 从一个城市生成策略迁移到另一个城市是否仍有效 |
| Criticality preservation | 城市上下文生成是否保留高风险局部场景 |

### 6.7 推荐投稿

- 主线：TR-C，如果强调城市交通系统、ODD、低空基础设施和场景数据集。
- 备选：T-ITS，如果强调 OpenSCENARIO-like 场景接口和智能系统评测。

---

## 7. Paper F4：面向山东高速应急救援的无人机-地面资源协同调配

### 7.1 论文题目

**Scenario-Aware UAV-Ground Resource Allocation for Highway Emergency Response**

### 7.2 背景

山东高速已有低空巡检和应急处置业务基础。山东高速集团公开信息显示，其综合巡检飞行服务系统已在重点区域部署无人值守平台和行业无人机，用于路况巡查、道路巡检、应急处置和数据分析等环节 [15]。这说明高速场景不是纯想象，而是已有应用入口。

高速公路应急资源调配研究指出，现有工作仍存在几个问题：运营阶段的路侧小型/微型应急设施点选址不足，事故初期常假设信息完备但实际并不成立，事故后交通状态不确定且时变，设施选址、资源配置和调度的一体化优化仍不足 [16]。交通事件监测中已有 space-time network UAV routing 研究 [17]，灾害应急通信中也有 UAV real-time deployment and resource allocation 研究 [18]，但它们还没有和高速应急事故场景覆盖、现场侦察信息价值、地面救援资源调配形成统一闭环。

这正适合引入 UAV：无人机先到达事故现场获取态势，地面清障、消防、救援和管制资源再动态调度。

### 7.3 方法

提出 **Scenario-Aware UAV-Ground Emergency Dispatch**：

- **事故场景生成**：基于 F1/F3 的高速场景库生成事故类型、车流状态、天气、路段几何、障碍物和次生风险。
- **UAV 侦察层**：无人机从服务区、收费站或无人值守平台起飞，快速确认事故位置、拥堵长度、可通行车道和危险品风险。
- **地面资源调配层**：调度清障车、消防、救护、交警、养护车辆和临时管制资源。
- **信息价值建模**：将 UAV 侦察减少的不确定性写入调度目标，即 UAV 不只是拍照，而是降低错误派遣和响应延误。
- **滚动优化**：事故信息随时间更新，调度策略滚动重算。

### 7.4 Problem formulation

令高速路段集合为 $\mathcal{L}$，事故集合为 $\mathcal{I}(t)$，UAV 集合为 $\mathcal{U}$，地面救援资源集合为 $\mathcal{G}$，服务站/无人值守平台集合为 $\mathcal{B}$。

决策变量包括：

- UAV 派遣 $x_{u,i}(t)$：无人机 $u$ 是否侦察事故 $i$。
- 地面资源派遣 $y_{g,i}(t)$：资源 $g$ 是否前往事故 $i$。
- 起飞/出车时间 $s_u(t), s_g(t)$。
- 信息更新动作 $a_i(t)$：是否等待 UAV 进一步确认，还是直接派遣。

目标函数：

$$
\min
\mathbb{E}\left[
\beta_1 T_{\text{response}}+
\beta_2 T_{\text{clearance}}+
\beta_3 C_{\text{dispatch}}+
\beta_4 R_{\text{secondary}}+
\beta_5 U_{\text{uncertainty}}
\right].
$$

其中 $U_{\text{uncertainty}}$ 表示事故信息不确定性，UAV 侦察可以降低该项。

### 7.5 Baselines

| Baseline | 对比目的 |
|----------|----------|
| Ground-only dispatch | 没有无人机侦察 |
| Nearest-resource dispatch | 最近资源优先 |
| Static facility allocation | 固定设施点配置 |
| Two-stage stochastic optimization | 先估计事故再派遣 |
| UAV-first heuristic | 无人机先侦察，再地面派遣 |
| Scenario-aware rolling optimization | 主方法 |

### 7.6 创新点

1. 将场景覆盖和高速应急调度打通，而不是只做资源调配。
2. 将 UAV 侦察建模为降低事故信息不确定性的决策动作。
3. 支持山东高速真实业务语境：无人值守平台、路况巡查、应急处置和工单流转。
4. 统一优化 response time、clearance time、secondary accident risk 和 dispatch cost。

### 7.7 如何评价

| 指标 | 含义 |
|------|------|
| First-view time | UAV 第一次获取事故画面的时间 |
| Response time | 第一批救援资源到达时间 |
| Clearance time | 事故清障完成时间 |
| Wrong dispatch rate | 错派、漏派或资源不足比例 |
| Secondary accident risk | 次生事故风险 proxy |
| Congestion delay | 事故导致的总延误 |
| UAV information value | 有 UAV 侦察相比无 UAV 的不确定性降低 |

### 7.8 推荐投稿

- 主线：TR-C first，因为重点是高速应急交通系统运营、资源配置和运输网络韧性。
- 备选：T-ITS，如果进一步强调无人机平台、通信、视频识别、工单系统和在线智能调度。

---

## 8. 统一实验平台、数据来源与评价指标

### 8.1 实验平台

| 层级 | 推荐实现 | 用途 |
|------|----------|------|
| 轻量仿真 | Python / PyBullet / custom 3D grid | 7600 万次级别快速探索 |
| UAV 仿真 | AirSim、Flightmare | 视觉、动力学、传感器验证 [8] [9] |
| 场景语言 | Scenic-like DSL、JSON schema | 可复现场景生成 [4] |
| 城市数据 | OpenStreetMap、POI、道路等级、建筑轮廓 | 城市到局部场景生成 |
| 高速应急 | 山东高速公开案例、事故统计、合成事故流 | 应急资源调配实验 |

F1/F2 的主实验应优先用轻量仿真，以保证大规模探索。AirSim / Flightmare 用于小规模高保真验证，不作为全部实验依赖。

### 8.2 数据来源

- **Synthetic UAV scenario benchmark**：程序生成 50m x 50m x 50m 局部空间。
- **Exploration logs**：已有 7600 万次探索日志，用于 coverage hole 和 failure taxonomy。
- **OSM / POI / building data**：用于城市功能区和局部障碍组合。
- **山东高速公开业务信息**：用于应用背景和部署假设 [15]。
- **高速事故与应急资源公开研究**：用于事故类型、资源调配阶段和评价指标 [16]。

### 8.3 统一指标

| 指标组 | 指标 |
|--------|------|
| 覆盖 | parameter coverage、t-wise coverage、ODD coverage、coverage gain |
| 安全 | collision rate、near-miss rate、minimum distance、constraint violation |
| 危险生成 | criticality、failure discovery rate、acceleration factor、feasible criticality |
| 训练价值 | sample efficiency、held-out success rate、robustness under ODD shift |
| 应急价值 | first-view time、response time、clearance time、wrong dispatch rate |

---

## 9. 推荐投稿路径与优先级

### 9.1 第一阶段：先做 F1 + F2

第一阶段建议直接围绕 “UAV 安全关键场景覆盖 + 加速测试” 做两篇：

1. **F1 benchmark paper**  
   更稳，适合作为后续所有 UAV 论文的实验底座。即使算法没有特别强，也可以凭 taxonomy、coverage metric、数据集和可复现实验成立。

2. **F2 method paper**  
   冲 AAAI / ICRA / IROS 的方法贡献。亮点是从 Shuo Feng 的自动驾驶 accelerated testing 迁移到 UAV 3D 场景，并加入 coverage-guided feasible criticality。

### 9.2 第二阶段：再做 F3 + F4

F3 和 F4 更适合在 F1/F2 有工具基础后推进：

- **F3** 解决城市整体和局部场景关联，可以投 TR-C / T-ITS。
- **F4** 做山东高速应急救援应用，可以投 TR-C，强调 transportation operation 和 emergency response。

### 9.3 与现有论文线的关系

| 论文 | Paper F 的支撑方式 |
|------|--------------------|
| Paper B | 提供 peak / shock / highway emergency demand scenarios |
| Paper C | 提供局部 3D 遮挡、视角覆盖和重建难度场景 |
| Paper E | 提供自然语言任务、地图实体和安全约束场景 |

Paper F 最适合成为整个 UAV 研究线的 “scenario infrastructure paper”。

---

## 10. 参考文献

[1] Shuo Feng, Xintao Yan, Haowei Sun, Yiheng Feng, and Henry X. Liu. “Intelligent Driving Intelligence Test for Autonomous Vehicles with Naturalistic and Adversarial Environment.” *Nature Communications*, 12:748, 2021. DOI: 10.1038/s41467-021-21007-8. URL: <https://doi.org/10.1038/s41467-021-21007-8>

[2] Shuo Feng, Yiheng Feng, Chunhui Yu, Yi Zhang, and Henry X. Liu. “Testing Scenario Library Generation for Connected and Automated Vehicles, Part I: Methodology.” *IEEE Transactions on Intelligent Transportation Systems*, 22(3):1573-1582, 2021. DOI: 10.1109/TITS.2020.2972211. URL: <https://doi.org/10.1109/TITS.2020.2972211>

[3] Shuo Feng, Yiheng Feng, Haowei Sun, Shan Bao, Yi Zhang, and Henry X. Liu. “Testing Scenario Library Generation for Connected and Automated Vehicles, Part II: Case Studies.” *IEEE Transactions on Intelligent Transportation Systems*, 22(9):5635-5647, 2021. DOI: 10.1109/TITS.2020.2988309. URL: <https://doi.org/10.1109/TITS.2020.2988309>

[4] Daniel J. Fremont, Tommaso Dreossi, Shromona Ghosh, Xiangyu Yue, Alberto L. Sangiovanni-Vincentelli, and Sanjit A. Seshia. “Scenic: A Language for Scenario Specification and Scene Generation.” *Proceedings of the 40th ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI)*, 2019. DOI: 10.1145/3314221.3314633. URL: <https://people.eecs.berkeley.edu/~sseshia/pubs/b2hd-fremont-pldi19.html>

[5] Chejian Xu, Wenhao Ding, Weijie Lyu, Zuxin Liu, Shuai Wang, Yihan He, Hanjiang Hu, Ding Zhao, and Bo Li. “SafeBench: A Benchmarking Platform for Safety Evaluation of Autonomous Vehicles.” *Advances in Neural Information Processing Systems 35 (NeurIPS 2022) Datasets and Benchmarks Track*, 2022. URL: <https://proceedings.neurips.cc/paper_files/paper/2022/hash/a48ad12d588c597f4725a8b84af647b5-Abstract-Datasets_and_Benchmarks.html>

[6] ASAM. “ASAM OpenSCENARIO DSL: Key Terminology and Conceptual Overview.” URL: <https://publications.pages.asam.net/standards/ASAM_OpenSCENARIO/ASAM_OpenSCENARIO_DSL/latest/conceptual-overview/key_terms.html>

[7] ASAM. “ASAM OpenODD: Model to ASAM OpenSCENARIO DSL Mapping Reference.” URL: <https://publications.pages.asam.net/standards/ASAM_OpenODD/ASAM_OpenODD/latest/specification/09_openscenario_dsl/09_01_overview.html>

[8] Shital Shah, Debadeepta Dey, Chris Lovett, and Ashish Kapoor. “AirSim: High-Fidelity Visual and Physical Simulation for Autonomous Vehicles.” *Field and Service Robotics*, Springer Proceedings in Advanced Robotics, 2017; arXiv:1705.05065. URL: <https://arxiv.org/abs/1705.05065>

[9] Yunlong Song, Selim Naji, Elia Kaufmann, Antonio Loquercio, and Davide Scaramuzza. “Flightmare: A Flexible Quadrotor Simulator.” *Proceedings of the 4th Conference on Robot Learning (CoRL)*, PMLR 155, 2021. URL: <https://proceedings.mlr.press/v155/song21a.html>

[10] Justin Nakama, Ricky Parada, Joao P. Matos-Carvalho, Fabio Azevedo, Dario Pedro, and Luis Campos. “Autonomous Environment Generator for UAV-Based Simulation.” *Applied Sciences*, 11(5):2185, 2021. DOI: 10.3390/app11052185. URL: <https://doi.org/10.3390/app11052185>

[11] Yash Vardhan Pant, Max Z. Li, Alena Rodionova, Rhudii A. Quaye, Houssam Abbas, Megan S. Ryerson, and Rahul Mangharam. “FADS: A Framework for Autonomous Drone Safety Using Temporal Logic-Based Trajectory Planning.” *Transportation Research Part C: Emerging Technologies*, 130:103275, 2021. DOI: 10.1016/j.trc.2021.103275. URL: <https://doi.org/10.1016/j.trc.2021.103275>

[12] Keyu Chen, Yuheng Lei, Hao Cheng, Haoran Wu, Wenchao Sun, and Sifa Zheng. “FREA: Feasibility-Guided Generation of Safety-Critical Scenarios with Reasonable Adversariality.” arXiv:2406.02983, 2024. URL: <https://arxiv.org/abs/2406.02983>

[13] Wenhao Ding, Chejian Xu, Mansur Arief, Haohong Lin, Bo Li, and Ding Zhao. “A Survey on Safety-Critical Driving Scenario Generation: A Methodological Perspective.” arXiv:2202.02215, 2022. URL: <https://arxiv.org/abs/2202.02215>

[14] CARLA Team. “Digital Twin Tool: Procedural Generation from OpenStreetMap.” CARLA Simulator Documentation. URL: <https://carla.readthedocs.io/en/0.9.16/adv_digital_twin/>

[15] 山东高速集团有限公司. “‘山东高速综合巡检飞行服务系统’上线运行.” 2025. URL: <https://www.sdhsg.com/article/72553>

[16] 赵祥模, 赵一飞, 吕能超, 等. “高速公路交通事故应急关键资源调配研究综述.” *交通运输工程学报*, 2024. DOI: 10.19818/j.cnki.1671-1637.2024.06.001. URL: <https://transport.chd.edu.cn/cn/article/doi/10.19818/j.cnki.1671-1637.2024.06.001>

[17] Jisheng Zhang, Limin Jia, Shuyun Niu, Fan Zhang, Lu Tong, and Xuesong Zhou. “A Space-Time Network-Based Modeling Framework for Dynamic Unmanned Aerial Vehicle Routing in Traffic Incident Monitoring Applications.” *Sensors*, 15(6):13874-13898, 2015. DOI: 10.3390/s150613874. URL: <https://doi.org/10.3390/s150613874>

[18] Tan Do-Duy, Long D. Nguyen, Trung Q. Duong, Saeed Khosravirad, and Holger Claussen. “Joint Optimisation of Real-Time Deployment and Resource Allocation for UAV-Aided Disaster Emergency Communications.” *IEEE Journal on Selected Areas in Communications*, 39(11):3411-3424, 2021. DOI: 10.1109/JSAC.2021.3088662. URL: <https://doi.org/10.1109/JSAC.2021.3088662>

---

## 附录：本次执行计划

### 第 1 步：冻结 Paper F 总定位

- 将 Paper F 定位为 UAV safety-critical scenario engineering。
- 明确它不是 Paper B/C/E 的重复，而是实验基础设施和场景方法论文群。
- 采用 F1-F4 四篇递进论文结构。

### 第 2 步：先做 F1 benchmark

- 定义 UAV 场景 taxonomy。
- 设计 `scenario.json` schema。
- 整理已有 7600 万次探索日志。
- 统计 coverage holes、failure modes 和 invalid scenario rate。
- 输出 CovUAV-Bench v0.1。

### 第 3 步：推进 F2 加速测试算法

- 实现 random / grid / LHS / BO / CMA-ES / RL adversarial baselines。
- 实现 coverage memory、criticality score 和 feasible criticality filter。
- 对比 failure discovery rate、coverage gain 和 acceleration factor。
- 用 held-out test 验证训练价值。

### 第 4 步：扩展 F3 城市到局部场景

- 接入 OSM、道路等级、建筑密度和 POI。
- 选择济南/青岛/山东高速重点路段作为 case study。
- 将 city-level ODD 映射到 local 50m x 50m x 50m test cell。
- 建立城市功能区覆盖指标。

### 第 5 步：扩展 F4 高速应急应用

- 把山东高速巡检/应急处置作为应用背景。
- 设计事故场景、UAV 侦察、地面救援资源协同调配流程。
- 对比 ground-only、nearest-resource、UAV-first heuristic 和 scenario-aware rolling optimization。
- 重点报告 first-view time、response time、clearance time 和 wrong dispatch rate。

### 第 6 步：投稿节奏

- 先投 F1/F2，形成 benchmark + method 双核心。
- F1 若工具和数据完整，优先 T-ITS / ITSC / IROS benchmark。
- F2 若算法结果强，优先 AAAI / ICRA / IROS。
- F3/F4 等 F1/F2 工具稳定后，再转 TR-C / T-ITS。

### 第 7 步：近期一周任务

- 写 F1 的正式实验任务书。
- 冻结场景维度：障碍物、空间结构、环境扰动、任务类型、风险标签。
- 从 7600 万次探索日志中抽样 1-5 万条做初步 coverage analysis。
- 画第一版场景 taxonomy 图和 coverage heatmap。
