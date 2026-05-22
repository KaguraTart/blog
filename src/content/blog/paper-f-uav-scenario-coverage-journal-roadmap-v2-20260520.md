---
title: "Paper F 期刊规划 v2：UAV 安全关键场景工程的期刊优先路线"
description: "在不考虑博士论文结构的前提下，重新规划 Paper F 的期刊优先产出路线，聚焦 UAV 安全关键场景覆盖、加速测试、风险保证与高速应急应用。"
pubDate: 2026-05-20
updatedDate: 2026-05-23
tags: ["Paper F", "期刊规划", "UAV", "场景生成", "场景覆盖", "Safety-Critical", "加速测试", "风险保证", "TR-C", "T-ITS"]
category: Tech
---

# Paper F 期刊规划 v2：UAV 安全关键场景工程的期刊优先路线

> 当前关注点先回到 **期刊论文产出**，不按博士论文来组织。  
> 结论：Paper F 不应拆成很多薄论文，而应先做一篇完整、扎实、可复现的 **T-ITS 主力期刊**，再根据实验资产分化出 TR-C 应用论文和风险保证方法论文。

---

## 1. 核心判断

Paper F 的大方向仍然成立：**UAV safety-critical scenario engineering**。但期刊论文和博士论文章节的逻辑不同。期刊审稿人不会因为路线完整而买单，他们更关心：

- 问题是否足够具体；
- 方法是否有明确技术增量；
- 实验是否足够扎实；
- baselines 是否强；
- 结论是否能支撑一个独立期刊故事；
- 与目标期刊 scope 是否贴合。

因此，当前应从“规划 4-5 篇”改成：

> **先把 F1 场景覆盖 benchmark 和 F2 危险场景加速生成合并为一篇主力 T-ITS；后续再从同一套平台分化出 TR-C 应急应用、T-RO/T-ASE 风险保证、TR-C/T-ITS 城市 ODD 场景生成。**

### 1.1 2026-05-22 写作校准：F 系列要拆清“测试方法论文”和“交通系统论文”

Paper F 很容易写散，因为它同时有场景生成、覆盖度、危险场景加速测试、城市 ODD、山东高速应急应用。最新校准是：

- **F-J1 不要硬写成 TR-C。** 它的第一性问题是 safety-critical scenario testing：如何系统性发现 UAV obstacle avoidance / low-altitude navigation 的危险但有效场景。主投 T-ITS 更自然，因为重点是智能交通系统中的安全测试、仿真评估和场景覆盖。
- **F-J2 才是 TR-C 应用论文。** 它必须写成高速交通应急运行问题：事故发现、无人机侦察、地面资源调配、响应时间、覆盖范围、信息延迟和交通恢复。
- **F-J4 只有在 city-level ODD 能反馈到交通规划/运行控制时才投 TR-C。** 如果只是把 OSM 转成局部障碍组合，更像仿真工具或 benchmark，不够 TR-C。

因此 F 系列的“故事”要分两种：

| 论文 | 系统故事 | 必须被什么支撑 |
|------|----------|----------------|
| F-J1 | 低空 ITS 安全测试缺少覆盖度和危险场景生成标准 | coverage metric、invalid scenario rate、failure discovery、planner cross-test、多 seed 统计 |
| F-J2 | 高速应急处置需要 UAV-ground 协同缩短黄金响应时间 | 真实高速拓扑/事故 proxy、资源调配模型、响应时间、覆盖率、拥堵下可达性、敏感性分析 |
| F-J3 | 场景覆盖如何转成风险保证证据 | coverage-to-risk bound、置信区间、rare-event estimation、可靠性指标 |
| F-J4 | 城市整体 ODD 如何决定局部低空测试场景 | OSM/POI/building/road/airspace 映射、局部风险保真度、跨城市泛化 |

交通期刊版本必须避免只说“我们生成了更多危险场景”。更强的结论应该是：

- 哪些城市结构或高速路段更容易诱发 UAV failure？
- 哪类障碍组合对不同 planner 最危险？
- 覆盖度提升是否真的减少未发现风险，而不是只增加样本数量？
- 在应急应用中，无人机侦察能否降低信息不完备带来的调度损失？
- 当天气、通信或起降点受限时，系统何时需要地面资源兜底？

原因很直接：单独 benchmark 容易被认为工程平台偏多，单独 accelerated testing 又会被追问测试场景空间是否定义清楚。两者合并后，论文从“我生成危险场景”升级为：

> **我定义了 UAV 安全关键场景空间，能度量覆盖，能发现 coverage holes，并能用覆盖引导的方法更高效地生成真实、危险、可行的测试场景。**

这更像一篇期刊论文。

### 1.2 2026-05-23 整理：F 系列当前只推进两条主线

当前 Paper F 不按博士论文目录铺开，先按期刊产出收束为两条主线。F-J3 和 F-J4 保留，但不抢 F-J1 的实验资源。

| 论文 | 主投 | 当前角色 | 近期策略 |
|------|------|----------|----------|
| F-J1 | T-ITS | coverage-guided accelerated testing | 主推进；必须使用 7600 万次探索日志、coverage metric、强 baseline 和 cross-planner evaluation |
| F-J2 | TR-C | 山东高速应急救援资源调配 | F-J1 平台稳定后启动；重点是真实高速拓扑、事故 proxy、响应时间和资源瓶颈 |
| F-J3 | T-RO / T-ASE / T-ITS | coverage-to-risk assurance | 暂缓；等 F-J1 形成 failure distribution 和 coverage statistics 后再证明风险边界 |
| F-J4 | TR-C / T-ITS | city-level ODD to local UAV scenario | 暂缓；等 OSM/POI/building/airspace pipeline 足够稳再做 |

F-J1 的第一版论文大纲建议固定为：

1. **Scenario space**：定义 UAV local test cell、障碍物语法、动态因素、任务目标和 invalid scenario 判定。
2. **Coverage metric**：把几何覆盖、语义覆盖、动力学覆盖、风险覆盖和 failure-mode 覆盖分开统计。
3. **Accelerated generation**：用 coverage holes 和 failure likelihood 引导采样，过滤不真实或不可执行场景。
4. **Benchmark protocol**：统一地图 seed、planner set、控制器参数、随机种子、失败阈值和统计检验。
5. **Main experiments**：比较 random、grid/LHS、BO、CMA-ES、RL adversarial、Scenic-style constrained generation 和本文方法。
6. **Failure analysis**：说明哪些障碍组合、速度/高度条件、遮挡和动态障碍最容易触发 failure。

这次整理后的判断是：F-J1 先追求“一篇能投 T-ITS 的安全测试期刊论文”，不要同时承诺城市规划、风险理论和山东高速应用。F-J2 可以在 F-J1 的场景库和风险指标成熟后，把故事切换成 TR-C 所需的交通应急运行闭环。

---

## 2. 期刊优先论文组合

建议暂时规划 **3 篇主力期刊 + 1 篇储备期刊**，而不是同时推进 5 篇。

| 编号 | 论文定位 | 建议题目 | 主投 | 优先级 |
|------|----------|----------|------|--------|
| F-J1 | 主力方法 + benchmark | Coverage-Guided Accelerated Testing for Safety-Critical UAV Navigation Scenarios | T-ITS | 最高 |
| F-J2 | 交通应急应用 | Scenario-Aware UAV-Ground Resource Allocation for Highway Emergency Response | TR-C | 高 |
| F-J3 | 风险保证方法 | Coverage-to-Risk Assurance for UAV Safety-Critical Scenario Testing | T-RO / T-ASE / T-ITS | 中高 |
| F-J4 | 城市场景生成 | City2Local-UAV: Hierarchical Scenario Generation from Urban ODDs to Local Obstacle Compositions | TR-C / T-ITS | 中 |

**执行顺序建议：F-J1 -> F-J2 -> F-J3 -> F-J4。**

F-J1 是平台和算法底座。F-J2 最接近交通期刊应用价值。F-J3 用来冲更方法/理论的机器人或自动化期刊。F-J4 只有在 OSM/城市数据管线成熟后再做，否则容易变成“地图转换工具”。

---

## 3. 文献格局与缺口

### 3.1 自动驾驶场景工程已经成熟，但 UAV 迁移不足

自动驾驶领域已经有完整的场景工程链条。ISO 34502 给出了 automated driving systems 的 scenario-based safety evaluation framework [1]，ASAM OpenSCENARIO 和 OpenODD 提供了可执行场景与 ODD 描述标准 [2] [3]。Shuo Feng 的 accelerated testing 和 testing scenario library generation 进一步说明，安全关键测试不能靠自然随机样本，而要用数据驱动方式提高 critical event 的采样效率 [4] [5] [6]。

近几年顶会也在持续推进这一方向：Scenic 用概率程序语言表达场景分布和约束 [7]；SafeBench 做了 safety evaluation benchmark [8]；ScenarioNet 从真实驾驶数据中抽取大规模 traffic scenarios [9]；AdvSim、KING、ChatScene、FREA 则分别从传感器扰动、梯度优化、LLM 知识生成和 feasible adversariality 的角度生成安全关键场景 [10] [11] [12] [13]。

但这些工作大多面向地面自动驾驶，UAV 场景有明显差异：

- UAV 是三维运动，场景维度包括高度、航迹倾角、风场、电量、视场遮挡和飞行动力学；
- UAV 的危险事件包括撞楼、撞线、穿越禁飞区、低空走廊冲突、起降失败和应急现场误入；
- UAV 安全测试很少有成熟 ODD taxonomy；
- UAV benchmark 多数重视仿真和控制任务，较少回答“场景覆盖了什么风险”。

### 3.2 UAV 仿真已有基础，但 coverage-oriented safety testing 仍空

AirSim 和 Flightmare 是无人机仿真的重要基础 [14] [15]。AvoidBench 已经针对 vision-based multi-rotor obstacle avoidance 提出高保真 benchmark [16]。OmniDrones 和 Aerial Gym 则说明 GPU 并行 UAV 仿真和大规模强化学习训练正在成熟 [17] [18]。FADS 证明 temporal logic safety specification 可以进入 drone safety pipeline [19]。

这些工作给 Paper F 提供了工具基础，但它们还没有解决期刊论文最关键的空白：

> **如何定义 UAV 安全关键场景空间、如何度量覆盖、如何高效生成既危险又可行的长尾场景、如何把测试覆盖转换为可解释的风险评估。**

这就是 F-J1/F-J3 的机会。

### 3.3 TR-C/T-ITS 的区别决定了论文切法

TR-C 的 intellectual core 在 transportation side，强调 emerging technologies 对 transportation system planning、design、operation、control 和 logistics 的影响 [20]。T-ITS 明确覆盖 sensing、communications、controls、planning、design、implementation、AI 和 transportation systems 中的信息技术应用 [21]。

因此：

- **F-J1 投 T-ITS**：因为它是 ITS safety evaluation / scenario generation / UAV navigation testing。
- **F-J2 投 TR-C**：因为它是高速交通应急系统运营与资源调配。
- **F-J3 投 T-RO/T-ASE/T-ITS**：取决于理论和实验重心，偏机器人安全测试可投 T-RO/T-ASE，偏交通系统可投 T-ITS。
- **F-J4 投 TR-C/T-ITS**：如果强调城市低空交通 ODD 和交通系统影响，投 TR-C；如果强调场景接口和仿真评测，投 T-ITS。

### 3.4 还能写哪些期刊方向

继续深挖后，可以多准备 4 个“候选分叉”，但不建议现在同时开写。它们更适合作为 F-J1 的实验平台成熟后的自然外溢。

| 分叉 | 可写题目 | 核心卖点 | 候选期刊 | 当前建议 |
|------|----------|----------|----------|----------|
| F-J5 | Scenario-Based Safety Case for Low-Altitude UAV Operations | 把 coverage、criticality、failure evidence 组织成 safety case | Reliability Engineering & System Safety / IEEE Transactions on Reliability / Safety Science | 等 F-J1 有结果后再写 |
| F-J6 | Cross-Simulator Transfer of UAV Safety-Critical Scenarios | 研究轻量仿真到 AirSim / Flightmare / AvoidBench 的场景迁移 | Robotics and Autonomous Systems / Journal of Field Robotics / T-RO | 需要真实或高保真验证 |
| F-J7 | Knowledge-Guided UAV Scenario Generation | 用 LLM / VLM / 知识图谱生成语义危险场景 | T-ITS / T-IV / IEEE Open Journal of ITS | 可和 Paper E 联动，但别喧宾夺主 |
| F-J8 | Multi-UAV Corridor Stress Testing | 专门生成低空 corridor 冲突、拥堵、起降瓶颈场景 | T-ITS / TR-C / T-IV | 可和 Paper B 联动 |

其中 **F-J5 最值得保留**。如果后续 F-J1 只停留在“发现更多失败”，期刊价值仍偏实验；但如果能把场景覆盖转成 reliability / safety assurance evidence，就能投 Reliability Engineering & System Safety 或 IEEE Transactions on Reliability 这种安全可靠性期刊 [28] [29]。Safety Science 也可作为备选，但它更偏安全管理、人因、组织和事故预防，如果论文仍是纯算法，不建议首投 [30]。

F-J6 适合在有真实无人机或高保真仿真结果时写。Journal of Field Robotics 和 Robotics and Autonomous Systems 都更看重机器人系统在真实或高保真环境中的自主性、可靠性和实验深度 [31] [32]。如果只有轻量仿真，先不要投这类期刊。

F-J7 不建议现在作为主线，因为它会和 Paper E 的 LLM/LTL 方向发生重叠。它更适合后续作为“knowledge-guided scenario generation”扩展：LLM 负责提出语义危险场景，Cov-ATUAV 负责验证、过滤和量化 coverage。

F-J8 则是 Paper B 的 stress-test 版本。它不再优化百架 UAV 调度，而是生成最能暴露 corridor congestion、vertiport bottleneck、charging bottleneck 和 conflict-resolution failure 的测试场景。这个方向可投 T-ITS 或 TR-C，但必须和 Paper B 的调度贡献切开。

### 3.5 候选期刊地图

| 期刊 | 最适合的 Paper F 切法 | 为什么合适 | 风险 |
|------|------------------------|------------|------|
| IEEE T-ITS | F-J1 / F-J4 / F-J8 | scope 覆盖 ITS 中的 sensing、control、AI、planning 和 transportation systems [21] | 需要把 UAV 写成 low-altitude ITS，不是普通机器人 |
| IEEE T-IV | F-J1 / F-J7 / F-J8 | intelligent vehicle 与 automated mobility 语境可接收安全测试和场景生成 [26] | 地面车辆色彩强，UAV 需要说明 vehicle/traffic relevance |
| TR-C | F-J2 / F-J4 / F-J8 | 强调 emerging technologies 对 transportation operations、control、logistics 的影响 [20] | 不适合纯算法 benchmark |
| TR-E | F-J2 | 适合物流、配送、供应链和应急资源运输调配 [33] | 若 UAV 技术细节太多会偏离 logistics |
| T-ASE | F-J3 / F-J5 | automation systems、testing、evaluation、reliability framing 较合适 [27] | 需要方法有自动化系统泛化价值 |
| T-RO | F-J3 / F-J6 | 机器人安全测试、规划、真实系统验证可投 [34] | 仅 synthetic benchmark 不够 |
| IEEE Transactions on Reliability | F-J5 | 适合 reliability modeling、risk quantification、assurance [28] | 需要严肃统计保证，不能只是实验表 |
| Reliability Engineering & System Safety | F-J5 | 适合安全关键系统、风险评估和可靠性工程 [29] | 需要从算法性能提升转成 safety evidence |
| Safety Science | F-J2 / F-J5 | 适合应急安全、事故预防、安全管理 [30] | 纯 UAV 算法不适合 |
| Robotics and Autonomous Systems / JFR | F-J6 | 适合 autonomous robotic systems 和 field/high-fidelity validation [31] [32] | 需要系统实验强于论文叙事 |
| IEEE Open Journal of ITS | F-J1 / F-J2 | 可作为快速开放获取备选 [35] | 影响力和定位通常低于 T-ITS |

**当前首投排序不变：F-J1 首投 T-ITS，F-J2 首投 TR-C，F-J3 视理论强度选择 T-ASE / T-RO / T-ITS，F-J5 作为可靠性期刊储备。**

---

## 4. 首篇主力期刊：F-J1

### 4.1 建议题目

**Coverage-Guided Accelerated Testing for Safety-Critical UAV Navigation Scenarios**

### 4.2 投稿目标

主投：**IEEE Transactions on Intelligent Transportation Systems**。  
备选：IEEE Transactions on Automation Science and Engineering、IEEE Transactions on Robotics、Robotics and Autonomous Systems。

T-ITS 最适合，因为论文可以写成 intelligent transportation safety testing：UAV 是低空交通参与者，场景生成服务于低空 ITS 的安全验证。

### 4.3 核心问题

现有 UAV 避障/导航论文通常报告 success rate、collision rate 或 trajectory length，但很少说明测试场景是否覆盖了安全关键 ODD。随机生成的场景又有两个问题：大量样本不危险，危险样本中又有很多物理不可行或任何算法都无法避免。

F-J1 要回答：

> 如何在有限仿真预算下，覆盖 UAV 低空运行 ODD，并优先生成真实、危险、可行、能区分算法能力的安全关键场景？

### 4.4 方法设计

方法名建议：**Cov-ATUAV: Coverage-Guided Accelerated Testing for UAVs**。

整体 pipeline：

```text
UAV ODD taxonomy
  -> scenario parameterization
  -> coverage memory
  -> criticality and feasibility scoring
  -> adaptive scenario generation
  -> planner evaluation and coverage update
```

核心模块：

| 模块 | 作用 |
|------|------|
| Scenario grammar | 定义障碍物、空间结构、动态体、风场、传感器噪声、任务类型 |
| Coverage memory | 记录 parameter bins、pairwise/t-wise coverage、failure modes |
| Criticality score | 综合 exposure、challenge、near-miss、constraint violation |
| Feasibility filter | 排除不可避免碰撞、物理不合理和任务无意义场景 |
| Adaptive generator | 在 coverage holes 和 high-criticality regions 中生成新样本 |
| Evaluation harness | 对多个 UAV planners 统一评测并输出 ranking stability |

### 4.5 数据与平台

- 主实验：50m x 50m x 50m 局部 UAV test cell。
- 已有资产：7600 万次探索日志，用于统计 coverage holes、失败模式和初始场景分布。
- 轻量仿真：自建 3D grid / PyBullet / custom dynamics，用于大规模搜索。
- 高保真验证：AirSim、Flightmare、AvoidBench 或 Aerial Gym，用于少量 cross-simulator validation [14] [15] [16] [18]。

7600 万次探索不能写成最终结果，但可以写成：

> “We initialize and validate our scenario coverage analysis using a large-scale exploration log containing over 76 million simulated rollouts.”

### 4.6 Baselines

| Baseline | 作用 |
|----------|------|
| Random scenario generation | 基础采样效率 |
| Grid sampling | 均匀离散覆盖 |
| Latin hypercube sampling | 参数空间覆盖 |
| Scenic-style constrained sampling | 约束式场景生成 [7] |
| SafeBench-style template sampling | 模板式 safety benchmark [8] |
| Bayesian optimization | 黑盒 failure search |
| CMA-ES / cross-entropy method | 连续参数危险搜索 |
| AdvSim/KING-style adversarial editing | 对抗轨迹/障碍扰动 [10] [11] |
| FREA-style feasible adversarial generation | 合理对抗样本 [13] |

### 4.7 UAV planners

至少要测三类 planner，否则期刊会质疑只对一个算法过拟合：

| Planner | 代表 |
|---------|------|
| Classical | A* / RRT* / artificial potential field / 3DVFH* |
| Optimization | MPC / safe corridor / B-spline trajectory optimization |
| Learning-based | PPO / SAC / imitation learning / vision-based policy |

如果算力有限，第一版保底选择：RRT*、MPC-lite、PPO policy、一个 vision-based baseline。

### 4.8 指标

| 指标 | 说明 |
|------|------|
| Coverage gain | 每 1000 次测试新增覆盖 |
| Failure discovery rate | 单位预算发现 collision / near-miss / timeout 的比例 |
| Acceleration factor | 达到同等失败发现率所需测试次数减少倍数 |
| Feasible criticality | 危险且可规避场景比例 |
| Invalid scenario rate | 物理不可行或无意义样本比例 |
| Planner ranking stability | 不同 seeds / scenario subsets 下 planner 排名稳定性 |
| Cross-simulator transfer | 在轻量仿真中发现的场景能否迁移到高保真仿真 |

### 4.9 期刊可发表的最低结果

F-J1 至少需要证明：

1. 相比 random / grid / LHS，Cov-ATUAV 在相同预算下显著提高 coverage gain。
2. 相比 BO / CMA-ES / adversarial baselines，Cov-ATUAV 降低 invalid scenario rate。
3. 相比纯 failure search，Cov-ATUAV 生成的场景能更稳定地区分不同 UAV planners。
4. 在 AirSim / Flightmare / AvoidBench 中至少验证一部分高风险场景可迁移。
5. 输出 scenario schema、seed、benchmark split 和评测脚本，增强 T-ITS 可复现性。

---

## 5. 第二篇期刊：F-J2 高速应急应用

### 5.1 建议题目

**Scenario-Aware UAV-Ground Resource Allocation for Highway Emergency Response**

### 5.2 投稿目标

主投：**Transportation Research Part C: Emerging Technologies**。  
备选：IEEE Transactions on Intelligent Transportation Systems。

这一篇必须把 UAV 放进 transportation operation，而不是写成“无人机调度算法”。山东高速综合巡检飞行服务系统已经在巡查、巡检、应急处置和数据分析中使用无人值守平台和行业无人机 [22]。高速应急资源调配研究也指出事故初期信息不完备、交通状态时变、设施选址和资源配置联动不足等问题 [23]。

### 5.3 核心问题

高速事故应急不是单纯“最近资源派过去”。事故刚发生时，事故类型、拥堵长度、车道封闭、危险品、二次事故风险都不确定。UAV 的价值不是飞得快而已，而是能提前降低信息不确定性，减少错派和延误。

F-J2 要回答：

> 在高速事故应急中，如何利用 UAV 侦察降低信息不确定性，并与地面清障、救护、消防、交警、养护资源协同调度，从而减少 response time、clearance time 和 secondary risk？

### 5.4 方法设计

方法名建议：**SAFER-UAV: Scenario-Aware Fast Emergency Response with UAVs**。

核心结构：

```text
Incident scenario generator
  -> UAV first-view dispatch
  -> incident state belief update
  -> ground resource rolling allocation
  -> congestion / clearance simulator
  -> emergency performance evaluation
```

关键是把 F-J1 的场景库转成应急任务场景：

- 事故类型：追尾、侧翻、危化品、占道施工、恶劣天气、拥堵次生事故。
- 路段几何：直线、弯道、匝道、服务区、收费站、桥梁、隧道入口。
- 不确定信息：事故严重程度、可通行车道、人员伤亡、资源需求、拥堵长度。
- 资源类型：UAV、清障车、救护车、消防、交警、养护车、临时管制设备。

### 5.5 Baselines

| Baseline | 作用 |
|----------|------|
| Ground-only dispatch | 无 UAV 情况 |
| Nearest-resource dispatch | 最近资源优先 |
| Fixed plan / rule-based dispatch | 当前实践近似 |
| UAV-first then dispatch | 先看后派的简单策略 |
| Two-stage stochastic programming | 随机优化基线 |
| Rolling horizon optimization | 强优化基线 |
| SAFER-UAV full | 主方法 |

### 5.6 指标

| 指标 | 说明 |
|------|------|
| First-view time | UAV 首次获取事故画面时间 |
| Response time | 第一批资源到达时间 |
| Clearance time | 清障完成时间 |
| Wrong dispatch rate | 错派、漏派或资源不足比例 |
| Secondary accident risk | 次生事故风险 proxy |
| Traffic delay | 事故导致总延误 |
| Information value of UAV | UAV 侦察带来的不确定性下降和调度收益 |
| Coverage of critical assets | UAV/地面资源对服务区、桥梁、隧道、事故多发段的覆盖能力 |
| Robustness to information delay | 图像回传、事件确认、通信延迟增加时性能下降幅度 |
| Equity across road segments | 偏远路段和核心路段的响应时间差异 |

TR-C 版还需要一张 **system implication table**：给定 UAV 数量、起降点数量、地面资源配置和事故强度，报告系统何时从“UAV 侦察有明显收益”转为“地面资源或路网拥堵成为主瓶颈”。这张表比单纯平均响应时间更像交通系统论文。

### 5.7 期刊可发表的最低结果

F-J2 至少需要：

1. 明确展示 UAV 侦察降低信息不确定性，而不是只缩短距离。
2. 在 peak / night / bad weather / multi-incident 场景下优于 ground-only 和 nearest-resource。
3. 与滚动优化或随机优化 baseline 对比，说明实时性和性能 tradeoff。
4. 写出 transportation implications：无人值守平台布设、资源预置、应急响应制度。

---

## 6. 第三篇期刊：F-J3 风险保证方法

### 6.1 建议题目

**Coverage-to-Risk Assurance for UAV Safety-Critical Scenario Testing**

### 6.2 投稿目标

主投视结果而定：

- 偏机器人安全测试：T-RO / IEEE Transactions on Automation Science and Engineering。
- 偏交通智能系统：T-ITS。
- 偏统计保证和学习风险：Machine Learning / Artificial Intelligence journal 方向。

### 6.3 为什么需要这篇

F-J1 能回答“怎么生成和覆盖场景”，但期刊审稿人还可能追问：

> 你覆盖了这些场景，能说明系统有多安全吗？覆盖度和真实风险之间是什么关系？

这就是 F-J3 的位置。它不是另一个 benchmark，而是把场景覆盖、重要性采样、scenario approach 和 conformal risk control 接起来。Campi 和 Garatti 的 scenario approach 给了随机场景约束下可行性概率保证 [24]，Conformal Risk Control 则提供了 distribution-free 风险控制框架 [25]。这些可以被改造为 UAV safety testing 的统计保证。

### 6.4 方法设计

方法名建议：**CovRisk-UAV**。

核心想法：

- 将 UAV scenario space 分成 coverage cells；
- 在每个 cell 内估计 failure / near-miss / violation risk；
- 使用 importance weighting 修正 accelerated testing 的采样偏差；
- 用 conformal risk control 给出 finite-sample risk upper bound；
- 对 planner ranking 给出置信区间，而不是只给平均 collision rate。

形式上，可以定义目标风险：

$$
R(\pi)=\mathbb{E}_{s\sim P_{\text{ODD}}}[\ell(\pi,s)],
$$

其中 $\pi$ 是 UAV planner，$s$ 是场景，$\ell$ 是 collision、near-miss 或 constraint violation loss。

由于测试场景来自加速分布 $Q(s)$，需要 importance correction：

$$
\hat{R}(\pi)=
\frac{1}{N}\sum_{i=1}^{N}
\frac{P_{\text{ODD}}(s_i)}{Q(s_i)}
\ell(\pi,s_i).
$$

再用 conformal / scenario bounds 给出：

$$
P(R(\pi)\leq \hat{R}_{\alpha}(\pi))\geq 1-\alpha.
$$

### 6.5 Baselines

| Baseline | 对比目的 |
|----------|----------|
| Empirical failure rate | 无置信保证 |
| Bootstrap confidence interval | 统计基线 |
| Importance sampling only | 只修正采样偏差 |
| Scenario approach only | 只做可行性概率界 |
| Conformal risk control | 风险控制基线 |
| CovRisk-UAV full | coverage-aware risk bound |

### 6.6 期刊可发表的最低结果

1. 在 synthetic known-risk 场景中验证风险上界校准有效。
2. 在 F-J1 场景库中给不同 planner 的风险置信区间。
3. 说明 accelerated testing 不能直接用原始 failure rate，需要分布修正。
4. 证明 coverage-aware risk bound 比 naive random testing 更紧或更稳定。

---

## 7. 第四篇期刊：F-J4 城市 ODD 到局部场景

### 7.1 建议题目

**City2Local-UAV: Hierarchical Scenario Generation from Urban ODDs to Local Obstacle Compositions**

### 7.2 投稿目标

主投：TR-C / T-ITS。  
这个方向暂时不建议排第一，因为它需要更多城市数据处理和案例支撑。

### 7.3 核心问题

局部 50m x 50m x 50m 场景虽然可控，但真实低空交通风险来自城市结构：道路等级、建筑密度、桥梁、服务区、禁飞区、医院、学校、立交和事故多发点。F-J4 要把 city-level ODD 和 local obstacle composition 建立映射。

### 7.4 方法设计

```text
City ODD
  -> functional zone and road segment extraction
  -> local UAV test-cell sampling
  -> obstacle grammar instantiation
  -> coverage-aware scenario selection
  -> simulator-ready scenario package
```

### 7.5 期刊可发表的最低结果

1. 至少两个城市或高速区域 case study。
2. 能证明 city-aware generation 比纯随机局部生成更真实。
3. 能证明生成的局部场景在 coverage 和 criticality 上优于人工模板。
4. 输出城市功能区到局部场景组合的可复现 pipeline。

---

## 8. 推荐路线：先写哪一篇

当前最应该集中火力的是 **F-J1**，不是同时开四篇。

### 8.1 为什么 F-J1 最优先

- 它把已有 7600 万次探索日志转化成论文资产。
- 它能吸收 F1 benchmark 和 F2 accelerated testing，体量够期刊。
- 它对后续三篇都有复用价值。
- 它最容易形成完整实验闭环：场景定义、生成方法、baselines、planners、metrics、跨仿真验证。

### 8.2 F-J1 的主线贡献应收束为三条

1. **UAV safety-critical scenario coverage taxonomy**  
   定义 UAV 低空 ODD、场景参数、coverage metric 和 failure taxonomy。

2. **Coverage-guided accelerated testing algorithm**  
   在 coverage holes 和 high-criticality regions 中生成危险但可行的场景。

3. **Reusable benchmark and evaluation protocol**  
   用多 planner、多 baseline、多仿真层级证明该 benchmark 能稳定评估 UAV safety。

不要把贡献写成 6-8 条。期刊引言里三条最清楚。

### 8.3 F-J1 最可能被拒的点

| 风险 | 原因 | 处理 |
|------|------|------|
| 被认为只是仿真平台 | benchmark 没有算法贡献 | 必须突出 coverage-guided accelerated testing |
| 被认为从自动驾驶照搬 | 缺少 UAV 特性 | 强调 3D dynamics、wind、battery、low-altitude obstacles、landing/emergency tasks |
| 被认为危险场景不真实 | adversarial 过强 | 加 feasibility 和 naturalness filter |
| 被认为只对单 planner 有效 | 过拟合 | 至少 4 类 planner |
| 被认为缺少交通系统意义 | UAV 只是机器人 | 写成 low-altitude ITS safety evaluation |

---

## 9. 近期实验任务拆解

### 第 1 周：冻结 F-J1 problem formulation

- 固定目标期刊：T-ITS。
- 固定主标题和三条贡献。
- 冻结 50m x 50m x 50m test cell。
- 定义场景参数表：geometry、obstacle、dynamic agent、weather、sensor、task、risk label。

### 第 2-3 周：处理 7600 万次探索日志

- 抽样 1-5 万条做初步分析。
- 统计 coverage holes。
- 聚类 failure modes：collision、near-miss、timeout、oscillation、energy violation、infeasible scene。
- 输出两张核心图：coverage heatmap 和 failure taxonomy。

### 第 4-6 周：实现 baseline generators

- random、grid、LHS。
- Scenic-style constrained generator。
- BO / CMA-ES。
- adversarial obstacle editing。
- feasible criticality filter。

### 第 7-9 周：实现 Cov-ATUAV

- coverage memory。
- criticality score。
- feasibility filter。
- adaptive generator。
- planner evaluation harness。

### 第 10-12 周：主实验

- 对比 failure discovery rate、coverage gain、invalid rate、acceleration factor。
- 测试 RRT*、MPC-lite、PPO、vision policy。
- 做 AirSim / Flightmare / AvoidBench 子集迁移验证。

### 第 13-16 周：写 T-ITS 初稿

- Introduction 聚焦 low-altitude ITS safety testing。
- Related work 分成 scenario-based safety evaluation、safety-critical scenario generation、UAV simulation and obstacle avoidance。
- Experiments 用主表 + coverage 图 + failure discovery 曲线 + cross-simulator transfer。

---

## 10. 当前不建议做的事

- 不要马上写 5 篇论文标题然后并行推进。
- 不要先做 F-J4 城市 ODD，因为数据管线会拖慢首篇产出。
- 不要把山东高速应急和 F-J1 混在一篇里，否则 T-ITS 主线会散。
- 不要把 7600 万次探索写成最终成果，它现在是数据资产，不是结论。
- 不要只报告 collision rate，必须报告 coverage、criticality、invalid rate 和 planner ranking stability。

---

## 11. 参考文献

[1] International Organization for Standardization. “ISO 34502:2022 Road vehicles — Test scenarios for automated driving systems — Scenario based safety evaluation framework.” 2022. URL: <https://www.iso.org/standard/78951.html>

[2] ASAM. “ASAM OpenSCENARIO DSL: Key Terminology and Conceptual Overview.” URL: <https://publications.pages.asam.net/standards/ASAM_OpenSCENARIO/ASAM_OpenSCENARIO_DSL/latest/conceptual-overview/key_terms.html>

[3] ASAM. “ASAM OpenODD: Model to ASAM OpenSCENARIO DSL Mapping Reference.” URL: <https://publications.pages.asam.net/standards/ASAM_OpenODD/ASAM_OpenODD/latest/specification/09_openscenario_dsl/09_01_overview.html>

[4] Shuo Feng, Xintao Yan, Haowei Sun, Yiheng Feng, and Henry X. Liu. “Intelligent Driving Intelligence Test for Autonomous Vehicles with Naturalistic and Adversarial Environment.” *Nature Communications*, 12:748, 2021. DOI: 10.1038/s41467-021-21007-8. URL: <https://doi.org/10.1038/s41467-021-21007-8>

[5] Shuo Feng, Yiheng Feng, Chunhui Yu, Yi Zhang, and Henry X. Liu. “Testing Scenario Library Generation for Connected and Automated Vehicles, Part I: Methodology.” *IEEE Transactions on Intelligent Transportation Systems*, 22(3):1573-1582, 2021. DOI: 10.1109/TITS.2020.2972211. URL: <https://doi.org/10.1109/TITS.2020.2972211>

[6] Shuo Feng, Yiheng Feng, Haowei Sun, Shan Bao, Yi Zhang, and Henry X. Liu. “Testing Scenario Library Generation for Connected and Automated Vehicles, Part II: Case Studies.” *IEEE Transactions on Intelligent Transportation Systems*, 22(9):5635-5647, 2021. DOI: 10.1109/TITS.2020.2988309. URL: <https://doi.org/10.1109/TITS.2020.2988309>

[7] Daniel J. Fremont, Tommaso Dreossi, Shromona Ghosh, Xiangyu Yue, Alberto L. Sangiovanni-Vincentelli, and Sanjit A. Seshia. “Scenic: A Language for Scenario Specification and Scene Generation.” *Proceedings of the 40th ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI)*, 2019. DOI: 10.1145/3314221.3314633. URL: <https://people.eecs.berkeley.edu/~sseshia/pubs/b2hd-fremont-pldi19.html>

[8] Chejian Xu, Wenhao Ding, Weijie Lyu, Zuxin Liu, Shuai Wang, Yihan He, Hanjiang Hu, Ding Zhao, and Bo Li. “SafeBench: A Benchmarking Platform for Safety Evaluation of Autonomous Vehicles.” *Advances in Neural Information Processing Systems 35 (NeurIPS 2022) Datasets and Benchmarks Track*, 2022. URL: <https://proceedings.neurips.cc/paper_files/paper/2022/hash/a48ad12d588c597f4725a8b84af647b5-Abstract-Datasets_and_Benchmarks.html>

[9] Quanyi Li, Zhenghao Peng, Lan Feng, Zhizheng Liu, Chenda Duan, Wenjie Mo, and Bolei Zhou. “ScenarioNet: Open-Source Platform for Large-Scale Traffic Scenario Simulation and Modeling.” *Advances in Neural Information Processing Systems 36 (NeurIPS 2023) Datasets and Benchmarks Track*, 2023. URL: <https://proceedings.neurips.cc/paper_files/paper/2023/hash/0c26a501df8fb919a0350e2df06b5d39-Abstract-Datasets_and_Benchmarks.html>

[10] Jingkang Wang, Ava Pun, James Tu, Sivabalan Manivasagam, Abbas Sadat, Sergio Casas, Mengye Ren, and Raquel Urtasun. “AdvSim: Generating Safety-Critical Scenarios for Self-Driving Vehicles.” *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2021. DOI: 10.1109/CVPR46437.2021.00978. URL: <https://openaccess.thecvf.com/content/CVPR2021/html/Wang_AdvSim_Generating_Safety-Critical_Scenarios_for_Self-Driving_Vehicles_CVPR_2021_paper.html>

[11] Niklas Hanselmann, Katrin Renz, Kashyap Chitta, Apratim Bhattacharyya, and Andreas Geiger. “KING: Generating Safety-Critical Driving Scenarios for Robust Imitation via Kinematics Gradients.” *European Conference on Computer Vision (ECCV)*, 2022. DOI: 10.1007/978-3-031-19839-7_20. URL: <https://is.mpg.de/ps/publications/king_geiger2022>

[12] Jiawei Zhang, Chejian Xu, and Bo Li. “ChatScene: Knowledge-Enabled Safety-Critical Scenario Generation for Autonomous Vehicles.” *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2024, pp. 15459-15469. URL: <https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_ChatScene_Knowledge-Enabled_Safety-Critical_Scenario_Generation_for_Autonomous_Vehicles_CVPR_2024_paper.html>

[13] Keyu Chen, Yuheng Lei, Hao Cheng, Haoran Wu, Wenchao Sun, and Sifa Zheng. “FREA: Feasibility-Guided Generation of Safety-Critical Scenarios with Reasonable Adversariality.” arXiv:2406.02983, 2024. URL: <https://arxiv.org/abs/2406.02983>

[14] Shital Shah, Debadeepta Dey, Chris Lovett, and Ashish Kapoor. “AirSim: High-Fidelity Visual and Physical Simulation for Autonomous Vehicles.” *Field and Service Robotics*, Springer Proceedings in Advanced Robotics, 2017; arXiv:1705.05065. URL: <https://arxiv.org/abs/1705.05065>

[15] Yunlong Song, Selim Naji, Elia Kaufmann, Antonio Loquercio, and Davide Scaramuzza. “Flightmare: A Flexible Quadrotor Simulator.” *Proceedings of the 4th Conference on Robot Learning (CoRL)*, PMLR 155, 2021. URL: <https://proceedings.mlr.press/v155/song21a.html>

[16] Hang Yu, Guido C. H. E. de Croon, and Christophe De Wagter. “AvoidBench: A High-Fidelity Vision-Based Obstacle Avoidance Benchmarking Suite for Multi-Rotors.” arXiv:2301.07430, 2023. URL: <https://arxiv.org/abs/2301.07430>

[17] Botian Xu, Feng Gao, Chao Yu, Ruize Zhang, Yi Wu, and Yu Wang. “OmniDrones: An Efficient and Flexible Platform for Reinforcement Learning in Drone Control.” *IEEE Robotics and Automation Letters*, 9(3):2838-2844, 2024. DOI: 10.1109/LRA.2024.3356168. URL: <https://ieeexplore.ieee.org/document/10409589/>

[18] Mihir Kulkarni, Theodor J. L. Forgaard, and Kostas Alexis. “Aerial Gym: Isaac Gym Simulator for Aerial Robots.” arXiv:2305.16510, 2023. URL: <https://arxiv.org/abs/2305.16510>

[19] Yash Vardhan Pant, Max Z. Li, Alena Rodionova, Rhudii A. Quaye, Houssam Abbas, Megan S. Ryerson, and Rahul Mangharam. “FADS: A Framework for Autonomous Drone Safety Using Temporal Logic-Based Trajectory Planning.” *Transportation Research Part C: Emerging Technologies*, 130:103275, 2021. DOI: 10.1016/j.trc.2021.103275. URL: <https://doi.org/10.1016/j.trc.2021.103275>

[20] Elsevier. “Transportation Research Part C: Emerging Technologies: Aims & Scope.” URL: <https://www.sciencedirect.com/journal/transportation-research-part-c-emerging-technologies>

[21] IEEE Intelligent Transportation Systems Society. “IEEE Transactions on Intelligent Transportation Systems (T-ITS): Scope.” URL: <https://ieee-itss.org/pub/t-its/>

[22] 山东高速集团有限公司. “‘山东高速综合巡检飞行服务系统’上线运行.” 2025. URL: <https://www.sdhsg.com/article/72553>

[23] 赵祥模, 赵一飞, 吕能超, 等. “高速公路交通事故应急关键资源调配研究综述.” *交通运输工程学报*, 2024. DOI: 10.19818/j.cnki.1671-1637.2024.06.001. URL: <https://transport.chd.edu.cn/cn/article/doi/10.19818/j.cnki.1671-1637.2024.06.001>

[24] Marco C. Campi and Simone Garatti. “The Exact Feasibility of Randomized Solutions of Uncertain Convex Programs.” *SIAM Journal on Optimization*, 19(3):1211-1230, 2008. DOI: 10.1137/07069821X. URL: <https://epubs.siam.org/doi/10.1137/07069821X>

[25] Anastasios N. Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei, and Tal Schuster. “Conformal Risk Control.” *International Conference on Learning Representations (ICLR)*, 2024. URL: <https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf>

[26] IEEE Intelligent Transportation Systems Society. “IEEE Transactions on Intelligent Vehicles.” URL: <https://ieee-itss.org/pub/t-iv/>

[27] IEEE Robotics and Automation Society. “IEEE Transactions on Automation Science and Engineering.” URL: <https://www.ieee-ras.org/publications/t-ase>

[28] IEEE Reliability Society. “IEEE Transactions on Reliability.” URL: <https://rs.ieee.org/publications/transactions-on-reliability/>

[29] Elsevier. “Reliability Engineering & System Safety: Aims & Scope.” URL: <https://www.sciencedirect.com/journal/reliability-engineering-and-system-safety>

[30] Elsevier. “Safety Science: Aims & Scope.” URL: <https://www.sciencedirect.com/journal/safety-science>

[31] Wiley. “Journal of Field Robotics: Overview.” URL: <https://onlinelibrary.wiley.com/journal/15564967>

[32] Elsevier. “Robotics and Autonomous Systems: Aims & Scope.” URL: <https://www.sciencedirect.com/journal/robotics-and-autonomous-systems>

[33] Elsevier. “Transportation Research Part E: Logistics and Transportation Review: Aims & Scope.” URL: <https://www.sciencedirect.com/journal/transportation-research-part-e-logistics-and-transportation-review>

[34] IEEE Robotics and Automation Society. “IEEE Transactions on Robotics.” URL: <https://www.ieee-ras.org/publications/t-ro>

[35] IEEE Intelligent Transportation Systems Society. “IEEE Open Journal of Intelligent Transportation Systems.” URL: <https://ieee-itss.org/pub/oj-its/>

---

## 附录：本次优化结论

1. 期刊优先时，Paper F 不应平铺成很多小论文。
2. 首篇应合并 benchmark 和 accelerated testing，形成 T-ITS 主力论文。
3. 山东高速应急应用应独立成 TR-C，不应混入首篇。
4. 风险保证论文可以作为中后期高水平方法期刊储备。
5. 城市 ODD 到局部场景生成暂排第四，等数据管线稳定后再推进。
