---
title: "Nature / Nature Communications 级低空自主系统论文规划 v1：从工程系统到可证伪科学问题"
description: "基于现有 Paper A/B/C 与后续低空云脑、具身智能、推理加速路线，结合联网调研和三个独立 Claude 严格评审，规划真正可能冲 Nature / Nature Communications 级别的低空 UAV 论文方向。"
pubDate: 2026-05-28
updatedDate: 2026-05-28
tags: ["低空规划", "UAV", "Nature Communications", "Nature", "安全验证", "Rare Event", "复杂系统", "相变", "具身智能", "推理加速"]
category: Tech
---

# Nature / Nature Communications 级低空自主系统论文规划 v1：从工程系统到可证伪科学问题

本文目标不是继续扩展工程选题，而是严格判断：在当前已有三篇文章基础上，哪些方向有机会被提升到 **Nature / Nature Communications** 级别。

当前已经在做的三篇文章是：

| 文章 | 当前属性 | 正常投稿定位 |
|---|---|---|
| Paper A：无冲突路径规划 / PPO-MAPPO / 低空冲突消解 | 战术安全控制 | T-ITS / T-RO / ICRA / IROS |
| Paper B：百架 UAV 三层分层调度 | 城市低空交通系统运营 | TR-C / T-ITS |
| Paper C：Fisher 信息驱动 UAV 3DGS 主动感知 | 主动感知与数字孪生 | T-RO / ICRA / IROS / CVPR 方向 |

结论先写清楚：**以当前形态，A/B/C 都不是 Nature 级论文。**  
要冲 Nature Communications，必须从“工程系统做得更好”转成“发现、验证并解释一个可证伪的科学规律”。

---

## 1. Nature / Nature Communications 的门槛

Nature 正刊要求文章具有 outstanding scientific importance，并且能引起本领域外科学家的兴趣 [1]。Nature Communications 是多学科期刊，目标是发表各领域内具有重要推进意义的高质量研究 [2]。这意味着低空 UAV 论文要进入这个层级，不能只满足：

- 提出一个新调度器；
- 提出一个新强化学习避碰方法；
- 提出一个低空大模型 agent；
- 在仿真里比 baseline 高几个点；
- 做一个系统 demo 或 benchmark。

Nature / Nature Communications 更关心：

| 需要回答的问题 | 工程论文常见写法 | Nature 级写法 |
|---|---|---|
| 这个问题为什么重要 | 某个算法更好 | 解决一个跨系统共性的科学瓶颈 |
| 贡献是什么 | 提出方法 | 发现规律、证明边界、建立测量方法 |
| 证据是什么 | 仿真曲线和消融 | 理论 + 统计置信 + 真实世界校准 |
| 是否可证伪 | 不容易 | 必须能被其他数据、城市、硬件复现或推翻 |
| 影响范围 | UAV/ITS/Robotics 社区 | 交通科学、复杂系统、安全认证、具身智能共同关心 |

对我们来说，核心判断是：

> A/B/C/G/I/J/K 本身大多是工程论文；只有当它们服务于 rare-event safety measurement、capacity phase transition、energy-intelligence scaling law 这类科学问题时，才可能进入 Nature Communications 视野。

---

## 2. 相关高等级先例

### 2.1 自动驾驶安全验证先例

Shuo Feng 等人的 NADE 工作发表在 Nature Communications，提出用 naturalistic 和 adversarial 环境测试自动驾驶智能 [3]。后续 Dense Reinforcement Learning for Safety Validation 发表在 Nature 正刊，核心不是“训练了一个更强 agent”，而是把 autonomous vehicle safety validation 变成了 accelerated rare-event measurement，并声称在不损失无偏性的前提下实现 10^3 到 10^5 倍加速 [4]。

这给我们的直接启发是：**低空 UAV 方向最有 Nature Communications 机会的不是规划器本身，而是可校准的安全验证方法。**

Nature Communications 还发表了 “curse of rarity” 相关工作，明确 rare safety-critical events 在高维空间中的稀有性会阻碍深度模型学习与验证 [5]。这与低空 UAV 的核心难题一致：碰撞、近失、非合作飞行器侵入、通信失效和风扰组合都是低概率但高后果事件。

### 2.2 群体机器人与复杂系统先例

Nature Communications 近期也发表了 swarm robotics / collective intelligence 方向文章。例如 snail-inspired robotic swarms 展示了户外非结构环境中的群体适应能力 [6]；collective intelligence model for swarm robotics applications 将 swarm robotics 放进更一般的集体智能建模框架 [7]。

这说明 Nature Communications 对机器人系统并不排斥，但它看重的不是“机器人系统能跑”，而是系统背后的 **collective adaptation、scaling、emergence、phase transition、universal behaviour**。

### 2.3 低空大模型与具身智能背景

低空经济大模型综述已经把低空系统拆成设施网络、信息网络、航路网络和服务网络，并指出大模型需要与边缘计算、通信网络和可信自主系统结合 [8]。SINGER、FlightGPT、UAV-VLN、OpenVLA、RT-2 等工作说明 aerial embodied intelligence、VLA/VLN、robot foundation model 正在快速发展 [9] [10] [11] [12] [13]。

但这些工作也带来风险：**LLM/VLA/Agent 方向过热，单纯做 LowAltitudeGPT 或 CloudBrain-Agent 很容易被认为是工程包装。** 要冲 Nature Communications，模型必须成为“测量科学规律的仪器”，而不是论文主角。

---

## 3. 三个独立 Claude 评审结论

本轮拉了三个独立 Claude 评审代理，分别从 Nature 编辑、复杂交通/安全科学、具身 AI/机器人/边缘智能三个角度严格评审。三个评审的共识如下。

| 评审视角 | 最认可方向 | 明确拒绝方向 | 关键理由 |
|---|---|---|---|
| Nature / Nat Commun 编辑视角 | B、C；D 只有在可认证 rare-event 估计时成立 | A、G、I、J、K | 工程性能提升不是 Nature 级贡献，必须有普适规律或测量方法 |
| 复杂系统 / 交通安全视角 | D 第一，B 第二，C 第三 | G、I、J、K，未重构的 A | D 有 rare-event estimator 先例；B 有容量相变潜力 |
| 具身 AI / 边缘智能视角 | B+K 融合，D，I 的 scaling-law 版本 | 单独的 G/J/K，普通 I | I/J/K 默认是工程；只有能量-智能或具身 scaling law 才可能升级 |

三个评审给出的严格共识：

1. **Nature 正刊目前不现实。**
2. **Nature Communications 不是完全没机会，但必须换问题定义。**
3. **D：低空安全关键 rare-event accelerated testing 是最强候选。**
4. **B：低空交通容量相变 / 标度律是第二候选。**
5. **B+K：低空具身集群 energy-intelligence scaling law 是高风险高收益候选。**
6. **A、G、I、J、K 作为独立工程论文不应冲 Nature Communications。**
7. **C 只有在提出跨场景主动感知信息-代价理论边界，并有真实 UAV 外场验证时，才有边界机会。**

---

## 4. 最推荐主线：低空自主空域的可认证稀有事件安全测量

### 4.1 建议题目

**Certifiable Rare-Event Safety Measurement for Autonomous Low-Altitude Airspace**

中文可写为：

**面向自主低空空域的可认证稀有安全事件测量**

### 4.2 核心科学问题

低空自主系统的安全事故是 rare event。真实世界里，灾难性事件很少出现，但一旦出现后果严重。直接用 Monte Carlo 或普通仿真估计事故率会遇到 curse of rarity：大多数样本都是正常飞行，真正有价值的安全关键样本被海量正常样本淹没 [5]。

核心问题是：

> 能否构建一种对低空 UAV 多智能体系统适用的加速安全验证方法，在显著减少样本量的同时，仍能给出可校准、带置信区间、可被真实数据验证的风险估计？

这不是 “生成危险场景”，而是 **测量低空自主系统失效概率**。

### 4.3 目标贡献

| 贡献 | 必须达到的形式 |
|---|---|
| 低空 rare-event 空间定义 | 非合作 UAV、通信退化、定位误差、风扰、走廊冲突、vertiport 近失、应急插入 |
| 加速采样理论 | importance sampling / rare-event density / adversarial but naturalistic distribution |
| 估计器保证 | unbiasedness 或 bounded bias；variance reduction；confidence interval |
| 仿真与真实校准 | 仿真 failure distribution 与真实/物理近失事件对齐 |
| 可认证输出 | failure rate、LoWC/NMAC risk、scenario criticality、policy-specific safety envelope |

### 4.4 与现有 Paper A/B/C 的关系

| 已有文章 | 在 Nature Communications 主线中的角色 |
|---|---|
| Paper A | 被测 conflict resolver / safety controller，不作为主贡献 |
| Paper B | 提供低空交通流、队列和密度条件，用于产生系统级风险状态 |
| Paper C | 提供感知不确定性、地图缺失和 3DGS 场景误差，用于构造 perception-induced risk |
| Paper D / F | 成为主论文核心：scenario coverage + accelerated rare-event safety validation |
| Paper G/I/J/K | 仅作为可选被测智能系统或工程支撑，不作为 Nature 级主贡献 |

### 4.5 数据来源

Nature Communications 级别不能只靠自研仿真。建议采用三层数据：

| 数据层 | 来源 | 作用 |
|---|---|---|
| Public safety proxy | NASA ASRS 数据库，包含航空前线人员和 UAS crew 的自愿安全报告 [14] |
| Public UAS reports | FAA UAS sighting reports / FOIA electronic reading room [15] |
| Air traffic baseline | OpenSky Network ADS-B / Mode S crowdsourced air traffic data [16] |
| Urban environment | OpenStreetMap / VGI 城市路网、建筑、POI 与语义功能区 [17] [18] |
| Controlled physical data | 室内/室外多 UAV testbed，注入非合作 UAV、通信延迟、定位噪声 |
| Simulation exposure | 自研低空 corridor/world generator，扩展到 10^7-10^8 等效暴露样本 |

这里要诚实：ASRS、FAA UAS sighting、OpenSky 都不是完美的低空 UAV fleet 数据。它们的作用是 **校准风险类型、空间分布和近失事件统计先验**。真正的系统级 failure rate 仍需要仿真和硬件在环补足。

### 4.6 实验设计

主实验不应写成 “我们方法更安全”，而应写成 “我们是否能可靠测量安全风险”。

| 实验 | 问题 | 成功标准 |
|---|---|---|
| Brute-force Monte Carlo 对照 | 加速估计是否无偏或可校准 | 在小规模可暴力枚举场景中，估计值落入 Monte Carlo 置信区间 |
| 加速倍数实验 | 是否真正缓解 rare-event curse | 在同等误差下样本量减少 10^3 级别以上 |
| 方差缩减实验 | estimator 是否稳定 | 多 seed 下 CI 更窄，variance reduction 显著 |
| 跨算法被测 | 对 A*/RRT*/ORCA/CBF/MAPPO 是否适用 | 不依赖单一 planner |
| 跨城市拓扑 | 是否跨城市泛化 | OSM-derived 多城市拓扑下保持校准 |
| 硬件在环 | 是否有现实锚点 | 真实多机/受控近失事件排序与仿真 criticality 一致 |
| 反事实验证 | 找到的危险场景是否真实关键 | 修改通信/风/路径等变量后，风险变化符合模型预测 |

### 4.7 评价指标

- accident / collision probability estimate；
- Loss of Well Clear rate；
- Near Mid-Air Collision proxy；
- variance reduction ratio；
- acceleration factor；
- calibration error；
- effective sample size；
- confidence interval width；
- scenario naturalness；
- failure-mode coverage；
- sim-to-real rank correlation。

### 4.8 致命风险

| 风险 | 严重性 | 缓解 |
|---|---|---|
| 没有真实数据锚点 | 致命 | 先拿 ASRS/FAA/OpenSky proxy + 自建物理 testbed |
| 加速采样有偏 | 致命 | 小规模 brute-force 校准 + 理论 estimator 修正 |
| 危险场景不 naturalistic | 高 | 用真实报告和城市结构约束采样分布 |
| 只证明某个 planner 不安全 | 高 | 至少评估 5 类 planner / policy |
| 论文被看成仿真 benchmark | 致命 | 主线必须是 risk measurement，不是 benchmark 排名 |

---

## 5. 第二候选：低空交通容量相变与拥堵崩溃规律

### 5.1 建议题目

**Capacity Phase Transitions in Autonomous Low-Altitude Traffic Networks**

中文可写为：

**自主低空交通网络中的容量相变与拥堵崩溃规律**

### 5.2 核心科学问题

Paper B 当前是一个三层调度系统。要冲 Nature Communications，必须从“调度器”转成“复杂系统规律”：

> 低空交通网络是否存在从自由流到拥堵崩溃的临界点？这个临界点由 demand intensity、vertiport capacity、charging capacity、corridor separation、communication reliability 哪些变量共同决定？是否存在跨城市拓扑可复现的 scaling law？

这类似地面交通的 traffic fundamental diagram，但对象变成了三维低空 corridor + vertiport + charging + fleet scheduling。

### 5.3 可能的科学发现

| 科学命题 | 需要验证的形式 |
|---|---|
| 存在低空交通临界负载 | 当 demand/capacity 超过阈值，backlog、delay、LoWC risk 非线性上升 |
| bottleneck 主导机制可切换 | 低负载由 demand 主导，中负载由 charging 主导，高负载由 corridor/vertiport 主导 |
| 三层调度改变相变点 | H-LyraUAV 不只是更优，而是扩展稳定区域 |
| 多模态 fallback 改变临界行为 | UAV-ground transfer 把 abrupt collapse 变成 smoother degradation |
| 城市拓扑影响临界指数 | 网格、放射、带状、高速走廊城市具有不同 capacity scaling |

### 5.4 实验最低门槛

- 10 / 20 / 50 / 100 / 200 / 500 / 1000 UAV 连续尺度扫描；
- 5 / 10 / 20 / 50 vertiports；
- low / medium / peak / shock demand；
- 多城市 OSM 拓扑；
- charging capacity、corridor separation、communication degradation 扫描；
- 至少一个真实 OD proxy：NYC TLC、Chicago taxi、物流订单 proxy、应急事件 proxy；
- 报告 phase diagram，而不是单一性能表。

### 5.5 为什么这可能够 Nat Commun

它不再是 “我们提出 H-LyraUAV”，而是：

> 我们发现自主低空交通系统的稳定运行存在可预测的容量边界和相变规律，并给出了可解释的队列稳定理论与跨城市实证验证。

如果这个规律能被不同城市、不同调度策略、不同交通模式复现，就有 Nature Communications 可能。

### 5.6 致命风险

最大风险是结果退化成经典 queueing / network flow 已知结论。  
如果只是 “负载越大延误越大”，没有新意。必须证明低空系统因为 3D separation、charging、vertiport、multimodal transfer、communication degradation 的耦合，产生了地面交通或传统排队网中没有被充分描述的临界行为。

---

## 6. 高风险第三候选：低空具身集群的能量-智能协同标度律

### 6.1 建议题目

**Energy-Intelligence Scaling Laws in Embodied Low-Altitude Swarms**

中文可写为：

**低空具身集群中的能量-智能协同标度律**

### 6.2 为什么不是普通 K / I / J

单独的 K 是推理加速，默认是系统工程。  
单独的 I 是 aerial VLA/VLN，默认是机器人/AI 工程。  
单独的 J 是 LowAltitudeGPT 微调，默认是领域模型工程。

但如果把它们组合成科学问题，就有一个可能：

> 在低空多 UAV 具身系统中，任务成功率、集体协调质量、通信开销、推理延迟和能耗之间是否存在可复现的 Pareto 前沿或 scaling law？

这条路的 Nature Communications 机会来自 “law”，不是来自 “LLM”。

### 6.3 可测变量

| 变量 | 例子 |
|---|---|
| 系统规模 | UAV 数量 N = 5-500 |
| 智能资源 | 模型大小、token budget、planning horizon、tool-call depth |
| 能源资源 | onboard compute power、communication energy、flight energy |
| 任务质量 | success、delay、LoWC risk、coverage、mapping quality |
| 架构策略 | cloud-only、edge-cloud split、onboard fallback、hybrid agent |

### 6.4 最低证据门槛

- 至少三类任务：冲突消解、应急调度、主动感知；
- 至少三类部署：cloud GPU、edge workstation、onboard Jetson / embedded GPU；
- 至少四个模型尺度：small / 8B / 14B / 32B / API teacher；
- 跨任务绘制 energy-intelligence Pareto frontier；
- 给出理论解释：为何某些 split policy 接近下界；
- 至少小规模真实 UAV 或硬件在环验证。

### 6.5 严格评审判断

这是高风险方向。当前还不能直接开写 Nature Communications。它适合作为 12-24 个月后的战略方向，前提是我们先有：

1. Paper A/B/C/D/G 工具链；
2. 可记录的 cloud-brain workload；
3. 真实或半真实 UAV 硬件平台；
4. 完整能耗与延迟测量。

---

## 7. 不建议单独冲 Nature Communications 的方向

| 方向 | 严格判断 | 更合适去处 |
|---|---|---|
| Paper A：PPO/MAPPO 无冲突规划 | 工程算法，除非并入 D 的风险测量 | T-ITS / T-RO / ICRA / IROS |
| Paper C：FIM-3DGS 主动感知 | 强方法论文，但 Nature 级需跨域信息理论规律 | T-RO / ICRA / IROS / CVPR 相关 |
| Paper G：CloudBrain-Agent | 系统集成，容易被认为是 agent hype | AAAI / IJCAI / T-ITS |
| Paper I：Aerial VLA/VLN | 普通版本是具身导航工程 | CoRL / RSS / ICRA / IROS |
| Paper J：LowAltitudeGPT | 垂类微调，不是科学发现 | T-ITS / Applied Intelligence / AAAI workshop |
| Paper K：推理加速 | 系统优化，不是低空科学本身 | MLSys / SenSys / TMC / IoT Journal |

这些方向不是不值得做，而是 **不应以 Nature Communications 为直接目标**。它们应作为 D/B/B+K 主线的支撑工具或对口工程论文。

---

## 8. 推荐执行路线

### 8.1 立即执行：Nature Communications 预研包

优先做 D：低空 rare-event safety measurement。

第 1-2 周：

- 整理 ASRS / FAA UAS sighting / OpenSky / OSM 数据可用性；
- 定义低空 rare-event taxonomy；
- 从现有 7600 万次探索日志中提取 failure modes；
- 确认是否能做小规模物理多机 testbed。

第 3-6 周：

- 实现 brute-force Monte Carlo 小规模 ground truth；
- 实现 importance sampling / adversarial naturalistic sampling；
- 证明 estimator 的 bias / variance / confidence interval；
- 对 A*/RRT*/ORCA/CBF/MAPPO 做首轮风险估计。

第 7-10 周：

- 跨城市 OSM 拓扑复现实验；
- 加入 communication degradation、wind、positioning error；
- 与 FAA/ASRS event taxonomy 做分布对齐；
- 做 hardware-in-loop 或小规模实飞。

第 11-16 周：

- 形成 Nature Communications 风格 narrative：
  - rare-event safety 是低空自主系统部署瓶颈；
  - 本文提出可校准加速测量方法；
  - 能以 10^3 级别样本效率估计多 UAV 失效风险；
  - 方法跨 planner、城市、扰动类型保持校准；
  - 结果给出低空安全认证启示。

### 8.2 同步准备：B 的容量相变版本

Paper B 继续按 TR-C 推进，但实验要多留一套 Nature Communications 数据：

- 做连续密度扫描；
- 记录 backlog / delay / risk 的非线性变化；
- 画 phase diagram；
- 分析 stable/unstable boundary；
- 尝试提取 scaling exponent；
- 比较 UAV-only、ground-only、multimodal fallback 的临界行为。

### 8.3 不建议做的事

- 不要把 CloudBrain-Agent 包装成 Nature 级；
- 不要写 “低空 AGI” 作为核心卖点；
- 不要承诺从零训练低空 foundation model；
- 不要只用仿真 claiming certifiable safety；
- 不要把 3DGS 主动感知单独包装成 Nature Communications，除非补理论边界和真实外场。

---

## 9. Nature Communications 版本的论文骨架

建议先以 D 为主线写预稿。

### 9.1 Abstract

问题：低空自主系统部署受限于稀有安全关键事件难以观测和验证。  
方法：提出可校准的 rare-event accelerated safety measurement framework。  
结果：在多个低空空域、planner、扰动条件下，以显著更少样本估计 collision / LoWC / near-miss risk。  
意义：为 autonomous low-altitude airspace safety certification 提供可复现测量方法。

### 9.2 Introduction

叙事链：

1. 低空经济依赖大量 UAV 安全运行；
2. 安全关键事件稀有，直接测试成本不可接受；
3. 自动驾驶已有 accelerated validation 先例，但低空空域更高维、更非结构化；
4. 现有 UAV 仿真/规划/调度工作缺少可校准风险测量；
5. 本文提出低空 rare-event safety measurement，并验证其统计有效性。

### 9.3 Methods

- Low-altitude event space；
- Naturalistic prior；
- Criticality function；
- Accelerated sampling distribution；
- Risk estimator；
- Confidence interval；
- Sim-to-real calibration；
- Planner/policy under test。

### 9.4 Results

| Figure | 内容 |
|---|---|
| Fig. 1 | 低空 rare-event safety measurement 框架总图 |
| Fig. 2 | rare-event taxonomy 与真实报告/仿真事件对应 |
| Fig. 3 | 小规模 brute-force 校准：估计值 vs Monte Carlo ground truth |
| Fig. 4 | 加速倍数与方差缩减 |
| Fig. 5 | 跨 planner 风险测量 |
| Fig. 6 | 跨城市/扰动泛化 |
| Fig. 7 | 硬件在环或真实近失排序验证 |

### 9.5 Discussion

必须讨论：

- 方法不能替代真实监管认证，但可显著提高预认证测试效率；
- 仿真分布与真实世界存在偏差；
- ASRS/FAA 数据是 voluntary reporting，存在 selection bias；
- 未来需要真实低空运营日志；
- 对低空监管、场景库、UAV planner benchmarking 的意义。

---

## 10. 最终判断

当前最值得投入 Nature Communications 预研的是：

1. **主线 D：可认证低空 rare-event safety measurement。**
2. **副线 B：低空交通 capacity phase transition / congestion collapse law。**
3. **远期高风险 B+K：低空具身集群 energy-intelligence scaling law。**

当前不建议直接冲 Nature Communications 的是：

- A 的 PPO/MAPPO conflict resolver；
- C 的普通 FIM-3DGS NBV；
- G 的 CloudBrain-Agent；
- I 的普通 aerial VLA/VLN；
- J 的 LowAltitudeGPT；
- K 的普通推理加速。

一句话：

> Nature Communications 级论文不能写成“我们做了一个更强的低空 UAV 系统”，必须写成“我们发现并验证了低空自主系统安全、容量或能耗智能之间的可证伪规律”。

---

## 11. 参考文献

[1] Nature. *Editorial criteria and processes.* URL: <https://www.nature.com/nature/for-authors/editorial-criteria-and-processes>

[2] Nature Communications. *Aims & Scope.* URL: <https://www.nature.com/ncomms/aims>

[3] Shuo Feng, Xintao Yan, Haowei Sun, Yiheng Feng, and Henry X. Liu. “Intelligent Driving Intelligence Test for Autonomous Vehicles with Naturalistic and Adversarial Environment.” *Nature Communications*, 12:748, 2021. DOI: 10.1038/s41467-021-21007-8. URL: <https://www.nature.com/articles/s41467-021-21007-8>

[4] Shuo Feng et al. “Dense Reinforcement Learning for Safety Validation of Autonomous Vehicles.” *Nature*, 615:620-627, 2023. DOI: 10.1038/s41586-023-05732-2. URL: <https://www.nature.com/articles/s41586-023-05732-2>

[5] Shuo Feng et al. “Curse of Rarity for Autonomous Vehicles.” *Nature Communications*, 15:4808, 2024. DOI: 10.1038/s41467-024-49194-0. URL: <https://www.nature.com/articles/s41467-024-49194-0>

[6] Da Zhao, Haobo Luo, Yuxiao Tu, Chongxi Meng, and Tin Lun Lam et al. “Snail-Inspired Robotic Swarms: A Hybrid Connector Drives Collective Adaptation in Unstructured Outdoor Environments.” *Nature Communications*, 15:3647, 2024. DOI: 10.1038/s41467-024-47788-2. URL: <https://www.nature.com/articles/s41467-024-47788-2>

[7] Alessandro Nitti, Marco D. de Tullio, Ivan Federico, and Giuseppe Carbone et al. “A Collective Intelligence Model for Swarm Robotics Applications.” *Nature Communications*, 16:6572, 2025. DOI: 10.1038/s41467-025-61985-7. URL: <https://www.nature.com/articles/s41467-025-61985-7>

[8] Jinpeng Hu, Wei Wang, Yuxiao Liu, and Jing Zhang. “Large Model in Low-Altitude Economy: Applications and Challenges.” *Big Data and Cognitive Computing*, 10(1):33, 2026. DOI: 10.3390/bdcc10010033. URL: <https://www.mdpi.com/2504-2289/10/1/33>

[9] Maximilian Adang, JunEn Low, Ola Shorinwa, and Mac Schwager. “SINGER: An Onboard Generalist Vision-Language Navigation Policy for Drones.” arXiv:2509.18610, 2025. URL: <https://arxiv.org/abs/2509.18610>

[10] Hengxing Cai et al. “FlightGPT: Towards Generalizable and Interpretable UAV Vision-and-Language Navigation with Vision-Language Models.” *EMNLP*, 2025. DOI: 10.18653/v1/2025.emnlp-main.338. URL: <https://aclanthology.org/2025.emnlp-main.338/>

[11] Pranav Saxena, Nishant Raghuvanshi, and Neena Goveas. “UAV-VLN: End-to-End Vision Language guided Navigation for UAVs.” arXiv:2504.21432, 2025. URL: <https://arxiv.org/abs/2504.21432>

[12] Moo Jin Kim et al. “OpenVLA: An Open-Source Vision-Language-Action Model.” arXiv:2406.09246, 2024. URL: <https://arxiv.org/abs/2406.09246>

[13] Anthony Brohan et al. “RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control.” arXiv:2307.15818, 2023. URL: <https://arxiv.org/abs/2307.15818>

[14] NASA. *Aviation Safety Reporting System Database Online.* URL: <https://asrsdbol.arc.nasa.gov/>

[15] Federal Aviation Administration. *Unmanned Aerial System (UAS) & Small Unmanned Aerial System (sUAS) FOIA Electronic Reading Room.* URL: <https://www.faa.gov/foia/electronic_reading_room/uas>

[16] Martin Strohmeier, Xavier Olive, Jannis Lübbe, Matthias Schäfer, and Vincent Lenders. “Crowdsourced Air Traffic Data from the OpenSky Network 2019-20.” *Earth System Science Data*, 2021. URL: <https://essd.copernicus.org/articles/13/357/2021/>

[17] Michael F. Goodchild. “Citizens as Sensors: The World of Volunteered Geography.” *GeoJournal*, 69:211-221, 2007. DOI: 10.1007/s10708-007-9111-y.

[18] Geoff Boeing. “OSMnx: New Methods for Acquiring, Constructing, Analyzing, and Visualizing Complex Street Networks.” *Computers, Environment and Urban Systems*, 65:126-139, 2017. DOI: 10.1016/j.compenvurbsys.2017.05.004.

---

## 12. 附录：本次执行计划

1. 先不改 A/B/C 当前投稿路线，继续把它们按 TR-C/T-ITS/T-RO/ICRA 做扎实。
2. 以 Paper D 为 Nature Communications 预研主线，重命名为 low-altitude rare-event safety measurement。
3. 同步改造 Paper B 的实验，让它输出 capacity phase diagram 和 congestion collapse evidence。
4. 暂缓单独写 LowAltitudeGPT / CloudBrain-Agent / AerialVLA 的 Nature 版本，只把它们作为被测智能系统或数据生成工具。
5. 两周内完成数据可得性审计：ASRS、FAA UAS sightings、OpenSky、OSM、现有 7600 万次探索日志、可用硬件平台。
6. 四周内完成一个小规模 proof-of-concept：brute-force Monte Carlo vs accelerated estimator。
7. 如果估计器无法校准，立即降级为 T-ITS / T-RO 安全测试论文，不继续冲 Nature Communications。
