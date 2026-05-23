---
title: "Low-altitude UAV research blog cultural roadmap: a complete plan from blog to journal"
description: "Systematically sort out the research value of 18 low-altitude UAV-related articles in the blog, identify the five directions with the greatest publication potential, and provide their respective innovation point statements, target journals, supplementary experiment lists, and suggested timelines."
pubDate: 2026-05-15
tags: ["Thesis planning", "Research roadmap", "UAV", "low altitude", "Submission strategy", "T-ITS", "ICRA"]
category: Tech
---

# Low Altitude UAV Research Blog Cultural Roadmap: A complete plan from blog to journal

> This article is not a technical introduction, but a **research management document**: re-examine the blog content accumulated in the past and find out which ones are worth publishing in journals, which ones are still missing, and which ones need to be tested from scratch. It is also a reckoning with one’s own research context.

---

## 0. Background and starting point

At present, the blog has accumulated **27 articles**, including 18 core articles related to low-altitude UAV, covering path planning, conflict resolution, multi-machine scheduling, perception reconstruction, digital twins, LLM/VLM planning and other directions.

Published paper basis: Journal of Advanced Transportation (SCI Q3), Q-learning for highway ramp control (DOI: 10.1155/2023/4771946), which established the research tone of "Reinforcement Learning × Traffic System".

**Objective of this article:**

1. Identify the 5–6 directions in blog content that are most valuable for publication
2. Provide actionable information for each direction: statement of innovation points, differences from existing work, target journals/conferences, list of supplementary experiments, and timeline of suggestions
3. Provide an overall submission roadmap for 12 months
4. Make this document a living research management tool (the version number is reflected in the file name)

---

## 1. Blog content panoramic map

### 1.1 Three major research lines

```
主线一：路径规划 × 冲突消解 × 多机调度
├── uav-urban-route-planning        （路径规划算法综述）
├── uav-conflict-resolution         （CD&R 机制综述+架构）
├── uav-conflict-env-construction   （仿真环境工程）
├── marl-kat-uav-conflict ★         （KAT MARL 框架）
├── large-scale-uav-scheduling ★    （三层百机调度）
└── urban-uav-3d-spatial-modeling   （3D空域建模参考）

主线二：感知 × 环境重建 × 数字孪生
├── uav-digital-twin-semantic-mapping ★  （五层数字孪生）
├── uav-semantic-mapping-functional-zoning ★（多源语义融合）
├── uav-nerf-gs-planning                 （NeRF/3DGS规划集成）
├── next-best-view-nerf-3dgs ★           （信息论NBV）
├── information-theory-active-perception （理论基础）
└── uav-multimodal-sim-data-synthesis    （多模态仿真工程）

主线三：LLM/VLM × 语义规划 × 形式验证
├── llm-uav-semantic-planning ★          （LTL/STL形式验证）
├── llm-guided-uav-planning-frontiers    （规划前沿概念）
├── hierarchical-vlm-uav-planning        （分层VLM架构）
└── vlm-uav-navigation-foundations       （VLN综述）

延伸：地面交通
├── carla-sumo-rl-lane-change ★          （PPO变道，已有实验）
└── traffic-signal-control               （信号控制反思）
```

★ = Paper candidates analyzed in this article

### 1.2 Maturity Assessment Summary List| Article | Theoretical Framework | Experimental Support | Comprehensive Maturity | Thesis Feasibility |
|------|----------|----------|----------|-----------|
| marl-kat-uav-conflict | ★★★★★ | ★★☆☆☆ | ★★★★☆ | High (just make up the experiment) |
| large-scale-uav-scheduling | ★★★★☆ | ★★☆☆☆ | ★★★☆☆ | High (complementary scale experiments) |
| next-best-view-nerf-3dgs | ★★★★★ | ★★★☆☆ | ★★★★☆ | High (complement online experiment) |
| uav-semantic-mapping-functional-zoning | ★★★★☆ | ★★☆☆☆ | ★★★☆☆ | Medium (Supplementary GIS data) |
| llm-uav-semantic-planning | ★★★★☆ | ★☆☆☆☆ | ★★★☆☆ | Medium (supplementary evaluation data set) |
| carla-sumo-rl-lane-change | ★★★☆☆ | ★★★★☆ | ★★★★☆ | High (experimented) |

---

## 2. Tier 1: Highest publication potential (recommended to submit within 6–12 months)

### Paper A: Large-Scale Urban UAV Conflict Resolution—KAT-MARL Framework

**Source article:** `marl-kat-uav-conflict` + `uav-conflict-resolution` + `uav-conflict-env-construction`

**Target journal:** IEEE Transactions on Intelligent Transportation Systems (T-ITS, SCI Q1, IF ≈ 8.5)

#### Core innovation point (Novelty Claim)

The **KAT (Knowledge-Attention-Transfer) framework** is proposed to replace explicit message passing with graph attention network (GAT) to achieve implicit multi-machine coordination without communication constraints:- **Implicit communication mechanism:** Each UAV only observes the neighborhood status, and automatically extracts the most relevant neighbor information through GAT's attention weight, without broadcasting messages
- **CTDE training paradigm:** Centralized training (Critic accesses global state) + decentralized execution (Actor only uses local observations)
- **ORCA covers the bottom layer:** The two-level security guarantee of learning strategy and geometric analysis method (ORCA) ensures strict no collision

Core formula system:

GAT attention weight:
$$e_{ij} = \text{LeakyReLU}\!\left(\mathbf{a}^\top[\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_j]\right)$$

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k\in\mathcal{N}_i}\exp(e_{ik})}$$

Aggregate neighbor information:
$$\mathbf{h}_i' = \sigma\!\left(\sum_{j\in\mathcal{N}_i}\alpha_{ij}\mathbf{W}\mathbf{h}_j\right)$$

QMIX centralized value functions:
$$Q_{tot}(\boldsymbol{\tau}, \mathbf{a}) = f_\theta\!\left(Q_1(\tau_1, a_1),\ldots,Q_N(\tau_N, a_N),\mathbf{s}\right)$$

The weight of $f_\theta$ is non-negative (monotonicity constraint), ensuring the IGM (individual-global maximization) condition.

#### Differentiation from existing work

| Method | Communication Requirements | Scale | Real-time | Security Guarantee |
|------|---------|------|--------|---------|
| MADDPG | None | <20 | Bad | None |
| QMIX | None | <20 | Medium | None |
| CommNet | Full Broadcast | <50 | Poor | None |
| ORCA | None | Large | Excellent | Yes |
| **KAT (this article)** | **None** | **50+** | **Good** | **Yes (double layer)** |

#### Supplementary experiment list- [ ] **Scale ablation:** 20 / 50 / 100 UAVs are trained + tested separately, and the success rate, average delay, and calculation delay are recorded
- [ ] **Baseline comparison:** ORCA-only, MADDPG, QMIX (without GAT), QMIX+GAT (with GAT and without ORCA cover)
- [ ] **Scenario:** Construct a simulation map based on the real road network of Shanghai Lujiazui or Beijing CBD
- [ ] **Indicators:** Mission success rate (goal achievement rate), average additional delay (seconds), conflict rate (conflicts/UAV/minute), inference delay (ms)
- [ ] **Visualization:** Attention weight heat map, showing the pattern of UAV paying attention to neighbors

#### Timeline

```
2026/06  搭建仿真环境（基于 existing uav-conflict-env-construction）
2026/07  训练 KAT 模型 + 基线对比实验
2026/08  写稿（Introduction / Method / Experiment / Conclusion）
2026/09  内部审阅 + 语言润色
2026/09  投稿 IEEE T-ITS（Regular Paper，通常 3–6 个月审回）
```

---

### Paper B: Three-layer hierarchical scheduling system for hundreds of drones

**Source article:** `large-scale-uav-scheduling` + `uav-urban-route-planning`

**Target journal:** IEEE T-ITS or Transportation Research Part C (SCI Q1, IF ≈ 7.6)

#### Core innovation points

A **three-layer hierarchical architecture** is proposed to decompose the urban scheduling problem of 100+ UAVs into three sub-problems that can be independently optimized and operated collaboratively:

**Macroscopic layer (task allocation):** GNN encoded airspace map state + ACO (ant colony optimization) allocates tasks to UAVs to optimize global throughput

Macro level objective function:
$$\min\;\sum_{k=1}^{N}\!\left(w_1 T_k + w_2 \mathcal{E}_k\right) + w_3\cdot\text{Congestion}(G)$$

**Meso layer (conflict coordination):** QMIX multi-agent coordination, speed/height adjustment based on macro path to resolve conflicts

Meso-level decentralized decision-making, local strategy for each UAV:
$$\pi_k(a_k \mid \tau_k) = \text{softmax}(Q_k(\tau_k, \cdot;\theta_k))$$

**Micro layer (trajectory execution):** ORCA geometric analysis + MPC rolling optimization to achieve centimeter-level accurate trackingMPC rolling optimization (prediction step size $H$):
$$\min_{\mathbf{u}_{0:H-1}}\sum_{t=0}^{H-1}\!\left\|\mathbf{x}_t - \mathbf{x}_{ref}\right\|_Q^2 + \|\mathbf{u}_t\|_R^2$$

#### Supplementary experiment list

- [ ] **Scale expansion curve:** 20/50/100/200 UAV, recording system throughput (UAV/min), end-to-end latency, computing resources (CPU/GPU)
- [ ] **Baseline comparison:** FCFS (first come first served), centralized MILP (optimal but slow), two-tier architecture (no macro layer)
- [ ] **Scenario diversity:** High-density logistics scenario (uniform demand) vs sudden peak scenario (Poisson arrival)
- [ ] **Theoretical analysis:** Gives the theoretical derivation of the upper bound of system throughput (based on queuing theory)

#### Timeline

```
2026/07  实现三层框架代码 + 集成测试
2026/08  规模扩展实验（需要较长训练时间）
2026/10  写稿
2026/11  投稿 Transportation Research Part C
```

---

### Paper C: Information theory-driven 3DGS active sensing planning

**Source article:** `next-best-view-nerf-3dgs-exploration` + `information-theory-active-perception-foundations` + `uav-nerf-gs-planning`

**Target Conference:** ICRA 2026 (ends approx. 2026/09) or IROS 2026

#### Core innovation points

Use **Fisher Information Matrix (FIM)** as the proxy target selected by Next-Best-View to drive **3D Gaussian Splatting (3DGS)** active convergence reconstruction:

**Information gain quantification:** The next viewpoint $\mathbf{v}^*$ is chosen to maximize the expected information gain with respect to the scene parameter $\boldsymbol{\Theta}$:

$$\mathbf{v}^* = \arg\max_{\mathbf{v}\in\mathcal{V}_{free}} \mathcal{I}(\boldsymbol{\Theta}; \mathbf{y}_\mathbf{v})$$Using the Cramér-Rao lower bound, the inverse FIM matrix gives a lower bound on the parameter estimate uncertainty:

$$\text{Cov}(\hat{\boldsymbol{\Theta}}) \succeq \mathbf{F}(\boldsymbol{\Theta})^{-1}$$

** Differentiable approximation of 3DGS FIM: ** For each Gaussian $\mathcal{G}_i$, its FIM with respect to the mean $\boldsymbol{\mu}_i$ can be approximated as:

$$\mathbf{F}_i(\boldsymbol{\mu}_i) \approx \sum_{\mathbf{r}\in\mathcal{R}(\mathbf{v})} \frac{\partial \hat{C}(\mathbf{r})}{\partial \boldsymbol{\mu}_i}\!\left(\frac{\partial \hat{C}(\mathbf{r})}{\partial \boldsymbol{\mu}_i}\right)^\top \frac{1}{\sigma_n^2}$$

**Greedy strategy real-time:** The global optimal NBV search is NP-hard, using greedy serialization + pruning (distance constraint + occlusion detection) to achieve real-time decision-making (<50 ms/step).

#### Comparison with existing methods

| Method | Objective function | Expression | Real-time | Information assurance |
|------|---------|------|--------|---------|
| Frontier | Coverage | Voxels | Good | None |
| Entropy Minimization | Occupied Entropy | Voxels | Medium | Weak |
| ActiveGAMER | Reconstruction Quality | 3DGS | Poor | None |
| **This article (FIM-3DGS)** | **Fisher information** | **3DGS** | **Good** | **CRB theoretical guarantee** |

#### Supplementary experiment list- [ ] **Online reconstruction experiment:** AirSim urban scene, UAV autonomous flight + online 3DGS update
- [ ] **Metrics:** PSNR/SSIM (reconstruction quality), coverage (%), average information gain per step, total flight distance
- [ ] **Baseline:** Random exploration, Frontier-based, ActiveGAMER, SO-NeRF
- [ ] **Ablation:** FIM proxy target vs pure coverage target vs pure reconstruction quality target

#### Timeline

```
2026/06  实现 FIM-3DGS 可微近似模块
2026/07  AirSim 在线实验
2026/08  写稿（ICRA 格式，8页）
2026/09  投稿 ICRA 2026
```

---

### Paper D: Multi-source semantic fusion + functional partition-driven UAV trajectory planning

**Source article:** `uav-semantic-mapping-functional-zoning` + `uav-digital-twin-semantic-mapping`

**Target Journal:** IEEE T-ITS or Transportation Research Part C

#### Core innovation points

**Multi-source data fusion pipeline:**

$$\mathcal{M}_{semantic} = \mathcal{F}_{fusion}(\mathcal{I}_{RS},\; \mathcal{G}_{OSM},\; \mathcal{P}_{POI},\; \mathcal{D}_{census})$$

Among them, $\mathcal{I}_{RS}$ is the semantic segmentation result of remote sensing images, $\mathcal{G}_{OSM}$ is the road/building GIS vector, $\mathcal{P}_{POI}$ is the point of interest (business/hospital/school), and $\mathcal{D}_{census}$ is the demographic data.

**Urban functional zoning risk model:**

Define the basic risk coefficient $\lambda_z$ for each functional area type $z \in \{\text{residential}, \text{commercial}, \text{industrial}, \text{green space}, \text{water}\}$, combining the time period factor $\delta(t)$ (morning and evening peak vs. night) and floor density $\rho_{bld}$:$$\mathcal{R}(x, y, t) = \lambda_{z(x,y)} \cdot \delta(t) \cdot \rho_{bld}(x,y)$$

**Risk-aware route cost function:**

Embedding the functional partition risk graph into A* edge weights:

$$d(u,v) = \ell_{uv}\cdot\!\left(1 + \beta_1\mathcal{R}_{uv} + \beta_2 TI_{uv}\right)$$

Where $TI_{uv}$ is the corridor turbulence intensity (extracted from the wind field model), $\beta_1, \beta_2$ are trade-off coefficients.

**Differentiation from existing work:**
- Existing work using **population density** as a ground risk proxy → static, coarse-grained
- This article uses **Functional zoning type × time period factor × building density** three-dimensional risk model → dynamic, fine-grained, and can be migrated across cities (unified functional zoning standards)

#### Supplementary experiment list

- [ ] **Data acquisition:** Guangzhou/Shenzhen CBD GIS data (OSM open source + high-resolution remote sensing images)
- [ ] **Baseline comparison:** Pure shortest path (Dijkstra), population density weighting, building occlusion weighting
- [ ] **Indicators:** Risk exposure points (REI = $\int \mathcal{R}(\boldsymbol{\xi}(t))\,\mathrm{d}t$), path length, flight time
- [ ] **Pareto curve:** REI vs path length trade-off front
- [ ] **Generalization experiment:** Training weight parameters in Beijing/Shanghai, testing in Guangzhou (cross-city transferability)

#### Timeline

```
2026/07  GIS 数据采集与预处理
2026/08  功能分区模型实现 + 航路规划实验
2026/09  写稿
2026/11  投稿 Transportation Research Part C
```

---

## 3. Tier 2: Requires more additional work (12–18 months)

### Paper E: UAV mission planning with LLM + formal verification

**Source article:** `llm-uav-semantic-planning` + `llm-guided-uav-planning-frontiers`

**Target:** ICRA/IROS or IJCAI 2027

#### Core innovation points

**Closed loop pipeline:**

```
自然语言任务描述
       ↓ LLM 转译
LTL/STL 形式规范
       ↓ 模型检测（NuSMV / Breach）
验证通过 → 执行
验证失败 → 反馈给 LLM → 迭代修正
```**LTL Specification Example ("Avoid flying over the hospital before reaching point B"):**

$$\varphi = \Box(\neg \text{Hospital}) \;\wedge\; \Diamond(\text{Waypoint}_B)$$

**Main Challenges:**
- Translation accuracy of LLM → LTL (requires construction of evaluation data set: natural language-form specification pair)
- Computational overhead of model checking in large state spaces (requires state space abstraction technology)
- LLM Hallucination leads to unsatisfiable specifications (requires pre-processing of satisfiability checks)

#### Supplementary work list

- [ ] Build UAV mission NL→LTL dataset (~500 pairs)
- [ ] Measure the translation accuracy of GPT-4o / Llama-3
- [ ] Implement the NuSMV interface to verify urban UAV scene specifications
- [ ] Design Hallucination detection + repair module

---

### Paper F: CARLA-SUMO multi-agent lane changing RL (ground extension)

**Source article:** `carla-sumo-rl-lane-change` (270k steps of PPO experimental results)

**Target:** Transportation Research Part C

#### Extension direction

- Current status: Single-agent PPO, converged at 270k steps
- Extension: multi-agent (5–10 cars changing lanes simultaneously) + uncertainty quantification (Dropout/Ensemble)
- Sim2Real: Validating policy generalization on nuScenes/Waymo datasets

---

## 4. Summary of key research gaps in various directions| Direction | Blog Status | Biggest Gap | Difficulty of Making Up |
|------|---------|---------|---------|
| Paper A (KAT-MARL) | Complete theoretical framework, clear equation derivation | Lack of large-scale simulation experimental data | ★★☆ (3–4 months) |
| Paper B (three-tier scheduling) | Clear architectural design and complete logic | Lack of 100+ scale expansion experiments | ★★★ (4–5 months) |
| Paper C (FIM-3DGS) | Deep information theory derivation and good understanding of 3DGS | Lack of online closed-loop implementation and experimentation | ★★★ (3–4 months) |
| Paper D (Functional Partition) | Clear multi-source integration logic | Lack of real GIS data and experiments | ★★☆ (3–4 months) |
| Paper E (LLM+formal verification) | Complete pipeline design | Missing evaluation data set, translation accuracy unknown | ★★★★ (6–8 months) |
| Paper F (CARLA lane change) | Experimental results available | Multi-agent scenarios need to be expanded | ★★☆ (3–4 months) |

---

## 5. Submission strategy and journal selection guide

### List of target journals/conferences

| Journal / Conference | Field | IF / Acceptance Rate | Review Cycle | Suitable for Paper |
|------------|------|------------|---------|-----------|
| **IEEE T-ITS** | Transportation Intelligence Systems | 8.5 / ~20% | 3–6 months | A, B, D |
| **TR Part C** | Transportation Science and Engineering | 7.6 / ~18% | April–June | B, D, F |
| **IEEE T-ASE** | Automation Science and Engineering | 5.9 / ~22% | 3–5 months | A |
| **IEEE RAL** | Robot Express | 4.6 / ~30% | 2–3 months | C |
| **ICRA** | Robot Conference | ~30% | Once a year | C, E |
| **IROS** | Robot Summit | ~40% | Once a year | C, E |
| **IJCAI** | AI Summit | ~15% | Once a year | E |

### Progressive submission path suggestionsBased on the basis of published SCI Q3 papers, the **incremental improvement** strategy is recommended:

```
阶段一（2026）：冲刺 Q1 期刊
  → Paper A → IEEE T-ITS（同赛道，优势最大）
  → Paper C → IEEE RAL 或 ICRA（快速发表）

阶段二（2026–2027）：扩展并提升
  → Paper B → Transportation Research Part C
  → Paper D → IEEE T-ITS（第二篇，建立系列感）

阶段三（2027–）：攻顶会
  → Paper E → ICRA/IROS 或 IJCAI（高风险高回报）
```

**Key Tips:**
- T-ITS has high acceptance of cross-research on "UAV × Urban Transportation System", which is consistent with the field of published papers, and reviewers have the highest recognition of the background.
- ICRA deadlines are usually in September of the previous year, so plan ahead
- It is recommended to preprint on arXiv before submission (increasing acceptance in the transportation field)

---

## 6. 12-month submission roadmap

```
时间        Paper A（KAT-MARL）     Paper C（FIM-3DGS）    Paper D（功能分区）     Paper B（三层调度）
─────────────────────────────────────────────────────────────────────────────────────────────────
2026/05    ▶ 环境搭建                ▶ FIM模块实现
2026/06    实验训练                  实验训练（AirSim）
2026/07    基线对比                  写稿启动              ▶ GIS数据采集
2026/08    写稿                      写稿完成              实验 + 写稿           ▶ 框架实现
2026/09    ◉ 投 T-ITS               ◉ 投 ICRA/RAL
2026/10                                                    写稿                  规模实验
2026/11                                                    ◉ 投 TR Part C
2026/12                                                                          写稿
2027/01                                                                          ◉ 投 TR Part C
─────────────────────────────────────────────────────────────────────────────────────────────────
◉ = 投稿节点   ▶ = 工作启动
```

---

## 7. Maintenance Agreement for this document

**File naming convention:** `research-roadmap_v{version number}_{year, month, day}.md`

- Current version: `research-roadmap_v1_20260515.md`
- Next update (after Paper A submission): `research-roadmap_v2_20260930.md`
- After receiving review comments: `research-roadmap_v3_202611xx.md`

**Modified content for each update:**
1. Corresponding to Paper’s Timeline (actual progress vs. plan)
2. Supplement the completion status of the experiment list (hit ✅)
3. Summary of review comments and response strategies
4. New paper opportunities (such as newly discovered research gaps)

> Use version management for the research plan itself because the direction of the research will be continuously adjusted with the emergence of experimental results, review comments, and new papers. This document should be living, not disposable.

---

**Appendix: Quick check on the correspondence between blog posts and Paper**| Blog Post | Corresponding Paper |
|---------|----------|
| marl-kat-uav-conflict | A (main) |
| uav-conflict-resolution | A (reference) |
| uav-conflict-env-construction | A (experimental environment) |
| large-scale-uav-scheduling | B (main) |
| uav-urban-route-planning | B (reference) |
| next-best-view-nerf-3dgs-exploration | C (main) |
| information-theory-active-perception | C (theoretical basis) |
| uav-nerf-gs-planning | C (reference) |
| uav-semantic-mapping-functional-zoning | D (main) |
| uav-digital-twin-semantic-mapping | D (reference) |
| llm-uav-semantic-planning | E (main) |
| llm-guided-uav-planning-frontiers | E (reference) |
| carla-sumo-rl-lane-change | F (main) |