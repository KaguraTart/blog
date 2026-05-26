---
title: "Research Roadmap v2: Comprehensive upgrade of top journal strategy and organization of low-altitude transportation paper groups"
description: "Under the Q1 top issue goal, the paper routes for low-altitude UAV, low-altitude transportation cloud brain, scene coverage, scheduling and formal planning are reorganized, and the short-term priorities, submission positioning, transportation system narrative boundaries and special planning entrances are clarified."
pubDate: 2026-05-15
updatedDate: 2026-05-23
tags: ["Thesis planning", "Research roadmap", "top publication strategy", "T-ITS", "TR Part C", "T-RO", "UAV", "low altitude"]
category: Tech
sourceHash: "1e0fdb9e8c70b09d0ee2f5dec5a08bf897624441"
---

# Research Roadmap v2: Comprehensive upgrade of top journal strategy and organization of low-altitude transportation paper groups

> **v1 → v2 trigger: ** The teacher explicitly requires that all papers must be published in SCI Q1 top issues (IF ≥ 7). v1 includes "rapid publication" paths such as RA-L (IF 4.6) and ICRA conferences, which have been moved up to the three top publication matrices of IEEE T-ITS, TR Part C, and IEEE T-RO.

---

## 0. v1 → v2 core changes overview

### 0.1 All submissions to journals have been moved up

| Paper | v1 Goal | v1 IF | **v2 Goal** | **v2 IF** | Upgrade Amount |
|-------|----------|-------|------------|-----------|---------|
| A: KAT-MARL conflict resolution | IEEE T-ITS | 8.5 | **IEEE T-ITS (Keep)** | 8.5 | — |
| B: Three-layer Scheduling | TR Part C | 8.5 | **TR Part C / T-ITS (Hold)** | 8.5 | — |
| **C: FIM-3DGS Active Sensing** | **RA-L/ICRA** | **4.6** | **IEEE T-ITS → TR-C** | **8.5** | **Major Upgrade** |
| D: Functional Partition Planning | T-ITS / TR-C | 8.5 | **TR Part C (maintained)** | 8.5 | — |
| **E: VERA-UAV Formal Language Planning** | **ICRA/IJCAI** | **Conference** | **AAAI first + T-ITS extension** | **Conference + 8.5** | Conference method first, journal extension later |
| **F: UAV safety critical scenario engineering** | **TR Part C** | **8.5** | **T-ITS first + TR-C emergency expansion** | **8.5** | New independent low-altitude safety test route |

### 0.2 Overall extension of timeline- v1: 12 month window (2026/05 – 2027/01), mainly due to RA-L fast track
- **v2: 24–30 month window (2026/06 – 2029/06)**, the review cycle of top journals is longer and the experiments need to be more solid

### 0.3 Workload increase estimate

| Paper | v1 workload | v2 workload | Increment reasons |
|-------|----------|---------|---------|
| A | March–April | June–August | Experimental scale from 50 → 200 UAV, queuing theory analysis |
| B | April–May | August–October | Multi-scenario generalization test + real map data |
| **C** | **3–4 months** | **12–15 months** | **Completely restructured into a low-level economy ITS paper** |
| D | March–April | June–August | Generalization of Gadot City + Actual Flight Case |
| **E** | **6–August** | **8–December** | **First do the AAAI method paper, and then expand it to the ITS system paper** |
| F | March–April | August–December | 76 million exploration log cleaning, coverage metric, accelerated testing, real high-speed emergency expansion |

### 0.4 2026-05-22 Calibration: Transportation Journal is not "storytelling", but a closed loop of system problems

This time the road map needs to be recalibrated. The transportation field does pay more attention to problem narrative and system significance than the pure algorithm field, but it cannot be understood as "just telling a round story". A more accurate standard is:

> Transportation papers should tell a credible system story, but this story must be supported by models, experiments, indicators, and boundary conditions.

Therefore, all subsequent plans that are partial to TR-C/T-ITS must be checked according to the following chain:

```text
真实交通系统问题
  -> 现实假设与边界条件
  -> 数学建模 / 运行机制
  -> 强 baseline 与消融
  -> 交通含义指标
  -> 敏感性 / 泛化 / 失败分析
  -> 对运行控制、规划设计或管理政策的启示
```

Not all papers need to use TR-C logic. The core of strong algorithm-driven AAAI / ICLR / robotics method papers is still the novelty of the algorithm, theoretical properties, benchmark difficulty and reproducibility. Only when the goal is TR-C / T-ITS / transportation journal, it is necessary to put "transportation system significance" into the main line.| Thesis | Main positioning | Whether to use a transportation system narrative | Current writing calibration |
|------|--------|--------------------|--------------|
| Paper A: KAT-MARL conflict resolution | T-ITS / Low-altitude traffic safety control | Yes, but the algorithm cannot be weakened | Changed from "New MARL algorithm" to "Verification of low-altitude conflict resolution system under communication degradation, non-cooperative UAV, high-density corridor" |
| Paper B: Three-layer scheduling of hundreds of UAVs | TR-C first | Strong need | Focus on capacity, delay, queue stability, vertiport/charging/corridor bottleneck and multimodal fallback |
| Paper C: FIM-3DGS Active Sensing | Algorithm + Traffic Enablement Technology | Conditionally required | If you vote for T-ITS/TR-C, you must prove that active sensing improves traffic task indicators such as inspection, emergency response, and distribution; otherwise, keep the robot perception algorithm paper |
| Paper D: Semantic functional area planning | TR-C / Urban low-altitude planning | Need | Focus on ODD, urban functional areas, risk exposure, planning suggestions, not pure semantic segmentation |
| Paper E: VERA-UAV | AAAI / Formal Language Planning | No forced application | First follow the AI planning / verification paper; follow-up ITS expansion plus traffic operation scenarios |
| Paper F: Scenario coverage and emergency | T-ITS + TR-C bifurcation | F-J1 is partially needed, F-J2 is strongly needed | F-J1 writes safety testing benchmark; F-J2 writes a traffic operation paper on Shandong expressway emergency resource allocation |
| Paper G/G1: Low-altitude traffic cloud brain LLM Agent | AAAI/IJCAI first, T-ITS extension | G1 is not mandatory, journal expansion is required | G1 maintains the agent/tool-use/verification method contribution; journal version supplement system indicators and operational inspiration |

The minimum experimental hardness requirements for the transportation journal version have also been uniformly increased:- At least 5 random seeds, the main table reports mean ± std or bootstrap confidence interval.
- Baseline cannot just put no-control/greedy, it must include strong classical methods, heuristic methods and learning methods in the problem field.
- Indicators cannot only report reward, accuracy, and success rate; they must include traffic meaning indicators such as conflict count, LoWC, NMAC, delay, extra distance, energy, throughput, resource utilization, and runtime.
- Generalization must be done: train low density and test high density, train small scale and test large scale, train fixed topology and test new topology, train cooperative traffic and test non-cooperative/degraded communication traffic.
- There must be a failure case analysis showing at what density, communication loss rate, non-cooperative behavior or resource bottleneck the system failed.

---

### 0.5 2026-05-23 Organization: reading order and priority of current planning documents

The current general roadmap is retained as the "Research Matrix Entry", and the specific implementation is subject to the B/E/F/G/G1 special document. The recommended reading order is as follows:| Priority | Documentation | Current positioning | Recent actions |
|--------|------|----------|----------|
| P0 | Paper G1: CloudBrain-Agent complete thesis plan | AAAI / IJCAI first | First implement the verifiable agent, CloudBrain-Bench, tool chain and main experiment |
| P1 | Paper B: Three-layer hierarchical scheduling of hundreds of UAVs | TR-C first | Build synthetic queuing benchmark, Lyapunov scheduler and strong baseline |
| P1 | Paper F: UAV safety critical scenario engineering | T-ITS first, TR-C emergency expansion | Complete first F-J1: coverage metric + accelerated testing |
| P2 | Paper E: VERA-UAV | AAAI method paper, subsequent expansion of T-ITS | Condensed into typed IR + LTL/STL + verifier repair, no major paper on transportation system first |
| P3 | Paper C / Paper D | Pending further data and task convergence | Retain the direction, but do not compete with B/F/G1 for recent experimental resources |

This edition requires special clarification: **The old Paper F = CARLA-SUMO multi-agent lane changing RL line is no longer counted in the current group of low-altitude UAV papers. ** If the ground autonomous driving direction is redone in the future, it can be restored as an independent ground transportation paper; currently Paper F refers specifically to UAV safety-critical scenario engineering.

The recommended execution sequence in the near future is:

1. Do G1 first, because it can unify Paper B’s scheduler, Paper E’s verifier, and Paper F’s scenario stress test into a “low-altitude traffic cloud brain” tool chain.
2. Start the synthetic benchmark of B simultaneously, because it is the core base for subsequent TR-C system papers and G1 scheduling tools.
3. F-J1 advances after having exploration logs and scene generation scripts to avoid falling into too many real application narratives at the beginning.
4. E Keep AAAI method papers and do not expand them into a large system of low-altitude transportation journals in advance.

---## 1. Blog content panoramic map (consistent with v1)

The three main lines of research remain unchanged (see v1 for details):
- Main line one: path planning × conflict resolution × multi-machine scheduling
- Main Line 2: Perception × Environment Reconstruction × Digital Twin
- Main Line 3: LLM/VLM × Semantic Planning × Formal Verification

---

## 2. Tier 1: Core top journal papers (within 24 months)

### Paper A: Large-Scale Urban UAV Conflict Resolution — KAT-MARL (Maintaining Top Issue Positioning)

**Target journal:** IEEE Transactions on Intelligent Transportation Systems (T-ITS, IF 8.5 Q1)

**Changes from v1:** Experimental scale upgrade, theoretical analysis expansion

#### v2 new requirements

- Experiment size from 100 UAV → **200 UAV** (to meet T-ITS’s preference for large-scale simulation)
- Added **queuing theory analysis**: prove the upper bound of the system throughput of the KAT framework
- Added **real road network mapping**: expanded from CBD simulation to 2–3 real cities (Shanghai Lujiazui, Beijing CBD, Shenzhen Futian)
- Added **robustness experiments**: communication delay, sensor noise, UAV failure scenarios

#### v2 Timeline
```
2026/06–07  实验环境搭建（基于 uav-conflict-env-construction）
2026/08–10  训练 KAT + 200 UAV 规模扩展实验
2026/11     真实城市路网泛化实验
2026/12     排队论理论分析与证明
2027/01–02  写稿（25 页 T-ITS 格式）+ 内部审阅
2027/03     ◉ 投稿 IEEE T-ITS
2027/09     收到审稿意见（4–6 月审回）
2027/12     接受目标
```

---

### Paper B: Three-tier hierarchical scheduling of hundreds of drones (maintaining top publication positioning)

**Target Journal:** Transportation Research Part C or IEEE T-ITS (IF 8.5 Q1)

**Changes from v1:** Add the mathematical foundation of queuing theory and add multi-modal transportation scenarios

#### v2 new requirements

- **Theoretical enhancements:** Queuing theory + Lyapunov stability proof
- **Multi-modal extension:** UAV + ground vehicle joint dispatch (enhance TR-C’s fit for transportation systems)
- **Real data:** Comparison with Meituan/JD unmanned delivery pilot data (if available)

#### v2 Timeline
```
2026/08–09  三层框架代码实现
2026/10–12  规模扩展实验（20/50/100/200 UAV）
2027/01     排队论与 Lyapunov 分析
2027/02–03  写稿
2027/04     ◉ 投稿 TR Part C
2027/10     接受目标
```

---

### Paper C: FIM-3DGS active sensing - **Major reconstruction (see v2 special document for details)**

**Target journal:** IEEE T-ITS (preferred) → TR Part C (preferred), IF 8.5 Q1**Reason for reconstruction:** v1 positioning RA-L is too low, the teacher requires top publication

**v2 core changes (see `paper-c-fim-3dgs-uav-active-perception_v2_20260515.md` for details): **

1. **Positioning upgrade:** From "Perception Algorithm Paper" → "Low Altitude Economic Enabling Technology"
2. **Evaluation expansion:** Single perception indicator → five-layer indicator system (perception/planning/task/system/economy)
3. **Case Study:** Three new transportation application cases (building inspection, last-mile delivery, emergency response)
4. **Experimental expansion:** Added SUMO + AirSim joint simulation + multi-UAV system-level experiment
5. **Dataset contribution:** Self-built UAV-Delivery-Dataset open source data set

#### v2 Timeline
```
2026/06–10  五阶段实验（核心算法 + 三案例 + 多机系统级）
2026/11–12  数据整合 + 初稿（22 页 T-ITS 格式）
2027/01–02  润色 + 内部审阅
2027/03     ◉ 投稿 IEEE T-ITS
2027/09     收到审稿意见
2027/12     接受 / 转 TR-C
2028/06     最终发表
```

For detailed Paper C planning, see [Paper C v2 special document](/blog/paper-c-fim-3dgs-uav-active-perception_v2_20260515/).

---

### Paper D: Multi-source semantic fusion + functional partition-driven UAV trajectory planning (maintaining top publication positioning)

**Target Journal:** Transportation Research Part C (IF 8.5 Q1)

**Changes from v1:** Multi-city generalization experiment expansion

#### v2 new requirements

- **Multi-city generalization:** Training + testing in 5 cities (Beijing, Shanghai, Guangzhou, Shenzhen, Wuhan)
- **Real flight case:** Cooperation with a UAV delivery pilot or public data reproduction
- **Risk quantification:** Introducing actuarial risk assessment (insurance/compensation perspective)

#### v2 Timeline
```
2026/07–09  GIS 数据采集（5 城市）
2026/10–12  功能分区模型 + 多城市实验
2027/01     真实飞行案例对比
2027/02–03  写稿
2027/04     ◉ 投稿 TR Part C
2027/10     接受目标
```

---

## 3. Tier 2: Top journal papers with greater technical challenges

### Paper E: VERA-UAV formal language planning (AAAI first, then ITS extension)

**v1 Target:** ICRA/IJCAI (Conference)

**Current target:** AAAI/IJCAI first, T-ITS extension backup**Reason for calibration:** The core contribution of Paper E is AI planning/verification, and it should not be forced to become a large and scattered transportation system paper for the sake of top publication. The AAAI version prioritizes answering "How natural language UAV tasks form executable safe trajectories via typed IR, LTL/STL, validator counterexamples, and symbolic fallbacks."

#### Current closing direction

- **Method main line:** NL instruction -> typed TaskIR -> LTL/STL -> verifier -> counterexample/robustness repair -> trajectory verification.
- **Theoretical Bounds:** Does not claim LLM completeness; prove relative completeness under assumptions of finite DSL, decidable verifier and complete underlying planner.
- **Experiment boundary:** The main experiment uses synthetic controlled benchmark; AirSim, real logistics, and multi-UAV ITS indicators will be put into subsequent expansion.
- **Submission strategy:** The main article of AAAI emphasizes methods, theories, benchmarks and strong baselines; T-ITS is expanded to include traffic operation indicators and real low-altitude scenarios.

#### v2 Timeline
```
2026/06–07  冻结 TaskIR DSL、任务生成器和验证器接口
2026/08–09  实现 Direct LLM / NL2LTL-style / LTLCodeGen-style / VERA-UAV baselines
2026/10     跑主实验、消融和泛化测试
2026/11     完成理论证明、图表和初稿
2026/12     ◉ 投稿 AAAI / IJCAI 对应批次
2027/03     根据结果扩展 T-ITS 版本
```

---

### Paper F: UAV safety-critical scenario engineering and emergency applications (replacing the old CARLA-SUMO line)

**Current Target:** F-J1 is the main candidate for IEEE T-ITS; F-J2 is the main candidate for TR-C.

**Positioning change:** The current Paper F no longer refers to the CARLA-SUMO lane change RL, but focuses on UAV safety-critical scenario engineering as a journal priority: first establish reproducible safety-critical scenario coverage and accelerated testing papers, and then extend the same platform to Shandong highway emergency rescue resource deployment.

#### Current new requirements- **Scene space:** Clearly define the 50m x 50m x 50m UAV test cell, obstacle combination, dynamic obstacles, wind field, visual area occlusion, no-fly zone and mission objectives.
- **Existing experimental assets:** 76 million exploration logs can only be written as "available basis" and cannot be written as final experimental results; they need to be cleaned into failure taxonomy, coverage holes and planner stress cases.
- **Method main line:** coverage metric -> coverage-guided sampler -> danger-validity filter -> accelerated testing -> cross-planner evaluation.
- **Strong baseline:** random generation, grid/LHS sampling, Bayesian optimization, CMA-ES, RL adversarial generation, Scenic-style constrained generation.
- **Traffic Expansion:** F-J2 only introduced Shandong Highway Emergency, focusing on accident discovery, UAV reconnaissance, ground resource deployment, response time and traffic recovery.

#### v2 Timeline
```
2026/06–07  整理 7600 万次探索日志，冻结场景空间和 coverage metric
2026/08–10  实现 accelerated testing 与强 baseline
2026/11     cross-planner evaluation、failure taxonomy、统计检验
2026/12–2027/01  写 F-J1 初稿
2027/02     ◉ 投稿 IEEE T-ITS
2027/03–06  扩展山东高速应急资源调配 F-J2
```

---

## 4. Overall 30-month submission roadmap for top issues

```
─────────────────────────────────────────────────────────────────────────────────────────
时间        A (T-ITS)    B (TR-C)     C (T-ITS)    D (TR-C)     E (AAAI)     F (T-ITS/TR-C)
─────────────────────────────────────────────────────────────────────────────────────────
2026/06    ▶ 环境搭建                  ▶ 算法实现                              ▶ 日志清洗
2026/07    实验训练                    AirSim搭建    ▶ GIS采集
2026/08    实验                        案例1巡检    实验          
2026/09                  ▶ 框架实现    案例2配送                                加速测试
2026/10                  规模实验      案例3应急    多城市实验    ▶ 数据集     baseline
2026/11                  实验          多机系统级   案例研究
2026/12                  实验          初稿         案例研究      数据集完成    
2027/01                  理论分析      润色         写稿          实验          F-J1 写稿
2027/02                  写稿          润色         润色          实验          ◉ 投 T-ITS
2027/03    ◉ 投 T-ITS               ◉ 投 T-ITS                              F-J2 启动
2027/04                  ◉ 投 TR-C                ◉ 投 TR-C
2027/05                                                          实验
2027/06                                                          多UAV案例
2027/07                                                          写稿
2027/08                                                          写稿
2027/09    审稿意见                  审稿意见                   ◉ 投 T-ITS    审稿意见
2027/10                  接受目标                   接受目标                    接受目标
2027/11
2027/12    接受目标                  接受/转TR-C
2028/03                                                          接受目标
2028/06                              最终发表
─────────────────────────────────────────────────────────────────────────────────────────
◉ = 投稿节点   ▶ = 工作启动
```

**Core Rhythm:**
- **Second half of 2026:** G1 / E / F-J1 form the first batch of runnable experiments to avoid all work being pushed to the spring of 2027 at the same time.
- **Spring 2027:** A/B/C/D will continue to be promoted as the main line of systematic papers in the top issue.
- **2027 first half:** F-J2 differentiates from the F-J1 platform into a high-speed emergency resource deployment TR-C version.
- **H1 2028:** Main intake period.

---

## 5. Detailed explanation of top journal matrix| Journal | Field | IF | Acceptance rate | Review cycle | v2 adaptation Paper |
|------|------|-----|--------|---------|--------------|
| **IEEE T-ITS** | ITS General | 8.5 | ~20% | 4–6 months | A, C, F-J1, G/G1 Journal Extensions |
| **TR Part C** | New Transportation Technologies | 8.5 | ~18% | 4–6 months | B, D, F-J2 |
| **IEEE T-RO** | Robotics | 7.4 | ~25% | 6–10 months | C Prep |
| **TR Part B** | Transportation Methodology | 6.0 | ~15% | 6–8 Months | B Prep |
| **Transportation Science** | Transportation Science | 5.4 | ~12% | 6–10 months | B investment |

**v2 Submission Matrix Principles:**
- **Q1 with IF ≥ 8 preferred** (T-ITS, TR-C)
- **Prepare to invest in Q1** (T-RO) of the same fund with IF ≥ 7
- **Journals with IF < 7 will no longer be considered**

---

## 6. Risk Assessment and Alternatives

### 6.1 Key risks of top publication strategy

**Risk 1: The review period exceeds the PhD graduation window**
- The first round of review takes place from April to June for the top issue, and revisions may be delayed until 12+ months
- **Response:** Centralized submission in the spring of 2027, with 12 months reserved for revisions
- **Bottom line:** At least 2 articles accepted, the rest can be graduated with "submitted/in review" status

**Risk 2: Excessive experimental workload**
- The total workload of v2 is about 50–60 months (if serial), requiring team/cooperation division of labor
- **Response:** Prioritize G1/B/F-J1/E in the near future, and only retain concept and data entrances in other directions to avoid resource dilution

**Risk 3: Time lost in switching after rejection**
- One round of rejection + transfer = about 6 months of loss
- **Response:** Prepare TR-C / T-ITS dual framing in cover letter in advance

### 6.2 Priority of alternative submissions| Paper | First choice | Alternative 1 | Alternative 2 |
|-------|------|------|------|
| A | T-ITS | TR Part C | IEEE T-Cyber |
| B | TR Part C | T-ITS | TR Part B |
| C | T-ITS | TR Part C | IEEE T-RO |
| D | TR Part C | T-ITS | TR Part D (Environment) |
| E | AAAI/IJCAI | T-ITS | IEEE T-SMC |
| F | T-ITS | TR Part C | T-ASE / T-RO |

---

## 7. One-sentence summary of the report to the teacher

> "The current paper group has been reorganized into the main line of low-altitude UAV/low-altitude transportation cloud brain: G1 is the first to invest in AAAI/IJCAI, B is the main investment in TR-C, F-J1 is the main investment in T-ITS, E maintains AAAI method papers and reserves T-ITS expansion. Transportation journal papers must be supported by system issues, mathematical models, strong baselines, traffic indicators and failure analysis, and no longer rely solely on directional narratives."

---

## 8. v1 document processing instructions

- **v1 (`research-roadmap_v1_20260515.md`):** Reserved as a historical archive to record the design of the "Rapid Publishing Hybrid Strategy"
- **v2 (this document):** The currently effective planning document
- **Trigger conditions for the next update:** ① Complete the experimental data of Paper A ② Receive the first review comments ③ The teacher adjusts the direction

---

**Appendix: Correspondence between blog posts and Paper (consistent with v1)**| Blog Post | Corresponding Paper |
|---------|----------|
| marl-kat-uav-conflict | A (main) |
| uav-conflict-resolution | A (reference) |
| uav-conflict-env-construction | A (experimental environment) |
| large-scale-uav-scheduling | B (main) |
| uav-urban-route-planning | B (reference) |
| next-best-view-nerf-3dgs-exploration | C (main) |
| information-theory-active-perception | C (theoretical basis) |
| uav-nerf-gs-planning | C (reference) |
| **paper-c-fim-3dgs-uav-active-perception_v2_20260515** | **C Special Planning (v2)** |
| uav-semantic-mapping-functional-zoning | D (main) |
| uav-digital-twin-semantic-mapping | D (reference) |
| llm-uav-semantic-planning | E (main) |
| llm-guided-uav-planning-frontiers | E (reference) |
| paper-b-hierarchical-uav-scheduling-trc-plan-v1-20260519 | B special planning |
| paper-e-vera-uav-experiment-taskbook-v1-20260517 | E special task book |
| paper-f-uav-scenario-coverage-journal-roadmap-v2-20260520 | F Special Planning |
| paper-g-low-altitude-cloud-brain-llm-roadmap-v1-20260520 | G total route |
| paper-g1-cloudbrain-agent-full-paper-plan-v1-20260520 |G1 first complete thesis proposal |
| carla-sumo-rl-lane-change | Old F line, currently not included in the low-altitude UAV paper group |