---
title: "Paper F Paper Group Planning v1: UAV safety critical scenario generation, coverage and emergency application"
description: "Multiple paper routes are planned for UAV safety-critical scene generation, scene coverage, city-local scene correlation, and high-speed emergency rescue resource allocation directions."
pubDate: 2026-05-19
tags: ["Paper F", "UAV", "scene generation", "scene coverage", "Safety-Critical", "accelerated testing", "emergency rescue", "TR-C", "T-ITS"]
category: Tech
---

# Paper F Paper group planning v1: UAV safety critical scenario generation, coverage and emergency application

> Overall judgment: In addition to Paper B’s hundred-shelf-level system scheduling, Paper C’s 3DGS/FIM active sensing, and Paper E’s LLM/LTL language planning, you can also open a separate **UAV safety-critical scenario engineering** paper line.  
> The core of this line is not to create another obstacle avoidance algorithm, but to answer: **How ​​to systematically generate, cover, filter and reuse key UAV safety scenarios so that subsequent training, testing, emergency dispatch and paper experiments have a credible scenario base. **

---

## 1. Overall judgment: What other directions can be written?

There are currently several paper lines focusing on different issues:

| Thesis line | Core objects | Already covered | Should not be repeated |
|--------|----------|----------|----------|
| Paper B | Hundreds of UAV fleet | Three-layer hierarchical scheduling, queuing theory, Lyapunov, multi-modal transportation | No longer writing large-scale fleet scheduling separately |
| Paper C | UAV active sensing | 3DGS, Fisher information, next-best-view | No longer focused on mapping/perspective selection |
| Paper E | Language to planning | LLM, TaskIR, LTL/STL, formal verification | No longer focused on natural language task planning |
| Paper F | Scene Engineering | Scene generation, coverage, dangerous scenes, emergency applications | New directions |

The value of Paper F is that it can become the **experimental infrastructure** for the previous papers:

- Paper B requires urban needs and emergency scenarios.
- Paper C requires controllable local 3D scenes and occlusion/perspective coverage.
- Paper E requires task semantics, map entities, and safety constraints.
- Paper F can provide unified scenario grammar, coverage metric, criticality score and benchmark.For the "FENG SHOU" direction you mentioned, it is recommended that the standard correspond to **Shuo Feng**'s automated driving accelerated testing / testing scenario library generation work. The core idea is: safety-critical events are extremely rare in natural data, so we cannot rely solely on ordinary random testing, but must use data methods to construct more dangerous but still reasonable scenarios, thereby accelerating testing and safety verification [1] [2] [3]. This idea is very suitable for migration to UAV obstacle avoidance, low-altitude channel flight and high-speed emergency inspection.

---

## 2. Paper group hierarchical design

Paper F is recommended to be planned as 4 progressive papers:

| Level | Paper | One sentence positioning | Priority |
|------|------|------------|--------|
| F1 | CovUAV-Bench | UAV safety critical scene coverage benchmark | Highest |
| F2 | Coverage-Guided Accelerated Testing | Coverage-guided dangerous scene acceleration generation algorithm | Highest |
| F3 | City2Local-UAV | Create hierarchical scene generation from overall city ODD to local obstacle combination | Medium to High |
| F4 | Scenario-Aware Emergency Response | Do Shandong high-speed UAV-ground resource collaborative emergency deployment | Medium to high |

It is recommended to do **F1 + F2** first. F1 provides the dataset, metrics, and problem definition, and F2 provides the algorithmic contributions. F3 and F4 can be used as extensions: F3 turns the benchmark into a city-level system, and F4 turns the scene engineering into real traffic emergency business.

---

## 3. Common Background: Why scene coverage is the foundation for UAV safety research

A common weakness in UAV security research is: the algorithm is beautifully done, but the experimental scenario is too arbitrary. Just because an obstacle avoidance algorithm is successful in 20 manual scenarios does not mean that it covers the long-tail risks in urban low-altitude operations.A clear consensus has been reached in the field of autonomous driving: crashes/near-crash on real roads are rare events, and relying directly on natural testing would be extremely inefficient. Therefore, Shuo Feng et al. proposed a naturalistic and adversarial environment, using natural distribution to maintain authenticity, and using adversarial distribution to increase the probability of dangerous events, thereby accelerating intelligent driving testing [1]. They further proposed testing scenario library generation, defining the test scenario library under ODD as a set of representative and critical scenarios, and using criticality to take into account exposure frequency and maneuver challenge [2] [3]. Ding et al.'s review of safety-critical scenario generation also classified the field into three types of methods: data-driven, adversarial and knowledge-based, and pointed out that fidelity, efficiency, diversity, transferability and controllability are core challenges [13].

The UAV scenario requires this set of ideas even more for four reasons:

1. **Three-dimensional space is a higher dimension. **
   UAVs are not just about flat lanes, but also about altitude, obstacle volume, wind field, electrical charge, sensor field of view and flight dynamics.

2. **Hazardous events are more difficult to collect. **
   There are very few samples of real collisions with buildings, line collisions, entering no-fly zones, crossing bridges or high-speed accident scenes, and training cannot rely on real accident data.

3. **Ordinary random generation wastes computing power. **
   A large number of random scenarios are either too simple, physically infeasible, or dangerous but inevitable, making them inefficient for training and evaluation.

4. **There is no unified measurement of scene coverage. **
   Existing UAV papers often report success rate/collision rate, but rarely indicate which obstacle combinations, local geometries, task difficulties, and ODD boundaries are covered by the test set.

Therefore, the common scientific questions for Paper F are:

> How to build a UAV scenario generation and evaluation system that is real, controllable, and reproducible, and can effectively cover key safety long-tail risks?

---

## 4. Paper F1: UAV safety critical scene coverage Benchmark

### 4.1 Thesis title**CovUAV-Bench: A Coverage-Oriented Benchmark for Safety-Critical UAV Navigation Scenarios**

### 4.2 Background

SafeBench already provides a unified safety-critical benchmark in autonomous driving, integrating multiple types of scene templates, scene generation algorithms and evaluation indicators [5]. Scenic also proved that using probabilistic programs to express scene distribution, hard constraints and soft constraints is a feasible route [4]. There has been preliminary work on UAV simulation environment generation. For example, Nakama et al. proposed an automated UAV simulation environment generator [10]. FADS also showed that temporal-logic safety specification can enter the autonomous drone safety pipeline [11]. However, in the UAV field, there is still a lack of a coverage-oriented benchmark for 3D obstacle avoidance, low-altitude corridors, urban local spaces, and emergency tasks.

The goal of F1 is not to propose the strongest planner, but to define how the UAV scene space is covered by the system.

### 4.3 Method

Construct a basic test space of 50m x 50m x 50m, starting with local scenes and then expanding to urban blocks:- **Scene Objects**: Building blocks, towers, wires, trees, bridges, temporary obstacles, dynamic UAVs, ground vehicles, personnel areas.
- **Spatial structure**: open space, narrow passage, urban canyon, under-bridge, landing zone, highway shoulder, accident zone.
- **Environmental Disturbance**: Wind, visibility, sensor noise, GPS offset, communication delays.
- **Task type**: point-to-point navigation, inspection pass, emergency hover, landing, return-to-home.
- **Executable format**: Save it as `scenario.json` and add simulator adapter. It can be converted to AirSim, Flightmare, PyBullet or self-built lightweight simulation later.

Scene coverage is defined as:

$$
Coverage(S)=
\sum_{k=1}^{K} w_k \cdot
\frac{|B_k(S)|}{|B_k(\Omega)|},
$$

Where $\Omega$ is the discretized scene space of the target ODD, $B_k(S)$ is the bin covered by the sample set $S$ on the $k$th class attribute dimension, and $w_k$ is the dimension weight.

The existing **76 million explorations** can be written as "Existing exploration log assets" for statistics:

- Which scene combinations are frequently explored.
- Which combinations are still coverage holes.
- Which combinations trigger collision / near-miss / timeout.
- Which combinations are invalid training samples.

Note: 76 million explorations are only written as "available experimental basis" and cannot be written as verified conclusions.

### 4.4 Baselines| Baseline | Purpose |
|----------|------|
| Random scenario sampling | The most basic coverage baseline |
| Grid sampling | Uniform discretization of parameter space |
| Latin hypercube sampling | More efficient parameter coverage |
| Scenic-style constrained sampling | Constrained scene generation baseline [4] |
| SafeBench-style template suite | Templated security scenario baseline [5] |

### 4.5 Innovation points

1. Propose UAV scene coverage taxonomy: ODD, obstacle combination, dynamic disturbance, task type, risk level.
2. Give a coverage-oriented benchmark instead of just a few manual maps.
3. Convert exploration log into coverage holes and critical scenario seeds.
4. Provide a unified scene interface for subsequent Paper B/C/E.

### 4.6 How to evaluate

| Indicator | Meaning |
|------|------|
| Parameter coverage | Parameter bin coverage ratio |
| Pairwise / t-wise coverage | Multi-attribute combination coverage |
| Critical scenario density | Number of near-miss / collisions discovered per unit test budget |
| Invalid scenario rate | The proportion of physically infeasible or mission meaningless scenarios |
| Planner ranking stability | Is the algorithm ranking stable under different random seeds |
| Replay reproducibility | Whether the same result can be reproduced with the same seed |

### 4.7 Recommended contributions

- Main line: T-ITS / IEEE ITSC / IROS benchmark-oriented paper.
- Alternative: RA-L + ICRA, if the benchmark has both high-quality open source tools and small-scale verification of real UAVs.

---

## 5. Paper F2: Accelerate the generation of dangerous scenes guided by coverage### 5.1 Thesis title

**Coverage-Guided Accelerated Testing for Safety-Critical UAV Obstacle Avoidance**

### 5.2 Background

The core of accelerated testing for autonomous driving is not to "create inevitable crash scenarios", but to improve the sampling efficiency of safety-critical events while maintaining scene authenticity and actionability [1] [2] [3]. If the generated scenario is not feasible for any planner, then it cannot help differentiate the algorithm's capabilities; if the generated scenario is too safe, it cannot expose system weaknesses.

UAV obstacle avoidance training also has the same problem:

- A large number of randomly generated scenes without security pressure.
- Confrontation generation tends to produce obstacle layouts that cannot be reasonably avoided.
- Manual curriculum has limited coverage and cannot explain whether long-tail risks are covered.
- RL training wastes budget on a lot of invalid scenarios.

### 5.3 Method

Proposed **CGAT-UAV: Coverage-Guided Accelerated Testing for UAVs**.

The algorithm consists of four modules:

1. **Scenario encoder**
   Encode the scene into structured vectors: number of obstacles, minimum channel width, target direction, dynamic obstacle speed, wind intensity, sensor noise, battery margin, etc.

2. **Coverage memory**
   Maintain coverage bins, failure types, and planner performance for explored scenes.

3. **Criticality score**
   Refer to Feng’s criticality idea and combine the degree of risk with the frequency of exposure [2]:

   $$
   Crit(s)=P_{\text{exposure}}(s)\cdot R_{\text{challenge}}(s)\cdot F_{\text{feasible}}(s).
   $$

   Among them, $F_{\text{feasible}}(s)$ is used to punish inevitable collisions and physically unreasonable scenarios.4. **Adaptive generator**
   Generate new scenes in coverage holes and high-criticality regions using Bayesian optimization, CMA-ES, RL editing, or cross-entropy methods.

### 5.4 Baselines

| Baseline | Comparison purpose |
|----------|----------|
| Random generation | Test acceleration rate |
| Grid / Latin hypercube sampling | Coverage efficiency |
| Bayesian optimization | Black box dangerous search |
| CMA-ES | Continuous Parametric Hazard Search |
| RL adversarial scenario generation | Learning hazard generation |
| Scenic constrained generation | Rules and constraints generation [4] |
| FREA-style feasibility-guided generation | Compare the idea of "reasonable antagonism" [12] |

### 5.5 Innovation points

1. Migrate accelerated testing from autonomous driving to UAV 3D obstacle avoidance.
2. Optimize **coverage, criticality, and feasibility** at the same time to avoid pursuing only the collision rate.
3. Propose a coverage-guided curriculum to train planners with dangerous but solvable scenarios.
4. The test acceleration rate is given: the number of simulations required to reach the same confidence interval is significantly reduced.

### 5.6 How to evaluate| Indicator | Meaning |
|------|------|
| Acceleration factor | The multiple reduction in the number of tests required to achieve the same failure discovery rate compared to random testing |
| Failure discovery rate | The ratio of collision / near-miss / timeout discovered per unit budget |
| Feasible criticality | Proportion of danger and feasible obstacle avoidance strategies |
| Naturalness score | Whether the scene conforms to ODD prior |
| Coverage gain per 1k tests | New coverage every 1000 tests |
| Training efficiency | After training with generated scenarios, planner’s improvement in held-out test |

### 5.7 Recommended contributions

- Mainline: AAAI/ICRA/IROS.
- Alternative: T-ITS, if more emphasis is placed on traffic safety testing and benchmarking.

---

## 6. Paper F3: Hierarchical generation of urban overall scenes to local obstacle combinations

### 6.1 Thesis title

**City2Local-UAV: Hierarchical Scenario Generation from Urban ODDs to Local Obstacle Compositions**

### 6.2 Background

F1 and F2 address a local 3D test space, but real urban low-altitude flights are not isolated boxes. Why a local scene appears depends on the overall structure of the city: road grades, building density, functional areas, bridges, service areas, interchanges, hospitals, schools, no-fly zones and emergency points.

ASAM OpenODD/OpenSCENARIO provides a standardized idea from ODD, current operating domain to executable scenario description [6] [7]. The UAV field can learn from this level of abstraction, but will need to incorporate three-dimensional obstacles, airspace constraints, and low-altitude mission semantics.

### 6.3 Method

Propose a three-layer generation pipeline from city to local:

```text
City-level ODD
  -> district / road / highway segment selection
  -> local 50m x 50m x 50m UAV test cell
  -> concrete obstacle composition
  -> simulator executable scenario
```

Specific modules:- **City ODD parser**: Extract city/highway semantics from OSM, road grades, building outlines, POIs, service areas, bridges and highway entrances.
- **Local cell sampler**: Select typical local cells, such as high-rise canyons, service areas, overpasses, toll stations, highway shoulders, and accident bottleneck areas.
- **Obstacle grammar**: Use rules to generate local obstacle combinations, such as buildings + wires + trees + parked vehicles + personnel restricted areas.
- **Coverage controller**: Monitors the coverage of different urban functional areas and local combinations.

### 6.4 Baselines

| Baseline | Comparison purpose |
|----------|----------|
| Pure random local generation | Does not consider urban context |
| OSM-to-map direct conversion | Only converts the map, does not control scene coverage |
| CARLA / OSM digital twin generation | Ground autonomous driving digital twin baseline [14] |
| Manual scenario templates | Manual rule templates |
| CityEngine / procedural city generation | Procedural city generation baseline |

### 6.5 Innovation points

1. Associate the urban ODD with the UAV local safety test cell.
2. Propose hierarchical scene generation of “global city semantics -> local obstacle composition”.
3. Extend scene coverage from local parameters to urban functional area coverage.
4. Support real city case studies, such as Jinan, Qingdao, and Shandong key highway hubs.

### 6.6 How to evaluate| Indicator | Meaning |
|------|------|
| ODD coverage | Urban functional areas, road grades, building density coverage |
| Local composition diversity | Local obstacle combination diversity |
| Realism score | Consistency with OSM/POI/Building Statistics |
| Transferability | Is the policy generated from one city still valid when moved to another city |
| Criticality preservation | Whether urban context generation preserves high-risk local scenes |

### 6.7 Recommended contributions

- Mainline: TR-C, if urban transportation systems, ODD, low-altitude infrastructure and scene datasets are emphasized.
- Alternative: T-ITS, if OpenSCENARIO-like scenario interface and intelligent system evaluation are emphasized.

---

## 7. Paper F4: UAV-ground resource collaborative deployment for Shandong highway emergency rescue

### 7.1 Thesis title

**Scenario-Aware UAV-Ground Resource Allocation for Highway Emergency Response**

### 7.2 Background

Shandong Expressway already has a business foundation for low-altitude inspection and emergency response. Public information from Shandong Hi-Speed ​​Group shows that its comprehensive inspection flight service system has deployed unattended platforms and industrial drones in key areas for road condition inspections, road inspections, emergency response and data analysis [15]. This shows that high-speed scenarios are not pure imagination, but have application entrances.

Research on highway emergency resource allocation pointed out that there are still several problems in the existing work: insufficient site selection of roadside small/micro emergency facilities during the operation phase, complete information is often assumed in the early stage of the accident but is not actually true, traffic status after the accident is uncertain and time-varying, and the integrated optimization of facility site selection, resource allocation and dispatch is still insufficient [16]. There have been studies on space-time network UAV routing in traffic incident monitoring [17], and there have been studies on UAV real-time deployment and resource allocation in disaster emergency communications [18], but they have not yet formed a unified closed loop with high-speed emergency accident scene coverage, on-site reconnaissance information value, and ground rescue resource allocation.

This is suitable for the introduction of UAV: ​​the drone first arrives at the accident scene to obtain the situation, and then the ground clearance, firefighting, rescue and control resources are dynamically dispatched.

### 7.3 MethodProposed **Scenario-Aware UAV-Ground Emergency Dispatch**:

- **Accident scene generation**: Based on the F1/F3 high-speed scene library, the accident type, traffic flow status, weather, road section geometry, obstacles and secondary risks are generated.
- **UAV reconnaissance layer**: UAVs take off from service areas, toll stations or unmanned platforms to quickly confirm accident locations, congestion lengths, passable lanes and hazardous materials risks.
- **Ground resource allocation layer**: dispatches tow trucks, firefighting, ambulance, traffic police, maintenance vehicles and temporary control resources.
- **Information Value Modeling**: Write the uncertainty reduction of UAV reconnaissance into the dispatch goal, that is, the UAV does not just take pictures, but reduces false dispatch and response delays.
- **Rolling Optimization**: Accident information is updated over time, and scheduling strategies are recalculated on a rolling basis.

### 7.4 Problem formulation

Let the highway section set be $\mathcal{L}$, the accident set be $\mathcal{I}(t)$, the UAV set be $\mathcal{U}$, the ground rescue resource set be $\mathcal{G}$, and the service station/unattended platform set be $\mathcal{B}$.

Decision variables include:

- UAV dispatch $x_{u,i}(t)$: Whether UAV $u$ scouts incident $i$.
- Ground resource dispatch $y_{g,i}(t)$: whether resource $g$ is heading to incident $i$.
- Take-off/departure time $s_u(t), s_g(t)$.
- Information update action $a_i(t)$: whether to wait for further confirmation from the UAV or dispatch directly.

Objective function:

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

Among them, $U_{\text{uncertainty}}$ represents the uncertainty of accident information, which can be reduced by UAV reconnaissance.

### 7.5 Baselines| Baseline | Comparison purpose |
|----------|----------|
| Ground-only dispatch | No drone reconnaissance |
| Nearest-resource dispatch | Nearest resources first |
| Static facility allocation | Fixed facility allocation |
| Two-stage stochastic optimization | Estimate the accident before dispatching |
| UAV-first heuristic | UAV reconnaissance first, then ground dispatch |
| Scenario-aware rolling optimization | Main method |

### 7.6 Innovation points

1. Connect scene coverage and high-speed emergency dispatch, instead of just resource allocation.
2. Model UAV reconnaissance as a decision-making action that reduces uncertainty in incident information.
3. Support the real business context of Shandong Expressway: unattended platform, road condition inspection, emergency response and work order circulation.
4. Unified optimization of response time, clearance time, secondary accident risk and dispatch cost.

### 7.7 How to evaluate

| Indicator | Meaning |
|------|------|
| First-view time | The time when the UAV first acquired the accident footage |
| Response time | Arrival time of the first batch of rescue resources |
| Clearance time | Accident clearance completion time |
| Wrong dispatch rate | The proportion of wrong dispatch, missed dispatch or insufficient resources |
| Secondary accident risk | Secondary accident risk proxy |
| Congestion delay | Total delay caused by accident |
| UAV information value | Reconnaissance with UAV reduces uncertainty compared to without UAV |

### 7.8 Recommended contributions

- Main line: TR-C first, as the focus is on high-speed emergency transportation system operations, resource allocation and transportation network resilience.
- Alternative: T-ITS, if further emphasis is placed on drone platforms, communications, video recognition, work order systems and online intelligent dispatch.

---

## 8. Unify experimental platform, data sources and evaluation indicators

### 8.1 Experimental platform| Hierarchy | Recommended implementation | Purpose |
|------|----------|------|
| Lightweight simulation | Python / PyBullet / custom 3D grid | 76 million levels of rapid exploration |
| UAV simulation | AirSim, Flightmare | Vision, dynamics, sensor verification [8] [9] |
| Scenario language | Scenic-like DSL, JSON schema | Reproducible scene generation [4] |
| City data | OpenStreetMap, POI, road grades, building outlines | City to local scene generation |
| High-speed emergency | Shandong expressway open cases, accident statistics, synthetic accident flow | Emergency resource allocation experiment |

The main experiment of F1/F2 should prioritize lightweight simulation to ensure large-scale exploration. AirSim/Flightmare is used for small-scale high-fidelity verification and is not relied upon for all experiments.

### 8.2 Data source

- **Synthetic UAV scenario benchmark**: Procedurally generated 50m x 50m x 50m local space.
- **Exploration logs**: 76 million exploration logs for coverage holes and failure taxonomy.
- **OSM/POI/building data**: for urban functional areas and local barrier combinations.
- **Shandong Expressway Public Business Information**: used for application background and deployment assumptions [15].
- **High-speed accident and emergency resource disclosure research**: used for accident types, resource allocation stages and evaluation indicators [16].

### 8.3 Unified indicators| Indicator Group | Indicator |
|--------|------|
| Coverage | parameter coverage, t-wise coverage, ODD coverage, coverage gain |
| Safety | collision rate, near-miss rate, minimum distance, constraint violation |
| Danger generation | criticality, failure discovery rate, acceleration factor, feasible criticality |
| Training value | sample efficiency, held-out success rate, robustness under ODD shift |
| Emergency value | first-view time, response time, clearance time, wrong dispatch rate |

---

## 9. Recommended submission path and priority

### 9.1 The first stage: do F1 + F2 first

In the first phase, it is recommended to write two articles directly around "UAV safety critical scene coverage + accelerated testing":

1. **F1 benchmark paper**
   More stable, suitable as an experimental base for all subsequent UAV papers. Even if the algorithm is not particularly strong, it can still be established based on taxonomy, coverage metric, data set and reproducible experiments.

2. **F2 method paper**
   Methodological contributions to AAAI/ICRA/IROS. The highlight is the migration from Shuo Feng's accelerated testing of autonomous driving to UAV 3D scenes and the addition of coverage-guided feasible criticality.

### 9.2 Phase 2: Do F3 + F4 again

F3 and F4 are more suitable for advancement after F1/F2 has a tool foundation:

- **F3** To solve the relationship between the overall city and local scenes, you can vote for TR-C / T-ITS.
- **F4** For Shandong highway emergency rescue applications, TR-C can be selected, emphasizing transportation operation and emergency response.### 9.3 Relationship with existing paper lines

| Paper | How to support Paper F |
|------|--------------------|
| Paper B | Provides peak / shock / highway emergency demand scenarios |
| Paper C | Provides local 3D occlusion, perspective coverage and reconstruction difficult scenes |
| Paper E | Provides natural language tasks, map entities and safety constraint scenarios |

Paper F is best suited to be the “scenario infrastructure paper” for the entire UAV research line.

---

## 10. References

[1] Shuo Feng, Xintao Yan, Haowei Sun, Yiheng Feng, and Henry X. Liu. “Intelligent Driving Intelligence Test for Autonomous Vehicles with Naturalistic and Adversarial Environment.” *Nature Communications*, 12:748, 2021. DOI: 10.1038/s41467-021-21007-8. URL: <https://doi.org/10.1038/s41467-021-21007-8>

[2] Shuo Feng, Yiheng Feng, Chunhui Yu, Yi Zhang, and Henry X. Liu. "Testing Scenario Library Generation for Connected and Automated Vehicles, Part I: Methodology." *IEEE Transactions on Intelligent Transportation Systems*, 22(3):1573-1582, 2021. DOI: 10.1109/TITS.2020.2972211. URL: <https://doi.org/10.1109/TITS.2020.2972211>[3] Shuo Feng, Yiheng Feng, Haowei Sun, Shan Bao, Yi Zhang, and Henry X. Liu. "Testing Scenario Library Generation for Connected and Automated Vehicles, Part II: Case Studies." *IEEE Transactions on Intelligent Transportation Systems*, 22(9):5635-5647, 2021. DOI: 10.1109/TITS.2020.2988309. URL: <https://doi.org/10.1109/TITS.2020.2988309>

[4] Daniel J. Fremont, Tommaso Dreossi, Shromona Ghosh, Xiangyu Yue, Alberto L. Sangiovanni-Vincentelli, and Sanjit A. Seshia. “Scenic: A Language for Scenario Specification and Scene Generation.” *Proceedings of the 40th ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI)*, 2019. DOI: 10.1145/3314221.3314633. URL: <https://people.eecs.berkeley.edu/~sseshia/pubs/b2hd-fremont-pldi19.html>[5] Chejian Xu, Wenhao Ding, Weijie Lyu, Zuxin Liu, Shuai Wang, Yihan He, Hanjiang Hu, Ding Zhao, and Bo Li. “SafeBench: A Benchmarking Platform for Safety Evaluation of Autonomous Vehicles.” *Advances in Neural Information Processing Systems 35 (NeurIPS 2022) Datasets and Benchmarks Track*, 2022. URL: <https://proceedings.neurips.cc/paper_files/paper/2022/hash/a48ad12d588c597f4725a8b84af647b5-Abstract-Datasets_and_Benchmarks.html>

[6] ASAM. “ASAM OpenSCENARIO DSL: Key Terminology and Conceptual Overview.” URL: <https://publications.pages.asam.net/standards/ASAM_OpenSCENARIO/ASAM_OpenSCENARIO_DSL/latest/conceptual-overview/key_terms.html>

[7] ASAM. “ASAM OpenODD: Model to ASAM OpenSCENARIO DSL Mapping Reference.” URL: <https://publications.pages.asam.net/standards/ASAM_OpenODD/ASAM_OpenODD/latest/specification/09_openscenario_dsl/09_01_overview.html>[8] Shital Shah, Debadeepta Dey, Chris Lovett, and Ashish Kapoor. “AirSim: High-Fidelity Visual and Physical Simulation for Autonomous Vehicles.” *Field and Service Robotics*, Springer Proceedings in Advanced Robotics, 2017; arXiv:1705.05065. URL: <https://arxiv.org/abs/1705.05065>

[9] Yunlong Song, Selim Naji, Elia Kaufmann, Antonio Loquercio, and Davide Scaramuzza. “Flightmare: A Flexible Quadrotor Simulator.” *Proceedings of the 4th Conference on Robot Learning (CoRL)*, PMLR 155, 2021. URL: <https://proceedings.mlr.press/v155/song21a.html>

[10] Justin Nakama, Ricky Parada, Joao P. Matos-Carvalho, Fabio Azevedo, Dario Pedro, and Luis Campos. “Autonomous Environment Generator for UAV-Based Simulation.” *Applied Sciences*, 11(5):2185, 2021. DOI: 10.3390/app11052185. URL: <https://doi.org/10.3390/app11052185>[11] Yash Vardhan Pant, Max Z. Li, Alena Rodionova, Rhudii A. Quaye, Houssam Abbas, Megan S. Ryerson, and Rahul Mangharam. “FADS: A Framework for Autonomous Drone Safety Using Temporal Logic-Based Trajectory Planning.” *Transportation Research Part C: Emerging Technologies*, 130:103275, 2021. DOI: 10.1016/j.trc.2021.103275. URL: <https://doi.org/10.1016/j.trc.2021.103275>

[12] Keyu Chen, Yuheng Lei, Hao Cheng, Haoran Wu, Wenchao Sun, and Sifa Zheng. "FREA: Feasibility-Guided Generation of Safety-Critical Scenarios with Reasonable Adversariality." arXiv:2406.02983, 2024. URL: <https://arxiv.org/abs/2406.02983>

[13] Wenhao Ding, Chejian Xu, Mansur Arief, Haohong Lin, Bo Li, and Ding Zhao. “A Survey on Safety-Critical Driving Scenario Generation: A Methodological Perspective.” arXiv:2202.02215, 2022. URL: <https://arxiv.org/abs/2202.02215>[14] CARLA Team. “Digital Twin Tool: Procedural Generation from OpenStreetMap.” CARLA Simulator Documentation. URL: <https://carla.readthedocs.io/en/0.9.16/adv_digital_twin/>

[15] Shandong Expressway Group Co., Ltd. “‘Shandong Expressway Comprehensive Inspection Flight Service System’ goes online.” 2025. URL: <https://www.sdhsg.com/article/72553>

[16] Zhao Xiangmo, Zhao Yifei, Lu Nengchao, et al. "A review of research on key resource allocation for highway traffic accident emergency." *Transactions of Transportation Engineering*, 2024. DOI: 10.19818/j.cnki.1671-1637.2024.06.001. URL: <https://transport.chd.edu.cn/cn/article/doi/10.19818/j.cnki.1671-1637.2024.06.001>

[17] Jisheng Zhang, Limin Jia, Shuyun Niu, Fan Zhang, Lu Tong, and Xuesong Zhou. "A Space-Time Network-Based Modeling Framework for Dynamic Unmanned Aerial Vehicle Routing in Traffic Incident Monitoring Applications." *Sensors*, 15(6):13874-13898, 2015. DOI: 10.3390/s150613874. URL: <https://doi.org/10.3390/s150613874>[18] Tan Do-Duy, Long D. Nguyen, Trung Q. Duong, Saeed Khosravirad, and Holger Claussen. "Joint Optimization of Real-Time Deployment and Resource Allocation for UAV-Aided Disaster Emergency Communications." *IEEE Journal on Selected Areas in Communications*, 39(11):3411-3424, 2021. DOI: 10.1109/JSAC.2021.3088662. URL: <https://doi.org/10.1109/JSAC.2021.3088662>

---

## Appendix: This execution plan

### Step 1: Freeze Paper F Total Positioning

- Position Paper F as UAV safety-critical scenario engineering.
- Make it clear that it is not a duplicate of Paper B/C/E, but a group of Experimental Infrastructure and Scenario Methods papers.
- Adopt the structure of four progressive papers from F1 to F4.

### Step 2: Do F1 benchmark first

- Define UAV scenario taxonomy.
- Design the `scenario.json` schema.
- Organized 76 million exploration logs.
- Statistics coverage holes, failure modes and invalid scenario rate.
- Export CovUAV-Bench v0.1.

### Step 3: Advance the F2 Accelerated Testing Algorithm- Implement random/grid/LHS/BO/CMA-ES/RL adversarial baselines.
- Implement coverage memory, criticality score and feasible criticality filter.
- Compare failure discovery rate, coverage gain and acceleration factor.
- Use held-out test to verify the training value.

### Step 4: Extend F3 city to local scene

- Access OSM, road grades, building density and POIs.
- Select key sections of Jinan/Qingdao/Shandong Expressway as case study.
- Map city-level ODD to local 50m x 50m x 50m test cell.
- Establish urban functional area coverage indicators.

### Step 5: Expand F4 High-Speed Emergency Application

- Take Shandong expressway inspection/emergency response as the application background.
- Design accident scenarios, UAV reconnaissance, and ground rescue resource collaborative deployment processes.
- Compare ground-only, nearest-resource, UAV-first heuristic and scenario-aware rolling optimization.
- Focus on reporting first-view time, response time, clearance time and wrong dispatch rate.

### Step 6: Submission rhythm

- Invest in F1/F2 first to form a benchmark + method dual core.
- F1 If the tools and data are complete, priority will be given to T-ITS / ITSC / IROS benchmark.
- F2 If the algorithm result is strong, AAAI / ICRA / IROS will be given priority.
- F3/F4 Wait until the F1/F2 tool is stable before switching to TR-C / T-ITS.

### Step 7: Tasks for the Recent Week-Write a formal experimental assignment for F1.
- Freeze scene dimensions: obstacles, spatial structures, environmental disturbances, task types, risk labels.
- Sampling 10,000 to 50,000 exploration logs from 76 million exploration logs for preliminary coverage analysis.
- Draw the first version of the scene taxonomy diagram and coverage heatmap.