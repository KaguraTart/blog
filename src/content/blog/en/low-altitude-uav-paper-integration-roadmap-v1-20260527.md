---
title: "Low Altitude Planning Thesis Matrix v1: Integration of already written articles, subsequent topic selection and Zotero literature list"
description: "Integrate conflict-free path planning, three-layer scheduling of hundreds of UAVs, information theory-driven 3DGS active sensing planning and other written directions, plan subsequent low-altitude planning paper groups, and provide top journals and highly relevant arXiv references for 2021-2026."
pubDate: 2026-05-27
updatedDate: 2026-05-27
tags: ["low altitude planning", "UAV", "Thesis planning", "Zotero", "T-ITS", "TR-C", "T-RO", "AAAI", "ICRA", "3DGS", "MARL"]
category: Tech
sourceHash: "2c029666156e0305f30131a09abf73c03a2ccbd7"
---

# Low Altitude Planning Thesis Matrix v1: Integration of already written articles, subsequent topic selection and Zotero literature list

> This article reintegrates the low-altitude UAV papers that have been written so far into a **paper portfolio**.  
> The goal is not to spread out and write a lot of ideas, but to clarify: which articles have already taken shape, which ones can continue to be made into top journal/regular conference papers, and what literature support, experimental assets and submission positioning are needed for each paper.

---

## 1. There are currently articles and main line positioning

There are currently three core articles that have formed the basis of content:

| Number | Existing content | Current positioning | Recommended main investment | Core judgment |
|---|---|---|---|---|
| Paper A | Conflict-free path planning / PPO / MAPPO / Multi-UAV conflict resolution | Robust conflict resolution in low-altitude air route network | IEEE T-ITS / IEEE T-RO / ICRA-IROS | You cannot just write PPO, you must write "Safety-efficiency trade-off under non-cooperative UAVs, communication degradation, and high-density corridors" |
| Paper B | Hundreds of UAV three-layer dispatching | Urban low-altitude logistics/emergency system operation dispatching | TR-C first, T-ITS backup | This is a transportation system paper, focusing on capacity, delay, queue stability, vertiport/charging/corridor bottleneck |
| Paper C | UAV 3DGS active sensing planning driven by information theory | Active sensing + low-altitude digital twin + planning closed loop | T-RO / T-ITS / ICRA-IROS | If submitted to a transportation journal, it must be proven that active sensing improves inspection, emergency, obstacle avoidance or operational control indicators |

These three articles have been able to form a very stable low-altitude planning triangle:

```text
Paper A：战术安全
  多 UAV conflict resolution / no-conflict planning / PPO-MAPPO / CBF / RMADER

Paper B：系统运营
  hundred-UAV scheduling / queue stability / Lyapunov / multimodal logistics

Paper C：环境认知
  3DGS active perception / Fisher information / NBV / safe reconstruction
```

It is best for subsequent new papers to expand around this triangle and not to start in a completely unrelated direction.

---

## 2. Overall submission judgment

The low-altitude planning direction can be divided into three categories of papers, and different categories have different review standards:| Type | Representative papers | Review attention | Recommended venues |
|---|---|---|---|
| Transportation system papers | Paper B, emergency resource allocation, low-altitude road network planning | Real traffic problems, system indicators, data/simulation credibility, policy or operational implications | TR-C, T-ITS |
| Robot planning paper | Paper A, Paper C, digital twin planning | Algorithm novelty, real-time, security, hardware/simulation verification | T-RO, RA-L+ICRA/IROS, T-ITS |
| AI method papers | VERA-UAV, CloudBrain-Agent, scene acceleration generation | benchmark difficulty, theory/verification mechanism, model generalization, reproducibility | AAAI, IJCAI, NeurIPS/ICLR workshop, T-ITS extension |

The official positioning of TR-C emphasizes transportation systems and emerging technologies, and the intellectual core is on the transportation side [1]; T-ITS covers sensing, communications, controls, planning, design, implementation and other modern transportation system technologies [2]. Therefore:

- **Paper B/Emergency resource allocation/Low-altitude road network planning**: Prioritize writing according to the transportation system operation logic of TR-C.
- **Paper A / Paper C**: You can vote for T-RO or ICRA/IROS; if you switch to T-ITS, you need to supplement the transportation system indicators.
- **Paper E/G type LLM-Agent**: The first article is more suitable for AAAI/IJCAI, and the journal version is extended to T-ITS.

---

## 3. Paper matrix: It is recommended to form 7 articles that can be advanced

### 3.1 Paper A: Robust conflict-free planning of low-altitude air route network

**Suggested topic:** Robust Conflict-Free UAV Corridor Planning under Non-Cooperative Traffic and Communication Degradation**Corresponding to existing articles:** Conflict-free path planning, PPO/MAPPO, UAV conflict resolution, UAV conflict env construction.

**Core question:** In an urban low-altitude airway network, how can multiple UAVs maintain separation safety while controlling delays, extra distances, and throughput losses under the conditions of local observation, communication delays, positioning errors, and non-cooperative aircraft insertion.

**Method Route:**

- strategic layer: initial path and time slot allocation based on route network;
- tactical layer: MAPPO/PPO output speed, height or lateral offset action;
- safety shield: CBF-QP / ORCA / RMADER-style trajectory check;
- fallback layer: switch to conservative priority rule when communication degrades;
- Evaluation: 30/50 aircraft are trained and 100/200 aircraft are tested, covering four scenarios: cooperative, non-cooperative, communication-loss, and high-density corridor.

**Key References:**

The multi-agent stable training of MAPPO/PPO can be supported by Yu et al. [3]; MAT and FACMAC provide stronger MARL baseline [4,5]; HAPPO/HATRPO provides trust-region multi-agent policy optimization reference [6]. On the robot side, EGO-Swarm, MADER, RMADER, RACER, PANTHER and GCOPTER respectively support decentralized swarm planning, trajectory sharing under delay, collaborative exploration, perception-aware planning and multicopter trajectory optimization [7-12].

**Innovation suggestions:**1. Upgrade “PPO conflict-free path planning” from a simple RL task to low-altitude traffic corridor safety control.
2. Introduce communication degradation and non-cooperative UAV to form the actual operating boundary that T-ITS is more concerned about.
3. Use learning policy + formal/safety shield to avoid the lack of security of pure RL.
4. Trafficization of indicators: LoWC, NMAC, conflict count, average delay, extra distance, throughput, runtime.

### 3.2 Paper B: Three-layer hierarchical scheduling of hundreds of UAVs

**Suggested topic:** H-LyraUAV: Queue-Stable Hierarchical Scheduling for Hundred-Scale Low-Altitude UAV Logistics

**Corresponding to existing articles:** Paper B three-tier scheduling planning.

**Core question:** How can a hundred-level UAV fleet operate stably, efficiently, and safely under dynamic requirements, limited vertiport/charging/corridor capacity, and multi-modal transportation constraints.

**Method Route:**

- macro layer: demand queue, fleet repositioning, mode choice;
- meso layer: vertiport, charging pad, corridor slot scheduling;
- micro layer: energy/safety/conflict-aware trajectory feasibility;
- Theory: Lyapunov drift-plus-penalty guarantees queue stability and cost-backlog tradeoff;
- data: synthetic city grid + OSM/POI/NYC TLC/Chicago taxi/SUMO enhancement.

**Key References:**TR-C low-altitude UAV delivery traffic management has directly discussed resource allocation and conflict resolution in low-altitude urban space [13]; passenger-centric UAM, fairness and operational efficiency research support service quality framing [14]; charging-station delivery network, capacity-constrained UAM scheduling, safe learning scheduling support infrastructure capacity and safe online scheduling [15-17]; truck-drone / UAV-UGV multi-modal delivery support multimodal extension [18,19].

**Innovation suggestions:**

1. Hundred-shelf-level online three-layer scheduling closed loop instead of offline routing/network design.
2. Queue stability becomes the main line of theory, and the learning module only makes predictions or value estimates.
3. Evaluate delay, throughput, backlog, charging utilization, vertiport bottleneck, and corridor congestion at the same time.
4. The conclusion of the traffic system can answer: when does it need to limit traffic, where is the bottleneck, and when is UAV-only inferior to multimodal fallback.

### 3.3 Paper C: FIM-3DGS UAV active sensing planning

**Suggested topic:** FIM-3DGS: Fisher-Information-Driven Active Perception Planning for Safe UAV Reconstruction

**Corresponds to existing articles:** Paper C, Next-Best-View and NeRF/3DGS, Information Theory Active Sensing.

**Core question:** Under limited flight time, energy and safety constraints, how can UAV actively select viewpoints to make the 3DGS map converge faster and serve low-altitude planning tasks.

**Method Route:**- scene representation: incremental 3D Gaussian Splatting;
- information metric: build Fisher Information / expected information gain for Gaussian parameters or rendered Jacobian;
- planner: NBV candidate generation + safe corridor / CBF constraint;
- task coupling: The reconstruction quality is not only reported on PSNR/SSIM, but also on obstacle recall, planning collision rate, and inspection coverage;
- baselines: ActiveNeRF, FisherRF, GS-Planner, HGS-Planner, POp-GS, frontier exploration.

**Key References:**

The original 3DGS text provides real-time explicit radiance field representation [20]; ActiveNeRF is an early representative of neural rendering active perception [21]; FisherRF directly supports Fisher information active view selection, and has 3DGS backend 70 fps results [22]; GS-Planner, HGS-Planner, POp-GS and NVF support the 3DGS/NBV competition line of 2024-2025 [23-26].

**Innovation suggestions:**

1. Upgrade from “3DGS NBV” to “active perception serving UAV security planning”.
2. Use Fisher information to connect CRB / reconstruction uncertainty / planning safety.
3. Expand from visual indicators to traffic/robot task indicators: path feasibility rate, obstacle recall rate, emergency inspection coverage rate.
4. Do cross-scenario generalization on MatrixCity/AirSim/self-built urban low-altitude cells.

### 3.4 Paper D: Low-altitude safety critical scene coverage and accelerated testing**Suggested topic:** Coverage-Guided Accelerated Testing for Safety-Critical Low-Altitude UAV Navigation

**Corresponding to existing articles:** Paper F scene coverage, dangerous scene generation, 76 million exploration logs.

**Core question:** How to define the test scene space of low-altitude UAV obstacle avoidance/planning algorithm, how to measure coverage, and how to efficiently discover dangerous but effective failure scenarios.

**Method Route:**

- scenario grammar: local 50m x 50m x 50m cell, obstacle combination, dynamic obstacles, wind disturbance, target point, start and end points;
- coverage metric: geometry coverage, semantic coverage, dynamics coverage, risk coverage, failure-mode coverage;
- accelerated testing: actively sampling from coverage holes and failure likelihood;
- invalid filtering: filtering is unreal, unsafe, invalid, and unexecutable;
- cross-planner evaluation: A*/RRT*/MPC/ORCA/MAPPO/CBF-shielded planner.

**Key References:**

Shuo Feng's NADE and testing scenario library generation are core references for accelerated testing and security-critical scenario libraries [27-29]; SafeBench provides a benchmark platform and security assessment protocol reference [30].

**Innovation suggestions:**

1. Migrate from autonomous driving scenario engineering to low-altitude UAV 3D scene space.
2. Model the three objectives of coverage, criticality, and feasibility simultaneously.
3. Prove coverage space and failure taxonomy using 76 million exploration logs.
4. Let the results answer: Which combinations of obstacles are the most dangerous, which planners generalize the worst, and whether increased coverage really reduces unknown risks.### 3.5 Paper E: Verification of error-correcting UAV language planning

**Suggested topic:** VERA-UAV: Verification-and-Repair Language Planning for Low-Altitude UAV Tasks

**Corresponding to existing articles:** Paper E.

**Core problem:** LLM can convert natural language tasks into UAV executable task specifications, but it is prone to producing plans that are unexecutable, semantic mismatch, or violate safety constraints. Requires typed IR, LTL/STL, validators, and counterexample feedback loops.

**Method Route:**

- NL instruction -> typed TaskIR;
- TaskIR -> LTL/STL;
- Spot/RTAMT verification;
- counterexample/robustness feedback;
- local LLM iterative repair;
- final trajectory verification.

**Key References:**

Lang2LTL, NL2LTL, LTLCodeGen, and ConformalNL2LTL respectively support NL-to-LTL grounding, system demonstration, code-generation-style temporal logic generation and conformal correctness guarantee [31-34].

**Innovation suggestions:**

1. It is not just NL2LTL, but the UAV trajectory can perform closed loop.
2. Typed TaskIR reduces language ambiguity and improves interpretability.
3. Counterexample feedback and STL robustness feedback give repair a specific direction.
4. The AAAI/IJCAI version focuses on AI planning/verification; T-ITS is extended to connect to low-altitude traffic operation scenarios.

### 3.6 Paper G: Low-altitude traffic cloud brain LLM Agent

**Suggested topic:** CloudBrain-Agent: Tool-Augmented LLM Agents for Low-Altitude Traffic Operation

**Corresponding to existing articles:** Paper G/G1.**Core question:** The low-altitude traffic cloud brain cannot be just a chat model, but a verifiable agent that can call the scheduler, path planner, verifier, simulator and risk assessor.

**Method Route:**

- LLM is responsible for task understanding, tool selection, status summary and interpretation;
- Tools include Paper A conflict resolver, Paper B scheduler, Paper C active mapper, Paper D scenario tester, Paper E verifier;
- LowAltitudeIR as unified intermediate representation;
- The technical route gives priority to ordinary large models + agent + skills + MCP/tool-use, and will focus on the field of LoRA/SFT later;
- In the first stage of deployment, the API is called to form a benchmark, and in the second stage, the local Qwen/DeepSeek model is used for reproduction and cost control.

**Key References:**

UrbanGPT, UniST, and TrafficGPT show that transportation/urban spatiotemporal tasks have begun to move closer to foundation models and agent frameworks [35-37]; although DriveLM is autonomous driving, its Graph VQA task form can learn from the multi-step reasoning of low-altitude traffic cloud brain [38].

**Innovation suggestions:**

1. The low-altitude traffic cloud brain is not a "vertical chat model", but a tool-augmented verifiable agent.
2. Use unified IR to connect scheduling, planning, sensing, verification, and scenario testing.
3. Do the agent benchmark first, and then decide whether to fine-tune the vertical model to reduce the risk of the first article.
4. Evaluation indicators include tool-call accuracy, task success, safety violation, repair success, latency, and human auditability.

### 3.7 Paper H: Urban low-altitude ODD and semantic functional area planning

**Suggested topic:** ODD2Route: Semantic Operational-Design-Domain Modeling for Low-Altitude UAV Route Planning

**This is a new article that can be written in a new direction. ****Core question:** How does the overall urban scene map to local low-altitude route planning? How to determine the risk, capacity and service strategy of low-altitude air routes based on different functional areas, building density, road structure, crowd activities, no-fly zones and emergency facility distribution?

**Method Route:**

- city-level ODD: OSM road/building/POI/land-use + population/demand proxy;
- local test cell: sample local 3D obstacle/traffic scenario from city ODD;
- route risk model: construction canyons, schools and hospitals, transportation hubs, highway sections, no-fly zones;
- planning output: risk-aware corridor, altitude layer, emergency landing site, charging/vertiport candidates;
- Evaluation: Cross-city generalization, comparing naive shortest path, risk-aware A*, multi-objective MILP, and learning-based route recommender.

**Literature support:**

This article can be supported by Paper B’s TR-C/UAM literature [13-19], Paper D’s scenario coverage literature [27-30], and Paper C’s 3D/digital twin literature [20-26]. The difficulty lies not in the complexity of the algorithm, but in the trustworthy definition of city-level ODD to local scenario/route risk.

**Innovation suggestions:**

1. Establish a computable mapping between the "overall city scene" and "local obstacle combination".
2. Use ODD coverage to interpret scene coverage instead of randomly generating scenes.
3. Provide a bridge between urban low-altitude planning, route design and test scenario library for TR-C/T-ITS.

---

## 4. Recommendation priority| Priority | Articles | Recent Actions | Reasons |
|---|---|---|---|
| P0 | Paper B | Freeze problem formulation, queue model, and experimental benchmark first | Most similar to the TR-C system paper, and most suitable for low-altitude economy/emergency |
| P0 | Paper A | Rewrite PPO/MAPPO into a robust low-altitude conflict resolution paper | Already have the basis of the algorithm, but need traffic indicators and strong baseline |
| P1 | Paper C | Converged to Fisher + 3DGS + safe planning, no longer expand too much | The algorithm is very innovative and can be used in robots/AI/ITS |
| P1 | Paper D | Reuse 76 million exploration logs for coverage-guided testing | The data assets are unique and can easily form a reproducible benchmark |
| P2 | Paper E | Maintain the AAAI/IJCAI method paper route | Suitable for short and fast work but control the scope of experiments |
| P2 | Paper G | Start after the Paper A/B/C/D/E tool interface is stable | CloudBrain-Agent needs to rely on the previous module, otherwise it will be empty |
| P3 | Paper H | As a subsequent extension of TR-C/T-ITS | Requires mature urban data pipeline and ODD definitions |

---

## 5. Zotero organizes status

Target Zotero collection name:

```text
低空规划论文参考
```

A BibTeX file that can be imported into Zotero has been generated locally:

```text
zotero/low-altitude-planning-references-20260527.bib
```

There is no Zotero MCP/connector that can be called in the current environment, and there is no Zotero connector that can be installed, so it is not possible to directly write to the Zotero collection in this round. It has been confirmed that the Zotero executable file and `~/Zotero/zotero.sqlite` exist on this machine, but it is not recommended to directly modify the Zotero SQLite database, as it can easily damage the library structure and synchronization status. The safe approach is:1. Create a new collection in Zotero: `Low Altitude Planning Paper Reference`.
2. Import `zotero/low-altitude-planning-references-20260527.bib`.
3. If you later connect to the Zotero MCP or Better BibTeX automation interface, you can change the script to write directly to the collection.

---

## 6. Follow-up execution plan

### 6.1 Week 1: Freeze Paper Matrix

- Confirm whether Paper A/B/C is the main force of the current three articles.
- Confirm whether Paper D regards the 76 million exploration logs as a core asset.
- Confirm whether Paper E/G continues to be AAAI/IJCAI first.
- Import Zotero collection into BibTeX and add PDF.

### 6.2 Week 2-3: Completing the literature matrix

- Compile at least 25 highly relevant documents for each main article.
- Each article forms a `related work matrix`: problem, method, data, metric, gap, our angle.
- For Paper A/B/C, mark the papers that “must reproduce the baseline” and “only serve as related work”.

### 6.3 Weeks 4-8: Advance the three experimental lines of Paper B/A/C first

- Paper B: synthetic UAM queuing benchmark + FCFS/greedy/MILP/backpressure/MARL baseline.
- Paper A: corridor conflict simulation + ORCA/CBF/RMADER/MAPPO baseline.
- Paper C: 3DGS NBV pipeline + FisherRF/ActiveNeRF/GS-Planner/POp-GS baseline.

### 6.4 Weeks 9-12: Deciding on your first submission- If Paper B’s queue stability and hundred-shelf level results are the most stable: vote for TR-C first.
- If Paper A has the strongest conflict safety and generalization: vote for T-ITS/T-RO first.
- If Paper C has the strongest Fisher + 3DGS theoretical and visual results: vote for T-RO/ICRA/IROS first.
- If D has the best coverage/failure discovery data: invest in T-ITS first.

---

## 7. References

[1] Elsevier. *Transportation Research Part C: Emerging Technologies: Aims and Scope.* URL: <https://www.sciencedirect.com/journal/transportation-research-part-c-emerging-technologies>

[2] IEEE Intelligent Transportation Systems Society. *IEEE Transactions on Intelligent Transportation Systems: Scope.* URL: <https://ieee-itss.org/pub/t-its/>

[3] Chao Yu, Akash Velu, Eugene Vinitsky, Yu Wang, Alexandre M. Bayen, and Yi Wu. “The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games.” *Advances in Neural Information Processing Systems*, 2022. URL: <https://arxiv.org/abs/2103.01955>[4] Muning Wen, Jakub Grudzien Kuba, Runji Lin, Weinan Zhang, Ying Wen, Jun Wang, and Yaodong Yang. “Multi-Agent Reinforcement Learning is a Sequence Modeling Problem.” *NeurIPS*, 2022. URL: <https://proceedings.neurips.cc/paper_files/paper/2022/hash/69413f87e5a34897cd010ca698097d0a-Abstract-Conference.html>

[5] Bei Peng, Tabish Rashid, Christian Schroeder de Witt, Pierre-Alexandre Kamienny, Philip Torr, Wendelin Boehmer, and Shimon Whiteson. “FACMAC: Factored Multi-Agent Centralized Policy Gradients.” *NeurIPS*, 2021. URL: <https://proceedings.neurips.cc/paper/2021/hash/65b9eea6e1cc6bb9f0cd2a47751a186f-Abstract.html>

[6] Jakub Grudzien Kuba, Ruiqing Chen, Muning Wen, Ying Wen, Fudan Sun, Jun Wang, and Yaodong Yang. "Trust Region Policy Optimization in Multi-Agent Reinforcement Learning." arXiv:2109.11251, 2021. URL: <https://arxiv.org/abs/2109.11251>[7] Boyu Zhou, Xin Zhou, Jun Zhang, Fei Gao, and Shaojie Shen. "EGO-Swarm: A Fully Autonomous and Decentralized Quadrotor Swarm System in Cluttered Environments." *ICRA*, 2021. DOI: 10.1109/ICRA48506.2021.9561902. URL: <https://arxiv.org/abs/2011.04183>

[8] Jesus Tordesillas, Brett T. Lopez, and Jonathan P. How. “MADER: Trajectory Planner in Multiagent and Dynamic Environments.” *IEEE Transactions on Robotics*, 38(1):463-476, 2022. URL: <https://arxiv.org/abs/2010.11061>

[9] Kota Kondo, Reinaldo Figueroa, Juan Rached, Jesus Tordesillas, Parker C. Lusk, and Jonathan P. How. “Robust MADER: Decentralized Multiagent Trajectory Planner Robust to Communication Delay in Dynamic Environments.” arXiv:2303.06222, 2023. URL: <https://arxiv.org/abs/2303.06222>[10] Boyu Zhou, Hao Xu, and Shaojie Shen. "RACER: Rapid Collaborative Exploration With a Decentralized Multi-UAV System." *IEEE Transactions on Robotics*, 2023. DOI: 10.1109/TRO.2023.3236945. URL: <https://arxiv.org/abs/2209.08533>

[11] Jesus Tordesillas and Jonathan P. How. “PANTHER: Perception-Aware Trajectory Planner in Dynamic Environments.” *IEEE Access*, 10:22662-22677, 2022. DOI: 10.1109/ACCESS.2022.3154037. URL: <https://arxiv.org/abs/2103.06372>

[12] Zhepei Wang, Xin Zhou, Chao Xu, and Fei Gao. "Geometrically Constrained Trajectory Optimization for Multicopters." *IEEE Transactions on Robotics*, 38(5):3259-3278, 2022. DOI: 10.1109/TRO.2022.3160022. URL: <https://arxiv.org/abs/2103.00190>[13] Ang Li, Mark Hansen, and Bo Zou. “Traffic Management and Resource Allocation for UAV-Based Parcel Delivery in Low-Altitude Urban Space.” *Transportation Research Part C: Emerging Technologies*, 143:103808, 2022. DOI: 10.1016/j.trc.2022.103808. URL: <https://doi.org/10.1016/j.trc.2022.103808>

[14] Mehdi Bennaceur, Rémi Delmas, and Youssef Hamadi. “Passenger-Centric Urban Air Mobility: Fairness Trade-Offs and Operational Efficiency.” *Transportation Research Part C: Emerging Technologies*, 136:103519, 2022. DOI: 10.1016/j.trc.2021.103519. URL: <https://doi.org/10.1016/j.trc.2021.103519>

[15] Roberto Pinto and Alexandra Lagorio. “Point-to-Point Drone-Based Delivery Network Design with Intermediate Charging Stations.” *Transportation Research Part C: Emerging Technologies*, 135:103506, 2022. DOI: 10.1016/j.trc.2021.103506. URL: <https://doi.org/10.1016/j.trc.2021.103506>[16] Qinshuang Wei, Gustav Nilsson, and Samuel Coogan. “Capacity-Constrained Urban Air Mobility Scheduling.” arXiv:2107.02900, 2021. URL: <https://arxiv.org/abs/2107.02900>

[17] Surya Murthy, Natasha A. Neogi, and Suda Bharadwaj. “Scheduling for Urban Air Mobility Using Safe Learning.” arXiv:2209.15457, NASA NTRS, 2022. URL: <https://arxiv.org/abs/2209.15457>

[18] Jiahao Xing, Tong Guo, and Lu Tong. "Reliable Truck-Drone Routing with Dynamic Synchronization: A High-Dimensional Network Programming Approach." *Transportation Research Part C: Emerging Technologies*, 165:104698, 2024. DOI: 10.1016/j.trc.2024.104698. URL: <https://doi.org/10.1016/j.trc.2024.104698>

[19] Bolong Zhou, Wei Zeng, and Hai Yang. "Multi-Trip UAV-UGV Delivery Network Design with Release Times." *Transportation Research Part C: Emerging Technologies*, 181:105389, 2025. DOI: 10.1016/j.trc.2025.105389. URL: <https://doi.org/10.1016/j.trc.2025.105389>[20] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. “3D Gaussian Splatting for Real-Time Radiance Field Rendering.” *ACM Transactions on Graphics / SIGGRAPH*, 42(4), 2023. DOI: 10.1145/3592433. URL: <https://arxiv.org/abs/2308.04079>

[21] Xuran Pan, Zihang Lai, Shiji Song, and Gao Huang. "ActiveNeRF: Learning Where to See with Uncertainty Estimation." *ECCV*, 2022. URL: <https://arxiv.org/abs/2209.08546>

[22] Wen Jiang, Boshu Lei, and Kostas Daniilidis. “FisherRF: Active View Selection and Mapping with Radiance Fields Using Fisher Information.” *ECCV*, 2024. DOI: 10.1007/978-3-031-72624-8_24. URL: <https://eccv.ecva.net/virtual/2024/oral/1226>

[23] Rui Jin, Yuman Gao, Yingjian Wang, Haojian Lu, and Fei Gao. "GS-Planner: A Gaussian-Splatting-Based Planning Framework for Active High-Fidelity Reconstruction." arXiv:2405.10142, 2024. URL: <https://arxiv.org/abs/2405.10142>[24] Zijun Xu, Rui Jin, Ke Wu, Yi Zhao, Zhiwei Zhang, Jieru Zhao, Fei Gao, Zhongxue Gan, and Wenchao Ding. "HGS-Planner: Hierarchical Planning Framework for Active Scene Reconstruction Using 3D Gaussian Splatting." arXiv:2409.17624, 2024. URL: <https://arxiv.org/abs/2409.17624>

[25] Joey Wilson, Marcelino Almeida, Sachit Mahajan, Martin Labrie, Maani Ghaffari, Omid Ghasemalizadeh, Min Sun, Cheng-Hao Kuo, and Arnab Sen. “POp-GS: Next Best View in 3D-Gaussian Splatting with P-Optimality.” *CVPR*, 2025. URL: <https://cvpr.thecvf.com/virtual/2025/poster/34708>

[26] Shangjie Xue, Jesse Dill, Pranay Mathur, Frank Dellaert, Panagiotis Tsiotras, and Danfei Xu. “Neural Visibility Field for Uncertainty-Driven Active Mapping.” *CVPR*, 2024. URL: <https://arxiv.org/abs/2406.06948>[27] Shuo Feng, Xintao Yan, Haowei Sun, Yiheng Feng, and Henry X. Liu. “Intelligent Driving Intelligence Test for Autonomous Vehicles with Naturalistic and Adversarial Environment.” *Nature Communications*, 12:748, 2021. DOI: 10.1038/s41467-021-21007-8. URL: <https://www.nature.com/articles/s41467-021-21007-8>

[28] Shuo Feng, Yiheng Feng, Chao Yu, Yi Zhang, and Henry X. Liu. “Testing Scenario Library Generation for Connected and Automated Vehicles, Part I: Methodology.” *IEEE Transactions on Intelligent Transportation Systems*, 2021. DOI: 10.1109/TITS.2020.2972211. URL: <https://doi.org/10.1109/TITS.2020.2972211>

[29] Shuo Feng, Yiheng Feng, Haowei Sun, Yi Zhang, and Henry X. Liu. "Testing Scenario Library Generation for Connected and Automated Vehicles, Part II: Case Studies." *IEEE Transactions on Intelligent Transportation Systems*, 2021. DOI: 10.1109/TITS.2020.2988309. URL: <https://doi.org/10.1109/TITS.2020.2988309>[30] Chejian Xu, Wenhao Ding, Baiming Li, Xinqing Chen, Ding Zhao, Bo Li, Jiajun Liu, and Hang Zhao. “SafeBench: A Benchmarking Platform for Safety Evaluation of Autonomous Vehicles.” *NeurIPS Datasets and Benchmarks*, 2022. URL: <https://proceedings.neurips.cc/paper_files/paper/2022/hash/a48ad12d588c597f4725a8b84af647b5-Abstract-Datasets_and_Benchmarks.html>

[31] Jason Liu, Ankit Shah, Eric Rosen, and Stefanie Tellex. “Lang2LTL: Translating Natural Language Commands to Temporal Robot Task Specification.” *PMLR/CoRL*, 229, 2023. URL: <https://proceedings.mlr.press/v229/liu23d.html>

[32] Francesco Fuggitti and Tathagata Chakraborti. “NL2LTL: A Python Package for Converting Natural Language Instructions to Linear Temporal Logic Formulas.” *AAAI Demonstration*, 37(13):16428-16430, 2023. DOI: 10.1609/aaai.v37i13.27068. URL: <https://ojs.aaai.org/index.php/AAAI/article/view/27068>[33] Behrad Rabiei and Mahesh A. Kumar. “LTLCodeGen: Code Generation of Syntactically Correct Temporal Logic for Robot Task Planning.” arXiv:2503.07902, 2025. URL: <https://arxiv.org/abs/2503.07902>

[34] Jun Wang, David Smith Sundarsingh, Jyotirmoy V. Deshmukh, and Yiannis Kantaros. “ConformalNL2LTL: Translating Natural Language Instructions into Temporal Logic Formulas with Conformal Correctness Guarantees.” arXiv:2504.21022, 2025. URL: <https://arxiv.org/abs/2504.21022>

[35] Zhonghang Li, Lianghao Xia, Jiabin Tang, Yong Xu, Lei Shi, Long Xia, Dawei Yin, and Chao Huang. "UrbanGPT: Spatio-Temporal Large Language Models." arXiv:2403.00813, 2024. URL: <https://arxiv.org/abs/2403.00813>

[36] Yuan Yuan, Jingtao Ding, Jie Feng, Depeng Jin, and Yong Li. “UniST: A Prompt-Empowered Universal Model for Urban Spatio-Temporal Prediction.” *KDD*, 2024. DOI: 10.1145/3637528.3671662. URL: <https://dblp.org/rec/conf/kdd/0032D0J024>[37] Jinhui Ouyang, Yijie Zhu, Xiang Yuan, and Di Wu. “TrafficGPT: Towards Multi-Scale Traffic Analysis and Generation with Spatial-Temporal Agent Framework.” arXiv:2405.05985, 2024. URL: <https://arxiv.org/abs/2405.05985>

[38] Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Jens Beisswenger, Ping Luo, Andreas Geiger, and Hongyang Li. “DriveLM: Driving with Graph Visual Question Answering.” *ECCV*, 2024. URL: <https://github.com/OpenDriveLab/DriveLM>