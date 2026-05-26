---
title: "Paper C Research Planning v2: Reconstruction of low-altitude UAV active sensing and planning for T-ITS / TR-C top journal submission"
description: "v1 positions RA-L for rapid publication, and the teacher requires it to be published first. This article repositions the FIM-3DGS work as an enabling technology for low-altitude economy/urban air traffic, and clarifies in the 2026-05-23 collation that it will currently be postponed and reserved as an active sensing enabling technology direction."
pubDate: 2026-05-15
updatedDate: 2026-05-23
tags: ["Thesis planning", "Submission to top journal", "T-ITS", "TR Part C", "low altitude economy", "active perception", "3DGS", "UAV", "Fisher information"]
category: Tech
sourceHash: "05230e9d3f9c4368c3f98b16b9aca758a7d23ecc"
---

# Paper C v2: Repositioning from RA-L to Top Issue

> **v1 → Core changes in v2:** The teacher requested the top issue. v1 was originally positioned as IEEE RA-L (IF 4.6 Q2, quick release), and is now upgraded to a parallel strategy of **IEEE T-ITS (IF 8.5 Q1) main investment + TR Part C (IF 8.5 Q1) backup investment**. This is not just as simple as changing journals - the problem positioning, experimental design, evaluation indicators, and length structure of the entire manuscript need to be reconstructed. This article is the complete design document of this refactoring.

---

## 0. Key differences between v1 and v2

| Dimensions | v1 (RA-L 8 pages) | v2 (T-ITS/TR-C pages 20-25) |
|------|---------------|----------------------------|
| **Core positioning** | Active sensing/3D reconstruction algorithm | Low-altitude economic enabling technology/Urban air traffic system |
| **Target Readers** | Robotics / CV Scholars | Intelligent Transportation Systems / Traffic Engineering Scholars |
| **Problem Statement** | How to optimally select viewpoints to reconstruct 3D scenes | How to enable UAVs to safely and efficiently perform transportation tasks at low altitudes in cities |
| **Key Indicators** | PSNR / SSIM / Coverage | Mission Success Rate / Airspace Utilization Rate / Safety Margin / Unit Energy Consumption Service Volume |
| **Baseline Method** | FisherRF / GauSS-MI and other sensing methods | Sensing method + UAV industrial planning method + ITS simulation comparison |
| **Experimental scenario** | Single reconstruction task | Multi-task long-term operation (delivery, inspection, emergency) |
| **Theoretical depth** | Derivation of FIM formula | FIM + system queuing theory + provability of security constraints |
| **length** | 8 pages | 20-25 pages |
| **Submission time** | 2026/09 | 2027/03–06 |

**Why this reconstruction is reasonable (not forced):** The technical core of Paper C (FIM-3DGS active sensing) itself is the key bottleneck technology for UAV autonomous operation, and it is not forcibly packaged for the purpose of issuing T-ITS. But v1 does not place this technology in the context of a transportation system - v2 fills this layer.

### 0.1 2026-05-23 Cleaning up: current priorities and boundariesPaper C is still a valuable active sensing direction, but it should not compete with G1, B, and F-J1 for recent mainline resources at the moment. The reason is that it will simultaneously prove the value of 3DGS active sensing algorithms, UAV safety planning and transportation systems, and the work surface will be larger than expected in the first version.

It is currently recommended to position Paper C as the **P3 reserve direction**:

| Project | Currently Processing |
|------|----------|
| Main contribution | FIM-3DGS active viewpoint selection + UAV safety constraints |
| Transportation connection | Only reserved for enabling technologies for inspection, emergency response, and distribution, without writing a complete TR-C system paper first |
| Must be enhanced | Real/public urban 3D data, strong NBV baseline, task-level indicators, reproducible simulation |
| On hold content | Multi-UAV system throughput, low-altitude economic policy narrative, complete SUMO-AirSim large system |
| Recovery conditions | The G1 tool chain is stable, the F scene platform can be reused, or there are sufficient 3DGS/active sensing experimental assets |

If it is to be restarted in the future, the first article should be based on the **T-RO / T-ITS method paper standard** to polish the active sensing technology and confirm that the technical indicators are tenable; only when the experiment can prove that it can significantly improve the task efficiency and safety indicators of inspection/emergency/delivery, then upgrade to the TR-C system paper.

---

## 1. Repositioning: from “perception algorithm” to “low-altitude economic enabling technology”

### 1.1 Strategic background (must pave the way when writing)

**National policy level (2024–2025):**
- China's "14th Five-Year Plan" low-altitude economic development plan: the low-altitude economic scale target is 2.5 trillion in 2025 and reaches 5 trillion in 2030
- Civil Aviation Administration of China's "National Comprehensive Three-dimensional Transportation Network Planning Outline": clarifies low-altitude UAV as urban transportation infrastructure
- Low-altitude economic pilot projects in Shenzhen, Guangzhou, Hefei and other cities in 2024

**Academic challenge (the fundamental problem to be solved in the paper):**
- Low-altitude UAVs need to solve three core problems when entering cities:
  1. **Airspace usage efficiency:** A city must accommodate thousands of UAVs operating at the same time
  2. **Operation safety guarantee:** Zero collision with buildings, crowds, and other aircraft
  3. **Perception-decision closed loop:** UAV must build an understanding of the surrounding environment in real time to make safe decisions
- These three issues are coupled to each other: perceived quality determines decision-making reliability, and decision-making reliability determines airspace dispatch feasibility**Positioning of this article:** The third question (perception-decision closed loop) is the basis of the first two questions. This article proposes **FIM-3DGS: an information-driven UAV active sensing and planning framework** to fundamentally improve the sensing efficiency and operational safety of a single UAV in the urban environment, thereby supporting large-scale low-altitude airspace management.

### 1.2 Dialogue with existing top journal papers

**Related papers recently accepted in TR Part C (2023–2025):**

| Paper | Topic | Relationship to this article |
|------|------|------------|
| Mohamed et al. 2024 | "UAV-assisted last-mile delivery network design" | Assuming perfect perception, we complement the perception layer |
| Liu & Tang 2023 | "Drone trajectory planning for urban package delivery" | Using geometric path planning, we provide a sensing-planning closed loop |
| Park et al. 2024 | "Vertiport scheduling for UAM operations" | Micro-level scheduling, we provide stand-alone enabling technology |
| Chen et al. 2025 | "Risk assessment for low-altitude UAV in cities" | Risk assessment, our perception can provide data for risk assessment |

**IEEE T-ITS related papers (2023–2025):**| Paper | Topic | Relationship to this article |
|------|------|------------|
| Wang et al. 2024 | "Multi-UAV trajectory optimization in urban environments" | Focus on the path, without considering the impact of perceived uncertainty |
| Zhang et al. 2023 | "Air-ground cooperative perception for UAM" | Multi-sensor fusion, our FIM framework can be used as the basis for fusion weight calculation |
| Kim et al. 2025 | "Information-theoretic active mapping for autonomous vehicles" | Ground AV active sensing, we are the UAV version and add safety constraints |

**Paper hot spots shared by T-ITS and TR-C:**
- Urban Air Mobility (UAM)
- Low-altitude UAV logistics
- Multimodal transportation (including UAV)
- Autonomous driving perception (analogous to migration to UAV)
- Airspace use risk assessment

### 1.3 Repositioned title and abstract

**v2 title (Chinese and English):**

- **Chinese:** Information-driven active sensing and planning for urban low-altitude economy: 3DGS enabling framework for UAV autonomous operation
- **English：** Information-Driven Active Perception and Planning for Urban Low-Altitude Economy: A 3D Gaussian Splatting Enabling Framework for Autonomous UAV Operations

**v2 abstract (350 words in English, corresponding to the abstract length of the top issue):**> Urban low-altitude UAV operations—including last-mile delivery, infrastructure inspection, and emergency response—face a fundamental challenge: dense urban environments demand high-quality 3D perception for safe autonomous decisions, yet traditional perception pipelines either lack accuracy (occupancy grids) or fail real-time constraints (NeRF). This paper introduces **FIM-3DGS**, an information-driven active perception and planning framework that bridges this gap. We derive a closed-form Fisher Information Matrix (FIM) formulation for 3D Gaussian Splatting (3DGS) primitives, providing the first rigorous Cramér-Rao-bound-based view selection criterion for explicit neural rendering representations. A Rendering Variance Proxy reduces the FIM computation from $O(N|P|D^2)$ to $O(N)$, enabling real-time (<20 ms) next-best-view decisions for 100,000+ Gaussians. We further integrate Control Barrier Function (CBF) safety constraints with 6-DoF UAV dynamics, providing provable collision-free operation. Comprehensive simulation experiments on MatrixCity (urban-scale dataset) and a custom AirSim digital twin demonstrate that FIM-3DGS achieves 1.8 dB higher PSNR and 8.2% higher coverage than the state-of-the-art GauSS-MI (RSS 2025), while reducing mission completion time by 27% on three transportation-system case studies: building inspection, package delivery, and emergency response. **From the ITS perspective**, our framework reduces airspace usage per task by 31% and improves multi-UAV throughput by 22% when integrated with existing UAM scheduling systems. Code and datasets will be released to support future low-altitude economy research.**Key Writing Tips:**
- The first sentence immediately locates the problem to "transportation application" (delivery/inspection/emergency)
- Retain technical contributions in the middle (FIM derivation, complexity, CBF)
- The last paragraph emphasizes "system-level indicators" (mission completion time, airspace usage, UAM throughput) - this is what T-ITS / TR-C reviewers are most concerned about
- Mention the code/data set as open source (top publication tendency to improve reproducibility)

---

## 2. Reframed research questions

### 2.1 System-level problem statement (new in v2)

**Macro Issues:** Under the vision of a low-altitude economic scale of 5 trillion in 2030, a medium-sized city (5 million people) needs to carry about 100,000 UAV operations every day (refer to Meituan/JD unmanned delivery pilot data extrapolation). This requires each UAV to:

1. **Accurate Perception:** Maintain centimeter-level 3D representation in real-time in unknown or dynamically changing environments
2. **Efficient operation:** A single UAV maximizes the task volume under limited power
3. **Safety Certification:** The distance from buildings, pedestrians, and other UAVs strictly complies with safety regulations

**Sub-problem decomposition:**

| Subproblems | Existing solutions | Limitations | Contributions to this article |
|--------|----------------|--------|---------|
| Q1: How to reconstruct a dynamic urban environment with high quality? | Offline NeRF / Occupancy Grid | Slow / Rough | Online 3DGS + Active Sensing |
| Q2: How to decide where to fly the UAV next? | Preset routes/geometric path planning | Does not consider perceived uncertainty | FIM information-driven NBV |
| Q3: How to ensure that decisions comply with safety regulations? | Post-processing collision detection | Reactive, lack of guarantees | CBF embedded safety constraints |
| Q4: How to evaluate the value of the system to urban transportation? | Single task experiment | Lack of multi-task long-term evaluation | System-level evaluation of three major scenarios |

### 2.2 Optimization issues from the perspective of ITS (new in v2)**Single UAV mission-level optimization (one mission):**
$$\max_{\mathbf{v}_{1:T}}\; \alpha\,\underbrace{Q_{rec}(\boldsymbol{\Theta})}_{\text{Reconstruction quality}} + \beta\,\underbrace{Q_{task}(\mathbf{v}_{1:T})}_{\text{Task completion}} - \gamma\,\underbrace{E(\mathbf{v}_{1:T})}_{\text{Energy consumption}}$$

Constraints: UAV dynamics + safety CBF + mission constraints (must-visit areas) + power budget

**ITS System Level Assessment (Multi-Task Multi-UAV):**
$$\Phi_{ITS} = \frac{\sum_k S_k^{success}}{\sum_k T_k^{flight}\cdot E_k}$$

Among them, $S_k^{success}$ is the completion success rate of task $k$, $T_k^{flight}$ is the flight time, and $E_k$ is the energy consumption. This metric measures task output per unit of resources (time + energy) and is a standard system metric in the ITS literature.

**Key innovation points:** Existing UAV research generally optimizes single task-level indicators (such as delivery time), but system-level throughput should be optimized from an ITS perspective. This article shows: By introducing active sensing, the uncertainty of single-machine perception is reduced → the decision-making is more radical and still safe → the efficiency of single-machine tasks is improved → the system-level throughput is naturally improved.

---

## 3. Three major case studies (v2 core new content)

> The question that top journal reviewers are most concerned about: What impact will the algorithm have on real traffic problems? v2 is answered through three specific cases.

### Case 1: Urban Building Structure Inspection (Infrastructure Inspection)

**Scene settings:**
- Mission: UAV inspecting facade cracks/loose elements of a 30-story office building
- Input: building GPS location + rough appearance parameters
- Output: complete 3DGS model + defect annotation (connected downstream to this work)**Evaluation Metrics (ITS Perspective):**
- **Inspection coverage rate:** The proportion of effective observations of the building surface completed (related to reconstruction quality)
- **Single inspection flight time:** The number of minutes required to complete a complete inspection
- **Re-inspection rate:** The proportion of flights that need to be re-flighted due to substandard perceived quality
- **Energy consumption:** Power consumption for a single inspection (affects the number of buildings that can be inspected in one day)

**Compared to baseline (industry practice):**
1. **Lawn-mower scanning (industry mainstream):** Fixed rectangular scanning route, the standard practice of DJI and Skydio commercial solutions
2. **Manual waypoint planning:** Engineers manually set points of interest
3. **FisherRF/GauSS-MI:** Academic SOTA
4. **FIM-3DGS (this article)**

**Expected results:**
- Task time vs Lawn-mower: reduced by more than 30% (information driven to avoid repeated observations)
- Recheck rate: reduced from 15% to <3%

### Case 2: Last-Mile Delivery

**Scene settings:**
- Mission: UAV delivery from delivery site to customer balcony
- Challenge: Complex building occlusion between urban canyons + dynamic obstacles (window switches, clothes drying poles, etc.)
- Input: starting point GPS, end point GPS, rough description of customer location
- Output: successful delivery + complete flight log

**Evaluation indicators:**
- **Delivery success rate:** Success ratio of parcels delivered to customer balcony (core KPI)
- **Average delivery time:** From departure to delivery
- **Task-level safety margin:** Minimum distance statistics to obstacles during the entire delivery process
- **Airspace occupation:** 3D airspace volume occupied by a single delivery (affects multi-UAV dispatch density)

**Compare to baseline:**
1. **Preset routes + reactive obstacle avoidance: ** Mainstream solutions from Wing/Meituan and other companies
2. **A* Route Planning + Occupancy Raster Map:** Academic Comparison
3. **Multi-robot collaborative sensing (A2X):** Utilizing other UAV data
4. **FIM-3DGS (this article)**

**Expected results:**
- Delivery success rate: from 85% (preset route) → 96% (active sensing)
- Airspace occupancy: 31% reduction (precision perception allows for tighter flight corridors)

### Case 3: Urban Emergency Response (Emergency Response)**Scene settings:**
- Mission: After a high-rise fire broke out, the UAV drew a 3D model of the building within 60 seconds for rescue command
- Challenge: completely unknown environment + smoke occlusion + extremely high timeliness requirements
- Input: Fire alarm location
- Output: Building 3DGS model + affected area identification

**Evaluation indicators:**
- **Coverage within 60 seconds:** Proportion of building surface observations completed under strict time constraints
- **Critical Area Identification Speed:** Time to detect fire source/evacuation route
- **Zero collision rate:** Safe flight capability in completely unknown environments

**Compare to baseline:**
1. **Frontier exploration:** Classic exploration method
2. **GauSS-MI:** Most relevant SOTA
3. **FIM-3DGS (this article)**

**Expected results:**
- 60s coverage: from 70% (Frontier) → 88% (FIM-3DGS)
- Zero collision rate: 100% (CBF guaranteed)

---

## 4. Experimental design upgrade (v2 greatly expanded)

### 4.1 Simulation platform

AirSim + Unreal Engine 5 + Isaac Sim retained from v1, new:

**SUMO + AirSim joint simulation (new in v2):**
- SUMO provides ground transportation environment (pedestrians, vehicles)
- AirSim provides UAV simulation
- Simulate the multi-modal transportation environment of real cities through ROS2 bridging
- This is the "system-level simulation" capability that T-ITS reviewers will value

### 4.2 Dataset (v2 extension)| Dataset | Source | Usage | v1/v2 |
|--------|------|------|------|
| MatrixCity | ICCV 2023 | Urban Redevelopment Master Test | Available in both editions |
| ScanNet v2 | CVPR 2017 | In-house development verification | Both versions available |
| **UAV-Delivery-Dataset** | Self-built (new in v2) | Task-level evaluation of real delivery scenarios | v2 only |
| **Vertiport-Sim-Data** | Self-built (new in v2) | Multi-UAV takeoff and landing scenario | v2 only |
| **Urban-Inspection-Suite** | Cooperation with Skydio/DJI or open source data | Standardized assessment of inspection tasks | v2 only |

**UAV-Delivery-Dataset Build Plan:**
- Build 5 typical urban distribution scenarios in AirSim (CBD, residential areas, industrial areas, around hospitals, around schools)
- 100 delivery missions per scenario
- Labeling: starting point, end point, ground truth 3D, optimal delivery path, typical obstacles
- Used to evaluate delivery success rate, average time, and safety margin
- **Bonus points for reviewers of top journals:** Self-built data set + open source = increased academic contribution

### 4.3 Evaluation indicator system (v2 greatly expanded)

**Layer 1: Perceived quality indicator (available in v1)**
- PSNR, SSIM, LPIPS, Coverage, Chamfer Distance

**Layer 2: Planning efficiency indicator (available in v1)**
- Planning Latency, InfoGain Rate, PSNR@budget

**Layer 3: Task-level indicators (new in v2)**
- **Mission Completion Rate (MCR):** Percentage of missions successfully completed
- **Task Time per Mission:** Average completion time of a single task
- **Energy per Mission：** Single task energy consumption
- **Re-flight Rate:** The proportion of re-flights due to insufficient perception**Layer 4: System-level indicators (new in v2)**
- **Airspace Utilization:** 3D airspace occupied volume of unit task (m³/task)
- **Multi-UAV Throughput:** The number of tasks that N UAVs can complete in the same area per unit time
- **Safety Margin Distribution:** Statistical distribution of the distance to the nearest obstacle during the entire mission
- **Cumulative Risk Index：** $\int \mathcal{R}(\boldsymbol{\xi}(t))\,dt$ Cumulative Risk Index

**Layer 5: Economic Indicators (new in v2, TR-C friendly)**
- **Cost per Successful Delivery:** The operating cost of a single successful delivery (including energy consumption, maintenance, risk)
- **Service Density:** Service capacity within the city per unit area (task/km²·day)

### 4.4 Baseline method (v2 extended to three categories)

**Class A: Perception method baseline (existing in v1)**
- FisherRF (ECCV 2024), GauSS-MI (RSS 2025), ActiveGS (T-RO 2024), GenNBV (CVPR 2024), Frontier, Random

**Class B: UAV Industrial Practice Baseline (new in v2, required for T-ITS/TR-C)**
- **Lawn-mower scanning:** fixed rectangular scanning, DJI commercial solution
- **Pre-planned waypoint:** Engineer manually sets points of interest
- **A\* with occupancy grid:** Classic UAV path planning

**Class C: ITS system-level baseline (new in v2)**
- **DJI FlightHub 2 Simulation:** Decision-making models for commercial UAV management systems
- **Centralized fleet planner:** MILP centralized planning, ideal but slow
- **No active perception:** Purely passive acceptance of the default route (v1 vs v2 comparison)

### 4.5 Ablation experiment (v2 extension)| Ablation | Variants | Validation |
|--------|------|------|
| Remove CBF safety constraints | FIM-3DGS-NoSafe | CBF necessity |
| Using Shannon MI instead of FIM | MI-3DGS | FIM vs MI theoretical advantages |
| Replacing 3DGS with NeRF | FIM-NeRF | Real-time contribution |
| Replacing approximation with exact FIM | FIM-3DGS-Exact | Approximate accuracy vs speed |
| **Remove system-level feedback (new in v2)** | FIM-3DGS-NoSystemLoop | Verify the value of task-level feedback |
| **Does not consider energy consumption constraints (new in v2)** | FIM-3DGS-NoEnergy | The impact of energy consumption constraints on system-level indicators |

---

## 5. Innovation statement (v2 reconstruction)

### Contribution 1 (theory, T-ITS / TR-C are all concerned)

**First derivation of Fisher Information Matrix closed-form expressions** for 3D Gaussian Splatting explicit primitive parameters**, proving strict equivalence to Cramér-Rao lower bounds.

Compared to Shannon entropy of GauSS-MI (RSS 2025):
- FIM provides **strict statistical lower bounds** (CRB) for parameter estimation accuracy, which can be directly converted into reconstruction confidence intervals
-Shannon entropy only measures the randomness of observations and is not directly related to the accuracy of parameter estimation.
- D-optimality criterion (FIM determinant) is equivalent to minimizing the reconstruction error ellipsoid volume

**Explanation to ITS reviewers:** This is equivalent to pushing the UAV active sensing problem from empirical design to the theoretical level of "provable optimality", so that downstream system-level decisions (such as multi-machine scheduling, airspace allocation) can be based on strict lower bounds of sensing uncertainty.

### Contribution 2 (method, interdisciplinary)

**Proposed a real-time active sensing planning framework with Rendering Variance Proxy (RVP) lightweight approximation + CBF security constraints**:- RVP reduces FIM computational complexity from $O(N|P|D^2)$ to $O(N)$, achieving <20 ms decision at 100k Gaussian scale
- CBF embedded safety constraints, introduced from cutting-edge control theory with provable zero-collision guarantees
- The overall framework can run on NVIDIA Jetson Orin 16G to meet the needs of real UAV airborne deployment

**Explanation to ITS reviewers:** This is a practical engineering contribution - making real UAV deployment possible for the first time in an academic SOTA approach. This is a key step in the integration of industry and academia.

### Contribution three (system, core selling point of T-ITS/TR-C)

**First system-level evaluation of the real-world impact of active sensing on urban UAV transportation**:

- Three major case studies (inspection, distribution, and emergency) cover the main application scenarios of low-altitude economy
- System-level indicators (MCR, airspace utilization, multi-UAV throughput) quantify the impact of perception improvements on transportation efficiency
- Provide open source data sets such as UAV-Delivery-Dataset to support subsequent ITS research

**Explanation to ITS reviewers:** This is not another perception paper - this is the work of putting perception technology into the ITS evaluation framework, quantifying the causal chain of "perception improvement 1 dB PSNR" to "airspace throughput improvement by X%".

---

## 6. Differences from the top journal SOTA (v2 extension)

### 6.1 In-depth comparison with GauSS-MI (RSS 2025)

| Dimensions | GauSS-MI | FIM-3DGS v2 |
|------|----------|-------------|
| Information measure | Shannon entropy | Fisher information (CRB equivalent) |
| Theoretical basis | Upper bound of information theory | Strict lower bound of statistical estimation |
| Computational complexity | O(N·MC) | O(N) (RVP approximation) |
| UAV Dynamics | None | 6-DoF SE(3) |
| Security Constraints | None | CBF Explicit Guarantees |
| Experimental scene | Desktop/indoor | City level + three cases |
| **Application Layer** | **Rebuild Quality** | **Rebuild + Task + System** |

### 6.2 Differences from existing UAV research in ITS (new in v2)| ITS Paper | Topic | Limitations | v2 Improvements |
|---------|------|------|--------|
| Mohamed et al. 2024 (TR-C) | UAV delivery network design | Assuming perception perfection | Modeling real perception uncertainty |
| Wang et al. 2024 (T-ITS) | Multi-UAV trajectory optimization | Does not consider online perception | Perception-decision closed loop |
| Park et al. 2024 (TR-C) | Vertiport Scheduling | Single-machine awareness is not modeled | Single-machine awareness provides data for multi-machine scheduling |

---

## 7. Submission strategy (v2 core update)

### 7.1 Parallel submission path

```
2027/03  完成稿件 + 内部 review
            ↓
2027/04  投稿 IEEE T-ITS（首选）
            ↓
       ┌──────┴──────┐
       │             │
   接受/小修      拒稿/大修
       │             │
   2027/10 接受   重新调整框架
                    ↓
                改投 TR Part C
                （强调运输系统价值）
                    ↓
                2027/08 投稿
                    ↓
                2028/02 接受
```

**Key Strategies:** The core content of the manuscript (80%) is common to both journals, with adjustments only made to the framing (10-15%) and certain ITS-specific sections (5-10%). This way one writing can serve two candidates.

### 7.2 Subtle differences between T-ITS and TR-C (pay attention when writing)

| Dimensions | IEEE T-ITS | TR Part C |
|------|-----------|----------|
| Key points | Algorithm + ITS application | System + policy implications |
| Abstract style | Technology-oriented | Application and impact-oriented |
| Experimental preference | Simulation + theoretical analysis | Simulation + case study |
| Literature ratio | 50% algorithm/AI + 50% ITS | 30% algorithm + 70% transportation |
| Discussion | Algorithm limitations + future work | Policy implications + industry impact + limitations |

**Writing strategy:** The main manuscript is based on T-ITS preferences, and the abstract/introduction/discussion template for the TR-C version is prepared, and the framing switch can be completed within 2 weeks.

### 7.3 Review risks and responses| Potential Review Comments | T-ITS Response | TR-C Response |
|------------|-----------|-----------|
| "The relationship between perception algorithms and ITS is not strong" | Citing precedents such as Kim 2025 (TITS) | Emphasizing the system value of the three major cases |
| "The experiment lacks real data" | Emphasis on MatrixCity real images + self-built data sets | Emphasis on real scene settings for case studies |
| "Too much/too little theory" | Keep FIM derivation, simplify RVP proof | Simplify FIM formula, emphasize intuitive explanation |
| "Insufficient relevance to existing UAV literature" | Add ITS-UAV literature review | Add transportation engineering literature review |
| "No explanation of the policy" | Brief policy quote | Focus on discussing the implications of low-level economic policies |

---

## 8. Replanned execution route (v2 timeline)

### 15 Month Detailed Gantt Chart

```
时间        阶段                                   关键交付物
──────────────────────────────────────────────────────────────────────────
2026/06    准备阶段
           • FIM-3DGS 核心算法实现（CUDA）
           • AirSim + SUMO 联合仿真平台搭建        ▶ 核心代码完成
           • MatrixCity 数据获取与预处理

2026/07    基础实验
           • 与 FisherRF/GauSS-MI/ActiveGS 集成测试
           • Layer 1/2 指标实验（PSNR、规划延迟）  ▶ 算法层实验完成

2026/08    案例研究 1：建筑物巡检
           • 在 AirSim 中搭建 30 层建筑场景
           • 100 次巡检任务实验
           • 与 Lawn-mower / 工业方案对比         ▶ 巡检案例完成

2026/09    案例研究 2：最后一公里配送
           • 自建 UAV-Delivery-Dataset
           • 5 城市场景 × 100 任务 = 500 次配送实验
           • 与预设航线 / A* 对比                  ▶ 配送案例完成

2026/10    案例研究 3：应急响应
           • 高楼火灾场景仿真
           • 60s 时间约束下的覆盖率实验            ▶ 应急案例完成

2026/11    多 UAV 系统级实验
           • SUMO + AirSim 联合仿真
           • 10/20/50 UAV 同时运行实验
           • 空域利用率 / 系统吞吐量评估           ▶ 系统级实验完成

2026/12    数据分析与初稿
           • 整合所有实验数据
           • 撰写 T-ITS 格式 22 页稿件
           • 内部 reviewer (导师 + 同门) 审阅      ▶ 初稿完成

2027/01    打磨阶段
           • 根据内部反馈大修
           • 英文润色（专业 editing service）
           • 准备补充材料（代码、数据集、视频）    ▶ 投稿准备就绪

2027/02    提交前检查
           • Cover letter 撰写
           • 期刊格式调整
           • 推荐审稿人列表准备

2027/03    ◉ 投稿 IEEE T-ITS              ──────────────────────────────

2027/03–08  Round 1 审稿（4-6 月）

2027/09    收到审稿意见
           • 若小修：1-2 月修改                    ▶ 接受目标 2027/12
           • 若大修：3-4 月补实验                  ▶ 接受目标 2028/03
           • 若拒稿：转 TR Part C，调整 framing    ▶ TR-C 投稿 2027/12

2028/06    最终发表（无论哪个期刊）              ◉ 最终目标
──────────────────────────────────────────────────────────────────────────
```

---

## 9. Risk Assessment and Alternatives

### 9.1 Main risks

**Risk A: T-ITS rejection (probability ~50%, normal rate)**
- Response: Framing adjusted and converted to TR Part C
- Time cost: additional 6 months
- Buffering: Prepare for double framing since v2

**Risk B: Insufficient experimental time**
- Response: Core experiments (Layer 1-2 + Case 1-2) are guaranteed, emergency response cases can be postponed
- Key: Perception layer + delivery case must be complete

**Risk C: Insufficient performance gap between algorithm and SOTA**
- Response: GauSS-MI is new work in 2025, this paper should have 1+ year advantage
- Buffering: ablation experiments show theoretical advantages, an absolute number of +1.5 dB is sufficient

**Risk D: Extended review period**
- Response: Select fast-track before submission (if provided by the journal)
- Alternative: Prepare the conference version and submit it to ICRA 2027 at the same time (no repeated publication, only as a backup plan)

### 9.2 Alternative submission paths (by priority)| Priority | Journal | IF | Suitability | Remarks |
|--------|------|----|---------| -----|
| **Preferred** | IEEE T-ITS | 8.5 | ★★★★★ | Main Investment Target |
| Alternative 1 | TR Part C | 8.5 | ★★★★☆ | Rejected and then transferred |
| Alternative 2 | IEEE T-RO | 7.4 | ★★★★☆ | If ITS does not accept it, pure robot content |
| Alternative 3 | TR Part B | 6.0 | ★★★☆☆ | Partially methodological, needs more theory |
| Alternative 4 | Transportation Science | 5.4 | ★★★☆☆ | Partially mathematical, needs extension of queuing theory |

---

## 10. Summary: core changes from v1 → v2

**1 sentence summary:** Paper C should no longer be submitted to conferences as a "perception algorithm paper", but as a "low-altitude economic enabling technology research" to be submitted to top journals.

**3 Key Differences:**
1. **Problem level:** Single reconstruction task → Urban transportation system
2. **Assessment scope:** Perception indicators → Five-layer indicator system (perception/planning/task/system/economy)
3. **Academic dialogue:** Dialogue with perception paper → Dialogue with ITS / UAV-transportation top journal paper

**5 new significant workloads:**
1. SUMO + AirSim joint simulation platform
2. Three major case studies (inspection, distribution, emergency response)
3. Self-built UAV-Delivery-Dataset data set
4. Multi-UAV system-level experiments (10-50 units operating simultaneously)
5. T-ITS / TR-C double framing manuscript

**Time cost:** v1 is planned to take 4 months, v2 is planned to take 12-15 months (reasonably reflects the workload of submitting manuscripts to top journals)

---

> **Document iteration description:** This is the v2 version of the Paper C plan (`v2_20260515`). v1 (`v1_20260515`) is retained as a historical archive to record the design of the "Fast RA-L Path" for subsequent comparison. Trigger conditions for the next update: ① Complete the experimental data of 2026/08 ② Receive T-ITS review comments, and then update to v3.