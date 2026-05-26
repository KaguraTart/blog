---
title: "Paper B Planning v1: Three-layer hierarchical scheduling of hundreds of UAVs for TR-C"
description: "Investigate whether Paper B is more suitable for TR Part C, and plan the background, related methods, problem definition, algorithm route, experimental data, expected conclusions, innovation points and promotion plan."
pubDate: 2026-05-19
updatedDate: 2026-05-23
tags: ["Paper B", "TR-C", "T-ITS", "UAV", "UAM", "hierarchical scheduling", "queuing theory", "Lyapunov", "multimodal transport"]
category: Tech
sourceHash: "ed9e6e52930a3853c8677d15a6be553b48ecb4cc"
---

# Paper B Planning v1: Three-layer hierarchical scheduling of hundreds of UAVs for TR-C

> Conclusion: **Paper B is more suitable for main investment Transportation Research Part C: Emerging Technologies, IEEE T-ITS as an alternative or change of investment direction. **
> The core reason is not that TR-C is "better", but that the essence of Paper B's problem is transportation system operation: under the constraints of dynamic demand, limited vertiport/charging/corridor capacity, and multi-modal transportation, let a hundred-level UAV fleet serve urban logistics/emergency tasks stably, efficiently, and safely.

---

## 1. Background and submission judgment

The concerns of Paper B can be summarized as:

> In an urban low-altitude economic scenario, how to schedule a 100-level UAV fleet to maintain long-term task queue stability and minimize delays, energy consumption, airspace congestion, and operating costs under the conditions of dynamic orders, limited take-off and landing points, limited charging resources, low-altitude corridor capacity constraints, and ground transportation coordination?

This is not a single-machine path planning, nor is it simply multi-agent collision avoidance. The real research object is **transportation service system**:

- Demand side: Orders arrive randomly, and there are differences in deadlines, priorities, starting and ending points, and cargo/emergency types.
- Supply side: UAV power, load, current location, maintenance status and available airspace change over time.
- Infrastructure side: limited capacity for vertiports, charging pads, low-altitude corridors, handover points and ground vehicles.
- System side: It is necessary to optimize throughput, delay, queue backlog, resource utilization, energy and safety at the same time.Therefore, it is more reasonable to invest mainly in TR-C. The official scope of TR-C clearly emphasizes the impact of emerging technologies on transportation system planning, design, operation, control and maintenance, and explains that the intellectual core of the journal is on the transportation side, rather than the individual technology itself; it also welcomes the integration approach of operations research, control systems, complex networks, computer science and AI, and pays special attention to multimodal / intermodal transportation, on-demand transport, ITS, logistics, aviation, resource management and open data sets [1]. These keywords almost exactly cover Paper B.

T-ITS is also available as an alternative. T-ITS scope covers sensing, communications, controls, planning, design, implementation, AI, formal methods, multi-agent systems and multimodal transportation in modern transportation systems [2]. But T-ITS is more likely to require the flavor of “intelligent transportation system technology implementation”, such as communication, sensing, control, deployment architecture or intelligent system closed loop. If Paper B ultimately emphasizes Lyapunov-regulated online scheduling, GNN/MARL control and real-time system implementation, you can turn to T-ITS; if it emphasizes transport capacity, queue stability, infrastructure bottlenecks and the value of multi-modal logistics systems, you should vote for TR-C.

**Current recommendations: TR-C first, T-ITS backup. **

### 1.1 2026-05-22 Writing Calibration: Paper B must be a transportation system operation paper

Paper B is the most suitable for absorbing the logic of "traffic journals are not just about algorithms". It cannot be written as "We propose a new UAV scheduling algorithm", but should be written as:> For urban low-altitude logistics/emergency services, how can a hundred-level UAV fleet maintain fleet stability, reduce delays, control safety risks, and identify system bottlenecks under dynamic demand, limited vertiport/charging/corridor capacity, and multi-modal transportation constraints?

This main line determines how the full text is written:

| module | cannot just be written as | TR-C version should be written as |
|------|------------|------------------|
| Background | UAV scheduling is difficult | Operational control issues for low-altitude transportation services under peak demand, infrastructure capacity, and safety isolation constraints |
| gap | Existing algorithms are not good enough | Existing research deals with single-point problems of routing/network design/resource allocation and lacks closed-loop and stability guarantees for hundreds of rack-level online systems |
| Method | New hierarchical scheduling algorithm | Unified operation control framework of macro demand queue, meso airspace/takeoff and landing/charging resources, and micro energy/safety constraints |
| Experiment | Reward or higher success rate | Systematic improvement of delay, throughput, queue backlog, deadline violation, resource utilization, energy, conflict risk |
| Conclusion | The method is better than the baseline | Under what demand intensity the system is stable, which resource becomes the bottleneck first, when multi-modal fallback is necessary, and whether strategic current limiting is needed |

Therefore, Paper B's experiment answers the system question rather than just proving that the model score is higher:- **Capacity Boundary**: Under low / medium / peak / shock demand, when does the system enter the unstable zone?
- **Bottleneck Attribution**: Does the delay mainly come from vertiport, charging, corridor congestion, or fleet repositioning?
- **Multi-modal value**: When is UAV-only not enough? How does ground fallback reduce deadline violations?
- **Theoretical correspondence**: Can the backlog/cost tradeoff of Lyapunov drift-plus-penalty be observed in experiments?
- **Management Inspiration**: If you only add one resource, should you add UAV, charging pad, vertiport slot, or corridor capacity?

### 1.2 2026-05-23 Organizing: Minimum submission version and boundaries

The minimum submittable version of Paper B should be a **TR-C transportation system operation paper**, not a mixture of "scheduler + MARL + airspace simulation + low-altitude platform demonstration". The first edition must break through the system problem: how limited UAV, vertiport, charging pad and corridor capacity jointly determine delay, throughput, queue stability and service reliability under dynamic demand.| Must be completed | Postponed to extended version |
|----------|--------------|
| synthetic city UAM queuing benchmark | AirSim/UE-level high-fidelity visual simulation |
| Lyapunov-regulated online scheduler | Real flight deployment or hardware closed loop |
| 20 / 50 / 100 / 200 UAV scalability | LLM dispatcher as main algorithm |
| vertiport / charging / corridor bottleneck analysis | Complete communication protocol and link layer simulation |
| FCFS, greedy, rolling MILP, ALNS, backpressure, MARL/GNN baselines | Multi-city policy evaluation and full economic analysis |
| stability, cost-delay tradeoff, deadline violation, runtime | real business order system access |

The first version of the experimental package recommends freezing to five deliverables:1. **Benchmark generator**: Generate urban areas, vertiport, charging pad, corridor, demand flow, deadline, ground fallback and shock demand.
2. **System model**: Output reproducible experimental logs of demand queue, vertiport queue, charging queue, corridor virtual queue, and deadline virtual queue.
3. **H-LyraUAV core**: implements drift-plus-penalty decision-making. The learning module only provides demand/service time/risk estimates and is not used as a source of stability.
4. **Baseline suite**: Each baseline uses the same set of demand traces, capacity settings, UAV fleet and random seeds.
5. **TR-C result package**: The main table reports delay, 95th delay, deadline violation, throughput, backlog, resource utilization, energy, conflict proxy, runtime; the appendix table reports bottleneck attribution and sensitivity.

This boundary can make Paper B's system story consistent with its experimental responsibilities: first prove that "hundred-shelf-level low-altitude logistics/emergency service systems can operate stably online" before talking about real maps, real orders, or more complex agents.

---

## 2. Current method genealogy

Paper B needs to put relevant methods into the pedigree of transportation engineering, rather than just talking along the lines of UAV/MARL.| Method line | Representative method | Inspiration for Paper B | Limitations |
|--------|----------|-------------------|------|
| Traditional OR | MILP, time-space network, network design, ALNS, rolling horizon | Suitable for expressing capacity, time window, path, synchronization and infrastructure constraints | Large-scale online scheduling is difficult to solve in real time |
| UAM / UTM | vertiport scheduling, capacity-constrained scheduling, conflict detection and resolution | Provides capacity, airspace conflict and corridor management perspectives | Most are single-layer, single-mode or small and medium-sized |
| Multi-modal logistics | truck-drone, UAV-UGV, ground-air transfer, ridesharing-UAM | Prove that UAV must be embedded in the urban transportation system instead of flying in isolation | Mostly offline routing / network design, lack of online queue stability |
| Learning scheduling | MARL, GNN, safe learning, demand prediction | Scalable to hundreds of racks, suitable for dynamic needs and high-dimensional states | Lack of explainable stability guarantee, reviewers will question security |
| Queuing theory and Lyapunov | open queuing network, backpressure, drift-plus-penalty | Can prove backlog stability and cost-delay tradeoff | Need to be combined with actual UAV energy, capacity, path constraints |Existing TR-C papers have covered many "single point capabilities": low-altitude UAV parcel traffic management and resource allocation [3], UAM passenger-centric fairness and operational efficiency [4], relay charging station network design [5], truck-drone reliable synchronization routing [6], UAV-UGV multi-trip delivery network design [7], and UAM ridesharing dynamic scheduling [8]. The opportunity for Paper B lies in converging these capabilities into a **hundred-shelf-level online hierarchical scheduling system**.

---

## 3. Current papers and citable literature

### 3.1 Venue and framing literature| Number | Literature/source | Core information | Positioning role for Paper B |
|------|-----------|----------|------------------------|
| [1] | TR-C official aims & scope | transportation-side intellectual core; focus on operation, control, multimodal, logistics, aviation, open datasets | Support the main investment TR-C |
| [2] | IEEE T-ITS official scope | ITS sensing, communications, controls, planning, AI, multi-agent systems | Support T-ITS alternatives |
| [15] | Machine Learning for UAV-Aided ITS, T-ITS 2024 | UAV can serve ITS's traffic monitoring, emergency response, infrastructure inspection | Support T-ITS alternative framing |
| [18] | 4D trajectory planning for UAV teams, T-ITS 2024 | UAV teams have been published in ITS/T-ITS | Explain that T-ITS can be invested, but it needs to be more intelligent system |

### 3.2 TR-C/UAM/UAV Scheduling Paper| Number | Literature | Methods | Inspiration for Paper B |
|------|------|------|------------------|
| [3] | Li, Hansen & Zou, TR-C 2022 | Traffic management, path conflict, resource allocation, VCG mechanism of low-altitude UAV parcel delivery | Directly stating that low-altitude airspace resource allocation is a legal topic of TR-C |
| [4] | Bennaceur, Delmas & Hamadi, TR-C 2022 | passenger-centric UAM, fairness and operational efficiency | Supporting service quality, fairness, passenger/cargo metrics |
| [5] | Pinto & Lagorio, TR-C 2022 | drone network design with intermediate charging stations | Support charging infrastructure into formulation |
| [6] | Xing, Guo & Tong, TR-C 2024 | truck-drone reliable routing with dynamic synchronization | Support multi-modal synchronization and uncertain travel time |
| [7] | Zhou, Zeng & Yang, TR-C 2025 | UAV-UGV multi-trip delivery network design with release times | Support UAV + UGV multi-trip delivery network |
| [8] | Li, Zhang, Xiao & Li, TR-C 2025 | UAM ridesharing dynamic scheduling and multimodal mobility-on-demand | Support air-ground multimodal service architecture |
| [9] | Wei, Nilsson & Coogan, arXiv 2021 | capacity-constrained UAM scheduling with uncertain travel time and limited landing capacity | support capacity-constrained scheduling formulation |
| [10] | Murthy et al., EPTCS/arXiv 2022 | safe learning for UAM scheduling with hard/soft deadlines | Support safe online scheduling baseline |
| [11] | NASA vertiport FCFS scheduling, 2020 | vertiport capacity and throughput under FCFS | as FCFS and queuing capacity baseline |
| [16] | Liu, Liu & Huang, arXiv 2024 | real-time UAV delivery scheduling management middleware | Support real execution system and UAV/AGV/ground staff collaboration |### 3.3 Queuing theory, Lyapunov and the basis of system stability

| Number | Literature | Core contribution | Contribution to Paper B |
|------|------|----------|-------------------|
| [12] | Grippa et al., Autonomous Robots 2019 | drone delivery job assignment and dimensioning; use queuing theory to analyze stability and workload policy | Support UAV delivery queuing model |
| [13] | Neely, 2010 | stochastic network optimization and Lyapunov drift-plus-penalty | Support $O(1/V)$ cost and $O(V)$ backlog tradeoff |
| [14] | Tassiulas & Ephremides, IEEE TAC 1992 | constrained queuing systems and throughput-optimal scheduling | supporting backpressure / stability theoretical tradition |
| [17] | Vertiport placement with vehicle sizing and queuing, 2023 | open-network queuing for vertiport infrastructure and service rates | Support vertiport queue / charging queue modeling |

**Literature Judgment:** The existing literature has fully proven that “UAV/UAM + transportation systems + scheduling + multimodal + queuing” is a legitimate topic for TR-C/T-ITS. Paper B can no longer be written as “MARL Scheduling Algorithm for Hundreds of UAVs”, but must be written as “Low-altitude Traffic System Operation Control and Stability Guarantee”.

---

## 4. Current problemThere are four main gaps left by existing work.

1. **Lack of a hundred-shelf-level online three-layer scheduling closed loop. **
   TR-C already has low-altitude airspace resource allocation, truck-drone routing, UAM ridesharing and UAV-UGV network design [3,6,7,8], but most of these works deal with a certain layer of routing, network design, resource allocation or ridesharing, lacking a unified online framework from macro demand queue to meso corridor/vertiport resource to micro UAV energy/safety.

2. **Lack of queue stability/service guarantee. **
   Learned scheduling and heuristics can improve empirical performance, but TR-C reviewers question system operability if they cannot account for whether queues are stable under peak demand. Neely's Lyapunov optimization and Tassiulas-Ephremides' constrained queuing scheduling provide theoretical foundations [13,14], but have not been systematically used for multi-modal scheduling of hundreds of low-altitude UAVs.

3. **Lack of UAV fleet control from a multi-modal transport perspective. **
   Truck-drone, UAV-UGV, and ridesharing-UAM papers have proven that ground-air integration is the mainstream direction [6, 7, 8], but most of the existing research is offline route/network design. Paper B should treat ground mode as an online fallback and capacity buffer: when the low-altitude corridor or charging queue is congested, the task can be transferred to the UGV/truck/ground courier.4. **Lack of experimental benchmark. **
   TR-C scope places special emphasis on open science and large-scale datasets [1]. If Paper B only performs internal simulation and does not release synthetic benchmark schema, OD demand, corridor capacity and reproducible seeds, it will weaken the persuasiveness.

---

## 5. Our approach: H-LyraUAV

The method name is tentatively decided:

**H-LyraUAV: Hierarchical Lyapunov-Regulated UAV Scheduling for Multimodal Urban Logistics**

Where H stands for hierarchical and Lyra stands for Lyapunov-regulated routing and assignment.

### 5.1 Three-tier architecture

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

| Hierarchy | Time scale | Decision | Core state | Output |
|------|----------|------|----------|------|
| Macro level | 1-5 min | Task partitioning, UAV repositioning, mode split | Regional demand queue, power distribution, OD demand forecast | Regional dispatch target |
| Mesolayer | 5-30 s | vertiport slot, corridor route, charging slot | take-off and landing queue, corridor congestion, charging queue | executable schedule |
| Microscopic layer | 0.1-5 s | Speed, altitude, local avoidance, emergency return | Neighboring UAVs, obstacles, remaining power | Safe trajectory correction |

### 5.2 Core Mechanism

The key to H-LyraUAV is not "end-to-end scheduling with a large model", but to limit the learning module to prediction and stratification, and build stability on Lyapunov queue control:- **Queueing model**: Demand, vertiport, charging and corridor are represented by real or virtual queues.
- **Lyapunov drift-plus-penalty**: Select assignment / mode / route / charging in each time window to minimize the weighted sum of queue drift and operating costs.
- **Learning-assisted prediction**: GNN/temporal model predicts future OD demand, service time, corridor risk and ground travel time, but is not used as a source of stability proof.
- **Multimodal fallback**: When UAV-only causes queue explosion or deadline risk to increase, the system automatically enables UGV/truck/ground courier or mixed mode.

---

## 6. Problem Formulation

### 6.1 Collections and States

Let the UAV set be $\mathcal{U}$, the dynamic mission set be $\mathcal{R}(t)$, the vertiport set be $\mathcal{V}$, the low-altitude corridor set be $\mathcal{E}$, and the ground transportation mode set be $\mathcal{G}$.

The state of each UAV $u\in\mathcal{U}$ at time $t$ is:

$$
s_u(t)=(l_u(t), b_u(t), a_u(t), \kappa_u(t)),
$$

Where $l_u(t)$ is the position, $b_u(t)$ is the power, $a_u(t)$ is the available status, $\kappa_u(t)$ is the load/task capacity.

Each task $r\in\mathcal{R}(t)$ contains:

$$
r=(o_r,d_r,\omega_r,\delta_r,\pi_r,\eta_r),
$$

Among them, $o_r,d_r$ is the starting and ending point, $\omega_r$ is the cargo/passenger/emergency type, $\delta_r$ is the deadline, $\pi_r$ is the priority, and $\eta_r$ is the set of acceptable transportation modes.### 6.2 Queue definition

The system maintains the following real and virtual queues:

| Queue | Meaning |
|------|------|
| $Q_i(t)$ | Unserved demand queue for area $i$ |
| $B_v(t)$ | Vertiport $v$'s take-off/waiting queue |
| $C_v(t)$ | charging queue of vertiport $v$ |
| $Z_e(t)$ | Congestion/safety interval of corridor $e$ virtual queue |
| $D_i(t)$ | deadline violation virtual queue in area $i$ |

For example, the regional demand queue can be written as:

$$
Q_i(t+1)=\max[Q_i(t)-\mu_i(t),0]+A_i(t),
$$

Where $A_i(t)$ is the newly arrived demand, $\mu_i(t)$ is the number of demands that complete the service within the time window.

### 6.3 Decision variables

Each scheduling cycle needs to decide:

| Decision Making | Symbols | Meaning |
|------|------|------|
| assignment | $x_{u,r}(t)$ | Whether UAV $u$ serves task $r$ |
| mode choice | $m_r(t)$ | UAV-only, ground-only or mixed mode |
| departure time | $s_u(t)$ | departure/departure/transfer time |
| route / corridor | $p_u(t)$ | Select low-altitude corridor or ground path |
| charging decision | $c_u(t)$ | Whether to charge and which vertiport to charge |

### 6.4 Optimization goals

The long-term goal is to minimize average system cost:

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
$$Where $W(t)$ is delay, $E(t)$ is energy consumption, $O(t)$ is operating cost, $S(t)$ is safety/congestion penalty, $M(t)$ is multi-modal transportation penalty.

### 6.5 Constraints

Constraints include:

- queue stability: All real queues and critical virtual queues must be strongly stable.
- battery: $b_u(t)$ is not lower than the safe return threshold.
- Payload: The weight of the mission cargo cannot exceed the capacity of the UAV or ground vehicle.
- time window: High-priority tasks must meet the deadline or enter the deadline virtual queue.
- vertiport capacity: The pad / parking / charging capacity of each vertiport has an upper limit.
- Corridor separation: The time and space intervals of UAVs in the same corridor meet safety requirements.
- multimodal transfer feasibility: The handover time, location and capacity of UAV and UGV/truck/ground courier are feasible.

### 6.6 Theoretical Objectives

Define Lyapunov function:

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

Solve drift-plus-penalty for each time window:

$$
\Delta(\Theta(t)) + V\cdot \mathbb{E}[Cost(t)\mid \Theta(t)].
$$

Expectancy theory conclusion:

- H-LyraUAV can keep the queue stable if the arrival rate is inside the system capacity region.
- Compared with the optimal stationary randomized policy, the long-term average cost reaches an approximate $O(1/V)$.
- The average backlog is $O(V)$, forming an interpretable cost-delay tradeoff [13,14].

---

## 7. Experimental data source### 7.1 Main experiment: program generation benchmark

The main experiment does not rely on real UAV flight data, but builds a reproducible synthetic UAM queuing benchmark:

- City map: `50x50` to `200x200` grid, including building, no-fly zones, corridors, vertiports, charging pads.
- Demand flow: Poisson / non-homogeneous Poisson / bursty demand, supports morning peak, evening peak, shock demand.
-Task types: parcel delivery, medical delivery, inspection, emergency supply.
- UAV fleet: 20 / 50 / 100 / 200 units, heterogeneous batteries, load, speed, charging time.
- Infrastructure: 5 / 10 / 20 vertiports, different pad, parking, charging capacity.
- Multi-modal mode: UAV-only, ground-only, UAV-ground mixed mode.

### 7.2 Real augmented data

To enhance the convincingness of TR-C, experiments can use public traffic data as demand proxy and ground mode travel time:

| Data source | Purpose |
|--------|------|
| OpenStreetMap | Road network, POI, building density, candidate vertiport / transfer point |
| NYC TLC Taxi Trip Data | OD demand proxy, time period demand profile |
| Chicago Taxi Trips / Divvy / public mobility data | Cross-city generalization demand proxy |
| SUMO | Ground vehicle travel time, congestion, ground fallback costs |
| AirSim or lightweight UAV simulator | Supplementary verification of micro-safety, flight time and energy consumption |Conferences such as AAAI can only conduct synthetic benchmarks; but TR-C requires a case study quality, so it is recommended to conduct at least one city case: San Francisco, New York or Chicago. Li et al.'s low-altitude UAV parcel traffic management using the San Francisco case study [3] can be used as an alignment object.

---

## 8. Experimental design and comparison

### 8.1 Baselines

| Baseline | Description | Purpose |
|----------|------|------|
| FCFS vertiport scheduling | Allocate takeoff and landing resources in order of arrival [11] | Traditional operation baseline |
| Greedy nearest UAV | The nearest available UAV to pick up the nearest task | Simple online dispatch |
| MILP rolling horizon | small-scale rolling optimization | small-scale upper bound |
| ALNS / heuristic dispatch | Refer to TR-C multimodal routing literature [7,8] | Strong OR heuristic |
| Queue-only backpressure | Scheduling based only on queue difference | Theoretically stable baseline |
| MARL / GNN dispatch | Learning allocation, no Lyapunov virtual queues | Learning baseline |
| H-LyraUAV full | Three-layer layering + Lyapunov + learning prediction + multimodal fallback | Main method |

### 8.2 Metrics| Indicator | Meaning |
|------|------|
| Average delay | Average task completion delay |
| 95th percentile delay | Long tail service quality |
| Deadline violation rate | Ratio of overtime tasks |
| Throughput | Number of tasks completed per unit time |
| Queue backlog | Demand, vertiport, charging, corridor queue length |
| Queue stability | Is the backlog bounded over time |
| Vertiport utilization | Take-off and landing/stopping resource utilization rate |
| Charging utilization | Charging resource utilization rate |
| Airspace conflict rate | corridor safety interval conflict rate |
| Energy per delivery | Energy consumption per order |
| Ground-UAV transfer success | Multi-modal handover success rate |
| Runtime | Single-step scheduling time |
| Bottleneck contribution | How much delay is contributed by vertiport / charging / corridor / fleet repositioning |
| Capacity margin | How far is the current demand intensity from the system instability zone |
| Service equity | delay gap for different areas/priority tasks to prevent optimization of only popular areas |

It is not recommended that the TR-C version main table only reports algorithm performance rankings, but also provides a separate **system diagnostics table**: reporting average backlog, 95% delay, deadline violation, major bottlenecks, and whether multimodal fallback is triggered under different demand multipliers. In this way, the conclusion can come to "how the system operates" rather than "which model is stronger".

### 8.3 Ablation| Ablation | Purpose |
|------|------|
| no Lyapunov virtual queues | Verify stability component contribution |
| no multimodal fallback | Verify the value of ground mode as a capacity buffer |
| no hierarchical decomposition | Verify the scalability of the three-tier structure |
| no demand prediction | Verify learning prediction module contribution |
| no charging queue modeling | Verify whether the charging bottleneck must be modeled explicitly |
| 20 / 50 / 100 / 200 UAV scaling | Verification of hundreds of rack-level scalability |

### 8.4 Scene design

Run at least four types of demand scenarios:

1. **Low demand**: The system is lightly loaded, verifying that H-LyraUAV does not sacrifice efficiency.
2. **Peak demand**: Demand is close to the capacity region, verify queue stability.
3. **Shock demand**: sudden emergency orders, verify deadline virtual queue and multimodal fallback.
4. **Infrastructure bottleneck**: charging pads or vertiport pads are deliberately reduced to verify resource bottleneck identification capabilities.

It is additionally recommended to add two types of generalizations:

5. **Scale generalization**: Training or parameter adjustment is at 50/100 UAV, and testing is at 200 UAV, indicating that the hierarchical structure is not only effective for fixed scale.
6. **Topology generalization**: The OSM-derived city graph was measured from the rule grid_city, indicating that the conclusion is not an accidental result of a toy map.

---

## 9. Expected success and innovation

### 9.1 Expected success

This section is for pre-registration expectations and does not write actual experimental results.1. **Maintain queue stability under peak demand. **
   It is expected that H-LyraUAV keeps the demand queue, vertiport queue and charging queue bounded in peak demand, while greedy/MARL-only is more prone to backlog accumulation.

2. **Reduce delays and deadline violations. **
   Compared with FCFS and greedy, H-LyraUAV is expected to reduce average delay, 95th percentile delay and deadline violation rate.

3. **Improve real-time performance and scalability. **
   Compared with MILP rolling horizon, H-LyraUAV is expected to maintain second- or sub-second online decision-making in 100/200 UAV scenarios.

4. **Theoretical explanation reserved. **
   Compared with MARL/GNN-only, the advantage of H-LyraUAV is not just the experience score, but it can explain the cost-delay tradeoff and stability boundary.

5. **Show the system value of multi-modal fallback. **
   It is expected that UAV-ground mixed mode reduces missed deadlines and queue backlog in charging bottleneck or corridor congestion scenarios.

### 9.2 Innovation points

1. **Low-altitude traffic system scheduling paper based on TR-C framing. **
   Paper B does not treat UAVs as isolated robots, but as a fleet of hundreds of UAVs as part of the urban transportation service system.

2. **Hundred-shelf-level three-layer queue-stable multimodal scheduling framework. **
   Unify macro task queues, meso infrastructure resources, and micro safety/energy constraints in an online framework.

3. **Lyapunov-regulated learning-assisted dispatch. **
   The learning module is used to predict demand and cost, and the Lyapunov module provides stability and cost-delay tradeoff.4. **Multimodal transport capacity buffering mechanism. **
   Use UGV/truck/ground courier as fallback under airspace/charging bottleneck instead of attached baseline.

5. **Open synthetic UAM queuing benchmark. **
   Aligning TR-C's preferences for open data, reproducible benchmarks, and transferability [1].

---

## 10. References

[1] Elsevier. “Transportation Research Part C: Emerging Technologies: Aims & Scope.” URL: <https://www.sciencedirect.com/journal/transportation-research-part-c-emerging-technologies>

[2] IEEE Intelligent Transportation Systems Society. “IEEE Transactions on Intelligent Transportation Systems (T-ITS): Scope.” URL: <https://ieee-itss.org/pub/t-its/>

[3] Ang Li, Mark Hansen, and Bo Zou. "Traffic management and resource allocation for UAV-based parcel delivery in low-altitude urban space." *Transportation Research Part C: Emerging Technologies*, 143:103808, 2022. DOI: 10.1016/j.trc.2022.103808. URL: <https://www.sciencedirect.com/science/article/pii/S0968090X22002339>[4] Mehdi Bennaceur, Rémi Delmas, and Youssef Hamadi. “Passenger-centric Urban Air Mobility: Fairness trade-offs and operational efficiency.” *Transportation Research Part C: Emerging Technologies*, 136:103519, 2022. DOI: 10.1016/j.trc.2021.103519. URL: <https://www.sciencedirect.com/science/article/pii/S0968090X21005015>

[5] Roberto Pinto and Alexandra Lagorio. “Point-to-point drone-based delivery network design with intermediate charging stations.” *Transportation Research Part C: Emerging Technologies*, 135:103506, 2022. DOI: 10.1016/j.trc.2021.103506. URL: <https://doi.org/10.1016/j.trc.2021.103506>[6] Jiahao Xing, Tong Guo, and Lu (Carol) Tong. "Reliable truck-drone routing with dynamic synchronization: A high-dimensional network programming approach." *Transportation Research Part C: Emerging Technologies*, 165:104698, 2024. DOI: 10.1016/j.trc.2024.104698. URL: <https://www.sciencedirect.com/science/article/pii/S0968090X24002195>

[7] Bolong Zhou, Wenjia Zeng, and Hai Yang. “Multi-trip UAV-UGV delivery network design with release times.” *Transportation Research Part C: Emerging Technologies*, 181:105389, 2025. DOI: 10.1016/j.trc.2025.105389. URL: <https://doi.org/10.1016/j.trc.2025.105389>

[8] Shanghan Li, Tengfei Zhang, Yiyong Xiao, and Daqing Li. “On-demand ridesharing based on dynamic scheduling in urban air mobility.” *Transportation Research Part C: Emerging Technologies*, 175:105111, 2025. DOI: 10.1016/j.trc.2025.105111. URL: <https://www.sciencedirect.com/science/article/pii/S0968090X25001159>[9] Qinshuang Wei, Gustav Nilsson, and Samuel Coogan. “Capacity-Constrained Urban Air Mobility Scheduling.” arXiv:2107.02900, 2021. URL: <https://arxiv.org/abs/2107.02900>

[10] Surya Murthy, Natasha A. Neogi, and Suda Bharadwaj. “Scheduling for Urban Air Mobility using Safe Learning.” *Electronic Proceedings in Theoretical Computer Science*, 371:86-102, 2022; arXiv:2209.15457. DOI: 10.4204/EPTCS.371.7. URL: <https://arxiv.org/abs/2209.15457>

[11] Nelson M. Guerreiro, George E. Hagen, Jeffrey M. Maddalon, and Ricky W. Butler. “Capacity and Throughput of Urban Air Mobility Vertiports with a First-Come, First-Served Vertiport Scheduling Algorithm.” NASA Technical Reports Server, AIAA Aviation 2020 Forum, 2020. URL: <https://ntrs.nasa.gov/citations/20205001421>[12] Pasquale Grippa, Doris A. Behrens, Friederike Wall, and Christian Bettstetter. “Drone delivery systems: job assignment and dimensioning.” *Autonomous Robots*, 43:261-274, 2019. DOI: 10.1007/s10514-018-9768-8. URL: <https://link.springer.com/article/10.1007/s10514-018-9768-8>

[13] Michael J. Neely. *Stochastic Network Optimization with Application to Communication and Queueing Systems.* Synthesis Lectures on Communication Networks, Morgan & Claypool Publishers, 2010. DOI: 10.2200/S00271ED1V01Y201006CNT007. URL: <https://doi.org/10.2200/S00271ED1V01Y201006CNT007>

[14] Leandros Tassiulas and Anthony Ephremides. "Stability Properties of Constrained Queueing Systems and Scheduling Policies for Maximum Throughput in Multihop Radio Networks." *IEEE Transactions on Automatic Control*, 37(12):1936-1948, 1992. DOI: 10.1109/9.182479. URL: <https://doi.org/10.1109/9.182479>[15] Akbar Telikani, Arupa Sarkar, Bo Du, and Jun Shen. "Machine Learning for UAV-Aided ITS: A Review With Comparative Study." *IEEE Transactions on Intelligent Transportation Systems*, 25(11):15388-15406, 2024. DOI: 10.1109/TITS.2024.3422039. URL: <https://ieeexplore.ieee.org/document/10622103/>

[16] Han Liu, Tian Liu, and Kai Huang. "A Real-Time System for Scheduling and Managing UAV Delivery in Urban Areas." arXiv:2412.11590, 2024. URL: <https://arxiv.org/abs/2412.11590>

[17] Jose Escribano Macias, Carl Khalife, Joseph Slim, and Panagiotis Angeloudis. “An integrated vertiport placement model considering vehicle sizing and queuing: A case study in London.” *Journal of Air Transport Management*, 113:102486, 2023. DOI: 10.1016/j.jairtraman.2023.102486. URL: <https://www.sciencedirect.com/science/article/pii/S0969699723001291>[18] Blanca Lopez Palomino, Javier Muñoz Mendi, Fernando Quevedo Vallejo, Concepción Alicia Monje Micharet, Luis Santiago Garrido Bullon, and Luis Enrique Moreno Lorente. “4D Trajectory Planning Based on Fast Marching Square for UAV Teams.” *IEEE Transactions on Intelligent Transportation Systems*, 25(6):5703-5717, 2024. DOI: 10.1109/TITS.2023.3336008. URL: <https://doi.org/10.1109/TITS.2023.3336008>

---

## Appendix: Execution Plan

### Week 1: Freeze paper positioning and problem formulation

- Clarify the main investment TR-C and the alternative T-ITS.
- Freeze title, first draft of abstract, core questions and three-layer architecture diagram.
- Complete the definition of sets, queues, decisions, goals and constraints for problem formulation.

### Week 2-3: Supplement 25+ documents and related work matrix

- Extended TR-C / T-ITS / UAM / UAV delivery / queuing / Lyapunov documentation.
- Output related work matrix: problem, method, scale, mode, limitation of each paper.
- Identify the differences between Paper B and each type of job.

### Weeks 4-6: Implementing the synthetic UAM queuing benchmark- Implement map, vertiport, corridor, charging pad, and OD demand generator.
- Supports 20/50/100/200 UAV and low/medium/peak/shock demand.
- Output manifest, seed, scenario config to ensure reproducibility.

### Weeks 7-9: Implementing baselines

- FCFS vertiport scheduling.
- Greedy nearest UAV.
- MILP rolling horizon small-scale upper bound.
-ALNS/heuristic dispatch.
- Queue-only backpressure.
- MARL/GNN dispatch without Lyapunov.

### Weeks 10-12: Implementing H-LyraUAV and Ablation

- Implement macro queue-aware assignment.
- Implement meso corridor / vertiport / charging scheduling.
- Implement microscopic energy/safety constraint interface.
- Implement no-Lyapunov, no-multimodal, no-hierarchy, no-demand-prediction, no-charging-queue ablations.

### Weeks 13-15: Runner Experiment

-Running 20/50/100/200 UAV scalability.
- Run peak / shock / bottleneck scenarios.
- Output the main table: delay, deadline violation, throughput, queue backlog, resource utilization, runtime.
- Output key graphs: queue trajectory, cost-delay tradeoff, scalability curve, multimodal fallback contribution.

### Week 16: Writing the first draft of TR-C- Introduction starts with transportation operation problem.
- Put three-layer architecture, Lyapunov theorem and algorithm in Method.
- Experiments emphasize system performance, resource utilization, and open benchmark.
- Discussion writes about low-altitude economics, vertiport capacity, charging infrastructure and multimodal logistics implications.

### T-ITS investment change strategy

If the TR-C framing is not strong enough, or the experimental results show that the algorithm/control contribution is stronger than the system transportation insight, the T-ITS version is retained:

- Abstract puts more emphasis on intelligent transportation system, online control, and AI-assisted scheduling.
- Introduction Add sensing / communication / real-time implementation.
- Experiments increase runtime, communication delay, distributed execution and controller robustness.
- Discussion reduces policy/operational implications and increases intelligent systems deployment and ITS integration.