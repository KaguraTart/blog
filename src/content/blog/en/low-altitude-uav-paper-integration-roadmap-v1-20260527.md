---
title: "Low-altitude planning paper matrix v2: three papers in progress, follow-up embodied low-altitude and large model routes"
description: "With three ongoing papers on conflict-free path planning, three-layer scheduling of hundreds of UAVs, and information theory-driven 3DGS active sensing planning as the core, we will re-plan the follow-up paper route on embodied low-altitude, low-altitude cloud brain, vertebral large model fine-tuning, inference acceleration, and software and hardware collaboration."
pubDate: 2026-05-27
updatedDate: 2026-05-28
tags: ["low altitude planning", "UAV", "Thesis planning", "Zotero", "T-ITS", "TR-C", "T-RO", "AAAI", "ICRA", "3DGS", "MARL", "Embodied AI", "VLA", "LLM", "Inference acceleration"]
category: Tech
sourceHash: "46302954b2010a293b0edfe46fe54f398cd68bf3"
---

# Low-altitude planning paper matrix v2: three papers in progress, follow-up embodied low-altitude and large model routes

> This article reintegrates the low-altitude UAV papers that have been written so far into a **paper portfolio**.  
> The goal is not to spread out and write a lot of ideas, but to clarify: which articles have already taken shape, which ones can continue to be made into top journal/regular conference papers, and what literature support, experimental assets and submission positioning are needed for each paper.

---

## 0. 2026-05-28 Correction Conclusion

The current focus needs to be changed: instead of "planning 7-10 papers at the same time", it is first to acknowledge that **three papers are already being worked on**, and subsequent papers must naturally grow from the assets of these three papers.

The three articles I’m currently working on are:

| Status | Article | Role | The main line that cannot be deviated from |
|---|---|---|---|
| Already working on | Paper A: Conflict-free path planning / PPO-MAPPO / Low-altitude conflict resolution | Tactical security layer | High-density low-altitude corridors, non-cooperative UAVs, communication/positioning degradation, safety-efficiency trade-offs |
| Already in progress | Paper B: Three-layer hierarchical scheduling of 100 UAVs | System operation layer | 100-level fleet, queue stability, vertiport/charging/corridor bottleneck, multi-modal scheduling |
| Already working on | Paper C: Information theory-driven UAV 3DGS active sensing | Environmental cognitive layer | 3DGS / Fisher information / NBV / safe reconstruction / planning-aware mapping |

Don’t start a new paper for subsequent papers that have nothing to do with the direction. The correct route is:1. **First make A/B/C into three main papers that can be submitted. **
2. **Follow-up Paper D/F/G/H/I are only used as extensions of A/B/C**: scene coverage supports A/C, low-altitude cloud brain connects A/B/C in series, embodied low-altitude connects C’s perception and A’s control into a closed loop, and model fine-tuning and inference accelerate the implementation of the service cloud brain.
3. **General AGI directions cannot be written as empty claims**. A more stable expression is "towards general embodied low-altitude intelligence": starting from domain Agent, tool invocation, simulation feedback, VLA/VLN, world model, and device-side reasoning, and gradually approaching general embodied intelligence.
4. **It is not recommended to train vertical foundation model from scratch in the first stage**. First use an ordinary large model + Agent + Skills/MCP + RAG + verifier + simulator post-processing to form a reproducible experimental closed loop; wait until there are enough tool call trajectories, failure samples, and simulation feedback before doing LoRA/SFT/DPO/GRPO fine-tuning.

This means that subsequent papers should be divided into two levels:

| Level | Thesis goal | Whether to start in the near future |
|---|---|---|
| A/B/C Main Layer | Already in progress, the experimental closed loop must be completed first | Immediately |
| D scene overlay | Provides benchmark, failure taxonomy and safety-critical data to A/C | Recent |
| G Cloud Brain Agent Layer | Turn A/B/C/D/E into tools to build a verifiable low-altitude traffic cloud brain | Mid-term |
| H Embodied low-altitude layer | Make UAV VLN/VLA/world model and connect to universal embodied AI | Mid-term and later |
| I model training layer | Training LowAltitudeGPT / tool-use / LowAltitudeIR / simulation feedback | Wait for the data to stabilize |
| J inference acceleration layer | vLLM/TensorRT-LLM/quantization/device-cloud collaboration/hardware deployment | Wait until the agent workload is stable |

---

## 1. There are currently articles and main line positioning

There are currently three core articles that have formed the basis of content:| Number | Existing content | Current positioning | Recommended main investment | Core judgment |
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

## 3. Paper matrix: 3 papers already in progress + follow-up extension route

The reading of this section needs to be changed: Paper A/B/C are the three main works that are already being done, not "new directions in the future". Paper D/E/G/H/I/J are writable extensions, but the startup sequence must obey the experimental asset maturity of A/B/C.

### 3.1 Paper A: Robust conflict-free planning of low-altitude air route network**Suggested topic:** Robust Conflict-Free UAV Corridor Planning under Non-Cooperative Traffic and Communication Degradation

**Corresponding to existing articles:** Conflict-free path planning, PPO/MAPPO, UAV conflict resolution, UAV conflict env construction.

**Core question:** In an urban low-altitude airway network, how can multiple UAVs maintain separation safety while controlling delays, extra distances, and throughput losses under the conditions of local observation, communication delays, positioning errors, and non-cooperative aircraft insertion.

**Method Route:**

- strategic layer: initial path and time slot allocation based on route network;
- tactical layer: MAPPO/PPO output speed, height or lateral offset action;
- safety shield: CBF-QP / ORCA / RMADER-style trajectory check;
- fallback layer: switch to conservative priority rule when communication degrades;
- Evaluation: 30/50 aircraft are trained and 100/200 aircraft are tested, covering four scenarios: cooperative, non-cooperative, communication-loss, and high-density corridor.

**Key References:**The multi-agent stable training of MAPPO/PPO can be supported by Yu et al. [3]; MAT and FACMAC provide stronger MARL baseline [4,5]; HAPPO/HATRPO provides trust-region multi-agent policy optimization reference [6]. On the robot side, EGO-Swarm, MADER, RMADER, RACER, PANTHER and GCOPTER respectively support decentralized swarm planning, trajectory sharing under delay, collaborative exploration, perception-aware planning and multicopter trajectory optimization [7-12].

**Innovation suggestions:**

1. Upgrade “PPO conflict-free path planning” from a simple RL task to low-altitude traffic corridor safety control.
2. Introduce communication degradation and non-cooperative UAV to form the actual operating boundary that T-ITS is more concerned about.
3. Use learning policy + formal/safety shield to avoid the lack of security of pure RL.
4. Trafficization of indicators: LoWC, NMAC, conflict count, average delay, extra distance, throughput, runtime.

### 3.2 Paper B: Three-layer hierarchical scheduling of hundreds of UAVs

**Suggested topic:** H-LyraUAV: Queue-Stable Hierarchical Scheduling for Hundred-Scale Low-Altitude UAV Logistics

**Corresponding to existing articles:** Paper B three-tier scheduling planning.

**Core question:** How can a hundred-level UAV fleet operate stably, efficiently, and safely under dynamic requirements, limited vertiport/charging/corridor capacity, and multi-modal transportation constraints.

**Method Route:**- macro layer: demand queue, fleet repositioning, mode choice;
- meso layer: vertiport, charging pad, corridor slot scheduling;
- micro layer: energy/safety/conflict-aware trajectory feasibility;
- Theory: Lyapunov drift-plus-penalty guarantees queue stability and cost-backlog tradeoff;
- data: synthetic city grid + OSM/POI/NYC TLC/Chicago taxi/SUMO enhancement.

**Key References:**

TR-C low-altitude UAV delivery traffic management has directly discussed resource allocation and conflict resolution in low-altitude urban space [13]; passenger-centric UAM, fairness and operational efficiency research support service quality framing [14]; charging-station delivery network, capacity-constrained UAM scheduling, safe learning scheduling support infrastructure capacity and safe online scheduling [15-17]; truck-drone / UAV-UGV multi-modal delivery support multimodal extension [18,19].

**Innovation suggestions:**1. Hundred-shelf-level online three-layer scheduling closed loop instead of offline routing/network design.
2. Queue stability becomes the main line of theory, and the learning module only makes predictions or value estimates.
3. Evaluate delay, throughput, backlog, charging utilization, vertiport bottleneck, and corridor congestion at the same time.
4. The conclusion of the traffic system can answer: when does it need to limit traffic, where is the bottleneck, and when is UAV-only inferior to multimodal fallback.

### 3.3 Paper C: FIM-3DGS UAV active sensing planning

**Suggested topic:** FIM-3DGS: Fisher-Information-Driven Active Perception Planning for Safe UAV Reconstruction

**Corresponds to existing articles:** Paper C, Next-Best-View and NeRF/3DGS, Information Theory Active Sensing.

**Core question:** Under limited flight time, energy and safety constraints, how can UAV actively select viewpoints to make the 3DGS map converge faster and serve low-altitude planning tasks.

**Method Route:**

- scene representation: incremental 3D Gaussian Splatting;
- information metric: build Fisher Information / expected information gain for Gaussian parameters or rendered Jacobian;
- planner: NBV candidate generation + safe corridor / CBF constraint;
- task coupling: The reconstruction quality is not only reported on PSNR/SSIM, but also on obstacle recall, planning collision rate, and inspection coverage;
- baselines: ActiveNeRF, FisherRF, GS-Planner, HGS-Planner, POp-GS, frontier exploration.

**Key References:**The original 3DGS text provides real-time explicit radiance field representation [20]; ActiveNeRF is an early representative of neural rendering active perception [21]; FisherRF directly supports Fisher information active view selection, and has 3DGS backend 70 fps results [22]; GS-Planner, HGS-Planner, POp-GS and NVF support the 3DGS/NBV competition line of 2024-2025 [23-26].

**Innovation suggestions:**

1. Upgrade from “3DGS NBV” to “active perception serving UAV security planning”.
2. Use Fisher information to connect CRB / reconstruction uncertainty / planning safety.
3. Expand from visual indicators to traffic/robot task indicators: path feasibility rate, obstacle recall rate, emergency inspection coverage rate.
4. Do cross-scenario generalization on MatrixCity/AirSim/self-built urban low-altitude cells.

### 3.4 Paper D: Low-altitude safety critical scene coverage and accelerated testing

**Suggested topic:** Coverage-Guided Accelerated Testing for Safety-Critical Low-Altitude UAV Navigation

**Corresponding to existing articles:** Paper F scene coverage, dangerous scene generation, 76 million exploration logs.

**Core question:** How to define the test scene space of low-altitude UAV obstacle avoidance/planning algorithm, how to measure coverage, and how to efficiently discover dangerous but effective failure scenarios.

**Method Route:**- scenario grammar: local 50m x 50m x 50m cell, obstacle combination, dynamic obstacles, wind disturbance, target point, start and end points;
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
4. Let the results answer: Which combinations of obstacles are the most dangerous, which planners generalize the worst, and whether increased coverage really reduces unknown risks.

### 3.5 Paper E: Verification of error-correcting UAV language planning

**Suggested topic:** VERA-UAV: Verification-and-Repair Language Planning for Low-Altitude UAV Tasks

**Corresponding to existing articles:** Paper E.

**Core problem:** LLM can convert natural language tasks into UAV executable task specifications, but it is prone to producing plans that are unexecutable, semantic mismatch, or violate safety constraints. Requires typed IR, LTL/STL, validators, and counterexample feedback loops.**Method Route:**

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

**Corresponding to existing articles:** Paper G/G1.

**Core question:** The low-altitude traffic cloud brain cannot be just a chat model, but a verifiable agent that can call the scheduler, path planner, verifier, simulator and risk assessor.

**Method Route:**- LLM is responsible for task understanding, tool selection, status summary and interpretation;
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

**This is a new article that can be written in a new direction. **

**Core question:** How does the overall urban scene map to local low-altitude route planning? How to determine the risk, capacity and service strategy of low-altitude air routes based on different functional areas, building density, road structure, crowd activities, no-fly zones and emergency facility distribution?**Method Route:**

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

### 3.8 Paper I: Embodied low-altitude intelligence and Aerial VLA/VLN

**Suggested topic:** Embodied Low-Altitude Intelligence: Vision-Language-Action Planning for UAVs in Urban Airspace

**This is the mid- to long-term direction that is most worth retaining after the new survey. **The current main line of embodied intelligence has moved from “LLM speaking” to “VLM/VLA directly connecting perception, language and action”. RT-2 clearly proposed the vision-language-action model, putting vision, language and robot action into the same model paradigm [44]; OpenVLA and Octo showed that the open source VLA / generalist robot policy can be pre-trained with large-scale robot trajectories and then fine-tuned with a small amount of target domain data [42,43]. Directly related work has also begun to appear in the UAV direction: SINGER uses Gaussian Splatting to generate language embedding flight simulation data, train onboard drone VLN policy, and do hardware experiments [39]; FlightGPT uses SFT + GRPO to do UAV VLN, and verifies generalization and interpretable reasoning on CityNav [40]; UAV-VLN connects natural language, visual perception and feasible trajectory planning [41].

**Our writable gap:**

Most of the existing aerial VLN/VLA focus on “giving a language target and letting the drone fly near the target”. This is not the capability required by the low-altitude traffic cloud brain. Low-altitude scenarios require the model to understand at the same time:

- Urban low-altitude ODD, airspace structure, no-fly zones and risk areas;
- Multi-UAV traffic status, corridor capacity, and collision avoidance rules;
- Priority of emergency tasks, inspection tasks, and logistics tasks;
- Incomplete visual maps, positioning errors, communication degradation and non-cooperative targets;
- The output must be verifiable, controllable, and degradable, rather than an end-to-end black box operation.

**Suggested method:**

```text
multimodal observation
  = UAV RGB/depth/semantic map/3DGS local map
  + low-altitude traffic state
  + natural-language mission
  + city ODD metadata

LLM/VLM/VLA policy
  -> LowAltitudeIR
  -> skill selection
  -> waypoint / velocity / route command
  -> verifier + safety shield
  -> simulator or hardware feedback
```

**Recommended version to do first:**

Don't initially train end-to-end AerialVLA. First make a **hybrid embodied agent**:

- High-level managers use the Qwen/DeepSeek/API model for task understanding and tool invocation;
- The middle layer calls the conflict resolver of Paper A, the scheduler of Paper B, the active mapper of Paper C, and the verifier of Paper E;
- The lower layer uses traditional controller / MPC / CBF shield to ensure real-time security;
- Training data comes from simulation trajectories, expert planners, failure repair logs and manual annotation tasks.

**Available targets:**- ICRA/IROS/T-RO: Emphasis on embodied navigation, hardware closed loop, and sim-to-real.
- AAAI/IJCAI: Emphasize agent planning, tool-use, verification feedback.
- T-ITS: Emphasis on low-altitude traffic operations, emergency response, conflict resolution and system indicators.

### 3.9 Paper J: LowAltitudeGPT training and fine-tuning route

**Suggested topic:** LowAltitudeGPT: Tool-Use and Simulation-Feedback Tuning for Low-Altitude Traffic Intelligence

**Core judgment:**

It’s not time to train the “large low-altitude traffic model” from scratch now. There are three problems with this:

1. The amount of data is insufficient to support foundation model-level contributions;
2. The review will ask whether the model contribution exceeds that of ordinary large models + RAG + tool-use;
3. The training cost is high, but it is not necessarily more valuable than the agent/verifier/simulator closed loop.

A more feasible route is **ordinary large model + Agent + Skills/MCP + RAG + verifier + simulator** to run through it first, and then precipitate the running logs into trainable data. MCP is essentially a standard interface that exposes tools and context to LLM, and is suitable for unified access to schedulers, planners, verifiers, simulators, databases and document libraries [47].

A review of large low-altitude economic models also breaks down low-altitude systems into facility networks, information networks, route networks, and service networks, and emphasizes that large models need to be combined with edge computing, 6G/ISAC, and trusted distributed intelligence [50]. This shows that our paper cannot just be written as "training a chat model", but must be written as a closed loop of models, tools, networks, operation control and system evaluation.

**Model selection suggestions:**| Stage | Recommended model | Reason |
|---|---|---|
| Solution exploration / data generation / teacher | High-capability API model | Quickly generate tasks, tool traces, counterexample explanations and evaluation samples first, without using the API as a final reproducible dependency |
| Local reproducible experiments | Qwen3-8B / Qwen3-14B / Qwen3-32B | Qwen3 officially supports local operation, deployment, quantification and training processes, with good Chinese language, tool calling and engineering ecology [45] |
| Reasoning/Mathematics/Constraint Interpretation | DeepSeek-R1-Distill-Qwen-14B / 32B | The DeepSeek-R1 series emphasizes RL-motivated reasoning capabilities. The distill version can be deployed locally and is fine-tuned based on the Qwen/Llama open source model [46] |
| Multi-modal low-altitude perception | Qwen-VL / Qwen3-VL / other open source VLM | Semantic understanding for pictures, video frames, maps, track charts, and 3DGS render |
| Edge-end small model | Qwen3-4B / 8B quantitative version, SLM | Used for end-side status summary, anomaly detection, low-latency fallback |

**Training data design:**

| Data type | Source | Training target |
|---|---|---|
| NL mission -> LowAltitudeIR | Manual template + API teacher + real task rewriting | Task parsing and structured representation |
| tool-use trace | Paper A/B/C/D/E tool call log | Learn when to call scheduling, planning, verification, simulation |
| verifier counterexample | Spot/RTAMT/CBF/simulator feedback | Learn to fix unexecutable or dangerous plans |
| simulation rollout | SUMO/AirSim/self-developed low-altitude simulation | Learn to explain system bottlenecks from results |
| failure case | collision, LoWC, timeout, queue explosion, insufficient energy consumption | Learn risk diagnosis and emergency de-escalation |
| human audit data | Manual selection of more reasonable solutions | DPO/preference optimization |

**Training Phase:**1. **RAG + prompt baseline**: No fine-tuning, only use the literature library, regulations, system description and tool schema.
2. **LoRA/QLoRA SFT**: training NL-to-IR, tool-call, result interpretation, and counterexample repair.
3. **DPO/IPO**: Use manual preferences or verifier scoring preferences to optimize "safe, executable, concise, and explainable".
4. **GRPO/RL-style tuning**: Use simulation to reward training task success rate, low violation, low latency and format compliance. FlightGPT’s SFT + GRPO route can be used as a UAV VLN reference [40].
5. **distillation**: Distill API teacher/32B model capabilities to 8B/4B for local and edge deployment.

**Evaluation indicators:**

-task success;
- LowAltitudeIR exact/semantic match;
- tool-call precision/recall;
- executable plan rate;
- safety violation rate;
- repair success rate;
- hallucination rate;
- latency / token cost;
- cross-city / cross-task generalization;
- human audit pass rate.

### 3.10 Paper K: Low-altitude cloud brain inference acceleration and software and hardware collaboration

**Suggested topic:** Edge-Cloud Co-Optimized Inference for Low-Altitude Traffic Cloud-Brain Agents

**Why can this article be written:**If we want to do both software and hardware in the future, inference acceleration cannot just be engineering optimization. It needs to be written as **Real-time intelligent service problem under low-altitude traffic system constraints**: There are large models and global states on the cloud side, low latency and privacy/communication constraints on the edge side, and power consumption, computing power, heat dissipation and real-time control constraints on the drone side. General-Purpose Aerial Intelligent Agents have given a direct signal in the direction of hardware-software co-design: the 14B model onboard runs about 5-6 tokens/sec, has a peak power consumption of about 220W, and adopts a bidirectional cognitive architecture of slow LLM planning + fast reaction control [51].

**System Architecture:**

```text
cloud brain
  - full LLM / VLM
  - global scheduler
  - long-horizon planner
  - batch simulation evaluator

edge station / vertiport
  - quantized 8B/14B model
  - local RAG cache
  - route/conflict verifier
  - streaming state summarizer

onboard UAV
  - tiny policy / controller
  - VIO / obstacle avoidance
  - emergency fallback
  - compressed semantic state uplink
```

**Accelerated technology route:**

- Server: vLLM/PagedAttention/continuous batching/prefix cache. The core value of PagedAttention is to reduce KV cache waste and improve batch serving throughput [48].
- NVIDIA GPU production deployment: TensorRT-LLM, performing LLM inference with TensorRT engines, Python/C++ runtime, and GPU optimization [49].
- End/edge: AWQ/GPTQ/GGUF INT4/INT8, KV cache compression, speculative decoding, small model router.
- Tool call optimization: caching tool schema, caching static RAG search results, and compiling high-frequency tool calls into deterministic skills.
- Operator direction: attention kernel, paged KV cache, prefill/decode separation, batch scheduler, MoE expert routing, vision encoder caching.

**Thesis points available:**1. **System paper**: latency/cost/energy profiling of low-altitude cloud brain agent workload.
2. **Algorithm-System Paper**: Dynamic selection of API/cloud 32B/edge 14B/onboard 4B based on task risk.
3. **Operator/Inference paper**: KV cache and batching optimization for low-altitude traffic with multiple agents, multiple tools, long context, and streaming status updates.
4. **Hardware collaboration paper**: Jetson Orin / RTX workstation / cloud GPU three-tier deployment, evaluating tokens/sec, end-to-end latency, energy per decision, and safety fallback rate.

**Recommended venue:**

- Partial Transportation Systems: T-ITS / IEEE IoT Journal.
- Partial edge intelligence: IEEE TMC / IEEE Internet of Things Journal / ACM TECS.
- Partial robotic system: IROS/ICRA system paper.
- Partial operators and systems: Starting from MLSys / SC workshop / DAC/DATE workshop, it is not recommended to go directly to the top system conference at the beginning.

---

## 4. Recommendation priority| Priority | Articles | Recent Actions | Reasons |
|---|---|---|---|
| P0-Active | Paper B | Freeze problem formulation, queue model, experimental benchmark | Most similar to the TR-C system paper, and most suitable for low-altitude economy/emergency |
| P0-Active | Paper A | Rewriting PPO/MAPPO into Robust Low-altitude Conflict Resolution Paper | Already have algorithm basis, but need traffic indicators and strong baseline |
| P0-Active | Paper C | Converged to Fisher + 3DGS + safe planning, no longer expand too much | The algorithm is innovative and can be used in robots/AI/ITS |
| P1-Support | Paper D | Reuse 76 million exploration logs and do coverage-guided testing | Provide safety-critical scenarios, failure taxonomy and benchmark for A/C |
| P1-Bridge | Paper G | Make the tool interface and CloudBrain-Agent benchmark first | String A/B/C/D/E into a low-altitude cloud brain instead of an empty chat model |
| P2-Embodied | Paper I | Making aerial VLN/VLA small-scale pilot: simulation data, expert trajectories, end-to-end/hybrid strategy comparison | This is the main line leading to embodied AGI, but it requires A/C perception and security tools to be stabilized first |
| P2-Model | Paper J | Precipitate LowAltitudeIR, tool trace, verifier feedback, and then do LoRA/SFT/GRPO | First have data closed loop, then fine-tune the vertical model |
| P3-System | Paper K | Wait for the CloudBrain-Agent workload to be fixed before doing vLLM/TensorRT/quantization/end-cloud collaboration | The software and hardware direction can be written, but it requires a real workload to be like a paper |
| P3-Planning | Paper H | As a subsequent extension of TR-C/T-ITS | Requires mature urban data pipeline and ODD definition |

**Execution order suggestions:**1. Do not change the current main battlefield: A/B/C continue to advance.
2. Supplement Paper D first, because it directly enhances the experimental credibility of A/C and can also generate subsequent model training data.
3. Make Paper G again and package A/B/C/D/E into a tool-based cloud brain.
4. Paper I/J/K Don’t rush to start a big project; do a small pilot and data schema first. Before starting the actual question, you must answer: Where does the data come from, what are the evaluation indicators, and whether it can be stronger than ordinary large models + tool calls.

---

## 4.1 Literature support matrix

In order to avoid document stacking, the current 51 references are used in a closed manner according to the direction of the paper:| Directions | Documentation Groups | Usage |
|---|---|---|
| Submission and transportation system positioning | [1,2] | Determine the framing differences of TR-C / T-ITS |
| Paper A: Multi-agent conflict resolution | [3-12] | PPO/MAPPO, MAT/FACMAC/HAPPO and EGO-Swarm/MADER/RMADER/RACER/PANTHER/GCOPTER baseline |
| Paper B: Hundreds of UAV Scheduling | [13-19] | Low-altitude delivery resource allocation, UAM scheduling, safe learning, truck-drone/UAV-UGV multi-modal delivery |
| Paper C: 3DGS active sensing | [20-26] | 3DGS, ActiveNeRF, FisherRF, GS-Planner, HGS-Planner, POp-GS, NVF |
| Paper D: Safety-critical scenario coverage | [27-30] | Shuo Feng accelerated testing, scenario library, SafeBench |
| Paper E: Language Planning and Verification | [31-34] | Lang2LTL, NL2LTL, LTLCodeGen, ConformalNL2LTL |
| Paper G: Low-altitude cloud brain Agent | [35-38,47,50,51] | UrbanGPT/UniST/TrafficGPT/DriveLM, MCP, low-altitude economic large model review, aerial intelligent agent |
| Paper I: Embodied Low Altitude / Aerial VLA | [39-44] | SINGER, FlightGPT, UAV-VLN, OpenVLA, Octo, RT-2 |
| Paper J: Model training and fine-tuning | [40,45,46,47,50] | SFT/GRPO reference, Qwen3, DeepSeek-R1, MCP/tool-use, low-altitude large model system positioning |
| Paper K: Inference acceleration and software and hardware collaboration | [45,48,49,51] | Qwen3 deployment ecology, vLLM/PagedAttention, TensorRT-LLM, onboard 14B aerial agent hardware constraints|---

## 5. Zotero organizes status

Target Zotero collection name:

```text
低空规划论文参考
```

Currently, two levels of organization have been completed:

| Project | Status |
|---|---|
| Zotero collection | already exists, collection key is `FVHS3SKY`, local treeViewID is `C17` |
| Zotero local selection link | `zotero://select/library/collections/FVHS3SKY` |
| Imported documents | 51 top-level items |
| item type distribution | `journalArticle` 17 items, `conferencePaper` 11 items, `document/preprint/webpage` 23 items |
| Local backup BibTeX | `zotero/low-altitude-planning-references-20260527.bib`; Increment: `zotero/low-altitude-planning-references-update-20260528.bib` |

The import method uses Zotero's local connector server instead of writing `zotero.sqlite` directly. The specific process is:

1. Use `pandoc` to check that BibTeX can be parsed as CSL JSON.
2. Import `zotero/low-altitude-planning-references-20260527.bib` through Zotero local `/connector/import`.
3. Update the target collection of the imported session to `C17 / Low Altitude Planning Paper Reference` through `/connector/updateSession`.
4. Use Zotero local API and read-only SQLite to double verify that there are 51 top-level documents in the collection.If you continue to add documents in the future, it is recommended to update the local BibTeX first, and then import Zotero through the same connector import/updateSession process. Do not modify SQLite directly.

---

## 6. Follow-up execution plan

### 6.1 Week 1: Freeze three papers in progress

- It is clear that Paper A/B/C is the current active pipeline, and subsequent papers will no longer be written with the same priority.
- Paper A: Freeze conflict scenarios, action space, baseline and traffic indicators.
- Paper B: frozen queue model, Lyapunov objective, synthetic benchmark and TR-C framing.
- Paper C: Freeze theoretical interfaces and planning-aware metrics for FIM/3DGS/NBV.
- The initial import of Zotero collection and the incremental import on 2026-05-28 have been completed; the next step is to add PDF, summary notes and priority tags for each article.

### 6.2 Weeks 2-3: Supplementing the literature matrix and subsequent route novelty checking

- Compile at least 25 highly relevant documents for each main article.
- Each article forms a `related work matrix`: problem, method, data, metric, gap, our angle.
- For Paper A/B/C, mark the papers that “must reproduce the baseline” and “only serve as related work”.
- Perform novelty checking on Paper I/J/K separately:
  - Paper I: aerial VLN, AerialVLA, SINGER, FlightGPT, OpenVLA, Octo, RT-2;
  - Paper J: Qwen3, DeepSeek-R1, tool-use tuning, MCP, RAG, simulation-feedback training;
  - Paper K: vLLM, TensorRT-LLM, quantification, KV cache, edge-cloud deployment.

### 6.3 Weeks 4-8: Advance the three experimental lines of Paper B/A/C first- Paper B: synthetic UAM queuing benchmark + FCFS/greedy/MILP/backpressure/MARL baseline.
- Paper A: corridor conflict simulation + ORCA/CBF/RMADER/MAPPO baseline.
- Paper C: 3DGS NBV pipeline + FisherRF/ActiveNeRF/GS-Planner/POp-GS baseline.
- Paper D: Only make a light pilot, organize 76 million exploration logs into coverage/failure taxonomy, and do not compete for A/B/C resources.

### 6.4 Weeks 9-12: Constructing the minimum closed loop of low-altitude cloud brain

- Expose A/B/C/D/E as tool interfaces: scheduler, conflict resolver, active mapper, scenario tester, verifier.
- Define `LowAltitudeIR` to unify mission, airspace, UAV, resource, risk and tool call results.
- First use API teacher + local Qwen/DeepSeek to make CloudBrain-Agent baseline, and don’t rush to fine-tune it.
- Collect tool trace, failure repair, and simulation rollout as training data for Paper J.

### 6.5 Weeks 13-20: Deciding on submission and model routes- If Paper B’s queue stability and hundred-shelf level results are the most stable: vote for TR-C first.
- If Paper A has the strongest conflict safety and generalization: vote for T-ITS/T-RO first.
- If Paper C has the strongest Fisher + 3DGS theoretical and visual results: vote for T-RO/ICRA/IROS first.
- If Paper D has the best coverage/failure discovery data: invest in T-ITS first.
- If CloudBrain-Agent can already call A/B/C/D/E tools stably: start the AAAI/IJCAI version.
- If 5k-20k high-quality tool traces / verifier feedback / simulation rollout have been accumulated: start LowAltitudeGPT LoRA/SFT.
- If the agent workload is fixed and latency becomes a bottleneck: Start Paper K’s vLLM/TensorRT/edge quantization experiment.

---

## 7. References

[1] Elsevier. *Transportation Research Part C: Emerging Technologies: Aims and Scope.* URL: <https://www.sciencedirect.com/journal/transportation-research-part-c-emerging-technologies>

[2] IEEE Intelligent Transportation Systems Society. *IEEE Transactions on Intelligent Transportation Systems: Scope.* URL: <https://ieee-itss.org/pub/t-its/>[3] Chao Yu, Akash Velu, Eugene Vinitsky, Yu Wang, Alexandre M. Bayen, and Yi Wu. “The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games.” *Advances in Neural Information Processing Systems*, 2022. URL: <https://arxiv.org/abs/2103.01955>

[4] Muning Wen, Jakub Grudzien Kuba, Runji Lin, Weinan Zhang, Ying Wen, Jun Wang, and Yaodong Yang. “Multi-Agent Reinforcement Learning is a Sequence Modeling Problem.” *NeurIPS*, 2022. URL: <https://proceedings.neurips.cc/paper_files/paper/2022/hash/69413f87e5a34897cd010ca698097d0a-Abstract-Conference.html>

[5] Bei Peng, Tabish Rashid, Christian Schroeder de Witt, Pierre-Alexandre Kamienny, Philip Torr, Wendelin Boehmer, and Shimon Whiteson. “FACMAC: Factored Multi-Agent Centralized Policy Gradients.” *NeurIPS*, 2021. URL: <https://proceedings.neurips.cc/paper/2021/hash/65b9eea6e1cc6bb9f0cd2a47751a186f-Abstract.html>[6] Jakub Grudzien Kuba, Ruiqing Chen, Muning Wen, Ying Wen, Fudan Sun, Jun Wang, and Yaodong Yang. "Trust Region Policy Optimization in Multi-Agent Reinforcement Learning." arXiv:2109.11251, 2021. URL: <https://arxiv.org/abs/2109.11251>

[7] Boyu Zhou, Xin Zhou, Jun Zhang, Fei Gao, and Shaojie Shen. "EGO-Swarm: A Fully Autonomous and Decentralized Quadrotor Swarm System in Cluttered Environments." *ICRA*, 2021. DOI: 10.1109/ICRA48506.2021.9561902. URL: <https://arxiv.org/abs/2011.04183>

[8] Jesus Tordesillas, Brett T. Lopez, and Jonathan P. How. “MADER: Trajectory Planner in Multiagent and Dynamic Environments.” *IEEE Transactions on Robotics*, 38(1):463-476, 2022. URL: <https://arxiv.org/abs/2010.11061>[9] Kota Kondo, Reinaldo Figueroa, Juan Rached, Jesus Tordesillas, Parker C. Lusk, and Jonathan P. How. “Robust MADER: Decentralized Multiagent Trajectory Planner Robust to Communication Delay in Dynamic Environments.” arXiv:2303.06222, 2023. URL: <https://arxiv.org/abs/2303.06222>

[10] Boyu Zhou, Hao Xu, and Shaojie Shen. "RACER: Rapid Collaborative Exploration With a Decentralized Multi-UAV System." *IEEE Transactions on Robotics*, 2023. DOI: 10.1109/TRO.2023.3236945. URL: <https://arxiv.org/abs/2209.08533>

[11] Jesus Tordesillas and Jonathan P. How. “PANTHER: Perception-Aware Trajectory Planner in Dynamic Environments.” *IEEE Access*, 10:22662-22677, 2022. DOI: 10.1109/ACCESS.2022.3154037. URL: <https://arxiv.org/abs/2103.06372>[12] Zhepei Wang, Xin Zhou, Chao Xu, and Fei Gao. "Geometrically Constrained Trajectory Optimization for Multicopters." *IEEE Transactions on Robotics*, 38(5):3259-3278, 2022. DOI: 10.1109/TRO.2022.3160022. URL: <https://arxiv.org/abs/2103.00190>

[13] Ang Li, Mark Hansen, and Bo Zou. “Traffic Management and Resource Allocation for UAV-Based Parcel Delivery in Low-Altitude Urban Space.” *Transportation Research Part C: Emerging Technologies*, 143:103808, 2022. DOI: 10.1016/j.trc.2022.103808. URL: <https://doi.org/10.1016/j.trc.2022.103808>

[14] Mehdi Bennaceur, Rémi Delmas, and Youssef Hamadi. “Passenger-Centric Urban Air Mobility: Fairness Trade-Offs and Operational Efficiency.” *Transportation Research Part C: Emerging Technologies*, 136:103519, 2022. DOI: 10.1016/j.trc.2021.103519. URL: <https://doi.org/10.1016/j.trc.2021.103519>[15] Roberto Pinto and Alexandra Lagorio. “Point-to-Point Drone-Based Delivery Network Design with Intermediate Charging Stations.” *Transportation Research Part C: Emerging Technologies*, 135:103506, 2022. DOI: 10.1016/j.trc.2021.103506. URL: <https://doi.org/10.1016/j.trc.2021.103506>

[16] Qinshuang Wei, Gustav Nilsson, and Samuel Coogan. “Capacity-Constrained Urban Air Mobility Scheduling.” arXiv:2107.02900, 2021. URL: <https://arxiv.org/abs/2107.02900>

[17] Surya Murthy, Natasha A. Neogi, and Suda Bharadwaj. “Scheduling for Urban Air Mobility Using Safe Learning.” arXiv:2209.15457, NASA NTRS, 2022. URL: <https://arxiv.org/abs/2209.15457>[18] Jiahao Xing, Tong Guo, and Lu Tong. "Reliable Truck-Drone Routing with Dynamic Synchronization: A High-Dimensional Network Programming Approach." *Transportation Research Part C: Emerging Technologies*, 165:104698, 2024. DOI: 10.1016/j.trc.2024.104698. URL: <https://doi.org/10.1016/j.trc.2024.104698>

[19] Bolong Zhou, Wei Zeng, and Hai Yang. "Multi-Trip UAV-UGV Delivery Network Design with Release Times." *Transportation Research Part C: Emerging Technologies*, 181:105389, 2025. DOI: 10.1016/j.trc.2025.105389. URL: <https://doi.org/10.1016/j.trc.2025.105389>

[20] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. “3D Gaussian Splatting for Real-Time Radiance Field Rendering.” *ACM Transactions on Graphics / SIGGRAPH*, 42(4), 2023. DOI: 10.1145/3592433. URL: <https://arxiv.org/abs/2308.04079>[21] Xuran Pan, Zihang Lai, Shiji Song, and Gao Huang. "ActiveNeRF: Learning Where to See with Uncertainty Estimation." *ECCV*, 2022. URL: <https://arxiv.org/abs/2209.08546>

[22] Wen Jiang, Boshu Lei, and Kostas Daniilidis. “FisherRF: Active View Selection and Mapping with Radiance Fields Using Fisher Information.” *ECCV*, 2024. DOI: 10.1007/978-3-031-72624-8_24. URL: <https://eccv.ecva.net/virtual/2024/oral/1226>

[23] Rui Jin, Yuman Gao, Yingjian Wang, Haojian Lu, and Fei Gao. "GS-Planner: A Gaussian-Splatting-Based Planning Framework for Active High-Fidelity Reconstruction." arXiv:2405.10142, 2024. URL: <https://arxiv.org/abs/2405.10142>

[24] Zijun Xu, Rui Jin, Ke Wu, Yi Zhao, Zhiwei Zhang, Jieru Zhao, Fei Gao, Zhongxue Gan, and Wenchao Ding. "HGS-Planner: Hierarchical Planning Framework for Active Scene Reconstruction Using 3D Gaussian Splatting." arXiv:2409.17624, 2024. URL: <https://arxiv.org/abs/2409.17624>[25] Joey Wilson, Marcelino Almeida, Sachit Mahajan, Martin Labrie, Maani Ghaffari, Omid Ghasemalizadeh, Min Sun, Cheng-Hao Kuo, and Arnab Sen. “POp-GS: Next Best View in 3D-Gaussian Splatting with P-Optimality.” *CVPR*, 2025. URL: <https://cvpr.thecvf.com/virtual/2025/poster/34708>

[26] Shangjie Xue, Jesse Dill, Pranay Mathur, Frank Dellaert, Panagiotis Tsiotras, and Danfei Xu. “Neural Visibility Field for Uncertainty-Driven Active Mapping.” *CVPR*, 2024. URL: <https://arxiv.org/abs/2406.06948>

[27] Shuo Feng, Xintao Yan, Haowei Sun, Yiheng Feng, and Henry X. Liu. “Intelligent Driving Intelligence Test for Autonomous Vehicles with Naturalistic and Adversarial Environment.” *Nature Communications*, 12:748, 2021. DOI: 10.1038/s41467-021-21007-8. URL: <https://www.nature.com/articles/s41467-021-21007-8>[28] Shuo Feng, Yiheng Feng, Chao Yu, Yi Zhang, and Henry X. Liu. “Testing Scenario Library Generation for Connected and Automated Vehicles, Part I: Methodology.” *IEEE Transactions on Intelligent Transportation Systems*, 2021. DOI: 10.1109/TITS.2020.2972211. URL: <https://doi.org/10.1109/TITS.2020.2972211>

[29] Shuo Feng, Yiheng Feng, Haowei Sun, Yi Zhang, and Henry X. Liu. "Testing Scenario Library Generation for Connected and Automated Vehicles, Part II: Case Studies." *IEEE Transactions on Intelligent Transportation Systems*, 2021. DOI: 10.1109/TITS.2020.2988309. URL: <https://doi.org/10.1109/TITS.2020.2988309>[30] Chejian Xu, Wenhao Ding, Baiming Li, Xinqing Chen, Ding Zhao, Bo Li, Jiajun Liu, and Hang Zhao. “SafeBench: A Benchmarking Platform for Safety Evaluation of Autonomous Vehicles.” *NeurIPS Datasets and Benchmarks*, 2022. URL: <https://proceedings.neurips.cc/paper_files/paper/2022/hash/a48ad12d588c597f4725a8b84af647b5-Abstract-Datasets_and_Benchmarks.html>

[31] Jason Liu, Ankit Shah, Eric Rosen, and Stefanie Tellex. “Lang2LTL: Translating Natural Language Commands to Temporal Robot Task Specification.” *PMLR/CoRL*, 229, 2023. URL: <https://proceedings.mlr.press/v229/liu23d.html>

[32] Francesco Fuggitti and Tathagata Chakraborti. “NL2LTL: A Python Package for Converting Natural Language Instructions to Linear Temporal Logic Formulas.” *AAAI Demonstration*, 37(13):16428-16430, 2023. DOI: 10.1609/aaai.v37i13.27068. URL: <https://ojs.aaai.org/index.php/AAAI/article/view/27068>[33] Behrad Rabiei and Mahesh A. Kumar. “LTLCodeGen: Code Generation of Syntactically Correct Temporal Logic for Robot Task Planning.” arXiv:2503.07902, 2025. URL: <https://arxiv.org/abs/2503.07902>

[34] Jun Wang, David Smith Sundarsingh, Jyotirmoy V. Deshmukh, and Yiannis Kantaros. “ConformalNL2LTL: Translating Natural Language Instructions into Temporal Logic Formulas with Conformal Correctness Guarantees.” arXiv:2504.21022, 2025. URL: <https://arxiv.org/abs/2504.21022>

[35] Zhonghang Li, Lianghao Xia, Jiabin Tang, Yong Xu, Lei Shi, Long Xia, Dawei Yin, and Chao Huang. "UrbanGPT: Spatio-Temporal Large Language Models." arXiv:2403.00813, 2024. URL: <https://arxiv.org/abs/2403.00813>

[36] Yuan Yuan, Jingtao Ding, Jie Feng, Depeng Jin, and Yong Li. “UniST: A Prompt-Empowered Universal Model for Urban Spatio-Temporal Prediction.” *KDD*, 2024. DOI: 10.1145/3637528.3671662. URL: <https://dblp.org/rec/conf/kdd/0032D0J024>[37] Jinhui Ouyang, Yijie Zhu, Xiang Yuan, and Di Wu. “TrafficGPT: Towards Multi-Scale Traffic Analysis and Generation with Spatial-Temporal Agent Framework.” arXiv:2405.05985, 2024. URL: <https://arxiv.org/abs/2405.05985>

[38] Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Jens Beisswenger, Ping Luo, Andreas Geiger, and Hongyang Li. “DriveLM: Driving with Graph Visual Question Answering.” *ECCV*, 2024. URL: <https://github.com/OpenDriveLab/DriveLM>

[39] Maximilian Adang, JunEn Low, Ola Shorinwa, and Mac Schwager. “SINGER: An Onboard Generalist Vision-Language Navigation Policy for Drones.” arXiv:2509.18610, 2025. URL: <https://arxiv.org/abs/2509.18610>[40] Hengxing Cai, Jinhan Dong, Jingjun Tan, Jingcheng Deng, Sihang Li, Zhifeng Gao, Haidong Wang, Zicheng Su, Agachai Sumalee, and Renxin Zhong. "FlightGPT: Towards Generalizable and Interpretable UAV Vision-and-Language Navigation with Vision-Language Models." *EMNLP*, 2025. DOI: 10.18653/v1/2025.emnlp-main.338. URL: <https://aclanthology.org/2025.emnlp-main.338/>

[41] Pranav Saxena, Nishant Raghuvanshi, and Neena Goveas. “UAV-VLN: End-to-End Vision Language guided Navigation for UAVs.” arXiv:2504.21432, 2025. URL: <https://arxiv.org/abs/2504.21432>[42] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. “OpenVLA: An Open-Source Vision-Language-Action Model." arXiv:2406.09246, 2024. URL: <https://arxiv.org/abs/2406.09246>

[43] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, Jianlan Luo, You Liang Tan, Lawrence Yunliang Chen, Pannag Sanketi, Quan Vuong, Ted Xiao, Dorsa Sadigh, Chelsea Finn, and Sergey Levine. “Octo: An Open-Source Generalist Robot Policy." arXiv:2405.12213, 2024. URL: <https://arxiv.org/abs/2405.12213>[44] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. “RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control.” arXiv:2307.15818, 2023. URL: <https://arxiv.org/abs/2307.15818>

[45] Qwen Team. “Qwen3 Technical Report.” arXiv:2505.09388, 2025; QwenLM/Qwen3 official repository. URL: <https://arxiv.org/abs/2505.09388>; <https://github.com/QwenLM/Qwen3>

[46] DeepSeek-AI. “DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.” arXiv:2501.12948, 2025. URL: <https://arxiv.org/abs/2501.12948>; model card: <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B>

[47] OpenAI. “Model Context Protocol (MCP): OpenAI Agents SDK.” Official documentation, 2026. URL: <https://openai.github.io/openai-agents-js/guides/mcp/>[48] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. “Efficient Memory Management for Large Language Model Serving with PagedAttention.” arXiv:2309.06180, 2023. URL: <https://arxiv.org/abs/2309.06180>

[49] NVIDIA. “NVIDIA TensorRT-LLM.” Official documentation, 2026. URL: <https://docs.nvidia.com/tensorrt-llm/index.html>

[50] Jinpeng Hu, Wei Wang, Yuxiao Liu, and Jing Zhang. "Large Model in Low-Altitude Economy: Applications and Challenges." *Big Data and Cognitive Computing*, 10(1):33, 2026. DOI: 10.3390/bdcc10010033. URL: <https://www.mdpi.com/2504-2289/10/1/33>

[51] Ji Zhao and Xiao Lin. "General-Purpose Aerial Intelligent Agents Empowered by Large Language Models." arXiv:2503.08302, 2025. URL: <https://arxiv.org/abs/2503.08302>