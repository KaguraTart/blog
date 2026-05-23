---
title: "Paper E Experimental Task Book v2: Verification and Error Correction UAV Language Planning for AAAI"
description: "v2 focuses on submissions to AAAI top conferences: Supplementing 30+ real and citable regular conference/top journal/key preprint documents, deepening the experimental indicators, comparison and ablation schemes, and reproducible experimental protocols of VERA-UAV, and providing mathematical proof of relative completeness."
pubDate: 2026-05-17
updatedDate: 2026-05-23
tags: ["Paper E", "AAAI", "UAV", "LLM", "LTL", "STL", "Formal verification", "Experimental task book", "Proof of completeness"]
category: Tech
---

# Paper E Experimental Task Book v2: Verification and Error Correction UAV Language Planning for AAAI

> This file still uses the file name `paper-e-vera-uav-experiment-taskbook-v1-20260517.md` because this round requires "direct modification on the V1 version". The text, title and release notes have all been upgraded to **v2**. This article is not a final paper draft, but an executable experimental task statement: clarify the research positioning of Paper E, real citable documents, algorithm solutions, data construction, comparative experiments, ablation experiments, evaluation indicators, theoretical completeness boundaries and subsequent AAAI/T-ITS promotion plans. Supplementary focus on 2026-05-19 is: data leakage prevention, failure taxonomy, parameter budgeting, indicator formulas, chart planning and AAAI compliance risks.

---

## 1. Research background and goals

Urban low-altitude UAV mission planning is moving from "engineer-preset routes" to "natural language mission-driven". In actual applications, operators are more likely to give the following instructions:

- "First check the east facade of Building 3, then go to the landing point on the roof and wait."
- "Avoid the air above the hospital and reach the temporary delivery area within 30 seconds."
- "If the south corridor is occupied, bypass the west corridor but keep a safe distance of more than 20 meters throughout."

These instructions simultaneously include semantic understanding, temporal order, spatial constraints, continuous trajectory safety, and reachability judgments. Large language models (LLM) are good at understanding natural language and generating candidate plans, but they cannot guarantee that the output plan is executable in physical space, nor can it guarantee that aviation safety constraints are met. Formal methods are good at giving verifiable semantics, such as Linear Temporal Logic (LTL) and Signal Temporal Logic (STL), but direct handwritten specifications require professional knowledge and are difficult to serve non-expert operators.

Existing work has proven that natural language to LTL translation can significantly reduce the threshold for writing robot task specifications. For example, Lang2LTL converts complex navigation commands into LTL and performs generalization evaluation in unseen environments [1]; NL2LTL provides an open source Python package from natural language to LTL [2]; LTLCodeGen uses code generation to improve the grammatical correctness of LTL and integrates it into robot path planning [3]; ConformalNL2LTL further attempts to use conformal prediction to guarantee translation accuracy [4]. These works provide an important foundation for this study.But for low-altitude UAV scenarios, just doing NL-to-LTL translation is not enough. UAV missions have three additional requirements:

1. **Continuous safety constraints**: Constraints such as flight altitude, speed, obstacle distance, time window, etc. are naturally constraints on continuous signals, and are more suitable to be evaluated by STL robustness.
2. **Executable Trajectory Closed Loop**: Correct specifications do not mean that the trajectory is feasible and must be verified by maps, dynamics and planners.
3. **Errors can be fixed**: LLM errors should not only be judged as errors, but should be converted into counterexample or robustness feedback by the verifier, and then drive LLM correction.

Therefore, this article proposes **VERA-UAV**: a verification and error-correcting neuro-symbolic planning framework for UAV natural language tasks. The AAAI version prioritizes answering a core question:

> Given a natural language UAV mission, how can a native open source LLM generate verifiable, repairable, and executable LTL/STL mission specifications and trajectories, rather than just generating textual plans that appear reasonable but are not provably safe?

The AAAI main conference version focuses on AI planning, neuro-symbolic verification and LLM self-repair. System-level content such as AirSim, real low-altitude logistics, and multi-UAV airspace throughput will be put into subsequent T-ITS extended versions.

---

## 2. Problem definition and core assumptions

### 2.1 Input and output

Given a UAV task instance:

$$
\mathcal{I} = (x_{\text{NL}}, \mathcal{M}, s_0)
$$

Among them, $x_{\text{NL}}$ is the natural language task instruction, $\mathcal{M}$ is the urban low-altitude map with semantic annotation, and $s_0$ is the UAV initial state. The map contains buildings, no-fly zones, passable airspace, landing points, inspection targets, dynamic obstacles and altitude levels.

System output:

$$
\mathcal{O} = (\text{TaskIR}, \varphi_{\text{LTL}}, \varphi_{\text{STL}}, \tau, r)
$$Where TaskIR is the structured intermediate representation, $\varphi_{\text{LTL}}$ is the discrete timing task specification, $\varphi_{\text{STL}}$ is the continuous trajectory constraint, $\tau$ is the candidate trajectory, and $r$ is the verification result. If the task cannot be met, the system should output `UNSAT` or `NEED_CLARIFICATION` instead of forcibly generating an unsafe trajectory.

### 2.2 Task Type

The AAAI main experiment covers six types of tasks:

| Type | Example | Key Difficulties |
|------|------|----------|
| Reach-avoid | Reach A, avoid B | Basic reachability and obstacle avoidance |
| Ordered waypoints | First to A, then to B | Temporal order |
| Patrol / inspection | Patrol A, B, C | Multi-target coverage |
| Time-window delivery | Arrive at A within 30 seconds | Continuous time constraints |
| Emergency landing | If the road ahead is unreachable, go to the nearest landing point | Conditions and alternative strategies |
| Ambiguous / impossible | "Go to that safe place" or mutually exclusive constraints | Clarification and unsatisfiable detection |

### 2.3 Core Assumptions

This article does not assume that LLM is reliable by itself. Instead, this article assumes that LLMs often make the following mistakes:

- Generate LTL/STL with illegal syntax.
- Missing security constraints in natural language.
- Reference to an entity that does not exist in the map.
- Give a sequence of tasks that satisfies the text but is not executable.
- Violation of minimum distance, height, or time window constraints on continuous trajectories.

The core assumption of VERA-UAV is: **If the verifier can convert these errors into structured counterexamples, unsat core and robustness feedback, the correction success rate of local open source LLM will be significantly higher than that of pure prompt retry; further, if the system retains the symbol enumeration fallback within the limited DSL, the algorithm can obtain relative completeness, rather than basing completeness on LLM reliability. **

---

## 3. Related work and citable papers

### 3.1 Overview of literature mapOne of the problems with v1 is that there are too few references, and it is easy for reviewers to think that "it is just a UAV application based on Lang2LTL / LTLCodeGen". v2 expands related work into five lines: natural language to temporal logic, LLM planning and self-healing, STL/formal verification, shielding and security agent, UAV-VLN and low-altitude applications. The table below lists **37 strongly relevant documents**, each of which is cited in this article.| Number | Literature | Venue / status | Relationship to this article |
|------|------|----------------|------------|
| [1] | Lang2LTL | CoRL 2023/PMLR | Direct starting point for NL-to-LTL grounding |
| [2] | NL2LTL | AAAI 2023 Demo | template/tool baseline |
| [3] | LTLCodeGen | IROS 2025/arXiv | The strongest direct baseline, code generation guaranteed syntax |
| [4] | ConformalNL2LTL | arXiv 2025 | Translation credibility and rejection mechanism reference |
| [5] | NL2SpaTiaL | arXiv 2025/2026 | Structured logic tree and spatial relationship inspiration |
| [6] | T3 Planner | arXiv 2025 | Self-study formal LLM + STL motion planning direct competition |
| [7] | SENTINEL | arXiv 2025/2026 | Multi-layered formal safety evaluation |
| [8] | LogicGuard | arXiv 2025 | Temporal-logic critic and security constraint generation |
| [9] | Pro2Guard | arXiv 2025 | probabilistic runtime monitoring |
| [10] | Generalized Planning in PDDL Domains with LLMs | AAAI 2024 | The value of verifier/debugging feedback for planning |
| [11] | Critical investigation of LLM planning | NeurIPS 2023 | Explain that LLM has limited direct planning capabilities |
| [12] | LLM+P | arXiv 2023 | Framework reference for LLM + classical planner |
| [13] | PlanBench | NeurIPS 2023Datasets & Benchmarks | LLM planning benchmark design reference |
| [14] | ReAct | ICLR 2023 | reasoning-action loop baseline |
| [15] | SayCan | CoRL 2022 | affordance-grounded LLM planning baseline |
| [16] | Code as Policies | ICRA 2023 | LLM generates executable program policies |
| [17] | ProgPrompt | ICRA 2023 / Autonomous Robots | situated robot task plan generation |
| [18] | Temporal-Logic-Based Reactive Mission and Motion Planning | IEEE T-RO 2009 | Robot LTL planning classic foundation |
| [19] | Synthesis for Robots | Annual Review 2018 | Review of formal synthesis and robot behavior feedback |
| [20] | Monitoring Temporal Properties of Continuous Signals | FORMATS/FTRTFT 2004 | STL Starting Point |
| [21] | Robustness of temporal logic specifications | Theoretical Computer Science 2009 | Fundamentals of robustness semantics |
| [22] | Robust satisfaction over real-valued signals | FORMATS 2010 | STL robustness calculation basis |
| [23] | Reactive Synthesis from STL Specifications | HSCC 2015 | STL and control/planning coupling |
| [24] | Diagnosis and Repair for STL Synthesis | HSCC 2016 | specification diagnosis/repair theoretical reference |
| [25] | Spot 2.0 | ATVA 2016 | LTL/omega-automata tool |
| [26] | RTAMT | STTT 2024 / arXiv 2025 | STL robustness monitor |
| [27] | PRISM 4.0 | CAV 2011 | probabilistic model checking tool |
| [28] | Safe RL via Shielding | AAAI 2018 | shield guarantees safe classics will work |
| [29] | Probabilistic Shielding | AAAI 2025 | Probabilistic Security Assurance and Shielding |
| [30] | AerialVLN | ICCV 2023 | UAV Visual Language Navigation Benchmark |
| [31] | Realistic UAV-VLN | ICLR 2025 | More realistic UAV-VLN platforms, benchmarks and methods |
| [32] | ASMA | RA-L/arXiv 2024 | CBF Security Constraints Reference in UAV-VLN |
| [33] | LogisticsVLN | arXiv 2025 | Low-altitude delivery language navigation application scenario |
| [34] | UAV-VLN Survey | arXiv 2026 | UAV-VLN Research Roadmap and Challenges |
| [35] | Qwen3 Technical Report | arXiv 2025 | Basis for local open source model selection |
| [36] | DeepSeek-R1 | arXiv 2025 | Basis for selection of inferential open source models |
| [37] | vLLM/PagedAttention | SOSP 2023 | Multi-model local inference implementation basis |### 3.2 Key gaps in existing work

Lang2LTL, NL2LTL, LTLCodeGen and ConformalNL2LTL jointly demonstrate that NL-to-LTL is no longer a blank direction [1-4]. Therefore, Paper E cannot just claim "we translate natural language to LTL". The real potential points of difference are:

1. **Extension from translation correctness to execution correctness**: LTLCodeGen already handles syntax correctness and path generation [3], but the UAV's altitude, speed, obstacle distance and time window require STL robustness, not just LTL formula validity.
2. **Expand from single generation to verification and error correction closed loop**: T3 Planner, LogicGuard, SENTINEL and Pro2Guard explain that formal feedback is becoming a hot spot for embodied LLM safety [6-9]. VERA-UAV needs to more explicitly treat counterexamples, unsat core, and robustness trace as repair signals.
3. **Extension from LLM heuristic to relatively complete algorithms**: LLM self-healing itself is not provably complete; completeness must come from limited DSL, decidable verifiers and symbolic enumeration fallbacks, not from the model "might think right".
4. **Extension from terrestrial navigation to low-altitude UAVs**: AerialVLN and ICLR 2025’s realistic UAV-VLN work emphasizes the differences between UAVs and terrestrial VLNs: three-dimensional motion, continuous dynamics, airspace safety, and resource constraints [30,31]. This is exactly the motivation behind VERA-UAV’s use of STL.

### 3.3 Submission and journal extension constraints

The official description of AAAI-26 Main Technical Track requires the main text to have up to 7 pages of technical content, and requires the author to complete a reproducibility checklist [38]. Therefore, the AAAI version must focus on methods, core experiments, and reproducibility, and cannot expand too much system engineering content.The scope of T-ITS covers sensing, communications, controls, planning, design and implementation in modern transportation systems, as well as methodological directions such as Artificial Intelligence, and requires journal expansion to have clear new contributions relative to conference papers [39]. Therefore, subsequent ITS journal editions should add urban low-altitude transportation system metrics such as airspace utilization, mission throughput, multi-UAV coordination, communication latency, and operational safety gains.

---

## 4. Proposed algorithm: VERA-UAV

### 4.1 Overall process

The full name of VERA-UAV is tentatively determined as:

**VERA-UAV: Verification-Enhanced Repair for Autonomous UAV Language Planning**

The system process is as follows:

```text
Natural-language UAV instruction
        ↓
Local open-source LLM
        ↓
Typed TaskIR
        ↓
TaskIR-to-LTL/STL compiler
        ↓
Spot / RTAMT / optional PRISM verification
        ↓
Counterexample + unsat core + robustness feedback
        ↓
LLM repair + symbolic enumerative fallback
        ↓
A* / RRT* / MPC-lite trajectory generation
        ↓
Final trajectory verification
        ↓
Executable trajectory or UNSAT / NEED_CLARIFICATION
```

Compared with v1, the key change in v2 is the addition of **symbolic enumerative fallback**: LLM is still the main candidate generator, but when LLM fails in multiple rounds of repairs, the system will enumerate candidate repairs within a limited TaskIR DSL. This design is the basis for the subsequent "relative completeness" proof.

### 4.2 Typed TaskIR

TaskIR is a structured interface between natural language and formal logic. It avoids having LLM output arbitrary LTL/STL strings directly, thereby reducing syntax errors and entity grounding errors.

The TaskIR field is designed as follows:| Field | Meaning | Example |
|------|------|------|
| `entities` | Objects involved in the directive | `building_3`, `hospital_zone`, `landing_pad_A` |
| `goals` | Goals to be achieved | `reach(landing_pad_A)` |
| `avoid` | Areas that must be avoided | `avoid(hospital_zone)` |
| `sequence` | Sub-target sequence | `inspect(B3_east) -> land(A)` |
| `metric_bounds` | Continuous constraints | `altitude in [20,120]`, `distance_to_obstacle >= 10` |
| `time_windows` | Time window | `reach(A) within 30s` |
| `fallbacks` | Alternative strategies | `if blocked, reach nearest_safe_pad` |
| `uncertainty` | Ambiguous or missing fields | `NEED_CLARIFICATION(target="safe place")` |

### 4.3 TaskIR to LTL/STL compilation

LTL is used to express discrete timing structures:

$$
\varphi_{\text{LTL}} =
G(\neg collision) \wedge F(reach(goal)) \wedge G(\neg enter(no\_fly\_zone))
$$

STL is used to express continuous signal constraints:

$$
\varphi_{\text{STL}} =
G_{[0,T]}(d_{\text{obs}}(t) \ge d_{\min})
\wedge
G_{[0,T]}(h_{\min} \le h(t) \le h_{\max})
\wedge
F_{[0,30]}(reach(goal))
$$

Where $d_{\text{obs}}(t)$ is the distance from the UAV to the nearest obstacle, and $h(t)$ is the flight altitude. RTAMT or equivalent STL monitor output robustness:$$
\rho(\tau, \varphi_{\text{STL}}) > 0
$$

Indicates that the trajectory $\tau$ satisfies the specification; if $\rho \le 0$, the verifier returns the violation clause, the violation time, and the minimum safety margin.

### 4.4 Counterexample driver repair

Instead of just returning `pass/fail`, the validator returns a structured diagnostic:

```json
{
  "status": "FAILED",
  "stage": "STL_ROBUSTNESS",
  "violated_clause": "G[0,T](distance_to_obstacle >= 10)",
  "counterexample_trace": [
    {"t": 14.2, "x": 38, "y": 51, "z": 30, "distance_to_obstacle": 6.4}
  ],
  "robustness": -3.6,
  "repair_hint": "Increase safety margin or route around building_7 west side."
}
```

LLM's repair prompt does not require free play, but requires it to only modify relevant fields in TaskIR:

```text
你生成的 TaskIR 在 STL 验证中失败。
失败子句：G[0,T](distance_to_obstacle >= 10)
反例：t=14.2s 时距离 building_7 仅 6.4m。
请只修改 route constraint 或 safety margin，不要改变用户原始目标。
输出新的 TaskIR JSON。
```

The focus of this design is to reduce the search space of LLM and make the repair behavior explainable, recordable, and reproducible.

If LLM repair fails after consecutive $K_{\mathrm{LLM}}$ rounds, the symbol enumeration fallback is entered. The enumeration scope is bounded by the TaskIR DSL depth, map entity set, allowed constraint template, and maximum task horizon. The enumerator prioritizes the expansion of the most relevant fields based on diagnostic results, such as safe distance, detour side, time window, target sequence, and fallback landing pad.

### 4.5 Trajectory generation

The AAAI version uses a lightweight reproducible trajectory generator:

- 2D grid A*: for basic reach-avoid and sequential tasks.
- 3D grid A*: used for height levels and urban low-altitude corridors.
- RRT*: for continuous spatial supplementary verification.
- MPC-lite/trajectory smoothing: used to check whether turning radius, speed change and height change satisfy simplified dynamics constraints.

The trajectory generator is not the innovation of this article. Its function is to advance the specification translation problem to the level of "whether the executable track really exists".

---

## 5. Proof of theoretical properties and relative completeness

v1 only says "verification error correction can improve reliability", but there is no mathematical boundary. v2 makes the algorithmic properties clear: VERA-UAV does not claim that LLM itself is complete, but rather claims to have **relative completeness** under the assumptions of a finite DSL, a decidable verifier, and a complete underlying planner.

### 5.1 Formal setting

Discretize the urban low-altitude map into a limited weighted map:

$$
G=(V,E,w), \quad |V|<\infty, \quad |E|<\infty.
$$Each node $v\in V$ carries a set of atomic propositions $L(v)$, such as `goal_A`, `building_7_margin`, `no_fly_zone`, `altitude_layer_3`. Trajectories are finite sequences:

$$
\tau = (v_0, v_1, \ldots, v_T), \quad (v_t,v_{t+1})\in E.
$$

TaskIR DSL is defined as a limited syntax:

$$
\mathcal{D}_{H,D} = \{\psi: \mathrm{depth}(\psi)\le D,\ \mathrm{horizon}(\psi)\le H,\ \mathrm{entities}(\psi)\subseteq \mathcal{E}(\mathcal{M})\}.
$$

Compiler $C$ compiles TaskIR to LTL/STL specification:

$$
C(\psi)=(\varphi_{\mathrm{LTL}},\varphi_{\mathrm{STL}}).
$$

Verifier $V$ determines whether candidate trajectories meet specifications:

$$
V(\tau, C(\psi)) =
\begin{cases}
\mathrm{PASS}, & \tau \models \varphi_{\mathrm{LTL}}\ \land\ \rho(\tau,\varphi_{\mathrm{STL}})>0,\\
\mathrm{FAIL}(\eta), & \text{otherwise},
\end{cases}
$$

where $\eta$ is a counterexample, unsat core, or robustness trace.

### 5.2 Algorithm pseudocode

```text
Algorithm VERA-UAV
Input: natural language x_NL, map M, initial state s0
Output: verified trajectory tau or UNSAT / NEED_CLARIFICATION

1: Q ← LLM_PROPOSE(x_NL, M)
2: Q ← TYPECHECK_AND_RANK(Q)
3: Visited ← ∅
4: for iter = 1 ... B do
5:     if Q has no unvisited candidate:
6:         Q ← Q ∪ SYMBOLIC_ENUMERATE_NEXT(D, H)
7:         if Q still has no unvisited candidate:
8:             return UNSAT
9:     ψ ← POP_UNVISITED(Q, Visited)
10:    Visited ← Visited ∪ {ψ}
11:    if ψ has missing entity or underspecified field:
12:        η ← type / grounding diagnostic
13:        Q ← Q ∪ REPAIR(ψ, η)
14:        if all remaining candidates require the same external information:
15:            return NEED_CLARIFICATION
16:        continue
17:    (φ_LTL, φ_STL) ← COMPILE(ψ)
18:    if compiler or syntax verifier fails:
19:        η ← compiler diagnostic
20:        Q ← Q ∪ REPAIR(ψ, η)
21:        continue
22:    τ ← COMPLETE_PLANNER(G, s0, φ_LTL, φ_STL)
23:    if τ exists and VERIFY(τ, φ_LTL, φ_STL) = PASS:
24:        return τ
25:    η ← counterexample / unsat core / robustness trace
26:    Q ← Q ∪ LLM_REPAIR(ψ, η)
27:    if LLM repair budget exhausted:
28:        Q ← Q ∪ SYMBOLIC_ENUMERATE(ψ, η, D, H)
29: return UNSAT
```

### 5.3 Theorem 1: Terminability

**Theorem 1 (Termination).** If the TaskIR DSL $\mathcal{D}_{H,D}$ is finite and the algorithm sets a finite candidate budget $B$, then VERA-UAV must return verified trajectory, `UNSAT` or `NEED_CLARIFICATION` in finite steps.**Proof sketch.** Each time an unvisited candidate TaskIR pops up in the queue $Q$, and is used to avoid repeated expansion through `Visited`. The maximum number of rounds of LLM repair is limited, the symbol enumeration space $\mathcal{D}_{H,D}$ is limited, and the outer loop can be executed at most $B$ times. Therefore the algorithm cannot run infinitely. Each branch either returns or enters the next finite loop. Certification completed.

### 5.4 Theorem 2: Safety and reliability

**Theorem 2 (Soundness).** If VERA-UAV returns a trajectory $\tau$, then given the map model, monitor semantics, and trajectory discretization accuracy, $\tau$ satisfies the compiled LTL/STL specification:

$$
\tau \models \varphi_{\mathrm{LTL}}
\quad \text{and} \quad
\rho(\tau,\varphi_{\mathrm{STL}})>0.
$$

**Proof sketch.** The algorithm returns the trajectory only after passing final verification on line 23. The final verification consists of LTL layer verification and STL robustness check. If any check fails, the algorithm simply generates a diagnosis and continues repair without returning to the trajectory. Therefore all return trajectories satisfy the above conditions. Certification completed.

### 5.5 Theorem 3: Relative completeness

**Theorem 3 (Relative Completeness).** For task instances that do not require external clarification, assume:

1. There is an equivalent or sufficiently fidelity TaskIR $\psi^\star \in \mathcal{D}_{H,D}$ for the user’s intention;
2. Compiler $C$ can generate semantically preserved LTL/STL specifications for all TaskIRs in $\mathcal{D}_{H,D}$;
3. The underlying planner is complete in searching for trajectories satisfying $C(\psi)$ on the finite graph $G$;
4. The symbolic enumerator will enumerate all candidates in $\mathcal{D}_{H,D}$ within a limited time;
5. The final validator is reliable for bounded LTL/STL semantics.If there is a trajectory $\tau^\star$ that satisfies $C(\psi^\star)$, then when the candidate budget $B \ge |\mathcal{D}_{H,D}|$ is, VERA-UAV will eventually return a trajectory $\tau$ that satisfies the specification.

**Proof sketch.** According to Assumption 4, the symbolic enumeration fallback will enumerate to $\psi^\star$. By Assumption 2, $C(\psi^\star)$ remains semantic. According to Assumption 3, the underlying planner will find trajectories that satisfy $C(\psi^\star)$. According to Assumption 5, the final validator will accept this trajectory. The algorithm returns this trajectory according to lines 23-24 of the algorithm. VERA-UAV is therefore relatively complete under this limited DSL and model assumptions. Certification completed.

### 5.6 Completeness Boundary

This theorem does not mean that VERA-UAV is absolutely complete for any natural language and any continuous dynamics in the real world. It only states: **As long as the target task can be represented by a limited TaskIR DSL, and the underlying search space and verification semantics cover the task, VERA-UAV will inevitably generate the correct answer without relying on LLM, and can also find a feasible solution through the symbolic fallback. **

This is also the key theoretical positioning of this article relative to Lang2LTL, LTLCodeGen, and T3 Planner: LLM is an efficient proposal generator, not a source of completeness.

---

## 6. Data sources and data set construction

### 6.1 Master data source

The AAAI main experiment uses procedural generation of urban UAV grid/world data and does not rely on AirSim or real flight data. There are three reasons for doing this:

1. Controllable: can systematically generate tasks such as reachable, unreachable, ambiguous, conflicting, and tight time windows.
2. Reproducible: Maps, tasks and random seeds can be completely open source.
3. Adapt to the length of AAAI: focus on serving AI method evaluation rather than heavy-duty simulation engineering.

### 6.2 Map generation

Each map contains:- Grid size: `50x50x3` to `100x100x5`.
- Semantic objects: buildings, hospitals, schools, logistics stations, landing points, inspection surfaces, no-fly zones.
- Airspace structure: levels, flight corridors, temporary closed areas.
- Dynamic elements: optionally adding moving obstacles or temporary no-fly zones.
- OSM style naming: such as `hospital_zone_2`, `building_7_east_face`, are only used as a semantic naming reference and are not relied upon by the main experiment.

### 6.3 Sample fields

Each sample contains:

| Field | Description |
|------|------|
| `instruction_id` | Sample number |
| `map_id` | Map number |
| `natural_language_instruction` | Natural language UAV tasks |
| `entity_annotations` | Map entities are aligned with directive entities |
| `gold_task_ir` | The gold standard for manual or rule generation TaskIR |
| `gold_ltl` | Gold Standard LTL |
| `gold_stl` | The gold standard STL |
| `satisfiability_label` | `SAT`, `UNSAT`, `NEED_CLARIFICATION` |
| `reference_trajectory` | If SAT, give a feasible trajectory |
| `failure_type` | If it fails, mark the failure type |
| `oracle_cost` | Shortest path or minimum cost trajectory cost |

### 6.4 Data scale

v2 recommended AAAI main experiment scale:

| Split | Quantity | Purpose |
|------|------|------|
| Train / prompt pool | 800 | few-shot examples, template debugging |
| Dev | 250 | prompt, repair strategy, threshold selection |
| Test | 400 | Final Report |
| Stress test | 150 | Long combination, fuzzy, unsatisfiable, tight time window |

The test set cannot participate in prompt selection. All lab reports have fixed random seeds and task lists.

### 6.5 Data generation protocol and leakage preventionIn order for synthetic benchmarks to stand up to AAAI reviewers, data generation must be managed from day one as "reproducible benchmarks" rather than "ad hoc experimental scripts":

1. **Freeze the generator first, then generate the test**: The map generator, task template, language paraphrase rules and failure injection rules are debugged on dev first, freeze the commit hash and then generate the test / stress test.
2. **Split by map level**: The test map cannot share the same `map_id`, entity coordinates or obstacle layout with train/dev. Only abstract task types are allowed to be shared.
3. **Split by entity naming level**: At least 30% of the tasks in the test use entity naming patterns that have not appeared in the training set, such as `clinic_zone`, `sky_corridor_E2`, `temporary_pad_17`.
4. **Split by template combination level**: Keep some unseen combinations in the test, such as `ordered inspection + time window + emergency fallback`, to prevent the model from only remembering a single template mapping.
5. **Fixed random seed and manifest**: Each split outputs `manifest.jsonl`, recording generator version, seed, map hash, task template id, paraphrase id and satisfiability label.
6. **Prohibit test prompt pollution**: few-shot examples can only come from train/prompt pool; dev is only used for threshold and prompt strategy selection; test and stress test are only run once and the results are locked.

### 6.6 Failure taxonomy

Each failure sample should record the first failure stage and the final failure stage to facilitate explanation of what VERA-UAV fixed:| Failure type | Definition | Main Attribution Module |
|--------------|------|--------------|
| `syntax_error` | LTL/STL cannot be parsed or type mismatch | LLM/compiler |
| `entity_error` | Reference to a non-existent, ambiguous or mismatched map entity | grounding |
| `semantic_miss` | Missing key user constraints, such as no-fly zones or time windows | TaskIR generation |
| `unsat_missed` | gold is UNSAT, but the system returns an executable plan | verifier / decision policy |
| `false_unsat` | gold is SAT, but the system error outputs UNSAT | planner / search budget |
| `ltl_violation` | Discrete timing sequence, arrival and avoidance are not satisfied | planner / LTL compiler |
| `stl_violation` | height, distance, speed, time window robustness non-positive | trajectory / STL monitor |
| `repair_regression` | Repair a constraint and then destroy the originally satisfied constraints | repair loop |
| `timeout` | Exceeded preset inference or planning budget | system budget |

Not only the average score is reported in the final paper, but also a stacked histogram of the failure taxonomy. In this way, even if the overall improvement is not large enough, it can still prove that the method has a clear effect on critical safety failure types.

---

## 7. Experimental platform and implementation configuration

### 7.1 Hardware

Currently designed with 4 RTX 4090s and 24GB of video memory each. This study does not rely on closed source APIs, and the main experiments all use local open source models.

### 7.2 Model

Main experimental model:

- Qwen3-8B: lightweight local model baseline [35].
- Qwen3-14B: master model [35].
- DeepSeek-R1-Distill-Qwen-14B: Inference enhanced model [36].

Optional capped models:- 32B quantitative model, used as appendix or supplementary results; not required as a requirement for AAAI main conclusions.

Local inference uses vLLM/PagedAttention or HuggingFace Transformers. The PagedAttention design of vLLM is suitable for throughput experiments under multiple prompts and multiple repair rounds [37].

### 7.3 Software modules

| Module | Candidate Tool | Function |
|------|----------|------|
| LLM inference | vLLM/transformers | Local model inference |
| LTL validation | Spot | LTL parsing, automata, satisfiability analysis |
| STL monitoring | RTAMT or self-implemented monitor | STL robustness |
| Probabilistic checking | PRISM | Optional uncertain environment verification |
| Planning | A* / RRT* / MPC-lite | Trajectory generation |
| Logging | JSONL + CSV | Logging every round of build, verification and repair |

### 7.4 Operation record

Each task instance must record:

- original instruction.
- TaskIR, LTL, STL per round.
- Validator output.
- Fix prompt.
- Final trajectory.
- Running time, number of tokens, and graphics memory configuration.
- Paired comparison id of baseline and VERA-UAV on the same task.

These records serve the AAAI reproducibility checklist [38].

### 7.5 Pre-registration parameter budget

In order to avoid adjusting parameters after the experiment, this task document recommends fixing the following budget before the first round of formal testing:| Parameters | Recommended values | Description |
|------|--------|------|
| `K_LLM` | 3 | Up to three rounds of LLM repair per task |
| `B` | 256 | VERA-UAV Total Candidate TaskIR Budget |
| `D` | 4 | TaskIR DSL maximum nesting depth |
| `H` | 8 | Discrete task horizon / sub-goal upper limit |
| `T_plan` | 30s | Single task planning timeout |
| `T_llm` | 20s | Single round LLM inference timeout |
| decoding temperature | 0.2 | Low randomness in main experiment; only report temperature sensitivity in appendix |
| top-p | 0.9 | Co-fixed with temperature |
| max new tokens | 1024 | Prevent the difference in output lengths of different models from affecting runtime |

If these values need to be changed for formal experiments, the reasons must be recorded on dev first, and then the configuration must be refrozen. The test results cannot determine parameters inversely.

---

## 8. Comparative Experimental Design

### 8.1 Baseline list| Method | Description | Purpose |
|------|------|------|
| Direct LLM planning | LLM direct output waypoint / action sequence | Check whether plain text planning is unsafe |
| ReAct-style planning | reasoning-action loop, no formal verification | compared with general LLM agent planning [14] |
| SayCan-style affordance filtering | LLM score + feasible skill filter | Compare affordance grounding [15] |
| Prompt-only NL-to-LTL/STL | LLM outputs LTL/STL directly, without typed IR and verification fixes | Check prompt Project upper limit |
| NL2LTL-style template baseline | Generate LTL based on template matching | Compared with traditional template method [2] |
| LTLCodeGen-style baseline | LLM generates logic function code and then compiles it into LTL | Check the syntax correctness route [3] |
| T3-style self-correction | LLM + STL verifier, multiple rounds of self-correction | Compared with the recent direct competition route [6] |
| VERA-UAV without repair | Use TaskIR and verify, but do not repair after failure | Separate verification and repair contributions |
| VERA-UAV LLM-only repair | typed IR + LLM repair, unsigned enum fallback | Verify fallback's contribution to completeness |
| VERA-UAV full | full typed IR + verification + counterexample repair + symbolic fallback | main method |

### 8.2 Main experiment

The main experiment answers five questions:1. Is VERA-UAV easier to generate executable plans than baseline?
2. Does VERA-UAV reduce security breach rates?
3. Is the STL robustness of VERA-UAV significantly higher?
4. Are the number of repair rounds and extra inference overhead of VERA-UAV acceptable?
5. Does symbolic enumeration fallback really improve the "failed task recovery rate" and relative completeness?

Main results table suggestions:| Method | Syntax valid ↑ | Semantic F1 ↑ | ESS ↑ | FSR ↓ | Mean robustness ↑ | Optimality gap ↓ | Repair success ↑ | Runtime ↓ |
|--------|----------------|---------------|-------|-------|-------------------|------------------|------------------|-----------|
| Direct LLM | TBD | TBD | TBD | TBD | TBD | TBD | N/A | TBD |
| ReAct-style | TBD | TBD | TBD | TBD | TBD | TBD | N/A | TBD |
| SayCan-style | TBD | TBD | TBD | TBD | TBD | TBD | N/A | TBD |
| Prompt-only | TBD | TBD | TBD | TBD | TBD | TBD | N/A | TBD |
| NL2LTL-style | TBD | TBD | TBD | TBD | TBD | TBD | N/A | TBD |
| LTLCodeGen-style | TBD | TBD | TBD | TBD | TBD | TBD | N/A | TBD |
| T3-style | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| VERA-UAV no repair | TBD | TBD | TBD | TBD | TBD | TBD | 0 | TBD |
| VERA-UAV LLM-only repair | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| VERA-UAV full | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |`TBD` in the table is the data to be filled in for the experiment and must not be forged in the assignment letter.

### 8.3 Experimental results evaluation protocol

v2 clarifies the main indicators and statistical judgments to avoid the subsequent risk of "reporting whichever indicator is good".

**Primary metric 1: Executable Safety Success (ESS)**

A task is counted as ESS=1 only if it simultaneously meets the following conditions:

- Generated TaskIR has no type error.
- LTL/STL compilable.
- planner finds trajectories.
- Final trajectory passes LTL check.
- STL robustness is positive.
- No collisions, no-fly zone entries, altitude violations or time window failures.

**Primary metric 2: False Safe Rate (FSR)**

FSR measures the proportion of unsafe or unsatisfactory tasks that the system misjudges as safely executable:

$$
\mathrm{FSR} = \frac{\#\{\mathrm{unsafe\ but\ returned\ as\ executable}\}}{\#\{\mathrm{all\ returned\ executable}\}}.
$$

In the AAAI paper, FSR should be regarded as the most critical negative indicator in the direction of security. The main selling point of VERA-UAV is not to have "output" for all tasks, but to avoid false security.

**Statistical Test**

- For binary indicators such as ESS, FSR, and UNSAT detection, use paired McNemar test.
- For continuous indicators such as robustness, optimality gap, runtime, etc., use paired bootstrap 95% CI and Wilcoxon signed-rank test.
- Multiple baseline comparisons use Holm-Bonferroni correction.
- Conclusions are only written into the main text when $p<0.05$ and the effect size reaches the pre-registration threshold.

**Success Criteria**

The minimum conditions for the establishment of AAAI’s main conclusion:1. The ESS of VERA-UAV full is significantly higher than that of LTLCodeGen-style and T3-style baseline.
2. The FSR of VERA-UAV full is significantly lower than all LLM-only baselines.
3. After removing STL robustness feedback, failures related to continuous safety constraints increase significantly.
4. Symbolic fallback provides measurable gains in LLM repair failure samples.

### 8.4 Generalization experiment

Generalization dimension:

- No map seen.
- No entity name seen.
- Natural language paraphrase.
- Longer timing combinations.
- Tighter time window.
- Unsatisfied task ratio increase.

Generalization experiments focus on reporting whether VERA-UAV can identify unsatisfiable or ambiguous tasks, rather than outputting error trajectories.

### 8.5 Case study

Prepare at least three visualization cases:

1. **Syntax repair case**: LLM output is illegal STL, Spot/RTAMT reports an error, system repair.
2. **Trajectory safety case**: LTL is satisfied but STL robustness is negative, and the system turns positive after detouring.
3. **Unsatisfiable case**: User requirements are conflicting, and the system outputs `UNSAT`.

### 8.6 AAAI Main Text Chart Plan

AAAI's main text space is very tight, and the charts must serve the core argument. It is recommended that only five types of charts be included in the main text, and appendix is ​​used for the others:| Diagram | Target | Placement |
|------|------|----------|
| Figure 1: VERA-UAV pipeline | A glance at the closed loop of typed IR, verification, repair, and fallback | Method |
| Table 1: Core literature positioning matrix | Proves that this article is not a simple NL-to-LTL application | Related Work |
| Table 2: Main experiment results | paired comparison of ESS, FSR, robustness, runtime | Experiments |
| Figure 2: failure taxonomy stacked chart | illustrates which failure types the method mainly reduces | Experiments |
| Figure 3: Case study trajectory | Shows how counterexample feedback can correct negative robustness to positive | Experiments / Appendix |

It is not recommended to enlarge the prompt section, the complete DSL grammar or all map screenshots in the main article. These contents should be placed in the code/data appendix so as not to crowd out the contribution argument.

---

## 9. Ablation Experiment Design| Ablation | Variant | Purpose |
|--------|------|------|
| Remove typed IR | Direct LTL/STL generation | Verify whether structured intermediate representation improves reliability |
| Remove counterexample feedback | Generic retry | Verify whether counterexample is more effective than normal retry |
| Remove STL robustness feedback | LTL-only verification | The importance of verifying continuous safety constraints |
| one-shot repair | Repair at most 1 time | Evaluate the benefits of repair rounds |
| iterative repair | Repair up to 3 times | Evaluate the upper limit of multiple rounds of repair |
| Different model sizes | Qwen3-8B / Qwen3-14B / DeepSeek-R1-Distill-Qwen-14B | Evaluate the relationship between model capability and verification framework |
| Remove UNSAT detection | Force trace generation | Verify the contribution of denial-of-answer capability to security |
| Remove symbol fallback | LLM-only repair | Verify the contribution of relative completeness components to failure recovery |
| Remove planner final verification | Only verify formulas but not trajectories | Prove execution of closed loop is not optional |

The core of the ablation experiment is not to "prove that the components are effective", but to find out which components contribute the most to the safety and performability indicators that AAAI reviewers are most concerned about.

---

## 10. Evaluation indicators

### 10.1 Specification generation indicators| Indicators | Definition |
|------|------|
| Syntax validity | Is LTL/STL acceptable to the parser |
| Entity grounding accuracy | Whether the command entity is correctly mapped to the map entity |
| Semantic F1 | Generate precision / recall / F1 of TaskIR field and gold TaskIR |
| Semantic match | Whether the generated specification is equivalent or approximately equivalent to gold TaskIR / gold formula |
| UNSAT detection accuracy | Whether the unsatisfiable task is correctly identified |
| Clarification accuracy | Whether the fuzzy task triggers `NEED_CLARIFICATION` |
| False executable rate | The proportion of unsatisfiable or ambiguous tasks that are incorrectly executed |

### 10.2 Planning execution indicators

| Indicators | Definition |
|------|------|
| ESS | Proportion of tasks that simultaneously satisfy semantics, feasible trajectories, LTL, STL, and safety constraints |
| FSR | Proportion of unsafe tasks incorrectly marked as safe to execute |
|Mean STL robustness |The average robustness of the final trajectory against the STL specification |
| Worst-case STL robustness | Distribution of minimum robustness per trajectory |
| Minimum safety margin | Minimum obstacle distance in trajectory |
| Optimality gap | $(J(\tau)-J^\star)/J^\star$ |
| Path length / flight time | Trajectory cost and flight time |

### 10.3 Repair efficiency indicator| Indicators | Definition |
|------|------|
| Repair success rate | Repair success rate after verification failure |
| Fail-to-pass conversion | The proportion of initial failed samples that pass after being repaired |
| Average repair rounds | Average repair rounds |
| Fallback contribution | Proportion of LLM repair failure but symbolic fallback success |
| Runtime overhead | Extra time caused by repair mechanism |
| Token overhead | Fix the token increment caused by prompt and diagnosis |

### 10.4 Indicator calculation details

The main experiment needs to implement the following indicators directly in the code to avoid manual arrangement during the paper writing stage:

**Semantic F1**

Flatten TaskIR into a set of field-level constraints $\mathcal{C}$, such as `reach(A)`, `avoid(zone_B)`, `time_window(A,30)`. Let the prediction set be $\hat{\mathcal{C}}$ and the gold standard set be $\mathcal{C}^\star$:

$$
P = \frac{|\hat{\mathcal{C}}\cap \mathcal{C}^\star|}{|\hat{\mathcal{C}}|}, \quad
R = \frac{|\hat{\mathcal{C}}\cap \mathcal{C}^\star|}{|\mathcal{C}^\star|}, \quad
F1 = \frac{2PR}{P+R}.
$$

**Safety violation rate**

$$
\mathrm{SVR} =
\frac{\#\{\tau: collision \lor nofly \lor altitude\_violation \lor \rho(\tau,\varphi_{\mathrm{STL}})\le 0\}}
{\#\{\mathrm{returned\ trajectories}\}}.
$$

**Optimality gap**When gold or oracle planner can give the optimal cost $J^\star$:

$$
\mathrm{Gap}(\tau)=\frac{J(\tau)-J^\star}{\max(J^\star,\epsilon)}.
$$

If the task is UNSAT or NEED_CLARIFICATION, the optimality gap is not calculated and is counted separately in the recognition accuracy.

**Repair efficiency**

$$
\mathrm{FailToPass} =
\frac{\#\{\mathrm{initial\ fail,\ final\ pass}\}}
{\#\{\mathrm{initial\ fail}\}},
\quad
\mathrm{FallbackContribution} =
\frac{\#\{\mathrm{LLM\ repair\ fail,\ symbolic\ fallback\ pass}\}}
{\#\{\mathrm{final\ pass}\}}.
$$

These formulas should be output as machine-readable CSV fields in the experiment script and only formatted in the paper table.

---

## 11. Expected experimental conclusions

This section is pre-registration expectations, not experimental results.

### 11.1 Main expectations

VERA-UAV full is expected to be higher than all baselines on ESS and lower on FSR/safety violation rate. The reason is that baseline usually only optimizes the local correctness of the language to the specification, while VERA-UAV incorporates "whether the specification can produce a safe trajectory" into the closed loop.

### 11.2 Counterexample feedback expectations

Counterexample feedback is expected to significantly reduce the proportion of unexecutable plans. Compared with generic retry, structured counterexamples can tell LLM which clause, which moment, and which entity caused the failure, thereby reducing undirected retries.

### 11.3 Typed IR expectations

Typed IR is expected to improve semantic consistency and interpretability. Generating LTL/STL directly is prone to missing parentheses, operators, entity references, and constraints; TaskIR exposes these errors as missing fields or type errors in advance.

### 11.4 STL robustness expectedSTL robustness feedback is expected to be most critical for continuous safety constraints. The LTL layer can prove discrete properties such as "final arrival" and "avoid no-fly zone", but cannot fully express flight altitude, minimum distance and time window margin. STL robustness can provide quantified safety boundaries and is the key point that distinguishes UAV from ordinary ground navigation tasks.

### 11.5 Expected model size

Stronger local models are expected to improve initial TaskIR quality, but the validation repair framework is also helpful for smaller models. In other words, this article should not write the contribution as "a certain large model is stronger", but should write it as "the verification error correction mechanism improves the reliability of different open source models."

---

## 12. Issues discovered during self-audit and v2 fixes

### Main issues with 12.1 v1

1. **Insufficient literature coverage**: v1 only lists 12 references, which is not enough to support the positioning of AAAI.
2. **The novelty boundary is not sharp enough**: v1 is easily understood as "UAV version NL-to-LTL", and the difference from Lang2LTL and LTLCodeGen is not strong enough.
3. **Experimental indicators are not judged enough**: v1 only lists general indicators and does not define ESS, FSR, statistical tests and success criteria.
4. **The completeness statement is too weak**: v1 does not explain why the algorithm is not pure heuristic.
5. **Synthetic data risk not adequately mitigated**: v1 does not explain why synthetic data still supports the AAAI methodological conclusions.

### Repair strategy for 12.2 v2

1. Expand to 30+ strongly relevant documents, and use a literature matrix to clarify the relationship between each article and this article.
2. Narrow the contribution from "translation" to "execution closed loop + STL robustness + counterexample repair + relatively complete fallback".
3. Define reproducible indicators such as ESS, FSR, optimality gap, fail-to-pass conversion, etc.
4. Give the theorem of termination, safety, reliability and relative completeness, and make it clear that completeness comes from finite DSL and symbolic enumeration, not LLM.
5. Put AirSim/real logistics into T-ITS extension, and the AAAI main article adheres to the methodological positioning of synthetic controlled benchmark.

### 12.3 2026-05-19 Second self-examination and reinforcementAfter continuing the review in this round, it is believed that Paper E still has four issues that are easy for reviewers to ask, and the corresponding constraints have been added to the task book:

1. **Data Credibility**: Just saying "program-generated data" is not enough. It is necessary to clarify generator freeze, map-level segmentation, entity naming-level segmentation and test prompt pollution prevention.
2. **Failure Explanatory Power**: Only reporting ESS/FSR is not enough. Failure taxonomy needs to be recorded to prove that the method reduces safety-related failures rather than just improving the average score.
3. **Reproducible Parameters**: Just using Qwen3 / DeepSeek is not enough. You need to fix the number of repair rounds, candidate budget, DSL depth, planning timeout and decoding parameters.
4. **Paper presentation strategy**: AAAI has limited space, so you need to determine the main text diagram in advance, otherwise it will be easy to scatter the main line.

These four points do not change the core contribution of VERA-UAV, but they can advance the mission statement from an "idea route" to a state where "experiments and papers can be directly organized".

### 12.4 2026-05-23 Finishing: AAAI main line wrapping up

Paper E should be prioritized as a **AAAI / IJCAI method paper**, rather than writing a complete ITS system paper in advance. The core issue is: how the UAV mission plan generated by LLM can be turned into an executable, verifiable, and interpretable trajectory plan through typed IR, temporal logic verification, counterexample repair, and symbolic fallback.

The first version of the paper only retains three contributions:

1. **Typed TaskIR**: Convert natural language UAV instructions into intermediate representations that can be inspected for entities, actions, timing constraints, security constraints, and resource constraints.
2. **LTL/STL + verifier + trajectory closure**: Not only verifies the formula syntax, but also verifies whether the specification can generate a trajectory that satisfies safety constraints.
3. **Counterexample / robustness repair with finite DSL fallback**: Use counterexample, unsat core and STL robustness feedback to repair; when LLM cannot be repaired, use finite DSL enumeration to give relative completeness.

Do not promise the following in advance in the main article:- Does not do complete multi-UAV traffic management;
- No real logistics system deployment;
- Do not rely on AirSim high-fidelity simulation as the main experiment;
- Do not write ITS policies or low-altitude economic system revelations as AAAI main contributions.

The minimum experimental matrix is recommended to be frozen as:

| Dimensions | First Edition Settings |
|------|------------|
| Task family | patrol, delivery, inspection, avoidance, temporal ordering, UNSAT / ambiguous |
| Map | Procedurally generated city grid / obstacle / no-fly-zone / charging-point |
| baselines | Direct LLM planning, ReAct / prompt-only, NL2LTL-style, LTLCodeGen-style, VERA-UAV no repair, VERA-UAV full |
| Main indicators | ESS, FSR, safety violation rate, repair success, fail-to-pass conversion, runtime |
| Ablation | no typed IR, no counterexample, no STL robustness, one-shot vs iterative repair, no symbolic fallback |
| Generalization | unseen map, unseen entity naming, longer horizon, harder constraints, UNSAT detection |

T-ITS extensions can be placed in subsequent versions: integrating Paper B’s fleet scheduling, Paper F’s stress scenarios and low-altitude traffic system indicators. But the AAAI version must keep the questions clean, otherwise it will be pushed for boundaries by both AI reviewers and traffic reviewers.

---

## 13. Risks and Alternatives| Risk | Impact | Alternatives |
|------|------|----------|
| Novelty is considered only for NL-to-LTL applications | AAAI has high risk of rejection | Emphasis on STL robustness, counterexample repair, and executable trajectory closure |
| LTLCodeGen baseline is too strong | The main result has insufficient advantages | Use UAV continuous constraints and unsatisfiable detection as differentiation indicators |
| Insufficient local model capabilities | Low initial translation quality | Use Qwen3-14B/DeepSeek-R1-Distill-Qwen-14B and report repair gains |
| Dataset is considered too synthetic | Application credibility is insufficient | Add OSM style naming, real city block layout statistics, but do not rely on real flights |
| The number of repair rounds causes the runtime to be too high | Real-time performance is questioned | Report one-shot and up to three rounds of repair, set timeout and fallback |
| STL monitor is complex to implement | Affects progress | Implement discrete-time STL subset first, then connect to RTAMT |
| AAAI lacks space | The story is divergent | The main text only contains methods and core experiments, and ITS plans to expand the appendix |
| AAAI is sensitive to LLM generated text policies | Compliance risks in paper writing | The final submitted text must be manually rewritten and reviewed by the author. The LLM output is only used as an experimental subject or internal writing aid, and the unreviewed generated text is not directly used as the text of the paper [38] |
| Relative completeness is considered to be too strong an assumption | The theoretical contribution is weakened | In the main text, it is clearly written as relative completeness, and limited DSL, bounded horizon, and complete planner are used as theorem assumptions instead of absolute guarantees in the real world |
| The stress test is too difficult, causing the main results to decline | The average indicator is not good-looking | The main test and the stress test are reported separately. The stress test is used to analyze the robust boundary and is not mixed with the main conclusion in the same average value |

---

## 14. References[1] Jason Xinyu Liu, Ziyi Yang, Ifrah Idrees, Sam Liang, Benjamin Schornstein, Stefanie Tellex, and Ankit Shah. “Grounding Complex Natural Language Commands for Temporal Tasks in Unseen Environments.” *Proceedings of The 7th Conference on Robot Learning*, PMLR 229:1084-1110, 2023. URL: <https://proceedings.mlr.press/v229/liu23d.html>

[2] Francesco Fuggitti and Tathagata Chakraborti. "NL2LTL -- a Python Package for Converting Natural Language (NL) Instructions to Linear Temporal Logic (LTL) Formulas." *Proceedings of the AAAI Conference on Artificial Intelligence*, 37(13):16428-16430, 2023. DOI: 10.1609/aaai.v37i13.27068. URL: <https://ojs.aaai.org/index.php/AAAI/article/view/27068>[3] Behrad Rabiei, Mahesh Kumar A. R., Zhirui Dai, Surya L. S. R. Pilla, Qiyue Dong, and Nikolay Atanasov. “LTLCodeGen: Code Generation of Syntactically Correct Temporal Logic for Robot Task Planning.” arXiv:2503.07902, 2025; project page reports IROS 2025. URL: <https://arxiv.org/abs/2503.07902>; <https://existentialrobotics.org/LTLCodeGen/>

[4] Jun Wang, David Smith Sundarsingh, Jyotirmoy V. Deshmukh, and Yiannis Kantaros. “ConformalNL2LTL: Translating Natural Language Instructions into Temporal Logic Formulas with Conformal Correctness Guarantees.” arXiv:2504.21022, 2025. URL: <https://arxiv.org/abs/2504.21022>

[5] Licheng Luo, Kaier Liang, Yu Xia, and Mingyu Cai. "NL2SpaTiaL: Generating Geometric Spatio-Temporal Logic Specifications from Natural Language for Manipulation Tasks." arXiv:2512.13670, 2025; revised 2026. URL: <https://arxiv.org/abs/2512.13670>[6] Jia Li and Guoxiang Zhao. "T3 Planner: A Self-Correcting LLM Framework for Robotic Motion Planning with Temporal Logic." arXiv:2510.16767, 2025. URL: <https://arxiv.org/abs/2510.16767>

[7] Simon Sinong Zhan, Yao Liu, Philip Wang, Zinan Wang, Qineng Wang, Zhian Ruan, Xiangyu Shi, Xinyu Cao, Frank Yang, Kangrui Wang, Huajie Shao, Manling Li, and Qi Zhu. "SENTINEL: A Multi-Level Formal Framework for Safety Evaluation of LLM-based Embodied Agents." arXiv:2510.12985, 2025. URL: <https://arxiv.org/abs/2510.12985>

[8] Anand Gokhale, Vaibhav Srivastava, and Francesco Bullo. “LogicGuard: Improving Embodied LLM agents through Temporal Logic based Critics.” arXiv:2507.03293, 2025. URL: <https://arxiv.org/abs/2507.03293>

[9] Haoyu Wang, Christopher M. Poskitt, Jun Sun, and Jiali Wei. “Pro2Guard: Proactive Runtime Enforcement of LLM Agent Safety via Probabilistic Model Checking.” arXiv:2508.00500, 2025. URL: <https://arxiv.org/abs/2508.00500>[10] Tom Silver, Soham Dan, Kavitha Srinivas, Joshua B. Tenenbaum, Leslie Kaelbling, and Michael Katz. "Generalized Planning in PDDL Domains with Pretrained Large Language Models." *Proceedings of the AAAI Conference on Artificial Intelligence*, 38(18):20256-20264, 2024. DOI: 10.1609/aaai.v38i18.30006. URL: <https://ojs.aaai.org/index.php/AAAI/article/view/30006>

[11] Karthik Valmeekam, Matthew Marquez, Sarath Sreedharan, and Subbarao Kambhampati. “On the Planning Abilities of Large Language Models: A Critical Investigation.” *Advances in Neural Information Processing Systems*, 2023. URL: <https://arxiv.org/abs/2305.15771>

[12] Bo Liu, Yuqian Jiang, Xiaohan Zhang, Qiang Liu, Shiqi Zhang, Joydeep Biswas, and Peter Stone. "LLM+P: Empowering Large Language Models with Optimal Planning Proficiency." arXiv:2304.11477, 2023. URL: <https://arxiv.org/abs/2304.11477>[13] Karthik Valmeekam, Matthew Marquez, Alberto Olmo, Sarath Sreedharan, and Subbarao Kambhampati. “PlanBench: An Extensible Benchmark for Evaluating Large Language Models on Planning and Reasoning about Change.” *Advances in Neural Information Processing Systems, Datasets and Benchmarks Track*, 2023. URL: <https://openreview.net/forum?id=YXogl4uQUO>

[14] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. "ReAct: Synergizing Reasoning and Acting in Language Models." *International Conference on Learning Representations (ICLR)*, 2023. URL: <https://openreview.net/forum?id=WE_vluYUL-X>

[15] Michael Ahn et al. “Do As I Can, Not As I Say: Grounding Language in Robotic Affordances.” *Conference on Robot Learning (CoRL)*, 2022. URL: <https://arxiv.org/abs/2204.01691>[16] Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng. "Code as Policies: Language Model Programs for Embodied Control." *IEEE International Conference on Robotics and Automation (ICRA)*, 2023. URL: <https://arxiv.org/abs/2209.07753>

[17] Ishika Singh, Valts Blukis, Arsalan Mousavian, Ankit Goyal, Danfei Xu, Jonathan Tremblay, Dieter Fox, Jesse Thomason, and Animesh Garg. “ProgPrompt: Generating Situated Robot Task Plans using Large Language Models.” *IEEE International Conference on Robotics and Automation (ICRA)*, 2023; extended version in *Autonomous Robots*, 2023. URL: <https://arxiv.org/abs/2209.11302>

[18] Hadas Kress-Gazit, Georgios E. Fainekos, and George J. Pappas. “Temporal-Logic-Based Reactive Mission and Motion Planning.” *IEEE Transactions on Robotics*, 25(6):1370-1381, 2009. DOI: 10.1109/TRO.2009.2030225.[19] Hadas Kress-Gazit, Morteza Lahijanian, and Vasumathi Raman. “Synthesis for Robots: Guarantees and Feedback for Robot Behavior.” *Annual Review of Control, Robotics, and Autonomous Systems*, 1:211-236, 2018. DOI: 10.1146/annurev-control-060117-105838.

[20] Oded Maler and Dejan Nickovic. “Monitoring Temporal Properties of Continuous Signals.” *FORMATS/FTRTFT*, 2004. DOI: 10.1007/978-3-540-30206-3_12.

[21] Georgios E. Fainekos and George J. Pappas. "Robustness of Temporal Logic Specifications for Continuous-Time Signals." *Theoretical Computer Science*, 410(42):4262-4291, 2009. DOI: 10.1016/j.tcs.2009.06.021.

[22] Alexandre Donze and Oded Maler. “Robust Satisfaction of Temporal Logic over Real-Valued Signals.” *FORMATS*, 2010. DOI: 10.1007/978-3-642-15297-9_12.[23] Vasumathi Raman, Alexandre Donze, Dorsa Sadigh, Richard M. Murray, and Sanjit A. Seshia. “Reactive Synthesis from Signal Temporal Logic Specifications.” *Hybrid Systems: Computation and Control (HSCC)*, 2015. DOI: 10.1145/2728606.2728628.

[24] Shromona Ghosh, Dorsa Sadigh, Pierluigi Nuzzo, Vasumathi Raman, Alexandre Donze, Alberto L. Sangiovanni-Vincentelli, and Sanjit A. Seshia. “Diagnosis and Repair for Synthesis from Signal Temporal Logic Specifications.” *Hybrid Systems: Computation and Control (HSCC)*, 2016. DOI: 10.1145/2883817.2883847.

[25] Alexandre Duret-Lutz, Alexandre Lewkowicz, Amaury Fauchille, Thibault Michaud, Étienne Renault, and Laurent Xu. “Spot 2.0 -- A Framework for LTL and omega-Automata Manipulation.” *Automated Technology for Verification and Analysis (ATVA)*, 2016. URL: <https://spot.lre.epita.fr/>[26] Tomoya Yamaguchi, Bardh Hoxha, and Dejan Nickovic. "RTAMT -- Runtime Robustness Monitors with Application to CPS and Robotics." *International Journal on Software Tools for Technology Transfer*, 26(1):79-99, 2024; arXiv:2501.18608, 2025. DOI: 10.1007/S10009-023-00720-3. URL: <https://arxiv.org/abs/2501.18608>; code: <https://github.com/nickovic/rtamt>

[27] Marta Kwiatkowska, Gethin Norman, and David Parker. “PRISM 4.0: Verification of Probabilistic Real-time Systems.” *Computer Aided Verification (CAV)*, 2011. URL: <https://www.prismmodelchecker.org/bibitem.php?key=KNP11>

[28] Mohammed Alshiekh, Roderick Bloem, Rüdiger Ehlers, Bettina Könighofer, Scott Niekum, and Ufuk Topcu. “Safe Reinforcement Learning via Shielding.” *Proceedings of the AAAI Conference on Artificial Intelligence*, 2018. URL: <https://ojs.aaai.org/index.php/AAAI/article/view/11797>[29] Edwin Hamel-De le Court, Francesco Belardinelli, and Alexander W. Goodall. “Probabilistic Shielding for Safe Reinforcement Learning.” *Proceedings of the AAAI Conference on Artificial Intelligence*, 39(15):16091-16099, 2025. DOI: 10.1609/aaai.v39i15.33767. URL: <https://ojs.aaai.org/index.php/AAAI/article/view/33767>

[30] Shubo Liu, Hongsheng Zhang, Yuankai Qi, Peng Wang, Yanning Zhang, and Qi Wu. "AerialVLN: Vision-and-Language Navigation for UAVs." *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2023, pp. 15384-15394. URL: <https://openaccess.thecvf.com/content/ICCV2023/html/Liu_AerialVLN_Vision-and-Language_Navigation_for_UAVs_ICCV_2023_paper.html>[31] Xiangyu Wang, Donglin Yang, Ziqin Wang, Hohin Kwan, Jinyu Chen, Wenjun Wu, Hongsheng Li, Yue Liao, and Si Liu. "Towards Realistic UAV Vision-Language Navigation: Platform, Benchmark, and Methodology." *International Conference on Learning Representations (ICLR)*, 2025. URL: <https://openreview.net/forum?id=rUvCIvI4eB>; arXiv:2410.07087.

[32] Sourav Sanyal and Kaushik Roy. "ASMA: An Adaptive Safety Margin Algorithm for Vision-Language Drone Navigation via Scene-Aware Control Barrier Functions." arXiv:2409.10283, 2024; accepted by *IEEE Robotics and Automation Letters*. URL: <https://arxiv.org/abs/2409.10283>

[33] Xinyuan Zhang, Yonglin Tian, Fei Lin, Yue Liu, Jing Ma, Kornélia Sára Szatmáry, and Fei-Yue Wang. “LogisticsVLN: Vision-Language Navigation For Low-Altitude Terminal Delivery Based on Agentic UAVs.” arXiv:2505.03460, 2025. URL: <https://arxiv.org/abs/2505.03460>[34] Hanxuan Chen, Jie Zheng, Siqi Yang, Tianle Zeng, Siwei Feng, Songsheng Cheng, Ruilong Ren, Hanzhong Guo, Shuai Yuan, Xiangyue Wang, Kangli Wang, and Ji Pei. "Vision-and-Language Navigation for UAVs: Progress, Challenges, and a Research Roadmap." arXiv:2604.13654, 2026. URL: <https://arxiv.org/abs/2604.13654>

[35] Qwen Team. “Qwen3 Technical Report.” arXiv:2505.09388, 2025. URL: <https://arxiv.org/abs/2505.09388>

[36] DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv:2501.12948, 2025. URL: <https://arxiv.org/abs/2501.12948>

[37] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. “Efficient Memory Management for Large Language Model Serving with PagedAttention.” *ACM Symposium on Operating Systems Principles (SOSP)*, 2023. URL: <https://arxiv.org/abs/2309.06180>[38] AAAI. “AAAI-26 Main Technical Track: Call for Papers” and “AAAI-26 Reproducibility Checklist.” 2025. URL: <https://aaai.org/conference/aaai/aaai-26/main-technical-track-call/>; <https://aaai.org/conference/aaai/aaai-26/reproducibility-checklist/>

[39] IEEE Intelligent Transportation Systems Society. “IEEE Transactions on Intelligent Transportation Systems (T-ITS): Scope.” URL: <https://ieee-itss.org/pub/t-its/>

---

## 15. Appendix: Current AAAI Priority Promotion Plan

### 15.1 Positioning of the paper

The AAAI version is first narrowed down to an AI method paper:

**Counterexample-Guided Verified Language-to-STL Planning for UAVs**

The core is not "LLM can plan UAVs", but: after the local open source LLM generates UAV mission specifications, a formal verifier is used to generate counterexample diagnosis, and then the LLM is driven to correct the specifications or plan, and finally generates a verifiable trajectory.

### 15.2 AAAI Contribution Statement

AAAI advocates three contributions:

1. A typed IR to LTL/STL UAV mission specification compilation chain covering arrival, avoidance, sequence, inspection, time window, altitude and distance constraints.
2. A verification-guided repair loop that converts syntax errors, missing grounding, unsatisfiable, unsafe trajectories, and low STL robustness into structured counterexample feedback.
3. A UAV-NL2STL benchmark, including natural language tasks, maps, gold standard specifications, executable traces, and failure diagnostic labels.

### 15.3 Timeline| Time | Task | Output |
|------|------|------|
| 2026-05-18 to 2026-05-24 | Complete the core literature table and freeze the benchmark schema | Related work table + dataset spec |
| 2026-05-25 to 2026-06-07 | Implement map/task generator, gold TaskIR/LTL/STL template, basic planner | Data generation script + baseline planner |
| 2026-06-08 to 2026-06-21 | Implement Spot/RTAMT verifier and counterexample feedback | verifier module |
| 2026-06-22 to 2026-07-05 | Run local model, baseline, no-repair/full repair preliminary experiment | First moderator result table |
| 2026-07-06 to 2026-07-19 | Main experiment, ablation, generalization, failure case statistics | Complete experiment table and figures |
| 2026-07-20 to AAAI abstract deadline | Complete abstract, introduction, method, figure 1, main results table | AAAI first draft |
| AAAI full text before deadline | Compressed to 7 pages, add appendix, reproducibility, anonymous repository | Submission package |

As of 2026-05-19, the official CFP of the AAAI-27 Main Technical Track has not been retrieved on the AAAI official website; currently, the 7-page technical content, reproducibility checklist and code/data appendix requirements of the AAAI-26 Main Technical Track are still prioritized as the basis for inversion [38]. Once the AAAI-27 CFP is released, this timeline needs to be updated as soon as possible, especially the abstract deadline, full text deadline, supplementary material deadline and LLM generated text policy.

### 15.4 Subsequent T-ITS extensions

When AAAI is later expanded into T-ITS, the new content must be clearly different from the conference version. It is recommended to add:- AirSim/SUMO or Low Altitude Logistics Digital Twin Experiment.
- Multi-UAV coordination and airspace conflict arbitration.
- Traffic system indicators: mission throughput, airspace occupancy, safety margin, delivery/inspection completion rate, communication delay robustness.
- Edge deployment experiment: latency-energy trade-off for 4-bit / 8-bit models on Jetson or 4090.
- Title changed from AAAI’s “verified planning method” to “safe low-altitude UAV operation for intelligent transportation systems”.

---

**Version Notes:** The content of this article has been updated to `v2`, but the file name continues to be `v1-20260517` to meet the requirement of "modify directly on the V1 version" of this round. Incremental optimization on 2026-05-19 complements data leakage prevention, failure taxonomy, parameter budgeting, indicator formulas, chart planning and AAAI compliance risks. In the next version, it is recommended to update to `v3-YYYYMMDD` after completing the data set schema and the first round of baseline running, focusing on replacing the `TBD` table and supplementing real experimental results and failure cases.