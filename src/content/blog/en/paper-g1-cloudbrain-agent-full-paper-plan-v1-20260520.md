---
title: "Paper G1 complete paper proposal v1: Verifiable LLM Agent for low-altitude traffic cloud brain"
description: "Completely plan the research questions, submission positioning, algorithm design, data construction, model selection, local deployment, experimental plan, evaluation indicators, expected conclusions, chart design, risk control and execution plan for the first CloudBrain-Agent conference paper."
pubDate: 2026-05-20
updatedDate: 2026-05-23
tags: ["Paper G1", "CloudBrain-Agent", "Low altitude traffic cloud brain", "LLM Agent", "MCP", "Tool Use", "AAAI", "IJCAI", "UAV", "Formal verification"]
category: Tech
---

# Paper G1 Complete paper proposal v1: Verifiable LLM Agent for low-altitude traffic cloud brain

> Core judgment: The first paper should not be written as "fine-tuning a large low-altitude traffic model", but should be written as a **verifiable, reproducible, and deployable low-altitude traffic LLM Agent method paper**.  
> Recommended topic: **CloudBrain-Agent: Tool-Augmented and Verification-Guided LLM Agents for Low-Altitude Traffic Operation**.

---

## 1. Paper positioning and submission judgment

### 1.1 Positioning in one sentence

This article studies the large model agent in the low-altitude traffic cloud brain: given a natural language task, urban low-altitude airspace status, UAV fleet status and safety constraints, how the LLM agent can generate safe, executable and interpretable low-altitude traffic operation decisions through structured intermediate representation, tool invocation, formal verification and simulation feedback.

### 1.2 Recommended contributions

Preferred: **AAAI/IJCAI Master**.  
Alternatives: AAMAS, IROS/ICRA workshop, T-ITS follow-up expansion.

According to the time point of 2026-05-20, the specific session needs to be aligned with the next round of AAAI/IJCAI CFP; this article is still designed in the style of the AAAI/IJCAI main conference, because AAAI emphasizes AI methods, application fields and reproducibility, and the IJCAI-ECAI AI and Robotics track clearly focuses on robot agents, generative AI, reasoning, structured modeling and action consequences [1] [2].

### 1.3 Why is this article more suitable to do first than "Low-altitude traffic large model fine-tuning"

Directly fine-tuning a LowAltitudeGPT will encounter three review risks:

1. LoRA, QLoRA, and DPO are already mature training paradigms. Simply changing domain data is not enough to constitute the main contribution [3] [4] [5].
2. Low-altitude traffic is a safety-critical system, and it is difficult to convince reviewers that LLM directly outputs control actions.
3. Real low-altitude traffic operation data is scarce. If you focus on "large model training" in the first article, you will be asked about data scale, training budget, and model novelty.Therefore the first article should focus on **Agent + Tools + Verifier + Simulator Feedback**. The large model is not the final controller, but a layer of task understanding, tool orchestration, counterexample repair, and interpretation. This setting is naturally connected with agent/tool-use/planning work such as ReAct, ToolLLM, LLM+P [6] [7] [8], and can also catch up with TrafficGPT’s discussion of the interaction between traffic foundation model and LLM [9].

### 1.4 2026-05-22 Writing Calibration: Don’t write G1 as a TR-C story, but keep the traffic system evidence

The first investment in G1 is AAAI/IJCAI, so the main contribution must be the AI agent method rather than the transportation journal-style system narrative. A more accurate way to write it is:

> CloudBrain-Agent is an AI agent method evaluated in a safety-critical low-altitude traffic domain.

In other words, the traffic scene provides real difficulty and safety constraints, but the paper still needs to answer questions in the agent field: whether the tool call is reliable, whether the state is consistent, whether the counterexample repair is effective, whether the model is an illusion, and whether the evaluation is reproducible.

At the same time, G1 cannot only report `task_success` and `tool_call_accuracy`. Because low-altitude traffic is a safety-critical area, traffic system evidence must be preserved from the first version of the experiment:| Level | AAAI/IJCAI main text focus | Follow-up T-ITS expansion focus |
|------|---------------------|---------------------|
| Agent capabilities | IR validity, tool-call accuracy, repair success, hallucination rate | human confirmation, operator workload, stateful consistency |
| Safety | safety violation, NFZ violation, battery violation | LoWC/NMAC proxy, risk ratio, weather/communication degradation |
| Efficiency | executable decision, latency, runtime | delay, extra distance, energy, throughput |
| Generalization | unseen city, stress, UNSAT/ambiguous tasks | high-density corridor, non-cooperative UAV, communication loss, real context city split |
| System Enlightenment | When is verifier feedback necessary | Which scenarios must be returned from LLM agent deterministic solver/human supervisor |

Therefore, the boundary conditions of G1 must be written clearly:

- Do not claim real deployment;
- Does not claim end-to-end automatic control;
- LLM is not claimed to be a replacement scheduler/planner/validator;
- only claims that the LLM agent is responsible for task understanding, orchestration, repair and interpretation in the tool chain and verification feedback;
- The transportation system conclusions are only written as "observable operational implications" and are not exaggerated into policy recommendations.

### 1.5 2026-05-23 Compilation: frozen list of submission versions

The first version of G1 submission must freeze three claims to avoid turning into a low-altitude platform specification:1. **Domain-grounded tool-use benchmark**: CloudBrain-Bench not only tests the JSON format, but also tests the function selection, parameter grounding, state dependency, policy compliance and multi-round consistency in the low-altitude transportation chain.
2. **Verifier-guided repair**: Safety errors, unexecutable errors and ambiguous tasks in low-altitude traffic missions must be converted into structured repair signals through LTL/STL verifier, route planner and simulator feedback.
3. **Local-deployable agent implementation**: The main experiment must be reproducible on the local open source model, and the API model only serves as teacher or upper bound.

The first part must be completed:| Modules | Freeze Requirements |
|------|----------|
| LowAltitudeIR | Fixed schema, type checker, error codes and JSON examples |
| Tools | At least 6: airspace query, fleet status, assignment, route planner, LTL/STL verifier, scenario simulator / risk estimator |
| CloudBrain-Bench | dev/validation/test/stress split, covering SAT, UNSAT, ambiguous, resource-limited, stress scenarios |
| Baselines | Direct LLM, JSON-only, ReAct, LLM+P / planner-only, tool-use no verifier, CloudBrain full |
| Metrics | task success, tool-call accuracy, executable decision, safety violation, repair success, hallucination rate, latency/cost |
| Ablations | no IR, no verifier, no simulator, no repair, API teacher vs local model |
| Data layer | synthetic master data + OSM/FAA/OD/SUMO real context fields, do not write real data as deployed system |

The first suspended content:

- Complete MCP productization;
- Multi-agent collaboration as main contribution;
-Write the LowAltitudeGPT fine-tuning model as the main method;
- Real UAV deployment or flight;
- VLA/world model/embodied AGI proposition.

The function of this frozen list is to control the boundaries of the paper: G1 only proves that "verifiable LLM agent in the key domain of low-altitude traffic safety" is established, and subsequent G2/G3/G4 will deal with fine-tuning, multi-agent and embodied expansion respectively.

---## 2. Draft abstract

Urban low-altitude traffic operations require real-time decision-making among dynamic tasks, limited airspace resources, UAV status constraints, and safety rules. Large language models have the ability to understand natural language and decompose complex tasks, but if used directly for UAV scheduling and path planning, they will produce hallucinations, unexecutable plans, and safety violations. This article proposes **CloudBrain-Agent**, a tool enhancement and verification guidance LLM agent framework for low-altitude traffic cloud brain. CloudBrain-Agent parses natural language tasks and system states into typed `LowAltitudeIR`, invokes airspace query, UAV allocation, path planning, LTL/STL verification, scenario simulation and risk assessment tools, and iteratively fixes decisions using verifier counterexample and simulation feedback. We build **CloudBrain-Bench** to cover emergency distribution, inspections, no-fly zone avoidance, corridor congestion, charging bottlenecks, multi-mode fallback and unsatisfactory tasks. The experiment will compare direct LLM, prompt-only ReAct, tool-use without verification, LLM+P, TrafficGPT-style orchestration and CloudBrain-Agent full. The pre-registration expectation is that CloudBrain-Agent significantly outperforms prompt-only and tool-only baselines in task success, executable decision rate, safety violation rate, hallucination rate, and repair success, while maintaining acceptable local deployment latency.

---

## 3. Research questions and core hypotheses

### 3.1 Research questions

**RQ1:** Can the LLM agent stably generate decision chains of the correct type and tool-executable in low-altitude traffic missions?

**RQ2:** Can formal verification and simulation feedback significantly reduce unexecutable plans, security violations, and hallucinations in LLM?

**RQ3:** Compared with directly fine-tuning the vertical model, can the solution of general LLM + typed IR + MCP/tools + verifier form a reproducible, deployable, and scalable research system faster?**RQ4:** Can the local open source model approach the closed source strong model performance under the data and rule feedback generated by the teacher API, and support the subsequent LowAltitudeGPT paper?

### 3.2 Core Assumptions

H1: typed `LowAltitudeIR` can significantly improve structured output quality and tool-call accuracy.  
H2: Verification-guided repair can significantly improve the executable decision rate and reduce the safety violation rate.  
H3: Simulator feedback is most critical for generalization of unseen dangerous scenes.  
H4: There is no need to train the vertical foundation model in the first stage; the general model + agent tool layer + verifier post-processing is enough to complete the G1 paper.  
H5: After the local Qwen3 / DeepSeek-R1-Distill model is deployed through vLLM, it can be used as a reproducible main experimental model; API models such as GPT-5.2 serve as teachers and performance upper limits [10] [11] [12].

---

## 4. Paper contribution design

It is recommended that the final contribution of the paper be written in three articles to avoid being scattered:

1. **CloudBrain-Agent framework**
   A typed tool-use LLM agent is proposed for low-altitude traffic cloud brain, which unifies natural language tasks, urban airspace status, UAV fleet status and safety constraints into `LowAltitudeIR`.

2. **Verification-guided repair for low-altitude traffic operation**
   Transform failure feedback from LTL/STL verifiers, route planners, and simulators into structured counterexamples that drive LLM repair tool invocations, task constraints, and path/scheduling recommendations.3. **CloudBrain-Bench and evaluation protocol**
   Build a low-altitude traffic cloud brain benchmark, covering indicators such as tool-call accuracy, executable decision, safety violation, repair success, generalization, latency and human trust.

It is not recommended to write the contribution as "We trained a large low-altitude traffic model". Fine-tuning can be done as an experimental extension or as the next G2.

### 4.1 Paper positioning matrix after the second round of research

After online research, the best entry point for G1 should be more clearly **domain-grounded agent evaluation + safety verification**, rather than general LLM applications. AgentBench proves that LLM agents need to evaluate reasoning and decision-making in an interactive environment [34]; BFCL explains that function calling needs to check function selection, parameters, parallel calls and relevance detection [35]; $\tau$-bench further emphasizes multi-round interaction, API, domain policy and consistency index `pass^k` [36]; ToolSandbox points out that state dependency, canonicalization and insufficient information are the key difficulties of tool-based agents. [37].

The inspiration for G1 from these works is: CloudBrain-Bench cannot only evaluate "whether JSON is output", but also evaluates the agent's **status update, rule compliance, tool dependency, failure repair and multi-round consistency** in the low-altitude transportation chain.| Already directed | Representative work | Limitations | Differences in G1 |
|----------|----------|------|-----------|
| General agent benchmark | AgentBench, $\tau$-bench, ToolSandbox [34] [36] [37] | Does not include low-altitude traffic safety constraints and UAV tool chain | Domain tools, policy, verifier for UTM/UAV |
| function calling benchmark | BFCL [35] | Focus on the correctness of function calls and not care about physical executability and security | Tool calls must go through planner/verifier/simulator |
| LLM + traffic | TrafficGPT, ITS LLM survey [9] [13] [14] | Multi-focus ground traffic or traffic model interaction | Extension to low-altitude airspace, UAV fleet and formal safety |
| NL-to-LTL / robot task spec | Lang2LTL, LTLCodeGen, ConformalNL2LTL [21] [22] [23] | Mainly solve specification generation | Put specification verification into the complete cloud brain decision-making closed loop |
| UTM/UAM simulation | NASA TCL4, CORUS-XUAM, AAM-Gym [38] [39] [40] | LLM agent tool orchestration is usually not studied | Support CloudBrain-Bench with UTM/UAM concepts and scenarios |

---

## 5. Related work framework

### 5.1 LLM for transportation

TrafficGPT explains that LLM can be used as an interaction and processing entrance for traffic foundation models, but also points out that traffic numerical data, simulation and model interaction cannot be generated solely by plain text [9]. Recent ITS reviews further place LLM in traffic semantic interfaces, decision aids, and multi-source data understanding [13] [14]. UrbanGPT and UniST represent the direction of urban space-time foundation model and are suitable for supporting urban state understanding, but they are not low-altitude UAV operation tool chains [15] [16].### 5.2 LLM agents and tool use

ReAct interweaves reasoning trace and action and is the basis of the agent loop in this article [6]. Toolformer and ToolLLM prove that LLM can learn API/tool ​​usage, but they do not solve the problems of low-altitude traffic safety verification and mission executability [7][17]. MCP and OpenAI Agents SDK provide a more standard tool connection method, which helps make scheduler, planner, verifier and simulator into replaceable tools [18] [19].

After the second round of research, related work should also add the agent evaluation system: AgentBench is a multi-environment LLM-as-agent benchmark [34]; BFCL specifically evaluates function calling and relevance detection [35]; $\tau$-bench uses multiple rounds of user-agent-tool interaction and `pass^k` to evaluate reliability [36]; ToolSandbox emphasizes tool execution status, implicit dependencies and insufficient information scenarios [37]. The G1 evaluation protocol should incorporate these ideas but change the environment to a low-altitude traffic cloud brain.

### 5.3 LLM planning and formal verification

LLM+P and PlanBench show that LLM alone is not reliable for planning and needs to be combined with external planners, formal representations and evaluation protocols [8] [20]. Lang2LTL, LTLCodeGen and ConformalNL2LTL illustrate that the translation of natural language to temporal logic is developing, but they mainly focus on specification generation and incomplete coverage of scheduling, routing, simulation and risk closed loops in the low-altitude traffic cloud brain [21] [22] [23]. Spot and RTAMT can be used as LTL/STL verification tools respectively [24] [25].

### 5.4 UAV, UTM, and simulation dataFAA UTM defines low-altitude UAV traffic management as a collaborative ecology that supports flight planning, authorization, surveillance, and conflict management [26]. FAA UAS Facility Maps provide an altitude reference that can be quickly approved for Part 107 operations in controlled airspace, and are suitable for airspace rules proxy [27]. OSM/Overpass, NYC TLC OD data, SUMO, AirSim and Flightmare can jointly support the synthetic-to-real benchmark [28] [29] [30] [31] [32].

To enhance low-altitude traffic credibility, G1 should further cite the NASA TCL4 Nevada flight tests: this test includes BVLOS, urban canyon, weather front, concert emergency response and CNS issue scenarios, and is suitable as a source for scenario taxonomy and human-system information quality discussions [38]. European CORUS-XUAM provides U-space/UAM operational concept, U3/U4 service models, ATM-U-space coordination, vertiport guidance and human-in-the-loop evidence [39]. AAM-Gym can be used as a simulation control for advanced air mobility AI testbed, especially corridor separation assurance [40].

---

## 6. Problem Formulation

### 6.1 System status

At the discrete decision time $t$, the low-altitude traffic cloud brain receives the system status:

$$
S_t = \langle \mathcal{U}_t, \mathcal{R}_t, \mathcal{A}_t, \mathcal{M}, \mathcal{C}_t, \mathcal{H}_t \rangle
$$

Among them:- $\mathcal{U}_t$: A collection of UAVs. Each UAV has position, power, load, speed, and mission status.
- $\mathcal{R}_t$: task collection, including distribution, inspection, emergency response, return, and charging.
- $\mathcal{A}_t$: Airspace status, including corridor, no-fly zone, altitude, weather, and capacity.
- $\mathcal{M}$: City map, including OSM road network, POI, buildings, and functional areas.
- $\mathcal{C}_t$: safety and operational constraints, including LTL/STL, deadline, distance, energy.
- $\mathcal{H}_t$: historical events, failure cases, human feedback and verifier feedback.

Natural language instructions are denoted $q_t$. The goal is to generate executable decisions:

$$
\pi_t = \langle z_t, a_{1:k}, y_t, e_t \rangle
$$

Where $z_t$ is `LowAltitudeIR`, $a_{1:k}$ is the tool call sequence, $y_t$ is the scheduling/path/risk decision, and $e_t$ is the explanation.

### 6.2 Safe Executable Targets

A decision $\pi_t$ is considered successful if and only if:

1. **Schema validity**: $z_t$ satisfies the `LowAltitudeIR` type constraint.
2. **Tool executability**: All tool call parameters are legal and return non-error results.
3. **Planning feasibility**: Scheduling and path planning are executable.
4. **Temporal safety**: LTL/STL specifications verified.
5. **Simulation robustness**: Does not trigger collisions, no-fly zone violations, or deadline violations in specified scenario seeds.
6. **Human interpretability**: Interpretation does not involve non-existent entities, tools or rules.

formal:$$
\text{Success}(\pi_t) =
\mathbb{1}[
V_\text{schema}(z_t)
\land V_\text{tool}(a_{1:k})
\land V_\text{plan}(y_t)
\land V_\text{logic}(y_t)
\land V_\text{sim}(y_t)
]
$$

### 6.3 What this article does not do

- Prevent LLM from directly outputting low-level control variables.
- Can be deployed directly without claiming real airspace.
- Do not disguise synthetic data as real operational data.
- Do not train the low-altitude traffic foundation model from scratch.

---

## 7. Method: CloudBrain-Agent

### 7.1 Overall architecture

```text
User instruction + System state
  -> Context builder / RAG
  -> LLM planner
  -> LowAltitudeIR
  -> Tool router
  -> Scheduler / Route planner / Verifier / Simulator / Risk assessor
  -> Counterexample & robustness feedback
  -> Repair agent
  -> Final verified decision + explanation
```

### 7.2 LowAltitudeIR

`LowAltitudeIR` is the key to the paper. It is more strict than normal JSON output and must be able to connect tools and validators.

```json
{
  "task_id": "task_00042",
  "intent": "emergency_delivery",
  "priority": "high",
  "entities": {
    "origin": "hospital_A",
    "destination": "accident_site_3",
    "candidate_uavs": ["uav_03", "uav_07"]
  },
  "constraints": {
    "deadline_sec": 600,
    "avoid_zones": ["school_zone_2", "nfz_temp_1"],
    "altitude_min_m": 30,
    "altitude_max_m": 120,
    "min_separation_m": 10,
    "battery_reserve_ratio": 0.2
  },
  "tool_plan": [
    {"tool": "query_airspace", "args": {"region": "downtown"}},
    {"tool": "assign_uav", "args": {"objective": "min_delay_safe"}},
    {"tool": "plan_route", "args": {"planner": "astar_3d"}},
    {"tool": "verify_ltl_stl", "args": {"logic": ["avoid_nfz", "meet_deadline"]}},
    {"tool": "simulate_scenario", "args": {"stress_level": "medium"}}
  ],
  "fallback_policy": "ground_transfer_or_human_confirm"
}
```

Field-level constraints:

| Fields | Types | Constraints |
|------|------|------|
| `intent` | enum | delivery / patrol / inspection / emergency / return / charge |
| `priority` | enum | low / normal / high / critical |
| `entities` | object | Must refer to an entity that exists in the map or UAV state |
| `constraints` | object | must be able to be translated into planner/verifier input |
| `tool_plan` | list | The tool name must come from the registry, and the parameters must conform to the schema |
| `fallback_policy` | enum | Triggered when unreachable, unsafe, timeout |

### 7.2.1 LowAltitudeIR v0.1 detailed field specifications

In the first version, do not design the IR too large, but ensure that each field can be consumed by tools, evaluated by indicators, and analyzed and attributed by errors. It is recommended to split the IR into 9 top-level fields:| Top-level fields | Required | Type | Description | Metrics that failure will affect |
|----------|------|------|------|------------------|
| `task_id` | Yes | string | Unique task ID in the dataset | traceability |
| `intent` | Yes | enum | Task intent: delivery, inspection, patrol, emergency, return, charge, monitoring | IR field F1 |
| `priority` | yes | enum | low, normal, high, critical | policy compliance |
| `entities` | Yes | object | origin, destination, candidate_uavs, sensitive_zones, handoff_points | hallucination rate |
| `constraints` | Yes | object | Time, altitude, distance, battery, no-fly zone, capacity, weather risk | safety violation rate |
| `tool_plan` | Yes | list | Linearized plan for tool-call DAG | tool-call accuracy |
| `verification_specs` | yes | object | LTL/STL specifications and interpretable natural language rules | verified decision rate |
| `fallback_policy` | yes | enum | ground_transfer, wait, human_confirm, safe_refusal | safe refusal accuracy |
| `explanation_plan` | no | object | tool results and constraints that need to be referenced in the explanation | human trust score |

Entity field recommendations are specific to:| Fields | Examples | Checking methods |
|------|------|----------|
| `origin` | `hospital_A` | Must exist in `city_state.entities` |
| `destination` | `accident_site_3` | Must exist in task or map |
| `candidate_uavs` | `["uav_03", "uav_07"]` | Must exist in `uav_state` and status is available |
| `avoid_zones` | `["school_zone_2", "nfz_temp_1"]` | Must exist in airspace/map |
| `handoff_points` | `["metro_station_4"]` | Required for multimodal fallback |

Constraint field recommendations are specific to:

| Field | Unit | Default | Description |
|------|------|------|------|
| `deadline_sec` | second | null | Empty if there is no deadline |
| `altitude_min_m` | meter | 30 | Minimum flight altitude |
| `altitude_max_m` | meter | 120 | Maximum altitude, subject to airspace proxy |
| `min_separation_m` | meter | 10 | Minimum distance to obstacles/UAV/sensitive zone |
| `battery_reserve_ratio` | ratio | 0.2 | Minimum remaining battery ratio after arrival |
| `max_risk_level` | enum | medium | low, medium, high |
| `corridor_capacity_required` | int | 1 | Minimum capacity occupied by corridor |

### 7.2.2 LowAltitudeIR verification sequence

IR verification should be hierarchical to facilitate error analysis:1. **JSON validity**: Whether it can be parsed into JSON.
2. **Schema validity**: Whether the field type, enum, and required fields are correct.
3. **Entity grounding**: Whether all entities exist in the current state.
4. **Constraint grounding**: Whether constraints can be converted into planner/verifier parameters.
5. **Tool dependency**: Whether the tool input depends on the previous tool output.
6. **Policy compatibility**: whether priority, fallback, and human confirm comply with the rules.

Each level of failure must be written to `error_type`, for example:

```json
{
  "valid": false,
  "stage": "entity_grounding",
  "error_type": "nonexistent_destination",
  "field": "entities.destination",
  "value": "hospital_X",
  "allowed_entities": ["hospital_A", "hospital_B", "accident_site_3"]
}
```

### 7.3 Tool Registry

The first version of the tool should be made into a Python function and then packaged into an MCP server. The advantage of MCP is the standardized tools/context interface, which allows different models and agent runtimes to reuse the same set of tools [18] [19].| Tool | Required | Input | Output | Failure Type |
|------|------|------|------|----------|
| `query_city_state` | Yes | region, time | POI, buildings, ground graph | unknown_region |
| `query_airspace` | Yes | region, altitude, time | corridor, NFZ, ceiling | restricted_airspace |
| `assign_uav` | yes | task, UAV states | selected UAV / none | no_available_uav |
| `plan_route` | yes | start, goal, constraints | path / unreachable | no_path |
| `verify_ltl_stl` | yes | path, temporal specs | pass/fail/counterexample | spec_violation |
| `simulate_scenario` | Yes | decision, scenario seed | success/risk/collision | sim_failure |
| `risk_assess` | yes | decision, state | risk score, reasons | high_risk |
| `explain_decision` | Optional | decision trace | explanation | hallucinated_explanation |

### 7.3.1 Tool API Contract

All tools return uniformly:

```json
{
  "ok": true,
  "tool": "plan_route",
  "request_id": "tool_00042_03",
  "result": {},
  "warnings": [],
  "error": null,
  "provenance": {
    "input_hash": "sha256:...",
    "data_sources": ["city_osm", "airspace_rules"],
    "timestamp": "2026-05-20T12:00:00Z"
  }
}
```

On failure:

```json
{
  "ok": false,
  "tool": "plan_route",
  "request_id": "tool_00042_03",
  "result": null,
  "warnings": [],
  "error": {
    "type": "no_path",
    "message": "No feasible path avoiding nfz_temp_1 within altitude range.",
    "recoverable": true,
    "suggested_actions": ["relax_deadline", "choose_ground_transfer", "human_confirm"]
  },
  "provenance": {
    "input_hash": "sha256:...",
    "data_sources": ["city_grid", "airspace_rules"]
  }
}
```

Specific tool input and output:| Tool | Key input fields | Key output fields | Recoverable failures | Non-recoverable failures |
|------|--------------|--------------|------------|--------------|
| `query_city_state` | `region_id`, `bbox`, `time` | `pois`, `buildings`, `roads`, `sensitive_zones` | `partial_map` | `unknown_region` |
| `query_airspace` | `bbox`, `altitude_range`, `time` | `corridors`, `nfz`, `capacity`, `ceiling` | `capacity_low` | `restricted_airspace` |
| `assign_uav` | `task`, `uav_states`, `objective` | `uav_id`, `assignment_score`, `reason` | `low_battery_candidates` | `no_available_uav` |
| `plan_route` | `start`, `goal`, `avoid`, `altitude_range` | `waypoints`, `length_m`, `eta_sec`, `energy_est` | `deadline_risk` | `no_path` |
| `verify_ltl_stl` | `trajectory`, `specs` | `pass`, `violations`, `robustness`, `counterexample` | `negative_robustness` | `invalid_spec` |
| `simulate_scenario` | `decision`, `scenario_seed`, `stress_level` | `success`, `events`, `min_distance`, `delay`, `risk` | `near_miss` | `collision` |
| `risk_assess` | `decision`, `weather`, `traffic`, `history` | `risk_score`, `risk_level`, `top_reasons` | `medium_risk` | `high_risk_no_override` |### 7.3.2 Tool dependency DAG

Tool calls are not arbitrary sequences and should satisfy dependencies:

```text
query_city_state
  -> query_airspace
      -> assign_uav
          -> plan_route
              -> verify_ltl_stl
                  -> simulate_scenario
                      -> risk_assess
                          -> explain_decision
```

Situations that are allowed to be skipped:

- `simulate_scenario` can be turned off in dev-mini, but must be turned on for the main experiment.
- `risk_assess` can be merged into `simulate_scenario`, but paper metrics are still reported separately.
- `explain_decision` does not affect task success, but affects human trust and hallucination.

### 7.3.3 Minimum implementation version

In the first version every tool can be deterministic:

| Tool | Minimal Algorithm | Complex Version |
|------|----------|----------|
| `query_city_state` | Read entities from JSON/GeoJSON | OSM/Overture dynamic query |
| `query_airspace` | Rule template + polygon intersection | UTM/U-space service simulation |
| `assign_uav` | greedy min ETA with battery filter | MILP / Lyapunov scheduler |
| `plan_route` | 3D A* grid | RRT* / MPC-lite |
| `verify_ltl_stl` | Handwritten rules + RTAMT/Spot | Complete temporal logic monitor |
| `simulate_scenario` | discrete-time kinematics | AirSim/Flightmare |
| `risk_assess` | weighted rule score | learned risk model |

### 7.4 Verification-guided repair

The key to CloudBrain-Agent is not to generate it once, but to fix the closed loop:

```text
for i in 1..K:
  z_i = LLM(q, S, feedback_{i-1})
  if not schema_valid(z_i):
      feedback_i = schema_error(z_i)
      continue
  trace_i = execute_tools(z_i)
  verdict_i = verify_and_simulate(trace_i)
  if verdict_i.pass:
      return decision_i
  feedback_i = compress_counterexample(verdict_i)
return safe_refusal_or_human_confirm
```

Counterexample feedback must be structured, not just “failed.” For example:

```json
{
  "failure_type": "stl_robustness_negative",
  "violated_constraint": "always distance_to_school_zone > 30m",
  "counterexample_time_sec": 142,
  "offending_segment": ["p17", "p18", "p19"],
  "suggested_repair": "increase detour radius or choose corridor_C"
}
```### 7.5 Safety memory

Safety memory records three types of information:

1. **Known unsafe patterns**: For example, low battery + high wind speed + tight deadline.
2. **Repair cases**: failed IR, counterexample, successful repair IR.
3. **Human interventions**: manual confirmation, rejection, and reassignment.

The first article does not require complex long-term memory, and only needs to implement retrieval: given the current task, retrieve similar failure cases as few-shot repair context.

### 7.6 Algorithm pseudocode

It is recommended to put a simplified algorithm in the main text of the paper and the complete version in the appendix.

```text
Algorithm 1: CloudBrain-Agent
Input:
  q: natural-language instruction
  S: low-altitude traffic state
  T: typed tool registry
  R: rule and memory retriever
  K: maximum repair iterations

1: C <- BuildContext(q, S, R)
2: feedback <- null
3: for k = 0 ... K do
4:     z <- LLM_Generate_IR(q, C, feedback)
5:     schema_report <- ValidateIR(z, S, T)
6:     if schema_report fails then
7:         feedback <- Compress(schema_report)
8:         continue
9:     trace <- ExecuteToolPlan(z.tool_plan, T)
10:    if trace has unrecoverable tool error then
11:        return SafeRefusal(trace.error)
12:    verdict <- VerifyAndSimulate(z, trace)
13:    if verdict.pass then
14:        explanation <- ExplainDecision(z, trace, verdict)
15:        return VerifiedDecision(z, trace, verdict, explanation)
16:    feedback <- CompressCounterexample(verdict)
17: return HumanConfirmOrSafeRefusal(feedback)
```

### 7.7 Complexity and running time expectations

Suppose the map grid size is $G = X \times Y \times Z$, the number of candidate UAVs is $|\mathcal{U}|$, and the number of tool call rounds is $K$.

| Module | Main complexity | Optimization methods |
|------|------------|----------|
| IR generation | $O(K \cdot C_\text{LLM})$ | Cache prompt, short feedback, low temperature |
| UAV assignment | $O(|\mathcal{U}|)$ greedy | Pre-filtering is not available UAV |
| 3D A* | $O(G \log G)$ | corridor mask, hierarchical grid, heuristic |
| STL monitoring | $O(T \cdot |\Phi|)$ | Vectorized trajectory check |
| Simulation | $O(T \cdot N_\text{agents})$ | Batch seeds, early stop |
| Retrieval | $O(\log M)$ approximate | FAISS/Qdrant |

The first article does not need to pursue extreme real-time performance, but it must report end-to-end latency. Suggested goals:- dev-mini: single task 5-20 seconds;
- Local 14B: single task 10-40 seconds;
- API upper bound: 5-30 seconds for a single task;
- Batch evaluation: asynchronous and concurrent, but each sample records independent latency.

---

## 8. Data sources and CloudBrain-Bench build

### 8.1 Data composition

The first main data set is recommended to be called **CloudBrain-Bench**.| Data layer | Source | Whether the main experiment depends on | Function |
|--------|------|----------------|------|
| Synthetic city grid | Procedurally generated | Yes | Controllable, reproducible, scalable |
| OSM city context | OSM / Overpass | Yes | POI, road, building, functional area naming |
| Overture Maps context | Overture Places / Buildings / Transportation | Optional enhancements | High quality POIs, buildings, road topology and stable entity IDs |
| Real airspace grids | FAA UAS Facility Map polygons + UAS Data Dictionary | Yes | Real UASFM geometry, ceiling, airspace/airport/LAANC fields |
| OD demand proxy | NYC TLC / Chicago taxi Optional | Optional | Generate demand hotspots and peak tasks |
| Ground traffic | SUMO | Optional enhancement | Ground fallback travel time |
| Aviation weather | NOAA Aviation Weather Data API METAR + Open-Meteo | Optional enhancements | Real aviation weather, wind speed, visibility, precipitation and weather risk |
| Real UAV flight telemetry | DJI Matrice 100 package-delivery flight dataset | Optional calibration | Energy consumption/ETA calibration for position, current, voltage, wind, speed, load, altitude |
| UTM flight-test context | NASA TCL4 reports | Optional enhancements | City canyon, BVLOS, weather front, emergency response scenarios taxonomy |
| UAV dynamics | Self-built lightweight simulator | Yes | Path, energy consumption, collision, delay |
| Visualssimulator | AirSim/Flightmare | Optional supplements | Subsequent visual/dynamic extensions |OSM/Overpass is suitable for querying urban features [28]; Overture Maps provides places, buildings and transportation layers through GeoParquet, which can complement POI, building and road topology [41]. The airspace layer should not just be written as an abstract proxy: the FAA UAS Facility Maps official page provides UASFM data entry for data providers. The data dictionary clarifies fields such as geometry, center latitude/longitude, `CEILING`, airspace class, airport identifiers and LAANC readiness [27] [43]. The weather layer can use the NOAA Aviation Weather Data API to pull aviation weather observations such as METAR, and then use Open-Meteo to supplement historical/grid weather features [42] [44]. The real UAV dynamics layer can use the DJI Matrice 100 small package delivery flight data released by Scientific Data; this data contains hundreds of in-flight position, energy use, wind, load, altitude and speed changes, which can be used to calibrate energy consumption and ETA instead of specifying the battery model out of thin air [45]. NYC TLC and SUMO still only serve as demand and ground fallback proxies [29] [30]; AirSim and Flightmare supplement closed-loop simulation [31] [32].

### 8.1.1 Real data feasibility judgment

The conclusion after the second search is not that "there is no real data", but that **real data exists at different levels, and there is a lack of a public and complete low-altitude commercial operation closed loop**.| Data issues | 2026-05-21 Public availability | Ways that can be used for G1 | What cannot be claimed |
|----------|---------------------------|------------------|----------------|
| City Map/POI/Buildings/Roads | High | OSM/Overture Real City Context | Not equal to real drone corridor |
| UAS Airspace Altitude Grid | High | FAA UASFM polygon, ceiling, airspace/LAANC fields | UASFM does not equal flight authorization |
| Aviation Meteorology | High | NOAA METAR, Open-Meteo wind and rain characteristics | Airport weather is not equal to block-level low-altitude wind fields |
| UAV real flight energy consumption/position | Medium | DJI M100 delivery telemetry calibrated energy consumption and ETA | Not equal to 100 real operation schedules |
| UTM test flight scenarios and human-machine information flow | Medium | NASA TCL4 scenario taxonomy and UTM information requirements | Reporting is not equal to public raw UTM fleet trajectories |
| Commercial delivery order flow/track log | Low | Only use FAA operational background and motivation for future cooperation | Cannot forge Zipline/Wing/Flytrex order tracks |
| Remote ID exposes real-time trajectories at scale | Low | Not used as the primary data source | Remote ID cannot be used as a ready-made public dataset fleet |

The FAA Part 135 page states that the United States already has approval routes and approved operating entities for package delivery drone operations, so the research question is not purely hypothetical [46]. However, public operational order flows, airspace conflict records, and commercial flight logs are usually not released with the approval page. Remote ID should also not be treated as an off-the-shelf open source trajectory library: GAO in 2024 still recommended that the FAA identify pathways that provide real-time, networked drone location/status data [47]. Therefore the strong statement of G1 should be:> We build a real-context and real-flight-calibrated low-altitude agent benchmark, while leaving fully real operational fleet logs to future operator collaboration.

### 8.1.2 Data tiering strategy

CloudBrain-Bench recommends breaking it down into three levels of confidence:

| Hierarchy | Name | Data composition | Role in the paper |
|------|------|----------|----------------|
| L1 | `Synthetic-Controlled` | Program city, program airspace, program task | Controllable master contrast, ablation, statistical stability |
| L2 | `Real-Context` | OSM/Overture + FAA UASFM + NOAA/Open-Meteo + program tasks | Main experiment priority layer, proving real context grounding |
| L3 | `Real-Flight-Calibrated` | L2 + DJI M100 flight energy consumption/ETA parameter calibration | Calibration analysis and real flight sensitivity verification |

It is not recommended to write L3 as "real operational benchmark". A more stable method of disassembly is:

- **Tasks and gold trace**: still generated by deterministic generator, planner, verifier, ensuring SAT/UNSAT true value.
- **City/Airspace/Weather Context**: As realistic as possible, verify that the agent is grounding to real entities and real airspace fields.
- **Energy consumption/ETA model**: Use real flight data to fit or bucket calibration to verify that safety judgments are not based on arbitrary energy consumption parameters.

### 8.1.3 Real data acquisition recipe

In order to make the first paper reproducible, it is recommended to write the data acquisition as a fixed pipeline:| Step | Input | Operation | Output |
|------|------|------|------|
| 1 | City bounding box | Use Overpass to query hospital, school, park, police, fire_station, building, road | `city_osm.geojson` |
| 2 | The same bbox | Use Overture Places/Buildings to supplement POI and building footprint, retain stable entity id | `city_overture.parquet` |
| 3 | FAA UASFM data download / bbox | Read UASFM polygon, `CEILING`, airport/airspace/LAANC fields | `uasfm_cells.geojson` |
| 4 | nearest ICAO stations + time window | Query NOAA METAR JSON and extract wind, visibility, precip/weather tokens | `aviation_weather.parquet` |
| 5 | Latitude and longitude and time period | Query Open-Meteo historical/forecast weather as a non-airport supplement | `weather_grid.parquet` |
| 6 | Scientific Data DJI M100 files | Parse position, voltage, current, wind, payload, altitude, speed | `uav_flight_calibration.parquet` |
| 7 | Selected cities and dates | Sample NYC TLC / Chicago taxi OD to form demand heatmap | `od_proxy.parquet` |
| 8 | OSM road graph | Import SUMO, estimate ground fallback travel time | `ground_time_matrix.parquet` |
| 9 | UTM/UASFM/CORUS/NASA TCL4 | ManualOrganize rule templates and scenario taxonomy | `airspace_rules.yaml` |
| 10 | real context + calibrated UAV params | Program generation UAV tasks, NFZ, corridor, charging, weather risk | `cloudbrain_samples.jsonl` |
| 11 | samples | planner/verifier/simulator automatically annotates SAT/UNSAT, gold trace, counterexample | `cloudbrain_gold.jsonl` |The main experiment minimally relies on steps 1, 3, 9, 10, and 11; steps 4-8 provide real weather, energy consumption calibration, and ground fallback. Each capture needs to save the original file snapshot, data field version and download date to prevent subsequent changes in FAA/NOAA/map data from causing irreproducibility.

### 8.1.4 How to map real data to benchmark

| Real fields | Map to CloudBrain | Usage |
|----------|-----------|----------|
| OSM `amenity=hospital/school/fire_station` | `origin`, `destination`, `sensitive_zones` | Command entity grounding |
| Overture building footprint | obstacle polygons | route planner/simulator |
| UASFM `SHAPE`, `CEILING` | altitude cap cells | `query_airspace` tool return |
| UASFM airport/airspace fields | airspace provenance | Explanation and policy fields |
| NOAA METAR wind/visibility/weather | weather risk | `risk_assess`, stress scenarios |
| M100 position/speed/altitude | route/ETA calibration bins | ETA distribution |
| M100 current/voltage/payload/wind | energy model calibration | battery reserve check |

### 8.1.5 Real flight calibration task

Use DJI M100 data to only do what you can support:1. Divide into buckets according to payload, cruise speed, altitude and wind.
2. Obtain flight energy or energy consumption proxy from the voltage and current integration.
3. Fit `energy_per_meter`, `eta_multiplier` or conservative quantile lookup.
4. Map the synthetic planner’s route length to energy estimate and battery reserve verdict.
5. Report in the appendix whether the safety decision changes under calibrated and uncalibrated energy models.

It is recommended that the first version use conservative quantiles instead of complex black box energy consumption networks:

$$
E_\text{route} = L_\text{route} \cdot q_{0.9}(e \mid v, h, p, w)
$$

Where $e$ represents the energy consumption per unit distance, $v$ represents the speed, $h$ represents the height, $p$ represents the load, and $w$ represents the wind condition. In this way, real flight data can be integrated into the safety check without turning G1 into an energy consumption modeling paper.

### 8.1.6 Real data split design

| Split | Data layer | Function |
|-------|--------|------|
| `test_synthetic_controlled` | L1 | Main ablation, controllable difficulty |
| `test_real_context_city_a` | L2 | Real city/airspace/weather context |
| `test_real_context_city_b` | L2 | unseen city generalization |
| `test_real_weather_stress` | L2 | METAR/Open-Meteo Weather Risk |
| `test_energy_calibrated` | L3 | battery/ETA safety after real flight calibration |The main table of the paper can be given to both L1 and L2; L3 is recommended as a calibration analysis table or appendix. If the L2 effect is stable, the summary can be written as "real-context benchmark". If L3 is also stable, write “real-flight-calibrated evaluation” again.

### 8.1.7 Real data acquisition code sketch

Do not mix UASFM and METAR loader in agent tools. Prepare offline data first:

```python
def load_uasfm_cells(path: Path, bbox: BoundingBox) -> gpd.GeoDataFrame:
    cells = gpd.read_file(path)
    cells = cells.to_crs("EPSG:4326")
    clipped = cells[cells.geometry.intersects(bbox.to_polygon())].copy()
    keep = ["CEILING", "UNIT", "GLOBAL_ID", "APT1_ICAO", "AIRSPACE_1", "geometry"]
    return clipped[keep]


def fetch_metar_snapshot(station_ids: list[str], hours: int) -> pd.DataFrame:
    response = requests.get(
        "https://aviationweather.gov/api/data/metar",
        params={"ids": ",".join(station_ids), "format": "json", "hours": hours},
        timeout=30,
    )
    response.raise_for_status()
    rows = response.json()
    return normalize_metar_rows(rows)
```

Real flight calibration loader:

```python
def build_energy_calibration(flights: Iterable[FlightLog]) -> EnergyCalibrationTable:
    rows = []
    for flight in flights:
        energy_j = integrate_power(flight.voltage_v, flight.current_a, flight.time_sec)
        path_length_m = trajectory_length(flight.position_xyz)
        rows.append(
            {
                "payload_g": flight.payload_g,
                "cruise_speed_mps": flight.programmed_speed_mps,
                "altitude_m": flight.programmed_altitude_m,
                "wind_bin": wind_bin(flight.wind_speed_mps),
                "energy_per_meter_j": energy_j / max(path_length_m, 1.0),
            }
        )
    return EnergyCalibrationTable.from_rows(rows, quantile=0.9)
```

### 8.1.8 City and scene parameters

It is recommended that the first version fix 4 types of urban layout:

| City Type | Grid Size | POI Characteristics | Low Altitude Risk | Usage |
|----------|----------|----------|----------|------|
| `grid_city` | 50 x 50 x 6 | Regular road network, uniform POI | Low | sanity check |
| `downtown_city` | 80 x 80 x 8 | High building density, intensive hospitals/schools | High | Main experiment |
| `suburban_city` | 100 x 100 x 5 | POI sparse, long distance | medium | battery/deadline |
| `mixed_city` | 120 x 120 x 10 | Mixed commercial areas, residential areas, transportation hubs | High | unseen generalization |

Spatial scale:| Parameters | Default | Range |
|------|------|------|
| cell size | 10 m | 5-20 m |
| altitude layers | 6 | 3-12 |
| max altitude | 120 m | 60-150 m |
| corridor width | 20 m | 10-40 m |
| no-fly zones | 3-12 per map | 0-20 |
| sensitive zones | 5-30 per map | 0-50 |
| charging pads | 3-10 per map | 1-20 |
| UAV count | 10 / 30 / 50 | 5-100 |

Task parameters:

| Parameters | Default | Description |
|------|------|------|
| deadline tightness | medium | loose / medium / tight / impossible |
| priority distribution | 60/25/10/5 | normal/high/critical/low adjustable |
| battery distribution | beta-like | Create low-battery edge cases |
| weather risk | none/low/medium/high | stress split medium to increase |
| demand burst | 1x / 2x / 4x | test corridor and scheduler |

### 8.1.9 Rule Template

The first version only has 8 types of rules, which is enough to write a paper and can be reproduced:| Rule ID | Natural Language | LTL/STL/Program Check |
|---------|----------|------------------|
| R1 | Not entering the temporary no-fly zone | `G not_in_nfz` |
| R2 | Always maintain the minimum safe distance | STL robustness: `dist_to_obstacle > d_min` |
| R3 | The altitude remains within the allowed range | `G altitude_min <= z <= altitude_max` |
| R4 | Arrive before deadline | `F[0, deadline] at_goal` |
| R5 | Reserve battery after return/arrival | program check |
| R6 | corridor capacity does not exceed limit | capacity monitor |
| R7 | critical tasks take priority but cannot be overridden safety | policy check |
| R8 | Triggered when insufficient information or UNSAT safe refusal/human confirm | refusal check |

### 8.1.10 Data quality control

CloudBrain-Bench must avoid "LLM generated spam tags". It is recommended that each sample record four types of quality fields:

| Field | Description |
|------|------|
| `generation_seed` | Random seeds for replicating experiments |
| `source_provenance` | OSM/Overture/Rule Template/Program Generation Source |
| `label_verifier` | Which checker does the SAT/UNSAT label come from |
| `human_review_status` | unchecked / sampled_pass / sampled_fail / corrected |

Sampling inspection strategy:

- Randomly check at least 30 items for each scenario type;
- Randomly check at least 20 items for each failure mode;
- Increase the sampling rate of stress and UNSAT samples to 15%-20%;
- Only natural language and explanations are manually modified, and planner/verifier tags are not manually modified to avoid introducing subjective tags.### 8.2 Sample format

Each sample includes:

```json
{
  "sample_id": "cb_000001",
  "data_tier": "real_context",
  "city_seed": 12,
  "scenario_type": "emergency_delivery_with_nfz",
  "instruction": "请优先派一架无人机把急救包送到 accident_site_3，避开学校和临时禁飞区，10 分钟内到达。",
  "source_provenance": {
    "map_sources": ["osm", "overture"],
    "airspace_sources": ["faa_uasfm"],
    "weather_sources": ["noaa_metar", "open_meteo"],
    "task_source": "deterministic_generator"
  },
  "real_context": {
    "city_id": "pittsburgh_bbox_01",
    "uasfm_snapshot": "faa_uasfm_2026_05",
    "weather_snapshot": "metar_kpit_2026_05_20T12Z"
  },
  "energy_calibration_version": "dji_m100_q90_v0",
  "state": {
    "uavs": "...",
    "airspace": "...",
    "map": "...",
    "tasks": "..."
  },
  "gold_ir": "...",
  "gold_tool_trace": "...",
  "gold_decision": "...",
  "logic_specs": ["G not_in_nfz", "F[0,600] arrive_destination"],
  "label": "SAT",
  "failure_modes": []
}
```

### 8.3 Scene type

| Scene | Proportion | Difficulty |
|------|------|------|
| Normal delivery | 15% | Normal scheduling and path planning |
| Emergency delivery | 15% | priority, deadline, risk tradeoff |
| Patrol / inspection | 10% | Multiple waypoint temporal constraints |
| No-fly zone avoidance | 15% | LTL/STL safety constraints |
| Corridor congestion | 10% | Airspace capacity and latency |
| Charging bottleneck | 10% | Power constraints and fallback |
| Weather / wind risk | 10% | Risk assessment and rejection |
| Multimodal fallback | 10% | UAV-ground transfer |
| UNSAT / ambiguous tasks | 5% | Safe rejections and clarifications |

### 8.4 Data scale

Feasible scale of the first version:

| Split | Number of samples | Purpose |
|-------|--------|------|
| Dev-mini | 200 | Quick debugging pipeline |
| Train-like | 3000 | few-shot, RAG, repair memory, not used for main training |
| Validation | 1000 | prompt/model selection |
| Test-seen-city | 1000 | Main Test |
| Test-unseen-city | 1000 | Generalization |
| Test-stress | 1000 | Dangerous scene stress test |

There are about 7200 samples in total, which is enough to support the first benchmark/method paper. Subsequent G2 fine-tuning will be expanded to 50k-100k tool traces.

### 8.5 Gold label generationgold should not all be generated by LLM. Recommended process:

1. Procedurally generate cities, missions, UAV status and rules.
2. The rule template generates gold `LowAltitudeIR`.
3. Call deterministic tools to get the gold tool trace.
4. planner/verifier/simulator determines SAT/UNSAT.
5. The LLM teacher is only responsible for natural language paraphrase and a small amount of explanation text.
6. Sampling 5%-10% for manual inspection, focusing on high-risk and UNSAT samples.

### 8.6 Data file organization

It is recommended that the final open source or internal reproduction experiment use the following structure:

```text
data/cloudbrain_bench/
  README.md
  schemas/
    low_altitude_ir.schema.json
    tool_result.schema.json
    sample.schema.json
  raw/
    osm/
    overture/
    uasfm/
    aviation_weather/
    weather/
    uav_flight_calibration/
    od_proxy/
  processed/
    city_states/
    airspace_rules/
    uasfm_cells/
    weather_risk_tables/
    energy_calibration_tables/
    uav_states/
  splits/
    dev_mini.jsonl
    train_like.jsonl
    validation.jsonl
    test_seen_city.jsonl
    test_unseen_city.jsonl
    test_stress.jsonl
    test_real_context_city_a.jsonl
    test_real_context_city_b.jsonl
    test_real_weather_stress.jsonl
    test_energy_calibrated.jsonl
    test_unsat.jsonl
  gold/
    gold_ir.jsonl
    gold_tool_traces.jsonl
    gold_verdicts.jsonl
  metadata/
    split_stats.csv
    scenario_taxonomy.yaml
    data_sources.yaml
```

### 8.7 Statistics must be reported

Paper Table 2 reports at least:

| Statistical items | Must report |
|--------|----------|
| Total number of samples | total / per split |
| Scenario type distribution | 9 categories scenario type |
| SAT/UNSAT ratio | overall + per scenario |
| Number of cities | seen / unseen |
| Data level | L1/L2/L3 sample number and proportion |
| True context field coverage | OSM/Overture/UASFM/NOAA/Open-Meteo snapshot coverage |
| UAV number distribution | min / median / max |
| Number of constraints | Average number of constraints per task |
| tool call length | gold trace average length |
| Failure type distribution | no_path/nfz/battery/deadline/ambiguity |
| Manual sampling ratio | pass / corrected |

---

## 9. Model selection and deployment plan

### 9.1 The first article does not recommend training a large vertical model first

The main line of G1 is the agent method and verification closed loop. Vertical model training is placed in subsequent G2. G1 can contain a lightweight SFT pre-experiment, but it should not be critical to the success or failure of the paper.

### 9.2 Recommended model matrix| Role | Model | Usage | Required |
|------|------|------|----------|
| Teacher / upper bound | GPT-5.2 or equivalent API | Generate data, strengthen baseline, error analysis | Yes |
| Local main model | Qwen3-14B / Qwen3-32B | Main experiment reproducible agent | Yes |
| Local reasoning model | DeepSeek-R1-Distill-Qwen-14B/32B | repair, counterexample reasoning | Yes |
| Small latency model | Qwen3-8B | Low latency ablation | Optional |
| Embedding | Qwen3-Embedding / BGE-M3 | RAG and safety memory retrieval | Yes |

GPT-5.2 is officially positioned as suitable for coding and agentic tasks, and can be used as a strong teacher and closed-source upper limit [10]. The Qwen3 technical report emphasizes reasoning, instruction following, agent and multilingual capabilities, and is suitable as a local open source main model [11]. DeepSeek-R1 provides 14B/32B reasoning models distilled to Qwen/Llama, which is suitable for counterexample repair [12].

### 9.3 Local or API

Recommended **Hybrid Architecture**:

| Stage | API | Local |
|------|-----|------|
| Week 1-2 | Quick verification prompt, schema, tool design | Synchronous deployment Qwen3-14B |
| Week 3-5 | teacher generates paraphrase and difficult samples | main run dev/validation |
| Week 6-8 | Do upper-bound baseline | Main experiment and reproducible results |
| Before submission | Minor error analysis | All core experiments are locally reproducible |The main table of the paper recommends using the local model as the main model and the API model as the upper bound. This not only has a strong effect, but also avoids reviewers questioning whether it can be reproduced.

### 9.4 Fast processing implementation

Deployment recommendations:

```text
vLLM server
  -> OpenAI-compatible endpoint
  -> Agent runtime
  -> Tool registry / MCP servers
  -> verifier / simulator
```

vLLM provides OpenAI-compatible server, which allows local Qwen/DeepSeek and API models to share a calling interface [33].

### 9.5 Prompt and inference configuration

To ensure reproducibility, all models must have fixed inference parameters:

| purpose | temperature | top_p | max tokens | repair K | description |
|------|-------------|-------|------------|----------|------|
| Direct LLM | 0.2 | 0.9 | 2048 | 0 | Direct output decision |
| JSON-only | 0.0 | 1.0 | 2048 | 0 | Structured output to reduce randomness |
| ReAct | 0.2 | 0.9 | 4096 | 0 | Allow reasoning/action |
| CloudBrain no repair | 0.0 | 1.0 | 4096 | 0 | Single IR + tools |
| CloudBrain full | 0.0 first, 0.2 repair | 1.0 | 4096 | 3 | The repair wheel can be slightly loosened |

It is recommended to split prompt into four sections:

1. **System role**: You are the low-altitude traffic cloud brain agent and do not directly output control quantities.
2. **IR schema**: Give `LowAltitudeIR` JSON schema and enum.
3. **Tool registry**: Lists available tools, input and output, and failure types.
4. **Current task/state**: Current natural language task, UAV status, map, airspace rules, and historical feedback.

The output format must be fixed:

```json
{
  "low_altitude_ir": {},
  "rationale_summary": "one paragraph only",
  "uncertainty": {
    "needs_human_confirmation": false,
    "missing_information": []
  }
}
```Don't leave model outputs complete chain-of-thought; papers and systems only save short rationale summaries, tool trajectories, and verifier feedback.

### 9.6 API and local cost recording

Save each experiment:

| Field | Description |
|------|------|
| `model_name` | API or local model name |
| `endpoint_type` | api/local_vllm |
| `prompt_tokens` | Enter token |
| `completion_tokens` | Output tokens |
| `wall_time_sec` | End-to-end time |
| `llm_time_sec` | LLM call time |
| `tool_time_sec` | Tool execution time |
| `repair_rounds` | Number of repair rounds |
| `estimated_cost_usd` | API estimated cost, local can be filled with 0 or GPU-hour |

This supports the deployment analysis of Table 5.

---

## 10. Baselines

### 10.1 main baseline| Baseline | Description | Questions to answer |
|----------|------|--------------|
| Direct LLM | The model directly outputs decision text | How unreliable is LLM naked running |
| JSON-only LLM | Only requires the output of JSON IR, no tool for execution | Is typed output sufficient |
| ReAct prompting | ReAct style tool invocation, no schema/verifier | Is reasoning-action loop sufficient |
| Tool-use only | There is a tool call, but no verification repair | Is the tool sufficient |
| BFCL-style function calling | Only evaluates whether the function name and parameters are correct, and does not perform physical verification | Whether the success of function calling equals the success of the cloud brain |
| Tau-bench-style policy agent | Has tools and policy rules, but no UAV planner/verifier | Is domain policy following sufficient |
| ToolSandbox-style stateful tool agent | Stateful tool execution and information deficiency handling | The contribution of stateful tool execution to low-altitude tasks |
| LLM+P style | LLM is converted into a planning problem, and the planner solves it | How much can the external planner solve |
| TrafficGPT-style | LLM calls vehicles, no UAV formal safety | Traffic LLM orchestration baseline |
| CloudBrain-Agent w/o simulator | Remove simulation stress testing | simulator feedback contribution |
| CloudBrain-Agent w/o repair | Stop on failure | repair loop contribution |
| CloudBrain-Agent full | Complete method | Main method of this article |

### 10.2 Model baseline| Model | Settings |
|-------|------|
| GPT-5.2 | API upper bound |
| Qwen3-14B | local main |
| Qwen3-32B | local stronger |
| DeepSeek-R1-Distill-Qwen-14B | local repair reasoning |
| Qwen3-8B | small local |

### 10.3 Baseline implementation details

In order to avoid baselines being considered unfair by reviewers, each baseline must clearly enter permissions:

| Baseline | Visible natural language | Visible status | Callable tools | Visible verifier feedback | Repairable |
|----------|--------------|----------|------------|------------------------|--------|
| Direct LLM | Yes | Summary Status | No | No | No |
| JSON-only | Yes | Full status | No | No | No |
| ReAct | Yes | Complete status | Yes | Tool error without counterexample | No |
| Tool-use only | Yes | Full status | Yes | Tool error | No |
| LLM+P style | Yes | Complete status | planner | planner result | No |
| CloudBrain w/o verifier | yes | full status | yes | no | no |
| CloudBrain w/o simulator | yes | full status | yes | verifier only | yes |
| CloudBrain full | yes | full status | yes | verifier + simulator | yes |

Fairness principle:- All methods use the same base model;
- Use the same test split for all methods;
- All methods have the same maximum token budget;
- The maximum number of tool calls for ReAct and CloudBrain is the same;
- Only CloudBrain full uses structured counterexample since this is the contribution of this article.

---

## 11. Experimental design

### 11.1 Experiment 1: Main results

Question: Is CloudBrain-Agent full better than direct LLM, ReAct, tool-use only, and LLM+P?

Data: Test-seen-city, Test-unseen-city, Test-stress.

Indicators:

-Task success rate
-Executable decision rate
- Safety violation rate
-Tool-call accuracy
- Hallucination rate
- Repair success rate
-Latency

### 11.2 Experiment 2: Ablation experiment

| Ablation | Removal of content | Expected impact |
|----------|----------|----------|
| no typed IR | Free text tool call | tool-call accuracy decreased |
| no verifier | No LTL/STL checks | safety violation rising |
| no simulator | No scene stress test | stress pass decrease |
| no repair | No iteration after verification failure | executable rate decrease |
| no memory | Do not retrieve historical failure cases | repair success decreased |
| no RAG | Do not retrieve rules/map context | hallucination rise |

### 11.3 Experiment 3: Counterexample repair analysis

Repair path after statistical verifier/simulator failure:- First repair success rate
- 2nd repair success rate
- 3rd repair success rate
- New violation rate after repair
- The most common failure types: NFZ, deadline, battery, corridor, entity hallucination

### 11.4 Experiment 4: Model and Deployment Analysis

Compare API to native models:

| Model | Indicators |
|------|------|
| GPT-5.2 | Cap effects, costs, delays |
| Qwen3-14B | Locally reproducible main results |
| Qwen3-32B | Local strong model |
| DeepSeek-R1-Distill-Qwen-14B | repair special ability |
| Qwen3-8B | Low latency tradeoff |

### 11.5 Experiment 5: Generalization

Generalization dimension:

- unseen city layout
- unseen POI names
- unseen no-fly zone shape
-unseen tool combination
-unseen emergency scenario
-higher UAV density
-higher demand shock

### 11.6 Experiment 6: Safe rejection of human-machine collaboration

Test whether the model can refuse execution or request human confirmation when UNSAT or insufficient information is available.

Example:

- deadline impossible
- all UAV battery insufficient
- destination inside NFZ
- missing destination
- conflict between priority and safety rule

### 11.7 Experiment 7: Agent reliability and multi-round consistency

Referring to the `pass^k` idea of $\tau$-bench, run the same task $k$ times repeatedly to evaluate whether the agent can complete the task stably [36]. In low-altitude traffic missions, one success but multiple random failures are not safe enough, so it is recommended to report:| Indicator | Meaning |
|------|------|
| `pass@1` | Single run success rate |
| `pass^3` | Proportion of success for the same task 3 times in a row |
| `pass^5` | The proportion of the same task being successful for 5 consecutive times |
| policy compliance | Whether to comply with airspace/security/manual confirmation rules |
| state consistency | Whether the internal state is consistent with the tool return after multiple rounds of tool calls |
| Insufficient-information handling | Whether to clarify/reject when information is insufficient, rather than hallucinatory completion |

This part will make G1 not just a "traffic application", but a transferable contribution to general agent reliability.

### 11.8 Experiment 8: Task Difficulty Stratification

To avoid the main results being obscured by simple samples, reporting is stratified by difficulty:

| Difficulty | Definition | Sample Characteristics |
|------|------|----------|
| Easy | Single task, no NFZ, loose deadline | Normal delivery |
| Medium | 1-2 safety constraints, normal power | NFZ or battery single factor |
| Hard | Multiple constraints, tight deadline, corridor congestion | emergency + NFZ + charging |
| Extreme | High risk or close to UNSAT | stress split |
| UNSAT | No feasible safety solution | safe refusal / human confirm |

The main table reports overall, and the appendix reports per difficulty. The benefit of CloudBrain is expected to be greatest on Hard/Extreme/UNSAT.

### 11.9 Experiment 9: Misattribution

Each failed sample is automatically attributed to the first failure stage:| Stage | Error Type |
|------|----------|
| IR | invalid JSON, schema missing, wrong enum |
| Grounding | nonexistent entity, wrong zone, wrong UAV |
| Tool | wrong tool, wrong order, invalid arguments |
| Planning | no path, wrong UAV, battery infeasible |
| Verification | NFZ, altitude, distance, deadline, capacity |
| Simulation | collision, near miss, weather risk, delay |
| Policy | unsafe override, missing human confirm, wrong refusal |
| Explanation | hallucinated reason, unsupported claim |

It is recommended to use stacked bar for error analysis diagram: distribution of failure stages of different baselines. This can clearly explain what CloudBrain has fixed.

---

## 12. Definition of evaluation indicators

### 12.1 Structured output indicators

**IR exact match**：

$$
\text{IR-EM} = \frac{1}{N}\sum_i \mathbb{1}[z_i = z_i^\*]
$$

**IR field F1**: Calculate precision, recall, and F1 respectively for fields such as intent, entities, constraints, and tool plan.

### 12.2 Tool call indicator

**Tool-call accuracy**：

$$
\text{TCA} = \frac{\#\text{correct tool calls}}{\#\text{all tool calls}}
$$

Correct requirements:

- The tool name is correct;
- The parameter schema is correct;
- The entity referenced by the parameter exists;
- The calling sequence satisfies dependencies.**Tool dependency success**:

$$
\text{TDS} = \frac{\#\text{tool chains satisfying all data dependencies}}{\#\text{tool chains}}
$$

It measures whether the agent first queries airspace/city status, then plans and verifies, rather than relying on downstream tools.

### 12.3 Executability Indicators

**Executable decision rate**：

$$
\text{EDR} = \frac{\#\text{planner executable decisions}}{N}
$$

**Task success rate**：

$$
\text{TSR} = \frac{\#\text{fully verified and simulated successful tasks}}{N}
$$

### 12.4 Security indicators

**Safety violation rate**：

$$
\text{SVR} = \frac{\#\text{safety violated tasks}}{N}
$$

Violation types include:

- no-fly zone intrusion;
- altitude violation;
- min separation violation;
- battery reserve violation;
- deadline violation;
- unsafe fallback;
- hallucinated permission.

The extended version of low-altitude transportation recommends further transporting safety indicators:| Indicators | Definition | Purpose |
|------|------|------|
| LoWC proxy | The ratio below well-clear separation at any time | Measuring the risk of loss of separation |
| NMAC proxy | Number of times below near-mid-air-collision threshold | Measure of severe near-mid risk |
| Risk ratio | The proportion of risk events relative to the rule-based safe baseline | Make different scenarios comparable |
| Safe-refusal precision | The proportion of rejections/requests for manual confirmation that are truly unsafe to execute | Preventing the agent from being overly conservative |

The AAAI/IJCAI main text can only report SVR and violation type breakdown; the T-ITS extension should report LoWC/NMAC proxy and risk ratio.

### 12.5 Hallucination Indicator

**Hallucination rate**：

$$
\text{HR} = \frac{\#\text{outputs containing nonexistent entity/tool/rule}}{N}
$$

### 12.6 Repair indicators

**Repair success rate**：

$$
\text{RSR} = \frac{\#\text{failed first attempts repaired within K iterations}}{\#\text{failed first attempts}}
$$

It is recommended that $K=3$ and report the marginal gain for each round.

**Consistency success**:

$$
\text{pass}^k = \frac{\#\text{tasks successful in all } k \text{ repeated runs}}{N}
$$

This metric is more suitable for safety-critical agents than `pass@1` because low-altitude traffic cloud brains require stable compliance with the rules rather than occasional success [36].

### 12.7 Statistical tests

At least 3 random seeds per experiment. Main results report:- mean ± standard error;
-paired bootstrap 95% confidence interval;
- McNemar test or bootstrap test compares success/failure indicators;
- Report median, p90, p95 for latency.

### 12.8 Main result table template

Paper Table 3 can be filled directly in this format:

| Method | Model | TSR ↑ | EDR ↑ | SVR ↓ | HR ↓ | TCA ↑ | RSR ↑ | pass^3 ↑ | p95 Latency ↓ |
|--------|-------|-------|-------|-------|------|-------|-------|----------|---------------|
| Direct LLM | Qwen3-14B | - | - | - | - | N/A | N/A | - | - |
| JSON-only | Qwen3-14B | - | - | - | - | N/A | N/A | - | - |
| ReAct | Qwen3-14B | - | - | - | - | - | N/A | - | - |
| Tool-use only | Qwen3-14B | - | - | - | - | - | N/A | - | - |
| LLM+P | Qwen3-14B | - | - | - | - | - | N/A | - | - |
| CloudBrain w/o repair | Qwen3-14B | - | - | - | - | - | N/A | - | - |
| CloudBrain full | Qwen3-14B | - | - | - | - | - | - | - | - |

### 12.9 Ablation table template| Variant | TSR ↑ | EDR ↑ | SVR ↓ | TCA ↑ | RSR ↑ | Main explanation |
|---------|-------|-------|-------|-------|-------|----------|
| Full | - | - | - | - | - | Full method |
| no typed IR | - | - | - | - | - | Test structured interface |
| no verifier | - | - | - | - | - | Test formal verification |
| no simulator | - | - | - | - | - | Pressure measurement feedback |
| no repair | - | - | - | - | N/A | Test counterexample repair |
| no memory | - | - | - | - | - | Test failure case retrieval |
| no RAG | - | - | - | - | - | Test rules/map context |

### 12.10 Minimum success threshold

Before entering thesis writing, it is recommended to at least meet these thresholds:

| Indicators | Minimum threshold | Reasons |
|------|----------|------|
| CloudBrain full TSR | More than 10 percentage points higher than ReAct | Method master income |
| SVR | More than 30% lower than Direct LLM | Security Critical Value |
| TCA | Over 85% | Reliable tool calls |
| RSR | More than 40% | Counterexample repair is effective |
| pass^3 | Significantly higher than tool-use only | Multi-round stability |
| p95 latency | local 14B less than 60 seconds | deployable narrative |

---

## 13. Expected experimental conclusions

The following are pre-registration expectations, not experimental results:1. CloudBrain-Agent full is expected to be better than direct LLM, ReAct and tool-use only in task success, executable decision rate and safety violation rate.
2. It is expected that typed `LowAltitudeIR` will mainly improve tool-call accuracy, IR field F1 and hallucination rate.
3. Verifier feedback is expected to mainly improve the executable decision rate and repair success rate.
4. It is expected that simulator feedback is most critical in stress scenarios, especially corridor congestion, wind risk, and NFZ edge cases.
5. Local Qwen3-14B/32B is expected to serve as a reproducible master model, but GPT-5.2 is still upper-bound.
6. DeepSeek-R1-Distill-Qwen is expected to outperform the ordinary instruct model in counterexample repair.

---

## 14. Chart Plan| ID | Type | Content | Priority |
|----|------|------|--------|
| Fig. 1 | Architecture diagram | CloudBrain-Agent’s closed loop from instruction to verified decision | High |
| Fig. 2 | Data generation flow chart | OSM/FAA/OD/SUMO/simulator to CloudBrain-Bench | High |
| Fig. 3 | Main results histogram | TSR, EDR, SVR, HR comparison | High |
| Fig. 4 | Repair curve | Improved success rate of repair iteration 1-3 | High |
| Fig. 5 | Generalization heat map | Performance on seen/unseen city, stress, UNSAT | Medium |
| Fig. 6 | Agent consistency curve | `pass@1`, `pass^3`, `pass^5` and state consistency | Medium |
| Table 1 | Comparison of related work | LLM traffic, tool-use, planning, formal verification, this article | High |
| Table 2 | Data set statistics | Scenario type, SAT/UNSAT, city, number of tasks | High |
| Table 3 | Baseline main results | Comparison of all indicators | High |
| Table 4 | Ablation | Performance changes after component removal | High |
| Table 5 | Model deployment | Effect, latency, cost of API vs local | Medium |
| Table 6 | Data source reproducibility | URL of each type of data, license, whether the main experiment depends on it, fallback | Medium |

---

## 15. Paper structure planning

Compressed by the main text of pages 7-8 of AAAI/IJCAI:

### Abstract

150-200 words. Highlight the key to low-altitude traffic safety, LLM unreliability, CloudBrain-Agent, benchmark, and core results.

### 1 Introduction

Content:- Low-altitude traffic cloud background;
- Risks of direct decision-making by LLM;
- The necessity of tool calling and verification closed loop;
- Three contributions of this article;
- Fig. 1 hero figure.

### 2 Related Work

Three paragraphs:

1. LLM for transportation and spatio-temporal intelligence;
2. LLM agents, tool use, and planning;
3. Formal verification and UAV/UTM simulation.

### 3 Problem Setup

Define states, tasks, `LowAltitudeIR`, tools, success conditions and security constraints.

### 4 Methods

Introducing CloudBrain-Agent:

- context builder;
- LowAltitudeIR parser;
- tool router;
- verifier/simulator;
- repair loop;
-safety memory.

### 5 CloudBrain-Bench

Introduces data sources, generation processes, scenario types, splits, gold labels, and reproducibility.

### 6 Experiments

Main results, ablation, repair analysis, generalization, model deployment.

### 7 Conclusion

Summarize the contributions and write honestly about the limitations: synthetic benchmark, real airspace deployment has not been verified, and human-in-the-loop is still needed.

---

## 16. Implementation route

### 16.1 Minimum Viable System

Only do these in the first month:

```text
cloudbrain/
  ir/schema.py
  tools/city.py
  tools/airspace.py
  tools/scheduler.py
  tools/planner.py
  tools/verifier.py
  tools/simulator.py
  agent/runner.py
  data/generator.py
  eval/metrics.py
```

### 16.2 Recommended technology stack| Modules | Technology |
|------|------|
| Agent runtime | Python + Pydantic + LiteLLM/OpenAI client |
| Local model | vLLM OpenAI-compatible server |
| Tool protocol | Python functions first, MCP wrapper second |
| IR validation | Pydantic JSON Schema |
| Planner | 3D A* first, RRT* optional |
| Verifier | Spot for LTL, RTAMT for STL |
| Simulator | lightweight grid/corridor simulator |
| RAG | Qdrant/FAISS + Qwen3-Embedding/BGE-M3 |
| Storage | JSONL + Parquet + DuckDB |
| Evaluation | pandas + scipy + bootstrap |

### 16.2.1 Code module interface

It is recommended that each module expose the minimum interface:

```python
class LowAltitudeIR(BaseModel):
    task_id: str
    intent: Literal["delivery", "inspection", "patrol", "emergency", "return", "charge", "monitoring"]
    priority: Literal["low", "normal", "high", "critical"]
    entities: dict
    constraints: dict
    tool_plan: list[dict]
    verification_specs: dict
    fallback_policy: str

class ToolResult(BaseModel):
    ok: bool
    tool: str
    request_id: str
    result: dict | None
    warnings: list[str]
    error: dict | None
    provenance: dict
```

Runner main function:

```python
def run_agent(sample: dict, model: str, config: AgentConfig) -> AgentTrace:
    ...
```

Evaluation main function:

```python
def evaluate_trace(sample: dict, trace: AgentTrace) -> dict:
    return {
        "task_success": ...,
        "executable_decision": ...,
        "safety_violation": ...,
        "tool_call_accuracy": ...,
        "hallucination": ...,
        "repair_success": ...,
        "latency_sec": ...,
    }
```

### 16.2.2 Experimental command design

It is recommended that these commands can be used to reproduce the problem after future implementation:

```bash
python -m cloudbrain.data.generate --config configs/data/dev_mini.yaml
python -m cloudbrain.eval.run --split dev_mini --method direct_llm --model qwen3-14b
python -m cloudbrain.eval.run --split dev_mini --method react --model qwen3-14b
python -m cloudbrain.eval.run --split dev_mini --method cloudbrain_full --model qwen3-14b
python -m cloudbrain.eval.aggregate --runs runs/dev_mini --out results/dev_mini.csv
python -m cloudbrain.figures.make_all --results results/main.csv --out figures/
```

### 16.2.3 Configuration file template

```yaml
experiment:
  name: cloudbrain_main_qwen3_14b
  seed: 42
  split: test_seen_city
  max_repair_rounds: 3

model:
  provider: local_vllm
  name: qwen3-14b
  temperature: 0.0
  top_p: 1.0
  max_tokens: 4096

tools:
  enable_city: true
  enable_airspace: true
  enable_scheduler: true
  enable_planner: true
  enable_verifier: true
  enable_simulator: true
  enable_risk: true

evaluation:
  bootstrap_samples: 1000
  report_pass_k: [1, 3, 5]
  latency_percentiles: [50, 90, 95]
```

### 16.3 10-week execution plan| Week | Goals | Deliverables |
|----|------|--------|
| 1 | Freeze problem formulation and IR schema | `LowAltitudeIR v0.1` |
| 2 | Implement city/airspace/UAV/task generator | 200 dev samples |
| 3 | Implement planner/verifier/simulator | deterministic gold labels |
| 4 | Implement agent runner and direct/ReAct baselines | dev-mini results |
| 5 | Extending CloudBrain-Bench to 3000+ | validation split |
| 6 | Run local Qwen3-14B and GPT-5.2 upper bound | main baseline table draft |
| 7 | Implement repair loop, memory, ablation | ablation results |
| 8 | run unseen/stress/UNSAT | generalization figures |
| 9 | Statistical tests, error analysis, graphs | camera-ready figures draft |
| 10 | Writing the first draft of AAAI/IJCAI | full paper draft |

### 16.4 Weekly Acceptance Criteria| Week | Commands that must be run | Acceptance criteria |
|----|----------------|----------|
| 1 | schema validation script | 20 handwritten IRs all verified correctly |
| 2 | data generator | 200 samples generated, split stats no empty fields |
| 3 | tool unit tests | planner/verifier/simulator at least 30 unit tests |
| 4 | dev-mini baseline | direct/ReAct/CloudBrain no repair run through |
| 5 | validation split | 3000+ samples, gold label generation completed |
| 6 | model matrix | Qwen3-14B and GPT upper bound have results |
| 7 | ablation | no IR/no verifier/no repair/no simulator executable |
| 8 | stress/UNSAT | stress and safe refusal indicators can be calculated |
| 9 | figures | 6 figures and 6 tables automatically generated draft |
| 10 | paper draft | The main text is complete, and the appendices include schema and data description |

### 16.5 Recommended code directory v1

It is recommended that the first version of the code base be kept small and clear, and serve thesis experiments first, rather than making it a large platform at the beginning.

```text
cloudbrain-agent/
  pyproject.toml
  README.md
  configs/
    data/
      dev_mini.yaml
      main_bench.yaml
    experiments/
      direct_llm.yaml
      react.yaml
      cloudbrain_full.yaml
      ablation_no_verifier.yaml
    models/
      local_qwen3_14b.yaml
      api_gpt52.yaml
  data/
    cloudbrain_bench/
      schemas/
      splits/
      gold/
      metadata/
  src/
    cloudbrain/
      __init__.py
      ir/
        schema.py
        validators.py
        errors.py
      state/
        city_state.py
        airspace_state.py
        uav_state.py
        task_state.py
      tools/
        base.py
        registry.py
        city.py
        airspace.py
        scheduler.py
        planner.py
        verifier.py
        simulator.py
        risk.py
      agent/
        prompts.py
        llm_client.py
        runner.py
        repair.py
        memory.py
        traces.py
      data/
        generator.py
        osm_loader.py
        overture_loader.py
        weather_loader.py
        split.py
        quality.py
      eval/
        run.py
        metrics.py
        aggregate.py
        bootstrap.py
        error_analysis.py
      figures/
        main_results.py
        ablations.py
        repair_curve.py
      utils/
        io.py
        geometry.py
        hashing.py
        timing.py
  tests/
    test_ir_schema.py
    test_tool_registry.py
    test_planner.py
    test_verifier.py
    test_metrics.py
```

### 16.6 Pydantic schema code details

`LowAltitudeIR` should use strong type constraints and try to block errors after LLM output and before tool execution.

```python
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class Intent(str, Enum):
    delivery = "delivery"
    inspection = "inspection"
    patrol = "patrol"
    emergency = "emergency"
    return_home = "return"
    charge = "charge"
    monitoring = "monitoring"


class Priority(str, Enum):
    low = "low"
    normal = "normal"
    high = "high"
    critical = "critical"


class EntityRefs(BaseModel):
    origin: str | None = None
    destination: str | None = None
    candidate_uavs: list[str] = Field(default_factory=list)
    avoid_zones: list[str] = Field(default_factory=list)
    sensitive_zones: list[str] = Field(default_factory=list)
    handoff_points: list[str] = Field(default_factory=list)


class OperationConstraints(BaseModel):
    deadline_sec: int | None = Field(default=None, ge=1)
    altitude_min_m: float = Field(default=30.0, ge=0)
    altitude_max_m: float = Field(default=120.0, ge=0)
    min_separation_m: float = Field(default=10.0, ge=0)
    battery_reserve_ratio: float = Field(default=0.2, ge=0, le=1)
    max_risk_level: Literal["low", "medium", "high"] = "medium"
    corridor_capacity_required: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def check_altitude_range(self) -> "OperationConstraints":
        if self.altitude_min_m >= self.altitude_max_m:
            raise ValueError("altitude_min_m must be lower than altitude_max_m")
        return self


class ToolCallSpec(BaseModel):
    tool: str
    args: dict = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)


class VerificationSpecs(BaseModel):
    ltl: list[str] = Field(default_factory=list)
    stl: list[str] = Field(default_factory=list)
    program_rules: list[str] = Field(default_factory=list)


class LowAltitudeIR(BaseModel):
    task_id: str
    intent: Intent
    priority: Priority
    entities: EntityRefs
    constraints: OperationConstraints
    tool_plan: list[ToolCallSpec]
    verification_specs: VerificationSpecs
    fallback_policy: Literal[
        "ground_transfer",
        "wait",
        "human_confirm",
        "safe_refusal",
        "ground_transfer_or_human_confirm",
    ]
    explanation_plan: dict = Field(default_factory=dict)

    @field_validator("tool_plan")
    @classmethod
    def check_nonempty_tool_plan(cls, value: list[ToolCallSpec]) -> list[ToolCallSpec]:
        if not value:
            raise ValueError("tool_plan must contain at least one tool call")
        return value
```

Entity grounding should not be written in Pydantic but done separately as it relies on the current map and UAV state.

```python
def validate_entity_grounding(ir: LowAltitudeIR, state: SystemState) -> ValidationReport:
    errors: list[ValidationErrorItem] = []

    known_entities = state.known_entity_ids()
    known_uavs = state.known_uav_ids()

    for field_name in ["origin", "destination"]:
        value = getattr(ir.entities, field_name)
        if value is not None and value not in known_entities:
            errors.append(
                ValidationErrorItem(
                    stage="entity_grounding",
                    field=f"entities.{field_name}",
                    value=value,
                    error_type="unknown_entity",
                )
            )

    for uav_id in ir.entities.candidate_uavs:
        if uav_id not in known_uavs:
            errors.append(
                ValidationErrorItem(
                    stage="entity_grounding",
                    field="entities.candidate_uavs",
                    value=uav_id,
                    error_type="unknown_uav",
                )
            )

    return ValidationReport(valid=not errors, errors=errors)
```

### 16.7 ToolRegistry code detailsAll tools implement the same interface, making it easy to replace the deterministic / learned / external MCP version.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter
from typing import Any


class ToolErrorType(str, Enum):
    unknown_region = "unknown_region"
    restricted_airspace = "restricted_airspace"
    no_available_uav = "no_available_uav"
    no_path = "no_path"
    spec_violation = "spec_violation"
    sim_failure = "sim_failure"
    high_risk = "high_risk"
    invalid_arguments = "invalid_arguments"


class ToolError(BaseModel):
    type: ToolErrorType | str
    message: str
    recoverable: bool = True
    suggested_actions: list[str] = Field(default_factory=list)


class ToolResult(BaseModel):
    ok: bool
    tool: str
    request_id: str
    result: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)
    error: ToolError | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    latency_sec: float = 0.0


class BaseTool(ABC):
    name: str

    @abstractmethod
    def run(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        ...


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def execute(self, call: ToolCallSpec, context: ToolContext) -> ToolResult:
        start = perf_counter()
        if call.tool not in self._tools:
            return ToolResult(
                ok=False,
                tool=call.tool,
                request_id=context.next_request_id(call.tool),
                error=ToolError(
                    type="unknown_tool",
                    message=f"Tool {call.tool} is not registered.",
                    recoverable=True,
                    suggested_actions=["choose a registered tool"],
                ),
            )
        result = self._tools[call.tool].run(call.args, context)
        result.latency_sec = perf_counter() - start
        return result
```

### 16.8 Planner and simulator minimal code design

The first version of 3D A* only needs to support grid, NFZ mask, altitude range, and battery/length estimation.

```python
def plan_route_astar(
    grid: Grid3D,
    start: GridNode,
    goal: GridNode,
    avoid_mask: set[GridNode],
    altitude_min_layer: int,
    altitude_max_layer: int,
) -> RoutePlan:
    open_set = PriorityQueue()
    open_set.put((0.0, start))
    came_from: dict[GridNode, GridNode] = {}
    g_score = {start: 0.0}

    while not open_set.empty():
        _, current = open_set.get()
        if current == goal:
            return reconstruct_route(came_from, current)

        for nxt in grid.neighbors_26(current):
            if nxt in avoid_mask:
                continue
            if not altitude_min_layer <= nxt.z <= altitude_max_layer:
                continue
            tentative = g_score[current] + grid.edge_cost(current, nxt)
            if tentative < g_score.get(nxt, float("inf")):
                came_from[nxt] = current
                g_score[nxt] = tentative
                priority = tentative + euclidean_distance(nxt, goal)
                open_set.put((priority, nxt))

    return RoutePlan(ok=False, failure_type="no_path")
```

The lightweight simulator can be advanced in discrete time:

```python
def simulate_route(
    route: RoutePlan,
    scenario: ScenarioState,
    dt_sec: float = 1.0,
) -> SimulationResult:
    events: list[SimEvent] = []
    min_distance = float("inf")
    elapsed = 0.0

    for segment in route.segments:
        for pose in interpolate_segment(segment, dt_sec):
            elapsed += dt_sec
            distance = scenario.min_distance_to_obstacles(pose)
            min_distance = min(min_distance, distance)

            if scenario.inside_no_fly_zone(pose):
                events.append(SimEvent(time=elapsed, type="nfz_intrusion", pose=pose))
            if distance < scenario.min_separation_m:
                events.append(SimEvent(time=elapsed, type="separation_violation", pose=pose))
            if scenario.weather_risk_at(pose, elapsed) == "high":
                events.append(SimEvent(time=elapsed, type="weather_risk", pose=pose))

    return SimulationResult(
        success=not any(e.is_terminal for e in events),
        events=events,
        min_distance_m=min_distance,
        elapsed_sec=elapsed,
    )
```

### 16.9 Verifier minimal code design

The first version can divide LTL/STL into two layers: common rules are ensured by the program checker to ensure stability, and complex expressions are handed over to Spot/RTAMT.

```python
def verify_common_rules(
    trajectory: Trajectory,
    specs: VerificationSpecs,
    scenario: ScenarioState,
) -> VerificationResult:
    violations: list[Violation] = []

    if "G not_in_nfz" in specs.ltl:
        for t, pose in trajectory.iter_poses():
            if scenario.inside_no_fly_zone(pose):
                violations.append(
                    Violation(
                        rule="G not_in_nfz",
                        time_sec=t,
                        failure_type="nfz_intrusion",
                        recoverable=True,
                    )
                )

    for stl_spec in specs.stl:
        if stl_spec.startswith("distance_to_obstacle"):
            robustness = min(
                scenario.distance_to_nearest_obstacle(pose) - scenario.min_separation_m
                for _, pose in trajectory.iter_poses()
            )
            if robustness < 0:
                violations.append(
                    Violation(
                        rule=stl_spec,
                        time_sec=trajectory.time_of_min_distance(scenario),
                        failure_type="negative_robustness",
                        robustness=robustness,
                        recoverable=True,
                    )
                )

    return VerificationResult(pass_=not violations, violations=violations)
```

Counterexample compression should be short, do not put the entire track back into prompt:

```python
def compress_counterexample(verdict: VerificationResult) -> dict:
    first = next(iter(verdict.violations))
    return {
        "failure_type": first.failure_type,
        "violated_rule": first.rule,
        "time_sec": first.time_sec,
        "robustness": first.robustness,
        "suggested_repair": suggest_repair(first),
    }
```

### 16.10 Agent runner code details

`run_agent` needs to save the trace completely to facilitate reproducible experiments and error analysis.

```python
def run_agent(sample: Sample, model: ChatModel, tools: ToolRegistry, cfg: AgentConfig) -> AgentTrace:
    trace = AgentTrace(sample_id=sample.sample_id, method=cfg.method, model=model.name)
    context = build_context(sample, cfg)
    feedback: dict | None = None

    for repair_round in range(cfg.max_repair_rounds + 1):
        llm_output = model.generate(
            messages=build_messages(sample.instruction, context, feedback),
            temperature=cfg.temperature_for_round(repair_round),
            max_tokens=cfg.max_tokens,
        )
        trace.add_llm_call(llm_output, repair_round=repair_round)

        parse_report = parse_low_altitude_ir(llm_output.text)
        if not parse_report.ok:
            feedback = {"stage": "parse", "errors": parse_report.errors}
            trace.add_validation_failure(feedback)
            continue

        ir = parse_report.ir
        validation = validate_ir_all(ir, sample.state, tools)
        if not validation.valid:
            feedback = {"stage": "validation", "errors": validation.to_prompt_feedback()}
            trace.add_validation_failure(feedback)
            continue

        tool_trace = execute_tool_plan(ir, tools, sample.state)
        trace.add_tool_trace(tool_trace)

        if tool_trace.has_unrecoverable_error:
            trace.final_status = "safe_refusal"
            trace.final_reason = tool_trace.first_unrecoverable_error.type
            return trace

        verdict = verify_and_simulate(ir, tool_trace, sample.state)
        trace.add_verdict(verdict)

        if verdict.pass_:
            trace.final_status = "success"
            trace.final_decision = build_final_decision(ir, tool_trace, verdict)
            return trace

        feedback = compress_counterexample(verdict)

    trace.final_status = "human_confirm_or_safe_refusal"
    trace.final_reason = "max_repair_rounds_exceeded"
    return trace
```

Trace JSONL Each line is recommended to save:

```json
{
  "sample_id": "cb_000001",
  "method": "cloudbrain_full",
  "model": "qwen3-14b",
  "final_status": "success",
  "repair_rounds": 1,
  "llm_calls": [],
  "tool_calls": [],
  "validation_errors": [],
  "verifier_verdicts": [],
  "latency": {
    "total_sec": 18.4,
    "llm_sec": 13.2,
    "tool_sec": 4.1
  }
}
```

### 16.11 Evaluation code details

Metric should be automatically calculated from trace and gold to avoid manual subjective judgment.

```python
def compute_tool_call_accuracy(gold: list[ToolCallSpec], pred: list[ToolCallRecord]) -> float:
    if not gold:
        return 1.0 if not pred else 0.0
    matched = 0
    for gold_call, pred_call in zip(gold, pred):
        if gold_call.tool != pred_call.tool:
            continue
        if not args_compatible(gold_call.args, pred_call.args):
            continue
        matched += 1
    return matched / max(len(gold), len(pred), 1)


def compute_safety_violation(trace: AgentTrace) -> bool:
    if trace.final_status not in {"success", "safe_refusal"}:
        return True
    for verdict in trace.verifier_verdicts:
        if any(v.is_safety_critical for v in verdict.violations):
            return True
    for event in trace.sim_events:
        if event.type in {"collision", "nfz_intrusion", "separation_violation"}:
            return True
    return False


def evaluate_trace(sample: Sample, trace: AgentTrace) -> MetricRow:
    return MetricRow(
        sample_id=sample.sample_id,
        method=trace.method,
        model=trace.model,
        task_success=trace.final_status == "success",
        executable_decision=trace.has_executable_route(),
        safety_violation=compute_safety_violation(trace),
        hallucination=trace.has_unknown_entity_or_tool(),
        tool_call_accuracy=compute_tool_call_accuracy(sample.gold_tool_trace, trace.tool_calls),
        repair_success=trace.first_attempt_failed_and_later_succeeded(),
        latency_sec=trace.latency.total_sec,
    )
```

`pass^k` calculation method:

```python
def compute_pass_k(rows: list[MetricRow], k: int) -> float:
    by_sample = group_by(rows, key=lambda row: row.sample_id)
    success_count = 0
    for sample_id, sample_rows in by_sample.items():
        repeated = sorted(sample_rows, key=lambda row: row.repeat_id)[:k]
        if len(repeated) == k and all(row.task_success for row in repeated):
            success_count += 1
    return success_count / len(by_sample)
```

### 16.12 Unit Test Plan

The first stage of testing should cover the basic issues of "not writing wrong paper experiments":| Test files | Test content |
|----------|----------|
| `test_ir_schema.py` | enum, required fields, altitude range, power ratio |
| `test_entity_grounding.py` | No UAV/POI/NFZ can be caught |
| `test_tool_registry.py` | Unregistered tool, duplicate registration, error return format |
| `test_planner.py` | Simple reachability, NFZ blocking, no path reachability |
| `test_verifier.py` | NFZ, deadline, distance robustness |
| `test_simulator.py` | collision, near miss, weather risk |
| `test_agent_runner.py` | schema fail -> repair, verifier fail -> repair, unrecoverable -> refusal |
| `test_metrics.py` | TSR, SVR, TCA, RSR, pass^k calculation |

Recommended minimum test example:

```python
def test_invalid_altitude_range_is_rejected() -> None:
    with pytest.raises(ValueError):
        OperationConstraints(altitude_min_m=120, altitude_max_m=30)


def test_unknown_uav_is_entity_grounding_error(simple_state: SystemState) -> None:
    ir = make_valid_ir()
    ir.entities.candidate_uavs = ["uav_missing"]
    report = validate_entity_grounding(ir, simple_state)
    assert not report.valid
    assert next(iter(report.errors)).error_type == "unknown_uav"


def test_repair_success_metric() -> None:
    trace = make_trace(statuses=["validation_failed", "verifier_failed", "success"])
    assert trace.first_attempt_failed_and_later_succeeded()
```

### 16.13 First version implementation priority

Do not do all modules at the same time. It is recommended to sort by "minimum evidence chain of the paper":| Priority | Module | Why do it first |
|--------|------|------------|
| P0 | `LowAltitudeIR` schema + validators | Typed IR contributions cannot be proven without it |
| P0 | deterministic tools + trace logging | all experiments rely on |
| P0 | 3D A* + basic verifier | Support executable/safety indicators |
| P0 | direct/ReAct/CloudBrain baseline runner | Form the first main table |
| P1 | simulator stress seeds | Support safety-critical narrative |
| P1 | repair loop + counterexample compression | Core method of this article |
| P1 | metrics + aggregation | Prevent results from being reproducible |
| P2 | OSM/Overture/Open-Meteo loaders | Enhanced realism |
| P2 | MCP wrapper | Enhance the engineering narrative without blocking the paper |
| P3 | AirSim/Flightmare | Subsequent expansion, without blocking G1 |

### 16.14 MCP wrapper code details

The first version of the tools can be run through the Python registry first, and then the same batch of tools can be packaged into an MCP server. The advantage of this is that the paper experiments are not stuck in MCP engineering details, but the system narrative can naturally connect to the "cloud brain tool ecology".

The tool naming recommendations for MCP server are consistent with those of Python registry:| MCP tool | Python tool | Description |
|----------|-------------|------|
| `cloudbrain.query_city_state` | `query_city_state` | City entity and map state |
| `cloudbrain.query_airspace` | `query_airspace` | corridor, NFZ, height, capacity |
| `cloudbrain.assign_uav` | `assign_uav` | UAV-task assignment |
| `cloudbrain.plan_route` | `plan_route` | 3D route planner |
| `cloudbrain.verify_ltl_stl` | `verify_ltl_stl` | safety verifier |
| `cloudbrain.simulate_scenario` | `simulate_scenario` | stress simulator |
| `cloudbrain.risk_assess` | `risk_assess` | risk scoring |

MCP wrapper pseudo code:

```python
from mcp.server.fastmcp import FastMCP

from cloudbrain.tools.registry import build_default_registry
from cloudbrain.tools.base import ToolContext


mcp = FastMCP("cloudbrain-tools")
registry = build_default_registry()


@mcp.tool()
def query_airspace(region: str, altitude_min_m: float, altitude_max_m: float, time_sec: int) -> dict:
    context = ToolContext.from_active_scenario()
    result = registry.execute_by_name(
        "query_airspace",
        {
            "region": region,
            "altitude_range": [altitude_min_m, altitude_max_m],
            "time_sec": time_sec,
        },
        context,
    )
    return result.model_dump()


@mcp.tool()
def plan_route(start: str, goal: str, avoid_zones: list[str], altitude_min_m: float, altitude_max_m: float) -> dict:
    context = ToolContext.from_active_scenario()
    result = registry.execute_by_name(
        "plan_route",
        {
            "start": start,
            "goal": goal,
            "avoid_zones": avoid_zones,
            "altitude_range": [altitude_min_m, altitude_max_m],
        },
        context,
    )
    return result.model_dump()


if __name__ == "__main__":
    mcp.run()
```

MCP engineering constraints:

- MCP tool return value still uses `ToolResult` schema to avoid two sets of protocols.
- MCP server does not directly read and write experimental results, but only executes tools; traces are saved by agent runner.
- If the MCP call fails, the agent runner must be able to fallback to the Python registry to ensure that the experiment is not interrupted.
- It is recommended to use Python registry and MCP wrapper to store the main results of the thesis experiments in the system demonstration or appendix.

### 16.15 Data generator code details

The data generator needs to be deterministic, and the core inputs are seed and config.

```python
def generate_sample(seed: int, cfg: DataGenConfig) -> Sample:
    rng = np.random.default_rng(seed)
    context = load_context_bundle(cfg.context, rng)
    city = generate_city_layout(rng, cfg.city, context.map_snapshot)
    airspace = generate_airspace(city, rng, cfg.airspace, context.uasfm_snapshot)
    uavs = generate_uav_fleet(city, rng, cfg.fleet)
    task = generate_task(city, airspace, uavs, rng, cfg.task, context.weather_snapshot)

    gold_ir = build_gold_ir(task, city, airspace, uavs, cfg.rules)
    tool_context = ToolContext(
        city=city,
        airspace=airspace,
        uavs=uavs,
        weather=context.weather_snapshot,
        energy_calibration=context.energy_calibration,
    )
    gold_trace = execute_gold_tool_trace(gold_ir, tool_context)
    verdict = verify_and_simulate(gold_ir, gold_trace, tool_context)

    instruction = paraphrase_instruction(task, gold_ir, rng, cfg.language)

    return Sample(
        sample_id=f"cb_{seed:08d}",
        data_tier=context.data_tier,
        generation_seed=seed,
        city_id=context.city_id,
        scenario_type=task.scenario_type,
        instruction=instruction,
        source_provenance=context.provenance,
        real_context=context.real_context_metadata(),
        energy_calibration_version=context.energy_calibration.version,
        state=SystemState(city=city, airspace=airspace, uavs=uavs, tasks=[task]),
        gold_ir=gold_ir,
        gold_tool_trace=gold_trace,
        gold_verdict=verdict,
        label="SAT" if verdict.pass_ else "UNSAT",
        failure_modes=verdict.failure_modes,
    )
```

Split generation must ensure no information leakage:

```python
def assign_split(sample: Sample, cfg: SplitConfig) -> str:
    if sample.data_tier == "real_flight_calibrated":
        return "test_energy_calibrated"
    if sample.data_tier == "real_context" and sample.city_id in cfg.real_context_holdout_city_ids:
        return "test_real_context_city_b"
    if sample.data_tier == "real_context" and sample.has_weather_stress:
        return "test_real_weather_stress"
    if sample.data_tier == "real_context":
        return "test_real_context_city_a"
    if sample.city_id in cfg.unseen_city_ids:
        return "test_unseen_city"
    if sample.scenario_type in cfg.stress_scenario_types:
        return "test_stress"
    if sample.label == "UNSAT":
        return "test_unsat"
    bucket = stable_hash(sample.sample_id) % 100
    if bucket < 10:
        return "validation"
    if bucket < 20:
        return "test_seen_city"
    return "train_like"
```The generator must output split stats:

```json
{
  "split": "test_stress",
  "num_samples": 1000,
  "sat_rate": 0.74,
  "scenario_counts": {
    "emergency_delivery_with_nfz": 210,
    "corridor_congestion": 180
  },
  "avg_tool_trace_len": 5.8,
  "avg_constraints_per_task": 4.2
}
```

### 16.16 vLLM and local model startup solution

The native model recommends exposing the OpenAI-compatible endpoint via vLLM. This way `llm_client.py` only maintains one interface.

Example startup command:

```bash
vllm serve Qwen/Qwen3-14B \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen3-14b \
  --tensor-parallel-size 1 \
  --max-model-len 32768
```

DeepSeek repair specialist:

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --host 0.0.0.0 \
  --port 8001 \
  --served-model-name deepseek-r1-distill-qwen-14b \
  --tensor-parallel-size 1 \
  --max-model-len 32768
```

Unified client pseudocode:

```python
class ChatModel:
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def generate(self, messages: list[dict], temperature: float, max_tokens: int) -> LLMOutput:
        start = perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency = perf_counter() - start
        first_choice = next(iter(response.choices))
        content = first_choice.message.content or ""
        usage = response.usage
        return LLMOutput(text=content, latency_sec=latency, usage=usage.model_dump() if usage else {})
```

Running records must be written to `model_manifest.json`:

```json
{
  "model": "qwen3-14b",
  "provider": "local_vllm",
  "base_url": "http://localhost:8000/v1",
  "temperature": 0.0,
  "top_p": 1.0,
  "max_tokens": 4096,
  "system_fingerprint": "local",
  "prompt_version": "cloudbrain_v0.3"
}
```

### 16.17 Caching, Logging and Reproduction

In order to control API cost and experiment time, all LLM calls, tool calls, and verifier results are cached.

Cache key:

```python
def cache_key(prefix: str, payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"
```

Recommended cache directory:

```text
runs/
  20260520_cloudbrain_main/
    config.yaml
    model_manifest.json
    samples.jsonl
    traces.jsonl
    metrics.jsonl
    aggregate.csv
    cache/
      llm/
      tools/
      verifier/
    logs/
      run.log
      errors.log
```

Each trace must contain:

- `sample_id`
- `method`
- `model`
- `prompt_version`
- `config_hash`
- `random_seed`
- `repair_rounds`
- `final_status`
- `metric_row`
- `all_tool_results`
- `all_verifier_results`

### 16.18 CI and Quality Access Control

Even if there is only planning and experimental code in the first phase, quality gates must be set up:

```yaml
checks:
  formatting:
    - ruff format --check src tests
  lint:
    - ruff check src tests
  typing:
    - mypy src
  unit_tests:
    - pytest tests -q
  smoke:
    - python -m cloudbrain.data.generate --config configs/data/dev_mini.yaml --limit 20
    - python -m cloudbrain.eval.run --split dev_mini --method cloudbrain_full --limit 5 --mock-llm
```

Smoke test only uses mock LLM to ensure that CI does not depend on GPU or API key. Mock LLM returns fixed IR for validation toolchain, metrics and trace logging.

### 16.19 Experiment matrix configuration

In the end, the main experiment ran at least this matrix:| Split | Method | Model | Repeat |
|-------|--------|-------|--------|
| validation | direct_llm/react/cloudbrain_full | qwen3-14b | 1 |
| test_seen_city | all main baselines | qwen3-14b | 3 |
| test_unseen_city | all main baselines | qwen3-14b | 3 |
| test_stress | all main baselines | qwen3-14b | 3 |
| test_unsat | direct_llm/react/cloudbrain_full | qwen3-14b | 3 |
| test_seen_city | cloudbrain_full | qwen3-8b / qwen3-32b / deepseek-repair / GPT upper bound | 3 |

Automatically generate experimental tasks:

```python
def build_experiment_matrix(cfg: MatrixConfig) -> list[ExperimentJob]:
    jobs = []
    for split in cfg.splits:
        for method in cfg.methods_for_split(split):
            for model in cfg.models_for_method(method):
                for repeat_id in range(cfg.repeats):
                    jobs.append(
                        ExperimentJob(
                            split=split,
                            method=method,
                            model=model,
                            repeat_id=repeat_id,
                            seed=stable_seed(split, method, model, repeat_id),
                        )
                    )
    return jobs
```

### 16.20 Engineering materials that should be included in the appendix of the paper

To improve reproducibility, the G1 appendix contains at least:| Appendix | Contents |
|------|------|
| A | Full `LowAltitudeIR` JSON schema |
| B | Tool registry schema and error taxonomy |
| C | Data generation config and scenario taxonomy |
| D | Prompt templates for each baseline |
| E | Full metric definitions and bootstrap procedure |
| F | Extra ablation and per-scenario results |
| G | Failure case visualizations |
| H | Compute budget, model versions, cache policy |

---

## 17. Risks and Alternatives

### 17.1 Risk: The effect is not obvious

Alternative: Increase the stress/UNSAT ratio to make the value of verifier repair more prominent; and report the benefits under different task difficulties.

### 17.2 Risk: Local model is too weak

Alternative: Qwen3-32B is used for the main experiment, and Qwen3-14B is used as the deployable version; GPT-5.2 is only used for the upper bound. DeepSeek-R1-Distill can also be used as a repair-only specialist.

### 17.3 Risk: Data perceived as too synthetic

Alternative: Split the main experiment into three layers: `Synthetic-Controlled`, `Real-Context`, and `Real-Flight-Calibrated`, which report program true values, real city/airspace/weather grounding, and real flight energy consumption calibration respectively. The paper description is clearly written as benchmark + method, and does not exaggerate UASFM, METAR or DJI M100 data into real commercial fleet operations.

### 17.4 Risk: MCP implementation slows down progressAlternative: The first version of the tool first uses Python function registry, and describes the interface as MCP-compatible when submitting. Real MCP server release source appendix or subsequent version.

### 17.5 Risk: Formal verification is too heavy

Alternative: LTL first performs discrete event constraints, and STL first performs four types of continuous constraints: distance, height, deadline, and battery. Do not initially cover all low-altitude regulations.

### 17.6 Risk: Agent benchmark contribution is not general enough

Alternative: Split the tasks of CloudBrain-Bench into general agent evaluation dimensions: tool selection, argument grounding, state dependency, policy compliance, counterexample repair, `pass^k` consistency. This way, even reviewers unfamiliar with low-altitude transportation can understand its contribution to safety-critical agent evaluation.

---

## 18. References

[1] AAAI. “AAAI-26 Main Technical Track: Call for Papers.” URL: <https://aaai.org/conference/aaai/aaai-26/main-technical-track-call/>

[2] IJCAI-ECAI 2026. “Call for Papers — AI and Robotics Special Track.” URL: <https://2026.ijcai.org/ijcai-ecai-2026-call-for-papers-ai-and-robotics/>[3] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. "LoRA: Low-Rank Adaptation of Large Language Models." *International Conference on Learning Representations (ICLR)*, 2022. URL: <https://openreview.net/forum?id=nZeVKeeFYf9>

[4] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. “QLoRA: Efficient Finetuning of Quantized LLMs.” *Advances in Neural Information Processing Systems 36 (NeurIPS)*, 2023. URL: <https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html>[5] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. “Direct Preference Optimization: Your Language Model is Secretly a Reward Model.” *Advances in Neural Information Processing Systems 36 (NeurIPS)*, 2023. URL: <https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html>

[6] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. "ReAct: Synergizing Reasoning and Acting in Language Models." *International Conference on Learning Representations (ICLR)*, 2023. URL: <https://openreview.net/forum?id=WE_vluYUL-X>[7] Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu, and Maosong Sun. “ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs.” *International Conference on Learning Representations (ICLR)*, 2024. URL: <https://openreview.net/forum?id=dHng2O0Jjr>

[8] Bo Liu, Yuqian Jiang, Xiaohan Zhang, Qiang Liu, Shiqi Zhang, Joydeep Biswas, and Peter Stone. "LLM+P: Empowering Large Language Models with Optimal Planning Proficiency." arXiv:2304.11477, 2023. URL: <https://arxiv.org/abs/2304.11477>[9] Siyao Zhang, Daocheng Fu, Wenzhe Liang, Zhao Zhang, Bin Yu, Pinlong Cai, and Baozhen Yao. “TrafficGPT: Viewing, Processing and Interacting with Traffic Foundation Models.” *Transport Policy*, 150:95-105, 2024. DOI: 10.1016/j.tranpol.2024.03.006. URL: <https://www.sciencedirect.com/science/article/pii/S0967070X24000726>

[10] OpenAI. “GPT-5.2 Model.” *OpenAI API Documentation*, 2025. URL: <https://platform.openai.com/docs/models/gpt-5.2>

[11] Qwen Team. “Qwen3 Technical Report.” arXiv:2505.09388, 2025. URL: <https://arxiv.org/abs/2505.09388>

[12] DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv:2501.12948, 2025. URL: <https://arxiv.org/abs/2501.12948>[13] Sebastian Wandelt, Changhong Zheng, Shuang Wang, Yucheng Liu, and Xiaoqian Sun. "Large Language Models for Intelligent Transportation: A Review of the State of the Art and Challenges." *Applied Sciences*, 14(17):7455, 2024. DOI: 10.3390/app14177455. URL: <https://www.mdpi.com/2076-3417/14/17/7455>

[14] Doaa Mahmud, Hadeel Hajmohamed, Shamma Almentheri, Shamma Alqaydi, Lameya Aldhaheri, Ruhul Amin Khalil, and Nasir Saeed. "Integrating LLMs With ITS: Recent Advances, Potentials, Challenges, and Future Directions." *IEEE Transactions on Intelligent Transportation Systems*, 26(5):5674-5709, 2025. DOI: 10.1109/TITS.2025.3528116. URL: <https://ieeexplore.ieee.org/document/10851302>

[15] Zhonghang Li, Lianghao Xia, Jiabin Tang, Yong Xu, Lei Shi, Long Xia, Dawei Yin, and Chao Huang. "UrbanGPT: Spatio-Temporal Large Language Models." arXiv:2403.00813, 2024. URL: <https://arxiv.org/abs/2403.00813>[16] Yuan Yuan, Jingtao Ding, Jie Feng, Depeng Jin, and Yong Li. “UniST: A Prompt-Empowered Universal Model for Urban Spatio-Temporal Prediction.” *Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)*, 2024. DOI: 10.1145/3637528.3671662. URL: <https://arxiv.org/abs/2402.11838>

[17] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. “Toolformer: Language Models Can Teach Themselves to Use Tools.” *Advances in Neural Information Processing Systems 36 (NeurIPS)*, 2023. URL: <https://proceedings.neurips.cc/paper_files/paper/2023/hash/d842425e4bf79ba039352da0f658a906-Abstract-Conference.html>

[18] OpenAI. “Model Context Protocol (MCP) — OpenAI Agents SDK.” URL: <https://openai.github.io/openai-agents-js/guides/mcp/>[19] OpenAI. “Tools — OpenAI Agents SDK.” URL: <https://openai.github.io/openai-agents-js/guides/tools/>

[20] Karthik Valmeekam, Matthew Marquez, Alberto Olmo, Sarath Sreedharan, and Subbarao Kambhampati. “PlanBench: An Extensible Benchmark for Evaluating Large Language Models on Planning and Reasoning about Change.” *Advances in Neural Information Processing Systems 36 (NeurIPS) Datasets and Benchmarks Track*, 2023. URL: <https://openreview.net/forum?id=YXogl4uQUO>

[21] Bo Liu, Yuqian Jiang, Xiaohan Zhang, Qiang Liu, Shiqi Zhang, Joydeep Biswas, and Peter Stone. "Lang2LTL: Translating Natural Language Commands to Temporal Specification with Large Language Models." *Conference on Robot Learning (CoRL)*, PMLR 229, 2023. URL: <https://proceedings.mlr.press/v229/liu23d.html>[22] Behrad Rabiei, Mahesh Kumar A. R., Zhirui Dai, Surya L. S. R. Pilla, Qiyue Dong, and Nikolay Atanasov. “LTLCodeGen: Code Generation of Syntactically Correct Temporal Logic for Robot Task Planning.” arXiv:2503.07902, 2025. URL: <https://arxiv.org/abs/2503.07902>

[23] Jun Wang, David Smith Sundarsingh, Jyotirmoy V. Deshmukh, and Yiannis Kantaros. “ConformalNL2LTL: Translating Natural Language Instructions into Temporal Logic Formulas with Conformal Correctness Guarantees.” arXiv:2504.21022, 2025. URL: <https://arxiv.org/abs/2504.21022>

[24] Alexandre Duret-Lutz, Alexandre Lewkowicz, Amaury Fauchille, Thibault Michaud, Etienne Renault, and Laurent Xu. “Spot 2.0: A Framework for LTL and ω-Automata Manipulation.” *International Symposium on Automated Technology for Verification and Analysis (ATVA)*, 2016. URL: <https://spot.lre.epita.fr/>[25] Bardh Hoxha, Houssam Abbas, and Georgios Fainekos. “RTAMT: Online Robustness Monitors from STL.” arXiv:2005.11827, 2020. URL: <https://arxiv.org/abs/2005.11827>

[26] Federal Aviation Administration. “Unmanned Aircraft System Traffic Management (UTM).” URL: <https://www.faa.gov/uas/advanced_operations/traffic_management>

[27] Federal Aviation Administration. “UAS Facility Maps.” URL: <https://www.faa.gov/uas/commercial_operators/uas_facility_maps>

[28] OpenStreetMap / Overpass API. “OpenStreetMap and the Overpass API.” URL: <https://dev.overpass-api.de/overpass-doc/en/preface/preface.html>

[29] New York City Taxi and Limousine Commission. “TLC Trip Record Data.” URL: <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page>

[30] Eclipse SUMO. “SUMO Documentation.” URL: <https://sumo.dlr.de/docs/index.html>[31] Shital Shah, Debadeepta Dey, Chris Lovett, and Ashish Kapoor. “AirSim: High-Fidelity Visual and Physical Simulation for Autonomous Vehicles.” *Field and Service Robotics*, Springer Proceedings in Advanced Robotics, 2017; arXiv:1705.05065. URL: <https://arxiv.org/abs/1705.05065>

[32] Yunlong Song, Selim Naji, Elia Kaufmann, Antonio Loquercio, and Davide Scaramuzza. “Flightmare: A Flexible Quadrotor Simulator.” *Conference on Robot Learning (CoRL)*, PMLR 155, 2021. URL: <https://proceedings.mlr.press/v155/song21a.html>

[33] vLLM Team. “OpenAI-Compatible Server.” *vLLM Documentation*. URL: <https://docs.vllm.ai/serving/openai_compatible_server.html>[34] Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, and others. "AgentBench: Evaluating LLMs as Agents." *International Conference on Learning Representations (ICLR)*, 2024. URL: <https://openreview.net/forum?id=zAdUB0aCTQ>

[35] Ivan Ortega, Fanjia Yan, Huanzhi Mao, Charlie Cheng-Jie Ji, Tianjun Zhang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. “Berkeley Function-Calling Leaderboard.” UC Berkeley Sky Computing Lab project page, 2024/2025. URL: <https://sky.cs.berkeley.edu/project/berkeley-function-calling-leaderboard/>

[36] Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. “$\tau$-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.” arXiv:2406.12045, 2024. URL: <https://arxiv.org/abs/2406.12045>[37] Jiarui Lu, Thomas Holleis, Yizhe Zhang, Bernhard Aumayer, Feng Nan, Felix Bai, Shuang Ma, Shen Ma, Mengyu Li, Guoli Yin, Zirui Wang, and Ruoming Pang. "ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities." arXiv:2408.04682, revised 2025. URL: <https://arxiv.org/abs/2408.04682>

[38] Lynne Martin, Cynthia Wolter, Kimberly Jobe, Mariah Manzano, Stefan Bladin, Michele Cencetti, Lauren Claudatos, Joey Mercer, and Jeffrey Homola. “TCL4 UTM (UAS Traffic Management) Nevada 2019 Flight Tests, Airspace Operations Laboratory (AOL) Report.” NASA Technical Memorandum NASA/TM-2020-220516, 2020. URL: <https://ntrs.nasa.gov/citations/20205003361>

[39] EUROCONTROL. “CORUS-XUAM: Concept of Operations for European UTM Systems — Extension for Urban Air Mobility.” Project page, 2023. URL: <https://www.eurocontrol.int/project/corus-xuam>[40] Marc Brittain, Luis E. Alvarez, Kara Breeden, and Ian Jessen. “AAM-Gym: Artificial Intelligence Testbed for Advanced Air Mobility.” *IEEE/AIAA Digital Avionics Systems Conference (DASC)*, 2022; arXiv:2206.04513. URL: <https://arxiv.org/abs/2206.04513>

[41] Overture Maps Foundation. “Overture Maps Documentation: Places, Buildings, and Transportation Data.” URL: <https://docs.overturemaps.org/>

[42] Open-Meteo. “Weather Forecast API and Historical Forecast API Documentation.” URL: <https://open-meteo.com/en/docs>

[43] Federal Aviation Administration. “UAS Data Delivery System Data Dictionary.” PDF, 2022. URL: <https://www.faa.gov/sites/faa.gov/files/2022-08/UAS_Data_Delivery_System_Data_Dictionary.pdf>

[44] Aviation Weather Center. “Data API.” National Oceanic and Atmospheric Administration, documentation page. URL: <https://aviationweather.gov/data/api/>[45] Thiago A. Rodrigues, Jay Patrikar, Arnav Choudhry, Jacob Feldgoise, Vaibhav Arcot, Aradhana Gahlaut, Sophia Lau, Brady Moon, Bastian Wagner, H. Scott Matthews, Sebastian Scherer, and Constantine Samaras. "In-flight positional and energy use data set of a DJI Matrice 100 quadcopter for small package delivery." *Scientific Data*, 8:155, 2021. DOI: 10.1038/s41597-021-00930-x. URL: <https://www.nature.com/articles/s41597-021-00930-x>

[46] Federal Aviation Administration. “Package Delivery by Drone (Part 135).” URL: <https://www.faa.gov/uas/advanced_operations/package_delivery_drone>

[47] U.S. Government Accountability Office. “Drones: Actions Needed to Better Support Remote Identification in the National Airspace.” GAO-24-106158, 2024. URL: <https://www.gao.gov/products/gao-24-106158>

---

## Appendix: This execution plan

### A. Do it immediately1. Create `LowAltitudeIR v0.1` schema.
2. Implement 6 deterministic tools: city, airspace, scheduler, planner, verifier, and simulator.
3. Generate 200 dev-mini samples.
4. Run four baselines: direct LLM, JSON-only, ReAct, and CloudBrain-Agent without repair.

### B. Passing Criteria for the First Round of Experiments

If the following conditions are met on dev-mini, the complete benchmark will be entered:

- The tool-call accuracy of CloudBrain-Agent full exceeds ReAct baseline;
- safety violation rate is lower than direct LLM;
- The repair loop can repair at least some verifier failures;
- The average running time of each task does not exceed an acceptable threshold, such as less than 30 seconds for a local 14B model.

### C. Next article connection

The tool traces, repair traces, failure cases and human review data generated by G1 will directly become the SFT/DPO data of G2 LowAltitudeGPT. In other words, G1 is not only a paper, but also a data factory for vertical model training data.