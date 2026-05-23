---
title: "Paper G Planning v1: LLM Agent and model fine-tuning route in low-altitude traffic cloud brain"
description: "Plan how to train or fine-tune LLM to make it a verifiable agent in the low-altitude traffic cloud brain, and form the first AAAI/IJCAI conference paper, follow-up transportation journals and general embodied agent transformation route."
pubDate: 2026-05-20
updatedDate: 2026-05-23
tags: ["Paper G", "Low altitude traffic cloud brain", "LLM Agent", "Model fine-tuning", "Tool Use", "AAAI", "IJCAI", "UAV", "AGI"]
category: Tech
---

# Paper G Planning v1: LLM Agent and model fine-tuning route in low-altitude traffic cloud brain

> Overall judgment: This route should not be a "large low-altitude traffic chat model" first, but should be a **verifiable LLM Agent in the low-altitude traffic cloud brain**.  
> Prioritize AAAI / IJCAI for the first article: Put LLM in the position of "task understanding, tool invocation, planning and repair, verification closed loop, and scheduling explanation" instead of directly committing to training a large-scale foundation model.

---

## 1. Overall judgment: Why build the Agent cloud brain first instead of directly training the large model?

If you directly write "fine-tuning a low-altitude traffic LLM", conference reviewers are likely to ask three questions:

1. **What is the contribution of the model? **
   LoRA / SFT / DPO itself is already a standard training process [14] [15] [16]. Simply replacing the data with low-altitude traffic corpus is difficult to support the AAAI / IJCAI main conference.

2. **Why is LLM more necessary than existing scheduling/planning models? **
   Low-altitude traffic operation involves scheduling, path planning, risk assessment, formal verification and simulation feedback. The advantage of LLM is not to replace these models, but to decompose complex tasks into callable tool chains.

3. **How ​​to ensure safety? **
   The low-altitude traffic cloud brain is a safety-critical system. Directly outputting control actions from LLM runs the risk of hallucination and non-verification. The first paper must put the verifier, simulator and risk estimator into a closed loop.

Therefore, it is not recommended that the first article of Paper G be called “LowAltitudeGPT”. A better first article would be:

> **CloudBrain-Agent: Tool-Augmented and Verification-Guided LLM Agents for Low-Altitude Traffic Operation**

Its core contribution is not "the model becomes smarter", but:

- Construct an agentic decision pipeline for low-altitude traffic cloud brain;
- Let LLM learn to call low-altitude vehicles;
- Use verifiers and emulators to correct errors;
- Output executable, interpretable, and auditable scheduling/planning decisions.This is close to the idea of ​​TrafficGPT: TrafficGPT has pointed out that LLM itself is difficult to handle traffic numerical data and simulation interaction, so it needs to be combined with traffic foundation models [1]. The difference in Paper G is that we have expanded the object from ground transportation to low-altitude transportation, and further added UAV status, airspace constraints, formal verification and safety closed loop.

From a broader review of traffic intelligence, LLM has been discussed as a semantic interface, reasoning module and traffic decision-making auxiliary component in ITS [2] [3]; UrbanGPT and UniST illustrate that urban spatio-temporal prediction is transitioning to a spatio-temporal foundation model [4] [5]. Paper G does not directly repeat these directions, but combines "urban spatiotemporal intelligence + UAV operating tools + verifiable agents" into a low-altitude traffic cloud brain.

### 1.1 2026-05-22 Writing calibration: G1 is an AI agent paper, and journal expansion requires a complete transportation system narrative.

Paper G could easily be written off as a "low-altitude traffic large model story." This route requires distinguishing between two evaluation criteria:

| Stages | Goals | Main review logic | Mistakes that cannot be made |
|------|------|--------------|--------------|
| G1 AAAI/IJCAI | Verifiable LLM Agent method | tool use, planning, verification, benchmark, reproducibility | Sacrificing method clarity for traffic narrative, or writing the agent as a platform display |
| G2 T-ITS/T-IV | LLM fine-tuning in the low-altitude transportation field | Domain data, deployment reproducibility, and traffic decision-making assistance capabilities | Only general LoRA/SFT, no transportation chain and safety indicators |
| G3 AAMAS/T-ITS | Multi-agent cloud-brain collaboration | Multi-role collaboration, communication, conflict handling, human-machine collaboration | Multi-agent is just multiple prompts, without system status and responsibility boundaries |
| Journal extended version | Significance of transportation system operation | Safety, efficiency, capacity, delay, resource utilization, management inspiration | Only report accuracy/tool-call success, do not answer traffic questions |Therefore, the main line of G1 is still strong AI methods: typed IR, tool-use, verifier repair, and stateful evaluation.
However, all low-altitude traffic-related indicators must be retained from the beginning to facilitate subsequent expansion to T-ITS:

- Security: LoWC/NMAC proxy, no-fly-zone violation, battery reserve violation.
- Efficiency: delay, extra distance, energy, throughput, runtime.
- Operation management: safe refusal rate, human confirmation rate, ambiguous-task handling.
- Robustness: communication loss, weather disturbance, non-cooperative UAV, unseen city/topology.
- System enlightenment: Under what conditions the LLM agent needs to exit to the deterministic solver or human supervisor.

### 1.2 2026-05-23 Arrangement: The order of G routes

Paper G is an umbrella roadmap, and what is really to be completed in the near future is **G1 CloudBrain-Agent**. Currently, the fastest and most submittable route is not to train a large vertical model first, but to use a general strong model + typed IR + tool chain + verifier + simulator feedback to form a reproducible closed loop. The vertical model training is placed in G2, and the tool-call traces, repair traces and failure cases generated by G1 are used as data.| Stage | Whether to train the model | Recommended model/deployment | Goal |
|------|--------------|---------------|------|
| G1 now | Not as a main contributor to training | Local vLLM runs Qwen / DeepSeek, API model does teacher / upper bound | Prove that agent tool calling, verification repair and low-altitude task benchmark are effective |
| G2 next | LoRA / SFT / DPO | Fine-tuning Qwen / Llama / DeepSeek series with G1 traces | Forming LowAltitudeGPT domain cognitive module |
| G3 later | Optional multi-agent trajectory distillation | Multi-role agent + shared memory + verifier | Research airspace monitoring, scheduling, risk, emergency, human-machine collaboration |
| G4 long-term | Multi-modal / world model / VLA | Depends on data and computing power | Migrating to embodied traffic intelligence |

Deployment strategy recommendations are as follows:

- **Local open source model for main experiment**: reproducible, controllable cost, easy to report latency and hardware conditions; it is recommended to use vLLM / llama.cpp as the inference service.
- **API model as teacher and upper-bound**: used to generate high-quality initial samples, difficult example repair demonstrations and upper-bound baseline; API results and local model results should be reported separately in the paper.
- **MCP first develops the interface style, not productization first**: The first version first implements the Python tool registry and JSON schema; after the tool is stable, it will be packaged into an MCP-compatible server to avoid pushing the engineering complexity into the main line of the paper.
- **Vertical model training does not grab the main line of G1**: G1's contribution is the agent architecture and verification closed loop; G2 only distills the running trajectory into the local model.

This sequence can quickly form a closed loop that can be submitted: first let the system run, evaluate, and explain failures, and then decide which capabilities are worthy of fine-tuning into the model.

---

## 2. System definition of low-altitude traffic cloud brainThe "low-altitude traffic cloud brain" in this article is not a general intelligent platform, but a **cognitive operation layer** for urban low-altitude UAV operation:

```text
Human / operator instruction
  -> CloudBrain LLM Agent
  -> LowAltitudeIR
  -> traffic tools / UAV tools / verifier / simulator
  -> safe decision proposal
  -> human approval or autonomous execution
```

### 2.1 Input

The low-altitude traffic cloud brain receives multi-source status:

| Input | Example |
|------|------|
| Natural Language Task | "Prioritize emergency deliveries near hospitals and avoid schools and no-fly zones." |
| UAV status | Position, power, load, mission status, communication status |
| Airspace status | corridor capacity, no-fly zones, temporary controls, weather, wind fields |
| Transportation needs | Delivery orders, inspection tasks, emergencies, passenger/cargo priority |
| Scene status | Safety-critical scenarios, accident scenarios, coverage holes of Paper F |
| Formal constraints | LTL/STL safety rules, time windows, minimum heights, minimum intervals |

### 2.2 Output

The cloud brain does not directly output "flight actions", but outputs auditable intermediate decisions:

| Output | Example |
|------|------|
| LowAltitudeIR | Structured tasks, entities, constraints, tool call plans |
| Tool calling sequence | Query airspace, call scheduler, call path planner, run verifier |
| Scheduling recommendations | Which UAV performs which task and whether to trigger ground fallback |
| Security Diagnosis | Which constraints may be violated and whether manual confirmation is required |
| Explanation text | Explain in natural language why it is scheduled this way |

### 2.3 Cloud Brain is not an end-to-end controller

The boundaries of the low-altitude traffic cloud brain must be clearly written:

- LLM does semantic understanding, task decomposition, tool selection, interpretation and repair.
- The scheduler performs fleet assignment and resource optimization, corresponding to Paper B.
- The validator does LTL/STL security checks, corresponding to Paper E.
- Scenario simulator and risk generator provide stress testing, corresponding to Paper F.
- The trajectory controller is still executed by the traditional planning/MPC/safety control module.

This avoids the reviewer's doubt that "LLM control of UAV is unsafe".

---

## 3. Overview of research routes: from domain LLM to general embodied agent

Paper G can be divided into 4 stages.| Stages | Thesis | Objectives | Key Questions |
|------|------|------|----------|
| G1 | CloudBrain-Agent | AAAI / IJCAI | LLM How to reliably call tools in low-altitude traffic cloud brain and pass verification closed-loop repair |
| G2 | LowAltitudeGPT | T-ITS / T-IV | How to fine-tune a local open source LLM to become a low-altitude traffic decision-making cognitive module |
| G3 | Multi-Agent Cloud Brain | AAMAS / IJCAI / T-ITS | How multiple full-time agents collaborate to manage low-altitude traffic |
| G4 | World-Model / VLA Extension | Long-term route | How to shift from domain agent to embodied general intelligence |

The recommended sequence is **G1 -> G2 -> G3 -> G4**.

G1 first solves "whether the system can run, whether it can be safely closed-loop, and whether it can hold meetings." G2 then distills the agent trajectory into a domain model. G3 uses multi-agent collaboration. The AGI transformation is only discussed in G4 and will not be exaggerated in the first article.

---

## 4. Paper G1: CloudBrain-Agent, the first conference paper for AAAI/IJCAI

### 4.1 Question

**CloudBrain-Agent: Tool-Augmented and Verification-Guided LLM Agents for Low-Altitude Traffic Operation**

### 4.2 Target Meeting

First pitch: AAAI/IJCAI.  
Alternatives: AAMAS, ICRA/IROS workshop, T-ITS fast journal extension.AAAI-26 Main Technical Track encourages work across AI technology directions and important application areas such as transportation. The main text is limited to 7 pages of technical content and requires a reproducibility checklist [34]. The AI ​​and Robotics special track of IJCAI-ECAI 2026 explicitly focuses on robot agents, generative AI, robot control, structured modeling, reasoning and how to perform/avoid the consequences of actions [35]. Therefore G1 should be written as an AI agent / planning / tool-use / verification paper rather than a system engineering demonstration.

### 4.3 Core Issues

G1 wants to answer:

> Given a low-altitude traffic operation task, how to make the LLM agent reliably understand the task, select tools, call the scheduling/planning/verification module, and fix errors under counterexample feedback, thereby outputting safe, executable, and explainable cloud brain decisions?

### 4.4 Method

Proposed **CloudBrain-Agent**, including five modules:

| Module | Function |
|------|------|
| LowAltitudeIR parser | Convert natural language tasks and system states into structured representations |
| Tool planner | Planning tool calling sequence |
| Tool executor | Call scheduler, path planner, verifier, simulator, risk assessor |
| Verifier feedback loop | Convert failed tool calls, unsatisfiable constraints, and STL robustness failures into repair feedback |
| Safety memory | Save known hazard scenarios, failure cases, manual decisions and rule constraints |

Behavior form of CloudBrain-Agent:

```text
Observe -> Think -> Select Tool -> Execute -> Verify -> Repair -> Decide
```

This inherits ReAct’s reasoning-action loop [6], but adds two low-altitude traffic-specific mechanisms:

1. **Tool calls must be type-safe**: Each tool input and output is checked against the `LowAltitudeIR` schema.
2. **Decisions must pass verifier**: Any scheduling or path recommendations must undergo security verification or simulation stress testing.### 4.5 LowAltitudeIR

LowAltitudeIR is the key public interface of G1:

```json
{
  "intent": "emergency_delivery",
  "entities": ["uav_12", "hospital_zone", "landing_pad_A"],
  "constraints": {
    "avoid": ["school_zone", "temporary_no_fly_zone"],
    "deadline_sec": 600,
    "min_obstacle_distance_m": 10,
    "altitude_range_m": [30, 120]
  },
  "tool_plan": [
    "query_airspace",
    "assign_uav",
    "plan_route",
    "verify_stl",
    "simulate_scenario"
  ],
  "fallback": "ground_vehicle_transfer_if_unreachable"
}
```

LowAltitudeIR should be compatible with three existing paper lines:

- Paper B: Task queue, UAV allocation, vertiport / charging / corridor resources.
- Paper E: TaskIR, LTL/STL, verification and error correction; related citation bases include Lang2LTL, LTLCodeGen and ConformalNL2LTL [20] [21] [22].
- Paper F: Scene generation, coverage holes, dangerous scene stress testing.

### 4.6 Tool Collection

G1's tools do not need to be built on real systems at the beginning. You can first build reproducible experimental tools:

| Tool | Input | Output |
|------|------|------|
| `query_airspace` | Region, time, mission type | corridor, no-fly zone, weather, capacity |
| `assign_uav` | Task, UAV status, priority | UAV-task assignment |
| `plan_route` | start, end, constraint | path or `UNREACHABLE` |
| `verify_ltl_stl` | Task specification, trajectory | pass / fail / counterexample |
| `simulate_scenario` | scenario seed, strategy | success, collision, delay, risk |
| `risk_assess` | Tasks and scenarios | Risk level, main constraints |
| `explain_decision` | Decision trajectory | Human-readable explanation |

### 4.7 Baselines| Baseline | Description |
|----------|------|
| Direct LLM decision | LLM directly gives scheduling/path suggestions |
| Prompt-only ReAct | ReAct style tool call, but without type constraints and verifier [6] |
| Toolformer / ToolLLM-style tool-use | Learn to call tools, but do not perform low-level security verification [7] [8] |
| TrafficGPT-style orchestration | LLM calls the traffic model, but without UAV constraints and formal verification [1] |
| LLM+P / classical planner | LLM conversion problem, solved by external planner [10] |
| VERA-UAV only | Only language to specification verification, no cloud brain multi-tool scheduling |
| CloudBrain-Agent full | LowAltitudeIR + tool-use + verifier + simulator feedback |

PlanBench and subsequent critical studies on LLM planning capabilities have shown that simply letting LLM plan verbally is not reliable and that external planners, constraint checks, and reproducible experimental tasks must be introduced [11] [12]. At the same time, AerialVLN and realistic UAV-VLN work can be used as a benchmark source for low-altitude visual language navigation [23] [24]; DriveLM, LMDrive, DriveVLM and LaMPilot can be used as a horizontal reference for autonomous driving VLM/LLM benchmark and closed-loop decision-making paradigm [25] [26] [27] [28].

### 4.8 Evaluation indicators| Indicator | Meaning |
|------|------|
| Task success rate | Cloud brain task completion ratio |
| Tool-call accuracy | Whether the tool selection and parameters are correct |
| Executable decision rate | Whether the output can be executed by the scheduler/planner |
| Safety violation rate | Whether the no-fly zone, distance, altitude, deadline is violated |
| Hallucination rate | Whether to reference a non-existent entity, tool or state |
| Repair success rate | Whether it can be repaired after verification fails |
| Simulator stress pass rate | Pass rate in Paper F hazardous scenario |
| Latency | Single task decision time |
| Generalization | Performance on unseen cities/unseen tasks/unseen tool combinations |

### 4.9 Expected innovation points

1. Propose `LowAltitudeIR` and typed tool-use agent architecture for low-altitude traffic cloud brain.
2. Unify scheduling, path planning, formal verification and scenario simulation into the LLM agent decision-making closed loop.
3. Propose verification-guided repair so that LLM no longer relies solely on prompt retry.
4. Build a low-altitude traffic cloud brain benchmark, covering task decomposition, tool invocation, scheduling, verification and interpretation.

---

## 5. Paper G2: LowAltitudeGPT, LLM fine-tuning in the field of low-altitude traffic

### 5.1 Question

**LowAltitudeGPT: Instruction Tuning LLMs for Low-Altitude Traffic Decision Support**

### 5.2 Goals

G2 is the model fine-tuning paper. The goal is to distill the agent running trajectory, artificial rules, simulation feedback and verification and repair data in G1 into a local open source model, so that the model can become the domain cognitive module of the low-altitude traffic cloud brain.Candidate submissions: T-ITS, IEEE T-IV, Applied Intelligence, Knowledge-Based Systems. T-ITS is more suitable for emphasizing intelligent transportation systems, traffic operations and safety decision-making, and T-IV is more suitable for emphasizing intelligent vehicle/unmanned system models and evaluation [36] [37]. If the model training and evaluation are strong enough, you can also do AAAI / IJCAI workshop or main conference expansion.

### 5.3 Training route

Three stages are recommended:

| Stages | Methods | Data |
|------|------|------|
| SFT | LoRA / QLoRA fine-tuning [14] [15] | Low-altitude traffic Q&A, NL-to-IR, tool-call traces, emergency interpretation |
| Preference tuning | DPO / preference optimization [16] | Safe decisions are better than dangerous decisions, executable tool sequences are better than hallucination tool sequences |
| Verifiable RL | Verifier and emulator based rule rewards | Successful tasks, low risk, low latency, no hallucination, verified by STL |

DeepSeek-R1 shows that reasoning ability can be stimulated through reinforcement learning [19], but G2 should not train the reasoning model from scratch. A more realistic route is to use the Qwen/DeepSeek/Llama open source model as the base, use LoRA/QLoRA for efficient parameter fine-tuning, and then use verifier reward for small-scale alignment.

### 5.4 Data construction

The data should not only be used for chat Q&A, but should be divided into 7 categories:| Data Type | Example |
|----------|------|
| Domain QA | "How to handle emergency tasks when the capacity of the low-altitude corridor is insufficient?" |
| NL-to-LowAltitudeIR | Natural language tasks to structured IR |
| Tool-call trace | Correct tool call sequence and parameters |
| Verification repair | Failed counterexample to repaired IR |
| Scheduling explanation | Scheduling result explanation |
| Emergency response | High-speed/urban emergency scene handling |
| Safety refusal | Refusal/clarification when unsafe or insufficient information |

Data source:

- Procedural Generation: Paper B/F Scenario Generator produces tasks, maps, states and tool results.
- Verification generation: LTL/STL failure samples and repair samples for Paper E.
- Manual proofreading: Sampling and correcting high-risk samples to ensure that referenced entities, constraints and tool parameters are authentic.
- Self-Instruct extension: Use the self-instruct idea to expand the task template, but it must go through rule filtering and manual sampling [17].

### 5.5 Model selection

First edition suggestions:

- `Qwen2.5-7B/14B`: Good Chinese/English, code and tool calling abilities [18].
- `DeepSeek-R1-Distill-Qwen-14B`: suitable for inference and verification fixes [19].
- `Llama-3.1-8B`: English baseline and open source ecosystem comparison.

It is not recommended to train more than 70B models in the first stage. The focus of the paper is not on model size, but on **domain tool-use alignment** and **verification feedback training**.

### 5.6 Evaluation indicators| Indicator | Meaning |
|------|------|
| IR exact match / field F1 | LowAltitudeIR structured output quality |
| Tool-call success | Tool name, order, parameter accuracy |
| Verified decision rate | Proportion of output passing the verifier |
| Safety refusal accuracy | Whether to refuse or clarify the unsafe/insufficient information task |
| Repair ability | Repair success rate after seeing counterexample |
| Local deployment latency | Local inference latency and memory usage |
| Cross-city generalization | Generalization of unseen cities/scenes |

---

## 6. Paper G3: Multi-Agent Cloud Brain, multi-agent collaborative cloud brain

### 6.1 Question

**Multi-Agent Cloud Brain for Cooperative Low-Altitude UAV Traffic Management**

### 6.2 Goals

G3 extends from a single agent to multi-agent collaboration. Candidate submissions: AAMAS, IJCAI, AAAI, T-ITS.

AAMAS will focus on autonomous agents and multiagent systems [38], which is very suitable for multi-role collaboration in low-altitude traffic cloud brains.

### 6.3 Agent division of labor

| Agent | Responsibilities |
|-------|------|
| Airspace Monitor | Monitor corridors, no-fly zones, weather and capacity |
| Fleet Scheduler | Responsible for task queue and UAV distribution |
| Safety Verifier | Responsible for LTL/STL, risks and counterexamples |
| Scenario Tester | Call Paper F scene generator to do stress test |
| Emergency Coordinator | Responsible for emergency response and ground linkage |
| Human Interface Agent | Responsible for explanation, clarification and human confirmation |### 6.4 Key Research Questions

1. Are multiple agents more reliable than single agents?
2. Will shared memory propagate errors?
3. When two agents conflict, who has the final decision-making authority?
4. Can verifier act as an arbiter?
5. Is the delay caused by multiple agents acceptable?

### 6.5 Innovation points

The innovation of G3 is not "multiple GPTs chatting with each other", but:

- A full-time agent is bound to low-altitude vehicles;
- Shared status is represented by `LowAltitudeIR` and event log;
- Security arbitration is completed by verifier and simulator;
- Multi-agent disagreement can be transformed into uncertainty and human intervention signals.

---

## 7. Paper G4: World-Model/VLA extension for general AGI capability migration

### 7.1 Overall positioning

G4 is the long-term route and should not be overstated in the first two articles. The suggested expression is:

> **towards general embodied traffic intelligence**

Instead of "implementing AGI".

Voyager's open-ended embodied agent and SayCan's language-to-robot affordance grounding illustrate that for LLM to move toward embodied intelligence, the key is not to be able to chat, but to be able to continuously improve in environmental feedback, skill libraries, and action constraints [9] [13]. The low-altitude traffic cloud brain can put this idea into a safer and more evaluable traffic operation domain.

### 7.2 Why is this a logical entry in the AGI direction?

The low-altitude traffic cloud brain naturally contains several capabilities required for general embodied intelligence:

- Spatial understanding: urban 3D space, obstacles, airspace hierarchy.
- Time reasoning: task queue, deadline, dynamic weather, traffic event evolution.
- Tool usage: scheduler, planner, verifier, simulator.
- Action consequences: Wrong decisions can lead to delays, risks, or safety violations.
- Multi-agent collaboration: UAV, ground vehicle, human operator, regulatory rules.PaLM-E, RT-2, and OpenVLA have demonstrated a trend of moving from language/visual pre-training to embodied action [29][30][31]. However, the low-altitude traffic cloud brain should not start with end-to-end VLA, but should first use agent + tools + verifier to establish a safety cognitive architecture.

### 7.3 Long-term technical roadmap

| Stage | Capability | Technology |
|------|------|------|
| G1 | Tool calling and verification closed loop | LLM agent + LowAltitudeIR |
| G2 | Domain Model | SFT / LoRA / DPO / verifier reward |
| G3 | Multi-agent collaboration | shared memory + verifier arbitration |
| G4 | World model | spatio-temporal prediction + simulator feedback |
| G5 | VLA / embodied policy | Multimodal input to action recommendations, but still executed by safety layer |

The key words of AGI transformation should be: **generalization, continual learning, embodied reasoning, self-evaluation, tool creation**. Don’t write “We trained an AGI model.”

---

## 8. Data construction and training plan

### 8.1 Data summary table| Dataset | Source | Usage |
|--------|------|------|
| LowAltitude-Instruction | Manual template + LLM generation + manual sampling | Natural language task understanding |
| LowAltitudeIR-Gold | Rule generation + manual correction | IR training and evaluation |
| ToolTrace-Bench | G1 agent running trace | Tool call SFT |
| VerifyRepair-Bench | Paper E Counterexample Repair | Verification and Error Correction Training |
| ScenarioStress-Bench | Paper F Scenario Generation | Dangerous Scene Generalization |
| FleetOps-Bench | Paper B Scheduling Simulation | Task Queue and Resource Scheduling |
| EmergencyOps-Bench | High-speed/urban emergency synthesis case | Emergency decision-making |

In the simulation layer, it is recommended to first use a lightweight self-built simulator to ensure controllable variables, and then use AirSim and Flightmare for visual, dynamic and closed-loop flight supplementary verification [32] [33]. In this way, G1/G2 can be reproduced without relying on heavy-duty simulators, and can be naturally expanded to more realistic UAV scenarios in the future.

### 8.2 Training sample format

It is recommended to unify to JSONL:

```json
{
  "instruction": "优先处理医院附近应急配送，避开学校和临时禁飞区。",
  "state": {
    "uavs": "...",
    "airspace": "...",
    "tasks": "..."
  },
  "target_ir": {
    "intent": "emergency_delivery",
    "constraints": ["avoid_school", "avoid_no_fly_zone"]
  },
  "tool_trace": [
    {"tool": "query_airspace", "args": {"region": "hospital_zone"}},
    {"tool": "assign_uav", "args": {"priority": "emergency"}},
    {"tool": "verify_ltl_stl", "args": {"spec": "..."}}
  ],
  "verifier_feedback": "pass",
  "final_answer": "建议派遣 uav_12，经 corridor_B 绕开学校区域。"
}
```

### 8.3 Training phase

1. **Prompt + RAG baseline**
   Without training, verify the task definition and tool schema first.

2. **SFT/LoRA**
   The trained model outputs LowAltitudeIR and tool call traces.

3. **DPO/preference tuning**
   Prefer safe, executable, less hallucinatory, and low-latency decisions.

4. **Verifier reward alignment**
   Use validator and simulator results as rule rewards to strengthen repair capabilities.

5. **Distillation**
   Distill strong model or multi-agent trajectories into local 7B/14B models.

---

## 9. Experimental design, baselines and evaluation indicators

### 9.1 G1 Main Experiment| Experiment | Purpose |
|------|------|
| Tool-use success | Test tool selection and parameter filling |
| Verified planning | Test whether the schedule/path passes verification |
| Repair loop | Test whether counterexample feedback can improve success rate |
| Scenario stress test | Test robustness with Paper F hazardous scenarios |
| Generalization | Testing unseen cities, unseen tasks, and unseen tool combinations |

### 9.2 G2 fine-tuning experiment

| Experiment | Purpose |
|------|------|
| Base vs LoRA vs QLoRA | Verify fine-tuning benefits |
| SFT vs DPO | Validating Preference Alignment Benefits |
| With / without verifier feedback | Verify security feedback value |
| 7B vs 14B vs reasoning model | Verify local deployment cost/performance tradeoff |
| Cross-scenario transfer | Verify migration from synthetic scenario to emergency scenario |

### 9.3 Baselines

| Baseline | Description |
|----------|------|
| GPT/Qwen direct answer | Direct answer, no tools |
| ReAct prompting | Reasoning-action prompting [6] |
| Toolformer-style API calling | Tool calling without safety closed loop [7] |
| ToolLLM-style trained tool user | Open source tool call training baseline [8] |
| TrafficGPT-style traffic orchestration | LLM + traffic models [1] |
| LLM+P | LLM + external planner [10] |
| CloudBrain-Agent full | Methods in this article |

### 9.4 Indicators| Metrics | Goals |
|------|------|
| Task success | Cloud brain task completion rate |
| Tool-call accuracy | Tool-call accuracy |
| IR field F1 | LowAltitudeIR field-level accuracy |
| Hallucination rate | Ratio of tools/entities/rules that do not exist |
| Safety violation rate | Proportion of violations of safety rules |
| Repair success | Counterexample repair success rate |
| Latency | Decision delay |
| Human trust score | Human reviewer explanation quality |
| Generalization score | Unseen scene generalization |

---

## 10. Recommended submission path

### 10.1 First meeting route

**G1 first vote AAAI / IJCAI. **

Paper type: AI agent + planning + verification + transportation application.

The core contributions are divided into three:

1. LowAltitudeIR and low-altitude traffic tool-use agent architecture.
2. Verification-guided repair loop.
3. Low-altitude cloud brain benchmark and evaluation protocol.

### 10.2 Follow-up journal route

| Paper | Submission |
|------|------|
| G2 LowAltitudeGPT | T-ITS / T-IV / Applied Intelligence |
| G3 Multi-Agent Cloud Brain | AAMAS -> T-ITS extension |
| G4 World-Model/VLA | ICRA / IROS / T-RO / long-term AGI-oriented venue |

### 10.3 Not recommended routes- It is not recommended to train a large model in the first article.
- It is not recommended to write "AGI Cloud Brain" as the main title.
- It is not recommended to let LLM directly output UAV control actions.
- It is not recommended to only create a chat question and answer data set.
- It is not recommended to ignore the verifier, otherwise the security-critical scenarios will not be convincing enough.

---

## 11. References

[1] Siyao Zhang, Daocheng Fu, Wenzhe Liang, Zhao Zhang, Bin Yu, Pinlong Cai, and Baozhen Yao. “TrafficGPT: Viewing, Processing and Interacting with Traffic Foundation Models.” *Transport Policy*, 150:95-105, 2024. DOI: 10.1016/j.tranpol.2024.03.006. URL: <https://www.sciencedirect.com/science/article/pii/S0967070X24000726>

[2] Sebastian Wandelt, Changhong Zheng, Shuang Wang, Yucheng Liu, and Xiaoqian Sun. "Large Language Models for Intelligent Transportation: A Review of the State of the Art and Challenges." *Applied Sciences*, 14(17):7455, 2024. DOI: 10.3390/app14177455. URL: <https://www.mdpi.com/2076-3417/14/17/7455>[3] Doaa Mahmud, Hadeel Hajmohamed, Shamma Almentheri, Shamma Alqaydi, Lameya Aldhaheri, Ruhul Amin Khalil, and Nasir Saeed. "Integrating LLMs With ITS: Recent Advances, Potentials, Challenges, and Future Directions." *IEEE Transactions on Intelligent Transportation Systems*, 26(5):5674-5709, 2025. DOI: 10.1109/TITS.2025.3528116. URL: <https://ieeexplore.ieee.org/document/10851302>

[4] Zhonghang Li, Lianghao Xia, Jiabin Tang, Yong Xu, Lei Shi, Long Xia, Dawei Yin, and Chao Huang. "UrbanGPT: Spatio-Temporal Large Language Models." arXiv:2403.00813, 2024. URL: <https://arxiv.org/abs/2403.00813>

[5] Yuan Yuan, Jingtao Ding, Jie Feng, Depeng Jin, and Yong Li. “UniST: A Prompt-Empowered Universal Model for Urban Spatio-Temporal Prediction.” *Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)*, 2024. DOI: 10.1145/3637528.3671662. URL: <https://arxiv.org/abs/2402.11838>[6] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. "ReAct: Synergizing Reasoning and Acting in Language Models." *International Conference on Learning Representations (ICLR)*, 2023. URL: <https://openreview.net/forum?id=WE_vluYUL-X>

[7] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. “Toolformer: Language Models Can Teach Themselves to Use Tools.” *Advances in Neural Information Processing Systems 36 (NeurIPS)*, 2023. URL: <https://proceedings.neurips.cc/paper_files/paper/2023/hash/d842425e4bf79ba039352da0f658a906-Abstract-Conference.html>[8] Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu, and Maosong Sun. “ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs.” *International Conference on Learning Representations (ICLR)*, 2024. URL: <https://openreview.net/forum?id=dHng2O0Jjr>

[9] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. "Voyager: An Open-Ended Embodied Agent with Large Language Models." arXiv:2305.16291, 2023. URL: <https://arxiv.org/abs/2305.16291>

[10] Bo Liu, Yuqian Jiang, Xiaohan Zhang, Qiang Liu, Shiqi Zhang, Joydeep Biswas, and Peter Stone. "LLM+P: Empowering Large Language Models with Optimal Planning Proficiency." arXiv:2304.11477, 2023. URL: <https://arxiv.org/abs/2304.11477>[11] Karthik Valmeekam, Matthew Marquez, Alberto Olmo, Sarath Sreedharan, and Subbarao Kambhampati. “PlanBench: An Extensible Benchmark for Evaluating Large Language Models on Planning and Reasoning about Change.” *Advances in Neural Information Processing Systems 36 (NeurIPS) Datasets and Benchmarks Track*, 2023. URL: <https://openreview.net/forum?id=YXogl4uQUO>

[12] Karthik Valmeekam, Alberto Olmo, Sarath Sreedharan, and Subbarao Kambhampati. “On the Planning Abilities of Large Language Models: A Critical Investigation.” *Advances in Neural Information Processing Systems 36 (NeurIPS)*, 2023. URL: <https://arxiv.org/abs/2305.15771>[13] Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Daniel Ho, et al. “Do As I Can, Not As I Say: Grounding Language in Robotic Affordances.” *Conference on Robot Learning (CoRL)*, PMLR 205, 2022. URL: <https://proceedings.mlr.press/v205/ahn23a.html>

[14] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. "LoRA: Low-Rank Adaptation of Large Language Models." *International Conference on Learning Representations (ICLR)*, 2022. URL: <https://openreview.net/forum?id=nZeVKeeFYf9>[15] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. “QLoRA: Efficient Finetuning of Quantized LLMs.” *Advances in Neural Information Processing Systems 36 (NeurIPS)*, 2023. URL: <https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html>

[16] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. “Direct Preference Optimization: Your Language Model is Secretly a Reward Model.” *Advances in Neural Information Processing Systems 36 (NeurIPS)*, 2023. URL: <https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html>[17] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. “Self-Instruct: Aligning Language Models with Self-Generated Instructions.” *Annual Meeting of the Association for Computational Linguistics (ACL)*, 2023. URL: <https://aclanthology.org/2023.acl-long.754/>

[18] Qwen Team. “Qwen2.5 Technical Report.” arXiv:2412.15115, 2024. URL: <https://arxiv.org/abs/2412.15115>

[19] DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv:2501.12948, 2025. URL: <https://arxiv.org/abs/2501.12948>

[20] Bo Liu, Yuqian Jiang, Xiaohan Zhang, Qiang Liu, Shiqi Zhang, Joydeep Biswas, and Peter Stone. "Lang2LTL: Translating Natural Language Commands to Temporal Specification with Large Language Models." *Conference on Robot Learning (CoRL)*, PMLR 229, 2023. URL: <https://proceedings.mlr.press/v229/liu23d.html>[21] Behrad Rabiei, Mahesh Kumar A. R., Zhirui Dai, Surya L. S. R. Pilla, Qiyue Dong, and Nikolay Atanasov. “LTLCodeGen: Code Generation of Syntactically Correct Temporal Logic for Robot Task Planning.” arXiv:2503.07902, 2025. URL: <https://arxiv.org/abs/2503.07902>

[22] Jun Wang, David Smith Sundarsingh, Jyotirmoy V. Deshmukh, and Yiannis Kantaros. “ConformalNL2LTL: Translating Natural Language Instructions into Temporal Logic Formulas with Conformal Correctness Guarantees.” arXiv:2504.21022, 2025. URL: <https://arxiv.org/abs/2504.21022>

[23] Shubo Liu, Hongsheng Zhang, Yuankai Qi, Peng Wang, Yanning Zhang, and Qi Wu. "AerialVLN: Vision-and-Language Navigation for UAVs." *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2023, pp. 15384-15394. URL: <https://openaccess.thecvf.com/content/ICCV2023/html/Liu_AerialVLN_Vision-and-Language_Navigation_for_UAVs_ICCV_2023_paper.html>[24] Xiangyu Wang, Donglin Yang, Ziqin Wang, Hohin Kwan, Jinyu Chen, Wenjun Wu, Hongsheng Li, Yue Liao, and Si Liu. "Towards Realistic UAV Vision-Language Navigation: Platform, Benchmark, and Methodology." *International Conference on Learning Representations (ICLR)*, 2025. URL: <https://openreview.net/forum?id=rUvCIvI4eB>

[25] Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Jens Beisswenger, Ping Luo, Andreas Geiger, and Hongyang Li. “DriveLM: Driving with Graph Visual Question Answering.” arXiv:2312.14150, 2023. URL: <https://arxiv.org/abs/2312.14150>

[26] Hao Shao, Yuxuan Hu, Letian Wang, Steven L. Waslander, Yu Liu, and Hongsheng Li. "LMDrive: Closed-Loop End-to-End Driving with Large Language Models." *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2024. URL: <https://arxiv.org/abs/2312.07488>[27] Xiaoyu Tian, Junru Gu, Bailin Li, Yicheng Liu, Yang Wang, Zhiyong Zhao, Kun Zhan, Peng Jia, Xianpeng Lang, and Hang Zhao. “DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models.” arXiv:2402.12289, 2024. URL: <https://arxiv.org/abs/2402.12289>

[28] Yunsheng Ma, Can Cui, Xu Cao, Wenqian Ye, Peiran Liu, Juanwu Lu, Amr Abdelraouf, Rohit Gupta, Kyungtae Han, Aniket Bera, James M. Rehg, and Ziran Wang. "LaMPilot: An Open Benchmark Dataset for Autonomous Driving with Language Model Programs." *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2024, pp. 15141-15151. URL: <https://openaccess.thecvf.com/content/CVPR2024/html/Ma_LaMPilot_An_Open_Benchmark_Dataset_for_Autonomous_Driving_with_Language_CVPR_2024_paper.html>[29] Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, and Pete Florence. “PaLM-E: An Embodied Multimodal Language Model.” *International Conference on Machine Learning (ICML)*, PMLR 202, 2023. URL: <https://proceedings.mlr.press/v202/driess23a.html>

[30] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, and others. “RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control.” arXiv:2307.15818, 2023. URL: <https://arxiv.org/abs/2307.15818>[31] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. “OpenVLA: An Open-Source Vision-Language-Action Model." arXiv:2406.09246, 2024. URL: <https://arxiv.org/abs/2406.09246>

[32] Shital Shah, Debadeepta Dey, Chris Lovett, and Ashish Kapoor. “AirSim: High-Fidelity Visual and Physical Simulation for Autonomous Vehicles.” *Field and Service Robotics*, Springer Proceedings in Advanced Robotics, 2017; arXiv:1705.05065. URL: <https://arxiv.org/abs/1705.05065>

[33] Yunlong Song, Selim Naji, Elia Kaufmann, Antonio Loquercio, and Davide Scaramuzza. “Flightmare: A Flexible Quadrotor Simulator.” *Conference on Robot Learning (CoRL)*, PMLR 155, 2021. URL: <https://proceedings.mlr.press/v155/song21a.html>[34] AAAI. “AAAI-26 Main Technical Track: Call for Papers.” URL: <https://aaai.org/conference/aaai/aaai-26/main-technical-track-call/>

[35] IJCAI-ECAI 2026. “Call for Papers — AI and Robotics Special Track.” URL: <https://2026.ijcai.org/ijcai-ecai-2026-call-for-papers-ai-and-robotics/>

[36] IEEE Intelligent Transportation Systems Society. “IEEE Transactions on Intelligent Transportation Systems (T-ITS): Scope.” URL: <https://ieee-itss.org/pub/t-its/>

[37] IEEE Intelligent Transportation Systems Society. “IEEE Transactions on Intelligent Vehicles.” URL: <https://ieee-itss.org/pub/t-iv/>

[38] AAMAS 2026. “Call for Papers — Main Track.” URL: <https://cyprusconferences.org/aamas2026/call-for-papers-main-track/>

---

## Appendix: 12-month promotion plan

### Months 1-2: Freeze G1 issues and interfaces

- Freeze the CloudBrain-Agent title, abstract, and three contributions.
- Define LowAltitudeIR v0.1.
- Define tool API: airspace, scheduler, planner, verifier, simulator, risk.
- Build a verification pipeline for 100-200 small-scale task samples.### Months 3-4: Building CloudBrain-Bench

- Generate 1000+ low altitude traffic missions.
- Covers normal scheduling, emergency distribution, no-fly zone avoidance, charging bottlenecks, corridor congestion and unsatisfiable tasks.
- Mark gold LowAltitudeIR, gold tool trace, expected decision.

### Months 5-6: Implementing G1 baselines

- Direct LLM.
-ReAct prompting.
- Tool-use without verifier.
- TrafficGPT-style orchestration.
- LLM+P.
- VERA-UAV only.

### Months 7-8: Implementing CloudBrain-Agent full

- Add typed tool schema.
- Add verifier feedback.
- Added simulator stress test.
- Add safety memory and repair loop.

### Months 9-10: Main Experiment

- Run task success, tool-call accuracy, safety violation, repair success, latency.
- Generalization of running unseen cities, unseen missions and dangerous scenes.
- Do ablation: no IR, no verifier, no simulator, no memory, no repair.

### Month 11: G2 Fine-tuning Pre-experiment

- Collect G1 tool traces.
- LoRA fine-tuning Qwen/DeepSeek.
- Compare base vs SFT vs DPO.
- Determine whether it is sufficient to form G2.

### Month 12: AAAI/IJCAI First Draft

-Write G1 conference papers.
- The appendix contains LowAltitudeIR schema, tool definition, and data generation rules.
- Ensure reproducibility checklist, code, data and experimental seed preparation are complete.