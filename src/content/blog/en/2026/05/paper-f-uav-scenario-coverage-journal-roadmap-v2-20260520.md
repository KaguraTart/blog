---
title: "Paper F Journal Planning v2: Journal Priority Route for UAV Safety-Critical Scenario Engineering"
description: "Without considering the structure of the doctoral thesis, the journal's priority output route for Paper F will be re-planned, focusing on UAV safety critical scenario coverage, accelerated testing, risk assurance and high-speed emergency applications."
pubDate: 2026-05-20
updatedDate: 2026-05-23
tags: ["Paper F", "Journal planning", "UAV", "scene generation", "scene coverage", "Safety-Critical", "accelerated testing", "risk guarantee", "TR-C", "T-ITS"]
category: Tech
sourceHash: "a0e4aeb6f8bbbaba5c9d96869c63d0b8f954d370"
---

# Paper F Journal Planning v2: Journal Priority Route for UAV Safety-Critical Scenario Engineering

> The current focus returns to **journal paper output**, which is not organized by doctoral theses.  
> Conclusion: Paper F should not be broken into many thin papers, but should be made into a complete, solid, and reproducible **T-ITS main journal**, and then differentiated into TR-C application papers and risk assurance method papers based on experimental assets.

---

## 1. Core judgment

The general direction of Paper F still holds: **UAV safety-critical scenario engineering**. But the logic of a journal article and a doctoral thesis chapter is different. Journal reviewers will not pay for a complete route, they are more concerned about:

- Is the question specific enough?
- Whether the method has clear technical increments;
- Whether the experiment is solid enough;
- Whether baselines are strong;
- Whether the conclusion supports an independent journal story;
- Whether it matches the scope of the target journal.

Therefore, the current "Planning 4-5" should be changed to:

> **First merge the F1 scene coverage benchmark and F2 dangerous scene accelerated generation into a main T-ITS; later, TR-C emergency application, T-RO/T-ASE risk assurance, and TR-C/T-ITS urban ODD scene generation will be differentiated from the same platform. **

### 1.1 2026-05-22 Writing calibration: F series needs to separate the "test method paper" and "traffic system paper"

Paper F is easy to write loosely because it also has scene generation, coverage, dangerous scene accelerated testing, urban ODD, and Shandong highway emergency applications. The latest calibration is:- **F-J1 should not be written as TR-C. ** Its primary issue is safety-critical scenario testing: how to systematically discover dangerous but effective scenarios of UAV obstacle avoidance / low-altitude navigation. It is more natural to invest in T-ITS as the main focus is on safety testing, simulation evaluation and scenario coverage in intelligent transportation systems.
- **F-J2 is the TR-C application paper. ** It must be written as high-speed traffic emergency operation issues: accident detection, drone reconnaissance, ground resource allocation, response time, coverage, information delay and traffic recovery.
- **F-J4 Only throw TR-C if city-level ODD can be fed back to traffic planning/operational control. ** If you just convert OSM into a local obstacle combination, it will be more like a simulation tool or benchmark, but not TR-C enough.

Therefore, the "story" of the F series can be divided into two types:

| Thesis | System Story | What must be supported |
|------|----------|----------------|
| F-J1 | Low-altitude ITS safety testing lacks coverage and dangerous scenario generation standards | coverage metric, invalid scenario rate, failure discovery, planner cross-test, multi-seed statistics |
| F-J2 | High-speed emergency response requires UAV-ground collaboration to shorten golden response time | Real high-speed topology/accident proxy, resource allocation model, response time, coverage, accessibility under congestion, sensitivity analysis |
| F-J3 | How to convert scene coverage into risk assurance evidence | coverage-to-risk bound, confidence interval, rare-event estimation, reliability index |
| F-J4 | How the overall city ODD determines local low-altitude test scenarios | OSM/POI/building/road/airspace mapping, local risk fidelity, cross-city generalization |

The traffic journal version must avoid just saying "we generated more dangerous scenarios". A stronger conclusion would be:- Which urban structures or highway sections are more likely to induce UAV failure?
- Which combinations of obstacles are most dangerous for different planners?
- Does increased coverage actually reduce undetected risk, rather than just increasing sample size?
- In emergency applications, can drone reconnaissance reduce dispatch losses caused by incomplete information?
- When are weather, communications or landing points restricted, when will the system require ground resources to back it up?

The reason is very straightforward: benchmark alone is easy to be considered as too many engineering platforms, and accelerated testing alone will be questioned whether the test scenario space is clearly defined. After the two are merged, the paper is upgraded from "I generate dangerous scenarios" to:

> **I defined the UAV safety-critical scene space, which can measure coverage, discover coverage holes, and use coverage-guided methods to more efficiently generate realistic, dangerous, and feasible test scenarios. **

This is more like a journal article.

### 1.2 2026-05-23 Compilation: The F series currently only advances two main lines

Currently, Paper F is not expanded according to the doctoral thesis catalogue, but is first condensed into two main lines based on journal output. The F-J3 and F-J4 are retained but do not take away the experimental resources of the F-J1.| Thesis | Lead Investor | Current Role | Recent Strategies |
|------|------|----------|----------|
| F-J1 | T-ITS | coverage-guided accelerated testing | Main push; must use 76 million exploration logs, coverage metric, strong baseline and cross-planner evaluation |
| F-J2 | TR-C | Shandong high-speed emergency rescue resource allocation | F-J1 platform will be started after stabilization; the focus is on real high-speed topology, accident proxy, response time and resource bottlenecks |
| F-J3 | T-RO / T-ASE / T-ITS | coverage-to-risk assurance | Suspended; wait until F-J1 forms failure distribution and coverage statistics before proving the risk boundary |
| F-J4 | TR-C / T-ITS | city-level ODD to local UAV scenario | Suspended; wait until OSM/POI/building/airspace pipeline is stable enough |

The recommended outline for the first version of F-J1 is fixed at:1. **Scenario space**: Define UAV local test cell, obstacle grammar, dynamic factors, mission objectives and invalid scenario determination.
2. **Coverage metric**: Separate statistics on geometric coverage, semantic coverage, dynamics coverage, risk coverage and failure-mode coverage.
3. **Accelerated generation**: Use coverage holes and failure likelihood to guide sampling and filter out unrealistic or unexecutable scenarios.
4. **Benchmark protocol**: Unify map seed, planner set, controller parameters, random seeds, failure thresholds and statistical tests.
5. **Main experiments**: Compare random, grid/LHS, BO, CMA-ES, RL adversarial, Scenic-style constrained generation and this method.
6. **Failure analysis**: Explain which combinations of obstacles, speed/height conditions, occlusions and dynamic obstacles are most likely to trigger failure.

The judgment after this sorting is: F-J1 first pursues "a security testing journal paper that can be submitted to T-ITS", and should not commit to urban planning, risk theory and Shandong high-speed applications at the same time. F-J2 can switch the story to the traffic emergency operation closed loop required by TR-C after the F-J1's scenario library and risk indicators mature.

---

## 2. Journal priority paper portfolio

It is recommended to temporarily plan **3 main journals + 1 reserve journal** instead of promoting 5 articles at the same time.| Number | Paper positioning | Suggested topic | Lead investor | Priority |
|------|----------|----------|------|--------|
| F-J1 | Workhorse method + benchmark | Coverage-Guided Accelerated Testing for Safety-Critical UAV Navigation Scenarios | T-ITS | Highest |
| F-J2 | Traffic Emergency Application | Scenario-Aware UAV-Ground Resource Allocation for Highway Emergency Response | TR-C | High |
| F-J3 | Coverage-to-Risk Assurance for UAV Safety-Critical Scenario Testing | T-RO / T-ASE / T-ITS | Medium to High |
| F-J4 | Urban scene generation | City2Local-UAV: Hierarchical Scenario Generation from Urban ODDs to Local Obstacle Compositions | TR-C / T-ITS | Medium |

**Execution order recommendation: F-J1 -> F-J2 -> F-J3 -> F-J4. **

F-J1 is the platform and algorithm base. F-J2 is closest to transportation journal application value. F-J3 Robotics or automation journals used to update methods/theories. F-J4 can only be done after the OSM/city data pipeline has matured, otherwise it will easily become a "map conversion tool".

---

## 3. Literature pattern and gaps

### 3.1 Autonomous driving scenario engineering has matured, but UAV migration is insufficientThere is already a complete scene engineering chain in the field of autonomous driving. ISO 34502 provides a scenario-based safety evaluation framework for automated driving systems [1], and ASAM OpenSCENARIO and OpenODD provide executable scenario and ODD description standards [2] [3]. Shuo Feng's accelerated testing and testing scenario library generation further illustrates that safety-critical testing cannot rely on natural random samples, but must use a data-driven approach to improve the sampling efficiency of critical events [4] [5] [6].

In recent years, top conferences have also continued to promote this direction: Scenic uses probabilistic programming language to express scene distribution and constraints [7]; SafeBench has made a safety evaluation benchmark [8]; ScenarioNet extracts large-scale traffic scenarios from real driving data [9]; AdvSim, KING, ChatScene, and FREA generate safety-critical scenarios from the perspectives of sensor perturbation, gradient optimization, LLM knowledge generation, and feasible adversariality respectively [10] [11] [12] [13].

However, most of these works are oriented towards ground autonomous driving, and UAV scenarios are significantly different:

- UAV is a three-dimensional movement, and the scene dimensions include altitude, track inclination, wind field, power, field of view occlusion and flight dynamics;
- UAV dangerous events include collision with buildings, collision with lines, crossing no-fly zones, conflicts in low-altitude corridors, failure to take off and land, and accidental entry into emergency sites;
- UAV safety testing rarely matures ODD taxonomy;
- UAV benchmark mostly focuses on simulation and control tasks, and rarely answers "what risks does the scenario cover?"

### 3.2 UAV simulation has a foundation, but coverage-oriented safety testing is still emptyAirSim and Flightmare are important foundations for UAV simulation [14][15]. AvoidBench has proposed a high-fidelity benchmark for vision-based multi-rotor obstacle avoidance [16]. OmniDrones and Aerial Gym illustrate that GPU parallel UAV simulation and large-scale reinforcement learning training are maturing [17] [18]. FADS proves that temporal logic safety specification can enter the drone safety pipeline [19].

These works provide the tool foundation for Paper F, but they have not yet addressed the most critical gaps in journal papers:

> **How to define UAV safety-critical scenario space, how to measure coverage, how to efficiently generate long-tail scenarios that are both dangerous and feasible, and how to convert test coverage into interpretable risk assessments. **

This is the opportunity for the F-J1/F-J3.

### 3.3 The difference between TR-C/T-ITS determines how to cut the paper

The intellectual core of TR-C is on the transportation side, emphasizing the impact of emerging technologies on transportation system planning, design, operation, control and logistics [20]. T-ITS explicitly covers information technology applications in sensing, communications, controls, planning, design, implementation, AI, and transportation systems [21].

therefore:- **F-J1 Vote for T-ITS**: Because it is ITS safety evaluation / scenario generation / UAV navigation testing.
- **F-J2 voted for TR-C**: Because it is high-speed traffic emergency system operation and resource allocation.
- **F-J3 is eligible for T-RO/T-ASE/T-ITS**: Depending on the focus of theory and experiment, T-RO/T-ASE can be selected for robot safety testing, and T-ITS can be selected for transportation systems.
- **F-J4 votes for TR-C/T-ITS**: If the emphasis is on urban low-altitude traffic ODD and the impact of traffic systems, vote for TR-C; if the emphasis is on scene interfaces and simulation evaluation, vote for T-ITS.

### 3.4 What other journal directions can I write about?

After continuing to dig deeper, you can prepare 4 more "candidate forks", but it is not recommended to start writing at the same time now. They are more suitable as a natural spillover as the F-J1’s experimental platform matures.| Forks | Writable topics | Core selling points | Candidate journals | Current recommendations |
|------|----------|----------|----------|----------|
| F-J5 | Scenario-Based Safety Case for Low-Altitude UAV Operations | Organize coverage, criticality, and failure evidence into safety cases | Reliability Engineering & System Safety / IEEE Transactions on Reliability / Safety Science | Wait for F-J1 to have results before writing |
| F-J6 | Cross-Simulator Transfer of UAV Safety-Critical Scenarios | Study scenario transfer from lightweight simulation to AirSim / Flightmare / AvoidBench | Robotics and Autonomous Systems / Journal of Field Robotics / T-RO | Realistic or high-fidelity verification required |
| F-J7 | Knowledge-Guided UAV Scenario Generation | Use LLM/VLM/Knowledge Graph to generate semantic hazard scenes | T-ITS / T-IV / IEEE Open Journal of ITS | Can be linked with Paper E, but don’t overwhelm it |
| F-J8 | Multi-UAV Corridor Stress Testing | Specially generated low-altitude corridor conflict, congestion, take-off and landing bottleneck scenarios | T-ITS / TR-C / T-IV | Can be linked with Paper B |Among them, the **F-J5 is the most worth keeping**. If the follow-up F-J1 only stops at "finding more failures", the value of the journal will still be experimental; but if the scenario coverage can be converted into reliability/safety assurance evidence, it can be submitted to safety and reliability journals such as Reliability Engineering & System Safety or IEEE Transactions on Reliability [28] [29]. Safety Science can also be used as an alternative, but it is more focused on safety management, human factors, organization and accident prevention. If the paper is still purely algorithmic, it is not recommended for first submission [30].

F-J6 is suitable for writing when there are real drones or high-fidelity simulation results. Journal of Field Robotics and Robotics and Autonomous Systems both value the autonomy, reliability, and experimental depth of robotic systems in real or high-fidelity environments [31] [32]. If you only have lightweight simulation, don't submit to this type of journal yet.

F-J7 is not recommended as the main line now because it will overlap with the LLM/LTL direction of Paper E. It is more suitable for subsequent extensions as "knowledge-guided scenario generation": LLM is responsible for proposing semantic hazard scenarios, and Cov-ATUAV is responsible for validating, filtering and quantifying coverage.

F-J8 is the stress-test version of Paper B. It no longer optimizes the scheduling of hundreds of UAVs, but generates test scenarios that best expose corridor congestion, vertiport bottleneck, charging bottleneck, and conflict-resolution failure. T-ITS or TR-C can be voted in this direction, but it must be cut off from Paper B's scheduling contribution.

### 3.5 Candidate Journal Map| Journal | The most suitable Paper F cut | Why it’s appropriate | Risks |
|------|------------------------|------------|------|
| IEEE T-ITS | F-J1 / F-J4 / F-J8 | scope covers sensing, control, AI, planning and transportation systems in ITS [21] | UAV needs to be written as low-altitude ITS, not an ordinary robot |
| IEEE T-IV | F-J1 / F-J7 / F-J8 | Intelligent vehicle and automated mobility context can receive safety testing and scenario generation [26] | Ground vehicles have strong color, UAV needs to explain vehicle/traffic relevance |
| TR-C | F-J2 / F-J4 / F-J8 | Emphasize the impact of emerging technologies on transportation operations, control, and logistics [20] | Not suitable for pure algorithm benchmark |
| TR-E | F-J2 | Suitable for logistics, distribution, supply chain and emergency resource transportation deployment [33] | If the UAV has too many technical details, it will deviate from logistics |
| T-ASE | F-J3 / F-J5 | automation systems, testing, evaluation, and reliability framing are more suitable [27] | The method needs to have generalization value for automation systems |
| T-RO | F-J3 / F-J6 | Robot safety testing, planning, and real system verification can be submitted [34] | Synthetic benchmark alone is not enough |
| IEEE Transactions on Reliability | F-J5 | Suitable for reliability modeling, risk quantification, assurance [28] | Serious statistical guarantee is required, not just experimental tables |
| Reliability Engineering & System Safety | F-J5 |Suitable for safety-critical systems, risk assessment and reliability engineering [29] | Need to convert from algorithm performance improvement to safety evidence |
| Safety Science | F-J2 / F-J5 | Suitable for emergency safety, accident prevention, safety management [30] | Pure UAV algorithm is not suitable |
| Robotics and Autonomous Systems / JFR | F-J6 | Suitable for autonomous robotic systems and field/high-fidelity validation [31] [32] | System experiments are required to be stronger than thesis narrative |
| IEEE Open Journal of ITS | F-J1 / F-J2 | Can be used as a fast open access alternative [35] | Generally lower impact and positioning than T-ITS |**The current first selection order remains unchanged: F-J1 is the first selection for T-ITS, F-J2 is the first selection for TR-C, F-J3 selects T-ASE/T-RO/T-ITS depending on the theoretical strength, and F-J5 is reserved for reliable journals. **

---

## 4. The first major journal: F-J1

### 4.1 Suggested topics

**Coverage-Guided Accelerated Testing for Safety-Critical UAV Navigation Scenarios**

### 4.2 Submission goals

Main contributor: **IEEE Transactions on Intelligent Transportation Systems**.  
Alternatives: IEEE Transactions on Automation Science and Engineering, IEEE Transactions on Robotics, Robotics and Autonomous Systems.

T-ITS is most suitable because the paper can be written as intelligent transportation safety testing: UAVs are low-altitude transportation participants, and scenario generation serves the safety verification of low-altitude ITS.

### 4.3 Core Issues

Existing UAV obstacle avoidance/navigation papers typically report success rate, collision rate, or trajectory length, but rarely indicate whether the test scenario covers safety-critical ODDs. Randomly generated scenes have two problems: a large number of samples are not dangerous, and many dangerous samples are physically infeasible or cannot be avoided by any algorithm.

F-J1 wants to answer:

> How to cover UAV low-altitude operation ODD under a limited simulation budget, and prioritize the generation of safety-critical scenarios that are real, dangerous, feasible, and can distinguish algorithm capabilities?

### 4.4 Method design

Method name suggestion: **Cov-ATUAV: Coverage-Guided Accelerated Testing for UAVs**.

Overall pipeline:

```text
UAV ODD taxonomy
  -> scenario parameterization
  -> coverage memory
  -> criticality and feasibility scoring
  -> adaptive scenario generation
  -> planner evaluation and coverage update
```

Core modules:| Module | Function |
|------|------|
| Scenario grammar | Define obstacles, spatial structures, dynamic bodies, wind fields, sensor noise, task types |
| Coverage memory | Record parameter bins, pairwise/t-wise coverage, failure modes |
| Criticality score | Comprehensive exposure, challenge, near-miss, constraint violation |
| Feasibility filter | Exclude inevitable collisions, unreasonable physics and meaningless mission scenarios |
| Adaptive generator | Generate new samples in coverage holes and high-criticality regions |
| Evaluation harness | Unified evaluation of multiple UAV planners and output ranking stability |

### 4.5 Data and Platform

- Main experiment: 50m x 50m x 50m local UAV test cell.
- Existing assets: 76 million exploration logs, used to count coverage holes, failure modes and initial scene distribution.
- Lightweight simulation: self-built 3D grid / PyBullet / custom dynamics for large-scale search.
- High-fidelity validation: AirSim, Flightmare, AvoidBench or Aerial Gym for small-scale cross-simulator validation [14] [15] [16] [18].

76 million explorations cannot be written as the final result, but it can be written as:

> “We initialize and validate our scenario coverage analysis using a large-scale exploration log containing over 76 million simulated rollouts.”

### 4.6 Baselines| Baseline | Function |
|----------|------|
| Random scenario generation | Basic sampling efficiency |
| Grid sampling | Uniform discrete coverage |
| Latin hypercube sampling | Parameter space coverage |
| Scenic-style constrained sampling | Constrained scene generation [7] |
| SafeBench-style template sampling | template-style safety benchmark [8] |
| Bayesian optimization | black box failure search |
| CMA-ES / cross-entropy method | Continuous parameter hazard search |
| AdvSim/KING-style adversarial editing | Adversarial trajectory/obstacle perturbation [10] [11] |
| FREA-style feasible adversarial generation | Reasonable adversarial example [13] |

### 4.7 UAV planners

At least three types of planners must be tested, otherwise the journal will question overfitting of only one algorithm:

| Planner | Representative |
|---------|------|
| Classical | A* / RRT* / artificial potential field / 3DVFH* |
| Optimization | MPC / safe corridor / B-spline trajectory optimization |
| Learning-based | PPO / SAC / imitation learning / vision-based policy |

If computing power is limited, the first version’s guaranteed choices are: RRT*, MPC-lite, PPO policy, and a vision-based baseline.

### 4.8 Indicators| Indicator | Description |
|------|------|
| Coverage gain | New coverage every 1000 tests |
| Failure discovery rate | The ratio of collision / near-miss / timeout discovered per unit budget |
| Acceleration factor | The multiple reduction in the number of tests required to achieve the same failure discovery rate |
| Feasible criticality | Proportion of dangerous and avoidable scenes |
| Invalid scenario rate | Physically infeasible or meaningless sample ratio |
| Planner ranking stability | Planner ranking stability under different seeds / scenario subsets |
| Cross-simulator transfer | Can scenarios discovered in lightweight simulation be transferred to high-fidelity simulation |

### 4.9 Minimum results that a journal can publish

F-J1 requires at least proof of:

1. Compared with random/grid/LHS, Cov-ATUAV significantly improves coverage gain under the same budget.
2. Compared with BO/CMA-ES/adversarial baselines, Cov-ATUAV reduces invalid scenario rate.
3. Compared with pure failure search, the scenes generated by Cov-ATUAV can distinguish different UAV planners more stably.
4. Verify that at least some of the high-risk scenarios can be migrated in AirSim / Flightmare / AvoidBench.
5. Output scenario schema, seed, benchmark split and evaluation script to enhance the reproducibility of T-ITS.

---

## 5. The second journal: F-J2 high-speed emergency application

### 5.1 Suggested topics

**Scenario-Aware UAV-Ground Resource Allocation for Highway Emergency Response**

### 5.2 Submission goalsMain investor: **Transportation Research Part C: Emerging Technologies**.  
Alternative: IEEE Transactions on Intelligent Transportation Systems.

This article must put UAV into transportation operation, rather than writing it as a "UAV scheduling algorithm." The Shandong Expressway Comprehensive Inspection Flight Service System already uses unattended platforms and industrial drones in inspections, inspections, emergency response, and data analysis [22]. Research on high-speed emergency resource allocation also points out problems such as incomplete information in the early stage of an accident, time-varying traffic conditions, and insufficient linkage between facility location selection and resource allocation [23].

### 5.3 Core Issues

High-speed accident emergency response is not simply about “sending the nearest resources.” When an accident first occurs, the type of accident, congestion length, lane closures, hazardous materials, and secondary accident risks are all uncertain. The value of UAV is not just to fly fast, but to reduce information uncertainty in advance and reduce mis-delivery and delays.

F-J2 wants to answer:

> In high-speed accident emergency, how to use UAV reconnaissance to reduce information uncertainty and coordinate with ground clearance, rescue, firefighting, traffic police, and maintenance resources to reduce response time, clearance time and secondary risk?

### 5.4 Method design

Method name suggestion: **SAFER-UAV: Scenario-Aware Fast Emergency Response with UAVs**.

Core structure:

```text
Incident scenario generator
  -> UAV first-view dispatch
  -> incident state belief update
  -> ground resource rolling allocation
  -> congestion / clearance simulator
  -> emergency performance evaluation
```

The key is to convert the F-J1 scene library into emergency mission scenarios:

- Accident types: rear-end collision, rollover, hazardous chemicals, road-occupying construction, bad weather, congestion secondary accidents.
- Road section geometry: straight lines, curves, ramps, service areas, toll stations, bridges, tunnel entrances.
- Uncertain information: accident severity, passable lanes, casualties, resource requirements, congestion length.
- Resource types: UAV, tow truck, ambulance, firefighting, traffic police, maintenance vehicle, temporary control equipment.

### 5.5 Baselines| Baseline | Function |
|----------|------|
| Ground-only dispatch | No UAV situation |
| Nearest-resource dispatch | Nearest resources first |
| Fixed plan / rule-based dispatch | Current practice approximation |
| UAV-first then dispatch | A simple strategy to watch first and then dispatch |
| Two-stage stochastic programming | Stochastic optimization baseline |
| Rolling horizon optimization | Strong optimization baseline |
| SAFER-UAV full | Main method |

### 5.6 Indicators

| Indicator | Description |
|------|------|
| First-view time | The time when the UAV first acquired the accident footage |
| Response time | Arrival time of the first batch of resources |
| Clearance time | Clearance completion time |
| Wrong dispatch rate | The proportion of wrong dispatch, missed dispatch or insufficient resources |
| Secondary accident risk | Secondary accident risk proxy |
| Traffic delay | Total delays caused by accidents |
| Information value of UAV | Uncertainty reduction and scheduling benefits brought by UAV reconnaissance |
| Coverage of critical assets | UAV/ground resources coverage capabilities for service areas, bridges, tunnels, and accident-prone sections |
| Robustness to information delay | Performance degradation when image return, event confirmation, and communication delays increase |
| Equity across road segments | Difference in response time between remote road segments and core road segments |

The TR-C version also requires a **system implication table**: given the number of UAVs, the number of takeoff and landing points, ground resource configuration and accident intensity, report when the system changes from "UAV reconnaissance has obvious benefits" to "ground resources or road network congestion becomes the main bottleneck". This table is more like a transportation systems paper than just average response times.

### 5.7 Minimum results that a journal can publish

F-J2 requires at least:1. Explicitly demonstrate that UAV reconnaissance reduces information uncertainty, rather than just reducing distance.
2. Better than ground-only and nearest-resource in peak / night / bad weather / multi-incident scenarios.
3. Compare with rolling optimization or random optimization baseline to illustrate the real-time and performance tradeoff.
4. Write down transportation implications: unattended platform deployment, resource preset, emergency response system.

---

## 6. The third journal: F-J3 Risk Assurance Method

### 6.1 Suggested topics

**Coverage-to-Risk Assurance for UAV Safety-Critical Scenario Testing**

### 6.2 Submission goals

The main bet depends on the result:

- Bias robot safety testing: T-RO/IEEE Transactions on Automation Science and Engineering.
- Partial traffic intelligence system: T-ITS.
- Partial statistical guarantees and learning risks: Machine Learning / Artificial Intelligence journal direction.

### 6.3 Why is this article needed?

F-J1 can answer "how to generate and cover scenes", but journal reviewers may also ask:

> Now that you've covered these scenarios, can you tell how secure the system is? What is the relationship between coverage and true risk?

That's where the F-J3 comes in. It is not another benchmark, but connects scene coverage, importance sampling, scenario approach and conformal risk control. Campi and Garatti's scenario approach gives a feasibility probability guarantee under random scenario constraints [24], and Conformal Risk Control provides a distribution-free risk control framework [25]. These can be adapted into statistical guarantees for UAV safety testing.

### 6.4 Method Design

Method name suggestion: **CovRisk-UAV**.

Core idea:- Divide UAV scenario space into coverage cells;
- Estimate failure / near-miss / violation risk within each cell;
- Use importance weighting to correct the sampling bias of accelerated testing;
- Use conformal risk control to give finite-sample risk upper bound;
- Give confidence intervals for planner ranking instead of just average collision rate.

Formally, target risk can be defined:

$$
R(\pi)=\mathbb{E}_{s\sim P_{\text{ODD}}}[\ell(\pi,s)],
$$

Where $\pi$ is the UAV planner, $s$ is the scenario, and $\ell$ is collision, near-miss or constraint violation loss.

Since the test scenario comes from the accelerated distribution $Q(s)$, importance correction is required:

$$
\hat{R}(\pi)=
\frac{1}{N}\sum_{i=1}^{N}
\frac{P_{\text{ODD}}(s_i)}{Q(s_i)}
\ell(\pi,s_i).
$$

Using conformal / scenario bounds again gives:

$$
P(R(\pi)\leq \hat{R}_{\alpha}(\pi))\geq 1-\alpha.
$$

### 6.5 Baselines| Baseline | Comparison purpose |
|----------|----------|
| Empirical failure rate | No confidence guarantee |
| Bootstrap confidence interval | Statistical baseline |
| Importance sampling only | Correct sampling bias only |
| Scenario approach only | Only feasibility probability bound |
| Conformal risk control | Risk control baseline |
| CovRisk-UAV full | coverage-aware risk bound |

### 6.6 Minimum results that a journal can publish

1. Verify that the risk upper bound calibration is valid in synthetic known-risk scenarios.
2. Give risk confidence intervals to different planners in the F-J1 scenario library.
3. Explain that accelerated testing cannot directly use the original failure rate and needs distribution correction.
4. Prove that coverage-aware risk bound is tighter or more stable than naive random testing.

---

## 7. The fourth journal: F-J4 urban ODD to local scene

### 7.1 Suggested topics

**City2Local-UAV: Hierarchical Scenario Generation from Urban ODDs to Local Obstacle Compositions**

### 7.2 Submission goals

Main cast: TR-C/T-ITS.  
It is not recommended to rank first in this direction for the time being, because it requires more urban data processing and case support.

### 7.3 Core Issues

Although the local 50m x 50m x 50m scene is controllable, the real low-altitude traffic risks come from the urban structure: road grades, building density, bridges, service areas, no-fly zones, hospitals, schools, interchanges and accident-prone points. F-J4 needs to map city-level ODD to local obstacle composition.

### 7.4 Method design

```text
City ODD
  -> functional zone and road segment extraction
  -> local UAV test-cell sampling
  -> obstacle grammar instantiation
  -> coverage-aware scenario selection
  -> simulator-ready scenario package
```

### 7.5 Minimum results that the journal can publish1. At least two city or highway area case studies.
2. It can be proved that city-aware generation is more realistic than purely random local generation.
3. It can be proved that the generated local scene is better than the artificial template in terms of coverage and criticality.
4. Output a reproducible pipeline from urban functional areas to local scene combinations.

---

## 8. Recommended route: Which article to write first

The one that should be most focused on right now is **F-J1**, not four chapters at the same time.

### 8.1 Why F-J1 is the priority

- It converts 76 million exploration logs into paper assets.
- It can absorb F1 benchmark and F2 accelerated testing, and is large enough for journals.
- It has reuse value for the next three articles.
- It is the easiest to form a complete experimental closed loop: scenario definition, generation method, baselines, planners, metrics, and cross-simulation verification.

### 8.2 The main line contribution of F-J1 should be reduced to three

1. **UAV safety-critical scenario coverage taxonomy**
   Define UAV low altitude ODD, scenario parameters, coverage metric and failure taxonomy.

2. **Coverage-guided accelerated testing algorithm**
   Generate dangerous but feasible scenarios in coverage holes and high-criticality regions.

3. **Reusable benchmark and evaluation protocol**
   Using multiple planners, multiple baselines, and multiple simulation levels proves that this benchmark can stably evaluate UAV safety.

Don’t write 6-8 contributions. The three items in the journal introduction are the clearest.

### 8.3 F-J1 The point most likely to be rejected| Risk | Cause | Treatment |
|------|------|------|
| Considered just a simulation platform | benchmark has no algorithmic contribution | coverage-guided accelerated testing must be highlighted |
| Considered copied from autonomous driving | Lack of UAV features | Emphasis on 3D dynamics, wind, battery, low-altitude obstacles, landing/emergency tasks |
| The dangerous scene is considered unrealistic | adversarial too strong | Add feasibility and naturalness filters |
| Considered only valid for a single planner | Overfitting | At least 4 types of planners |
| Considered to lack significance as a transportation system | UAVs are just robots | Written as low-altitude ITS safety evaluation |

---

## 9. Dismantling of recent experimental tasks

### Week 1: Freeze F-J1 problem formulation

- Fixed target journal: T-ITS.
- Fixed main title and three contributions.
- Freeze 50m x 50m x 50m test cell.
- Define the scene parameter table: geometry, obstacle, dynamic agent, weather, sensor, task, risk label.

### Weeks 2-3: Processing 76 million exploration logs

- Sampling 10,000-50,000 items for preliminary analysis.
- Statistical coverage holes.
- Clustering failure modes: collision, near-miss, timeout, oscillation, energy violation, infeasible scene.
- Output two core maps: coverage heatmap and failure taxonomy.

### Weeks 4-6: Implementing baseline generators- random, grid, LHS.
- Scenic-style constrained generator.
-BO/CMA-ES.
- adversarial obstacle editing.
- feasible criticality filter.

### Weeks 7-9: Implementing Cov-ATUAV

- coverage memory.
- criticality score.
- feasibility filter.
-adaptive generator.
- planner evaluation harness.

### Weeks 10-12: Main Experiment

- Compare failure discovery rate, coverage gain, invalid rate, acceleration factor.
- Test RRT*, MPC-lite, PPO, vision policy.
- Do AirSim / Flightmare / AvoidBench subset migration verification.

### Weeks 13-16: Writing First Draft of T-ITS

- Introduction focuses on low-altitude ITS safety testing.
- Related work is divided into scenario-based safety evaluation, safety-critical scenario generation, UAV simulation and obstacle avoidance.
- Experiments use main table + coverage graph + failure discovery curve + cross-simulator transfer.

---

## 10. Things not currently recommended- Don’t write 5 essay titles at once and then work on them in parallel.
- Don't do F-J4 city ODD first, because the data pipeline will slow down the first article output.
- Don't mix Shandong Expressway Emergency and F-J1 in one article, otherwise the main line of T-ITS will be scattered.
- Don’t write 76 million explorations as the final result, it is now a data asset, not a conclusion.
- Don't just report collision rate, you must report coverage, criticality, invalid rate and planner ranking stability.

---

## 11. References

[1] International Organization for Standardization. "ISO 34502:2022 Road vehicles — Test scenarios for automated driving systems — Scenario based safety evaluation framework." 2022. URL: <https://www.iso.org/standard/78951.html>

[2] ASAM. “ASAM OpenSCENARIO DSL: Key Terminology and Conceptual Overview.” URL: <https://publications.pages.asam.net/standards/ASAM_OpenSCENARIO/ASAM_OpenSCENARIO_DSL/latest/conceptual-overview/key_terms.html>

[3] ASAM. “ASAM OpenODD: Model to ASAM OpenSCENARIO DSL Mapping Reference.” URL: <https://publications.pages.asam.net/standards/ASAM_OpenODD/ASAM_OpenODD/latest/specification/09_openscenario_dsl/09_01_overview.html>[4] Shuo Feng, Xintao Yan, Haowei Sun, Yiheng Feng, and Henry X. Liu. “Intelligent Driving Intelligence Test for Autonomous Vehicles with Naturalistic and Adversarial Environment.” *Nature Communications*, 12:748, 2021. DOI: 10.1038/s41467-021-21007-8. URL: <https://doi.org/10.1038/s41467-021-21007-8>

[5] Shuo Feng, Yiheng Feng, Chunhui Yu, Yi Zhang, and Henry X. Liu. "Testing Scenario Library Generation for Connected and Automated Vehicles, Part I: Methodology." *IEEE Transactions on Intelligent Transportation Systems*, 22(3):1573-1582, 2021. DOI: 10.1109/TITS.2020.2972211. URL: <https://doi.org/10.1109/TITS.2020.2972211>[6] Shuo Feng, Yiheng Feng, Haowei Sun, Shan Bao, Yi Zhang, and Henry X. Liu. "Testing Scenario Library Generation for Connected and Automated Vehicles, Part II: Case Studies." *IEEE Transactions on Intelligent Transportation Systems*, 22(9):5635-5647, 2021. DOI: 10.1109/TITS.2020.2988309. URL: <https://doi.org/10.1109/TITS.2020.2988309>

[7] Daniel J. Fremont, Tommaso Dreossi, Shromona Ghosh, Xiangyu Yue, Alberto L. Sangiovanni-Vincentelli, and Sanjit A. Seshia. “Scenic: A Language for Scenario Specification and Scene Generation.” *Proceedings of the 40th ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI)*, 2019. DOI: 10.1145/3314221.3314633. URL: <https://people.eecs.berkeley.edu/~sseshia/pubs/b2hd-fremont-pldi19.html>[8] Chejian Xu, Wenhao Ding, Weijie Lyu, Zuxin Liu, Shuai Wang, Yihan He, Hanjiang Hu, Ding Zhao, and Bo Li. “SafeBench: A Benchmarking Platform for Safety Evaluation of Autonomous Vehicles.” *Advances in Neural Information Processing Systems 35 (NeurIPS 2022) Datasets and Benchmarks Track*, 2022. URL: <https://proceedings.neurips.cc/paper_files/paper/2022/hash/a48ad12d588c597f4725a8b84af647b5-Abstract-Datasets_and_Benchmarks.html>

[9] Quanyi Li, Zhenghao Peng, Lan Feng, Zhizheng Liu, Chenda Duan, Wenjie Mo, and Bolei Zhou. “ScenarioNet: Open-Source Platform for Large-Scale Traffic Scenario Simulation and Modeling.” *Advances in Neural Information Processing Systems 36 (NeurIPS 2023) Datasets and Benchmarks Track*, 2023. URL: <https://proceedings.neurips.cc/paper_files/paper/2023/hash/0c26a501df8fb919a0350e2df06b5d39-Abstract-Datasets_and_Benchmarks.html>[10] Jingkang Wang, Ava Pun, James Tu, Sivabalan Manivasagam, Abbas Sadat, Sergio Casas, Mengye Ren, and Raquel Urtasun. "AdvSim: Generating Safety-Critical Scenarios for Self-Driving Vehicles." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2021. DOI: 10.1109/CVPR46437.2021.00978. URL: <https://openaccess.thecvf.com/content/CVPR2021/html/Wang_AdvSim_Generating_Safety-Critical_Scenarios_for_Self-Driving_Vehicles_CVPR_2021_paper.html>

[11] Niklas Hanselmann, Katrin Renz, Kashyap Chitta, Apratim Bhattacharyya, and Andreas Geiger. “KING: Generating Safety-Critical Driving Scenarios for Robust Imitation via Kinematics Gradients.” *European Conference on Computer Vision (ECCV)*, 2022. DOI: 10.1007/978-3-031-19839-7_20. URL: <https://is.mpg.de/ps/publications/king_geiger2022>[12] Jiawei Zhang, Chejian Xu, and Bo Li. "ChatScene: Knowledge-Enabled Safety-Critical Scenario Generation for Autonomous Vehicles." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2024, pp. 15459-15469. URL: <https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_ChatScene_Knowledge-Enabled_Safety-Critical_Scenario_Generation_for_Autonomous_Vehicles_CVPR_2024_paper.html>

[13] Keyu Chen, Yuheng Lei, Hao Cheng, Haoran Wu, Wenchao Sun, and Sifa Zheng. "FREA: Feasibility-Guided Generation of Safety-Critical Scenarios with Reasonable Adversariality." arXiv:2406.02983, 2024. URL: <https://arxiv.org/abs/2406.02983>

[14] Shital Shah, Debadeepta Dey, Chris Lovett, and Ashish Kapoor. “AirSim: High-Fidelity Visual and Physical Simulation for Autonomous Vehicles.” *Field and Service Robotics*, Springer Proceedings in Advanced Robotics, 2017; arXiv:1705.05065. URL: <https://arxiv.org/abs/1705.05065>[15] Yunlong Song, Selim Naji, Elia Kaufmann, Antonio Loquercio, and Davide Scaramuzza. “Flightmare: A Flexible Quadrotor Simulator.” *Proceedings of the 4th Conference on Robot Learning (CoRL)*, PMLR 155, 2021. URL: <https://proceedings.mlr.press/v155/song21a.html>

[16] Hang Yu, Guido C. H. E. de Croon, and Christophe De Wagter. “AvoidBench: A High-Fidelity Vision-Based Obstacle Avoidance Benchmarking Suite for Multi-Rotors.” arXiv:2301.07430, 2023. URL: <https://arxiv.org/abs/2301.07430>

[17] Botian Xu, Feng Gao, Chao Yu, Ruize Zhang, Yi Wu, and Yu Wang. "OmniDrones: An Efficient and Flexible Platform for Reinforcement Learning in Drone Control." *IEEE Robotics and Automation Letters*, 9(3):2838-2844, 2024. DOI: 10.1109/LRA.2024.3356168. URL: <https://ieeexplore.ieee.org/document/10409589/>[18] Mihir Kulkarni, Theodor J. L. Forgaard, and Kostas Alexis. “Aerial Gym: Isaac Gym Simulator for Aerial Robots.” arXiv:2305.16510, 2023. URL: <https://arxiv.org/abs/2305.16510>

[19] Yash Vardhan Pant, Max Z. Li, Alena Rodionova, Rhudii A. Quaye, Houssam Abbas, Megan S. Ryerson, and Rahul Mangharam. “FADS: A Framework for Autonomous Drone Safety Using Temporal Logic-Based Trajectory Planning.” *Transportation Research Part C: Emerging Technologies*, 130:103275, 2021. DOI: 10.1016/j.trc.2021.103275. URL: <https://doi.org/10.1016/j.trc.2021.103275>

[20] Elsevier. “Transportation Research Part C: Emerging Technologies: Aims & Scope.” URL: <https://www.sciencedirect.com/journal/transportation-research-part-c-emerging-technologies>

[21] IEEE Intelligent Transportation Systems Society. “IEEE Transactions on Intelligent Transportation Systems (T-ITS): Scope.” URL: <https://ieee-itss.org/pub/t-its/>[22] Shandong Expressway Group Co., Ltd. “‘Shandong Expressway Comprehensive Inspection Flight Service System’ goes online.” 2025. URL: <https://www.sdhsg.com/article/72553>

[23] Zhao Xiangmo, Zhao Yifei, Lu Nengchao, et al. "A review of research on key resource allocation for highway traffic accident emergency." *Transactions of Transportation Engineering*, 2024. DOI: 10.19818/j.cnki.1671-1637.2024.06.001. URL: <https://transport.chd.edu.cn/cn/article/doi/10.19818/j.cnki.1671-1637.2024.06.001>

[24] Marco C. Campi and Simone Garatti. “The Exact Feasibility of Randomized Solutions of Uncertain Convex Programs.” *SIAM Journal on Optimization*, 19(3):1211-1230, 2008. DOI: 10.1137/07069821X. URL: <https://epubs.siam.org/doi/10.1137/07069821X>

[25] Anastasios N. Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei, and Tal Schuster. “Conformal Risk Control.” *International Conference on Learning Representations (ICLR)*, 2024. URL: <https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf>[26] IEEE Intelligent Transportation Systems Society. “IEEE Transactions on Intelligent Vehicles.” URL: <https://ieee-itss.org/pub/t-iv/>

[27] IEEE Robotics and Automation Society. “IEEE Transactions on Automation Science and Engineering.” URL: <https://www.ieee-ras.org/publications/t-ase>

[28] IEEE Reliability Society. “IEEE Transactions on Reliability.” URL: <https://rs.ieee.org/publications/transactions-on-reliability/>

[29] Elsevier. “Reliability Engineering & System Safety: Aims & Scope.” URL: <https://www.sciencedirect.com/journal/reliability-engineering-and-system-safety>

[30] Elsevier. “Safety Science: Aims & Scope.” URL: <https://www.sciencedirect.com/journal/safety-science>

[31] Wiley. “Journal of Field Robotics: Overview.” URL: <https://onlinelibrary.wiley.com/journal/15564967>

[32] Elsevier. “Robotics and Autonomous Systems: Aims & Scope.” URL: <https://www.sciencedirect.com/journal/robotics-and-autonomous-systems>[33] Elsevier. “Transportation Research Part E: Logistics and Transportation Review: Aims & Scope.” URL: <https://www.sciencedirect.com/journal/transportation-research-part-e-logistics-and-transportation-review>

[34] IEEE Robotics and Automation Society. “IEEE Transactions on Robotics.” URL: <https://www.ieee-ras.org/publications/t-ro>

[35] IEEE Intelligent Transportation Systems Society. “IEEE Open Journal of Intelligent Transportation Systems.” URL: <https://ieee-itss.org/pub/oj-its/>

---

## Appendix: Conclusion of this optimization

1. When journal priority is given, Paper F should not be spread out into many small papers.
2. The first article should combine benchmark and accelerated testing to form the main paper of T-ITS.
3. Shandong expressway emergency applications should be independent as TR-C and should not be mixed into the first article.
4. Risk assurance papers can be used as reserves for high-level method journals in the mid- to late-stage period.
5. Urban ODD to local scene generation is temporarily ranked fourth, and will be advanced after the data pipeline is stabilized.