---
title: "Paper: RL-Based Cooperative Optimization of Channelization and Ramp Metering in Weaving Areas"
description: "A first-author SCI Q3 paper introducing a reinforcement learning approach to coordinate channelization design and ramp metering for urban expressway weaving areas."
pubDate: 2023-04-10
tags: ["Traffic Engineering", "Reinforcement Learning", "Expressway", "SUMO", "SCI Q3"]
category: Paper
doi: "10.1155/2023/4771946"
journal: "Journal of Advanced Transportation"
---

# RL-Based Cooperative Optimization of Channelization and Ramp Metering

**Authors:** Diantao Deng, Bo Yu, Duo Xu, Yuren Chen, You Kong  
**Journal:** *Journal of Advanced Transportation*, 2023  
**DOI:** [10.1155/2023/4771946](https://doi.org/10.1155/2023/4771946)  
**Impact Factor:** 2.3 | **Category:** SCI Q3

---

## Motivation

Urban expressway weaving areas are notorious for congestion. When vehicles need to merge or diverge across multiple lanes in a short distance, conflicts arise — and conventional single-strategy controls (either lane markings *or* ramp signals, never both together) typically fail to handle them effectively.

The key insight of this paper: **channelization (how lanes are physically divided) and ramp metering (how vehicles are admitted from on-ramps) are not independent problems.** Optimizing them jointly — rather than in isolation — can unlock significant performance gains.

## Method

The proposed framework uses a **Q-learning** agent to dynamically coordinate both strategies:

1. **Channelization strategies** — two types of lane-marking configurations that guide how vehicles merge/diverge
2. **Ramp metering** — adaptive signal control at the on-ramp to regulate inflow
3. **Cooperative mode** — Q-learning decides the optimal combination of both in real time

The environment is built in **SUMO** (Simulation of Urban Mobility), with real traffic data collected via **UAV aerial surveys** used to calibrate and validate the simulation.

## Results

The cooperative method significantly outperforms all alternatives. Lane-3 — the most heavily impacted by merge conflicts — sees a dramatic **37% improvement** in average vehicle speed:

- **Lane-1:** +14.51% average speed increase
- **Lane-2:** +14.81% average speed increase
- **Lane-3:** +37.03% average speed increase

## Key Takeaways

- **Joint optimization beats isolated strategies.** Traffic control is a systems problem; treating it as such pays dividends.
- **Q-learning is viable for traffic signal control** even without a full dynamics model — the agent learns the optimal policy purely from reward signals in simulation.
- **SUMO + Python co-simulation** provides a practical platform for developing and testing RL-based traffic controllers before real-world deployment.
- **UAV-based data collection** offers a scalable way to obtain ground-truth traffic data for simulation calibration.

## Related Work

This paper draws on prior SUMO simulation research from the broader traffic engineering community, and sits alongside other RL-based signal control work in the literature. The SUMO-Python co-simulation pipeline developed here became the foundation for the [Simulation Platform project](/) referenced in my About page.

---

*Full paper available at: [https://doi.org/10.1155/2023/4771946](https://doi.org/10.1155/2023/4771946)*
