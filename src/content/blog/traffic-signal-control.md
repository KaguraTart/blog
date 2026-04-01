---
title: "Rethinking Traffic Signal Control: From Fixed Timing to Adaptive Intelligence"
description: "A reflection on the evolution of traffic signal control — from loop detectors and fixed plans to reinforcement learning and connected autonomous vehicles."
pubDate: 2026-04-02
tags: ["Traffic Engineering", "Reinforcement Learning", "Adaptive Control", "Smart City"]
category: Tech
---

# Rethinking Traffic Signal Control: From Fixed Timing to Adaptive Intelligence

Traffic signals are everywhere — we encounter them dozens of times a day, usually without a second thought. But if you've ever sat at a red light at 2 AM with no one in sight, or found yourself in a "green wave" that flows perfectly from one intersection to the next, you've already felt the consequences of how traffic signals are (or aren't) being optimized.

After spending years working with traffic simulation tools like SUMO and CARLA, and digging into the research around reinforcement learning for signal control, I've come to see this problem as one of the most interesting and underexplored challenges in urban mobility. Here's my honest reflection on where we are, and where we might be heading.

## The Traditional Approach: Fixed-Time and Actuated Control

Most traffic signals today still operate on one of two paradigms:

**Fixed-time control** assigns green phases according to pre-programmed schedules, typically derived from historical traffic counts. These schedules are often updated once a year — if at all. They're robust in the sense that they're predictable and easy to operate, but they're fundamentally reactive to the past, not the present.

**Actuated control** adds loop detectors or video cameras at intersections. When a vehicle is detected, the signal extends the green phase. It's better than fixed-time, but it's still fundamentally local — each intersection optimizes itself in isolation, with no awareness of what's happening upstream or downstream.

Both approaches share a core limitation: **they optimize for the intersection, not the network.** A green light that clears one intersection might create a queue that spills back and blocks three others. Traffic is a system, not a collection of independent nodes.

## The Network-Wide Problem: Why Coordination Changes Everything

Think about what happens during a typical morning rush hour. Vehicles pour in from residential areas onto arterial roads, and unless those arterial signals are coordinated, the result is a phenomenon called **progressive band failure** — the exact opposite of a green wave. Stop-and-go traffic emerges not because of high demand, but because of poor signal timing.

This is where **SCOOT** (Split Cycle Offset Optimization Technique) and **SCATS** (Sydney Coordinated Adaptive Traffic System) made their mark. Developed in the 1980s, these systems use real-time detection data to adjust cycle lengths, splits, and offsets across a network of intersections. They're genuinely effective — cities running SCOOT report 10–20% reductions in delay.

But here's the catch: SCOOT and SCATS are still based on **traffic flow models** — macroscopic or mesoscopic approximations of how vehicles move. These models were calibrated for conventional traffic. They struggle with:

- **Oversaturated conditions** (when demand exceeds capacity)
- **Non-recurrent congestion** (incidents, construction, events)
- **Mixed traffic** (human-driven vehicles sharing lanes with autonomous ones)
- **Long-range dependencies** (a bottleneck 3 intersections upstream)

The model-based approach has hit a ceiling. To go further, we need to step outside the model's comfort zone.

## Reinforcement Learning: A Different Kind of Optimizer

This is where my own research experience intersects with the broader picture. When I worked on the SUMO-Python co-simulation platform for urban expressway ramp metering, I started asking: can an agent learn to control traffic signals purely from experience, without an explicit model?

The idea behind **reinforcement learning (RL)** for traffic signal control is elegant:

- The **agent** is the traffic signal controller
- The **state** is the current traffic condition — queue lengths, waiting times, vehicle positions, possibly vehicle-to-infrastructure (V2I) data
- The **action** is the signal phase to switch to
- The **reward** is a combination of metrics: minimize total delay, maximize throughput, penalize queue overflow

The agent doesn't need to know the underlying dynamics of traffic flow. It learns a control policy directly from interactions with the environment — just like how AlphaGo learned to play Go without being told what the "best move" was at each step.

### What Makes It Hard

It's not all smooth sailing. Traffic signal RL faces several practical challenges:

**Sample efficiency.** Unlike a game where millions of self-play episodes are feasible, real-world deployment requires the agent to learn in simulation first. Building a faithful simulation is non-trivial — lane-changing behavior, driver aggressiveness, pedestrian unpredictability, all need to be modeled.

**Multi-agent coordination.** A single intersection is one thing. But a network of 50 intersections, each with its own RL agent, creates a multi-agent RL problem. The agents need to coordinate, not just individually optimize. Each agent's action affects its neighbors' observations.

**Safety and interpretability.** Traffic control is safety-critical. You can't let a learning agent experiment freely on a real intersection. The baseline must be safe, and learning must be constrained — e.g., conservative policy updates, human-in-the-loop fallback, or safety shields.

**Generalization.** An RL agent trained on morning rush hour data may fail spectacularly at noon or on a holiday weekend. Distribution shift is a real problem.

### Promising Directions

Despite the challenges, I'm genuinely excited about where this is heading. A few directions I find particularly promising:

**Graph neural networks for spatial awareness.** Rather than feeding each intersection a flat vector of its own queue lengths, GNNs let agents communicate over the network topology — sharing information about what's happening at neighboring intersections. This is how my internship work at Bosch China approached trajectory generation, and the approach transfers naturally to signal control.

**Hybrid physics-informed RL.** Combining first-principles traffic models (e.g., store-and-forward or cell transmission models) with RL gives you the best of both: the model provides structure and safety constraints, while RL handles the fine-grained optimization. This is where my SCI paper on expressway ramp metering sits — Q-learning backed by SUMO simulation, with channelization modeling.

**V2I and CAV-enabled control.** As connected autonomous vehicles (CAVs) penetrate the market, the feedback loop changes dramatically. Instead of inferring traffic state from sparse loop detectors, signals can receive real-time position and speed data from every vehicle in the network. This isn't just incremental improvement — it fundamentally changes what's observable and controllable.

## What We've Built and What Remains

In my own work — from the SUMO-CARLA fusion platform to the RL-based ramp metering paper — I've seen firsthand both the potential and the gaps. Simulation platforms are maturing rapidly. SUMO's TraCI interface lets you script everything in Python. CARLA adds the sensor fidelity needed for perception-based control. The tools are no longer the bottleneck.

What remains open, in my view:

1. **Benchmark environments** — we need standardized traffic network benchmarks with consistent metrics, like how ML has ImageNet and GLUE. The literature is full of single-intersection toy problems that don't transfer to real deployment.

2. **Fairness and equity** — most RL signal controllers optimize average delay. But a signal that serves the dominant traffic flow might systematically penalize pedestrians, cyclists, or vehicles on minor approaches. Multi-objective RL with fairness constraints is underexplored.

3. **Transfer from simulation to reality.** This is the last-mile problem. A policy that works in SUMO often fails in the real world due to sim-to-real gap. Domain randomization, system identification, and robust RL are all part of the solution.

4. **Public acceptance.** Adaptive signals that change their behavior non-deterministically can confuse drivers. There needs to be a human factors research thread alongside the control theory one.

## Closing Thoughts

Traffic signal control is one of those problems that looks simple on the surface but is deceptively deep. It's a control problem, a networking problem, a fairness problem, and increasingly a machine learning problem. The fact that 19th-century timing mechanisms still run most of the world's intersections is a testament both to their reliability and to how hard it is to do better.

I'm optimistic. The convergence of cheap sensors, V2X communication, better simulation, and smarter RL algorithms is creating a genuine opportunity to rethink urban mobility at the most fundamental level — one green light at a time.

---

*If you're working on traffic signal control, RL for transportation, or SUMO/CARLA simulation, feel free to reach out. Always happy to exchange ideas.*
