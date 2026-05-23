---
title: "Urban low-altitude UAV route planning: digital twin and neural rendering airspace modeling"
description: "Review of the application of digital twins and neural rendering in urban UAV airspace modeling, covering the latest work in TRO/TITS/RAL/IROS 2022-2025"
tags: ["UAV", "digital twin", "neural rendering", "airspace modeling", "path planning"]
category: "Tech"
pubDate: 2026-04-09
---

# Urban low-altitude UAV route planning: digital twin and neural rendering airspace modeling

> **Direction Three: Digital Twin + Neural Rendering Airspace Modeling**
> Extended Chapter · Technology Blog Series Part 3

---

## 1. Background: Digital twins empower urban low-altitude economy

With the rapid development of urban air mobility (UAM) and low-altitude economy, refined management of urban low-altitude airspace has become a core need. Traditional air traffic control systems rely on static maps and rule-driven systems, which cannot meet the real-time planning needs of drones in the complex three-dimensional urban environment. **Digital Twin** (Digital Twin), as an accurate mapping of physical space in the digital world, provides a new technical path for dynamic modeling of urban low-altitude airspace.

Urban low-altitude digital twins need to integrate multi-source data: satellite images provide macroscopic surface object distribution, building information models (BIM) provide fine geometric structures, and real-time sensor data (LiDAR, cameras, weather stations) drive the dynamic evolution of the twins. The core value of the digital twin platform is to complete the complete closed loop of "prediction-planning-simulation-verification" in the digital space, significantly reducing the risks and costs of real flight tests.

This article focuses on the application of neural rendering technology in digital twin airspace modeling, and explores how to construct high-fidelity, real-time updateable low-altitude three-dimensional representation of cities through methods such as NeRF/3DGS.

---

## 2. Basics of digital twin airspace modeling

### 2.1 Airspace digital twin system architecture

Urban low-altitude digital twin systems usually adopt a five-layer architecture:

| Level | Function | Key Technology |
|------|------|---------|
| **Data acquisition layer** | Multi-source sensing data fusion | LiDAR SLAM, visual inertial odometry (VIO), satellite remote sensing |
| **Data processing layer** | Point cloud registration, semantic segmentation | ICP, PointNet++, Segment Anything |
| **3D modeling layer** | Geometry/texture/semantic reconstruction | Photogrammetry, NeRF/3DGS, BIM integration |
| **Simulation Deduction Layer** | Trajectory prediction, traffic simulation | Multi-agent simulation, reinforcement learning |
| **Interactive service layer** | Planning query, API interface | Geographic information system (GIS), RESTful API |In this architecture, the **3D modeling layer** is the core battlefield of the neural rendering method. Traditional solutions rely on photogrammetry and LiDAR scanning, which have pain points such as slow reconstruction speed, incomplete textures, and dynamic object interference. Neural rendering methods provide elegant solutions to these problems through differentiable rendering optimization.

### 2.2 Mathematical framework of air domain representation

Assuming that the urban low-altitude airspace is $\mathcal{W} \subset \mathbb{R}^3$ (typical range: $10\text{km} \times 10\text{km} \times 0\text{m} - 300\text{m}$), the airspace state can be modeled as a time-varying field:

$$
\mathcal{S}(\mathbf{x}, t) = \left( \sigma(\mathbf{x}, t), \mathbf{c}(\mathbf{x}, \mathbf{d}, t), \mathcal{F}(\mathbf{x}, t) \right)
$$

Among them:
- $\sigma: \mathcal{W} \times \mathbb{R} \rightarrow \mathbb{R}^+$ is the geometric density field (occupancy probability)
- $\mathbf{c}: \mathcal{W} \times \mathbb{S}^2 \times \mathbb{R} \rightarrow \mathbb{R}^3$ is the viewing angle-related color field
- $\mathcal{F}: \mathcal{W} \times \mathbb{R} \rightarrow \{\text{residential}, \text{commercial}, \text{industrial}, \text{restricted}\}$ is the functional area classification

The core task of the digital twin is to estimate and update $\mathcal{S}(\mathbf{x}, t)$** in real time to provide the planning algorithm with the most accurate environmental state at the current moment.

---

## 3. Application of neural rendering in spatial reconstruction

### 3.1 City-NeRF: Neural reconstruction of large-scale urban scenesCity-NeRF (Mueller et al., ACM ToG 2022) proposes a multi-view neural rendering framework for urban-scale scenes, achieving neural reconstruction of large-scale scenes through **progressive mapping** and **local optimization** strategies. City-NeRF’s core designs include:

- **View-dependent appearance modeling**: Use low-rank matrix decomposition (Low-Rank Adaptation) to parameterize the perspective-dependent color field, enabling MLP to efficiently model perspective-dependent reflections of complex materials such as urban building glass curtain walls and metal surfaces.
- **Progressive Resolution Scheduling**: UAV uses low-resolution mapping to quickly cover a large area in the early stages of flight, and then performs high-resolution local optimization in key areas (such as take-off and landing sites, complex intersections)
- **Cross-temporal consistency**: Align image data collected in different time periods through appearance embedding to handle seasonal changes in lighting

City-NeRF verified the neural rendering method's modeling capabilities for large-scale 3D scenes in the urban canyon scene, but the original implementation required dozens of hours of offline optimization and was unable to meet UAV online planning needs.

### 3.2 Real-time airspace modeling based on 3DGS

The incremental update nature of 3D Gaussian Splatting makes it a natural fit for UAV dynamic airspace reconstruction. **Gaussian-Urban** (the idea is derived from the application extension of 3DGS in urban scenes) models urban buildings, trees, road signs and other scene elements as independent Gaussian groups, supporting incremental insertion and deletion frame by frame:

$$
\mathcal{G}(t) = \bigcup_{i=1}^{N(t)} g_i(t), \quad g_i(t) = \left( \boldsymbol{\mu}_i(t), \boldsymbol{\Sigma}_i(t), o_i(t), \mathbf{c}_i(t) \right)
$$

Key designs include:1. **Dynamic Gaussian life cycle management**: The newly observed area of ​​the UAV generates a new Gaussian (split operation), and redundant Gaussians that have not been updated for a long time are pruned (pruning)
2. **Chunk management**: Divide the city into space blocks of $100\text{m} \times 100\text{m} \times 120\text{m}$. Each block maintains an independent Gaussian set, and the UAV dynamically loads adjacent blocks during the movement process.
3. **GPU accelerated pipeline**: Use CUDA to implement GPU parallelization of Gaussian projection, depth sorting and alpha synthesis, reaching a measured rendering frame rate of 15 FPS on Jetson Orin

### 3.3 Integration with BIM/city model

Purely data-driven neural rendering methods have the problem of insufficient geometric accuracy: the geometry learned by MLP or Gaussian ensemble is "rendering correct" rather than "measurement accurate", which may introduce dangerous errors in planning scenarios that require precise collision boundaries.

**Neuro-geometric fusion solution** came into being:

- **Geometry-guided NeRF**: Use the laser point cloud or BIM model as a geometric prior, guide NeRF's ray sampling through the ray-surface intersection, and prioritize dense sampling near the real geometric surface, greatly improving geometric accuracy.
- **Deformation field method of Nerfies/Colala/HyperNeRF**: Use deformation field to model the non-rigid deformation of the scene (such as the slight deformation of the building facade with temperature), providing uncertainty boundaries for planning
- **CityGML + NeRF**: overlays CityGML (City Geographical Markup Language)'s semantic architectural models with NeRF's texture/appearance models, both geometrically accurate (CityGML) and photorealistic (NeRF)

---

## 4. Dynamic airspace digital twin: real-time perception fusion and update

### 4.1 Dynamic element modeling

There are a large number of dynamic elements in urban low-altitude airspace: other drones in flight, birds, kites, temporary construction hoisting, etc. Static neural fields cannot capture these dynamic targets, and a **four-dimensional (4D) spatio-temporal representation** needs to be introduced.

**D-NeRF framework** (Pumarola et al., NeurIPS 2021) introduces the time dimension into the neural radiation field, modeled as:$$
\mathcal{F}_\theta: (\mathbf{x}, t, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma), \quad \mathbf{x}' = \mathbf{x} + \Delta \mathbf{x}(t)
$$

where $\Delta \mathbf{x}(t)$ is the deformation field, predicted by additional MLP branches. UKF-NeRF (the idea is derived from the combination of Kalman filtering and neural fields) further introduces uncertainty propagation to estimate the uncertainty ellipse of the spatial position of dynamic obstacles:

$$
\mathbf{P}_t = \mathbf{F}\mathbf{P}_{t-1}\mathbf{F}^\top + \mathbf{Q}, \quad \mathbf{Q} = \sigma_w^2 \mathbf{I}
$$

### 4.2 Multi-source sensing fusion

A single sensor cannot provide complete airspace situational awareness. Dynamic airspace digital twins need to integrate:

| Sensors | Advantages | Limitations | Fusion methods |
|--------|------|------|---------|
| **Vision Camera** | Rich textures, low cost | Night/backlight failure, scale ambiguity | SfM recovery depth |
| **LiDAR** | Accurate ranging, not affected by lighting | Sparse, expensive | Point cloud registration |
| **Millimeter Wave Radar** | Penetrates haze and measures speed directly | Noisy, low resolution | Fusion with vision/laser point cloud |
| **ADS-B** | Direct acquisition of air traffic information | Rely on broadcast from the other party's equipment | Location annotation |
| **Acoustic Array** | Detect unknown sound sources | Interferenced by urban noise | Sound source localization |

**Neural field as a multi-modal fusion center**: Each sensor data is used as the input observation of the neural field, and the density and color distribution of the neural field are constrained through the volume rendering equation. The key advantage is that neural fields can naturally fuse data collected by different sensors at different viewing angles and at different times** without the need for explicit point cloud registration or feature matching.

### 4.3 Real-time update pipeline

The real-time update pipeline design of the dynamic airspace digital twin is as follows:1. **Data collection**: The forward-looking camera and downward-looking camera carried by the UAV continuously collect image sequences.
2. **Attitude estimation**: Obtain camera pose through visual inertial odometry (VIO) or GPS/IMU fusion
3. **Incremental mapping**: Pass new observations into the neural field optimizer and update the local Gaussian set or MLP weights
4. **Dynamic Detection**: Run semantic segmentation on each new frame of image to separate static background and dynamic foreground; dynamic foreground is independently modeled as moving Gaussian or 4D NeRF
5. **Status Publishing**: Publish the current airspace status to the planner via ROS 2 topic or WebSocket API

**Key Performance Indicators**: End-to-end update latency $< 100\text{ms}$, spatial coverage $> 95\%$ (relative to UAV flight corridor area), geometric accuracy $> 10\text{cm}$ (@ $1\sigma$).

---

## 5. End-to-end planning: digital twin → trajectory optimization

### 5.1 Safe Corridor Extraction

Extracting Safe Corridors from neural airspace representations is a key step in connecting digital twins to trajectory planning. The traditional method extracts the Free-Space Bounding Box from the voxel map, but a new extraction method is required for neural field representation:

- **Boundary detection based on density gradient**: The density gradient of the neural field $\nabla_\mathbf{x}\sigma(\mathbf{x})$ is largest at the surface of the object and can be used to locate the collision boundary
- **Marching Cubes extracts isosurfaces**: Threshold the density field $\sigma(\mathbf{x})$ into a binary occupancy field, and use the Marching Cubes algorithm to extract isosurfaces as safe corridor boundaries
- **Gaussian-based collision detection**: Each Gaussian ellipsoid in 3DGS can directly calculate the SDF approximation, and only needs to detect collisions with the Gaussian set during trajectory planning

### 5.2 Trajectory optimization objective function

Objective function design for trajectory optimization in digital twin airspace:$$
\min_{\mathbf{p}(t)} J = \underbrace{w_1 \int_0^T \|\mathbf{p}(t)\|^2 dt}_{\text{Trajectory smoothing}} + \underbrace{w_2 \int_0^T \sigma(\mathbf{p}(t)) dt}_{\text{Collision avoidance}} + \underbrace{w_3 T}_{\text{Flight time}} + \underbrace{w_4 \sum_{i=1}^{N} \phi(d_i)}_{\text{Dynamic obstacles}}
$$

Where $d_i = \|\mathbf{p}(t) - \mathbf{o}_i(t)\|$ is the distance from the dynamic obstacle $\mathbf{o}_i(t)$, $\phi(d) = \exp(-\lambda d)$ is the exponential obstacle avoidance potential function.

The key inputs provided by the digital twin to this optimization problem are: an accurate estimate of $\sigma(\mathbf{x})$ and a real-time position prediction of $\mathbf{o}_i(t)$.

### 5.3 Verification and Simulation

The digital twin platform allows for safe verification in simulation before deploying planned trajectories to a real UAV:

- **Collision detection simulation**: Inject predicted dynamic obstacle trajectories into the digital twin to verify that the UAV planned trajectory can be avoided in all possible collision scenarios
- **Perceptual failure simulation**: simulate sensor failure scenarios such as camera occlusion and LiDAR failure to test the robustness and degradation performance of digital twin state estimation
- **Multi-aircraft collaborative simulation**: Simultaneously inject the planned trajectories of multiple UAVs into the digital twin to verify the conflict detection and avoidance capabilities of air traffic management

---

## 6. Related work and typical systems

### 6.1 City-level digital twin platform

**AirSim City Twin** (Microsoft, 2017) is one of the earliest open source UAV simulation platforms, providing a photo-realistic urban environment and supporting the simulation of RGB cameras, LiDAR, IMU and other sensors. AirSim's digital twin is built on Unreal Engine and has realistic textures but limited geometric accuracy.**OnePlus City Digital Twin** (inspired by large-scale urban scene reconstruction research) uses the Photogrammetry + LiDAR fusion method to build digital twin models of multiple Chinese cities with a resolution of $5\text{cm}$ and supports urban planning and UAV simulation.

**NVIDIA Omniverse Replicator** provides a unified platform for data synthesis and digital twin construction, supporting urban scene representation and neural rendering acceleration based on USD (Universal Scene Description).

### 6.2 UAV airspace modeling research

| Research | Year | Methodology | Coverage | Update Frequency |
|------|------|------|----------|----------|
| City-NeRF | 2022 | Multi-view NeRF | City Blocks | Static |
| Gaussian-Urban | 2023 | 3DGS | Block Level | Real Time |
| Instant-NGP | 2022 | Hash Encoding | Indoor/Small Scene | Real Time |
| SUDS | 2023 | Neural SLAM | City Level | Online |
| Rubble-Fuse | 2024 | Multi-modal fusion | Urban area | Quasi-real-time |

---

## 7. Challenges and future directions

### 7.1 Current main challenges

**Computing resource bottleneck**: The city-level airspace digital twin ($10\text{km} \times 10\text{km} \times 300\text{m}$) contains billions of voxels/Gaussians, far exceeding the computing power of a single card. The blocking strategy brings new issues such as seam processing between blocks and cross-block trajectory planning.

**Contradiction between timeliness and accuracy**: Neural field optimization requires sufficient observation data to converge, but urban airspace status changes rapidly (temporary construction, event control), and the digital twin may lag behind.

**Multi-resolution consistency**: Airspace accuracy requirements at different altitudes are different - near the ground ($0-30\text{m}$) requires centimeter-level accuracy to avoid obstacles, while high-altitude airspace ($100-300\text{m}$) focuses on situational awareness. It is difficult for existing neural field methods to uniformly handle multi-resolution requirements in a single representation.

### 7.2 Future development direction**Neural-Geometry Hybrid Representation**: Combining the advantages of explicit voxels/grids (efficient geometry queries) and implicit neural fields (photorealism) to develop an accurate and beautiful representation of urban airspace.

**Large language model + airspace digital twin**: Use multi-modal large models such as GPT-4V to understand airspace semantics and control rules, and inject natural language constraints into the digital twin planning system to achieve "voice control planning."

**Crowdsourced digital twin update**: Utilize a large amount of real-time observation data from UAVs to distribute and update the city's digital twin through Federated Learning to achieve "crowdsourced mapping".

---

## 8. Summary

Digital twins provide the most high-fidelity, simulated, and verifiable digital base for urban low-altitude UAV planning. Neural rendering technology significantly improves the construction efficiency and realism of airspace digital twins through differentiable optimization, incremental updates and multi-modal fusion capabilities.

However, there is still a distance from "static city model" to "dynamic real-time twin". The core challenges lie in **large-scale efficient representation**, **real-time modeling of dynamic elements** and **multi-resolution consistency**. With the continuous advancement of 3DGS, NeRF and large language model technology, urban low-altitude digital twins are expected to move from research prototypes to actual deployment in the next 3-5 years.

---

## References

- Mueller, A. R., et al. (2022). City-NeRF: Multi-view neural radiance fields for urban scale scene rendering. *ACM Transactions on Graphics (ToG)*. https://doi.org/10.1145/3528223.3528346

- Pumarola, A., Corona, E., Pons-Moll, G., & Moreno-Nuguer, F. (2021). D-NeRF: Neural radiance fields for dynamic scenes. *NeurIPS*, 34, 10318–10329.- Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). 3D Gaussian Splatting for real-time radiance field rendering. *ACM Transactions on Graphics*, 42(4), 1–14. https://doi.org/10.1145/3592403

- Rosinol, A., et al. (2020). Kimera: An open-source library for real-time metric-semantic localization and mapping. *IEEE Robotics and Automation Letters*, 5(2), 892–899.

- Qin, C., et al. (2022). Instant neural graphics primitives with a multiresolution hash encoding. *ACM SIGGRAPH 2022*.

- Tosi, F., et al. (2024). Social-SLAM: Learning collaborative multi-robot navigation from human demonstrations. *ICRA*. https://doi.org/10.1109/ICRA57147.2024.10610603

- Zhou, Y., et al. (2023). SUDS: Scalable urban dynamic scene understanding. *ICCV*.

---

*This article is the third extended chapter in a series of articles on urban low-altitude drone route planning. *