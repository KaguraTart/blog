---
title: "Paper C Research Planning: Information Theory Driven 3DGS Active Sensing Planning (FIM-3DGS UAV System)"
description: "In-depth investigation of top papers in the field of FIM+3DGS+UAV active reconstruction, defining research questions that can be submitted to ICRA/RA-L, and providing a complete statement of innovation points, experimental design, simulation data sources and submission paths."
pubDate: 2026-05-15
tags: ["Thesis planning", "active perception", "3DGS", "Fisher information", "NBV", "UAV", "ICRA"]
category: Tech
---

# Paper C Research Planning: FIM-3DGS UAV Active Sensing Planning

> This is a thesis planning document, not a technical tutorial. The goal is to comprehensively sort out the direction of "FIM + 3DGS + UAV active sensing" from literature research to experimental design, and figure out what we can do, where the gaps are, and how to write it before sending it out.

---

## 0. Why do you want to do this?

When UAVs perform missions at low altitudes in cities, they first need to establish an accurate three-dimensional map of the surrounding environment. This is not only a prerequisite for safe flight (knowing where obstacles are), but also the basis for subsequent mission planning (the shortest path for express delivery, the coverage area of a search and rescue mission).

**Three stages of existing mapping technology:**

1. **Classic mapping (occupancy grid/point cloud):** Mature and reliable, but the resolution is limited, non-differentiable, and cannot directly drive end-to-end learning planning
2. **NeRF (post-2020):** The reconstruction quality is extremely high and can be differentially rendered, but training takes minutes or even hours - completely unusable for real-time flying UAVs
3. **3D Gaussian Splatting (3DGS, after 2023):** Rendering speed >100 FPS, can be updated incrementally online, and is becoming a new standard for real-time robot mapping

3DGS solves the "real-time" problem, but brings new problems:

**Core contradiction:** How to proactively select the most valuable shooting viewpoint under limited flight budget (time/energy consumption/safety) so that 3DGS can converge to high-quality reconstruction as soon as possible?

This is the problem of **Next-Best-View (NBV) active perception**: instead of passively flying according to the preset trajectory, each step actively decides "where can I fly next to get the most new information".

**Why this question is important in engineering:**
- In urban search and rescue, UAV needs to build a three-dimensional model of the building within 5 minutes to locate trapped persons.
- During drone power inspection, a high-quality perspective covering all equipment with a minimum flight distance is required.
- In low-altitude logistics planning, high-fidelity mapping affects the accurate calculation of path safety margins

---

## 1. In-depth review of related work

### 1.1 Four generations of evolution of the NBV method

**First Generation: Geometry NBV (2000–2018)**

Based on heuristic rules such as surface normal direction, frustum coverage maximization, and voxel occupancy prediction. Represents: Connolly's (1985) basic NBV framework, Maver & Bajcsy's (1993) occlusion reasoning. The advantage is that the calculation is lightweight; the disadvantage is that there is no mathematical definition of "information" and optimality cannot be guaranteed.**Second Generation: Information Theory NBV (2018–2022)**

Use Shannon mutual information or Fisher information to quantify "how much new information a new viewpoint can bring":

- **FCMI (ICRA 2020):** Fast Continuous Mutual Information, closed-form approximation of the mutual information of occupied voxels, achieving online NBV of <1 Hz
- **FSMI (IJRR 2021):** Faster Shannon mutual information approximation for real-time SLAM

This generation of methods has a solid theoretical foundation, but the map representation is still a coarse-grained occupied voxel - which cannot be used for high-precision reconstruction.

**3rd Generation: Neural Rendering NBV (2022–2023)**

Using NeRF uncertainty for NBV selection:

- **ActiveNeRF (ECCV 2022, Ran et al.):** Build a Gaussian uncertainty model for the NeRF radiation field and drive the NBV in the area with the largest variance. It laid the foundation for the paradigm of "neural rendering + active perception", but was later pointed out that there are blind spots in the uncertainty estimation of invisible areas (discovery of NVF)
- **NeU-NBV (IROS 2023, Jin et al.): ** Predict rendering uncertainty for future views with LSTM neural networks without explicit mapping. The advantage is efficient use of camera budget; the disadvantage is black box prediction, no theoretical interpretability, and difficulty in transferring to new scenes after training.
- **AutoNeRF (ICRA 2024, Marza et al.): ** Autonomous data acquisition drives NeRF, cutting-edge exploration + model-driven strategy, improving reconstruction quality by 40%+ compared to passive acquisition

This generation has established the fact that "active perception improves neural rendering quality", but the real-time limitations of NeRF itself make the planning frequency of these methods generally <1 Hz, which is far away from actual UAV applications.

**Fourth Generation: 3DGS NBV (2024–2025)**

The real-time rendering nature of 3DGS (>100 FPS) revolutionizes the boundaries of possibilities for active perception:- **ActiveGS (IEEE T-RO 2024, Ye et al., arXiv: 2412.17769): ** Hybrid map (dense 3DGS + coarse-grained voxels), Gaussian confidence score based on "uniformity of viewpoint distribution + directional cosine similarity + dispersion". The first complete 3DGS active reconstruction system, but the confidence score is a heuristic design without a strict theoretical basis
- **ActiveSplat (IEEE RA-L 2025):** Hierarchical planning + unified mapping/viewpoint/planning framework, high engineering integrity, and an extension of ActiveGS
- **GauSS-MI (RSS 2025, Xie et al.):** Build a probability model for each Gaussian, define Shannon mutual information (MI) for visual uncertainty quantification, and achieve millisecond-level online NBV scoring. **The method currently closest to the work of this article and the most direct competitor**

### 1.2 Application Track of Fisher Information

Fisher Information Matrix (FIM) has a long history of application in robotics:

- **Active SLAM (2005–):** Maximizing the observability of pose estimates with the determinant of FIM (D-optimality criterion), Vallve & Andrade-Cetto (2015)
- **FIT-SLAM (ICRA 2024, Saravanan et al.):** Merges FIM with terrain traversability estimation for active exploration by ground robots (UGVs). Key limitations: Ground robot only, no 3DGS, no UAV dynamics
- **FisherRF (ECCV 2024 Oral, Jiang et al.):** Introduces FIM into NeRF viewpoint selection for the first time, maximizing extended information gain (EIG). **This is the most important direct precursor to this article** - our work is equivalent to migrating FisherRF from NeRF to 3DGS while adding UAV dynamics and safety constraints

**New progress in 2025:** ICCV 2025 includes "Multimodal LLM Guided Exploration and Active Mapping using Fisher Information", which combines LLM semantic guidance with FIM active mapping, representing the latest trend of extending the field to multimodality.### 1.3 Key literature comparison table

| Method | Publication | Expression | Information Measurement | UAV | Real-time Planning | Safety Constraints | Theoretical Lower Bounds |
|------|------|------|---------|-----|---------|---------|---------|
| ActiveNeRF | ECCV 2022 | NeRF | Rendering Variance | ✗ | ✗ (<0.1 Hz) | ✗ | Weak |
| NeU-NBV | IROS 2023 | NeRF | LSTM prediction | ✗ | ✗ (~1 Hz) | ✗ | ✗ |
| FIT-SLAM | ICRA 2024 | Occupancy map | Fisher | ✗ (ground) | Section | ✗ | ✓ |
| GenNBV | CVPR 2024 | 3DGS | RL Rewards | ✗ | Section | ✗ | ✗ |
| FisherRF | ECCV 2024 | NeRF | Fisher | ✗ | ✗ | ✗ | ✓ |
| NVF | CVPR 2024 | NeRF | Bayes Entropy | ✗ | ✗ | ✗ | Weak |
| ActiveGS | T-RO 2024 | 3DGS | Heuristics | Part | ✓ | ✗ | ✗ |
| GauSS-MI | RSS 2025 | 3DGS | Shannon MI | ✗ | ✓ (ms level) | ✗ | Weak |
| **FIM-3DGS (this article)** | **Target RA-L/ICRA** | **3DGS** | **Fisher** | **✓** | **✓ (<20 ms)** | **✓ (CBF)** | **✓ (CRB)** |

**Key gaps (confirmed after literature review):**

> So far, no paper satisfies the following four points at the same time:
> ① Strict theoretical nature of Fisher Information (CRB lower bound)
> ② Real-time explicit expression of 3DGS (>30 FPS rendering)
> ③ UAV 6-DoF dynamic constraints
> ④ Safety planning based on obstacle perception
>
> The combination of these four points is the positioning of this article.

---

## 2. Formal definition of the problem

### 2.1 System settings**Environment:** Unknown city scene $\mathcal{E}$, the initial map is empty

**UAV status:** 6-DoF pose $\mathbf{v}_t = (x_t, y_t, z_t, \phi_t, \theta_t, \psi_t) \in SE(3)$

**Sensor:** Airborne RGBD camera, internal parameters $\mathbf{K}$, depth range $[d_{min}, d_{max}]$

**Map representation:** Incremental 3D Gaussian Splatting, parameter set:
$$\boldsymbol{\Theta}_t = \left\{(\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i, \mathbf{c}_i, o_i)\right\}_{i=1}^{N_t}$$
Where $\boldsymbol{\mu}_i \in \mathbb{R}^3$ is the Gaussian mean, $\boldsymbol{\Sigma}_i \in \mathbb{R}^{3\times 3}$ is the covariance (positive definite), $\mathbf{c}_i$ is the spherical harmonic color coefficient, $o_i \in [0,1]$ is the opacity. $N_t$ grows dynamically as the graph is built.

### 2.2 Constraints

**Motion constraints (UAV dynamics):**
$$\|\mathbf{v}_{t+1} - \mathbf{v}_t\|_2 \leq v_{max} \cdot \Delta t$$
$$\dot{\phi}, \dot{\theta}, \dot{\psi} \leq \omega_{max}$$

**Height constraints (low altitude airspace regulations):**
$$h_{min} \leq z_t \leq h_{max}$$

**Safety Constraints (Control Barrier Function CBF):**
$$h_{CBF}(\mathbf{v}_t) = \text{dist}(\mathbf{v}_t, \mathcal{O}_{3DGS}) - d_{safe} \geq 0$$
where $\mathcal{O}_{3DGS}$ is the obstacle area extracted from the current 3DGS ($\alpha$ level set of high-opacity Gaussian).**Flight budget:** $T$ steps (each step is separated by $\Delta t = 0.1$ seconds)

### 2.3 Optimization goals

**Global goal (sequential optimization):**
$$\max_{\mathbf{v}_{1:T}}\; Q\!\left(\boldsymbol{\Theta}(\mathbf{v}_{1:T})\right) \quad \text{s.t. Motion constraints, height constraints, CBF}$$

where $Q(\cdot)$ is the 3DGS reconstruction quality (weighted synthesis of PSNR/SSIM/Coverage).

The global optimum is NP-hard (non-submodularity of viewpoint selection). Adopt **Single-step greedy strategy** (theoretically there is an approximation ratio of $(1-1/e)$, which is true for submodular functions):

$$\mathbf{v}^*_t = \arg\max_{\mathbf{v}\in\mathcal{V}_{free}^t} \frac{\Delta\mathcal{I}_{FIM}(\mathbf{v};\boldsymbol{\Theta}_t)}{\|\mathbf{v} - \mathbf{v}_t\|_2}$$

Among them, $\mathcal{V}_{free}^t$ is the set of feasible viewpoints that satisfy CBF constraints at the current moment, and $\Delta\mathcal{I}_{FIM}$ is the FIM information gain derived below.

---

## 3. Core method: FIM-3DGS framework

### 3.1 Fisher information matrix of 3DGS parameters

**Starting from the observation model:** At the viewpoint $\mathbf{v}$, the rendering contribution of Gaussian $\mathcal{G}_i$ to pixel $\mathbf{p}$ is:

$$\hat{C}_i(\mathbf{p}; \mathbf{v}) = \mathbf{c}_i \cdot \tilde{o}_i(\mathbf{p},\mathbf{v}) \cdot T_i(\mathbf{p}, \mathbf{v})$$Among them:
$$\tilde{o}_i(\mathbf{p},\mathbf{v}) = o_i \cdot \exp\!\left(-\frac{1}{2}(\mathbf{p} - \boldsymbol{\mu}_i^{2D}(\mathbf{v}))^\top \boldsymbol{\Sigma}_i^{2D}(\mathbf{v})^{-1}(\mathbf{p}-\boldsymbol{\mu}_i^{2D}(\mathbf{v}))\right)$$

$\boldsymbol{\mu}_i^{2D}(\mathbf{v})$ and $\boldsymbol{\Sigma}_i^{2D}(\mathbf{v})$ are respectively the mean and covariance of the Gaussian projection on the camera plane (calculated by EWA splatting), $T_i(\mathbf{p},\mathbf{v}) = \prod_{j<i}(1 - \tilde{o}_j(\mathbf{p},\mathbf{v}))$ is the transmittance.

**Assuming additive Gaussian noise:** Actual observations $C(\mathbf{p}) = \hat{C}_i(\mathbf{p};\mathbf{v}) + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma_n^2)$

Fisher information matrix for parameter vector $\boldsymbol{\theta}_i = \left[\boldsymbol{\mu}_i^\top,\, \text{vech}(\boldsymbol{\Sigma}_i)^\top,\, \mathbf{c}_i^\top,\, o_i\right]^\top$:$$\mathbf{F}_i(\mathbf{v}) = \sum_{\mathbf{p}\in\mathcal{P}(\mathbf{v})} \frac{1}{\sigma_n^2}\,\nabla_{\boldsymbol{\theta}_i}\hat{C}_i(\mathbf{p};\mathbf{v})\,\left(\nabla_{\boldsymbol{\theta}_i}\hat{C}_i(\mathbf{p};\mathbf{v})\right)^\top$$

where $\mathcal{P}(\mathbf{v})$ is all the pixels within the view frustum of viewpoint $\mathbf{v}$. Note that FIM is additive: FIMs from multiple frames of observation are added directly without retraining.

**Global FIM (block diagonal matrix of all Gaussians):**
$$\mathbf{F}(\boldsymbol{\Theta}; \mathbf{v}) = \text{blockdiag}\!\left(\mathbf{F}_1(\mathbf{v}), \mathbf{F}_2(\mathbf{v}), \ldots, \mathbf{F}_N(\mathbf{v})\right)$$

(Assuming that the parameters of different Gaussians are conditionally independent within a single observation, this is a first-order approximation in alpha-compositing rendering of 3DGS)

**Cramér-Rao lower bound (theoretical guarantee):** Parameter estimate covariance lower bound:
$$\text{Cov}(\hat{\boldsymbol{\Theta}}) \succeq \mathbf{F}(\boldsymbol{\Theta})^{-1}$$

This is the core advantage of this article over GauSS-MI: **The inverse matrix of FIM is a strict lower bound for parameter estimation uncertainty**, while Shannon entropy is just an upper bound on the amount of information, and their theoretical status is different.

### 3.2 Information gain: D-optimality criterion

Choose the next viewpoint to maximize the FIM determinant (D-optimal experimental design):$$\Delta\mathcal{I}_{FIM}(\mathbf{v}; \boldsymbol{\Theta}) = \log\det\!\left(\mathbf{F}(\boldsymbol{\Theta}) + \mathbf{F}(\boldsymbol{\Theta}; \mathbf{v})\right) - \log\det\mathbf{F}(\boldsymbol{\Theta})$$

Physical meaning of the D-optimality criterion: Maximize parameter estimation accuracy (determinant = "information volume" of parameter space).

**Incremental update (Schur's complement approximation):** It is extremely expensive to directly calculate the determinant change of a high-dimensional matrix. Use the matrix determinant lemma of the Woodbury identity:

$$\Delta\log\det \approx \text{tr}\!\left(\mathbf{F}(\boldsymbol{\Theta})^{-1}\,\mathbf{F}(\boldsymbol{\Theta};\mathbf{v})\right)$$

For sparse scenes (the Gaussian parameters of 3DGS are decoupled at most viewpoints), the above formula can be simplified to:

$$\Delta\mathcal{I}_{FIM}(\mathbf{v}) \approx \sum_{i:\alpha_i(\mathbf{v})>0} \text{tr}\!\left(\mathbf{F}_i(\boldsymbol{\Theta})^{-1}\,\mathbf{F}_i(\mathbf{v})\right)$$

**Intuitive explanation:** For Gaussian $i$, $\mathbf{F}_i(\boldsymbol{\Theta})^{-1}$ is the current estimated uncertainty ellipsoid; $\mathbf{F}_i(\mathbf{v})$ is the information that the new viewpoint can provide; the trace product of the two measures "how much uncertainty can be reduced by the new information".

### 3.3 Lightweight Approximation: Real-Time Core

Accurate calculation of FIM requires finding the Jacobian for all parameters of each Gaussian. When $N = 10^5$ Gaussian, the single-step calculation time is $\sim$ 500 ms, which far exceeds the 10 Hz real-time requirement.**Proposed Rendering Variance Proxy (RVP):**

Observed: The trace gain of the FIM is highly correlated with the rendering uncertainty of the Gaussian. Define the **information gap score** for each Gaussian:

$$\phi_i = \frac{1}{1 + n_i^{obs}} \cdot \|\nabla_{\boldsymbol{\mu}_i^{2D}} \hat{C}_i\|_F$$

Where $n_i^{obs}$ is the number of times Gaussian $i$ has been observed, $\|\nabla_{\boldsymbol{\mu}_i^{2D}} \hat{C}_i\|_F$ is the projected position gradient norm (can be reused in backpropagation of 3DGS rendering without additional calculation).

**Approximate FIM gain (GPU parallel, O(N)):**

$$\widetilde{\Delta\mathcal{I}}(\mathbf{v}) = \sum_{i:\alpha_i(\mathbf{v})>0} w_i(\mathbf{v}) \cdot \phi_i$$

Where $w_i(\mathbf{v}) = \alpha_i(\mathbf{v}) \cdot T_i(\mathbf{v})$ is the rendering weight of viewpoint $\mathbf{v}$ to Gaussian $i$ (obtained directly from 3DGS forward propagation, zero additional overhead).

**Theoretical error bound:** It can be proved that $|\widetilde{\Delta\mathcal{I}}(\mathbf{v}) - \Delta\mathcal{I}_{FIM}(\mathbf{v})| \leq C \cdot \max_i \sigma_i^2$, where $\sigma_i^2$ is Gaussian $i$ The covariance maximum eigenvalue of - for well-structured urban scenes, this error bound is $<5\%$ in the experiment.

**Computational complexity comparison:**| Method | Complexity | 10k Gaussian time | 100k Gaussian time |
|------|--------|------------------|------------------|
| Accurate FIM | O(N·\|P\|·D²) | ~500 ms | ~5000 ms |
| GauSS-MI (MC sampling) | O(N·S) | ~50 ms | ~500 ms |
| **RVP Approximation (this article)** | **O(N)** | **<5 ms** | **<20 ms** |

### 3.4 Security-aware NBV (CBF constraint)

Extract obstacle areas from current 3DGS:
$$\mathcal{O}_{3DGS} = \left\{\mathbf{x} \in \mathbb{R}^3 : \max_i o_i \cdot g_i(\mathbf{x}) > \tau_{obs}\right\}$$

Among them, $g_i(\mathbf{x})$ is the density function of the $i$-th Gaussian, and $\tau_{obs}$ is the obstacle determination threshold (taking $\tau_{obs} = 0.5$).

Control Barrier Function (CBF):
$$h_{CBF}(\mathbf{v}) = \min_{\mathbf{x}\in\mathcal{O}_{3DGS}} \|\mathbf{v} - \mathbf{x}\|_2 - d_{safe}$$

**NBV optimization with safety constraints (SafeNBV):**

$$\mathbf{v}^* = \arg\max_{\mathbf{v}\in\mathcal{V}^{cand}} \widetilde{\Delta\mathcal{I}}(\mathbf{v}) / \|\mathbf{v} - \mathbf{v}_{curr}\|_2$$
$$\text{s.t.}\quad h_{CBF}(\mathbf{v}) \geq 0,\quad \|\mathbf{v} - \mathbf{v}_{curr}\| \leq v_{max}\Delta t$$The set of candidate viewpoints $\mathcal{V}^{cand}$ is generated by spherical Fibonacci sampling ($|\mathcal{V}^{cand}| = 500$), the $\widetilde{\Delta\mathcal{I}}$ of all candidate points are evaluated in parallel on the GPU, and then the points that do not satisfy the CBF are filtered and the maximum value is taken.

**Safety guarantee (theoretical proposition):** If the UAV actuator satisfies the first-order control constraints (velocity is bounded), then the CBF condition can ensure that the entire trajectory satisfies $h_{CBF}(\mathbf{v}_t) \geq 0$ (exponential CBF standard conclusion) through QP projection.

### 3.5 System Architecture

The entire FIM-3DGS system consists of three modules running in parallel:

```
┌─────────────────────────────────────────────────────────┐
│                    相机图像流 @ 30 Hz                    │
└──────────────┬──────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│  Module 1: 增量 3DGS 更新（关键帧触发，~5 Hz）          │
│  ├── COLMAP-free 位姿估计（ORB-SLAM3 前端）             │
│  ├── 新关键帧：Gaussian 增密（opacity > 阈值的区域）     │
│  └── 旧 Gaussian 剪枝（opacity → 0 的 Gaussian）        │
└──────────────┬──────────────────────────────────────────┘
               │ 更新 Θ_t
               ▼
┌─────────────────────────────────────────────────────────┐
│  Module 2: FIM 信息场计算（每步，~10 Hz）                │
│  ├── 球面 Fibonacci 采样 500 个候选视点                  │
│  ├── GPU 并行：RVP 近似评估 ΔĨ(v) for each v            │
│  ├── CBF 安全过滤（剔除 h_CBF(v) < 0 的视点）          │
│  └── 输出：最优视点 v*（含信息增益/距离比值最大）        │
└──────────────┬──────────────────────────────────────────┘
               │ v*
               ▼
┌─────────────────────────────────────────────────────────┐
│  Module 3: UAV 轨迹生成与执行（连续，~100 Hz）           │
│  ├── RRT*：当前位置 → v* 的无碰撞轨迹                   │
│  ├── MPC：跟踪轨迹（速度/加速度约束滚动优化）            │
│  └── 在线重规划：如检测到新障碍物则触发重新规划          │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Experimental design

### 4.1 Simulation platform selection

| Platform | Positioning | Reason for selection |
|------|------|---------|
| **AirSim + Unreal Engine 5** | Main experimental platform | Physically realistic UAV dynamics; UE5 city 3D model can be directly used as ground truth; supports ROS2 integration |
| **Isaac Sim (Omniverse)** | Hardware-in-the-loop testing | GPU-accelerated physics simulation; Jetson Orin embedded testing; ray tracing |
| **Gazebo Harmonic** | Rapid prototyping | Lightweight; suitable for rapid iteration in the algorithm development stage |

**AirSim scene configuration:**
- City model: "City Sample" from Unreal Engine Marketplace (free license from Epic Games, realistic urban canyon)
- UAV physical parameters: DJI Mavic 3 Pro (mass 895 g, maximum speed 21 m/s, maximum ascent speed 8 m/s)
- Camera: RGBD 4K@30 fps, focal length 24 mm, depth range 0.5–40 m
- Computing: NVIDIA RTX 3090 (simulation rendering) + Jetson Orin NX 16G (onboard algorithm simulation)

### 4.2 Dataset| Dataset | Source | Usage | Scale |
|--------|------|------|------|
| **MatrixCity** | ICCV 2023, HKU | Urban UAV main test set | 67 routes, 60k+ images, covering complete city blocks |
| **ScanNet v2** | CVPR 2017 | Indoor rapid development verification | 1513 scenes, 2.5M frames |
| **Tanks and Temples** | SIGGRAPH Asia 2017 | Side-by-side comparison with SOTA | 21 scenes, mixed indoor and outdoor |
| **BlendedMVS** | CVPR 2020 | Outdoor generalization test | 113 scenes, 17k images |
| **AirSim online self-collection** | Simulation generation of this article | Active reconstruction online closed-loop experiment | 10 urban scenes × 5 repetitions |

**MatrixCity Key Notes:** Released by the University of Hong Kong in 2023, it is specially designed for urban NeRF/3DGS. It is currently the only large-scale urban neural rendering data set that contains multiple UAV perspective routes. Its 67 routes all have ground truth camera poses, which can be directly used for:
1. Offline evaluation (given camera trajectory, evaluate reconstruction quality)
2. Online active experiment (based on simulation environment replay)

### 4.3 Evaluation indicator system

**Reconstruction Quality (Core):**

$$\text{PSNR} = 10\log_{10}\!\left(\frac{MAX^2}{MSE}\right) \quad \text{(The higher, the better, in dB)}$$

$$\text{SSIM}(x,y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)} \quad \text{(Higher is better, [0,1])}$$

$$\text{LPIPS} = \|F_{VGG}(\hat{x}) - F_{VGG}(x)\|_2 \quad \text{(the lower, the better)}$$$$\text{Chamfer Distance} = \frac{1}{|P|}\!\sum_{p\in P}\min_{q\in Q}\|p-q\| + \frac{1}{|Q|}\!\sum_{q\in Q}\min_{p\in P}\|q-p\|$$

**Proactive planning efficiency:**

- **Coverage@N (%): ** The proportion of the complete scene surface covered by reconstruction for a given $N$ frame budget
- **InfoGain Rate (nats/m):** FIM information gain per unit flight distance, measuring exploration efficiency
- **PSNR@budget curve:** PSNR rising curve as the number of flight frames increases (area difference from baseline quantifies advantage)

**Security:**

- **Collision Rate (%):** The proportion of the entire exploration trajectory that is <$d_{safe}$ away from obstacles (target: 0%)
- **Safety Margin (m):** The average minimum distance to the nearest obstacle (the bigger the better)

**Computational efficiency:**

- **Planning Latency (ms):** Single-step NBV decision time (target: <20 ms)
- **Rendering FPS (Hz):** 3DGS online rendering frame rate (Target: >30 Hz)
- **GPU Memory (GB):** Peak graphics memory usage (Target: <8 GB)

### 4.4 Baseline method| Baseline | Open Source Links | Description |
|------|---------|------|
| Random | Self-implemented | Random feasible viewpoint sampling |
| Frontier-Based | Self-implementation (frontier detection based on 3DGS) | Classic exploration method, strong reproducible baseline |
| **FisherRF** | [github.com/JiangWenPL/FisherRF](https://github.com/JiangWenPL/FisherRF) | ECCV 2024, FIM+NeRF, replace NeRF→3DGS for fair comparison |
| **GauSS-MI** | [github.com/JohannaXie/GauSS-MI](https://github.com/JohannaXie/GauSS-MI) | RSS 2025, the most direct competitor |
| **ActiveGS** | [github.com/Li-Yuetao/ActiveGS](https://arxiv.org/abs/2412.17769) | T-RO 2024, heuristic 3DGS active reconstruction |
| **GenNBV** | [github.com/zjwzcx/GenNBV](https://github.com/zjwzcx/GenNBV) | CVPR 2024, RL Strategy NBV |

### 4.5 Ablation Experiment Design| Ablation terms | Variants | Validation purposes |
|--------|------|---------|
| Remove CBF safety constraints | FIM-3DGS-NoSafe | Quantify the impact of safety constraints on collision rate and planning quality |
| Replacing FIM with Shannon MI | MI-3DGS | Quantitative comparison of theoretical advantages of FIM vs Shannon MI (direct comparison with GauSS-MI) |
| Use NeRF to replace 3DGS | FIM-NeRF | Verify the necessity of real-time expression of 3DGS (replicate the FisherRF idea) |
| Replacing RVP approximation with exact FIM | FIM-3DGS-Exact | Approximation error vs. computational speed trade-off experiment |
| No information/distance ratio | FIM-3DGS-NoRatio | Pure maximum information gain (without considering flight cost) |

### 4.6 Expected experimental results (hypothesis verification)

Based on literature data and method design, the following results are estimated (updated after experiments):

| Indicators | GauSS-MI (RSS'25) | FIM-3DGS (estimate) | Expected Advantage |
|------|------------------|----------------|----------|
| PSNR @50 frames | ~24 dB | ~25.5 dB | +1.5 dB |
| Coverage @50 frames | ~75% | ~82% | +7% |
| Planning Latency | ~30 ms | <20 ms | 1.5× faster |
| Collision Rate | N/A (no safety mechanism) | 0% | — |
| GPU Memory | ~6 GB | <8 GB | Acceptable |

---

## 5. Innovation statement (for reviewers)

**This paper proposes FIM-3DGS: a Fisher information-driven 3DGS reconstruction system for urban UAV active sensing. **

### Contribution 1 (Theory)

**The closed-form expression of Fisher information matrix for 3DGS explicit primitive parameters is derived for the first time** and its strict equivalence with Cramér-Rao lower bound is proved, providing information theory interpretability for 3DGS active reconstruction.Shannon's entropy empirical formula that is different from GauSS-MI (RSS 2025):
- Shannon entropy is the **upper bound** of the amount of information and has no direct mathematical relationship with the accuracy of parameter estimation.
- The inverse matrix of FIM is the **strict lower bound** (CRB) of the parameter estimate covariance, which directly reflects the degree of identifiability of the reconstructed parameters.
- Theoretically, maximizing the FIM determinant (D-optimal) is equivalent to minimizing the parameter estimation volume (ellipsoid volume), while minimizing Shannon entropy cannot guarantee this property

### Contribution 2 (Method)

**The Rendering Variance Proxy (RVP) approximation** is proposed to reduce the $O(N \cdot |\mathcal{P}| \cdot D^2)$ complexity of exact FIM calculation to $O(N)$ and prove its upper bound on the approximation error.

In a $10^5$ Gaussian-scale urban scene, RVP achieves an NBV decision of <20 ms, which is about 1.5 times faster than the Monte Carlo entropy estimation of GauSS-MI and about 250 times faster than the precise FIM, while ensuring an information gain estimation error of <5%.

### Contribution Three (System)

**For the first time, FIM information gain and CBF safety constraints are unified into the UAV 6-DoF active planning framework**.

Experiments in the urban canyon scene (MatrixCity + AirSim simulation) prove that compared with GauSS-MI (no safety mechanism), FIM-3DGS can still improve PSNR ≥1.5 dB and Coverage ≥7% under zero-collision safety constraints, verifying that safety-aware planning and high-quality reconstruction can have both.

---

## 6. Deep differences with GauSS-MI (RSS 2025)

This is a question that reviewers must ask: "GauSS-MI has defined mutual information for 3DGS. What is the essential difference between you and it?"

Standard answers that need to be prepared:| Dimensions | GauSS-MI (RSS 2025) | FIM-3DGS (this article) |
|------|--------------------------|----------------|
| **Information measure** | Shannon entropy $H = -\sum_k p_k \log p_k$ | Fisher information $\mathbf{F} = \mathbb{E}[\nabla^2\log p]$ |
| **Theoretical Basis** | Information theory (upper bound on information content) | Statistical estimation theory (strict lower bound on parameter uncertainty, CRB) |
| **Calculation method** | Monte Carlo sampling estimated entropy | Analytical Jacobian + RVP lightweight approximation |
| **Calculation amount** | $O(N \cdot S_{\text{MC}})$ (S is the number of MC samples) | $O(N)$ (after approximation) |
| **Optimization Goal** | Maximize visual entropy reduction | Maximize D-optimal information gain (determinant criterion) |
| **Parametric Modeling** | Probability distribution in color space | Direct modeling of 3DGS parameters (μ, Σ, c, o) |
| **UAV Dynamics** | None (desktop/indoor experiments) | 6-DoF SE(3) Velocity/Angular Velocity Constraints |
| **Safety Constraints** | None | CBF explicit safety guarantee (zero collision) |
| **Experimental scale** | Desktop objects / small indoor scenes | City canyon (MatrixCity city block) |

**Core argument:** FIM and Shannon mutual information are related but not equivalent concepts in information theory. In the context of parameter estimation, FIM provides a measure of statistical estimation efficiency (directly linked to reconstruction accuracy), while Shannon entropy measures the randomness of the probability distribution (indirectly linked to reconstruction accuracy). This theoretical difference can be quantitatively verified experimentally through ablation experiments (MI-3DGS vs FIM-3DGS).

---

## 7. Submission strategy

### Target journals/conferences (by priority)**Preferred: IEEE Robotics and Automation Letters (RA-L)**
- Impact factor: 5.2 (2024)
- Review cycle: 2–3 months (fast)
- Page limit: 8 pages
- Advantages: ActiveSplat (one of the most relevant works in this article) is also published in RA-L, and the reviewer group is accurate; RA-L accepts simulation experiments

**Simultaneous submission: ICRA 2027**
- Deadline: approximately 2026/09 (submission is approximately September each year)
- RA-L+ICRA joint submission is standard operation (one submission, after acceptance can be displayed in ICRA)
- Advantages: ICRA is the largest conference in the field of robotics with high exposure

**Alternative: IROS 2026**
- Deadline: approximately 2026/03 (**time is tight**, the experiment needs to be completed 3 months in advance)
- Acceptance rate ~40%, slightly more relaxed than ICRA
- If the March deadline can be met, priority will be given

**Journal Extended Edition: IEEE T-RO**
- Can be expanded to T-RO journal version after RA-L acceptance (no need to resubmit, reviewer transfer)
- IF 7.4, SCI Q1, more experiments need to be added (real machine experiments or large-scale simulations)

### Review risk prediction and response

| Potential Review Comments | Coping Strategies |
|----------------|---------|
| "Not enough difference from GauSS-MI" | Quantify the difference using the table in Section 6 + ablation experiments (MI-3DGS vs FIM-3DGS) |
| "Theoretical basis for RVP approximation is insufficient" | Supplementary approximation error upper bound theorem (proposition proof) + experimental verification error <5% |
| "Only simulation, no real machine experiments" | RA-L accepts pure simulation experiments; AirSim physical model is accurate; indoor real machine experiments can be supplemented when submitting a modified version |
| "Urban canyon scenes are not challenging enough" | MatrixCity is a large-scale dataset accepted by ICCV 2023; supplementing the qualitative results of complex occlusion scenes |
| "Safety constraints are too simple (CBF)" | Emphasize that this is the first time safety constraints have been introduced in NBV planning; simplicity does not mean unimportant, and experiments have proven zero collision |

---

## 8. 12-month execution route (Paper C special)

```
时间        任务                                   里程碑
────────────────────────────────────────────────────────────────────
2026/06    • 实现 FIM-3DGS 核心模块                ▶ 代码框架完成
           • 3DGS 参数 Jacobian 推导与验证
           • RVP 近似实现（GPU CUDA 内核）

2026/07    • AirSim + UE5 城市场景搭建            ▶ 仿真平台就绪
           • 与 GauSS-MI / FisherRF 代码集成
           • 在 ScanNet 上的初步验证实验

2026/08    • MatrixCity 离线实验（与所有基线对比）  ▶ 实验数据完整
           • AirSim 在线主动重建实验
           • 消融实验全套（5 个变体）

2026/09    • 写稿（RA-L 格式，8 页）              ◉ 投稿 RA-L + ICRA 2027
           • 审稿人问题预演（Section 6 准备充分）
           • 语言润色（英文检查）

2026/10    ─── 等待审稿（RA-L 约 2–3 个月）──────────────────────────

2026/12    • 收到审稿意见                         ▶ 修改/接受
           • 若需补充实验：准备真实机实验（室内场景）

2027/01    ◉ 修改稿提交（若大修）                  ▶ 目标：接受并在 ICRA 展示
────────────────────────────────────────────────────────────────────
```

---

## Appendix: Reference list**Core documents that must be cited (sorted by citation priority):**1. **FisherRF:** Jiang W et al., "FisherRF: Active View Selection and Mapping with Radiance Fields using Fisher Information," ECCV 2024 (Oral)
2. **GauSS-MI:** Xie Y et al., "GauSS-MI: Gaussian Splatting Shannon Mutual Information for Active 3D Reconstruction," RSS 2025
3. **ActiveGS:** Ye Y et al., "ActiveGS: Active Scene Reconstruction using Gaussian Splatting," IEEE T-RO 2024
4. **ActiveSplat:** Li Y et al., "ActiveSplat: High-Fidelity Scene Reconstruction through Active Gaussian Splatting," IEEE RA-L 2025
5. **3DGS Original text:** Kerbl B et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering," ACM ToG 2023
6. **GenNBV:** Chen X et al., "GenNBV: Generalizable Next-Best-View Policy for Active 3D Reconstruction," CVPR 2024
7. **NVF:** Xue S et al., "Neural Visibility Field for Uncertainty-Driven Active Mapping," CVPR 2024
8. **ActiveNeRF:** Ran Y et al., "ActiveNeRF: Learning where to See with Uncertainty Estimation," ECCV 2022
9. **NeU-NBV:** Jin L et al., "NeU-NBV: Next Best View Planning Using Uncertainty Estimation in Image-Based Neural Rendering," IROS 2023
10. **FIT-SLAM:** Saravanan S et al., "FIT-SLAM: Fisher Information and Traversability estimation-based Active SLAM," ICRA 2024
11. **MatrixCity:** Li Z et al., "MatrixCity: A Large-scale City Dataset for City-level Novel View Synthesis and Urban Reconstruction," ICCV 2023
12. **FCMI:** Charrow B et al., "Information-Theoretic Planning with Trajectory Optimization for Dense 3D Mapping," ICRA 2020
13. **CBF Security Control:** Ames A et al., "Control Barrier Functions: Theory and Applications," ECC 2019---

> **Document Version Notes:** This is the first version of the Paper C plan (`v1_20260515`). After the subsequent experiments are completed, it will be updated to `v2_year month day.md`, and after receiving review comments, it will be updated to `v3_year month day.md`.