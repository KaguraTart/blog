---
title: "Urban low-altitude UAV route planning: multi-modal simulation data synthesis"
description: "Overview of the application of multimodal data synthesis and simulation platforms in urban UAV planning, covering the latest work of NeurIPS/ICRA/IROS/TRO 2022-2025"
tags: ["UAV", "Multimodal simulation", "Data synthesis", "Sim2Real", "reinforcement learning"]
category: "Tech"
pubDate: 2026-04-09
---

# Urban low-altitude UAV route planning: multi-modal simulation data synthesis

> **Direction 5: Multi-modal simulation data synthesis**
> Extended Chapter · Technical Blog Series Part 5

---

## 1. Background: The dual dilemma of data scarcity and security constraints

The training of urban low-altitude UAV planning algorithms (especially planners based on deep reinforcement learning) faces the dual dilemma of data scarcity and safety constraints:

**Data Scarcity**: The cost of collecting real flight data is high - it requires a lot of manpower control and site security, and the corner cases of complex urban scenes (extreme weather, sudden obstacles, signal interference) are difficult to cover with the system. Public data sets (such as MAVNet, UZH-FPV) are limited in scale and difficult to support the training of end-to-end deep learning models.

**Safety Constraints**: The reinforcement learning planner will produce a lot of "exploratory" behavior in the early stages of training. Direct training on real UAVs may lead to accidents such as collisions and loss of control. The simulation environment provides a **zero-risk training venue**, but the simulation-reality gap (Sim2Real Gap) makes the strategies trained in the simulation completely ineffective on the real UAV.

Multi-modal simulation data synthesis emerged as the times require - by building a high-fidelity multi-sensor simulation environment, systematically generating large-scale and diverse training data, while using Domain Randomization and Sim2Real migration technology to bridge the gap between simulation and reality.

---

## 2. Multi-modal sensor simulation

### 2.1 Why multimodality is needed

There are inherent capability boundaries for a single sensor. The safe operation of urban low-altitude UAV requires **redundant sensing capabilities**:

| Sensors | Core Competencies | Key Limitations | Complementarities |
|--------|---------|---------|--------|
| **RGB camera** | Texture recognition, semantic understanding | Failure at night, no depth information | Provide semantic segmentation capabilities |
| **LiDAR** | Accurate ranging, 3D mapping | Sparse, high cost | Provide accurate geometry |
| **Millimeter wave radar** | All-weather, direct speed measurement | Noisy, low resolution | Provide moving target detection |
| **Thermal Imaging** | Pedestrian detection, night vision | Temperature difference ambiguity, low resolution | Provide vulnerable road user detection |
| **Ultrasonic** | Obstacle avoidance at short range | Short range, susceptible to interference | Provide accurate close range perception |**Multimodal fusion** is not simply "installing a few more sensors", but designing a **fusion strategy** to make multi-source information complementary and redundant, and improve the system's **fault tolerance** (Fault Tolerance) - when a certain sensor fails, the system can still rely on other sensors to operate safely.

### 2.2 Sensor simulation principle

**RGB Camera Simulation** Based on Physically-based Rendering (PBR) pipeline:

$$
L_o(\mathbf{x}, \omega_o) = \int_{\Omega} f_r(\omega_i, \omega_o) \cdot L_i(\omega_i) \cdot \cos\theta_i \, d\omega_i
$$

Where $f_r$ is the bidirectional reflection distribution function (BRDF), $L_i$ is the incident irradiance, and the PBR pipeline generates photorealistic images by simulating the physical interaction of light and scene materials. Unreal Engine 5's Nanite virtual geometry system and Lumen global illumination system are currently the closest real-time rendering solutions to physical reality.

**LiDAR simulation** is usually based on raycasting: emitting rays from the LiDAR position along each scan line direction, detecting the intersection with the scene geometry, and returning the distance and reflection intensity:

$$
d = \min_{t > 0} \{ t : \mathbf{o} + t\omega \in \mathcal{O} \}
$$

Where $\mathcal{O}$ is the scene occupied geometry. High-end LiDAR simulations (such as NVIDIA FLIPS) can also simulate physical effects such as Multi-Echo and Waveform Broadening.

**Millimeter wave radar simulation** is based on the electromagnetic wave propagation model to simulate the multipath effect (Multipath), shadowing attenuation (Shadowing) and ground reflection (Ground Bounce) of the signal:

$$
P_r = P_t \cdot \frac{G_t G_r \lambda^2}{(4\pi)^3 R^4} \cdot \sigma \cdot L_{\text{atm}} \cdot L_{\text{multipath}}
$$Where $P_r$ is the received power, $R$ is the target distance, $\sigma$ is the radar cross section (RCS), and $L_{\text{multipath}}$ is the multipath fading factor.

### 2.3 Multimodal spatiotemporal synchronization

The key engineering challenge for multimodal data synthesis is spatiotemporal synchronization - each sensor data needs to be aligned in a unified time and coordinate system:

- **Hardware synchronization**: Each sensor shares the same clock trigger (such as GPS-PPS), and the timestamp error $< 1\text{ms}$
- **Software Timestamp Alignment**: Posterior time alignment based on sensor delay model (camera exposure delay, LiDAR scan cycle)
- **Spatial alignment**: Calibrate the external parameters of each sensor ($\mathbf{T}_{\text{camera}}^{\text{body}}$, $\mathbf{T}_{\text{lidar}}^{\text{body}}$, etc.) through the calibration board or CAD model, and unify the data to the airborne coordinate system

---

## 3. Comparison and selection of simulation platforms

### 3.1 Mainstream platform Hengping| Platform | Rendering engine | Multi-modal support | Physical simulation | Open source | UAV specialization | Applicable scenarios |
|------|----------|-----------|----------|------|----------|----------|
| **AirSim** | Unreal Engine | RGB-D / LiDAR / IMU | PX4 SITL | ✅ | ✅ Excellent | Aerial Path Planning |
| **Gazebo** | Ogre3D | Camera / LiDAR / IMU | ODE/Bullet | ✅ | ✅ Rich | Universal Robot Simulation |
| **Flightmare** | Unity | Camera / LiDAR / Events | - | ✅ | ✅ Excellent | UAV High Speed Flight |
| **Isaac Sim** | Omniverse | Full Modal | PhysX | Partial | General | Industrial Simulation |
| **SORDAMS** | Self-developed | Camera / LiDAR | Self-developed | ❌ | ✅ | Military-grade UAV simulation |
| **CAVS** | Self-researched | Full-mode | Self-researched | ✅ | ✅ | Low-altitude UTM research |
| **NeuroSIM** | Neural Rendering | Camera (NeRF) | - | Under Research | Exploratory | Neural Perception Training |

### 3.2 AirSim in-depth analysis

Microsoft AirSim is one of the most widely used UAV simulation platforms currently. It is built on Unreal Engine and provides photo-realistic urban scene simulation capabilities.

**Core Architecture**:
- **AirSim Plugin**: A plug-in that runs in Unreal Engine and handles sensor simulation, flight physics and API interfaces
- **PX4 SITL**: communicates with AirSim through the MAVLink protocol, supporting complete PX4 flight control firmware in-the-loop simulation
- **RPC Communication**: Provides Python/C++ API to support research-level flexible control**Advantages**:
- Photo-realistic rendering, the urban canyon scene is realistic
- Supports a variety of aircraft (MultiRotor, FixedWing, Rover)
- Rich sensor models (camera distortion, motion blur, depth of field)
- Dynamic changes in weather, lighting, and time

**Limitations**:
- Depends on Unreal Engine (large commercial engine, steep learning curve)
- Limited Linux support (mostly for Windows)
- The accuracy of physical simulation is not as good as that of professional robot simulators

### 3.3 Flightmare: High-speed UAV simulation

Flightmare developed by ETH Zurich is optimized for **high-speed UAV maneuver** scenarios and supports simulation of $10\text{m/s}^2+$ acceleration. It is an ideal tool for Aggressive Flight research.

Flightmare Features:
- **Modular Rendering Pipeline**: Interchangeable rendering engines (Unity/OpenGL), supporting large-scale urban environments
- **Large-scale scene library**: preset various scenes such as cities, forests, warehouses, etc.
- **Event Camera Simulation**: Supports event-based sensor (Event Camera) simulation, suitable for high-speed maneuvering scenes

### 3.4 Emerging Directions: Neural Simulation

**UniSim** (Zhou et al., NeurIPS 2023 / arxiv preprint) first proposed the concept of neural perception simulation, using neural radiation fields to model static backgrounds + explicit geometry to model dynamic objects, to achieve photo-realistic and controllable sensor data generation. UniSim’s core pipeline:

1. Collect a small amount of real-world data (about 20 minutes of driving video)
2. Train NeRF static background model + dynamic object explicit model
3. Adjust camera trajectories, add/delete objects, modify weather, and generate new scenes in NeRF
4. Neural rendering outputs RGB, depth, normal vector and other sensory data

The simulation data generated by this method is highly close to the real data, significantly narrowing the Sim2Real gap, but real-time performance is still a bottleneck (the current generation speed is about 0.1 FPS, non-real-time).

---

## 4. Domain Randomization and Sim2Real migration

### 4.1 Domain Randomization PrincipleThe core idea of ​​Domain Randomization (DR) is to randomize a large number of non-key attributes in simulation, forcing the learning algorithm to focus on the understanding of key attributes (geometric structure, semantic information), thereby generalizing to the real world.

**Typical randomization parameters**:

| Category | Parameters | Randomization range |
|------|------|---------|
| **Appearance** | Textures, lighting, weather | Color/intensity randomization, dynamic lighting |
| **Geometry** | Object size, position, orientation | Random position of non-key objects |
| **Sensor** | Internal parameters, noise, external parameters | Camera focus offset, LiDAR noise level |
| **Dynamics** | Mass, wind disturbance, delay | Parameters $\pm 20\%$ Random |
| **Background** | Scene complexity, number of objects | Random interference object density |

### 4.2 Online Domain Adaptation

The problem with pure DR is that over-randomization leads to inefficient training - the policy trains well in simple scenarios but degrades in complex scenarios. **Online Adaptation** (Online Adaptation) method continuously updates simulation parameters during the simulation-real migration process:

**Meta-Sim** (Kar et al., NeurIPS 2019) uses reinforcement learning to automatically learn the optimal Domain Randomization parameter distribution, with the goal of maximizing evaluation performance on real data:

$$
\theta^* = \arg\max_\theta \mathbb{E}_{\mathbf{s} \sim p_\theta} \left[ \text{Performance}(\pi_\theta, \text{Real}) \right]
$$

**SimBot** (Zhang et al., CoRL 2021) adopts a domain adaptation method to collect a small amount of interaction data of real robots at the same time during the training process, and uses these data to correct the simulator parameters:

$$
p_{\text{real}} \approx \alpha \cdot p_{\text{sim}} + (1-\alpha) \cdot p_{\text{real,obs}}
$$

### 4.3 Task-related vs. task-irrelevant randomizationNot all randomization is good for generalization. **Grounding SBIR** (Singh et al., 2023) distinguishes two types of randomization:

- **Task-Relevant Randomization**: Randomization that directly changes strategic decisions, such as obstacle locations (affecting obstacle avoidance decisions). This type of randomization **must be retained** and is a necessary signal for learning generalization strategies
- **Task-Irrelevant randomization**: Randomization that does not change strategic decisions, such as ground texture changes (does not affect flight path). This type of randomization can reduce ** and avoid wasting training capacity

Policy gradient can automatically identify task-related randomization parameters to achieve efficient DR distribution learning.

---

## 5. Digital asset construction: city-level 3D asset generation

### 5.1 Automated scene asset pipeline

Building city-scale simulation scenes requires a large number of 3D assets (buildings, trees, road infrastructure). Manual modeling is extremely expensive (a single detailed architectural model requires 2-5 man-days) and requires **Procedural Generation** (Procedural Generation) technology.

**Sat2Map**: Automatic reconstruction of 3D city models from satellite/aerial imagery:

1. Semantic segmentation: extract building roofs, roads, and vegetation areas
2. Monocular height estimation: predict the height of each building (based on shadow analysis or deep models such as Midas)
3. Grid reconstruction: Stretch the 2D semantic mask along the height direction to generate building exterior walls
4. Texture mapping: Sampling textures from original images or satellite libraries

**Procedural Modeling**: Use L-system or rule grammar to generate building facades and urban street scenes:

$$
\text{Building} ::= \text{Base} + \text{Floor}^N + \text{Roof}, \quad N \sim \text{Uniform}(3, 30)
$$

By adjusting the parameter distribution (number of floors, roof type, facade material), urban building groups with different styles can be generated.

### 5.2 Asset Quality Assessment

The quality of synthetic assets directly affects the effectiveness of Sim2Real migration. **Quality assessment dimensions** include:| Dimensions | Evaluation indicators | Methods |
|------|---------|------|
| **Geometry Accuracy** | RMSE vs LiDAR ground truth | Quantization after point cloud registration |
| **Texture Authenticity** | FID vs Real Image | Fréchet Inception Distance |
| **Semantic Consistency** | Segmentation Accuracy | SegAcc on Synthetic Image |
| **Physical plausibility** | Object size distribution | Comparison with GT statistics |

**SynthCity** (Griffiths & Boehm, 2023) provides a large-scale synthetic data set of 9 types of urban assets, including point clouds, images, and semantic annotations, which can be used as a benchmark for simulated asset quality.

---

## 6. Data quality assessment and multi-modal consistency

### 6.1 Authenticity Measurement

The distribution gap (Domain Gap) between simulation data and real data determines the upper limit of Sim2Real migration effect. Quantitative assessment methods include:

**FID (Fréchet Inception Distance)**: Extract image features through Inception-v3, and calculate the Fréchet distance between the real image feature distribution $\mathcal{N}(\mu_r, \Sigma_r)$ and the simulated image feature distribution $\mathcal{N}(\mu_s, \Sigma_s)$:

$$
\text{FID} = \|\mu_r - \mu_s\|^2 + \text{Tr}\left( \Sigma_r + \Sigma_s - 2\sqrt{\Sigma_r \Sigma_s} \right)
$$

The lower the FID, the closer the simulation image is to the real image. Typical target: FID $< 30$ (difficult to distinguish with the naked eye).

**SSIM/PSNR**: Structural similarity and peak signal-to-noise ratio, pixel-by-pixel evaluation of image quality, suitable for rendering quality comparison of the same scene.

**Perceptual Distance**: Perceptual Loss based on the VGG/ResNet feature layer, which is more in line with the subjective evaluation of the human eye than pixel-level indicators.

### 6.2 Multimodal Consistency ConstraintsMulti-modal simulation data must meet the **cross-modal consistency** constraint - the RGB image, depth map, and LiDAR point cloud of the same scene must be consistent with each other, and there cannot be a self-contradiction such as "the camera sees the wall but the LiDAR does not hit the wall."

**Consistency Verification Pipeline**:

1. **Geometry Consistency Check**: For each 3D point, verify that its projected coordinate depth in the RGB image is consistent with the depth map/LiDAR measurement (error $< 1\%$)
2. **Semantic consistency check**: RGB segmentation results and LiDAR reflection intensity classification results should be consistent (for example, metal railings should be classified as "hard obstacles" in both modalities)
3. **Temporal consistency check**: Optical flow/point cloud motion between adjacent frames should conform to the physical motion model (uniform speed/uniform acceleration assumption)

Data that violates consistency constraints will mislead multi-modal fusion learning and needs to be automatically detected and filtered after data generation.

---

## 7. Planning-simulation closed loop: reinforcement learning training

### 7.1 Reinforcement learning training in simulation

Reinforcement learning (RL) provides a learning paradigm for end-to-end UAV planning without the need for manual design of cost functions. Typical RL training pipeline:

1. **Simulation environment initialization**: Load the city 3D model and generate random take-off and landing points and obstacle configurations
2. **Strategy interaction**: UAV strategy $\pi_\theta(a_t | s_t)$ interacts with the environment in the simulation and collects trajectory data $\{s_t, a_t, r_t, s_{t+1}\}$
3. **Policy Update**: Use PPO (Proximal Policy Optimization) or SAC (Soft Actor-Critic) algorithm to update policy parameters
4. **Domain Randomization**: Randomize scenario configuration in each round of training to improve strategy generalization capabilities
5. **Sim2Real Transfer**: Deploy the trained strategy to a real UAV, which may require a small amount of real data fine-tuning (Transfer RL)

**Key reward function design**:

$$
r_t = r_{\text{progress}} + r_{\text{safety}} + r_{\text{efficiency}} + r_{\text{comfort}}
$$- $r_{\text{progress}} = \Delta d_{\text{goal}}$: Positive reward for progress toward the goal
- $r_{\text{safety}} = -10$ if collision: collision penalty (large negative reward)
- $r_{\text{efficiency}} = -0.01 \cdot T$: time penalty (encourages quick arrival)
- $r_{\text{comfort}} = -0.1 \cdot \|\mathbf{a}_t\|$: acceleration penalty (suppresses sharp turns)

### 7.2 Simulation to real migration strategy

Even with Domain Randomization, simulation-real gaps may still exist. The following strategies can improve migration success rates:

**Conservative Deployment**:
- First conduct safety verification on real UAV at low speed and low altitude
- Gradually expand the flight envelope only after safety is confirmed

**Task-Relevant Feature Alignment**:
- Analyze sensor data feature distribution (depth statistics, edge density) of real UAVs
- Adjust simulation parameters to match the distribution of key features

**Meta-Learning**:
- Use MAML (Model-Agnostic Meta-Learning) to train the strategy to quickly adapt to a small amount of real data
- Train the basic policy $\pi_0$ in simulation and fine-tune it to $\pi^*$ in the real environment

### 7.3 Virtual-real closed-loop case: Aggressive Flight

The autonomous UAV racing projects in **AlphaPilot** (sponsored by Lockheed Martin) and **SUAS Competition** demonstrate a mature simulation-training-deployment closed loop:1. Use DOMAIN_RANDOMIZE in Flightmare/AirSim to configure random lighting, wind disturbance, and obstacle locations
2. Use PPO to train the end-to-end strategy (directly output the motor speed), and the rewards include lap time, collision penalty, and comfort
3. The training strategy reaches $> 15\text{m/s}$ traversal speed in simulation
4. Deploy to real UAV and use Online Adaptation to compensate for residual Sim2Real gaps
5. Key skills: **Safety Shield** - Combining RL policy output with emergency obstacle avoidance based on geometric planning, the policy is only responsible for high-level decision-making

---

## 8. Future directions and frontier exploration

### 8.1 Neural Simulator: Learnable Physics Engine

Traditional simulators rely on manually designed physical models and are difficult to capture complex interactions (fluid-structure interaction, flexible body deformation). **Learned Physics Engine** (Learned Physics Engine) learns physical laws from data through neural networks:

**Graph Network Simulator (GNS)** (Sanchez-Gonzalez et al., ICML 2020) uses graph neural networks to model particle system interactions and can learn the evolution rules of fluid, rigid body, and multi-body systems. If GNS is extended to aerodynamic modeling, it is possible to achieve **data-driven UAV flight dynamics simulation**.

### 8.2 Internet-scale data + generative AI

Large Language Model (LLM) and Diffusion Model bring new possibilities for simulation data generation:

- **LLM generates scene description**: input "Beijing CBD evening peak intersection, 5 cars, 10 pedestrians", GPT-4V can generate detailed scene configuration (location, speed, behavior pattern)
- **Diffusion model generation texture**: Use ControlNet / Stable Diffusion to automatically generate realistic textures based on architectural line drawings, reducing manual modeling
- **NeRF scene cloning**: Take a 5-minute city video with your mobile phone and automatically reconstruct it into a navigable NeRF scene, which can be used directly as a simulation environment

### 8.3 Federated Simulation: Distributed Collaborative MappingIn the future, urban UAV clusters may form a **federated simulation network**: each UAV collects data in flight and updates a shared city digital twin, and other UAVs download the latest twin and train in the updated simulation environment. This not only protects data privacy (the original image does not leave the local area), but also achieves distributed accumulation of knowledge.

---

## 9. Summary

Multimodal simulation data synthesis is the key technical foundation for urban low-altitude UAV planning algorithms to move from research to implementation. Through high-fidelity sensor simulation (RGB, LiDAR, millimeter wave, thermal imaging), programmatic generation of diverse scene assets and strict Domain Randomization strategy, large-scale training data sets can be systematically constructed in the simulation environment.

The core challenge of Sim2Real migration is the **perception gap** and the **dynamic gap**. The perceptual gap can be alleviated through neural rendering (UniSim) and perceptual consistency evaluation; the dynamic gap can be compensated through online adaptation and meta-learning.

As neural simulators, learnable physics engines, and generative AI technologies mature, future simulation data synthesis will be more automated, high-fidelity, and low-cost. The vision of **Simulation as Ground Truth** is gradually becoming possible.

---

## References

- Shah, S., Dey, D., Lovett, C., & Kapoor, A. (2018). AirSim: High-fidelity visual and physical simulation for autonomous vehicles. *Field and Service Robotics*. https://doi.org/10.1007/978-3-319-67361-5_40

- Zhou, Y., et al. (2023). UniSim: A neural closed-loop sensor simulator. *CVPR* (or arxiv:2308.01812, venue to be confirmed). https://doi.org/10.1109/CVPR52729.2023.00571- Kar, A., et al. (2019). Meta-sim: Learning to generate synthetic datasets. *ICCV*. https://doi.org/10.1109/ICCV.2019.00393

- Sanchez-Gonzalez, A., et al. (2020). Learning to simulate complex physics with graph networks. *ICML*. https://doi.org/10.5555/3524938.3525750

- Zhang, J., et al. (2021). SimBot: Enabling autonomous robots with vision-language models via robotic simulators. *CoRL*.

- Du, Y., et al. (2023). Learning policies from simulation with adversarial domain randomization. *ICRA*. https://doi.org/10.1109/ICRA57147.2024.10610923

- Antonini, A., et al. (2020). Winter is coming: Learning to navigate safely in unseen environments. *ICRA*. https://doi.org/10.1109/ICRA40945.2020.9196643

- Song, Y., et al. (2023). Diffusion-LM: Controllable text generation through diffusion models. *NeurIPS*.- Griffith, S., & Boehm, J. (2023). SynthCity: A large-scale synthetic point cloud for urban scenes. *ISPRS Journal of Photogrammetry and Remote Sensing*. https://doi.org/10.1016/j.isprsjprs.2023.04.015

- Lois, C., et al. (2020). Flightmare: A flexible quadrotor simulator with modular perception. *IROS*.

---

*This article is the fifth extended chapter in a series of articles on urban low-altitude drone route planning. Complete series 🎉*