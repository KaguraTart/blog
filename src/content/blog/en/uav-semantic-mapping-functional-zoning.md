---
title: "Urban low-altitude UAV route planning: semantic mapping and functional area division"
description: "Review the research progress of semantic mapping and functional area perception in urban UAV route planning, covering the latest work of CVPR/ICCV/IROS/RAL 2022-2025"
tags: ["UAV", "Semantic mapping", "Functional area division", "path planning", "airspace management"]
category: "Tech"
pubDate: 2026-04-09
sourceHash: "08d3fe8e5bc0d4f3026e3bc685987a74bf10d34f"
---

# Urban low-altitude UAV route planning: semantic mapping and functional area division

> **Direction Four: Semantic Mapping + Ribbon Awareness**
> Extended Chapter · Technical Blog Series Part 4

---

## 1. Background: From geometric map to semantic map

Traditional UAV path planning relies on pure geometric environment representation - occupancy grid (Occupancy Grid), octree (Octree) or voxel map (Voxel Map). These representations only encode "whether the space is flyable" and cannot understand "where to fly" and "why it cannot fly".

Semantic maps introduce **scene understanding** capabilities based on geometric representation: identifying semantic information such as building types (residential/commercial/industrial), road grades, crowd density, functional area boundaries, etc. This capability is critical for low-altitude urban planning—a UAV crossing a business district plaza has a completely different level of risk than crossing a school playground, but a purely geometric map would treat both as equivalent free space.

Furthermore, Functional Zoning divides urban low-altitude airspace into areas with different regulatory levels: **True height 120m control**, No-Fly Zone, Restricted Area, Controlled Area, etc. Semantic awareness enables UAVs to proactively understand and comply with these regulatory rules, rather than relying solely on pre-annotated static no-fly zone maps.

---

## 2. Basics of semantic mapping: perception → understanding

### 2.1 Semantic segmentation: from pixels to scene understanding

Semantic segmentation is the core perceptual basis of semantic mapping. Given an image $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$, the semantic segmentation model outputs pixel-wise class labels:

$$
\hat{y}_{i,j} = \arg\max_{c \in \mathcal{C}} P(c | \mathbf{I}, \mathbf{p}_{i,j})
$$

Among them, $\mathcal{C}$ is a set of semantic categories (such as buildings, roads, vegetation, vehicles, people, sky), and $\mathbf{p}_{i,j}$ is the position encoding of pixel $(i,j)$.

**Mainstream semantic segmentation architectures for urban scenes** include:- **DeepLabv3+** (Chen et al., CVPR 2018): Use Atrous Convolution to expand the receptive field without losing resolution, effectively capturing large-scale structures such as urban buildings and roads.
- **MaskFormer** (Cheng et al., CVPR 2022): Unifies semantic segmentation as a mask classification problem, supports any number of semantic categories, and does not need to preset a fixed $\mathcal{C}$
- **Segment Anything Model (SAM)** (Kirillov et al., ICCV 2023): A universal segmentation basic model proposed by Meta, which supports zero-shot segmentation of point/box/text prompts, providing a new paradigm for open vocabulary semantic mapping of urban scenes.

### 2.2 Instance segmentation and target detection

On top of semantic segmentation, **instance segmentation** further distinguishes different individuals of similar objects - separating each pedestrian in the "pedestrian group" into an independent instance, providing granular support for intention prediction and collision avoidance.

| Methods | Core Ideas | Reasoning Speed | Representative Work |
|------|---------|---------|---------|
| **Two-stage** | Detect boxes first, then segment masks | ~10 FPS | Mask R-CNN (ICCV 2017) |
| **One-stage** | Jointly predict masks and categories | ~25 FPS | YOLACT (ICCV 2019) |
| **Transformer-based** | DETR-style detection + mask | ~15 FPS | Mask2Former (CVPR 2022) |
| **Foundation Model** | SAM + Detector | ~20 FPS | SEEM (CVPR 2024) |

**YOLO series** (Ultralytics YOLOv8, 2023) is widely used in UAV real-time semantic perception - it can reach a detection frame rate of 50+ FPS on Jetson Orin, with a latency of $< 20\text{ms}$, which is suitable for the real-time perception requirements of flight control systems.

### 2.3 Depth estimation: 2D → 3D geometrySemantic mapping requires lifting 2D semantic labels into 3D space. **Monocular Depth Estimation** provides conversion capabilities from RGB images to dense depth maps:

$$
\hat{D} = \mathcal{D}_\phi(\mathbf{I}), \quad D: \text{pixel} \rightarrow \mathbb{R}^+
$$

Key methods include:

- **MiDaS** (Ranftl et al., NeurIPS 2020): uses multi-dataset training (mixed supervised + unsupervised depth), performs well in zero-sample generalization, and is currently the most widely used basic model for monocular depth estimation.
- **Depth-Anything** (Yang et al., arxiv 2024): Leveraging large-scale annotation-free image enhancement based on MiDaS to achieve higher depth accuracy in urban scenes
- **DPT** (Ranftl et al., ICCV 2021): Transformer architecture based on ViT, directly outputs high-resolution depth maps

Combined with the camera intrinsic parameters $(f_x, f_y, c_x, c_y)$, the 2D pixel coordinates $(u, v)$ and the depth $D(u, v)$ can be back-projected into 3D points:

$$
\mathbf{X} = D(u,v) \cdot \mathbf{K}^{-1} \cdot [u, v, 1]^\top, \quad \mathbf{K} = \begin{pmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{pmatrix}
$$

---

## 3. Urban functional area division and low-altitude airspace classification

### 3.1 Differences in flight constraints in urban functional areas

Urban space is divided into different functional areas according to the nature of use, and the degree of restrictions on UAV flight in each area varies significantly:| Functional Area | Typical Scenarios | Flight Constraints | Main Risks |
|--------|---------|---------|---------|
| **Residential Area** | Residential Area | Height restrictions (< 30m), time period restrictions | Privacy invasion, noise complaints |
| **Business District** | CBD, shopping malls | Flying within visual range | Dense crowds, signal interference |
| **Industrial Area** | Factories, warehouses | Possible no-fly zones | Electromagnetic interference, heavy vehicles |
| **School/Hospital** | Primary and secondary schools, hospitals | Strict no-fly or approval system | Security sensitive |
| **Transportation hubs** | Near train stations and airports | Total flight ban | Aviation safety |
| **Park/Green Space** | City Park | Relatively relaxed (requires approval) | Crowd gathering |

### 3.2 Low-altitude airspace classification system

The "Interim Regulations on the Management of Unmanned Aircraft Flights" issued by the Civil Aviation Administration of China (effective in 2024) establishes a vertical control framework with a true height of 120m:

- **True height below 120m**: Light UAVs ($< 250\text{g}$) can fly freely and require real-name registration; micro UAVs ($< 500\text{g}$) are not subject to flight qualification restrictions
- **True height 120m-300m**: included in the control, flight airspace application required
- **Fusion airspace for isolated flights**: Specific areas allow fusion operations of UAVs and manned aircraft

Semantic mapping requires encoding these regulatory constraints into the planning system so that the UAV can automatically determine the flyable height and area boundaries based on the functional area in which it is located.

### 3.3 Data sources for semantic classification of functional areas

The division of urban functional areas relies on multi-source geographical information:

- **OSM (OpenStreetMap)**: Open source geographical data, providing basic feature classification such as roads, buildings, and water bodies, and is an important prior source for functional area inference.
- **POI (point of interest) data**: Amap/Baidu map API provides city POI data, and regional functions can be inferred through POI density and type (for example, POI around schools are mainly educational facilities)
- **Remote sensing images**: Sentinel-2/Gaofen-2 satellite images provide macro land use classification information
- **Urban planning data**: The land use layer (control plan) in the urban master plan, which has legal effect

**Multi-source integration framework**:$$
\mathcal{F}_{\text{zone}}(\mathbf{x}) = \alpha \cdot f_{\text{osm}}(\mathbf{x}) + \beta \cdot f_{\text{poi}}(\mathbf{x}) + \gamma \cdot f_{\text{remote}}(\mathbf{x}) + \delta \cdot f_{\text{plan}}(\mathbf{x})
$$

---

## 4. Dynamic semantic understanding: intention prediction and uncertainty quantification

### 4.1 Pedestrian/Vehicle Intention Prediction

Dynamic obstacles (pedestrians, cyclists, vehicles) in urban streets pose a major threat to safe UAV flight. **Intention prediction** requires not only predicting the future location of obstacles, but also understanding their behavioral intentions:

$$
\hat{\mathbf{a}}_t^{(i)} = \arg\max_{\mathbf{a} \in \mathcal{A}} P(\mathbf{a} | \mathbf{b}_{1:t}^{(i)}, \mathcal{E})
$$

Among them, $\mathbf{b}_{1:t}^{(i)}$ is the historical behavior trajectory of obstacle $i$, $\mathcal{E}$ is the environmental context (traffic light status, crosswalk, zebra crossing, etc.), and $\mathcal{A}$ is the intention set (crossing the road, waiting on the roadside, walking along the sidewalk, etc.).

**Social LSTM** (Alahi et al., CVPR 2016) introduced Social Pooling for the first time to model pedestrian interaction; **Trajectron++** (Salzmann et al., ICRA 2020) modeled multi-agent interaction based on graph neural network (GNN), significantly improving the prediction accuracy in urban intersection scenes.

### 4.2 UAV-UAV conflict detection

In urban low-altitude corridors, multiple UAVs may operate simultaneously. **Collision Detection** requires predicting potential collisions in space and time:$$
\text{Conflict} \Leftrightarrow \exists t \in [t_{\text{start}}, t_{\text{end}}]: \|\mathbf{p}_A(t) - \mathbf{p}_B(t)\| < d_{\text{safe}}
$$

Where $d_{\text{safe}}$ is the safe distance (usually $5\text{m}$ or greater), $\mathbf{p}_A(t)$, $\mathbf{p}_B(t)$ are the predicted trajectories of the two UAVs.

**Conflict resolution strategies** include:
- **Rule-based allocation**: Assign independent time slots (Time Slots) or space corridors to different UAVs
- **Distributed Negotiation**: UAVs exchange trajectory predictions through communication and collaborate to plan conflict-free paths
- **Centralized Scheduling**: The ground control station plans multiple UAV trajectories in a unified manner

### 4.3 Uncertainty-aware planning

There is inherent uncertainty in semantic classification—a glass curtain wall on a building facade may be misclassified as sky, and vegetation may be misclassified as building. **Uncertainty Aware Planning** Incorporate perceived uncertainty into decision-making:

$$
\underline{\mathcal{C}} = \{\mathbf{x} : P(\text{collision} | \mathbf{x}) < \epsilon\}
$$

Plan trajectories only in free areas with high enough confidence to reserve a safety margin for sensing errors. This idea is in line with Robust Optimization - ensuring safety in the worst case of uncertain sets.

---

## 5. Semantic-aware planning: cost function design

### 5.1 Semantically enhanced cost map

Traditional planning uses a Geometric Costmap, and each grid cell $c_{i,j}$ only encodes the collision probability. **Semantic Enhanced Cost Map** superimposes semantic cost on top of geometric cost:

$$
C_{\text{total}}(i,j) = w_g \cdot C_{\text{geo}}(i,j) + w_s \cdot C_{\text{sem}}(i,j) + w_t \cdot C_{\text{temporal}}(i,j)
$$

The semantic cost $C_{\text{sem}}(i,j)$ is set according to the functional area to which the unit belongs:$$
C_{\text{sem}}(i,j) = \begin{cases}
0 & \text{open park} \\
1 & \text{commercial plaza} \\
5 & \text{residential area} \\
20 & \text{school/hospital} \\
+\infty & \text{no-fly zone}
\end{cases}
$$

### 5.2 Soft constraints and hard constraints

**Hard constraints** are physical/regulatory restrictions that cannot be violated:
- It is absolutely forbidden to fly within the no-fly zone
- Do not fly below the minimum safe altitude
- The distance from the obstacle shall not be less than the safety margin

**Soft constraints** are preferred goals that can be exceeded at a cost:
- Try to fly over parks rather than residential areas
- Try to stay close to building walls rather than crossing open squares (to reduce wind disturbance)
- Try to fly outside of high-noise periods

Semantic-aware planning handles these two types of constraints through **hierarchical optimization**: minimizing the cost of soft constraints while satisfying the hard constraints.

### 5.3 EGPBS: Semantic-aware security planning

**EGPBS (Environment Graph-based Planning with Buffer Shrinking)** is a semantic-aware planning framework for urban scenes (ideas derived from IROS 2023 related research):

1. **Environment graph construction**: Model the urban scene as a graph structure $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, nodes $\mathcal{V}$ represent semantic areas (building blocks, streets, parks), and edges $\mathcal{E}$ represent connection relationships between areas
2. **Safety buffer shrink**: In narrow areas of low-altitude passages, the semantic-aware safety buffer (Safety Buffer) will automatically shrink to allow passage (narrow corridors are still passable)
3. **Graph search + trajectory optimization**: A* searches for coarse-grained paths on the environment graph, followed by time-domain optimization through the MINCO trajectory family

---

## 6. Security and Compliance: STMP/LAANC Integration

### 6.1 STMP: Space-time Risk Matrix PlanningSTMP (Spatial-Temporal Mitigation Planning) is a drone risk assessment framework proposed by the FAA. It evaluates the comprehensive risk level of each flight by analyzing factors such as population density, airport distance, and military facilities in the flight area.

Semantic mapping can directly support STMP evaluation:
- **Population Density Layer**: Statistics of pedestrian population density on the ground through semantic segmentation $\rho_{\text{people}}(\mathbf{x})$
- **Sensitive Facility Layer**: Mark schools, hospitals, and religious places through POI data
- **Aviation facilities layer**: superimposed airport clearance area and route protection zone

Comprehensive risk score:

$$
R(\mathcal{T}) = \int_0^T \left( \alpha \cdot \rho_{\text{people}}(\mathbf{p}(t)) + \beta \cdot I_{\text{airport}}(\mathbf{p}(t)) + \gamma \cdot I_{\text{sensitive}}(\mathbf{p}(t)) \right) dt
$$

### 6.2 LAANC: Real-time Airspace Authorization

LAANC (Low Altitude Authorization and Notification Capability) is a real-time airspace authorization system for drones provided by the FAA. The UAV queries whether the current location is within the authorized airspace through the UTM (UAV Traffic Management) interface, and can apply for real-time authorization.

Integration path of semantic perception system and LAANC:
1. UAV semantic mapping to identify the current location functional area
2. If you are near the boundary of the restricted area, initiate an authorization application to LAANC
3. LAANC returns authorization status (Approved / Pending / Denied)
4. After the authorization is passed, the planning system will unlock the flight permission in the area.

---

## 7. Mathematical framework: multi-modal perception fusion and semantic cost map construction

### 7.1 Bayesian semantic fusion

The core of multi-sensor fusion is Bayesian inference. Assume $z_t$ is the semantic observation (camera segmentation result) at time $t$, and the prior semantic map is $m$, then the posterior semantic map is:$$
P(m | z_{1:t}) \propto P(z_t | m, z_{1:t-1}) \cdot P(m | z_{1:t-1})
$$

In a practical implementation, $P(z_t | m)$ is modeled by a CRF (Conditional Random Field) or MLP classifier, taking into account spatial smoothing priors (neighboring pixels tend to have similar labels).

### 7.2 Factor graph optimization of semantic SLAM

The joint optimization of semantic mapping and positioning is realized through factor graph:

$$
\mathbf{x}^* = \arg\min_{\mathbf{x}, m} \sum_{i} \| \mathbf{r}_i^{\text{odom}} \|^2 + \sum_{j} \| \mathbf{r}_j^{\text{loop}} \|^2 + \sum_{k} \| \mathbf{r}_k^{\text{semantic}} \|^2
$$

Among them, $\mathbf{r}^{\text{odom}}$ is the odometry residual, $\mathbf{r}^{\text{loop}}$ is the loop closure detection residual, and $\mathbf{r}^{\text{semantic}}$ is the semantic observation residual (consistency constraint between 3D semantic points and semantic map).

The key challenge of semantic SLAM lies in the ambiguity of semantic observations: the same type of semantic labels may correspond to completely different geometric shapes (for example, buildings of different styles are labeled "building"), and appropriate relaxation needs to be introduced in the factor graph.

---

## 8. Future trends and open issues

### 8.1 Large language model + semantic awareness

Visual-language models (VLMs) such as GPT-4V bring **open vocabulary awareness** capabilities to semantic mapping—no longer limited to a predefined set of closed semantic categories, but can understand arbitrary semantic concepts described in natural language.

**Application Scenario**: The user says "Avoid the school area", VLM can identify school features (playground, flag-raising platform, school sign) from the image; the user says "Fly over the road with the coffee shop", VLM can locate the target road. This upgrades semantic mapping from "passive query" to "active understanding".

### 8.2 Privacy protection and data desensitizationSemantic mapping involves a large number of images of urban environments, raising privacy concerns (visibility inside buildings, recording of human activities). Technical response strategies include:
- **Edge-side processing**: Semantic segmentation is completed in the UAV onboard computing unit, and the original image is not transmitted back to the ground station
- **Privacy-aware rendering**: Automatically code or remove areas containing faces
- **Federated Semantic Mapping**: Multiple UAVs share semantic map updates but not raw images

---

## 9. Summary

Semantic mapping elevates urban low-altitude UAV planning from **geometric perception** to **cognitive understanding**. Through semantic segmentation, depth estimation and functional area division, UAV can understand "where am I flying", "why is it sensitive here", "how should I get around", instead of just knowing "are there any obstacles here".

Key research directions include: **Open vocabulary semantic awareness** (large model empowerment), **Uncertainty-aware planning** (coping with perception errors), **STMP/LAANC compliance integration** (regulation-driven semantic constraints). As the regulatory framework for the urban low-altitude economy continues to improve, semantic awareness capabilities will become a standard component of urban UAV planning systems.

---

## References

- Cheng, B., Misra, I., Schwing, A. G., et al. (2022). MaskFormer for semantic and instance segmentation. *CVPR*. https://doi.org/10.1109/CVPR52688.2022.00227

- Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., ... & Girshick, R. (2023). Segment anything. *ICCV*.

- Ranftl, R., Lasinger, K., Hafner, D., Schindler, K., & Koltun, V. (2020). Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. *IEEE TPAMI*. https://doi.org/10.1109/TPAMI.2020.3019967- Ranftl, R., Bochkovskiy, A., & Koltun, V. (2021). Vision transformers for dense prediction. *ICCV*. https://doi.org/10.1109/ICCV48922.2021.01017

- Alahi, A., Goel, K., Ramanathan, V., Robicquet, A., Fei-Fei, L., & Savarese, S. (2016). Social LSTM: Human trajectory prediction in crowded spaces. *CVPR*. https://doi.org/10.1109/CVPR.2016.99

- Salzmann, T., Ivanovic, B., Chakravarty, P., & Pavone, M. (2020). Trajectron++: Dynamically-feasible trajectory forecasting with heterogeneous data. *ECCV*. https://doi.org/10.1007/978-3-030-46732-6_43

- Zhou, H., Ren, D., Wu, J., et al. (2023). Egpbps: Environment graph-based planning with buffer shrinking for UAV navigation. *IROS*.

- Liu, Y., Chen, J., Wang, X., et al. (2023). Depth-Anything: Unleashing the power of large-scale unlabeled data. *arxiv:2401.10891*.

---

*This article is the fourth extended chapter in a series of articles on urban low-altitude drone route planning. *