---
title: "Urban low-altitude UAV route planning: three-dimensional spatial modeling"
description: "Systematically review the three-dimensional space modeling methods in urban low-altitude UAV route planning, covering 3D occupancy grid, urban canyon effect and airspace layered model"
tags: ['UAV', '路径规划', '城市空域']
category: Tech
pubDate: 2026-04-08T14:54:00+08:00
sourceHash: "5207e2b900596685eafb34524553f682fb5b6948"
---

## Introduction

Urban low-altitude UAV route planning is one of the core basic technologies to achieve safe and efficient urban air transportation (UAM, Urban Air Mobility). Different from suburban open areas, the urban environment has distinctive characteristics such as complex three-dimensional geometric structure, severe attenuation of GNSS signals, and strong disturbance of the flow field by buildings, which puts forward higher requirements for spatial modeling methods. This article focuses on the first part of the urban low-altitude UAV route planning series - three-dimensional space modeling. It discusses in depth the three-dimensional occupancy grid (3D Occupancy Grid) and octree (Octree) representation, the physical modeling of the urban canyon effect (Urban Canyon), and the airspace hierarchical model that draws on traditional aviation control, supplemented by a comparative analysis of engineering implementation.

## 1. Three-dimensional occupancy grid and octree representation

### 1.1 From two dimensions to three dimensions: mathematical definition

The classic occupancy grid (Occupancy Grid) was proposed by Moravec and Elfes (1985). Its core idea is to discretize continuous space into finite grids and encode the occupancy state of each grid with probability values. In the two-dimensional case, the space is divided into square cells with side length $\Delta$, and the occupancy probability of each cell $m_i$ is recorded as $P(m_i | Z_{1:t})$, where $Z_{1:t}$ is all sensor observations up to time $t$. Sensor updates follow the Bayesian recursive formula:

$$
P(m_i | Z_t, Z_{1:t-1}) = \frac{P(Z_t | m_i, Z_{1:t-1}) P(m_i | Z_{1:t-1})}{P(Z_t | Z_{1:t-1})}
$$

In actual engineering, in order to avoid numerical underflow and simplify calculations, **logarithmic odds (log-odds)** are usually expressed:

$$
l(m_i) = \log \frac{P(m_i)}{1 - P(m_i)}
$$

After each sensor measurement, the additive update rule is:

$$
l(m_i)_{\text{new}} = l(m_i)_{\text{old}} + \Delta l
$$

where $\Delta l$ is determined by the sensor model (positive when occupied, negative when idle). This method converts multiplication into addition, greatly improving real-time performance.The three-dimensional occupancy grid extends the above definition from a plane to a volumetric space $\mathbb{R}^3$, dividing the space into cubic units (voxels) of edge length $\Delta$. Assume $V_i \subset \mathbb{R}^3$ represents the $i$th voxel, then its occupancy probability is $P(v_i | Z_{1:t})$. The direct storage complexity of three-dimensional rasters is $O(N^3)$ ($N$ is the number of single-dimensional rasters), which is unacceptable in typical urban scenarios - for example, the total number of voxels covering an urban area of ​​$1\,\text{km}^3$ at a resolution of $0.1\,\text{m}$ is as high as $10^{13}$ in total.

### 1.2 Octree: Spatial index for adaptive resolution

**Octree** is the standard solution to address the above storage challenges. The OctoMap library proposed by Hornung et al. (2013) is a milestone implementation of this method in the field of robotics. The space division logic of the octree is as follows: the root node covers the entire three-dimensional space, and each internal node is recursively subdivided into 8 sub-nodes of equal volume (corresponding to the $2 \times 2 \times 2$ division of the three-dimensional space) until the preset maximum depth $d_{\max}$ or the minimum voxel size $\Delta_{\min}$ is reached.

Assume the side length of the root node is $L_0$, and the side length of the voxel at depth $d$ is:

$$
L_d = \frac{L_0}{2^d}
$$

The maximum number of nodes for depth $d$ is $8^d$, but since the octree only splits the occupied or observed space, unknown/free areas can be represented by a single node, so the actual number of nodes is much smaller than the full raster. OctoMap's storage model further uses **Probabilistic OcTree**: each node stores an occupancy probability value $P(n)$, which is continuously modified through Bayesian update. The probability of an idle node is $P_{\text{occ}}$, the probability of an occupied node is close to $1$, and the corresponding node in the unknown area does not exist in the tree (implicit encoding).

Hornung et al.'s (2013) experiments show that in a typical indoor environment, OctoMap's memory consumption is about **1/50** of dense three-dimensional rasters of the same resolution, while supporting dynamic updates and arbitrary resolution queries.

### 1.3 Octree and multi-granularity perceptionZeng et al. (2020) proposed a multi-granularity environment perception algorithm based on octree occupancy grids on Multimedia Tools and Applications, pointing out that although the point cloud model is rich in information, there is a lot of redundancy in path planning. They use octrees to provide a unified probabilistic representation of data from different sensors (RGB-D, LiDAR, etc.), retaining high-resolution geometric information at the leaf node level and providing low-resolution global structure perception at the coarse node level. This idea is particularly important for the construction of large-scale city-level maps - centimeter-level obstacle avoidance is required in the near distance, and macro-path decision-making at the 100-meter level is required in the distance.

Thomas et al. (2021) further proposed **Spatiotemporal Occupancy Grid Maps (SOGM)** in the arXiv paper (arXiv: 2108.10585), which embeds the time prediction of dynamic obstacles into the grid representation, providing effective occupancy prediction capabilities for people and vehicles moving in the urban environment, and is of great value for real-time obstacle avoidance planning.

## 2. Urban Canyon Effect: Physical Modeling and Navigation Challenges

Urban Canyon refers to an urban micro-landform with dense buildings and narrow streets. It is one of the most challenging operating environments for low-altitude drones. Its physical effects can be understood in three dimensions.

### 2.1 GNSS signal attenuation and multipath effect

In urban canyons, dense high-rise buildings form a "canyon" structure, and GNSS satellite signals face two types of serious interference:

- **Non-line-of-sight propagation (NLOS)**: The direct signal is blocked by the building, and the drone can only receive the signal reflected or diffracted by the wall, causing the pseudo-range measurement value to be systematically larger;
- **Multipath**: The signal superposition of multiple reflection paths causes carrier phase solution errors and positioning jitter.

The UrbanNav Dataset (Wen et al., 2021; GitHub: IPNL-POLYU/UrbanNavDataset) measured the positioning performance of low-cost sensors in urban canyons in Tokyo and Hong Kong. The results showed that in deep canyon areas, the single point positioning (SPP) error can reach tens of meters. Even if a dual-frequency GNSS receiver is used, without NLOS detection and elimination, the horizontal positioning accuracy will still be difficult to meet the sub-meter requirements for UAV hovering accuracy. The aspect ratio (AR = building height / street width) of urban canyons is the dominant factor affecting GNSS accuracy - the larger the AR, the lower the signal availability.

### 2.2 Turbulence and wind field disturbanceFluid dynamics within urban canyons exhibit a high degree of heterogeneity. The classic study by Rotach (1995) in *Boundary-Layer Meteorology* quantified the statistical profile of turbulence inside canyons, noting that turbulent kinetic energy (TKE) in street canyons is **2-5 times** higher than in open suburbs, and that the standard deviation of the vertical velocity component $\sigma_w$ can reach $0.3$–$0.6$ times the mean wind speed near the surface. Key physical mechanisms include:

- **Building Wake**: The airflow forms periodic shedding vortices (Kármán vortex street) on the leeward side after bypassing the building, generating significant unsteady lift and lateral forces;
- **Canyon Circulation (Street Canyon Circulation)**: When the incoming flow is orthogonal to the canyon axis, a double vortex ring structure with opposite directions is formed inside the street, and the net vertical wind speed component is significantly amplified in this area;
- **Inertial Subrange**: The energy spectrum of turbulence energy in the inertial subrange follows the $-5/3$ rule (Kolmogorov's law). Small-scale turbulence constitutes a continuous disturbance to the UAV attitude control bandwidth.

For UAV control design, the characteristic frequency range of turbulence intensity is crucial. The disturbance in the $1$–$10\,\text{Hz}$ frequency band is the most significant in urban canyons, which requires the attitude loop bandwidth of the flight control system to be no less than $20\,\text{Hz}$, which is not easy to implement on an embedded platform.

### 2.3 Bernoulli wind acceleration effect

In narrow streets, the Bernoulli effect cannot be ignored. When airflow is forced through a channel with reduced cross-sectional area, the wind speed increases significantly in local areas according to the continuity equation $A_1 ​​v_1 = A_2 v_2$. Wind speeds at the narrowest points between buildings in urban canyons can be **1.5–3 times** as high as in open areas. In addition, the "Venturi Effect" between building facades will locally produce suction toward the center of the street, posing a threat to the lateral stability of the drone.

In practical planning, it is recommended to model the **equivalent wind disturbance** in urban canyons as mean wind $\bar{u}$ superimposed with random turbulence components $\tilde{u}$:

$$
u_{\text{eff}}(t) = \bar{u} + \sigma_u \cdot \xi(t)
$$

where $\xi(t)$ is Gaussian white noise obeying the standard normal distribution, and $\sigma_u$ is determined from an empirical formula based on the canyon aspect ratio and local street geometry.## 3. Airspace layered model

### 3.1 Enlightenment from traditional aviation control

The traditional civil air traffic control system has adopted altitude layer (Altitude Layer) management for decades: with $1000\,\text{ft}$ (approximately $300\,\text{m}$) as the basic altitude interval, the airspace below $29000\,\text{ft}$ is divided into multiple control sectors, with each layer serving aircraft of different types and speeds. In the context of UAM, urban low-altitude UAVs need to coexist with ground pedestrians, buildings, helicopter landing pads, and traditional general aircraft within a vertical range of **$0$–$120\,\text{m}$** (approximately $0$–$400\,\text{ft}$). Therefore, layered design becomes inevitable.

NASA's UTM (UAS Traffic Management) project research (2016-2024) and FAA's UAM ConOps V2.0 (2023) both pointed out that hierarchical management is the core means to avoid large-scale drone conflicts. Drawing on this idea in urban scenarios, the following three-layer scheme can be designed.

### 3.2 Urban scene height layer division scheme

| Altitude level | Vertical range | Main functions | Aircraft type | Typical speed |
|--------|----------|----------|-----------|----------|
| **G Floor** | Ground $\sim 30\,\text{m}$ | Express delivery on sidewalks, robot delivery | Micro multi-rotor | $0$–$5\,\text{m/s}$ |
| **L level** | $30$–$80\,\text{m}$ | Community logistics, urban aerial photography, low-rise shuttle | Small multi-rotor/composite wing | $5$–$15\,\text{m/s}$ |
| **U Tier** | $80$–$120\,\text{m}$ | Intercity express, emergency response, high-rise shuttle | Medium eVTOL/fixed wing | $15$–$30\,\text{m/s}$ |

> Note: The specific altitude boundaries need to be adjusted according to local airspace management regulations (my country is based on the "Interim Regulations on Unmanned Aircraft Flight Management" 2023) and urban planning.

The design principles of this layered scheme are as follows:1. **Functional isolation**: The G layer focuses on the safety of terminal distribution (to avoid direct conflicts with people), the L layer is the urban mainstream application layer, and the U layer is close to the height of traditional general aviation to be compatible with transition;
2. **Flow separation**: The upstream and downstream directions are further separated horizontally at the same altitude, and the one-way route loop is designed with reference to the five-sided approach logic of air traffic control;
3. **Dynamic adjustment**: The layered boundary can be dynamically translated according to real-time traffic density, and the FAA's xTM (extensible Traffic Management) framework has provided a standardized interface for this.

### 3.3 Fusion of layered and three-dimensional raster maps

The height layered model needs to be deeply integrated with the three-dimensional occupancy grid: in the planning stage, **Layer Masking** is performed on the octree map based on the layered boundaries, and paths are only searched for in the flyable voxels of the layer where the current task is located and adjacent layers; during dynamic re-planning, if there is congestion on a certain layer, it can be automatically switched to the adjacent layer for detours. This mechanism has been initially verified in NASA's UTM corridor (Corridor) concept.

## 4. Engineering trade-offs for Octree/PCL point clouds/voxels

In engineering practice, the choice of three-dimensional representation method requires a trade-off between accuracy, memory, calculation speed, and update frequency. The following is a systematic comparison.| Metrics | Dense 3D Raster | Octree | Raw Point Cloud (PCL) | Hash Voxel |
|------|------------|----------------|--------------|----------------------|
| **Memory efficiency** | Low (fixed $O(N^3)$) | High (adaptive splitting) | Medium (only points saved, no topology) | High (sparse hash index) |
| **Query Complexity** | $O(1)$ | $O(\log N)$ | $O(N)$ (exhaustive) or $O(\log N)$ (with kd-tree) | $O(1)$ mean |
| **Dynamic update** | Slow (full reconstruction) | Fast (incremental node splitting) | Fast (append points) | Fast (hash insertion) |
| **Resolution Consistency** | Global Consistency | Hierarchical Adaptation | No Grid Structure | Global Consistency |
| **Collision Detection** | Fast (array index) | Medium (tree search) | Slow (point-model detection) | Fast (hash lookup) |
| **Engineering Ecology** | ROS nav_msgs | OctoMap / PCL Octree | PCL / Open3D | OctoMap (configurable) |
| **Applicable scenarios** | Small range and high precision | Large range and multiple resolutions | Real-time sensing/mapping | Sparse large-scale scenes |

**Octree's core advantage** lies in its dual characteristics of **adaptive resolution + probabilistic representation**: it is both a spatial index structure and a probabilistic update framework, which is particularly suitable for the perception needs of "accurate obstacles near and rough obstacles far away" in urban scenes. The OctoMap library (Hornung et al., 2013; DOI: 10.1007/s10514-012-9321-0) is a testament to its engineering maturity, both in terms of activity on GitHub and the number of academic citations (more than 5,000 times according to Google Scholar).

**The advantage of point cloud** is that it preserves the original sensor data without loss and is suitable for perception algorithms based on deep learning (3D target detection, semantic segmentation, etc.) as input. The PCL (Point Cloud Library) library and the Open3D library provide a mature point cloud processing tool chain, but the point cloud itself does not encode occupied/idle semantic information and requires additional steps to convert into a flyable area.**Hash voxels** (such as OctoMap's `OcTree Key` hash index scheme) perform well in scenarios that require extremely fast query speed and sparse scenes. The memory overhead is close to that of an octree but the query is more efficient. It has been a hot topic in cutting-edge research in recent years.

In actual urban scenarios, the **recommended solution** uses OctoMap's probabilistic octree as the underlying storage, uses the original point cloud as the sensing input, continuously corrects the occupancy probability through the incremental update mechanism, and uses a hash index to accelerate nearest neighbor queries. This combination has been proven in advanced SLAM systems such as LIO-SAM to achieve robust real-time mapping in urban canyons (see LIO-SAM-6AXIS-UrbanNav adapted version).

## 5. Summary and Outlook

This article systematically sorts out the core elements of three-dimensional space modeling in urban low-altitude UAV route planning:

- **Three-dimensional occupancy grid and octree** provide a unified environment representation framework based on probability theory. As an open source implementation, OctoMap has been widely verified in academia and industry;
- **Urban Canyon Effect** imposes constraints on the UAV planning system from the three physical dimensions of GNSS attenuation, turbulence statistics and Bernoulli wind acceleration, and needs to be explicitly modeled in route planning;
- **Airspace layered model** draws on traditional aviation control ideas and divides $0$–$120\,\text{m}$ vertical airspace into three layers: G/L/U in urban scenarios, providing a structural framework for large-scale drone traffic management;
- Project selection should make a comprehensive trade-off between memory efficiency, query speed and dynamic update capabilities. The combination of OctoMap + point cloud is the current mainstream technology route.

Subsequent chapters will gradually delve into topics such as **Path Planning Algorithm** (the application of sampling algorithms such as RRT*/BIT* in three-dimensional octree maps), **Real-time Trajectory Optimization** (Model Predictive Control under Wind Disturbance in Urban Canyons), and **Multi-aircraft Collaborative Obstacle Avoidance** to build a complete urban low-altitude route planning technology system.

---

## References- Hornung, A., Wurm, K. M., Bennewitz, M., Stachniss, C., & Burgard, W. (2013). OctoMap: An efficient probabilistic 3D mapping framework based on octrees. *Autonomous Robots*, 34(3), 189–206. https://doi.org/10.1007/s10514-012-9321-0

- Thomas, H., Farr, R., Yang, C., Chen, Y., & Leonard, J. J. (2021). Learning Spatiotemporal Occupancy Grid Maps for Lifelong Navigation in Dynamic Scenes (arXiv: 2108.10585). arXiv. https://arxiv.org/abs/2108.10585

- Wen, W., Zhang, G., & Hsu, L. T. (2021). *UrbanNav: An Open-sourcing Localization Dataset for Benchmarking Positioning Algorithms Designed for Urban Canyons* [Dataset and documentation]. GitHub repository: https://github.com/IPNL-POLYU/UrbanNavDataset

- Rotach, M. W. (1995). Profiles of turbulence statistics in and above an urban street canyon. *Atmospheric Environment*, 29(13), 1473–1486. https://doi.org/10.1016/1352-2310(95)00084-D- Zeng, T., Si, B., & Zhao, J. (2020). Multi-granularity environment perception based on octree occupancy grid. *Multimedia Tools and Applications*, 79, 27875–27896. https://doi.org/10.1007/s11042-020-09302-w

- Moravec, H. P., & Elfes, A. (1985). High resolution maps from wide angle sonar. *Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)*, 116–121. https://doi.org/10.1109/ROBOT.1985.1087316

- U.S. Department of Transportation / Federal Aviation Administration. (2023). *Urban Air Mobility (UAM) Concept of Operations*, Version 2.0. FAA. https://www.faa.gov/air_traffic/nas_management/nas_research/models/uam_conops

- NASA Aeronautics Research Mission Directorate. (2023). *UAS Traffic Management (UTM) Project Summary*. NASA. https://utm.arc.nasa.gov/- Hrabar, S., & Sukhatme, G. S. (2004). A comparison of two camera configurations for optic-flow based navigation through urban canyons. *Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 3943–3948. https://doi.org/10.1109/IROS.2004.1389989