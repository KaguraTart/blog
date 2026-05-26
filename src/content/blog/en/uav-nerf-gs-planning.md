---
title: "Urban low-altitude UAV route planning: NeRF and 3DGS neural rendering methods"
description: "Overview of the application of NeRF/3DGS in urban UAV active sensing and route planning, covering the latest work of CVPR/ICCV/NeurIPS/IROS/ICRA 2022-2025"
tags: ["UAV", "NRF", "3DGS", "active perception", "path planning"]
category: "Tech"
pubDate: 2026-04-08
sourceHash: "5557d17ae8bb31a91500f574bc8cc486e4e032d1"
---

# Urban low-altitude UAV route planning: NeRF and 3DGS neural rendering methods

> **Direction 1: NeRF/3DGS + UAV active sensing planning**
> Extended Chapter · Technical Blog Series Part 1

---

## 1. Background: bottleneck of traditional environment representation

One of the core challenges for low-altitude unmanned aerial vehicle (UAV) online route planning in urban scenes is **how to construct and update the environment representation in real time under limited computing power**. Traditional methods rely on voxel grid (Voxel Grid) or octree (Octree) as spatial representation, and their limitations have become increasingly prominent in recent years:

| Dimensions | Voxel/Octree | NeRF/3DGS |
|------|------------|-----------|
| **Memory complexity** | $O(N^3)$ number of voxels, $N$ determines the upper limit of resolution | Continuously differentiable MLP, no fixed resolution constraints |
| **Update speed** | Incremental update requires rewriting local voxels, which wastes storage in empty areas | Point/Gaussian incremental insertion, $\Delta t = O(1)$ Local update |
| **Occlusion Reasoning** | Only geometric occupancy, no texture/semantic information, weak prediction ability | Implicit continuous density field naturally supports ray casting and occlusion prediction |
| **Rendering Quality** | Requires additional texture mapping for visualization | End-to-end differentiable rendering, Photo-realistic |

Specifically, UAVs need to handle multi-story building facades, cantilevered structures, dynamic vehicles, and pedestrians while flying through urban canyons. The voxel method faces a resolution-memory trade-off after discretizing continuous space: increasing the resolution to capture small obstacles (such as wires, branches) will lead to memory explosion; reducing the resolution will introduce the risk of collision. The continuous radiation field representation introduced by Mip-NeRF (Barron et al., 2021) provides a new solution to this dilemma, and the rise of 3D Gaussian Splatting (Kerbl et al., 2023) further makes real-time rendering possible.

---

## 2. NeRF Basics: From MLP to Volume Rendering

### 2.1 Implicit 3D scene representationThe core idea of NeRF (Neural Radiance Fields, Mildenhall et al., 2020) is to use an MLP network
$\mathcal{F}_\theta: (\mathbf{x}, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma)$ maps 3D position $\mathbf{x} \in \mathbb{R}^3$ and view direction $\mathbf{d} \in \mathbb{R}^2$ to color $\mathbf{c} \in \mathbb{R}^3$ and bulk density $\sigma \in \mathbb{R}^+$. The original NeRF adopts a standard 8-layer fully connected network (256 channels per layer) and uses Positional Encoding to map $\mathbf{x}$ and $\mathbf{d}$ to high-frequency space to capture detailed textures in the scene. This MLP is optimized through a large number of images with known camera poses to learn an implicit geometric and appearance representation of the scene.

For UAV online planning scenarios, the core question is: **How ​​to incrementally update this MLP during flight**? The original NeRF requires several hours of offline training and cannot meet real-time needs. This has driven the emergence of fast mapping methods such as Instant-NGP (Müller et al., 2022), which uses Multi-Resolution Hash Encoding to compress mapping time from hours to seconds. In addition, NICE-SLAM (Zhu et al., 2022) achieves real-time reconstruction through hierarchical feature grids, and its multi-resolution architecture is particularly suitable for the incremental update scenario of UAVs.

### 2.2 Volume rendering equation

Given a ray $r(t) = o + t\mathbf{d}$ emanating from the camera optical center $o$ along direction $\mathbf{d}$, NeRF's volume rendering equation performs alpha synthesis on sampling $K$ points along the ray:$$
\hat{C}(\mathbf{r}) = \sum_{i=1}^{K} T_i \cdot \alpha_i \cdot \mathbf{c}_i, \quad T_i = \prod_{j=1}^{i-1}(1 - \alpha_j), \quad \alpha_i = 1 - \exp(-\sigma_i \delta_i)
$$

Where $\delta_i = t_{i+1} - t_i$ is the distance between adjacent sampling points, $T_i$ is the transmittance (transmittance), which represents the probability that there is no obstruction from the optical center to the $i$th sampling point. The rendered color $\hat{C}$ is differentiable with respect to $\theta$, allowing end-to-end optimization of scene representation via photometric loss $\mathcal{L} = \| \hat{C} - C_{\text{GT}} \|^2_2$. In actual implementation, perceptual loss or SSIM is usually added to improve rendering quality.

**Optimization objective function** can be written as:

$$
\theta^* = \arg\min_\theta \sum_{\text{rays}} \| \hat{C}(\mathbf{r}; \theta) - C_{\text{GT}}(\mathbf{r}) \|^2_2
$$

### 2.3 Essential differences from Occupancy Grid

Occupancy Grid models each voxel as a discrete binary variable $p \in \{0, 1\}$ (occupied/idle), while NeRF models the density $\sigma$ as a continuous volumetric density (Volumetric Density). This design has two key advantages:

1. **Anti-noise**: Real LIDAR point clouds have measurement noise, discrete occupancy rasters are difficult to handle, and volumetric density can naturally model uncertainty.
2. **differentiable geometry**: the gradient of the density field $\nabla_\mathbf{x}\sigma$ directly gives the direction of the surface normal vector without additional SDF calculationsHowever, the **black box characteristics** of MLP make it difficult to directly query "whether a certain space is occupied" during planning - the voxel density must be estimated through ray integration, which is less efficient. This is an important motivation for the rise of 3DGS: it replaces implicit MLP with explicit Gaussian primitives, achieving a spatial query complexity of $O(N)$ while maintaining differentiable rendering capabilities.

---

## 3. 3D Gaussian Splatting: a new paradigm for real-time rendering

### 3.1 From MLP to differentiable Gaussian ellipsoid

3D Gaussian Splatting (3DGS, Kerbl et al., 2023) replaces NeRF's MLP network with a set of differentiable Gaussian ellipsoids, achieving >30 FPS differentiable rendering on a single consumer-grade GPU, and won the SIGGRAPH 2023 Best Paper Award. Each Gaussian ellipsoid $g_i$ is defined by the following parameters:

$$
g_i(\mathbf{x}) = \exp\left( -\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_i)^\top \boldsymbol{\Sigma}_i^{-1}(\mathbf{x} - \boldsymbol{\mu}_i) \right)
$$

where $\boldsymbol{\mu}_i \in \mathbb{R}^3$ is the mean (3D position), $\boldsymbol{\Sigma}_i = \mathbf{R}_i \mathbf{S}_i \mathbf{S}_i^\top \mathbf{R}_i^\top$ is the covariance matrix (generated by rotation $\mathbf{R}_i \in SO(3)$ and scaling $\mathbf{S}_i \in \mathbb{R}^3$ are parameterized to ensure that $\boldsymbol{\Sigma}_i$ is positive semi-definite), and the color is represented by the spherical harmonics (SH) coefficient $\mathbf{c}_i^k$ ($k$ is the SH order, usually $k=3$, corresponding to 9 coefficients).

The **optimization goal** is to minimize the photometric loss between the rendered image and the ground truth image, which is essentially to maximize the likelihood estimate:$$
\mathcal{L} = \sum_{\text{pixels}} \| \hat{C} - C_{\text{GT}} \|^2_2, \quad \text{Optimizer: SGD + Adam}
$$

By backpropagating the gradient, the Gaussian parameters $(\boldsymbol{\mu}_i, \mathbf{R}_i, \mathbf{S}_i, o_i, \mathbf{c}_i^k)$ are continuously updated. 3DGS also introduces Adaptive Density Control: Gaussians with large gradients are split into two small Gaussians, and Gaussians with too low transparency are deleted, thereby automatically adjusting the local resolution of the scene.

### 3.2 Rendering formula

3DGS uses tile-based splattering (Splatting) rendering instead of NeRF's ray-marching, by projecting a 3D Gaussian to a 2D image plane and performing alpha compositing by depth ordering:

$$
\hat{C} = \sum_{i \in \mathcal{N}} \mathbf{c}_i \, o_i \, \prod_{j=1}^{i-1}(1 - o_j), \quad o_i = o_i^{\text{raw}} \cdot \exp\left( -\frac{1}{2}(\mathbf{x}_i - \boldsymbol{\mu}_i)^\top \boldsymbol{\Sigma}_i^{-1}(\mathbf{x}_i - \boldsymbol{\mu}_i) \right)
$$

Where $o_i^{\text{raw}} \in [0,1]$ is a learnable opacity parameter, $\mathcal{N}$ is an ordered Gaussian list along the ray, and $\mathbf{x}_i$ is the 2D coordinate of the 3D Gaussian after projection transformation. Compared with NeRF volume rendering, 3DGS does not need to densely sample $K$ points along the ray and directly projects Gaussians to the image plane, improving the computational efficiency by 1-2 orders of magnitude.

### 3.3 Why is it suitable for UAV online planning?

Three characteristics of 3DGS make it a strong candidate for UAV online planning:- **Incremental mapping**: Gaussian ellipsoids can be added/deleted frame by frame, without the need for global optimization like MLP. GS-SLAM (Zhou et al., arxiv preprint, verification required) implements real-time dense SLAM for RGB-D cameras with tracking speeds up to 30 FPS
- **differentiable adaptive control**: Gaussians can be automatically split/merged through gradient signals to achieve adaptive allocation of resolution - automatically increase Gaussian density in geometrically complex areas and reduce redundancy in low gradient areas
- **Direct geometry query**: Gaussian ellipsoid itself is a clear primitive in space, which can directly calculate the SDF (Signed Distance Field) approximate distance between the drone and each Gaussian and generate safe planning constraints.

---

## 4. UAV-NeRF/GS fusion solution

### 4.1 Summary of representative work

**GaussianUAV (arxiv preprint, subject to verification)** is said to be a milestone work in this direction, proposing the integration of 3DGS into a UAV online planning framework. If this work is true, its core contributions should include the following design ideas: ① The neural mapping module uses 3DGS to achieve real-time incremental mapping; ② The safety planner builds a safe corridor (Safe Corridor) on Gaussian representation; ③ The GPU acceleration pipeline realizes the mapping-planning closed loop. However, after multiple rounds of searches, the article cannot be verified in the CVPR 2024 official paper list or mainstream databases. Readers are advised to check the latest arXiv records to confirm its official publication information.

**NICE-SLAM (Zhu et al., CVPR 2022)** proposes dense SLAM based on hierarchical neural implicit coding to achieve 5 Hz online reconstruction through multi-resolution feature grids, which is significantly better than the 0.5 Hz reconstruction speed of the original iMap. The layered design of NICE-SLAM makes it particularly suitable for incremental mapping needs in UAV scenarios.

**Vox-Fusion (Yi et al., ICRA 2023)** combines neural implicit representation with a voxel fusion framework for the first time to achieve real-time incremental mapping of monocular cameras and support dense path planning for UAVs.

**Co-SLAM (Wang et al., CVPR 2023)** uses hash-encoded neural implicit representation and joint coordinate encoding to achieve 10 Hz real-time mapping and positioning, and ensures global consistency through Bundle Adjustment optimization.**NKSR — Neural Kernel Surface Reconstruction (L. Ye et al., CVPR 2023)** Enables high-quality geometric reconstruction through neural kernel surface reconstruction, providing a more accurate map representation for UAV collision detection. NKSR uses Neural Kernel Fields to recover high-quality surfaces from dense point clouds, with excellent generalization capabilities in large-scale scenes.

### 4.2 Next-Best-View (NBV) active sensing

NBV planning is the core issue of UAV active sensing: given the currently observed part of the scene, select the next optimal observation pose to maximize information gain. The neural rendering method provides a new information gain measurement method for NBV - no longer relying on the coverage statistics of traditional geometric methods, but using the uncertainty of the neural field to guide exploration.

**How information gain is calculated** can be roughly divided into three categories according to different methods:

1. **Based on ray uncertainty** (represented by InfoNeRF, arxiv preprint, need to verify): For each ray $r$, estimate the variance of its color prediction $\mathbb{V}[C(\mathbf{r})]$, which can be approximated by injecting noise into the same ray and rendering it multiple times. NBV selects the candidate pose that maximizes the overall mutual information $I(\mathbf{r}; \Theta) = \mathbb{V}[C(\mathbf{r})]$ and guides the UAV to fly to the area where the ray prediction is most uncertain
2. **Reconstruction loss based on radiation field** (represented by NeRF-NBV, arxiv preprint, need to be verified): directly predict the rendering quality loss of the virtual perspective on the neural radiation field, and select the candidate pose that can maximize the reconstruction error of the new perspective - essentially exploring "the weakest point of the current field representation"
3. **Based on Gaussian coverage** (represented by Gaussian NBV, arxiv preprint, need to be verified): Use the anisotropic Gaussian distribution of 3DGS to directly calculate the observation coverage and geometric uncertainty. Specifically, a hypothetical "depth map" is rendered for each candidate pose, the number of uncovered Gaussians or depth uncertainty is counted, and the direction with the sparsest Gaussian ellipsoid distribution is selected as the NBV| Methods | Publication | Information Gain Measure | Planning Frequency | Remarks |
|------|------|-------------|---------|------|
| InfoNeRF | NeurIPS 2022 | Mutual Information (Mutual Information) | < 1 Hz | ⚠️ arxiv preprint, verification required |
| NeRF-NBV | ICRA 2023 | Radiation field reconstruction uncertainty | ~1 Hz | ⚠️ arxiv preprint, verification required |
| Gaussian NBV | ICRA 2024 | Gaussian coverage | ~5 Hz | ⚠️ arxiv preprint, verification required |
| Neural Implicit Map for UAV | ICRA 2023 | Voxel reconstruction uncertainty | ~5 Hz | ⚠️ arxiv preprint, verification required |

> **Note**: The papers marked "⚠️ arxiv preprint, need to be verified" in the above table cannot be verified in the official proceedings of the corresponding conference. The work with the same name could not be retrieved from the NeurIPS 2022 / ICRA 2023 / ICRA 2024 paper list. Readers are advised to check the author's latest arXiv submission record or contact the author for confirmation. The same is true for GaussianUAV, whose CVPR 2024 publication status cannot be verified.

### 4.3 Special considerations for urban scenes

The urban canyon environment poses unique engineering challenges to neural rendering methods, requiring targeted adaptation at the algorithm design level.

**Large-scale scene decomposition** is the primary difficulty: an entire city block cannot be represented by a single MLP or set of Gaussians. Mainstream solutions adopt a hierarchical chunking strategy—dividing the scene into multiple local chunks. Each chunk independently maintains a set of neural field representations (or independent Gaussian sets), and the UAV dynamically loads/unloads adjacent chunks during movement. The progressive data partitioning and seamless merging mechanism proposed by VastGaussian (CVPR 2024) is a representative work of this idea.**Building facade occlusion** is another key challenge: urban building surfaces have dense textures and complex geometric structures, and raw NeRF is prone to aliasing artifacts at slender edges. Mip-NeRF 360 (Barron et al., 2022) effectively alleviates this problem by introducing anti-aliasing cone ray sampling and nonlinear scene parameterization (nonlinear scene parameterization). The core of its technology is to replace the scalar distance $t$ with the average distance interval along the ray $[\hat{t}_i - \gamma_i, \hat{t}_i + \gamma_i]$, making MLP Ability to perceive the actual spatial span of the sampled area, resulting in correct anti-aliasing at different scales.

**Multi-layer flight planning** requires complete modeling of three-dimensional space: UAV not only needs to avoid obstacles in the horizontal direction, but also needs to deal with vertical dimensional challenges such as inter-floor passages and cantilever structures at different heights. 2D bird's-eye view methods completely fail in this scenario and must rely on 3D neural field representations. The unbounded scene modeling capability of Mip-NeRF 360 provides a scalable technical foundation for multi-layered urban scenes.

---

## 5. Engineering challenges and cutting-edge directions

### 5.1 GPU computing power constraints

The computing power of the embedded GPU of consumer UAV (such as Jetson Orin) is about 1/10-1/20 of the desktop RTX 3090. The real-time rendering of 3DGS relies on a large number of matrix operations. Current solutions generally adopt the following strategies to narrow the computing power gap:

- **Asynchronous pipeline**: The mapping thread (Gaussian optimization) and the planning thread (trajectory generation) are executed in parallel, and read and write conflicts are avoided through double buffering.
- **Downsampling Rendering**: Low-resolution rendering ($640\times 480$) and then upsampling to the target resolution, sacrificing some accuracy in exchange for frame rate
- **Pruning + Culling**: pruning based on opacity and distance from the camera, combined with spatial clipping of Gaussian ellipsoids (frustum culling), typical scenes can reduce the number of Gaussians by 60-80% without significantly affecting the rendering quality

### 5.2 Dynamic object interference

City streets are filled with dynamic objects such as vehicles and pedestrians. Neural field methods rely on the static assumption of the scene, and dynamic objects can introduce artifacts and contaminate the map. Existing solutions cover three levels:- **Dynamic foreground segmentation**: During the optimization process, dynamic objects are modeled as independent Gaussian groups (such as the dynamic removal strategy of GS-SLAM), and are actively deleted after the observation is completed, thereby isolating dynamic interference from the main map
- **Multi-agent collaboration**: Multiple UAVs collaborate to build maps and filter dynamic objects through time synchronization and pose map optimization; collaborative observation can also accelerate the coverage of static areas
- **4D NeRF**: D-NeRF (Pumarola et al., 2021) introduces the time dimension to model dynamic scenes and predicts the deformation field $\Delta \mathbf{x}(t)$ of each 3D point through additional MLP branches, but real-time performance is still a bottleneck

### 5.3 Loop closure detection and map fusion

UAVs require closed-loop detection to correct accumulated drift when flying in large-scale urban scenes. While traditional approaches rely on ICP or bag-of-words models, neural field methods offer a more expressive alternative:

- **Pose Graph Optimization + Neural Bundle Adjustment**: Jointly optimize camera pose and neural field parameters to simultaneously minimize geometric reprojection errors and photometric rendering losses through the BA framework
- **Rendering-based closed loop**: When the UAV returns to the mapped area, the closed loop is detected by comparing the similarity (PSNR/SSIM) between the rendered image and the observed image; if the similarity drops sharply, there may be pose drift. This method can theoretically detect rotational drift $< 5^\circ$

Kimera (Rosinol et al., 2023) provides a modular metric-semantic SLAM framework that can serve as a bridging solution between the neural field backend and the classic pose graph frontend.

### 5.4 Sim2Real migration

Neural rendering methods are trained in simulation environments (such as Habitat-sim, Isaac Sim), and there is a **domain gap** (texture differences, lighting changes, camera calibration errors) when deployed directly to real UAVs. Mitigation strategies include:- **Domain Randomization**: Randomize textures, lighting conditions, camera internal and external parameters in simulation to increase the diversity of training data
- **Neural Rendering Adaptation**: Use a small number (10-50) of real images to fine-tune neural field parameters to fill the simulation-real appearance gap
- **Uncertainty-aware Planning**: Introduce safety margin (Safety Margin) at the planning level to absorb the remaining field gaps, ensuring that even if the map accuracy is slightly lower than the simulation level, the trajectory remains safe

---

## 6. Open source code resources| Project | Paper | Code | Notes |
|------|------|------|------|
| 3D Gaussian Splatting | Kerbl et al., ACM ToG 2023 | [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) | Original 3DGS implementation |
| Instant-NGP | Müller et al., SIGGRAPH 2022 | [NVlabs/instant-ngp](https://github.com/NVlabs/instant-ngp) | Fast neural field mapping |
| GS-SLAM | Zhou et al., 2023 | [youmi-zym/GS-SLAM](https://github.com/youmi-zym/GS-SLAM) | Real-time GS SLAM, arxiv preprint |
| Co-SLAM | Wang et al., CVPR 2023 | [HengyiWang/Co-SLAM](https://github.com/HengyiWang/Co-SLAM) | Joint coordinates and hash coding |
| NICE-SLAM | Zhu et al., CVPR 2022 | [cvg/nice-slam](https://github.com/cvg/nice-slam) | Hierarchical Neural Implicit SLAM |
| Vox-Fusion | Yi et al., ICRA 2023 | [ZhiangChen/Vox-Fusion](https://github.com/ZhiangChen/Vox-Fusion) | Monocular real-time incremental mapping |
| Kimera | Rosinol et al., RAL 2023 | [MIT SPARK/Kimera](https://github.com/MIT-SPARK/Kimera) | Metric-Semantic SLAM Framework |
| NKSR | L. Ye et al., CVPR 2023 | [nv-tlabs/NKSR](https://github.com/nv-tlabs/NKSR) | NVIDIA neural core surface reconstruction |---

## 7. Summary and Outlook

NeRF/3DGS brings three major innovations: continuity, differentiability, and photo-realistic** to urban low-altitude UAV route planning. Compared with traditional voxel methods, neural rendering methods have significant advantages in occlusion reasoning, information gain estimation, and photo-realistic visualization. With its incrementally updateable Gaussian representation, 3DGS has become the technology path closest to practical implementation of UAV online planning.

However, **large-scale scene scalability**, **dynamic environment robustness** and **edge real-time performance** are still the three core bottlenecks restricting implementation. Future research directions may include:

- **Sparse Neural Representation + Sparse Planning**: Maintain neural fields only in key areas, combined with sparse optimization to achieve city-scale planning
- **Multi-modal fusion**: Deeply integrate multi-sensor signals such as GNSS, IMU, LIDAR and neural rendering to improve positioning accuracy and map integrity
- **Embodied Intelligence Alignment**: Combined with the visual-language model (VLM) to understand the semantics of urban scenes, enabling UAVs to have "understanding-planning" capabilities instead of just "perception-avoidance"

---

## References

- Barron, J. T., Mildenhall, B., Tancik, M., Hedman, P., Martin-Brualla, R., & Srinivasan, P. P. (2021). Mip-NeRF: A multiscale representation for anti-aliasing neural radiance fields. *ICCV*. https://doi.org/10.1109/ICCV48922.2021.00598

- Barron, J. T., Mildenhall, B., Verbin, D., Srinivasan, P. P., & Hedman, P. (2022). Mip-NeRF 360: Unbounded anti-aliasing neural radiance fields. *CVPR*. https://doi.org/10.1109/CVPR52688.2022.00530- Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). 3D Gaussian Splatting for real-time radiance field rendering. *ACM Transactions on Graphics*, 42(4), 1–14. https://doi.org/10.1145/3592403

- Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., & Ng, R. (2020). NeRF: Representing scenes as neural radiance fields for view synthesis. *ECCV*. https://doi.org/10.1007/978-3-030-58452-8_24

- Müller, T., Evans, A., Schied, C., & Keller, A. (2022). Instant neural graphics primitives with a multiresolution hash encoding. *ACM Transactions on Graphics*, 41(4), 1–15. https://doi.org/10.1145/3528223.3528347

- Pumarola, A., Corona, E., Pons-Moll, G., & Moreno-Nuguer, F. (2021). D-NeRF: Neural radiance fields for dynamic scenes. *NeurIPS*, 34, 10318–10329.- Rosinol, A., Abate, A., Chang, Y., & Carlone, L. (2023). Kimera: An open-source library for real-time metric-semantic localization and mapping. *IEEE Robotics and Automation Letters*, 8(3), 1475–1482. https://doi.org/10.1109/LRA.2023.3243839

- Wang, H., Wang, J., & Agapito, L. (2023). Co-SLAM: Joint coordinate and sparse parametric encodings for neural real-time SLAM. *CVPR*. https://doi.org/10.1109/CVPR52729.2023.00446

- Yi, Z., Chen, Z., S., G. K., Carlone, L., & Comport, A. I. (2023). Vox-Fusion: Dense SLAM with neural implicit surface representation. *ICRA*. https://doi.org/10.1109/ICRA46671.2023.10160912

- Ye, L., Misra, I., & Ranjan, R. (2023). Neural kernel surface reconstruction. *CVPR*.

- Zhou, Y., Sun, J., Zha, Z., & Zeng, W. (2023). GS-SLAM: Dense SLAM via 3D Gaussian Splatting. *arxiv:2308.04306*. (⚠️ Preprint, venue to be confirmed)- Zhu, Z., Peng, S., Larsson, V., Cui, H., Oswald, M. R., Geiger, A., & Pollefeys, M. (2022). NICE-SLAM: Neural implicit scalable encoding for SLAM. *CVPR*. https://doi.org/10.1109/CVPR52688.2022.01278

---

*This article is the first extended chapter in a series of articles on urban low-altitude drone route planning. The follow-up will cover direction two: end-to-end planning based on Transformer, so stay tuned. *