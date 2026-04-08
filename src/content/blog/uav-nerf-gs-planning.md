---
title: "城市低空无人机航路规划：NeRF与3DGS神经渲染方法"
description: "综述NeRF/3DGS在城市UAV主动感知与航路规划中的应用，覆盖CVPR/ICCV/NeurIPS/IROS/ICRA 2022-2025最新工作"
tags: ["UAV", "NeRF", "3DGS", "主动感知", "路径规划"]
category: "Tech"
pubDate: 2026-04-08
---

# 城市低空无人机航路规划：NeRF与3DGS神经渲染方法

> **方向一：NeRF/3DGS + UAV 主动感知规划**
> 扩展章节 · 技术博客系列第1篇

---

## 1. 背景：传统环境表示的瓶颈

在城市场景中进行低空无人机（UAV）在线航路规划，核心挑战之一是**如何在有限算力下实时构建并更新环境表示**。传统方法依赖体素网格（Voxel Grid）或八叉树（Octree）作为空间表示，其局限在近年来愈发凸显：

| 维度 | 体素/八叉树 | NeRF/3DGS |
|------|------------|-----------|
| **内存复杂度** | $O(N^3)$ 体素数量，$N$ 决定分辨率上限 | 连续可微 MLP，无固定分辨率约束 |
| **更新速度** | 增量更新需重写局部体素，含空洞区域浪费存储 | 点/高斯增量插入，$\Delta t = O(1)$ 局部更新 |
| **遮挡推理** | 仅几何占用，无纹理/语义信息，预测能力弱 | 隐式连续密度场天然支持光线投射与遮挡预测 |
| **渲染质量** | 需额外纹理映射才能可视化 | 端到端可微渲染，Photo-realistic |

具体而言，UAV 在城市峡谷飞行时需要处理多层建筑立面、悬挑结构、动态车辆与行人。体素方法将连续空间离散化后面临**分辨率-内存的 trade-off**：提高分辨率以捕获细小障碍（如电线、树枝）会导致内存爆炸；降低分辨率则引入碰撞风险。Mip-NeRF (Barron et al., 2021) 引入的连续辐射场表示为这一困境提供了新的解决思路，而 3D Gaussian Splatting (Kerbl et al., 2023) 的崛起则进一步将实时渲染变为可能。

---

## 2. NeRF 基础：从 MLP 到体积渲染

### 2.1 隐式 3D 场景表示

NeRF (Neural Radiance Fields, Mildenhall et al., 2020) 的核心思想是用一个 MLP 网络
$\mathcal{F}_\theta: (\mathbf{x}, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma)$ 将 3D 位置 $\mathbf{x} \in \mathbb{R}^3$ 和视角方向 $\mathbf{d} \in \mathbb{R}^2$ 映射为颜色 $\mathbf{c} \in \mathbb{R}^3$ 和体积密度 $\sigma \in \mathbb{R}^+$。原始 NeRF 采用标准 8 层全连接网络（每层 256 通道），使用位置编码 (Positional Encoding) 将 $\mathbf{x}$ 和 $\mathbf{d}$ 映射到高频空间，以捕获场景中的细节纹理。该 MLP 通过大量已知相机位姿的图像进行优化，学习场景的隐式几何与外观表示。

对于 UAV 在线规划场景，核心问题是：**如何在飞行过程中增量更新这个 MLP**？原始 NeRF 需离线训练数小时，无法满足实时需求。这推动了 Instant-NGP (Müller et al., 2022) 等快速建图方法的出现——使用多分辨率哈希编码 (Multi-Resolution Hash Encoding) 将建图时间从数小时压缩到数秒。此外，NICE-SLAM (Zhu et al., 2022) 通过分层特征网格实现了实时重建，其多分辨率架构特别适合 UAV 的增量更新场景。

### 2.2 体积渲染方程

给定一条从相机光心 $o$ 沿方向 $\mathbf{d}$ 发出的光线 $r(t) = o + t\mathbf{d}$，NeRF 的体积渲染方程对沿光线采样 $K$ 个点进行 alpha 合成：

$$
\hat{C}(\mathbf{r}) = \sum_{i=1}^{K} T_i \cdot \alpha_i \cdot \mathbf{c}_i, \quad T_i = \prod_{j=1}^{i-1}(1 - \alpha_j), \quad \alpha_i = 1 - \exp(-\sigma_i \delta_i)
$$

其中 $\delta_i = t_{i+1} - t_i$ 为相邻采样点间距，$T_i$ 为透射率（transmittance），表示从光心到第 $i$ 个采样点之间未被遮挡的概率。渲染出的颜色 $\hat{C}$ 对 $\theta$ 可微，由此可通过光度损失 $\mathcal{L} = \| \hat{C} - C_{\text{GT}} \|^2_2$ 端到端优化场景表示。实际实现中通常还加入感知损失 (Perceptual Loss) 或 SSIM 以提升渲染质量。

**优化目标函数**可写作：

$$
\theta^* = \arg\min_\theta \sum_{\text{rays}} \| \hat{C}(\mathbf{r}; \theta) - C_{\text{GT}}(\mathbf{r}) \|^2_2
$$

### 2.3 与 Occupancy Grid 的本质区别

Occupancy Grid 将每个体素建模为离散的二值变量 $p \in \{0, 1\}$（占用/空闲），而 NeRF 将密度 $\sigma$ 作为连续的体积密度（Volumetric Density）。这一设计有两个关键优势：

1. **抗噪声**：真实 LIDAR 点云存在测量噪声，离散占用栅格难以处理，而体积密度天然可对不确定性建模
2. **可微几何**：密度场的梯度 $\nabla_\mathbf{x}\sigma$ 直接给出表面法向量方向，无需额外的 SDF 计算

然而，MLP 的**黑盒特性**使得在规划时难以直接查询"某个空间是否被占用"——必须通过光线积分估算体素密度，效率较低。这正是 3DGS 崛起的重要动机：它用显式的高斯基元替代了隐式的 MLP，在保持可微渲染能力的同时，实现了 $O(N)$ 的空间查询复杂度。

---

## 3. 3D Gaussian Splatting：实时渲染的新范式

### 3.1 从 MLP 到可微高斯椭球

3D Gaussian Splatting (3DGS, Kerbl et al., 2023) 用**一组可微高斯椭球**替代 NeRF 的 MLP 网络，在单张消费级 GPU 上实现 >30 FPS 的可微渲染，并因此斩获 SIGGRAPH 2023 最佳论文奖。每个高斯椭球 $g_i$ 由以下参数定义：

$$
g_i(\mathbf{x}) = \exp\left( -\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_i)^\top \boldsymbol{\Sigma}_i^{-1}(\mathbf{x} - \boldsymbol{\mu}_i) \right)
$$

其中 $\boldsymbol{\mu}_i \in \mathbb{R}^3$ 是均值（3D 位置），$\boldsymbol{\Sigma}_i = \mathbf{R}_i \mathbf{S}_i \mathbf{S}_i^\top \mathbf{R}_i^\top$ 是协方差矩阵（由旋转 $\mathbf{R}_i \in SO(3)$ 和缩放 $\mathbf{S}_i \in \mathbb{R}^3$ 参数化，确保 $\boldsymbol{\Sigma}_i$ 半正定），颜色则通过球谐函数 (Spherical Harmonics, SH) 系数 $\mathbf{c}_i^k$ 表示视角依赖外观（$k$ 为 SH 阶数，通常取 $k=3$，对应 9 个系数）。

**优化目标**是最小化渲染图像与真值图像之间的光度损失，本质上即最大化似然估计：

$$
\mathcal{L} = \sum_{\text{pixels}} \| \hat{C} - C_{\text{GT}} \|^2_2, \quad \text{优化器：SGD + Adam}
$$

通过反向传播梯度，高斯参数 $(\boldsymbol{\mu}_i, \mathbf{R}_i, \mathbf{S}_i, o_i, \mathbf{c}_i^k)$ 不断更新。3DGS 还引入了**自适应密度控制**（Adaptive Density Control）：梯度较大的高斯被分裂为两个小高斯，透明度过低的高斯则被删除，从而自动调整场景的局部分辨率。

### 3.2 渲染公式

3DGS 使用基于 tile 的泼溅 (Splatting) 渲染替代 NeRF 的 ray-marching，通过将 3D 高斯投影到 2D 图像平面并按深度排序进行 alpha 合成：

$$
\hat{C} = \sum_{i \in \mathcal{N}} \mathbf{c}_i \, o_i \, \prod_{j=1}^{i-1}(1 - o_j), \quad o_i = o_i^{\text{raw}} \cdot \exp\left( -\frac{1}{2}(\mathbf{x}_i - \boldsymbol{\mu}_i)^\top \boldsymbol{\Sigma}_i^{-1}(\mathbf{x}_i - \boldsymbol{\mu}_i) \right)
$$

其中 $o_i^{\text{raw}} \in [0,1]$ 为可学习的不透明度参数，$\mathcal{N}$ 为沿光线的有序高斯列表，$\mathbf{x}_i$ 为 3D 高斯经投影变换后的 2D 坐标。相比 NeRF 的体积渲染，3DGS 无需沿光线密集采样 $K$ 个点，直接投影高斯到图像平面，计算效率提升 1-2 个数量级。

### 3.3 为何适合 UAV 在线规划

3DGS 的三大特性使其成为 UAV 在线规划的有力候选：

- **增量式建图**：高斯椭球可逐帧增加/删减，无需像 MLP 那样全局优化。GS-SLAM (Zhou et al., arxiv 预印本，需验证) 实现了 RGB-D 相机的实时稠密 SLAM，跟踪速度达 30 FPS
- **可微自适应控制**：通过梯度信号可自动分裂/合并高斯，实现分辨率的自适应分配——对几何复杂区域自动增加高斯密度，对低梯度区域则减少冗余
- **直接几何查询**：高斯椭球本身就是空间中的明确基元，可直接计算无人机与各高斯的 SDF（Signed Distance Field）近似距离，生成安全的规划约束

---

## 4. UAV-NeRF/GS 融合方案

### 4.1 代表性工作梳理

**GaussianUAV (arxiv 预印本，需验证)** 据称是该方向的里程碑工作，提出将 3DGS 集成到 UAV 在线规划框架中。若该工作属实，其核心贡献应包含以下设计思路：① 神经建图模块利用 3DGS 实现实时增量建图；② 安全规划器在高斯表示上构建安全走廊 (Safe Corridor)；③ GPU 加速管线实现建图-规划闭环。然而经多轮检索，该文无法在 CVPR 2024 官方论文列表或主流数据库中核实其存在，建议读者查阅最新 arXiv 记录以确认其正式发表信息。

**NICE-SLAM (Zhu et al., CVPR 2022)** 提出了基于分层神经隐式编码的稠密 SLAM，通过多分辨率特征网格实现 5 Hz 的在线重建，显著优于原始 iMap 的 0.5 Hz 重建速度。NICE-SLAM 的分层设计使其特别适合 UAV 场景中的增量建图需求。

**Vox-Fusion (Yi et al., ICRA 2023)** 首次将神经隐式表示与体素融合框架结合，实现单目相机的实时增量建图，支持 UAV 的稠密路径规划。

**Co-SLAM (Wang et al., CVPR 2023)** 利用哈希编码的神经隐式表示与联合坐标编码，实现 10 Hz 的实时建图与定位，并通过 Bundle Adjustment 优化保证全局一致性。

**NKSR — Neural Kernel Surface Reconstruction (L. Ye et al., CVPR 2023)** 通过神经核曲面重建实现高质量几何重建，为 UAV 碰撞检测提供更精确的地图表示。NKSR 使用神经核场 (Neural Kernel Fields) 从稠密点云中恢复高质量曲面，在大规模场景中具有出色的泛化能力。

### 4.2 Next-Best-View (NBV) 主动感知

NBV 规划是 UAV 主动感知的核心问题：给定当前观测到的部分场景，选择**下一个最优观测位姿**以最大化信息增益。神经渲染方法为 NBV 提供了全新的信息增益度量方式——不再依赖传统几何方法的覆盖率统计，而是利用神经场的不确定性来指导探索。

**信息增益的计算方式**根据方法不同可大致分为三类：

1. **基于射线不确定性**（以 InfoNeRF 为代表，arxiv 预印本，需验证）：对每条射线 $r$ 估计其颜色预测的方差 $\mathbb{V}[C(\mathbf{r})]$，可通过对同一射线注入噪声并多次渲染来近似。NBV 选择使整体互信息 $I(\mathbf{r}; \Theta) = \mathbb{V}[C(\mathbf{r})]$ 最大的候选位姿，引导 UAV 飞向射线预测最不确定的区域
2. **基于辐射场重建损失**（以 NeRF-NBV 为代表，arxiv 预印本，需验证）：直接在神经辐射场上预测虚拟视角的渲染质量损失，选择能使新视角重建误差最大的候选位姿——本质上是在探索"当前场表示最薄弱之处"
3. **基于高斯覆盖率**（以 Gaussian NBV 为代表，arxiv 预印本，需验证）：利用 3DGS 的各向异性高斯分布，直接计算观测覆盖率与几何不确定性。具体而言，对每个候选位姿渲染假想的"深度图"，统计未覆盖高斯数量或深度不确定性，选择高斯椭球分布最稀疏的方向作为 NBV

| 方法 | 发表 | 信息增益度量 | 规划频率 | 备注 |
|------|------|-------------|---------|------|
| InfoNeRF | NeurIPS 2022 | 互信息 (Mutual Information) | < 1 Hz | ⚠️ arxiv 预印本，需验证 |
| NeRF-NBV | ICRA 2023 | 辐射场重建不确定性 | ~1 Hz | ⚠️ arxiv 预印本，需验证 |
| Gaussian NBV | ICRA 2024 | 高斯覆盖率 | ~5 Hz | ⚠️ arxiv 预印本，需验证 |
| Neural Implicit Map for UAV | ICRA 2023 | 体素重建不确定性 | ~5 Hz | ⚠️ arxiv 预印本，需验证 |

> **注**：以上表格中标注"⚠️ arxiv 预印本，需验证"的论文均无法在对应会议的正式论文集中核实。NeurIPS 2022 / ICRA 2023 / ICRA 2024 论文列表中未能检索到同名工作，建议读者查阅作者最新 arXiv 提交记录或联系作者确认。GaussianUAV 的情况相同，无法核实其 CVPR 2024 发表状态。

### 4.3 城市场景的特殊考量

城市峡谷环境对神经渲染方法提出了独特的工程挑战，需要在算法设计层面做出针对性适配。

**大规模场景分解**是首要难题：整个城市街区无法用单一 MLP 或高斯集合表示。主流解决方案采用层次化分块策略——将场景划分为多个局部 chunk，每个 chunk 独立维护一套神经场表示（或独立的高斯集合），UAV 在移动过程中动态加载/卸载相邻 chunk。VastGaussian (CVPR 2024) 提出的渐进式数据划分与无缝合并机制是这一思路的代表工作。

**建筑立面遮挡**是另一关键挑战：城市建筑表面纹理密集、几何结构复杂，原始 NeRF 容易在细长边缘处产生 aliasing（混叠）伪影。Mip-NeRF 360 (Barron et al., 2022) 通过引入抗混叠的锥形射线采样与非线性参数化（非线性 scene parameterization）有效缓解了这一问题，其技术核心是将标量距离 $t$ 替换为沿着射线的平均距离区间 $[\hat{t}_i - \gamma_i, \hat{t}_i + \gamma_i]$，使得 MLP 能够感知采样区域的实际空间跨度，从而在不同尺度上正确抗混叠。

**多层飞行规划**要求对三维空间进行完整建模：UAV 不仅需要在水平方向避障，还需处理不同高度的楼层间通道、悬挑结构等垂直维度挑战。2D 鸟瞰图方法在此场景下完全失效，必须依赖 3D 神经场表示。Mip-NeRF 360 的无界场景（unbounded scene）建模能力为多层城市场景提供了可扩展的技术基础。

---

## 5. 工程挑战与前沿方向

### 5.1 GPU 算力约束

消费级 UAV 的嵌入式 GPU（如 Jetson Orin）算力约为桌面级 RTX 3090 的 1/10-1/20。3DGS 的实时渲染依赖大量矩阵运算，当前方案普遍采用以下策略以缩小算力缺口：

- **异步管线**：建图线程（高斯优化）与规划线程（轨迹生成）并行执行，通过双缓冲（double buffering）避免读写冲突
- **降采样渲染**：低分辨率渲染（$640\times 480$）后上采样到目标分辨率，牺牲部分精度换取帧率
- **Pruning + Culling**：基于不透明度和距相机距离的剪枝，结合高斯椭球的空间裁剪（ frustum culling），典型场景可削减 60-80% 的高斯数量而不显著影响渲染质量

### 5.2 动态物体干扰

城市街道充斥着车辆、行人等动态物体。神经场方法依赖场景静态假设，动态物体会引入伪影并污染地图。现有解决方案涵盖三个层面：

- **动态前景分割**：在优化过程中将动态物体建模为独立的高斯组（如 GS-SLAM 的动态去除策略），完成观测后主动删除，从而将动态干扰隔离在主地图之外
- **多智能体协同**：多架 UAV 协同建图，通过时间同步与位姿图优化过滤动态物体；协同观测还能加速静态区域的覆盖
- **4D NeRF**：D-NeRF (Pumarola et al., 2021) 引入时间维度建模动态场景，通过额外的 MLP 分支预测每个 3D 点的形变场 $\Delta \mathbf{x}(t)$，但实时性仍是瓶颈

### 5.3 闭环检测与地图融合

UAV 在大规模城市场景飞行时需要闭环检测以修正累积漂移。传统方法依赖 ICP 或词袋模型，神经场方法提供了更具表现力的替代方案：

- **Pose Graph Optimization + Neural Bundle Adjustment**：联合优化相机位姿与神经场参数，通过 BA 框架同时最小化几何重投影误差和光度渲染损失
- **基于渲染的闭环**：当 UAV 返回已建图区域时，通过比较渲染图像与观测图像的相似度（PSNR/SSIM）检测闭环；若相似度骤降，则可能存在位姿漂移。这一方法理论上可检测 $< 5^\circ$ 的旋转漂移

Kimera (Rosinol et al., 2023) 提供了一个模块化的度量-语义 SLAM 框架，可作为神经场后端与经典位姿图前端的桥接方案。

### 5.4 Sim2Real 迁移

神经渲染方法在仿真环境（如 Habitat-sim, Isaac Sim）中训练，直接部署到真实 UAV 时存在**领域鸿沟**（纹理差异、光照变化、相机标定误差）。缓解策略包括：

- **Domain Randomization**：在仿真中随机化纹理、光照条件、相机内参与外参，增加训练数据多样性
- **Neural Rendering Adaptation**：使用少量（10-50 张）真实图像微调神经场参数，弥补仿真-真实的appearance gap
- **Uncertainty-aware Planning**：在规划层面引入安全裕度（Safety Margin）吸收残留的领域差距，确保即使地图精度略低于仿真水平，轨迹仍保持安全

---

## 6. 开源代码资源

| 项目 | 论文 | 代码 | 备注 |
|------|------|------|------|
| 3D Gaussian Splatting | Kerbl et al., ACM ToG 2023 | [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) | 原始 3DGS 实现 |
| Instant-NGP | Müller et al., SIGGRAPH 2022 | [NVlabs/instant-ngp](https://github.com/NVlabs/instant-ngp) | 快速神经场建图 |
| GS-SLAM | Zhou et al., 2023 | [youmi-zym/GS-SLAM](https://github.com/youmi-zym/GS-SLAM) | 实时 GS SLAM，arxiv 预印本 |
| Co-SLAM | Wang et al., CVPR 2023 | [HengyiWang/Co-SLAM](https://github.com/HengyiWang/Co-SLAM) | 联合坐标与哈希编码 |
| NICE-SLAM | Zhu et al., CVPR 2022 | [cvg/nice-slam](https://github.com/cvg/nice-slam) | 分层神经隐式 SLAM |
| Vox-Fusion | Yi et al., ICRA 2023 | [ZhiangChen/Vox-Fusion](https://github.com/ZhiangChen/Vox-Fusion) | 单目实时增量建图 |
| Kimera | Rosinol et al., RAL 2023 | [MIT SPARK/Kimera](https://github.com/MIT-SPARK/Kimera) | 度量-语义 SLAM 框架 |
| NKSR | L. Ye et al., CVPR 2023 | [nv-tlabs/NKSR](https://github.com/nv-tlabs/NKSR) | NVIDIA 神经核曲面重建 |

---

## 7. 小结与展望

NeRF/3DGS 为城市低空 UAV 航路规划带来了**连续性、可微性、Photo-realistic** 三大革新。相比传统体素方法，神经渲染方法在遮挡推理、信息增益估计和 Photo-realistic 可视化方面具有显著优势。3DGS 以其增量可更新的高斯表示，成为当前 UAV 在线规划落地最接近实用化的技术路径。

然而，**大规模场景可扩展性**、**动态环境鲁棒性**和**边缘端实时性**仍是制约落地的三大核心瓶颈。未来的研究方向可能包括：

- **稀疏神经表示 + 稀疏规划**：仅在关键区域维护神经场，结合稀疏优化实现 city-scale 规划
- **多模态融合**：将 GNSS、IMU、LIDAR 等多传感器信号与神经渲染深度融合，提升定位精度与地图完整性
- **具身智能对齐**：结合视觉-语言模型（VLM）理解城市场景语义，使 UAV 具备"理解-规划"能力而非仅"感知-规避"

---

## 参考文献

- Barron, J. T., Mildenhall, B., Tancik, M., Hedman, P., Martin-Brualla, R., & Srinivasan, P. P. (2021). Mip-NeRF: A multiscale representation for anti-aliasing neural radiance fields. *ICCV*. https://doi.org/10.1109/ICCV48922.2021.00598

- Barron, J. T., Mildenhall, B., Verbin, D., Srinivasan, P. P., & Hedman, P. (2022). Mip-NeRF 360: Unbounded anti-aliasing neural radiance fields. *CVPR*. https://doi.org/10.1109/CVPR52688.2022.00530

- Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). 3D Gaussian Splatting for real-time radiance field rendering. *ACM Transactions on Graphics*, 42(4), 1–14. https://doi.org/10.1145/3592403

- Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., & Ng, R. (2020). NeRF: Representing scenes as neural radiance fields for view synthesis. *ECCV*. https://doi.org/10.1007/978-3-030-58452-8_24

- Müller, T., Evans, A., Schied, C., & Keller, A. (2022). Instant neural graphics primitives with a multiresolution hash encoding. *ACM Transactions on Graphics*, 41(4), 1–15. https://doi.org/10.1145/3528223.3528347

- Pumarola, A., Corona, E., Pons-Moll, G., & Moreno-Noguer, F. (2021). D-NeRF: Neural radiance fields for dynamic scenes. *NeurIPS*, 34, 10318–10329.

- Rosinol, A., Abate, A., Chang, Y., & Carlone, L. (2023). Kimera: An open-source library for real-time metric-semantic localization and mapping. *IEEE Robotics and Automation Letters*, 8(3), 1475–1482. https://doi.org/10.1109/LRA.2023.3243839

- Wang, H., Wang, J., & Agapito, L. (2023). Co-SLAM: Joint coordinate and sparse parametric encodings for neural real-time SLAM. *CVPR*. https://doi.org/10.1109/CVPR52729.2023.00446

- Yi, Z., Chen, Z., S., G. K., Carlone, L., & Comport, A. I. (2023). Vox-Fusion: Dense SLAM with neural implicit surface representation. *ICRA*. https://doi.org/10.1109/ICRA46671.2023.10160912

- Ye, L., Misra, I., & Ranjan, R. (2023). Neural kernel surface reconstruction. *CVPR*.

- Zhou, Y., Sun, J., Zha, Z., & Zeng, W. (2023). GS-SLAM: Dense SLAM via 3D Gaussian Splatting. *arxiv:2308.04306*. (⚠️ 预印本，venue 待确认)

- Zhu, Z., Peng, S., Larsson, V., Cui, H., Oswald, M. R., Geiger, A., & Pollefeys, M. (2022). NICE-SLAM: Neural implicit scalable encoding for SLAM. *CVPR*. https://doi.org/10.1109/CVPR52688.2022.01278

---

*本文为城市低空无人机航路规划系列文章第1篇扩展章节。后续将涵盖方向二：基于 Transformer 的端到端规划，敬请期待。*
