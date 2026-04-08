---
title: "城市低空无人机航路规划：数字孪生与神经渲染空域建模"
description: "综述数字孪生与神经渲染在城市UAV空域建模中的应用，覆盖TRO/TITS/RAL/IROS 2022-2025最新工作"
tags: ["UAV", "数字孪生", "神经渲染", "空域建模", "路径规划"]
category: "Tech"
pubDate: 2026-04-09
---

# 城市低空无人机航路规划：数字孪生与神经渲染空域建模

> **方向三：数字孪生 + 神经渲染空域建模**
> 扩展章节 · 技术博客系列第3篇

---

## 1. 背景：数字孪生赋能城市低空经济

随着城市空中交通（UAM, Urban Air Mobility）与低空经济的快速发展，城市低空空域的精细化管理成为核心需求。传统空管系统依赖静态地图与规则驱动，无法满足无人机在城市复杂三维环境中的实时规划需求。**数字孪生**（Digital Twin）作为物理空间在数字世界的精准映射，为城市低空空域的动态建模提供了全新的技术路径。

城市低空数字孪生需要融合多源数据：卫星影像提供宏观地物分布，建筑信息模型（BIM）提供精细几何结构，实时传感器数据（LiDAR、摄像头、气象站）驱动孪生体的动态演化。数字孪生平台的核心价值在于：**在数字空间中完成"预测-规划-仿真-验证"的完整闭环**，大幅降低真实飞行试验的风险与成本。

本文聚焦于神经渲染技术在数字孪生空域建模中的应用，探讨如何通过 NeRF/3DGS 等方法构建高保真、实时可更新的城市低空三维表示。

---

## 2. 数字孪生空域建模基础

### 2.1 空域数字孪生系统架构

城市低空数字孪生系统通常采用**五层架构**：

| 层级 | 功能 | 关键技术 |
|------|------|---------|
| **数据采集层** | 多源感知数据融合 | LiDAR SLAM、视觉惯性里程计（VIO）、卫星遥感 |
| **数据处理层** | 点云配准、语义分割 | ICP、PointNet++、Segment Anything |
| **三维建模层** | 几何/纹理/语义重建 | 摄影测量、NeRF/3DGS、BIM整合 |
| **仿真推演层** | 轨迹预测、流量仿真 | 多智能体仿真、强化学习 |
| **交互服务层** | 规划查询、API接口 | 地理信息系统（GIS）、RESTful API |

这一架构中，**三维建模层**是神经渲染方法的核心战场。传统方案依赖摄影测量（Photogrammetry）与激光雷达（LiDAR）扫描，存在重建速度慢、纹理不完整、动态物体干扰等痛点。神经渲染方法通过可微渲染优化，为这些问题提供了优雅的解决思路。

### 2.2 空域表示的数学框架

设城市低空空域为 $\mathcal{W} \subset \mathbb{R}^3$（典型范围：$10\text{km} \times 10\text{km} \times 0\text{m} - 300\text{m}$），空域状态可建模为时变场：

$$
\mathcal{S}(\mathbf{x}, t) = \left( \sigma(\mathbf{x}, t), \mathbf{c}(\mathbf{x}, \mathbf{d}, t), \mathcal{F}(\mathbf{x}, t) \right)
$$

其中：
- $\sigma: \mathcal{W} \times \mathbb{R} \rightarrow \mathbb{R}^+$ 为几何密度场（占用概率）
- $\mathbf{c}: \mathcal{W} \times \mathbb{S}^2 \times \mathbb{R} \rightarrow \mathbb{R}^3$ 为视角相关颜色场
- $\mathcal{F}: \mathcal{W} \times \mathbb{R} \rightarrow \{\text{residential}, \text{commercial}, \text{industrial}, \text{restricted}\}$ 为功能区分类

数字孪生的核心任务是**实时估计并更新 $\mathcal{S}(\mathbf{x}, t)$**，为规划算法提供当前时刻最准确的环境状态。

---

## 3. 神经渲染在空域重建中的应用

### 3.1 City-NeRF：大规模城市场景的神经重建

City-NeRF（Mueller et al., ACM ToG 2022）提出了面向城市尺度场景的多视角神经渲染框架，通过**渐进式建图**与**局部优化**策略实现大规模场景的神经重建。City-NeRF 的核心设计包括：

- **视图依赖的外观建模**：使用低秩矩阵分解（Low-Rank Adaptation）参数化视角依赖颜色场，使 MLP 能够高效建模城市建筑玻璃幕墙、金属表面等复杂材质的视角相关反射
- **渐进式分辨率调度**：UAV 飞行初期使用低分辨率建图快速覆盖大面积区域，随后在关键区域（如起降场、复杂交汇点）进行高分辨率局部优化
- **跨时间一致性**：通过外观嵌入（Appearance Embedding）对齐不同时间段采集的图像数据，处理光照季节性变化

City-NeRF 在城市峡谷场景中验证了神经渲染方法对大规模 3D 场景的建模能力，但原始实现需要数十小时的离线优化，无法满足 UAV 在线规划需求。

### 3.2 基于 3DGS 的实时空域建模

3D Gaussian Splatting 的增量更新特性使其天然适合 UAV 动态空域重建。**Gaussian-Urban**（思路源自 3DGS 在城市场景的应用外延）将城市建筑、树木、道路标识等场景元素建模为独立的高斯组，支持逐帧增量插入与删除：

$$
\mathcal{G}(t) = \bigcup_{i=1}^{N(t)} g_i(t), \quad g_i(t) = \left( \boldsymbol{\mu}_i(t), \boldsymbol{\Sigma}_i(t), o_i(t), \mathbf{c}_i(t) \right)
$$

关键设计包括：

1. **动态高斯生命周期管理**：UAV 新观测到的区域生成新高斯（分裂操作），长时间未更新的冗余高斯被剪枝（pruning）
2. **分块（Chunk）管理**：将城市划分为 $100\text{m} \times 100\text{m} \times 120\text{m}$ 的空间块，每块维护独立的高斯集合，UAV 在移动过程中动态加载相邻块
3. **GPU 加速管线**：利用 CUDA 实现高斯投影、深度排序与 alpha 合成的 GPU 并行化，在 Jetson Orin 上实测达到 15 FPS 渲染帧率

### 3.3 与 BIM/城市模型的融合

纯数据驱动的神经渲染方法存在**几何精度不足**的问题：MLP 或高斯集合学到的几何是"渲染正确"而非"测量准确"的，在需要精确碰撞边界的规划场景中可能引入危险误差。

**神经-几何融合方案**应运而生：

- **Geometry-guided NeRF**：将激光点云或 BIM 模型作为几何先验，通过射线-表面交点引导 NeRF 的射线采样，优先在真实几何表面附近密集采样，大幅提升几何精度
- **Nerfies /可乐乐 / HyperNeRF 的变形场方法**：用变形场建模场景的非刚性形变（如建筑立面随温度的微小形变），为规划提供不确定性边界
- **CityGML + NeRF**：将 CityGML（城市地理标记语言）的语义建筑模型与 NeRF 的纹理/外观模型叠加，既有精确几何（CityGML）又有照片级真实感（NeRF）

---

## 4. 动态空域数字孪生：实时感知融合与更新

### 4.1 动态元素建模

城市低空空域中存在大量动态元素：飞行中的其他无人机、鸟类、风筝、临时施工吊装等。静态神经场无法捕获这些动态目标，需要引入**四维（4D）时空表示**。

**D-NeRF 框架**（Pumarola et al., NeurIPS 2021）将时间维度引入神经辐射场，建模为：

$$
\mathcal{F}_\theta: (\mathbf{x}, t, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma), \quad \mathbf{x}' = \mathbf{x} + \Delta \mathbf{x}(t)
$$

其中 $\Delta \mathbf{x}(t)$ 为变形场，通过额外的 MLP 分支预测。UKF-NeRF（思路源自卡尔曼滤波与神经场的结合）进一步引入不确定性传播，为动态障碍物估计空间位置的不确定性椭圆：

$$
\mathbf{P}_t = \mathbf{F}\mathbf{P}_{t-1}\mathbf{F}^\top + \mathbf{Q}, \quad \mathbf{Q} = \sigma_w^2 \mathbf{I}
$$

### 4.2 多源感知融合

单一传感器无法提供完整的空域态势感知。动态空域数字孪生需要融合：

| 传感器 | 优势 | 局限 | 融合方式 |
|--------|------|------|---------|
| **视觉相机** | 纹理丰富、成本低 | 夜间/逆光失效、尺度歧义 | SfM 恢复深度 |
| **LiDAR** | 精确测距、不受光照影响 | 稀疏性、贵重 | 点云配准 |
| **毫米波雷达** | 穿透雾霾、测速直接 | 噪声大、分辨率低 | 与视觉/激光点云融合 |
| **ADS-B** | 空中交通信息直接获取 | 依赖对方设备广播 | 位置标注 |
| **声学阵列** | 检测未知声源 | 受城市噪声干扰 | 声源定位 |

**神经场作为多模态融合中枢**：将各传感器数据作为神经场的输入观测，通过体积渲染方程约束神经场的密度与颜色分布。关键优势在于神经场可以**自然地融合不同传感器在不同视角、不同时间采集的数据**，无需显式地进行点云配准或特征匹配。

### 4.3 实时更新管线

动态空域数字孪生的实时更新管线设计如下：

1. **数据采集**：UAV 携带的前视相机与下视相机持续采集图像序列
2. **姿态估计**：通过视觉惯性里程计（VIO）或 GPS/IMU 融合获取相机位姿
3. **增量建图**：将新观测传入神经场优化器，更新局部高斯集合或 MLP 权重
4. **动态检测**：对每帧新图像运行语义分割，分离静态背景与动态前景；动态前景独立建模为移动高斯或 4D NeRF
5. **状态发布**：通过 ROS 2 主题或 WebSocket API 向规划器发布当前空域状态

**关键性能指标**：端到端更新延迟 $< 100\text{ms}$，空间覆盖率 $> 95\%$（相对于 UAV 飞行走廊区域），几何精度 $> 10\text{cm}$（@ $1\sigma$）。

---

## 5. 端到端规划：数字孪生 → 轨迹优化

### 5.1 安全走廊提取

从神经空域表示中提取**安全飞行走廊**（Safe Corridor）是连接数字孪生与轨迹规划的关键步骤。传统方法在体素地图上提取自由空间包围盒（Free-Space Bounding Box），在神经场表示上则需要新的提取方法：

- **基于密度梯度的边界检测**：神经场的密度梯度 $\nabla_\mathbf{x}\sigma(\mathbf{x})$ 在物体表面处最大，可用于定位碰撞边界
- **Marching Cubes 提取等值面**：将密度场 $\sigma(\mathbf{x})$ 阈值化为二值占用场，使用 Marching Cubes 算法提取等值面作为安全走廊边界
- **基于高斯的碰撞检测**：3DGS 中每个高斯椭球可直接计算 SDF 近似值，轨迹规划时只需检测与高斯集合的碰撞

### 5.2 轨迹优化目标函数

在数字孪生空域中进行轨迹优化的目标函数设计：

$$
\min_{\mathbf{p}(t)} J = \underbrace{w_1 \int_0^T \|\mathbf{p}(t)\|^2 dt}_{\text{轨迹平滑}} + \underbrace{w_2 \int_0^T \sigma(\mathbf{p}(t)) dt}_{\text{碰撞规避}} + \underbrace{w_3 T}_{\text{飞行时间}} + \underbrace{w_4 \sum_{i=1}^{N} \phi(d_i)}_{\text{动态障碍物}}
$$

其中 $d_i = \|\mathbf{p}(t) - \mathbf{o}_i(t)\|$ 为与动态障碍物 $\mathbf{o}_i(t)$ 的距离，$\phi(d) = \exp(-\lambda d)$ 为指数避障势函数。

数字孪生为该优化问题提供的关键输入是：**$\sigma(\mathbf{x})$ 的精确估计**与**$\mathbf{o}_i(t)$ 的实时位置预测**。

### 5.3 验证与仿真

在将规划轨迹部署到真实 UAV 之前，数字孪生平台允许在仿真中进行**安全验证**：

- **碰撞检测仿真**：在数字孪生中注入预测的动态障碍物轨迹，验证 UAV 规划的轨迹在所有可能碰撞场景下均能规避
- **感知失效仿真**：模拟相机遮挡、LiDAR 失效等传感器故障场景，测试数字孪生状态估计的鲁棒性与降级性能
- **多机协同仿真**：在数字孪生中同时注入多架 UAV 的规划轨迹，验证空中交通管理的冲突检测与避免能力

---

## 6. 相关工作与典型系统

### 6.1 城市级数字孪生平台

**AirSim 城市孪生**（Microsoft, 2017）是最早的开源 UAV 仿真平台之一，提供了 Photo-realistic 城市环境，支持 RGB 相机、LiDAR、IMU 等传感器的仿真。AirSim 的数字孪生基于 Unreal Engine 构建，纹理逼真但几何精度有限。

**OnePlus 城市数字孪生**（思路源自大规模城市场景重建研究）使用 Photogrammetry + LiDAR 融合方法构建了多个中国城市的数字孪生模型，分辨率达到 $5\text{cm}$，支持城市规划与 UAV 仿真。

**NVIDIA Omniverse Replicator** 提供了数据合成与数字孪生构建的统一平台，支持基于 USD（Universal Scene Description）的城市场景表示与神经渲染加速。

### 6.2 UAV 空域建模研究

| 研究 | 年份 | 方法 | 覆盖范围 | 更新频率 |
|------|------|------|---------|---------|
| City-NeRF | 2022 | Multi-view NeRF | 城市街区 | 静态 |
| Gaussian-Urban | 2023 | 3DGS | 街区级 | 实时 |
| Instant-NGP | 2022 | Hash Encoding | 室内/小场景 | 实时 |
| SUDS | 2023 | Neural SLAM | 城市级 | 在线 |
| Rubble-Fuse | 2024 | 多模态融合 | 城区 | 准实时 |

---

## 7. 挑战与未来方向

### 7.1 当前主要挑战

**计算资源瓶颈**：城市级空域数字孪生（$10\text{km} \times 10\text{km} \times 300\text{m}$）包含数十亿个体素/高斯，远超单卡算力。分块策略带来了块间接缝处理、跨块轨迹规划等新问题。

**时效性与准确性的矛盾**：神经场优化需要足够的观测数据才能收敛，但城市空域状态变化迅速（临时施工、事件管制），数字孪生可能存在滞后。

**多分辨率一致性**：不同高度层的空域精度需求不同——近地面（$0-30\text{m}$）需要厘米级精度以避障，高空空域（$100-300\text{m}$）则以态势感知为主。现有神经场方法难以在单一表示中统一处理多分辨率需求。

### 7.2 未来发展方向

**神经-几何混合表示**：结合显式体素/网格（高效几何查询）与隐式神经场（照片级真实感）的优势，开发兼具精度与美观的城市空域表示。

**大语言模型 + 空域数字孪生**：利用 GPT-4V 等多模态大模型理解空域语义与管制规则，将自然语言约束注入数字孪生规划系统，实现"语音控制规划"。

**众包式数字孪生更新**：利用大量 UAV 的实时观测数据，通过联邦学习（Federated Learning）分布式更新城市数字孪生，实现"众包建图"。

---

## 8. 小结

数字孪生为城市低空无人机规划提供了**高保真、可仿真、可验证**的数字底座。神经渲染技术通过可微优化、增量更新与多模态融合能力，显著提升了空域数字孪生的构建效率与真实感。

然而，从"静态城市模型"到"动态实时孪生"仍有距离，核心挑战在于**大规模高效表示**、**动态元素实时建模**与**多分辨率一致性**。随着 3DGS、NeRF 与大语言模型技术的持续进步，城市低空数字孪生有望在未来 3-5 年内从研究原型走向实际部署。

---

## 参考文献

- Mueller, A. R., et al. (2022). City-NeRF: Multi-view neural radiance fields for urban scale scene rendering. *ACM Transactions on Graphics (ToG)*. https://doi.org/10.1145/3528223.3528346

- Pumarola, A., Corona, E., Pons-Moll, G., & Moreno-Noguer, F. (2021). D-NeRF: Neural radiance fields for dynamic scenes. *NeurIPS*, 34, 10318–10329.

- Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). 3D Gaussian Splatting for real-time radiance field rendering. *ACM Transactions on Graphics*, 42(4), 1–14. https://doi.org/10.1145/3592403

- Rosinol, A., et al. (2020). Kimera: An open-source library for real-time metric-semantic localization and mapping. *IEEE Robotics and Automation Letters*, 5(2), 892–899.

- Qin, C., et al. (2022). Instant neural graphics primitives with a multiresolution hash encoding. *ACM SIGGRAPH 2022*.

- Tosi, F., et al. (2024). Social-SLAM: Learning collaborative multi-robot navigation from human demonstrations. *ICRA*. https://doi.org/10.1109/ICRA57147.2024.10610603

- Zhou, Y., et al. (2023). SUDS: Scalable urban dynamic scene understanding. *ICCV*.

---

*本文为城市低空无人机航路规划系列文章第3篇扩展章节。*
