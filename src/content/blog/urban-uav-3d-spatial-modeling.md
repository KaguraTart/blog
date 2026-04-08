---
title: '城市低空无人机航路规划：三维空间建模'
tags: ['UAV', '路径规划', '城市空域']
category: Tech
pubDate: '2026-04-08'
---

## 引言

城市低空无人机航路规划是实现安全、高效城市空中交通（UAM, Urban Air Mobility）的核心基础技术之一。与郊区开阔地带不同，城区环境具有三维几何结构复杂、GNSS信号严重衰减、流场受建筑群强烈扰动等鲜明特征，对空间建模方法提出了更高要求。本文聚焦城市低空无人机航路规划系列的第一部分——三维空间建模，依次深入讨论三维占用栅格（3D Occupancy Grid）与八叉树（Octree）表示、城市峡谷效应（Urban Canyon）的物理建模，以及借鉴传统航空管制的空域分层模型，并辅以工程实现的对比分析。

## 1. 三维占用栅格与八叉树表示

### 1.1 从二维到三维：数学定义

经典占用栅格（Occupancy Grid）由 Moravec 和 Elfes（1985）提出，其核心思想是将连续空间离散化为有限栅格，用概率值编码每个栅格的占据状态。在二维情形下，空间被划分为边长为 $\Delta$ 的正方形单元，每个单元 $m_i$ 的占据概率记为 $P(m_i | Z_{1:t})$，其中 $Z_{1:t}$ 为截至时刻 $t$ 的全部传感器观测。传感器更新遵循贝叶斯递归公式：

$$
P(m_i | Z_t, Z_{1:t-1}) = \frac{P(Z_t | m_i, Z_{1:t-1}) P(m_i | Z_{1:t-1})}{P(Z_t | Z_{1:t-1})}
$$

实际工程中，为避免数值下溢并简化计算，通常采用**对数赔率（log-odds）**表示：

$$
l(m_i) = \log \frac{P(m_i)}{1 - P(m_i)}
$$

每次传感器测量后，加法更新规则为：

$$
l(m_i)_{\text{new}} = l(m_i)_{\text{old}} + \Delta l
$$

其中 $\Delta l$ 由传感器模型决定（占据时为正值，空闲时为负值）。该方法将乘法转化为加法，大幅提升实时性。

三维占用栅格将上述定义从平面扩展至体积空间 $\mathbb{R}^3$，将空间划分为边长为 $\Delta$ 的立方体单元（体素，voxel）。设 $V_i \subset \mathbb{R}^3$ 表示第 $i$ 个体素，则其占据概率为 $P(v_i | Z_{1:t})$。三维栅格的直接存储复杂度为 $O(N^3)$（$N$ 为单维栅格数），在典型城市场景下不可接受——例如，以 $0.1\,\text{m}$ 分辨率覆盖 $1\,\text{km}^3$ 城区的体素总数高达 $10^{13}$ 量级。

### 1.2 八叉树：自适应分辨率的空间索引

**八叉树（Octree）** 是应对上述存储挑战的标准方案。Hornung 等（2013）提出的 OctoMap 库是这一方法在机器人领域的里程碑式实现。八叉树的空间划分逻辑如下：根节点覆盖整个三维空间，每个内部节点递归细分为 8 个等体积的子节点（对应三维空间的 $2 \times 2 \times 2$ 分割），直到达到预设的最大深度 $d_{\max}$ 或最小体素尺寸 $\Delta_{\min}$。

设根节点边长为 $L_0$，深度 $d$ 处的体素边长为：

$$
L_d = \frac{L_0}{2^d}
$$

深度 $d$ 的最大节点数为 $8^d$，但由于八叉树仅对**被占据或被观测的空间**进行分裂，未知/空闲区域可用单一节点表示，因此实际节点数远小于完整栅格。OctoMap 的存储模型进一步采用**概率八叉树（Probabilistic OcTree）**：每个节点存储一个占据概率值 $P(n)$，通过贝叶斯更新持续修正。空闲节点的概率为 $P_{\text{occ}}$，占据节点概率接近 $1$，未知区域对应节点不存在于树中（隐式编码）。

Hornung 等（2013）的实验表明，在典型室内环境中，OctoMap 的内存消耗约为同分辨率稠密三维栅格的 **1/50**，同时支持动态更新和任意分辨率查询。

### 1.3 八叉树与多粒度感知

Zeng 等（2020）在 Multimedia Tools and Applications 上提出了基于八叉树占用栅格的多粒度环境感知算法，指出点云模型虽然信息丰富，但在路径规划中存在大量冗余。他们利用八叉树对不同传感器（RGB-D、LiDAR 等）数据进行统一概率表示，在树叶节点层级保留高分辨率几何信息，在粗节点层级提供低分辨率的全局结构感知。这一思路对城市级大规模地图构建尤为重要——近处需要厘米级精度避障，远处需要百米级宏观路径决策。

Thomas 等（2021）在 arXiv 论文（arXiv: 2108.10585）中进一步提出**时空占用栅格地图（Spatiotemporal Occupancy Grid Maps, SOGM）**，将动态障碍物的时间预测嵌入栅格表示，为城市环境中运动的人和车辆提供了有效的占据预测能力，对实时避障规划具有重要价值。

## 2. 城市峡谷效应：物理建模与导航挑战

城市峡谷（Urban Canyon）是指高楼林立、街道狭窄的城市微观地貌形态，是低空无人机面临最具挑战性的运行环境之一。其物理效应可从三个维度理解。

### 2.1 GNSS 信号衰减与多径效应

在城市峡谷中，密集的高层建筑形成"峡谷"结构，GNSS 卫星信号面临两类严重干扰：

- **非视距传播（NLOS）**：直射信号被建筑遮挡，无人机仅能接收到经墙面反射或衍射的信号，导致伪距测量值系统性偏大；
- **多径效应（Multipath）**：多个反射路径的信号叠加，引起载波相位解算错误和定位抖动。

UrbanNav 数据集（Wen et al., 2021; GitHub: IPNL-POLYU/UrbanNavDataset）在东京和香港实测了低成本传感器在城市峡谷中的定位性能，结果显示在深峡谷区域，单点定位（SPP）误差可达数十米。即使采用双频 GNSS 接收机，若不进行 NLOS 检测与排除，水平定位精度仍难以满足无人机悬停精度的亚米级要求。城市峡谷的**高宽比（Aspect Ratio, AR = 建筑高度 / 街道宽度）**是影响 GNSS 精度的主导因素——AR 越大，信号可用性越低。

### 2.2 湍流与风场扰动

城市峡谷内的流体动力学呈现高度非均匀性。Rotach（1995）在 *Boundary-Layer Meteorology* 上的经典研究量化了峡谷内部的湍流统计轮廓，指出街道峡谷内湍流动能（TKE）比开阔郊区高出 **2-5 倍**，且垂直速度分量的标准差 $\sigma_w$ 在近地面可达平均风速的 $0.3$–$0.6$ 倍。关键物理机制包括：

- **建筑尾流（Building Wake）**：气流绕过建筑物后在背风面形成周期性脱落涡（Kármán 涡街），产生显著的非定常升力和侧向力；
- **峡谷循环（Street Canyon Circulation）**：当来流与峡谷轴线正交时，街道内部形成方向相反的双涡环结构，净垂直风速分量在此区域内被显著放大；
- **惯性子区（Inertial Subrange）**：湍流能在惯性子区的能谱遵循 $-5/3$ 法则（Kolmogorov 定律），小尺度湍流对无人机姿态控制带宽构成持续扰动。

对无人机控制设计而言，湍流强度的特征频率范围至关重要。城市峡谷中 $1$–$10\,\text{Hz}$ 频段的扰动最为显著，要求飞控系统的姿态环带宽不低于 $20\,\text{Hz}$，这在嵌入式平台上实现起来并非易事。

### 2.3 Bernoulli 风加速效应

在狭窄街道中，Bernoulli 效应不可忽略。当气流被迫通过横截面积减小的通道时，根据连续性方程 $A_1 v_1 = A_2 v_2$，风速在局部区域显著增加。城市峡谷中建筑间距最窄处的风速可达开阔处的 **1.5–3 倍**。此外，建筑立面间的"狭管效应（Venturi Effect）"还会在局部产生指向街道中心的吸力，对无人机的侧向稳定性构成威胁。

在实际规划中，建议将城市峡谷中的**等效风扰动**建模为均值风 $\bar{u}$ 叠加随机湍流分量 $\tilde{u}$：

$$
u_{\text{eff}}(t) = \bar{u} + \sigma_u \cdot \xi(t)
$$

其中 $\xi(t)$ 为服从标准正态分布的高斯白噪声，$\sigma_u$ 根据峡谷高宽比和局部街道几何从经验公式中确定。

## 3. 空域分层模型

### 3.1 传统航空管制的启示

传统民航空管系统采用高度分层（Altitude Layer）管理已有数十年历史：以 $1000\,\text{ft}$（约 $300\,\text{m}$）为基本高度层间隔，将 $29000\,\text{ft}$ 以下的空域划分为多个管制扇区，每层服务于不同类型和速度的航空器。UAM 语境下，城市低空无人机需要在 **$0$–$120\,\text{m}$**（约 $0$–$400\,\text{ft}$）的垂直范围内与地面行人、建筑、直升机起降坪、传统通用航空器共存，分层设计因此成为必然。

NASA 的 UTM（UAS Traffic Management）项目研究（2016-2024）和 FAA 的 UAM ConOps V2.0（2023）均指出：分层管理是避免大规模无人机冲突的核心手段。在城市场景下借鉴这一思想，可设计如下三层方案。

### 3.2 城市场景高度层划分方案

| 高度层 | 垂直范围 | 主要功能 | 飞行器类型 | 典型速度 |
|--------|----------|----------|-----------|---------|
| **G 层** | 地面 $\sim 30\,\text{m}$ | 人行道上快递配送、机器人配送 | 微小型多旋翼 | $0$–$5\,\text{m/s}$ |
| **L 层** | $30$–$80\,\text{m}$ | 社区物流、城市航拍、低层穿梭 | 小型多旋翼/复合翼 | $5$–$15\,\text{m/s}$ |
| **U 层** | $80$–$120\,\text{m}$ | 城际快运、应急响应、高层穿梭 | 中型 eVTOL/固定翼 | $15$–$30\,\text{m/s}$ |

> 注：具体高度边界需根据各地空域管理法规（我国依据《无人驾驶航空器飞行管理暂行条例》2023）及城市规划进行调整。

该分层方案的设计原则如下：

1. **功能隔离**：G 层侧重末端配送的安全性（避免与人直接冲突），L 层为城市主流应用层，U 层接近传统通用航空高度以兼容过渡；
2. **流分离**：上下行方向在同一高度层内进一步水平分离，参考航空管制的五边进近逻辑设计单向航路环；
3. **动态调整**：分层边界可根据实时流量密度动态平移，FAA 的 xTM（extensible Traffic Management）框架已为此提供了标准化接口。

### 3.3 分层与三维栅格地图的融合

高度分层模型需要与三维占用栅格深度整合：在规划阶段，依据分层边界对八叉树地图进行**高度层掩膜（Layer Masking）**，仅在当前任务所在层及相邻层的可飞行体素中搜索路径；在动态重新规划时，若某层出现拥堵，可自动切换至相邻层绕行。这一机制在 NASA 的 UTM 走廊（Corridor）概念中已有初步验证。

## 4. Octree / PCL 点云 / 体素的工程权衡

在工程实践中，选择何种三维表示方法需要在精度、内存、计算速度和更新频率之间做出权衡。以下为系统化对比。

| 指标 | 稠密三维栅格 | 八叉树（Octree） | 原始点云（PCL） | 哈希体素（Hash Voxel） |
|------|------------|----------------|--------------|----------------------|
| **内存效率** | 低（固定 $O(N^3)$） | 高（自适应分裂） | 中（仅存点，无拓扑） | 高（稀疏哈希索引） |
| **查询复杂度** | $O(1)$ | $O(\log N)$ | $O(N)$（穷举）或 $O(\log N)$（配合 kd-tree） | $O(1)$ 均值 |
| **动态更新** | 慢（全量重建） | 快（增量节点分裂） | 快（追加点） | 快（哈希插入） |
| **分辨率一致性** | 全局一致 | 层次自适应 | 无栅格结构 | 全局一致 |
| **碰撞检测** | 快（数组索引） | 中（树搜索） | 慢（点-模型检测） | 快（哈希查找） |
| **工程生态** | ROS nav_msgs | OctoMap / PCL Octree | PCL / Open3D | OctoMap（可配置） |
| **适用场景** | 小范围高精度 | 大范围多分辨率 | 实时感知/建图 | 稀疏大规模场景 |

**Octree 的核心优势**在于其**自适应分辨率 + 概率表示**的双重特性：它既是空间索引结构，又是概率更新框架，特别适合城市场景下"近处障碍精确、远处障碍粗略"的感知需求。OctoMap 库（Hornung et al., 2013; DOI: 10.1007/s10514-012-9321-0）在 GitHub 上的活跃度和学术引用量（据 Google Scholar 统计超过 5000 次）均证明了其工程成熟度。

**点云的优势**在于无损保留原始传感器数据，适合基于深度学习的感知算法（3D 目标检测、语义分割等）作为输入。PCL（Point Cloud Library）库和 Open3D 库提供了成熟的点云处理工具链，但点云本身不编码占据/空闲的语义信息，需额外步骤转换为可飞行区域。

**哈希体素**（如 OctoMap 的 `OcTree Key` 哈希索引方案）在需要极快查询速度且场景稀疏的场景下表现优异，内存开销接近八叉树但查询更高效，是近年前沿研究的热点方向。

在实际城市场景中，**推荐方案**是以 OctoMap 的概率八叉树为底层存储，以原始点云为感知输入，通过增量更新机制持续修正占据概率，同时利用哈希索引加速最近邻查询。这一组合在 LIO-SAM 等先进 SLAM 系统中已被验证可在城市峡谷中实现鲁棒的实时建图（参见 LIO-SAM-6AXIS-UrbanNav 适配版本）。

## 5. 总结与展望

本文系统梳理了城市低空无人机航路规划中三维空间建模的核心要素：

- **三维占用栅格与八叉树**提供了从概率论出发的统一环境表示框架，OctoMap 作为开源实现已在学术界和工业界获得广泛验证；
- **城市峡谷效应**从 GNSS 衰减、湍流统计和 Bernoulli 风加速三个物理维度对无人机规划系统提出约束，需要在航路规划中显式建模；
- **空域分层模型**借鉴传统航空管制思想，在城市场景下将 $0$–$120\,\text{m}$ 垂直空域划分为 G/L/U 三层，为大规模无人机流量管理提供了结构性框架；
- 工程选型应在内存效率、查询速度和动态更新能力之间综合权衡，OctoMap + 点云的组合是当前主流技术路线。

后续章节将逐步深入**路径规划算法**（RRT*/BIT* 等采样类算法在三维八叉树地图中的应用）、**实时轨迹优化**（城市峡谷风扰动下的模型预测控制）以及**多机协同避障**等议题，构建完整的城市低空航路规划技术体系。

---

## 参考文献

- Hornung, A., Wurm, K. M., Bennewitz, M., Stachniss, C., & Burgard, W. (2013). OctoMap: An efficient probabilistic 3D mapping framework based on octrees. *Autonomous Robots*, 34(3), 189–206. https://doi.org/10.1007/s10514-012-9321-0

- Thomas, H., Farr, R., Yang, C., Chen, Y., & Leonard, J. J. (2021). Learning Spatiotemporal Occupancy Grid Maps for Lifelong Navigation in Dynamic Scenes (arXiv: 2108.10585). arXiv. https://arxiv.org/abs/2108.10585

- Wen, W., Zhang, G., & Hsu, L. T. (2021). *UrbanNav: An Open-sourcing Localization Dataset for Benchmarking Positioning Algorithms Designed for Urban Canyons* [Dataset and documentation]. GitHub repository: https://github.com/IPNL-POLYU/UrbanNavDataset

- Rotach, M. W. (1995). Profiles of turbulence statistics in and above an urban street canyon. *Atmospheric Environment*, 29(13), 1473–1486. https://doi.org/10.1016/1352-2310(95)00084-D

- Zeng, T., Si, B., & Zhao, J. (2020). Multi-granularity environment perception based on octree occupancy grid. *Multimedia Tools and Applications*, 79, 27875–27896. https://doi.org/10.1007/s11042-020-09302-w

- Moravec, H. P., & Elfes, A. (1985). High resolution maps from wide angle sonar. *Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)*, 116–121. https://doi.org/10.1109/ROBOT.1985.1087316

- U.S. Department of Transportation / Federal Aviation Administration. (2023). *Urban Air Mobility (UAM) Concept of Operations*, Version 2.0. FAA. https://www.faa.gov/air_traffic/nas_management/nas_research/models/uam_conops

- NASA Aeronautics Research Mission Directorate. (2023). *UAS Traffic Management (UTM) Project Summary*. NASA. https://utm.arc.nasa.gov/

- Hrabar, S., & Sukhatme, G. S. (2004). A comparison of two camera configurations for optic-flow based navigation through urban canyons. *Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 3943–3948. https://doi.org/10.1109/IROS.2004.1389989
