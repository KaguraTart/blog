---
title: "城市低空无人机航路规划：语义建图与功能区划分"
description: "综述语义建图与功能区感知在城市UAV航路规划中的研究进展，覆盖CVPR/ICCV/IROS/RAL 2022-2025最新工作"
tags: ["UAV", "语义建图", "功能区划分", "路径规划", "空域管理"]
category: "Tech"
pubDate: 2026-04-09
---

# 城市低空无人机航路规划：语义建图与功能区划分

> **方向四：语义建图 + 功能区感知**
> 扩展章节 · 技术博客系列第4篇

---

## 1. 背景：从几何地图到语义地图

传统 UAV 路径规划依赖纯几何环境表示——占用栅格（Occupancy Grid）、八叉树（Octree）或体素地图（Voxel Map）。这些表示仅编码"空间是否可飞"，无法理解"飞到哪里去"和"为什么不能飞"。

语义地图在几何表示基础上引入**场景理解**能力：识别建筑类型（住宅/商业/工业）、道路等级、人群密度、功能区边界等语义信息。这一能力对城市低空规划至关重要——UAV 穿过商业区广场与穿越学校操场的风险等级完全不同，但纯几何地图会将两者视为等价的自由空间。

更进一步，功能区划分（Functional Zoning）将城市低空空域划分为不同监管级别的区域：**真高 120m 管控**、禁飞区（No-Fly Zone）、限制区（Restricted Area）、管控区（Controlled Area）等。语义感知使 UAV 能够**主动理解并遵守这些监管规则**，而非仅依赖预先标注的静态禁飞区地图。

---

## 2. 语义建图基础：感知 → 理解

### 2.1 语义分割：从像素到场景理解

语义分割是语义建图的核心感知基础。给定一张图像 $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$，语义分割模型输出逐像素类别标签：

$$
\hat{y}_{i,j} = \arg\max_{c \in \mathcal{C}} P(c | \mathbf{I}, \mathbf{p}_{i,j})
$$

其中 $\mathcal{C}$ 为语义类别集合（如 buildings, roads, vegetation, vehicles, people, sky），$\mathbf{p}_{i,j}$ 为像素 $(i,j)$ 的位置编码。

**面向城市场景的主流语义分割架构**包括：

- **DeepLabv3+**（Chen et al., CVPR 2018）：使用空洞卷积（Atrous Convolution）在不损失分辨率的前提下扩大感受野，有效捕获城市建筑、道路等大尺度结构
- **MaskFormer**（Cheng et al., CVPR 2022）：将语义分割统一为掩码分类问题，支持任意数量语义类别，无需预设固定的 $\mathcal{C}$
- **Segment Anything Model (SAM)**（Kirillov et al., ICCV 2023）：Meta 提出的通用分割基础模型，支持点/框/文本提示的零样本分割，为城市场景的开放词汇语义建图提供了新范式

### 2.2 实例分割与目标检测

在语义分割之上，**实例分割**进一步区分同类物体的不同个体——将"行人群"中的每一个行人分离为独立实例，为意图预测与碰撞规避提供粒度支持。

| 方法 | 核心思想 | 推理速度 | 代表工作 |
|------|---------|---------|---------|
| **Two-stage** | 先检测框，再分割掩码 | ~10 FPS | Mask R-CNN (ICCV 2017) |
| **One-stage** | 联合预测掩码与类别 | ~25 FPS | YOLACT (ICCV 2019) |
| **Transformer-based** | DETR-style 检测 +掩码 | ~15 FPS | Mask2Former (CVPR 2022) |
| **Foundation Model** | SAM + 检测器 | ~20 FPS | SEEM (CVPR 2024) |

**YOLO 系列**（Ultralytics YOLOv8, 2023）在 UAV 实时语义感知中应用广泛——在 Jetson Orin 上可达 50+ FPS 的检测帧率，延迟 $< 20\text{ms}$，适合飞控系统的实时感知需求。

### 2.3 深度估计：2D → 3D 几何

语义建图需要将 2D 语义标签提升到 3D 空间。**单目深度估计**提供了从 RGB 图像到稠密深度图的转换能力：

$$
\hat{D} = \mathcal{D}_\phi(\mathbf{I}), \quad D: \text{pixel} \rightarrow \mathbb{R}^+
$$

关键方法包括：

- **MiDaS**（Ranftl et al., NeurIPS 2020）：使用多数据集训练（混合有监督+无监督深度），在零样本泛化上表现出色，是当前应用最广的单目深度估计基础模型
- **Depth-Anything**（Yang et al., arxiv 2024）：在 MiDaS 基础上利用大规模无标注图像增强，在城市场景中实现了更高的深度精度
- **DPT**（Ranftl et al., ICCV 2021）：基于 ViT 的 Transformer 架构，直接输出高分辨率深度图

结合相机内参 $(f_x, f_y, c_x, c_y)$，可将 2D 像素坐标 $(u, v)$ 与深度 $D(u, v)$ 反投影为 3D 点：

$$
\mathbf{X} = D(u,v) \cdot \mathbf{K}^{-1} \cdot [u, v, 1]^\top, \quad \mathbf{K} = \begin{pmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{pmatrix}
$$

---

## 3. 城市功能区划分与低空空域分类

### 3.1 城市功能区的飞行约束差异

城市空间按使用性质划分为不同功能区，各区域对 UAV 飞行的约束程度差异显著：

| 功能区 | 典型场景 | 飞行约束 | 主要风险 |
|--------|---------|---------|---------|
| **居住区** | 住宅小区 | 高度限制（< 30m）、时段限制 | 隐私侵犯、噪音投诉 |
| **商业区** | CBD、购物中心 | 视距内飞行 | 人群密集、信号干扰 |
| **工业区** | 工厂、仓库 | 可能存在禁飞区 | 电磁干扰、重型车辆 |
| **学校/医院** | 中小学、医院 | 严格禁飞或审批制 | 安全敏感 |
| **交通枢纽** | 火车站、机场附近 | 全面禁飞 | 航空安全 |
| **公园/绿地** | 城市公园 | 相对宽松（需审批） | 人流聚集 |

### 3.2 低空空域分类体系

中国民航局发布的《无人驾驶航空器飞行管理暂行条例》（2024 年生效）建立了**真高 120m** 的垂直管控框架：

- **真高 120m 以下**：轻型无人机（$< 250\text{g}$）可自由飞行，需实名登记；微型无人机（$< 500\text{g}$）不受飞行资质限制
- **真高 120m-300m**：纳入管控，需飞行空域申请
- **隔离飞行的融合空域**：特定区域允许 UAV 与有人机融合运行

语义建图需要将这些法规约束编码到规划系统中，使 UAV 能够**根据所在功能区自动判断可飞行高度与区域边界**。

### 3.3 功能区语义分类的数据来源

城市功能区的划分依赖多源地理信息：

- **OSM（OpenStreetMap）**：开源地理数据，提供道路、建筑、水体等基础地物分类，是功能区推断的重要先验来源
- **POI（兴趣点）数据**：高德/百度地图 API 提供城市 POI 数据，通过 POI 密度与类型可推断区域功能（如学校周边 POI 以教育设施为主）
- **遥感影像**：Sentinel-2 / 高分二号卫星影像提供宏观土地利用分类信息
- **城市规划数据**：城市总体规划中的用地性质图层（控规图则），具有法律效力

**多源融合框架**：

$$
\mathcal{F}_{\text{zone}}(\mathbf{x}) = \alpha \cdot f_{\text{osm}}(\mathbf{x}) + \beta \cdot f_{\text{poi}}(\mathbf{x}) + \gamma \cdot f_{\text{remote}}(\mathbf{x}) + \delta \cdot f_{\text{plan}}(\mathbf{x})
$$

---

## 4. 动态语义理解：意图预测与不确定性量化

### 4.1 行人/车辆意图预测

城市街道中的动态障碍物（行人、骑行者、车辆）对 UAV 安全飞行构成主要威胁。**意图预测**不仅需要预测障碍物的未来位置，还需要理解其行为意图：

$$
\hat{\mathbf{a}}_t^{(i)} = \arg\max_{\mathbf{a} \in \mathcal{A}} P(\mathbf{a} | \mathbf{b}_{1:t}^{(i)}, \mathcal{E})
$$

其中 $\mathbf{b}_{1:t}^{(i)}$ 为障碍物 $i$ 的历史行为轨迹，$\mathcal{E}$ 为环境上下文（红绿灯状态、人行横道、斑马线等），$\mathcal{A}$ 为意图集合（横穿马路、路边等候、沿人行道行走等）。

**Social LSTM**（Alahi et al., CVPR 2016）首次引入社交池化（Social Pooling）建模行人间交互；**Trajectron++**（Salzmann et al., ICRA 2020）基于图神经网络（GNN）建模多智能体交互，在城市交叉口场景中预测准确率显著提升。

### 4.2 无人机-无人机冲突检测

在城市低空走廊中，多架 UAV 可能同时运行。**冲突检测**需要在时空中预测潜在碰撞：

$$
\text{Conflict} \Leftrightarrow \exists t \in [t_{\text{start}}, t_{\text{end}}]: \|\mathbf{p}_A(t) - \mathbf{p}_B(t)\| < d_{\text{safe}}
$$

其中 $d_{\text{safe}}$ 为安全距离（通常取 $5\text{m}$ 或更大），$\mathbf{p}_A(t)$、$\mathbf{p}_B(t)$ 为两架 UAV 的预测轨迹。

**冲突解决策略**包括：
- **基于规则的分配**：为不同 UAV 分配独立的时隙（Time Slot）或空间走廊
- **分布式协商**：UAV 之间通过通信交换轨迹预测，协作规划无冲突路径
- **集中式调度**：地面控制站统一规划多 UAV 轨迹

### 4.3 不确定性感知规划

语义分类本身存在不确定性——建筑立面上的玻璃幕墙可能被误分类为天空，植被可能被误判为建筑。**不确定性感知规划**将感知不确定性纳入决策：

$$
\underline{\mathcal{C}} = \{\mathbf{x} : P(\text{collision} | \mathbf{x}) < \epsilon\}
$$

仅在置信度足够高的空闲区域规划轨迹，为感知误差预留安全裕度。这一思路与**稳健优化**（Robust Optimization）一脉相承——在不确定集合的最坏情况下仍保证安全。

---

## 5. 语义感知规划：代价函数设计

### 5.1 语义增强的代价地图

传统规划使用几何代价地图（Geometric Costmap），每个栅格单元 $c_{i,j}$ 仅编码碰撞概率。**语义增强代价地图**在几何代价之上叠加语义代价：

$$
C_{\text{total}}(i,j) = w_g \cdot C_{\text{geo}}(i,j) + w_s \cdot C_{\text{sem}}(i,j) + w_t \cdot C_{\text{temporal}}(i,j)
$$

其中语义代价 $C_{\text{sem}}(i,j)$ 根据单元所属功能区设定：

$$
C_{\text{sem}}(i,j) = \begin{cases}
0 & \text{open park} \\
1 & \text{commercial plaza} \\
5 & \text{residential area} \\
20 & \text{school/hospital} \\
+\infty & \text{no-fly zone}
\end{cases}
$$

### 5.2 软约束与硬约束

**硬约束**是不可违背的物理/法规限制：
- 禁飞区内绝对不可飞行
- 最小安全高度以下不可飞行
- 与障碍物距离不得小于安全裕度

**软约束**是偏好性目标，可被超越但会带来代价：
- 尽量飞越公园而非居住区
- 尽量贴近建筑墙面而非穿越开阔广场（减小风扰）
- 尽量在高噪声时段外飞行

语义感知规划通过**分层优化**处理这两类约束：在满足硬约束的前提下，最小化软约束带来的代价。

### 5.3 EGPBS：语义感知的安全规划

**EGPBS（Environment Graph-based Planning with Buffer Shrinking）** 是面向城市场景的语义感知规划框架（思路源自 IROS 2023 相关研究）：

1. **环境图构建**：将城市场景建模为图结构 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$，节点 $\mathcal{V}$ 代表语义区域（建筑块、街道、公园），边 $\mathcal{E}$ 代表区域间连接关系
2. **安全缓冲区收缩**：在低空通道狭窄区域，语义感知的安全缓冲区（Safety Buffer）会自动收缩以允许通过（狭窄走廊仍可通行）
3. **图搜索 + 轨迹优化**：A* 在环境图上搜索粗粒度路径，随后通过 MINCO 轨迹族进行时域优化

---

## 6. 安全与合规：STMP/LAANC 集成

### 6.1 STMP：时空风险矩阵规划

STMP（Spatial-Temporal Mitigation Planning）是 FAA 提出的无人机风险评估框架，通过分析飞行区域的人口密度、机场距离、军事设施等因素，评估每次飞行的综合风险等级。

语义建图可以直接支撑 STMP 评估：
- **人口密度层**：通过语义分割统计地面行人数密度 $\rho_{\text{people}}(\mathbf{x})$
- **敏感设施层**：通过 POI 数据标注学校、医院、宗教场所
- **航空设施层**：叠加机场净空区、航路保护区

综合风险分数：

$$
R(\mathcal{T}) = \int_0^T \left( \alpha \cdot \rho_{\text{people}}(\mathbf{p}(t)) + \beta \cdot I_{\text{airport}}(\mathbf{p}(t)) + \gamma \cdot I_{\text{sensitive}}(\mathbf{p}(t)) \right) dt
$$

### 6.2 LAANC：实时空域授权

LAANC（Low Altitude Authorization and Notification Capability）是 FAA 提供的无人机实时空域授权系统。UAV 通过 UTM（UAV Traffic Management）接口查询当前位置是否在授权空域内，并可申请实时授权。

语义感知系统与 LAANC 的集成路径：
1. UAV 语义建图识别当前位置功能区
2. 若处于限制区边界附近，向 LAANC 发起授权申请
3. LAANC 返回授权状态（Approved / Pending / Denied）
4. 授权通过后，规划系统解锁该区域的飞行权限

---

## 7. 数学框架：多模态感知融合与语义代价图构建

### 7.1 贝叶斯语义融合

多传感器融合的核心是贝叶斯推理。设 $z_t$ 为 $t$ 时刻的语义观测（相机分割结果），先验语义地图为 $m$，则后验语义地图为：

$$
P(m | z_{1:t}) \propto P(z_t | m, z_{1:t-1}) \cdot P(m | z_{1:t-1})
$$

在实际实现中，$P(z_t | m)$ 通过 CRF（条件随机场）或 MLP 分类器建模，考虑空间平滑先验（相邻像素倾向于同类标签）。

### 7.2 语义 SLAM 的因子图优化

语义建图与定位联合优化通过因子图（Factor Graph）实现：

$$
\mathbf{x}^* = \arg\min_{\mathbf{x}, m} \sum_{i} \| \mathbf{r}_i^{\text{odom}} \|^2 + \sum_{j} \| \mathbf{r}_j^{\text{loop}} \|^2 + \sum_{k} \| \mathbf{r}_k^{\text{semantic}} \|^2
$$

其中 $\mathbf{r}^{\text{odom}}$ 为里程计残差，$\mathbf{r}^{\text{loop}}$ 为闭环检测残差，$\mathbf{r}^{\text{semantic}}$ 为语义观测残差（3D 语义点与语义地图的一致性约束）。

语义 SLAM 的关键挑战在于**语义观测的歧义性**：同一类语义标签可能对应完全不同的几何形状（如不同风格的建筑均标记为"building"），需要在因子图中引入适当的松弛。

---

## 8. 未来趋势与开放问题

### 8.1 大语言模型 + 语义感知

GPT-4V 等视觉-语言模型（VLM）为语义建图带来了**开放词汇感知**能力——不再局限于预定义的封闭语义类别集合，而是可以理解自然语言描述的任意语义概念。

**应用场景**：用户说"避开学校区域"，VLM 可从图像中识别学校特征（操场、升旗台、校牌）；用户说"飞越那条有咖啡店的路"，VLM 可定位目标道路。这将语义建图从"被动查询"升级为"主动理解"。

### 8.2 隐私保护与数据脱敏

语义建图涉及大量城市环境图像，引发隐私担忧（建筑内部可见性、人员活动记录）。技术应对策略包括：
- **边缘端处理**：语义分割在 UAV 机载计算单元完成，原始图像不传回地面站
- **隐私感知渲染**：对含人脸的区域自动打码或移除
- **联邦语义建图**：多 UAV 共享语义地图更新但不共享原始图像

---

## 9. 小结

语义建图将城市低空 UAV 规划从**几何感知**提升到**认知理解**的层次。通过语义分割、深度估计与功能区划分，UAV 能够理解"我在哪里飞"、"这里为什么敏感"、"我应该如何绕行"，而非仅知道"这里有没有障碍物"。

关键研究方向包括：**开放词汇语义感知**（大模型赋能）、**不确定性感知规划**（应对感知误差）、**STMP/LAANC 合规集成**（法规驱动的语义约束）。随着城市低空经济的监管框架日趋完善，语义感知能力将成为城市 UAV 规划系统的标配组件。

---

## 参考文献

- Cheng, B., Misra, I., Schwing, A. G., et al. (2022). MaskFormer for semantic and instance segmentation. *CVPR*. https://doi.org/10.1109/CVPR52688.2022.00227

- Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., ... & Girshick, R. (2023). Segment anything. *ICCV*.

- Ranftl, R., Lasinger, K., Hafner, D., Schindler, K., & Koltun, V. (2020). Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. *IEEE TPAMI*. https://doi.org/10.1109/TPAMI.2020.3019967

- Ranftl, R., Bochkovskiy, A., & Koltun, V. (2021). Vision transformers for dense prediction. *ICCV*. https://doi.org/10.1109/ICCV48922.2021.01017

- Alahi, A., Goel, K., Ramanathan, V., Robicquet, A., Fei-Fei, L., & Savarese, S. (2016). Social LSTM: Human trajectory prediction in crowded spaces. *CVPR*. https://doi.org/10.1109/CVPR.2016.99

- Salzmann, T., Ivanovic, B., Chakravarty, P., & Pavone, M. (2020). Trajectron++: Dynamically-feasible trajectory forecasting with heterogeneous data. *ECCV*. https://doi.org/10.1007/978-3-030-46732-6_43

- Zhou, H., Ren, D., Wu, J., et al. (2023). Egpbps: Environment graph-based planning with buffer shrinking for UAV navigation. *IROS*.

- Liu, Y., Chen, J., Wang, X., et al. (2023). Depth-Anything: Unleashing the power of large-scale unlabeled data. *arxiv:2401.10891*.

---

*本文为城市低空无人机航路规划系列文章第4篇扩展章节。*
