---
title: "城市低空无人机航路规划：高密度 CBD 场景下的理论与算法"
description: "系统性解析城市低空（300m 以下）无人机航路规划的核心挑战与解决方案，重点讨论高密度 CBD 场景下的冲突消解、4D 轨迹优化与多机协同算法，附完整数学推导"
pubDate: 2026-04-06
tags: ["UAV", "路径规划", "城市空域", "优化算法", "UTM", "MILP", "强化学习"]
category: Tech
---

# 城市低空无人机航路规划：高密度 CBD 场景下的理论与算法

> 本文系统梳理城市低空无人机航路规划的理论体系，重点面向高密度 CBD 场景，涵盖从问题建模、单机规划、多机协同到实际部署的全链路算法，提供完整的数学推导与工程权衡分析。

---

## 1. 引言：为什么 CBD 特别难？

城市低空经济正在快速崛起。从外卖配送、应急物资到城市安防，无人机的应用场景日益丰富。然而，当我们将无人机从郊区开阔地带入高密度 CBD 时，规划问题的复杂度呈现指数级上升。

### 1.1 城市低空空域定义

本文所讨论的**城市低空**特指距地面 300m 以下的空域（Above Ground Level, AGL），这是大多数城市法规规定的无人机可飞行高度上限。在这一范围内，存在三类截然不同的空域层级：

- **地面层（0–30m）**：建筑物立面干扰区，GPS 信号严重衰减，视觉特征丰富但光照条件多变
- **建筑层（30–120m）**：城市峡谷主体区域，风场受建筑物尾流效应影响显著
- **过渡层（120–300m）**：相对开阔，但仍受城市热岛效应和进出港航线约束

### 1.2 CBD 的三重核心挑战

高密度 CBD 与郊区开阔地的本质区别体现在三个维度：

**① 密集建筑群带来的三维空间约束**

摩天大楼形成垂直壁障，传统的二维路径规划算法不再适用。需要三维占用表示 + 动态安全半径调整——越靠近建筑物，无人机遭遇突发气流的风险越高，所需的安全裕量反而越大。

**② 高密度飞行需求带来的多机冲突**

一条商业走廊可能在同一时段承载数十架无人机同时起降，冲突概率随密度平方增长。传统的"先规划后执行"串行模式无法满足实时性要求。

**③ 动态障碍的不可预测性**

施工塔吊、无人机临时禁飞区、低空鸟群——CBD 的动态障碍在时间和空间上均不可预测，规划算法必须具备在线 replanning 能力。

### 1.3 与郊区规划的本质区别

| 维度 | 郊区开阔地 | 高密度 CBD |
|------|-----------|-----------|
| 空间密度 | 低（可自由选择航路） | 高（航路高度约束） |
| 障碍类型 | 静态为主 | 静态 + 动态混合 |
| 风场特性 | 均匀层流 | 湍流、城市峡谷效应 |
| 多机密度 | 低（稀疏分布） | 高（密集汇聚） |
| GPS 精度 | 良好（开阔天空） | 下降（多径效应） |
| 监管约束 | 相对宽松 | 严格分层管理 |

---

## 2. 问题建模：把飞行问题变成数学问题

### 2.1 空间建模

#### 3D 占用栅格（Occupancy Grid）

将三维空间离散化为体素（Voxel）网格，每个体素标记为占用或空闲：

$$
O(\mathbf{x}) = \begin{cases}
1 & \text{if voxel at } \mathbf{x} \text{ is occupied} \\
0 & \text{otherwise}
\end{cases}
$$

占用栅格从 GIS 数据、建筑物 CAD 模型或 SLAM 在线建图构建。在实际系统中，通常采用**八叉树（Octree）** 表示以节省存储——稀疏区域用大节点，密集区域递归细分。

#### 城市峡谷（Urban Canyon）效应

CBD 的建筑物形成峡谷地形，导致三个物理效应：

1. **风加速效应**：气流在峡谷内加速（Bernoulli 效应），风速可达开阔地的 1.5–2 倍
2. **湍流增强**：建筑物尾流产生随机湍流，导致无人机姿态扰动
3. **GPS 多径衰落**：信号在建筑物立面反射，造成定位偏差

数学上用一个**风险场** $\mathcal{R}(\mathbf{x}) \in [0,1]$ 来量化第 $i$ 个体素的风险：

$$
\mathcal{R}_i = w_1 \cdot \mathbb{I}[h_i < H_{building}] + w_2 \cdot TI_i + w_3 \cdot \text{GPS\_error}_i
$$

其中 $TI_i$ 为第 $i$ 区的湍流强度指数，$H_{building}$ 为周边建筑物高度。

#### 空域分层（Layered Airspace）模型

借鉴航空管制的成熟经验，CBD 低空空域可分层管理：

$$
\mathcal{A} = \bigcup_{k=1}^{K} \mathcal{L}_k, \quad \mathcal{L}_k = \{ \mathbf{x} : h \in [h_k^{min}, h_k^{max}] \}
$$

典型 $K=4$ 分层方案（基于飞行方向分配高度）：

| 高度层 | 高度范围 | 适用方向 |
|-------|---------|---------|
| $\mathcal{L}_1$ | 50–80m | 东向飞行 |
| $\mathcal{L}_2$ | 80–120m | 西向飞行 |
| $\mathcal{L}_3$ | 120–180m | 南向飞行 |
| $\mathcal{L}_4$ | 180–300m | 北向飞行 |

这种**方向-高度绑定**策略从源头减少了迎面冲突的概率。

### 2.2 4D 轨迹定义

无人机轨迹是一个从时间到三维空间的映射：

$$
\boldsymbol{\xi}: [t_0, t_f] \to \mathbb{R}^3, \quad \boldsymbol{\xi}(t) = (x(t), y(t), z(t))
$$

状态向量包括位置和速度：

$$
\mathbf{s}(t) = \begin{bmatrix} \boldsymbol{\xi}(t) & \dot{\boldsymbol{\xi}}(t) & \ddot{\boldsymbol{\xi}}(t) \end{bmatrix}^\top
$$

**安全分离约束**（水平 + 垂直同时满足）：

$$
\|\mathbf{p}_i(t) - \mathbf{p}_j(t)\|_{xy} \geq d_h, \quad |z_i(t) - z_j(t)| \geq d_v
$$

典型值：$d_h = 50\text{m}$（水平），$d_v = 20\text{m}$（垂直），参考 NASA UTM 标准。

### 2.3 优化问题的一般形式

单机航路规划可建模为带约束的最优控制问题：

$$
\min_{\boldsymbol{\xi}(t)} \int_{t_0}^{t_f} \mathcal{L}(\boldsymbol{\xi}(t), \dot{\boldsymbol{\xi}}(t), t) \, dt
$$

约束条件包括：

- **碰撞避免**：$O(\boldsymbol{\xi}(t)) = 0, \quad \forall t \in [t_0, t_f]$
- **动力学约束**：$\|\ddot{\boldsymbol{\xi}}(t)\| \leq a_{max}, \quad \|\dot{\boldsymbol{\xi}}(t)\| \leq v_{max}$
- **分离约束**：$\|\mathbf{p}_i(t) - \mathbf{p}_j(t)\| \geq d_{safe}, \quad \forall (i,j) \in \text{adjacent pairs}$
- **边界条件**：$\boldsymbol{\xi}(t_0) = \mathbf{p}_{start}, \quad \boldsymbol{\xi}(t_f) = \mathbf{p}_{goal}$

---

## 3. 单机路径规划算法

### 3.1 图搜索：A* 算法

A* 是离散航路点规划中最经典的算法，通过启发式搜索在状态空间中找到代价最优路径。

**空域图构建**：将连续空间离散化为节点（关键航路点）和边（合法飞行廊道）。常用方法包括：

- **Visibility Graph**：节点为起点、终点和所有多边形障碍的顶点，边为可见连线（不穿越障碍物）
- **Waypoint Graph**：预定义航路点网络，节点分布在建筑物间隔和交叉口附近

**代价评估函数**（A* 的核心）：

$$
f(n) = g(n) + h(n)
$$

其中：

- **实际代价**（从起点累计）：$g(n) = g(parent(n)) + d(parent(n), n)$
- **启发代价**（到目标的估计，从不高估）：$h(n) = \|\mathbf{p}_n - \mathbf{p}_{goal}\|_2 = \sqrt{(x_n - x_g)^2 + (y_n - y_g)^2 + (z_n - z_g)^2}$（欧氏距离，保证可采纳性）

**面向城市风险加权的边代价**：

$$
d(u, v) = \ell_{uv} \cdot \left(1 + \beta \cdot \mathcal{R}_{uv}\right)
$$

其中 $\ell_{uv}$ 是航段长度，$\mathcal{R}_{uv} \in [0,1]$ 是廊道地面风险评分（来自 2.1 节的风险场），$\beta$ 为风险权重系数（典型值 0.5–2.0）。

### 3.2 采样规划：RRT* 算法

当状态空间维度较高且障碍物形状复杂时，基于随机采样的 RRT* 算法更具优势。

**算法核心步骤**：

1. **随机采样**：在自由空间中均匀采样一个状态 $\mathbf{x}_{rand}$
2. **最近邻搜索**：$x_{nearest} = \arg\min_{x \in \mathcal{T}} \|x - x_{rand}\|_2$
3. **Steer（导向）**：以步长 $\delta$ 朝随机状态延伸：$x_{new} = x_{nearest} + \delta \cdot \frac{x_{rand} - x_{nearest}}{\|x_{rand} - x_{nearest}\|_2}$
4. **重连（Rewire）**：检查是否能以更低价廉的父节点替换现有父节点

**最近邻球半径**（渐近最优半径）：

$$
r_n = \gamma_{\text{RRT}^*} \left(\frac{\log n}{n}\right)^{1/d}
$$

其中 $n$ 为当前树节点数，$d$ 为空间维度，$\gamma_{\text{RRT}^*}$ 为与空间体积相关的常数。

**重连条件**（替换父节点当且仅当新路径更便宜）：

$$
c(x_{near}) > c(x_{new}) + d(x_{new}, x_{near})
$$

**渐近最优性保证**（RRT* 区别于普通 RRT 的关键）：

$$
\lim_{n \to \infty} c(\xi^*_n) = c^* \quad \text{(almost surely)}
$$

即随着采样点增加，RRT* 能以概率 1 收敛到全局最优解。

### 3.3 人工势场法（APF）

APF 将无人机视为在势能场中运动的粒子，引力指向目标，斥力来自障碍物。

**引力势能**（目标方向，二次势阱）：

$$
U_{att}(\mathbf{p}) = \frac{1}{2} k_{att} \|\mathbf{p} - \mathbf{p}_{goal}\|^2
$$

**斥力势能**（障碍物影响范围内激活）：

$$
U_{rep}(\mathbf{p}) = \begin{cases}
\dfrac{1}{2} k_{rep}\left(\dfrac{1}{\rho(\mathbf{p})} - \dfrac{1}{\rho_0}\right)^2 & \rho(\mathbf{p}) \leq \rho_0 \\
0 & \rho(\mathbf{p}) > \rho_0
\end{cases}
$$

其中 $\rho(\mathbf{p})$ 为到最近障碍物的距离，$\rho_0$ 为影响半径，$k_{rep}$ 为斥力增益。

**合力为势能的负梯度**：

$$
\mathbf{F}(\mathbf{p}) = -\nabla U_{att}(\mathbf{p}) - \nabla U_{rep}(\mathbf{p})
$$

显式梯度分量：

$$
\nabla U_{att} = k_{att}(\mathbf{p} - \mathbf{p}_{goal}), \quad \nabla U_{rep} = k_{rep}\left(\frac{1}{\rho} - \frac{1}{\rho_0}\right)\frac{1}{\rho^2}\nabla\rho \quad (\rho \leq \rho_0)
$$

APF 的优点是计算量极小（$O(n)$，$n$ 为障碍物数），适合实时控制；缺点是在窄通道中容易陷入局部极小——这是 CBD 密集建筑群场景的致命缺陷，通常需要引入**随机扰动**或**势场隧道**机制。

### 3.4 快速行进平方法（FM²）

Fast Marching Square (FM²) 以**速度图**驱动波前传播，自动生成平滑 4D 轨迹，特别适合城市峡谷地形。

**Eikonal 方程**（波前到达时间满足）：

$$
|\nabla T(\mathbf{x})|^2 \cdot v^2(\mathbf{x}) = 1
$$

其中 $T(\mathbf{x})$ 为波前到达 $\mathbf{x}$ 的时间，$v(\mathbf{x})$ 为该点的速度。

**基于净空场 $\rho(\mathbf{x})$ 的速度图**：

$$
v(\mathbf{x}) = c \cdot \rho(\mathbf{x}) = c \cdot \min_{obs}\|\mathbf{x} - \mathbf{x}_{obs}\|
$$

障碍物附近 $\rho \to 0$，速度趋近于零；开阔区域速度最大。

**从到达时间场提取路径**（沿负梯度下降）：

$$
\dot{\boldsymbol{\xi}}(s) = -\frac{\nabla T(\boldsymbol{\xi}(s))}{|\nabla T(\boldsymbol{\xi}(s))|}
$$

**4D 扩展——时空冲突区时变速度图**：

$$
v(\mathbf{x}, t) = v_0(\mathbf{x}) \cdot \phi_{conflict}(\mathbf{x}, t)
$$

其中 $\phi_{conflict}$ 在时空冲突区衰减至接近零，自动绕开其他无人机的预测轨迹。

FM² 的优势在于路径天然平滑（波前传播的数学性质保证），且可以方便地融合多目标（速度、能耗、风险）到同一速度图中。

---

## 4. 高密度场景核心问题：冲突探测与解除

当多架无人机同时在 CBD 空域运行时，冲突探测与解除（CD/R）是保证飞行安全的核心技术。

### 4.1 冲突探测（CD - Conflict Detection）

采用 NASA UTM 标准的两层分离保护盾：**水平分离 $d_h$ + 垂直分离 $d_v$**，两者同时触发才判定为冲突：

$$
\text{Conflict}_{ij} \iff \|\Delta\mathbf{p}_{ij}(t)\|_{xy} < d_h \;\wedge\; |\Delta z_{ij}(t)| < d_v
$$

在常速假设下（预测窗口内速度恒定）：

$$
\mathbf{p}_i(t) = \mathbf{p}_i^0 + \mathbf{v}_i t, \quad \mathbf{p}_j(t) = \mathbf{p}_j^0 + \mathbf{v}_j t
$$

**最近接近点时间（CPA）**：

$$
t_{CPA} = -\frac{(\Delta\mathbf{p}_{ij}^0) \cdot \Delta\mathbf{v}_{ij}}{\|\Delta\mathbf{v}_{ij}\|^2}, \quad \Delta\mathbf{v}_{ij} = \mathbf{v}_i - \mathbf{v}_j
$$

**CPA 处的最小分离距离**：

$$
d_{min} = \|\Delta\mathbf{p}_{ij}(t_{CPA})\|
$$

冲突判定：$t_{CPA} > 0$ 且 $t_{CPA} < T_{lookahead}$ 且 $d_{min} < d_h$。

### 4.2 冲突解除（CR - Conflict Resolution）

三大经典策略：

#### 策略一：速度调整（Speed Adjustment）

对 UAV $i$ 施加速度缩放因子 $\alpha_i$：

最优 $\alpha_i^*$ 在最小化速度偏差的同时满足时间分离约束：

$$
\alpha_i^* = \arg\min_\alpha |\alpha - 1| \quad \text{s.t.} \quad |t_{CPA}(\alpha) - t_{CPA}^*| \geq \Delta t_{sep}
$$

#### 策略二：航向偏转（Heading Change）

在水平面内施加航向角扰动 $\Delta\theta$：

最小满足分离的偏转角：

$$
\Delta\theta^* = \arccos\left(\frac{d_h^2 - \|\Delta\mathbf{p}_{ij}^0\|^2 + (v_i \Delta t)^2}{2 \cdot \|\Delta\mathbf{p}_{ij}^0\| \cdot v_i \Delta t}\right)
$$

#### 策略三：高度层分离（Altitude Layer Assignment）

基于方向的静态高度分配（见 2.1 节的分层模型），从源头避免水平面内的迎面冲突，是 CBD 场景最推荐的方案。

### 4.3 集中式 vs 分散式协调

| 架构 | 优点 | 缺点 | 适用规模 |
|-----|------|------|---------|
| **集中式 UTM** | 全局最优 | 通信开销 $O(N^2)$ | <20 架 |
| **分散式 VO/ORCA** | 无中心通信 | 仅局部最优 | 20–200 架 |
| **混合 CTDE** | 可扩展性强 | 训练复杂 | >200 架 |

**Velocity Obstacle (VO)**：由 UAV $j$ 诱导的 VO 是导致碰撞的速度集合：

$$
VO_j = \left\{ \mathbf{v} : \mathbf{v} - \mathbf{v}_j \in \mathcal{D}_{ij} \right\}
$$

其中 $\mathcal{D}_{ij}$ 为以 $\|\Delta\mathbf{p}_{ij}\|$ 为半径的安全盘。

**ORCA（最优互惠避碰）**：每个智能体负责一半避让义务：

$$
ORCA_{i|j} = \left\{ \mathbf{v} : \left(\mathbf{v} - \left(\mathbf{v}_j + \frac{\mathbf{u}}{2}\right)\right) \cdot \mathbf{u} \geq 0 \right\}
$$

其中 $\mathbf{u}$ 为达到 VO 边界所需的最小速度改变量。

可行速度集：

$$
\mathcal{V}_i^{ORCA} = \bigcap_{j \neq i} ORCA_{i|j} \cap \mathcal{V}_i^{dyn}
$$

其中 $\mathcal{V}_i^{dyn}$ 编码了无人机的最大速度和加速度约束。

---

## 5. 基于图论的城市空域建模

### 5.1 航路网络图构建

将城市低空空域抽象为加权有向图：

$$
G = (V, E, W), \quad W: E \to \mathbb{R}_{+}
$$

**复合边权重**（多目标标量化）：

$$
W(e_{ij}) = w_1 d_{ij} + w_2 \Delta t_{ij} + w_3 \mathcal{R}_{ij} + w_4 \mathcal{E}_{ij}
$$

其中 $d_{ij}$ 为欧氏距离，$\Delta t_{ij}$ 为风场修正后的时间代价，$\mathcal{R}_{ij}$ 为地面风险评分（2.1 节），$\mathcal{E}_{ij}$ 为能耗代价。

**廊道容量约束**（最大同时容纳无人机数）：

$$
\text{load}(e_{ij}, t) \leq C_{ij}, \quad \forall t
$$

**空域占用张量**（时空占用栅格的 4D 扩展）：

$$
A_{x,y,z,t} = 1 \iff \exists \text{ UAV at voxel } (x,y,z) \text{ during slot } t
$$

### 5.2 能耗模型（旋翼无人机）

旋翼无人机的能耗模型由叶片动量理论（BEMT）推导。

**悬停功率**（Blade Element Momentum Theory）：

$$
P_{hover} = \sqrt{\frac{(mg)^3}{2\rho_{air} A_r}}
$$

其中 $m$ 为无人机质量，$g$ 为重力加速度，$\rho_{air}$ 为空气密度，$A_r$ 为旋翼盘面积。

**前飞功率**（Zeng et al., 2019）：

$$
P(v) = P_0\!\left(1 + \frac{3v^2}{U_{tip}^2}\right) + P_i\!\left(\sqrt{1 + \frac{v^4}{4v_0^4}} - \frac{v^2}{2v_0^2}\right)^{\!\frac{1}{2}} + \frac{1}{2}d_0\,\rho_{air}\,s\,A\,v^3
$$

其中 $P_0$ 为叶片轮廓功率，$P_i$ 为诱导功率（悬停），$U_{tip}$ 为旋翼翼尖速度，$v_0$ 为悬停诱导速度，$d_0$ 为机身阻力比，$s$ 为实度，$A$ 为旋翼盘面积。

**航段能耗**（以速度 $v$ 飞过长度 $\ell$ 的航段）：

$$
\mathcal{E}_{ij} = \frac{\ell_{ij}}{v} \cdot P(v)
$$

**能效最优巡航速度**（单位距离能耗最小）：

$$
v^* = \arg\min_v \frac{P(v)}{v}
$$

通常 $v^*$ 在 $8\text{--}12\text{m/s}$ 范围内（取决于无人机重量和旋翼配置）。

---

## 6. 风场影响与城市峡谷效应

### 6.1 风场建模

**城市峡谷风速 Weibull 分布**：

$$
f(v_w; k, \lambda) = \frac{k}{\lambda}\left(\frac{v_w}{\lambda}\right)^{k-1} e^{-(v_w/\lambda)^k}
$$

其中 $k$ 为形状参数（典型值 $1.5\text{--}2.5$），$\lambda$ 为尺度参数（来自当地实测数据）。

**对数风剖面**（地面层，高度低于屋顶）：

$$
\bar{u}(z) = \frac{u^*}{\kappa}\ln\!\left(\frac{z - d_0}{z_0}\right)
$$

其中 $u^*$ 为摩擦速度，$\kappa = 0.41$（冯·卡门常数），$d_0$ 为零平面位移，$z_0$ 为粗糙长度。

**有效地速**（考虑侧风影响）：

$$
v_{ground} = v_{air} + \mathbf{v}_w \cdot \hat{\mathbf{e}}_{ij} = v_{air} + v_w \cos\theta_w
$$

**风场修正后的航行时间**：

$$
t_{ij} = \frac{d_{ij}}{v_{air} + v_w \cos\theta_w}
$$

**湍流强度指数**（量化廊道风风险）：

$$
TI = \frac{\sigma_u}{\bar{u}}, \quad \sigma_u = \sqrt{\overline{u'^2}}
$$

### 6.2 安全半径动态调整

近建筑物时涡流加剧，保护半径随高度余量缩小而增大：

$$
d_{safe}(h) = d_{base} + \frac{k \cdot H_{building}}{h - H_{building} + \epsilon}
$$

其中 $h$ 为飞行高度，$H_{building}$ 为建筑物高度，$\epsilon$ 为正则项（避免除零）。

**综合障碍-净空约束**：

$$
\rho(\mathbf{p}(t)) \geq d_{safe}(z(t)), \quad \forall t \in [t_0, t_f]
$$

---

## 7. 多机协同优化：MILP 方法

### 7.1 问题建模为整数规划

为每架 UAV $k$ 同时分配路径和时隙，求解全局协同最优：

**目标函数**（最小化总完成时间与能耗加权和）：

$$
\min \sum_{k=1}^{N} \left(w_1 T_k + w_2 \mathcal{E}_k\right)
$$

**决策变量**：
- $x_{ij}^k \in \{0,1\}$：UAV $k$ 是否经过边 $(i,j) \in E$

**流量守恒约束**（每架无人机进入和离开每个中间节点）：

$$
\sum_{j:(i,j)\in E} x_{ij}^k - \sum_{j:(j,i)\in E} x_{ji}^k = b_i^k \quad \forall i \in V,\; \forall k
$$

其中 $b_i^k = 1$（若 $i$ 为 $k$ 的起点），$b_i^k = -1$（若 $i$ 为 $k$ 的终点），否则 $b_i^k = 0$。

**廊道容量约束**（同时最多 $C_{ij}$ 架 UAV）：

$$
\sum_{k=1}^{N} x_{ij}^k \leq C_{ij} \quad \forall (i,j) \in E
$$

**时间一致性约束**（到达时间与路径一致）：

$$
t_j^k \geq t_i^k + \frac{d_{ij}}{v_{max}} \cdot x_{ij}^k \quad \forall (i,j)\in E,\; \forall k
$$

**时间分离约束**（共享节点时 UAV 间保持时间间隔）：

$$
|t_i^k - t_i^l| \geq \Delta t_{sep} \quad \forall i \in V,\; k \neq l \text{ sharing node } i
$$

利用大 M 法线性化（$z_{il}^k \in \{0,1\}$ 为序贯变量）：

$$
t_i^k - t_i^l \geq \Delta t_{sep} - M(1 - z_{il}^k), \quad t_i^l - t_i^k \geq \Delta t_{sep} - M z_{il}^k
$$

**MINLP 扩展**（当速度 $v_{ij}^k$ 也是优化变量时，能耗同时被优化）：

$$
\min_{x, t, v} \sum_k \sum_{(i,j)} x_{ij}^k \cdot \frac{d_{ij}}{v_{ij}^k} \cdot P(v_{ij}^k) \quad \text{s.t. above constraints} + \quad v_{min} \leq v_{ij}^k \leq v_{max}
$$

MILP/MINLP 的优势在于**全局最优性保证**，缺点是计算复杂度随 UAV 数量指数增长，通常用于离线大规模协同规划或离线基准算法对比。

---

## 8. 强化学习方法（MARL）

当 UAV 规模超过 50 架时，MILP 的计算时间难以满足实时性要求。MARL（多智能体强化学习）提供了从数据中学习实时策略的替代路径。

### 8.1 MARL 方案架构

**状态空间**：本地位置 + 速度 + 邻居相对状态

**动作空间**：离散方向集合 $\{\text{N, NE, E, SE, S, SW, W, NW}\}$ + 速度档位

**个体奖励设计**：

$$
r_i^t = r_{arrive}\cdot\mathbf{1}[goal] - c_{step} - c_{conflict}\cdot\mathbf{1}[conflict] - c_{detour}\cdot\|\mathbf{p}_i^t - \mathbf{p}_{direct}\|
$$

包含到达奖励、步进代价、冲突惩罚和绕路惩罚。

**DQN 更新（离散动作空间）**：

$$
Q(s,a;\theta) \leftarrow Q(s,a;\theta) + \alpha\left[r + \gamma Q(s', \arg\max_{a'}Q(s',a';\theta); \theta^-) - Q(s,a;\theta)\right]
$$

其中 $\theta$ 为在线网络参数，$\theta^-$ 为目标网络参数（定期同步）。

### 8.2 图注意力机制（GAT）——建模邻居 UAV

CBD 空域的通信拓扑是动态的，需要注意力机制自适应加权邻居影响：

$$
e_{ij} = \frac{(\mathbf{W}_Q \mathbf{h}_i)(\mathbf{W}_K \mathbf{h}_j)^T}{\sqrt{d_k}}, \quad \alpha_{ij} = \frac{\exp(e_{ij})}{\sum_l \exp(e_{il})}
$$

$$
\mathbf{h}_i^{attn} = \sum_j \alpha_{ij} (\mathbf{W}_V \mathbf{h}_j)
$$

### 8.3 PPO 策略梯度（混合动作空间）

**PPO Clip 目标函数**：

$$
\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_t\right)\right]
$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 为概率比，$\hat{A}_t$ 为优势函数估计。

**CTDE 架构（集中式训练 + 分散式执行）**：

- ** Critic（批评者）**：训练时使用全局状态 $s$ 估计价值函数
- ** Actor（行动者）**：执行时仅使用本地观测 $o_i$，保证在线执行的可扩展性

---

## 9. 轨迹平滑与动力学约束

### 9.1 Bézier 曲线轨迹生成

A* / RRT* 输出的路径是离散的航路点序列，需要用连续曲线拟合以满足无人机的动力学可行性。

**$n$ 次 Bézier 曲线**：

$$
\boldsymbol{\xi}(u) = \sum_{i=0}^{n} \binom{n}{i}(1-u)^{n-i}u^i \mathbf{P}_i, \quad u \in [0,1]
$$

其中 $\mathbf{P}_i$ 为控制点序列。次数 $n$ 通常取 3–5（阶数越高曲线越灵活，但计算量越大）。

**速度曲线**（Bézier 的一阶导）：

$$
\dot{\boldsymbol{\xi}}(u) = n\sum_{i=0}^{n-1}\binom{n-1}{i}(1-u)^{n-1-i}u^i(\mathbf{P}_{i+1}-\mathbf{P}_i)
$$

**曲率约束**（限制向心加速度）：

$$
\kappa = \frac{\|\dot{\boldsymbol{\xi}} \times \ddot{\boldsymbol{\xi}}\|}{\|\dot{\boldsymbol{\xi}}\|^3} \leq \frac{a_{max}}{v^2}
$$

### 9.2 Minimum Snap 轨迹平滑

四旋翼无人机的动力学约束使其**最小Snap（最小四阶导数积分）**轨迹在工程中被广泛采用：

$$
\min \int_{t_0}^{t_f} \left\|\frac{d^4 \boldsymbol{\xi}}{dt^4}\right\|^2 dt
$$

转化为二次规划（分段多项式形式）：

$$
\min_{\mathbf{c}} \mathbf{c}^T \mathbf{Q} \mathbf{c} \quad \text{s.t. } \mathbf{A}_{eq}\mathbf{c} = \mathbf{b}_{eq}
$$

其中 $\mathbf{Q}$ 编码了 Snap 的平方积分，约束矩阵 $\mathbf{A}_{eq}$ 施加了航路点通过和连续性条件。

Minimum Snap 的优点是**解析最优解存在**（QP 有闭式解），计算量 $O(n)$，非常适合实时在线计算。

---

## 10. 实际部署挑战与工程考量

### 10.1 实时性 vs 最优性的权衡

| 算法 | 典型计算时间 | 最优性 | 适用场景 |
|-----|------------|--------|---------|
| A* | $10\text{--}100\text{ms}$ | 全局最优 | 稀疏障碍，$N<5$ |
| RRT* | $100\text{--}500\text{ms}$ | 渐近最优 | 密集障碍，$N<3$ |
| FM² | $50\text{--}200\text{ms}$ | 全局最优 | 实时平滑需求 |
| MILP | $1\text{--}60\text{s}$ | 全局最优 | 离线规划，$N<20$ |
| ORCA | $<1\text{ms}$ | 局部最优 | 实时避碰，$N>20$ |
| MARL | $<10\text{ms}$（查表） | 局部最优 | 超大规模，$N>50$ |

### 10.2 通信延迟的影响

分布式协调算法的预测精度受通信延迟 $\tau$ 严重制约。延迟 $\tau$ 意味着智能体使用的是 $t - \tau$ 时刻的邻居状态进行预测：

$$
\hat{\mathbf{p}}_j(t) = \mathbf{p}_j(t-\tau) + \mathbf{v}_j(t-\tau) \cdot \tau
$$

预测误差随 $\tau$ 增大而增大，当 $\tau > 500\text{ms}$ 时，基于预测的冲突解除算法性能显著下降。

### 10.3 GPS 精度与多源融合

| 定位方式 | 水平精度 | 垂直精度 | 可用性 |
|---------|---------|---------|-------|
| 普通 GPS | $3\text{--}5\text{m}$ | $5\text{--}10\text{m}$ | 全天候 |
| GPS + GLONASS | $2\text{--}3\text{m}$ | $4\text{--}6\text{m}$ | 全天候 |
| RTK GPS | $2\text{--}3\text{cm}$ | $3\text{--}5\text{cm}$ | 需基站 |
| 视觉里程计 | $0.1\text{--}0.5\text{m}$ | $0.2\text{--}1\text{m}$ | 依赖光照 |
| LiDAR SLAM | $0.05\text{--}0.2\text{m}$ | $0.1\text{--}0.3\text{m}$ | 室内/隧道 |

城市峡谷中 GPS 多径效应使普通 GPS 精度降至 $10\text{m}$ 以上，**必须融合 IMU 或视觉里程计**才能满足 CBD 环境的飞行安全需求。

### 10.4 监管框架

| 地区 | 法规 | 关键约束 |
|-----|------|---------|
| 中国 | CAAC《无人驾驶航空器飞行管理暂行条例》 | 实名登记，120m 高度上限 |
| 欧盟 | EASA U-Space | U-Space 服务商注册 |
| 美国 | FAA UTM CONOPS | Remote ID，LAANC 授权 |

---

## 11. 总结与展望

### 各方法对比

| 算法 | 计算复杂度 | 最优性 | 实时性 | 多机扩展性 |
|-----|-----------|-------|--------|-----------|
| A* | $O(b^d)$ | ✅ 全局最优 | ⭐⭐⭐⭐ | ⭐（差） |
| RRT* | $O(n \log n)$ | 渐近最优 | ⭐⭐⭐ | ⭐（差） |
| APF | $O(n)$ | ❌ 局部极小 | ⭐⭐⭐⭐⭐ | ⭐（差） |
| FM² | $O(n \log n)$ | ✅ 全局最优 | ⭐⭐⭐⭐ | ⭐⭐（中） |
| MILP | $O(2^N)$ | ✅ 全局最优 | ⭐（差） | ⭐（差） |
| ORCA | $O(N^2)$ | ❌ 局部最优 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐（良） |
| MARL | $O(1)$ | ❌ 局部最优 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐（优） |

### 未来方向

1. **数字孪生实时映射**：利用传感器网络实时构建城市低空数字孪生，为规划算法提供高分辨率的动态环境模型
2. **在线 MARL + 元学习**：通过元学习（MAML/RL²）实现快速策略适应，新城市环境仅需少量样本即可部署
3. **大语言模型（LLM）任务分解**：LLM 作为任务规划器，将高层任务指令分解为底层航路点序列，结合传统规划算法执行
4. **V2X（无人机-设施-无人机）通信**：5G/6G 网络下的低延迟 V2X 通信，使分散式算法能获取更精确的邻居状态信息

---

> 📚 **相关阅读**
> - [多智能体强化学习与图注意力网络：UAV 集群冲突消解的端到端方案](/blog/marl-kat-uav-conflict/)
> - [UAV 集群冲突消解：算法全景图](/blog/uav-conflict-resolution/)
> - [城市低空 UTM 冲突环境构建：从理论到代码实现](/blog/uav-conflict-env-construction/)

---

*本文的理论框架综合参考了 UTM (NASA)、EASA U-Space、CAAC 相关技术规范，以及 Schopper et al. (2019)、Zeng et al. (2019)、Zhou et al. (2019) 等学术成果。*
