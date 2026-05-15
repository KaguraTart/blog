---
title: "城市低空无人机航路规划：高密度 CBD 场景下的理论与算法"
description: "系统解析城市低空 UAV 航路规划的核心难题与求解思路，涵盖 A*、RRT*、APF、FM²、MILP、ORCA 和 MARL 方法，附完整数学推导与方程。"
pubDate: 2026-05-15
tags: ["UAV", "路径规划", "城市空域", "优化算法", "UTM", "冲突消解"]
category: Tech
---

# 城市低空无人机航路规划：高密度 CBD 场景下的理论与算法

> 当数百架无人机同时穿梭于摩天大楼之间，航路规划早已不是"从 A 点飞到 B 点"的简单问题——它是一个在三维空间、时间、能量、安全性之间寻找平衡的**高维约束优化难题**。

---

## 引言：为什么 CBD 特别难？

城市低空空域通常定义为地面以上 **0–300 m**（AGL）的飞行区间，这一高度层恰好是无人机物流、巡检、应急响应等应用的主战场。而 CBD（Central Business District，中央商务区）是其中最复杂的子场景，原因有三：

**1. 密集建筑群形成"城市峡谷"**

高楼林立使可用飞行廊道极为狭窄，视线遮蔽导致 GPS 精度下降，建筑物边缘还会产生强烈的绕流湍流——在 50 m 以下的低空，这些湍流完全可以令小型多旋翼失去控制。

**2. 高密度 UAV 引发密集冲突**

郊区场景中，同一时段可能只有数架无人机；而在成熟的城市空中交通管理（UTM）体系下，CBD 上空的无人机数量可能达到每分钟 40 架以上。这意味着冲突探测与解除（Conflict Detection & Resolution, CD&R）成为系统的核心瓶颈，而非边缘功能。

**3. 动态障碍与多约束耦合**

除建筑物外，无人机还要应对临时禁飞区、有人机航线、实时风场变化，以及来自地面人群密度的安全风险——所有这些因素共同作用，使得任何单一的路径规划算法都难以单独应对。

---

## 1. 问题建模：把飞行问题变成数学问题

### 1.1 三维占用栅格（3D Occupancy Grid）

将城市空间离散化为体素（Voxel）网格，每个体素记录其占用状态：

$$
O(x,y,z) = \begin{cases} 1 & \text{障碍物 / 禁飞区} \\ 0 & \text{可飞行} \end{cases}
$$

体素分辨率通常为 1–5 m，CBD 核心区域可细化至 0.5 m。建筑物高度数据来源于 GIS（地理信息系统）数据库，结合实时传感器实现动态更新。

### 1.2 4D 轨迹的数学定义

单架 UAV 的飞行轨迹是一条关于时间参数化的空间曲线：

$$
\boldsymbol{\xi}(t) = \bigl(x(t),\; y(t),\; z(t)\bigr), \quad t \in [t_0,\, t_f]
$$

引入时间维度后，轨迹升维为 4D 时空曲线 $\boldsymbol{\xi}^{4D}(t) = (x,y,z,t)$，这正是所谓 **4D 轨迹规划**的核心思想：通过时间调度（到达某点的时刻）来规避空间上的冲突，比纯粹的空间绕行代价更低。

多机系统中，任意两架 UAV 在任意时刻均须满足安全分离约束：

$$
\|\boldsymbol{\xi}_i(t) - \boldsymbol{\xi}_j(t)\|_2 \geq d_{sep}, \quad \forall\, i \neq j,\; \forall\, t \in [t_0, t_f]
$$

其中 $d_{sep}$ 为最小安全间距，典型取值为 5–30 m（视飞行速度和 GPS 精度而定）。

### 1.3 多目标优化问题的一般形式

航路规划本质上是一个带约束的多目标优化问题：

$$
\min_{\boldsymbol{\xi}}\; J(\boldsymbol{\xi}) = w_1 J_{len} + w_2 J_{time} + w_3 J_{energy} + w_4 J_{risk}
$$

其中各分项含义如下：

| 分项 | 含义 | 典型度量 |
|------|------|----------|
| $J_{len}$ | 路径长度 | $\int_{t_0}^{t_f}\|\dot{\boldsymbol{\xi}}\|\,\mathrm{d}t$ |
| $J_{time}$ | 飞行时间 | $t_f - t_0$ |
| $J_{energy}$ | 能量消耗 | $\int P(v)\,\mathrm{d}t$ |
| $J_{risk}$ | 地面风险 | 飞越人口密度区的积分 |

约束条件（缺一不可）：

- **无障碍**：$O(\boldsymbol{\xi}(t)) = 0,\;\forall t$
- **动力学**：$\dot{\mathbf{x}} = f(\mathbf{x}, \mathbf{u})$（UAV 运动学 / 动力学模型）
- **安全分离**：$\|\boldsymbol{\xi}_i(t)-\boldsymbol{\xi}_j(t)\| \geq d_{sep},\;\forall i\neq j$
- **边界条件**：$\boldsymbol{\xi}(t_0)=\mathbf{p}_{start},\;\boldsymbol{\xi}(t_f)=\mathbf{p}_{goal}$
- **速度限制**：$v_{min} \leq \|\dot{\boldsymbol{\xi}}(t)\| \leq v_{max}$

---

## 2. 单机路径规划算法

在处理多机协同之前，先理解单机场景下的核心算法。

### 2.1 A* 算法：图搜索的基石

A* 在离散化的空域图（Waypoint Graph 或 Visibility Graph）上搜索最短路径。每个节点 $n$ 的评估值为：

$$
f(n) = g(n) + h(n)
$$

其中 $g(n)$ 是从起点到节点 $n$ 的**实际累积代价**：

$$
g(n) = g(\text{parent}) + d(\text{parent},\, n)
$$

$h(n)$ 是从 $n$ 到目标的**可容许启发函数**（admissible heuristic，永不高估真实代价）。城市 3D 空间中常用欧氏距离启发：

$$
h(n) = \|\mathbf{p}_n - \mathbf{p}_{goal}\|_2 = \sqrt{(x_n-x_g)^2+(y_n-y_g)^2+(z_n-z_g)^2}
$$

在城市场景中，仅考虑几何距离是不够的。引入**地面风险加权边代价**：

$$
d(u,v) = \ell_{uv}\cdot\bigl(1 + \beta\cdot\mathcal{R}_{uv}\bigr)
$$

其中 $\ell_{uv}$ 为廊道段长，$\mathcal{R}_{uv}\in[0,1]$ 为该廊道的地面风险评分（综合人口密度、建筑物类型、事故后果等因素），$\beta$ 为风险权重系数。这使得 A* 倾向于选择飞越低风险区域（如河流、公园）的路径，即便稍微绕路。

> A* 的局限：空域图的质量决定了解的质量。在高密度 CBD 中，图的节点数可达数十万，且图的构建本身就是一个挑战。

### 2.2 RRT* 算法：概率完备的渐近最优规划

RRT*（Rapidly-exploring Random Tree Star）通过在连续空间随机采样来探索可行路径，特别适合高维、复杂障碍物场景。

**最近邻查询**——在树 $\mathcal{T}$ 中找到离随机采样点最近的节点：

$$
x_{nearest} = \arg\min_{x \in \mathcal{T}} \|x - x_{rand}\|_2
$$

**步进扩展**——从 $x_{nearest}$ 向 $x_{rand}$ 方向延伸步长 $\delta$：

$$
x_{new} = x_{nearest} + \delta \cdot \frac{x_{rand} - x_{nearest}}{\|x_{rand} - x_{nearest}\|_2}
$$

**RRT* 的核心改进——重连（Rewire）：** 在以 $x_{new}$ 为圆心、半径 $r_n$ 的球内寻找所有近邻节点：

$$
r_n = \gamma_{RRT^*}\!\left(\frac{\log n}{n}\right)^{1/d}
$$

其中 $n$ 为当前树的节点数，$d$ 为空间维度（3D 场景 $d=3$），$\gamma_{RRT^*}$ 是与自由空间体积相关的常数。该半径随采样点增加而**收缩**，保证渐近最优性。

代价更新：

$$
c(x_{new}) = c(x_{near}) + d(x_{near},\, x_{new})
$$

若通过 $x_{new}$ 能降低 $x_{near}$ 的代价，则执行重连：

$$
\text{若 } c(x_{near}) > c(x_{new}) + d(x_{new},\, x_{near}),\text{ 则将 } x_{near} \text{ 的父节点改为 } x_{new}
$$

随着采样次数趋于无穷，RRT* 保证以概率 1 收敛到最优解：

$$
\lim_{n\to\infty} c(\xi_n^*) = c^* \quad \text{（几乎必然）}
$$

### 2.3 人工势场法（APF）：实时性之王

APF 把目标构造为引力场，把障碍物构造为斥力场，UAV 在合力作用下运动。

**引力势**（二次势阱，拉向目标）：

$$
U_{att}(\mathbf{p}) = \frac{1}{2}k_{att}\|\mathbf{p} - \mathbf{p}_{goal}\|^2
$$

**斥力势**（在障碍物影响半径 $\rho_0$ 内激活）：

$$
U_{rep}(\mathbf{p}) = \begin{cases} \dfrac{1}{2}k_{rep}\!\left(\dfrac{1}{\rho(\mathbf{p})}-\dfrac{1}{\rho_0}\right)^{\!2} & \rho(\mathbf{p}) \leq \rho_0 \\[8pt] 0 & \rho(\mathbf{p}) > \rho_0 \end{cases}
$$

其中 $\rho(\mathbf{p})=\min_{obs}\|\mathbf{p}-\mathbf{p}_{obs}\|$ 为 UAV 到最近障碍物的净空距离。

**合力**（总势场的负梯度）：

$$
\mathbf{F}(\mathbf{p}) = -\nabla U_{att}(\mathbf{p}) - \nabla U_{rep}(\mathbf{p})
$$

显式梯度分量：

$$
\nabla U_{att} = k_{att}\,(\mathbf{p}-\mathbf{p}_{goal})
$$

$$
\nabla U_{rep} = k_{rep}\!\left(\frac{1}{\rho}-\frac{1}{\rho_0}\right)\!\frac{1}{\rho^2}\,\nabla\rho \qquad (\rho\leq\rho_0)
$$

APF 计算量极小（每步 $O(1)$），天然适合实时避障。但在 CBD 峡谷中有一个致命弱点：**局部极小值**——当引力和斥力恰好平衡时，UAV 会卡在一个非目标点无法前进。改进方案包括随机扰动、谐波势场或与 RRT 结合的 PF-RRT 算法。

### 2.4 快速行进平方法（FM²）：波前传播的优雅

FM²（Fast Marching Square）通过求解 Eikonal 方程来生成平滑轨迹，特别适合 4D 冲突回避。

**Eikonal 方程**——描述波前到达时间 $T(\mathbf{x})$ 的偏微分方程：

$$
|\nabla T(\mathbf{x})|^2 \cdot v^2(\mathbf{x}) = 1
$$

其中 $v(\mathbf{x})$ 是空间中的传播速度。构造**基于净空距离的速度图**，使波前在障碍物附近自然减速：

$$
v(\mathbf{x}) = c\cdot\rho(\mathbf{x}) = c\cdot\min_{obs}\|\mathbf{x}-\mathbf{x}_{obs}\|
$$

求解 $T(\mathbf{x})$ 后，路径通过对 $T$ 场做梯度下降提取：

$$
\dot{\boldsymbol{\xi}}(s) = -\frac{\nabla T\bigl(\boldsymbol{\xi}(s)\bigr)}{\left|\nabla T\bigl(\boldsymbol{\xi}(s)\bigr)\right|}
$$

**扩展到 4D 冲突回避：** 引入时变速度图，在已被其他 UAV 占用的时空区域令 $v\to 0$：

$$
v(\mathbf{x},t) = v_0(\mathbf{x})\cdot\phi_{conflict}(\mathbf{x},t)
$$

当 $\phi_{conflict}\to 0$ 时，波前自然绕开该时空冲突体积，实现无碰撞的 4D 路径。FM² 生成的路径天然平滑（$C^\infty$ 连续），无需额外的平滑后处理。

---

## 3. 高密度场景的核心难题：冲突探测与解除（CD&R）

高密度 UAV 场景的根本挑战不是找到一条路，而是**保证所有路同时安全**。

### 3.1 冲突探测（Conflict Detection）

定义 UAV $i$ 和 $j$ 之间的相对位置向量：

$$
\Delta\mathbf{p}_{ij}(t) = \mathbf{p}_i(t) - \mathbf{p}_j(t)
$$

**冲突判定条件**（水平 **且** 垂直分离同时违反）：

$$
\text{Conflict}_{ij} \iff \|\Delta\mathbf{p}_{ij}(t)\|_{xy} < d_h \;\wedge\; |\Delta z_{ij}(t)| < d_v
$$

参考 NASA UTM CONOPS 典型参数：水平分离标准 $d_h=30\,\text{m}$，垂直分离标准 $d_v=10\,\text{m}$。

实践中，系统需要在飞行前**预测**冲突而非等冲突发生再响应。在前瞻窗口 $[0, T_h]$ 内假设 UAV 匀速飞行：

$$
\mathbf{p}_i(t) = \mathbf{p}_i^0 + \mathbf{v}_i t, \quad \mathbf{p}_j(t) = \mathbf{p}_j^0 + \mathbf{v}_j t
$$

**最近接近点（CPA, Closest Point of Approach）时刻：**

$$
t_{CPA} = -\frac{\Delta\mathbf{p}_{ij}^0 \cdot \Delta\mathbf{v}_{ij}}{\|\Delta\mathbf{v}_{ij}\|^2}, \qquad \Delta\mathbf{v}_{ij} = \mathbf{v}_i - \mathbf{v}_j
$$

CPA 处的最小间距：

$$
d_{min} = \|\Delta\mathbf{p}_{ij}(t_{CPA})\|
$$

当 $d_{min} < d_{sep}$ 且 $t_{CPA}\in[0, T_h]$ 时，判定存在**预测冲突**，需要立即触发解除机制。

### 3.2 冲突解除（Conflict Resolution）

解除策略分为三类，可单独使用或组合使用：

**策略一：速度调整（Speed Adjustment）**

对 UAV $i$ 施加速度缩放因子 $\alpha$，在动力学允许范围内减速或加速：

$$
\mathbf{v}_i^{new} = \alpha\,\mathbf{v}_i, \quad \alpha\in\!\left[\frac{v_{min}}{v_i},\;\frac{v_{max}}{v_i}\right]
$$

最优 $\alpha$ 在满足分离约束的前提下最小化对原计划的偏离：

$$
\alpha^* = \arg\min_\alpha\;|\alpha-1| \quad \text{s.t. } d_{min}^{new}(\alpha)\geq d_{sep}
$$

**策略二：航向偏转（Heading Change）**

在水平面内将 UAV $i$ 的飞行方向旋转 $\delta\psi$：

$$
\mathbf{v}_i^{new} = v_i\begin{pmatrix}\cos(\psi_i+\delta\psi)\\\sin(\psi_i+\delta\psi)\\0\end{pmatrix}
$$

$$
\delta\psi^* = \arg\min_{|\delta\psi|}\;\delta\psi \quad \text{s.t. } d_{min}(\delta\psi)\geq d_{sep}
$$

**策略三：高度层分离（Altitude Layer Separation）**

CBD 场景中，依照飞行方向分配固定高度层是最高效的系统性方案：

$$
z_{layer}(k) = z_{base} + k\cdot\Delta z_{layer}, \quad k\in\{0,1,\ldots,N_{layer}-1\}
$$

典型配置：东行 $\to z_1$，西行 $\to z_2$，北行 $\to z_3$，南行 $\to z_4$，层间距 $\Delta z_{layer}=10\,\text{m}$。这将三维碰撞问题降维为二维问题，大幅降低系统复杂度。

### 3.3 分散式协调：速度障碍与 ORCA

集中式 UTM 能获得全局最优解，但通信开销随无人机数量 $N$ 以 $O(N^2)$ 增长，在极高密度场景中面临瓶颈。分散式方案中，**速度障碍（Velocity Obstacle, VO）**及其改进 **ORCA** 是最成熟的框架。

**速度障碍**定义——UAV $i$ 因 UAV $j$ 存在而被封禁的速度集合（所有会在时间窗口 $\tau$ 内导致碰撞的速度）：

$$
VO_{ij}^\tau = \left\{\mathbf{v}_i \;\middle|\; \exists\, t\in[0,\tau],\; \mathbf{p}_i+\mathbf{v}_i t \;\in\; \mathbf{p}_j+\mathbf{v}_j t \oplus \mathcal{D}(d_{sep})\right\}
$$

其中 $\mathcal{D}(r)$ 是半径为 $r$ 的圆盘/球体，$\oplus$ 是 Minkowski 和。

**最优互惠碰撞规避（ORCA）**——每个智能体只承担"一半"的规避责任，避免过度保守。ORCA 为智能体 $i$ 相对于 $j$ 定义一个半空间约束：

$$
ORCA_{ij} = \left\{\mathbf{v} \;\middle|\; \bigl(\mathbf{v}-\mathbf{v}_{opt}^i\bigr)\cdot\hat{\mathbf{n}}_{ij} \geq \tfrac{1}{2}u_{ij}\right\}
$$

其中 $u_{ij}$ 是最小速度变化量的大小，$\hat{\mathbf{n}}_{ij}$ 指向 $VO_{ij}$ 边界的法方向。

智能体 $i$ 的可行速度集（对所有邻居约束取交集，再与动力学约束求交）：

$$
\mathcal{V}_i^{ORCA} = \bigcap_{j\neq i} ORCA_{ij} \;\cap\; \mathcal{V}_{dyn}
$$

其中 $\mathcal{V}_{dyn}$ 编码最大速度、加速度等动力学约束。ORCA 在 40 架/分钟以上的密度场景中已实现 100% 成功率，计算复杂度 $O(N^2)$，适合实时部署。

---

## 4. 图论建模：城市空域网络

### 4.1 航路网络图的构建

城市空域被建模为**加权有向图**：

$$
G = (V,\; E,\; W), \quad W: E \to \mathbb{R}_+
$$

- **节点** $V$：道路交叉口上空、建筑物顶部、关键中转点
- **边** $E$：两节点间的合法飞行廊道（需通过碰撞检测验证）
- **边权** $W$：多目标标量化加权

$$
W(e_{ij}) = w_1\, d_{ij} + w_2\,\Delta t_{ij} + w_3\,\mathcal{R}_{ij} + w_4\,\mathcal{E}_{ij}, \quad \sum_{k} w_k = 1
$$

廊道容量约束（同一时刻通行 UAV 数不超过上限）：

$$
\text{load}(e_{ij},\, t) \leq C_{ij}, \quad \forall\, t
$$

整个空域的占用状态可用四维张量描述（$N_x\times N_y\times N_z$ 为体素网格，$N_t$ 为时间槽数量）：

$$
\mathbf{A} \in \{0,1\}^{N_x\times N_y\times N_z\times N_t}, \quad A_{x,y,z,t} = 1 \iff \exists\text{ UAV 占用体素}(x,y,z)\text{ 在时间槽 }t
$$

### 4.2 旋翼无人机能耗模型

能耗是航路规划的重要优化目标，需要精确建模。

**悬停功率**（叶素动量理论推导）：

$$
P_{hover} = \sqrt{\frac{(mg)^3}{2\,\rho_{air}\, A_r}}
$$

其中 $m$ 为无人机质量，$g$ 为重力加速度，$\rho_{air}$ 为空气密度，$A_r$ 为旋翼盘面积。

**前飞功率模型**（Zeng et al. 2019，三项物理分量）：

$$
P(v) = \underbrace{P_0\!\left(1+\frac{3v^2}{U_{tip}^2}\right)}_{\text{叶片型阻}} + \underbrace{P_i\!\left(\sqrt{1+\frac{v^4}{4v_0^4}}-\frac{v^2}{2v_0^2}\right)^{\!\frac{1}{2}}}_{\text{诱导功率}} + \underbrace{\frac{1}{2}\,d_0\,\rho_{air}\,s\,A\,v^3}_{\text{机身阻力}}
$$

参数含义：$P_0$ 为悬停叶片型阻功率，$P_i$ 为悬停诱导功率，$U_{tip}$ 为旋翼桨尖速度，$v_0$ 为悬停诱导速度，$d_0$ 为机身阻力系数，$s$ 为旋翼实度，$A$ 为旋翼盘面积。

飞越段 $e_{ij}$（长度 $\ell_{ij}$，速度 $v$）的能耗：

$$
\mathcal{E}_{ij} = \frac{\ell_{ij}}{v}\cdot P(v)
$$

**最优巡航速度**（单位距离能耗最小）：

$$
v^* = \arg\min_v \frac{P(v)}{v}
$$

对典型小型多旋翼（$m\approx 1\,\text{kg}$），$v^*$ 通常在 8–12 m/s 之间。

---

## 5. 风场与城市峡谷效应

### 5.1 城市风场建模

城市峡谷中的风速分布远比郊野复杂，Weibull 分布被广泛用于统计建模：

$$
f(v_w;\, k,\, \lambda) = \frac{k}{\lambda}\!\left(\frac{v_w}{\lambda}\right)^{k-1}\!\exp\!\left[-\!\left(\frac{v_w}{\lambda}\right)^k\right]
$$

其中形状参数 $k\approx 1.5$–$2.5$（城区湍流较强时取较小值），$\lambda$ 为尺度参数（由局部气象测量定标）。

近地面风速的对数廓线（适用于屋顶高度以下的表面层）：

$$
\bar{u}(z) = \frac{u^*}{\kappa}\ln\!\left(\frac{z - d_0}{z_0}\right), \quad \kappa = 0.41 \text{（von Kármán 常数）}
$$

其中 $u^*$ 为摩擦速度，$d_0$ 为零平面位移高度，$z_0$ 为粗糙长度。

风场对航路规划的定量影响：

**风修正行驶时间**（沿廊道方向分量 $v_w\cos\theta_w$）：

$$
t_{ij} = \frac{d_{ij}}{v_{air} + v_w\cos\theta_w}
$$

**含风阻的段能耗积分**（真空速 = 地速 $-$ 风速）：

$$
\mathcal{E}_{ij}^{wind} = \int_0^{t_{ij}} P\!\left(\|\mathbf{v}_{UAV}(t) - \mathbf{v}_w(t)\|\right)\mathrm{d}t
$$

**湍流强度指数**（量化廊道风险，用于边权 $\mathcal{R}_{ij}$ 的风险分量）：

$$
TI = \frac{\sigma_u}{\bar{u}}, \qquad \sigma_u = \sqrt{\overline{u'^2}}
$$

$TI > 0.3$ 的廊道通常被标记为高风险，规划器会主动避开或提高该段的边权。

### 5.2 动态安全半径

建筑物附近的绕流湍流随高度余量减小而急剧增强，因此安全净空距离不应是固定常数，而应随飞行高度动态调整：

$$
d_{safe}(h) = d_{base} + \frac{k\cdot H_{bld}}{h - H_{bld} + \epsilon}
$$

其中 $h$ 为当前飞行高度，$H_{bld}$ 为附近建筑物高度，$\epsilon$ 为防止分母为零的正则化项。这一公式意味着：UAV 与建筑物顶部的高度余量越小，要求的横向净空距离越大。

动态净空约束：

$$
\rho\bigl(\mathbf{p}(t)\bigr) \geq d_{safe}\bigl(z(t)\bigr), \quad \forall\, t \in [t_0, t_f]
$$

---

## 6. 多机协同优化：MILP 全局建模

对于 $N$ 架无人机的联合路径与时隙分配问题，可建立**混合整数线性规划（MILP）**模型，在小到中等规模（$N\leq 50$）时求得全局最优解。

**目标函数**（最小化所有无人机的总完成时间与能耗）：

$$
\min\;\sum_{k=1}^{N}\!\left(w_1\, T_k + w_2\,\mathcal{E}_k\right)
$$

**决策变量：**
- $x_{ij}^k \in \{0,1\}$：无人机 $k$ 是否选择廊道 $(i,j)$
- $t_i^k \geq 0$：无人机 $k$ 到达节点 $i$ 的时刻

**约束一——流量守恒**（每架无人机进出中间节点各一次）：

$$
\sum_{j:(i,j)\in E}x_{ij}^k - \sum_{j:(j,i)\in E}x_{ji}^k = b_i^k, \quad \forall\, i\in V,\;\forall\, k
$$

其中 $b_i^k\in\{+1,\, 0,\, -1\}$ 分别对应起点、中间节点、终点。

**约束二——廊道容量**：

$$
\sum_{k=1}^{N} x_{ij}^k \leq C_{ij}, \quad \forall\,(i,j)\in E
$$

**约束三——时间一致性**（到达时间与行驶时间匹配）：

$$
t_j^k \geq t_i^k + \frac{d_{ij}}{v_{max}}\cdot x_{ij}^k, \quad \forall\,(i,j)\in E,\;\forall\, k
$$

**约束四——时间分离**（同一节点上不同无人机须保持时间间隔 $\Delta t_{sep}$，Big-M 线性化）：

$$
t_i^k - t_i^l \geq \Delta t_{sep} - M(1 - z_{kl}^i)
$$

$$
t_i^l - t_i^k \geq \Delta t_{sep} - M\, z_{kl}^i
$$

其中 $z_{kl}^i \in \{0,1\}$ 是无人机 $k$、$l$ 在节点 $i$ 的时序排序变量，$M$ 为足够大的常数（Big-M 方法）。

当速度也作为决策变量时，问题升级为**混合整数非线性规划（MINLP）**：

$$
\min_{x,\, t,\, v}\;\sum_k\sum_{(i,j)} x_{ij}^k\cdot\frac{d_{ij}}{v_{ij}^k}\cdot P(v_{ij}^k), \quad v_{min}\leq v_{ij}^k\leq v_{max}
$$

MINLP 是 NP-hard 问题，实用中常用启发式算法（随机分形搜索 SFS、猎豹优化 MCO 等）近似求解。

---

## 7. 强化学习方案：MARL 与注意力机制

当无人机规模超过百架，MILP 的计算复杂度不可接受。**多智能体强化学习（MARL）**提供了一种训练离线、推理极快的替代方案。

### 7.1 奖励函数设计

每架无人机 $i$ 在时间步 $t$ 获得的奖励：

$$
r_i^t = r_{arrive}\cdot\mathbf{1}[goal] - c_{step} - c_{conflict}\cdot\mathbf{1}[conflict] - c_{detour}\cdot\|\mathbf{p}_i^t - \mathbf{p}_{direct}\|
$$

各项含义：$r_{arrive}$ 为到达目标的正奖励；$c_{step}$ 为每步飞行的时间惩罚；$c_{conflict}\cdot\mathbf{1}[conflict]$ 为发生冲突时的惩罚；$c_{detour}$ 为偏离直线的绕行惩罚。

### 7.2 Double-DQN 更新（离散动作空间）

$$
Q(s,a;\theta) \leftarrow Q(s,a;\theta) + \alpha\!\left[r + \gamma\, Q\!\left(s',\,\arg\max_{a'}Q(s',a';\theta);\,\theta^-\right) - Q(s,a;\theta)\right]
$$

在线网络 $\theta$ 选择动作，目标网络 $\theta^-$ 评估价值，解耦选择与评估以减少过估计偏差。

### 7.3 注意力机制：建模邻机影响

CBD 中每架无人机的决策需感知周围邻机状态。**注意力机制**允许智能体 $i$ 动态地对邻居 $j$ 的影响进行加权：

$$
e_{ij} = \frac{(\mathbf{W}_Q\mathbf{h}_i)(\mathbf{W}_K\mathbf{h}_j)^\top}{\sqrt{d_k}}
$$

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_l \exp(e_{il})}, \qquad \mathbf{h}_i^{attn} = \sum_j \alpha_{ij}\,(\mathbf{W}_V\mathbf{h}_j)
$$

注意力权重 $\alpha_{ij}$ 反映了邻机 $j$ 对智能体 $i$ 决策的相关性，距离近、速度冲突大的邻机自然获得更高权重。

### 7.4 PPO 策略梯度（连续 / 混合动作空间）

$$
\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\hat{A}_t,\;\mathrm{clip}\!\left(r_t(\theta),\,1-\varepsilon,\,1+\varepsilon\right)\hat{A}_t\right)\right]
$$

其中概率比：

$$
r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}
$$

Clip 操作将更新步长限制在 $[1-\varepsilon,\, 1+\varepsilon]$ 范围内（通常 $\varepsilon=0.2$），防止策略更新过大导致训练崩溃。

**集中训练、分散执行（CTDE）范式：**
- **训练阶段**：评价网络 $V(s^{global};\phi)$ 使用全局状态，能够感知所有智能体信息
- **执行阶段**：策略网络 $\pi_\theta(a_i\mid o_i)$ 只使用智能体 $i$ 的局部观测，无需通信

---

## 8. 轨迹平滑：Bézier 曲线与 Minimum Snap

路径规划输出的往往是一系列离散航路点，直接跟踪这些路径点会产生不可行的急转弯。需要通过**轨迹平滑**生成动力学可行的连续轨迹。

### 8.1 Bézier 曲线

$n$ 阶 Bézier 曲线由 $n+1$ 个控制点 $\{\mathbf{P}_i\}$ 定义：

$$
\boldsymbol{\xi}(u) = \sum_{i=0}^{n}\binom{n}{i}(1-u)^{n-i}u^i\,\mathbf{P}_i, \quad u \in [0,1]
$$

速度（对参数 $u$ 求导）：

$$
\dot{\boldsymbol{\xi}}(u) = n\sum_{i=0}^{n-1}\binom{n-1}{i}(1-u)^{n-1-i}u^i\,(\mathbf{P}_{i+1}-\mathbf{P}_i)
$$

Bézier 曲线天然具备**凸包性质**——曲线始终在控制点的凸包内，便于障碍物碰撞检查。曲率约束（限制向心加速度）：

$$
\kappa = \frac{\|\dot{\boldsymbol{\xi}}\times\ddot{\boldsymbol{\xi}}\|}{\|\dot{\boldsymbol{\xi}}\|^3} \leq \frac{a_{max}}{v^2}
$$

### 8.2 Minimum Snap：四旋翼的标准方案

对于四旋翼无人机，**最小化 Snap**（加速度的二阶导数）等价于最小化所需推力的变化率，产生最优的飞行动态：

$$
\min\;\int_{t_0}^{t_f}\!\left\|\frac{d^4\boldsymbol{\xi}}{dt^4}\right\|^2\!\mathrm{d}t
$$

将轨迹表示为分段多项式 $\boldsymbol{\xi}_k(t)=\sum_{j=0}^{m}c_{kj}t^j$，上述无限维优化问题可化为**二次规划（QP）**：

$$
\min_{\mathbf{c}}\;\mathbf{c}^\top\mathbf{Q}\mathbf{c} \quad \text{s.t. }\mathbf{A}_{eq}\mathbf{c} = \mathbf{b}_{eq}
$$

矩阵 $\mathbf{Q}$ 编码 Snap 积分（可解析计算），等式约束 $\mathbf{A}_{eq}\mathbf{c}=\mathbf{b}_{eq}$ 强制轨迹通过所有路径点并保证各段之间的位置、速度、加速度连续性。

---

## 9. 方法横向对比

| 方法 | 完备性 | 最优性 | 时间复杂度 | 实时性 | 多机扩展性 |
|------|--------|--------|------------|--------|------------|
| **A\*** | 完备 | 最优（离散图） | $O(b^d)$ | 中 | 差 |
| **RRT\*** | 概率完备 | 渐近最优 | $O(n\log n)$ | 较好 | 中 |
| **APF** | 不完备 | 无保证 | $O(1)$/步 | 极好 | 好 |
| **FM²** | 完备 | 最优（连续） | $O(N\log N)$ | 中 | 中 |
| **MILP** | 完备 | 全局最优 | NP-hard | 差 | 中（$N\leq50$） |
| **ORCA** | 概率完备 | 局部最优 | $O(N^2)$ | 极好 | 极好 |
| **MARL+Attn** | 概率完备 | 近似 | 训练重，推理快 | 好 | 极好 |

**选型建议：**

- **小规模、高安全要求**（$N\leq 20$）→ MILP 全局最优
- **中等规模、实时性敏感**（$20 < N \leq 100$）→ A\* / RRT\* + ORCA 冲突解除
- **大规模、高密度**（$N > 100$）→ MARL + 注意力机制（推理延迟 $< 10\,\text{ms}$）

---

## 10. 总结与展望

城市低空，尤其是 CBD 场景下的高密度 UAV 航路规划，是多学科交叉的系统工程难题。本文梳理了从**单机路径规划**（A\*、RRT\*、APF、FM²）到**多机冲突消解**（CD&R、ORCA、MILP）再到**学习型方法**（MARL、PPO、注意力）的完整方法链，并给出了各核心环节的精确数学表达。

**三个主要的未解挑战：**

1. **实时在线 Replanning**：当突发禁飞区或无人机故障时，系统需在 200 ms 内完成所有受影响轨迹的重新规划。目前 MILP 远达不到这个要求，MARL 是最有前景的候选。

2. **数字孪生与感知融合**：精确的实时城市三维地图（含动态建筑施工、临时围挡、气象信息）是航路规划质量的基础，数字孪生技术有望实现厘米级、亚秒级的空域状态同步。

3. **监管框架的技术化落地**：中国民航局（CAAC）低空管理法规、欧洲 EASA U-Space 以及美国 FAA UTM CONOPS 均对冲突解除时间、飞行计划提交格式、紧急降落程序等有明确要求，算法设计需与监管边界深度耦合。

> 从数学角度看，城市低空航路规划是一个**非凸、非线性、混合整数、多智能体、实时约束**的优化问题。没有哪个单一框架能"一键解决"——工程实践中，往往是多层次的混合架构：战略层用图规划，战术层用 ORCA，紧急层用 APF，共同构成一个鲁棒的空中交通管理系统。

---

**主要参考文献：**

1. Karaman, S., & Frazzoli, E. (2011). *Sampling-based algorithms for optimal motion planning.* International Journal of Robotics Research, 30(7), 846–894.
2. Van den Berg, J., Guy, S. J., Lin, M., & Manocha, D. (2011). *Reciprocal n-body collision avoidance.* Robotics Research, 3–19.
3. Zeng, Y., Xu, J., & Zhang, R. (2019). *Energy minimization for wireless communication with rotary-wing UAV.* IEEE Transactions on Wireless Communications, 18(4), 2329–2345.
4. Mueller, M. W., Hehn, M., & D'Andrea, R. (2015). *A computationally efficient motion primitive for quadrocopter trajectory generation.* IEEE Transactions on Robotics, 31(6), 1294–1310.
5. Brittain, M., & Wei, P. (2019). *Autonomous air traffic controller: A deep multi-agent reinforcement learning approach.* arXiv:1905.01303.
6. Bertram, J., & Wei, P. (2020). *Distributed computational guidance for high-density urban air mobility.* AIAA Aviation Forum.
7. Valavanis, K. P., & Vachtsevanos, G. J. (Eds.). (2015). *Handbook of Unmanned Aerial Vehicles.* Springer.
8. Augugliaro, F., Schoellig, A. P., & D'Andrea, R. (2012). *Generation of collision-free trajectories for a quadrocopter fleet.* IEEE/RSJ IROS, 3977–3982.
