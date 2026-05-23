---
title: "Urban low-altitude drone route planning: theory and algorithm in high-density CBD scenarios"
description: "Systematically analyzes the core problems and solution ideas of urban low-altitude UAV route planning, covering A*, RRT*, APF, FM², MILP, ORCA and MARL methods, with complete mathematical derivation and equations."
pubDate: 2026-05-15
tags: ["UAV", "path planning", "urban airspace", "Optimization algorithm", "UTM", "conflict resolution"]
category: Tech
---

# Urban low-altitude UAV route planning: theory and algorithm in high-density CBD scenarios

> When hundreds of drones shuttle between skyscrapers at the same time, route planning is no longer a simple problem of "flying from point A to point B" - it is a **high-dimensional constrained optimization problem** that seeks a balance between three-dimensional space, time, energy, and safety.

---

## Introduction: Why is CBD so difficult?

Urban low-altitude airspace is usually defined as the flight range **0–300 m** (AGL) above the ground. This height level happens to be the main battlefield for UAV logistics, inspection, emergency response and other applications. The CBD (Central Business District) is the most complex sub-scenario for three reasons:

**1. Dense buildings form an "urban canyon"**

High-rise buildings make the available flight corridors extremely narrow, and the line of sight is blocked, which reduces the accuracy of GPS. The edges of the buildings also generate strong turbulence. At low altitudes below 50 meters, these turbulences can completely cause a small multi-rotor to lose control.

**2. High-density UAVs cause intensive conflicts**

In a suburban scene, there may be only a few drones flying at the same time; while under a mature urban air traffic management (UTM) system, the number of drones over the CBD may reach more than 40 drones per minute. This means that Conflict Detection & Resolution (CD&R) becomes the core bottleneck of the system rather than a peripheral function.

**3. Dynamic obstacle and multi-constraint coupling**

In addition to buildings, drones also have to deal with temporary no-fly zones, manned aircraft routes, real-time wind field changes, and safety risks from crowd density on the ground - all of which work together to make it difficult for any single path planning algorithm to deal with alone.

---

## 1. Problem modeling: turning flight problems into mathematical problems

### 1.1 3D Occupancy Grid

Discretize urban space into a voxel grid, and each voxel records its occupancy status:

$$
O(x,y,z) = \begin{cases} 1 & \text{Obstacle / No-Fly Zone} \\ 0 & \text{Flyable} \end{cases}
$$

Voxel resolution is typically 1–5 m and the CBD core region can be refined to 0.5 m. Building height data comes from a GIS (Geographic Information System) database and is dynamically updated combined with real-time sensors.

### 1.2 Mathematical definition of 4D trajectory

The flight trajectory of a single UAV is a space curve parameterized with respect to time:$$
\boldsymbol{\xi}(t) = \bigl(x(t),\; y(t),\; z(t)\bigr), \quad t \in [t_0,\, t_f]
$$

After introducing the time dimension, the trajectory dimension is increased to a 4D space-time curve $\boldsymbol{\xi}^{4D}(t) = (x,y,z,t)$. This is the core idea of the so-called **4D trajectory planning**: avoiding spatial conflicts through time scheduling (the moment of arrival at a certain point) is cheaper than pure space detours.

In a multi-machine system, any two UAVs must meet the safe separation constraints at any time:

$$
\|\boldsymbol{\xi}_i(t) - \boldsymbol{\xi}_j(t)\|_2 \geq d_{sep}, \quad \forall\, i \neq j,\; \forall\, t \in [t_0, t_f]
$$

Where $d_{sep}$ is the minimum safe separation, with a typical value of 5–30 m (depending on flight speed and GPS accuracy).

### 1.3 General form of multi-objective optimization problem

Route planning is essentially a constrained multi-objective optimization problem:

$$
\min_{\boldsymbol{\xi}}\; J(\boldsymbol{\xi}) = w_1 J_{len} + w_2 J_{time} + w_3 J_{energy} + w_4 J_{risk}
$$

The meaning of each sub-item is as follows:

| Breakdown | Meaning | Typical measures |
|------|------|----------|
| $J_{len}$ | Path length | $\int_{t_0}^{t_f}\|\dot{\boldsymbol{\xi}}\|\,\mathrm{d}t$ |
| $J_{time}$ | Flight time | $t_f - t_0$ |
| $J_{energy}$ | Energy consumption | $\int P(v)\,\mathrm{d}t$ |
| $J_{risk}$ | Ground risk | Points for flying over populated areas |

Constraints (all are indispensable):- **Accessibility**: $O(\boldsymbol{\xi}(t)) = 0,\;\forall t$
- **Kinematics**: $\dot{\mathbf{x}} = f(\mathbf{x}, \mathbf{u})$ (UAV kinematics/dynamics model)
- **Safe separation**: $\|\boldsymbol{\xi}_i(t)-\boldsymbol{\xi}_j(t)\| \geq d_{sep},\;\forall i\neq j$
- **Boundary conditions**: $\boldsymbol{\xi}(t_0)=\mathbf{p}_{start},\;\boldsymbol{\xi}(t_f)=\mathbf{p}_{goal}$
- **Speed limit**: $v_{min} \leq \|\dot{\boldsymbol{\xi}}(t)\| \leq v_{max}$

---

## 2. Single machine path planning algorithm

Before dealing with multi-machine collaboration, first understand the core algorithm in a single-machine scenario.

### 2.1 A* algorithm: the cornerstone of graph search

A* searches for the shortest path on a discretized spatial graph (Waypoint Graph or Visibility Graph). The evaluation value of each node $n$ is:

$$
f(n) = g(n) + h(n)
$$

Where $g(n)$ is the **actual cumulative cost** from the starting point to node $n$:

$$
g(n) = g(\text{parent}) + d(\text{parent},\, n)
$$

$h(n)$ is the admissible heuristic function from $n$ to the target (never overestimate the true cost). Commonly used Euclidean distance heuristics in urban 3D space:

$$
h(n) = \|\mathbf{p}_n - \mathbf{p}_{goal}\|_2 = \sqrt{(x_n-x_g)^2+(y_n-y_g)^2+(z_n-z_g)^2}
$$

In urban scenarios, only considering geometric distance is not enough. Introducing **Ground Risk Weighted Edge Cost**:

$$
d(u,v) = \ell_{uv}\cdot\bigl(1 + \beta\cdot\mathcal{R}_{uv}\bigr)
$$Among them, $\ell_{uv}$ is the corridor segment length, $\mathcal{R}_{uv}\in[0,1]$ is the ground risk score of the corridor (combining factors such as population density, building type, accident consequences, etc.), and $\beta$ is the risk weight coefficient. This makes A* tend to choose paths that fly over low-risk areas (e.g. rivers, parks), even if they are slightly detoured.

> Limitations of A*: The quality of the airspace map determines the quality of the understanding. In high-density CBD, the number of nodes in the graph can reach hundreds of thousands, and the construction of the graph itself is a challenge.

### 2.2 RRT* algorithm: probabilistically complete asymptotic optimal planning

RRT* (Rapidly-exploring Random Tree Star) explores feasible paths by randomly sampling in continuous space, which is particularly suitable for high-dimensional and complex obstacle scenes.

**Nearest neighbor query** - Find the node closest to the random sampling point in the tree $\mathcal{T}$:

$$
x_{nearest} = \arg\min_{x \in \mathcal{T}} \|x - x_{rand}\|_2
$$

**Step expansion** - Extend step size $\delta$ from $x_{nearest}$ to $x_{rand}$ direction:

$$
x_{new} = x_{nearest} + \delta \cdot \frac{x_{rand} - x_{nearest}}{\|x_{rand} - x_{nearest}\|_2}
$$

The core improvement of **RRT* - Rewire:** Find all neighboring nodes in the sphere with $x_{new}$ as the center and radius $r_n$:

$$
r_n = \gamma_{RRT^*}\!\left(\frac{\log n}{n}\right)^{1/d}
$$

Where $n$ is the number of nodes of the current tree, $d$ is the space dimension (3D scene $d=3$), and $\gamma_{RRT^*}$ is a constant related to the free space volume. This radius shrinks as the sampling points increase, ensuring asymptotic optimality.

Cost update:

$$
c(x_{new}) = c(x_{near}) + d(x_{near},\, x_{new})
$$

If the cost of $x_{near}$ can be reduced through $x_{new}$, reconnection is performed:$$
\text{If } c(x_{near}) > c(x_{new}) + d(x_{new},\, x_{near}),\text{ then change the parent node of } x_{near} \text{ to } x_{new}
$$

As the number of sampling approaches infinity, RRT* is guaranteed to converge to the optimal solution with probability 1:

$$
\lim_{n\to\infty} c(\xi_n^*) = c^* \quad \text{(almost certainly)}
$$

### 2.3 Artificial Potential Field Method (APF): The King of Real-time

APF constructs the target as a gravitational field and the obstacles as a repulsive field, and the UAV moves under the action of the resultant force.

**Gravitational potential** (quadratic potential well, pulling towards the target):

$$
U_{att}(\mathbf{p}) = \frac{1}{2}k_{att}\|\mathbf{p} - \mathbf{p}_{goal}\|^2
$$

**Repulsion Potential** (activated within obstacle influence radius $\rho_0$):

$$
U_{rep}(\mathbf{p}) = \begin{cases} \dfrac{1}{2}k_{rep}\!\left(\dfrac{1}{\rho(\mathbf{p})}-\dfrac{1}{\rho_0}\right)^{\!2} & \rho(\mathbf{p}) \leq \rho_0 \\[8pt] 0 & \rho(\mathbf{p}) > \rho_0 \end{cases}
$$

Where $\rho(\mathbf{p})=\min_{obs}\|\mathbf{p}-\mathbf{p}_{obs}\|$ is the clearance distance from the UAV to the nearest obstacle.

**Resultant force** (negative gradient of the total potential field):

$$
\mathbf{F}(\mathbf{p}) = -\nabla U_{att}(\mathbf{p}) - \nabla U_{rep}(\mathbf{p})
$$

Explicit gradient components:

$$
\nabla U_{att} = k_{att}\,(\mathbf{p}-\mathbf{p}_{goal})
$$$$
\nabla U_{rep} = k_{rep}\!\left(\frac{1}{\rho}-\frac{1}{\rho_0}\right)\!\frac{1}{\rho^2}\,\nabla\rho \qquad (\rho\leq\rho_0)
$$

The online update of APF is usually lightweight and suitable for real-time obstacle avoidance; however, if the nearest obstacle distance is calculated directly $\rho(\mathbf{p})=\min_{obs}\|\mathbf{p}-\mathbf{p}_{obs}\|$ according to the aforementioned definition, the naive implementation usually requires at least traversing the obstacle set at each step, which is about $O(n_{obs})$. Single-step queries can only be approximately $O(1)$) if distance fields, ESDFs, or raster queries have been precomputed. But it still has an Achilles heel in the CBD Canyon: **Local Minimum** - When gravity and repulsion are exactly balanced, the UAV will get stuck at a non-target point and be unable to move forward. Improvements include random perturbations, harmonic potential fields, or the PF-RRT algorithm combined with RRT.

### 2.4 Fast Traveling Square Method (FM²): The Elegance of Wavefront Propagation

FM² (Fast Marching Square) generates smooth trajectories by solving the Eikonal equation, which is particularly suitable for 4D conflict avoidance.

**Eikonal equation** - a partial differential equation describing the wavefront arrival time $T(\mathbf{x})$:

$$
|\nabla T(\mathbf{x})|^2 \cdot v^2(\mathbf{x}) = 1
$$

where $v(\mathbf{x})$ is the velocity of propagation in space. Construct a **clearance distance-based velocity map** so that the wavefront naturally decelerates near obstacles:

$$
v(\mathbf{x}) = c\cdot\rho(\mathbf{x}) = c\cdot\min_{obs}\|\mathbf{x}-\mathbf{x}_{obs}\|
$$

After solving $T(\mathbf{x})$, the path is extracted by gradient descent on the $T$ field:

$$
\dot{\boldsymbol{\xi}}(s) = -\frac{\nabla T\bigl(\boldsymbol{\xi}(s)\bigr)}{\left|\nabla T\bigl(\boldsymbol{\xi}(s)\bigr)\right|}
$$**Extended to 4D conflict avoidance:** Introducing a time-varying velocity map, letting $v\to 0$ in the space-time region already occupied by other UAVs:

$$
v(\mathbf{x},t) = v_0(\mathbf{x})\cdot\phi_{conflict}(\mathbf{x},t)
$$

When $\phi_{conflict}\to 0$, the wavefront naturally bypasses the space-time conflict volume, achieving a collision-free 4D path. The paths generated by FM² are naturally smooth ($C^\infty$ continuous) and require no additional smoothing post-processing.

---

## 3. The core problem of high-density scenes: conflict detection and resolution (CD&R)

The fundamental challenge in high-density UAV scenarios is not to find a path, but to ensure that all paths are safe simultaneously.

### 3.1 Conflict Detection

Define the relative position vector between UAV $i$ and $j$:

$$
\Delta\mathbf{p}_{ij}(t) = \mathbf{p}_i(t) - \mathbf{p}_j(t)
$$

**Conflict determination conditions** (horizontal **and** vertical separation are violated simultaneously):

$$
\text{Conflict}_{ij} \iff \|\Delta\mathbf{p}_{ij}(t)\|_{xy} < d_h \;\wedge\; |\Delta z_{ij}(t)| < d_v
$$

Refer to NASA UTM CONOPS typical parameters: horizontal separation standard $d_h=30\,\text{m}$, vertical separation standard $d_v=10\,\text{m}$.

In practice, the system needs to predict conflicts before flight rather than waiting for conflicts to occur before responding. Assume that the UAV flies at a constant speed within the look-ahead window $[0, T_h]$:

$$
\mathbf{p}_i(t) = \mathbf{p}_i^0 + \mathbf{v}_i t, \quad \mathbf{p}_j(t) = \mathbf{p}_j^0 + \mathbf{v}_j t
$$

**Closest Point of Approach (CPA, Closest Point of Approach) time:**$$
t_{CPA} = -\frac{\Delta\mathbf{p}_{ij}^0 \cdot \Delta\mathbf{v}_{ij}}{\|\Delta\mathbf{v}_{ij}\|^2}, \qquad \Delta\mathbf{v}_{ij} = \mathbf{v}_i - \mathbf{v}_j
$$

Minimum spacing at CPA:

$$
d_{min} = \|\Delta\mathbf{p}_{ij}(t_{CPA})\|
$$

When $d_{min} < d_{sep}$ and $t_{CPA}\in[0, T_h]$, it is determined that there is a **prediction conflict** and the release mechanism needs to be triggered immediately.

### 3.2 Conflict Resolution

Disarm strategies fall into three categories and can be used individually or in combination:

**Strategy 1: Speed Adjustment**

Apply a velocity scaling factor $\alpha$ to UAV $i$, decelerating or accelerating within the allowed range of dynamics:

$$
\mathbf{v}_i^{new} = \alpha\,\mathbf{v}_i, \quad \alpha\in\!\left[\frac{v_{min}}{v_i},\;\frac{v_{max}}{v_i}\right]
$$

The optimal $\alpha$ minimizes the deviation from the original plan while satisfying the separation constraints:

$$
\alpha^* = \arg\min_\alpha\;|\alpha-1| \quad \text{s.t. } d_{min}^{new}(\alpha)\geq d_{sep}
$$

**Strategy 2: Heading Change**

Rotate the flight direction of UAV $i$ by $\delta\psi$ in the horizontal plane:

$$
\mathbf{v}_i^{new} = v_i\begin{pmatrix}\cos(\psi_i+\delta\psi)\\\sin(\psi_i+\delta\psi)\\0\end{pmatrix}
$$$$
\delta\psi^* = \arg\min_{|\delta\psi|}\;\delta\psi \quad \text{s.t. } d_{min}(\delta\psi)\geq d_{sep}
$$

**Strategy Three: Altitude Layer Separation**

In the CBD scenario, assigning fixed altitudes according to the flight direction is the most efficient systematic solution:

$$
z_{layer}(k) = z_{base} + k\cdot\Delta z_{layer}, \quad k\in\{0,1,\ldots,N_{layer}-1\}
$$

Typical configuration: eastbound $\to z_1$, westbound $\to z_2$, northbound $\to z_3$, southbound $\to z_4$, layer spacing $\Delta z_{layer}=10\,\text{m}$. This reduces the dimensionality of the three-dimensional collision problem to a two-dimensional problem, greatly reducing the system complexity.

### 3.3 Decentralized Coordination: Speed Barriers and ORCA

Centralized UTM can obtain the global optimal solution, but the communication overhead increases with $O(N^2)$ as the number of UAVs $N$, facing a bottleneck in extremely high-density scenarios. Among decentralized solutions, **Velocity Obstacle (VO)** and its improvement **ORCA** are the most mature frameworks.

**Speed Barrier** Definition - UAV $i$ The set of velocities that are prohibited due to the presence of UAV $j$ (all velocities that would cause a collision within the time window $\tau$):

$$
VO_{ij}^\tau = \left\{\mathbf{v}_i \;\middle|\; \exists\, t\in[0,\tau],\; \mathbf{p}_i+\mathbf{v}_i t \;\in\; \mathbf{p}_j+\mathbf{v}_j t \oplus \mathcal{D}(d_{sep})\right\}
$$

where $\mathcal{D}(r)$ is a disk/sphere of radius $r$ and $\oplus$ is the Minkowski sum.

**Optimal Reciprocal Collision Avoidance (ORCA)** - Each agent only bears "half" of the avoidance responsibility to avoid being overly conservative. ORCA defines a half-space constraint for agent $i$ relative to $j$:$$
ORCA_{ij} = \left\{\mathbf{v} \;\middle|\; \bigl(\mathbf{v}-\mathbf{v}_{opt}^i\bigr)\cdot\hat{\mathbf{n}}_{ij} \geq \tfrac{1}{2}u_{ij}\right\}
$$

where $u_{ij}$ is the size of the minimum velocity change, and $\hat{\mathbf{n}}_{ij}$ points to the normal direction of the $VO_{ij}$ boundary.

The set of feasible velocities for agent $i$ (intersect all neighbor constraints and then intersect with dynamic constraints):

$$
\mathcal{V}_i^{ORCA} = \bigcap_{j\neq i} ORCA_{ij} \;\cap\; \mathcal{V}_{dyn}
$$

Among them, $\mathcal{V}_{dyn}$ encodes dynamic constraints such as maximum velocity and acceleration. ORCA has achieved a 100% success rate in density scenarios of more than 40 frames/minute, with a computational complexity of $O(N^2)$, making it suitable for real-time deployment.

---

## 4. Graph theory modeling: urban airspace network

### 4.1 Construction of route network diagram

Urban airspace is modeled as a weighted directed graph:

$$
G = (V,\; E,\; W), \quad W: E \to \mathbb{R}_+
$$

- **Node** $V$: above road intersections, tops of buildings, key transfer points
- **Edge** $E$: Legal flight corridor between two nodes (needs to pass collision detection verification)
- **Edge weight** $W$: multi-objective scalar weighting

$$
W(e_{ij}) = w_1\, d_{ij} + w_2\,\Delta t_{ij} + w_3\,\mathcal{R}_{ij} + w_4\,\mathcal{E}_{ij}, \quad \sum_{k} w_k = 1
$$

Corridor capacity constraints (the number of UAVs passing at the same time does not exceed the upper limit):

$$
\text{load}(e_{ij},\, t) \leq C_{ij}, \quad \forall\, t
$$

The occupancy status of the entire airspace can be described by a four-dimensional tensor ($N_x\times N_y\times N_z$ is the voxel grid, $N_t$ is the number of time slots):$$
\mathbf{A} \in \{0,1\}^{N_x\times N_y\times N_z\times N_t}, \quad A_{x,y,z,t} = 1 \iff \exists\text{ UAV occupied voxel}(x,y,z)\text{ in time slot }t
$$

### 4.2 Rotor UAV energy consumption model

Energy consumption is an important optimization goal for route planning and requires accurate modeling.

**Hover Power** (derived from leaf element momentum theory):

$$
P_{hover} = \sqrt{\frac{(mg)^3}{2\,\rho_{air}\, A_r}}
$$

Where $m$ is the mass of the drone, $g$ is the gravity acceleration, $\rho_{air}$ is the air density, and $A_r$ is the rotor disk area.

**Forward flight power model** (Zeng et al. 2019, three physical components):

$$
P(v) = \underbrace{P_0\!\left(1+\frac{3v^2}{U_{tip}^2}\right)}_{\text{Blade resistance}} + \underbrace{P_i\!\left(\sqrt{1+\frac{v^4}{4v_0^4}}-\frac{v^2}{2v_0^2}\right)^{\!\frac{1}{2}}}_{\text{Induction power}} + \underbrace{\frac{1}{2}\,d_0\,\rho_{air}\,s\,A\,v^3}_{\text{Body resistance}}
$$

Parameter meaning: $P_0$ is the hovering blade type resistance power, $P_i$ is the hovering induced power, $U_{tip}$ is the rotor tip speed, $v_0$ is the hovering induced speed, $d_0$ is the fuselage drag coefficient, $s$ is the rotor solidity, and $A$ is the rotor disk area.

Energy consumption of flyby segment $e_{ij}$ (length $\ell_{ij}$, speed $v$):

$$
\mathcal{E}_{ij} = \frac{\ell_{ij}}{v}\cdot P(v)
$$

**Optimum cruising speed** (minimum energy consumption per unit distance):

$$
v^* = \arg\min_v \frac{P(v)}{v}
$$

For a typical small multicopter ($m\approx 1\,\text{kg}$), $v^*$ is typically between 8–12 m/s.

---## 5. Wind field and urban canyon effect

### 5.1 Urban wind field modeling

The wind speed distribution in urban canyons is much more complex than in the countryside, and the Weibull distribution is widely used in statistical modeling:

$$
f(v_w;\, k,\, \lambda) = \frac{k}{\lambda}\!\left(\frac{v_w}{\lambda}\right)^{k-1}\!\exp\!\left[-\!\left(\frac{v_w}{\lambda}\right)^k\right]
$$

Among them, the shape parameter $k\approx 1.5$–$2.5$ (the smaller value is taken when the turbulence in urban areas is strong), and $\lambda$ is the scale parameter (calibrated by local meteorological measurements).

Logarithmic profile of near-surface wind speed (for surface layers below roof height):

$$
\bar{u}(z) = \frac{u^*}{\kappa}\ln\!\left(\frac{z - d_0}{z_0}\right), \quad \kappa = 0.41 \text{(von Kármán constant)}
$$

Where $u^*$ is the friction speed, $d_0$ is the zero plane displacement height, and $z_0$ is the roughness length.

Quantitative impact of wind fields on route planning:

**Wind corrected travel time** (along corridor direction component $v_w\cos\theta_w$):

$$
t_{ij} = \frac{d_{ij}}{v_{air} + v_w\cos\theta_w}
$$

**Segment energy consumption integral including wind resistance** (True air speed = Ground speed $-$ Wind speed):

$$
\mathcal{E}_{ij}^{wind} = \int_0^{t_{ij}} P\!\left(\|\mathbf{v}_{UAV}(t) - \mathbf{v}_w(t)\|\right)\mathrm{d}t
$$

**Turbulence Intensity Index** (quantifies corridor risk, risk component for edge weights $\mathcal{R}_{ij}$):

$$
TI = \frac{\sigma_u}{\bar{u}}, \qquad \sigma_u = \sqrt{\overline{u'^2}}
$$

Corridors with $TI > 0.3$ are usually marked as high risk, and the planner will actively avoid or increase the edge weight of this segment.

### 5.2 Dynamic safety radiusThe turbulence around buildings increases sharply as the height margin decreases. Therefore, the safe clearance distance should not be a fixed constant, but should be dynamically adjusted with the flight altitude:

$$
d_{safe}(h) = d_{base} + \frac{k\cdot H_{bld}}{h - H_{bld} + \epsilon}
$$

Where $h$ is the current flight height, $H_{bld}$ is the height of nearby buildings, and $\epsilon$ is the regularization term to prevent the denominator from being zero. This formula means that the smaller the height margin between the UAV and the top of the building, the greater the lateral clearance required.

Dynamic headroom constraints:

$$
\rho\bigl(\mathbf{p}(t)\bigr) \geq d_{safe}\bigl(z(t)\bigr), \quad \forall\, t \in [t_0, t_f]
$$

---

## 6. Multi-machine collaborative optimization: MILP global modeling

For the joint path and time slot allocation problem of $N$ drones, a **Mixed Integer Linear Programming (MILP)** model can be established to obtain the global optimal solution at small to medium scale ($N\leq 50$).

**Objective function** (minimize the total completion time and energy consumption of all drones):

$$
\min\;\sum_{k=1}^{N}\!\left(w_1\, T_k + w_2\,\mathcal{E}_k\right)
$$

**Decision variables:**
- $x_{ij}^k \in \{0,1\}$: whether drone $k$ selects corridor $(i,j)$
- $t_i^k \geq 0$: The time when drone $k$ arrives at node $i$

**Constraint 1 - Traffic Conservation** (Each drone enters and exits the intermediate node once):

$$
\sum_{j:(i,j)\in E}x_{ij}^k - \sum_{j:(j,i)\in E}x_{ji}^k = b_i^k, \quad \forall\, i\in V,\;\forall\, k
$$

Among them, $b_i^k\in\{+1,\, 0,\, -1\}$ correspond to the starting point, intermediate node and end point respectively.

**Constraint 2 - Corridor Capacity**:

$$
\sum_{k=1}^{N} x_{ij}^k \leq C_{ij}, \quad \forall\,(i,j)\in E
$$**Constraint 3 - Time consistency** (arrival time matches travel time):

$$
t_j^k \geq t_i^k + \frac{d_{ij}}{v_{max}}\cdot x_{ij}^k, \quad \forall\,(i,j)\in E,\;\forall\, k
$$

**Constraint 4 - Time separation** (different drones on the same node must maintain a time interval $\Delta t_{sep}$, Big-M linearization):

$$
t_i^k - t_i^l \geq \Delta t_{sep} - M(1 - z_{kl}^i)
$$

$$
t_i^l - t_i^k \geq \Delta t_{sep} - M\, z_{kl}^i
$$

Among them, $z_{kl}^i \in \{0,1\}$ is the time series ordering variable of UAV $k$, $l$ at node $i$, and $M$ is a sufficiently large constant (Big-M method).

When speed is also used as a decision variable, the problem is upgraded to Mixed Integer Nonlinear Programming (MINLP):

$$
\min_{x,\, t,\, v}\;\sum_k\sum_{(i,j)} x_{ij}^k\cdot\frac{d_{ij}}{v_{ij}^k}\cdot P(v_{ij}^k), \quad v_{min}\leq v_{ij}^k\leq v_{max}
$$

MINLP is an NP-hard problem that is approximately solved by commonly used heuristic algorithms in practice (random fractal search SFS, cheetah optimization MCO, etc.).

---

## 7. Reinforcement learning solution: MARL and attention mechanism

When the scale of UAVs exceeds a hundred, the computational complexity of MILP is unacceptable. **Multi-agent reinforcement learning (MARL)** provides an alternative for offline training and extremely fast inference.

### 7.1 Reward function design

The reward received by each drone $i$ at time step $t$:

$$
r_i^t = r_{arrive}\cdot\mathbf{1}[goal] - c_{step} - c_{conflict}\cdot\mathbf{1}[conflict] - c_{detour}\cdot\|\mathbf{p}_i^t - \mathbf{p}_{direct}\|
$$The meaning of each item: $r_{arrive}$ is the positive reward for reaching the target; $c_{step}$ is the time penalty for each flight step; $c_{conflict}\cdot\mathbf{1}[conflict]$ is the penalty when a conflict occurs; $c_{detour}$ is the detour penalty for deviating from the straight line.

### 7.2 Double-DQN update (discrete action space)

$$
Q(s,a;\theta) \leftarrow Q(s,a;\theta) + \alpha\!\left[r + \gamma\, Q\!\left(s',\,\arg\max_{a'}Q(s',a';\theta);\,\theta^-\right) - Q(s,a;\theta)\right]
$$

The online network $\theta$ selects actions, and the target network $\theta^-$ evaluates values, decoupling selection and evaluation to reduce overestimation bias.

### 7.3 Attention Mechanism: Modeling Neighbor Influence

The decision-making of each drone in the CBD requires sensing the status of its surrounding neighbors. The **attention mechanism** allows agent $i$ to dynamically weight the influence of neighbors $j$:

$$
e_{ij} = \frac{(\mathbf{W}_Q\mathbf{h}_i)(\mathbf{W}_K\mathbf{h}_j)^\top}{\sqrt{d_k}}
$$

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_l \exp(e_{il})}, \qquad \mathbf{h}_i^{attn} = \sum_j \alpha_{ij}\,(\mathbf{W}_V\mathbf{h}_j)
$$

The attention weight $\alpha_{ij}$ reflects the relevance of the neighbor $j$ to the decision-making of the agent $i$. Neighbors with close distances and large speed conflicts naturally receive higher weights.

### 7.4 PPO policy gradient (continuous/mixed action space)$$
\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\hat{A}_t,\;\mathrm{clip}\!\left(r_t(\theta),\,1-\varepsilon,\,1+\varepsilon\right)\hat{A}_t\right)\right]
$$

where the probability ratio is:

$$
r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}
$$

The Clip operation limits the update step size to the range of $[1-\varepsilon,\, 1+\varepsilon]$ (usually $\varepsilon=0.2$) to prevent training from crashing due to excessive policy updates.

**Centralized Training, Decentralized Execution (CTDE) Paradigm:**
- **Training phase**: Evaluation network $V(s^{global};\phi)$ uses global state and can perceive all agent information
- **Execution phase**: Policy network $\pi_\theta(a_i\mid o_i)$ only uses local observations of agent $i$, without communication

---

## 8. Trajectory Smoothing: Bézier Curve and Minimum Snap

The output of path planning is often a series of discrete waypoints, and directly tracking these waypoints will produce unfeasible sharp turns. It is necessary to generate dynamically feasible continuous trajectories through **trajectory smoothing**.

### 8.1 Bézier Curve

A Bézier curve of order $n$ is defined by $n+1$ control points $\{\mathbf{P}_i\}$:

$$
\boldsymbol{\xi}(u) = \sum_{i=0}^{n}\binom{n}{i}(1-u)^{n-i}u^i\,\mathbf{P}_i, \quad u \in [0,1]
$$

Velocity (derivative with respect to parameter $u$):

$$
\dot{\boldsymbol{\xi}}(u) = n\sum_{i=0}^{n-1}\binom{n-1}{i}(1-u)^{n-1-i}u^i\,(\mathbf{P}_{i+1}-\mathbf{P}_i)
$$Bézier curves naturally have convex hull properties - the curve is always within the convex hull of the control points, which facilitates obstacle collision checking. Curvature constraints (limiting centripetal acceleration):

$$
\kappa = \frac{\|\dot{\boldsymbol{\xi}}\times\ddot{\boldsymbol{\xi}}\|}{\|\dot{\boldsymbol{\xi}}\|^3} \leq \frac{a_{max}}{v^2}
$$

### 8.2 Minimum Snap: The standard solution for quadcopters

For a quadcopter UAV, minimizing Snap (the second derivative of acceleration) is equivalent to minimizing the rate of change of required thrust, resulting in optimal flight dynamics:

$$
\min\;\int_{t_0}^{t_f}\!\left\|\frac{d^4\boldsymbol{\xi}}{dt^4}\right\|^2\!\mathrm{d}t
$$

Expressing the trajectory as a piecewise polynomial $\boldsymbol{\xi}_k(t)=\sum_{j=0}^{m}c_{kj}t^j$, the above infinite-dimensional optimization problem can be reduced to **quadratic programming (QP)**:

$$
\min_{\mathbf{c}}\;\mathbf{c}^\top\mathbf{Q}\mathbf{c} \quad \text{s.t. }\mathbf{A}_{eq}\mathbf{c} = \mathbf{b}_{eq}
$$

The matrix $\mathbf{Q}$ encodes the Snap integral (can be calculated analytically), and the equality constraint $\mathbf{A}_{eq}\mathbf{c}=\mathbf{b}_{eq}$ forces the trajectory to pass through all path points and ensures the continuity of position, velocity, and acceleration between segments.

---

## 9. Horizontal comparison of methods| Method | Completeness | Optimality | Time complexity | Real-time | Multi-machine scalability |
|------|--------|--------|------------|--------|------------|
| **A\*** | Complete | Optimal (discrete graph) | $O(b^d)$ | Medium | Poor |
| **RRT\*** | Probabilistically complete | Asymptotically optimal | $O(n\log n)$ | Better | Medium |
| **APF** | Incomplete | No guarantee | $O(1)$/step | Excellent | Good |
| **FM²** | Complete | Optimal (continuous) | $O(N\log N)$ | Medium | Medium |
| **MILP** | Complete | Global Optimal | NP-hard | Poor | Medium ($N\leq50$) |
| **ORCA** | Probabilistically complete | Local optimum | $O(N^2)$ | Excellent | Excellent |
| **MARL+Attn** | Complete probability | Approximate | Heavy training, fast inference | Good | Excellent |

**Selection suggestions:**

- **Small scale, high security requirements** ($N\leq 20$) → MILP global optimal
- **Medium scale, real-time sensitive** ($20 < N \leq 100$) → A\* / RRT\* + ORCA conflict resolution
- **Large scale, high density** ($N > 100$) → MARL + attention mechanism (inference delay $< 10\,\text{ms}$)

---

## 10. Summary and Outlook

Urban low-altitude, especially high-density UAV route planning in CBD scenarios, is a multidisciplinary system engineering problem. This article sorts out the complete method chain from **single-machine path planning** (A\*, RRT\*, APF, FM²) to **multi-machine conflict resolution** (CD&R, ORCA, MILP) to **learning methods** (MARL, PPO, attention), and gives the precise mathematical expression of each core link.

**Three main unsolved challenges:**

1. **Real-time online Replanning**: When a sudden no-fly zone or drone failure occurs, the system needs to complete the replanning of all affected trajectories within 200 ms. Currently MILP falls far short of this requirement and MARL is the most promising candidate.2. **Digital twins and perception fusion**: Accurate real-time three-dimensional urban maps (including dynamic building construction, temporary enclosures, and meteorological information) are the basis for the quality of route planning. Digital twin technology is expected to achieve centimeter-level and sub-second-level airspace status synchronization.

3. **Technical implementation of the regulatory framework**: The Civil Aviation Administration of China (CAAC) low-altitude management regulations, European EASA U-Space, and American FAA UTM CONOPS all have clear requirements for conflict resolution time, flight plan submission format, emergency landing procedures, etc., and algorithm design needs to be deeply coupled with regulatory boundaries.

> From a mathematical perspective, urban low-altitude air route planning is a non-convex, non-linear, mixed integer, multi-agent, real-time constrained optimization problem. No single framework can "solve it with one click" - in engineering practice, it is often a multi-level hybrid architecture: map planning is used at the strategic layer, ORCA is used at the tactical layer, and APF is used at the emergency layer, which together form a robust air traffic management system.

---

**Main references:**1. Karaman, S., & Frazzoli, E. (2011). *Sampling-based algorithms for optimal motion planning.* International Journal of Robotics Research, 30(7), 846–894.
2. Van den Berg, J., Guy, S. J., Lin, M., & Manocha, D. (2011). *Reciprocal n-body collision avoidance.* Robotics Research, 3–19.
3. Zeng, Y., Xu, J., & Zhang, R. (2019). *Energy minimization for wireless communication with rotary-wing UAV.* IEEE Transactions on Wireless Communications, 18(4), 2329–2345.
4. Mueller, M. W., Hehn, M., & D'Andrea, R. (2015). *A computationally efficient motion primitive for quadrocopter trajectory generation.* IEEE Transactions on Robotics, 31(6), 1294–1310.
5. Brittain, M., & Wei, P. (2019). *Autonomous air traffic controller: A deep multi-agent reinforcement learning approach.* arXiv:1905.01303.
6. Bertram, J., & Wei, P. (2020). *Distributed computational guidance for high-density urban air mobility.* AIAA Aviation Forum.
7. Valavanis, K. P., & Vachtsevanos, G. J. (Eds.). (2015). *Handbook of Unmanned Aerial Vehicles.* Springer.
8. Augugliaro, F., Schoellig, A. P., & D'Andrea, R. (2012). *Generation of collision-free trajectories for a quadrocopter fleet.* IEEE/RSJ IROS, 3977–3982.