---
title: "Paper E 实验任务书 v2：面向 AAAI 的验证纠错式 UAV 语言规划"
description: "v2 聚焦 AAAI 顶会投稿：补充 30+ 篇真实可引用定会/顶刊/关键预印本文献，深化 VERA-UAV 的实验指标、对比与消融方案、可复现实验协议，并给出相对完备性的数学证明。"
pubDate: 2026-05-17
updatedDate: 2026-05-23
tags: ["Paper E", "AAAI", "UAV", "LLM", "LTL", "STL", "形式化验证", "实验任务书", "完备性证明"]
category: Tech
---

# Paper E 实验任务书 v2：面向 AAAI 的验证纠错式 UAV 语言规划

> 本文件仍沿用 `paper-e-vera-uav-experiment-taskbook-v1-20260517.md` 文件名，是因为本轮要求“直接在 V1 版本上修改”。正文、标题与版本说明均已升级为 **v2**。本文不是最终论文草稿，而是可执行的实验任务书：明确 Paper E 的研究定位、真实可引用文献、算法方案、数据构建、对比实验、消融实验、评价指标、理论完备性边界与后续 AAAI/T-ITS 推进计划。2026-05-19 的补充重点是：数据防泄漏、failure taxonomy、参数预算、指标公式、图表计划和 AAAI 合规风险。

---

## 1. 研究背景与目标

城市低空 UAV 任务规划正在从“工程师预设航线”走向“自然语言任务驱动”。在实际应用中，操作者更可能给出如下指令：

- “先检查 3 号楼东侧外立面，再去楼顶降落点等待。”
- “避开医院上空，30 秒内到达临时投送区。”
- “如果南侧走廊被占用，就绕行西侧通道，但全程保持 20 米以上安全距离。”

这些指令同时包含语义理解、时序顺序、空间约束、连续轨迹安全和可达性判断。大语言模型（LLM）擅长理解自然语言与生成候选计划，但不能保证输出计划在物理空间中可执行，也不能保证满足航空安全约束。形式化方法则擅长给出可验证语义，例如 Linear Temporal Logic（LTL）和 Signal Temporal Logic（STL），但直接手写规格需要专业知识，难以服务非专家操作者。

已有工作已经证明，自然语言到 LTL 的翻译可以显著降低机器人任务规格编写门槛。例如 Lang2LTL 将复杂导航命令转为 LTL 并在未见环境中进行泛化评测 [1]；NL2LTL 提供了自然语言到 LTL 的开源 Python 包 [2]；LTLCodeGen 用代码生成方式提升 LTL 语法正确性并接入机器人路径规划 [3]；ConformalNL2LTL 进一步尝试用 conformal prediction 给出翻译正确率保证 [4]。这些工作为本研究提供了重要基础。

但对于低空 UAV 场景，仅做 NL-to-LTL 翻译还不够。UAV 任务有三个额外要求：

1. **连续安全约束**：飞行高度、速度、障碍距离、时间窗等约束天然是连续信号上的约束，更适合用 STL robustness 评价。
2. **可执行轨迹闭环**：规格正确不等于轨迹可行，必须经过地图、动力学和规划器验证。
3. **错误可修复**：LLM 的错误不应只被判错，而应被验证器转化为 counterexample 或 robustness feedback，再驱动 LLM 修正。

因此，本文拟提出 **VERA-UAV**：一个面向 UAV 自然语言任务的验证纠错式 neuro-symbolic planning 框架。AAAI 版本优先回答一个核心问题：

> 给定自然语言 UAV 任务，如何让本地开源 LLM 生成可验证、可修复、可执行的 LTL/STL 任务规格与轨迹，而不是只生成看似合理但不可证明安全的文本计划？

AAAI 主会版本聚焦 AI planning、neuro-symbolic verification 和 LLM self-repair。AirSim、真实低空物流、多 UAV 空域吞吐量等系统级内容放到后续 T-ITS 扩展版本。

---

## 2. 问题定义与核心假设

### 2.1 输入与输出

给定一个 UAV 任务实例：

$$
\mathcal{I} = (x_{\text{NL}}, \mathcal{M}, s_0)
$$

其中 $x_{\text{NL}}$ 是自然语言任务指令，$\mathcal{M}$ 是带语义标注的城市低空地图，$s_0$ 是 UAV 初始状态。地图包含建筑、禁飞区、可通行空域、降落点、检查目标、动态障碍和高度层。

系统输出：

$$
\mathcal{O} = (\text{TaskIR}, \varphi_{\text{LTL}}, \varphi_{\text{STL}}, \tau, r)
$$

其中 TaskIR 是结构化中间表示，$\varphi_{\text{LTL}}$ 是离散时序任务规格，$\varphi_{\text{STL}}$ 是连续轨迹约束，$\tau$ 是候选轨迹，$r$ 是验证结果。若任务不可满足，系统应输出 `UNSAT` 或 `NEED_CLARIFICATION`，而不是强行生成不安全轨迹。

### 2.2 任务类型

AAAI 主实验覆盖六类任务：

| 类型 | 示例 | 关键难点 |
|------|------|----------|
| Reach-avoid | 到达 A，避开 B | 基本可达性与避障 |
| Ordered waypoints | 先到 A，再到 B | 时序顺序 |
| Patrol / inspection | 巡检 A、B、C | 多目标覆盖 |
| Time-window delivery | 30 秒内到达 A | 连续时间约束 |
| Emergency landing | 若前路不可达则去最近降落点 | 条件与备选策略 |
| Ambiguous / impossible | “去那个安全的地方”或互斥约束 | 澄清与不可满足检测 |

### 2.3 核心假设

本文不假设 LLM 自身可靠。相反，本文假设 LLM 经常会犯以下错误：

- 生成语法不合法的 LTL/STL。
- 漏掉自然语言中的安全约束。
- 引用地图中不存在的实体。
- 给出满足文本但不可执行的任务顺序。
- 在连续轨迹上违反最小距离、高度或时间窗约束。

VERA-UAV 的核心假设是：**如果验证器能够把这些错误转化为结构化反例、unsat core 和 robustness feedback，本地开源 LLM 的修正成功率会显著高于纯 prompt 重试；进一步，如果系统保留有限 DSL 内的符号枚举 fallback，则算法可以获得相对完备性，而不是把完备性建立在 LLM 可靠性之上。**

---

## 3. 相关工作与可引用论文

### 3.1 文献图谱总览

v1 的问题之一是参考文献太少，容易被审稿人认为“只是在 Lang2LTL / LTLCodeGen 上换了一个 UAV 应用”。v2 将相关工作扩展成五条线：自然语言到时序逻辑、LLM 规划与自修复、STL/形式化验证、shielding 与安全 agent、UAV-VLN 与低空应用。下表列出 **37 篇强相关文献**，每一篇都在本文中被引用。

| 编号 | 文献 | Venue / status | 与本文关系 |
|------|------|----------------|------------|
| [1] | Lang2LTL | CoRL 2023 / PMLR | NL-to-LTL grounding 的直接起点 |
| [2] | NL2LTL | AAAI 2023 Demo | template/tool baseline |
| [3] | LTLCodeGen | IROS 2025 / arXiv | 最强直接 baseline，代码生成保证语法 |
| [4] | ConformalNL2LTL | arXiv 2025 | 翻译可信度与拒答机制参考 |
| [5] | NL2SpaTiaL | arXiv 2025/2026 | 结构化逻辑树与空间关系启发 |
| [6] | T3 Planner | arXiv 2025 | 自修正式 LLM + STL motion planning 直接竞争 |
| [7] | SENTINEL | arXiv 2025/2026 | 多层 formal safety evaluation |
| [8] | LogicGuard | arXiv 2025 | temporal-logic critic 与安全约束生成 |
| [9] | Pro2Guard | arXiv 2025 | probabilistic runtime monitoring |
| [10] | Generalized Planning in PDDL Domains with LLMs | AAAI 2024 | verifier/debugging feedback 对 planning 的价值 |
| [11] | Critical investigation of LLM planning | NeurIPS 2023 | 说明 LLM 直接规划能力有限 |
| [12] | LLM+P | arXiv 2023 | LLM + classical planner 的框架参考 |
| [13] | PlanBench | NeurIPS 2023 Datasets & Benchmarks | LLM planning benchmark 设计参考 |
| [14] | ReAct | ICLR 2023 | reasoning-action loop baseline |
| [15] | SayCan | CoRL 2022 | affordance-grounded LLM planning baseline |
| [16] | Code as Policies | ICRA 2023 | LLM 生成可执行程序策略 |
| [17] | ProgPrompt | ICRA 2023 / Autonomous Robots | situated robot task plan generation |
| [18] | Temporal-Logic-Based Reactive Mission and Motion Planning | IEEE T-RO 2009 | 机器人 LTL planning 经典基础 |
| [19] | Synthesis for Robots | Annual Review 2018 | 形式化合成与机器人行为反馈综述 |
| [20] | Monitoring Temporal Properties of Continuous Signals | FORMATS/FTRTFT 2004 | STL 起点 |
| [21] | Robustness of temporal logic specifications | Theoretical Computer Science 2009 | robustness semantics 基础 |
| [22] | Robust satisfaction over real-valued signals | FORMATS 2010 | STL robustness 计算基础 |
| [23] | Reactive Synthesis from STL Specifications | HSCC 2015 | STL 与控制/规划耦合 |
| [24] | Diagnosis and Repair for STL Synthesis | HSCC 2016 | specification diagnosis/repair 理论参照 |
| [25] | Spot 2.0 | ATVA 2016 | LTL/omega-automata 工具 |
| [26] | RTAMT | STTT 2024 / arXiv 2025 | STL robustness monitor |
| [27] | PRISM 4.0 | CAV 2011 | probabilistic model checking 工具 |
| [28] | Safe RL via Shielding | AAAI 2018 | shield 保证安全的经典定会工作 |
| [29] | Probabilistic Shielding | AAAI 2025 | 概率安全保证与 shielding |
| [30] | AerialVLN | ICCV 2023 | UAV 视觉语言导航基准 |
| [31] | Realistic UAV-VLN | ICLR 2025 | 更真实 UAV-VLN 平台、基准与方法 |
| [32] | ASMA | RA-L / arXiv 2024 | UAV-VLN 中 CBF 安全约束参考 |
| [33] | LogisticsVLN | arXiv 2025 | 低空配送语言导航应用场景 |
| [34] | UAV-VLN Survey | arXiv 2026 | UAV-VLN 研究路线图与挑战 |
| [35] | Qwen3 Technical Report | arXiv 2025 | 本地开源模型选择依据 |
| [36] | DeepSeek-R1 | arXiv 2025 | 推理型开源模型选择依据 |
| [37] | vLLM / PagedAttention | SOSP 2023 | 多模型本地推理实现依据 |

### 3.2 现有工作的关键空白

Lang2LTL、NL2LTL、LTLCodeGen 和 ConformalNL2LTL 共同说明 NL-to-LTL 已经不是空白方向 [1-4]。因此，Paper E 不能只声称“我们把自然语言翻译为 LTL”。真正有潜力的差异点在于：

1. **从翻译正确性扩展到执行正确性**：LTLCodeGen 已经处理语法正确和路径生成 [3]，但 UAV 的高度、速度、障碍距离和时间窗需要 STL robustness，而不只是 LTL formula validity。
2. **从单次生成扩展到验证纠错闭环**：T3 Planner、LogicGuard、SENTINEL 和 Pro2Guard 说明 formal feedback 正在成为 embodied LLM safety 的热点 [6-9]。VERA-UAV 需要更明确地把反例、unsat core 和 robustness trace 作为修复信号。
3. **从 LLM heuristic 扩展到相对完备算法**：LLM 自修复本身不可证明完备；完备性必须来自有限 DSL、可判定验证器和符号枚举 fallback，而不是来自模型“可能会想对”。
4. **从地面导航扩展到低空 UAV**：AerialVLN 和 ICLR 2025 的 realistic UAV-VLN 工作强调 UAV 与地面 VLN 的差异：三维运动、连续动力学、空域安全和资源约束 [30,31]。这正是 VERA-UAV 使用 STL 的动机。

### 3.3 投稿与期刊延展约束

AAAI-26 Main Technical Track 官方说明要求主文最多 7 页技术内容，并要求作者完成 reproducibility checklist [38]。因此 AAAI 版本必须聚焦方法、核心实验和可复现性，不能展开过多系统工程内容。

T-ITS 的 scope 覆盖现代交通系统中的 sensing、communications、controls、planning、design 和 implementation，也覆盖 Artificial Intelligence 等方法方向，并要求期刊扩展相对会议论文有明确新增贡献 [39]。因此，后续 ITS 期刊版本应增加城市低空交通系统指标，例如空域利用率、任务吞吐量、多 UAV 协同、通信延迟和运行安全收益。

---

## 4. 拟提出算法：VERA-UAV

### 4.1 总体流程

VERA-UAV 的全称暂定为：

**VERA-UAV: Verification-Enhanced Repair for Autonomous UAV Language Planning**

系统流程如下：

```text
Natural-language UAV instruction
        ↓
Local open-source LLM
        ↓
Typed TaskIR
        ↓
TaskIR-to-LTL/STL compiler
        ↓
Spot / RTAMT / optional PRISM verification
        ↓
Counterexample + unsat core + robustness feedback
        ↓
LLM repair + symbolic enumerative fallback
        ↓
A* / RRT* / MPC-lite trajectory generation
        ↓
Final trajectory verification
        ↓
Executable trajectory or UNSAT / NEED_CLARIFICATION
```

与 v1 相比，v2 的关键变化是加入 **symbolic enumerative fallback**：LLM 仍然是主要候选生成器，但当 LLM 多轮修复失败时，系统会在有限 TaskIR DSL 内枚举候选修复。这一设计是后续“相对完备性”证明的基础。

### 4.2 Typed TaskIR

TaskIR 是自然语言和形式逻辑之间的结构化接口。它避免让 LLM 直接输出任意 LTL/STL 字符串，从而减少语法错误和 entity grounding 错误。

TaskIR 字段设计如下：

| 字段 | 含义 | 示例 |
|------|------|------|
| `entities` | 指令中涉及的对象 | `building_3`, `hospital_zone`, `landing_pad_A` |
| `goals` | 需要达成的目标 | `reach(landing_pad_A)` |
| `avoid` | 必须避开的区域 | `avoid(hospital_zone)` |
| `sequence` | 子目标顺序 | `inspect(B3_east) -> land(A)` |
| `metric_bounds` | 连续约束 | `altitude in [20,120]`, `distance_to_obstacle >= 10` |
| `time_windows` | 时间窗 | `reach(A) within 30s` |
| `fallbacks` | 备选策略 | `if blocked, reach nearest_safe_pad` |
| `uncertainty` | 模糊或缺失字段 | `NEED_CLARIFICATION(target="safe place")` |

### 4.3 TaskIR 到 LTL/STL 编译

LTL 用于表达离散时序结构：

$$
\varphi_{\text{LTL}} =
G(\neg collision) \wedge F(reach(goal)) \wedge G(\neg enter(no\_fly\_zone))
$$

STL 用于表达连续信号约束：

$$
\varphi_{\text{STL}} =
G_{[0,T]}(d_{\text{obs}}(t) \ge d_{\min})
\wedge
G_{[0,T]}(h_{\min} \le h(t) \le h_{\max})
\wedge
F_{[0,30]}(reach(goal))
$$

其中 $d_{\text{obs}}(t)$ 是 UAV 到最近障碍物的距离，$h(t)$ 是飞行高度。RTAMT 或等价 STL monitor 输出 robustness：

$$
\rho(\tau, \varphi_{\text{STL}}) > 0
$$

表示轨迹 $\tau$ 满足规格；若 $\rho \le 0$，验证器返回违反子句、违反时刻和最小安全裕度。

### 4.4 反例驱动修复

验证器不只返回 `pass/fail`，而是返回结构化诊断：

```json
{
  "status": "FAILED",
  "stage": "STL_ROBUSTNESS",
  "violated_clause": "G[0,T](distance_to_obstacle >= 10)",
  "counterexample_trace": [
    {"t": 14.2, "x": 38, "y": 51, "z": 30, "distance_to_obstacle": 6.4}
  ],
  "robustness": -3.6,
  "repair_hint": "Increase safety margin or route around building_7 west side."
}
```

LLM 的 repair prompt 不要求自由发挥，而要求它只修改 TaskIR 中相关字段：

```text
你生成的 TaskIR 在 STL 验证中失败。
失败子句：G[0,T](distance_to_obstacle >= 10)
反例：t=14.2s 时距离 building_7 仅 6.4m。
请只修改 route constraint 或 safety margin，不要改变用户原始目标。
输出新的 TaskIR JSON。
```

该设计的重点是减少 LLM 的搜索空间，让修复行为可解释、可记录、可复现。

如果连续 $K_{\mathrm{LLM}}$ 轮 LLM 修复仍失败，则进入符号枚举 fallback。枚举范围由 TaskIR DSL 深度、地图实体集、允许约束模板和最大任务 horizon 限定。枚举器按诊断结果优先扩展最相关字段，例如安全距离、绕行侧、时间窗、目标顺序和 fallback landing pad。

### 4.5 轨迹生成

AAAI 版本使用轻量可复现轨迹生成器：

- 2D grid A*：用于基础 reach-avoid 与顺序任务。
- 3D grid A*：用于高度层与城市低空走廊。
- RRT*：用于连续空间补充验证。
- MPC-lite / trajectory smoothing：用于检查转弯半径、速度变化和高度变化是否满足简化动力学约束。

轨迹生成器不是本文的创新点。它的作用是把规格翻译问题推进到“可执行轨迹是否真的存在”的层面。

---

## 5. 理论性质与相对完备性证明

v1 只说“验证纠错能提升可靠性”，但没有数学边界。v2 将算法性质写清楚：VERA-UAV 不声称 LLM 本身完备，而是声称在有限 DSL、可判定验证器和完备底层规划器假设下具有 **relative completeness**。

### 5.1 形式化设定

令城市低空地图离散化为有限带权图：

$$
G=(V,E,w), \quad |V|<\infty, \quad |E|<\infty.
$$

每个节点 $v\in V$ 带有原子命题集合 $L(v)$，例如 `goal_A`、`building_7_margin`、`no_fly_zone`、`altitude_layer_3`。轨迹是有限序列：

$$
\tau = (v_0, v_1, \ldots, v_T), \quad (v_t,v_{t+1})\in E.
$$

TaskIR DSL 定义为有限语法：

$$
\mathcal{D}_{H,D} = \{\psi: \mathrm{depth}(\psi)\le D,\ \mathrm{horizon}(\psi)\le H,\ \mathrm{entities}(\psi)\subseteq \mathcal{E}(\mathcal{M})\}.
$$

编译器 $C$ 将 TaskIR 编译成 LTL/STL 规格：

$$
C(\psi)=(\varphi_{\mathrm{LTL}},\varphi_{\mathrm{STL}}).
$$

验证器 $V$ 判断候选轨迹是否满足规格：

$$
V(\tau, C(\psi)) =
\begin{cases}
\mathrm{PASS}, & \tau \models \varphi_{\mathrm{LTL}}\ \land\ \rho(\tau,\varphi_{\mathrm{STL}})>0,\\
\mathrm{FAIL}(\eta), & \text{otherwise},
\end{cases}
$$

其中 $\eta$ 是 counterexample、unsat core 或 robustness trace。

### 5.2 算法伪代码

```text
Algorithm VERA-UAV
Input: natural language x_NL, map M, initial state s0
Output: verified trajectory tau or UNSAT / NEED_CLARIFICATION

1: Q ← LLM_PROPOSE(x_NL, M)
2: Q ← TYPECHECK_AND_RANK(Q)
3: Visited ← ∅
4: for iter = 1 ... B do
5:     if Q has no unvisited candidate:
6:         Q ← Q ∪ SYMBOLIC_ENUMERATE_NEXT(D, H)
7:         if Q still has no unvisited candidate:
8:             return UNSAT
9:     ψ ← POP_UNVISITED(Q, Visited)
10:    Visited ← Visited ∪ {ψ}
11:    if ψ has missing entity or underspecified field:
12:        η ← type / grounding diagnostic
13:        Q ← Q ∪ REPAIR(ψ, η)
14:        if all remaining candidates require the same external information:
15:            return NEED_CLARIFICATION
16:        continue
17:    (φ_LTL, φ_STL) ← COMPILE(ψ)
18:    if compiler or syntax verifier fails:
19:        η ← compiler diagnostic
20:        Q ← Q ∪ REPAIR(ψ, η)
21:        continue
22:    τ ← COMPLETE_PLANNER(G, s0, φ_LTL, φ_STL)
23:    if τ exists and VERIFY(τ, φ_LTL, φ_STL) = PASS:
24:        return τ
25:    η ← counterexample / unsat core / robustness trace
26:    Q ← Q ∪ LLM_REPAIR(ψ, η)
27:    if LLM repair budget exhausted:
28:        Q ← Q ∪ SYMBOLIC_ENUMERATE(ψ, η, D, H)
29: return UNSAT
```

### 5.3 定理 1：终止性

**Theorem 1 (Termination).** 若 TaskIR DSL $\mathcal{D}_{H,D}$ 有限，且算法设置有限候选预算 $B$，则 VERA-UAV 必定在有限步内返回 verified trajectory、`UNSAT` 或 `NEED_CLARIFICATION`。

**Proof sketch.** 队列 $Q$ 中每次弹出一个未访问候选 TaskIR，并通过 `Visited` 避免重复展开。LLM repair 的最大轮数有限，符号枚举空间 $\mathcal{D}_{H,D}$ 有限，外层循环最多执行 $B$ 次。因此算法不可能无限运行。每个分支要么返回，要么进入下一轮有限循环。证毕。

### 5.4 定理 2：安全可靠性

**Theorem 2 (Soundness).** 若 VERA-UAV 返回轨迹 $\tau$，则在给定地图模型、监控器语义和轨迹离散化精度下，$\tau$ 满足编译后的 LTL/STL 规格：

$$
\tau \models \varphi_{\mathrm{LTL}}
\quad \text{and} \quad
\rho(\tau,\varphi_{\mathrm{STL}})>0.
$$

**Proof sketch.** 算法只有在第 23 行通过最终验证后才返回轨迹。最终验证包含 LTL 层验证和 STL robustness 检查。若任一检查失败，算法只会生成诊断并继续修复，不会返回该轨迹。因此所有返回轨迹都满足上述条件。证毕。

### 5.5 定理 3：相对完备性

**Theorem 3 (Relative Completeness).** 对于不需要外部澄清的任务实例，假设：

1. 用户意图存在一个等价或足够保真的 TaskIR $\psi^\star \in \mathcal{D}_{H,D}$；
2. 编译器 $C$ 对 $\mathcal{D}_{H,D}$ 内所有 TaskIR 都能生成语义保持的 LTL/STL 规格；
3. 底层 planner 对有限图 $G$ 上满足 $C(\psi)$ 的轨迹搜索是完备的；
4. 符号枚举器会在有限时间内枚举 $\mathcal{D}_{H,D}$ 中所有候选；
5. 最终验证器对 bounded LTL/STL 语义是可靠的。

若存在轨迹 $\tau^\star$ 满足 $C(\psi^\star)$，则在候选预算 $B \ge |\mathcal{D}_{H,D}|$ 时，VERA-UAV 最终会返回某条满足规格的轨迹 $\tau$。

**Proof sketch.** 根据假设 4，符号枚举 fallback 会枚举到 $\psi^\star$。根据假设 2，$C(\psi^\star)$ 保持语义。根据假设 3，底层 planner 会找到满足 $C(\psi^\star)$ 的轨迹。根据假设 5，最终验证器会接受该轨迹。根据算法第 23-24 行，算法会返回该轨迹。因此 VERA-UAV 在该有限 DSL 和模型假设下相对完备。证毕。

### 5.6 完备性边界

这一定理不是说 VERA-UAV 对现实世界中的任意自然语言和任意连续动力学绝对完备。它只说明：**只要目标任务能被有限 TaskIR DSL 表示，且底层搜索空间与验证语义覆盖了该任务，VERA-UAV 不依赖 LLM 必然生成正确答案，也能通过符号 fallback 找到可行解。**

这也是本文相对 Lang2LTL、LTLCodeGen、T3 Planner 的关键理论定位：LLM 是高效 proposal generator，而不是完备性来源。

---

## 6. 数据来源与数据集构建

### 6.1 主数据来源

AAAI 主实验使用程序生成城市 UAV grid/world 数据，不依赖 AirSim 或真实飞行数据。这样做有三个原因：

1. 可控：可以系统性生成可达、不可达、模糊、冲突、时间窗紧张等任务。
2. 可复现：地图、任务和随机种子可完全开源。
3. 适配 AAAI 篇幅：重点服务 AI 方法评估，而不是重型仿真工程。

### 6.2 地图生成

每个地图包含：

- 栅格尺寸：`50x50x3` 到 `100x100x5`。
- 语义对象：建筑、医院、学校、物流站、降落点、检查面、禁飞区。
- 空域结构：高度层、飞行走廊、临时封闭区。
- 动态元素：可选加入移动障碍或临时禁飞区。
- OSM 风格命名：如 `hospital_zone_2`、`building_7_east_face`，仅作为语义命名参考，不作为主实验依赖。

### 6.3 样本字段

每条样本包含：

| 字段 | 说明 |
|------|------|
| `instruction_id` | 样本编号 |
| `map_id` | 地图编号 |
| `natural_language_instruction` | 自然语言 UAV 任务 |
| `entity_annotations` | 地图实体与指令实体对齐 |
| `gold_task_ir` | 人工或规则生成的金标准 TaskIR |
| `gold_ltl` | 金标准 LTL |
| `gold_stl` | 金标准 STL |
| `satisfiability_label` | `SAT`, `UNSAT`, `NEED_CLARIFICATION` |
| `reference_trajectory` | 若 SAT，则给出一条可行轨迹 |
| `failure_type` | 若失败，标注失败类型 |
| `oracle_cost` | 最短路或最小代价轨迹成本 |

### 6.4 数据规模

v2 建议 AAAI 主实验规模：

| Split | 数量 | 用途 |
|------|------|------|
| Train / prompt pool | 800 | few-shot 示例、模板调试 |
| Dev | 250 | prompt、repair 策略、阈值选择 |
| Test | 400 | 最终报告 |
| Stress test | 150 | 长组合、模糊、不可满足、紧时间窗 |

测试集不能参与 prompt 选择。所有实验报告固定随机种子和任务列表。

### 6.5 数据生成协议与防泄漏

为了让 synthetic benchmark 在 AAAI 审稿中站得住，数据生成必须从第一天就按“可复现 benchmark”而不是“临时实验脚本”来管理：

1. **先冻结 generator，再生成 test**：地图生成器、任务模板、语言 paraphrase 规则和失败注入规则先在 dev 上调试，冻结 commit hash 后再生成 test / stress test。
2. **按地图级切分**：test 地图不能与 train / dev 共享同一个 `map_id`、实体坐标或障碍布局，只允许共享抽象任务类型。
3. **按实体命名级切分**：test 中至少 30% 任务使用训练集中未出现过的实体命名模式，例如 `clinic_zone`、`sky_corridor_E2`、`temporary_pad_17`。
4. **按模板组合级切分**：test 中保留一部分未见组合，例如 `ordered inspection + time window + emergency fallback`，避免模型只记住单模板映射。
5. **固定随机种子与 manifest**：每个 split 输出 `manifest.jsonl`，记录 generator version、seed、map hash、task template id、paraphrase id 和 satisfiability label。
6. **禁止 test prompt 污染**：few-shot 示例只能来自 train / prompt pool；dev 只用于阈值与 prompt 策略选择；test 和 stress test 只运行一次并锁定结果。

### 6.6 Failure taxonomy

每条失败样本要记录首个失败阶段和最终失败阶段，便于解释 VERA-UAV 到底修复了什么：

| Failure type | 定义 | 主要归因模块 |
|--------------|------|--------------|
| `syntax_error` | LTL/STL 无法解析或类型不匹配 | LLM / compiler |
| `entity_error` | 引用了不存在、歧义或错配的地图实体 | grounding |
| `semantic_miss` | 漏掉用户关键约束，如禁飞区或时间窗 | TaskIR generation |
| `unsat_missed` | gold 为 UNSAT，但系统返回可执行计划 | verifier / decision policy |
| `false_unsat` | gold 为 SAT，但系统错误输出 UNSAT | planner / search budget |
| `ltl_violation` | 离散时序顺序、到达、避让不满足 | planner / LTL compiler |
| `stl_violation` | 高度、距离、速度、时间窗 robustness 非正 | trajectory / STL monitor |
| `repair_regression` | 修复一个约束后破坏原本满足的约束 | repair loop |
| `timeout` | 超过预设推理或规划预算 | system budget |

最终论文中不只报告平均分，还要报告 failure taxonomy 的堆叠柱状图。这样即使总体提升不够大，也能证明方法在关键安全失败类型上有明确作用。

---

## 7. 实验平台与实现配置

### 7.1 硬件

当前按 4 张 RTX 4090、每张 24GB 显存设计。本研究不依赖闭源 API，主实验全部使用本地开源模型。

### 7.2 模型

主实验模型：

- Qwen3-8B：轻量本地模型 baseline [35]。
- Qwen3-14B：主模型 [35]。
- DeepSeek-R1-Distill-Qwen-14B：推理增强模型 [36]。

可选上限模型：

- 32B 量化模型，用作 appendix 或补充结果；不作为 AAAI 主结论必要条件。

本地推理使用 vLLM / PagedAttention 或 HuggingFace Transformers。vLLM 的 PagedAttention 设计适合多 prompt、多修复轮次下的吞吐实验 [37]。

### 7.3 软件模块

| 模块 | 候选工具 | 作用 |
|------|----------|------|
| LLM inference | vLLM / transformers | 本地模型推理 |
| LTL validation | Spot | LTL parsing、automata、可满足性分析 |
| STL monitoring | RTAMT 或自实现 monitor | STL robustness |
| Probabilistic checking | PRISM | 可选不确定环境验证 |
| Planning | A* / RRT* / MPC-lite | 轨迹生成 |
| Logging | JSONL + CSV | 记录每轮生成、验证和修复 |

### 7.4 运行记录

每个任务实例必须记录：

- 原始 instruction。
- 每轮 TaskIR、LTL、STL。
- 验证器输出。
- 修复 prompt。
- 最终轨迹。
- 运行时间、token 数、显存配置。
- baseline 与 VERA-UAV 在同一任务上的 paired comparison id。

这些记录服务 AAAI reproducibility checklist [38]。

### 7.5 预注册参数预算

为了避免实验后调参，本任务书建议在第一轮正式 test 前固定以下预算：

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `K_LLM` | 3 | 每个任务最多三轮 LLM repair |
| `B` | 256 | VERA-UAV 总候选 TaskIR 预算 |
| `D` | 4 | TaskIR DSL 最大嵌套深度 |
| `H` | 8 | 离散任务 horizon / 子目标上限 |
| `T_plan` | 30s | 单任务规划超时 |
| `T_llm` | 20s | 单轮 LLM 推理超时 |
| decoding temperature | 0.2 | 主实验低随机性；只在附录报告温度敏感性 |
| top-p | 0.9 | 与 temperature 共同固定 |
| max new tokens | 1024 | 防止不同模型输出长度差异影响 runtime |

若正式实验需要改动这些值，必须先在 dev 上记录原因，然后重新冻结配置。test 结果不能反向决定参数。

---

## 8. 对比实验设计

### 8.1 Baseline 列表

| 方法 | 描述 | 目的 |
|------|------|------|
| Direct LLM planning | LLM 直接输出 waypoint / action sequence | 检验纯文本规划是否不安全 |
| ReAct-style planning | reasoning-action loop，无形式化验证 | 对照通用 LLM agent planning [14] |
| SayCan-style affordance filtering | LLM score + feasible skill filter | 对照 affordance grounding [15] |
| Prompt-only NL-to-LTL/STL | LLM 直接输出 LTL/STL，无 typed IR 和验证修复 | 检验 prompt 工程上限 |
| NL2LTL-style template baseline | 基于模板匹配生成 LTL | 对照传统 template 方法 [2] |
| LTLCodeGen-style baseline | LLM 生成逻辑函数代码，再编译为 LTL | 对照语法正确性路线 [3] |
| T3-style self-correction | LLM + STL verifier，多轮自修复 | 对照最近直接竞争路线 [6] |
| VERA-UAV without repair | 使用 TaskIR 和验证，但失败后不修复 | 分离验证与修复贡献 |
| VERA-UAV LLM-only repair | typed IR + LLM repair，无符号枚举 fallback | 检验 fallback 对完备性的贡献 |
| VERA-UAV full | 完整 typed IR + verification + counterexample repair + symbolic fallback | 主方法 |

### 8.2 主实验

主实验回答五个问题：

1. VERA-UAV 是否比 baseline 更容易生成可执行计划？
2. VERA-UAV 是否降低安全违规率？
3. VERA-UAV 的 STL robustness 是否显著更高？
4. VERA-UAV 的修复轮数和额外推理开销是否可接受？
5. 符号枚举 fallback 是否真的提升了“失败任务恢复率”和相对完备性？

主结果表建议：

| Method | Syntax valid ↑ | Semantic F1 ↑ | ESS ↑ | FSR ↓ | Mean robustness ↑ | Optimality gap ↓ | Repair success ↑ | Runtime ↓ |
|--------|----------------|---------------|-------|-------|-------------------|------------------|------------------|-----------|
| Direct LLM | TBD | TBD | TBD | TBD | TBD | TBD | N/A | TBD |
| ReAct-style | TBD | TBD | TBD | TBD | TBD | TBD | N/A | TBD |
| SayCan-style | TBD | TBD | TBD | TBD | TBD | TBD | N/A | TBD |
| Prompt-only | TBD | TBD | TBD | TBD | TBD | TBD | N/A | TBD |
| NL2LTL-style | TBD | TBD | TBD | TBD | TBD | TBD | N/A | TBD |
| LTLCodeGen-style | TBD | TBD | TBD | TBD | TBD | TBD | N/A | TBD |
| T3-style | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| VERA-UAV no repair | TBD | TBD | TBD | TBD | TBD | TBD | 0 | TBD |
| VERA-UAV LLM-only repair | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| VERA-UAV full | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

表中 `TBD` 是实验待填数据，不得在任务书中伪造。

### 8.3 实验结果评定协议

v2 明确主指标与统计判定，避免后续“看哪个指标好就报哪个”的风险。

**Primary metric 1：Executable Safety Success（ESS）**

一个任务只有同时满足以下条件才计为 ESS=1：

- 生成的 TaskIR 无类型错误。
- LTL/STL 可编译。
- planner 找到轨迹。
- 最终轨迹通过 LTL 检查。
- STL robustness 为正。
- 没有碰撞、禁飞区进入、高度越界或时间窗失败。

**Primary metric 2：False Safe Rate（FSR）**

FSR 衡量系统把不安全或不可满足任务误判为安全可执行的比例：

$$
\mathrm{FSR} = \frac{\#\{\mathrm{unsafe\ but\ returned\ as\ executable}\}}{\#\{\mathrm{all\ returned\ executable}\}}.
$$

AAAI 论文中，FSR 应作为安全方向最关键的负指标。VERA-UAV 的主要卖点不是让所有任务都“有输出”，而是避免虚假安全。

**统计检验**

- 对 ESS、FSR、UNSAT detection 等二值指标，用 paired McNemar test。
- 对 robustness、optimality gap、runtime 等连续指标，用 paired bootstrap 95% CI 和 Wilcoxon signed-rank test。
- 多 baseline 比较使用 Holm-Bonferroni 校正。
- 结论只在 $p<0.05$ 且效应量达到预注册阈值时写入主文。

**成功判据**

AAAI 主结论成立的最低条件：

1. VERA-UAV full 的 ESS 显著高于 LTLCodeGen-style 和 T3-style baseline。
2. VERA-UAV full 的 FSR 显著低于所有 LLM-only baseline。
3. 去掉 STL robustness feedback 后，连续安全约束相关 failure 明显上升。
4. 符号 fallback 对 LLM 修复失败样本提供可测增益。

### 8.4 泛化实验

泛化维度：

- 未见地图。
- 未见实体命名。
- 自然语言 paraphrase。
- 更长时序组合。
- 更紧时间窗。
- 不可满足任务比例提升。

泛化实验重点报告 VERA-UAV 是否能识别不可满足或模糊任务，而不是输出错误轨迹。

### 8.5 Case study

至少准备三个可视化案例：

1. **语法修复案例**：LLM 输出不合法 STL，Spot/RTAMT 报错，系统修复。
2. **轨迹安全案例**：LTL 满足但 STL robustness 为负，系统绕行后转正。
3. **不可满足案例**：用户要求互相矛盾，系统输出 `UNSAT`。

### 8.6 AAAI 主文图表计划

AAAI 主文空间很紧，图表必须服务核心论证。建议主文只放五类图表，其他放 appendix：

| 图表 | 目标 | 放置位置 |
|------|------|----------|
| Figure 1：VERA-UAV pipeline | 一眼说明 typed IR、verification、repair、fallback 的闭环 | Method |
| Table 1：核心文献定位矩阵 | 证明本文不是简单 NL-to-LTL 应用 | Related Work |
| Table 2：主实验结果 | ESS、FSR、robustness、runtime 的 paired comparison | Experiments |
| Figure 2：failure taxonomy 堆叠图 | 说明方法主要减少哪些失败类型 | Experiments |
| Figure 3：case study 轨迹图 | 展示反例反馈如何把负 robustness 修到正 | Experiments / Appendix |

主文不建议放大段 prompt、完整 DSL grammar 或所有地图截图。这些内容应放 code/data appendix，以免挤占贡献论证。

---

## 9. 消融实验设计

| 消融项 | 变体 | 目的 |
|--------|------|------|
| 去掉 typed IR | Direct LTL/STL generation | 验证结构化中间表示是否提升可靠性 |
| 去掉 counterexample feedback | Generic retry | 验证反例是否比普通重试更有效 |
| 去掉 STL robustness feedback | LTL-only verification | 验证连续安全约束的重要性 |
| one-shot repair | 最多修复 1 次 | 评估修复轮数收益 |
| iterative repair | 最多修复 3 次 | 评估多轮修复上限 |
| 不同模型规模 | Qwen3-8B / Qwen3-14B / DeepSeek-R1-Distill-Qwen-14B | 评估模型能力与验证框架的关系 |
| 去掉 UNSAT 检测 | 强制生成轨迹 | 验证拒答能力对安全性的贡献 |
| 去掉符号 fallback | LLM-only repair | 验证相对完备性组件对失败恢复的贡献 |
| 去掉 planner final verification | 只验证公式不验证轨迹 | 证明执行闭环不是可选项 |

消融实验的核心不是“证明组件都有效”，而是找出哪些组件对 AAAI 审稿人最关心的安全与可执行性指标贡献最大。

---

## 10. 评价指标

### 10.1 规格生成指标

| 指标 | 定义 |
|------|------|
| Syntax validity | LTL/STL 是否可被 parser 接受 |
| Entity grounding accuracy | 指令实体是否正确映射到地图实体 |
| Semantic F1 | 生成 TaskIR 字段与 gold TaskIR 的 precision / recall / F1 |
| Semantic match | 生成规格是否与 gold TaskIR / gold formula 等价或近似等价 |
| UNSAT detection accuracy | 不可满足任务是否被正确识别 |
| Clarification accuracy | 模糊任务是否触发 `NEED_CLARIFICATION` |
| False executable rate | 不可满足或模糊任务被错误执行的比例 |

### 10.2 规划执行指标

| 指标 | 定义 |
|------|------|
| ESS | 同时满足语义、轨迹可行、LTL、STL、安全约束的任务比例 |
| FSR | 不安全任务被错误标为安全可执行的比例 |
| Mean STL robustness | 最终轨迹对 STL 规格的平均 robustness |
| Worst-case STL robustness | 每条轨迹最小 robustness 的分布 |
| Minimum safety margin | 轨迹中最小障碍距离 |
| Optimality gap | $(J(\tau)-J^\star)/J^\star$ |
| Path length / flight time | 轨迹代价与飞行时间 |

### 10.3 修复效率指标

| 指标 | 定义 |
|------|------|
| Repair success rate | 验证失败后修复成功比例 |
| Fail-to-pass conversion | 初始失败样本经修复后通过的比例 |
| Average repair rounds | 平均修复轮数 |
| Fallback contribution | LLM repair 失败但 symbolic fallback 成功的比例 |
| Runtime overhead | 修复机制带来的额外时间 |
| Token overhead | 修复 prompt 与诊断带来的 token 增量 |

### 10.4 指标计算细则

主实验需要在代码中直接实现以下指标，避免论文写作阶段再手工整理：

**Semantic F1**

将 TaskIR 展平成字段级约束集合 $\mathcal{C}$，例如 `reach(A)`、`avoid(zone_B)`、`time_window(A,30)`。设预测集合为 $\hat{\mathcal{C}}$，金标准集合为 $\mathcal{C}^\star$：

$$
P = \frac{|\hat{\mathcal{C}}\cap \mathcal{C}^\star|}{|\hat{\mathcal{C}}|}, \quad
R = \frac{|\hat{\mathcal{C}}\cap \mathcal{C}^\star|}{|\mathcal{C}^\star|}, \quad
F1 = \frac{2PR}{P+R}.
$$

**Safety violation rate**

$$
\mathrm{SVR} =
\frac{\#\{\tau: collision \lor nofly \lor altitude\_violation \lor \rho(\tau,\varphi_{\mathrm{STL}})\le 0\}}
{\#\{\mathrm{returned\ trajectories}\}}.
$$

**Optimality gap**

当 gold 或 oracle planner 能给出最优代价 $J^\star$ 时：

$$
\mathrm{Gap}(\tau)=\frac{J(\tau)-J^\star}{\max(J^\star,\epsilon)}.
$$

若任务为 UNSAT 或 NEED_CLARIFICATION，不计算 optimality gap，而单独计入识别准确率。

**Repair efficiency**

$$
\mathrm{FailToPass} =
\frac{\#\{\mathrm{initial\ fail,\ final\ pass}\}}
{\#\{\mathrm{initial\ fail}\}},
\quad
\mathrm{FallbackContribution} =
\frac{\#\{\mathrm{LLM\ repair\ fail,\ symbolic\ fallback\ pass}\}}
{\#\{\mathrm{final\ pass}\}}.
$$

这些公式应在实验脚本中输出为机器可读 CSV 字段，并在论文表格中只做格式化展示。

---

## 11. 预期实验结论

本节为预注册预期，不是实验结果。

### 11.1 主预期

预期 VERA-UAV full 在 ESS 上高于所有 baseline，并在 FSR / safety violation rate 上更低。原因是 baseline 通常只优化语言到规格的局部正确性，而 VERA-UAV 把“规格是否能产生安全轨迹”纳入闭环。

### 11.2 反例反馈预期

预期 counterexample feedback 显著降低不可执行计划比例。与 generic retry 相比，结构化反例能告诉 LLM 哪个子句、哪个时刻、哪个实体导致失败，从而减少无方向重试。

### 11.3 Typed IR 预期

预期 typed IR 提高语义一致性和可解释性。直接生成 LTL/STL 容易出现括号、运算符、实体引用和约束遗漏问题；TaskIR 将这些错误提前暴露为字段缺失或类型错误。

### 11.4 STL robustness 预期

预期 STL robustness feedback 对连续安全约束最关键。LTL 层可以证明“最终到达”和“避免禁飞区”等离散性质，但不能充分表达飞行高度、最小距离和时间窗裕度。STL robustness 能提供量化安全边界，是 UAV 区别于普通地面导航任务的关键点。

### 11.5 模型规模预期

预期更强本地模型能提升初始 TaskIR 质量，但验证修复框架对较小模型也有帮助。换句话说，本文不应把贡献写成“某个大模型更强”，而应写成“验证纠错机制提升不同开源模型的可靠性”。

---

## 12. 自审发现的问题与 v2 修复

### 12.1 v1 的主要问题

1. **文献覆盖不足**：v1 只列出 12 条参考文献，不足以支撑 AAAI 顶会定位。
2. **新颖性边界不够锐利**：v1 容易被理解为“UAV 版本 NL-to-LTL”，与 Lang2LTL、LTLCodeGen 差异不够强。
3. **实验指标不够可判定**：v1 只列出一般指标，没有定义 ESS、FSR、统计检验和成功判据。
4. **完备性表述过弱**：v1 没有说明算法为什么不是纯 heuristic。
5. **synthetic data 风险没有充分缓解**：v1 没有说明为什么合成数据仍能支持 AAAI 方法论结论。

### 12.2 v2 的修复策略

1. 扩展到 30+ 篇强相关文献，并用文献矩阵明确每篇与本文的关系。
2. 将贡献从“翻译”收窄为“执行闭环 + STL robustness + 反例修复 + 相对完备 fallback”。
3. 定义 ESS、FSR、optimality gap、fail-to-pass conversion 等可复现指标。
4. 给出终止性、安全可靠性和相对完备性定理，明确完备性来自有限 DSL 与符号枚举，而不是 LLM。
5. 把 AirSim/真实物流放到 T-ITS 扩展，AAAI 主文坚持 synthetic controlled benchmark 的方法论定位。

### 12.3 2026-05-19 二次自审补强

本轮继续审阅后，认为 Paper E 还存在四个容易被审稿人追问的问题，并已在任务书中补齐对应约束：

1. **数据可信度**：只说“程序生成数据”不够，需要明确 generator freeze、地图级切分、实体命名级切分和 test prompt 防污染。
2. **失败解释力**：只报告 ESS / FSR 不够，需要记录 failure taxonomy，证明方法减少的是安全相关失败，而不是只提升平均分。
3. **可复现参数**：只说使用 Qwen3 / DeepSeek 不够，需要固定 repair 轮数、候选预算、DSL 深度、规划超时和 decoding 参数。
4. **论文呈现策略**：AAAI 篇幅有限，需要提前确定主文图表，否则容易把主线写散。

这四点不改变 VERA-UAV 的核心贡献，但能把任务书从“想法路线”推进到“可以直接组织实验和论文”的状态。

### 12.4 2026-05-23 整理：AAAI 主线收束

Paper E 当前应当优先收束为 **AAAI / IJCAI 方法论文**，不要提前写成完整 ITS 系统论文。核心问题是：LLM 生成的 UAV 任务规划如何经过 typed IR、时序逻辑验证、反例修复和符号 fallback，变成可执行、可验证、可解释的轨迹方案。

第一版论文只保留三条贡献：

1. **Typed TaskIR**：把自然语言 UAV 指令转换成实体、动作、时序约束、安全约束和资源约束均可检查的中间表示。
2. **LTL/STL + verifier + trajectory closure**：不仅验证公式语法，还验证规格能否生成满足安全约束的轨迹。
3. **Counterexample / robustness repair with finite DSL fallback**：利用反例、unsat core 和 STL robustness 反馈修复；当 LLM 无法修复时，用有限 DSL 枚举给出 relative completeness。

主文中不要提前承诺以下内容：

- 不做完整多 UAV 交通管理；
- 不做真实物流系统部署；
- 不把 AirSim 高保真仿真作为主实验依赖；
- 不把 ITS 政策或低空经济系统启示写成 AAAI 主贡献。

最小实验矩阵建议冻结为：

| 维度 | 第一版设置 |
|------|------------|
| 任务族 | patrol、delivery、inspection、avoidance、temporal ordering、UNSAT / ambiguous |
| 地图 | 程序生成 city grid / obstacle / no-fly-zone / charging-point |
| baselines | Direct LLM planning、ReAct / prompt-only、NL2LTL-style、LTLCodeGen-style、VERA-UAV no repair、VERA-UAV full |
| 主指标 | ESS、FSR、safety violation rate、repair success、fail-to-pass conversion、runtime |
| 消融 | no typed IR、no counterexample、no STL robustness、one-shot vs iterative repair、no symbolic fallback |
| 泛化 | unseen map、unseen entity naming、longer horizon、harder constraints、UNSAT detection |

T-ITS 扩展可以放在后续版本：接入 Paper B 的 fleet scheduling、Paper F 的 stress scenarios 和低空交通系统指标。但 AAAI 版本必须保持问题干净，否则会同时被 AI 审稿人和交通审稿人追问边界。

---

## 13. 风险与备选方案

| 风险 | 影响 | 备选方案 |
|------|------|----------|
| 新颖性被认为只是 NL-to-LTL 应用 | AAAI 拒稿风险高 | 强调 STL robustness、反例修复、可执行轨迹闭环 |
| LTLCodeGen baseline 太强 | 主结果优势不足 | 用 UAV 连续约束和不可满足检测作为差异化指标 |
| 本地模型能力不足 | 初始翻译质量低 | 使用 Qwen3-14B / DeepSeek-R1-Distill-Qwen-14B，并报告修复收益 |
| 数据集被认为过于合成 | 应用可信度不足 | 加入 OSM 风格命名、真实城市区块布局统计，但不依赖真实飞行 |
| 修复轮数导致 runtime 过高 | 实时性受质疑 | 报告 one-shot 和最多三轮修复，设置超时和 fallback |
| STL monitor 实现复杂 | 影响进度 | 先实现离散时间 STL 子集，再接 RTAMT |
| AAAI 篇幅不足 | 故事发散 | 主文只放方法和核心实验，ITS 扩展放附录计划 |
| AAAI 对 LLM 生成文本政策敏感 | 论文写作合规风险 | 最终投稿文本必须由作者人工重写和审校，LLM 输出只作为实验对象或内部写作辅助，不把未审校生成文本直接作为论文正文 [38] |
| 相对完备性被认为假设过强 | 理论贡献被削弱 | 在主文明确写成 relative completeness，并把有限 DSL、bounded horizon、complete planner 作为 theorem assumptions，而非现实世界绝对保证 |
| Stress test 过难导致主结果下降 | 平均指标不好看 | 主 test 与 stress test 分开报告，stress test 用于分析鲁棒边界，不与主结论混在同一平均值中 |

---

## 14. 参考文献

[1] Jason Xinyu Liu, Ziyi Yang, Ifrah Idrees, Sam Liang, Benjamin Schornstein, Stefanie Tellex, and Ankit Shah. “Grounding Complex Natural Language Commands for Temporal Tasks in Unseen Environments.” *Proceedings of The 7th Conference on Robot Learning*, PMLR 229:1084-1110, 2023. URL: <https://proceedings.mlr.press/v229/liu23d.html>

[2] Francesco Fuggitti and Tathagata Chakraborti. “NL2LTL -- a Python Package for Converting Natural Language (NL) Instructions to Linear Temporal Logic (LTL) Formulas.” *Proceedings of the AAAI Conference on Artificial Intelligence*, 37(13):16428-16430, 2023. DOI: 10.1609/aaai.v37i13.27068. URL: <https://ojs.aaai.org/index.php/AAAI/article/view/27068>

[3] Behrad Rabiei, Mahesh Kumar A. R., Zhirui Dai, Surya L. S. R. Pilla, Qiyue Dong, and Nikolay Atanasov. “LTLCodeGen: Code Generation of Syntactically Correct Temporal Logic for Robot Task Planning.” arXiv:2503.07902, 2025; project page reports IROS 2025. URL: <https://arxiv.org/abs/2503.07902>; <https://existentialrobotics.org/LTLCodeGen/>

[4] Jun Wang, David Smith Sundarsingh, Jyotirmoy V. Deshmukh, and Yiannis Kantaros. “ConformalNL2LTL: Translating Natural Language Instructions into Temporal Logic Formulas with Conformal Correctness Guarantees.” arXiv:2504.21022, 2025. URL: <https://arxiv.org/abs/2504.21022>

[5] Licheng Luo, Kaier Liang, Yu Xia, and Mingyu Cai. “NL2SpaTiaL: Generating Geometric Spatio-Temporal Logic Specifications from Natural Language for Manipulation Tasks.” arXiv:2512.13670, 2025; revised 2026. URL: <https://arxiv.org/abs/2512.13670>

[6] Jia Li and Guoxiang Zhao. “T3 Planner: A Self-Correcting LLM Framework for Robotic Motion Planning with Temporal Logic.” arXiv:2510.16767, 2025. URL: <https://arxiv.org/abs/2510.16767>

[7] Simon Sinong Zhan, Yao Liu, Philip Wang, Zinan Wang, Qineng Wang, Zhian Ruan, Xiangyu Shi, Xinyu Cao, Frank Yang, Kangrui Wang, Huajie Shao, Manling Li, and Qi Zhu. “SENTINEL: A Multi-Level Formal Framework for Safety Evaluation of LLM-based Embodied Agents.” arXiv:2510.12985, 2025. URL: <https://arxiv.org/abs/2510.12985>

[8] Anand Gokhale, Vaibhav Srivastava, and Francesco Bullo. “LogicGuard: Improving Embodied LLM agents through Temporal Logic based Critics.” arXiv:2507.03293, 2025. URL: <https://arxiv.org/abs/2507.03293>

[9] Haoyu Wang, Christopher M. Poskitt, Jun Sun, and Jiali Wei. “Pro2Guard: Proactive Runtime Enforcement of LLM Agent Safety via Probabilistic Model Checking.” arXiv:2508.00500, 2025. URL: <https://arxiv.org/abs/2508.00500>

[10] Tom Silver, Soham Dan, Kavitha Srinivas, Joshua B. Tenenbaum, Leslie Kaelbling, and Michael Katz. “Generalized Planning in PDDL Domains with Pretrained Large Language Models.” *Proceedings of the AAAI Conference on Artificial Intelligence*, 38(18):20256-20264, 2024. DOI: 10.1609/aaai.v38i18.30006. URL: <https://ojs.aaai.org/index.php/AAAI/article/view/30006>

[11] Karthik Valmeekam, Matthew Marquez, Sarath Sreedharan, and Subbarao Kambhampati. “On the Planning Abilities of Large Language Models: A Critical Investigation.” *Advances in Neural Information Processing Systems*, 2023. URL: <https://arxiv.org/abs/2305.15771>

[12] Bo Liu, Yuqian Jiang, Xiaohan Zhang, Qiang Liu, Shiqi Zhang, Joydeep Biswas, and Peter Stone. “LLM+P: Empowering Large Language Models with Optimal Planning Proficiency.” arXiv:2304.11477, 2023. URL: <https://arxiv.org/abs/2304.11477>

[13] Karthik Valmeekam, Matthew Marquez, Alberto Olmo, Sarath Sreedharan, and Subbarao Kambhampati. “PlanBench: An Extensible Benchmark for Evaluating Large Language Models on Planning and Reasoning about Change.” *Advances in Neural Information Processing Systems, Datasets and Benchmarks Track*, 2023. URL: <https://openreview.net/forum?id=YXogl4uQUO>

[14] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. “ReAct: Synergizing Reasoning and Acting in Language Models.” *International Conference on Learning Representations (ICLR)*, 2023. URL: <https://openreview.net/forum?id=WE_vluYUL-X>

[15] Michael Ahn et al. “Do As I Can, Not As I Say: Grounding Language in Robotic Affordances.” *Conference on Robot Learning (CoRL)*, 2022. URL: <https://arxiv.org/abs/2204.01691>

[16] Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng. “Code as Policies: Language Model Programs for Embodied Control.” *IEEE International Conference on Robotics and Automation (ICRA)*, 2023. URL: <https://arxiv.org/abs/2209.07753>

[17] Ishika Singh, Valts Blukis, Arsalan Mousavian, Ankit Goyal, Danfei Xu, Jonathan Tremblay, Dieter Fox, Jesse Thomason, and Animesh Garg. “ProgPrompt: Generating Situated Robot Task Plans using Large Language Models.” *IEEE International Conference on Robotics and Automation (ICRA)*, 2023; extended version in *Autonomous Robots*, 2023. URL: <https://arxiv.org/abs/2209.11302>

[18] Hadas Kress-Gazit, Georgios E. Fainekos, and George J. Pappas. “Temporal-Logic-Based Reactive Mission and Motion Planning.” *IEEE Transactions on Robotics*, 25(6):1370-1381, 2009. DOI: 10.1109/TRO.2009.2030225.

[19] Hadas Kress-Gazit, Morteza Lahijanian, and Vasumathi Raman. “Synthesis for Robots: Guarantees and Feedback for Robot Behavior.” *Annual Review of Control, Robotics, and Autonomous Systems*, 1:211-236, 2018. DOI: 10.1146/annurev-control-060117-105838.

[20] Oded Maler and Dejan Nickovic. “Monitoring Temporal Properties of Continuous Signals.” *FORMATS/FTRTFT*, 2004. DOI: 10.1007/978-3-540-30206-3_12.

[21] Georgios E. Fainekos and George J. Pappas. “Robustness of Temporal Logic Specifications for Continuous-Time Signals.” *Theoretical Computer Science*, 410(42):4262-4291, 2009. DOI: 10.1016/j.tcs.2009.06.021.

[22] Alexandre Donze and Oded Maler. “Robust Satisfaction of Temporal Logic over Real-Valued Signals.” *FORMATS*, 2010. DOI: 10.1007/978-3-642-15297-9_12.

[23] Vasumathi Raman, Alexandre Donze, Dorsa Sadigh, Richard M. Murray, and Sanjit A. Seshia. “Reactive Synthesis from Signal Temporal Logic Specifications.” *Hybrid Systems: Computation and Control (HSCC)*, 2015. DOI: 10.1145/2728606.2728628.

[24] Shromona Ghosh, Dorsa Sadigh, Pierluigi Nuzzo, Vasumathi Raman, Alexandre Donze, Alberto L. Sangiovanni-Vincentelli, and Sanjit A. Seshia. “Diagnosis and Repair for Synthesis from Signal Temporal Logic Specifications.” *Hybrid Systems: Computation and Control (HSCC)*, 2016. DOI: 10.1145/2883817.2883847.

[25] Alexandre Duret-Lutz, Alexandre Lewkowicz, Amaury Fauchille, Thibault Michaud, Étienne Renault, and Laurent Xu. “Spot 2.0 -- A Framework for LTL and omega-Automata Manipulation.” *Automated Technology for Verification and Analysis (ATVA)*, 2016. URL: <https://spot.lre.epita.fr/>

[26] Tomoya Yamaguchi, Bardh Hoxha, and Dejan Nickovic. “RTAMT -- Runtime Robustness Monitors with Application to CPS and Robotics.” *International Journal on Software Tools for Technology Transfer*, 26(1):79-99, 2024; arXiv:2501.18608, 2025. DOI: 10.1007/S10009-023-00720-3. URL: <https://arxiv.org/abs/2501.18608>; code: <https://github.com/nickovic/rtamt>

[27] Marta Kwiatkowska, Gethin Norman, and David Parker. “PRISM 4.0: Verification of Probabilistic Real-time Systems.” *Computer Aided Verification (CAV)*, 2011. URL: <https://www.prismmodelchecker.org/bibitem.php?key=KNP11>

[28] Mohammed Alshiekh, Roderick Bloem, Rüdiger Ehlers, Bettina Könighofer, Scott Niekum, and Ufuk Topcu. “Safe Reinforcement Learning via Shielding.” *Proceedings of the AAAI Conference on Artificial Intelligence*, 2018. URL: <https://ojs.aaai.org/index.php/AAAI/article/view/11797>

[29] Edwin Hamel-De le Court, Francesco Belardinelli, and Alexander W. Goodall. “Probabilistic Shielding for Safe Reinforcement Learning.” *Proceedings of the AAAI Conference on Artificial Intelligence*, 39(15):16091-16099, 2025. DOI: 10.1609/aaai.v39i15.33767. URL: <https://ojs.aaai.org/index.php/AAAI/article/view/33767>

[30] Shubo Liu, Hongsheng Zhang, Yuankai Qi, Peng Wang, Yanning Zhang, and Qi Wu. “AerialVLN: Vision-and-Language Navigation for UAVs.” *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2023, pp. 15384-15394. URL: <https://openaccess.thecvf.com/content/ICCV2023/html/Liu_AerialVLN_Vision-and-Language_Navigation_for_UAVs_ICCV_2023_paper.html>

[31] Xiangyu Wang, Donglin Yang, Ziqin Wang, Hohin Kwan, Jinyu Chen, Wenjun Wu, Hongsheng Li, Yue Liao, and Si Liu. “Towards Realistic UAV Vision-Language Navigation: Platform, Benchmark, and Methodology.” *International Conference on Learning Representations (ICLR)*, 2025. URL: <https://openreview.net/forum?id=rUvCIvI4eB>; arXiv:2410.07087.

[32] Sourav Sanyal and Kaushik Roy. “ASMA: An Adaptive Safety Margin Algorithm for Vision-Language Drone Navigation via Scene-Aware Control Barrier Functions.” arXiv:2409.10283, 2024; accepted by *IEEE Robotics and Automation Letters*. URL: <https://arxiv.org/abs/2409.10283>

[33] Xinyuan Zhang, Yonglin Tian, Fei Lin, Yue Liu, Jing Ma, Kornélia Sára Szatmáry, and Fei-Yue Wang. “LogisticsVLN: Vision-Language Navigation For Low-Altitude Terminal Delivery Based on Agentic UAVs.” arXiv:2505.03460, 2025. URL: <https://arxiv.org/abs/2505.03460>

[34] Hanxuan Chen, Jie Zheng, Siqi Yang, Tianle Zeng, Siwei Feng, Songsheng Cheng, Ruilong Ren, Hanzhong Guo, Shuai Yuan, Xiangyue Wang, Kangli Wang, and Ji Pei. “Vision-and-Language Navigation for UAVs: Progress, Challenges, and a Research Roadmap.” arXiv:2604.13654, 2026. URL: <https://arxiv.org/abs/2604.13654>

[35] Qwen Team. “Qwen3 Technical Report.” arXiv:2505.09388, 2025. URL: <https://arxiv.org/abs/2505.09388>

[36] DeepSeek-AI. “DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.” arXiv:2501.12948, 2025. URL: <https://arxiv.org/abs/2501.12948>

[37] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. “Efficient Memory Management for Large Language Model Serving with PagedAttention.” *ACM Symposium on Operating Systems Principles (SOSP)*, 2023. URL: <https://arxiv.org/abs/2309.06180>

[38] AAAI. “AAAI-26 Main Technical Track: Call for Papers” and “AAAI-26 Reproducibility Checklist.” 2025. URL: <https://aaai.org/conference/aaai/aaai-26/main-technical-track-call/>; <https://aaai.org/conference/aaai/aaai-26/reproducibility-checklist/>

[39] IEEE Intelligent Transportation Systems Society. “IEEE Transactions on Intelligent Transportation Systems (T-ITS): Scope.” URL: <https://ieee-itss.org/pub/t-its/>

---

## 15. 附录：当前 AAAI 优先推进计划

### 15.1 论文定位

AAAI 版先收窄为一篇 AI 方法论文：

**Counterexample-Guided Verified Language-to-STL Planning for UAVs**

核心不是“LLM 会规划无人机”，而是：本地开源 LLM 生成 UAV 任务规格后，用形式化验证器产生反例诊断，再驱动 LLM 修正规格或计划，最终生成可验证轨迹。

### 15.2 AAAI 贡献表述

AAAI 主张三点贡献：

1. 一个 typed IR 到 LTL/STL 的 UAV 任务规格编译链，覆盖到达、避让、顺序、巡检、时间窗、高度和距离约束。
2. 一个 verification-guided repair loop，将语法错误、grounding 缺失、不可满足、轨迹不安全、STL robustness 过低都转成结构化反例反馈。
3. 一个 UAV-NL2STL benchmark，包含自然语言任务、地图、金标准规格、可执行轨迹和失败诊断标签。

### 15.3 时间线

| 时间 | 任务 | 产出 |
|------|------|------|
| 2026-05-18 至 2026-05-24 | 完成核心文献表、冻结 benchmark schema | Related work table + dataset spec |
| 2026-05-25 至 2026-06-07 | 实现地图/任务生成器、gold TaskIR/LTL/STL 模板、基础 planner | 数据生成脚本 + baseline planner |
| 2026-06-08 至 2026-06-21 | 实现 Spot/RTAMT 验证器和 counterexample feedback | verifier module |
| 2026-06-22 至 2026-07-05 | 跑本地模型、baseline、no-repair/full repair 初实验 | 初版主结果表 |
| 2026-07-06 至 2026-07-19 | 主实验、消融、泛化、失败案例统计 | 完整实验表和图 |
| 2026-07-20 至 AAAI 摘要截止 | 完成 abstract、introduction、method、figure 1、主结果表 | AAAI 初稿 |
| AAAI 全文截止前 | 压缩到 7 页，补 appendix、reproducibility、匿名仓库 | 投稿包 |

截至 2026-05-19，尚未在 AAAI 官网检索到 AAAI-27 Main Technical Track 官方 CFP；当前仍优先以 AAAI-26 Main Technical Track 的 7 页技术内容、reproducibility checklist 和 code/data appendix 要求作为倒排依据 [38]。一旦 AAAI-27 CFP 发布，需要第一时间更新本时间线，尤其是摘要截止、全文截止、supplementary material 截止和 LLM 生成文本政策。

### 15.4 后续 T-ITS 扩展

AAAI 后扩展成 T-ITS 时，新增内容必须明显区别于会议版。建议新增：

- AirSim / SUMO 或低空物流数字孪生实验。
- 多 UAV 协同和空域冲突仲裁。
- 交通系统指标：任务吞吐量、空域占用、安全裕度、配送/巡检完成率、通信延迟鲁棒性。
- 边缘部署实验：4-bit / 8-bit 模型在 Jetson 或 4090 上的 latency-energy trade-off。
- 标题从 AAAI 的 “verified planning method” 改为 “safe low-altitude UAV operation for intelligent transportation systems”。

---

**版本说明：** 本文内容已更新为 `v2`，但文件名沿用 `v1-20260517` 以满足本轮“直接在 V1 版本上修改”的要求。2026-05-19 的增量优化补充了数据防泄漏、failure taxonomy、参数预算、指标公式、图表计划和 AAAI 合规风险。下一版建议在完成数据集 schema 和第一轮 baseline 跑通后更新为 `v3-YYYYMMDD`，重点替换 `TBD` 表格、补充真实实验结果和失败案例。
