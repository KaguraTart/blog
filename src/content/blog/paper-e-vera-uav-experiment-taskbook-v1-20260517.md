---
title: "Paper E 实验任务书 v1：面向 AAAI 的验证纠错式 UAV 语言规划"
description: "完整记录 Paper E 的研究背景、真实可引用相关工作、算法设计、数据来源、对比实验、消融实验、预期结论与 AAAI 优先推进计划。"
pubDate: 2026-05-17
tags: ["Paper E", "AAAI", "UAV", "LLM", "LTL", "STL", "形式化验证", "实验任务书"]
category: Tech
---

# Paper E 实验任务书 v1：面向 AAAI 的验证纠错式 UAV 语言规划

> 本文是 Paper E 的实验任务书，不是最终论文草稿。目标是把“LLM + 形式化验证 + UAV 任务规划”收敛成一条可以优先冲刺 AAAI 主会的实验路线，并为后续扩展 IEEE Transactions on Intelligent Transportation Systems（T-ITS）期刊论文预留系统级空间。

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

VERA-UAV 的核心假设是：**如果验证器能够把这些错误转化为结构化反例和 robustness feedback，本地开源 LLM 的修正成功率会显著高于纯 prompt 重试。**

---

## 3. 相关工作与可引用论文

### 3.1 自然语言到时序逻辑

Lang2LTL 是最接近本研究的问题起点。Liu 等人在 CoRL 2023 提出模块化系统，将自然语言导航命令 grounding 到 LTL，并在 21 个 city-scaled environments 和真实机器人环境中验证泛化能力 [1]。该工作证明了“自然语言命令 → LTL 任务规格 → 机器人执行”的路线可行，但主要处理地面导航中的 LTL 任务，不强调连续 UAV 轨迹的 STL robustness，也没有把验证失败的 counterexample 作为 LLM 修复信号。

NL2LTL 是 Fuggitti 和 Chakraborti 在 AAAI 2023 Demonstrations 中发布的 Python 包，支持将自然语言输入转为 LTL 公式，并基于 DECLARE templates 提供可扩展模式 [2]。该工作更偏工具接口与工作流管理场景，给本文提供了 template baseline 的参考。

LTLCodeGen 是 Rabiei 等人在 2025 年提出的机器人任务规划方法，使用 LLM 生成预定义逻辑函数形式的代码，再得到语法正确的 LTL 公式，并结合 semantic occupancy map 与 motion planning 生成路径 [3]。它是本文最强的直接 baseline 之一。本文与 LTLCodeGen 的差异在于：我们不仅关注 LTL 语法正确性和路径生成，还关注 UAV 连续安全约束、STL robustness、反例诊断和迭代修复。

ConformalNL2LTL 提出用 conformal prediction 为 NL-to-LTL 翻译提供用户指定成功率保证 [4]。该工作说明“翻译可信度”本身已经成为前沿问题。本文不直接复现其 conformal guarantee，而将其作为 uncertainty-aware translation 的重要参照，并在实验中比较“拒答/请求澄清”能力。

### 3.2 形式化验证与运行时监控工具

Spot 是一个用于 LTL、omega-automata manipulation 和 model checking 的 C++/Python 工具库，可用于 LTL 语法检查、公式转换与自动机分析 [5]。本文计划用 Spot 检查 LTL 公式可解析性、构造 automata，并作为 LTL 层错误诊断工具。

RTAMT 是面向 Signal Temporal Logic 的 runtime robustness monitoring 工具，可计算 STL 规格在离散或连续信号轨迹上的定量 robustness [6]。这正好对应 UAV 轨迹中的高度、距离、时间窗和安全裕度约束。

PRISM 是经典 probabilistic model checker，用于带概率和实时性质的系统验证 [7]。AAAI 主实验中 PRISM 不作为必需模块，但可在不确定动态障碍或传感器噪声场景中作为可选扩展。

### 3.3 LLM 具身规划与机器人任务执行

ReAct 将 reasoning 和 acting 交替组织，使 LLM 能够在环境反馈中逐步推理和行动 [8]。SayCan 将语言模型的任务偏好与机器人 affordance 结合，筛选物理上可执行的动作 [9]。Code as Policies 则把语言模型输出为可执行机器人控制代码 [10]。这些工作说明 LLM 可以作为高层任务分解器，但它们本身不提供形式化安全保证。

### 3.4 投稿与期刊延展约束

AAAI-26 Main Technical Track 官方说明要求主文最多 7 页技术内容，并要求作者完成 reproducibility checklist [11]。因此 AAAI 版本必须聚焦方法、核心实验和可复现性，不能展开过多系统工程内容。

T-ITS 的 scope 覆盖现代交通系统中的 sensing、communications、controls、planning、design 和 implementation，也覆盖 Artificial Intelligence 等方法方向，并要求期刊扩展相对会议论文有明确新增贡献 [12]。因此，后续 ITS 期刊版本应增加城市低空交通系统指标，例如空域利用率、任务吞吐量、多 UAV 协同、通信延迟和运行安全收益。

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
Spot / RTAMT verification
        ↓
Counterexample + robustness feedback
        ↓
Local LLM repair loop
        ↓
A* / RRT* / MPC-lite trajectory generation
        ↓
Final trajectory verification
        ↓
Executable trajectory or UNSAT / NEED_CLARIFICATION
```

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

### 4.5 轨迹生成

AAAI 版本使用轻量可复现轨迹生成器：

- 2D grid A*：用于基础 reach-avoid 与顺序任务。
- 3D grid A*：用于高度层与城市低空走廊。
- RRT*：用于连续空间补充验证。
- MPC-lite / trajectory smoothing：用于检查转弯半径、速度变化和高度变化是否满足简化动力学约束。

轨迹生成器不是本文的创新点。它的作用是把规格翻译问题推进到“可执行轨迹是否真的存在”的层面。

---

## 5. 数据来源与数据集构建

### 5.1 主数据来源

AAAI 主实验使用程序生成城市 UAV grid/world 数据，不依赖 AirSim 或真实飞行数据。这样做有三个原因：

1. 可控：可以系统性生成可达、不可达、模糊、冲突、时间窗紧张等任务。
2. 可复现：地图、任务和随机种子可完全开源。
3. 适配 AAAI 篇幅：重点服务 AI 方法评估，而不是重型仿真工程。

### 5.2 地图生成

每个地图包含：

- 栅格尺寸：`50x50x3` 到 `100x100x5`。
- 语义对象：建筑、医院、学校、物流站、降落点、检查面、禁飞区。
- 空域结构：高度层、飞行走廊、临时封闭区。
- 动态元素：可选加入移动障碍或临时禁飞区。
- OSM 风格命名：如 `hospital_zone_2`、`building_7_east_face`，仅作为语义命名参考，不作为主实验依赖。

### 5.3 样本字段

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

### 5.4 数据规模

v1 任务书建议 AAAI 主实验规模：

| Split | 数量 | 用途 |
|------|------|------|
| Train / prompt pool | 700 | few-shot 示例、模板调试 |
| Dev | 200 | prompt 与 repair 策略选择 |
| Test | 300 | 最终报告 |
| Stress test | 100 | 长组合、模糊、不可满足任务 |

测试集不能参与 prompt 选择。所有实验报告固定随机种子和任务列表。

---

## 6. 实验平台与实现配置

### 6.1 硬件

当前按 4 张 RTX 4090、每张 24GB 显存设计。本研究不依赖闭源 API，主实验全部使用本地开源模型。

### 6.2 模型

主实验模型：

- Qwen3-8B：轻量本地模型 baseline。
- Qwen3-14B：主模型。
- DeepSeek-R1-Distill-Qwen-14B：推理增强模型。

可选上限模型：

- 32B 量化模型，用作 appendix 或补充结果；不作为 AAAI 主结论必要条件。

### 6.3 软件模块

| 模块 | 候选工具 | 作用 |
|------|----------|------|
| LLM inference | vLLM / transformers | 本地模型推理 |
| LTL validation | Spot | LTL parsing、automata、可满足性分析 |
| STL monitoring | RTAMT 或自实现 monitor | STL robustness |
| Probabilistic checking | PRISM | 可选不确定环境验证 |
| Planning | A* / RRT* / MPC-lite | 轨迹生成 |
| Logging | JSONL + CSV | 记录每轮生成、验证和修复 |

### 6.4 运行记录

每个任务实例必须记录：

- 原始 instruction。
- 每轮 TaskIR、LTL、STL。
- 验证器输出。
- 修复 prompt。
- 最终轨迹。
- 运行时间、token 数、显存配置。

这些记录服务 AAAI reproducibility checklist [11]。

---

## 7. 对比实验设计

### 7.1 Baseline 列表

| 方法 | 描述 | 目的 |
|------|------|------|
| Direct LLM planning | LLM 直接输出 waypoint / action sequence | 检验纯文本规划是否不安全 |
| Prompt-only NL-to-LTL/STL | LLM 直接输出 LTL/STL，无 typed IR 和验证修复 | 检验 prompt 工程上限 |
| NL2LTL-style template baseline | 基于模板匹配生成 LTL | 对照传统 template 方法 [2] |
| LTLCodeGen-style baseline | LLM 生成逻辑函数代码，再编译为 LTL | 对照语法正确性路线 [3] |
| VERA-UAV without repair | 使用 TaskIR 和验证，但失败后不修复 | 分离验证与修复贡献 |
| VERA-UAV full | 完整 typed IR + verification + counterexample repair | 主方法 |

### 7.2 主实验

主实验回答三个问题：

1. VERA-UAV 是否比 baseline 更容易生成可执行计划？
2. VERA-UAV 是否降低安全违规率？
3. VERA-UAV 的修复轮数和额外推理开销是否可接受？

主结果表建议：

| Method | Syntax valid | Semantic match | Executable success | Safety violation | Mean STL robustness | Repair rounds | Runtime |
|--------|--------------|----------------|--------------------|------------------|--------------------|---------------|---------|
| Direct LLM | TBD | TBD | TBD | TBD | TBD | N/A | TBD |
| Prompt-only | TBD | TBD | TBD | TBD | TBD | N/A | TBD |
| NL2LTL-style | TBD | TBD | TBD | TBD | TBD | N/A | TBD |
| LTLCodeGen-style | TBD | TBD | TBD | TBD | TBD | N/A | TBD |
| VERA-UAV no repair | TBD | TBD | TBD | TBD | TBD | 0 | TBD |
| VERA-UAV full | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

表中 `TBD` 是实验待填数据，不得在任务书中伪造。

### 7.3 泛化实验

泛化维度：

- 未见地图。
- 未见实体命名。
- 自然语言 paraphrase。
- 更长时序组合。
- 更紧时间窗。
- 不可满足任务比例提升。

泛化实验重点报告 VERA-UAV 是否能识别不可满足或模糊任务，而不是输出错误轨迹。

### 7.4 Case study

至少准备三个可视化案例：

1. **语法修复案例**：LLM 输出不合法 STL，Spot/RTAMT 报错，系统修复。
2. **轨迹安全案例**：LTL 满足但 STL robustness 为负，系统绕行后转正。
3. **不可满足案例**：用户要求互相矛盾，系统输出 `UNSAT`。

---

## 8. 消融实验设计

| 消融项 | 变体 | 目的 |
|--------|------|------|
| 去掉 typed IR | Direct LTL/STL generation | 验证结构化中间表示是否提升可靠性 |
| 去掉 counterexample feedback | Generic retry | 验证反例是否比普通重试更有效 |
| 去掉 STL robustness feedback | LTL-only verification | 验证连续安全约束的重要性 |
| one-shot repair | 最多修复 1 次 | 评估修复轮数收益 |
| iterative repair | 最多修复 3 次 | 评估多轮修复上限 |
| 不同模型规模 | Qwen3-8B / Qwen3-14B / DeepSeek-R1-Distill-Qwen-14B | 评估模型能力与验证框架的关系 |
| 去掉 UNSAT 检测 | 强制生成轨迹 | 验证拒答能力对安全性的贡献 |

消融实验的核心不是“证明组件都有效”，而是找出哪些组件对 AAAI 审稿人最关心的安全与可执行性指标贡献最大。

---

## 9. 评价指标

### 9.1 规格生成指标

| 指标 | 定义 |
|------|------|
| Syntax validity | LTL/STL 是否可被 parser 接受 |
| Entity grounding accuracy | 指令实体是否正确映射到地图实体 |
| Semantic match | 生成规格是否与 gold TaskIR / gold formula 等价或近似等价 |
| UNSAT detection accuracy | 不可满足任务是否被正确识别 |
| Clarification accuracy | 模糊任务是否触发 `NEED_CLARIFICATION` |

### 9.2 规划执行指标

| 指标 | 定义 |
|------|------|
| Executable success | 是否生成满足任务且可执行的轨迹 |
| Safety violation rate | 是否违反禁飞区、碰撞、高度、距离约束 |
| Mean STL robustness | 最终轨迹对 STL 规格的平均 robustness |
| Minimum safety margin | 轨迹中最小障碍距离 |
| Path length / flight time | 轨迹代价 |

### 9.3 修复效率指标

| 指标 | 定义 |
|------|------|
| Repair success rate | 验证失败后修复成功比例 |
| Average repair rounds | 平均修复轮数 |
| Runtime overhead | 修复机制带来的额外时间 |
| Token overhead | 修复 prompt 与诊断带来的 token 增量 |

---

## 10. 预期实验结论

本节为预注册预期，不是实验结果。

### 10.1 主预期

预期 VERA-UAV full 在 executable success 和 safety violation rate 上优于所有 baseline。原因是 baseline 通常只优化语言到规格的局部正确性，而 VERA-UAV 把“规格是否能产生安全轨迹”纳入闭环。

### 10.2 反例反馈预期

预期 counterexample feedback 显著降低不可执行计划比例。与 generic retry 相比，结构化反例能告诉 LLM 哪个子句、哪个时刻、哪个实体导致失败，从而减少无方向重试。

### 10.3 Typed IR 预期

预期 typed IR 提高语义一致性和可解释性。直接生成 LTL/STL 容易出现括号、运算符、实体引用和约束遗漏问题；TaskIR 将这些错误提前暴露为字段缺失或类型错误。

### 10.4 STL robustness 预期

预期 STL robustness feedback 对连续安全约束最关键。LTL 层可以证明“最终到达”和“避免禁飞区”等离散性质，但不能充分表达飞行高度、最小距离和时间窗裕度。STL robustness 能提供量化安全边界，是 UAV 区别于普通地面导航任务的关键点。

### 10.5 模型规模预期

预期更强本地模型能提升初始 TaskIR 质量，但验证修复框架对较小模型也有帮助。换句话说，本文不应把贡献写成“某个大模型更强”，而应写成“验证纠错机制提升不同开源模型的可靠性”。

---

## 11. 风险与备选方案

| 风险 | 影响 | 备选方案 |
|------|------|----------|
| 新颖性被认为只是 NL-to-LTL 应用 | AAAI 拒稿风险高 | 强调 STL robustness、反例修复、可执行轨迹闭环 |
| LTLCodeGen baseline 太强 | 主结果优势不足 | 用 UAV 连续约束和不可满足检测作为差异化指标 |
| 本地模型能力不足 | 初始翻译质量低 | 使用 Qwen3-14B / DeepSeek-R1-Distill-Qwen-14B，并报告修复收益 |
| 数据集被认为过于合成 | 应用可信度不足 | 加入 OSM 风格命名、真实城市区块布局统计，但不依赖真实飞行 |
| 修复轮数导致 runtime 过高 | 实时性受质疑 | 报告 one-shot 和最多三轮修复，设置超时和 fallback |
| STL monitor 实现复杂 | 影响进度 | 先实现离散时间 STL 子集，再接 RTAMT |
| AAAI 篇幅不足 | 故事发散 | 主文只放方法和核心实验，ITS 扩展放附录计划 |

---

## 12. 参考文献

[1] Jason Xinyu Liu, Ziyi Yang, Ifrah Idrees, Sam Liang, Benjamin Schornstein, Stefanie Tellex, and Ankit Shah. “Grounding Complex Natural Language Commands for Temporal Tasks in Unseen Environments.” *Proceedings of The 7th Conference on Robot Learning*, PMLR 229:1084-1110, 2023. URL: <https://proceedings.mlr.press/v229/liu23d.html>

[2] Francesco Fuggitti and Tathagata Chakraborti. “NL2LTL -- a Python Package for Converting Natural Language (NL) Instructions to Linear Temporal Logic (LTL) Formulas.” *Proceedings of the AAAI Conference on Artificial Intelligence*, 37(13):16428-16430, 2023. DOI: 10.1609/aaai.v37i13.27068. URL: <https://ojs.aaai.org/index.php/AAAI/article/view/27068>

[3] Behrad Rabiei, Mahesh Kumar A. R., Zhirui Dai, Surya L. S. R. Pilla, Qiyue Dong, and Nikolay Atanasov. “LTLCodeGen: Code Generation of Syntactically Correct Temporal Logic for Robot Task Planning.” arXiv:2503.07902, 2025; project page reports IROS 2025. URL: <https://arxiv.org/abs/2503.07902>; <https://existentialrobotics.org/LTLCodeGen/>

[4] Jun Wang, David Smith Sundarsingh, Jyotirmoy V. Deshmukh, and Yiannis Kantaros. “ConformalNL2LTL: Translating Natural Language Instructions into Temporal Logic Formulas with Conformal Correctness Guarantees.” arXiv:2504.21022, 2025. URL: <https://arxiv.org/abs/2504.21022>

[5] Alexandre Duret-Lutz et al. “Spot: A platform for LTL and omega-automata manipulation.” Official documentation. URL: <https://spot.lre.epita.fr/>

[6] Dejan Nickovic, Tomoya Yamaguchi, Bardh Hoxha, and collaborators. “RTAMT -- Runtime Robustness Monitors with Application to CPS and Robotics.” arXiv:2501.18608, 2025. URL: <https://arxiv.org/abs/2501.18608>; code: <https://github.com/nickovic/rtamt>

[7] Marta Kwiatkowska, Gethin Norman, and David Parker. “PRISM 4.0: Verification of Probabilistic Real-time Systems.” *Computer Aided Verification (CAV)*, 2011. URL: <https://www.prismmodelchecker.org/bibitem.php?key=KNP11>

[8] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. “ReAct: Synergizing Reasoning and Acting in Language Models.” *International Conference on Learning Representations (ICLR)*, 2023. URL: <https://openreview.net/forum?id=WE_vluYUL-X>

[9] Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Daniel Ho, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Irpan, Eric Jang, Rosario Jauregui Ruano, Kyle Jeffrey, Sally Jesmonth, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Kuang-Huei Lee, Sergey Levine, Yao Lu, Linda Luu, Carolina Parada, Peter Pastor, Jornell Quiambao, Kanishka Rao, Jarek Rettinghouse, Diego Reyes, Pierre Sermanet, Nicolas Sievers, Clayton Tan, Alexander Toshev, Vincent Vanhoucke, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Mengyuan Yan, and Andy Zeng. “Do As I Can, Not As I Say: Grounding Language in Robotic Affordances.” *Conference on Robot Learning (CoRL)*, 2022. URL: <https://arxiv.org/abs/2204.01691>

[10] Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng. “Code as Policies: Language Model Programs for Embodied Control.” *IEEE International Conference on Robotics and Automation (ICRA)*, 2023. URL: <https://arxiv.org/abs/2209.07753>

[11] AAAI. “AAAI-26 Main Technical Track: Call for Papers” and “AAAI-26 Reproducibility Checklist.” 2025. URL: <https://aaai.org/conference/aaai/aaai-26/main-technical-track-call/>; <https://aaai.org/conference/aaai/aaai-26/reproducibility-checklist/>

[12] IEEE Intelligent Transportation Systems Society. “IEEE Transactions on Intelligent Transportation Systems (T-ITS): Scope.” URL: <https://ieee-itss.org/pub/t-its/>

---

## 13. 附录：当前 AAAI 优先推进计划

### 13.1 论文定位

AAAI 版先收窄为一篇 AI 方法论文：

**Counterexample-Guided Verified Language-to-STL Planning for UAVs**

核心不是“LLM 会规划无人机”，而是：本地开源 LLM 生成 UAV 任务规格后，用形式化验证器产生反例诊断，再驱动 LLM 修正规格或计划，最终生成可验证轨迹。

### 13.2 AAAI 贡献表述

AAAI 主张三点贡献：

1. 一个 typed IR 到 LTL/STL 的 UAV 任务规格编译链，覆盖到达、避让、顺序、巡检、时间窗、高度和距离约束。
2. 一个 verification-guided repair loop，将语法错误、grounding 缺失、不可满足、轨迹不安全、STL robustness 过低都转成结构化反例反馈。
3. 一个 UAV-NL2STL benchmark，包含自然语言任务、地图、金标准规格、可执行轨迹和失败诊断标签。

### 13.3 时间线

| 时间 | 任务 | 产出 |
|------|------|------|
| 2026-05-18 至 2026-05-24 | 完成核心文献表、冻结 benchmark schema | Related work table + dataset spec |
| 2026-05-25 至 2026-06-07 | 实现地图/任务生成器、gold TaskIR/LTL/STL 模板、基础 planner | 数据生成脚本 + baseline planner |
| 2026-06-08 至 2026-06-21 | 实现 Spot/RTAMT 验证器和 counterexample feedback | verifier module |
| 2026-06-22 至 2026-07-05 | 跑本地模型、baseline、no-repair/full repair 初实验 | 初版主结果表 |
| 2026-07-06 至 2026-07-19 | 主实验、消融、泛化、失败案例统计 | 完整实验表和图 |
| 2026-07-20 至 AAAI 摘要截止 | 完成 abstract、introduction、method、figure 1、主结果表 | AAAI 初稿 |
| AAAI 全文截止前 | 压缩到 7 页，补 appendix、reproducibility、匿名仓库 | 投稿包 |

AAAI-27 的官方 CFP 尚需等待发布；当前倒排参考 AAAI-26 的官方提交节奏与页面限制 [11]。

### 13.4 后续 T-ITS 扩展

AAAI 后扩展成 T-ITS 时，新增内容必须明显区别于会议版。建议新增：

- AirSim / SUMO 或低空物流数字孪生实验。
- 多 UAV 协同和空域冲突仲裁。
- 交通系统指标：任务吞吐量、空域占用、安全裕度、配送/巡检完成率、通信延迟鲁棒性。
- 边缘部署实验：4-bit / 8-bit 模型在 Jetson 或 4090 上的 latency-energy trade-off。
- 标题从 AAAI 的 “verified planning method” 改为 “safe low-altitude UAV operation for intelligent transportation systems”。

---

**版本说明：** 本文为 `v1-20260517`。下一版建议在完成数据集 schema 和第一轮 baseline 跑通后更新为 `v2-YYYYMMDD`，重点替换 `TBD` 表格、补充真实实验结果和失败案例。
