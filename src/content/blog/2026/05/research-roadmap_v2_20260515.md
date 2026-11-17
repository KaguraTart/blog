---
title: "研究路线图 v2：顶刊战略全面升级与低空交通论文群整理"
description: "在 Q1 顶刊目标下重新整理低空 UAV、低空交通云脑、场景覆盖、调度与形式化规划论文路线，明确近期优先级、投稿定位、交通系统叙事边界和专项规划入口。"
pubDate: 2026-05-15
updatedDate: 2026-05-23
tags: ["论文规划", "研究路线图", "顶刊战略", "T-ITS", "TR Part C", "T-RO", "UAV", "低空"]
category: Tech
---

# 研究路线图 v2：顶刊战略全面升级与低空交通论文群整理

> **v1 → v2 触发：** 老师明确要求所有论文必须发 SCI Q1 顶刊（IF ≥ 7）。v1 中包含 RA-L（IF 4.6）、ICRA 会议等"快速发表"路径，被整体上移到 IEEE T-ITS、TR Part C、IEEE T-RO 这三个顶刊矩阵。

---

## 0. v1 → v2 核心变更总览

### 0.1 投稿期刊全面上移

| Paper | v1 目标 | v1 IF | **v2 目标** | **v2 IF** | 升级幅度 |
|-------|---------|-------|------------|-----------|---------|
| A: KAT-MARL 冲突消解 | IEEE T-ITS | 8.5 | **IEEE T-ITS（保持）** | 8.5 | — |
| B: 三层调度 | TR Part C | 8.5 | **TR Part C / T-ITS（保持）** | 8.5 | — |
| **C: FIM-3DGS 主动感知** | **RA-L / ICRA** | **4.6** | **IEEE T-ITS → TR-C** | **8.5** | **重大升级** |
| D: 功能分区规划 | T-ITS / TR-C | 8.5 | **TR Part C（保持）** | 8.5 | — |
| **E: VERA-UAV 形式化语言规划** | **ICRA / IJCAI** | **会议** | **AAAI first + T-ITS extension** | **会议 + 8.5** | 先会议方法，后期刊扩展 |
| **F: UAV 安全关键场景工程** | **TR Part C** | **8.5** | **T-ITS first + TR-C 应急扩展** | **8.5** | 新增独立低空安全测试路线 |

### 0.2 时间线整体延长

- v1：12 个月窗口（2026/05 – 2027/01），主要因为有 RA-L 快速通道
- **v2：24–30 个月窗口（2026/06 – 2029/06）**，顶刊审稿周期更长、实验需要更扎实

### 0.3 工作量增加预估

| Paper | v1 工作量 | v2 工作量 | 增量原因 |
|-------|----------|---------|---------|
| A | 3–4 月 | 6–8 月 | 实验规模从 50 → 200 UAV，加排队论分析 |
| B | 4–5 月 | 8–10 月 | 加多场景泛化测试 + 真实地图数据 |
| **C** | **3–4 月** | **12–15 月** | **完全重构为低空经济 ITS 论文** |
| D | 3–4 月 | 6–8 月 | 加多城市泛化 + 实际飞行案例 |
| **E** | **6–8 月** | **8–12 月** | **先做 AAAI 方法论文，再扩展为 ITS 系统论文** |
| F | 3–4 月 | 8–12 月 | 7600 万次探索日志清洗、coverage metric、accelerated testing、真实高速应急扩展 |

### 0.4 2026-05-22 校准：交通期刊不是“讲故事”，而是系统问题闭环

这次需要把路线图重新校准一下。交通领域确实比纯算法领域更重视问题叙事和系统意义，但不能理解成“故事讲圆就可以”。更准确的标准是：

> 交通论文要讲一个可信的系统故事，但这个故事必须被模型、实验、指标和边界条件支撑起来。

因此，后续所有偏 TR-C / T-ITS 的规划都要按下面这条链检查：

```text
真实交通系统问题
  -> 现实假设与边界条件
  -> 数学建模 / 运行机制
  -> 强 baseline 与消融
  -> 交通含义指标
  -> 敏感性 / 泛化 / 失败分析
  -> 对运行控制、规划设计或管理政策的启示
```

不是所有论文都要套 TR-C 逻辑。强算法驱动的 AAAI / ICLR / robotics 方法论文，核心仍然是算法新意、理论性质、benchmark 难度和可复现性。只有当目标是 TR-C / T-ITS / transportation journal 时，才必须把“交通系统意义”放到主线。

| 论文 | 主定位 | 是否套交通系统叙事 | 当前写作校准 |
|------|--------|--------------------|--------------|
| Paper A：KAT-MARL 冲突消解 | T-ITS / 低空交通安全控制 | 是，但不能弱化算法 | 从“新 MARL 算法”改成“通信退化、非合作 UAV、高密度 corridor 下的低空冲突消解系统验证” |
| Paper B：百架 UAV 三层调度 | TR-C first | 强烈需要 | 重点是 capacity、delay、queue stability、vertiport/charging/corridor bottleneck 和 multimodal fallback |
| Paper C：FIM-3DGS 主动感知 | 算法 + 交通使能技术 | 有条件需要 | 如果投 T-ITS/TR-C，必须证明主动感知改善巡检、应急、配送等交通任务指标；否则保持机器人感知算法论文 |
| Paper D：语义功能区规划 | TR-C / 城市低空规划 | 需要 | 重点是 ODD、城市功能区、风险暴露、规划建议，不是单纯语义分割 |
| Paper E：VERA-UAV | AAAI / 形式化语言规划 | 不强行套 | 先按 AI planning / verification 论文做；后续 ITS 扩展再加交通运行场景 |
| Paper F：场景覆盖与应急 | T-ITS + TR-C 分叉 | F-J1 部分需要，F-J2 强烈需要 | F-J1 写 safety testing benchmark；F-J2 写山东高速应急资源调配的交通运营论文 |
| Paper G/G1：低空交通云脑 LLM Agent | AAAI/IJCAI first，T-ITS extension | G1 不强行套，期刊扩展需要 | G1 保持 agent/tool-use/verification 方法贡献；期刊版补系统指标和运行启示 |

交通期刊版本的最低实验硬度要求也统一提高：

- 至少 5 个随机种子，主表报告 mean ± std 或 bootstrap confidence interval。
- baseline 不能只放 no-control / greedy，必须包括该问题领域的强经典方法、启发式方法和学习式方法。
- 指标不能只报 reward、accuracy、success rate；必须加入 conflict count、LoWC、NMAC、delay、extra distance、energy、throughput、resource utilization、runtime 等交通含义指标。
- 必须做泛化：训练低密度测试高密度、训练小规模测试大规模、训练固定拓扑测试新拓扑、训练合作 traffic 测试非合作/通信退化 traffic。
- 必须有失败案例分析，说明系统在什么密度、通信丢失率、非合作行为或资源瓶颈下失效。

---

### 0.5 2026-05-23 整理：当前规划文档的阅读顺序与优先级

当前总路线图保留为“研究矩阵入口”，具体执行以 B/E/F/G/G1 专项文档为准。建议阅读顺序如下：

| 优先级 | 文档 | 当前定位 | 近期动作 |
|--------|------|----------|----------|
| P0 | Paper G1：CloudBrain-Agent 完整论文方案 | AAAI / IJCAI first | 先实现可验证 agent、CloudBrain-Bench、工具链和主实验 |
| P1 | Paper B：百架 UAV 三层分层调度 | TR-C first | 建 synthetic queueing benchmark、Lyapunov 调度器和强 baseline |
| P1 | Paper F：UAV 安全关键场景工程 | T-ITS first，TR-C 应急扩展 | 先完成 F-J1：coverage metric + accelerated testing |
| P2 | Paper E：VERA-UAV | AAAI 方法论文，T-ITS 后续扩展 | 收束为 typed IR + LTL/STL + verifier repair，不先做交通系统大论文 |
| P3 | Paper C / Paper D | 待进一步数据和任务收敛 | 保留方向，但不与 B/F/G1 抢近期实验资源 |

这一版需要特别澄清：**旧的 Paper F = CARLA-SUMO 多智能体变道 RL 线不再计入当前低空 UAV 论文群。** 如果以后重新做地面自动驾驶方向，它可以作为独立地面交通论文恢复；当前 Paper F 专指 UAV safety-critical scenario engineering。

近期执行顺序建议为：

1. 先做 G1，因为它能把 Paper B 的调度器、Paper E 的验证器、Paper F 的场景压力测试统一成“低空交通云脑”工具链。
2. 同步启动 B 的 synthetic benchmark，因为它是后续 TR-C 系统论文和 G1 调度工具的核心底座。
3. F-J1 在有探索日志和场景生成脚本后推进，避免一开始陷入过多真实应用叙事。
4. E 保持 AAAI 方法论文，不要提前膨胀成低空交通期刊大系统。

---

## 1. 博客内容全景地图（与 v1 一致）

三大研究主线保持不变（详见 v1）：
- 主线一：路径规划 × 冲突消解 × 多机调度
- 主线二：感知 × 环境重建 × 数字孪生
- 主线三：LLM/VLM × 语义规划 × 形式验证

---

## 2. Tier 1：核心顶刊论文（24 个月内）

### Paper A：大规模城市 UAV 冲突消解 — KAT-MARL（保持顶刊定位）

**目标期刊：** IEEE Transactions on Intelligent Transportation Systems（T-ITS，IF 8.5 Q1）

**与 v1 的变化：** 实验规模升级，理论分析扩展

#### v2 新增要求

- 实验规模从 100 UAV → **200 UAV**（满足 T-ITS 对大规模仿真的偏好）
- 增加**排队论理论分析**：证明 KAT 框架的系统吞吐量上界
- 增加**真实路网映射**：从 CBD 仿真扩展到 2–3 个真实城市（上海陆家嘴、北京 CBD、深圳福田）
- 增加**鲁棒性实验**：通信延迟、传感器噪声、UAV 失效场景

#### v2 时间线
```
2026/06–07  实验环境搭建（基于 uav-conflict-env-construction）
2026/08–10  训练 KAT + 200 UAV 规模扩展实验
2026/11     真实城市路网泛化实验
2026/12     排队论理论分析与证明
2027/01–02  写稿（25 页 T-ITS 格式）+ 内部审阅
2027/03     ◉ 投稿 IEEE T-ITS
2027/09     收到审稿意见（4–6 月审回）
2027/12     接受目标
```

---

### Paper B：百架无人机三层分层调度（保持顶刊定位）

**目标期刊：** Transportation Research Part C 或 IEEE T-ITS（IF 8.5 Q1）

**与 v1 的变化：** 加入排队论数学基础，加入多模态运输场景

#### v2 新增要求

- **理论增强：** 排队论 + Lyapunov 稳定性证明
- **多模态扩展：** UAV + 地面车辆联合调度（增强 TR-C 对运输系统的契合度）
- **真实数据：** 与美团/京东无人配送试点数据对比（如能获取）

#### v2 时间线
```
2026/08–09  三层框架代码实现
2026/10–12  规模扩展实验（20/50/100/200 UAV）
2027/01     排队论与 Lyapunov 分析
2027/02–03  写稿
2027/04     ◉ 投稿 TR Part C
2027/10     接受目标
```

---

### Paper C：FIM-3DGS 主动感知 — **重大重构（详见 v2 专项文档）**

**目标期刊：** IEEE T-ITS（首选）→ TR Part C（备投），IF 8.5 Q1

**重构原因：** v1 定位 RA-L 太低，老师要求顶刊

**v2 核心变化（详见 `paper-c-fim-3dgs-uav-active-perception_v2_20260515.md`）：**

1. **定位升级：** 从"感知算法论文"→"低空经济使能技术"
2. **评估扩展：** 单一感知指标 → 五层指标体系（感知/规划/任务/系统/经济）
3. **案例研究：** 新增三大运输应用案例（建筑巡检、最后一公里配送、应急响应）
4. **实验扩展：** 新增 SUMO + AirSim 联合仿真 + 多 UAV 系统级实验
5. **数据集贡献：** 自建 UAV-Delivery-Dataset 开源数据集

#### v2 时间线
```
2026/06–10  五阶段实验（核心算法 + 三案例 + 多机系统级）
2026/11–12  数据整合 + 初稿（22 页 T-ITS 格式）
2027/01–02  润色 + 内部审阅
2027/03     ◉ 投稿 IEEE T-ITS
2027/09     收到审稿意见
2027/12     接受 / 转 TR-C
2028/06     最终发表
```

详细的 Paper C 规划见 [Paper C v2 专项文档](/blog/paper-c-fim-3dgs-uav-active-perception_v2_20260515/)。

---

### Paper D：多源语义融合 + 功能分区驱动的 UAV 轨迹规划（保持顶刊定位）

**目标期刊：** Transportation Research Part C（IF 8.5 Q1）

**与 v1 的变化：** 多城市泛化实验扩展

#### v2 新增要求

- **多城市泛化：** 在 5 个城市（北京、上海、广州、深圳、武汉）训练+测试
- **真实飞行案例：** 与某 UAV 配送试点合作或公开数据复现
- **风险量化：** 引入精算式风险评估（保险/赔付视角）

#### v2 时间线
```
2026/07–09  GIS 数据采集（5 城市）
2026/10–12  功能分区模型 + 多城市实验
2027/01     真实飞行案例对比
2027/02–03  写稿
2027/04     ◉ 投稿 TR Part C
2027/10     接受目标
```

---

## 3. Tier 2：技术挑战较大的顶刊论文

### Paper E：VERA-UAV 形式化语言规划（先 AAAI，后 ITS 扩展）

**v1 目标：** ICRA / IJCAI（会议）

**当前目标：** AAAI / IJCAI first，T-ITS extension backup

**校准理由：** Paper E 的核心贡献是 AI planning / verification，不应为了顶刊强行变成大而散的交通系统论文。AAAI 版本优先回答“自然语言 UAV 任务如何经 typed IR、LTL/STL、验证器反例和符号 fallback 形成可执行安全轨迹”。

#### 当前收束方向

- **方法主线：** NL instruction -> typed TaskIR -> LTL/STL -> verifier -> counterexample/robustness repair -> trajectory verification。
- **理论边界：** 不声称 LLM 完备；在有限 DSL、可判定验证器和完备底层 planner 假设下证明 relative completeness。
- **实验边界：** 主实验使用 synthetic controlled benchmark；AirSim、真实物流、多 UAV ITS 指标放入后续扩展。
- **投稿策略：** AAAI 主文强调方法、理论、benchmark 与强 baseline；T-ITS 扩展再加入交通运行指标和真实低空场景。

#### v2 时间线
```
2026/06–07  冻结 TaskIR DSL、任务生成器和验证器接口
2026/08–09  实现 Direct LLM / NL2LTL-style / LTLCodeGen-style / VERA-UAV baselines
2026/10     跑主实验、消融和泛化测试
2026/11     完成理论证明、图表和初稿
2026/12     ◉ 投稿 AAAI / IJCAI 对应批次
2027/03     根据结果扩展 T-ITS 版本
```

---

### Paper F：UAV 安全关键场景工程与应急应用（替代旧 CARLA-SUMO 线）

**当前目标：** F-J1 主投 IEEE T-ITS；F-J2 主投 TR-C。

**定位变化：** 当前 Paper F 不再指 CARLA-SUMO 变道 RL，而是围绕 UAV safety-critical scenario engineering 做期刊优先路线：先建立可复现的安全关键场景覆盖与加速测试论文，再把同一平台扩展到山东高速应急救援资源调配。

#### 当前新增要求

- **场景空间：** 明确定义 50m x 50m x 50m UAV test cell、障碍组合、动态障碍、风场、可视域遮挡、禁飞区和任务目标。
- **已有实验资产：** 7600 万次探索日志只能写成“可用基础”，不能写成最终实验结果；需要清洗成 failure taxonomy、coverage holes 和 planner stress cases。
- **方法主线：** coverage metric -> coverage-guided sampler -> danger-validity filter -> accelerated testing -> cross-planner evaluation。
- **强 baseline：** random generation、grid/LHS sampling、Bayesian optimization、CMA-ES、RL adversarial generation、Scenic-style constrained generation。
- **交通扩展：** F-J2 才引入山东高速应急，关注事故发现、UAV 侦察、地面资源调配、响应时间和交通恢复。

#### v2 时间线
```
2026/06–07  整理 7600 万次探索日志，冻结场景空间和 coverage metric
2026/08–10  实现 accelerated testing 与强 baseline
2026/11     cross-planner evaluation、failure taxonomy、统计检验
2026/12–2027/01  写 F-J1 初稿
2027/02     ◉ 投稿 IEEE T-ITS
2027/03–06  扩展山东高速应急资源调配 F-J2
```

---

## 4. 总体 30 个月顶刊投稿路线图

```
─────────────────────────────────────────────────────────────────────────────────────────
时间        A (T-ITS)    B (TR-C)     C (T-ITS)    D (TR-C)     E (AAAI)     F (T-ITS/TR-C)
─────────────────────────────────────────────────────────────────────────────────────────
2026/06    ▶ 环境搭建                  ▶ 算法实现                              ▶ 日志清洗
2026/07    实验训练                    AirSim搭建    ▶ GIS采集
2026/08    实验                        案例1巡检    实验          
2026/09                  ▶ 框架实现    案例2配送                                加速测试
2026/10                  规模实验      案例3应急    多城市实验    ▶ 数据集     baseline
2026/11                  实验          多机系统级   案例研究
2026/12                  实验          初稿         案例研究      数据集完成    
2027/01                  理论分析      润色         写稿          实验          F-J1 写稿
2027/02                  写稿          润色         润色          实验          ◉ 投 T-ITS
2027/03    ◉ 投 T-ITS               ◉ 投 T-ITS                              F-J2 启动
2027/04                  ◉ 投 TR-C                ◉ 投 TR-C
2027/05                                                          实验
2027/06                                                          多UAV案例
2027/07                                                          写稿
2027/08                                                          写稿
2027/09    审稿意见                  审稿意见                   ◉ 投 T-ITS    审稿意见
2027/10                  接受目标                   接受目标                    接受目标
2027/11
2027/12    接受目标                  接受/转TR-C
2028/03                                                          接受目标
2028/06                              最终发表
─────────────────────────────────────────────────────────────────────────────────────────
◉ = 投稿节点   ▶ = 工作启动
```

**核心节奏：**
- **2026 下半年：** G1 / E / F-J1 形成首批可跑实验，避免所有工作同时压到 2027 春季。
- **2027 春季：** A / B / C / D 继续作为顶刊系统论文主线推进。
- **2027 上半年：** F-J2 从 F-J1 平台分化为高速应急资源调配 TR-C 版本。
- **2028 上半年：** 主要接收期。

---

## 5. 顶刊期刊矩阵详解

| 期刊 | 领域 | IF | 接收率 | 审稿周期 | v2 适配 Paper |
|------|------|-----|--------|---------|--------------|
| **IEEE T-ITS** | ITS 综合 | 8.5 | ~20% | 4–6 月 | A, C, F-J1，G/G1 期刊扩展 |
| **TR Part C** | 运输新技术 | 8.5 | ~18% | 4–6 月 | B, D, F-J2 |
| **IEEE T-RO** | 机器人 | 7.4 | ~25% | 6–10 月 | C 备投 |
| **TR Part B** | 运输方法论 | 6.0 | ~15% | 6–8 月 | B 备投 |
| **Transportation Science** | 运输科学 | 5.4 | ~12% | 6–10 月 | B 备投 |

**v2 投稿矩阵原则：**
- **首选 IF ≥ 8 的 Q1**（T-ITS、TR-C）
- **备投同档 IF ≥ 7 的 Q1**（T-RO）
- **不再考虑 IF < 7 的期刊**

---

## 6. 风险评估与备选方案

### 6.1 顶刊战略的关键风险

**风险 1：审稿周期超出博士毕业窗口**
- 顶刊 4–6 月一轮审稿，加修改可能拖到 12+ 月
- **应对：** 2027 春季集中投稿，给修改预留 12 个月
- **底线：** 至少 2 篇接受，剩余可以"submitted/in review"状态毕业

**风险 2：实验工作量过大**
- v2 总工作量约 50–60 月（如果串行），需要团队/合作分工
- **应对：** 近期优先 G1 / B / F-J1 / E，其他方向只保留概念和数据入口，避免资源摊薄

**风险 3：拒稿后转投损失时间**
- 一轮拒稿 + 转投 = 约 6 个月损失
- **应对：** 在 cover letter 中预先准备 TR-C / T-ITS 双 framing

### 6.2 备选投稿优先级

| Paper | 首选 | 备选1 | 备选2 |
|-------|------|------|------|
| A | T-ITS | TR Part C | IEEE T-Cyber |
| B | TR Part C | T-ITS | TR Part B |
| C | T-ITS | TR Part C | IEEE T-RO |
| D | TR Part C | T-ITS | TR Part D（环境） |
| E | AAAI / IJCAI | T-ITS | IEEE T-SMC |
| F | T-ITS | TR Part C | T-ASE / T-RO |

---

## 7. 给老师汇报的一句话总结

> "当前论文群已重新整理为低空 UAV/低空交通云脑主线：G1 先冲 AAAI/IJCAI，B 主投 TR-C，F-J1 主投 T-ITS，E 保持 AAAI 方法论文并预留 T-ITS 扩展。交通期刊论文必须用系统问题、数学模型、强 baseline、交通指标和失败分析支撑，不再只靠方向叙事。"

---

## 8. v1 文档处理说明

- **v1（`research-roadmap_v1_20260515.md`）：** 作为历史归档保留，记录"快速发表混合策略"的设计
- **v2（本文档）：** 当前生效的规划文档
- **下一次更新触发条件：** ① 完成 Paper A 实验数据 ② 收到第一篇审稿意见 ③ 老师调整方向

---

**附录：博客文章与 Paper 对应关系（与 v1 一致）**

| 博客文章 | 对应 Paper |
|---------|-----------|
| marl-kat-uav-conflict | A（主） |
| uav-conflict-resolution | A（参考） |
| uav-conflict-env-construction | A（实验环境） |
| large-scale-uav-scheduling | B（主） |
| uav-urban-route-planning | B（参考） |
| next-best-view-nerf-3dgs-exploration | C（主） |
| information-theory-active-perception | C（理论基础） |
| uav-nerf-gs-planning | C（参考） |
| **paper-c-fim-3dgs-uav-active-perception_v2_20260515** | **C 专项规划（v2）** |
| uav-semantic-mapping-functional-zoning | D（主） |
| uav-digital-twin-semantic-mapping | D（参考） |
| llm-uav-semantic-planning | E（主） |
| llm-guided-uav-planning-frontiers | E（参考） |
| paper-b-hierarchical-uav-scheduling-trc-plan-v1-20260519 | B 专项规划 |
| paper-e-vera-uav-experiment-taskbook-v1-20260517 | E 专项任务书 |
| paper-f-uav-scenario-coverage-journal-roadmap-v2-20260520 | F 专项规划 |
| paper-g-low-altitude-cloud-brain-llm-roadmap-v1-20260520 | G 总路线 |
| paper-g1-cloudbrain-agent-full-paper-plan-v1-20260520 | G1 首篇完整论文方案 |
| carla-sumo-rl-lane-change | 旧 F 线，当前暂不计入低空 UAV 论文群 |
