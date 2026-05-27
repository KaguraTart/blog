---
title: "低空规划论文矩阵 v1：已写文章整合、后续选题与 Zotero 文献清单"
description: "整合无冲突路径规划、百架 UAV 三层调度、信息论驱动 3DGS 主动感知规划等已写方向，规划后续低空规划论文群，并给出 2021-2026 年顶会顶刊与高相关 arXiv 参考文献。"
pubDate: 2026-05-27
updatedDate: 2026-05-27
tags: ["低空规划", "UAV", "论文规划", "Zotero", "T-ITS", "TR-C", "T-RO", "AAAI", "ICRA", "3DGS", "MARL"]
category: Tech
---

# 低空规划论文矩阵 v1：已写文章整合、后续选题与 Zotero 文献清单

> 本文把目前已经写过的低空 UAV 论文方向重新整合成一个 **paper portfolio**。  
> 目标不是再散开写很多想法，而是明确：哪些文章已经有雏形，哪些可以继续做成顶刊/定会论文，每篇论文需要什么文献支撑、实验资产和投稿定位。

---

## 1. 当前已有文章与主线定位

目前已经形成内容基础的核心文章有三条：

| 编号 | 已有内容 | 当前定位 | 推荐主投 | 核心判断 |
|---|---|---|---|---|
| Paper A | 无冲突路径规划 / PPO / MAPPO / 多 UAV 冲突消解 | 低空航路网中的鲁棒冲突消解 | IEEE T-ITS / IEEE T-RO / ICRA-IROS | 不能只写 PPO，要写成“非合作 UAV、通信退化、高密度走廊下的安全-效率折中” |
| Paper B | 百架次 UAV 三层调度 | 城市低空物流/应急系统运营调度 | TR-C first, T-ITS backup | 这是交通系统论文，重点是 capacity、delay、queue stability、vertiport/charging/corridor bottleneck |
| Paper C | 信息论驱动的 UAV 3DGS 主动感知规划 | 主动感知 + 低空数字孪生 + 规划闭环 | T-RO / T-ITS / ICRA-IROS | 如果投交通期刊，要证明主动感知提升巡检、应急、避障或运行控制指标 |

这三篇已经能形成一个很稳定的低空规划三角：

```text
Paper A：战术安全
  多 UAV conflict resolution / no-conflict planning / PPO-MAPPO / CBF / RMADER

Paper B：系统运营
  hundred-UAV scheduling / queue stability / Lyapunov / multimodal logistics

Paper C：环境认知
  3DGS active perception / Fisher information / NBV / safe reconstruction
```

后续新增论文最好围绕这三角扩展，不要另起完全无关的方向。

---

## 2. 总体投稿判断

低空规划方向可以分成三类论文，不同类别的评审标准不同：

| 类型 | 代表论文 | 审稿关注 | 推荐 venues |
|---|---|---|---|
| 交通系统论文 | Paper B、应急资源调配、低空路网规划 | 真实交通问题、系统指标、数据/仿真可信度、政策或运营启示 | TR-C、T-ITS |
| 机器人规划论文 | Paper A、Paper C、数字孪生规划 | 算法新意、实时性、安全性、硬件/仿真验证 | T-RO、RA-L+ICRA/IROS、T-ITS |
| AI 方法论文 | VERA-UAV、CloudBrain-Agent、场景加速生成 | benchmark 难度、理论/验证机制、模型泛化、可复现性 | AAAI、IJCAI、NeurIPS/ICLR workshop、T-ITS extension |

TR-C 的官方定位强调 transportation systems and emerging technologies，且 intellectual core 在 transportation side [1]；T-ITS 覆盖 sensing、communications、controls、planning、design、implementation 等现代交通系统技术 [2]。因此：

- **Paper B / 应急资源调配 / 低空路网规划**：优先按 TR-C 的 transportation system operation 逻辑写。
- **Paper A / Paper C**：可以投 T-RO 或 ICRA/IROS；若转 T-ITS，需要补交通系统指标。
- **Paper E/G 类 LLM-Agent**：首篇更适合 AAAI/IJCAI，期刊版再扩展到 T-ITS。

---

## 3. 论文矩阵：建议形成 7 篇可推进文章

### 3.1 Paper A：低空航路网鲁棒无冲突规划

**建议题目：** Robust Conflict-Free UAV Corridor Planning under Non-Cooperative Traffic and Communication Degradation

**对应已有文章：** 无冲突路径规划、PPO/MAPPO、UAV conflict resolution、UAV conflict env construction。

**核心问题：** 城市低空航路网中，多 UAV 在局部观测、通信延迟、定位误差、非合作飞行器插入条件下，如何保持 separation safety，同时控制延误、额外距离和吞吐损失。

**方法路线：**

- strategic layer：基于航路网的初始路径与时隙分配；
- tactical layer：MAPPO/PPO 输出速度、高度或横向偏移动作；
- safety shield：CBF-QP / ORCA / RMADER-style trajectory check；
- fallback layer：通信退化时切换 conservative priority rule；
- evaluation：训练 30/50 架，测试 100/200 架，覆盖 cooperative、non-cooperative、communication-loss、high-density corridor 四类场景。

**关键参考：**

MAPPO/PPO 的多智能体稳定训练可由 Yu et al. [3] 支撑；MAT 与 FACMAC 提供更强 MARL baseline [4,5]；HAPPO/HATRPO 给出 trust-region multi-agent policy optimization 参考 [6]。机器人侧，EGO-Swarm、MADER、RMADER、RACER、PANTHER 和 GCOPTER 分别支撑 decentralized swarm planning、trajectory sharing under delay、collaborative exploration、perception-aware planning 和 multicopter trajectory optimization [7-12]。

**创新点建议：**

1. 把 “PPO 无冲突路径规划” 从单纯 RL 任务升级为低空交通 corridor safety control。
2. 引入通信退化与非合作 UAV，形成 T-ITS 更关心的实际运行边界。
3. 采用 learning policy + formal/safety shield，避免纯 RL 安全性不足。
4. 指标交通化：LoWC、NMAC、conflict count、average delay、extra distance、throughput、runtime。

### 3.2 Paper B：百架 UAV 三层分层调度

**建议题目：** H-LyraUAV: Queue-Stable Hierarchical Scheduling for Hundred-Scale Low-Altitude UAV Logistics

**对应已有文章：** Paper B 三层调度规划。

**核心问题：** 动态需求、有限 vertiport/charging/corridor 容量、多模态转运约束下，百架级 UAV fleet 如何稳定、高效、安全地运行。

**方法路线：**

- macro layer：demand queue、fleet repositioning、mode choice；
- meso layer：vertiport、charging pad、corridor slot scheduling；
- micro layer：energy/safety/conflict-aware trajectory feasibility；
- theory：Lyapunov drift-plus-penalty 保证 queue stability 和 cost-backlog tradeoff；
- data：synthetic city grid + OSM/POI/NYC TLC/Chicago taxi/SUMO 增强。

**关键参考：**

TR-C 低空 UAV delivery traffic management 已直接讨论 low-altitude urban space 的资源分配和冲突消解 [13]；passenger-centric UAM、公平性与运营效率研究支撑 service quality framing [14]；charging-station delivery network、capacity-constrained UAM scheduling、安全学习调度支撑 infrastructure capacity 和 safe online scheduling [15-17]；truck-drone / UAV-UGV 多模态配送支撑 multimodal extension [18,19]。

**创新点建议：**

1. 百架级在线三层调度闭环，而不是离线 routing/network design。
2. queue stability 成为理论主线，学习模块只做预测或价值估计。
3. 同时评估 delay、throughput、backlog、charging utilization、vertiport bottleneck、corridor congestion。
4. 交通系统结论能回答：什么时候需要限流，哪里是瓶颈，UAV-only 何时不如 multimodal fallback。

### 3.3 Paper C：FIM-3DGS UAV 主动感知规划

**建议题目：** FIM-3DGS: Fisher-Information-Driven Active Perception Planning for Safe UAV Reconstruction

**对应已有文章：** Paper C、Next-Best-View 与 NeRF/3DGS、信息论主动感知。

**核心问题：** 在有限飞行时间、能量和安全约束下，UAV 如何主动选择视点，使 3DGS 地图更快收敛并服务低空规划任务。

**方法路线：**

- scene representation：incremental 3D Gaussian Splatting；
- information metric：对 Gaussian 参数或渲染 Jacobian 构建 Fisher Information / expected information gain；
- planner：NBV candidate generation + safe corridor / CBF constraint；
- task coupling：重建质量不仅报 PSNR/SSIM，还报 obstacle recall、planning collision rate、inspection coverage；
- baselines：ActiveNeRF、FisherRF、GS-Planner、HGS-Planner、POp-GS、frontier exploration。

**关键参考：**

3DGS 原文给出实时显式 radiance field 表示 [20]；ActiveNeRF 是神经渲染主动感知早期代表 [21]；FisherRF 直接支撑 Fisher information active view selection，并已有 3DGS backend 70 fps 结果 [22]；GS-Planner、HGS-Planner、POp-GS 和 NVF 支撑 2024-2025 的 3DGS/NBV 竞争线 [23-26]。

**创新点建议：**

1. 从 “3DGS NBV” 升级为 “服务 UAV 安全规划的 active perception”。
2. 用 Fisher 信息连接 CRB / reconstruction uncertainty / planning safety。
3. 从视觉指标扩展到交通/机器人任务指标：路径可行率、障碍召回率、应急巡检覆盖率。
4. 在 MatrixCity / AirSim / 自建城市低空 cell 上做跨场景泛化。

### 3.4 Paper D：低空安全关键场景覆盖与加速测试

**建议题目：** Coverage-Guided Accelerated Testing for Safety-Critical Low-Altitude UAV Navigation

**对应已有文章：** Paper F 场景覆盖、危险场景生成、7600 万次探索日志。

**核心问题：** 低空 UAV 避障/规划算法的测试场景空间如何定义、如何度量覆盖、如何高效发现危险但有效的 failure 场景。

**方法路线：**

- scenario grammar：局部 50m x 50m x 50m cell，障碍物组合、动态障碍、风扰、目标点、起终点；
- coverage metric：geometry coverage、semantic coverage、dynamics coverage、risk coverage、failure-mode coverage；
- accelerated testing：从 coverage holes 和 failure likelihood 主动采样；
- invalid filtering：过滤不真实、不安全无效、不可执行任务；
- cross-planner evaluation：A*/RRT*/MPC/ORCA/MAPPO/CBF-shielded planner。

**关键参考：**

Shuo Feng 的 NADE 和 testing scenario library generation 是加速测试与安全关键场景库的核心参考 [27-29]；SafeBench 提供 benchmark 平台与安全评估 protocol 参考 [30]。

**创新点建议：**

1. 从自动驾驶 scenario engineering 迁移到低空 UAV 3D 场景空间。
2. 把 coverage、criticality、feasibility 三个目标同时建模。
3. 用 7600 万次探索日志证明覆盖空间和 failure taxonomy。
4. 让结果能回答：哪些障碍组合最危险，哪些 planner 泛化最差，覆盖度提升是否真的减少未知风险。

### 3.5 Paper E：验证纠错式 UAV 语言规划

**建议题目：** VERA-UAV: Verification-and-Repair Language Planning for Low-Altitude UAV Tasks

**对应已有文章：** Paper E。

**核心问题：** LLM 能把自然语言任务转换成 UAV 可执行任务规格，但容易产生不可执行、语义错配或违反安全约束的计划。需要 typed IR、LTL/STL、验证器和反例反馈闭环。

**方法路线：**

- NL instruction -> typed TaskIR；
- TaskIR -> LTL/STL；
- Spot / RTAMT 验证；
- counterexample / robustness feedback；
- local LLM iterative repair；
- final trajectory verification。

**关键参考：**

Lang2LTL、NL2LTL、LTLCodeGen、ConformalNL2LTL 分别支撑 NL-to-LTL grounding、系统演示、code-generation-style temporal logic generation 和 conformal correctness guarantee [31-34]。

**创新点建议：**

1. 不是单纯 NL2LTL，而是 UAV 轨迹可执行闭环。
2. typed TaskIR 降低语言歧义，提高可解释性。
3. 反例反馈和 STL robustness feedback 让 repair 有具体方向。
4. AAAI/IJCAI 版本聚焦 AI planning / verification；T-ITS 扩展再接低空交通运行场景。

### 3.6 Paper G：低空交通云脑 LLM Agent

**建议题目：** CloudBrain-Agent: Tool-Augmented LLM Agents for Low-Altitude Traffic Operation

**对应已有文章：** Paper G / G1。

**核心问题：** 低空交通云脑不能只是聊天模型，而应是能调用调度器、路径规划器、验证器、仿真器和风险评估器的可验证 agent。

**方法路线：**

- LLM 负责任务理解、工具选择、状态摘要和解释；
- tools 包括 Paper A conflict resolver、Paper B scheduler、Paper C active mapper、Paper D scenario tester、Paper E verifier；
- LowAltitudeIR 作为统一中间表示；
- 技术路线优先普通大模型 + agent + skills + MCP/tool-use，后续再做领域 LoRA/SFT；
- 部署上第一阶段调用 API 形成 benchmark，第二阶段本地 Qwen/DeepSeek 系模型做复现和成本控制。

**关键参考：**

UrbanGPT、UniST、TrafficGPT 说明交通/城市时空任务已经开始向 foundation model 和 agent framework 靠近 [35-37]；DriveLM 虽是自动驾驶，但其 Graph VQA 任务形式可借鉴到低空交通 cloud brain 的多步推理 [38]。

**创新点建议：**

1. 低空交通云脑不是“垂类聊天模型”，而是 tool-augmented verifiable agent。
2. 用统一 IR 把调度、规划、感知、验证、场景测试串起来。
3. 先做 agent benchmark，再决定是否微调垂类模型，降低第一篇风险。
4. 评价指标包括 tool-call accuracy、task success、safety violation、repair success、latency、human auditability。

### 3.7 Paper H：城市低空 ODD 与语义功能区规划

**建议题目：** ODD2Route: Semantic Operational-Design-Domain Modeling for Low-Altitude UAV Route Planning

**这是还可以补写的一篇新方向。**

**核心问题：** 城市整体场景如何映射到局部低空航路规划？不同功能区、建筑密度、道路结构、人群活动、禁飞区和应急设施分布，如何决定低空航路的风险、容量和服务策略？

**方法路线：**

- city-level ODD：OSM road/building/POI/land-use + population/demand proxy；
- local test cell：从 city ODD 采样局部 3D obstacle/traffic scenario；
- route risk model：建筑峡谷、学校医院、交通枢纽、高速路段、禁飞区；
- planning output：risk-aware corridor、altitude layer、emergency landing site、charging/vertiport candidates；
- evaluation：跨城市泛化，比较 naive shortest path、risk-aware A*、multi-objective MILP、learning-based route recommender。

**文献支撑：**

这一篇可以从 Paper B 的 TR-C/UAM 文献 [13-19]、Paper D 的 scenario coverage 文献 [27-30] 和 Paper C 的 3D/digital twin 文献 [20-26] 共同支撑。它的难点不在算法复杂，而在 city-level ODD 到 local scenario / route risk 的定义要可信。

**创新点建议：**

1. 把“城市整体场景”和“局部障碍组合”建立可计算映射。
2. 用 ODD coverage 解释场景覆盖，而不是随机生成场景。
3. 为 TR-C/T-ITS 提供城市低空规划、航路设计和测试场景库之间的桥梁。

---

## 4. 推荐优先级

| 优先级 | 文章 | 近期动作 | 原因 |
|---|---|---|---|
| P0 | Paper B | 先冻结 problem formulation、队列模型、实验 benchmark | 最像 TR-C 系统论文，和低空经济/应急最贴合 |
| P0 | Paper A | 把 PPO/MAPPO 改写成鲁棒低空冲突消解论文 | 已有算法基础，但需要交通指标和强 baseline |
| P1 | Paper C | 收敛到 Fisher + 3DGS + safe planning，不再扩太多 | 算法新意较强，能投机器人/AI/ITS |
| P1 | Paper D | 复用 7600 万次探索日志，做 coverage-guided testing | 数据资产独特，容易形成可复现 benchmark |
| P2 | Paper E | 保持 AAAI/IJCAI 方法论文路线 | 适合做短平快但要控制实验范围 |
| P2 | Paper G | 等 Paper A/B/C/D/E 工具接口稳定后启动 | CloudBrain-Agent 需要依赖前面模块，否则容易空 |
| P3 | Paper H | 作为 TR-C/T-ITS 后续扩展 | 需要真实城市数据 pipeline 和 ODD 定义成熟 |

---

## 5. Zotero 整理状态

目标 Zotero collection 名称：

```text
低空规划论文参考
```

当前已完成两层整理：

| 项目 | 状态 |
|---|---|
| Zotero collection | 已存在，collection key 为 `FVHS3SKY`，本地 treeViewID 为 `C17` |
| Zotero 本地选择链接 | `zotero://select/library/collections/FVHS3SKY` |
| 已导入文献 | 38 条 top-level items |
| item type 分布 | `journalArticle` 16 条，`conferencePaper` 10 条，`document/preprint/webpage` 12 条 |
| 本地备份 BibTeX | `zotero/low-altitude-planning-references-20260527.bib` |

导入方式采用 Zotero 本地 connector server，而不是直接写 `zotero.sqlite`。具体流程是：

1. 用 `pandoc` 检查 BibTeX 可解析为 CSL JSON。
2. 通过 Zotero 本地 `/connector/import` 导入 `zotero/low-altitude-planning-references-20260527.bib`。
3. 通过 `/connector/updateSession` 把导入 session 的目标 collection 更新为 `C17 / 低空规划论文参考`。
4. 用 Zotero local API 与只读 SQLite 双重验证 collection 中有 38 条 top-level 文献。

后续如果继续补文献，建议仍然先更新本地 BibTeX，再通过同样的 connector import/updateSession 流程导入 Zotero。不要直接修改 SQLite。

---

## 6. 后续执行计划

### 6.1 第 1 周：冻结论文矩阵

- 确认 Paper A/B/C 是否作为当前三篇主力。
- 确认 Paper D 是否把 7600 万次探索日志作为核心资产。
- 确认 Paper E/G 是否继续保持 AAAI/IJCAI first。
- 已完成 Zotero collection 初始导入；下一步补 PDF、摘要备注和每篇文章的优先级标签。

### 6.2 第 2-3 周：补文献矩阵

- 每篇主力文章至少整理 25 篇高相关文献。
- 每篇文章形成 `related work matrix`：problem、method、data、metric、gap、our angle。
- 对 Paper A/B/C 分别标出 “必须复现 baseline” 和 “只作为 related work” 的论文。

### 6.3 第 4-8 周：先推进 Paper B/A/C 三条实验线

- Paper B：synthetic UAM queueing benchmark + FCFS/greedy/MILP/backpressure/MARL baseline。
- Paper A：corridor conflict simulation + ORCA/CBF/RMADER/MAPPO baseline。
- Paper C：3DGS NBV pipeline + FisherRF/ActiveNeRF/GS-Planner/POp-GS baseline。

### 6.4 第 9-12 周：决定第一篇投稿

- 如果 Paper B 的 queue stability 和百架级结果最稳：先投 TR-C。
- 如果 Paper A 的 conflict safety 和泛化最强：先投 T-ITS/T-RO。
- 如果 Paper C 的 Fisher + 3DGS 理论和视觉结果最强：先投 T-RO/ICRA/IROS。
- 如果 D 的 coverage/failure discovery 数据最好：先投 T-ITS。

---

## 7. 参考文献

[1] Elsevier. *Transportation Research Part C: Emerging Technologies: Aims and Scope.* URL: <https://www.sciencedirect.com/journal/transportation-research-part-c-emerging-technologies>

[2] IEEE Intelligent Transportation Systems Society. *IEEE Transactions on Intelligent Transportation Systems: Scope.* URL: <https://ieee-itss.org/pub/t-its/>

[3] Chao Yu, Akash Velu, Eugene Vinitsky, Yu Wang, Alexandre M. Bayen, and Yi Wu. “The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games.” *Advances in Neural Information Processing Systems*, 2022. URL: <https://arxiv.org/abs/2103.01955>

[4] Muning Wen, Jakub Grudzien Kuba, Runji Lin, Weinan Zhang, Ying Wen, Jun Wang, and Yaodong Yang. “Multi-Agent Reinforcement Learning is a Sequence Modeling Problem.” *NeurIPS*, 2022. URL: <https://proceedings.neurips.cc/paper_files/paper/2022/hash/69413f87e5a34897cd010ca698097d0a-Abstract-Conference.html>

[5] Bei Peng, Tabish Rashid, Christian Schroeder de Witt, Pierre-Alexandre Kamienny, Philip Torr, Wendelin Boehmer, and Shimon Whiteson. “FACMAC: Factored Multi-Agent Centralised Policy Gradients.” *NeurIPS*, 2021. URL: <https://proceedings.neurips.cc/paper/2021/hash/65b9eea6e1cc6bb9f0cd2a47751a186f-Abstract.html>

[6] Jakub Grudzien Kuba, Ruiqing Chen, Muning Wen, Ying Wen, Fudan Sun, Jun Wang, and Yaodong Yang. “Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning.” arXiv:2109.11251, 2021. URL: <https://arxiv.org/abs/2109.11251>

[7] Boyu Zhou, Xin Zhou, Jun Zhang, Fei Gao, and Shaojie Shen. “EGO-Swarm: A Fully Autonomous and Decentralized Quadrotor Swarm System in Cluttered Environments.” *ICRA*, 2021. DOI: 10.1109/ICRA48506.2021.9561902. URL: <https://arxiv.org/abs/2011.04183>

[8] Jesus Tordesillas, Brett T. Lopez, and Jonathan P. How. “MADER: Trajectory Planner in Multiagent and Dynamic Environments.” *IEEE Transactions on Robotics*, 38(1):463-476, 2022. URL: <https://arxiv.org/abs/2010.11061>

[9] Kota Kondo, Reinaldo Figueroa, Juan Rached, Jesus Tordesillas, Parker C. Lusk, and Jonathan P. How. “Robust MADER: Decentralized Multiagent Trajectory Planner Robust to Communication Delay in Dynamic Environments.” arXiv:2303.06222, 2023. URL: <https://arxiv.org/abs/2303.06222>

[10] Boyu Zhou, Hao Xu, and Shaojie Shen. “RACER: Rapid Collaborative Exploration With a Decentralized Multi-UAV System.” *IEEE Transactions on Robotics*, 2023. DOI: 10.1109/TRO.2023.3236945. URL: <https://arxiv.org/abs/2209.08533>

[11] Jesus Tordesillas and Jonathan P. How. “PANTHER: Perception-Aware Trajectory Planner in Dynamic Environments.” *IEEE Access*, 10:22662-22677, 2022. DOI: 10.1109/ACCESS.2022.3154037. URL: <https://arxiv.org/abs/2103.06372>

[12] Zhepei Wang, Xin Zhou, Chao Xu, and Fei Gao. “Geometrically Constrained Trajectory Optimization for Multicopters.” *IEEE Transactions on Robotics*, 38(5):3259-3278, 2022. DOI: 10.1109/TRO.2022.3160022. URL: <https://arxiv.org/abs/2103.00190>

[13] Ang Li, Mark Hansen, and Bo Zou. “Traffic Management and Resource Allocation for UAV-Based Parcel Delivery in Low-Altitude Urban Space.” *Transportation Research Part C: Emerging Technologies*, 143:103808, 2022. DOI: 10.1016/j.trc.2022.103808. URL: <https://doi.org/10.1016/j.trc.2022.103808>

[14] Mehdi Bennaceur, Rémi Delmas, and Youssef Hamadi. “Passenger-Centric Urban Air Mobility: Fairness Trade-Offs and Operational Efficiency.” *Transportation Research Part C: Emerging Technologies*, 136:103519, 2022. DOI: 10.1016/j.trc.2021.103519. URL: <https://doi.org/10.1016/j.trc.2021.103519>

[15] Roberto Pinto and Alexandra Lagorio. “Point-to-Point Drone-Based Delivery Network Design with Intermediate Charging Stations.” *Transportation Research Part C: Emerging Technologies*, 135:103506, 2022. DOI: 10.1016/j.trc.2021.103506. URL: <https://doi.org/10.1016/j.trc.2021.103506>

[16] Qinshuang Wei, Gustav Nilsson, and Samuel Coogan. “Capacity-Constrained Urban Air Mobility Scheduling.” arXiv:2107.02900, 2021. URL: <https://arxiv.org/abs/2107.02900>

[17] Surya Murthy, Natasha A. Neogi, and Suda Bharadwaj. “Scheduling for Urban Air Mobility Using Safe Learning.” arXiv:2209.15457, NASA NTRS, 2022. URL: <https://arxiv.org/abs/2209.15457>

[18] Jiahao Xing, Tong Guo, and Lu Tong. “Reliable Truck-Drone Routing with Dynamic Synchronization: A High-Dimensional Network Programming Approach.” *Transportation Research Part C: Emerging Technologies*, 165:104698, 2024. DOI: 10.1016/j.trc.2024.104698. URL: <https://doi.org/10.1016/j.trc.2024.104698>

[19] Bolong Zhou, Wei Zeng, and Hai Yang. “Multi-Trip UAV-UGV Delivery Network Design with Release Times.” *Transportation Research Part C: Emerging Technologies*, 181:105389, 2025. DOI: 10.1016/j.trc.2025.105389. URL: <https://doi.org/10.1016/j.trc.2025.105389>

[20] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. “3D Gaussian Splatting for Real-Time Radiance Field Rendering.” *ACM Transactions on Graphics / SIGGRAPH*, 42(4), 2023. DOI: 10.1145/3592433. URL: <https://arxiv.org/abs/2308.04079>

[21] Xuran Pan, Zihang Lai, Shiji Song, and Gao Huang. “ActiveNeRF: Learning Where to See with Uncertainty Estimation.” *ECCV*, 2022. URL: <https://arxiv.org/abs/2209.08546>

[22] Wen Jiang, Boshu Lei, and Kostas Daniilidis. “FisherRF: Active View Selection and Mapping with Radiance Fields Using Fisher Information.” *ECCV*, 2024. DOI: 10.1007/978-3-031-72624-8_24. URL: <https://eccv.ecva.net/virtual/2024/oral/1226>

[23] Rui Jin, Yuman Gao, Yingjian Wang, Haojian Lu, and Fei Gao. “GS-Planner: A Gaussian-Splatting-Based Planning Framework for Active High-Fidelity Reconstruction.” arXiv:2405.10142, 2024. URL: <https://arxiv.org/abs/2405.10142>

[24] Zijun Xu, Rui Jin, Ke Wu, Yi Zhao, Zhiwei Zhang, Jieru Zhao, Fei Gao, Zhongxue Gan, and Wenchao Ding. “HGS-Planner: Hierarchical Planning Framework for Active Scene Reconstruction Using 3D Gaussian Splatting.” arXiv:2409.17624, 2024. URL: <https://arxiv.org/abs/2409.17624>

[25] Joey Wilson, Marcelino Almeida, Sachit Mahajan, Martin Labrie, Maani Ghaffari, Omid Ghasemalizadeh, Min Sun, Cheng-Hao Kuo, and Arnab Sen. “POp-GS: Next Best View in 3D-Gaussian Splatting with P-Optimality.” *CVPR*, 2025. URL: <https://cvpr.thecvf.com/virtual/2025/poster/34708>

[26] Shangjie Xue, Jesse Dill, Pranay Mathur, Frank Dellaert, Panagiotis Tsiotras, and Danfei Xu. “Neural Visibility Field for Uncertainty-Driven Active Mapping.” *CVPR*, 2024. URL: <https://arxiv.org/abs/2406.06948>

[27] Shuo Feng, Xintao Yan, Haowei Sun, Yiheng Feng, and Henry X. Liu. “Intelligent Driving Intelligence Test for Autonomous Vehicles with Naturalistic and Adversarial Environment.” *Nature Communications*, 12:748, 2021. DOI: 10.1038/s41467-021-21007-8. URL: <https://www.nature.com/articles/s41467-021-21007-8>

[28] Shuo Feng, Yiheng Feng, Chao Yu, Yi Zhang, and Henry X. Liu. “Testing Scenario Library Generation for Connected and Automated Vehicles, Part I: Methodology.” *IEEE Transactions on Intelligent Transportation Systems*, 2021. DOI: 10.1109/TITS.2020.2972211. URL: <https://doi.org/10.1109/TITS.2020.2972211>

[29] Shuo Feng, Yiheng Feng, Haowei Sun, Yi Zhang, and Henry X. Liu. “Testing Scenario Library Generation for Connected and Automated Vehicles, Part II: Case Studies.” *IEEE Transactions on Intelligent Transportation Systems*, 2021. DOI: 10.1109/TITS.2020.2988309. URL: <https://doi.org/10.1109/TITS.2020.2988309>

[30] Chejian Xu, Wenhao Ding, Baiming Li, Xinqing Chen, Ding Zhao, Bo Li, Jiajun Liu, and Hang Zhao. “SafeBench: A Benchmarking Platform for Safety Evaluation of Autonomous Vehicles.” *NeurIPS Datasets and Benchmarks*, 2022. URL: <https://proceedings.neurips.cc/paper_files/paper/2022/hash/a48ad12d588c597f4725a8b84af647b5-Abstract-Datasets_and_Benchmarks.html>

[31] Jason Liu, Ankit Shah, Eric Rosen, and Stefanie Tellex. “Lang2LTL: Translating Natural Language Commands to Temporal Robot Task Specification.” *PMLR / CoRL*, 229, 2023. URL: <https://proceedings.mlr.press/v229/liu23d.html>

[32] Francesco Fuggitti and Tathagata Chakraborti. “NL2LTL: A Python Package for Converting Natural Language Instructions to Linear Temporal Logic Formulas.” *AAAI Demonstration*, 37(13):16428-16430, 2023. DOI: 10.1609/aaai.v37i13.27068. URL: <https://ojs.aaai.org/index.php/AAAI/article/view/27068>

[33] Behrad Rabiei and Mahesh A. Kumar. “LTLCodeGen: Code Generation of Syntactically Correct Temporal Logic for Robot Task Planning.” arXiv:2503.07902, 2025. URL: <https://arxiv.org/abs/2503.07902>

[34] Jun Wang, David Smith Sundarsingh, Jyotirmoy V. Deshmukh, and Yiannis Kantaros. “ConformalNL2LTL: Translating Natural Language Instructions into Temporal Logic Formulas with Conformal Correctness Guarantees.” arXiv:2504.21022, 2025. URL: <https://arxiv.org/abs/2504.21022>

[35] Zhonghang Li, Lianghao Xia, Jiabin Tang, Yong Xu, Lei Shi, Long Xia, Dawei Yin, and Chao Huang. “UrbanGPT: Spatio-Temporal Large Language Models.” arXiv:2403.00813, 2024. URL: <https://arxiv.org/abs/2403.00813>

[36] Yuan Yuan, Jingtao Ding, Jie Feng, Depeng Jin, and Yong Li. “UniST: A Prompt-Empowered Universal Model for Urban Spatio-Temporal Prediction.” *KDD*, 2024. DOI: 10.1145/3637528.3671662. URL: <https://dblp.org/rec/conf/kdd/0032D0J024>

[37] Jinhui Ouyang, Yijie Zhu, Xiang Yuan, and Di Wu. “TrafficGPT: Towards Multi-Scale Traffic Analysis and Generation with Spatial-Temporal Agent Framework.” arXiv:2405.05985, 2024. URL: <https://arxiv.org/abs/2405.05985>

[38] Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Jens Beisswenger, Ping Luo, Andreas Geiger, and Hongyang Li. “DriveLM: Driving with Graph Visual Question Answering.” *ECCV*, 2024. URL: <https://github.com/OpenDriveLab/DriveLM>
