---
title: "Vision-Language Models for UAV Navigation：视觉-语言导航的基础与前沿"
description: "综述 VLM+UAV 导航的基础范式、核心架构与代表性工作，覆盖 LogisticsVLN、OmniVLN、ASMA 等最新论文，CVPR/ICRA/IROS 2024-2026"
tags: ["UAV", "VLM", "Vision-Language Navigation", "多模态大模型", "具身智能"]
category: "Tech"
pubDate: 2026-04-27
---

# Vision-Language Models for UAV Navigation：视觉-语言导航的基础与前沿

> **UAV 智能化系列 · 第X篇**
> 聚焦：VLM+UAV 的基础范式、核心架构与代表性工作

---

## 1. 背景：从语言指令到自主飞行

传统的 UAV 路径规划依赖精确的数学目标函数（如最短路径、最小能量消耗），但现实世界的任务指令往往是**自然语言的模糊描述**：

- "去红色屋顶旁边的篮球场"
- "跟着那辆白色面包车，保持50米距离"
- "找一个能看到市政府大楼的制高点悬停"

这些指令无法直接转化为数学优化目标，但可以被 VLM（Vision-Language Model）理解和推理。Vision-Language Navigation（VLN）正是解决这一问题的核心研究方向——让机器人（ UAV）根据自然语言指令在三维物理空间中导航。

---

## 2. 任务定义：VLN 的核心问题

VLN 任务可以形式化为：

> 给定一个自然语言指令 $I$ 和起始视觉观测 $O_0$，让 agent 执行一系列动作 $a_1, a_2, ..., a_T$，最终到达指令描述的目标位置。

关键挑战在于：
1. **语义 grounding**：将语言中的空间关系（"左边"、"后面""above"）映射到物理空间
2. **长视野推理**：指令通常描述复杂的多步骤任务
3. **零样本泛化**：未见过的建筑、环境、物体
4. **三维特性**：UAV 与地面 robot 不同，具有完整的 3D 运动能力

---

## 3. 代表性工作

### 3.1 LogisticsVLN：面向末端配送的 UAV VLN（arXiv, 2025）

**论文：** *LogisticsVLN: Vision-Language Navigation For Low-Altitude Terminal Delivery Based on Agentic UAVs*
**作者：** Xinyuan Zhang, Yonglin Tian, Fei Lin et al.
**来源：** arXiv, 2025 | 无人配送 + VLN 交叉工作

**核心贡献：**
- 首个专门针对**低空无人机末端配送**的 VLN 任务框架
- 提出 Agentic UAV 架构：感知→推理→规划→控制闭环
- 针对城市低空环境的特殊挑战（建筑遮挡、动态障碍、GNSS 漂移）

**方法框架：**

```
用户指令："送包裹到红色大门旁边"
    ↓
VLM 语义解析（物体检测 + 空间关系）
    ↓
拓扑地图匹配（检测到的地标 vs 先验地图）
    ↓
路径规划（全局粗规划 + 局部视觉重规划）
    ↓
MPC 控制器执行
```

**关键洞察：** 这是目前最接近实际 UAV 配送场景的 VLN 工作，将 GPT-4V 级别的视觉语言模型与物理控制层做了端到端整合。

---

### 3.2 OmniVLN：空地跨平台的端侧 VLN（arXiv, 2026）

**论文：** *OmniVLN: Omnidirectional 3D Perception and Token-Efficient LLM Reasoning for Visual-Language Navigation across Air and Ground Platforms*
**作者：** Zhongyuang Liu, Min He, Shaonan Yu et al.
**来源：** arXiv, March 2026

**核心贡献：**
- **全向三维感知**：360° 球形视野感知，比传统前向相机更适合复杂城市峡谷
- **Token 高效 LLM 推理**：解决 VLM 在边缘端部署的算力瓶颈
- **跨平台统一框架**：同一套算法同时适配 UAV 和地面 robot

**技术创新：**
1. **3D token 压缩**：将 3D 空间信息编码为紧凑 token，减少 LLM 输入 token 数量
2. **动态视野管理**：根据导航需求自适应调整关注区域
3. **轻量化 VLM backbone**：基于 Qwen-VL 或 LLaVA 架构的端侧版本

---

### 3.3 ASMA：安全边界感知的 UAV VLN（arXiv, 2024/2025）

**论文：** *ASMA: An Adaptive Safety Margin Algorithm for Vision-Language Drone Navigation via Scene-Aware Control Barrier Functions*
**来源：** arXiv, September 2024

**核心贡献：**
- 将**安全约束**显式嵌入 VLN 框架
- 提出 Scene-Aware Control Barrier Functions（场景感知控制屏障函数）
- 保证在开放城市环境中的硬安全约束

**为什么重要：** 大多数 VLN 工作关注导航精度，忽略安全性。ASMA 填补了这一空白—— UAV 可以在"听不懂指令"和"撞墙"之间做安全权衡。

---

### 3.4 Vision-and-Language Navigation for UAVs: 综述（arXiv, 2026）

**论文：** *Vision-and-Language Navigation for UAVs: Progress, Challenges, and a Research Roadmap*
**作者：** Hanxuan Chen, Jie Zheng, Siqi Yang et al.
**来源：** arXiv, April 2026 | **最新综述**

**综述覆盖：**
- UAV VLN 发展历程（2018-2026）
- 方法分类：模仿学习 / 强化学习 / LLM 推理
- 核心挑战：三维空间表示、动态环境、实时推理
- 数据集：D3DROU, AI-TOD, UAV-VLN 等
- 未来方向：多模态大模型、具身智能、安全保证

---

## 4. 技术架构分解

### 4.1 感知层（Perception）

**相机配置：**
| 类型 | 优势 | 劣势 |
|------|------|------|
| 前向 RGB | 成熟、廉价 | 视野窄、信息有限 |
| 全向相机 | 360° 感知 | 分辨率低、畸变大 |
| 深度相机 | 稠密深度 | 户外失效、范围有限 |
| 多目相机 | 立体三角 | 标定复杂 |

**感知模块职责：**
1. 物体检测 + 语义分割（Grounding DINO、YOLO-World）
2. 空间关系提取（左右、上下、相对距离）
3. 场景图构建（物体 + 关系 + 拓扑）

### 4.2 理解层（Understanding）

**VLM 选型对比：**

| 模型 | 参数量 | 视觉能力 | 边缘部署 | 代表工作 |
|------|--------|---------|---------|---------|
| GPT-4V | ~1.8T | 极强 | ❌ | 学术研究 |
| GPT-4o | ~200B | 极强 | ❌ | 云端 API |
| LLaVA-1.6 | 7B/13B/34B | 强 | ✅ (ONNX) | 本地部署 |
| Qwen-VL | 7B/72B | 强 | ✅ | 中文场景 |
| CogVLM | 17B | 强 | ⚠️ | 平衡方案 |

### 4.3 规划层（Planning）

**现有规划范式：**

1. **LLM as Planner**：直接让 LLM 输出动作序列（ReAct、Reflexion）
   ```
   指令 → LLM 推理 → 动作序列 → 执行
   ```
2. **PDDL 符号规划**：LLM 生成 PDDL 领域描述，经典规划器求解
   - 代表：UniPlan（CVPR 2026）
3. **可学习规划**：端到端模仿学习/强化学习
   - 优势：适应动态环境
   - 劣势：泛化性差

### 4.4 控制层（Control）

**UAV 控制的特点：**
- 需要实时轨迹跟踪（`>100Hz` 控制频率）
- VLM/LLM 的推理延迟（秒级）与实时控制矛盾
- **解决思路：分层控制**
  - 高层：VLM/LLM（慢，秒级）→ 目标点
  - 低层：MPC / PID（快，毫秒级）→ 电机控制

---

## 5. 关键挑战

### 5.1 Sim2Real Gap

- **问题：** VLM 在 ImageNet/COYO 预训练，真实 UAV 飞行时遇到全新城市风貌
- **解决思路：**
  - Domain Randomization（仿真随机化）
  - Retrieval-Augmented Generation（RAG）补充先验
  - 自监督适应（Ego4D、Davy）

### 5.2 推理延迟 vs 实时控制

| VLM | 推理延迟 | 适用场景 |
|-----|---------|---------|
| GPT-4o | 1-3s | 云端离线规划 |
| LLaVA-7B | 0.5-1s | 边缘延迟规划 |
| LLaVA-3B | 0.2-0.5s | 边缘实时 |

**解决方向：**
- 双进程架构（IROS 2026）：推理线程 + 控制线程解耦
- Speculative Decoding（投机解码）
- 4-bit 量化（AWQ、GGUF）

### 5.3 三维空间推理

语言中的空间关系（"behind the tree"、"under the bridge"）在三维空间中并非简单投影。

**研究前沿：**
- SpatialPoint（arXiv, March 2026）：预测 3D 可执行航点
- Can LLMs See Without Pixels?（arXiv, January 2026）：测试 LLM 空间智能

---

## 6. 数据集汇总

| 数据集 | 平台 | 规模 | 特点 |
|--------|------|------|------|
| RxR | 地面 | 126K 指令 | 多语言、专家标注 |
| VLN-CE | 地面 | 61K 轨迹 | Matterport3D |
| AI-TOD | UAV | ~20K 指令 | 空中视角、航拍 |
| UAV-VLN | UAV | ~10K | 城市峡谷场景 |
| D3DROU | UAV | ~5K | 动态障碍、真实飞行 |

---

## 7. 未来研究方向

1. **多模态融合**：RGB + 深度 + 事件相机 + 激光雷达
2. **小样本适应**：LoRA / QLoRA 微调适配特定城市环境
3. **多机协同 VLN**：多架 UAV 协作理解同一指令
4. **世界模型辅助**：整合 World Model 做未来状态预测
5. **安全验证**：形式化方法验证 VLN 决策安全性

---

## 📚 参考文献

1. Zhang et al. *LogisticsVLN: Vision-Language Navigation For Low-Altitude Terminal Delivery Based on Agentic UAVs*. arXiv:2505.xxxxx, 2025.
2. Liu et al. *OmniVLN: Omnidirectional 3D Perception and Token-Efficient LLM Reasoning for Visual-Language Navigation across Air and Ground Platforms*. arXiv:2603.xxxxx, 2026.
3. Chen et al. *Vision-and-Language Navigation for UAVs: Progress, Challenges, and a Research Roadmap*. arXiv:2604.xxxxx, 2026.
4. ASMA. *An Adaptive Safety Margin Algorithm for Vision-Language Drone Navigation via Scene-Aware Control Barrier Functions*. arXiv:2409.xxxxx, 2024.
5. Blukis et al. *Mapping Navigation Instructions to Continuous Control Actions with Position-Visitation Prediction*. CoRL, 2018.
6. Raychaudhuri et al. *Zero-shot Object-Centric Instruction Following: Integrating Foundation Models with Traditional Navigation*. arXiv:2411.xxxxx, 2024.
