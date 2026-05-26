---
title: "Vision-Language Models for UAV Navigation: The foundation and frontier of vision-language navigation"
description: "Overview of the basic paradigm, core architecture and representative work of VLM+UAV navigation, covering the latest papers such as LogisticsVLN, OmniVLN, and ASMA"
tags: ["UAV", "VLM", "Vision-Language Navigation", "Multimodal large model", "embodied intelligence"]
category: "Tech"
pubDate: 2026-04-27
sourceHash: "afcc4f7205fc6b593288c445afbd3bcab294c159"
---

# Vision-Language Models for UAV Navigation: The foundation and frontier of vision-language navigation

> **UAV Intelligent Series · Part X**
> Focus: Basic paradigm, core architecture and representative work of VLM+UAV

---

## 1. Background: From verbal commands to autonomous flight

Traditional UAV path planning relies on precise mathematical objective functions (such as shortest path, minimum energy consumption), but real-world mission instructions are often fuzzy descriptions of natural language:

- "Go to the basketball court next to the red roof"
- "Follow the white van and keep a distance of 50 meters"
- "Find a high point where you can see the city government building and hover"

These instructions cannot be directly converted into mathematical optimization goals, but they can be understood and reasoned by VLM (Vision-Language Model). Vision-Language Navigation (VLN) is the core research direction to solve this problem - allowing robots (UAV) to navigate in three-dimensional physical space according to natural language instructions.

---

## 2. Task Definition: Core Issues of VLN

The VLN task can be formalized as:

> Given a natural language instruction $I$ and a starting visual observation $O_0$, let the agent perform a series of actions $a_1, a_2, ..., a_T$, and finally reach the target position described by the instruction.

The key challenges are:
1. **Semantic grounding**: Mapping spatial relationships in language ("left", "back", "above") to physical space
2. **Long Horizon Reasoning**: Instructions often describe complex multi-step tasks
3. **Zero-sample generalization**: Unseen buildings, environments, and objects
4. **Three-dimensional characteristics**: UAV, unlike ground robots, has complete 3D movement capabilities

---

## 3. Representative work

### 3.1 LogisticsVLN: UAV VLN for terminal distribution (arXiv, 2025)**Paper:** *LogisticsVLN: Vision-Language Navigation For Low-Altitude Terminal Delivery Based on Agentic UAVs*
**Author:** Xinyuan Zhang, Yonglin Tian, Fei Lin, Yue Liu, Jing Ma, Kornélia Sára Szatmáry, Fei-Yue Wang
**Source:** arXiv:2505.03460, May 2025

**Core contribution:**
- The first VLN mission framework specifically targeted at **Low Altitude UAV Terminal Delivery**
- Proposed Agentic UAV architecture: perception → reasoning → planning → control closed loop
- Special challenges for urban low-altitude environments (building occlusion, dynamic obstacles, GNSS drift)

**Method Framework:**

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

**Key insights:** This is currently the VLN work closest to actual UAV delivery scenarios, integrating the GPT-4V level visual language model with the physical control layer end-to-end.

---

### 3.2 OmniVLN: Open-ground cross-platform end-side VLN (arXiv, 2026)

**Paper:** *OmniVLN: Omnidirectional 3D Perception and Token-Efficient LLM Reasoning for Visual-Language Navigation across Air and Ground Platforms*
**Author:** Zhongyuang Liu, Min He, Shaonan Yu et al.
**Source:** arXiv, March 2026

**Core contribution:**
- **Omnidirectional 3D Perception**: 360° spherical field of view perception, more suitable for complex urban canyons than traditional forward-facing cameras
- **Token Efficient LLM Inference**: Solve the computing power bottleneck of VLM deployment at the edge
- **Cross-platform unified framework**: The same set of algorithms adapts to both UAV and ground robots**Technological Innovation:**
1. **3D token compression**: Encode 3D spatial information into compact tokens to reduce the number of LLM input tokens
2. **Dynamic field of view management**: Adaptively adjust the area of interest according to navigation needs
3. **Lightweight VLM backbone**: client-side version based on Qwen-VL or LLaVA architecture

---

### 3.3 ASMA: Security Boundary-Aware UAV VLN (arXiv, 2024)

**Paper:** *ASMA: An Adaptive Safety Margin Algorithm for Vision-Language Drone Navigation via Scene-Aware Control Barrier Functions*
**Source:** arXiv, September 2024

**Core contribution:**
- Explicitly embed **security constraints** into the VLN framework
- Proposed Scene-Aware Control Barrier Functions (scene-aware control barrier function)
- Ensure hard security constraints in open urban environments

**Why it matters:** Most VLN efforts focus on navigation accuracy and ignore safety. ASMA fills this gap - UAVs can make safety trade-offs between "not understanding instructions" and "hitting the wall".

---

### 3.4 Vision-and-Language Navigation for UAVs: Overview (arXiv, 2026)

**Paper:** *Vision-and-Language Navigation for UAVs: Progress, Challenges, and a Research Roadmap*
**Author:** Hanxuan Chen, Jie Zheng, Siqi Yang et al.
**Source:** arXiv:2604.xxxxx, April 2026

**Overview Coverage:**
- UAV VLN development history (2018-2026)
- Method classification: imitation learning / reinforcement learning / LLM inference
- Core challenges: three-dimensional space representation, dynamic environment, real-time reasoning
- Data sets: D3DROU, AI-TOD, UAV-VLN, etc.
- Future directions: multi-modal large models, embodied intelligence, and safety assurance

---## 4. Technical architecture decomposition

### 4.1 Perception layer (Perception)

**Camera Configuration:**

| Type | Advantages | Disadvantages |
|------|------|------|
| Forward-facing RGB | Mature, cheap | Narrow field of view, limited information |
| Omnidirectional camera | 360° perception | Low resolution, large distortion |
| Depth camera | Dense depth | Failure outdoors, limited range |
| Multi-camera | Stereo triangulation | Complex calibration |

**Perception module responsibilities:**

1. Object detection + semantic segmentation (Grounding DINO, YOLO-World)
2. Spatial relationship extraction (left and right, up and down, relative distance)
3. Scene graph construction (object + relationship + topology)

### 4.2 Understanding layer

**VLM selection comparison:**

| Model | Parameter volume | Vision capabilities | Edge deployment | Representative work |
|------|--------|---------|---------|---------|
| GPT-4V | ~1.8T | Extremely strong | ❌ | Academic research |
| GPT-4o | ~200B | Extremely strong | ❌ | Cloud API |
| LLaVA-1.6 | 7B/13B/34B | Strong | ✅ (ONNX) | Local deployment |
| Qwen-VL | 7B/72B | Strong | ✅ | Chinese scene |
| CogVLM | 17B | Strong | ⚠️ | Balanced Solution |

### 4.3 Planning layer (Planning)

**Existing planning paradigm:**

1. **LLM as Planner**: Directly let LLM output action sequences (ReAct, Reflexion)
   ```
   Instruction → LLM Reasoning → Action Sequence → Execution
   ```
2. **PDDL symbolic planning**: LLM generates PDDL domain description, solved by classic planner
   - Representative: UniPlan
3. **Learnable Planning**: End-to-end imitation learning/reinforcement learning
   - Advantages: Adapt to dynamic environments
   - Disadvantages: poor generalization

### 4.4 Control layer (Control)

**UAV Control Features:**- Requires real-time trajectory tracking (`>100Hz` control frequency)
- The inference delay (second level) of VLM/LLM is inconsistent with real-time control
- **Solution idea: hierarchical control**
  - High level: VLM/LLM (slow, second level) → target point
  - Low level: MPC/PID (fast, millisecond level) → motor control

---

## 5. Key Challenges

### 5.1 Sim2Real Gap

- **Issue:** VLM is pre-trained on ImageNet/COCO and encounters a new urban landscape during real UAV flight
- **Solution ideas:**
  - Domain Randomization (simulation randomization)
  - Retrieval-Augmented Generation (RAG) supplementary prior
  - Self-supervised adaptation (Ego4D, DyTap)

### 5.2 Inference delay vs real-time control

| VLM | Inference delay | Applicable scenarios |
|-----|---------|---------|
| GPT-4o | 1-3s | Cloud offline planning |
| LLaVA-7B | 0.5-1s | Edge delay planning |
| LLaVA-3B | 0.2-0.5s | Edge real-time |

**Solution direction:**

- Dual-process architecture: Decoupling of reasoning thread and control thread
- Speculative Decoding
- 4-bit quantization (AWQ, GGUF)

### 5.3 Three-dimensional spatial reasoning

The spatial relationships in language ("behind the tree", "under the bridge") are not simple projections in three-dimensional space.

**Research Frontiers:**
- SpatialPoint: predict 3D executable waypoints
- Can LLMs See Without Pixels?: Testing LLM spatial intelligence

---

## 6. Data set summary| Dataset | Platform | Scale | Features |
|--------|------|------|------|
| RxR | Ground | 126K commands | Multi-language, expert annotation |
| VLN-CE | Ground | 61K trajectories | Matterport3D |
| AI-TOD | UAV | ~20K commands | Aerial perspective, aerial photography |
| UAV-VLN | UAV | ~10K | Urban Canyon Scene |
| D3DROU | UAV | ~5K | Dynamic obstacles, real flight |

---

## 7. Future research directions

1. **Multi-modal fusion**: RGB + Depth + Event Camera + LiDAR
2. **Small sample adaptation**: LoRA / QLoRA fine-tuning to adapt to specific urban environments
3. **Multiple UAV collaboration VLN**: Multiple UAVs collaborate to understand the same command
4. **World Model Assistance**: Integrate World Model to predict future states
5. **Security Verification**: Formal method to verify VLN decision security

---

## 📚 References1. Zhang et al. *LogisticsVLN: Vision-Language Navigation For Low-Altitude Terminal Delivery Based on Agentic UAVs*. arXiv:2505.03460, 2025.
2. Liu et al. *OmniVLN: Omnidirectional 3D Perception and Token-Efficient LLM Reasoning for Visual-Language Navigation across Air and Ground Platforms*. arXiv, 2026.
3. Chen et al. *Vision-and-Language Navigation for UAVs: Progress, Challenges, and a Research Roadmap*. arXiv, 2026.
4. ASMA. *An Adaptive Safety Margin Algorithm for Vision-Language Drone Navigation via Scene-Aware Control Barrier Functions*. arXiv, 2024.
5. Blukis et al. *Mapping Navigation Instructions to Continuous Control Actions with Position-Visitation Prediction*. CoRL, 2018.
6. Raychaudhuri et al. *Zero-shot Object-Centric Instruction Following: Integrating Foundation Models with Traditional Navigation*. arXiv, 2024.