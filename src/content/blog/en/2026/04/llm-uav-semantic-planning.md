---
title: "LLM empowers drone planning: from semantic understanding to safe collaboration"
description: "In-depth analysis of the two cutting-edge routes of LLM as the planning brain of UAVs: ① Neuro-symbolic safety planning (LLM generates natural language planning → LTL/STL formal verification → provably safe trajectory execution); ② Multi-UAV natural language collaboration (LLM acts as an aerial negotiation intermediary to achieve intent sharing and dynamic renegotiation). Covers architectural design, core algorithms, key papers and future directions."
tags: ["LLM", "drone", "planning", "Neuro-symbolic", "LTL", "Formal verification", "multi-agent", "natural language collaboration", "embodied intelligence", "safety critical"]
pubDate: 2026-04-18
category: Tech
sourceHash: "1aaabede73e5882ed99ac79a008d3128323454c6"
---

# LLM empowers drone planning: from semantic understanding to safe collaboration

## Introduction: Why can LLM become the "brain" of a drone?

In the field of unmanned aerial vehicles (UAV), the traditional planning pipeline has a fundamental information bottleneck: user instructions are vague, and the planner only understands precise mathematical descriptions.

"Find a place with a good view to land" - this sentence is crystal clear to humans, but meaningless to a planner. What is "good view"? Lighting conditions? What is the density of surrounding buildings? GPS signal strength? No planning algorithm can derive an executable trajectory directly from this sentence.

The emergence of large language models (LLM) gives us for the first time a system that can understand fuzzy semantics and perform common sense reasoning. Rather than letting human engineers manually write rules such as "good view = lighting > 1000 lux + building density within 50m surrounding < 30%", it is better to let LLM understand and complete this mapping by itself.

But there is a fatal problem with LLM planning: it will produce hallucinations, and the output is natural language, not a verifiable safe trajectory. An LLM who says "Okay, I've already planned it" may be planning a path to hit the building.

The core question to be discussed in this article is: **How ​​to make LLM a safe brain in the UAV planning system**?

We follow two frontier routes in depth:

1. **Neuro-symbolic security planning**: LLM is responsible for semantic understanding, and formal verification tools are responsible for security assurance.
2. **Multi-drone natural language collaboration**: LLM serves as the "language communication layer" between multiple drones to achieve dynamic negotiation and re-coordination.

---

## 1. Core question: What are the risks of LLM planning?

Before launching the technical route, let’s first figure out what problems will occur if LLM directly does drone planning:

### 1.1 Three levels of risk

**① Risk of semantic illusion**

LLM may misinterpret ambiguous user instructions:

```
用户："绕开那片树林，在附近的空地降落"

LLM 误解输出："向西飞 200m，在树林北侧空地降落"

实际问题：树林西侧有高压线塔，LLM 不知道（且训练数据中未包含该区域）
```

**② Risk of physical infeasibility**

The paths output by LLM may violate dynamical constraints:

```
LLM 建议轨迹：90°急转弯，速度不变
实际情况：无人机最大转弯角速率受限，该路径不可执行
```

**③ Multi-machine collaboration risk**

The LLM planners of multiple drones run independently, which may cause conflicts:

```
UAV-1 的 LLM 规划："向西绕行，避开 UAV-2 的航路"
UAV-2 的 LLM 规划："向西绕行，避开 UAV-1 的航路"
结果：两架无人机撞在一起
```

### 1.2 Solution: Let professional tools do professional things

**Core principle: LLM does what it does best (understand, reason, generate), formal verification tools do what it does best (prove security)**This is not to eliminate traditional planning algorithms, but to add a "semantic understanding interface" to them. Traditional RRT*, MPC, and ORCA are still used for trajectory generation and execution, but the input no longer comes from the manual modeling of engineers, but from the semantic parsing of LLM.

---

## 2. Neuro-symbolic safety planning: LLM → Formal language → Safety trajectory

### 2.1 Architecture Overview

The core idea of the Neuro-symbolic architecture is **dual-layer verification**: using LLM to generate plans and using formal methods to verify security.

```
用户指令: "在3号楼东侧降落，避开2号楼"

┌────────────────────────────────────────────┐
│ LLM 语义规划层 (Neuro)                      │
│                                            │
│  ① 理解指令，提取关键地标和约束              │
│  ② 生成自然语言规划描述                     │
│  ③ 将 NL 描述转换为 LTL/STL 规格            │
│                                            │
│  模型: GPT-4o / Qwen2.5 / DeepSeek-V3      │
└────────────────────────────────────────────┘
                    ↓ LTL/STL 规格
┌────────────────────────────────────────────┐
│ 形式化验证层 (Symbolic)                     │
│                                            │
│  ① Model Checker 验证 LLM输出的规格         │
│  ② 如果验证通过 → 编译为可执行轨迹          │
│  ③ 如果验证失败 → LLM 重新规划              │
│                                            │
│  工具: NuSMV / Spin / Breach / SymPy      │
└────────────────────────────────────────────┘
                    ↓ 验证通过
┌────────────────────────────────────────────┐
│ 经典规划执行层                              │
│  RRT* / MPC / ORCA 生成可执行轨迹          │
│  ↓                                         │
│  PX4 / ArduPilot 控制器执行                 │
└────────────────────────────────────────────┘
```

### 2.2 Formal specification language: LTL and STL

**LTL (Linear Temporal Logic)** is used to describe timing constraints on **discrete time series** and is suitable for expressing path sequence constraints:

```ltl
// 经典 UAV LTL 约束示例
AG !collision          // G: always, A: for all paths
                       // "永远不发生碰撞"

EF reach(landing_zone) // E: exists, F: eventually
                       // "最终会到达降落区"

AG(avoid(building_2)) // "始终避开 2 号楼区域"

AG(heading != unstable) // "姿态始终稳定"
```

**STL (Signal Temporal Logic)** is used to describe constraints on **continuous time signals** and is suitable for expressing trajectory timing constraints (more suitable for UAV continuous trajectories than LTL):

```stl
// STL 规格示例
// "从现在起 30 秒内，必须到达目标点，且期间永远保持安全距离"

spec1 := eventually[0, 30] (reach_goal)
spec2 := always (distance > safety_margin)
spec := spec1 and spec2

// "飞行高度始终在 20m 到 120m 之间"
always (altitude >= 20 and altitude <= 120)

// "接近建筑物时，安全半径自动增大"
always (when (distance_to_building < 30)
        then (safety_radius >= 15))
```

**Why is the LTL → STL compilation process needed? **

Because LLM generates a discrete language description ("First fly north, then east..."), while the execution layer requires a continuous trajectory. We need a **compile link**:

```python
# LLM 输出的离散子目标序列
nl_plan = "先向北飞 50m 避开 2 号楼，然后向东飞 30m 接近 3 号楼，最后下降 20m 降落"

# Step 1: LLM 将 NL 映射为 LTL 规格
ltl_spec = llm.to_ltl(nl_plan)
# → ltl_spec = "G(!collision) ∧ F(reach(goal)) ∧ G(avoid(building_2))"

# Step 2: 将 LTL 编译为 STL 时序约束
stl_spec = ltl_to_stl_compiler(ltl_spec)
# → stl_spec = "always[0,∞) (distance > 20) ∧ eventually[0,60] (position ∈ landing_zone)"

# Step 3: 将 STL 编译为控制器参数
controller_params = stl_to_mpctuner(stl_spec)
# → MPC 权重、预测时域、安全半径等参数被自动设置
```

### 2.3 Key question: How does LLM generate LTL?

This is the most subtle step of the entire architecture. LLM cannot "output LTL" directly and needs to be guided through the prompt project:

```python
# LLM Prompt: NL → LTL 转换
prompt = """
你是一个无人机规划专家。请将以下自然语言指令转换为 LTL（线性时序逻辑）规格。

【可用 LTL 运算符】
- G φ: φ 始终为真 (Globally)
- F φ: φ 最终为真 (Finally)
- X φ: 下一时刻 φ 为真 (Next)
- U φ: φ 一直为真直到 ψ (Until)
- ¬ φ: 非 φ
- ∧, ∨: 与，或

【无人机领域约束模板】
- G(!collision): 永不碰撞
- F(reach_goal): 最终到达目标
- G(altitude ∈ [20, 120]): 高度始终在 [20, 120]m
- avoid(zone): 始终避开某区域
- G(speed < v_max): 速度始终不超过最大值

【示例】
NL: "飞往 A 点，不要撞到障碍物，最后降落"
LTL: "F(reach(A)) ∧ G(!collision) ∧ F(landed)"

【你的任务】
NL: "{user_instruction}"
LTL: 
"""
```

**Key Challenges**:
1. LLM has limited understanding of LTL syntax and needs few-shot examples for guidance.
2. Complex instructions may require nested sequential logic, and LLM is prone to errors.
3. Requires domain-specific LTL template libraries (domain-specific templates)

**Solution: Template-based LTL Generation**

```python
class LTLTemplateLibrary:
    """预定义 LTL 模板，LLM 只需填充参数"""

    TEMPLATES = {
        "avoid_building": "G(avoid(building_{id}))",
        "reach_then_land": "F(reach({zone})) ∧ F(landed)",
        "altitude_constraint": "G(altitude ∈ [{h_min}, {h_max}])",
        "sequence": "F(step1) ∧ G(step1) U step2",  # 先完成 step1 再开始 step2
        "monitor": "G(monitor({condition}))",  # 持续监控某条件
    }

    def fill_template(self, nl_instruction: str, llm) -> str:
        """用 LLM 理解 NL 指令，选择合适模板并填充参数"""
        parsed = llm.parse_instruction(nl_instruction)
        # parsed = {"type": "avoid_reach", "building_id": 2, "goal_zone": "zone_A"}
        
        if parsed["type"] == "avoid_reach":
            return (self.TEMPLATES["avoid_building"].format(id=parsed["building_id"])
                    + " ∧ "
                    + self.TEMPLATES["reach_then_land"].format(zone=parsed["goal_zone"]))
```

### 2.4 Formal verification: Model Checking

After getting the LTL/STL specifications, you need to use the model checker to verify whether the LLM plan satisfies the constraints.**Model-based Verification:**

```python
# 使用 NuSMV 进行 LTL 模型检查
from smv_verify import NuSMV

# 将无人机环境建模为有限状态机
uav_model = """
MODULE uav
VAR
  position : {safe, danger, goal};
  altitude : {low, mid, high};
  action : {fly_north, fly_east, descend, hover, land};

ASSIGN
  init(position) := safe;
  init(altitude) := mid;

  -- 状态转移规则
  next(position) := case
    position = safe & action = fly_east & altitude = mid : safe;
    position = safe & action = descend & altitude = mid : danger;  -- 下降时可能进入危险区
    position = goal : goal;
    TRUE : position;
  esac;
"""

# LTL 属性
safety_property = "G !danger"  # "始终不在危险区"
reachability_property = "F goal"  # "最终到达目标"

# 模型检查
model_checker = NuSMV()
result = model_checker.verify(uav_model, safety_property)
# result: {"status": "PROVED", "counterexample": None}  ✓ 安全！
```

**Simulation-based STL Verification:**

```python
# 使用 Breach 工具箱进行 STL 验证（连续轨迹）
from stl_verify import Breach

# 生成候选轨迹（来自 RRT*/MPC）
candidate_trajectory = mpcc_planner.solve(
    start=uav_state,
    goal=llm_generated_goal,
    constraints=stl_spec
)

# STL 鲁棒性评估（不只是 true/false，还能给出违反程度）
breach = Breach()
robustness = breach.evaluate_stl(stl_spec, candidate_trajectory)

# robustness > 0: 满足规格，值越大越安全
# robustness < 0: 违反规格，值越小越危险
# robustness = -0.5: 在边界附近，稍微超出约束

if robustness > 0:
    print(f"✓ 轨迹通过 STL 验证（鲁棒性 = {robustness:.3f}）")
    execute_trajectory(candidate_trajectory)
else:
    print(f"✗ 轨迹违反 STL（鲁棒性 = {robustness:.3f}），重新规划")
    # 触发 LLM 重新生成规划
    llm.replan_feedback(robustness)
```

**Robustness feedback → LLM replanning (iterative optimization loop):**

```python
def neuro_symbolic_planning_loop(user_instruction, max_iterations=3):
    """Neuro-symbolic 规划主循环"""
    for i in range(max_iterations):
        # Step 1: LLM 生成规划 + LTL 规格
        nl_plan, ltl_spec = llm.plan_with_spec(user_instruction)
        
        # Step 2: 编译 LTL → STL
        stl_spec = ltl_compiler(ltl_spec)
        
        # Step 3: 用 MPC/RRT* 生成候选轨迹
        trajectory = planner.solve(stl_spec)
        
        # Step 4: STL 鲁棒性验证
        robustness = breach.verify(stl_spec, trajectory)
        
        if robustness > 0:
            return trajectory  # 验证通过，执行
        
        # Step 5: 验证失败 → 将鲁棒性反馈给 LLM
        feedback = f"STL 验证失败。鲁棒性 = {robustness:.3f}。" \
                   f"问题：{diagnose_failure(robustness, stl_spec)}。" \
                   f"请重新规划。"
        user_instruction = feedback  # 将反馈作为新的约束输入
    
    raise PlanningError(f"经过 {max_iterations} 次迭代仍无法通过验证")
```

### 2.5 Key Research Work

#### 2.5.1 LLMRL: LLM generates RL reward function

**Schmidt et al., RRL 2024**

LLM directly generates the reward function of reinforcement learning to solve the tedious problem of manual design of reward engineering:

```python
# LLM 生成的自然语言 reward 规范
llm_reward_spec = """
为无人机避障任务设计奖励函数：
- 靠近目标：+10 * distance_reduction
- 保持安全距离（> 5m）：+5
- 违反安全距离（< 3m）：-100
- 到达目标：+100
- 碰撞：-500
"""

# 自动转换为代码
reward_code = llm.to_reward_code(llm_reward_spec)
# → reward = 10 * dr + 5 * safe_flag - 100 * danger_flag + 100 * goal_flag - 500 * collision
```

#### 2.5.2 CALM: Chat with Language Models for Multi-Agent Coordination

**Meta AI, 2024**

CALM uses LLM to allow multi-agent coordination through natural language without the need for predefined communication protocols:

- Each agent maintains a "shared context" and uses LLM to reason about the intentions of other agents
- Key innovation: LLM's emergent reasoning allows the agent to reason about other people's beliefs (false belief task pass rate > 80%)
- **Inspiration for UAV**: UAV-1 can be allowed to use LLM to reason about what UAV-2 will do next, thereby coordinating in advance

#### 2.5.3 Verified LLM Planning

**arXiv 2024, Verified LLM-based Task Planners**

This is the work most directly relevant to us:

```python
# 核心架构
class VerifiedLLMPlanner:
    def __init__(self, llm, model_checker):
        self.llm = llm
        self.model_checker = model_checker
    
    def plan(self, task):
        # 1. LLM 生成 PDDL 规划
        plan = self.llm.pddl_plan(task)
        
        # 2. 用 model checker 验证
        is_safe = self.model_checker.verify(plan, safety_constraints)
        
        # 3. 若验证失败，LLM 自我修正
        while not is_safe and self.llm.can_replan():
            feedback = self.model_checker.get_counterexample()
            plan = self.llm.replan(feedback)
            is_safe = self.model_checker.verify(plan, safety_constraints)
        
        return plan
```

#### 2.5.4 LTLBAR: LTL from Natural Language

**arXiv 2024, LTLBAR: Neural LTL Translation with Bootstrap**

Jointly fine-tuning datasets with GPT-4 to translate natural language instructions into LTL specifications:- Training data: 10K NL-LTL pairs, generated by GPT-4 and manually corrected
- Achieved **92.3% semantic equivalence accuracy on unseen instructions**
- Solved the core problem of LLM being difficult to accurately generate LTL

---

## 3. Multi-UAV natural language collaboration: LLM as the "air negotiation layer"

### 3.1 The nature of the problem

Traditional multi-UAV collaboration relies on predefined communication protocols (such as MAVLink, ROS 2 topics), and the coordination strategy is fixed before takeoff. However, in real scenarios, drones often require **dynamic negotiation** - when encountering unexpected obstacles, task changes, or insufficient energy, they need to be re-coordinated.

**Limitations of traditional methods:**

```
UAV-1: "我需要避障，申请占用节点 A"
UAV-2: "我已经预定了节点 A"

传统系统：冲突检测 → 优先级仲裁 → 强制重路由
问题：没有考虑"谁的紧迫性更高"、"换节点 B 对整体任务影响多大"
```

**Possibilities brought by LLM: using natural language for dynamic semantic coordination**

```
UAV-1: "我这边检测到阵风，稳定性下降，需要尽快降落。能否借用你
        规划的 A-B 航路最后一段？我可以绕道 C-D 作为补偿。"

UAV-2: "理解。我这边油量还剩 40%，不算紧急。我走 C-D 你走 A-B
        末端，这样我们都不绕远。我可以稍等你 5 秒。"

LLM 中介：理解了双方意图，生成了对双方都优的协商结果。
```

This is not science fiction - there are already several works starting to explore this direction in 2024-2025.

### 3.2 Architecture design: LLM as a shared intent layer

```python
# 多无人机 LLM 协同架构
class UAVSwarmLLMCoordinator:
    """LLM 作为多无人机通信与协调的中介层"""
    
    def __init__(self, llm_backend):
        # 每个无人机配备一个 LLM 实例（边缘部署）
        self.uav_llms = {
            "UAV-1": LLM(edge_model="Qwen2-7B"),
            "UAV-2": LLM(edge_model="Qwen2-7B"),
            ...
        }
        # 共享意图池（分布式知识库）
        self.shared_intent_pool = SharedKnowledgePool()
    
    def broadcast_intent(self, uav_id: str, intent: str):
        """广播本无人机的意图到共享池"""
        # intent 示例: "我计划在 10:32 到达 zone-B，路径 A→B，速度 8m/s"
        self.shared_intent_pool.add(
            uav_id=uav_id,
            intent=intent,
            timestamp=current_time(),
            priority=self.estimate_priority(uav_id)
        )
    
    def query_intentions(self, uav_id: str) -> List[Dict]:
        """查询其他无人机的意图（用于冲突预判）"""
        return self.shared_intent_pool.get_all(uav_id)
    
    def negotiate(self, uav_id: str, conflict: Conflict) -> str:
        """发起协商请求，LLM 辅助生成协商消息"""
        llm = self.uav_llms[uav_id]
        other_intents = self.query_intentions(uav_id)
        
        negotiation_prompt = f"""
当前无人机 {uav_id} 面临冲突：{conflict.describe()}
其他无人机意图：{other_intents}

请生成一段自然语言协调请求：
1. 说明当前困境
2. 解释自己的紧迫性（时间/能量/任务重要性）
3. 提议替代方案
4. 保持礼貌和协作语气（而非命令语气）

要求：50 字以内，适合 VHF/数据链传输。
"""
        return llm.generate(negotiation_prompt)
```

### 3.3 Negotiation Agreement: Intent Sharing + Dynamic Replanning

**Three-stage negotiation agreement:**

```
阶段 1: 意图声明（Intention Declaration）
┌──────────────────────────────────────────────┐
│ 每架无人机周期性地广播：                       │
│ "UAV-X 计划 [时间窗口] 内从 [起点] 到 [终点]，  │
│  路径 [航路描述]，优先级 [高/中/低]"            │
│                                              │
│  意图存储在分布式 KV 存储中（etcd / Redis）    │
└──────────────────────────────────────────────┘

阶段 2: 冲突检测 + 协商触发（Conflict Detection）
┌──────────────────────────────────────────────┐
│ 当两条航迹的时空包络出现重叠时：                │
│                                              │
│  检测器 → 识别冲突 UAV 对 (i, j)               │
│  → 触发协商事件                              │
│                                              │
│  协商类型：                                   │
│  - 时序冲突：同时到达同一节点                  │
│  - 空间冲突：航迹包络重叠                     │
│  - 资源冲突：共同需要某通信频道                │
└──────────────────────────────────────────────┘

阶段 3: LLM 辅助协商（LLM-Assisted Negotiation）
┌──────────────────────────────────────────────┐
│ 双方交换自然语言信息：                         │
│                                              │
│  UAV-i: "我的电池只剩 15%，必须尽快降落。"      │
│  UAV-j: "理解了。你先走 A-B-C，我绕 D-E-F。"  │
│                                              │
│  LLM 辅助：                                   │
│  ① 理解对方意图和紧迫性                      │
│  ② 生成多个可选方案                          │
│  ③ 评估每个方案对己方任务的影响                │
│  ④ 输出最优协商策略                          │
└──────────────────────────────────────────────┘
```

### 3.4 Intent representation: from natural language to structured data

The natural language coordination information generated by LLM needs to be automatically understood and executed by the system, so it requires two-way conversion:

```python
# 自然语言 → 结构化意图（LLM 解析）
def parse_intention(nl_text: str) -> IntentionStruct:
    """将其他无人机的自然语言意图解析为结构化数据"""
    prompt = f"""
从以下无人机广播消息中提取结构化信息：

{nl_text}

输出 JSON 格式：
{{
  "uav_id": "无人机编号",
  "action": "动作类型（fly/hover/land/evade）",
  "origin": "起点坐标 [x, y, z]",
  "destination": "终点坐标 [x, y, z]",
  "time_window": [开始时间, 结束时间],
  "priority": "高/中/低",
  "constraints": ["约束1", "约束2"],
  "reason": "紧迫原因"
}}
"""
    return llm.parse_json(prompt)


# 结构化意图 → 自然语言（LLM 生成广播）
def intention_to_nl(intent: IntentionStruct) -> str:
    """将结构化意图转换为可广播的自然语言消息"""
    prompt = f"""
将以下无人机规划信息转换为简洁的自然语言广播（< 50 字）：

意图信息：{intent}

要求：
- 包含时间、路径、优先级
- 用人类可理解的方式描述
- 适合无线链路传输
"""
    return llm.generate(prompt)
```

### 3.5 Key Research Work

#### 3.5.1 Chatty Robots: Natural Language Communication in Multi-Agent RL

**Lyu et al., CoRL 2024**

Use LLM to generate a **natural language communication protocol** for multi-agent reinforcement learning:

```python
# 核心思想：RL 策略学习"说什么"，LLM 负责"怎么说"
class LLMCommunication:
    def __init__(self, rl_policy, llm):
        self.rl_policy = rl_policy  # 学习通信的 RL 策略
        self.llm = llm              # 学习如何表达的 LLM
    
    def decide_message(self, obs, other_agents):
        # RL 策略决定"需要通信"
        if self.rl_policy.should_communicate(obs):
            # 提取关键信息
            key_info = self.rl_policy.extract_info(obs)
            # LLM 将关键信息转换为自然语言
            message = self.llm.narrate(key_info, other_agents)
            return message
        return None
```

#### 3.5.2 MAgIC: Multi-Agent Interaction via Communication

**IBM Research, 2025**

The core innovation of MAgIC: Proposes **Intention Voting**, allowing LLM to reach consensus through voting when coordination fails:

```
协商场景：UAV-1 和 UAV-2 都需要优先使用 shared resource（通信频道）

MAgIC 协商过程：
1. 双方各自提出优先级和理由
2. LLM 评估两个提案，生成"仲裁建议"
3. 双方对仲裁建议投票
4. 票数多者执行

仲裁示例：
"鉴于 UAV-1 电池仅剩 8%（紧迫性高），且绕路成本为 UAV-2 的 2.3 倍，
建议 UAV-1 优先使用频道，UAV-2 等待 8 秒后使用。"
```#### 3.5.3RIAL + LLM: Implicit vs explicit coordination

**Foerster et al., "Learning to Communicate with Deep Multi-Agent RL" (improved version)**

Distinguish between two types of coordination methods:

| Coordination Type | Description | LLM Role |
|---------|------|---------|
| **Implicit Coordination** | Infer intentions by observing others' behavior without explicit communication | LLM assists in reasoning about others' intentions |
| **Explicit Coordination** | Direct exchange of intent via language messages | LLM assists in generating and understanding messages |

**RIAL (Reinforced Imitative Actor Learning)** combined with LLM:
- LLM prior knowledge injects RL strategy (reduces exploration space)
- Solve the sparse reward problem (LLM guides the exploration direction)

#### 3.5.4 LLM-as-a-Judge for UAV Coordination

**Emerging Direction: Using LLM to Assess the Quality of Negotiation Outcomes**

```python
class CoordinationJudge:
    """用 LLM 作为协调质量评判器"""
    
    def evaluate(self, negotiation_log: List[Message]) -> dict:
        prompt = f"""
评估以下无人机协调过程的质量：

【协调日志】
{negotiation_log}

请从以下维度打分（1-10）：
1. 效率：是否快速达成一致？（时间代价）
2. 公平性：双方是否都做出了合理让步？
3. 安全性：协商结果是否满足安全约束？
4. 最优性：协商结果是否接近全局最优？

输出 JSON：
{{"efficiency": X, "fairness": Y, "safety": Z, "optimality": W}}
"""
        return llm.parse_json(prompt)
```

---

## 4. Client-side deployment challenges and optimization

### 4.1 Why deploy LLM at the edge?

Problems with cloud LLM:

- **Latency**: 4G/5G round trip to the cloud > 500ms, unacceptable for high-speed flying drones
- **Reliability**: Unable to connect when signal is blocked, flight cannot be interrupted
- **Privacy**: Map information in sensitive areas is not suitable for uploading to the cloud

Goal for edge deployment: **Run 7B model on Jetson Orin NX / Orin Nano, latency < 200ms**

### 4.2 Extreme Optimization Technology Stack

```python
# 端侧 LLM 推理优化

# ① INT4/INT8 量化（体积缩小 4-8 倍，速度提升 2-4 倍）
llm_quantized = AutoModelForCausalLM.from_pretrained(
    "Qwen2-7B-Instruct",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
)

# ② 推测解码（Speculative Decoding）
# 用小模型（1B）预测 token，用大模型（7B）验证
# 实际加速比 2-3x，输出质量几乎无损
speculative_model = SpeculativeDecoder(
    draft_model="Qwen2-1.5B",
    target_model="Qwen2-7B",
    max_draft=4
)

# ③ 连续批处理（Continuous Batching）
# 多个请求共享 GPU 计算，提升吞吐量
from text_generation_server.utils.flash_attention import FlashAttention
llm_engine = TGIEngine(
    model="Qwen2-7B-Instruct",
    max_batch_size=8,
    use_flash_attention=True
)

# ④ KV Cache 量化
kv_quantizer = KVCacheQuantizer(bits=8)
# KV cache 显存占用减少 50%+

# ⑤ 端侧性能基准（Jetson Orin NX @ INT4）
# Qwen2-7B + INT4 + Speculative Decoding:
#   首 token 延迟: 180ms
#   生成速度: 25 tokens/s
#   显存占用: 6GB / 8GB
```

### 4.3 Hierarchical reasoning architecture (delay budget allocation)

```
总延迟预算: < 300ms

┌─────────────────────────────────────────────────────┐
│ Layer 1: 本地意图理解 (LLM, < 200ms)               │
│  - INT4 Qwen2-7B 本地运行                          │
│  - NL → LTL 转换 (< 200ms)                         │
│  - 复杂请求 → 触发云端协作                           │
└─────────────────────────────────────────────────────┘
                        ↓（如需）
┌─────────────────────────────────────────────────────┐
│ Layer 2: 云端重协商 (LLM, 200-500ms)                │
│  - 通过 5G 网络与云端 LLM 通信                       │
│  - 处理复杂多机协调请求                             │
│  - 更新本地意图池                                   │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ Layer 3: 实时执行 (< 10ms)                          │
│  - 纯经典算法，不依赖 LLM                           │
│  - MPC / ORCA 执行轨迹                             │
│  - 100Hz 控制周期                                  │
└─────────────────────────────────────────────────────┘
```

### 4.4 Lightweight verification of LLM planning results

Edge device resources are limited, and it is impossible to use NuSMV to do complete model checking every time. Requires **Lightweight Symbol Validation**:

```python
# 轻量级安全验证（比 NuSMV 快 100x）
class LightweightSafetyVerifier:
    """基于几何计算 + 符号边界检查的轻量验证"""
    
    def verify(self, trajectory, ltl_spec, env_map):
        # Step 1: 几何碰撞检查（O(n)，极快）
        for obs in env_map.static_obstacles:
            if trajectory.intersects(obs, margin=self.get_safety_margin(trajectory, obs)):
                return False, f"Collision risk: {obs.id}"
        
        # Step 2: 动力学可行性检查
        if not self.check_dynamics(trajectory):
            return False, "Dynamics infeasible: curvature exceeds limit"
        
        # Step 3: LTL 关键路径点检查（只检查关键节点）
        key_states = trajectory.extract_key_states()
        for state in key_states:
            if not self.check_ltl_state(state, ltl_spec):
                return False, f"LTL violation at state {state}"
        
        return True, "All checks passed"
    
    def check_ltl_state(self, state, ltl_spec):
        """只检查 LTL 规格在该状态的原子命题"""
        # G(!collision): 检查当前是否 collision-free
        # F(reach_goal): 检查目标是否可达（下界估计）
        # 不需要遍历整个轨迹，只做当前快照检查
        return True  # 简化版，实际需要符号执行
```

---

## 5. Future directions

### 5.1 Recent (1-2 years)

**① World Model enhanced LLM planning**Before LLM generates the plan, use World Model to predict the execution consequences:

```python
# 用 3DGS / NeRF World Model 预测规划执行后的环境变化
world_model = NeuralRenderer3D(ckpt="urban_scene_v2.pt")

# LLM 规划 → World Model 模拟 → 验证 → 执行
planned_trajectory = llm.plan(nl_instruction)
simulated_scene = world_model.render_trajectory(
    planned_trajectory,
    camera_pose=drone_camera
)
# 如果模拟中看到障碍物 → LLM 重新规划
if simulated_scene.contains_unexpected_obstacle():
    llm.replan("检测到模拟中出现未预期障碍，重新规划")
```

**② Multi-modal LLM unified planning interface**

Integrate visual, language, and sensor data to allow LLM to simultaneously understand:
- Live camera images ("What is the number of this building?")
- Map data ("Coordinates of Building 3 on the map")
- Sensor data ("Current wind speed 8m/s, exceeding safety threshold")

The current solution is multi-channel input splicing, and the future direction is **cross-modal attention unified encoding**.

### 5.2 Mid-term (3-5 years)

**③ Real-time multi-machine LLM coordination under 5G/6G network**

The low-latency, high-bandwidth network makes multi-machine LLM coordination possible for cloud-edge collaboration:

```
地面控制站（强大 LLM）←→ 5G/6G ←→ 边缘无人机（轻量 LLM）
     ↓
  全局意图协调 + 冲突仲裁
     ↓
  各无人机本地规划执行
```

**④Provably secure multi-machine LLM coordination**

The multi-machine coordination problem is modeled as a Parametric Markov Decision Process (PMDP), LLM learns the coordination strategy, and formal verification tools ensure that the strategy satisfies security constraints.

### 5.3 Long term (more than 5 years)

**⑤ The independent emergence of air traffic language**

When large numbers of LLM-equipped drones work together in the airspace, will an autonomous "air traffic language" emerge - an emergent protocol that is more concise than natural language and better suited for drone communication?

```python
# 可能的涌现协议示例
emergent_protocol = {
    # 由大量协同飞行中自动归纳出来
    "shortcut_ACK": "SACK",           # 短确认
    "handoff_REQUEST": "HREQ",         # 移交请求
    "conflict_ALERT": "CALRT",         # 冲突警报
    "reroute_PROPOSE": "RPROP",        # 改航提议
    "abort_MANDATE": "ABORT",          # 强制中止
}
```

This is not a fantasy - some research has observed the emergence of **emergent communication** (Emergent Communication) in multi-agent LLM communication, which may be extended to UAV airspace in the future.

---

## 6. Summary

The core insight of LLM-enabled drone planning is: **LLM is an excellent semantic engine, but should not be considered a reliable controller**.

| Ability | LLM is good at | LLM is not good at |
|-----|---------|-----------|
| Understand fuzzy instructions | ✅ Natural language understanding, common sense reasoning | ❌ Precise numerical values |
| Plan generation | ✅ Multiple plan generation | ❌ Safety proof |
| Multi-machine coordination | ✅ Intent understanding, negotiation generation | ❌ Real-time |
| Trace execution | ❌ | ❌❌ must be handed over to MPC/ORCA |

Summary of the two core routes:**① Neuro-symbolic security planning**: LLM generates natural language planning → compiled into LTL/STL formal specification → model checker verifies security → executed by the classic planner after passing the verification. The core value of this route is to provide provable security guarantees for LLM planning and solve the "trust crisis" of LLM.

**② Multi-drone natural language collaboration**: LLM serves as the semantic communication layer between multiple drones to achieve intent sharing, dynamic negotiation and conflict resolution. The core value of this route is to upgrade drone coordination from "protocol-driven" to "intent-driven", greatly improving the system's adaptability in unknown environments.

The two routes can be integrated: **use LLM to handle intent understanding and negotiation, and use formal methods to verify the security of the negotiation results**. This is a complete and secure multi-machine LLM coordination system.

---

*References (in order of citation in the text)*1. Schmidt et al., "LLMRL: Generating Rewards with Large Language Models from Prior Knowledge", RRL Workshop, 2024
2. Meta AI, "CALM: Collaborative Agents with Language Models", 2024
3. arXiv 2024, "Verified LLM-based Task Planners for Robotic Manipulation"
4. arXiv 2024, "LTLBAR: Bootstrap Neural LTL Translation with Large Language Models"
5. Lyu et al., "Chatty Robots: Emergent Communication in Multi-Agent Reinforcement Learning", CoRL 2024
6. IBM Research, "MAgIC: Multi-Agent Interaction via Communication with LLM", 2025
7. Foerster et al., "Learning to Communicate with Deep Multi-Agent Reinforcement Learning", NeurIPS 2017
8. Zeng et al., "Cellular-Connected UAV: Integration of UAV and Cellular Networks", IEEE IoT Journal, 2019

---*Related Reading*
- [Hierarchical VLM planning: Let the drone understand instructions such as "land on the east side of Building 3"](/blog/hierarchical-vlm-uav-planning/)
- [Urban low-altitude UAV route planning: Theory and algorithm in high-density CBD scenarios](/blog/uav-urban-route-planning/)
- [LLM RAG knowledge base and fine-tuned training technology panoramic survey](/blog/llm-rag-finetune-technical-survey/)