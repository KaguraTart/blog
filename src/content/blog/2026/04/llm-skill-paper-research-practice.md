---
title: "从幻觉到实事求是的学术研究工作流：我用 OpenClaw Skills 搭了一套论文追踪系统"
description: "记录我如何设计 paper-research + paper-verifier 两个 OpenClaw Skill，构建一套强调「真实可验证」的学术文献研究工作流。核心原则：不生成虚假文献，手动搜索 + 工具辅助整理，配合 Zotero 管理，形成从检索到综述的完整闭环。"
tags: ["OpenClaw", "Skill", "学术研究", "LLM", "Zotero", "文献管理", "RAG", "知识管理"]
pubDate: 2026-04-15
---

# 从幻觉到实事求是的学术研究工作流：我用 OpenClaw Skills 搭了一套论文追踪系统

## 0. 缘起：一次差点越线的教训

2026 年 3 月，我让 AI 帮我写论文。过程很顺利——论文结构清晰、图表精美、数据看起来非常合理。

然后我发现：**那些数据是 AI 生成的，不是真实实验跑出来的。**

92% 成功率、500 无人机的可扩展性分析——看起来无懈可击，但全是"合理推演"。如果就这样提交，就是学术造假。

这次经历让我彻底想清楚了一件事：**LLM 最大的危险不是答不上来，而是答得很自信但答错了。在学术场景下，这种自信是致命的。**

所以我重新设计了文献研究的工作流。核心原则只有一条：

> **不生成任何不可验证的内容。所有文献，必须手动搜索、真实存在；所有数据，必须有据可查。**

这就是 `paper-research` 和 `paper-verifier` 这两个 Skill 的设计初衷。

## 1. 问题的核心：为什么 AI 直接搜文献靠不住？

Google Scholar 有严格的反爬虫机制。Selenium / Playwright 自动化操作容易被封 IP、触发验证码，结果不稳定。更关键的是：**AI 搜索出来的论文，可能根本不存在**——LLM 擅长"听起来合理"的幻觉式回答，在文献综述里，这是最危险的雷区。

常见的虚假文献套路：
- 编造论文标题和作者
- 捏造不存在的期刊会议
- 夸大引用数或影响因子
- 把普通期刊说成一区顶刊

所以，我选择了一个看起来"笨"但绝对可靠的方式：**人机协作，以人为主**。

## 2. 整体设计：四步闭环工作流

```
手动搜索文献
     ↓
交互式收集元数据
     ↓
LLM 辅助生成综述
     ↓
Zotero 统一管理
     ↓
paper-verifier 交叉验证（可选）
```

**核心逻辑**：人负责"找真实的东西"，LLM 负责"高效地整理它们"。LLM 永远不生成内容，只组织和呈现人输入的真实信息。

## 3. paper-research Skill：完整的文献研究工作流

### 3.1 快速初始化

```bash
# 创建研究工作区
./scripts/setup_manual_search.sh ./my_research "UAV path planning"
```

这会在指定目录生成三个文件：

```
my_research/
├── 搜索指南.md         # 告诉你去哪搜、怎么搜
├── 文献记录模板.json   # 论文元数据存储格式
└── 收集文献.sh         # 交互式收集脚本
```

### 3.2 搜索指南：带策略的手动检索指导

`搜索指南.md` 不是简单的"去 Google Scholar 搜"——它包含：

**推荐数据库**：
- Google Scholar（综合，最全）
- IEEE Xplore（工科权威）
- Web of Science（顶刊索引）
- ACM Digital Library（计算机）

**搜索策略**：
```bash
# 示例：2023-2025 年 UAV 路径规划相关一区论文
"UAV path planning" AND "low altitude" AND year:2023..2025
site:ieeexplore.ieee.org "urban air mobility"
```

**筛选标准**：
- 按引用数排序，优先高被引论文
- 验证期刊是否在中科院一区（fenqubiao.com 查询）
- 记录 DOI 或可访问链接

**中科院一区参考列表**（内置在 Skill 里）：

| 类型 | 代表期刊 |
|------|---------|
| IEEE Transactions | TRO, TITS, TAE, TCST |
| 机器人顶刊 | Automatica, JFR, RAS |
| 航空顶会 | ICRA, IROS, AIAA SciTech |

### 3.3 交互式文献收集

有两种方式，推荐用交互式脚本：

```bash
cd my_research
./收集文献.sh
```

脚本会逐步提示输入每篇论文的信息：

```
===== 文献收集 =====
论文标题: Multi-Agent Path Planning for UAV Swarms
作者: Zhang S, Li M, Wang W
年份: 2024
期刊/会议: IEEE Transactions on Robotics
DOI: 10.1109/TRO.2024.3391285
引用数: 45
关键词: UAV, path planning, multi-agent, reinforcement learning
一句话总结: 用集中式训练+分布式执行的框架解决
            无人机集群路径冲突问题...
继续添加下一篇? (y/n):
```

如果嫌慢，也可以直接编辑 `文献记录模板.json`：

```json
[
  {
    "title": "Multi-Agent Path Planning for UAV Swarms",
    "authors": ["Zhang S", "Li M", "Wang W"],
    "year": 2024,
    "venue": "IEEE Transactions on Robotics",
    "doi": "10.1109/TRO.2024.3391285",
    "citations": 45,
    "keywords": ["UAV", "path planning", "multi-agent"],
    "summary": "集中式训练+分布式执行框架解决无人机集群路径冲突"
  }
]
```

### 3.4 从 JSON 生成 Markdown 文献综述

收集完成后，一键生成结构化综述：

```bash
python3 scripts/paper_collection.py \
    --input 文献记录模板.json \
    --output-md 文献综述.md
```

输出格式大致如下：

```markdown
## 1. 研究背景

### 1.1 无人机集群路径规划

无人机集群路径规划是城市低空空域管理的核心问题...
Zhang et al. (2024) 提出的集中式训练+分布式执行框架...

### 1.2 多智能体强化学习

MARL 是解决分布式协同决策的主流方法...

## 2. 方法论分类

### 2.1 基于优化的方法
...

### 2.2 基于学习的方法
...

## 3. 关键文献汇总

| 论文 | 年份 | venue | 方法 | 贡献 |
|------|------|-------|------|------|
| Zhang et al. | 2024 | TRO | MARL | 提出 MGAT-AC 架构 |
...
```

LLM 的作用在这里体现得恰到好处——**不是替你找论文，而是把人工收集的真实信息组织成结构化的综述文本**。

### 3.5 Zotero 同步：建立个人文献数据库

收集的文献可以一键导入 Zotero：

```bash
export ZOTERO_LIBRARY_ID="你的图书馆ID"
export ZOTERO_API_KEY="你的API Key"

python3 scripts/zotero_manager.py \
    --library-id $ZOTERO_LIBRARY_ID \
    --api-key $ZOTERO_API_KEY \
    batch-add --file 收集的文献.json
```

这样，所有手动收集的真实文献都会同步到 Zotero，形成可复用的个人文献库。每次做新研究，都可以在已有库的基础上增量添加。

## 4. paper-verifier Skill：真实性交叉验证

收集完文献，还需要验证。这是第二个 Skill——**论文真实性核查工具**。

### 4.1 为什么需要验证？

手动搜索也有失误的可能：
- DOI 填错了
- 作者名拼写不对
- 期刊名写成了缩写而非全称
- 引用数记错了（差了几十次）
- 以为是一区结果其实只是 CCF-B

所以在综述定稿前，用 paper-verifier 做一次交叉核验。

### 4.2 DOI + 元数据核查

```bash
python3 scripts/verify_papers.py \
    --input papers.json \
    --output verification_report.md
```

`papers.json` 格式：

```json
{
  "title": "Multi-Agent Path Planning for UAV Swarms",
  "authors": "Zhang S, Li M",
  "year": 2024,
  "venue": "IEEE Transactions on Robotics",
  "doi": "10.1109/TRO.2024.3391285",
  "citations": 45
}
```

脚本会通过两个 API 核查：

**Crossref API** — 获取官方元数据：
- 标题是否完全匹配
- 作者列表是否正确
- 年份是否一致
- 期刊全称是什么

**Semantic Scholar API** — 二次验证：
- 论文是否真实存在
- 实际引用数是多少
- 获取论文摘要用于核对内容

### 4.3 核查报告格式

```
## 核查报告

### ✓ Zhang et al. (2024) - TRO
- DOI: 10.1109/TRO.2024.3391285 → 有效
- 期刊: IEEE Transactions on Robotics → 中科院一区 ✓
- 引用数: 声称45 → 实际52（Semantic Scholar）
  ⚠️ 引用数有出入，差7次
- 作者: Zhang S, Li M, Wang W → 核对通过 ✓

### ✗ Li et al. (2023) - ICRA
- DOI: 10.1109/ICRA.2023.1001234 → 有效
- ⚠️ 论文标题不匹配：实际为 "Single-Agent ..."
  → 请核实是否填错了论文
```

通过这种方式，任何手动输入的错误都能被及时发现。

## 5. 两个 Skill 的文件结构

```
~/.openclaw/workspace/skills/
├── paper-research/          # 文献研究工作流
│   ├── SKILL.md            # 使用说明
│   ├── references/
│   │   └── zotero_setup.md # Zotero 配置指南
│   └── scripts/
│       ├── setup_manual_search.sh    # 工作区初始化
│       ├── paper_collection.py       # 文献收集+综述生成
│       ├── zotero_manager.py         # Zotero API 同步
│       ├── search_scholar.py         # Scholar 搜索（需API）
│       ├── search_serpapi.py         # SerpAPI 搜索
│       └── generate_summary.py       # 综述文本生成
│
└── paper-verifier/         # 真实性核查
    ├── SKILL.md
    └── scripts/
        └── verify_papers.py # 核心核查脚本
```

## 6. 与 RAG 知识库的联动

收集的真实文献，不仅可以生成综述，还可以进入 **LLMRAG 知识库**，用于后续的问答和写作辅助。

整个联动流程：

```
手动搜索 → 收集到 JSON → 生成 Markdown 综述
    ↓
导入 Zotero（长期管理）
    ↓
添加到 LLMRA G知识库（向量检索）
    ↓
后续论文写作时，RAG 问答检索真实文献
```

这样，文献研究不是一次性的工作，而是形成了**可积累、可检索、可复用的知识资产**。

## 7. 设计哲学总结

这个工作流的设计，回答了一个根本问题：**LLM 在学术研究中的边界在哪里？**

| LLM 擅长 | LLM 不擅长（必须人来） |
|---------|----------------------|
| 整理和组织信息 | 判断文献是否真实存在 |
| 生成结构化文本 | 验证 DOI / 引用数的准确性 |
| 查找知识空白和关联 | 在权威数据库检索 |
| 润色和改写 | 决定哪些论文值得纳入综述 |

**底线**：凡是需要"真实性保证"的环节，都由人负责；LLM 只负责人在真实输入之上的整理和生成。

这也和我之前论文事件的教训一脉相承：**不让 LLM 产生任何无法追溯到源头的内容**。每个数据、每篇文献，都必须有迹可循。

## 8. 未来扩展方向

- [ ] 接入 SerpAPI 实现半自动 Google Scholar 搜索（减少手动操作）
- [ ] 与 Zotero API 双向同步，自动拉取已读论文元数据
- [ ] 构建领域专属的一区期刊/会议知识库，内置引用格式模板
- [ ] 将已验证文献自动转为 Obsidian 笔记格式，融入双链笔记生态

---

*Skill 路径：`~/.openclaw/workspace/skills/paper-research/` 和 `paper-verifier/`*

*作者：Kagura Tart | 2026-04-15 | 写给未来的自己和所有需要做文献综述的研究者*
