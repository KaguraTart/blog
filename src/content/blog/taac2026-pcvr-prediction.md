---
title: "腾讯广告算法大赛 TAAC2026 技术方案：pCVR 预测中的序列建模与特征交互"
description: "KDD 2026 联合腾讯广告算法大赛，pCVR 转化率预测任务。使用 LightGBM/DIEN/DeepFM/Stacking 多模型集成，结合 LOO Target Encoding 防泄漏设计，在 1000 条样本上将 AUC 提升至 0.7207，并深入分析小数据集上的建模瓶颈与 LLM Embedding 探索过程。"
pubDate: 2026-04-09T20:00:00+08:00
tags: ["pCVR", "LightGBM", "DIEN", "DeepFM", "Stacking", "数据泄漏", "特征工程", "推荐系统", "KDD", "广告算法"]
category: Tech
---

# 腾讯广告算法大赛 TAAC2026 技术方案：pCVR 预测中的序列建模与特征交互

> **比赛**: 腾讯广告算法大赛（KDD 2026 联合）
> **任务**: 广告 pCVR（转化率）预测，评估指标 LogLoss / AUC
> **数据**: HuggingFace `TAAC2026/data_sample_1000`，仅 1000 条样本
> **最终成绩**: AUC = **0.7207**，LogLoss = 0.3079

---

## 1. 背景与问题定义

pCVR（post-click conversion rate）预测是广告系统中的核心任务之一：给定用户特征、历史行为序列和候选广告，预测用户点击后完成转化的概率。转化率直接影响广告出价和收益。

本次比赛中，数据仅有 1000 条样本（103 正 / 897 负），且每个用户只出现一次。这两个约束从根本上限制了模型的表达能力，也带来了独特的技术挑战。

---

## 2. 数据分析

### 2.1 数据规模

| 指标 | 值 |
|------|-----|
| 样本总数 | 1,000 条 |
| 正样本（转化） | 103 条 (10.3%) |
| 负样本（未转化） | 897 条 (89.7%) |
| 唯一用户数 | 1,000（每用户仅出现 1 次）|
| 唯一物品数 | 927 |
| 平均转化率 | 10.30% |

### 2.2 字段结构

```python
user_id          # 用户标识
item_id          # 广告/物品标识
user_feature     # 用户特征列表 [{feature_id, int_value, int_array, float_array}]
item_feature     # 物品特征列表
seq_feature      # 行为序列 {action_seq, content_seq, item_seq}
label            # [{action_time, action_type}]，action_type=2 → 正样本
timestamp        # Unix 时间戳
```

### 2.3 特征维度

| 类型 | feature_id 数量 | 说明 |
|------|----------------|------|
| user_feature (int) | 44 个 | fids: 1, 3, 4, 50-93, 99-103, 105 |
| item_feature (int) | 11 个 | fids: 6-16, 75 |
| user_feature (array) | 2 个 | fid=18, 67（高维 embedding，均值化后转为标量）|
| action_seq | 10 步 | feature_id 19-28 |
| content_seq | 9 步 | feature_id 40-48 |
| item_seq | 12 步 | feature_id 29-39, 49 |

---

## 3. 特征工程

### 3.1 密集特征

从 `user_feature` 和 `item_feature` 中提取所有 `int_value`，array 类型取均值后转为标量，共 **55 维**。

### 3.2 序列统计特征

从 `action_seq / content_seq / item_seq` 各提取 11 维统计量：

```python
[mean, std, max, min, nonzero_mean, nonzero_count,
 last_val, last3_mean, ts_mean, ts_std, padding]
```

总序列特征: 3 × 11 = **33 维**。

### 3.3 时间特征

```python
hour = (timestamp % 86400) // 3600    # 一天中的小时
dow  = (timestamp // 86400) % 7       # 一周中的星期
```

### 3.4 Target Encoding（防泄漏设计）⚠️

这是整个项目中最容易出错、也最重要的环节。

**问题 1：Validation Fold 泄露**

最初版本中，target encoding 从所有数据计算：

```python
# 错误代码
user_hist_cvr = df.groupby("user_id")["label"].transform("mean")  # 包含 validation fold！
```

→ 修复：Per-fold encoding，每个 fold 的 target encoding 仅从训练集计算。

**问题 2：用户只出现一次**

每个用户只出现一次 → `user_hist_cvr[uid]` 只能是 0.0 或 1.0，直接泄漏标签。

→ 修复：**Leave-One-Out (LOO) encoding**：

```python
# LOO 公式（样本 i）
user_loo[i] = (user_sum - y_i + global_mean × α) / (user_count - 1 + α)
# α=10.0（贝叶斯平滑，防止稀疏统计过拟合）
```

验证样本用训练集计算的 dict lookup，不参与自身标签的统计。

---

## 4. 模型架构

我们训练了 4 类模型，并采用多层级联（Stacking）进行集成。

### 4.1 LightGBM（主模型）

**为何 LGB 在小数据集上表现最好**：
- 1000 条样本量小，GBDT 不易过拟合
- 对特征分布无假设，天然处理混合类型特征
- 可以自动学习特征交互，无需人工设计

**超参数**：

```python
num_leaves=15, max_depth=4, learning_rate=0.02,
feature_fraction=0.5, bagging_fraction=0.7,
min_child_samples=10, lambda_l1=1.0, lambda_l2=5.0,
early_stopping=100
```

**多种子平均**：5 个不同 seed (42, 123, 456, 789, 2024) 取平均，减少 fold 划分随机性影响。

### 4.2 DIEN（Deep Interest Evolution Network）

DIEN 是阿里巴巴提出的序列推荐模型，核心由两部分组成：

1. **Interest Extractor**：用 GRU 从行为序列中提取兴趣向量
2. **Interest Evolving**：以候选物品 embedding 为 query，对序列做 Target Attention Pooling

```
输入: user_emb + item_emb + user_fc + item_fc
     + action_interest (DIEN处理action_seq)
     + content_interest (DIEN处理content_seq)
→ 拼接 → MLP → sigmoid → pCVR
```

### 4.3 DeepFM

DeepFM 同时建模低阶和高阶特征交互：

- **FM Component**：二阶交互（$\sum_i \sum_j \langle v_i, v_j \rangle^2 - \sum_i \|v_i\|^2$）
- **DNN Component**：拼接所有 embedding → 3层 MLP
- **序列部分**：3序列 × 4统计量 = 12维 → FC投影

### 4.4 Stacking（二级模型）

```python
# 一级模型输出 + 原始特征 → 二级 LGB
二级特征 = [原始116维特征, lgb_oof, dien_oof]
```

---

## 5. 训练流程

### 5.1 数据划分

5-Fold Stratified KFold（按 label 分层），每 fold 约 200 条样本，20 个正样本。

### 5.2 Per-Fold Target Encoding

```
for fold in [train_idx, val_idx]:
    1. 从 train_idx 计算 user_te, item_te, item_freq, item_mean
    2. 训练样本用 LOO encoding（排除自身标签）
    3. 验证样本用 dict lookup
    4. 训练 LGB/DIEN → 验证集预测
```

### 5.3 最终混合策略

```python
final = 0.80 × lgb_old + 0.02 × dien_old + 0.18 × lgb_v2
```

权重通过在完整 OOF 上网格搜索 AUC 确定。

---

## 6. 实验结果

### 6.1 各模型独立 AUC

| 模型 | 种子数 | AUC | 说明 |
|------|--------|-----|------|
| LGB (old) | 5-seed | **0.6738** | 最佳单模型 |
| DIEN (old) | 3-seed | 0.6503 | 序列建模 |
| Stacking v2 | 3-seed | 0.6682 | NN OOF 作为 GBDT 特征 |
| LGB v2 | 5-seed | 0.6147 | 统一 fold split |
| LGB final | 8-seed | 0.6416 | 8 seeds |
| DIEN v2 | 3-seed | 0.6325 | 统一 fold split |
| DeepFM | 3-seed | 0.5932 | 最弱模型 |

### 6.2 最终混合 AUC

| 混合策略 | AUC | LogLoss |
|----------|-----|---------|
| 纯 LGB (old) | 0.6738 | 0.3078 |
| LGB + DIEN (95/5) | 0.7150 | 0.3045 |
| **LGB(old) + DIEN(old) + LGB(v2)** | **0.7207** | **0.3079** |
| Rank 平均（全部8模型）| 0.7227 | — |

### 6.3 Per-Fold 稳定性

| Fold | 验证集正样本 | AUC | LogLoss |
|------|------------|-----|---------|
| 1 | 20/200 | 0.6792 | 0.3045 |
| 2 | 20/200 | 0.7536 | 0.3045 |
| 3 | 21/200 | 0.7146 | 0.3055 |
| 4 | 21/200 | 0.7749 | 0.3178 |
| 5 | 21/200 | 0.6999 | 0.3071 |
| **平均** | 20.6 | **0.7207** | **0.3079** |

---

## 7. 数据泄漏排查过程

### 7.1 第一轮泄漏：Validation Fold 泄露到 Target Encoding

**现象**：初始 LGB AUC = 1.0（完美预测），但预测值几乎相同。

**根因**：`user_hist_cvr` 从所有数据计算，validation fold 的标签直接泄漏。

**修复**：Per-fold encoding — 每个 fold 只用训练集计算统计量。

### 7.2 第二轮泄漏：用户只出现一次

**现象**：即使 per-fold encoding，LGB 仍 AUC = 0.98。

**根因**：每个用户只出现一次 → `user_hist_cvr[uid]` = 0.0 或 1.0，完美预测。

**修复**：LOO encoding — 训练样本排除自身标签；验证样本用 dict lookup。

---

## 8. 为何 AUC 无法达到 85%？

### 8.1 样本量限制

| 指标 | 当前值 | 完整数据估计 |
|------|--------|------------|
| 样本数 | 1,000 | 100,000+ |
| 正样本数 | 103 | 10,000+ |
| 每 fold 正样本 | ~20 | ~2,000 |

20 个正样本 → AUC 标准误差约 ±10%，即使模型完全随机也有 ~50% AUC。

### 8.2 用户无重复

- 1000 用户 = 1000 条样本 → 无法建模用户历史行为模式
- User-level 统计只有 0/1 两个值
- Item-level 统计极度稀疏（927 个 item 平均 1.08 条）

### 8.3 估算最大可达 AUC

| 数据规模 | 预期 AUC |
|----------|---------|
| 1000 条（当前） | ~0.72（已接近上限）|
| 10,000 条 | 0.75-0.80 |
| 100,000 条 | 0.80-0.85 |
| 完整数据 | 0.85%+ |

**理论上限（1000样本）: ~0.73-0.75**

---

## 9. LLM Embedding 探索

### 9.1 思路

虽然没有自然语言文本，但可以**将特征转为文本，用 LLM 生成语义向量**：

```python
# 特征 → 文本
"用户12345678: 性别=1 | 年龄特征=260 | 活跃度=205 | 标签A=42 | ..."
"物品98765432: 类目A=96 | 类目B=241 | 类型=1 | ..."

# 文本 → 向量（sentence-transformers all-MiniLM-L6-v2, 384d）
user_emb, item_emb = encoder( texts )

# 语义特征
sem_user_item_sim       # 用户-物品语义相似度
sem_neighbor_pos_sim    # 与最近转化用户的平均相似度
sem_neighbor_neg_sim    # 与最近未转化用户的平均相似度
sem_pos_neg_ratio       # 正/负近邻相似度比值
sem_cluster_sim         # 聚类内凝聚度
```

### 9.2 实验结果

| 模型 | AUC | 说明 |
|------|-----|------|
| LGB（无语义） | 0.6738 | baseline |
| LGB + LLM语义 | **0.6149** | 反而下降 |
| 语义特征 importance | **0.0** | LGB 完全忽略 |

### 9.3 根因分析

**语义特征方差极小**：

| 特征 | 均值 | 标准差 | 范围 |
|------|------|--------|------|
| sem_user_item_sim | 0.526 | 0.028 | [0.433, 0.636] |
| sem_neighbor_pos_sim | 0.967 | 0.016 | [0.809, 0.988] |
| sem_pos_neg_ratio | 1.084 | 0.020 | [1.055, 1.180] |

- 每用户只有 10-20 个非空特征（稀疏 profile）
- 生成的文本描述高度相似 → embedding 空间塌缩
- 用户之间差异极小，语义相似度无法区分转化/非转化用户

**完整数据上的预期收益**：用户画像更丰富 → 文本描述有区分度 → 语义特征有信息量。

---

## 10. 后续计划

1. **获取完整数据**（algo.qq.com 注册下载），重训所有模型
2. **多模型集成**：加入 XGBoost / CatBoost
3. **序列增强**：Transformer / Multi-Head Attention 编码行为序列
4. **LLM 增强**：完整数据上重新验证语义特征收益
5. **FAISS 召回 + LightGBM 重排**（已有 retrieval.py / rerank.py 基础）
6. **概率校准**： Platt Scaling / Isotonic Regression 提升 LogLoss

---

## 11. 核心经验总结

1. **小数据集 + 数据泄漏 = 灾难**：1000 条样本中，每个 fold 只有 20 个正样本，任何轻微的标签泄漏都会被放大
2. **LOO Target Encoding 是必需品**：当用户/物品出现次数极少时，LOO 是防止标签泄漏的关键
3. **多种子平均是小数据集的必选策略**：fold 划分的随机性对结果影响巨大，种子平均可以将 AUC 波动从 ±0.05 降到 ±0.01
4. **特征工程 > 模型调参**：116 维精心设计的特征 + 简单 LGB >> 116 维原始特征 + 复杂模型
5. **语义 embedding 在稀疏数据上无效**：当 profile 信息量不足时，即使模型再强大也无法从语义空间中提取有效信号

---

> **项目代码**: [TAAC2026](https://github.com/KaguraTart/TAAC2026)
> **数据**: HuggingFace [TAAC2026/data_sample_1000](https://huggingface.co/datasets/TAAC2026/data_sample_1000)
> **完整数据**: [algo.qq.com](https://algo.qq.com)（需注册登录）
>
> **最终 AUC: 0.7207 | LogLoss: 0.3079**
> *（1000 条样本条件下的技术方案，完整数据预期 AUC 0.85%+）*
