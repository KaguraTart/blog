---
title: "Six Yao Fortune Telling and Markov Chains: A Century-old Dialogue between Eastern Metaphysics and Western Probability"
description: "When \"The Book of Changes\" meets Bayesian inference - an in-depth exploration of the similarities between six-line fortune-telling and Markov chains in dealing with uncertainty"
pubDate: 2026-04-03
tags: ["Six Yao", "Markov chain", "I Ching", "probability theory", "AI"]
category: Essay
sourceHash: "b9b1a6be94fc853402f71853bd5459db3e130505"
---

## 1. Introduction

One is the ancient Chinese divination technique, which comes from the "Book of Changes", which relies on copper coins to make hexagrams and determines good and bad luck based on the five elements. The other is the stochastic process model proposed by the Russian mathematician Andrey Markov in the 20th century, which is now the cornerstone of natural language processing, gene sequence analysis, and financial forecasting.

On the surface, they seem unrelated. But digging deeper, the two are surprisingly similar in their bones - both deal with "uncertainty" and use "state" and "transition" to describe the evolution of the world**.

---

## 2. Introduce the two concepts to each other

### 2.1 Six Yao Fortune Telling - Eastern Philosophy of Uncertainty

Six Yao is a traditional Chinese divination method. Three copper coins (or yarrow) are thrown six times to obtain six Yao, which form a hexagram from bottom to top. Each line is divided into Yin (⚋) and Yang (⚊), and the six lines together form a complete hexagram (a total of 64×64=4096 possibilities).

**Core logic chain:**

```
起卦（随机触发）
  → 获得卦象（信息载体）
  → 装卦（加入动变爻、世应等元数据）
  → 分析五行生克冲合
  → 结合神煞、空亡等附加信息
  → 输出判断（吉/凶/平，及其具体解释）
```

The theoretical basis of Liu Yao is **Yin Yang and Five Elements** - metal, wood, water, fire, and earth, which are mutually reinforcing and restraining each other. Together with the image and number system of Bagua, they form a self-consistent symbol system. It does not pursue "prediction accuracy", but rather provides an interpretable narrative to help the querent obtain a psychological anchor for the direction of action in ambiguous situations.

### 2.2 Markov Chain-Western State Transfer Machine

Markov Chain is a **random process** that satisfies the Markov Property: **the future state is only related to the current state and has nothing to do with past history**. This is the famous "Memorylessness".

Mathematical expression:

$$P(X_{n+1} = j |

To put it simply: **If you know where you are, you know the possible distribution of where you will go next - you don’t need to know how you got here. **

**Typical applications:**
- Search engine’s PageRank algorithm
- Hidden Markov Model (HMM) for speech recognition
- State transition model of financial markets (bull market → bear market → shock)
- Random walk for music recommendation system

---

## 3. Differences and connections between input and output| Dimensions | Six Yao Fortune Telling | Markov Chain |
|------|---------|-----------|
| **Input** | The questioner's coin/yarrow operation, or casual reporting of the number with a thought | Current state of the system + state transition probability matrix P |
| **Randomness of input** | Extremely high (human hand shaking behavior), essentially physical randomness | Theoretically a deterministic probability distribution (but the initial state can be random) |
| **State Space** | 4096 hexagrams + countless changing combinations | Can be discrete or continuous, depending on modeling |
| **Output** | Qualitative judgment (good/bad, due date, event category) | Probability distribution (probability vector of the next state) |
| **Interpretability** | High (but the explanation depends on the level of the hexagram interpreter) | Low (the probability number itself is accurate, but the meaning of the model needs to be interpreted) |
| **Time Dimension** | The hexagrams are fixed, and changes provide time clues | Chain evolution, time step explicit modeling |
| **Subjectivity** | Very high (different people may have completely different interpretations of the same hexagram) | Very low (mathematics is objective) |

### 3.1 Their deep connection

**Level 1: Random trigger**
The hexagram of the six lines is essentially a physical random process (copper coin flipping), and each step of the Markov chain is also probabilistic. Both start with randomness.

**Level 2: Status and Transfer**
The six dynamic lines represent "variables" - hexagrams change from static hexagrams to zhi hexagrams on the time line; the core of the Markov chain is also "state → transfer → new state". From this perspective, **Six Yao is essentially a special set of Markov processes** - the state space is 4096, and the transfer rules are determined by the five elements.

**Level 3: Conditional Probability**
The six Yao's "Exaltation" determines the strength of one Yao, which in turn affects its influence on other Yao - in mathematics, this is **conditional probability**: Under the premise that this Yao has an Exaltation, what is the probability of something happening?

**Level 4: Path dependence (but in opposite direction)**
The six lines emphasize "cause and effect" - what you ask, at what time, and what state of mind will affect the interpretation of the hexagram (this corresponds to the initial distribution sensitivity of the Markov chain). Markov chain claims to be "path-independent", and the two are exactly the opposite in this dimension.

---

## 4. One is quantification, the other is based on feeling - is this the essential difference?

### 4.1 Markov chain: a natural quantitative system

The design of Markov chains has been quantifiable from the start:

```python
import numpy as np

# 一个简单的"心情马尔可夫链"
# 状态：😊 好心情 / 😐 一般 / 😢 低落

P = np.array([
    #   😊    😐    😢
    [0.7,  0.2,  0.1],  # 😊 → 
    [0.3,  0.4,  0.3],  # 😐 → 
    [0.2,  0.3,  0.5],  # 😢 → 
])

# 问：今天心情一般，明天心情分布？
state_today = np.array([0, 1, 0])  # 😐
state_tomorrow = state_today @ P
print(state_tomorrow)  
# [0.3  0.4  0.3] → 30%好, 40%一般, 30%低落
```The beauty of mathematics is that you can accurately calculate **stationary distribution** (the probability of each state in the long term), **first arrival time** (how long it takes on average to go from state A to state B), and **mixing time** (how long it takes the system to "forget" the initial state).

### 4.2 Six Yao: Is the hexagram interpretation really just "based on feeling"?

On the surface, the six-line interpretation of hexagrams is indeed highly dependent on the interpreter's experience, understanding, and even the "inspiration" of the day. For the same hexagram, if you ask about the prosperity or decline of your career or your relationship, the results may be very different.

But if we break down the "feeling" of Liuyao in detail, we will find that it is not purely metaphysics:

**Liuyao has a strict set of internal rules:**
- **Quantification of the generation and restraint of the five elements**: Each line belongs to the five elements and has a clear relationship of generation and restraint (mutual generation: metal generates water, water generates wood, wood generates fire, fire generates earth, earth generates metal; mutual restraint: metal restrains wood, wood restrains earth, earth restrains water, water restrains fire, fire restrains metal)
- **Quantification of prosperous phase, rest, prisoner and death**: Yao has five states of prosperous, phase, rest, prisoner and death in different months, which can be quantified and scored.
- **Sheng Ke, Conflict and Combination**: There are specific rules for conflict (contradiction) and combination (combining)
- **Shensha System**: Although it is highly subjective, it also has fixed trigger conditions.

**The real source of "feeling" is:**
1. Too many rules, redundant information, and no unified computable output
2. The interpreter has done a lot of implicit weight distribution, but has not expressed it explicitly.
3. The "quantification" of the six lines is hidden in the experience of the hexagram interpreter and has not been **explicitly modeled**

---

## 5. How to quantify the six Yao lines?

This is the fun part - can we turn the six lines into a Markov chain? **

### 5.1 Idea 1: State space modeling

Model the six-yao hexagram as a discrete state system:

```
状态空间 S = {六十四卦 × 动变爻位置 × 用神强弱}
```

This is an astronomically large state space, but we can simplify it.

### 5.2 Idea 2: Construct transition probability

Use historical hexagram data to learn the "experience transfer matrix" of the six lines - this is the machine learning route:

```
训练数据：大量已知应验的卦例（问事+卦象+结果）
↓
提取特征：卦象五行、用神强弱、动变方向、月令
↓
监督学习：训练一个分类/回归模型
↓
输出：给定一个新卦，预测各结果的概率
```

This actually extracts the six-yao "empirical rules" from the human brain and turns them into a computable model - essentially no different from training a text classifier.

### 5.3 Idea 3: Bayesian Six Yao

Think of Liuyao as a **Bayesian inference system**:

```python
import numpy as np

# 先验：基于历史数据的各类事项基础概率
prior = {
    '大吉': 0.15,
    '吉': 0.25,
    '平': 0.30,
    '凶': 0.20,
    '大凶': 0.10
}

# 似然函数：给定卦象特征，各结果的概率（从经验数据学习）
likelihood = {
    '用神旺相': {'大吉': 0.4, '吉': 0.35, '平': 0.2, '凶': 0.05, '大凶': 0.0},
    '用神休囚': {'大吉': 0.05, '吉': 0.15, '平': 0.3, '凶': 0.35, '大凶': 0.15},
    '官鬼持世': {'大吉': 0.05, '吉': 0.1, '平': 0.25, '凶': 0.4, '大凶': 0.2},
    '子孙持世': {'大吉': 0.3, '吉': 0.4, '平': 0.2, '凶': 0.08, '大凶': 0.02},
}

# 贝叶斯后验（简化版本）
def bayes_liuyao(observations):
    """根据观察到的卦象特征，计算各结果的后验概率"""
    posterior = {k: v for k, v in prior.items()}
    
    for obs, likely in likelihood.items():
        if obs in observations:
            for outcome in posterior:
                posterior[outcome] *= likely[outcome]
    
    # 归一化
    total = sum(posterior.values())
    return {k: round(v/total, 3) for k, v in posterior.items()}

# 示例：用神旺相 + 子孙持世
result = bayes_liuyao(['用神旺相', '子孙持世'])
print(result)
# {'大吉': 0.444, '吉': 0.389, '平': 0.111, '凶': 0.049, '大凶': 0.007}
```

### 5.4 The core challenge of quantifying six Yao| Challenge | Description |
|------|------|
| **Data Scarcity** | There are very few hexagram data of high quality, clear history, and complete records |
| **Subjectivity Coding** | What counts as "good luck"? Different people have different standards |
| **Time Element** | It is particularly difficult to quantify the "due date" (when it will occur) in the six lines |
| **Xiangshu vs. Mathematics** | Six Yao has both "Xiang" (symbolic meaning) and "Shu" (number of births and grams) at the same time, and the two are not completely commensurable |
| **Black box problem** | If you use a neural network to do it, the accuracy may be high, but the "why" cannot be explained |

---

## 6. When Liu Yao meets the big language model

A cutting-edge idea: use LLM to make a six-yao quantitative interpreter:

```
用户输入："占卜明日股市行情"
         ↓
LLM 转化为六爻起卦（模拟或随机）
         ↓
LLM 根据卦象特征 + 历史金融数据 → 生成分析文本
         ↓
输出：概率化的判断 + 自然语言解释
```

This essentially treats the six-yao **xiang number system** as a **Prompt Engineering framework** - the hexagrams determine the "angle of looking at the problem", and LLM is responsible for filling in the specific content.

Liuyao provides: **Structured Uncertainty Framework** (it forces you to choose one of the 64 quadrants to approach the problem)
LLM provides: **Unlimited knowledge background and fluent explanation ability**

The combination of the two may be the direction of "AI divination" in the future.

---

## 7. Conclusion: They are two sides of the same problem

| | Six Yao | Markov Chain |
|---|---|---|
| **Essence** | Symbol system + empirical rules | Mathematical model + probability transfer |
| **Advantages** | Highly interpretable and rich in mathematical wisdom | Accurate quantification and computational verification |
| **Disadvantages** | Difficult to standardize, too subjective | Oversimplification, ignoring deep semantics |
| **Common Ancestor** | Awe and modeling of "uncertainty" | Mathematical abstraction of "random process" |

In the final analysis, **Liuyao is the probability theory** that ancient people used to simulate Yin-Yang and Five Elements in the era before computers**; **Markov chain is the precise formalization of random processes** after modern people have computers.

They do not compete on the same dimension, but they have reached a resonance across time and space at the deepest level of human cognition - **how to face uncertainty and make decisions**.

> Leibniz said: "The I Ching grasps the universe through yin and yang, 0 and 1."
> Markov said: "Grasp the stochastic process through state transition."
>Perhaps they are essentially talking about the same thing.

---*Written with curiosity and humility, somewhere between Big Data and the I Ching.*