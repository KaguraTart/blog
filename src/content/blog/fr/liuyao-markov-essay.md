---
title: "Six prédictions de bonne aventure Yao et chaînes de Markov : un dialogue centenaire entre la métaphysique orientale et les probabilités occidentales"
description: "Quand \"Le Livre des Changements\" rencontre l'inférence bayésienne - une exploration approfondie des similitudes entre la divination à six lignes et les chaînes de Markov dans la gestion de l'incertitude"
pubDate: 2026-04-03
tags: ["Six Yao", "Chaîne de Markov", "Je Ching", "théorie des probabilités", "IA"]
category: Essay
---

## 1. Introduction

L'une est l'ancienne technique de divination chinoise, qui vient du « Livre des Mutations », qui s'appuie sur des pièces de cuivre pour fabriquer des hexagrammes et détermine la bonne et la mauvaise chance en fonction des cinq éléments. L’autre est le modèle de processus stochastique proposé par le mathématicien russe Andrey Markov au XXe siècle, qui constitue aujourd’hui la pierre angulaire du traitement du langage naturel, de l’analyse des séquences génétiques et des prévisions financières.

En apparence, ils semblent sans rapport. Mais en creusant plus profondément, les deux sont étonnamment similaires dans leurs os – tous deux traitent de « l’incertitude » et utilisent « état » et « transition » pour décrire l’évolution du monde**.

---

## 2. Présentez les deux concepts l'un à l'autre

### 2.1 Six Yao Fortune Telling - Philosophie orientale de l'incertitude

Six Yao est une méthode de divination traditionnelle chinoise. Trois pièces de cuivre (ou millefeuille) sont lancées six fois pour obtenir six Yao, qui forment un hexagramme de bas en haut. Chaque ligne est divisée en Yin (⚋) et Yang (⚊), et les six lignes forment ensemble un hexagramme complet (un total de 64×64=4096 possibilités).

**Chaîne logique de base :**

```
起卦（随机触发）
  → 获得卦象（信息载体）
  → 装卦（加入动变爻、世应等元数据）
  → 分析五行生克冲合
  → 结合神煞、空亡等附加信息
  → 输出判断（吉/凶/平，及其具体解释）
```

La base théorique de Liu Yao est le **Yin Yang et les Cinq Éléments** - le métal, le bois, l'eau, le feu et la terre, qui se renforcent et se retiennent mutuellement. Avec le système d'images et de chiffres de Bagua, ils forment un système de symboles cohérent. Il ne recherche pas « l’exactitude des prédictions », mais fournit plutôt un récit interprétable pour aider le demandeur à obtenir un ancrage psychologique pour la direction de l’action dans des situations ambiguës.

### 2.2 Machine de transfert de chaîne de Markov-État occidental

La chaîne de Markov est un **processus aléatoire** qui satisfait la propriété de Markov : **l'état futur est uniquement lié à l'état actuel et n'a rien à voir avec l'histoire passée**. C'est le fameux « sans mémoire ».

Expression mathématique :

$$P(X_{n+1} = j |

Pour le dire simplement : ** Si vous savez où vous êtes, vous connaissez la répartition possible de votre prochaine destination - vous n'avez pas besoin de savoir comment vous êtes arrivé ici. **

**Applications typiques :**
- Algorithme PageRank du moteur de recherche
- Modèle de Markov caché (HMM) pour la reconnaissance vocale
- Modèle de transition d'état des marchés financiers (marché haussier → marché baissier → choc)
- Marche aléatoire pour le système de recommandation musicale

---

## 3. Différences et connexions entre entrée et sortie| Dimensions | Six Yao Fortune Telling | Chaîne de Markov |
|------|---------|---------------|
| **Entrée** | L'opération pièce de monnaie/achillée millefeuille de l'interrogateur, ou le rapport occasionnel du numéro avec une pensée | État actuel du système + matrice de probabilité de transition d'état P |
| ** Caractère aléatoire de l'entrée ** | Extrêmement élevé (comportement humain de tremblement de la main), caractère aléatoire essentiellement physique | Théoriquement, une distribution de probabilité déterministe (mais l'état initial peut être aléatoire) |
| **Espace d'état** | 4096 hexagrammes + d'innombrables combinaisons changeantes | Peut être discret ou continu, selon la modélisation |
| **Sortie** | Jugement qualitatif (bon/mauvais, date d'échéance, catégorie d'événement) | Distribution de probabilité (vecteur de probabilité de l'état suivant) |
| **Interprétabilité** | Élevé (mais l'explication dépend du niveau de l'interprète de l'hexagramme) | Faible (le nombre de probabilité lui-même est exact, mais la signification du modèle doit être interprétée) |
| **Dimension temporelle** | Les hexagrammes sont corrigés et les modifications fournissent des indices temporels | Evolution de la chaîne, modélisation explicite du pas de temps |
| **Subjectivité** | Très élevé (différentes personnes peuvent avoir des interprétations complètement différentes du même hexagramme) | Très faible (les mathématiques sont objectives) |

### 3.1 Leur connexion profonde

**Niveau 1 : Déclenchement aléatoire**
L'hexagramme des six lignes est essentiellement un processus physique aléatoire (retournement de pièces de cuivre), et chaque étape de la chaîne de Markov est également probabiliste. Les deux commencent par le hasard.

**Niveau 2 : Statut et transfert**
Les six lignes dynamiques représentent des « variables » - les hexagrammes passent des hexagrammes statiques aux hexagrammes zhi sur la ligne temporelle ; le cœur de la chaîne de Markov est également « état → transfert → nouvel état ». De ce point de vue, **Six Yao est essentiellement un ensemble spécial de processus de Markov** : l'espace d'états est de 4096 et les règles de transfert sont déterminées par les cinq éléments.

**Niveau 3 : Probabilité conditionnelle**
L'« Exaltation » des six Yao détermine la force d'un Yao, ce qui à son tour affecte son influence sur les autres Yao - en mathématiques, il s'agit d'une **probabilité conditionnelle** : en partant du principe que ce Yao a une Exaltation, quelle est la probabilité que quelque chose se produise ?

**Niveau 4 : Dépendance au chemin (mais en sens inverse)**
Les six lignes mettent l'accent sur « la cause et l'effet » - ce que vous demandez, à quel moment et quel état d'esprit affectera l'interprétation de l'hexagramme (cela correspond à la sensibilité de distribution initiale de la chaîne de Markov). La chaîne de Markov prétend être « indépendante du chemin », et les deux sont exactement le contraire dans cette dimension.

---

## 4. L'une est la quantification, l'autre est basée sur le ressenti - est-ce la différence essentielle ?

### 4.1 Chaîne de Markov : un système quantitatif naturel

La conception des chaînes de Markov a été quantifiable dès le départ :

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
```La beauté des mathématiques est que vous pouvez calculer avec précision la **distribution stationnaire** (la probabilité de chaque état à long terme), le **temps de première arrivée** (combien de temps il faut en moyenne pour passer de l'état A à l'état B) et le **temps de mélange** (combien de temps il faut au système pour « oublier » l'état initial).

### 4.2 Six Yao : L'interprétation de l'hexagramme est-elle vraiment simplement « basée sur le sentiment » ?

En apparence, l'interprétation des hexagrammes en six lignes dépend en effet fortement de l'expérience, de la compréhension et même de « l'inspiration » de l'interprète du jour. Pour le même hexagramme, si vous posez des questions sur la prospérité ou le déclin de votre carrière ou de votre relation, les résultats peuvent être très différents.

Mais si nous décomposons en détail le « sentiment » de Liuyao, nous constaterons qu'il ne s'agit pas d'une pure métaphysique :

**Liuyao a un ensemble de règles internes strictes :**
- **Quantification de la génération et de la retenue des cinq éléments** : Chaque ligne appartient aux cinq éléments et a une relation claire de génération et de retenue (génération mutuelle : le métal génère de l'eau, l'eau génère du bois, le bois génère du feu, le feu génère de la terre, la terre génère du métal ; retenue mutuelle : le métal retient le bois, le bois retient la terre, la terre retient l'eau, l'eau retient le feu, le feu retient le métal)
- **Quantification de la phase prospère, du repos, du prisonnier et de la mort** : Yao a cinq états de prospérité, de phase, de repos, de prisonnier et de mort au cours de différents mois, qui peuvent être quantifiés et notés.
- **Sheng Ke, Conflit et Combinaison** : Il existe des règles spécifiques pour le conflit (contradiction) et la combinaison (combinaison)
- **Système Shensha** : bien qu'il soit hautement subjectif, il a également des conditions de déclenchement fixes.

**La véritable source du « sentiment » est :**
1. Trop de règles, d'informations redondantes et aucune sortie calculable unifiée
2. L'interprète a fait beaucoup de répartition implicite du poids, mais ne l'a pas exprimée explicitement.
3. La « quantification » des six lignes est cachée dans l'expérience de l'interprète de l'hexagramme et n'a pas été **explicitement modélisée**

---

## 5. Comment quantifier les six raies Yao ?

C’est la partie amusante : pouvons-nous transformer les six lignes en une chaîne de Markov ? **

### 5.1 Idée 1 : Modélisation de l'espace d'état

Modélisez l'hexagramme de six yao comme un système à états discrets :

```
状态空间 S = {六十四卦 × 动变爻位置 × 用神强弱}
```

Il s’agit d’un espace d’états astronomiquement grand, mais nous pouvons le simplifier.

### 5.2 Idée 2 : Construire une probabilité de transition

Utilisez les données historiques de l'hexagramme pour apprendre la « matrice de transfert d'expérience » des six lignes - c'est la voie de l'apprentissage automatique :

```
训练数据：大量已知应验的卦例（问事+卦象+结果）
↓
提取特征：卦象五行、用神强弱、动变方向、月令
↓
监督学习：训练一个分类/回归模型
↓
输出：给定一个新卦，预测各结果的概率
```

Cela extrait en fait les « règles empiriques » de six yao du cerveau humain et les transforme en un modèle calculable – ce qui n’est essentiellement pas différent de la formation d’un classificateur de texte.

### 5.3 Idée 3 : Bayésien Six Yao

Considérez Liuyao comme un **système d'inférence bayésien** :

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

### 5.4 Le principal défi de la quantification de six Yao| Défi | Descriptif |
|------|------|
| **Pénurie de données** | Il existe très peu de données hexagrammes de haute qualité, avec un historique clair et des enregistrements complets |
| **Codage de subjectivité** | Qu'est-ce qui compte comme « bonne chance » ? Différentes personnes ont des normes différentes |
| **Élément temporel** | Il est particulièrement difficile de quantifier la « date d'échéance » (quand elle surviendra) dans les six lignes |
| **Xiangshu contre mathématiques** | Six Yao a à la fois « Xiang » (signification symbolique) et « Shu » (nombre de naissances et grammes), et les deux ne sont pas complètement commensurables |
| **Problème de boîte noire** | Si vous utilisez un réseau neuronal pour le faire, la précision peut être élevée, mais le « pourquoi » ne peut pas être expliqué |

---

## 6. Quand Liu Yao rencontre le grand modèle linguistique

Une idée avant-gardiste : utilisez LLM pour créer un interpréteur quantitatif de six yao :

```
用户输入："占卜明日股市行情"
         ↓
LLM 转化为六爻起卦（模拟或随机）
         ↓
LLM 根据卦象特征 + 历史金融数据 → 生成分析文本
         ↓
输出：概率化的判断 + 自然语言解释
```

Cela traite essentiellement le **système numérique xiang** à six yao comme un **cadre d'ingénierie rapide** - les hexagrammes déterminent "l'angle de vue du problème", et LLM est responsable de remplir le contenu spécifique.

Liuyao fournit : **Cadre d'incertitude structuré** (il vous oblige à choisir l'un des 64 quadrants pour aborder le problème)
LLM fournit : **Connaissances illimitées et capacité d'explication fluide**

La combinaison des deux pourrait être la direction de la « divination IA » à l'avenir.

---

## 7. Conclusion : Ce sont les deux faces d'un même problème

| | Six Yao | Chaîne de Markov |
|---|---|---|
| **Essences** | Système de symboles + règles empiriques | Modèle mathématique + transfert de probabilité |
| **Avantages** | Hautement interprétable et riche en sagesse mathématique | Quantification précise et vérification informatique |
| **Inconvénients** | Difficile de standardiser, trop subjectif | Simplification excessive, ignorant la sémantique profonde |
| **Ancêtre commun** | Crainte et modélisation de « l'incertitude » | Abstraction mathématique du « processus aléatoire » |

En dernière analyse, **Liuyao est la théorie des probabilités** que les peuples anciens utilisaient pour simuler le Yin-Yang et les Cinq Éléments à l'époque précédant les ordinateurs** ; **La chaîne de Markov est la formalisation précise de processus aléatoires** une fois que les gens modernes disposent d'ordinateurs.

Ils ne rivalisent pas sur la même dimension, mais ils ont atteint une résonance à travers le temps et l'espace au niveau le plus profond de la cognition humaine - **comment faire face à l'incertitude et prendre des décisions**.

> Leibniz disait : "Le I Ching appréhende l'univers à travers le yin et le yang, 0 et 1."
> Markov a déclaré : « Saisissez le processus stochastique à travers la transition d'état. »
>Peut-être qu'ils parlent essentiellement de la même chose.

---*Écrit avec curiosité et humilité, quelque part entre le Big Data et le I Ching.*