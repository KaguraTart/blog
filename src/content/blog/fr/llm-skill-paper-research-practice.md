---
title: "De l'illusion au flux de travail pratique de recherche universitaire : j'ai construit un système de suivi papier à l'aide d'OpenClaw Skills"
description: "Enregistrez comment j'ai conçu deux compétences OpenClaw, recherche sur papier + vérification sur papier, pour créer un ensemble de flux de travail de recherche de documents académiques qui mettent l'accent sur « réel et vérifiable ». Principes de base : ne pas générer de fausse littérature, recherche manuelle + tri assisté par outils, et coopérer avec la direction de Zotero pour former une boucle fermée complète de la récupération à la révision."
tags: ["Griffe Ouverte", "Compétence", "recherche universitaire", "LLM", "Zotéro", "Gestion des documents", "CHIFFON", "gestion des connaissances"]
pubDate: 2026-04-15
sourceHash: "c4999eb3aaaa694755ba1afe0971a2a88935fdb8"
---

# De l'illusion au workflow de recherche académique pratique : j'ai utilisé OpenClaw Skills pour créer un système de suivi papier

## 0. Origine : Une leçon en franchissant presque la ligne

En mars 2026, j'ai demandé à AI de rédiger mon article à ma place. Le processus s'est déroulé sans heurts : le document est clairement structuré, les chiffres sont intéressants et les données semblent très raisonnables.

Puis j'ai découvert : **Ces données ont été générées par l'IA, et non à partir d'expériences réelles. **

Taux de réussite de 92%, analyse d'évolutivité de 500 drones - cela semble impeccable, mais tout cela n'est qu'une "déduction raisonnable". Si vous le soumettez ainsi, c'est une fraude académique.

Cette expérience m'a fait réfléchir complètement à une chose : **Le plus grand danger en LLM n'est pas de ne pas être capable de répondre correctement à la question, mais de répondre avec confiance mais de se tromper de réponse. Dans un cadre universitaire, cette confiance peut être fatale. **

J'ai donc repensé mon flux de travail de recherche documentaire. Il n’y a qu’un seul principe fondamental :

> **Ne générez aucun contenu invérifiable. Tous les documents doivent être recherchés manuellement et exister réellement ; toutes les données doivent être bien documentées. **

C'est l'intention originale de conception des deux compétences « recherche sur papier » et « vérificateur sur papier ».

## 1. Le cœur du problème : pourquoi la recherche documentaire directe d’IA n’est-elle pas fiable ?

Google Scholar dispose de mécanismes anti-crawler stricts. Les opérations automatisées de Selenium/Playwright sont sujettes au blocage IP, au déclenchement de codes de vérification et à des résultats instables. Plus important encore : **Les articles recherchés par l'IA peuvent ne pas exister du tout** - LLM est doué pour les réponses hallucinatoires qui « semblent raisonnables ». Dans les revues de littérature, c’est le champ de mines le plus dangereux.

Routines courantes de la fausse littérature :
- Composer le titre et l'auteur de l'article
- Fabriquer des conférences de revues inexistantes
- Exagérer le nombre de citations ou les facteurs d'impact
- Appeler les revues ordinaires les meilleures revues du domaine

J'ai donc choisi une méthode qui paraît "stupide" mais qui est absolument fiable : **Collaboration homme-machine, centrée sur les personnes**.

## 2. Conception globale : flux de travail en boucle fermée en quatre étapes

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

**Logique de base** : les gens sont responsables de « trouver des choses réelles », et LLM est responsable de « les organiser efficacement ». LLM ne génère jamais de contenu, organise et présente uniquement des informations réelles saisies par des humains.

## 3. Compétence de recherche papier : flux de travail complet de recherche documentaire

### 3.1 Initialisation rapide

```bash
# 创建研究工作区
./scripts/setup_manual_search.sh ./my_research "UAV path planning"
```

Cela générera trois fichiers dans le répertoire spécifié :

```
my_research/
├── 搜索指南.md         # 告诉你去哪搜、怎么搜
├── 文献记录模板.json   # 论文元数据存储格式
└── 收集文献.sh         # 交互式收集脚本
```

### 3.2 Guide de recherche : Guide de recherche manuelle avec stratégie« Search Guide.md » ne consiste pas seulement à « rechercher sur Google Scholar » : il contient :

**Base de données recommandée** :
- Google Scholar (complet, le plus complet)
- IEEE Xplore (autorité d'ingénierie)
- Web of Science (Index des meilleurs journaux)
- Bibliothèque numérique ACM (ordinateur)

**Stratégie de recherche** : 
```bash
# 示例：2023-2025 年 UAV 路径规划相关一区论文
"UAV path planning" AND "low altitude" AND year:2023..2025
site:ieeexplore.ieee.org "urban air mobility"
```

**Critères de filtrage** :
- Trier par nombre de citations, en donnant la priorité aux articles les plus cités
- Vérifiez si la revue se trouve dans le district 1 de l'Académie chinoise des sciences (requête sur fenqubiao.com)
- Enregistrer le DOI ou le lien accessible

**Liste de référence pour le domaine 1 de l'Académie chinoise des sciences** (intégrée à Skill) :

| Tapez | Journal représentatif |
|------|--------------|
| Transactions IEEE | TRO, SEINS, TAE, TCST |
| Principaux problèmes liés aux robots | Automatique, JFR, RAS |
| Sommet de l'aviation | ICRA, IROS, AIAA SciTech |

### 3.3 Collection de littérature interactive

Il existe deux manières, et les scripts interactifs sont recommandés :

```bash
cd my_research
./收集文献.sh
```

Le script demandera progressivement des informations sur chaque article :

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

S'il est trop lent, vous pouvez également éditer directement le `documentation record template.json` :

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

### 3.4 Générer du Markdown à partir de la revue de la littérature JSON

Une fois la collecte terminée, générez une revue structurée en un clic :

```bash
python3 scripts/paper_collection.py \
    --input 文献记录模板.json \
    --output-md 文献综述.md
```

Le format de sortie est à peu près le suivant :

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

Le rôle du LLM se reflète parfaitement ici : il ne s'agit pas de trouver des articles pour vous, mais d'organiser les informations réelles collectées manuellement dans un texte de révision structuré**.

### 3.5 Synchronisation Zotero : constitution d'une base de données documentaire personnelle

La littérature collectée peut être importée dans Zotero en un seul clic :

```bash
export ZOTERO_LIBRARY_ID="你的图书馆ID"
export ZOTERO_API_KEY="你的API Key"

python3 scripts/zotero_manager.py \
    --library-id $ZOTERO_LIBRARY_ID \
    --api-key $ZOTERO_API_KEY \
    batch-add --file 收集的文献.json
```

De cette manière, tous les documents réels collectés manuellement seront synchronisés avec Zotero pour former une bibliothèque de documents personnels réutilisables. Chaque fois que vous effectuez une nouvelle recherche, vous pouvez ajouter progressivement des bibliothèques existantes.

## 4. Compétence du vérificateur papier : validation croisée d'authenticitéAprès avoir collecté la littérature, il reste encore à la vérifier. Il s'agit de la deuxième compétence - **Outil de vérification de l'authenticité de la thèse**.

### 4.1 Pourquoi la vérification est-elle requise ?

Les recherches manuelles peuvent également générer des erreurs :
- DOI mal renseigné
- Le nom de l'auteur est mal orthographié
- Le nom de la revue est écrit sous forme d'abréviation au lieu du nom complet
- Le nombre de citations a été mal enregistré (des dizaines de fois)
- Je pensais que c'était la Zone 1 mais il s'est avéré que c'était juste du CCF-B.

Par conséquent, avant de finaliser l’examen, utilisez un vérificateur papier pour effectuer une vérification croisée.

### 4.2 DOI + Vérification des métadonnées

```bash
python3 scripts/verify_papers.py \
    --input papers.json \
    --output verification_report.md
```

Format `papers.json` :

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

Le script sera vérifié via deux API :

**API Crossref** — Obtenez les métadonnées officielles :
- Le titre correspond-il exactement ?
- La liste des auteurs est-elle correcte ?
- Les années sont-elles cohérentes ?
- Quel est le nom complet de la revue ?

**API Semantic Scholar** — Vérification secondaire :
- Si le document existe réellement
- Quel est le nombre réel de citations ?
- Obtenir le résumé de l'article pour vérifier le contenu

### 4.3 Format du rapport de vérification

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

De cette manière, toute erreur de saisie manuelle peut être détectée rapidement.

## 5. Structure de fichiers de deux compétences

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

## 6. Lien avec la base de connaissances RAG

Les documents réels collectés peuvent non seulement générer une révision, mais également entrer dans la **base de connaissances LLMRAG** pour des questions et réponses ultérieures et une aide à la rédaction.

L'ensemble du processus de liaison :

```
手动搜索 → 收集到 JSON → 生成 Markdown 综述
    ↓
导入 Zotero（长期管理）
    ↓
添加到 LLMRA G知识库（向量检索）
    ↓
后续论文写作时，RAG 问答检索真实文献
```

De cette manière, la recherche documentaire n’est pas une tâche ponctuelle, mais un atout de connaissances qui peut être accumulé, récupéré et réutilisé.

## 7. Résumé de la philosophie de conception

La conception de ce workflow répond à une question fondamentale : **Où sont les limites du LLM dans la recherche universitaire ? **

| LLM est bon en | Le LLM n'est pas bon pour (les gens doivent venir) |
|---------|----------------------|
| Rassembler et organiser les informations | Déterminer si le document existe réellement |
| Générer du texte structuré | Vérifier l'exactitude du DOI/du nombre de citations |
| Trouver les lacunes et les connexions dans les connaissances | Rechercher des bases de données faisant autorité |
| Polissage et réécriture | Décider quels articles méritent d'être inclus dans la revue |**En résumé** : tous les liens qui nécessitent une « garantie d'authenticité » sont gérés par des humains ; LLM est uniquement responsable du tri et de la génération des humains sur la base d'entrées réelles.

Ceci est également conforme à la leçon tirée de mon précédent incident papier : **Ne laissez pas LLM produire quoi que ce soit qui ne puisse pas être retracé jusqu'à la source**. Chaque donnée et chaque document doivent être traçables.

## 8. Orientations futures d'expansion

- [ ] Accédez à SerpAPI pour implémenter la recherche semi-automatique de Google Scholar (réduire les opérations manuelles)
-[ ] Synchronisation bidirectionnelle avec l'API Zotero pour extraire automatiquement les métadonnées des articles lus
- [ ] Créez une base de connaissances de revues/conférences spécifique à un domaine avec des modèles de format de citation intégrés
-[ ] Convertissez automatiquement les documents vérifiés au format de note Obsidian et intégrez-les à l'écosystème de notes à double chaîne

---

*Parcours de compétences : `~/.openclaw/workspace/skills/paper-research/` et `paper-verifier/`*

*Auteur : Tarte Kagura | 2026-04-15 | Écrit pour mon futur moi et tous les chercheurs qui ont besoin de faire une revue de la littérature*