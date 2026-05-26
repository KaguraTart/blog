---
title: "Planification de mission UAV guidée par LLM : la frontière de l'inférence à l'exécution"
description: "Analyse approfondie des trois principaux paradigmes de LLM pour la planification de missions de drones : LLM en tant que planificateur, LLM+PDDL symbol planning et LLM+RAG, couvrant des travaux de pointe tels que VoxPoser, ActiveGAMER et l'architecture à double processus."
tags: ["drone", "LLM", "planification des missions", "PDDL", "intelligence incarnée", "de bout en bout"]
category: "Tech"
pubDate: 2026-04-27
sourceHash: "da6a65e942e29f4e1a718bd909a7275cfe43fa9b"
---

# Planification de mission de drone guidée par LLM : la frontière de l'inférence à l'exécution

> **Série intelligente UAV · Chapitre X+1**
> Pleins feux : LLM comme planificateur de mission, intégration de planification symbolique, architecture d'inférence en temps réel

---

## 1. Pourquoi le LLM est-il adapté à la planification de missions de drones ?

Le défi de la planification des missions des drones réside dans l'**incertitude du monde ouvert** :

```
传统规划（基于模型）：
输入：精确目标状态 + 精确环境模型
输出：最优动作序列
局限：模型不准就崩溃，无法处理语言目标

LLM 规划（基于知识）：
输入：自然语言指令 + 视觉观测 + 世界知识
输出：可执行动作序列
优势：泛化性强、零样本理解新任务
```

Avantages du LLM :
- **Connaissance du monde** : la pré-formation contient de riches connaissances physiques ("L'eau coule", "Les voitures sont plus rapides que les gens")
- **Inférence Zero-shot** : pas besoin de s'entraîner séparément pour chaque tâche
- **Planification en plusieurs étapes** : Décomposer des tâches complexes en chaînes de sous-objectifs (Chaîne de pensée)

---

## 2. Le paradigme du LLM pour la planification des tâches

### 2.1 Paradigme 1 : LLM en tant que planificateur (actions de sortie directe)

**Travail représentatif :**

**ReAct (Raisonnement + Agir)**
- Idée centrale : le LLM alterne « raisonnement » et « action »
- Chaque étape : `obs → réfléchir → action → next_obs`
- Applicable à : Scénarios avec un statut observable et un retour environnemental clair
- Adaptation sur drone : nécessite une action rapide → boucle obs

**SayCan (PaLM-SayCan, 2022)**
- Combiner la « description de capacité » de LLM avec la « faisabilité » physique
- Le robot dit "ce qu'il peut faire", et le LLM décide "ce qu'il doit faire"
- **Enlightenment :** Le drone peut filtrer les actions irréalisables en fonction de son propre statut (puissance, restrictions de vol)

---

### 2.2 Paradigme 2 : Planification des symboles LLM + PDDL

**PDDL (Planning Domain Definition Language)** est un langage classique de planification de tâches robotiques qui modélise les tâches sous forme de problèmes symboliques discrets.

**Idée de base :**
```
VLM 感知 → PDDL problem 生成 → 经典规划器 → UAV 动作序列
```

**Avantages :**
- Les résultats de la planification peuvent être expliqués et vérifiés
- Preuve mathématique pour garantir l'achèvement de la tâche
- Adapté aux scénarios critiques pour la sécurité (vols dans l'espace aérien urbain)

**Défi :**
- La modélisation PDDL elle-même est un goulot d'étranglement (nécessite des experts du domaine)
- La dynamique continue des drones n'est pas totalement compatible avec les hypothèses discrètes du PDDL
- **Idée de solution :** PDDL gère la décomposition des tâches de haut niveau, MPC gère l'exécution de trajectoires de bas niveau

---### 2.3 Paradigme 3 : LLM + RAG (génération améliorée par récupération)

** MPC génératif (arXiv, 2026) **

**Article :** *GenerativeMPC : MPC du corps entier guidé par VLM-RAG avec impédance virtuelle pour la manipulation mobile bimanuelle*
**Auteur :** Marcelino Julio Fernando et al.
**Source :** arXiv, avril 2026

**Idée de base :**
```
VLM 感知当前场景 → 检索相关操作知识库 → RAG 生成操作建议 → MPC 执行
```

**Technologie clé :**
1. **Récupération de connaissances** : récupérez les exemples les plus pertinents pour le scénario actuel à partir de la base de connaissances opérationnelles (y compris les données d'expérience de contrôle du robot)
2. **Impédance virtuelle** : générez des paramètres de contrôle de conformité pour éviter les collisions rigides
3. **Filtrage RAG** : assurez-vous que la sortie LLM est physiquement exécutable

**Adaptation sur drone :**
- Rechercher les codes du bâtiment (restrictions de hauteur, zones d'exclusion aérienne)
- Récupérer l'expérience historique de la mission (paramètres de vol dans des conditions météorologiques similaires)
- Récupérer les protocoles de sécurité (distance minimale d'évitement d'obstacles, procédures d'urgence)

---

## 3. Architecture de raisonnement en temps réel

### 3.1 Architecture à double processus (arXiv, 2026)

**Article :** *Une architecture à double processus pour la navigation intérieure basée sur VLM en temps réel*
**Auteur :** Joonhee Lee, Hyunseung Shin, Jeonggil Ko
**Source :** arXiv :2601.19401, janvier 2026

**Conception de base :**

```
┌─────────────────────────────────────────────┐
│           System Architecture               │
│                                             │
│  Process 1 (Slow): VLM Reasoning Thread     │
│  ┌─────────────────────────────────────┐   │
│  │ VLM: "What should I do next?"       │   │
│  │ Frequency: ~0.2-1 Hz                 │   │
│  │ Output: Navigation goal / decision  │   │
│  └─────────────────────────────────────┘   │
│              ↓ goal                        │
│  Process 2 (Fast): Control Execution Thread│
│  ┌─────────────────────────────────────┐   │
│  │ MPC: Track trajectory to goal        │   │
│  │ Frequency: ~100 Hz                   │   │
│  │ Output: Motor control signals        │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

**Principes de conception :**
- **Quick Process** (MPC) : réponse au niveau de la milliseconde, traitement de l'évitement d'obstacles en temps réel
- **Slow Process** (VLM) : Raisonnement de deuxième niveau, traitement des décisions de haut niveau
- **Découplage critique** : VLM n'est pas sur le chemin critique et n'affecte pas la fréquence de contrôle

---

### 3.2 Cadre de planification hiérarchique

**Niveau élevé (LLM/VLM, deuxième niveau) : **
```
任务理解 → 子目标分解 → 全局路径规划 → 授权低层执行
```

**Couche intermédiaire (optimisation différenciable, niveau 100 ms) : **
```
RRT*/MPC → 局部路径重规划 → 平滑轨迹生成
```**Couche basse (PID/MPC, niveau milliseconde) : **
```
姿态控制 → 电机分配 → 执行
```

---

## 4. Profondeur de l'algorithme clé

### 4.1 VoxPoser : carte de valeurs 3D synthétique LLM

**Article :** *VoxPoser : Cartes de valeurs 3D composables pour la manipulation robotique avec des modèles de langage*
**Auteur :** Wenlong Huang, Chen Wang, Ruohan Zhang, Yunzhu Li, Jiajun Wu, Li Fei-Fei
**Source :** arXiv :2307.05973, juillet 2023

**Contribution de base :**
- Sortie LLM **Carte thermique spatiale 3D** (carte de valeurs 3D composable)
- Encodage par carte thermique "où aller" et "ce qu'il faut éviter"
- Directement utilisé comme fonction de récompense pour l'optimisation de trajectoire

**Extension sur le drone :**
- Carte thermique d'occupation 3D de sortie VLM
- Fonction de coût MPC basée sur la carte thermique
- VoxPoser pour UAV = "accessibilité spatiale 3D du langage"

**Remarque :** VoxPoser a été publié sur arXiv. Aucun rapport clair sur les publications de la conférence n'a été trouvé jusqu'à présent.

---

### 4.2 CoNVO (optimisation conditionnelle de la valeur neuronale)

Combinez la planification LLM avec l'itération de valeur :
- LLM fournit des **préférences prioritaires** (quelles actions sont les plus raisonnables)
- L'itération de valeur fournit une **garantie d'optimalité**
- Plus robuste que la planification LLM pure et plus flexible que la planification pure

---

## 5. Planification assistée par un modèle mondial

### 5.1 Pourquoi un modèle mondial ?

Les connaissances du LLM sont statiques, mais l'environnement drone est dynamique :
- Le vent va changer
- Les obstacles se déplaceront
- Les signaux GNSS peuvent dériver

Le modèle mondial permet aux drones de **prédire l'avenir** : 
```
当前状态 + 动作 → 世界模型 → 预测未来状态序列
LLM 在预测的未来状态序列上做规划（Plan over imagined futures）
```

### 5.2 Représentant papier**Série Dreamer** (Daniel Hafner, Jürg Widmer, etc.)
- Basé sur le modèle dynamique RSSM
- Faire un apprentissage par renforcement sur un futur imaginé
- Vérifié sur les robots (bras de robots, véhicules sans pilote)

**VMP (planification de mouvements vidéo)**
- Utiliser des modèles de génération vidéo pour la planification de mouvements
- Générer des images futures → extraire des vecteurs de mouvement → contrôler le drone

---

## 6. Sécurité et authentification

### 6.1 Pourquoi la sécurité est essentielle

Lorsque les drones volent dans les villes, une mauvaise prise de décision peut causer des **victimes humaines**. Il existe une contradiction fondamentale entre les résultats probabilistes du LLM et les garanties déterministes requises par la sécurité aérienne.

### 6.2 Cadre de sécurité

**CBF（Fonctions de barrière de contrôle）：**
- ASMA présente CBF au drone VLN
- S'assurer que l'état dangereux n'est jamais accessible

**Vérification formelle：**
- Utilisez TLA+ / NuSMV pour la vérification de la machine d'état
- Les résultats de la planification LLM sont exécutés après vérification du modèle

**Blindage :**
- Protecteur de couche inférieure (Shield) : surveille la sortie LLM et intercepte les actions dangereuses
- LLM de niveau supérieur : concentrez-vous sur l'achèvement des tâches et ne tenez pas compte des détails de sécurité
- **Architecture "Ange Gardien" de conduite autonome**

---

## 7. Points chauds frontaliers et orientations futures

### 7.1 VLA de bout en bout (Vision-Langage-Action)

**Dernière tendance :** Évitez la conception hiérarchique « détection → planification → contrôle » et générez un **jeton d'action** directement à partir de VLM.

Travail représentatif :
- **RT-2** (Google Robotics) : Affinez directement l'action de sortie de VLM
- **π₀** (Intelligence Physique) : VLA pour robots humanoïdes
- **Version UAV** (émergente) : idées similaires appliquées aux drones

**Défi :**
- Continuité de l'espace d'action vs discrétion du langage
- Difficulté de vérification de sécurité (boîte noire de bout en bout)
- Pénurie de données (nécessite des données de téléopération robot à grande échelle)

### 7.2 Planification LLM collaborative multi-machines

**SysNav (arXiv, mars 2026)****Article :** *SysNav : La coopération systématique à plusieurs niveaux permet la navigation d'objets dans le monde réel et entre modes de réalisation*
**Auteur :** Haokun Zhu et al.
**Source :** arXiv :2603.xxxxx, mars 2026

**Contribution de base :**
- Navigation collaborative multi-agents sur différentes plateformes robotiques
- LLM fait une coordination de haut niveau (qui va dans quelle zone)
- Fusion de perception distribuée (chaque agent partage la vision)

### 7.3 Intelligence physique × UAV

- **Modèles de base pour la manipulation** → **Modèles de base pour le vol**
- Un modèle de pré-entraînement dédié au "cerveau UAV" pourrait apparaître dans le futur
- Similaire à LLaVA mais spécialisé dans le raisonnement spatial 3D + dynamique de vol

---

## 8. Résumé et suggestions

| Dimensions | Meilleur actuel | Orientations futures |
|------|---------|---------|
| Paradigme de planification | Architecture à double processus (réalisable en temps réel) | VLA de bout en bout (objectif à long terme) |
| Connaissance du monde | RAG (fiable mais lent) | Modèle mondial (rapide mais nécessite une formation) |
| Sécurité | CBF + Blindage | Vérification formelle (entièrement garantie) |
| Déploiement périphérique | LLaVA 4 bits (à peine en temps réel) | Puces à usage spécial (NPU/TPU) |

**Conseils pour vous :**
1. **Le chemin le plus rapide vers les résultats** : architecture à double processus + LLaVA-7B + plateforme UAV
2. **La plus grande marge d'innovation** : VLM + cadre de vérification de sécurité (presque personne ne le fait actuellement)
3. **Disposition à long terme** : collectez vos propres données de contrôle de drone et entraînez un modèle VLA dédié

---

## 📚 Références1. Lee et coll. *Une architecture à double processus pour la navigation intérieure basée sur VLM en temps réel*. arXiv : 2601.19401, 2026.
2. Fernando et coll. *GenerativeMPC : MPC du corps entier guidé par VLM-RAG avec impédance virtuelle*. arXiv, 2026.
3. Huang et coll. *VoxPoser : cartes de valeurs 3D composables pour la manipulation robotique avec des modèles de langage*. arXiv :2307.05973, 2023.
4. Brohan et coll. *RT-2 : Les modèles Vision-Langage-Action transfèrent les connaissances du Web vers le contrôle robotique*. arXiv, 2023.
5. Zhu et coll. *SysNav : la coopération systématique à plusieurs niveaux permet une navigation d'objets dans le monde réel et entre modes de réalisation*. arXiv, 2026.
6. Ahn et coll. *Faites ce que je peux et pas ce que je dis : ancrer le langage dans les moyens robotiques*. arXiv, 2022.