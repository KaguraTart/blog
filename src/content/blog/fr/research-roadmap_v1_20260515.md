---
title: "Feuille de route culturelle du blog de recherche sur les drones à basse altitude : un plan complet du blog au journal"
description: "Triez systématiquement la valeur de recherche de 18 articles sur les drones à basse altitude dans le blog, identifiez les cinq directions ayant le plus grand potentiel de publication et fournissez leurs déclarations de points d'innovation respectives, les revues cibles, les listes d'expériences supplémentaires et les délais suggérés."
pubDate: 2026-05-15
tags: ["Planification de thèse", "Feuille de route de la recherche", "drone", "basse altitude", "Stratégie de soumission", "T-ITS", "ICRA"]
category: Tech
---

# Blog de recherche sur les drones à basse altitude Feuille de route culturelle : un plan complet du blog au journal

> Cet article n'est pas une introduction technique, mais un **document de gestion de la recherche** : réexaminez le contenu des blogs accumulés dans le passé et découvrez lesquels valent la peine d'être publiés dans des revues, lesquels manquent encore et lesquels doivent être testés à partir de zéro. C’est aussi une prise en compte de son propre contexte de recherche.

---

## 0. Contexte et point de départ

À l'heure actuelle, le blog a accumulé **27 articles**, dont 18 articles principaux liés aux drones à basse altitude, couvrant la planification de trajectoire, la résolution de conflits, la planification multi-machines, la reconstruction de la perception, les jumeaux numériques, la planification LLM/VLM et d'autres directions.

Base papier publiée : Journal of Advanced Transportation (SCI Q3), Q-learning for Highway Ramp Control (DOI : 10.1155/2023/4771946), qui a établi le ton de recherche « Reinforcement Learning × Traffic System ».

**Objectif de cet article :**

1. Identifiez les 5 à 6 directions du contenu du blog qui sont les plus utiles pour la publication
2. Fournir des informations exploitables pour chaque direction : déclaration des points d'innovation, différences par rapport aux travaux existants, revues/conférences cibles, liste des expériences supplémentaires et chronologie des suggestions.
3. Fournir une feuille de route globale de soumission sur 12 mois
4. Faire de ce document un outil vivant de gestion de la recherche (le numéro de version est reflété dans le nom du fichier)

---

## 1. Carte panoramique du contenu du blog

### 1.1 Trois grands axes de recherche

```
主线一：路径规划 × 冲突消解 × 多机调度
├── uav-urban-route-planning        （路径规划算法综述）
├── uav-conflict-resolution         （CD&R 机制综述+架构）
├── uav-conflict-env-construction   （仿真环境工程）
├── marl-kat-uav-conflict ★         （KAT MARL 框架）
├── large-scale-uav-scheduling ★    （三层百机调度）
└── urban-uav-3d-spatial-modeling   （3D空域建模参考）

主线二：感知 × 环境重建 × 数字孪生
├── uav-digital-twin-semantic-mapping ★  （五层数字孪生）
├── uav-semantic-mapping-functional-zoning ★（多源语义融合）
├── uav-nerf-gs-planning                 （NeRF/3DGS规划集成）
├── next-best-view-nerf-3dgs ★           （信息论NBV）
├── information-theory-active-perception （理论基础）
└── uav-multimodal-sim-data-synthesis    （多模态仿真工程）

主线三：LLM/VLM × 语义规划 × 形式验证
├── llm-uav-semantic-planning ★          （LTL/STL形式验证）
├── llm-guided-uav-planning-frontiers    （规划前沿概念）
├── hierarchical-vlm-uav-planning        （分层VLM架构）
└── vlm-uav-navigation-foundations       （VLN综述）

延伸：地面交通
├── carla-sumo-rl-lane-change ★          （PPO变道，已有实验）
└── traffic-signal-control               （信号控制反思）
```

★ = Candidats papier analysés dans cet article

### 1.2 Liste récapitulative de l'évaluation de la maturité| Article | Cadre théorique | Soutien expérimental | Maturité globale | Faisabilité de la thèse |
|------|----------|----------|--------------|---------------|
| marl-kat-uav-conflit | ★★★★★ | ★★☆☆☆ | ★★★★☆ | Élevé (inventez simplement l'expérience) |
| planification de drones à grande échelle | ★★★★☆ | ★★☆☆☆ | ★★★☆☆ | Élevé (expériences à échelle complémentaire) |
| prochaine-meilleure-vue-nerf-3dgs | ★★★★★ | ★★★☆☆ | ★★★★☆ | Élevé (complément à l'expérience en ligne) |
| uav-cartographie-sémantique-zonage-fonctionnel | ★★★★☆ | ★★☆☆☆ | ★★★☆☆ | Moyen (Données SIG supplémentaires) |
| llm-uav-planification-sémantique | ★★★★☆ | ★☆☆☆☆ | ★★★☆☆ | Moyen (ensemble de données d'évaluation supplémentaire) |
| carla-sumo-rl-lane-changement | ★★★☆☆ | ★★★★☆ | ★★★★☆ | Élevé (expérimenté) |

---

## 2. Niveau 1 : Potentiel de publication le plus élevé (il est recommandé de le soumettre dans un délai de 6 à 12 mois)

### Document A : Résolution des conflits urbains à grande échelle liés aux drones – Cadre KAT-MARL

**Article source :** `marl-kat-uav-conflict` + `uav-conflict-resolution` + `uav-conflict-env-construction`

**Journal cible :** Transactions IEEE sur les systèmes de transport intelligents (T-ITS, SCI Q1, IF ≈ 8.5)

#### Point d'innovation principal (allégation de nouveauté)

Le framework **KAT (Knowledge-Attention-Transfer)** est proposé pour remplacer la transmission de messages explicites par un réseau d'attention graphique (GAT) afin d'obtenir une coordination multi-machine implicite sans contraintes de communication :- **Mécanisme de communication implicite :** Chaque drone observe uniquement l'état du quartier et extrait automatiquement les informations les plus pertinentes sur les voisins grâce au poids d'attention du GAT, sans diffuser de messages.
- **Paradigme de formation CTDE :** Formation centralisée (le critique accède à l'état global) + exécution décentralisée (l'acteur utilise uniquement les observations locales)
- **ORCA couvre la couche inférieure :** La garantie de sécurité à deux niveaux de la stratégie d'apprentissage et de la méthode d'analyse géométrique (ORCA) garantit une stricte absence de collision

Système de formule de base :

Poids d’attention GAT :
$$e_{ij} = \text{LeakyReLU}\!\left(\mathbf{a}^\top[\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_j]\right)$$

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k\in\mathcal{N}_i}\exp(e_{ik})}$$

Informations agrégées sur les voisins :
$$\mathbf{h}_i' = \sigma\!\left(\sum_{j\in\mathcal{N}_i}\alpha_{ij}\mathbf{W}\mathbf{h}_j\right)$$

Fonctions de valeur centralisées QMIX :
$$Q_{tot}(\boldsymbol{\tau}, \mathbf{a}) = f_\theta\!\left(Q_1(\tau_1, a_1),\ldots,Q_N(\tau_N, a_N),\mathbf{s}\right)$$

Le poids de $f_\theta$ est non négatif (contrainte de monotonie), assurant la condition IGM (maximisation individuelle-globale).

#### Différenciation par rapport aux travaux existants

| Méthode | Exigences de communication | Échelle | En temps réel | Garantie de sécurité |
|------|---------|------|--------|---------|
| MADDPG | Aucun | <20 | Mauvais | Aucun |
| QMIX | Aucun | <20 | Moyen | Aucun |
| CommNet | Diffusion complète | <50 | Pauvre | Aucun |
| ORQUE | Aucun | Grand | Excellent | Oui |
| **KAT (cet article)** | **Aucun** | **50+** | **Bien** | **Oui (double couche)** |

#### Liste d'expériences supplémentaires- [ ] **Ablation à grande échelle :** 20/50/100 drones sont entraînés + testés séparément, et le taux de réussite, le délai moyen et le délai de calcul sont enregistrés
- [ ] **Comparaison de référence :** ORCA uniquement, MADDPG, QMIX (sans GAT), QMIX+GAT (avec GAT et sans couverture ORCA)
- [ ] **Scénario :** Construire une carte de simulation basée sur le réseau routier réel de Shanghai Lujiazui ou du CBD de Pékin
- [ ] **Indicateurs :** Taux de réussite des missions (taux d'atteinte des objectifs), délai supplémentaire moyen (secondes), taux de conflits (conflits/UAV/minute), délai d'inférence (ms)
- [ ] **Visualisation :** Carte thermique du poids de l'attention, montrant le modèle du drone prêtant attention aux voisins

#### Chronologie

```
2026/06  搭建仿真环境（基于 existing uav-conflict-env-construction）
2026/07  训练 KAT 模型 + 基线对比实验
2026/08  写稿（Introduction / Method / Experiment / Conclusion）
2026/09  内部审阅 + 语言润色
2026/09  投稿 IEEE T-ITS（Regular Paper，通常 3–6 个月审回）
```

---

### Paper B : Système de planification hiérarchique à trois niveaux pour des centaines de drones

**Article source :** `large-scale-uav-scheduling` + `uav-urban-route-planning`

**Journal cible :** IEEE T-ITS ou Transportation Research Part C (SCI Q1, IF ≈ 7.6)

#### Points d'innovation fondamentaux

Une **architecture hiérarchique à trois couches** est proposée pour décomposer le problème de planification urbaine de plus de 100 drones en trois sous-problèmes qui peuvent être optimisés indépendamment et exploités en collaboration :

**Couche macroscopique (allocation des tâches) :** État de la carte de l'espace aérien codé GNN + ACO (optimisation des colonies de fourmis) attribue des tâches aux drones pour optimiser le débit global

Fonction objectif au niveau macro :
$$\min\;\sum_{k=1}^{N}\!\left(w_1 T_k + w_2 \mathcal{E}_k\right) + w_3\cdot\text{Congestion}(G)$$

**Couche méso (coordination des conflits) :** Coordination multi-agents QMIX, ajustement vitesse/hauteur basé sur le chemin macro pour résoudre les conflits

Prise de décision décentralisée au niveau méso, stratégie locale pour chaque drone :
$$\pi_k(a_k \mid \tau_k) = \text{softmax}(Q_k(\tau_k, \cdot;\theta_k))$$

**Micro couche (exécution de trajectoire) :** Analyse géométrique ORCA + optimisation du roulement MPC pour obtenir un suivi précis au centimètre prèsOptimisation continue MPC (taille du pas de prédiction $H$) :
$$\min_{\mathbf{u}_{0:H-1}}\sum_{t=0}^{H-1}\!\left\|\mathbf{x}_t - \mathbf{x}_{ref}\right\|_Q^2 + \|\mathbf{u}_t\|_R^2$$

#### Liste d'expériences supplémentaires

- [ ] **Courbe d'expansion d'échelle :** 20/50/100/200 UAV, débit du système d'enregistrement (UAV/min), latence de bout en bout, ressources informatiques (CPU/GPU)
- [ ] **Comparaison de référence :** FCFS (premier arrivé, premier servi), MILP centralisé (optimal mais lent), architecture à deux niveaux (pas de couche macro)
- [ ] **Diversité des scénarios :** Scénario logistique à haute densité (demande uniforme) vs scénario de pointe soudaine (arrivée de Poisson)
- [ ] **Analyse théorique :** Donne le calcul théorique de la limite supérieure du débit du système (basée sur la théorie des files d'attente)

#### Chronologie

```
2026/07  实现三层框架代码 + 集成测试
2026/08  规模扩展实验（需要较长训练时间）
2026/10  写稿
2026/11  投稿 Transportation Research Part C
```

---

### Article C : Planification de la détection active 3DGS basée sur la théorie de l'information

**Article source :** `next-best-view-nerf-3dgs-exploration` + `information-theory-active-perception-foundations` + `uav-nerf-gs-planning`

**Conférence cible :** ICRA 2026 (se termine env. 2026/09) ou IROS 2026

#### Points d'innovation fondamentaux

Utilisez **Fisher Information Matrix (FIM)** comme cible proxy sélectionnée par Next-Best-View pour piloter la reconstruction de convergence active **3D Gaussian Splatting (3DGS)** :

**Quantification du gain d'informations :** Le point de vue suivant $\mathbf{v}^*$ est choisi pour maximiser le gain d'informations attendu par rapport au paramètre de scène $\boldsymbol{\Theta}$ :

$$\mathbf{v}^* = \arg\max_{\mathbf{v}\in\mathcal{V}_{free}} \mathcal{I}(\boldsymbol{\Theta}; \mathbf{y}_\mathbf{v})$$En utilisant la borne inférieure de Cramér-Rao, la matrice FIM inverse donne une borne inférieure sur l’incertitude de l’estimation des paramètres :

$$\text{Cov}(\hat{\boldsymbol{\Theta}}) \succeq \mathbf{F}(\boldsymbol{\Theta})^{-1}$$

** approximation différentiable du FIM 3DGS : ** Pour chaque $\mathcal{G}_i$ gaussien, son FIM par rapport à la moyenne $\boldsymbol{\mu}_i$ peut être approché comme :

$$\mathbf{F}_i(\boldsymbol{\mu}_i) \approx \sum_{\mathbf{r}\in\mathcal{R}(\mathbf{v})} \frac{\partial \hat{C}(\mathbf{r})}{\partial \boldsymbol{\mu}_i}\!\left(\frac{\partial \hat{C}(\mathbf{r})}{\partial \boldsymbol{\mu}_i}\right)^\top \frac{1}{\sigma_n^2}$$

**Stratégie gourmande en temps réel :** La recherche globale optimale de NBV est NP-difficile, utilisant la sérialisation gourmande + l'élagage (contrainte de distance + détection d'occlusion) pour obtenir une prise de décision en temps réel (<50 ms/pas).

#### Comparaison avec les méthodes existantes

| Méthode | Fonction objectif | Expressions | En temps réel | Assurance de l'information |
|------|---------|------|--------|---------|
| Frontière | Couverture | Voxels | Bon | Aucun |
| Minimisation de l'entropie | Entropie occupée | Voxels | Moyen | Faible |
| ActifGAMER | Qualité de la reconstruction | 3DGS | Pauvre | Aucun |
| **Cet article (FIM-3DGS)** | **Informations sur les pêcheurs** | **3DGS** | **Bien** | **Garantie théorique CRB** |

#### Liste d'expériences supplémentaires-[ ] **Expérience de reconstruction en ligne :** Scène urbaine AirSim, vol autonome de drone + mise à jour 3DGS en ligne
- [ ] **Mesures :** PSNR/SSIM (qualité de la reconstruction), couverture (%), gain moyen d'informations par pas, distance de vol totale
-[ ] **Référence :** Exploration aléatoire, basée sur Frontier, ActiveGAMER, SO-NeRF
- [ ] **Ablation :** Cible proxy FIM vs cible de couverture pure vs cible de qualité de reconstruction pure

#### Chronologie

```
2026/06  实现 FIM-3DGS 可微近似模块
2026/07  AirSim 在线实验
2026/08  写稿（ICRA 格式，8页）
2026/09  投稿 ICRA 2026
```

---

### Article D : Fusion sémantique multi-sources + planification de trajectoire de drone basée sur des partitions fonctionnelles

**Article source :** `uav-semantic-mapping-fonctionnel-zoning` + `uav-digital-twin-semantic-mapping`

**Journal cible :** IEEE T-ITS ou Transportation Research Part C

#### Points d'innovation fondamentaux

**Pipeline de fusion de données multi-sources :**

$$\mathcal{M}_{sémantique} = \mathcal{F}_{fusion}(\mathcal{I}_{RS},\; \mathcal{G}_{OSM},\; \mathcal{P}_{POI},\; \mathcal{D}_{census})$$

Parmi eux, $\mathcal{I}_{RS}$ est le résultat de la segmentation sémantique des images de télédétection, $\mathcal{G}_{OSM}$ est le vecteur SIG route/bâtiment, $\mathcal{P}_{POI}$ est le point d'intérêt (entreprise/hôpital/école) et $\mathcal{D}_{census}$ est les données démographiques.

**Modèle de risque de zonage fonctionnel urbain :**

Définissez le coefficient de risque de base $\lambda_z$ pour chaque type de zone fonctionnelle $z \in \{\text{residential}, \text{commercial}, \text{industrial}, \text{green space}, \text{water}\}$, en combinant le facteur de période $\delta(t)$ (pointe du matin et du soir par rapport à la nuit) et la densité du sol $\rho_{bld}$ :$$\mathcal{R}(x, y, t) = \lambda_{z(x,y)} \cdot \delta(t) \cdot \rho_{bld}(x,y)$$

**Fonction de coût d'itinéraire tenant compte des risques :**

Intégration du graphique de risque de partition fonctionnelle dans les pondérations de bord A* :

$$d(u,v) = \ell_{uv}\cdot\!\left(1 + \beta_1\mathcal{R}_{uv} + \beta_2 TI_{uv}\right)$$

Où $TI_{uv}$ est l'intensité de turbulence du couloir (extraite du modèle de champ de vent), $\beta_1, \beta_2$ sont des coefficients de compromis.

**Différenciation par rapport aux travaux existants :**
- Travaux existants utilisant la **densité de population** comme indicateur du risque au sol → statique, à gros grains
- Cet article utilise **Type de zonage fonctionnel × facteur de période × densité des bâtiments** modèle de risque tridimensionnel → dynamique, à granularité fine et pouvant être migré d'une ville à l'autre (normes de zonage fonctionnel unifiées)

#### Liste d'expériences supplémentaires

- [ ] **Acquisition de données :** Données SIG CBD Guangzhou/Shenzhen (OSM open source + images de télédétection haute résolution)
- [ ] **Comparaison de référence :** Chemin le plus court pur (Dijkstra), pondération de la densité de population, pondération de l'occlusion du bâtiment
- [ ] **Indicateurs :** Points d'exposition au risque (REI = $\int \mathcal{R}(\boldsymbol{\xi}(t))\,\mathrm{d}t$), longueur du trajet, temps de vol
- [ ] **Courbe de Pareto :** Front de compromis entre REI et longueur de trajet
- [ ] **Expérience de généralisation :** Paramètres de poids d'entraînement à Pékin/Shanghai, tests à Guangzhou (transférabilité entre les villes)

#### Chronologie

```
2026/07  GIS 数据采集与预处理
2026/08  功能分区模型实现 + 航路规划实验
2026/09  写稿
2026/11  投稿 Transportation Research Part C
```

---

## 3. Niveau 2 : nécessite davantage de travail supplémentaire (12 à 18 mois)

### Paper E : Planification de mission UAV avec LLM + vérification formelle

**Article source :** `llm-uav-semantic-planning` + `llm-guided-uav-planning-frontiers`

**Cible :** ICRA/IROS ou IJCAI 2027

#### Points d'innovation fondamentaux

** Pipeline en boucle fermée :**

```
自然语言任务描述
       ↓ LLM 转译
LTL/STL 形式规范
       ↓ 模型检测（NuSMV / Breach）
验证通过 → 执行
验证失败 → 反馈给 LLM → 迭代修正
```**Exemple de spécification LTL ("Évitez de survoler l'hôpital avant d'atteindre le point B") :**

$$\varphi = \Box(\neg \text{Hôpital}) \;\wedge\; \Diamond(\text{Waypoint}_B)$$

**Principaux défis :**
- Précision de la traduction de LLM → LTL (nécessite la construction d'un ensemble de données d'évaluation : paire de spécifications langage naturel-forme)
- Charge de calcul liée à la vérification de modèles dans de grands espaces d'états (nécessite une technologie d'abstraction de l'espace d'états)
- LLM Hallucination conduit à des spécifications insatisfaisantes (nécessite un pré-traitement des contrôles de satisfiabilité)

#### Liste de travail supplémentaire

- [ ] Construire l'ensemble de données de la mission UAV NL → LTL (~ 500 paires)
-[ ] Mesurer la précision de la traduction de GPT-4o / Llama-3
-[ ] Implémenter l'interface NuSMV pour vérifier les spécifications des scènes de drones urbains
- [ ] Conception Détection d'hallucinations + module de réparation

---

### Paper F : RL de changement de voie multi-agent CARLA-SUMO (extension au sol)

**Article source :** `carla-sumo-rl-lane-change` (270 000 étapes de résultats expérimentaux PPO)

**Cible :** Recherche sur les transports, partie C

#### Sens d'extension

- Statut actuel : PPO à agent unique, convergé à 270 000 étapes
- Extension : multi-agents (5 à 10 voitures changeant de voie simultanément) + quantification des incertitudes (Dropout/Ensemble)
- Sim2Real : Validation de la généralisation des politiques sur les jeux de données nuScenes/Waymo

---

## 4. Résumé des principales lacunes de la recherche dans diverses directions| Itinéraire | Statut du blog | Le plus grand écart | Difficulté à se réconcilier |
|------|---------|---------|---------|
| Papier A (KAT-MARL) | Cadre théorique complet, dérivation claire des équations | Manque de données expérimentales de simulation à grande échelle | ★★☆ (3–4 mois) |
| Papier B (planification à trois niveaux) | Conception architecturale claire et logique complète | Manque de plus de 100 expériences d'expansion à grande échelle | ★★★ (4 à 5 mois) |
| Papier C (FIM-3DGS) | Dérivation approfondie de la théorie de l'information et bonne compréhension du 3DGS | Manque de mise en œuvre et d'expérimentation en ligne en boucle fermée | ★★★ (3-4 mois) |
| Papier D (Partition fonctionnelle) | Logique d'intégration multi-source claire | Manque de données et d'expériences SIG réelles | ★★☆ (3–4 mois) |
| Papier E (LLM+vérification formelle) | Conception complète de pipeline | Ensemble de données d'évaluation manquant, précision de la traduction inconnue | ★★★★ (6 à 8 mois) |
| Papier F (changement de voie CARLA) | Résultats expérimentaux disponibles | Les scénarios multi-agents doivent être étendus | ★★☆ (3–4 mois) |

---

## 5. Stratégie de soumission et guide de sélection des revues

### Liste des revues/conférences cibles

| Revue / Conférence | Champ | IF / Taux d'acceptation | Cycle de révision | Convient au papier |
|------------|------|------------|---------|-----------|
| **IEEE T-ITS** | Systèmes de renseignement sur les transports | 8,5 / ~20% | 3 à 6 mois | A, B, D |
| **TR Partie C** | Science et ingénierie des transports | 7,6 / ~18% | Avril-juin | B, D, F |
| **IEEE T-ASE** | Science et ingénierie de l'automatisation | 5,9 / ~22% | 3 à 5 mois | Un |
| **IEEERAL** | Robots Express | 4,6 / ~30% | 2 à 3 mois | C |
| **ICRA** | Conférence sur les robots | ~30% | Une fois par an | C, E |
| **IROS** | Sommet des robots | ~40% | Une fois par an | C, E |
| **IJCAI** | Sommet sur l'IA | ~15% | Une fois par an | E |

### Suggestions de chemin de soumission progressiveSur la base des articles publiés par SCI Q3, la stratégie d'**amélioration incrémentielle** est recommandée :

```
阶段一（2026）：冲刺 Q1 期刊
  → Paper A → IEEE T-ITS（同赛道，优势最大）
  → Paper C → IEEE RAL 或 ICRA（快速发表）

阶段二（2026–2027）：扩展并提升
  → Paper B → Transportation Research Part C
  → Paper D → IEEE T-ITS（第二篇，建立系列感）

阶段三（2027–）：攻顶会
  → Paper E → ICRA/IROS 或 IJCAI（高风险高回报）
```

**Conseils clés :**
- T-ITS bénéficie d'une grande acceptation des recherches croisées sur « UAV × Système de transport urbain », ce qui est cohérent avec le domaine des articles publiés, et les évaluateurs ont la plus haute reconnaissance du contexte.
- Les dates limites de l'ICRA sont généralement en septembre de l'année précédente, alors planifiez à l'avance
- Il est recommandé de préimprimer sur arXiv avant de soumettre (acceptation croissante dans le domaine des transports)

---

## 6. Feuille de route de soumission sur 12 mois

```
时间        Paper A（KAT-MARL）     Paper C（FIM-3DGS）    Paper D（功能分区）     Paper B（三层调度）
─────────────────────────────────────────────────────────────────────────────────────────────────
2026/05    ▶ 环境搭建                ▶ FIM模块实现
2026/06    实验训练                  实验训练（AirSim）
2026/07    基线对比                  写稿启动              ▶ GIS数据采集
2026/08    写稿                      写稿完成              实验 + 写稿           ▶ 框架实现
2026/09    ◉ 投 T-ITS               ◉ 投 ICRA/RAL
2026/10                                                    写稿                  规模实验
2026/11                                                    ◉ 投 TR Part C
2026/12                                                                          写稿
2027/01                                                                          ◉ 投 TR Part C
─────────────────────────────────────────────────────────────────────────────────────────────────
◉ = 投稿节点   ▶ = 工作启动
```

---

## 7. Contrat de maintenance pour ce document

**Convention de dénomination des fichiers :** `research-roadmap_v{numéro de version}_{année, mois, jour}.md`

- Version actuelle : `research-roadmap_v1_20260515.md`
- Prochaine mise à jour (après la soumission de l'article A) : `research-roadmap_v2_20260930.md`
- Après avoir reçu les commentaires de révision : `research-roadmap_v3_202611xx.md`

**Contenu modifié pour chaque mise à jour :**
1. Correspondant au calendrier de Paper (progrès réel par rapport au plan)
2. Complétez l'état d'achèvement de la liste d'expériences (appuyez sur ✅)
3. Résumé des commentaires de l'examen et des stratégies de réponse
4. Nouvelles opportunités de papier (telles que les lacunes de recherche récemment découvertes)

> Utiliser la gestion des versions pour le plan de recherche lui-même, car l'orientation de la recherche sera continuellement ajustée avec l'émergence de résultats expérimentaux, de commentaires de revue et de nouveaux articles. Ce document doit être vivant et non jetable.

---

**Annexe : Vérification rapide de la correspondance entre les articles du blog et Paper**| Article de blog | Document correspondant |
|---------|----------|
| marl-kat-uav-conflit | A (principal) |
| résolution de conflit de drones | A (référence) |
| drone-conflit-env-construction | A (environnement expérimental) |
| planification de drones à grande échelle | B (principal) |
| drone-urban-route-planification | B (référence) |
| prochaine-meilleure-vue-nerf-3dgs-exploration | C (principal) |
| théorie-de l'information-perception-active | C (base théorique) |
| drone-nerf-gs-planification | C (référence) |
| uav-cartographie-sémantique-zonage-fonctionnel | D (principal) |
| cartographie-sémantique-twin-nuav-nuav | D (référence) |
| llm-uav-planification-sémantique | E (principal) |
| llm-guidé-UAV-planification-frontières | E (référence) |
| carla-sumo-rl-lane-changement | F (principal) |