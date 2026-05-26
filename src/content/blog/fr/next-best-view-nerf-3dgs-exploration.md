---
title: "La planification Next-Best-View rencontre NeRF/3DGS : la frontière de l'information de la détection active"
description: "Explication détaillée des méthodes de pointe NBV + NeRF/3DGS : cartographie gaussienne active ActiveGAMER, cible proxy SO-NeRF, collecte de données autonome AutoNeRF, couvrant la frontière d'intersection des champs de détection active et de rayonnement neuronal"
tags: ["drone", "FRN", "3DGS", "Meilleure vue suivante", "perception active", "Éclaboussures gaussiennes"]
category: "Tech"
pubDate: 2026-04-27
sourceHash: "030ae332e13fe029a870b23907cdc45d7e0018c0"
---

# Next-Best-View Planning rencontre NeRF/3DGS : la frontière de l'information de la détection active

> **Série de planification de la perception des drones·Partie X+1**
> Focus : méthodes de pointe NBV + NeRF/3DGS, ActiveGAMER, SO-NeRF, exploration active air-sol

---

## 1. Concept de base : Pourquoi NeRF/3DGS est-il un partenaire idéal pour NBV ?

La planification NBV traditionnelle a une faiblesse fatale : **Elle ne sait pas « à quoi ressemble l'invisible »**.

Vous déduisez où le plus d'informations sont basées sur les observations actuelles - mais pour les endroits qui n'ont pas été observés, vous ne pouvez vous fier qu'à l'heuristique (« choisissez un endroit où vous n'êtes jamais allé »).

**NeRF/3DGS change ceci :**

```
传统方法：
  "我前方10米有个物体，但背面我完全看不到"
  → 只能假设背面 = 未知，启发式选个点去看看

NeRF/3DGS：
  "我有个神经辐射场，已经隐式编码了前+背面的大致形状"
  → 可以渲染背面的大致外观，评估信息增益的真实上限
```

C'est pourquoi **NeRF/3DGS est parfait comme « modèle génératif »** pour la détection active : il peut « imaginer » à quoi ressemblerait une région non observée sous n'importe quel angle de vue et être utilisé pour calculer le véritable gain d'informations.

---

## 2. ActiveGAMER : Reconstruction de carte gaussienne active (arXiv, 2025)

**Article :** *ActiveGAMER : Cartographie gaussienne active grâce à un rendu efficace*
**Auteur :** Liyan Chen, Huangying Zhan, Kevin Chen, Xiangyu Xu, Qingan Yan, Changjiang Cai, Yi Xu
**Source :** arXiv :2501.06897, janvier 2025 | **CVPR 2025**

**Contribution de base :**
- Le premier système complet de **Perception Active + Splatting Gaussien 3D**
- Validé en simulation et environnement réel (bras robotique Franka + plateforme drone)
- Implémentation de la **planification NBV en temps réel** (accélération du rendu parallèle GPU)

**Architecture du système :**

```
┌──────────────────────────────────────────────────────────┐
│                  ActiveGAMER Pipeline                   │
│                                                          │
│  Step 1: 初始建图（稀疏视角覆盖）                         │
│  → 3DGS 初始重建（有明显空洞）                           │
│                                                          │
│  Step 2: NBV 选择（主动感知循环）                        │
│  ┌────────────────────────────────────────────────────┐ │
│  │ 候选视角渲染（并行 ray casting through Gaussians）  │ │
│  │ → 渲染深度图 + 渲染 RGB + 渲染不确定性图             │ │
│  │ → 信息增益评估（基于深度不确定度）                   │ │
│  │ → 选择信息增益最大的下一视角                         │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
│  Step 3: 移动 + 精细建图                                  │
│  → UAV 飞行到新视角                                      │
│  → 增量插入新 Gaussians                                  │
│  → 自适应致密化（只加有信息的区域）                       │
│                                                          │
│  Loop: 返回 Step 2，直到覆盖率达到阈值                    │
└──────────────────────────────────────────────────────────┘
```

**Technologie clé :**

### 2.1 Gain d'informations basé sur l'incertitude

**Point clé :** Les paramètres gaussiens du 3DGS ont par nature des **moyennes et des covariances** (distribution gaussienne), et le gain d'informations des observations peut être calculé directement à partir de la distribution des paramètres.**Calcul du gain d'informations :**
$$
\Delta I \approx \sum_{p \in \text{pixels}} \sigma^2_{\text{rendered}}(p)
$$

Autrement dit : la somme des variances des pixels rendus = la quantité d'informations que la perspective peut fournir.

- Grande variance de rendu → La carte de cette zone est encore approximative et des observations supplémentaires sont nécessaires
- Petite variance de rendu → La carte de cette zone est déjà très bonne, mais le bénéfice de l'observation est faible

### 2.2 Évaluation efficace du point de vue des candidats

Le nombre de points de vue candidats dans les méthodes traditionnelles est faible (des dizaines) car chacun doit être entièrement restitué.

**Accélération ActiveGAMER :**
1. Utilisez le **casting de rayons basé sur les éclaboussures** (sans suivre tous les détails)
2. Évaluation par lots et parallèle de centaines de perspectives de candidats
3. Effectuez un rendu complet uniquement sur les candidats top-K
4. Le cycle NBV global est d'environ **10 Hz** (peut être en temps réel !)

### 2.3 Densification adaptative

Toutes les nouvelles perspectives ne valent pas la peine d'ajouter des gaussiennes :
- **Zone d'information élevée** : discontinuité de profondeur, grands changements d'angle de vue → densification
- **Zone d'information faible** : zone de chevauchement, texture clairsemée → sauter

**C'est aussi le plus proche de la direction de votre blog existant ! ** Votre uav-nerf-gs-planning peut citer directement cet article.

---

## 3. SO-NeRF : NeRF NBV pour les cibles proxy (arXiv, 2023)

**Article :** *SO-NeRF : Planification Active View pour NeRF à l'aide d'objectifs de substitution*
**Auteur :** Keifer Lee, Shubham Gupta, Sunglyoung Kim, Bhargav Makwana, Chao Chen, Chen Feng
**Source :** arXiv :2312.XXXXX, décembre 2023

**Contribution de base :**
- Proposition d'**objectifs de substitution** pour résoudre la non-convexité dans l'optimisation NBV
- Evite le problème de l'optimisation directe de la qualité de la reconstruction (non différenciable, calcul lourd)

**Remarque :** SO-NeRF a été publié sur arXiv, et aucun record de publication clair n'a été trouvé.

**Méthode :**

```
传统 NBV：
  目标：max 重建质量（需要完整重建才能评估）
  局限：不可微、慢、需要多次渲染

SO-NeRF：
  目标：max 代理目标（可微、快速）
  代理：渲染深度的不连续性 + 视角覆盖度
  核心：深度梯度 = 物体边界 = 需要更多信息的地方
```**Intuition :** Les lieux présentant de grands dégradés dans la carte de profondeur rendue (mutations de profondeur = limites de l'objet) sont des lieux qui n'ont pas encore été modélisés.

**Différences par rapport à ActiveGAMER :**
- SO-NeRF utilise des gradients de profondeur comme proxy (pas besoin de modifier NeRF lui-même)
- ActiveGAMER avec variance gaussienne (nécessite le cadre probabiliste GS)
- Les deux peuvent se compléter : SO-NeRF fait la sélection des candidats et ActiveGAMER fait la mise au point

---

## 4. AutoNeRF : Collecte de données autonome (arXiv, 2024)

**Article :** *AutoNeRF : Entraînement aux représentations de scènes implicites avec des agents autonomes*
**Auteur :** Pierre Marza, Laetitia Matignon, Olivier Simonin, Dhruv Batra, Christian Wolf, Devendra Singh Chaplot
**Source :**arXiv, 2024

**Contribution de base :**
- Laissez le **agent (robot) décider indépendamment où collecter les données d'entraînement NeRF**
- Vérifié dans l'environnement de simulation Habitat-sim
- Comparaison de plusieurs stratégies actives : aléatoire / basée sur les frontières / basée sur un modèle

**Principales conclusions :**
- Une stratégie simple basée sur les frontières est déjà bien meilleure que le hasard
- Le type de prédiction du modèle (prédire la qualité de nouvelles perspectives à l'aide de NeRF) peut être encore amélioré
- **Collection active vs collection passive** : La qualité de la reconstruction finale est améliorée de 40 %+

**Inspiration sur les drones :**
- La perspective aérienne du drone rend la frontière (limite explorée-inexplorée) plus grande que celle des robots terrestres
- Le NBV aérien doit prendre en compte la **direction verticale** (pas seulement le mouvement horizontal)
- Au sommet du bâtiment et sous la structure en surplomb se trouve la « frontière » unique du drone

---

## 5. Perception active utilisant NeRF (arXiv, 2023)**Article :** *Perception active à l'aide de champs de rayonnement neuronal*
**Auteur :** Siming He, Christopher D. Hsu, Dexter Ong, Yifei Simon Shao, Pratik Chaudhari
**Source :** arXiv :2310.09892, octobre 2023

**Il s'agit d'un article de base sur la théorie de l'information que vous pouvez directement citer dans votre blog ! **

**Contribution de base :**
Dérivez des **premiers principes** ce que la détection active devrait maximiser :

> **Maximiser l'information mutuelle des observations passées vers les observations futures**
> $$\max_a \quad I(Z_{passé} \cup Z_{nouveau}(a); Y)$$

Parmi eux :
- $Z_{past}$ = observations de capteurs existantes
- $Z_{new}(a)$ = nouvelle observation qui sera obtenue après l'exécution de l'action $a$
- $Y$ = état complet de l'environnement

**Trois éléments clés :**

```
1. Scene Representation（场景表示）
   → NeRF 捕获几何 + 外观 + 语义
   → 可以从任意视角渲染合成图像

2. Generative Model（生成模型）
   → NeRF 就是生成模型！给定 pose → 渲染 image
   → 给合成观测评估信息增益

3. Information-Driven Planner（信息驱动规划器）
   → 采样可行的机器人轨迹
   → 在每条轨迹的末端视角渲染
   → 选择渲染图像信息增益最大的轨迹
```

---

## 6. De l'objet à la scène : mise à l'échelle du NBV

### 6.1 NBV à objet unique → NBV au niveau de la scène

Les premiers travaux de NBV se sont concentrés sur la reconstruction complète d'objets uniques :
- L'objet est placé sur le plateau tournant et tourné selon un angle spécifique pour prendre des photos
- Objectif : Couvrir toutes les perspectives et obtenir un modèle 3D complet

**Votre travail sur le drone s'effectue au niveau de la scène :**
- Canyon urbain entier/espace intérieur
- Vous ne pouvez pas le faire un par un, vous avez besoin d'un plan global
- L'**exploration basée sur les frontières** devient la stratégie principale

### 6.2 Exploration basée sur les frontières + gain d'informations

**Frontière** = La frontière entre les zones explorées et inexplorées.

```
经典 Frontier 探索：
  1. 从当前地图提取所有 frontier 点
  2. 选择最近的 frontier → 飞过去
  3. 扩大已知区域
  4. 重复

Frontier + Information Gain：
  1. 从当前地图提取所有 frontier 点
  2. 预测每个 frontier 的信息增益（用 NeRF/3DGS 渲染）
  3. 选择 info/max(distance) 最大的 frontier（权衡信息 + 能量）
  4. 飞过去
  5. 重复
```

**Conception fonctionnelle de compromis :**

$$
\text{score}(f) = \frac{\text{InformationGain}(f)}{\text{TravelCost}(f)} = \frac{I(f)}{\|p_{current} - f\|_2}
$$

Il s'agit en fait du critère du **"rapport information/distance maximale"** dans l'exploration des drones pour garantir l'efficacité du vol.

---

## 7. Applications spécifiques dans les scénarios de drones### 7.1 Exploration du canyon urbain

**Caractéristiques de la scène :**
- Il y a des immeubles de grande hauteur des deux côtés et le ciel est ouvert au sommet
- Le fond est la rue, le signal GNSS est faible
- Le côté est la façade du bâtiment, avec une haute densité d'informations

**Conseils stratégiques NBV :**

```
Phase 1: 建立初始地图
  → 沿建筑边缘飞行，捕获立面纹理
  → 初始重建完成约 30-40%

Phase 2: 填充立面细节
  → 选择立面渲染不确定度大的区域
  → 飞到近处做精细扫描

Phase 3: 顶部覆盖
  → 飞行到建筑顶面高度
  → 俯视捕获屋顶结构

Phase 4: 精细化
  → 重复，直到渲染不确定度全面低于阈值
```

### 7.2 Correspondance avec votre emploi existant

| Ce que vous avez écrit sur votre blog | Correspondant aux composants du système NBV |
|------------------|-----------------|
| Modélisation spatiale 3D (Octree/Grille d'occupation) | Contraintes d'accessibilité + Détection de collision |
| Cartographie NeRF/3DGS | Représentation de scène activement consciente |
| SLAM sémantique | NBV sémantique (prioriser l'analyse des objets « importants ») |
| Boucle fermée de données de simulation | Amélioration des données de détection active |

---

## 8. Détails techniques clés

### 8.1 Résumé des méthodes d'estimation de l'incertitude

| Méthode | Méthode de calcul | Scénarios applicables | En temps réel |
|------|---------|---------|--------|
| **Abandon de Monte Carlo** | Propagation multiple vers l'avant, variance comme incertitude | NeRF (nécessite une modification du réseau) | Lent |
| **Dégradé de substitution** | Rendre le dégradé de profondeur comme proxy | SO-NeRF | Rapide |
| **Variance gaussienne** | Propre propagation de covariance de GS | 3DGS (ActiveGAMER) | Moyen |
| **Aléatoire + Épistémique** | Séparer l'incertitude liée au bruit et l'incertitude relative aux connaissances | Général | Moyen |

### 8.2 Génération de trajectoires candidats

La NBV ne consiste pas seulement à choisir un point, mais à choisir une **trajectoire réalisable** :
- Le drone a des contraintes de vitesse/accélération maximales
- La faisabilité cinétique doit être prise en compte (RRT*/BIT*/MPC)
- Générez généralement d'abord les points finaux candidats, puis vérifiez la faisabilité de la trajectoire

---

## 9. Défis et questions ouvertes

### 9.1 Goulot d'étranglement informatique

Le principal coût de calcul de la NBV :
- **Évaluation des candidats** (des centaines de candidats × rendu = goulot d'étranglement)
- **Calcul du gain d'informations** (nécessite plusieurs rendus)
- **Boucle d'optimisation NBV** (nécessite généralement 10 à 50 itérations)**Solution :**
- Criblage rapide avec rendu basse résolution dès le début
- Évaluation précise à haute résolution des 10 meilleurs candidats uniquement
- Parallélisation GPU (candidat au rendu parallèle)

### 9.2 Environnement dynamique

Les méthodes NBV existantes supposent pour la plupart un environnement statique. Mais dans le canyon urbain :
- La voiture bouge
- Les piétons vont et viennent
- Le bâtiment est peut-être en construction

**QUESTIONS OUVERTES :**
- Comment les objets dynamiques sont-ils inclus dans les calculs de gain d'information ?
- Que dois-je faire si la zone modélisée est bloquée par des objets dynamiques ?
- Compromis entre mises à jour incrémentielles en ligne et reconstructions complètes périodiques ?

### 9.3 NBV sémantique

La plupart des méthodes NBV actuelles ne considèrent que le gain d'informations géométriques. Mais :
- "Ce bâtiment est un musée, plus important qu'un parking"
- "Il y a des panneaux d'affichage sur cette façade, qui ont une densité d'informations plus élevée que le mur blanc."

**Solution :**
- Ajouter **NeRF sémantique** à NeRF/3DGS
- Gain d'information = gain géométrique × poids sémantique
- Semblable à ce que vous avez écrit dans uav-semantic-mapping.md !

---

## 10. Itinéraire de recherche recommandé

**Route A (résultats rapides) :**
1. Basé sur votre article uav-nerf-gs-planning
2. Connectez-vous au module de calcul du gain d'informations d'ActiveGAMER
3. Validez sur votre plateforme de simulation de drone existante
4. Charge de travail estimée : 2-3 mois

**Voie B (étude systématique) :**
1. Mettre en œuvre FIT-SLAM (SLAM actif basé sur FIM)
2. Remplacez la représentation cartographique par votre système 3DGS
3. Ajoutez des poids sémantiques
4. Vérification sur un vrai drone
5. Charge de travail estimée : 6 à 12 mois

**Route C (exploration des frontières) :**
1. Combinez VLM (Direction 1) pour faire "Semantic NBV"
2. VLM évalue l'importance sémantique de chaque frontière
3. Gain informationnel = gain géométrique + gain sémantique
4. Charge de travail estimée : plus de 12 mois, mais il reste encore beaucoup de place pour l'innovation

---

## 📚 Références1. Chen et coll. *ActiveGAMER : cartographie gaussienne active grâce à un rendu efficace*. arXiv : 2501.06897, janvier 2025.
2. Lee et coll. *SO-NeRF : Planification Active View pour NeRF à l'aide d'objectifs de substitution*. arXiv :2312.XXXXX, décembre 2023.
3. Lui et coll. *Perception active utilisant les champs de rayonnement neuronal*. arXiv :2310.09892, octobre 2023.
4. Marza et coll. *AutoNeRF : Formation de représentations de scènes implicites avec des agents autonomes*. arXiv, 2024.
5. Saravanan et coll. *FIT-SLAM : SLAM actif basé sur les informations de Fisher et l'estimation de la traversabilité*. arXiv :2401.09322, janvier 2024.
6. Zhan et coll. *Estimation active de la pose humaine via un agent UAV autonome*. arXiv, 2024.
7. Chaplot et coll. *Apprentissage de l'exploration visuelle pour la navigation à longue distance*. NeuroIPS, 2020.