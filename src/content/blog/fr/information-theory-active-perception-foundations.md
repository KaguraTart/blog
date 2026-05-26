---
title: "Perception active du point de vue de la théorie de l'information : limites inférieures de Fisher Information et Cramér-Rao"
description: "Expliquer le fondement de la théorie de l'information de la détection active à partir des premiers principes : informations de Fisher, limite inférieure de Cramér-Rao, informations mutuelles et son application dans les travaux SLAM tels que FIT-SLAM et la modélisation continue de l'information."
tags: ["drone", "perception active", "théorie de l'information", "Informations sur les pêcheurs", "CLAQUER", "Cramér-Rao"]
category: "Tech"
pubDate: 2026-04-27
sourceHash: "ea628455d699760ba54122071b05535aa55cf481"
---

# Perception active du point de vue de la théorie de l'information : Fisher Information et limite inférieure de Cramér-Rao

> **Série de planification de la perception des drones · Partie X**
> Focus : Bases de la théorie de l'information, cadre de détection active, calcul de l'information de Fisher et application en SLAM

---

## 1. Qu'est-ce que la perception active ?

La perception traditionnelle est **passive** : le robot reçoit les données des capteurs et met à jour un modèle de l'environnement.

La **perception active** va encore plus loin : le robot **choisit activement « où chercher »** pour maximiser la valeur de la tâche.

```
被动感知：
传感器 → 数据 → 地图更新（机器人不动）

主动感知：
当前地图 → 信息价值评估 → 最优下一视角选择 → 移动 → 传感器 → 地图更新
                ↑
           核心问题：如何量化"信息价值"？
```

Pour les drones, la détection active est particulièrement critique :
- **Contrainte énergétique** : Voler consomme de l'énergie et ne peut pas voler de manière aléatoire.
- **Large champ de vision** : Lorsque vous vous déplacez dans les airs, le champ de vision change radicalement et il est crucial de choisir la trajectoire optimale.
- **Espace tridimensionnel** : les bâtiments, les montagnes et les arbres doivent tous être observés sous plusieurs angles pour une modélisation complète.

---

## 2. Fondement mathématique de la théorie de l'information

### 2.1 Informations sur les pêcheurs

Étant donné un modèle probabiliste $p(x|\theta)$, où $\theta$ est le paramètre à estimer, **Fisher Information** mesure la quantité d'informations sur $\theta$ portées par les données d'observation $X$ :

$$
I(\theta) = \mathbb{E}_X \left[ \left( \frac{\partial}{\partial \theta} \log p(X|\theta) \right)^2 \right] = - \mathbb{E}_X \left[ \frac{\partial^2}{\partial \theta^2} \log p(X|\theta) \right]
$$

**Compréhension intuitive :**
- Si $\log p(x|\theta)$ change **très fortement** près de $\theta$, cela signifie que les données sont très sensibles à $\theta$ → Fisher Information **large**
- Si $\log p(x|\theta)$ change **à plat** autour de $\theta$, les données ne sont pas sensibles à $\theta$ → Fisher Information **small**

**Forme scalaire ou matricielle :**- Scalaire : $I(\theta)$ (paramètre unidimensionnel)
- Matrice : **Fisher Information Matrix (FIM)** $I(\boldsymbol{\theta})$ (paramètres multidimensionnels)

$$
[I(\boldsymbol{\theta})]_{ij} = \mathbb{E} \left[ \frac{\partial \log p(X|\boldsymbol{\theta})}{\partial \theta_i} \cdot \frac{\partial \log p(X|\boldsymbol{\theta})}{\partial \theta_j} \right]
$$

FIM est le tenseur métrique riemannien dans l'espace des paramètres, qui détermine avec quelle précision vous pouvez estimer les paramètres.

---

### 2.2 Borne inférieure de Cramér-Rao (CRLB)

La limite inférieure de Cramér-Rao est une application essentielle de Fisher Information : **donne une limite inférieure optimale sur la variance d'un estimateur sans biais**.

$$
\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}
$$

**Signification physique :** Quelle que soit la méthode d'estimation que vous utilisez (tant qu'elle est impartiale), la précision de l'estimation ne peut pas dépasser 1 $/I(\theta)$.

**Signification en SLAM :**
- La borne inférieure de la covariance de la pose du robot $\mathbf{x}$ est déterminée par FIM
- $[\text{Cov}(\mathbf{x})]^{-1} \preceq I(\mathbf{x})$
- Plus l'inverse de FIM est petit → plus l'estimation est précise

---

### 2.3 Informations mutuelles

L'information mutuelle mesure la dépendance statistique entre deux variables aléatoires :

$$
I(X; Y) = \int \int p(x,y) \log \frac{p(x,y)}{p(x)p(y)} \, dx \, dy = H(X) - H(X|Y)
$$

**Signification en perception active :**
- $X$ = futures observations des capteurs
- $Y$ = l'état incertain de la carte actuelle

Maximiser $I(X; Y)$ = choisir la perspective où les observations futures réduiront le mieux l'incertitude de la carte actuelle.Il s'agit de la définition de la théorie de l'information du « **Gain d'information** » dans la perception active.

---

## 3. Cadre de détection active

### 3.1 Problème principal : Next-Best-View (NBV)

Le problème central de la détection active est la **planification NBV** : étant donné la zone actuellement observée, où devrions-nous aller ensuite pour réduire le plus efficacement l'incertitude ?

**Forme mathématique du problème NBV :**

$$
a^* = \arg\max_{a \in \mathcal{A}} \quad \mathbb{E}_{z \sim p(z|x, a)} \left[ \log \det I(\theta_{new}(x, z)) \right] - \log \det I(\theta_{old}(x))
$$

Autrement dit : choisissez l'action $a$ telle que le déterminant du FIM (une mesure scalaire de l'incertitude globale) après l'exécution soit maximisé.

---

### 3.2 Trois composants majeurs du système de détection active

**Cadre de perception active de la théorie de l'information** propose trois composants d'un système de perception active :

```
┌─────────────────────────────────────────────────────────┐
│                   Active Perception System              │
│                                                         │
│  Component 1: 状态估计 & 地图表示                        │
│  (State Estimation & Map Representation)               │
│  → 当前已观测区域的完整表示（几何 + 语义）               │
│                                                         │
│  Component 2: 未来观测合成                               │
│  (Generative Model of Future Observations)              │
│  → 给定候选动作，生成未来会看到的图像/传感器数据         │
│                                                         │
│  Component 3: 信息驱动的规划                              │
│  (Information-Driven Planning)                          │
│  → 在候选轨迹上计算互信息，选择最优                     │
└─────────────────────────────────────────────────────────┘
```

**Pourquoi avez-vous besoin du composant 2 (modèle généré) ? **
- Vous ne pouvez pas vraiment prendre l'avion et essayer tous les endroits (trop cher)
- Vous avez besoin d'un modèle pour "imaginer" ce que vous verriez en volant vers chaque emplacement candidat
- **NeRF/3DGS sont des modèles génératifs parfaits** (déjà écrit à ce sujet dans votre blog !)

---

## 4. Application des informations sur les pêcheurs dans SLAM

### 4.1 FIM en SLAM

En SLAM visuel, le robot doit estimer simultanément :
- **Pose** $\mathbf{x}_k$ (où est la caméra)
- **Map Point** $\mathbf{m}_i$ (où est le point 3D dans l'espace)

Modèle d'observation : $z_{k,i} = h(\mathbf{x}_k, \mathbf{m}_i) + \mathbf{n}$

- $h(\cdot)$ est la fonction de projection (coordonnées de l'image 3D → 2D)
- $\mathbf{n} \sim \mathcal{N}(0, \Sigma)$ est le bruit de mesure**Informations sur les pêcheurs observés :**
$$
I(\mathbf{x}_k, \mathbf{m}_i) = \frac{\partial h^\top}{\partial [\mathbf{x}_k, \mathbf{m}_i]} \Sigma^{-1} \frac{\partial h}{\partial [\mathbf{x}_k, \mathbf{m}_i]}
$$

**Informations clés :**
- En observant le même point 3D, **différentes perspectives** produisent différentes informations de pêcheur
- Plus la profondeur d'observation est profonde (plus on s'éloigne), plus la quantité d'informations est faible
- Plus la ligne de base d'observation est grande (plus le changement d'angle de vue est important), plus la quantité d'informations est grande

**C'est pourquoi les drones doivent choisir activement leur perspective ! **

---

### 4.2 Interprétation des articles classiques

#### **FIT-SLAM (arXiv, janvier 2024)**

**Article :** *FIT-SLAM -- SLAM actif basé sur les informations de Fisher et l'estimation de la traversabilité pour l'exploration dans les environnements 3D*
**Auteur :** Suchetan Saravanan, Corentin Chauffaut, Caroline Chanel, Damien Vivet
**Source :** arXiv :2401.09322, janvier 2024

**Contribution de base :**
- Introduire explicitement **Fisher Information** dans la fonction objectif de **Active SLAM**
- Pensez également à la **Traversabilité** : non seulement "voir clairement", mais aussi "voler".
- Ciblé sur **environnement 3D** (non planaire), adapté à l'exploration de drones dans des canyons urbains complexes

**Remarque :** Cet article a été publié sur arXiv (il a été soumis à l'IEEE ICARA 2024). Aucun enregistrement clair de publication n’a été trouvé lors de la conférence suprême. La version arXiv doit être notée lors de la citation.

---#### **Planification Active View pour Visual SLAM : modélisation continue de l'information (arXiv, 2022/2023)**

**Article :** *Planification Active View pour le SLAM visuel dans les environnements extérieurs basée sur la modélisation continue de l'information*
**Auteur :** Zhihao Wang, Haoyao Chen, Shiwu Zhang, Yunjiang Lou
**Source :** arXiv :2211.xxxxx, 2022

**Contribution de base :**
- Proposition de **modélisation continue de l'information** pour remplacer les grilles d'information discrètes
- Optimiser la vue suivante sur un espace continu plutôt que sur un ensemble discret de points candidats
- Modéliser l'incertitude spatiale en utilisant le **Processus Gaussien (GP)**

**Informations clés :**

Les méthodes traditionnelles discrétisent l'espace en points candidats → le gain d'information n'est évalué que sur cet ensemble limité de points

Méthode continue : utilisez GP pour représenter "la quantité d'informations à n'importe quelle position", puis **optimisez directement dans l'espace continu**

$$
\mu(a) = \text{Action prédite par GP} a \text{La quantité d'informations à} \\
\sigma(a) = \text{Incertitude de prédiction du GP} \\
\text{Fonction d'acquisition : } a^* = \arg\max_a \, \mu(a) + \beta \sigma(a)
$$

**Avantages par rapport aux drones :**
- L'espace de mouvement du drone est continu et ne doit pas être forcé à se discrétiser
- Possibilité d'optimiser des trajectoires complètes 6-DoF plutôt que de simples sélections de waypoints discrets

---

## 5. Calcul du gain d'informations pour la détection active

### 5.1 Gain d'informations basé sur les informations des pêcheurs

**Gain d'information** = changement FIM avant et après l'action :

$$
\Delta I(a) = \det I(\theta_{après}) - \det I(\theta_{avant})
$$Mais le calcul réel ne nécessite pas de véritable reconstruction, mais simplement :
1. Prédire les observations sous un nouvel angle
2. Calculer la FIM des observations nouvellement ajoutées
3. Utilisez le **Complément Schur** pour mettre à jour efficacement le FIM total

### 5.2 Estimation Monte Carlo de l'information mutuelle

Les informations mutuelles $I(X; Y)$ ne peuvent généralement pas être calculées de manière analytique et nécessitent l'utilisation de méthodes de Monte Carlo :

$$
\hat{I}(X; Y) = \frac{1}{N} \sum_{i=1}^N \log \frac{p(x_i|y_i)}{p(x_i)}
$$

En perception active :
- Échantillonnez plusieurs versions de carte possibles à partir d'une distribution incertaine de la carte actuelle
- Pour chaque action candidate, calculez l'**information mutuelle moyenne**
- Sélectionnez l'action avec la plus grande information mutuelle

---

## 6. Théorie de l'information vs autres principes

| Lignes directrices | Avantages | Inconvénients |
|------|------|------|
| **Informations sur les pêcheurs** | Limite inférieure étroite et optimale théorique | Calcul complexe, nécessite un modèle probabiliste |
| **Information mutuelle** | Intuitif et simple à mesurer | Grande variance d'estimation |
| **Entropie** | Intuitif | Impossible de gérer les distributions continues |
| **Basé sur la distance** | Simple et rapide | Ne prend pas en compte l'occlusion/l'apparence |
| **Basé sur la couverture** | Simple | Ne prend pas en compte la densité de l'information |

**Meilleure pratique :** Combinaison de plusieurs critères
- **SÉCURITÉ** : vérification des collisions basée sur la distance
- **Efficacité** : couverture basée sur l'entropie
- **Précision** : précision de pose basée sur FIM

---

## 7. Connexions à votre travail existant

Vous avez déjà écrit sur votre blog :
- **NeRF/3DGS + UAV** : Représentation de l'environnement (exactement Composant 1 de la détection active !)
- **Semantic SLAM** : Cartes avec sémantique (FIM sémantique > FIM géométrique)
- **Digital Twin** : un modèle d'environnement mis à jour en temps réel

**Cela signifie :**
Vous disposez déjà de la **couche de représentation cartographique** du cadre de détection active, et en ajoutant la **couche d'évaluation du gain d'informations**, vous pouvez créer un système de détection active complet !

**Extension naturelle :**
```
你已有的 NeRF/3DGS 地图
    ↓ + FIT-SLAM 的 FIM 计算方法
    ↓ + GP-based continuous NBV 优化
= 你的主动感知 UAV 系统
```

---

## 📚 Références1. Saravanan et coll. *FIT-SLAM - SLAM actif basé sur les informations de Fisher et l'estimation de la traversabilité pour l'exploration dans les environnements 3D*. arXiv :2401.09322, janvier 2024.
2. Wang et coll. *Planification Active View pour le SLAM visuel dans les environnements extérieurs basée sur la modélisation continue des informations*. arXiv, 2022.
3. Chen et coll. *ActiveGAMER : cartographie gaussienne active grâce à un rendu efficace*. arXiv : 2501.06897, janvier 2025.
4. Lee et coll. *SO-NeRF : Planification Active View pour NeRF à l'aide d'objectifs de substitution*. arXiv :2312.XXXXX, décembre 2023.
5. Lui et coll. *Perception active utilisant les champs de rayonnement neuronal*. arXiv :2310.09892, octobre 2023.
6. Marza et coll. *AutoNeRF : Formation de représentations de scènes implicites avec des agents autonomes*. arXiv, 2024.
7. Chaplot et coll. *Apprentissage de l'exploration visuelle pour la navigation à longue distance*. NeuroIPS, 2020.