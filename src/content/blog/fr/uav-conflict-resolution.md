---
title: "Un examen des algorithmes de résolution de conflits pour la planification de la trajectoire des drones"
description: "Analyse approfondie des algorithmes d'identification et de résolution des conflits dans les systèmes multi-UAV, couvrant les méthodes géométriques, les méthodes d'optimisation, la collaboration multi-machines et les méthodes d'apprentissage, des algorithmes classiques à l'examen de pointe des systèmes de progrès"
pubDate: 2026-04-07T11:12:59+08:00
tags: ["drone", "planification du chemin", "résolution de conflits", "multi-agent", "Présentation de l'algorithme"]
category: Tech
sourceHash: "bdea72e467b5ee1ca0825e4536706f6e89e09f1b"
---

# Présentation de l'algorithme de résolution des conflits de planification de trajectoire d'UAV

> À mesure que les véhicules aériens sans pilote (UAV) évoluent du fonctionnement d'une seule machine vers une collaboration en cluster, les conflits de trajectoire sont devenus un problème majeur inévitable lorsque plusieurs drones effectuent des tâches dans le même espace aérien. **La résolution des conflits** fait référence à l'ajustement de la trajectoire ou de la prise de décision de chaque drone pour éliminer l'état de conflit et continuer à accomplir la mission tout en garantissant la sécurité des vols. Cet article trie systématiquement les cadres algorithmiques traditionnels pour l'identification et la résolution des conflits, des méthodes géométriques à l'apprentissage par renforcement profond, et explore les idées fondamentales, les avantages, les inconvénients et l'évolution de chaque technologie.

---

## 1. Définition et classification des conflits

### 1.1 Qu'est-ce qu'un conflit de chemin ?

Dans un système multi-UAV, **Conflit** fait référence à un état dans lequel deux ou plusieurs drones occupent simultanément la même position spatiale (ou moins qu'une distance d'isolement sûre) dans la dimension espace-temps. Formellement :

$$
\exists \, i, j, \, i \neq j, \quad \| \mathbf{p}_i(t) - \mathbf{p}_j(t) \| < d_{coffre-fort}
$$

Parmi eux, $\mathbf{p}_i(t)$ est la position du $i$-ième drone, et $d_{safe}$ est la distance d'isolement de sécurité (généralement 5 à 50 m, selon le scénario de mission).

### 1.2 Classification des conflits

| Tapez | Descriptif | Scénario typique |
|------|------|--------------|
| **Conflit spatial** | Les trajectoires se croisent dans l'espace | Routes croisées, vols opposés |
| **Conflit temporel et spatial** | Les trajectoires se chevauchent dans la dimension temporelle | Entrez dans le même espace aérien l'un après l'autre |
| **Conflit de vitesse** | La vitesse relative dépasse le seuil de sécurité | Scénario de rattrapage |
| **Conflit élevé** | Conflit dans le sens vertical | Carrefour de levage |
| **Conflits dynamiques** | Conflits provoqués par des obstacles en mouvement (autres aéronefs) | Rencontres aériennes |

### 1.3 Mesures du conflit

- **Time-to-Conflict** : prédisez le temps restant avant qu'un conflit ne se produise
- **Probabilité de conflit** : évaluation des risques de conflit en tenant compte de l'incertitude
- **Distance de séparation minimale** : La distance la plus proche entre les trajectoires
- **Resolution Time** : le temps nécessaire pour que l'action de résolution prenne effet

---

## 2. Algorithme d'identification des conflitsL'identification des conflits est une étape préliminaire à la résolution des conflits, et son noyau est la **prédiction des conflits** : déterminer si un conflit se produira avant qu'il ne se produise réellement.

### 2.1 Méthode de prédiction géométrique

La méthode la plus intuitive est la détection spatiale basée sur des calculs géométriques :

```python
import numpy as np

def detect_conflict_2D(traj_i, traj_j, safe_radius=5.0):
    """
    检测两条轨迹是否发生空间冲突
    
    traj_i, traj_j: shape (N, 3) 的轨迹数组，每行为 (x, y, z)
    safe_radius: 安全隔离距离 (m)
    返回: (是否冲突, 最小间隔距离, 冲突时间点索引)
    """
    min_dist = float('inf')
    conflict_time = -1
    
    for t in range(len(traj_i)):
        dist = np.linalg.norm(traj_i[t] - traj_j[t])
        if dist < min_dist:
            min_dist = dist
            if dist < safe_radius:
                conflict_time = t
    
    is_conflict = min_dist < safe_radius
    return is_conflict, min_dist, conflict_time
```

### 2.2 Obstacle de vitesse

**Velocity Obstacle (VO)** est la méthode de détection et de prédiction de conflits la plus classique dans le domaine de la robotique. Il a été introduit dans le domaine des drones par Fioretti & Fraichard (1999).

Idée de base : Construire une « zone interdite » dans l’espace de vitesse. Si le vecteur vitesse actuel du drone se situe dans cette zone, un conflit se produira certainement.

$$
VO_{ij} = \{ \mathbf{v} \mid \lambda(\mathbf{p}_j - \mathbf{p}_i, \mathbf{v} - \mathbf{v}_j) \cap D_{ij} \neq \varnothing \}
$$

Où $D_{ij}$ est un cylindre avec $\mathbf{p}_j - \mathbf{p}_i$ comme axe et le rayon comme distance de sécurité, et $\lambda$ est le demi-rayon.

```python
def velocity_obstacle(p_i, v_i, p_j, v_j, r_safe=5.0):
    """
    计算第 i 架 UAV 的速度障碍区域
    p_i, p_j: 位置向量
    v_i, v_j: 速度向量
    r_safe: 安全半径
    """
    rel_pos = p_j - p_i
    rel_vel = v_i - v_j
    dist = np.linalg.norm(rel_pos)
    
    if dist == 0:
        return None
    
    # 相对位置的夹角（安全圆柱的视角）
    theta = np.arcsin(r_safe / dist)
    
    # 障碍扇区的两条边向量
    dir_pos = rel_pos / dist
    perp_dir = np.array([-dir_pos[1], dir_pos[0]])
    
    # 两条边界速度向量
    v_left  = v_j + np.linalg.norm(v_j) * (np.cos(theta) * dir_pos + np.sin(theta) * perp_dir)
    v_right = v_j + np.linalg.norm(v_j) * (np.cos(theta) * dir_pos - np.sin(theta) * perp_dir)
    
    return v_left, v_right  # VO 的两条边界

def is_in_vo(v_i, v_left, v_right, v_j):
    """判断速度 v_i 是否落在 VO 区域内"""
    # 转换到相对坐标系
    rel_v = v_i - v_j
    rel_left  = v_left - v_j
    rel_right = v_right - v_j
    
    # 检查 rel_v 是否在 rel_left 和 rel_right 之间
    cross_left  = np.cross(rel_left,  rel_v)
    cross_right = np.cross(rel_right, rel_v)
    
    return np.sign(cross_left) == np.sign(cross_right)
```

### 2.3 Détection des conflits tenant compte de l'incertitude

Dans les systèmes réels, les informations de localisation contiennent souvent des incertitudes telles que des erreurs GPS et du bruit des capteurs. **Détection probabiliste des conflits** Introduit la distribution de probabilité dans le jugement des conflits :

$$
P_{conflit} = \int\int \mathbb{1}(\| \mathbf{p}_i - \mathbf{p}_j \| < d_{safe}) \cdot f_i(\mathbf{p}_i) \cdot f_j(\mathbf{p}_j) \, d\mathbf{p}_i \, d\mathbf{p}_j
$$

où $f_i, f_j$ est la fonction de densité de probabilité de l'emplacement (généralement supposée être une distribution gaussienne). Une alarme de conflit est déclenchée lorsque $P_{conflict} > P_{threshold}$.Les méthodes courantes incluent :
- **Monte Carlo Sampling** : Statistiques des ratios de conflits après avoir échantillonné un grand nombre de distributions de probabilité
- **Outil de validation linéaire (LVT)** : approximation analytique des conflits de probabilité sous l'hypothèse d'une distribution gaussienne
- **Stochastic Reachable Set** : représentation d'ensemble basée sur la théorie du contrôle stochastique

---

## 3. Algorithme de résolution des conflits

### 3.1 Méthode géométrique

#### 3.1.1 Méthode Rate Obstacle (Correction Rate Obstacle / VO)

Stratégie d'élimination basée sur VO : trouvez une vitesse cible $\mathbf{v}_{new}$ qui peut éviter la zone VO :

```python
def vo_resolution(p_i, v_i, p_j, v_j, v_max=20.0, r_safe=5.0):
    """
    基于 Velocity Obstacle 的冲突消解
    返回满足速度约束且避开 VO 的新速度
    """
    vo = velocity_obstacle(p_i, v_i, p_j, v_j, r_safe)
    if vo is None:
        return v_i
    
    v_left, v_right = vo
    
    # 所有候选速度（速度空间中均匀采样）
    best_v = v_i
    min_dist_to_vo = float('inf')
    
    for speed in np.linspace(0, v_max, 20):
        for angle in np.linspace(0, 2*np.pi, 36):
            v_candidate = speed * np.array([np.cos(angle), np.sin(angle)])
            
            # 跳过落在 VO 内的速度
            if is_in_vo(v_candidate, v_left, v_right, v_j):
                continue
            
            # 选择最接近原始速度方向且最"远离"VO 的速度
            dist_to_original = np.linalg.norm(v_candidate - v_i)
            # 到 VO 边界的距离
            dist_to_vo = min(
                np.linalg.norm(v_candidate - v_left),
                np.linalg.norm(v_candidate - v_right)
            )
            
            # 优化目标：尽量接近原速度，同时远离 VO
            score = dist_to_vo - 0.5 * dist_to_original
            if dist_to_vo > min_dist_to_vo and dist_to_vo > 1.0:
                min_dist_to_vo = dist_to_vo
                best_v = v_candidate
    
    return best_v
```

#### 3.1.2 Méthode du champ de potentiel artificiel (champ de potentiel artificiel)

Considérez les drones comme des particules chargées se déplaçant dans un « champ potentiel » :
- **Target Point** génère une attraction
- **les obstacles/autres drones** génèrent une force répulsive

$$
\mathbf{F}_{total} = \mathbf{F}_{att} + \sum_j \mathbf{F}_{rep,j}
$$

Parmi eux :
$$
\mathbf{F}_{att} = k_{att} \cdot (\mathbf{p}_{goal} - \mathbf{p}_i)
$$
$$
\mathbf{F}_{rep,j} = k_{rep} \cdot \frac{\mathbf{p}_i - \mathbf{p}_j}{\| \mathbf{p}_i - \mathbf{p}_j \|^3} \cdot (\| \mathbf{p}_i - \mathbf{p}_j \| - d_{safe})
$$

**Avantages** : Calcul rapide, adapté au contrôle en temps réel
**Inconvénients** : Facile de tomber dans les minima locaux (deux drones oscilleront lorsqu'ils "ne pourront pas se pousser")

**Orientations d'amélioration** :
- Mise en forme du champ potentiel : ajustez la forme du champ potentiel pour éviter les minima locaux
- Plusieurs champs de potentiel virtuels : introduisez des obstacles virtuels pour guider les chemins autour des zones de piège
- Méthode hybride : combinée avec A* ou RRT*, utilisant le champ de potentiel pour un réglage fin local

#### 3.1.3 Méthode du diagramme de VoronoïLe diagramme de Voronoï est utilisé pour diviser l'espace en plusieurs zones, et chaque drone vole au sein de son unité de Voronoï, garantissant ainsi que la distance par rapport aux autres drones est toujours supérieure à sa distance par rapport à la limite de Voronoï :

1. Construisez le diagramme de Voronoï du moment actuel en temps réel
2. Chaque drone sélectionne un waypoint proche du point le plus éloigné (point optimal) de son unité Voronoi
3. Déplacez-vous en direction du waypoint et passez au nouveau chemin de Voronoi si un conflit est détecté.

```python
from scipy.spatial import Voronoi
import numpy as np

def voronoi_resolution(positions, v_i, p_i, v_max=20.0):
    """
    基于 Voronoi 图的多机冲突消解
    positions: 所有无人机位置 (N, 2)
    """
    vor = Voronoi(positions)
    
    # 找到当前无人机 i 的 Voronoi 单元
    region_idx = vor.point_region[np.where(vor.point_region == vor.point_region[0])[0][0]]
    # 简化的 Voronoi 路径选择：取 Voronoi 顶点的方向
    vertices = [vor.vertices[v] for v in vor.regions[region_idx] if v >= 0]
    
    if not vertices:
        return v_i
    
    # 选择最接近原始速度方向的安全顶点
    best_dir = v_i / np.linalg.norm(v_i)
    best_vertex = None
    max_projection = -float('inf')
    
    for v in vertices:
        direction = v - p_i
        if np.linalg.norm(direction) < 0.01:
            continue
        direction = direction / np.linalg.norm(direction)
        projection = np.dot(direction, best_dir)
        
        if projection > max_projection:
            max_projection = projection
            best_vertex = v
    
    if best_vertex is None:
        return v_i
    
    return np.clip(best_vertex - p_i, -v_max, v_max)
```

### 3.2 Méthode d'optimisation

#### 3.2.1 Programmation linéaire en nombres entiers mixtes (MILP)

MILP est un cadre classique qui formalise la planification de trajectoires multi-UAV en tant que problème d'optimisation mathématique et a été lancé dans le domaine des UAV par Schouwenaars et al. (2001).

**Idée de base** : La trajectoire continue est représentée par des polynômes par morceaux ou des séquences de points de cheminement fixes, les contraintes de conflit et les contraintes de sécurité sont exprimées par des inégalités linéaires, et des variables entières binaires sont introduites pour représenter la logique de commutation des segments de vol :

$$
\min \quad \sum_i \sum_k \| \mathbf{v}_{i,k} - \mathbf{v}_{i,k-1} \|^2 + \lambda \sum_i \sum_k \| \mathbf{v}_{i,k} - \mathbf{v}_{i,k}^{pref} \|^2
$$

**Contraintes** :
- Contraintes cinématiques : $\| \mathbf{v}_{i,k} \| \leq v_{max}$, $\| \mathbf{a}_{i,k} \| \leq a_{max}$
- Contraintes d'évitement des conflits :
  - Si $\| \mathbf{p}_{i,k} - \mathbf{p}_{j,k} \| < d_{safe}$, alors la variable binaire correspondante $\delta_{ijk} = 1$
  - Introduire la contrainte OR : $\sum_j \delta_{ijk} \leq 0$ (force tous les $\delta$ à 0, c'est-à-dire aucun conflit)
- Contraintes de réalisation des tâches : $\| \mathbf{p}_{i,K} - \mathbf{p}_{objectif,i} \| < \varepsilon$

```python
# 概念性 MILP 冲突约束（伪代码）
"""
minimize: sum_i sum_k (v_i,k - v_pref_i,k)^2

subject to:
    for each UAV i, segment k:
        p_i,k+1 = p_i,k + v_i,k * dt          # 运动学
        norm(v_i,k) <= v_max                   # 速度限幅
        norm(a_i,k) <= a_max                   # 加速度限幅
        
    for each UAV pair (i,j), segment k:
        norm(p_i,k - p_j,k)^2 >= d_safe^2 OR delta_ik = 0
        M * delta_ik >= norm(p_i,k - p_j,k)^2 - d_safe^2
        sum_j delta_jk <= 0                     # 所有 delta 必须为 0
"""
```**Avantages** : Solution globale optimale, garantissant que les contraintes strictes sont satisfaites
**Inconvénients** : La complexité informatique des solveurs MILP (CPLEX, Gurobi) augmente de façon exponentielle avec le nombre de drones, ce qui rend difficile la résolution de scénarios avec plus de 5 à 10 drones en temps réel.

#### 3.2.2 Approche de fenêtre dynamique (DWA)

DWA, emprunté à la planification des mouvements des robots, échantillonne l'espace des vitesses $(v, \omega)$ et évalue chaque vitesse candidate :
1. **Trajectoire vers la cible**
2. **Sécurité en cas de collision** (jugée en simulant des trajectoires à court terme)
3. ** Vitesse d'accessibilité **

```python
def dwa_resolution(p_i, v_i, v_goal, obstacles,
                   v_max=3.0, v_min=0.0,
                   a_max=2.0, dt=0.1, predict_time=2.0,
                   safe_radius=1.5):
    """
    Dynamic Window Approach 用于 UAV 冲突消解
    """
    # 1. 构建动态窗口（当前可达速度集）
   Vw = []
    for v in np.arange(max(0, v_i[0] - a_max*dt), min(v_max, v_i[0] + a_max*dt), 0.1):
        for w in np.arange(v_i[1] - a_max*dt, v_i[1] + a_max*dt, 0.1):
            Vw.append((v, w))
    
    best_score = -float('inf')
    best_v = v_i
    
    for (v, w) in Vw:
        # 2. 预测轨迹
        traj = []
        p_pred = p_i.copy()
        v_pred = np.array([v, w])
        for t in np.arange(0, predict_time, dt):
            traj.append(p_pred.copy())
            p_pred = p_pred + v_pred * dt
        
        # 3. 碰撞检测
        collision = False
        for p_obs in obstacles:
            for p_t in traj:
                if np.linalg.norm(p_t - p_obs) < safe_radius:
                    collision = True
                    break
            if collision:
                break
        
        if collision:
            continue
        
        # 4. 评分函数
        score_heading = np.linalg.norm(p_pred - v_goal)  # 越小越好
        score_velocity = v  # 越大越好（偏好高速）
        score_clearance = min([np.linalg.norm(p_t - p_obs)
                               for p_obs in obstacles for p_t in traj])
        
        total_score = (
            2.0 * (1.0 / (score_heading + 1e-6)) +
            1.0 * score_velocity +
            0.5 * score_clearance
        )
        
        if total_score > best_score:
            best_score = total_score
            best_v = np.array([v, w])
    
    return best_v
```

#### 3.2.3 Contrôle prédictif de modèle distribué (DMPC)

Le DMPC est une méthode courante pour les essaims multi-UAV à grande échelle, où chaque UAV :
1. Construire un modèle de prédiction local basé sur les informations locales et la communication avec les voisins
2. Résoudre des problèmes d'optimisation locale dans un domaine à temps fini
3. Effectuez uniquement la première étape de contrôle, puis effectuez une ré-optimisation continue

**Contraintes de cohérence fondamentales** :
$$
\sum_{j \in \mathcal{N}_i} a_{ij} (\mathbf{x}_i[k+k_p|k] - \mathbf{x}_j[k+k_p|k]) = 0, \quad \forall k_p \in \{1, \dots, N_p\}
$$

Où $\mathcal{N}_i$ est l'ensemble des voisins de l'UAV $i$, et $a_{ij}$ est le poids de la matrice de contiguïté.

Le principal avantage du DMPC est **l'évolutivité** : chaque drone n'a besoin que de communiquer avec ses voisins, et la quantité de calcul n'augmente pas de façon exponentielle avec le nombre de drones mondiaux.

### 3.3 Méthode de collaboration multi-machines

#### 3.3.1 Algorithme de cohérence basé sur la théorie des graphes

Modélisez le système multi-UAV sous la forme du graphique $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ :
- Le nœud $v_i \in \mathcal{V}$ représente le drone
- Edge $e_{ij} \in \mathcal{E}$ représente le lien de communication

**Protocole de consensus** :
$$
\dot{\mathbf{x}}_i = \sum_{j \in \mathcal{N}_i} a_{ij} (\mathbf{x}_j - \mathbf{x}_i)
$$Lorsqu'elle est appliquée à la résolution de conflits, la **priorité** ou la **fonction de coût** de chaque drone est utilisée comme variable d'état, et l'action de résolution est sélectionnée après convergence consensuelle.

Comparaison des topologies courantes :
- **Topologie du voisin le plus proche** : trafic $\mathcal{O}(N)$, mais convergence lente
- **Topologie entièrement connectée** : convergence rapide, mais trafic $\mathcal{O}(N^2)$
- **Pondération Metropolis-Hastings** : équilibre la vitesse de convergence et les frais généraux de communication

#### 3.3.2 Basé sur le marché

Idée d'algorithme bionique : traitez la zone de conflit comme une « ressource », et chaque drone est en compétition pour le droit d'utiliser la ressource par le biais d'enchères :
1. **Enchère** : Chaque drone calcule l'urgence et le coût de sa propre mission
2. **Enchères** : Le plus offrant obtient la priorité, les autres drones attendent ou font le tour
3. **Règlement (allocation)** : mettez à jour le tableau d'allocation des ressources, répétez jusqu'à ce qu'il n'y ait plus de conflit

```python
import heapq

def auction_based_resolution(uavs, conflict_zone, max_iterations=20):
    """
    基于拍卖的多机冲突消解
    uavs: List[UAV] - 无人机列表
    conflict_zone: 冲突区域中心及半径
    """
    allocation = {}  # zone_id -> winner_uav_id
    unallocated = list(uavs)
    
    for iteration in range(max_iterations):
        if not unallocated:
            break
        
        # 每次拍卖冲突区域使用权
        bids = []
        for uav in unallocated:
            urgency = uav.task_deadline - time.now()
            cost = uav.compute_detour_cost(conflict_zone)
            bid = urgency / (cost + 1e-6)
            bids.append((bid, uav))
        
        # 最高出价者胜出
        bids.sort(reverse=True)
        winner_bid, winner = bids[0]
        
        allocation[conflict_zone.id] = winner.id
        unallocated.remove(winner)
        
        # 对非胜出者计算绕行路径
        for uav in unallocated:
            uav.compute_alternative_path(conflict_zone)
    
    return allocation
```

#### 3.3.3 Méthode de la théorie des jeux

Modélisez la résolution des conflits sous la forme d'un **jeu non coopératif** :
- Chaque drone est un **Joueur**
- La trajectoire de chaque drone est **Stratégie**
- Minimiser son propre risque de conflit et le coût de son vol est une **Fonction avantage (utilitaire)**

**Nash Equilibrium** est un ensemble de combinaisons de stratégies dans lesquelles aucun joueur ne peut obtenir de meilleurs rendements en changeant unilatéralement sa stratégie :

$$
\forall i, \quad \mathbf{s}_i^* \in \arg\min_{\mathbf{s}_i \in \mathcal{S}_i}
\mathcal{J}_i(\mathbf{s}_i^*, \mathbf{s}_{-i}^*)
$$

L'**équilibre corrélé** est plus facile à résoudre de manière distribuée que l'équilibre de Nash et est plus pratique dans les clusters d'UAV.

### 3.4 Méthode d'apprentissage

#### 3.4.1 Apprentissage par renforcement (RL)

Ces dernières années, l'**Deep Reinforcement Learning (DRL)** a réalisé des progrès significatifs dans la résolution des conflits entre les clusters de drones. Cadre typique :- **Espace d'état $\mathcal{S}$** : positions, vitesses, points cibles, obstacles de tous les drones
- **Espace d'action $\mathcal{A}$** : changement de vitesse $(\Delta v_x, \Delta v_y, \Delta v_z)$ ou changement d'angle de cap
- **Fonction de récompense $\mathcal{R}$** :
  - Collision : $r_{collision} = -100$
  - Proche de la cible : $r_{progress} = +10 \cdot \Delta dist$
  - Maintenir une distance de sécurité : $r_{safety} = +5$ (quand $\|p_i - p_j\| > d_{safe}$)
  - Consommation d'énergie : $r_{énergie} = -0,1 \cdot \|\Delta v\|^2$

**MADDPG (Multi-Agent DDPG)** est l'un des frameworks RL multi-machines les plus couramment utilisés :

```python
# MADDPG 核心思路（伪代码）
class MADDPGAgent:
    def __init__(self, state_dim, action_dim, lr_actor=1e-4, lr_critic=1e-3):
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim * n_agents, action_dim * n_agents)
        self.target_actor = copy_network(self.actor)
        self.target_critic = copy_network(self.critic)
        self.replay_buffer = ReplayBuffer(capacity=1e6)
    
    def select_action(self, state, noise=0.1):
        action = self.actor.forward(state)
        action += noise * np.random.randn(action.shape)
        return np.clip(action, -1, 1)
    
    def update(self, batch):
        # 从全局视角更新 Critic（这是 MADDPG 的关键创新）
        states, actions, rewards, next_states = batch
        
        # 目标网络更新
        target_actions = [self.target_actor.forward(ns) for ns in next_states]
        target_Q = self.target_critic.forward(
            torch.cat(states), torch.cat(target_actions))
        
        # 均值聚集（Mean Aggregation）：所有智能体的目标 Q 值取平均
        target_Q = sum(target_Q) / n_agents
        
        # 策略更新
        ...
```

#### 3.4.2 Mécanisme d'attention (Attention)

**Graph Attention Network (GAT)** est utilisé pour modéliser la relation d'importance relative entre les drones :

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        self.W = nn.Linear(in_features, out_features)
        self.a = nn.Linear(2 * out_features, 1)
        
        self.leaky_relu = nn.LeakyReLU(alpha)
    
    def forward(self, h, adj):
        """
        h: (N, D_in) - N 个节点的特征
        adj: (N, N) - 邻接矩阵
        """
        Wh = self.W(h)  # (N, D_out)
        N = Wh.size(0)
        
        # 计算注意力系数
        a_input = torch.cat([
            Wh.repeat(1, N).view(N, N, -1),
            Wh.repeat(N, 1).view(N, N, -1)
        ], dim=2)  # (N, N, 2*D_out)
        
        e = self.leaky_relu(self.a(a_input).squeeze(2))  # (N, N)
        
        # Mask 掉非邻接节点
        e = e.masked_fill(adj == 0, -1e9)
        attention = F.softmax(e, dim=1)  # (N, N)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # 特征聚集
        h_out = torch.matmul(attention, Wh)  # (N, D_out)
        return F.elu(h_out)
```

Grâce au GAT, les drones peuvent apprendre de manière adaptative quels avions voisins ont le plus grand impact sur leurs propres décisions, réalisant ainsi une « coordination douce » : il n'est pas nécessaire de communiquer avec tous les drones, mais doit uniquement prêter attention aux avions voisins avec des poids d'attention élevés.

#### 3.4.3 Apprentissage par imitation

Former le réseau politique à l’aide de trajectoires expertes (solutions du DMPC ou méthodes géométriques) :

$$
\mathcal{L} = -\mathbb{E}_{(s,a) \sim d_{\pi^*}}[\log \pi_\theta(a \mid s)]
$$

La méthode **DAgger (Dataset Aggregation)** peut collecter de manière itérative des données annotées par des experts pour résoudre le problème de changement de distribution.

---

## 4. Comparaison d'algorithmes et guide de sélection| Catégorie d'algorithme | En temps réel | Évolutivité | Optimalité | Gestion de l'incertitude | Scénarios d'application typiques |
|---------|--------|---------|--------|----------------|---------------------|
| **VO / Géométrie** | ⭐⭐⭐⭐⭐ | ⭐⭐ | Local optimal | ❌ | 2 à 5 images, conflits de rattrapage |
| **Méthode du champ potentiel** | ⭐⭐⭐⭐⭐ | ⭐⭐ | Optimale locale | ❌ | Évitement d'obstacles en temps réel, obstacles dynamiques |
| **Voronoï** | ⭐⭐⭐ | ⭐⭐⭐ | Local optimal | ❌ | Planification du chemin du cluster clairsemé |
| **MILP** | ⭐ | ⭐⭐ | **Optimum mondial** | ⚠️ Évolutif | ≤10 racks planification hors ligne |
| **DMPC** | ⭐⭐⭐ | ⭐⭐⭐⭐ | Sous-optimal | ⚠️ Intégré | Cluster de 10 à 50 racks |
| **Théorie des graphes/enchères** | ⭐⭐⭐ | ⭐⭐⭐⭐ | Sous-optimal | ❌ | Allocation de tâches de cluster à grande échelle |
| **Théorie des jeux** | ⭐⭐ | ⭐⭐⭐ | Équilibre pertinent | ⚠️ Évolutif | Scénarios compétitifs |
| **DRL** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Stratégie optimale | ✅ Intégré | Plus de 50 clusters, de bout en bout |
| **GAT+RL** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Stratégie optimale | ✅ Intégré | Cluster hétérogène à très grande échelle |

**Suggestions de sélection** :
- **5 ou moins** : Priorisez VO ou DMPC, le calcul est rapide et la qualité de la solution est garantie
- **5–50** : DMPC + Graph Coherence Protocol, ou MADDPG
- **Plus de 50 racks** : GAT + Attention + RL, la stratégie de bout en bout est le choix le plus réaliste
- **Parties concurrentes** : Cadre de la théorie des jeux (équilibre de Nash/équilibre corrélé)
- **Incertitude significative** (erreurs GPS, perturbation du vent) : détection probabiliste de conflits + MPC robuste

---

## 5. Progrès et tendances à la frontière

### 5.1 Apprentissage par renforcement de bout en boutLa plateforme [SMARTS](https://github.com/hijkzzz/SMARTS) et le projet [AlphaPilot](https://www.microsoft.com/en-us/research/project/alpha-pilot/) publiés par Google DeepMind en 2023 ont favorisé l'application du RL de bout en bout dans les clusters d'UAV, démontrant qu'un réseau politique unique, de la perception à la prise de décision, peut traiter directement les données brutes des capteurs.

### 5.2 Apprentissage fédéré

Dans le but de protéger la confidentialité des données de chaque drone, l'expérience de plusieurs drones est regroupée via un apprentissage fédéré :
1. Stratégie de formation locale pour chaque drone
2. Téléchargez uniquement des dégradés au lieu de données brutes sur le serveur central
3. Émettez de nouvelles stratégies après avoir regroupé les mises à jour

Il résout les problèmes de dispersion des données et de difficulté d’acquisition d’étiquettes dans les clusters de drones.

### 5.3 MPC robuste et incertain

Ces dernières années, le **MPC basé sur des tubes** et le **MPC basé sur des scénarios** modélisent l'incertitude sous forme de perturbations limitées ou de scénarios probabilistes pour contraindre explicitement la robustesse des problèmes d'optimisation :

$$
\forall \omega \in \mathbb{W} : \quad \mathbf{x}[k+1] = A\mathbf{x}[k] + B\mathbf{u}[k] + E\omega[k]
$$

Précalculez des « ensembles invariants » pour garantir que les contraintes de sécurité sont toujours respectées dans le pire des cas.

### 5.4 Résolution de conflits multi-objectifs

Dans les tâches réelles, la résolution des conflits doit également prendre en compte :
- **Taux d'achèvement des tâches** : ne vous contentez pas de faire le tour et de provoquer l'expiration de la tâche.
- **Consommation d'énergie** : les drones à puissance limitée doivent minimiser la distance de vol supplémentaire
- **Délai de communication** : le retard des informations dans les systèmes distribués peut conduire à des erreurs de jugement
- **Équité** : Certains drones ne peuvent pas toujours céder (problème de faim)

La recherche de **frontière optimale de Pareto** est l'outil de base pour résoudre les conflits multi-objectifs.

---

## 6. Résumé

La résolution des conflits dans la planification de la trajectoire des drones est un problème transversal couvrant le calcul géométrique, la théorie de l'optimisation, les systèmes distribués et l'apprentissage automatique. De la première méthode de barrière de vitesse géométrique à l'optimisation globale MILP, en passant par le MPC distribué et l'apprentissage par renforcement profond, la principale force motrice de l'évolution des algorithmes a toujours été :

> **Comment trouver des trajectoires plus sûres pour plus de drones dans un temps plus court et dans une plus grande incertitude. **La tendance future sera celle de l'**architecture hybride** : utiliser des méthodes d'apprentissage pour prendre des décisions locales rapides, utiliser des méthodes d'optimisation pour vérifier les trajectoires globales et utiliser des protocoles de communication pour assurer la cohérence de la collaboration multi-machines. La combinaison des trois peut véritablement réaliser un vol autonome d’essaims de drones sûr, efficace et évolutif.

---

**Références** (triées par heure) :1. van den Berg, J., Lin, M. et Manocha, D. (2008). *Obstacles à vitesse réciproque pour la navigation multi-agents en temps réel.* Conférence internationale IEEE sur la robotique et l'automatisation (ICRA).
2. Richards, A. et How, JP (2002). *Planification de trajectoire d'avion avec évitement de collision à l'aide d'une programmation linéaire mixte en nombres entiers.* Conférence de guidage, de navigation et de contrôle (GNC) de l'AIAA.
3. Alonso-Mora, J. et al. (2018). *Évitement des collisions basé sur l'optimisation pour les systèmes multi-véhicules.* Transactions IEEE sur la robotique (TRO).
4. Everett, M. et coll. (2021). *Évitement des collisions dans un trafic dense grâce à un apprentissage par renforcement profond.* Conférence internationale IEEE sur la robotique et l'automatisation (ICRA).
5. Zhou, M. et coll. (2019). *Une enquête sur la planification de trajectoire pour les drones dans des environnements encombrés.* Transactions IEEE sur les systèmes de transport intelligents (T-ITS).
6. Lowe, R. et coll. (2017). *Acteur-critique multi-agents pour les environnements mixtes coopératifs-compétitifs (MADDPG).* Conférence sur les systèmes de traitement de l'information neuronale (NeurIPS).
7. Foerster, J. et coll. (2018). *Gradients politiques multi-agents contrefactuels (COMA).* Conférence AAAI sur l'intelligence artificielle.
8. Rashid, T. et coll. (2018). *QMIX : Factorisation de fonctions de valeur monotones pour un apprentissage par renforcement multi-agents approfondi.* Conférence internationale sur l'apprentissage automatique (ICML).
9. Veličković, P., et al. (2018). *Réseaux d'attention graphique.* Conférence internationale sur les représentations d'apprentissage (ICLR).
10. Yan, C. et coll. (2025). *Apprentissage par renforcement multi-agents avec attention spatio-temporelle pour le flocage avec prévention des collisions d'une flotte évolutive de drones à voilure fixe.* Transactions IEEE sur les systèmes de transport intelligents (T-ITS).
11. Huo, D. et coll. (2023). *Contrôle prédictif de suivi de trajectoire de modèle sans collision pour les drones dans un environnement d'obstacles.* Transactions IEEE sur les systèmes aérospatiaux et électroniques (TAES).
12. Fan, T. et coll. (2020). *Didistribué un système d'évitement des collisions multi-robots via un apprentissage par renforcement profond pour la navigation dans des scénarios complexes.* The International Journal of Robotics Research (IJRR).
13. Jiang, C. et coll. (2024). *Contrôle prédictif de modèle basé sur l'échantillonnage distribué via la propagation des croyances pour la navigation en formation multi-robots.* Lettres de robotique et d'automatisation IEEE (RA-L).
14. Goeckner, A., et al. (2024). *Apprentissage par renforcement multi-agents basé sur un réseau neuronal graphique pour une coordination distribuée résiliente des systèmes multi-robots.* Conférence internationale IEEE/RSJ sur les robots et systèmes intelligents (IROS).