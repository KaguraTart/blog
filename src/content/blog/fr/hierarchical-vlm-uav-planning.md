---
title: "Planification hiérarchique du VLM : laissez le drone comprendre des instructions telles que \"atterrir du côté est du bâtiment 3\"."
description: "Analyse approfondie de l'application du modèle vision-langage-action (VLA) dans la planification de trajectoire d'UAV, passant au peigne fin l'itinéraire d'évolution de la planification sémantique unique de bout en bout à la planification sémantique hiérarchique, couvrant des travaux clés tels que RT-2, OpenVLA, Compositional Foundation Models, LangStrands, etc., analysant pourquoi l'architecture hiérarchique est la solution optimale pour UAV VLA et donnant des directives de mise en œuvre."
tags: ["VLM", "VLA", "intelligence incarnée", "drone", "planification hiérarchique", "RT-2", "OuvertVLA", "Modèle de fondation", "Robotique"]
pubDate: 2026-04-15
---

# Planification hiérarchique VLM : Laissez le drone comprendre des instructions telles que "atterrir du côté est du bâtiment 3"

## 1. Question : Pourquoi le « direct de bout en bout » ne fonctionne-t-il pas pour les drones ?

Imaginez que vous donnez l'ordre à un drone : **"Contournez le bâtiment 2 et atterrissez dans l'espace ouvert du côté est du bâtiment 3"**.

Cette phrase est simple pour les humains, mais elle contient trois niveaux de sens :

1. **Couche sémantique** : recherchez les emplacements du bâtiment 2 et du bâtiment 3
2. **Raisonnement spatial** : en partant de l'emplacement actuel, contournez le bâtiment 2 et atteignez le côté est du bâtiment 3.
3. **Exécution physique** : générez des trajectoires 3D fluides et contrôlez les moteurs pour un suivi en temps réel

Si vous utilisez un VLA (Vision-Langage-Action) pur de bout en bout pour émettre directement les signaux de commande du moteur à partir des images de la caméra, vous serez confronté à deux problèmes fondamentaux :

- **Inadéquation des fréquences de sortie** : l'inférence du modèle de langage de VLA prend des centaines de millisecondes à la fois, mais le contrôle du drone nécessite plus de 100 Hz de signaux en temps réel.
- **Sécurité incontrôlable** : Le modèle de boîte noire de bout en bout ne peut garantir que la trajectoire générée répond aux contraintes physiques. Que dois-je faire si je heurte un bâtiment ?

Par conséquent, la **planification hiérarchique VLM** est devenue un choix courant pour l'industrie et le monde universitaire.

## 2. Présentation de l'architecture en couches

```
用户指令: "绕过2号楼，在3号楼东侧降落"

┌─────────────────────────────────────────────┐
│ Layer 1: 语义理解层 (VLM/LLM)               │
│ 输入: 文本指令 + 地图/图像                   │
│ 输出: "先向北飞，绕过2号楼，再向东飞..."       │
│       (子目标序列)                            │
│ 模型: GPT-4V / LLaVA / Gemini 2.0 Flash    │
└─────────────────────────────────────────────┘
                    ↓ 子目标
┌─────────────────────────────────────────────┐
│ Layer 2: 轨迹规划层 (MPC / RRT* / ESDF)    │
│ 输入: 当前状态 + 子目标                      │
│ 输出: 空间路径点序列 (x,y,z,t)              │
│ 特点: 有理论安全性保证（可源性、碰撞检测）     │
└─────────────────────────────────────────────┘
                    ↓ 路径点
┌─────────────────────────────────────────────┐
│ Layer 3: 控制执行层 (PID / 非线性MPC)        │
│ 输入: 期望轨迹 + 当前状态                    │
│ 输出: 电机转速 / PWM 信号                    │
│ 频率: 100-400Hz (实时)                      │
└─────────────────────────────────────────────┘
```

Les avantages de cette superposition : **Chaque couche est découplée et peut être entraînée/optimisée indépendamment en utilisant la méthode la plus adaptée à cette couche**, sans avoir besoin d'une formation de bout en bout en lien complet.

## 3. Explication détaillée de chaque couche

### 3.1 Couche 1 : Couche de compréhension sémantique - laissez VLM comprendre les instructions

Il s’agit de la couche la plus « intelligente » de l’architecture en couches, et là où les grands modèles sont les plus utiles.

**Mission principale :**
- Analyser les instructions en langage naturel pour extraire les repères et contraintes clés ("Building 2", "East Side", "Bypass")
- Aligner les informations cartographiques 2D/3D avec les observations visuelles
- Séquence de sous-objectifs sémantiques de sortie (liste de sous-objectifs)

**Technologie clé : mise à la terre visuelle**

Mapper une référence abstraite dans le langage ("Côté Est du Bâtiment 3") à un emplacement concret dans la carte/image :

```python
# 伪代码示例
def parse_instruction(instruction: str, map_image: Image, ego_view: Image):
    # Step 1: 用 VLM 理解指令中的地标
    landmarks = vlm.extract_landmarks(
        instruction,  # "绕过2号楼，在3号楼东侧降落"
        map_image    # 鸟瞰地图
    )
    # → ["building_2 (polygon: [[x1,y1],...])", 
    #    "landing_zone: east_side_of_building_3"]

    # Step 2: 将地标转换为坐标
    goal_pose = convert_to_waypoints(landmarks, ego_view)

    # Step 3: 生成子目标序列
    subgoals = plan_subgoal_sequence(
        current_pose, goal_pose,
        constraints=[avoid(building_2), approach(building_3, side=east)]
    )
    # → ["fly_north_50m", "turn_east", "descend_20m", "hover_at_landing_zone"]

    return subgoals
```

**Sélection du modèle :**| Modèle | Avantages | Inconvénients | Scénarios applicables |
|------|------|------|--------------|
| GPT-4V / GPT-4o | Forte capacité de raisonnement, forte multimodalité | Connexion Internet requise, latence élevée | Cloud, insensible à la latence |
| Gémeaux 2.0 Flash | Gratuit, rapide, prend en charge le déploiement local | Compréhension générale des instructions chinoises | Déploiement en périphérie locale |
| LLaVA 7B | Déployable localement, open source | Faible dans la compréhension d’instructions complexes | Drone de bord |
| Qwen2-VL | Adapté aux chinois, open source | Le déploiement Edge doit être quantifié | Scénarios nationaux |

**Progrès récents (2024-2025) :**

- **LLaVA-Plan** — Ajoute un chef de planification basé sur LLaVA, spécialisé dans la décomposition des tâches
- **GPT-4o Live Voice** — compréhension des commandes vocales de bout en bout, aucun ASR requis, aucune interruption

### 3.2 Couche 2 : Couche de planification de trajectoire – de la sémantique aux chemins spatiaux

Cette couche reçoit les sous-objectifs de la couche sémantique et génère des chemins géométriques.

**Méthode classique (style sans apprentissage) :**

```python
# RRT* 全局路径规划
path = rrt_star(
    start=current_pose,
    goal=subgoal,
    obstacles=building_2_obstacle,  # 来自语义层的输出
    max_iterations=1000,
    connection_radius=5.0
)

# ESDF 局部避障（实时）
esdf_map = build_esdf_from_lidar(ego_view)
safe_direction = esdf_map.gradient_at(current_position)

# MPC 轨迹优化
trajectory = mpc.optimize(
    horizon=20,
    dynamics=uav_dynamics,
    obstacles=esdf_map,
    cost=[trajectory_smoothness, progress_to_goal, control_effort]
)
```

**Méthode d'apprentissage (apprentissage par renforcement) :**

```python
# 策略网络：输入当前状态+目标 → 输出控制动作
policy = PPO(
    obs_dim=state_dim,      # 位置、速度、姿态、附近障碍物
    act_dim=action_dim,     # 速度指令 (vx, vy, vz)
)

# 在仿真中训练，通过 Domain Randomization 提升泛化性
# DR参数：风速、延迟、传感器噪声、空气质量
env.set_domain_randomization(
    wind=(0, 5),           # m/s
    comm_latency=(0, 100), # ms
    sensor_noise=(0, 0.05) # 归一化噪声
)
```

**Pourquoi ne pas remplacer cette couche par du RL pur ? **

Parce que les trajectoires RL pures n'ont aucune garantie de sécurité théorique - RL peut trouver des chemins qui « semblent réalisables mais qui heurtent en réalité un mur ». La combinaison ESDF + MPC offre une garantie d'accessibilité : **Tant que MPC peut trouver une solution, elle ne rencontrera pas d'obstacles**.

### 3.3 Couche 3 : Couche d'exécution du contrôle - suivi stable en temps réel

Cette couche est la plus mature et la théorie du contrôle traditionnelle est tout à fait suffisante :

```python
# 非线性 MPC 控制
class UAVController:
    def __init__(self):
        self.mpc = NMPC(
            horizon=10,          # 预测 10 步 (~1秒)
            dt=0.1,             # 控制周期 10Hz
            Q=diag([1,1,1]),    # 位置误差权重
            R=diag([0.1,0.1])   # 控制量权重
        )

    def control(self, state, ref_trajectory):
        # ref_trajectory 来自 Layer 2
        u = self.mpc.solve(state, ref_trajectory)
        return self.motor_mixer.mix(u)  # 转换为电机转速

    def safety_check(self, state):
        # 实时安全兜底：如果状态危险，强制悬停
        if state.altitude < 2.0 and state.speed < 0.5:
            return "LANDING"
        return "FLY"
```

## 4. Travaux de recherche clés

### 4.1 Modèles de base compositionnels pour la planification hiérarchique

**Ajay et coll., arXiv :2309.08587 (2023)**

Cet article constitue un ouvrage de base très important et propose le concept de « modèle de base combiné » :- **Idée de base** : utilisez plusieurs modèles de base dédiés à combiner, chaque couche fait une chose et utilisez la combinaison pour effectuer des tâches complexes
- **Architecture** : encodeur visuel + modèle de langage + décodeur d'actions, cascade hiérarchique
- **Expérience** : Vérifié sur un bras de robot, prouvant que la superposition se généralise mieux que le pur de bout en bout

**Pourquoi c'est inspirant pour les UAV :** La planification hiérarchique permet à chaque couche de réutiliser indépendamment des modèles pré-entraînés, sans avoir besoin de recycler l'ensemble du système pour les scénarios d'UAV.

### 4.2 LangStrands — Robot contrôlé par le langage naturel

**LangStrands (2024)** – Utiliser le langage naturel pour contrôler les robots afin d'effectuer des tâches de reconnaissance/opération dans des scénarios industriels :

-Prend en charge les instructions complexes : "Vérifiez d'abord l'équipement dans la zone A, et si une anomalie est détectée, signalez-vous à l'emplacement B"
- Analyser les instructions dans Task Graph, prenant en charge les branches et boucles conditionnelles
- Prend en charge la collaboration multi-robots, chaque robot reçoit différentes sous-tâches

**Références provenant des drones :** Les idées d'analyse de la carte de mission de LangStrands peuvent être directement transférées aux drones, telles que des tâches complexes telles que « première reconnaissance de 5 points cibles, puis retour à la base ».

### 4.3 Arbre de pensées incarné — Planification assistée par un modèle mondial

**Xu et coll., arXiv :2512.08188 (2025)**

- Utiliser le modèle mondial pour prédire les conséquences physiques (changements d'état environnemental) après l'exécution de l'action
- Utilisez l'Arbre des Pensées pour rechercher la séquence optimale de sous-objectifs avant l'exécution
- Plus basé sur la physique que la planification VLM pure, évitant les chemins « qui semblent corrects mais physiquement impossibles »

**Valeur pour les drones :** Lorsque le drone vole dans les airs, World Model peut prédire l'impact des rafales, la capacité de vol stationnaire après la dégradation de la batterie et planifier une trajectoire plus sûre à l'avance.

### 4.4 OpenVLA — VLA de robots open source

**OpenVLA (2024)** — Modèle VLA open source publié par l'UC Berkeley :

- Paramètres 7B, prenant en charge 97 types d'actions de robot
- Formé sur 220 000 données réelles de robots
- Peut fonctionner sur un GPU grand public (RTX 3090)
- **Potentiel pour les drones** : Bien qu'actuellement OpenVLA soit principalement destiné aux bras robotiques, l'architecture du VLA (codage visuel + LLM + tête d'action) peut être complètement migrée vers des scénarios de drones.### 4.5 Embodied Arena — Plateforme d'évaluation unifiée de l'intelligence incorporée

**Ni et al., arXiv :2509.15273 (2025)**

- Couvre plus de 250 tâches d'intelligence incorporée et des normes d'évaluation unifiées
- Y compris la navigation intérieure, l'exploitation, le vol aérien et d'autres types de tâches
- Fournit des références de performances pour la superposition de drones (précision, latence, taux de réussite pour chaque couche)

**Importance :** Avec une plate-forme d'évaluation unifiée, chaque couche de l'architecture en couches peut être évaluée indépendamment et l'optimisation peut être basée sur des preuves.

## 5. Sim2Real : Comment transférer des stratégies entraînées vers de vrais drones

Un avantage clé de l'architecture en couches : **Chaque couche peut être Sim2Real** indépendamment, éliminant ainsi le besoin d'une migration de bout en bout en liaison complète.

### 5.1 Analyse de difficulté Sim2Real de chaque couche

| Niveau | Environnement de formation | Difficulté de migration | Principaux défis |
|------|---------|---------|---------|
| Couche 1 (VLM) | N'importe quelle image/carte | **Faible** | VLM a été pré-entraîné et a une forte généralisation |
| Couche 2 (RL) | AirSim / Flightmare | **Moyen** | Inadéquation des paramètres aérodynamiques |
| Couche 3 (MPC) | Réglage des paramètres du drone réel | **Faible** | Il suffit de calibrer les paramètres du moteur |

### 5.2 Stratégie Sim2Real pour la couche 2

** Randomisation de domaine (DR) :**
```python
# 仿真训练时随机化关键物理参数
class SimEnv:
    def reset(self):
        self.wind = random.uniform(-3, 3)      # m/s 阵风
        self.motor_lag = random.uniform(0.8, 1.2)  # 电机响应系数
        self.battery_level = random.uniform(0.7, 1.0)  # 电池状态
        self.gps_noise = random.uniform(0, 0.5)  # GPS 噪声 (m)
```

**Calibrage Real2Sim (calibrage de la machine réelle) : **
```python
# 在真实无人机上跑系统辨识
def calibrate_dynamics(real_uav):
    # 激励信号：阶跃输入
    for amplitude in [0.1, 0.3, 0.5]:
        response = real_uav.step_input(thrust=amplitude)
        # 拟合真实电机响应曲线
        motor_model.fit(step_responses)

    # 将标定参数写回仿真
    sim_env.set_dynamics(motor_model.params)
```

### 5.3 Cas pratique : Sim2Real de MADER

**MADER (Multi-Agent DEep Reinforcement learning for Aerial Swarms)** est l'un des meilleurs travaux réalisés par UAV Sim2Real ces dernières années :

- Utilisez MADDPG dans AirSim pour entraîner des stratégies d'évitement d'obstacles coordonnées sur plusieurs machines
- **Key Trick** : ajoutez un délai du capteur (20 à 50 ms) pendant l'entraînement pour permettre à la stratégie d'apprendre à fonctionner sous délai
- **Résultats** : Zéro migration d'échantillon vers un vrai drone Tello, taux de réussite en matière d'évitement d'obstacles > 85 %

## 6. Implémentation de l'ingénierie : construire un drone VLM hiérarchique à partir de zéro### 6.1 Pile technologique recommandée

```
硬件:                      软件:
- Pixhawk 飞控 (或 Crazyflie)   - PX4 / ArduPilot 固件
- Jetson Orin NX (边缘计算)     - ROS 2 Humble
- Livox 激光雷达 / RealSense    - 深度相机 + IMU

软件分层:
┌──────────────────────────────────┐
│ VLM (Layer 1): LLaVA 7B/Qwen2-VL │
│ 推理引擎: llm.cpp / vLLM          │
│ 推理硬件: Jetson Orin NX (INT8)   │
└──────────────────────────────────┘
┌──────────────────────────────────┐
│ 规划器 (Layer 2):                │
│ - 全局: RRT* (OMPL)             │
│ - 局部: OSQP / Crocoddyl (MPC)   │
└──────────────────────────────────┘
┌──────────────────────────────────┐
│ 控制器 (Layer 3): PX4 SITL /     │
│              Ardupilot guided mode │
└──────────────────────────────────┘
```

### 6.2 Interface de messages ROS 2

```python
# Layer 1 → Layer 2 消息
class SubgoalMsg(Message):
    position: Point  # 目标点
    constraints: List[Constraint]  # 避障约束
    priority: int  # 优先级
    timeout: float  # 超时时间

# Layer 2 → Layer 3 消息
class TrajectoryMsg(Message):
    waypoints: List[PoseStamped]  # 路径点序列
    velocities: List[float]       # 期望速度
    start_time: Time             # 计划开始时间
```

### 6.3 Budget de retard (garantie en temps réel)

```
总延迟预算: < 500ms (可接受)
├─ 图像采集: 30ms (30fps)
├─ VLM 推理: 200-400ms (LLaVA 7B @ INT8)
├─ 轨迹规划: 50ms (RRT* + MPC)
└─ 控制器跟踪: 实时 (100Hz)
```

Si l'inférence VLM est trop lente, vous pouvez :
1. Grâce au raisonnement en continu, la couche de planification peut obtenir les résultats intermédiaires à l'avance
2. Utilisez un modèle léger (LLaVA 3B / Qwen2-VL 2B)
3. Cachez les résultats de planification des instructions couramment utilisées (valables lorsque la carte est corrigée)

## 7. Défis actuels et orientations futures

### 7.1 Défis fondamentaux

1. **Latence d'inférence VLM** : LLaVA 7B déduit environ 200 à 400 ms sur le GPU Edge, dépassant les exigences en matière de réponse de sécurité
2. **Ambigüité de la commande** : La commande ambiguë « atterrir dans un endroit sûr » est difficile à gérer pour VLM.
3. **Accumulation d'erreurs multicouches** : Erreur sémantique de la couche 1 → Déviation de chemin de la couche 2 → Gigue de contrôle de la couche 3
4. **Obstacles dynamiques** : la fréquence de mise à jour de la carte ESDF de la couche 2 ne peut pas suivre les obstacles à grande vitesse (tels que les oiseaux en vol)

### 7.2 Orientations futures

- **Fusion de commandes multimodales** : Compréhension conjointe de la voix + geste + point de regard, sauvegarde en cas d'échec d'une seule modalité
- **Lifelong Learning** : mettre à jour les stratégies de couche 2 en ligne pendant le vol pour s'adapter aux environnements inconnus
- **VLM coopératif multi-drones** : un VLM coordonne plusieurs drones au lieu que chacun fonctionne indépendamment
- **World Model Prediction** : utilisez un modèle génératif pour prédire le flux du trafic aérien dans les 5 prochaines secondes et l'éviter à l'avance

## 8. Résumé

La planification hiérarchique VLM est actuellement la voie la plus réalisable pour le renseignement des drones :

- **La couche 1 (VLM)** est responsable de la compréhension sémantique et de l'appel de grands modèles cloud ou Edge.
- **La couche 2 (Planificateur)** est responsable de la conversion de la sémantique en géométrie et peut être combinée avec RL + MPC
- **La couche 3 (contrôleur)** est responsable du suivi en temps réel, et la théorie du contrôle traditionnelle est tout à fait suffisante.

Le principal avantage de cette architecture : **Laissez le grand modèle faire ce pour quoi il est bon (compréhension), laissez la méthode classique faire ce pour quoi elle est bonne (planification de la sécurité)**, chacun effectuant ses propres tâches, au lieu d'utiliser un modèle de boîte noire de bout en bout pour supporter tous les risques.

---

*Références (par ordre de citation dans le texte)*1. Ajay et al., « Modèles de fondation compositionnels pour la planification hiérarchique », arXiv : 2309.08587, 2023
2. Padalkar et al., « OpenVLA : Modèle Open-Source Vision-Langage-Action », 2024
3. Liu et al., « Aligner le cyberespace avec le monde physique : une enquête complète sur l'IA incorporée », arXiv : 2407.06886, 2024
4. Xu et al., « Arbre de pensées incarné : planification de manipulation délibérée avec un modèle mondial incarné », arXiv : 2512.08188, 2025
5. Ni et al., « Embodied Arena : Une plateforme d'évaluation complète pour l'IA incorporée », arXiv :2509.15273, 2025
6. Mu et al., « Computing de bord IoMT amélioré par l'IA : optimisation de la trajectoire des drones », arXiv : 2512.20902, 2025
7. Zhou et al., « OmniShow : Unifier les conditions multimodales pour l'interaction homme-objet », arXiv :2604.11804, 2026

*Auteur : Tarte Kagura | 2026-04-15*