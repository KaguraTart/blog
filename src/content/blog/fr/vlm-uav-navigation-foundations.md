---
title: "Modèles vision-langage pour la navigation par drone : fondement et frontière de la navigation vision-langage"
description: "Présentation du paradigme de base, de l'architecture de base et des travaux représentatifs de la navigation VLM+UAV, couvrant les derniers articles tels que LogisticsVLN, OmniVLN et ASMA"
tags: ["drone", "VLM", "Navigation en langage visuel", "Grand modèle multimodal", "intelligence incarnée"]
category: "Tech"
pubDate: 2026-04-27
---

# Modèles vision-langage pour la navigation par drone : fondement et frontière de la navigation vision-langage

> **Série intelligente UAV · Partie X**
> Focus : Paradigme de base, architecture de base et travail représentatif de VLM+UAV

---

## 1. Contexte : Des commandes verbales au vol autonome

La planification traditionnelle de la trajectoire d'un drone repose sur des fonctions objectives mathématiques précises (telles que le chemin le plus court, la consommation d'énergie minimale), mais les instructions de mission du monde réel sont souvent des descriptions floues du langage naturel :

- "Allez au terrain de basket à côté du toit rouge"
- "Suivez la camionnette blanche et gardez une distance de 50 mètres"
- "Trouvez un point culminant où vous pouvez voir le bâtiment du gouvernement de la ville et survoler"

Ces instructions ne peuvent pas être directement converties en objectifs d'optimisation mathématique, mais elles peuvent être comprises et raisonnées par VLM (Vision-Language Model). La navigation en langage vision (VLN) est la principale direction de recherche pour résoudre ce problème - permettant aux robots (UAV) de naviguer dans un espace physique tridimensionnel selon des instructions en langage naturel.

---

## 2. Définition des tâches : problèmes fondamentaux du VLN

La tâche VLN peut être formalisée comme :

> Étant donné une instruction en langage naturel $I$ et une observation visuelle de départ $O_0$, laissez l'agent effectuer une série d'actions $a_1, a_2, ..., a_T$, et enfin atteindre la position cible décrite par l'instruction.

Les principaux défis sont les suivants :
1. **Fondement sémantique** : cartographier les relations spatiales du langage ("gauche", "arrière", "au-dessus") avec l'espace physique
2. **Raisonnement à long horizon** : les instructions décrivent souvent des tâches complexes en plusieurs étapes
3. **Généralisation à échantillon nul** : bâtiments, environnements et objets invisibles
4. **Caractéristiques tridimensionnelles** : le drone, contrairement aux robots terrestres, possède des capacités complètes de mouvement 3D.

---

## 3. Travail représentatif

### 3.1 LogisticsVLN : UAV VLN pour la distribution des terminaux (arXiv, 2025)**Papier :** *LogisticsVLN : Navigation en langage visuel pour la livraison de terminaux à basse altitude basée sur des drones agents*
**Auteur :** Xinyuan Zhang, Yonglin Tian, Fei Lin, Yue Liu, Jing Ma, Kornélia Sára Szatmáry, Fei-Yue Wang
**Source :** arXiv :2505.03460, mai 2025

**Contribution de base :**
- Le premier cadre de mission VLN spécifiquement destiné à la **livraison de terminaux de drones à basse altitude**
- Proposition d'architecture de drone agentique : perception → raisonnement → planification → contrôle en boucle fermée
- Défis particuliers pour les environnements urbains à basse altitude (occlusion de bâtiments, obstacles dynamiques, dérive GNSS)

**Cadre de méthode :**

```
用户指令："送包裹到红色大门旁边"
    ↓
VLM 语义解析（物体检测 + 空间关系）
    ↓
拓扑地图匹配（检测到的地标 vs 先验地图）
    ↓
路径规划（全局粗规划 + 局部视觉重规划）
    ↓
MPC 控制器执行
```

**Points clés :** Il s'agit actuellement du travail VLN le plus proche des scénarios réels de livraison d'UAV, intégrant le modèle de langage visuel de niveau GPT-4V à la couche de contrôle physique de bout en bout.

---

### 3.2 OmniVLN : VLN côté extrémité multiplateforme ouvert (arXiv, 2026)

**Article :** *OmniVLN : Perception 3D omnidirectionnelle et raisonnement LLM efficace par jeton pour la navigation en langage visuel sur les plates-formes aériennes et terrestres*
**Auteur :** Zhongyuang Liu, Min He, Shaonan Yu et al.
**Source :** arXiv, mars 2026

**Contribution de base :**
- **Perception 3D omnidirectionnelle** : perception d'un champ de vision sphérique à 360°, plus adaptée aux canyons urbains complexes que les caméras traditionnelles orientées vers l'avant
- **Inférence LLM efficace par jeton** : résolvez le goulot d'étranglement en matière de puissance de calcul du déploiement VLM à la périphérie
- **Cadre unifié multiplateforme** : le même ensemble d'algorithmes s'adapte à la fois aux drones et aux robots au sol**Innovation technologique :**
1. **Compression de jetons 3D** : codez les informations spatiales 3D en jetons compacts pour réduire le nombre de jetons d'entrée LLM
2. **Gestion dynamique du champ de vision** : ajustez de manière adaptative la zone d'intérêt en fonction des besoins de navigation
3. **Backbone VLM léger** : version côté client basée sur l'architecture Qwen-VL ou LLaVA

---

### 3.3 ASMA : UAV VLN prenant en compte les limites de sécurité (arXiv, 2024)

**Article :** *ASMA : Un algorithme de marge de sécurité adaptatif pour la navigation des drones en langage visuel via des fonctions de barrière de contrôle sensibles à la scène*
**Source :** arXiv, septembre 2024

**Contribution de base :**
- Intégrer explicitement les **contraintes de sécurité** dans le framework VLN
- Fonctions proposées de barrière de contrôle sensible à la scène (fonction de barrière de contrôle sensible à la scène)
- Assurer des contraintes de sécurité strictes en milieu urbain ouvert

**Pourquoi c'est important :** La plupart des efforts du VLN se concentrent sur la précision de la navigation et ignorent la sécurité. L'ASMA comble cette lacune : les drones peuvent faire des compromis en matière de sécurité entre « ne pas comprendre les instructions » et « heurter le mur ».

---

### 3.4 Navigation visuelle et linguistique pour les drones : présentation (arXiv, 2026)

**Article :** *Navigation visuelle et linguistique pour les drones : progrès, défis et feuille de route de recherche*
**Auteur :** Hanxuan Chen, Jie Zheng, Siqi Yang et al.
**Source :** arXiv :2604.xxxxx, avril 2026

**Couverture générale :**
- Historique du développement du drone VLN (2018-2026)
- Classification des méthodes : apprentissage par imitation / apprentissage par renforcement / inférence LLM
- Principaux enjeux : représentation spatiale tridimensionnelle, environnement dynamique, raisonnement en temps réel
- Jeux de données : D3DROU, AI-TOD, UAV-VLN, etc.
- Orientations futures : grands modèles multimodaux, intelligence incorporée et assurance de la sécurité

---## 4. Décomposition de l'architecture technique

### 4.1 Couche de perception (Perception)

**Configuration de la caméra :**

| Tapez | Avantages | Inconvénients |
|------|------|------|
| RVB orienté vers l'avant | Mature, pas cher | Champ de vision étroit, informations limitées |
| Caméra omnidirectionnelle | Perception à 360° | Basse résolution, grande distorsion |
| Caméra de profondeur | Profondeur dense | Panne en extérieur, portée limitée |
| Multi-caméra | Triangulation stéréo | Étalonnage complexe |

**Responsabilités du module de perception :**

1. Détection d'objets + segmentation sémantique (Grounding DINO, YOLO-World)
2. Extraction des relations spatiales (gauche et droite, haut et bas, distance relative)
3. Construction de graphes de scènes (objet + relation + topologie)

### 4.2 Comprendre la couche

**Comparaison de sélection VLM :**

| Modèle | Volume des paramètres | Capacités visuelles | Déploiement périphérique | Travail représentatif |
|------|--------|---------|---------|---------|
| GPT-4V | ~1,8T | Extrêmement fort | ❌ | Recherche académique |
| GPT-4o | ~200B | Extrêmement fort | ❌ | API Cloud |
| LLaVA-1.6 | 7B/13B/34B | Fort | ✅ (ONNX) | Déploiement local |
| Qwen-VL | 7B/72B | Fort | ✅ | scène chinoise |
| CogVLM | 17B | Fort | ⚠️ | Solution équilibrée |

### 4.3 Couche de planification (Planification)

**Paradigme de planification existant :**

1. **LLM en tant que planificateur** : laissez directement LLM produire des séquences d'actions (ReAct, Reflexion)
   ```
   Instruction → Raisonnement LLM → Séquence d'actions → Exécution
   ```
2. **Planification symbolique PDDL** : LLM génère une description du domaine PDDL, résolue par le planificateur classique
   - Représentant : UniPlan
3. **Planification apprenable** : apprentissage par imitation/apprentissage par renforcement de bout en bout
   - Avantages : S'adapter aux environnements dynamiques
   - Inconvénients : mauvaise généralisation

### 4.4 Couche de contrôle (Contrôle)

**Fonctionnalités de contrôle des drones :**- Nécessite un suivi de trajectoire en temps réel (fréquence de contrôle «> 100 Hz»)
- Le délai d'inférence (deuxième niveau) de VLM/LLM n'est pas cohérent avec le contrôle en temps réel
- **Idée de solution : contrôle hiérarchique**
  - Niveau élevé : VLM/LLM (lent, deuxième niveau) → point cible
  - Niveau bas : MPC/PID (niveau rapide, milliseconde) → contrôle moteur

---

## 5. Principaux défis

### 5.1 Écart Sim2Real

- **Problème :** VLM est pré-entraîné sur ImageNet/COCO et rencontre un nouveau paysage urbain lors d'un vol réel d'UAV
- **Idées de solutions :**
  - Randomisation de domaine (randomisation de simulation)
  - Génération augmentée par récupération (RAG) supplémentaire avant
  - Adaptation auto-supervisée (Ego4D, DyTap)

### 5.2 Délai d'inférence vs contrôle en temps réel

| VLM | Délai d'inférence | Scénarios applicables |
|-----|---------|---------|
| GPT-4o | 1-3s | Planification hors ligne du cloud |
| LLaVA-7B | 0,5-1s | Planification des délais Edge |
| LLaVA-3B | 0,2-0,5 s | Bord en temps réel |

**Orientation de la solution :**

- Architecture double processus : Découplage du thread de raisonnement et du thread de contrôle
- Décodage spéculatif
- Quantification 4 bits (AWQ, GGUF)

### 5.3 Raisonnement spatial tridimensionnel

Les relations spatiales du langage (« derrière l'arbre », « sous le pont ») ne sont pas de simples projections dans l'espace tridimensionnel.

**Frontières de la recherche :**
- SpatialPoint : prédire des waypoints exécutables en 3D
- Les LLM peuvent-ils voir sans pixels ? : tester l'intelligence spatiale du LLM

---

## 6. Résumé de l'ensemble de données| Ensemble de données | Plateforme | Échelle | Caractéristiques |
|--------|------|------|------|
| RxR | Sol | 126 000 commandes | Annotation experte multilingue |
| VLN-CE | Sol | 61K trajectoires | Matterport3D |
| AI-TOD | Drone | ~ 20 000 commandes | Perspective aérienne, photographie aérienne |
| Drone-VLN | Drone | ~10 000 | Scène du canyon urbain |
| D3DROU | Drone | ~5K | Obstacles dynamiques, vol réel |

---

## 7. Orientations futures de la recherche

1. **Fusion multimodale** : RVB + Profondeur + Caméra événementielle + LiDAR
2. **Adaptation sur petits échantillons** : ajustement fin de LoRA / QLoRA pour s'adapter à des environnements urbains spécifiques
3. **VLN de collaboration avec plusieurs drones** : plusieurs drones collaborent pour comprendre la même commande
4. **Assistance au modèle mondial** : intégrez le modèle mondial pour prédire les états futurs
5. **Vérification de sécurité** : méthode formelle pour vérifier la sécurité des décisions VLN

---

## 📚 Références1. Zhang et coll. *LogisticsVLN : navigation en langage visuel pour la livraison de terminaux à basse altitude basée sur des drones agents*. arXiv :2505.03460, 2025.
2. Liu et coll. *OmniVLN : perception 3D omnidirectionnelle et raisonnement LLM efficace par jetons pour la navigation en langage visuel sur les plates-formes aériennes et terrestres*. arXiv, 2026.
3. Chen et coll. *Navigation visuelle et linguistique pour les drones : progrès, défis et feuille de route de recherche*. arXiv, 2026.
4. ASMA. *Un algorithme de marge de sécurité adaptatif pour la navigation des drones en langage visuel via des fonctions de barrière de contrôle sensibles à la scène*. arXiv, 2024.
5. Blukis et coll. *Mappage des instructions de navigation aux actions de contrôle continu avec prédiction de position-visite*. CoRL, 2018.
6. Raychaudhuri et coll. *Instruction zéro-shot centrée sur l'objet suivante : intégration de modèles de base avec la navigation traditionnelle*. arXiv, 2024.