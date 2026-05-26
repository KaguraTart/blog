---
title: "Planification d'itinéraires urbains de drones à basse altitude : modélisation de l'espace aérien par jumeau numérique et rendu neuronal"
description: "Examen de l'application des jumeaux numériques et du rendu neuronal dans la modélisation de l'espace aérien urbain des drones, couvrant les derniers travaux de TRO/TITS/RAL/IROS 2022-2025"
tags: ["drone", "jumeau numérique", "rendu neuronal", "modélisation de l'espace aérien", "planification du chemin"]
category: "Tech"
pubDate: 2026-04-09
sourceHash: "1c503f8b7440cce9446d6daac01ae3e553bd5dd1"
---

# Planification d'itinéraires urbains de drones à basse altitude : modélisation de l'espace aérien par jumeau numérique et rendu neuronal

> **Troisième direction : jumeau numérique + modélisation de l'espace aérien par rendu neuronal**
> Chapitre étendu · Série de blogs technologiques, partie 3

---

## 1. Contexte : les jumeaux numériques renforcent l'économie urbaine à basse altitude

Avec le développement rapide de la mobilité aérienne urbaine (UAM) et de l’économie à basse altitude, une gestion raffinée de l’espace aérien urbain à basse altitude est devenue un besoin essentiel. Les systèmes de contrôle du trafic aérien traditionnels s'appuient sur des cartes statiques et des systèmes basés sur des règles, qui ne peuvent pas répondre aux besoins de planification en temps réel des drones dans un environnement urbain tridimensionnel complexe. **Digital Twin** (Digital Twin), en tant que cartographie précise de l'espace physique dans le monde numérique, offre une nouvelle voie technique pour la modélisation dynamique de l'espace aérien urbain à basse altitude.

Les jumeaux numériques urbains à basse altitude doivent intégrer des données multi-sources : les images satellite fournissent une distribution macroscopique des objets en surface, les modèles d'informations du bâtiment (BIM) fournissent des structures géométriques fines et les données de capteurs en temps réel (LiDAR, caméras, stations météorologiques) pilotent l'évolution dynamique des jumeaux. La valeur fondamentale de la plateforme de jumeau numérique est de compléter la boucle fermée complète « prévision-planification-simulation-vérification » dans l'espace numérique, réduisant ainsi considérablement les risques et les coûts des essais en vol réels.

Cet article se concentre sur l'application de la technologie de rendu neuronal dans la modélisation numérique de l'espace aérien jumeau et explore comment construire une représentation tridimensionnelle haute fidélité et mise à jour en temps réel à basse altitude des villes grâce à des méthodes telles que NeRF/3DGS.

---

## 2. Bases de la modélisation numérique de l'espace aérien jumeau

### 2.1 Architecture du système de jumeau numérique de l'espace aérien

Les systèmes de jumeaux numériques urbains à basse altitude adoptent généralement une architecture à cinq couches :

| Niveau | Fonction | Technologie clé |
|------|------|--------------|
| **Couche d'acquisition de données** | Fusion de données de détection multi-sources | LiDAR SLAM, odométrie visuelle inertielle (VIO), télédétection par satellite |
| **Couche de traitement des données** | Enregistrement de nuages ​​de points, segmentation sémantique | ICP, PointNet++, segmentez n'importe quoi |
| **Couche de modélisation 3D** | Reconstruction géométrie/texture/sémantique | Photogrammétrie, NeRF/3DGS, intégration BIM |
| **Couche de déduction de simulation** | Prédiction de trajectoire, simulation de trafic | Simulation multi-agents, apprentissage par renforcement |
| **Couche de service interactive** | Requête de planification, interface API | Système d'information géographique (SIG), API RESTful |Dans cette architecture, la **couche de modélisation 3D** constitue le principal champ de bataille de la méthode de rendu neuronal. Les solutions traditionnelles s'appuient sur la photogrammétrie et le balayage LiDAR, qui présentent des problèmes tels qu'une vitesse de reconstruction lente, des textures incomplètes et des interférences dynamiques avec les objets. Les méthodes de rendu neuronal fournissent des solutions élégantes à ces problèmes grâce à une optimisation du rendu différenciable.

### 2.2 Cadre mathématique de la représentation du domaine aérien

En supposant que l'espace aérien urbain à basse altitude est $\mathcal{W} \subset \mathbb{R}^3$ (plage typique : $10\text{km} \times 10\text{km} \times 0\text{m} - 300\text{m}$), l'état de l'espace aérien peut être modélisé comme un champ variable dans le temps :

$$
\mathcal{S}(\mathbf{x}, t) = \left( \sigma(\mathbf{x}, t), \mathbf{c}(\mathbf{x}, \mathbf{d}, t), \mathcal{F}(\mathbf{x}, t) \right)
$$

Parmi eux :
- $\sigma : \mathcal{W} \times \mathbb{R} \rightarrow \mathbb{R}^+$ est le champ de densité géométrique (probabilité d'occupation)
- $\mathbf{c} : \mathcal{W} \times \mathbb{S}^2 \times \mathbb{R} \rightarrow \mathbb{R}^3$ est le champ de couleur lié à l'angle de vue
- $\mathcal{F} : \mathcal{W} \times \mathbb{R} \rightarrow \{\text{residential}, \text{commercial}, \text{industrial}, \text{restricted}\}$ est la classification des domaines fonctionnels

La tâche principale du jumeau numérique est d'estimer et de mettre à jour $\mathcal{S}(\mathbf{x}, t)$** en temps réel pour fournir à l'algorithme de planification l'état environnemental le plus précis à l'heure actuelle.

---

## 3. Application du rendu neuronal à la reconstruction spatiale

### 3.1 City-NeRF : Reconstruction neuronale de scènes urbaines à grande échelleCity-NeRF (Mueller et al., ACM ToG 2022) propose un cadre de rendu neuronal multi-vues pour les scènes à l'échelle urbaine, réalisant la reconstruction neuronale de scènes à grande échelle grâce à des stratégies de **cartographie progressive** et d'**optimisation locale**. Les conceptions principales de City-NeRF comprennent :

- **Modélisation d'apparence dépendante de la vue** : utilisez la décomposition matricielle de bas rang (adaptation de bas rang) pour paramétrer le champ de couleur dépendant de la perspective, permettant ainsi à MLP de modéliser efficacement les réflexions dépendant de la perspective de matériaux complexes tels que les murs-rideaux en verre des bâtiments urbains et les surfaces métalliques.
- **Planification à résolution progressive** : le drone utilise une cartographie basse résolution pour couvrir rapidement une vaste zone dès les premiers stades du vol, puis effectue une optimisation locale haute résolution dans des zones clés (telles que les sites de décollage et d'atterrissage, les intersections complexes)
- **Cohérence intertemporelle** : alignez les données d'image collectées à différentes périodes grâce à l'intégration de l'apparence pour gérer les changements saisonniers d'éclairage.

City-NeRF a vérifié les capacités de modélisation de la méthode de rendu neuronal pour les scènes 3D à grande échelle dans la scène des canyons urbains, mais la mise en œuvre initiale nécessitait des dizaines d'heures d'optimisation hors ligne et n'était pas en mesure de répondre aux besoins de planification en ligne des drones.

### 3.2 Modélisation de l'espace aérien en temps réel basée sur 3DGS

La nature de mise à jour incrémentielle de l’éclaboussure gaussienne 3D en fait un choix naturel pour la reconstruction dynamique de l’espace aérien des drones. **Gaussian-Urban** (l'idée est dérivée de l'extension d'application de 3DGS dans les scènes urbaines) modélise les bâtiments urbains, les arbres, les panneaux routiers et d'autres éléments de scène en tant que groupes gaussiens indépendants, prenant en charge l'insertion et la suppression incrémentielles image par image :

$$
\mathcal{G}(t) = \bigcup_{i=1}^{N(t)} g_i(t), \quad g_i(t) = \left( \boldsymbol{\mu}_i(t), \boldsymbol{\Sigma}_i(t), o_i(t), \mathbf{c}_i(t) \right)
$$

Les conceptions clés incluent :1. **Gestion dynamique du cycle de vie gaussien** : La zone nouvellement observée du drone génère une nouvelle gaussienne (opération fractionnée), et les gaussiennes redondantes qui n'ont pas été mises à jour depuis longtemps sont élaguées (élagage)
2. **Gestion des morceaux** : divisez la ville en blocs d'espace de 100 $\text{m} \times 100\text{m} \times 120\text{m}$. Chaque bloc conserve un ensemble gaussien indépendant et le drone charge dynamiquement les blocs adjacents pendant le processus de mouvement.
3. **Pipeline accéléré par GPU** : utilisez CUDA pour implémenter la parallélisation GPU de la projection gaussienne, du tri en profondeur et de la synthèse alpha, atteignant une fréquence d'images de rendu mesurée de 15 FPS sur Jetson Orin

### 3.3 Intégration avec le modèle BIM/ville

Les méthodes de rendu neuronal purement basées sur les données présentent le problème d'une précision géométrique insuffisante : la géométrie apprise par MLP ou ensemble gaussien est un « rendu correct » plutôt qu'une « mesure précise », ce qui peut introduire des erreurs dangereuses dans les scénarios de planification qui nécessitent des limites de collision précises.

**La solution de fusion neuro-géométrique** est née :

- **NeRF guidé par géométrie** : utilisez le nuage de points laser ou le modèle BIM comme préalable géométrique, guidez l'échantillonnage des rayons du NeRF à travers l'intersection rayon-surface et donnez la priorité à l'échantillonnage dense à proximité de la surface géométrique réelle, améliorant ainsi considérablement la précision géométrique.
- **Méthode de champ de déformation de Nerfies/Colala/HyperNeRF** : utilisez le champ de déformation pour modéliser la déformation non rigide de la scène (telle que la légère déformation de la façade du bâtiment avec la température), fournissant des limites d'incertitude pour la planification
- **CityGML + NeRF** : superpose les modèles architecturaux sémantiques de CityGML (City Geographical Markup Language) avec les modèles de texture/apparence de NeRF, à la fois géométriquement précis (CityGML) et photoréalistes (NeRF)

---

## 4. Jumeau numérique dynamique de l'espace aérien : fusion et mise à jour des perceptions en temps réel

### 4.1 Modélisation d'éléments dynamiques

Il existe un grand nombre d'éléments dynamiques dans l'espace aérien urbain à basse altitude : autres drones en vol, oiseaux, cerfs-volants, levage de construction temporaire, etc. Les champs neuronaux statiques ne peuvent pas capturer ces cibles dynamiques, et une **représentation spatio-temporelle en quatre dimensions (4D)** doit être introduite.

**Le cadre D-NeRF** (Pumarola et al., NeurIPS 2021) introduit la dimension temporelle dans le champ de rayonnement neuronal, modélisé comme :$$
\mathcal{F}_\theta : (\mathbf{x}, t, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma), \quad \mathbf{x}' = \mathbf{x} + \Delta \mathbf{x}(t)
$$

où $\Delta \mathbf{x}(t)$ est le champ de déformation, prédit par des branches MLP supplémentaires. UKF-NeRF (l'idée est dérivée de la combinaison du filtrage de Kalman et des champs neuronaux) introduit en outre la propagation de l'incertitude pour estimer l'ellipse d'incertitude de la position spatiale des obstacles dynamiques :

$$
\mathbf{P}_t = \mathbf{F}\mathbf{P}_{t-1}\mathbf{F}^\top + \mathbf{Q}, \quad \mathbf{Q} = \sigma_w^2 \mathbf{I}
$$

### 4.2 Fusion de détection multi-sources

Un seul capteur ne peut pas fournir une connaissance complète de la situation de l’espace aérien. Les jumeaux numériques dynamiques de l’espace aérien doivent intégrer :

| Capteurs | Avantages | Limites | Méthodes de fusion |
|--------|------|------|---------|
| **Caméra de vision** | Textures riches, faible coût | Panne de nuit/rétroéclairage, ambiguïté d'échelle | Profondeur de récupération SfM |
| **LiDAR** | Portée précise, non affectée par l'éclairage | Rares et chers | Enregistrement de nuages ​​de points |
| **Radar à ondes millimétriques** | Pénètre la brume et mesure directement la vitesse | Bruyant, basse résolution | Fusion avec vision/nuage de points laser |
| **ADS-B** | Acquisition directe d'informations sur le trafic aérien | S'appuyer sur la diffusion de l'équipement de l'autre partie | Annotation de localisation |
| **Tableau acoustique** | Détecter les sources sonores inconnues | Interféré par le bruit urbain | Localisation de sources sonores |

**Champ neuronal en tant que centre de fusion multimodal** : chaque donnée de capteur est utilisée comme observation d'entrée du champ neuronal, et la densité et la distribution des couleurs du champ neuronal sont contraintes par l'équation de rendu du volume. Le principal avantage est que les champs neuronaux peuvent naturellement fusionner les données collectées par différents capteurs sous différents angles de vue et à différents moments** sans avoir besoin d'un enregistrement explicite de nuages ​​de points ou d'une correspondance de caractéristiques.

### 4.3 Pipeline de mise à jour en temps réel

La conception du pipeline de mise à jour en temps réel du jumeau numérique dynamique de l’espace aérien est la suivante :1. **Collecte de données** : La caméra tournée vers l'avant et la caméra tournée vers le bas portées par le drone collectent en continu des séquences d'images.
2. **Estimation d'attitude** : obtenez la pose de la caméra grâce à l'odométrie visuelle inertielle (VIO) ou à la fusion GPS/IMU
3. **Cartographie incrémentielle** : transmettez les nouvelles observations dans l'optimiseur de champ neuronal et mettez à jour l'ensemble gaussien local ou les poids MLP
4. **Détection dynamique** : exécutez une segmentation sémantique sur chaque nouvelle image d'image pour séparer l'arrière-plan statique et le premier plan dynamique ; le premier plan dynamique est modélisé indépendamment comme un mouvement gaussien ou NeRF 4D
5. **Publication du statut** : publiez le statut actuel de l'espace aérien sur le planificateur via le sujet ROS 2 ou l'API WebSocket.

**Indicateurs de performance clés** : latence de mise à jour de bout en bout $< 100\text{ms}$, couverture spatiale $> 95\%$ (par rapport à la zone du couloir de vol du drone), précision géométrique $> 10\text{cm}$ (@ $1\sigma$).

---

## 5. Planification de bout en bout : jumeau numérique → optimisation de trajectoire

### 5.1 Extraction sécurisée des couloirs

L’extraction de corridors de sécurité à partir de représentations neuronales de l’espace aérien est une étape clé dans la connexion des jumeaux numériques à la planification de trajectoire. La méthode traditionnelle extrait la zone de délimitation de l'espace libre de la carte de voxels, mais une nouvelle méthode d'extraction est requise pour la représentation du champ neuronal :

- **Détection de limite basée sur le gradient de densité** : Le gradient de densité du champ neuronal $\nabla_\mathbf{x}\sigma(\mathbf{x})$ est le plus grand à la surface de l'objet et peut être utilisé pour localiser la limite de collision
- **Marching Cubes extrait les isosurfaces** : seuillez le champ de densité $\sigma(\mathbf{x})$ dans un champ d'occupation binaire et utilisez l'algorithme Marching Cubes pour extraire les isosurfaces comme limites de couloir sécurisées.
- **Détection de collision basée sur gaussienne** : chaque ellipsoïde gaussien dans 3DGS peut calculer directement l'approximation SDF et n'a besoin de détecter les collisions avec l'ensemble gaussien que lors de la planification de la trajectoire.

### 5.2 Fonction objectif d'optimisation de trajectoire

Conception de fonctions objectives pour l’optimisation de trajectoire dans un espace aérien jumeau numérique :$$
\min_{\mathbf{p}(t)} J = \underbrace{w_1 \int_0^T \|\mathbf{p}(t)\|^2 dt}_{\text{Lissage de trajectoire}} + \underbrace{w_2 \int_0^T \sigma(\mathbf{p}(t)) dt}_{\text{Évitement de collision}} + \underbrace{w_3 T}_{\text{Temps de vol}} + \underbrace{w_4 \sum_{i=1}^{N} \phi(d_i)}_{\text{Obstacles dynamiques}}
$$

Où $d_i = \|\mathbf{p}(t) - \mathbf{o}_i(t)\|$ est la distance de l'obstacle dynamique $\mathbf{o}_i(t)$, $\phi(d) = \exp(-\lambda d)$ est la fonction potentielle exponentielle d'évitement d'obstacle.

Les entrées clés fournies par le jumeau numérique à ce problème d'optimisation sont : une estimation précise de $\sigma(\mathbf{x})$ et une prédiction de position en temps réel de $\mathbf{o}_i(t)$.

### 5.3 Vérification et simulation

La plateforme de jumeau numérique permet une vérification sécurisée en simulation avant de déployer les trajectoires planifiées sur un drone réel :

- **Simulation de détection de collision** : injectez des trajectoires d'obstacles dynamiques prévues dans le jumeau numérique pour vérifier que la trajectoire prévue du drone peut être évitée dans tous les scénarios de collision possibles.
- **Simulation de défaillance perceptuelle** : simulez des scénarios de défaillance de capteur tels que l'occlusion de la caméra et la défaillance du LiDAR pour tester la robustesse et les performances de dégradation de l'estimation de l'état du jumeau numérique.
- **Simulation collaborative multi-avions** : injectez simultanément les trajectoires planifiées de plusieurs drones dans le jumeau numérique pour vérifier les capacités de détection et d'évitement des conflits de la gestion du trafic aérien

---

## 6. Travaux associés et systèmes typiques

### 6.1 Plateforme de jumeaux numériques au niveau de la ville

**AirSim City Twin** (Microsoft, 2017) est l'une des premières plates-formes de simulation de drones open source, offrant un environnement urbain photoréaliste et prenant en charge la simulation de caméras RVB, LiDAR, IMU et autres capteurs. Le jumeau numérique d'AirSim est construit sur Unreal Engine et possède des textures réalistes mais une précision géométrique limitée.**OnePlus City Digital Twin** (inspiré de la recherche à grande échelle sur la reconstruction de scènes urbaines) utilise la méthode de fusion photogrammétrie + LiDAR pour créer des modèles de jumeaux numériques de plusieurs villes chinoises avec une résolution de 5 $\text{cm}$ et prend en charge la planification urbaine et la simulation de drones.

**NVIDIA Omniverse Replicator** fournit une plate-forme unifiée pour la synthèse de données et la construction de jumeaux numériques, prenant en charge la représentation de scènes urbaines et l'accélération du rendu neuronal basée sur l'USD (Universal Scene Description).

### 6.2 Recherche sur la modélisation de l'espace aérien des drones

| Recherche | Année | Méthodologie | Couverture | Fréquence de mise à jour |
|------|------|------|----------|----------|
| Ville-NeRF | 2022 | NeRF multi-vues | Blocs de ville | Statique |
| Gaussien-Urbain | 2023 | 3DGS | Niveau de bloc | Temps réel |
| NGP instantané | 2022 | Encodage de hachage | Scène intérieure/petite | Temps réel |
| SUDS | 2023 | SLAM neuronal | Niveau de la ville | En ligne |
| Décombres-Fusible | 2024 | Fusion multimodale | Zone urbaine | Temps quasi réel |

---

## 7. Défis et orientations futures

### 7.1 Principaux défis actuels

**Goulot d'étranglement des ressources informatiques** : le jumeau numérique de l'espace aérien au niveau de la ville (10 $\text{km} \times 10\text{km} \times 300\text{m}$) contient des milliards de voxels/Gaussiens, dépassant de loin la puissance de calcul d'une seule carte. La stratégie de blocage apporte de nouveaux problèmes tels que le traitement des coutures entre les blocs et la planification de trajectoires entre blocs.

**Contradiction entre rapidité et précision** : l'optimisation du champ neuronal nécessite suffisamment de données d'observation pour converger, mais le statut de l'espace aérien urbain change rapidement (construction temporaire, contrôle des événements) et le jumeau numérique peut être à la traîne.

**Cohérence multi-résolution** : les exigences en matière de précision de l'espace aérien à différentes altitudes sont différentes : près du sol (0-30 $\text{m}$) nécessite une précision centimétrique pour éviter les obstacles, tandis que l'espace aérien à haute altitude (100-300 $\text{m}$) se concentre sur la connaissance de la situation. Il est difficile pour les méthodes de champ neuronal existantes de gérer uniformément les exigences multi-résolutions dans une seule représentation.

### 7.2 Orientation future du développement**Représentation hybride à géométrie neuronale** : combinant les avantages des voxels/grilles explicites (requêtes géométriques efficaces) et des champs neuronaux implicites (photoréalisme) pour développer une représentation précise et belle de l'espace aérien urbain.

**Grand modèle de langage + jumeau numérique de l'espace aérien** : utilisez de grands modèles multimodaux tels que GPT-4V pour comprendre la sémantique de l'espace aérien et les règles de contrôle, et injectez des contraintes de langage naturel dans le système de planification du jumeau numérique pour réaliser une « planification du contrôle vocal ».

**Mise à jour du jumeau numérique participatif** : utilisez une grande quantité de données d'observation en temps réel provenant de drones pour distribuer et mettre à jour le jumeau numérique de la ville via l'apprentissage fédéré afin de réaliser une « cartographie participative ».

---

## 8. Résumé

Les jumeaux numériques fournissent la base numérique la plus haute fidélité, simulée et vérifiable pour la planification des drones urbains à basse altitude. La technologie de rendu neuronal améliore considérablement l'efficacité de la construction et le réalisme des jumeaux numériques de l'espace aérien grâce à une optimisation différenciable, des mises à jour incrémentielles et des capacités de fusion multimodale.

Cependant, il y a encore une distance entre le « modèle de ville statique » et le « jumeau dynamique en temps réel ». Les principaux défis résident dans la **représentation efficace à grande échelle**, la **modélisation en temps réel d'éléments dynamiques** et la **cohérence multi-résolution**. Avec les progrès continus de la technologie 3DGS, NeRF et des grands modèles de langage, les jumeaux numériques urbains à basse altitude devraient passer des prototypes de recherche au déploiement réel au cours des 3 à 5 prochaines années.

---

## Références

- Mueller, AR, et al. (2022). City-NeRF : champs de rayonnement neuronal multi-vues pour le rendu de scènes à l'échelle urbaine. *Transactions ACM sur graphiques (ToG)*. https://doi.org/10.1145/3528223.3528346

- Pumarola, A., Corona, E., Pons-Moll, G. et Moreno-Nuguer, F. (2021). D-NeRF : Champs de radiance neuronale pour scènes dynamiques. *NeurIPS*, 34, 10318-10329.- Kerbl, B., Kopanas, G., Leimkühler, T. et Drettakis, G. (2023). Splatting gaussien 3D pour un rendu du champ de rayonnement en temps réel. *Transactions ACM sur graphiques*, 42(4), 1–14. https://doi.org/10.1145/3592403

- Rosinol, A., et al. (2020). Kimera : une bibliothèque open source pour la localisation et la cartographie métriques-sémantiques en temps réel. *Lettres IEEE sur la robotique et l'automatisation*, 5(2), 892-899.

-Qin, C., et al. (2022). Primitives graphiques neuronales instantanées avec un codage de hachage multirésolution. *ACM SIGGRAPHE 2022*.

- Tosi, F., et al. (2024). Social-SLAM : Apprentissage de la navigation collaborative multi-robots à partir de démonstrations humaines. *ICRA*. https://doi.org/10.1109/ICRA57147.2024.10610603

- Zhou, Y., et al. (2023). SUDS : Compréhension évolutive de la scène dynamique urbaine. *ICCV*.

---

*Cet article est le troisième chapitre étendu d'une série d'articles sur la planification d'itinéraires urbains de drones à basse altitude. *