---
title: "Planification d'itinéraires urbains de drones à basse altitude : synthèse de données de simulation multimodale"
description: "Aperçu de l'application des plateformes de synthèse et de simulation de données multimodales dans la planification urbaine des drones, couvrant les derniers travaux de NeurIPS/ICRA/IROS/TRO 2022-2025"
tags: ["drone", "Simulation multimodale", "Synthèse des données", "Sim2Réel", "apprentissage par renforcement"]
category: "Tech"
pubDate: 2026-04-09
sourceHash: "946877605d6074e428721e879d238873c8cb1d00"
---

# Planification d'itinéraires urbains de drones à basse altitude : synthèse de données de simulation multimodale

> **Orientation 5 : Synthèse de données de simulation multimodale**
> Chapitre étendu · Série de blogs techniques, partie 5

---

## 1. Contexte : Le double dilemme de la rareté des données et des contraintes de sécurité

La formation des algorithmes de planification de drones urbains à basse altitude (en particulier les planificateurs basés sur l’apprentissage par renforcement profond) est confrontée au double dilemme de la rareté des données et des contraintes de sécurité :

**Rareté des données** : le coût de la collecte de données de vol réelles est élevé : cela nécessite beaucoup de main-d'œuvre, de contrôle et de sécurité du site, et les cas particuliers des scènes urbaines complexes (conditions météorologiques extrêmes, obstacles soudains, interférences de signal) sont difficiles à couvrir avec le système. Les ensembles de données publiques (tels que MAVNet, UZH-FPV) sont limités en échelle et difficiles à prendre en charge la formation de modèles d'apprentissage profond de bout en bout.

**Contraintes de sécurité** : Le planificateur d'apprentissage par renforcement produira de nombreux comportements « exploratoires » dans les premières étapes de la formation. Un entraînement direct sur de vrais drones peut entraîner des accidents tels que des collisions et des pertes de contrôle. L'environnement de simulation offre un **lieu de formation à risque zéro**, mais l'écart simulation-réalité (Sim2Real Gap) rend les stratégies entraînées dans la simulation complètement inefficaces sur le drone réel.

La synthèse de données de simulation multimodale a émergé au fur et à mesure que les temps l'exigeaient - en créant un environnement de simulation multicapteur haute fidélité, générant systématiquement des données de formation diverses et à grande échelle, tout en utilisant la randomisation de domaine et la technologie de migration Sim2Real pour combler le fossé entre la simulation et la réalité.

---

## 2. Simulation de capteurs multimodaux

### 2.1 Pourquoi la multimodalité est nécessaire

Il existe des limites de capacité inhérentes pour un seul capteur. L’exploitation sûre des drones urbains à basse altitude nécessite des **capacités de détection redondantes** :

| Capteurs | Compétences de base | Limites clés | Complémentarités |
|--------|---------|---------|--------|
| **Caméra RVB** | Reconnaissance de textures, compréhension sémantique | Panne la nuit, aucune information sur la profondeur | Fournir des capacités de segmentation sémantique |
| **LiDAR** | Télémétrie précise, cartographie 3D | Rares et coûteux | Fournir une géométrie précise |
| **Radar à ondes millimétriques** | Mesure directe de la vitesse par tous temps | Bruyant, basse résolution | Fournir une détection de cible mobile |
| **Imagerie thermique** | Détection des piétons, vision nocturne | Ambiguïté de différence de température, basse résolution | Assurer la détection des usagers de la route vulnérables |
| **Ultrasons** | Évitement d'obstacles à courte portée | Courte portée, sensible aux interférences | Fournit une perception précise à courte distance |La **fusion multimodale** ne consiste pas simplement à « installer quelques capteurs supplémentaires », mais à concevoir une **stratégie de fusion** pour rendre les informations multi-sources complémentaires et redondantes, et améliorer la **tolérance aux pannes** (Fault Tolerance) du système - lorsqu'un certain capteur tombe en panne, le système peut toujours compter sur d'autres capteurs pour fonctionner en toute sécurité.

### 2.2 Principe de simulation de capteur

**Simulation de caméra RVB** Basée sur un pipeline de rendu physique (PBR) :

$$
L_o(\mathbf{x}, \omega_o) = \int_{\Omega} f_r(\omega_i, \omega_o) \cdot L_i(\omega_i) \cdot \cos\theta_i \, d\omega_i
$$

Où $f_r$ est la fonction de distribution de réflexion bidirectionnelle (BRDF), $L_i$ est l'irradiance incidente et le pipeline PBR génère des images photoréalistes en simulant l'interaction physique de la lumière et des matériaux de la scène. Le système de géométrie virtuelle Nanite et le système d'éclairage global Lumen d'Unreal Engine 5 sont actuellement les solutions de rendu en temps réel les plus proches de la réalité physique.

**La simulation LiDAR** est généralement basée sur le raycasting : émettre des rayons depuis la position LiDAR le long de chaque direction de ligne de balayage, détecter l'intersection avec la géométrie de la scène et renvoyer la distance et l'intensité de réflexion :

$$
d = \min_{t > 0} \{ t : \mathbf{o} + t\omega \in \mathcal{O} \}
$$

Où $\mathcal{O}$ est la géométrie occupée par la scène. Les simulations LiDAR haut de gamme (telles que NVIDIA FLIPS) peuvent également simuler des effets physiques tels que le multi-écho et l'élargissement de la forme d'onde.

**La simulation radar à ondes millimétriques** est basée sur le modèle de propagation des ondes électromagnétiques pour simuler l'effet multitrajet (Multipath), l'atténuation d'ombrage (Shadowing) et la réflexion au sol (Ground Bounce) du signal :

$$
P_r = P_t \cdot \frac{G_t G_r \lambda^2}{(4\pi)^3 R^4} \cdot \sigma \cdot L_{\text{atm}} \cdot L_{\text{multipath}}
$$Où $P_r$ est la puissance reçue, $R$ est la distance cible, $\sigma$ est la section efficace radar (RCS) et $L_{\text{multipath}}$ est le facteur d'évanouissement par trajets multiples.

### 2.3 Synchronisation spatio-temporelle multimodale

Le principal défi technique pour la synthèse de données multimodales est la synchronisation spatio-temporelle : chaque donnée de capteur doit être alignée dans un système de temps et de coordonnées unifié :

- **Synchronisation matérielle** : chaque capteur partage le même déclencheur d'horloge (tel que GPS-PPS) et l'erreur d'horodatage $< 1\text{ms}$
- **Alignement de l'horodatage logiciel** : alignement du temps postérieur basé sur le modèle de retard du capteur (délai d'exposition de la caméra, cycle de balayage LiDAR)
- **Alignement spatial** : calibrez les paramètres externes de chaque capteur ($\mathbf{T}_{\text{camera}}^{\text{body}}$, $\mathbf{T}_{\text{lidar}}^{\text{body}}$, etc.) via la carte d'étalonnage ou le modèle CAO, et unifiez les données avec le système de coordonnées aéroporté.

---

## 3. Comparaison et sélection des plateformes de simulation

### 3.1 Plateforme grand public Hengping| Plateforme | Moteur de rendu | Prise en charge multimodale | Simulation physique | Source ouverte | Spécialisation drone | Scénarios applicables |
|------|----------|-----------|----------|------|----------|----------|
| **AirSim** | Moteur irréel | RVB-D / LiDAR / IMU | PX4 SITL | ✅ | ✅Excellent | Planification du chemin aérien |
| **Gazébo** | Ogre3D | Caméra / LiDAR / IMU | ODE/Balle | ✅ | ✅ Riche | Simulation universelle de robots |
| **Volmar** | Unité | Caméra / LiDAR / Événements | - | ✅ | ✅Excellent | Vol à grande vitesse d'un drone |
| **Isaac Sim** | Omnivers | Modal complet | PhysX | Partielle | Général | Simulation industrielle |
| **SORDAMES** | Auto-développé | Caméra / LiDAR | Auto-développé | ❌ | ✅ | Simulation de drone de qualité militaire |
| **CAVS** | Auto-recherché | Mode complet | Auto-recherché | ✅ | ✅ | Recherche UTM à basse altitude |
| **NeuroSIM** | Rendu neuronal | Caméra (NeRF) | - | En cours de recherche | Exploratoire | Formation à la perception neuronale |

### 3.2 Analyse approfondie d'AirSim

Microsoft AirSim est actuellement l'une des plates-formes de simulation de drones les plus utilisées. Il est construit sur Unreal Engine et offre des capacités de simulation de scènes urbaines photoréalistes.

**Architecture de base** :
- **AirSim Plugin** : Un plug-in qui fonctionne dans Unreal Engine et gère la simulation de capteurs, la physique du vol et les interfaces API
- **PX4 SITL** : communique avec AirSim via le protocole MAVLink, prenant en charge la simulation complète du micrologiciel de commande de vol PX4 en boucle
- **RPC Communication** : fournit une API Python/C++ pour prendre en charge un contrôle flexible au niveau de la recherche**Avantages** :
- Rendu photo-réaliste, la scène du canyon urbain est réaliste
- Prend en charge une variété d'avions (MultiRotor, FixedWing, Rover)
- Modèles de capteurs riches (distorsion de la caméra, flou de mouvement, profondeur de champ)
- Changements dynamiques de la météo, de l'éclairage et de l'heure

**Limites** :
- Dépend d'Unreal Engine (gros moteur commercial, courbe d'apprentissage abrupte)
- Prise en charge limitée de Linux (principalement pour Windows)
- La précision de la simulation physique n'est pas aussi bonne que celle des simulateurs de robots professionnels

### 3.3 Flightmare : simulation de drone à grande vitesse

Flightmare développé par l'ETH Zurich est optimisé pour les scénarios de **manœuvres de drones à grande vitesse** et prend en charge la simulation d'une accélération de 10 $\text{m/s}^2+$. C'est un outil idéal pour la recherche sur les vols agressifs.

Caractéristiques de Flightmare :
- **Modular Rendering Pipeline** : moteurs de rendu interchangeables (Unity/OpenGL), prenant en charge les environnements urbains à grande échelle
- **Bibliothèque de scènes à grande échelle** : prédéfinissez diverses scènes telles que des villes, des forêts, des entrepôts, etc.
- **Event Camera Simulation** : prend en charge la simulation de capteur basée sur les événements (Event Camera), adaptée aux scènes de manœuvres à grande vitesse

### 3.4 Orientations émergentes : simulation neuronale

**UniSim** (Zhou et al., NeurIPS 2023 / arxiv preprint) a d'abord proposé le concept de simulation de perception neuronale, utilisant des champs de rayonnement neuronal pour modéliser des arrière-plans statiques + une géométrie explicite pour modéliser des objets dynamiques, afin d'obtenir une génération de données de capteur photoréaliste et contrôlable. Le pipeline principal d’UniSim :

1. Collectez une petite quantité de données du monde réel (environ 20 minutes de vidéo de conduite)
2. Former le modèle d'arrière-plan statique NeRF + le modèle explicite d'objet dynamique
3. Ajustez les trajectoires des caméras, ajoutez/supprimez des objets, modifiez la météo et générez de nouvelles scènes dans NeRF
4. Le rendu neuronal produit RVB, profondeur, vecteur normal et autres données sensorielles

Les données de simulation générées par cette méthode sont très proches des données réelles, réduisant considérablement l'écart Sim2Real, mais les performances en temps réel restent un goulot d'étranglement (la vitesse de génération actuelle est d'environ 0,1 FPS, en temps non réel).

---

## 4. Randomisation de domaine et migration Sim2Real

### 4.1 Principe de randomisation des domainesL'idée principale de la randomisation de domaine (DR) est de randomiser un grand nombre d'attributs non clés dans la simulation, obligeant l'algorithme d'apprentissage à se concentrer sur la compréhension des attributs clés (structure géométrique, informations sémantiques), généralisant ainsi au monde réel.

**Paramètres de randomisation typiques** :

| Catégorie | Paramètres | Plage de randomisation |
|------|------|--------------|
| **Apparence** | Textures, éclairage, météo | Randomisation couleur/intensité, éclairage dynamique |
| **Géométrie** | Taille, position, orientation de l'objet | Position aléatoire des objets non clés |
| **Capteur** | Paramètres internes, bruit, paramètres externes | Décalage de mise au point de la caméra, niveau de bruit LiDAR |
| **Dynamique** | Masse, perturbation du vent, retard | Paramètres $\pm 20\%$ Aléatoire |
| **Contexte** | Complexité de la scène, nombre d'objets | Densité d'objets d'interférence aléatoire |

### 4.2 Adaptation de domaine en ligne

Le problème avec la DR pure est que la randomisation excessive conduit à une formation inefficace : la politique entraîne bien dans des scénarios simples mais se dégrade dans des scénarios complexes. La méthode **Online Adaptation** (Online Adaptation) met à jour en permanence les paramètres de simulation pendant le processus de migration simulation-réel :

**Meta-Sim** (Kar et al., NeurIPS 2019) utilise l'apprentissage par renforcement pour apprendre automatiquement la distribution optimale des paramètres de randomisation de domaine, dans le but de maximiser les performances d'évaluation sur des données réelles :

$$
\theta^* = \arg\max_\theta \mathbb{E}_{\mathbf{s} \sim p_\theta} \left[ \text{Performance}(\pi_\theta, \text{Real}) \right]
$$

**SimBot** (Zhang et al., CoRL 2021) adopte une méthode d'adaptation de domaine pour collecter une petite quantité de données d'interaction de robots réels en même temps pendant le processus de formation, et utilise ces données pour corriger les paramètres du simulateur :

$$
p_{\text{real}} \approx \alpha \cdot p_{\text{sim}} + (1-\alpha) \cdot p_{\text{real,obs}}
$$

### 4.3 Randomisation liée à la tâche ou non pertinente à la tâcheToute randomisation n’est pas bonne pour la généralisation. **Grounding SBIR** (Singh et al., 2023) distingue deux types de randomisation :

- ** Randomisation pertinente pour les tâches ** : randomisation qui modifie directement les décisions stratégiques, telles que l'emplacement des obstacles (affectant les décisions d'évitement des obstacles). Ce type de randomisation **doit être conservé** et constitue un signal nécessaire à l'apprentissage des stratégies de généralisation
- **Randomisation sans rapport avec les tâches** : randomisation qui ne modifie pas les décisions stratégiques, telles que les changements de texture du sol (n'affecte pas la trajectoire de vol). Ce type de randomisation peut réduire ** et éviter de gaspiller la capacité de formation

Le gradient de politique peut identifier automatiquement les paramètres de randomisation liés aux tâches pour obtenir un apprentissage efficace de la distribution DR.

---

## 5. Construction d'actifs numériques : génération d'actifs 3D au niveau de la ville

### 5.1 Pipeline automatisé d'actifs de scène

Construire des scènes de simulation à l’échelle d’une ville nécessite un grand nombre d’actifs 3D (bâtiments, arbres, infrastructures routières). La modélisation manuelle est extrêmement coûteuse (un seul modèle architectural détaillé nécessite 2 à 5 jours-homme) et nécessite la technologie de **Génération procédurale** (Génération procédurale).

**Sat2Map** : Reconstruction automatique de modèles de villes 3D à partir d'images satellite/aériennes :

1. Segmentation sémantique : extraire les toits des bâtiments, les routes et les zones de végétation
2. Estimation monoculaire de la hauteur : prédire la hauteur de chaque bâtiment (basée sur l'analyse des ombres ou des modèles profonds tels que Midas)
3. Reconstruction de la grille : étirez le masque sémantique 2D dans le sens de la hauteur pour générer les murs extérieurs du bâtiment.
4. Cartographie de texture : échantillonnage de textures à partir d'images originales ou de bibliothèques satellites

**Modélisation procédurale** : utilisez le système L ou la grammaire des règles pour générer des façades de bâtiments et des scènes de rues urbaines :

$$
\text{Bâtiment} ::= \text{Base} + \text{Étage}^N + \text{Toit}, \quad N \sim \text{Uniform}(3, 30)
$$

En ajustant la répartition des paramètres (nombre d'étages, type de toit, matériau de façade), des groupes de bâtiments urbains avec différents styles peuvent être générés.

### 5.2 Évaluation de la qualité des actifs

La qualité des actifs synthétiques affecte directement l'efficacité de la migration Sim2Real. **Les dimensions de l'évaluation de la qualité** comprennent :| Dimensions | Indicateurs d'évaluation | Méthodes |
|------|---------|------|
| **Précision de la géométrie** | Vérité terrain RMSE vs LiDAR | Quantification après enregistrement du nuage de points |
| **Authenticité de la texture** | FID vs image réelle | Distance de départ de Fréchet |
| **Cohérence sémantique** | Précision de la segmentation | SegAcc sur Image Synthétique |
| ** Plausibilité physique ** | Distribution de la taille des objets | Comparaison avec les statistiques GT |

**SynthCity** (Griffiths & Boehm, 2023) fournit un ensemble de données synthétiques à grande échelle de 9 types d'actifs urbains, notamment des nuages ​​de points, des images et des annotations sémantiques, qui peuvent être utilisés comme référence pour la qualité des actifs simulés.

---

## 6. Évaluation de la qualité des données et cohérence multimodale

### 6.1 Mesure de l'authenticité

L'écart de distribution (Domain Gap) entre les données de simulation et les données réelles détermine la limite supérieure de l'effet de migration Sim2Real. Les méthodes d'évaluation quantitative comprennent :

**FID (Fréchet Inception Distance)** : extrayez les caractéristiques de l'image via Inception-v3 et calculez la distance de Fréchet entre la distribution des caractéristiques de l'image réelle $\mathcal{N}(\mu_r, \Sigma_r)$ et la distribution des caractéristiques de l'image simulée $\mathcal{N}(\mu_s, \Sigma_s)$ :

$$
\text{FID} = \|\mu_r - \mu_s\|^2 + \text{Tr}\left( \Sigma_r + \Sigma_s - 2\sqrt{\Sigma_r \Sigma_s} \right)
$$

Plus le FID est bas, plus l’image de simulation est proche de l’image réelle. Cible typique : FID $< 30$ (difficile à distinguer à l'œil nu).

**SSIM/PSNR** : similarité structurelle et rapport signal/bruit maximal, évaluation pixel par pixel de la qualité de l'image, adapté à la comparaison de la qualité de rendu de la même scène.

**Distance perceptuelle** : perte de perception basée sur la couche de fonctionnalités VGG/ResNet, qui est plus conforme à l'évaluation subjective de l'œil humain qu'aux indicateurs au niveau des pixels.

### 6.2 Contraintes de cohérence multimodaleLes données de simulation multimodale doivent respecter la contrainte de **cohérence intermodale** : l'image RVB, la carte de profondeur et le nuage de points LiDAR de la même scène doivent être cohérents les uns avec les autres, et il ne peut y avoir d'auto-contradiction telle que « la caméra voit le mur mais le LiDAR ne touche pas le mur ».

**Pipeline de vérification de cohérence** :

1. **Contrôle de cohérence géométrique** : Pour chaque point 3D, vérifiez que la profondeur de ses coordonnées projetées dans l'image RVB est cohérente avec la carte de profondeur/mesure LiDAR (erreur $< 1\%$)
2. **Contrôle de cohérence sémantique** : les résultats de la segmentation RVB et les résultats de la classification de l'intensité de réflexion LiDAR doivent être cohérents (par exemple, les garde-corps métalliques doivent être classés comme « obstacles durs » dans les deux modalités)
3. **Contrôle de cohérence temporelle** : le flux optique/le mouvement du nuage de points entre des images adjacentes doit être conforme au modèle de mouvement physique (hypothèse de vitesse/accélération uniforme)

Les données qui violent les contraintes de cohérence induiront en erreur l'apprentissage par fusion multimodale et doivent être automatiquement détectées et filtrées après la génération des données.

---

## 7. Boucle fermée planification-simulation : formation par apprentissage par renforcement

### 7.1 Formation par apprentissage par renforcement en simulation

L'apprentissage par renforcement (RL) fournit un paradigme d'apprentissage pour la planification de bout en bout des drones sans avoir besoin de conception manuelle des fonctions de coût. Pipeline de formation RL typique :

1. **Initialisation de l'environnement de simulation** : chargez le modèle 3D de la ville et générez des points de décollage et d'atterrissage aléatoires et des configurations d'obstacles
2. **Interaction stratégique** : la stratégie UAV $\pi_\theta(a_t | s_t)$ interagit avec l'environnement dans la simulation et collecte les données de trajectoire $\{s_t, a_t, r_t, s_{t+1}\}$
3. **Mise à jour de la politique** : utilisez l'algorithme PPO (Proximal Policy Optimization) ou SAC (Soft Actor-Critic) pour mettre à jour les paramètres de la politique.
4. ** Randomisation du domaine ** : randomisez la configuration des scénarios à chaque cycle de formation pour améliorer les capacités de généralisation de la stratégie
5. **Sim2Real Transfer** : Déployez la stratégie entraînée sur un drone réel, ce qui peut nécessiter une petite quantité de réglage fin des données réelles (Transfer RL)

**Conception de la fonction de récompense clé** :

$$
r_t = r_{\text{progress}} + r_{\text{sécurité}} + r_{\text{efficacité}} + r_{\text{confort}}
$$- $r_{\text{progress}} = \Delta d_{\text{goal}}$ : récompense positive pour la progression vers l'objectif
- $r_{\text{safety}} = -10$ si collision : pénalité de collision (grosse récompense négative)
- $r_{\text{efficiency}} = -0,01 \cdot T$ : pénalité de temps (encourage l'arrivée rapide)
- $r_{\text{confort}} = -0,1 \cdot \|\mathbf{a}_t\|$ : pénalité d'accélération (supprime les virages serrés)

### 7.2 Simulation d'une stratégie de migration réelle

Même avec la randomisation de domaine, des écarts entre la simulation et la réalité peuvent encore exister. Les stratégies suivantes peuvent améliorer les taux de réussite de la migration :

**Déploiement conservateur** :
- Effectuer d'abord une vérification de sécurité sur un vrai drone à basse vitesse et basse altitude
- Élargir progressivement le domaine de vol seulement après que la sécurité soit confirmée

**Alignement des fonctionnalités pertinentes pour les tâches** :
- Analyser la distribution des caractéristiques des données des capteurs (statistiques de profondeur, densité des bords) de drones réels
- Ajuster les paramètres de simulation pour correspondre à la distribution des fonctionnalités clés

**Méta-apprentissage** :
- Utiliser MAML (Model-Agnostic Meta-Learning) pour entraîner la stratégie afin de s'adapter rapidement à une petite quantité de données réelles
- Entraîner la politique de base $\pi_0$ en simulation et l'affiner à $\pi^*$ dans l'environnement réel

### 7.3 Cas virtuel-réel en boucle fermée : vol agressif

Les projets de course de drones autonomes dans **AlphaPilot** (sponsorisé par Lockheed Martin) et **SUAS Competition** démontrent une boucle fermée de simulation-entraînement-déploiement mature :1. Utilisez DOMAIN_RANDOMIZE dans Flightmare/AirSim pour configurer l'éclairage aléatoire, les perturbations du vent et l'emplacement des obstacles
2. Utilisez PPO pour entraîner la stratégie de bout en bout (sortie directement de la vitesse du moteur), et les récompenses incluent le temps au tour, la pénalité de collision et le confort
3. La stratégie d'entraînement atteint une vitesse de déplacement de $> 15\text{m/s}$ en simulation
4. Déployez sur un drone réel et utilisez l'adaptation en ligne pour compenser les lacunes résiduelles de Sim2Real.
5. Compétences clés : **Bouclier de sécurité** - Combinant les résultats de la politique RL avec l'évitement d'obstacles d'urgence basé sur une planification géométrique, la politique n'est responsable que de la prise de décision de haut niveau.

---

## 8. Orientations futures et exploration des frontières

### 8.1 Neural Simulator : moteur physique apprenable

Les simulateurs traditionnels s'appuient sur des modèles physiques conçus manuellement et sont difficiles à capturer des interactions complexes (interaction fluide-structure, déformation flexible du corps). **Learned Physics Engine** (Learned Physics Engine) apprend les lois physiques à partir de données via des réseaux de neurones :

**Graph Network Simulator (GNS)** (Sanchez-Gonzalez et al., ICML 2020) utilise des réseaux de neurones graphiques pour modéliser les interactions des systèmes de particules et peut apprendre les règles d'évolution des systèmes fluides, rigides et multi-corps. Si le GNS est étendu à la modélisation aérodynamique, il est possible de réaliser une **simulation de la dynamique de vol des drones basée sur les données**.

### 8.2 Données à l'échelle d'Internet + IA générative

Le modèle LLM (Large Language Model) et le modèle de diffusion offrent de nouvelles possibilités pour la génération de données de simulation :

- **LLM génère une description de la scène** : saisissez « intersection de pointe du soir du CBD de Pékin, 5 voitures, 10 piétons », GPT-4V peut générer une configuration de scène détaillée (emplacement, vitesse, comportement)
- **Texture de génération de modèle de diffusion** : utilisez ControlNet / Stable Diffusion pour générer automatiquement des textures réalistes basées sur des dessins au trait architecturaux, réduisant ainsi la modélisation manuelle
- **Clonage de scène NeRF** : prenez une vidéo de ville de 5 minutes avec votre téléphone portable et reconstruisez-la automatiquement en une scène NeRF navigable, qui peut être utilisée directement comme environnement de simulation

### 8.3 Simulation fédérée : cartographie collaborative distribuéeÀ l'avenir, les clusters de drones urbains pourraient former un **réseau de simulation fédéré** : chaque drone collecte des données en vol et met à jour un jumeau numérique de ville partagé, et d'autres drones téléchargent le dernier jumeau et s'entraînent dans l'environnement de simulation mis à jour. Cela protège non seulement la confidentialité des données (l’image originale ne quitte pas la zone locale), mais permet également une accumulation distribuée de connaissances.

---

## 9. Résumé

La synthèse des données de simulation multimodale constitue la base technique clé permettant aux algorithmes de planification des drones urbains à basse altitude de passer de la recherche à la mise en œuvre. Grâce à la simulation de capteurs haute fidélité (RVB, LiDAR, ondes millimétriques, imagerie thermique), à ​​la génération programmatique de divers actifs de scène et à une stratégie stricte de randomisation de domaine, des ensembles de données de formation à grande échelle peuvent être systématiquement construits dans l'environnement de simulation.

Le principal défi de la migration Sim2Real est le **écart de perception** et le **écart dynamique**. L'écart de perception peut être atténué grâce au rendu neuronal (UniSim) et à l'évaluation de la cohérence perceptuelle ; l’écart dynamique peut être compensé par l’adaptation en ligne et le méta-apprentissage.

À mesure que les simulateurs neuronaux, les moteurs physiques apprenables et les technologies d’IA générative mûriront, la future synthèse des données de simulation sera plus automatisée, plus fidèle et moins coûteuse. La vision de la **Simulation comme vérité terrain** devient progressivement possible.

---

## Références

- Shah, S., Dey, D., Lovett, C. et Kapoor, A. (2018). AirSim : Simulation visuelle et physique haute fidélité pour véhicules autonomes. *Robotique de terrain et de service*. https://doi.org/10.1007/978-3-319-67361-5_40

- Zhou, Y., et al. (2023). UniSim : Un simulateur de capteur neuronal en boucle fermée. *CVPR* (ou arxiv:2308.01812, lieu à confirmer). https://doi.org/10.1109/CVPR52729.2023.00571- Kar, A., et al. (2019). Meta-sim : Apprendre à générer des ensembles de données synthétiques. *ICCV*. https://doi.org/10.1109/ICCV.2019.00393

- Sánchez-Gonzalez, A., et al. (2020). Apprendre à simuler une physique complexe avec des réseaux de graphes. *ICML*. https://doi.org/10.5555/3524938.3525750

-Zhang, J., et al. (2021). SimBot : activer des robots autonomes avec des modèles de langage de vision via des simulateurs robotiques. *CoRL*.

- Du, Y., et al. (2023). Apprentissage des politiques à partir de la simulation avec randomisation de domaine contradictoire. *ICRA*. https://doi.org/10.1109/ICRA57147.2024.10610923

-Antonini, A., et al. (2020). L'hiver arrive : apprendre à naviguer en toute sécurité dans des environnements invisibles. *ICRA*. https://doi.org/10.1109/ICRA40945.2020.9196643

- Song, Y., et al. (2023). Diffusion-LM : Génération de texte contrôlable via des modèles de diffusion. *NeurIPS*.- Griffith, S. et Boehm, J. (2023). SynthCity : Un nuage de points synthétiques à grande échelle pour les scènes urbaines. *Journal ISPRS de photogrammétrie et de télédétection*. https://doi.org/10.1016/j.isprsjprs.2023.04.015

- Lois, C., et coll. (2020). Flightmare : Un simulateur quadrotor flexible avec une perception modulaire. *IROS*.

---

*Cet article est le cinquième chapitre étendu d'une série d'articles sur la planification d'itinéraires urbains à basse altitude pour les drones. Série complète 🎉*