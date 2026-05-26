---
title: "Planification d'itinéraires urbains de drones à basse altitude : modélisation spatiale tridimensionnelle"
description: "Examiner systématiquement les méthodes de modélisation spatiale tridimensionnelle dans la planification d'itinéraires urbains de drones à basse altitude, couvrant la grille d'occupation 3D, l'effet canyon urbain et le modèle en couches de l'espace aérien."
tags: ['UAV', '路径规划', '城市空域']
category: Tech
pubDate: 2026-04-08T14:54:00+08:00
sourceHash: "5207e2b900596685eafb34524553f682fb5b6948"
---

##Présentation

La planification d'itinéraires urbains de drones à basse altitude est l'une des technologies de base de base pour parvenir à un transport aérien urbain sûr et efficace (UAM, Urban Air Mobility). Différent des zones ouvertes suburbaines, l'environnement urbain présente des caractéristiques distinctives telles qu'une structure géométrique tridimensionnelle complexe, une atténuation sévère des signaux GNSS et une forte perturbation du champ d'écoulement par les bâtiments, ce qui impose des exigences plus élevées aux méthodes de modélisation spatiale. Cet article se concentre sur la première partie de la série de planification d'itinéraires urbains à basse altitude pour les drones : la modélisation spatiale tridimensionnelle. Il aborde en profondeur la grille d'occupation tridimensionnelle (3D Occupancy Grid) et la représentation octree (Octree), la modélisation physique de l'effet canyon urbain (Urban Canyon) et le modèle hiérarchique de l'espace aérien qui s'appuie sur le contrôle aéronautique traditionnel, complété par une analyse comparative de la mise en œuvre de l'ingénierie.

## 1. Grille d'occupation tridimensionnelle et représentation octree

### 1.1 De deux dimensions à trois dimensions : définition mathématique

La grille d'occupation classique (Occupancy Grid) a été proposée par Moravec et Elfes (1985). Son idée principale est de discrétiser l'espace continu en grilles finies et de coder l'état d'occupation de chaque grille avec des valeurs de probabilité. Dans le cas bidimensionnel, l'espace est divisé en cellules carrées de longueur latérale $\Delta$, et la probabilité d'occupation de chaque cellule $m_i$ est enregistrée comme $P(m_i | Z_{1:t})$, où $Z_{1:t}$ représente toutes les observations de capteurs jusqu'au temps $t$. Les mises à jour des capteurs suivent la formule récursive bayésienne :

$$
P(m_i | Z_t, Z_{1:t-1}) = \frac{P(Z_t | m_i, Z_{1:t-1}) P(m_i | Z_{1:t-1})}{P(Z_t | Z_{1:t-1})}
$$

Dans l'ingénierie réelle, afin d'éviter le sous-débordement numérique et de simplifier les calculs, les **cotes logarithmiques (log-odds)** sont généralement exprimées :

$$
l(m_i) = \log \frac{P(m_i)}{1 - P(m_i)}
$$

Après chaque mesure du capteur, la règle de mise à jour additive est :

$$
l(m_i)_{\text{nouveau}} = l(m_i)_{\text{old}} + \Delta l
$$

où $\Delta l$ est déterminé par le modèle du capteur (positif lorsqu'il est occupé, négatif lorsqu'il est inactif). Cette méthode convertit la multiplication en addition, améliorant considérablement les performances en temps réel.La grille d'occupation tridimensionnelle étend la définition ci-dessus d'un plan à un espace volumétrique $\mathbb{R}^3$, divisant l'espace en unités cubiques (voxels) de longueur de bord $\Delta$. Supposons que $V_i \subset \mathbb{R}^3$ représente le $i$ème voxel, alors sa probabilité d'occupation est $P(v_i | Z_{1:t})$. La complexité de stockage direct des rasters tridimensionnels est de $O(N^3)$ ($N$ est le nombre de rasters unidimensionnels), ce qui est inacceptable dans les scénarios urbains typiques - par exemple, le nombre total de voxels couvrant une zone urbaine de ​​$1\,\text{km}^3$ à une résolution de $0,1\,\text{m}$ atteint 10$^{13}$ au total.

### 1.2 Octree : Indice spatial pour une résolution adaptative

**Octree** est la solution standard pour relever les défis de stockage ci-dessus. La bibliothèque OctoMap proposée par Hornung et al. (2013) constitue une mise en œuvre marquante de cette méthode dans le domaine de la robotique. La logique de division spatiale de l'octree est la suivante : le nœud racine couvre tout l'espace tridimensionnel, et chaque nœud interne est subdivisé de manière récursive en 8 sous-nœuds de volume égal (correspondant à la division $2 \times 2 \times 2$ de l'espace tridimensionnel) jusqu'à ce que la profondeur maximale prédéfinie $d_{\max}$ ou la taille minimale du voxel $\Delta_{\min}$ soit atteinte.

Supposons que la longueur du côté du nœud racine est $L_0$ et que la longueur du côté du voxel en profondeur $d$ est :

$$
L_d = \frac{L_0}{2^d}
$$

Le nombre maximum de nœuds pour la profondeur $d$ est de $8^d$, mais comme l'octree ne divise que l'espace occupé ou observé, les zones inconnues/libres peuvent être représentées par un seul nœud, de sorte que le nombre réel de nœuds est beaucoup plus petit que le raster complet. Le modèle de stockage d'OctoMap utilise en outre **Probabilistic OcTree** : chaque nœud stocke une valeur de probabilité d'occupation $P(n)$, qui est continuellement modifiée via une mise à jour bayésienne. La probabilité d'un nœud inactif est $P_{\text{occ}}$, la probabilité d'un nœud occupé est proche de $1$ et le nœud correspondant dans la zone inconnue n'existe pas dans l'arbre (codage implicite).

Les expériences de Hornung et al. (2013) montrent que dans un environnement intérieur typique, la consommation de mémoire d'OctoMap est d'environ **1/50** de rasters tridimensionnels denses de même résolution, tout en prenant en charge les mises à jour dynamiques et les requêtes de résolution arbitraire.

### 1.3 Octree et perception multi-granularitéZeng et coll. (2020) ont proposé un algorithme de perception de l'environnement multi-granularité basé sur des grilles d'occupation octree sur des outils et applications multimédias, soulignant que bien que le modèle de nuage de points soit riche en informations, il existe une grande redondance dans la planification des chemins. Ils utilisent des octrees pour fournir une représentation probabiliste unifiée des données provenant de différents capteurs (RVB-D, LiDAR, etc.), en conservant les informations géométriques à haute résolution au niveau du nœud feuille et en fournissant une perception de la structure globale à basse résolution au niveau du nœud grossier. Cette idée est particulièrement importante pour la construction de cartes urbaines à grande échelle : un évitement d'obstacles au niveau centimétrique est nécessaire à courte distance, et une prise de décision macro-trajet au niveau de 100 mètres est nécessaire à distance.

Thomas et coll. (2021) ont en outre proposé des **Cartes de grille d'occupation spatio-temporelle (SOGM)** dans l'article arXiv (arXiv : 2108.10585), qui intègrent la prédiction temporelle des obstacles dynamiques dans la représentation de la grille, fournissant des capacités efficaces de prédiction d'occupation pour les personnes et les véhicules se déplaçant dans l'environnement urbain, et sont d'une grande valeur pour la planification d'évitement d'obstacles en temps réel.

## 2. Effet Canyon urbain : modélisation physique et défis de navigation

Urban Canyon fait référence à un micro-terrain urbain avec des bâtiments denses et des rues étroites. Il s’agit de l’un des environnements opérationnels les plus difficiles pour les drones à basse altitude. Ses effets physiques peuvent être compris en trois dimensions.

### 2.1 Atténuation du signal GNSS et effet multitrajet

Dans les canyons urbains, les immeubles de grande hauteur denses forment une structure de « canyon », et les signaux des satellites GNSS sont confrontés à deux types d'interférences graves :

- **Propagation sans visibilité directe (NLOS)** : Le signal direct est bloqué par le bâtiment, et le drone ne peut recevoir que le signal réfléchi ou diffracté par le mur, ce qui fait que la valeur de mesure de pseudo-portée est systématiquement plus grande ;
- **Multitrajet** : La superposition de signaux de plusieurs chemins de réflexion provoque des erreurs de solution de phase porteuse et une gigue de positionnement.

L'ensemble de données UrbanNav (Wen et al., 2021 ; GitHub : IPNL-POLYU/UrbanNavDataset) a mesuré les performances de positionnement de capteurs à faible coût dans les canyons urbains de Tokyo et de Hong Kong. Les résultats ont montré que dans les zones de canyons profonds, l’erreur de positionnement d’un seul point (SPP) peut atteindre des dizaines de mètres. Même si un récepteur GNSS bi-fréquence est utilisé, sans détection et élimination NLOS, la précision du positionnement horizontal sera toujours difficile à répondre aux exigences inférieures au mètre pour la précision du vol stationnaire des drones. Le rapport hauteur/largeur (AR = hauteur du bâtiment / largeur de la rue) des canyons urbains est le facteur dominant affectant la précision du GNSS : plus l'AR est grand, plus la disponibilité du signal est faible.

### 2.2 Turbulences et perturbations du champ de ventLa dynamique des fluides au sein des canyons urbains présente un degré élevé d'hétérogénéité. L'étude classique de Rotach (1995) dans *Boundary-Layer Meteorology* a quantifié le profil statistique de la turbulence à l'intérieur des canyons, notant que l'énergie cinétique turbulente (TKE) dans les canyons de rue est **2 à 5 fois** plus élevée que dans les banlieues ouvertes, et que l'écart type de la composante de vitesse verticale $\sigma_w$ peut atteindre 0,3$ à 0,6$ fois la vitesse moyenne du vent près de la surface. Les principaux mécanismes physiques comprennent :

- **Sillage du bâtiment** : le flux d'air forme des vortex périodiques (rue des vortex de Kármán) du côté sous le vent après avoir contourné le bâtiment, générant une portance instable et des forces latérales importantes ;
- **Circulation du canyon (Circulation du canyon dans la rue)** : Lorsque le flux entrant est orthogonal à l'axe du canyon, une structure à double anneau vortex avec des directions opposées se forme à l'intérieur de la rue, et la composante verticale nette de la vitesse du vent est considérablement amplifiée dans cette zone ;
- **Sous-plage inertielle** : Le spectre énergétique de l'énergie de turbulence dans la sous-plage inertielle suit la règle $-5/3$ (loi de Kolmogorov). Les turbulences à petite échelle constituent une perturbation continue de la bande passante de contrôle d'attitude des drones.

Pour la conception du contrôle des drones, la plage de fréquences caractéristique de l’intensité des turbulences est cruciale. La perturbation dans la bande de fréquence $1$–$10\,\text{Hz}$ est la plus importante dans les canyons urbains, ce qui nécessite que la bande passante de la boucle d'attitude du système de commandes de vol ne soit pas inférieure à 20$\,\text{Hz}$, ce qui n'est pas facile à mettre en œuvre sur une plateforme embarquée.

### 2.3 Effet d'accélération du vent de Bernoulli

Dans les rues étroites, l’effet Bernoulli ne peut être ignoré. Lorsque le flux d'air est forcé à travers un canal de section transversale réduite, la vitesse du vent augmente considérablement dans les zones locales selon l'équation de continuité $A_1 ​​​​​​v_1 = A_2 v_2$. La vitesse du vent aux points les plus étroits entre les bâtiments dans les canyons urbains peut être **1,5 à 3 fois** plus élevée que dans les zones ouvertes. De plus, « l'effet Venturi » entre les façades des bâtiments produira localement une succion vers le centre de la rue, menaçant la stabilité latérale du drone.

Dans la planification pratique, il est recommandé de modéliser la **perturbation éolienne équivalente** dans les canyons urbains sous forme de vent moyen $\bar{u}$ superposé à des composantes de turbulence aléatoires $\tilde{u}$ :

$$
u_{\text{eff}}(t) = \bar{u} + \sigma_u \cdot \xi(t)
$$

où $\xi(t)$ est un bruit blanc gaussien obéissant à la distribution normale standard, et $\sigma_u$ est déterminé à partir d'une formule empirique basée sur le rapport d'aspect du canyon et la géométrie de la rue locale.## 3. Modèle en couches de l'espace aérien

### 3.1 Lumières issues du contrôle aéronautique traditionnel

Le système de contrôle du trafic aérien civil traditionnel a adopté la gestion des couches d'altitude (Altitude Layer) depuis des décennies : avec 1 000 $\,\text{ft}$ (environ 300 $\,\text{m}$) comme intervalle d'altitude de base, l'espace aérien inférieur à 29 000 $\,\text{ft}$ est divisé en plusieurs secteurs de contrôle, chaque couche desservant des avions de différents types et vitesses. Dans le contexte de l'UAM, les drones urbains à basse altitude doivent coexister avec les piétons au sol, les bâtiments, les aires d'atterrissage d'hélicoptères et les avions généraux traditionnels dans une plage verticale de **0$ à 120$\,\text{m}$** (environ 0$ à 400$\,\text{ft}$). La conception en couches devient donc inévitable.

Les projets de recherche UTM (UAS Traffic Management) de la NASA (2016-2024) et UAM ConOps V2.0 de la FAA (2023) ont tous deux souligné que la gestion hiérarchique est le principal moyen d'éviter les conflits de drones à grande échelle. En s’appuyant sur cette idée dans des scénarios urbains, le schéma suivant à trois niveaux peut être conçu.

### 3.2 Schéma de division des couches de hauteur des scènes urbaines

| Niveau d'altitude | Plage verticale | Fonctions principales | Type d'avion | Vitesse typique |
|--------|----------|----------|---------------|----------|
| **Étage G** | Sol $\sim 30\,\text{m}$ | Livraison express sur les trottoirs, livraison robotisée | Micro multi-rotor | $0$–$5\,\text{m/s}$ |
| **Niveau L** | 30$–80$\,\text{m}$ | Logistique communautaire, photographie aérienne urbaine, navette basse | Petite aile multi-rotors/composite | 5$–15$\,\text{m/s}$ |
| **Niveau U** | 80$ – 120$\,\text{m}$ | Intercity express, intervention d'urgence, navette en hauteur | Moyen eVTOL/voile fixe | 15$–30$\,\text{m/s}$ |

> Remarque : les limites d'altitude spécifiques doivent être ajustées en fonction des réglementations locales de gestion de l'espace aérien (la Chine s'appuie sur les « Règlements provisoires sur la gestion des vols sans pilote » 2023) et de l'urbanisme.

Les principes de conception de ce schéma en couches sont les suivants :1. **Isolement fonctionnel** : la couche G se concentre sur la sécurité de la distribution des terminaux (pour éviter les conflits directs avec les personnes), la couche L est la couche d'application grand public urbaine et la couche U est proche de la hauteur de l'aviation générale traditionnelle pour être compatible avec la transition ;
2. **Séparation des flux** : les directions amont et aval sont en outre séparées horizontalement à la même altitude, et la boucle de route unidirectionnelle est conçue en référence à la logique d'approche à cinq côtés du contrôle de la circulation aérienne ;
3. **Ajustement dynamique** : la limite en couches peut être traduite dynamiquement en fonction de la densité du trafic en temps réel, et le cadre xTM (extensible Traffic Management) de la FAA a fourni une interface standardisée pour cela.

### 3.3 Fusion de cartes raster en couches et tridimensionnelles

Le modèle en couches de hauteur doit être profondément intégré à la grille d'occupation tridimensionnelle : lors de la phase de planification, le **masquage des couches** est effectué sur la carte octree en fonction des limites des couches, et les chemins ne sont recherchés que dans les voxels pilotables de la couche où se trouve la tâche actuelle et des couches adjacentes ; lors de la replanification dynamique, s'il y a une congestion sur une certaine couche, elle peut être automatiquement basculée vers la couche adjacente pour les détours. Ce mécanisme a été initialement vérifié dans le concept de corridor UTM (Corridor) de la NASA.

## 4. Compromis d'ingénierie pour les nuages de points/voxels Octree/PCL

Dans la pratique de l'ingénierie, le choix d'une méthode de représentation tridimensionnelle nécessite un compromis entre précision, mémoire, vitesse de calcul et fréquence de mise à jour. Ce qui suit est une comparaison systématique.| Métriques | Trame 3D dense | Octarbre | Nuage de points bruts (PCL) | Voxel de hachage |
|------|------------|----------------|--------------|----------------------|
| **Efficacité de la mémoire** | Faible (fixe $O(N^3)$) | Élevé (fractionnement adaptatif) | Medium (uniquement les points enregistrés, pas de topologie) | Élevé (indice de hachage clairsemé) |
| **Complexité des requêtes** | $O(1)$ | $O(\log N)$ | $O(N)$ (exhaustif) ou $O(\log N)$ (avec kd-tree) | $O(1)$ signifie |
| **Mise à jour dynamique** | Lent (reconstruction complète) | Rapide (séparation incrémentielle des nœuds) | Rapide (ajouter des points) | Rapide (insertion de hachage) |
| **Cohérence de la résolution** | Cohérence mondiale | Adaptation hiérarchique | Aucune structure de grille | Cohérence mondiale |
| **Détection de collision** | Rapide (index du tableau) | Moyen (recherche arborescente) | Lent (détection de modèle ponctuel) | Rapide (recherche de hachage) |
| **Écologie de l'ingénierie** | ROS nav_msgs | OctoMap / PCL Octree | PCL/Open3D | OctoMap (configurable) |
| **Scénarios applicables** | Petite portée et haute précision | Large gamme et résolutions multiples | Détection/cartographie en temps réel | Scènes clairsemées à grande échelle |

**Le principal avantage d'Octree** réside dans sa double caractéristique de **résolution adaptative + représentation probabiliste** : il s'agit à la fois d'une structure d'index spatial et d'un cadre de mise à jour probabiliste, particulièrement adapté aux besoins de perception des « obstacles précis à proximité et des obstacles rugueux au loin » dans les scènes urbaines. La bibliothèque OctoMap (Hornung et al., 2013 ; DOI : 10.1007/s10514-012-9321-0) témoigne de sa maturité en ingénierie, tant en termes d'activité sur GitHub que de nombre de citations académiques (plus de 5 000 fois selon Google Scholar).

**L'avantage du nuage de points** est qu'il préserve sans perte les données originales du capteur et convient aux algorithmes de perception basés sur le deep learning (détection de cibles 3D, segmentation sémantique, etc.) en entrée. La bibliothèque PCL (Point Cloud Library) et la bibliothèque Open3D fournissent une chaîne d'outils de traitement de nuages ​​de points mature, mais le nuage de points lui-même n'encode pas les informations sémantiques occupées/inactives et nécessite des étapes supplémentaires pour être converti en une zone pilotable.Les **voxels de hachage** (tels que le schéma d'index de hachage « OcTree Key » d'OctoMap) fonctionnent bien dans les scénarios qui nécessitent une vitesse de requête extrêmement rapide et des scènes clairsemées. La surcharge mémoire est proche de celle d'un octree mais la requête est plus efficace. C’est un sujet brûlant dans la recherche de pointe ces dernières années.

Dans les scénarios urbains réels, la **solution recommandée** utilise l'octree probabiliste d'OctoMap comme stockage sous-jacent, utilise le nuage de points d'origine comme entrée de détection, corrige en permanence la probabilité d'occupation via le mécanisme de mise à jour incrémentielle et utilise un index de hachage pour accélérer les requêtes du voisin le plus proche. Cette combinaison a été éprouvée dans les systèmes SLAM avancés tels que LIO-SAM pour obtenir une cartographie robuste en temps réel dans les canyons urbains (voir la version adaptée de LIO-SAM-6AXIS-UrbanNav).

## 5. Résumé et perspectives

Cet article trie systématiquement les éléments essentiels de la modélisation spatiale tridimensionnelle dans la planification d'itinéraires urbains de drones à basse altitude :

- **La grille d'occupation tridimensionnelle et l'octree** fournissent un cadre de représentation de l'environnement unifié basé sur la théorie des probabilités. En tant qu'implémentation open source, OctoMap a été largement vérifiée dans le monde universitaire et industriel ;
- **L'effet Urban Canyon** impose des contraintes au système de planification des drones à partir des trois dimensions physiques de l'atténuation GNSS, des statistiques de turbulence et de l'accélération du vent de Bernoulli, et doit être explicitement modélisé dans la planification d'itinéraire ;
- Le **modèle en couches de l'espace aérien** s'appuie sur les idées traditionnelles de contrôle de l'aviation et divise l'espace aérien vertical de 0 $ à 120 $\,\text{m}$ en trois couches : G/L/U dans les scénarios urbains, fournissant un cadre structurel pour la gestion du trafic de drones à grande échelle ;
- La sélection des projets doit faire un compromis global entre l'efficacité de la mémoire, la vitesse des requêtes et les capacités de mise à jour dynamique. La combinaison d'OctoMap + nuage de points est la voie technologique dominante actuelle.

Les chapitres suivants aborderont progressivement des sujets tels que **l'algorithme de planification de chemin** (l'application d'algorithmes d'échantillonnage tels que RRT*/BIT* dans des cartes octree tridimensionnelles), **l'optimisation de trajectoire en temps réel** (modèle de contrôle prédictif sous perturbation du vent dans les canyons urbains) et **l'évitement collaboratif d'obstacles multi-avions** pour construire un système technologique complet de planification d'itinéraires urbains à basse altitude.

---

## Références- Hornung, A., Wurm, KM, Bennewitz, M., Stachniss, C. et Burgard, W. (2013). OctoMap : Un cadre de cartographie 3D probabiliste efficace basé sur les octrees. *Robots autonomes*, 34(3), 189-206. https://doi.org/10.1007/s10514-012-9321-0

- Thomas, H., Farr, R., Yang, C., Chen, Y. et Leonard, JJ (2021). Apprentissage de cartes de grille d'occupation spatio-temporelle pour une navigation tout au long de la vie dans des scènes dynamiques (arXiv : 2108.10585). arXiv. https://arxiv.org/abs/2108.10585

- Wen, W., Zhang, G. et Hsu, LT (2021). *UrbanNav : un ensemble de données de localisation open source pour l'analyse comparative des algorithmes de positionnement conçus pour les canyons urbains* [Ensemble de données et documentation]. Dépôt GitHub : https://github.com/IPNL-POLYU/UrbanNavDataset

- Rotach, M.W. (1995). Profils de statistiques de turbulence dans et au-dessus d'un canyon de rue urbain. *Environnement atmosphérique*, 29(13), 1473-1486. https://doi.org/10.1016/1352-2310(95)00084-D- Zeng, T., Si, B. et Zhao, J. (2020). Perception de l'environnement multi-granularité basée sur une grille d'occupation octree. *Outils et applications multimédias*, 79, 27875-27896. https://doi.org/10.1007/s11042-020-09302-w

- Moravec, HP et Elfes, A. (1985). Cartes haute résolution provenant d'un sonar grand angle. *Actes de la Conférence internationale de l'IEEE sur la robotique et l'automatisation (ICRA)*, 116-121. https://doi.org/10.1109/ROBOT.1985.1087316

- Département américain des transports / Federal Aviation Administration. (2023). *Concept d'opérations de mobilité aérienne urbaine (UAM)*, version 2.0. FAA. https://www.faa.gov/air_traffic/nas_management/nas_research/models/uam_conops

- Direction des missions de recherche aéronautique de la NASA. (2023). *Résumé du projet de gestion du trafic UAS (UTM)*. NASA. https://utm.arc.nasa.gov/- Hrabar, S. et Sukhatme, GS (2004). Une comparaison de deux configurations de caméras pour la navigation basée sur le flux optique dans les canyons urbains. *Actes de la Conférence internationale IEEE/RSJ sur les robots et systèmes intelligents (IROS)*, 3943-3948. https://doi.org/10.1109/IROS.2004.1389989