---
title: "Planification d'itinéraires urbains de drones à basse altitude : cartographie sémantique et division des zones fonctionnelles"
description: "Examiner les progrès de la recherche sur la cartographie sémantique et la perception des zones fonctionnelles dans la planification d'itinéraires urbains de drones, couvrant les derniers travaux du CVPR/ICCV/IROS/RAL 2022-2025."
tags: ["drone", "Cartographie sémantique", "Division des domaines fonctionnels", "planification du chemin", "gestion de l'espace aérien"]
category: "Tech"
pubDate: 2026-04-09
---

# Planification d'itinéraires urbains de drones à basse altitude : cartographie sémantique et division des zones fonctionnelles

> **Quatrième direction : cartographie sémantique + sensibilisation au ruban**
> Chapitre étendu · Série de blogs techniques, partie 4

---

## 1. Contexte : De la carte géométrique à la carte sémantique

La planification traditionnelle des trajectoires des drones repose sur une représentation géométrique pure de l'environnement - grille d'occupation (Occupancy Grid), octree (Octree) ou carte de voxels (Voxel Map). Ces représentations codent uniquement « si l'espace est pilotable » et ne peuvent pas comprendre « où voler » et « pourquoi il ne peut pas voler ».

Les cartes sémantiques introduisent des capacités de **compréhension de scène** basées sur une représentation géométrique : identification d'informations sémantiques telles que les types de bâtiments (résidentiels/commerciaux/industriels), le niveau des routes, la densité de la foule, les limites des zones fonctionnelles, etc. Cette capacité est essentielle pour la planification urbaine à basse altitude : un drone traversant une place d'un quartier d'affaires présente un niveau de risque complètement différent de celui traversant une cour de récréation d'école, mais une carte purement géométrique traiterait les deux comme un espace libre équivalent.

De plus, le zonage fonctionnel divise l'espace aérien urbain à basse altitude en zones avec différents niveaux de réglementation : **Contrôle de la hauteur réelle à 120 m**, zone d'exclusion aérienne, zone réglementée, zone contrôlée, etc. La conscience sémantique permet aux drones de comprendre et de se conformer de manière proactive à ces règles réglementaires, plutôt que de s'appuyer uniquement sur des cartes statiques pré-annotées des zones d'exclusion aérienne.

---

## 2. Bases de la cartographie sémantique : perception → compréhension

### 2.1 Segmentation sémantique : des pixels à la compréhension de la scène

La segmentation sémantique est la base perceptuelle fondamentale de la cartographie sémantique. Étant donné une image $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$, le modèle de segmentation sémantique génère des étiquettes de classe par pixel :

$$
\hat{y}_{i,j} = \arg\max_{c \in \mathcal{C}} P(c | \mathbf{I}, \mathbf{p}_{i,j})
$$

Parmi eux, $\mathcal{C}$ est un ensemble de catégories sémantiques (telles que les bâtiments, les routes, la végétation, les véhicules, les personnes, le ciel), et $\mathbf{p}_{i,j}$ est le codage de position du pixel $(i,j)$.

**Les architectures de segmentation sémantique grand public pour les scènes urbaines** incluent :- **DeepLabv3+** (Chen et al., CVPR 2018) : utilisez Atrous Convolution pour étendre le champ de réception sans perdre en résolution, capturant efficacement des structures à grande échelle telles que des bâtiments urbains et des routes.
- **MaskFormer** (Cheng et al., CVPR 2022) : unifie la segmentation sémantique en tant que problème de classification de masques, prend en charge n'importe quel nombre de catégories sémantiques et n'a pas besoin de prédéfinir un $\mathcal{C}$ fixe.
- **Segment Anything Model (SAM)** (Kirillov et al., ICCV 2023) : Un modèle de base de segmentation universelle proposé par Meta, qui prend en charge la segmentation zéro des invites de point/boîte/texte, fournissant un nouveau paradigme pour la cartographie sémantique à vocabulaire ouvert des scènes urbaines.

### 2.2 Segmentation d'instance et détection de cible

En plus de la segmentation sémantique, la **segmentation d'instance** distingue davantage les différents individus d'objets similaires, en séparant chaque piéton du « groupe de piétons » en une instance indépendante, fournissant ainsi une prise en charge granulaire pour la prédiction des intentions et l'évitement des collisions.

| Méthodes | Idées fondamentales | Vitesse de raisonnement | Travail représentatif |
|------|---------|---------|---------|
| **En deux étapes** | Détectez d'abord les boîtes, puis segmentez les masques | ~10 FPS | Masque R-CNN (ICCV 2017) |
| **Une étape** | Prédire conjointement les masques et les catégories | ~25 FPS | YOLACT (ICCV 2019) |
| **Basé sur un transformateur** | Détection de style DETR + masque | ~15 FPS | Mask2Former (CVPR 2022) |
| **Modèle de base** | SAM + Détecteur | ~20 FPS | SEEM (CVPR 2024) |

La **série YOLO** (Ultralytics YOLOv8, 2023) est largement utilisée dans la perception sémantique en temps réel des drones : elle peut atteindre une fréquence d'images de détection de plus de 50 FPS sur Jetson Orin, avec une latence de $< 20\text{ms}$, ce qui convient aux exigences de perception en temps réel des systèmes de contrôle de vol.

### 2.3 Estimation de la profondeur : géométrie 2D → 3DLa cartographie sémantique nécessite de déplacer les étiquettes sémantiques 2D dans l'espace 3D. **L'estimation de la profondeur monoculaire** offre des capacités de conversion d'images RVB en cartes de profondeur denses :

$$
\hat{D} = \mathcal{D}_\phi(\mathbf{I}), \quad D : \text{pixel} \rightarrow \mathbb{R}^+
$$

Les méthodes clés comprennent :

- **MiDaS** (Ranftl et al., NeurIPS 2020) : utilise un entraînement multi-ensembles de données (profondeur mixte supervisée + non supervisée), fonctionne bien en généralisation à échantillon nul et est actuellement le modèle de base le plus largement utilisé pour l'estimation de la profondeur monoculaire.
- **Depth-Anything** (Yang et al., arxiv 2024) : optimisation d'image à grande échelle sans annotation basée sur MiDaS pour obtenir une plus grande précision de profondeur dans les scènes urbaines
- **DPT** (Ranftl et al., ICCV 2021) : architecture de transformateur basée sur ViT, produit directement des cartes de profondeur haute résolution

Combinés avec les paramètres intrinsèques de la caméra $(f_x, f_y, c_x, c_y)$, les coordonnées des pixels 2D $(u, v)$ et la profondeur $D(u, v)$ peuvent être rétroprojetées en points 3D :

$$
\mathbf{X} = D(u,v) \cdot \mathbf{K}^{-1} \cdot [u, v, 1]^\top, \quad \mathbf{K} = \begin{pmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{pmatrix}
$$

---

## 3. Division des zones fonctionnelles urbaines et classification de l'espace aérien à basse altitude

### 3.1 Différences de contraintes de vol dans les zones fonctionnelles urbaines

L'espace urbain est divisé en différentes zones fonctionnelles selon la nature de l'utilisation, et le degré de restrictions sur le vol des drones dans chaque zone varie considérablement :| Domaine fonctionnel | Scénarios typiques | Contraintes de vol | Principaux risques |
|--------|---------|---------|---------|
| **Zone résidentielle** | Quartier résidentiel | Restrictions de hauteur (< 30 m), restrictions de période | Atteinte à la vie privée et plaintes concernant le bruit |
| **Quartier des affaires** | CBD, centres commerciaux | Voler à portée visuelle | Foules denses, interférences de signal |
| **Zone industrielle** | Usines, entrepôts | Zones d'exclusion aérienne possibles | Interférences électromagnétiques, véhicules lourds |
| **École/Hôpital** | Ecoles primaires et secondaires, hôpitaux | Système strict d'interdiction de vol ou d'approbation | Sensible à la sécurité |
| **Pôles de transport** | A proximité des gares et aéroports | Interdiction totale de vol | Sécurité aérienne |
| **Parc/Espace vert** | Parc de la ville | Relativement détendu (nécessite une approbation) | Rassemblement de foule |

### 3.2 Système de classification de l'espace aérien à basse altitude

Le « Règlement provisoire sur la gestion des vols d'avions sans pilote » publié par l'Administration de l'aviation civile de Chine (en vigueur en 2024) établit un cadre de contrôle vertical avec une hauteur réelle de 120 m :

- **Hauteur réelle inférieure à 120 m** : les drones légers ($< 250\text{g}$) peuvent voler librement et nécessitent un enregistrement sous leur nom réel ; les micro-UAV ($< 500\text{g}$) ne sont pas soumis aux restrictions de qualification de vol
- **Hauteur réelle 120 m-300 m** : inclus dans le contrôle, application de l'espace aérien de vol requise
- **Espace aérien de fusion pour vols isolés** : des zones spécifiques permettent les opérations de fusion de drones et d'avions pilotés

La cartographie sémantique nécessite d'encoder ces contraintes réglementaires dans le système de planification afin que le drone puisse déterminer automatiquement la hauteur de vol et les limites de la zone en fonction de la zone fonctionnelle dans laquelle il se trouve.

### 3.3 Sources de données pour la classification sémantique des domaines fonctionnels

Le découpage des zones fonctionnelles urbaines s’appuie sur des informations géographiques multi-sources :

- **OSM (OpenStreetMap)** : données géographiques open source, fournissant une classification des caractéristiques de base telles que les routes, les bâtiments et les plans d'eau, et constituent une source préalable importante pour l'inférence de zones fonctionnelles.
- **Données POI (point d'intérêt)** : l'API de carte Amap/Baidu fournit des données sur les POI de la ville, et les fonctions régionales peuvent être déduites grâce à la densité et au type de POI (par exemple, les POI autour des écoles sont principalement des établissements d'enseignement)
- **Images de télédétection** : les images satellites Sentinel-2/Gaofen-2 fournissent des informations sur la classification macro de l'utilisation des terres.
- **Données d'urbanisme** : La couche d'occupation du sol (plan de contrôle) dans le plan directeur d'urbanisme, qui a un effet juridique

**Cadre d'intégration multi-source** :$$
\mathcal{F}_{\text{zone}}(\mathbf{x}) = \alpha \cdot f_{\text{osm}}(\mathbf{x}) + \beta \cdot f_{\text{poi}}(\mathbf{x}) + \gamma \cdot f_{\text{remote}}(\mathbf{x}) + \delta \cdot f_{\text{plan}}(\mathbf{x})
$$

---

## 4. Compréhension sémantique dynamique : prédiction des intentions et quantification des incertitudes

### 4.1 Prédiction des intentions des piétons/véhicules

Les obstacles dynamiques (piétons, cyclistes, véhicules) dans les rues urbaines constituent une menace majeure pour la sécurité du vol des drones. **La prédiction des intentions** nécessite non seulement de prédire l'emplacement futur des obstacles, mais également de comprendre leurs intentions comportementales :

$$
\hat{\mathbf{a}}_t^{(i)} = \arg\max_{\mathbf{a} \in \mathcal{A}} P(\mathbf{a} | \mathbf{b}_{1:t}^{(i)}, \mathcal{E})
$$

Parmi eux, $\mathbf{b}_{1:t}^{(i)}$ est la trajectoire comportementale historique de l'obstacle $i$, $\mathcal{E}$ est le contexte environnemental (état des feux de circulation, passage pour piétons, passage piéton, etc.), et $\mathcal{A}$ est l'intention définie (traverser la route, attendre sur le bord de la route, marcher le long du trottoir, etc.).

**Social LSTM** (Alahi et al., CVPR 2016) a introduit pour la première fois le Social Pooling pour modéliser l'interaction des piétons ; **Trajectron++** (Salzmann et al., ICRA 2020) a modélisé l'interaction multi-agents basée sur un réseau neuronal graphique (GNN), améliorant considérablement la précision des prédictions dans les scènes d'intersection urbaines.

### 4.2 Détection des conflits drone-UAV

Dans les corridors urbains à basse altitude, plusieurs drones peuvent fonctionner simultanément. **La détection des collisions** nécessite de prédire les collisions potentielles dans l'espace et dans le temps :$$
\text{Conflit} \Leftrightarrow \exists t \in [t_{\text{start}}, t_{\text{end}}] : \|\mathbf{p}_A(t) - \mathbf{p}_B(t)\| < d_{\text{coffre-fort}}
$$

Où $d_{\text{safe}}$ est la distance de sécurité (généralement 5 $\text{m}$ ou plus), $\mathbf{p}_A(t)$, $\mathbf{p}_B(t)$ sont les trajectoires prédites des deux drones.

**Les stratégies de résolution des conflits** comprennent :
- **Allocation basée sur des règles** : attribuez des plages horaires indépendantes (Time Slots) ou des couloirs spatiaux à différents drones
- **Négociation distribuée** : les drones échangent des prédictions de trajectoire via la communication et collaborent pour planifier des chemins sans conflit
- **Planification centralisée** : la station de contrôle au sol planifie plusieurs trajectoires de drones de manière unifiée

### 4.3 Planification tenant compte des incertitudes

Il existe une incertitude inhérente à la classification sémantique : un mur-rideau en verre sur une façade de bâtiment peut être classé à tort comme ciel, et la végétation peut être classée à tort comme bâtiment. **Planification tenant compte de l'incertitude** Intégrez l'incertitude perçue dans la prise de décision :

$$
\underline{\mathcal{C}} = \{\mathbf{x} : P(\text{collision} | \mathbf{x}) < \epsilon\}
$$

Planifiez les trajectoires uniquement dans les zones libres avec un niveau de confiance suffisamment élevé pour réserver une marge de sécurité aux erreurs de détection. Cette idée est conforme à l'optimisation robuste - assurer la sécurité dans le pire des cas d'ensembles incertains.

---

## 5. Planification sémantique : conception de fonctions de coût

### 5.1 Carte des coûts sémantiquement améliorée

La planification traditionnelle utilise une carte de coûts géométrique et chaque cellule de la grille $c_{i,j}$ code uniquement la probabilité de collision. **Semantic Enhanced Cost Map** superpose le coût sémantique au coût géométrique :

$$
C_{\text{total}}(i,j) = w_g \cdot C_{\text{geo}}(i,j) + w_s \cdot C_{\text{sem}}(i,j) + w_t \cdot C_{\text{temporel}}(i,j)
$$

Le coût sémantique $C_{\text{sem}}(i,j)$ est fixé en fonction du domaine fonctionnel auquel appartient l'unité :$$
C_{\text{sem}}(i,j) = \begin{cases}
0 & \text{parc ouvert} \\
1 & \text{place commerciale} \\
5 & \text{quartier résidentiel} \\
20 & \text{école/hôpital} \\
+\infty & \text{zone d'exclusion aérienne}
\fin{cas}
$$

### 5.2 Contraintes douces et contraintes dures

**Les contraintes strictes** sont des restrictions physiques/réglementaires qui ne peuvent être enfreintes :
- Il est absolument interdit de voler dans la zone d'exclusion aérienne
- Ne volez pas en dessous de l'altitude minimale de sécurité
- La distance à l'obstacle ne doit pas être inférieure à la marge de sécurité

Les **contraintes souples** sont des objectifs privilégiés qui peuvent être dépassés à un coût :
- Essayez de survoler les parcs plutôt que les zones résidentielles
- Essayez de rester près des murs du bâtiment plutôt que de traverser des places ouvertes (pour réduire les perturbations dues au vent)
- Essayez de voler en dehors des périodes très bruyantes

La planification sémantique gère ces deux types de contraintes grâce à une **optimisation hiérarchique** : minimiser le coût des contraintes souples tout en satisfaisant les contraintes strictes.

### 5.3 EGPBS : planification de la sécurité basée sur la sémantique

**EGPBS (Environment Graph-based Planning with Buffer Shrinking)** est un cadre de planification sémantique pour les scènes urbaines (idées dérivées de la recherche liée à l'IROS 2023) :

1. **Construction de graphe d'environnement** : modélisez la scène urbaine sous forme de structure graphique $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, les nœuds $\mathcal{V}$ représentent des zones sémantiques (blocs de construction, rues, parcs) et les bords $\mathcal{E}$ représentent les relations de connexion entre les zones.
2. **Rétrécissement du tampon de sécurité** : Dans les zones étroites des passages à basse altitude, le tampon de sécurité sémantique (Safety Buffer) se rétrécira automatiquement pour permettre le passage (les couloirs étroits sont toujours praticables)
3. **Recherche de graphique + optimisation de trajectoire** : A* recherche des chemins à gros grains sur le graphe d'environnement, suivi d'une optimisation dans le domaine temporel via la famille de trajectoires MINCO

---

## 6. Sécurité et conformité : intégration STMP/LAANC

### 6.1 STMP : Planification de la matrice de risques spatio-temporellesSTMP (Spatial-Temporal Mitigation Planning) est un cadre d'évaluation des risques liés aux drones proposé par la FAA. Il évalue le niveau de risque global de chaque vol en analysant des facteurs tels que la densité de population, la distance de l'aéroport et les installations militaires dans la zone de vol.

Le mappage sémantique peut directement prendre en charge l'évaluation STMP :
- **Couche de densité de population** : Statistiques de densité de population piétonne au sol par segmentation sémantique $\rho_{\text{people}}(\mathbf{x})$
- **Couche d'installation sensible** : marquez les écoles, les hôpitaux et les lieux religieux grâce aux données de POI
- **Couche d'installations aéronautiques** : zone de dégagement aéroportuaire et zone de protection de route superposées

Score de risque global :

$$
R(\mathcal{T}) = \int_0^T \left( \alpha \cdot \rho_{\text{people}}(\mathbf{p}(t)) + \beta \cdot I_{\text{airport}}(\mathbf{p}(t)) + \gamma \cdot I_{\text{sensitive}}(\mathbf{p}(t)) \right) dt
$$

### 6.2 LAANC : Autorisation de l'espace aérien en temps réel

LAANC (Low Altitude Authorisation and Notification Capability) est un système d'autorisation de l'espace aérien en temps réel pour les drones fourni par la FAA. L'UAV demande si l'emplacement actuel se trouve dans l'espace aérien autorisé via l'interface UTM (UAV Traffic Management) et peut demander une autorisation en temps réel.

Parcours d'intégration du système de perception sémantique et du LAANC :
1. Cartographie sémantique du drone pour identifier la zone fonctionnelle de localisation actuelle
2. Si vous êtes à proximité de la limite de la zone réglementée, initiez une demande d'autorisation auprès du LAANC
3. LAANC renvoie le statut d'autorisation (Approuvé / En attente / Refusé)
4. Une fois l'autorisation obtenue, le système de planification débloquera l'autorisation de vol dans la zone.

---

## 7. Cadre mathématique : fusion de perceptions multimodales et construction de cartes de coûts sémantiques

### 7.1 Fusion sémantique bayésienne

Le cœur de la fusion multi-capteurs est l’inférence bayésienne. Supposons que $z_t$ soit l'observation sémantique (résultat de la segmentation de la caméra) au temps $t$ et que la carte sémantique antérieure est $m$, alors la carte sémantique postérieure est :$$
P(m | z_{1:t}) \propto P(z_t | m, z_{1:t-1}) \cdot P(m | z_{1:t-1})
$$

Dans une implémentation pratique, $P(z_t | m)$ est modélisé par un classificateur CRF (Conditional Random Field) ou MLP, en tenant compte des a priori du lissage spatial (les pixels voisins ont tendance à avoir des étiquettes similaires).

### 7.2 Optimisation du graphe factoriel du SLAM sémantique

L'optimisation conjointe de la cartographie sémantique et du positionnement est réalisée à travers un graphe factoriel :

$$
\mathbf{x}^* = \arg\min_{\mathbf{x}, m} \sum_{i} \| \mathbf{r}_i^{\text{odom}} \|^2 + \sum_{j} \| \mathbf{r}_j^{\text{loop}} \|^2 + \sum_{k} \| \mathbf{r}_k^{\text{sémantique}} \|^2
$$

Parmi eux, $\mathbf{r}^{\text{odom}}$ est le résidu d'odométrie, $\mathbf{r}^{\text{loop}}$ est le résidu de détection de fermeture de boucle, et $\mathbf{r}^{\text{semantic}}$ est le résidu d'observation sémantique (contrainte de cohérence entre les points sémantiques 3D et la carte sémantique).

Le principal défi du SLAM sémantique réside dans l'ambiguïté des observations sémantiques : le même type d'étiquettes sémantiques peut correspondre à des formes géométriques complètement différentes (par exemple, des bâtiments de styles différents sont étiquetés « bâtiment »), et une relaxation appropriée doit être introduite dans le graphe factoriel.

---

## 8. Tendances futures et questions en suspens

### 8.1 Grand modèle de langage + conscience sémantique

Les modèles de langage visuel (VLM) tels que GPT-4V apportent des capacités de **conscience ouverte du vocabulaire** au mappage sémantique : ils ne se limitent plus à un ensemble prédéfini de catégories sémantiques fermées, mais peuvent comprendre des concepts sémantiques arbitraires décrits en langage naturel.

**Scénario d'application** : L'utilisateur dit « Éviter la zone scolaire », VLM peut identifier les caractéristiques de l'école (aire de jeux, plate-forme de lever de drapeau, panneau scolaire) à partir de l'image ; l'utilisateur dit "Survolez la route avec le café", VLM peut localiser la route cible. Cela fait passer le mappage sémantique de « requête passive » à « compréhension active ».

### 8.2 Protection de la vie privée et désensibilisation des donnéesLa cartographie sémantique implique un grand nombre d'images d'environnements urbains, soulevant des problèmes de confidentialité (visibilité à l'intérieur des bâtiments, enregistrement des activités humaines). Les stratégies de réponse technique comprennent :
- **Traitement Edge-side** : la segmentation sémantique est terminée dans l'unité informatique embarquée du drone et l'image originale n'est pas retransmise à la station au sol.
- **Rendu respectueux de la confidentialité** : codez ou supprimez automatiquement les zones contenant des visages
- **Cartographie sémantique fédérée** : plusieurs drones partagent des mises à jour de cartes sémantiques mais pas d'images brutes.

---

## 9. Résumé

La cartographie sémantique élève la planification urbaine des drones à basse altitude de la **perception géométrique** à la **compréhension cognitive**. Grâce à la segmentation sémantique, à l'estimation de la profondeur et à la division des zones fonctionnelles, les drones peuvent comprendre « où je vole », « pourquoi est-ce sensible ici », « comment dois-je me déplacer », au lieu de simplement savoir « y a-t-il des obstacles ici ».

Les principales orientations de recherche comprennent : **Conscience sémantique du vocabulaire ouvert** (autonomisation des grands modèles), **Planification tenant compte de l'incertitude** (faire face aux erreurs de perception), **Intégration de la conformité STMP/LAANC** (contraintes sémantiques basées sur la réglementation). À mesure que le cadre réglementaire de l’économie urbaine à basse altitude continue de s’améliorer, les capacités de sensibilisation sémantique deviendront un élément standard des systèmes de planification urbaine des drones.

---

## Références

- Cheng, B., Misra, I., Schwing, A.G., et al. (2022). MaskFormer pour la segmentation sémantique et d'instance. *CVPR*. https://doi.org/10.1109/CVPR52688.2022.00227

- Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., ... & Girshick, R. (2023). Segmentez n'importe quoi. *ICCV*.

- Ranftl, R., Lasinger, K., Hafner, D., Schindler, K. et Koltun, V. (2020). Vers une estimation robuste de la profondeur monoculaire : mélange d’ensembles de données pour un transfert d’ensembles de données croisés sans prise de vue. *IEEETPAMI*. https://doi.org/10.1109/TPAMI.2020.3019967- Ranftl, R., Bochkovskiy, A. et Koltun, V. (2021). Transformateurs de vision pour une prédiction dense. *ICCV*. https://doi.org/10.1109/ICCV48922.2021.01017

- Alahi, A., Goel, K., Ramanathan, V., Robicquet, A., Fei-Fei, L. et Savarese, S. (2016). Social LSTM : Prédiction de trajectoire humaine dans des espaces très fréquentés. *CVPR*. https://doi.org/10.1109/CVPR.2016.99

- Salzmann, T., Ivanovic, B., Chakravarty, P. et Pavone, M. (2020). Trajectron++ : prévision de trajectoire dynamiquement réalisable avec des données hétérogènes. *ECCV*. https://doi.org/10.1007/978-3-030-46732-6_43

- Zhou, H., Ren, D., Wu, J. et al. (2023). Egpbps : planification basée sur des graphiques d'environnement avec réduction de la mémoire tampon pour la navigation des drones. *IROS*.

- Liu, Y., Chen, J., Wang, X. et al. (2023). Depth-Anything : libérer la puissance des données non étiquetées à grande échelle. *arxiv:2401.10891*.

---

*Cet article est le quatrième chapitre étendu d'une série d'articles sur la planification d'itinéraires urbains à basse altitude pour les drones. *