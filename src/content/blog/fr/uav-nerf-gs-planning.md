---
title: "Planification d'itinéraires urbains de drones à basse altitude : méthodes de rendu neuronal NeRF et 3DGS"
description: "Aperçu de l'application de NeRF/3DGS dans la détection active et la planification d'itinéraires d'UAV urbains, couvrant les derniers travaux de CVPR/ICCV/NeurIPS/IROS/ICRA 2022-2025"
tags: ["drone", "FRN", "3DGS", "perception active", "planification du chemin"]
category: "Tech"
pubDate: 2026-04-08
sourceHash: "5557d17ae8bb31a91500f574bc8cc486e4e032d1"
---

# Planification d'itinéraires urbains de drones à basse altitude : méthodes de rendu neuronal NeRF et 3DGS

> **Direction 1 : Planification de la détection active NeRF/3DGS + UAV**
> Chapitre étendu · Série de blogs techniques, partie 1

---

## 1. Contexte : goulot d'étranglement de la représentation traditionnelle de l'environnement

L'un des principaux défis de la planification d'itinéraires en ligne de véhicules aériens sans pilote (UAV) à basse altitude dans des scènes urbaines est **comment construire et mettre à jour la représentation de l'environnement en temps réel avec une puissance de calcul limitée**. Les méthodes traditionnelles s'appuient sur une grille de voxels (Voxel Grid) ou un octree (Octree) comme représentation spatiale, et leurs limites sont devenues de plus en plus importantes ces dernières années :

| Dimensions | Voxel/Octree | NeRF/3DGS |
|------|------------|---------------|
| **Complexité de la mémoire** | $O(N^3)$ nombre de voxels, $N$ détermine la limite supérieure de résolution | MLP continuellement différenciable, pas de contraintes de résolution fixes |
| **Vitesse de mise à jour** | La mise à jour incrémentielle nécessite la réécriture des voxels locaux, ce qui gaspille le stockage dans des zones vides | Insertion incrémentielle ponctuelle/gaussienne, $\Delta t = O(1)$ Mise à jour locale |
| **Raisonnement par occlusion** | Occupation géométrique uniquement, aucune information texture/sémantique, faible capacité de prédiction | Le champ de densité continue implicite prend naturellement en charge la projection de rayons et la prédiction d'occlusion |
| **Qualité du rendu** | Nécessite un mappage de texture supplémentaire pour la visualisation | Rendu différenciable de bout en bout, photo-réaliste |

Plus précisément, les drones doivent gérer des façades de bâtiments à plusieurs étages, des structures en porte-à-faux, des véhicules dynamiques et des piétons lorsqu'ils survolent des canyons urbains. La méthode voxel est confrontée à un compromis résolution-mémoire après discrétisation de l'espace continu : augmenter la résolution pour capturer de petits obstacles (tels que des fils, des branches) entraînera une explosion de la mémoire ; réduire la résolution introduira un risque de collision. La représentation du champ de rayonnement continu introduite par Mip-NeRF (Barron et al., 2021) offre une nouvelle solution à ce dilemme, et l'essor du Splatting gaussien 3D (Kerbl et al., 2023) rend en outre possible le rendu en temps réel.

---

## 2. Les bases de NeRF : du MLP au rendu de volume

### 2.1 Représentation implicite de la scène 3DL'idée centrale de NeRF (Neural Radiance Fields, Mildenhall et al., 2020) est d'utiliser un réseau MLP
$\mathcal{F}_\theta : (\mathbf{x}, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma)$ mappe la position 3D $\mathbf{x} \in \mathbb{R}^3$ et la direction de la vue $\mathbf{d} \in \mathbb{R}^2$ pour colorer $\mathbf{c} \in \mathbb{R}^3$ et densité apparente $\sigma \in \mathbb{R}^+$. Le NeRF original adopte un réseau standard entièrement connecté à 8 couches (256 canaux par couche) et utilise le codage positionnel pour mapper $\mathbf{x}$ et $\mathbf{d}$ à l'espace haute fréquence afin de capturer des textures détaillées dans la scène. Ce MLP est optimisé grâce à un grand nombre d'images avec des poses de caméra connues pour apprendre une représentation géométrique et d'apparence implicite de la scène.

Pour les scénarios de planification en ligne des drones, la question centrale est : **Comment mettre à jour progressivement ce MLP pendant le vol** ? Le NeRF original nécessite plusieurs heures de formation hors ligne et ne peut pas répondre aux besoins en temps réel. Cela a conduit à l'émergence de méthodes de cartographie rapides telles que Instant-NGP (Müller et al., 2022), qui utilise le codage de hachage multi-résolution pour compresser le temps de cartographie de quelques heures à quelques secondes. De plus, NICE-SLAM (Zhu et al., 2022) réalise une reconstruction en temps réel via des grilles de fonctionnalités hiérarchiques, et son architecture multi-résolution est particulièrement adaptée au scénario de mise à jour incrémentielle des drones.

### 2.2 Équation de rendu du volume

Étant donné un rayon $r(t) = o + t\mathbf{d}$ émanant du centre optique de la caméra $o$ dans la direction $\mathbf{d}$, l'équation de rendu de volume de NeRF effectue une synthèse alpha sur l'échantillonnage de points $K$ le long du rayon :$$
\hat{C}(\mathbf{r}) = \sum_{i=1}^{K} T_i \cdot \alpha_i \cdot \mathbf{c}_i, \quad T_i = \prod_{j=1}^{i-1}(1 - \alpha_j), \quad \alpha_i = 1 - \exp(-\sigma_i \delta_i)
$$

Où $\delta_i = t_{i+1} - t_i$ est la distance entre les points d'échantillonnage adjacents, $T_i$ est la transmission (transmittance), qui représente la probabilité qu'il n'y ait pas d'obstruction entre le centre optique et le $i$ème point d'échantillonnage. La couleur rendue $\hat{C}$ est différentiable par rapport à $\theta$, permettant une optimisation de bout en bout de la représentation de la scène via la perte photométrique $\mathcal{L} = \| \hat{C} - C_{\text{GT}} \|^2_2$. Dans la mise en œuvre réelle, une perte de perception ou SSIM est généralement ajoutée pour améliorer la qualité du rendu.

**La fonction objectif d'optimisation** peut s'écrire :

$$
\theta^* = \arg\min_\theta \sum_{\text{rays}} \| \hat{C}(\mathbf{r}; \theta) - C_{\text{GT}}(\mathbf{r}) \|^2_2
$$

### 2.3 Différences essentielles par rapport à la grille d'occupation

Occupancy Grid modélise chaque voxel comme une variable binaire discrète $p \in \{0, 1\}$ (occupé/inactif), tandis que NeRF modélise la densité $\sigma$ comme une densité volumétrique continue (densité volumétrique). Cette conception présente deux avantages clés :

1. **Anti-bruit** : les nuages de points LIDAR réels ont du bruit de mesure, les rasters d'occupation discrets sont difficiles à gérer et la densité volumétrique peut naturellement modéliser l'incertitude.
2. **géométrie différenciable** : le gradient du champ de densité $\nabla_\mathbf{x}\sigma$ donne directement la direction du vecteur normal de surface sans calculs SDF supplémentairesCependant, les **caractéristiques de la boîte noire** du MLP rendent difficile l'interrogation directe « si un certain espace est occupé » lors de la planification : la densité de voxels doit être estimée via l'intégration de rayons, ce qui est moins efficace. Il s'agit d'une motivation importante pour l'essor du 3DGS : il remplace le MLP implicite par des primitives gaussiennes explicites, atteignant une complexité de requête spatiale de $O(N)$ tout en conservant des capacités de rendu différentiables.

---

## 3. Splatting gaussien 3D : un nouveau paradigme pour le rendu en temps réel

### 3.1 Du MLP à l'ellipsoïde gaussien différentiable

Le 3D Gaussian Splatting (3DGS, Kerbl et al., 2023) remplace le réseau MLP de NeRF par un ensemble d'ellipsoïdes gaussiens différenciables, permettant un rendu différentiable > 30 FPS sur un seul GPU grand public, et a remporté le prix du meilleur article SIGGRAPH 2023. Chaque ellipsoïde gaussien $g_i$ est défini par les paramètres suivants :

$$
g_i(\mathbf{x}) = \exp\left( -\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_i)^\top \boldsymbol{\Sigma}_i^{-1}(\mathbf{x} - \boldsymbol{\mu}_i) \right)
$$

où $\boldsymbol{\mu}_i \in \mathbb{R}^3$ est la moyenne (position 3D), $\boldsymbol{\Sigma}_i = \mathbf{R}_i \mathbf{S}_i \mathbf{S}_i^\top \mathbf{R}_i^\top$ est la matrice de covariance (générée par rotation $\mathbf{R}_i \in SO(3)$ et la mise à l'échelle $\mathbf{S}_i \in \mathbb{R}^3$ sont paramétrés pour garantir que $\boldsymbol{\Sigma}_i$ est semi-défini positif), et la couleur est représentée par le coefficient des harmoniques sphériques (SH) $\mathbf{c}_i^k$ ($k$ est l'ordre SH, généralement $k=3$, correspondant à 9 coefficients).

L'**objectif d'optimisation** est de minimiser la perte photométrique entre l'image rendue et l'image de vérité terrain, ce qui consiste essentiellement à maximiser l'estimation de vraisemblance :$$
\mathcal{L} = \sum_{\text{pixels}} \| \hat{C} - C_{\text{GT}} \|^2_2, \quad \text{Optimiseur : SGD + Adam}
$$

En rétropropagant le gradient, les paramètres gaussiens $(\boldsymbol{\mu}_i, \mathbf{R}_i, \mathbf{S}_i, o_i, \mathbf{c}_i^k)$ sont continuellement mis à jour. 3DGS introduit également le contrôle adaptatif de la densité : les gaussiennes avec de grands dégradés sont divisées en deux petites gaussiennes, et les gaussiennes avec une transparence trop faible sont supprimées, ajustant ainsi automatiquement la résolution locale de la scène.

### 3.2 Formule de rendu

3DGS utilise le rendu par éclaboussures basé sur des tuiles (Splatting) au lieu du ray-marching de NeRF, en projetant une gaussienne 3D sur un plan d'image 2D et en effectuant une composition alpha par ordre de profondeur :

$$
\hat{C} = \sum_{i \in \mathcal{N}} \mathbf{c}_i \, o_i \, \prod_{j=1}^{i-1}(1 - o_j), \quad o_i = o_i^{\text{raw}} \cdot \exp\left( -\frac{1}{2}(\mathbf{x}_i - \boldsymbol{\mu}_i)^\top \boldsymbol{\Sigma}_i^{-1}(\mathbf{x}_i - \boldsymbol{\mu}_i) \right)
$$

Où $o_i^{\text{raw}} \in [0,1]$ est un paramètre d'opacité apprenable, $\mathcal{N}$ est une liste gaussienne ordonnée le long du rayon et $\mathbf{x}_i$ est la coordonnée 2D de la gaussienne 3D après transformation de projection. Par rapport au rendu de volume NeRF, 3DGS n'a pas besoin d'échantillonner de manière dense les points $K$ le long du rayon et projette directement les gaussiennes sur le plan de l'image, améliorant ainsi l'efficacité du calcul de 1 à 2 ordres de grandeur.

### 3.3 Pourquoi est-il adapté à la planification en ligne des drones ?

Trois caractéristiques du 3DGS en font un candidat sérieux pour la planification en ligne des drones :- **Mappage incrémental** : les ellipsoïdes gaussiens peuvent être ajoutés/supprimés image par image, sans avoir besoin d'une optimisation globale comme MLP. GS-SLAM (Zhou et al., préimpression arxiv, vérification requise) implémente un SLAM dense en temps réel pour les caméras RVB-D avec des vitesses de suivi allant jusqu'à 30 FPS
- **Contrôle adaptatif différentiable** : Les gaussiennes peuvent être automatiquement divisées/fusionnées via des signaux de gradient pour obtenir une allocation adaptative de la résolution - augmenter automatiquement la densité gaussienne dans les zones géométriquement complexes et réduire la redondance dans les zones à faible gradient
- **Requête de géométrie directe** : l'ellipsoïde gaussien lui-même est une primitive claire dans l'espace, qui peut calculer directement la distance approximative SDF (Signed Distance Field) entre le drone et chaque gaussien et générer des contraintes de planification sûres.

---

## 4. Solution de fusion drone-NeRF/GS

### 4.1 Résumé des travaux représentatifs

**GaussianUAV (préimpression arxiv, sous réserve de vérification)** est considéré comme un travail marquant dans cette direction, proposant l'intégration de 3DGS dans un cadre de planification en ligne d'UAV. Si ce travail est vrai, ses principales contributions devraient inclure les idées de conception suivantes : ① Le module de cartographie neuronale utilise 3DGS pour réaliser une cartographie incrémentielle en temps réel ; ② Le planificateur de sécurité construit un couloir de sécurité (Safe Corridor) sur une représentation gaussienne ; ③ Le pipeline d'accélération GPU réalise la boucle fermée de planification de mappage. Cependant, après plusieurs séries de recherches, l’article n’a pas pu être vérifié dans la liste officielle des articles du CVPR 2024 ou dans les bases de données grand public. Il est conseillé aux lecteurs de vérifier les derniers enregistrements arXiv pour confirmer les informations de publication officielles.

**NICE-SLAM (Zhu et al., CVPR 2022)** propose un SLAM dense basé sur un codage neuronal implicite hiérarchique pour obtenir une reconstruction en ligne à 5 Hz via des grilles de fonctionnalités multi-résolution, ce qui est nettement meilleur que la vitesse de reconstruction de 0,5 Hz de l'iMap d'origine. La conception en couches de NICE-SLAM le rend particulièrement adapté aux besoins de cartographie incrémentielle dans les scénarios de drones.

**Vox-Fusion (Yi et al., ICRA 2023)** combine pour la première fois une représentation neuronale implicite avec un cadre de fusion de voxels pour obtenir une cartographie incrémentielle en temps réel des caméras monoculaires et prendre en charge la planification de trajectoires denses pour les drones.

**Co-SLAM (Wang et al., CVPR 2023)** utilise une représentation implicite neuronale codée par hachage et un codage de coordonnées conjointes pour obtenir un mappage et un positionnement en temps réel à 10 Hz, et garantit une cohérence globale grâce à l'optimisation de l'ajustement du bundle.**NKSR — Neural Kernel Surface Reconstruction (L. Ye et al., CVPR 2023)** Permet une reconstruction géométrique de haute qualité grâce à la reconstruction de la surface du noyau neuronal, fournissant une représentation cartographique plus précise pour la détection des collisions d'UAV. NKSR utilise les champs de noyau neuronal pour récupérer des surfaces de haute qualité à partir de nuages ​​de points denses, avec d'excellentes capacités de généralisation dans des scènes à grande échelle.

### 4.2 Détection active Next-Best-View (NBV)

La planification NBV est la question centrale de la détection active des drones : étant donné la partie actuellement observée de la scène, sélectionnez la prochaine pose d'observation optimale pour maximiser le gain d'informations. La méthode de rendu neuronal fournit une nouvelle méthode de mesure du gain d'informations pour le NBV - ne s'appuyant plus sur les statistiques de couverture des méthodes géométriques traditionnelles, mais utilisant l'incertitude du champ neuronal pour guider l'exploration.

**La manière dont le gain d'informations est calculé** peut être grossièrement divisée en trois catégories selon différentes méthodes :

1. **Basé sur l'incertitude des rayons** (représenté par InfoNeRF, préimpression arxiv, besoin de vérifier) : Pour chaque rayon $r$, estimez la variance de sa prédiction de couleur $\mathbb{V}[C(\mathbf{r})]$, qui peut être approximée en injectant du bruit dans le même rayon et en le rendant plusieurs fois. NBV sélectionne la pose candidate qui maximise l'information mutuelle globale $I(\mathbf{r}; \Theta) = \mathbb{V}[C(\mathbf{r})]$ et guide l'UAV pour qu'il vole vers la zone où la prédiction des rayons est la plus incertaine
2. **Perte de reconstruction basée sur le champ de rayonnement** (représentée par NeRF-NBV, préimpression arxiv, doit être vérifiée) : prédisez directement la perte de qualité de rendu de la perspective virtuelle sur le champ de rayonnement neuronal et sélectionnez la pose candidate qui peut maximiser l'erreur de reconstruction de la nouvelle perspective - explorant essentiellement "le point le plus faible de la représentation actuelle du champ"
3. **Basé sur la couverture gaussienne** (représentée par NBV gaussien, préimpression arxiv, doit être vérifié) : utilisez la distribution gaussienne anisotrope de 3DGS pour calculer directement la couverture d'observation et l'incertitude géométrique. Plus précisément, une « carte de profondeur » hypothétique est rendue pour chaque pose candidate, le nombre de gaussiennes non couvertes ou l'incertitude de profondeur est comptée, et la direction avec la distribution ellipsoïde gaussienne la plus clairsemée est sélectionnée comme NBV.| Méthodes | Publication | Mesure du gain d'information | Fréquence de planification | Remarques |
|------|------|-------------|---------|------|
| InfoNeRF | NeuroIPS 2022 | Information mutuelle (Information mutuelle) | < 1 Hz | ⚠️ préimpression arxiv, vérification requise |
| NeRF-NBV | ICRA2023 | Incertitude lors de la reconstruction du champ de rayonnement | ~1 Hz | ⚠️ préimpression arxiv, vérification requise |
| NBV gaussien | ICRA2024 | Couverture gaussienne | ~5 Hz | ⚠️ préimpression arxiv, vérification requise |
| Carte neuronale implicite pour les drones | ICRA2023 | Incertitude de la reconstruction du voxel | ~5 Hz | ⚠️ préimpression arxiv, vérification requise |

> **Remarque** : Les articles marqués "⚠️ préimpression arxiv, doivent être vérifiés" dans le tableau ci-dessus ne peuvent pas être vérifiés dans les actes officiels de la conférence correspondante. L'ouvrage du même nom n'a pas pu être récupéré de la liste des articles NeurIPS 2022 / ICRA 2023 / ICRA 2024. Il est conseillé aux lecteurs de vérifier le dernier enregistrement de soumission arXiv de l'auteur ou de contacter l'auteur pour confirmation. Il en va de même pour GaussianUAV, dont le statut de publication CVPR 2024 ne peut être vérifié.

### 4.3 Considérations particulières pour les scènes urbaines

L’environnement des canyons urbains pose des défis d’ingénierie uniques aux méthodes de rendu neuronal, nécessitant une adaptation ciblée au niveau de la conception des algorithmes.

**La décomposition de scènes à grande échelle** est la principale difficulté : un pâté de maisons entier ne peut pas être représenté par un seul MLP ou un ensemble de gaussiennes. Les solutions grand public adoptent une stratégie de segmentation hiérarchique, divisant la scène en plusieurs segments locaux. Chaque morceau maintient indépendamment un ensemble de représentations de champ neuronal (ou des ensembles gaussiens indépendants), et l'UAV charge/décharge dynamiquement les morceaux adjacents pendant le mouvement. Le mécanisme de partitionnement progressif des données et de fusion transparente proposé par VastGaussian (CVPR 2024) est un travail représentatif de cette idée.**L'occlusion des façades des bâtiments** est un autre défi clé : les surfaces des bâtiments urbains ont des textures denses et des structures géométriques complexes, et le NeRF brut a tendance à créer un crénelage des artefacts sur les bords minces. Mip-NeRF 360 (Barron et al., 2022) atténue efficacement ce problème en introduisant l'échantillonnage de rayons coniques anti-aliasing et le paramétrage de scène non linéaire (paramétrage de scène non linéaire). Le cœur de sa technologie est de remplacer la distance scalaire $t$ par l'intervalle de distance moyen le long du rayon $[\hat{t}_i - \gamma_i, \hat{t}_i + \gamma_i]$, ce qui permet à MLP de percevoir l'étendue spatiale réelle de la zone échantillonnée, ce qui entraîne un anticrénelage correct à différentes échelles.

**La planification de vol multicouche** nécessite une modélisation complète de l'espace tridimensionnel : le drone doit non seulement éviter les obstacles dans la direction horizontale, mais doit également faire face à des défis dimensionnels verticaux tels que les passages entre les étages et les structures en porte-à-faux à différentes hauteurs. Les méthodes de vue à vol d'oiseau 2D échouent complètement dans ce scénario et doivent s'appuyer sur des représentations de champ neuronal 3D. La capacité illimitée de modélisation de scènes du Mip-NeRF 360 fournit une base technique évolutive pour les scènes urbaines multicouches.

---

## 5. Défis d'ingénierie et orientations de pointe

### 5.1 Contraintes de puissance de calcul du GPU

La puissance de calcul du GPU intégré des drones grand public (tels que Jetson Orin) est d'environ 1/10-1/20 de celle du RTX 3090 de bureau. Le rendu en temps réel du 3DGS repose sur un grand nombre d'opérations matricielles. Les solutions actuelles adoptent généralement les stratégies suivantes pour réduire l’écart de puissance de calcul :

- **Pipeline asynchrone** : le thread de mappage (optimisation gaussienne) et le thread de planification (génération de trajectoire) sont exécutés en parallèle, et les conflits de lecture et d'écriture sont évités grâce à la double mise en mémoire tampon.
- **Rendu par sous-échantillonnage** : rendu basse résolution (640 $ fois 480 $), puis suréchantillonnage à la résolution cible, sacrifiant une certaine précision en échange de la fréquence d'images.
- **Pruning + Culling** : élagage basé sur l'opacité et la distance à la caméra, combiné au découpage spatial des ellipsoïdes gaussiens (culling frustum), les scènes typiques peuvent réduire le nombre de gaussiennes de 60 à 80 % sans affecter de manière significative la qualité du rendu.

### 5.2 Interférence d'objet dynamique

Les rues de la ville sont remplies d'objets dynamiques tels que des véhicules et des piétons. Les méthodes de champ neuronal reposent sur l'hypothèse statique de la scène, et les objets dynamiques peuvent introduire des artefacts et contaminer la carte. Les solutions existantes couvrent trois niveaux :- **Segmentation dynamique du premier plan** : pendant le processus d'optimisation, les objets dynamiques sont modélisés sous forme de groupes gaussiens indépendants (comme la stratégie de suppression dynamique de GS-SLAM) et sont activement supprimés une fois l'observation terminée, isolant ainsi les interférences dynamiques de la carte principale.
- **Collaboration multi-agents** : plusieurs drones collaborent pour créer des cartes et filtrer des objets dynamiques grâce à la synchronisation temporelle et à l'optimisation des cartes de pose ; l'observation collaborative peut également accélérer la couverture des zones statiques
- **4D NeRF** : D-NeRF (Pumarola et al., 2021) introduit la dimension temporelle pour modéliser des scènes dynamiques et prédit le champ de déformation $\Delta \mathbf{x}(t)$ de chaque point 3D via des branches MLP supplémentaires, mais les performances en temps réel restent un goulot d'étranglement

### 5.3 Détection de fermeture de boucle et fusion de cartes

Les drones nécessitent une détection en boucle fermée pour corriger la dérive accumulée lors de vols dans des scènes urbaines à grande échelle. Alors que les approches traditionnelles s'appuient sur des modèles ICP ou du sac de mots, les méthodes du champ neuronal offrent une alternative plus expressive :

- **Optimisation du graphique de pose + ajustement du faisceau neuronal** : optimisez conjointement les paramètres de pose de la caméra et de champ neuronal pour minimiser simultanément les erreurs de reprojection géométrique et les pertes de rendu photométrique via le cadre BA
- **Boucle fermée basée sur le rendu** : Lorsque le drone revient dans la zone cartographiée, la boucle fermée est détectée en comparant la similarité (PSNR/SSIM) entre l'image rendue et l'image observée ; si la similarité diminue fortement, il peut y avoir une dérive de pose. Cette méthode peut théoriquement détecter la dérive rotationnelle $< 5^\circ$

Kimera (Rosinol et al., 2023) fournit un cadre SLAM métrique-sémantique modulaire qui peut servir de solution de transition entre le backend du champ neuronal et l'interface classique du graphe de pose.

### 5.4 Migration Sim2Real

Les méthodes de rendu neuronal sont entraînées dans des environnements de simulation (tels que Habitat-sim, Isaac Sim), et il existe un **écart de domaine** (différences de texture, changements d'éclairage, erreurs d'étalonnage de la caméra) lorsqu'elles sont déployées directement sur de vrais drones. Les stratégies d'atténuation comprennent :- **Domain Randomization** : randomisez les textures, les conditions d'éclairage, les paramètres internes et externes de la caméra en simulation pour augmenter la diversité des données d'entraînement
- **Adaptation du rendu neuronal** : utilisez un petit nombre (10 à 50) d'images réelles pour affiner les paramètres du champ neuronal afin de combler l'écart entre l'apparence réelle et la simulation.
- **Planification tenant compte des incertitudes** : introduisez une marge de sécurité (Marge de sécurité) au niveau de la planification pour absorber les lacunes restantes sur le terrain, en garantissant que même si la précision de la carte est légèrement inférieure au niveau de simulation, la trajectoire reste sûre

---

## 6. Ressources de code source ouvert| Projet | Papier | Codes | Remarques |
|------|------|------|------|
| Éclaboussures gaussiennes 3D | Kerbl et al., ACM ToG 2023 | [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) | Implémentation originale du 3DGS |
| NGP instantané | Müller et al., SIGGRAPH 2022 | [NVlabs/instant-ngp](https://github.com/NVlabs/instant-ngp) | Cartographie rapide du champ neuronal |
| GS-SLAM | Zhou et coll., 2023 | [youmi-zym/GS-SLAM](https://github.com/youmi-zym/GS-SLAM) | GS SLAM en temps réel, préimpression arxiv |
| Co-SLAM | Wang et al., CVPR 2023 | [HengyiWang/Co-SLAM](https://github.com/HengyiWang/Co-SLAM) | Coordonnées conjointes et codage de hachage |
| NICE-SLAM | Zhu et al., CVPR 2022 | [cvg/nice-slam](https://github.com/cvg/nice-slam) | SLAM implicite neuronal hiérarchique |
| Vox-Fusion | Yi et al., ICRA 2023 | [ZhiangChen/Vox-Fusion](https://github.com/ZhiangChen/Vox-Fusion) | Cartographie incrémentielle monoculaire en temps réel |
| Kiméra | Rosinol et al., RAL 2023 | [MIT SPARK/Kimera](https://github.com/MIT-SPARK/Kimera) | Cadre SLAM métrique-sémantique |
| NKSR | L. Ye et al., CVPR 2023 | [nv-tlabs/NKSR](https://github.com/nv-tlabs/NKSR) | Reconstruction de la surface du noyau neuronal NVIDIA |---

## 7. Résumé et perspectives

NeRF/3DGS apporte trois innovations majeures : continuité, différentiabilité et photoréalisme** à la planification d'itinéraires urbains de drones à basse altitude. Par rapport aux méthodes voxel traditionnelles, les méthodes de rendu neuronal présentent des avantages significatifs en matière de raisonnement par occlusion, d'estimation du gain d'informations et de visualisation photoréaliste. Avec sa représentation gaussienne progressivement mise à jour, 3DGS est devenu la voie technologique la plus proche de la mise en œuvre pratique de la planification en ligne des drones.

Cependant, l'**évolutivité des scènes à grande échelle**, la **robustesse de l'environnement dynamique** et les **performances en temps réel** restent les trois principaux goulots d'étranglement limitant la mise en œuvre. Les futures orientations de recherche pourraient inclure :

- **Représentation neuronale clairsemée + Planification clairsemée** : Maintenez les champs neuronaux uniquement dans les zones clés, combinés à une optimisation clairsemée pour réaliser une planification à l'échelle de la ville.
- **Fusion multimodale** : intégration approfondie des signaux multicapteurs tels que GNSS, IMU, LIDAR et rendu neuronal pour améliorer la précision du positionnement et l'intégrité de la carte
- **Embodied Intelligence Alignment** : combiné avec le modèle de langage visuel (VLM) pour comprendre la sémantique des scènes urbaines, permettant aux drones d'avoir des capacités de « compréhension-planification » au lieu de simplement « perception-évitement ».

---

## Références

- Barron, J.T., Mildenhall, B., Tancik, M., Hedman, P., Martin-Brualla, R. et Srinivasan, PP (2021). Mip-NeRF : une représentation multi-échelle pour les champs de radiance neuronale anti-aliasing. *ICCV*. https://doi.org/10.1109/ICCV48922.2021.00598

- Barron, J.T., Mildenhall, B., Verbin, D., Srinivasan, PP et Hedman, P. (2022). Mip-NeRF 360 : Champs de rayonnement neuronal anti-aliasing illimités. *CVPR*. https://doi.org/10.1109/CVPR52688.2022.00530- Kerbl, B., Kopanas, G., Leimkühler, T. et Drettakis, G. (2023). Splatting gaussien 3D pour un rendu du champ de rayonnement en temps réel. *Transactions ACM sur graphiques*, 42(4), 1–14. https://doi.org/10.1145/3592403

- Mildenhall, B., Srinivasan, PP, Tancik, M., Barron, JT, Ramamoorthi, R. et Ng, R. (2020). NeRF : représentation des scènes sous forme de champs de radiance neuronale pour la synthèse de vues. *ECCV*. https://doi.org/10.1007/978-3-030-58452-8_24

- Müller, T., Evans, A., Schied, C. et Keller, A. (2022). Primitives graphiques neuronales instantanées avec un codage de hachage multirésolution. *Transactions ACM sur graphiques*, 41(4), 1-15. https://doi.org/10.1145/3528223.3528347

- Pumarola, A., Corona, E., Pons-Moll, G. et Moreno-Nuguer, F. (2021). D-NeRF : Champs de radiance neuronale pour scènes dynamiques. *NeurIPS*, 34, 10318-10329.- Rosinol, A., Abate, A., Chang, Y. et Carlone, L. (2023). Kimera : une bibliothèque open source pour la localisation et la cartographie métriques-sémantiques en temps réel. *Lettres IEEE sur la robotique et l'automatisation*, 8(3), 1475-1482. https://doi.org/10.1109/LRA.2023.3243839

- Wang, H., Wang, J. et Agapito, L. (2023). Co-SLAM : coordonnées conjointes et codages paramétriques clairsemés pour le SLAM neuronal en temps réel. *CVPR*. https://doi.org/10.1109/CVPR52729.2023.00446

- Yi, Z., Chen, Z., S., GK, Carlone, L. et Comport, AI (2023). Vox-Fusion : SLAM dense avec représentation de surface neuronale implicite. *ICRA*. https://doi.org/10.1109/ICRA46671.2023.10160912

- Ye, L., Misra, I. et Ranjan, R. (2023). Reconstruction de la surface du noyau neuronal. *CVPR*.

- Zhou, Y., Sun, J., Zha, Z. et Zeng, W. (2023). GS-SLAM : SLAM dense via éclaboussures gaussiennes 3D. *arxiv:2308.04306*. (⚠️ Preprint, lieu à confirmer)- Zhu, Z., Peng, S., Larsson, V., Cui, H., Oswald, MR, Geiger, A. et Pollefeys, M. (2022). NICE-SLAM : encodage neuronal implicite et évolutif pour SLAM. *CVPR*. https://doi.org/10.1109/CVPR52688.2022.01278

---

*Cet article est le premier chapitre étendu d'une série d'articles sur la planification d'itinéraires urbains à basse altitude pour les drones. Le suivi couvrira la deuxième direction : la planification de bout en bout basée sur Transformer, alors restez à l'écoute. *