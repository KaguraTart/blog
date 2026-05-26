---
title: "Planification d'itinéraires urbains de drones à basse altitude : théorie et algorithme dans des scénarios CBD à haute densité"
description: "Analyse systématiquement les principaux problèmes et les idées de solutions de la planification d'itinéraires urbains de drones à basse altitude, couvrant les méthodes A*, RRT*, APF, FM², MILP, ORCA et MARL, avec une dérivation mathématique et des équations complètes."
pubDate: 2026-05-15
tags: ["drone", "planification du chemin", "espace aérien urbain", "Algorithme d'optimisation", "UTM", "résolution de conflits"]
category: Tech
sourceHash: "5588745289f1f698abd6def7ed9650375344a695"
---

# Planification d'itinéraires urbains de drones à basse altitude : théorie et algorithme dans des scénarios CBD à haute densité

> Lorsque des centaines de drones font la navette entre les gratte-ciel en même temps, la planification d'itinéraire n'est plus un simple problème de « voler d'un point A à un point B » : il s'agit d'un **problème d'optimisation contraint de haute dimension** qui recherche un équilibre entre l'espace tridimensionnel, le temps, l'énergie et la sécurité.

---

## Introduction : Pourquoi le CBD est-il si difficile ?

L'espace aérien urbain à basse altitude est généralement défini comme la plage de vol **0–300 m** (AGL) au-dessus du sol. Ce niveau de hauteur constitue le principal champ de bataille pour la logistique, l’inspection, les interventions d’urgence et d’autres applications des drones. Le CBD (Central Business District) est le sous-scénario le plus complexe pour trois raisons :

**1. Des bâtiments denses forment un « canyon urbain »**

Les immeubles de grande hauteur rendent les couloirs de vol disponibles extrêmement étroits et la ligne de vue est bloquée, ce qui réduit la précision du GPS. Les bords des bâtiments génèrent également de fortes turbulences. À basse altitude, en dessous de 50 mètres, ces turbulences peuvent complètement faire perdre le contrôle à un petit multirotor.

**2. Les drones à haute densité provoquent des conflits intenses**

Dans une scène de banlieue, seuls quelques drones peuvent voler en même temps ; tandis que dans le cadre d’un système de gestion du trafic aérien urbain (UTM) mature, le nombre de drones au-dessus du CBD peut atteindre plus de 40 drones par minute. Cela signifie que la détection et la résolution des conflits (CD&R) deviennent le principal goulot d'étranglement du système plutôt qu'une fonction périphérique.

**3. Couplage obstacle dynamique et multi-contraintes**

En plus des bâtiments, les drones doivent également faire face à des zones d'exclusion aérienne temporaires, à des itinéraires d'avions habités, à des changements de champ de vent en temps réel et à des risques pour la sécurité liés à la densité de la foule au sol - tous ces éléments travaillant ensemble pour rendre difficile la gestion par un algorithme de planification de trajectoire unique.

---

## 1. Modélisation de problèmes : transformer des problèmes de vol en problèmes mathématiques

### 1.1 Grille d'occupation 3D

Discrétisez l'espace urbain dans une grille de voxels, et chaque voxel enregistre son statut d'occupation :

$$
O(x,y,z) = \begin{cases} 1 & \text{Obstacle / Zone d'exclusion aérienne} \\ 0 & \text{Flyable} \end{cases}
$$

La résolution du voxel est généralement de 1 à 5 m et la région centrale du CBD peut être affinée à 0,5 m. Les données de hauteur des bâtiments proviennent d'une base de données SIG (Système d'information géographique) et sont mises à jour dynamiquement en combinaison avec des capteurs en temps réel.

### 1.2 Définition mathématique de la trajectoire 4D

La trajectoire de vol d'un seul drone est une courbe spatiale paramétrée par rapport au temps :$$
\boldsymbol{\xi}(t) = \bigl(x(t),\; y(t),\; z(t)\bigr), \quad t \in [t_0,\, t_f]
$$

Après avoir introduit la dimension temporelle, la dimension de la trajectoire est augmentée jusqu'à une courbe espace-temps 4D $\boldsymbol{\xi}^{4D}(t) = (x,y,z,t)$. C'est l'idée centrale de ce qu'on appelle la **planification de trajectoire 4D** : éviter les conflits spatiaux grâce à la planification temporelle (le moment d'arrivée à un certain point) est moins cher que de purs détours spatiaux.

Dans un système multi-machines, deux drones doivent à tout moment respecter les contraintes de séparation de sécurité :

$$
\|\boldsymbol{\xi}_i(t) - \boldsymbol{\xi}_j(t)\|_2 \geq d_{sep}, \quad \forall\, i \neq j,\; \pour tous\, t \in [t_0, t_f]
$$

Où $d_{sep}$ est la séparation minimale de sécurité, avec une valeur typique de 5 à 30 m (en fonction de la vitesse de vol et de la précision du GPS).

### 1.3 Forme générale du problème d'optimisation multi-objectifs

La planification d'itinéraires est essentiellement un problème d'optimisation multi-objectifs contraint :

$$
\min_{\boldsymbol{\xi}}\; J(\boldsymbol{\xi}) = w_1 J_{len} + w_2 J_{time} + w_3 J_{énergie} + w_4 J_{risque}
$$

La signification de chaque sous-élément est la suivante :

| Répartition | Signification | Mesures typiques |
|------|------|----------|
| $J_{len}$ | Longueur du chemin | $\int_{t_0}^{t_f}\|\dot{\boldsymbol{\xi}}\|\,\mathrm{d}t$ |
| $J_{heure}$ | Temps de vol | $t_f - t_0$ |
| $J_{énergie}$ | Consommation d'énergie | $\int P(v)\,\mathrm{d}t$ |
| $J_{risque}$ | Risque au sol | Points pour survoler des zones peuplées |

Contraintes (toutes sont indispensables) :- **Accessibilité** : $O(\boldsymbol{\xi}(t)) = 0,\;\forall t$
- **Cinématique** : $\dot{\mathbf{x}} = f(\mathbf{x}, \mathbf{u})$ (modèle cinématique/dynamique du drone)
- **Séparation sécurisée** : $\|\boldsymbol{\xi}_i(t)-\boldsymbol{\xi}_j(t)\| \geq d_{sep},\;\forall i\neq j$
- **Conditions aux limites** : $\boldsymbol{\xi}(t_0)=\mathbf{p}_{start},\;\boldsymbol{\xi}(t_f)=\mathbf{p}_{goal}$
- **Limite de vitesse** : $v_{min} \leq \|\dot{\boldsymbol{\xi}}(t)\| \leq v_{max}$

---

## 2. Algorithme de planification de chemin de machine unique

Avant d’aborder la collaboration multi-machines, comprenez d’abord l’algorithme de base dans un scénario mono-machine.

### 2.1 Algorithme A* : la pierre angulaire de la recherche de graphes

A* recherche le chemin le plus court sur un graphe spatial discrétisé (Waypoint Graph ou Visibility Graph). La valeur d'évaluation de chaque nœud $n$ est :

$$
f(n) = g(n) + h(n)
$$

Où $g(n)$ est le **coût cumulé réel** du point de départ au nœud $n$ :

$$
g(n) = g(\text{parent}) + d(\text{parent},\, n)
$$

$h(n)$ est la fonction heuristique admissible de $n$ à la cible (ne surestimez jamais le coût réel). Heuristiques de distance euclidienne couramment utilisées dans l'espace urbain 3D :

$$
h(n) = \|\mathbf{p}_n - \mathbf{p}_{objectif}\|_2 = \sqrt{(x_n-x_g)^2+(y_n-y_g)^2+(z_n-z_g)^2}
$$

Dans les scénarios urbains, il ne suffit pas de considérer uniquement la distance géométrique. Présentation du **Coût pondéré du risque au sol** :

$$
d(u,v) = \ell_{uv}\cdot\bigl(1 + \beta\cdot\mathcal{R}_{uv}\bigr)
$$Parmi eux, $\ell_{uv}$ est la longueur du segment du corridor, $\mathcal{R}_{uv}\in[0,1]$ est le score de risque au sol du corridor (combinant des facteurs tels que la densité de population, le type de bâtiment, les conséquences des accidents, etc.) et $\beta$ est le coefficient de pondération des risques. Cela amène A* à choisir des itinéraires qui survolent des zones à faible risque (par exemple des rivières, des parcs), même s'ils sont légèrement détournés.

> Limites de A* : La qualité de la carte de l'espace aérien détermine la qualité de la compréhension. Dans le CBD haute densité, le nombre de nœuds dans le graphe peut atteindre des centaines de milliers, et la construction du graphe lui-même est un défi.

### 2.2 Algorithme RRT* : planification optimale asymptotique complète de manière probabiliste

RRT* (Rapidly-exploring Random Tree Star) explore les chemins réalisables en échantillonnant aléatoirement dans un espace continu, ce qui est particulièrement adapté aux scènes d'obstacles complexes et de grande dimension.

**Requête du voisin le plus proche** - Recherchez le nœud le plus proche du point d'échantillonnage aléatoire dans l'arborescence $\mathcal{T}$ :

$$
x_{le plus proche} = \arg\min_{x \in \mathcal{T}} \|x - x_{rand}\|_2
$$

**Expansion de l'étape** - Étendre la taille du pas $\delta$ de la direction $x_{le plus proche}$ à la direction $x_{rand}$ :

$$
x_{nouveau} = x_{le plus proche} + \delta \cdot \frac{x_{rand} - x_{le plus proche}}{\|x_{rand} - x_{le plus proche}\|_2}
$$

L'amélioration principale de **RRT* - Rewire :** Recherchez tous les nœuds voisins de la sphère avec $x_{new}$ comme centre et rayon $r_n$ :

$$
r_n = \gamma_{RRT^*}\!\left(\frac{\log n}{n}\right)^{1/d}
$$

Où $n$ est le nombre de nœuds de l'arbre actuel, $d$ est la dimension de l'espace (scène 3D $d=3$) et $\gamma_{RRT^*}$ est une constante liée au volume d'espace libre. Ce rayon diminue à mesure que les points d'échantillonnage augmentent, garantissant une optimalité asymptotique.

Mise à jour des coûts :

$$
c(x_{nouveau}) = c(x_{près de}) + d(x_{près de},\, x_{nouveau})
$$

Si le coût de $x_{near}$ peut être réduit via $x_{new}$, la reconnexion est effectuée :$$
\text{Si } c(x_{near}) > c(x_{new}) + d(x_{new},\, x_{near}),\text{ puis changez le nœud parent de } x_{near} \text{ en } x_{new}
$$

Comme le nombre d’échantillonnages tend vers l’infini, RRT* est assuré de converger vers la solution optimale avec une probabilité de 1 :

$$
\lim_{n\to\infty} c(\xi_n^*) = c^* \quad \text{(presque certainement)}
$$

### 2.3 Méthode du champ de potentiel artificiel (APF) : le roi du temps réel

L'APF construit la cible comme un champ gravitationnel et les obstacles comme un champ répulsif, et le drone se déplace sous l'action de la force résultante.

**Potentiel gravitationnel** (puits de potentiel quadratique, tirant vers la cible) :

$$
U_{att}(\mathbf{p}) = \frac{1}{2}k_{att}\|\mathbf{p} - \mathbf{p}_{goal}\|^2
$$

**Potentiel de répulsion** (activé dans le rayon d'influence de l'obstacle $\rho_0$) :

$$
U_{rep}(\mathbf{p}) = \begin{cases} \dfrac{1}{2}k_{rep}\!\left(\dfrac{1}{\rho(\mathbf{p})}-\dfrac{1}{\rho_0}\right)^{\!2} & \rho(\mathbf{p}) \leq \rho_0 \\[8pt] 0 & \rho(\mathbf{p}) > \rho_0 \end{cases}
$$

Où $\rho(\mathbf{p})=\min_{obs}\|\mathbf{p}-\mathbf{p}_{obs}\|$ est la distance de dégagement entre le drone et l'obstacle le plus proche.

**Force résultante** (gradient négatif du champ de potentiel total) :

$$
\mathbf{F}(\mathbf{p}) = -\nabla U_{att}(\mathbf{p}) - \nabla U_{rep}(\mathbf{p})
$$

Composants de dégradé explicites :

$$
\nabla U_{att} = k_{att}\,(\mathbf{p}-\mathbf{p}_{goal})
$$$$
\nabla U_{rep} = k_{rep}\!\left(\frac{1}{\rho}-\frac{1}{\rho_0}\right)\!\frac{1}{\rho^2}\,\nabla\rho \qquad (\rho\leq\rho_0)
$$

La mise à jour en ligne de l'APF est généralement légère et adaptée à l'évitement d'obstacles en temps réel ; cependant, si la distance d'obstacle la plus proche est calculée directement $\rho(\mathbf{p})=\min_{obs}\|\mathbf{p}-\mathbf{p}_{obs}\|$ selon la définition susmentionnée, l'implémentation naïve nécessite généralement au moins de traverser l'obstacle défini à chaque étape, ce qui est d'environ $O(n_{obs})$. Les requêtes en une seule étape ne peuvent coûter environ $O(1)$) que si les champs de distance, les ESDF ou les requêtes raster ont été précalculés. Mais il a toujours un talon d'Achille dans le CBD Canyon : **Minimum local** - Lorsque la gravité et la répulsion sont exactement équilibrées, le drone restera coincé à un point non ciblé et sera incapable d'avancer. Les améliorations incluent les perturbations aléatoires, les champs de potentiel harmonique ou l'algorithme PF-RRT combiné avec RRT.

### 2.4 Méthode des carrés à déplacement rapide (FM²) : l'élégance de la propagation du front d'onde

FM² (Fast Marching Square) génère des trajectoires fluides en résolvant l'équation Eikonal, particulièrement adaptée à l'évitement des conflits 4D.

**Équation eikonale** - une équation aux dérivées partielles décrivant le temps d'arrivée du front d'onde $T(\mathbf{x})$ :

$$
|\nabla T(\mathbf{x})|^2 \cdot v^2(\mathbf{x}) = 1
$$

où $v(\mathbf{x})$ est la vitesse de propagation dans l'espace. Construisez une **carte de vitesse basée sur la distance de dégagement** afin que le front d'onde décélère naturellement à proximité des obstacles :

$$
v(\mathbf{x}) = c\cdot\rho(\mathbf{x}) = c\cdot\min_{obs}\|\mathbf{x}-\mathbf{x}_{obs}\|
$$

Après avoir résolu $T(\mathbf{x})$, le chemin est extrait par descente de gradient sur le champ $T$ :

$$
\dot{\boldsymbol{\xi}}(s) = -\frac{\nabla T\bigl(\boldsymbol{\xi}(s)\bigr)}{\left|\nabla T\bigl(\boldsymbol{\xi}(s)\bigr)\right|}
$$**Étendu à l'évitement des conflits 4D :** Introduction d'une carte de vitesse variable dans le temps, laissant $v\à 0$ dans la région spatio-temporelle déjà occupée par d'autres drones :

$$
v(\mathbf{x},t) = v_0(\mathbf{x})\cdot\phi_{conflit}(\mathbf{x},t)
$$

Lorsque $\phi_{conflict}\atteint 0$, le front d'onde contourne naturellement le volume de conflit spatio-temporel, obtenant ainsi un chemin 4D sans collision. Les chemins générés par FM² sont naturellement lisses ($C^\infty$ continu) et ne nécessitent aucun post-traitement de lissage supplémentaire.

---

## 3. Le problème central des scènes haute densité : la détection et la résolution des conflits (CD&R)

Le défi fondamental dans les scénarios de drones à haute densité n’est pas de trouver un chemin, mais de garantir que tous les chemins sont sûrs simultanément.

### 3.1 Détection des conflits

Définissez le vecteur de position relative entre le drone $i$ et $j$ :

$$
\Delta\mathbf{p}_{ij}(t) = \mathbf{p}_i(t) - \mathbf{p}_j(t)
$$

**Conditions de détermination des conflits** (les séparations horizontales **et** verticales sont violées simultanément) :

$$
\text{Conflit}_{ij} \iff \|\Delta\mathbf{p}_{ij}(t)\|_{xy} < d_h \;\wedge\; |\Delta z_{ij}(t)| < d_v
$$

Reportez-vous aux paramètres typiques de la NASA UTM CONOPS : norme de séparation horizontale $d_h=30\,\text{m}$, norme de séparation verticale $d_v=10\,\text{m}$.

En pratique, le système doit prédire les conflits avant le vol plutôt que d'attendre que des conflits surviennent avant de réagir. Supposons que le drone vole à une vitesse constante dans la fenêtre d'anticipation $[0, T_h]$ :

$$
\mathbf{p}_i(t) = \mathbf{p}_i^0 + \mathbf{v}_i t, \quad \mathbf{p}_j(t) = \mathbf{p}_j^0 + \mathbf{v}_j t
$$

**Heure du point d'approche le plus proche (CPA, point d'approche le plus proche) :**$$
t_{CPA} = -\frac{\Delta\mathbf{p}_{ij}^0 \cdot \Delta\mathbf{v}_{ij}}{\|\Delta\mathbf{v}_{ij}\|^2}, \qquad \Delta\mathbf{v}_{ij} = \mathbf{v}_i - \mathbf{v}_j
$$

Espacement minimum au CPA :

$$
d_{min} = \|\Delta\mathbf{p}_{ij}(t_{CPA})\|
$$

Lorsque $d_{min} < d_{sep}$ et $t_{CPA}\in[0, T_h]$, il est déterminé qu'il existe un **conflit de prédiction** et le mécanisme de libération doit être déclenché immédiatement.

### 3.2 Résolution des conflits

Les stratégies de désarmement se répartissent en trois catégories et peuvent être utilisées individuellement ou en combinaison :

**Stratégie 1 : Ajustement de la vitesse**

Appliquez un facteur d'échelle de vitesse $\alpha$ au drone $i$, en décélérant ou en accélérant dans la plage de dynamique autorisée :

$$
\mathbf{v}_i^{nouveau} = \alpha\,\mathbf{v}_i, \quad \alpha\in\!\left[\frac{v_{min}}{v_i},\;\frac{v_{max}}{v_i}\right]
$$

Le $\alpha$ optimal minimise l'écart par rapport au plan d'origine tout en satisfaisant les contraintes de séparation :

$$
\alpha^* = \arg\min_\alpha\;|\alpha-1| \quad \text{s.t. } d_{min}^{nouveau}(\alpha)\geq d_{sep}
$$

**Stratégie 2 : Cap sur le changement**

Faites pivoter la direction de vol du drone $i$ de $\delta\psi$ dans le plan horizontal :

$$
\mathbf{v}_i^{nouveau} = v_i\begin{pmatrix}\cos(\psi_i+\delta\psi)\\\sin(\psi_i+\delta\psi)\\0\end{pmatrix}
$$$$
\delta\psi^* = \arg\min_{|\delta\psi|}\;\delta\psi \quad \text{s.t. } d_{min}(\delta\psi)\geq d_{sep}
$$

**Troisième stratégie : séparation des couches d'altitude**

Dans le scénario CBD, l’attribution d’altitudes fixes en fonction de la direction du vol est la solution systématique la plus efficace :

$$
z_{couche}(k) = z_{base} + k\cdot\Delta z_{couche}, \quad k\in\{0,1,\ldots,N_{couche}-1\}
$$

Configuration typique : direction est $\to z_1$, direction ouest $\to z_2$, direction nord $\to z_3$, direction sud $\to z_4$, espacement des couches $\Delta z_{layer}=10\,\text{m}$. Cela réduit la dimensionnalité du problème de collision tridimensionnelle à un problème bidimensionnel, réduisant ainsi considérablement la complexité du système.

### 3.3 Coordination décentralisée : barrières de vitesse et ORCA

L'UTM centralisé peut obtenir la solution optimale globale, mais la surcharge de communication augmente avec $O(N^2)$ comme nombre de drones $N$, faisant face à un goulot d'étranglement dans des scénarios de densité extrêmement élevée. Parmi les solutions décentralisées, **Velocity Obstacle (VO)** et son amélioration **ORCA** sont les frameworks les plus matures.

**Définition de la barrière de vitesse** - UAV $i$ L'ensemble des vitesses interdites en raison de la présence du UAV $j$ (toutes les vitesses qui provoqueraient une collision dans la fenêtre de temps $\tau$) :

$$
VO_{ij}^\tau = \left\{\mathbf{v}_i \;\middle|\; \exists\, t\in[0,\tau],\; \mathbf{p}_i+\mathbf{v}_i t \;\in\; \mathbf{p}_j+\mathbf{v}_j t \oplus \mathcal{D}(d_{sep})\right\}
$$

où $\mathcal{D}(r)$ est un disque/sphère de rayon $r$ et $\oplus$ est la somme de Minkowski.

**Évitement optimal des collisions réciproques (ORCA)** - Chaque agent n'assume que « la moitié » de la responsabilité d'évitement pour éviter d'être trop conservateur. ORCA définit une contrainte de demi-espace pour l'agent $i$ par rapport à $j$ :$$
ORCA_{ij} = \left\{\mathbf{v} \;\middle|\; \bigl(\mathbf{v}-\mathbf{v}_{opt}^i\bigr)\cdot\hat{\mathbf{n}}_{ij} \geq \tfrac{1}{2}u_{ij}\right\}
$$

où $u_{ij}$ est la taille du changement de vitesse minimum et $\hat{\mathbf{n}}_{ij}$ pointe vers la direction normale de la limite $VO_{ij}$.

L'ensemble des vitesses réalisables pour l'agent $i$ (coupe toutes les contraintes voisines puis coupe les contraintes dynamiques) :

$$
\mathcal{V}_i^{ORCA} = \bigcap_{j\neq i} ORCA_{ij} \;\cap\; \mathcal{V}_{dyn}
$$

Parmi eux, $\mathcal{V}_{dyn}$ code des contraintes dynamiques telles que la vitesse et l'accélération maximales. ORCA a atteint un taux de réussite de 100 % dans des scénarios de densité supérieure à 40 images/minute, avec une complexité de calcul de $O(N^2)$, ce qui le rend adapté au déploiement en temps réel.

---

## 4. Modélisation de la théorie des graphes : réseau d'espace aérien urbain

### 4.1 Construction du schéma du réseau routier

L'espace aérien urbain est modélisé sous la forme d'un graphe orienté pondéré :

$$
G = (V,\; E,\; W), \quad W : E \to \mathbb{R}_+
$$

- **Nœud** $V$ : au-dessus des intersections routières, des sommets des bâtiments, des points de transfert clés
- **Edge** $E$ : couloir de vol légal entre deux nœuds (doit réussir la vérification de détection de collision)
- **Edge Weight** $W$ : pondération scalaire multi-objectifs

$$
W(e_{ij}) = w_1\, d_{ij} + w_2\,\Delta t_{ij} + w_3\,\mathcal{R}_{ij} + w_4\,\mathcal{E}_{ij}, \quad \sum_{k} w_k = 1
$$

Contraintes de capacité du couloir (le nombre de drones passant en même temps ne dépasse pas la limite supérieure) :

$$
\text{load}(e_{ij},\, t) \leq C_{ij}, \quad \forall\, t
$$

L'état d'occupation de l'ensemble de l'espace aérien peut être décrit par un tenseur à quatre dimensions ($N_x\times N_y\times N_z$ est la grille de voxels, $N_t$ est le nombre de créneaux horaires) :$$
\mathbf{A} \in \{0,1\}^{N_x\times N_y\times N_z\times N_t}, \quad A_{x,y,z,t} = 1 \iff \exists\text{ voxel occupé par le drone}(x,y,z)\text{ dans la tranche horaire }t
$$

### 4.2 Modèle de consommation d'énergie du rotor du drone

La consommation d'énergie est un objectif d'optimisation important pour la planification d'itinéraires et nécessite une modélisation précise.

**Hover Power** (dérivé de la théorie de l'élan des éléments feuilles) :

$$
P_{hover} = \sqrt{\frac{(mg)^3}{2\,\rho_{air}\, A_r}}
$$

Où $m$ est la masse du drone, $g$ est l'accélération gravitationnelle, $\rho_{air}$ est la densité de l'air et $A_r$ est la surface du disque du rotor.

**Modèle de puissance de vol avancé** (Zeng et al. 2019, trois composants physiques) :

$$
P(v) = \underbrace{P_0\!\left(1+\frac{3v^2}{U_{tip}^2}\right)}_{\text{Résistance de la lame}} + \underbrace{P_i\!\left(\sqrt{1+\frac{v^4}{4v_0^4}}-\frac{v^2}{2v_0^2}\right)^{\!\frac{1}{2}}}_{\text{Puissance d'induction}} + \underbrace{\frac{1}{2}\,d_0\,\rho_{air}\,s\,A\,v^3}_{\text{Résistance corporelle}}
$$

Signification du paramètre : $P_0$ est la puissance de résistance du type de pale en vol stationnaire, $P_i$ est la puissance induite en vol stationnaire, $U_{tip}$ est la vitesse de pointe du rotor, $v_0$ est la vitesse induite en vol stationnaire, $d_0$ est le coefficient de traînée du fuselage, $s$ est la solidité du rotor et $A$ est la surface du disque du rotor.

Consommation d'énergie du segment de survol $e_{ij}$ (longueur $\ell_{ij}$, vitesse $v$) :

$$
\mathcal{E}_{ij} = \frac{\ell_{ij}}{v}\cdot P(v)
$$

**Vitesse de croisière optimale** (consommation d'énergie minimale par unité de distance) :

$$
v^* = \arg\min_v \frac{P(v)}{v}
$$

Pour un petit multicoptère typique ($m\approx 1\,\text{kg}$), $v^*$ se situe généralement entre 8 et 12 m/s.

---## 5. Champ de vent et effet canyon urbain

### 5.1 Modélisation des champs de vent urbains

La distribution de la vitesse du vent dans les canyons urbains est beaucoup plus complexe qu'à la campagne, et la distribution de Weibull est largement utilisée en modélisation statistique :

$$
f(v_w;\, k,\, \lambda) = \frac{k}{\lambda}\!\left(\frac{v_w}{\lambda}\right)^{k-1}\!\exp\!\left[-\!\left(\frac{v_w}{\lambda}\right)^k\right]
$$

Parmi eux, le paramètre de forme $k\environ 1,5$ – $2,5$ (la valeur la plus petite est prise lorsque les turbulences en zone urbaine sont fortes), et $\lambda$ est le paramètre d'échelle (calibré par des mesures météorologiques locales).

Profil logarithmique de la vitesse du vent près de la surface (pour les couches superficielles situées en dessous de la hauteur du toit) :

$$
\bar{u}(z) = \frac{u^*}{\kappa}\ln\!\left(\frac{z - d_0}{z_0}\right), \quad \kappa = 0,41 \text{(constante de von Kármán)}
$$

Où $u^*$ est la vitesse de frottement, $d_0$ est la hauteur de déplacement du plan nul et $z_0$ est la longueur de rugosité.

Impact quantitatif des champs de vent sur la planification des itinéraires :

**Temps de trajet corrigé du vent** (le long de la composante de direction du couloir $v_w\cos\theta_w$) :

$$
t_{ij} = \frac{d_{ij}}{v_{air} + v_w\cos\theta_w}
$$

**Consommation d'énergie du segment intégrale, y compris la résistance au vent** (Vitesse de l'air réelle = Vitesse sol $-$ Vitesse du vent) :

$$
\mathcal{E}_{ij}^{vent} = \int_0^{t_{ij}} P\!\left(\|\mathbf{v}_{UAV}(t) - \mathbf{v}_w(t)\|\right)\mathrm{d}t
$$

**Indice d'intensité de turbulence** (quantifie le risque de couloir, composante de risque pour les pondérations de bord $\mathcal{R}_{ij}$) :

$$
TI = \frac{\sigma_u}{\bar{u}}, \qquad \sigma_u = \sqrt{\overline{u'^2}}
$$

Les couloirs avec $TI > 0,3$ sont généralement marqués comme à haut risque, et le planificateur évitera ou augmentera activement le poids des bords de ce segment.

### 5.2 Rayon de sécurité dynamiqueLa turbulence autour des bâtiments augmente fortement à mesure que la marge de hauteur diminue. Par conséquent, la distance de sécurité ne doit pas être une constante fixe, mais doit être ajustée dynamiquement en fonction de l’altitude de vol :

$$
d_{safe}(h) = d_{base} + \frac{k\cdot H_{bld}}{h - H_{bld} + \epsilon}
$$

Où $h$ est la hauteur de vol actuelle, $H_{bld}$ est la hauteur des bâtiments à proximité et $\epsilon$ est le terme de régularisation pour empêcher le dénominateur d'être nul. Cette formule signifie que plus la marge de hauteur entre le drone et le sommet du bâtiment est petite, plus le dégagement latéral requis est grand.

Contraintes de marge dynamique :

$$
\rho\bigl(\mathbf{p}(t)\bigr) \geq d_{safe}\bigl(z(t)\bigr), \quad \forall\, t \in [t_0, t_f]
$$

---

## 6. Optimisation collaborative multi-machines : modélisation globale MILP

Pour le problème conjoint d'allocation de trajectoire et de créneau horaire des drones $N$, un modèle de **Programmation Linéaire Entiers Mixtes (MILP)** peut être établi pour obtenir la solution optimale globale à petite et moyenne échelle ($N\leq 50$).

**Fonction objectif** (minimiser le temps de réalisation total et la consommation d'énergie de tous les drones) :

$$
\min\;\sum_{k=1}^{N}\!\left(w_1\, T_k + w_2\,\mathcal{E}_k\right)
$$

**Variables de décision :**
- $x_{ij}^k \in \{0,1\}$ : si le drone $k$ sélectionne le couloir $(i,j)$
- $t_i^k \geq 0$ : L'heure à laquelle le drone $k$ arrive au nœud $i$

**Contrainte 1 - Conservation du trafic** (Chaque drone entre et sort une fois du nœud intermédiaire) :

$$
\sum_{j:(i,j)\in E}x_{ij}^k - \sum_{j:(j,i)\in E}x_{ji}^k = b_i^k, \quad \forall\, i\in V,\;\forall\, k
$$

Parmi eux, $b_i^k\in\{+1,\, 0,\, -1\}$ correspondent respectivement au point de départ, au nœud intermédiaire et au point final.

**Contrainte 2 - Capacité du corridor** :

$$
\sum_{k=1}^{N} x_{ij}^k \leq C_{ij}, \quad \forall\,(i,j)\in E
$$**Contrainte 3 - Cohérence temporelle** (l'heure d'arrivée correspond au temps de trajet) :

$$
t_j^k \geq t_i^k + \frac{d_{ij}}{v_{max}}\cdot x_{ij}^k, \quad \forall\,(i,j)\in E,\;\forall\, k
$$

**Contrainte 4 - Séparation temporelle** (les différents drones sur le même nœud doivent maintenir un intervalle de temps $\Delta t_{sep}$, linéarisation Big-M) :

$$
t_i^k - t_i^l \geq \Delta t_{sep} - M(1 - z_{kl}^i)
$$

$$
t_i^l - t_i^k \geq \Delta t_{sep} - M\, z_{kl}^i
$$

Parmi eux, $z_{kl}^i \in \{0,1\}$ est la variable de classement des séries chronologiques de l'UAV $k$, $l$ au nœud $i$ et $M$ est une constante suffisamment grande (méthode Big-M).

Lorsque la vitesse est également utilisée comme variable de décision, le problème est mis à niveau vers la programmation non linéaire en nombres entiers mixtes (MINLP) :

$$
\min_{x,\, t,\, v}\;\sum_k\sum_{(i,j)} x_{ij}^k\cdot\frac{d_{ij}}{v_{ij}^k}\cdot P(v_{ij}^k), \quad v_{min}\leq v_{ij}^k\leq v_{max}
$$

MINLP est un problème NP-difficile qui est résolu approximativement par des algorithmes heuristiques couramment utilisés dans la pratique (recherche fractale aléatoire SFS, optimisation de guépard MCO, etc.).

---

## 7. Solution d'apprentissage par renforcement : MARL et mécanisme d'attention

Lorsque l’échelle des drones dépasse la centaine, la complexité informatique du MILP est inacceptable. **L'apprentissage par renforcement multi-agents (MARL)** offre une alternative à la formation hors ligne et à l'inférence extrêmement rapide.

### 7.1 Conception de la fonction de récompense

La récompense reçue par chaque drone $i$ au pas de temps $t$ :

$$
r_i^t = r_{arriver}\cdot\mathbf{1}[objectif] - c_{étape} - c_{conflit}\cdot\mathbf{1}[conflit] - c_{détour}\cdot\|\mathbf{p}_i^t - \mathbf{p}_{direct}\|
$$La signification de chaque élément : $r_{arrive}$ est la récompense positive pour avoir atteint l'objectif ; $c_{step}$ est la pénalité de temps pour chaque étape de vol ; $c_{conflict}\cdot\mathbf{1}[conflict]$ est la pénalité lorsqu'un conflit survient ; $c_{detour}$ est la pénalité de détour pour s'écarter de la ligne droite.

### 7.2 Mise à jour Double-DQN (espace d'action discret)

$$
Q(s,a;\theta) \leftarrow Q(s,a;\theta) + \alpha\!\left[r + \gamma\, Q\!\left(s',\,\arg\max_{a'}Q(s',a';\theta);\,\theta^-\right) - Q(s,a;\theta)\right]
$$

Le réseau en ligne $\theta$ sélectionne les actions et le réseau cible $\theta^-$ évalue les valeurs, dissociant la sélection et l'évaluation pour réduire les biais de surestimation.

### 7.3 Mécanisme d'attention : modélisation de l'influence du voisin

La prise de décision de chaque drone dans le CBD nécessite de détecter le statut de ses voisins environnants. Le **mécanisme d'attention** permet à l'agent $i$ de pondérer dynamiquement l'influence des voisins $j$ :

$$
e_{ij} = \frac{(\mathbf{W}_Q\mathbf{h}_i)(\mathbf{W}_K\mathbf{h}_j)^\top}{\sqrt{d_k}}
$$

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_l \exp(e_{il})}, \qquad \mathbf{h}_i^{attn} = \sum_j \alpha_{ij}\,(\mathbf{W}_V\mathbf{h}_j)
$$

Le poids d'attention $\alpha_{ij}$ reflète la pertinence du voisin $j$ dans la prise de décision de l'agent $i$. Les voisins proches et en conflit de vitesse important reçoivent naturellement des pondérations plus élevées.

### 7.4 Dégradé de politique PPO (espace d'action continu/mixte)$$
\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\hat{A}_t,\;\mathrm{clip}\!\left(r_t(\theta),\,1-\varepsilon,\,1+\varepsilon\right)\hat{A}_t\right)\right]
$$

où le rapport de probabilité est :

$$
r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}
$$

L'opération Clip limite la taille de l'étape de mise à jour à la plage $[1-\varepsilon,\, 1+\varepsilon]$ (généralement $\varepsilon=0,2$) pour éviter que la formation ne plante en raison de mises à jour excessives des politiques.

**Paradigme de formation centralisée et d'exécution décentralisée (CTDE) :**
- **Phase de formation** : le réseau d'évaluation $V(s^{global};\phi)$ utilise l'état global et peut percevoir toutes les informations sur les agents
- **Phase d'exécution** : Le réseau de politiques $\pi_\theta(a_i\mid o_i)$ utilise uniquement les observations locales de l'agent $i$, sans communication

---

## 8. Lissage de trajectoire : courbe de Bézier et accrochage minimum

Le résultat de la planification d'un chemin est souvent une série de points de cheminement discrets, et le suivi direct de ces points de cheminement produira des virages serrés irréalisables. Il est nécessaire de générer des trajectoires continues réalisables dynamiquement grâce au **lissage de trajectoire**.

### 8.1 Courbe de Bézier

Une courbe de Bézier d'ordre $n$ est définie par $n+1$ points de contrôle $\{\mathbf{P}_i\}$ :

$$
\boldsymbol{\xi}(u) = \sum_{i=0}^{n}\binom{n}{i}(1-u)^{n-i}u^i\,\mathbf{P}_i, \quad u \in [0,1]
$$

Vitesse (dérivée par rapport au paramètre $u$) :

$$
\dot{\boldsymbol{\xi}}(u) = n\sum_{i=0}^{n-1}\binom{n-1}{i}(1-u)^{n-1-i}u^i\,(\mathbf{P}_{i+1}-\mathbf{P}_i)
$$Les courbes de Bézier ont naturellement des propriétés d'enveloppe convexe : la courbe se trouve toujours à l'intérieur de l'enveloppe convexe des points de contrôle, ce qui facilite la vérification des collisions avec des obstacles. Contraintes de courbure (limitant l'accélération centripète) :

$$
\kappa = \frac{\|\dot{\boldsymbol{\xi}}\times\ddot{\boldsymbol{\xi}}\|}{\|\dot{\boldsymbol{\xi}}\|^3} \leq \frac{a_{max}}{v^2}
$$

### 8.2 Minimum Snap : La solution standard pour les quadricoptères

Pour un drone quadricoptère, minimiser Snap (la dérivée seconde de l'accélération) équivaut à minimiser le taux de changement de poussée requise, ce qui entraîne une dynamique de vol optimale :

$$
\min\;\int_{t_0}^{t_f}\!\left\|\frac{d^4\boldsymbol{\xi}}{dt^4}\right\|^2\!\mathrm{d}t
$$

En exprimant la trajectoire sous la forme d'un polynôme par morceaux $\boldsymbol{\xi}_k(t)=\sum_{j=0}^{m}c_{kj}t^j$, le problème d'optimisation de dimension infinie ci-dessus peut être réduit à de la **programmation quadratique (QP)** :

$$
\min_{\mathbf{c}}\;\mathbf{c}^\top\mathbf{Q}\mathbf{c} \quad \text{s.t. }\mathbf{A}_{eq}\mathbf{c} = \mathbf{b}_{eq}
$$

La matrice $\mathbf{Q}$ code l'intégrale Snap (peut être calculée analytiquement) et la contrainte d'égalité $\mathbf{A}_{eq}\mathbf{c}=\mathbf{b}_{eq}$ force la trajectoire à passer par tous les points du trajet et assure la continuité de la position, de la vitesse et de l'accélération entre les segments.

---

## 9. Comparaison horizontale des méthodes| Méthode | Complétude | Optimalité | Complexité temporelle | En temps réel | Évolutivité multi-machines |
|------|--------|--------|------------|--------|------------|
| **A\*** | Terminé | Optimal (graphe discret) | $O(b^d)$ | Moyen | Pauvre |
| **RRT\*** | Probablement complet | Asymptotiquement optimal | $O(n\log n)$ | Mieux | Moyen |
| **APF** | Incomplet | Aucune garantie | $O(1)$/étape | Excellent | Bon |
| **FM²** | Terminé | Optimal (continu) | $O(N\log N)$ | Moyen | Moyen |
| **MILP** | Terminé | Globale optimale | NP-dur | Pauvre | Moyen ($N\leq50$) |
| **ORQUE** | Probablement complet | Optimale locale | $O(N^2)$ | Excellent | Excellent |
| **MARL+À l'attention** | Probabilité complète | approximatif | Entraînement intensif, inférence rapide | Bon | Excellent |

**Suggestions de sélection :**

- **Exigences de sécurité élevées et à petite échelle** ($N\leq 20$) → MILP global optimal
- **Échelle moyenne, sensible au temps réel** ($20 < N \leq 100$) → A\* / RRT\* + résolution de conflits ORCA
- **Grande échelle, haute densité** ($N > 100$) → MARL + mécanisme d'attention (délai d'inférence $< 10\,\text{ms}$)

---

## 10. Résumé et perspectives

La planification d'itinéraires urbains à basse altitude, en particulier à haute densité, pour les drones dans les scénarios CBD, est un problème d'ingénierie système multidisciplinaire. Cet article trie la chaîne complète des méthodes, depuis la **planification de chemin sur une seule machine** (A\*, RRT\*, APF, FM²) jusqu'à la **résolution de conflits multi-machines** (CD&R, ORCA, MILP) jusqu'aux **méthodes d'apprentissage** (MARL, PPO, attention), et donne l'expression mathématique précise de chaque maillon central.

**Trois principaux défis non résolus :**

1. **Replanification en ligne en temps réel** : Lorsqu'une zone d'exclusion aérienne soudaine ou une panne de drone se produit, le système doit terminer la replanification de toutes les trajectoires affectées dans un délai de 200 ms. Actuellement, MILP est loin de répondre à cette exigence et MARL est le candidat le plus prometteur.2. **Jumeaux numériques et fusion des perceptions** : des cartes urbaines tridimensionnelles précises en temps réel (y compris la construction dynamique de bâtiments, les enceintes temporaires et les informations météorologiques) constituent la base de la qualité de la planification des itinéraires. La technologie des jumeaux numériques devrait permettre une synchronisation de l’état de l’espace aérien au niveau centimétrique et inférieur à la seconde.

3. **Mise en œuvre technique du cadre réglementaire** : Les réglementations de gestion à basse altitude de l'Administration de l'aviation civile de Chine (CAAC), l'U-Space européen de l'EASA et l'UTM CONOPS américain de la FAA ont tous des exigences claires en matière de temps de résolution des conflits, de format de soumission du plan de vol, de procédures d'atterrissage d'urgence, etc., et la conception des algorithmes doit être profondément liée aux limites réglementaires.

> D'un point de vue mathématique, la planification de routes aériennes urbaines à basse altitude est un problème d'optimisation contraint en temps réel, non convexe, non linéaire, mixte, multi-agents. Aucun cadre unique ne peut « le résoudre en un seul clic » - dans la pratique de l'ingénierie, il s'agit souvent d'une architecture hybride à plusieurs niveaux : la planification cartographique est utilisée au niveau stratégique, ORCA est utilisée au niveau tactique et APF est utilisé au niveau d'urgence, qui forment ensemble un système robuste de gestion du trafic aérien.

---

**Principales références :**1. Karaman, S. et Frazzoli, E. (2011). *Algorithmes basés sur l'échantillonnage pour une planification de mouvement optimale.* International Journal of Robotics Research, 30(7), 846-894.
2. Van den Berg, J., Guy, SJ, Lin, M. et Manocha, D. (2011). *Évitement réciproque des collisions à n corps.* Robotics Research, 3–19.
3. Zeng, Y., Xu, J. et Zhang, R. (2019). *Minimisation de l'énergie pour la communication sans fil avec les drones à voilure tournante.* IEEE Transactions on Wireless Communications, 18(4), 2329-2345.
4. Mueller, MW, Hehn, M. et D'Andrea, R. (2015). *Une primitive de mouvement informatiquement efficace pour la génération de trajectoires de quadricoptères.* IEEE Transactions on Robotics, 31(6), 1294-1310.
5. Brittain, M. et Wei, P. (2019). *Contrôleur aérien autonome : une approche approfondie d'apprentissage par renforcement multi-agents.* arXiv :1905.01303.
6. Bertram, J. et Wei, P. (2020). *Guide informatique distribué pour la mobilité aérienne urbaine à haute densité.* AIAA Aviation Forum.
7. Valavanis, KP et Vachtsevanos, GJ (Eds.). (2015). *Manuel des véhicules aériens sans pilote.* Springer.
8. Auggliaro, F., Schoellig, AP et D'Andrea, R. (2012). *Génération de trajectoires sans collision pour une flotte de quadricoptères.* IEEE/RSJ IROS, 3977-3982.