---
title: "Papier C Planification de la recherche v2 : Reconstruction de la détection active des drones à basse altitude et planification pour la soumission à la revue T-ITS/TR-C"
description: "La v1 positionne RA-L pour une publication rapide, et l'enseignant exige qu'il soit publié en premier. Cet article repositionne le travail du FIM-3DGS en tant que technologie habilitante pour l'économie à basse altitude/le trafic aérien urbain, et précise dans la compilation du 23/05/2026 qu'il sera actuellement reporté et réservé en tant qu'orientation technologique habilitante de détection active."
pubDate: 2026-05-15
updatedDate: 2026-05-23
tags: ["Planification de thèse", "Soumission au journal principal", "T-ITS", "TR Partie C", "économie de basse altitude", "perception active", "3DGS", "drone", "Informations sur les pêcheurs"]
category: Tech
sourceHash: "05230e9d3f9c4368c3f98b16b9aca758a7d23ecc"
---

# Paper C v2 : Repositionnement de RA-L vers Top Issue

> **v1 → Modifications fondamentales dans la v2 :** L'enseignant a demandé le numéro principal. La v1 était initialement positionnée comme IEEE RA-L (IF 4.6 Q2, version rapide), et est maintenant mise à niveau vers une stratégie parallèle d'**investissement principal IEEE T-ITS (IF 8.5 Q1) + investissement de sauvegarde TR Part C (IF 8.5 Q1)**. Cela n’est pas aussi simple que de changer de revue : la position du problème, la conception expérimentale, les indicateurs d’évaluation et la structure de longueur de l’ensemble du manuscrit doivent être reconstruits. Cet article est le document de conception complet de cette refactorisation.

---

## 0. Principales différences entre la v1 et la v2

| Dimensions | v1 (RA-L 8 pages) | v2 (T-ITS/TR-C pages 20 à 25) |
|------|--------------|----------------------------|
| **Positionnement de base** | Algorithme de détection active/reconstruction 3D | Technologie habilitante économique à basse altitude/Système de trafic aérien urbain |
| **Lecteurs cibles** | Robotique / CV Boursiers | Chercheurs en systèmes de transport intelligents/ingénierie du trafic |
| **Énoncé du problème** | Comment sélectionner de manière optimale les points de vue pour reconstruire des scènes 3D | Comment permettre aux drones d'effectuer des tâches de transport en toute sécurité et efficacement à basse altitude dans les villes |
| **Indicateurs clés** | PSNR / SSIM / Couverture | Taux de réussite des missions / Taux d'utilisation de l'espace aérien / Marge de sécurité / Consommation d'énergie unitaire Volume de service |
| **Méthode de référence** | FisherRF / GauSS-MI et autres méthodes de détection | Méthode de détection + méthode de planification industrielle des drones + comparaison de simulation ITS |
| **Scénario expérimental** | Tâche de reconstruction unique | Multi-task long-term operation (delivery, inspection, emergency) |
| **Profondeur théorique** | Dérivation de la formule FIM | FIM + théorie des files d'attente du système + prouvabilité des contraintes de sécurité |
| **longueur** | 8 pages | 20-25pages |
| **Heure de soumission** | 2026/09 | 2027/03-06 |

**Pourquoi cette reconstruction est raisonnable (non forcée) :** Le noyau technique du Paper C (détection active FIM-3DGS) lui-même est la technologie clé de goulot d'étranglement pour le fonctionnement autonome des drones, et il n'est pas emballé de force dans le but de délivrer des T-ITS. Mais la v1 ne place pas cette technologie dans le contexte d’un système de transport – la v2 remplit cette couche.

### 0.1 2026-05-23 Faire le ménage : priorités et limites actuellesLe papier C reste une direction de détection active précieuse, mais il ne devrait pas pour le moment rivaliser avec G1, B et F-J1 pour les ressources principales récentes. La raison en est qu'il prouvera simultanément la valeur des algorithmes de détection active 3DGS, de la planification de la sécurité des drones et des systèmes de transport, et que la surface de travail sera plus grande que prévu dans la première version.

Il est actuellement recommandé de positionner le papier C comme **direction de réserve P3** :

| Projet | En cours de traitement |
|------|----------|
| Contribution principale | Sélection active du point de vue FIM-3DGS + contraintes de sécurité des drones |
| Connexion de transport | Uniquement réservé aux technologies habilitantes pour l'inspection, les interventions d'urgence et la distribution, sans rédiger au préalable un document complet sur le système TR-C |
| Doit être amélioré | Real/public urban 3D data, strong NBV baseline, task-level indicators, reproducible simulation |
| Contenu en attente | Débit du système multi-UAV, récit de politique économique à basse altitude, grand système SUMO-AirSim complet |
| Conditions de récupération | La chaîne d'outils G1 est stable, la plate-forme de scène F peut être réutilisée ou il existe suffisamment d'actifs expérimentaux 3DGS/détection active |

S'il doit être redémarré à l'avenir, le premier article devrait être basé sur la **norme de papier de méthode T-RO / T-ITS** pour peaufiner la technologie de détection active et confirmer que les indicateurs techniques sont tenables ; Ce n'est que lorsque l'expérience peut prouver qu'elle peut améliorer considérablement l'efficacité des tâches et les indicateurs de sécurité d'inspection/d'urgence/de livraison qu'elle peut passer au document du système TR-C.

---

## 1. Repositionnement : de « l’algorithme de perception » à la « technologie économique habilitante à basse altitude »

### 1.1 Contexte stratégique (doit ouvrir la voie lors de la rédaction)

**Niveau politique national (2024-2025) :**
- Plan de développement économique des basses altitudes du « 14e plan quinquennal » de la Chine : l'objectif d'échelle économique des basses altitudes est de 2,5 billions en 2025 et atteint 5 billions en 2030
- Le « Plan global de planification du réseau de transport tridimensionnel national » de l'Administration de l'aviation civile de Chine : clarifie les drones à basse altitude en tant qu'infrastructure de transport urbain
- Projets pilotes économiques à basse altitude à Shenzhen, Guangzhou, Hefei et d'autres villes en 2024

**Défi académique (le problème fondamental à résoudre dans l'article) :**
- Les drones à basse altitude doivent résoudre trois problèmes fondamentaux lorsqu'ils entrent dans les villes :
  1. **Efficacité de l'utilisation de l'espace aérien :** Une ville doit accueillir des milliers de drones opérant en même temps
  2. **Garantie de sécurité de fonctionnement :** Aucune collision avec des bâtiments, des foules et d'autres avions
  3. **Boucle fermée perception-décision :** Les drones doivent acquérir une compréhension de l'environnement environnant en temps réel pour prendre des décisions sûres
- Ces trois enjeux sont couplés les uns aux autres : la qualité perçue détermine la fiabilité décisionnelle, et la fiabilité décisionnelle détermine la faisabilité du dispatching de l'espace aérien.**Positionnement de cet article :** La troisième question (boucle fermée perception-décision) est à la base des deux premières questions. Cet article propose **FIM-3DGS : un cadre de détection et de planification active d'UAV basé sur l'information** pour améliorer fondamentalement l'efficacité de détection et la sécurité opérationnelle d'un seul UAV dans l'environnement urbain, prenant ainsi en charge la gestion à grande échelle de l'espace aérien à basse altitude.

### 1.2 Dialogue avec les principaux articles de revues existants

**Articles connexes récemment acceptés dans TR Part C (2023-2025) :**

| Papier | Sujet | Relation avec cet article |
|------|------|------------|
| Mohamed et coll. 2024 | "Conception d'un réseau de livraison du dernier kilomètre assisté par drone" | En supposant une perception parfaite, nous complétons la couche de perception |
| Liu et Tang 2023 | "Planification de trajectoire de drone pour la livraison de colis en milieu urbain" | En utilisant la planification de chemins géométriques, nous fournissons une boucle fermée de planification de détection |
| Parc et coll. 2024 | « Planification de vertiports pour les opérations UAM » | Planification au niveau micro, nous fournissons une technologie habilitante autonome |
| Chen et coll. 2025 | "Évaluation des risques liés aux drones à basse altitude dans les villes" | Évaluation des risques, notre perception peut fournir des données pour l'évaluation des risques |

**Documents liés à l'IEEE T-ITS (2023-2025) :**| Papier | Sujet | Relation avec cet article |
|------|------|------------|
| Wang et coll. 2024 | "Multi-UAV trajectory optimization in urban environments" | Focus on the path, without considering the impact of perceived uncertainty |
| Zhang et coll. 2023 | "Perception coopérative air-sol pour l'UAM" | Fusion multi-capteurs, notre cadre FIM peut être utilisé comme base pour le calcul du poids de fusion |
| Kim et coll. 2025 | "Information-theoretic active mapping for autonomous vehicles" | Détection active AV au sol, nous sommes la version UAV et ajoutons des contraintes de sécurité |

**Points chauds du papier partagés par T-ITS et TR-C :**
- Mobilité Aérienne Urbaine (UAM)
- Logistique drones basse altitude
- Transport multimodal (y compris drone)
- Autonomous driving perception (analogous to migration to UAV)
- Évaluation des risques liés à l'utilisation de l'espace aérien

### 1.3 Titre et résumé repositionnés

**Titre v2 (chinois et anglais) :**

- **Chinois :** Détection active et planification basées sur l'information pour l'économie urbaine à basse altitude : cadre habilitant 3DGS pour le fonctionnement autonome des drones
- **Anglais：** Perception active et planification basées sur l'information pour une économie urbaine à basse altitude : un cadre permettant les éclaboussures gaussiennes 3D pour les opérations de drones autonomes

**v2 abstract (350 words in English, corresponding to the abstract length of the top issue):**> Urban low-altitude UAV operations—including last-mile delivery, infrastructure inspection, and emergency response—face a fundamental challenge: dense urban environments demand high-quality 3D perception for safe autonomous decisions, yet traditional perception pipelines either lack accuracy (occupancy grids) or fail real-time constraints (NeRF). Cet article présente **FIM-3DGS**, un cadre de perception et de planification active basé sur l'information qui comble cette lacune. We derive a closed-form Fisher Information Matrix (FIM) formulation for 3D Gaussian Splatting (3DGS) primitives, providing the first rigorous Cramér-Rao-bound-based view selection criterion for explicit neural rendering representations. A Rendering Variance Proxy reduces the FIM computation from $O(N|P|D^2)$ to $O(N)$, enabling real-time (<20 ms) next-best-view decisions for 100,000+ Gaussians. Nous intégrons davantage les contraintes de sécurité de la fonction de barrière de contrôle (CBF) avec 6-DoDynamique du drone F, offrant un fonctionnement prouvé sans collision. Des expériences de simulation complètes sur MatrixCity (ensemble de données à l'échelle urbaine) et un jumeau numérique AirSim personnalisé démontrent que FIM-3DGS atteint un PSNR 1,8 dB plus élevé et une couverture 8,2 % plus élevée que le GauSS-MI de pointe (RSS 2025), tout en réduisant le temps d'achèvement de la mission de 27 % sur trois études de cas de systèmes de transport : inspection des bâtiments, livraison de colis et intervention d'urgence. **Du point de vue ITS**, notre cadre réduit l'utilisation de l'espace aérien par tâche de 31 % et améliore le débit multi-UAV de 22 % lorsqu'il est intégré aux systèmes de planification UAM existants. Du code et des ensembles de données seront publiés pour soutenir les futures recherches sur l’économie à basse altitude.**Conseils de rédaction clés :**
- La première phrase situe immédiatement le problème à « application de transport » (livraison/inspection/urgence)
- Conserver les contributions techniques au milieu (dérivation FIM, complexité, CBF)
- Le dernier paragraphe met l'accent sur les "indicateurs au niveau du système" (durée d'achèvement de la mission, utilisation de l'espace aérien, débit UAM) - c'est ce qui préoccupe le plus les réviseurs des T-ITS/TR-C.
- Mentionner le code/l'ensemble de données comme open source (principale tendance de publication à améliorer la reproductibilité)

---

## 2. Questions de recherche recadrées

### 2.1 Énoncé du problème au niveau du système (nouveau dans la v2)

**Problèmes macro :** Dans la perspective d'une échelle économique à basse altitude de 5 000 milliards en 2030, une ville de taille moyenne (5 millions d'habitants) doit effectuer environ 100 000 opérations de drones chaque jour (voir l'extrapolation des données du pilote de livraison sans pilote Meituan/JD). Cela nécessite que chaque drone :

1. **Perception précise :** Maintenez une représentation 3D au niveau centimétrique en temps réel dans des environnements inconnus ou en évolution dynamique
2. **Fonctionnement efficace :** Un seul drone maximise le volume de tâches sous une puissance limitée
3. **Certification de sécurité :** La distance par rapport aux bâtiments, aux piétons et aux autres drones est strictement conforme aux règles de sécurité

**Décomposition des sous-problèmes :**

| Sous-problèmes | Solutions existantes | Limites | Contributions à cet article |
|--------|----------------|--------|---------|
| Q1 : Comment reconstruire un environnement urbain dynamique et de haute qualité ? | NeRF / Grille d'occupation hors ligne | Lent / Rugueux | 3DGS en ligne + détection active |
| Q2 : Comment décider où faire voler le drone ensuite ? | Itinéraires prédéfinis/planification de chemin géométrique | Ne tient pas compte de l'incertitude perçue | NBV basée sur l'information FIM |
| Q3 : Comment s’assurer que les décisions sont conformes aux règles de sécurité ? | Détection de collision post-traitement | Réactif, manque de garanties | Contraintes de sécurité intégrées CBF |
| Q4 : Comment évaluer la valeur du système pour le transport urbain ? | Expérience à tâche unique | Absence d'évaluation multi-tâches à long terme | Évaluation au niveau du système de trois scénarios majeurs |

### 2.2 Problèmes d'optimisation du point de vue des ITS (nouveau dans la v2)**Optimisation au niveau de la mission d'un seul drone (une mission) :**
$$\max_{\mathbf{v}_{1:T}}\; \alpha\,\underbrace{Q_{rec}(\boldsymbol{\Theta})}_{\text{Qualité de la reconstruction}} + \beta\,\underbrace{Q_{tâche}(\mathbf{v}_{1:T})}_{\text{Achèvement de la tâche}} - \gamma\,\underbrace{E(\mathbf{v}_{1:T})}_{\text{Consommation d'énergie}}$$

Contraintes : dynamique du drone + sécurité CBF + contraintes de mission (zones incontournables) + budget puissance

**Évaluation du niveau du système ITS (Multi-Task Multi-UAV) :**
$$\Phi_{ITS} = \frac{\sum_k S_k^{succès}}{\sum_k T_k^{vol}\cdot E_k}$$

Parmi eux, $S_k^{success}$ est le taux de réussite de la tâche $k$, $T_k^{flight}$ est le temps de vol et $E_k$ est la consommation d'énergie. Cette métrique mesure le rendement des tâches par unité de ressources (temps + énergie) et constitue une métrique système standard dans la littérature ITS.

**Points d'innovation clés :** Les recherches existantes sur les drones optimisent généralement les indicateurs au niveau d'une tâche unique (tels que le délai de livraison), mais le débit au niveau du système doit être optimisé du point de vue des STI. Cet article montre : En introduisant la détection active, l'incertitude de la perception d'une seule machine est réduite → la prise de décision est plus radicale et toujours sûre → l'efficacité des tâches sur une seule machine est améliorée → le débit au niveau du système est naturellement amélioré.

---

## 3. Trois études de cas majeures (nouveau contenu de base v2)

> La question qui préoccupe le plus les principaux critiques de revues : quel impact l'algorithme aura-t-il sur les problèmes réels de trafic ? La v2 trouve une réponse à travers trois cas spécifiques.

### Cas 1 : Inspection des structures des bâtiments urbains (Inspection des infrastructures)

**Paramètres de scène :**
- Mission : UAV inspectant les fissures de façade/éléments desserrés d'un immeuble de bureaux de 30 étages
- Entrée : localisation GPS du bâtiment + paramètres d'apparence approximatifs
- Sortie : modèle 3DGS complet + annotation de défauts (connecté en aval de ce travail)**Mesures d'évaluation (perspective ITS) :**
- **Taux de couverture des inspections :** La proportion d'observations effectives de la surface du bâtiment réalisées (liée à la qualité de la reconstruction)
- **Durée de vol d'inspection unique :** Le nombre de minutes nécessaires pour effectuer une inspection complète
- **Taux de réinspection :** La proportion de vols qui doivent être réexécutés en raison d'une qualité perçue inférieure aux normes
- **Consommation d'énergie :** Consommation d'énergie pour une seule inspection (affecte le nombre de bâtiments pouvant être inspectés en une journée)

**Par rapport à la référence (pratique de l'industrie) :**
1. **Numérisation sur tondeuse à gazon (grand public de l'industrie) :** Itinéraire de numérisation rectangulaire fixe, pratique standard des solutions commerciales DJI et Skydio
2. **Planification manuelle des points de cheminement :** Les ingénieurs définissent manuellement les points d'intérêt
3. **FisherRF/GauSS-MI :** SOTA académique
4. **FIM-3DGS (cet article)**

**Résultats attendus :**
- Temps de travail vs tondeuse à gazon : réduit de plus de 30 % (informations pilotées pour éviter les observations répétées)
- Taux de revérification : réduit de 15 % à <3 %

### Cas 2 : Livraison du dernier kilomètre

**Paramètres de scène :**
- Mission : Livraison de drones du site de livraison au balcon du client
- Challenge : Occlusion complexe de bâtiments entre canyons urbains + obstacles dynamiques (interrupteurs de fenêtres, poteaux de séchage de vêtements, etc.)
- Saisie : GPS du point de départ, GPS du point final, description approximative de l'emplacement du client
- Résultat : livraison réussie + carnet de vol complet

**Indicateurs d'évaluation :**
- **Taux de réussite de livraison :** Taux de réussite des colis livrés au balcon du client (KPI principal)
- **Délai moyen de livraison :** Du départ à la livraison
- **Marge de sécurité au niveau de la tâche :** Statistiques de distance minimale par rapport aux obstacles pendant tout le processus de livraison
- **Occupation de l'espace aérien :** volume de l'espace aérien 3D occupé par une seule livraison (affecte la densité de répartition multi-UAV)

**Comparer à la référence :**
1. **Itinéraires prédéfinis + évitement d'obstacles réactif : ** Solutions grand public de Wing/Meituan et d'autres sociétés
2. **A* Planification d'itinéraire + Carte raster d'occupation :** Comparaison académique
3. **Détection collaborative multi-robots (A2X) :** Utilisation d'autres données de drones
4. **FIM-3DGS (cet article)**

**Résultats attendus :**
- Taux de réussite de livraison : de 85 % (itinéraire prédéfini) → 96 % (détection active)
- Occupation de l'espace aérien : réduction de 31 % (la perception de la précision permet des couloirs de vol plus étroits)

### Cas 3 : Intervention d'urgence urbaine (Intervention d'urgence)**Paramètres de scène :**
- Mission : Après le déclenchement d'un incendie dans un immeuble, le drone a dessiné un modèle 3D du bâtiment en 60 secondes pour le commandement de sauvetage
- Défi : environnement totalement inconnu + occlusion de fumée + exigences de rapidité extrêmement élevées
- Entrée : Localisation de l'alarme incendie
- Résultat : Construction du modèle 3DGS + identification de la zone affectée

**Indicateurs d'évaluation :**
- **Couverture en 60 secondes :** Proportion d'observations de surfaces de bâtiments réalisées sous des contraintes de temps strictes
- **Vitesse d'identification des zones critiques :** Temps nécessaire pour détecter la source d'incendie/la voie d'évacuation
- **Taux de collision nul :** Capacité de vol sûre dans des environnements complètement inconnus

**Comparer à la référence :**
1. **Exploration des frontières :** Méthode d'exploration classique
2. **GauSS-MI :** SOTA le plus pertinent
3. **FIM-3DGS (cet article)**

**Résultats attendus :**
- Couverture années 60 : de 70% (Frontier) → 88% (FIM-3DGS)
- Taux de collision zéro : 100 % (garantie CBF)

---

## 4. Mise à niveau de la conception expérimentale (v2 considérablement étendue)

### 4.1 Plateforme de simulation

AirSim + Unreal Engine 5 + Isaac Sim conservés de la v1, nouveau :

**Simulation conjointe SUMO + AirSim (nouveau dans la v2) :**
- SUMO assure l'environnement de transport terrestre (piétons, véhicules)
- AirSim fournit une simulation de drone
- Simulez l'environnement de transport multimodal de villes réelles grâce au pontage ROS2
- Il s'agit de la capacité de « simulation au niveau du système » que les évaluateurs du T-ITS apprécieront

### 4.2 Ensemble de données (extension v2)| Ensemble de données | Source | Utilisation | v1/v2 |
|--------|------|------|------|
| MatriceCity | ICCV 2023 | Test de Master en Réaménagement Urbain | Disponible dans les deux éditions |
| ScanNet v2 | CVPR 2017 | Vérification du développement en interne | Les deux versions disponibles |
| **Ensemble de données de livraison d'UAV** | Auto-construit (nouveau dans la v2) | Évaluation au niveau des tâches de scénarios de livraison réels | v2 uniquement |
| **Vertiport-Sim-Données** | Auto-construit (nouveau dans la v2) | Scénario de décollage et d'atterrissage multi-UAV | v2 uniquement |
| **Suite-Inspection-Urbaine** | Coopération avec Skydio/DJI ou données open source | Évaluation standardisée des tâches d'inspection | v2 uniquement |

**Plan de construction de l'ensemble de données de livraison d'UAV :**
- Construire 5 scénarios types de distribution urbaine dans AirSim (CBD, zones résidentielles, zones industrielles, autour des hôpitaux, autour des écoles)
- 100 missions de livraison par scénario
- Étiquetage : point de départ, point final, vérité terrain 3D, chemin de livraison optimal, obstacles typiques
- Utilisé pour évaluer le taux de réussite de la livraison, le délai moyen et la marge de sécurité
- **Points bonus pour les évaluateurs des meilleures revues :** Ensemble de données auto-construit + open source = contribution académique accrue

### 4.3 Système d'indicateurs d'évaluation (v2 considérablement étendu)

**Couche 1 : Indicateur de qualité perçue (disponible en v1)**
- PSNR, SSIM, LPIPS, couverture, distance de chanfrein

**Couche 2 : Indicateur d'efficacité de la planification (disponible en v1)**
- Latence de planification, taux InfoGain, PSNR@budget

**Couche 3 : Indicateurs au niveau des tâches (nouveau dans la version v2)**
- **Taux d'achèvement des missions (MCR) :** Pourcentage de missions terminées avec succès
- **Temps de tâche par mission :** Temps d'achèvement moyen d'une seule tâche
- **Énergie par mission：** Consommation d'énergie pour une seule tâche
- **Taux de revol :** La proportion de revols dus à une perception insuffisante**Couche 4 : Indicateurs au niveau du système (nouveau dans la version 2)**
- **Utilisation de l'espace aérien :** Volume occupé par l'espace aérien 3D de la tâche unitaire (m³/tâche)
- **Débit multi-UAV :** Le nombre de tâches que N drones peuvent effectuer dans la même zone par unité de temps
- **Répartition de la marge de sécurité :** Répartition statistique de la distance jusqu'à l'obstacle le plus proche pendant toute la mission
- **Indice de risque cumulatif：** $\int \mathcal{R}(\boldsymbol{\xi}(t))\,dt$ Indice de risque cumulé

**Couche 5 : Indicateurs économiques (nouveau dans la v2, compatible TR-C)**
- **Coût par livraison réussie :** Le coût d'exploitation d'une seule livraison réussie (y compris la consommation d'énergie, la maintenance et les risques)
- **Densité de service :** Capacité de service au sein de la ville par unité de surface (tâche/km²·jour)

### 4.4 Méthode de base (v2 étendue à trois catégories)

**Classe A : Baseline de la méthode de perception (existant dans la v1)**
- FisherRF (ECCV 2024), GauSS-MI (RSS 2025), ActiveGS (T-RO 2024), GenNBV (CVPR 2024), Frontier, Random

**Classe B : Base de référence en matière de pratiques industrielles pour les drones (nouveau dans la version 2, requis pour T-ITS/TR-C)**
- **Scan de tondeuse à gazon :** balayage rectangulaire fixe, solution commerciale DJI
- **Waypoint pré-planifié :** L'ingénieur définit manuellement les points d'intérêt
- **A\* avec grille d'occupation :** Planification de trajectoire de drone classique

**Classe C : référence de base au niveau du système ITS (nouveau dans la version 2)**
- **DJI FlightHub 2 Simulation :** Modèles de prise de décision pour les systèmes de gestion de drones commerciaux
- **Planificateur de flotte centralisé :** Planification centralisée MILP, idéale mais lente
- **Aucune perception active :** Acceptation purement passive de l'itinéraire par défaut (comparaison v1 vs v2)

### 4.5 Expérience d'ablation (extension v2)| Ablation | Variantes | Validation |
|--------|------|------|
| Supprimer les contraintes de sécurité du CBF | FIM-3DGS-NoSafe | nécessité du CBF |
| Utiliser Shannon MI au lieu de FIM | MI-3DGS | Avantages théoriques FIM vs MI |
| Remplacement du 3DGS par NeRF | FIM-NeRF | Contribution en temps réel |
| Remplacement de l'approximation par FIM exact | FIM-3DGS-Exact | Précision approximative par rapport à la vitesse |
| **Supprimer les commentaires au niveau du système (nouveau dans la v2)** | FIM-3DGS-NoSystemLoop | Vérifier la valeur des commentaires au niveau des tâches |
| **Ne prend pas en compte les contraintes de consommation d'énergie (nouveau dans la v2)** | FIM-3DGS-NoEnergy | L'impact des contraintes de consommation d'énergie sur les indicateurs au niveau du système |

---

## 5. Déclaration d'innovation (reconstruction v2)

### Contribution 1 (la théorie, les T-ITS / TR-C sont tous concernés)

**Première dérivation d'expressions fermées de Fisher Information Matrix** pour les paramètres primitifs explicites du Splatting gaussien 3D**, prouvant une stricte équivalence avec les limites inférieures de Cramér-Rao.

Par rapport à l'entropie de Shannon de GauSS-MI (RSS 2025) :
- FIM fournit des **limites inférieures statistiques strictes** (CRB) pour la précision de l'estimation des paramètres, qui peuvent être directement converties en intervalles de confiance de reconstruction
-L'entropie de Shannon mesure uniquement le caractère aléatoire des observations et n'est pas directement liée à la précision de l'estimation des paramètres.
- Le critère D-optimalité (déterminant FIM) équivaut à minimiser l'erreur de reconstruction du volume de l'ellipsoïde

**Explication aux examinateurs ITS :** Cela équivaut à pousser le problème de détection active des drones de la conception empirique au niveau théorique de « l'optimalité prouvable », de sorte que les décisions au niveau du système en aval (telles que la planification multi-machines, l'attribution de l'espace aérien) puissent être basées sur des limites inférieures strictes d'incertitude de détection.

### Contribution 2 (méthode, interdisciplinaire)

**Proposition d'un cadre de planification de détection active en temps réel avec approximation légère du Rendering Variance Proxy (RVP) + contraintes de sécurité CBF** :- RVP réduit la complexité de calcul FIM de $O(N|P|D^2)$ à $O(N)$, obtenant une décision <20 ms à une échelle gaussienne de 100 000
- Contraintes de sécurité intégrées CBF, introduites à partir d'une théorie de contrôle de pointe avec des garanties zéro collision prouvables
- Le cadre global peut fonctionner sur NVIDIA Jetson Orin 16G pour répondre aux besoins d'un véritable déploiement aéroporté d'UAV

**Explication aux évaluateurs ITS :** Il s'agit d'une contribution d'ingénierie pratique - rendant possible le déploiement réel d'UAV pour la première fois dans une approche SOTA académique. Il s’agit d’une étape clé dans l’intégration de l’industrie et du monde universitaire.

### Troisième contribution (système, principal argument de vente du T-ITS/TR-C)

**Première évaluation au niveau du système de l'impact réel de la détection active sur le transport urbain par drone** :

- Trois études de cas majeures (inspection, distribution et urgence) couvrent les principaux scénarios d'application de l'économie à basse altitude
- Les indicateurs au niveau du système (MCR, utilisation de l'espace aérien, débit multi-UAV) quantifient l'impact des améliorations de la perception sur l'efficacité des transports
- Fournir des ensembles de données open source tels que UAV-Delivery-Dataset pour soutenir les recherches ultérieures sur les STI

**Explication aux évaluateurs ITS :** Il ne s'agit pas d'un autre document de perception - il s'agit du travail consistant à intégrer la technologie de perception dans le cadre d'évaluation ITS, en quantifiant la chaîne causale de « l'amélioration de la perception de 1 dB PSNR » à « l'amélioration du débit de l'espace aérien de X % ».

---

## 6. Différences avec le journal principal SOTA (extension v2)

### 6.1 Comparaison approfondie avec GauSS-MI (RSS 2025)

| Dimensions | GauSS-MI | FIM-3DGS v2 |
|------|----------|-------------|
| Mesure d'information | Entropie de Shannon | Informations sur les pêcheurs (équivalent CRB) |
| Base théorique | Limite supérieure de la théorie de l'information | Limite inférieure stricte de l'estimation statistique |
| Complexité informatique | O(N·MC) | O(N) (approximation RVP) |
| Dynamique des drones | Aucun | 6-DoF SE(3) |
| Contraintes de sécurité | Aucun | Garanties explicites du CBF |
| Scène expérimentale | Bureau/intérieur | Niveau de la ville + trois cas |
| **Couche d'application** | **Reconstruire la qualité** | **Reconstruire + Tâche + Système** |

### 6.2 Différences par rapport à la recherche existante sur les drones dans ITS (nouveau dans la v2)| Papier ITS | Sujet | Limites | Améliorations v2 |
|---------|------|------|--------|
| Mohamed et coll. 2024 (TR-C) | Conception d'un réseau de livraison de drones | En supposant la perfection de la perception | Modélisation de l'incertitude de perception réelle |
| Wang et coll. 2024 (T-ITS) | Optimisation de trajectoire multi-UAV | Ne prend pas en compte la perception en ligne | Boucle fermée perception-décision |
| Parc et coll. 2024 (TR-C) | Planification des vertiports | La connaissance d'une seule machine n'est pas modélisée | La connaissance d'une seule machine fournit des données pour la planification multi-machines |

---

## 7. Stratégie de soumission (mise à jour principale v2)

### 7.1 Chemin de soumission parallèle

```
2027/03  完成稿件 + 内部 review
            ↓
2027/04  投稿 IEEE T-ITS（首选）
            ↓
       ┌──────┴──────┐
       │             │
   接受/小修      拒稿/大修
       │             │
   2027/10 接受   重新调整框架
                    ↓
                改投 TR Part C
                （强调运输系统价值）
                    ↓
                2027/08 投稿
                    ↓
                2028/02 接受
```

**Stratégies clés :** Le contenu de base du manuscrit (80 %) est commun aux deux revues, avec des ajustements uniquement apportés au cadrage (10-15 %) et à certaines sections spécifiques à l'ITS (5-10 %). De cette façon, un seul écrit peut servir à deux candidats.

### 7.2 Différences subtiles entre T-ITS et TR-C (faites attention lors de l'écriture)

| Dimensions | IEEE T-ITS | TR Partie C |
|------|-----------|--------------|
| Points clés | Algorithme + application ITS | Implications système + politique |
| Style abstrait | Orienté technologie | Orienté application et impact |
| Préférence expérimentale | Simulation + analyse théorique | Simulation + étude de cas |
| Taux de littérature | 50 % algorithme/IA + 50 % ITS | 30% algorithme + 70% transport |
| Discussions | Limites de l'algorithme + travaux futurs | Implications politiques + impact sur l'industrie + limites |

**Stratégie d'écriture :** Le manuscrit principal est basé sur les préférences de T-ITS, et le modèle de résumé/introduction/discussion pour la version TR-C est préparé, et le changement de cadrage peut être effectué dans les 2 semaines.

### 7.3 Examiner les risques et les réponses| Commentaires potentiels sur l'examen | Réponse T-ITS | Réponse TR-C |
|------------|-----------|---------------|
| "La relation entre les algorithmes de perception et les ITS n'est pas forte" | Citant des précédents tels que Kim 2025 (TITS) | Souligner la valeur systémique des trois cas majeurs |
| "L'expérience manque de données réelles" | Accent sur les images réelles de MatrixCity + les ensembles de données auto-construits | Accent sur les décors de scènes réelles pour les études de cas |
| "Trop/pas assez de théorie" | Conserver la dérivation FIM, simplifier la preuve RVP | Simplifiez la formule FIM, mettez l'accent sur l'explication intuitive |
| « Pertinence insuffisante par rapport à la littérature existante sur les drones » | Ajouter une revue de la littérature ITS-UAV | Ajouter une revue de la littérature sur l'ingénierie des transports |
| "Aucune explication de la politique" | Brève citation politique | Focus sur la discussion des implications des politiques économiques de bas niveau |

---

## 8. Itinéraire d'exécution replanifié (chronologie v2)

### Diagramme de Gantt détaillé sur 15 mois

```
时间        阶段                                   关键交付物
──────────────────────────────────────────────────────────────────────────
2026/06    准备阶段
           • FIM-3DGS 核心算法实现（CUDA）
           • AirSim + SUMO 联合仿真平台搭建        ▶ 核心代码完成
           • MatrixCity 数据获取与预处理

2026/07    基础实验
           • 与 FisherRF/GauSS-MI/ActiveGS 集成测试
           • Layer 1/2 指标实验（PSNR、规划延迟）  ▶ 算法层实验完成

2026/08    案例研究 1：建筑物巡检
           • 在 AirSim 中搭建 30 层建筑场景
           • 100 次巡检任务实验
           • 与 Lawn-mower / 工业方案对比         ▶ 巡检案例完成

2026/09    案例研究 2：最后一公里配送
           • 自建 UAV-Delivery-Dataset
           • 5 城市场景 × 100 任务 = 500 次配送实验
           • 与预设航线 / A* 对比                  ▶ 配送案例完成

2026/10    案例研究 3：应急响应
           • 高楼火灾场景仿真
           • 60s 时间约束下的覆盖率实验            ▶ 应急案例完成

2026/11    多 UAV 系统级实验
           • SUMO + AirSim 联合仿真
           • 10/20/50 UAV 同时运行实验
           • 空域利用率 / 系统吞吐量评估           ▶ 系统级实验完成

2026/12    数据分析与初稿
           • 整合所有实验数据
           • 撰写 T-ITS 格式 22 页稿件
           • 内部 reviewer (导师 + 同门) 审阅      ▶ 初稿完成

2027/01    打磨阶段
           • 根据内部反馈大修
           • 英文润色（专业 editing service）
           • 准备补充材料（代码、数据集、视频）    ▶ 投稿准备就绪

2027/02    提交前检查
           • Cover letter 撰写
           • 期刊格式调整
           • 推荐审稿人列表准备

2027/03    ◉ 投稿 IEEE T-ITS              ──────────────────────────────

2027/03–08  Round 1 审稿（4-6 月）

2027/09    收到审稿意见
           • 若小修：1-2 月修改                    ▶ 接受目标 2027/12
           • 若大修：3-4 月补实验                  ▶ 接受目标 2028/03
           • 若拒稿：转 TR Part C，调整 framing    ▶ TR-C 投稿 2027/12

2028/06    最终发表（无论哪个期刊）              ◉ 最终目标
──────────────────────────────────────────────────────────────────────────
```

---

## 9. Évaluation des risques et alternatives

### 9.1 Principaux risques

**Risque A : rejet du T-ITS (probabilité ~50 %, taux normal)**
- Réponse : Cadrage ajusté et converti en TR Partie C
- Coût en temps : 6 mois supplémentaires
- Buffering : Préparez-vous au double cadrage depuis la v2

**Risque B : durée d'expérimentation insuffisante**
- Réponse : les expériences de base (couche 1-2 + cas 1-2) sont garanties, les cas d'intervention d'urgence peuvent être reportés
- Clé : La couche de perception + le dossier de livraison doivent être complets

**Risque C : écart de performances insuffisant entre l'algorithme et SOTA**
- Réponse : GauSS-MI est un nouveau travail en 2025, cet article devrait avoir un avantage de plus d'un an
- Buffering : les expériences d'ablation montrent des avantages théoriques, un nombre absolu de +1,5 dB suffit

**Risque D : période d'examen prolongée**
- Réponse : Sélectionnez la procédure accélérée avant la soumission (si fournie par la revue)
- Alternative : Préparer la version de la conférence et la soumettre à l'ICRA 2027 en même temps (pas de publication répétée, uniquement comme plan de secours)

### 9.2 Chemins de soumission alternatifs (par priorité)| Priorité | Revue | SI | Aptitude | Remarques |
|--------|------|----|---------| -----|
| **Préféré** | IEEE T-ITS | 8.5 | ★★★★★ | Principal objectif d'investissement |
| Alternative 1 | TR Partie C | 8.5 | ★★★★☆ | Rejeté puis transféré |
| Alternative 2 | IEEE T-RO | 7.4 | ★★★★☆ | Si ITS ne l'accepte pas, du contenu robotique pur |
| Alternative 3 | TR Partie B | 6.0 | ★★★☆☆ | Partiellement méthodologique, nécessite plus de théorie |
| Alternative 4 | Sciences des transports | 5.4 | ★★★☆☆ | Partiellement mathématique, nécessite une extension de la théorie des files d'attente |

---

## 10. Résumé : changements fondamentaux de la v1 à la v2

**Résumé en 1 phrase :** L'article C ne devrait plus être soumis à des conférences en tant que « article sur l'algorithme de perception », mais en tant que « recherche technologique économique à basse altitude » devant être soumis aux meilleures revues.

**3 différences clés :**
1. **Niveau de problème :** Tâche de reconstruction unique → Système de transport urbain
2. **Portée de l'évaluation :** Indicateurs de perception → Système d'indicateurs à cinq niveaux (perception/planification/tâche/système/économie)
3. **Dialogue académique :** Dialogue avec l'article de perception → Dialogue avec l'article de revue de premier plan sur le transport ITS/UAV

**5 nouvelles charges de travail importantes :**
1. Plateforme de simulation commune SUMO + AirSim
2. Trois études de cas majeures (inspection, distribution, intervention d'urgence)
3. Ensemble de données UAV-Delivery-Dataset auto-construit
4. Expériences multi-UAV au niveau du système (10 à 50 unités fonctionnant simultanément)
5. Manuscrit à double cadrage T-ITS / TR-C

**Coût en temps :** la v1 devrait prendre 4 mois, la v2 devrait prendre 12 à 15 mois (ce qui reflète raisonnablement la charge de travail liée à la soumission de manuscrits aux meilleures revues)

---

> **Description de l'itération du document :** Il s'agit de la version v2 du plan Paper C (`v2_20260515`). v1 (`v1_20260515`) est conservé comme archive historique pour enregistrer la conception du « Fast RA-L Path » pour une comparaison ultérieure. Conditions de déclenchement pour la prochaine mise à jour : ① Compléter les données expérimentales du 2026/08 ② Recevoir les commentaires de révision du T-ITS, puis mettre à jour vers la v3.