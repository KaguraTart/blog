---
title: "Paper F Paper Group Planning v1 : génération de scénarios critiques pour la sécurité des drones, couverture et application d'urgence"
description: "Plusieurs itinéraires papier sont prévus pour la génération de scènes critiques pour la sécurité des drones, la couverture de scènes, la corrélation de scènes ville-locale et les instructions d'allocation des ressources de sauvetage d'urgence à grande vitesse."
pubDate: 2026-05-19
tags: ["Papier F", "drone", "génération de scène", "couverture de la scène", "Critique pour la sécurité", "tests accélérés", "secours d'urgence", "TR-C", "T-ITS"]
category: Tech
sourceHash: "e8583115b5944094ad19a72285ecf76f319d06d8"
---

# Paper F Paper group planning v1 : Génération de scénarios critiques pour la sécurité des drones, couverture et application d'urgence

> Jugement global : en plus de la planification du système sur cent niveaux du papier B, de la détection active 3DGS/FIM du papier C et de la planification linguistique LLM/LTL du papier E, vous pouvez également ouvrir une ligne de papier séparée pour l'**ingénierie de scénarios critiques pour la sécurité des drones**.  
> L'essentiel de cette ligne n'est pas de créer un autre algorithme d'évitement d'obstacles, mais de répondre : **Comment générer, couvrir, filtrer et réutiliser systématiquement les scénarios clés de sécurité des drones afin que la formation, les tests, l'envoi d'urgence et les expériences papier ultérieurs disposent d'une base de scénarios crédible. **

---

## 1. Jugement global : Quelles autres instructions peuvent être écrites ?

Il existe actuellement plusieurs lignes de papier axées sur différentes problématiques :

| Ligne de thèse | Objets de base | Déjà couvert | Ne devrait pas être répété |
|--------|----------|--------------|----------|
| Papier B | Des centaines de flottes de drones | Ordonnancement hiérarchique à trois niveaux, théorie des files d'attente, Lyapunov, transport multimodal | Ne plus écrire séparément la planification de la flotte à grande échelle |
| Papier C | Détection active des drones | 3DGS, informations Fisher, prochaine meilleure vue | Ne se concentre plus sur la cartographie/sélection de perspectives |
| Papier E | Du langage à la planification | LLM, TaskIR, LTL/STL, vérification formelle | Ne se concentre plus sur la planification des tâches en langage naturel |
| Papier F | Ingénierie de scène | Génération de scènes, couverture, scènes dangereuses, applications d'urgence | Nouvelles orientations |

La valeur de Paper F est qu'il peut devenir l'**infrastructure expérimentale** pour les articles précédents :

- L'article B nécessite des besoins urbains et des scénarios d'urgence.
- Le papier C nécessite des scènes 3D locales contrôlables et une couverture d'occlusion/perspective.
- L'article E nécessite une sémantique des tâches, des entités cartographiques et des contraintes de sécurité.
- Paper F peut fournir une grammaire de scénario unifiée, une mesure de couverture, un score de criticité et un benchmark.Pour la direction « FENG SHOU » que vous avez mentionnée, il est recommandé que la norme corresponde au travail de génération de bibliothèque de tests/scénarios de test accélérés de conduite automatisée de **Shuo Feng**. L'idée centrale est la suivante : les événements critiques pour la sécurité sont extrêmement rares dans les données naturelles, nous ne pouvons donc pas nous fier uniquement à des tests aléatoires ordinaires, mais devons utiliser des méthodes de données pour construire des scénarios plus dangereux mais toujours raisonnables, accélérant ainsi les tests et la vérification de la sécurité [1] [2] [3]. Cette idée est très adaptée à la migration vers l'évitement d'obstacles par drones, le vol de canal à basse altitude et l'inspection d'urgence à grande vitesse.

---

## 2. Conception hiérarchique des groupes de papier

Il est recommandé de planifier l'épreuve F en 4 épreuves progressives :

| Niveau | Papier | Positionnement en une phrase | Priorité |
|------|------|------------|--------|
| F1 | Banc CovUAV | Référence de couverture des scènes critiques pour la sécurité des drones | Le plus haut |
| F2 | Tests accélérés guidés par la couverture | Algorithme de génération d'accélération de scènes dangereuses guidé par la couverture | Le plus haut |
| F3 | City2Local-UAV | Créer une génération de scène hiérarchique depuis l'ODD global de la ville jusqu'à la combinaison d'obstacles locaux | Moyen à élevé |
| F4 | Intervention d'urgence adaptée aux scénarios | Déploiement d'urgence collaboratif des ressources au sol d'UAV à grande vitesse du Shandong | Moyen à élevé |

Il est recommandé de faire **F1 + F2** en premier. F1 fournit l'ensemble de données, les métriques et la définition du problème, et F2 fournit les contributions algorithmiques. F3 et F4 peuvent être utilisés comme extensions : F3 transforme la référence en un système à l'échelle de la ville, et F4 transforme l'ingénierie de scène en une véritable activité d'urgence routière.

---

## 3. Contexte commun : Pourquoi la couverture de scène est le fondement de la recherche sur la sécurité des drones

Une faiblesse courante dans la recherche sur la sécurité des drones est la suivante : l’algorithme est magnifiquement conçu, mais le scénario expérimental est trop arbitraire. Ce n’est pas parce qu’un algorithme d’évitement d’obstacles réussit dans 20 scénarios manuels qu’il couvre les risques à longue traîne dans les opérations urbaines à basse altitude.Un consensus clair a été atteint dans le domaine de la conduite autonome : les accidents/quasi-accidents sur des routes réelles sont des événements rares, et s'appuyer directement sur des tests naturels serait extrêmement inefficace. Par conséquent, Shuo Feng et al. ont proposé un environnement naturaliste et contradictoire, utilisant la distribution naturelle pour maintenir l'authenticité et utilisant la distribution contradictoire pour augmenter la probabilité d'événements dangereux, accélérant ainsi les tests de conduite intelligents [1]. Ils ont en outre proposé de générer une bibliothèque de scénarios de test, en définissant la bibliothèque de scénarios de test sous ODD comme un ensemble de scénarios représentatifs et critiques, et en utilisant la criticité pour prendre en compte la fréquence d'exposition et le défi de manœuvre [2] [3]. L'examen de Ding et al. sur la génération de scénarios critiques pour la sécurité a également classé le domaine en trois types de méthodes : basées sur les données, contradictoires et basées sur les connaissances, et a souligné que la fidélité, l'efficacité, la diversité, la transférabilité et la contrôlabilité sont des défis majeurs [13].

Le scénario des drones nécessite encore plus cet ensemble d’idées pour quatre raisons :

1. **L'espace tridimensionnel est une dimension supérieure. **
   Les drones ne concernent pas seulement les voies plates, mais aussi l'altitude, le volume des obstacles, le champ de vent, la charge électrique, le champ de vision des capteurs et la dynamique de vol.

2. **Les événements dangereux sont plus difficiles à collecter. **
   Il existe très peu d’échantillons de collisions réelles avec des bâtiments, de collisions de lignes, d’entrée dans des zones d’exclusion aérienne, de traversées de ponts ou de scènes d’accidents à grande vitesse, et la formation ne peut pas s’appuyer sur des données réelles d’accidents.

3. **La génération aléatoire ordinaire gaspille de la puissance de calcul. **
   Un grand nombre de scénarios aléatoires sont soit trop simples, soit physiquement irréalisables, soit dangereux mais inévitables, ce qui les rend inefficaces pour la formation et l'évaluation.

4. **Il n’existe pas de mesure unifiée de la couverture des scènes. **
   Les articles existants sur les drones font souvent état du taux de réussite/taux de collision, mais indiquent rarement quelles combinaisons d'obstacles, géométries locales, difficultés de tâche et limites ODD sont couvertes par l'ensemble de tests.

Par conséquent, les questions scientifiques courantes pour l’article F sont :

> Comment construire un système de génération et d'évaluation de scénarios de drones qui soit réel, contrôlable et reproductible, et capable de couvrir efficacement les principaux risques de sécurité à longue traîne ?

---

## 4. Document F1 : Couverture des scènes critiques pour la sécurité des drones

### 4.1 Titre de la thèse**CovUAV-Bench : une référence axée sur la couverture pour les scénarios de navigation de drones critiques pour la sécurité**

### 4.2 Contexte

SafeBench fournit déjà une référence unifiée pour la sécurité dans la conduite autonome, intégrant plusieurs types de modèles de scène, d'algorithmes de génération de scène et d'indicateurs d'évaluation [5]. Scenic a également prouvé que l'utilisation de programmes probabilistes pour exprimer la distribution de scènes, les contraintes dures et les contraintes douces est une voie réalisable [4]. Des travaux préliminaires ont été réalisés sur la génération d'environnements de simulation d'UAV. Par exemple, Nakama et al. ont proposé un générateur automatisé d'environnement de simulation d'UAV [10]. FADS a également montré que les spécifications de sécurité de logique temporelle peuvent entrer dans le pipeline de sécurité des drones autonomes [11]. Cependant, dans le domaine des drones, il manque encore une référence orientée couverture pour l’évitement d’obstacles 3D, les couloirs à basse altitude, les espaces urbains locaux et les tâches d’urgence.

Le but de F1 n'est pas de proposer le planificateur le plus performant, mais de définir comment l'espace de la scène du drone est couvert par le système.

### 4.3 Méthode

Construire un espace de test de base de 50 mx 50 mx 50 m, en commençant par les scènes locales puis en l'étendant aux îlots urbains :- **Objets de scène** : blocs de construction, tours, fils, arbres, ponts, obstacles temporaires, drones dynamiques, véhicules terrestres, zones du personnel.
- **Structure spatiale** : espace ouvert, passage étroit, canyon urbain, sous-pont, zone d'atterrissage, accotement d'autoroute, zone d'accident.
- **Perturbation environnementale** : Vent, visibilité, bruit des capteurs, décalage GPS, retards de communication.
- **Type de tâche** : navigation point à point, passe d'inspection, vol stationnaire d'urgence, atterrissage, retour au domicile.
- **Format exécutable** : enregistrez-le sous `scenario.json` et ajoutez un adaptateur de simulateur. Il peut être converti ultérieurement en AirSim, Flightmare, PyBullet ou en simulation légère auto-construite.

La couverture de scène est définie comme :

$$
Couverture(S)=
\sum_{k=1}^{K} w_k \cdot
\frac{|B_k(S)|}{|B_k(\Omega)|},
$$

Où $\Omega$ est l'espace de scène discrétisé de l'ODD cible, $B_k(S)$ est le compartiment couvert par l'ensemble d'échantillons $S$ sur la $k$ème dimension d'attribut de classe, et $w_k$ est le poids de la dimension.

Les **76 millions d'explorations** existantes peuvent être écrites sous la forme "Actifs de journaux d'exploration existants" à des fins statistiques :

- Quelles combinaisons de scènes sont fréquemment explorées.
- Quelles combinaisons sont encore des trous de couverture.
- Quelles combinaisons déclenchent une collision / un quasi-accident / un délai d'attente.
- Quelles combinaisons sont des échantillons d'entraînement invalides.

Remarque : 76 millions d'explorations sont uniquement écrites comme « base expérimentale disponible » et ne peuvent pas être écrites comme des conclusions vérifiées.

### 4.4 Lignes de base| Référence | Objectif |
|--------------|------|
| Échantillonnage de scénarios aléatoires | La couverture de base la plus basique |
| Échantillonnage de grille | Discrétisation uniforme de l'espace des paramètres |
| Échantillonnage d'hypercube latin | Couverture des paramètres plus efficace |
| Échantillonnage contraint de style scénique | Base de référence pour la génération de scènes contraintes [4] |
| Suite de modèles de style SafeBench | Base de référence du scénario de sécurité modélisé [5] |

### 4.5 Points d'innovation

1. Proposer une taxonomie de couverture de scènes de drones : ODD, combinaison d'obstacles, perturbation dynamique, type de tâche, niveau de risque.
2. Donnez une référence axée sur la couverture au lieu de seulement quelques cartes manuelles.
3. Convertissez le journal d'exploration en trous de couverture et en graines de scénarios critiques.
4. Fournissez une interface de scène unifiée pour les documents B/C/E ultérieurs.

### 4.6 Comment évaluer

| Indicateur | Signification |
|------|------|
| Couverture des paramètres | Taux de couverture des bacs de paramètres |
| Couverture par paire/t-wise | Couverture combinée multi-attributs |
| Densité des scénarios critiques | Nombre de quasi-accidents/collisions découverts par budget de test unitaire |
| Taux de scénario invalide | La proportion de scénarios physiquement irréalisables ou dénués de sens pour la mission |
| Stabilité du classement du planificateur | Le classement de l'algorithme est-il stable sous différentes graines aléatoires |
| Reproductibilité de la relecture | Si le même résultat peut être reproduit avec la même graine |

### 4.7 Contributions recommandées

- Axe principal : papier orienté benchmark T-ITS / IEEE ITSC / IROS.
- Alternative : RA-L + ICRA, si le benchmark dispose à la fois d'outils open source de haute qualité et d'une vérification à petite échelle de drones réels.

---

## 5. Paper F2 : Accélérer la génération de scènes dangereuses guidée par la couverture### 5.1 Titre de la thèse

**Tests accélérés guidés par la couverture pour éviter les obstacles critiques aux drones**

### 5.2 Contexte

L'essentiel des tests accélérés pour la conduite autonome n'est pas de « créer des scénarios d'accident inévitables », mais d'améliorer l'efficacité de l'échantillonnage des événements critiques pour la sécurité tout en préservant l'authenticité et la possibilité d'action de la scène [1] [2] [3]. Si le scénario généré n’est réalisable pour aucun planificateur, il ne peut alors pas aider à différencier les capacités de l’algorithme ; si le scénario généré est trop sûr, il ne peut pas révéler les faiblesses du système.

La formation à l’évitement d’obstacles par drone présente également le même problème :

- Un grand nombre de scènes générées aléatoirement sans pression de sécurité.
- La génération de confrontations a tendance à produire des configurations d'obstacles qui ne peuvent être raisonnablement évitées.
- Le programme manuel a une couverture limitée et ne peut pas expliquer si les risques à longue traîne sont couverts.
- La formation RL gaspille du budget sur de nombreux scénarios invalides.

### 5.3 Méthode

**CGAT-UAV proposé : tests accélérés guidés par la couverture pour les drones**.

L'algorithme se compose de quatre modules :

1. **Encodeur de scénario**
   Encodez la scène en vecteurs structurés : nombre d'obstacles, largeur minimale du canal, direction cible, vitesse dynamique des obstacles, intensité du vent, bruit du capteur, marge de la batterie, etc.

2. **Mémoire de couverture**
   Maintenez les catégories de couverture, les types de pannes et les performances du planificateur pour les scènes explorées.

3. **Score de criticité**
   Référez-vous à l’idée de criticité de Feng et combinez le degré de risque avec la fréquence d’exposition [2] :

   $$
   Crit(s)=P_{\text{exposure}}(s)\cdot R_{\text{challenge}}(s)\cdot F_{\text{faisable}}(s).
   $$

   Parmi eux, $F_{\text{feasible}}(s)$ est utilisé pour punir les collisions inévitables et les scénarios physiquement déraisonnables.4. **Générateur adaptatif**
   Générez de nouvelles scènes dans les trous de couverture et les régions à haute criticité à l'aide des méthodes d'optimisation bayésienne, CMA-ES, d'édition RL ou d'entropie croisée.

### 5.4 Lignes de base

| Référence | Objectif de comparaison |
|----------|----------|
| Génération aléatoire | Taux d'accélération des tests |
| Grille / Échantillonnage d'hypercube latin | Efficacité de la couverture |
| Optimisation bayésienne | Recherche dangereuse de boîte noire |
| CMA-ES | Recherche paramétrique continue des dangers |
| Génération de scénarios contradictoires RL | Génération de risques d'apprentissage |
| Génération contrainte scénique | Génération de règles et de contraintes [4] |
| Génération guidée par la faisabilité de style FREA | Comparez l'idée d'« antagonisme raisonnable » [12] |

### 5.5 Points d'innovation

1. Migrer les tests accélérés de la conduite autonome vers l'évitement d'obstacles 3D par drone.
2. Optimisez simultanément la **couverture, la criticité et la faisabilité** pour éviter de rechercher uniquement le taux de collision.
3. Proposer un programme d'études guidé par la couverture pour former les planificateurs à des scénarios dangereux mais résolubles.
4. Le taux d'accélération du test est donné : le nombre de simulations nécessaires pour atteindre le même intervalle de confiance est considérablement réduit.

### 5.6 Comment évaluer| Indicateur | Signification |
|------|------|
| Facteur d'accélération | La réduction multiple du nombre de tests requis pour obtenir le même taux de découverte de pannes par rapport aux tests aléatoires |
| Taux de découverte d'échecs | Le rapport collision / quasi-accident / délai d'attente découvert par unité de budget |
| Criticité réalisable | Proportion de danger et stratégies réalisables pour éviter les obstacles |
| Score de naturalité | Si la scène est conforme à ODD avant |
| Gain de couverture pour 1 000 tests | Nouvelle couverture tous les 1000 tests |
| Efficacité de la formation | Après une formation avec des scénarios générés, amélioration du planificateur en test de tenue |

### 5.7 Contributions recommandées

- Ligne principale : AAAI/ICRA/IROS.
- Alternative : T-ITS, si l'accent est mis davantage sur les tests et l'analyse comparative de la sécurité routière.

---

## 6. Paper F3 : Génération hiérarchique de scènes globales urbaines vers des combinaisons d'obstacles locaux

### 6.1 Titre de la thèse

**City2Local-UAV : Génération de scénarios hiérarchiques depuis les ODD urbains jusqu'aux compositions d'obstacles locaux**

### 6.2 Contexte

F1 et F2 abordent un espace de test local en 3D, mais les véritables vols urbains à basse altitude ne sont pas des boîtes isolées. La raison pour laquelle une scène locale apparaît dépend de la structure globale de la ville : niveaux des routes, densité des bâtiments, zones fonctionnelles, ponts, zones de service, échangeurs, hôpitaux, écoles, zones d'exclusion aérienne et points d'urgence.

ASAM OpenODD/OpenSCENARIO fournit une idée standardisée depuis ODD, le domaine d'exploitation actuel jusqu'à la description de scénario exécutable [6] [7]. Le domaine des drones peut apprendre de ce niveau d’abstraction, mais devra intégrer des obstacles tridimensionnels, des contraintes d’espace aérien et une sémantique de mission à basse altitude.

### 6.3 Méthode

Proposer un pipeline de production à trois niveaux, de la ville au local :

```text
City-level ODD
  -> district / road / highway segment selection
  -> local 50m x 50m x 50m UAV test cell
  -> concrete obstacle composition
  -> simulator executable scenario
```

Modules spécifiques :- **City ODD analyseur** : extrayez la sémantique des villes/autoroutes d'OSM, des niveaux de route, des contours des bâtiments, des POI, des zones de service, des ponts et des entrées d'autoroute.
- **Échantillonneur de cellules locales** : sélectionnez des cellules locales typiques, telles que des canyons de grande hauteur, des zones de service, des viaducs, des gares de péage, des accotements d'autoroute et des zones de goulot d'étranglement accidentées.
- **Grammaire des obstacles** : utilisez des règles pour générer des combinaisons d'obstacles locaux, telles que des bâtiments + des câbles + des arbres + des véhicules garés + des zones réservées au personnel.
- **Contrôleur de couverture** : surveille la couverture de différentes zones fonctionnelles urbaines et combinaisons locales.

### 6.4 Lignes de base

| Référence | Objectif de comparaison |
|----------|----------|
| Génération locale aléatoire pure | Ne tient pas compte du contexte urbain |
| Conversion directe OSM vers carte | Convertit uniquement la carte, ne contrôle pas la couverture de la scène |
| Génération de jumeau numérique CARLA / OSM | Base de référence du jumeau numérique de conduite autonome au sol [14] |
| Modèles de scénarios manuels | Modèles de règles manuelles |
| CityEngine / génération procédurale de villes | Base de référence pour la génération procédurale de la ville |

### 6.5 Points d'innovation

1. Associer l’ODD urbain à la cellule locale de test de sécurité du drone.
2. Proposer une génération de scène hiérarchique de « sémantique globale de la ville -> composition d'obstacles locaux ».
3. Étendre la couverture de la scène des paramètres locaux à la couverture des zones fonctionnelles urbaines.
4. Soutenir des études de cas de villes réelles, telles que les principaux carrefours routiers de Jinan, Qingdao et Shandong.

### 6.6 Comment évaluer| Indicateur | Signification |
|------|------|
| Couverture IMPAIRE | Zones fonctionnelles urbaines, nivellement des routes, densité de construction |
| Diversité de la composition locale | Diversité des combinaisons d'obstacles locaux |
| Score de réalisme | Cohérence avec les statistiques OSM/POI/Building |
| Transférabilité | La politique générée à partir d'une ville est-elle toujours valable lorsqu'elle est déplacée vers une autre ville |
| Préservation de la criticité | La génération du contexte urbain préserve-t-elle les scènes locales à haut risque |

### 6.7 Contributions recommandées

- Ligne principale : TR-C, si l'accent est mis sur les systèmes de transport urbain, l'ODD, les infrastructures à basse altitude et les ensembles de données de scène.
- Alternative : T-ITS, si l'accent est mis sur l'interface de scénario de type OpenSCENARIO et l'évaluation intelligente du système.

---

## 7. Document F4 : Déploiement collaboratif de ressources drones au sol pour les secours d'urgence sur l'autoroute du Shandong

### 7.1 Titre de la thèse

**Allocation de ressources UAV au sol en fonction des scénarios pour les interventions d'urgence sur les autoroutes**

### 7.2 Contexte

Shandong Expressway dispose déjà d’une base commerciale pour l’inspection à basse altitude et les interventions d’urgence. Les informations publiques du Shandong Hi-Speed ​​​​Group montrent que son système complet de services de vols d'inspection a déployé des plates-formes sans surveillance et des drones industriels dans des zones clés pour les inspections de l'état des routes, les inspections routières, les interventions d'urgence et l'analyse des données [15]. Cela montre que les scénarios à grande vitesse ne sont pas une pure imagination, mais ont des entrées d'application.

Les recherches sur l'allocation des ressources d'urgence sur les autoroutes ont souligné qu'il existe encore plusieurs problèmes dans les travaux existants : sélection insuffisante du site des petites/micro installations d'urgence en bord de route pendant la phase d'exploitation, des informations complètes sont souvent supposées au début de l'accident mais ne sont pas réellement vraies, l'état du trafic après l'accident est incertain et variable dans le temps, et l'optimisation intégrée de la sélection du site des installations, de l'allocation des ressources et de la répartition est encore insuffisante [16]. Il y a eu des études sur le routage des drones sur les réseaux spatio-temporels dans la surveillance des incidents de circulation [17], ainsi que des études sur le déploiement en temps réel des drones et l'allocation des ressources dans les communications d'urgence en cas de catastrophe [18], mais elles n'ont pas encore formé une boucle fermée unifiée avec une couverture à grande vitesse des lieux d'accidents d'urgence, une valeur d'information de reconnaissance sur site et une allocation des ressources de sauvetage au sol.

Ceci est adapté à l'introduction du drone : le drone arrive d'abord sur les lieux de l'accident pour obtenir la situation, puis les ressources de garde au sol, de lutte contre l'incendie, de sauvetage et de contrôle sont réparties de manière dynamique.

### 7.3 Méthode**Répartition d'urgence au sol par drone en fonction des scénarios** :

- **Génération de scènes d'accident** : sur la base de la bibliothèque de scènes à grande vitesse F1/F3, le type d'accident, l'état du trafic, la météo, la géométrie des sections de route, les obstacles et les risques secondaires sont générés.
- **Couche de reconnaissance des drones** : les drones décollent des zones de service, des gares de péage ou des plates-formes sans pilote pour confirmer rapidement les emplacements des accidents, la longueur des embouteillages, les voies praticables et les risques liés aux matières dangereuses.
- **Couche d'allocation des ressources au sol** : répartit les dépanneuses, les pompiers, les ambulances, la police de la circulation, les véhicules de maintenance et les ressources de contrôle temporaires.
- **Modélisation de la valeur de l'information** : écrivez la réduction de l'incertitude de la reconnaissance des drones dans l'objectif de répartition, c'est-à-dire que le drone ne se contente pas de prendre des photos, mais réduit les faux délais de répartition et de réponse.
- **Optimisation continue** : les informations sur les accidents sont mises à jour au fil du temps et les stratégies de planification sont recalculées sur une base continue.

### 7.4 Formulation du problème

Supposons que l'ensemble de tronçons d'autoroute soit $\mathcal{L}$, l'ensemble d'accidents soit $\mathcal{I}(t)$, l'ensemble d'UAV soit $\mathcal{U}$, l'ensemble de ressources de sauvetage au sol soit $\mathcal{G}$ et l'ensemble de station-service/plate-forme sans surveillance soit $\mathcal{B}$.

Les variables de décision comprennent :

- Envoi d'UAV $x_{u,i}(t)$ : si l'UAV $u$ détecte l'incident $i$.
- Répartition des ressources au sol $y_{g,i}(t)$ : si la ressource $g$ se dirige vers l'incident $i$.
- Heure de décollage/départ $s_u(t), s_g(t)$.
- Action de mise à jour des informations $a_i(t)$ : s'il faut attendre une confirmation supplémentaire du drone ou envoyer directement.

Fonction objectif :

$$
\min
\mathbb{E}\gauche[
\beta_1 T_{\text{response}}+
\beta_2 T_{\text{autorisation}}+
\beta_3 C_{\text{dispatch}}+
\beta_4 R_{\text{secondaire}}+
\beta_5 U_{\text{incertitude}}
\droite].
$$

Parmi eux, $U_{\text{uncertainty}}$ représente l'incertitude des informations sur les accidents, qui peut être réduite par la reconnaissance par drone.

### 7.5 Lignes de base| Référence | Objectif de comparaison |
|----------|----------|
| Expédition au sol uniquement | Pas de reconnaissance par drone |
| Répartition de la ressource la plus proche | Ressources les plus proches en premier |
| Allocation d'installations statiques | Allocation d'installation fixe |
| Optimisation stochastique en deux étapes | Estimer l'accident avant expédition |
| Heuristique d'abord pour les drones | Reconnaissance par drone d'abord, puis répartition au sol |
| Optimisation continue tenant compte des scénarios | Méthode principale |

### 7.6 Points d'innovation

1. Connectez la couverture des lieux et la répartition des urgences à grande vitesse, au lieu de simplement allouer des ressources.
2. Modéliser la reconnaissance par drone comme une action décisionnelle qui réduit l’incertitude des informations sur les incidents.
3. Soutenir le contexte commercial réel de l'autoroute du Shandong : plate-forme sans surveillance, inspection de l'état des routes, intervention d'urgence et circulation des ordres de travail.
4. Optimisation unifiée du temps de réponse, du temps de dédouanement, du risque d'accident secondaire et des coûts d'expédition.

### 7.7 Comment évaluer

| Indicateur | Signification |
|------|------|
| Heure de première vue | Le moment où le drone a acquis pour la première fois les images de l'accident |
| Temps de réponse | Heure d'arrivée du premier lot de ressources de secours |
| Délai de dédouanement | Délai d'achèvement de l'intervention en cas d'accident |
| Taux d'expédition erroné | La proportion d'envois erronés, d'envois manqués ou de ressources insuffisantes |
| Risque d'accident secondaire | Indicateur de risque d'accident secondaire |
| Retard de congestion | Retard total causé par un accident |
| Valeur des informations sur les drones | La reconnaissance avec un drone réduit l'incertitude par rapport à sans drone |

### 7.8 Contributions recommandées

- Axe principal : TR-C en premier, car l'accent est mis sur l'exploitation du système de transport d'urgence à grande vitesse, l'allocation des ressources et la résilience du réseau de transport.
- Alternative : T-ITS, si l'accent est davantage mis sur les plateformes de drones, les communications, la reconnaissance vidéo, les systèmes d'ordres de travail et la répartition intelligente en ligne.

---

## 8. Unifier la plateforme expérimentale, les sources de données et les indicateurs d'évaluation

### 8.1 Plateforme expérimentale| Hiérarchie | Implémentation recommandée | Objectif |
|------|----------|------|
| Simulation légère | Python / PyBullet / grille 3D personnalisée | 76 millions de niveaux d'exploration rapide |
| Simulation de drone | AirSim, Flightmare | Vision, dynamique, vérification des capteurs [8] [9] |
| Langue du scénario | DSL de type scénique, schéma JSON | Génération de scènes reproductibles [4] |
| Données de la ville | OpenStreetMap, POI, niveaux de route, contours de bâtiments | Génération de la ville à la scène locale |
| Urgence à grande vitesse | Cas ouverts sur l'autoroute du Shandong, statistiques d'accidents, flux d'accidents synthétiques | Expérience d'allocation de ressources d'urgence |

L’expérience principale de F1/F2 devrait donner la priorité à la simulation légère pour garantir une exploration à grande échelle. AirSim/Flightmare est utilisé pour la vérification haute fidélité à petite échelle et n'est pas utilisé pour toutes les expériences.

### 8.2 Source de données

- **Référence de scénario de drone synthétique** : espace local généré de manière procédurale de 50 m x 50 m x 50 m.
- **Journaux d'exploration** : 76 millions de journaux d'exploration pour les trous de couverture et la taxonomie des échecs.
- **Données OSM/POI/bâtiment** : pour les zones fonctionnelles urbaines et les combinaisons de barrières locales.
- **Shandong Expressway Public Business Information** : utilisé pour l'arrière-plan de l'application et les hypothèses de déploiement [15].
- **Recherche sur la divulgation des ressources en matière d'accidents et d'urgence à grande vitesse** : utilisée pour les types d'accidents, les étapes d'allocation des ressources et les indicateurs d'évaluation [16].

### 8.3 Indicateurs unifiés| Groupe d'indicateurs | Indicateur |
|--------|------|
| Couverture | couverture des paramètres, couverture t-wise, couverture ODD, gain de couverture |
| Sécurité | taux de collision, taux de quasi-accidents, distance minimale, violation de contrainte |
| Génération de danger | criticité, taux de découverte de pannes, facteur d'accélération, criticité réalisable |
| Valeur de la formation | efficacité de l'échantillon, taux de réussite, robustesse sous changement ODD |
| Valeur d'urgence | délai de première visualisation, temps de réponse, délai de dédouanement, taux d'expédition erroné |

---

## 9. Chemin de soumission recommandé et priorité

### 9.1 La première étape : faites d'abord F1 + F2

Dans la première phase, il est recommandé d'écrire deux articles directement autour de « Couverture des scènes critiques pour la sécurité des drones + tests accélérés » :

1. **Document de référence F1**
   Plus stable, adapté comme base expérimentale pour tous les articles ultérieurs sur les drones. Même si l’algorithme n’est pas particulièrement puissant, il peut toujours être établi sur la base d’une taxonomie, d’une métrique de couverture, d’un ensemble de données et d’expériences reproductibles.

2. **Document de méthode F2**
   Contributions méthodologiques à AAAI/ICRA/IROS. Le point culminant est la migration des tests accélérés de conduite autonome de Shuo Feng vers des scènes 3D de drones et l'ajout d'une criticité réalisable guidée par la couverture.

### 9.2 Phase 2 : Refaire F3 + F4

F3 et F4 sont plus adaptés à l'avancement une fois que F1/F2 dispose d'une base d'outils :

- **F3** Pour résoudre la relation entre la ville dans son ensemble et les scènes locales, vous pouvez voter pour TR-C / T-ITS.
- **F4** Pour les applications de sauvetage d'urgence sur l'autoroute du Shandong, TR-C peut être sélectionné, mettant l'accent sur les opérations de transport et les interventions d'urgence.### 9.3 Relation avec les lignes papier existantes

| Papier | Comment soutenir Paper F |
|------|--------------------|
| Papier B | Fournit des scénarios de demande d'urgence de pointe/choc/autoroute |
| Papier C | Fournit une occlusion 3D locale, une couverture de perspective et une reconstruction de scènes difficiles |
| Papier E | Fournit des tâches en langage naturel, cartographie des entités et des scénarios de contraintes de sécurité |

Le document F est le mieux adapté pour servir de « document d’infrastructure de scénario » pour l’ensemble de la ligne de recherche sur les drones.

---

## 10. Références

[1] Shuo Feng, Xintao Yan, Haowei Sun, Yiheng Feng et Henry X. Liu. « Test d'intelligence de conduite intelligente pour les véhicules autonomes dans un environnement naturaliste et conflictuel. » *Nature Communications*, 12:748, 2021. DOI : 10.1038/s41467-021-21007-8. URL : <https://doi.org/10.1038/s41467-021-21007-8>

[2] Shuo Feng, Yiheng Feng, Chunhui Yu, Yi Zhang et Henry X. Liu. "Génération de bibliothèque de scénarios de test pour les véhicules connectés et automatisés, partie I : méthodologie." *Transactions IEEE sur les systèmes de transport intelligents*, 22(3):1573-1582, 2021. DOI : 10.1109/TITS.2020.2972211. URL : <https://doi.org/10.1109/TITS.2020.2972211>[3] Shuo Feng, Yiheng Feng, Haowei Sun, Shan Bao, Yi Zhang et Henry X. Liu. "Génération de bibliothèque de scénarios de test pour les véhicules connectés et automatisés, partie II : études de cas." *Transactions IEEE sur les systèmes de transport intelligents*, 22(9):5635-5647, 2021. DOI : 10.1109/TITS.2020.2988309. URL : <https://doi.org/10.1109/TITS.2020.2988309>

[4] Daniel J. Fremont, Tommaso Dreossi, Shromona Ghosh, Xiangyu Yue, Alberto L. Sangiovanni-Vincentelli et Sanjit A. Seshia. « Scenic : un langage pour la spécification de scénarios et la génération de scènes. » *Actes de la 40e conférence ACM SIGPLAN sur la conception et la mise en œuvre de langages de programmation (PLDI)*, 2019. DOI : 10.1145/3314221.3314633. URL : <https://people.eecs.berkeley.edu/~sseshia/pubs/b2hd-fremont-pldi19.html>[5] Chejian Xu, Wenhao Ding, Weijie Lyu, Zuxin Liu, Shuai Wang, Yihan He, Hanjiang Hu, Ding Zhao et Bo Li. « SafeBench : une plateforme d'analyse comparative pour l'évaluation de la sécurité des véhicules autonomes. » *Avances in Neural Information Processing Systems 35 (NeurIPS 2022) Piste des ensembles de données et des benchmarks*, 2022. URL : <https://proceedings.neurips.cc/paper_files/paper/2022/hash/a48ad12d588c597f4725a8b84af647b5-Abstract-Datasets_and_Benchmarks.html>

[6] ASAM. « ASAM OpenSCENARIO DSL : terminologie clé et aperçu conceptuel. » URL : <https://publications.pages.asam.net/standards/ASAM_OpenSCENARIO/ASAM_OpenSCENARIO_DSL/latest/conceptual-overview/key_terms.html>

[7] ASAM. « ASAM OpenODD : modèle vers la référence de mappage DSL ASAM OpenSCENARIO. » URL : <https://publications.pages.asam.net/standards/ASAM_OpenODD/ASAM_OpenODD/latest/spécification/09_openscenario_dsl/09_01_overview.html>[8] Shital Shah, Debadeepta Dey, Chris Lovett et Ashish Kapoor. "AirSim : simulation visuelle et physique haute fidélité pour les véhicules autonomes." *Robotique de terrain et de service*, Springer Proceedings in Advanced Robotics, 2017 ; arXiv : 1705.05065. URL : <https://arxiv.org/abs/1705.05065>

[9] Yunlong Song, Selim Naji, Elia Kaufmann, Antonio Loquercio et Davide Scaramuzza. « Flightmare : un simulateur de quadrirotor flexible. » *Actes de la 4e Conférence sur l'apprentissage des robots (CoRL)*, PMLR 155, 2021. URL : <https://proceedings.mlr.press/v155/song21a.html>

[10] Justin Nakama, Ricky Parada, Joao P. Matos-Carvalho, Fabio Azevedo, Dario Pedro et Luis Campos. « Générateur d'environnement autonome pour la simulation basée sur des drones. » *Sciences appliquées*, 11(5):2185, 2021. DOI : 10.3390/app11052185. URL : <https://doi.org/10.3390/app11052185>[11] Yash Vardhan Pant, Max Z. Li, Alena Rodionova, Rhudii A. Quaye, Houssam Abbas, Megan S. Ryerson et Rahul Mangharam. « FADS : un cadre pour la sécurité des drones autonomes utilisant la planification de trajectoire basée sur la logique temporelle. » *Recherche sur les transports, partie C : technologies émergentes*, 130 :103275, 2021. DOI : 10.1016/j.trc.2021.103275. URL : <https://doi.org/10.1016/j.trc.2021.103275>

[12] Keyu Chen, Yuheng Lei, Hao Cheng, Haoran Wu, Wenchao Sun et Sifa Zheng. "FREA : Génération guidée par la faisabilité de scénarios critiques pour la sécurité avec une rivalité raisonnable." arXiv :2406.02983, 2024. URL : <https://arxiv.org/abs/2406.02983>

[13] Wenhao Ding, Chejian Xu, Mansur Arief, Haohong Lin, Bo Li et Ding Zhao. « Une enquête sur la génération de scénarios de conduite critiques pour la sécurité : une perspective méthodologique. » arXiv :2202.02215, 2022. URL : <https://arxiv.org/abs/2202.02215>[14] Équipe CARLA. « Outil de jumeau numérique : génération procédurale à partir d'OpenStreetMap. » Documentation du simulateur CARLA. URL : <https://carla.readthedocs.io/en/0.9.16/adv_digital_twin/>

[15] Shandong Expressway Group Co., Ltd. « Le système de service de vol d’inspection complet de l’autoroute du Shandong est mis en ligne. » 2025. URL : <https://www.sdhsg.com/article/72553>

[16] Zhao Xiangmo, Zhao Yifei, Lu Nengchao et al. "Un examen de la recherche sur l'allocation des ressources clés en cas d'urgence en cas d'accident de la route." *Transactions d'ingénierie des transports*, 2024. DOI : 10.19818/j.cnki.1671-1637.2024.06.001. URL : <https://transport.chd.edu.cn/cn/article/doi/10.19818/j.cnki.1671-1637.2024.06.001>

[17] Jisheng Zhang, Limin Jia, Shuyun Niu, Fan Zhang, Lu Tong et Xuesong Zhou. "Un cadre de modélisation basé sur un réseau spatio-temporel pour le routage dynamique des véhicules aériens sans pilote dans les applications de surveillance des incidents de la circulation." *Capteurs*, 15(6):13874-13898, 2015. DOI : 10.3390/s150613874. URL : <https://doi.org/10.3390/s150613874>[18] Tan Do-Duy, Long D. Nguyen, Trung Q. Duong, Saeed Khosravirad et Holger Claussen. "Optimisation conjointe du déploiement en temps réel et de l'allocation des ressources pour les communications d'urgence en cas de catastrophe assistées par drone." *Journal IEEE sur certains domaines des communications*, 39(11):3411-3424, 2021. DOI : 10.1109/JSAC.2021.3088662. URL : <https://doi.org/10.1109/JSAC.2021.3088662>

---

## Annexe : Ce plan d'exécution

### Étape 1 : Geler le papier F Positionnement total

- Exposé de position F en tant qu'ingénierie de scénarios critiques pour la sécurité des drones.
- Précisez clairement qu'il ne s'agit pas d'un double de l'article B/C/E, mais d'un groupe d'articles sur l'infrastructure expérimentale et les méthodes de scénarios.
- Adopter la structure de quatre épreuves progressives de F1 à F4.

### Étape 2 : faites d'abord un benchmark F1

- Définir la taxonomie des scénarios UAV.
- Concevoir le schéma `scenario.json`.
- Organisation de 76 millions de journaux d'exploration.
- Trous de couverture des statistiques, modes de défaillance et taux de scénarios invalides.
- Exporter CovUAV-Bench v0.1.

### Étape 3 : Faire progresser l'algorithme de test accéléré F2- Implémenter des lignes de base contradictoires aléatoires/grille/LHS/BO/CMA-ES/RL.
- Implémenter la mémoire de couverture, le score de criticité et le filtre de criticité réalisable.
- Comparez le taux de découverte de pannes, le gain de couverture et le facteur d'accélération.
- Utilisez un test de tenue pour vérifier la valeur de la formation.

### Étape 4 : Étendre la ville F3 à la scène locale

- Accédez à OSM, aux niveaux de route, à la densité des bâtiments et aux POI.
- Sélectionner les sections clés de l'autoroute Jinan/Qingdao/Shandong comme étude de cas.
- Cartographier l'ODD au niveau de la ville avec une cellule de test locale de 50 mx 50 mx 50 m.
- Etablir des indicateurs de couverture des zones fonctionnelles urbaines.

### Étape 5 : Développez l'application d'urgence haute vitesse F4

- Prenez l'inspection/intervention d'urgence de l'autoroute du Shandong comme arrière-plan de la demande.
- Concevoir des scénarios d'accident, des processus de reconnaissance d'UAV et de déploiement collaboratif des ressources de sauvetage au sol.
- Comparez l'optimisation heuristique et basée sur les scénarios au sol uniquement, à la ressource la plus proche, basée sur les drones.
- Concentrez-vous sur le reporting du temps de première visualisation, du temps de réponse, du temps de dédouanement et du taux de répartition erroné.

### Étape 6 : Rythme de soumission

- Investissez d'abord dans F1/F2 pour former un benchmark + méthode dual core.
- F1 Si les outils et les données sont complets, la priorité sera donnée au benchmark T-ITS/ITSC/IROS.
- F2 Si le résultat de l'algorithme est fort, AAAI / ICRA / IROS seront prioritaires.
- F3/F4 Attendez que l'outil F1/F2 soit stable avant de passer à TR-C / T-ITS.

### Étape 7 : Tâches de la semaine récente-Rédiger une mission expérimentale formelle pour la F1.
- Geler les dimensions de la scène : obstacles, structures spatiales, perturbations environnementales, types de tâches, étiquettes de risque.
- Échantillonnage de 10 000 à 50 000 journaux d'exploration à partir de 76 millions de journaux d'exploration pour une analyse préliminaire de la couverture.
- Dessinez la première version du diagramme de taxonomie de scène et de la carte thermique de couverture.