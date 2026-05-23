---
title: "Paper F Journal Planning v2 : Itinéraire prioritaire du journal pour l'ingénierie de scénarios critiques pour la sécurité des drones"
description: "Sans tenir compte de la structure de la thèse de doctorat, l'itinéraire de sortie prioritaire de la revue pour l'article F sera repensé, en se concentrant sur la couverture des scénarios critiques pour la sécurité des drones, les tests accélérés, l'assurance des risques et les applications d'urgence à grande vitesse."
pubDate: 2026-05-20
updatedDate: 2026-05-23
tags: ["Papier F", "Planification du journal", "drone", "génération de scène", "couverture de la scène", "Critique pour la sécurité", "tests accélérés", "garantie de risque", "TR-C", "T-ITS"]
category: Tech
---

# Paper F Journal Planning v2 : Itinéraire prioritaire du journal pour l'ingénierie de scénarios critiques pour la sécurité des drones

> L'accent actuel revient sur la **production d'articles de revues**, qui n'est pas organisée par thèses de doctorat.  
> Conclusion : le papier F ne doit pas être divisé en plusieurs papiers minces, mais doit être transformé en un **journal principal T-ITS** complet, solide et reproductible, puis différencié en documents d'application TR-C et documents de méthode d'assurance des risques basés sur des actifs expérimentaux.

---

## 1. Jugement fondamental

L'orientation générale du papier F est toujours valable : **Ingénierie de scénarios critiques pour la sécurité des drones**. Mais la logique d’un article de revue et d’un chapitre de thèse de doctorat est différente. Les évaluateurs de revues ne paieront pas pour un parcours complet, ils se préoccupent davantage de :

- La question est-elle suffisamment précise ?
- Si la méthode comporte des incréments techniques clairs ;
- Si l'expérience est suffisamment solide ;
- Si les lignes de base sont solides ;
- Si la conclusion soutient un article de journal indépendant ;
- Si cela correspond au périmètre de la revue cible.

Par conséquent, la « Planification 4-5 » actuelle devrait être modifiée comme suit :

> **Fusionner d'abord le benchmark de couverture de scène F1 et la génération accélérée de scènes dangereuses F2 dans un T-ITS principal ; plus tard, l'application d'urgence TR-C, l'assurance des risques T-RO/T-ASE et la génération de scènes ODD urbaines TR-C/T-ITS seront différenciées à partir de la même plate-forme. **

### 1.1 2026-05-22 Étalonnage d'écriture : la série F doit séparer le « papier de la méthode de test » et le « papier du système de circulation »

Le papier F est facile à écrire de manière vague car il propose également des applications de génération de scènes, de couverture, de tests accélérés de scènes dangereuses, d'ODD urbain et d'urgence sur les autoroutes du Shandong. Le dernier étalonnage est :- **F-J1 ne doit pas être écrit TR-C. ** Son principal enjeu est le test de scénarios critiques pour la sécurité : comment découvrir systématiquement des scénarios dangereux mais efficaces d'évitement d'obstacles/navigation à basse altitude par des drones. Il est plus naturel d’investir dans les T-ITS car l’accent est mis principalement sur les tests de sécurité, l’évaluation des simulations et la couverture de scénarios dans les systèmes de transport intelligents.
- **F-J2 est le document de candidature TR-C. ** Il doit être rédigé sous forme de questions d'opérations d'urgence sur le trafic à grande vitesse : détection des accidents, reconnaissance des drones, allocation des ressources au sol, temps de réponse, couverture, retard de l'information et récupération du trafic.
- **F-J4 Lancez TR-C uniquement si l'ODD au niveau de la ville peut être renvoyé à la planification du trafic/au contrôle opérationnel. ** Si vous convertissez simplement OSM en une combinaison d'obstacles locaux, cela ressemblera davantage à un outil de simulation ou à un benchmark, mais pas suffisamment à TR-C.

Par conséquent, « l'histoire » de la série F peut être divisée en deux types :

| Thèse | Histoire du système | Ce qui doit être soutenu |
|------|----------|----------------|
| F-J1 | Les tests de sécurité des STI à basse altitude manquent de couverture et de normes de génération de scénarios dangereux | métrique de couverture, taux de scénarios invalides, découverte d'échecs, tests croisés du planificateur, statistiques multi-graines |
| F-J2 | Une intervention d'urgence à grande vitesse nécessite une collaboration UAV-sol pour raccourcir le temps de réponse en or | Topologie réelle à haut débit/proxy d'accident, modèle d'allocation des ressources, temps de réponse, couverture, accessibilité en cas de congestion, analyse de sensibilité |
| F-J3 | Comment convertir la couverture de scène en preuve d'assurance des risques | limite de couverture au risque, intervalle de confiance, estimation d'événements rares, indice de fiabilité |
| F-J4 | Comment l'ODD global de la ville détermine les scénarios de tests locaux à basse altitude | Cartographie OSM/POI/bâtiment/route/espace aérien, fidélité aux risques locaux, généralisation inter-villes |

La version du journal de trafic doit éviter de se contenter de dire "nous avons généré des scénarios plus dangereux". Une conclusion plus forte serait :- Quelles structures urbaines ou tronçons d'autoroute sont les plus susceptibles de provoquer une panne de drone ?
- Quelles combinaisons d'obstacles sont les plus dangereuses pour les différents planificateurs ?
- Une couverture accrue réduit-elle réellement les risques non détectés, plutôt que de simplement augmenter la taille de l'échantillon ?
- Dans les applications d'urgence, la reconnaissance par drone peut-elle réduire les pertes de répartition causées par des informations incomplètes ?
- Quand les conditions météorologiques, les communications ou les points d'atterrissage sont-ils limités, quand le système aura-t-il besoin de ressources au sol pour le sauvegarder ?

La raison est très simple : le benchmark seul est facile à considérer comme un trop grand nombre de plates-formes d'ingénierie, et les tests accélérés seuls seront remis en question si l'espace des scénarios de test est clairement défini. Une fois les deux fusionnés, le document passe de « Je génère des scénarios dangereux » à :

> **J'ai défini l'espace de scène critique pour la sécurité des drones, qui peut mesurer la couverture, découvrir des trous de couverture et utiliser des méthodes guidées par la couverture pour générer plus efficacement des scénarios de test réalistes, dangereux et réalisables. **

Cela ressemble plus à un article de journal.

### 1.2 2026-05-23 Compilation : La série F n'avance actuellement que deux lignes principales

Actuellement, l'article F n'est pas développé en fonction du catalogue des thèses de doctorat, mais est d'abord condensé en deux lignes principales basées sur les publications des revues. Les F-J3 et F-J4 sont conservés mais n'enlèvent pas les moyens expérimentaux du F-J1.| Thèse | Investisseur principal | Rôle actuel | Stratégies récentes |
|------|------|----------|--------------|
| F-J1 | T-ITS | tests accélérés guidés par la couverture | Poussée principale ; doit utiliser 76 millions de journaux d'exploration, une mesure de couverture, une base de référence solide et une évaluation inter-planificateurs |
| F-J2 | TR-C | Allocation des ressources de secours d'urgence à grande vitesse du Shandong | La plate-forme F-J1 sera démarrée après stabilisation ; l'accent est mis sur la topologie réelle à grande vitesse, le proxy d'accident, le temps de réponse et les goulots d'étranglement des ressources |
| F-J3 | T-RO / T-ASE / T-ITS | assurance couverture-risque | Suspendu; attendre que F-J1 forme des statistiques de répartition des pannes et de couverture avant de prouver la limite du risque |
| F-J4 | TR-C / T-ITS | scénario ODD au niveau de la ville vers le scénario UAV local | Suspendu; attendre que le pipeline OSM/POI/bâtiment/espace aérien soit suffisamment stable |

Le plan recommandé pour la première version du F-J1 est fixé à :1. **Espace de scénario** : définissez la cellule de test locale du drone, la grammaire des obstacles, les facteurs dynamiques, les objectifs de mission et la détermination de scénarios invalides.
2. **Métrique de couverture** : statistiques distinctes sur la couverture géométrique, la couverture sémantique, la couverture dynamique, la couverture des risques et la couverture en mode de défaillance.
3. **Génération accélérée** : utilisez les trous de couverture et la probabilité de défaillance pour guider l'échantillonnage et filtrer les scénarios irréalistes ou inexécutables.
4. **Protocole de référence** : Unifiez les graines de carte, l'ensemble de planificateurs, les paramètres du contrôleur, les graines aléatoires, les seuils de défaillance et les tests statistiques.
5. **Expériences principales** : Comparez la génération aléatoire, grille/LHS, BO, CMA-ES, RL contradictoire, contrainte de style scénique et cette méthode.
6. **Analyse des défaillances** : expliquez quelles combinaisons d'obstacles, de conditions de vitesse/hauteur, d'occlusions et d'obstacles dynamiques sont les plus susceptibles de déclencher une défaillance.

Le jugement après ce tri est le suivant : F-J1 poursuit d'abord "un article de journal sur les tests de sécurité qui peut être soumis au T-ITS", et ne devrait pas s'engager en même temps dans l'urbanisme, la théorie des risques et les applications à grande vitesse du Shandong. Le F-J2 peut basculer l'histoire vers la boucle fermée des opérations d'urgence routière requise par TR-C une fois que la bibliothèque de scénarios et les indicateurs de risque du F-J1 ont mûri.

---

## 2. Portefeuille de papiers prioritaires du journal

Il est recommandé de prévoir temporairement **3 revues principales + 1 revue de réserve** au lieu de promouvoir 5 articles en même temps.| Numéro | Positionnement du papier | Sujet suggéré | Investisseur principal | Priorité |
|------|----------|----------|------|--------|
| F-J1 | Méthode Workhorse + benchmark | Tests accélérés guidés par la couverture pour les scénarios de navigation de drones critiques pour la sécurité | T-ITS | Le plus haut |
| F-J2 | Application d'urgence routière | Allocation des ressources UAV au sol en fonction des scénarios pour les interventions d'urgence sur les autoroutes | TR-C | Élevé |
| F-J3 | Assurance couverture-risque pour les tests de scénarios critiques pour la sécurité des drones | T-RO / T-ASE / T-ITS | Moyen à élevé |
| F-J4 | Génération de scènes urbaines | City2Local-UAV : Génération de scénarios hiérarchiques depuis les ODD urbains jusqu'aux compositions d'obstacles locaux | TR-C / T-ITS | Moyen |

**Recommandation d'ordre d'exécution : F-J1 -> F-J2 -> F-J3 -> F-J4. **

F-J1 est la plateforme et la base de l'algorithme. F-J2 est le plus proche de la valeur d'application du journal de transport. F-J3 Revues de robotique ou d'automatisation utilisées pour mettre à jour les méthodes/théories. F-J4 ne peut être réalisé qu'une fois que le pipeline de données OSM/ville a mûri, sinon il deviendra facilement un « outil de conversion de carte ».

---

## 3. Modèle de littérature et lacunes

### 3.1 L'ingénierie des scénarios de conduite autonome a mûri, mais la migration des drones est insuffisanteIl existe déjà une chaîne d’ingénierie complète dans le domaine de la conduite autonome. ISO 34502 fournit un cadre d'évaluation de la sécurité basé sur des scénarios pour les systèmes de conduite automatisés [1], et ASAM OpenSCENARIO et OpenODD fournissent des normes de scénarios exécutables et de description ODD [2] [3]. La génération accélérée de bibliothèques de tests et de scénarios de tests de Shuo Feng illustre en outre que les tests critiques pour la sécurité ne peuvent pas s'appuyer sur des échantillons aléatoires naturels, mais doivent utiliser une approche basée sur les données pour améliorer l'efficacité de l'échantillonnage des événements critiques [4] [5] [6].

Ces dernières années, de grandes conférences ont également continué à promouvoir cette direction : Scenic utilise un langage de programmation probabiliste pour exprimer la distribution et les contraintes des scènes [7] ; SafeBench a réalisé un référentiel d'évaluation de la sécurité [8] ; ScenarioNet extrait des scénarios de trafic à grande échelle à partir de données de conduite réelles [9] ; AdvSim, KING, ChatScene et FREA génèrent des scénarios critiques pour la sécurité du point de vue de la perturbation des capteurs, de l'optimisation du gradient, de la génération de connaissances LLM et de l'adversarialité réalisable respectivement [10] [11] [12] [13].

Cependant, la plupart de ces travaux sont orientés vers la conduite autonome au sol, et les scénarios de drones sont sensiblement différents :

- L'UAV est un mouvement tridimensionnel, et les dimensions de la scène incluent l'altitude, l'inclinaison de la trajectoire, le champ de vent, la puissance, l'occlusion du champ de vision et la dynamique de vol ;
- Les événements dangereux liés aux drones comprennent les collisions avec des bâtiments, les collisions avec des lignes, le franchissement de zones d'exclusion aérienne, les conflits dans les couloirs à basse altitude, l'échec du décollage et de l'atterrissage et l'entrée accidentelle dans des sites d'urgence ;
- Les tests de sécurité des drones font rarement mûrir la taxonomie ODD ;
- Le benchmark UAV se concentre principalement sur les tâches de simulation et de contrôle, et répond rarement « quels risques le scénario couvre-t-il ? »

### 3.2 La simulation de drone a une base, mais les tests de sécurité axés sur la couverture sont encore videsAirSim et Flightmare sont des bases importantes pour la simulation de drones [14][15]. ÉvitezBench a proposé une référence haute fidélité pour l'évitement d'obstacles multi-rotors basé sur la vision [16]. OmniDrones et Aerial Gym illustrent que la simulation de drones parallèles sur GPU et la formation par apprentissage par renforcement à grande échelle arrivent à maturité [17] [18]. FADS prouve que les spécifications de sécurité de la logique temporelle peuvent entrer dans le pipeline de sécurité des drones [19].

Ces travaux constituent la base de l'outil pour l'article F, mais ils n'ont pas encore comblé les lacunes les plus critiques des articles de revues :

> **Comment définir l'espace des scénarios critiques pour la sécurité des drones, comment mesurer la couverture, comment générer efficacement des scénarios à longue traîne qui sont à la fois dangereux et réalisables, et comment convertir la couverture des tests en évaluations des risques interprétables. **

C'est l'occasion pour le F-J1/F-J3.

### 3.3 La différence entre TR-C/T-ITS détermine comment couper le papier

Le noyau intellectuel du TR-C se situe du côté des transports, mettant l'accent sur l'impact des technologies émergentes sur la planification, la conception, l'exploitation, le contrôle et la logistique des systèmes de transport [20]. Les T-ITS couvrent explicitement les applications des technologies de l'information dans les domaines de la détection, des communications, des contrôles, de la planification, de la conception, de la mise en œuvre, de l'IA et des systèmes de transport [21].

donc:- **F-J1 Vote pour T-ITS** : Parce qu'il s'agit d'une évaluation de la sécurité des ITS/de la génération de scénarios/des tests de navigation d'UAV.
- **F-J2 a voté pour TR-C** : Parce qu'il s'agit du fonctionnement du système d'urgence routière à grande vitesse et de l'allocation des ressources.
- **F-J3 est éligible pour T-RO/T-ASE/T-ITS** : en fonction de l'orientation de la théorie et de l'expérience, T-RO/T-ASE peut être sélectionné pour les tests de sécurité des robots, et T-ITS peut être sélectionné pour les systèmes de transport.
- **F-J4 vote pour TR-C/T-ITS** : Si l'accent est mis sur le trafic urbain à basse altitude ODD et l'impact des systèmes de circulation, votez pour TR-C ; si l'accent est mis sur les interfaces de scène et l'évaluation de la simulation, votez pour T-ITS.

### 3.4 Sur quelles autres orientations du journal puis-je écrire ?

Après avoir continué à creuser plus profondément, vous pouvez préparer 4 « fourchettes candidates » supplémentaires, mais il n'est pas recommandé de commencer à écrire en même temps maintenant. Ils conviennent davantage comme retombées naturelles à mesure que la plate-forme expérimentale du F-J1 mûrit.| Fourchettes | Sujets inscriptibles | Principaux arguments de vente | Journaux des candidats | Recommandations actuelles |
|------|----------|----------|--------------|--------------|
| F-J5 | Dossier de sécurité basé sur des scénarios pour les opérations d'UAV à basse altitude | Organiser les preuves de couverture, de criticité et de défaillance dans des dossiers de sécurité | Ingénierie de la fiabilité et sécurité des systèmes / Transactions IEEE sur la fiabilité / Science de la sécurité | Attendez que F-J1 ait des résultats avant d'écrire |
| F-J6 | Transfert entre simulateurs de scénarios critiques pour la sécurité des drones | Transfert de scénario d'étude de la simulation légère vers AirSim / Flightmare / ÉvitezBench | Robotique et systèmes autonomes / Journal of Field Robotics / T-RO | Vérification réaliste ou haute fidélité requise |
| F-J7 | Génération de scénarios d'UAV guidée par les connaissances | Utilisez LLM/VLM/Knowledge Graph pour générer des scènes de danger sémantique | T-ITS / T-IV / IEEE Journal ouvert des ITS | Peut être lié au Paper E, mais ne le submergez pas |
| F-J8 | Tests de contrainte de couloir multi-UAV | Scénarios spécialement générés de conflits de couloirs à basse altitude, de congestion, de goulots d'étranglement au décollage et à l'atterrissage | T-ITS / TR-C / T-IV | Peut être lié au papier B |Parmi eux, le **F-J5 est le plus intéressant à conserver**. Si le suivi F-J1 ne s'arrête qu'à « trouver d'autres échecs », la valeur du journal restera encore expérimentale ; mais si la couverture du scénario peut être convertie en preuves d'assurance de fiabilité/sécurité, elle peut être soumise à des revues de sécurité et de fiabilité telles que Reliability Engineering & System Safety ou IEEE Transactions on Reliability [28] [29]. La science de la sécurité peut également être utilisée comme alternative, mais elle est davantage axée sur la gestion de la sécurité, les facteurs humains, l'organisation et la prévention des accidents. Si l’article est encore purement algorithmique, il n’est pas recommandé pour une première soumission [30].

Le F-J6 convient à l'écriture lorsqu'il y a de vrais drones ou des résultats de simulation haute fidélité. Le Journal of Field Robotics et Robotics and Autonomous Systems valorisent tous deux l'autonomie, la fiabilité et la profondeur expérimentale des systèmes robotiques dans des environnements réels ou haute fidélité [31] [32]. Si vous ne disposez que d'une simulation légère, ne soumettez pas encore à ce type de revue.

F-J7 n'est pas recommandé comme ligne principale maintenant car il chevauchera la direction LLM/LTL de l'article E. Il est plus approprié pour des extensions ultérieures en tant que « génération de scénarios guidée par la connaissance » : LLM est chargé de proposer des scénarios de risque sémantique, et Cov-ATUAV est responsable de la validation, du filtrage et de la quantification de la couverture.

F-J8 est la version de test de résistance du papier B. Il n'optimise plus la planification de centaines de drones, mais génère des scénarios de test qui exposent au mieux la congestion des couloirs, le goulot d'étranglement des vertiports, le goulot d'étranglement de la charge et l'échec de la résolution des conflits. T-ITS ou TR-C peuvent être votés dans ce sens, mais ils doivent être coupés de la contribution de programmation du Paper B.

### 3.5 Carte du journal du candidat| Revue | Le papier F coupé le plus approprié | Pourquoi c'est approprié | Risques |
|------|--------------|------------|------|
| IEEE T-ITS | F-J1 / F-J4 / F-J8 | le champ d'application couvre les systèmes de détection, de contrôle, d'IA, de planification et de transport dans les ITS [21] | Le drone doit être écrit comme un ITS à basse altitude, et non comme un robot ordinaire |
| IEEE T-IV | F-J1 / F-J7 / F-J8 | Le contexte des véhicules intelligents et de la mobilité automatisée peut faire l'objet de tests de sécurité et de génération de scénarios [26] | Les véhicules terrestres ont une couleur forte, le drone doit expliquer la pertinence du véhicule/du trafic |
| TR-C | F-J2 / F-J4 / F-J8 | Mettre l'accent sur l'impact des technologies émergentes sur les opérations de transport, le contrôle et la logistique [20] | Ne convient pas au benchmark d'algorithmes purs |
| TR-E | F-J2 | Convient au déploiement de la logistique, de la distribution, de la chaîne d'approvisionnement et du transport de ressources d'urgence [33] | Si le drone comporte trop de détails techniques, il s'écartera de la logistique |
| T-ASE | F-J3 / F-J5 | les systèmes d'automatisation, les tests, l'évaluation et la définition de la fiabilité sont plus adaptés [27] | La méthode doit avoir une valeur de généralisation pour les systèmes d'automatisation |
| T-RO | F-J3 / F-J6 | Les tests de sécurité des robots, la planification et la vérification réelle du système peuvent être soumis [34] | Le benchmark synthétique à lui seul ne suffit pas |
| Transactions IEEE sur la fiabilité | F-J5 | Convient pour la modélisation de la fiabilité, la quantification des risques et l'assurance [28] | Il faut une garantie statistique sérieuse, pas seulement des tableaux expérimentaux |
| Ingénierie de fiabilité et sécurité des systèmes | F-J5 |Convient aux systèmes critiques pour la sécurité, à l'évaluation des risques et à l'ingénierie de fiabilité [29] | Nécessité de passer de l'amélioration des performances des algorithmes aux preuves de sécurité |
| Science de la sécurité | F-J2 / F-J5 | Convient pour la sécurité d'urgence, la prévention des accidents et la gestion de la sécurité [30] | L'algorithme UAV pur ne convient pas |
| Robotique et Systèmes Autonomes / JFR | F-J6 | Convient aux systèmes robotiques autonomes et à la validation sur le terrain/haute fidélité [31] [32] | Les expériences système doivent être plus fortes que le récit de la thèse |
| Journal ouvert IEEE des ITS | F-J1 / F-J2 | Peut être utilisé comme alternative à accès libre et rapide [35] | Impact et positionnement généralement inférieurs à ceux de T-ITS |**Le premier ordre de sélection actuel reste inchangé : F-J1 est la première sélection pour T-ITS, F-J2 est la première sélection pour TR-C, F-J3 sélectionne T-ASE/T-RO/T-ITS en fonction de la force théorique et F-J5 est réservé aux revues fiables. **

---

## 4. La première grande revue : F-J1

### 4.1 Thèmes suggérés

**Tests accélérés guidés par la couverture pour les scénarios de navigation de drones critiques pour la sécurité**

### 4.2 Objectifs de soumission

Contributeur principal : **Transactions IEEE sur les systèmes de transport intelligents**.  
Alternatives : Transactions IEEE sur la science et l'ingénierie de l'automatisation, Transactions IEEE sur la robotique, la robotique et les systèmes autonomes.

Le T-ITS est le plus approprié car le document peut être rédigé comme un test de sécurité des transports intelligents : les drones participent au transport à basse altitude, et la génération de scénarios sert à la vérification de la sécurité des ITS à basse altitude.

### 4.3 Problèmes fondamentaux

Les documents existants sur l'évitement d'obstacles et la navigation des drones rapportent généralement le taux de réussite, le taux de collision ou la longueur de la trajectoire, mais indiquent rarement si le scénario de test couvre les ODD critiques pour la sécurité. Les scènes générées aléatoirement présentent deux problèmes : un grand nombre d’échantillons ne sont pas dangereux, et de nombreux échantillons dangereux sont physiquement irréalisables ou ne peuvent être évités par aucun algorithme.

F-J1 veut répondre :

> Comment couvrir les opérations ODD de drones à basse altitude avec un budget de simulation limité et donner la priorité à la génération de scénarios critiques pour la sécurité qui sont réels, dangereux, réalisables et capables de distinguer les capacités des algorithmes ?

### 4.4 Conception de la méthode

Suggestion de nom de méthode : **Cov-ATUAV : tests accélérés guidés par la couverture pour les drones**.

Pipeline global :

```text
UAV ODD taxonomy
  -> scenario parameterization
  -> coverage memory
  -> criticality and feasibility scoring
  -> adaptive scenario generation
  -> planner evaluation and coverage update
```

Modules de base :| Module | Fonction |
|------|------|
| Grammaire du scénario | Définir les obstacles, les structures spatiales, les corps dynamiques, les champs de vent, le bruit des capteurs, les types de tâches |
| Mémoire de couverture | Bacs de paramètres d'enregistrement, couverture par paire/t, modes de défaillance |
| Score de criticité | Exposition complète, défi, quasi-accident, violation de contrainte |
| Filtre de faisabilité | Exclure les collisions inévitables, les physiques déraisonnables et les scénarios de mission dénués de sens |
| Générateur adaptatif | Générez de nouveaux échantillons dans les trous de couverture et les régions à haute criticité |
| Harnais d'évaluation | Évaluation unifiée de plusieurs planificateurs d'UAV et stabilité du classement des sorties |

### 4.5 Données et plateforme

- Expérience principale : cellule de test locale de drones de 50 m x 50 m x 50 m.
- Actifs existants : 76 millions de journaux d'exploration, utilisés pour compter les trous de couverture, les modes de défaillance et la distribution initiale des scènes.
- Simulation légère : grille 3D auto-construite / PyBullet / dynamique personnalisée pour une recherche à grande échelle.
- Validation haute-fidélité : AirSim, Flightmare, ÉvitezBench ou Aerial Gym pour une validation cross-simulateur à petite échelle [14] [15] [16] [18].

76 millions d'explorations ne peuvent pas être écrites comme le résultat final, mais cela peut s'écrire comme suit :

> « Nous initialisons et validons notre analyse de couverture de scénarios à l'aide d'un journal d'exploration à grande échelle contenant plus de 76 millions de déploiements simulés. »

### 4.6 Lignes de base| Référence | Fonction |
|--------------|------|
| Génération de scénarios aléatoires | Efficacité d'échantillonnage de base |
| Échantillonnage de grille | Couverture discrète uniforme |
| Échantillonnage d'hypercube latin | Couverture de l'espace paramétrique |
| Échantillonnage contraint de style scénique | Génération de scènes contraintes [7] |
| Échantillonnage de modèles de style SafeBench | référence de sécurité de type modèle [8] |
| Optimisation bayésienne | recherche d'échec de la boîte noire |
| CMA-ES / méthode d'entropie croisée | Recherche continue de dangers paramétriques |
| Édition contradictoire de style AdvSim/KING | Trajectoire adverse/perturbation d'obstacles [10] [11] |
| Génération contradictoire réalisable de style FREA | Exemple contradictoire raisonnable [13] |

### 4.7 Planificateurs de drones

Au moins trois types de planificateurs doivent être testés, sinon la revue remettra en question le surajustement d'un seul algorithme :

| Planificateur | Représentant |
|--------------|------|
| Classique | A* / RRT* / champ de potentiel artificiel / 3DVFH* |
| Optimisation | MPC / corridor sécurisé / optimisation de trajectoire B-spline |
| Basé sur l'apprentissage | PPO / SAC / apprentissage par imitation / politique basée sur une vision |

Si la puissance de calcul est limitée, les choix garantis de la première version sont : RRT*, MPC-lite, politique PPO et une base de référence basée sur la vision.

### 4.8 Indicateurs| Indicateur | Descriptif |
|------|------|
| Gain de couverture | Nouvelle couverture tous les 1000 tests |
| Taux de découverte d'échecs | Le rapport collision / quasi-accident / délai d'attente découvert par unité de budget |
| Facteur d'accélération | La réduction multiple du nombre de tests nécessaires pour atteindre le même taux de découverte de pannes |
| Criticité réalisable | Proportion de scènes dangereuses et évitables |
| Taux de scénario invalide | Rapport d'échantillonnage physiquement irréalisable ou dénué de sens |
| Stabilité du classement du planificateur | Stabilité du classement du planificateur sous différents sous-ensembles de graines/scénarios |
| Transfert entre simulateurs | Les scénarios découverts dans la simulation légère peuvent-ils être transférés vers la simulation haute fidélité |

### 4.9 Résultats minimaux qu'une revue peut publier

F-J1 nécessite au moins une preuve de :

1. Par rapport au aléatoire/grille/LHS, Cov-ATUAV améliore considérablement le gain de couverture avec le même budget.
2. Par rapport aux lignes de base BO/CMA-ES/adversaires, Cov-ATUAV réduit le taux de scénarios invalides.
3. Par rapport à la recherche d'échec pure, les scènes générées par Cov-ATUAV peuvent distinguer les différents planificateurs d'UAV de manière plus stable.
4. Vérifiez qu'au moins certains des scénarios à haut risque peuvent être migrés dans AirSim/Flightmare/AvitBench.
5. Schéma de scénario de sortie, graine, répartition de référence et script d'évaluation pour améliorer la reproductibilité des T-ITS.

---

## 5. Le deuxième journal : application d'urgence à grande vitesse F-J2

### 5.1 Thèmes suggérés

**Allocation de ressources UAV au sol en fonction des scénarios pour les interventions d'urgence sur les autoroutes**

### 5.2 Objectifs de soumissionInvestisseur principal : **Recherche sur les transports Partie C : Technologies émergentes**.  
Alternative : Transactions IEEE sur les systèmes de transport intelligents.

Cet article doit mettre l'UAV en opération de transport, plutôt que de l'écrire comme un « algorithme de planification d'UAV ». Le système de service de vol d'inspection complet de l'autoroute du Shandong utilise déjà des plates-formes sans surveillance et des drones industriels pour les inspections, les inspections, les interventions d'urgence et l'analyse des données [22]. Les recherches sur l'allocation des ressources d'urgence à grande vitesse mettent également en évidence des problèmes tels que des informations incomplètes dès les premiers stades d'un accident, des conditions de circulation variables dans le temps et un lien insuffisant entre la sélection de l'emplacement des installations et l'allocation des ressources [23].

### 5.3 Problèmes fondamentaux

Les interventions d’urgence à grande vitesse en cas d’accident ne consistent pas simplement à « envoyer les ressources les plus proches ». Lorsqu'un accident survient pour la première fois, le type d'accident, la longueur de la congestion, les fermetures de voies, les matières dangereuses et les risques d'accident secondaire sont tous incertains. L’intérêt des drones n’est pas seulement de voler vite, mais aussi de réduire à l’avance l’incertitude des informations et de réduire les erreurs de livraison et les retards.

F-J2 veut répondre :

> En cas d'urgence suite à un accident à grande vitesse, comment utiliser la reconnaissance par drone pour réduire l'incertitude des informations et se coordonner avec les ressources de garde au sol, de sauvetage, de lutte contre les incendies, de police de la circulation et de maintenance afin de réduire le temps de réponse, le temps de dégagement et les risques secondaires ?

### 5.4 Conception de la méthode

Suggestion de nom de méthode : **SAFER-UAV : réponse d'urgence rapide basée sur un scénario avec des drones**.

Structure de base :

```text
Incident scenario generator
  -> UAV first-view dispatch
  -> incident state belief update
  -> ground resource rolling allocation
  -> congestion / clearance simulator
  -> emergency performance evaluation
```

La clé est de convertir la bibliothèque de scènes du F-J1 en scénarios de missions d’urgence :

- Types d'accidents : collision arrière, renversement, produits chimiques dangereux, travaux de construction occupant la route, intempéries, accidents secondaires liés aux embouteillages.
- Géométrie des tronçons routiers : lignes droites, courbes, rampes, aires de service, gares de péage, ponts, entrées de tunnels.
- Informations incertaines : gravité de l'accident, voies praticables, victimes, besoins en ressources, durée de la congestion.
- Types de ressources : drone, dépanneuse, ambulance, pompiers, police de la circulation, véhicule de maintenance, équipement de contrôle temporaire.

### 5.5 Lignes de base| Référence | Fonction |
|--------------|------|
| Expédition au sol uniquement | Pas de situation de drone |
| Répartition de la ressource la plus proche | Ressources les plus proches en premier |
| Plan fixe / expédition basée sur des règles | Rapprochement des pratiques actuelles |
| UAV d'abord puis expédition | Une stratégie simple à surveiller d'abord puis à expédier |
| Programmation stochastique en deux étapes | Référence d'optimisation stochastique |
| Optimisation de l'horizon glissant | Base de référence d'optimisation solide |
| SAFER-UAV complet | Méthode principale |

### 5.6 Indicateurs

| Indicateur | Descriptif |
|------|------|
| Heure de première vue | Le moment où le drone a acquis pour la première fois les images de l'accident |
| Temps de réponse | Heure d'arrivée du premier lot de ressources |
| Délai de dédouanement | Délai d'achèvement du dédouanement |
| Taux d'expédition erroné | La proportion d'envois erronés, d'envois manqués ou de ressources insuffisantes |
| Risque d'accident secondaire | Indicateur de risque d'accident secondaire |
| Retard de circulation | Total des retards causés par des accidents |
| Valeur informationnelle du drone | Réduction de l'incertitude et avantages en matière de planification apportés par la reconnaissance par drone |
| Couverture des actifs critiques | Capacités de couverture des ressources drones/sol pour les zones de service, les ponts, les tunnels et les sections sujettes aux accidents |
| Robustesse au délai d'information | Dégradation des performances lorsque le retour d'image, la confirmation d'événement et les délais de communication augmentent |
| Équité entre les segments routiers | Différence de temps de réponse entre les segments routiers éloignés et les segments routiers principaux |

La version TR-C nécessite également un **tableau d'implication du système** : compte tenu du nombre de drones, du nombre de points de décollage et d'atterrissage, de la configuration des ressources au sol et de l'intensité des accidents, signaler quand le système passe de « la reconnaissance des drones a des avantages évidents » à « les ressources au sol ou la congestion du réseau routier deviennent le principal goulot d'étranglement ». Ce tableau ressemble plus à un document sur les systèmes de transport qu'à de simples temps de réponse moyens.

### 5.7 Résultats minimaux qu'une revue peut publier

F-J2 nécessite au moins :1. Démontrer explicitement que la reconnaissance par drone réduit l’incertitude de l’information, plutôt que de simplement réduire la distance.
2. Mieux que les ressources au sol uniquement et les plus proches dans les scénarios de pointe/nuit/mauvais temps/incidents multiples.
3. Comparez avec l'optimisation continue ou l'optimisation aléatoire pour illustrer le compromis entre le temps réel et les performances.
4. Notez les implications en matière de transport : déploiement de plate-forme sans surveillance, préréglage des ressources, système d'intervention d'urgence.

---

## 6. La troisième revue : Méthode d'assurance des risques F-J3

### 6.1 Thèmes suggérés

**Assurance couverture des risques pour les tests de scénarios critiques pour la sécurité des drones**

### 6.2 Objectifs de soumission

La mise principale dépend du résultat :

- Tests de sécurité des robots biaisés : Transactions T-RO/IEEE sur la science et l'ingénierie de l'automatisation.
- Système d'intelligence partielle du trafic : T-ITS.
- Garanties statistiques partielles et risques d'apprentissage : Direction revue Machine Learning / Intelligence Artificielle.

### 6.3 Pourquoi cet article est-il nécessaire ?

F-J1 peut répondre « comment générer et couvrir des scènes », mais les critiques de journaux peuvent également demander :

> Maintenant que vous avez couvert ces scénarios, pouvez-vous déterminer le degré de sécurité du système ? Quelle est la relation entre la couverture et le risque réel ?

C'est là que le F-J3 entre en jeu. Il ne s'agit pas d'une autre référence, mais il relie la couverture des scènes, l'échantillonnage par importance, l'approche par scénario et le contrôle conforme des risques. L'approche par scénario de Campi et Garatti donne une garantie de probabilité de faisabilité sous des contraintes de scénario aléatoires [24], et le contrôle conforme des risques fournit un cadre de contrôle des risques sans distribution [25]. Celles-ci peuvent être adaptées en garanties statistiques pour les tests de sécurité des drones.

### 6.4 Conception de la méthode

Suggestion de nom de méthode : **CovRisk-UAV**.

Idée de base :- Diviser l'espace du scénario UAV en cellules de couverture ;
- Estimer le risque de défaillance/quasi-accident/violation au sein de chaque cellule ;
- Utiliser la pondération par importance pour corriger le biais d'échantillonnage des tests accélérés ;
- Utiliser un contrôle de risque conforme pour donner une limite supérieure de risque pour un échantillon fini ;
- Donnez des intervalles de confiance pour le classement du planificateur au lieu d'un simple taux de collision moyen.

Formellement, le risque cible peut être défini :

$$
R(\pi)=\mathbb{E}_{s\sim P_{\text{ODD}}}[\ell(\pi,s)],
$$

Où $\pi$ est le planificateur du drone, $s$ est le scénario et $\ell$ est la perte de collision, de quasi-accident ou de violation de contrainte.

Puisque le scénario de test provient de la distribution accélérée $Q(s)$, une correction d'importance est nécessaire :

$$
\hat{R}(\pi)=
\frac{1}{N}\sum_{i=1}^{N}
\frac{P_{\text{ODD}}(s_i)}{Q(s_i)}
\ell(\pi,s_i).
$$

Utiliser à nouveau des limites conformes/scénarios donne :

$$
P(R(\pi)\leq \hat{R}_{\alpha}(\pi))\geq 1-\alpha.
$$

### 6.5 Lignes de base| Référence | Objectif de comparaison |
|----------|----------|
| Taux d'échec empirique | Aucune garantie de confiance |
| Intervalle de confiance bootstrap | Base statistique |
| Échantillonnage d'importance uniquement | Corriger le biais d'échantillonnage uniquement |
| Approche par scénario uniquement | Seule la probabilité de faisabilité est limitée |
| Contrôle conforme des risques | Base de référence en matière de contrôle des risques |
| CovRisk-UAV complet | lié au risque et conscient de la couverture |

### 6.6 Résultats minimaux qu'une revue peut publier

1. Vérifiez que l’étalonnage de la limite supérieure du risque est valide dans les scénarios synthétiques de risque connu.
2. Donnez des intervalles de confiance aux différents planificateurs dans la bibliothèque de scénarios F-J1.
3. Expliquez que les tests accélérés ne peuvent pas utiliser directement le taux d'échec d'origine et nécessitent une correction de distribution.
4. Prouver que la limite de risque tenant compte de la couverture est plus stricte ou plus stable que les tests aléatoires naïfs.

---

## 7. Le quatrième journal : F-J4 urbain ODD vers la scène locale

### 7.1 Thèmes suggérés

**City2Local-UAV : Génération de scénarios hiérarchiques depuis les ODD urbains jusqu'aux compositions d'obstacles locaux**

### 7.2 Objectifs de soumission

Acteurs principaux : TR-C/T-ITS.  
Il n'est pas recommandé pour le moment de se classer en premier dans cette direction, car cela nécessite davantage de traitement de données urbaines et de prise en charge des cas.

### 7.3 Problèmes fondamentaux

Bien que la scène locale de 50 m x 50 m x 50 m soit contrôlable, les véritables risques de circulation à basse altitude proviennent de la structure urbaine : niveaux des routes, densité des bâtiments, ponts, aires de service, zones d'exclusion aérienne, hôpitaux, écoles, échangeurs et points sujets aux accidents. F-J4 doit mapper l'ODD au niveau de la ville à la composition des obstacles locaux.

### 7.4 Conception de la méthode

```text
City ODD
  -> functional zone and road segment extraction
  -> local UAV test-cell sampling
  -> obstacle grammar instantiation
  -> coverage-aware scenario selection
  -> simulator-ready scenario package
```

### 7.5 Résultats minimaux que la revue peut publier1. Au moins deux études de cas de villes ou de zones routières.
2. Il peut être prouvé que la génération consciente de la ville est plus réaliste que la génération locale purement aléatoire.
3. Il peut être prouvé que la scène locale générée est meilleure que le modèle artificiel en termes de couverture et de criticité.
4. Produisez un pipeline reproductible depuis les zones fonctionnelles urbaines vers les combinaisons de scènes locales.

---

## 8. Itinéraire recommandé : quel article écrire en premier

Celui sur lequel on devrait se concentrer le plus en ce moment est **F-J1**, et non quatre chapitres en même temps.

### 8.1 Pourquoi F-J1 est la priorité

- Il convertit 76 millions de journaux d'exploration en actifs papier.
- Il peut absorber les tests de référence F1 et les tests accélérés F2, et est suffisamment grand pour les revues.
- Il a une valeur de réutilisation pour les trois prochains articles.
- C'est le moyen le plus simple de former une boucle fermée expérimentale complète : définition du scénario, méthode de génération, lignes de base, planificateurs, métriques et vérification par simulation croisée.

### 8.2 La contribution principale du F-J1 devrait être réduite à trois

1. **Taxonomie de couverture des scénarios critiques pour la sécurité des drones**
   Définir l'ODD à basse altitude du drone, les paramètres du scénario, la métrique de couverture et la taxonomie des défaillances.

2. **Algorithme de test accéléré guidé par la couverture**
   Générez des scénarios dangereux mais réalisables dans les trous de couverture et les régions à haute criticité.

3. **Référence réutilisable et protocole d'évaluation**
   L'utilisation de plusieurs planificateurs, de plusieurs références et de plusieurs niveaux de simulation prouve que cette référence peut évaluer de manière stable la sécurité des drones.

N'écrivez pas 6 à 8 contributions. Les trois éléments de l’introduction du journal sont les plus clairs.

### 8.3 F-J1 Le point le plus susceptible d'être rejeté| Risque | Parce que | Traitement |
|------|------|------|
| Considéré comme une simple plateforme de simulation | benchmark n'a aucune contribution algorithmique | les tests accélérés guidés par la couverture doivent être mis en avant |
| Considéré comme copié de la conduite autonome | Manque de fonctionnalités de drone | Accent sur la dynamique 3D, le vent, la batterie, les obstacles à basse altitude, les tâches d'atterrissage/d'urgence |
| La scène dangereuse est considérée comme irréaliste | contradictoire trop fort | Ajouter des filtres de faisabilité et de naturel |
| Considéré comme valable uniquement pour un seul agenda | Surapprentissage | Au moins 4 types de planificateurs |
| Considéré comme manquant d'importance en tant que système de transport | Les drones ne sont que des robots | Écrit comme évaluation de la sécurité des STI à basse altitude |

---

## 9. Démantèlement des tâches expérimentales récentes

### Semaine 1 : Geler la formulation du problème F-J1

- Journal cible fixe : T-ITS.
- Titre principal fixe et trois contributions.
- Congeler une cellule de test de 50 m x 50 m x 50 m.
- Définir la table des paramètres de la scène : géométrie, obstacle, agent dynamique, météo, capteur, tâche, étiquette de risque.

### Semaines 2-3 : Traitement de 76 millions de journaux d'exploration

- Échantillonnage de 10 000 à 50 000 éléments pour analyse préliminaire.
- Trous de couverture statistique.
- Modes de défaillance du clustering : collision, quasi-accident, délai d'attente, oscillation, violation d'énergie, scène infaisable.
- Générez deux cartes principales : la carte thermique de couverture et la taxonomie des échecs.

### Semaines 4 à 6 : Mise en œuvre de générateurs de référence- aléatoire, grille, LHS.
- Générateur contraint de style scénique.
-BO/CMA-ES.
- édition d'obstacles contradictoires.
- filtre de criticité réalisable.

### Semaines 7 à 9 : mise en œuvre de Cov-ATUAV

- mémoire de couverture.
- score de criticité.
- filtre de faisabilité.
-générateur adaptatif.
- harnais d'évaluation du planificateur.

### Semaines 10 à 12 : Expérience principale

- Comparez le taux de découverte de pannes, le gain de couverture, le taux d'invalidité et le facteur d'accélération.
- Test RRT*, MPC-lite, PPO, politique de vision.
- Effectuer la vérification de la migration des sous-ensembles AirSim/Flightmare/AvitBench.

### Semaines 13 à 16 : Rédaction de la première version du T-ITS

- L'introduction se concentre sur les tests de sécurité ITS à basse altitude.
- Les travaux connexes sont divisés en évaluation de la sécurité basée sur des scénarios, génération de scénarios critiques pour la sécurité, simulation d'UAV et évitement d'obstacles.
- Les expériences utilisent le tableau principal + le graphique de couverture + la courbe de découverte des pannes + le transfert entre simulateurs.

---

## 10. Choses non recommandées actuellement- N'écrivez pas 5 titres de dissertation à la fois pour ensuite travailler dessus en parallèle.
- Ne faites pas d'abord F-J4 city ODD, car le pipeline de données ralentira la sortie du premier article.
- Ne mélangez pas Shandong Expressway Emergency et F-J1 dans un seul article, sinon la ligne principale du T-ITS sera dispersée.
- N'écrivez pas 76 millions d'explorations comme résultat final, c'est désormais un actif de données, pas une conclusion.
- Ne vous contentez pas de signaler le taux de collisions, vous devez également signaler la couverture, la criticité, le taux d'invalidité et la stabilité du classement du planificateur.

---

## 11. Références

[1] Organisation internationale de normalisation. "ISO 34502:2022 Véhicules routiers — Scénarios de test pour les systèmes de conduite automatisés — Cadre d'évaluation de la sécurité basé sur des scénarios." 2022. URL : <https://www.iso.org/standard/78951.html>

[2] ASAM. « ASAM OpenSCENARIO DSL : terminologie clé et aperçu conceptuel. » URL : <https://publications.pages.asam.net/standards/ASAM_OpenSCENARIO/ASAM_OpenSCENARIO_DSL/latest/conceptual-overview/key_terms.html>

[3] ASAM. « ASAM OpenODD : modèle vers la référence de mappage DSL ASAM OpenSCENARIO. » URL : <https://publications.pages.asam.net/standards/ASAM_OpenODD/ASAM_OpenODD/latest/spécification/09_openscenario_dsl/09_01_overview.html>[4] Shuo Feng, Xintao Yan, Haowei Sun, Yiheng Feng et Henry X. Liu. « Test d'intelligence de conduite intelligente pour les véhicules autonomes dans un environnement naturaliste et conflictuel. » *Nature Communications*, 12:748, 2021. DOI : 10.1038/s41467-021-21007-8. URL : <https://doi.org/10.1038/s41467-021-21007-8>

[5] Shuo Feng, Yiheng Feng, Chunhui Yu, Yi Zhang et Henry X. Liu. "Génération de bibliothèque de scénarios de test pour les véhicules connectés et automatisés, partie I : méthodologie." *Transactions IEEE sur les systèmes de transport intelligents*, 22(3):1573-1582, 2021. DOI : 10.1109/TITS.2020.2972211. URL : <https://doi.org/10.1109/TITS.2020.2972211>[6] Shuo Feng, Yiheng Feng, Haowei Sun, Shan Bao, Yi Zhang et Henry X. Liu. "Génération de bibliothèque de scénarios de test pour les véhicules connectés et automatisés, partie II : études de cas." *Transactions IEEE sur les systèmes de transport intelligents*, 22(9):5635-5647, 2021. DOI : 10.1109/TITS.2020.2988309. URL : <https://doi.org/10.1109/TITS.2020.2988309>

[7] Daniel J. Fremont, Tommaso Dreossi, Shromona Ghosh, Xiangyu Yue, Alberto L. Sangiovanni-Vincentelli et Sanjit A. Seshia. « Scenic : un langage pour la spécification de scénarios et la génération de scènes. » *Actes de la 40e conférence ACM SIGPLAN sur la conception et la mise en œuvre de langages de programmation (PLDI)*, 2019. DOI : 10.1145/3314221.3314633. URL : <https://people.eecs.berkeley.edu/~sseshia/pubs/b2hd-fremont-pldi19.html>[8] Chejian Xu, Wenhao Ding, Weijie Lyu, Zuxin Liu, Shuai Wang, Yihan He, Hanjiang Hu, Ding Zhao et Bo Li. « SafeBench : une plateforme d'analyse comparative pour l'évaluation de la sécurité des véhicules autonomes. » *Avances in Neural Information Processing Systems 35 (NeurIPS 2022) Piste des ensembles de données et des benchmarks*, 2022. URL : <https://proceedings.neurips.cc/paper_files/paper/2022/hash/a48ad12d588c597f4725a8b84af647b5-Abstract-Datasets_and_Benchmarks.html>

[9] Quanyi Li, Zhenghao Peng, Lan Feng, Zhizheng Liu, Chenda Duan, Wenjie Mo et Bolei Zhou. « ScenarioNet : plate-forme open source pour la simulation et la modélisation de scénarios de trafic à grande échelle. » *Avances in Neural Information Processing Systems 36 (NeurIPS 2023) Piste des ensembles de données et des benchmarks*, 2023. URL : <https://proceedings.neurips.cc/paper_files/paper/2023/hash/0c26a501df8fb919a0350e2df06b5d39-Abstract-Datasets_and_Benchmarks.html>[10] Jingkang Wang, Ava Pun, James Tu, Sivabalan Manivasagam, Abbas Sadat, Sergio Casas, Mengye Ren et Raquel Urtasun. "AdvSim : Génération de scénarios critiques pour la sécurité pour les véhicules autonomes." *Actes de la conférence IEEE/CVF sur la vision par ordinateur et la reconnaissance de formes (CVPR)*, 2021. DOI : 10.1109/CVPR46437.2021.00978. URL : <https://openaccess.thecvf.com/content/CVPR2021/html/Wang_AdvSim_Generating_Safety-Critical_Scenarios_for_Self-Driving_Vehicles_CVPR_2021_paper.html>

[11] Niklas Hanselmann, Katrin Renz, Kashyap Chitta, Apratim Bhattacharyya et Andreas Geiger. « KING : Génération de scénarios de conduite critiques pour la sécurité pour une imitation robuste via des gradients cinématiques. » *Conférence européenne sur la vision par ordinateur (ECCV)*, 2022. DOI : 10.1007/978-3-031-19839-7_20. URL : <https://is.mpg.de/ps/publications/king_geiger2022>[12] Jiawei Zhang, Chejian Xu et Bo Li. "ChatScene : Génération de scénarios critiques pour la sécurité basée sur les connaissances pour les véhicules autonomes." *Actes de la conférence IEEE/CVF sur la vision par ordinateur et la reconnaissance de formes (CVPR)*, 2024, pp. 15459-15469. URL : <https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_ChatScene_Knowledge-Enabled_Safety-Critical_Scenario_Generation_for_Autonomous_Vehicles_CVPR_2024_paper.html>

[13] Keyu Chen, Yuheng Lei, Hao Cheng, Haoran Wu, Wenchao Sun et Sifa Zheng. "FREA : Génération guidée par la faisabilité de scénarios critiques pour la sécurité avec une rivalité raisonnable." arXiv :2406.02983, 2024. URL : <https://arxiv.org/abs/2406.02983>

[14] Shital Shah, Debadeepta Dey, Chris Lovett et Ashish Kapoor. "AirSim : simulation visuelle et physique haute fidélité pour les véhicules autonomes." *Robotique de terrain et de service*, Springer Proceedings in Advanced Robotics, 2017 ; arXiv : 1705.05065. URL : <https://arxiv.org/abs/1705.05065>[15] Yunlong Song, Selim Naji, Elia Kaufmann, Antonio Loquercio et Davide Scaramuzza. « Flightmare : un simulateur de quadrirotor flexible. » *Actes de la 4e Conférence sur l'apprentissage des robots (CoRL)*, PMLR 155, 2021. URL : <https://proceedings.mlr.press/v155/song21a.html>

[16] Hang Yu, Guido C.H.E. de Croon et Christophe De Wagter. «AvoidBench : une suite d'analyse comparative d'évitement d'obstacles basée sur la vision haute fidélité pour les multi-rotors.» arXiv :2301.07430, 2023. URL : <https://arxiv.org/abs/2301.07430>

[17] Botian Xu, Feng Gao, Chao Yu, Ruize Zhang, Yi Wu et Yu Wang. "OmniDrones : une plate-forme efficace et flexible pour l'apprentissage par renforcement dans le contrôle des drones." *Lettres IEEE sur la robotique et l'automatisation*, 9(3):2838-2844, 2024. DOI : 10.1109/LRA.2024.3356168. URL : <https://ieeexplore.ieee.org/document/10409589/>[18] Mihir Kulkarni, Theodor J. L. Forgaard et Kostas Alexis. "Aerial Gym : Isaac Gym Simulator pour robots aériens." arXiv :2305.16510, 2023. URL : <https://arxiv.org/abs/2305.16510>

[19] Yash Vardhan Pant, Max Z. Li, Alena Rodionova, Rhudii A. Quaye, Houssam Abbas, Megan S. Ryerson et Rahul Mangharam. « FADS : un cadre pour la sécurité des drones autonomes utilisant la planification de trajectoire basée sur la logique temporelle. » *Recherche sur les transports, partie C : technologies émergentes*, 130 :103275, 2021. DOI : 10.1016/j.trc.2021.103275. URL : <https://doi.org/10.1016/j.trc.2021.103275>

[20] Elsevier. « Recherche sur les transports, partie C : Technologies émergentes : objectifs et portée. » URL : <https://www.sciencedirect.com/journal/transportation-research-part-c-emerging-technologies>

[21] Société des systèmes de transport intelligents IEEE. "Transactions IEEE sur les systèmes de transport intelligents (T-ITS) : portée." URL : <https://ieee-itss.org/pub/t-its/>[22] Shandong Expressway Group Co., Ltd. « Le système de service de vol d’inspection complet de l’autoroute du Shandong est mis en ligne. » 2025. URL : <https://www.sdhsg.com/article/72553>

[23] Zhao Xiangmo, Zhao Yifei, Lu Nengchao et al. "Un examen de la recherche sur l'allocation des ressources clés en cas d'urgence en cas d'accident de la route." *Transactions d'ingénierie des transports*, 2024. DOI : 10.19818/j.cnki.1671-1637.2024.06.001. URL : <https://transport.chd.edu.cn/cn/article/doi/10.19818/j.cnki.1671-1637.2024.06.001>

[24] Marco C. Campi et Simone Garatti. "La faisabilité exacte des solutions randomisées de programmes convexes incertains." *SIAM Journal sur l'optimisation*, 19(3):1211-1230, 2008. DOI : 10.1137/07069821X. URL : <https://epubs.siam.org/doi/10.1137/07069821X>

[25] Anastasios N. Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei et Tal Schuster. «Contrôle conforme des risques». *Conférence internationale sur les représentations de l'apprentissage (ICLR)*, 2024. URL : <https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf>[26] Société des systèmes de transport intelligents IEEE. «Transactions IEEE sur les véhicules intelligents». URL : <https://ieee-itss.org/pub/t-iv/>

[27] Société de robotique et d'automatisation IEEE. « Transactions IEEE sur la science et l'ingénierie de l'automatisation. » URL : <https://www.ieee-ras.org/publications/t-ase>

[28] Société de fiabilité IEEE. « Transactions IEEE sur la fiabilité ». URL : <https://rs.ieee.org/publications/transactions-on-reliability/>

[29] Elsevier. « Ingénierie de la fiabilité et sécurité des systèmes : objectifs et portée. URL : <https://www.sciencedirect.com/journal/reliability-engineering-and-system-safety>

[30] Elsevier. « Science de la sécurité : objectifs et portée. » URL : <https://www.sciencedirect.com/journal/safety-science>

[31] Wiley. « Journal of Field Robotics : aperçu. URL : <https://onlinelibrary.wiley.com/journal/15564967>

[32] Elsevier. «Robotique et systèmes autonomes : objectifs et portée." URL : <https://www.sciencedirect.com/journal/robotics-and-autonomous-systems>[33] Elsevier. « Recherche sur les transports, partie E : Examen de la logistique et des transports : objectifs et portée. URL : <https://www.sciencedirect.com/journal/transportation-research-part-e-logistics-and-transportation-review>

[34] Société de robotique et d'automatisation IEEE. « Transactions IEEE sur la robotique ». URL : <https://www.ieee-ras.org/publications/t-ro>

[35] Société des systèmes de transport intelligents IEEE. «Journal ouvert IEEE des systèmes de transport intelligents». URL : <https://ieee-itss.org/pub/oj-its/>

---

## Annexe : Conclusion de cette optimisation

1. Lorsque la priorité est donnée à une revue, l'article F ne doit pas être réparti dans de nombreux petits articles.
2. Le premier article doit combiner des tests de référence et des tests accélérés pour former le document principal du T-ITS.
3. Les demandes d'urgence sur l'autoroute du Shandong doivent être indépendantes du TR-C et ne doivent pas être mélangées au premier article.
4. Les documents d'assurance des risques peuvent être utilisés comme réserves pour les revues de méthodes de haut niveau à un stade intermédiaire ou avancé.
5. L'ODD urbain vers la génération de scènes locales est temporairement classé quatrième et sera avancé une fois le pipeline de données stabilisé.