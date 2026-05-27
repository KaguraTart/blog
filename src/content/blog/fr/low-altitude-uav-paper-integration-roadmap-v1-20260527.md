---
title: "Matrice de thèse de planification à basse altitude v1 : intégration d'articles déjà écrits, sélection de sujets ultérieurs et liste de littérature Zotero"
description: "Intégrez la planification de trajectoires sans conflit, la planification à trois niveaux de centaines de drones, la planification de détection active 3DGS basée sur la théorie de l'information et d'autres instructions écrites, planifiez les groupes de documents de planification ultérieurs à basse altitude et fournissez les meilleures revues et références arXiv très pertinentes pour 2021-2026."
pubDate: 2026-05-27
updatedDate: 2026-05-27
tags: ["planification à basse altitude", "drone", "Planification de thèse", "Zotéro", "T-ITS", "TR-C", "T-RO", "AAAI", "ICRA", "3DGS", "MARNE"]
category: Tech
sourceHash: "210cd0f4500f05c3002dd5603dcac92ea4664512"
---

# Matrice de thèse de planification à basse altitude v1 : intégration d'articles déjà écrits, sélection de sujets ultérieurs et liste de littérature Zotero

> Cet article réintègre les articles sur les drones à basse altitude qui ont été rédigés jusqu'à présent dans un **portefeuille d'articles**.  
> L'objectif n'est pas d'éparpiller et d'écrire beaucoup d'idées, mais de clarifier : quels articles ont déjà pris forme, lesquels peuvent continuer à être transformés en articles de premier plan dans des revues/conférences régulières, et quels supports littéraires, actifs expérimentaux et positionnement de soumission sont nécessaires pour chaque article.

---

## 1. Il y a actuellement des articles et un positionnement de la ligne principale

Il existe actuellement trois articles principaux qui constituent la base du contenu :

| Numéro | Contenu existant | Positionnement actuel | Investissement principal recommandé | Jugement fondamental |
|---|---|---|---|---|
| Papier A | Planification de trajectoire sans conflit / PPO / MAPPO / Résolution de conflits multi-UAV | Résolution robuste des conflits dans le réseau de routes aériennes à basse altitude | IEEE T-ITS/IEEE T-RO/ICRA-IROS | Vous ne pouvez pas simplement écrire PPO, vous devez écrire « Compromis sécurité-efficacité sous les drones non coopératifs, dégradation des communications et couloirs à haute densité » |
| Papier B | Des centaines de drones de répartition à trois couches | Répartition des opérations de logistique/système d'urgence urbain à basse altitude | TR-C d'abord, sauvegarde T-ITS | Il s'agit d'un document sur le système de transport, axé sur la capacité, les retards, la stabilité des files d'attente, les goulots d'étranglement des vertiports/chargements/couloirs |
| Papier C | Planification de la détection active du drone 3DGS basée sur la théorie de l'information | Détection active + jumeau numérique basse altitude + planification en boucle fermée | T-RO / T-ITS / ICRA-IROS | S'il est soumis à un journal de transport, il doit être prouvé que la détection active améliore les indicateurs d'inspection, d'urgence, d'évitement d'obstacles ou de contrôle opérationnel |

Ces trois articles ont pu former un triangle de planification à basse altitude très stable :

```text
Paper A：战术安全
  多 UAV conflict resolution / no-conflict planning / PPO-MAPPO / CBF / RMADER

Paper B：系统运营
  hundred-UAV scheduling / queue stability / Lyapunov / multimodal logistics

Paper C：环境认知
  3DGS active perception / Fisher information / NBV / safe reconstruction
```

Il est préférable que les nouveaux articles ultérieurs s’étendent autour de ce triangle et ne démarrent pas dans une direction totalement indépendante.

---

## 2. Jugement global de soumission

La direction de la planification à basse altitude peut être divisée en trois catégories d'articles, et différentes catégories ont des normes d'évaluation différentes :| Tapez | Documents représentatifs | Attention à l'examen | Lieux recommandés |
|---|---|---|---|
| Documents sur le système de transport | Document B, allocation de ressources d'urgence, planification du réseau routier à basse altitude | Problèmes de trafic réels, indicateurs du système, crédibilité des données/simulations, implications politiques ou opérationnelles | TR-C, T-ITS |
| Document de planification des robots | Papier A, Papier C, planification du jumeau numérique | Nouveauté d'algorithme, temps réel, sécurité, vérification matériel/simulation | T-RO, RA-L+ICRA/IROS, T-ITS |
| Documents sur la méthode IA | VERA-UAV, CloudBrain-Agent, génération d'accélération de scène | difficulté du benchmark, théorie/mécanisme de vérification, généralisation du modèle, reproductibilité | Atelier AAAI, IJCAI, NeurIPS/ICLR, extension T-ITS |

Le positionnement officiel du TR-C met l'accent sur les systèmes de transport et les technologies émergentes, et le noyau intellectuel est du côté des transports [1] ; Les T-ITS couvrent la détection, les communications, les contrôles, la planification, la conception, la mise en œuvre et d'autres technologies de systèmes de transport modernes [2]. Par conséquent :

- **Papier B/Allocation des ressources d'urgence/Planification du réseau routier à basse altitude** : Prioriser l'écriture selon la logique de fonctionnement du système de transport du TR-C.
- **Papier A / Papier C** : Vous pouvez voter pour T-RO ou ICRA/IROS ; si vous passez au T-ITS, vous devez compléter les indicateurs du système de transport.
- **Paper E/G type LLM-Agent** : Le premier article est plus adapté à AAAI/IJCAI, et la version revue est étendue à T-ITS.

---

## 3. Matrice papier : Il est recommandé de former 7 articles pouvant être avancés

### 3.1 Document A : Planification robuste et sans conflit du réseau de routes aériennes à basse altitude

**Sujet suggéré :** Planification robuste d'un couloir de drones sans conflit dans un contexte de dégradation du trafic et des communications non coopérative**Correspondant aux articles existants :** Planification de chemin sans conflit, PPO/MAPPO, résolution de conflits UAV, construction d'environnement de conflit UAV.

**Question principale :** Dans un réseau aérien urbain à basse altitude, comment plusieurs drones peuvent-ils maintenir la sécurité de séparation tout en contrôlant les retards, les distances supplémentaires et les pertes de débit dans des conditions d'observation locale, de retards de communication, d'erreurs de positionnement et d'insertion d'aéronefs non coopératifs.

**Itinéraire de la méthode :**

- couche stratégique : attribution initiale de chemins et de créneaux horaires en fonction du réseau de routes ;
- couche tactique : vitesse de sortie MAPPO/PPO, hauteur ou action de décalage latéral ;
- bouclier de sécurité : contrôle de trajectoire type CBF-QP / ORCA / RMADER ;
- couche de secours : passer à une règle de priorité conservatrice lorsque la communication se dégrade ;
- Évaluation : 30/50 avions sont formés et 100/200 avions sont testés, couvrant quatre scénarios : coopératif, non coopératif, perte de communication et couloir à haute densité.

**Références clés :**

La formation stable multi-agents de MAPPO/PPO peut être soutenue par Yu et al. [3] ; MAT et FACMAC fournissent une base de référence MARL plus solide [4,5] ; HAPPO/HATRPO fournit une référence d'optimisation de politique multi-agent de région de confiance [6]. Du côté des robots, EGO-Swarm, MADER, RMADER, RACER, PANTHER et GCOPTER prennent respectivement en charge la planification d'essaim décentralisée, le partage de trajectoire sous délai, l'exploration collaborative, la planification consciente de la perception et l'optimisation de trajectoire multicoptère [7-12].

**Suggestions d'innovations :**1. Améliorer la « planification de trajectoire sans conflit PPO » d'une simple tâche RL au contrôle de la sécurité des couloirs de circulation à basse altitude.
2. Introduire une dégradation des communications et des drones non coopératifs pour former la limite opérationnelle réelle qui préoccupe davantage les T-ITS.
3. Utilisez une politique d'apprentissage + un bouclier formel/de sécurité pour éviter le manque de sécurité du RL pur.
4. Trafic des indicateurs : LoWC, NMAC, nombre de conflits, délai moyen, distance supplémentaire, débit, temps d'exécution.

### 3.2 Article B : Planification hiérarchique à trois niveaux de centaines de drones

**Sujet suggéré :** H-LyraUAV : Planification hiérarchique stable en file d'attente pour la logistique des drones à basse altitude à l'échelle des centaines

**Correspondant aux articles existants :** Planification d'ordonnancement à trois niveaux de l'article B.

**Question principale :** Comment une flotte d'UAV d'une centaine de niveaux peut-elle fonctionner de manière stable, efficace et sûre dans des conditions dynamiques, une capacité limitée de vertiport/chargement/couloir et des contraintes de transport multimodal.

**Itinéraire de la méthode :**

- couche macro : file d'attente des demandes, repositionnement de la flotte, choix du mode ;
- couche méso : vertiport, socle de chargement, planification des emplacements de couloir ;
- microcouche : faisabilité de la trajectoire énergie/sécurité/conscience des conflits ;
- Théorie : la dérive de Lyapunov plus pénalité garantit la stabilité de la file d'attente et un compromis coût-arriéré ;
- données : réseau urbain synthétique + amélioration OSM/POI/NYC TLC/Chicago taxi/SUMO.

**Références clés :**La gestion du trafic de livraison d'UAV à basse altitude TR-C a directement discuté de l'allocation des ressources et de la résolution des conflits dans l'espace urbain à basse altitude [13] ; la recherche sur l'UAM, l'équité et l'efficacité opérationnelle centrée sur les passagers soutient le cadrage de la qualité des services [14] ; réseau de livraison de bornes de recharge, planification UAM à capacité limitée, capacité d'infrastructure de soutien à la planification d'apprentissage sécurisée et planification en ligne sécurisée [15-17] ; La livraison multimodale camion-drone / UAV-UGV prend en charge l'extension multimodale [18,19].

**Suggestions d'innovations :**

1. Planification en ligne à trois niveaux en boucle fermée sur cent niveaux au lieu d'une conception de routage/réseau hors ligne.
2. La stabilité de la file d'attente devient l'axe principal de la théorie et le module d'apprentissage fait uniquement des prédictions ou des estimations de valeurs.
3. Évaluez simultanément les retards, le débit, l'arriéré, l'utilisation de la recharge, le goulot d'étranglement des vertiports et la congestion des couloirs.
4. La conclusion du système de trafic peut répondre : quand doit-il limiter le trafic, où se trouve le goulot d'étranglement et quand les drones uniquement sont-ils inférieurs au repli multimodal.

### 3.3 Document C : Planification de la détection active du drone FIM-3DGS

**Sujet suggéré :** FIM-3DGS : Planification de la perception active basée sur les informations des pêcheurs pour la reconstruction sûre des drones

**Correspond aux articles existants :** Paper C, Next-Best-View et NeRF/3DGS, Information Theory Active Sensing.

**Question principale :** Sous des contraintes de temps de vol, d'énergie et de sécurité limitées, comment les drones peuvent-ils sélectionner activement des points de vue pour faire converger la carte 3DGS plus rapidement et servir aux tâches de planification à basse altitude.

**Itinéraire de la méthode :**- représentation de la scène : éclaboussures gaussiennes 3D incrémentielles ;
- métrique d'information : construire des informations de Fisher / gain d'informations attendu pour les paramètres gaussiens ou jacobiens rendus ;
- planificateur : génération de candidats NBV + corridor sécurisé / contrainte CBF ;
- couplage de tâches : la qualité de la reconstruction n'est pas seulement rapportée sur le PSNR/SSIM, mais également sur le rappel des obstacles, la planification du taux de collision et la couverture d'inspection ;
- lignes de base : ActiveNeRF, FisherRF, GS-Planner, HGS-Planner, POp-GS, exploration des frontières.

**Références clés :**

Le texte original du 3DGS fournit une représentation explicite du champ de radiance en temps réel [20] ; ActiveNeRF est l'un des premiers représentants de la perception active du rendu neuronal [21] ; FisherRF prend directement en charge la sélection de vue active des informations Fisher et offre des résultats backend 3DGS à 70 ips [22] ; GS-Planner, HGS-Planner, POp-GS et NVF soutiennent la ligne de compétition 3DGS/NBV de 2024-2025 [23-26].

**Suggestions d'innovations :**

1. Passer de « 3DGS NBV » à « perception active au service de la planification de la sécurité des drones ».
2. Utilisez les informations de Fisher pour relier CRB / incertitude de reconstruction / sécurité de planification.
3. Passer des indicateurs visuels aux indicateurs de trafic/tâches du robot : taux de faisabilité du chemin, taux de rappel des obstacles, taux de couverture des inspections d'urgence.
4. Faire une généralisation inter-scénarios sur les cellules urbaines à basse altitude auto-construites de MatrixCity/AirSim/auto-construites.

### 3.4 Document D : Couverture des scènes critiques pour la sécurité à basse altitude et tests accélérés**Sujet suggéré :** Tests accélérés guidés par la couverture pour la navigation de drones à basse altitude critique en matière de sécurité

**Correspondant aux articles existants :** Couverture de scènes Paper F, génération de scènes dangereuses, 76 millions de journaux d'exploration.

**Question principale :** Comment définir l'espace de la scène de test de l'algorithme d'évitement/planification d'obstacles des drones à basse altitude, comment mesurer la couverture et comment découvrir efficacement des scénarios de défaillance dangereux mais efficaces.

**Itinéraire de la méthode :**

- grammaire du scénario : cellule locale de 50 m x 50 m x 50 m, combinaison d'obstacles, obstacles dynamiques, perturbation du vent, point cible, points de départ et d'arrivée ;
- métrique de couverture : couverture géométrique, couverture sémantique, couverture dynamique, couverture des risques, couverture des modes de défaillance ;
- tests accélérés : échantillonnage actif à partir des trous de couverture et probabilité de défaillance ;
- filtrage invalide : le filtrage est irréel, dangereux, invalide et inexécutable ;
- évaluation multi-planificateurs : Planificateur blindé A*/RRT*/MPC/ORCA/MAPPO/CBF.

**Références clés :**

Le NADE de Shuo Feng et la génération de bibliothèques de scénarios de test sont des références essentielles pour les tests accélérés et les bibliothèques de scénarios critiques pour la sécurité [27-29] ; SafeBench fournit une plate-forme de référence et un protocole de référence d'évaluation de la sécurité [30].

**Suggestions d'innovations :**

1. Migrez de l’ingénierie de scénarios de conduite autonome vers l’espace de scène 3D d’UAV à basse altitude.
2. Modélisez simultanément les trois objectifs de couverture, de criticité et de faisabilité.
3. Prouvez l'espace de couverture et la taxonomie des échecs à l'aide de 76 millions de journaux d'exploration.
4. Laissons les résultats répondre : quelles combinaisons d'obstacles sont les plus dangereuses, quelles planificateurs généralisent les pires et si une couverture accrue réduit réellement les risques inconnus.### 3.5 Epreuve E : Vérification de la planification linguistique du drone avec correction d'erreurs

**Sujet suggéré :** VERA-UAV : Planification du langage de vérification et de réparation pour les tâches d'UAV à basse altitude

**Correspondant aux articles existants :** Article E.

**Problème principal :** LLM peut convertir des tâches en langage naturel en spécifications de tâches exécutables pour drones, mais il est susceptible de produire des plans inexécutables, présentant une inadéquation sémantique ou violant les contraintes de sécurité. Nécessite des boucles de rétroaction IR, LTL/STL, de validation et de contre-exemple.

**Itinéraire de la méthode :**

- Instruction NL -> tapé TaskIR ;
- TaskIR -> LTL/STL ;
- Vérification Spot/RTAMT ;
- retour d'expérience contre-exemple/robustesse ;
- réparation itérative locale LLM ;
- vérification finale de la trajectoire.

**Références clés :**

Lang2LTL, NL2LTL, LTLCodeGen et ConformalNL2LTL prennent respectivement en charge la mise à la terre NL-LTL, la démonstration du système, la génération de logique temporelle de type génération de code et la garantie d'exactitude conforme [31-34].

**Suggestions d'innovations :**

1. Il ne s'agit pas seulement de NL2LTL, mais la trajectoire du drone peut effectuer une boucle fermée.
2. Typed TaskIR réduit l’ambiguïté du langage et améliore l’interprétabilité.
3. Les retours de contre-exemple et les retours de robustesse STL donnent à la réparation une direction spécifique.
4. La version AAAI/IJCAI se concentre sur la planification/vérification de l'IA ; T-ITS est étendu pour se connecter à des scénarios d’exploitation du trafic à basse altitude.

### 3.6 Article G : Agent LLM du cerveau des nuages de trafic à basse altitude

**Sujet suggéré :** CloudBrain-Agent : agents LLM améliorés par des outils pour l'exploitation du trafic à basse altitude

**Correspondant aux articles existants :** Papier G/G1.**Question principale :** Le cerveau du nuage de trafic à basse altitude ne peut pas être simplement un modèle de chat, mais un agent vérifiable qui peut appeler le planificateur, le planificateur de trajet, le vérificateur, le simulateur et l'évaluateur de risques.

**Itinéraire de la méthode :**

- LLM est responsable de la compréhension des tâches, de la sélection des outils, du résumé de l'état et de l'interprétation ;
- Les outils incluent le résolveur de conflits Paper A, le planificateur Paper B, le mappeur actif Paper C, le testeur de scénario Paper D, le vérificateur Paper E ;
- LowAltitudeIR comme représentation intermédiaire unifiée ;
- Le parcours technique donne la priorité aux grands modèles ordinaires + agent + compétences + MCP/utilisation d'outils, et se concentrera ultérieurement sur le domaine de LoRA/SFT ;
- Dans la première étape du déploiement, l'API est appelée pour former un benchmark, et dans la deuxième étape, le modèle local Qwen/DeepSeek est utilisé pour la reproduction et le contrôle des coûts.

**Références clés :**

UrbanGPT, UniST et TrafficGPT montrent que les tâches spatio-temporelles de transport/urbain ont commencé à se rapprocher des modèles de base et des cadres d'agents [35-37] ; bien que DriveLM soit une conduite autonome, sa forme de tâche Graph VQA peut apprendre du raisonnement en plusieurs étapes du cerveau des nuages ​​​​de trafic à basse altitude [38].

**Suggestions d'innovations :**

1. Le cerveau du nuage de trafic à basse altitude n'est pas un « modèle de conversation verticale », mais un agent vérifiable augmenté par un outil.
2. Utilisez l'IR unifié pour connecter la planification, la détection, la vérification et les tests de scénarios.
3. Effectuez d'abord l'analyse comparative de l'agent, puis décidez s'il convient d'affiner le modèle vertical pour réduire le risque du premier article.
4. Les indicateurs d'évaluation incluent la précision des appels d'outils, la réussite des tâches, les violations de sécurité, la réussite des réparations, la latence et l'auditabilité humaine.

### 3.7 Article H : ODD urbain à basse altitude et planification des zones fonctionnelles sémantiques

**Sujet suggéré :** ODD2Route : Modélisation sémantique de domaine de conception opérationnelle pour la planification d'itinéraires de drones à basse altitude

** Ceci est un nouvel article qui peut être écrit dans une nouvelle direction. ****Question principale :** Comment la scène urbaine globale s'adapte-t-elle à la planification d'itinéraires locaux à basse altitude ? Comment déterminer le risque, la capacité et la stratégie de service des routes aériennes à basse altitude en fonction des différents domaines fonctionnels, de la densité des bâtiments, de la structure routière, des activités de foule, des zones d'exclusion aérienne et de la répartition des installations d'urgence ?

**Itinéraire de la méthode :**

- ODD au niveau de la ville : OSM route/bâtiment/POI/utilisation du sol + proxy population/demande ;
- cellule de test locale : échantillon d'un scénario d'obstacle/trafic local en 3D provenant de l'ODD de la ville ;
- modèle de risque routier : canyons de construction, écoles et hôpitaux, pôles de transport, tronçons d'autoroute, zones d'exclusion aérienne ;
- résultats de la planification : corridor conscient des risques, couche d'altitude, site d'atterrissage d'urgence, candidats à la recharge/vertiport ;
- Évaluation : généralisation à travers les villes, comparaison du chemin le plus court naïf, A* conscient des risques, MILP multi-objectif et recommandation d'itinéraire basée sur l'apprentissage.

**Support littéraire :**

Cet article peut être étayé par la littérature TR-C/UAM de l’article B [13-19], la littérature de couverture de scénarios de l’article D [27-30] et la littérature sur les jumeaux 3D/numériques de l’article C [20-26]. La difficulté ne réside pas dans la complexité de l’algorithme, mais dans la définition fiable de l’ODD au niveau de la ville par rapport au risque de scénario/itinéraire local.

**Suggestions d'innovations :**

1. Établir une cartographie calculable entre la « scène urbaine globale » et la « combinaison d'obstacles locaux ».
2. Utilisez la couverture ODD pour interpréter la couverture de scène au lieu de générer des scènes de manière aléatoire.
3. Fournir un pont entre la planification urbaine à basse altitude, la conception d'itinéraires et la bibliothèque de scénarios de test pour TR-C/T-ITS.

---

## 4. Priorité des recommandations| Priorité | Articles | Actions récentes | Raisons |
|---|---|---|---|
| P0 | Papier B | Geler d'abord la formulation du problème, le modèle de file d'attente et le test de référence expérimental | Le plus similaire au document du système TR-C et le plus adapté à l'économie/à l'urgence à basse altitude |
| P0 | Papier A | Réécrire PPO/MAPPO dans un document solide sur la résolution des conflits à basse altitude | J'ai déjà la base de l'algorithme, mais j'ai besoin d'indicateurs de trafic et d'une base de référence solide |
| P1 | Papier C | Convergé vers Fisher + 3DGS + planification sécurisée, ne s'étend plus trop | L'algorithme est très innovant et peut être utilisé dans les robots/IA/ITS |
| P1 | Papier D | Réutilisez 76 millions de journaux d'exploration pour des tests guidés par la couverture | Les actifs de données sont uniques et peuvent facilement former une référence reproductible |
| P2 | Papier E | Maintenir le parcours papier de la méthode AAAI/IJCAI | Convient pour un travail court et rapide mais contrôle la portée des expériences |
| P2 | Papier G | Commencez une fois que l'interface de l'outil Paper A/B/C/D/E est stable | CloudBrain-Agent doit s'appuyer sur le module précédent, sinon il sera vide |
| P3 | Papier H | En tant qu'extension ultérieure de TR-C/T-ITS | Nécessite un pipeline de données urbaines mature et des définitions ODD |

---

## 5. Zotero organise le statut

Nom de la collection Zotero cible :

```text
低空规划论文参考
```

Actuellement, deux niveaux d'organisation ont été réalisés :| Projet | Statut |
|---|---|
| Collection Zotéro | existe déjà, la clé de collection est `FVHS3SKY`, le treeViewID local est `C17` |
| Lien de sélection locale Zotero | `zotero://select/library/collections/FVHS3SKY` |
| Documents importés | 38 éléments de premier niveau |
| répartition des types d'articles | `journalArticle` 16 éléments, `conferencePaper` 10 éléments, `document/preprint/webpage` 12 éléments |
| Sauvegarde locale BibTeX | `zotero/références-de-planification-basse-altitude-20260527.bib` |

La méthode d'importation utilise le serveur de connecteurs local de Zotero au lieu d'écrire directement « zotero.sqlite ». Le processus spécifique est :

1. Utilisez `pandoc` pour vérifier que BibTeX peut être analysé en tant que CSL JSON.
2. Importez `zotero/low-altitude-planning-references-20260527.bib` via Zotero local `/connector/import`.
3. Mettez à jour la collection cible de la session importée vers « C17 / Low Altitude Planning Paper Reference » via « /connector/updateSession ».
4. Utilisez l'API locale Zotero et SQLite en lecture seule pour vérifier qu'il existe 38 documents de niveau supérieur dans la collection.

Si vous continuez à ajouter des documents à l'avenir, il est recommandé de mettre à jour d'abord le BibTeX local, puis d'importer Zotero via le même processus d'importation/mise à jour du connecteur. Ne modifiez pas SQLite directement.

---

## 6. Plan d'exécution de suivi

### 6.1 Semaine 1 : Geler la matrice du papier- Confirmez si le papier A/B/C est la force principale des trois articles actuels.
- Confirmez si le papier D considère les 76 millions de journaux d'exploration comme un actif essentiel.
- Confirmez si le papier E/G continue d'être AAAI/IJCAI en premier.
- L'importation initiale de la collection Zotero est terminée ; l'étape suivante consiste à ajouter des PDF, des notes récapitulatives et des balises de priorité pour chaque article.

### 6.2 Semaine 2-3 : Compléter la matrice de littérature

- Compiler au moins 25 documents très pertinents pour chaque article principal.
- Chaque article forme une « matrice de travail associée » : problème, méthode, données, métrique, écart, notre angle.
- Pour l'épreuve A/B/C, cochez les épreuves qui « doivent reproduire la ligne de base » et « servir uniquement de travail connexe ».

### 6.3 Semaines 4 à 8 : Avancez d'abord les trois lignes expérimentales du papier B/A/C

- Papier B : benchmark de mise en file d'attente UAM synthétique + référence FCFS/greedy/MILP/backpression/MARL.
- Papier A : simulation de conflit de corridor + référence ORCA/CBF/RMADER/MAPPO.
- Papier C : pipeline 3DGS NBV + référence FisherRF/ActiveNeRF/GS-Planner/POp-GS.

### 6.4 Semaines 9 à 12 : Décider de votre première soumission

- Si la stabilité de la file d'attente du papier B et les résultats au niveau d'une centaine d'étagères sont les plus stables : votez d'abord pour TR-C.
- Si l'article A présente la sécurité et la généralisation des conflits les plus fortes : votez d'abord pour T-ITS/T-RO.
- Si l'article C a les résultats théoriques et visuels Fisher + 3DGS les plus forts : votez d'abord pour T-RO/ICRA/IROS.
- Si D dispose des meilleures données de couverture/découverte des pannes : investissez d'abord dans les T-ITS.

---

## 7. Références[1] Elsevier. *Recherche sur les transports, partie C : Technologies émergentes : objectifs et portée.* URL : <https://www.sciencedirect.com/journal/transportation-research-part-c-emerging-technologies>

[2] Société des systèmes de transport intelligents IEEE. *Transactions IEEE sur les systèmes de transport intelligents : portée.* URL : <https://ieee-itss.org/pub/t-its/>

[3] Chao Yu, Akash Velu, Eugene Vinitsky, Yu Wang, Alexandre M. Bayen et Yi Wu. «L'efficacité surprenante du PPO dans les jeux multi-agents coopératifs.» *Avances dans les systèmes de traitement de l'information neuronale*, 2022. URL : <https://arxiv.org/abs/2103.01955>

[4] Muning Wen, Jakub Grudzien Kuba, Runji Lin, Weinan Zhang, Ying Wen, Jun Wang et Yaodong Yang. "L'apprentissage par renforcement multi-agents est un problème de modélisation de séquence." *NeurIPS*, 2022. URL : <https://proceedings.neurips.cc/paper_files/paper/2022/hash/69413f87e5a34897cd010ca698097d0a-Abstract-Conference.html>[5] Bei Peng, Tabish Rashid, Christian Schroeder de Witt, Pierre-Alexandre Kamienny, Philip Torr, Wendelin Boehmer et Shimon Whiteson. « FACMAC : Gradients de politiques centralisés multi-agents factorisés. » *NeurIPS*, 2021. URL : <https://proceedings.neurips.cc/paper/2021/hash/65b9eea6e1cc6bb9f0cd2a47751a186f-Abstract.html>

[6] Jakub Grudzien Kuba, Ruiqing Chen, Muning Wen, Ying Wen, Fudan Sun, Jun Wang et Yaodong Yang. «Optimisation des politiques de région de confiance dans l'apprentissage par renforcement multi-agents». arXiv :2109.11251, 2021. URL : <https://arxiv.org/abs/2109.11251>

[7] Boyu Zhou, Xin Zhou, Jun Zhang, Fei Gao et Shaojie Shen. "EGO-Swarm : un système d'essaim à quatre rotors entièrement autonome et décentralisé dans des environnements encombrés." *ICRA*, 2021. DOI : 10.1109/ICRA48506.2021.9561902. URL : <https://arxiv.org/abs/2011.04183>[8] Jesus Tordesillas, Brett T. Lopez et Jonathan P. How. « MADER : Planificateur de trajectoire dans des environnements multiagents et dynamiques. » *Transactions IEEE sur la robotique*, 38(1):463-476, 2022. URL : <https://arxiv.org/abs/2010.11061>

[9] Kota Kondo, Reinaldo Figueroa, Juan Rached, Jesus Tordesillas, Parker C. Lusk et Jonathan P. How. "MADER robuste : planificateur de trajectoire multiagent décentralisé robuste aux retards de communication dans les environnements dynamiques." arXiv :2303.06222, 2023. URL : <https://arxiv.org/abs/2303.06222>

[10] Boyu Zhou, Hao Xu et Shaojie Shen. "RACER : Exploration collaborative rapide avec un système multi-UAV décentralisé." *Transactions IEEE sur la robotique*, 2023. DOI : 10.1109/TRO.2023.3236945. URL : <https://arxiv.org/abs/2209.08533>[11] Jésus Tordesillas et Jonathan P. Comment. « PANTHER : Planificateur de trajectoire sensible à la perception dans des environnements dynamiques. » *Accès IEEE*, 10 : 22662-22677, 2022. DOI : 10.1109/ACCESS.2022.3154037. URL : <https://arxiv.org/abs/2103.06372>

[12] Zhepei Wang, Xin Zhou, Chao Xu et Fei Gao. «Optimisation de trajectoire géométriquement contrainte pour les multicoptères». *Transactions IEEE sur la robotique*, 38(5):3259-3278, 2022. DOI : 10.1109/TRO.2022.3160022. URL : <https://arxiv.org/abs/2103.00190>

[13] Ang Li, Mark Hansen et Bo Zou. «Gestion du trafic et allocation des ressources pour la livraison de colis par drone dans l'espace urbain à basse altitude.» *Recherche sur les transports, partie C : technologies émergentes*, 143 : 103808, 2022. DOI : 10.1016/j.trc.2022.103808. URL : <https://doi.org/10.1016/j.trc.2022.103808>[14] Mehdi Bennaceur, Rémi Delmas et Youssef Hamadi. « Mobilité aérienne urbaine centrée sur les passagers : compromis en matière d'équité et d'efficacité opérationnelle. » *Recherche sur les transports, partie C : technologies émergentes*, 136 :103519, 2022. DOI : 10.1016/j.trc.2021.103519. URL : <https://doi.org/10.1016/j.trc.2021.103519>

[15] Roberto Pinto et Alexandra Lagorio. « Conception d'un réseau de livraison point à point basé sur des drones avec des stations de recharge intermédiaires. » *Recherche sur les transports, partie C : technologies émergentes*, 135 :103506, 2022. DOI : 10.1016/j.trc.2021.103506. URL : <https://doi.org/10.1016/j.trc.2021.103506>

[16] Qinshuang Wei, Gustav Nilsson et Samuel Coogan. « Planification de la mobilité aérienne urbaine à capacité limitée. » arXiv :2107.02900, 2021. URL : <https://arxiv.org/abs/2107.02900>[17] Surya Murthy, Natasha A. Neogi et Suda Bharadwaj. « Planification de la mobilité aérienne urbaine grâce à un apprentissage sécurisé. » arXiv :2209.15457, NASA NTRS, 2022. URL : <https://arxiv.org/abs/2209.15457>

[18] Jiahao Xing, Tong Guo et Lu Tong. "Routage camion-drone fiable avec synchronisation dynamique : une approche de programmation réseau de grande dimension." *Recherche sur les transports, partie C : technologies émergentes*, 165 :104698, 2024. DOI : 10.1016/j.trc.2024.104698. URL : <https://doi.org/10.1016/j.trc.2024.104698>

[19] Bolong Zhou, Wei Zeng et Hai Yang. "Conception de réseau de livraison UAV-UGV multi-voyages avec délais de sortie." *Recherche sur les transports, partie C : technologies émergentes*, 181 : 105389, 2025. DOI : 10.1016/j.trc.2025.105389. URL : <https://doi.org/10.1016/j.trc.2025.105389>[20] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler et George Drettakis. « Éclaboussures gaussiennes 3D pour le rendu du champ de rayonnement en temps réel. » *Transactions ACM sur graphiques / SIGGRAPH*, 42(4), 2023. DOI : 10.1145/3592433. URL : <https://arxiv.org/abs/2308.04079>

[21] Xuran Pan, Zihang Lai, Shiji Song et Gao Huang. "ActiveNeRF : Apprendre où voir avec une estimation de l'incertitude." *ECCV*, 2022. URL : <https://arxiv.org/abs/2209.08546>

[22] Wen Jiang, Boshu Lei et Kostas Daniilidis. « FisherRF : sélection de vue active et cartographie avec des champs de radiance à l'aide des informations de Fisher. » *ECCV*, 2024. DOI : 10.1007/978-3-031-72624-8_24. URL : <https://eccv.ecva.net/virtual/2024/oral/1226>

[23] Rui Jin, Yuman Gao, Yingjian Wang, Haojian Lu et Fei Gao. "GS-Planner : un cadre de planification basé sur les éclaboussures gaussiennes pour une reconstruction active haute fidélité." arXiv :2405.10142, 2024. URL : <https://arxiv.org/abs/2405.10142>[24] Zijun Xu, Rui Jin, Ke Wu, Yi Zhao, Zhiwei Zhang, Jieru Zhao, Fei Gao, Zhongxue Gan et Wenchao Ding. "HGS-Planner : cadre de planification hiérarchique pour la reconstruction de scènes actives à l'aide d'éclaboussures gaussiennes 3D." arXiv :2409.17624, 2024. URL : <https://arxiv.org/abs/2409.17624>

[25] Joey Wilson, Marcelino Almeida, Sachit Mahajan, Martin Labrie, Maani Ghaffari, Omid Ghasemalizadeh, Min Sun, Cheng-Hao Kuo et Arnab Sen. « POp-GS : prochaine meilleure vue en éclaboussures gaussiennes 3D avec P-Optimality. » *CVPR*, 2025. URL : <https://cvpr.thecvf.com/virtual/2025/poster/34708>

[26] Shangjie Xue, Jesse Dill, Pranay Mathur, Frank Dellaert, Panagiotis Tsiotras et Danfei Xu. « Champ de visibilité neuronale pour la cartographie active basée sur l'incertitude. » *CVPR*, 2024. URL : <https://arxiv.org/abs/2406.06948>[27] Shuo Feng, Xintao Yan, Haowei Sun, Yiheng Feng et Henry X. Liu. « Test d'intelligence de conduite intelligente pour les véhicules autonomes dans un environnement naturaliste et conflictuel. » *Nature Communications*, 12:748, 2021. DOI : 10.1038/s41467-021-21007-8. URL : <https://www.nature.com/articles/s41467-021-21007-8>

[28] Shuo Feng, Yiheng Feng, Chao Yu, Yi Zhang et Henry X. Liu. « Génération d'une bibliothèque de scénarios de tests pour les véhicules connectés et automatisés, partie I : Méthodologie. » *Transactions IEEE sur les systèmes de transport intelligents*, 2021. DOI : 10.1109/TITS.2020.2972211. URL : <https://doi.org/10.1109/TITS.2020.2972211>

[29] Shuo Feng, Yiheng Feng, Haowei Sun, Yi Zhang et Henry X. Liu. "Génération de bibliothèque de scénarios de test pour les véhicules connectés et automatisés, partie II : études de cas." *Transactions IEEE sur les systèmes de transport intelligents*, 2021. DOI : 10.1109/TITS.2020.2988309. URL : <https://doi.org/10.1109/TITS.2020.2988309>[30] Chejian Xu, Wenhao Ding, Baiming Li, Xinqing Chen, Ding Zhao, Bo Li, Jiajun Liu et Hang Zhao. « SafeBench : une plateforme d'analyse comparative pour l'évaluation de la sécurité des véhicules autonomes. » *Ensembles de données et benchmarks NeurIPS*, 2022. URL : <https://proceedings.neurips.cc/paper_files/paper/2022/hash/a48ad12d588c597f4725a8b84af647b5-Abstract-Datasets_and_Benchmarks.html>

[31] Jason Liu, Ankit Shah, Eric Rosen et Stefanie Tellex. « Lang2LTL : traduction des commandes en langage naturel en spécifications de tâches temporelles de robot. » *PMLR/CoRL*, 229, 2023. URL : <https://proceedings.mlr.press/v229/liu23d.html>

[32] Francesco Fuggitti et Tathagata Chakraborti. "NL2LTL : un package Python pour convertir des instructions en langage naturel en formules logiques temporelles linéaires." *Démonstration AAAI*, 37(13):16428-16430, 2023. DOI : 10.1609/aaai.v37i13.27068. URL : <https://ojs.aaai.org/index.php/AAAI/article/view/27068>[33] Behrad Rabiei et Mahesh A. Kumar. « LTLCodeGen : génération de code de logique temporelle syntaxiquement correcte pour la planification des tâches du robot. » arXiv :2503.07902, 2025. URL : <https://arxiv.org/abs/2503.07902>

[34] Jun Wang, David Smith Sundarsingh, Jyotirmoy V. Deshmukh et Yiannis Kantaros. "ConformalNL2LTL : traduction d'instructions en langage naturel en formules logiques temporelles avec des garanties d'exactitude conforme." arXiv :2504.21022, 2025. URL : <https://arxiv.org/abs/2504.21022>

[35] Zhonghang Li, Lianghao Xia, Jiabin Tang, Yong Xu, Lei Shi, Long Xia, Dawei Yin et Chao Huang. "UrbanGPT : grands modèles de langage spatio-temporels." arXiv :2403.00813, 2024. URL : <https://arxiv.org/abs/2403.00813>

[36] Yuan Yuan, Jingtao Ding, Jie Feng, Depeng Jin et Yong Li. « UniST : un modèle universel optimisé pour la prévision spatio-temporelle urbaine. » *KDD*, 2024. DOI : 10.1145/3637528.3671662. URL : <https://dblp.org/rec/conf/kdd/0032D0J024>[37] Jinhui Ouyang, Yijie Zhu, Xiang Yuan et Di Wu. "TrafficGPT : vers une analyse et une génération de trafic à plusieurs échelles avec un cadre d'agents spatio-temporels." arXiv :2405.05985, 2024. URL : <https://arxiv.org/abs/2405.05985>

[38] Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Jens Beisswenger, Ping Luo, Andreas Geiger et Hongyang Li. « DriveLM : Conduire avec des réponses visuelles aux questions sous forme de graphique. » *ECCV*, 2024. URL : <https://github.com/OpenDriveLab/DriveLM>