---
title: "Matrice de documents de planification à basse altitude v2 : trois articles en cours, suivi incarné d'itinéraires à basse altitude et de grands modèles"
description: "Avec trois articles en cours sur la planification de trajectoires sans conflit, la planification à trois couches de centaines d'UAV et la planification de détection active 3DGS basée sur la théorie de l'information comme noyau, nous replanifierons l'itinéraire de l'article de suivi sur le cerveau nuageux incorporé à basse altitude et à basse altitude, le réglage fin du grand modèle vertébral, l'accélération de l'inférence et la collaboration logicielle et matérielle."
pubDate: 2026-05-27
updatedDate: 2026-05-28
tags: ["planification à basse altitude", "drone", "Planification de thèse", "Zotéro", "T-ITS", "TR-C", "T-RO", "AAAI", "ICRA", "3DGS", "MARNE", "IA incarnée", "VLA", "LLM", "Accélération d'inférence"]
category: Tech
sourceHash: "46302954b2010a293b0edfe46fe54f398cd68bf3"
---

# Matrice de documents de planification à basse altitude v2 : trois articles en cours, suivi incarné d'itinéraires à basse altitude et de grands modèles

> Cet article réintègre les articles sur les drones à basse altitude qui ont été rédigés jusqu'à présent dans un **portefeuille d'articles**.  
> L'objectif n'est pas d'éparpiller et d'écrire beaucoup d'idées, mais de clarifier : quels articles ont déjà pris forme, lesquels peuvent continuer à être transformés en articles de premier plan dans des revues/conférences régulières, et quels supports littéraires, actifs expérimentaux et positionnement de soumission sont nécessaires pour chaque article.

---

## 0. 2026-05-28 Correction Conclusion

L'orientation actuelle doit être modifiée : au lieu de "planifier 7 à 10 articles en même temps", il s'agit d'abord de reconnaître que **trois articles sont déjà en cours d'élaboration**, et que les articles suivants doivent naturellement se développer à partir des atouts de ces trois articles.

Les trois articles sur lesquels je travaille actuellement sont :

| Statut | Article | Rôle | La ligne principale qui ne peut pas être déviée de |
|---|---|---|---|
| Je travaille déjà sur | Document A : Planification de trajectoires sans conflit / PPO-MAPPO / Résolution des conflits à basse altitude | Couche de sécurité tactique | Corridors à haute densité et basse altitude, drones non coopératifs, dégradation de la communication/positionnement, compromis entre sécurité et efficacité |
| Déjà en cours | Article B : Planification hiérarchique à trois niveaux de 100 drones | Couche d'exploitation du système | Flotte de 100 niveaux, stabilité des files d'attente, goulot d'étranglement vertiport/charge/couloir, planification multimodale |
| Je travaille déjà sur | Article C : Détection active UAV 3DGS basée sur la théorie de l'information | Couche cognitive environnementale | 3DGS / Informations sur les pêcheurs / NBV / reconstruction sécurisée / cartographie tenant compte de la planification |

Ne commencez pas un nouvel article pour des articles ultérieurs qui n'ont rien à voir avec la direction. Le bon itinéraire est :1. **Commencez par diviser A/B/C en trois documents principaux qui peuvent être soumis. **
2. **Les documents de suivi D/F/G/H/I ne sont utilisés que comme extensions de A/B/C** : la couverture de scène prend en charge A/C, le cerveau nuageux à basse altitude connecte A/B/C en série, la basse altitude incarnée connecte la perception de C et le contrôle de A dans une boucle fermée, et le réglage fin du modèle et l'inférence accélèrent la mise en œuvre du cerveau de service cloud.
3. **Les instructions générales de l'AGI ne peuvent pas être écrites sous forme de déclarations creuses**. Une expression plus stable est « vers une intelligence générale incarnée à basse altitude » : en commençant par l'agent de domaine, l'invocation d'outils, le retour de simulation, le VLA/VLN, le modèle mondial et le raisonnement côté appareil, et en se rapprochant progressivement de l'intelligence générale incarnée.
4. **Il n'est pas recommandé de former un modèle de fondation verticale à partir de zéro dans la première étape**. Utilisez d'abord un grand modèle ordinaire + Agent + Compétences/MCP + RAG + vérificateur + post-traitement simulateur pour former une boucle fermée expérimentale reproductible ; attendez qu'il y ait suffisamment de trajectoires d'appel d'outils, d'échantillons de défaillance et de retours de simulation avant d'effectuer les réglages précis de LoRA/SFT/DPO/GRPO.

Cela signifie que les articles ultérieurs doivent être divisés en deux niveaux :

| Niveau | Objectif de la thèse | Faut-il commencer dans un avenir proche |
|---|---|---|
| Couche principale A/B/C | Déjà en cours, la boucle fermée expérimentale doit d'abord être complétée | Immédiatement |
| Superposition de scène D | Fournit des données de référence, une taxonomie des pannes et des données critiques pour la sécurité à A/C | Récent |
| Couche d'agent cérébral G Cloud | Transformez A/B/C/D/E en outils pour créer un cerveau de nuage de trafic à basse altitude vérifiable | Moyen terme |
| H Couche incorporée de basse altitude | Créez un modèle UAV VLN/VLA/monde et connectez-vous à l'IA universelle incarnée | Mi-parcours et plus tard |
| Je modélise la couche de formation | Formation LowAltitudeGPT / utilisation des outils / LowAltitudeIR / retour d'expérience de simulation | Attendez que les données se stabilisent |
| Couche d'accélération d'inférence J | vLLM/TensorRT-LLM/quantisation/collaboration appareil-cloud/déploiement matériel | Attendez que la charge de travail de l'agent soit stable |

---

## 1. Il y a actuellement des articles et un positionnement de la ligne principale

Il existe actuellement trois articles principaux qui constituent la base du contenu :| Numéro | Contenu existant | Positionnement actuel | Investissement principal recommandé | Jugement fondamental |
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

## 3. Matrice papier : 3 articles déjà en cours + parcours d'extension de suivi

La lecture de cette section doit être modifiée : les articles A/B/C sont les trois principaux travaux déjà en cours, et non « de nouvelles orientations pour le futur ». Les papiers D/E/G/H/I/J sont des extensions inscriptibles, mais la séquence de démarrage doit obéir à la maturité des actifs expérimentaux de A/B/C.

### 3.1 Document A : Planification robuste et sans conflit du réseau de routes aériennes à basse altitude**Sujet suggéré :** Planification robuste d'un couloir de drones sans conflit dans un contexte de dégradation du trafic et des communications non coopérative

**Correspondant aux articles existants :** Planification de chemin sans conflit, PPO/MAPPO, résolution de conflits UAV, construction d'environnement de conflit UAV.

**Question principale :** Dans un réseau aérien urbain à basse altitude, comment plusieurs drones peuvent-ils maintenir la sécurité de séparation tout en contrôlant les retards, les distances supplémentaires et les pertes de débit dans des conditions d'observation locale, de retards de communication, d'erreurs de positionnement et d'insertion d'aéronefs non coopératifs.

**Itinéraire de la méthode :**

- couche stratégique : attribution initiale de chemins et de créneaux horaires en fonction du réseau de routes ;
- couche tactique : vitesse de sortie MAPPO/PPO, hauteur ou action de décalage latéral ;
- bouclier de sécurité : contrôle de trajectoire type CBF-QP / ORCA / RMADER ;
- couche de secours : passer à une règle de priorité conservatrice lorsque la communication se dégrade ;
- Évaluation : 30/50 avions sont formés et 100/200 avions sont testés, couvrant quatre scénarios : coopératif, non coopératif, perte de communication et couloir à haute densité.

**Références clés :**La formation stable multi-agents de MAPPO/PPO peut être soutenue par Yu et al. [3] ; MAT et FACMAC fournissent une base de référence MARL plus solide [4,5] ; HAPPO/HATRPO fournit une référence d'optimisation de politique multi-agent de région de confiance [6]. Du côté des robots, EGO-Swarm, MADER, RMADER, RACER, PANTHER et GCOPTER prennent respectivement en charge la planification d'essaim décentralisée, le partage de trajectoire sous délai, l'exploration collaborative, la planification consciente de la perception et l'optimisation de trajectoire multicoptère [7-12].

**Suggestions d'innovations :**

1. Améliorer la « planification de trajectoire sans conflit PPO » d'une simple tâche RL au contrôle de la sécurité des couloirs de circulation à basse altitude.
2. Introduire une dégradation des communications et des drones non coopératifs pour former la limite opérationnelle réelle qui préoccupe davantage les T-ITS.
3. Utilisez une politique d'apprentissage + un bouclier formel/de sécurité pour éviter le manque de sécurité du RL pur.
4. Trafic des indicateurs : LoWC, NMAC, nombre de conflits, délai moyen, distance supplémentaire, débit, temps d'exécution.

### 3.2 Article B : Planification hiérarchique à trois niveaux de centaines de drones

**Sujet suggéré :** H-LyraUAV : Planification hiérarchique stable en file d'attente pour la logistique des drones à basse altitude à l'échelle des centaines

**Correspondant aux articles existants :** Planification d'ordonnancement à trois niveaux de l'article B.

**Question principale :** Comment une flotte d'UAV d'une centaine de niveaux peut-elle fonctionner de manière stable, efficace et sûre dans des conditions dynamiques, une capacité limitée de vertiport/chargement/couloir et des contraintes de transport multimodal.

**Itinéraire de la méthode :**- couche macro : file d'attente des demandes, repositionnement de la flotte, choix du mode ;
- couche méso : vertiport, socle de chargement, planification des emplacements de couloir ;
- microcouche : faisabilité de la trajectoire énergie/sécurité/conscience des conflits ;
- Théorie : la dérive de Lyapunov plus pénalité garantit la stabilité de la file d'attente et un compromis coût-arriéré ;
- données : réseau urbain synthétique + amélioration OSM/POI/NYC TLC/Chicago taxi/SUMO.

**Références clés :**

La gestion du trafic de livraison d'UAV à basse altitude TR-C a directement discuté de l'allocation des ressources et de la résolution des conflits dans l'espace urbain à basse altitude [13] ; la recherche sur l'UAM, l'équité et l'efficacité opérationnelle centrée sur les passagers soutient le cadrage de la qualité des services [14] ; réseau de livraison de bornes de recharge, planification UAM à capacité limitée, capacité d'infrastructure de soutien à la planification d'apprentissage sécurisée et planification en ligne sécurisée [15-17] ; La livraison multimodale camion-drone / UAV-UGV prend en charge l'extension multimodale [18,19].

**Suggestions d'innovations :**1. Planification en ligne à trois niveaux en boucle fermée sur cent niveaux au lieu d'une conception de routage/réseau hors ligne.
2. La stabilité de la file d'attente devient l'axe principal de la théorie et le module d'apprentissage fait uniquement des prédictions ou des estimations de valeurs.
3. Évaluez simultanément les retards, le débit, l'arriéré, l'utilisation de la recharge, le goulot d'étranglement des vertiports et la congestion des couloirs.
4. La conclusion du système de trafic peut répondre : quand doit-il limiter le trafic, où se trouve le goulot d'étranglement et quand les drones uniquement sont-ils inférieurs au repli multimodal.

### 3.3 Document C : Planification de la détection active du drone FIM-3DGS

**Sujet suggéré :** FIM-3DGS : Planification de la perception active basée sur les informations des pêcheurs pour la reconstruction sûre des drones

**Correspond aux articles existants :** Paper C, Next-Best-View et NeRF/3DGS, Information Theory Active Sensing.

**Question principale :** Sous des contraintes de temps de vol, d'énergie et de sécurité limitées, comment les drones peuvent-ils sélectionner activement des points de vue pour faire converger la carte 3DGS plus rapidement et servir aux tâches de planification à basse altitude.

**Itinéraire de la méthode :**

- représentation de la scène : éclaboussures gaussiennes 3D incrémentielles ;
- métrique d'information : construire des informations de Fisher / gain d'informations attendu pour les paramètres gaussiens ou jacobiens rendus ;
- planificateur : génération de candidats NBV + corridor sécurisé / contrainte CBF ;
- couplage de tâches : la qualité de la reconstruction n'est pas seulement rapportée sur le PSNR/SSIM, mais également sur le rappel des obstacles, la planification du taux de collision et la couverture d'inspection ;
- lignes de base : ActiveNeRF, FisherRF, GS-Planner, HGS-Planner, POp-GS, exploration des frontières.

**Références clés :**Le texte original du 3DGS fournit une représentation explicite du champ de radiance en temps réel [20] ; ActiveNeRF est l'un des premiers représentants de la perception active du rendu neuronal [21] ; FisherRF prend directement en charge la sélection de vue active des informations Fisher et offre des résultats backend 3DGS à 70 ips [22] ; GS-Planner, HGS-Planner, POp-GS et NVF soutiennent la ligne de compétition 3DGS/NBV de 2024-2025 [23-26].

**Suggestions d'innovations :**

1. Passer de « 3DGS NBV » à « perception active au service de la planification de la sécurité des drones ».
2. Utilisez les informations de Fisher pour relier CRB / incertitude de reconstruction / sécurité de planification.
3. Passer des indicateurs visuels aux indicateurs de trafic/tâches du robot : taux de faisabilité du chemin, taux de rappel des obstacles, taux de couverture des inspections d'urgence.
4. Faire une généralisation inter-scénarios sur les cellules urbaines à basse altitude auto-construites de MatrixCity/AirSim/auto-construites.

### 3.4 Document D : Couverture des scènes critiques pour la sécurité à basse altitude et tests accélérés

**Sujet suggéré :** Tests accélérés guidés par la couverture pour la navigation de drones à basse altitude critique en matière de sécurité

**Correspondant aux articles existants :** Couverture de scènes Paper F, génération de scènes dangereuses, 76 millions de journaux d'exploration.

**Question principale :** Comment définir l'espace de la scène de test de l'algorithme d'évitement/planification d'obstacles des drones à basse altitude, comment mesurer la couverture et comment découvrir efficacement des scénarios de défaillance dangereux mais efficaces.

**Itinéraire de la méthode :**- grammaire du scénario : cellule locale de 50 m x 50 m x 50 m, combinaison d'obstacles, obstacles dynamiques, perturbation du vent, point cible, points de départ et d'arrivée ;
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
4. Laissons les résultats répondre : quelles combinaisons d'obstacles sont les plus dangereuses, quelles planificateurs généralisent les pires et si une couverture accrue réduit réellement les risques inconnus.

### 3.5 Epreuve E : Vérification de la planification linguistique du drone avec correction d'erreurs

**Sujet suggéré :** VERA-UAV : Planification du langage de vérification et de réparation pour les tâches d'UAV à basse altitude

**Correspondant aux articles existants :** Article E.

**Problème principal :** LLM peut convertir des tâches en langage naturel en spécifications de tâches exécutables pour drones, mais il est susceptible de produire des plans inexécutables, présentant une inadéquation sémantique ou violant les contraintes de sécurité. Nécessite des boucles de rétroaction IR, LTL/STL, de validation et de contre-exemple.**Itinéraire de la méthode :**

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

**Correspondant aux articles existants :** Papier G/G1.

**Question principale :** Le cerveau du nuage de trafic à basse altitude ne peut pas être simplement un modèle de chat, mais un agent vérifiable qui peut appeler le planificateur, le planificateur de trajet, le vérificateur, le simulateur et l'évaluateur de risques.

**Itinéraire de la méthode :**- LLM est responsable de la compréhension des tâches, de la sélection des outils, du résumé de l'état et de l'interprétation ;
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

** Ceci est un nouvel article qui peut être écrit dans une nouvelle direction. **

**Question principale :** Comment la scène urbaine globale s'adapte-t-elle à la planification d'itinéraires locaux à basse altitude ? Comment déterminer le risque, la capacité et la stratégie de service des routes aériennes à basse altitude en fonction des différents domaines fonctionnels, de la densité des bâtiments, de la structure routière, des activités de foule, des zones d'exclusion aérienne et de la répartition des installations d'urgence ?**Itinéraire de la méthode :**

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

### 3.8 Article I : Intelligence embarquée à basse altitude et VLA/VLN aériens

**Sujet suggéré :** Intelligence incorporée à basse altitude : vision, langage et planification d'action pour les drones dans l'espace aérien urbain

**Il s'agit de l'orientation à moyen et long terme qu'il convient de retenir après la nouvelle enquête. **La ligne principale actuelle de l’intelligence incarnée est passée du « LLM parlant » au « VLM/VLA reliant directement la perception, le langage et l’action ». RT-2 a clairement proposé le modèle vision-langage-action, plaçant la vision, le langage et l'action du robot dans le même paradigme de modèle [44] ; OpenVLA et Octo ont montré que la politique open source VLA/robots généralistes peut être pré-entraînée avec des trajectoires de robots à grande échelle, puis affinée avec une petite quantité de données de domaine cible [42,43]. Des travaux directement liés ont également commencé à apparaître dans le domaine des drones : SINGER utilise le Splatting gaussien pour générer un langage intégrant des données de simulation de vol, former la politique VLN du drone à bord et réaliser des expériences matérielles [39] ; FlightGPT utilise SFT + GRPO pour faire du UAV VLN, et vérifie la généralisation et le raisonnement interprétable sur CityNav [40] ; UAV-VLN relie le langage naturel, la perception visuelle et la planification de trajectoire réalisable [41].

**Notre écart d'écriture :**

La plupart des VLN/VLA aériens existants se concentrent sur « donner une cible linguistique et laisser le drone voler près de la cible ». Ce n’est pas la capacité requise par le cerveau des nuages ​​​​de trafic à basse altitude. Les scénarios à basse altitude nécessitent que le modèle comprenne à la fois :

- ODD urbain à basse altitude, structure de l'espace aérien, zones d'exclusion aérienne et zones à risques ;
- Statut du trafic multi-UAV, capacité du couloir et règles d'évitement des collisions ;
- Priorité des tâches d'urgence, des tâches d'inspection et des tâches logistiques ;
- Cartes visuelles incomplètes, erreurs de positionnement, dégradation de la communication et cibles non coopératives ;
- Le résultat doit être vérifiable, contrôlable et dégradable, plutôt qu'une opération de boîte noire de bout en bout.

**Méthode suggérée :**

```text
multimodal observation
  = UAV RGB/depth/semantic map/3DGS local map
  + low-altitude traffic state
  + natural-language mission
  + city ODD metadata

LLM/VLM/VLA policy
  -> LowAltitudeIR
  -> skill selection
  -> waypoint / velocity / route command
  -> verifier + safety shield
  -> simulator or hardware feedback
```

**Version recommandée à faire en premier :**

Ne formez pas initialement AerialVLA de bout en bout. Créez d’abord un **agent incorporé hybride** :

- Les managers de haut niveau utilisent le modèle Qwen/DeepSeek/API pour la compréhension des tâches et l'invocation des outils ;
- La couche intermédiaire appelle le résolveur de conflits du papier A, le planificateur du papier B, le mappeur actif du papier C et le vérificateur du papier E ;
- La couche inférieure utilise un contrôleur traditionnel/MPC/CBF Shield pour garantir la sécurité en temps réel ;
- Les données de formation proviennent de trajectoires de simulation, de planificateurs experts, de journaux de réparation de pannes et de tâches d'annotation manuelles.

**Cibles disponibles :**- ICRA/IROS/T-RO : accent sur la navigation intégrée, la boucle fermée matérielle et la simulation vers le réel.
- AAAI/IJCAI : mettre l'accent sur la planification des agents, l'utilisation des outils et les commentaires de vérification.
- T-ITS : accent sur les opérations de trafic à basse altitude, les interventions d'urgence, la résolution des conflits et les indicateurs système.

### 3.9 Article J : Formation LowAltitudeGPT et itinéraire de réglage fin

**Sujet suggéré :** LowAltitudeGPT : utilisation d'outils et réglage des commentaires de simulation pour l'intelligence du trafic à basse altitude

**Jugement fondamental :**

Il n’est pas temps de former à partir de zéro le « grand modèle de trafic à basse altitude ». Cela pose trois problèmes :

1. La quantité de données est insuffisante pour soutenir les contributions au niveau du modèle de base ;
2. L'examen demandera si la contribution du modèle dépasse celle des grands modèles ordinaires + RAG + utilisation d'outils ;
3. Le coût de la formation est élevé, mais il n’est pas nécessairement plus précieux que la boucle fermée agent/vérificateur/simulateur.

Un itinéraire plus réalisable est **grand modèle ordinaire + Agent + Compétences/MCP + RAG + vérificateur + simulateur** pour l'exécuter d'abord, puis précipiter les journaux d'exécution en données pouvant être entraînées. MCP est essentiellement une interface standard qui expose les outils et le contexte à LLM, et convient pour un accès unifié aux ordonnanceurs, planificateurs, vérificateurs, simulateurs, bases de données et bibliothèques de documents [47].

Un examen des grands modèles économiques à basse altitude décompose également les systèmes à basse altitude en réseaux d'installations, réseaux d'information, réseaux de routes et réseaux de services, et souligne que les grands modèles doivent être combinés avec l'informatique de pointe, la 6G/ISAC et une intelligence distribuée fiable [50]. Cela montre que notre article ne peut pas simplement être rédigé comme « la formation d'un modèle de chat », mais doit être rédigé comme une boucle fermée de modèles, d'outils, de réseaux, de contrôle des opérations et d'évaluation du système.

**Suggestions de sélection de modèles :**| Scène | Modèle recommandé | Raison |
|---|---|---|
| Exploration de solutions / génération de données / enseignant | Modèle d'API haute capacité | Générez rapidement des tâches, des traces d'outils, des explications de contre-exemples et des échantillons d'évaluation en premier, sans utiliser l'API comme dépendance finale reproductible |
| Expériences locales reproductibles | Qwen3-8B / Qwen3-14B / Qwen3-32B | Qwen3 prend officiellement en charge les processus locaux d'exploitation, de déploiement, de quantification et de formation, avec une bonne langue chinoise, un bon appel d'outils et une écologie d'ingénierie [45] |
| Raisonnement/Mathématiques/Interprétation des contraintes | DeepSeek-R1-Distill-Qwen-14B / 32B | La série DeepSeek-R1 met l'accent sur les capacités de raisonnement motivées par RL. La version distill peut être déployée localement et est affinée sur la base du modèle open source Qwen/Llama [46] |
| Perception multimodale à basse altitude | Qwen-VL / Qwen3-VL / autre VLM open source | Compréhension sémantique des images, des images vidéo, des cartes, des graphiques de suivi et du rendu 3DGS |
| Petit modèle Edge-End | Qwen3-4B / 8B version quantitative, SLM | Utilisé pour le résumé de l'état final, la détection des anomalies et le repli à faible latence |

**Conception des données de formation :**

| Type de données | Source | Objectif de formation |
|---|---|---|
| Mission NL -> LowAltitudeIR | Modèle manuel + professeur API + réécriture de tâches réelles | Analyse des tâches et représentation structurée |
| trace d'utilisation des outils | Journal d'appels de l'outil papier A/B/C/D/E | Apprenez quand appeler planification, planification, vérification, simulation |
| contre-exemple du vérificateur | Retour d'information Spot/RTAMT/CBF/simulateur | Apprenez à réparer les plans inexécutables ou dangereux |
| déploiement de simulations | SUMO/AirSim/simulation à basse altitude auto-développée | Apprenez à expliquer les goulots d'étranglement du système à partir des résultats |
| cas d'échec | collision, LoWC, timeout, explosion de file d'attente, consommation d'énergie insuffisante | Apprenez le diagnostic des risques et la désescalade des urgences |
| données d'audit humain | Sélection manuelle de solutions plus raisonnables | Optimisation DPO/préférences |

**Phase de formation :**1. **RAG + invite de base** : pas de réglage fin, utilisez uniquement la bibliothèque de littérature, les réglementations, la description du système et le schéma de l'outil.
2. **LoRA/QLoRA SFT** : formation NL-to-IR, appel d'outil, interprétation des résultats et réparation de contre-exemples.
3. **DPO/IPO** : utilisez les préférences manuelles ou les préférences de notation du vérificateur pour optimiser « sûr, exécutable, concis et explicable ».
4. **Réglage de style GRPO/RL** : utilisez la simulation pour récompenser le taux de réussite des tâches de formation, la faible violation, la faible latence et la conformité du format. La route SFT + GRPO de FlightGPT peut être utilisée comme référence UAV VLN [40].
5. **distillation** : distillez les capacités du modèle enseignant/32B de l'API vers 8B/4B pour un déploiement local et périphérique.

**Indicateurs d'évaluation :**

-réussite de la tâche ;
- Correspondance exacte/sémantique LowAltitudeIR ;
- précision/rappel d'appel d'outil ;
- taux du plan exécutable ;
- taux d'infractions à la sécurité ;
- taux de réussite des réparations ;
- taux d'hallucinations ;
- latence/coût du jeton ;
- généralisation cross-city/cross-tâches ;
- taux de réussite à l'audit humain.

### 3.10 Article K : Accélération de l'inférence cérébrale dans les nuages à basse altitude et collaboration logicielle et matérielle

**Sujet suggéré :** Inférence co-optimisée Edge-Cloud pour le trafic à basse altitude Agents Cloud-Brain

**Pourquoi cet article peut-il être écrit :**Si nous voulons créer à la fois du logiciel et du matériel à l’avenir, l’accélération de l’inférence ne peut pas simplement être une optimisation technique. Il doit être écrit comme suit : **Problème de service intelligent en temps réel sous contraintes du système de trafic à basse altitude** : il existe de grands modèles et des états globaux du côté du cloud, de faibles contraintes de latence et de confidentialité/communication du côté de la périphérie, et des contraintes de consommation d'énergie, de puissance de calcul, de dissipation thermique et de contrôle en temps réel du côté des drones. Les agents intelligents aériens à usage général ont donné un signal direct en direction de la co-conception matériel-logiciel : le modèle 14B intégré fonctionne à environ 5 à 6 jetons/sec, a une consommation électrique maximale d'environ 220 W et adopte une architecture cognitive bidirectionnelle de planification LLM lente + contrôle de réaction rapide [51].

**Architecture du système :**

```text
cloud brain
  - full LLM / VLM
  - global scheduler
  - long-horizon planner
  - batch simulation evaluator

edge station / vertiport
  - quantized 8B/14B model
  - local RAG cache
  - route/conflict verifier
  - streaming state summarizer

onboard UAV
  - tiny policy / controller
  - VIO / obstacle avoidance
  - emergency fallback
  - compressed semantic state uplink
```

**Voie technologique accélérée :**

- Serveur : vLLM/PagedAttention/traitement continu/cache de préfixe. La valeur fondamentale de PagedAttention est de réduire le gaspillage du cache KV et d'améliorer le débit de traitement par lots [48].
- Déploiement de production GPU NVIDIA : TensorRT-LLM, réalisation d'inférence LLM avec les moteurs TensorRT, runtime Python/C++ et optimisation GPU [49].
- Fin/bord : AWQ/GPTQ/GGUF INT4/INT8, compression cache KV, décodage spéculatif, routeur petit modèle.
- Optimisation des appels d'outils : mise en cache du schéma de l'outil, mise en cache des résultats de recherche RAG statiques et compilation des appels d'outils à haute fréquence en compétences déterministes.
- Direction de l'opérateur : noyau d'attention, cache KV paginé, séparation pré-remplissage/décodage, planificateur de lots, routage expert MoE, mise en cache de l'encodeur de vision.

**Points de thèse disponibles :**1. **Document système** : profilage de latence/coût/énergie de la charge de travail des agents cloud brain à basse altitude.
2. **Algorithm-System Paper** : Sélection dynamique de l'API/cloud 32B/edge 14B/onboard 4B en fonction du risque de la tâche.
3. **Document opérateur/inférence** : optimisation du cache KV et du traitement par lots pour le trafic à basse altitude avec plusieurs agents, plusieurs outils, un contexte long et des mises à jour d'état en streaming.
4. **Document de collaboration matérielle** : déploiement à trois niveaux Jetson Orin/station de travail RTX/GPU cloud, évaluation des jetons/s, de la latence de bout en bout, de l'énergie par décision et du taux de repli de sécurité.

**Lieu recommandé :**

- Systèmes de transport partiels : T-ITS / IEEE IoT Journal.
- Intelligence de pointe partielle : IEEE TMC / IEEE Internet of Things Journal / ACM TECS.
- Système robotique partiel : article système IROS/ICRA.
- Opérateurs et systèmes partiels : à partir de l'atelier MLSys / SC / atelier DAC/DATE, il n'est pas recommandé d'accéder directement à la conférence système supérieure au début.

---

## 4. Priorité des recommandations| Priorité | Articles | Actions récentes | Raisons |
|---|---|---|---|
| P0-Actif | Papier B | Formulation du problème de gel, modèle de file d'attente, benchmark expérimental | Le plus similaire au document du système TR-C et le plus adapté à l'économie/à l'urgence à basse altitude |
| P0-Actif | Papier A | Réécrire le PPO/MAPPO dans un document robuste sur la résolution des conflits à basse altitude | Dispose déjà d'une base d'algorithme, mais nécessite des indicateurs de trafic et une base de référence solide |
| P0-Actif | Papier C | Convergé vers Fisher + 3DGS + planification sécurisée, ne s'étend plus trop | L'algorithme est innovant et peut être utilisé dans les robots/IA/ITS |
| Prise en charge P1 | Papier D | Réutilisez 76 millions de journaux d'exploration et effectuez des tests guidés par la couverture | Fournir des scénarios critiques pour la sécurité, une taxonomie des pannes et des références pour la climatisation |
| Pont P1 | Papier G | Créez d'abord l'interface de l'outil et le benchmark CloudBrain-Agent | Enchaînez A/B/C/D/E dans un cerveau nuageux à basse altitude au lieu d'un modèle de discussion vide |
| P2-Incarné | Papier I | Réalisation d'un pilote aérien VLN/VLA à petite échelle : données de simulation, trajectoires expertes, comparaison de stratégies de bout en bout/hybrides | Il s’agit de la ligne principale menant à l’AGI incarnée, mais elle nécessite d’abord de stabiliser la perception de l’A/C et les outils de sécurité |
| Modèle P2 | Papier J | Précipitez LowAltitudeIR, la trace de l'outil, les commentaires du vérificateur, puis effectuez LoRA/SFT/GRPO | Ayez d'abord des données en boucle fermée, puis affinez le modèle vertical |
| Système P3 | Papier K | Attendez que la charge de travail CloudBrain-Agent soit corrigée avant de procéder à une collaboration vLLM/TensorRT/quantization/end-cloud | L'orientation logicielle et matérielle peut être écrite, mais cela nécessite une réelle charge de travail pour être comme un papier |
| Planification P3 | Papier H | En tant qu'extension ultérieure de TR-C/T-ITS | Nécessite un pipeline de données urbaines mature et une définition ODD |

**Suggestions d'ordre d'exécution :**1. Ne changez pas le champ de bataille principal actuel : A/B/C continue d'avancer.
2. Complétez d'abord le document D, car il améliore directement la crédibilité expérimentale de la climatisation et peut également générer des données de formation de modèle ultérieures.
3. Créez à nouveau Paper G et regroupez A/B/C/D/E dans un cerveau cloud basé sur des outils.
4. Paper I/J/K Ne vous précipitez pas pour démarrer un grand projet ; faites d'abord un petit pilote et un schéma de données. Avant de commencer la vraie question, vous devez répondre : d'où viennent les données, quels sont les indicateurs d'évaluation et si elles peuvent être plus puissantes que les grands modèles ordinaires + les appels d'outils.

---

## 4.1 Matrice de support de la littérature

Afin d'éviter l'empilement des documents, les références actuelles 51 sont utilisées de manière fermée selon le sens du papier :| Itinéraire | Groupes de documentation | Utilisation |
|---|---|---|
| Positionnement du système de soumission et de transport | [1,2] | Déterminer les différences de cadrage de TR-C / T-ITS |
| Article A : Résolution de conflits multi-agents | [3-12] | Base de référence PPO/MAPPO, MAT/FACMAC/HAPPO et EGO-Swarm/MADER/RMADER/RACER/PANTHER/GCOPTER |
| Document B : Planification de centaines de drones | [13-19] | Allocation des ressources de livraison à basse altitude, planification UAM, apprentissage sécurisé, livraison multimodale camion-drone/UAV-UGV |
| Article C : Détection active 3DGS | [20-26] | 3DGS, ActiveNeRF, FisherRF, GS-Planner, HGS-Planner, POp-GS, NVF |
| Document D : Couverture des scénarios critiques pour la sécurité | [27-30] | Tests accélérés Shuo Feng, bibliothèque de scénarios, SafeBench |
| Papier E : Planification et vérification linguistiques | [31-34] | Lang2LTL, NL2LTL, LTLCodeGen, ConformalNL2LTL |
| Document G : Agent cerveau nuageux à basse altitude | [35-38,47,50,51] | UrbanGPT/UniST/TrafficGPT/DriveLM, MCP, revue de grands modèles économiques à basse altitude, agent intelligent aérien |
| Article I : VLA incarnée à basse altitude/aérienne | [39-44] | SINGER, FlightGPT, UAV-VLN, OpenVLA, Octo, RT-2 |
| Article J : Formation et mise au point du modèle | [40,45,46,47,50] | Référence SFT/GRPO, Qwen3, DeepSeek-R1, MCP/utilisation d'outils, positionnement de système grand modèle à basse altitude |
| Paper K : Accélération d'inférence et collaboration logicielle et matérielle | [45,48,49,51] | Écologie de déploiement Qwen3, vLLM/PagedAttention, TensorRT-LLM, contraintes matérielles de l'agent aérien 14B intégré|---

## 5. Zotero organise le statut

Nom de la collection Zotero cible :

```text
低空规划论文参考
```

Actuellement, deux niveaux d'organisation ont été réalisés :

| Projet | Statut |
|---|---|
| Collection Zotéro | existe déjà, la clé de collection est `FVHS3SKY`, le treeViewID local est `C17` |
| Lien de sélection locale Zotero | `zotero://select/library/collections/FVHS3SKY` |
| Documents importés | 51 éléments de premier niveau |
| répartition des types d'articles | `journalArticle` 17 éléments, `conferencePaper` 11 éléments, `document/preprint/webpage` 23 éléments |
| Sauvegarde locale BibTeX | `zotero/low-altitude-planning-references-20260527.bib` ; Incrément : `zotero/low-altitude-planning-references-update-20260528.bib` |

La méthode d'importation utilise le serveur de connecteurs local de Zotero au lieu d'écrire directement « zotero.sqlite ». Le processus spécifique est :

1. Utilisez `pandoc` pour vérifier que BibTeX peut être analysé en tant que CSL JSON.
2. Importez `zotero/low-altitude-planning-references-20260527.bib` via Zotero local `/connector/import`.
3. Mettez à jour la collection cible de la session importée vers « C17 / Low Altitude Planning Paper Reference » via « /connector/updateSession ».
4. Utilisez l'API locale Zotero et SQLite en lecture seule pour vérifier qu'il existe 51 documents de niveau supérieur dans la collection.Si vous continuez à ajouter des documents à l'avenir, il est recommandé de mettre à jour d'abord le BibTeX local, puis d'importer Zotero via le même processus d'importation/mise à jour du connecteur. Ne modifiez pas SQLite directement.

---

## 6. Plan d'exécution de suivi

### 6.1 Semaine 1 : Geler trois articles en cours

- Il est clair que l'article A/B/C est le pipeline actif actuel, et les articles suivants ne seront plus rédigés avec la même priorité.
- Papier A : Geler les scénarios de conflit, l'espace d'action, les indicateurs de référence et de trafic.
- Papier B : modèle de file d'attente gelée, objectif Lyapunov, benchmark synthétique et cadrage TR-C.
- Papier C : Geler les interfaces théoriques et les métriques tenant compte de la planification pour FIM/3DGS/NBV.
- L'importation initiale de la collection Zotero et l'importation incrémentielle du 2026-05-28 sont terminées ; l'étape suivante consiste à ajouter des PDF, des notes récapitulatives et des balises de priorité pour chaque article.

### 6.2 Semaines 2-3 : Complément de la matrice de littérature et vérification ultérieure de la nouveauté de l'itinéraire

- Compiler au moins 25 documents très pertinents pour chaque article principal.
- Chaque article forme une « matrice de travail associée » : problème, méthode, données, métrique, écart, notre angle.
- Pour l'épreuve A/B/C, cochez les épreuves qui « doivent reproduire la ligne de base » et « servir uniquement de travail connexe ».
- Effectuez la vérification de la nouveauté sur le papier I/J/K séparément :
  - Papier I : VLN aérien, AerialVLA, SINGER, FlightGPT, OpenVLA, Octo, RT-2 ;
  - Papier J : Qwen3, DeepSeek-R1, réglage de l'utilisation des outils, MCP, RAG, formation simulation-feedback ;
  - Paper K : vLLM, TensorRT-LLM, quantification, cache KV, déploiement edge-cloud.

### 6.3 Semaines 4 à 8 : Avancez d'abord les trois lignes expérimentales du papier B/A/C- Papier B : benchmark de mise en file d'attente UAM synthétique + référence FCFS/greedy/MILP/backpression/MARL.
- Papier A : simulation de conflit de corridor + référence ORCA/CBF/RMADER/MAPPO.
- Papier C : pipeline 3DGS NBV + référence FisherRF/ActiveNeRF/GS-Planner/POp-GS.
- Papier D : réalisez uniquement un pilote léger, organisez 76 millions de journaux d'exploration dans une taxonomie de couverture/défaillance et ne rivalisez pas pour les ressources A/B/C.

### 6.4 Semaines 9 à 12 : Construction de la boucle fermée minimale du cerveau des nuages à basse altitude

- Exposer A/B/C/D/E comme interfaces d'outils : planificateur, résolveur de conflits, mappeur actif, testeur de scénarios, vérificateur.
- Définir « LowAltitudeIR » pour unifier les résultats des appels de mission, d'espace aérien, de drone, de ressources, de risques et d'outils.
- Utilisez d'abord l'enseignant d'API + Qwen/DeepSeek local pour créer la référence CloudBrain-Agent, et ne vous précipitez pas pour l'affiner.
- Collectez la trace des outils, la réparation des pannes et le déploiement de la simulation en tant que données de formation pour Paper J.

### 6,5 semaines 13 à 20 : décider de la soumission et des itinéraires modèles- Si la stabilité de la file d'attente du papier B et les résultats au niveau d'une centaine d'étagères sont les plus stables : votez d'abord pour TR-C.
- Si l'article A présente la sécurité et la généralisation des conflits les plus fortes : votez d'abord pour T-ITS/T-RO.
- Si l'article C a les résultats théoriques et visuels Fisher + 3DGS les plus forts : votez d'abord pour T-RO/ICRA/IROS.
- Si le papier D possède les meilleures données de couverture/découverte de pannes : investissez d'abord dans les T-ITS.
- Si CloudBrain-Agent peut déjà appeler les outils A/B/C/D/E de manière stable : démarrez la version AAAI/IJCAI.
- Si 5 000 à 20 000 traces d'outils de haute qualité, commentaires du vérificateur et déploiement de simulation ont été accumulés : démarrez LowAltitudeGPT LoRA/SFT.
- Si la charge de travail de l'agent est fixe et que la latence devient un goulot d'étranglement : démarrez l'expérience de quantification vLLM/TensorRT/edge de Paper K.

---

## 7. Références

[1] Elsevier. *Recherche sur les transports, partie C : Technologies émergentes : objectifs et portée.* URL : <https://www.sciencedirect.com/journal/transportation-research-part-c-emerging-technologies>

[2] Société des systèmes de transport intelligents IEEE. *Transactions IEEE sur les systèmes de transport intelligents : portée.* URL : <https://ieee-itss.org/pub/t-its/>[3] Chao Yu, Akash Velu, Eugene Vinitsky, Yu Wang, Alexandre M. Bayen et Yi Wu. «L'efficacité surprenante du PPO dans les jeux multi-agents coopératifs.» *Avances dans les systèmes de traitement de l'information neuronale*, 2022. URL : <https://arxiv.org/abs/2103.01955>

[4] Muning Wen, Jakub Grudzien Kuba, Runji Lin, Weinan Zhang, Ying Wen, Jun Wang et Yaodong Yang. "L'apprentissage par renforcement multi-agents est un problème de modélisation de séquence." *NeurIPS*, 2022. URL : <https://proceedings.neurips.cc/paper_files/paper/2022/hash/69413f87e5a34897cd010ca698097d0a-Abstract-Conference.html>

[5] Bei Peng, Tabish Rashid, Christian Schroeder de Witt, Pierre-Alexandre Kamienny, Philip Torr, Wendelin Boehmer et Shimon Whiteson. « FACMAC : Gradients de politiques centralisés multi-agents factorisés. » *NeurIPS*, 2021. URL : <https://proceedings.neurips.cc/paper/2021/hash/65b9eea6e1cc6bb9f0cd2a47751a186f-Abstract.html>[6] Jakub Grudzien Kuba, Ruiqing Chen, Muning Wen, Ying Wen, Fudan Sun, Jun Wang et Yaodong Yang. «Optimisation des politiques de région de confiance dans l'apprentissage par renforcement multi-agents». arXiv :2109.11251, 2021. URL : <https://arxiv.org/abs/2109.11251>

[7] Boyu Zhou, Xin Zhou, Jun Zhang, Fei Gao et Shaojie Shen. "EGO-Swarm : un système d'essaim à quatre rotors entièrement autonome et décentralisé dans des environnements encombrés." *ICRA*, 2021. DOI : 10.1109/ICRA48506.2021.9561902. URL : <https://arxiv.org/abs/2011.04183>

[8] Jesus Tordesillas, Brett T. Lopez et Jonathan P. How. « MADER : Planificateur de trajectoire dans des environnements multiagents et dynamiques. » *Transactions IEEE sur la robotique*, 38(1):463-476, 2022. URL : <https://arxiv.org/abs/2010.11061>[9] Kota Kondo, Reinaldo Figueroa, Juan Rached, Jesus Tordesillas, Parker C. Lusk et Jonathan P. How. "MADER robuste : planificateur de trajectoire multiagent décentralisé robuste aux retards de communication dans les environnements dynamiques." arXiv :2303.06222, 2023. URL : <https://arxiv.org/abs/2303.06222>

[10] Boyu Zhou, Hao Xu et Shaojie Shen. "RACER : Exploration collaborative rapide avec un système multi-UAV décentralisé." *Transactions IEEE sur la robotique*, 2023. DOI : 10.1109/TRO.2023.3236945. URL : <https://arxiv.org/abs/2209.08533>

[11] Jésus Tordesillas et Jonathan P. Comment. « PANTHER : Planificateur de trajectoire sensible à la perception dans des environnements dynamiques. » *Accès IEEE*, 10 : 22662-22677, 2022. DOI : 10.1109/ACCESS.2022.3154037. URL : <https://arxiv.org/abs/2103.06372>[12] Zhepei Wang, Xin Zhou, Chao Xu et Fei Gao. «Optimisation de trajectoire géométriquement contrainte pour les multicoptères». *Transactions IEEE sur la robotique*, 38(5):3259-3278, 2022. DOI : 10.1109/TRO.2022.3160022. URL : <https://arxiv.org/abs/2103.00190>

[13] Ang Li, Mark Hansen et Bo Zou. «Gestion du trafic et allocation des ressources pour la livraison de colis par drone dans l'espace urbain à basse altitude.» *Recherche sur les transports, partie C : technologies émergentes*, 143 : 103808, 2022. DOI : 10.1016/j.trc.2022.103808. URL : <https://doi.org/10.1016/j.trc.2022.103808>

[14] Mehdi Bennaceur, Rémi Delmas et Youssef Hamadi. « Mobilité aérienne urbaine centrée sur les passagers : compromis en matière d'équité et d'efficacité opérationnelle. » *Recherche sur les transports, partie C : technologies émergentes*, 136 :103519, 2022. DOI : 10.1016/j.trc.2021.103519. URL : <https://doi.org/10.1016/j.trc.2021.103519>[15] Roberto Pinto et Alexandra Lagorio. « Conception d'un réseau de livraison point à point basé sur des drones avec des stations de recharge intermédiaires. » *Recherche sur les transports, partie C : technologies émergentes*, 135 :103506, 2022. DOI : 10.1016/j.trc.2021.103506. URL : <https://doi.org/10.1016/j.trc.2021.103506>

[16] Qinshuang Wei, Gustav Nilsson et Samuel Coogan. « Planification de la mobilité aérienne urbaine à capacité limitée. » arXiv :2107.02900, 2021. URL : <https://arxiv.org/abs/2107.02900>

[17] Surya Murthy, Natasha A. Neogi et Suda Bharadwaj. « Planification de la mobilité aérienne urbaine grâce à un apprentissage sécurisé. » arXiv :2209.15457, NASA NTRS, 2022. URL : <https://arxiv.org/abs/2209.15457>[18] Jiahao Xing, Tong Guo et Lu Tong. "Routage camion-drone fiable avec synchronisation dynamique : une approche de programmation réseau de grande dimension." *Recherche sur les transports, partie C : technologies émergentes*, 165 :104698, 2024. DOI : 10.1016/j.trc.2024.104698. URL : <https://doi.org/10.1016/j.trc.2024.104698>

[19] Bolong Zhou, Wei Zeng et Hai Yang. "Conception de réseau de livraison UAV-UGV multi-voyages avec délais de sortie." *Recherche sur les transports, partie C : technologies émergentes*, 181 : 105389, 2025. DOI : 10.1016/j.trc.2025.105389. URL : <https://doi.org/10.1016/j.trc.2025.105389>

[20] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler et George Drettakis. « Éclaboussures gaussiennes 3D pour le rendu du champ de rayonnement en temps réel. » *Transactions ACM sur graphiques / SIGGRAPH*, 42(4), 2023. DOI : 10.1145/3592433. URL : <https://arxiv.org/abs/2308.04079>[21] Xuran Pan, Zihang Lai, Shiji Song et Gao Huang. "ActiveNeRF : Apprendre où voir avec une estimation de l'incertitude." *ECCV*, 2022. URL : <https://arxiv.org/abs/2209.08546>

[22] Wen Jiang, Boshu Lei et Kostas Daniilidis. « FisherRF : sélection de vue active et cartographie avec des champs de radiance à l'aide des informations de Fisher. » *ECCV*, 2024. DOI : 10.1007/978-3-031-72624-8_24. URL : <https://eccv.ecva.net/virtual/2024/oral/1226>

[23] Rui Jin, Yuman Gao, Yingjian Wang, Haojian Lu et Fei Gao. "GS-Planner : un cadre de planification basé sur les éclaboussures gaussiennes pour une reconstruction active haute fidélité." arXiv :2405.10142, 2024. URL : <https://arxiv.org/abs/2405.10142>

[24] Zijun Xu, Rui Jin, Ke Wu, Yi Zhao, Zhiwei Zhang, Jieru Zhao, Fei Gao, Zhongxue Gan et Wenchao Ding. "HGS-Planner : cadre de planification hiérarchique pour la reconstruction de scènes actives à l'aide d'éclaboussures gaussiennes 3D." arXiv :2409.17624, 2024. URL : <https://arxiv.org/abs/2409.17624>[25] Joey Wilson, Marcelino Almeida, Sachit Mahajan, Martin Labrie, Maani Ghaffari, Omid Ghasemalizadeh, Min Sun, Cheng-Hao Kuo et Arnab Sen. « POp-GS : prochaine meilleure vue en éclaboussures gaussiennes 3D avec P-Optimality. » *CVPR*, 2025. URL : <https://cvpr.thecvf.com/virtual/2025/poster/34708>

[26] Shangjie Xue, Jesse Dill, Pranay Mathur, Frank Dellaert, Panagiotis Tsiotras et Danfei Xu. « Champ de visibilité neuronale pour la cartographie active basée sur l'incertitude. » *CVPR*, 2024. URL : <https://arxiv.org/abs/2406.06948>

[27] Shuo Feng, Xintao Yan, Haowei Sun, Yiheng Feng et Henry X. Liu. « Test d'intelligence de conduite intelligente pour les véhicules autonomes dans un environnement naturaliste et conflictuel. » *Nature Communications*, 12:748, 2021. DOI : 10.1038/s41467-021-21007-8. URL : <https://www.nature.com/articles/s41467-021-21007-8>[28] Shuo Feng, Yiheng Feng, Chao Yu, Yi Zhang et Henry X. Liu. « Génération d'une bibliothèque de scénarios de tests pour les véhicules connectés et automatisés, partie I : Méthodologie. » *Transactions IEEE sur les systèmes de transport intelligents*, 2021. DOI : 10.1109/TITS.2020.2972211. URL : <https://doi.org/10.1109/TITS.2020.2972211>

[29] Shuo Feng, Yiheng Feng, Haowei Sun, Yi Zhang et Henry X. Liu. "Génération de bibliothèque de scénarios de test pour les véhicules connectés et automatisés, partie II : études de cas." *Transactions IEEE sur les systèmes de transport intelligents*, 2021. DOI : 10.1109/TITS.2020.2988309. URL : <https://doi.org/10.1109/TITS.2020.2988309>[30] Chejian Xu, Wenhao Ding, Baiming Li, Xinqing Chen, Ding Zhao, Bo Li, Jiajun Liu et Hang Zhao. « SafeBench : une plateforme d'analyse comparative pour l'évaluation de la sécurité des véhicules autonomes. » *Ensembles de données et benchmarks NeurIPS*, 2022. URL : <https://proceedings.neurips.cc/paper_files/paper/2022/hash/a48ad12d588c597f4725a8b84af647b5-Abstract-Datasets_and_Benchmarks.html>

[31] Jason Liu, Ankit Shah, Eric Rosen et Stefanie Tellex. « Lang2LTL : traduction des commandes en langage naturel en spécifications de tâches temporelles de robot. » *PMLR/CoRL*, 229, 2023. URL : <https://proceedings.mlr.press/v229/liu23d.html>

[32] Francesco Fuggitti et Tathagata Chakraborti. "NL2LTL : un package Python pour convertir des instructions en langage naturel en formules logiques temporelles linéaires." *Démonstration AAAI*, 37(13):16428-16430, 2023. DOI : 10.1609/aaai.v37i13.27068. URL : <https://ojs.aaai.org/index.php/AAAI/article/view/27068>[33] Behrad Rabiei et Mahesh A. Kumar. « LTLCodeGen : génération de code de logique temporelle syntaxiquement correcte pour la planification des tâches du robot. » arXiv :2503.07902, 2025. URL : <https://arxiv.org/abs/2503.07902>

[34] Jun Wang, David Smith Sundarsingh, Jyotirmoy V. Deshmukh et Yiannis Kantaros. "ConformalNL2LTL : traduction d'instructions en langage naturel en formules logiques temporelles avec des garanties d'exactitude conforme." arXiv :2504.21022, 2025. URL : <https://arxiv.org/abs/2504.21022>

[35] Zhonghang Li, Lianghao Xia, Jiabin Tang, Yong Xu, Lei Shi, Long Xia, Dawei Yin et Chao Huang. "UrbanGPT : grands modèles de langage spatio-temporels." arXiv :2403.00813, 2024. URL : <https://arxiv.org/abs/2403.00813>

[36] Yuan Yuan, Jingtao Ding, Jie Feng, Depeng Jin et Yong Li. « UniST : un modèle universel optimisé pour la prévision spatio-temporelle urbaine. » *KDD*, 2024. DOI : 10.1145/3637528.3671662. URL : <https://dblp.org/rec/conf/kdd/0032D0J024>[37] Jinhui Ouyang, Yijie Zhu, Xiang Yuan et Di Wu. "TrafficGPT : vers une analyse et une génération de trafic à plusieurs échelles avec un cadre d'agents spatio-temporels." arXiv :2405.05985, 2024. URL : <https://arxiv.org/abs/2405.05985>

[38] Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Jens Beisswenger, Ping Luo, Andreas Geiger et Hongyang Li. « DriveLM : Conduire avec des réponses visuelles aux questions sous forme de graphique. » *ECCV*, 2024. URL : <https://github.com/OpenDriveLab/DriveLM>

[39] Maximilian Adang, JunEn Low, Ola Shorinwa et Mac Schwager. « SINGER : Une politique de navigation vision-langage généraliste embarquée pour les drones. » arXiv :2509.18610, 2025. URL : <https://arxiv.org/abs/2509.18610>[40] Hengxing Cai, Jinhan Dong, Jingjun Tan, Jingcheng Deng, Sihang Li, Zhifeng Gao, Haidong Wang, Zicheng Su, Agachai Sumalee et Renxin Zhong. "FlightGPT : vers une navigation vision et langage d'UAV généralisable et interprétable avec des modèles de vision et de langage." *EMNLP*, 2025. DOI : 10.18653/v1/2025.emnlp-main.338. URL : <https://aclanthology.org/2025.emnlp-main.338/>

[41] Pranav Saxena, Nishant Raghuvanshi et Neena Goveas. «UAV-VLN : navigation guidée par langage de vision de bout en bout pour les drones.» arXiv :2504.21432, 2025. URL : <https://arxiv.org/abs/2504.21432>[42] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang et Chelsea Finn. "OpenVLA : un modèle vision-langage-action open source." arXiv :2406.09246, 2024. URL : <https://arxiv.org/abs/2406.09246>

[43] Équipe Octo Model, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, Jianlan Luo, You Liang Tan, Lawrence Yunliang Chen, Pannag Sanketi, Quan Vuong, Ted Xiao, Dorsa Sadigh, Chelsea Finn et Sergey Levine. "Octo : une politique de robot généraliste open source." arXiv :2405.12213, 2024. URL : <https://arxiv.org/abs/2405.12213>[44] Anthony Brohan, Noah Brown, le juge Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn et al. « RT-2 : Les modèles Vision-Langage-Action transfèrent les connaissances Web vers le contrôle robotique. » arXiv :2307.15818, 2023. URL : <https://arxiv.org/abs/2307.15818>

[45] Équipe Qwen. «Rapport technique Qwen3.» arXiv :2505.09388, 2025 ; Dépôt officiel QwenLM/Qwen3. URL : <https://arxiv.org/abs/2505.09388> ; <https://github.com/QwenLM/Qwen3>

[46] DeepSeek-AI. "DeepSeek-R1 : Inciter la capacité de raisonnement dans les LLM via l'apprentissage par renforcement." arXiv :2501.12948, 2025. URL : <https://arxiv.org/abs/2501.12948> ; carte modèle : <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B>

[47] OpenAI. « Protocole de contexte de modèle (MCP) : SDK OpenAI Agents. » Documentation officielle, 2026. URL : <https://openai.github.io/openai-agents-js/guides/mcp/>[48] ​​Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang et Ion Stoica. «Gestion efficace de la mémoire pour la diffusion de modèles de langage étendus avec PagedAttention.» arXiv :2309.06180, 2023. URL : <https://arxiv.org/abs/2309.06180>

[49] Nvidia. "NVIDIA TensorRT-LLM." Documentation officielle, 2026. URL : <https://docs.nvidia.com/tensorrt-llm/index.html>

[50] Jinpeng Hu, Wei Wang, Yuxiao Liu et Jing Zhang. "Grand modèle dans l'économie à basse altitude : applications et défis." *Mégadonnées et informatique cognitive*, 10(1):33, 2026. DOI : 10.3390/bdcc10010033. URL : <https://www.mdpi.com/2504-2289/10/1/33>

[51] Ji Zhao et Xiao Lin. "Agents intelligents aériens à usage général dotés de grands modèles de langage." arXiv :2503.08302, 2025. URL : <https://arxiv.org/abs/2503.08302>