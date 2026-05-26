---
title: "Papier E Livre de tâches expérimentales v2 : Vérification et correction d'erreurs Planification du langage UAV pour AAAI"
description: "La v2 se concentre sur les soumissions aux principales conférences de l'AAAI : complétant plus de 30 conférences régulières réelles et citables/journaux de premier plan/documents pré-imprimés clés, approfondissant les indicateurs expérimentaux, les schémas de comparaison et d'ablation et les protocoles expérimentaux reproductibles de VERA-UAV, et fournissant une preuve mathématique d'exhaustivité relative."
pubDate: 2026-05-17
updatedDate: 2026-05-23
tags: ["Papier E", "AAAI", "drone", "LLM", "LTL", "STL", "Vérification formelle", "Cahier de tâches expérimentales", "Preuve d'exhaustivité"]
category: Tech
sourceHash: "5a168001b03609769dacbd93bde621f2660a2405"
---

# Paper E Experimental Task Book v2 : Vérification et correction d'erreurs Planification du langage UAV pour AAAI

> Ce fichier utilise toujours le nom de fichier `paper-e-vera-uav-experiment-taskbook-v1-20260517.md` car ce tour nécessite une "modification directe sur la version V1". Le texte, le titre et les notes de version ont tous été mis à niveau vers **v2**. Cet article n'est pas une ébauche finale, mais un énoncé de tâche expérimentale exécutable : clarifier le positionnement de recherche de l'article E, les documents réels citables, les solutions algorithmiques, la construction de données, les expériences comparatives, les expériences d'ablation, les indicateurs d'évaluation, les limites théoriques d'exhaustivité et les plans de promotion AAAI/T-ITS ultérieurs. L'accent supplémentaire sur le 19/05/2026 est : la prévention des fuites de données, la taxonomie des échecs, la budgétisation des paramètres, les formules d'indicateurs, la planification des graphiques et les risques de conformité AAAI.

---

## 1. Contexte et objectifs de la recherche

La planification des missions urbaines de drones à basse altitude passe des « itinéraires prédéfinis par l'ingénieur » à des « itinéraires axés sur la mission en langage naturel ». Dans les applications réelles, les opérateurs sont plus susceptibles de donner les instructions suivantes :

- "Vérifiez d'abord la façade est du bâtiment 3, puis rendez-vous au point d'atterrissage sur le toit et attendez."
- "Évitez l'air au-dessus de l'hôpital et atteignez la zone d'accouchement temporaire dans les 30 secondes."
- "Si le couloir sud est occupé, contourner le couloir ouest mais garder une distance de sécurité supérieure à 20 mètres partout."

Ces instructions incluent simultanément la compréhension sémantique, l'ordre temporel, les contraintes spatiales, la sécurité continue de la trajectoire et les jugements d'accessibilité. Les grands modèles linguistiques (LLM) sont efficaces pour comprendre le langage naturel et générer des plans candidats, mais ils ne peuvent pas garantir que le plan de sortie soit exécutable dans l'espace physique, ni garantir que les contraintes de sécurité aérienne soient respectées. Les méthodes formelles sont efficaces pour donner une sémantique vérifiable, telle que la logique temporelle linéaire (LTL) et la logique temporelle du signal (STL), mais les spécifications manuscrites directes nécessitent des connaissances professionnelles et sont difficiles à servir aux opérateurs non experts.

Les travaux existants ont prouvé que la traduction du langage naturel vers LTL peut réduire considérablement le seuil d'écriture des spécifications des tâches du robot. Par exemple, Lang2LTL convertit les commandes de navigation complexes en LTL et effectue une évaluation de généralisation dans des environnements invisibles [1] ; NL2LTL fournit un package Python open source du langage naturel au LTL [2] ; LTLCodeGen utilise la génération de code pour améliorer l'exactitude grammaticale de LTL et l'intègre dans la planification du chemin du robot [3] ; ConformalNL2LTL tente en outre d'utiliser la prédiction conforme pour garantir l'exactitude de la traduction [4]. Ces travaux constituent une base importante pour cette étude.Mais pour les scénarios de drones à basse altitude, il ne suffit pas d’effectuer une simple conversion NL vers LTL. Les missions de drones ont trois exigences supplémentaires :

1. **Contraintes de sécurité continues** : Les contraintes telles que l'altitude de vol, la vitesse, la distance d'obstacle, la fenêtre temporelle, etc. sont naturellement des contraintes sur les signaux continus et sont plus adaptées pour être évaluées par la robustesse STL.
2. **Boucle fermée de trajectoire exécutable** : Des spécifications correctes ne signifient pas que la trajectoire est réalisable et doivent être vérifiées par des cartes, des dynamiques et des planificateurs.
3. **Les erreurs peuvent être corrigées** : les erreurs LLM doivent non seulement être jugées comme des erreurs, mais doivent être converties en contre-exemple ou en retour de robustesse par le vérificateur, puis conduire à la correction LLM.

Par conséquent, cet article propose **VERA-UAV** : un cadre de planification neuro-symbolique de vérification et de correction d'erreurs pour les tâches en langage naturel des drones. La version AAAI donne la priorité à la réponse à une question centrale :

> Étant donné une mission de drone en langage naturel, comment un LLM open source natif peut-il générer des spécifications et des trajectoires de mission LTL/STL vérifiables, réparables et exécutables, plutôt que de simplement générer des plans textuels qui semblent raisonnables mais dont la sécurité n'est pas prouvée ?

La version principale de la conférence AAAI se concentre sur la planification de l'IA, la vérification neuro-symbolique et l'auto-réparation LLM. Le contenu au niveau du système tel qu'AirSim, la véritable logistique à basse altitude et le débit de l'espace aérien multi-UAV sera intégré dans les versions étendues ultérieures du T-ITS.

---

## 2. Définition du problème et hypothèses de base

### 2.1 Entrée et sortie

Étant donné une instance de tâche UAV :

$$
\mathcal{I} = (x_{\text{NL}}, \mathcal{M}, s_0)
$$

Parmi eux, $x_{\text{NL}}$ est l'instruction de tâche en langage naturel, $\mathcal{M}$ est la carte urbaine à basse altitude avec annotation sémantique et $s_0$ est l'état initial du drone. La carte contient les bâtiments, les zones d'exclusion aérienne, l'espace aérien praticable, les points d'atterrissage, les cibles d'inspection, les obstacles dynamiques et les niveaux d'altitude.

Sortie système :

$$
\mathcal{O} = (\text{TaskIR}, \varphi_{\text{LTL}}, \varphi_{\text{STL}}, \tau, r)
$$Où TaskIR est la représentation intermédiaire structurée, $\varphi_{\text{LTL}}$ est la spécification de tâche de synchronisation discrète, $\varphi_{\text{STL}}$ est la contrainte de trajectoire continue, $\tau$ est la trajectoire candidate et $r$ est le résultat de la vérification. Si la tâche ne peut pas être accomplie, le système doit afficher « UNSAT » ou « NEED_CLARIFICATION » au lieu de générer de force une trajectoire dangereuse.

### 2.2 Type de tâche

L’expérience principale AAAI couvre six types de tâches :

| Tapez | Exemple | Principales difficultés |
|------|------|----------|
| Atteindre-éviter | Atteignez A, évitez B | Accessibilité de base et évitement d'obstacles |
| Points de cheminement ordonnés | D’abord vers A, puis vers B | Ordre temporel |
| Patrouille / inspection | Patrouille A, B, C | Couverture multi-cibles |
| Livraison à créneau horaire | Arrivez à A dans les 30 secondes | Contraintes de temps continues |
| Atterrissage d'urgence | Si la route devant vous est inaccessible, rendez-vous au point d'atterrissage le plus proche | Conditions et stratégies alternatives |
| Ambigu / impossible | "Allez dans cet endroit sûr" ou des contraintes mutuellement exclusives | Clarification et détection insatisfaisante |

### 2.3 Hypothèses fondamentales

Cet article ne suppose pas que LLM soit fiable en soi. Au lieu de cela, cet article suppose que les LLM commettent souvent les erreurs suivantes :

- Générez du LTL/STL avec une syntaxe illégale.
- Contraintes de sécurité manquantes en langage naturel.
- Référence à une entité qui n'existe pas dans la carte.
- Donner une séquence de tâches qui satisfait au texte mais qui n'est pas exécutable.
- Violation des contraintes minimales de distance, de hauteur ou de fenêtre temporelle sur les trajectoires continues.

L'hypothèse de base de VERA-UAV est la suivante : ** Si le vérificateur peut convertir ces erreurs en contre-exemples structurés, en retours de noyau non saturé et de robustesse, le taux de réussite de la correction du LLM open source local sera nettement supérieur à celui d'une simple nouvelle tentative d'invite ; en outre, si le système conserve le repli d'énumération de symboles dans le DSL limité, l'algorithme peut obtenir une exhaustivité relative, plutôt que de baser l'exhaustivité sur la fiabilité LLM. **

---

## 3. Travaux connexes et articles citables

### 3.1 Aperçu de la carte de la littératureL'un des problèmes avec la v1 est qu'il y a trop peu de références, et il est facile pour les évaluateurs de penser que "c'est juste une application UAV basée sur Lang2LTL / LTLCodeGen". La v2 étend les travaux connexes en cinq lignes : langage naturel à la logique temporelle, planification LLM et auto-réparation, STL/vérification formelle, agent de blindage et de sécurité, UAV-VLN et applications à basse altitude. Le tableau ci-dessous répertorie **37 documents très pertinents**, chacun étant cité dans cet article.| Numéro | Littérature | Lieu / statut | Relation avec cet article |
|------|------|----------------|------------|
| [1] | Lang2LTL | CoRL 2023/PMLR | Point de départ direct pour la mise à la terre NL-LTL |
| [2] | NL2LTL | Démo AAAI 2023 | référence de modèle/outil |
| [3] | LTLCodeGen | IROS 2025/arXiv | La base de référence directe la plus solide, la génération de code garantie syntaxe |
| [4] | ConformeNL2LTL | arXiv 2025 | Référence sur la crédibilité et le mécanisme de rejet de la traduction |
| [5] | NL2SpaTial | arXiv 2025/2026 | Arbre logique structuré et inspiration de relations spatiales |
| [6] | Planificateur T3 | arXiv 2025 | Auto-apprentissage formel LLM + STL planification de mouvement concours direct |
| [7] | SENTINELLE | arXiv 2025/2026 | Évaluation formelle de la sécurité à plusieurs niveaux |
| [8] | LogiqueGarde | arXiv 2025 | Critique de la logique temporelle et génération de contraintes de sécurité |
| [9] | Pro2Garde | arXiv 2025 | surveillance probabiliste du temps d'exécution |
| [10] | Planification généralisée dans les domaines PDDL avec LLM | AAAI 2024 | La valeur des commentaires du vérificateur/débogage pour la planification |
| [11] | Enquête critique sur la planification du LLM | NeurIPS 2023 | Expliquez que LLM a des capacités de planification directe limitées |
| [12] | LLM+P | arXiv 2023 | Cadre de référence pour LLM + planificateur classique |
| [13] | PlanBanc | NeurIPS 2023Ensembles de données et références | Référence de conception de référence de planification LLM |
| [14] | Réagir | ICLR 2023 | ligne de base de la boucle raisonnement-action |
| [15] | SayCan | CoRL 2022 | Base de référence pour la planification LLM fondée sur les moyens financiers |
| [16] | Code en tant que politiques | ICRA2023 | LLM génère des politiques de programme exécutables |
| [17] | ProgPrompt | ICRA 2023 / Robots autonomes | génération de plans de tâches de robots situés |
| [18] | Mission réactive basée sur la logique temporelle et planification de mouvement | IEEE T-RO 2009 | Robot LTL planification fondation classique |
| [19] | Synthèse pour Robots | Bilan annuel 2018 | Revue de la synthèse formelle et du feedback comportemental des robots |
| [20] | Surveillance des propriétés temporelles des signaux continus | FORMATS/FTRTFT 2004 | Point de départ STL |
| [21] | Robustesse des spécifications de logique temporelle | Informatique théorique 2009 | Fondamentaux de la sémantique de robustesse |
| [22] | Satisfaction robuste par rapport aux signaux à valeur réelle | FORMATS 2010 | Base de calcul de robustesse STL |
| [23] | Synthèse réactive à partir des spécifications STL | CCSS 2015 | STL et couplage contrôle/planification |
| [24] | Diagnostic et réparation des synthés STLest | CCSS 2016 | spécification diagnostic/réparation référence théorique |
| [25] | Spot 2.0 | ATVA 2016 | Outil LTL/oméga-automates |
| [26] | RTAM | STTT 2024 / arXiv 2025 | Moniteur de robustesse STL |
| [27] | PRISME 4.0 | CAV 2011 | outil de vérification de modèles probabilistes |
| [28] | RL sécurisé via blindage | AAAI 2018 | Shield garantit que les classiques sûrs fonctionneront |
| [29] | Blindage probabiliste | AAAI 2025 | Assurance de sécurité et blindage probabilistes |
| [30] | AntenneVLN | ICCV 2023 | Référence de navigation en langage visuel pour les drones |
| [31] | Drone réaliste-VLN | ICLR 2025 | Plateformes, benchmarks et méthodes UAV-VLN plus réalistes |
| [32] | ASMA | RA-L/arXiv 2024 | Référence des contraintes de sécurité CBF dans UAV-VLN |
| [33] | LogistiqueVLN | arXiv 2025 | Scénario d'application de navigation linguistique de livraison à basse altitude |
| [34] | Enquête drone-VLN | arXiv 2026 | Feuille de route et défis de la recherche UAV-VLN |
| [35] | Rapport technique Qwen3 | arXiv 2025 | Base de sélection du modèle open source local |
| [36] | DeepSeek-R1 | arXiv 2025 | Base de sélection de modèles open source inférentiels |
| [37] | vLLM/PagedAttention | SOSP 2023 | Base de mise en œuvre de l'inférence locale multimodèle |### 3.2 Principales lacunes des travaux existants

Lang2LTL, NL2LTL, LTLCodeGen et ConformalNL2LTL démontrent conjointement que NL-to-LTL n'est plus une direction vide [1-4]. Par conséquent, Paper E ne peut pas simplement prétendre « nous traduisons le langage naturel en LTL ». Les véritables points de différence potentiels sont :

1. **Extension de l'exactitude de la traduction à l'exactitude de l'exécution** : LTLCodeGen gère déjà l'exactitude de la syntaxe et la génération de chemin [3], mais l'altitude, la vitesse, la distance d'obstacle et la fenêtre temporelle du drone nécessitent la robustesse STL, pas seulement la validité de la formule LTL.
2. ** Passer de la génération unique à la boucle fermée de vérification et de correction d'erreurs ** : T3 Planner, LogicGuard, SENTINEL et Pro2Guard expliquent que le retour d'information formel est en train de devenir un point chaud pour la sécurité LLM incorporée [6-9]. VERA-UAV doit traiter plus explicitement les contre-exemples, le noyau non saturé et les traces de robustesse comme signaux de réparation.
3. **Extension de l'heuristique LLM à des algorithmes relativement complets** : l'auto-guérison LLM elle-même n'est pas prouvée comme étant complète ; l'exhaustivité doit provenir d'un DSL limité, de vérificateurs décidables et de solutions de repli d'énumération symbolique, et non du modèle « pourrait penser correctement ».
4. **Extension de la navigation terrestre aux drones à basse altitude** : les travaux réalistes d'AerialVLN et de l'ICLR 2025 sur les drones-VLN mettent l'accent sur les différences entre les drones et les VLN terrestres : mouvement tridimensionnel, dynamique continue, sécurité de l'espace aérien et contraintes de ressources [30,31]. C’est exactement la motivation derrière l’utilisation du STL par VERA-UAV.

### 3.3 Contraintes de soumission et d'extension de journal

La description officielle de la piste technique principale AAAI-26 exige que le texte principal contienne jusqu'à 7 pages de contenu technique et oblige l'auteur à remplir une liste de contrôle de reproductibilité [38]. Par conséquent, la version AAAI doit se concentrer sur les méthodes, les expériences de base et la reproductibilité, et ne peut pas étendre trop le contenu de l’ingénierie système.Le champ d'application des T-ITS couvre la détection, les communications, les contrôles, la planification, la conception et la mise en œuvre dans les systèmes de transport modernes, ainsi que les orientations méthodologiques telles que l'intelligence artificielle, et nécessite l'expansion de la revue pour avoir de nouvelles contributions claires par rapport aux articles de conférence [39]. Par conséquent, les éditions ultérieures de la revue ITS devraient ajouter des mesures du système de transport urbain à basse altitude telles que l'utilisation de l'espace aérien, le débit des missions, la coordination multi-UAV, la latence des communications et les gains en matière de sécurité opérationnelle.

---

## 4. Algorithme proposé : VERA-UAV

### 4.1 Processus global

Le nom complet de VERA-UAV est provisoirement déterminé comme suit :

**VERA-UAV : Réparation améliorée par vérification pour la planification linguistique des drones autonomes**

Le processus système est le suivant :

```text
Natural-language UAV instruction
        ↓
Local open-source LLM
        ↓
Typed TaskIR
        ↓
TaskIR-to-LTL/STL compiler
        ↓
Spot / RTAMT / optional PRISM verification
        ↓
Counterexample + unsat core + robustness feedback
        ↓
LLM repair + symbolic enumerative fallback
        ↓
A* / RRT* / MPC-lite trajectory generation
        ↓
Final trajectory verification
        ↓
Executable trajectory or UNSAT / NEED_CLARIFICATION
```

Par rapport à la v1, le changement clé dans la v2 est l'ajout d'un **repli énumératif symbolique** : LLM est toujours le principal générateur de candidats, mais lorsque LLM échoue lors de plusieurs cycles de réparations, le système énumérera les réparations candidates dans un DSL TaskIR limité. Cette conception constitue la base de la preuve ultérieure de « complétude relative ».

### 4.2 TaskIR typé

TaskIR est une interface structurée entre langage naturel et logique formelle. Cela évite que LLM génère directement des chaînes LTL/STL arbitraires, réduisant ainsi les erreurs de syntaxe et les erreurs de mise à la terre d'entité.

Le champ TaskIR est conçu comme suit :| Champ | Signification | Exemple |
|------|------|------|
| `entités` | Objets impliqués dans la directive | `bâtiment_3`, `hospital_zone`, `landing_pad_A` |
| `objectifs` | Objectifs à atteindre | `reach(landing_pad_A)` |
| `éviter` | Zones à éviter | `éviter (zone_hôpital)` |
| `séquence` | Séquence sous-cible | `inspecter(B3_east) -> atterrir(A)` |
| `metric_bounds` | Contraintes continues | `altitude en [20 120]`, `distance_à_obstacle >= 10` |
| `time_windows` | Fenêtre de temps | `atteindre (A) dans les 30 secondes` |
| `replis` | Stratégies alternatives | `si bloqué, atteignez le plus proche_safe_pad` |
| `incertitude` | Champs ambigus ou manquants | `NEED_CLARIFICATION(target="endroit sûr")` |

### 4.3 Compilation TaskIR vers LTL/STL

LTL est utilisé pour exprimer des structures temporelles discrètes :

$$
\varphi_{\text{LTL}} =
G(\neg collision) \wedge F(reach(goal)) \wedge G(\neg enter(no\_fly\_zone))
$$

STL est utilisé pour exprimer des contraintes de signal continues :

$$
\varphi_{\text{STL}} =
G_{[0,T]}(d_{\text{obs}}(t) \ge d_{\min})
\coin
G_{[0,T]}(h_{\min} \le h(t) \le h_{\max})
\coin
F_{[0,30]}(atteindre(objectif))
$$

Où $d_{\text{obs}}(t)$ est la distance entre le drone et l'obstacle le plus proche, et $h(t)$ est l'altitude de vol. Robustesse de sortie du moniteur RTAMT ou STL équivalent :$$
\rho(\tau, \varphi_{\text{STL}}) > 0
$$

Indique que la trajectoire $\tau$ satisfait la spécification ; si $\rho \le 0$, le vérificateur renvoie la clause de violation, le temps de violation et la marge de sécurité minimale.

### 4.4 Réparation du pilote de contre-exemple

Au lieu de simplement renvoyer « réussite/échec », le validateur renvoie un diagnostic structuré :

```json
{
  "status": "FAILED",
  "stage": "STL_ROBUSTNESS",
  "violated_clause": "G[0,T](distance_to_obstacle >= 10)",
  "counterexample_trace": [
    {"t": 14.2, "x": 38, "y": 51, "z": 30, "distance_to_obstacle": 6.4}
  ],
  "robustness": -3.6,
  "repair_hint": "Increase safety margin or route around building_7 west side."
}
```

L'invite de réparation de LLM ne nécessite pas de jeu libre, mais nécessite uniquement la modification des champs pertinents dans TaskIR :

```text
你生成的 TaskIR 在 STL 验证中失败。
失败子句：G[0,T](distance_to_obstacle >= 10)
反例：t=14.2s 时距离 building_7 仅 6.4m。
请只修改 route constraint 或 safety margin，不要改变用户原始目标。
输出新的 TaskIR JSON。
```

L'objectif de cette conception est de réduire l'espace de recherche de LLM et de rendre le comportement de réparation explicable, enregistrable et reproductible.

Si la réparation LLM échoue après des tours $K_{\mathrm{LLM}}$ consécutifs, le repli de l'énumération des symboles est entré. La portée de l'énumération est délimitée par la profondeur DSL de TaskIR, l'ensemble d'entités de carte, le modèle de contrainte autorisé et l'horizon de tâche maximal. L'enquêteur donne la priorité à l'expansion des champs les plus pertinents en fonction des résultats du diagnostic, tels que la distance de sécurité, le côté du détour, la fenêtre temporelle, la séquence cible et l'aire d'atterrissage de repli.

### 4.5 Génération de trajectoire

La version AAAI utilise un générateur de trajectoire léger et reproductible :

- Grille 2D A* : pour les tâches de base à éviter et séquentielles.
- Grille 3D A* : utilisée pour les niveaux d'altitude et les corridors urbains de basse altitude.
- RRT* : pour une vérification supplémentaire spatiale continue.
- MPC-lite/lissage de trajectoire : utilisé pour vérifier si le rayon de braquage, le changement de vitesse et le changement de hauteur satisfont aux contraintes dynamiques simplifiées.

Le générateur de trajectoire n'est pas l'innovation de cet article. Sa fonction est de faire progresser le problème de traduction des spécifications jusqu'au niveau de « si la piste exécutable existe réellement ».

---

## 5. Preuve des propriétés théoriques et de l'exhaustivité relative

La v1 indique uniquement que "la correction des erreurs de vérification peut améliorer la fiabilité", mais il n'y a pas de limite mathématique. La v2 clarifie les propriétés algorithmiques : VERA-UAV ne prétend pas que le LLM lui-même est complet, mais prétend plutôt avoir une **exhaustivité relative** sous les hypothèses d'un DSL fini, d'un vérificateur décidable et d'un planificateur sous-jacent complet.

### 5.1 Cadre formel

Discrétiser la carte urbaine de basse altitude en une carte pondérée limitée :

$$
G=(V,E,w), \quad |V|<\infty, \quad |E|<\infty.
$$Chaque nœud $v\in V$ transporte un ensemble de propositions atomiques $L(v)$, telles que `goal_A`, `building_7_margin`, `no_fly_zone`, `altitude_layer_3`. Les trajectoires sont des séquences finies :

$$
\tau = (v_0, v_1, \ldots, v_T), \quad (v_t,v_{t+1})\in E.
$$

TaskIR DSL est défini comme une syntaxe limitée :

$$
\mathcal{D}_{H,D} = \{\psi : \mathrm{profondeur}(\psi)\le D,\ \mathrm{horizon}(\psi)\le H,\ \mathrm{entities}(\psi)\subseteq \mathcal{E}(\mathcal{M})\}.
$$

Le compilateur $C$ compile TaskIR selon la spécification LTL/STL :

$$
C(\psi)=(\varphi_{\mathrm{LTL}},\varphi_{\mathrm{STL}}).
$$

Le vérificateur $V$ détermine si les trajectoires candidates répondent aux spécifications :

$$
V(\tau, C(\psi)) =
\begin{cas}
\mathrm{PASS}, & \tau \models \varphi_{\mathrm{LTL}}\ \land\ \rho(\tau,\varphi_{\mathrm{STL}})>0,\\
\mathrm{ÉCHEC}(\eta), & \text{sinon},
\fin{cas}
$$

où $\eta$ est un contre-exemple, un noyau non saturé ou une trace de robustesse.

### 5.2 Pseudocode de l'algorithme

```text
Algorithm VERA-UAV
Input: natural language x_NL, map M, initial state s0
Output: verified trajectory tau or UNSAT / NEED_CLARIFICATION

1: Q ← LLM_PROPOSE(x_NL, M)
2: Q ← TYPECHECK_AND_RANK(Q)
3: Visited ← ∅
4: for iter = 1 ... B do
5:     if Q has no unvisited candidate:
6:         Q ← Q ∪ SYMBOLIC_ENUMERATE_NEXT(D, H)
7:         if Q still has no unvisited candidate:
8:             return UNSAT
9:     ψ ← POP_UNVISITED(Q, Visited)
10:    Visited ← Visited ∪ {ψ}
11:    if ψ has missing entity or underspecified field:
12:        η ← type / grounding diagnostic
13:        Q ← Q ∪ REPAIR(ψ, η)
14:        if all remaining candidates require the same external information:
15:            return NEED_CLARIFICATION
16:        continue
17:    (φ_LTL, φ_STL) ← COMPILE(ψ)
18:    if compiler or syntax verifier fails:
19:        η ← compiler diagnostic
20:        Q ← Q ∪ REPAIR(ψ, η)
21:        continue
22:    τ ← COMPLETE_PLANNER(G, s0, φ_LTL, φ_STL)
23:    if τ exists and VERIFY(τ, φ_LTL, φ_STL) = PASS:
24:        return τ
25:    η ← counterexample / unsat core / robustness trace
26:    Q ← Q ∪ LLM_REPAIR(ψ, η)
27:    if LLM repair budget exhausted:
28:        Q ← Q ∪ SYMBOLIC_ENUMERATE(ψ, η, D, H)
29: return UNSAT
```

### 5.3 Théorème 1 : Terminabilité

**Théorème 1 (Terminaison).** Si le TaskIR DSL $\mathcal{D}_{H,D}$ est fini et que l'algorithme définit un budget candidat fini $B$, alors VERA-UAV doit renvoyer une trajectoire vérifiée, `UNSAT` ou `NEED_CLARIFICATION` par étapes finies.**Croquis de preuve.** Chaque fois qu'un candidat TaskIR non visité apparaît dans la file d'attente $Q$, et est utilisé pour éviter une expansion répétée via « Visité ». Le nombre maximum de tours de réparation LLM est limité, l'espace d'énumération des symboles $\mathcal{D}_{H,D}$ est limité et la boucle externe peut être exécutée au plus $B$ fois. L’algorithme ne peut donc pas fonctionner indéfiniment. Chaque branche renvoie ou entre dans la boucle finie suivante. Certification terminée.

### 5.4 Théorème 2 : Sécurité et fiabilité

**Théorème 2 (solidité).** Si VERA-UAV renvoie une trajectoire $\tau$, alors étant donné le modèle de carte, la sémantique du moniteur et la précision de discrétisation de la trajectoire, $\tau$ satisfait la spécification LTL/STL compilée :

$$
\tau \models \varphi_{\mathrm{LTL}}
\quad \text{et} \quad
\rho(\tau,\varphi_{\mathrm{STL}})>0.
$$

**Croquis de preuve.** L'algorithme renvoie la trajectoire uniquement après avoir réussi la vérification finale à la ligne 23. La vérification finale consiste en une vérification de la couche LTL et une vérification de la robustesse STL. Si une vérification échoue, l'algorithme génère simplement un diagnostic et poursuit la réparation sans revenir à la trajectoire. Par conséquent, toutes les trajectoires de retour satisfont aux conditions ci-dessus. Certification terminée.

### 5.5 Théorème 3 : Complétude relative

**Théorème 3 (exhaustivité relative).** Pour les instances de tâches qui ne nécessitent pas de clarification externe, supposons :

1. Il existe un TaskIR $\psi^\star \in \mathcal{D}_{H,D}$ équivalent ou suffisamment fidèle pour l'intention de l'utilisateur ;
2. Le compilateur $C$ peut générer des spécifications LTL/STL sémantiquement préservées pour tous les TaskIR dans $\mathcal{D}_{H,D}$ ;
3. Le planificateur sous-jacent est complet dans la recherche de trajectoires satisfaisant $C(\psi)$ sur le graphe fini $G$ ;
4. L'énumérateur symbolique énumérera tous les candidats en $\mathcal{D}_{H,D}$ dans un temps limité ;
5. Le validateur final est fiable pour la sémantique LTL/STL limitée.S'il existe une trajectoire $\tau^\star$ qui satisfait $C(\psi^\star)$, alors lorsque le budget candidat $B \ge |\mathcal{D}_{H,D}|$ l'est, VERA-UAV renverra finalement une trajectoire $\tau$ qui satisfait à la spécification.

**Croquis de preuve.** Selon l'hypothèse 4, le repli de l'énumération symbolique énumérera à $\psi^\star$. D'après l'hypothèse 2, $C(\psi^\star)$ reste sémantique. Selon l'hypothèse 3, le planificateur sous-jacent trouvera des trajectoires qui satisfont $C(\psi^\star)$. Selon l'hypothèse 5, le validateur final acceptera cette trajectoire. L'algorithme renvoie cette trajectoire selon les lignes 23-24 de l'algorithme. VERA-UAV est donc relativement complet dans le cadre de ces hypothèses limitées du DSL et du modèle. Certification terminée.

### 5.6 Limite d'exhaustivité

Ce théorème ne signifie pas que VERA-UAV est absolument complet pour tout langage naturel et toute dynamique continue dans le monde réel. Il indique simplement : ** Tant que la tâche cible peut être représentée par un DSL TaskIR limité et que l'espace de recherche sous-jacent et la sémantique de vérification couvrent la tâche, VERA-UAV générera inévitablement la bonne réponse sans s'appuyer sur LLM, et pourra également trouver une solution réalisable grâce au repli symbolique. **

C'est également le positionnement théorique clé de cet article par rapport à Lang2LTL, LTLCodeGen et T3 Planner : LLM est un générateur de propositions efficace, pas une source d'exhaustivité.

---

## 6. Sources de données et construction d'ensembles de données

### 6.1 Source de données principale

L'expérience principale de l'AAAI utilise la génération procédurale de données de grille/monde de drones urbains et ne s'appuie pas sur AirSim ou sur des données de vol réelles. Il y a trois raisons à cela :

1. Contrôlable : peut générer systématiquement des tâches telles que des fenêtres de temps accessibles, inaccessibles, ambiguës, conflictuelles et serrées.
2. Reproductible : les cartes, les tâches et les graines aléatoires peuvent être entièrement open source.
3. S'adapter à la durée de l'AAAI : se concentrer sur l'évaluation des méthodes d'IA plutôt que sur l'ingénierie de simulation lourde.

### 6.2 Génération de cartes

Chaque carte contient :- Taille de la grille : « 50x50x3 » à « 100x100x5 ».
- Objets sémantiques : bâtiments, hôpitaux, écoles, stations logistiques, points d'atterrissage, surfaces d'inspection, zones d'exclusion aérienne.
- Structure de l'espace aérien : niveaux, couloirs de vol, zones temporairement fermées.
- Éléments dynamiques : ajout optionnel d'obstacles mobiles ou de zones d'exclusion aérienne temporaires.
- Nommage de style OSM : tels que `hospital_zone_2`, `building_7_east_face`, ne sont utilisés que comme référence de dénomination sémantique et ne sont pas utilisés par l'expérience principale.

### 6.3 Exemples de champs

Chaque échantillon contient :

| Champ | Descriptif |
|------|------|
| `id_instruction` | Numéro d'échantillon |
| `map_id` | Numéro de carte |
| `instruction_langue_naturelle` | Tâches de drone en langage naturel |
| `entité_annotations` | Les entités cartographiques sont alignées sur les entités directives |
| `gold_task_ir` | La référence en matière de génération manuelle ou de règles TaskIR |
| `gold_ltl` | LTL standard d’or |
| `or_stl` | L'étalon-or STL |
| `label_satisfiabilité` | `SAT`, `UNSAT`, `NEED_CLARIFICATION` |
| `trajectoire_référence` | Si SAT, donnez une trajectoire réalisable |
| `type_d'échec` | En cas d'échec, marquez le type d'échec |
| `oracle_cost` | Chemin le plus court ou coût de trajectoire à coût minimum |

### 6.4 Échelle des données

Échelle d'expérimentation principale AAAI recommandée v2 :

| Divisé | Quantité | Objectif |
|------|------|------|
| Train / pool d'invites | 800 | quelques exemples, débogage de modèles |
| Développeur | 250 | rapide, stratégie de réparation, sélection de seuil |
| Test | 400 | Rapport final |
| Test de résistance | 150 | Combinaison longue, floue, insatisfaisante, fenêtre temporelle serrée |

L'ensemble de test ne peut pas participer à la sélection rapide. Tous les rapports de laboratoire contiennent des graines aléatoires et des listes de tâches fixes.

### 6.5 Protocole de génération de données et prévention des fuitesPour que les benchmarks synthétiques résistent aux évaluateurs de l'AAAI, la génération de données doit être gérée dès le premier jour comme des « benchmarks reproductibles » plutôt que des « scripts expérimentaux ad hoc » :

1. **Gelez d'abord le générateur, puis générez le test** : Le générateur de carte, le modèle de tâche, les règles de paraphrase du langage et les règles d'injection d'échecs sont d'abord débogués sur le développement, gèlent le hachage de validation, puis génèrent le test/test de résistance.
2. ** Divisé par niveau de carte ** : La carte de test ne peut pas partager le même `map_id`, les mêmes coordonnées d'entité ou la même disposition des obstacles avec train/dev. Seuls les types de tâches abstraites peuvent être partagés.
3. **Répartir par niveau de dénomination d'entité** : Au moins 30 % des tâches du test utilisent des modèles de dénomination d'entité qui ne sont pas apparus dans l'ensemble de formation, tels que `clinic_zone`, `sky_corridor_E2`, `temporary_pad_17`.
4. **Répartir par niveau de combinaison de modèle** : conservez certaines combinaisons invisibles dans le test, telles que « inspection ordonnée + fenêtre de temps + repli d'urgence », pour empêcher le modèle de se souvenir d'un seul mappage de modèle.
5. **Correction de la graine aléatoire et du manifeste** : chaque division génère `manifest.jsonl`, la version du générateur d'enregistrement, la graine, le hachage de carte, l'identifiant du modèle de tâche, l'identifiant de paraphrase et l'étiquette de satisfiabilité.
6. **Interdire la pollution par les invites de test** : les exemples de quelques tirs ne peuvent provenir que d'un pool de trains/invites ; dev n'est utilisé que pour la sélection de seuil et de stratégie rapide ; les tests et les tests de résistance ne sont exécutés qu'une seule fois et les résultats sont verrouillés.

### 6.6 Taxonomie des échecs

Chaque échantillon de défaillance doit enregistrer la première étape de défaillance et la phase de défaillance finale pour faciliter l'explication de ce que VERA-UAV a corrigé :| Type de panne | Définition | Module d'attribution principal |
|--------------|------|--------------|
| `erreur_syntaxe` | LTL/STL ne peut pas être analysé ou ne correspond pas au type | LLM/compilateur |
| `erreur_entité` | Référence à une entité cartographique inexistante, ambiguë ou incompatible | mise à la terre |
| `sémantique_miss` | Contraintes utilisateur clés manquantes, telles que les zones d'exclusion aérienne ou les fenêtres horaires | Génération TaskIR |
| `unsat_missed` | l'or est UNSAT, mais le système renvoie un plan exécutable | vérificateur / politique de décision |
| `false_unsat` | l'or est SAT, mais l'erreur système génère UNSAT | planificateur / budget de recherche |
| `ltl_violation` | La séquence temporelle discrète, l'arrivée et l'évitement ne sont pas satisfaits | planificateur / compilateur LTL |
| `stl_violation` | hauteur, distance, vitesse, fenêtre temporelle robustesse non positive | trajectoire / moniteur STL |
| `réparation_régression` | Réparer une contrainte puis détruire les contraintes initialement satisfaites | boucle de réparation |
| `délai d'attente` | Dépassement du budget d'inférence ou de planification prédéfini | budget du système |

Non seulement le score moyen est indiqué dans l'article final, mais également un histogramme empilé de la taxonomie des échecs. De cette manière, même si l’amélioration globale n’est pas suffisamment importante, elle peut néanmoins prouver que la méthode a un effet clair sur les types de défaillances critiques en matière de sécurité.

---

## 7. Plateforme expérimentale et configuration de l'implémentation

### 7.1 Matériel

Actuellement conçu avec 4 RTX 4090 et 24 Go de mémoire vidéo chacun. Cette étude ne s'appuie pas sur des API fermées et les principales expériences utilisent toutes des modèles open source locaux.

### 7.2 Modèle

Modèle expérimental principal :

- Qwen3-8B : baseline de modèle local léger [35].
- Qwen3-14B : modèle maître [35].
- DeepSeek-R1-Distill-Qwen-14B : Modèle amélioré d'inférence [36].

Modèles coiffés en option :- Modèle quantitatif 32B, utilisé en annexe ou en résultats supplémentaires ; Cette étude n’est pas requise pour les principales conclusions de l’AAAI.

L'inférence locale utilise les transformateurs vLLM/PagedAttention ou HuggingFace. La conception PagedAttention de vLLM convient aux expériences de débit sous plusieurs invites et plusieurs cycles de réparation [37].

### 7.3 Modules logiciels

| Module | Outil candidat | Fonction |
|------|----------|------|
| Inférence LLM | vLLM/transformateurs | Inférence de modèle local |
| Validation LTL | Spot | Analyse LTL, automates, analyse de satisfiabilité |
| Surveillance STL | RTAMT ou moniteur auto-implémenté | Robustesse STL |
| Vérification probabiliste | PRISME | Vérification facultative en environnement incertain |
| Planification | A* / RRT* / MPC-lite | Génération de trajectoire |
| Journalisation | JSONL + CSV | Enregistrement de chaque cycle de construction, de vérification et de réparation |

### 7.4 Dossier d'opération

Chaque instance de tâche doit enregistrer :

- notice originale.
- TaskIR, LTL, STL par tour.
- Sortie du validateur.
- Correction de l'invite.
- Trajectoire finale.
- Durée d'exécution, nombre de jetons et configuration de la mémoire graphique.
- ID de comparaison apparié de la ligne de base et du VERA-UAV sur la même tâche.

Ces enregistrements servent à la liste de contrôle de reproductibilité AAAI [38].

### 7.5 Budget des paramètres de pré-inscription

Afin d'éviter d'ajuster les paramètres après l'expérimentation, ce document de tâche recommande de fixer le budget suivant avant la première série de tests formels :| Paramètres | Valeurs recommandées | Descriptif |
|------|--------|------|
| `K_LLM` | 3 | Jusqu'à trois cycles de réparation LLM par tâche |
| 'B' | 256 | Budget total TaskIR du candidat VERA-UAV |
| 'D' | 4 | Profondeur d'imbrication maximale TaskIR DSL |
| 'H' | 8 | Horizon de tâche discret / limite supérieure du sous-objectif |
| `T_plan` | années 30 | Délai d'expiration de la planification d'une tâche unique |
| `T_llm` | 20 ans | Délai d'inférence LLM à un seul tour |
| température de décodage | 0,2 | Faible caractère aléatoire dans l'expérience principale ; signaler uniquement la sensibilité à la température en annexe |
| haut-p | 0,9 | Co-fixé à la température |
| maximum de nouveaux jetons | 1024 | Empêcher que la différence entre les longueurs de sortie des différents modèles n'affecte le temps d'exécution |

Si ces valeurs doivent être modifiées pour des expériences formelles, les raisons doivent d'abord être enregistrées sur le développement, puis la configuration doit être recongelée. Les résultats des tests ne peuvent pas déterminer les paramètres inversement.

---

## 8. Conception expérimentale comparative

### 8.1 Liste de référence| Méthode | Descriptif | Objectif |
|------|------|------|
| Planification directe du LLM | Waypoint de sortie directe LLM / séquence d'action | Vérifiez si la planification en texte brut est dangereuse |
| Planification de style ReAct | boucle raisonnement-action, pas de vérification formelle | par rapport à la planification générale des agents LLM [14] |
| Filtrage des moyens de type SayCan | Score LLM + filtre de compétences réalisables | Comparez la mise à la terre des moyens [15] |
| Invite uniquement NL vers LTL/STL | LLM génère directement LTL/STL, sans IR tapé ni correctifs de vérification | Vérifier l'invite Limite supérieure du projet |
| Base de référence du modèle de style NL2LTL | Générer du LTL basé sur la correspondance de modèles | Par rapport à la méthode de modèle traditionnelle [2] |
| Base de référence de style LTLCodeGen | LLM génère du code de fonction logique puis le compile en LTL | Vérifiez l'exactitude de la syntaxe [3] |
| Autocorrection de style T3 | Vérificateur LLM + STL, plusieurs cycles d'autocorrection | Par rapport à la récente voie de concurrence directe [6] |
| VERA-UAV sans réparation | Utilisez TaskIR et vérifiez, mais ne réparez pas après un échec | Contributions séparées pour la vérification et la réparation |
| Réparation VERA-UAV LLM uniquement | Réparation IR + LLM typée, repli d'énumération non signé | Vérifier la contribution du repli à l'exhaustivité |
| VERA-UAV complet | IR entièrement typé + vérification + réparation de contre-exemple + repli symbolique | méthode principale |

### 8.2 Expérience principale

L'expérience principale répond à cinq questions :1. VERA-UAV est-il plus facile à générer des plans exécutables que la ligne de base ?
2. VERA-UAV réduit-il les taux de failles de sécurité ?
3. La robustesse STL du VERA-UAV est-elle significativement plus élevée ?
4. Le nombre de cycles de réparation et les frais généraux d'inférence supplémentaires du VERA-UAV sont-ils acceptables ?
5. Le repli de l'énumération symbolique améliore-t-il réellement le « taux de récupération des tâches ayant échoué » et l'exhaustivité relative ?

Principales suggestions de tableaux de résultats :| Méthode | Syntaxe valide ↑ | Sémantique F1 ↑ | ESS ↑ | FRS ↓ | Robustesse moyenne ↑ | Écart d'optimalité ↓ | Réussite de la réparation ↑ | Durée d'exécution ↓ |
|--------|----------------|---------------|-------|-------|-------------------|------------------|------------------|---------------|
| LLM direct | À déterminer | À déterminer | À déterminer | À déterminer | À déterminer | À déterminer | N/A | À déterminer |
| Style ReAct | À déterminer | À déterminer | À déterminer | À déterminer | À déterminer | À déterminer | N/A | À déterminer |
| Style SayCan | À déterminer | À déterminer | À déterminer | À déterminer | À déterminer | À déterminer | N/A | À déterminer |
| Invite uniquement | À déterminer | À déterminer | À déterminer | À déterminer | À déterminer | À déterminer | N/A | À déterminer |
| Style NL2LTL | À déterminer | À déterminer | À déterminer | À déterminer | À déterminer | À déterminer | N/A | À déterminer |
| Style LTLCodeGen | À déterminer | À déterminer | À déterminer | À déterminer | À déterminer | À déterminer | N/A | À déterminer |
| Style T3 | À déterminer | À déterminer | À déterminer | À déterminer | À déterminer | À déterminer | À déterminer | À déterminer |
| VERA-UAV pas de réparation | À déterminer | À déterminer | À déterminer | À déterminer | À déterminer | À déterminer | 0 | À déterminer |
| Réparation VERA-UAV LLM uniquement | À déterminer | À déterminer | À déterminer | À déterminer | À déterminer | À déterminer | À déterminer | À déterminer |
| VERA-UAV complet | À déterminer | À déterminer | À déterminer | À déterminer | À déterminer | À déterminer | À déterminer | À déterminer |« À déterminer » dans le tableau correspondent aux données à renseigner pour l'expérience et ne doivent pas être falsifiées dans la lettre de mission.

### 8.3 Protocole d'évaluation des résultats expérimentaux

La v2 clarifie les principaux indicateurs et jugements statistiques pour éviter le risque ultérieur de « déclarer le bon indicateur ».

**Mesure principale 1 : Succès de sécurité exécutable (ESS)**

Une tâche est comptée comme ESS=1 uniquement si elle remplit simultanément les conditions suivantes :

- Le TaskIR généré n'a aucune erreur de type.
-Compilable LTL/STL.
- Le planificateur trouve des trajectoires.
- La trajectoire finale réussit le contrôle LTL.
- La robustesse STL est positive.
- Pas de collisions, d'entrées dans des zones d'exclusion aérienne, de violations d'altitude ou de défaillances de fenêtres horaires.

**Mesure principale 2 : taux de fausse sécurité (FSR)**

FSR mesure la proportion de tâches dangereuses ou insatisfaisantes que le système juge à tort comme exécutables en toute sécurité :

$$
\mathrm{FSR} = \frac{\#\{\mathrm{incertain\ mais\ renvoyé\ comme\ exécutable}\}}{\#\{\mathrm{tous\ renvoyé\ exécutable}\}}.
$$

Dans le document de l’AAAI, le FSR doit être considéré comme l’indicateur négatif le plus critique en matière de sécurité. Le principal argument de vente du VERA-UAV n'est pas d'avoir un « rendement » pour toutes les tâches, mais d'éviter une fausse sécurité.

**Test statistique**

- Pour les indicateurs binaires tels que la détection ESS, FSR et UNSAT, utilisez le test McNemar apparié.
- Pour les indicateurs continus tels que la robustesse, l'écart d'optimalité, le temps d'exécution, etc., utilisez le bootstrap apparié IC à 95 % et le test de rang signé de Wilcoxon.
- Plusieurs comparaisons de base utilisent la correction Holm-Bonferroni.
- Les conclusions ne sont écrites dans le texte principal que lorsque $p<0,05$ et que la taille de l'effet atteint le seuil de pré-enregistrement.

**Critères de réussite**

Les conditions minimales pour l’établissement de la conclusion principale de l’AAAI :1. L'ESS du VERA-UAV complet est nettement supérieur à celui de la ligne de base de style LTLCodeGen et de style T3.
2. Le FSR du VERA-UAV complet est nettement inférieur à celui de toutes les références LLM uniquement.
3. Après suppression du feedback de robustesse STL, les défaillances liées aux contraintes de sécurité continues augmentent considérablement.
4. Le repli symbolique fournit des gains mesurables dans les échantillons d'échec de réparation LLM.

### 8.4 Expérience de généralisation

Dimension de généralisation :

- Aucune carte vue.
- Aucun nom d'entité vu.
- Paraphrase en langage naturel.
- Combinaisons de timing plus longues.
- Fenêtre temporelle plus serrée.
- Augmentation du taux de tâches insatisfaites.

Les expériences de généralisation se concentrent sur la question de savoir si VERA-UAV peut identifier des tâches insatisfaisantes ou ambiguës, plutôt que sur la production de trajectoires d'erreur.

### 8.5 Étude de cas

Préparez au moins trois cas de visualisation :

1. **Cas de réparation de syntaxe** : la sortie LLM est un STL illégal, Spot/RTAMT signale une erreur, réparation du système.
2. **Cas de sécurité de la trajectoire** : LTL est satisfait mais la robustesse STL est négative et le système devient positif après un détour.
3. **Cas insatisfaisant** : les exigences de l'utilisateur sont contradictoires et le système affiche « UNSAT ».

### 8.6 Plan de graphique de texte principal AAAI

L'espace de texte principal de l'AAAI est très restreint et les graphiques doivent servir l'argument principal. Il est recommandé que seuls cinq types de graphiques soient inclus dans le texte principal, et des annexes sont utilisées pour les autres :| Diagramme | Cible | Placement |
|------|------|----------|
| Figure 1 : Pipeline VERA-UAV | Un coup d'œil sur la boucle fermée des IR tapés, de la vérification, de la réparation et du repli | Méthode |
| Tableau 1 : Matrice de positionnement de la littérature de base | Prouve que cet article n'est pas une simple application NL-to-LTL | Travaux connexes |
| Tableau 2 : Principaux résultats de l'expérience | comparaison appariée de ESS, FSR, robustesse, runtime | Expériences |
| Figure 2 : graphique empilé de taxonomie des échecs | illustre les types de défaillances que la méthode réduit principalement | Expériences |
| Figure 3 : Parcours de l'étude de cas | Montre comment les commentaires contre-exemples peuvent corriger la robustesse négative en positive | Expériences / Annexe |

Il n'est pas recommandé d'agrandir la section d'invite, la grammaire DSL complète ou toutes les captures d'écran de la carte dans l'article principal. Ces contenus doivent être placés dans l'annexe code/données afin de ne pas évincer l'argument de contribution.

---

## 9. Conception d'expériences d'ablation| Ablation | Variante | Objectif |
|--------|------|------|
| Supprimer l'IR tapé | Génération directe LTL/STL | Vérifier si la représentation intermédiaire structurée améliore la fiabilité |
| Supprimer les commentaires contre-exemples | Nouvelle tentative générique | Vérifier si le contre-exemple est plus efficace que la nouvelle tentative normale |
| Supprimer les commentaires sur la robustesse STL | Vérification LTL uniquement | L'importance de vérifier les contraintes de sécurité en continu |
| réparation en un seul coup | Réparer au maximum 1 fois | Évaluer les avantages des tournées de réparation |
| réparation itérative | Réparer jusqu'à 3 fois | Évaluer la limite supérieure de plusieurs cycles de réparation |
| Différentes tailles de modèles | Qwen3-8B / Qwen3-14B / DeepSeek-R1-Distill-Qwen-14B | Évaluer la relation entre la capacité du modèle et le cadre de vérification |
| Supprimer la détection UNSAT | Forcer la génération de traces | Vérifier la contribution de la capacité de refus de réponse à la sécurité |
| Supprimer le symbole de secours | Réparation LLM uniquement | Vérifier la contribution des composants d'exhaustivité relative à la reprise après incident |
| Supprimer la vérification finale du planificateur | Vérifiez uniquement les formules mais pas les trajectoires | Prouver que l'exécution d'une boucle fermée n'est pas facultative |

Le cœur de l'expérience d'ablation n'est pas de « prouver que les composants sont efficaces », mais de découvrir quels composants contribuent le plus aux indicateurs de sécurité et de performance qui préoccupent le plus les évaluateurs de l'AAAI.

---

## 10. Indicateurs d'évaluation

### 10.1 Indicateurs de génération de spécifications| Indicateurs | Définition |
|------|------|
| Validité de la syntaxe | LTL/STL est-il acceptable pour l'analyseur |
| Précision de la mise à la terre de l'entité | Si l'entité de commande est correctement mappée à l'entité de carte |
| Sémantique F1 | Générer précision/rappel/F1 du champ TaskIR et or TaskIR |
| Correspondance sémantique | Si la spécification générée est équivalente ou approximativement équivalente à la formule Gold TaskIR / Gold |
| Précision de détection UNSAT | Si la tâche insatisfaisante est correctement identifiée |
| Précision des clarifications | Si la tâche floue déclenche `NEED_CLARIFICATION` |
| Taux de faux exécutables | La proportion de tâches insatisfaisantes ou ambiguës qui sont mal exécutées |

### 10.2 Indicateurs d'exécution de la planification

| Indicateurs | Définition |
|------|------|
| ESS | Proportion de tâches qui satisfont simultanément à la sémantique, aux trajectoires réalisables, au LTL, au STL et aux contraintes de sécurité |
| FRS | Proportion de tâches dangereuses marquées à tort comme étant sûres à exécuter |
|Robustesse moyenne STL |La robustesse moyenne de la trajectoire finale par rapport à la spécification STL |
| Robustesse STL dans le pire des cas | Répartition de la robustesse minimale par trajectoire |
| Marge minimale de sécurité | Distance minimale d'obstacle dans la trajectoire |
| Écart d'optimalité | $(J(\tau)-J^\star)/J^\star$ |
| Longueur du trajet / temps de vol | Coût de la trajectoire et temps de vol |

### 10.3 Indicateur d'efficacité des réparations| Indicateurs | Définition |
|------|------|
| Taux de réussite des réparations | Taux de réussite des réparations après échec de la vérification |
| Conversion échouée | La proportion d'échantillons initiaux défectueux qui réussissent après avoir été réparés |
| Tours de réparation moyens | Tours de réparation moyens |
| Contribution de secours | Proportion d'échec de réparation LLM mais succès de repli symbolique |
| Frais généraux d'exécution | Temps supplémentaire causé par le mécanisme de réparation |
| Frais généraux de jeton | Correction de l'incrément de jeton provoqué par l'invite et le diagnostic |

### 10.4 Détails du calcul de l'indicateur

L'expérience principale doit implémenter les indicateurs suivants directement dans le code pour éviter toute disposition manuelle pendant la phase de rédaction du papier :

**F1 sémantique**

Aplatissez TaskIR en un ensemble de contraintes au niveau du champ $\mathcal{C}$, telles que `reach(A)`, `avoid(zone_B)`, `time_window(A,30)`. Supposons que l'ensemble de prédictions soit $\hat{\mathcal{C}}$ et que l'ensemble de référence soit $\mathcal{C}^\star$ :

$$
P = \frac{|\hat{\mathcal{C}}\cap \mathcal{C}^\star|}{|\hat{\mathcal{C}}|}, \quad
R = \frac{|\hat{\mathcal{C}}\cap \mathcal{C}^\star|}{|\mathcal{C}^\star|}, \quad
F1 = \frac{2PR}{P+R}.
$$

**Taux d'infractions à la sécurité**

$$
\mathrm{SVR} =
\frac{\#\{\tau : collision \lor nofly \lor altitude\_violation \lor \rho(\tau,\varphi_{\mathrm{STL}})\le 0\}}
{\#\{\mathrm{trajectoires retournées}\}}.
$$

**Écart d'optimalité**Quand Gold ou Oracle Planner peuvent donner le coût optimal $J^\star$ :

$$
\mathrm{Gap}(\tau)=\frac{J(\tau)-J^\star}{\max(J^\star,\epsilon)}.
$$

Si la tâche est UNSAT ou NEED_CLARIFICATION, l'écart d'optimalité n'est pas calculé et est compté séparément dans la précision de la reconnaissance.

**Efficacité de la réparation**

$$
\mathrm{FailToPass} =
\frac{\#\{\mathrm{initial\ échec,\ final\ réussite}\}}
{\#\{\mathrm{initial\ fail}\}},
\quad
\mathrm{Contribution de repli} =
\frac{\#\{\mathrm{LLM\ réparation\ échec,\ symbolique\ secours\ réussite}\}}
{\#\{\mathrm{final\ pass}\}}.
$$

Ces formules doivent être générées sous forme de champs CSV lisibles par machine dans le script d'expérience et formatées uniquement dans le tableau papier.

---

## 11. Conclusions expérimentales attendues

Cette section concerne les attentes préalables à l'inscription et non les résultats expérimentaux.

### 11.1 Principales attentes

Le VERA-UAV complet devrait être supérieur à toutes les lignes de base sur l'ESS et inférieur au taux de violation du FSR/sécurité. La raison en est que la ligne de base optimise généralement uniquement l'exactitude locale du langage par rapport à la spécification, tandis que VERA-UAV intègre « si la spécification peut produire une trajectoire sûre » dans la boucle fermée.

### 11.2 Attentes en matière de feedback contre-exemple

Les contre-exemples devraient réduire considérablement la proportion de plans non exécutables. Par rapport aux nouvelles tentatives génériques, les contre-exemples structurés peuvent indiquer à LLM quelle clause, à quel moment et quelle entité a causé l'échec, réduisant ainsi les tentatives non dirigées.

### 11.3 Attentes IR typées

L'IR typé devrait améliorer la cohérence sémantique et l'interprétabilité. La génération directe de LTL/STL est sujette à l'absence de parenthèses, d'opérateurs, de références d'entité et de contraintes ; TaskIR expose ces erreurs sous forme de champs manquants ou d'erreurs de saisie à l'avance.

### 11.4 Robustesse STL attendueLe retour d’information sur la robustesse STL devrait être le plus critique pour les contraintes de sécurité continues. La couche LTL peut prouver des propriétés discrètes telles que « l'arrivée finale » et « éviter la zone d'exclusion aérienne », mais ne peut pas exprimer pleinement l'altitude de vol, la distance minimale et la marge de la fenêtre temporelle. La robustesse du STL peut fournir des limites de sécurité quantifiées et constitue le point clé qui distingue les drones des tâches de navigation au sol ordinaires.

### 11.5 Taille attendue du modèle

Des modèles locaux plus solides devraient améliorer la qualité initiale de TaskIR, mais le cadre de réparation de validation est également utile pour les modèles plus petits. En d'autres termes, cet article ne devrait pas écrire la contribution comme « un certain grand modèle est plus fort », mais devrait l'écrire comme « le mécanisme de correction des erreurs de vérification améliore la fiabilité des différents modèles open source ».

---

## 12. Problèmes découverts lors de l'auto-audit et des correctifs v2

### Principaux problèmes avec la version 12.1 v1

1. **Couverture bibliographique insuffisante** : la v1 ne répertorie que 12 références, ce qui n'est pas suffisant pour soutenir le positionnement de l'AAAI.
2. **La limite de nouveauté n'est pas assez nette** : la v1 est facilement comprise comme « version UAV NL-to-LTL », et la différence entre Lang2LTL et LTLCodeGen n'est pas assez forte.
3. **Les indicateurs expérimentaux ne sont pas suffisamment jugés** : la v1 ne répertorie que les indicateurs généraux et ne définit pas l'ESS, le FSR, les tests statistiques et les critères de réussite.
4. **La déclaration d'exhaustivité est trop faible** : la v1 n'explique pas pourquoi l'algorithme n'est pas purement heuristique.
5. **Le risque lié aux données synthétiques n'est pas suffisamment atténué** : la version v1 n'explique pas pourquoi les données synthétiques soutiennent toujours les conclusions méthodologiques de l'AAAI.

### Stratégie de réparation pour 12.2 v2

1. Développez jusqu'à plus de 30 documents très pertinents et utilisez une matrice de littérature pour clarifier la relation entre chaque article et cet article.
2. Réduire la contribution de « traduction » à « exécution en boucle fermée + robustesse STL + réparation de contre-exemple + repli relativement complet ».
3. Définir des indicateurs reproductibles tels que ESS, FSR, écart d'optimalité, conversion fail-to-pass, etc.
4. Donnez le théorème de terminaison, de sécurité, de fiabilité et d'exhaustivité relative, et précisez que l'exhaustivité vient d'un DSL fini et d'une énumération symbolique, et non d'un LLM.
5. Mettez AirSim/real logistique dans l'extension T-ITS, et l'article principal de l'AAAI adhère au positionnement méthodologique du benchmark synthétique contrôlé.

### 12.3 2026-05-19 Deuxième auto-examen et renforcementAprès avoir poursuivi l'examen dans ce cycle, on pense que l'épreuve E comporte encore quatre questions faciles à poser pour les évaluateurs, et les contraintes correspondantes ont été ajoutées au cahier de tâches :

1. **Crédibilité des données** : Il ne suffit pas de dire « données générées par le programme ». Il est nécessaire de clarifier le gel du générateur, la segmentation au niveau de la carte, la segmentation au niveau de la dénomination des entités et de tester rapidement la prévention de la pollution.
2. **Pouvoir explicatif des échecs** : La seule déclaration ESS/FSR n'est pas suffisante. La taxonomie des échecs doit être enregistrée pour prouver que la méthode réduit les échecs liés à la sécurité plutôt que de simplement améliorer le score moyen.
3. **Paramètres reproductibles** : La simple utilisation de Qwen3 / DeepSeek ne suffit pas. Vous devez fixer le nombre de cycles de réparation, le budget du candidat, la profondeur DSL, le délai d'attente de planification et les paramètres de décodage.
4. **Stratégie de présentation papier** : AAAI a un espace limité, vous devez donc déterminer le diagramme de texte principal à l'avance, sinon il sera facile de disperser la ligne principale.

Ces quatre points ne changent pas la contribution fondamentale de VERA-UAV, mais ils peuvent faire passer l'énoncé de mission d'une « route d'idées » à un état où « les expériences et les articles peuvent être directement organisés ».

### 12.4 2026-05-23 Finition : conclusion de la ligne principale AAAI

L'article E doit être priorisé en tant que **document de méthode AAAI / IJCAI**, plutôt que de rédiger à l'avance un document complet sur le système ITS. Le problème principal est : comment le plan de mission du drone généré par LLM peut être transformé en un plan de trajectoire exécutable, vérifiable et interprétable via une IR typée, une vérification de la logique temporelle, une réparation de contre-exemple et un repli symbolique.

La première version de l’article ne retient que trois contributions :

1. **TaskIR typé** : convertissez les instructions UAV en langage naturel en représentations intermédiaires qui peuvent être inspectées pour les entités, les actions, les contraintes de temps, les contraintes de sécurité et les contraintes de ressources.
2. **LTL/STL + vérificateur + fermeture de trajectoire** : vérifie non seulement la syntaxe de la formule, mais vérifie également si la spécification peut générer une trajectoire qui satisfait aux contraintes de sécurité.
3. **Réparation de contre-exemple/de robustesse avec un repli DSL fini** : utilisez un contre-exemple, un noyau non saturé et un retour de robustesse STL pour réparer ; lorsque LLM ne peut pas être réparé, utilisez une énumération DSL finie pour donner une exhaustivité relative.

Ne promettez pas ce qui suit à l’avance dans l’article principal :- Ne gère pas complètement le trafic multi-UAV ;
- Pas de véritable déploiement de système logistique ;
- Ne comptez pas sur la simulation haute fidélité AirSim comme expérience principale ;
- N'écrivez pas les politiques STI ou les révélations du système économique à basse altitude comme contributions principales de l'AAAI.

Il est recommandé de figer la matrice expérimentale minimale comme :

| Dimensions | Paramètres de la première édition |
|------|------------|
| Famille de tâches | patrouille, livraison, inspection, évitement, ordonnancement temporel, UNSAT / ambigu |
| Carte | Réseau urbain généré de manière procédurale / obstacle / zone d'exclusion aérienne / point de recharge |
| lignes de base | Planification LLM directe, ReAct / invite uniquement, style NL2LTL, style LTLCodeGen, VERA-UAV sans réparation, VERA-UAV complet |
| Principaux indicateurs | ESS, FSR, taux de violations de sécurité, réussite des réparations, conversion échouée, durée d'exécution |
| Ablation | pas d'IR typé, pas de contre-exemple, pas de robustesse STL, réparation ponctuelle ou itérative, pas de repli symbolique |
| Généralisation | carte invisible, dénomination d'entité invisible, horizon plus long, contraintes plus strictes, détection UNSAT |

Les extensions T-ITS peuvent être placées dans les versions ultérieures : intégrant la planification de la flotte de Paper B, les scénarios de stress de Paper F et les indicateurs du système de trafic à basse altitude. Mais la version AAAI doit garder les questions claires, sinon elle sera repoussée à la fois par les évaluateurs de l'IA et par les évaluateurs du trafic.

---

## 13. Risques et alternatives| Risque | Impact | Alternatives |
|------|------|----------|
| La nouveauté est prise en compte uniquement pour les applications NL vers LTL | AAAI présente un risque élevé de rejet | Accent sur la robustesse STL, la réparation de contre-exemples et la fermeture de trajectoire exécutable |
| La ligne de base LTLCodeGen est trop forte | Le résultat principal présente des avantages insuffisants | Utiliser les contraintes continues des drones et la détection insatisfaisante comme indicateurs de différenciation |
| Capacités de modèle local insuffisantes | Faible qualité initiale de la traduction | Utilisez Qwen3-14B/DeepSeek-R1-Distill-Qwen-14B et signalez les gains de réparation |
| L'ensemble de données est considéré comme trop synthétique | La crédibilité de la candidature est insuffisante | Ajoutez un nom de style OSM, de véritables statistiques de disposition des pâtés de maisons, mais ne vous appuyez pas sur des vols réels |
| Le nombre de cycles de réparation rend le temps d'exécution trop élevé | Les performances en temps réel sont remises en question | Signalez une réparation unique et jusqu'à trois cycles de réparation, définissez un délai d'attente et une solution de secours |
| Le moniteur STL est complexe à mettre en œuvre | Affecte les progrès | Implémentez d'abord le sous-ensemble STL à temps discret, puis connectez-vous à RTAMT |
| AAAI manque d'espace | L'histoire est divergente | Le texte principal ne contient que des méthodes et des expériences de base, et ITS prévoit d'élargir l'annexe |
| AAAI est sensible aux politiques de texte générées par LLM | Risques de conformité dans la rédaction papier | Le texte final soumis doit être réécrit manuellement et révisé par l'auteur. Le résultat du LLM n'est utilisé que comme sujet expérimental ou comme aide à l'écriture interne, et le texte généré non révisé n'est pas directement utilisé comme texte de l'article [38] |
| L'exhaustivité relative est considérée comme une hypothèse trop forte | L'apport théorique est affaibli | Dans le texte principal, il est clairement écrit comme étant une exhaustivité relative, et un DSL limité, un horizon délimité et un planificateur complet sont utilisés comme hypothèses de théorème au lieu de garanties absolues dans le monde réel |
| Le test de résistance est trop difficile, ce qui entraîne une baisse des principaux résultats | L'indicateur moyen n'est pas beau | Le test principal et le test de résistance sont rapportés séparément. Le test de résistance est utilisé pour analyser la frontière robuste et n'est pas mélangé avec la conclusion principale dans la même valeur moyenne |

---

## 14. Références[1] Jason Xinyu Liu, Ziyi Yang, Ifrah Idrees, Sam Liang, Benjamin Schornstein, Stefanie Tellex et Ankit Shah. "Mise à la terre de commandes complexes en langage naturel pour des tâches temporelles dans des environnements invisibles." *Actes de la 7e conférence sur l'apprentissage des robots*, PMLR 229 : 1084-1110, 2023. URL : <https://proceedings.mlr.press/v229/liu23d.html>

[2] Francesco Fuggitti et Tathagata Chakraborti. "NL2LTL - un package Python pour convertir des instructions en langage naturel (NL) en formules de logique temporelle linéaire (LTL)." *Actes de la conférence AAAI sur l'intelligence artificielle*, 37(13):16428-16430, 2023. DOI : 10.1609/aaai.v37i13.27068. URL : <https://ojs.aaai.org/index.php/AAAI/article/view/27068>[3] Behrad Rabiei, Mahesh Kumar AR, Zhirui Dai, Surya LSR Pilla, Qiyue Dong et Nikolay Atanasov. « LTLCodeGen : génération de code de logique temporelle syntaxiquement correcte pour la planification des tâches du robot. » arXiv :2503.07902, 2025 ; la page du projet rapporte IROS 2025. URL : <https://arxiv.org/abs/2503.07902> ; <https://existentialrobotics.org/LTLCodeGen/>

[4] Jun Wang, David Smith Sundarsingh, Jyotirmoy V. Deshmukh et Yiannis Kantaros. "ConformalNL2LTL : traduction d'instructions en langage naturel en formules logiques temporelles avec des garanties d'exactitude conforme." arXiv :2504.21022, 2025. URL : <https://arxiv.org/abs/2504.21022>

[5] Licheng Luo, Kaier Liang, Yu Xia et Mingyu Cai. "NL2SpaTiaL : Génération de spécifications logiques spatio-temporelles géométriques à partir du langage naturel pour les tâches de manipulation." arXiv :2512.13670, 2025 ; révisé 2026. URL : <https://arxiv.org/abs/2512.13670>[6] Jia Li et Guoxiang Zhao. "T3 Planner : un cadre LLM auto-correctif pour la planification de mouvements robotiques avec logique temporelle." arXiv :2510.16767, 2025. URL : <https://arxiv.org/abs/2510.16767>

[7] Simon Sinong Zhan, Yao Liu, Philip Wang, Zinan Wang, Qineng Wang, Zhian Ruan, Xiangyu Shi, Xinyu Cao, Frank Yang, Kangrui Wang, Huajie Shao, Manling Li et Qi Zhu. "SENTINEL : un cadre formel à plusieurs niveaux pour l'évaluation de la sécurité des agents incorporés basés sur LLM." arXiv :2510.12985, 2025. URL : <https://arxiv.org/abs/2510.12985>

[8] Anand Gokhale, Vaibhav Srivastava et Francesco Bullo. « LogicGuard : Amélioration des agents LLM incorporés grâce à des critiques basées sur la logique temporelle. » arXiv :2507.03293, 2025. URL : <https://arxiv.org/abs/2507.03293>

[9] Haoyu Wang, Christopher M. Poskitt, Jun Sun et Jiali Wei. « Pro2Guard : application proactive du temps d'exécution de la sécurité des agents LLM via la vérification de modèle probabiliste. » arXiv :2508.00500, 2025. URL : <https://arxiv.org/abs/2508.00500>[10] Tom Silver, Soham Dan, Kavitha Srinivas, Joshua B. Tenenbaum, Leslie Kaelbling et Michael Katz. "Planification généralisée dans les domaines PDDL avec de grands modèles de langage pré-entraînés." *Actes de la conférence AAAI sur l'intelligence artificielle*, 38(18):20256-20264, 2024. DOI : 10.1609/aaai.v38i18.30006. URL : <https://ojs.aaai.org/index.php/AAAI/article/view/30006>

[11] Karthik Valmeekam, Matthew Marquez, Sarath Sreedharan et Subbarao Kambhampati. « Sur les capacités de planification des grands modèles de langage : une enquête critique. » *Avances dans les systèmes de traitement de l'information neuronale*, 2023. URL : <https://arxiv.org/abs/2305.15771>

[12] Bo Liu, Yuqian Jiang, Xiaohan Zhang, Qiang Liu, Shiqi Zhang, Joydeep Biswas et Peter Stone. "LLM+P : Renforcer les grands modèles de langage avec une maîtrise optimale de la planification." arXiv :2304.11477, 2023. URL : <https://arxiv.org/abs/2304.11477>[13] Karthik Valmeekam, Matthew Marquez, Alberto Olmo, Sarath Sreedharan et Subbarao Kambhampati. « PlanBench : un référentiel extensible pour évaluer de grands modèles de langage sur la planification et le raisonnement sur le changement. » *Progrès dans les systèmes de traitement de l'information neuronale, les ensembles de données et les benchmarks*, 2023. URL : <https://openreview.net/forum?id=YXogl4uQUO>

[14] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan et Yuan Cao. "ReAct : mettre en synergie le raisonnement et l'action dans les modèles linguistiques." *Conférence internationale sur les représentations de l'apprentissage (ICLR)*, 2023. URL : <https://openreview.net/forum?id=WE_vluYUL-X>

[15] Michael Ahn et coll. "Faites ce que je peux, pas ce que je dis : ancrer le langage dans les moyens robotiques." *Conférence sur l'apprentissage des robots (CoRL)*, 2022. URL : <https://arxiv.org/abs/2204.01691>[16] Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence et Andy Zeng. "Code en tant que politiques : programmes de modèles de langage pour un contrôle incorporé." *Conférence internationale de l'IEEE sur la robotique et l'automatisation (ICRA)*, 2023. URL : <https://arxiv.org/abs/2209.07753>

[17] Ishika Singh, Valts Blukis, Arsalan Mousavian, Ankit Goyal, Danfei Xu, Jonathan Tremblay, Dieter Fox, Jesse Thomason et Animesh Garg. « ProgPrompt : Génération de plans de tâches de robot localisés à l'aide de grands modèles de langage. » *Conférence internationale de l'IEEE sur la robotique et l'automatisation (ICRA)*, 2023 ; version étendue dans *Autonomous Robots*, 2023. URL : <https://arxiv.org/abs/2209.11302>

[18] Hadas Kress-Gazit, Georgios E. Fainekos et George J. Pappas. « Mission réactive et planification de mouvement basées sur la logique temporelle. » *Transactions IEEE sur la robotique*, 25(6):1370-1381, 2009. DOI : 10.1109/TRO.2009.2030225.[19] Hadas Kress-Gazit, Morteza Lahijanian et Vasumathi Raman. « Synthèse pour les robots : garanties et commentaires sur le comportement des robots. » *Revue annuelle du contrôle, de la robotique et des systèmes autonomes*, 1:211-236, 2018. DOI : 10.1146/annurev-control-060117-105838.

[20] Oded Maler et Dejan Nickovic. « Surveillance des propriétés temporelles des signaux continus. » *FORMATS/FTRTFT*, 2004. DOI : 10.1007/978-3-540-30206-3_12.

[21] Georgios E. Fainekos et George J. Pappas. "Robustesse des spécifications de logique temporelle pour les signaux à temps continu." *Informatique théorique*, 410(42):4262-4291, 2009. DOI : 10.1016/j.tcs.2009.06.021.

[22] Alexandre Donzé et Oded Maler. « Satisfaction robuste de la logique temporelle par rapport aux signaux à valeur réelle. » *FORMATS*, 2010. DOI : 10.1007/978-3-642-15297-9_12.[23] Vasumathi Raman, Alexandre Donze, Dorsa Sadigh, Richard M. Murray et Sanjit A. Seshia. « Synthèse réactive à partir des spécifications de la logique temporelle du signal. » *Systèmes hybrides : calcul et contrôle (HSCC)*, 2015. DOI : 10.1145/2728606.2728628.

[24] Shromona Ghosh, Dorsa Sadigh, Pierluigi Nuzzo, Vasumathi Raman, Alexandre Donze, Alberto L. Sangiovanni-Vincentelli et Sanjit A. Seshia. « Diagnostic et réparation pour la synthèse à partir des spécifications de la logique temporelle du signal. » *Systèmes hybrides : calcul et contrôle (HSCC)*, 2016. DOI : 10.1145/2883817.2883847.

[25] Alexandre Duret-Lutz, Alexandre Lewkowicz, Amaury Fauchille, Thibault Michaud, Étienne Renault et Laurent Xu. «Spot 2.0 - Un cadre pour la manipulation de LTL et d'oméga-automates.» *Technologie automatisée de vérification et d'analyse (ATVA)*, 2016. URL : <https://spot.lre.epita.fr/>[26] Tomoya Yamaguchi, Bardh Hoxha et Dejan Nickovic. "RTAMT - Moniteurs de robustesse d'exécution avec application aux CPS et à la robotique." *Journal international sur les outils logiciels pour le transfert de technologie*, 26(1):79-99, 2024 ; arXiv : 2501.18608, 2025. DOI : 10.1007/S10009-023-00720-3. URL : <https://arxiv.org/abs/2501.18608> ; code : <https://github.com/nickovic/rtamt>

[27] Marta Kwiatkowska, Gethin Norman et David Parker. «PRISM 4.0 : Vérification des systèmes probabilistes en temps réel.» *Vérification assistée par ordinateur (CAV)*, 2011. URL : <https://www.prismmodelchecker.org/bibitem.php?key=KNP11>

[28] Mohammed Alshiekh, Roderick Bloem, Rüdiger Ehlers, Bettina Könighofer, Scott Niekum et Ufuk Topcu. « Apprentissage par renforcement sécurisé via le blindage. » *Actes de la conférence AAAI sur l'intelligence artificielle*, 2018. URL : <https://ojs.aaai.org/index.php/AAAI/article/view/11797>[29] Edwin Hamel-De le Court, Francesco Belardinelli et Alexander W. Goodall. « Protection probabiliste pour un apprentissage par renforcement sûr. » *Actes de la conférence AAAI sur l'intelligence artificielle*, 39(15):16091-16099, 2025. DOI : 10.1609/aaai.v39i15.33767. URL : <https://ojs.aaai.org/index.php/AAAI/article/view/33767>

[30] Shubo Liu, Hongsheng Zhang, Yuankai Qi, Peng Wang, Yanning Zhang et Qi Wu. "AerialVLN : navigation visuelle et linguistique pour les drones." *Conférence internationale IEEE/CVF sur la vision par ordinateur (ICCV)*, 2023, pp. URL : <https://openaccess.thecvf.com/content/ICCV2023/html/Liu_AerialVLN_Vision-and-Language_Navigation_for_UAVs_ICCV_2023_paper.html>[31] Xiangyu Wang, Donglin Yang, Ziqin Wang, Hohin Kwan, Jinyu Chen, Wenjun Wu, Hongsheng Li, Yue Liao et Si Liu. "Vers une navigation réaliste en langage vision pour drones : plate-forme, référence et méthodologie." *Conférence internationale sur les représentations de l'apprentissage (ICLR)*, 2025. URL : <https://openreview.net/forum?id=rUvCIvI4eB> ; arXiv :2410.07087.

[32] Sourav Sanyal et Kaushik Roy. "ASMA : un algorithme de marge de sécurité adaptatif pour la navigation des drones en langage visuel via des fonctions de barrière de contrôle sensibles à la scène." arXiv : 2409.10283, 2024 ; accepté par *IEEE Robotics and Automation Letters*. URL : <https://arxiv.org/abs/2409.10283>

[33] Xinyuan Zhang, Yonglin Tian, Fei Lin, Yue Liu, Jing Ma, Kornélia Sára Szatmáry et Fei-Yue Wang. « LogisticsVLN : navigation en langage visuel pour la livraison de terminaux à basse altitude basée sur des drones agents. » arXiv :2505.03460, 2025. URL : <https://arxiv.org/abs/2505.03460>[34] Hanxuan Chen, Jie Zheng, Siqi Yang, Tianle Zeng, Siwei Feng, Songsheng Cheng, Ruilong Ren, Hanzhong Guo, Shuai Yuan, Xiangyue Wang, Kangli Wang et Ji Pei. « Navigation visuelle et linguistique pour les drones : progrès, défis et feuille de route de recherche. » arXiv :2604.13654, 2026. URL : <https://arxiv.org/abs/2604.13654>

[35] Équipe Qwen. «Rapport technique Qwen3.» arXiv :2505.09388, 2025. URL : <https://arxiv.org/abs/2505.09388>

[36] DeepSeek-AI. "DeepSeek-R1 : Inciter la capacité de raisonnement dans les LLM via l'apprentissage par renforcement." arXiv :2501.12948, 2025. URL : <https://arxiv.org/abs/2501.12948>

[37] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang et Ion Stoica. «Gestion efficace de la mémoire pour la diffusion de modèles de langage étendus avec PagedAttention.» *Symposium ACM sur les principes des systèmes d'exploitation (SOSP)*, 2023. URL : <https://arxiv.org/abs/2309.06180>[38] AAAI. « Piste technique principale AAAI-26 : appel à communications » et « Liste de contrôle de reproductibilité AAAI-26 ». 2025. URL : <https://aaai.org/conference/aaai/aaai-26/main-technical-track-call/> ; <https://aaai.org/conference/aaai/aaai-26/reproducibility-checklist/>

[39] Société des systèmes de transport intelligents IEEE. "Transactions IEEE sur les systèmes de transport intelligents (T-ITS) : portée." URL : <https://ieee-itss.org/pub/t-its/>

---

## 15. Annexe : Plan de promotion prioritaire actuel de l'AAAI

### 15.1 Positionnement du papier

La version AAAI est d'abord réduite à un article sur la méthode de l'IA :

** Planification linguistique vérifiée vers STL guidée par contre-exemple pour les drones **

Le noyau n'est pas "LLM peut planifier des drones", mais : une fois que le LLM open source local a généré les spécifications de la mission du drone, un vérificateur formel est utilisé pour générer un diagnostic de contre-exemple, puis le LLM est amené à corriger les spécifications ou le plan, et génère enfin une trajectoire vérifiable.

### 15.2 Relevé de contribution AAAI

AAAI préconise trois contributions :

1. Une chaîne de compilation des spécifications de mission dactylographiée IR à LTL/STL UAV couvrant les contraintes d'arrivée, d'évitement, de séquence, d'inspection, de fenêtre horaire, d'altitude et de distance.
2. Une boucle de réparation guidée par vérification qui convertit les erreurs de syntaxe, la mise à la terre manquante, les trajectoires insatisfaisantes et dangereuses et la faible robustesse STL en retour de contre-exemple structuré.
3. Un benchmark UAV-NL2STL, comprenant des tâches en langage naturel, des cartes, des spécifications de référence, des traces exécutables et des étiquettes de diagnostic de panne.

### 15.3 Chronologie| Temps | Tâche | Sortie |
|------|------|------|
| 2026-05-18 au 2026-05-24 | Complétez le tableau de la littérature de base et figez le schéma de référence | Table de travail associée + spécification de l'ensemble de données |
| 2026-05-25 au 2026-06-07 | Implémenter un générateur de cartes/tâches, un modèle Gold TaskIR/LTL/STL, un planificateur de base | Script de génération de données + planificateur de base |
| 2026-06-08 au 2026-06-21 | Implémenter le vérificateur Spot/RTAMT et les contre-exemples | module vérificateur |
| 2026-06-22 au 2026-07-05 | Exécuter un modèle local, une ligne de base, une expérience préliminaire sans réparation/réparation complète | Tableau des résultats du premier modérateur |
| 2026-07-06 au 2026-07-19 | Expérience principale, ablation, généralisation, statistiques des cas d'échec | Tableau d'expérimentation complet et figures |
| 2026-07-20 à la date limite des résumés AAAI | Résumé complet, introduction, méthode, figure 1, tableau principal des résultats | Première ébauche de l'AAAI |
| Texte intégral AAAI avant la date limite | Compressé à 7 pages, ajouter une annexe, reproductibilité, référentiel anonyme | Dossier de soumission |

Au 2026-05-19, l'AFC officiel de la piste technique principale AAAI-27 n'a pas été récupéré sur le site officiel de l'AAAI ; actuellement, le contenu technique de 7 pages, la liste de contrôle de reproductibilité et les exigences de l'annexe code/données de la piste technique principale AAAI-26 sont toujours prioritaires comme base pour l'inversion [38]. Une fois l'appel à propositions AAAI-27 publié, ce calendrier doit être mis à jour dès que possible, en particulier la date limite des résumés, la date limite du texte intégral, la date limite des documents supplémentaires et la politique de texte généré par LLM.

### 15.4 Extensions ultérieures des T-ITS

Lorsque AAAI sera ensuite étendu à T-ITS, le nouveau contenu devra être clairement différent de la version de la conférence. Il est recommandé d'ajouter :- Expérience AirSim/SUMO ou Low Altitude Logistics Digital Twin.
- Coordination multi-UAV et arbitrage des conflits dans l'espace aérien.
- Indicateurs du système de trafic : débit des missions, occupation de l'espace aérien, marge de sécurité, taux d'achèvement des livraisons/inspections, robustesse des délais de communication.
- Expérience de déploiement Edge : compromis latence-énergie pour les modèles 4 bits / 8 bits sur Jetson ou 4090.
- Le titre est passé de « méthode de planification vérifiée » de l'AAAI à « Exploitation sûre d'UAV à basse altitude pour les systèmes de transport intelligents ».

---

**Notes de version :** Le contenu de cet article a été mis à jour vers `v2`, mais le nom du fichier continue d'être `v1-20260517` pour répondre à l'exigence de "modifier directement sur la version V1" de ce cycle. L'optimisation incrémentielle du 19/05/2026 complète la prévention des fuites de données, la taxonomie des échecs, la budgétisation des paramètres, les formules d'indicateurs, la planification des graphiques et les risques de conformité AAAI. Dans la prochaine version, il est recommandé de mettre à jour vers « v3-AAAAMMJJ » après avoir terminé le schéma de l'ensemble de données et le premier cycle d'exécution de la ligne de base, en se concentrant sur le remplacement de la table « À déterminer » et en complétant les résultats expérimentaux réels et les cas d'échec.