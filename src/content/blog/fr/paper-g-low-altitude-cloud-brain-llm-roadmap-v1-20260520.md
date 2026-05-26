---
title: "Paper G Planning v1 : Agent LLM et modèle d'itinéraire de réglage fin dans le cerveau du nuage de trafic à basse altitude"
description: "Planifiez comment entraîner ou affiner le LLM pour en faire un agent vérifiable dans le cerveau du nuage de trafic à basse altitude, et formez le premier document de conférence AAAI/IJCAI, des revues de transport de suivi et un itinéraire général de transformation des agents incarnés."
pubDate: 2026-05-20
updatedDate: 2026-05-23
tags: ["Papier G", "Cerveau de nuage de trafic à basse altitude", "Agent LLM", "Mise au point du modèle", "Utilisation des outils", "AAAI", "IJCAI", "drone", "AGI"]
category: Tech
sourceHash: "f37d501e3b68b293076f6f90f7f7c3da5c91513d"
---

# Paper G Planning v1 : Agent LLM et modèle d'itinéraire de réglage fin dans le cerveau du nuage de trafic à basse altitude

> Jugement global : cet itinéraire ne doit pas d'abord être un "grand modèle de discussion sur le trafic à basse altitude", mais doit être un **agent LLM vérifiable dans le cerveau du nuage de trafic à basse altitude**.  
> Donner la priorité à AAAI / IJCAI pour le premier article : placez LLM dans la position de « compréhension des tâches, invocation d'outils, planification et réparation, vérification en boucle fermée et explication de la planification » au lieu de s'engager directement dans la formation d'un modèle de base à grande échelle.

---

## 1. Jugement global : pourquoi créer d'abord le cerveau du cloud d'agent au lieu de former directement le grand modèle ?

Si vous écrivez directement « affiner un LLM sur le trafic à basse altitude », les évaluateurs de la conférence sont susceptibles de poser trois questions :

1. **Quelle est la contribution du modèle ? **
   LoRA / SFT / DPO lui-même est déjà un processus de formation standard [14] [15] [16]. Le simple remplacement des données par un corpus de trafic à basse altitude est difficile à soutenir la conférence principale AAAI/IJCAI.

2. **Pourquoi le LLM est-il plus nécessaire que les modèles de planification/ordonnancement existants ? **
   L'exploitation du trafic à basse altitude implique la planification, la planification des itinéraires, l'évaluation des risques, la vérification formelle et le retour d'expérience de la simulation. L’avantage du LLM n’est pas de remplacer ces modèles, mais de décomposer des tâches complexes en chaînes d’outils appelables.

3. **Comment assurer la sécurité ? **
   Le cerveau du nuage de trafic à basse altitude est un système critique pour la sécurité. La sortie directe d’actions de contrôle à partir de LLM présente un risque d’hallucination et de non-vérification. Le premier article doit mettre le vérificateur, le simulateur et l’estimateur de risques dans une boucle fermée.

Par conséquent, il n’est pas recommandé que le premier article du Paper G soit intitulé « LowAltitudeGPT ». Un meilleur premier article serait :

> **CloudBrain-Agent : agents LLM améliorés par des outils et guidés par la vérification pour l'exploitation du trafic à basse altitude**

Sa principale contribution n’est pas « le modèle devient plus intelligent », mais :

- Construire un pipeline de décision agentique pour le cerveau des nuages ​​de trafic à basse altitude ;
- Laissez LLM apprendre à appeler les véhicules à basse altitude ;
- Utiliser des vérificateurs et des émulateurs pour corriger les erreurs ;
- Produire des décisions de planification/planification exécutables, interprétables et vérifiables.Ceci est proche de l'idée de TrafficGPT : TrafficGPT a souligné que LLM lui-même est difficile à gérer les données numériques de trafic et l'interaction de simulation, il doit donc être combiné avec des modèles de base de trafic [1]. La différence dans l'article G est que nous avons étendu l'objet du transport terrestre au transport à basse altitude, et ajouté le statut des drones, les contraintes de l'espace aérien, la vérification formelle et la boucle fermée de sécurité.

À partir d'un examen plus large de l'intelligence routière, LLM a été discuté comme une interface sémantique, un module de raisonnement et un composant auxiliaire de prise de décision en matière de trafic dans ITS [2] [3] ; UrbanGPT et UniST illustrent que la prévision spatio-temporelle urbaine est en train de passer à un modèle de fondation spatio-temporelle [4] [5]. L'article G ne répète pas directement ces directions, mais combine « l'intelligence spatio-temporelle urbaine + les outils d'exploitation des drones + les agents vérifiables » dans un cerveau nuageux de trafic à basse altitude.

### 1.1 2026-05-22 Calibrage de l'écriture : G1 est un article d'agent d'IA, et l'expansion du journal nécessite un récit complet du système de transport.

L'article G pourrait facilement être considéré comme une « histoire de grand modèle de trafic à basse altitude ». Cette voie nécessite de distinguer deux critères d’évaluation :

| Étapes | Objectifs | Logique de révision principale | Des erreurs impossibles à commettre |
|------|------|--------------|--------------|
| G1 AAAI/IJCAI | Méthode d'agent LLM vérifiable | utilisation des outils, planification, vérification, benchmark, reproductibilité | Sacrifier la clarté de la méthode pour le récit du trafic ou écrire l'agent sous forme d'affichage de plate-forme |
| G2 T-ITS/T-IV | Mise au point du LLM dans le domaine du transport à basse altitude | Données de domaine, reproductibilité du déploiement et capacités d'aide à la prise de décision en matière de trafic | Uniquement LoRA/SFT général, pas de chaîne de transport ni d'indicateurs de sécurité |
| G3 AAMAS/T-ITS | Collaboration cloud-cerveau multi-agents | Collaboration multi-rôles, communication, gestion des conflits, collaboration homme-machine | Le multi-agent n'est que plusieurs invites, sans état du système ni limites de responsabilité |
| Version étendue du journal | Importance de l'exploitation du système de transport | Sécurité, efficacité, capacité, délais, utilisation des ressources, inspiration de gestion | Signalez uniquement la précision/le succès de l'appel d'outil, ne répondez pas aux questions sur le trafic |Par conséquent, la ligne principale du G1 reste les méthodes d’IA puissantes : IR typée, utilisation d’outils, réparation du vérificateur et évaluation avec état.
Cependant, tous les indicateurs liés au trafic à basse altitude doivent être conservés dès le départ pour faciliter l'expansion ultérieure des T-ITS :

- Sécurité : proxy LoWC/NMAC, violation de la zone d'exclusion aérienne, violation de la réserve de batterie.
- Efficacité : délai, distance supplémentaire, énergie, débit, durée d'exécution.
- Gestion des opérations : taux de refus sécurisé, taux de confirmation humaine, gestion des tâches ambiguës.
- Robustesse : perte de communication, perturbation météo, drone non coopératif, ville/topologie invisible.
- Éclairage du système : dans quelles conditions l'agent LLM doit sortir vers le solveur déterministe ou le superviseur humain.

### 1.2 2026-05-23 Disposition : L'ordre des routes G

Paper G est une feuille de route générale, et ce qui doit réellement être achevé dans un avenir proche est **G1 CloudBrain-Agent**. Actuellement, la voie la plus rapide et la plus soumise n'est pas de former d'abord un grand modèle vertical, mais d'utiliser un modèle général fort + IR typé + chaîne d'outils + vérificateur + retour du simulateur pour former une boucle fermée reproductible. La formation du modèle vertical est placée dans G2, et les traces d'appel d'outil, les traces de réparation et les cas de défaillance générés par G1 sont utilisés comme données.| Scène | S'il faut entraîner le modèle | Modèle/déploiement recommandé | Objectif |
|------|--------------|--------------------|------|
| G1 maintenant | Pas en tant que contributeur principal à la formation | Le vLLM local exécute Qwen/DeepSeek, le modèle API fait l'enseignant/la limite supérieure | Prouver que l'appel aux outils d'agent, la réparation de vérification et le benchmark des tâches à basse altitude sont efficaces |
| G2 suivant | LoRA/SFT/DPO | Affiner les séries Qwen / Llama / DeepSeek avec les traces G1 | Formation du module cognitif du domaine LowAltitudeGPT |
| G3 plus tard | Distillation de trajectoire multi-agents en option | Agent multi-rôle + mémoire partagée + vérificateur | Recherche sur la surveillance de l'espace aérien, la planification, les risques, les urgences, la collaboration homme-machine |
| G4 long terme | Multimodal / modèle mondial / VLA | Dépend des données et de la puissance de calcul | Migrer vers l'intelligence embarquée du trafic |

Les recommandations de stratégie de déploiement sont les suivantes :

- **Modèle open source local pour l'expérience principale** : coût reproductible et contrôlable, rapport de latence et de conditions matérielles faciles à signaler ; il est recommandé d'utiliser vLLM / llama.cpp comme service d'inférence.
- **Modèle API en tant qu'enseignant et limite supérieure** : utilisé pour générer des échantillons initiaux de haute qualité, des exemples de démonstrations de réparation difficiles et une ligne de base de limite supérieure ; Les résultats de l'API et les résultats du modèle local doivent être rapportés séparément dans le document.
- **MCP développe d'abord le style d'interface, pas la production en premier** : La première version implémente d'abord le registre de l'outil Python et le schéma JSON ; une fois l'outil stable, il sera intégré à un serveur compatible MCP pour éviter de placer la complexité technique dans la ligne principale du document.
- **La formation du modèle vertical ne saisit pas la ligne principale de G1** : la contribution de G1 est l'architecture de l'agent et la vérification en boucle fermée ; G2 distille uniquement la trajectoire de course dans le modèle local.

Cette séquence peut rapidement former une boucle fermée qui peut être soumise : laissez d'abord le système s'exécuter, évaluer et expliquer les pannes, puis décider quelles fonctionnalités méritent d'être affinées dans le modèle.

---

## 2. Définition du système du cerveau du nuage de trafic à basse altitudeLe « cerveau du nuage de trafic à basse altitude » dans cet article n'est pas une plate-forme intelligente générale, mais une **couche d'opérations cognitives** pour le fonctionnement des drones urbains à basse altitude :

```text
Human / operator instruction
  -> CloudBrain LLM Agent
  -> LowAltitudeIR
  -> traffic tools / UAV tools / verifier / simulator
  -> safe decision proposal
  -> human approval or autonomous execution
```

### 2.1 Entrée

Le cerveau du nuage de trafic à basse altitude reçoit un statut multi-source :

| Entrée | Exemple |
|------|------|
| Tâche en langage naturel | "Donner la priorité aux livraisons d'urgence à proximité des hôpitaux et éviter les écoles et les zones d'exclusion aérienne." |
| Statut du drone | Position, puissance, charge, état de mission, état de communication |
| Statut de l'espace aérien | capacité du corridor, zones d'exclusion aérienne, contrôles temporaires, météo, champs de vent |
| Besoins en transport | Ordres de livraison, tâches d'inspection, urgences, priorité passagers/fret |
| Statut de la scène | Scénarios critiques pour la sécurité, scénarios d'accidents, trous de couverture du papier F |
| Contraintes formelles | Règles de sécurité LTL/STL, fenêtres horaires, hauteurs minimales, intervalles minimaux |

### 2.2 Sortie

Le cerveau nuageux ne produit pas directement des « actions de vol », mais génère des décisions intermédiaires vérifiables :

| Sortie | Exemple |
|------|------|
| Faible AltitudeIR | Tâches structurées, entités, contraintes, plans d'appel d'outils |
| Séquence d'appel d'outil | Interroger l'espace aérien, planificateur d'appels, planificateur de chemin d'appel, exécuter le vérificateur |
| Recommandations de planification | Quel drone effectue quelle tâche et s'il faut déclencher un repli au sol |
| Diagnostic de sécurité | Quelles contraintes peuvent être violées et si une confirmation manuelle est requise |
| Texte d'explication | Expliquez en langage naturel pourquoi il est programmé de cette façon |

### 2.3 Cloud Brain n'est pas un contrôleur de bout en bout

Les limites du cerveau des nuages de trafic à basse altitude doivent être clairement écrites :

- LLM effectue la compréhension sémantique, la décomposition des tâches, la sélection, l'interprétation et la réparation des outils.
- Le planificateur effectue l'affectation de la flotte et l'optimisation des ressources, correspondant au papier B.
- Le validateur effectue des contrôles de sécurité LTL/STL, correspondant au papier E.
- Le simulateur de scénarios et le générateur de risques fournissent des tests de résistance, correspondant au papier F.
- Le contrôleur de trajectoire est toujours exécuté par le module traditionnel de planification/MPC/contrôle de sécurité.

Cela évite au critique le doute selon lequel "le contrôle LLM du drone n'est pas sûr".

---

## 3. Aperçu des axes de recherche : du domaine LLM à l'agent incarné général

L'épreuve G peut être divisée en 4 étapes.| Étapes | Thèse | Objectifs | Questions clés |
|------|------|------|--------------|
| G1 | Agent CloudBrain | AAAI/IJCAI | LLM Comment appeler de manière fiable des outils dans le cerveau du nuage de trafic à basse altitude et réussir la réparation en boucle fermée de vérification |
| G2 | BasseAltitudeGPT | T-ITS / T-IV | Comment affiner un LLM open source local pour devenir un module cognitif de prise de décision en matière de trafic à basse altitude |
| G3 | Cerveau cloud multi-agents | AAMAS / IJCAI / T-ITS | Comment plusieurs agents à temps plein collaborent pour gérer le trafic à basse altitude |
| G4 | Modèle mondial / Extension VLA | Itinéraire à long terme | Comment passer d'un agent de domaine à une intelligence générale incarnée |

La séquence recommandée est **G1 -> G2 -> G3 -> G4**.

G1 détermine d'abord « si le système peut fonctionner, s'il peut fonctionner en boucle fermée en toute sécurité et s'il peut tenir des réunions ». G2 distille ensuite la trajectoire de l'agent dans un modèle de domaine. G3 utilise la collaboration multi-agents. La transformation AGI n'est abordée que dans G4 et ne sera pas exagérée dans le premier article.

---

## 4. Article G1 : CloudBrain-Agent, le premier article de conférence pour AAAI/IJCAI

### 4.1 Question

**CloudBrain-Agent : agents LLM améliorés par des outils et guidés par la vérification pour le fonctionnement du trafic à basse altitude**

### 4.2 Objectif atteint

Premier pitch : AAAI/IJCAI.  
Alternatives : AAMAS, atelier ICRA/IROS, extension rapide du journal T-ITS.La piste technique principale AAAI-26 encourage le travail dans les directions technologiques de l'IA et dans les domaines d'application importants tels que le transport. Le texte principal est limité à 7 pages de contenu technique et nécessite une liste de contrôle de reproductibilité [34]. Le parcours spécial IA et robotique de l'IJCAI-ECAI 2026 se concentre explicitement sur les agents robots, l'IA générative, le contrôle des robots, la modélisation structurée, le raisonnement et la manière d'effectuer/éviter les conséquences des actions [35]. Par conséquent, G1 doit être rédigé comme un document d'agent d'IA/planification/utilisation d'outils/vérification plutôt que comme une démonstration d'ingénierie système.

### 4.3 Problèmes fondamentaux

G1 veut répondre :

> Étant donné une tâche d'exploitation du trafic à basse altitude, comment faire en sorte que l'agent LLM comprenne la tâche de manière fiable, sélectionne les outils, appelle le module de planification/planification/vérification et corrige les erreurs sous contre-exemple, produisant ainsi des décisions cloud brain sûres, exécutables et explicables ?

### 4.4 Méthode

**CloudBrain-Agent** proposé, comprenant cinq modules :

| Module | Fonction |
|------|------|
| Analyseur LowAltitudeIR | Convertir les tâches en langage naturel et les états du système en représentations structurées |
| Planificateur d'outils | Séquence d'appel de l'outil de planification |
| Exécuteur d'outils | Planificateur d'appels, planificateur de parcours, vérificateur, simulateur, évaluateur de risques |
| Boucle de rétroaction du vérificateur | Convertissez les appels d'outils ayant échoué, les contraintes insatisfaisantes et les échecs de robustesse STL en retours de réparation |
| Mémoire de sécurité | Enregistrez les scénarios de danger connus, les cas de défaillance, les décisions manuelles et les contraintes de règles |

Forme de comportement de CloudBrain-Agent :

```text
Observe -> Think -> Select Tool -> Execute -> Verify -> Repair -> Decide
```

Celui-ci hérite de la boucle raisonnement-action de ReAct [6], mais ajoute deux mécanismes spécifiques au trafic à basse altitude :

1. **Les appels d'outils doivent être de type sécurisé** : chaque entrée et sortie d'outil est vérifiée par rapport au schéma `LowAltitudeIR`.
2. **Les décisions doivent passer par le vérificateur** : toute recommandation de planification ou de chemin doit être soumise à une vérification de sécurité ou à des tests de résistance par simulation.### 4.5 LowAltitudeIR

LowAltitudeIR est l'interface publique clé de G1 :

```json
{
  "intent": "emergency_delivery",
  "entities": ["uav_12", "hospital_zone", "landing_pad_A"],
  "constraints": {
    "avoid": ["school_zone", "temporary_no_fly_zone"],
    "deadline_sec": 600,
    "min_obstacle_distance_m": 10,
    "altitude_range_m": [30, 120]
  },
  "tool_plan": [
    "query_airspace",
    "assign_uav",
    "plan_route",
    "verify_stl",
    "simulate_scenario"
  ],
  "fallback": "ground_vehicle_transfer_if_unreachable"
}
```

LowAltitudeIR devrait être compatible avec trois lignes papier existantes :

- Papier B : File d'attente des tâches, allocation des drones, ressources vertiport/chargement/couloir.
- Epreuve E : TaskIR, LTL/STL, vérification et correction d'erreurs ; les bases de citations associées incluent Lang2LTL, LTLCodeGen et ConformalNL2LTL [20] [21] [22].
- Epreuve F : Génération de scènes, trous de couverture, tests de stress sur scènes dangereuses.

### 4.6 Collection d'outils

Les outils de G1 n'ont pas besoin d'être construits sur des systèmes réels au départ. Vous pouvez d’abord construire des outils expérimentaux reproductibles :

| Outil | Entrée | Sortie |
|------|------|------|
| `query_airspace` | Région, heure, type de mission | couloir, zone d'exclusion aérienne, météo, capacité |
| `assign_uav` | Tâche, statut du drone, priorité | Affectation des tâches du drone |
| `plan_route` | début, fin, contrainte | chemin ou `INACCESSIBLE` |
| `verify_ltl_stl` | Spécification de la tâche, trajectoire | réussite / échec / contre-exemple |
| `simulate_scenario` | graine de scénario, stratégie | succès, collision, retard, risque |
| `risk_assess` | Tâches et scénarios | Niveau de risque, principales contraintes |
| `explain_decision` | Trajectoire décisionnelle | Explication lisible par l'homme |

### 4.7 Lignes de base| Référence | Descriptif |
|--------------|------|
| Décision directe LLM | LLM donne directement des suggestions de planification/chemin |
| ReAct en mode invite uniquement | Appel à l'outil de style ReAct, mais sans contraintes de type ni vérificateur [6] |
| Utilisation d'outils de style Toolformer / ToolLLM | Apprenez à appeler des outils, mais n'effectuez pas de vérification de sécurité de bas niveau [7] [8] |
| Orchestration de style TrafficGPT | LLM appelle le modèle de trafic, mais sans contraintes UAV ni vérification formelle [1] |
| LLM+P / planificateur classique | Problème de conversion LLM, résolu par un planificateur externe [10] |
| VERA-UAV uniquement | Seule la vérification de la langue aux spécifications, pas de planification multi-outils cloud brain |
| CloudBrain-Agent complet | LowAltitudeIR + utilisation de l'outil + vérificateur + feedback du simulateur |

PlanBench et les études critiques ultérieures sur les capacités de planification LLM ont montré que le simple fait de laisser LLM planifier verbalement n'est pas fiable et que des planificateurs externes, des contrôles de contraintes et des tâches expérimentales reproductibles doivent être introduits [11] [12]. Dans le même temps, AerialVLN et les travaux réalistes d’UAV-VLN peuvent être utilisés comme source de référence pour la navigation en langage visuel à basse altitude [23] [24] ; DriveLM, LMDrive, DriveVLM et LaMPilot peuvent être utilisés comme référence horizontale pour le benchmark VLM/LLM de conduite autonome et le paradigme de prise de décision en boucle fermée [25] [26] [27] [28].

### 4.8 Indicateurs d'évaluation| Indicateur | Signification |
|------|------|
| Taux de réussite des tâches | Taux d'achèvement des tâches du cerveau cloud |
| Précision de l'appel d'outil | Si la sélection de l'outil et les paramètres sont corrects |
| Taux de décision exécutable | Si la sortie peut être exécutée par le planificateur/planificateur |
| Taux d'infractions à la sécurité | Si la zone d'exclusion aérienne, la distance, l'altitude et le délai sont violés |
| Taux d'hallucinations | Qu'il s'agisse de référencer une entité, un outil ou un état inexistant |
| Taux de réussite des réparations | S'il peut être réparé après l'échec de la vérification |
| Taux de réussite au stress du simulateur | Taux de réussite dans le scénario hasardeux du papier F |
| Latence | Temps de décision pour une seule tâche |
| Généralisation | Performances sur des villes invisibles/tâches invisibles/combinaisons d'outils invisibles |

### 4.9 Points d'innovation attendus

1. Proposer « LowAltitudeIR » et une architecture d'agent utilisant des outils typés pour le cerveau des nuages de trafic à basse altitude.
2. Unifiez la planification, la planification du chemin, la vérification formelle et la simulation de scénarios dans la boucle fermée de prise de décision des agents LLM.
3. Proposer une réparation guidée par vérification afin que LLM ne repose plus uniquement sur une nouvelle tentative rapide.
4. Construire une référence cérébrale pour les nuages ​​de trafic à basse altitude, couvrant la décomposition des tâches, l'invocation d'outils, la planification, la vérification et l'interprétation.

---

## 5. Paper G2 : LowAltitudeGPT, mise au point du LLM dans le domaine du trafic à basse altitude

### 5.1 Question

**LowAltitudeGPT : LLM de réglage des instructions pour l'aide à la décision en matière de trafic à basse altitude**

### 5.2 Objectifs

G2 est le papier de mise au point du modèle. L'objectif est de distiller la trajectoire de fonctionnement de l'agent, les règles artificielles, les retours de simulation et les données de vérification et de réparation dans G1 dans un modèle open source local, afin que le modèle puisse devenir le module cognitif de domaine du cerveau du nuage de trafic à basse altitude.Candidatures : T-ITS, IEEE T-IV, Applied Intelligence, Knowledge-Based Systems. Le T-ITS est plus adapté pour mettre l'accent sur les systèmes de transport intelligents, les opérations de circulation et la prise de décision en matière de sécurité, et le T-IV est plus adapté pour mettre l'accent sur les modèles et l'évaluation de véhicules intelligents/systèmes sans pilote [36] [37]. Si la formation et l'évaluation du modèle sont suffisamment solides, vous pouvez également organiser un atelier AAAI / IJCAI ou une extension de la conférence principale.

### 5.3 Parcours de formation

Trois étapes sont recommandées :

| Étapes | Méthodes | Données |
|------|------|------|
| SFT | Ajustement LoRA / QLoRA [14] [15] | Questions et réponses sur le trafic à basse altitude, NL-to-IR, traces d'appels d'outils, interprétation d'urgence |
| Réglage des préférences | DPO / optimisation des préférences [16] | Des décisions sûres valent mieux que des décisions dangereuses, les séquences d'outils exécutables valent mieux que les séquences d'outils d'hallucination |
| RL vérifiable | Récompenses de règles basées sur un vérificateur et un émulateur | Tâches réussies, faible risque, faible latence, pas d'hallucination, vérifiées par STL |

DeepSeek-R1 montre que la capacité de raisonnement peut être stimulée par l’apprentissage par renforcement [19], mais G2 ne devrait pas entraîner le modèle de raisonnement à partir de zéro. Une voie plus réaliste consiste à utiliser le modèle open source Qwen/DeepSeek/Llama comme base, à utiliser LoRA/QLoRA pour un réglage fin efficace des paramètres, puis à utiliser la récompense du vérificateur pour un alignement à petite échelle.

### 5.4 Construction des données

Les données ne doivent pas uniquement être utilisées pour les questions-réponses du chat, mais doivent être divisées en 7 catégories :| Type de données | Exemple |
|--------------|------|
| Assurance qualité du domaine | "Comment gérer les tâches d'urgence lorsque la capacité du couloir de basse altitude est insuffisante ?" |
| NL-à-LowAltitudeIR | Tâches en langage naturel vers IR structuré |
| Trace d'appel d'outil | Corriger la séquence d'appel d'outil et les paramètres |
| Réparation de vérification | Échec du contre-exemple pour réparer l'IR |
| Explication de la planification | Explication du résultat de la planification |
| Intervention d'urgence | Gestion des situations d'urgence à grande vitesse/urbaines |
| Refus de sécurité | Refus/clarification en cas d'informations dangereuses ou insuffisantes |

Source des données :

- Génération procédurale : le générateur de scénarios Paper B/F produit des tâches, des cartes, des états et des résultats d'outils.
- Génération de vérification : échantillons de défaillance LTL/STL et échantillons de réparation pour le papier E.
- Relecture manuelle : échantillonnage et correction d'échantillons à haut risque pour garantir que les entités référencées, les contraintes et les paramètres de l'outil sont authentiques.
- Extension d'auto-instruction : utilisez l'idée d'auto-instruction pour étendre le modèle de tâche, mais elle doit passer par un filtrage de règles et un échantillonnage manuel [17].

### 5.5 Sélection du modèle

Suggestions de première édition :

- `Qwen2.5-7B/14B` : Bonnes capacités d'appel de code et d'outils en chinois/anglais [18].
- `DeepSeek-R1-Distill-Qwen-14B` : adapté aux correctifs d'inférence et de vérification [19].
- `Llama-3.1-8B` : comparaison de la base de référence en anglais et de l'écosystème open source.

Il n'est pas recommandé de former plus de 70 modèles B dans la première étape. L'article ne se concentre pas sur la taille du modèle, mais sur **l'alignement de l'utilisation des outils de domaine** et la **formation aux retours de vérification**.

### 5.6 Indicateurs d'évaluation| Indicateur | Signification |
|------|------|
| Correspondance exacte IR/champ F1 | Qualité de sortie structurée LowAltitudeIR |
| Succès de l'appel d'outil | Nom de l'outil, ordre, précision des paramètres |
| Taux de décision vérifié | Proportion de résultats réussissant le vérificateur |
| Précision du refus de sécurité | Faut-il refuser ou clarifier la tâche d'information dangereuse/insuffisante |
| Capacité de réparation | Taux de réussite des réparations après avoir vu un contre-exemple |
| Latence de déploiement local | Latence d'inférence locale et utilisation de la mémoire |
| Généralisation interurbaine | Généralisation de villes/scènes invisibles |

---

## 6. Article G3 : Multi-Agent Cloud Brain, cerveau cloud collaboratif multi-agents

### 6.1 Question

**Cerveau cloud multi-agents pour la gestion coopérative du trafic de drones à basse altitude**

### 6.2 Objectifs

G3 s'étend de la collaboration mono-agent à la collaboration multi-agents. Candidatures : AAMAS, IJCAI, AAAI, T-ITS.

AAMAS se concentrera sur les agents autonomes et les systèmes multi-agents [38], qui sont très adaptés à la collaboration multirôle dans les cerveaux des nuages ​​de trafic à basse altitude.

### 6.3 Division du travail des agents

| Agent | Responsabilités |
|-------|------|
| Moniteur de l'espace aérien | Surveiller les couloirs, les zones d'exclusion aérienne, la météo et la capacité |
| Planificateur de flotte | Responsable de la file d'attente des tâches et de la distribution des drones |
| Vérificateur de sécurité | Responsable LTL/STL, risques et contre-exemples |
| Testeur de scénarios | Appelez le générateur de scène Paper F pour faire un test de résistance |
| Coordonnateur d'urgence | Responsable des interventions d'urgence et de la liaison au sol |
| Agent d'interface humaine | Responsable de l'explication, de la clarification et de la confirmation humaine |### 6.4 Questions clés de recherche

1. Plusieurs agents sont-ils plus fiables qu’un seul agent ?
2. La mémoire partagée propagera-t-elle les erreurs ?
3. Lorsque deux agents sont en conflit, qui a le pouvoir de décision final ?
4. Le vérificateur peut-il agir comme arbitre ?
5. Le retard causé par plusieurs agents est-il acceptable ?

### 6.5 Points d'innovation

L'innovation de G3 ne réside pas dans "plusieurs GPT discutant entre eux", mais :

- Un agent à temps plein est lié aux véhicules à basse altitude ;
- L'état partagé est représenté par « LowAltitudeIR » et le journal des événements ;
- L'arbitrage de sécurité est réalisé par vérificateur et simulateur ;
- Les désaccords multi-agents peuvent se transformer en incertitude et en signaux d'intervention humaine.

---

## 7. Article G4 : Extension World-Model/VLA pour la migration générale des fonctionnalités AGI

### 7.1 Positionnement global

Le G4 est la voie à long terme et ne doit pas être surestimé dans les deux premiers articles. L'expression proposée est :

> **vers une intelligence embarquée générale du trafic**

Au lieu de "implémenter AGI".

L'agent incarné ouvert de Voyager et la base d'accessibilité du langage au robot de SayCan illustrent que pour que LLM évolue vers l'intelligence incarnée, la clé n'est pas d'être capable de discuter, mais d'être capable d'améliorer continuellement les commentaires environnementaux, les bibliothèques de compétences et les contraintes d'action [9] [13]. Le cerveau du nuage de trafic à basse altitude peut intégrer cette idée dans un domaine d’exploitation du trafic plus sûr et plus évaluable.

### 7.2 Pourquoi s'agit-il d'une entrée logique dans le sens AGI ?

Le cerveau du nuage de trafic à basse altitude contient naturellement plusieurs capacités requises pour l’intelligence incorporée générale :

- Compréhension spatiale : espace urbain 3D, obstacles, hiérarchie des espaces aériens.
- Raisonnement temporel : file d'attente des tâches, date limite, météo dynamique, évolution des événements de trafic.
- Utilisation des outils : ordonnanceur, planificateur, vérificateur, simulateur.
- Conséquences des actions : de mauvaises décisions peuvent entraîner des retards, des risques ou des violations de la sécurité.
- Collaboration multi-agents : drone, véhicule terrestre, opérateur humain, règles réglementaires.PaLM-E, RT-2 et OpenVLA ont démontré une tendance à passer de la pré-formation linguistique/visuelle à l'action incarnée [29][30][31]. Cependant, le cerveau du nuage de trafic à basse altitude ne devrait pas commencer par un VLA de bout en bout, mais devrait d'abord utiliser agent + outils + vérificateur pour établir une architecture cognitive de sécurité.

### 7.3 Feuille de route technique à long terme

| Scène | Capacité | Technologie |
|------|------|------|
| G1 | Appel d'outil et vérification en boucle fermée | Agent LLM + LowAltitudeIR |
| G2 | Modèle de domaine | Récompense SFT / LoRA / DPO / vérificateur |
| G3 | Collaboration multi-agents | mémoire partagée + arbitrage vérificateur |
| G4 | Modèle mondial | prédiction spatio-temporelle + retour d'expérience du simulateur |
| G5 | VLA / politique incarnée | Contribution multimodale aux recommandations d'action, mais toujours exécutée par la couche de sécurité |

Les mots clés de la transformation AGI devraient être : **généralisation, apprentissage continu, raisonnement incarné, auto-évaluation, création d'outils**. N'écrivez pas « Nous avons formé un modèle AGI ».

---

## 8. Construction des données et plan de formation

### 8.1 Tableau récapitulatif des données| Ensemble de données | Source | Utilisation |
|--------|------|------|
| Instruction basse altitude | Modèle manuel + génération LLM + échantillonnage manuel | Compréhension des tâches en langage naturel |
| LowAltitudeIR-Or | Génération de règles + correction manuelle | Formation et évaluation RI |
| ToolTrace-Banc | Agent G1 exécutant la trace | Outil d'appel SFT |
| VérifierRepair-Bench | Réparation du contre-exemple du papier E | Formation à la vérification et à la correction d'erreurs |
| ScénarioStress-Bench | Génération de scénarios Paper F | Généralisation des scènes dangereuses |
| FleetOps-Bench | Simulation de planification du papier B | File d'attente des tâches et planification des ressources |
| Banc d'opérations d'urgence | Cas de synthèse grande vitesse/urgence urbaine | Prise de décision d'urgence |

Dans la couche de simulation, il est recommandé d'utiliser d'abord un simulateur léger auto-construit pour garantir des variables contrôlables, puis d'utiliser AirSim et Flightmare pour une vérification supplémentaire de vol visuel, dynamique et en boucle fermée [32] [33]. De cette manière, G1/G2 peuvent être reproduits sans recourir à des simulateurs lourds et peuvent être naturellement étendus à des scénarios de drones plus réalistes à l’avenir.

### 8.2 Format de l'exemple de formation

Il est recommandé d'unifier en JSONL :

```json
{
  "instruction": "优先处理医院附近应急配送，避开学校和临时禁飞区。",
  "state": {
    "uavs": "...",
    "airspace": "...",
    "tasks": "..."
  },
  "target_ir": {
    "intent": "emergency_delivery",
    "constraints": ["avoid_school", "avoid_no_fly_zone"]
  },
  "tool_trace": [
    {"tool": "query_airspace", "args": {"region": "hospital_zone"}},
    {"tool": "assign_uav", "args": {"priority": "emergency"}},
    {"tool": "verify_ltl_stl", "args": {"spec": "..."}}
  ],
  "verifier_feedback": "pass",
  "final_answer": "建议派遣 uav_12，经 corridor_B 绕开学校区域。"
}
```

### 8.3 Phase de formation

1. **Invite + ligne de base RAG**
   Sans formation, vérifiez d’abord la définition de la tâche et le schéma de l’outil.

2. **SFT/LoRA**
   Le modèle entraîné génère des traces LowAltitudeIR et d’appel d’outil.

3. **DPO/réglage des préférences**
   Préférez des décisions sûres, exécutables, moins hallucinatoires et à faible latence.

4. **Alignement des récompenses du vérificateur**
   Utilisez les résultats du validateur et du simulateur comme récompenses de règles pour renforcer les capacités de réparation.

5. **Distillation**
   Distiller des modèles forts ou des trajectoires multi-agents en modèles locaux 7B/14B.

---

## 9. Conception expérimentale, données de référence et indicateurs d'évaluation

### 9.1 Expérience principale G1| Expérience | Objectif |
|------|------|
| Succès de l'utilisation des outils | Sélection des outils de test et remplissage des paramètres |
| Planification vérifiée | Tester si le planning/le chemin réussit la vérification |
| Boucle de réparation | Testez si les commentaires contre-exemples peuvent améliorer le taux de réussite |
| Test de résistance par scénario | Testez la robustesse avec les scénarios dangereux Paper F |
| Généralisation | Tester des villes invisibles, des tâches invisibles et des combinaisons d'outils invisibles |

### 9.2 Expérience de réglage fin du G2

| Expérience | Objectif |
|------|------|
| Base contre LoRA contre QLoRA | Vérifier les avantages du réglage fin |
| SFT contre DPO | Validation des avantages de l'alignement des préférences |
| Avec/sans feedback du vérificateur | Vérifier la valeur du retour de sécurité |
| 7B vs 14B vs modèle de raisonnement | Vérifier le compromis coût/performance du déploiement local |
| Transfert entre scénarios | Vérifier la migration du scénario synthétique vers le scénario d'urgence |

### 9.3 Lignes de base

| Référence | Descriptif |
|--------------|------|
| Réponse directe GPT/Qwen | Réponse directe, aucun outil |
| Invite ReAct | Invite au raisonnement-action [6] |
| Appel d'API de style Toolformer | Appel d'outil sans boucle fermée de sécurité [7] |
| Utilisateur d'outils formé de style ToolLLM | Base de référence pour la formation aux appels aux outils open source [8] |
| Orchestration du trafic de style TrafficGPT | LLM + modèles de trafic [1] |
| LLM+P | LLM + planificateur externe [10] |
| CloudBrain-Agent complet | Méthodes dans cet article |

### 9.4 Indicateurs| Métriques | Objectifs |
|------|------|
| Succès de la tâche | Taux d'achèvement des tâches du cerveau cloud |
| Précision de l'appel d'outil | Précision de l'appel d'outil |
| Champ IR F1 | Précision au niveau du champ LowAltitudeIR |
| Taux d'hallucinations | Ratio outils/entités/règles qui n'existent pas |
| Taux d'infractions à la sécurité | Proportion de violations des règles de sécurité |
| Succès de la réparation | Taux de réussite des réparations contre-exemple |
| Latence | Retard de décision |
| Score de confiance humaine | Qualité des explications de l'examinateur humain |
| Score de généralisation | Généralisation de scène invisible |

---

## 10. Chemin de soumission recommandé

### 10.1 Parcours du premier rendez-vous

**G1 premier vote AAAI / IJCAI. **

Type de papier : agent IA + planification + vérification + demande de transport.

Les contributions principales sont divisées en trois :

1. LowAltitudeIR et architecture d'agent utilisant des outils de trafic à basse altitude.
2. Boucle de réparation guidée par vérification.
3. Protocole d’évaluation et d’évaluation du cerveau des nuages ​​à basse altitude.

### 10.2 Itinéraire du journal de suivi

| Papier | Soumission |
|------|------|
| G2 basse altitudeGPT | T-ITS / T-IV / Intelligence Appliquée |
| Cerveau cloud multi-agent G3 | AAMAS -> Extension T-ITS |
| Modèle mondial G4/VLA | ICRA / IROS / T-RO / lieu orienté AGI longue durée |

### 10.3 Itinéraires non recommandés- Il n'est pas recommandé de former un grand modèle dans le premier article.
- Il n'est pas recommandé d'écrire "AGI Cloud Brain" comme titre principal.
- Il n'est pas recommandé de laisser LLM produire directement les actions de contrôle du drone.
- Il n'est pas recommandé de créer uniquement un ensemble de données de questions et réponses de chat.
- Il n'est pas recommandé d'ignorer le vérificateur, sinon les scénarios critiques pour la sécurité ne seront pas suffisamment convaincants.

---

## 11. Références

[1] Siyao Zhang, Daocheng Fu, Wenzhe Liang, Zhao Zhang, Bin Yu, Pinlong Cai et Baozhen Yao. « TrafficGPT : affichage, traitement et interaction avec les modèles de base de trafic. » *Politique des transports*, 150 :95-105, 2024. DOI : 10.1016/j.tranpol.2024.03.006. URL : <https://www.sciencedirect.com/science/article/pii/S0967070X24000726>

[2] Sebastian Wandelt, Changhong Zheng, Shuang Wang, Yucheng Liu et Xiaoqian Sun. « Grands modèles linguistiques pour le transport intelligent : un examen de l'état de l'art et des défis. » *Sciences appliquées*, 14(17):7455, 2024. DOI : 10.3390/app14177455. URL : <https://www.mdpi.com/2076-3417/14/17/7455>[3] Doaa Mahmud, Hadeel Hajmohamed, Shamma Almentheri, Shamma Alqaydi, Lameya Aldhaheri, Ruhul Amin Khalil et Nasir Saeed. "Intégrer les LLM avec les ITS : avancées récentes, potentiels, défis et orientations futures." *Transactions IEEE sur les systèmes de transport intelligents*, 26(5):5674-5709, 2025. DOI : 10.1109/TITS.2025.3528116. URL : <https://ieeexplore.ieee.org/document/10851302>

[4] Zhonghang Li, Lianghao Xia, Jiabin Tang, Yong Xu, Lei Shi, Long Xia, Dawei Yin et Chao Huang. "UrbanGPT : grands modèles de langage spatio-temporels." arXiv :2403.00813, 2024. URL : <https://arxiv.org/abs/2403.00813>

[5] Yuan Yuan, Jingtao Ding, Jie Feng, Depeng Jin et Yong Li. « UniST : un modèle universel optimisé pour la prévision spatio-temporelle urbaine. » *Actes de la conférence ACM SIGKDD sur la découverte des connaissances et l'exploration de données (KDD)*, 2024. DOI : 10.1145/3637528.3671662. URL : <https://arxiv.org/abs/2402.11838>[6] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan et Yuan Cao. "ReAct : mettre en synergie le raisonnement et l'action dans les modèles linguistiques." *Conférence internationale sur les représentations de l'apprentissage (ICLR)*, 2023. URL : <https://openreview.net/forum?id=WE_vluYUL-X>

[7] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda et Thomas Scialom. "Toolformer : les modèles de langage peuvent apprendre eux-mêmes à utiliser des outils." *Avances in Neural Information Processing Systems 36 (NeurIPS)*, 2023. URL : <https://proceedings.neurips.cc/paper_files/paper/2023/hash/d842425e4bf79ba039352da0f658a906-Abstract-Conference.html>[8] Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Runchu Tian, ​​Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu et Maosong Sun. "ToolLLM : Faciliter les grands modèles de langage pour maîtriser plus de 16 000 API du monde réel." *Conférence internationale sur les représentations de l'apprentissage (ICLR)*, 2024. URL : <https://openreview.net/forum?id=dHng2O0Jjr>

[9] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan et Anima Anandkumar. "Voyager : un agent incarné ouvert avec de grands modèles de langage." arXiv :2305.16291, 2023. URL : <https://arxiv.org/abs/2305.16291>

[10] Bo Liu, Yuqian Jiang, Xiaohan Zhang, Qiang Liu, Shiqi Zhang, Joydeep Biswas et Peter Stone. "LLM+P : Renforcer les grands modèles de langage avec une maîtrise optimale de la planification." arXiv :2304.11477, 2023. URL : <https://arxiv.org/abs/2304.11477>[11] Karthik Valmeekam, Matthew Marquez, Alberto Olmo, Sarath Sreedharan et Subbarao Kambhampati. « PlanBench : un référentiel extensible pour évaluer de grands modèles de langage sur la planification et le raisonnement sur le changement. » *Avances in Neural Information Processing Systems 36 (NeurIPS) Datasets and Benchmarks Track*, 2023. URL : <https://openreview.net/forum?id=YXogl4uQUO>

[12] Karthik Valmeekam, Alberto Olmo, Sarath Sreedharan et Subbarao Kambhampati. « Sur les capacités de planification des grands modèles de langage : une enquête critique. » *Avances dans les systèmes de traitement de l'information neuronale 36 (NeurIPS)*, 2023. URL : <https://arxiv.org/abs/2305.15771>[13] Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Daniel Ho et al. "Faites ce que je peux, pas ce que je dis : ancrer le langage dans les moyens robotiques." *Conférence sur l'apprentissage des robots (CoRL)*, PMLR 205, 2022. URL : <https://proceedings.mlr.press/v205/ahn23a.html>

[14] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang et Weizhu Chen. "LoRA : Adaptation de bas rang de grands modèles de langage." *Conférence internationale sur les représentations de l'apprentissage (ICLR)*, 2022. URL : <https://openreview.net/forum?id=nZeVKeeFYf9>[15] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman et Luke Zettlemoyer. « QLoRA : réglage fin efficace des LLM quantifiés. » *Avances in Neural Information Processing Systems 36 (NeurIPS)*, 2023. URL : <https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html>

[16] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning et Chelsea Finn. "Optimisation directe des préférences : votre modèle linguistique est secrètement un modèle de récompense." *Avances in Neural Information Processing Systems 36 (NeurIPS)*, 2023. URL : <https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html>[17] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi et Hannaneh Hajishirzi. « Auto-instruction : aligner les modèles de langage avec des instructions auto-générées. » *Réunion annuelle de l'Association for Computational Linguistics (ACL)*, 2023. URL : <https://aclanthology.org/2023.acl-long.754/>

[18] Équipe Qwen. «Rapport technique Qwen2.5». arXiv :2412.15115, 2024. URL : <https://arxiv.org/abs/2412.15115>

[19] DeepSeek-AI. "DeepSeek-R1 : Inciter la capacité de raisonnement dans les LLM via l'apprentissage par renforcement." arXiv :2501.12948, 2025. URL : <https://arxiv.org/abs/2501.12948>

[20] Bo Liu, Yuqian Jiang, Xiaohan Zhang, Qiang Liu, Shiqi Zhang, Joydeep Biswas et Peter Stone. "Lang2LTL : traduction des commandes en langage naturel en spécifications temporelles avec de grands modèles de langage." *Conférence sur l'apprentissage des robots (CoRL)*, PMLR 229, 2023. URL : <https://proceedings.mlr.press/v229/liu23d.html>[21] Behrad Rabiei, Mahesh Kumar AR, Zhirui Dai, Surya LSR Pilla, Qiyue Dong et Nikolay Atanasov. « LTLCodeGen : génération de code de logique temporelle syntaxiquement correcte pour la planification des tâches du robot. » arXiv :2503.07902, 2025. URL : <https://arxiv.org/abs/2503.07902>

[22] Jun Wang, David Smith Sundarsingh, Jyotirmoy V. Deshmukh et Yiannis Kantaros. "ConformalNL2LTL : traduction d'instructions en langage naturel en formules logiques temporelles avec des garanties d'exactitude conforme." arXiv :2504.21022, 2025. URL : <https://arxiv.org/abs/2504.21022>

[23] Shubo Liu, Hongsheng Zhang, Yuankai Qi, Peng Wang, Yanning Zhang et Qi Wu. "AerialVLN : navigation visuelle et linguistique pour les drones." *Conférence internationale IEEE/CVF sur la vision par ordinateur (ICCV)*, 2023, pp. URL : <https://openaccess.thecvf.com/content/ICCV2023/html/Liu_AerialVLN_Vision-and-Language_Navigation_for_UAVs_ICCV_2023_paper.html>[24] Xiangyu Wang, Donglin Yang, Ziqin Wang, Hohin Kwan, Jinyu Chen, Wenjun Wu, Hongsheng Li, Yue Liao et Si Liu. "Vers une navigation réaliste en langage vision pour drones : plate-forme, référence et méthodologie." *Conférence internationale sur les représentations de l'apprentissage (ICLR)*, 2025. URL : <https://openreview.net/forum?id=rUvCIvI4eB>

[25] Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Jens Beisswenger, Ping Luo, Andreas Geiger et Hongyang Li. « DriveLM : Conduire avec des réponses visuelles aux questions sous forme de graphique. » arXiv :2312.14150, 2023. URL : <https://arxiv.org/abs/2312.14150>

[26] Hao Shao, Yuxuan Hu, Letian Wang, Steven L. Waslander, Yu Liu et Hongsheng Li. "LMDrive : conduite de bout en bout en boucle fermée avec de grands modèles de langage." *Conférence IEEE/CVF sur la vision par ordinateur et la reconnaissance de formes (CVPR)*, 2024. URL : <https://arxiv.org/abs/2312.07488>[27] Xiaoyu Tian, ​​Junru Gu, Bailin Li, Yicheng Liu, Yang Wang, Zhiyong Zhao, Kun Zhan, Peng Jia, Xianpeng Lang et Hang Zhao. « DriveVLM : la convergence de la conduite autonome et des modèles de langage à grande vision. » arXiv :2402.12289, 2024. URL : <https://arxiv.org/abs/2402.12289>

[28] Yunsheng Ma, Can Cui, Xu Cao, Wenqian Ye, Peiran Liu, Juanwu Lu, Amr Abdelraouf, Rohit Gupta, Kyungtae Han, Aniket Bera, James M. Rehg et Ziran Wang. "LaMPilot : un ensemble de données de référence ouvert pour la conduite autonome avec des programmes de modèle de langage." *Conférence IEEE/CVF sur la vision par ordinateur et la reconnaissance de formes (CVPR)*, 2024, pp. 15141-15151. URL : <https://openaccess.thecvf.com/content/CVPR2024/html/Ma_LaMPilot_An_Open_Benchmark_Dataset_for_Autonomous_Driving_with_Language_CVPR_2024_paper.html>[29] Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch et Pete Florence. « PaLM-E : un modèle de langage multimodal incorporé. » *Conférence internationale sur l'apprentissage automatique (ICML)*, PMLR 202, 2023. URL : <https://proceedings.mlr.press/v202/driess23a.html>

[30] Anthony Brohan, Noah Brown, le juge Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence et d'autres. « RT-2 : Les modèles Vision-Langage-Action transfèrent les connaissances Web vers le contrôle robotique. » arXiv :2307.15818, 2023. URL : <https://arxiv.org/abs/2307.15818>[31] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang et Chelsea Finn. "OpenVLA : un modèle vision-langage-action open source." arXiv :2406.09246, 2024. URL : <https://arxiv.org/abs/2406.09246>

[32] Shital Shah, Debadeepta Dey, Chris Lovett et Ashish Kapoor. "AirSim : simulation visuelle et physique haute fidélité pour les véhicules autonomes." *Robotique de terrain et de service*, Springer Proceedings in Advanced Robotics, 2017 ; arXiv : 1705.05065. URL : <https://arxiv.org/abs/1705.05065>

[33] Yunlong Song, Selim Naji, Elia Kaufmann, Antonio Loquercio et Davide Scaramuzza. « Flightmare : un simulateur de quadrirotor flexible. » *Conférence sur l'apprentissage des robots (CoRL)*, PMLR 155, 2021. URL : <https://proceedings.mlr.press/v155/song21a.html>[34] AAAI. « Piste technique principale AAAI-26 : appel à communications. » URL : <https://aaai.org/conference/aaai/aaai-26/main-technical-track-call/>

[35] IJCAI-ECAI 2026. « Appel à communications – Volet spécial IA et robotique ». URL : <https://2026.ijcai.org/ijcai-ecai-2026-call-for-papers-ai-and-robotics/>

[36] Société des systèmes de transport intelligents IEEE. "Transactions IEEE sur les systèmes de transport intelligents (T-ITS) : portée." URL : <https://ieee-itss.org/pub/t-its/>

[37] Société des systèmes de transport intelligents IEEE. «Transactions IEEE sur les véhicules intelligents». URL : <https://ieee-itss.org/pub/t-iv/>

[38] AAMAS 2026. « Appel à communications – piste principale ». URL : <https://cyprusconferences.org/aamas2026/call-for-papers-main-track/>

---

## Annexe : plan de promotion sur 12 mois

### Mois 1-2 : Geler les problèmes et les interfaces du G1

- Gelez le titre, le résumé et trois contributions de CloudBrain-Agent.
- Définir LowAltitudeIR v0.1.
- Définir l'API des outils : espace aérien, ordonnanceur, planificateur, vérificateur, simulateur, risque.
- Construire un pipeline de vérification pour 100 à 200 échantillons de tâches à petite échelle.### Mois 3-4 : Création de CloudBrain-Bench

- Générez plus de 1000 missions de trafic à basse altitude.
- Couvre la planification normale, la distribution d'urgence, l'évitement des zones d'exclusion aérienne, les goulots d'étranglement en matière de recharge, la congestion des couloirs et les tâches insatisfaisantes.
- Marque or LowAltitudeIR, trace outil or, décision attendue.

### Mois 5-6 : Mise en œuvre des références G1

- LLM direct.
-Invite ReAct.
- Utilisation de l'outil sans vérificateur.
- Orchestration de style TrafficGPT.
- LLM+P.
- VERA-UAV uniquement.

### Mois 7-8 : Implémentation complète de CloudBrain-Agent

- Ajouter un schéma d'outil typé.
- Ajouter les commentaires du vérificateur.
- Ajout d'un test de stress sur simulateur.
-Ajouter une mémoire de sécurité et une boucle de réparation.

### Mois 9-10 : Expérience principale

- Réussite de l'exécution de la tâche, précision des appels d'outil, violation de la sécurité, réussite de la réparation, latence.
- Généralisation de la gestion de villes invisibles, de missions invisibles et de scènes dangereuses.
- Faire l'ablation : pas d'IR, pas de vérificateur, pas de simulateur, pas de mémoire, pas de réparation.

### Mois 11 : Pré-expérience de réglage fin du G2

- Collectez les traces de l'outil G1.
- LoRA affinant Qwen/DeepSeek.
- Comparez la base, la SFT et la DPO.
- Déterminer s'il suffit de former G2.

### Mois 12 : Première ébauche AAAI/IJCAI

-Rédiger des articles de conférence G1.
- L'annexe contient le schéma LowAltitudeIR, la définition des outils et les règles de génération de données.
- S'assurer que la liste de contrôle de reproductibilité, le code, les données et la préparation expérimentale des semences sont complets.