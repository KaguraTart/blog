---
title: "Paper B Planning v1 : Planification hiérarchique à trois niveaux de centaines de drones pour TR-C"
description: "Déterminez si l'article B est plus adapté à la partie C du TR et planifiez le contexte, les méthodes associées, la définition du problème, l'itinéraire de l'algorithme, les données expérimentales, les conclusions attendues, les points d'innovation et le plan de promotion."
pubDate: 2026-05-19
updatedDate: 2026-05-23
tags: ["Papier B", "TR-C", "T-ITS", "drone", "UAM", "planification hiérarchique", "théorie des files d'attente", "Liapounov", "transport multimodal"]
category: Tech
---

# Paper B Planning v1 : Planification hiérarchique à trois niveaux de centaines de drones pour TR-C

> Conclusion : **Le papier B est plus adapté à l'investissement principal Transportation Research Part C : Emerging Technologies, IEEE T-ITS comme alternative ou changement d'orientation d'investissement. **
> La raison principale n'est pas que le TR-C est « meilleur », mais que l'essence du problème de l'article B est l'exploitation du système de transport : sous les contraintes d'une demande dynamique, d'une capacité limitée de vertiport/chargement/couloir et d'un transport multimodal, laissez une flotte de drones d'une centaine de niveaux assurer les tâches de logistique urbaine/d'urgence de manière stable, efficace et sûre.

---

## 1. Contexte et jugement de soumission

Les préoccupations du document B peuvent être résumées comme suit :

> Dans un scénario économique urbain à basse altitude, comment planifier une flotte de drones de 100 niveaux pour maintenir la stabilité de la file d'attente des tâches à long terme et minimiser les retards, la consommation d'énergie, la congestion de l'espace aérien et les coûts d'exploitation dans des conditions d'ordres dynamiques, de points de décollage et d'atterrissage limités, de ressources de recharge limitées, de contraintes de capacité des couloirs de basse altitude et de coordination des transports terrestres ?

Il ne s’agit pas d’une planification de chemin sur une seule machine, ni d’un simple évitement de collision multi-agents. Le véritable objet de recherche est le **système de services de transport** :

- Côté demande : les commandes arrivent de manière aléatoire et il existe des différences en termes de délais, de priorités, de points de départ et d'arrivée, ainsi que de types de fret/d'urgence.
- Côté offre : puissance du drone, charge, emplacement actuel, état de maintenance et évolution de l'espace aérien disponible au fil du temps.
- Côté infrastructures : capacité limitée pour les vertiports, les bornes de recharge, les couloirs de basse altitude, les points de transfert et les véhicules terrestres.
- Côté système : il est nécessaire d'optimiser à la fois le débit, les délais, l'arriéré de files d'attente, l'utilisation des ressources, l'énergie et la sécurité.Il est donc plus raisonnable d’investir principalement dans le TR-C. La portée officielle de TR-C met clairement l'accent sur l'impact des technologies émergentes sur la planification, la conception, l'exploitation, le contrôle et la maintenance des systèmes de transport, et explique que le noyau intellectuel de la revue se situe du côté des transports, plutôt que de la technologie individuelle elle-même ; il accueille également favorablement l'approche d'intégration de la recherche opérationnelle, des systèmes de contrôle, des réseaux complexes, de l'informatique et de l'IA, et accorde une attention particulière au transport multimodal/intermodal, au transport à la demande, aux ITS, à la logistique, à l'aviation, à la gestion des ressources et aux ensembles de données ouvertes [1]. Ces mots-clés couvrent presque exactement l’article B.

T-ITS est également disponible comme alternative. Le champ d'application des T-ITS couvre la détection, les communications, les contrôles, la planification, la conception, la mise en œuvre, l'IA, les méthodes formelles, les systèmes multi-agents et le transport multimodal dans les systèmes de transport modernes [2]. Mais les STI-T nécessiteront probablement une approche de « mise en œuvre de technologies de systèmes de transport intelligents », comme la communication, la détection, le contrôle, l’architecture de déploiement ou la boucle fermée d’un système intelligent. Si l’article B met finalement l’accent sur la planification en ligne réglementée par Lyapunov, le contrôle GNN/MARL et la mise en œuvre de systèmes en temps réel, vous pouvez vous tourner vers T-ITS ; s’il met l’accent sur la capacité de transport, la stabilité des files d’attente, les goulots d’étranglement des infrastructures et la valeur des systèmes logistiques multimodaux, vous devriez voter pour le TR-C.

**Recommandations actuelles : TR-C en premier, sauvegarde T-ITS. **

### 1.1 2026-05-22 Calibrage d'écriture : le papier B doit être un document d'exploitation du système de transport

L'article B est le plus approprié pour absorber la logique du « les journaux de trafic ne sont pas seulement une question d'algorithmes ». Il ne peut pas être écrit comme « Nous proposons un nouvel algorithme de planification de drones », mais doit être écrit comme suit :> Pour les services urbains de logistique/d'urgence à basse altitude, comment une flotte de drones d'une centaine de niveaux peut-elle maintenir la stabilité de la flotte, réduire les retards, contrôler les risques de sécurité et identifier les goulots d'étranglement du système face à une demande dynamique, une capacité limitée de vertiport/charge/couloir et des contraintes de transport multimodal ?

Cette ligne principale détermine la manière dont le texte intégral est rédigé :

| modules | ne peut pas simplement être écrit comme | La version TR-C doit être écrite comme |
|------|------------|------------------|
| Contexte | La planification des drones est difficile | Problèmes de contrôle opérationnel pour les services de transport à basse altitude dans des conditions de pointe de demande, de capacité des infrastructures et de contraintes d'isolement de sécurité |
| écart | Les algorithmes existants ne sont pas assez bons | Les recherches existantes portent sur des problèmes ponctuels de routage/conception de réseau/allocation de ressources et manquent de garanties de boucle fermée et de stabilité pour des centaines de systèmes en ligne au niveau rack |
| Méthode | Nouvel algorithme de planification hiérarchique | Cadre de contrôle des opérations unifié de la file d'attente de macro-demandes, des ressources d'espace aérien/de décollage et d'atterrissage/de recharge méso et des contraintes de micro-énergie/sécurité |
| Expérience | Récompense ou taux de réussite plus élevé | Amélioration systématique des délais, du débit, de l'arriéré de files d'attente, du non-respect des délais, de l'utilisation des ressources, de l'énergie, du risque de conflit |
| Conclusion | La méthode est meilleure que la ligne de base | Sous quelle intensité de demande le système est stable, quelle ressource devient le goulot d'étranglement en premier, quand un repli multimodal est nécessaire et si une limitation stratégique du courant est nécessaire |

Par conséquent, l'expérience de l'article B répond à la question du système plutôt que de simplement prouver que le score du modèle est plus élevé :- ** Limite de capacité ** : en cas de demande faible/moyenne/crête/choc, quand le système entre-t-il dans la zone instable ?
- **Attribution des goulots d'étranglement** : le retard provient-il principalement du vertiport, de la tarification, de la congestion des couloirs ou du repositionnement de la flotte ?
- **Valeur multimodale** : Quand les drones seuls ne suffisent-ils pas ? Comment le repli au sol réduit-il les violations des délais ?
- **Correspondance théorique** : Le compromis retard/coût de la dérive de Lyapunov plus pénalité peut-il être observé dans les expériences ?
- **Inspiration de gestion** : Si vous n'ajoutez qu'une seule ressource, devriez-vous ajouter un drone, un socle de chargement, un emplacement de vertiport ou une capacité de couloir ?

### 1.2 2026-05-23 Organisation : version de soumission minimale et limites

La version minimale à soumettre du papier B doit être un **document d'exploitation du système de transport TR-C**, et non un mélange de « planificateur + MARL + simulation de l'espace aérien + démonstration de plate-forme à basse altitude ». La première édition doit résoudre le problème du système : comment la capacité limitée des drones, des vertiports, des bornes de recharge et des couloirs détermine conjointement les délais, le débit, la stabilité de la file d'attente et la fiabilité du service dans un contexte de demande dynamique.| Doit être complété | Reporté à la version étendue |
|--------------|--------------|
| référence de file d'attente UAM de ville synthétique | Simulation visuelle haute fidélité au niveau AirSim/UE |
| Planificateur en ligne réglementé par Lyapunov | Déploiement en vol réel ou boucle fermée matérielle |
| Évolutivité du drone 20/50/100/200 | Répartiteur LLM comme algorithme principal |
| analyse des goulots d'étranglement des vertiports, des recharges et des corridors | Protocole de communication complet et simulation de couche liaison |
| FCFS, gourmand, MILP roulant, ALNS, contre-pression, lignes de base MARL/GNN | Évaluation de la politique multi-villes et analyse économique complète |
| stabilité, compromis coût-délai, non-respect des délais, durée d'exécution | accès au système de commandes commerciales réelles |

La première version du package expérimental recommande de se limiter à cinq livrables :1. **Générateur de référence** : Générez des zones urbaines, un vertiport, une borne de recharge, un couloir, un flux de demande, un délai, un repli au sol et une demande de choc.
2. **Modèle système** : génère des journaux expérimentaux reproductibles de la file d'attente de demande, de la file d'attente de vertiport, de la file d'attente de facturation, de la file d'attente virtuelle de couloir et de la file d'attente virtuelle d'échéance.
3. **Noyau H-LyraUAV** : met en œuvre la prise de décision de dérive plus pénalité. Le module d'apprentissage fournit uniquement des estimations de la demande/du temps de service/des risques et n'est pas utilisé comme source de stabilité.
4. **Suite de référence** : chaque référence utilise le même ensemble de traces de demande, de paramètres de capacité, de flotte de drones et de graines aléatoires.
5. **Package de résultats TR-C** : le tableau principal rapporte le retard, le 95e retard, le non-respect des délais, le débit, le retard, l'utilisation des ressources, l'énergie, le proxy de conflit, le temps d'exécution ; le tableau en annexe rend compte de l’attribution et de la sensibilité des goulots d’étranglement.

Cette frontière peut rendre l'histoire du système de Paper B cohérente avec ses responsabilités expérimentales : prouver d'abord que « des systèmes de logistique/service d'urgence à basse altitude à basse altitude peuvent fonctionner de manière stable en ligne » avant de parler de cartes réelles, d'ordres réels ou d'agents plus complexes.

---

## 2. Généalogie des méthodes actuelles

L'article B doit intégrer les méthodes pertinentes dans le pedigree de l'ingénierie des transports, plutôt que de simplement parler du type UAV/MARL.| Ligne de méthode | Méthode représentative | Inspiration pour le papier B | Limites |
|--------|----------|---------|------|
| OU traditionnelle | MILP, réseau espace-temps, conception de réseau, ALNS, horizon glissant | Convient pour exprimer les contraintes de capacité, de fenêtre temporelle, de chemin, de synchronisation et d'infrastructure | La planification en ligne à grande échelle est difficile à résoudre en temps réel |
| UAM/UTM | planification de vertiports, planification avec contraintes de capacité, détection et résolution de conflits | Fournit des perspectives en matière de capacité, de conflits dans l'espace aérien et de gestion des corridors | La plupart sont monocouches, monomodes ou de petite et moyenne taille |
| Logistique multimodale | camion-drone, UAV-UGV, transfert sol-air, covoiturage-UAM | Prouver que les drones doivent être intégrés au système de transport urbain au lieu de voler de manière isolée | Routage/conception de réseau principalement hors ligne, manque de stabilité de la file d'attente en ligne |
| Planification de l'apprentissage | MARL, GNN, apprentissage sécurisé, prévision de la demande | Évolutif jusqu'à des centaines de racks, adapté aux besoins dynamiques et aux états de grande dimension | Manque de garantie de stabilité explicable, les évaluateurs remettront en question la sécurité |
| Théorie des files d'attente et Lyapunov | réseau de files d'attente ouvert, contre-pression, dérive plus pénalité | Peut prouver la stabilité du carnet de commandes et le compromis coût-retard | Doit être combiné avec les contraintes réelles d'énergie, de capacité et de trajectoire du drone |Les articles TR-C existants ont couvert de nombreuses « capacités ponctuelles » : gestion du trafic de colis d'UAV à basse altitude et allocation des ressources [3], équité et efficacité opérationnelle centrées sur les passagers UAM [4], conception de réseau de stations de recharge relais [5], routage de synchronisation fiable camion-drone [6], conception de réseau de livraison multi-voyages UAV-UGV [7] et planification dynamique de covoiturage UAM [8]. L'opportunité pour Paper B réside dans la convergence de ces capacités dans un **système de planification hiérarchique en ligne à cent niveaux**.

---

## 3. Articles actuels et littérature citable

### 3.1 Documentation sur le lieu et le cadrage| Numéro | Littérature/source | Informations de base | Rôle de positionnement pour l'article B |
|------|-----------|--------------|----------------------------------|
| [1] | Objectifs et portée officiels du TR-C | noyau intellectuel du côté des transports; focus sur l'exploitation, le contrôle, le multimodal, la logistique, l'aviation, les jeux de données ouverts | Soutenir l'investissement principal TR-C |
| [2] | Portée officielle IEEE T-ITS | Détection ITS, communications, contrôles, planification, IA, systèmes multi-agents | Prise en charge des alternatives T-ITS |
| [15] | Apprentissage automatique pour les ITS assistés par drones, T-ITS 2024 | L'UAV peut servir à la surveillance du trafic, aux interventions d'urgence et à l'inspection des infrastructures d'ITS | Prise en charge du cadrage alternatif T-ITS |
| [18] | Planification de trajectoire 4D pour les équipes de drones, T-ITS 2024 | Les équipes de drones ont été publiées dans ITS/T-ITS | Expliquez que les T-ITS peuvent être investis, mais qu'il doit s'agir d'un système plus intelligent |

### 3.2 Document de planification TR-C/UAM/UAV| Numéro | Littérature | Méthodes | Inspiration pour le papier B |
|------|------|------|------------------|
| [3] | Li, Hansen & Zou, TR-C 2022 | Gestion du trafic, conflit de chemin, allocation des ressources, mécanisme VCG de livraison de colis par drone à basse altitude | Déclarant directement que l'allocation des ressources de l'espace aérien à basse altitude est un sujet juridique du TR-C |
| [4] | Bennaceur, Delmas & Hamadi, TR-C 2022 | UAM centrée sur les passagers, équité et efficacité opérationnelle | Soutenir la qualité du service, l'équité et les paramètres passagers/fret |
| [5] | Pinto & Lagorio, TR-C 2022 | conception d'un réseau de drones avec bornes de recharge intermédiaires | Soutenir l'infrastructure de recharge dans la formulation |
| [6] | Xing, Guo et Tong, TR-C 2024 | routage fiable camion-drone avec synchronisation dynamique | Prise en charge de la synchronisation multimodale et du temps de trajet incertain |
| [7] | Zhou, Zeng et Yang, TR-C 2025 | Conception d'un réseau de livraison multi-voyages UAV-UGV avec délais de lancement | Prise en charge du réseau de livraison multi-voyages UAV + UGV |
| [8] | Li, Zhang, Xiao et Li, TR-C 2025 | Planification dynamique de covoiturage UAM et mobilité multimodale à la demande | Soutenir l'architecture de service multimodal air-sol |
| [9] | Wei, Nilsson et Coogan, arXiv 2021 | planification UAM à capacité limitée avech temps de trajet incertain et capacité d'atterrissage limitée | prendre en charge la formulation de planification avec des capacités limitées |
| [10] | Murthy et al., EPTCS/arXiv 2022 | apprentissage sécurisé pour la planification UAM avec des délais fermes/souples | Prise en charge de la planification en ligne sécurisée |
| [11] | Planification FCFS des vertiports de la NASA, 2020 | capacité et débit du vertiport sous FCFS | comme FCFS et référence de capacité de file d'attente |
| [16] | Liu, Liu et Huang, arXiv 2024 | middleware de gestion de planification de livraison de drones en temps réel | Prise en charge du système d'exécution réel et de la collaboration UAV/AGV/personnel au sol |### 3.3 Théorie des files d'attente, Lyapunov et base de la stabilité du système

| Numéro | Littérature | Contribution de base | Contribution à l'article B |
|------|------|----------|---------|
| [12] | Grippa et al., Robots autonomes 2019 | attribution et dimensionnement des tâches de livraison par drone ; utiliser la théorie des files d'attente pour analyser la politique de stabilité et de charge de travail | Prise en charge du modèle de file d'attente de livraison d'UAV |
| [13] | Neely, 2010 | optimisation du réseau stochastique et dérive de Lyapunov plus pénalité | Prise en charge du coût de O(1/V)$ et du compromis de retard de $O(V)$ |
| [14] | Tassiulas et Éphrémides, IEEE TAC 1992 | systèmes de files d'attente contraints et planification à débit optimal | supportant la contre-pression/stabilité tradition théorique |
| [17] | Placement du Vertiport avec dimensionnement des véhicules et file d'attente, 2023 | file d'attente en réseau ouvert pour les infrastructures vertiportaires et les tarifs de service | Prise en charge de la modélisation de la file d'attente de vertiport/de la file d'attente de chargement |

**Jugement de la littérature :** La littérature existante a pleinement prouvé que « UAV/UAM + systèmes de transport + planification + multimodal + file d'attente » est un sujet légitime pour les TR-C/T-ITS. L'article B ne peut plus être rédigé sous la forme « Algorithme de planification MARL pour des centaines d'UAV », mais doit être rédigé sous la forme « Contrôle du fonctionnement et garantie de stabilité du système de trafic à basse altitude ».

---

## 4. Problème actuelIl existe quatre principales lacunes dans les travaux existants.

1. ** Absence d'une boucle fermée de planification en ligne à trois niveaux au niveau d'une centaine d'étagères. **
   TR-C dispose déjà d'une allocation de ressources dans l'espace aérien à basse altitude, d'un routage camion-drone, d'un covoiturage UAM et d'une conception de réseau UAV-UGV [3,6,7,8], mais la plupart de ces travaux traitent d'une certaine couche de routage, de conception de réseau, d'allocation de ressources ou de covoiturage, manquant d'un cadre en ligne unifié allant de la file d'attente de demande macro aux ressources de corridor/vertiport méso en passant par l'énergie/sécurité des micro UAV.

2. **Manque de stabilité de la file d'attente/garantie de service. **
   La planification et les heuristiques apprises peuvent améliorer les performances empiriques, mais les évaluateurs du TR-C remettent en question l'opérabilité du système s'ils ne peuvent pas déterminer si les files d'attente sont stables en cas de demande de pointe. L'optimisation Lyapunov de Neely et la planification des files d'attente contraintes de Tassiulas-Ephremides fournissent des fondements théoriques [13,14], mais n'ont pas été systématiquement utilisées pour la planification multimodale de centaines de drones à basse altitude.

3. **Manque de contrôle de la flotte de drones du point de vue du transport multimodal. **
   Les articles sur les camions-drones, les UAV-UGV et le covoiturage-UAM ont prouvé que l'intégration sol-air est la direction dominante [6, 7, 8], mais la plupart des recherches existantes portent sur la conception d'itinéraires/réseaux hors ligne. L'article B devrait traiter le mode terrestre comme une solution de repli en ligne et un tampon de capacité : lorsque le couloir de basse altitude ou la file d'attente de chargement est encombré, la tâche peut être transférée à l'UGV/camion/courrier terrestre.4. **Manque de référence expérimentale. **
   La portée du TR-C met un accent particulier sur la science ouverte et les ensembles de données à grande échelle [1]. Si le papier B effectue uniquement une simulation interne et ne publie pas de schéma de référence synthétique, de demande de DO, de capacité de couloir et de semences reproductibles, cela affaiblira son pouvoir de persuasion.

---

## 5. Notre approche : H-LyraUAV

Le nom de la méthode est provisoirement décidé :

**H-LyraUAV : planification hiérarchique des drones régulés par Lyapunov pour la logistique urbaine multimodale**

Où H signifie hiérarchique et Lyra signifie routage et affectation régulés par Lyapunov.

### 5.1 Architecture à trois niveaux

```text
Dynamic urban logistics / emergency demand
        ↓
Macro layer: regional demand queues + fleet repositioning
        ↓
Meso layer: corridor / vertiport / charging resource scheduling
        ↓
Micro layer: UAV energy, safety separation, local conflict avoidance
        ↓
Multimodal execution: UAV-only / ground-only / UAV-ground mixed mode
```

| Hiérarchie | Échelle de temps | Décision | État de base | Sortie |
|------|----------|------|---------|------|
| Niveau macro | 1-5 minutes | Partitionnement des tâches, repositionnement du drone, répartition des modes | File d'attente de la demande régionale, distribution d'énergie, prévision de la demande OD | Objectif d'expédition régional |
| Mésocouche | 5-30 s | emplacement de vertiport, itinéraire de couloir, emplacement de recharge | file d'attente au décollage et à l'atterrissage, congestion des couloirs, file d'attente pour la recharge | calendrier exécutable |
| Couche microscopique | 0,1-5 s | Vitesse, altitude, évitement local, retour d'urgence | Drones voisins, obstacles, puissance restante | Correction de trajectoire sûre |

### 5.2 Mécanisme de base

La clé du H-LyraUAV n'est pas « la planification de bout en bout avec un grand modèle », mais de limiter le module d'apprentissage à la prédiction et à la stratification, et de renforcer la stabilité du contrôle de file d'attente Lyapunov :- **Modèle de file d'attente** : La demande, le vertiport, la tarification et le corridor sont représentés par des files d'attente réelles ou virtuelles.
- **Dérive de Lyapunov plus pénalité** : sélectionnez l'affectation/le mode/l'itinéraire/la facturation dans chaque fenêtre horaire pour minimiser la somme pondérée de la dérive de la file d'attente et des coûts d'exploitation.
- **Prévision assistée par apprentissage** : le modèle GNN/temporel prédit la demande future de DO, le temps de service, le risque de couloir et le temps de trajet au sol, mais n'est pas utilisé comme source de preuve de stabilité.
- **Repli multimodal** : lorsque le drone uniquement entraîne une explosion de la file d'attente ou une augmentation du risque de délai, le système active automatiquement le mode UGV/camion/courrier terrestre ou mixte.

---

## 6. Formulation du problème

### 6.1 Collections et états

Supposons que l'ensemble d'UAV soit $\mathcal{U}$, l'ensemble de missions dynamiques soit $\mathcal{R}(t)$, l'ensemble de vertiports soit $\mathcal{V}$, l'ensemble de couloirs à basse altitude soit $\mathcal{E}$ et l'ensemble de modes de transport terrestre soit $\mathcal{G}$.

L'état de chaque drone $u\in\mathcal{U}$ au temps $t$ est :

$$
s_u(t)=(l_u(t), b_u(t), a_u(t), \kappa_u(t)),
$$

Où $l_u(t)$ est la position, $b_u(t)$ est la puissance, $a_u(t)$ est l'état disponible, $\kappa_u(t)$ est la capacité de charge/tâche.

Chaque tâche $r\in\mathcal{R}(t)$ contient :

$$
r=(o_r,d_r,\omega_r,\delta_r,\pi_r,\eta_r),
$$

Parmi eux, $o_r,d_r$ est le point de départ et d'arrivée, $\omega_r$ est le type cargo/passager/urgence, $\delta_r$ est la date limite, $\pi_r$ est la priorité et $\eta_r$ est l'ensemble des modes de transport acceptables.### 6.2 Définition de la file d'attente

Le système gère les files d'attente réelles et virtuelles suivantes :

| File d'attente | Signification |
|------|------|
| $Q_i(t)$ | File d'attente de demandes non desservies pour la zone $i$ |
| $B_v(t)$ | File d'attente/décollage du Vertiport $v$ |
| $C_v(t)$ | file d'attente de chargement du vertiport $v$ |
| $Z_e(t)$ | Encombrement/intervalle de sécurité de la file d'attente virtuelle $e$ du couloir |
| $D_i(t)$ | file d'attente virtuelle de non-respect du délai dans la zone $i$ |

Par exemple, la file d’attente de demande régionale peut s’écrire :

$$
Q_i(t+1)=\max[Q_i(t)-\mu_i(t),0]+A_i(t),
$$

Où $A_i(t)$ est la demande nouvellement arrivée, $\mu_i(t)$ est le nombre de demandes qui terminent le service dans la fenêtre de temps.

### 6.3 Variables de décision

Chaque cycle de planification doit décider :

| Prise de décision | Symboles | Signification |
|------|------|------|
| mission | $x_{u,r}(t)$ | Si le drone $u$ sert la tâche $r$ |
| choix des modes | $m_r(t)$ | Mode drone uniquement, sol uniquement ou mixte |
| heure de départ | $s_u(t)$ | heure de départ/départ/transfert |
| itinéraire / couloir | $p_u(t)$ | Sélectionnez un couloir de basse altitude ou un chemin au sol |
| décision de taxation | $c_u(t)$ | S'il faut charger et quel vertiport charger |

### 6.4 Objectifs d'optimisation

L’objectif à long terme est de minimiser le coût moyen du système :

$$
\min_{\pi}
\limsup_{T\à\infty}
\frac{1}{T}\sum_{t=0}^{T-1}
\mathbb{E}\gauche[
\alpha_1 W(t)+
\alpha_2 E(t)+
\alpha_3O(t)+
\alpha_4 S(t)+
\alpha_5M(t)
\droite],
$$Où $W(t)$ est le retard, $E(t)$ est la consommation d'énergie, $O(t)$ est le coût d'exploitation, $S(t)$ est la pénalité de sécurité/encombrement, $M(t)$ est la pénalité de transport multimodal.

### 6.5 Contraintes

Les contraintes incluent :

- stabilité de la file d'attente : toutes les files d'attente réelles et les files d'attente virtuelles critiques doivent être fortement stables.
- batterie : $b_u(t)$ n'est pas inférieur au seuil de retour sûr.
- Charge utile : Le poids de la cargaison de mission ne peut pas dépasser la capacité du drone ou du véhicule terrestre.
- fenêtre temporelle : les tâches hautement prioritaires doivent respecter le délai ou entrer dans la file d'attente virtuelle des délais.
- capacité du vertiport : La capacité d'aire de stationnement/parking/charge de chaque vertiport a une limite supérieure.
- Séparation des couloirs : Les intervalles temporels et spatiaux des drones dans un même couloir répondent aux exigences de sécurité.
- Faisabilité du transfert multimodal : le délai de transfert, l'emplacement et la capacité des drones et des UGV/camions/courriers terrestres sont réalisables.

### 6.6 Objectifs théoriques

Définir la fonction Lyapunov :

$$
L(\Thêta(t)) =
\frac{1}{2}\gauche(
\sum_i Q_i(t)^2+
\sum_v B_v(t)^2+
\sum_v C_v(t)^2+
\sum_e Z_e(t)^2+
\sum_i D_i(t)^2
\droite).
$$

Résolvez la dérive et la pénalité pour chaque fenêtre temporelle :

$$
\Delta(\Theta(t)) + V\cdot \mathbb{E}[Coût(t)\mid \Theta(t)].
$$

Conclusion de la théorie des attentes :

- H-LyraUAV peut maintenir la file d'attente stable si le taux d'arrivée se situe dans la région de capacité du système.
- Par rapport à la politique randomisée stationnaire optimale, le coût moyen à long terme atteint environ $O(1/V)$.
- Le backlog moyen est de $O(V)$, formant un compromis coût-délai interprétable [13,14].

---

## 7. Source de données expérimentale### 7.1 Expérience principale : benchmark de génération de programmes

L'expérience principale ne s'appuie pas sur des données de vol réelles d'UAV, mais construit un benchmark synthétique reproductible de mise en file d'attente UAM :

- Plan de la ville : grille de « 50x50 » à « 200x200 », comprenant les bâtiments, les zones d'exclusion aérienne, les couloirs, les vertiports, les bornes de recharge.
- Flux de demande : Poisson / Poisson non homogène / demande en rafale, supporte la pointe du matin, la pointe du soir, la demande de choc.
-Types de tâches : livraison de colis, livraison médicale, inspection, approvisionnement d'urgence.
- Flotte de drones : 20 / 50 / 100 / 200 unités, batteries hétérogènes, charge, vitesse, temps de charge.
- Infrastructure : 5 / 10 / 20 vertiports, différents pads, parking, capacité de recharge.
- Mode multimodal : mode mixte drone-sol uniquement, drone-sol uniquement.

### 7.2 Données réelles augmentées

Pour renforcer la conviction du TR-C, les expériences peuvent utiliser les données de trafic public comme proxy de la demande et du temps de trajet en mode terrestre :

| Data source | Purpose |
|--------|------|
| OpenStreetMap | Réseau routier, POI, densité de construction, vertiport/point de transfert candidat |
| NYC TLC Taxi Trip Data | Proxy de demande OD, profil de demande par période |
| Chicago Taxi Trips / Divvy / données de mobilité publique | Proxy de la demande de généralisation interurbaine |
| SUMO | Temps de trajet des véhicules terrestres, congestion, coûts de repli au sol |
| AirSim ou simulateur de drone léger | Vérification complémentaire de la micro-sécurité, du temps de vol et de la consommation d'énergie |Les conférences telles que l'AAAI ne peuvent effectuer que des benchmarks synthétiques ; mais le TR-C requiert une étude de cas de qualité, il est donc recommandé de mener au moins un cas urbain : San Francisco, New York ou Chicago. La gestion du trafic de colis de drones à basse altitude de Li et al. utilisant l'étude de cas de San Francisco [3] peut être utilisée comme objet d'alignement.

---

## 8. Conception expérimentale et comparaison

### 8.1 Baselines

| Baseline | Descriptif | Purpose |
|--------------|------|------|
| Planification des vertiports FCFS | Allouer les ressources de décollage et d'atterrissage par ordre d'arrivée [11] | Base de référence des opérations traditionnelles |
| Greedy nearest UAV | Le drone disponible le plus proche pour récupérer la tâche la plus proche | Simple online dispatch |
| MILP rolling horizon | optimisation du roulement à petite échelle | limite supérieure à petite échelle |
| ALNS / répartition heuristique | Se référer à la littérature sur le routage multimodal TR-C [7,8] | Strong OR heuristic |
| Contre-pression dans la file d'attente uniquement | Planification basée uniquement sur la différence de file d'attente | Ligne de base théoriquement stable |
| MARL / GNN dispatch | Allocation d'apprentissage, pas de files d'attente virtuelles Lyapunov | Learning baseline |
| H-LyraUAV full | Superposition à trois couches + Lyapunov + prédiction d'apprentissage + repli multimodal | Main method |

### 8.2 Metrics| Indicateur | Signification |
|------|------|
| Délai moyen | Délai moyen d'achèvement des tâches |
| Retard du 95e percentile | Qualité de service à longue traîne |
| Taux de non-respect des délais | Ratio des heures supplémentaires |
| Débit | Nombre de tâches réalisées par unité de temps |
| Arriéré de file d'attente | Demande, vertiport, tarification, longueur de file d'attente du couloir |
| Stabilité de la file d'attente | L'arriéré est-il limité dans le temps |
| Utilisation du vertiport | Taux d'utilisation des ressources de décollage et d'atterrissage/arrêt |
| Utilisation de la recharge | Taux d'utilisation des ressources de facturation |
| Taux de conflits dans l'espace aérien | taux de conflits dans les intervalles de sécurité du corridor |
| Énergie par livraison | Consommation d'énergie par commande |
| Succès du transfert sol-UAV | Taux de réussite du transfert multimodal |
| Durée d'exécution | Temps de planification en une seule étape |
| Contribution aux goulots d'étranglement | Dans quelle mesure le repositionnement du vertiport / de la recharge / du corridor / de la flotte contribue-t-il au retard |
| Marge de capacité | Quelle est la distance entre l'intensité de la demande actuelle et la zone d'instabilité du système |
| Équité des services | retarder l'écart pour différents domaines/tâches prioritaires afin d'empêcher l'optimisation des seuls domaines populaires |

Il n'est pas recommandé que le tableau principal de la version TR-C rapporte uniquement les classements des performances des algorithmes, mais fournisse également un **tableau de diagnostic du système** distinct : rapportant le retard moyen, le retard de 95 %, le non-respect des délais, les goulots d'étranglement majeurs et si le repli multimodal est déclenché sous différents multiplicateurs de demande. De cette façon, la conclusion peut arriver à « comment le système fonctionne » plutôt qu'à « quel modèle est le plus fort ».

### 8.3 Ablation| Ablation | Objectif |
|------|------|
| pas de files d'attente virtuelles Lyapunov | Vérifier la contribution du composant de stabilité |
| pas de repli multimodal | Vérifier la valeur du mode sol en tant que tampon de capacité |
| pas de décomposition hiérarchique | Vérifier l'évolutivité de la structure à trois niveaux |
| aucune prévision de la demande | Vérifier la contribution du module de prédiction d'apprentissage |
| pas de modélisation de file d'attente de chargement | Vérifier si le goulot d'étranglement de charge doit être modélisé explicitement |
| Mise à l'échelle du drone 20/50/100/200 | Vérification de centaines d'évolutivités au niveau du rack |

### 8.4 Conception de la scène

Exécutez au moins quatre types de scénarios de demande :

1. **Faible demande** : le système est légèrement chargé, ce qui vérifie que le H-LyraUAV ne sacrifie pas l'efficacité.
2. **Demande de pointe** : la demande est proche de la région de capacité, vérifiez la stabilité de la file d'attente.
3. **Demande de choc** : commandes d'urgence soudaines, vérification de la file d'attente virtuelle des délais et repli multimodal.
4. **Goulot d'étranglement de l'infrastructure** : les bornes de recharge ou les vertiports sont délibérément réduits pour vérifier les capacités d'identification des goulots d'étranglement des ressources.

Il est en outre recommandé d'ajouter deux types de généralisations :

5. **Généralisation à l'échelle** : la formation ou l'ajustement des paramètres est à 50/100 UAV et les tests à 200 UAV, ce qui indique que la structure hiérarchique n'est pas seulement efficace pour une échelle fixe.
6. **Généralisation de la topologie** : Le graphique de ville dérivé d'OSM a été mesuré à partir de la règle grid_city, indiquant que la conclusion n'est pas le résultat accidentel d'une carte de jouet.

---

## 9. Succès et innovation attendus

### 9.1 Succès attendu

Cette section est destinée aux attentes préalables à l'inscription et n'écrit pas les résultats expérimentaux réels.1. ** Maintenir la stabilité de la file d'attente en cas de demande de pointe. **
   On s'attend à ce que le H-LyraUAV maintienne la file d'attente de demande, la file d'attente de vertiport et la file d'attente de charge limitées à la demande de pointe, tandis que le gourmand/MARL uniquement est plus sujet à l'accumulation d'arriérés.

2. **Réduisez les retards et les non-respects des délais. **
   Comparé au FCFS et au gourmand, le H-LyraUAV devrait réduire le retard moyen, le retard du 95e percentile et le taux de non-respect des délais.

3. **Améliorez les performances et l'évolutivité en temps réel. **
   Par rapport à l’horizon mobile MILP, le H-LyraUAV devrait maintenir une prise de décision en ligne en une seconde ou une seconde dans des scénarios 100/200 UAV.

4. **Explication théorique réservée. **
   Par rapport au MARL/GNN uniquement, l'avantage du H-LyraUAV n'est pas seulement le score d'expérience, mais il peut expliquer le compromis coût-délai et la limite de stabilité.

5. **Afficher la valeur système du repli multimodal. **
   On s'attend à ce que le mode mixte drone-sol réduise les délais manqués et les retards dans les files d'attente dans les scénarios de goulot d'étranglement de recharge ou de congestion des couloirs.

### 9.2 Points d'innovation

1. **Document de planification du système de trafic à basse altitude basé sur le cadrage TR-C. **
   L’article B ne traite pas les drones comme des robots isolés, mais comme une flotte de centaines de drones faisant partie du système de services de transport urbain.

2. **Cadre de planification multimodale stable à trois couches sur cent niveaux. **
   Unifiez les macro-files d'attente de tâches, les ressources d'infrastructure méso et les micro-contraintes de sécurité/énergie dans un cadre en ligne.

3. **Répartition assistée par apprentissage réglementée par Lyapunov. **
   Le module d'apprentissage est utilisé pour prédire la demande et les coûts, et le module Lyapunov offre un compromis stabilité et coût-délai.4. **Mécanisme tampon de capacité de transport multimodal. **
   Utilisez l’UGV/camion/courrier terrestre comme solution de repli sous l’espace aérien/goulot d’étranglement de chargement au lieu de la ligne de base attachée.

5. ** Ouvrir le benchmark de file d'attente UAM synthétique. **
   Aligner les préférences de TR-C en matière de données ouvertes, de références reproductibles et de transférabilité [1].

---

## 10. Références

[1] Elsevier. « Recherche sur les transports, partie C : Technologies émergentes : objectifs et portée. » URL : <https://www.sciencedirect.com/journal/transportation-research-part-c-emerging-technologies>

[2] Société des systèmes de transport intelligents IEEE. "Transactions IEEE sur les systèmes de transport intelligents (T-ITS) : portée." URL : <https://ieee-itss.org/pub/t-its/>

[3] Ang Li, Mark Hansen et Bo Zou. "Gestion du trafic et allocation des ressources pour la livraison de colis par drone dans l'espace urbain à basse altitude." *Recherche sur les transports, partie C : technologies émergentes*, 143 : 103808, 2022. DOI : 10.1016/j.trc.2022.103808. URL : <https://www.sciencedirect.com/science/article/pii/S0968090X22002339>[4] Mehdi Bennaceur, Rémi Delmas et Youssef Hamadi. « Mobilité aérienne urbaine centrée sur les passagers : compromis en matière d'équité et d'efficacité opérationnelle. » *Recherche sur les transports, partie C : technologies émergentes*, 136 :103519, 2022. DOI : 10.1016/j.trc.2021.103519. URL : <https://www.sciencedirect.com/science/article/pii/S0968090X21005015>

[5] Roberto Pinto et Alexandra Lagorio. "Conception d'un réseau de livraison point à point basé sur des drones avec des stations de recharge intermédiaires." *Recherche sur les transports, partie C : technologies émergentes*, 135 :103506, 2022. DOI : 10.1016/j.trc.2021.103506. URL : <https://doi.org/10.1016/j.trc.2021.103506>[6] Jiahao Xing, Tong Guo et Lu (Carol) Tong. "Routage camion-drone fiable avec synchronisation dynamique : une approche de programmation réseau de grande dimension." *Recherche sur les transports, partie C : technologies émergentes*, 165 :104698, 2024. DOI : 10.1016/j.trc.2024.104698. URL : <https://www.sciencedirect.com/science/article/pii/S0968090X24002195>

[7] Bolong Zhou, Wenjia Zeng et Hai Yang. "Conception d'un réseau de livraison d'UAV-UGV multi-voyages avec délais de sortie." *Recherche sur les transports, partie C : technologies émergentes*, 181 : 105389, 2025. DOI : 10.1016/j.trc.2025.105389. URL : <https://doi.org/10.1016/j.trc.2025.105389>

[8] Shanghan Li, Tengfei Zhang, Yiyong Xiao et Daqing Li. « Covoiturage à la demande basé sur une programmation dynamique dans la mobilité aérienne urbaine. » *Recherche sur les transports, partie C : technologies émergentes*, 175 :105111, 2025. DOI : 10.1016/j.trc.2025.105111. URL : <https://www.sciencedirect.com/science/article/pii/S0968090X25001159>[9] Qinshuang Wei, Gustav Nilsson et Samuel Coogan. « Planification de la mobilité aérienne urbaine à capacité limitée. » arXiv :2107.02900, 2021. URL : <https://arxiv.org/abs/2107.02900>

[10] Surya Murthy, Natasha A. Neogi et Suda Bharadwaj. « Planification de la mobilité aérienne urbaine à l'aide d'un apprentissage sécurisé. » *Actes électroniques en informatique théorique*, 371 : 86-102, 2022 ; arXiv :2209.15457. DOI : 10.4204/EPTCS.371.7. URL : <https://arxiv.org/abs/2209.15457>

[11] Nelson M. Guerreiro, George E. Hagen, Jeffrey M. Maddalon et Ricky W. Butler. « Capacité et débit des vertiports de mobilité aérienne urbaine avec un algorithme de planification de vertiports premier arrivé, premier servi. » Serveur de rapports techniques de la NASA, Forum AIAA Aviation 2020, 2020. URL : <https://ntrs.nasa.gov/citations/20205001421>[12] Pasquale Grippa, Doris A. Behrens, Friederike Wall et Christian Bettstetter. « Systèmes de livraison par drone : affectation des tâches et dimensionnement. » *Robots autonomes*, 43 : 261-274, 2019. DOI : 10.1007/s10514-018-9768-8. URL : <https://link.springer.com/article/10.1007/s10514-018-9768-8>

[13] Michael J. Neely. *Optimisation stochastique des réseaux avec application aux systèmes de communication et de file d'attente.* Conférences de synthèse sur les réseaux de communication, Morgan & Claypool Publishers, 2010. DOI : 10.2200/S00271ED1V01Y201006CNT007. URL : <https://doi.org/10.2200/S00271ED1V01Y201006CNT007>

[14] Leandros Tassiulas et Anthony Ephremides. "Propriétés de stabilité des systèmes de file d'attente contraintes et politiques de planification pour un débit maximal dans les réseaux radio multi-sauts." *Transactions IEEE sur le contrôle automatique*, 37(12) :1936-1948, 1992. DOI : 10.1109/9.182479. URL : <https://doi.org/10.1109/9.182479>[15] Akbar Telikani, Arupa Sarkar, Bo Du et Jun Shen. "Apprentissage automatique pour les ITS assistés par UAV : une revue avec étude comparative." *Transactions IEEE sur les systèmes de transport intelligents*, 25(11):15388-15406, 2024. DOI : 10.1109/TITS.2024.3422039. URL : <https://ieeexplore.ieee.org/document/10622103/>

[16] Han Liu, Tian Liu et Kai Huang. "Un système en temps réel pour planifier et gérer la livraison de drones dans les zones urbaines." arXiv :2412.11590, 2024. URL : <https://arxiv.org/abs/2412.11590>

[17] José Escribano Macias, Carl Khalife, Joseph Slim et Panagiotis Angeloudis. "Un modèle intégré de placement de vertiport prenant en compte le dimensionnement des véhicules et la file d'attente : une étude de cas à Londres." *Journal of Air Transport Management*, 113:102486, 2023. DOI : 10.1016/j.jairtraman.2023.102486. URL : <https://www.sciencedirect.com/science/article/pii/S0969699723001291>[18] Blanca Lopez Palomino, Javier Muñoz Mendi, Fernando Quevedo Vallejo, Concepción Alicia Monje Micharet, Luis Santiago Garrido Bullon et Luis Enrique Moreno Lorente. « Planification de trajectoire 4D basée sur Fast Marching Square pour les équipes de drones. » *Transactions IEEE sur les systèmes de transport intelligents*, 25(6):5703-5717, 2024. DOI : 10.1109/TITS.2023.3336008. URL : <https://doi.org/10.1109/TITS.2023.3336008>

---

## Annexe : Plan d'exécution

### Semaine 1 : Geler le positionnement du papier et la formulation du problème

- Clarifier le principal investissement TR-C et le T-ITS alternatif.
- Geler le titre, la première ébauche du résumé, les questions principales et le diagramme d'architecture à trois niveaux.
- Compléter la définition des ensembles, des files d'attente, des décisions, des objectifs et des contraintes pour la formulation des problèmes.

### Semaine 2-3 : Compléter plus de 25 documents et la matrice de travail associée

- Documentation étendue TR-C / T-ITS / UAM / UAV / mise en file d'attente / Lyapunov.
- Matrice de travail liée aux résultats : problème, méthode, échelle, mode, limitation de chaque article.
- Identifier les différences entre le papier B et chaque type de travail.

### Semaines 4 à 6 : Implémentation du benchmark de file d'attente UAM synthétique- Implémenter une carte, un vertiport, un couloir, une borne de recharge et un générateur de demande OD.
- Prend en charge les UAV 20/50/100/200 et la demande faible/moyenne/crête/choc.
- Manifeste de sortie, graine, configuration de scénario pour garantir la reproductibilité.

### Semaines 7 à 9 : mise en œuvre des références

- Planification des vertiports FCFS.
- UAV gourmand le plus proche.
- Limite supérieure à petite échelle de l'horizon roulant MILP.
-ALNS/envoi heuristique.
- Contre-pression de file d'attente uniquement.
- Envoi MARL/GNN sans Lyapunov.

### Semaines 10 à 12 : Mise en œuvre du H-LyraUAV et de l'ablation

- Implémenter une affectation prenant en compte la file d'attente des macros.
- Implémenter la planification du corridor méso/vertiport/chargement.
- Implémenter une interface microscopique énergie/contrainte de sécurité.
- Mettre en œuvre des ablations sans Lyapunov, sans multimodal, sans hiérarchie, sans prévision de la demande et sans file d'attente de facturation.

### Semaines 13 à 15 : Expérience des coureurs

-Extensibilité d'UAV 20/50/100/200.
- Exécuter des scénarios de pointe/choc/goulot d'étranglement.
- Afficher le tableau principal : délai, non-respect des délais, débit, retard dans la file d'attente, utilisation des ressources, temps d'exécution.
- Graphiques clés de sortie : trajectoire de file d'attente, compromis coût-délai, courbe d'évolutivité, contribution de repli multimodale.

### Semaine 16 : Rédaction de la première version du TR-C- L'introduction commence par le problème d'exploitation du transport.
- Mettre l'architecture à trois couches, le théorème de Lyapunov et l'algorithme dans Méthode.
- Les expériences mettent l'accent sur les performances du système, l'utilisation des ressources et le benchmark ouvert.
- Discussion écrit sur l'économie à basse altitude, la capacité des vertiports, l'infrastructure de recharge et les implications logistiques multimodales.

### Stratégie de changement d'investissement T-ITS

Si le cadrage TR-C n'est pas assez fort, ou si les résultats expérimentaux montrent que la contribution de l'algorithme/contrôle est plus forte que la compréhension du transport du système, la version T-ITS est retenue :

- Abstract met davantage l'accent sur le système de transport intelligent, le contrôle en ligne et la planification assistée par l'IA.
- Introduction Ajouter la détection/communication/implémentation en temps réel.
- Les expériences augmentent le temps d'exécution, le délai de communication, l'exécution distribuée et la robustesse du contrôleur.
- La discussion réduit les implications politiques/opérationnelles et augmente le déploiement de systèmes intelligents et l'intégration des STI.