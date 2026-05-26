---
title: "Repenser le contrôle des feux de circulation : du timing fixe à l'intelligence adaptative"
description: "Une réflexion sur l'évolution du contrôle des feux de circulation — des détecteurs de boucle et plans fixes à l'apprentissage par renforcement et aux véhicules autonomes connectés."
pubDate: 2026-04-02
tags: ["Ingénierie du trafic", "Apprentissage par renforcement", "Contrôle adaptatif", "Ville intelligente"]
category: Tech
sourceHash: "ca62dbbd7b3acf68e773871e122fe3a9b1d895c9"
---

# Repenser le contrôle des feux de circulation : du timing fixe à l'intelligence adaptative

Les feux de circulation sont partout : nous les rencontrons des dizaines de fois par jour, généralement sans y penser. Mais si vous vous êtes déjà assis à un feu rouge à 2 heures du matin sans personne en vue, ou si vous vous êtes retrouvé dans une « vague verte » qui coule parfaitement d'une intersection à l'autre, vous avez déjà ressenti les conséquences de l'optimisation (ou non) des feux de circulation.

Après avoir passé des années à travailler avec des outils de simulation de trafic comme SUMO et CARLA, et à approfondir la recherche sur l'apprentissage par renforcement pour le contrôle des signaux, j'en suis venu à considérer ce problème comme l'un des défis les plus intéressants et sous-explorés de la mobilité urbaine. Voici ma réflexion honnête sur où nous en sommes et vers où nous pourrions nous diriger.

## L'approche traditionnelle : contrôle à temps fixe et actionné

Aujourd’hui, la plupart des feux de circulation fonctionnent encore selon l’un des deux paradigmes suivants :**Contrôle à temps fixe** attribue des phases vertes selon des horaires préprogrammés, généralement dérivés des décomptes historiques du trafic. Ces calendriers sont souvent mis à jour une fois par an, voire pas du tout. Ils sont robustes dans le sens où ils sont prévisibles et faciles à utiliser, mais ils sont fondamentalement réactifs au passé et non au présent.

**Le contrôle actionné** ajoute des détecteurs de boucle ou des caméras vidéo aux intersections. Lorsqu'un véhicule est détecté, le signal prolonge la phase verte. C'est mieux que l'heure fixe, mais cela reste fondamentalement local : chaque intersection s'optimise de manière isolée, sans aucune conscience de ce qui se passe en amont ou en aval.

Les deux approches partagent une limite fondamentale : **elles optimisent pour l'intersection, pas pour le réseau.** Un feu vert qui dégage une intersection peut créer une file d'attente qui déborde et en bloque trois autres. Le trafic est un système et non un ensemble de nœuds indépendants.## Le problème à l'échelle du réseau : pourquoi la coordination change tout

Pensez à ce qui se passe pendant une heure de pointe matinale typique. Les véhicules affluent des zones résidentielles vers les artères, et à moins que ces signaux artériels ne soient coordonnés, le résultat est un phénomène appelé **défaillance progressive de la bande** — l'exact opposé d'une vague verte. Le trafic stop-and-go apparaît non pas en raison d’une forte demande, mais en raison d’un mauvais timing des signaux.

C'est là que **SCOOT** (Split Cycle Offset Optimization Technique) et **SCATS** (Sydney Cooperative Adaptive Traffic System) ont fait leur marque. Développés dans les années 1980, ces systèmes utilisent des données de détection en temps réel pour ajuster la longueur, les divisions et les décalages des cycles sur un réseau d'intersections. Ils sont véritablement efficaces : les villes qui utilisent SCOOT signalent une réduction des retards de 10 à 20 %.Mais voici le problème : SCOOT et SCATS sont toujours basés sur des **modèles de flux de trafic** – des approximations macroscopiques ou mésoscopiques de la façon dont les véhicules se déplacent. Ces modèles ont été calibrés pour le trafic conventionnel. Ils luttent avec :

- **Conditions de sursaturation** (lorsque la demande dépasse la capacité)
- **Congestions non récurrentes** (incidents, chantiers, événements)
- **Trafic mixte** (véhicules à conduite humaine partageant des voies avec des véhicules autonomes)
- **Dépendances longue portée** (un goulot d'étranglement 3 intersections en amont)

L’approche basée sur un modèle a atteint un plafond. Pour aller plus loin, il faut sortir de la zone de confort du modèle.

## Apprentissage par renforcement : un autre type d'optimiseurC’est là que ma propre expérience de recherche recoupe une vision plus large. Lorsque j'ai travaillé sur la plateforme de cosimulation SUMO-Python pour le comptage des rampes d'accès aux autoroutes urbaines, j'ai commencé à me demander : un agent peut-il apprendre à contrôler les feux de circulation uniquement par expérience, sans modèle explicite ?

L'idée derrière l'**apprentissage par renforcement (RL)** pour le contrôle des feux de circulation est élégante :

- Le **agent** est le contrôleur des feux de circulation
- L'**état** correspond à l'état actuel du trafic : longueurs d'attente, temps d'attente, positions des véhicules, éventuellement données véhicule-infrastructure (V2I).
- L'**action** est la phase du signal vers laquelle passer
- La **récompense** est une combinaison de métriques : minimiser le délai total, maximiser le débit, pénaliser le débordement de file d'attente.L'agent n'a pas besoin de connaître la dynamique sous-jacente du flux de trafic. Il apprend une politique de contrôle directement à partir des interactions avec l'environnement – ​​tout comme la façon dont AlphaGo a appris à jouer au Go sans qu'on lui dise quel était le « meilleur coup » à chaque étape.

### Ce qui rend les choses difficiles

Tout n’est pas facile. Le feu tricolore RL est confronté à plusieurs défis pratiques :

**Exemple d'efficacité.** Contrairement à un jeu dans lequel des millions d'épisodes de jeu autonome sont réalisables, le déploiement dans le monde réel nécessite que l'agent apprenne d'abord en simulation. Construire une simulation fidèle n'est pas trivial : le comportement de changement de voie, l'agressivité du conducteur, l'imprévisibilité des piétons, tous doivent être modélisés.

**Coordination multi-agents.** Une seule intersection est une chose. Mais un réseau de 50 intersections, chacune avec son propre agent RL, crée un problème RL multi-agents. Les agents doivent se coordonner, pas seulement optimiser individuellement. L'action de chaque agent affecte les observations de ses voisins.**Sécurité et interprétabilité.** Le contrôle de la circulation est essentiel à la sécurité. Vous ne pouvez pas laisser un agent d’apprentissage expérimenter librement sur une intersection réelle. La ligne de base doit être sûre et l'apprentissage doit être limité – par exemple, des mises à jour de politiques conservatrices, un repli de l'humain dans la boucle ou des boucliers de sécurité.

**Généralisation.** Un agent RL formé sur les données des heures de pointe du matin peut échouer de manière spectaculaire à midi ou un week-end férié. Le changement de distribution est un réel problème.

### Orientations prometteuses

Malgré les défis, je suis vraiment enthousiasmé par la direction que prendront les choses. Quelques orientations que je trouve particulièrement prometteuses :**Représentez graphiquement les réseaux de neurones pour une connaissance spatiale.** Plutôt que de fournir à chaque intersection un vecteur plat de ses propres longueurs de file d'attente, les GNN permettent aux agents de communiquer via la topologie du réseau, en partageant des informations sur ce qui se passe aux intersections voisines. C'est ainsi que mon stage chez Bosch Chine a abordé la génération de trajectoire, et cette approche se transfère naturellement au contrôle du signal.

**RL hybride basé sur la physique.** La combinaison de modèles de trafic de principes fondamentaux (par exemple, modèles de stockage et de retransmission ou de transmission cellulaire) avec RL vous offre le meilleur des deux : le modèle fournit des contraintes de structure et de sécurité, tandis que RL gère l'optimisation fine. C'est là que se trouve mon article SCI sur le comptage des rampes d'autoroute - Q-learning soutenu par la simulation SUMO, avec modélisation de canalisation.**Contrôle V2I et CAV.** À mesure que les véhicules autonomes connectés (CAV) pénètrent le marché, la boucle de rétroaction change radicalement. Au lieu de déduire l’état du trafic à partir de détecteurs à boucle clairsemée, les signaux peuvent recevoir des données de position et de vitesse en temps réel de chaque véhicule du réseau. Il ne s’agit pas seulement d’une amélioration progressive : cela change fondamentalement ce qui est observable et contrôlable.

## Ce que nous avons construit et ce qui reste

Dans mon propre travail – de la plate-forme de fusion SUMO-CARLA au papier de mesure de rampe basé sur RL – j'ai pu constater par moi-même à la fois le potentiel et les lacunes. Les plateformes de simulation évoluent rapidement. L'interface TraCI de SUMO vous permet de tout scripter en Python. CARLA ajoute la fidélité du capteur nécessaire au contrôle basé sur la perception. Les outils ne sont plus le goulot d’étranglement.

Ce qui reste ouvert, à mon avis :1. **Environnements de référence** — nous avons besoin de références de réseau de trafic standardisées avec des métriques cohérentes, comme la manière dont ML a ImageNet et GLUE. La littérature regorge de problèmes de jouets à intersection unique qui ne se transfèrent pas à un déploiement réel.

2. **Justice et équité** — la plupart des contrôleurs de signaux RL optimisent le délai moyen. Mais un signal qui dessert le flux de circulation dominant pourrait systématiquement pénaliser les piétons, les cyclistes ou les véhicules sur les approches mineures. Le RL multi-objectifs avec des contraintes d’équité est sous-exploré.

3. **Transfert de la simulation à la réalité.** C'est le problème du dernier kilomètre. Une politique qui fonctionne dans SUMO échoue souvent dans le monde réel en raison d'un écart entre la simulation et le réel. La randomisation du domaine, l'identification du système et un RL robuste font tous partie de la solution.4. **Acceptation du public.** Les signaux adaptatifs qui modifient leur comportement de manière non déterministe peuvent dérouter les conducteurs. Il doit y avoir un fil de recherche sur les facteurs humains parallèlement à celui sur la théorie du contrôle.

## Réflexions finales

Le contrôle des feux de circulation est l’un de ces problèmes qui semblent simples en surface mais qui sont trompeusement profonds. Il s'agit d'un problème de contrôle, d'un problème de réseau, d'un problème d'équité et, de plus en plus, d'un problème d'apprentissage automatique. Le fait que les mécanismes de chronométrage du XIXe siècle fonctionnent encore sur la plupart des intersections du monde témoigne à la fois de leur fiabilité et de la difficulté de faire mieux.

Je suis optimiste. La convergence de capteurs bon marché, de communications V2X, d’une meilleure simulation et d’algorithmes RL plus intelligents crée une véritable opportunité de repenser la mobilité urbaine au niveau le plus fondamental : un feu vert à la fois.

---*Si vous travaillez sur le contrôle des feux de circulation, le RL pour les transports ou la simulation SUMO/CARLA, n'hésitez pas à nous contacter. Toujours heureux d'échanger des idées.*