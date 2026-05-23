---
title: "Article : Optimisation coopérative basée sur RL de la canalisation et de la mesure de rampe dans les zones de tissage"
description: "Un article du premier auteur SCI Q3 présente une approche d'apprentissage par renforcement pour coordonner la conception de la canalisation et le comptage des rampes pour les zones de tissage d'autoroutes urbaines."
pubDate: 2023-04-10
tags: ["Ingénierie du trafic", "Apprentissage par renforcement", "voie express", "SUMO", "SCI Q3"]
category: Paper
doi: "10.1155/2023/4771946"
journal: "Journal of Advanced Transportation"
---

# Optimisation coopérative basée sur RL de la canalisation et de la mesure de rampe

**Auteurs :** Diantao Deng, Bo Yu, Duo Xu, Yuren Chen, You Kong
**Journal :** *Journal of Advanced Transportation*, 2023
**DOI :** [10.1155/2023/4771946](https://doi.org/10.1155/2023/4771946)
**Facteur d'impact :** 2,3 | **Catégorie :** SCI Q3

---

## Motivation

Les zones de tissage d’autoroutes urbaines sont connues pour leurs embouteillages. Lorsque les véhicules doivent fusionner ou diverger sur plusieurs voies sur une courte distance, des conflits surviennent – ​​et les contrôles conventionnels à stratégie unique (que ce soit le marquage des voies *ou* les signaux de rampe, jamais les deux ensemble) ne parviennent généralement pas à les gérer efficacement.

L'idée clé de cet article : **la canalisation (la manière dont les voies sont physiquement divisées) et la mesure des rampes (la manière dont les véhicules sont admis depuis les rampes d'accès) ne sont pas des problèmes indépendants.** Leur optimisation conjointe, plutôt que isolée, peut générer des gains de performances significatifs.

## MéthodeLe cadre proposé utilise un agent **Q-learning** pour coordonner dynamiquement les deux stratégies :

1. **Stratégies de canalisation** — deux types de configurations de marquage au sol qui guident la façon dont les véhicules fusionnent/divergent
2. **Mesure de rampe** — contrôle adaptatif du signal au niveau de la rampe d'accès pour réguler le débit entrant
3. **Mode coopératif** — Q-learning décide de la combinaison optimale des deux en temps réel

L'environnement est construit en **SUMO** (Simulation of Urban Mobility), avec des données de trafic réelles collectées via des **enquêtes aériennes UAV** utilisées pour calibrer et valider la simulation.

## Résultats

La méthode coopérative surpasse largement toutes les alternatives. La voie 3, la plus fortement touchée par les conflits de fusion, connaît une amélioration spectaculaire de **37 %** de la vitesse moyenne des véhicules :

- **Voie 1 :** +14,51 % d'augmentation de la vitesse moyenne
- **Voie-2 :** +14,81 % d'augmentation de la vitesse moyenne
- **Voie-3 :** +37,03 % d'augmentation de la vitesse moyenne

## Points clés à retenir- **L'optimisation conjointe bat les stratégies isolées.** Le contrôle du trafic est un problème de système ; le traiter comme tel rapporte des dividendes.
- **Le Q-learning est viable pour le contrôle des feux de circulation** même sans modèle dynamique complet : l'agent apprend la politique optimale uniquement à partir des signaux de récompense en simulation.
- **La co-simulation SUMO + Python** fournit une plate-forme pratique pour développer et tester des contrôleurs de trafic basés sur RL avant le déploiement dans le monde réel.
- La **collecte de données basée sur des drones** offre un moyen évolutif d'obtenir des données de trafic réelles sur le terrain pour l'étalonnage de la simulation.

## Travaux connexes

Cet article s'appuie sur des recherches antérieures sur la simulation SUMO menées par la communauté plus large de l'ingénierie du trafic et s'inscrit aux côtés d'autres travaux de contrôle des signaux basés sur RL dans la littérature. Le pipeline de co-simulation SUMO-Python développé ici est devenu la base du [projet Plateforme de simulation](/) référencé dans ma page À propos.

---*Article complet disponible sur : [https://doi.org/10.1155/2023/4771946](https://doi.org/10.1155/2023/4771946)*