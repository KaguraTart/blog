---
title: "Feuille de route de recherche v2 : Mise à niveau complète de la stratégie des principales revues et organisation des groupes de journaux sur les transports à basse altitude"
description: "Dans le cadre de l'objectif principal du premier trimestre, les itinéraires papier pour les drones à basse altitude, le cerveau des nuages ​​de transport à basse altitude, la couverture des scènes, la planification et la planification formelle sont réorganisés, et les priorités à court terme, le positionnement des soumissions, les limites narratives du système de transport et les entrées de planification spéciales sont clarifiées."
pubDate: 2026-05-15
updatedDate: 2026-05-23
tags: ["Planification de thèse", "Feuille de route de la recherche", "meilleure stratégie de publication", "T-ITS", "TR Partie C", "T-RO", "drone", "basse altitude"]
category: Tech
sourceHash: "1e0fdb9e8c70b09d0ee2f5dec5a08bf897624441"
---

# Feuille de route de recherche v2 : mise à niveau complète de la stratégie des principales revues et organisation des groupes de journaux sur les transports à basse altitude

> Déclencheur **v1 → v2 : ** L'enseignant exige explicitement que tous les articles soient publiés dans les principaux numéros du SCI Q1 (IF ≥ 7). La v1 inclut des chemins de « publication rapide » tels que les conférences RA-L (IF 4.6) et ICRA, qui ont été déplacées vers les trois principales matrices de publication de l'IEEE T-ITS, de la TR Part C et de l'IEEE T-RO.

---

## 0. Aperçu des modifications principales v1 → v2

### 0.1 Toutes les soumissions aux revues ont été avancées

| Papier | Objectif v1 | v1 SI | **Objectif v2** | **v2 SI** | Montant de la mise à niveau |
|-------|----------|-------|------------|-----------|---------|
| A : Résolution du conflit KAT-MARL | IEEE T-ITS | 8.5 | **IEEE T-ITS (Conserver)** | 8.5 | — |
| B : Planification à trois couches | TR Partie C | 8.5 | **TR Partie C / T-ITS (Maintenir)** | 8.5 | — |
| **C : Détection active FIM-3DGS** | **RA-L/ICRA** | **4.6** | **IEEE T-ITS → TR-C** | **8,5** | **Mise à niveau majeure** |
| D : Planification des partitions fonctionnelles | T-ITS / TR-C | 8.5 | **TR Partie C (maintenu)** | 8.5 | — |
| **E : Planification du langage formel VERA-UAV** | **ICRA/IJCAI** | **Conférence** | **AAAI premier + extension T-ITS** | **Conférence + 8,5** | Méthode de conférence d'abord, extension de journal plus tard |
| **F : Ingénierie de scénarios critiques pour la sécurité des drones** | **TR Partie C** | **8,5** | **Premier T-ITS + extension d'urgence TR-C** | **8,5** | Nouvelle route d'essai de sécurité indépendante à basse altitude |

### 0.2 Extension globale du calendrier- v1 : fenêtre de 12 mois (2026/05 – 2027/01), principalement en raison de la procédure accélérée RA-L
- **v2 : fenêtre de 24 à 30 mois (2026/06 – 2029/06)**, le cycle d'évaluation des principales revues est plus long et les expériences doivent être plus solides

### 0,3 Estimation de l'augmentation de la charge de travail

| Papier | charge de travail v1 | charge de travail v2 | Raisons de l'incrément |
|-------|----------|---------|-------------|
| Un | Mars-avril | Juin-août | Échelle expérimentale de 50 → 200 drones, analyse de la théorie des files d'attente |
| B | Avril-mai | Août-octobre | Test de généralisation multi-scénarios + données cartographiques réelles |
| **C** | **3-4 mois** | **12 à 15 mois** | **Complètement restructuré en un papier ITS d'économie de bas niveau** |
| D | Mars-avril | Juin-août | Généralisation de Gadot City + Flight Case réel |
| **E** | **6-août** | **8-décembre** | **Rédigez d'abord le document sur la méthode AAAI, puis développez-le vers le document sur le système ITS** |
| F | Mars-avril | Août-décembre | 76 millions de nettoyages de journaux d'exploration, métrique de couverture, tests accélérés, véritable expansion d'urgence à grande vitesse |

### 0.4 2026-05-22 Calibrage : Transportation Journal n'est pas une « narration », mais une boucle fermée de problèmes système

Cette fois, la feuille de route doit être recalibrée. Le domaine des transports accorde effectivement plus d'attention à la narration des problèmes et à l'importance du système que le domaine des algorithmes purs, mais il ne peut pas être compris comme « simplement raconter une histoire ronde ». Une norme plus précise est la suivante :

> Les articles sur les transports doivent raconter une histoire crédible du système, mais cette histoire doit être étayée par des modèles, des expériences, des indicateurs et des conditions aux limites.

Par conséquent, tous les plans ultérieurs partiels au TR-C/T-ITS doivent être vérifiés selon la chaîne suivante :

```text
真实交通系统问题
  -> 现实假设与边界条件
  -> 数学建模 / 运行机制
  -> 强 baseline 与消融
  -> 交通含义指标
  -> 敏感性 / 泛化 / 失败分析
  -> 对运行控制、规划设计或管理政策的启示
```

Tous les articles n'ont pas besoin d'utiliser la logique TR-C. Le cœur des documents sur les méthodes AAAI / ICLR / robotique basés sur des algorithmes reste la nouveauté de l'algorithme, les propriétés théoriques, la difficulté du benchmark et la reproductibilité. Ce n'est que lorsque l'objectif est TR-C / T-ITS / journal des transports qu'il est nécessaire de mettre « l'importance du système de transport » dans l'axe principal.| Thèse | Positionnement principal | Faut-il utiliser un récit sur le système de transport | Calibrage d'écriture actuel |
|------|--------|----------|--------------|
| Document A : Résolution du conflit KAT-MARL | T-ITS / Contrôle de sécurité du trafic à basse altitude | Oui, mais l'algorithme ne peut pas être affaibli | Passé de « Nouvel algorithme MARL » à « Vérification du système de résolution des conflits à basse altitude en cas de dégradation des communications, drone non coopératif, couloir à haute densité » |
| Article B : Planification à trois niveaux de centaines de drones | TR-C en premier | Fort besoin | Focus sur la capacité, les délais, la stabilité des files d'attente, les goulots d'étranglement des vertiports/chargements/couloirs et le repli multimodal |
| Document C : Détection active FIM-3DGS | Algorithme + Technologie d'activation du trafic | Requis sous condition | Si vous votez pour le T-ITS/TR-C, vous devez prouver que la détection active améliore les indicateurs de tâches de circulation tels que l'inspection, les interventions d'urgence et la distribution ; sinon, gardez le papier sur l'algorithme de perception du robot |
| Papier D : Planification des zones fonctionnelles sémantiques | TR-C / Aménagement urbain à basse altitude | Besoin | Focus sur les ODD, les zones fonctionnelles urbaines, l'exposition aux risques, les suggestions de planification, et non sur la segmentation sémantique pure |
| Article E : VERA-UAV | AAAI / Planification du langage formel | Pas d'application forcée | Suivez d'abord le document de planification/vérification de l'IA ; suivi de l'expansion des STI et des scénarios d'exploitation du trafic |
| Document F : Couverture de scénarios et urgence | Bifurcation T-ITS + TR-C | F-J1 est partiellement nécessaire, F-J2 est fortement nécessaire | F-J1 rédige des références de tests de sécurité ; F-J2 rédige un article sur l'exploitation du trafic sur l'allocation des ressources d'urgence sur l'autoroute du Shandong |
| Papier G/G1 : Agent LLM du cerveau des nuages ​​de trafic à basse altitude | AAAI/IJCAI en premier, extension T-ITS | G1 n'est pas obligatoire, l'expansion du journal est requise | G1 maintient la contribution de l'agent/de l'utilisation de l'outil/de la méthode de vérification ; journal version supplément indicateurs système et inspiration opérationnelle |

Les exigences minimales de dureté expérimentale pour la version du journal de transport ont également été uniformément augmentées :- Au moins 5 graines aléatoires, le tableau principal rapporte la moyenne ± intervalle de confiance standard ou bootstrap.
- La ligne de base ne peut pas simplement être sans contrôle/gourmand, elle doit inclure de solides méthodes classiques, des méthodes heuristiques et des méthodes d'apprentissage dans le domaine du problème.
- Les indicateurs ne peuvent pas uniquement rendre compte de la récompense, de la précision et du taux de réussite ; ils doivent inclure des indicateurs de trafic tels que le nombre de conflits, le LoWC, le NMAC, le retard, la distance supplémentaire, l'énergie, le débit, l'utilisation des ressources et la durée d'exécution.
- La généralisation doit être faite : former une faible densité et tester une haute densité, former à petite échelle et tester à grande échelle, former une topologie fixe et tester une nouvelle topologie, former un trafic coopératif et tester un trafic de communication non coopératif/dégradé.
- Il doit y avoir une analyse de cas de défaillance montrant à quelle densité, taux de perte de communication, comportement non coopératif ou goulot d'étranglement des ressources le système a échoué.

---

### 0.5 2026-05-23 Organisation : ordre de lecture et priorité des documents de planification en cours

La feuille de route générale actuelle est conservée sous le nom de « Research Matrix Entry », et la mise en œuvre spécifique est soumise au document spécial B/E/F/G/G1. L’ordre de lecture recommandé est le suivant :| Priorité | Documents | Positionnement actuel | Actions récentes |
|--------|------|----------|--------------|
| P0 | Paper G1 : Plan de thèse complet de CloudBrain-Agent | AAAI / IJCAI en premier | Implémentez d'abord l'agent vérifiable, CloudBrain-Bench, la chaîne d'outils et l'expérience principale |
| P1 | Article B : Planification hiérarchique à trois niveaux de centaines de drones | TR-C en premier | Créer une référence de file d'attente synthétique, un planificateur Lyapunov et une base de référence solide |
| P1 | Document F : Ingénierie de scénarios critiques pour la sécurité des drones | T-ITS d'abord, extension d'urgence TR-C | Terminer le premier F-J1 : métrique de couverture + tests accélérés |
| P2 | Article E : VERA-UAV | Document de méthode AAAI, extension ultérieure de T-ITS | Condensé en réparation dactylographiée IR + LTL/STL + vérificateur, pas d'article majeur sur le système de transport en premier |
| P3 | Papier C / Papier D | Dans l'attente d'une convergence plus poussée des données et des tâches | Conserver la direction, mais ne pas rivaliser avec B/F/G1 pour les ressources expérimentales récentes |

Cette édition nécessite une clarification particulière : **L'ancien article F = ligne RL de changement de voie multi-agent CARLA-SUMO n'est plus compté dans le groupe actuel des articles sur les drones à basse altitude. ** Si la direction de conduite autonome au sol est refaite à l'avenir, elle pourra être restaurée en tant que document de transport terrestre indépendant ; actuellement, le document F fait spécifiquement référence à l'ingénierie de scénarios critiques pour la sécurité des drones.

La séquence d’exécution recommandée dans un avenir proche est la suivante :

1. Faites G1 en premier, car il peut unifier le planificateur du papier B, le vérificateur du papier E et le test de résistance du scénario du papier F dans une chaîne d'outils de « cerveau de nuage de trafic à basse altitude ».
2. Démarrez simultanément le benchmark synthétique de B, car il constitue la base principale des documents ultérieurs du système TR-C et des outils de planification G1.
3. F-J1 avance après avoir eu des journaux d'exploration et des scripts de génération de scènes pour éviter de tomber dans trop de récits d'application réels au début.
4. E Conservez les documents de la méthode AAAI et ne les développez pas à l'avance dans un vaste système de journaux de transport à basse altitude.

---## 1. Carte panoramique du contenu du blog (conforme à la v1)

Les trois grands axes de recherche restent inchangés (voir v1 pour plus de détails) :
- Axe principal 1 : planification du chemin × résolution des conflits × planification multi-machines
- Axe principal 2 : Perception × Reconstruction de l'environnement × Jumeau numérique
- Axe principal 3 : LLM/VLM × Planification sémantique × Vérification formelle

---

## 2. Niveau 1 : articles de revues de premier plan (dans les 24 mois)

### Document A : Résolution des conflits urbains à grande échelle liés aux drones – KAT-MARL (Maintenir le positionnement des principaux problèmes)

**Journal cible :** Transactions IEEE sur les systèmes de transport intelligents (T-ITS, IF 8.5 Q1)

**Modifications par rapport à la v1 :** Mise à niveau à l'échelle expérimentale, expansion de l'analyse théorique

#### v2 nouvelles exigences

- Taille de l'expérience de 100 UAV → **200 UAV** (pour répondre à la préférence de T-ITS pour la simulation à grande échelle)
- Ajout de **analyse de la théorie des files d'attente** : prouver la limite supérieure du débit du système du framework KAT
- Ajout de la **cartographie réelle du réseau routier** : extension de la simulation CBD à 2 à 3 villes réelles (Shanghai Lujiazui, Beijing CBD, Shenzhen Futian)
- Ajout d'**expériences de robustesse** : délai de communication, bruit des capteurs, scénarios de défaillance du drone

#### Chronologie v2
```
2026/06–07  实验环境搭建（基于 uav-conflict-env-construction）
2026/08–10  训练 KAT + 200 UAV 规模扩展实验
2026/11     真实城市路网泛化实验
2026/12     排队论理论分析与证明
2027/01–02  写稿（25 页 T-ITS 格式）+ 内部审阅
2027/03     ◉ 投稿 IEEE T-ITS
2027/09     收到审稿意见（4–6 月审回）
2027/12     接受目标
```

---

### Article B : Planification hiérarchique à trois niveaux de centaines de drones (maintien du meilleur positionnement de publication)

**Journal cible :** Transportation Research Part C ou IEEE T-ITS (IF 8.5 Q1)

**Modifications par rapport à la version 1 :** Ajoutez les fondements mathématiques de la théorie des files d'attente et ajoutez des scénarios de transport multimodal.

#### v2 nouvelles exigences

- **Améliorations théoriques :** Théorie des files d'attente + preuve de stabilité de Lyapunov
- **Extension multimodale :** Répartition conjointe UAV + véhicule terrestre (améliorer l'adéquation du TR-C aux systèmes de transport)
- **Données réelles :** Comparaison avec les données du pilote de livraison sans pilote Meituan/JD (si disponible)

#### Chronologie v2
```
2026/08–09  三层框架代码实现
2026/10–12  规模扩展实验（20/50/100/200 UAV）
2027/01     排队论与 Lyapunov 分析
2027/02–03  写稿
2027/04     ◉ 投稿 TR Part C
2027/10     接受目标
```

---

### Paper C : Détection active FIM-3DGS - **Reconstruction majeure (voir le document spécial v2 pour plus de détails)**

**Journal cible :** IEEE T-ITS (de préférence) → TR Part C (de préférence), IF 8.5 Q1**Raison de la reconstruction :** Le positionnement v1 RA-L est trop bas, l'enseignant exige une publication supérieure

**Modifications principales de la v2 (voir `paper-c-fim-3dgs-uav-active-perception_v2_20260515.md` pour plus de détails) : **

1. **Mise à niveau du positionnement :** Extrait de « Document sur l'algorithme de perception » → « Technologie habilitante économique à basse altitude »
2. **Élargissement de l'évaluation :** Indicateur de perception unique → système d'indicateurs à cinq niveaux (perception/planification/tâche/système/économie)
3. **Étude de cas :** Trois nouveaux cas d'application en matière de transport (inspection de bâtiments, livraison du dernier kilomètre, intervention d'urgence)
4. **Extension expérimentale :** Ajout de la simulation conjointe SUMO + AirSim + expérience au niveau du système multi-UAV
5. **Contribution à l'ensemble de données :** Ensemble de données open source UAV-Delivery-Dataset auto-construit

#### Chronologie v2
```
2026/06–10  五阶段实验（核心算法 + 三案例 + 多机系统级）
2026/11–12  数据整合 + 初稿（22 页 T-ITS 格式）
2027/01–02  润色 + 内部审阅
2027/03     ◉ 投稿 IEEE T-ITS
2027/09     收到审稿意见
2027/12     接受 / 转 TR-C
2028/06     最终发表
```

Pour une planification détaillée du Paper C, voir [Document spécial Paper C v2] (/blog/paper-c-fim-3dgs-uav-active-perception_v2_20260515/).

---

### Article D : Fusion sémantique multi-sources + planification de la trajectoire du drone basée sur des partitions fonctionnelles (maintien du positionnement de publication en tête)

**Journal cible :** Recherche sur les transports, partie C (IF 8.5 Q1)

**Modifications par rapport à la version 1 :** Extension de l'expérience de généralisation multi-villes

#### v2 nouvelles exigences

- **Généralisation multi-villes :** Formation + tests dans 5 villes (Pékin, Shanghai, Guangzhou, Shenzhen, Wuhan)
- **Flight case réel :** Coopération avec un pilote de livraison d'UAV ou reproduction de données publiques
- **Quantification des risques :** Introduction de l'évaluation actuarielle des risques (perspective assurance/rémunération)

#### Chronologie v2
```
2026/07–09  GIS 数据采集（5 城市）
2026/10–12  功能分区模型 + 多城市实验
2027/01     真实飞行案例对比
2027/02–03  写稿
2027/04     ◉ 投稿 TR Part C
2027/10     接受目标
```

---

## 3. Niveau 2 : meilleurs articles de revues présentant de plus grands défis techniques

### Paper E : Planification du langage formel VERA-UAV (AAAI d'abord, puis extension ITS)

**cible v1 :** ICRA/IJCAI (conférence)

**Cible actuelle :** AAAI/IJCAI en premier, sauvegarde de l'extension T-ITS**Raison de l'étalonnage :** La principale contribution de l'article E est la planification/vérification de l'IA, et il ne devrait pas être forcé de devenir un article important et dispersé sur le système de transport dans le but d'une publication de premier ordre. La version AAAI donne la priorité à la réponse « Comment les tâches UAV en langage naturel forment des trajectoires sûres exécutables via des IR typés, LTL/STL, des contre-exemples de validateur et des solutions de repli symboliques ».

#### Sens de fermeture actuel

- **Ligne principale de la méthode :** Instruction NL -> TaskIR tapé -> LTL/STL -> vérificateur -> contre-exemple/réparation de robustesse -> vérification de trajectoire.
- ** Limites théoriques : ** Ne revendique pas l'exhaustivité du LLM ; prouver l'exhaustivité relative sous les hypothèses de DSL fini, de vérificateur décidable et de planificateur sous-jacent complet.
- ** Limite de l'expérience : ** L'expérience principale utilise un benchmark synthétique contrôlé ; Les indicateurs AirSim, logistique réelle et ITS multi-UAV seront étendus ultérieurement.
- **Stratégie de soumission :** L'article principal de l'AAAI met l'accent sur les méthodes, les théories, les références et les bases de référence solides ; Le T-ITS est élargi pour inclure des indicateurs de fonctionnement du trafic et des scénarios réels à basse altitude.

#### Chronologie v2
```
2026/06–07  冻结 TaskIR DSL、任务生成器和验证器接口
2026/08–09  实现 Direct LLM / NL2LTL-style / LTLCodeGen-style / VERA-UAV baselines
2026/10     跑主实验、消融和泛化测试
2026/11     完成理论证明、图表和初稿
2026/12     ◉ 投稿 AAAI / IJCAI 对应批次
2027/03     根据结果扩展 T-ITS 版本
```

---

### Paper F : Ingénierie de scénarios critiques pour la sécurité des drones et applications d'urgence (remplacement de l'ancienne gamme CARLA-SUMO)

**Cible actuelle :** F-J1 est le principal candidat pour l'IEEE T-ITS ; F-J2 est le principal candidat pour TR-C.

**Changement de positionnement :** L'article F actuel ne fait plus référence au RL de changement de voie CARLA-SUMO, mais se concentre sur l'ingénierie de scénarios critiques pour la sécurité des drones en tant que priorité du journal : établissez d'abord une couverture reproductible des scénarios critiques pour la sécurité et des documents de tests accélérés, puis étendez la même plate-forme au déploiement des ressources de secours d'urgence sur l'autoroute du Shandong.

#### Nouvelles exigences actuelles- **Espace de scène :** Définissez clairement la cellule de test du drone de 50 m x 50 m x 50 m, la combinaison d'obstacles, les obstacles dynamiques, le champ de vent, l'occlusion de la zone visuelle, la zone d'exclusion aérienne et les objectifs de la mission.
- **Actifs expérimentaux existants :** 76 millions de journaux d'exploration ne peuvent être rédigés que comme « base disponible » et ne peuvent pas être rédigés comme résultats expérimentaux finaux ; ils doivent être nettoyés en taxonomie des échecs, en trous de couverture et en cas de stress du planificateur.
- **Ligne principale de la méthode :** métrique de couverture -> échantillonneur guidé par la couverture -> filtre danger-validité -> tests accélérés -> évaluation inter-planificateurs.
- **Base de référence forte :** génération aléatoire, échantillonnage grille/LHS, optimisation bayésienne, CMA-ES, génération contradictoire RL, génération contrainte de style Scenic.
- **Expansion du trafic :** Le F-J2 n'a introduit que l'urgence sur l'autoroute du Shandong, en se concentrant sur la découverte des accidents, la reconnaissance des drones, le déploiement des ressources au sol, le temps de réponse et la récupération du trafic.

#### Chronologie v2
```
2026/06–07  整理 7600 万次探索日志，冻结场景空间和 coverage metric
2026/08–10  实现 accelerated testing 与强 baseline
2026/11     cross-planner evaluation、failure taxonomy、统计检验
2026/12–2027/01  写 F-J1 初稿
2027/02     ◉ 投稿 IEEE T-ITS
2027/03–06  扩展山东高速应急资源调配 F-J2
```

---

## 4. Feuille de route globale de soumission sur 30 mois pour les principaux problèmes

```
─────────────────────────────────────────────────────────────────────────────────────────
时间        A (T-ITS)    B (TR-C)     C (T-ITS)    D (TR-C)     E (AAAI)     F (T-ITS/TR-C)
─────────────────────────────────────────────────────────────────────────────────────────
2026/06    ▶ 环境搭建                  ▶ 算法实现                              ▶ 日志清洗
2026/07    实验训练                    AirSim搭建    ▶ GIS采集
2026/08    实验                        案例1巡检    实验          
2026/09                  ▶ 框架实现    案例2配送                                加速测试
2026/10                  规模实验      案例3应急    多城市实验    ▶ 数据集     baseline
2026/11                  实验          多机系统级   案例研究
2026/12                  实验          初稿         案例研究      数据集完成    
2027/01                  理论分析      润色         写稿          实验          F-J1 写稿
2027/02                  写稿          润色         润色          实验          ◉ 投 T-ITS
2027/03    ◉ 投 T-ITS               ◉ 投 T-ITS                              F-J2 启动
2027/04                  ◉ 投 TR-C                ◉ 投 TR-C
2027/05                                                          实验
2027/06                                                          多UAV案例
2027/07                                                          写稿
2027/08                                                          写稿
2027/09    审稿意见                  审稿意见                   ◉ 投 T-ITS    审稿意见
2027/10                  接受目标                   接受目标                    接受目标
2027/11
2027/12    接受目标                  接受/转TR-C
2028/03                                                          接受目标
2028/06                              最终发表
─────────────────────────────────────────────────────────────────────────────────────────
◉ = 投稿节点   ▶ = 工作启动
```

**Rythme de base :**
- **Deuxième semestre 2026 :** G1 / E / F-J1 forment le premier lot d'expériences exécutables pour éviter que tous les travaux soient repoussés au printemps 2027 en même temps.
- **Printemps 2027 :** A/B/C/D continuera d'être promu comme la principale ligne d'articles systématiques dans le numéro principal.
- **Premier semestre 2027 :** Le F-J2 se différencie de la plate-forme F-J1 en une version TR-C de déploiement de ressources d'urgence à grande vitesse.
- **S1 2028 :** Période principale de réception.

---

## 5. Explication détaillée de la matrice du journal principal| Revue | Champ | SI | Taux d'acceptation | Cycle de révision | v2 adaptation Papier |
|------|------|-----|--------|---------|--------------|
| **IEEE T-ITS** | ITS Général | 8.5 | ~20% | 4 à 6 mois | Extensions de journaux A, C, F-J1, G/G1 |
| **TR Partie C** | Nouvelles technologies de transport | 8.5 | ~18% | 4 à 6 mois | B, D, F-J2 |
| **IEEE T-RO** | Robotique | 7.4 | ~25% | 6 à 10 mois | Préparation C |
| **TR Partie B** | Méthodologie de transport | 6.0 | ~15% | 6 à 8 mois | Préparation B |
| **Sciences des transports** | Sciences des transports | 5.4 | ~12% | 6 à 10 mois | Investissement B |

**Principes de la matrice de soumission v2 :**
- **Q1 avec IF ≥ 8 préféré** (T-ITS, TR-C)
- **Préparez-vous à investir au T1** (T-RO) du même fonds avec IF ≥ 7
- **Les revues avec IF < 7 ne seront plus prises en compte**

---

## 6. Évaluation des risques et alternatives

### 6.1 Principaux risques de la stratégie de publication principale

**Risque 1 : La période d'évaluation dépasse la fenêtre d'obtention du doctorat**
- Le premier cycle de révision a lieu d'avril à juin pour le numéro principal, et les révisions peuvent être retardées jusqu'à 12 mois et plus.
- **Réponse :** Soumission centralisée au printemps 2027, avec 12 mois réservés aux révisions
- **En résumé :** Au moins 2 articles acceptés, le reste peut être obtenu avec le statut "soumis/en cours de révision".

**Risque 2 : charge de travail expérimentale excessive**
- La charge de travail totale de la v2 est d'environ 50 à 60 mois (si en série), ce qui nécessite une division du travail en équipe/coopération
- **Réponse :** Donner la priorité à G1/B/F-J1/E dans un avenir proche et conserver uniquement les entrées de concepts et de données dans d'autres directions pour éviter la dilution des ressources

**Risque 3 : temps perdu lors du changement après un rejet**
- Un round de rejet + transfert = environ 6 mois de perte
- **Réponse :** Préparez à l'avance le double cadrage TR-C / T-ITS dans la lettre de motivation

### 6.2 Priorité des soumissions alternatives| Papier | Premier choix | Alternative 1 | Alternative 2 |
|-------|------|------|------|
| Un | T-ITS | TR Partie C | IEEE T-Cyber ​​|
| B | TR Partie C | T-ITS | TR Partie B |
| C | T-ITS | TR Partie C | IEEE T-RO |
| D | TR Partie C | T-ITS | TR Partie D (Environnement) |
| E | AAAI/IJCAI | T-ITS | IEEE T-SMC |
| F | T-ITS | TR Partie C | T-ASE / T-RO |

---

## 7. Résumé en une phrase du rapport à l'enseignant

> « Le groupe de papier actuel a été réorganisé en la ligne principale des drones à basse altitude/cerveau de nuage de transport à basse altitude : G1 est le premier à investir dans AAAI/IJCAI, B est le principal investissement dans TR-C, F-J1 est le principal investissement dans T-ITS, E maintient les articles sur la méthode AAAI et réserve l'expansion de T-ITS. récits."

---

## 8. Instructions de traitement des documents v1

- **v1 (`research-roadmap_v1_20260515.md`) :** Réservé en tant qu'archive historique pour enregistrer la conception de la « Stratégie hybride de publication rapide »
- **v2 (ce document) :** Le document de planification actuellement en vigueur
- **Conditions de déclenchement de la prochaine mise à jour :** ① Compléter les données expérimentales de l'épreuve A ② Recevoir les premiers commentaires de révision ③ L'enseignant ajuste la direction

---

**Annexe : Correspondance entre les articles du blog et Paper (conforme à la v1)**| Article de blog | Document correspondant |
|---------|----------|
| marl-kat-uav-conflit | A (principal) |
| résolution de conflit de drones | A (référence) |
| drone-conflit-env-construction | A (environnement expérimental) |
| planification de drones à grande échelle | B (principal) |
| drone-urban-route-planification | B (référence) |
| prochaine-meilleure-vue-nerf-3dgs-exploration | C (principal) |
| théorie-de l'information-perception-active | C (base théorique) |
| drone-nerf-gs-planification | C (référence) |
| **papier-c-fim-3dgs-uav-active-perception_v2_20260515** | **C Planification spéciale (v2)** |
| uav-cartographie-sémantique-zonage-fonctionnel | D (principal) |
| cartographie-sémantique-twin-nuav-nuav | D (référence) |
| llm-uav-planification-sémantique | E (principal) |
| llm-guidé-UAV-planification-frontières | E (référence) |
| papier-b-hiérarchique-uav-scheduling-trc-plan-v1-20260519 | B planification spéciale |
| paper-e-vera-uav-experiment-taskbook-v1-20260517 | E livre de tâches spéciales |
| paper-f-uav-scenario-coverage-journal-roadmap-v2-20260520 | F Planification spéciale |
| paper-g-basse-altitude-cloud-brain-llm-roadmap-v1-20260520 | G itinéraire total |
| paper-g1-cloudbrain-agent-full-paper-plan-v1-20260520 |G1 premier projet de thèse complet |
| carla-sumo-rl-lane-changement | Ancienne ligne F, actuellement non incluse dans le groupe papier des drones à basse altitude |