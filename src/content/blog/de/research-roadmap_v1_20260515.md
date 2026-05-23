---
title: "Kulturelle Roadmap für UAV-Forschungsblogs in geringer Höhe: ein vollständiger Plan vom Blog bis zum Journal"
description: "Sortieren Sie systematisch den Forschungswert von 18 Artikeln zum Thema UAV in geringer Höhe im Blog, identifizieren Sie die fünf Richtungen mit dem größten Veröffentlichungspotenzial und stellen Sie ihre jeweiligen Innovationspunktaussagen, Zielzeitschriften, ergänzenden Experimentlisten und vorgeschlagenen Zeitpläne bereit."
pubDate: 2026-05-15
tags: ["Abschlussarbeitsplanung", "Forschungs-Roadmap", "UAV", "geringe Höhe", "Einreichungsstrategie", "T-ITS", "ICRA"]
category: Tech
---

# Low Altitude UAV Research Blog Cultural Roadmap: Ein vollständiger Plan vom Blog bis zum Journal

> Dieser Artikel ist keine technische Einführung, sondern ein **Dokument zum Forschungsmanagement**: Untersuchen Sie die in der Vergangenheit gesammelten Blog-Inhalte noch einmal und finden Sie heraus, welche es wert sind, in Zeitschriften veröffentlicht zu werden, welche noch fehlen und welche von Grund auf getestet werden müssen. Es ist auch eine Auseinandersetzung mit dem eigenen Forschungskontext.

---

## 0. Hintergrund und Ausgangspunkt

Derzeit hat der Blog **27 Artikel** gesammelt, darunter 18 Kernartikel zum Thema UAV in geringer Höhe, die sich mit Pfadplanung, Konfliktlösung, Planung mehrerer Maschinen, Wahrnehmungsrekonstruktion, digitalen Zwillingen, LLM/VLM-Planung und anderen Bereichen befassen.

Veröffentlichte Papierbasis: Journal of Advanced Transportation (SCI Q3), Q-Learning for Highway Ramp Control (DOI: 10.1155/2023/4771946), das den Forschungston von „Reinforcement Learning × Traffic System“ festlegte.

**Ziel dieses Artikels:**

1. Identifizieren Sie die 5–6 Richtungen in Blog-Inhalten, die für die Veröffentlichung am wertvollsten sind
2. Stellen Sie umsetzbare Informationen für jede Richtung bereit: Angabe der Innovationspunkte, Unterschiede zu bestehenden Arbeiten, Zielzeitschriften/Konferenzen, Liste ergänzender Experimente und Zeitplan für Vorschläge
3. Stellen Sie eine allgemeine Roadmap für die Einreichung für 12 Monate bereit
4. Machen Sie dieses Dokument zu einem lebendigen Forschungsmanagement-Tool (die Versionsnummer spiegelt sich im Dateinamen wider).

---

## 1. Panoramakarte für Bloginhalte

### 1.1 Drei große Forschungslinien

```
主线一：路径规划 × 冲突消解 × 多机调度
├── uav-urban-route-planning        （路径规划算法综述）
├── uav-conflict-resolution         （CD&R 机制综述+架构）
├── uav-conflict-env-construction   （仿真环境工程）
├── marl-kat-uav-conflict ★         （KAT MARL 框架）
├── large-scale-uav-scheduling ★    （三层百机调度）
└── urban-uav-3d-spatial-modeling   （3D空域建模参考）

主线二：感知 × 环境重建 × 数字孪生
├── uav-digital-twin-semantic-mapping ★  （五层数字孪生）
├── uav-semantic-mapping-functional-zoning ★（多源语义融合）
├── uav-nerf-gs-planning                 （NeRF/3DGS规划集成）
├── next-best-view-nerf-3dgs ★           （信息论NBV）
├── information-theory-active-perception （理论基础）
└── uav-multimodal-sim-data-synthesis    （多模态仿真工程）

主线三：LLM/VLM × 语义规划 × 形式验证
├── llm-uav-semantic-planning ★          （LTL/STL形式验证）
├── llm-guided-uav-planning-frontiers    （规划前沿概念）
├── hierarchical-vlm-uav-planning        （分层VLM架构）
└── vlm-uav-navigation-foundations       （VLN综述）

延伸：地面交通
├── carla-sumo-rl-lane-change ★          （PPO变道，已有实验）
└── traffic-signal-control               （信号控制反思）
```

★ = In diesem Artikel analysierte Papierkandidaten

### 1.2 Zusammenfassungsliste zur Reifebewertung| Artikel | Theoretischer Rahmen | Experimentelle Unterstützung | Umfassende Reife | Machbarkeit der Abschlussarbeit |
|------|----------|----------|----------|-----------|
| Marl-Kat-UAV-Konflikt | ★★★★★ | ★★☆☆☆ | ★★★★☆ | Hoch (erfinden Sie einfach das Experiment) |
| Großraum-UAV-Planung | ★★★★☆ | ★★☆☆☆ | ★★★☆☆ | Hoch (komplementäre Skalenexperimente) |
| next-best-view-nerf-3dgs | ★★★★★ | ★★★☆☆ | ★★★★☆ | Hoch (Ergänzung zum Online-Experiment) |
| UAV-Semantic-Mapping-Functional-Zoning | ★★★★☆ | ★★☆☆☆ | ★★★☆☆ | Mittel (Zusätzliche GIS-Daten) |
| llm-uav-semantische-planung | ★★★★☆ | ★☆☆☆☆ | ★★★☆☆ | Mittel (ergänzender Auswertungsdatensatz) |
| carla-sumo-rl-spurwechsel | ★★★☆☆ | ★★★★☆ | ★★★★☆ | Hoch (experimentiert) |

---

## 2. Stufe 1: Höchstes Veröffentlichungspotenzial (empfohlene Einreichung innerhalb von 6–12 Monaten)

### Papier A: Groß angelegte urbane UAV-Konfliktlösung – KAT-MARL-Framework

**Quellenartikel:** „marl-kat-uav-conflict“ + „uav-conflict-resolution“ + „uav-conflict-env-construction“.

**Zieljournal:** IEEE Transactions on Intelligent Transportation Systems (T-ITS, SCI Q1, IF ≈ 8,5)

#### Kerninnovationspunkt (Novelty Claim)

Das **KAT-Framework (Knowledge-Attention-Transfer)** soll die explizite Nachrichtenübermittlung durch ein Graph Attention Network (GAT) ersetzen, um eine implizite Koordination mehrerer Maschinen ohne Kommunikationsbeschränkungen zu erreichen:- **Impliziter Kommunikationsmechanismus:** Jedes UAV beobachtet nur den Nachbarschaftsstatus und extrahiert automatisch die relevantesten Nachbarinformationen durch das Aufmerksamkeitsgewicht von GAT, ohne Nachrichten zu senden
- **CTDE-Trainingsparadigma:** Zentralisiertes Training (Kritiker greift auf globalen Zustand zu) + dezentrale Ausführung (Akteur nutzt nur lokale Beobachtungen)
- **ORCA deckt die unterste Ebene ab:** Die zweistufige Sicherheitsgarantie der Lernstrategie und der geometrischen Analysemethode (ORCA) gewährleistet strikte Kollisionsfreiheit

Kernformelsystem:

GAT-Aufmerksamkeitsgewicht:
$$e_{ij} = \text{LeakyReLU}\!\left(\mathbf{a}^\top[\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_j]\right)$$

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k\in\mathcal{N}_i}\exp(e_{ik})}$$

Aggregierte Nachbarinformationen:
$$\mathbf{h}_i' = \sigma\!\left(\sum_{j\in\mathcal{N}_i}\alpha_{ij}\mathbf{W}\mathbf{h}_j\right)$$

Zentralisierte QMIX-Wertfunktionen:
$$Q_{tot}(\boldsymbol{\tau}, \mathbf{a}) = f_\theta\!\left(Q_1(\tau_1, a_1),\ldots,Q_N(\tau_N, a_N),\mathbf{s}\right)$$

Das Gewicht von $f_\theta$ ist nicht negativ (Monotonizitätsbeschränkung), wodurch die IGM-Bedingung (individuell-globale Maximierung) gewährleistet ist.

#### Abgrenzung zum bestehenden Werk

| Methode | Kommunikationsanforderungen | Maßstab | Echtzeit | Sicherheitsgarantie |
|------|---------|------|--------|---------|
| MADDPG | Keine | <20 | Schlecht | Keine |
| QMIX | Keine | <20 | Mittel | Keine |
| CommNet | Vollständige Übertragung | <50 | Schlecht | Keine |
| ORCA | Keine | Groß | Ausgezeichnet | Ja |
| **KAT (dieser Artikel)** | **Keine** | **50+** | **Gut** | **Ja (doppelschichtig)** |

#### Ergänzende Experimentliste- [ ] **Maßstabsablation:** 20/50/100 UAVs werden separat trainiert und getestet, und die Erfolgsquote, die durchschnittliche Verzögerung und die Berechnungsverzögerung werden aufgezeichnet
- [ ] **Basisvergleich:** Nur ORCA, MADDPG, QMIX (ohne GAT), QMIX+GAT (mit GAT und ohne ORCA-Abdeckung)
- [ ] **Szenario:** Erstellen Sie eine Simulationskarte basierend auf dem realen Straßennetz von Shanghai Lujiazui oder Beijing CBD
- [ ] **Indikatoren:** Missionserfolgsrate (Zielerreichungsrate), durchschnittliche zusätzliche Verzögerung (Sekunden), Konfliktrate (Konflikte/UAV/Minute), Inferenzverzögerung (ms)
- [ ] **Visualisierung:** Aufmerksamkeitsgewichtungs-Heatmap, die das Muster des UAV zeigt, das auf Nachbarn achtet

#### Zeitleiste

```
2026/06  搭建仿真环境（基于 existing uav-conflict-env-construction）
2026/07  训练 KAT 模型 + 基线对比实验
2026/08  写稿（Introduction / Method / Experiment / Conclusion）
2026/09  内部审阅 + 语言润色
2026/09  投稿 IEEE T-ITS（Regular Paper，通常 3–6 个月审回）
```

---

### Papier B: Dreischichtiges hierarchisches Planungssystem für Hunderte von Drohnen

**Quellenartikel:** „Großraum-UAV-Planung“ + „UAV-Stadtroutenplanung“.

**Zielzeitschrift:** IEEE T-ITS oder Transportation Research Teil C (SCI Q1, IF ≈ 7.6)

#### Kerninnovationspunkte

Es wird eine **dreischichtige hierarchische Architektur** vorgeschlagen, um das städtische Planungsproblem von über 100 UAVs in drei Unterprobleme zu zerlegen, die unabhängig voneinander optimiert und gemeinsam betrieben werden können:

**Makroskopische Ebene (Aufgabenzuweisung):** GNN-codierter Luftraumkartenstatus + ACO (Ameisenkolonieoptimierung) weist UAVs Aufgaben zu, um den globalen Durchsatz zu optimieren

Zielfunktion auf Makroebene:
$$\min\;\sum_{k=1}^{N}\!\left(w_1 T_k + w_2 \mathcal{E}_k\right) + w_3\cdot\text{Überlastung}(G)$$

**Meso-Schicht (Konfliktkoordination):** QMIX-Multiagentenkoordination, Geschwindigkeits-/Höhenanpassung basierend auf dem Makropfad zur Lösung von Konflikten

Dezentrale Entscheidungsfindung auf Mesoebene, lokale Strategie für jedes UAV:
$$\pi_k(a_k \mid \tau_k) = \text{softmax}(Q_k(\tau_k, \cdot;\theta_k))$$

**Mikroschicht (Flugbahnausführung):** ORCA-Geometrieanalyse + MPC-Rolloptimierung für eine zentimetergenaue VerfolgungMPC-Rolling-Optimierung (Vorhersageschrittgröße $H$):
$$\min_{\mathbf{u}_{0:H-1}}\sum_{t=0}^{H-1}\!\left\|\mathbf{x}_t - \mathbf{x}_{ref}\right\|_Q^2 + \|\mathbf{u}_t\|_R^2$$

#### Ergänzende Experimentliste

- [ ] **Skalenerweiterungskurve:** 20/50/100/200 UAV, Aufzeichnungssystemdurchsatz (UAV/min), End-to-End-Latenz, Rechenressourcen (CPU/GPU)
- [ ] **Basisvergleich:** FCFS (Wer zuerst kommt, mahlt zuerst), zentralisiertes MILP (optimal, aber langsam), zweistufige Architektur (keine Makroebene)
- [ ] **Szenariovielfalt:** Logistikszenario mit hoher Dichte (gleichmäßige Nachfrage) vs. plötzliches Spitzenszenario (Poisson-Ankunft)
- [ ] **Theoretische Analyse:** Gibt die theoretische Ableitung der Obergrenze des Systemdurchsatzes (basierend auf der Warteschlangentheorie)

#### Zeitleiste

```
2026/07  实现三层框架代码 + 集成测试
2026/08  规模扩展实验（需要较长训练时间）
2026/10  写稿
2026/11  投稿 Transportation Research Part C
```

---

### Papier C: Informationstheoriegesteuerte 3DGS-Aktiverfassungsplanung

**Quellenartikel:** „next-best-view-nerf-3dgs-exploration“ + „information-theory-active-perception-foundations“ + „uav-nerf-gs-planning“.

**Zielkonferenz:** ICRA 2026 (endet ca. 2026/09) oder IROS 2026

#### Kerninnovationspunkte

Verwenden Sie **Fisher Information Matrix (FIM)** als von Next-Best-View ausgewähltes Proxy-Ziel, um **3D Gaussian Splatting (3DGS)** aktive Konvergenzrekonstruktion voranzutreiben:

**Quantifizierung des Informationsgewinns:** Der nächste Standpunkt $\mathbf{v}^*$ wird gewählt, um den erwarteten Informationsgewinn in Bezug auf den Szenenparameter $\boldsymbol{\Theta}$ zu maximieren:

$$\mathbf{v}^* = \arg\max_{\mathbf{v}\in\mathcal{V}_{free}} \mathcal{I}(\boldsymbol{\Theta}; \mathbf{y}_\mathbf{v})$$Unter Verwendung der Cramér-Rao-Untergrenze ergibt die inverse FIM-Matrix eine Untergrenze für die Parameterschätzungsunsicherheit:

$$\text{Cov}(\hat{\boldsymbol{\Theta}}) \succeq \mathbf{F}(\boldsymbol{\Theta})^{-1}$$

** Differenzierbare Näherung von 3DGS FIM: ** Für jedes Gaußsche $\mathcal{G}_i$ kann sein FIM in Bezug auf den Mittelwert $\boldsymbol{\mu}_i$ wie folgt angenähert werden:

$$\mathbf{F}_i(\boldsymbol{\mu}_i) \ approx \sum_{\mathbf{r}\in\mathcal{R}(\mathbf{v})} \frac{\partial \hat{C}(\mathbf{r})}{\partial \boldsymbol{\mu}_i}\!\left(\frac{\partial \hat{C}(\mathbf{r})}{\partial \boldsymbol{\mu}_i}\right)^\top \frac{1}{\sigma_n^2}$$

**Gierige Strategie in Echtzeit:** Die globale optimale NBV-Suche ist NP-hart und verwendet gierige Serialisierung + Bereinigung (Entfernungsbeschränkung + Okklusionserkennung), um eine Entscheidungsfindung in Echtzeit zu erreichen (<50 ms/Schritt).

#### Vergleich mit bestehenden Methoden

| Methode | Zielfunktion | Ausdruck | Echtzeit | Informationssicherheit |
|------|---------|------|--------|---------|
| Grenze | Abdeckung | Voxel | Gut | Keine |
| Entropieminimierung | Besetzte Entropie | Voxel | Mittel | Schwach |
| ActiveGAMER | Rekonstruktionsqualität | 3DGS | Schlecht | Keine |
| **Dieser Artikel (FIM-3DGS)** | **Fischerinformationen** | **3DGS** | **Gut** | **Theoretische CRB-Garantie** |

#### Ergänzende Experimentliste- [ ] **Online-Rekonstruktionsexperiment:** AirSim-Stadtszene, autonomer UAV-Flug + Online-3DGS-Update
- [ ] **Metriken:** PSNR/SSIM (Rekonstruktionsqualität), Abdeckung (%), durchschnittlicher Informationsgewinn pro Schritt, Gesamtflugstrecke
- [ ] **Grundlage:** Zufällige Erkundung, Frontier-basiert, ActiveGAMER, SO-NeRF
- [ ] **Ablation:** FIM-Proxy-Ziel vs. reines Abdeckungsziel vs. reines Rekonstruktionsqualitätsziel

#### Zeitleiste

```
2026/06  实现 FIM-3DGS 可微近似模块
2026/07  AirSim 在线实验
2026/08  写稿（ICRA 格式，8页）
2026/09  投稿 ICRA 2026
```

---

### Papier D: Semantische Fusion aus mehreren Quellen + funktionale, partitionsgesteuerte UAV-Trajektorienplanung

**Quellenartikel:** „UAV-Semantic-Mapping-Functional-Zoning“ + „UAV-Digital-Twin-Semantic-Mapping“.

**Zielzeitschrift:** IEEE T-ITS oder Transportation Research Teil C

#### Kerninnovationspunkte

**Datenfusionspipeline aus mehreren Quellen:**

$$\mathcal{M}_{semantic} = \mathcal{F}_{fusion}(\mathcal{I}_{RS},\; \mathcal{G}_{OSM},\; \mathcal{P}_{POI},\; \mathcal{D}_{census})$$

Darunter ist $\mathcal{I}_{RS}$ das semantische Segmentierungsergebnis von Fernerkundungsbildern, $\mathcal{G}_{OSM}$ ist der Straßen-/Gebäude-GIS-Vektor, $\mathcal{P}_{POI}$ ist der Punkt von Interesse (Unternehmen/Krankenhaus/Schule) und $\mathcal{D}_{Volkszählung}$ sind die demografischen Daten.

**Risikomodell der städtischen funktionalen Zoneneinteilung:**

Definieren Sie den grundlegenden Risikokoeffizienten $\lambda_z$ für jeden Funktionsbereichstyp $z \in \{\text{Wohngebiet}, \text{Gewerbegebiet}, \text{Industriegebiet}, \text{Grünfläche}, \text{Wasser}\}$ und kombinieren Sie dabei den Zeitraumfaktor $\delta(t)$ (Morgen- und Abendgipfel vs. Nacht) und die Bodendichte $\rho_{bld}$:$$\mathcal{R}(x, y, t) = \lambda_{z(x,y)} \cdot \delta(t) \cdot \rho_{bld}(x,y)$$

**Risikobewusste Routenkostenfunktion:**

Einbetten des funktionalen Partitionsrisikodiagramms in A*-Kantengewichte:

$$d(u,v) = \ell_{uv}\cdot\!\left(1 + \beta_1\mathcal{R}_{uv} + \beta_2 TI_{uv}\right)$$

Dabei ist $TI_{uv}$ die Turbulenzintensität des Korridors (extrahiert aus dem Windfeldmodell), $\beta_1, \beta_2$ sind Kompromisskoeffizienten.

**Abgrenzung zu bestehenden Arbeiten:**
- Vorhandene Arbeiten unter Verwendung der **Bevölkerungsdichte** als Bodenrisiko-Proxy → statisch, grobkörnig
- Dieser Artikel verwendet das dreidimensionale Risikomodell **Funktioneller Zoneneinteilungstyp × Zeitraumfaktor × Gebäudedichte** → dynamisch, feinkörnig und kann über Städte hinweg migriert werden (einheitliche funktionale Zoneneinteilungsstandards)

#### Ergänzende Experimentliste

- [ ] **Datenerfassung:** Guangzhou/Shenzhen CBD GIS-Daten (OSM Open Source + hochauflösende Fernerkundungsbilder)
- [ ] **Basislinienvergleich:** Reiner kürzester Weg (Dijkstra), Gewichtung der Bevölkerungsdichte, Gewichtung der Gebäudeverdeckung
- [ ] **Indikatoren:** Risikoexpositionspunkte (REI = $\int \mathcal{R}(\boldsymbol{\xi}(t))\,\mathrm{d}t$), Weglänge, Flugzeit
- [ ] **Pareto-Kurve:** Kompromissfront zwischen REI und Pfadlänge
- [ ] **Generalisierungsexperiment:** Trainingsgewichtsparameter in Peking/Shanghai, Test in Guangzhou (städteübergreifende Übertragbarkeit)

#### Zeitleiste

```
2026/07  GIS 数据采集与预处理
2026/08  功能分区模型实现 + 航路规划实验
2026/09  写稿
2026/11  投稿 Transportation Research Part C
```

---

## 3. Stufe 2: Erfordert mehr zusätzliche Arbeit (12–18 Monate)

### Papier E: UAV-Missionsplanung mit LLM + formale Verifizierung

**Quellenartikel:** „llm-uav-semantic-planning“ + „llm-guided-uav-planning-frontiers“.

**Ziel:** ICRA/IROS oder IJCAI 2027

#### Kerninnovationspunkte

**Pipeline mit geschlossenem Kreislauf:**

```
自然语言任务描述
       ↓ LLM 转译
LTL/STL 形式规范
       ↓ 模型检测（NuSMV / Breach）
验证通过 → 执行
验证失败 → 反馈给 LLM → 迭代修正
```**Beispiel für eine LTL-Spezifikation („Vermeiden Sie das Überfliegen des Krankenhauses, bevor Sie Punkt B erreichen“):**

$$\varphi = \Box(\neg \text{Krankenhaus}) \;\wedge\; \Diamond(\text{Wegpunkt}_B)$$

**Hauptherausforderungen:**
- Übersetzungsgenauigkeit von LLM → LTL (erfordert die Erstellung eines Bewertungsdatensatzes: Paar aus natürlicher Sprache und Formspezifikation)
- Rechenaufwand für die Modellprüfung in großen Zustandsräumen (erfordert Zustandsraum-Abstraktionstechnologie)
- LLM-Halluzination führt zu unerfüllbaren Spezifikationen (erfordert Vorverarbeitung von Erfüllbarkeitsprüfungen)

#### Ergänzungswerkliste

- [ ] Erstellen Sie einen NL→LTL-Datensatz für die UAV-Mission (~500 Paare)
- [ ] Messen Sie die Übersetzungsgenauigkeit von GPT-4o / Llama-3
- [ ] Implementieren Sie die NuSMV-Schnittstelle, um Spezifikationen für städtische UAV-Szenen zu überprüfen
- [ ] Design-Halluzinationserkennungs- und Reparaturmodul

---

### Papier F: CARLA-SUMO Multi-Agent-Spurwechsel RL (Bodenverlängerung)

**Quellenartikel:** „carla-sumo-rl-lane-change“ (270.000 Schritte PPO-Versuchsergebnisse)

**Ziel:** Verkehrsforschung Teil C

#### Erstreckungsrichtung

– Aktueller Status: Einzelagenten-PPO, konvergiert in 270.000 Schritten
- Erweiterung: Multi-Agent (5–10 Autos wechseln gleichzeitig die Spur) + Unsicherheitsquantifizierung (Dropout/Ensemble)
- Sim2Real: Validierung der Richtlinienverallgemeinerung für nuScenes/Waymo-Datensätze

---

## 4. Zusammenfassung der wichtigsten Forschungslücken in verschiedenen Richtungen| Richtung | Blog-Status | Größte Lücke | Schwierigkeiten beim Schminken |
|------|---------|---------|---------|
| Papier A (KAT-MARL) | Vollständiger theoretischer Rahmen, klare Gleichungsableitung | Mangel an groß angelegten experimentellen Simulationsdaten | ★★☆ (3–4 Monate) |
| Papier B (dreistufige Planung) | Klare architektonische Gestaltung und vollständige Logik | Fehlen von mehr als 100 Skalenerweiterungsexperimenten | ★★★ (4–5 Monate) |
| Papier C (FIM-3DGS) | Tiefgreifende Ableitung der Informationstheorie und gutes Verständnis von 3DGS | Mangel an Online-Closed-Loop-Implementierung und -Experimenten | ★★★ (3–4 Monate) |
| Papier D (Funktionale Partition) | Klare Multi-Source-Integrationslogik | Mangel an echten GIS-Daten und Experimenten | ★★☆ (3–4 Monate) |
| Papier E (LLM+formale Verifizierung) | Komplettes Rohrleitungsdesign | Fehlender Auswertungsdatensatz, Übersetzungsgenauigkeit unbekannt | ★★★★ (6–8 Monate) |
| Papier F (CARLA-Spurwechsel) | Experimentelle Ergebnisse verfügbar | Multiagentenszenarien müssen ausgebaut werden | ★★☆ (3–4 Monate) |

---

## 5. Einreichungsstrategie und Leitfaden zur Zeitschriftenauswahl

### Liste der Zielzeitschriften/Konferenzen

| Zeitschrift / Konferenz | Feld | IF / Akzeptanzrate | Überprüfungszyklus | Geeignet für Papier |
|------------|------|------------|---------|-----------|
| **IEEE T-ITS** | Transportintelligenzsysteme | 8,5 / ~20 % | 3–6 Monate | A, B, D |
| **TR Teil C** | Verkehrswissenschaft und -technik | 7,6 / ~18 % | April–Juni | B, D, F |
| **IEEE T-ASE** | Automatisierungswissenschaft und -technik | 5,9 / ~22 % | 3–5 Monate | A |
| **IEEE RAL** | Roboter-Express | 4,6 / ~30 % | 2–3 Monate | C |
| **ICRA** | Roboterkonferenz | ~30% | Einmal im Jahr | C, E |
| **IROS** | Robotergipfel | ~40% | Einmal im Jahr | C, E |
| **IJCAI** | KI-Gipfel | ~15% | Einmal im Jahr | E |

### Vorschläge für progressive EinreichungspfadeBasierend auf veröffentlichten SCI Q3-Papieren wird die Strategie der **inkrementellen Verbesserung** empfohlen:

```
阶段一（2026）：冲刺 Q1 期刊
  → Paper A → IEEE T-ITS（同赛道，优势最大）
  → Paper C → IEEE RAL 或 ICRA（快速发表）

阶段二（2026–2027）：扩展并提升
  → Paper B → Transportation Research Part C
  → Paper D → IEEE T-ITS（第二篇，建立系列感）

阶段三（2027–）：攻顶会
  → Paper E → ICRA/IROS 或 IJCAI（高风险高回报）
```

**Wichtige Tipps:**
- T-ITS genießt eine hohe Akzeptanz der übergreifenden Forschung zum Thema „UAV × Urban Transportation System“, die mit dem Bereich der veröffentlichten Arbeiten übereinstimmt, und die Gutachter genießen die höchste Anerkennung des Hintergrunds.
- ICRA-Fristtermine liegen normalerweise im September des Vorjahres, planen Sie also im Voraus
- Es wird empfohlen, vor der Einreichung einen Vorabdruck auf arXiv vorzunehmen (erhöhte Akzeptanz im Transportbereich).

---

## 6. 12-monatige Einreichungs-Roadmap

```
时间        Paper A（KAT-MARL）     Paper C（FIM-3DGS）    Paper D（功能分区）     Paper B（三层调度）
─────────────────────────────────────────────────────────────────────────────────────────────────
2026/05    ▶ 环境搭建                ▶ FIM模块实现
2026/06    实验训练                  实验训练（AirSim）
2026/07    基线对比                  写稿启动              ▶ GIS数据采集
2026/08    写稿                      写稿完成              实验 + 写稿           ▶ 框架实现
2026/09    ◉ 投 T-ITS               ◉ 投 ICRA/RAL
2026/10                                                    写稿                  规模实验
2026/11                                                    ◉ 投 TR Part C
2026/12                                                                          写稿
2027/01                                                                          ◉ 投 TR Part C
─────────────────────────────────────────────────────────────────────────────────────────────────
◉ = 投稿节点   ▶ = 工作启动
```

---

## 7. Wartungsvertrag für dieses Dokument

**Dateinamenskonvention:** `research-roadmap_v{Versionsnummer}_{Jahr, Monat, Tag}.md`

- Aktuelle Version: „research-roadmap_v1_20260515.md“.
- Nächstes Update (nach der Einreichung von Paper A): „research-roadmap_v2_20260930.md“.
- Nach Erhalt der Rezensionskommentare: „research-roadmap_v3_202611xx.md“.

**Geänderter Inhalt für jedes Update:**
1. Entspricht dem Zeitplan von Paper (tatsächlicher Fortschritt vs. Plan)
2. Ergänzen Sie den Abschlussstatus der Experimentliste (klicken Sie auf ✅).
3. Zusammenfassung der Bewertungskommentare und Antwortstrategien
4. Neue Möglichkeiten für Veröffentlichungen (z. B. neu entdeckte Forschungslücken)

> Nutzen Sie die Versionsverwaltung für den Forschungsplan selbst, da die Richtung der Forschung kontinuierlich angepasst wird, wenn experimentelle Ergebnisse, Rezensionskommentare und neue Arbeiten auftauchen. Dieses Dokument sollte lebendig und nicht wegwerfbar sein.

---

**Anhang: Kurzüberprüfung der Korrespondenz zwischen Blogbeiträgen und Paper**| Blogbeitrag | Entsprechendes Papier |
|---------|----------|
| Marl-Kat-UAV-Konflikt | A (Haupt) |
| UAV-Konfliktlösung | A (Referenz) |
| uav-conflict-env-construction | A (experimentelle Umgebung) |
| Großraum-UAV-Planung | B (Haupt) |
| uav-urban-routenplanung | B (Referenz) |
| next-best-view-nerf-3dgs-exploration | C (Haupt) |
| Informationstheorie-aktive-Wahrnehmung | C (theoretische Basis) |
| uav-nerf-gs-planung | C (Referenz) |
| UAV-Semantic-Mapping-Functional-Zoning | D (Haupt) |
| uav-digital-twin-semantic-mapping | D (Referenz) |
| llm-uav-semantische-planung | E (Haupt) |
| llm-guided-uav-planung-frontiers | E (Referenz) |
| carla-sumo-rl-spurwechsel | F (Haupt) |