---
title: "Forschungs-Roadmap v2: Umfassende Aktualisierung der Top-Journal-Strategie und Organisation von Paper-Gruppen für Tiefgebirgstransporte"
description: "Im Rahmen des Q1-Hauptproblemziels werden die Papierrouten für UAVs in geringer Höhe, das Cloud Brain für den Transport in geringer Höhe, die Szenenabdeckung, die Terminplanung und die formelle Planung neu organisiert und die kurzfristigen Prioritäten, die Einreichungspositionierung, die erzählerischen Grenzen des Transportsystems und die besonderen Planungseingänge geklärt."
pubDate: 2026-05-15
updatedDate: 2026-05-23
tags: ["Abschlussarbeitsplanung", "Forschungs-Roadmap", "Top-Publikationsstrategie", "T-ITS", "TR Teil C", "T-RO", "UAV", "geringe Höhe"]
category: Tech
---

# Forschungs-Roadmap v2: Umfassende Aktualisierung der Top-Journal-Strategie und Organisation von Papiergruppen zum Thema Tieftransport

> **v1 → v2-Auslöser: ** Der Lehrer fordert ausdrücklich, dass alle Arbeiten in den Top-Ausgaben von SCI Q1 veröffentlicht werden müssen (IF ≥ 7). v1 umfasst „Rapid Publication“-Pfade wie RA-L (IF 4.6) und ICRA-Konferenzen, die in die drei Top-Publikationsmatrizen IEEE T-ITS, TR Part C und IEEE T-RO verschoben wurden.

---

## 0. v1 → v2 Kernänderungen – Übersicht

### 0.1 Alle Einreichungen bei Zeitschriften wurden nach oben verschoben

| Papier | v1 Ziel | v1 WENN | **v2-Ziel** | **v2 WENN** | Upgrade-Betrag |
|-------|----------|-------|------------|-----------|---------|
| A: KAT-MARL-Konfliktlösung | IEEE T-ITS | 8,5 | **IEEE T-ITS (Behalten)** | 8,5 | — |
| B: Dreischichtige Planung | TR Teil C | 8,5 | **TR Teil C / T-ITS (Halten)** | 8,5 | — |
| **C: FIM-3DGS Active Sensing** | **RA-L/ICRA** | **4,6** | **IEEE T-ITS → TR-C** | **8,5** | **Großes Upgrade** |
| D: Funktionale Partitionsplanung | T-ITS / TR-C | 8,5 | **TR Teil C (beibehalten)** | 8,5 | — |
| **E: VERA-UAV Formale Sprachplanung** | **ICRA/IJCAI** | **Konferenz** | **AAAI zuerst + T-ITS-Erweiterung** | **Konferenz + 8,5** | Zuerst Konferenzmethode, später Journalerweiterung |
| **F: Entwicklung sicherheitskritischer UAV-Szenarien** | **TR Teil C** | **8,5** | **T-ITS zuerst + TR-C Notfallerweiterung** | **8,5** | Neue unabhängige Sicherheitstestroute in geringer Höhe |

### 0,2 Gesamtverlängerung der Zeitleiste- v1: 12-Monats-Fenster (2026/05 – 2027/01), hauptsächlich aufgrund des RA-L-Fast-Tracks
- **v2: 24–30-Monats-Fenster (2026/06 – 2029/06)**, der Begutachtungszyklus der Top-Zeitschriften ist länger und die Experimente müssen solider sein

### 0,3 Schätzung der Arbeitsbelastungssteigerung

| Papier | v1-Arbeitslast | v2-Arbeitslast | Gründe für die Erhöhung |
|-------|----------|---------|---------|
| A | März–April | Juni–August | Experimenteller Maßstab von 50 → 200 UAV, Analyse der Warteschlangentheorie |
| B | April–Mai | August–Oktober | Multi-Szenario-Generalisierungstest + echte Kartendaten |
| **C** | **3–4 Monate** | **12–15 Monate** | **Komplett umstrukturiert in ein Low-Level-Economy-ITS-Papier** |
| D | März–April | Juni–August | Verallgemeinerung von Gadot City + tatsächlicher Flugfall |
| **E** | **6.–August** | **8–Dezember** | **Führen Sie zuerst das AAAI-Methodenpapier durch und erweitern Sie es dann zum ITS-Systempapier** |
| F | März–April | August–Dezember | 76 Millionen Erkundungsprotokollbereinigung, Abdeckungsmetrik, beschleunigte Tests, echte Hochgeschwindigkeits-Notfallerweiterung |

### 0.4 22.05.2026 Kalibrierung: Transportation Journal ist kein „Storytelling“, sondern ein geschlossener Kreislauf von Systemproblemen

Diesmal muss die Roadmap neu kalibriert werden. Der Transportbereich legt zwar mehr Wert auf Problemerzählung und Systembedeutung als der reine Algorithmusbereich, kann aber nicht als „nur das Erzählen einer runden Geschichte“ verstanden werden. Ein genauerer Standard ist:

> Transportpapiere sollten eine glaubwürdige Systemgeschichte erzählen, aber diese Geschichte muss durch Modelle, Experimente, Indikatoren und Randbedingungen gestützt werden.

Daher müssen alle nachfolgenden Pläne, die Teil von TR-C/T-ITS sind, gemäß der folgenden Kette überprüft werden:

```text
真实交通系统问题
  -> 现实假设与边界条件
  -> 数学建模 / 运行机制
  -> 强 baseline 与消融
  -> 交通含义指标
  -> 敏感性 / 泛化 / 失败分析
  -> 对运行控制、规划设计或管理政策的启示
```

Nicht alle Arbeiten müssen die TR-C-Logik verwenden. Der Kern starker algorithmusgesteuerter AAAI-/ICLR-/Robotik-Methodenpapiere ist immer noch die Neuheit des Algorithmus, die theoretischen Eigenschaften, die Benchmark-Schwierigkeit und die Reproduzierbarkeit. Nur wenn das Ziel TR-C / T-ITS / Transportjournal ist, ist es notwendig, „Bedeutung des Transportsystems“ in die Hauptzeile zu setzen.| These | Hauptpositionierung | Ob eine Verkehrssystemerzählung verwendet werden soll | Aktuelle Schreibkalibrierung |
|------|--------|------|--------------|
| Papier A: KAT-MARL-Konfliktlösung | T-ITS / Verkehrssicherheitskontrolle in geringer Höhe | Ja, aber der Algorithmus kann nicht geschwächt werden | Geändert von „Neuer MARL-Algorithmus“ zu „Überprüfung eines Konfliktlösungssystems in geringer Höhe bei Kommunikationsverschlechterung, nicht kooperativem UAV, Korridor mit hoher Dichte“ |
| Papier B: Dreischichtige Planung von Hunderten von UAVs | TR-C zuerst | Starkes Bedürfnis | Konzentrieren Sie sich auf Kapazität, Verzögerung, Warteschlangenstabilität, Vertiport-/Lade-/Korridor-Engpass und multimodalen Fallback |
| Papier C: FIM-3DGS Active Sensing | Algorithmus + Verkehrsermöglichungstechnologie | Bedingt erforderlich | Wenn Sie für T-ITS/TR-C stimmen, müssen Sie nachweisen, dass die aktive Erfassung die Indikatoren für Verkehrsaufgaben wie Inspektion, Notfallreaktion und Verteilung verbessert; andernfalls bewahren Sie das Papier zum Roboterwahrnehmungsalgorithmus | auf
| Papier D: Semantische Funktionsbereichsplanung | TR-C / Städtische Tieflandplanung | Brauchen | Fokus auf ODD, städtische Funktionsbereiche, Risikoexposition, Planungsvorschläge, nicht reine semantische Segmentierung |
| Papier E: VERA-UAV | AAAI / Formale Sprachplanung | Keine Zwangsanwendung | Befolgen Sie zunächst das KI-Planungs-/Verifizierungspapier. Nachfolge-ITS-Ausbau plus Verkehrsbetriebsszenarien |
| Papier F: Szenarioabdeckung und Notfall | T-ITS + TR-C-Bifurkation | F-J1 wird teilweise benötigt, F-J2 wird dringend benötigt | F-J1 schreibt Benchmark für Sicherheitstests; F-J2 schreibt ein Papier zum Verkehrsbetrieb über die Zuweisung von Notfallressourcen auf der Shandong-Schnellstraße |
| Papier G/G1: Low-Altitude Traffic Cloud Brain LLM Agent | AAAI/IJCAI zuerst, T-ITS-Erweiterung | G1 ist nicht obligatorisch, Journalerweiterung ist erforderlich | G1 behält den Beitrag des Agenten/der Toolnutzung/der Verifizierungsmethode bei; Zeitschriftenversion, Ergänzungssystem, Indikatoren und betriebliche Inspiration |

Auch die minimalen experimentellen Härteanforderungen für die Transportzapfenversion wurden einheitlich erhöht:- Mindestens 5 zufällige Seeds, die Haupttabelle gibt den Mittelwert ± Standard oder Bootstrap-Konfidenzintervall an.
- Baseline darf nicht einfach nur „keine Kontrolle/gierig“ sein, sondern muss starke klassische Methoden, heuristische Methoden und Lernmethoden im Problemfeld umfassen.
- Indikatoren können nicht nur Belohnung, Genauigkeit und Erfolgsquote melden; Sie müssen Verkehrsbedeutungsindikatoren wie Konfliktanzahl, LoWC, NMAC, Verzögerung, zusätzliche Entfernung, Energie, Durchsatz, Ressourcennutzung und Laufzeit enthalten.
- Es muss eine Verallgemeinerung durchgeführt werden: Trainieren Sie niedrige Dichte und testen Sie hohe Dichte, trainieren Sie kleinen Maßstab und testen Sie großen Maßstab, trainieren Sie feste Topologie und testen Sie neue Topologie, trainieren Sie kooperativen Verkehr und testen Sie nicht kooperativen/beeinträchtigten Kommunikationsverkehr.
- Es muss eine Fehlerfallanalyse vorliegen, aus der hervorgeht, bei welcher Dichte, Kommunikationsverlustrate, nicht kooperativem Verhalten oder Ressourcenengpass das System ausgefallen ist.

---

### 0,5 23.05.2026 Organisation: Lesereihenfolge und Priorität der aktuellen Planungsdokumente

Die aktuelle allgemeine Roadmap wird als „Research Matrix Entry“ beibehalten und die spezifische Umsetzung unterliegt dem B/E/F/G/G1-Sonderdokument. Die empfohlene Lesereihenfolge ist wie folgt:| Priorität | Dokumentation | Aktuelle Positionierung | Aktuelle Aktionen |
|--------|------|----------|----------|
| P0 | Papier G1: CloudBrain-Agent vollständiger Abschlussarbeitsplan | AAAI / IJCAI zuerst | Implementieren Sie zunächst den überprüfbaren Agenten, CloudBrain-Bench, die Toolkette und das Hauptexperiment |
| P1 | Papier B: Dreischichtige hierarchische Planung von Hunderten von UAVs | TR-C zuerst | Erstellen Sie einen synthetischen Warteschlangen-Benchmark, einen Lyapunov-Scheduler und eine starke Baseline |
| P1 | Papier F: Entwicklung sicherheitskritischer UAV-Szenarien | T-ITS zuerst, TR-C-Notfallerweiterung | Schließen Sie den ersten F-J1 ab: Abdeckungsmetrik + beschleunigte Tests |
| P2 | Papier E: VERA-UAV | AAAI-Methodenpapier, anschließende Erweiterung von T-ITS | Verdichtet in getippte IR + LTL/STL + Verifier-Reparatur, kein Hauptpapier zum Transportsystem zuerst |
| P3 | Papier C / Papier D | Bis zur weiteren Daten- und Aufgabenkonvergenz | Behalten Sie die Richtung bei, aber konkurrieren Sie nicht mit B/F/G1 um aktuelle experimentelle Ressourcen |

Diese Ausgabe bedarf einer besonderen Klarstellung: **Die alte Zeile „Paper F = CARLA-SUMO Multi-Agent Lane Changing RL“ wird nicht mehr in der aktuellen Gruppe der Low-Altitude-UAV-Papiere gezählt. ** Wenn die Richtung des autonomen Bodenfahrens in Zukunft erneuert wird, kann sie als unabhängiges Bodentransportpapier wiederhergestellt werden; Derzeit bezieht sich Papier F speziell auf die Entwicklung sicherheitskritischer UAV-Szenarien.

Die empfohlene Ausführungsreihenfolge in naher Zukunft ist:

1. Führen Sie zuerst G1 durch, da dadurch der Planer von Papier B, der Verifizierer von Papier E und der Szenario-Stresstest von Papier F in einer Toolkette für das „Cloud-Gehirn für den Verkehr in geringer Höhe“ vereint werden können.
2. Starten Sie gleichzeitig den synthetischen Benchmark von B, da dieser die Kernbasis für nachfolgende TR-C-Systempapiere und G1-Planungstools darstellt.
3. F-J1 schreitet voran, nachdem Erkundungsprotokolle und Szenengenerierungsskripte vorliegen, um zu vermeiden, dass man zu Beginn in zu viele reale Anwendungserzählungen verfällt.
4. E Bewahren Sie AAAI-Methodenpapiere auf und erweitern Sie sie nicht im Voraus zu einem großen System von Tiefflug-Transportjournalen.

---## 1. Panoramakarte für Bloginhalte (im Einklang mit Version 1)

Die drei Hauptforschungslinien bleiben unverändert (Einzelheiten siehe Version 1):
- Hauptzeile eins: Pfadplanung × Konfliktlösung × Planung für mehrere Maschinen
- Hauptlinie 2: Wahrnehmung × Umgebungsrekonstruktion × Digitaler Zwilling
- Hauptzeile 3: LLM/VLM × Semantische Planung × Formale Verifizierung

---

## 2. Stufe 1: Top-Zeitschriftenbeiträge (innerhalb von 24 Monaten)

### Papier A: Groß angelegte urbane UAV-Konfliktlösung – KAT-MARL (Aufrechterhaltung der Top-Themenpositionierung)

**Zieljournal:** IEEE Transactions on Intelligent Transportation Systems (T-ITS, IF 8.5 Q1)

**Änderungen gegenüber Version 1:** Upgrade des experimentellen Maßstabs, Erweiterung der theoretischen Analyse

#### v2 neue Anforderungen

- Experimentgröße von 100 UAV → **200 UAV** (um der Präferenz von T-ITS für groß angelegte Simulationen gerecht zu werden)
- **Analyse der Warteschlangentheorie** hinzugefügt: Beweisen Sie die Obergrenze des Systemdurchsatzes des KAT-Frameworks
- **reale Straßennetzkartierung** hinzugefügt: von der CBD-Simulation auf 2–3 reale Städte erweitert (Shanghai Lujiazui, Beijing CBD, Shenzhen Futian)
- **Robustheitsexperimente** hinzugefügt: Kommunikationsverzögerung, Sensorrauschen, UAV-Fehlerszenarien

#### v2 Zeitleiste
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

### Papier B: Dreistufige hierarchische Planung von Hunderten von Drohnen (Aufrechterhaltung der Top-Positionierung der Veröffentlichung)

**Zielzeitschrift:** Transportation Research Teil C oder IEEE T-ITS (IF 8.5 Q1)

**Änderungen gegenüber Version 1:** Fügen Sie die mathematische Grundlage der Warteschlangentheorie hinzu und fügen Sie multimodale Transportszenarien hinzu

#### v2 neue Anforderungen

- **Theoretische Verbesserungen:** Warteschlangentheorie + Lyapunov-Stabilitätsnachweis
- **Multimodale Erweiterung:** Gemeinsamer Versand von UAV und Bodenfahrzeug (verbessert die Eignung von TR-C für Transportsysteme)
- **Echte Daten:** Vergleich mit Daten unbemannter Lieferpiloten von Meituan/JD (falls verfügbar)

#### v2 Zeitleiste
```
2026/08–09  三层框架代码实现
2026/10–12  规模扩展实验（20/50/100/200 UAV）
2027/01     排队论与 Lyapunov 分析
2027/02–03  写稿
2027/04     ◉ 投稿 TR Part C
2027/10     接受目标
```

---

### Papier C: Aktive FIM-3DGS-Erkennung – **Umfassende Rekonstruktion (Einzelheiten siehe spezielles Dokument zu Version 2)**

**Zieljournal:** IEEE T-ITS (bevorzugt) → TR Teil C (bevorzugt), IF 8.5 Q1**Grund für die Rekonstruktion:** v1-Positionierung RA-L ist zu niedrig, der Lehrer verlangt Top-Veröffentlichung

**v2-Kernänderungen (siehe „paper-c-fim-3dgs-uav-active-perception_v2_20260515.md“ für Details): **

1. **Positionierungs-Upgrade:** Aus „Perception Algorithm Paper“ → „Low Altitude Economic Enabling Technology“
2. **Bewertungserweiterung:** Einzelner Wahrnehmungsindikator → fünfschichtiges Indikatorensystem (Wahrnehmung/Planung/Aufgabe/System/Wirtschaft)
3. **Fallstudie:** Drei neue Transportanwendungsfälle (Gebäudeinspektion, Zustellung auf der letzten Meile, Notfallmaßnahmen)
4. **Experimentelle Erweiterung:** Gemeinsame SUMO + AirSim-Simulation + Multi-UAV-Experiment auf Systemebene hinzugefügt
5. **Beitrag zum Datensatz:** Selbst erstellter Open-Source-Datensatz UAV-Delivery-Dataset

#### v2 Zeitleiste
```
2026/06–10  五阶段实验（核心算法 + 三案例 + 多机系统级）
2026/11–12  数据整合 + 初稿（22 页 T-ITS 格式）
2027/01–02  润色 + 内部审阅
2027/03     ◉ 投稿 IEEE T-ITS
2027/09     收到审稿意见
2027/12     接受 / 转 TR-C
2028/06     最终发表
```

Eine detaillierte Planung für Paper C finden Sie im [Paper C v2-Sonderdokument](/blog/paper-c-fim-3dgs-uav-active-perception_v2_20260515/).

---

### Papier D: Semantische Fusion aus mehreren Quellen + funktionale, partitionsgesteuerte UAV-Trajektorienplanung (Aufrechterhaltung der Top-Positionierung der Veröffentlichung)

**Zielzeitschrift:** Transportforschung Teil C (IF 8.5 Q1)

**Änderungen gegenüber Version 1:** Erweiterung des Generalisierungsexperiments für mehrere Städte

#### v2 neue Anforderungen

- **Verallgemeinerung auf mehrere Städte:** Training + Tests in 5 Städten (Peking, Shanghai, Guangzhou, Shenzhen, Wuhan)
- **Realer Flugfall:** Zusammenarbeit mit einem UAV-Lieferpiloten oder öffentliche Datenreproduktion
- **Risikoquantifizierung:** Einführung einer versicherungsmathematischen Risikobewertung (Versicherungs-/Entschädigungsperspektive)

#### v2 Zeitleiste
```
2026/07–09  GIS 数据采集（5 城市）
2026/10–12  功能分区模型 + 多城市实验
2027/01     真实飞行案例对比
2027/02–03  写稿
2027/04     ◉ 投稿 TR Part C
2027/10     接受目标
```

---

## 3. Stufe 2: Top-Zeitschriftenbeiträge mit größeren technischen Herausforderungen

### Papier E: Formale VERA-UAV-Sprachplanung (zuerst AAAI, dann ITS-Erweiterung)

**v1-Ziel:** ICRA/IJCAI (Konferenz)

**Aktuelles Ziel:** AAAI/IJCAI zuerst, Sicherung der T-ITS-Erweiterung**Grund für die Kalibrierung:** Der Kernbeitrag von Paper E ist die KI-Planung/-Verifizierung, und es sollte nicht gezwungen werden, ein großes und verstreutes Transportsystempapier zu werden, um eine Spitzenpublikation zu erreichen. Die AAAI-Version priorisiert die Antwort „Wie UAV-Aufgaben in natürlicher Sprache ausführbare sichere Flugbahnen über typisierte IR, LTL/STL, Validator-Gegenbeispiele und symbolische Fallbacks bilden.“

#### Aktuelle Schließrichtung

- **Hauptzeile der Methode:** NL-Anweisung -> typisiertes TaskIR -> LTL/STL -> Verifizierer -> Gegenbeispiel/Robustheitsreparatur -> Trajektorienüberprüfung.
- **Theoretische Grenzen:** Erhebt keinen Anspruch auf LLM-Vollständigkeit; Beweisen Sie die relative Vollständigkeit unter der Annahme eines endlichen DSL, eines entscheidbaren Prüfers und eines vollständigen zugrunde liegenden Planers.
- **Experimentgrenze:** Das Hauptexperiment verwendet einen synthetischen kontrollierten Benchmark; AirSim, reale Logistik und Multi-UAV-ITS-Indikatoren werden später erweitert.
- **Einreichungsstrategie:** Der Hauptartikel von AAAI betont Methoden, Theorien, Benchmarks und starke Grundlagen; T-ITS wird um Verkehrsbetriebsindikatoren und reale Tiefflugszenarien erweitert.

#### v2 Zeitleiste
```
2026/06–07  冻结 TaskIR DSL、任务生成器和验证器接口
2026/08–09  实现 Direct LLM / NL2LTL-style / LTLCodeGen-style / VERA-UAV baselines
2026/10     跑主实验、消融和泛化测试
2026/11     完成理论证明、图表和初稿
2026/12     ◉ 投稿 AAAI / IJCAI 对应批次
2027/03     根据结果扩展 T-ITS 版本
```

---

### Papier F: UAV-Sicherheitsszenariotechnik und Notfallanwendungen (ersetzt die alte CARLA-SUMO-Linie)

**Aktuelles Ziel:** F-J1 ist der Hauptkandidat für IEEE T-ITS; F-J2 ist der Hauptkandidat für TR-C.

**Positionierungsänderung:** Das aktuelle Papier F bezieht sich nicht mehr auf den CARLA-SUMO-Spurwechsel-RL, sondern konzentriert sich auf die Entwicklung sicherheitskritischer UAV-Szenarien als Zeitschriftenpriorität: Erstellen Sie zunächst eine reproduzierbare Abdeckung sicherheitskritischer Szenarien und beschleunigte Testpapiere und erweitern Sie dann dieselbe Plattform auf den Einsatz von Notfallrettungsressourcen auf der Autobahn Shandong.

#### Aktuelle neue Anforderungen- **Szenenraum:** Definieren Sie klar die 50 x 50 x 50 m große UAV-Testzelle, die Hinderniskombination, die dynamischen Hindernisse, das Windfeld, die Sichtbereichsverdeckung, die Flugverbotszone und die Missionsziele.
- **Vorhandene experimentelle Vermögenswerte:** 76 Millionen Explorationsprotokolle können nur als „verfügbare Basis“ und nicht als endgültige experimentelle Ergebnisse geschrieben werden; Sie müssen in Fehlertaxonomie, Abdeckungslücken und Planer-Stressfälle bereinigt werden.
- **Hauptzeile der Methode:** Abdeckungsmetrik -> Abdeckungsgesteuerter Sampler -> Gefahrenvaliditätsfilter -> beschleunigtes Testen -> planerübergreifende Bewertung.
- **Starke Basislinie:** Zufallsgenerierung, Raster-/LHS-Stichprobe, Bayes'sche Optimierung, CMA-ES, kontradiktorische RL-Generierung, eingeschränkte Generierung im szenischen Stil.
- **Verkehrsausweitung:** F-J2 führte nur den Shandong Highway Emergency ein und konzentrierte sich auf Unfallerkennung, UAV-Aufklärung, Einsatz von Bodenressourcen, Reaktionszeit und Verkehrswiederherstellung.

#### v2 Zeitleiste
```
2026/06–07  整理 7600 万次探索日志，冻结场景空间和 coverage metric
2026/08–10  实现 accelerated testing 与强 baseline
2026/11     cross-planner evaluation、failure taxonomy、统计检验
2026/12–2027/01  写 F-J1 初稿
2027/02     ◉ 投稿 IEEE T-ITS
2027/03–06  扩展山东高速应急资源调配 F-J2
```

---

## 4. Insgesamt 30-monatige Einreichungs-Roadmap für Top-Themen

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

**Kernrhythmus:**
- **Zweites Halbjahr 2026:** G1 / E / F-J1 bilden den ersten Stapel ausführbarer Experimente, um zu vermeiden, dass alle Arbeiten gleichzeitig auf das Frühjahr 2027 verschoben werden.
- **Frühjahr 2027:** A/B/C/D werden weiterhin als Hauptlinie systematischer Beiträge in der Top-Ausgabe beworben.
- **Erste Jahreshälfte 2027:** F-J2 unterscheidet sich von der F-J1-Plattform in eine TR-C-Version mit Hochgeschwindigkeits-Notfallressourcenbereitstellung.
- **H1 2028:** Haupteinnahmezeitraum.

---

## 5. Detaillierte Erläuterung der Top-Journal-Matrix| Tagebuch | Feld | WENN | Akzeptanzrate | Überprüfungszyklus | v2-Anpassungspapier |
|------|------|-----|--------|---------|--------------|
| **IEEE T-ITS** | ITS General | 8,5 | ~20 % | 4–6 Monate | A-, C-, F-J1-, G/G1-Journalerweiterungen |
| **TR Teil C** | Neue Transporttechnologien | 8,5 | ~18% | 4–6 Monate | B, D, F-J2 |
| **IEEE T-RO** | Robotik | 7,4 | ~25 % | 6–10 Monate | C-Vorbereitung |
| **TR Teil B** | Transportmethodik | 6,0 | ~15% | 6–8 Monate | B-Vorbereitung |
| **Verkehrswissenschaft** | Verkehrswissenschaft | 5,4 | ~12% | 6–10 Monate | B-Investition |

**V2-Einreichungsmatrix-Prinzipien:**
- **Q1 mit IF ≥ 8 bevorzugt** (T-ITS, TR-C)
- **Bereiten Sie sich auf die Investition in Q1** (T-RO) desselben Fonds mit IF ≥ 7 vor
- **Zeitschriften mit IF < 7 werden nicht mehr berücksichtigt**

---

## 6. Risikobewertung und Alternativen

### 6.1 Hauptrisiken der Top-Publikationsstrategie

**Risiko 1: Der Überprüfungszeitraum überschreitet das Promotionsfenster**
- Die erste Überprüfungsrunde für die Top-Ausgabe findet von April bis Juni statt und Überarbeitungen können sich um mehr als 12 Monate verzögern
- **Antwort:** Zentralisierte Einreichung im Frühjahr 2027, wobei 12 Monate für Überarbeitungen reserviert sind
- **Fazit:** Mindestens 2 Artikel akzeptiert, der Rest kann mit dem Status „eingereicht/in Prüfung“ bewertet werden

**Risiko 2: Übermäßige experimentelle Arbeitsbelastung**
- Die Gesamtarbeitsbelastung von Version 2 beträgt etwa 50–60 Monate (falls seriell), was eine Arbeitsteilung zwischen Team und Zusammenarbeit erfordert
- **Antwort:** Priorisieren Sie G1/B/F-J1/E in naher Zukunft und behalten Sie nur Konzept- und Dateneingaben in andere Richtungen bei, um eine Ressourcenverwässerung zu vermeiden

**Risiko 3: Zeitverlust beim Wechsel nach Ablehnung**
- Eine Runde Ablehnung + Transfer = ca. 6 Monate Verlust
- **Antwort:** Bereiten Sie das TR-C/T-ITS-Dual-Framing im Anschreiben im Voraus vor

### 6.2 Priorität alternativer Einreichungen| Papier | Erste Wahl | Alternative 1 | Alternative 2 |
|-------|------|------|------|
| A | T-ITS | TR Teil C | IEEE T-Cyber ​​|
| B | TR Teil C | T-ITS | TR Teil B |
| C | T-ITS | TR Teil C | IEEE T-RO |
| D | TR Teil C | T-ITS | TR Teil D (Umwelt) |
| E | AAAI/IJCAI | T-ITS | IEEE T-SMC |
| F | T-ITS | TR Teil C | T-ASE / T-RO |

---

## 7. Zusammenfassung des Berichts an den Lehrer in einem Satz

> „Die aktuelle Papiergruppe wurde in die Hauptlinie UAV für niedrige Flughöhen/Transport in niedriger Flughöhe umstrukturiert. Cloud Brain: G1 ist der erste, der in AAAI/IJCAI investiert, B ist die Hauptinvestition in TR-C, F-J1 ist die Hauptinvestition in T-ITS, E verwaltet AAAI-Methodenpapiere und reserviert die T-ITS-Erweiterung. Transportzeitschriftenpapiere müssen durch Systemprobleme, mathematische Modelle, starke Basislinien, Verkehrsindikatoren und Fehleranalysen unterstützt werden und dürfen sich nicht mehr nur auf die Richtung verlassen Erzählungen.

---

## 8. Anweisungen zur Dokumentverarbeitung v1

- **v1 (`research-roadmap_v1_20260515.md`):** Reserviert als historisches Archiv zur Aufzeichnung des Designs der „Rapid Publishing Hybrid Strategy“
- **v2 (dieses Dokument):** Das derzeit gültige Planungsdokument
- **Auslösebedingungen für das nächste Update:** ① Vervollständigen Sie die experimentellen Daten von Papier A. ② Erhalten Sie die ersten Überprüfungskommentare. ③ Der Lehrer passt die Richtung an

---

**Anhang: Korrespondenz zwischen Blogbeiträgen und Paper (im Einklang mit Version 1)**| Blogbeitrag | Entsprechendes Papier |
|---------|----------|
| Marl-Kat-UAV-Konflikt | A (Haupt) |
| UAV-Konfliktlösung | A (Referenz) |
| uav-conflict-env-construction | A (experimentelle Umgebung) |
| Großraum-UAV-Planung | B (Haupt) |
| uav-urban-routenplanung | B (Referenz) |
| next-best-view-nerf-3dgs-exploration | C (Haupt) |
| Informationstheorie-aktive-Wahrnehmung | C (theoretische Basis) |
| uav-nerf-gs-planung | C (Referenz) |
| **paper-c-fim-3dgs-uav-active-perception_v2_20260515** | **C Spezialplanung (v2)** |
| UAV-Semantic-Mapping-Functional-Zoning | D (Haupt) |
| uav-digital-twin-semantic-mapping | D (Referenz) |
| llm-uav-semantische-planung | E (Haupt) |
| llm-guided-uav-planung-frontiers | E (Referenz) |
| paper-b-hierarchical-uav-scheduling-trc-plan-v1-20260519 | B Sonderplanung |
| paper-e-vera-uav-experiment-taskbook-v1-20260517 | E-Sonderaufgabenbuch |
| paper-f-uav-scenario-coverage-journal-roadmap-v2-20260520 | F Sonderplanung |
| paper-g-low-altitude-cloud-brain-llm-roadmap-v1-20260520 | G Gesamtstrecke |
| paper-g1-cloudbrain-agent-full-paper-plan-v1-20260520 |G1 erster vollständiger Thesenvorschlag |
| carla-sumo-rl-spurwechsel | Alte F-Linie, derzeit nicht in der Papiergruppe für UAVs in geringer Höhe enthalten |