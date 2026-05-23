---
title: "LLM-gesteuerte UAV-Missionsplanung: die Grenze von der Schlussfolgerung bis zur Ausführung"
description: "Eingehende Analyse der drei Hauptparadigmen von LLM für die UAV-Missionsplanung: LLM als Planer, LLM+PDDL-Symbolplanung und LLM+RAG, einschließlich modernster Arbeiten wie VoxPoser, ActiveGAMER und Dual-Process-Architektur."
tags: ["UAV", "LLM", "Missionsplanung", "PDDL", "verkörperte Intelligenz", "Ende-zu-Ende"]
category: "Tech"
pubDate: 2026-04-27
---

# LLM-gesteuerte UAV-Missionsplanung: Die Grenze von der Schlussfolgerung bis zur Ausführung

> **UAV Intelligent Series · Kapitel X+1**
> Spotlight: LLM als Missionsplaner, symbolische Planungsintegration, Echtzeit-Inferenzarchitektur

---

## 1. Warum ist LLM für die Planung von UAV-Missionen geeignet?

Die Herausforderung bei der Planung von UAV-Missionen liegt in der **Unsicherheit der offenen Welt**:

```
传统规划（基于模型）：
输入：精确目标状态 + 精确环境模型
输出：最优动作序列
局限：模型不准就崩溃，无法处理语言目标

LLM 规划（基于知识）：
输入：自然语言指令 + 视觉观测 + 世界知识
输出：可执行动作序列
优势：泛化性强、零样本理解新任务
```

Vorteile von LLM:
- **Weltwissen**: Das Vortraining enthält umfangreiches physikalisches Wissen („Wasser fließt“, „Autos sind schneller als Menschen“)
- **Zero-Shot-Inferenz**: Es ist nicht erforderlich, für jede Aufgabe separat zu trainieren
- **Mehrstufige Planung**: Komplexe Aufgaben in Teilzielketten zerlegen (Chain-of-Thought)

---

## 2. LLMs Paradigma für die Aufgabenplanung

### 2.1 Paradigma 1: LLM als Planer (Aktionen direkt ausgeben)

**Repräsentative Arbeit:**

**ReAct (Begründung + Handeln)**
- Kerngedanke: LLM wechselt „Argumentation“ und „Handlung“ ab
- Jeder Schritt: „obs → think → action → next_obs“.
- Anwendbar auf: Szenarien mit beobachtbarem Status und klarem Umgebungsfeedback
- Anpassung an UAV: erfordert schnelles Handeln→OBS-Schleife

**SayCan (PaLM-SayCan, 2022)**
- Kombinieren Sie die „Fähigkeitsbeschreibung“ von LLM mit der physischen „Machbarkeit“.
- Der Roboter sagt „was er tun kann“ und der LLM entscheidet, „was er tun soll“
- **Erleuchtung:** UAV kann undurchführbare Aktionen basierend auf seinem eigenen Status (Leistung, Flugbeschränkungen) filtern.

---

### 2.2 Paradigma 2: LLM + PDDL-Symbolplanung

**PDDL (Planning Domain Definition Language)** ist eine klassische Roboter-Aufgabenplanungssprache, die Aufgaben als diskrete symbolische Probleme modelliert.

**Kernidee:**
```
VLM 感知 → PDDL problem 生成 → 经典规划器 → UAV 动作序列
```

**Vorteile:**
- Planungsergebnisse können erläutert und überprüft werden
- Mathematischer Beweis, um die Erledigung der Aufgabe sicherzustellen
- Geeignet für sicherheitskritische Szenarien (städtische Luftraumflüge)

**Herausforderung:**
- Die PDDL-Modellierung selbst ist ein Flaschenhals (erfordert Domänenexperten)
- Die kontinuierliche Dynamik von UAVs ist nicht vollständig mit den diskreten Annahmen von PDDL kompatibel
- **Lösungsidee:** PDDL übernimmt die Aufgabenzerlegung auf hoher Ebene, MPC übernimmt die Trajektorienausführung auf niedriger Ebene

---### 2.3 Paradigma 3: LLM + RAG (Retrieval Enhanced Generation)

**GenerativeMPC (arXiv, 2026)**

**Artikel:** *GenerativeMPC: VLM-RAG-gesteuerter Ganzkörper-MPC mit virtueller Impedanz für bimanuelle mobile Manipulation*
**Autor:** Marcelino Julio Fernando et al.
**Quelle:** arXiv, April 2026

**Kernidee:**
```
VLM 感知当前场景 → 检索相关操作知识库 → RAG 生成操作建议 → MPC 执行
```

**Schlüsseltechnologie:**
1. **Wissensabruf**: Rufen Sie Beispiele ab, die für das aktuelle Szenario am relevantesten sind, aus der betrieblichen Wissensdatenbank (einschließlich Erfahrungsdaten zur Robotersteuerung).
2. **Virtuelle Impedanz**: Generieren Sie Compliance-Kontrollparameter, um starre Kollisionen zu vermeiden
3. **RAG-Filterung**: Stellen Sie sicher, dass die LLM-Ausgabe physisch ausführbar ist

**Anpassung an UAV:**
- Suchen Sie nach Bauvorschriften (Höhenbeschränkungen, Flugverbotszonen)
- Historische Missionserfahrungen abrufen (Flugparameter unter ähnlichen Wetterbedingungen)
- Sicherheitsprotokolle abrufen (Mindestabstand zur Hindernisvermeidung, Notfallmaßnahmen)

---

## 3. Echtzeit-Argumentationsarchitektur

### 3.1 Dual-Prozess-Architektur (arXiv, 2026)

**Artikel:** *Eine Dual-Prozess-Architektur für Echtzeit-VLM-basierte Indoor-Navigation*
**Autor:** Joonhee Lee, Hyunseung Shin, Jeonggil Ko
**Quelle:** arXiv:2601.19401, Januar 2026

**Kerndesign:**

```
┌─────────────────────────────────────────────┐
│           System Architecture               │
│                                             │
│  Process 1 (Slow): VLM Reasoning Thread     │
│  ┌─────────────────────────────────────┐   │
│  │ VLM: "What should I do next?"       │   │
│  │ Frequency: ~0.2-1 Hz                 │   │
│  │ Output: Navigation goal / decision  │   │
│  └─────────────────────────────────────┘   │
│              ↓ goal                        │
│  Process 2 (Fast): Control Execution Thread│
│  ┌─────────────────────────────────────┐   │
│  │ MPC: Track trajectory to goal        │   │
│  │ Frequency: ~100 Hz                   │   │
│  │ Output: Motor control signals        │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

**Designprinzipien:**
- **Quick Process** (MPC): Reaktion auf Millisekundenebene, Verarbeitung der Hindernisvermeidung in Echtzeit
- **Slow Process** (VLM): Argumentation der zweiten Ebene, Verarbeitung von Entscheidungen auf hoher Ebene
- **Entkopplung kritisch**: VLM befindet sich nicht auf dem kritischen Pfad und hat keinen Einfluss auf die Steuerfrequenz

---

### 3.2 Hierarchischer Planungsrahmen

**Hohe Ebene (LLM/VLM, zweite Ebene): **
```
任务理解 → 子目标分解 → 全局路径规划 → 授权低层执行
```

**Mittlere Schicht (differenzierbare Optimierung, 100-ms-Ebene): **
```
RRT*/MPC → 局部路径重规划 → 平滑轨迹生成
```**Untere Schicht (PID/MPC, Millisekundenebene): **
```
姿态控制 → 电机分配 → 执行
```

---

## 4. Tiefe des Schlüsselalgorithmus

### 4.1 VoxPoser: Synthetische 3D-Wertekarte von LLM

**Artikel:** *VoxPoser: Zusammensetzbare 3D-Wertekarten für die Robotermanipulation mit Sprachmodellen*
**Autor:** Wenlong Huang, Chen Wang, Ruohan Zhang, Yunzhu Li, Jiajun Wu, Li Fei-Fei
**Quelle:** arXiv:2307.05973, Juli 2023

**Kernbeitrag:**
- LLM-Ausgabe **räumliche 3D-Wärmekarte** (zusammensetzbare 3D-Wertekarte)
- Heatmap-Kodierung „Wohin gehen“ und „Was man meiden sollte“
- Wird direkt als Belohnungsfunktion zur Flugbahnoptimierung verwendet

**Erweiterung auf UAV:**
- VLM gibt eine 3D-Belegungs-Heatmap aus
- Heatmap-gesteuerte MPC-Kostenfunktion
- VoxPoser für UAV = „3D-Raumangebot aus Sprache“

**Hinweis:** VoxPoser wurde auf arXiv veröffentlicht. Bisher wurden keine eindeutigen Aufzeichnungen über Konferenzpublikationen gefunden.

---

### 4.2 CoNVO (Conditional Neural Value Optimization)

Kombinieren Sie LLM-Planung mit Werteiteration:
- LLM bietet **vorherige Präferenzen** (welche Aktionen sinnvoller sind)
- Wertiteration bietet **Optimalitätsgarantie**
- Robuster als reine LLM-Planung und flexibler als reine Planung

---

## 5. Weltmodellgestützte Planung

### 5.1 Warum Weltmodell?

Das Wissen über das LLM ist statisch, aber die UAV-Umgebung ist dynamisch:
- Der Wind wird sich ändern
- Hindernisse werden sich bewegen
- GNSS-Signale können driften

Das Weltmodell ermöglicht es UAVs, **die Zukunft vorherzusagen**: 
```
当前状态 + 动作 → 世界模型 → 预测未来状态序列
LLM 在预测的未来状态序列上做规划（Plan over imagined futures）
```

### 5.2 Papiervertreter**Dreamer-Serie** (Daniel Hafner, Jürg Widmer, etc.)
- Basierend auf dem dynamischen RSSM-Modell
- Machen Sie verstärkendes Lernen über die imaginäre Zukunft
- Verifiziert an Robotern (Roboterarme, unbemannte Fahrzeuge)

**VMP (Video-Bewegungsplanung)**
- Verwenden Sie Videogenerierungsmodelle für die Bewegungsplanung
- Zukünftige Frames generieren → Bewegungsvektoren extrahieren → UAV steuern

---

## 6. Sicherheit und Authentifizierung

### 6.1 Warum Sicherheit der Schlüssel ist

Wenn UAVs in Städten fliegen, kann eine schlechte Entscheidungsfindung zu **menschlichen Verlusten** führen. Es besteht ein grundlegender Widerspruch zwischen dem probabilistischen Ergebnis von LLM und den deterministischen Garantien, die für die Flugsicherheit erforderlich sind.

### 6.2 Sicherheitsrahmen

**CBF (Kontrollbarrierenfunktionen):**
- ASMA führt CBF in UAV VLN ein
- Stellen Sie sicher, dass der unsichere Zustand niemals erreichbar ist

**Formelle Verifizierung:**
- Verwenden Sie TLA+ / NuSMV zur Überprüfung der Zustandsmaschine
- Die Ergebnisse der LLM-Planung werden nach der Modellüberprüfung ausgeführt

**Abschirmung:**
- Unterschichtschutz (Shield): Überwacht die LLM-Ausgabe und fängt unsichere Aktionen ab
- LLM der oberen Ebene: Konzentrieren Sie sich auf die Aufgabenerfüllung und berücksichtigen Sie keine Sicherheitsdetails
- **Autonomes Fahren ähnliche „Schutzengel“-Architektur**

---

## 7. Grenz-Hotspots und zukünftige Richtungen

### 7.1 End-to-End-VLA (Vision-Language-Action)

**Neuester Trend:** Überspringen Sie das hierarchische Design von „Erkennung → Planung → Steuerung“ und geben Sie **Aktionstoken** direkt aus VLM aus.

Repräsentative Arbeit:
- **RT-2** (Google Robotics): Passen Sie die Ausgabeaktion von VLM direkt an
- **π₀** (Physische Intelligenz): VLA für humanoide Roboter
- **UAV-Version** (im Entstehen begriffen): Ähnliche Ideen werden auf Drohnen angewendet

**Herausforderung:**
- Kontinuität des Handlungsraums vs. Diskretion der Sprache
- Schwierigkeiten bei der Sicherheitsüberprüfung (End-to-End-Blackbox)
- Datenknappheit (erfordert umfangreiche Roboter-Teleoperationsdaten)

### 7.2 Kollaborative LLM-Planung mit mehreren Maschinen

**SysNav (arXiv, März 2026)****Artikel:** *SysNav: Mehrstufige systematische Zusammenarbeit ermöglicht reale, verkörperungsübergreifende Objektnavigation*
**Autor:** Haokun Zhu et al.
**Quelle:** arXiv:2603.xxxxx, März 2026

**Kernbeitrag:**
- Kollaborative Navigation mit mehreren Agenten über verschiedene Roboterplattformen hinweg
- LLM übernimmt die Koordination auf hoher Ebene (wer geht in welchen Bereich)
- Verteilte Wahrnehmungsfusion (jeder Agent teilt die Vision)

### 7.3 Physische Intelligenz × UAV

- **Grundlagenmodelle für Manipulation** → **Grundlagenmodelle für Flug**
- Möglicherweise wird in Zukunft ein spezielles „UAV-Gehirn“-Vortrainingsmodell erscheinen
- Ähnlich wie LLaVA, aber spezialisiert auf dreidimensionales räumliches Denken + Flugdynamik

---

## 8. Zusammenfassung und Vorschläge

| Abmessungen | Aktuelle Besten | Zukünftige Richtungen |
|------|---------|---------|
| Planungsparadigma | Dual-Prozess-Architektur (echtzeitfähig) | End-to-End-VLA (langfristiges Ziel) |
| Weltwissen | RAG (zuverlässig, aber langsam) | Weltmodell (schnell, erfordert aber Schulung) |
| Sicherheit | CBF + Abschirmung | Formale Verifizierung (vollständig garantiert) |
| Edge-Bereitstellung | 4-Bit-LLaVA (kaum Echtzeit) | Spezialchips (NPU/TPU) |

**Tipp für Sie:**
1. **Der schnellste Weg zu Ergebnissen**: Dual-Prozess-Architektur + LLaVA-7B + UAV-Plattform
2. **Der größte Raum für Innovation**: VLM + Sicherheitsüberprüfungs-Framework (das macht derzeit fast niemand)
3. **Langfristiges Layout**: Sammeln Sie Ihre eigenen UAV-Steuerungsdaten und trainieren Sie ein dediziertes VLA-Modell

---

## 📚 Referenzen1. Lee et al. *Eine Dual-Prozess-Architektur für Echtzeit-VLM-basierte Indoor-Navigation*. arXiv:2601.19401, 2026.
2. Fernando et al. *GenerativeMPC: VLM-RAG-gesteuerter Ganzkörper-MPC mit virtueller Impedanz*. arXiv, 2026.
3. Huang et al. *VoxPoser: Zusammensetzbare 3D-Wertekarten für die Robotermanipulation mit Sprachmodellen*. arXiv:2307.05973, 2023.
4. Brohan et al. *RT-2: Vision-Language-Action-Modelle übertragen Webwissen auf Robotersteuerung*. arXiv, 2023.
5. Zhu et al. *SysNav: Mehrstufige systematische Zusammenarbeit ermöglicht reale, verkörperungsübergreifende Objektnavigation*. arXiv, 2026.
6. Ahn et al. *Tu, was ich kann und nicht, was ich sage: Sprache in robotischen Errungenschaften verankern*. arXiv, 2022.