---
title: "Vision-Language-Modelle für die UAV-Navigation: Grundlage und Grenze der Vision-Language-Navigation"
description: "Überblick über das grundlegende Paradigma, die Kernarchitektur und die repräsentative Arbeit der VLM+UAV-Navigation, einschließlich der neuesten Veröffentlichungen wie LogisticsVLN, OmniVLN und ASMA"
tags: ["UAV", "VLM", "Vision-Sprachnavigation", "Multimodales Großmodell", "verkörperte Intelligenz"]
category: "Tech"
pubDate: 2026-04-27
sourceHash: "afcc4f7205fc6b593288c445afbd3bcab294c159"
---

# Vision-Language-Modelle für die UAV-Navigation: Die Grundlage und Grenze der Vision-Language-Navigation

> **UAV Intelligent Series · Teil X**
> Fokus: Grundlegendes Paradigma, Kernarchitektur und repräsentative Arbeit von VLM+UAV

---

## 1. Hintergrund: Von verbalen Befehlen zum autonomen Flug

Die traditionelle UAV-Pfadplanung basiert auf präzisen mathematischen Zielfunktionen (z. B. kürzester Pfad, minimaler Energieverbrauch), aber reale Missionsanweisungen sind oft unscharfe Beschreibungen natürlicher Sprache:

- „Gehen Sie zum Basketballplatz neben dem roten Dach“
- „Folgen Sie dem weißen Transporter und halten Sie einen Abstand von 50 Metern ein“
- „Suchen Sie einen hohen Punkt, von dem aus Sie das Regierungsgebäude der Stadt sehen und schweben können.“

Diese Anweisungen können nicht direkt in mathematische Optimierungsziele umgewandelt werden, sie können jedoch durch VLM (Vision-Language Model) verstanden und begründet werden. Die Vision-Language-Navigation (VLN) ist die zentrale Forschungsrichtung zur Lösung dieses Problems und ermöglicht es Robotern (UAV), im dreidimensionalen physischen Raum gemäß Anweisungen in natürlicher Sprache zu navigieren.

---

## 2. Aufgabenstellung: Kernthemen der VLN

Die VLN-Aufgabe kann wie folgt formalisiert werden:

> Lassen Sie den Agenten anhand einer natürlichsprachlichen Anweisung $I$ und einer anfänglichen visuellen Beobachtung $O_0$ eine Reihe von Aktionen $a_1, a_2, ..., a_T$ ausführen und schließlich die durch die Anweisung beschriebene Zielposition erreichen.

Die wichtigsten Herausforderungen sind:
1. **Semantische Begründung**: räumliche Beziehungen in der Sprache („links“, „hinten“, „oben“) auf den physischen Raum abbilden
2. **Long Horizon Reasoning**: Anweisungen beschreiben oft komplexe mehrstufige Aufgaben
3. **Zero-Sample-Generalisierung**: Unsichtbare Gebäude, Umgebungen und Objekte
4. **Dreidimensionale Eigenschaften**: UAV verfügt im Gegensatz zu Bodenrobotern über vollständige 3D-Bewegungsfähigkeiten

---

## 3. Repräsentative Arbeit

### 3.1 LogisticsVLN: UAV VLN für Terminalverteilung (arXiv, 2025)**Artikel:** *LogisticsVLN: Vision-Language-Navigation für die Terminalzustellung in geringer Höhe auf Basis von Agenten-UAVs*
**Autor:** Xinyuan Zhang, Yonglin Tian, Fei Lin, Yue Liu, Jing Ma, Kornélia Sára Szatmáry, Fei-Yue Wang
**Quelle:** arXiv:2505.03460, Mai 2025

**Kernbeitrag:**
- Das erste VLN-Missionsrahmenwerk, das speziell auf die **Lieferung von UAV-Terminals in geringer Höhe** ausgerichtet ist
- Vorgeschlagene Agenten-UAV-Architektur: Wahrnehmung → Argumentation → Planung → Regelkreis
- Besondere Herausforderungen für städtische Umgebungen in geringer Höhe (Gebäudeverdeckung, dynamische Hindernisse, GNSS-Drift)

**Methodenrahmen:**

```
用户指令："送包裹到红色大门旁边"
    ↓
VLM 语义解析（物体检测 + 空间关系）
    ↓
拓扑地图匹配（检测到的地标 vs 先验地图）
    ↓
路径规划（全局粗规划 + 局部视觉重规划）
    ↓
MPC 控制器执行
```

**Wichtige Erkenntnisse:** Dies ist derzeit die VLN-Arbeit, die den tatsächlichen UAV-Bereitstellungsszenarien am nächsten kommt und das visuelle Sprachmodell auf GPT-4V-Ebene durchgängig mit der physischen Kontrollschicht integriert.

---

### 3.2 OmniVLN: Offenes, plattformübergreifendes, endseitiges VLN (arXiv, 2026)

**Aufsatz:** *OmniVLN: Omnidirektionale 3D-Wahrnehmung und tokeneffizientes LLM-Argument für die visuelle Sprachnavigation über Luft- und Bodenplattformen hinweg*
**Autor:** Zhongyuang Liu, Min He, Shaonan Yu et al.
**Quelle:** arXiv, März 2026

**Kernbeitrag:**
- **Omnidirektionale 3D-Wahrnehmung**: 360° sphärische Sichtfeldwahrnehmung, besser geeignet für komplexe Stadtschluchten als herkömmliche nach vorne gerichtete Kameras
- **Token-effiziente LLM-Inferenz**: Lösen Sie den Engpass bei der Rechenleistung der VLM-Bereitstellung am Edge
- **Plattformübergreifendes einheitliches Framework**: Der gleiche Satz von Algorithmen passt sich sowohl an UAVs als auch an Bodenroboter an**Technologische Innovation:**
1. **3D-Token-Komprimierung**: Codieren Sie räumliche 3D-Informationen in kompakte Token, um die Anzahl der LLM-Eingabe-Tokens zu reduzieren
2. **Dynamische Sichtfeldverwaltung**: Passen Sie den Interessenbereich adaptiv an die Navigationsanforderungen an
3. **Leichtes VLM-Backbone**: Clientseitige Version basierend auf der Qwen-VL- oder LLaVA-Architektur

---

### 3.3 ASMA: Security Boundary-Aware UAV VLN (arXiv, 2024)

**Artikel:** *ASMA: Ein adaptiver Sicherheitsmargenalgorithmus für die Vision-Language-Drohnennavigation über szenenbewusste Kontrollbarrierenfunktionen*
**Quelle:** arXiv, September 2024

**Kernbeitrag:**
- Integrieren Sie explizit **Sicherheitsbeschränkungen** in das VLN-Framework
- Vorgeschlagene szenenbezogene Kontrollbarrierenfunktionen (szenenbezogene Kontrollbarrierenfunktion)
- Sorgen Sie für strenge Sicherheitsbeschränkungen in offenen städtischen Umgebungen

**Warum es wichtig ist:** Die meisten VLN-Bemühungen konzentrieren sich auf die Navigationsgenauigkeit und ignorieren die Sicherheit. ASMA füllt diese Lücke – UAVs können Sicherheitskompromisse zwischen „Anweisungen nicht verstehen“ und „gegen die Wand stoßen“ eingehen.

---

### 3.4 Vision-and-Language-Navigation für UAVs: Überblick (arXiv, 2026)

**Papier:** *Vision-and-Language-Navigation für UAVs: Fortschritte, Herausforderungen und eine Forschungs-Roadmap*
**Autor:** Hanxuan Chen, Jie Zheng, Siqi Yang et al.
**Quelle:** arXiv:2604.xxxxx, April 2026

**Übersichtsabdeckung:**
- Entwicklungsgeschichte des UAV VLN (2018-2026)
- Methodenklassifizierung: Nachahmungslernen / Verstärkungslernen / LLM-Inferenz
- Kernherausforderungen: dreidimensionale Raumdarstellung, dynamische Umgebung, Echtzeit-Argumentation
- Datensätze: D3DROU, AI-TOD, UAV-VLN usw.
- Zukünftige Richtungen: multimodale Großmodelle, verkörperte Intelligenz und Sicherheitsgarantie

---## 4. Zerlegung der technischen Architektur

### 4.1 Wahrnehmungsschicht (Wahrnehmung)

**Kamerakonfiguration:**

| Geben Sie | ein Vorteile | Nachteile |
|------|------|------|
| Nach vorne gerichtetes RGB | Ausgereift, günstig | Enges Sichtfeld, begrenzte Informationen |
| Omnidirektionale Kamera | 360°-Wahrnehmung | Geringe Auflösung, große Verzerrung |
| Tiefenkamera | Dichte Tiefe | Ausfall im Freien, eingeschränkte Reichweite |
| Multikamera | Stereo-Triangulation | Komplexe Kalibrierung |

**Verantwortlichkeiten des Wahrnehmungsmoduls:**

1. Objekterkennung + semantische Segmentierung (Grounding DINO, YOLO-World)
2. Extraktion der räumlichen Beziehung (links und rechts, oben und unten, relativer Abstand)
3. Aufbau eines Szenendiagramms (Objekt + Beziehung + Topologie)

### 4.2 Ebene verstehen

**VLM-Auswahlvergleich:**

| Modell | Parametervolumen | Sehfähigkeiten | Edge-Bereitstellung | Repräsentative Arbeit |
|------|--------|---------|---------|---------|
| GPT-4V | ~1,8T | Extrem stark | ❌ | Akademische Forschung |
| GPT-4o | ~200B | Extrem stark | ❌ | Cloud-API |
| LLaVA-1.6 | 7B/13B/34B | Stark | ✅ (ONNX) | Lokale Bereitstellung |
| Qwen-VL | 7B/72B | Stark | ✅ | Chinesische Szene |
| CogVLM | 17B | Stark | ⚠️ | Ausgewogene Lösung |

### 4.3 Planungsebene (Planung)

**Bestehendes Planungsparadigma:**

1. **LLM als Planer**: Aktionssequenzen direkt von LLM ausgeben lassen (ReAct, Reflexion)
   „
   Anweisung → LLM-Argumentation → Aktionssequenz → Ausführung
   „
2. **Symbolische PDDL-Planung**: LLM generiert eine PDDL-Domänenbeschreibung, gelöst durch den klassischen Planer
   - Vertreter: UniPlan
3. **Lernbare Planung**: Durchgängiges Nachahmungslernen/Verstärkungslernen
   - Vorteile: Anpassung an dynamische Umgebungen
   - Nachteile: schlechte Verallgemeinerung

### 4.4 Kontrollschicht (Kontrolle)

**UAV-Steuerungsfunktionen:**- Erfordert Echtzeit-Trajektorienverfolgung (Steuerfrequenz „>100 Hz“)
- Die Inferenzverzögerung (zweite Ebene) von VLM/LLM ist nicht mit der Echtzeitsteuerung vereinbar
- **Lösungsidee: hierarchische Steuerung**
  - Hohe Stufe: VLM/LLM (langsam, zweite Stufe) → Zielpunkt
  - Niedriges Niveau: MPC/PID (schnelles, Millisekunden-Niveau) → Motorsteuerung

---

## 5. Wichtigste Herausforderungen

### 5.1 Sim2Real-Lücke

- **Problem:** VLM ist auf ImageNet/COCO vorab trainiert und trifft während eines echten UAV-Flugs auf eine neue Stadtlandschaft
- **Lösungsideen:**
  - Domänen-Randomisierung (Simulations-Randomisierung)
  - Retrieval-Augmented Generation (RAG) zusätzliche Priorität
  - Selbstüberwachte Anpassung (Ego4D, DyTap)

### 5.2 Inferenzverzögerung vs. Echtzeitsteuerung

| VLM | Inferenzverzögerung | Anwendbare Szenarien |
|-----|---------|---------|
| GPT-4o | 1-3s | Cloud-Offline-Planung |
| LLaVA-7B | 0,5-1s | Kantenverzögerungsplanung |
| LLaVA-3B | 0,2-0,5s | Edge-Echtzeit |

**Lösungsrichtung:**

- Dual-Prozess-Architektur: Entkopplung von Argumentations-Thread und Kontroll-Thread
- Spekulative Dekodierung
- 4-Bit-Quantisierung (AWQ, GGUF)

### 5.3 Dreidimensionales räumliches Denken

Die räumlichen Beziehungen in der Sprache („hinter dem Baum“, „unter der Brücke“) sind keine einfachen Projektionen im dreidimensionalen Raum.

**Forschungsgrenzen:**
- SpatialPoint: Vorhersage ausführbarer 3D-Wegpunkte
- Können LLMs ohne Pixel sehen?: Testen der räumlichen Intelligenz von LLM

---

## 6. Zusammenfassung des Datensatzes| Datensatz | Plattform | Maßstab | Funktionen |
|--------|------|------|------|
| RxR | Boden | 126.000 Befehle | Mehrsprachige, fachmännische Anmerkungen |
| VLN-CE | Boden | 61K Flugbahnen | Matterport3D |
| AI-TOD | UAV | ~20.000 Befehle | Luftperspektive, Luftfotografie |
| UAV-VLN | UAV | ~10K | Urban Canyon-Szene |
| D3DROU | UAV | ~5K | Dynamische Hindernisse, echter Flug |

---

## 7. Zukünftige Forschungsrichtungen

1. **Multimodale Fusion**: RGB + Tiefe + Ereigniskamera + LiDAR
2. **Anpassung kleiner Stichproben**: LoRA/QLoRA-Feinabstimmung zur Anpassung an bestimmte städtische Umgebungen
3. **Mehrere UAV-Zusammenarbeit VLN**: Mehrere UAVs arbeiten zusammen, um denselben Befehl zu verstehen
4. **Weltmodellunterstützung**: Integrieren Sie das Weltmodell, um zukünftige Zustände vorherzusagen
5. **Sicherheitsüberprüfung**: Formale Methode zur Überprüfung der VLN-Entscheidungssicherheit

---

## 📚 Referenzen1. Zhang et al. *LogisticsVLN: Vision-Language-Navigation für die Terminalzustellung in geringer Höhe auf Basis von Agenten-UAVs*. arXiv:2505.03460, 2025.
2. Liu et al. *OmniVLN: Omnidirektionale 3D-Wahrnehmung und tokeneffizientes LLM-Argumentation für die visuelle Sprachnavigation über Luft- und Bodenplattformen hinweg*. arXiv, 2026.
3. Chen et al. *Vision-and-Language-Navigation für UAVs: Fortschritte, Herausforderungen und eine Forschungs-Roadmap*. arXiv, 2026.
4. ASMA. *Ein adaptiver Sicherheitsmargenalgorithmus für die Vision-Language-Drohnennavigation über szenenbewusste Kontrollbarrierenfunktionen*. arXiv, 2024.
5. Blukis et al. *Zuordnung von Navigationsanweisungen zu kontinuierlichen Steueraktionen mit Positionsvisitationsvorhersage*. CoRL, 2018.
6. Raychaudhuri et al. *Zero-Shot Object-Centric Instruction Following: Integration von Foundation-Modellen mit traditioneller Navigation*. arXiv, 2024.