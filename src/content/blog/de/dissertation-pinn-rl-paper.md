---
title: "Artikel: RL-basierte kooperative Optimierung der Kanalisierung und Rampenmessung in Webbereichen"
description: "In einem SCI Q3-Papier des Erstautors wird ein Reinforcement-Learning-Ansatz zur Koordinierung des Kanalisierungsdesigns und der Rampenmessung für städtische Schnellstraßenbereiche vorgestellt."
pubDate: 2023-04-10
tags: ["Verkehrstechnik", "Verstärkungslernen", "Schnellstraße", "SUMO", "SCI Q3"]
category: Paper
doi: "10.1155/2023/4771946"
journal: "Journal of Advanced Transportation"
sourceHash: "a9a545059b6f7eb95d0926a5116d27e524cbb70d"
---

# RL-basierte kooperative Optimierung der Kanalisierung und Rampenmessung

**Autoren:** Diantao Deng, Bo Yu, Duo Xu, Yuren Chen, You Kong
**Zeitschrift:** *Journal of Advanced Transportation*, 2023
**DOI:** [10.1155/2023/4771946](https://doi.org/10.1155/2023/4771946)
**Auswirkungsfaktor:** 2,3 | **Kategorie:** SCI Q3

---

## Motivation

Städtische Schnellstraßenbereiche sind berüchtigt für Staus. Wenn Fahrzeuge in kurzer Entfernung auf mehreren Fahrspuren zusammenlaufen oder abzweigen müssen, kommt es zu Konflikten – und herkömmliche Steuerungen mit nur einer Strategie (entweder Fahrspurmarkierungen *oder* Rampensignale, niemals beides zusammen) sind in der Regel nicht in der Lage, diese effektiv zu bewältigen.

Die wichtigste Erkenntnis dieses Dokuments: **Kanalisierung (wie Fahrspuren physisch unterteilt sind) und Rampenmessung (wie Fahrzeuge von Auffahrten zugelassen werden) sind keine unabhängigen Probleme.** Ihre gemeinsame Optimierung – und nicht isoliert – kann erhebliche Leistungssteigerungen ermöglichen.

## MethodeDas vorgeschlagene Framework verwendet einen **Q-Learning**-Agenten, um beide Strategien dynamisch zu koordinieren:

1. **Kanalisierungsstrategien** – zwei Arten von Fahrspurmarkierungskonfigurationen, die steuern, wie Fahrzeuge zusammen-/auseinanderlaufen
2. **Rampenmessung** – adaptive Signalsteuerung an der Auffahrt zur Regulierung des Zuflusses
3. **Kooperativer Modus** – Q-Learning entscheidet in Echtzeit über die optimale Kombination beider

Die Umgebung wird in **SUMO** (Simulation of Urban Mobility) erstellt, wobei reale Verkehrsdaten, die über **UAV-Luftaufnahmen** gesammelt werden, zur Kalibrierung und Validierung der Simulation verwendet werden.

## Ergebnisse

Die kooperative Methode übertrifft alle Alternativen deutlich. Spur 3 – die am stärksten von Zusammenfahrkonflikten betroffene Spur – verzeichnet eine dramatische **37 %ige Verbesserung** der durchschnittlichen Fahrzeuggeschwindigkeit:

- **Spur-1:** Steigerung der Durchschnittsgeschwindigkeit um +14,51 %
- **Spur 2:** Steigerung der Durchschnittsgeschwindigkeit um +14,81 %
- **Spur 3:** Steigerung der Durchschnittsgeschwindigkeit um +37,03 %

## Wichtige Erkenntnisse- **Gemeinsame Optimierung schlägt isolierte Strategien.** Verkehrskontrolle ist ein Systemproblem; Es als solches zu behandeln, zahlt sich aus.
- **Q-Learning ist für die Ampelsteuerung sinnvoll**, auch ohne ein vollständiges Dynamikmodell – der Agent lernt die optimale Richtlinie ausschließlich aus Belohnungssignalen in der Simulation.
- **SUMO + Python-Co-Simulation** bietet eine praktische Plattform zum Entwickeln und Testen von RL-basierten Verkehrscontrollern vor dem realen Einsatz.
- **UAV-basierte Datenerfassung** bietet eine skalierbare Möglichkeit, reale Verkehrsdaten für die Simulationskalibrierung zu erhalten.

## Verwandte Arbeit

Dieses Papier stützt sich auf frühere SUMO-Simulationsforschungen aus der breiteren Verkehrstechnik-Community und steht neben anderen RL-basierten Signalsteuerungsarbeiten in der Literatur. Die hier entwickelte SUMO-Python-Co-Simulationspipeline wurde zur Grundlage für das [Simulationsplattform-Projekt](/), auf das auf meiner About-Seite verwiesen wird.

---*Vollständiger Artikel verfügbar unter: [https://doi.org/10.1155/2023/4771946](https://doi.org/10.1155/2023/4771946)*