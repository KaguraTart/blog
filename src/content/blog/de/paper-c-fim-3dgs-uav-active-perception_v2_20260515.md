---
title: "Papier C Forschungsplanung v2: Rekonstruktion der aktiven UAV-Erfassung und -Planung in geringer Höhe für die Einreichung bei T-ITS / TR-C-Top-Journalen"
description: "v1 positioniert RA-L für eine schnelle Veröffentlichung, und der Lehrer verlangt, dass es zuerst veröffentlicht wird. In diesem Artikel wird die FIM-3DGS-Arbeit als Basistechnologie für die Wirtschaft in geringer Höhe und den städtischen Flugverkehr neu positioniert und in der Zusammenstellung vom 23.05.2026 klargestellt, dass sie derzeit verschoben und als Technologierichtung für aktive Sensorik reserviert wird."
pubDate: 2026-05-15
updatedDate: 2026-05-23
tags: ["Abschlussarbeitsplanung", "Einreichung bei Top-Journal", "T-ITS", "TR Teil C", "Wirtschaft in geringer Höhe", "aktive Wahrnehmung", "3DGS", "UAV", "Fisher-Informationen"]
category: Tech
---

# Paper C v2: Neupositionierung von RA-L zur Top-Ausgabe

> **v1 → Kernänderungen in v2:** Der Lehrer hat die oberste Ausgabe angefordert. v1 war ursprünglich als IEEE RA-L (IF 4.6 Q2, Schnellveröffentlichung) positioniert und wird jetzt auf eine parallele Strategie aus **IEEE T-ITS (IF 8.5 Q1) Hauptinvestition + TR Teil C (IF 8.5 Q1) Backup-Investition** aktualisiert. Dabei geht es nicht nur um einen Zeitschriftenwechsel, sondern auch um die Positionierung des Problems, das experimentelle Design, die Bewertungsindikatoren und die Längenstruktur des gesamten Manuskripts. Dieser Artikel ist das vollständige Designdokument dieses Refactorings.

---

## 0. Hauptunterschiede zwischen v1 und v2

| Abmessungen | v1 (RA-L 8 Seiten) | v2 (T-ITS/TR-C Seiten 20-25) |
|------|---------------|---------------------------|
| **Kernpositionierung** | Aktiver Sensor-/3D-Rekonstruktionsalgorithmus | Wirtschaftliche Technologie für niedrige Flughöhen/Städtisches Flugverkehrssystem |
| **Zielleser** | Robotik / CV-Stipendiaten | Wissenschaftler für intelligente Verkehrssysteme / Verkehrstechnik |
| **Problemstellung** | So wählen Sie optimale Standpunkte zur Rekonstruktion von 3D-Szenen aus | Wie man UAVs in die Lage versetzt, Transportaufgaben in geringer Höhe in Städten sicher und effizient durchzuführen |
| **Schlüsselindikatoren** | PSNR / SSIM / Abdeckung | Missionserfolgsrate / Luftraumauslastungsrate / Sicherheitsmarge / Energieverbrauch der Einheit Servicevolumen |
| **Basismethode** | FisherRF / GauSS-MI und andere Erfassungsmethoden | Erfassungsmethode + UAV-Industrieplanungsmethode + ITS-Simulationsvergleich |
| **Experimentelles Szenario** | Einzelne Rekonstruktionsaufgabe | Multitasking-Langzeitbetrieb (Lieferung, Inspektion, Notfall) |
| **Theoretische Tiefe** | Ableitung der FIM-Formel | FIM + Systemwarteschlangentheorie + Beweisbarkeit von Sicherheitsbeschränkungen |
| **Länge** | 8 Seiten | 20-25 Seiten |
| **Einreichungszeit** | 2026/09 | 2027/03–06 |

**Warum diese Rekonstruktion sinnvoll (nicht erzwungen) ist:** Der technische Kern von Paper C (aktive FIM-3DGS-Erkennung) selbst ist die entscheidende Engpasstechnologie für den autonomen Betrieb von UAVs und wird zum Zwecke der Ausstellung von T-ITS nicht zwangsweise verpackt. Aber v1 stellt diese Technologie nicht in den Kontext eines Transportsystems – v2 füllt diese Ebene.

### 0,1 23.05.2026 Aufräumen: aktuelle Prioritäten und GrenzenPapier C ist immer noch eine wertvolle aktive Sensorrichtung, sollte jedoch derzeit nicht mit G1, B und F-J1 um aktuelle Hauptressourcen konkurrieren. Der Grund dafür ist, dass es gleichzeitig den Wert der aktiven 3DGS-Sensoralgorithmen, der UAV-Sicherheitsplanung und der Transportsysteme beweisen wird und die Arbeitsfläche in der ersten Version größer sein wird als erwartet.

Derzeit wird empfohlen, Papier C als **P3-Reserverichtung** zu positionieren:

| Projekt | Wird derzeit bearbeitet |
|------|----------|
| Hauptbeitrag | FIM-3DGS aktive Blickwinkelauswahl + UAV-Sicherheitseinschränkungen |
| Verkehrsanbindung | Nur für die Aktivierung von Technologien zur Inspektion, Notfallreaktion und Verteilung reserviert, ohne vorher ein vollständiges TR-C-Systempapier zu verfassen |
| Muss verbessert werden | Reale/öffentliche städtische 3D-Daten, starke NBV-Basislinie, Indikatoren auf Aufgabenebene, reproduzierbare Simulation |
| Inhalt in der Warteschleife | Multi-UAV-Systemdurchsatz, wirtschaftspolitisches Narrativ in geringer Höhe, komplettes SUMO-AirSim-Großsystem |
| Wiederherstellungsbedingungen | Die G1-Toolkette ist stabil, die F-Szenenplattform kann wiederverwendet werden oder es sind ausreichend 3DGS/aktive Sensor-Experimente vorhanden |

Wenn es in Zukunft neu gestartet werden soll, sollte der erste Artikel auf dem **T-RO / T-ITS-Methodenpapierstandard** basieren, um die aktive Sensortechnologie zu verbessern und zu bestätigen, dass die technischen Indikatoren haltbar sind; Erst wenn das Experiment beweisen kann, dass es die Aufgabeneffizienz und Sicherheitsindikatoren für Inspektion/Notfall/Lieferung erheblich verbessern kann, erfolgt ein Upgrade auf das TR-C-Systempapier.

---

## 1. Neupositionierung: vom „Wahrnehmungsalgorithmus“ zur „Technologie zur Förderung der Wirtschaftlichkeit in geringer Höhe“

### 1.1 Strategischer Hintergrund (muss beim Schreiben den Weg ebnen)

**Nationale politische Ebene (2024–2025):**
- Chinas „14. Fünfjahresplan“ für die wirtschaftliche Entwicklung in geringer Höhe: Das Ziel für die wirtschaftliche Entwicklung in geringer Höhe liegt bei 2,5 Billionen im Jahr 2025 und erreicht 5 Billionen im Jahr 2030
- „National Comprehensive Three-dimensional Transportation Network Planning Outline“ der Zivilluftfahrtbehörde Chinas: Präzisiert UAVs in geringer Höhe als städtische Transportinfrastruktur
- Wirtschaftliche Pilotprojekte in geringer Höhe in Shenzhen, Guangzhou, Hefei und anderen Städten im Jahr 2024

**Akademische Herausforderung (das grundlegende Problem, das in der Arbeit gelöst werden soll):**
- UAVs in geringer Höhe müssen beim Eindringen in Städte drei Kernprobleme lösen:
  1. **Effizienz der Luftraumnutzung:** Eine Stadt muss Tausende von UAVs gleichzeitig betreiben können
  2. **Betriebssicherheitsgarantie:** Keine Kollision mit Gebäuden, Menschenmengen und anderen Flugzeugen
  3. **Geschlossener Wahrnehmungs-Entscheidungs-Kreislauf:** UAV muss in Echtzeit ein Verständnis der Umgebung aufbauen, um sichere Entscheidungen treffen zu können
- Diese drei Aspekte hängen miteinander zusammen: Die wahrgenommene Qualität bestimmt die Entscheidungszuverlässigkeit, und die Entscheidungszuverlässigkeit bestimmt die Durchführbarkeit des Luftraumabfertigungsdienstes**Positionierung dieses Artikels:** Die dritte Frage (geschlossener Wahrnehmungs-Entscheidungs-Kreislauf) ist die Grundlage der ersten beiden Fragen. Dieser Artikel schlägt **FIM-3DGS vor: ein informationsgesteuertes aktives UAV-Erfassungs- und Planungs-Framework**, um die Erfassungseffizienz und Betriebssicherheit eines einzelnen UAV in der städtischen Umgebung grundlegend zu verbessern und so ein groß angelegtes Luftraummanagement in geringer Höhe zu unterstützen.

### 1.2 Dialog mit bestehenden Top-Zeitschriftenbeiträgen

**Verwandte Arbeiten, die kürzlich in TR Teil C (2023–2025) aufgenommen wurden:**

| Papier | Thema | Beziehung zu diesem Artikel |
|------|------|------------|
| Mohamed et al. 2024 | „UAV-unterstütztes Netzwerkdesign für die Zustellung auf der letzten Meile“ | Eine perfekte Wahrnehmung vorausgesetzt, ergänzen wir die Wahrnehmungsschicht |
| Liu & Tang 2023 | „Drohnenflugbahnplanung für die städtische Paketzustellung“ | Mithilfe der geometrischen Pfadplanung stellen wir einen geschlossenen Sensor-Planungs-Kreislauf bereit
| Park et al. 2024 | „Vertiport-Planung für UAM-Operationen“ | Für die Planung auf Mikroebene bieten wir eigenständige Enabling-Technologie |
| Chen et al. 2025 | „Risikobewertung für UAVs in geringer Höhe in Städten“ | Risikobewertung, unsere Wahrnehmung kann Daten zur Risikobewertung liefern |

**IEEE T-ITS-bezogene Dokumente (2023–2025):**| Papier | Thema | Beziehung zu diesem Artikel |
|------|------|------------|
| Wang et al. 2024 | „Multi-UAV-Flugbahnoptimierung in städtischen Umgebungen“ | Konzentrieren Sie sich auf den Weg, ohne die Auswirkungen der wahrgenommenen Unsicherheit zu berücksichtigen |
| Zhang et al. 2023 | „Kooperative Luft-Boden-Wahrnehmung für UAM“ | Multisensor-Fusion, unser FIM-Framework kann als Grundlage für die Berechnung des Fusionsgewichts verwendet werden |
| Kim et al. 2025 | „Informationstheoretische aktive Kartierung für autonome Fahrzeuge“ | Boden-AV-Aktiverkennung, wir sind die UAV-Version und fügen Sicherheitsbeschränkungen hinzu |

**Von T-ITS und TR-C geteilte Papier-Hotspots:**
- Urban Air Mobility (UAM)
- UAV-Logistik in geringer Höhe
- Multimodaler Transport (einschließlich UAV)
- Wahrnehmung des autonomen Fahrens (analog zur Migration zum UAV)
- Risikobewertung der Luftraumnutzung

### 1.3 Titel und Zusammenfassung neu positioniert

**V2-Titel (Chinesisch und Englisch):**

- **Chinesisch:** Informationsgesteuerte aktive Erfassung und Planung für die städtische Wirtschaft in geringer Höhe: 3DGS-Rahmenwerk für den autonomen UAV-Betrieb
- **Englisch:** Informationsgesteuerte aktive Wahrnehmung und Planung für die städtische Wirtschaft in geringer Höhe: Ein 3D-Gaußsches Splatting-Enabling-Framework für autonome UAV-Operationen

**V2-Zusammenfassung (350 Wörter in Englisch, entsprechend der Zusammenfassungslänge der Top-Ausgabe):**> Städtische UAV-Operationen in geringer Höhe – einschließlich Lieferung auf der letzten Meile, Infrastrukturinspektion und Notfallmaßnahmen – stehen vor einer grundlegenden Herausforderung: Dichte städtische Umgebungen erfordern eine qualitativ hochwertige 3D-Wahrnehmung für sichere autonome Entscheidungen, doch herkömmlichen Wahrnehmungspipelines mangelt es entweder an Genauigkeit (Belegungsraster) oder sie scheitern an Echtzeitbeschränkungen (NeRF). In diesem Artikel wird **FIM-3DGS** vorgestellt, ein informationsgesteuertes aktives Wahrnehmungs- und Planungsframework, das diese Lücke schließt. Wir leiten eine geschlossene Fisher-Information-Matrix-Formulierung (FIM) für 3D-Gaußsche Splatting-Primitive (3DGS) ab und stellen das erste strenge Cramér-Rao-gebundene Ansichtsauswahlkriterium für explizite neuronale Rendering-Darstellungen bereit. Ein Rendering-Varianz-Proxy reduziert die FIM-Berechnung von $O(N|P|D^2)$ auf $O(N)$ und ermöglicht Echtzeit-Entscheidungen (<20 ms) für die nächstbeste Ansicht für mehr als 100.000 Gauß-Funktionen. Darüber hinaus integrieren wir die Sicherheitseinschränkungen der Control Barrier Function (CBF) in 6-DoF UAV-Dynamik, die einen nachweislich kollisionsfreien Betrieb ermöglicht. Umfassende Simulationsexperimente an MatrixCity (Stadtdatensatz) und einem benutzerdefinierten digitalen Zwilling von AirSim zeigen, dass FIM-3DGS ein um 1,8 dB höheres PSNR und eine um 8,2 % höhere Abdeckung als das hochmoderne GauSS-MI (RSS 2025) erreicht und gleichzeitig die Missionsabschlusszeit um 27 % verkürzt, und zwar in drei Fallstudien zu Transportsystemen: Gebäudeinspektion, Paketzustellung und Notfallreaktion. **Aus ITS-Sicht** reduziert unser Framework die Luftraumnutzung pro Aufgabe um 31 % und verbessert den Multi-UAV-Durchsatz um 22 %, wenn es in bestehende UAM-Planungssysteme integriert wird. Code und Datensätze werden veröffentlicht, um die zukünftige Forschung zur Wirtschaft in geringer Höhe zu unterstützen.**Wichtige Schreibtipps:**
- Der erste Satz ordnet das Problem sofort der „Transportanwendung“ (Lieferung/Inspektion/Notfall) zu.
- Technische Beiträge in der Mitte behalten (FIM-Ableitung, Komplexität, CBF)
- Der letzte Absatz betont „Indikatoren auf Systemebene“ (Missionsabschlusszeit, Luftraumnutzung, UAM-Durchsatz) – das ist es, worüber sich die T-ITS/TR-C-Prüfer am meisten Sorgen machen
- Erwähnen Sie den Code/Datensatz als Open Source (oberste Veröffentlichungstendenz zur Verbesserung der Reproduzierbarkeit)

---

## 2. Neu formulierte Forschungsfragen

### 2.1 Problemstellung auf Systemebene (neu in Version 2)

**Makroprobleme:** Unter der Vision eines wirtschaftlichen Ausmaßes in geringer Höhe von 5 Billionen im Jahr 2030 muss eine mittelgroße Stadt (5 Millionen Einwohner) täglich etwa 100.000 UAV-Einsätze durchführen (siehe Datenextrapolation zum unbemannten Lieferpiloten von Meituan/JD). Dies erfordert, dass jedes UAV:

1. **Genaue Wahrnehmung:** Behalten Sie die zentimetergenaue 3D-Darstellung in Echtzeit in unbekannten oder sich dynamisch ändernden Umgebungen bei
2. **Effizienter Betrieb:** Ein einzelnes UAV maximiert das Aufgabenvolumen bei begrenzter Leistung
3. **Sicherheitszertifizierung:** Der Abstand zu Gebäuden, Fußgängern und anderen UAVs entspricht strikt den Sicherheitsvorschriften

**Unterproblemzerlegung:**

| Teilprobleme | Bestehende Lösungen | Einschränkungen | Beiträge zu diesem Artikel |
|--------|----------------|--------|---------|
| F1: Wie kann eine dynamische städtische Umgebung mit hoher Qualität rekonstruiert werden? | Offline-NeRF / Belegungsraster | Langsam / Rau | Online 3DGS + Active Sensing |
| F2: Wie kann man entscheiden, wohin das UAV als nächstes fliegen soll? | Voreingestellte Routen/geometrische Pfadplanung | Berücksichtigt nicht die wahrgenommene Unsicherheit | FIM-informationsgesteuertes NBV |
| F3: Wie kann sichergestellt werden, dass Entscheidungen den Sicherheitsvorschriften entsprechen? | Kollisionserkennung nach der Verarbeitung | Reaktiv, Mangel an Garantien | CBF eingebettete Sicherheitseinschränkungen |
| F4: Wie lässt sich der Wert des Systems für den städtischen Verkehr bewerten? | Einzelaufgabenexperiment | Fehlende Multitasking-Langzeitbewertung | Bewertung von drei Hauptszenarien auf Systemebene |

### 2.2 Optimierungsthemen aus Sicht von ITS (neu in v2)**Einzelne UAV-Missionsoptimierung (eine Mission):**
$$\max_{\mathbf{v}_{1:T}}\; \alpha\,\underbrace{Q_{rec}(\boldsymbol{\Theta})}_{\text{Rekonstruktionsqualität}} + \beta\,\underbrace{Q_{Aufgabe}(\mathbf{v}_{1:T})}_{\text{Aufgabenabschluss}} - \gamma\,\underbrace{E(\mathbf{v}_{1:T})}_{\text{Energie Verbrauch}}$$

Einschränkungen: UAV-Dynamik + Sicherheits-CBF + Missionseinschränkungen (unbedingt besuchte Gebiete) + Energiebudget

**Bewertung der ITS-Systemebene (Multi-Task-Multi-UAV):**
$$\Phi_{ITS} = \frac{\sum_k S_k^{Erfolg}}{\sum_k T_k^{Flug}\cdot E_k}$$

Darunter ist $S_k^{success}$ die Abschlusserfolgsrate der Aufgabe $k$, $T_k^{flight}$ ist die Flugzeit und $E_k$ ist der Energieverbrauch. Diese Metrik misst die Aufgabenleistung pro Ressourceneinheit (Zeit + Energie) und ist eine Standardsystemmetrik in der ITS-Literatur.

**Wichtige Innovationspunkte:** Bestehende UAV-Forschung optimiert im Allgemeinen einzelne Indikatoren auf Aufgabenebene (z. B. Lieferzeit), der Durchsatz auf Systemebene sollte jedoch aus ITS-Perspektive optimiert werden. Dieser Artikel zeigt: Durch die Einführung der aktiven Erfassung wird die Unsicherheit der Einzelmaschinenwahrnehmung verringert → die Entscheidungsfindung ist radikaler und dennoch sicher → die Effizienz von Einzelmaschinenaufgaben wird verbessert → der Durchsatz auf Systemebene wird natürlich verbessert.

---

## 3. Drei große Fallstudien (neue Kerninhalte der Version 2)

> Die Frage, die Top-Rezensenten von Fachzeitschriften am meisten beschäftigt: Welche Auswirkungen wird der Algorithmus auf echte Verkehrsprobleme haben? v2 wird anhand von drei konkreten Fällen beantwortet.

### Fall 1: Inspektion städtischer Gebäudestrukturen (Infrastrukturinspektion)

**Szeneneinstellungen:**
- Mission: UAV-Inspektion von Fassadenrissen/losen Elementen eines 30-stöckigen Bürogebäudes
- Eingabe: GPS-Standort des Gebäudes + grobe Erscheinungsbildparameter
- Ausgabe: vollständiges 3DGS-Modell + Fehleranmerkung (im Anschluss an diese Arbeit)**Bewertungsmetriken (ITS-Perspektive):**
- **Inspektionsabdeckungsrate:** Der Anteil der effektiven durchgeführten Beobachtungen der Gebäudeoberfläche (bezogen auf die Sanierungsqualität)
- **Einzelinspektionsflugzeit:** Die Anzahl der Minuten, die für die Durchführung einer vollständigen Inspektion erforderlich sind
- **Re-Inspektionsrate:** Der Anteil der Flüge, die aufgrund einer wahrgenommenen minderwertigen Qualität erneut geflogen werden müssen
- **Energieverbrauch:** Stromverbrauch für eine einzelne Inspektion (beeinflusst die Anzahl der Gebäude, die an einem Tag inspiziert werden können)

**Im Vergleich zum Ausgangswert (Branchenpraxis):**
1. **Rasenmäher-Scannen (Branchen-Mainstream):** Feste rechteckige Scanroute, die Standardpraxis kommerzieller DJI- und Skydio-Lösungen
2. **Manuelle Wegpunktplanung:** Ingenieure legen Points of Interest manuell fest
3. **FisherRF/GauSS-MI:** Akademisches SOTA
4. **FIM-3DGS (dieser Artikel)**

**Erwartete Ergebnisse:**
- Arbeitszeit im Vergleich zum Rasenmäher: um mehr als 30 % reduziert (Informationsgesteuert, um wiederholte Beobachtungen zu vermeiden)
- Wiederholungsrate: von 15 % auf <3 % reduziert

### Fall 2: Lieferung auf der letzten Meile

**Szeneneinstellungen:**
- Mission: UAV-Lieferung vom Lieferort zum Kundenbalkon
- Herausforderung: Komplexe Gebäudeverdeckung zwischen Häuserschluchten + dynamische Hindernisse (Fensterschalter, Wäscheständer usw.)
- Eingabe: Startpunkt-GPS, Endpunkt-GPS, grobe Beschreibung des Kundenstandorts
- Ausgabe: erfolgreiche Lieferung + vollständiges Flugprotokoll

**Bewertungsindikatoren:**
- **Zustellungserfolgsquote:** Erfolgsquote der an den Kundenbalkon gelieferten Pakete (Kern-KPI)
- **Durchschnittliche Lieferzeit:** Von der Abreise bis zur Lieferung
- **Sicherheitsspielraum auf Aufgabenebene:** Mindestabstandsstatistik zu Hindernissen während des gesamten Liefervorgangs
- **Luftraumbelegung:** 3D-Luftraumvolumen, das von einer einzelnen Lieferung belegt wird (beeinflusst die Versanddichte mehrerer UAVs)

**Im Vergleich zum Ausgangswert:**
1. **Voreingestellte Routen + reaktive Hindernisvermeidung: ** Mainstream-Lösungen von Wing/Meituan und anderen Unternehmen
2. **A* Routenplanung + Belegungsrasterkarte:** Akademischer Vergleich
3. **Kollaborative Erfassung mehrerer Roboter (A2X):** Nutzung anderer UAV-Daten
4. **FIM-3DGS (dieser Artikel)**

**Erwartete Ergebnisse:**
- Zustellungserfolgsrate: von 85 % (voreingestellte Route) → 96 % (aktive Erkennung)
- Luftraumbelegung: Reduzierung um 31 % (präzise Wahrnehmung ermöglicht engere Flugkorridore)

### Fall 3: Städtische Notfallreaktion (Notfallreaktion)**Szeneneinstellungen:**
- Mission: Nachdem ein Hochhausbrand ausgebrochen war, zeichnete das UAV innerhalb von 60 Sekunden ein 3D-Modell des Gebäudes für das Rettungskommando
- Herausforderung: völlig unbekannte Umgebung + Raucheinschluss + extrem hohe Aktualitätsanforderungen
- Eingabe: Standort des Brandmelders
- Ausgabe: Erstellen eines 3DGS-Modells + Identifizierung des betroffenen Bereichs

**Bewertungsindikatoren:**
- **Abdeckung innerhalb von 60 Sekunden:** Anteil der Gebäudeoberflächenbeobachtungen, die unter strengen Zeitvorgaben durchgeführt wurden
- **Geschwindigkeit zur Identifizierung kritischer Bereiche:** Zeit zum Erkennen der Brandquelle/des Evakuierungswegs
- **Keine Kollisionsrate:** Sichere Flugfähigkeit in völlig unbekannten Umgebungen

**Im Vergleich zum Ausgangswert:**
1. **Grenzexploration:** Klassische Explorationsmethode
2. **GaSS-MI:** Wichtigstes SOTA
3. **FIM-3DGS (dieser Artikel)**

**Erwartete Ergebnisse:**
- 60er-Abdeckung: von 70 % (Frontier) → 88 % (FIM-3DGS)
- Nullkollisionsrate: 100 % (CBF garantiert)

---

## 4. Experimentelles Design-Upgrade (v2 stark erweitert)

### 4.1 Simulationsplattform

AirSim + Unreal Engine 5 + Isaac Sim aus v1 beibehalten, neu:

**SUMO + AirSim gemeinsame Simulation (neu in v2):**
- SUMO bietet Bodentransportumgebung (Fußgänger, Fahrzeuge)
- AirSim bietet UAV-Simulation
- Simulieren Sie die multimodale Transportumgebung realer Städte durch ROS2-Bridging
- Dies ist die Fähigkeit zur „Simulation auf Systemebene“, die T-ITS-Prüfer zu schätzen wissen

### 4.2 Datensatz (v2-Erweiterung)| Datensatz | Quelle | Verwendung | v1/v2 |
|--------|------|------|------|
| MatrixCity | ICCV 2023 | Stadtsanierungs-Meistertest | Erhältlich in beiden Editionen |
| ScanNet v2 | CVPR 2017 | Eigene Entwicklungsverifizierung | Beide Versionen verfügbar |
| **UAV-Delivery-Dataset** | Selbstgebaut (neu in v2) | Evaluierung realer Lieferszenarien auf Aufgabenebene | Nur v2 |
| **Vertiport-Sim-Daten** | Selbstgebaut (neu in v2) | Multi-UAV-Start- und Lande-Szenario | Nur v2 |
| **Urban-Inspection-Suite** | Zusammenarbeit mit Skydio/DJI oder Open-Source-Daten | Standardisierte Beurteilung von Prüfaufgaben | Nur v2 |

**UAV-Delivery-Dataset-Erstellungsplan:**
- Erstellen Sie 5 typische städtische Verteilungsszenarien in AirSim (CBD, Wohngebiete, Industriegebiete, rund um Krankenhäuser, rund um Schulen)
- 100 Liefermissionen pro Szenario
- Beschriftung: Startpunkt, Endpunkt, Ground Truth 3D, optimaler Lieferweg, typische Hindernisse
- Wird verwendet, um die Erfolgsquote der Zustellung, die durchschnittliche Zeit und den Sicherheitsspielraum zu bewerten
- **Bonuspunkte für Gutachter von Top-Zeitschriften:** Selbst erstellter Datensatz + Open Source = erhöhter akademischer Beitrag

### 4.3 Bewertungsindikatorensystem (v2 stark erweitert)

**Ebene 1: Indikator für wahrgenommene Qualität (verfügbar in Version 1)**
- PSNR, SSIM, LPIPS, Abdeckung, Fasenabstand

**Ebene 2: Planungseffizienzindikator (verfügbar in Version 1)**
- Planungslatenz, InfoGain-Rate, PSNR@budget

**Ebene 3: Indikatoren auf Aufgabenebene (neu in Version 2)**
- **Mission Completion Rate (MCR):** Prozentsatz der erfolgreich abgeschlossenen Missionen
- **Aufgabenzeit pro Mission:** Durchschnittliche Abschlusszeit einer einzelnen Aufgabe
- **Energie pro Mission:** Energieverbrauch einer einzelnen Aufgabe
- **Rückflugrate:** Der Anteil der Rückflüge aufgrund unzureichender Wahrnehmung**Ebene 4: Indikatoren auf Systemebene (neu in Version 2)**
- **Luftraumnutzung:** 3D-Luftraumvolumen der Einheitsaufgabe (m³/Aufgabe)
- **Multi-UAV-Durchsatz:** Die Anzahl der Aufgaben, die N UAVs im gleichen Bereich pro Zeiteinheit erledigen können
- **Sicherheitsmargenverteilung:** Statistische Verteilung der Entfernung zum nächsten Hindernis während der gesamten Mission
- **Kumulativer Risikoindex:** $\int \mathcal{R}(\boldsymbol{\xi}(t))\,dt$ Kumulativer Risikoindex

**Ebene 5: Wirtschaftsindikatoren (neu in Version 2, TR-C-freundlich)**
- **Kosten pro erfolgreicher Lieferung:** Die Betriebskosten einer einzelnen erfolgreichen Lieferung (einschließlich Energieverbrauch, Wartung, Risiko)
- **Servicedichte:** Servicekapazität innerhalb der Stadt pro Flächeneinheit (Aufgabe/km²·Tag)

### 4.4 Baseline-Methode (v2 auf drei Kategorien erweitert)

**Klasse A: Basislinie der Wahrnehmungsmethode (vorhanden in Version 1)**
- FisherRF (ECCV 2024), GauSS-MI (RSS 2025), ActiveGS (T-RO 2024), GenNBV (CVPR 2024), Frontier, Random

**Klasse B: UAV Industrial Practice Baseline (neu in Version 2, erforderlich für T-ITS/TR-C)**
- **Rasenmäher-Scannen:** festes rechteckiges Scannen, kommerzielle DJI-Lösung
- **Vorgeplanter Wegpunkt:** Der Techniker legt die Points of Interest manuell fest
- **A\* mit Belegungsraster:** Klassische UAV-Pfadplanung

**Klasse C: ITS-Basislinie auf Systemebene (neu in Version 2)**
- **DJI FlightHub 2 Simulation:** Entscheidungsmodelle für kommerzielle UAV-Managementsysteme
- **Zentraler Flottenplaner:** MILP zentralisierte Planung, ideal, aber langsam
- **Keine aktive Wahrnehmung:** Rein passive Akzeptanz der Standardroute (Vergleich v1 vs. v2)

### 4.5 Ablationsexperiment (v2-Erweiterung)| Ablation | Varianten | Validierung |
|--------|------|------|
| CBF-Sicherheitseinschränkungen entfernen | FIM-3DGS-NoSafe | CBF Notwendigkeit |
| Verwendung von Shannon MI anstelle von FIM | MI-3DGS | Theoretische Vorteile von FIM vs. MI |
| 3DGS durch NeRF ersetzen | FIM-NeRF | Echtzeitbeitrag |
| Näherung durch exaktes FIM | ersetzen FIM-3DGS-Exakt | Ungefähre Genauigkeit vs. Geschwindigkeit |
| **Feedback auf Systemebene entfernen (neu in Version 2)** | FIM-3DGS-NoSystemLoop | Überprüfen Sie den Wert des Feedbacks auf Aufgabenebene |
| **Berücksichtigt keine Einschränkungen des Energieverbrauchs (neu in Version 2)** | FIM-3DGS-NoEnergy | Die Auswirkungen von Energieverbrauchsbeschränkungen auf Indikatoren auf Systemebene |

---

## 5. Innovationserklärung (V2-Rekonstruktion)

### Beitrag 1 (Theorie, T-ITS / TR-C sind alle betroffen)

**Erste Ableitung geschlossener Fisher-Information-Matrix-Ausdrücke** für explizite 3D-Gauß-Splatting-Primitivparameter**, die die strikte Äquivalenz zu den unteren Cramér-Rao-Grenzen beweist.

Im Vergleich zur Shannon-Entropie von GauSS-MI (RSS 2025):
- FIM bietet **strikte statistische Untergrenzen** (CRB) für die Genauigkeit der Parameterschätzung, die direkt in Rekonstruktionskonfidenzintervalle umgewandelt werden können
-Shannons Entropie misst nur die Zufälligkeit von Beobachtungen und steht nicht in direktem Zusammenhang mit der Genauigkeit der Parameterschätzung.
- Das D-Optimalitätskriterium (FIM-Determinante) entspricht der Minimierung des Rekonstruktionsfehlers des Ellipsoidvolumens

**Erklärung für ITS-Gutachter:** Dies ist gleichbedeutend damit, das UAV-Aktiverfassungsproblem vom empirischen Design auf die theoretische Ebene der „nachweisbaren Optimalität“ zu verschieben, sodass nachgelagerte Entscheidungen auf Systemebene (z. B. Planung mehrerer Maschinen, Luftraumzuweisung) auf strengen Untergrenzen der Erfassungsunsicherheit basieren können.

### Beitrag 2 (Methode, interdisziplinär)

**Vorschlag für ein Echtzeit-Aktiverfassungsplanungs-Framework mit leichter RVP-Approximation (Rendering Variance Proxy) + CBF-Sicherheitseinschränkungen**:- RVP reduziert die FIM-Rechenkomplexität von $O(N|P|D^2)$ auf $O(N)$ und erreicht eine Entscheidung von <20 ms auf der 100k-Gauß-Skala
- Integrierte CBF-Sicherheitseinschränkungen, eingeführt auf der Grundlage modernster Steuerungstheorie mit nachweisbaren Nullkollisionsgarantien
– Das Gesamtframework kann auf NVIDIA Jetson Orin 16G ausgeführt werden, um den Anforderungen eines echten UAV-Einsatzes in der Luft gerecht zu werden

**Erklärung für ITS-Gutachter:** Dies ist ein praktischer technischer Beitrag, der erstmals einen echten UAV-Einsatz in einem akademischen SOTA-Ansatz ermöglicht. Dies ist ein wichtiger Schritt bei der Integration von Industrie und Wissenschaft.

### Beitrag drei (System, Kernverkaufsargument von T-ITS/TR-C)

**Erste Bewertung der realen Auswirkungen der aktiven Sensorik auf den städtischen UAV-Transport auf Systemebene**:

- Drei große Fallstudien (Inspektion, Verteilung und Notfall) decken die wichtigsten Anwendungsszenarien der Tieflandwirtschaft ab
- Indikatoren auf Systemebene (MCR, Luftraumnutzung, Multi-UAV-Durchsatz) quantifizieren die Auswirkungen von Wahrnehmungsverbesserungen auf die Transporteffizienz
- Bereitstellung von Open-Source-Datensätzen wie UAV-Delivery-Dataset zur Unterstützung nachfolgender ITS-Forschung

**Erklärung für ITS-Gutachter:** Dies ist kein weiteres Wahrnehmungspapier – es handelt sich um die Arbeit, Wahrnehmungstechnologie in den ITS-Bewertungsrahmen einzubinden und die Kausalkette von „Wahrnehmungsverbesserung 1 dB PSNR“ bis „Verbesserung des Luftraumdurchsatzes um X %“ zu quantifizieren.

---

## 6. Unterschiede zum Top-Journal SOTA (v2-Erweiterung)

### 6.1 Detaillierter Vergleich mit GauSS-MI (RSS 2025)

| Abmessungen | GauSS-MI | FIM-3DGS v2 |
|------|----------|-------------|
| Informationsmaßnahme | Shannon-Entropie | Fisher-Informationen (CRB-Äquivalent) |
| Theoretische Basis | Obergrenze der Informationstheorie | Strenge Untergrenze der statistischen Schätzung |
| Rechenkomplexität | O(N·MC) | O(N) (RVP-Näherung) |
| UAV-Dynamik | Keine | 6-DoF SE(3) |
| Sicherheitsbeschränkungen | Keine | Explizite CBF-Garantien |
| Experimentelle Szene | Desktop/Innenraum | Stadtebene + drei Fälle |
| **Anwendungsschicht** | **Wiederaufbauqualität** | **Neuaufbau + Aufgabe + System** |

### 6.2 Unterschiede zur bestehenden UAV-Forschung in ITS (neu in v2)| ITS-Papier | Thema | Einschränkungen | v2-Verbesserungen |
|---------|------|------|--------|
| Mohamed et al. 2024 (TR-C) | Design eines UAV-Liefernetzwerks | Vorausgesetzt, dass die Wahrnehmung perfekt ist | Modellierung realer Wahrnehmungsunsicherheit |
| Wang et al. 2024 (T-ITS) | Multi-UAV-Flugbahnoptimierung | Berücksichtigt nicht die Online-Wahrnehmung | Wahrnehmungs-Entscheidungs-Schleife |
| Park et al. 2024 (TR-C) | Vertiport-Planung | Das Bewusstsein für eine einzelne Maschine wird nicht modelliert | Single-Machine-Awareness liefert Daten für die Multi-Machine-Planung |

---

## 7. Einreichungsstrategie (v2-Kernupdate)

### 7.1 Paralleler Einreichungspfad

```
2027/03  完成稿件 + 内部 review
            ↓
2027/04  投稿 IEEE T-ITS（首选）
            ↓
       ┌──────┴──────┐
       │             │
   接受/小修      拒稿/大修
       │             │
   2027/10 接受   重新调整框架
                    ↓
                改投 TR Part C
                （强调运输系统价值）
                    ↓
                2027/08 投稿
                    ↓
                2028/02 接受
```

**Schlüsselstrategien:** Der Kerninhalt des Manuskripts (80 %) ist in beiden Zeitschriften gleich, wobei Anpassungen nur am Rahmen (10–15 %) und bestimmten ITS-spezifischen Abschnitten (5–10 %) vorgenommen wurden. Auf diese Weise kann eine Schrift zwei Kandidaten bedienen.

### 7.2 Subtile Unterschiede zwischen T-ITS und TR-C (beim Schreiben beachten)

| Abmessungen | IEEE T-ITS | TR Teil C |
|------|-----------|----------|
| Wichtige Punkte | Algorithmus + ITS-Anwendung | Auswirkungen auf das System und die Politik |
| Abstrakter Stil | Technologieorientiert | Anwendungs- und wirkungsorientiert |
| Experimentelle Präferenz | Simulation + theoretische Analyse | Simulation + Fallstudie |
| Literaturverhältnis | 50 % Algorithmus/KI + 50 % ITS | 30 % Algorithmus + 70 % Transport |
| Diskussion | Algorithmusbeschränkungen + zukünftige Arbeit | Politische Implikationen + Auswirkungen auf die Branche + Einschränkungen |

**Schreibstrategie:** Das Hauptmanuskript basiert auf den T-ITS-Präferenzen, und die Zusammenfassungs-/Einführungs-/Diskussionsvorlage für die TR-C-Version ist vorbereitet, und der Rahmenwechsel kann innerhalb von 2 Wochen abgeschlossen werden.

### 7.3 Risiken und Reaktionen überprüfen| Mögliche Bewertungskommentare | T-ITS-Antwort | TR-C-Antwort |
|------------|-----------|-----------|
| „Die Beziehung zwischen Wahrnehmungsalgorithmen und ITS ist nicht stark“ | Unter Berufung auf Präzedenzfälle wie Kim 2025 (TITS) | Betonung des Systemwerts der drei Hauptfälle |
| „Dem Experiment fehlen reale Daten“ | Schwerpunkt auf MatrixCity-Realbildern + selbst erstellten Datensätzen | Schwerpunkt auf realen Szeneneinstellungen für Fallstudien |
| „Zu viel/zu wenig Theorie“ | FIM-Ableitung beibehalten, RVP-Beweis vereinfachen | Vereinfachen Sie die FIM-Formel und betonen Sie die intuitive Erklärung |
| „Unzureichende Relevanz für die vorhandene UAV-Literatur“ | ITS-UAV-Literaturübersicht hinzufügen | Literaturrecherche zur Verkehrstechnik hinzufügen |
| „Keine Erklärung der Richtlinie“ | Kurzes politisches Zitat | Konzentrieren Sie sich auf die Diskussion der Auswirkungen wirtschaftspolitischer Maßnahmen auf niedriger Ebene |

---

## 8. Neu geplante Ausführungsroute (V2-Zeitleiste)

### Detailliertes Gantt-Diagramm für 15 Monate

```
时间        阶段                                   关键交付物
──────────────────────────────────────────────────────────────────────────
2026/06    准备阶段
           • FIM-3DGS 核心算法实现（CUDA）
           • AirSim + SUMO 联合仿真平台搭建        ▶ 核心代码完成
           • MatrixCity 数据获取与预处理

2026/07    基础实验
           • 与 FisherRF/GauSS-MI/ActiveGS 集成测试
           • Layer 1/2 指标实验（PSNR、规划延迟）  ▶ 算法层实验完成

2026/08    案例研究 1：建筑物巡检
           • 在 AirSim 中搭建 30 层建筑场景
           • 100 次巡检任务实验
           • 与 Lawn-mower / 工业方案对比         ▶ 巡检案例完成

2026/09    案例研究 2：最后一公里配送
           • 自建 UAV-Delivery-Dataset
           • 5 城市场景 × 100 任务 = 500 次配送实验
           • 与预设航线 / A* 对比                  ▶ 配送案例完成

2026/10    案例研究 3：应急响应
           • 高楼火灾场景仿真
           • 60s 时间约束下的覆盖率实验            ▶ 应急案例完成

2026/11    多 UAV 系统级实验
           • SUMO + AirSim 联合仿真
           • 10/20/50 UAV 同时运行实验
           • 空域利用率 / 系统吞吐量评估           ▶ 系统级实验完成

2026/12    数据分析与初稿
           • 整合所有实验数据
           • 撰写 T-ITS 格式 22 页稿件
           • 内部 reviewer (导师 + 同门) 审阅      ▶ 初稿完成

2027/01    打磨阶段
           • 根据内部反馈大修
           • 英文润色（专业 editing service）
           • 准备补充材料（代码、数据集、视频）    ▶ 投稿准备就绪

2027/02    提交前检查
           • Cover letter 撰写
           • 期刊格式调整
           • 推荐审稿人列表准备

2027/03    ◉ 投稿 IEEE T-ITS              ──────────────────────────────

2027/03–08  Round 1 审稿（4-6 月）

2027/09    收到审稿意见
           • 若小修：1-2 月修改                    ▶ 接受目标 2027/12
           • 若大修：3-4 月补实验                  ▶ 接受目标 2028/03
           • 若拒稿：转 TR Part C，调整 framing    ▶ TR-C 投稿 2027/12

2028/06    最终发表（无论哪个期刊）              ◉ 最终目标
──────────────────────────────────────────────────────────────────────────
```

---

## 9. Risikobewertung und Alternativen

### 9.1 Hauptrisiken

**Risiko A: T-ITS-Ablehnung (Wahrscheinlichkeit ~50 %, normale Rate)**
- Antwort: Framing angepasst und in TR Teil C umgewandelt
- Zeitaufwand: zusätzliche 6 Monate
- Pufferung: Bereiten Sie sich seit Version 2 auf Double Framing vor

**Risiko B: Unzureichende Versuchszeit**
- Reaktion: Kernexperimente (Schicht 1-2 + Fall 1-2) sind garantiert, Notfallreaktionsfälle können verschoben werden
- Schlüssel: Wahrnehmungsschicht + Lieferfall müssen vollständig sein

**Risiko C: Unzureichende Leistungslücke zwischen Algorithmus und SOTA**
- Antwort: GauSS-MI ist eine neue Arbeit im Jahr 2025, dieses Papier sollte einen Vorsprung von mehr als einem Jahr haben
- Pufferung: Ablationsexperimente zeigen theoretische Vorteile, eine absolute Zahl von +1,5 dB ist ausreichend

**Risiko D: Verlängerter Überprüfungszeitraum**
- Antwort: Wählen Sie Fast-Track vor der Einreichung (falls von der Zeitschrift bereitgestellt)
- Alternative: Konferenzversion vorbereiten und gleichzeitig bei ICRA 2027 einreichen (keine wiederholte Veröffentlichung, nur als Backup-Plan)

### 9.2 Alternative Einreichungswege (nach Priorität)| Priorität | Tagebuch | WENN | Eignung | Bemerkungen |
|--------|------|----|---------| -----|
| **Bevorzugt** | IEEE T-ITS | 8,5 | ★★★★★ | Hauptinvestitionsziel |
| Alternative 1 | TR Teil C | 8,5 | ★★★★☆ | Abgelehnt und dann übertragen |
| Alternative 2 | IEEE T-RO | 7,4 | ★★★★☆ | Wenn der ITS dies nicht akzeptiert, ist der reine Roboterinhalt |
| Alternative 3 | TR Teil B | 6,0 | ★★★☆☆ | Teilweise methodisch, bedarf mehr Theorie |
| Alternative 4 | Verkehrswissenschaft | 5,4 | ★★★☆☆ | Teilweise mathematisch, erfordert eine Erweiterung der Warteschlangentheorie |

---

## 10. Zusammenfassung: Kernänderungen von v1 → v2

**Zusammenfassung in einem Satz:** Papier C sollte nicht mehr als „Perception Algorithm Paper“ auf Konferenzen eingereicht werden, sondern als „Low-Höhe Economic Enabling Technology Research“, das bei Top-Zeitschriften eingereicht werden soll.

**3 Hauptunterschiede:**
1. **Problemebene:** Einzelne Sanierungsaufgabe → Städtisches Verkehrssystem
2. **Bewertungsumfang:** Wahrnehmungsindikatoren → Fünfschichtiges Indikatorensystem (Wahrnehmung/Planung/Aufgabe/System/Ökonomie)
3. **Akademischer Dialog:** Dialog mit Perception Paper → Dialog mit ITS / UAV-Transportation Top-Journalpapier

**5 neue erhebliche Arbeitslasten:**
1. Gemeinsame Simulationsplattform SUMO + AirSim
2. Drei große Fallstudien (Inspektion, Verteilung, Notfallmaßnahmen)
3. Selbst erstellter UAV-Delivery-Dataset-Datensatz
4. Multi-UAV-Experimente auf Systemebene (10–50 Einheiten gleichzeitig im Betrieb)
5. T-ITS / TR-C-Doppelrahmungsmanuskript

**Zeitaufwand:** v1 soll 4 Monate dauern, v2 soll 12–15 Monate dauern (spiegelt angemessen den Arbeitsaufwand wider, der mit der Einreichung von Manuskripten bei Top-Zeitschriften verbunden ist)

---

> **Beschreibung der Dokumentiteration:** Dies ist die v2-Version des Paper C-Plans (`v2_20260515`). v1 („v1_20260515“) wird als historisches Archiv aufbewahrt, um den Entwurf des „Fast RA-L Path“ für einen späteren Vergleich aufzuzeichnen. Auslösebedingungen für das nächste Update: ① Vervollständigen Sie die experimentellen Daten von 2026/08. ② Erhalten Sie T-ITS-Überprüfungskommentare und aktualisieren Sie dann auf Version 3.