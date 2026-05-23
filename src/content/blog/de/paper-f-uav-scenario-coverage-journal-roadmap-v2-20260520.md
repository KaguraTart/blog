---
title: "Paper F Journal Planning v2: Journal Priority Route for UAV Safety-Critical Scenario Engineering"
description: "Ohne Berücksichtigung der Struktur der Doktorarbeit wird der vorrangige Ausgabeweg der Zeitschrift für Paper F neu geplant, wobei der Schwerpunkt auf der Abdeckung sicherheitskritischer UAV-Szenarien, beschleunigten Tests, Risikosicherung und Hochgeschwindigkeits-Notfallanwendungen liegt."
pubDate: 2026-05-20
updatedDate: 2026-05-23
tags: ["Papier F", "Journalplanung", "UAV", "Szenengenerierung", "Szenenberichterstattung", "Sicherheitskritisch", "beschleunigtes Testen", "Risikogarantie", "TR-C", "T-ITS"]
category: Tech
---

# Paper F Journal Planning v2: Journal Priority Route for UAV Safety-Critical Scenario Engineering

> Der aktuelle Fokus liegt wieder auf **Zeitschriftenpapierausgabe**, die nicht nach Doktorarbeiten geordnet ist.  
> Fazit: Papier F sollte nicht in viele dünne Papiere zerlegt werden, sondern in ein vollständiges, solides und reproduzierbares **T-ITS-Hauptjournal** umgewandelt und dann auf der Grundlage experimenteller Vermögenswerte in TR-C-Anwendungspapiere und Papiere zur Risikosicherungsmethode differenziert werden.

---

## 1. Kernurteil

Die allgemeine Ausrichtung von Paper F lautet immer noch: **Sicherheitskritische UAV-Szenarienentwicklung**. Aber die Logik eines Zeitschriftenartikels und eines Kapitels einer Doktorarbeit ist unterschiedlich. Zeitschriftenrezensenten zahlen nicht für eine komplette Route, sie sind eher besorgt über:

- Ist die Frage konkret genug?
- Ob die Methode klare technische Fortschritte aufweist;
- Ob das Experiment solide genug ist;
- Ob die Grundlinien stark sind;
- Ob die Schlussfolgerung eine unabhängige Zeitschriftengeschichte unterstützt;
- Ob es zum Umfang des Zieljournals passt.

Daher sollte die aktuelle „Planung 4-5“ geändert werden in:

> **Führen Sie zunächst den Benchmark für die F1-Szenenabdeckung und die beschleunigte Generierung gefährlicher F2-Szenen in einem Haupt-T-ITS zusammen. Später werden die TR-C-Notfallanwendung, die T-RO/T-ASE-Risikosicherung und die TR-C/T-ITS-Urban-ODD-Szenengenerierung von derselben Plattform unterschieden. **

### 1.1 22.05.2026 Kalibrierung schreiben: F-Serie muss das „Testmethodenpapier“ und das „Verkehrssystempapier“ trennen.

Papier F ist leicht zu schreiben, da es auch Szenengenerierung, Abdeckung, beschleunigte Tests gefährlicher Szenen, städtisches ODD und Notfallanwendungen für Shandong-Autobahnen bietet. Die neueste Kalibrierung ist:- **F-J1 sollte nicht als TR-C geschrieben werden. ** Sein Hauptthema ist das Testen sicherheitskritischer Szenarien: Wie kann man systematisch gefährliche, aber effektive Szenarien der UAV-Hindernisvermeidung/Navigation in geringer Höhe entdecken? Es ist natürlicher, in T-ITS zu investieren, da der Schwerpunkt auf Sicherheitstests, Simulationsbewertung und Szenarioabdeckung in intelligenten Transportsystemen liegt.
- **F-J2 ist das TR-C-Bewerbungspapier. ** Es muss als Problem des Notfallbetriebs im Hochgeschwindigkeitsverkehr geschrieben werden: Unfallerkennung, Drohnenaufklärung, Zuweisung von Bodenressourcen, Reaktionszeit, Abdeckung, Informationsverzögerung und Verkehrswiederherstellung.
- **F-J4 TR-C nur auslösen, wenn ODD auf Stadtebene an die Verkehrsplanung/Betriebssteuerung zurückgegeben werden kann. ** Wenn Sie OSM einfach in eine lokale Hinderniskombination umwandeln, ähnelt es eher einem Simulationstool oder Benchmark, aber nicht TR-C genug.

Daher kann die „Geschichte“ der F-Serie in zwei Typen unterteilt werden:

| These | Systemgeschichte | Was muss unterstützt werden |
|------|----------|----------------|
| F-J1 | Bei ITS-Sicherheitstests in geringer Höhe mangelt es an Abdeckung und Standards für die Generierung gefährlicher Szenarien | Abdeckungsmetrik, Rate ungültiger Szenarien, Fehlererkennung, Planer-Kreuztest, Multi-Seed-Statistiken |
| F-J2 | Eine schnelle Notfallreaktion erfordert eine UAV-Boden-Zusammenarbeit, um die goldene Reaktionszeit zu verkürzen | Echte Hochgeschwindigkeitstopologie/Unfall-Proxy, Ressourcenzuteilungsmodell, Reaktionszeit, Abdeckung, Zugänglichkeit bei Überlastung, Sensitivitätsanalyse |
| F-J3 | So wandeln Sie die Berichterstattung über den Unfallort in Nachweise zur Risikosicherung um | Deckungs-Risiko-Grenze, Konfidenzintervall, Schätzung seltener Ereignisse, Zuverlässigkeitsindex |
| F-J4 | Wie das ODD der gesamten Stadt lokale Testszenarien in geringer Höhe bestimmt | OSM/POI/Gebäude-/Straßen-/Luftraumkartierung, lokale Risikotreue, stadtübergreifende Generalisierung |

In der Version des Verkehrstagebuchs muss vermieden werden, einfach nur zu sagen: „Wir haben gefährlichere Szenarien generiert“. Eine stärkere Schlussfolgerung wäre:- Welche städtischen Strukturen oder Autobahnabschnitte führen eher zu einem UAV-Ausfall?
- Welche Hinderniskombinationen sind für verschiedene Planer am gefährlichsten?
- Reduziert eine erhöhte Abdeckung tatsächlich das unerkannte Risiko, anstatt nur die Stichprobengröße zu erhöhen?
- Kann die Drohnenaufklärung bei Notfalleinsätzen Versandverluste aufgrund unvollständiger Informationen reduzieren?
- Wann sind Wetter-, Kommunikations- oder Landepunkte eingeschränkt, wann benötigt das System Bodenressourcen, um es zu unterstützen?

Der Grund ist ganz einfach: Benchmark allein kann leicht als zu viele Engineering-Plattformen angesehen werden, und beschleunigtes Testen allein wird in Frage gestellt, ob der Testszenarioraum klar definiert ist. Nachdem die beiden zusammengeführt wurden, wird das Papier von „Ich erzeuge gefährliche Szenarien“ auf Folgendes aktualisiert:

> **Ich habe den sicherheitskritischen UAV-Szenenraum definiert, der die Abdeckung messen, Abdeckungslücken entdecken und abdeckungsgesteuerte Methoden verwenden kann, um realistischere, gefährliche und realisierbare Testszenarien effizienter zu generieren. **

Dies ähnelt eher einem Zeitschriftenartikel.

### 1.2 23.05.2026 Zusammenstellung: Die F-Serie erweitert derzeit nur zwei Hauptlinien

Derzeit wird Paper F nicht entsprechend dem Dissertationskatalog erweitert, sondern zunächst auf der Grundlage der Zeitschriftenausgabe in zwei Hauptzeilen verdichtet. Die Modelle F-J3 und F-J4 bleiben erhalten, nehmen aber nicht die experimentellen Ressourcen des F-J1 weg.| These | Hauptinvestor | Aktuelle Rolle | Aktuelle Strategien |
|------|------|----------|----------|
| F-J1 | T-ITS | Abdeckungsgesteuertes beschleunigtes Testen | Hauptstoß; muss 76 Millionen Explorationsprotokolle, Abdeckungsmetrik, starke Basislinie und planerübergreifende Auswertung verwenden |
| F-J2 | TR-C | Ressourcenzuweisung für Hochgeschwindigkeits-Notfallrettung in Shandong | Die F-J1-Plattform wird nach der Stabilisierung gestartet. Der Fokus liegt auf echter Hochgeschwindigkeitstopologie, Unfall-Proxy, Reaktionszeit und Ressourcenengpässen |
| F-J3 | T-RO / T-ASE / T-ITS | Deckungs-Risiko-Versicherung | Ausgesetzt; Warten Sie, bis F-J1 Fehlerverteilungs- und Abdeckungsstatistiken erstellt, bevor Sie die Risikogrenze nachweisen |
| F-J4 | TR-C / T-ITS | ODD auf Stadtebene zum lokalen UAV-Szenario | Ausgesetzt; Warten Sie, bis die OSM/POI/Building/Airspace-Pipeline stabil genug ist |

Der empfohlene Umriss für die erste Version von F-J1 ist festgelegt auf:1. **Szenarioraum**: Definieren Sie die lokale UAV-Testzelle, die Hindernisgrammatik, dynamische Faktoren, Missionsziele und die Bestimmung ungültiger Szenarien.
2. **Abdeckungsmetrik**: Separate Statistiken zur geometrischen Abdeckung, semantischen Abdeckung, dynamischen Abdeckung, Risikoabdeckung und Fehlermodusabdeckung.
3. **Beschleunigte Generierung**: Nutzen Sie Abdeckungslücken und Ausfallwahrscheinlichkeiten, um die Stichprobe zu steuern und unrealistische oder nicht ausführbare Szenarien herauszufiltern.
4. **Benchmark-Protokoll**: Karten-Seed, Planersatz, Controller-Parameter, Zufalls-Seeds, Fehlerschwellenwerte und statistische Tests vereinheitlichen.
5. **Hauptexperimente**: Vergleichen Sie die Zufalls-, Gitter-/LHS-, BO-, CMA-ES-, RL-Kontroll- und szenisch-stilbeschränkte Generierung mit dieser Methode.
6. **Fehleranalyse**: Erklären Sie, welche Kombinationen aus Hindernissen, Geschwindigkeits-/Höhenbedingungen, Verdeckungen und dynamischen Hindernissen am wahrscheinlichsten einen Fehler auslösen.

Das Urteil nach dieser Sortierung lautet: F-J1 verfolgt zunächst „eine Zeitschriftenarbeit zur Sicherheitsprüfung, die bei T-ITS eingereicht werden kann“ und sollte sich nicht gleichzeitig auf Stadtplanung, Risikotheorie und Shandong-Hochgeschwindigkeitsanwendungen festlegen. F-J2 kann die Story auf den von TR-C geforderten geschlossenen Verkehrsnotfallbetrieb umstellen, nachdem die Szenariobibliothek und die Risikoindikatoren des F-J1 ausgereift sind.

---

## 2. Journal-Prioritätspapier-Portfolio

Es wird empfohlen, vorübergehend **3 Hauptzeitschriften + 1 Reservezeitschrift** einzuplanen, anstatt 5 Artikel gleichzeitig zu bewerben.| Nummer | Papierpositionierung | Vorgeschlagenes Thema | Hauptinvestor | Priorität |
|------|----------|----------|------|--------|
| F-J1 | Workhorse-Methode + Benchmark | Abdeckungsgesteuerte beschleunigte Tests für sicherheitskritische UAV-Navigationsszenarien | T-ITS | Höchste |
| F-J2 | Verkehrsnotfallanwendung | Szenariobewusste UAV-Bodenressourcenzuweisung für Notfallmaßnahmen auf Autobahnen | TR-C | Hoch |
| F-J3 | Coverage-to-Risk-Assurance für sicherheitskritische UAV-Szenariotests | T-RO / T-ASE / T-ITS | Mittel bis hoch |
| F-J4 | Generierung urbaner Szenen | City2Local-UAV: Hierarchische Szenariogenerierung von städtischen ODDs bis hin zu lokalen Hinderniszusammensetzungen | TR-C / T-ITS | Mittel |

**Empfehlung für die Ausführungsreihenfolge: F-J1 -> F-J2 -> F-J3 -> F-J4. **

F-J1 ist die Plattform- und Algorithmusbasis. F-J2 kommt dem Anwendungswert des Transportjournals am nächsten. F-J3 Robotik- oder Automatisierungszeitschriften zur Aktualisierung von Methoden/Theorien. F-J4 kann erst durchgeführt werden, nachdem die OSM/Stadtdatenpipeline ausgereift ist, andernfalls wird es leicht zu einem „Kartenkonvertierungstool“.

---

## 3. Literaturmuster und Lücken

### 3.1 Die Entwicklung autonomer Fahrszenarien ist ausgereift, aber die UAV-Migration reicht nicht ausIm Bereich des autonomen Fahrens gibt es bereits eine komplette Szene-Engineering-Kette. ISO 34502 bietet ein szenariobasiertes Sicherheitsbewertungs-Framework für automatisierte Fahrsysteme [1], und ASAM OpenSCENARIO und OpenODD stellen ausführbare Szenario- und ODD-Beschreibungsstandards bereit [2] [3]. Shuo Fengs beschleunigte Tests und die Generierung von Testszenarienbibliotheken verdeutlichen weiter, dass sicherheitskritische Tests nicht auf natürlichen Zufallsstichproben basieren können, sondern einen datengesteuerten Ansatz verwenden müssen, um die Stichprobeneffizienz kritischer Ereignisse zu verbessern [4] [5] [6].

In den letzten Jahren haben auch Top-Konferenzen diese Richtung weiter vorangetrieben: Scenic verwendet probabilistische Programmiersprachen, um Szenenverteilung und -beschränkungen auszudrücken [7]; SafeBench hat einen Sicherheitsbewertungs-Benchmark erstellt [8]; ScenarioNet extrahiert groß angelegte Verkehrsszenarien aus realen Fahrdaten [9]; AdvSim, KING, ChatScene und FREA generieren sicherheitskritische Szenarien aus den Perspektiven Sensorstörung, Gradientenoptimierung, LLM-Wissensgenerierung bzw. mögliche Gegnerschaft [10] [11] [12] [13].

Die meisten dieser Arbeiten sind jedoch auf das autonome Fahren am Boden ausgerichtet, und die UAV-Szenarien unterscheiden sich erheblich:

- UAV ist eine dreidimensionale Bewegung und die Szenendimensionen umfassen Höhe, Spurneigung, Windfeld, Leistung, Sichtfeldverdeckung und Flugdynamik;
- Zu den gefährlichen UAV-Ereignissen gehören Kollisionen mit Gebäuden, Kollisionen mit Leitungen, das Überqueren von Flugverbotszonen, Konflikte in Korridoren in geringer Höhe, fehlende Starts und Landungen sowie das versehentliche Betreten von Notfallstandorten.
- UAV-Sicherheitstests führen selten zur Reife der ODD-Taxonomie;
- Der UAV-Benchmark konzentriert sich hauptsächlich auf Simulations- und Steuerungsaufgaben und antwortet selten: „Welche Risiken deckt das Szenario ab?“

### 3.2 UAV-Simulation hat eine Grundlage, aber abdeckungsorientierte Sicherheitstests sind noch leerAirSim und Flightmare sind wichtige Grundlagen für die UAV-Simulation [14][15]. AvoidBench hat einen High-Fidelity-Benchmark für die visionsbasierte Hindernisvermeidung mit mehreren Rotoren vorgeschlagen [16]. OmniDrones und Aerial Gym veranschaulichen, dass GPU-parallele UAV-Simulationen und umfangreiches Reinforcement-Learning-Training ausgereift sind [17] [18]. FADS beweist, dass zeitliche logische Sicherheitsspezifikationen in die Drohnensicherheitspipeline einfließen können [19].

Diese Arbeiten bilden die Werkzeuggrundlage für Paper F, sie haben jedoch noch nicht die kritischsten Lücken in Zeitschriftenarbeiten behoben:

> **Wie man den sicherheitskritischen UAV-Szenarioraum definiert, wie man die Abdeckung misst, wie man effizient Long-Tail-Szenarien generiert, die sowohl gefährlich als auch machbar sind, und wie man die Testabdeckung in interpretierbare Risikobewertungen umwandelt. **

Dies ist die Gelegenheit für die F-J1/F-J3.

### 3.3 Der Unterschied zwischen TR-C/T-ITS bestimmt, wie das Papier geschnitten wird

Der intellektuelle Kern von TR-C liegt auf der Transportseite und betont die Auswirkungen neuer Technologien auf die Planung, das Design, den Betrieb, die Steuerung und die Logistik von Transportsystemen [20]. T-ITS deckt explizit Informationstechnologieanwendungen in den Bereichen Sensorik, Kommunikation, Steuerung, Planung, Design, Implementierung, KI und Transportsysteme ab [21].

daher:- **F-J1 Abstimmung für T-ITS**: Weil es um ITS-Sicherheitsbewertung/Szenarioerstellung/UAV-Navigationstests geht.
- **F-J2 hat für TR-C gestimmt**: Weil es sich um den Betrieb des Hochgeschwindigkeits-Notfallsystems und die Ressourcenzuweisung handelt.
- **F-J3 ist für T-RO/T-ASE/T-ITS geeignet**: Je nach Schwerpunkt von Theorie und Experiment kann T-RO/T-ASE für Robotersicherheitstests und T-ITS für Transportsysteme ausgewählt werden.
- **F-J4 stimmt für TR-C/T-ITS**: Wenn der Schwerpunkt auf städtischem Tieflandverkehr und den Auswirkungen von Verkehrssystemen liegt, stimmen Sie für TR-C; Wenn der Schwerpunkt auf Szenenschnittstellen und Simulationsauswertung liegt, stimmen Sie für T-ITS.

### 3.4 Über welche anderen Zeitschriftenrichtungen kann ich schreiben?

Nachdem Sie weiter tiefer gegraben haben, können Sie vier weitere „Kandidatengabeln“ vorbereiten, es wird jedoch nicht empfohlen, jetzt gleichzeitig mit dem Schreiben zu beginnen. Sie eignen sich besser als natürliches Spillover, wenn die experimentelle Plattform des F-J1 ausgereift ist.| Gabeln | Beschreibbare Themen | Kernverkaufsargumente | Kandidatenzeitschriften | Aktuelle Empfehlungen |
|------|----------|----------|----------|----------|
| F-J5 | Szenariobasierter Sicherheitsnachweis für UAV-Einsätze in geringer Höhe | Organisieren Sie Abdeckungs-, Kritikalitäts- und Fehlernachweise in Sicherheitsfällen | Zuverlässigkeitstechnik und Systemsicherheit / IEEE-Transaktionen zur Zuverlässigkeits- / Sicherheitswissenschaft | Warten Sie, bis F-J1 Ergebnisse hat, bevor Sie | schreiben
| F-J6 | Cross-Simulator-Übertragung sicherheitskritischer UAV-Szenarien | Übertragung des Studienszenarios von der Leichtbausimulation auf AirSim/Flightmare/AvoidBench | Robotik und autonome Systeme / Journal of Field Robotics / T-RO | Realistische oder hochpräzise Verifizierung erforderlich |
| F-J7 | Wissensgesteuerte UAV-Szenariogenerierung | Verwenden Sie LLM/VLM/Knowledge Graph, um semantische Gefahrenszenen zu generieren | T-ITS / T-IV / IEEE Open Journal of ITS | Kann mit Paper E verknüpft werden, aber nicht überfordern |
| F-J8 | Multi-UAV-Korridor-Stresstest | Speziell generierte Konflikt-, Stau-, Start- und Landeengpassszenarien für Korridore in geringer Höhe | T-ITS / TR-C / T-IV | Kann mit Paper B | verknüpft werdenUnter ihnen ist die **F-J5 diejenige, die es am meisten wert ist, behalten zu werden**. Wenn die Folgearbeit F-J1 nur beim „Finden weiterer Fehler“ aufhört, wird der Wert des Journals immer noch experimentell sein; Wenn die Szenarioabdeckung jedoch in Zuverlässigkeits-/Sicherheitsnachweise umgewandelt werden kann, kann sie bei Sicherheits- und Zuverlässigkeitszeitschriften wie Reliability Engineering & System Safety oder IEEE Transactions on Reliability [28] [29] eingereicht werden. Alternativ kann auch Safety Science eingesetzt werden, der Schwerpunkt liegt jedoch stärker auf Sicherheitsmanagement, menschlichen Faktoren, Organisation und Unfallverhütung. Wenn die Arbeit noch rein algorithmisch ist, wird sie nicht zur Ersteinreichung empfohlen [30].

F-J6 eignet sich zum Schreiben, wenn echte Drohnen oder High-Fidelity-Simulationsergebnisse vorhanden sind. Sowohl das Journal of Field Robotics als auch Robotics und Autonomous Systems schätzen die Autonomie, Zuverlässigkeit und experimentelle Tiefe von Robotersystemen in realen oder High-Fidelity-Umgebungen [31] [32]. Wenn Sie nur über eine leichtgewichtige Simulation verfügen, reichen Sie diese noch nicht bei dieser Art von Zeitschrift ein.

F-J7 wird derzeit nicht als Hauptlinie empfohlen, da es sich mit der LLM/LTL-Richtung von Paper E überschneiden wird. Es eignet sich besser für spätere Erweiterungen als „wissensbasierte Szenariogenerierung“: LLM ist für das Vorschlagen semantischer Gefahrenszenarien verantwortlich, und Cov-ATUAV ist für die Validierung, Filterung und Quantifizierung der Abdeckung verantwortlich.

F-J8 ist die Stresstestversion von Paper B. Es optimiert nicht mehr die Planung von Hunderten von UAVs, sondern generiert Testszenarien, die Korridorüberlastungen, Vertiport-Engpässe, Ladeengpässe und Konfliktlösungsfehler am besten aufdecken. Über T-ITS oder TR-C kann in diese Richtung abgestimmt werden, es muss jedoch vom Planungsbeitrag von Papier B abgeschnitten werden.

### 3.5 Karte des Kandidatentagebuchs| Tagebuch | Das am besten geeignete Papier F-Schnitt | Warum es angemessen ist | Risiken |
|------|---------|------------|------|
| IEEE T-ITS | F-J1 / F-J4 / F-J8 | Der Geltungsbereich umfasst Sensor-, Steuerungs-, KI-, Planungs- und Transportsysteme in ITS [21] | UAV muss als Low-Altitude-ITS geschrieben werden, nicht als gewöhnlicher Roboter |
| IEEE T-IV | F-J1 / F-J7 / F-J8 | Der Kontext intelligenter Fahrzeuge und automatisierter Mobilität kann Sicherheitstests und Szenariogenerierung unterzogen werden [26] | Bodenfahrzeuge haben eine starke Farbe, UAV muss Fahrzeug-/Verkehrsrelevanz erklären |
| TR-C | F-J2 / F-J4 / F-J8 | Betonen Sie die Auswirkungen neuer Technologien auf Transportabläufe, Kontrolle und Logistik [20] | Nicht für reinen Algorithmus-Benchmark geeignet |
| TR-E | F-J2 | Geeignet für den Einsatz in den Bereichen Logistik, Vertrieb, Lieferkette und Notfallressourcentransport [33] | Wenn das UAV zu viele technische Details aufweist, weicht es von der Logistik ab |
| T-ASE | F-J3 / F-J5 | Automatisierungssysteme, Tests, Evaluierung und Zuverlässigkeitsrahmen sind besser geeignet [27] | Die Methode muss einen Generalisierungswert für Automatisierungssysteme haben |
| T-RO | F-J3 / F-J6 | Robotersicherheitstests, Planung und reale Systemverifizierung können eingereicht werden [34] | Synthetischer Benchmark allein reicht nicht aus |
| IEEE-Transaktionen zur Zuverlässigkeit | F-J5 | Geeignet für Zuverlässigkeitsmodellierung, Risikoquantifizierung, Absicherung [28] | Es sind ernsthafte statistische Garantien erforderlich, nicht nur experimentelle Tabellen |
| Zuverlässigkeitstechnik und Systemsicherheit | F-J5 |Geeignet für sicherheitskritische Systeme, Risikobewertung und Zuverlässigkeitstechnik [29] | Notwendigkeit der Umstellung von der Verbesserung der Algorithmusleistung auf Sicherheitsnachweise |
| Sicherheitswissenschaft | F-J2 / F-J5 | Geeignet für Notfallsicherheit, Unfallverhütung, Sicherheitsmanagement [30] | Reiner UAV-Algorithmus ist nicht geeignet |
| Robotik und autonome Systeme / JFR | F-J6 | Geeignet für autonome Robotersysteme und Feld-/High-Fidelity-Validierung [31] [32] | Systemexperimente müssen stärker sein als Thesennarrative |
| IEEE Open Journal of ITS | F-J1 / F-J2 | Kann als schnelle Open-Access-Alternative verwendet werden [35] | Im Allgemeinen geringere Auswirkungen und Positionierung als T-ITS |**Die aktuelle Reihenfolge der ersten Auswahl bleibt unverändert: F-J1 ist die erste Auswahl für T-ITS, F-J2 ist die erste Auswahl für TR-C, F-J3 wählt T-ASE/T-RO/T-ITS abhängig von der theoretischen Stärke aus und F-J5 ist für zuverlässige Zeitschriften reserviert. **

---

## 4. Die erste große Zeitschrift: F-J1

### 4.1 Vorgeschlagene Themen

**Abdeckungsgesteuerte beschleunigte Tests für sicherheitskritische UAV-Navigationsszenarien**

### 4.2 Einreichungsziele

Hauptmitwirkender: **IEEE-Transaktionen auf intelligenten Transportsystemen**.  
Alternativen: IEEE Transactions on Automation Science and Engineering, IEEE Transactions on Robotics, Robotics and Autonomous Systems.

T-ITS ist am besten geeignet, da die Arbeit als intelligente Transportsicherheitsprüfung geschrieben werden kann: UAVs sind Transportteilnehmer in geringer Höhe, und die Szenariogenerierung dient der Sicherheitsüberprüfung von ITS in geringer Höhe.

### 4.3 Kernthemen

Vorhandene UAV-Hindernisvermeidungs-/Navigationspapiere berichten typischerweise über Erfolgsrate, Kollisionsrate oder Flugbahnlänge, geben aber selten an, ob das Testszenario sicherheitskritische ODDs abdeckt. Zufällig generierte Szenen weisen zwei Probleme auf: Eine große Anzahl von Proben ist nicht gefährlich, und viele gefährliche Proben sind physikalisch nicht realisierbar oder können von keinem Algorithmus vermieden werden.

F-J1 möchte antworten:

> Wie lässt sich der ODD-Betrieb von UAVs in geringer Höhe mit einem begrenzten Simulationsbudget abdecken und die Generierung sicherheitskritischer Szenarien priorisieren, die real, gefährlich und machbar sind und die Fähigkeiten des Algorithmus unterscheiden können?

### 4.4 Methodendesign

Vorschlag für einen Methodennamen: **Cov-ATUAV: Coverage-Guided Accelerated Testing for UAVs**.

Gesamtpipeline:

```text
UAV ODD taxonomy
  -> scenario parameterization
  -> coverage memory
  -> criticality and feasibility scoring
  -> adaptive scenario generation
  -> planner evaluation and coverage update
```

Kernmodule:| Modul | Funktion |
|------|------|
| Szenariogrammatik | Definieren Sie Hindernisse, räumliche Strukturen, dynamische Körper, Windfelder, Sensorgeräusche, Aufgabentypen |
| Abdeckungsspeicher | Parameter-Bins, paarweise/t-weise Abdeckung, Fehlermodi aufzeichnen |
| Kritikalitätsbewertung | Umfassende Exposition, Herausforderung, Beinahe-Unfall, Einschränkungsverletzung |
| Machbarkeitsfilter | Unvermeidliche Kollisionen, unvernünftige Physik und bedeutungslose Missionsszenarien ausschließen |
| Adaptiver Generator | Generieren Sie neue Proben in Abdeckungslöchern und Regionen mit hoher Kritikalität |
| Bewertungsgeschirr | Einheitliche Auswertung mehrerer UAV-Planer und Ausgaberanking-Stabilität |

### 4.5 Daten und Plattform

- Hauptexperiment: 50 m x 50 m x 50 m große lokale UAV-Testzelle.
- Vorhandene Vermögenswerte: 76 Millionen Explorationsprotokolle, die zur Zählung von Abdeckungslöchern, Fehlermodi und der anfänglichen Szeneverteilung verwendet werden.
- Leichte Simulation: selbst erstelltes 3D-Gitter / PyBullet / benutzerdefinierte Dynamik für groß angelegte Suche.
- High-Fidelity-Validierung: AirSim, Flightmare, AvoidBench oder Aerial Gym für die Cross-Simulator-Validierung im kleinen Maßstab [14] [15] [16] [18].

76 Millionen Erkundungen können nicht als Endergebnis geschrieben werden, aber es kann wie folgt geschrieben werden:

> „Wir initialisieren und validieren unsere Szenario-Abdeckungsanalyse anhand eines umfangreichen Explorationsprotokolls mit über 76 Millionen simulierten Rollouts.“

### 4.6 Grundlinien| Grundlinie | Funktion |
|----------|------|
| Zufällige Szenariogenerierung | Grundlegende Stichprobeneffizienz |
| Rasterprobenahme | Einheitliche diskrete Abdeckung |
| Lateinisches Hypercube-Sampling | Parameterraumabdeckung |
| Eingeschränktes Sampling im szenischen Stil | Eingeschränkte Szenengenerierung [7] |
| Vorlagen-Sampling im SafeBench-Stil | Sicherheits-Benchmark im Template-Stil [8] |
| Bayesianische Optimierung | Black-Box-Fehlersuche |
| CMA-ES / Kreuzentropiemethode | Kontinuierliche Parameter-Gefahrensuche |
| Kontroverse Bearbeitung im AdvSim/KING-Stil | Gegnerische Flugbahn/Hindernisstörung [10] [11] |
| Machbare gegnerische Generierung im FREA-Stil | Angemessenes gegnerisches Beispiel [13] |

### 4.7 UAV-Planer

Es müssen mindestens drei Arten von Planern getestet werden, andernfalls stellt die Zeitschrift die Überanpassung nur eines Algorithmus in Frage:

| Planer | Vertreter |
|---------|------|
| Klassik | A* / RRT* / künstliches Potentialfeld / 3DVFH* |
| Optimierung | MPC / sicherer Korridor / B-Spline-Trajektorienoptimierung |
| Lernbasiert | PPO / SAC / Nachahmungslernen / visionsbasierte Politik |

Wenn die Rechenleistung begrenzt ist, sind die garantierten Optionen der ersten Version: RRT*, MPC-lite, PPO-Richtlinie und eine visionsbasierte Basislinie.

### 4.8 Indikatoren| Indikator | Beschreibung |
|------|------|
| Abdeckungsgewinn | Neue Abdeckung alle 1000 Tests |
| Fehlererkennungsrate | Das Verhältnis von Kollision/Beinahe-Unfall/entdeckter Zeitüberschreitung pro Einheitsbudget |
| Beschleunigungsfaktor | Die mehrfache Reduzierung der Anzahl der erforderlichen Tests, um die gleiche Fehlererkennungsrate zu erreichen |
| Machbare Kritikalität | Anteil gefährlicher und vermeidbarer Szenen |
| Ungültiger Szenariopreis | Physikalisch undurchführbares oder bedeutungsloses Stichprobenverhältnis |
| Stabilität des Planerrankings | Stabilität des Planner-Rankings unter verschiedenen Seeds/Szenario-Teilmengen |
| Cross-Simulator-Transfer | Können in der Leichtbausimulation entdeckte Szenarien auf die High-Fidelity-Simulation übertragen werden |

### 4.9 Mindestergebnisse, die eine Zeitschrift veröffentlichen kann

F-J1 erfordert mindestens den Nachweis von:

1. Im Vergleich zu Random/Grid/LHS verbessert Cov-ATUAV den Abdeckungsgewinn bei gleichem Budget erheblich.
2. Im Vergleich zu BO/CMA-ES/kontradiktorischen Basislinien reduziert Cov-ATUAV die Rate ungültiger Szenarien.
3. Im Vergleich zur reinen Fehlersuche können die von Cov-ATUAV generierten Szenen verschiedene UAV-Planer stabiler unterscheiden.
4. Stellen Sie sicher, dass zumindest einige der Hochrisikoszenarien in AirSim/Flightmare/AvoidBench migriert werden können.
5. Ausgabeszenarioschema, Seed, Benchmark-Split und Evaluierungsskript zur Verbesserung der Reproduzierbarkeit von T-ITS.

---

## 5. Das zweite Journal: F-J2-Hochgeschwindigkeits-Notfallanwendung

### 5.1 Vorgeschlagene Themen

**Szenariobewusste Zuweisung von UAV-Bodenressourcen für Notfallmaßnahmen auf der Autobahn**

### 5.2 EinreichungszieleHauptinvestor: **Transportation Research Teil C: Neue Technologien**.  
Alternative: IEEE-Transaktionen auf intelligenten Transportsystemen.

In diesem Artikel geht es darum, UAV in den Transportbetrieb zu versetzen, anstatt es als „UAV-Planungsalgorithmus“ zu schreiben. Das Shandong Expressway Comprehensive Inspection Flight Service System nutzt bereits unbeaufsichtigte Plattformen und Industriedrohnen für Inspektionen, Inspektionen, Notfallmaßnahmen und Datenanalysen [22]. Untersuchungen zur Ressourcenzuweisung für Hochgeschwindigkeitsnotfälle weisen auch auf Probleme wie unvollständige Informationen in der Frühphase eines Unfalls, zeitlich variierende Verkehrsbedingungen und eine unzureichende Verknüpfung zwischen der Standortwahl der Einrichtung und der Ressourcenzuweisung hin [23].

### 5.3 Kernthemen

Bei der Notfallhilfe bei Hochgeschwindigkeitsunfällen geht es nicht nur darum, „die nächstgelegenen Einsatzkräfte zu entsenden“. Wenn sich ein Unfall zum ersten Mal ereignet, sind die Art des Unfalls, die Länge des Staus, die Sperrung der Fahrspuren, gefährliche Stoffe und die Risiken eines Folgeunfalls ungewiss. Der Wert von UAV besteht nicht nur darin, schnell zu fliegen, sondern auch darin, die Informationsunsicherheit im Voraus zu verringern und Fehllieferungen und Verzögerungen zu reduzieren.

F-J2 möchte antworten:

> Wie kann bei einem Unfall mit hoher Geschwindigkeit die UAV-Aufklärung genutzt werden, um die Informationsunsicherheit zu verringern und mit den Ressourcen für Bodenfreiheit, Rettung, Brandbekämpfung, Verkehrspolizei und Wartung koordiniert zu werden, um Reaktionszeit, Räumungszeit und Sekundärrisiken zu reduzieren?

### 5.4 Methodendesign

Vorschlag für einen Methodennamen: **SAFER-UAV: Szenariobewusste schnelle Notfallreaktion mit UAVs**.

Kernstruktur:

```text
Incident scenario generator
  -> UAV first-view dispatch
  -> incident state belief update
  -> ground resource rolling allocation
  -> congestion / clearance simulator
  -> emergency performance evaluation
```

Der Schlüssel besteht darin, die F-J1-Szenenbibliothek in Notfallszenarien umzuwandeln:

- Unfallarten: Auffahrunfall, Überschlag, gefährliche Chemikalien, Straßenbebauung, schlechtes Wetter, Folgeunfälle im Stau.
- Geometrie des Straßenabschnitts: Geraden, Kurven, Rampen, Raststätten, Mautstellen, Brücken, Tunneleinfahrten.
- Unsichere Informationen: Unfallschwere, befahrbare Fahrspuren, Opfer, Ressourcenbedarf, Staulänge.
- Ressourcentypen: UAV, Abschleppwagen, Krankenwagen, Feuerwehr, Verkehrspolizei, Wartungsfahrzeug, temporäre Kontrollausrüstung.

### 5.5 Grundlinien| Grundlinie | Funktion |
|----------|------|
| Nur-Boden-Versand | Keine UAV-Situation |
| Versand der nächstgelegenen Ressource | Nächste Ressourcen zuerst |
| Fester Plan / regelbasierter Versand | Aktuelle Praxisnäherung |
| UAV-zuerst, dann versenden | Eine einfache Strategie, zuerst zuzuschauen und dann zu versenden |
| Zweistufige stochastische Programmierung | Stochastische Optimierungsbasislinie |
| Optimierung des rollenden Horizonts | Starke Optimierungsbasislinie |
| SAFER-UAV voll | Hauptmethode |

### 5.6 Indikatoren

| Indikator | Beschreibung |
|------|------|
| Erstansichtszeit | Der Zeitpunkt, als das UAV zum ersten Mal das Unfallmaterial aufnahm |
| Reaktionszeit | Ankunftszeit des ersten Ressourcenstapels |
| Räumungszeit | Fertigstellungszeit der Räumung |
| Falsche Versandrate | Der Anteil falscher Sendungen, verpasster Sendungen oder unzureichender Ressourcen |
| Sekundäres Unfallrisiko | Sekundärer Unfallrisiko-Proxy |
| Verkehrsverzögerung | Gesamtverzögerungen durch Unfälle |
| Informationswert von UAV | Unsicherheitsreduzierung und Planungsvorteile durch UAV-Aufklärung |
| Abdeckung kritischer Vermögenswerte | UAV-/Bodenressourcen-Abdeckungsfunktionen für Raststätten, Brücken, Tunnel und unfallgefährdete Abschnitte |
| Robustheit gegenüber Informationsverzögerung | Leistungseinbußen bei zunehmender Bildrückgabe, Ereignisbestätigung und Kommunikationsverzögerungen |
| Gerechtigkeit über Straßensegmente hinweg | Unterschied in der Reaktionszeit zwischen abgelegenen Straßenabschnitten und Kernstraßenabschnitten |

Die TR-C-Version erfordert außerdem eine **Systemimplikationstabelle**: Geben Sie unter Berücksichtigung der Anzahl der UAVs, der Anzahl der Start- und Landepunkte, der Konfiguration der Bodenressourcen und der Unfallintensität an, wann sich das System von „UAV-Aufklärung hat offensichtliche Vorteile“ zu „Bodenressourcen oder Überlastung des Straßennetzes wird zum Hauptengpass“ ändert. Diese Tabelle ähnelt eher einem Transportsystempapier als nur durchschnittlichen Antwortzeiten.

### 5.7 Mindestergebnisse, die eine Zeitschrift veröffentlichen kann

F-J2 erfordert mindestens:1. Zeigen Sie ausdrücklich, dass die UAV-Aufklärung die Informationsunsicherheit verringert und nicht nur die Entfernung verringert.
2. Besser als nur am Boden und mit der nächstgelegenen Ressource in Spitzen-/Nacht-/Schlechtwetter-/Multi-Unfall-Szenarien.
3. Vergleichen Sie die Basislinie mit der rollierenden Optimierung oder der zufälligen Optimierung, um den Kompromiss zwischen Echtzeit und Leistung zu veranschaulichen.
4. Notieren Sie die Auswirkungen auf den Transport: unbeaufsichtigte Plattformbereitstellung, Ressourcenvoreinstellung, Notfallreaktionssystem.

---

## 6. Die dritte Zeitschrift: F-J3 Risk Assurance Method

### 6.1 Vorgeschlagene Themen

**Coverage-to-Risk-Assurance für sicherheitskritische UAV-Szenariotests**

### 6.2 Einreichungsziele

Die Hauptwette hängt vom Ergebnis ab:

- Bias-Robotersicherheitstests: T-RO/IEEE-Transaktionen zu Automatisierungswissenschaft und -technik.
- Teilweises Verkehrsaufklärungssystem: T-ITS.
- Teilweise statistische Garantien und Lernrisiken: Zeitschriftenrichtung Maschinelles Lernen / Künstliche Intelligenz.

### 6.3 Warum wird dieser Artikel benötigt?

F-J1 kann antworten „wie man Szenen generiert und abdeckt“, aber Zeitschriftenrezensenten fragen möglicherweise auch:

> Können Sie nun, nachdem Sie diese Szenarien behandelt haben, sagen, wie sicher das System ist? Welcher Zusammenhang besteht zwischen Deckung und tatsächlichem Risiko?

Hier kommt der F-J3 ins Spiel. Er ist kein weiterer Maßstab, sondern verbindet Szenenabdeckung, Wichtigkeitsstichprobe, Szenarioansatz und konforme Risikokontrolle. Der Szenarioansatz von Campi und Garatti bietet eine Machbarkeitswahrscheinlichkeitsgarantie unter zufälligen Szenariobeschränkungen [24], und die konforme Risikokontrolle bietet ein verteilungsfreies Risikokontrollrahmenwerk [25]. Diese können in statistische Garantien für UAV-Sicherheitstests umgewandelt werden.

### 6.4 Methodendesign

Vorschlag für einen Methodennamen: **CovRisk-UAV**.

Kernidee:- Teilen Sie den Raum des UAV-Szenarios in Abdeckungszellen auf;
- Schätzung des Ausfall-/Beinahe-Unfall-/Verletzungsrisikos innerhalb jeder Zelle;
- Verwenden Sie die Wichtigkeitsgewichtung, um die Stichprobenverzerrung beschleunigter Tests zu korrigieren.
- Verwenden Sie eine konforme Risikokontrolle, um eine Risikoobergrenze für endliche Stichproben festzulegen.
- Geben Sie Konfidenzintervalle für die Rangfolge der Planer an, anstatt nur die durchschnittliche Kollisionsrate.

Formal kann das Zielrisiko definiert werden:

$$
R(\pi)=\mathbb{E}_{s\sim P_{\text{ODD}}}[\ell(\pi,s)],
$$

Dabei ist $\pi$ der UAV-Planer, $s$ das Szenario und $\ell$ der Verlust durch Kollision, Beinaheunfall oder Einschränkungsverletzung.

Da das Testszenario aus der beschleunigten Verteilung $Q(s)$ stammt, ist eine Wichtigkeitskorrektur erforderlich:

$$
\hat{R}(\pi)=
\frac{1}{N}\sum_{i=1}^{N}
\frac{P_{\text{ODD}}(s_i)}{Q(s_i)}
\ell(\pi,s_i).
$$

Die erneute Verwendung konformer / Szenario-Grenzen ergibt:

$$
P(R(\pi)\leq \hat{R}_{\alpha}(\pi))\geq 1-\alpha.
$$

### 6.5 Grundlinien| Grundlinie | Vergleichszweck |
|----------|----------|
| Empirische Ausfallrate | Keine Vertrauensgarantie |
| Bootstrap-Konfidenzintervall | Statistische Basis |
| Nur Wichtigkeitsstichprobe | Nur korrekte Stichprobenverzerrung |
| Nur Szenarioansatz | Nur Machbarkeitswahrscheinlichkeitsgrenze |
| Konforme Risikokontrolle | Grundlinie der Risikokontrolle |
| CovRisk-UAV voll | absicherungsbewusste Risikogrenze |

### 6.6 Mindestergebnisse, die eine Zeitschrift veröffentlichen kann

1. Stellen Sie sicher, dass die Kalibrierung der oberen Risikogrenze in synthetischen Szenarien mit bekanntem Risiko gültig ist.
2. Geben Sie verschiedenen Planern in der F-J1-Szenariobibliothek Risikokonfidenzintervalle an.
3. Erklären Sie, dass beschleunigte Tests die ursprüngliche Fehlerrate nicht direkt nutzen können und eine Verteilungskorrektur erfordern.
4. Beweisen Sie, dass die deckungsbezogene Risikogrenze enger oder stabiler ist als naive Zufallstests.

---

## 7. Das vierte Journal: F-J4 urban ODD zur lokalen Szene

### 7.1 Vorgeschlagene Themen

**City2Local-UAV: Hierarchische Szenariogenerierung von städtischen ODDs bis hin zu lokalen Hinderniszusammensetzungen**

### 7.2 Einreichungsziele

Hauptdarsteller: TR-C/T-ITS.  
Es wird vorerst nicht empfohlen, in dieser Richtung an erster Stelle zu stehen, da dies eine stärkere städtische Datenverarbeitung und Fallunterstützung erfordert.

### 7.3 Kernthemen

Obwohl die lokale Szene mit den Maßen 50 x 50 x 50 m kontrollierbar ist, ergeben sich die tatsächlichen Verkehrsrisiken in geringer Höhe aus der städtischen Struktur: Straßengefälle, Bebauungsdichte, Brücken, Raststätten, Flugverbotszonen, Krankenhäuser, Schulen, Verkehrsknotenpunkte und unfallgefährdete Punkte. F-J4 muss ODD auf Stadtebene der lokalen Hinderniszusammensetzung zuordnen.

### 7.4 Methodendesign

```text
City ODD
  -> functional zone and road segment extraction
  -> local UAV test-cell sampling
  -> obstacle grammar instantiation
  -> coverage-aware scenario selection
  -> simulator-ready scenario package
```

### 7.5 Mindestergebnisse, die die Zeitschrift veröffentlichen kann1. Mindestens zwei Fallstudien zu Städten oder Autobahnen.
2. Es kann nachgewiesen werden, dass eine stadtbewusste Generation realistischer ist als eine rein zufällige lokale Generation.
3. Es kann bewiesen werden, dass die generierte lokale Szene hinsichtlich Abdeckung und Kritikalität besser ist als die künstliche Vorlage.
4. Geben Sie eine reproduzierbare Pipeline von städtischen Funktionsbereichen zu lokalen Szenenkombinationen aus.

---

## 8. Empfohlene Route: Welchen Artikel Sie zuerst schreiben sollten

Dasjenige, auf das man sich im Moment am meisten konzentrieren sollte, ist **F-J1**, nicht vier Kapitel gleichzeitig.

### 8.1 Warum F-J1 Priorität hat

- Es wandelt 76 Millionen Explorationsprotokolle in Papierwerte um.
- Es kann F1-Benchmark- und F2-beschleunigte Tests aufnehmen und ist groß genug für Zeitschriften.
- Es hat einen Wiederverwendungswert für die nächsten drei Artikel.
- Am einfachsten ist es, einen vollständigen experimentellen geschlossenen Regelkreis zu bilden: Szenariodefinition, Generierungsmethode, Basislinien, Planer, Metriken und simulationsübergreifende Überprüfung.

### 8.2 Der Hauptlinienbeitrag von F-J1 sollte auf drei reduziert werden

1. **Taxonomie der Abdeckung sicherheitskritischer UAV-Szenarien**
   Definieren Sie UAV-ODD in geringer Höhe, Szenarioparameter, Abdeckungsmetrik und Fehlertaxonomie.

2. **Coverage-gesteuerter beschleunigter Testalgorithmus**
   Generieren Sie gefährliche, aber realisierbare Szenarien in Abdeckungslücken und Regionen mit hoher Kritikalität.

3. **Wiederverwendbares Benchmark- und Bewertungsprotokoll**
   Die Verwendung mehrerer Planer, mehrerer Basislinien und mehrerer Simulationsebenen beweist, dass dieser Benchmark die UAV-Sicherheit stabil bewerten kann.

Schreiben Sie nicht 6-8 Beiträge. Die drei Punkte in der Zeitschrifteneinleitung sind die klarsten.

### 8.3 F-J1 Der Punkt, der am wahrscheinlichsten abgelehnt wird| Risiko | Ursache | Behandlung |
|------|------|------|
| Gilt nur als Simulationsplattform | Benchmark hat keinen algorithmischen Beitrag | Abdeckungsgesteuerte beschleunigte Tests müssen hervorgehoben werden |
| Gilt als vom autonomen Fahren kopiert | Fehlende UAV-Funktionen | Schwerpunkt auf 3D-Dynamik, Wind, Batterie, Hindernissen in geringer Höhe, Landung/Notfallaufgaben |
| Die gefährliche Szene gilt als unrealistisch | gegnerisch zu stark | Machbarkeits- und Natürlichkeitsfilter hinzufügen |
| Gilt nur für einen einzelnen Planer | Überanpassung | Mindestens 4 Arten von Planern |
| Gilt als Transportsystem ohne Bedeutung | UAVs sind nur Roboter | Verfasst als Low-Altitude ITS Safety Evaluation |

---

## 9. Abbau neuer experimenteller Aufgaben

### Woche 1: F-J1-Problemformulierung einfrieren

- Festes Zieljournal: T-ITS.
- Haupttitel und drei Beiträge korrigiert.
- 50 m x 50 m x 50 m große Testzelle einfrieren.
- Definieren Sie die Szenenparametertabelle: Geometrie, Hindernis, dynamischer Agent, Wetter, Sensor, Aufgabe, Risikobezeichnung.

### Wochen 2–3: Verarbeitung von 76 Millionen Erkundungsprotokollen

- Probenahme von 10.000 bis 50.000 Artikeln zur vorläufigen Analyse.
- Lücken in der statistischen Erfassung.
- Clustering-Fehlermodi: Kollision, Beinahe-Unfall, Timeout, Oszillation, Energieverletzung, undurchführbare Szene.
- Ausgabe von zwei Kernkarten: Abdeckungs-Heatmap und Fehlertaxonomie.

### Wochen 4–6: Implementierung von Baseline-Generatoren- Zufall, Raster, LHS.
- Beschränkter Generator im szenischen Stil.
-BO/CMA-ES.
- Bearbeitung kontroverser Hindernisse.
- Machbarer Kritikalitätsfilter.

### Wochen 7–9: Implementierung von Cov-ATUAV

- Abdeckungsspeicher.
- Kritikalitätsbewertung.
- Machbarkeitsfilter.
-adaptiver Generator.
- Planer-Bewertungsgeschirr.

### Wochen 10–12: Hauptexperiment

- Vergleichen Sie die Fehlererkennungsrate, den Abdeckungsgewinn, die Ungültigkeitsrate und den Beschleunigungsfaktor.
- Testen Sie RRT*, MPC-lite, PPO, Vision-Richtlinie.
- Führen Sie eine AirSim-/Flightmare-/AvoidBench-Subset-Migrationsüberprüfung durch.

### Wochen 13–16: Schreiben des ersten Entwurfs von T-ITS

- Die Einführung konzentriert sich auf ITS-Sicherheitstests in geringer Höhe.
- Die damit verbundenen Arbeiten gliedern sich in szenariobasierte Sicherheitsbewertung, Generierung sicherheitskritischer Szenarien, UAV-Simulation und Hindernisvermeidung.
- Experimente verwenden Haupttabelle + Abdeckungsdiagramm + Fehlererkennungskurve + simulatorübergreifende Übertragung.

---

## 10. Dinge, die derzeit nicht empfohlen werden- Schreiben Sie nicht 5 Aufsatztitel auf einmal und arbeiten Sie dann parallel daran.
- Führen Sie F-J4 City ODD nicht zuerst aus, da die Datenpipeline die Ausgabe des ersten Artikels verlangsamt.
- Mischen Sie Shandong Expressway Emergency und F-J1 nicht in einem Artikel, da sonst die Hauptlinie von T-ITS verstreut wird.
- Schreiben Sie 76 Millionen Untersuchungen nicht als Endergebnis, es handelt sich jetzt um einen Datenbestand und nicht um eine Schlussfolgerung.
- Melden Sie nicht nur die Kollisionsrate, sondern auch die Abdeckung, die Kritikalität, die Ungültigkeitsrate und die Stabilität des Planerrankings.

---

## 11. Referenzen

[1] Internationale Organisation für Normung. „ISO 34502:2022 Straßenfahrzeuge – Testszenarien für automatisierte Fahrsysteme – Szenariobasierter Sicherheitsbewertungsrahmen.“ 2022. URL: <https://www.iso.org/standard/78951.html>

[2] ASAM. „ASAM OpenSCENARIO DSL: Schlüsselterminologie und konzeptioneller Überblick.“ URL: <https://publications.pages.asam.net/standards/ASAM_OpenSCENARIO/ASAM_OpenSCENARIO_DSL/latest/conceptual-overview/key_terms.html>

[3] ASAM. „ASAM OpenODD: Modell zur ASAM OpenSCENARIO DSL-Zuordnungsreferenz.“ URL: <https://publications.pages.asam.net/standards/ASAM_OpenODD/ASAM_OpenODD/latest/pecification/09_openscenario_dsl/09_01_overview.html>[4] Shuo Feng, Xintao Yan, Haowei Sun, Yiheng Feng und Henry X. Liu. „Intelligenter Fahrintelligenztest für autonome Fahrzeuge in naturalistischer und kontroverser Umgebung.“ *Nature Communications*, 12:748, 2021. DOI: 10.1038/s41467-021-21007-8. URL: <https://doi.org/10.1038/s41467-021-21007-8>

[5] Shuo Feng, Yiheng Feng, Chunhui Yu, Yi Zhang und Henry X. Liu. „Testen der Szenariobibliotheksgenerierung für vernetzte und automatisierte Fahrzeuge, Teil I: Methodik.“ *IEEE Transactions on Intelligent Transportation Systems*, 22(3):1573-1582, 2021. DOI: 10.1109/TITS.2020.2972211. URL: <https://doi.org/10.1109/TITS.2020.2972211>[6] Shuo Feng, Yiheng Feng, Haowei Sun, Shan Bao, Yi Zhang und Henry X. Liu. „Testen der Szenariobibliotheksgenerierung für vernetzte und automatisierte Fahrzeuge, Teil II: Fallstudien.“ *IEEE Transactions on Intelligent Transportation Systems*, 22(9):5635-5647, 2021. DOI: 10.1109/TITS.2020.2988309. URL: <https://doi.org/10.1109/TITS.2020.2988309>

[7] Daniel J. Fremont, Tommaso Dreossi, Shromona Ghosh, Xiangyu Yue, Alberto L. Sangiovanni-Vincentelli und Sanjit A. Seshia. „Szenisch: Eine Sprache zur Szenariospezifikation und Szenengenerierung.“ *Vorträge der 40. ACM SIGPLAN-Konferenz zum Thema Programmiersprachendesign und -implementierung (PLDI)*, 2019. DOI: 10.1145/3314221.3314633. URL: <https://people.eecs.berkeley.edu/~sseshia/pubs/b2hd-fremont-pldi19.html>[8] Chejian Xu, Wenhao Ding, Weijie Lyu, Zuxin Liu, Shuai Wang, Yihan He, Hanjiang Hu, Ding Zhao und Bo Li. „SafeBench: Eine Benchmarking-Plattform zur Sicherheitsbewertung autonomer Fahrzeuge.“ *Advances in Neural Information Processing Systems 35 (NeurIPS 2022) Datasets and Benchmarks Track*, 2022. URL: <https://proceedings.neurips.cc/paper_files/paper/2022/hash/a48ad12d588c597f4725a8b84af647b5-Abstract-Datasets_and_Benchmarks.html>

[9] Quanyi Li, Zhenghao Peng, Lan Feng, Zhizheng Liu, Chenda Duan, Wenjie Mo und Bolei Zhou. „ScenarioNet: Open-Source-Plattform für groß angelegte Simulation und Modellierung von Verkehrsszenarien.“ *Advances in Neural Information Processing Systems 36 (NeurIPS 2023) Datasets and Benchmarks Track*, 2023. URL: <https://proceedings.neurips.cc/paper_files/paper/2023/hash/0c26a501df8fb919a0350e2df06b5d39-Abstract-Datasets_and_Benchmarks.html>[10] Jingkang Wang, Ava Pun, James Tu, Sivabalan Manivasagam, Abbas Sadat, Sergio Casas, Mengye Ren und Raquel Urtasun. „AdvSim: Generierung sicherheitskritischer Szenarien für selbstfahrende Fahrzeuge.“ *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2021. DOI: 10.1109/CVPR46437.2021.00978. URL: <https://openaccess.thecvf.com/content/CVPR2021/html/Wang_AdvSim_Generating_Safety-Critical_Scenarios_for_Self-Driving_Vehicles_CVPR_2021_paper.html>

[11] Niklas Hanselmann, Katrin Renz, Kashyap Chitta, Apratim Bhattacharyya und Andreas Geiger. „KING: Generierung sicherheitskritischer Fahrszenarien für robuste Nachahmung über Kinematikgradienten.“ *European Conference on Computer Vision (ECCV)*, 2022. DOI: 10.1007/978-3-031-19839-7_20. URL: <https://is.mpg.de/ps/publications/king_geiger2022>[12] Jiawei Zhang, Chejian Xu und Bo Li. „ChatScene: Wissensbasierte sicherheitskritische Szenariogenerierung für autonome Fahrzeuge.“ *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2024, S. 15459-15469. URL: <https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_ChatScene_Knowledge-Enabled_Safety-Critical_Scenario_Generation_for_Autonomous_Vehicles_CVPR_2024_paper.html>

[13] Keyu Chen, Yuheng Lei, Hao Cheng, Haoran Wu, Wenchao Sun und Sifa Zheng. „FREA: Machbarkeitsgesteuerte Generierung sicherheitskritischer Szenarien mit angemessener Adversarialität.“ arXiv:2406.02983, 2024. URL: <https://arxiv.org/abs/2406.02983>

[14] Shital Shah, Debadeepta Dey, Chris Lovett und Ashish Kapoor. „AirSim: Hochpräzise visuelle und physikalische Simulation für autonome Fahrzeuge.“ *Field and Service Robotics*, Springer Proceedings in Advanced Robotics, 2017; arXiv:1705.05065. URL: <https://arxiv.org/abs/1705.05065>[15] Yunlong Song, Selim Naji, Elia Kaufmann, Antonio Loquercio und Davide Scaramuzza. „Flightmare: Ein flexibler Quadrocopter-Simulator.“ *Proceedings of the 4th Conference on Robot Learning (CoRL)*, PMLR 155, 2021. URL: <https://proceedings.mlr.press/v155/song21a.html>

[16] Hang Yu, Guido C. H. E. de Croon und Christophe De Wagter. „AvoidBench: Eine High-Fidelity Vision-basierte Benchmarking-Suite zur Hindernisvermeidung für Multirotoren.“ arXiv:2301.07430, 2023. URL: <https://arxiv.org/abs/2301.07430>

[17] Botian Xu, Feng Gao, Chao Yu, Ruize Zhang, Yi Wu und Yu Wang. „OmniDrones: Eine effiziente und flexible Plattform für verstärkendes Lernen in der Drohnensteuerung.“ *IEEE Robotics and Automation Letters*, 9(3):2838-2844, 2024. DOI: 10.1109/LRA.2024.3356168. URL: <https://ieeexplore.ieee.org/document/10409589/>[18] Mihir Kulkarni, Theodor J. L. Forgaard und Kostas Alexis. „Aerial Gym: Isaac Gym Simulator für Flugroboter.“ arXiv:2305.16510, 2023. URL: <https://arxiv.org/abs/2305.16510>

[19] Yash Vardhan Pant, Max Z. Li, Alena Rodionova, Rhudii A. Quaye, Houssam Abbas, Megan S. Ryerson und Rahul Mangharam. „FADS: Ein Rahmen für die Sicherheit autonomer Drohnen unter Verwendung einer auf zeitlicher Logik basierenden Flugbahnplanung.“ *Transportation Research Part C: Emerging Technologies*, 130:103275, 2021. DOI: 10.1016/j.trc.2021.103275. URL: <https://doi.org/10.1016/j.trc.2021.103275>

[20] Elsevier. „Verkehrsforschung Teil C: Neue Technologien: Ziele und Umfang.“ URL: <https://www.sciencedirect.com/journal/transportation-research-part-c-emerging-technologies>

[21] IEEE Intelligent Transportation Systems Society. „IEEE-Transaktionen auf intelligenten Transportsystemen (T-ITS): Umfang.“ URL: <https://ieee-itss.org/pub/t-its/>[22] Shandong Expressway Group Co., Ltd. „‚Shandong Expressway Comprehensive Inspection Flight Service System‘ geht online.“ 2025. URL: <https://www.sdhsg.com/article/72553>

[23] Zhao Xiangmo, Zhao Yifei, Lu Nengchao et al. „Eine Überprüfung der Forschung zur Zuweisung wichtiger Ressourcen für Notfälle bei Verkehrsunfällen auf der Autobahn.“ *Transactions of Transportation Engineering*, 2024. DOI: 10.19818/j.cnki.1671-1637.2024.06.001. URL: <https://transport.chd.edu.cn/cn/article/doi/10.19818/j.cnki.1671-1637.2024.06.001>

[24] Marco C. Campi und Simone Garatti. „Die genaue Machbarkeit randomisierter Lösungen unsicherer konvexer Programme.“ *SIAM Journal on Optimization*, 19(3):1211-1230, 2008. DOI: 10.1137/07069821X. URL: <https://epubs.siam.org/doi/10.1137/07069821X>

[25] Anastasios N. Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei und Tal Schuster. „Konforme Risikokontrolle.“ *International Conference on Learning Representations (ICLR)*, 2024. URL: <https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf>[26] IEEE Intelligent Transportation Systems Society. „IEEE-Transaktionen auf intelligenten Fahrzeugen.“ URL: <https://ieee-itss.org/pub/t-iv/>

[27] IEEE Robotics and Automation Society. „IEEE-Transaktionen zur Automatisierungswissenschaft und -technik.“ URL: <https://www.ieee-ras.org/publications/t-ase>

[28] IEEE Reliability Society. „IEEE-Transaktionen zur Zuverlässigkeit.“ URL: <https://rs.ieee.org/publications/transactions-on-reliability/>

[29] Elsevier. „Zuverlässigkeitstechnik und Systemsicherheit: Ziele und Umfang.“ URL: <https://www.sciencedirect.com/journal/reliability-engineering-and-system-safety>

[30] Elsevier. „Sicherheitswissenschaft: Ziele und Umfang.“ URL: <https://www.sciencedirect.com/journal/safety-science>

[31] Wiley. „Journal of Field Robotics: Überblick.“ URL: <https://onlinelibrary.wiley.com/journal/15564967>

[32] Elsevier. „Robotik und autonome Systeme: Ziele und Umfang.“ URL: <https://www.sciencedirect.com/journal/robotics-and-autonomous-systems>[33] Elsevier. „Verkehrsforschung Teil E: Logistik- und Transportrückblick: Ziele und Umfang.“ URL: <https://www.sciencedirect.com/journal/transportation-research-part-e-logistics-and-transportation-review>

[34] IEEE Robotics and Automation Society. „IEEE-Transaktionen zur Robotik.“ URL: <https://www.ieee-ras.org/publications/t-ro>

[35] IEEE Intelligent Transportation Systems Society. „IEEE Open Journal of Intelligent Transportation Systems.“ URL: <https://ieee-itss.org/pub/oj-its/>

---

## Anhang: Fazit dieser Optimierung

1. Wenn Zeitschriftenpriorität eingeräumt wird, sollte Papier F nicht in viele kleine Papiere aufgeteilt werden.
2. Der erste Artikel sollte Benchmark- und beschleunigte Tests kombinieren, um das Hauptpapier von T-ITS zu bilden.
3. Notfallanträge für die Shandong-Schnellstraße sollten als TR-C unabhängig sein und nicht in den ersten Artikel eingemischt werden.
4. Risikosicherungspapiere können als Reserve für hochrangige Methodenzeitschriften im mittleren bis späten Stadium verwendet werden.
5. Urban ODD zur lokalen Szenengenerierung steht vorübergehend an vierter Stelle und wird weiterentwickelt, sobald die Datenpipeline stabilisiert ist.