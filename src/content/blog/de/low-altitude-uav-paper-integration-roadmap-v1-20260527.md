---
title: "Low Altitude Planning Thesis Matrix v1: Integration bereits geschriebener Artikel, anschließende Themenauswahl und Zotero-Literaturliste"
description: "Integrieren Sie konfliktfreie Pfadplanung, dreistufige Planung von Hunderten von UAVs, informationstheoretisch gestützte 3DGS-Aktiverfassungsplanung und andere schriftliche Anweisungen, planen Sie nachfolgende Planungspapiergruppen für geringe Höhen und stellen Sie Top-Zeitschriften und hochrelevante arXiv-Referenzen für 2021–2026 bereit."
pubDate: 2026-05-27
updatedDate: 2026-05-27
tags: ["Planung in geringer Höhe", "UAV", "Abschlussarbeitsplanung", "Zotero", "T-ITS", "TR-C", "T-RO", "AAAI", "ICRA", "3DGS", "MERGEL"]
category: Tech
sourceHash: "2c029666156e0305f30131a09abf73c03a2ccbd7"
---

# Low Altitude Planning Thesis Matrix v1: Integration bereits geschriebener Artikel, anschließende Themenauswahl und Zotero-Literaturliste

> Dieser Artikel fasst die bisher verfassten Arbeiten zu UAVs in geringer Flughöhe wieder in ein **Papierportfolio** zusammen.  
> Das Ziel besteht nicht darin, viele Ideen auszubreiten und zu schreiben, sondern zu klären: Welche Artikel haben bereits Gestalt angenommen, welche können weiterhin zu Top-Zeitschriften-/regulären Konferenzbeiträgen gemacht werden und welche Literaturunterstützung, experimentellen Ressourcen und Einreichungspositionierung werden für jeden Beitrag benötigt.

---

## 1. Es gibt derzeit Artikel und Hauptzeilenpositionierung

Derzeit gibt es drei Kernartikel, die die inhaltliche Grundlage bilden:

| Nummer | Vorhandener Inhalt | Aktuelle Positionierung | Empfohlene Hauptinvestition | Kernurteil |
|---|---|---|---|---|
| Papier A | Konfliktfreie Pfadplanung / PPO / MAPPO / Multi-UAV-Konfliktlösung | Robuste Konfliktlösung im Tiefflugroutennetz | IEEE T-ITS / IEEE T-RO / ICRA-IROS | Sie können nicht einfach PPO schreiben, Sie müssen „Kompromiss zwischen Sicherheit und Effizienz bei nicht kooperativen UAVs, Kommunikationsverschlechterung und Korridoren mit hoher Dichte“ schreiben
| Papier B | Hunderte von UAV-Dreischicht-Dispatching | Städtische Tieflandlogistik/Notfallsystemeinsatzplanung | TR-C zuerst, T-ITS-Backup | Dies ist ein Papier zum Transportsystem, das sich auf Kapazität, Verzögerung, Warteschlangenstabilität, Vertiport/Lade-/Korridor-Engpass konzentriert |
| Papier C | UAV 3DGS aktive Sensorplanung basierend auf Informationstheorie | Aktive Erfassung + digitaler Zwilling in geringer Höhe + geschlossener Planungskreislauf | T-RO / T-ITS / ICRA-IROS | Bei der Einreichung in einem Transportjournal muss nachgewiesen werden, dass die aktive Sensorik die Inspektions-, Notfall-, Hindernisvermeidungs- oder Betriebskontrollindikatoren verbessert |

Diese drei Artikel konnten ein sehr stabiles Planungsdreieck in geringer Höhe bilden:

```text
Paper A：战术安全
  多 UAV conflict resolution / no-conflict planning / PPO-MAPPO / CBF / RMADER

Paper B：系统运营
  hundred-UAV scheduling / queue stability / Lyapunov / multimodal logistics

Paper C：环境认知
  3DGS active perception / Fisher information / NBV / safe reconstruction
```

Für nachfolgende neue Arbeiten ist es am besten, sich um dieses Dreieck herum auszudehnen und nicht in einer völlig fremden Richtung zu beginnen.

---

## 2. Gesamturteil zur Einreichung

Die Planungsrichtung für niedrige Höhen kann in drei Kategorien von Papieren unterteilt werden, und verschiedene Kategorien haben unterschiedliche Überprüfungsstandards:| Geben Sie | ein Repräsentative Papiere | Überprüfen Sie die Aufmerksamkeit | Empfohlene Veranstaltungsorte |
|---|---|---|---|
| Papiere zum Transportsystem | Papier B, Notfallressourcenzuweisung, Planung des Straßennetzes in geringer Höhe | Echte Verkehrsprobleme, Systemindikatoren, Glaubwürdigkeit der Daten/Simulationen, politische oder betriebliche Auswirkungen | TR-C, T-ITS |
| Roboterplanungspapier | Papier A, Papier C, digitale Zwillingsplanung | Algorithmusneuheit, Echtzeit, Sicherheit, Hardware-/Simulationsverifizierung | T-RO, RA-L+ICRA/IROS, T-ITS |
| KI-Methodenpapiere | VERA-UAV, CloudBrain-Agent, Szenenbeschleunigungsgenerierung | Benchmark-Schwierigkeit, Theorie/Verifizierungsmechanismus, Modellverallgemeinerung, Reproduzierbarkeit | AAAI, IJCAI, NeurIPS/ICLR-Workshop, T-ITS-Erweiterung |

Die offizielle Positionierung von TR-C betont Transportsysteme und neue Technologien, und der intellektuelle Kern liegt auf der Transportseite [1]; T-ITS umfasst Sensorik, Kommunikation, Steuerung, Planung, Design, Implementierung und andere moderne Transportsystemtechnologien [2]. Deshalb:

- **Papier B/Zuweisung von Notfallressourcen/Planung von Straßennetzen in geringer Höhe**: Priorisieren Sie das Schreiben gemäß der Betriebslogik des Transportsystems von TR-C.
- **Papier A / Papier C**: Sie können für T-RO oder ICRA/IROS stimmen; Wenn Sie auf T-ITS umsteigen, müssen Sie die Verkehrssystemindikatoren ergänzen.
- **Papier-LLM-Agent vom Typ E/G**: Der erste Artikel ist besser für AAAI/IJCAI geeignet und die Zeitschriftenversion ist auf T-ITS erweitert.

---

## 3. Papiermatrix: Es wird empfohlen, 7 Artikel zu bilden, die erweitert werden können

### 3.1 Papier A: Robuste, konfliktfreie Planung eines Flugroutennetzes in geringer Höhe

**Vorgeschlagenes Thema:** Robuste konfliktfreie UAV-Korridorplanung unter nicht kooperativer Verkehrs- und Kommunikationsverschlechterung**Entsprechend vorhandenen Artikeln:** Konfliktfreie Pfadplanung, PPO/MAPPO, UAV-Konfliktlösung, UAV-Konfliktumgebungskonstruktion.

**Kernfrage:** Wie können in einem städtischen Luftstraßennetz in geringer Höhe mehrere UAVs die Trennungssicherheit aufrechterhalten und gleichzeitig Verzögerungen, zusätzliche Entfernungen und Durchsatzverluste unter den Bedingungen lokaler Beobachtung, Kommunikationsverzögerungen, Positionierungsfehlern und nicht kooperativem Flugzeugeinsatz kontrollieren?

**Methodenroute:**

- strategische Ebene: anfängliche Pfad- und Zeitfensterzuweisung basierend auf dem Streckennetz;
- taktische Ebene: MAPPO/PPO-Ausgabegeschwindigkeit, Höhe oder seitliche Versatzaktion;
- Sicherheitsschild: Flugbahnkontrolle im CBF-QP-/ORCA-/RMADER-Stil;
- Fallback-Schicht: Wechsel zur konservativen Prioritätsregel, wenn die Kommunikation nachlässt;
- Bewertung: 30/50 Flugzeuge werden trainiert und 100/200 Flugzeuge getestet, wobei vier Szenarien abgedeckt werden: kooperativ, nicht kooperativ, Kommunikationsverlust und Korridor mit hoher Dichte.

**Wichtige Referenzen:**

Das stabile Multi-Agent-Training von MAPPO/PPO kann von Yu et al. unterstützt werden. [3]; MAT und FACMAC bieten eine stärkere MARL-Basislinie [4,5]; HAPPO/HATRPO bietet eine Referenz zur Multi-Agent-Richtlinienoptimierung für Vertrauensregionen [6]. Auf der Roboterseite unterstützen EGO-Swarm, MADER, RMADER, RACER, PANTHER und GCOPTER jeweils die dezentrale Schwarmplanung, die gemeinsame Nutzung von Flugbahnen unter Verzögerung, die kollaborative Erkundung, die wahrnehmungsbewusste Planung und die Flugbahnoptimierung von Multikoptern [7-12].

**Innovationsvorschläge:**1. Upgrade der „PPO-konfliktfreien Pfadplanung“ von einer einfachen RL-Aufgabe auf eine Sicherheitskontrolle für Verkehrskorridore in geringer Höhe.
2. Einführung von Kommunikationsverschlechterungen und nicht kooperativen UAVs, um die eigentliche Betriebsgrenze zu bilden, die T-ITS mehr Sorgen bereitet.
3. Verwenden Sie Lernrichtlinien + formelles/Sicherheitsschild, um die mangelnde Sicherheit von reinem RL zu vermeiden.
4. Trafficisierung von Indikatoren: LoWC, NMAC, Konfliktanzahl, durchschnittliche Verzögerung, zusätzliche Entfernung, Durchsatz, Laufzeit.

### 3.2 Papier B: Dreischichtige hierarchische Planung von Hunderten von UAVs

**Vorgeschlagenes Thema:** H-LyraUAV: Warteschlangenstabile hierarchische Planung für die UAV-Logistik in geringer Höhe im Hundertmaßstab

**Entsprechend vorhandenen Artikeln:** Papier B dreistufige Terminplanung.

**Kernfrage:** Wie kann eine UAV-Flotte mit hundert Ebenen unter dynamischen Anforderungen, begrenzter Vertiport-/Lade-/Korridorkapazität und multimodalen Transportbeschränkungen stabil, effizient und sicher arbeiten?

**Methodenroute:**

- Makroebene: Nachfragewarteschlange, Neupositionierung der Flotte, Moduswahl;
- Mesoschicht: Vertiport, Ladepad, Korridor-Slot-Planung;
- Mikroebene: Machbarkeit der Energie-/Sicherheits-/konfliktbewussten Entwicklung;
- Theorie: Lyapunov-Drift plus Strafe garantiert Warteschlangenstabilität und Kosten-Rückstands-Kompromiss;
- Daten: synthetisches Stadtgitter + OSM/POI/NYC TLC/Chicago Taxi/SUMO-Erweiterung.

**Wichtige Referenzen:**Das TR-C-Drohnen-Lieferverkehrsmanagement in geringer Höhe hat die Ressourcenzuweisung und Konfliktlösung im städtischen Raum in geringer Höhe direkt erörtert [13]; passagierzentriertes UAM, Fairness und betriebliche Effizienzforschung unterstützen die Gestaltung der Servicequalität [14]; Ladestations-Liefernetzwerk, kapazitätsbeschränkte UAM-Planung, sichere Lernplanung, unterstützende Infrastrukturkapazität und sichere Online-Planung [15-17]; LKW-Drohne / UAV-UGV multimodale Zustellung unterstützt multimodale Erweiterung [18,19].

**Innovationsvorschläge:**

1. Online-Scheduling mit drei Schichten auf Hunderten von Ebenen anstelle von Offline-Routing/Netzwerkdesign.
2. Die Warteschlangenstabilität wird zum Hauptthema der Theorie, und das Lernmodul macht nur Vorhersagen oder Wertschätzungen.
3. Bewerten Sie gleichzeitig Verzögerung, Durchsatz, Rückstand, Ladeauslastung, Vertiport-Engpass und Korridorüberlastung.
4. Die Schlussfolgerung des Verkehrssystems kann antworten: Wann muss der Verkehr begrenzt werden, wo liegt der Engpass und wann ist der Einsatz von UAVs dem multimodalen Fallback unterlegen?

### 3.3 Papier C: FIM-3DGS UAV-Aktiverfassungsplanung

**Vorgeschlagenes Thema:** FIM-3DGS: Fisher-Information-gesteuerte aktive Wahrnehmungsplanung für eine sichere UAV-Rekonstruktion

**Entspricht bestehenden Artikeln:** Paper C, Next-Best-View und NeRF/3DGS, Information Theory Active Sensing.

**Kernfrage:** Wie können UAVs unter begrenzten Flugzeit-, Energie- und Sicherheitsbeschränkungen aktiv Aussichtspunkte auswählen, um die Konvergenz der 3DGS-Karte zu beschleunigen und Planungsaufgaben in geringer Höhe zu erfüllen?

**Methodenroute:**- Szenendarstellung: inkrementelles 3D-Gaußsches Splatting;
- Informationsmetrik: Fisher-Information erstellen / erwarteter Informationsgewinn für Gaußsche Parameter oder gerenderte Jacobi-Parameter;
- Planer: Generierung von NBV-Kandidaten + sicherer Korridor / CBF-Beschränkung;
- Aufgabenkopplung: Die Rekonstruktionsqualität wird nicht nur über PSNR/SSIM berichtet, sondern auch über Hindernisabruf, Planungskollisionsrate und Inspektionsabdeckung;
- Basislinien: ActiveNeRF, FisherRF, GS-Planner, HGS-Planner, POp-GS, Grenzerkundung.

**Wichtige Referenzen:**

Der ursprüngliche 3DGS-Text bietet eine explizite Darstellung des Strahlungsfeldes in Echtzeit [20]; ActiveNeRF ist ein früher Vertreter der neuronalen Wiedergabe aktiver Wahrnehmung [21]; FisherRF unterstützt direkt die aktive Ansichtsauswahl von Fisher-Informationen und liefert 3DGS-Backend-Ergebnisse mit 70 fps [22]; GS-Planner, HGS-Planner, POp-GS und NVF unterstützen die 3DGS/NBV-Wettbewerbslinie von 2024–2025 [23–26].

**Innovationsvorschläge:**

1. Upgrade von „3DGS NBV“ auf „aktive Wahrnehmung im Dienste der UAV-Sicherheitsplanung“.
2. Verwenden Sie Fisher-Informationen, um CRB/Rekonstruktionsunsicherheit/Planungssicherheit miteinander zu verbinden.
3. Erweitern Sie von visuellen Indikatoren zu Verkehrs-/Roboteraufgabenindikatoren: Pfaddurchführbarkeitsrate, Hinderniserkennungsrate, Abdeckungsrate für Notfallinspektionen.
4. Führen Sie eine szenarioübergreifende Verallgemeinerung auf MatrixCity/AirSim/selbstgebauten städtischen Zellen in geringer Höhe durch.

### 3.4 Papier D: Abdeckung sicherheitskritischer Szenen in geringer Höhe und beschleunigte Tests**Vorgeschlagenes Thema:** Abdeckungsgesteuerte beschleunigte Tests für sicherheitskritische UAV-Navigation in geringer Höhe

**Entsprechend vorhandenen Artikeln:** Berichterstattung über F-Szenen in Papierform, Generierung gefährlicher Szenen, 76 Millionen Erkundungsprotokolle.

**Kernfrage:** Wie definiert man den Testszenenraum des UAV-Hindernisvermeidungs-/Planungsalgorithmus in geringer Höhe, wie misst man die Abdeckung und wie erkennt man effizient gefährliche, aber effektive Ausfallszenarien.

**Methodenroute:**

- Szenariogrammatik: lokale 50 m x 50 m x 50 m große Zelle, Hinderniskombination, dynamische Hindernisse, Windstörung, Zielpunkt, Start- und Endpunkte;
- Abdeckungsmetrik: Geometrieabdeckung, semantische Abdeckung, Dynamikabdeckung, Risikoabdeckung, Fehlermodusabdeckung;
- beschleunigtes Testen: aktive Probenahme von Abdeckungslücken und Ausfallwahrscheinlichkeit;
- Ungültige Filterung: Die Filterung ist unwirklich, unsicher, ungültig und nicht ausführbar;
- planerübergreifende Auswertung: A*/RRT*/MPC/ORCA/MAPPO/CBF-geschirmter Planer.

**Wichtige Referenzen:**

Shuo Fengs NADE und die Generierung von Testszenariobibliotheken sind zentrale Referenzen für beschleunigte Tests und sicherheitskritische Szenariobibliotheken [27-29]; SafeBench bietet eine Benchmark-Plattform und eine Referenz zum Sicherheitsbewertungsprotokoll [30].

**Innovationsvorschläge:**

1. Migration von der Entwicklung autonomer Fahrszenarien zum UAV-3D-Szenenraum in geringer Höhe.
2. Modellieren Sie gleichzeitig die drei Ziele Abdeckung, Kritikalität und Machbarkeit.
3. Nachweis des Abdeckungsraums und der Fehlertaxonomie anhand von 76 Millionen Erkundungsprotokollen.
4. Lassen Sie die Ergebnisse beantworten: Welche Kombinationen von Hindernissen sind am gefährlichsten, welche Planer verallgemeinern die schlimmsten und ob eine erhöhte Abdeckung unbekannte Risiken wirklich reduziert.### 3.5 Papier E: Überprüfung der fehlerkorrigierenden UAV-Sprachplanung

**Vorgeschlagenes Thema:** VERA-UAV: Verifizierungs- und Reparatursprachenplanung für UAV-Aufgaben in geringer Höhe

**Entsprechend vorhandenen Artikeln:** Papier E.

**Kernproblem:** LLM kann Aufgaben in natürlicher Sprache in UAV-ausführbare Aufgabenspezifikationen umwandeln, ist jedoch anfällig für die Erstellung von Plänen, die nicht ausführbar sind, semantische Diskrepanzen aufweisen oder gegen Sicherheitsbeschränkungen verstoßen. Erfordert typisierte IR, LTL/STL, Validatoren und Gegenbeispiel-Feedbackschleifen.

**Methodenroute:**

- NL-Anweisung -> typisiertes TaskIR;
- TaskIR -> LTL/STL;
- Spot-/RTAMT-Verifizierung;
- Gegenbeispiel/Robustheits-Feedback;
- lokale iterative LLM-Reparatur;
- Endgültige Überprüfung der Flugbahn.

**Wichtige Referenzen:**

Lang2LTL, NL2LTL, LTLCodeGen und ConformalNL2LTL unterstützen jeweils NL-zu-LTL-Erdung, Systemdemonstration, zeitliche Logikgenerierung im Code-Generierungsstil und konforme Korrektheitsgarantie [31-34].

**Innovationsvorschläge:**

1. Es handelt sich nicht nur um NL2LTL, sondern die UAV-Flugbahn kann einen geschlossenen Regelkreis ausführen.
2. Typisiertes TaskIR reduziert Sprachmehrdeutigkeiten und verbessert die Interpretierbarkeit.
3. Gegenbeispiel-Feedback und STL-Robustheits-Feedback geben der Reparatur eine bestimmte Richtung.
4. Die AAAI/IJCAI-Version konzentriert sich auf die KI-Planung/-Verifizierung; T-ITS wird erweitert, um eine Verbindung zu Verkehrsbetriebsszenarien in geringer Höhe herzustellen.

### 3.6 Papier G: Low-Altitude Traffic Cloud Brain LLM Agent

**Vorgeschlagenes Thema:** CloudBrain-Agent: Tool-erweiterte LLM-Agenten für den Verkehrsbetrieb in geringer Höhe

**Entsprechend vorhandenen Artikeln:** Papier G/G1.**Kernfrage:** Das Cloud-Gehirn für den Verkehr in geringer Höhe kann nicht nur ein Chat-Modell sein, sondern ein überprüfbarer Agent, der den Planer, Pfadplaner, Verifizierer, Simulator und Risikobewerter anrufen kann.

**Methodenroute:**

- LLM ist verantwortlich für das Aufgabenverständnis, die Werkzeugauswahl, die Statuszusammenfassung und die Interpretation;
- Zu den Tools gehören der Konfliktlöser für Paper A, der Scheduler für Paper B, der aktive Mapper für Paper C, der Szenariotester für Paper D und der Verifier für Paper E;
- LowAltitudeIR als einheitliche Zwischendarstellung;
- Der technische Weg gibt gewöhnlichen großen Modellen + Agent + Fähigkeiten + MCP/Tool-Nutzung Vorrang und wird sich später auf den Bereich LoRA/SFT konzentrieren;
- In der ersten Phase der Bereitstellung wird die API aufgerufen, um einen Benchmark zu bilden, und in der zweiten Phase wird das lokale Qwen/DeepSeek-Modell zur Reproduktion und Kostenkontrolle verwendet.

**Wichtige Referenzen:**

UrbanGPT, UniST und TrafficGPT zeigen, dass Transport-/städtische räumlich-zeitliche Aufgaben begonnen haben, sich den Grundmodellen und Agentenrahmen anzunähern [35-37]; Obwohl es sich bei DriveLM um autonomes Fahren handelt, kann seine Graph-VQA-Aufgabenform aus der mehrstufigen Argumentation des Cloud-Gehirns für den Verkehr in geringer Höhe lernen [38].

**Innovationsvorschläge:**

1. Das Cloud-Gehirn für den Verkehr in geringer Höhe ist kein „vertikales Chat-Modell“, sondern ein überprüfbarer Agent, der durch Tools erweitert wird.
2. Verwenden Sie eine einheitliche IR, um Terminplanung, Planung, Erfassung, Verifizierung und Szenariotests zu verbinden.
3. Führen Sie zuerst den Agenten-Benchmark durch und entscheiden Sie dann, ob das vertikale Modell optimiert werden soll, um das Risiko des ersten Artikels zu verringern.
4. Zu den Bewertungsindikatoren gehören die Genauigkeit des Werkzeugaufrufs, der Aufgabenerfolg, die Sicherheitsverletzung, der Reparaturerfolg, die Latenz und die Überprüfbarkeit durch den Menschen.

### 3.7 Papier H: Urbane ODD in geringer Höhe und semantische Funktionsraumplanung

**Vorgeschlagenes Thema:** ODD2Route: Semantische Operational-Design-Domain-Modellierung für die Routenplanung von UAVs in geringer Höhe

**Dies ist ein neuer Artikel, der in eine neue Richtung geschrieben werden kann. ****Kernfrage:** Wie lässt sich die städtische Gesamtszene auf die lokale Routenplanung in geringer Höhe abbilden? Wie lässt sich das Risiko, die Kapazität und die Servicestrategie von Flugrouten in geringer Höhe auf der Grundlage unterschiedlicher Funktionsbereiche, Gebäudedichte, Straßenstruktur, Menschenansammlungen, Flugverbotszonen und der Verteilung von Notfalleinrichtungen bestimmen?

**Methodenroute:**

- ODD auf Stadtebene: OSM-Proxy für Straße/Gebäude/POI/Landnutzung + Bevölkerung/Nachfrage;
- Lokale Testzelle: Beispiel eines lokalen 3D-Hindernis-/Verkehrsszenarios aus der ODD der Stadt;
- Routenrisikomodell: Bauschluchten, Schulen und Krankenhäuser, Verkehrsknotenpunkte, Autobahnabschnitte, Flugverbotszonen;
- Planungsergebnisse: risikobewusster Korridor, Höhenschicht, Notlandeplatz, Lade-/Vertiport-Kandidaten;
- Auswertung: Stadtübergreifende Generalisierung, Vergleich von naivem kürzestem Weg, risikobewusstem A*, Multi-Ziel-MILP und lernbasierter Routenempfehlung.

**Literaturunterstützung:**

Dieser Artikel kann durch die TR-C/UAM-Literatur von Paper B [13-19], die Szenario-Coverage-Literatur von Paper D [27-30] und die 3D-/digitale Zwillingsliteratur von Paper C [20-26] unterstützt werden. Die Schwierigkeit liegt nicht in der Komplexität des Algorithmus, sondern in der vertrauenswürdigen Definition des ODD auf Stadtebene zum lokalen Szenario-/Routenrisiko.

**Innovationsvorschläge:**

1. Erstellen Sie eine berechenbare Zuordnung zwischen der „gesamten Stadtszene“ und der „lokalen Hinderniskombination“.
2. Verwenden Sie die ODD-Abdeckung, um die Szenenabdeckung zu interpretieren, anstatt zufällig Szenen zu generieren.
3. Bereitstellung einer Brücke zwischen städtischer Tiefgebirgsplanung, Routendesign und Testszenariobibliothek für TR-C/T-ITS.

---

## 4. Empfehlungspriorität| Priorität | Artikel | Letzte Aktionen | Gründe |
|---|---|---|---|
| P0 | Papier B | Problemformulierung, Warteschlangenmodell und experimentellen Benchmark zuerst einfrieren | Am ähnlichsten dem TR-C-Systempapier und am besten geeignet für Wirtschaft/Notfälle in geringer Höhe |
| P0 | Papier A | Schreiben Sie PPO/MAPPO in ein robustes Papier zur Konfliktlösung in geringer Höhe um | Sie verfügen bereits über die Grundlage des Algorithmus, benötigen jedoch Verkehrsindikatoren und eine starke Basislinie |
| P1 | Papier C | Konvergiert zu Fisher + 3DGS + sichere Planung, nicht mehr zu stark erweitern | Der Algorithmus ist sehr innovativ und kann in Robotern/KI/ITS | eingesetzt werden
| P1 | Papier D | Wiederverwendung von 76 Millionen Explorationsprotokollen für abdeckungsgesteuerte Tests | Die Datenbestände sind einzigartig und können leicht einen reproduzierbaren Benchmark bilden |
| P2 | Papier E | Behalten Sie die Papierroute der AAAI/IJCAI-Methode bei | Geeignet für kurze und schnelle Arbeiten, aber Kontrolle des Experimentierumfangs |
| P2 | Papier G | Starten Sie, nachdem die Schnittstelle des Papier-A/B/C/D/E-Werkzeugs stabil ist | CloudBrain-Agent muss sich auf das vorherige Modul verlassen, sonst ist es leer |
| P3 | Papier H | Als nachträgliche Erweiterung von TR-C/T-ITS | Erfordert ausgereifte städtische Datenpipeline und ODD-Definitionen |

---

## 5. Zotero organisiert den Status

Name der Ziel-Zotero-Sammlung:

```text
低空规划论文参考
```

Eine BibTeX-Datei, die in Zotero importiert werden kann, wurde lokal generiert:

```text
zotero/low-altitude-planning-references-20260527.bib
```

In der aktuellen Umgebung gibt es keinen Zotero MCP/Connector, der aufgerufen werden kann, und es gibt keinen Zotero Connector, der installiert werden kann, daher ist es in dieser Runde nicht möglich, direkt in die Zotero-Sammlung zu schreiben. Es wurde bestätigt, dass die ausführbare Zotero-Datei und „~/Zotero/zotero.sqlite“ auf diesem Computer vorhanden sind. Es wird jedoch nicht empfohlen, die Zotero SQLite-Datenbank direkt zu ändern, da dies leicht die Bibliotheksstruktur und den Synchronisierungsstatus beschädigen kann. Der sichere Ansatz ist:1. Erstellen Sie eine neue Sammlung in Zotero: „Low Altitude Planning Paper Reference“.
2. Importieren Sie „zotero/low-altitude-planning-references-20260527.bib“.
3. Wenn Sie später eine Verbindung zur Zotero MCP- oder Better BibTeX-Automatisierungsschnittstelle herstellen, können Sie das Skript so ändern, dass es direkt in die Sammlung schreibt.

---

## 6. Follow-up-Ausführungsplan

### 6.1 Woche 1: Papiermatrix einfrieren

- Bestätigen Sie, ob Papier A/B/C die Hauptkraft der aktuellen drei Artikel ist.
- Bestätigen Sie, ob Papier D die 76 Millionen Explorationsprotokolle als einen Kernwert betrachtet.
- Bestätigen Sie zunächst, ob Papier E/G weiterhin AAAI/IJCAI ist.
- Zotero-Sammlung in BibTeX importieren und PDF hinzufügen.

### 6.2 Woche 2-3: Vervollständigung der Literaturmatrix

- Stellen Sie für jeden Hauptartikel mindestens 25 Dokumente mit hoher Relevanz zusammen.
- Jeder Artikel bildet eine „Matrix der zugehörigen Arbeit“: Problem, Methode, Daten, Metrik, Lücke, unser Blickwinkel.
- Markieren Sie für Arbeit A/B/C die Arbeiten, die „die Grundlinie wiedergeben müssen“ und „nur als verwandte Arbeit dienen“.

### 6.3 Wochen 4–8: Führen Sie zunächst die drei experimentellen Zeilen von Papier B/A/C weiter

- Papier B: synthetischer UAM-Warteschlangen-Benchmark + FCFS/Greedy/MILP/BackPressure/MARL-Basislinie.
- Papier A: Korridorkonfliktsimulation + ORCA/CBF/RMADER/MAPPO-Basislinie.
- Papier C: 3DGS NBV-Pipeline + FisherRF/ActiveNeRF/GS-Planner/POp-GS-Basislinie.

### 6.4 Wochen 9–12: Entscheidung über Ihre erste Einreichung- Wenn die Warteschlangenstabilität und die Hunderte-Regal-Level-Ergebnisse von Papier B am stabilsten sind: Stimmen Sie zuerst für TR-C.
- Wenn Papier A die stärkste Konfliktsicherheit und Verallgemeinerung aufweist: stimmen Sie zuerst für T-ITS/T-RO.
- Wenn Papier C die stärksten theoretischen und visuellen Ergebnisse von Fisher + 3DGS aufweist: stimmen Sie zuerst für T-RO/ICRA/IROS.
- Wenn D über die besten Abdeckungs-/Fehlererkennungsdaten verfügt: Investieren Sie zuerst in T-ITS.

---

## 7. Referenzen

[1] Elsevier. *Transportation Research Teil C: Neue Technologien: Ziele und Umfang.* URL: <https://www.sciencedirect.com/journal/transportation-research-part-c-emerging-technologies>

[2] IEEE Intelligent Transportation Systems Society. *IEEE-Transaktionen auf intelligenten Transportsystemen: Geltungsbereich.* URL: <https://ieee-itss.org/pub/t-its/>

[3] Chao Yu, Akash Velu, Eugene Vinitsky, Yu Wang, Alexandre M. Bayen und Yi Wu. „Die überraschende Wirksamkeit von PPO in kooperativen Multi-Agent-Spielen.“ *Fortschritte in neuronalen Informationsverarbeitungssystemen*, 2022. URL: <https://arxiv.org/abs/2103.01955>[4] Muning Wen, Jakub Grudzien Kuba, Runji Lin, Weinan Zhang, Ying Wen, Jun Wang und Yaodong Yang. „Multi-Agent Reinforcement Learning ist ein Problem der Sequenzmodellierung.“ *NeurIPS*, 2022. URL: <https://proceedings.neurips.cc/paper_files/paper/2022/hash/69413f87e5a34897cd010ca698097d0a-Abstract-Conference.html>

[5] Bei Peng, Tabish Rashid, Christian Schroeder de Witt, Pierre-Alexandre Kamienny, Philip Torr, Wendelin Boehmer und Shimon Whiteson. „FACMAC: Faktorisierte zentralisierte Multi-Agent-Policy-Gradienten.“ *NeurIPS*, 2021. URL: <https://proceedings.neurips.cc/paper/2021/hash/65b9eea6e1cc6bb9f0cd2a47751a186f-Abstract.html>

[6] Jakub Grudzien Kuba, Ruiqing Chen, Muning Wen, Ying Wen, Fudan Sun, Jun Wang und Yaodong Yang. „Optimierung der Trust-Region-Richtlinie beim Multi-Agent-Reinforcement-Learning.“ arXiv:2109.11251, 2021. URL: <https://arxiv.org/abs/2109.11251>[7] Boyu Zhou, Xin Zhou, Jun Zhang, Fei Gao und Shaojie Shen. „EGO-Swarm: Ein vollständig autonomes und dezentrales Quadrotor-Schwarmsystem in überfüllten Umgebungen.“ *ICRA*, 2021. DOI: 10.1109/ICRA48506.2021.9561902. URL: <https://arxiv.org/abs/2011.04183>

[8] Jesus Tordesillas, Brett T. Lopez und Jonathan P. How. „MADER: Trajektorienplaner in Multiagenten- und dynamischen Umgebungen.“ *IEEE Transactions on Robotics*, 38(1):463-476, 2022. URL: <https://arxiv.org/abs/2010.11061>

[9] Kota Kondo, Reinaldo Figueroa, Juan Rached, Jesus Tordesillas, Parker C. Lusk und Jonathan P. How. „Robust MADER: Dezentraler Multiagenten-Trajektorienplaner, robust gegenüber Kommunikationsverzögerungen in dynamischen Umgebungen.“ arXiv:2303.06222, 2023. URL: <https://arxiv.org/abs/2303.06222>[10] Boyu Zhou, Hao Xu und Shaojie Shen. „RACER: Schnelle kollaborative Erkundung mit einem dezentralen Multi-UAV-System.“ *IEEE Transactions on Robotics*, 2023. DOI: 10.1109/TRO.2023.3236945. URL: <https://arxiv.org/abs/2209.08533>

[11] Jesus Tordesillas und Jonathan P. How. „PANTHER: Wahrnehmungsbewusster Trajektorienplaner in dynamischen Umgebungen.“ *IEEE Access*, 10:22662-22677, 2022. DOI: 10.1109/ACCESS.2022.3154037. URL: <https://arxiv.org/abs/2103.06372>

[12] Zhepei Wang, Xin Zhou, Chao Xu und Fei Gao. „Geometrisch eingeschränkte Flugbahnoptimierung für Multikopter.“ *IEEE Transactions on Robotics*, 38(5):3259-3278, 2022. DOI: 10.1109/TRO.2022.3160022. URL: <https://arxiv.org/abs/2103.00190>[13] Ang Li, Mark Hansen und Bo Zou. „Verkehrsmanagement und Ressourcenzuweisung für die UAV-basierte Paketzustellung im städtischen Raum in geringer Höhe.“ *Transportation Research Part C: Emerging Technologies*, 143:103808, 2022. DOI: 10.1016/j.trc.2022.103808. URL: <https://doi.org/10.1016/j.trc.2022.103808>

[14] Mehdi Bennaceur, Rémi Delmas und Youssef Hamadi. „Passagierzentrierte städtische Luftmobilität: Fairness-Kompromisse und betriebliche Effizienz.“ *Transportation Research Part C: Emerging Technologies*, 136:103519, 2022. DOI: 10.1016/j.trc.2021.103519. URL: <https://doi.org/10.1016/j.trc.2021.103519>

[15] Roberto Pinto und Alexandra Lagorio. „Drohnenbasierter Punkt-zu-Punkt-Liefernetzwerkentwurf mit Zwischenladestationen.“ *Transportation Research Part C: Emerging Technologies*, 135:103506, 2022. DOI: 10.1016/j.trc.2021.103506. URL: <https://doi.org/10.1016/j.trc.2021.103506>[16] Qinshuang Wei, Gustav Nilsson und Samuel Coogan. „Kapazitätsbeschränkte städtische Flugmobilitätsplanung.“ arXiv:2107.02900, 2021. URL: <https://arxiv.org/abs/2107.02900>

[17] Surya Murthy, Natasha A. Neogi und Suda Bharadwaj. „Planung für urbane Luftmobilität durch sicheres Lernen.“ arXiv:2209.15457, NASA NTRS, 2022. URL: <https://arxiv.org/abs/2209.15457>

[18] Jiahao Xing, Tong Guo und Lu Tong. „Zuverlässiges Truck-Drohnen-Routing mit dynamischer Synchronisierung: Ein hochdimensionaler Netzwerkprogrammierungsansatz.“ *Transportation Research Part C: Emerging Technologies*, 165:104698, 2024. DOI: 10.1016/j.trc.2024.104698. URL: <https://doi.org/10.1016/j.trc.2024.104698>

[19] Bolong Zhou, Wei Zeng und Hai Yang. „Multi-Trip UAV-UGV Delivery Network Design mit Release-Zeiten.“ *Transportation Research Part C: Emerging Technologies*, 181:105389, 2025. DOI: 10.1016/j.trc.2025.105389. URL: <https://doi.org/10.1016/j.trc.2025.105389>[20] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler und George Drettakis. „3D-Gaußsches Splatting für Echtzeit-Radiance-Field-Rendering.“ *ACM Transactions on Graphics / SIGGRAPH*, 42(4), 2023. DOI: 10.1145/3592433. URL: <https://arxiv.org/abs/2308.04079>

[21] Xuran Pan, Zihang Lai, Shiji Song und Gao Huang. „ActiveNeRF: Mit Unsicherheitsschätzung lernen, wo man sehen kann.“ *ECCV*, 2022. URL: <https://arxiv.org/abs/2209.08546>

[22] Wen Jiang, Boshu Lei und Kostas Daniilidis. „FisherRF: Aktive Ansichtsauswahl und Kartierung mit Strahlungsfeldern unter Verwendung von Fisher-Informationen.“ *ECCV*, 2024. DOI: 10.1007/978-3-031-72624-8_24. URL: <https://eccv.ecva.net/virtual/2024/oral/1226>

[23] Rui Jin, Yuman Gao, Yingjian Wang, Haojian Lu und Fei Gao. „GS-Planner: Ein auf Gauß-Splatting basierendes Planungsrahmenwerk für die aktive High-Fidelity-Rekonstruktion.“ arXiv:2405.10142, 2024. URL: <https://arxiv.org/abs/2405.10142>[24] Zijun Xu, Rui Jin, Ke Wu, Yi Zhao, Zhiwei Zhang, Jieru Zhao, Fei Gao, Zhongxue Gan und Wenchao Ding. „HGS-Planner: Hierarchischer Planungsrahmen für die Rekonstruktion aktiver Szenen mithilfe von 3D-Gauß-Splatting.“ arXiv:2409.17624, 2024. URL: <https://arxiv.org/abs/2409.17624>

[25] Joey Wilson, Marcelino Almeida, Sachit Mahajan, Martin Labrie, Maani Ghaffari, Omid Ghasemalizadeh, Min Sun, Cheng-Hao Kuo und Arnab Sen. „POp-GS: Next Best View in 3D-Gaussian Splatting with P-Optimality.“ *CVPR*, 2025. URL: <https://cvpr.thecvf.com/virtual/2025/poster/34708>

[26] Shangjie Xue, Jesse Dill, Pranay Mathur, Frank Dellaert, Panagiotis Tsiotras und Danfei Xu. „Neuronales Sichtbarkeitsfeld für unsicheres aktives Mapping.“ *CVPR*, 2024. URL: <https://arxiv.org/abs/2406.06948>[27] Shuo Feng, Xintao Yan, Haowei Sun, Yiheng Feng und Henry X. Liu. „Intelligenter Fahrintelligenztest für autonome Fahrzeuge in naturalistischer und kontroverser Umgebung.“ *Nature Communications*, 12:748, 2021. DOI: 10.1038/s41467-021-21007-8. URL: <https://www.nature.com/articles/s41467-021-21007-8>

[28] Shuo Feng, Yiheng Feng, Chao Yu, Yi Zhang und Henry X. Liu. „Testen der Szenariobibliotheksgenerierung für vernetzte und automatisierte Fahrzeuge, Teil I: Methodik.“ *IEEE Transactions on Intelligent Transportation Systems*, 2021. DOI: 10.1109/TITS.2020.2972211. URL: <https://doi.org/10.1109/TITS.2020.2972211>

[29] Shuo Feng, Yiheng Feng, Haowei Sun, Yi Zhang und Henry X. Liu. „Testen der Szenariobibliotheksgenerierung für vernetzte und automatisierte Fahrzeuge, Teil II: Fallstudien.“ *IEEE Transactions on Intelligent Transportation Systems*, 2021. DOI: 10.1109/TITS.2020.2988309. URL: <https://doi.org/10.1109/TITS.2020.2988309>[30] Chejian Xu, Wenhao Ding, Baiming Li, Xinqing Chen, Ding Zhao, Bo Li, Jiajun Liu und Hang Zhao. „SafeBench: Eine Benchmarking-Plattform zur Sicherheitsbewertung autonomer Fahrzeuge.“ *NeurIPS Datasets and Benchmarks*, 2022. URL: <https://proceedings.neurips.cc/paper_files/paper/2022/hash/a48ad12d588c597f4725a8b84af647b5-Abstract-Datasets_and_Benchmarks.html>

[31] Jason Liu, Ankit Shah, Eric Rosen und Stefanie Tellex. „Lang2LTL: Übersetzen natürlicher Sprachbefehle in zeitliche Roboteraufgabenspezifikationen.“ *PMLR/CoRL*, 229, 2023. URL: <https://proceedings.mlr.press/v229/liu23d.html>

[32] Francesco Fuggitti und Tathagata Chakraborti. „NL2LTL: Ein Python-Paket zum Konvertieren natürlichsprachlicher Anweisungen in lineare temporale Logikformeln.“ *AAAI Demonstration*, 37(13):16428-16430, 2023. DOI: 10.1609/aaai.v37i13.27068. URL: <https://ojs.aaai.org/index.php/AAAI/article/view/27068>[33] Behrad Rabiei und Mahesh A. Kumar. „LTLCodeGen: Codegenerierung syntaktisch korrekter zeitlicher Logik für die Roboteraufgabenplanung.“ arXiv:2503.07902, 2025. URL: <https://arxiv.org/abs/2503.07902>

[34] Jun Wang, David Smith Sundarsingh, Jyotirmoy V. Deshmukh und Yiannis Kantaros. „ConformalNL2LTL: Übersetzen von Anweisungen in natürlicher Sprache in temporale Logikformeln mit konformen Korrektheitsgarantien.“ arXiv:2504.21022, 2025. URL: <https://arxiv.org/abs/2504.21022>

[35] Zhonghang Li, Lianghao Xia, Jiabin Tang, Yong Xu, Lei Shi, Long Xia, Dawei Yin und Chao Huang. „UrbanGPT: Räumlich-zeitliche große Sprachmodelle.“ arXiv:2403.00813, 2024. URL: <https://arxiv.org/abs/2403.00813>

[36] Yuan Yuan, Jingtao Ding, Jie Feng, Depeng Jin und Yong Li. „UniST: Ein Prompt-gestütztes Universalmodell für urbane räumlich-zeitliche Vorhersagen.“ *KDD*, 2024. DOI: 10.1145/3637528.3671662. URL: <https://dblp.org/rec/conf/kdd/0032D0J024>[37] Jinhui Ouyang, Yijie Zhu, Xiang Yuan und Di Wu. „TrafficGPT: Auf dem Weg zur mehrskaligen Verkehrsanalyse und -generierung mit dem räumlich-zeitlichen Agenten-Framework.“ arXiv:2405.05985, 2024. URL: <https://arxiv.org/abs/2405.05985>

[38] Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Jens Beisswenger, Ping Luo, Andreas Geiger und Hongyang Li. „DriveLM: Fahren mit grafischer Beantwortung visueller Fragen.“ *ECCV*, 2024. URL: <https://github.com/OpenDriveLab/DriveLM>