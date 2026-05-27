---
title: "Planungspapiermatrix für niedrige Höhen v2: Drei Papiere sind in Arbeit, Folgearbeiten umfassen Modellrouten für niedrige Höhen und große Routen"
description: "Mit drei laufenden Arbeiten zur konfliktfreien Pfadplanung, zur dreischichtigen Planung von Hunderten von UAVs und zur informationstheoretischen 3DGS-Aktiverfassungsplanung als Kernstück werden wir die Route der Folgepapiere zum verkörperten Tiefflug- und Tiefflug-Wolkengehirn, zur Feinabstimmung großer Wirbelmodelle, zur Inferenzbeschleunigung sowie zur Software- und Hardware-Zusammenarbeit neu planen."
pubDate: 2026-05-27
updatedDate: 2026-05-28
tags: ["Planung in geringer Höhe", "UAV", "Abschlussarbeitsplanung", "Zotero", "T-ITS", "TR-C", "T-RO", "AAAI", "ICRA", "3DGS", "MERGEL", "Verkörperte KI", "VLA", "LLM", "Inferenzbeschleunigung"]
category: Tech
sourceHash: "46302954b2010a293b0edfe46fe54f398cd68bf3"
---

# Planungspapiermatrix für niedrige Höhen v2: Drei Papiere sind in Arbeit, Folgearbeiten umfassen Modellrouten für niedrige Höhen und große Routen

> Dieser Artikel fasst die bisher verfassten Arbeiten zu UAVs in geringer Flughöhe wieder in ein **Papierportfolio** zusammen.  
> Das Ziel besteht nicht darin, viele Ideen auszubreiten und zu schreiben, sondern zu klären: Welche Artikel haben bereits Gestalt angenommen, welche können weiterhin zu Top-Zeitschriften-/regulären Konferenzbeiträgen gemacht werden und welche Literaturunterstützung, experimentellen Ressourcen und Einreichungspositionierung werden für jeden Beitrag benötigt.

---

## 0. 28.05.2026 Korrekturfazit

Der aktuelle Fokus muss geändert werden: Anstatt „7-10 Arbeiten gleichzeitig zu planen“, muss zunächst anerkannt werden, dass **bereits an drei Arbeiten gearbeitet wird**, und nachfolgende Arbeiten müssen natürlich aus dem Vermögen dieser drei Arbeiten erwachsen.

Die drei Artikel, an denen ich derzeit arbeite, sind:

| Status | Artikel | Rolle | Die Hauptzeile, von der nicht abgewichen werden kann |
|---|---|---|---|
| Arbeite bereits an | Papier A: Konfliktfreie Pfadplanung / PPO-MAPPO / Konfliktlösung in geringer Höhe | Taktische Sicherheitsschicht | Hochdichte Korridore in geringer Höhe, nicht kooperative UAVs, Verschlechterung der Kommunikation/Positionierung, Kompromisse zwischen Sicherheit und Effizienz |
| Bereits in Bearbeitung | Papier B: Dreischichtige hierarchische Planung von 100 UAVs | Systembetriebsschicht | 100-Ebenen-Flotte, Warteschlangenstabilität, Vertiport-/Lade-/Korridor-Engpass, multimodale Planung |
| Arbeite bereits an | Papier C: Informationstheoriegesteuerte aktive UAV 3DGS-Erkennung | Umweltkognitive Schicht | 3DGS / Fisher-Informationen / NBV / sichere Rekonstruktion / planungsbewusste Kartierung |

Beginnen Sie keine neue Arbeit für nachfolgende Arbeiten, die nichts mit der Richtung zu tun haben. Die richtige Route ist:1. **Machen Sie zunächst A/B/C zu drei Hauptpapieren, die eingereicht werden können. **
2. **Folgepapier D/F/G/H/I werden nur als Erweiterungen von A/B/C verwendet**: Die Szenenabdeckung unterstützt A/C, das Cloud-Brain in geringer Höhe verbindet A/B/C in Reihe, das verkörperte Low-Altitude-Cloud-Gehirn verbindet die Wahrnehmung von C und die Kontrolle von A in einem geschlossenen Regelkreis, und die Feinabstimmung und Inferenz des Modells beschleunigt die Implementierung des Cloud-Gehirns des Dienstes.
3. **Allgemeine AGI-Anweisungen können nicht als leere Ansprüche formuliert werden**. Ein stabilerer Ausdruck ist „in Richtung allgemeiner verkörperter Intelligenz in geringer Höhe“: ausgehend vom Domänenagenten, Werkzeugaufruf, Simulationsfeedback, VLA/VLN, Weltmodell und geräteseitigem Denken, bis hin zur schrittweisen Annäherung an allgemeine verkörperte Intelligenz.
4. **Es wird nicht empfohlen, das vertikale Fundamentmodell in der ersten Phase von Grund auf zu trainieren**. Verwenden Sie zunächst ein gewöhnliches großes Modell + Agent + Skills/MCP + RAG + Verifizierer + Simulator-Nachbearbeitung, um einen reproduzierbaren experimentellen geschlossenen Regelkreis zu bilden. Warten Sie, bis genügend Tool-Call-Trajektorien, Fehlerbeispiele und Simulationsfeedback vorliegen, bevor Sie die LoRA/SFT/DPO/GRPO-Feinabstimmung durchführen.

Dies bedeutet, dass nachfolgende Arbeiten in zwei Ebenen unterteilt werden sollten:

| Ebene | Ziel der Abschlussarbeit | Ob in naher Zukunft damit begonnen werden soll |
|---|---|---|
| A/B/C Hauptschicht | Der experimentelle geschlossene Kreislauf ist bereits im Gange und muss zunächst abgeschlossen werden | Sofort |
| D-Szenen-Overlay | Bietet Benchmark-, Fehlertaxonomie- und sicherheitskritische Daten für Klimaanlagen | Neu |
| G Cloud Brain Agent-Schicht | Verwandeln Sie A/B/C/D/E in Werkzeuge, um ein überprüfbares Cloud-Gehirn für den Tiefflugverkehr aufzubauen | Halbzeit |
| H Verkörperte Tiefschicht | Erstellen Sie ein UAV-VLN-/VLA-/Weltmodell und stellen Sie eine Verbindung zur universellen verkörperten KI her Halbzeit und später |
| Ich modelliere die Trainingsschicht | Training LowAltitudeGPT / Werkzeugnutzung / LowAltitudeIR / Simulationsfeedback | Warten Sie, bis sich die Daten stabilisiert haben |
| J-Inferenzbeschleunigungsschicht | vLLM/TensorRT-LLM/Quantisierung/Geräte-Cloud-Zusammenarbeit/Hardwarebereitstellung | Warten Sie, bis die Arbeitslast des Agenten stabil ist |

---

## 1. Es gibt derzeit Artikel und Hauptzeilenpositionierung

Derzeit gibt es drei Kernartikel, die die inhaltliche Grundlage bilden:| Nummer | Vorhandener Inhalt | Aktuelle Positionierung | Empfohlene Hauptinvestition | Kernurteil |
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

## 3. Paper-Matrix: 3 Papers bereits in Arbeit + Follow-up-Erweiterungsroute

Die Lesart dieses Abschnitts muss geändert werden: Auf Papier A/B/C handelt es sich um die drei Hauptarbeiten, die bereits durchgeführt werden, nicht um „neue Richtungen in der Zukunft“. Papier D/E/G/H/I/J sind beschreibbare Erweiterungen, aber die Startsequenz muss der experimentellen Asset-Reife von A/B/C entsprechen.

### 3.1 Papier A: Robuste, konfliktfreie Planung eines Flugroutennetzes in geringer Höhe**Vorgeschlagenes Thema:** Robuste konfliktfreie UAV-Korridorplanung unter nicht kooperativer Verkehrs- und Kommunikationsverschlechterung

**Entsprechend vorhandenen Artikeln:** Konfliktfreie Pfadplanung, PPO/MAPPO, UAV-Konfliktlösung, UAV-Konfliktumgebungskonstruktion.

**Kernfrage:** Wie können in einem städtischen Luftstraßennetz in geringer Höhe mehrere UAVs die Trennungssicherheit aufrechterhalten und gleichzeitig Verzögerungen, zusätzliche Entfernungen und Durchsatzverluste unter den Bedingungen lokaler Beobachtung, Kommunikationsverzögerungen, Positionierungsfehlern und nicht kooperativem Flugzeugeinsatz kontrollieren?

**Methodenroute:**

- strategische Ebene: anfängliche Pfad- und Zeitfensterzuweisung basierend auf dem Streckennetz;
- taktische Ebene: MAPPO/PPO-Ausgabegeschwindigkeit, Höhe oder seitliche Versatzaktion;
- Sicherheitsschild: Flugbahnkontrolle im CBF-QP-/ORCA-/RMADER-Stil;
- Fallback-Schicht: Wechsel zur konservativen Prioritätsregel, wenn die Kommunikation nachlässt;
- Bewertung: 30/50 Flugzeuge werden trainiert und 100/200 Flugzeuge getestet, wobei vier Szenarien abgedeckt werden: kooperativ, nicht kooperativ, Kommunikationsverlust und Korridor mit hoher Dichte.

**Wichtige Referenzen:**Das stabile Multi-Agent-Training von MAPPO/PPO kann von Yu et al. unterstützt werden. [3]; MAT und FACMAC bieten eine stärkere MARL-Basislinie [4,5]; HAPPO/HATRPO bietet eine Referenz zur Multi-Agent-Richtlinienoptimierung für Vertrauensregionen [6]. Auf der Roboterseite unterstützen EGO-Swarm, MADER, RMADER, RACER, PANTHER und GCOPTER jeweils die dezentrale Schwarmplanung, die gemeinsame Nutzung von Flugbahnen unter Verzögerung, die kollaborative Erkundung, die wahrnehmungsbewusste Planung und die Flugbahnoptimierung von Multikoptern [7-12].

**Innovationsvorschläge:**

1. Upgrade der „PPO-konfliktfreien Pfadplanung“ von einer einfachen RL-Aufgabe auf eine Sicherheitskontrolle für Verkehrskorridore in geringer Höhe.
2. Einführung von Kommunikationsverschlechterungen und nicht kooperativen UAVs, um die eigentliche Betriebsgrenze zu bilden, die T-ITS mehr Sorgen bereitet.
3. Verwenden Sie Lernrichtlinien + formelles/Sicherheitsschild, um die mangelnde Sicherheit von reinem RL zu vermeiden.
4. Trafficisierung von Indikatoren: LoWC, NMAC, Konfliktanzahl, durchschnittliche Verzögerung, zusätzliche Entfernung, Durchsatz, Laufzeit.

### 3.2 Papier B: Dreischichtige hierarchische Planung von Hunderten von UAVs

**Vorgeschlagenes Thema:** H-LyraUAV: Warteschlangenstabile hierarchische Planung für die UAV-Logistik in geringer Höhe im Hundertmaßstab

**Entsprechend vorhandenen Artikeln:** Papier B dreistufige Terminplanung.

**Kernfrage:** Wie kann eine UAV-Flotte mit hundert Ebenen unter dynamischen Anforderungen, begrenzter Vertiport-/Lade-/Korridorkapazität und multimodalen Transportbeschränkungen stabil, effizient und sicher arbeiten?

**Methodenroute:**- Makroebene: Nachfragewarteschlange, Neupositionierung der Flotte, Moduswahl;
- Mesoschicht: Vertiport, Ladepad, Korridor-Slot-Planung;
- Mikroebene: Machbarkeit der Energie-/Sicherheits-/konfliktbewussten Entwicklung;
- Theorie: Lyapunov-Drift plus Strafe garantiert Warteschlangenstabilität und Kosten-Rückstands-Kompromiss;
- Daten: synthetisches Stadtgitter + OSM/POI/NYC TLC/Chicago Taxi/SUMO-Erweiterung.

**Wichtige Referenzen:**

Das TR-C-Drohnen-Lieferverkehrsmanagement in geringer Höhe hat die Ressourcenzuweisung und Konfliktlösung im städtischen Raum in geringer Höhe direkt erörtert [13]; passagierzentriertes UAM, Fairness und betriebliche Effizienzforschung unterstützen die Gestaltung der Servicequalität [14]; Ladestations-Liefernetzwerk, kapazitätsbeschränkte UAM-Planung, sichere Lernplanung, unterstützende Infrastrukturkapazität und sichere Online-Planung [15-17]; LKW-Drohne / UAV-UGV multimodale Zustellung unterstützt multimodale Erweiterung [18,19].

**Innovationsvorschläge:**1. Online-Scheduling mit drei Schichten auf Hunderten von Ebenen anstelle von Offline-Routing/Netzwerkdesign.
2. Die Warteschlangenstabilität wird zum Hauptthema der Theorie, und das Lernmodul macht nur Vorhersagen oder Wertschätzungen.
3. Bewerten Sie gleichzeitig Verzögerung, Durchsatz, Rückstand, Ladeauslastung, Vertiport-Engpass und Korridorüberlastung.
4. Die Schlussfolgerung des Verkehrssystems kann antworten: Wann muss der Verkehr begrenzt werden, wo liegt der Engpass und wann ist der Einsatz von UAVs dem multimodalen Fallback unterlegen?

### 3.3 Papier C: FIM-3DGS UAV-Aktiverfassungsplanung

**Vorgeschlagenes Thema:** FIM-3DGS: Fisher-Information-gesteuerte aktive Wahrnehmungsplanung für eine sichere UAV-Rekonstruktion

**Entspricht bestehenden Artikeln:** Paper C, Next-Best-View und NeRF/3DGS, Information Theory Active Sensing.

**Kernfrage:** Wie können UAVs unter begrenzten Flugzeit-, Energie- und Sicherheitsbeschränkungen aktiv Aussichtspunkte auswählen, um die Konvergenz der 3DGS-Karte zu beschleunigen und Planungsaufgaben in geringer Höhe zu erfüllen?

**Methodenroute:**

- Szenendarstellung: inkrementelles 3D-Gaußsches Splatting;
- Informationsmetrik: Fisher-Information erstellen / erwarteter Informationsgewinn für Gaußsche Parameter oder gerenderte Jacobi-Parameter;
- Planer: Generierung von NBV-Kandidaten + sicherer Korridor / CBF-Beschränkung;
- Aufgabenkopplung: Die Rekonstruktionsqualität wird nicht nur über PSNR/SSIM berichtet, sondern auch über Hindernisabruf, Planungskollisionsrate und Inspektionsabdeckung;
- Basislinien: ActiveNeRF, FisherRF, GS-Planner, HGS-Planner, POp-GS, Grenzerkundung.

**Wichtige Referenzen:**Der ursprüngliche 3DGS-Text bietet eine explizite Darstellung des Strahlungsfeldes in Echtzeit [20]; ActiveNeRF ist ein früher Vertreter der neuronalen Wiedergabe aktiver Wahrnehmung [21]; FisherRF unterstützt direkt die aktive Ansichtsauswahl von Fisher-Informationen und liefert 3DGS-Backend-Ergebnisse mit 70 fps [22]; GS-Planner, HGS-Planner, POp-GS und NVF unterstützen die 3DGS/NBV-Wettbewerbslinie von 2024–2025 [23–26].

**Innovationsvorschläge:**

1. Upgrade von „3DGS NBV“ auf „aktive Wahrnehmung im Dienste der UAV-Sicherheitsplanung“.
2. Verwenden Sie Fisher-Informationen, um CRB/Rekonstruktionsunsicherheit/Planungssicherheit miteinander zu verbinden.
3. Erweitern Sie von visuellen Indikatoren zu Verkehrs-/Roboteraufgabenindikatoren: Pfaddurchführbarkeitsrate, Hinderniserkennungsrate, Abdeckungsrate für Notfallinspektionen.
4. Führen Sie eine szenarioübergreifende Verallgemeinerung auf MatrixCity/AirSim/selbstgebauten städtischen Zellen in geringer Höhe durch.

### 3.4 Papier D: Abdeckung sicherheitskritischer Szenen in geringer Höhe und beschleunigte Tests

**Vorgeschlagenes Thema:** Abdeckungsgesteuerte beschleunigte Tests für sicherheitskritische UAV-Navigation in geringer Höhe

**Entsprechend vorhandenen Artikeln:** Berichterstattung über F-Szenen in Papierform, Generierung gefährlicher Szenen, 76 Millionen Erkundungsprotokolle.

**Kernfrage:** Wie definiert man den Testszenenraum des UAV-Hindernisvermeidungs-/Planungsalgorithmus in geringer Höhe, wie misst man die Abdeckung und wie erkennt man effizient gefährliche, aber effektive Ausfallszenarien.

**Methodenroute:**- Szenariogrammatik: lokale 50 m x 50 m x 50 m große Zelle, Hinderniskombination, dynamische Hindernisse, Windstörung, Zielpunkt, Start- und Endpunkte;
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
4. Lassen Sie die Ergebnisse beantworten: Welche Kombinationen von Hindernissen sind am gefährlichsten, welche Planer verallgemeinern die schlimmsten und ob eine erhöhte Abdeckung unbekannte Risiken wirklich reduziert.

### 3.5 Papier E: Überprüfung der fehlerkorrigierenden UAV-Sprachplanung

**Vorgeschlagenes Thema:** VERA-UAV: Verifizierungs- und Reparatursprachenplanung für UAV-Aufgaben in geringer Höhe

**Entsprechend vorhandenen Artikeln:** Papier E.

**Kernproblem:** LLM kann Aufgaben in natürlicher Sprache in UAV-ausführbare Aufgabenspezifikationen umwandeln, ist jedoch anfällig für die Erstellung von Plänen, die nicht ausführbar sind, semantische Diskrepanzen aufweisen oder gegen Sicherheitsbeschränkungen verstoßen. Erfordert typisierte IR, LTL/STL, Validatoren und Gegenbeispiel-Feedbackschleifen.**Methodenroute:**

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

**Entsprechend vorhandenen Artikeln:** Papier G/G1.

**Kernfrage:** Das Cloud-Gehirn für den Verkehr in geringer Höhe kann nicht nur ein Chat-Modell sein, sondern ein überprüfbarer Agent, der den Planer, Pfadplaner, Verifizierer, Simulator und Risikobewerter anrufen kann.

**Methodenroute:**- LLM ist verantwortlich für das Aufgabenverständnis, die Werkzeugauswahl, die Statuszusammenfassung und die Interpretation;
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

**Dies ist ein neuer Artikel, der in eine neue Richtung geschrieben werden kann. **

**Kernfrage:** Wie lässt sich die städtische Gesamtszene auf die lokale Routenplanung in geringer Höhe abbilden? Wie lässt sich das Risiko, die Kapazität und die Servicestrategie von Flugrouten in geringer Höhe auf der Grundlage unterschiedlicher Funktionsbereiche, Gebäudedichte, Straßenstruktur, Menschenansammlungen, Flugverbotszonen und der Verteilung von Notfalleinrichtungen bestimmen?**Methodenroute:**

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

### 3.8 Papier I: Verkörperte Intelligenz in geringer Höhe und Luft-VLA/VLN

**Vorgeschlagenes Thema:** Verkörperte Intelligenz in geringer Höhe: Vision-Sprache-Aktionsplanung für UAVs im städtischen Luftraum

**Dies ist die mittel- bis langfristige Richtung, die nach der neuen Umfrage am meisten beibehalten werden sollte. **Die derzeitige Hauptlinie der verkörperten Intelligenz hat sich von „LLM-Sprechen“ zu „VLM/VLA, das Wahrnehmung, Sprache und Handlung direkt verbindet“ entwickelt. RT-2 schlug eindeutig das Vision-Sprache-Aktion-Modell vor und brachte Vision, Sprache und Roboteraktion in dasselbe Modellparadigma [44]; OpenVLA und Octo zeigten, dass die Open-Source-VLA-/generalistische Roboterrichtlinie mit großen Robotertrajektorien vorab trainiert und dann mit einer kleinen Menge an Zieldomänendaten verfeinert werden kann [42,43]. Es gibt auch direkt verwandte Arbeiten in Richtung UAV: ​​SINGER verwendet Gaußsches Splatting, um Flugsimulationsdaten zur Spracheinbettung zu generieren, die VLN-Richtlinien für Borddrohnen zu trainieren und Hardwareexperimente durchzuführen [39]; FlightGPT verwendet SFT + GRPO, um UAV VLN durchzuführen, und überprüft die Verallgemeinerung und interpretierbare Argumentation auf CityNav [40]; UAV-VLN verbindet natürliche Sprache, visuelle Wahrnehmung und realisierbare Flugbahnplanung [41].

**Unsere beschreibbare Lücke:**

Die meisten bestehenden VLN/VLA aus der Luft konzentrieren sich darauf, „ein Sprachziel anzugeben und die Drohne in der Nähe des Ziels fliegen zu lassen“. Dies ist nicht die Fähigkeit, die das Gehirn der Verkehrswolke in geringer Höhe benötigt. Szenarien in geringer Höhe erfordern, dass das Modell gleichzeitig Folgendes versteht:

- Städtische ODD in geringer Höhe, Luftraumstruktur, Flugverbotszonen und Risikogebiete;
- Multi-UAV-Verkehrsstatus, Korridorkapazität und Kollisionsvermeidungsregeln;
- Priorität von Notfallaufgaben, Inspektionsaufgaben und Logistikaufgaben;
- Unvollständige visuelle Karten, Positionierungsfehler, Kommunikationsverschlechterung und nicht kooperative Ziele;
- Die Ausgabe muss überprüfbar, kontrollierbar und abbaubar sein und darf kein durchgängiger Black-Box-Vorgang sein.

**Vorgeschlagene Methode:**

```text
multimodal observation
  = UAV RGB/depth/semantic map/3DGS local map
  + low-altitude traffic state
  + natural-language mission
  + city ODD metadata

LLM/VLM/VLA policy
  -> LowAltitudeIR
  -> skill selection
  -> waypoint / velocity / route command
  -> verifier + safety shield
  -> simulator or hardware feedback
```

**Empfohlene Version zuerst:**

Trainieren Sie zunächst nicht End-to-End-AerialVLA. Erstellen Sie zunächst einen **hybriden verkörperten Agenten**:

- Führungskräfte auf hoher Ebene nutzen das Qwen/DeepSeek/API-Modell zum Verständnis von Aufgaben und zum Aufrufen von Tools;
- Die mittlere Schicht ruft den Konfliktlöser von Papier A, den Planer von Papier B, den aktiven Mapper von Papier C und den Verifizierer von Papier E auf;
- Die untere Schicht verwendet einen herkömmlichen Controller/MPC/CBF-Schutz, um Echtzeitsicherheit zu gewährleisten;
- Trainingsdaten stammen aus Simulationsverläufen, Expertenplanern, Fehlerreparaturprotokollen und manuellen Anmerkungsaufgaben.

**Verfügbare Ziele:**- ICRA/IROS/T-RO: Schwerpunkt auf integrierter Navigation, Hardware-Closed-Loop und Sim-to-Real.
- AAAI/IJCAI: Betonen Sie Agentenplanung, Tool-Nutzung und Verifizierungs-Feedback.
- T-ITS: Schwerpunkt auf Verkehrsbetrieb in geringer Höhe, Notfallmaßnahmen, Konfliktlösung und Systemindikatoren.

### 3.9 Paper J: LowAltitudeGPT-Training und Feinabstimmung der Route

**Vorgeschlagenes Thema:** LowAltitudeGPT: Tool-Nutzung und Simulations-Feedback-Tuning für Verkehrsinformationen in geringer Höhe

**Kernurteil:**

Es ist jetzt nicht an der Zeit, das „große Tieflandverkehrsmodell“ von Grund auf zu trainieren. Dabei gibt es drei Probleme:

1. Die Datenmenge reicht nicht aus, um Beiträge auf Stiftungsmodellebene zu unterstützen.
2. Bei der Überprüfung wird gefragt, ob der Modellbeitrag den von gewöhnlichen großen Modellen + RAG + Werkzeugnutzung übersteigt;
3. Die Schulungskosten sind hoch, aber sie sind nicht unbedingt wertvoller als der geschlossene Regelkreis von Agent/Prüfer/Simulator.

Ein praktikablerer Weg ist **normales großes Modell + Agent + Skills/MCP + RAG + Verifizierer + Simulator**, um es zuerst zu durchlaufen und dann die laufenden Protokolle in trainierbare Daten umzuwandeln. MCP ist im Wesentlichen eine Standardschnittstelle, die Tools und Kontext für LLM verfügbar macht und sich für den einheitlichen Zugriff auf Scheduler, Planer, Prüfer, Simulatoren, Datenbanken und Dokumentbibliotheken eignet [47].

Eine Überprüfung großer Wirtschaftsmodelle in geringer Höhe unterteilt auch Systeme in geringer Höhe in Anlagennetzwerke, Informationsnetzwerke, Routennetzwerke und Servicenetzwerke und betont, dass große Modelle mit Edge Computing, 6G/ISAC und vertrauenswürdiger verteilter Intelligenz kombiniert werden müssen [50]. Dies zeigt, dass unser Artikel nicht nur als „Training eines Chat-Modells“ geschrieben werden kann, sondern als geschlossener Kreislauf von Modellen, Tools, Netzwerken, Betriebssteuerung und Systembewertung geschrieben werden muss.

**Vorschläge zur Modellauswahl:**| Bühne | Empfohlenes Modell | Grund |
|---|---|---|
| Lösungserkundung / Datengenerierung / Lehrer | Hochleistungsfähiges API-Modell | Generieren Sie zunächst schnell Aufgaben, Werkzeugspuren, Gegenbeispielerklärungen und Bewertungsbeispiele, ohne die API als endgültige reproduzierbare Abhängigkeit zu verwenden |
| Lokal reproduzierbare Experimente | Qwen3-8B / Qwen3-14B / Qwen3-32B | Qwen3 unterstützt offiziell lokale Betriebs-, Bereitstellungs-, Quantifizierungs- und Schulungsprozesse mit guter chinesischer Sprache, Werkzeugaufruf und technischer Ökologie [45] |
| Argumentation/Mathematik/Constraint-Interpretation | DeepSeek-R1-Distill-Qwen-14B / 32B | Die DeepSeek-R1-Serie legt den Schwerpunkt auf RL-motivierte Denkfähigkeiten. Die Destillationsversion kann lokal bereitgestellt werden und basiert auf dem Qwen/Llama-Open-Source-Modell [46] |
| Multimodale Wahrnehmung in geringer Höhe | Qwen-VL / Qwen3-VL / andere Open-Source-VLM | Semantisches Verständnis für Bilder, Videobilder, Karten, Streckendiagramme und 3DGS-Rendering |
| Kantenende kleines Modell | Qwen3-4B / 8B quantitative Version, SLM | Wird für die Statuszusammenfassung auf der Endseite, die Erkennung von Anomalien und den Fallback mit geringer Latenz verwendet |

**Trainingsdatendesign:**

| Datentyp | Quelle | Trainingsziel |
|---|---|---|
| NL-Mission -> LowAltitudeIR | Manuelle Vorlage + API-Lehrer + Umschreiben echter Aufgaben | Aufgabenanalyse und strukturierte Darstellung |
| Werkzeugnutzungsspur | Papier-A/B/C/D/E-Tool-Anrufprotokoll | Erfahren Sie, wann Sie Terminplanung, Planung, Überprüfung und Simulation anrufen müssen |
| Verifizierer-Gegenbeispiel | Spot/RTAMT/CBF/Simulator-Feedback | Lernen Sie, nicht ausführbare oder gefährliche Pläne zu reparieren |
| Simulations-Rollout | SUMO/AirSim/selbstentwickelte Tiefflugsimulation | Lernen Sie, Systemengpässe aus Ergebnissen zu erklären |
| Fehlerfall | Kollision, LoWC, Timeout, Warteschlangenexplosion, unzureichender Energieverbrauch | Erfahren Sie mehr über Risikodiagnose und Deeskalation im Notfall |
| menschliche Prüfdaten | Manuelle Auswahl sinnvollerer Lösungen | DPO/Präferenzoptimierung |

**Trainingsphase:**1. **RAG + Prompt Baseline**: Keine Feinabstimmung, nur Literaturbibliothek, Vorschriften, Systembeschreibung und Toolschema verwenden.
2. **LoRA/QLoRA SFT**: Training von NL-zu-IR, Tool-Call, Ergebnisinterpretation und Gegenbeispielreparatur.
3. **DPO/IPO**: Verwenden Sie manuelle Einstellungen oder Verifier-Scoring-Einstellungen, um „sicher, ausführbar, prägnant und erklärbar“ zu optimieren.
4. **Optimierung im GRPO/RL-Stil**: Verwenden Sie Simulation, um die Erfolgsquote von Trainingsaufgaben, geringe Verstöße, geringe Latenz und Formatkonformität zu belohnen. Die SFT + GRPO-Route von FlightGPT kann als UAV-VLN-Referenz verwendet werden [40].
5. **Destillation**: Destillieren Sie die API-Lehrer-/32B-Modellfunktionen auf 8B/4B für die lokale und Edge-Bereitstellung.

**Bewertungsindikatoren:**

-Aufgabenerfolg;
- Genaue/semantische Übereinstimmung mit LowAltitudeIR;
- Werkzeugaufrufgenauigkeit/-rückruf;
- ausführbarer Planpreis;
- Rate von Sicherheitsverstößen;
- Reparaturerfolgsquote;
- Halluzinationsrate;
- Latenz/Token-Kosten;
- stadt-/aufgabenübergreifende Verallgemeinerung;
- Erfolgsquote bei menschlichen Audits.

### 3.10 Paper K: Cloud-Gehirn-Inferenzbeschleunigung in geringer Höhe und Software- und Hardware-Zusammenarbeit

**Vorgeschlagenes Thema:** Edge-Cloud-Co-optimierte Inferenz für Cloud-Brain-Agenten für Verkehr in geringer Höhe

**Warum kann dieser Artikel geschrieben werden:**Wenn wir in Zukunft sowohl Software als auch Hardware nutzen wollen, kann die Inferenzbeschleunigung nicht nur eine technische Optimierung sein. Es muss als **Problem intelligenter Echtzeitdienste unter Einschränkungen des Verkehrssystems in geringer Höhe** geschrieben werden: Auf der Cloud-Seite gibt es große Modelle und globale Zustände, auf der Edge-Seite niedrige Latenz- und Datenschutz-/Kommunikationsbeschränkungen und auf der Drohnenseite Einschränkungen hinsichtlich Stromverbrauch, Rechenleistung, Wärmeableitung und Echtzeitsteuerung. General-Purpose Aerial Intelligent Agents haben ein direktes Signal in Richtung Hardware-Software-Codesign gegeben: Das 14B-Modell an Bord läuft etwa 5–6 Token/Sekunde, hat einen Spitzenstromverbrauch von etwa 220 W und übernimmt eine bidirektionale kognitive Architektur aus langsamer LLM-Planung und schneller Reaktionskontrolle [51].

**Systemarchitektur:**

```text
cloud brain
  - full LLM / VLM
  - global scheduler
  - long-horizon planner
  - batch simulation evaluator

edge station / vertiport
  - quantized 8B/14B model
  - local RAG cache
  - route/conflict verifier
  - streaming state summarizer

onboard UAV
  - tiny policy / controller
  - VIO / obstacle avoidance
  - emergency fallback
  - compressed semantic state uplink
```

**Beschleunigter Technologieweg:**

- Server: vLLM/PagedAttention/Continuous Batching/Präfix-Cache. Der Kernwert von PagedAttention besteht darin, die KV-Cache-Verschwendung zu reduzieren und den Batch-Serving-Durchsatz zu verbessern [48].
- NVIDIA GPU-Produktionsbereitstellung: TensorRT-LLM, Durchführung von LLM-Inferenz mit TensorRT-Engines, Python/C++-Laufzeit und GPU-Optimierung [49].
- Ende/Edge: AWQ/GPTQ/GGUF INT4/INT8, KV-Cache-Komprimierung, spekulative Dekodierung, kleiner Modell-Router.
- Tool-Call-Optimierung: Tool-Schema zwischenspeichern, statische RAG-Suchergebnisse zwischenspeichern und hochfrequente Tool-Calls in deterministische Fähigkeiten kompilieren.
- Bedienerrichtung: Aufmerksamkeitskernel, ausgelagerter KV-Cache, Vorfüllungs-/Dekodierungstrennung, Stapelplaner, MoE-Expertenrouting, Vision-Encoder-Caching.

**Abschlussarbeitspunkte verfügbar:**1. **Systempapier**: Latenz-/Kosten-/Energieprofilierung der Arbeitslast von Cloud-Brain-Agenten in geringer Höhe.
2. **Algorithmus-System-Papier**: Dynamische Auswahl von API/Cloud 32B/Edge 14B/Onboard 4B basierend auf dem Aufgabenrisiko.
3. **Operator-/Inferenzpapier**: KV-Cache- und Batch-Optimierung für Verkehr in geringer Höhe mit mehreren Agenten, mehreren Tools, langem Kontext und Streaming-Statusaktualisierungen.
4. **Papier zur Hardware-Zusammenarbeit**: Dreistufige Bereitstellung von Jetson Orin / RTX-Workstation / Cloud-GPU, Bewertung von Tokens/Sek., End-to-End-Latenz, Energie pro Entscheidung und Sicherheits-Fallback-Rate.

**Empfohlener Veranstaltungsort:**

– Teiltransportsysteme: T-ITS / IEEE IoT Journal.
- Teilweise Edge Intelligence: IEEE TMC / IEEE Internet of Things Journal / ACM TECS.
- Teilrobotersystem: IROS/ICRA-Systempapier.
- Teiloperatoren und -systeme: Ausgehend vom MLSys / SC-Workshop / DAC/DATE-Workshop wird nicht empfohlen, zu Beginn direkt zur Top-Systemkonferenz zu gehen.

---

## 4. Empfehlungspriorität| Priorität | Artikel | Letzte Aktionen | Gründe |
|---|---|---|---|
| P0-Aktiv | Papier B | Problemformulierung einfrieren, Warteschlangenmodell, experimenteller Benchmark | Am ähnlichsten dem TR-C-Systempapier und am besten geeignet für Wirtschaft/Notfälle in geringer Höhe |
| P0-Aktiv | Papier A | Umschreiben von PPO/MAPPO in ein robustes Papier zur Konfliktlösung in geringer Höhe | Verfügt bereits über eine Algorithmusbasis, benötigt jedoch Verkehrsindikatoren und eine starke Basislinie |
| P0-Aktiv | Papier C | Konvergiert zu Fisher + 3DGS + sichere Planung, nicht mehr zu stark erweitern | Der Algorithmus ist innovativ und kann in Robotern/KI/ITS | eingesetzt werden
| P1-Unterstützung | Papier D | 76 Millionen Explorationsprotokolle wiederverwenden und abdeckungsgesteuerte Tests durchführen | Bereitstellung sicherheitskritischer Szenarien, Fehlertaxonomie und Benchmark für Klimaanlagen |
| P1-Brücke | Papier G | Erstellen Sie zunächst die Tool-Schnittstelle und den CloudBrain-Agent-Benchmark | String A/B/C/D/E in ein Wolkenhirn in geringer Höhe statt in ein leeres Chat-Modell |
| P2-verkörpert | Papier I | Durchführung eines kleinen VLN/VLA-Piloten aus der Luft: Simulationsdaten, Expertenflugbahnen, End-to-End-/Hybrid-Strategievergleich | Dies ist die Hauptlinie, die zur verkörperten AGI führt, erfordert jedoch zunächst die Stabilisierung der A/C-Wahrnehmung und der Sicherheitsinstrumente |
| P2-Modell | Papier J | Präzipitieren Sie LowAltitudeIR, Werkzeugverfolgung, Verifizierer-Feedback und führen Sie dann LoRA/SFT/GRPO aus | Führen Sie zunächst einen geschlossenen Datenkreislauf durch und optimieren Sie dann das vertikale Modell |
| P3-System | Papier K | Warten Sie, bis die CloudBrain-Agent-Arbeitslast behoben ist, bevor Sie vLLM/TensorRT/Quantization/End-Cloud-Zusammenarbeit | durchführen Die Software- und Hardwarerichtung kann geschrieben werden, aber es erfordert einen echten Arbeitsaufwand, um wie ein Papier zu sein |
| P3-Planung | Papier H | Als nachträgliche Erweiterung von TR-C/T-ITS | Erfordert eine ausgereifte städtische Datenpipeline und ODD-Definition |

**Vorschläge zur Ausführungsreihenfolge:**1. Das aktuelle Hauptschlachtfeld nicht ändern: A/B/C rücken weiter vor.
2. Ergänzen Sie zuerst Papier D, da es die experimentelle Glaubwürdigkeit von A/C direkt erhöht und auch nachfolgende Modelltrainingsdaten generieren kann.
3. Erstellen Sie erneut Papier G und packen Sie A/B/C/D/E in ein werkzeugbasiertes Cloud-Gehirn.
4. Papier I/J/K Beeilen Sie sich nicht, ein großes Projekt zu starten; Erstellen Sie zunächst ein kleines Pilot- und Datenschema. Bevor Sie mit der eigentlichen Frage beginnen, müssen Sie antworten: Woher kommen die Daten, was sind die Bewertungsindikatoren und ob sie stärker sein können als gewöhnliche große Modelle + Toolaufrufe.

---

## 4.1 Literaturunterstützungsmatrix

Um eine Dokumentenstapelung zu vermeiden, werden die derzeit 51 Referenzen entsprechend der Richtung des Papiers geschlossen verwendet:| Wegbeschreibung | Dokumentationsgruppen | Verwendung |
|---|---|---|
| Positionierung von Einreichungs- und Transportsystemen | [1,2] | Bestimmen Sie die Rahmenunterschiede von TR-C / T-ITS |
| Papier A: Multi-Agenten-Konfliktlösung | [3-12] | PPO/MAPPO, MAT/FACMAC/HAPPO und EGO-Swarm/MADER/RMADER/RACER/PANTHER/GCOPTER Basislinie |
| Papier B: Hunderte von UAV-Planungen | [13-19] | Zuweisung von Ressourcen für die Lieferung in geringer Höhe, UAM-Planung, sicheres Lernen, multimodale Lieferung per Lkw-Drohne/UAV-UGV |
| Papier C: Aktive 3DGS-Sensorik | [20-26] | 3DGS, ActiveNeRF, FisherRF, GS-Planner, HGS-Planner, POp-GS, NVF |
| Papier D: Abdeckung sicherheitskritischer Szenarien | [27-30] | Shuo Feng beschleunigte Tests, Szenariobibliothek, SafeBench |
| Papier E: Sprachplanung und -überprüfung | [31-34] | Lang2LTL, NL2LTL, LTLCodeGen, ConformalNL2LTL |
| Papier G: Cloud Brain Agent in geringer Höhe | [35-38,47,50,51] | UrbanGPT/UniST/TrafficGPT/DriveLM, MCP, Überprüfung großer wirtschaftlicher Modelle in geringer Höhe, intelligenter Luftagent |
| Papier I: Verkörpertes Low Altitude / Aerial VLA | [39-44] | SINGER, FlightGPT, UAV-VLN, OpenVLA, Octo, RT-2 |
| Papier J: Modellschulung und Feinabstimmung | [40,45,46,47,50] | SFT/GRPO-Referenz, Qwen3, DeepSeek-R1, MCP/Werkzeugnutzung, Positionierung großer Modellsysteme in geringer Höhe |
| Paper K: Inferenzbeschleunigung und Software- und Hardware-Zusammenarbeit | [45,48,49,51] | Qwen3-Bereitstellungsökologie, vLLM/PagedAttention, TensorRT-LLM, integrierte 14B-Aerial-Agent-Hardwareeinschränkungen|---

## 5. Zotero organisiert den Status

Name der Ziel-Zotero-Sammlung:

```text
低空规划论文参考
```

Derzeit sind zwei Organisationsebenen abgeschlossen:

| Projekt | Status |
|---|---|
| Zotero-Sammlung | existiert bereits, Sammlungsschlüssel ist „FVHS3SKY“, lokale TreeViewID ist „C17“ |
| Link zur lokalen Zotero-Auswahl | `zotero://select/library/collections/FVHS3SKY` |
| Importierte Dokumente | 51 Top-Level-Artikel |
| Artikeltypverteilung | „journalArticle“ 17 Elemente, „conferencePaper“ 11 Elemente, „document/preprint/webpage“ 23 Elemente |
| Lokale Sicherung von BibTeX | `zotero/low-altitude-planung-references-20260527.bib`; Inkrement: `zotero/low-altitude-planung-references-update-20260528.bib` |

Die Importmethode verwendet den lokalen Connector-Server von Zotero, anstatt „zotero.sqlite“ direkt zu schreiben. Der spezifische Prozess ist:

1. Verwenden Sie „pandoc“, um zu überprüfen, ob BibTeX als CSL JSON geparst werden kann.
2. Importieren Sie „zotero/low-altitude-planning-references-20260527.bib“ über Zotero local „/connector/import“.
3. Aktualisieren Sie die Zielsammlung der importierten Sitzung über „/connector/updateSession“ auf „C17 / Low Altitude Planning Paper Reference“.
4. Verwenden Sie die lokale Zotero-API und schreibgeschütztes SQLite, um doppelt zu überprüfen, ob die Sammlung 51 Dokumente der obersten Ebene enthält.Wenn Sie in Zukunft weiterhin Dokumente hinzufügen, wird empfohlen, zuerst das lokale BibTeX zu aktualisieren und dann Zotero über denselben Connector-Import/UpdateSession-Prozess zu importieren. Ändern Sie SQLite nicht direkt.

---

## 6. Follow-up-Ausführungsplan

### 6.1 Woche 1: Drei laufende Arbeiten einfrieren

- Es ist klar, dass Paper A/B/C die derzeit aktive Pipeline ist und nachfolgende Papers nicht mehr mit derselben Priorität geschrieben werden.
- Papier A: Konfliktszenarien, Aktionsraum, Baseline und Verkehrsindikatoren einfrieren.
- Papier B: Frozen-Queue-Modell, Lyapunov-Ziel, synthetischer Benchmark und TR-C-Framing.
- Papier C: Theoretische Schnittstellen und planungsbewusste Metriken für FIM/3DGS/NBV einfrieren.
– Der Erstimport der Zotero-Sammlung und der inkrementelle Import am 28.05.2026 wurden abgeschlossen; Der nächste Schritt besteht darin, für jeden Artikel PDF-Dateien, zusammenfassende Notizen und Prioritäts-Tags hinzuzufügen.

### 6.2 Wochen 2-3: Ergänzung der Literaturmatrix und anschließende Überprüfung der Routenneuheit

- Stellen Sie für jeden Hauptartikel mindestens 25 Dokumente mit hoher Relevanz zusammen.
- Jeder Artikel bildet eine „Matrix der zugehörigen Arbeit“: Problem, Methode, Daten, Metrik, Lücke, unser Blickwinkel.
- Markieren Sie für Arbeit A/B/C die Arbeiten, die „die Grundlinie wiedergeben müssen“ und „nur als verwandte Arbeit dienen“.
- Führen Sie die Neuheitsprüfung für Papier I/J/K separat durch:
  - Papier I: Aerial VLN, AerialVLA, SINGER, FlightGPT, OpenVLA, Octo, RT-2;
  - Paper J: Qwen3, DeepSeek-R1, Tool-Use-Tuning, MCP, RAG, Simulations-Feedback-Training;
  - Paper K: vLLM, TensorRT-LLM, Quantifizierung, KV-Cache, Edge-Cloud-Bereitstellung.

### 6.3 Wochen 4–8: Führen Sie zunächst die drei experimentellen Zeilen von Papier B/A/C weiter- Papier B: synthetischer UAM-Warteschlangen-Benchmark + FCFS/Greedy/MILP/BackPressure/MARL-Basislinie.
- Papier A: Korridorkonfliktsimulation + ORCA/CBF/RMADER/MAPPO-Basislinie.
- Papier C: 3DGS NBV-Pipeline + FisherRF/ActiveNeRF/GS-Planner/POp-GS-Basislinie.
- Papier D: Erstellen Sie nur einen leichten Piloten, organisieren Sie 76 Millionen Explorationsprotokolle in der Abdeckungs-/Fehlertaxonomie und konkurrieren Sie nicht um A/B/C-Ressourcen.

### 6,4 Wochen 9–12: Aufbau des minimalen geschlossenen Kreislaufs des Wolkengehirns in geringer Höhe

- Stellen Sie A/B/C/D/E als Tool-Schnittstellen bereit: Scheduler, Konfliktlöser, aktiver Mapper, Szenariotester, Verifizierer.
- Definieren Sie „LowAltitudeIR“, um Missions-, Luftraum-, UAV-, Ressourcen-, Risiko- und Tool-Call-Ergebnisse zu vereinheitlichen.
- Verwenden Sie zunächst den API-Lehrer + lokales Qwen/DeepSeek, um die CloudBrain-Agent-Basislinie zu erstellen, und beeilen Sie sich nicht mit der Feinabstimmung.
- Sammeln Sie Werkzeugverfolgung, Fehlerreparatur und Simulations-Rollout als Trainingsdaten für Paper J.

### 6,5 Wochen 13–20: Entscheidung über Einreichung und Modellrouten- Wenn die Warteschlangenstabilität und die Hunderte-Regal-Level-Ergebnisse von Papier B am stabilsten sind: Stimmen Sie zuerst für TR-C.
- Wenn Papier A die stärkste Konfliktsicherheit und Verallgemeinerung aufweist: stimmen Sie zuerst für T-ITS/T-RO.
- Wenn Papier C die stärksten theoretischen und visuellen Ergebnisse von Fisher + 3DGS aufweist: stimmen Sie zuerst für T-RO/ICRA/IROS.
- Wenn Papier D über die besten Abdeckungs-/Fehlererkennungsdaten verfügt: Investieren Sie zuerst in T-ITS.
- Wenn CloudBrain-Agent A/B/C/D/E-Tools bereits stabil aufrufen kann: Starten Sie die AAAI/IJCAI-Version.
– Wenn 5.000–20.000 hochwertige Werkzeugspuren/Verifizierer-Feedback/Simulations-Rollout gesammelt wurden: Starten Sie LowAltitudeGPT LoRA/SFT.
- Wenn die Arbeitslast des Agenten festgelegt ist und die Latenz zum Engpass wird: Starten Sie das vLLM/TensorRT/Edge-Quantisierungsexperiment von Paper K.

---

## 7. Referenzen

[1] Elsevier. *Transportation Research Teil C: Neue Technologien: Ziele und Umfang.* URL: <https://www.sciencedirect.com/journal/transportation-research-part-c-emerging-technologies>

[2] IEEE Intelligent Transportation Systems Society. *IEEE-Transaktionen auf intelligenten Transportsystemen: Geltungsbereich.* URL: <https://ieee-itss.org/pub/t-its/>[3] Chao Yu, Akash Velu, Eugene Vinitsky, Yu Wang, Alexandre M. Bayen und Yi Wu. „Die überraschende Wirksamkeit von PPO in kooperativen Multi-Agent-Spielen.“ *Fortschritte in neuronalen Informationsverarbeitungssystemen*, 2022. URL: <https://arxiv.org/abs/2103.01955>

[4] Muning Wen, Jakub Grudzien Kuba, Runji Lin, Weinan Zhang, Ying Wen, Jun Wang und Yaodong Yang. „Multi-Agent Reinforcement Learning ist ein Problem der Sequenzmodellierung.“ *NeurIPS*, 2022. URL: <https://proceedings.neurips.cc/paper_files/paper/2022/hash/69413f87e5a34897cd010ca698097d0a-Abstract-Conference.html>

[5] Bei Peng, Tabish Rashid, Christian Schroeder de Witt, Pierre-Alexandre Kamienny, Philip Torr, Wendelin Boehmer und Shimon Whiteson. „FACMAC: Faktorisierte zentralisierte Multi-Agent-Policy-Gradienten.“ *NeurIPS*, 2021. URL: <https://proceedings.neurips.cc/paper/2021/hash/65b9eea6e1cc6bb9f0cd2a47751a186f-Abstract.html>[6] Jakub Grudzien Kuba, Ruiqing Chen, Muning Wen, Ying Wen, Fudan Sun, Jun Wang und Yaodong Yang. „Optimierung der Trust-Region-Richtlinie beim Multi-Agent-Reinforcement-Learning.“ arXiv:2109.11251, 2021. URL: <https://arxiv.org/abs/2109.11251>

[7] Boyu Zhou, Xin Zhou, Jun Zhang, Fei Gao und Shaojie Shen. „EGO-Swarm: Ein vollständig autonomes und dezentrales Quadrotor-Schwarmsystem in überfüllten Umgebungen.“ *ICRA*, 2021. DOI: 10.1109/ICRA48506.2021.9561902. URL: <https://arxiv.org/abs/2011.04183>

[8] Jesus Tordesillas, Brett T. Lopez und Jonathan P. How. „MADER: Trajektorienplaner in Multiagenten- und dynamischen Umgebungen.“ *IEEE Transactions on Robotics*, 38(1):463-476, 2022. URL: <https://arxiv.org/abs/2010.11061>[9] Kota Kondo, Reinaldo Figueroa, Juan Rached, Jesus Tordesillas, Parker C. Lusk und Jonathan P. How. „Robust MADER: Dezentraler Multiagenten-Trajektorienplaner, robust gegenüber Kommunikationsverzögerungen in dynamischen Umgebungen.“ arXiv:2303.06222, 2023. URL: <https://arxiv.org/abs/2303.06222>

[10] Boyu Zhou, Hao Xu und Shaojie Shen. „RACER: Schnelle kollaborative Erkundung mit einem dezentralen Multi-UAV-System.“ *IEEE Transactions on Robotics*, 2023. DOI: 10.1109/TRO.2023.3236945. URL: <https://arxiv.org/abs/2209.08533>

[11] Jesus Tordesillas und Jonathan P. How. „PANTHER: Wahrnehmungsbewusster Trajektorienplaner in dynamischen Umgebungen.“ *IEEE Access*, 10:22662-22677, 2022. DOI: 10.1109/ACCESS.2022.3154037. URL: <https://arxiv.org/abs/2103.06372>[12] Zhepei Wang, Xin Zhou, Chao Xu und Fei Gao. „Geometrisch eingeschränkte Flugbahnoptimierung für Multikopter.“ *IEEE Transactions on Robotics*, 38(5):3259-3278, 2022. DOI: 10.1109/TRO.2022.3160022. URL: <https://arxiv.org/abs/2103.00190>

[13] Ang Li, Mark Hansen und Bo Zou. „Verkehrsmanagement und Ressourcenzuweisung für die UAV-basierte Paketzustellung im städtischen Raum in geringer Höhe.“ *Transportation Research Part C: Emerging Technologies*, 143:103808, 2022. DOI: 10.1016/j.trc.2022.103808. URL: <https://doi.org/10.1016/j.trc.2022.103808>

[14] Mehdi Bennaceur, Rémi Delmas und Youssef Hamadi. „Passagierzentrierte städtische Luftmobilität: Fairness-Kompromisse und betriebliche Effizienz.“ *Transportation Research Part C: Emerging Technologies*, 136:103519, 2022. DOI: 10.1016/j.trc.2021.103519. URL: <https://doi.org/10.1016/j.trc.2021.103519>[15] Roberto Pinto und Alexandra Lagorio. „Drohnenbasierter Punkt-zu-Punkt-Liefernetzwerkentwurf mit Zwischenladestationen.“ *Transportation Research Part C: Emerging Technologies*, 135:103506, 2022. DOI: 10.1016/j.trc.2021.103506. URL: <https://doi.org/10.1016/j.trc.2021.103506>

[16] Qinshuang Wei, Gustav Nilsson und Samuel Coogan. „Kapazitätsbeschränkte städtische Flugmobilitätsplanung.“ arXiv:2107.02900, 2021. URL: <https://arxiv.org/abs/2107.02900>

[17] Surya Murthy, Natasha A. Neogi und Suda Bharadwaj. „Planung für urbane Luftmobilität durch sicheres Lernen.“ arXiv:2209.15457, NASA NTRS, 2022. URL: <https://arxiv.org/abs/2209.15457>[18] Jiahao Xing, Tong Guo und Lu Tong. „Zuverlässiges Truck-Drohnen-Routing mit dynamischer Synchronisierung: Ein hochdimensionaler Netzwerkprogrammierungsansatz.“ *Transportation Research Part C: Emerging Technologies*, 165:104698, 2024. DOI: 10.1016/j.trc.2024.104698. URL: <https://doi.org/10.1016/j.trc.2024.104698>

[19] Bolong Zhou, Wei Zeng und Hai Yang. „Multi-Trip UAV-UGV Delivery Network Design mit Release-Zeiten.“ *Transportation Research Part C: Emerging Technologies*, 181:105389, 2025. DOI: 10.1016/j.trc.2025.105389. URL: <https://doi.org/10.1016/j.trc.2025.105389>

[20] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler und George Drettakis. „3D-Gaußsches Splatting für Echtzeit-Radiance-Field-Rendering.“ *ACM Transactions on Graphics / SIGGRAPH*, 42(4), 2023. DOI: 10.1145/3592433. URL: <https://arxiv.org/abs/2308.04079>[21] Xuran Pan, Zihang Lai, Shiji Song und Gao Huang. „ActiveNeRF: Mit Unsicherheitsschätzung lernen, wo man sehen kann.“ *ECCV*, 2022. URL: <https://arxiv.org/abs/2209.08546>

[22] Wen Jiang, Boshu Lei und Kostas Daniilidis. „FisherRF: Aktive Ansichtsauswahl und Kartierung mit Strahlungsfeldern unter Verwendung von Fisher-Informationen.“ *ECCV*, 2024. DOI: 10.1007/978-3-031-72624-8_24. URL: <https://eccv.ecva.net/virtual/2024/oral/1226>

[23] Rui Jin, Yuman Gao, Yingjian Wang, Haojian Lu und Fei Gao. „GS-Planner: Ein auf Gauß-Splatting basierendes Planungsrahmenwerk für die aktive High-Fidelity-Rekonstruktion.“ arXiv:2405.10142, 2024. URL: <https://arxiv.org/abs/2405.10142>

[24] Zijun Xu, Rui Jin, Ke Wu, Yi Zhao, Zhiwei Zhang, Jieru Zhao, Fei Gao, Zhongxue Gan und Wenchao Ding. „HGS-Planner: Hierarchischer Planungsrahmen für die Rekonstruktion aktiver Szenen mithilfe von 3D-Gauß-Splatting.“ arXiv:2409.17624, 2024. URL: <https://arxiv.org/abs/2409.17624>[25] Joey Wilson, Marcelino Almeida, Sachit Mahajan, Martin Labrie, Maani Ghaffari, Omid Ghasemalizadeh, Min Sun, Cheng-Hao Kuo und Arnab Sen. „POp-GS: Next Best View in 3D-Gaussian Splatting with P-Optimality.“ *CVPR*, 2025. URL: <https://cvpr.thecvf.com/virtual/2025/poster/34708>

[26] Shangjie Xue, Jesse Dill, Pranay Mathur, Frank Dellaert, Panagiotis Tsiotras und Danfei Xu. „Neuronales Sichtbarkeitsfeld für unsicheres aktives Mapping.“ *CVPR*, 2024. URL: <https://arxiv.org/abs/2406.06948>

[27] Shuo Feng, Xintao Yan, Haowei Sun, Yiheng Feng und Henry X. Liu. „Intelligenter Fahrintelligenztest für autonome Fahrzeuge in naturalistischer und kontroverser Umgebung.“ *Nature Communications*, 12:748, 2021. DOI: 10.1038/s41467-021-21007-8. URL: <https://www.nature.com/articles/s41467-021-21007-8>[28] Shuo Feng, Yiheng Feng, Chao Yu, Yi Zhang und Henry X. Liu. „Testen der Szenariobibliotheksgenerierung für vernetzte und automatisierte Fahrzeuge, Teil I: Methodik.“ *IEEE Transactions on Intelligent Transportation Systems*, 2021. DOI: 10.1109/TITS.2020.2972211. URL: <https://doi.org/10.1109/TITS.2020.2972211>

[29] Shuo Feng, Yiheng Feng, Haowei Sun, Yi Zhang und Henry X. Liu. „Testen der Szenariobibliotheksgenerierung für vernetzte und automatisierte Fahrzeuge, Teil II: Fallstudien.“ *IEEE Transactions on Intelligent Transportation Systems*, 2021. DOI: 10.1109/TITS.2020.2988309. URL: <https://doi.org/10.1109/TITS.2020.2988309>[30] Chejian Xu, Wenhao Ding, Baiming Li, Xinqing Chen, Ding Zhao, Bo Li, Jiajun Liu und Hang Zhao. „SafeBench: Eine Benchmarking-Plattform zur Sicherheitsbewertung autonomer Fahrzeuge.“ *NeurIPS Datasets and Benchmarks*, 2022. URL: <https://proceedings.neurips.cc/paper_files/paper/2022/hash/a48ad12d588c597f4725a8b84af647b5-Abstract-Datasets_and_Benchmarks.html>

[31] Jason Liu, Ankit Shah, Eric Rosen und Stefanie Tellex. „Lang2LTL: Übersetzen natürlicher Sprachbefehle in zeitliche Roboteraufgabenspezifikationen.“ *PMLR/CoRL*, 229, 2023. URL: <https://proceedings.mlr.press/v229/liu23d.html>

[32] Francesco Fuggitti und Tathagata Chakraborti. „NL2LTL: Ein Python-Paket zum Konvertieren natürlichsprachlicher Anweisungen in lineare temporale Logikformeln.“ *AAAI Demonstration*, 37(13):16428-16430, 2023. DOI: 10.1609/aaai.v37i13.27068. URL: <https://ojs.aaai.org/index.php/AAAI/article/view/27068>[33] Behrad Rabiei und Mahesh A. Kumar. „LTLCodeGen: Codegenerierung syntaktisch korrekter zeitlicher Logik für die Roboteraufgabenplanung.“ arXiv:2503.07902, 2025. URL: <https://arxiv.org/abs/2503.07902>

[34] Jun Wang, David Smith Sundarsingh, Jyotirmoy V. Deshmukh und Yiannis Kantaros. „ConformalNL2LTL: Übersetzen von Anweisungen in natürlicher Sprache in temporale Logikformeln mit konformen Korrektheitsgarantien.“ arXiv:2504.21022, 2025. URL: <https://arxiv.org/abs/2504.21022>

[35] Zhonghang Li, Lianghao Xia, Jiabin Tang, Yong Xu, Lei Shi, Long Xia, Dawei Yin und Chao Huang. „UrbanGPT: Räumlich-zeitliche große Sprachmodelle.“ arXiv:2403.00813, 2024. URL: <https://arxiv.org/abs/2403.00813>

[36] Yuan Yuan, Jingtao Ding, Jie Feng, Depeng Jin und Yong Li. „UniST: Ein Prompt-gestütztes Universalmodell für urbane räumlich-zeitliche Vorhersagen.“ *KDD*, 2024. DOI: 10.1145/3637528.3671662. URL: <https://dblp.org/rec/conf/kdd/0032D0J024>[37] Jinhui Ouyang, Yijie Zhu, Xiang Yuan und Di Wu. „TrafficGPT: Auf dem Weg zur mehrskaligen Verkehrsanalyse und -generierung mit dem räumlich-zeitlichen Agenten-Framework.“ arXiv:2405.05985, 2024. URL: <https://arxiv.org/abs/2405.05985>

[38] Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Jens Beisswenger, Ping Luo, Andreas Geiger und Hongyang Li. „DriveLM: Fahren mit grafischer Beantwortung visueller Fragen.“ *ECCV*, 2024. URL: <https://github.com/OpenDriveLab/DriveLM>

[39] Maximilian Adang, JunEn Low, Ola Shorinwa und Mac Schwager. „SINGER: Eine allgemeine Vision-Language-Navigationsrichtlinie für Drohnen an Bord.“ arXiv:2509.18610, 2025. URL: <https://arxiv.org/abs/2509.18610>[40] Hengxing Cai, Jinhan Dong, Jingjun Tan, Jingcheng Deng, Sihang Li, Zhifeng Gao, Haidong Wang, Zicheng Su, Agachai Sumalee und Renxin Zhong. „FlightGPT: Auf dem Weg zu einer verallgemeinerbaren und interpretierbaren UAV-Vision-and-Language-Navigation mit Vision-Language-Modellen.“ *EMNLP*, 2025. DOI: 10.18653/v1/2025.emnlp-main.338. URL: <https://aclanthology.org/2025.emnlp-main.338/>

[41] Pranav Saxena, Nishant Raghuvanshi und Neena Goveas. „UAV-VLN: End-to-End Vision Language gesteuerte Navigation für UAVs.“ arXiv:2504.21432, 2025. URL: <https://arxiv.org/abs/2504.21432>[42] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang und Chelsea Finn. „OpenVLA: Ein Open-Source-Vision-Sprache-Aktionsmodell.“ arXiv:2406.09246, 2024. URL: <https://arxiv.org/abs/2406.09246>

[43] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, Jianlan Luo, You Liang Tan, Lawrence Yunliang Chen, Pannag Sanketi, Quan Vuong, Ted Xiao, Dorsa Sadigh, Chelsea Finn und Sergey Levine. „Octo: Eine Open-Source-Richtlinie für generalistische Roboter.“ arXiv:2405.12213, 2024. URL: <https://arxiv.org/abs/2405.12213>[44] Anthony Brohan, Noah Brown, Richter Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. „RT-2: Vision-Language-Action-Modelle übertragen Webwissen auf Robotersteuerung.“ arXiv:2307.15818, 2023. URL: <https://arxiv.org/abs/2307.15818>

[45] Qwen-Team. „Technischer Bericht von Qwen3.“ arXiv:2505.09388, 2025; Offizielles QwenLM/Qwen3-Repository. URL: <https://arxiv.org/abs/2505.09388>; <https://github.com/QwenLM/Qwen3>

[46] DeepSeek-KI. „DeepSeek-R1: Anreize für die Denkfähigkeit in LLMs durch Reinforcement Learning schaffen.“ arXiv:2501.12948, 2025. URL: <https://arxiv.org/abs/2501.12948>; Modellkarte: <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B>

[47] OpenAI. „Model Context Protocol (MCP): OpenAI Agents SDK.“ Offizielle Dokumentation, 2026. URL: <https://openai.github.io/openai-agents-js/guides/mcp/>[48] ​​Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang und Ion Stoica. „Effiziente Speicherverwaltung für die Bereitstellung großer Sprachmodelle mit PagedAttention.“ arXiv:2309.06180, 2023. URL: <https://arxiv.org/abs/2309.06180>

[49] NVIDIA. „NVIDIA TensorRT-LLM.“ Offizielle Dokumentation, 2026. URL: <https://docs.nvidia.com/tensorrt-llm/index.html>

[50] Jinpeng Hu, Wei Wang, Yuxiao Liu und Jing Zhang. „Großes Modell in der Tieflandwirtschaft: Anwendungen und Herausforderungen.“ *Big Data and Cognitive Computing*, 10(1):33, 2026. DOI: 10.3390/bdcc10010033. URL: <https://www.mdpi.com/2504-2289/10/1/33>

[51] Ji Zhao und Xiao Lin. „Allzweck-intelligente Luftagenten, unterstützt durch große Sprachmodelle.“ arXiv:2503.08302, 2025. URL: <https://arxiv.org/abs/2503.08302>