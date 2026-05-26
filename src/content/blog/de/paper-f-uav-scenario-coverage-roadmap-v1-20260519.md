---
title: "Paper F Paper Group Planning v1: Erstellung, Abdeckung und Notfallanwendung sicherheitskritischer UAV-Szenarien"
description: "Es sind mehrere Papierrouten für die Generierung sicherheitskritischer UAV-Szenen, die Szenenabdeckung, die Korrelation zwischen Stadt- und Lokalszenen und Anweisungen zur Zuweisung von Hochgeschwindigkeits-Notfallrettungsressourcen geplant."
pubDate: 2026-05-19
tags: ["Papier F", "UAV", "Szenengenerierung", "Szenenberichterstattung", "Sicherheitskritisch", "beschleunigtes Testen", "Notfallrettung", "TR-C", "T-ITS"]
category: Tech
sourceHash: "e8583115b5944094ad19a72285ecf76f319d06d8"
---

# Papier F Papiergruppenplanung v1: Erstellung, Abdeckung und Notfallanwendung sicherheitskritischer UAV-Szenarien

> Gesamturteil: Zusätzlich zur Systemplanung auf Hunderten von Regalebenen von Paper B, der aktiven 3DGS/FIM-Erkennung von Paper C und der LLM/LTL-Sprachplanung von Paper E können Sie auch eine separate Papierlinie **UAV-sicherheitskritisches Szenario-Engineering** eröffnen.  
> Der Kern dieser Linie besteht nicht darin, einen weiteren Hindernisvermeidungsalgorithmus zu erstellen, sondern zu antworten: **Wie man systematisch wichtige UAV-Sicherheitsszenarien generiert, abdeckt, filtert und wiederverwendet, damit nachfolgende Schulungen, Tests, Notfalleinsätze und Papierexperimente eine glaubwürdige Szenariobasis haben. **

---

## 1. Gesamturteil: Welche weiteren Anweisungen können geschrieben werden?

Derzeit gibt es mehrere Papierlinien, die sich auf unterschiedliche Themen konzentrieren:

| Thesenzeile | Kernobjekte | Bereits abgedeckt | Sollte nicht wiederholt werden |
|--------|----------|----------|----------|
| Papier B | Hunderte UAV-Flotte | Dreischichtige hierarchische Planung, Warteschlangentheorie, Lyapunov, multimodaler Transport | Große Flottenplanung nicht mehr separat schreiben |
| Papier C | Aktive UAV-Erkennung | 3DGS, Fisher-Informationen, nächstbeste Ansicht | Nicht mehr auf Mapping/Perspektivauswahl konzentriert |
| Papier E | Sprache zur Planung | LLM, TaskIR, LTL/STL, formale Verifizierung | Nicht mehr auf die Aufgabenplanung in natürlicher Sprache konzentriert |
| Papier F | Szenentechnik | Szenengenerierung, Berichterstattung, gefährliche Szenen, Notfallanwendungen | Neue Wege |

Der Wert von Paper F besteht darin, dass es zur **experimentellen Infrastruktur** für die vorherigen Papers werden kann:

- Papier B erfordert städtische Bedürfnisse und Notfallszenarien.
- Paper C erfordert kontrollierbare lokale 3D-Szenen und Okklusions-/Perspektivabdeckung.
- Papier E erfordert Aufgabensemantik, Kartenentitäten und Sicherheitsbeschränkungen.
- Papier F kann eine einheitliche Szenariogrammatik, Abdeckungsmetrik, Kritikalitätsbewertung und Benchmark bereitstellen.Für die von Ihnen erwähnte „FENG SHOU“-Richtung wird empfohlen, dass der Standard der Arbeit von **Shuo Feng** für beschleunigte Tests/Testszenario-Bibliotheksgenerierung für automatisiertes Fahren entspricht. Die Kernidee ist: Sicherheitskritische Ereignisse sind in natürlichen Daten äußerst selten, daher können wir uns nicht nur auf gewöhnliche Zufallstests verlassen, sondern müssen Datenmethoden verwenden, um gefährlichere, aber dennoch vernünftige Szenarien zu konstruieren, wodurch Tests und Sicherheitsüberprüfungen beschleunigt werden [1] [2] [3]. Diese Idee eignet sich sehr gut für die Umstellung auf UAV-Hindernisvermeidung, Kanalflug in geringer Höhe und Hochgeschwindigkeits-Notfallinspektion.

---

## 2. Hierarchisches Design der Papiergruppe

Es wird empfohlen, Arbeit F als 4 progressive Arbeiten zu planen:

| Ebene | Papier | Ein-Satz-Positionierung | Priorität |
|------|------|------------|--------|
| F1 | CovUAV-Bank | Benchmark zur Abdeckung sicherheitskritischer UAV-Szenen | Höchste |
| F2 | Abdeckungsgesteuertes beschleunigtes Testen | Abdeckungsgesteuerter Algorithmus zur Generierung der Beschleunigung gefährlicher Szenen | Höchste |
| F3 | City2Local-UAV | Erstellen Sie eine hierarchische Szenengenerierung von der gesamten Stadt-ODD bis zur lokalen Hinderniskombination | Mittel bis hoch |
| F4 | Szenariobewusste Notfallreaktion | Führen Sie in Shandong einen kollaborativen Notfalleinsatz für Hochgeschwindigkeits-UAVs am Boden durch | Mittel bis hoch |

Es wird empfohlen, zuerst **F1 + F2** auszuführen. F1 stellt den Datensatz, die Metriken und die Problemdefinition bereit, und F2 liefert die algorithmischen Beiträge. F3 und F4 können als Erweiterungen verwendet werden: F3 verwandelt den Benchmark in ein System auf Stadtebene und F4 verwandelt die Szenentechnik in ein echtes Verkehrsnotfallgeschäft.

---

## 3. Gemeinsamer Hintergrund: Warum die Szenenabdeckung die Grundlage für die UAV-Sicherheitsforschung ist

Eine häufige Schwäche in der UAV-Sicherheitsforschung ist: Der Algorithmus ist schön gemacht, aber das experimentelle Szenario ist zu willkürlich. Nur weil ein Hindernisvermeidungsalgorithmus in 20 manuellen Szenarien erfolgreich ist, heißt das nicht, dass er die langfristigen Risiken bei städtischen Einsätzen in geringer Höhe abdeckt.Im Bereich des autonomen Fahrens besteht ein klarer Konsens: Unfälle/Beinahe-Unfälle auf realen Straßen sind seltene Ereignisse und es wäre äußerst ineffizient, sich direkt auf natürliche Tests zu verlassen. Daher haben Shuo Feng et al. schlug eine naturalistische und kontradiktorische Umgebung vor, die natürliche Verteilung nutzt, um die Authentizität aufrechtzuerhalten, und kontradiktorische Verteilung nutzt, um die Wahrscheinlichkeit gefährlicher Ereignisse zu erhöhen, wodurch intelligente Fahrtests beschleunigt werden [1]. Sie schlugen außerdem die Erstellung einer Testszenariobibliothek vor, definierten die Testszenariobibliothek unter ODD als eine Reihe repräsentativer und kritischer Szenarien und nutzten die Kritikalität, um die Expositionshäufigkeit und die Manöverherausforderung zu berücksichtigen [2] [3]. In der Übersicht von Ding et al. zur Generierung sicherheitskritischer Szenarien wurde das Feld ebenfalls in drei Arten von Methoden eingeteilt: datengesteuert, kontradiktorisch und wissensbasiert, und es wurde darauf hingewiesen, dass Treue, Effizienz, Diversität, Übertragbarkeit und Kontrollierbarkeit zentrale Herausforderungen sind [13].

Das UAV-Szenario erfordert diese Ideen aus vier Gründen noch mehr:

1. **Der dreidimensionale Raum ist eine höhere Dimension. **
   Bei UAVs geht es nicht nur um flache Fahrbahnen, sondern auch um Höhe, Hindernisvolumen, Windfeld, elektrische Ladung, Sensorsichtfeld und Flugdynamik.

2. **Gefährliche Ereignisse sind schwieriger zu erfassen. **
   Es gibt nur sehr wenige Beispiele für reale Kollisionen mit Gebäuden, Linienkollisionen, das Betreten von Flugverbotszonen, das Überqueren von Brücken oder Unfallstellen mit hoher Geschwindigkeit, und die Schulung kann sich nicht auf echte Unfalldaten stützen.

3. **Gewöhnliche Zufallsgenerierung verschwendet Rechenleistung. **
   Viele Zufallsszenarien sind entweder zu einfach, physikalisch nicht realisierbar oder gefährlich, aber unvermeidlich, was sie für Training und Bewertung ineffizient macht.

4. **Es gibt keine einheitliche Messung der Szenenabdeckung. **
   Vorhandene UAV-Papiere berichten häufig über Erfolgsraten/Kollisionsraten, geben jedoch selten an, welche Hinderniskombinationen, lokalen Geometrien, Aufgabenschwierigkeiten und ODD-Grenzen vom Testsatz abgedeckt werden.

Daher sind die häufigsten wissenschaftlichen Fragen für Arbeit F:

> Wie kann ein System zur Generierung und Bewertung von UAV-Szenarien aufgebaut werden, das real, kontrollierbar und reproduzierbar ist und wichtige langfristige Sicherheitsrisiken wirksam abdecken kann?

---

## 4. Papier F1: Benchmark für die Abdeckung sicherheitskritischer UAV-Szenen

### 4.1 Titel der Abschlussarbeit**CovUAV-Bench: Ein abdeckungsorientierter Benchmark für sicherheitskritische UAV-Navigationsszenarien**

### 4.2 Hintergrund

SafeBench bietet bereits einen einheitlichen sicherheitskritischen Benchmark für autonomes Fahren und integriert mehrere Arten von Szenenvorlagen, Szenengenerierungsalgorithmen und Bewertungsindikatoren [5]. Scenic hat auch bewiesen, dass die Verwendung probabilistischer Programme zur Darstellung der Szenenverteilung sowie harter und weicher Einschränkungen ein gangbarer Weg ist [4]. Es wurden Vorarbeiten zur Generierung einer UAV-Simulationsumgebung durchgeführt. Beispielsweise haben Nakama et al. schlug einen automatisierten UAV-Simulationsumgebungsgenerator vor [10]. FADS zeigte auch, dass zeitlich-logische Sicherheitsspezifikationen in die Sicherheitspipeline autonomer Drohnen einfließen können [11]. Allerdings mangelt es im UAV-Bereich immer noch an einem abdeckungsorientierten Maßstab für die Vermeidung von 3D-Hindernissen, Korridoren in geringer Höhe, städtischen lokalen Räumen und Notfallaufgaben.

Das Ziel von F1 besteht nicht darin, den stärksten Planer vorzuschlagen, sondern zu definieren, wie der UAV-Szenenraum vom System abgedeckt wird.

### 4.3 Methode

Konstruieren Sie einen grundlegenden Testraum von 50 x 50 x 50 m, beginnend mit lokalen Szenen und erweitern Sie ihn dann auf städtische Blöcke:- **Szenenobjekte**: Bausteine, Türme, Drähte, Bäume, Brücken, temporäre Hindernisse, dynamische UAVs, Bodenfahrzeuge, Personalbereiche.
- **Raumstruktur**: Freiraum, Engpass, Häuserschlucht, Unterbrücke, Landezone, Autobahnrand, Unfallzone.
- **Umweltstörungen**: Wind, Sicht, Sensorrauschen, GPS-Offset, Kommunikationsverzögerungen.
- **Aufgabentyp**: Punkt-zu-Punkt-Navigation, Inspektionspass, Notschwebeflug, Landung, Rückkehr zur Heimat.
- **Ausführbares Format**: Speichern Sie es als „scenario.json“ und fügen Sie den Simulatoradapter hinzu. Es kann später auf AirSim, Flightmare, PyBullet oder selbst erstellte Leichtbausimulation umgestellt werden.

Szenenabdeckung ist definiert als:

$$
Abdeckung(S)=
\sum_{k=1}^{K} w_k \cdot
\frac{|B_k(S)|}{|B_k(\Omega)|},
$$

Dabei ist $\Omega$ der diskretisierte Szenenraum des ungeraden Zielobjekts, $B_k(S)$ ist der vom Stichprobensatz $S$ auf der $k$-ten Klassenattributdimension abgedeckte Bin und $w_k$ ist das Dimensionsgewicht.

Die vorhandenen **76 Millionen Erkundungen** können für Statistiken als „Vorhandene Erkundungsprotokoll-Assets“ geschrieben werden:

- Welche Szenenkombinationen werden häufig untersucht?
- Welche Kombinationen sind noch Abdeckungslücken?
- Welche Kombinationen lösen Kollision / Beinahe-Unfall / Timeout aus?
- Welche Kombinationen sind ungültige Trainingsbeispiele?

Hinweis: 76 Millionen Untersuchungen werden nur als „verfügbare experimentelle Grundlage“ geschrieben und können nicht als verifizierte Schlussfolgerungen geschrieben werden.

### 4.4 Grundlinien| Grundlinie | Zweck |
|----------|------|
| Zufällige Szenario-Stichprobe | Die grundlegendste Deckungsgrundlinie |
| Rasterprobenahme | Einheitliche Diskretisierung des Parameterraums |
| Lateinisches Hypercube-Sampling | Effizientere Parameterabdeckung |
| Eingeschränktes Sampling im szenischen Stil | Eingeschränkte Basislinie für die Szenengenerierung [4] |
| Vorlagensuite im SafeBench-Stil | Vorlage für Sicherheitsszenario-Basislinie [5] |

### 4,5 Innovationspunkte

1. Schlagen Sie eine Taxonomie für die UAV-Szenenabdeckung vor: Ungerade, Hinderniskombination, dynamische Störung, Aufgabentyp, Risikoniveau.
2. Geben Sie einen abdeckungsorientierten Benchmark an, statt nur ein paar manuelle Karten.
3. Konvertieren Sie das Explorationsprotokoll in Abdeckungslöcher und kritische Szenario-Seeds.
4. Stellen Sie eine einheitliche Szenenschnittstelle für nachfolgende Paper B/C/E bereit.

### 4.6 So bewerten Sie

| Indikator | Bedeutung |
|------|------|
| Parameterabdeckung | Parameter-Bin-Abdeckungsverhältnis |
| Paarweise / t-weise Abdeckung | Abdeckung mehrerer Attributkombinationen |
| Kritische Szenariodichte | Anzahl der entdeckten Beinahe-Unfälle/Kollisionen pro Unit-Test-Budget |
| Ungültiger Szenariopreis | Der Anteil physikalisch undurchführbarer oder aufgabenbedeutungsloser Szenarien |
| Stabilität des Planerrankings | Ist das Ranking des Algorithmus unter verschiedenen Zufallsstartwerten stabil?
| Reproduzierbarkeit der Wiedergabe | Ob das gleiche Ergebnis mit demselben Seed reproduziert werden kann |

### 4.7 Empfohlene Beiträge

- Hauptlinie: Benchmark-orientiertes Papier von T-ITS / IEEE ITSC / IROS.
- Alternative: RA-L + ICRA, wenn der Benchmark sowohl über hochwertige Open-Source-Tools als auch über eine kleinräumige Verifizierung realer UAVs verfügt.

---

## 5. Papier F2: Beschleunigen Sie die Generierung gefährlicher Szenen anhand der Berichterstattung### 5.1 Titel der Abschlussarbeit

**Abdeckungsgesteuerte beschleunigte Tests zur Vermeidung sicherheitskritischer UAV-Hindernisse**

### 5.2 Hintergrund

Der Kern beschleunigter Tests für autonomes Fahren besteht nicht darin, „unvermeidliche Unfallszenarien zu schaffen“, sondern darin, die Stichprobeneffizienz sicherheitskritischer Ereignisse zu verbessern und gleichzeitig die Authentizität und Umsetzbarkeit der Szene zu wahren [1] [2] [3]. Wenn das generierte Szenario für keinen Planer realisierbar ist, kann es nicht dazu beitragen, die Fähigkeiten des Algorithmus zu differenzieren. Wenn das generierte Szenario zu sicher ist, können keine Systemschwächen aufgedeckt werden.

Das gleiche Problem besteht auch beim UAV-Hindernisvermeidungstraining:

- Eine große Anzahl zufällig generierter Szenen ohne Sicherheitsdruck.
- Die Generierung von Konfrontationen führt tendenziell zu Hindernisanordnungen, die nicht vernünftigerweise vermieden werden können.
- Der manuelle Lehrplan deckt nur begrenzt ab und kann nicht erklären, ob langfristige Risiken abgedeckt sind.
- RL-Training verschwendet Budget für viele ungültige Szenarien.

### 5.3 Methode

Vorgeschlagener **CGAT-UAV: Coverage-Guided Accelerated Testing for UAVs**.

Der Algorithmus besteht aus vier Modulen:

1. **Szenario-Encoder**
   Kodieren Sie die Szene in strukturierte Vektoren: Anzahl der Hindernisse, minimale Kanalbreite, Zielrichtung, dynamische Hindernisgeschwindigkeit, Windintensität, Sensorrauschen, Batteriereserve usw.

2. **Abdeckungsspeicher**
   Behalten Sie Abdeckungsklassen, Fehlertypen und Planerleistung für erkundete Szenen bei.

3. **Kritikalitätswert**
   Beziehen Sie sich auf Fengs Kritikalitätsidee und kombinieren Sie den Grad des Risikos mit der Häufigkeit der Exposition [2]:

   $$
   Krit(s)=P_{\text{Belichtung}}(s)\cdot R_{\text{Herausforderung}}(s)\cdot F_{\text{machbar}}(s).
   $$

   Unter anderem wird $F_{\text{feasible}}(s)$ verwendet, um unvermeidliche Kollisionen und physikalisch unzumutbare Szenarien zu bestrafen.4. **Adaptiver Generator**
   Generieren Sie neue Szenen in Abdeckungslöchern und Regionen mit hoher Kritikalität mithilfe von Bayes'schen Optimierungs-, CMA-ES-, RL-Bearbeitungs- oder Kreuzentropiemethoden.

### 5.4 Grundlinien

| Grundlinie | Vergleichszweck |
|----------|----------|
| Zufällige Generierung | Beschleunigungsrate testen |
| Raster-/Latein-Hyperwürfel-Sampling | Abdeckungseffizienz |
| Bayesianische Optimierung | Black-Box-Gefährdungssuche |
| CMA-ES | Kontinuierliche parametrische Gefahrensuche |
| Generierung eines kontradiktorischen RL-Szenarios | Gefahrengenerierung lernen |
| Szenisch eingeschränkte Generation | Generierung von Regeln und Einschränkungen [4] |
| Machbarkeitsgesteuerte Generierung im FREA-Stil | Vergleichen Sie die Idee des „vernünftigen Antagonismus“ [12] |

### 5,5 Innovationspunkte

1. Migrieren Sie beschleunigte Tests vom autonomen Fahren zur UAV-3D-Hindernisvermeidung.
2. Optimieren Sie gleichzeitig **Abdeckung, Kritikalität und Machbarkeit**, um zu vermeiden, dass nur die Kollisionsrate verfolgt wird.
3. Schlagen Sie einen abdeckungsorientierten Lehrplan vor, um Planer mit gefährlichen, aber lösbaren Szenarien zu schulen.
4. Die Testbeschleunigungsrate ist angegeben: Die Anzahl der Simulationen, die zum Erreichen des gleichen Konfidenzintervalls erforderlich sind, wird erheblich reduziert.

### 5.6 So bewerten Sie| Indikator | Bedeutung |
|------|------|
| Beschleunigungsfaktor | Die mehrfache Reduzierung der Anzahl der Tests, die erforderlich sind, um die gleiche Fehlererkennungsrate im Vergleich zu Zufallstests zu erreichen |
| Fehlererkennungsrate | Das Verhältnis von Kollision/Beinahe-Unfall/entdeckter Zeitüberschreitung pro Einheitsbudget |
| Machbare Kritikalität | Gefahrenanteil und umsetzbare Hindernisvermeidungsstrategien |
| Natürlichkeitsbewertung | Ob die Szene ODD prior entspricht |
| Abdeckungsgewinn pro 1.000 Tests | Neue Abdeckung alle 1000 Tests |
| Trainingseffizienz | Nach dem Training mit generierten Szenarien, Verbesserung des Planers im durchgehaltenen Test |

### 5.7 Empfohlene Beiträge

- Hauptlinie: AAAI/ICRA/IROS.
- Alternative: T-ITS, wenn mehr Gewicht auf Verkehrssicherheitstests und Benchmarking gelegt wird.

---

## 6. Papier F3: Hierarchische Generierung städtischer Gesamtszenen zu lokalen Hinderniskombinationen

### 6.1 Titel der Abschlussarbeit

**City2Local-UAV: Hierarchische Szenariogenerierung von städtischen ODDs bis hin zu lokalen Hinderniszusammensetzungen**

### 6.2 Hintergrund

F1 und F2 befassen sich mit einem lokalen 3D-Testraum, echte städtische Tiefflüge sind jedoch keine isolierten Boxen. Warum eine lokale Szene entsteht, hängt von der Gesamtstruktur der Stadt ab: Straßenniveau, Bebauungsdichte, Funktionsbereiche, Brücken, Raststätten, Verkehrsknotenpunkte, Krankenhäuser, Schulen, Flugverbotszonen und Notfallpunkte.

ASAM OpenODD/OpenSCENARIO bietet eine standardisierte Idee von ODD, der aktuellen Betriebsdomäne, bis hin zur ausführbaren Szenariobeschreibung [6] [7]. Der UAV-Bereich kann von dieser Abstraktionsebene lernen, muss jedoch dreidimensionale Hindernisse, Luftraumbeschränkungen und die Semantik von Missionen in geringer Höhe berücksichtigen.

### 6.3 Methode

Schlagen Sie eine dreischichtige Stromerzeugungspipeline von der Stadt zum Ort vor:

```text
City-level ODD
  -> district / road / highway segment selection
  -> local 50m x 50m x 50m UAV test cell
  -> concrete obstacle composition
  -> simulator executable scenario
```

Spezifische Module:- **Stadt-ODD-Parser**: Extrahieren Sie die Stadt-/Autobahnsemantik aus OSM, Straßenniveaus, Gebäudeumrissen, POIs, Raststätten, Brücken und Autobahnauffahrten.
- **Lokaler Zell-Sampler**: Wählen Sie typische lokale Zellen aus, z. B. Hochhausschluchten, Raststätten, Überführungen, Mautstellen, Autobahnstreifen und Unfallengpässe.
- **Hindernisgrammatik**: Verwenden Sie Regeln, um lokale Hinderniskombinationen zu generieren, z. B. Gebäude + Leitungen + Bäume + geparkte Fahrzeuge + Bereiche mit Personenbeschränkung.
- **Abdeckungscontroller**: Überwacht die Abdeckung verschiedener städtischer Funktionsbereiche und lokaler Kombinationen.

### 6.4 Grundlinien

| Grundlinie | Vergleichszweck |
|----------|----------|
| Rein zufällige lokale Generierung | Berücksichtigt nicht den städtischen Kontext |
| Direkte OSM-zu-Karte-Konvertierung | Konvertiert nur die Karte, steuert nicht die Szenenabdeckung |
| CARLA / OSM digitale Zwillingsgeneration | Basislinie des digitalen Zwillings für autonomes Fahren am Boden [14] |
| Manuelle Szenariovorlagen | Manuelle Regelvorlagen |
| CityEngine / prozedurale Stadtgenerierung | Basislinie für die prozedurale Stadtgenerierung |

### 6,5 Innovationspunkte

1. Verknüpfen Sie das städtische ODD mit der lokalen UAV-Sicherheitstestzelle.
2. Schlagen Sie eine hierarchische Szenengenerierung der „Semantik globaler Städte -> Zusammensetzung lokaler Hindernisse“ vor.
3. Erweitern Sie die Szenenabdeckung von lokalen Parametern auf die Abdeckung städtischer Funktionsbereiche.
4. Unterstützen Sie Fallstudien aus realen Städten wie den wichtigsten Autobahnknotenpunkten Jinan, Qingdao und Shandong.

### 6.6 So bewerten Sie| Indikator | Bedeutung |
|------|------|
| ODD-Abdeckung | Städtische Funktionsflächen, Straßenniveaus, Bebauungsdichte |
| Lokale Kompositionsvielfalt | Lokale Hinderniskombinationsvielfalt |
| Realismus-Score | Konsistenz mit OSM/POI/Gebäudestatistiken |
| Übertragbarkeit | Ist die von einer Stadt generierte Police auch dann noch gültig, wenn sie in eine andere Stadt verschoben wird?
| Erhaltung der Kritikalität | Ob die Schaffung städtischen Kontexts gefährdete lokale Szenen bewahrt |

### 6.7 Empfohlene Beiträge

- Hauptlinie: TR-C, wenn städtische Verkehrssysteme, ODD, Infrastruktur in geringer Höhe und Szenendatensätze im Vordergrund stehen.
- Alternative: T-ITS, wenn OpenSCENARIO-ähnliche Szenarioschnittstelle und intelligente Systemauswertung im Vordergrund stehen.

---

## 7. Papier F4: Gemeinsamer Einsatz von UAV-Bodenressourcen für die Notfallrettung auf der Autobahn Shandong

### 7.1 Titel der Abschlussarbeit

**Szenariobewusste Zuweisung von UAV-Bodenressourcen für Notfallmaßnahmen auf der Autobahn**

### 7.2 Hintergrund

Shandong Expressway verfügt bereits über eine Geschäftsgrundlage für Inspektionen in geringer Höhe und Notfallmaßnahmen. Öffentliche Informationen der Shandong Hi-Speed ​​​​Group zeigen, dass ihr umfassendes Inspektionsflugdienstsystem unbeaufsichtigte Plattformen und Industriedrohnen in Schlüsselbereichen für Straßenzustandsinspektionen, Straßeninspektionen, Notfallmaßnahmen und Datenanalysen eingesetzt hat [15]. Dies zeigt, dass Hochgeschwindigkeitsszenarien keine reine Fantasie sind, sondern Anwendungsmöglichkeiten haben.

Untersuchungen zur Zuweisung von Notfallressourcen auf Autobahnen haben gezeigt, dass es bei der bestehenden Arbeit immer noch mehrere Probleme gibt: unzureichende Standortauswahl für kleine/kleine Notfalleinrichtungen am Straßenrand während der Betriebsphase, vollständige Informationen werden oft in der Frühphase des Unfalls angenommen, sind aber nicht tatsächlich zutreffend, der Verkehrsstatus nach dem Unfall ist unsicher und zeitlich schwankend, und die integrierte Optimierung der Standortauswahl der Einrichtung, der Ressourcenzuweisung und des Einsatzes ist immer noch unzureichend [16]. Es gab Studien zum UAV-Routing im Raum-Zeit-Netzwerk bei der Überwachung von Verkehrsunfällen [17] und es gab Studien zum UAV-Echtzeiteinsatz und zur Ressourcenzuweisung in der Katastrophen-Notfallkommunikation [18], aber sie haben noch keinen einheitlichen geschlossenen Regelkreis mit der Abdeckung von Hochgeschwindigkeits-Notfallorten, dem Informationswert der Aufklärung vor Ort und der Ressourcenzuweisung für die Bodenrettung gebildet.

Dies eignet sich für die Einführung von UAV: ​​Die Drohne kommt zunächst am Unfallort an, um die Situation zu erfassen, und dann werden die Bodenfreiheits-, Brandbekämpfungs-, Rettungs- und Kontrollressourcen dynamisch entsandt.

### 7.3 MethodeVorgeschlagener **Szenariobewusster UAV-Boden-Notfalleinsatz**:

- **Generierung von Unfallszenen**: Basierend auf der F1/F3-Hochgeschwindigkeits-Szenenbibliothek werden Unfallart, Verkehrsflussstatus, Wetter, Straßenabschnittsgeometrie, Hindernisse und Sekundärrisiken generiert.
- **UAV-Aufklärungsschicht**: UAVs starten von Raststätten, Mautstellen oder unbemannten Plattformen, um schnell Unfallorte, Staulängen, befahrbare Fahrspuren und Gefahrgutrisiken zu ermitteln.
- **Zuweisungsschicht für Bodenressourcen**: Entsendet Abschleppwagen, Feuerwehr, Krankenwagen, Verkehrspolizei, Wartungsfahrzeuge und temporäre Kontrollressourcen.
- **Informationswertmodellierung**: Schreiben Sie die Unsicherheitsreduzierung der UAV-Aufklärung in das Versandziel, das heißt, das UAV macht nicht nur Bilder, sondern reduziert auch falsche Versand- und Antwortverzögerungen.
- **Laufende Optimierung**: Unfallinformationen werden im Laufe der Zeit aktualisiert und Planungsstrategien werden fortlaufend neu berechnet.

### 7.4 Problemformulierung

Der Autobahnabschnittssatz sei $\mathcal{L}$, der Unfallsatz sei $\mathcal{I}(t)$, der UAV-Satz sei $\mathcal{U}$, der Bodenrettungsressourcensatz sei $\mathcal{G}$ und der Tankstellen-/unbeaufsichtigte Bahnsteigsatz sei $\mathcal{B}$.

Zu den Entscheidungsvariablen gehören:

- UAV-Versand $x_{u,i}(t)$: Ob UAV $u$ den Vorfall $i$ erkundet.
- Bodenressourceneinsatz $y_{g,i}(t)$: ob Ressource $g$ auf dem Weg zum Vorfall $i$ ist.
- Start-/Abflugzeit $s_u(t), s_g(t)$.
- Informationsaktualisierungsaktion $a_i(t)$: ob auf eine weitere Bestätigung vom UAV gewartet oder direkt versendet werden soll.

Zielfunktion:

$$
\min
\mathbb{E}\left[
\beta_1 T_{\text{Antwort}}+
\beta_2 T_{\text{Spielraum}}+
\beta_3 C_{\text{Versand}}+
\beta_4 R_{\text{sekundär}}+
\beta_5 U_{\text{Unsicherheit}}
\richtig].
$$

Darunter stellt $U_{\text{Unsicherheit}}$ die Unsicherheit von Unfallinformationen dar, die durch UAV-Aufklärung reduziert werden kann.

### 7.5 Grundlinien| Grundlinie | Vergleichszweck |
|----------|----------|
| Nur-Boden-Versand | Keine Drohnenaufklärung |
| Versand der nächstgelegenen Ressource | Nächste Ressourcen zuerst |
| Statische Anlagenzuordnung | Feste Anlagenzuteilung |
| Zweistufige stochastische Optimierung | Schätzen Sie den Unfall vor dem Versand |
| UAV-First-Heuristik | Zuerst UAV-Aufklärung, dann Bodeneinsatz |
| Szenariobewusste Rolling-Optimierung | Hauptmethode |

### 7,6 Innovationspunkte

1. Verbinden Sie die Einsatzortabdeckung mit der Hochgeschwindigkeits-Notrufabwicklung statt nur der Ressourcenzuweisung.
2. Modellieren Sie die UAV-Aufklärung als Entscheidungsmaßnahme, die die Unsicherheit bei Vorfallinformationen verringert.
3. Unterstützen Sie den tatsächlichen Geschäftskontext der Shandong-Schnellstraße: unbeaufsichtigte Plattform, Inspektion des Straßenzustands, Notfallmaßnahmen und Arbeitsauftragsverteilung.
4. Einheitliche Optimierung von Reaktionszeit, Räumungszeit, sekundärem Unfallrisiko und Versandkosten.

### 7.7 So bewerten Sie

| Indikator | Bedeutung |
|------|------|
| Erstansichtszeit | Der Zeitpunkt, als das UAV zum ersten Mal das Unfallmaterial aufnahm |
| Reaktionszeit | Ankunftszeit der ersten Ladung Rettungsmittel |
| Räumungszeit | Fertigstellungszeit der Unfallbeseitigung |
| Falsche Versandrate | Der Anteil falscher Sendungen, verpasster Sendungen oder unzureichender Ressourcen |
| Sekundäres Unfallrisiko | Sekundärer Unfallrisiko-Proxy |
| Stauverzögerung | Gesamtverspätung durch Unfall |
| UAV-Informationswert | Die Aufklärung mit UAV verringert die Unsicherheit im Vergleich zur Aufklärung ohne UAV |

### 7.8 Empfohlene Beiträge

- Hauptlinie: TR-C zuerst, da der Schwerpunkt auf dem Betrieb von Hochgeschwindigkeits-Nottransportsystemen, der Ressourcenzuweisung und der Widerstandsfähigkeit des Transportnetzes liegt.
- Alternative: T-ITS, wenn der Schwerpunkt stärker auf Drohnenplattformen, Kommunikation, Videoerkennung, Arbeitsauftragssystemen und intelligentem Online-Dispatch liegt.

---

## 8. Experimentelle Plattform, Datenquellen und Bewertungsindikatoren vereinheitlichen

### 8.1 Experimentelle Plattform| Hierarchie | Empfohlene Implementierung | Zweck |
|------|----------|------|
| Leichtbausimulation | Python / PyBullet / benutzerdefiniertes 3D-Raster | 76 Millionen Levels der schnellen Erkundung |
| UAV-Simulation | AirSim, Flightmare | Vision, Dynamik, Sensorverifizierung [8] [9] |
| Szenariosprache | Szenisches DSL-, JSON-Schema | Reproduzierbare Szenengenerierung [4] |
| Stadtdaten | OpenStreetMap, POI, Straßengrade, Gebäudeumrisse | Stadt-zu-Lokal-Szene-Generierung |
| Hochgeschwindigkeits-Notfall | Offene Fälle auf der Shandong-Schnellstraße, Unfallstatistik, synthetischer Unfallfluss | Experiment zur Ressourcenzuweisung im Notfall |

Das Hauptexperiment von F1/F2 sollte der Leichtbausimulation Priorität einräumen, um eine groß angelegte Erkundung sicherzustellen. AirSim/Flightmare wird für die High-Fidelity-Verifizierung im kleinen Maßstab verwendet und ist nicht für alle Experimente verlässlich.

### 8.2 Datenquelle

- **Benchmark für synthetische UAV-Szenarien**: Prozedural generierter lokaler Raum von 50 m x 50 m x 50 m.
- **Explorationsprotokolle**: 76 Millionen Explorationsprotokolle für Abdeckungslücken und Fehlertaxonomie.
- **OSM/POI/Gebäudedaten**: für städtische Funktionsbereiche und lokale Barrierekombinationen.
- **Shandong Expressway Public Business Information**: Wird für Anwendungshintergrund und Bereitstellungsannahmen verwendet [15].
- **Forschung zur Offenlegung von Ressourcen bei Hochgeschwindigkeitsunfällen und Notfällen**: Wird für Unfalltypen, Ressourcenzuteilungsstufen und Bewertungsindikatoren verwendet [16].

### 8.3 Einheitliche Indikatoren| Indikatorgruppe | Indikator |
|--------|------|
| Abdeckung | Parameterabdeckung, t-weise Abdeckung, ODD-Abdeckung, Abdeckungsgewinn |
| Sicherheit | Kollisionsrate, Beinahe-Unfall-Rate, Mindestabstand, Einschränkungsverletzung |
| Gefahrenerzeugung | Kritikalität, Fehlererkennungsrate, Beschleunigungsfaktor, mögliche Kritikalität |
| Trainingswert | Stichprobeneffizienz, anhaltende Erfolgsquote, Robustheit bei ODD-Verschiebung |
| Notfallwert | First-View-Zeit, Reaktionszeit, Clearance-Zeit, Falschversandrate |

---

## 9. Empfohlener Einreichungspfad und Priorität

### 9.1 Die erste Stufe: Führen Sie zuerst F1 + F2 aus

In der ersten Phase wird empfohlen, zwei Artikel direkt zum Thema „Abdeckung sicherheitskritischer UAV-Szenen + beschleunigte Tests“ zu schreiben:

1. **F1-Benchmark-Papier**
   Stabiler, geeignet als experimentelle Basis für alle nachfolgenden UAV-Veröffentlichungen. Auch wenn der Algorithmus nicht besonders stark ist, kann er dennoch auf der Grundlage von Taxonomie, Abdeckungsmetrik, Datensatz und reproduzierbaren Experimenten erstellt werden.

2. **F2-Methodenpapier**
   Methodische Beiträge zu AAAI/ICRA/IROS. Der Höhepunkt ist die Migration von Shuo Fengs beschleunigten Tests des autonomen Fahrens zu UAV-3D-Szenen und die Hinzufügung einer abdeckungsgesteuerten realisierbaren Kritikalität.

### 9.2 Phase 2: Wiederholen Sie F3 + F4

F3 und F4 eignen sich besser für den Aufstieg, nachdem F1/F2 über eine Werkzeuggrundlage verfügt:

- **F3** Um die Beziehung zwischen der gesamten Stadt und den lokalen Szenen zu lösen, können Sie für TR-C / T-ITS stimmen.
- **F4** Für Notfallrettungsanwendungen auf der Autobahn Shandong kann TR-C ausgewählt werden, wobei der Schwerpunkt auf Transportbetrieb und Notfallreaktion liegt.### 9.3 Zusammenhang mit bestehenden Papierlinien

| Papier | So unterstützen Sie Paper F |
|------|------|
| Papier B | Bietet Spitzen-/Schock-/Notfall-Szenarien auf der Autobahn |
| Papier C | Bietet lokale 3D-Okklusion, perspektivische Abdeckung und Rekonstruktion schwieriger Szenen |
| Papier E | Bietet Aufgaben in natürlicher Sprache, Kartenentitäten und Sicherheitseinschränkungsszenarien |

Papier F eignet sich am besten als „Szenario-Infrastrukturpapier“ für die gesamte UAV-Forschungslinie.

---

## 10. Referenzen

[1] Shuo Feng, Xintao Yan, Haowei Sun, Yiheng Feng und Henry X. Liu. „Intelligenter Fahrintelligenztest für autonome Fahrzeuge in naturalistischer und kontroverser Umgebung.“ *Nature Communications*, 12:748, 2021. DOI: 10.1038/s41467-021-21007-8. URL: <https://doi.org/10.1038/s41467-021-21007-8>

[2] Shuo Feng, Yiheng Feng, Chunhui Yu, Yi Zhang und Henry X. Liu. „Testen der Szenariobibliotheksgenerierung für vernetzte und automatisierte Fahrzeuge, Teil I: Methodik.“ *IEEE Transactions on Intelligent Transportation Systems*, 22(3):1573-1582, 2021. DOI: 10.1109/TITS.2020.2972211. URL: <https://doi.org/10.1109/TITS.2020.2972211>[3] Shuo Feng, Yiheng Feng, Haowei Sun, Shan Bao, Yi Zhang und Henry X. Liu. „Testen der Szenariobibliotheksgenerierung für vernetzte und automatisierte Fahrzeuge, Teil II: Fallstudien.“ *IEEE Transactions on Intelligent Transportation Systems*, 22(9):5635-5647, 2021. DOI: 10.1109/TITS.2020.2988309. URL: <https://doi.org/10.1109/TITS.2020.2988309>

[4] Daniel J. Fremont, Tommaso Dreossi, Shromona Ghosh, Xiangyu Yue, Alberto L. Sangiovanni-Vincentelli und Sanjit A. Seshia. „Szenisch: Eine Sprache zur Szenariospezifikation und Szenengenerierung.“ *Vorträge der 40. ACM SIGPLAN-Konferenz zum Thema Programmiersprachendesign und -implementierung (PLDI)*, 2019. DOI: 10.1145/3314221.3314633. URL: <https://people.eecs.berkeley.edu/~sseshia/pubs/b2hd-fremont-pldi19.html>[5] Chejian Xu, Wenhao Ding, Weijie Lyu, Zuxin Liu, Shuai Wang, Yihan He, Hanjiang Hu, Ding Zhao und Bo Li. „SafeBench: Eine Benchmarking-Plattform zur Sicherheitsbewertung autonomer Fahrzeuge.“ *Advances in Neural Information Processing Systems 35 (NeurIPS 2022) Datasets and Benchmarks Track*, 2022. URL: <https://proceedings.neurips.cc/paper_files/paper/2022/hash/a48ad12d588c597f4725a8b84af647b5-Abstract-Datasets_and_Benchmarks.html>

[6] ASAM. „ASAM OpenSCENARIO DSL: Schlüsselterminologie und konzeptioneller Überblick.“ URL: <https://publications.pages.asam.net/standards/ASAM_OpenSCENARIO/ASAM_OpenSCENARIO_DSL/latest/conceptual-overview/key_terms.html>

[7] ASAM. „ASAM OpenODD: Modell zur ASAM OpenSCENARIO DSL-Zuordnungsreferenz.“ URL: <https://publications.pages.asam.net/standards/ASAM_OpenODD/ASAM_OpenODD/latest/pecification/09_openscenario_dsl/09_01_overview.html>[8] Shital Shah, Debadeepta Dey, Chris Lovett und Ashish Kapoor. „AirSim: Hochpräzise visuelle und physikalische Simulation für autonome Fahrzeuge.“ *Field and Service Robotics*, Springer Proceedings in Advanced Robotics, 2017; arXiv:1705.05065. URL: <https://arxiv.org/abs/1705.05065>

[9] Yunlong Song, Selim Naji, Elia Kaufmann, Antonio Loquercio und Davide Scaramuzza. „Flightmare: Ein flexibler Quadrocopter-Simulator.“ *Proceedings of the 4th Conference on Robot Learning (CoRL)*, PMLR 155, 2021. URL: <https://proceedings.mlr.press/v155/song21a.html>

[10] Justin Nakama, Ricky Parada, Joao P. Matos-Carvalho, Fabio Azevedo, Dario Pedro und Luis Campos. „Autonomer Umgebungsgenerator für UAV-basierte Simulation.“ *Applied Sciences*, 11(5):2185, 2021. DOI: 10.3390/app11052185. URL: <https://doi.org/10.3390/app11052185>[11] Yash Vardhan Pant, Max Z. Li, Alena Rodionova, Rhudii A. Quaye, Houssam Abbas, Megan S. Ryerson und Rahul Mangharam. „FADS: Ein Rahmen für die Sicherheit autonomer Drohnen unter Verwendung einer auf zeitlicher Logik basierenden Flugbahnplanung.“ *Transportation Research Part C: Emerging Technologies*, 130:103275, 2021. DOI: 10.1016/j.trc.2021.103275. URL: <https://doi.org/10.1016/j.trc.2021.103275>

[12] Keyu Chen, Yuheng Lei, Hao Cheng, Haoran Wu, Wenchao Sun und Sifa Zheng. „FREA: Machbarkeitsgesteuerte Generierung sicherheitskritischer Szenarien mit angemessener Adversarialität.“ arXiv:2406.02983, 2024. URL: <https://arxiv.org/abs/2406.02983>

[13] Wenhao Ding, Chejian Xu, Mansur Arief, Haohong Lin, Bo Li und Ding Zhao. „Eine Umfrage zur Erstellung sicherheitskritischer Fahrszenarien: Eine methodische Perspektive.“ arXiv:2202.02215, 2022. URL: <https://arxiv.org/abs/2202.02215>[14] CARLA-Team. „Digital Twin Tool: Prozedurale Generierung aus OpenStreetMap.“ Dokumentation zum CARLA-Simulator. URL: <https://carla.readthedocs.io/en/0.9.16/adv_digital_twin/>

[15] Shandong Expressway Group Co., Ltd. „‚Shandong Expressway Comprehensive Inspection Flight Service System‘ geht online.“ 2025. URL: <https://www.sdhsg.com/article/72553>

[16] Zhao Xiangmo, Zhao Yifei, Lu Nengchao et al. „Eine Überprüfung der Forschung zur Zuweisung wichtiger Ressourcen für Notfälle bei Verkehrsunfällen auf der Autobahn.“ *Transactions of Transportation Engineering*, 2024. DOI: 10.19818/j.cnki.1671-1637.2024.06.001. URL: <https://transport.chd.edu.cn/cn/article/doi/10.19818/j.cnki.1671-1637.2024.06.001>

[17] Jisheng Zhang, Limin Jia, Shuyun Niu, Fan Zhang, Lu Tong und Xuesong Zhou. „Ein raumzeitnetzwerkbasiertes Modellierungsframework für die dynamische unbemannte Luftfahrzeugführung in Anwendungen zur Überwachung von Verkehrsunfällen.“ *Sensors*, 15(6):13874-13898, 2015. DOI: 10.3390/s150613874. URL: <https://doi.org/10.3390/s150613874>[18] Tan Do-Duy, Long D. Nguyen, Trung Q. Duong, Saeed Khosravirad und Holger Claussen. „Gemeinsame Optimierung der Echtzeit-Bereitstellung und Ressourcenzuweisung für UAV-gestützte Katastrophen-Notfallkommunikation.“ *IEEE Journal on Selected Areas in Communications*, 39(11):3411-3424, 2021. DOI: 10.1109/JSAC.2021.3088662. URL: <https://doi.org/10.1109/JSAC.2021.3088662>

---

## Anhang: Dieser Ausführungsplan

### Schritt 1: Papier F Gesamtpositionierung einfrieren

- Positionspapier F als UAV-sicherheitskritisches Szenario-Engineering.
- Machen Sie deutlich, dass es sich nicht um ein Duplikat von Papier B/C/E handelt, sondern um eine Gruppe von Papieren zu experimenteller Infrastruktur und Szenariomethoden.
- Übernahme der Struktur von vier progressiven Arbeiten von F1 bis F4.

### Schritt 2: Führen Sie zuerst den F1-Benchmark durch

- Definieren Sie die Taxonomie des UAV-Szenarios.
- Entwerfen Sie das Schema „scenario.json“.
- 76 Millionen Erkundungsprotokolle organisiert.
- Statistikabdeckungslücken, Fehlermodi und ungültige Szenariorate.
- CovUAV-Bench v0.1 exportieren.

### Schritt 3: Weiterentwicklung des beschleunigten F2-Testalgorithmus- Implementieren Sie kontradiktorische Zufalls-/Gitter-/LHS-/BO-/CMA-ES-/RL-Baselines.
- Implementieren Sie Abdeckungsspeicher, Kritikalitätsbewertung und möglichen Kritikalitätsfilter.
- Vergleichen Sie die Fehlererkennungsrate, den Abdeckungsgewinn und den Beschleunigungsfaktor.
- Verwenden Sie einen ausgehaltenen Test, um den Trainingswert zu überprüfen.

### Schritt 4: Erweitern Sie die F3-Stadt auf die lokale Szene

- Greifen Sie auf OSM, Straßengrade, Bebauungsdichte und POIs zu.
- Wählen Sie wichtige Abschnitte der Schnellstraße Jinan/Qingdao/Shandong als Fallstudie aus.
- Ordnen Sie das ODD auf Stadtebene einer lokalen 50 x 50 x 50 m großen Testzelle zu.
- Festlegung von Indikatoren für die Abdeckung städtischer Funktionsgebiete.

### Schritt 5: Erweitern Sie die F4-Hochgeschwindigkeits-Notfallanwendung

- Nehmen Sie die Inspektion/Notfallreaktion der Shandong-Schnellstraße als Anwendungshintergrund.
- Entwerfen Sie Unfallszenarien, UAV-Aufklärung und kollaborative Einsatzprozesse für Bodenrettungsressourcen.
- Vergleichen Sie die heuristische und szenariobewusste Rolling-Optimierung nur am Boden, am nächsten zur Ressource, UAV-zuerst.
- Konzentrieren Sie sich auf die Meldung der First-View-Zeit, der Reaktionszeit, der Bearbeitungszeit und der Falschversandrate.

### Schritt 6: Einreichungsrhythmus

- Investieren Sie zuerst in F1/F2, um einen Benchmark + Methoden-Dual-Core zu bilden.
- F1 Wenn die Tools und Daten vollständig sind, erhält der T-ITS/ITSC/IROS-Benchmark Vorrang.
- F2 Wenn das Algorithmusergebnis stark ist, wird AAAI / ICRA / IROS Priorität eingeräumt.
- F3/F4 Warten Sie, bis das F1/F2-Werkzeug stabil ist, bevor Sie zu TR-C / T-ITS wechseln.

### Schritt 7: Aufgaben für die letzte Woche-Schreiben Sie eine formelle experimentelle Aufgabe für F1.
- Szenendimensionen einfrieren: Hindernisse, räumliche Strukturen, Umgebungsstörungen, Aufgabentypen, Risikobezeichnungen.
- Beprobung von 10.000 bis 50.000 Explorationsprotokollen aus 76 Millionen Explorationsprotokollen für eine vorläufige Abdeckungsanalyse.
- Zeichnen Sie die erste Version des Szenentaxonomiediagramms und der Abdeckungs-Heatmap.