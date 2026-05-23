---
title: "Paper B Planning v1: Dreischichtige hierarchische Planung von Hunderten von UAVs für TR-C"
description: "Untersuchen Sie, ob Papier B besser für TR Teil C geeignet ist, und planen Sie den Hintergrund, die zugehörigen Methoden, die Problemdefinition, die Algorithmusroute, die experimentellen Daten, die erwarteten Schlussfolgerungen, die Innovationspunkte und den Werbeplan."
pubDate: 2026-05-19
updatedDate: 2026-05-23
tags: ["Papier B", "TR-C", "T-ITS", "UAV", "UAM", "hierarchische Planung", "Warteschlangentheorie", "Ljapunow", "multimodaler Transport"]
category: Tech
---

# Paper B Planning v1: Dreischichtige hierarchische Planung von Hunderten von UAVs für TR-C

> Fazit: **Papier B eignet sich besser für die Hauptinvestition Transportation Research Teil C: Emerging Technologies, IEEE T-ITS als Alternative oder Änderung der Investitionsrichtung. **
> Der Hauptgrund ist nicht, dass TR-C „besser“ ist, sondern dass der Kern des Problems von Papier B im Betrieb des Transportsystems liegt: Unter den Einschränkungen der dynamischen Nachfrage, der begrenzten Vertiport-/Lade-/Korridorkapazität und des multimodalen Transports soll eine UAV-Flotte mit hundert Ebenen städtische Logistik-/Notfallaufgaben stabil, effizient und sicher erfüllen.

---

## 1. Hintergrund und Vorlageurteil

Die Bedenken von Papier B lassen sich wie folgt zusammenfassen:

> Wie kann in einem städtischen Wirtschaftsszenario in geringer Höhe eine UAV-Flotte mit 100 Ebenen geplant werden, um die Stabilität der Aufgabenwarteschlange langfristig aufrechtzuerhalten und Verzögerungen, Energieverbrauch, Überlastung des Luftraums und Betriebskosten unter den Bedingungen dynamischer Befehle, begrenzter Start- und Landepunkte, begrenzter Laderessourcen, Kapazitätsbeschränkungen für Korridore in geringer Höhe und Koordinierung des Bodentransports zu minimieren?

Dabei handelt es sich weder um eine Pfadplanung für eine einzelne Maschine noch einfach um die Kollisionsvermeidung mehrerer Agenten. Das eigentliche Forschungsobjekt ist **Transportdienstleistungssystem**:

- Nachfrageseite: Bestellungen treffen zufällig ein und es gibt Unterschiede bei Fristen, Prioritäten, Start- und Endpunkten sowie Fracht-/Notfallarten.
- Angebotsseite: UAV-Leistung, Last, aktueller Standort, Wartungsstatus und verfügbarer Luftraum ändern sich im Laufe der Zeit.
- Infrastrukturseite: begrenzte Kapazität für Vertiports, Ladestationen, Korridore in geringer Höhe, Übergabepunkte und Bodenfahrzeuge.
- Systemseitig: Es ist notwendig, gleichzeitig Durchsatz, Verzögerung, Warteschlangenrückstand, Ressourcennutzung, Energie und Sicherheit zu optimieren.Daher ist es sinnvoller, hauptsächlich in TR-C zu investieren. Der offizielle Umfang von TR-C betont deutlich die Auswirkungen neuer Technologien auf die Planung, das Design, den Betrieb, die Steuerung und die Wartung von Transportsystemen und erklärt, dass der intellektuelle Kern der Zeitschrift auf der Transportseite und nicht auf der einzelnen Technologie selbst liegt; Es begrüßt auch den Integrationsansatz von Operations Research, Kontrollsystemen, komplexen Netzwerken, Informatik und KI und legt besonderen Wert auf multimodalen/intermodalen Transport, On-Demand-Transport, ITS, Logistik, Luftfahrt, Ressourcenmanagement und offene Datensätze [1]. Diese Schlüsselwörter decken fast genau Artikel B ab.

Alternativ steht auch T-ITS zur Verfügung. Der Umfang von T-ITS umfasst Sensorik, Kommunikation, Steuerung, Planung, Design, Implementierung, KI, formale Methoden, Multiagentensysteme und multimodalen Transport in modernen Transportsystemen [2]. T-ITS erfordert jedoch eher die Variante einer „Implementierung intelligenter Transportsystemtechnologie“, wie z. B. Kommunikation, Sensorik, Steuerung, Bereitstellungsarchitektur oder einen geschlossenen Regelkreis eines intelligenten Systems. Wenn Paper B letztendlich den Schwerpunkt auf Lyapunov-regulierte Online-Terminplanung, GNN/MARL-Steuerung und Echtzeit-Systemimplementierung legt, können Sie sich an T-ITS wenden; Wenn es um Transportkapazität, Warteschlangenstabilität, Infrastrukturengpässe und den Wert multimodaler Logistiksysteme geht, sollten Sie für TR-C stimmen.

**Aktuelle Empfehlungen: TR-C zuerst, T-ITS-Backup. **

### 1.1 22.05.2026 Schreibkalibrierung: Papier B muss ein Papier zum Betrieb eines Transportsystems sein

Aufsatz B eignet sich am besten, um die Logik „Bei Verkehrsjournalen geht es nicht nur um Algorithmen“ zu verinnerlichen. Es kann nicht wie folgt geschrieben werden: „Wir schlagen einen neuen UAV-Planungsalgorithmus vor“, sondern sollte wie folgt geschrieben werden:> Wie kann eine hundertstufige UAV-Flotte für städtische Tieffluglogistik/Notfalldienste die Flottenstabilität aufrechterhalten, Verzögerungen reduzieren, Sicherheitsrisiken kontrollieren und Systemengpässe bei dynamischer Nachfrage, begrenzter Vertiport-/Lade-/Korridorkapazität und multimodalen Transportbeschränkungen identifizieren?

Diese Hauptzeile bestimmt, wie der Volltext geschrieben wird:

| Modul | kann nicht einfach als | geschrieben werden Die TR-C-Version sollte als | geschrieben werden
|------|------------|----|
| Hintergrund | UAV-Planung ist schwierig | Betriebskontrollprobleme für Transportdienste in geringer Höhe bei Spitzennachfrage, Infrastrukturkapazität und Sicherheitsisolationsbeschränkungen |
| Lücke | Bestehende Algorithmen sind nicht gut genug | Vorhandene Forschungsarbeiten befassen sich mit Einzelpunktproblemen des Routings/Netzwerkdesigns/der Ressourcenzuweisung und es fehlen geschlossene Regelkreise und Stabilitätsgarantien für Hunderte von Online-Systemen auf Rack-Ebene |
| Methode | Neuer hierarchischer Planungsalgorithmus | Einheitlicher Betriebskontrollrahmen für Makro-Nachfragewarteschlange, Meso-Luftraum/Start- und Lande-/Laderessourcen sowie Mikroenergie-/Sicherheitsbeschränkungen |
| Experiment | Belohnung oder höhere Erfolgsquote | Systematische Verbesserung von Verzögerung, Durchsatz, Warteschlangenrückstand, Fristverletzung, Ressourcenauslastung, Energie, Konfliktrisiko |
| Fazit | Die Methode ist besser als die Basislinie | Bei welcher Nachfrageintensität ist das System stabil, welche Ressource wird zuerst zum Engpass, wann ist ein multimodaler Fallback erforderlich und ob eine strategische Strombegrenzung erforderlich ist |

Daher beantwortet das Experiment von Paper B die Systemfrage und beweist nicht nur, dass die Modellbewertung höher ist:- **Kapazitätsgrenze**: Wann tritt das System bei niedrigem/mittlerem/Spitzen-/Schockbedarf in die instabile Zone ein?
- **Zuschreibung von Engpässen**: Ist die Verzögerung hauptsächlich auf Vertiport, Aufladung, Überlastung des Korridors oder eine Neupositionierung der Flotte zurückzuführen?
- **Multimodaler Wert**: Wann reicht UAV allein nicht aus? Wie reduziert Ground Fallback Fristverletzungen?
- **Theoretische Entsprechung**: Kann der Rückstand/Kosten-Kompromiss von Lyapunov-Drift plus Strafe in Experimenten beobachtet werden?
- **Management-Inspiration**: Wenn Sie nur eine Ressource hinzufügen, sollten Sie dann UAV, Ladestation, Vertiport-Steckplatz oder Korridorkapazität hinzufügen?

### 1.2 23.05.2026 Organisation: Mindesteinreichungsversion und -grenzen

Die mindestens einreichbare Version von Papier B sollte ein **TR-C-Transportsystem-Betriebspapier** sein, keine Mischung aus „Scheduler + MARL + Luftraumsimulation + Demonstration einer Plattform in geringer Höhe“. Die erste Ausgabe muss das Systemproblem lösen: Wie begrenzte UAV-, Vertiport-, Ladepad- und Korridorkapazitäten gemeinsam Verzögerung, Durchsatz, Warteschlangenstabilität und Servicezuverlässigkeit bei dynamischer Nachfrage bestimmen.| Muss ausgefüllt werden | Auf erweiterte Version verschoben |
|----------|--------------|
| Synthetic City UAM-Warteschlangen-Benchmark | Hochpräzise visuelle Simulation auf AirSim/UE-Ebene |
| Lyapunov-regulierter Online-Terminplaner | Echter Flugeinsatz oder Hardware-Closed-Loop |
| 20/50/100/200 UAV-Skalierbarkeit | LLM-Dispatcher als Hauptalgorithmus |
| Vertiport-/Lade-/Korridor-Engpassanalyse | Vollständige Kommunikationsprotokoll- und Link-Layer-Simulation |
| FCFS, gierig, rollendes MILP, ALNS, Gegendruck, MARL/GNN-Basislinien | Evaluierung der Multi-City-Politik und vollständige Wirtschaftsanalyse |
| Stabilität, Kosten-Verzögerungs-Kompromiss, Fristverletzung, Laufzeit | Zugang zum echten Geschäftsauftragssystem |

Die erste Version des experimentellen Pakets empfiehlt das Einfrieren auf fünf Ergebnisse:1. **Benchmark-Generator**: Stadtgebiete, Vertiport, Ladestation, Korridor, Nachfragefluss, Frist, Bodenrückfall und Schockbedarf generieren.
2. **Systemmodell**: Geben Sie reproduzierbare experimentelle Protokolle der Bedarfswarteschlange, der Vertiport-Warteschlange, der Ladewarteschlange, der virtuellen Korridorwarteschlange und der virtuellen Fristenwarteschlange aus.
3. **H-LyraUAV-Kern**: implementiert Drift-plus-Strafe-Entscheidungsfindung. Das Lernmodul liefert lediglich Bedarfs-/Servicezeit-/Risikoschätzungen und dient nicht als Stabilitätsquelle.
4. **Baseline-Suite**: Jede Baseline verwendet den gleichen Satz an Bedarfsspuren, Kapazitätseinstellungen, UAV-Flotte und zufälligen Seeds.
5. **TR-C-Ergebnispaket**: Die Haupttabelle meldet Verzögerung, 95. Verzögerung, Fristverletzung, Durchsatz, Rückstand, Ressourcennutzung, Energie, Konflikt-Proxy, Laufzeit; In der Anhangtabelle werden die Engpasszuordnung und -empfindlichkeit aufgeführt.

Diese Grenze kann die Systemgeschichte von Papier B mit seinen experimentellen Verantwortlichkeiten in Einklang bringen: Zunächst wird bewiesen, dass „Logistik-/Notfalldienstsysteme in geringer Höhe auf hundert Ebenen stabil online funktionieren können“, bevor über echte Karten, echte Befehle oder komplexere Agenten gesprochen wird.

---

## 2. Aktuelle Methode Genealogie

In Papier B müssen relevante Methoden in den Stammbaum der Verkehrstechnik eingebracht werden, anstatt nur nach dem Vorbild von UAV/MARL zu reden.| Methodenzeile | Repräsentative Methode | Inspiration für Papier B | Einschränkungen |
|--------|----------|-----|------|
| Traditionelles ODER | MILP, Zeit-Raum-Netzwerk, Netzwerkdesign, ALNS, Rolling Horizon | Geeignet zum Ausdrücken von Kapazitäts-, Zeitfenster-, Pfad-, Synchronisations- und Infrastrukturbeschränkungen | Eine groß angelegte Online-Planung ist in Echtzeit schwer zu lösen |
| UAM / UTM | Vertiport-Planung, kapazitätsbeschränkte Planung, Konflikterkennung und -lösung | Bietet Perspektiven für Kapazität, Luftraumkonflikte und Korridormanagement | Die meisten sind Single-Layer-, Single-Mode- oder kleine und mittlere |
| Multimodale Logistik | LKW-Drohne, UAV-UGV, Boden-Luft-Transfer, Mitfahrgelegenheit-UAM | Beweisen Sie, dass UAV in das städtische Transportsystem eingebettet werden muss, anstatt isoliert zu fliegen | Hauptsächlich Offline-Routing/Netzwerkdesign, mangelnde Online-Warteschlangenstabilität |
| Lernplanung | MARL, GNN, sicheres Lernen, Nachfragevorhersage | Skalierbar auf Hunderte von Racks, geeignet für dynamische Anforderungen und hochdimensionale Zustände | Mangel an erklärbarer Stabilitätsgarantie, Prüfer werden die Sicherheit in Frage stellen |
| Warteschlangentheorie und Lyapunov | offenes Warteschlangennetzwerk, Gegendruck, Drift plus Strafe | Kann die Stabilität des Rückstands und den Kompromiss zwischen Kosten und Verzögerung nachweisen | Muss mit der tatsächlichen UAV-Energie, Kapazität und Pfadbeschränkungen kombiniert werden |Vorhandene TR-C-Papiere haben viele „Single-Point-Fähigkeiten“ abgedeckt: UAV-Paketverkehrsmanagement und Ressourcenzuweisung in geringer Höhe [3], passagierzentrierte UAM-Fairness und Betriebseffizienz [4], Netzwerkdesign für Relais-Ladestationen [5], zuverlässiges Synchronisationsrouting für LKW-Drohnen [6], UAV-UGV-Netzwerkdesign für die Mehrfahrtenzustellung [7] und dynamische UAM-Fahrgemeinschaftsplanung [8]. Die Chance für Paper B liegt in der Konvergenz dieser Funktionen in einem **hierarchischen Online-Planungssystem mit Hunderten Regalebenen**.

---

## 3. Aktuelle Arbeiten und zitierfähige Literatur

### 3.1 Veranstaltungsort und Rahmenliteratur| Nummer | Literatur/Quelle | Kerninformationen | Positionierungsrolle für Papier B |
|------|-----------|----------|------------------------|
| [1] | Offizielle Ziele und Umfang von TR-C | verkehrsseitiger intellektueller Kern; Schwerpunkte: Betrieb, Steuerung, Multimodalität, Logistik, Luftfahrt, offene Datensätze | Unterstützen Sie die Hauptinvestition TR-C |
| [2] | Offizieller Geltungsbereich von IEEE T-ITS | ITS-Erkennung, Kommunikation, Steuerung, Planung, KI, Multiagentensysteme | Unterstützen Sie T-ITS-Alternativen |
| [15] | Maschinelles Lernen für UAV-gestützte ITS, T-ITS 2024 | UAV kann zur Verkehrsüberwachung, Notfallreaktion und Infrastrukturinspektion von ITS dienen | Unterstützen Sie T-ITS alternatives Framing |
| [18] | 4D-Flugbahnplanung für UAV-Teams, T-ITS 2024 | UAV-Teams wurden in ITS/T-ITS | veröffentlicht Erklären Sie, dass in T-ITS investiert werden kann, es aber ein intelligenteres System braucht |

### 3.2 TR-C/UAM/UAV-Planungspapier| Nummer | Literatur | Methoden | Inspiration für Papier B |
|------|------|------|----|
| [3] | Li, Hansen & Zou, TR-C 2022 | Verkehrsmanagement, Pfadkonflikt, Ressourcenzuweisung, VCG-Mechanismus der UAV-Paketzustellung in geringer Höhe | Es wird direkt darauf hingewiesen, dass die Zuteilung von Luftraumressourcen in geringer Höhe ein rechtliches Thema von TR-C | ist
| [4] | Bennaceur, Delmas & Hamadi, TR-C 2022 | passagierzentriertes UAM, Fairness und betriebliche Effizienz | Unterstützung von Servicequalität, Fairness, Passagier-/Frachtkennzahlen |
| [5] | Pinto & Lagorio, TR-C 2022 | Drohnennetzwerkdesign mit zwischengeschalteten Ladestationen | Unterstützung der Ladeinfrastruktur bei der Formulierung |
| [6] | Xing, Guo & Tong, TR-C 2024 | Zuverlässiges Routing per Lkw-Drohne mit dynamischer Synchronisierung | Unterstützung multimodaler Synchronisierung und ungewisser Reisezeit |
| [7] | Zhou, Zeng & Yang, TR-C 2025 | UAV-UGV-Multi-Trip-Delivery-Netzwerkdesign mit Release-Zeiten | Unterstützung des UAV + UGV-Multi-Trip-Liefernetzwerks |
| [8] | Li, Zhang, Xiao & Li, TR-C 2025 | Dynamische Planung von UAM-Mitfahrgelegenheiten und multimodale Mobilität auf Abruf | Unterstützung der multimodalen Luft-Boden-Dienstarchitektur |
| [9] | Wei, Nilsson & Coogan, arXiv 2021 | Kapazitätsbeschränkte UAM-Planung mith unsichere Reisezeit und begrenzte Landekapazität | Unterstützung der kapazitätsbeschränkten Planungsformulierung |
| [10] | Murthy et al., EPTCS/arXiv 2022 | sicheres Lernen für die UAM-Planung mit harten/weichen Fristen | Unterstützen Sie eine sichere Online-Planungsbasis |
| [11] | NASA-Vertiport-FCFS-Planung, 2020 | Vertiport-Kapazität und Durchsatz unter FCFS | als FCFS und Warteschlangenkapazitätsbasislinie |
| [16] | Liu, Liu & Huang, arXiv 2024 | Echtzeit-UAV-Lieferplanungs-Management-Middleware | Unterstützung eines echten Ausführungssystems und der Zusammenarbeit von UAV/AGV/Bodenpersonal |### 3.3 Warteschlangentheorie, Lyapunov und die Grundlagen der Systemstabilität

| Nummer | Literatur | Kernbeitrag | Beitrag zu Papier B |
|------|------|----------|-----|
| [12] | Grippa et al., Autonome Roboter 2019 | Zuweisung und Dimensionierung von Drohnenlieferungsaufträgen; Verwenden Sie die Warteschlangentheorie, um Stabilität und Workload-Richtlinien zu analysieren Unterstützung des UAV-Lieferwarteschlangenmodells |
| [13] | Neely, 2010 | stochastische Netzwerkoptimierung und Lyapunov-Drift-plus-Strafe | Unterstützen Sie $O(1/V)$-Kosten und $O(V)$-Backlog-Kompromiss |
| [14] | Tassiulas & Ephremides, IEEE TAC 1992 | eingeschränkte Warteschlangensysteme und durchsatzoptimale Planung | Unterstützung der theoretischen Tradition des Gegendrucks / der Stabilität |
| [17] | Vertiport-Platzierung mit Fahrzeuggröße und Warteschlangen, 2023 | Open-Network-Warteschlangen für Vertiport-Infrastruktur und Service-Tarife | Unterstützt die Modellierung von Vertiport-Warteschlangen/Ladewarteschlangen |

**Literatururteil:** Die vorhandene Literatur hat vollständig bewiesen, dass „UAV/UAM + Transportsysteme + Planung + multimodal + Warteschlangen“ ein legitimes Thema für TR-C/T-ITS ist. Papier B kann nicht mehr als „MARL-Planungsalgorithmus für Hunderte von UAVs“ geschrieben werden, sondern muss als „Betriebssteuerung und Stabilitätsgarantie für Verkehrssysteme in geringer Höhe“ geschrieben werden.

---

## 4. Aktuelles ProblemEs gibt vier Hauptlücken, die die bestehende Arbeit hinterlässt.

1. **Fehlen eines geschlossenen Regelkreises für die Online-Planung mit drei Ebenen auf Hunderten von Regalebenen. **
   TR-C verfügt bereits über Ressourcenzuweisung für den Luftraum in geringer Höhe, Routing von LKW-Drohnen, UAM-Mitfahrgelegenheit und UAV-UGV-Netzwerkdesign [3,6,7,8], aber die meisten dieser Arbeiten befassen sich mit einer bestimmten Ebene von Routing, Netzwerkdesign, Ressourcenzuteilung oder Mitfahrgelegenheit und es fehlt ein einheitliches Online-Framework von der Makro-Nachfragewarteschlange über Meso-Korridor-/Vertiport-Ressourcen bis hin zu Mikro-UAV-Energie/-Sicherheit.

2. **Mangelnde Warteschlangenstabilität/Servicegarantie. **
   Gelernte Planung und Heuristiken können die empirische Leistung verbessern, aber TR-C-Prüfer stellen die Funktionsfähigkeit des Systems in Frage, wenn sie nicht berücksichtigen können, ob Warteschlangen bei Spitzenlast stabil sind. Die Lyapunov-Optimierung von Neely und die eingeschränkte Warteschlangenplanung von Tassiulas-Ephremides liefern theoretische Grundlagen [13,14], wurden jedoch nicht systematisch für die multimodale Planung von Hunderten von UAVs in geringer Höhe eingesetzt.

3. **Mangelnde UAV-Flottenkontrolle aus Sicht des multimodalen Transports. **
   Papiere zu LKW-Drohnen, UAV-UGV und Mitfahrgelegenheiten-UAM haben bewiesen, dass die Boden-Luft-Integration die Hauptrichtung ist [6, 7, 8], aber der Großteil der vorhandenen Forschung befasst sich mit der Offline-Routen-/Netzwerkgestaltung. Papier B sollte den Bodenmodus als Online-Fallback und Kapazitätspuffer behandeln: Wenn der Korridor in geringer Höhe oder die Ladewarteschlange überlastet ist, kann die Aufgabe auf das UGV/den LKW/den Bodenkurier übertragen werden.4. **Fehlender experimenteller Benchmark. **
   Der TR-C-Bereich legt besonderen Wert auf offene Wissenschaft und große Datensätze [1]. Wenn Papier B nur eine interne Simulation durchführt und kein synthetisches Benchmark-Schema, OD-Nachfrage, Korridorkapazität und reproduzierbare Samen veröffentlicht, wird dies die Überzeugungskraft schwächen.

---

## 5. Unser Ansatz: H-LyraUAV

Der Methodenname wird vorläufig festgelegt:

**H-LyraUAV: Hierarchische Lyapunov-regulierte UAV-Planung für multimodale städtische Logistik**

Dabei steht H für hierarchisch und Lyra für Lyapunov-regulierte Weiterleitung und Zuweisung.

### 5.1 Dreischichtige Architektur

```text
Dynamic urban logistics / emergency demand
        ↓
Macro layer: regional demand queues + fleet repositioning
        ↓
Meso layer: corridor / vertiport / charging resource scheduling
        ↓
Micro layer: UAV energy, safety separation, local conflict avoidance
        ↓
Multimodal execution: UAV-only / ground-only / UAV-ground mixed mode
```

| Hierarchie | Zeitskala | Entscheidung | Kernzustand | Ausgabe |
|------|----------|------|----------|------|
| Makroebene | 1-5 Minuten | Aufgabenpartitionierung, UAV-Neupositionierung, Modusaufteilung | Regionale Nachfragewarteschlange, Stromverteilung, OD-Bedarfsprognose | Regionales Versandziel |
| Mesoschicht | 5-30 s | Vertiport-Slot, Korridorroute, Ladeslot | Start- und Landewarteschlange, Korridorüberlastung, Ladewarteschlange | ausführbarer Zeitplan |
| Mikroskopische Schicht | 0,1-5 s | Geschwindigkeit, Höhe, örtliche Vermeidung, Notrückkehr | Benachbarte UAVs, Hindernisse, verbleibende Leistung | Sichere Flugbahnkorrektur |

### 5.2 Kernmechanismus

Der Schlüssel zu H-LyraUAV liegt nicht in der „End-to-End-Planung mit einem großen Modell“, sondern darin, das Lernmodul auf Vorhersage und Schichtung zu beschränken und Stabilität auf der Lyapunov-Warteschlangensteuerung aufzubauen:- **Warteschlangenmodell**: Bedarf, Vertiport, Laden und Korridor werden durch reale oder virtuelle Warteschlangen dargestellt.
- **Lyapunov-Drift-plus-Strafe**: Wählen Sie in jedem Zeitfenster Zuweisung/Modus/Route/Laden aus, um die gewichtete Summe aus Warteschlangendrift und Betriebskosten zu minimieren.
- **Lerngestützte Vorhersage**: Das GNN/zeitliche Modell sagt den zukünftigen OD-Bedarf, die Servicezeit, das Korridorrisiko und die Bodenreisezeit voraus, wird jedoch nicht als Quelle für den Stabilitätsnachweis verwendet.
- **Multimodaler Fallback**: Wenn nur UAV zu einer Warteschlangenexplosion oder einem erhöhten Terminrisiko führt, aktiviert das System automatisch den UGV-/LKW-/Bodenkurier- oder gemischten Modus.

---

## 6. Problemformulierung

### 6.1 Sammlungen und Zustände

Der UAV-Satz sei $\mathcal{U}$, der dynamische Missionssatz sei $\mathcal{R}(t)$, der Vertiport-Satz sei $\mathcal{V}$, der Tiefflugkorridor-Satz sei $\mathcal{E}$ und der Bodentransportmodus-Satz sei $\mathcal{G}$.

Der Zustand jedes UAV $u\in\mathcal{U}$ zum Zeitpunkt $t$ ist:

$$
s_u(t)=(l_u(t), b_u(t), a_u(t), \kappa_u(t)),
$$

Dabei ist $l_u(t)$ die Position, $b_u(t)$ die Leistung, $a_u(t)$ der verfügbare Status und $\kappa_u(t)$ die Last-/Aufgabenkapazität.

Jede Aufgabe $r\in\mathcal{R}(t)$ enthält:

$$
r=(o_r,d_r,\omega_r,\delta_r,\pi_r,\eta_r),
$$

Darunter ist $o_r,d_r$ der Start- und Endpunkt, $\omega_r$ ist der Fracht-/Passagier-/Notfalltyp, $\delta_r$ ist die Frist, $\pi_r$ ist die Priorität und $\eta_r$ ist die Menge der akzeptablen Transportarten.### 6.2 Warteschlangendefinition

Das System verwaltet die folgenden realen und virtuellen Warteschlangen:

| Warteschlange | Bedeutung |
|------|------|
| $Q_i(t)$ | Nicht bediente Nachfragewarteschlange für Bereich $i$ |
| $B_v(t)$ | Start-/Wartewarteschlange von Vertiport $v$ |
| $C_v(t)$ | Ladewarteschlange von Vertiport $v$ |
| $Z_e(t)$ | Überlastungs-/Sicherheitsintervall der virtuellen Warteschlange des Korridors $e$ |
| $D_i(t)$ | Fristverletzung virtuelle Warteschlange im Bereich $i$ |

Die regionale Nachfragewarteschlange kann beispielsweise wie folgt geschrieben werden:

$$
Q_i(t+1)=\max[Q_i(t)-\mu_i(t),0]+A_i(t),
$$

Dabei ist $A_i(t)$ die neu eingetroffene Nachfrage und $\mu_i(t)$ die Anzahl der Anforderungen, die den Dienst innerhalb des Zeitfensters abschließen.

### 6.3 Entscheidungsvariablen

In jedem Planungszyklus muss Folgendes entschieden werden:

| Entscheidungsfindung | Symbole | Bedeutung |
|------|------|------|
| Aufgabe | $x_{u,r}(t)$ | Ob das UAV $u$ die Aufgabe $r$ erfüllt |
| Moduswahl | $m_r(t)$ | Nur UAV, nur Boden oder gemischter Modus |
| Abfahrtszeit | $s_u(t)$ | Abfahrt/Abfahrt/Transferzeit |
| Route / Korridor | $p_u(t)$ | Wählen Sie einen Korridor in geringer Höhe oder einen Bodenpfad |
| Gebührenentscheidung | $c_u(t)$ | Ob aufgeladen werden soll und welcher Vertiport aufgeladen werden soll |

### 6.4 Optimierungsziele

Das langfristige Ziel besteht darin, die durchschnittlichen Systemkosten zu minimieren:

$$
\min_{\pi}
\limsup_{T\to\infty}
\frac{1}{T}\sum_{t=0}^{T-1}
\mathbb{E}\left[
\alpha_1 W(t)+
\alpha_2 E(t)+
\alpha_3 O(t)+
\alpha_4 S(t)+
\alpha_5 M(t)
\richtig],
$$Dabei ist $W(t)$ die Verzögerung, $E(t)$ der Energieverbrauch, $O(t)$ die Betriebskosten, $S(t)$ der Sicherheits-/Überlastungsnachteil und $M(t)$ der multimodale Transportnachteil.

### 6.5 Einschränkungen

Zu den Einschränkungen gehören:

- Warteschlangenstabilität: Alle realen Warteschlangen und kritischen virtuellen Warteschlangen müssen stark stabil sein.
- Batterie: $b_u(t)$ ist nicht niedriger als der sichere Rückgabeschwellenwert.
- Nutzlast: Das Gewicht der Missionsfracht darf die Kapazität des UAV oder Bodenfahrzeugs nicht überschreiten.
- Zeitfenster: Aufgaben mit hoher Priorität müssen die Frist einhalten oder in die virtuelle Fristwarteschlange gelangen.
- Vertiport-Kapazität: Die Pad-/Park-/Ladekapazität jedes Vertiports hat eine Obergrenze.
- Korridortrennung: Die zeitlichen und räumlichen Abstände von UAVs im selben Korridor entsprechen den Sicherheitsanforderungen.
- Machbarkeit eines multimodalen Transfers: Übergabezeitpunkt, Standort und Kapazität von UAV und UGV/LKW/Bodenkurier sind machbar.

### 6.6 Theoretische Ziele

Definieren Sie die Lyapunov-Funktion:

$$
L(\Theta(t)) =
\frac{1}{2}\left(
\sum_i Q_i(t)^2+
\sum_v B_v(t)^2+
\sum_v C_v(t)^2+
\sum_e Z_e(t)^2+
\sum_i D_i(t)^2
\richtig).
$$

Lösen Sie Drift plus Strafe für jedes Zeitfenster auf:

$$
\Delta(\Theta(t)) + V\cdot \mathbb{E}[Cost(t)\mid \Theta(t)].
$$

Fazit der Erwartungstheorie:

- H-LyraUAV kann die Warteschlange stabil halten, wenn die Ankunftsrate innerhalb der Systemkapazitätsregion liegt.
- Verglichen mit der optimalen stationären randomisierten Politik erreichen die langfristigen Durchschnittskosten ungefähr $O(1/V)$.
- Der durchschnittliche Rückstand beträgt $O(V)$, was einen interpretierbaren Kosten-Verzögerungs-Kompromiss darstellt [13,14].

---

## 7. Experimentelle Datenquelle### 7.1 Hauptexperiment: Benchmark zur Programmgenerierung

Das Hauptexperiment basiert nicht auf echten UAV-Flugdaten, sondern erstellt einen reproduzierbaren synthetischen UAM-Warteschlangen-Benchmark:

- Stadtplan: Raster „50x50“ bis „200x200“, einschließlich Gebäude, Flugverbotszonen, Korridore, Vertiports, Ladestationen.
- Nachfragefluss: Poisson-/inhomogene Poisson-/Burst-Nachfrage, unterstützt Morgenspitze, Abendspitze, Schocknachfrage.
-Aufgabenarten: Paketzustellung, medizinische Zustellung, Inspektion, Notfallversorgung.
- UAV-Flotte: 20 / 50 / 100 / 200 Einheiten, heterogene Batterien, Ladung, Geschwindigkeit, Ladezeit.
- Infrastruktur: 5 / 10 / 20 Vertiports, verschiedene Pad, Parkplätze, Ladekapazität.
- Multimodaler Modus: Nur UAV, Nur Boden, UAV-Boden-Mischmodus.

### 7.2 Echte erweiterte Daten

Um die Überzeugungskraft von TR-C zu erhöhen, können Experimente öffentliche Verkehrsdaten als Nachfrage-Proxy und Reisezeit im Bodenmodus nutzen:

| Datenquelle | Zweck |
|--------|------|
| OpenStreetMap | Straßennetz, POI, Bebauungsdichte, möglicher Vertiport/Übergabepunkt |
| NYC TLC Taxifahrtdaten | OD-Nachfrage-Proxy, Zeitraum-Nachfrageprofil |
| Chicago Taxi Trips / Divvy / Daten zur öffentlichen Mobilität | Stadtübergreifender Generalisierungsnachfrage-Proxy |
| SUMO | Fahrzeit des Bodenfahrzeugs, Staus, Kosten für den Bodenersatz |
| AirSim oder leichter UAV-Simulator | Ergänzende Überprüfung von Mikrosicherheit, Flugzeit und Energieverbrauch |Konferenzen wie AAAI können nur synthetische Benchmarks durchführen; Da TR-C jedoch die Qualität einer Fallstudie erfordert, wird empfohlen, mindestens einen Stadtfall durchzuführen: San Francisco, New York oder Chicago. Das UAV-Paketverkehrsmanagement in geringer Höhe von Li et al. anhand der Fallstudie San Francisco [3] kann als Ausrichtungsobjekt verwendet werden.

---

## 8. Experimenteller Aufbau und Vergleich

### 8.1 Grundlinien

| Grundlinie | Beschreibung | Zweck |
|----------|------|------|
| FCFS-Vertiport-Planung | Weisen Sie Start- und Landeressourcen in der Reihenfolge ihres Eintreffens zu [11] | Basislinie für den traditionellen Betrieb |
| Gieriges nächstes UAV | Das nächstgelegene verfügbare UAV, um die nächstgelegene Aufgabe aufzunehmen | Einfacher Online-Versand |
| MILP rollender Horizont | kleinräumige Walzoptimierung | kleinräumige Obergrenze |
| ALNS / heuristischer Versand | Siehe TR-C-Literatur zum multimodalen Routing [7,8] | Starke ODER-Heuristik |
| Nur-Warteschlangen-Gegendruck | Planung nur basierend auf der Warteschlangendifferenz | Theoretisch stabile Basislinie |
| MARL/GNN-Versand | Lernzuweisung, keine virtuellen Lyapunov-Warteschlangen | Lerngrundlage |
| H-LyraUAV voll | Dreischichtige Schichtung + Lyapunov + Lernvorhersage + multimodaler Fallback | Hauptmethode |

### 8.2 Metriken| Indikator | Bedeutung |
|------|------|
| Durchschnittliche Verzögerung | Durchschnittliche Verzögerung bei der Aufgabenerledigung |
| 95. Perzentilverzögerung | Long-Tail-Servicequalität |
| Quote der Fristüberschreitungen | Anteil der Überstunden |
| Durchsatz | Anzahl der pro Zeiteinheit erledigten Aufgaben |
| Warteschlangenrückstand | Nachfrage, Vertiport, Aufladung, Länge der Korridorwarteschlange |
| Warteschlangenstabilität | Ist der Rückstand über die Zeit begrenzt |
| Vertiport-Nutzung | Ressourcenauslastung bei Start und Landung/Stopp |
| Ladeauslastung | Laderessourcennutzungsrate |
| Luftraumkonfliktrate | Konfliktrate im Korridor-Sicherheitsintervall |
| Energie pro Lieferung | Energieverbrauch pro Bestellung |
| Erfolgreicher Boden-UAV-Transfer | Erfolgsquote bei multimodaler Übergabe |
| Laufzeit | Einstufige Planungszeit |
| Engpassbeitrag | Wie viel Verzögerung wird durch Vertiport/Laden/Korridor/Flottenneupositionierung verursacht |
| Kapazitätsmarge | Wie weit ist die aktuelle Nachfrageintensität von der Systeminstabilitätszone entfernt |
| Servicegerechtigkeit | Verzögerungslücke für verschiedene Bereiche/Prioritätsaufgaben, um die Optimierung nur beliebter Bereiche zu verhindern |

Es wird nicht empfohlen, dass die Haupttabelle der TR-C-Version nur die Rangfolge der Algorithmusleistung meldet, sondern auch eine separate **Systemdiagnosetabelle** bereitstellt: Meldung des durchschnittlichen Rückstands, 95 % Verzögerung, Fristverletzung, größere Engpässe und ob ein multimodaler Fallback unter verschiedenen Nachfragemultiplikatoren ausgelöst wird. Auf diese Weise kann die Schlussfolgerung gezogen werden, „wie das System funktioniert“ und nicht, „welches Modell stärker ist“.

### 8.3 Ablation| Ablation | Zweck |
|------|------|
| keine virtuellen Lyapunov-Warteschlangen | Beitrag der Stabilitätskomponente überprüfen |
| kein multimodaler Fallback | Überprüfen Sie den Wert des Bodenmodus als Kapazitätspuffer |
| keine hierarchische Zerlegung | Überprüfen Sie die Skalierbarkeit der dreistufigen Struktur |
| keine Nachfrageprognose | Überprüfen Sie den Beitrag des Lernvorhersagemoduls |
| keine Modellierung der Ladewarteschlange | Überprüfen Sie, ob der Ladeengpass explizit modelliert werden muss |
| 20/50/100/200 UAV-Skalierung | Überprüfung der Skalierbarkeit von Hunderten von Rack-Ebenen |

### 8.4 Szenendesign

Führen Sie mindestens vier Arten von Nachfrageszenarien durch:

1. **Geringe Nachfrage**: Das System ist leicht belastet, was bestätigt, dass H-LyraUAV keine Effizienzeinbußen erleidet.
2. **Spitzennachfrage**: Die Nachfrage liegt nahe am Kapazitätsbereich. Überprüfen Sie die Stabilität der Warteschlange.
3. **Schockanforderung**: plötzliche Notfallbefehle, virtuelle Warteschlange zur Fristüberprüfung und multimodaler Fallback.
4. **Infrastrukturengpass**: Ladestationen oder Vertiport-Pads werden bewusst reduziert, um die Fähigkeit zur Identifizierung von Ressourcenengpässen zu überprüfen.

Es wird zusätzlich empfohlen, zwei Arten von Verallgemeinerungen hinzuzufügen:

5. **Skalenverallgemeinerung**: Das Training oder die Parameteranpassung erfolgt bei 50/100 UAV und das Testen bei 200 UAV, was darauf hinweist, dass die hierarchische Struktur nicht nur für feste Skalierungen wirksam ist.
6. **Verallgemeinerung der Topologie**: Das von OSM abgeleitete Stadtdiagramm wurde anhand der Regel „grid_city“ gemessen, was darauf hinweist, dass die Schlussfolgerung kein zufälliges Ergebnis einer Spielzeugkarte ist.

---

## 9. Erwarteter Erfolg und Innovation

### 9.1 Erwarteter Erfolg

Dieser Abschnitt dient den Erwartungen vor der Registrierung und enthält keine tatsächlichen Versuchsergebnisse.1. **Aufrechterhaltung der Warteschlangenstabilität bei Spitzenlast. **
   Es wird erwartet, dass H-LyraUAV die Nachfragewarteschlange, die Vertiport-Warteschlange und die Ladewarteschlange bei Spitzenbedarf begrenzt hält, während Greedy/nur MARL anfälliger für die Anhäufung von Rückständen ist.

2. **Reduzieren Sie Verzögerungen und Fristverletzungen. **
   Im Vergleich zu FCFS und Greedy wird erwartet, dass H-LyraUAV die durchschnittliche Verzögerung, die 95. Perzentilverzögerung und die Fristverletzungsrate reduziert.

3. **Verbessern Sie die Echtzeitleistung und Skalierbarkeit. **
   Im Vergleich zum MILP-Rollhorizont wird erwartet, dass H-LyraUAV in 100/200-UAV-Szenarien eine Online-Entscheidungsfindung im Sekunden- oder Subsekundenbereich ermöglicht.

4. **Theoretische Erläuterung vorbehalten. **
   Im Vergleich zu reinem MARL/GNN liegt der Vorteil von H-LyraUAV nicht nur in der Erfahrungsbewertung, sondern kann auch den Kosten-Verzögerungs-Kompromiss und die Stabilitätsgrenze erklären.

5. **Zeigt den Systemwert des multimodalen Fallbacks. **
   Es wird erwartet, dass der UAV-Boden-Mischmodus verpasste Fristen und Warteschlangenrückstände bei Ladeengpässen oder Korridorüberlastungsszenarien reduziert.

### 9.2 Innovationspunkte

1. **Planungspapier für Verkehrssysteme in geringer Höhe basierend auf TR-C-Rahmen. **
   Aufsatz B behandelt UAVs nicht als isolierte Roboter, sondern als eine Flotte von Hunderten von UAVs als Teil des städtischen Transportdienstleistungssystems.

2. ** Dreischichtiges, warteschlangenstabiles, multimodales Planungsframework auf Hunderter-Regal-Ebene. **
   Vereinheitlichen Sie Makro-Aufgabenwarteschlangen, Meso-Infrastrukturressourcen und Mikrosicherheits-/Energiebeschränkungen in einem Online-Framework.

3. **Lyapunov-regulierter lerngestützter Versand. **
   Das Lernmodul wird zur Vorhersage von Nachfrage und Kosten verwendet, und das Lyapunov-Modul sorgt für Stabilität und Kosten-Verzögerungs-Kompromiss.4. **Multimodaler Mechanismus zur Pufferung der Transportkapazität. **
   Verwenden Sie UGV/LKW/Bodenkurier als Ausweichlösung bei Luftraum-/Ladeengpässen anstelle der angeschlossenen Basislinie.

5. **Offener synthetischer UAM-Warteschlangen-Benchmark. **
   Angleichung der Präferenzen von TR-C für offene Daten, reproduzierbare Benchmarks und Übertragbarkeit [1].

---

## 10. Referenzen

[1] Elsevier. „Verkehrsforschung Teil C: Neue Technologien: Ziele und Umfang.“ URL: <https://www.sciencedirect.com/journal/transportation-research-part-c-emerging-technologies>

[2] IEEE Intelligent Transportation Systems Society. „IEEE-Transaktionen auf intelligenten Transportsystemen (T-ITS): Umfang.“ URL: <https://ieee-itss.org/pub/t-its/>

[3] Ang Li, Mark Hansen und Bo Zou. „Verkehrsmanagement und Ressourcenzuweisung für die UAV-basierte Paketzustellung im städtischen Raum in geringer Höhe.“ *Transportation Research Part C: Emerging Technologies*, 143:103808, 2022. DOI: 10.1016/j.trc.2022.103808. URL: <https://www.sciencedirect.com/science/article/pii/S0968090X22002339>[4] Mehdi Bennaceur, Rémi Delmas und Youssef Hamadi. „Passagierzentrierte urbane Luftmobilität: Fairness-Kompromisse und betriebliche Effizienz.“ *Transportation Research Part C: Emerging Technologies*, 136:103519, 2022. DOI: 10.1016/j.trc.2021.103519. URL: <https://www.sciencedirect.com/science/article/pii/S0968090X21005015>

[5] Roberto Pinto und Alexandra Lagorio. „Drohnenbasierter Punkt-zu-Punkt-Liefernetzwerkentwurf mit Zwischenladestationen.“ *Transportation Research Part C: Emerging Technologies*, 135:103506, 2022. DOI: 10.1016/j.trc.2021.103506. URL: <https://doi.org/10.1016/j.trc.2021.103506>[6] Jiahao Xing, Tong Guo und Lu (Carol) Tong. „Zuverlässiges Truck-Drohnen-Routing mit dynamischer Synchronisation: Ein hochdimensionaler Netzwerkprogrammierungsansatz.“ *Transportation Research Part C: Emerging Technologies*, 165:104698, 2024. DOI: 10.1016/j.trc.2024.104698. URL: <https://www.sciencedirect.com/science/article/pii/S0968090X24002195>

[7] Bolong Zhou, Wenjia Zeng und Hai Yang. „Multi-Trip-UAV-UGV-Liefernetzwerkdesign mit Release-Zeiten.“ *Transportation Research Part C: Emerging Technologies*, 181:105389, 2025. DOI: 10.1016/j.trc.2025.105389. URL: <https://doi.org/10.1016/j.trc.2025.105389>

[8] Shanghan Li, Tengfei Zhang, Yiyong Xiao und Daqing Li. „On-Demand-Mitfahrgelegenheit basierend auf dynamischer Planung in der urbanen Luftmobilität.“ *Transportation Research Part C: Emerging Technologies*, 175:105111, 2025. DOI: 10.1016/j.trc.2025.105111. URL: <https://www.sciencedirect.com/science/article/pii/S0968090X25001159>[9] Qinshuang Wei, Gustav Nilsson und Samuel Coogan. „Kapazitätsbeschränkte städtische Flugmobilitätsplanung.“ arXiv:2107.02900, 2021. URL: <https://arxiv.org/abs/2107.02900>

[10] Surya Murthy, Natasha A. Neogi und Suda Bharadwaj. „Planung für urbane Luftmobilität durch sicheres Lernen.“ *Electronic Proceedings in Theoretical Computer Science*, 371:86-102, 2022; arXiv:2209.15457. DOI: 10.4204/EPTCS.371.7. URL: <https://arxiv.org/abs/2209.15457>

[11] Nelson M. Guerreiro, George E. Hagen, Jeffrey M. Maddalon und Ricky W. Butler. „Kapazität und Durchsatz städtischer Luftmobilitäts-Vertiports mit einem „Wer zuerst kommt, mahlt zuerst“-Vertiport-Planungsalgorithmus.“ NASA Technical Reports Server, AIAA Aviation 2020 Forum, 2020. URL: <https://ntrs.nasa.gov/citations/20205001421>[12] Pasquale Grippa, Doris A. Behrens, Friederike Wall und Christian Bettstetter. „Drohnenliefersysteme: Aufgabenstellung und Dimensionierung.“ *Autonomous Robots*, 43:261-274, 2019. DOI: 10.1007/s10514-018-9768-8. URL: <https://link.springer.com/article/10.1007/s10514-018-9768-8>

[13] Michael J. Neely. *Stochastische Netzwerkoptimierung mit Anwendung auf Kommunikations- und Warteschlangensysteme.* Synthesis Lectures on Communication Networks, Morgan & Claypool Publishers, 2010. DOI: 10.2200/S00271ED1V01Y201006CNT007. URL: <https://doi.org/10.2200/S00271ED1V01Y201006CNT007>

[14] Leandros Tassiulas und Anthony Ephremides. „Stabilitätseigenschaften eingeschränkter Warteschlangensysteme und Planungsrichtlinien für maximalen Durchsatz in Multihop-Funknetzwerken.“ *IEEE Transactions on Automatic Control*, 37(12):1936-1948, 1992. DOI: 10.1109/9.182479. URL: <https://doi.org/10.1109/9.182479>[15] Akbar Telikani, Arupa Sarkar, Bo Du und Jun Shen. „Maschinelles Lernen für UAV-gestützte ITS: Ein Rückblick mit vergleichender Studie.“ *IEEE Transactions on Intelligent Transportation Systems*, 25(11):15388-15406, 2024. DOI: 10.1109/TITS.2024.3422039. URL: <https://ieeexplore.ieee.org/document/10622103/>

[16] Han Liu, Tian Liu und Kai Huang. „Ein Echtzeitsystem zur Planung und Verwaltung der UAV-Lieferung in städtischen Gebieten.“ arXiv:2412.11590, 2024. URL: <https://arxiv.org/abs/2412.11590>

[17] Jose Escribano Macias, Carl Khalife, Joseph Slim und Panagiotis Angeloudis. „Ein integriertes Vertiport-Platzierungsmodell unter Berücksichtigung von Fahrzeuggröße und Warteschlangen: Eine Fallstudie in London.“ *Journal of Air Transport Management*, 113:102486, 2023. DOI: 10.1016/j.jairtraman.2023.102486. URL: <https://www.sciencedirect.com/science/article/pii/S0969699723001291>[18] Blanca Lopez Palomino, Javier Muñoz Mendi, Fernando Quevedo Vallejo, Concepción Alicia Monje Micharet, Luis Santiago Garrido Bullon und Luis Enrique Moreno Lorente. „4D-Flugbahnplanung basierend auf Fast Marching Square für UAV-Teams.“ *IEEE Transactions on Intelligent Transportation Systems*, 25(6):5703-5717, 2024. DOI: 10.1109/TITS.2023.3336008. URL: <https://doi.org/10.1109/TITS.2023.3336008>

---

## Anhang: Ausführungsplan

### Woche 1: Papierpositionierung und Problemformulierung einfrieren

- Klärung der Hauptinvestition TR-C und der Alternative T-ITS.
- Titel, erster Entwurf der Zusammenfassung, Kernfragen und dreischichtiges Architekturdiagramm einfrieren.
- Vervollständigen Sie die Definition von Mengen, Warteschlangen, Entscheidungen, Zielen und Einschränkungen für die Problemformulierung.

### Woche 2–3: Ergänzen Sie mehr als 25 Dokumente und die zugehörige Arbeitsmatrix

- Erweiterte TR-C / T-ITS / UAM / UAV-Lieferung / Warteschlangen / Lyapunov-Dokumentation.
- Ausgabebezogene Arbeitsmatrix: Problem, Methode, Maßstab, Modus, Einschränkung jeder Arbeit.
- Identifizieren Sie die Unterschiede zwischen Papier B und den einzelnen Auftragstypen.

### Wochen 4–6: Implementierung des synthetischen UAM-Warteschlangen-Benchmarks- Implementieren Sie Karte, Vertiport, Korridor, Ladestation und OD-Bedarfsgenerator.
- Unterstützt 20/50/100/200 UAV und niedrige/mittlere/Spitzen-/Schock-Anforderungen.
- Ausgabe von Manifest, Seed und Szenariokonfiguration, um die Reproduzierbarkeit sicherzustellen.

### Wochen 7–9: Grundlinien umsetzen

- FCFS-Vertiport-Planung.
- Gieriges nächstgelegenes UAV.
- MILP-Rollhorizont, kleinräumige Obergrenze.
-ALNS/heuristischer Versand.
- Gegendruck nur in der Warteschlange.
- MARL/GNN-Versand ohne Lyapunov.

### Wochen 10–12: Implementierung von H-LyraUAV und Ablation

- Implementieren Sie eine makrowarteschlangenbewusste Zuweisung.
- Meso-Korridor/Vertiport/Ladeplanung implementieren.
- Implementierung einer mikroskopisch kleinen Schnittstelle für Energie-/Sicherheitsbeschränkungen.
- Implementieren Sie keine Lyapunov-, keine multimodalen, keine Hierarchie-, keine Bedarfsvorhersage- und keine Ladewarteschlangenablationen.

### Wochen 13–15: Läuferexperiment

-Ausführung einer UAV-Skalierbarkeit von 20/50/100/200.
- Führen Sie Spitzen-/Schock-/Engpassszenarien durch.
- Geben Sie die Haupttabelle aus: Verzögerung, Fristverletzung, Durchsatz, Warteschlangenrückstand, Ressourcennutzung, Laufzeit.
- Schlüsseldiagramme für die Ausgabe: Warteschlangenverlauf, Kosten-Verzögerungs-Kompromiss, Skalierbarkeitskurve, multimodaler Fallback-Beitrag.

### Woche 16: Schreiben des ersten Entwurfs von TR-C- Die Einführung beginnt mit dem Problem des Transportbetriebs.
- Fügen Sie die dreischichtige Architektur, das Lyapunov-Theorem und den Algorithmus in die Methode ein.
- Experimente betonen die Systemleistung, die Ressourcennutzung und den offenen Benchmark.
- In der Diskussion geht es um die Wirtschaftlichkeit in geringer Höhe, die Vertiport-Kapazität, die Ladeinfrastruktur und die Auswirkungen auf die multimodale Logistik.

### T-ITS-Investitionsänderungsstrategie

Wenn der TR-C-Rahmen nicht stark genug ist oder die experimentellen Ergebnisse zeigen, dass der Algorithmus-/Kontrollbeitrag stärker ist als die Systemtransporterkenntnisse, wird die T-ITS-Version beibehalten:

– Zusammenfassung legt mehr Wert auf intelligente Transportsysteme, Online-Steuerung und KI-gestützte Planung.
- Einführung Fügen Sie Sensorik/Kommunikation/Echtzeitimplementierung hinzu.
- Experimente erhöhen Laufzeit, Kommunikationsverzögerung, verteilte Ausführung und Controller-Robustheit.
- Durch die Diskussion werden politische/betriebliche Auswirkungen verringert und die Bereitstellung intelligenter Systeme sowie die ITS-Integration verbessert.