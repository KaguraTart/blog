---
title: "Paper G Planning v1: Feinabstimmung der Route des LLM-Agenten und des Modells im Wolkenhirn des Verkehrs in geringer Höhe"
description: "Planen Sie, wie LLM trainiert oder verfeinert werden kann, um es zu einem überprüfbaren Agenten im Gehirn der Verkehrswolke in geringer Höhe zu machen, und erstellen Sie das erste AAAI/IJCAI-Konferenzpapier, nachfolgende Transportzeitschriften und eine allgemeine Route zur Transformation des verkörperten Agenten."
pubDate: 2026-05-20
updatedDate: 2026-05-23
tags: ["Papier G", "Gehirn der Verkehrswolke in geringer Höhe", "LLM-Agent", "Feinabstimmung des Modells", "Werkzeuggebrauch", "AAAI", "IJCAI", "UAV", "AGI"]
category: Tech
---

# Paper G Planning v1: LLM-Agent und Modell-Feinabstimmung der Route im Cloud-Brain für den Verkehr in geringer Höhe

> Gesamturteil: Bei dieser Route sollte es sich zunächst nicht um ein „großes Chat-Modell für den Verkehr in geringer Höhe“ handeln, sondern es sollte sich um einen **überprüfbaren LLM-Agenten im Cloud-Gehirn für den Verkehr in geringer Höhe** handeln.  
> Priorisieren Sie AAAI/IJCAI für den ersten Artikel: Versetzen Sie LLM in die Position „Aufgabenverständnis, Werkzeugaufruf, Planung und Reparatur, Überprüfung im geschlossenen Regelkreis und Planungserklärung“, anstatt sich direkt auf das Training eines groß angelegten Grundlagenmodells zu verpflichten.

---

## 1. Gesamturteil: Warum zuerst das Agent-Cloud-Gehirn aufbauen, anstatt das große Modell direkt zu trainieren?

Wenn Sie direkt schreiben „Feinabstimmung eines LLM für den Verkehr in geringer Höhe“, werden Konferenzrezensenten wahrscheinlich drei Fragen stellen:

1. **Welchen Beitrag leistet das Modell? **
   LoRA / SFT / DPO selbst ist bereits ein Standard-Trainingsprozess [14] [15] [16]. Es ist schwierig, die AAAI/IJCAI-Hauptkonferenz zu unterstützen, indem man einfach die Daten durch einen Verkehrskorpus in geringer Höhe ersetzt.

2. **Warum ist LLM notwendiger als bestehende Terminierungs-/Planungsmodelle? **
   Der Betrieb des Verkehrs in geringer Höhe umfasst Terminplanung, Streckenplanung, Risikobewertung, formale Verifizierung und Simulationsfeedback. Der Vorteil von LLM besteht nicht darin, diese Modelle zu ersetzen, sondern komplexe Aufgaben in aufrufbare Werkzeugketten zu zerlegen.

3. **Wie kann die Sicherheit gewährleistet werden? **
   Das Gehirn der Verkehrswolke in geringer Höhe ist ein sicherheitskritisches System. Die direkte Ausgabe von Kontrollaktionen aus LLM birgt das Risiko von Halluzinationen und Nichtüberprüfungen. Die erste Arbeit muss den Verifizierer, den Simulator und den Risikoschätzer in einen geschlossenen Kreislauf bringen.

Daher wird nicht empfohlen, den ersten Artikel von Paper G „LowAltitudeGPT“ zu nennen. Ein besserer erster Artikel wäre:

> **CloudBrain-Agent: Tool-erweiterte und verifizierungsgesteuerte LLM-Agenten für den Verkehrsbetrieb in geringer Höhe**

Sein Kernbeitrag ist nicht „das Modell wird intelligenter“, sondern:

- Aufbau einer Agenten-Entscheidungspipeline für das Cloud-Brain des Verkehrs in geringer Höhe;
- Lassen Sie LLM lernen, Fahrzeuge in geringer Höhe anzurufen.
- Verwenden Sie Verifizierer und Emulatoren, um Fehler zu korrigieren;
- Ausführbare, interpretierbare und überprüfbare Terminierungs-/Planungsentscheidungen ausgeben.Dies kommt der Idee von TrafficGPT nahe: TrafficGPT hat darauf hingewiesen, dass LLM selbst schwierig mit numerischen Verkehrsdaten und Simulationsinteraktionen umzugehen ist und daher mit Verkehrsfundamentmodellen kombiniert werden muss [1]. Der Unterschied in Paper G besteht darin, dass wir das Objekt vom Bodentransport auf den Transport in geringer Höhe erweitert haben und darüber hinaus den UAV-Status, Luftraumbeschränkungen, formelle Überprüfung und einen geschlossenen Sicherheitskreislauf hinzugefügt haben.

In einer umfassenderen Untersuchung der Verkehrsintelligenz wurde LLM als semantische Schnittstelle, Argumentationsmodul und Hilfskomponente für die Verkehrsentscheidung in ITS diskutiert [2] [3]; UrbanGPT und UniST veranschaulichen, dass die städtische räumlich-zeitliche Vorhersage zu einem räumlich-zeitlichen Grundlagenmodell übergeht [4] [5]. Papier G wiederholt diese Anweisungen nicht direkt, sondern kombiniert „städtische raumzeitliche Intelligenz + UAV-Betriebswerkzeuge + überprüfbare Agenten“ zu einem Cloud-Gehirn für den Verkehr in geringer Höhe.

### 1.1 22.05.2026 Schreibkalibrierung: G1 ist ein KI-Agentenpapier, und die Journalerweiterung erfordert eine vollständige Erzählung über das Transportsystem.

Aufsatz G könnte leicht als „große Modellgeschichte für den Verkehr in geringer Höhe“ abgeschrieben werden. Bei diesem Weg ist zwischen zwei Bewertungskriterien zu unterscheiden:

| Stufen | Ziele | Hauptbewertungslogik | Fehler, die man nicht machen kann |
|------|------|--------------|--------------|
| G1 AAAI/IJCAI | Überprüfbare LLM-Agent-Methode | Werkzeugeinsatz, Planung, Verifizierung, Benchmark, Reproduzierbarkeit | Die Klarheit der Methode für die Verkehrserzählung opfern oder den Agenten als Plattformanzeige schreiben |
| G2 T-ITS/T-IV | LLM-Feinabstimmung im Bereich Tieftransport | Domänendaten, Reproduzierbarkeit der Bereitstellung und Unterstützungsfunktionen für die Entscheidungsfindung im Datenverkehr | Nur allgemeines LoRA/SFT, keine Transportkette und Sicherheitsindikatoren |
| G3 AAMAS/T-ITS | Cloud-Gehirn-Zusammenarbeit mit mehreren Agenten | Multi-Rollen-Kollaboration, Kommunikation, Konfliktbearbeitung, Mensch-Maschine-Kollaboration | Multi-Agent besteht lediglich aus mehreren Eingabeaufforderungen, ohne Systemstatus- und Verantwortungsgrenzen |
| Zeitschrift erweiterte Version | Bedeutung des Verkehrssystembetriebs | Sicherheit, Effizienz, Kapazität, Verzögerung, Ressourcennutzung, Management-Inspiration | Melden Sie nur die Genauigkeit/den Erfolg des Tool-Aufrufs, beantworten Sie keine Verkehrsfragen |Daher besteht die Hauptlinie von G1 immer noch aus starken KI-Methoden: typisierte IR, Tool-Nutzung, Verifier-Reparatur und zustandsbehaftete Auswertung.
Allerdings müssen alle verkehrsbezogenen Tiefhöhenindikatoren von Anfang an beibehalten werden, um eine spätere Erweiterung auf T-ITS zu ermöglichen:

- Sicherheit: LoWC/NMAC-Proxy, Verletzung der Flugverbotszone, Verletzung der Batteriereserve.
- Effizienz: Verzögerung, zusätzliche Distanz, Energie, Durchsatz, Laufzeit.
- Betriebsmanagement: sichere Ablehnungsrate, menschliche Bestätigungsrate, Umgang mit mehrdeutigen Aufgaben.
- Robustheit: Kommunikationsverlust, Wetterstörungen, nicht kooperatives UAV, unsichtbare Stadt/Topologie.
- Systemaufklärung: Unter welchen Bedingungen muss der LLM-Agent zum deterministischen Löser oder menschlichen Supervisor wechseln?

### 1.2 23.05.2026 Anordnung: Die Reihenfolge der G-Routen

Paper G ist eine übergeordnete Roadmap, und was in naher Zukunft tatsächlich fertiggestellt werden soll, ist **G1 CloudBrain-Agent**. Derzeit besteht der schnellste und am besten einreichbare Weg darin, nicht zuerst ein großes vertikales Modell zu trainieren, sondern ein allgemeines starkes Modell + typisierte IR + Werkzeugkette + Verifizierer + Simulator-Feedback zu verwenden, um einen reproduzierbaren geschlossenen Regelkreis zu bilden. Das vertikale Modelltraining wird in G2 platziert und die von G1 generierten Werkzeugaufrufspuren, Reparaturspuren und Fehlerfälle werden als Daten verwendet.| Bühne | Ob das Modell trainiert werden soll | Empfohlenes Modell/Bereitstellung | Ziel |
|------|--------------|---------------|------|
| G1 jetzt | Nicht als Hauptfaktor für die Ausbildung | Lokales vLLM führt Qwen/DeepSeek aus, API-Modell übernimmt Lehrer/Obergrenze | Beweisen Sie, dass Agent-Tool-Aufrufe, Überprüfungsreparaturen und Benchmarking für Aufgaben in geringer Höhe wirksam sind |
| G2 weiter | LoRA / SFT / DPO | Feinabstimmung der Qwen-/Llama-/DeepSeek-Serie mit G1-Spuren | Bildung des kognitiven LowAltitudeGPT-Domänenmoduls |
| G3 später | Optionale Trajektoriendestillation mit mehreren Wirkstoffen | Multi-Rollen-Agent + gemeinsam genutzter Speicher + Verifizierer | Forschungsluftraumüberwachung, Planung, Risiko, Notfall, Mensch-Maschine-Zusammenarbeit |
| G4 langfristig | Multimodal / Weltmodell / VLA | Abhängig von Daten und Rechenleistung | Migration zur verkörperten Verkehrsintelligenz |

Empfehlungen zur Bereitstellungsstrategie lauten wie folgt:

- **Lokales Open-Source-Modell für das Hauptexperiment**: reproduzierbare, kontrollierbare Kosten, einfache Meldung von Latenz- und Hardwarebedingungen; Es wird empfohlen, vLLM / llama.cpp als Inferenzdienst zu verwenden.
- **API-Modell als Lehrer und Obergrenze**: Wird zur Generierung hochwertiger Erstmuster, schwieriger Beispielreparaturdemonstrationen und Obergrenze-Grundlinie verwendet; API-Ergebnisse und lokale Modellergebnisse sollten im Papier separat angegeben werden.
- **MCP entwickelt zuerst den Schnittstellenstil, nicht zuerst die Produktisierung**: Die erste Version implementiert zuerst die Python-Tool-Registrierung und das JSON-Schema; Sobald das Tool stabil ist, wird es auf einen MCP-kompatiblen Server gepackt, um zu vermeiden, dass die technische Komplexität in den Hauptteil des Papiers gedrängt wird.
- **Das vertikale Modelltraining erfasst nicht die Hauptlinie von G1**: Der Beitrag von G1 ist die Agentenarchitektur und der geschlossene Verifizierungskreislauf; G2 destilliert nur die Laufbahn in das lokale Modell.

Diese Sequenz kann schnell einen geschlossenen Kreislauf bilden, der übermittelt werden kann: Lassen Sie das System zunächst laufen, bewerten und erklären Sie Fehler und entscheiden Sie dann, welche Funktionen eine Feinabstimmung im Modell erfordern.

---

## 2. Systemdefinition des Cloud-Gehirns für den Verkehr in geringer HöheDas „Cloud-Gehirn für den Verkehr in geringer Höhe“ in diesem Artikel ist keine allgemeine intelligente Plattform, sondern eine **kognitive Betriebsschicht** für den städtischen UAV-Betrieb in geringer Höhe:

```text
Human / operator instruction
  -> CloudBrain LLM Agent
  -> LowAltitudeIR
  -> traffic tools / UAV tools / verifier / simulator
  -> safe decision proposal
  -> human approval or autonomous execution
```

### 2.1 Eingabe

Das Gehirn der Verkehrswolke in geringer Höhe erhält den Multi-Source-Status:

| Eingabe | Beispiel |
|------|------|
| Aufgabe in natürlicher Sprache | „Priorisieren Sie Notlieferungen in der Nähe von Krankenhäusern und meiden Sie Schulen und Flugverbotszonen.“ |
| UAV-Status | Position, Leistung, Last, Missionsstatus, Kommunikationsstatus |
| Luftraumstatus | Korridorkapazität, Flugverbotszonen, temporäre Kontrollen, Wetter, Windfelder |
| Transportbedarf | Lieferaufträge, Inspektionsaufgaben, Notfälle, Passagier-/Frachtpriorität |
| Szenenstatus | Sicherheitskritische Szenarien, Unfallszenarien, Abdeckungslöcher von Paper F |
| Formale Zwänge | LTL/STL-Sicherheitsregeln, Zeitfenster, Mindesthöhen, Mindestabstände |

### 2.2 Ausgabe

Das Cloud Brain gibt nicht direkt „Flugaktionen“ aus, sondern überprüfbare Zwischenentscheidungen:

| Ausgabe | Beispiel |
|------|------|
| LowAltitudeIR | Strukturierte Aufgaben, Entitäten, Einschränkungen, Werkzeugaufrufpläne |
| Werkzeugaufrufsequenz | Luftraum abfragen, Anrufplaner, Anrufpfadplaner, Verifizierer ausführen |
| Planungsempfehlungen | Welches UAV führt welche Aufgabe aus und ob ein Bodenrückfall ausgelöst werden soll |
| Sicherheitsdiagnose | Welche Einschränkungen dürfen verletzt werden und ob eine manuelle Bestätigung erforderlich ist |
| Erläuterungstext | Erklären Sie in natürlicher Sprache, warum es so geplant ist |

### 2.3 Cloud Brain ist kein End-to-End-Controller

Die Grenzen des Gehirns der Verkehrswolke in geringer Höhe müssen klar beschrieben werden:

- LLM führt semantisches Verständnis, Aufgabenzerlegung, Werkzeugauswahl, Interpretation und Reparatur durch.
- Der Planer führt die Flottenzuweisung und Ressourcenoptimierung entsprechend Papier B durch.
- Der Validator führt LTL/STL-Sicherheitsprüfungen durch, entsprechend Paper E.
- Szenariosimulator und Risikogenerator bieten Stresstests entsprechend Paper F.
- Die Trajektoriensteuerung wird weiterhin vom herkömmlichen Planungs-/MPC-/Sicherheitssteuerungsmodul ausgeführt.

Dies vermeidet den Zweifel des Rezensenten, dass „die LLM-Steuerung von UAV unsicher ist“.

---

## 3. Überblick über Forschungswege: vom Domänen-LLM zum allgemeinen verkörperten Agenten

Papier G kann in 4 Phasen unterteilt werden.| Stufen | These | Ziele | Schlüsselfragen |
|------|------|------|----------|
| G1 | CloudBrain-Agent | AAAI / IJCAI | LLM So rufen Sie Werkzeuge in der Verkehrswolke in geringer Höhe zuverlässig auf und bestehen die Verifizierung im geschlossenen Regelkreis
| G2 | LowAltitudeGPT | T-ITS / T-IV | So optimieren Sie ein lokales Open-Source-LLM so, dass es zu einem kognitiven Modul für die Verkehrsentscheidung in geringer Höhe wird |
| G3 | Multi-Agent-Cloud-Gehirn | AAMAS / IJCAI / T-ITS | Wie mehrere Vollzeitagenten zusammenarbeiten, um den Verkehr in geringer Höhe zu bewältigen |
| G4 | Weltmodell / VLA-Erweiterung | Langfristige Route | Wie man vom Domänenagenten zur verkörperten allgemeinen Intelligenz wechselt |

Die empfohlene Reihenfolge ist **G1 -> G2 -> G3 -> G4**.

G1 klärt zunächst, „ob das System lauffähig ist, ob es sicher geschlossen werden kann und ob es Besprechungen abhalten kann.“ G2 destilliert dann die Agenten-Trajektorie in ein Domänenmodell. G3 nutzt die Zusammenarbeit mehrerer Agenten. Die AGI-Transformation wird nur in G4 besprochen und im ersten Artikel nicht übertrieben.

---

## 4. Paper G1: CloudBrain-Agent, das erste Konferenzpapier für AAAI/IJCAI

### 4.1 Frage

**CloudBrain-Agent: Tool-erweiterte und verifizierungsgesteuerte LLM-Agenten für den Verkehrsbetrieb in geringer Höhe**

### 4.2 Zielerreichung

Erster Pitch: AAAI/IJCAI.  
Alternativen: AAMAS, ICRA/IROS-Workshop, T-ITS-Fast-Journal-Erweiterung.AAAI-26 Main Technical Track fördert die Arbeit in allen Richtungen der KI-Technologie und wichtigen Anwendungsbereichen wie dem Transportwesen. Der Haupttext ist auf 7 Seiten technischen Inhalts beschränkt und erfordert eine Reproduzierbarkeits-Checkliste [34]. Der Spezialbereich KI und Robotik des IJCAI-ECAI 2026 konzentriert sich explizit auf Roboteragenten, generative KI, Robotersteuerung, strukturierte Modellierung, Argumentation und die Art und Weise, wie man die Konsequenzen von Handlungen ausführt/vermeidet [35]. Daher sollte G1 eher als KI-Agenten-/Planungs-/Tool-Nutzungs-/Verifizierungspapier und nicht als Systemtechnik-Demonstration geschrieben werden.

### 4.3 Kernthemen

G1 möchte antworten:

> Wie kann man bei einer Verkehrsbetriebsaufgabe in geringer Höhe dafür sorgen, dass der LLM-Agent die Aufgabe zuverlässig versteht, Werkzeuge auswählt, das Planungs-/Planungs-/Überprüfungsmodul aufruft und Fehler unter Gegenbeispiel-Feedback behebt, um dadurch sichere, ausführbare und erklärbare Cloud-Brain-Entscheidungen zu treffen?

### 4.4 Methode

Vorgeschlagener **CloudBrain-Agent**, einschließlich fünf Modulen:

| Modul | Funktion |
|------|------|
| LowAltitudeIR-Parser | Konvertieren Sie Aufgaben und Systemzustände in natürlicher Sprache in strukturierte Darstellungen |
| Werkzeugplaner | Aufrufreihenfolge des Planungstools |
| Werkzeugausführer | Anrufplaner, Pfadplaner, Verifizierer, Simulator, Risikobewerter |
| Verifizierer-Feedbackschleife | Konvertieren Sie fehlgeschlagene Tool-Aufrufe, unerfüllbare Einschränkungen und STL-Robustheitsfehler in Reparatur-Feedback |
| Sicherheitsspeicher | Speichern Sie bekannte Gefahrenszenarien, Fehlerfälle, manuelle Entscheidungen und Regeleinschränkungen |

Verhaltensform des CloudBrain-Agenten:

```text
Observe -> Think -> Select Tool -> Execute -> Verify -> Repair -> Decide
```

Dies erbt die Argumentations-Aktions-Schleife von ReAct [6], fügt jedoch zwei verkehrsspezifische Mechanismen für geringe Höhenlagen hinzu:

1. **Tool-Aufrufe müssen typsicher sein**: Jede Tool-Eingabe und -Ausgabe wird anhand des „LowAltitudeIR“-Schemas überprüft.
2. **Entscheidungen müssen den Prüfer bestehen**: Alle Planungs- oder Pfadempfehlungen müssen einer Sicherheitsüberprüfung oder einem Simulationsstresstest unterzogen werden.### 4.5 LowAltitudeIR

LowAltitudeIR ist die wichtigste öffentliche Schnittstelle von G1:

```json
{
  "intent": "emergency_delivery",
  "entities": ["uav_12", "hospital_zone", "landing_pad_A"],
  "constraints": {
    "avoid": ["school_zone", "temporary_no_fly_zone"],
    "deadline_sec": 600,
    "min_obstacle_distance_m": 10,
    "altitude_range_m": [30, 120]
  },
  "tool_plan": [
    "query_airspace",
    "assign_uav",
    "plan_route",
    "verify_stl",
    "simulate_scenario"
  ],
  "fallback": "ground_vehicle_transfer_if_unreachable"
}
```

LowAltitudeIR sollte mit drei bestehenden Papierlinien kompatibel sein:

- Papier B: Aufgabenwarteschlange, UAV-Zuteilung, Vertiport-/Lade-/Korridorressourcen.
- Papier E: TaskIR, LTL/STL, Verifizierung und Fehlerkorrektur; Zu den verwandten Zitierdatenbanken gehören Lang2LTL, LTLCodeGen und ConformalNL2LTL [20] [21] [22].
- Papier F: Szenengenerierung, Abdeckungslücken, Stresstests für gefährliche Szenen.

### 4.6 Werkzeugsammlung

Die Tools von G1 müssen zu Beginn nicht auf realen Systemen aufgebaut werden. Sie können zunächst reproduzierbare experimentelle Werkzeuge erstellen:

| Werkzeug | Eingabe | Ausgabe |
|------|------|------|
| `query_airspace` | Region, Zeit, Missionstyp | Korridor, Flugverbotszone, Wetter, Kapazität |
| `assign_uav` | Aufgabe, UAV-Status, Priorität | UAV-Aufgabenzuweisung |
| `plan_route` | Anfang, Ende, Einschränkung | Pfad oder „UNREACHABLE“ |
| `verify_ltl_stl` | Aufgabenspezifikation, Flugbahn | bestanden / nicht bestanden / Gegenbeispiel |
| `simulate_scenario` | Szenario-Seed, Strategie | Erfolg, Kollision, Verzögerung, Risiko |
| `risk_assess` | Aufgaben und Szenarien | Risikoniveau, Hauptbeschränkungen |
| `explain_decision` | Entscheidungsverlauf | Für Menschen lesbare Erklärung |

### 4.7 Grundlinien| Grundlinie | Beschreibung |
|----------|------|
| Direkte LLM-Entscheidung | LLM gibt direkt Planungs-/Pfadvorschläge |
| Nur-Prompt-ReAct | Werkzeugaufruf im ReAct-Stil, jedoch ohne Typbeschränkungen und Verifizierer [6] |
| Werkzeugverwendung im Toolformer / ToolLLM-Stil | Lernen Sie, Tools aufzurufen, führen Sie jedoch keine Sicherheitsüberprüfung auf niedriger Ebene durch [7] [8] |
| TrafficOrchestrierung im GPT-Stil | LLM nennt das Verkehrsmodell, jedoch ohne UAV-Einschränkungen und formale Verifizierung [1] |
| LLM+P / klassischer Planer | LLM-Konvertierungsproblem, gelöst durch externen Planer [10] |
| Nur VERA-UAV | Nur Überprüfung der Sprache gemäß Spezifikation, keine Cloud-Brain-Multitool-Planung |
| CloudBrain-Agent voll | LowAltitudeIR + Werkzeugnutzung + Prüfer + Simulator-Feedback |

PlanBench und nachfolgende kritische Studien zu LLM-Planungsfähigkeiten haben gezeigt, dass es nicht zuverlässig ist, LLM einfach mündlich planen zu lassen und dass externe Planer, Einschränkungsprüfungen und reproduzierbare experimentelle Aufgaben eingeführt werden müssen [11] [12]. Gleichzeitig können AerialVLN und realistische UAV-VLN-Arbeiten als Benchmark-Quelle für die visuelle Sprachnavigation in geringer Höhe verwendet werden [23] [24]; DriveLM, LMDrive, DriveVLM und LaMPilot können als horizontale Referenz für den VLM/LLM-Benchmark und das Entscheidungsfindungsparadigma für autonomes Fahren verwendet werden [25] [26] [27] [28].

### 4.8 Bewertungsindikatoren| Indikator | Bedeutung |
|------|------|
| Erfolgsquote der Aufgabe | Abschlussquote von Cloud-Gehirnaufgaben |
| Tool-Call-Genauigkeit | Ob die Werkzeugauswahl und die Parameter korrekt sind |
| Ausführbare Entscheidungsrate | Ob die Ausgabe vom Scheduler/Planer ausgeführt werden kann |
| Rate von Sicherheitsverstößen | Ob die Flugverbotszone, Distanz, Höhe, Frist verletzt wird |
| Halluzinationsrate | Ob auf eine nicht vorhandene Entität, ein nicht vorhandenes Tool oder einen nicht vorhandenen Status verwiesen werden soll |
| Reparaturerfolgsquote | Ob es repariert werden kann, nachdem die Überprüfung fehlgeschlagen ist |
| Stress-Erfolgsquote des Simulators | Erfolgsquote im Gefahrenszenario Papier F |
| Latenz | Entscheidungszeit für eine einzelne Aufgabe |
| Verallgemeinerung | Leistung bei unbekannten Städten/ungesehenen Aufgaben/ungesehenen Werkzeugkombinationen |

### 4.9 Erwartete Innovationspunkte

1. Schlagen Sie „LowAltitudeIR“ und eine typisierte Tool-Use-Agent-Architektur für das Cloud-Brain für den Verkehr in geringer Höhe vor.
2. Vereinheitlichen Sie Terminplanung, Pfadplanung, formale Verifizierung und Szenariosimulation im geschlossenen Entscheidungskreislauf des LLM-Agenten.
3. Schlagen Sie eine verifizierungsgesteuerte Reparatur vor, damit sich LLM nicht mehr nur auf einen sofortigen Wiederholungsversuch verlässt.
4. Erstellen Sie einen Brain-Benchmark für den Cloud-Verkehr in geringer Höhe, der Aufgabenzerlegung, Tool-Aufruf, Planung, Überprüfung und Interpretation abdeckt.

---

## 5. Papier G2: LowAltitudeGPT, LLM-Feinabstimmung im Bereich des Tiefflugverkehrs

### 5.1 Frage

**LowAltitudeGPT: Anweisungsoptimierungs-LLMs zur Entscheidungsunterstützung für den Verkehr in geringer Höhe**

### 5.2 Ziele

G2 ist das Modell-Feinabstimmungspapier. Das Ziel besteht darin, die Agentenlaufbahn, künstliche Regeln, Simulationsrückmeldungen sowie Verifizierungs- und Reparaturdaten in G1 in ein lokales Open-Source-Modell zu destillieren, sodass das Modell zum domänenkognitiven Modul des Cloud-Gehirns für den Verkehr in geringer Höhe werden kann.Kandidatenbeiträge: T-ITS, IEEE T-IV, Applied Intelligence, wissensbasierte Systeme. T-ITS eignet sich eher für den Schwerpunkt auf intelligenten Transportsystemen, Verkehrsabläufen und Sicherheitsentscheidungen, und T-IV eignet sich eher für den Schwerpunkt auf intelligenten Fahrzeug-/unbemannten Systemmodellen und deren Bewertung [36] [37]. Wenn die Modellschulung und -bewertung stark genug ist, können Sie auch einen AAAI/IJCAI-Workshop oder eine Erweiterung der Hauptkonferenz durchführen.

### 5.3 Trainingsroute

Drei Stufen werden empfohlen:

| Stufen | Methoden | Daten |
|------|------|------|
| SFT | LoRA / QLoRA-Feinabstimmung [14] [15] | Fragen und Antworten zum Verkehr in geringer Höhe, NL-zu-IR, Rückverfolgung von Werkzeugrufen, Notfalldolmetschung |
| Präferenzabstimmung | DPO / Präferenzoptimierung [16] | Sichere Entscheidungen sind besser als gefährliche Entscheidungen, ausführbare Werkzeugsequenzen sind besser als Halluzinations-Werkzeugsequenzen |
| Überprüfbare RL | Verifizierer- und Emulator-basierte Regelbelohnungen | Erfolgreiche Aufgaben, geringes Risiko, geringe Latenz, keine Halluzination, verifiziert durch STL |

DeepSeek-R1 zeigt, dass die Argumentationsfähigkeit durch verstärkendes Lernen stimuliert werden kann [19], G2 sollte das Argumentationsmodell jedoch nicht von Grund auf trainieren. Ein realistischerer Weg besteht darin, das Open-Source-Modell Qwen/DeepSeek/Llama als Basis zu verwenden, LoRA/QLoRA für eine effiziente Feinabstimmung der Parameter zu verwenden und dann die Verifiziererbelohnung für die Ausrichtung im kleinen Maßstab zu verwenden.

### 5.4 Datenkonstruktion

Die Daten sollten nicht nur für Chat-Fragen und Antworten verwendet werden, sondern in 7 Kategorien unterteilt werden:| Datentyp | Beispiel |
|----------|------|
| Domänen-QA | „Wie geht man mit Notfallaufgaben um, wenn die Kapazität des Tiefkorridors nicht ausreicht?“ |
| NL-zu-LowAltitudeIR | Aufgaben in natürlicher Sprache bis hin zu strukturiertem IR |
| Tool-Call-Trace | Korrekte Reihenfolge und Parameter des Werkzeugaufrufs |
| Überprüfungsreparatur | Fehlgeschlagenes Gegenbeispiel zur reparierten IR |
| Erklärung zur Terminplanung | Erklärung des Planungsergebnisses |
| Notfallreaktion | Umgang mit Hochgeschwindigkeits-/städtischen Notfällen |
| Sicherheitsverweigerung | Ablehnung/Klärung bei unsicheren oder unzureichenden Informationen |

Datenquelle:

- Prozedurale Generierung: Der Paper B/F Scenario Generator erstellt Aufgaben, Karten, Zustände und Tool-Ergebnisse.
- Verifizierungsgenerierung: LTL/STL-Fehlerproben und Reparaturproben für Paper E.
- Manuelles Korrekturlesen: Stichproben und Korrektur von Proben mit hohem Risiko, um sicherzustellen, dass referenzierte Entitäten, Einschränkungen und Werkzeugparameter authentisch sind.
- Self-Instruct-Erweiterung: Verwenden Sie die Self-Instruct-Idee, um die Aufgabenvorlage zu erweitern, sie muss jedoch eine Regelfilterung und eine manuelle Stichprobe durchlaufen [17].

### 5.5 Modellauswahl

Vorschläge für die Erstausgabe:

- „Qwen2.5-7B/14B“: Gutes Chinesisch/Englisch, Code- und Tool-Aufruffähigkeiten [18].
- „DeepSeek-R1-Distill-Qwen-14B“: geeignet für Inferenz- und Verifizierungskorrekturen [19].
- „Llama-3.1-8B“: Vergleich der englischen Basislinie und des Open-Source-Ökosystems.

Es wird nicht empfohlen, in der ersten Phase mehr als 70B-Modelle zu trainieren. Der Schwerpunkt des Artikels liegt nicht auf der Modellgröße, sondern auf der **Ausrichtung der Domänentoolnutzung** und dem **Training für Verifizierungsfeedback**.

### 5.6 Bewertungsindikatoren| Indikator | Bedeutung |
|------|------|
| IR exakte Übereinstimmung / Feld F1 | Strukturierte LowAltitudeIR-Ausgabequalität |
| Erfolgreicher Toolaufruf | Werkzeugname, Reihenfolge, Parametergenauigkeit |
| Verifizierte Entscheidungsrate | Anteil der Ausgabe, die den Prüfer passiert |
| Genauigkeit der Sicherheitsverweigerung | Ob die unsichere/unzureichende Informationsaufgabe abgelehnt oder geklärt werden soll |
| Reparaturfähigkeit | Reparatur-Erfolgsquote nach Betrachtung des Gegenbeispiels |
| Lokale Bereitstellungslatenz | Lokale Inferenzlatenz und Speichernutzung |
| Stadtübergreifende Verallgemeinerung | Verallgemeinerung unsichtbarer Städte/Szenen |

---

## 6. Paper G3: Multi-Agent Cloud Brain, kollaboratives Cloud Brain mit mehreren Agenten

### 6.1 Frage

**Multi-Agent Cloud Brain für kooperatives UAV-Verkehrsmanagement in geringer Höhe**

### 6.2 Ziele

G3 reicht von der Zusammenarbeit einzelner Agenten bis hin zur Zusammenarbeit mehrerer Agenten. Kandidateneinreichungen: AAMAS, IJCAI, AAAI, T-ITS.

AAMAS wird sich auf autonome Agenten und Multiagentensysteme konzentrieren [38], die sich sehr gut für die Zusammenarbeit mit mehreren Rollen in Cloud-Gehirnen für den Verkehr in geringer Höhe eignen.

### 6.3 Arbeitsteilung des Agenten

| Agent | Verantwortlichkeiten |
|-------|------|
| Luftraummonitor | Überwachen Sie Korridore, Flugverbotszonen, Wetter und Kapazität |
| Flottenplaner | Verantwortlich für die Aufgabenwarteschlange und UAV-Verteilung |
| Sicherheitsprüfer | Verantwortlich für LTL/STL, Risiken und Gegenbeispiele |
| Szenariotester | Rufen Sie den Paper F-Szenengenerator auf, um einen Stresstest durchzuführen |
| Notfallkoordinator | Verantwortlich für Notfallmaßnahmen und Bodenverbindung |
| Human Interface Agent | Verantwortlich für Erklärung, Klarstellung und menschliche Bestätigung |### 6.4 Schlüsselfragen der Forschung

1. Sind mehrere Agenten zuverlässiger als einzelne Agenten?
2. Wird der gemeinsam genutzte Speicher Fehler verbreiten?
3. Wer hat die endgültige Entscheidungsbefugnis, wenn zwei Agenten in Konflikt geraten?
4. Kann der Prüfer als Schiedsrichter fungieren?
5. Ist die durch mehrere Agenten verursachte Verzögerung akzeptabel?

### 6,5 Innovationspunkte

Die Innovation von G3 besteht nicht darin, dass „mehrere GPTs miteinander chatten“, sondern:

- Ein Vollzeitagent ist an Tieflandfahrzeuge gebunden;
- Der freigegebene Status wird durch „LowAltitudeIR“ und ein Ereignisprotokoll dargestellt.
- Die Sicherheitsschiedsgerichtsbarkeit wird durch Prüfer und Simulator durchgeführt;
- Meinungsverschiedenheiten zwischen mehreren Akteuren können in Unsicherheit und menschliche Interventionssignale umgewandelt werden.

---

## 7. Paper G4: World-Model/VLA-Erweiterung für die allgemeine AGI-Fähigkeitsmigration

### 7.1 Gesamtpositionierung

G4 ist der langfristige Weg und sollte in den ersten beiden Artikeln nicht überbewertet werden. Der vorgeschlagene Ausdruck ist:

> **auf dem Weg zu allgemeiner verkörperter Verkehrsintelligenz**

Anstatt „AGI zu implementieren“.

Der offene verkörperte Agent von Voyager und die Sprache-zu-Roboter-Affordance-Grundlage von SayCan veranschaulichen, dass der Schlüssel für die Entwicklung von LLM in Richtung verkörperter Intelligenz nicht darin liegt, chatten zu können, sondern in der Lage zu sein, das Umgebungsfeedback, die Fähigkeitsbibliotheken und die Handlungsbeschränkungen kontinuierlich zu verbessern [9] [13]. Das Gehirn der Verkehrswolke in geringer Höhe kann diese Idee in einen sichereren und besser auswertbaren Verkehrsbetriebsbereich umsetzen.

### 7.2 Warum ist dies ein logischer Einstieg in die AGI-Richtung?

Das Gehirn der Verkehrswolke in geringer Höhe verfügt natürlich über mehrere Fähigkeiten, die für die allgemeine verkörperte Intelligenz erforderlich sind:

- Räumliches Verständnis: städtischer 3D-Raum, Hindernisse, Luftraumhierarchie.
- Zeitberücksichtigung: Aufgabenwarteschlange, Frist, dynamisches Wetter, Entwicklung von Verkehrsereignissen.
- Tool-Nutzung: Planer, Planer, Verifizierer, Simulator.
- Handlungskonsequenzen: Fehlentscheidungen können zu Verzögerungen, Risiken oder Sicherheitsverstößen führen.
- Zusammenarbeit mehrerer Agenten: UAV, Bodenfahrzeug, menschlicher Bediener, regulatorische Vorschriften.PaLM-E, RT-2 und OpenVLA haben einen Trend des Übergangs vom sprachlichen/visuellen Vortraining zum verkörperten Handeln gezeigt [29][30][31]. Das Cloud-Gehirn des Verkehrs in geringer Höhe sollte jedoch nicht mit einer End-to-End-VLA beginnen, sondern zunächst Agenten + Tools + Verifizierer verwenden, um eine kognitive Sicherheitsarchitektur zu etablieren.

### 7.3 Langfristige technische Roadmap

| Bühne | Fähigkeit | Technologie |
|------|------|------|
| G1 | Werkzeugaufruf und Überprüfung im geschlossenen Regelkreis | LLM-Agent + LowAltitudeIR |
| G2 | Domänenmodell | SFT / LoRA / DPO / Verifizierer-Belohnung |
| G3 | Multi-Agenten-Zusammenarbeit | Shared Memory + Verifier-Arbitrierung |
| G4 | Weltmodell | räumlich-zeitliche Vorhersage + Simulator-Feedback |
| G5 | VLA / verkörperte Richtlinie | Multimodaler Input zu Handlungsempfehlungen, aber dennoch umgesetzt durch Safety Layer |

Die Schlüsselwörter der AGI-Transformation sollten sein: **Generalisierung, kontinuierliches Lernen, verkörpertes Denken, Selbstbewertung, Werkzeugerstellung**. Schreiben Sie nicht „Wir haben ein AGI-Modell trainiert.“

---

## 8. Datenaufbau und Trainingsplan

### 8.1 Datenübersichtstabelle| Datensatz | Quelle | Verwendung |
|--------|------|------|
| LowAltitude-Anleitung | Manuelle Vorlage + LLM-Generierung + manuelle Probenahme | Verständnis von Aufgaben in natürlicher Sprache |
| LowAltitudeIR-Gold | Regelgenerierung + manuelle Korrektur | IR-Schulung und -Bewertung |
| ToolTrace-Bank | G1-Agent, der Trace ausführt | Werkzeugaufruf SFT |
| VerifyRepair-Bench | Papier E Gegenbeispiel-Reparatur | Schulung zur Verifizierung und Fehlerkorrektur |
| SzenarioStress-Bank | Paper F-Szenariogenerierung | Verallgemeinerung gefährlicher Szenen |
| FleetOps-Bench | Papier B Planungssimulation | Aufgabenwarteschlange und Ressourcenplanung |
| EmergencyOps-Bench | Hochgeschwindigkeits-/Stadtnotfall-Synthesefall | Notfallentscheidungen |

In der Simulationsschicht wird empfohlen, zunächst einen leichten, selbstgebauten Simulator zu verwenden, um kontrollierbare Variablen sicherzustellen, und dann AirSim und Flightmare für die visuelle, dynamische und geschlossene Flugergänzungsüberprüfung zu verwenden [32] [33]. Auf diese Weise kann G1/G2 reproduziert werden, ohne auf Hochleistungssimulatoren angewiesen zu sein, und kann in Zukunft natürlich auf realistischere UAV-Szenarien erweitert werden.

### 8.2 Trainingsbeispielformat

Es wird empfohlen, auf JSONL zu vereinheitlichen:

```json
{
  "instruction": "优先处理医院附近应急配送，避开学校和临时禁飞区。",
  "state": {
    "uavs": "...",
    "airspace": "...",
    "tasks": "..."
  },
  "target_ir": {
    "intent": "emergency_delivery",
    "constraints": ["avoid_school", "avoid_no_fly_zone"]
  },
  "tool_trace": [
    {"tool": "query_airspace", "args": {"region": "hospital_zone"}},
    {"tool": "assign_uav", "args": {"priority": "emergency"}},
    {"tool": "verify_ltl_stl", "args": {"spec": "..."}}
  ],
  "verifier_feedback": "pass",
  "final_answer": "建议派遣 uav_12，经 corridor_B 绕开学校区域。"
}
```

### 8.3 Trainingsphase

1. **Prompt + RAG-Basislinie**
   Überprüfen Sie zunächst ohne Schulung die Aufgabendefinition und das Toolschema.

2. **SFT/LoRA**
   Das trainierte Modell gibt LowAltitudeIR- und Tool-Call-Traces aus.

3. **DPO/Präferenzabstimmung**
   Bevorzugen Sie sichere, ausführbare, weniger halluzinatorische Entscheidungen und Entscheidungen mit geringer Latenz.

4. **Ausrichtung der Verifizierer-Belohnung**
   Nutzen Sie Validator- und Simulatorergebnisse als Regelbelohnungen, um die Reparaturfähigkeiten zu stärken.

5. **Destillation**
   Destillieren Sie starke Modelle oder Trajektorien mit mehreren Agenten in lokale 7B/14B-Modelle.

---

## 9. Versuchsaufbau, Ausgangswerte und Bewertungsindikatoren

### 9.1 G1-Hauptexperiment| Experiment | Zweck |
|------|------|
| Erfolgreicher Werkzeugeinsatz | Testwerkzeugauswahl und Parameterfüllung |
| Verifizierte Planung | Testen Sie, ob der Zeitplan/Pfad die Überprüfung besteht |
| Schleife reparieren | Testen Sie, ob Gegenbeispiel-Feedback die Erfolgsquote verbessern kann |
| Szenario-Stresstest | Testen Sie die Robustheit mit Gefahrenszenarien von Paper F |
| Verallgemeinerung | Ungesehene Städte, ungesehene Aufgaben und ungesehene Werkzeugkombinationen testen |

### 9.2 G2-Feinabstimmungsexperiment

| Experiment | Zweck |
|------|------|
| Basis vs. LoRA vs. QLoRA | Vorteile der Feinabstimmung überprüfen |
| SFT vs. DPO | Validierung der Vorteile der Präferenzausrichtung |
| Mit/ohne Prüfer-Feedback | Sicherheitsfeedbackwert überprüfen |
| 7B vs. 14B vs. Argumentationsmodell | Überprüfen Sie den Kosten-Leistungs-Kompromiss für die lokale Bereitstellung |
| Szenarioübergreifender Transfer | Migration vom synthetischen Szenario zum Notfallszenario überprüfen |

### 9.3 Grundlinien

| Grundlinie | Beschreibung |
|----------|------|
| GPT/Qwen direkte Antwort | Direkte Antwort, kein Werkzeug |
| ReAct-Eingabeaufforderung | Aufforderung zum Denken und Handeln [6] |
| API-Aufruf im Toolformer-Stil | Werkzeugaufruf ohne Sicherheitsregelkreis [7] |
| Geschulter Tool-Benutzer im ToolLLM-Stil | Schulungsgrundlage für Open-Source-Tool-Aufrufe [8] |
| TrafficTraffic-Orchestrierung im GPT-Stil | LLM + Verkehrsmodelle [1] |
| LLM+P | LLM + externer Planer [10] |
| CloudBrain-Agent voll | Methoden in diesem Artikel |

### 9.4 Indikatoren| Metriken | Ziele |
|------|------|
| Aufgabenerfolg | Abschlussrate der Cloud-Gehirnaufgabe |
| Tool-Call-Genauigkeit | Tool-Call-Genauigkeit |
| IR-Feld F1 | LowAltitudeIR-Genauigkeit auf Feldebene |
| Halluzinationsrate | Verhältnis der nicht vorhandenen Tools/Entitäten/Regeln |
| Rate von Sicherheitsverstößen | Anteil der Verstöße gegen Sicherheitsvorschriften |
| Reparaturerfolg | Erfolgsquote der Gegenbeispielreparatur |
| Latenz | Entscheidungsverzögerung |
| Bewertung des menschlichen Vertrauens | Erklärungsqualität menschlicher Gutachter |
| Generalisierungsbewertung | Verallgemeinerung unsichtbarer Szenen |

---

## 10. Empfohlener Übermittlungspfad

### 10.1 Route des ersten Treffens

**G1-Erststimme AAAI / IJCAI. **

Papiertyp: KI-Agent + Planung + Verifizierung + Transportantrag.

Die Kernbeiträge sind in drei Teile gegliedert:

1. LowAltitudeIR- und Low-Altitude-Traffic-Tool-Use-Agent-Architektur.
2. Verifizierungsgeführte Reparaturschleife.
3. Benchmark- und Bewertungsprotokoll für das Cloud-Gehirn in geringer Höhe.

### 10.2 Follow-up-Journalroute

| Papier | Einreichung |
|------|------|
| G2 LowAltitudeGPT | T-ITS / T-IV / Angewandte Intelligenz |
| G3 Multi-Agent Cloud Brain | AAMAS -> T-ITS-Erweiterung |
| G4 Weltmodell/VLA | ICRA / IROS / T-RO / langfristiger AGI-orientierter Veranstaltungsort |

### 10.3 Nicht empfohlene Routen- Es wird nicht empfohlen, im ersten Artikel ein großes Modell zu trainieren.
- Es wird nicht empfohlen, „AGI Cloud Brain“ als Haupttitel zu schreiben.
- Es wird nicht empfohlen, LLM UAV-Steuerungsaktionen direkt ausgeben zu lassen.
- Es wird nicht empfohlen, nur einen Chat-Frage- und Antwortdatensatz zu erstellen.
- Es wird nicht empfohlen, den Prüfer zu ignorieren, da sonst die sicherheitskritischen Szenarien nicht überzeugend genug sind.

---

## 11. Referenzen

[1] Siyao Zhang, Daocheng Fu, Wenzhe Liang, Zhao Zhang, Bin Yu, Pinlong Cai und Baozhen Yao. „TrafficGPT: Anzeigen, Verarbeiten und Interagieren mit Traffic Foundation-Modellen.“ *Transport Policy*, 150:95-105, 2024. DOI: 10.1016/j.tranpol.2024.03.006. URL: <https://www.sciencedirect.com/science/article/pii/S0967070X24000726>

[2] Sebastian Wandelt, Changhong Zheng, Shuang Wang, Yucheng Liu und Xiaoqian Sun. „Große Sprachmodelle für intelligenten Transport: Ein Überblick über den Stand der Technik und Herausforderungen.“ *Applied Sciences*, 14(17):7455, 2024. DOI: 10.3390/app14177455. URL: <https://www.mdpi.com/2076-3417/14/17/7455>[3] Doaa Mahmud, Hadeel Hajmohamed, Shamma Almentheri, Shamma Alqaydi, Lameya Aldhaheri, Ruhul Amin Khalil und Nasir Saeed. „Integration von LLMs mit ITS: Aktuelle Fortschritte, Potenziale, Herausforderungen und zukünftige Richtungen.“ *IEEE Transactions on Intelligent Transportation Systems*, 26(5):5674-5709, 2025. DOI: 10.1109/TITS.2025.3528116. URL: <https://ieeexplore.ieee.org/document/10851302>

[4] Zhonghang Li, Lianghao Xia, Jiabin Tang, Yong Xu, Lei Shi, Long Xia, Dawei Yin und Chao Huang. „UrbanGPT: Räumlich-zeitliche große Sprachmodelle.“ arXiv:2403.00813, 2024. URL: <https://arxiv.org/abs/2403.00813>

[5] Yuan Yuan, Jingtao Ding, Jie Feng, Depeng Jin und Yong Li. „UniST: Ein Prompt-gestütztes Universalmodell für urbane räumlich-zeitliche Vorhersagen.“ *Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)*, 2024. DOI: 10.1145/3637528.3671662. URL: <https://arxiv.org/abs/2402.11838>[6] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan und Yuan Cao. „ReAct: Synergie zwischen Denken und Handeln in Sprachmodellen.“ *International Conference on Learning Representations (ICLR)*, 2023. URL: <https://openreview.net/forum?id=WE_vluYUL-X>

[7] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda und Thomas Scialom. „Toolformer: Sprachmodelle können sich selbst den Umgang mit Werkzeugen beibringen.“ *Advances in Neural Information Processing Systems 36 (NeurIPS)*, 2023. URL: <https://proceedings.neurips.cc/paper_files/paper/2023/hash/d842425e4bf79ba039352da0f658a906-Abstract-Conference.html>[8] Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Runchu Tian, ​​Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu und Maosong Sun. „ToolLLM: Erleichtert die Beherrschung großer Sprachmodelle von über 16.000 realen APIs.“ *International Conference on Learning Representations (ICLR)*, 2024. URL: <https://openreview.net/forum?id=dHng2O0Jjr>

[9] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan und Anima Anandkumar. „Voyager: Ein verkörperter Agent mit offenem Ende und großen Sprachmodellen.“ arXiv:2305.16291, 2023. URL: <https://arxiv.org/abs/2305.16291>

[10] Bo Liu, Yuqian Jiang, Xiaohan Zhang, Qiang Liu, Shiqi Zhang, Joydeep Biswas und Peter Stone. „LLM+P: Große Sprachmodelle mit optimaler Planungskompetenz stärken.“ arXiv:2304.11477, 2023. URL: <https://arxiv.org/abs/2304.11477>[11] Karthik Valmeekam, Matthew Marquez, Alberto Olmo, Sarath Sreedharan und Subbarao Kambhampati. „PlanBench: Ein erweiterbarer Benchmark zur Bewertung großer Sprachmodelle zur Planung und Begründung von Veränderungen.“ *Advances in Neural Information Processing Systems 36 (NeurIPS) Datasets and Benchmarks Track*, 2023. URL: <https://openreview.net/forum?id=YXogl4uQUO>

[12] Karthik Valmeekam, Alberto Olmo, Sarath Sreedharan und Subbarao Kambhampati. „Über die Planungsfähigkeiten großer Sprachmodelle: Eine kritische Untersuchung.“ *Advances in Neural Information Processing Systems 36 (NeurIPS)*, 2023. URL: <https://arxiv.org/abs/2305.15771>[13] Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Daniel Ho, et al. „Tue, was ich kann, nicht was ich sage: Sprache in robotischen Errungenschaften verankern.“ *Conference on Robot Learning (CoRL)*, PMLR 205, 2022. URL: <https://proceedings.mlr.press/v205/ahn23a.html>

[14] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang und Weizhu Chen. „LoRA: Low-Rank-Anpassung großer Sprachmodelle.“ *International Conference on Learning Representations (ICLR)*, 2022. URL: <https://openreview.net/forum?id=nZeVKeeFYf9>[15] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman und Luke Zettlemoyer. „QLoRA: Effiziente Feinabstimmung quantisierter LLMs.“ *Advances in Neural Information Processing Systems 36 (NeurIPS)*, 2023. URL: <https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html>

[16] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning und Chelsea Finn. „Direkte Präferenzoptimierung: Ihr Sprachmodell ist insgeheim ein Belohnungsmodell.“ *Advances in Neural Information Processing Systems 36 (NeurIPS)*, 2023. URL: <https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html>[17] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi und Hannaneh Hajishirzi. „Selbstunterricht: Sprachmodelle mit selbst generierten Anweisungen ausrichten.“ *Jahrestagung der Association for Computational Linguistics (ACL)*, 2023. URL: <https://aclanthology.org/2023.acl-long.754/>

[18] Qwen-Team. „Technischer Bericht zu Qwen2.5.“ arXiv:2412.15115, 2024. URL: <https://arxiv.org/abs/2412.15115>

[19] DeepSeek-KI. „DeepSeek-R1: Anreize für die Denkfähigkeit in LLMs durch Reinforcement Learning schaffen.“ arXiv:2501.12948, 2025. URL: <https://arxiv.org/abs/2501.12948>

[20] Bo Liu, Yuqian Jiang, Xiaohan Zhang, Qiang Liu, Shiqi Zhang, Joydeep Biswas und Peter Stone. „Lang2LTL: Übersetzen natürlicher Sprachbefehle in zeitliche Spezifikationen mit großen Sprachmodellen.“ *Conference on Robot Learning (CoRL)*, PMLR 229, 2023. URL: <https://proceedings.mlr.press/v229/liu23d.html>[21] Behrad Rabiei, Mahesh Kumar A. R., Zhirui Dai, Surya L. S. R. Pilla, Qiyue Dong und Nikolay Atanasov. „LTLCodeGen: Codegenerierung syntaktisch korrekter zeitlicher Logik für die Roboteraufgabenplanung.“ arXiv:2503.07902, 2025. URL: <https://arxiv.org/abs/2503.07902>

[22] Jun Wang, David Smith Sundarsingh, Jyotirmoy V. Deshmukh und Yiannis Kantaros. „ConformalNL2LTL: Übersetzen von Anweisungen in natürlicher Sprache in temporale Logikformeln mit konformen Korrektheitsgarantien.“ arXiv:2504.21022, 2025. URL: <https://arxiv.org/abs/2504.21022>

[23] Shubo Liu, Hongsheng Zhang, Yuankai Qi, Peng Wang, Yanning Zhang und Qi Wu. „AerialVLN: Vision-and-Language-Navigation für UAVs.“ *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2023, S. 15384-15394. URL: <https://openaccess.thecvf.com/content/ICCV2023/html/Liu_AerialVLN_Vision-and-Language_Navigation_for_UAVs_ICCV_2023_paper.html>[24] Xiangyu Wang, Donglin Yang, Ziqin Wang, Hohin Kwan, Jinyu Chen, Wenjun Wu, Hongsheng Li, Yue Liao und Si Liu. „Auf dem Weg zu einer realistischen UAV-Vision-Language-Navigation: Plattform, Benchmark und Methodik.“ *International Conference on Learning Representations (ICLR)*, 2025. URL: <https://openreview.net/forum?id=rUvCIvI4eB>

[25] Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Jens Beisswenger, Ping Luo, Andreas Geiger und Hongyang Li. „DriveLM: Fahren mit grafischer Beantwortung visueller Fragen.“ arXiv:2312.14150, 2023. URL: <https://arxiv.org/abs/2312.14150>

[26] Hao Shao, Yuxuan Hu, Letian Wang, Steven L. Waslander, Yu Liu und Hongsheng Li. „LMDrive: Closed-Loop-End-to-End-Fahren mit großen Sprachmodellen.“ *IEEE/CVF-Konferenz zu Computer Vision und Mustererkennung (CVPR)*, 2024. URL: <https://arxiv.org/abs/2312.07488>[27] Xiaoyu Tian, ​​Junru Gu, Bailin Li, Yicheng Liu, Yang Wang, Zhiyong Zhao, Kun Zhan, Peng Jia, Xianpeng Lang und Hang Zhao. „DriveVLM: Die Konvergenz von autonomem Fahren und großen Vision-Language-Modellen.“ arXiv:2402.12289, 2024. URL: <https://arxiv.org/abs/2402.12289>

[28] Yunsheng Ma, Can Cui, Xu Cao, Wenqian Ye, Peiran Liu, Juanwu Lu, Amr Abdelraouf, Rohit Gupta, Kyungtae Han, Aniket Bera, James M. Rehg und Ziran Wang. „LaMPilot: Ein offener Benchmark-Datensatz für autonomes Fahren mit Sprachmodellprogrammen.“ *IEEE/CVF-Konferenz zu Computer Vision und Mustererkennung (CVPR)*, 2024, S. 15141-15151. URL: <https://openaccess.thecvf.com/content/CVPR2024/html/Ma_LaMPilot_An_Open_Benchmark_Dataset_for_Autonomous_Driving_with_Language_CVPR_2024_paper.html>[29] Danny Driess, Fei Zeng, Igor Mordatch und Pete Florence. „PaLM-E: Ein verkörpertes multimodales Sprachmodell.“ *International Conference on Machine Learning (ICML)*, PMLR 202, 2023. URL: <https://proceedings.mlr.press/v202/driess23a.html>

[30] Anthony Brohan, Noah Brown, Richter Carbajal, Jewgen Tschebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence und andere. „RT-2: Vision-Language-Action-Modelle übertragen Webwissen auf Robotersteuerung.“ arXiv:2307.15818, 2023. URL: <https://arxiv.org/abs/2307.15818>[31] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang und Chelsea Finn. „OpenVLA: Ein Open-Source-Vision-Sprache-Aktionsmodell.“ arXiv:2406.09246, 2024. URL: <https://arxiv.org/abs/2406.09246>

[32] Shital Shah, Debadeepta Dey, Chris Lovett und Ashish Kapoor. „AirSim: Hochpräzise visuelle und physikalische Simulation für autonome Fahrzeuge.“ *Field and Service Robotics*, Springer Proceedings in Advanced Robotics, 2017; arXiv:1705.05065. URL: <https://arxiv.org/abs/1705.05065>

[33] Yunlong Song, Selim Naji, Elia Kaufmann, Antonio Loquercio und Davide Scaramuzza. „Flightmare: Ein flexibler Quadrocopter-Simulator.“ *Conference on Robot Learning (CoRL)*, PMLR 155, 2021. URL: <https://proceedings.mlr.press/v155/song21a.html>[34] AAAI. „AAAI-26 Main Technical Track: Call for Papers.“ URL: <https://aaai.org/conference/aaai/aaai-26/main-technical-track-call/>

[35] IJCAI-ECAI 2026. „Call for Papers – AI and Robotics Special Track.“ URL: <https://2026.ijcai.org/ijcai-ecai-2026-call-for-papers-ai-and-robotics/>

[36] IEEE Intelligent Transportation Systems Society. „IEEE-Transaktionen auf intelligenten Transportsystemen (T-ITS): Umfang.“ URL: <https://ieee-itss.org/pub/t-its/>

[37] IEEE Intelligent Transportation Systems Society. „IEEE-Transaktionen auf intelligenten Fahrzeugen.“ URL: <https://ieee-itss.org/pub/t-iv/>

[38] AAMAS 2026. „Call for Papers – Main Track.“ URL: <https://cyprusconferences.org/aamas2026/call-for-papers-main-track/>

---

## Anhang: 12-Monats-Werbeplan

### Monate 1–2: G1-Probleme und Schnittstellen einfrieren

- Frieren Sie den CloudBrain-Agent-Titel, die Zusammenfassung und drei Beiträge ein.
- Definieren Sie LowAltitudeIR v0.1.
- Definieren Sie die Tool-API: Luftraum, Scheduler, Planer, Prüfer, Simulator, Risiko.
- Erstellen Sie eine Verifizierungspipeline für 100–200 kleine Aufgabenbeispiele.### Monate 3–4: Aufbau von CloudBrain-Bench

- Generieren Sie über 1000 Verkehrsmissionen in geringer Höhe.
- Deckt normale Planung, Notfallverteilung, Vermeidung von Flugverbotszonen, Ladeengpässe, Korridorüberlastung und unerfüllbare Aufgaben ab.
- Markieren Sie Gold LowAltitudeIR, Gold-Werkzeugspur, erwartete Entscheidung.

### Monate 5–6: Implementierung der G1-Grundlinien

- Direktes LLM.
-ReAct-Eingabeaufforderung.
- Tool-Nutzung ohne Prüfer.
- Orchestrierung im TrafficGPT-Stil.
- LLM+P.
- Nur VERA-UAV.

### Monate 7–8: CloudBrain-Agent vollständig implementieren

- Typisiertes Werkzeugschema hinzufügen.
- Fügen Sie Prüfer-Feedback hinzu.
- Simulator-Stresstest hinzugefügt.
- Sicherheitsspeicher hinzufügen und Schleife reparieren.

### Monate 9–10: Hauptexperiment

- Erfolg der Ausführungsaufgabe, Genauigkeit des Werkzeugaufrufs, Sicherheitsverletzung, Reparaturerfolg, Latenz.
- Verallgemeinerung der Führung durch unsichtbare Städte, unsichtbare Missionen und gefährliche Szenen.
- Ablation durchführen: kein IR, kein Verifizierer, kein Simulator, kein Speicher, keine Reparatur.

### Monat 11: G2-Feinabstimmung vor dem Experiment

- Sammeln Sie G1-Werkzeugspuren.
- LoRA-Feinabstimmung von Qwen/DeepSeek.
- Vergleichen Sie Basis vs. SFT vs. DPO.
- Bestimmen Sie, ob es ausreicht, G2 zu bilden.

### Monat 12: Erster AAAI/IJCAI-Entwurf

-Schreiben Sie G1-Konferenzbeiträge.
– Der Anhang enthält das LowAltitudeIR-Schema, die Werkzeugdefinition und die Regeln zur Datengenerierung.
- Stellen Sie sicher, dass die Checkliste zur Reproduzierbarkeit, der Code, die Daten und die Vorbereitung des experimentellen Saatguts vollständig sind.