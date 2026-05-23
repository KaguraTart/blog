---
title: "Paper E Experimental Task Book v2: Verifizierung und Fehlerkorrektur UAV-Sprachplanung für AAAI"
description: "v2 konzentriert sich auf Einreichungen bei AAAI-Top-Konferenzen: Ergänzung von mehr als 30 echten und zitierfähigen regulären Konferenz-/Top-Journal-/wichtigen Preprint-Dokumenten, Vertiefung der experimentellen Indikatoren, Vergleichs- und Ablationsschemata und reproduzierbaren experimentellen Protokolle von VERA-UAV und Bereitstellung eines mathematischen Nachweises der relativen Vollständigkeit."
pubDate: 2026-05-17
updatedDate: 2026-05-23
tags: ["Papier E", "AAAI", "UAV", "LLM", "LTL", "STL", "Formale Überprüfung", "Experimentelles Aufgabenbuch", "Nachweis der Vollständigkeit"]
category: Tech
---

# Paper E Experimental Task Book v2: Verifizierung und Fehlerkorrektur UAV-Sprachplanung für AAAI

> Diese Datei verwendet immer noch den Dateinamen „paper-e-vera-uav-experiment-taskbook-v1-20260517.md“, da diese Runde eine „direkte Änderung an der V1-Version“ erfordert. Der Text, der Titel und die Versionshinweise wurden alle auf **v2** aktualisiert. Bei diesem Artikel handelt es sich nicht um einen endgültigen Papierentwurf, sondern um eine ausführbare experimentelle Aufgabenstellung: Klärung der Forschungspositionierung von Papier E, echte zitierfähige Dokumente, Algorithmuslösungen, Datenkonstruktion, Vergleichsexperimente, Ablationsexperimente, Bewertungsindikatoren, theoretische Vollständigkeitsgrenzen und nachfolgende AAAI/T-ITS-Förderpläne. Ergänzender Schwerpunkt am 19.05.2026 ist: Verhinderung von Datenlecks, Fehlertaxonomie, Parameterbudgetierung, Indikatorformeln, Diagrammplanung und AAAI-Compliance-Risiken.

---

## 1. Forschungshintergrund und Ziele

Die Planung städtischer UAV-Missionen in geringer Höhe wandelt sich von „vom Ingenieur voreingestellten Routen“ zu „missionsgesteuert in natürlicher Sprache“. In tatsächlichen Anwendungen geben Bediener eher die folgenden Anweisungen:

- „Überprüfen Sie zunächst die Ostfassade von Gebäude 3, gehen Sie dann zum Landepunkt auf dem Dach und warten Sie.“
- „Meiden Sie die Luft über dem Krankenhaus und erreichen Sie den temporären Entbindungsbereich innerhalb von 30 Sekunden.“
- „Wenn der Südkorridor besetzt ist, umgehen Sie den Westkorridor, halten Sie aber durchgehend einen Sicherheitsabstand von mehr als 20 Metern ein.“

Diese Anweisungen umfassen gleichzeitig semantisches Verständnis, zeitliche Reihenfolge, räumliche Einschränkungen, kontinuierliche Flugbahnsicherheit und Erreichbarkeitsbeurteilungen. Große Sprachmodelle (LLM) sind gut darin, natürliche Sprache zu verstehen und Kandidatenpläne zu generieren, sie können jedoch nicht garantieren, dass der Ausgabeplan im physischen Raum ausführbar ist, noch kann sie garantieren, dass Flugsicherheitsbeschränkungen eingehalten werden. Formale Methoden eignen sich gut zur Bereitstellung verifizierbarer Semantiken, wie beispielsweise die lineare zeitliche Logik (LTL) und die Signal-Temporale Logik (STL). Direkte handschriftliche Spezifikationen erfordern jedoch Fachwissen und sind für Laien nur schwer zu bedienen.

Bestehende Arbeiten haben gezeigt, dass die Übersetzung von natürlicher Sprache in LTL die Schwelle zum Schreiben von Roboteraufgabenspezifikationen erheblich senken kann. Lang2LTL wandelt beispielsweise komplexe Navigationsbefehle in LTL um und führt eine Generalisierungsbewertung in unsichtbaren Umgebungen durch [1]; NL2LTL bietet ein Open-Source-Python-Paket von natürlicher Sprache bis LTL [2]; LTLCodeGen verwendet Codegenerierung, um die grammatikalische Korrektheit von LTL zu verbessern und integriert es in die Roboterpfadplanung [3]; ConformalNL2LTL versucht außerdem, die konforme Vorhersage zu verwenden, um die Übersetzungsgenauigkeit zu gewährleisten [4]. Diese Arbeiten bilden eine wichtige Grundlage für diese Studie.Aber für UAV-Szenarien in geringer Höhe reicht es nicht aus, nur eine NL-zu-LTL-Übersetzung durchzuführen. Für UAV-Missionen gelten drei zusätzliche Anforderungen:

1. **Kontinuierliche Sicherheitseinschränkungen**: Einschränkungen wie Flughöhe, Geschwindigkeit, Hindernisentfernung, Zeitfenster usw. sind natürlich Einschränkungen für kontinuierliche Signale und eignen sich besser für die Bewertung der STL-Robustheit.
2. **Ausführbare Trajektorie im geschlossenen Regelkreis**: Korrekte Spezifikationen bedeuten nicht, dass die Trajektorie machbar ist und müssen durch Karten, Dynamik und Planer überprüft werden.
3. **Fehler können behoben werden**: LLM-Fehler sollten nicht nur als Fehler beurteilt werden, sondern vom Verifizierer in Gegenbeispiele oder Robustheitsrückmeldungen umgewandelt werden und dann die LLM-Korrektur vorantreiben.

Daher schlägt dieser Artikel **VERA-UAV** vor: ein neurosymbolisches Planungsframework zur Überprüfung und Fehlerkorrektur für UAV-Aufgaben in natürlicher Sprache. Die AAAI-Version priorisiert die Beantwortung einer Kernfrage:

> Wie kann bei einer UAV-Mission in natürlicher Sprache ein natives Open-Source-LLM überprüfbare, reparierbare und ausführbare LTL/STL-Missionsspezifikationen und -Trajektorien generieren, anstatt nur Textpläne zu erstellen, die vernünftig erscheinen, aber nicht nachweislich sicher sind?

Die AAAI-Hauptkonferenzversion konzentriert sich auf KI-Planung, neurosymbolische Verifizierung und LLM-Selbstreparatur. Inhalte auf Systemebene wie AirSim, echte Tieffluglogistik und Multi-UAV-Luftraumdurchsatz werden in nachfolgende erweiterte T-ITS-Versionen integriert.

---

## 2. Problemdefinition und Kernannahmen

### 2.1 Eingabe und Ausgabe

Gegeben eine UAV-Aufgabeninstanz:

$$
\mathcal{I} = (x_{\text{NL}}, \mathcal{M}, s_0)
$$

Darunter ist $x_{\text{NL}}$ die Aufgabenanweisung in natürlicher Sprache, $\mathcal{M}$ ist die städtische Karte in geringer Höhe mit semantischer Annotation und $s_0$ ist der UAV-Anfangszustand. Die Karte enthält Gebäude, Flugverbotszonen, passierbaren Luftraum, Landepunkte, Inspektionsziele, dynamische Hindernisse und Höhenstufen.

Systemausgabe:

$$
\mathcal{O} = (\text{TaskIR}, \varphi_{\text{LTL}}, \varphi_{\text{STL}}, \tau, r)
$$Dabei ist TaskIR die strukturierte Zwischendarstellung, $\varphi_{\text{LTL}}$ die diskrete Timing-Aufgabenspezifikation, $\varphi_{\text{STL}}$ die kontinuierliche Trajektorienbeschränkung, $\tau$ die Kandidatentrajektorie und $r$ das Verifizierungsergebnis. Wenn die Aufgabe nicht erfüllt werden kann, sollte das System „UNSAT“ oder „NEED_CLARIFICATION“ ausgeben, anstatt gewaltsam eine unsichere Flugbahn zu generieren.

### 2.2 Aufgabentyp

Das AAAI-Hauptexperiment umfasst sechs Arten von Aufgaben:

| Geben Sie | ein Beispiel | Hauptschwierigkeiten |
|------|------|----------|
| Reichweite-vermeiden | A erreichen, B meiden | Grundlegende Erreichbarkeit und Hindernisvermeidung |
| Geordnete Wegpunkte | Erst nach A, dann nach B | Zeitliche Ordnung |
| Patrouille / Inspektion | Patrouille A, B, C | Abdeckung mehrerer Ziele |
| Zeitfensterlieferung | Innerhalb von 30 Sekunden bei A ankommen | Kontinuierliche Zeitbeschränkungen |
| Notlandung | Wenn die Straße vor Ihnen nicht erreichbar ist, gehen Sie zum nächstgelegenen Landepunkt | Konditionen und Alternativstrategien |
| Mehrdeutig / unmöglich | „Gehen Sie an diesen sicheren Ort“ oder sich gegenseitig ausschließende Einschränkungen | Aufklärung und unbefriedigende Erkennung |

### 2.3 Kernannahmen

In diesem Artikel wird nicht davon ausgegangen, dass LLM an sich zuverlässig ist. Stattdessen geht dieser Artikel davon aus, dass LLMs häufig die folgenden Fehler machen:

- Generieren Sie LTL/STL mit illegaler Syntax.
– Fehlende Sicherheitsbeschränkungen in natürlicher Sprache.
– Verweis auf eine Entität, die in der Karte nicht vorhanden ist.
- Geben Sie eine Aufgabenfolge an, die dem Text genügt, aber nicht ausführbar ist.
- Verletzung von Mindestentfernungs-, Höhen- oder Zeitfensterbeschränkungen auf kontinuierlichen Flugbahnen.

Die Kernannahme von VERA-UAV lautet: **Wenn der Verifizierer diese Fehler in strukturierte Gegenbeispiele, ungesättigten Kern und Robustheitsrückmeldung umwandeln kann, ist die Korrekturerfolgsrate von lokalem Open-Source-LLM deutlich höher als die von reinen sofortigen Wiederholungsversuchen; Wenn das System außerdem den Symbolaufzählungs-Fallback innerhalb des begrenzten DSL beibehält, kann der Algorithmus eine relative Vollständigkeit erreichen, anstatt die Vollständigkeit auf der LLM-Zuverlässigkeit zu basieren. **

---

## 3. Verwandte Arbeiten und zitierfähige Arbeiten

### 3.1 Überblick über die LiteraturkarteEines der Probleme mit Version 1 besteht darin, dass es zu wenige Referenzen gibt und Rezensenten leicht denken können, dass es sich nur um eine UAV-Anwendung handelt, die auf Lang2LTL / LTLCodeGen basiert. v2 erweitert verwandte Arbeiten auf fünf Bereiche: natürliche Sprache zu zeitlicher Logik, LLM-Planung und Selbstheilung, STL/formale Verifizierung, Abschirmung und Sicherheitsagent, UAV-VLN und Anwendungen in geringer Höhe. Die folgende Tabelle listet **37 Dokumente mit hoher Relevanz** auf, die jeweils in diesem Artikel zitiert werden.| Nummer | Literatur | Veranstaltungsort / Status | Beziehung zu diesem Artikel |
|------|------|----------------|------------|
| [1] | Lang2LTL | CoRL 2023/PMLR | Direkter Ausgangspunkt für die NL-zu-LTL-Erdung |
| [2] | NL2LTL | AAAI 2023 Demo | Vorlage/Tool-Grundlinie |
| [3] | LTLCodeGen | IROS 2025/arXiv | Die stärkste direkte Baseline-Syntax mit garantierter Codegenerierung |
| [4] | KonformNL2LTL | arXiv 2025 | Referenz zur Übersetzungsglaubwürdigkeit und zum Ablehnungsmechanismus |
| [5] | NL2SpaTiaL | arXiv 2025/2026 | Strukturierter Logikbaum und Inspiration für räumliche Beziehungen |
| [6] | T3-Planer | arXiv 2025 | Selbststudium formaler LLM + STL-Bewegungsplanung direkter Wettbewerb |
| [7] | WÄCHTER | arXiv 2025/2026 | Mehrschichtige formale Sicherheitsbewertung |
| [8] | LogicGuard | arXiv 2025 | Zeitlogikkritiker und Generierung von Sicherheitsbeschränkungen |
| [9] | Pro2Guard | arXiv 2025 | probabilistische Laufzeitüberwachung |
| [10] | Generalisierte Planung in PDDL-Domänen mit LLMs | AAAI 2024 | Der Wert des Verifizierer-/Debugging-Feedbacks für die Planung |
| [11] | Kritische Untersuchung der LLM-Planung | NeurIPS 2023 | Erklären Sie, dass LLM über begrenzte direkte Planungsmöglichkeiten verfügt |
| [12] | LLM+P | arXiv 2023 | Rahmenreferenz für LLM + klassischer Planer |
| [13] | PlanBench | NeurIPS 2023Datensätze und Benchmarks | LLM-Planungs-Benchmark-Designreferenz |
| [14] | ReAct | ICLR 2023 | Grundlinie der Argumentations-Aktionsschleife |
| [15] | SayCan | CoRL 2022 | Affordance-basierte LLM-Planungsgrundlage |
| [16] | Code als Richtlinien | ICRA 2023 | LLM generiert ausführbare Programmrichtlinien |
| [17] | ProgPrompt | ICRA 2023 / Autonome Roboter | Generierung von Aufgabenplänen für ortsansässige Roboter |
| [18] | Auf Zeitlogik basierende reaktive Missions- und Bewegungsplanung | IEEE T-RO 2009 | Roboter-LTL-Planung klassisches Fundament |
| [19] | Synthese für Roboter | Jahresrückblick 2018 | Überprüfung der formalen Synthese und des Feedbacks zum Roboterverhalten |
| [20] | Überwachung zeitlicher Eigenschaften kontinuierlicher Signale | FORMATE/FTRTFT 2004 | STL-Startpunkt |
| [21] | Robustheit zeitlicher Logikspezifikationen | Theoretische Informatik 2009 | Grundlagen der Robustheitssemantik |
| [22] | Robuste Zufriedenheit über realwertige Signale | FORMATE 2010 | STL-Robustheitsberechnungsbasis |
| [23] | Reaktive Synthese aus STL-Spezifikationen | HSCC 2015 | AWL und Steuerungs-/Planungskopplung |
| [24] | Diagnose und Reparatur für STL Synthesist | HSCC 2016 | Spezifikation, Diagnose/Reparatur, theoretische Referenz |
| [25] | Spot 2.0 | ATVA 2016 | LTL/Omega-Automaten-Tool |
| [26] | RTAMT | STTT 2024 / arXiv 2025 | STL-Robustheitsmonitor |
| [27] | PRISM 4.0 | CAV 2011 | Tool zur probabilistischen Modellprüfung |
| [28] | Sicheres RL durch Abschirmung | AAAI 2018 | Shield garantiert, dass sichere Klassiker funktionieren |
| [29] | Probabilistische Abschirmung | AAAI 2025 | Probabilistische Sicherheit und Abschirmung |
| [30] | LuftbildVLN | ICCV 2023 | UAV Visual Language Navigation Benchmark |
| [31] | Realistische UAV-VLN | ICLR 2025 | Realistischere UAV-VLN-Plattformen, Benchmarks und Methoden |
| [32] | ASMA | RA-L/arXiv 2024 | Referenz zu CBF-Sicherheitseinschränkungen in UAV-VLN |
| [33] | LogistikVLN | arXiv 2025 | Anwendungsszenario für Sprachnavigation in geringer Höhe |
| [34] | UAV-VLN-Umfrage | arXiv 2026 | UAV-VLN-Forschungsfahrplan und Herausforderungen |
| [35] | Qwen3 Technischer Bericht | arXiv 2025 | Grundlage für die Auswahl lokaler Open-Source-Modelle |
| [36] | DeepSeek-R1 | arXiv 2025 | Grundlage für die Auswahl inferenzieller Open-Source-Modelle |
| [37] | vLLM/PagedAttention | SOSP 2023 | Basis für die Implementierung lokaler Inferenzen mit mehreren Modellen |### 3.2 Wesentliche Lücken in der bestehenden Arbeit

Lang2LTL, NL2LTL, LTLCodeGen und ConformalNL2LTL zeigen gemeinsam, dass NL-zu-LTL keine leere Richtung mehr ist [1-4]. Daher kann Paper E nicht einfach behaupten: „Wir übersetzen natürliche Sprache in LTL“. Die wirklich möglichen Unterschiede sind:

1. **Erweiterung von der Übersetzungskorrektheit zur Ausführungskorrektheit**: LTLCodeGen kümmert sich bereits um die Syntaxkorrektheit und Pfadgenerierung [3], aber die Höhe, Geschwindigkeit, Hindernisentfernung und das Zeitfenster des UAV erfordern STL-Robustheit und nicht nur die Gültigkeit der LTL-Formel.
2. **Erweitern Sie von der Einzelgenerierung auf die Verifizierung und Fehlerkorrektur im geschlossenen Regelkreis**: T3 Planner, LogicGuard, SENTINEL und Pro2Guard erklären, dass formelles Feedback zu einem Hotspot für die verkörperte LLM-Sicherheit wird [6-9]. VERA-UAV muss Gegenbeispiele, Unsat Core und Robustness Trace expliziter als Reparatursignale behandeln.
3. **Erweiterung von der LLM-Heuristik auf relativ vollständige Algorithmen**: Die LLM-Selbstheilung selbst ist nicht nachweislich vollständig; Vollständigkeit muss durch begrenztes DSL, entscheidbare Verifizierer und symbolische Aufzählungs-Fallbacks erreicht werden, nicht durch das Modell „könnte richtig denken“.
4. **Erweiterung von der terrestrischen Navigation auf UAVs in geringer Höhe**: Die realistische UAV-VLN-Arbeit von AerialVLN und ICLR 2025 betont die Unterschiede zwischen UAVs und terrestrischen VLNs: dreidimensionale Bewegung, kontinuierliche Dynamik, Luftraumsicherheit und Ressourcenbeschränkungen [30,31]. Genau das ist die Motivation hinter VERA-UAVs Einsatz von STL.

### 3.3 Einschränkungen bei Einreichung und Journalerweiterung

Die offizielle Beschreibung des AAAI-26 Main Technical Track verlangt, dass der Haupttext bis zu 7 Seiten technischen Inhalts umfasst und dass der Autor eine Reproduzierbarkeits-Checkliste ausfüllen muss [38]. Daher muss sich die AAAI-Version auf Methoden, Kernexperimente und Reproduzierbarkeit konzentrieren und darf nicht zu viele systemtechnische Inhalte erweitern.Der Umfang von T-ITS umfasst Sensorik, Kommunikation, Steuerung, Planung, Design und Implementierung in modernen Transportsystemen sowie methodische Richtungen wie künstliche Intelligenz und erfordert eine Erweiterung der Zeitschrift, um klare neue Beiträge im Vergleich zu Konferenzbeiträgen zu erhalten [39]. Daher sollten nachfolgende ITS-Journalausgaben Metriken des städtischen Tiefflugverkehrssystems wie Luftraumnutzung, Missionsdurchsatz, Multi-UAV-Koordination, Kommunikationslatenz und Betriebssicherheitsgewinne hinzufügen.

---

## 4. Vorgeschlagener Algorithmus: VERA-UAV

### 4.1 Gesamtprozess

Der vollständige Name von VERA-UAV wird vorläufig wie folgt ermittelt:

**VERA-UAV: Verifizierungsgestützte Reparatur für autonome UAV-Sprachplanung**

Der Systemprozess ist wie folgt:

```text
Natural-language UAV instruction
        ↓
Local open-source LLM
        ↓
Typed TaskIR
        ↓
TaskIR-to-LTL/STL compiler
        ↓
Spot / RTAMT / optional PRISM verification
        ↓
Counterexample + unsat core + robustness feedback
        ↓
LLM repair + symbolic enumerative fallback
        ↓
A* / RRT* / MPC-lite trajectory generation
        ↓
Final trajectory verification
        ↓
Executable trajectory or UNSAT / NEED_CLARIFICATION
```

Im Vergleich zu v1 ist die wichtigste Änderung in v2 die Hinzufügung eines **symbolischen enumerativen Fallbacks**: LLM ist immer noch der Hauptkandidatengenerator, aber wenn LLM in mehreren Reparaturrunden fehlschlägt, zählt das System mögliche Reparaturen innerhalb eines begrenzten TaskIR-DSL auf. Dieser Entwurf ist die Grundlage für den anschließenden Nachweis der „relativen Vollständigkeit“.

### 4.2 Typisiertes TaskIR

TaskIR ist eine strukturierte Schnittstelle zwischen natürlicher Sprache und formaler Logik. Dadurch wird vermieden, dass LLM beliebige LTL/STL-Strings direkt ausgibt, wodurch Syntaxfehler und Entity-Grounding-Fehler reduziert werden.

Das TaskIR-Feld ist wie folgt aufgebaut:| Feld | Bedeutung | Beispiel |
|------|------|------|
| „Entitäten“ | An der Direktive beteiligte Objekte | „building_3“, „hospital_zone“, „landing_pad_A“ |
| „Ziele“ | Zu erreichende Ziele | `reach(landing_pad_A)` |
| „vermeiden“ | Bereiche, die vermieden werden müssen | `avoid(hospital_zone)` |
| `Sequenz` | Unterzielsequenz | `inspect(B3_east) -> land(A)` |
| `metric_bounds` | Kontinuierliche Einschränkungen | `Höhe in [20,120]`, `distance_to_obstacle >= 10` |
| `time_windows` | Zeitfenster | „Reichweite(A) innerhalb von 30 Sekunden“ |
| „Fallbacks“ | Alternative Strategien | `Wenn blockiert, erreichen Sie das nächstgelegene_sichere_Pad` |
| „Unsicherheit“ | Mehrdeutige oder fehlende Felder | `NEED_CLARIFICATION(target="sicherer Ort")` |

### 4.3 Kompilierung von TaskIR zu LTL/STL

LTL wird verwendet, um diskrete Timing-Strukturen auszudrücken:

$$
\varphi_{\text{LTL}} =
G(\neg Kollision) \wedge F(reach(goal)) \wedge G(\neg enter(no\_fly\_zone))
$$

STL wird verwendet, um kontinuierliche Signalbeschränkungen auszudrücken:

$$
\varphi_{\text{STL}} =
G_{[0,T]}(d_{\text{obs}}(t) \ge d_{\min})
\keil
G_{[0,T]}(h_{\min} \le h(t) \le h_{\max})
\keil
F_{[0,30]}(erreichung(ziel))
$$

Dabei ist $d_{\text{obs}}(t)$ die Entfernung vom UAV zum nächsten Hindernis und $h(t)$ die Flughöhe. Robustheit der RTAMT- oder gleichwertigen STL-Monitorausgabe:$$
\rho(\tau, \varphi_{\text{STL}}) > 0
$$

Zeigt an, dass die Flugbahn $\tau$ die Spezifikation erfüllt; Wenn $\rho \le 0$, gibt der Prüfer die Verletzungsklausel, die Verletzungszeit und die minimale Sicherheitsmarge zurück.

### 4.4 Gegenbeispiel-Treiberreparatur

Anstatt nur „bestanden/nicht bestanden“ zurückzugeben, gibt der Validator eine strukturierte Diagnose zurück:

```json
{
  "status": "FAILED",
  "stage": "STL_ROBUSTNESS",
  "violated_clause": "G[0,T](distance_to_obstacle >= 10)",
  "counterexample_trace": [
    {"t": 14.2, "x": 38, "y": 51, "z": 30, "distance_to_obstacle": 6.4}
  ],
  "robustness": -3.6,
  "repair_hint": "Increase safety margin or route around building_7 west side."
}
```

Die Reparaturaufforderung von LLM erfordert kein freies Spielen, erfordert jedoch nur die Änderung relevanter Felder in TaskIR:

```text
你生成的 TaskIR 在 STL 验证中失败。
失败子句：G[0,T](distance_to_obstacle >= 10)
反例：t=14.2s 时距离 building_7 仅 6.4m。
请只修改 route constraint 或 safety margin，不要改变用户原始目标。
输出新的 TaskIR JSON。
```

Der Schwerpunkt dieses Entwurfs liegt darauf, den Suchraum von LLM zu reduzieren und das Reparaturverhalten erklärbar, aufzeichbar und reproduzierbar zu machen.

Wenn die LLM-Reparatur nach aufeinanderfolgenden $K_{\mathrm{LLM}}$-Runden fehlschlägt, wird der Symbolaufzählungs-Fallback eingegeben. Der Aufzählungsbereich wird durch die TaskIR-DSL-Tiefe, den Kartenentitätssatz, die zulässige Einschränkungsvorlage und den maximalen Aufgabenhorizont begrenzt. Der Enumerator priorisiert die Erweiterung der relevantesten Felder basierend auf Diagnoseergebnissen, wie z. B. Sicherheitsabstand, Umleitungsseite, Zeitfenster, Zielsequenz und Fallback-Landeplatz.

### 4.5 Trajektoriengenerierung

Die AAAI-Version verwendet einen leichten, reproduzierbaren Flugbahngenerator:

- 2D-Raster A*: für grundlegende Reichweitenvermeidungs- und sequentielle Aufgaben.
- 3D-Raster A*: Wird für Höhenniveaus und städtische Korridore in geringer Höhe verwendet.
- RRT*: zur kontinuierlichen räumlichen Zusatzverifikation.
- MPC-lite/Trajektorienglättung: Wird verwendet, um zu überprüfen, ob Wenderadius, Geschwindigkeitsänderung und Höhenänderung vereinfachte Dynamikbeschränkungen erfüllen.

Der Flugbahngenerator ist nicht die Neuerung dieses Artikels. Seine Funktion besteht darin, das Spezifikationsübersetzungsproblem auf die Ebene zu bringen, „ob der ausführbare Track wirklich existiert“.

---

## 5. Nachweis theoretischer Eigenschaften und relativer Vollständigkeit

v1 sagt nur „Verifizierungsfehlerkorrektur kann die Zuverlässigkeit verbessern“, aber es gibt keine mathematische Grenze. v2 macht die algorithmischen Eigenschaften deutlich: VERA-UAV behauptet nicht, dass LLM selbst vollständig ist, sondern behauptet vielmehr, dass es **relative Vollständigkeit** unter den Annahmen eines endlichen DSL, eines entscheidbaren Verifizierers und eines vollständigen zugrunde liegenden Planers hat.

### 5.1 Formaler Rahmen

Diskretisieren Sie die städtische Karte in geringer Höhe in eine begrenzt gewichtete Karte:

$$
G=(V,E,w), \quad |V|<\infty, \quad |E|<\infty.
$$Jeder Knoten $v\in V$ trägt eine Reihe atomarer Vorschläge $L(v)$, wie zum Beispiel „Ziel_A“, „Gebäude_7_Margin“, „Flugverbotszone“, „Altitude_Layer_3“. Trajektorien sind endliche Folgen:

$$
\tau = (v_0, v_1, \ldots, v_T), \quad (v_t,v_{t+1})\in E.
$$

TaskIR DSL ist als eingeschränkte Syntax definiert:

$$
\mathcal{D}_{H,D} = \{\psi: \mathrm{Tiefe}(\psi)\le D,\ \mathrm{Horizont}(\psi)\le H,\ \mathrm{Entitäten}(\psi)\subseteq \mathcal{E}(\mathcal{M})\}.
$$

Der Compiler $C$ kompiliert TaskIR gemäß der LTL/STL-Spezifikation:

$$
C(\psi)=(\varphi_{\mathrm{LTL}},\varphi_{\mathrm{STL}}).
$$

Der Prüfer $V$ bestimmt, ob die Trajektorien der Kandidaten den Spezifikationen entsprechen:

$$
V(\tau, C(\psi)) =
\begin{Fälle}
\mathrm{PASS}, & \tau \models \varphi_{\mathrm{LTL}}\ \land\ \rho(\tau,\varphi_{\mathrm{STL}})>0,\\
\mathrm{FAIL}(\eta), & \text{otherwise},
\end{Fälle}
$$

Dabei ist $\eta$ ein Gegenbeispiel, ein nicht gesättigter Kern oder eine Robustheitsspur.

### 5.2 Algorithmus-Pseudocode

```text
Algorithm VERA-UAV
Input: natural language x_NL, map M, initial state s0
Output: verified trajectory tau or UNSAT / NEED_CLARIFICATION

1: Q ← LLM_PROPOSE(x_NL, M)
2: Q ← TYPECHECK_AND_RANK(Q)
3: Visited ← ∅
4: for iter = 1 ... B do
5:     if Q has no unvisited candidate:
6:         Q ← Q ∪ SYMBOLIC_ENUMERATE_NEXT(D, H)
7:         if Q still has no unvisited candidate:
8:             return UNSAT
9:     ψ ← POP_UNVISITED(Q, Visited)
10:    Visited ← Visited ∪ {ψ}
11:    if ψ has missing entity or underspecified field:
12:        η ← type / grounding diagnostic
13:        Q ← Q ∪ REPAIR(ψ, η)
14:        if all remaining candidates require the same external information:
15:            return NEED_CLARIFICATION
16:        continue
17:    (φ_LTL, φ_STL) ← COMPILE(ψ)
18:    if compiler or syntax verifier fails:
19:        η ← compiler diagnostic
20:        Q ← Q ∪ REPAIR(ψ, η)
21:        continue
22:    τ ← COMPLETE_PLANNER(G, s0, φ_LTL, φ_STL)
23:    if τ exists and VERIFY(τ, φ_LTL, φ_STL) = PASS:
24:        return τ
25:    η ← counterexample / unsat core / robustness trace
26:    Q ← Q ∪ LLM_REPAIR(ψ, η)
27:    if LLM repair budget exhausted:
28:        Q ← Q ∪ SYMBOLIC_ENUMERATE(ψ, η, D, H)
29: return UNSAT
```

### 5.3 Satz 1: Terminierbarkeit

**Satz 1 (Beendigung).** Wenn der TaskIR DSL $\mathcal{D}_{H,D}$ endlich ist und der Algorithmus ein endliches Kandidatenbudget $B$ festlegt, muss VERA-UAV eine verifizierte Flugbahn, „UNSAT“ oder „NEED_CLARIFICATION“ in endlichen Schritten zurückgeben.**Beweisskizze.** Jedes Mal, wenn ein nicht besuchter Kandidat in der Warteschlange $Q$ auftaucht, wird TaskIR verwendet, um eine wiederholte Erweiterung durch „Besucht“ zu vermeiden. Die maximale Anzahl von Runden der LLM-Reparatur ist begrenzt, der Symbolaufzählungsraum $\mathcal{D}_{H,D}$ ist begrenzt und die äußere Schleife kann höchstens $B$ Mal ausgeführt werden. Daher kann der Algorithmus nicht unendlich laufen. Jeder Zweig kehrt entweder zurück oder tritt in die nächste endliche Schleife ein. Zertifizierung abgeschlossen.

### 5.4 Satz 2: Sicherheit und Zuverlässigkeit

**Theorem 2 (Solidität).** Wenn VERA-UAV eine Flugbahn $\tau$ zurückgibt, erfüllt $\tau$ angesichts des Kartenmodells, der Monitorsemantik und der Genauigkeit der Flugbahndiskretisierung die kompilierte LTL/STL-Spezifikation:

$$
\tau \models \varphi_{\mathrm{LTL}}
\quad \text{und} \quad
\rho(\tau,\varphi_{\mathrm{STL}})>0.
$$

**Beweisskizze.** Der Algorithmus gibt die Flugbahn erst zurück, nachdem die endgültige Überprüfung in Zeile 23 bestanden wurde. Die endgültige Überprüfung besteht aus der Überprüfung der LTL-Schicht und der Überprüfung der STL-Robustheit. Wenn eine Prüfung fehlschlägt, generiert der Algorithmus einfach eine Diagnose und setzt die Reparatur fort, ohne zur Flugbahn zurückzukehren. Daher erfüllen alle Rückkehrtrajektorien die oben genannten Bedingungen. Zertifizierung abgeschlossen.

### 5.5 Satz 3: Relative Vollständigkeit

**Theorem 3 (Relative Vollständigkeit).** Für Aufgabeninstanzen, die keiner externen Klärung bedürfen, gehen wir davon aus:

1. Es gibt einen äquivalenten oder ausreichend genauen TaskIR $\psi^\star \in \mathcal{D}_{H,D}$ für die Absicht des Benutzers;
2. Compiler $C$ kann semantisch erhaltene LTL/STL-Spezifikationen für alle TaskIRs in $\mathcal{D}_{H,D}$ generieren;
3. Der zugrunde liegende Planer ist mit der Suche nach Trajektorien, die $C(\psi)$ auf dem endlichen Graphen $G$ erfüllen, vollständig;
4. Der symbolische Enumerator zählt alle Kandidaten in $\mathcal{D}_{H,D}$ innerhalb einer begrenzten Zeit auf;
5. Der endgültige Validator ist für die begrenzte LTL/STL-Semantik zuverlässig.Wenn es eine Flugbahn $\tau^\star$ gibt, die $C(\psi^\star)$ erfüllt, dann wird VERA-UAV schließlich eine Flugbahn $\tau$ zurückgeben, die die Spezifikation erfüllt, wenn das Kandidatenbudget $B \ge |\mathcal{D}_{H,D}|$ ist.

**Beweisskizze.** Gemäß Annahme 4 wird der symbolische Aufzählungs-Fallback zu $\psi^\star$ aufzählen. Nach Annahme 2 bleibt $C(\psi^\star)$ semantisch. Gemäß Annahme 3 wird der zugrunde liegende Planer Trajektorien finden, die $C(\psi^\star)$ erfüllen. Gemäß Annahme 5 wird der endgültige Validator diese Flugbahn akzeptieren. Der Algorithmus gibt diese Flugbahn gemäß den Zeilen 23–24 des Algorithmus zurück. VERA-UAV ist daher unter diesen begrenzten DSL- und Modellannahmen relativ vollständig. Zertifizierung abgeschlossen.

### 5.6 Vollständigkeitsgrenze

Dieser Satz bedeutet nicht, dass VERA-UAV für jede natürliche Sprache und jede kontinuierliche Dynamik in der realen Welt absolut vollständig ist. Darin heißt es lediglich: **Solange die Zielaufgabe durch eine begrenzte TaskIR-DSL dargestellt werden kann und der zugrunde liegende Suchraum und die Verifizierungssemantik die Aufgabe abdecken, generiert VERA-UAV unweigerlich die richtige Antwort, ohne auf LLM angewiesen zu sein, und kann durch den symbolischen Fallback auch eine praktikable Lösung finden. **

Dies ist auch die zentrale theoretische Positionierung dieses Artikels im Vergleich zu Lang2LTL, LTLCodeGen und T3 Planner: LLM ist ein effizienter Vorschlagsgenerator und keine Quelle der Vollständigkeit.

---

## 6. Datenquellen und Datensatzaufbau

### 6.1 Stammdatenquelle

Das AAAI-Hauptexperiment nutzt die prozedurale Generierung städtischer UAV-Gitter-/Weltdaten und verlässt sich nicht auf AirSim oder reale Flugdaten. Dafür gibt es drei Gründe:

1. Kontrollierbar: Kann systematisch Aufgaben wie erreichbar, nicht erreichbar, mehrdeutig, widersprüchlich und enge Zeitfenster generieren.
2. Reproduzierbar: Karten, Aufgaben und Zufallsstartwerte können vollständig Open Source sein.
3. Anpassung an die Länge von AAAI: Konzentrieren Sie sich auf die Bewertung von KI-Methoden und nicht auf anspruchsvolle Simulationstechnik.

### 6.2 Kartenerstellung

Jede Karte enthält:- Rastergröße: „50x50x3“ bis „100x100x5“.
- Semantische Objekte: Gebäude, Krankenhäuser, Schulen, Logistikstationen, Landepunkte, Inspektionsflächen, Flugverbotszonen.
- Luftraumstruktur: Ebenen, Flugkorridore, vorübergehend gesperrte Bereiche.
- Dynamische Elemente: optionales Hinzufügen beweglicher Hindernisse oder temporärer Flugverbotszonen.
- Benennung im OSM-Stil: wie „hospital_zone_2“, „building_7_east_face“ werden nur als semantische Benennungsreferenz verwendet und werden im Hauptexperiment nicht herangezogen.

### 6.3 Beispielfelder

Jede Probe enthält:

| Feld | Beschreibung |
|------|------|
| `Anweisungs-ID` | Probennummer |
| `map_id` | Kartennummer |
| `natural_lingual_instruction` | UAV-Aufgaben in natürlicher Sprache |
| `entity_annotations` | Kartenentitäten werden an Direktivenentitäten | ausgerichtet
| `gold_task_ir` | Der Goldstandard für die manuelle oder Regelgenerierung TaskIR |
| `gold_ltl` | Gold Standard LTL |
| `gold_stl` | Der Goldstandard STL |
| `satisfiability_label` | „SAT“, „UNSAT“, „NEED_CLARIFICATION“ |
| `Referenztrajektorie` | Wenn SAT, geben Sie eine zulässige Flugbahn | an
| `failure_type` | Wenn dies fehlschlägt, markieren Sie den Fehlertyp |
| `oracle_cost` | Kosten für kürzesten Weg oder Flugbahn mit minimalen Kosten |

### 6.4 Datenskala

v2 empfohlene AAAI-Hauptversuchsskala:

| Geteilt | Menge | Zweck |
|------|------|------|
| Zug-/Promptpool | 800 | Wenige Beispiele, Vorlagen-Debugging |
| Entwickler | 250 | Eingabeaufforderung, Reparaturstrategie, Schwellenwertauswahl |
| Testen | 400 | Abschlussbericht |
| Stresstest | 150 | Lange Kombination, unscharf, unerfüllbar, enges Zeitfenster |

Der Testsatz kann nicht an der Eingabeaufforderungsauswahl teilnehmen. Alle Laborberichte verfügen über feste Zufallsstartwerte und Aufgabenlisten.

### 6.5 Datengenerierungsprotokoll und LeckageverhinderungDamit synthetische Benchmarks den AAAI-Prüfern standhalten können, muss die Datengenerierung vom ersten Tag an als „reproduzierbare Benchmarks“ und nicht als „Ad-hoc-experimentelle Skripte“ verwaltet werden:

1. **Zuerst den Generator einfrieren, dann den Test generieren**: Der Kartengenerator, die Aufgabenvorlage, die Sprachparaphrasierungsregeln und die Fehlerinjektionsregeln werden zuerst auf dem Entwickler debuggt, der Commit-Hash eingefroren und dann der Test/Stresstest generiert.
2. **Aufteilung nach Kartenebene**: Die Testkarte darf nicht dieselbe „map_id“, dieselben Entitätskoordinaten oder dasselbe Hindernislayout wie train/dev haben. Es dürfen nur abstrakte Aufgabentypen geteilt werden.
3. **Aufteilung nach Ebene der Entitätsbenennung**: Mindestens 30 % der Aufgaben im Test verwenden Entitätsbenennungsmuster, die nicht im Trainingssatz enthalten sind, z. B. „clinic_zone“, „sky_corridor_E2“, „temporary_pad_17“.
4. **Aufteilung nach Vorlagenkombinationsebene**: Behalten Sie einige unsichtbare Kombinationen im Test bei, z. B. „angeordnete Inspektion + Zeitfenster + Notfall-Fallback“, um zu verhindern, dass sich das Modell nur eine einzige Vorlagenzuordnung merkt.
5. **Zufälliger Seed und Manifest behoben**: Jede Aufteilung gibt „manifest.jsonl“, die Version des Aufzeichnungsgenerators, den Seed, den Karten-Hash, die Aufgabenvorlagen-ID, die Paraphrasen-ID und die Erfüllbarkeitsbezeichnung aus.
6. **Test-Prompt-Verschmutzung verbieten**: Beispiele mit wenigen Schüssen können nur aus Zug-/Prompt-Pools stammen; dev wird nur für die Auswahl von Schwellenwerten und Prompt-Strategien verwendet; Test und Stresstest werden nur einmal ausgeführt und die Ergebnisse werden gesperrt.

### 6.6 Fehlertaxonomie

Bei jeder Fehlerprobe sollten die erste Fehlerstufe und die letzte Fehlerstufe aufgezeichnet werden, um die Erklärung dessen zu erleichtern, was VERA-UAV behoben hat:| Fehlertyp | Definition | Hauptattributionsmodul |
|--------------|------|--------------|
| `syntax_error` | LTL/STL kann nicht geparst werden oder der Typ stimmt nicht überein | LLM/Compiler |
| `entity_error` | Verweis auf eine nicht vorhandene, mehrdeutige oder nicht übereinstimmende Kartenentität | Erdung |
| `semantic_miss` | Es fehlen wichtige Benutzerbeschränkungen wie Flugverbotszonen oder Zeitfenster | TaskIR-Generierung |
| `unsat_missed` | Gold ist UNSAT, aber das System gibt einen ausführbaren Plan zurück | Prüfer-/Entscheidungsrichtlinie |
| `false_unsat` | Gold ist SAT, aber der Systemfehler gibt UNSAT | aus Planer / Suchbudget |
| `ltl_violation` | Diskrete zeitliche Abfolge, Ankunft und Vermeidung sind nicht erfüllt | Planer / LTL-Compiler |
| `stl_violation` | Höhe, Distanz, Geschwindigkeit, Zeitfenster Robustheit nicht positiv | Flugbahn / STL-Monitor |
| `repair_regression` | Reparieren Sie eine Einschränkung und zerstören Sie dann die ursprünglich erfüllten Einschränkungen | Reparaturschleife |
| „Zeitüberschreitung“ | Voreingestelltes Rückschluss- oder Planungsbudget überschritten | Systembudget |

In der Abschlussarbeit wird nicht nur die durchschnittliche Punktzahl angegeben, sondern auch ein gestapeltes Histogramm der Fehlertaxonomie. Selbst wenn die Gesamtverbesserung nicht groß genug ist, kann auf diese Weise dennoch nachgewiesen werden, dass die Methode einen klaren Einfluss auf kritische Sicherheitsfehlertypen hat.

---

## 7. Experimentelle Plattform und Implementierungskonfiguration

### 7.1 Hardware

Derzeit mit 4 RTX 4090 und jeweils 24 GB Videospeicher ausgestattet. Diese Studie basiert nicht auf Closed-Source-APIs und die Hauptexperimente verwenden alle lokale Open-Source-Modelle.

### 7.2 Modell

Hauptversuchsmodell:

- Qwen3-8B: leichte lokale Modellbasis [35].
- Qwen3-14B: Mastermodell [35].
- DeepSeek-R1-Distill-Qwen-14B: Inferenz erweitertes Modell [36].

Optionale Modelle mit Kappe:- 32B quantitatives Modell, das als Anhang oder ergänzende Ergebnisse verwendet wird; nicht als Voraussetzung für AAAI-Hauptschlussfolgerungen erforderlich.

Lokale Inferenz verwendet vLLM/PagedAttention oder HuggingFace Transformers. Das PagedAttention-Design von vLLM eignet sich für Durchsatzexperimente unter mehreren Eingabeaufforderungen und mehreren Reparaturrunden [37].

### 7.3 Softwaremodule

| Modul | Kandidatentool | Funktion |
|------|----------|------|
| LLM-Schlussfolgerung | vLLM/Transformatoren | Lokale Modellinferenz |
| LTL-Validierung | Spot | LTL-Analyse, Automaten, Erfüllbarkeitsanalyse |
| STL-Überwachung | RTAMT oder selbstimplementierter Monitor | STL-Robustheit |
| Wahrscheinlichkeitsprüfung | PRISMA | Optionale Überprüfung unsicherer Umgebungen |
| Planung | A* / RRT* / MPC-lite | Trajektoriengenerierung |
| Protokollierung | JSONL + CSV | Protokollierung aller Bau-, Überprüfungs- und Reparaturrunden |

### 7.4 Betriebsprotokoll

Jede Aufgabeninstanz muss Folgendes aufzeichnen:

- Originalanleitung.
- TaskIR, LTL, STL pro Runde.
- Validator-Ausgabe.
- Eingabeaufforderung beheben.
- Endgültige Flugbahn.
- Laufzeit, Anzahl der Token und Grafikspeicherkonfiguration.
- Gepaarte Vergleichs-ID von Baseline und VERA-UAV für dieselbe Aufgabe.

Diese Aufzeichnungen dienen der AAAI-Reproduzierbarkeitscheckliste [38].

### 7.5 Parameterbudget für die Vorregistrierung

Um eine Anpassung der Parameter nach dem Experiment zu vermeiden, empfiehlt dieses Aufgabendokument, vor der ersten Runde des formellen Testens das folgende Budget festzulegen:| Parameter | Empfohlene Werte | Beschreibung |
|------|--------|------|
| `K_LLM` | 3 | Bis zu drei Runden LLM-Reparatur pro Aufgabe |
| „B“ | 256 | VERA-UAV Gesamtkandidaten-TaskIR-Budget |
| „D“ | 4 | Maximale Verschachtelungstiefe von TaskIR DSL |
| „H“ | 8 | Diskreter Aufgabenhorizont/Unterziel-Obergrenze |
| `T_plan` | 30er Jahre | Zeitüberschreitung bei der Einzelaufgabenplanung |
| `T_llm` | 20er Jahre | Einzelrunden-LLM-Inferenz-Timeout |
| Dekodierungstemperatur | 0,2 | Geringe Zufälligkeit im Hauptexperiment; Temperaturempfindlichkeit nur im Anhang angeben |
| top-p | 0,9 | Co-fixiert mit der Temperatur |
| max. neue Token | 1024 | Verhindern Sie, dass sich die unterschiedliche Ausgabelänge verschiedener Modelle auf die Laufzeit auswirkt |

Wenn diese Werte für formale Experimente geändert werden müssen, müssen die Gründe zuerst auf dev aufgezeichnet werden und dann muss die Konfiguration erneut eingefroren werden. Die Testergebnisse können Parameter nicht umgekehrt bestimmen.

---

## 8. Vergleichendes experimentelles Design

### 8.1 Baseline-Liste| Methode | Beschreibung | Zweck |
|------|------|------|
| Direkte LLM-Planung | LLM-Wegpunkt-/Aktionssequenz mit direkter Ausgabe | Prüfen Sie, ob die Klartextplanung unsicher ist |
| Planung im ReAct-Stil | Argumentations-Aktions-Schleife, keine formale Überprüfung | im Vergleich zur allgemeinen LLM-Agentenplanung [14] |
| Affordance-Filterung im SayCan-Stil | LLM-Score + machbarer Fähigkeitsfilter | Vergleiche Affordance Grounding [15] |
| Nur Eingabeaufforderung NL-zu-LTL/STL | LLM gibt LTL/STL direkt aus, ohne eingegebene IR und Überprüfungskorrekturen | Prüfen Sie die Eingabeaufforderung Projektobergrenze |
| Vorlagenbasislinie im NL2LTL-Stil | LTL basierend auf Vorlagenübereinstimmung generieren | Im Vergleich zur herkömmlichen Vorlagenmethode [2] |
| Basislinie im LTLCodeGen-Stil | LLM generiert Logikfunktionscode und kompiliert ihn dann in LTL | Überprüfen Sie die Syntax-Korrektheitsroute [3] |
| Selbstkorrektur im T3-Stil | LLM + STL-Verifizierer, mehrere Runden der Selbstkorrektur | Verglichen mit der jüngsten direkten Wettbewerbsroute [6] |
| VERA-UAV ohne Reparatur | Verwenden Sie TaskIR und überprüfen Sie es, aber führen Sie nach einem Fehler keine Reparatur durch | Getrennte Prüf- und Reparaturbeiträge |
| VERA-UAV LLM-reine Reparatur | typisierte IR + LLM-Reparatur, unsigned enum fallback | Überprüfen Sie den Beitrag des Fallbacks zur Vollständigkeit |
| VERA-UAV voll | Vollständige typisierte IR + Verifizierung + Gegenbeispielreparatur + symbolischer Fallback | Hauptmethode |

### 8.2 Hauptexperiment

Das Hauptexperiment beantwortet fünf Fragen:1. Ist es mit VERA-UAV einfacher, ausführbare Pläne zu erstellen als mit Basisplänen?
2. Reduziert VERA-UAV die Rate von Sicherheitsverletzungen?
3. Ist die STL-Robustheit von VERA-UAV deutlich höher?
4. Sind die Anzahl der Reparaturrunden und der zusätzliche Inferenzaufwand von VERA-UAV akzeptabel?
5. Verbessert der symbolische Aufzählungs-Fallback wirklich die „Wiederherstellungsrate fehlgeschlagener Aufgaben“ und die relative Vollständigkeit?

Vorschläge für die wichtigsten Ergebnistabellen:| Methode | Syntax gültig ↑ | Semantisches F1 ↑ | ESS ↑ | FSR ↓ | Mittlere Robustheit ↑ | Optimalitätslücke ↓ | Reparaturerfolg ↑ | Laufzeit ↓ |
|--------|----------------|---------------|-------|-------|-------------------|------------------|------------------|-----------|
| Direktes LLM | TBD | TBD | TBD | TBD | TBD | TBD | N/A | TBD |
| ReAct-Stil | TBD | TBD | TBD | TBD | TBD | TBD | N/A | TBD |
| SayCan-Stil | TBD | TBD | TBD | TBD | TBD | TBD | N/A | TBD |
| Nur Eingabeaufforderung | TBD | TBD | TBD | TBD | TBD | TBD | N/A | TBD |
| NL2LTL-Stil | TBD | TBD | TBD | TBD | TBD | TBD | N/A | TBD |
| LTLCodeGen-Stil | TBD | TBD | TBD | TBD | TBD | TBD | N/A | TBD |
| T3-Stil | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| VERA-UAV keine Reparatur | TBD | TBD | TBD | TBD | TBD | TBD | 0 | TBD |
| VERA-UAV LLM-reine Reparatur | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| VERA-UAV voll | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |„TBD“ in der Tabelle sind die für das Experiment einzugebenden Daten und dürfen im Auftragsschreiben nicht gefälscht werden.

### 8.3 Protokoll zur Auswertung der Versuchsergebnisse

v2 klärt die Hauptindikatoren und statistischen Beurteilungen, um das daraus resultierende Risiko zu vermeiden, „den Indikator zu melden, der gut ist“.

**Primäre Metrik 1: Executable Safety Success (ESS)**

Eine Aufgabe wird nur dann als ESS=1 gezählt, wenn sie gleichzeitig die folgenden Bedingungen erfüllt:

– Das generierte TaskIR weist keinen Typfehler auf.
- LTL/STL-kompilierbar.
- Planer findet Flugbahnen.
- Die endgültige Flugbahn besteht die LTL-Prüfung.
- STL-Robustheit ist positiv.
- Keine Kollisionen, Flugverbotszoneneintritte, Höhenverstöße oder Zeitfensterausfälle.

**Primäre Metrik 2: False Safe Rate (FSR)**

FSR misst den Anteil unsicherer oder unbefriedigender Aufgaben, die das System fälschlicherweise als sicher ausführbar einschätzt:

$$
\mathrm{FSR} = \frac{\#\{\mathrm{unsicher\ aber\ zurückgegeben\ als\ ausführbare Datei}\}}{\#\{\mathrm{alle\ zurückgegeben\ ausführbare Datei}\}}.
$$

Im AAAI-Papier sollte FSR als der kritischste negative Indikator in Richtung Sicherheit angesehen werden. Das Hauptverkaufsargument von VERA-UAV besteht nicht darin, für alle Aufgaben „Leistung“ zu haben, sondern darin, falsche Sicherheit zu vermeiden.

**Statistischer Test**

- Für binäre Indikatoren wie ESS-, FSR- und UNSAT-Erkennung verwenden Sie den gepaarten McNemar-Test.
- Für kontinuierliche Indikatoren wie Robustheit, Optimalitätslücke, Laufzeit usw. verwenden Sie den gepaarten Bootstrap-95-%-KI- und Wilcoxon-Signed-Rank-Test.
- Bei mehreren Basislinienvergleichen wird die Holm-Bonferroni-Korrektur verwendet.
- Schlussfolgerungen werden nur dann in den Haupttext geschrieben, wenn $p<0,05$ und die Effektgröße den Vorregistrierungsschwellenwert erreicht.

**Erfolgskriterien**

Die Mindestbedingungen für die Feststellung der Hauptschlussfolgerung der AAAI:1. Der ESS der VERA-UAV-Vollversion ist deutlich höher als der der Basislinie im LTLCodeGen-Stil und im T3-Stil.
2. Der FSR von VERA-UAV Full ist deutlich niedriger als bei allen reinen LLM-Basislinien.
3. Nach dem Entfernen des STL-Robustheitsfeedbacks nehmen Fehler im Zusammenhang mit kontinuierlichen Sicherheitsbeschränkungen erheblich zu.
4. Symbolischer Fallback sorgt für messbare Gewinne bei LLM-Reparaturfehlerproben.

### 8.4 Generalisierungsexperiment

Generalisierungsdimension:

- Keine Karte gesehen.
- Kein Entitätsname gesehen.
- Paraphrase in natürlicher Sprache.
- Längere Timing-Kombinationen.
- Engeres Zeitfenster.
- Erhöhung des Anteils unbefriedigter Aufgaben.

Generalisierungsexperimente konzentrieren sich auf die Meldung, ob VERA-UAV unerfüllbare oder mehrdeutige Aufgaben identifizieren kann, und nicht auf die Ausgabe von Fehlerverläufen.

### 8.5 Fallstudie

Bereiten Sie mindestens drei Visualisierungsfälle vor:

1. **Syntaxreparaturfall**: LLM-Ausgabe ist illegales STL, Spot/RTAMT meldet einen Fehler, Systemreparatur.
2. **Flugbahnsicherheitsnachweis**: LTL ist erfüllt, aber die STL-Robustheit ist negativ, und das System wird nach der Umleitung positiv.
3. **Unerfüllbarer Fall**: Benutzeranforderungen widersprechen sich und das System gibt „UNSAT“ aus.

### 8.6 AAAI-Haupttext-Diagrammplan

Der Haupttextraum von AAAI ist sehr begrenzt und die Diagramme müssen dem Kernargument dienen. Es wird empfohlen, nur fünf Arten von Diagrammen in den Haupttext aufzunehmen und für die anderen den Anhang zu verwenden:| Diagramm | Ziel | Platzierung |
|------|------|----------|
| Abbildung 1: VERA-UAV-Pipeline | Ein Blick auf den geschlossenen Kreislauf von getippter IR, Überprüfung, Reparatur und Fallback | Methode |
| Tabelle 1: Positionierungsmatrix der Kernliteratur | Beweist, dass es sich bei diesem Artikel nicht um eine einfache NL-zu-LTL-Anwendung handelt | Verwandte Arbeiten |
| Tabelle 2: Hauptexperimentergebnisse | paarweiser Vergleich von ESS, FSR, Robustheit, Laufzeit | Experimente |
| Abbildung 2: Stapeldiagramm der Fehlertaxonomie | veranschaulicht, welche Fehlertypen die Methode hauptsächlich reduziert | Experimente |
| Abbildung 3: Verlauf der Fallstudie | Zeigt, wie durch Gegenbeispiel-Feedback die negative Robustheit in eine positive | korrigiert werden kann Experimente / Anhang |

Es wird nicht empfohlen, den Eingabeaufforderungsabschnitt, die komplette DSL-Grammatik oder alle Karten-Screenshots im Hauptartikel zu vergrößern. Diese Inhalte sollten im Code-/Datenanhang platziert werden, um das Beitragsargument nicht zu verdrängen.

---

## 9. Design des Ablationsexperiments| Ablation | Variante | Zweck |
|--------|------|------|
| Entfernen Sie eingegebenes IR | Direkte LTL/STL-Generierung | Überprüfen Sie, ob eine strukturierte Zwischendarstellung die Zuverlässigkeit verbessert |
| Gegenbeispiel-Feedback entfernen | Allgemeiner Wiederholungsversuch | Überprüfen Sie, ob ein Gegenbeispiel effektiver ist als ein normaler Wiederholungsversuch |
| STL-Robustheitsfeedback entfernen | Nur LTL-Überprüfung | Die Bedeutung der Überprüfung kontinuierlicher Sicherheitsbeschränkungen |
| One-Shot-Reparatur | Maximal 1 Mal reparieren | Bewerten Sie die Vorteile von Reparaturrunden |
| iterative Reparatur | Bis zu 3 Mal reparieren | Bewerten Sie die Obergrenze mehrerer Reparaturrunden |
| Verschiedene Modellgrößen | Qwen3-8B / Qwen3-14B / DeepSeek-R1-Distill-Qwen-14B | Bewerten Sie die Beziehung zwischen Modellfähigkeit und Verifizierungsrahmen |
| UNSAT-Erkennung entfernen | Trace-Generierung erzwingen | Überprüfen Sie den Beitrag der Denial-of-Antwort-Fähigkeit zur Sicherheit |
| Symbol-Fallback entfernen | Nur LLM-Reparatur | Überprüfen Sie den Beitrag relativer Vollständigkeitskomponenten zur Wiederherstellung nach Fehlern |
| Endgültige Überprüfung des Planers entfernen | Überprüfen Sie nur Formeln, jedoch keine Trajektorien | Beweisen Sie, dass die Ausführung einer geschlossenen Schleife nicht optional ist |

Der Kern des Ablationsexperiments besteht nicht darin, „zu beweisen, dass die Komponenten wirksam sind“, sondern herauszufinden, welche Komponenten am meisten zu den Sicherheits- und Leistungsindikatoren beitragen, die AAAI-Gutachtern am meisten am Herzen liegen.

---

## 10. Bewertungsindikatoren

### 10.1 Indikatoren für die Spezifikationsgenerierung| Indikatoren | Definition |
|------|------|
| Syntaxgültigkeit | Ist LTL/STL für den Parser akzeptabel |
| Genauigkeit der Erdung des Unternehmens | Ob die Befehlsentität korrekt der Kartenentität zugeordnet ist |
| Semantisches F1 | Generieren Sie Präzision / Rückruf / F1 des TaskIR-Felds und des Gold-TaskIR |
| Semantische Übereinstimmung | Ob die generierte Spezifikation äquivalent oder annähernd äquivalent zu Gold TaskIR / Goldformel | ist
| UNSAT-Erkennungsgenauigkeit | Ob die unerfüllbare Aufgabe korrekt identifiziert wurde |
| Klärgenauigkeit | Ob die Fuzzy-Aufgabe „NEED_CLARIFICATION“ auslöst |
| Falsche Ausführrate | Der Anteil unerfüllbarer oder mehrdeutiger Aufgaben, die falsch ausgeführt werden |

### 10.2 Planungsausführungsindikatoren

| Indikatoren | Definition |
|------|------|
| ESS | Anteil der Aufgaben, die gleichzeitig Semantik, realisierbare Trajektorien, LTL, STL und Sicherheitsbeschränkungen erfüllen |
| FSR | Anteil unsicherer Aufgaben, die fälschlicherweise als sicher zur Ausführung markiert wurden |
|Mittlere STL-Robustheit |Die durchschnittliche Robustheit der endgültigen Flugbahn gegenüber der STL-Spezifikation |
| STL-Robustheit im ungünstigsten Fall | Verteilung der minimalen Robustheit pro Trajektorie |
| Mindestsicherheitsmarge | Mindestabstand von Hindernissen in der Flugbahn |
| Optimalitätslücke | $(J(\tau)-J^\star)/J^\star$ |
| Weglänge / Flugzeit | Flugbahnkosten und Flugzeit |

### 10.3 Reparatureffizienzanzeige| Indikatoren | Definition |
|------|------|
| Reparaturerfolgsquote | Reparaturerfolgsrate nach fehlgeschlagener Überprüfung |
| Fail-to-Pass-Konvertierung | Der Anteil der ersten fehlgeschlagenen Proben, die nach der Reparatur bestehen |
| Durchschnittliche Reparaturrunden | Durchschnittliche Reparaturrunden |
| Fallback-Beitrag | Anteil der LLM-Reparaturfehler, aber symbolischer Fallback-Erfolg |
| Laufzeitaufwand | Zusätzliche Zeit durch Reparaturmechanismus |
| Token-Overhead | Korrigieren Sie das durch Eingabeaufforderung und Diagnose verursachte Token-Inkrement |

### 10.4 Details zur Indikatorberechnung

Das Hauptexperiment muss die folgenden Indikatoren direkt im Code implementieren, um eine manuelle Anordnung während der Phase des Papierschreibens zu vermeiden:

**Semantisches F1**

Reduzieren Sie TaskIR in einen Satz von Einschränkungen auf Feldebene $\mathcal{C}$, wie „reach(A)“, „avoid(zone_B)“, „time_window(A,30)“. Der Vorhersagesatz sei $\hat{\mathcal{C}}$ und der Goldstandardsatz sei $\mathcal{C}^\star$:

$$
P = \frac{|\hat{\mathcal{C}}\cap \mathcal{C}^\star|}{|\hat{\mathcal{C}}|}, \quad
R = \frac{|\hat{\mathcal{C}}\cap \mathcal{C}^\star|}{|\mathcal{C}^\star|}, \quad
F1 = \frac{2PR}{P+R}.
$$

**Rate von Sicherheitsverstößen**

$$
\mathrm{SVR} =
\frac{\#\{\tau: Kollision \lor nofly \lor height\_violation \lor \rho(\tau,\varphi_{\mathrm{STL}})\le 0\}}
{\#\{\mathrm{zurückgegebene\ Trajektorien}\}}.
$$

**Optimalitätslücke**Wenn der Gold- oder Oracle-Planer die optimalen Kosten $J^\star$ angeben kann:

$$
\mathrm{Gap}(\tau)=\frac{J(\tau)-J^\star}{\max(J^\star,\epsilon)}.
$$

Wenn die Aufgabe UNSAT oder NEED_CLARIFICATION ist, wird die Optimalitätslücke nicht berechnet und separat in der Erkennungsgenauigkeit gezählt.

**Reparatureffizienz**

$$
\mathrm{FailToPass} =
\frac{\#\{\mathrm{initial\ fail,\ final\ pass}\}}
{\#\{\mathrm{initial\ fail}\}},
\quad
\mathrm{FallbackContribution} =
\frac{\#\{\mathrm{LLM\ Repair\ Fail,\ Symbolic\ Fallback\ Pass}\}}
{\#\{\mathrm{letzter\ Durchgang}\}}.
$$

Diese Formeln sollten als maschinenlesbare CSV-Felder im Experimentskript ausgegeben und nur in der Papiertabelle formatiert werden.

---

## 11. Erwartete experimentelle Schlussfolgerungen

In diesem Abschnitt handelt es sich um Erwartungen vor der Registrierung, nicht um experimentelle Ergebnisse.

### 11.1 Haupterwartungen

Es wird erwartet, dass die VERA-UAV-Vollauslastung bei ESS höher ist als alle Basiswerte und bei FSR/Sicherheitsverstößen niedriger ist. Der Grund dafür ist, dass Baseline normalerweise nur die lokale Korrektheit der Sprache zur Spezifikation optimiert, während VERA-UAV in den geschlossenen Regelkreis einbezieht, „ob die Spezifikation eine sichere Flugbahn erzeugen kann“.

### 11.2 Gegenbeispiel-Feedback-Erwartungen

Es wird erwartet, dass durch Gegenbeispiel-Feedback der Anteil nicht ausführbarer Pläne deutlich reduziert wird. Im Vergleich zu generischen Wiederholungsversuchen können strukturierte Gegenbeispiele LLM mitteilen, welche Klausel, welcher Zeitpunkt und welche Entität den Fehler verursacht hat, wodurch ungerichtete Wiederholungsversuche reduziert werden.

### 11.3 Typisierte IR-Erwartungen

Es wird erwartet, dass typisiertes IR die semantische Konsistenz und Interpretierbarkeit verbessert. Die direkte Generierung von LTL/STL ist anfällig für fehlende Klammern, Operatoren, Entitätsverweise und Einschränkungen; TaskIR deckt diese Fehler im Voraus als fehlende Felder oder Typfehler auf.

### 11.4 STL-Robustheit erwartetEs wird erwartet, dass das STL-Robustheitsfeedback für kontinuierliche Sicherheitsbeschränkungen am kritischsten ist. Die LTL-Schicht kann diskrete Eigenschaften wie „endgültige Ankunft“ und „Flugverbotszone vermeiden“ nachweisen, kann jedoch Flughöhe, Mindestentfernung und Zeitfensterspielraum nicht vollständig ausdrücken. Die STL-Robustheit kann quantifizierte Sicherheitsgrenzen liefern und ist der entscheidende Punkt, der UAV von gewöhnlichen Bodennavigationsaufgaben unterscheidet.

### 11,5 Erwartete Modellgröße

Es wird erwartet, dass stärkere lokale Modelle die anfängliche TaskIR-Qualität verbessern, aber das Validierungs-Reparatur-Framework ist auch für kleinere Modelle hilfreich. Mit anderen Worten, in diesem Artikel sollte der Beitrag nicht als „ein bestimmtes großes Modell ist stärker“ geschrieben werden, sondern als „Der Überprüfungsfehlerkorrekturmechanismus verbessert die Zuverlässigkeit verschiedener Open-Source-Modelle“.

---

## 12. Während der Selbstprüfung und v2-Korrekturen entdeckte Probleme

### Hauptprobleme mit 12.1 v1

1. **Unzureichende Literaturabdeckung**: v1 listet nur 12 Referenzen auf, was nicht ausreicht, um die Positionierung von AAAI zu unterstützen.
2. **Die Neuheitsgrenze ist nicht scharf genug**: v1 kann leicht als „UAV-Version NL-zu-LTL“ verstanden werden, und der Unterschied zu Lang2LTL und LTLCodeGen ist nicht stark genug.
3. **Experimentelle Indikatoren werden nicht ausreichend beurteilt**: v1 listet nur allgemeine Indikatoren auf und definiert nicht ESS, FSR, statistische Tests und Erfolgskriterien.
4. **Die Vollständigkeitsaussage ist zu schwach**: v1 erklärt nicht, warum der Algorithmus nicht rein heuristisch ist.
5. **Das Risiko synthetischer Daten wurde nicht ausreichend gemindert**: Version 1 erklärt nicht, warum synthetische Daten immer noch die methodischen Schlussfolgerungen des AAAI stützen.

### Reparaturstrategie für 12.2 v2

1. Erweitern Sie die Liste auf mehr als 30 relevante Dokumente und verwenden Sie eine Literaturmatrix, um die Beziehung zwischen jedem Artikel und diesem Artikel zu verdeutlichen.
2. Grenzen Sie den Beitrag von „Übersetzung“ auf „Ausführung geschlossener Regelkreis + STL-Robustheit + Gegenbeispielreparatur + relativ vollständiger Fallback“ ein.
3. Definieren Sie reproduzierbare Indikatoren wie ESS, FSR, Optimalitätslücke, Fail-to-Pass-Konvertierung usw.
4. Geben Sie den Satz von Terminierung, Sicherheit, Zuverlässigkeit und relativer Vollständigkeit an und machen Sie deutlich, dass Vollständigkeit von endlichem DSL und symbolischer Aufzählung herrührt, nicht von LLM.
5. Fügen Sie AirSim/reale Logistik in die T-ITS-Erweiterung ein, und der AAAI-Hauptartikel hält an der methodischen Positionierung des synthetisch kontrollierten Benchmarks fest.

### 12.3 19.05.2026 Zweite Selbstprüfung und StärkungNach der Fortsetzung der Überprüfung in dieser Runde wird davon ausgegangen, dass Papier E immer noch vier Probleme aufweist, die für die Gutachter leicht zu stellen sind, und die entsprechenden Einschränkungen wurden dem Aufgabenbuch hinzugefügt:

1. **Datenglaubwürdigkeit**: Es reicht nicht aus, nur „programmgenerierte Daten“ zu sagen. Es ist notwendig, das Einfrieren des Generators, die Segmentierung auf Kartenebene, die Segmentierung auf Ebene der Entitätsbenennung und die Vermeidung von Umweltverschmutzung durch Tests zu klären.
2. **Fehlererklärungskraft**: Die bloße Meldung von ESS/FSR reicht nicht aus. Die Fehlertaxonomie muss aufgezeichnet werden, um nachzuweisen, dass die Methode sicherheitsrelevante Fehler reduziert und nicht nur die Durchschnittsbewertung verbessert.
3. **Reproduzierbare Parameter**: Die bloße Verwendung von Qwen3 / DeepSeek reicht nicht aus. Sie müssen die Anzahl der Reparaturrunden, das Kandidatenbudget, die DSL-Tiefe, das Planungs-Timeout und die Dekodierungsparameter festlegen.
4. **Strategie für die Papierpräsentation**: AAAI hat nur begrenzten Platz, daher müssen Sie das Haupttextdiagramm im Voraus festlegen, da sonst die Hauptzeile leicht verstreut wird.

Diese vier Punkte ändern nichts am Kernbeitrag von VERA-UAV, können aber das Leitbild von einer „Ideenroute“ zu einem Zustand voranbringen, in dem „Experimente und Arbeiten direkt organisiert werden können“.

### 12.4 23.05.2026 Abschluss: Abschluss der AAAI-Hauptlinie

Papier E sollte als **AAAI/IJCAI-Methodenpapier** priorisiert werden, anstatt im Voraus ein vollständiges ITS-Systempapier zu schreiben. Das Kernproblem besteht darin, wie der von LLM generierte UAV-Missionsplan durch typisierte IR, zeitliche Logiküberprüfung, Gegenbeispielreparatur und symbolisches Fallback in einen ausführbaren, überprüfbaren und interpretierbaren Flugbahnplan umgewandelt werden kann.

Die erste Version des Papiers enthält nur drei Beiträge:

1. **Typisierte TaskIR**: Konvertieren Sie UAV-Anweisungen in natürlicher Sprache in Zwischendarstellungen, die auf Entitäten, Aktionen, Zeiteinschränkungen, Sicherheitseinschränkungen und Ressourceneinschränkungen untersucht werden können.
2. **LTL/STL + Prüfer + Trajektorienabschluss**: Verifiziert nicht nur die Formelsyntax, sondern auch, ob die Spezifikation eine Trajektorie generieren kann, die Sicherheitsbeschränkungen erfüllt.
3. **Gegenbeispiel/Robustheitsreparatur mit endlichem DSL-Fallback**: Zur Reparatur Gegenbeispiel, Unsat-Core und STL-Robustheitsfeedback verwenden; Wenn LLM nicht repariert werden kann, verwenden Sie eine endliche DSL-Aufzählung, um eine relative Vollständigkeit zu gewährleisten.

Versprechen Sie im Hauptartikel nicht vorab Folgendes:- Führt kein vollständiges Multi-UAV-Verkehrsmanagement durch;
- Kein wirklicher Einsatz eines Logistiksystems;
- Verlassen Sie sich nicht auf die High-Fidelity-Simulation AirSim als Hauptexperiment;
- Schreiben Sie keine ITS-Richtlinien oder Enthüllungen über das Wirtschaftssystem in geringer Höhe als AAAI-Hauptbeiträge.

Es wird empfohlen, die minimale experimentelle Matrix wie folgt einzufrieren:

| Abmessungen | Einstellungen der Erstausgabe |
|------|------------|
| Aufgabenfamilie | Patrouille, Lieferung, Inspektion, Vermeidung, zeitliche Ordnung, UNSAT / mehrdeutig |
| Karte | Prozedural generiertes Stadtnetz / Hindernis / Flugverbotszone / Ladepunkt |
| Grundlinien | Direkte LLM-Planung, ReAct / Nur Eingabeaufforderung, NL2LTL-Stil, LTLCodeGen-Stil, VERA-UAV keine Reparatur, VERA-UAV vollständig |
| Hauptindikatoren | ESS, FSR, Sicherheitsverletzungsrate, Reparaturerfolg, Fail-to-Pass-Konvertierung, Laufzeit |
| Ablation | keine typisierte IR, kein Gegenbeispiel, keine STL-Robustheit, One-Shot vs. iterative Reparatur, kein symbolischer Fallback |
| Verallgemeinerung | Unsichtbare Karte, unbekannte Benennung von Entitäten, längerer Horizont, strengere Einschränkungen, UNSAT-Erkennung |

T-ITS-Erweiterungen können in nachfolgenden Versionen platziert werden: Integration der Flottenplanung von Paper B, der Stressszenarien von Paper F und der Indikatoren des Verkehrssystems in geringer Höhe. Aber die AAAI-Version muss die Fragen sauber halten, sonst wird sie sowohl von KI-Prüfern als auch von Verkehrsprüfern an ihre Grenzen gebracht.

---

## 13. Risiken und Alternativen| Risiko | Auswirkungen | Alternativen |
|------|------|----------|
| Neuheit wird nur für NL-zu-LTL-Anwendungen berücksichtigt | AAAI hat ein hohes Ablehnungsrisiko | Schwerpunkt auf STL-Robustheit, Gegenbeispielreparatur und ausführbarem Trajektorienabschluss |
| Die LTLCodeGen-Basislinie ist zu stark | Das Hauptergebnis hat nicht genügend Vorteile | Verwenden Sie kontinuierliche UAV-Einschränkungen und unerfüllbare Erkennung als Differenzierungsindikatoren |
| Unzureichende lokale Modellfunktionen | Geringe Qualität der Erstübersetzung | Verwenden Sie Qwen3-14B/DeepSeek-R1-Distill-Qwen-14B und melden Sie Reparaturgewinne |
| Datensatz gilt als zu synthetisch | Die Glaubwürdigkeit der Bewerbung ist unzureichend | Fügen Sie Benennungen im OSM-Stil und Statistiken zum Layout echter Stadtblöcke hinzu, aber verlassen Sie sich nicht auf echte Flüge |
| Die Anzahl der Reparaturrunden führt zu einer zu hohen Laufzeit | Echtzeitleistung wird in Frage gestellt | Einmaliges und bis zu drei Reparaturrunden melden, Timeout und Fallback festlegen |
| STL-Monitor ist komplex zu implementieren | Beeinflusst den Fortschritt | Implementieren Sie zuerst die zeitdiskrete STL-Teilmenge und stellen Sie dann eine Verbindung zu RTAMT | her
| AAAI fehlt Platz | Die Geschichte ist divergent | Der Haupttext enthält nur Methoden und Kernexperimente, der Anhang soll vom ITS erweitert werden |
| AAAI reagiert empfindlich auf LLM-generierte Textrichtlinien | Compliance-Risiken beim Papierschreiben | Der endgültig eingereichte Text muss vom Autor manuell umgeschrieben und überprüft werden. Die LLM-Ausgabe wird nur als Versuchsgegenstand oder interne Schreibhilfe verwendet und der unrezensierte generierte Text wird nicht direkt als Text der Arbeit verwendet [38] |
| Die relative Vollständigkeit gilt als zu starke Annahme | Der theoretische Beitrag wird abgeschwächt | Im Haupttext wird es eindeutig als relative Vollständigkeit geschrieben, und begrenztes DSL, begrenzter Horizont und vollständiger Planer werden als Theoremannahmen anstelle von absoluten Garantien in der realen Welt verwendet |
| Der Stresstest ist zu schwierig, was zu einem Rückgang der Hauptergebnisse führt | Der durchschnittliche Indikator sieht nicht gut aus | Der Haupttest und der Stresstest werden separat berichtet. Der Stresstest dient zur Analyse der robusten Grenze und wird nicht mit der Hauptschlussfolgerung im gleichen Durchschnittswert | gemischt

---

## 14. Referenzen[1] Jason Xinyu Liu, Ziyi Yang, Ifrah Idrees, Sam Liang, Benjamin Schornstein, Stefanie Tellex und Ankit Shah. „Erdung komplexer natürlichsprachlicher Befehle für zeitliche Aufgaben in unsichtbaren Umgebungen.“ *Proceedings of The 7th Conference on Robot Learning*, PMLR 229:1084-1110, 2023. URL: <https://proceedings.mlr.press/v229/liu23d.html>

[2] Francesco Fuggitti und Tathagata Chakraborti. „NL2LTL – ein Python-Paket zum Konvertieren von Anweisungen in natürlicher Sprache (NL) in Formeln der linearen zeitlichen Logik (LTL).“ *Proceedings of the AAAI Conference on Artificial Intelligence*, 37(13):16428-16430, 2023. DOI: 10.1609/aaai.v37i13.27068. URL: <https://ojs.aaai.org/index.php/AAAI/article/view/27068>[3] Behrad Rabiei, Mahesh Kumar A. R., Zhirui Dai, Surya L. S. R. Pilla, Qiyue Dong und Nikolay Atanasov. „LTLCodeGen: Codegenerierung syntaktisch korrekter zeitlicher Logik für die Roboteraufgabenplanung.“ arXiv:2503.07902, 2025; Projektseitenberichte IROS 2025. URL: <https://arxiv.org/abs/2503.07902>; <https://existentialrobotics.org/LTLCodeGen/>

[4] Jun Wang, David Smith Sundarsingh, Jyotirmoy V. Deshmukh und Yiannis Kantaros. „ConformalNL2LTL: Übersetzen von Anweisungen in natürlicher Sprache in temporale Logikformeln mit konformen Korrektheitsgarantien.“ arXiv:2504.21022, 2025. URL: <https://arxiv.org/abs/2504.21022>

[5] Licheng Luo, Kaier Liang, Yu Xia und Mingyu Cai. „NL2SpaTiaL: Generierung geometrischer räumlich-zeitlicher Logikspezifikationen aus natürlicher Sprache für Manipulationsaufgaben.“ arXiv:2512.13670, 2025; überarbeitet 2026. URL: <https://arxiv.org/abs/2512.13670>[6] Jia Li und Guoxiang Zhao. „T3 Planner: Ein selbstkorrigierendes LLM-Framework für die Roboterbewegungsplanung mit zeitlicher Logik.“ arXiv:2510.16767, 2025. URL: <https://arxiv.org/abs/2510.16767>

[7] Simon Sinong Zhan, Yao Liu, Philip Wang, Zinan Wang, Qineng Wang, Zhian Ruan, Xiangyu Shi, Xinyu Cao, Frank Yang, Kangrui Wang, Huajie Shao, Manling Li und Qi Zhu. „SENTINEL: Ein mehrstufiger formaler Rahmen für die Sicherheitsbewertung von LLM-basierten verkörperten Wirkstoffen.“ arXiv:2510.12985, 2025. URL: <https://arxiv.org/abs/2510.12985>

[8] Anand Gokhale, Vaibhav Srivastava und Francesco Bullo. „LogicGuard: Verbesserung verkörperter LLM-Agenten durch auf Temporallogik basierende Kritiker.“ arXiv:2507.03293, 2025. URL: <https://arxiv.org/abs/2507.03293>

[9] Haoyu Wang, Christopher M. Poskitt, Jun Sun und Jiali Wei. „Pro2Guard: Proaktive Laufzeitdurchsetzung der LLM-Agentensicherheit durch probabilistische Modellprüfung.“ arXiv:2508.00500, 2025. URL: <https://arxiv.org/abs/2508.00500>[10] Tom Silver, Soham Dan, Kavitha Srinivas, Joshua B. Tenenbaum, Leslie Kaelbling und Michael Katz. „Generalisierte Planung in PDDL-Domänen mit vorab trainierten großen Sprachmodellen.“ *Proceedings of the AAAI Conference on Artificial Intelligence*, 38(18):20256-20264, 2024. DOI: 10.1609/aaai.v38i18.30006. URL: <https://ojs.aaai.org/index.php/AAAI/article/view/30006>

[11] Karthik Valmeekam, Matthew Marquez, Sarath Sreedharan und Subbarao Kambhampati. „Über die Planungsfähigkeiten großer Sprachmodelle: Eine kritische Untersuchung.“ *Fortschritte in neuronalen Informationsverarbeitungssystemen*, 2023. URL: <https://arxiv.org/abs/2305.15771>

[12] Bo Liu, Yuqian Jiang, Xiaohan Zhang, Qiang Liu, Shiqi Zhang, Joydeep Biswas und Peter Stone. „LLM+P: Große Sprachmodelle mit optimaler Planungskompetenz stärken.“ arXiv:2304.11477, 2023. URL: <https://arxiv.org/abs/2304.11477>[13] Karthik Valmeekam, Matthew Marquez, Alberto Olmo, Sarath Sreedharan und Subbarao Kambhampati. „PlanBench: Ein erweiterbarer Benchmark zur Bewertung großer Sprachmodelle zur Planung und Begründung von Veränderungen.“ *Fortschritte in neuronalen Informationsverarbeitungssystemen, Datensätzen und Benchmarks Track*, 2023. URL: <https://openreview.net/forum?id=YXogl4uQUO>

[14] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan und Yuan Cao. „ReAct: Synergie zwischen Denken und Handeln in Sprachmodellen.“ *International Conference on Learning Representations (ICLR)*, 2023. URL: <https://openreview.net/forum?id=WE_vluYUL-X>

[15] Michael Ahn et al. „Tue, was ich kann, nicht was ich sage: Sprache in robotischen Errungenschaften verankern.“ *Conference on Robot Learning (CoRL)*, 2022. URL: <https://arxiv.org/abs/2204.01691>[16] Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence und Andy Zeng. „Code als Richtlinien: Sprachmodellprogramme für verkörperte Kontrolle.“ *IEEE International Conference on Robotics and Automation (ICRA)*, 2023. URL: <https://arxiv.org/abs/2209.07753>

[17] Ishika Singh, Valts Blukis, Arsalan Mousavian, Ankit Goyal, Danfei Xu, Jonathan Tremblay, Dieter Fox, Jesse Thomason und Animesh Garg. „ProgPrompt: Generieren lokalisierter Roboteraufgabenpläne mithilfe großer Sprachmodelle.“ *IEEE International Conference on Robotics and Automation (ICRA)*, 2023; erweiterte Version in *Autonomous Robots*, 2023. URL: <https://arxiv.org/abs/2209.11302>

[18] Hadas Kress-Gazit, Georgios E. Fainekos und George J. Pappas. „Zeitlogisch-logikbasierte reaktive Missions- und Bewegungsplanung.“ *IEEE Transactions on Robotics*, 25(6):1370-1381, 2009. DOI: 10.1109/TRO.2009.2030225.[19] Hadas Kress-Gazit, Morteza Lahijanian und Vasumathi Raman. „Synthese für Roboter: Garantien und Feedback für Roboterverhalten.“ *Annual Review of Control, Robotics, and Autonomous Systems*, 1:211-236, 2018. DOI: 10.1146/annurev-control-060117-105838.

[20] Oded Maler und Dejan Nickovic. „Überwachung zeitlicher Eigenschaften kontinuierlicher Signale.“ *FORMATE/FTRTFT*, 2004. DOI: 10.1007/978-3-540-30206-3_12.

[21] Georgios E. Fainekos und George J. Pappas. „Robustheit zeitlicher Logikspezifikationen für zeitkontinuierliche Signale.“ *Theoretical Computer Science*, 410(42):4262-4291, 2009. DOI: 10.1016/j.tcs.2009.06.021.

[22] Alexandre Donze und Oded Maler. „Robuste Zufriedenheit der zeitlichen Logik gegenüber realwertigen Signalen.“ *FORMATE*, 2010. DOI: 10.1007/978-3-642-15297-9_12.[23] Vasumathi Raman, Alexandre Donze, Dorsa Sadigh, Richard M. Murray und Sanjit A. Seshia. „Reaktive Synthese aus signalzeitlichen Logikspezifikationen.“ *Hybrid Systems: Computation and Control (HSCC)*, 2015. DOI: 10.1145/2728606.2728628.

[24] Shromona Ghosh, Dorsa Sadigh, Pierluigi Nuzzo, Vasumathi Raman, Alexandre Donze, Alberto L. Sangiovanni-Vincentelli und Sanjit A. Seshia. „Diagnose und Reparatur für die Synthese anhand von Signal-Temporal-Logik-Spezifikationen.“ *Hybrid Systems: Computation and Control (HSCC)*, 2016. DOI: 10.1145/2883817.2883847.

[25] Alexandre Duret-Lutz, Alexandre Lewkowicz, Amaury Fauchille, Thibault Michaud, Étienne Renault und Laurent Xu. „Spot 2.0 – Ein Framework für die Manipulation von LTL und Omega-Automaten.“ *Automatisierte Technologie zur Verifizierung und Analyse (ATVA)*, 2016. URL: <https://spot.lre.epita.fr/>[26] Tomoya Yamaguchi, Bardh Hoxha und Dejan Nickovic. „RTAMT – Laufzeit-Robustheitsmonitore mit Anwendung auf CPS und Robotik.“ *International Journal on Software Tools for Technology Transfer*, 26(1):79-99, 2024; arXiv:2501.18608, 2025. DOI: 10.1007/S10009-023-00720-3. URL: <https://arxiv.org/abs/2501.18608>; Code: <https://github.com/nickovic/rtamt>

[27] Marta Kwiatkowska, Gethin Norman und David Parker. „PRISM 4.0: Verifizierung probabilistischer Echtzeitsysteme.“ *Computer Aided Verification (CAV)*, 2011. URL: <https://www.prismmodelchecker.org/bibitem.php?key=KNP11>

[28] Mohammed Alshiekh, Roderick Bloem, Rüdiger Ehlers, Bettina Könighofer, Scott Niekum und Ufuk Topcu. „Sicheres Verstärkungslernen durch Abschirmung.“ *Proceedings of the AAAI Conference on Artificial Intelligence*, 2018. URL: <https://ojs.aaai.org/index.php/AAAI/article/view/11797>[29] Edwin Hamel-De le Court, Francesco Belardinelli und Alexander W. Goodall. „Probabilistische Abschirmung für sicheres Verstärkungslernen.“ *Proceedings of the AAAI Conference on Artificial Intelligence*, 39(15):16091-16099, 2025. DOI: 10.1609/aaai.v39i15.33767. URL: <https://ojs.aaai.org/index.php/AAAI/article/view/33767>

[30] Shubo Liu, Hongsheng Zhang, Yuankai Qi, Peng Wang, Yanning Zhang und Qi Wu. „AerialVLN: Vision-and-Language-Navigation für UAVs.“ *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2023, S. 15384-15394. URL: <https://openaccess.thecvf.com/content/ICCV2023/html/Liu_AerialVLN_Vision-and-Language_Navigation_for_UAVs_ICCV_2023_paper.html>[31] Xiangyu Wang, Donglin Yang, Ziqin Wang, Hohin Kwan, Jinyu Chen, Wenjun Wu, Hongsheng Li, Yue Liao und Si Liu. „Auf dem Weg zu einer realistischen UAV-Vision-Language-Navigation: Plattform, Benchmark und Methodik.“ *International Conference on Learning Representations (ICLR)*, 2025. URL: <https://openreview.net/forum?id=rUvCIvI4eB>; arXiv:2410.07087.

[32] Sourav Sanyal und Kaushik Roy. „ASMA: Ein adaptiver Sicherheitsmargenalgorithmus für die Vision-Language-Drohnennavigation über szenenbewusste Kontrollbarrierenfunktionen.“ arXiv:2409.10283, 2024; akzeptiert von *IEEE Robotics and Automation Letters*. URL: <https://arxiv.org/abs/2409.10283>

[33] Xinyuan Zhang, Yonglin Tian, Fei Lin, Yue Liu, Jing Ma, Kornélia Sára Szatmáry und Fei-Yue Wang. „LogisticsVLN: Vision-Language-Navigation für die Terminalzustellung in geringer Höhe auf Basis von Agenten-UAVs.“ arXiv:2505.03460, 2025. URL: <https://arxiv.org/abs/2505.03460>[34] Hanxuan Chen, Jie Zheng, Siqi Yang, Tianle Zeng, Siwei Feng, Songsheng Cheng, Ruilong Ren, Hanzhong Guo, Shuai Yuan, Xiangyue Wang, Kangli Wang und Ji Pei. „Vision-and-Language-Navigation für UAVs: Fortschritte, Herausforderungen und eine Forschungs-Roadmap.“ arXiv:2604.13654, 2026. URL: <https://arxiv.org/abs/2604.13654>

[35] Qwen-Team. „Technischer Bericht von Qwen3.“ arXiv:2505.09388, 2025. URL: <https://arxiv.org/abs/2505.09388>

[36] DeepSeek-KI. „DeepSeek-R1: Anreize für die Denkfähigkeit in LLMs durch Reinforcement Learning schaffen.“ arXiv:2501.12948, 2025. URL: <https://arxiv.org/abs/2501.12948>

[37] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang und Ion Stoica. „Effiziente Speicherverwaltung für die Bereitstellung großer Sprachmodelle mit PagedAttention.“ *ACM Symposium on Operating Systems Principles (SOSP)*, 2023. URL: <https://arxiv.org/abs/2309.06180>[38] AAAI. „AAAI-26 Main Technical Track: Call for Papers“ und „AAAI-26 Reproducibility Checklist“. 2025. URL: <https://aaai.org/conference/aaai/aaai-26/main-technical-track-call/>; <https://aaai.org/conference/aaai/aaai-26/reproducibility-checklist/>

[39] IEEE Intelligent Transportation Systems Society. „IEEE-Transaktionen auf intelligenten Transportsystemen (T-ITS): Umfang.“ URL: <https://ieee-itss.org/pub/t-its/>

---

## 15. Anhang: Aktueller AAAI-Prioritätsförderungsplan

### 15.1 Positionierung des Papiers

Die AAAI-Version wird zunächst auf ein KI-Methodenpapier eingegrenzt:

**Gegenbeispiel-geführte verifizierte Sprache-zu-STL-Planung für UAVs**

Der Kern ist nicht „LLM kann UAVs planen“, sondern: Nachdem das lokale Open-Source-LLM UAV-Missionsspezifikationen generiert hat, wird ein formaler Verifizierer verwendet, um eine Gegenbeispieldiagnose zu generieren, und dann wird das LLM dazu veranlasst, die Spezifikationen oder den Plan zu korrigieren und schließlich eine überprüfbare Flugbahn zu generieren.

### 15.2 AAAI-Beitragserklärung

AAAI befürwortet drei Beiträge:

1. Eine typisierte IR-zu-LTL/STL-UAV-Missionsspezifikations-Kompilierungskette, die Ankunft, Vermeidung, Reihenfolge, Inspektion, Zeitfenster, Höhen- und Entfernungsbeschränkungen abdeckt.
2. Eine verifizierungsgesteuerte Reparaturschleife, die Syntaxfehler, fehlende Erdung, unerfüllbare, unsichere Trajektorien und geringe STL-Robustheit in strukturiertes Gegenbeispiel-Feedback umwandelt.
3. Ein UAV-NL2STL-Benchmark, einschließlich Aufgaben in natürlicher Sprache, Karten, Goldstandardspezifikationen, ausführbaren Ablaufverfolgungen und Fehlerdiagnosebezeichnungen.

### 15.3 Zeitleiste| Zeit | Aufgabe | Ausgabe |
|------|------|------|
| 18.05.2026 bis 24.05.2026 | Vervollständigen Sie die Kernliteraturtabelle und frieren Sie das Benchmark-Schema ein | Zugehörige Arbeitstabelle + Datensatzspezifikation |
| 25.05.2026 bis 07.06.2026 | Implementieren Sie einen Karten-/Aufgabengenerator, eine goldene TaskIR/LTL/STL-Vorlage, einen Basisplaner | Datengenerierungsskript + Baseline-Planer |
| 08.06.2026 bis 21.06.2026 | Implementieren Sie den Spot/RTAMT-Verifizierer und das Gegenbeispiel-Feedback | Prüfmodul |
| 22.06.2026 bis 05.07.2026 | Führen Sie ein lokales Modell, ein Basismodell, ein vorläufiges Experiment ohne Reparatur/vollständige Reparatur durch | Erste Moderator-Ergebnistabelle |
| 06.07.2026 bis 19.07.2026 | Hauptexperiment, Ablation, Generalisierung, Fehlerfallstatistik | Komplette Versuchstabelle und Abbildungen |
| 20.07.2026 bis AAAI-Abstract-Deadline | Vollständige Zusammenfassung, Einleitung, Methode, Abbildung 1, Hauptergebnistabelle | AAAI erster Entwurf |
| AAAI-Volltext vor Ablauf der Frist | Auf 7 Seiten komprimiert, Anhang hinzufügen, Reproduzierbarkeit, anonymes Repository | Einreichungspaket |

Mit Stand vom 19.05.2026 war die offizielle CFP des AAAI-27 Main Technical Track nicht auf der offiziellen AAAI-Website abrufbar; Derzeit werden der 7-seitige technische Inhalt, die Reproduzierbarkeits-Checkliste und die Code-/Datenanhangsanforderungen des AAAI-26 Main Technical Track immer noch als Grundlage für die Inversion priorisiert [38]. Sobald das AAAI-27 CFP veröffentlicht ist, muss dieser Zeitplan so schnell wie möglich aktualisiert werden, insbesondere die Abstract-Frist, die Volltext-Frist, die Frist für ergänzende Materialien und die LLM-Richtlinien für generierte Texte.

### 15.4 Nachträgliche T-ITS-Erweiterungen

Bei einer späteren Erweiterung von AAAI zu T-ITS müssen sich die neuen Inhalte deutlich von der Konferenzversion unterscheiden. Es wird empfohlen, Folgendes hinzuzufügen:- AirSim/SUMO oder Low Altitude Logistics Digital Twin Experiment.
- Multi-UAV-Koordination und Schlichtung von Luftraumkonflikten.
- Verkehrssystemindikatoren: Missionsdurchsatz, Luftraumbelegung, Sicherheitsmarge, Liefer-/Inspektionsabschlussrate, Robustheit der Kommunikationsverzögerung.
- Edge-Bereitstellungsexperiment: Latenz-Energie-Kompromiss für 4-Bit-/8-Bit-Modelle auf Jetson oder 4090.
- Der Titel wurde von AAAIs „verifizierter Planungsmethode“ in „Sicherer UAV-Betrieb in geringer Höhe für intelligente Transportsysteme“ geändert.

---

**Versionshinweise:** Der Inhalt dieses Artikels wurde auf „v2“ aktualisiert, der Dateiname lautet jedoch weiterhin „v1-20260517“, um die Anforderung „Direkt an der V1-Version ändern“ dieser Runde zu erfüllen. Die inkrementelle Optimierung am 19.05.2026 ergänzt Datenleckprävention, Fehlertaxonomie, Parameterbudgetierung, Indikatorformeln, Diagrammplanung und AAAI-Compliance-Risiken. In der nächsten Version wird empfohlen, nach Abschluss des Datensatzschemas und der ersten Runde der Baseline-Ausführung ein Update auf „v3-YYYYMMDD“ durchzuführen, wobei der Schwerpunkt auf dem Ersetzen der „TBD“-Tabelle und der Ergänzung realer experimenteller Ergebnisse und Fehlerfälle liegt.