---
title: "Städtische UAV-Routenplanung in geringer Höhe: semantische Kartierung und Funktionsbereichsaufteilung"
description: "Sehen Sie sich den Forschungsfortschritt der semantischen Kartierung und Funktionsbereichswahrnehmung in der städtischen UAV-Routenplanung an und decken Sie dabei die neuesten Arbeiten von CVPR/ICCV/IROS/RAL 2022–2025 ab"
tags: ["UAV", "Semantische Zuordnung", "Funktionsbereichsaufteilung", "Wegplanung", "Luftraummanagement"]
category: "Tech"
pubDate: 2026-04-09
sourceHash: "08d3fe8e5bc0d4f3026e3bc685987a74bf10d34f"
---

# Städtische UAV-Routenplanung in geringer Höhe: semantische Kartierung und Funktionsbereichsaufteilung

> **Richtung vier: Semantische Zuordnung + Ribbon-Bewusstsein**
> Erweitertes Kapitel · Technische Blog-Reihe Teil 4

---

## 1. Hintergrund: Von der geometrischen Karte zur semantischen Karte

Die herkömmliche UAV-Pfadplanung basiert auf einer rein geometrischen Umgebungsdarstellung – Belegungsgitter (Occupancy Grid), Octree (Octree) oder Voxelkarte (Voxel Map). Diese Darstellungen kodieren nur, „ob der Raum flugfähig ist“ und können nicht verstehen, „wohin man fliegen soll“ und „warum er nicht fliegen kann“.

Semantische Karten führen **Szenenverständnis**-Fähigkeiten ein, die auf geometrischer Darstellung basieren: Identifizierung semantischer Informationen wie Gebäudetypen (Wohn-/Gewerbe-/Industriegebäude), Straßenniveaus, Personendichte, Funktionsbereichsgrenzen usw. Diese Fähigkeit ist für die Stadtplanung in geringer Höhe von entscheidender Bedeutung – ein UAV, das einen Geschäftsviertelplatz überquert, birgt ein völlig anderes Risiko als das Überqueren eines Schulhofs, aber eine rein geometrische Karte würde beide als gleichwertigen freien Raum behandeln.

Darüber hinaus unterteilt Functional Zoning den städtischen Luftraum in geringer Höhe in Bereiche mit unterschiedlichen Regulierungsebenen: **Kontrolle der wahren Höhe von 120 m**, Flugverbotszone, Sperrgebiet, Kontrollgebiet usw. Semantisches Bewusstsein ermöglicht es UAVs, diese Regulierungsregeln proaktiv zu verstehen und einzuhalten, anstatt sich ausschließlich auf vorkommentierte statische Flugverbotszonenkarten zu verlassen.

---

## 2. Grundlagen der semantischen Abbildung: Wahrnehmung → Verstehen

### 2.1 Semantische Segmentierung: vom Pixel zum Szenenverständnis

Die semantische Segmentierung ist die zentrale Wahrnehmungsbasis der semantischen Zuordnung. Bei einem gegebenen Bild $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$ gibt das semantische Segmentierungsmodell pixelweise Klassenbeschriftungen aus:

$$
\hat{y}_{i,j} = \arg\max_{c \in \mathcal{C}} P(c | \mathbf{I}, \mathbf{p}_{i,j})
$$

Darunter ist $\mathcal{C}$ eine Reihe semantischer Kategorien (wie Gebäude, Straßen, Vegetation, Fahrzeuge, Menschen, Himmel) und $\mathbf{p}_{i,j}$ ist die Positionskodierung von Pixeln $(i,j)$.

**Zu den gängigen semantischen Segmentierungsarchitekturen für städtische Szenen gehören:- **DeepLabv3+** (Chen et al., CVPR 2018): Verwenden Sie Atrous Convolution, um das Empfangsfeld zu erweitern, ohne die Auflösung zu verlieren, und erfassen Sie so großflächige Strukturen wie städtische Gebäude und Straßen effektiv.
- **MaskFormer** (Cheng et al., CVPR 2022): Vereinheitlicht die semantische Segmentierung als Maskenklassifizierungsproblem, unterstützt eine beliebige Anzahl semantischer Kategorien und muss keinen festen $\mathcal{C}$ voreinstellen
- **Segment Anything Model (SAM)** (Kirillov et al., ICCV 2023): Ein von Meta vorgeschlagenes universelles Segmentierungs-Basismodell, das die Zero-Shot-Segmentierung von Punkt-/Box-/Text-Eingabeaufforderungen unterstützt und ein neues Paradigma für die semantische Zuordnung städtischer Szenen mit offenem Vokabular bietet.

### 2.2 Instanzsegmentierung und Zielerkennung

Zusätzlich zur semantischen Segmentierung unterscheidet die **Instanzsegmentierung** außerdem verschiedene Individuen ähnlicher Objekte – sie trennt jeden Fußgänger in der „Fußgängergruppe“ in eine unabhängige Instanz und bietet granulare Unterstützung für die Absichtsvorhersage und Kollisionsvermeidung.

| Methoden | Kernideen | Argumentationsgeschwindigkeit | Repräsentative Arbeit |
|------|---------|---------|---------|
| **Zweistufig** | Zuerst Boxen erkennen, dann Segmentmasken | ~10 FPS | Maske R-CNN (ICCV 2017) |
| **Einstufig** | Gemeinsam Masken und Kategorien vorhersagen | ~25 FPS | YOLACT (ICCV 2019) |
| **Transformatorbasiert** | Erkennung + Maske im DETR-Stil | ~15 FPS | Mask2Former (CVPR 2022) |
| **Grundlagenmodell** | SAM + Detektor | ~20 FPS | SEEM (CVPR 2024) |

**YOLO-Serie** (Ultralytics YOLOv8, 2023) wird häufig in der semantischen UAV-Echtzeitwahrnehmung verwendet – sie kann auf Jetson Orin eine Erkennungsbildrate von 50+ FPS erreichen, mit einer Latenz von $< 20\text{ms}$, was für die Echtzeitwahrnehmungsanforderungen von Flugsteuerungssystemen geeignet ist.

### 2.3 Tiefenschätzung: 2D → 3D-GeometrieFür die semantische Zuordnung müssen semantische 2D-Labels in den 3D-Raum übertragen werden. **Monokulare Tiefenschätzung** bietet Konvertierungsfunktionen von RGB-Bildern in dichte Tiefenkarten:

$$
\hat{D} = \mathcal{D}_\phi(\mathbf{I}), \quad D: \text{Pixel} \rightarrow \mathbb{R}^+
$$

Zu den wichtigsten Methoden gehören:

- **MiDaS** (Ranftl et al., NeurIPS 2020): nutzt Multi-Dataset-Training (gemischte überwachte + unüberwachte Tiefe), schneidet gut bei der Null-Stichproben-Generalisierung ab und ist derzeit das am weitesten verbreitete Basismodell für die monokulare Tiefenschätzung.
- **Depth-Anything** (Yang et al., arxiv 2024): Nutzung einer groß angelegten annotationsfreien Bildverbesserung auf Basis von MiDaS, um eine höhere Tiefengenauigkeit in städtischen Szenen zu erreichen
- **DPT** (Ranftl et al., ICCV 2021): Auf ViT basierende Transformer-Architektur gibt direkt hochauflösende Tiefenkarten aus

In Kombination mit den kamerainternen Parametern $(f_x, f_y, c_x, c_y)$ können die 2D-Pixelkoordinaten $(u, v)$ und die Tiefe $D(u, v)$ in 3D-Punkte zurückprojiziert werden:

$$
\mathbf{X} = D(u,v) \cdot \mathbf{K}^{-1} \cdot [u, v, 1]^\top, \quad \mathbf{K} = \begin{pmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{pmatrix}
$$

---

## 3. Städtische Funktionsgebietsaufteilung und Luftraumklassifizierung in geringer Höhe

### 3.1 Unterschiede in den Flugbeschränkungen in städtischen Funktionsräumen

Der städtische Raum ist je nach Art der Nutzung in verschiedene Funktionsbereiche unterteilt, und der Grad der Einschränkungen für den UAV-Flug in jedem Bereich variiert erheblich:| Funktionsbereich | Typische Szenarien | Flugbeschränkungen | Hauptrisiken |
|--------|---------|---------|---------|
| **Wohngebiet** | Wohngebiet | Höhenbeschränkungen (< 30 m), zeitliche Beschränkungen | Datenschutzverletzung, Lärmbeschwerden |
| **Geschäftsviertel** | CBD, Einkaufszentren | Fliegen in Sichtweite | Dichtes Gedränge, Signalstörungen |
| **Industriegebiet** | Fabriken, Lager | Mögliche Flugverbotszonen | Elektromagnetische Störungen, schwere Fahrzeuge |
| **Schule/Krankenhaus** | Grund- und weiterführende Schulen, Krankenhäuser | Strenges Flugverbot oder Genehmigungssystem | Sicherheitsrelevant |
| **Verkehrsknotenpunkte** | In der Nähe von Bahnhöfen und Flughäfen | Totales Flugverbot | Flugsicherheit |
| **Park/Grünfläche** | Stadtpark | Relativ entspannt (genehmigungspflichtig) | Menschenmenge |

### 3.2 Klassifizierungssystem für Lufträume in geringer Höhe

Die „Interim Regulations on the Management of Unmanned Aircraft Flights“, herausgegeben von der Zivilluftfahrtbehörde Chinas (gültig ab 2024), legen einen vertikalen Kontrollrahmen mit einer wahren Höhe von 120 m fest:

- **Tatsächliche Höhe unter 120 m**: Leichte UAVs ($< 250\text{g}$) können frei fliegen und erfordern eine Registrierung mit echtem Namen; Mikro-UAVs ($< 500\text{g}$) unterliegen keinen Flugqualifikationsbeschränkungen
- **Tatsächliche Höhe 120m-300m**: in der Kontrolle enthalten, Flugluftraumanwendung erforderlich
- **Fusionsluftraum für Einzelflüge**: Bestimmte Bereiche ermöglichen Fusionsoperationen von UAVs und bemannten Flugzeugen

Die semantische Kartierung erfordert die Kodierung dieser regulatorischen Einschränkungen in das Planungssystem, damit das UAV automatisch die flugfähige Höhe und die Flächengrenzen basierend auf dem Funktionsbereich, in dem es sich befindet, bestimmen kann.

### 3.3 Datenquellen zur semantischen Klassifizierung von Funktionsbereichen

Die Aufteilung städtischer Funktionsbereiche basiert auf geografischen Informationen aus mehreren Quellen:

- **OSM (OpenStreetMap)**: Geografische Open-Source-Daten, die eine grundlegende Merkmalsklassifizierung wie Straßen, Gebäude und Gewässer ermöglichen und eine wichtige vorherige Quelle für die Inferenz funktionaler Gebiete darstellen.
- **POI-Daten (Point of Interest)**: Die Amap/Baidu-Karten-API stellt POI-Daten für Städte bereit, und regionale Funktionen können durch POI-Dichte und -Typ abgeleitet werden (POIs in der Nähe von Schulen sind beispielsweise hauptsächlich Bildungseinrichtungen).
- **Fernerkundungsbilder**: Sentinel-2/Gaofen-2-Satellitenbilder liefern Makroinformationen zur Landnutzungsklassifizierung
- **Stadtplanungsdaten**: Die Landnutzungsebene (Kontrollplan) im städtischen Masterplan, die rechtliche Wirkung hat

**Multi-Source-Integrationsframework**:$$
\mathcal{F}_{\text{Zone}}(\mathbf{x}) = \alpha \cdot f_{\text{osm}}(\mathbf{x}) + \beta \cdot f_{\text{poi}}(\mathbf{x}) + \gamma \cdot f_{\text{remote}}(\mathbf{x}) + \delta \cdot f_{\text{plan}}(\mathbf{x})
$$

---

## 4. Dynamisches semantisches Verständnis: Absichtsvorhersage und Unsicherheitsquantifizierung

### 4.1 Vorhersage der Absicht von Fußgängern/Fahrzeugen

Dynamische Hindernisse (Fußgänger, Radfahrer, Fahrzeuge) in städtischen Straßen stellen eine große Bedrohung für den sicheren UAV-Flug dar. **Absichtsvorhersage** erfordert nicht nur die Vorhersage der zukünftigen Position von Hindernissen, sondern auch das Verständnis ihrer Verhaltensabsichten:

$$
\hat{\mathbf{a}}_t^{(i)} = \arg\max_{\mathbf{a} \in \mathcal{A}} P(\mathbf{a} | \mathbf{b}_{1:t}^{(i)}, \mathcal{E})
$$

Darunter ist $\mathbf{b}_{1:t}^{(i)}$ die historische Verhaltensbahn des Hindernisses $i$, $\mathcal{E}$ ist der Umgebungskontext (Ampelstatus, Zebrastreifen, Zebrastreifen usw.) und $\mathcal{A}$ ist die festgelegte Absicht (Straße überqueren, am Straßenrand warten, auf dem Bürgersteig gehen usw.).

**Social LSTM** (Alahi et al., CVPR 2016) führte erstmals Social Pooling ein, um die Interaktion von Fußgängern zu modellieren; **Trajectron++** (Salzmann et al., ICRA 2020) modellierte die Multi-Agenten-Interaktion auf Basis des Graph Neural Network (GNN) und verbesserte so die Vorhersagegenauigkeit in städtischen Kreuzungsszenen erheblich.

### 4.2 UAV-UAV-Konflikterkennung

In städtischen Korridoren in geringer Höhe können mehrere UAVs gleichzeitig betrieben werden. **Kollisionserkennung** erfordert die Vorhersage potenzieller Kollisionen in Raum und Zeit:$$
\text{Konflikt} \Leftrightarrow \exists t \in [t_{\text{start}}, t_{\text{end}}]: \|\mathbf{p}_A(t) - \mathbf{p}_B(t)\| < d_{\text{sicher}}
$$

Dabei ist $d_{\text{safe}}$ der Sicherheitsabstand (normalerweise $5\text{m}$ oder mehr), $\mathbf{p}_A(t)$, $\mathbf{p}_B(t)$ sind die vorhergesagten Flugbahnen der beiden UAVs.

**Konfliktlösungsstrategien** umfassen:
- **Regelbasierte Zuweisung**: Weisen Sie verschiedenen UAVs unabhängige Zeitfenster (Time Slots) oder Raumkorridore zu
- **Verteilte Verhandlung**: UAVs tauschen Flugbahnvorhersagen durch Kommunikation aus und arbeiten zusammen, um konfliktfreie Pfade zu planen
- **Zentralisierte Planung**: Die Bodenkontrollstation plant mehrere UAV-Flugbahnen auf einheitliche Weise

### 4.3 Unsicherheitsbewusste Planung

Bei der semantischen Klassifizierung besteht eine inhärente Unsicherheit – eine Glasfassade an einer Gebäudefassade kann fälschlicherweise als Himmel und Vegetation fälschlicherweise als Gebäude klassifiziert werden. **Unsicherheitsbewusste Planung** Beziehen Sie wahrgenommene Unsicherheit in die Entscheidungsfindung ein:

$$
\underline{\mathcal{C}} = \{\mathbf{x} : P(\text{Kollision} | \mathbf{x}) < \epsilon\}
$$

Planen Sie Flugbahnen nur in freien Bereichen mit ausreichender Zuverlässigkeit, um einen Sicherheitsspielraum für Erfassungsfehler zu reservieren. Diese Idee steht im Einklang mit Robust Optimization – Gewährleistung der Sicherheit im schlimmsten Fall unsicherer Mengen.

---

## 5. Semantikbewusste Planung: Kostenfunktionsdesign

### 5.1 Semantisch erweiterte Kostenkarte

Bei der herkömmlichen Planung wird eine geometrische Kostenkarte verwendet, und jede Gitterzelle $c_{i,j}$ kodiert nur die Kollisionswahrscheinlichkeit. **Semantic Enhanced Cost Map** überlagert die semantischen Kosten mit den geometrischen Kosten:

$$
C_{\text{total}}(i,j) = w_g \cdot C_{\text{geo}}(i,j) + w_s \cdot C_{\text{sem}}(i,j) + w_t \cdot C_{\text{temporal}}(i,j)
$$

Die semantischen Kosten $C_{\text{sem}}(i,j)$ werden entsprechend dem Funktionsbereich festgelegt, zu dem die Einheit gehört:$$
C_{\text{sem}}(i,j) = \begin{cases}
0 & \text{offener Park} \\
1 & \text{Geschäftsplatz} \\
5 & \text{Wohngebiet} \\
20 & \text{Schule/Krankenhaus} \\
+\infty & \text{Flugverbotszone}
\end{Fälle}
$$

### 5.2 Weiche Einschränkungen und harte Einschränkungen

**Harte Einschränkungen** sind physische/regulatorische Einschränkungen, die nicht verletzt werden dürfen:
- Es ist absolut verboten, innerhalb der Flugverbotszone zu fliegen
- Fliegen Sie nicht unter der Mindestsicherheitshöhe
- Der Abstand zum Hindernis darf den Sicherheitsabstand nicht unterschreiten

**Weiche Einschränkungen** sind bevorzugte Ziele, die mit Kosten übertroffen werden können:
- Versuchen Sie, Parks und nicht Wohngebiete zu überfliegen
- Versuchen Sie, in der Nähe von Gebäudewänden zu bleiben, anstatt offene Plätze zu überqueren (um Windstörungen zu reduzieren).
- Versuchen Sie, außerhalb von Zeiten mit hohem Lärmpegel zu fliegen

Semantikbewusste Planung behandelt diese beiden Arten von Einschränkungen durch **hierarchische Optimierung**: Minimierung der Kosten weicher Einschränkungen bei gleichzeitiger Erfüllung harter Einschränkungen.

### 5.3 EGPBS: Semantikbewusste Sicherheitsplanung

**EGPBS (Environment Graph-based Planning with Buffer Shrinking)** ist ein semantisches Planungsframework für städtische Szenen (Ideen abgeleitet aus IROS 2023-bezogener Forschung):

1. **Konstruktion eines Umgebungsgraphen**: Modellieren Sie die städtische Szene als Graphenstruktur $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, Knoten $\mathcal{V}$ repräsentieren semantische Bereiche (Gebäudeblöcke, Straßen, Parks) und Kanten $\mathcal{E}$ repräsentieren Verbindungsbeziehungen zwischen Bereichen
2. **Verkleinerung des Sicherheitspuffers**: In engen Bereichen von Passagen in geringer Höhe wird der semantische Sicherheitspuffer (Sicherheitspuffer) automatisch verkleinert, um den Durchgang zu ermöglichen (schmale Korridore sind weiterhin passierbar).
3. **Grafiksuche + Trajektorienoptimierung**: A* sucht nach grobkörnigen Pfaden im Umgebungsgraphen, gefolgt von einer Zeitbereichsoptimierung durch die MINCO-Trajektorienfamilie

---

## 6. Sicherheit und Compliance: STMP/LAANC-Integration

### 6.1 STMP: Raum-Zeit-RisikomatrixplanungSTMP (Spatial-Temporal Mitigation Planning) ist ein von der FAA vorgeschlagenes Rahmenwerk zur Risikobewertung von Drohnen. Es bewertet das umfassende Risikoniveau jedes Fluges durch die Analyse von Faktoren wie Bevölkerungsdichte, Flughafenentfernung und militärischen Einrichtungen im Fluggebiet.

Semantische Zuordnung kann die STMP-Auswertung direkt unterstützen:
- **Bevölkerungsdichteschicht**: Statistik der Fußgängerbevölkerungsdichte am Boden durch semantische Segmentierung $\rho_{\text{People}}(\mathbf{x})$
- **Sensible Facility Layer**: Markieren Sie Schulen, Krankenhäuser und religiöse Orte anhand von POI-Daten
- **Ebene „Luftfahrteinrichtungen“**: überlagerte Flughafenfreigabefläche und Streckenschutzzone

Umfassender Risiko-Score:

$$
R(\mathcal{T}) = \int_0^T \left( \alpha \cdot \rho_{\text{Menschen}}(\mathbf{p}(t)) + \beta \cdot I_{\text{Flughafen}}(\mathbf{p}(t)) + \gamma \cdot I_{\text{sensitive}}(\mathbf{p}(t)) \right) dt
$$

### 6.2 LAANC: Luftraumgenehmigung in Echtzeit

LAANC (Low Altitude Authorization and Notification Capability) ist ein von der FAA bereitgestelltes Echtzeit-Luftraumautorisierungssystem für Drohnen. Das UAV fragt über die UTM-Schnittstelle (UAV Traffic Management) ab, ob sich der aktuelle Standort innerhalb des autorisierten Luftraums befindet, und kann eine Echtzeitautorisierung beantragen.

Integrationspfad von semantischem Wahrnehmungssystem und LAANC:
1. UAV-Semantikkartierung zur Identifizierung des aktuellen Standortfunktionsbereichs
2. Wenn Sie sich in der Nähe der Grenze des Sperrgebiets befinden, stellen Sie einen Genehmigungsantrag beim LAANC
3. LAANC gibt den Autorisierungsstatus zurück (Genehmigt / Ausstehend / Verweigert)
4. Nachdem die Genehmigung erteilt wurde, wird das Planungssystem die Fluggenehmigung in dem Gebiet freischalten.

---

## 7. Mathematischer Rahmen: multimodale Wahrnehmungsfusion und semantische Kostenkartenkonstruktion

### 7.1 Bayesianische semantische Fusion

Der Kern der Multisensorfusion ist die Bayes'sche Inferenz. Angenommen, $z_t$ ist die semantische Beobachtung (Kamerasegmentierungsergebnis) zum Zeitpunkt $t$ und die vorherige semantische Karte ist $m$, dann lautet die hintere semantische Karte:$$
P(m | z_{1:t-1}) \propto P(z_t | m, z_{1:t-1}) \cdot P(m | z_{1:t-1})
$$

In einer praktischen Implementierung wird $P(z_t | m)$ durch einen CRF- (Conditional Random Field) oder MLP-Klassifikator modelliert, wobei räumliche Glättungsprioritäten berücksichtigt werden (benachbarte Pixel neigen dazu, ähnliche Beschriftungen zu haben).

### 7.2 Faktorgraphoptimierung des semantischen SLAM

Die gemeinsame Optimierung der semantischen Zuordnung und Positionierung wird durch den Faktorgraphen realisiert:

$$
\mathbf{x}^* = \arg\min_{\mathbf{x}, m} \sum_{i} \| \mathbf{r}_i^{\text{odom}} \|^2 + \sum_{j} \| \mathbf{r}_j^{\text{loop}} \|^2 + \sum_{k} \| \mathbf{r}_k^{\text{semantische}} \|^2
$$

Unter diesen ist $\mathbf{r}^{\text{odom}}$ das Odometrie-Residuum, $\mathbf{r}^{\text{loop}}$ das Schleifenschluss-Erkennungsresiduum und $\mathbf{r}^{\text{semantic}}$ das semantische Beobachtungsresiduum (Konsistenzbeschränkung zwischen semantischen 3D-Punkten und semantischer Karte).

Die größte Herausforderung des semantischen SLAM liegt in der Mehrdeutigkeit semantischer Beobachtungen: Die gleiche Art semantischer Bezeichnungen kann völlig unterschiedlichen geometrischen Formen entsprechen (z. B. werden Gebäude unterschiedlichen Stils mit „Gebäude“ bezeichnet), und im Faktordiagramm muss eine entsprechende Entspannung eingeführt werden.

---

## 8. Zukünftige Trends und offene Fragen

### 8.1 Großes Sprachmodell + semantisches Bewusstsein

Visual-Language-Modelle (VLMs) wie GPT-4V bringen **offenes Vokabularbewusstsein** in die semantische Abbildung – sie sind nicht mehr auf einen vordefinierten Satz geschlossener semantischer Kategorien beschränkt, sondern können beliebige semantische Konzepte verstehen, die in natürlicher Sprache beschrieben werden.

**Anwendungsszenario**: Der Benutzer sagt „Meiden Sie den Schulbereich“, VLM kann Schulmerkmale (Spielplatz, Fahnenhebeplattform, Schulschild) anhand des Bildes identifizieren; Sagt der Benutzer „Mit dem Café über die Straße fliegen“, kann VLM die Zielstraße lokalisieren. Dadurch wird die semantische Zuordnung von „passiver Abfrage“ zu „aktivem Verstehen“ verbessert.

### 8.2 Datenschutz und DatendesensibilisierungSemantische Kartierung umfasst eine große Anzahl von Bildern städtischer Umgebungen, was Bedenken hinsichtlich der Privatsphäre aufwirft (Sichtbarkeit innerhalb von Gebäuden, Aufzeichnung menschlicher Aktivitäten). Zu den technischen Reaktionsstrategien gehören:
- **Edge-Side-Verarbeitung**: Die semantische Segmentierung wird in der UAV-Onboard-Recheneinheit abgeschlossen und das Originalbild wird nicht zurück an die Bodenstation übertragen
- **Datenschutzbewusstes Rendering**: Bereiche mit Gesichtern automatisch kodieren oder entfernen
- **Federated Semantic Mapping**: Mehrere UAVs teilen semantische Kartenaktualisierungen, jedoch keine Rohbilder

---

## 9. Zusammenfassung

Semantische Kartierung hebt die urbane UAV-Planung in geringer Höhe von der **geometrischen Wahrnehmung** zum **kognitiven Verständnis**. Durch semantische Segmentierung, Tiefenschätzung und Funktionsbereichsaufteilung kann UAV verstehen, „wo fliege ich“, „warum es hier empfindlich ist“, „wie soll ich mich fortbewegen“, anstatt nur zu wissen, „gibt es hier irgendwelche Hindernisse“.

Zu den wichtigsten Forschungsrichtungen gehören: **Semantisches Bewusstsein für offenes Vokabular** (Befähigung großer Modelle), **Unsicherheitsbewusste Planung** (Umgang mit Wahrnehmungsfehlern), **STMP/LAANC-Compliance-Integration** (regulierungsgesteuerte semantische Einschränkungen). Da sich der regulatorische Rahmen für die städtische Tieflandwirtschaft weiter verbessert, werden semantische Bewusstseinsfähigkeiten zu einem Standardbestandteil städtischer UAV-Planungssysteme.

---

## Referenzen

- Cheng, B., Misra, I., Schwing, A. G., et al. (2022). MaskFormer für Semantik- und Instanzsegmentierung. *CVPR*. https://doi.org/10.1109/CVPR52688.2022.00227

- Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., ... & Girshick, R. (2023). Segmentieren Sie alles. *ICCV*.

- Ranftl, R., Lasinger, K., Hafner, D., Schindler, K. & Koltun, V. (2020). Auf dem Weg zu einer robusten monokularen Tiefenschätzung: Mischen von Datensätzen für eine datensatzübergreifende Null-Schuss-Übertragung. *IEEE TPAMI*. https://doi.org/10.1109/TPAMI.2020.3019967- Ranftl, R., Bochkovskiy, A. & Koltun, V. (2021). Vision-Transformatoren für dichte Vorhersagen. *ICCV*. https://doi.org/10.1109/ICCV48922.2021.01017

- Alahi, A., Goel, K., Ramanathan, V., Robicquet, A., Fei-Fei, L. & Savarese, S. (2016). Soziales LSTM: Vorhersage der menschlichen Flugbahn in überfüllten Räumen. *CVPR*. https://doi.org/10.1109/CVPR.2016.99

- Salzmann, T., Ivanovic, B., Chakravarty, P. & Pavone, M. (2020). Trajectron++: Dynamisch durchführbare Flugbahnvorhersage mit heterogenen Daten. *ECCV*. https://doi.org/10.1007/978-3-030-46732-6_43

- Zhou, H., Ren, D., Wu, J., et al. (2023). Egpbps: Umgebungsgraphbasierte Planung mit Pufferverkleinerung für die UAV-Navigation. *IROS*.

- Liu, Y., Chen, J., Wang, X., et al. (2023). Depth-Anything: Die Leistungsfähigkeit umfangreicher, unbeschrifteter Daten freisetzen. *arxiv:2401.10891*.

---

*Dieser Artikel ist das vierte erweiterte Kapitel einer Artikelreihe zur Routenplanung mit Drohnen in geringer Höhe in der Stadt. *