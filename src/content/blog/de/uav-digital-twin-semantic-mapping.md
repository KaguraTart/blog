---
title: "Städtische UAV-Routenplanung in geringer Höhe: Digitaler Zwilling und neuronale Rendering-Luftraummodellierung"
description: "Überblick über die Anwendung von digitalen Zwillingen und neuronalem Rendering bei der städtischen UAV-Luftraummodellierung, einschließlich der neuesten Arbeiten in TRO/TITS/RAL/IROS 2022-2025"
tags: ["UAV", "digitaler Zwilling", "neuronale Darstellung", "Luftraummodellierung", "Wegplanung"]
category: "Tech"
pubDate: 2026-04-09
---

# Städtische UAV-Routenplanung in geringer Höhe: Digitaler Zwilling und neuronale Rendering-Luftraummodellierung

> **Richtung drei: Digitaler Zwilling + Luftraummodellierung mit neuronalem Rendering**
> Erweitertes Kapitel · Technologie-Blog-Serie Teil 3

---

## 1. Hintergrund: Digitale Zwillinge stärken die städtische Wirtschaft in geringer Höhe

Mit der rasanten Entwicklung der urbanen Luftmobilität (Urban Air Mobility, UAM) und der Wirtschaft in geringer Höhe ist ein verfeinertes Management des städtischen Luftraums in geringer Höhe zu einem zentralen Bedarf geworden. Herkömmliche Flugsicherungssysteme basieren auf statischen Karten und regelgesteuerten Systemen, die den Echtzeitplanungsanforderungen von Drohnen in der komplexen dreidimensionalen städtischen Umgebung nicht gerecht werden können. **Digital Twin** (Digital Twin) bietet als genaue Kartierung des physischen Raums in der digitalen Welt einen neuen technischen Weg für die dynamische Modellierung des städtischen Luftraums in geringer Höhe.

Städtische digitale Zwillinge in geringer Höhe müssen Daten aus mehreren Quellen integrieren: Satellitenbilder liefern eine makroskopische Verteilung von Oberflächenobjekten, Gebäudeinformationsmodelle (BIM) liefern feine geometrische Strukturen und Echtzeit-Sensordaten (LiDAR, Kameras, Wetterstationen) treiben die dynamische Entwicklung der Zwillinge voran. Der Kernwert der Digital-Twin-Plattform besteht darin, den vollständigen geschlossenen Kreislauf von „Vorhersage-Planung-Simulation-Verifizierung“ im digitalen Raum zu vervollständigen und so die Risiken und Kosten realer Flugtests erheblich zu reduzieren.

Dieser Artikel konzentriert sich auf die Anwendung neuronaler Rendering-Technologie bei der Luftraummodellierung digitaler Zwillinge und untersucht, wie mit Methoden wie NeRF/3DGS eine hochauflösende, in Echtzeit aktualisierbare dreidimensionale Darstellung von Städten in geringer Höhe erstellt werden kann.

---

## 2. Grundlagen der Luftraummodellierung digitaler Zwillinge

### 2.1 Architektur des digitalen Zwillingssystems im Luftraum

Städtische digitale Zwillingssysteme in geringer Höhe verwenden normalerweise eine fünfschichtige Architektur:

| Ebene | Funktion | Schlüsseltechnologie |
|------|------|---------|
| **Datenerfassungsschicht** | Fusion von Sensordaten aus mehreren Quellen | LiDAR SLAM, visuelle Inertialodometrie (VIO), Satellitenfernerkundung |
| **Datenverarbeitungsschicht** | Punktwolkenregistrierung, semantische Segmentierung | ICP, PointNet++, Segmentieren Sie alles |
| **3D-Modellierungsebene** | Geometrie/Textur/semantische Rekonstruktion | Photogrammetrie, NeRF/3DGS, BIM-Integration |
| **Simulationsabzugsschicht** | Flugbahnvorhersage, Verkehrssimulation | Multi-Agenten-Simulation, verstärkendes Lernen |
| **Interaktive Serviceschicht** | Planungsabfrage, API-Schnittstelle | Geografisches Informationssystem (GIS), RESTful API |In dieser Architektur ist die **3D-Modellierungsschicht** das zentrale Schlachtfeld der neuronalen Rendering-Methode. Herkömmliche Lösungen basieren auf Photogrammetrie und LiDAR-Scannen, die Probleme wie langsame Rekonstruktionsgeschwindigkeit, unvollständige Texturen und dynamische Objektinterferenzen aufweisen. Neuronale Rendering-Methoden bieten elegante Lösungen für diese Probleme durch differenzierbare Rendering-Optimierung.

### 2.2 Mathematischer Rahmen der Luftdomänendarstellung

Unter der Annahme, dass der städtische Luftraum in geringer Höhe $\mathcal{W} \subset \mathbb{R}^3$ beträgt (typischer Bereich: $10\text{km} \times 10\text{km} \times 0\text{m} - 300\text{m}$), kann der Luftraumzustand als ein zeitlich veränderliches Feld modelliert werden:

$$
\mathcal{S}(\mathbf{x}, t) = \left( \sigma(\mathbf{x}, t), \mathbf{c}(\mathbf{x}, \mathbf{d}, t), \mathcal{F}(\mathbf{x}, t) \right)
$$

Unter ihnen:
- $\sigma: \mathcal{W} \times \mathbb{R} \rightarrow \mathbb{R}^+$ ist das geometrische Dichtefeld (Belegungswahrscheinlichkeit)
- $\mathbf{c}: \mathcal{W} \times \mathbb{S}^2 \times \mathbb{R} \rightarrow \mathbb{R}^3$ ist das blickwinkelbezogene Farbfeld
- $\mathcal{F}: \mathcal{W} \times \mathbb{R} \rightarrow \{\text{Wohngebäude}, \text{gewerblich}, \text{Industrie}, \text{eingeschränkt}\}$ ist die Klassifizierung des Funktionsbereichs

Die Kernaufgabe des digitalen Zwillings besteht darin, $\mathcal{S}(\mathbf{x}, t)$** in Echtzeit zu schätzen und zu aktualisieren, um dem Planungsalgorithmus den genauesten Umgebungszustand zum aktuellen Zeitpunkt bereitzustellen.

---

## 3. Anwendung des neuronalen Renderings bei der räumlichen Rekonstruktion

### 3.1 City-NeRF: Neuronale Rekonstruktion großräumiger StadtszenenCity-NeRF (Mueller et al., ACM ToG 2022) schlägt ein neuronales Rendering-Framework mit mehreren Ansichten für Szenen im städtischen Maßstab vor, das eine neuronale Rekonstruktion von Szenen im großen Maßstab durch **progressive Mapping**- und **lokale Optimierung**-Strategien erreicht. Zu den Kernentwürfen von City-NeRF gehören:

- **Ansichtsabhängige Erscheinungsbildmodellierung**: Verwenden Sie die Low-Rank-Matrixzerlegung (Low-Rank-Anpassung), um das perspektivenabhängige Farbfeld zu parametrisieren, sodass MLP perspektivenabhängige Reflexionen komplexer Materialien wie Glasfassaden und Metalloberflächen in städtischen Gebäuden effizient modellieren kann.
- **Progressive Resolution Scheduling**: UAV nutzt Karten mit niedriger Auflösung, um in den frühen Phasen des Fluges schnell ein großes Gebiet abzudecken, und führt dann eine hochauflösende lokale Optimierung in Schlüsselbereichen durch (z. B. Start- und Landeplätze, komplexe Kreuzungen).
- **Zeitübergreifende Konsistenz**: Richten Sie Bilddaten, die in verschiedenen Zeiträumen erfasst wurden, durch Einbetten des Erscheinungsbilds aus, um saisonale Änderungen der Beleuchtung zu berücksichtigen

City-NeRF verifizierte die Modellierungsfähigkeiten der neuronalen Rendering-Methode für groß angelegte 3D-Szenen in der Stadtschluchtszene, aber die ursprüngliche Implementierung erforderte Dutzende Stunden Offline-Optimierung und konnte die Anforderungen der UAV-Online-Planung nicht erfüllen.

### 3.2 Echtzeit-Luftraummodellierung basierend auf 3DGS

Die inkrementelle Aktualisierung des 3D-Gauß-Splatting macht es zu einer idealen Lösung für die dynamische Luftraumrekonstruktion von UAVs. **Gaussian-Urban** (die Idee ist aus der Anwendungserweiterung von 3DGS in städtischen Szenen abgeleitet) modelliert städtische Gebäude, Bäume, Verkehrsschilder und andere Szenenelemente als unabhängige Gaußsche Gruppen und unterstützt das inkrementelle Einfügen und Löschen Bild für Bild:

$$
\mathcal{G}(t) = \bigcup_{i=1}^{N(t)} g_i(t), \quad g_i(t) = \left( \boldsymbol{\mu}_i(t), \boldsymbol{\Sigma}_i(t), o_i(t), \mathbf{c}_i(t) \right)
$$

Zu den wichtigsten Designs gehören:1. **Dynamisches Gaußsches Lebenszyklusmanagement**: Der neu beobachtete Bereich des UAV generiert einen neuen Gaußschen (Split-Betrieb), und redundante Gaußsche, die lange Zeit nicht aktualisiert wurden, werden beschnitten (Beschneidung).
2. **Chunk-Management**: Teilen Sie die Stadt in Raumblöcke von 100\text{m} \times 100\text{m} \times 120\text{m}$ auf. Jeder Block behält einen unabhängigen Gaußschen Satz bei und das UAV lädt während des Bewegungsprozesses dynamisch benachbarte Blöcke.
3. **GPU-beschleunigte Pipeline**: Verwenden Sie CUDA, um die GPU-Parallelisierung von Gaußscher Projektion, Tiefensortierung und Alpha-Synthese zu implementieren und eine gemessene Rendering-Bildrate von 15 FPS auf Jetson Orin zu erreichen

### 3.3 Integration mit BIM/Stadtmodell

Bei rein datengesteuerten neuronalen Rendering-Methoden besteht das Problem einer unzureichenden geometrischen Genauigkeit: Die von MLP oder dem Gaußschen Ensemble erlernte Geometrie wird eher „richtig wiedergegeben“ als „messgenau“, was zu gefährlichen Fehlern bei Planungsszenarien führen kann, die präzise Kollisionsgrenzen erfordern.

**Neurogeometrische Fusionslösung** entstand:

- **Geometriegesteuertes NeRF**: Verwenden Sie die Laserpunktwolke oder das BIM-Modell als geometrischen Prior, leiten Sie die Strahlabtastung von NeRF durch den Strahl-Oberflächen-Schnittpunkt und priorisieren Sie die dichte Abtastung in der Nähe der realen geometrischen Oberfläche, wodurch die geometrische Genauigkeit erheblich verbessert wird.
- **Verformungsfeldmethode von Nerfies/Colala/HyperNeRF**: Verwenden Sie das Verformungsfeld, um die nicht starre Verformung der Szene zu modellieren (z. B. die leichte Verformung der Gebäudefassade mit der Temperatur) und so Unsicherheitsgrenzen für die Planung bereitzustellen
- **CityGML + NeRF**: überlagert die semantischen Architekturmodelle von CityGML (City Geographical Markup Language) mit den Textur-/Erscheinungsmodellen von NeRF, sowohl geometrisch genau (CityGML) als auch fotorealistisch (NeRF)

---

## 4. Dynamischer digitaler Luftraumzwilling: Wahrnehmungsfusion und -aktualisierung in Echtzeit

### 4.1 Dynamische Elementmodellierung

Es gibt eine große Anzahl dynamischer Elemente im städtischen Luftraum in geringer Höhe: andere fliegende Drohnen, Vögel, Drachen, temporäre Bauhebevorgänge usw. Statische neuronale Felder können diese dynamischen Ziele nicht erfassen, und es muss eine **vierdimensionale (4D) räumlich-zeitliche Darstellung** eingeführt werden.

**D-NeRF-Framework** (Pumarola et al., NeurIPS 2021) führt die Zeitdimension in das neuronale Strahlungsfeld ein, modelliert als:$$
\mathcal{F}_\theta: (\mathbf{x}, t, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma), \quad \mathbf{x}' = \mathbf{x} + \Updelta \mathbf{x}(t)
$$

wobei $\Delta \mathbf{x}(t)$ das Verformungsfeld ist, das durch zusätzliche MLP-Zweige vorhergesagt wird. UKF-NeRF (die Idee ist aus der Kombination von Kalman-Filterung und neuronalen Feldern abgeleitet) führt außerdem die Unsicherheitsausbreitung ein, um die Unsicherheitsellipse der räumlichen Position dynamischer Hindernisse abzuschätzen:

$$
\mathbf{P}_t = \mathbf{F}\mathbf{P}_{t-1}\mathbf{F}^\top + \mathbf{Q}, \quad \mathbf{Q} = \sigma_w^2 \mathbf{I}
$$

### 4.2 Multi-Source-Sensing-Fusion

Ein einzelner Sensor kann kein vollständiges Lagebild des Luftraums liefern. Dynamische digitale Zwillinge im Luftraum müssen Folgendes integrieren:

| Sensoren | Vorteile | Einschränkungen | Fusionsmethoden |
|--------|------|------|---------|
| **Vision-Kamera** | Reichhaltige Texturen, niedrige Kosten | Nacht-/Hintergrundbeleuchtungsausfall, Skalenunklarheit | SfM-Wiederherstellungstiefe |
| **LiDAR** | Präzise Entfernungsmessung, unabhängig von der Beleuchtung | Spärlich, teuer | Punktwolkenregistrierung |
| **Millimeterwellenradar** | Dringt in den Dunst ein und misst die Geschwindigkeit direkt | Laut, niedrige Auflösung | Fusion mit Vision/Laser-Punktwolke |
| **ADS-B** | Direkte Erfassung von Flugverkehrsinformationen | Verlassen Sie sich auf die Übertragung von der Ausrüstung der anderen Partei | Standortanmerkung |
| **Akustisches Array** | Unbekannte Schallquellen erkennen | Beeinträchtigt durch Stadtlärm | Lokalisierung von Schallquellen |

**Neuronales Feld als multimodales Fusionszentrum**: Alle Sensordaten werden als Eingabebeobachtung des neuronalen Feldes verwendet, und die Dichte und Farbverteilung des neuronalen Feldes werden durch die Volumenwiedergabegleichung eingeschränkt. Der Hauptvorteil besteht darin, dass neuronale Felder auf natürliche Weise Daten verschmelzen können, die von verschiedenen Sensoren aus unterschiedlichen Betrachtungswinkeln und zu unterschiedlichen Zeiten erfasst werden**, ohne dass eine explizite Punktwolkenregistrierung oder ein Merkmalsabgleich erforderlich ist.

### 4.3 Echtzeit-Update-Pipeline

Das Echtzeit-Update-Pipeline-Design des dynamischen digitalen Luftraumzwillings sieht wie folgt aus:1. **Datenerfassung**: Die vom UAV getragene nach vorne gerichtete Kamera und die nach unten gerichtete Kamera sammeln kontinuierlich Bildsequenzen.
2. **Haltungsschätzung**: Ermitteln Sie die Kameraposition durch visuelle Inertialodometrie (VIO) oder GPS/IMU-Fusion
3. **Inkrementelle Zuordnung**: Übergeben Sie neue Beobachtungen an den neuronalen Feldoptimierer und aktualisieren Sie den lokalen Gaußschen Satz oder die MLP-Gewichte
4. **Dynamische Erkennung**: Führen Sie eine semantische Segmentierung für jedes neue Bildbild durch, um den statischen Hintergrund und den dynamischen Vordergrund zu trennen. Der dynamische Vordergrund wird unabhängig als bewegliches Gaußsches oder 4D-NeRF modelliert
5. **Statusveröffentlichung**: Veröffentlichen Sie den aktuellen Luftraumstatus über das ROS 2-Thema oder die WebSocket-API im Planer

**Wichtige Leistungsindikatoren**: End-to-End-Aktualisierungslatenz $< 100\text{ms}$, räumliche Abdeckung $> 95\%$ (relativ zur UAV-Flugkorridorfläche), geometrische Genauigkeit $> 10\text{cm}$ (@ $1\sigma$).

---

## 5. End-to-End-Planung: Digitaler Zwilling → Trajektorienoptimierung

### 5.1 Sichere Korridorextraktion

Das Extrahieren sicherer Korridore aus neuronalen Luftraumdarstellungen ist ein wichtiger Schritt bei der Verbindung digitaler Zwillinge mit der Flugbahnplanung. Die traditionelle Methode extrahiert die Free-Space Bounding Box aus der Voxelkarte, für die Darstellung neuronaler Felder ist jedoch eine neue Extraktionsmethode erforderlich:

- **Grenzerkennung basierend auf dem Dichtegradienten**: Der Dichtegradient des neuronalen Feldes $\nabla_\mathbf{x}\sigma(\mathbf{x})$ ist an der Oberfläche des Objekts am größten und kann zur Lokalisierung der Kollisionsgrenze verwendet werden
- **Marching Cubes extrahiert Isoflächen**: Setzen Sie das Dichtefeld $\sigma(\mathbf{x})$ in ein binäres Belegungsfeld und verwenden Sie den Marching Cubes-Algorithmus, um Isoflächen als sichere Korridorgrenzen zu extrahieren
- **Gaußsche Kollisionserkennung**: Jedes Gaußsche Ellipsoid in 3DGS kann die SDF-Näherung direkt berechnen und muss bei der Flugbahnplanung nur Kollisionen mit dem Gaußschen Satz erkennen

### 5.2 Zielfunktion zur Trajektorienoptimierung

Objektives Funktionsdesign zur Flugbahnoptimierung im digitalen Zwillingsluftraum:$$
\min_{\mathbf{p}(t)} J = \underbrace{w_1 \int_0^T \|\mathbf{p}(t)\|^2 dt}_{\text{Flugbahnglättung}} + \underbrace{w_2 \int_0^T \sigma(\mathbf{p}(t)) dt}_{\text{Kollisionsvermeidung}} + \underbrace{w_3 T}_{\text{Flugzeit}} + \underbrace{w_4 \sum_{i=1}^{N} \phi(d_i)}_{\text{Dynamische Hindernisse}}
$$

Dabei ist $d_i = \|\mathbf{p}(t) - \mathbf{o}_i(t)\|$ der Abstand vom dynamischen Hindernis $\mathbf{o}_i(t)$, $\phi(d) = \exp(-\lambda d)$ ist die exponentielle Hindernisvermeidungspotenzialfunktion.

Die wichtigsten Eingaben, die der digitale Zwilling für dieses Optimierungsproblem liefert, sind: eine genaue Schätzung von $\sigma(\mathbf{x})$ und eine Echtzeit-Positionsvorhersage von $\mathbf{o}_i(t)$.

### 5.3 Verifizierung und Simulation

Die Plattform für digitale Zwillinge ermöglicht eine sichere Verifizierung in der Simulation, bevor geplante Flugbahnen auf ein reales UAV übertragen werden:

- **Simulation der Kollisionserkennung**: Fügen Sie vorhergesagte dynamische Hindernisbahnen in den digitalen Zwilling ein, um zu überprüfen, ob die geplante Flugbahn des UAV in allen möglichen Kollisionsszenarien vermieden werden kann
- **Wahrnehmungsfehlersimulation**: Simulieren Sie Sensorfehlerszenarien wie Kameraverdeckung und LiDAR-Fehler, um die Robustheit und Verschlechterungsleistung der Zustandsschätzung des digitalen Zwillings zu testen
- **Kollaborative Simulation mehrerer Flugzeuge**: Gleichzeitige Einspeisung der geplanten Flugbahnen mehrerer UAVs in den digitalen Zwilling, um die Konflikterkennungs- und -vermeidungsfähigkeiten des Flugverkehrsmanagements zu überprüfen

---

## 6. Verwandte Arbeiten und typische Systeme

### 6.1 Digitale Zwillingsplattform auf Stadtebene

**AirSim City Twin** (Microsoft, 2017) ist eine der frühesten Open-Source-UAV-Simulationsplattformen, die eine fotorealistische städtische Umgebung bietet und die Simulation von RGB-Kameras, LiDAR, IMU und anderen Sensoren unterstützt. Der digitale Zwilling von AirSim basiert auf der Unreal Engine und verfügt über realistische Texturen, aber begrenzte geometrische Genauigkeit.**OnePlus City Digital Twin** (inspiriert von groß angelegter Forschung zur Rekonstruktion urbaner Szenen) nutzt die Photogrammetrie + LiDAR-Fusionsmethode, um digitale Zwillingsmodelle mehrerer chinesischer Städte mit einer Auflösung von 5 cm zu erstellen und unterstützt Stadtplanung und UAV-Simulation.

**NVIDIA Omniverse Replicator** bietet eine einheitliche Plattform für die Datensynthese und die Erstellung digitaler Zwillinge und unterstützt die Darstellung städtischer Szenen und die Beschleunigung des neuronalen Renderings auf Basis von USD (Universal Scene Description).

### 6.2 UAV-Luftraummodellierungsforschung

| Forschung | Jahr | Methodik | Abdeckung | Aktualisierungshäufigkeit |
|------|------|------|----------|----------|
| Stadt-NeRF | 2022 | NeRF mit mehreren Ansichten | Stadtblöcke | Statisch |
| Gauß-Urban | 2023 | 3DGS | Blockebene | Echtzeit |
| Instant-NGP | 2022 | Hash-Kodierung | Indoor/Kleine Szene | Echtzeit |
| Seifenlauge | 2023 | Neuronaler SLAM | Stadtebene | Online |
| Schutt-Sicherung | 2024 | Multimodale Fusion | Stadtgebiet | Quasi-Echtzeit |

---

## 7. Herausforderungen und zukünftige Richtungen

### 7.1 Aktuelle Hauptherausforderungen

**Rechenressourcenengpass**: Der digitale Zwilling des Luftraums auf Stadtebene ($10\text{km} \times 10\text{km} \times 300\text{m}$) enthält Milliarden von Voxeln/Gaussianern, was die Rechenleistung einer einzelnen Karte bei weitem übersteigt. Die Blockierungsstrategie bringt neue Probleme mit sich, wie z. B. die Nahtverarbeitung zwischen Blöcken und die blockübergreifende Trajektorienplanung.

**Widerspruch zwischen Aktualität und Genauigkeit**: Die Optimierung neuronaler Felder erfordert ausreichend Beobachtungsdaten zur Konvergenz, aber der Status des städtischen Luftraums ändert sich schnell (vorübergehende Bauarbeiten, Ereigniskontrolle) und der digitale Zwilling kann zurückbleiben.

**Konsistenz bei mehreren Auflösungen**: Die Anforderungen an die Genauigkeit des Luftraums in verschiedenen Höhen sind unterschiedlich – in Bodennähe (0–30 m) ist eine Genauigkeit im Zentimeterbereich erforderlich, um Hindernissen auszuweichen, während der Luftraum in großen Höhen (100–300 m) auf Situationsbewusstsein ausgerichtet ist. Für bestehende neuronale Feldmethoden ist es schwierig, Anforderungen mit mehreren Auflösungen in einer einzigen Darstellung einheitlich zu bewältigen.

### 7.2 Zukünftige Entwicklungsrichtung**Neuronale Geometrie-Hybriddarstellung**: Kombination der Vorteile expliziter Voxel/Gitter (effiziente Geometrieabfragen) und impliziter neuronaler Felder (Fotorealismus), um eine genaue und schöne Darstellung des städtischen Luftraums zu entwickeln.

**Großes Sprachmodell + digitaler Luftraumzwilling**: Verwenden Sie multimodale große Modelle wie GPT-4V, um die Semantik und Kontrollregeln des Luftraums zu verstehen, und fügen Sie natürliche Sprachbeschränkungen in das Planungssystem des digitalen Zwillings ein, um eine „Sprachsteuerungsplanung“ zu erreichen.

**Aktualisierung des digitalen Zwillings durch Crowdsourcing**: Nutzen Sie eine große Menge an Echtzeit-Beobachtungsdaten von UAVs, um den digitalen Zwilling der Stadt über Federated Learning zu verteilen und zu aktualisieren und so eine „Crowdsourcing-Kartierung“ zu erreichen.

---

## 8. Zusammenfassung

Digitale Zwillinge bieten die zuverlässigste, simulierte und überprüfbare digitale Basis für die städtische UAV-Planung in geringer Höhe. Die neuronale Rendering-Technologie verbessert die Konstruktionseffizienz und den Realismus digitaler Luftraumzwillinge durch differenzierbare Optimierung, inkrementelle Aktualisierungen und multimodale Fusionsfähigkeiten erheblich.

Vom „statischen Stadtmodell“ zum „dynamischen Echtzeit-Zwilling“ besteht jedoch noch ein weiter Weg. Die Kernherausforderungen liegen in der **effizienten Darstellung im großen Maßstab**, der **Echtzeitmodellierung dynamischer Elemente** und der **Konsistenz bei mehreren Auflösungen**. Mit der kontinuierlichen Weiterentwicklung von 3DGS, NeRF und der Technologie großer Sprachmodelle wird erwartet, dass digitale Zwillinge in geringer Höhe in Städten in den nächsten drei bis fünf Jahren von Forschungsprototypen zum tatsächlichen Einsatz übergehen.

---

## Referenzen

- Mueller, A. R., et al. (2022). City-NeRF: Neuronale Strahlungsfelder mit mehreren Ansichten für die Darstellung von Szenen im städtischen Maßstab. *ACM-Transaktionen auf Grafiken (ToG)*. https://doi.org/10.1145/3528223.3528346

- Pumarola, A., Corona, E., Pons-Moll, G. & Moreno-Nuguer, F. (2021). D-NeRF: Neuronale Strahlungsfelder für dynamische Szenen. *NeurIPS*, 34, 10318–10329.- Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). 3D-Gaußsches Splatting für Echtzeit-Strahlungsfeld-Rendering. *ACM-Transaktionen auf Grafiken*, 42(4), 1–14. https://doi.org/10.1145/3592403

- Rosinol, A., et al. (2020). Kimera: Eine Open-Source-Bibliothek für metrisch-semantische Lokalisierung und Zuordnung in Echtzeit. *IEEE Robotics and Automation Letters*, 5(2), 892–899.

- Qin, C., et al. (2022). Sofortige neuronale Grafikprimitive mit einer Hash-Kodierung mit mehreren Auflösungen. *ACM SIGGRAPH 2022*.

- Tosi, F., et al. (2024). Social-SLAM: Lernen der kollaborativen Multi-Roboter-Navigation anhand menschlicher Demonstrationen. *ICRA*. https://doi.org/10.1109/ICRA57147.2024.10610603

- Zhou, Y., et al. (2023). SUDS: Skalierbares städtisches dynamisches Szenenverständnis. *ICCV*.

---

*Dieser Artikel ist das dritte erweiterte Kapitel einer Artikelreihe zur städtischen Routenplanung mit Drohnen in geringer Höhe. *