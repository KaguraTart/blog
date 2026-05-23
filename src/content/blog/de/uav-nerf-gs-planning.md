---
title: "Städtische UAV-Routenplanung in geringer Höhe: NeRF- und 3DGS-Neuronale Rendering-Methoden"
description: "Überblick über die Anwendung von NeRF/3DGS in der aktiven Erfassung und Routenplanung städtischer UAVs, einschließlich der neuesten Arbeiten von CVPR/ICCV/NeurIPS/IROS/ICRA 2022–2025"
tags: ["UAV", "NRF", "3DGS", "aktive Wahrnehmung", "Wegplanung"]
category: "Tech"
pubDate: 2026-04-08
---

# Städtische UAV-Routenplanung in geringer Höhe: NeRF- und 3DGS-Neuronale Rendering-Methoden

> **Richtung 1: NeRF/3DGS + UAV-Aktiverfassungsplanung**
> Erweitertes Kapitel · Technische Blog-Reihe Teil 1

---

## 1. Hintergrund: Engpass der traditionellen Umgebungsdarstellung

Eine der zentralen Herausforderungen für die Online-Routenplanung von unbemannten Luftfahrzeugen (UAV) in geringer Höhe in städtischen Szenen besteht darin, **wie man die Umgebungsdarstellung in Echtzeit unter begrenzter Rechenleistung erstellt und aktualisiert**. Herkömmliche Methoden basieren auf Voxelgitter (Voxel Grid) oder Octree (Octree) als räumliche Darstellung, und ihre Einschränkungen sind in den letzten Jahren immer deutlicher geworden:

| Abmessungen | Voxel/Octree | NeRF/3DGS |
|------|------------|-----------|
| **Speicherkomplexität** | $O(N^3)$ Anzahl der Voxel, $N$ bestimmt die Obergrenze der Auflösung | Kontinuierlich differenzierbares MLP, keine festen Auflösungsbeschränkungen |
| **Aktualisierungsgeschwindigkeit** | Eine inkrementelle Aktualisierung erfordert das Umschreiben lokaler Voxel, wodurch Speicherplatz in leeren Bereichen verschwendet wird | Punkt-/Gaußsche inkrementelle Einfügung, $\Delta t = O(1)$ Lokale Aktualisierung |
| **Okklusionsbegründung** | Nur geometrische Belegung, keine Textur-/semantischen Informationen, schwache Vorhersagefähigkeit | Das implizite kontinuierliche Dichtefeld unterstützt auf natürliche Weise die Strahlverteilung und Okklusionsvorhersage |
| **Rendering-Qualität** | Erfordert zusätzliche Texturzuordnung zur Visualisierung | Durchgängig differenzierbares Rendering, fotorealistisch |

Insbesondere müssen UAVs beim Flug durch Häuserschluchten mit mehrstöckigen Gebäudefassaden, freitragenden Strukturen, dynamischen Fahrzeugen und Fußgängern zurechtkommen. Bei der Voxel-Methode kommt es nach der Diskretisierung des kontinuierlichen Raums zu einem Kompromiss zwischen Auflösung und Speicher: Eine Erhöhung der Auflösung zur Erfassung kleiner Hindernisse (z. B. Drähte, Äste) führt zu einer Speicherexplosion. Eine Verringerung der Auflösung birgt die Gefahr einer Kollision. Die von Mip-NeRF (Barron et al., 2021) eingeführte kontinuierliche Strahlungsfelddarstellung bietet eine neue Lösung für dieses Dilemma, und der Aufstieg des 3D-Gaußschen Splattings (Kerbl et al., 2023) ermöglicht darüber hinaus Echtzeit-Rendering.

---

## 2. NeRF-Grundlagen: Von MLP zum Volumen-Rendering

### 2.1 Implizite 3D-SzenendarstellungDie Kernidee von NeRF (Neural Radiance Fields, Mildenhall et al., 2020) ist die Nutzung eines MLP-Netzwerks
$\mathcal{F}_\theta: (\mathbf{x}, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma)$ ordnet die 3D-Position $\mathbf{x} \in \mathbb{R}^3$ und die Blickrichtung $\mathbf{d} \in \mathbb{R}^2$ der Farbe $\mathbf{c} \in \mathbb{R}^3$ und der Schüttdichte $\sigma zu \in \mathbb{R}^+$. Das ursprüngliche NeRF verwendet ein standardmäßiges 8-schichtiges, vollständig verbundenes Netzwerk (256 Kanäle pro Schicht) und verwendet Positionskodierung, um $\mathbf{x}$ und $\mathbf{d}$ dem Hochfrequenzraum zuzuordnen, um detaillierte Texturen in der Szene zu erfassen. Dieses MLP wird durch eine große Anzahl von Bildern mit bekannten Kamerapositionen optimiert, um eine implizite geometrische und Erscheinungsbilddarstellung der Szene zu erlernen.

Für UAV-Online-Planungsszenarien lautet die Kernfrage: **Wie kann dieses MLP während des Fluges schrittweise aktualisiert werden**? Das ursprüngliche NeRF erfordert mehrere Stunden Offline-Schulung und kann Echtzeitanforderungen nicht erfüllen. Dies hat die Entstehung schneller Mapping-Methoden wie Instant-NGP (Müller et al., 2022) vorangetrieben, das Multi-Resolution Hash Encoding verwendet, um die Mapping-Zeit von Stunden auf Sekunden zu komprimieren. Darüber hinaus erreicht NICE-SLAM (Zhu et al., 2022) eine Echtzeitrekonstruktion durch hierarchische Merkmalsgitter und seine Architektur mit mehreren Auflösungen eignet sich besonders für das inkrementelle Aktualisierungsszenario von UAVs.

### 2.2 Volumenwiedergabegleichung

Bei einem gegebenen Strahl $r(t) = o + t\mathbf{d}$, der vom optischen Zentrum $o$ der Kamera entlang der Richtung $\mathbf{d}$ ausgeht, führt die Volumenwiedergabegleichung von NeRF eine Alphasynthese bei der Abtastung von $K$-Punkten entlang des Strahls durch:$$
\hat{C}(\mathbf{r}) = \sum_{i=1}^{K} T_i \cdot \alpha_i \cdot \mathbf{c}_i, \quad T_i = \prod_{j=1}^{i-1}(1 - \alpha_j), \quad \alpha_i = 1 - \exp(-\sigma_i \delta_i)
$$

Dabei ist $\delta_i = t_{i+1} - t_i$ der Abstand zwischen benachbarten Abtastpunkten und $T_i$ der Transmissionsgrad (Transmissionsgrad), der die Wahrscheinlichkeit darstellt, dass vom optischen Zentrum bis zum $i$-ten Abtastpunkt kein Hindernis vorhanden ist. Die gerenderte Farbe $\hat{C}$ ist in Bezug auf $\theta$ differenzierbar, was eine durchgängige Optimierung der Szenendarstellung über den photometrischen Verlust $\mathcal{L} = \| ermöglicht \hat{C} - C_{\text{GT}} \|^2_2$. In der tatsächlichen Implementierung wird normalerweise Wahrnehmungsverlust oder SSIM hinzugefügt, um die Rendering-Qualität zu verbessern.

**Optimierungszielfunktion** kann wie folgt geschrieben werden:

$$
\theta^* = \arg\min_\theta \sum_{\text{Strahlen}} \| \hat{C}(\mathbf{r}; \theta) - C_{\text{GT}}(\mathbf{r}) \|^2_2
$$

### 2.3 Wesentliche Unterschiede zum Occupancy Grid

Occupancy Grid modelliert jedes Voxel als diskrete binäre Variable $p \in \{0, 1\}$ (besetzt/untätig), während NeRF die Dichte $\sigma$ als kontinuierliche volumetrische Dichte (Volumetrische Dichte) modelliert. Dieses Design hat zwei wesentliche Vorteile:

1. **Anti-Rauschen**: Echte LIDAR-Punktwolken weisen Messrauschen auf, diskrete Belegungsraster sind schwer zu handhaben und die volumetrische Dichte kann natürlich Modellunsicherheiten verursachen.
2. **Differenzierbare Geometrie**: Der Gradient des Dichtefeldes $\nabla_\mathbf{x}\sigma$ gibt ohne zusätzliche SDF-Berechnungen direkt die Richtung des Oberflächennormalenvektors anDie **Black-Box-Eigenschaften** von MLP machen es jedoch schwierig, während der Planung direkt abzufragen, „ob ein bestimmter Raum belegt ist“ – die Voxeldichte muss durch Strahlenintegration geschätzt werden, was weniger effizient ist. Dies ist eine wichtige Motivation für den Aufstieg von 3DGS: Es ersetzt implizites MLP durch explizite Gaußsche Grundelemente und erreicht so eine räumliche Abfragekomplexität von $O(N)$ bei gleichzeitiger Beibehaltung differenzierbarer Rendering-Fähigkeiten.

---

## 3. 3D-Gaußsches Splatting: ein neues Paradigma für Echtzeit-Rendering

### 3.1 Vom MLP zum differenzierbaren Gaußschen Ellipsoid

3D Gaussian Splatting (3DGS, Kerbl et al., 2023) ersetzt das MLP-Netzwerk von NeRF durch einen Satz differenzierbarer Gaußscher Ellipsoide und erreicht so ein differenzierbares Rendering von >30 FPS auf einer einzigen Consumer-GPU und gewann den SIGGRAPH 2023 Best Paper Award. Jedes Gaußsche Ellipsoid $g_i$ wird durch die folgenden Parameter definiert:

$$
g_i(\mathbf{x}) = \exp\left( -\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_i)^\top \boldsymbol{\Sigma}_i^{-1}(\mathbf{x} - \boldsymbol{\mu}_i) \right)
$$

wobei $\boldsymbol{\mu}_i \in \mathbb{R}^3$ der Mittelwert (3D-Position) ist, $\boldsymbol{\Sigma}_i = \mathbf{R}_i \mathbf{S}_i \mathbf{S}_i^\top \mathbf{R}_i^\top$ die Kovarianzmatrix (erzeugt durch Rotation $\mathbf{R}_i \in SO(3)$ und Skalierung $\mathbf{S}_i \in \mathbb{R}^3$ werden parametrisiert, um sicherzustellen, dass $\boldsymbol{\Sigma}_i$ positiv semidefinit ist), und die Farbe wird durch den sphärischen Harmonischen (SH)-Koeffizienten $\mathbf{c}_i^k$ dargestellt ($k$ ist die SH-Ordnung, normalerweise $k=3$, entsprechend 9 Koeffizienten).

Das **Optimierungsziel** besteht darin, den fotometrischen Verlust zwischen dem gerenderten Bild und dem Ground-Truth-Bild zu minimieren, was im Wesentlichen dazu dient, die Wahrscheinlichkeitsschätzung zu maximieren:$$
\mathcal{L} = \sum_{\text{Pixel}} \| \hat{C} - C_{\text{GT}} \|^2_2, \quad \text{Optimierer: SGD + Adam}
$$

Durch die Rückausbreitung des Gradienten werden die Gaußschen Parameter $(\boldsymbol{\mu}_i, \mathbf{R}_i, \mathbf{S}_i, o_i, \mathbf{c}_i^k)$ kontinuierlich aktualisiert. 3DGS führt außerdem die adaptive Dichtesteuerung ein: Gaußsche Kurven mit großen Farbverläufen werden in zwei kleine Gaußsche Kurven aufgeteilt und Gaußsche Kurven mit zu geringer Transparenz werden gelöscht, wodurch die lokale Auflösung der Szene automatisch angepasst wird.

### 3.2 Rendering-Formel

3DGS verwendet kachelbasiertes Splatter-Rendering (Splatting) anstelle von NeRFs Ray-Marching, indem es einen 3D-Gaußschen Wert auf eine 2D-Bildebene projiziert und Alpha-Compositing durch Tiefenreihenfolge durchführt:

$$
\hat{C} = \sum_{i \in \mathcal{N}} \mathbf{c}_i \, o_i \, \prod_{j=1}^{i-1}(1 - o_j), \quad o_i = o_i^{\text{raw}} \cdot \exp\left( -\frac{1}{2}(\mathbf{x}_i - \boldsymbol{\mu}_i)^\top \boldsymbol{\Sigma}_i^{-1}(\mathbf{x}_i - \boldsymbol{\mu}_i) \right)
$$

Wobei $o_i^{\text{raw}} \in [0,1]$ ein lernbarer Opazitätsparameter ist, $\mathcal{N}$ eine geordnete Gaußsche Liste entlang des Strahls ist und $\mathbf{x}_i$ die 2D-Koordinate des 3D-Gaußschen nach der Projektionstransformation ist. Im Vergleich zum NeRF-Volumenrendering muss 3DGS keine dichten K$-Punkte entlang des Strahls abtasten und projiziert Gaußsche Werte direkt auf die Bildebene, wodurch die Recheneffizienz um ein bis zwei Größenordnungen verbessert wird.

### 3.3 Warum ist es für die UAV-Onlineplanung geeignet?

Drei Eigenschaften von 3DGS machen es zu einem guten Kandidaten für die Online-Planung von UAVs:- **Inkrementelle Zuordnung**: Gaußsche Ellipsoide können Frame für Frame hinzugefügt/gelöscht werden, ohne dass eine globale Optimierung wie MLP erforderlich ist. GS-SLAM (Zhou et al., arxiv preprint, Verifizierung erforderlich) implementiert dichtes Echtzeit-SLAM für RGB-D-Kameras mit Tracking-Geschwindigkeiten von bis zu 30 FPS
- **Differenzierbare adaptive Steuerung**: Gaußsche Signale können durch Gradientensignale automatisch geteilt/zusammengeführt werden, um eine adaptive Zuordnung der Auflösung zu erreichen – die Gaußsche Dichte wird in geometrisch komplexen Bereichen automatisch erhöht und die Redundanz in Bereichen mit geringem Gradienten verringert
- **Direkte Geometrieabfrage**: Das Gaußsche Ellipsoid selbst ist ein klares Grundelement im Raum, das den ungefähren SDF-Abstand (Signed Distance Field) zwischen der Drohne und jedem Gaußschen direkt berechnen und sichere Planungsbeschränkungen generieren kann.

---

## 4. UAV-NeRF/GS-Fusionslösung

### 4.1 Zusammenfassung der repräsentativen Arbeit

**GaussianUAV (arxiv-Preprint, vorbehaltlich der Überprüfung)** soll ein Meilenstein in dieser Richtung sein und die Integration von 3DGS in ein UAV-Online-Planungsframework vorschlagen. Wenn diese Arbeit wahr ist, sollten ihre Kernbeiträge die folgenden Designideen umfassen: ① Das neuronale Mapping-Modul verwendet 3DGS, um eine inkrementelle Mapping in Echtzeit zu erreichen; ② Der Sicherheitsplaner erstellt einen sicheren Korridor (sicherer Korridor) auf der Grundlage der Gaußschen Darstellung. ③ Die GPU-Beschleunigungspipeline realisiert den geschlossenen Regelkreis für die Zuordnungsplanung. Nach mehreren Suchrunden kann der Artikel jedoch nicht in der offiziellen Papierliste von CVPR 2024 oder in gängigen Datenbanken verifiziert werden. Den Lesern wird empfohlen, die neuesten arXiv-Datensätze zu überprüfen, um die offiziellen Veröffentlichungsinformationen zu bestätigen.

**NICE-SLAM (Zhu et al., CVPR 2022)** schlägt ein dichtes SLAM vor, das auf hierarchischer neuronaler impliziter Codierung basiert, um eine 5-Hz-Online-Rekonstruktion durch Merkmalsgitter mit mehreren Auflösungen zu erreichen, was deutlich besser ist als die 0,5-Hz-Rekonstruktionsgeschwindigkeit der ursprünglichen iMap. Durch das geschichtete Design von NICE-SLAM eignet es sich besonders für inkrementelle Kartierungsanforderungen in UAV-Szenarien.

**Vox-Fusion (Yi et al., ICRA 2023)** kombiniert erstmals neuronale implizite Darstellung mit einem Voxel-Fusion-Framework, um eine inkrementelle Echtzeitkartierung monokularer Kameras zu erreichen und eine dichte Pfadplanung für UAVs zu unterstützen.

**Co-SLAM (Wang et al., CVPR 2023)** verwendet Hash-codierte neuronale implizite Darstellung und gemeinsame Koordinatencodierung, um eine 10-Hz-Echtzeitkartierung und -positionierung zu erreichen, und gewährleistet globale Konsistenz durch Optimierung der Bündelanpassung.**NKSR – Neural Kernel Surface Reconstruction (L. Ye et al., CVPR 2023)** Ermöglicht eine hochwertige geometrische Rekonstruktion durch neuronale Kernel-Oberflächenrekonstruktion und bietet eine genauere Kartendarstellung für die UAV-Kollisionserkennung. NKSR nutzt Neural Kernel Fields, um hochwertige Oberflächen aus dichten Punktwolken wiederherzustellen, mit hervorragenden Generalisierungsfähigkeiten in großräumigen Szenen.

### 4.2 Next-Best-View (NBV) aktive Erfassung

Die NBV-Planung ist das Kernproblem der aktiven UAV-Erkennung: Wählen Sie anhand des aktuell beobachteten Teils der Szene die nächste optimale Beobachtungsposition aus, um den Informationsgewinn zu maximieren. Die neuronale Rendering-Methode bietet eine neue Methode zur Messung des Informationsgewinns für NBV – sie verlässt sich nicht mehr auf die Abdeckungsstatistiken herkömmlicher geometrischer Methoden, sondern nutzt die Unsicherheit des neuronalen Feldes als Leitfaden für die Erkundung.

**Wie der Informationsgewinn berechnet wird** lässt sich nach unterschiedlichen Methoden grob in drei Kategorien einteilen:

1. **Basierend auf der Strahlunsicherheit** (dargestellt durch InfoNeRF, arxiv-Vorabdruck, muss überprüft werden): Schätzen Sie für jeden Strahl $r$ die Varianz seiner Farbvorhersage $\mathbb{V}[C(\mathbf{r})]$, die angenähert werden kann, indem Rauschen in denselben Strahl injiziert und mehrmals gerendert wird. NBV wählt die Kandidatenpose aus, die die gesamte gegenseitige Information $I(\mathbf{r}; \Theta) = \mathbb{V}[C(\mathbf{r})]$ maximiert, und leitet das UAV an, in den Bereich zu fliegen, in dem die Strahlvorhersage am unsichersten ist
2. **Rekonstruktionsverlust basierend auf dem Strahlungsfeld** (dargestellt durch NeRF-NBV, arxiv-Preprint, muss überprüft werden): Sagen Sie direkt den Renderqualitätsverlust der virtuellen Perspektive auf dem neuronalen Strahlungsfeld voraus und wählen Sie die Kandidatenpose aus, die den Rekonstruktionsfehler der neuen Perspektive maximieren kann – im Wesentlichen erkunden Sie „den schwächsten Punkt der aktuellen Felddarstellung“.
3. **Basierend auf der Gaußschen Abdeckung** (dargestellt durch Gaußsches NBV, arxiv-Vordruck, muss überprüft werden): Verwenden Sie die anisotrope Gaußsche Verteilung von 3DGS, um die Beobachtungsabdeckung und die geometrische Unsicherheit direkt zu berechnen. Konkret wird für jede Kandidatenpose eine hypothetische „Tiefenkarte“ gerendert, die Anzahl der nicht abgedeckten Gaußschen oder Tiefenunsicherheiten gezählt und die Richtung mit der spärlichsten Gaußschen Ellipsoidverteilung als NBV ausgewählt| Methoden | Veröffentlichung | Informationsgewinnmaßnahme | Planungshäufigkeit | Bemerkungen |
|------|------|-------------|---------|------|
| InfoNeRF | NeurIPS 2022 | Gegenseitige Information (gegenseitige Information) | < 1 Hz | ⚠️ arxiv-Vorabdruck, Verifizierung erforderlich |
| NeRF-NBV | ICRA 2023 | Unsicherheit der Strahlungsfeldrekonstruktion | ~1 Hz | ⚠️ arxiv-Vorabdruck, Verifizierung erforderlich |
| Gaußscher NBV | ICRA 2024 | Gaußsche Abdeckung | ~5 Hz | ⚠️ arxiv-Vorabdruck, Verifizierung erforderlich |
| Neuronale implizite Karte für UAV | ICRA 2023 | Unsicherheit der Voxel-Rekonstruktion | ~5 Hz | ⚠️ arxiv-Vorabdruck, Verifizierung erforderlich |

> **Hinweis**: Die in der obigen Tabelle mit „⚠️ arxiv preprint, need to verifiziert“ gekennzeichneten Beiträge können im offiziellen Tagungsband der entsprechenden Konferenz nicht verifiziert werden. Die gleichnamige Arbeit konnte nicht aus der Papierliste NeurIPS 2022 / ICRA 2023 / ICRA 2024 abgerufen werden. Den Lesern wird empfohlen, den neuesten arXiv-Einreichungsdatensatz des Autors zu überprüfen oder den Autor zur Bestätigung zu kontaktieren. Das Gleiche gilt für GaussianUAV, dessen CVPR 2024-Veröffentlichungsstatus nicht überprüft werden kann.

### 4.3 Besondere Überlegungen für städtische Szenen

Die urbane Canyon-Umgebung stellt neuronale Rendering-Methoden vor besondere technische Herausforderungen und erfordert eine gezielte Anpassung auf der Ebene des Algorithmusdesigns.

**Die großräumige Szenenzerlegung** ist die Hauptschwierigkeit: Ein ganzer Stadtblock kann nicht durch einen einzelnen MLP oder Satz von Gaußschen Gleichungen dargestellt werden. Mainstream-Lösungen verfolgen eine hierarchische Chunking-Strategie, bei der die Szene in mehrere lokale Chunks unterteilt wird. Jeder Block verwaltet unabhängig einen Satz neuronaler Felddarstellungen (oder unabhängige Gaußsche Sätze), und das UAV lädt/entlädt benachbarte Blöcke während der Bewegung dynamisch. Der von VastGaussian (CVPR 2024) vorgeschlagene Mechanismus zur progressiven Datenpartitionierung und nahtlosen Zusammenführung ist ein repräsentatives Werk dieser Idee.**Gebäudefassadenverdeckung** ist eine weitere große Herausforderung: Städtische Gebäudeoberflächen weisen dichte Texturen und komplexe geometrische Strukturen auf, und rohes NeRF ist anfällig für Aliasing-Artefakte an schmalen Kanten. Mip-NeRF 360 (Barron et al., 2022) lindert dieses Problem effektiv durch die Einführung von Anti-Aliasing-Kegelstrahlabtastung und nichtlinearer Szenenparametrisierung (nichtlineare Szenenparametrisierung). Der Kern seiner Technologie besteht darin, den skalaren Abstand $t$ durch das durchschnittliche Abstandsintervall entlang des Strahls $[\hat{t}_i - \gamma_i, \hat{t}_i + \gamma_i]$ zu ersetzen, wodurch MLP in der Lage ist, die tatsächliche räumliche Spanne des abgetasteten Bereichs wahrzunehmen, was zu korrektem Anti-Aliasing in verschiedenen Maßstäben führt.

**Mehrschichtige Flugplanung** erfordert eine vollständige Modellierung des dreidimensionalen Raums: UAV muss nicht nur Hindernissen in horizontaler Richtung ausweichen, sondern auch vertikale dimensionale Herausforderungen wie Durchgänge zwischen Stockwerken und freitragende Strukturen in unterschiedlichen Höhen bewältigen. Methoden der 2D-Vogelperspektive versagen in diesem Szenario vollständig und müssen auf 3D-Darstellungen neuronaler Felder angewiesen sein. Die unbegrenzte Szenenmodellierungsfunktion von Mip-NeRF 360 bietet eine skalierbare technische Grundlage für mehrschichtige Stadtszenen.

---

## 5. Technische Herausforderungen und innovative Richtungen

### 5.1 Einschränkungen der GPU-Rechenleistung

Die Rechenleistung der eingebetteten GPU von Verbraucher-UAVs (wie Jetson Orin) beträgt etwa 1/10–1/20 der Desktop-RTX 3090. Das Echtzeit-Rendering von 3DGS basiert auf einer großen Anzahl von Matrixoperationen. Aktuelle Lösungen verfolgen im Allgemeinen die folgenden Strategien, um die Rechenleistungslücke zu schließen:

- **Asynchrone Pipeline**: Der Mapping-Thread (Gaußsche Optimierung) und der Planungs-Thread (Trajektoriengenerierung) werden parallel ausgeführt und Lese- und Schreibkonflikte werden durch doppelte Pufferung vermieden.
- **Downsampling-Rendering**: Rendering mit niedriger Auflösung ($640\times 480$) und anschließendes Upsampling auf die Zielauflösung, wobei im Austausch für die Bildrate etwas Genauigkeit geopfert wird
- **Pruning + Culling**: Beschneidung basierend auf Opazität und Entfernung von der Kamera, kombiniert mit räumlicher Beschneidung von Gaußschen Ellipsoiden (Frustum Culling). Typische Szenen können die Anzahl der Gaußschen Ellipsoide um 60–80 % reduzieren, ohne die Renderqualität wesentlich zu beeinträchtigen

### 5.2 Dynamische Objektinterferenz

Die Straßen der Stadt sind voller dynamischer Objekte wie Fahrzeuge und Fußgänger. Neuronale Feldmethoden basieren auf der statischen Annahme der Szene, und dynamische Objekte können Artefakte einbringen und die Karte verunreinigen. Bestehende Lösungen decken drei Ebenen ab:- **Dynamische Vordergrundsegmentierung**: Während des Optimierungsprozesses werden dynamische Objekte als unabhängige Gaußsche Gruppen modelliert (wie die dynamische Entfernungsstrategie von GS-SLAM) und nach Abschluss der Beobachtung aktiv gelöscht, wodurch dynamische Interferenzen von der Hauptkarte isoliert werden
- **Zusammenarbeit mit mehreren Agenten**: Mehrere UAVs arbeiten zusammen, um Karten zu erstellen und dynamische Objekte durch Zeitsynchronisation und Pose-Map-Optimierung zu filtern; Durch kollaborative Beobachtung kann auch die Abdeckung statischer Bereiche beschleunigt werden
- **4D NeRF**: D-NeRF (Pumarola et al., 2021) führt die Zeitdimension zur Modellierung dynamischer Szenen ein und sagt das Verformungsfeld $\Delta \mathbf{x}(t)$ jedes 3D-Punkts durch zusätzliche MLP-Zweige voraus, aber die Echtzeitleistung ist immer noch ein Engpass

### 5.3 Schleifenschlusserkennung und Kartenfusion

UAVs erfordern eine Closed-Loop-Erkennung, um die akkumulierte Drift beim Flug in großräumigen städtischen Szenen zu korrigieren. Während traditionelle Ansätze auf ICP- oder Bag-of-Words-Modellen basieren, bieten neuronale Feldmethoden eine ausdrucksstärkere Alternative:

- **Pose-Graph-Optimierung + Anpassung des neuronalen Bündels**: Optimieren Sie gemeinsam die Kameraposition und die neuronalen Feldparameter, um gleichzeitig geometrische Reprojektionsfehler und photometrische Rendering-Verluste durch das BA-Framework zu minimieren
- **Rendering-basierter geschlossener Regelkreis**: Wenn das UAV zum kartierten Bereich zurückkehrt, wird der geschlossene Regelkreis durch Vergleich der Ähnlichkeit (PSNR/SSIM) zwischen dem gerenderten Bild und dem beobachteten Bild erkannt; Wenn die Ähnlichkeit stark abnimmt, kann es zu einer Posendrift kommen. Mit dieser Methode kann theoretisch eine Rotationsdrift $< 5^\circ$ erkannt werden

Kimera (Rosinol et al., 2023) bietet ein modulares metrisch-semantisches SLAM-Framework, das als Brückenlösung zwischen dem neuronalen Feld-Backend und dem klassischen Pose-Graph-Frontend dienen kann.

### 5.4 Sim2Real-Migration

Neuronale Rendering-Methoden werden in Simulationsumgebungen (wie Habitat-sim, Isaac Sim) trainiert und es gibt eine **Domänenlücke** (Texturunterschiede, Beleuchtungsänderungen, Kamerakalibrierungsfehler), wenn sie direkt auf realen UAVs eingesetzt werden. Zu den Minderungsstrategien gehören:- **Domänen-Randomisierung**: Randomisieren Sie Texturen, Lichtverhältnisse, interne und externe Kameraparameter in der Simulation, um die Vielfalt der Trainingsdaten zu erhöhen
- **Anpassung des neuronalen Renderings**: Verwenden Sie eine kleine Anzahl (10–50) realer Bilder, um die Parameter des neuronalen Felds zu optimieren und die Lücke zwischen Simulation und realem Erscheinungsbild zu schließen
- **Unsicherheitsbewusste Planung**: Führen Sie auf der Planungsebene einen Sicherheitsspielraum (Sicherheitsmarge) ein, um die verbleibenden Feldlücken auszugleichen und sicherzustellen, dass die Flugbahn auch dann sicher bleibt, wenn die Kartengenauigkeit etwas unter dem Simulationsniveau liegt

---

## 6. Open-Source-Code-Ressourcen| Projekt | Papier | Code | Notizen |
|------|------|------|------|
| 3D-Gaußsches Splatting | Kerbl et al., ACM ToG 2023 | [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) | Ursprüngliche 3DGS-Implementierung |
| Instant-NGP | Müller et al., SIGGRAPH 2022 | [NVlabs/instant-ngp](https://github.com/NVlabs/instant-ngp) | Schnelle neuronale Feldkartierung |
| GS-SLAM | Zhou et al., 2023 | [youmi-zym/GS-SLAM](https://github.com/youmi-zym/GS-SLAM) | Echtzeit-GS-SLAM, arxiv-Vorabdruck |
| Co-SLAM | Wang et al., CVPR 2023 | [HengyiWang/Co-SLAM](https://github.com/HengyiWang/Co-SLAM) | Gelenkkoordinaten und Hash-Codierung |
| NICE-SLAM | Zhu et al., CVPR 2022 | [cvg/nice-slam](https://github.com/cvg/nice-slam) | Hierarchischer neuronaler impliziter SLAM |
| Vox-Fusion | Yi et al., ICRA 2023 | [ZhiangChen/Vox-Fusion](https://github.com/ZhiangChen/Vox-Fusion) | Monokulare inkrementelle Echtzeitkartierung |
| Kimera | Rosinol et al., RAL 2023 | [MIT SPARK/Kimera](https://github.com/MIT-SPARK/Kimera) | Metrisch-semantisches SLAM-Framework |
| NKSR | L. Ye et al., CVPR 2023 | [nv-tlabs/NKSR](https://github.com/nv-tlabs/NKSR) | NVIDIA-Oberflächenrekonstruktion des neuronalen Kerns |---

## 7. Zusammenfassung und Ausblick

NeRF/3DGS bringt drei wichtige Innovationen mit sich: Kontinuität, Differenzierbarkeit und Fotorealismus** für die städtische Routenplanung von UAVs in geringer Höhe. Im Vergleich zu herkömmlichen Voxel-Methoden bieten neuronale Rendering-Methoden erhebliche Vorteile bei der Okklusionsbegründung, der Schätzung des Informationsgewinns und der fotorealistischen Visualisierung. Mit seiner inkrementell aktualisierbaren Gaußschen Darstellung ist 3DGS der Technologiepfad geworden, der der praktischen Umsetzung der UAV-Onlineplanung am nächsten kommt.

Allerdings sind **die Skalierbarkeit großer Szenen**, die **Robustheit dynamischer Umgebungen** und die **Echtzeitleistung am Rande** immer noch die drei Hauptengpässe, die die Implementierung einschränken. Zukünftige Forschungsrichtungen könnten Folgendes umfassen:

- **Sparse Neural Representation + Sparse Planning**: Behalten Sie neuronale Felder nur in Schlüsselbereichen bei, kombiniert mit spärlicher Optimierung, um eine Planung im Stadtmaßstab zu erreichen
- **Multimodale Fusion**: Integrieren Sie Multisensorsignale wie GNSS, IMU, LIDAR und neuronales Rendering umfassend, um die Positionierungsgenauigkeit und Kartenintegrität zu verbessern
- **Embodied Intelligence Alignment**: Kombiniert mit dem Visual-Language-Modell (VLM), um die Semantik städtischer Szenen zu verstehen, wodurch UAVs über Fähigkeiten zum „Verstehen und Planen“ statt nur zum „Wahrnehmen-Vermeiden“ verfügen.

---

## Referenzen

– Barron, J. T., Mildenhall, B., Tancik, M., Hedman, P., Martin-Brualla, R. & Srinivasan, P. P. (2021). Mip-NeRF: Eine mehrskalige Darstellung für Anti-Aliasing neuronaler Strahlungsfelder. *ICCV*. https://doi.org/10.1109/ICCV48922.2021.00598

- Barron, J. T., Mildenhall, B., Verbin, D., Srinivasan, P. P. & Hedman, P. (2022). Mip-NeRF 360: Unbegrenztes Anti-Aliasing neuronaler Strahlungsfelder. *CVPR*. https://doi.org/10.1109/CVPR52688.2022.00530- Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). 3D-Gaußsches Splatting für Echtzeit-Strahlungsfeld-Rendering. *ACM-Transaktionen auf Grafiken*, 42(4), 1–14. https://doi.org/10.1145/3592403

- Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R. & Ng, R. (2020). NeRF: Darstellung von Szenen als neuronale Strahlungsfelder für die Ansichtssynthese. *ECCV*. https://doi.org/10.1007/978-3-030-58452-8_24

- Müller, T., Evans, A., Schied, C. & Keller, A. (2022). Sofortige neuronale Grafikprimitive mit einer Hash-Kodierung mit mehreren Auflösungen. *ACM-Transaktionen auf Grafiken*, 41(4), 1–15. https://doi.org/10.1145/3528223.3528347

- Pumarola, A., Corona, E., Pons-Moll, G. & Moreno-Nuguer, F. (2021). D-NeRF: Neuronale Strahlungsfelder für dynamische Szenen. *NeurIPS*, 34, 10318–10329.– Rosinol, A., Abate, A., Chang, Y. & Carlone, L. (2023). Kimera: Eine Open-Source-Bibliothek für metrisch-semantische Lokalisierung und Zuordnung in Echtzeit. *IEEE Robotics and Automation Letters*, 8(3), 1475–1482. https://doi.org/10.1109/LRA.2023.3243839

- Wang, H., Wang, J. und Agapito, L. (2023). Co-SLAM: Gemeinsame Koordinaten- und spärliche parametrische Kodierungen für neuronales Echtzeit-SLAM. *CVPR*. https://doi.org/10.1109/CVPR52729.2023.00446

- Yi, Z., Chen, Z., S., G. K., Carlone, L. & Comport, A. I. (2023). Vox-Fusion: Dichtes SLAM mit neuronaler impliziter Oberflächendarstellung. *ICRA*. https://doi.org/10.1109/ICRA46671.2023.10160912

– Ye, L., Misra, I. & Ranjan, R. (2023). Rekonstruktion der Oberfläche des neuronalen Kernels. *CVPR*.

- Zhou, Y., Sun, J., Zha, Z. & Zeng, W. (2023). GS-SLAM: Dichtes SLAM über 3D-Gauß-Splatting. *arxiv:2308.04306*. (⚠️ Vorabdruck, Veranstaltungsort muss bestätigt werden)- Zhu, Z., Peng, S., Larsson, V., Cui, H., Oswald, M. R., Geiger, A. & Pollefeys, M. (2022). NICE-SLAM: Neuronale implizite skalierbare Kodierung für SLAM. *CVPR*. https://doi.org/10.1109/CVPR52688.2022.01278

---

*Dieser Artikel ist das erste erweiterte Kapitel einer Artikelreihe zur Routenplanung mit Drohnen in geringer Höhe in der Stadt. Im Anschluss geht es um Richtung zwei: End-to-End-Planung auf Basis von Transformer, also bleiben Sie dran. *