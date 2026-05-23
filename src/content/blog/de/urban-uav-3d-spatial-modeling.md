---
title: "Städtische UAV-Routenplanung in geringer Höhe: dreidimensionale räumliche Modellierung"
description: "Überprüfen Sie systematisch die dreidimensionalen Raummodellierungsmethoden bei der Routenplanung von UAVs in geringer Höhe in Städten, die das 3D-Belegungsraster, den städtischen Canyon-Effekt und das Luftraum-Schichtenmodell abdecken"
tags: ['UAV', '路径规划', '城市空域']
category: Tech
pubDate: 2026-04-08T14:54:00+08:00
---

## Einführung

Die Routenplanung städtischer UAVs in geringer Höhe ist eine der wichtigsten Basistechnologien für einen sicheren und effizienten städtischen Luftverkehr (UAM, Urban Air Mobility). Im Gegensatz zu vorstädtischen Freiflächen weist die städtische Umgebung besondere Merkmale auf, wie z. B. eine komplexe dreidimensionale geometrische Struktur, eine starke Dämpfung von GNSS-Signalen und eine starke Störung des Strömungsfelds durch Gebäude, was höhere Anforderungen an räumliche Modellierungsmethoden stellt. Dieser Artikel konzentriert sich auf den ersten Teil der Reihe städtischer UAV-Routenplanung in geringer Höhe – dreidimensionale Raummodellierung. Es werden das dreidimensionale Belegungsgitter (3D Occupancy Grid) und die Octree-Darstellung (Octree), die physikalische Modellierung des Urban Canyon-Effekts (Urban Canyon) und das hierarchische Luftraummodell, das auf der traditionellen Luftfahrtsteuerung basiert, ausführlich erörtert, ergänzt durch eine vergleichende Analyse der technischen Umsetzung.

## 1. Dreidimensionales Belegungsgitter und Octree-Darstellung

### 1.1 Von zwei Dimensionen zu drei Dimensionen: mathematische Definition

Das klassische Belegungsraster (Occupancy Grid) wurde von Moravec und Elfes (1985) vorgeschlagen. Seine Kernidee besteht darin, den kontinuierlichen Raum in endliche Gitter zu diskretisieren und den Belegungszustand jedes Gitters mit Wahrscheinlichkeitswerten zu kodieren. Im zweidimensionalen Fall wird der Raum in quadratische Zellen mit der Seitenlänge $\Delta$ unterteilt und die Belegungswahrscheinlichkeit jeder Zelle $m_i$ wird als $P(m_i | Z_{1:t})$ aufgezeichnet, wobei $Z_{1:t}$ alle Sensorbeobachtungen bis zum Zeitpunkt $t$ sind. Sensoraktualisierungen folgen der Bayes'schen rekursiven Formel:

$$
P(m_i | Z_t, Z_{1:t-1}) = \frac{P(Z_t | m_i, Z_{1:t-1}) P(m_i | Z_{1:t-1})}{P(Z_t | Z_{1:t-1})}
$$

Um numerische Unterschreitungen zu vermeiden und Berechnungen zu vereinfachen, werden in der tatsächlichen Technik üblicherweise **logarithmische Quoten (Log-Odds)** ausgedrückt:

$$
l(m_i) = \log \frac{P(m_i)}{1 - P(m_i)}
$$

Nach jeder Sensormessung lautet die additive Aktualisierungsregel:

$$
l(m_i)_{\text{neu}} = l(m_i)_{\text{alt}} + \Updelta l
$$

wobei $\Delta l$ durch das Sensormodell bestimmt wird (positiv bei Belegung, negativ bei Leerlauf). Diese Methode wandelt Multiplikation in Addition um und verbessert so die Echtzeitleistung erheblich.Das dreidimensionale Belegungsgitter erweitert die obige Definition von einer Ebene auf einen volumetrischen Raum $\mathbb{R}^3$ und unterteilt den Raum in kubische Einheiten (Voxel) der Kantenlänge $\Delta$. Angenommen, $V_i \subset \mathbb{R}^3$ repräsentiert das $i$-te Voxel, dann ist seine Belegungswahrscheinlichkeit $P(v_i | Z_{1:t})$. Die direkte Speicherkomplexität dreidimensionaler Raster beträgt $O(N^3)$ ($N$ ist die Anzahl eindimensionaler Raster), was in typischen städtischen Szenarien nicht akzeptabel ist – beispielsweise beträgt die Gesamtzahl der Voxel, die ein Stadtgebiet von $1\,\text{km}^3$ abdecken, bei einer Auflösung von $0,1\,\text{m}$ insgesamt bis zu $10^{13}$.

### 1.2 Octree: Räumlicher Index für adaptive Auflösung

**Octree** ist die Standardlösung zur Bewältigung der oben genannten Speicherherausforderungen. Die von Hornung et al. vorgeschlagene OctoMap-Bibliothek. (2013) ist ein Meilenstein bei der Implementierung dieser Methode im Bereich der Robotik. Die Raumteilungslogik des Octree lautet wie folgt: Der Wurzelknoten deckt den gesamten dreidimensionalen Raum ab, und jeder interne Knoten wird rekursiv in 8 Unterknoten gleichen Volumens unterteilt (entsprechend der Aufteilung $2 \times 2 \times 2$ des dreidimensionalen Raums), bis die voreingestellte maximale Tiefe $d_{\max}$ oder die minimale Voxelgröße $\Delta_{\min}$ erreicht ist.

Angenommen, die Seitenlänge des Wurzelknotens beträgt $L_0$ und die Seitenlänge des Voxels in der Tiefe $d$ beträgt:

$$
L_d = \frac{L_0}{2^d}
$$

Die maximale Anzahl von Knoten für die Tiefe $d$ beträgt $8^d$, aber da der Octree nur den belegten oder beobachteten Raum aufteilt, können unbekannte/freie Bereiche durch einen einzelnen Knoten dargestellt werden, sodass die tatsächliche Anzahl der Knoten viel kleiner ist als das vollständige Raster. Das Speichermodell von OctoMap verwendet außerdem **Probabilistic OcTree**: Jeder Knoten speichert einen Belegungswahrscheinlichkeitswert $P(n)$, der durch Bayes'sche Aktualisierung kontinuierlich geändert wird. Die Wahrscheinlichkeit eines inaktiven Knotens beträgt $P_{\text{occ}}$, die Wahrscheinlichkeit eines besetzten Knotens liegt nahe bei $1$ und der entsprechende Knoten im unbekannten Bereich existiert nicht im Baum (implizite Codierung).

Die Experimente von Hornung et al. (2013) zeigen, dass der Speicherverbrauch von OctoMap in einer typischen Innenumgebung etwa **1/50** dichter dreidimensionaler Raster mit derselben Auflösung beträgt und gleichzeitig dynamische Aktualisierungen und Abfragen mit beliebiger Auflösung unterstützt.

### 1.3 Octree und MultigranularitätswahrnehmungZeng et al. (2020) schlug einen Algorithmus zur Umgebungswahrnehmung mit mehreren Granularitäten vor, der auf Octree-Belegungsgittern für Multimedia-Tools und -Anwendungen basiert, und wies darauf hin, dass das Punktwolkenmodell zwar reich an Informationen ist, es jedoch viele Redundanzen bei der Pfadplanung gibt. Sie verwenden Octrees, um eine einheitliche probabilistische Darstellung von Daten verschiedener Sensoren (RGB-D, LiDAR usw.) bereitzustellen, wobei hochauflösende geometrische Informationen auf Blattknotenebene erhalten bleiben und eine globale Strukturwahrnehmung mit niedriger Auflösung auf grober Knotenebene bereitgestellt werden. Diese Idee ist besonders wichtig für die Erstellung groß angelegter Karten auf Stadtebene – in der Nahentfernung ist eine Hindernisvermeidung auf Zentimeterebene und in der Ferne eine Makrowegentscheidung auf der 100-Meter-Ebene erforderlich.

Thomas et al. (2021) schlug im arXiv-Artikel (arXiv: 2108.10585) außerdem **Spatiotemporal Occupancy Grid Maps (SOGM)** vor, das die Zeitvorhersage dynamischer Hindernisse in die Gitterdarstellung einbettet, effektive Möglichkeiten zur Belegungsvorhersage für Menschen und Fahrzeuge bietet, die sich in der städtischen Umgebung bewegen, und von großem Wert für die Echtzeitplanung zur Vermeidung von Hindernissen ist.

## 2. Urban Canyon-Effekt: Herausforderungen bei der physikalischen Modellierung und Navigation

Urban Canyon bezeichnet eine städtische Mikrolandschaft mit dichter Bebauung und engen Straßen. Es handelt sich um eine der anspruchsvollsten Einsatzumgebungen für Drohnen in geringer Höhe. Seine physikalischen Wirkungen können dreidimensional verstanden werden.

### 2.1 GNSS-Signaldämpfung und Mehrwegeeffekt

In städtischen Schluchten bilden dichte Hochhäuser eine „Canyon“-Struktur, und GNSS-Satellitensignale sind zwei Arten schwerwiegender Störungen ausgesetzt:

- **Non-Line-of-Sight-Ausbreitung (NLOS)**: Das direkte Signal wird durch das Gebäude blockiert und die Drohne kann nur das von der Wand reflektierte oder gebeugte Signal empfangen, wodurch der Pseudoentfernungsmesswert systematisch größer wird;
- **Mehrweg**: Die Signalüberlagerung mehrerer Reflexionspfade führt zu Trägerphasenlösungsfehlern und Positionierungsjitter.

Der UrbanNav-Datensatz (Wen et al., 2021; GitHub: IPNL-POLYU/UrbanNavDataset) hat die Positionierungsleistung von kostengünstigen Sensoren in städtischen Schluchten in Tokio und Hongkong gemessen. Die Ergebnisse zeigten, dass in tiefen Schluchtgebieten der Fehler bei der Einzelpunktpositionierung (SPP) mehrere zehn Meter betragen kann. Selbst wenn ein Zweifrequenz-GNSS-Empfänger ohne NLOS-Erkennung und -Beseitigung verwendet wird, wird es immer noch schwierig sein, mit der horizontalen Positionierungsgenauigkeit die Submeter-Anforderungen für die Schwebegenauigkeit von UAVs zu erfüllen. Das Seitenverhältnis (AR = Gebäudehöhe / Straßenbreite) von Häuserschluchten ist der dominierende Faktor für die GNSS-Genauigkeit – je größer das AR, desto geringer die Signalverfügbarkeit.

### 2.2 Turbulenzen und WindfeldstörungenDie Strömungsdynamik in Stadtschluchten weist ein hohes Maß an Heterogenität auf. Die klassische Studie von Rotach (1995) in *Boundary-Layer Meteorology* quantifizierte das statistische Profil von Turbulenzen in Schluchten und stellte fest, dass die turbulente kinetische Energie (TKE) in Straßenschluchten **2-5 mal** höher ist als in offenen Vororten und dass die Standardabweichung der vertikalen Geschwindigkeitskomponente $\sigma_w$ das 0,3- bis 0,6-fache der mittleren Windgeschwindigkeit nahe der Oberfläche erreichen kann. Zu den wichtigsten physikalischen Mechanismen gehören:

- **Gebäudeschleppe**: Der Luftstrom bildet auf der Leeseite periodisch ablösende Wirbel (Kármán-Wirbelstraße), nachdem er das Gebäude umgangen hat, was erhebliche instabile Auftriebs- und Seitenkräfte erzeugt;
- **Canyon-Zirkulation (Straßen-Canyon-Zirkulation)**: Wenn die einströmende Strömung orthogonal zur Canyon-Achse verläuft, bildet sich innerhalb der Straße eine doppelte Wirbelringstruktur mit entgegengesetzten Richtungen, und die Netto-Vertikalwindgeschwindigkeitskomponente wird in diesem Bereich deutlich verstärkt;
- **Trägheitsunterbereich**: Das Energiespektrum der Turbulenzenergie im Trägheitsunterbereich folgt der $-5/3$-Regel (Kolmogorovs Gesetz). Turbulenzen im kleinen Maßstab stellen eine kontinuierliche Störung der Bandbreite der UAV-Lagesteuerung dar.

Für das UAV-Steuerungsdesign ist der charakteristische Frequenzbereich der Turbulenzintensität entscheidend. Die Störung im Frequenzband $1$–$10\,\text{Hz}$ ist in Stadtschluchten am bedeutendsten, was erfordert, dass die Lageschleifenbandbreite des Flugsteuerungssystems nicht weniger als $20\,\text{Hz}$ beträgt, was auf einer eingebetteten Plattform nicht einfach zu implementieren ist.

### 2.3 Bernoulli-Windbeschleunigungseffekt

In engen Gassen kann der Bernoulli-Effekt nicht ignoriert werden. Wenn der Luftstrom durch einen Kanal mit verringerter Querschnittsfläche gezwungen wird, erhöht sich die Windgeschwindigkeit in lokalen Bereichen gemäß der Kontinuitätsgleichung $A_1 ​​​​v_1 = A_2 v_2$ erheblich. Die Windgeschwindigkeiten an den engsten Stellen zwischen Gebäuden in Häuserschluchten können **1,5- bis 3-mal** so hoch sein wie in offenen Gebieten. Darüber hinaus wird durch den „Venturi-Effekt“ zwischen Gebäudefassaden lokal ein Sog zur Straßenmitte hin erzeugt, der die seitliche Stabilität der Drohne gefährdet.

In der praktischen Planung wird empfohlen, die **äquivalente Windstörung** in Häuserschluchten als mittleren Wind $\bar{u}$ zu modellieren, der mit zufälligen Turbulenzkomponenten $\tilde{u}$ überlagert ist:

$$
u_{\text{eff}}(t) = \bar{u} + \sigma_u \cdot \xi(t)
$$

wobei $\xi(t)$ das Gaußsche weiße Rauschen ist, das der Standardnormalverteilung folgt, und $\sigma_u$ aus einer empirischen Formel bestimmt wird, die auf dem Seitenverhältnis des Canyons und der lokalen Straßengeometrie basiert.## 3. Luftraum-Schichtenmodell

### 3.1 Aufklärung aus der traditionellen Luftfahrtkontrolle

Das traditionelle zivile Flugsicherungssystem nutzt seit Jahrzehnten die Verwaltung der Höhenschicht (Altitude Layer): Mit 1000 $ (ungefähr 300 m $) als grundlegendem Höhenintervall ist der Luftraum unter 29 000 ft $ in mehrere Kontrollsektoren unterteilt, wobei jede Schicht Flugzeuge unterschiedlichen Typs und unterschiedlicher Geschwindigkeit bedient. Im Kontext von UAM müssen städtische UAVs in geringer Höhe mit Fußgängern am Boden, Gebäuden, Hubschrauberlandeplätzen und herkömmlichen Flugzeugen in einem vertikalen Bereich von **0–120 m (ca. 0–400 m) koexistieren. Daher wird ein mehrschichtiges Design unvermeidlich.

Die UTM-Projektforschung (UAS Traffic Management) der NASA (2016–2024) und UAM ConOps V2.0 der FAA (2023) wiesen beide darauf hin, dass hierarchisches Management das wichtigste Mittel zur Vermeidung groß angelegter Drohnenkonflikte ist. Aufbauend auf dieser Idee in städtischen Szenarien kann das folgende dreischichtige Schema entworfen werden.

### 3.2 Aufteilungsschema für die Höhenschicht der städtischen Szene

| Höhenniveau | Vertikaler Bereich | Hauptfunktionen | Flugzeugtyp | Typische Geschwindigkeit |
|--------|----------|----------|-----------|----------|
| **G Etage** | Boden $\sim 30\,\text{m}$ | Expresszustellung auf Gehwegen, Roboterzustellung | Mikro-Mehrrotor | $0$–$5\,\text{m/s}$ |
| **L-Ebene** | $30$–$80\,\text{m}$ | Community-Logistik, urbane Luftaufnahmen, Low-Rise-Shuttle | Kleiner Mehrrotor-/Verbundflügel | $5$–$15\,\text{m/s}$ |
| **U-Stufe** | $80$–$120\,\text{m}$ | Intercity-Express, Notfallhilfe, Hochhaus-Shuttle | Mittlerer eVTOL/Starrflügel | $15$–$30\,\text{m/s}$ |

> Hinweis: Die spezifischen Höhengrenzen müssen entsprechend den örtlichen Luftraummanagementvorschriften (China basiert auf den „Interim Regulations on Unmanned Aircraft Flight Management“ 2023) und der Stadtplanung angepasst werden.

Die Gestaltungsprinzipien dieses Schichtschemas lauten wie folgt:1. **Funktionale Isolierung**: Die G-Schicht konzentriert sich auf die Sicherheit der Terminalverteilung (um direkte Konflikte mit Menschen zu vermeiden), die L-Schicht ist die städtische Mainstream-Anwendungsschicht und die U-Schicht liegt nahe an der Höhe der traditionellen allgemeinen Luftfahrt, um mit dem Übergang kompatibel zu sein;
2. **Strömungstrennung**: Die stromaufwärts und stromabwärts gerichteten Richtungen werden horizontal auf gleicher Höhe weiter getrennt, und die Einwegroutenschleife ist unter Bezugnahme auf die fünfseitige Anfluglogik der Flugsicherung konzipiert;
3. **Dynamische Anpassung**: Die Schichtgrenze kann entsprechend der Echtzeit-Verkehrsdichte dynamisch übersetzt werden, und das xTM-Framework (Extensible Traffic Management) der FAA stellt hierfür eine standardisierte Schnittstelle bereit.

### 3.3 Fusion von geschichteten und dreidimensionalen Rasterkarten

Das Höhenschichtmodell muss tief in das dreidimensionale Belegungsgitter integriert werden: In der Planungsphase wird die **Schichtmaskierung** auf der Octree-Karte basierend auf den Schichtgrenzen durchgeführt, und Pfade werden nur in den flugfähigen Voxeln der Schicht gesucht, in der sich die aktuelle Aufgabe befindet, und in angrenzenden Schichten; Bei der dynamischen Neuplanung kann bei Staus auf einer bestimmten Ebene automatisch auf die benachbarte Ebene umgeschaltet werden, um Umleitungen zu ermöglichen. Dieser Mechanismus wurde zunächst im UTM-Korridor-Konzept (Korridor) der NASA verifiziert.

## 4. Technische Kompromisse für Octree/PCL-Punktwolken/Voxel

In der technischen Praxis erfordert die Wahl der dreidimensionalen Darstellungsmethode einen Kompromiss zwischen Genauigkeit, Speicher, Berechnungsgeschwindigkeit und Aktualisierungsfrequenz. Es folgt ein systematischer Vergleich.| Metriken | Dichtes 3D-Raster | Octree | Rohpunktwolke (PCL) | Hash-Voxel |
|------|------------|----------------|--------------|--------|
| **Speichereffizienz** | Niedrig (fest $O(N^3)$) | Hoch (adaptive Aufteilung) | Mittel (nur Punkte gespeichert, keine Topologie) | Hoch (spärlicher Hash-Index) |
| **Abfragekomplexität** | $O(1)$ | $O(\log N)$ | $O(N)$ (erschöpfend) oder $O(\log N)$ (mit kd-Baum) | $O(1)$ bedeuten |
| **Dynamisches Update** | Langsam (vollständige Rekonstruktion) | Schnell (inkrementelle Knotenaufteilung) | Schnell (Punkte anhängen) | Schnell (Hash-Einfügung) |
| **Auflösungskonsistenz** | Globale Konsistenz | Hierarchische Anpassung | Keine Gitterstruktur | Globale Konsistenz |
| **Kollisionserkennung** | Schnell (Array-Index) | Mittel (Baumsuche) | Langsam (Punktmodellerkennung) | Schnell (Hash-Suche) |
| **Ingenieurökologie** | ROS nav_msgs | OctoMap / PCL Octree | PCL / Open3D | OctoMap (konfigurierbar) |
| **Anwendbare Szenarien** | Kleine Reichweite und hohe Präzision | Große Reichweite und mehrere Auflösungen | Echtzeit-Erfassung/Kartierung | Spärliche Großszenen |

**Octrees Hauptvorteil** liegt in seinen doppelten Eigenschaften von **adaptiver Auflösung + probabilistischer Darstellung**: Es handelt sich sowohl um eine räumliche Indexstruktur als auch um ein probabilistisches Aktualisierungsframework, das sich besonders für die Wahrnehmungsanforderungen von „genauen Hindernissen in der Nähe und groben Hindernissen in der Ferne“ in städtischen Szenen eignet. Die OctoMap-Bibliothek (Hornung et al., 2013; DOI: 10.1007/s10514-012-9321-0) ist ein Beweis für ihre technische Reife, sowohl im Hinblick auf die Aktivität auf GitHub als auch auf die Anzahl der akademischen Zitate (mehr als 5.000 Mal laut Google Scholar).

**Der Vorteil der Punktwolke** besteht darin, dass sie die ursprünglichen Sensordaten verlustfrei beibehält und sich für Wahrnehmungsalgorithmen eignet, die auf Deep Learning (3D-Zielerkennung, semantische Segmentierung usw.) als Eingabe basieren. Die PCL-Bibliothek (Point Cloud Library) und die Open3D-Bibliothek stellen eine ausgereifte Werkzeugkette für die Verarbeitung von Punktwolken bereit, aber die Punktwolke selbst kodiert keine semantischen Informationen zu belegten/untätigen Objekten und erfordert zusätzliche Schritte, um sie in einen flugfähigen Bereich umzuwandeln.**Hash-Voxel** (wie das „OcTree Key“-Hash-Indexschema von OctoMap) funktionieren gut in Szenarien, die eine extrem schnelle Abfragegeschwindigkeit und spärliche Szenen erfordern. Der Speicheraufwand ähnelt in etwa dem eines Octrees, die Abfrage ist jedoch effizienter. Es war in den letzten Jahren ein heißes Thema in der Spitzenforschung.

In tatsächlichen städtischen Szenarien verwendet die **empfohlene Lösung** den probabilistischen Octree von OctoMap als zugrunde liegenden Speicher, die ursprüngliche Punktwolke als Erfassungseingabe, korrigiert kontinuierlich die Belegungswahrscheinlichkeit durch den inkrementellen Aktualisierungsmechanismus und verwendet einen Hash-Index, um Abfragen nach nächsten Nachbarn zu beschleunigen. Diese Kombination hat sich in fortschrittlichen SLAM-Systemen wie LIO-SAM bewährt, um eine robuste Echtzeitkartierung in städtischen Schluchten zu erreichen (siehe angepasste Version von LIO-SAM-6AXIS-UrbanNav).

## 5. Zusammenfassung und Ausblick

In diesem Artikel werden die Kernelemente der dreidimensionalen Raummodellierung in der städtischen UAV-Routenplanung in geringer Höhe systematisch sortiert:

- **Dreidimensionales Belegungsgitter und Octree** bieten ein einheitliches Umgebungsdarstellungsgerüst basierend auf der Wahrscheinlichkeitstheorie. Als Open-Source-Implementierung wurde OctoMap in Wissenschaft und Industrie umfassend verifiziert;
- **Urban Canyon Effect** erlegt dem UAV-Planungssystem Einschränkungen durch die drei physikalischen Dimensionen GNSS-Dämpfung, Turbulenzstatistik und Bernoulli-Windbeschleunigung auf und muss in der Routenplanung explizit modelliert werden;
- **Das Schichtenmodell des Luftraums** basiert auf traditionellen Ideen der Luftfahrtkontrolle und unterteilt den vertikalen Luftraum zwischen 0 und 120 m² in drei Schichten: G/L/U in städtischen Szenarien und bietet einen strukturellen Rahmen für das groß angelegte Drohnenverkehrsmanagement;
- Die Projektauswahl sollte einen umfassenden Kompromiss zwischen Speichereffizienz, Abfragegeschwindigkeit und dynamischen Aktualisierungsmöglichkeiten eingehen. Die Kombination aus OctoMap + Punktwolke ist die aktuelle Mainstream-Technologieroute.

In den folgenden Kapiteln werden wir uns nach und nach mit Themen wie **Pfadplanungsalgorithmus** (die Anwendung von Abtastalgorithmen wie RRT*/BIT* in dreidimensionalen Octree-Karten), **Echtzeit-Trajektorienoptimierung** (modellprädiktive Steuerung bei Windstörungen in städtischen Schluchten) und **Kollaborative Hindernisvermeidung mit mehreren Flugzeugen** befassen, um ein vollständiges Technologiesystem für die Routenplanung in geringer Höhe in der Stadt aufzubauen.

---

## Referenzen- Hornung, A., Wurm, K. M., Bennewitz, M., Stachniss, C. & Burgard, W. (2013). OctoMap: Ein effizientes probabilistisches 3D-Mapping-Framework basierend auf Octrees. *Autonome Roboter*, 34(3), 189–206. https://doi.org/10.1007/s10514-012-9321-0

- Thomas, H., Farr, R., Yang, C., Chen, Y. und Leonard, J. J. (2021). Lernen von raumzeitlichen Belegungsgitterkarten für die lebenslange Navigation in dynamischen Szenen (arXiv: 2108.10585). arXiv. https://arxiv.org/abs/2108.10585

– Wen, W., Zhang, G. & Hsu, L. T. (2021). *UrbanNav: Ein Open-Source-Lokalisierungsdatensatz zum Benchmarking von Positionierungsalgorithmen für städtische Schluchten* [Datensatz und Dokumentation]. GitHub-Repository: https://github.com/IPNL-POLYU/UrbanNavDataset

- Rotach, M. W. (1995). Profile der Turbulenzstatistik in und über einer städtischen Straßenschlucht. *Atmospheric Environment*, 29(13), 1473–1486. https://doi.org/10.1016/1352-2310(95)00084-D- Zeng, T., Si, B. & Zhao, J. (2020). Umgebungswahrnehmung mit mehreren Granularitäten basierend auf dem Octree-Belegungsraster. *Multimedia-Tools und -Anwendungen*, 79, 27875–27896. https://doi.org/10.1007/s11042-020-09302-w

- Moravec, H. P. & Elfes, A. (1985). Hochauflösende Karten vom Weitwinkel-Sonar. *Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)*, 116–121. https://doi.org/10.1109/ROBOT.1985.1087316

- US-Verkehrsministerium / Federal Aviation Administration. (2023). *Urban Air Mobility (UAM) Betriebskonzept*, Version 2.0. FAA. https://www.faa.gov/air_traffic/nas_management/nas_research/models/uam_conops

- Direktion der NASA-Luftfahrtforschungsmission. (2023). *Zusammenfassung des Projekts „UAS Traffic Management (UTM)“*. NASA. https://utm.arc.nasa.gov/- Hrabar, S. & Sukhatme, G. S. (2004). Ein Vergleich zweier Kamerakonfigurationen für die auf optischem Fluss basierende Navigation durch städtische Schluchten. *Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 3943–3948. https://doi.org/10.1109/IROS.2004.1389989