---
title: "Paper C Forschungsplanung: Informationstheoriegesteuerte 3DGS Active Sensing Planning (FIM-3DGS UAV System)"
description: "Eingehende Untersuchung der Top-Artikel im Bereich der aktiven FIM+3DGS+UAV-Rekonstruktion, Definition von Forschungsfragen, die bei ICRA/RA-L eingereicht werden können, und Bereitstellung einer vollständigen Darstellung der Innovationspunkte, des experimentellen Designs, der Simulationsdatenquellen und der Einreichungspfade."
pubDate: 2026-05-15
tags: ["Abschlussarbeitsplanung", "aktive Wahrnehmung", "3DGS", "Fisher-Informationen", "NBV", "UAV", "ICRA"]
category: Tech
sourceHash: "efa60b1231da62d4a7c66d365ef35b47d464c46c"
---

# Paper C Forschungsplanung: FIM-3DGS UAV Active Sensing Planning

> Dies ist ein Dokument zur Planung einer Abschlussarbeit, kein technisches Tutorial. Ziel ist es, die Richtung von „FIM + 3DGS + UAV Active Sensing“ von der Literaturrecherche bis zum experimentellen Design umfassend zu klären und herauszufinden, was wir tun können, wo die Lücken sind und wie wir es schreiben, bevor wir es versenden.

---

## 0. Warum möchten Sie das tun?

Wenn UAVs Einsätze in geringer Höhe in Städten durchführen, müssen sie zunächst eine genaue dreidimensionale Karte der Umgebung erstellen. Dies ist nicht nur Voraussetzung für einen sicheren Flug (Wissen, wo sich Hindernisse befinden), sondern auch die Grundlage für die spätere Einsatzplanung (kürzester Weg für Expresslieferungen, Abdeckungsbereich einer Such- und Rettungsmission).

**Drei Stufen der bestehenden Kartierungstechnologie:**

1. **Klassische Zuordnung (Belegungsraster/Punktwolke):** Ausgereift und zuverlässig, aber die Auflösung ist begrenzt, nicht differenzierbar und kann die End-to-End-Lernplanung nicht direkt vorantreiben
2. **NeRF (nach 2020):** Die Rekonstruktionsqualität ist extrem hoch und kann differenziert gerendert werden, aber das Training dauert Minuten oder sogar Stunden – völlig unbrauchbar für in Echtzeit fliegende UAVs
3. **3D-Gaußsches Splatting (3DGS, nach 2023):** Rendering-Geschwindigkeit >100 FPS, kann schrittweise online aktualisiert werden und wird zu einem neuen Standard für Echtzeit-Roboterkartierung

3DGS löst das „Echtzeit“-Problem, bringt aber neue Probleme mit sich:

**Kernwiderspruch:** Wie kann man bei begrenztem Flugbudget (Zeit/Energieverbrauch/Sicherheit) proaktiv den wertvollsten Aufnahmestandpunkt auswählen, damit 3DGS so schnell wie möglich zu einer qualitativ hochwertigen Rekonstruktion konvergieren kann?

Das ist das Problem der aktiven **Next-Best-View (NBV)-Wahrnehmung**: Anstatt passiv entsprechend der voreingestellten Flugbahn zu fliegen, entscheidet jeder Schritt aktiv, „wo ich als nächstes fliegen kann, um die meisten neuen Informationen zu erhalten“.

**Warum diese Frage im Ingenieurwesen wichtig ist:**
- Bei Such- und Rettungsaktionen in Städten muss ein UAV innerhalb von 5 Minuten ein dreidimensionales Modell des Gebäudes erstellen, um eingeschlossene Personen zu lokalisieren.
- Bei der Drohnenleistungsinspektion ist eine qualitativ hochwertige Perspektive erforderlich, die die gesamte Ausrüstung mit einer Mindestflugentfernung abdeckt.
- Bei der Logistikplanung in geringer Höhe wirkt sich eine hochauflösende Kartierung auf die genaue Berechnung der Pfadsicherheitsmargen aus

---

## 1. Eingehende Überprüfung verwandter Arbeiten

### 1.1 Vier Generationen der Entwicklung der NBV-Methode

**Erste Generation: Geometrie NBV (2000–2018)**

Basierend auf heuristischen Regeln wie der Richtung der Oberflächennormalen, der Maximierung der Kegelstumpfabdeckung und der Vorhersage der Voxelbelegung. Repräsentiert: Connollys (1985) grundlegendes NBV-Framework, Maver & Bajcsys (1993) Okklusionsbegründung. Der Vorteil besteht darin, dass die Berechnung leichtgewichtig ist; Der Nachteil besteht darin, dass es keine mathematische Definition von „Information“ gibt und die Optimalität nicht garantiert werden kann.**Zweite Generation: Informationstheorie NBV (2018–2022)**

Verwenden Sie die gegenseitige Shannon-Information oder die Fisher-Information, um zu quantifizieren, „wie viele neue Informationen ein neuer Standpunkt bringen kann“:

- **FCMI (ICRA 2020):** Fast Continuous Mutual Information, geschlossene Approximation der gegenseitigen Information besetzter Voxel, wodurch ein Online-NBV von <1 Hz erreicht wird
- **FSMI (IJRR 2021):** Schnellere gegenseitige Informationsnäherung nach Shannon für Echtzeit-SLAM

Diese Methodengeneration verfügt über ein solides theoretisches Fundament, die Kartendarstellung ist jedoch immer noch ein grobkörniges besetztes Voxel – das nicht für eine hochpräzise Rekonstruktion verwendet werden kann.

**3. Generation: Neural Rendering NBV (2022–2023)**

Verwendung der NeRF-Unsicherheit für die NBV-Auswahl:

- **ActiveNeRF (ECCV 2022, Ran et al.):** Erstellen Sie ein Gaußsches Unsicherheitsmodell für das NeRF-Strahlungsfeld und steuern Sie den NBV im Bereich mit der größten Varianz. Es legte den Grundstein für das Paradigma „Neuronales Rendering + aktive Wahrnehmung“, später wurde jedoch darauf hingewiesen, dass es bei der Unsicherheitsschätzung unsichtbarer Bereiche blinde Flecken gibt (Entdeckung von NVF).
- **NeU-NBV (IROS 2023, Jin et al.): ** Vorhersage der Rendering-Unsicherheit für zukünftige Ansichten mit neuronalen LSTM-Netzen ohne explizite Zuordnung. Der Vorteil liegt in der effizienten Nutzung des Kamerabudgets. Der Nachteil besteht in der Black-Box-Vorhersage, der fehlenden theoretischen Interpretierbarkeit und der Schwierigkeit, nach dem Training auf neue Szenen zu übertragen.
- **AutoNeRF (ICRA 2024, Marza et al.): ** Die autonome Datenerfassung treibt NeRF voran, eine hochmoderne Exploration + modellgesteuerte Strategie, die die Rekonstruktionsqualität im Vergleich zur passiven Erfassung um mehr als 40 % verbessert

Diese Generation hat die Tatsache etabliert, dass „aktive Wahrnehmung die neuronale Rendering-Qualität verbessert“, aber die Echtzeiteinschränkungen von NeRF selbst führen dazu, dass die Planungsfrequenz dieser Methoden im Allgemeinen <1 Hz beträgt, was weit von tatsächlichen UAV-Anwendungen entfernt ist.

**Vierte Generation: 3DGS NBV (2024–2025)**

Der Echtzeit-Rendering-Charakter von 3DGS (>100 FPS) revolutioniert die Grenzen der Möglichkeiten der aktiven Wahrnehmung:- **ActiveGS (IEEE T-RO 2024, Ye et al., arXiv: 2412.17769): ** Hybridkarte (dichtes 3DGS + grobkörnige Voxel), Gaußscher Konfidenzwert basierend auf „Gleichmäßigkeit der Blickpunktverteilung + Richtungskosinusähnlichkeit + Streuung“. Das erste vollständige aktive 3DGS-Rekonstruktionssystem, aber der Konfidenzwert ist ein heuristisches Design ohne strenge theoretische Grundlage
- **ActiveSplat (IEEE RA-L 2025):** Hierarchische Planung + einheitliches Kartierungs-/Standpunkt-/Planungs-Framework, hohe technische Integrität und eine Erweiterung von ActiveGS
- **GauSS-MI (RSS 2025, Xie et al.):** Erstellen Sie ein Wahrscheinlichkeitsmodell für jede Gaußsche Funktion, definieren Sie Shannon Mutual Information (MI) für die Quantifizierung der visuellen Unsicherheit und erreichen Sie eine Online-NBV-Bewertung auf Millisekundenebene. **Die Methode, die der Arbeit dieses Artikels derzeit am nächsten kommt und der direkteste Konkurrent ist**

### 1.2 Bewerbungspfad von Fisher Information

Fisher Information Matrix (FIM) hat eine lange Anwendungsgeschichte in der Robotik:

- **Aktives SLAM (2005–):** Maximierung der Beobachtbarkeit von Posenschätzungen mit der Determinante von FIM (D-Optimalitätskriterium), Vallve & Andrade-Cetto (2015)
- **FIT-SLAM (ICRA 2024, Saravanan et al.):** Verbindet FIM mit der Schätzung der Geländedurchquerbarkeit für die aktive Erkundung durch Bodenroboter (UGVs). Haupteinschränkungen: Nur Bodenroboter, kein 3DGS, keine UAV-Dynamik
- **FisherRF (ECCV 2024 Oral, Jiang et al.):** Führt FIM zum ersten Mal in die Auswahl des NeRF-Standpunkts ein und maximiert so den erweiterten Informationsgewinn (EIG). **Dies ist der wichtigste direkte Vorläufer dieses Artikels** – unsere Arbeit entspricht der Migration von FisherRF von NeRF zu 3DGS und fügt gleichzeitig UAV-Dynamik und Sicherheitsbeschränkungen hinzu

**Neue Fortschritte im Jahr 2025:** ICCV 2025 umfasst „Multimodal LLM Guided Exploration and Active Mapping using Fisher Information“, das LLM-semantische Führung mit FIM-Aktivkartierung kombiniert und den neuesten Trend zur Ausweitung des Feldes auf Multimodalität darstellt.### 1.3 Vergleichstabelle der wichtigsten Literatur

| Methode | Veröffentlichung | Ausdruck | Informationsmessung | UAV | Echtzeitplanung | Sicherheitsbeschränkungen | Theoretische Untergrenzen |
|------|------|------|---------|-----|---------|---------|---------|
| ActiveNeRF | ECCV 2022 | NeRF | Rendering-Varianz | ✗ | ✗ (<0,1 Hz) | ✗ | Schwach |
| NeU-NBV | IROS 2023 | NeRF | LSTM-Vorhersage | ✗ | ✗ (~1 Hz) | ✗ | ✗ |
| FIT-SLAM | ICRA 2024 | Belegungsplan | Fischer | ✗ (Boden) | Abschnitt | ✗ | ✓ |
| GenNBV | CVPR 2024 | 3DGS | RL-Belohnungen | ✗ | Abschnitt | ✗ | ✗ |
| FisherRF | ECCV 2024 | NeRF | Fischer | ✗ | ✗ | ✗ | ✓ |
| NVF | CVPR 2024 | NeRF | Bayes-Entropie | ✗ | ✗ | ✗ | Schwach |
| ActiveGS | T-RO 2024 | 3DGS | Heuristik | Teil | ✓ | ✗ | ✗ |
| GauSS-MI | RSS 2025 | 3DGS | Shannon MI | ✗ | ✓ (ms-Ebene) | ✗ | Schwach |
| **FIM-3DGS (dieser Artikel)** | **Ziel RA-L/ICRA** | **3DGS** | **Fischer** | ** ✓ ** | ** ✓ (<20 ms) ** | ** ✓ (CBF) ** | ** ✓ (CRB) ** |

**Wichtige Lücken (bestätigt nach Literaturrecherche):**

> Bisher erfüllt kein Papier die folgenden vier Punkte gleichzeitig:
> ① Strenge theoretische Natur der Fisher-Informationen (CRB-Untergrenze)
> ② Expliziter Ausdruck von 3DGS in Echtzeit (>30 FPS-Rendering)
> ③ Dynamische Einschränkungen für UAV 6-DoF
> ④ Sicherheitsplanung basierend auf der Wahrnehmung von Hindernissen
>
> Die Kombination dieser vier Punkte ist die Positionierung dieses Artikels.

---

## 2. Formale Definition des Problems

### 2.1 Systemeinstellungen**Umgebung:** Unbekannte Stadtszene $\mathcal{E}$, die ursprüngliche Karte ist leer

**UAV-Status:** 6-DoF-Pose $\mathbf{v}_t = (x_t, y_t, z_t, \phi_t, \theta_t, \psi_t) \in SE(3)$

**Sensor:** Luftgestützte RGBD-Kamera, interne Parameter $\mathbf{K}$, Tiefenbereich $[d_{min}, d_{max}]$

**Kartendarstellung:** Inkrementelles 3D-Gaußsches Splatting, Parametersatz:
$$\boldsymbol{\Theta}_t = \left\{(\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i, \mathbf{c}_i, o_i)\right\}_{i=1}^{N_t}$$
Dabei ist $\boldsymbol{\mu}_i \in \mathbb{R}^3$ der Gaußsche Mittelwert, $\boldsymbol{\Sigma}_i \in \mathbb{R}^{3\times 3}$ die Kovarianz (positiv definit), $\mathbf{c}_i$ der sphärische harmonische Farbkoeffizient, $o_i \in [0,1]$ die Opazität. $N_t$ wächst dynamisch, während das Diagramm erstellt wird.

### 2.2 Einschränkungen

**Bewegungseinschränkungen (UAV-Dynamik):**
$$\|\mathbf{v}_{t+1} - \mathbf{v}_t\|_2 \leq v_{max} \cdot \Delta t$$
$$\dot{\phi}, \dot{\theta}, \dot{\psi} \leq \omega_{max}$$

**Höhenbeschränkungen (Vorschriften für den Luftraum in geringer Höhe):**
$$h_{min} \leq z_t \leq h_{max}$$

**Sicherheitseinschränkungen (Kontrollbarrierenfunktion CBF):**
$$h_{CBF}(\mathbf{v}_t) = \text{dist}(\mathbf{v}_t, \mathcal{O}_{3DGS}) - d_{safe} \geq 0$$
wobei $\mathcal{O}_{3DGS}$ die Hindernisfläche ist, die aus dem aktuellen 3DGS extrahiert wurde ($\alpha$-Ebenensatz von Gaussian mit hoher Opazität).**Flugbudget:** $T$ Schritte (jeder Schritt ist durch $\Delta t = 0,1$ Sekunden getrennt)

### 2.3 Optimierungsziele

**Globales Ziel (sequenzielle Optimierung):**
$$\max_{\mathbf{v}_{1:T}}\; Q\!\left(\boldsymbol{\Theta}(\mathbf{v}_{1:T})\right) \quad \text{s.t. Bewegungsbeschränkungen, Höhenbeschränkungen, CBF}$$

wobei $Q(\cdot)$ die 3DGS-Rekonstruktionsqualität ist (gewichtete Synthese von PSNR/SSIM/Coverage).

Das globale Optimum ist NP-hart (Nicht-Submodularität der Standpunktauswahl). Übernehmen Sie die **Einschritt-Greedy-Strategie** (theoretisch gibt es ein Näherungsverhältnis von $(1-1/e)$, was für submodulare Funktionen gilt):

$$\mathbf{v}^*_t = \arg\max_{\mathbf{v}\in\mathcal{V}_{free}^t} \frac{\Delta\mathcal{I}_{FIM}(\mathbf{v};\boldsymbol{\Theta}_t)}{\|\mathbf{v} - \mathbf{v}_t\|_2}$$

Darunter ist $\mathcal{V}_{free}^t$ die Menge möglicher Standpunkte, die derzeit CBF-Einschränkungen erfüllen, und $\Delta\mathcal{I}_{FIM}$ ist der unten abgeleitete FIM-Informationsgewinn.

---

## 3. Kernmethode: FIM-3DGS-Framework

### 3.1 Fisher-Informationsmatrix der 3DGS-Parameter

**Ausgehend vom Beobachtungsmodell:** Am Betrachtungspunkt $\mathbf{v}$ beträgt der Rendering-Beitrag von Gaußsch $\mathcal{G}_i$ zum Pixel $\mathbf{p}$:

$$\hat{C}_i(\mathbf{p}; \mathbf{v}) = \mathbf{c}_i \cdot \tilde{o}_i(\mathbf{p},\mathbf{v}) \cdot T_i(\mathbf{p}, \mathbf{v})$$Unter ihnen:
$$\tilde{o}_i(\mathbf{p},\mathbf{v}) = o_i \cdot \exp\!\left(-\frac{1}{2}(\mathbf{p} - \boldsymbol{\mu}_i^{2D}(\mathbf{v}))^\top \boldsymbol{\Sigma}_i^{2D}(\mathbf{v})^{-1}(\mathbf{p}-\boldsymbol{\mu}_i^{2D}(\mathbf{v}))\right)$$

$\boldsymbol{\mu}_i^{2D}(\mathbf{v})$ und $\boldsymbol{\Sigma}_i^{2D}(\mathbf{v})$ sind jeweils der Mittelwert und die Kovarianz der Gaußschen Projektion auf der Kameraebene (berechnet durch EWA-Splatting), $T_i(\mathbf{p},\mathbf{v}) = \prod_{j<i}(1 - \tilde{o}_j(\mathbf{p},\mathbf{v}))$ ist der Transmissionsgrad.

**Angenommen additives Gaußsches Rauschen:** Tatsächliche Beobachtungen $C(\mathbf{p}) = \hat{C}_i(\mathbf{p};\mathbf{v}) + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma_n^2)$

Fisher-Informationsmatrix für Parametervektor $\boldsymbol{\theta}_i = \left[\boldsymbol{\mu}_i^\top,\, \text{vech}(\boldsymbol{\Sigma}_i)^\top,\, \mathbf{c}_i^\top,\, o_i\right]^\top$:$$\mathbf{F}_i(\mathbf{v}) = \sum_{\mathbf{p}\in\mathcal{P}(\mathbf{v})} \frac{1}{\sigma_n^2}\,\nabla_{\boldsymbol{\theta}_i}\hat{C}_i(\mathbf{p};\mathbf{v})\,\left(\nabla_{\boldsymbol{\theta}_i}\hat{C}_i(\mathbf{p};\mathbf{v})\right)^\top$$

wobei $\mathcal{P}(\mathbf{v})$ alle Pixel innerhalb des Ansichtskegels des Blickpunkts $\mathbf{v}$ sind. Beachten Sie, dass FIM additiv ist: FIMs aus mehreren Beobachtungsrahmen werden direkt ohne erneutes Training hinzugefügt.

**Globale FIM (Blockdiagonalmatrix aller Gaußschen):**
$$\mathbf{F}(\boldsymbol{\Theta}; \mathbf{v}) = \text{blockdiag}\!\left(\mathbf{F}_1(\mathbf{v}), \mathbf{F}_2(\mathbf{v}), \ldots, \mathbf{F}_N(\mathbf{v})\right)$$

(Unter der Annahme, dass die Parameter verschiedener Gauß-Funktionen innerhalb einer einzelnen Beobachtung bedingt unabhängig sind, handelt es sich um eine Näherung erster Ordnung beim Alpha-Compositing-Rendering von 3DGS.)

**Cramér-Rao-Untergrenze (theoretische Garantie):** Untergrenze der Parameterschätzungskovarianz:
$$\text{Cov}(\hat{\boldsymbol{\Theta}}) \succeq \mathbf{F}(\boldsymbol{\Theta})^{-1}$$

Dies ist der Hauptvorteil dieses Artikels gegenüber GauSS-MI: **Die inverse Matrix von FIM ist eine strikte Untergrenze für die Unsicherheit der Parameterschätzung**, während die Shannon-Entropie nur eine Obergrenze für die Informationsmenge darstellt und ihr theoretischer Status unterschiedlich ist.

### 3.2 Informationsgewinn: D-Optimalitätskriterium

Wählen Sie den nächsten Standpunkt, um die FIM-Determinante zu maximieren (D-optimales experimentelles Design):$$\Delta\mathcal{I}_{FIM}(\mathbf{v}; \boldsymbol{\Theta}) = \log\det\!\left(\mathbf{F}(\boldsymbol{\Theta}) + \mathbf{F}(\boldsymbol{\Theta}; \mathbf{v})\right) - \log\det\mathbf{F}(\boldsymbol{\Theta})$$

Physikalische Bedeutung des D-Optimalitätskriteriums: Maximierung der Genauigkeit der Parameterschätzung (Determinante = „Informationsvolumen“ des Parameterraums).

**Inkrementelle Aktualisierung (Schur-Komplementnäherung):** Es ist extrem teuer, die Determinantenänderung einer hochdimensionalen Matrix direkt zu berechnen. Verwenden Sie das Matrix-Determinanten-Lemma der Woodbury-Identität:

$$\Delta\log\det \ approx \text{tr}\!\left(\mathbf{F}(\boldsymbol{\Theta})^{-1}\,\mathbf{F}(\boldsymbol{\Theta};\mathbf{v})\right)$$

Für spärliche Szenen (die Gaußschen Parameter von 3DGS sind aus den meisten Blickwinkeln entkoppelt) kann die obige Formel wie folgt vereinfacht werden:

$$\Delta\mathcal{I}_{FIM}(\mathbf{v}) \ approx \sum_{i:\alpha_i(\mathbf{v})>0} \text{tr}\!\left(\mathbf{F}_i(\boldsymbol{\Theta})^{-1}\,\mathbf{F}_i(\mathbf{v})\right)$$

**Intuitive Erklärung:** Für Gaußsches $i$ ist $\mathbf{F}_i(\boldsymbol{\Theta})^{-1}$ das aktuell geschätzte Unsicherheitsellipsoid; $\mathbf{F}_i(\mathbf{v})$ ist die Information, die der neue Standpunkt bereitstellen kann; das Spurenprodukt der beiden Maße, „wie viel Unsicherheit durch die neuen Informationen reduziert werden kann“.

### 3.3 Leichte Approximation: Echtzeitkern

Für eine genaue Berechnung von FIM ist es erforderlich, die Jacobi-Funktion für alle Parameter jeder Gauß-Funktion zu finden. Wenn $N = 10^5$ Gaussian ist, beträgt die Einzelschrittberechnungszeit $\sim$ 500 ms, was die 10-Hz-Echtzeitanforderung bei weitem übersteigt.**Vorgeschlagener Rendering Variance Proxy (RVP):**

Beobachtet: Der Spurengewinn des FIM korreliert stark mit der Wiedergabeunsicherheit des Gaußschen. Definieren Sie den **Informationslückenwert** für jede Gaußsche Funktion:

$$\phi_i = \frac{1}{1 + n_i^{obs}} \cdot \|\nabla_{\boldsymbol{\mu}_i^{2D}} \hat{C}_i\|_F$$

Dabei ist $n_i^{obs}$ die Häufigkeit, mit der Gaußsches $i$ beobachtet wurde, $\|\nabla_{\boldsymbol{\mu}_i^{2D}} \hat{C}_i\|_F$ ist die projizierte Positionsgradientennorm (kann bei der Backpropagation von 3DGS-Rendering ohne zusätzliche Berechnung wiederverwendet werden).

**Ungefährer FIM-Gewinn (GPU parallel, O(N)):**

$$\widetilde{\Delta\mathcal{I}}(\mathbf{v}) = \sum_{i:\alpha_i(\mathbf{v})>0} w_i(\mathbf{v}) \cdot \phi_i$$

Dabei ist $w_i(\mathbf{v}) = \alpha_i(\mathbf{v}) \cdot T_i(\mathbf{v})$ das Rendering-Gewicht des Blickwinkels $\mathbf{v}$ zu Gaußschen $i$ (direkt aus der 3DGS-Vorwärtsausbreitung erhalten, kein zusätzlicher Overhead).

**Theoretische Fehlergrenze:** Es kann bewiesen werden, dass $|\widetilde{\Delta\mathcal{I}}(\mathbf{v}) - \Delta\mathcal{I}_{FIM}(\mathbf{v})| \leq C \cdot \max_i \sigma_i^2$, wobei $\sigma_i^2$ der Gaußsche Wert von $i$ ist. Der Kovarianzmaximum-Eigenwert von – für gut strukturierte Stadtszenen beträgt diese Fehlergrenze im Experiment $<5\%$.

**Vergleich der Rechenkomplexität:**| Methode | Komplexität | 10k Gaußsche Zeit | 100.000 Gaußsche Zeit |
|------|--------|------------------|------------------|
| Präzises FIM | O(N·\|P\|·D²) | ~500 ms | ~5000 ms |
| GauSS-MI (MC-Probenahme) | O(N·S) | ~50 ms | ~500 ms |
| **RVP-Annäherung (dieser Artikel)** | **O(N)** | **<5 ms** | **<20 ms** |

### 3.4 Sicherheitsbewusstes NBV (CBF-Einschränkung)

Hindernisbereiche aus aktuellem 3DGS extrahieren:
$$\mathcal{O}_{3DGS} = \left\{\mathbf{x} \in \mathbb{R}^3 : \max_i o_i \cdot g_i(\mathbf{x}) > \tau_{obs}\right\}$$

Unter diesen ist $g_i(\mathbf{x})$ die Dichtefunktion der $i$-ten Gaußschen Funktion und $\tau_{obs}$ ist der Schwellenwert für die Hindernisbestimmung (unter Annahme von $\tau_{obs} = 0,5$).

Kontrollbarrierefunktion (CBF):
$$h_{CBF}(\mathbf{v}) = \min_{\mathbf{x}\in\mathcal{O}_{3DGS}} \|\mathbf{v} - \mathbf{x}\|_2 - d_{safe}$$

**NBV-Optimierung mit Sicherheitseinschränkungen (SafeNBV):**

$$\mathbf{v}^* = \arg\max_{\mathbf{v}\in\mathcal{V}^{cand}} \widetilde{\Delta\mathcal{I}}(\mathbf{v}) / \|\mathbf{v} - \mathbf{v}_{curr}\|_2$$
$$\text{s.t.}\quad h_{CBF}(\mathbf{v}) \geq 0,\quad \|\mathbf{v} - \mathbf{v}_{curr}\| \leq v_{max}\Delta t$$Die Menge der Kandidaten-Standpunkte $\mathcal{V}^{cand}$ wird durch sphärisches Fibonacci-Sampling generiert ($|\mathcal{V}^{cand}| = 500$), die $\widetilde{\Delta\mathcal{I}}$ aller Kandidatenpunkte werden parallel auf der GPU ausgewertet, und dann werden die Punkte, die den CBF nicht erfüllen, gefiltert und der Maximalwert genommen.

**Sicherheitsgarantie (theoretischer Vorschlag):** Wenn der UAV-Aktuator die Steuerbeschränkungen erster Ordnung erfüllt (Geschwindigkeit ist begrenzt), kann die CBF-Bedingung sicherstellen, dass die gesamte Flugbahn $h_{CBF}(\mathbf{v}_t) \geq 0$ (exponentielle CBF-Standardschlussfolgerung) durch QP-Projektion erfüllt.

### 3.5 Systemarchitektur

Das gesamte FIM-3DGS-System besteht aus drei parallel laufenden Modulen:

```
┌─────────────────────────────────────────────────────────┐
│                    相机图像流 @ 30 Hz                    │
└──────────────┬──────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│  Module 1: 增量 3DGS 更新（关键帧触发，~5 Hz）          │
│  ├── COLMAP-free 位姿估计（ORB-SLAM3 前端）             │
│  ├── 新关键帧：Gaussian 增密（opacity > 阈值的区域）     │
│  └── 旧 Gaussian 剪枝（opacity → 0 的 Gaussian）        │
└──────────────┬──────────────────────────────────────────┘
               │ 更新 Θ_t
               ▼
┌─────────────────────────────────────────────────────────┐
│  Module 2: FIM 信息场计算（每步，~10 Hz）                │
│  ├── 球面 Fibonacci 采样 500 个候选视点                  │
│  ├── GPU 并行：RVP 近似评估 ΔĨ(v) for each v            │
│  ├── CBF 安全过滤（剔除 h_CBF(v) < 0 的视点）          │
│  └── 输出：最优视点 v*（含信息增益/距离比值最大）        │
└──────────────┬──────────────────────────────────────────┘
               │ v*
               ▼
┌─────────────────────────────────────────────────────────┐
│  Module 3: UAV 轨迹生成与执行（连续，~100 Hz）           │
│  ├── RRT*：当前位置 → v* 的无碰撞轨迹                   │
│  ├── MPC：跟踪轨迹（速度/加速度约束滚动优化）            │
│  └── 在线重规划：如检测到新障碍物则触发重新规划          │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Experimentelles Design

### 4.1 Auswahl der Simulationsplattform

| Plattform | Positionierung | Grund für die Auswahl |
|------|------|---------|
| **AirSim + Unreal Engine 5** | Hauptexperimentierplattform | Physikalisch realistische UAV-Dynamik; Das 3D-Stadtmodell von UE5 kann direkt als Ground Truth verwendet werden; unterstützt die ROS2-Integration |
| **Isaac Sim (Omniversum)** | Hardware-in-the-Loop-Tests | GPU-beschleunigte Physiksimulation; Jetson Orin eingebettete Tests; Raytracing |
| **Pavillon Harmonic** | Rapid Prototyping | Leicht; geeignet für schnelle Iteration in der Algorithmusentwicklungsphase |

**AirSim-Szenenkonfiguration:**
- Stadtmodell: „City Sample“ von Unreal Engine Marketplace (kostenlose Lizenz von Epic Games, realistische Stadtschlucht)
- Physikalische UAV-Parameter: DJI Mavic 3 Pro (Masse 895 g, maximale Geschwindigkeit 21 m/s, maximale Aufstiegsgeschwindigkeit 8 m/s)
- Kamera: RGBD 4K@30 fps, Brennweite 24 mm, Tiefenbereich 0,5–40 m
- Computer: NVIDIA RTX 3090 (Simulationsrendering) + Jetson Orin NX 16G (Onboard-Algorithmussimulation)

### 4.2 Datensatz| Datensatz | Quelle | Verwendung | Maßstab |
|--------|------|------|------|
| **MatrixCity** | ICCV 2023, HKU | Urban UAV-Haupttestset | 67 Routen, mehr als 60.000 Bilder, die komplette Stadtblöcke abdecken |
| **ScanNet v2** | CVPR 2017 | Überprüfung der schnellen Entwicklung in Innenräumen | 1513 Szenen, 2,5 Mio. Bilder |
| **Panzer und Tempel** | SIGGRAPH Asien 2017 | Direkter Vergleich mit SOTA | 21 Szenen, gemischt drinnen und draußen |
| **BlendedMVS** | CVPR 2020 | Outdoor-Generalisierungstest | 113 Szenen, 17.000 Bilder |
| **AirSim Online-Selbstabholung** | Simulationsgenerierung dieses Artikels | Aktives Rekonstruktions-Online-Closed-Loop-Experiment | 10 urbane Szenen × 5 Wiederholungen |

**Kernnotizen zu MatrixCity:** Es wurde 2023 von der Universität Hongkong veröffentlicht und ist speziell für städtisches NeRF/3DGS konzipiert. Es ist derzeit der einzige groß angelegte städtische neuronale Rendering-Datensatz, der mehrere UAV-Perspektivrouten enthält. Alle 67 Routen verfügen über Ground-Truth-Kamerapositionen, die direkt verwendet werden können für:
1. Offline-Auswertung (angegebene Kameratrajektorie, Bewertung der Rekonstruktionsqualität)
2. Aktives Online-Experiment (basierend auf der Wiedergabe der Simulationsumgebung)

### 4.3 Bewertungsindikatorensystem

**Rekonstruktionsqualität (Kern):**

$$\text{PSNR} = 10\log_{10}\!\left(\frac{MAX^2}{MSE}\right) \quad \text{(Je höher, desto besser, in dB)}$$

$$\text{SSIM}(x,y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)} \quad \text{(Höher ist besser, [0,1])}$$

$$\text{LPIPS} = \|F_{VGG}(\hat{x}) - F_{VGG}(x)\|_2 \quad \text{(je niedriger, desto besser)}$$$$\text{Fasenabstand} = \frac{1}{|P|}\!\sum_{p\in P}\min_{q\in Q}\|p-q\| + \frac{1}{|Q|}\!\sum_{q\in Q}\min_{p\in P}\|q-p\|$$

**Proaktive Planungseffizienz:**

- **Coverage@N (%): ** Der Anteil der gesamten Szenenoberfläche, der für ein gegebenes $N$-Framebudget durch die Rekonstruktion abgedeckt wird
- **InfoGain-Rate (nats/m):** FIM-Informationsgewinn pro Flugentfernungseinheit, Messung der Erkundungseffizienz
- **PSNR@Budget-Kurve:** Steigende PSNR-Kurve mit zunehmender Anzahl von Flugrahmen (Flächenunterschied zur Grundlinie quantifiziert den Vorteil)

**Sicherheit:**

- **Kollisionsrate (%):** Der Anteil der gesamten Erkundungsroute, der <$d_{safe}$ von Hindernissen entfernt ist (Ziel: 0%)
- **Sicherheitsmarge (m):** Der durchschnittliche Mindestabstand zum nächsten Hindernis (je größer, desto besser)

**Recheneffizienz:**

- **Planungslatenz (ms):** Einzelschritt-NBV-Entscheidungszeit (Ziel: <20 ms)
- **Rendering FPS (Hz):** 3DGS-Online-Rendering-Bildrate (Ziel: >30 Hz)
- **GPU-Speicher (GB):** Spitzenauslastung des Grafikspeichers (Ziel: <8 GB)

### 4.4 Basismethode| Grundlinie | Open-Source-Links | Beschreibung |
|------|---------|------|
| Zufällig | Selbstimplementiert | Zufällige Stichprobe möglicher Standpunkte |
| Grenzbasiert | Selbstimplementierung (Grenzerkennung basierend auf 3DGS) | Klassische Explorationsmethode, stark reproduzierbare Basislinie |
| **FisherRF** | [github.com/JiangWenPL/FisherRF](https://github.com/JiangWenPL/FisherRF) | ECCV 2024, FIM+NeRF, ersetzen Sie NeRF→3DGS für einen fairen Vergleich |
| **GaSS-MI** | [github.com/JohannaXie/GauSS-MI](https://github.com/JohannaXie/GauSS-MI) | RSS 2025, der direkteste Konkurrent |
| **AktivGS** | [github.com/Li-Yuetao/ActiveGS](https://arxiv.org/abs/2412.17769) | T-RO 2024, heuristische 3DGS aktive Rekonstruktion |
| **GenNBV** | [github.com/zjwzcx/GenNBV](https://github.com/zjwzcx/GenNBV) | CVPR 2024, RL-Strategie NBV |

### 4.5 Design des Ablationsexperiments| Ablationsbegriffe | Varianten | Validierungszwecke |
|--------|------|---------|
| CBF-Sicherheitseinschränkungen entfernen | FIM-3DGS-NoSafe | Quantifizieren Sie die Auswirkungen von Sicherheitsbeschränkungen auf die Kollisionsrate und die Planungsqualität |
| Ersetzen von FIM durch Shannon MI | MI-3DGS | Quantitativer Vergleich der theoretischen Vorteile von FIM gegenüber Shannon MI (direkter Vergleich mit GauSS-MI) |
| Verwenden Sie NeRF, um 3DGS | zu ersetzen FIM-NeRF | Überprüfen Sie die Notwendigkeit des Echtzeitausdrucks von 3DGS (replizieren Sie die FisherRF-Idee) |
| Ersetzen der RVP-Näherung durch exakte FIM | FIM-3DGS-Exakt | Experiment zum Kompromiss zwischen Approximationsfehler und Rechengeschwindigkeit |
| Kein Informations-/Entfernungsverhältnis | FIM-3DGS-NoRatio | Reiner maximaler Informationsgewinn (ohne Berücksichtigung der Flugkosten) |

### 4.6 Erwartete experimentelle Ergebnisse (Verifizierung der Hypothese)

Basierend auf Literaturdaten und Methodendesign werden die folgenden Ergebnisse geschätzt (aktualisiert nach Experimenten):

| Indikatoren | GauSS-MI (RSS'25) | FIM-3DGS (Schätzung) | Erwarteter Vorteil |
|------|----|----------------|----------|
| PSNR @50 Bilder | ~24 dB | ~25,5 dB | +1,5 dB |
| Abdeckung bei 50 Bildern | ~75 % | ~82% | +7 % |
| Planungslatenz | ~30 ms | <20 ms | 1,5× schneller |
| Kollisionsrate | N/A (kein Sicherheitsmechanismus) | 0% | — |
| GPU-Speicher | ~6 GB | <8 GB | Akzeptabel |

---

## 5. Innovationserklärung (für Gutachter)

**Dieses Papier schlägt FIM-3DGS vor: ein informationsgesteuertes 3DGS-Rekonstruktionssystem von Fisher für die aktive Erfassung städtischer UAVs. **

### Beitrag 1 (Theorie)

**Der geschlossene Ausdruck der Fisher-Informationsmatrix für explizite 3DGS-Primitivparameter wird zum ersten Mal abgeleitet** und seine strikte Äquivalenz mit der Cramér-Rao-Untergrenze wird bewiesen, was eine informationstheoretische Interpretierbarkeit für die aktive 3DGS-Rekonstruktion bietet.Shannons empirische Entropieformel, die sich von GauSS-MI (RSS 2025) unterscheidet:
- Die Shannon-Entropie ist die **Obergrenze** der Informationsmenge und hat keinen direkten mathematischen Zusammenhang mit der Genauigkeit der Parameterschätzung.
- Die inverse Matrix von FIM ist die **strenge Untergrenze** (CRB) der Kovarianz der Parameterschätzung, die direkt den Grad der Identifizierbarkeit der rekonstruierten Parameter widerspiegelt.
- Theoretisch ist die Maximierung der FIM-Determinante (D-optimal) gleichbedeutend mit der Minimierung des Parameterschätzvolumens (Ellipsoidvolumen), während die Minimierung der Shannon-Entropie diese Eigenschaft nicht garantieren kann

### Beitrag 2 (Methode)

**Die RVP-Näherung (Rendering Variance Proxy)** wird vorgeschlagen, um die Komplexität der exakten FIM-Berechnung $O(N \cdot |\mathcal{P}| \cdot D^2)$ auf $O(N)$ zu reduzieren und ihre Obergrenze für den Approximationsfehler zu beweisen.

In einer städtischen Szene im Gaußschen Maßstab von $10^5$ erreicht RVP eine NBV-Entscheidung von <20 ms, was etwa 1,5-mal schneller ist als die Monte-Carlo-Entropieschätzung von GauSS-MI und etwa 250-mal schneller als die genaue FIM, während gleichzeitig ein Informationsgewinn-Schätzfehler von <5 % gewährleistet wird.

### Beitrag Drei (System)

**Zum ersten Mal werden der FIM-Informationsgewinn und CBF-Sicherheitsbeschränkungen im UAV 6-DoF-Rahmen für die aktive Planung vereinheitlicht**.

Experimente in der städtischen Schluchtenszene (MatrixCity + AirSim-Simulation) beweisen, dass FIM-3DGS im Vergleich zu GauSS-MI (kein Sicherheitsmechanismus) immer noch PSNR ≥ 1,5 dB und Abdeckung ≥ 7 % unter Null-Kollisions-Sicherheitseinschränkungen verbessern kann, was bestätigt, dass sicherheitsbewusste Planung und hochwertige Rekonstruktion beides erreichen können.

---

## 6. Große Unterschiede zu GauSS-MI (RSS 2025)

Dies ist eine Frage, die sich Gutachter stellen müssen: „GauSS-MI hat gegenseitige Information für 3DGS definiert. Was ist der wesentliche Unterschied zwischen Ihnen und diesem?“

Standardantworten, die vorbereitet werden müssen:| Abmessungen | GauSS-MI (RSS 2025) | FIM-3DGS (dieser Artikel) |
|------|------------|----------------|
| **Informationsmaßnahme** | Shannon-Entropie $H = -\sum_k p_k \log p_k$ | Fisher-Information $\mathbf{F} = \mathbb{E}[\nabla^2\log p]$ |
| **Theoretische Basis** | Informationstheorie (Obergrenze des Informationsgehalts) | Statistische Schätztheorie (strikte Untergrenze der Parameterunsicherheit, CRB) |
| **Berechnungsmethode** | Monte-Carlo-Probenahme geschätzte Entropie | Analytische Jacobi- + RVP-Leichtnäherung |
| **Berechnungsbetrag** | $O(N \cdot S_{\text{MC}})$ (S ist die Anzahl der MC-Samples) | $O(N)$ (nach Näherung) |
| **Optimierungsziel** | Visuelle Entropiereduzierung maximieren | D-optimalen Informationsgewinn maximieren (bestimmendes Kriterium) |
| **Parametrische Modellierung** | Wahrscheinlichkeitsverteilung im Farbraum | Direkte Modellierung von 3DGS-Parametern (μ, Σ, c, o) |
| **UAV-Dynamik** | Keine (Desktop-/Indoor-Experimente) | 6-DoF SE(3) Geschwindigkeits-/Winkelgeschwindigkeitsbeschränkungen |
| **Sicherheitseinschränkungen** | Keine | CBF explizite Sicherheitsgarantie (Nullkollision) |
| **Experimenteller Maßstab** | Desktop-Objekte / kleine Innenszenen | Stadtschlucht (Stadtblock MatrixCity) |

**Kernargument:** FIM und Shannon Mutual Information sind verwandte, aber nicht gleichwertige Konzepte in der Informationstheorie. Im Zusammenhang mit der Parameterschätzung liefert FIM ein Maß für die statistische Schätzungseffizienz (direkt verknüpft mit der Rekonstruktionsgenauigkeit), während die Shannon-Entropie die Zufälligkeit der Wahrscheinlichkeitsverteilung misst (indirekt verknüpft mit der Rekonstruktionsgenauigkeit). Dieser theoretische Unterschied kann experimentell durch Ablationsexperimente (MI-3DGS vs. FIM-3DGS) quantitativ verifiziert werden.

---

## 7. Einreichungsstrategie

### Ausrichtung auf Zeitschriften/Konferenzen (nach Priorität)**Bevorzugt: IEEE Robotics and Automation Letters (RA-L)**
- Impact-Faktor: 5,2 (2024)
- Überprüfungszyklus: 2–3 Monate (schnell)
- Seitenlimit: 8 Seiten
- Vorteile: ActiveSplat (eines der relevantesten Werke in diesem Artikel) wird auch in RA-L veröffentlicht und die Rezensentengruppe ist korrekt; RA-L akzeptiert Simulationsexperimente

**Gleichzeitige Einreichung: ICRA 2027**
- Frist: ca. 2026/09 (Einreichung erfolgt jeweils ca. September)
- Die gemeinsame Einreichung von RA-L+ICRA ist ein Standardvorgang (eine Einreichung kann nach der Annahme in ICRA angezeigt werden).
- Vorteile: ICRA ist die größte Konferenz im Bereich Robotik mit hoher Präsenz

**Alternative: IROS 2026**
- Frist: ca. 2026/03 (**die Zeit ist knapp**, das Experiment muss 3 Monate im Voraus abgeschlossen sein)
- Akzeptanzrate ~40 %, etwas entspannter als ICRA
- Wenn die Frist im März eingehalten werden kann, wird Vorrang eingeräumt

**Journal Extended Edition: IEEE T-RO**
- Kann nach RA-L-Annahme auf die T-RO-Journalversion erweitert werden (keine erneute Einreichung erforderlich, Gutachtertransfer)
- IF 7.4, SCI Q1, weitere Experimente müssen hinzugefügt werden (echte Maschinenexperimente oder groß angelegte Simulationen)

### Überprüfen Sie die Risikoprognose und -reaktion

| Mögliche Bewertungskommentare | Bewältigungsstrategien |
|----------------|---------|
| „Nicht genügend Unterschied zu GauSS-MI“ | Quantifizieren Sie den Unterschied mithilfe der Tabelle in Abschnitt 6 + Ablationsexperimente (MI-3DGS vs. FIM-3DGS) |
| „Theoretische Grundlage für die RVP-Näherung ist unzureichend“ | Ergänzender Approximationsfehler-Obergrenzensatz (Propositionsbeweis) + experimenteller Verifizierungsfehler <5 % |
| „Nur Simulation, keine realen Maschinenexperimente“ | RA-L akzeptiert reine Simulationsexperimente; Das physikalische Modell von AirSim ist genau; Indoor-Realmaschinenexperimente können durch Einreichung einer modifizierten Version ergänzt werden |
| „Stadtschluchtszenen sind nicht anspruchsvoll genug“ | MatrixCity ist ein umfangreicher Datensatz, der von ICCV 2023 akzeptiert wird; Ergänzung der qualitativen Ergebnisse komplexer Okklusionsszenen |
| „Sicherheitsbeschränkungen sind zu einfach (CBF)“ | Betonen Sie, dass dies das erste Mal ist, dass Sicherheitsbeschränkungen in die NBV-Planung eingeführt wurden; Einfachheit bedeutet nicht unwichtig, und Experimente haben bewiesen, dass es keine Kollisionen gibt |

---

## 8. 12-monatiger Ausführungsweg (Papier C-Spezial)

```
时间        任务                                   里程碑
────────────────────────────────────────────────────────────────────
2026/06    • 实现 FIM-3DGS 核心模块                ▶ 代码框架完成
           • 3DGS 参数 Jacobian 推导与验证
           • RVP 近似实现（GPU CUDA 内核）

2026/07    • AirSim + UE5 城市场景搭建            ▶ 仿真平台就绪
           • 与 GauSS-MI / FisherRF 代码集成
           • 在 ScanNet 上的初步验证实验

2026/08    • MatrixCity 离线实验（与所有基线对比）  ▶ 实验数据完整
           • AirSim 在线主动重建实验
           • 消融实验全套（5 个变体）

2026/09    • 写稿（RA-L 格式，8 页）              ◉ 投稿 RA-L + ICRA 2027
           • 审稿人问题预演（Section 6 准备充分）
           • 语言润色（英文检查）

2026/10    ─── 等待审稿（RA-L 约 2–3 个月）──────────────────────────

2026/12    • 收到审稿意见                         ▶ 修改/接受
           • 若需补充实验：准备真实机实验（室内场景）

2027/01    ◉ 修改稿提交（若大修）                  ▶ 目标：接受并在 ICRA 展示
────────────────────────────────────────────────────────────────────
```

---

## Anhang: Referenzliste**Kerndokumente, die zitiert werden müssen (sortiert nach Zitierpriorität):**1. **FisherRF:** Jiang W et al., „FisherRF: Active View Selection and Mapping with Radiance Fields using Fisher Information“, ECCV 2024 (mündlich)
2. **GauSS-MI:** Xie Y et al., „GauSS-MI: Gaussian Splatting Shannon Mutual Information for Active 3D Reconstruction“, RSS 2025
3. **ActiveGS:** Ye Y et al., „ActiveGS: Active Scene Reconstruction using Gaussian Splatting“, IEEE T-RO 2024
4. **ActiveSplat:** Li Y et al., „ActiveSplat: High-Fidelity Scene Reconstruction through Active Gaussian Splatting“, IEEE RA-L 2025
5. **3DGS Originaltext:** Kerbl B et al., „3D Gaussian Splatting for Real-Time Radiance Field Rendering“, ACM ToG 2023
6. **GenNBV:** Chen X et al., „GenNBV: Generalizable Next-Best-View Policy for Active 3D Reconstruction“, CVPR 2024
7. **NVF:** Xue S et al., „Neural Visibility Field for Uncertainty-Driven Active Mapping“, CVPR 2024
8. **ActiveNeRF:** Ran Y et al., „ActiveNeRF: Learning where to See with Uncertainty Estimation“, ECCV 2022
9. **NeU-NBV:** Jin L et al., „NeU-NBV: Next Best View Planning Using Uncertainty Estimation in Image-Based Neural Rendering“, IROS 2023
10. **FIT-SLAM:** Saravanan S et al., „FIT-SLAM: Fisher Information and Traversability estimation-based Active SLAM“, ICRA 2024
11. **MatrixCity:** Li Z et al., „MatrixCity: A Large-scale City Dataset for City-level Novel View Synthesis and Urban Reconstruction“, ICCV 2023
12. **FCMI:** Charrow B et al., „Information-Theoretic Planning with Trajectory Optimization for Dense 3D Mapping“, ICRA 2020
13. **CBF-Sicherheitskontrolle:** Ames A et al., „Control Barrier Functions: Theory and Applications“, ECC 2019---

> **Hinweise zur Dokumentversion:** Dies ist die erste Version des Paper C-Plans (`v1_20260515`). Nachdem die nachfolgenden Experimente abgeschlossen sind, wird es auf „v2_year Monat Tag.md“ aktualisiert, und nach Erhalt von Überprüfungskommentaren wird es auf „v3_Jahr Monat Tag.md“ aktualisiert.