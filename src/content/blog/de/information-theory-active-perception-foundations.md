---
title: "Aktive Wahrnehmung aus informationstheoretischer Sicht: Fisher Information und Cramér-Rao-Untergrenzen"
description: "Erklären Sie die informationstheoretischen Grundlagen der aktiven Sensorik anhand erster Prinzipien: Fisher-Information, Cramér-Rao-Untergrenze, gegenseitige Information und ihre Anwendung in SLAM-Arbeiten wie FIT-SLAM und Continuous Info Modeling."
tags: ["UAV", "aktive Wahrnehmung", "Informationstheorie", "Fisher-Informationen", "ZUSCHLAGEN", "Cramér-Rao"]
category: "Tech"
pubDate: 2026-04-27
sourceHash: "ea628455d699760ba54122071b05535aa55cf481"
---

# Aktive Wahrnehmung aus informationstheoretischer Sicht: Fisher Information und Cramér-Rao-Untergrenze

> **UAV-Wahrnehmungsplanungsreihe · Teil X**
> Schwerpunkt: Grundlagen der Informationstheorie, Active Sensing Framework, Berechnung von Fisher-Informationen und Anwendung in SLAM

---

## 1. Was ist aktive Wahrnehmung?

Die traditionelle Wahrnehmung ist **passiv**: Der Roboter empfängt Sensordaten und aktualisiert ein Modell der Umgebung.

**Aktive Wahrnehmung** geht noch einen Schritt weiter: Der Roboter wählt aktiv, „wo er suchen soll“**, um den Wert der Aufgabe zu maximieren.

```
被动感知：
传感器 → 数据 → 地图更新（机器人不动）

主动感知：
当前地图 → 信息价值评估 → 最优下一视角选择 → 移动 → 传感器 → 地图更新
                ↑
           核心问题：如何量化"信息价值"？
```

Für UAVs ist die aktive Erfassung besonders wichtig:
- **Energiebeschränkung**: Fliegen verbraucht Energie und kann nicht zufällig fliegen.
- **Weites Sichtfeld**: Bei der Bewegung in der Luft ändert sich das Sichtfeld drastisch und es ist entscheidend, den optimalen Weg zu wählen.
- **Dreidimensionaler Raum**: Gebäude, Berge und Bäume müssen für eine vollständige Modellierung aus mehreren Blickwinkeln betrachtet werden.

---

## 2. Mathematische Grundlagen der Informationstheorie

### 2.1 Fisher-Informationen

Bei einem Wahrscheinlichkeitsmodell $p(x|\theta)$, bei dem $\theta$ der zu schätzende Parameter ist, misst **Fisher Information** die Informationsmenge über $\theta$, die in den Beobachtungsdaten $X$ enthalten ist:

$$
I(\theta) = \mathbb{E}_X \left[ \left( \frac{\partial}{\partial \theta} \log p(X|\theta) \right)^2 \right] = - \mathbb{E}_X \left[ \frac{\partial^2}{\partial \theta^2} \log p(X|\theta) \right]
$$

**Intuitives Verständnis:**
- Wenn sich $\log p(x|\theta)$ in der Nähe von $\theta$ **sehr steil** ändert, bedeutet das, dass die Daten sehr empfindlich auf $\theta$ reagieren → Fisher-Informationen **groß**
- Wenn sich $\log p(x|\theta)$ um $\theta$ herum **flach** ändert, sind die Daten nicht empfindlich gegenüber $\theta$ → Fisher Information **klein**

**Skalare vs. Matrixform:**- Skalar: $I(\theta)$ (eindimensionaler Parameter)
- Matrix: **Fisher Information Matrix (FIM)** $I(\boldsymbol{\theta})$ (mehrdimensionale Parameter)

$$
[I(\boldsymbol{\theta})]_{ij} = \mathbb{E} \left[ \frac{\partial \log p(X|\boldsymbol{\theta})}{\partial \theta_i} \cdot \frac{\partial \log p(X|\boldsymbol{\theta})}{\partial \theta_j} \right]
$$

FIM ist der Riemannsche metrische Tensor im Parameterraum, der bestimmt, wie genau Sie Parameter schätzen können.

---

### 2.2 Cramér-Rao-Untergrenze (CRLB)

Die Cramér-Rao-Untergrenze ist eine Kernanwendung von Fisher Information: **gibt eine optimale Untergrenze für die Varianz eines erwartungstreuen Schätzers an**.

$$
\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}
$$

**Physikalische Bedeutung:** Unabhängig davon, welche Schätzmethode Sie verwenden (solange sie erwartungstreu ist), darf die Schätzgenauigkeit $1/I(\theta)$ nicht überschreiten.

**Bedeutung in SLAM:**
- Die untere Grenze der Kovarianz der Roboterpose $\mathbf{x}$ wird durch FIM bestimmt
- $[\text{Cov}(\mathbf{x})]^{-1} \preceq I(\mathbf{x})$
- Je kleiner der Kehrwert von FIM → desto genauer ist die Schätzung

---

### 2.3 Gegenseitige Information

Gegenseitige Information misst die statistische Abhängigkeit zwischen zwei Zufallsvariablen:

$$
I(X; Y) = \int \int p(x,y) \log \frac{p(x,y)}{p(x)p(y)} \, dx \, dy = H(X) - H(X|Y)
$$

**Bedeutung in aktiver Wahrnehmung:**
- $X$ = zukünftige Sensorbeobachtungen
- $Y$ = der unsichere Zustand der aktuellen Karte

Maximierung von $I(X; Y)$ = Auswahl der Perspektive, in der zukünftige Beobachtungen die Unsicherheit in der aktuellen Karte am besten reduzieren.Dies ist die informationstheoretische Definition von „**Informationsgewinn**“ in der aktiven Wahrnehmung.

---

## 3. Aktives Sensor-Framework

### 3.1 Kernthema: Next-Best-View (NBV)

Das Kernproblem der aktiven Erfassung ist die **NBV-Planung**: Wo sollten wir uns angesichts des aktuell beobachteten Gebiets als nächstes bewegen, um die Unsicherheit am effektivsten zu reduzieren?

**Mathematische Form des NBV-Problems:**

$$
a^* = \arg\max_{a \in \mathcal{A}} \quad \mathbb{E}_{z \sim p(z|x, a)} \left[ \log \det I(\theta_{new}(x, z)) \right] - \log \det I(\theta_{old}(x))
$$

Das heißt: Wählen Sie die Aktion $a$ so, dass die Determinante des FIM (ein skalares Maß für die Gesamtunsicherheit) nach der Ausführung maximiert wird.

---

### 3.2 Drei Hauptkomponenten des aktiven Sensorsystems

**Informationstheoretischer Rahmen für aktive Wahrnehmung** schlägt drei Komponenten eines aktiven Wahrnehmungssystems vor:

```
┌─────────────────────────────────────────────────────────┐
│                   Active Perception System              │
│                                                         │
│  Component 1: 状态估计 & 地图表示                        │
│  (State Estimation & Map Representation)               │
│  → 当前已观测区域的完整表示（几何 + 语义）               │
│                                                         │
│  Component 2: 未来观测合成                               │
│  (Generative Model of Future Observations)              │
│  → 给定候选动作，生成未来会看到的图像/传感器数据         │
│                                                         │
│  Component 3: 信息驱动的规划                              │
│  (Information-Driven Planning)                          │
│  → 在候选轨迹上计算互信息，选择最优                     │
└─────────────────────────────────────────────────────────┘
```

**Warum benötigen Sie Komponente 2 (generiertes Modell)? **
- Man kann nicht wirklich rausfliegen und jeden Ort ausprobieren (zu teuer)
- Sie benötigen ein Modell, um sich vorzustellen, was Sie sehen würden, wenn Sie zu jedem Kandidatenstandort fliegen würden
- **NeRF/3DGS sind perfekte generative Modelle** (in Ihrem Blog wurde bereits darüber geschrieben!)

---

## 4. Anwendung von Fisher-Informationen in SLAM

### 4.1 FIM im SLAM

Beim visuellen SLAM muss der Roboter gleichzeitig Folgendes schätzen:
- **Pose** $\mathbf{x}_k$ (wo ist die Kamera)
- **Kartenpunkt** $\mathbf{m}_i$ (wo ist der 3D-Punkt im Raum)

Beobachtungsmodell: $z_{k,i} = h(\mathbf{x}_k, \mathbf{m}_i) + \mathbf{n}$

- $h(\cdot)$ ist die Projektionsfunktion (3D → 2D-Bildkoordinaten)
- $\mathbf{n} \sim \mathcal{N}(0, \Sigma)$ ist das Messrauschen**Informationen zu beobachteten Fischern:**
$$
I(\mathbf{x}_k, \mathbf{m}_i) = \frac{\partial h^\top}{\partial [\mathbf{x}_k, \mathbf{m}_i]} \Sigma^{-1} \frac{\partial h}{\partial [\mathbf{x}_k, \mathbf{m}_i]}
$$

**Wichtige Erkenntnisse:**
- Bei der Beobachtung desselben 3D-Punkts ergeben **unterschiedliche Perspektiven** unterschiedliche Fisher-Informationen
- Je tiefer die Beobachtungstiefe (je weiter entfernt), desto geringer ist die Informationsmenge
- Je größer die Beobachtungsbasislinie (je größer die Änderung des Blickwinkels), desto größer ist die Informationsmenge

**Deshalb müssen UAVs ihre Perspektive aktiv wählen! **

---

### 4.2 Interpretation klassischer Arbeiten

#### **FIT-SLAM (arXiv, Januar 2024)**

**Artikel:** *FIT-SLAM – Fisher Information and Traversability Estimation-based Active SLAM für die Erkundung in 3D-Umgebungen*
**Autor:** Suchetan Saravanan, Corentin Chauffaut, Caroline Chanel, Damien Vivet
**Quelle:** arXiv:2401.09322, Januar 2024

**Kernbeitrag:**
- **Fisher-Informationen** explizit in die Zielfunktion von **Active SLAM** einführen
- Berücksichtigen Sie auch die **Passierbarkeit** – nicht nur „klar sehen“, sondern auch „fliegen“
- Auf **3D-Umgebung** (nicht planar) ausgerichtet, geeignet für UAV-Erkundung in komplexen Stadtschluchten

**Hinweis:** Dieses Papier wurde auf arXiv veröffentlicht (es wurde bei IEEE ICARA 2024 eingereicht). Auf der Spitzenkonferenz konnte kein eindeutiger Veröffentlichungsnachweis gefunden werden. Bei der Zitierung ist auf die arXiv-Version zu achten.

---#### **Active View Planning für Visual SLAM: Continuous Information Modeling (arXiv, 2022/2023)**

**Papier:** *Aktive Ansichtsplanung für visuelles SLAM in Außenumgebungen basierend auf kontinuierlicher Informationsmodellierung*
**Autor:** Zhihao Wang, Haoyao Chen, Shiwu Zhang, Yunjiang Lou
**Quelle:** arXiv:2211.xxxxx, 2022

**Kernbeitrag:**
- Vorgeschlagene **kontinuierliche Informationsmodellierung** als Ersatz für diskrete Informationsgitter
- Optimieren Sie die nächste Ansicht auf einem kontinuierlichen Raum und nicht auf einem diskreten Satz von Kandidatenpunkten
- Modellieren Sie räumliche Unsicherheit mithilfe des **Gaußschen Prozesses (GP)**

**Wichtige Erkenntnisse:**

Herkömmliche Methoden diskretisieren den Raum in Kandidatenpunkte → Der Informationsgewinn wird nur für diese begrenzte Menge von Punkten bewertet

Kontinuierliche Methode: Verwenden Sie GP, um „die Informationsmenge an jeder Position“ darzustellen, und optimieren Sie dann **direkt im kontinuierlichen Raum**

$$
\mu(a) = \text{GP vorhergesagte Aktion} a \text{Die Informationsmenge bei} \\
\sigma(a) = \text{Vorhersageunsicherheit von GP} \\
\text{Erfassungsfunktion: } a^* = \arg\max_a \, \mu(a) + \beta \sigma(a)
$$

**Vorteile gegenüber UAV:**
- Der Bewegungsraum des UAV ist kontinuierlich und sollte nicht zur Diskretisierung gezwungen werden
- Möglichkeit zur Optimierung kompletter 6-DoF-Trajektorien statt nur einzelner Wegpunktauswahlen

---

## 5. Berechnung des Informationsgewinns für die aktive Erfassung

### 5.1 Informationsgewinn basierend auf Fisher-Informationen

**Informationsgewinn** = FIM-Änderung vor und nach der Aktion:

$$
\Delta I(a) = \det I(\theta_{nach}) - \det I(\theta_{vor})
$$Die eigentliche Berechnung erfordert jedoch keine tatsächliche Rekonstruktion, sondern lediglich:
1. Beobachtungen aus einer neuen Perspektive vorhersagen
2. Berechnen Sie den FIM neu hinzugefügter Beobachtungen
3. Verwenden Sie das **Schur-Komplement**, um das Gesamt-FIM effizient zu aktualisieren

### 5.2 Monte-Carlo-Schätzung der gegenseitigen Information

Gegenseitige Information $I(X; Y)$ kann normalerweise nicht analytisch berechnet werden und erfordert den Einsatz von Monte-Carlo-Methoden:

$$
\hat{I}(X; Y) = \frac{1}{N} \sum_{i=1}^N \log \frac{p(x_i|y_i)}{p(x_i)}
$$

Bei aktiver Wahrnehmung:
- Probieren Sie mehrere mögliche Kartenversionen aus einer unsicheren Verteilung der aktuellen Karte aus
- Berechnen Sie für jede mögliche Aktion die **durchschnittliche gegenseitige Information**
- Wählen Sie die Aktion mit der größten gegenseitigen Information aus

---

## 6. Informationstheorie im Vergleich zu anderen Prinzipien

| Richtlinien | Vorteile | Nachteile |
|------|------|------|
| **Fischerinformationen** | Theoretisch optimale, enge untere Schranke | Komplexe Berechnung, erfordert Wahrscheinlichkeitsmodell |
| **Gegenseitige Information** | Intuitiv und einfach zu messen | Große Schätzvarianz |
| **Entropie** | Intuitiv | Kontinuierliche Verteilungen können nicht verarbeitet werden |
| **Entfernungsbasiert** | Einfach und schnell | Berücksichtigt keine Okklusion/Erscheinung |
| **Abdeckungsbasiert** | Einfach | Berücksichtigt nicht die Informationsdichte |

**Best Practice:** Kombination mehrerer Kriterien
- **SICHERHEIT**: abstandsbasierte Kollisionsprüfung
- **Effizienz**: Entropiebasierte Abdeckung
- **Genauigkeit**: FIM-basierte Posengenauigkeit

---

## 7. Verbindungen zu Ihrer bestehenden Arbeit

Du hast bereits in deinem Blog geschrieben:
- **NeRF/3DGS + UAV**: Umgebungsdarstellung (genau Komponente 1 der aktiven Erfassung!)
- **Semantischer SLAM**: Karten mit Semantik (Semantisches FIM > Geometrisches FIM)
- **Digital Twin**: ein in Echtzeit aktualisiertes Umgebungsmodell

**Das bedeutet:**
Sie verfügen bereits über die **Kartendarstellungsebene** des Active-Sensing-Frameworks, und durch Hinzufügen der **Informationsgewinn-Bewertungsebene** können Sie ein vollständiges Aktiv-Sensing-System aufbauen!

**Natürliche Erweiterung:**
```
你已有的 NeRF/3DGS 地图
    ↓ + FIT-SLAM 的 FIM 计算方法
    ↓ + GP-based continuous NBV 优化
= 你的主动感知 UAV 系统
```

---

## 📚 Referenzen1. Saravanan et al. *FIT-SLAM – Fisher Information and Traversability Estimation-basiertes aktives SLAM für die Erkundung in 3D-Umgebungen*. arXiv:2401.09322, Januar 2024.
2. Wang et al. *Aktive Ansichtsplanung für visuelles SLAM in Außenumgebungen basierend auf kontinuierlicher Informationsmodellierung*. arXiv, 2022.
3. Chen et al. *ActiveGAMER: Aktives Gaußsches Mapping durch effizientes Rendering*. arXiv:2501.06897, Januar 2025.
4. Lee et al. *SO-NeRF: Active View Planning für NeRF unter Verwendung von Ersatzzielen*. arXiv:2312.XXXXX, Dezember 2023.
5. He et al. *Aktive Wahrnehmung mithilfe neuronaler Strahlungsfelder*. arXiv:2310.09892, Oktober 2023.
6. Marza et al. *AutoNeRF: Training impliziter Szenendarstellungen mit autonomen Agenten*. arXiv, 2024.
7. Chaplot et al. *Visuelle Erkundung für die Navigation über große Entfernungen erlernen*. NeurIPS, 2020.