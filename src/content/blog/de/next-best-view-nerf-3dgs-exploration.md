---
title: "Next-Best-View-Planung trifft auf NeRF/3DGS: Die Informationsgrenze der aktiven Sensorik"
description: "Ausführliche Erläuterung der hochmodernen NBV + NeRF/3DGS-Methoden: Aktive Gaußsche Kartierung von ActiveGAMER, SO-NeRF-Proxy-Target, autonome Datenerfassung von AutoNeRF, die die Schnittgrenze von aktiven Erfassungs- und neuronalen Strahlungsfeldern abdeckt"
tags: ["UAV", "NRF", "3DGS", "Nächstbeste Ansicht", "aktive Wahrnehmung", "Gaußsches Splatting"]
category: "Tech"
pubDate: 2026-04-27
---

# Next-Best-View-Planung trifft auf NeRF/3DGS: die Informationsgrenze der aktiven Sensorik

> **UAV-Wahrnehmungsplanungsreihe·Teil X+1**
> Fokus: NBV + NeRF/3DGS Spitzenmethoden, ActiveGAMER, SO-NeRF, aktive Luft-Boden-Erkundung

---

## 1. Kernkonzept: Warum ist NeRF/3DGS ein perfekter Partner für NBV?

Die traditionelle NBV-Planung hat eine fatale Schwäche: **Sie weiß nicht, „wie das Unsichtbare aussieht“**.

Sie schließen aus aktuellen Beobachtungen, wo die meisten Informationen vorliegen. Bei Orten, die noch nicht beobachtet wurden, können Sie sich jedoch nur auf Heuristiken verlassen („Wählen Sie einen Ort, an dem Sie noch nie waren“).

**NeRF/3DGS ändert dies:**

```
传统方法：
  "我前方10米有个物体，但背面我完全看不到"
  → 只能假设背面 = 未知，启发式选个点去看看

NeRF/3DGS：
  "我有个神经辐射场，已经隐式编码了前+背面的大致形状"
  → 可以渲染背面的大致外观，评估信息增益的真实上限
```

Aus diesem Grund eignet sich **NeRF/3DGS perfekt als „generatives Modell“** für die aktive Erfassung – es kann sich „vorstellen“, wie eine unbeobachtete Region aus jedem Blickwinkel aussehen würde, und zur Berechnung des tatsächlichen Informationsgewinns verwendet werden.

---

## 2. ActiveGAMER: Aktive Gaußsche Kartenrekonstruktion (arXiv, 2025)

**Papier:** *ActiveGAMER: Aktives Gaußsches Mapping durch effizientes Rendering*
**Autor:** Liyan Chen, Huangying Zhan, Kevin Chen, Xiangyu Xu, Qingan Yan, Changjiang Cai, Yi Xu
**Quelle:** arXiv:2501.06897, Januar 2025 | **CVPR 2025**

**Kernbeitrag:**
- Das erste vollständige System aus **Aktiver Wahrnehmung + 3D-Gauß-Splatting**
- Validiert in Simulation und realer Umgebung (Franka-Roboterarm + UAV-Plattform)
- Implementierte **Echtzeit-NBV-Planung** (GPU-Parallel-Rendering-Beschleunigung)

**Systemarchitektur:**

```
┌──────────────────────────────────────────────────────────┐
│                  ActiveGAMER Pipeline                   │
│                                                          │
│  Step 1: 初始建图（稀疏视角覆盖）                         │
│  → 3DGS 初始重建（有明显空洞）                           │
│                                                          │
│  Step 2: NBV 选择（主动感知循环）                        │
│  ┌────────────────────────────────────────────────────┐ │
│  │ 候选视角渲染（并行 ray casting through Gaussians）  │ │
│  │ → 渲染深度图 + 渲染 RGB + 渲染不确定性图             │ │
│  │ → 信息增益评估（基于深度不确定度）                   │ │
│  │ → 选择信息增益最大的下一视角                         │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
│  Step 3: 移动 + 精细建图                                  │
│  → UAV 飞行到新视角                                      │
│  → 增量插入新 Gaussians                                  │
│  → 自适应致密化（只加有信息的区域）                       │
│                                                          │
│  Loop: 返回 Step 2，直到覆盖率达到阈值                    │
└──────────────────────────────────────────────────────────┘
```

**Schlüsseltechnologie:**

### 2.1 Informationsgewinn aufgrund von Unsicherheit

**Wichtige Erkenntnis:** Die Gaußschen Parameter von 3DGS weisen von Natur aus **Mittelwerte und Kovarianzen** (Gaußsche Verteilung) auf, und der Informationsgewinn von Beobachtungen kann direkt aus der Parameterverteilung berechnet werden.**Berechnung des Informationsgewinns:**
$$
\Delta I \ approx \sum_{p \in \text{Pixel}} \sigma^2_{\text{gerendert}}(p)
$$

Das heißt: die Summe der Varianzen der gerenderten Pixel = die Menge an Informationen, die die Perspektive liefern kann.

- Große Darstellungsvarianz → Die Karte dieses Gebiets ist noch grob und es sind weitere Beobachtungen erforderlich
- Geringe Darstellungsvarianz → Die Karte dieses Gebiets ist bereits sehr gut, der Beobachtungsvorteil ist jedoch gering

### 2.2 Effiziente Bewertung der Kandidatenperspektive

Die Anzahl der möglichen Standpunkte ist bei herkömmlichen Methoden gering (Dutzende), da jeder einzelne vollständig wiedergegeben werden muss.

**ActiveGAMER-Beschleunigung:**
1. Verwenden Sie **Splat-basiertes Raycasting** (ohne alle Details zu verfolgen)
2. Batch- und Parallelbewertung von Hunderten von Kandidatenperspektiven
3. Führen Sie ein vollständiges Rendering nur für Top-K-Kandidaten durch
4. Der gesamte NBV-Zyklus beträgt etwa **10 Hz** (kann in Echtzeit erfolgen!)

### 2.3 Adaptive Verdichtung

Nicht alle neuen Perspektiven sind es wert, Gaußsche Gleichungen hinzuzufügen:
- **Hoher Informationsbereich**: Tiefendiskontinuität, große Blickwinkeländerungen → Verdichtung
- **Bereich mit geringen Informationen**: überlappender Bereich, spärliche Textur → überspringen

**Dies kommt auch der Ausrichtung Ihres bestehenden Blogs am nächsten! ** Ihr UAV-NERF-GS-Planning kann diesen Artikel direkt zitieren.

---

## 3. SO-NeRF: NeRF NBV für Proxy-Ziele (arXiv, 2023)

**Papier:** *SO-NeRF: Active View Planning für NeRF unter Verwendung von Ersatzzielen*
**Autor:** Keifer Lee, Shubham Gupta, Sunglyoung Kim, Bhargav Makwana, Chao Chen, Chen Feng
**Quelle:** arXiv:2312.XXXXX, Dezember 2023

**Kernbeitrag:**
- Vorgeschlagene **Ersatzziele** zur Lösung von Nichtkonvexität bei der NBV-Optimierung
- Vermeidet das Problem der direkten Optimierung der Rekonstruktionsqualität (nicht differenzierbar, umfangreiche Berechnung)

**Hinweis:** SO-NeRF wurde auf arXiv veröffentlicht und es wurde kein eindeutiger Top-Veröffentlichungsdatensatz gefunden.

**Methode:**

```
传统 NBV：
  目标：max 重建质量（需要完整重建才能评估）
  局限：不可微、慢、需要多次渲染

SO-NeRF：
  目标：max 代理目标（可微、快速）
  代理：渲染深度的不连续性 + 视角覆盖度
  核心：深度梯度 = 物体边界 = 需要更多信息的地方
```**Intuition:** Die Orte mit großen Gradienten in der gerenderten Tiefenkarte (Tiefenmutationen = Objektgrenzen) sind Orte, die noch nicht modelliert wurden.

**Unterschiede zu ActiveGAMER:**
- SO-NeRF verwendet Tiefengradienten als Proxys (NeRF selbst muss nicht geändert werden)
- ActiveGAMER mit Gaußscher Varianz (erfordert GS-Wahrscheinlichkeitsrahmen)
- Die beiden können sich ergänzen: SO-NeRF führt das Kandidaten-Screening durch und ActiveGAMER übernimmt die Feinabstimmung

---

## 4. AutoNeRF: Autonome Datenerfassung (arXiv, 2024)

**Papier:** *AutoNeRF: Training impliziter Szenendarstellungen mit autonomen Agenten*
**Autor:** Pierre Marza, Laetitia Matignon, Olivier Simonin, Dhruv Batra, Christian Wolf, Devendra Singh Chaplot
**Quelle:** arXiv, 2024

**Kernbeitrag:**
- Lassen Sie den **Agenten (Roboter) unabhängig entscheiden, wo NeRF-Trainingsdaten erfasst werden**
- Verifiziert in der Habitat-Sim-Simulationsumgebung
- Vergleich mehrerer aktiver Strategien: zufällig / grenzbasiert / modellbasiert

**Wichtige Erkenntnisse:**
- Eine einfache grenzbasierte Strategie ist bereits viel besser als eine Zufallsstrategie
- Der Modellvorhersagetyp (Vorhersage der Qualität neuer Perspektiven mithilfe von NeRF) kann weiter verbessert werden
- **Aktive Sammlung vs. passive Sammlung**: Die endgültige Rekonstruktionsqualität wird um mehr als 40 % verbessert

**Inspiration zu UAV:**
- Die Luftperspektive des UAV macht die Grenze (erforschte-unerforschte Grenze) größer als die von Bodenrobotern
- Luft-NBV muss die **vertikale Richtung** berücksichtigen (nicht nur die horizontale Bewegung)
- Oben auf dem Gebäude und unter der überhängenden Struktur befindet sich die einzigartige „Grenze“ des UAV.

---

## 5. Aktive Wahrnehmung mit NeRF (arXiv, 2023)**Artikel:** *Aktive Wahrnehmung mithilfe neuronaler Strahlungsfelder*
**Autor:** Siming He, Christopher D. Hsu, Dexter Ong, Yifei Simon Shao, Pratik Chaudhari
**Quelle:** arXiv:2310.09892, Oktober 2023

**Dies ist ein Grundlagenpapier zur Informationstheorie, das Sie direkt in Ihrem Blog zitieren können! **

**Kernbeitrag:**
Leiten Sie aus **ersten Prinzipien** ab, was die aktive Wahrnehmung maximieren sollte:

> **Maximieren Sie die gegenseitige Information vergangener Beobachtungen für zukünftige Beobachtungen**
> $$\max_a \quad I(Z_{past} \cup Z_{new}(a); Y)$$

Unter ihnen:
- $Z_{past}$ = vorhandene Sensorbeobachtungen
- $Z_{new}(a)$ = neue Beobachtung, die nach Ausführung der Aktion $a$ erhalten wird
- $Y$ = vollständiger Zustand der Umgebung

**Drei Schlüsselkomponenten:**

```
1. Scene Representation（场景表示）
   → NeRF 捕获几何 + 外观 + 语义
   → 可以从任意视角渲染合成图像

2. Generative Model（生成模型）
   → NeRF 就是生成模型！给定 pose → 渲染 image
   → 给合成观测评估信息增益

3. Information-Driven Planner（信息驱动规划器）
   → 采样可行的机器人轨迹
   → 在每条轨迹的末端视角渲染
   → 选择渲染图像信息增益最大的轨迹
```

---

## 6. Vom Objekt zur Szene: Skalierung des NBV

### 6.1 Einzelobjekt-NBV → NBV auf Szenenebene

Frühe NBV-Arbeiten konzentrierten sich auf die vollständige Rekonstruktion einzelner Objekte:
- Das Objekt wird auf den Drehteller gelegt und in einen bestimmten Winkel gedreht, um Bilder aufzunehmen
- Ziel: Alle Perspektiven abdecken und ein vollständiges 3D-Modell erhalten

**Ihre UAV-Arbeit erfolgt auf Szenenebene:**
- Gesamte Stadtschlucht/Innenraum
- Man kann es nicht einzeln machen, man braucht einen Gesamtplan
- **Grenzbasierte Erkundung** wird zur Hauptstrategie

### 6.2 Grenzbasierte Erkundung + Informationsgewinn

**Grenze** = Die Grenze zwischen erforschten und unerforschten Gebieten.

```
经典 Frontier 探索：
  1. 从当前地图提取所有 frontier 点
  2. 选择最近的 frontier → 飞过去
  3. 扩大已知区域
  4. 重复

Frontier + Information Gain：
  1. 从当前地图提取所有 frontier 点
  2. 预测每个 frontier 的信息增益（用 NeRF/3DGS 渲染）
  3. 选择 info/max(distance) 最大的 frontier（权衡信息 + 能量）
  4. 飞过去
  5. 重复
```

**Kompromiss funktionales Design:**

$$
\text{score}(f) = \frac{\text{InformationGain}(f)}{\text{TravelCost}(f)} = \frac{I(f)}{\|p_{current} - f\|_2}
$$

Dies ist eigentlich das Kriterium „maximales Informations-/Entfernungsverhältnis“** bei der UAV-Erkundung, um die Flugeffizienz sicherzustellen.

---

## 7. Spezifische Anwendungen in UAV-Szenarien### 7.1 Urban Canyon-Erkundung

**Szenenmerkmale:**
- Auf beiden Seiten stehen Hochhäuser und oben ist der Himmel offen
- Unten ist die Straße, das GNSS-Signal ist schlecht
- Die Seite ist die Gebäudefassade mit hoher Informationsdichte

**NBV-Strategieberatung:**

```
Phase 1: 建立初始地图
  → 沿建筑边缘飞行，捕获立面纹理
  → 初始重建完成约 30-40%

Phase 2: 填充立面细节
  → 选择立面渲染不确定度大的区域
  → 飞到近处做精细扫描

Phase 3: 顶部覆盖
  → 飞行到建筑顶面高度
  → 俯视捕获屋顶结构

Phase 4: 精细化
  → 重复，直到渲染不确定度全面低于阈值
```

### 7.2 Korrespondenz zu Ihrer bestehenden Stelle

| Was Sie in Ihrem Blog geschrieben haben | Entsprechend NBV-Systemkomponenten |
|------------------|-----------------|
| 3D-Raummodellierung (Octree/Occupancy Grid) | Barrierefreiheitsbeschränkungen + Kollisionserkennung |
| NeRF/3DGS-Mapping | Aktiv bewusste Szenendarstellung |
| Semantischer SLAM | Semantikbewusstes NBV (Scannen „wichtiger“ Objekte priorisieren) |
| Simulationsdaten im geschlossenen Regelkreis | Aktive Sensordatenverbesserung |

---

## 8. Wichtige technische Details

### 8.1 Zusammenfassung der Methoden zur Unsicherheitsschätzung

| Methode | Berechnungsmethode | Anwendbare Szenarien | Echtzeit |
|------|---------|---------|--------|
| **Monte-Carlo-Aussteiger** | Mehrfache Vorwärtsausbreitung, Varianz als Unsicherheit | NeRF (erfordert Netzwerkmodifikation) | Langsam |
| **Ersatzgradient** | Tiefengradient als Proxy rendern | SO-NeRF | Schnell |
| **Gaußsche Varianz** | GS's eigene Kovarianzausbreitung | 3DGS (ActiveGAMER) | Mittel |
| **Aleatorisch + Epistemisch** | Getrennte Rauschunsicherheit und Wissensunsicherheit | Allgemein | Mittel |

### 8.2 Generierung von Kandidatentrajektorien

Beim NBV geht es nicht nur um die Auswahl eines Punktes, sondern um die Auswahl einer **machbaren Flugbahn**:
- Für das UAV gelten Höchstgeschwindigkeits-/Beschleunigungsbeschränkungen
- Kinetische Machbarkeit muss berücksichtigt werden (RRT*/BIT*/MPC)
- Generieren Sie normalerweise zuerst Kandidatenendpunkte und überprüfen Sie dann die Machbarkeit der Flugbahn

---

## 9. Herausforderungen und offene Fragen

### 9.1 Rechenengpass

Die wichtigsten Berechnungskosten des NBV:
- **Kandidatenbewertung** (Hunderte Kandidaten × Darstellung = Engpass)
- **Berechnung des Informationsgewinns** (erfordert mehrere Renderings)
- **NBV-Optimierungsschleife** (erfordert normalerweise 10–50 Iterationen)**Lösung:**
- Schnelles Screening mit Rendering in niedriger Auflösung zu Beginn
- Hochauflösende, genaue Bewertung nur der Top-10-Kandidaten
- GPU-Parallelisierung (Kandidat für paralleles Rendering)

### 9.2 Dynamische Umgebung

Bestehende NBV-Methoden gehen meist von einer statischen Umgebung aus. Aber in der Häuserschlucht:
- Das Auto fährt
- Fußgänger kommen und gehen
- Das Gebäude befindet sich möglicherweise im Bau

**OFFENE FRAGEN:**
- Wie werden dynamische Objekte in die Berechnung des Informationsgewinns einbezogen?
- Was soll ich tun, wenn der modellierte Bereich durch dynamische Objekte blockiert ist?
- Gibt es einen Kompromiss zwischen inkrementellen Online-Updates und regelmäßigen vollständigen Neuerstellungen?

### 9.3 Semantikbewusstes NBV

Die meisten aktuellen NBV-Methoden berücksichtigen nur den Gewinn geometrischer Informationen. Aber:
- „Dieses Gebäude ist ein Museum, wichtiger als ein Parkplatz“
- „An dieser Fassade sind Werbetafeln angebracht, die eine höhere Informationsdichte haben als die leere Wand.“

**Lösung:**
- **Semantisches NeRF** zu NeRF/3DGS hinzufügen
- Informationsgewinn = geometrischer Gewinn × semantisches Gewicht
- Ähnlich dem, was Sie in uav-semantic-mapping.md geschrieben haben!

---

## 10. Empfohlener Forschungsweg

**Route A (schnelle Ergebnisse):**
1. Basierend auf Ihrem UAV-NERF-GS-Planning-Artikel
2. Stellen Sie eine Verbindung zum ActiveGAMER-Modul zur Berechnung des Informationsgewinns her
3. Validieren Sie auf Ihrer vorhandenen UAV-Simulationsplattform
4. Geschätzter Arbeitsaufwand: 2-3 Monate

**Route B (Systematische Studie):**
1. Implementieren Sie FIT-SLAM (FIM-basiertes aktives SLAM).
2. Ersetzen Sie die Kartendarstellung durch Ihr 3DGS-System
3. Semantikbewusste Gewichte hinzufügen
4. Überprüfung an einem echten UAV
5. Geschätzter Arbeitsaufwand: 6-12 Monate

**Route C (Grenzexploration):**
1. Kombinieren Sie VLM (Richtung 1), um „Semantisches NBV“ durchzuführen.
2. VLM bewertet die semantische Bedeutung jeder Grenze
3. Informationsgewinn = geometrischer Gewinn + semantischer Gewinn
4. Geschätzter Arbeitsaufwand: 12+ Monate, aber es gibt viel Raum für Innovation

---

## 📚 Referenzen1. Chen et al. *ActiveGAMER: Aktives Gaußsches Mapping durch effizientes Rendering*. arXiv:2501.06897, Januar 2025.
2. Lee et al. *SO-NeRF: Active View Planning für NeRF unter Verwendung von Ersatzzielen*. arXiv:2312.XXXXX, Dezember 2023.
3. He et al. *Aktive Wahrnehmung mithilfe neuronaler Strahlungsfelder*. arXiv:2310.09892, Oktober 2023.
4. Marza et al. *AutoNeRF: Training impliziter Szenendarstellungen mit autonomen Agenten*. arXiv, 2024.
5. Saravanan et al. *FIT-SLAM: Auf Fisher Information and Traversability-Schätzung basierendes aktives SLAM*. arXiv:2401.09322, Januar 2024.
6. Zhan et al. *Aktive menschliche Posenschätzung über einen autonomen UAV-Agenten*. arXiv, 2024.
7. Chaplot et al. *Visuelle Erkundung für die Navigation über große Entfernungen erlernen*. NeurIPS, 2020.