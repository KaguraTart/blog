---
title: "Eine Überprüfung der Konfliktlösungsalgorithmen für die UAV-Pfadplanung"
description: "Eingehende Analyse von Konflikterkennungs- und -lösungsalgorithmen in Multi-UAV-Systemen, die geometrische Methoden, Optimierungsmethoden, Zusammenarbeit mit mehreren Maschinen und Lernmethoden abdeckt, von klassischen Algorithmen bis hin zur hochmodernen Systemüberprüfung"
pubDate: 2026-04-07T11:12:59+08:00
tags: ["Drohne", "Wegplanung", "Konfliktlösung", "Multi-Agent", "Algorithmusübersicht"]
category: Tech
---

# Überblick über den Konfliktlösungsalgorithmus zur UAV-Pfadplanung

> Da sich unbemannte Luftfahrzeuge (UAVs) vom Einzelmaschinenbetrieb zur Cluster-Kollaboration weiterentwickeln, sind Pfadkonflikte zu einem unvermeidlichen Kernproblem geworden, wenn mehrere UAVs Aufgaben im selben Luftraum ausführen. **Konfliktlösung** bezieht sich auf die Anpassung der Flugbahn oder Entscheidungsfindung jeder Drohne, um den Konfliktzustand zu beseitigen und die Mission weiterhin abzuschließen und gleichzeitig die Flugsicherheit zu gewährleisten. In diesem Artikel werden die gängigen Algorithmus-Frameworks zur Konflikterkennung und -lösung systematisch sortiert, von geometrischen Methoden bis hin zu Deep Reinforcement Learning, und die Kernideen, Vor- und Nachteile sowie die Entwicklung jeder Technologie untersucht.

---

## 1. Definition und Klassifizierung von Konflikten

### 1.1 Was ist ein Pfadkonflikt?

In einem Multi-UAV-System bezieht sich **Konflikt** auf einen Zustand, in dem zwei oder mehr UAVs gleichzeitig dieselbe räumliche Position (oder weniger als einen sicheren Isolationsabstand) in der Raum-Zeit-Dimension einnehmen. Formell:

$$
\exists \, i, j, \, i \neq j, \quad \| \mathbf{p}_i(t) - \mathbf{p}_j(t) \| < d_{sicher}
$$

Darunter ist $\mathbf{p}_i(t)$ die Position der $i$-ten Drohne und $d_{safe}$ der sichere Isolationsabstand (normalerweise 5–50 m, abhängig vom Missionsszenario).

### 1.2 Konfliktklassifizierung

| Geben Sie | ein Beschreibung | Typisches Szenario |
|------|------|---------|
| **Weltraumkonflikt** | Flugbahnen schneiden sich im Raum | Sich kreuzende Routen, gegensätzliche Flüge |
| **Zeit- und Raumkonflikt** | Trajektorien überschneiden sich in der Zeitdimension | Betreten Sie nacheinander denselben Luftraum |
| **Geschwindigkeitskonflikt** | Relativgeschwindigkeit überschreitet Sicherheitsschwelle | Aufholszenario |
| **Hoher Konflikt** | Konflikt in vertikaler Richtung | Hebekreuzung |
| **Dynamische Konflikte** | Konflikte durch sich bewegende Hindernisse (andere Luftfahrzeuge) | Begegnungen aus der Luft |

### 1.3 Konfliktmetriken

- **Zeit bis zum Konflikt**: Sagen Sie die verbleibende Zeit bis zum Auftreten eines Konflikts voraus
- **Konfliktwahrscheinlichkeit**: Konfliktrisikobewertung unter Berücksichtigung der Unsicherheit
- **Mindestabstand**: Der kürzeste Abstand zwischen Flugbahnen
- **Lösungszeit**: die Zeit, die erforderlich ist, damit die Lösungsaktion wirksam wird

---

## 2. KonflikterkennungsalgorithmusDie Konflikterkennung ist ein vorläufiger Schritt zur Konfliktlösung und ihr Kern ist die **Konfliktvorhersage** – die Feststellung, ob ein Konflikt auftreten wird, bevor er tatsächlich auftritt.

### 2.1 Geometrische Vorhersagemethode

Die intuitivste Methode ist die räumliche Erkennung auf Basis geometrischer Berechnungen:

```python
import numpy as np

def detect_conflict_2D(traj_i, traj_j, safe_radius=5.0):
    """
    检测两条轨迹是否发生空间冲突
    
    traj_i, traj_j: shape (N, 3) 的轨迹数组，每行为 (x, y, z)
    safe_radius: 安全隔离距离 (m)
    返回: (是否冲突, 最小间隔距离, 冲突时间点索引)
    """
    min_dist = float('inf')
    conflict_time = -1
    
    for t in range(len(traj_i)):
        dist = np.linalg.norm(traj_i[t] - traj_j[t])
        if dist < min_dist:
            min_dist = dist
            if dist < safe_radius:
                conflict_time = t
    
    is_conflict = min_dist < safe_radius
    return is_conflict, min_dist, conflict_time
```

### 2.2 Geschwindigkeitshindernis

**Velocity Obstacle (VO)** ist die klassischste Methode zur Konflikterkennung und -vorhersage im Robotikbereich. Es wurde von Fioretti & Fraichard (1999) in den UAV-Bereich eingeführt.

Kernidee: Konstruieren Sie einen „verbotenen Bereich“ im Geschwindigkeitsraum. Liegt der aktuelle Geschwindigkeitsvektor der Drohne in diesem Bereich, kommt es mit Sicherheit zu einem Konflikt.

$$
VO_{ij} = \{ \mathbf{v} \mid \lambda(\mathbf{p}_j - \mathbf{p}_i, \mathbf{v} - \mathbf{v}_j) \cap D_{ij} \neq \varnothing \}
$$

Dabei ist $D_{ij}$ ein Zylinder mit $\mathbf{p}_j - \mathbf{p}_i$ als Achse und dem Radius als Sicherheitsabstand und $\lambda$ ist der Halbstrahl.

```python
def velocity_obstacle(p_i, v_i, p_j, v_j, r_safe=5.0):
    """
    计算第 i 架 UAV 的速度障碍区域
    p_i, p_j: 位置向量
    v_i, v_j: 速度向量
    r_safe: 安全半径
    """
    rel_pos = p_j - p_i
    rel_vel = v_i - v_j
    dist = np.linalg.norm(rel_pos)
    
    if dist == 0:
        return None
    
    # 相对位置的夹角（安全圆柱的视角）
    theta = np.arcsin(r_safe / dist)
    
    # 障碍扇区的两条边向量
    dir_pos = rel_pos / dist
    perp_dir = np.array([-dir_pos[1], dir_pos[0]])
    
    # 两条边界速度向量
    v_left  = v_j + np.linalg.norm(v_j) * (np.cos(theta) * dir_pos + np.sin(theta) * perp_dir)
    v_right = v_j + np.linalg.norm(v_j) * (np.cos(theta) * dir_pos - np.sin(theta) * perp_dir)
    
    return v_left, v_right  # VO 的两条边界

def is_in_vo(v_i, v_left, v_right, v_j):
    """判断速度 v_i 是否落在 VO 区域内"""
    # 转换到相对坐标系
    rel_v = v_i - v_j
    rel_left  = v_left - v_j
    rel_right = v_right - v_j
    
    # 检查 rel_v 是否在 rel_left 和 rel_right 之间
    cross_left  = np.cross(rel_left,  rel_v)
    cross_right = np.cross(rel_right, rel_v)
    
    return np.sign(cross_left) == np.sign(cross_right)
```

### 2.3 Unsicherheitsbewusste Konflikterkennung

In tatsächlichen Systemen enthalten Standortinformationen häufig Unsicherheiten wie GPS-Fehler und Sensorrauschen. **Probabilistische Konflikterkennung** Führt die Wahrscheinlichkeitsverteilung in die Konfliktbeurteilung ein:

$$
P_{Konflikt} = \int\int \mathbb{1}(\| \mathbf{p}_i - \mathbf{p}_j \| < d_{sicher}) \cdot f_i(\mathbf{p}_i) \cdot f_j(\mathbf{p}_j) \, d\mathbf{p}_i \, d\mathbf{p}_j
$$

wobei $f_i, f_j$ die Wahrscheinlichkeitsdichtefunktion des Ortes ist (normalerweise wird davon ausgegangen, dass es sich um eine Gaußsche Verteilung handelt). Ein Konfliktalarm wird ausgelöst, wenn $P_{conflict} > P_{threshold}$.Zu den gängigen Methoden gehören:
- **Monte-Carlo-Stichprobe**: Statistik der Konfliktverhältnisse nach Stichprobe einer großen Anzahl von Wahrscheinlichkeitsverteilungen
- **Linear Validation Tool (LVT)**: Analytische Approximation von Wahrscheinlichkeitskonflikten unter der Annahme einer Gaußschen Verteilung
- **Stochastic Reachable Set**: Mengendarstellung basierend auf der stochastischen Kontrolltheorie

---

## 3. Konfliktlösungsalgorithmus

### 3.1 Geometrische Methode

#### 3.1.1 Rate Obstacle-Methode (Rate Obstacle / VO-Korrektur)

VO-basierte Eliminierungsstrategie: Finden Sie eine Zielgeschwindigkeit $\mathbf{v}_{new}$, die den VO-Bereich vermeiden kann:

```python
def vo_resolution(p_i, v_i, p_j, v_j, v_max=20.0, r_safe=5.0):
    """
    基于 Velocity Obstacle 的冲突消解
    返回满足速度约束且避开 VO 的新速度
    """
    vo = velocity_obstacle(p_i, v_i, p_j, v_j, r_safe)
    if vo is None:
        return v_i
    
    v_left, v_right = vo
    
    # 所有候选速度（速度空间中均匀采样）
    best_v = v_i
    min_dist_to_vo = float('inf')
    
    for speed in np.linspace(0, v_max, 20):
        for angle in np.linspace(0, 2*np.pi, 36):
            v_candidate = speed * np.array([np.cos(angle), np.sin(angle)])
            
            # 跳过落在 VO 内的速度
            if is_in_vo(v_candidate, v_left, v_right, v_j):
                continue
            
            # 选择最接近原始速度方向且最"远离"VO 的速度
            dist_to_original = np.linalg.norm(v_candidate - v_i)
            # 到 VO 边界的距离
            dist_to_vo = min(
                np.linalg.norm(v_candidate - v_left),
                np.linalg.norm(v_candidate - v_right)
            )
            
            # 优化目标：尽量接近原速度，同时远离 VO
            score = dist_to_vo - 0.5 * dist_to_original
            if dist_to_vo > min_dist_to_vo and dist_to_vo > 1.0:
                min_dist_to_vo = dist_to_vo
                best_v = v_candidate
    
    return best_v
```

#### 3.1.2 Methode des künstlichen Potenzialfeldes (Künstliches Potenzialfeld)

Stellen Sie sich Drohnen als geladene Teilchen vor, die sich in einem „Potentialfeld“ bewegen:
- **Zielpunkt** erzeugt Anziehung
- **Hindernisse/andere Drohnen** erzeugen abstoßende Kraft

$$
\mathbf{F}_{total} = \mathbf{F}_{att} + \sum_j \mathbf{F}_{rep,j}
$$

Unter ihnen:
$$
\mathbf{F}_{att} = k_{att} \cdot (\mathbf{p}_{Ziel} - \mathbf{p}_i)
$$
$$
\mathbf{F}_{rep,j} = k_{rep} \cdot \frac{\mathbf{p}_i - \mathbf{p}_j}{\| \mathbf{p}_i - \mathbf{p}_j \|^3} \cdot (\| \mathbf{p}_i - \mathbf{p}_j \| - d_{safe})
$$

**Vorteile**: Schnelle Berechnung, geeignet für Echtzeitsteuerung
**Nachteile**: Leicht in lokale Minima zu fallen (zwei Drohnen werden oszillieren, wenn sie sich nicht gegenseitig „schubsen“ können)

**Verbesserungsanweisungen**:
- Potenzialfeldformung: Passen Sie die Form des Potenzialfelds an, um lokale Minima zu vermeiden
- Mehrere virtuelle Potenzialfelder: Führen Sie virtuelle Hindernisse ein, um Wege um Fallenbereiche herum zu leiten
- Hybridmethode: kombiniert mit A* oder RRT*, Nutzung des Potenzialfeldes zur lokalen Feinabstimmung

#### 3.1.3 Voronoi-Diagramm-MethodeDas Voronoi-Diagramm wird verwendet, um den Raum in mehrere Bereiche zu unterteilen, und jede Drohne fliegt innerhalb ihrer Voronoi-Einheit, wodurch sichergestellt wird, dass der Abstand zu anderen Drohnen immer größer ist als ihr Abstand zur Voronoi-Grenze:

1. Konstruieren Sie das Voronoi-Diagramm des aktuellen Augenblicks in Echtzeit
2. Jede Drohne wählt einen Wegpunkt in der Nähe des am weitesten entfernten Punkts (optimaler Punkt) ihrer Voronoi-Einheit
3. Bewegen Sie sich in Richtung des Wegpunkts und wechseln Sie zum neuen Voronoi-Pfad, wenn ein Konflikt erkannt wird.

```python
from scipy.spatial import Voronoi
import numpy as np

def voronoi_resolution(positions, v_i, p_i, v_max=20.0):
    """
    基于 Voronoi 图的多机冲突消解
    positions: 所有无人机位置 (N, 2)
    """
    vor = Voronoi(positions)
    
    # 找到当前无人机 i 的 Voronoi 单元
    region_idx = vor.point_region[np.where(vor.point_region == vor.point_region[0])[0][0]]
    # 简化的 Voronoi 路径选择：取 Voronoi 顶点的方向
    vertices = [vor.vertices[v] for v in vor.regions[region_idx] if v >= 0]
    
    if not vertices:
        return v_i
    
    # 选择最接近原始速度方向的安全顶点
    best_dir = v_i / np.linalg.norm(v_i)
    best_vertex = None
    max_projection = -float('inf')
    
    for v in vertices:
        direction = v - p_i
        if np.linalg.norm(direction) < 0.01:
            continue
        direction = direction / np.linalg.norm(direction)
        projection = np.dot(direction, best_dir)
        
        if projection > max_projection:
            max_projection = projection
            best_vertex = v
    
    if best_vertex is None:
        return v_i
    
    return np.clip(best_vertex - p_i, -v_max, v_max)
```

### 3.2 Optimierungsmethode

#### 3.2.1 Gemischte ganzzahlige lineare Programmierung (MILP)

MILP ist ein klassisches Framework, das die Trajektorienplanung mit mehreren UAVs als mathematisches Optimierungsproblem formalisiert und im UAV-Bereich von Schouwenaars et al. entwickelt wurde. (2001).

**Kernidee**: Die kontinuierliche Flugbahn wird durch stückweise Polynome oder feste Wegpunktsequenzen dargestellt, Konfliktbeschränkungen und Sicherheitsbeschränkungen werden durch lineare Ungleichungen ausgedrückt und binäre ganzzahlige Variablen werden eingeführt, um die Schaltlogik der Flugsegmente darzustellen:

$$
\min \quad \sum_i \sum_k \| \mathbf{v}_{i,k} - \mathbf{v}_{i,k-1} \|^2 + \lambda \sum_i \sum_k \| \mathbf{v}_{i,k} - \mathbf{v}_{i,k}^{pref} \|^2
$$

**Einschränkungen**:
- Kinematische Einschränkungen: $\| \mathbf{v}_{i,k} \| \leq v_{max}$, $\| \mathbf{a}_{i,k} \| \leq a_{max}$
- Einschränkungen zur Konfliktvermeidung:
  - Wenn $\| \mathbf{p}_{i,k} - \mathbf{p}_{j,k} \| < d_{safe}$, dann ist die entsprechende Binärvariable $\delta_{ijk} = 1$
  - ODER-Beschränkung einführen: $\sum_j \delta_{ijk} \leq 0$ (zwingt, dass alle $\delta$ 0 sind, d. h. kein Konflikt)
- Einschränkungen bei der Aufgabenerledigung: $\| \mathbf{p}_{i,K} - \mathbf{p}_{Ziel,i} \| < \varepsilon$

```python
# 概念性 MILP 冲突约束（伪代码）
"""
minimize: sum_i sum_k (v_i,k - v_pref_i,k)^2

subject to:
    for each UAV i, segment k:
        p_i,k+1 = p_i,k + v_i,k * dt          # 运动学
        norm(v_i,k) <= v_max                   # 速度限幅
        norm(a_i,k) <= a_max                   # 加速度限幅
        
    for each UAV pair (i,j), segment k:
        norm(p_i,k - p_j,k)^2 >= d_safe^2 OR delta_ik = 0
        M * delta_ik >= norm(p_i,k - p_j,k)^2 - d_safe^2
        sum_j delta_jk <= 0                     # 所有 delta 必须为 0
"""
```**Vorteile**: Globale optimale Lösung, die sicherstellt, dass harte Einschränkungen erfüllt werden
**Nachteile**: Die Rechenkomplexität von MILP-Lösern (CPLEX, Gurobi) steigt exponentiell mit der Anzahl der Drohnen, was es schwierig macht, Szenarien mit mehr als 5–10 Drohnen in Echtzeit zu lösen

#### 3.2.2 Dynamischer Fensteransatz (DWA)

DWA, entlehnt aus der Roboterbewegungsplanung, tastet den Geschwindigkeitsraum $(v, \omega)$ ab und bewertet jede Kandidatengeschwindigkeit:
1. **Flugbahn zum Ziel**
2. **Kollisionssicherheit** (beurteilt durch Simulation kurzfristiger Flugbahnen)
3. **Geschwindigkeitserreichbarkeit**

```python
def dwa_resolution(p_i, v_i, v_goal, obstacles,
                   v_max=3.0, v_min=0.0,
                   a_max=2.0, dt=0.1, predict_time=2.0,
                   safe_radius=1.5):
    """
    Dynamic Window Approach 用于 UAV 冲突消解
    """
    # 1. 构建动态窗口（当前可达速度集）
   Vw = []
    for v in np.arange(max(0, v_i[0] - a_max*dt), min(v_max, v_i[0] + a_max*dt), 0.1):
        for w in np.arange(v_i[1] - a_max*dt, v_i[1] + a_max*dt, 0.1):
            Vw.append((v, w))
    
    best_score = -float('inf')
    best_v = v_i
    
    for (v, w) in Vw:
        # 2. 预测轨迹
        traj = []
        p_pred = p_i.copy()
        v_pred = np.array([v, w])
        for t in np.arange(0, predict_time, dt):
            traj.append(p_pred.copy())
            p_pred = p_pred + v_pred * dt
        
        # 3. 碰撞检测
        collision = False
        for p_obs in obstacles:
            for p_t in traj:
                if np.linalg.norm(p_t - p_obs) < safe_radius:
                    collision = True
                    break
            if collision:
                break
        
        if collision:
            continue
        
        # 4. 评分函数
        score_heading = np.linalg.norm(p_pred - v_goal)  # 越小越好
        score_velocity = v  # 越大越好（偏好高速）
        score_clearance = min([np.linalg.norm(p_t - p_obs)
                               for p_obs in obstacles for p_t in traj])
        
        total_score = (
            2.0 * (1.0 / (score_heading + 1e-6)) +
            1.0 * score_velocity +
            0.5 * score_clearance
        )
        
        if total_score > best_score:
            best_score = total_score
            best_v = np.array([v, w])
    
    return best_v
```

#### 3.2.3 Distributed Model Predictive Control (DMPC)

DMPC ist eine gängige Methode für große Multi-UAV-Schwärme, bei denen jedes UAV:
1. Erstellen Sie ein lokales Vorhersagemodell basierend auf lokalen Informationen und Nachbarkommunikation
2. Lösen Sie lokale Optimierungsprobleme im endlichen Zeitbereich
3. Führen Sie nur den ersten Steuerungsschritt durch und führen Sie dann eine fortlaufende Neuoptimierung durch

**Kernkonsistenzbeschränkungen**:
$$
\sum_{j \in \mathcal{N}_i} a_{ij} (\mathbf{x}_i[k+k_p|k] - \mathbf{x}_j[k+k_p|k]) = 0, \quad \forall k_p \in \{1, \dots, N_p\}
$$

Dabei ist $\mathcal{N}_i$ die Menge der Nachbarn von UAV $i$ und $a_{ij}$ das Gewicht der Adjazenzmatrix.

Der Hauptvorteil von DMPC ist die **Skalierbarkeit**: Jede Drohne muss nur mit ihren Nachbarn kommunizieren, und der Rechenaufwand steigt nicht exponentiell mit der Anzahl globaler Drohnen.

### 3.3 Methode zur Zusammenarbeit mit mehreren Maschinen

#### 3.3.1 Konsistenzalgorithmus basierend auf der Graphentheorie

Modellieren Sie das Multi-UAV-System als Graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:
- Knoten $v_i \in \mathcal{V}$ repräsentiert die Drohne
- Kante $e_{ij} \in \mathcal{E}$ stellt die Kommunikationsverbindung dar

**Konsensprotokoll**:
$$
\dot{\mathbf{x}}_i = \sum_{j \in \mathcal{N}_i} a_{ij} (\mathbf{x}_j - \mathbf{x}_i)
$$Bei der Konfliktlösung wird die **Priorität** oder **Kostenfunktion** jeder Drohne als Zustandsvariable verwendet und die Lösungsaktion wird nach Konsenskonvergenz ausgewählt.

Vergleich gängiger Topologien:
- **Topologie des nächsten Nachbarn**: $\mathcal{O}(N)$ Verkehr, aber langsame Konvergenz
- **Vollständig verbundene Topologie**: schnelle Konvergenz, aber $\mathcal{O}(N^2)$ Verkehr
- **Metropolis-Hastings-Gewichtung**: Gleicht Konvergenzgeschwindigkeit und Kommunikationsaufwand aus

#### 3.3.2 Marktbasiert

Idee des bionischen Algorithmus: Behandeln Sie das Konfliktgebiet als „Ressource“, und jede Drohne konkurriert durch eine Auktion um das Recht, die Ressource zu nutzen:
1. **Gebot**: Jede Drohne berechnet die Dringlichkeit und die Kosten ihrer eigenen Mission
2. **Auktion**: Der Meistbietende erhält Vorfahrt, andere Drohnen warten oder fliegen umher
3. **Abrechnung (Zuteilung)**: Aktualisieren Sie die Ressourcenzuteilungstabelle, wiederholen Sie den Vorgang, bis kein Konflikt mehr besteht

```python
import heapq

def auction_based_resolution(uavs, conflict_zone, max_iterations=20):
    """
    基于拍卖的多机冲突消解
    uavs: List[UAV] - 无人机列表
    conflict_zone: 冲突区域中心及半径
    """
    allocation = {}  # zone_id -> winner_uav_id
    unallocated = list(uavs)
    
    for iteration in range(max_iterations):
        if not unallocated:
            break
        
        # 每次拍卖冲突区域使用权
        bids = []
        for uav in unallocated:
            urgency = uav.task_deadline - time.now()
            cost = uav.compute_detour_cost(conflict_zone)
            bid = urgency / (cost + 1e-6)
            bids.append((bid, uav))
        
        # 最高出价者胜出
        bids.sort(reverse=True)
        winner_bid, winner = bids[0]
        
        allocation[conflict_zone.id] = winner.id
        unallocated.remove(winner)
        
        # 对非胜出者计算绕行路径
        for uav in unallocated:
            uav.compute_alternative_path(conflict_zone)
    
    return allocation
```

#### 3.3.3 Spieltheoretische Methode

Modellieren Sie Konfliktlösung als **nicht-kooperatives Spiel**:
- Jede Drohne ist ein **Spieler**
- Die Flugbahn jeder Drohne ist **Strategie**
- Die Minimierung des eigenen Konfliktrisikos und der Fluchtkosten ist **Nutzenfunktion (Nützlichkeit)**

**Nash-Gleichgewicht** ist eine Reihe von Strategiekombinationen, bei denen kein Spieler durch einseitige Änderung seiner Strategie bessere Renditen erzielen kann:

$$
\forall i, \quad \mathbf{s}_i^* \in \arg\min_{\mathbf{s}_i \in \mathcal{S}_i}
\mathcal{J}_i(\mathbf{s}_i^*, \mathbf{s}_{-i}^*)
$$

**Korreliertes Gleichgewicht** lässt sich leichter verteilt lösen als das Nash-Gleichgewicht und ist in UAV-Clustern praktischer.

### 3.4 Lernmethode

#### 3.4.1 Reinforcement Learning (RL)

In den letzten Jahren hat **Deep Reinforcement Learning (DRL)** erhebliche Fortschritte bei der Lösung von UAV-Clusterkonflikten gemacht. Typischer Rahmen:- **Zustandsraum $\mathcal{S}$**: Positionen, Geschwindigkeiten, Zielpunkte, Hindernisse aller UAVs
- **Aktionsraum $\mathcal{A}$**: Geschwindigkeitsänderung $(\Delta v_x, \Delta v_y, \Delta v_z)$ oder Kurswinkeländerung
- **Belohnungsfunktion $\mathcal{R}$**:
  - Kollision: $r_{collision} = -100$
  - Nahe am Ziel: $r_{progress} = +10 \cdot \Delta dist$
  - Halten Sie einen Sicherheitsabstand ein: $r_{safety} = +5$ (wenn $\|p_i - p_j\| > d_{safe}$)
  - Energieverbrauch: $r_{Energie} = -0,1 \cdot \|\Delta v\|^2$

**MADDPG (Multi-Agent DDPG)** ist eines der am häufigsten verwendeten Multi-Machine-RL-Frameworks:

```python
# MADDPG 核心思路（伪代码）
class MADDPGAgent:
    def __init__(self, state_dim, action_dim, lr_actor=1e-4, lr_critic=1e-3):
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim * n_agents, action_dim * n_agents)
        self.target_actor = copy_network(self.actor)
        self.target_critic = copy_network(self.critic)
        self.replay_buffer = ReplayBuffer(capacity=1e6)
    
    def select_action(self, state, noise=0.1):
        action = self.actor.forward(state)
        action += noise * np.random.randn(action.shape)
        return np.clip(action, -1, 1)
    
    def update(self, batch):
        # 从全局视角更新 Critic（这是 MADDPG 的关键创新）
        states, actions, rewards, next_states = batch
        
        # 目标网络更新
        target_actions = [self.target_actor.forward(ns) for ns in next_states]
        target_Q = self.target_critic.forward(
            torch.cat(states), torch.cat(target_actions))
        
        # 均值聚集（Mean Aggregation）：所有智能体的目标 Q 值取平均
        target_Q = sum(target_Q) / n_agents
        
        # 策略更新
        ...
```

#### 3.4.2 Aufmerksamkeitsmechanismus (Aufmerksamkeit)

**Graph Attention Network (GAT)** wird verwendet, um die relative Wichtigkeitsbeziehung zwischen UAVs zu modellieren:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        self.W = nn.Linear(in_features, out_features)
        self.a = nn.Linear(2 * out_features, 1)
        
        self.leaky_relu = nn.LeakyReLU(alpha)
    
    def forward(self, h, adj):
        """
        h: (N, D_in) - N 个节点的特征
        adj: (N, N) - 邻接矩阵
        """
        Wh = self.W(h)  # (N, D_out)
        N = Wh.size(0)
        
        # 计算注意力系数
        a_input = torch.cat([
            Wh.repeat(1, N).view(N, N, -1),
            Wh.repeat(N, 1).view(N, N, -1)
        ], dim=2)  # (N, N, 2*D_out)
        
        e = self.leaky_relu(self.a(a_input).squeeze(2))  # (N, N)
        
        # Mask 掉非邻接节点
        e = e.masked_fill(adj == 0, -1e9)
        attention = F.softmax(e, dim=1)  # (N, N)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # 特征聚集
        h_out = torch.matmul(attention, Wh)  # (N, D_out)
        return F.elu(h_out)
```

Durch GAT können Drohnen adaptiv lernen, welche Nachbarflugzeuge den größten Einfluss auf ihre eigenen Entscheidungen haben, und so eine **sanfte Koordination** erreichen – sie muss nicht mit allen Drohnen kommunizieren, sondern muss nur auf Nachbarflugzeuge mit hoher Aufmerksamkeitsgewichtung achten.

#### 3.4.3 Nachahmungslernen

Trainieren Sie das Richtliniennetzwerk mithilfe von Expertentrajektorien (Lösungen von DMPC oder geometrischen Methoden):

$$
\mathcal{L} = -\mathbb{E}_{(s,a) \sim d_{\pi^*}}[\log \pi_\theta(a \mid s)]
$$

Die Methode **DAgger (Dataset Aggregation)** kann von Experten kommentierte Daten iterativ sammeln, um das Verteilungsverschiebungsproblem zu lösen.

---

## 4. Algorithmusvergleich und Auswahlhilfe| Algorithmuskategorie | Echtzeit | Skalierbarkeit | Optimalität | Umgang mit Unsicherheit | Typische Anwendungsszenarien |
|---------|--------|---------|--------|----------------|---------------------|
| **VO / Geometrie** | ⭐⭐⭐⭐⭐ | ⭐⭐ | Lokal optimal | ❌ | 2–5 Frames, Aufholkonflikte |
| **Potentialfeldmethode** | ⭐⭐⭐⭐⭐ | ⭐⭐ | Lokales Optimum | ❌ | Echtzeit-Hindernisvermeidung, dynamische Hindernisse |
| **Voronoi** | ⭐⭐⭐ | ⭐⭐⭐ | Lokal optimal | ❌ | Sparse-Cluster-Pfadplanung |
| **MILP** | ⭐ | ⭐⭐ | **Globales Optimum** | ⚠️ Skalierbar | ≤10 Racks Offline-Planung |
| **DMPC** | ⭐⭐⭐ | ⭐⭐⭐⭐ | Suboptimal | ⚠️ Eingebaut | 10–50 Rack-Cluster |
| **Graphentheorie/Auktion** | ⭐⭐⭐ | ⭐⭐⭐⭐ | Suboptimal | ❌ | Aufgabenzuweisung für große Cluster |
| **Spieltheorie** | ⭐⭐ | ⭐⭐⭐ | Relevantes Gleichgewicht | ⚠️ Skalierbar | Wettbewerbsszenarien |
| **DRL** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Optimale Strategie | ✅ Eingebaut | Über 50 Cluster, End-to-End |
| **GAT+RL** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Optimale Strategie | ✅ Eingebaut | Ultragroßer heterogener Cluster |

**Auswahlvorschläge**:
- **5 oder weniger**: Priorisieren Sie VO oder DMPC, die Berechnung erfolgt schnell und die Qualität der Lösung ist garantiert
- **5–50**: DMPC + Graph Coherence Protocol oder MADDPG
- **Mehr als 50 Racks**: GAT + Attention + RL, eine End-to-End-Strategie ist die realistischste Wahl
- **Konkurrierende Parteien**: Spieltheoretischer Rahmen (Nash-Gleichgewicht/korreliertes Gleichgewicht)
- **Erhebliche Unsicherheit** (GPS-Fehler, Windstörung): probabilistische Konflikterkennung + robustes MPC

---

## 5. Grenzfortschritte und Trends

### 5.1 End-to-End-Lernen zur VerstärkungDie [SMARTS](https://github.com/hijkzzz/SMARTS)-Plattform und das von Google DeepMind im Jahr 2023 veröffentlichte Projekt [AlphaPilot](https://www.microsoft.com/en-us/research/project/alpha-pilot/) förderten die Anwendung von End-to-End-RL in UAV-Clustern und zeigten, dass ein einziges Richtliniennetzwerk von der Wahrnehmung bis zur Entscheidungsfindung rohe Sensordaten direkt verarbeiten kann.

### 5.2 Föderiertes Lernen

Unter der Voraussetzung, die Privatsphäre der Daten jeder Drohne zu schützen, wird die Erfahrung mehrerer Drohnen durch föderiertes Lernen zusammengefasst:
1. Lokale Trainingsstrategie für jedes UAV
2. Laden Sie nur Farbverläufe anstelle von Rohdaten auf den zentralen Server hoch
3. Geben Sie neue Strategien heraus, nachdem Sie Updates aggregiert haben

Es löst die Probleme der Datenstreuung und Schwierigkeiten bei der Etikettenerfassung in UAV-Clustern.

### 5.3 Unsicherer robuster MPC

In den letzten Jahren modellierten **röhrenbasierte MPC** und **szenariobasierte MPC** Unsicherheit als begrenzte Störungen oder probabilistische Szenarien, um die Robustheit bei Optimierungsproblemen explizit einzuschränken:

$$
\forall \omega \in \mathbb{W}: \quad \mathbf{x}[k+1] = A\mathbf{x}[k] + B\mathbf{u}[k] + E\omega[k]
$$

Berechnen Sie „invariante Mengen“ vorab, um sicherzustellen, dass die Sicherheitsbeschränkungen auch im schlimmsten Fall noch eingehalten werden.

### 5.4 Multiobjektive Konfliktlösung

Bei konkreten Aufgaben muss bei der Konfliktlösung auch Folgendes berücksichtigt werden:
- **Aufgabenabschlussrate**: Gehen Sie nicht einfach herum und verursachen Sie eine Zeitüberschreitung der Aufgabe.
- **Energieverbrauch**: UAVs mit begrenzter Leistung müssen die zusätzliche Flugdistanz minimieren
- **Kommunikationsverzögerung**: Informationsverzögerungen in verteilten Systemen können zu Fehleinschätzungen führen
- **Fairness**: Bestimmte UAVs können nicht immer nachgeben (hungriges Problem)

Die **Pareto-Optimierungsgrenzen**-Suche ist das zentrale Werkzeug zur Lösung von Konflikten mit mehreren Zielen.

---

## 6. Zusammenfassung

Die Konfliktlösung bei der UAV-Pfadplanung ist ein Querschnittsproblem, das geometrische Berechnungen, Optimierungstheorie, verteilte Systeme und maschinelles Lernen umfasst. Von der frühesten geometrischen Ratenbarriere-Methode über die globale MILP-Optimierung bis hin zu verteiltem MPC und Deep Reinforcement Learning war die zentrale Triebkraft für die Algorithmenentwicklung immer:

> **Wie man sicherere Flugbahnen für mehr Drohnen in kürzerer Zeit und unter größerer Unsicherheit findet. **Der zukünftige Trend wird **hybride Architektur** sein: Verwendung von Lernmethoden, um schnelle lokale Entscheidungen zu treffen, Verwendung von Optimierungsmethoden zur Überprüfung globaler Trajektorien und Verwendung von Kommunikationsprotokollen, um die Konsistenz der Zusammenarbeit mehrerer Maschinen sicherzustellen. Die Kombination der drei kann tatsächlich einen sicheren, effizienten und skalierbaren autonomen Flug von UAV-Schwärmen ermöglichen.

---

**Referenzen** (sortiert nach Zeit):1. van den Berg, J., Lin, M. & Manocha, D. (2008). *Reziproke Geschwindigkeitshindernisse für Echtzeit-Multiagentennavigation.* IEEE International Conference on Robotics and Automation (ICRA).
2. Richards, A. & How, J. P. (2002). *Flugzeugflugbahnplanung mit Kollisionsvermeidung unter Verwendung gemischter ganzzahliger linearer Programmierung.* AIAA Guidance, Navigation, and Control Conference (GNC).
3. Alonso-Mora, J., et al. (2018). *Optimierungsbasierte Kollisionsvermeidung für Mehrfahrzeugsysteme.* IEEE Transactions on Robotics (TRO).
4. Everett, M., et al. (2021). *Kollisionsvermeidung bei dichtem Verkehr mit tiefgreifendem Verstärkungslernen.* IEEE International Conference on Robotics and Automation (ICRA).
5. Zhou, M., et al. (2019). *Eine Umfrage zur Pfadplanung für UAVs in überfüllten Umgebungen.* IEEE Transactions on Intelligent Transportation Systems (T-ITS).
6. Lowe, R., et al. (2017). *Multi-Agent-Akteur-Kritiker für gemischte kooperativ-kompetitive Umgebungen (MADDPG).* Konferenz über neuronale Informationsverarbeitungssysteme (NeurIPS).
7. Foerster, J., et al. (2018). *Kontrafaktische Multi-Agent-Policy-Gradienten (COMA).* AAAI-Konferenz über künstliche Intelligenz.
8. Rashid, T., et al. (2018). *QMIX: Faktorisierung monotoner Wertfunktionen für tiefes Lernen zur Verstärkung mehrerer Agenten.* Internationale Konferenz für maschinelles Lernen (ICML).
9. Veličković, P., et al. (2018). *Graph-Aufmerksamkeitsnetzwerke.* Internationale Konferenz über lernende Repräsentationen (ICLR).
10. Yan, C., et al. (2025). *Multi-Agent-Verstärkungslernen mit räumlich-zeitlicher Aufmerksamkeit für die Beflockung mit Kollisionsvermeidung einer skalierbaren Starrflügel-UAV-Flotte.* IEEE-Transaktionen auf intelligenten Transportsystemen (T-ITS).
11. Huo, D., et al. (2023). *Kollisionsfreie modellprädiktive Flugbahnverfolgungssteuerung für UAVs in Hindernisumgebungen.* IEEE-Transaktionen zu Luft- und Raumfahrt- und elektronischen Systemen (TAES).
12. Fan, T., et al. (2020). *Diverteilte Multi-Roboter-Kollisionsvermeidung durch Deep Reinforcement Learning für die Navigation in komplexen Szenarien.* The International Journal of Robotics Research (IJRR).
13. Jiang, C., et al. (2024). *Verteilte, auf Stichproben basierende modellprädiktive Steuerung über Glaubensausbreitung für die Navigation in Multi-Roboter-Formationen.* IEEE Robotics and Automation Letters (RA-L).
14. Goeckner, A., et al. (2024). *Graph Neural Network-based Multi-Agent Reinforcement Learning for Resilient Distributed Coordination of Multi-Robot Systems.* Internationale IEEE/RSJ-Konferenz über intelligente Roboter und Systeme (IROS).