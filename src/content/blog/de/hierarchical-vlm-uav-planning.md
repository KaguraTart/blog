---
title: "Hierarchische VLM-Planung: Lassen Sie die Drohne Anweisungen wie „landen auf der Ostseite von Gebäude 3“ verstehen"
description: "Eingehende Analyse der Anwendung des Vision-Language-Action-Modells (VLA) in der UAV-Pfadplanung, Kämmen des Evolutionswegs von der einzelnen End-to-End- zur hierarchischen semantischen Planung, Abdeckung wichtiger Arbeiten wie RT-2, OpenVLA, Compositional Foundation Models, LangStrands usw., Analyse, warum hierarchische Architektur die optimale Lösung für UAV-VLA ist, und Bereitstellung von Implementierungsrichtlinien."
tags: ["VLM", "VLA", "verkörperte Intelligenz", "Drohne", "hierarchische Planung", "RT-2", "OpenVLA", "Stiftungsmodell", "Robotik"]
pubDate: 2026-04-15
sourceHash: "aee6b5f68ec2e1d8ceaef7083cebe6ce1b1d2c08"
---

# Hierarchische VLM-Planung: Lassen Sie die Drohne Anweisungen wie „landen auf der Ostseite von Gebäude 3“ verstehen

## 1. Frage: Warum funktioniert „Direct End-to-End“ bei Drohnen nicht?

Stellen Sie sich vor, Sie geben einer Drohne den Befehl: **„Gehen Sie um Gebäude 2 herum und landen Sie auf dem freien Platz an der Ostseite von Gebäude 3“**.

Dieser Satz ist für den Menschen einfach, enthält aber drei Bedeutungsebenen:

1. **Semantische Ebene**: Finden Sie die Standorte von Gebäude 2 und Gebäude 3
2. **Räumliche Überlegungen**: Umgehen Sie Gebäude 2 vom aktuellen Standort aus und erreichen Sie die Ostseite von Gebäude 3
3. **Physische Ausführung**: Erzeugen Sie sanfte 3D-Trajektorien und steuern Sie Motoren für Echtzeitverfolgung

Wenn Sie reines End-to-End-VLA (Vision-Language-Action) verwenden, um motorische Steuersignale direkt aus Kamerabildern auszugeben, stehen Sie vor zwei grundlegenden Problemen:

- **Nichtübereinstimmung der Ausgangsfrequenz**: Die Inferenz des Sprachmodells von VLA dauert jeweils Hunderte von Millisekunden, aber die Drohnensteuerung erfordert Echtzeitsignale mit mehr als 100 Hz
- **Unkontrollierbare Sicherheit**: Das End-to-End-Black-Box-Modell kann nicht garantieren, dass die generierte Flugbahn den physischen Einschränkungen entspricht. Was soll ich tun, wenn ich ein Gebäude treffe?

Daher ist die **hierarchische VLM-Planung** zu einer gängigen Wahl für Industrie und Wissenschaft geworden.

## 2. Überblick über die Schichtarchitektur

```
用户指令: "绕过2号楼，在3号楼东侧降落"

┌─────────────────────────────────────────────┐
│ Layer 1: 语义理解层 (VLM/LLM)               │
│ 输入: 文本指令 + 地图/图像                   │
│ 输出: "先向北飞，绕过2号楼，再向东飞..."       │
│       (子目标序列)                            │
│ 模型: GPT-4V / LLaVA / Gemini 2.0 Flash    │
└─────────────────────────────────────────────┘
                    ↓ 子目标
┌─────────────────────────────────────────────┐
│ Layer 2: 轨迹规划层 (MPC / RRT* / ESDF)    │
│ 输入: 当前状态 + 子目标                      │
│ 输出: 空间路径点序列 (x,y,z,t)              │
│ 特点: 有理论安全性保证（可源性、碰撞检测）     │
└─────────────────────────────────────────────┘
                    ↓ 路径点
┌─────────────────────────────────────────────┐
│ Layer 3: 控制执行层 (PID / 非线性MPC)        │
│ 输入: 期望轨迹 + 当前状态                    │
│ 输出: 电机转速 / PWM 信号                    │
│ 频率: 100-400Hz (实时)                      │
└─────────────────────────────────────────────┘
```

Die Vorteile dieser Schichtung: **Jede Schicht ist entkoppelt und kann unabhängig mit der für diese Schicht am besten geeigneten Methode trainiert/optimiert werden**, ohne dass ein vollständiges End-to-End-Training erforderlich ist.

## 3. Detaillierte Erklärung jeder Ebene

### 3.1 Schicht 1: Semantische Verständnisschicht – Lassen Sie VLM die Anweisungen verstehen

Dies ist die „intelligentste“ Schicht in der Schichtarchitektur und dort sind große Modelle am nützlichsten.

**Kernmission:**
- Analysieren Sie Anweisungen in natürlicher Sprache, um wichtige Orientierungspunkte und Einschränkungen zu extrahieren („Gebäude 2“, „Ostseite“, „Umgehungsstraße“)
- Richten Sie 2D-/3D-Karteninformationen mit visuellen Beobachtungen aus
- Ausgabe der semantischen Teilzielsequenz (Teilzielliste)

**Schlüsseltechnologie: Visuelle Erdung**

Ordnen Sie einen abstrakten Verweis in der Sprache („Ostseite von Gebäude 3“) einem konkreten Ort in der Karte/dem Bild zu:

```python
# 伪代码示例
def parse_instruction(instruction: str, map_image: Image, ego_view: Image):
    # Step 1: 用 VLM 理解指令中的地标
    landmarks = vlm.extract_landmarks(
        instruction,  # "绕过2号楼，在3号楼东侧降落"
        map_image    # 鸟瞰地图
    )
    # → ["building_2 (polygon: [[x1,y1],...])", 
    #    "landing_zone: east_side_of_building_3"]

    # Step 2: 将地标转换为坐标
    goal_pose = convert_to_waypoints(landmarks, ego_view)

    # Step 3: 生成子目标序列
    subgoals = plan_subgoal_sequence(
        current_pose, goal_pose,
        constraints=[avoid(building_2), approach(building_3, side=east)]
    )
    # → ["fly_north_50m", "turn_east", "descend_20m", "hover_at_landing_zone"]

    return subgoals
```

**Modellauswahl:**| Modell | Vorteile | Nachteile | Anwendbare Szenarien |
|------|------|------|----------|
| GPT-4V / GPT-4o | Starke Denkfähigkeit, starke Multimodalität | Internetverbindung erforderlich, hohe Latenz | Cloud, latenzunabhängig |
| Gemini 2.0 Flash | Kostenlos, schnell, unterstützt die lokale Bereitstellung | Allgemeines Verständnis chinesischer Anweisungen | Lokale Edge-Bereitstellung |
| LLaVA 7B | Lokal einsetzbar, Open Source | Schwach im Verständnis komplexer Anweisungen | Edge-Drohne |
| Qwen2-VL | Chinesisch freundlich, Open Source | Edge-Bereitstellung muss quantifiziert werden | Inländische Szenarien |

**Jüngste Fortschritte (2024–2025):**

- **LLaVA-Plan** – Fügt einen auf LLaVA basierenden Planungsleiter hinzu, der auf die Aufgabenzerlegung spezialisiert ist
- **GPT-4o Live Voice** – umfassende Sprachbefehlsverständlichkeit, kein ASR erforderlich, keine Unterbrechungen

### 3.2 Schicht 2: Flugbahnplanungsschicht – von der Semantik bis zu räumlichen Pfaden

Diese Schicht empfängt die Unterziele der semantischen Schicht und gibt geometrische Pfade aus.

**Klassische Methode (nicht-lernender Stil):**

```python
# RRT* 全局路径规划
path = rrt_star(
    start=current_pose,
    goal=subgoal,
    obstacles=building_2_obstacle,  # 来自语义层的输出
    max_iterations=1000,
    connection_radius=5.0
)

# ESDF 局部避障（实时）
esdf_map = build_esdf_from_lidar(ego_view)
safe_direction = esdf_map.gradient_at(current_position)

# MPC 轨迹优化
trajectory = mpc.optimize(
    horizon=20,
    dynamics=uav_dynamics,
    obstacles=esdf_map,
    cost=[trajectory_smoothness, progress_to_goal, control_effort]
)
```

**Lernmethode (Reinforcement Learning):**

```python
# 策略网络：输入当前状态+目标 → 输出控制动作
policy = PPO(
    obs_dim=state_dim,      # 位置、速度、姿态、附近障碍物
    act_dim=action_dim,     # 速度指令 (vx, vy, vz)
)

# 在仿真中训练，通过 Domain Randomization 提升泛化性
# DR参数：风速、延迟、传感器噪声、空气质量
env.set_domain_randomization(
    wind=(0, 5),           # m/s
    comm_latency=(0, 100), # ms
    sensor_noise=(0, 0.05) # 归一化噪声
)
```

**Warum diese Schicht nicht durch reines RL ersetzen? **

Da reine RL-Trajektorien keine theoretischen Sicherheitsgarantien haben, kann RL Wege finden, die „machbar aussehen, aber tatsächlich an eine Wand stoßen“. Die Kombination aus ESDF + MPC bietet eine Erreichbarkeitsgarantie: **Solange MPC eine Lösung finden kann, wird es nicht auf Hindernisse stoßen**.

### 3.3 Schicht 3: Kontrollausführungsschicht – stabile Echtzeitverfolgung

Diese Schicht ist am ausgereiftesten und die traditionelle Kontrolltheorie reicht völlig aus:

```python
# 非线性 MPC 控制
class UAVController:
    def __init__(self):
        self.mpc = NMPC(
            horizon=10,          # 预测 10 步 (~1秒)
            dt=0.1,             # 控制周期 10Hz
            Q=diag([1,1,1]),    # 位置误差权重
            R=diag([0.1,0.1])   # 控制量权重
        )

    def control(self, state, ref_trajectory):
        # ref_trajectory 来自 Layer 2
        u = self.mpc.solve(state, ref_trajectory)
        return self.motor_mixer.mix(u)  # 转换为电机转速

    def safety_check(self, state):
        # 实时安全兜底：如果状态危险，强制悬停
        if state.altitude < 2.0 and state.speed < 0.5:
            return "LANDING"
        return "FLY"
```

## 4. Wichtige Forschungsarbeit

### 4.1 Kompositionelle Grundlagenmodelle für die hierarchische Planung

**Ajay et al., arXiv:2309.08587 (2023)**

Dieser Artikel ist eine sehr wichtige Grundlagenarbeit und schlägt das Konzept des „kombinierten Basismodells“ vor:- **Kernidee**: Verwenden Sie mehrere dedizierte Grundmodelle zum Kombinieren, jede Ebene erledigt eine Aufgabe und verwenden Sie die Kombination, um komplexe Aufgaben zu erledigen
- **Architektur**: visueller Encoder + Sprachmodell + Aktionsdecoder, hierarchische Kaskadierung
- **Experiment**: An einem Roboterarm verifiziert, was beweist, dass sich Layering besser verallgemeinern lässt als reines End-to-End

**Warum es für UAV inspirierend ist:** Durch die hierarchische Planung kann jede Ebene vorab trainierte Modelle unabhängig wiederverwenden, ohne dass das gesamte System für UAV-Szenarien neu trainiert werden muss.

### 4.2 LangStrands – Roboter mit natürlicher Sprachsteuerung

**LangStrands (2024)** – Verwenden natürlicher Sprache zur Steuerung von Robotern zur Durchführung von Aufklärungs-/Betriebsaufgaben in industriellen Szenarien:

-Unterstützt komplexe Anweisungen: „Überprüfen Sie zuerst die Ausrüstung in Bereich A, und wenn Sie eine Anomalie feststellen, melden Sie diese an Standort B.“
- Analysieren Sie Anweisungen in Task Graph und unterstützen Sie bedingte Verzweigungen und Schleifen
- Unterstützt die Zusammenarbeit mehrerer Roboter, jeder Roboter erhält unterschiedliche Unteraufgaben

**Referenzen von UAVs:** Die Ideen zur Missionskartenanalyse von LangStrands können direkt auf UAVs übertragen werden, beispielsweise auf komplexe Aufgaben wie „zuerst Aufklärung von 5 Zielpunkten und dann Rückkehr zur Basis“.

### 4.3 Verkörperter Gedankenbaum – Weltmodellgestützte Planung

**Xu et al., arXiv:2512.08188 (2025)**

- Verwenden Sie das Weltmodell, um die physischen Folgen (Änderungen des Umweltzustands) nach der Aktionsausführung vorherzusagen
- Verwenden Sie Tree of Thoughts, um vor der Ausführung nach der optimalen Unterzielsequenz zu suchen
- Physikalischer fundierter als reine VLM-Planung, Vermeidung von „Sieht richtig aus, ist aber physikalisch unmöglich“-Pfaden

**Wert für UAVs:** Wenn das UAV in der Luft fliegt, kann World Model die Auswirkungen von Böen und die Schwebefähigkeit nach dem Entladen der Batterie vorhersagen und im Voraus eine sicherere Flugbahn planen.

### 4.4 OpenVLA – Open-Source-Roboter-VLA

**OpenVLA (2024)** – Open-Source-VLA-Modell veröffentlicht von UC Berkeley:

- 7B Parameter, die 97 Arten von Roboteraktionen unterstützen
- Trainiert anhand von 220.000 realen Roboterdaten
- Läuft auf Consumer-GPU (RTX 3090)
- **Potenzial für UAV**: Obwohl OpenVLA derzeit hauptsächlich auf Roboterarme abzielt, kann die Architektur von VLA (visuelle Codierung + LLM + Aktionskopf) vollständig auf UAV-Szenarien migriert werden### 4.5 Embodied Arena – Vereinheitlichte Evaluierungsplattform für verkörperte Intelligenz

**Ni et al., arXiv:2509.15273 (2025)**

- Deckt mehr als 250 verkörperte Geheimdienstaufgaben und einheitliche Bewertungsstandards ab
- Einschließlich Indoor-Navigation, Betrieb, Luftflug und andere Aufgabentypen
- Bietet Leistungsbenchmarks für die UAV-Schichtung (Genauigkeit, Latenz, Erfolgsquote für jede Schicht)

**Wichtigkeit:** Mit einer einheitlichen Bewertungsplattform kann jede Schicht der Schichtarchitektur unabhängig bewertet werden und die Optimierung kann auf Beweisen basieren.

## 5. Sim2Real: Wie man trainierte Strategien auf echte Drohnen überträgt

Ein wesentlicher Vorteil der Schichtenarchitektur: **Jede Schicht kann unabhängig voneinander Sim2Real sein**, wodurch die Notwendigkeit einer vollständigen End-to-End-Migration entfällt.

### 5.1 Sim2Real-Schwierigkeitsanalyse jeder Ebene

| Ebene | Trainingsumgebung | Migrationsschwierigkeit | Kernherausforderungen |
|------|---------|---------|---------|
| Schicht 1 (VLM) | Beliebiges Bild/Karte | **Niedrig** | VLM wurde vorab trainiert und verfügt über eine starke Verallgemeinerung |
| Schicht 2 (RL) | AirSim / Flightmare | **Mittel** | Aerodynamische Parameter stimmen nicht überein |
| Schicht 3 (MPC) | Echte Drohnen-Parameteranpassung | **Niedrig** | Kalibrieren Sie einfach die Motorparameter |

### 5.2 Sim2Real-Strategie für Layer 2

**Domänen-Randomisierung (DR):**
```python
# 仿真训练时随机化关键物理参数
class SimEnv:
    def reset(self):
        self.wind = random.uniform(-3, 3)      # m/s 阵风
        self.motor_lag = random.uniform(0.8, 1.2)  # 电机响应系数
        self.battery_level = random.uniform(0.7, 1.0)  # 电池状态
        self.gps_noise = random.uniform(0, 0.5)  # GPS 噪声 (m)
```

**Real2Sim-Kalibrierung (echte Maschinenkalibrierung): **
```python
# 在真实无人机上跑系统辨识
def calibrate_dynamics(real_uav):
    # 激励信号：阶跃输入
    for amplitude in [0.1, 0.3, 0.5]:
        response = real_uav.step_input(thrust=amplitude)
        # 拟合真实电机响应曲线
        motor_model.fit(step_responses)

    # 将标定参数写回仿真
    sim_env.set_dynamics(motor_model.params)
```

### 5.3 Praxisfall: MADERs Sim2Real

**MADER (Multi-Agent DEep Reinforcement Learning for Aerial Swarms)** ist eine der besten Arbeiten von UAV Sim2Real in den letzten Jahren:

- Verwenden Sie MADDPG in AirSim, um auf mehreren Maschinen koordinierte Strategien zur Vermeidung von Hindernissen zu trainieren
- **Schlüsseltrick**: Fügen Sie während des Trainings eine Sensorverzögerung (20–50 ms) hinzu, damit die Strategie lernt, unter Verzögerung zu funktionieren
- **Ergebnisse**: Keine Probenmigration zur echten Tello-Drohne, Erfolgsquote bei der Vermeidung von Hindernissen > 85 %

## 6. Technische Implementierung: Aufbau einer hierarchischen VLM-Drohne von Grund auf### 6.1 Empfohlener Technologie-Stack

```
硬件:                      软件:
- Pixhawk 飞控 (或 Crazyflie)   - PX4 / ArduPilot 固件
- Jetson Orin NX (边缘计算)     - ROS 2 Humble
- Livox 激光雷达 / RealSense    - 深度相机 + IMU

软件分层:
┌──────────────────────────────────┐
│ VLM (Layer 1): LLaVA 7B/Qwen2-VL │
│ 推理引擎: llm.cpp / vLLM          │
│ 推理硬件: Jetson Orin NX (INT8)   │
└──────────────────────────────────┘
┌──────────────────────────────────┐
│ 规划器 (Layer 2):                │
│ - 全局: RRT* (OMPL)             │
│ - 局部: OSQP / Crocoddyl (MPC)   │
└──────────────────────────────────┘
┌──────────────────────────────────┐
│ 控制器 (Layer 3): PX4 SITL /     │
│              Ardupilot guided mode │
└──────────────────────────────────┘
```

### 6.2 ROS 2-Nachrichtenschnittstelle

```python
# Layer 1 → Layer 2 消息
class SubgoalMsg(Message):
    position: Point  # 目标点
    constraints: List[Constraint]  # 避障约束
    priority: int  # 优先级
    timeout: float  # 超时时间

# Layer 2 → Layer 3 消息
class TrajectoryMsg(Message):
    waypoints: List[PoseStamped]  # 路径点序列
    velocities: List[float]       # 期望速度
    start_time: Time             # 计划开始时间
```

### 6.3 Verzögerungsbudget (Echtzeitgarantie)

```
总延迟预算: < 500ms (可接受)
├─ 图像采集: 30ms (30fps)
├─ VLM 推理: 200-400ms (LLaVA 7B @ INT8)
├─ 轨迹规划: 50ms (RRT* + MPC)
└─ 控制器跟踪: 实时 (100Hz)
```

Wenn die VLM-Inferenz zu langsam ist, können Sie:
1. Mithilfe von Streaming Reasoning kann die Planungsebene die Zwischenergebnisse im Voraus abrufen
2. Leichtes Modell verwenden (LLaVA 3B / Qwen2-VL 2B)
3. Zwischenspeichern der Planungsergebnisse häufig verwendeter Anweisungen (gültig, wenn die Karte fixiert ist)

## 7. Aktuelle Herausforderungen und zukünftige Richtungen

### 7.1 Kernherausforderungen

1. **VLM-Inferenzlatenz**: LLaVA 7B leitet auf der Edge-GPU etwa 200–400 ms ab und übertrifft damit die Anforderungen für die Sicherheitsreaktion
2. **Befehlsmehrdeutigkeit**: Der mehrdeutige Befehl „Lande an einem sicheren Ort“ ist für VLM schwierig zu handhaben
3. **Mehrschichtige Fehlerakkumulation**: Semantischer Fehler von Schicht 1 → Pfadabweichung von Schicht 2 → Kontrolljitter von Schicht 3
4. **Dynamische Hindernisse**: Die Aktualisierungshäufigkeit der ESDF-Karte der Ebene 2 kann mit Hochgeschwindigkeitshindernissen (z. B. fliegenden Vögeln) nicht mithalten.

### 7.2 Zukünftige Richtungen

- **Multimodale Befehlsfusion**: Gemeinsames Verständnis von Stimme + Geste + Blickpunkt, Backup, wenn eine einzelne Modalität ausfällt
- **Lebenslanges Lernen**: Aktualisieren Sie Layer-2-Strategien online während des Fluges, um sich an unbekannte Umgebungen anzupassen
- **Kooperatives VLM mit mehreren Drohnen**: Ein VLM koordiniert mehrere Drohnen, anstatt dass jede unabhängig voneinander arbeitet
- **Weltmodellvorhersage**: Verwenden Sie ein generatives Modell, um den Flugverkehrsfluss in den nächsten 5 Sekunden vorherzusagen und ihn im Voraus zu vermeiden

## 8. Zusammenfassung

Die hierarchische VLM-Planung ist derzeit der praktikabelste Weg für UAV-Intelligence:

- **Schicht 1 (VLM)** ist für das semantische Verständnis und den Aufruf großer Cloud- oder Edge-Modelle verantwortlich
- **Layer 2 (Planner)** ist für die Konvertierung von Semantik in Geometrie verantwortlich und kann mit RL + MPC kombiniert werden
- **Schicht 3 (Controller)** ist für die Echtzeitverfolgung verantwortlich und die traditionelle Steuerungstheorie ist völlig ausreichend.

Der Hauptvorteil dieser Architektur: **Lassen Sie das große Modell tun, was es gut kann (Verstehen), lassen Sie die klassische Methode tun, was sie kann (Sicherheitsplanung)**, wobei jeder seine eigenen Aufgaben erfüllt, anstatt ein Black-Box-End-to-End-Modell zu verwenden, um alle Risiken zu tragen.

---

*Referenzen (in der Reihenfolge der Zitierung im Text)*1. Ajay et al., „Compositional Foundation Models for Hierarchical Planning“, arXiv:2309.08587, 2023
2. Padalkar et al., „OpenVLA: Open-Source Vision-Language-Action Model“, 2024
3. Liu et al., „Aligning Cyber Space with Physical World: A Comprehensive Survey on Embodied AI“, arXiv:2407.06886, 2024
4. Xu et al., „Embodied Tree of Thoughts: Deliberate Manipulation Planning with Embodied World Model“, arXiv:2512.08188, 2025
5. Ni et al., „Embodied Arena: A Comprehensive Evaluation Platform for Embodied AI“, arXiv:2509.15273, 2025
6. Mu et al., „Embodied AI-Enhanced IoMT Edge Computing: UAV Trajectory Optimization“, arXiv:2512.20902, 2025
7. Zhou et al., „OmniShow: Unifying Multimodal Conditions for Human-Object Interaction“, arXiv:2604.11804, 2026

*Autor: Kagura Tart | 15.04.2026*