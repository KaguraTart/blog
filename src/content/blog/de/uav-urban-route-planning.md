---
title: "Routenplanung für städtische Drohnen in geringer Höhe: Theorie und Algorithmus in CBD-Szenarien mit hoher Dichte"
description: "Analysiert systematisch die Kernprobleme und Lösungsideen der städtischen UAV-Routenplanung in geringer Höhe und deckt die Methoden A*, RRT*, APF, FM², MILP, ORCA und MARL mit vollständiger mathematischer Ableitung und Gleichungen ab."
pubDate: 2026-05-15
tags: ["UAV", "Wegplanung", "städtischer Luftraum", "Optimierungsalgorithmus", "UTM", "Konfliktlösung"]
category: Tech
sourceHash: "5588745289f1f698abd6def7ed9650375344a695"
---

# Städtische UAV-Routenplanung in geringer Höhe: Theorie und Algorithmus in CBD-Szenarien mit hoher Dichte

> Wenn Hunderte von Drohnen gleichzeitig zwischen Wolkenkratzern pendeln, ist die Routenplanung nicht länger ein einfaches Problem des „Fliegens von Punkt A nach Punkt B“ – es ist ein **hochdimensionales eingeschränktes Optimierungsproblem**, das ein Gleichgewicht zwischen dreidimensionalem Raum, Zeit, Energie und Sicherheit anstrebt.

---

## Einleitung: Warum ist CBD so schwierig?

Der städtische Luftraum in geringer Höhe wird normalerweise als Flugreichweite **0–300 m** (AGL) über dem Boden definiert. Dieses Höhenniveau ist zufällig das Hauptschlachtfeld für UAV-Logistik, Inspektion, Notfallmaßnahmen und andere Anwendungen. Das CBD (Central Business District) ist aus drei Gründen das komplexeste Teilszenario:

**1. Dichte Bebauung bildet einen „Urban Canyon“**

Durch Hochhäuser sind die verfügbaren Flugkorridore extrem eng und die Sichtlinie ist blockiert, was die Genauigkeit des GPS verringert. Auch die Kanten der Gebäude erzeugen starke Turbulenzen. In geringen Höhen unter 50 Metern können diese Turbulenzen dazu führen, dass ein kleiner Multirotor völlig die Kontrolle verliert.

**2. UAVs mit hoher Dichte verursachen intensive Konflikte**

In einer Vorstadtszene fliegen möglicherweise nur wenige Drohnen gleichzeitig. Während bei einem ausgereiften UTM-System (Urban Air Traffic Management) die Anzahl der Drohnen über dem CBD mehr als 40 Drohnen pro Minute erreichen kann. Das bedeutet, dass die Konflikterkennung und -lösung (Conflict Detection & Resolution, CD&R) zum zentralen Engpass des Systems und nicht mehr zu einer Randfunktion wird.

**3. Dynamisches Hindernis und Multi-Constraint-Kopplung**

Neben Gebäuden müssen Drohnen auch mit temporären Flugverbotszonen, bemannten Flugrouten, Windfeldänderungen in Echtzeit und Sicherheitsrisiken aufgrund der Menschenansammlung am Boden umgehen – all diese Faktoren machen es für einen einzelnen Flugplanungsalgorithmus schwierig, allein damit umzugehen.

---

## 1. Problemmodellierung: Flugprobleme in mathematische Probleme umwandeln

### 1.1 3D-Belegungsraster

Diskretisieren Sie den städtischen Raum in einem Voxelgitter, und jedes Voxel zeichnet seinen Belegungsstatus auf:

$$
O(x,y,z) = \begin{cases} 1 & \text{Hindernis / Flugverbotszone} \\ 0 & \text{Flugbar} \end{cases}
$$

Die Voxelauflösung beträgt typischerweise 1–5 m und die CBD-Kernregion kann auf 0,5 m verfeinert werden. Die Gebäudehöhendaten stammen aus einer GIS-Datenbank (Geographic Information System) und werden in Kombination mit Echtzeitsensoren dynamisch aktualisiert.

### 1.2 Mathematische Definition der 4D-Flugbahn

Die Flugbahn eines einzelnen UAV ist eine zeitlich parametrisierte Raumkurve:$$
\boldsymbol{\xi}(t) = \bigl(x(t),\; y(t),\; z(t)\bigr), \quad t \in [t_0,\, t_f]
$$

Nach Einführung der Zeitdimension wird die Trajektoriendimension auf eine 4D-Raum-Zeit-Kurve $\boldsymbol{\xi}^{4D}(t) = (x,y,z,t)$ erhöht. Dies ist der Kerngedanke der sogenannten **4D-Trajektorienplanung**: Die Vermeidung räumlicher Konflikte durch Zeitplanung (der Zeitpunkt der Ankunft an einem bestimmten Punkt) ist günstiger als reine Raumumwege.

In einem System mit mehreren Maschinen müssen zwei beliebige UAVs zu jedem Zeitpunkt die Sicherheitsabstandsbeschränkungen erfüllen:

$$
\|\boldsymbol{\xi}_i(t) - \boldsymbol{\xi}_j(t)\|_2 \geq d_{sep}, \quad \forall\, i \neq j,\; \forall\, t \in [t_0, t_f]
$$

Dabei ist $d_{sep}$ der minimale sichere Abstand mit einem typischen Wert von 5–30 m (abhängig von Fluggeschwindigkeit und GPS-Genauigkeit).

### 1.3 Allgemeine Form eines Optimierungsproblems mit mehreren Zielen

Die Routenplanung ist im Wesentlichen ein eingeschränktes Optimierungsproblem mit mehreren Zielen:

$$
\min_{\boldsymbol{\xi}}\; J(\boldsymbol{\xi}) = w_1 J_{len} + w_2 J_{Zeit} + w_3 J_{Energie} + w_4 J_{Risiko}
$$

Die einzelnen Unterpunkte haben folgende Bedeutung:

| Aufschlüsselung | Bedeutung | Typische Maßnahmen |
|------|------|----------|
| $J_{len}$ | Pfadlänge | $\int_{t_0}^{t_f}\|\dot{\boldsymbol{\xi}}\|\,\mathrm{d}t$ |
| $J_{time}$ | Flugzeit | $t_f - t_0$ |
| $J_{Energie}$ | Energieverbrauch | $\int P(v)\,\mathrm{d}t$ |
| $J_{Risiko}$ | Bodenrisiko | Punkte für das Überfliegen besiedelter Gebiete |

Einschränkungen (alle sind unverzichtbar):- **Zugänglichkeit**: $O(\boldsymbol{\xi}(t)) = 0,\;\forall t$
- **Kinematik**: $\dot{\mathbf{x}} = f(\mathbf{x}, \mathbf{u})$ (UAV-Kinematik-/Dynamikmodell)
- **Sichere Trennung**: $\|\boldsymbol{\xi}_i(t)-\boldsymbol{\xi}_j(t)\| \geq d_{sep},\;\forall i\neq j$
- **Randbedingungen**: $\boldsymbol{\xi}(t_0)=\mathbf{p}_{start},\;\boldsymbol{\xi}(t_f)=\mathbf{p}_{Ziel}$
- **Geschwindigkeitsbegrenzung**: $v_{min} \leq \|\dot{\boldsymbol{\xi}}(t)\| \leq v_{max}$

---

## 2. Pfadplanungsalgorithmus für eine einzelne Maschine

Bevor Sie sich mit der Zusammenarbeit mehrerer Maschinen befassen, sollten Sie zunächst den Kernalgorithmus in einem Einzelmaschinenszenario verstehen.

### 2.1 A*-Algorithmus: der Eckpfeiler der Diagrammsuche

A* sucht nach dem kürzesten Weg auf einem diskretisierten räumlichen Diagramm (Wegpunktdiagramm oder Sichtbarkeitsdiagramm). Der Bewertungswert jedes Knotens $n$ ist:

$$
f(n) = g(n) + h(n)
$$

Wobei $g(n)$ die **tatsächlichen kumulierten Kosten** vom Startpunkt bis zum Knoten $n$ sind:

$$
g(n) = g(\text{parent}) + d(\text{parent},\, n)
$$

$h(n)$ ist die zulässige heuristische Funktion von $n$ zum Ziel (überschätzen Sie niemals die wahren Kosten). Häufig verwendete euklidische Distanzheuristik im städtischen 3D-Raum:

$$
h(n) = \|\mathbf{p}_n - \mathbf{p}_{Ziel}\|_2 = \sqrt{(x_n-x_g)^2+(y_n-y_g)^2+(z_n-z_g)^2}
$$

In städtischen Szenarien reicht die alleinige Berücksichtigung der geometrischen Entfernung nicht aus. Einführung von **Ground Risk Weighted Edge Cost**:

$$
d(u,v) = \ell_{uv}\cdot\bigl(1 + \beta\cdot\mathcal{R}_{uv}\bigr)
$$Darunter ist $\ell_{uv}$ die Länge des Korridorsegments, $\mathcal{R}_{uv}\in[0,1]$ ist der Bodenrisikowert des Korridors (kombiniert Faktoren wie Bevölkerungsdichte, Gebäudetyp, Unfallfolgen usw.) und $\beta$ ist der Risikogewichtskoeffizient. Dies führt dazu, dass A* dazu tendiert, Routen zu wählen, die über Gebiete mit geringem Risiko (z. B. Flüsse, Parks) fliegen, auch wenn diese über leichte Umwege führen.

> Einschränkungen von A*: Die Qualität der Luftraumkarte bestimmt die Qualität des Verständnisses. Bei CBD mit hoher Dichte kann die Anzahl der Knoten im Diagramm Hunderttausende erreichen, und die Erstellung des Diagramms selbst ist eine Herausforderung.

### 2.2 RRT*-Algorithmus: Wahrscheinlichkeitsvollständige asymptotische Optimalplanung

RRT* (Rapidly-exploring Random Tree Star) erkundet mögliche Pfade durch zufällige Stichproben im kontinuierlichen Raum, was sich besonders für hochdimensionale und komplexe Hindernisszenen eignet.

**Nächste-Nachbarn-Abfrage** – Finden Sie den Knoten, der dem zufälligen Stichprobenpunkt im Baum $\mathcal{T}$ am nächsten liegt:

$$
x_{nearest} = \arg\min_{x \in \mathcal{T}} \|x - x_{rand}\|_2
$$

**Schritterweiterung** – Erweitern Sie die Schrittgröße $\delta$ von der Richtung $x_{nearest}$ zur Richtung $x_{rand}$:

$$
x_{new} = x_{nearest} + \delta \cdot \frac{x_{rand} - x_{nearest}}{\|x_{rand} - x_{nearest}\|_2}
$$

Die Kernverbesserung von **RRT* – Rewire:** Finden Sie alle benachbarten Knoten in der Kugel mit $x_{new}$ als Mittelpunkt und dem Radius $r_n$:

$$
r_n = \gamma_{RRT^*}\!\left(\frac{\log n}{n}\right)^{1/d}
$$

Dabei ist $n$ die Anzahl der Knoten des aktuellen Baums, $d$ die Raumdimension (3D-Szene $d=3$) und $\gamma_{RRT^*}$ eine Konstante, die sich auf das freie Raumvolumen bezieht. Dieser Radius schrumpft mit zunehmender Anzahl von Abtastpunkten und gewährleistet so eine asymptotische Optimalität.

Kostenaktualisierung:

$$
c(x_{new}) = c(x_{near}) + d(x_{near},\, x_{new})
$$

Wenn die Kosten von $x_{near}$ durch $x_{new}$ reduziert werden können, wird eine erneute Verbindung durchgeführt:$$
\text{Wenn } c(x_{near}) > c(x_{new}) + d(x_{new},\, x_{near}),\text{ dann ändere den übergeordneten Knoten von } x_{near} \text{ in } x_{new}
$$

Wenn sich die Anzahl der Stichproben der Unendlichkeit nähert, konvergiert RRT* garantiert mit der Wahrscheinlichkeit 1 zur optimalen Lösung:

$$
\lim_{n\to\infty} c(\xi_n^*) = c^* \quad \text{(fast sicher)}
$$

### 2.3 Künstliche Potenzialfeldmethode (APF): Der König der Echtzeit

APF konstruiert das Ziel als Gravitationsfeld und die Hindernisse als abstoßendes Feld, und das UAV bewegt sich unter der Wirkung der resultierenden Kraft.

**Gravitationspotential** (quadratischer Potentialtopf, der zum Ziel zieht):

$$
U_{att}(\mathbf{p}) = \frac{1}{2}k_{att}\|\mathbf{p} - \mathbf{p}_{Ziel}\|^2
$$

**Abstoßungspotenzial** (aktiviert innerhalb des Hinderniseinflussradius $\rho_0$):

$$
U_{rep}(\mathbf{p}) = \begin{cases} \dfrac{1}{2}k_{rep}\!\left(\dfrac{1}{\rho(\mathbf{p})}-\dfrac{1}{\rho_0}\right)^{\!2} & \rho(\mathbf{p}) \leq \rho_0 \\[8pt] 0 & \rho(\mathbf{p}) > \rho_0 \end{cases}
$$

Dabei ist $\rho(\mathbf{p})=\min_{obs}\|\mathbf{p}-\mathbf{p}_{obs}\|$ der Abstand vom UAV zum nächsten Hindernis.

**Resultierende Kraft** (negativer Gradient des gesamten Potentialfeldes):

$$
\mathbf{F}(\mathbf{p}) = -\nabla U_{att}(\mathbf{p}) - \nabla U_{rep}(\mathbf{p})
$$

Explizite Gradientenkomponenten:

$$
\nabla U_{att} = k_{att}\,(\mathbf{p}-\mathbf{p}_{Ziel})
$$$$
\nabla U_{rep} = k_{rep}\!\left(\frac{1}{\rho}-\frac{1}{\rho_0}\right)\!\frac{1}{\rho^2}\,\nabla\rho \qquad (\rho\leq\rho_0)
$$

Das Online-Update von APF ist in der Regel leichtgewichtig und für die Echtzeit-Hindernisvermeidung geeignet; Wenn jedoch die Entfernung des nächsten Hindernisses direkt $\rho(\mathbf{p})=\min_{obs}\|\mathbf{p}-\mathbf{p}_{obs}\|$ gemäß der oben genannten Definition berechnet wird, erfordert die naive Implementierung normalerweise mindestens das Überqueren des Hindernissatzes bei jedem Schritt, der etwa $O(n_{obs})$ beträgt. Einzelschrittabfragen können nur ungefähr $O(1)$ betragen, wenn Distanzfelder, ESDFs oder Rasterabfragen vorberechnet wurden. Aber es hat immer noch eine Achillesferse im CBD Canyon: **Lokales Minimum** – Wenn Schwerkraft und Abstoßung genau im Gleichgewicht sind, bleibt das UAV an einem Nichtzielpunkt stecken und kann sich nicht vorwärts bewegen. Zu den Verbesserungen zählen zufällige Störungen, harmonische Potentialfelder oder der PF-RRT-Algorithmus in Kombination mit RRT.

### 2.4 Fast Traveling Square Method (FM²): Die Eleganz der Wellenfrontausbreitung

FM² (Fast Marching Square) erzeugt durch die Lösung der Eikonal-Gleichung glatte Trajektorien, die sich besonders zur 4D-Konfliktvermeidung eignen.

**Eikonal-Gleichung** – eine partielle Differentialgleichung, die die Ankunftszeit der Wellenfront $T(\mathbf{x})$ beschreibt:

$$
|\nabla T(\mathbf{x})|^2 \cdot v^2(\mathbf{x}) = 1
$$

wobei $v(\mathbf{x})$ die Ausbreitungsgeschwindigkeit im Raum ist. Erstellen Sie eine **auf Abstand basierende Geschwindigkeitskarte**, damit die Wellenfront in der Nähe von Hindernissen auf natürliche Weise abbremst:

$$
v(\mathbf{x}) = c\cdot\rho(\mathbf{x}) = c\cdot\min_{obs}\|\mathbf{x}-\mathbf{x}_{obs}\|
$$

Nach der Lösung von $T(\mathbf{x})$ wird der Pfad durch Gradientenabstieg im Feld $T$ extrahiert:

$$
\dot{\boldsymbol{\xi}}(s) = -\frac{\nabla T\bigl(\boldsymbol{\xi}(s)\bigr)}{\left|\nabla T\bigl(\boldsymbol{\xi}(s)\bigr)\right|}
$$**Erweitert auf 4D-Konfliktvermeidung:** Einführung einer zeitvariablen Geschwindigkeitskarte, die $v\auf 0$ in der Raum-Zeit-Region lässt, die bereits von anderen UAVs besetzt ist:

$$
v(\mathbf{x},t) = v_0(\mathbf{x})\cdot\phi_{Konflikt}(\mathbf{x},t)
$$

Wenn $\phi_{conflict}\to 0$ ist, umgeht die Wellenfront auf natürliche Weise das Raum-Zeit-Konfliktvolumen und erreicht einen kollisionsfreien 4D-Pfad. Die von FM² erzeugten Pfade sind von Natur aus glatt ($C^\infty$ kontinuierlich) und erfordern keine zusätzliche Glättungsnachbearbeitung.

---

## 3. Das Kernproblem von Szenen mit hoher Dichte: Konflikterkennung und -lösung (CD&R)

Die grundlegende Herausforderung bei UAV-Szenarien mit hoher Dichte besteht nicht darin, einen Pfad zu finden, sondern sicherzustellen, dass alle Pfade gleichzeitig sicher sind.

### 3.1 Konflikterkennung

Definieren Sie den relativen Positionsvektor zwischen UAV $i$ und $j$:

$$
\Delta\mathbf{p}_{ij}(t) = \mathbf{p}_i(t) - \mathbf{p}_j(t)
$$

**Konfliktbestimmungsbedingungen** (horizontale **und** vertikale Trennung werden gleichzeitig verletzt):

$$
\text{Konflikt}_{ij} \iff \|\Delta\mathbf{p}_{ij}(t)\|_{xy} < d_h \;\wedge\; |\Updelta z_{ij}(t)| < d_v
$$

Siehe NASA UTM CONOPS typische Parameter: horizontaler Trennungsstandard $d_h=30\,\text{m}$, vertikaler Trennungsstandard $d_v=10\,\text{m}$.

In der Praxis muss das System Konflikte vor dem Flug vorhersagen, anstatt darauf zu warten, dass Konflikte auftreten, bevor es reagiert. Nehmen Sie an, dass das UAV innerhalb des Vorausschaufensters $[0, T_h]$ mit konstanter Geschwindigkeit fliegt:

$$
\mathbf{p}_i(t) = \mathbf{p}_i^0 + \mathbf{v}_i t, \quad \mathbf{p}_j(t) = \mathbf{p}_j^0 + \mathbf{v}_j t
$$

**Closest Point of Approach (CPA, Closest Point of Approach) Zeit:**$$
t_{CPA} = -\frac{\Delta\mathbf{p}_{ij}^0 \cdot \Delta\mathbf{v}_{ij}}{\|\Delta\mathbf{v}_{ij}\|^2}, \qquad \Delta\mathbf{v}_{ij} = \mathbf{v}_i - \mathbf{v}_j
$$

Mindestabstand bei CPA:

$$
d_{min} = \|\Delta\mathbf{p}_{ij}(t_{CPA})\|
$$

Wenn $d_{min} < d_{sep}$ und $t_{CPA}\in[0, T_h]$, wird festgestellt, dass ein **Vorhersagekonflikt** vorliegt und der Freigabemechanismus sofort ausgelöst werden muss.

### 3.2 Konfliktlösung

Entwaffnungsstrategien lassen sich in drei Kategorien einteilen und können einzeln oder in Kombination eingesetzt werden:

**Strategie 1: Geschwindigkeitsanpassung**

Wenden Sie einen Geschwindigkeitsskalierungsfaktor $\alpha$ auf das UAV $i$ an und verlangsamen oder beschleunigen Sie innerhalb des zulässigen Dynamikbereichs:

$$
\mathbf{v}_i^{new} = \alpha\,\mathbf{v}_i, \quad \alpha\in\!\left[\frac{v_{min}}{v_i},\;\frac{v_{max}}{v_i}\right]
$$

Das optimale $\alpha$ minimiert die Abweichung vom ursprünglichen Plan und erfüllt gleichzeitig die Trennungsbeschränkungen:

$$
\alpha^* = \arg\min_\alpha\;|\alpha-1| \quad \text{s.t. } d_{min}^{new}(\alpha)\geq d_{sep}
$$

**Strategie 2: Kurswechsel**

Drehen Sie die Flugrichtung des UAV $i$ um $\delta\psi$ in der horizontalen Ebene:

$$
\mathbf{v}_i^{new} = v_i\begin{pmatrix}\cos(\psi_i+\delta\psi)\\\sin(\psi_i+\delta\psi)\\0\end{pmatrix}
$$$$
\delta\psi^* = \arg\min_{|\delta\psi|}\;\delta\psi \quad \text{s.t. } d_{min}(\delta\psi)\geq d_{sep}
$$

**Strategie Drei: Trennung der Höhenschichten**

Im CBD-Szenario ist die Zuordnung fester Höhen entsprechend der Flugrichtung die effizienteste systematische Lösung:

$$
z_{Schicht}(k) = z_{Basis} + k\cdot\Delta z_{Schicht}, \quad k\in\{0,1,\ldots,N_{Schicht}-1\}
$$

Typische Konfiguration: Richtung Osten $\bis z_1$, Richtung Westen $\bis z_2$, Richtung Norden $\bis z_3$, Richtung Süden $\bis z_4$, Schichtabstand $\Delta z_{Schicht}=10\,\text{m}$. Dadurch wird die Dimensionalität des dreidimensionalen Kollisionsproblems auf ein zweidimensionales Problem reduziert, wodurch die Systemkomplexität erheblich verringert wird.

### 3.3 Dezentrale Koordination: Geschwindigkeitsbarrieren und ORCA

Zentralisiertes UTM kann die global optimale Lösung erhalten, aber der Kommunikationsaufwand steigt mit $O(N^2)$ als Anzahl der UAVs $N$, was in Szenarien mit extrem hoher Dichte zu einem Engpass führt. Unter den dezentralen Lösungen sind **Velocity Obstacle (VO)** und seine Verbesserung **ORCA** die ausgereiftesten Frameworks.

**Geschwindigkeitsbarriere** Definition – UAV $i$ Der Satz von Geschwindigkeiten, die aufgrund der Anwesenheit von UAV $j$ verboten sind (alle Geschwindigkeiten, die innerhalb des Zeitfensters $\tau$ zu einer Kollision führen würden):

$$
VO_{ij}^\tau = \left\{\mathbf{v}_i \;\middle|\; \exists\, t\in[0,\tau],\; \mathbf{p}_i+\mathbf{v}_i t \;\in\; \mathbf{p}_j+\mathbf{v}_j t \oplus \mathcal{D}(d_{sep})\right\}
$$

wobei $\mathcal{D}(r)$ eine Scheibe/Kugel mit dem Radius $r$ und $\oplus$ die Minkowski-Summe ist.

**Optimale reziproke Kollisionsvermeidung (ORCA)** – Jeder Agent trägt nur „die Hälfte“ der Vermeidungsverantwortung, um nicht zu konservativ zu sein. ORCA definiert eine Halbraumbeschränkung für Agent $i$ relativ zu $j$:$$
ORCA_{ij} = \left\{\mathbf{v} \;\middle|\; \bigl(\mathbf{v}-\mathbf{v}_{opt}^i\bigr)\cdot\hat{\mathbf{n}}_{ij} \geq \tfrac{1}{2}u_{ij}\right\}
$$

wobei $u_{ij}$ die Größe der minimalen Geschwindigkeitsänderung ist und $\hat{\mathbf{n}}_{ij}$ auf die Normalenrichtung der $VO_{ij}$-Grenze zeigt.

Die Menge der zulässigen Geschwindigkeiten für den Agenten $i$ (schneidet alle Nachbarbeschränkungen und dann die dynamischen Beschränkungen):

$$
\mathcal{V}_i^{ORCA} = \bigcap_{j\neq i} ORCA_{ij} \;\cap\; \mathcal{V}_{dyn}
$$

Darunter kodiert $\mathcal{V}_{dyn}$ dynamische Einschränkungen wie maximale Geschwindigkeit und Beschleunigung. ORCA hat in Dichteszenarien von mehr als 40 Bildern/Minute eine Erfolgsquote von 100 % erreicht, mit einer Rechenkomplexität von $O(N^2)$, was es für den Einsatz in Echtzeit geeignet macht.

---

## 4. Graphentheoretische Modellierung: städtisches Luftraumnetzwerk

### 4.1 Erstellung eines Streckennetzdiagramms

Der städtische Luftraum wird als gewichteter gerichteter Graph modelliert:

$$
G = (V,\; E,\; W), \quad W: E \to \mathbb{R}_+
$$

- **Knoten** $V$: über Straßenkreuzungen, Gebäudedächern, wichtigen Übergabepunkten
- **Kante** $E$: Legaler Flugkorridor zwischen zwei Knoten (muss die Überprüfung der Kollisionserkennung bestehen)
- **Kantengewichtung** $W$: Skalare Gewichtung mit mehreren Zielen

$$
W(e_{ij}) = w_1\, d_{ij} + w_2\,\Delta t_{ij} + w_3\,\mathcal{R}_{ij} + w_4\,\mathcal{E}_{ij}, \quad \sum_{k} w_k = 1
$$

Einschränkungen der Korridorkapazität (die Anzahl der gleichzeitig passierenden UAVs überschreitet nicht die Obergrenze):

$$
\text{load}(e_{ij},\, t) \leq C_{ij}, \quad \forall\, t
$$

Der Belegungsstatus des gesamten Luftraums kann durch einen vierdimensionalen Tensor beschrieben werden ($N_x\times N_y\times N_z$ ist das Voxelgitter, $N_t$ ist die Anzahl der Zeitschlitze):$$
\mathbf{A} \in \{0,1\}^{N_x\times N_y\times N_z\times N_t}, \quad A_{x,y,z,t} = 1 \iff \exists\text{ UAV besetztes Voxel}(x,y,z)\text{ im Zeitfenster }t
$$

### 4.2 Rotor-UAV-Energieverbrauchsmodell

Der Energieverbrauch ist ein wichtiges Optimierungsziel für die Routenplanung und erfordert eine genaue Modellierung.

**Schwebekraft** (abgeleitet aus der Blattelement-Impulstheorie):

$$
P_{hover} = \sqrt{\frac{(mg)^3}{2\,\rho_{air}\, A_r}}
$$

Dabei ist $m$ die Masse der Drohne, $g$ die Erdbeschleunigung, $\rho_{air}$ die Luftdichte und $A_r$ die Rotorscheibenfläche.

**Vorwärtsflugleistungsmodell** (Zeng et al. 2019, drei physikalische Komponenten):

$$
P(v) = \underbrace{P_0\!\left(1+\frac{3v^2}{U_{tip}^2}\right)}_{\text{Klingenwiderstand}} + \underbrace{P_i\!\left(\sqrt{1+\frac{v^4}{4v_0^4}}-\frac{v^2}{2v_0^2}\right)^{\!\frac{1}{2}}}_{\text{Induktionsleistung}} + \underbrace{\frac{1}{2}\,d_0\,\rho_{Luft}\,s\,A\,v^3}_{\text{Körperwiderstand}}
$$

Parameterbedeutung: $P_0$ ist die Widerstandsleistung des Typs Schwebeblatt, $P_i$ ist die induzierte Schwebeleistung, $U_{tip}$ ist die Rotorspitzengeschwindigkeit, $v_0$ ist die induzierte Schwebegeschwindigkeit, $d_0$ ist der Rumpfwiderstandsbeiwert, $s$ ist die Rotorfestigkeit und $A$ ist die Rotorscheibenfläche.

Energieverbrauch des Vorbeiflugsegments $e_{ij}$ (Länge $\ell_{ij}$, Geschwindigkeit $v$):

$$
\mathcal{E}_{ij} = \frac{\ell_{ij}}{v}\cdot P(v)
$$

**Optimale Reisegeschwindigkeit** (minimaler Energieverbrauch pro Distanzeinheit):

$$
v^* = \arg\min_v \frac{P(v)}{v}
$$

Für einen typischen kleinen Multikopter ($m\ungefähr 1\,\text{kg}$) liegt $v^*$ typischerweise zwischen 8–12 m/s.

---## 5. Windfeld und städtischer Canyon-Effekt

### 5.1 Städtische Windfeldmodellierung

Die Windgeschwindigkeitsverteilung in städtischen Schluchten ist viel komplexer als auf dem Land, und die Weibull-Verteilung wird häufig in der statistischen Modellierung verwendet:

$$
f(v_w;\, k,\, \lambda) = \frac{k}{\lambda}\!\left(\frac{v_w}{\lambda}\right)^{k-1}\!\exp\!\left[-\!\left(\frac{v_w}{\lambda}\right)^k\right]
$$

Darunter ist der Formparameter $k\ca. 1,5$–$2,5$ (der kleinere Wert wird verwendet, wenn die Turbulenzen in städtischen Gebieten stark sind) und $\lambda$ ist der Skalenparameter (kalibriert durch lokale meteorologische Messungen).

Logarithmisches Profil der oberflächennahen Windgeschwindigkeit (für Oberflächenschichten unterhalb der Dachhöhe):

$$
\bar{u}(z) = \frac{u^*}{\kappa}\ln\!\left(\frac{z - d_0}{z_0}\right), \quad \kappa = 0,41 \text{(von Kármán-Konstante)}
$$

Dabei ist $u^*$ die Reibungsgeschwindigkeit, $d_0$ die Höhe der Nullebenenverschiebung und $z_0$ die Rauheitslänge.

Quantitativer Einfluss von Windfeldern auf die Routenplanung:

**Windkorrigierte Reisezeit** (entlang der Korridorrichtungskomponente $v_w\cos\theta_w$):

$$
t_{ij} = \frac{d_{ij}}{v_{air} + v_w\cos\theta_w}
$$

**Segment-Energieverbrauchsintegral einschließlich Windwiderstand** (Tatsächliche Luftgeschwindigkeit = Bodengeschwindigkeit $-$ Windgeschwindigkeit):

$$
\mathcal{E}_{ij}^{wind} = \int_0^{t_{ij}} P\!\left(\|\mathbf{v}_{UAV}(t) - \mathbf{v}_w(t)\|\right)\mathrm{d}t
$$

**Turbulenzintensitätsindex** (quantifiziert das Korridorrisiko, Risikokomponente für Kantengewichte $\mathcal{R}_{ij}$):

$$
TI = \frac{\sigma_u}{\bar{u}}, \qquad \sigma_u = \sqrt{\overline{u'^2}}
$$

Korridore mit $TI > 0,3$ werden normalerweise als Hochrisikokorridore markiert und der Planer wird die Kantengewichtung dieses Segments aktiv vermeiden oder erhöhen.

### 5.2 Dynamischer SicherheitsradiusDie Turbulenzen um Gebäude herum nehmen mit abnehmendem Höhenspielraum stark zu. Daher sollte der Sicherheitsabstand keine feste Konstante sein, sondern sich dynamisch an die Flughöhe anpassen:

$$
d_{safe}(h) = d_{base} + \frac{k\cdot H_{bld}}{h - H_{bld} + \epsilon}
$$

Dabei ist $h$ die aktuelle Flughöhe, $H_{bld}$ die Höhe benachbarter Gebäude und $\epsilon$ der Regularisierungsterm, um zu verhindern, dass der Nenner Null ist. Diese Formel bedeutet, dass der erforderliche seitliche Abstand umso größer ist, je kleiner der Höhenabstand zwischen dem UAV und der Gebäudeoberkante ist.

Dynamische Headroom-Einschränkungen:

$$
\rho\bigl(\mathbf{p}(t)\bigr) \geq d_{safe}\bigl(z(t)\bigr), \quad \forall\, t \in [t_0, t_f]
$$

---

## 6. Kollaborative Optimierung mehrerer Maschinen: Globale MILP-Modellierung

Für das gemeinsame Pfad- und Zeitschlitzzuweisungsproblem von $N$-Drohnen kann ein **Mixed Integer Linear Programming (MILP)**-Modell erstellt werden, um die global optimale Lösung im kleinen bis mittleren Maßstab ($N\leq 50$) zu erhalten.

**Zielfunktion** (Minimierung der gesamten Fertigstellungszeit und des Energieverbrauchs aller Drohnen):

$$
\min\;\sum_{k=1}^{N}\!\left(w_1\, T_k + w_2\,\mathcal{E}_k\right)
$$

**Entscheidungsvariablen:**
- $x_{ij}^k \in \{0,1\}$: ob Drohne $k$ Korridor $(i,j)$ auswählt
- $t_i^k \geq 0$: Der Zeitpunkt, zu dem die Drohne $k$ am Knoten $i$ ankommt

**Einschränkung 1 – Verkehrseinsparung** (Jede Drohne betritt und verlässt den Zwischenknoten einmal):

$$
\sum_{j:(i,j)\in E}x_{ij}^k - \sum_{j:(j,i)\in E}x_{ji}^k = b_i^k, \quad \forall\, i\in V,\;\forall\, k
$$

Unter diesen entsprechen $b_i^k\in\{+1,\, 0,\, -1\}$ dem Startpunkt, dem Zwischenknoten bzw. dem Endpunkt.

**Einschränkung 2 – Korridorkapazität**:

$$
\sum_{k=1}^{N} x_{ij}^k \leq C_{ij}, \quad \forall\,(i,j)\in E
$$**Einschränkung 3 – Zeitkonsistenz** (Ankunftszeit entspricht Reisezeit):

$$
t_j^k \geq t_i^k + \frac{d_{ij}}{v_{max}}\cdot x_{ij}^k, \quad \forall\,(i,j)\in E,\;\forall\, k
$$

**Einschränkung 4 – Zeitliche Trennung** (verschiedene Drohnen auf demselben Knoten müssen ein Zeitintervall $\Delta t_{sep}$ einhalten, Big-M-Linearisierung):

$$
t_i^k - t_i^l \geq \Delta t_{sep} - M(1 - z_{kl}^i)
$$

$$
t_i^l - t_i^k \geq \Delta t_{sep} - M\, z_{kl}^i
$$

Unter diesen ist $z_{kl}^i \in \{0,1\}$ die Zeitreihenordnungsvariable von UAV $k$, $l$ am Knoten $i$ und $M$ ist eine ausreichend große Konstante (Big-M-Methode).

Wenn Geschwindigkeit auch als Entscheidungsvariable verwendet wird, wird das Problem auf Mixed Integer Nonlinear Programming (MINLP) erweitert:

$$
\min_{x,\, t,\, v}\;\sum_k\sum_{(i,j)} x_{ij}^k\cdot\frac{d_{ij}}{v_{ij}^k}\cdot P(v_{ij}^k), \quad v_{min}\leq v_{ij}^k\leq v_{max}
$$

MINLP ist ein NP-schweres Problem, das durch in der Praxis häufig verwendete heuristische Algorithmen (zufällige fraktale Suche SFS, Gepardenoptimierung MCO usw.) näherungsweise gelöst wird.

---

## 7. Verstärkungslernlösung: MARL und Aufmerksamkeitsmechanismus

Wenn die Größe der UAVs 100 übersteigt, ist die Rechenkomplexität von MILP nicht akzeptabel. **Multi-Agent Reinforcement Learning (MARL)** bietet eine Alternative für Offline-Training und extrem schnelle Inferenz.

### 7.1 Design der Belohnungsfunktion

Die Belohnung, die jede Drohne $i$ zum Zeitpunkt $t$ erhält:

$$
r_i^t = r_{Ankunft}\cdot\mathbf{1}[Ziel] - c_{Schritt} - c_{Konflikt}\cdot\mathbf{1}[Konflikt] - c_{Umweg}\cdot\|\mathbf{p}_i^t - \mathbf{p}_{direkt}\|
$$Die Bedeutung jedes Elements: $r_{arrive}$ ist die positive Belohnung für das Erreichen des Ziels; $c_{step}$ ist die Zeitstrafe für jeden Flugschritt; $c_{conflict}\cdot\mathbf{1}[conflict]$ ist die Strafe, wenn ein Konflikt auftritt; $c_{detour}$ ist die Umwegstrafe für das Abweichen von der Geraden.

### 7.2 Double-DQN-Update (diskreter Aktionsraum)

$$
Q(s,a;\theta) \leftarrow Q(s,a;\theta) + \alpha\!\left[r + \gamma\, Q\!\left(s',\,\arg\max_{a'}Q(s',a';\theta);\,\theta^-\right) - Q(s,a;\theta)\right]
$$

Das Online-Netzwerk $\theta$ wählt Aktionen aus und das Zielnetzwerk $\theta^-$ wertet Werte aus, wodurch Auswahl und Bewertung entkoppelt werden, um Überschätzungsfehler zu reduzieren.

### 7.3 Aufmerksamkeitsmechanismus: Nachbareinfluss modellieren

Die Entscheidungsfindung jeder Drohne im CBD erfordert die Erfassung des Status ihrer umliegenden Nachbarn. Der **Aufmerksamkeitsmechanismus** ermöglicht es dem Agenten $i$, den Einfluss von Nachbarn $j$ dynamisch zu gewichten:

$$
e_{ij} = \frac{(\mathbf{W}_Q\mathbf{h}_i)(\mathbf{W}_K\mathbf{h}_j)^\top}{\sqrt{d_k}}
$$

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_l \exp(e_{il})}, \qquad \mathbf{h}_i^{attn} = \sum_j \alpha_{ij}\,(\mathbf{W}_V\mathbf{h}_j)
$$

Das Aufmerksamkeitsgewicht $\alpha_{ij}$ spiegelt die Relevanz des Nachbarn $j$ für die Entscheidungsfindung des Agenten $i$ wider. Nachbarn mit geringen Abständen und großen Geschwindigkeitskonflikten erhalten naturgemäß höhere Gewichte.

### 7.4 PPO-Richtliniengradient (kontinuierlicher/gemischter Aktionsraum)$$
\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\hat{A}_t,\;\mathrm{clip}\!\left(r_t(\theta),\,1-\varepsilon,\,1+\varepsilon\right)\hat{A}_t\right)\right]
$$

wobei das Wahrscheinlichkeitsverhältnis ist:

$$
r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}
$$

Der Clip-Vorgang begrenzt die Aktualisierungsschrittgröße auf den Bereich von $[1-\varepsilon,\, 1+\varepsilon]$ (normalerweise $\varepsilon=0,2$), um zu verhindern, dass das Training aufgrund übermäßiger Richtlinienaktualisierungen abstürzt.

**Zentralisiertes Training, dezentrale Ausführung (CTDE)-Paradigma:**
- **Trainingsphase**: Das Bewertungsnetzwerk $V(s^{global};\phi)$ nutzt den globalen Status und kann alle Agenteninformationen wahrnehmen
- **Ausführungsphase**: Das Richtliniennetzwerk $\pi_\theta(a_i\mid o_i)$ verwendet nur lokale Beobachtungen des Agenten $i$, ohne Kommunikation

---

## 8. Flugbahnglättung: Bézier-Kurve und minimaler Fang

Das Ergebnis der Pfadplanung ist häufig eine Reihe diskreter Wegpunkte, und die direkte Verfolgung dieser Wegpunkte führt zu undurchführbaren scharfen Kurven. Es ist notwendig, durch **Trajektorienglättung** dynamisch realisierbare kontinuierliche Trajektorien zu erzeugen.

### 8.1 Bézier-Kurve

Eine Bézier-Kurve der Ordnung $n$ wird durch $n+1$ Kontrollpunkte $\{\mathbf{P}_i\}$ definiert:

$$
\boldsymbol{\xi}(u) = \sum_{i=0}^{n}\binom{n}{i}(1-u)^{n-i}u^i\,\mathbf{P}_i, \quad u \in [0,1]
$$

Geschwindigkeit (Ableitung nach Parameter $u$):

$$
\dot{\boldsymbol{\xi}}(u) = n\sum_{i=0}^{n-1}\binom{n-1}{i}(1-u)^{n-1-i}u^i\,(\mathbf{P}_{i+1}-\mathbf{P}_i)
$$Bézier-Kurven haben von Natur aus konvexe Hülleneigenschaften – die Kurve liegt immer innerhalb der konvexen Hülle der Kontrollpunkte, was die Kollisionsprüfung von Hindernissen erleichtert. Krümmungsbeschränkungen (Begrenzung der Zentripetalbeschleunigung):

$$
\kappa = \frac{\|\dot{\boldsymbol{\xi}}\times\ddot{\boldsymbol{\xi}}\|}{\|\dot{\boldsymbol{\xi}}\|^3} \leq \frac{a_{max}}{v^2}
$$

### 8.2 Minimum Snap: Die Standardlösung für Quadcopter

Bei einem Quadrocopter-UAV ist die Minimierung von Snap (der zweiten Ableitung der Beschleunigung) gleichbedeutend mit der Minimierung der Änderungsrate des erforderlichen Schubs, was zu einer optimalen Flugdynamik führt:

$$
\min\;\int_{t_0}^{t_f}\!\left\|\frac{d^4\boldsymbol{\xi}}{dt^4}\right\|^2\!\mathrm{d}t
$$

Wenn man die Flugbahn als stückweises Polynom $\boldsymbol{\xi}_k(t)=\sum_{j=0}^{m}c_{kj}t^j$ ausdrückt, kann das obige unendlichdimensionale Optimierungsproblem auf **quadratische Programmierung (QP)** reduziert werden:

$$
\min_{\mathbf{c}}\;\mathbf{c}^\top\mathbf{Q}\mathbf{c} \quad \text{s.t. }\mathbf{A}_{eq}\mathbf{c} = \mathbf{b}_{eq}
$$

Die Matrix $\mathbf{Q}$ kodiert das Snap-Integral (kann analytisch berechnet werden), und die Gleichheitsbeschränkung $\mathbf{A}_{eq}\mathbf{c}=\mathbf{b}_{eq}$ zwingt die Trajektorie, durch alle Pfadpunkte zu verlaufen und stellt die Kontinuität von Position, Geschwindigkeit und Beschleunigung zwischen Segmenten sicher.

---

## 9. Horizontaler Methodenvergleich| Methode | Vollständigkeit | Optimalität | Zeitkomplexität | Echtzeit | Skalierbarkeit auf mehreren Maschinen |
|------|--------|--------|------------|--------|------------|
| **A\*** | Komplett | Optimal (diskreter Graph) | $O(b^d)$ | Mittel | Schlecht |
| **RRT\*** | Wahrscheinlichkeitsvollständig | Asymptotisch optimal | $O(n\log n)$ | Besser | Mittel |
| **APF** | Unvollständig | Keine Garantie | $O(1)$/Schritt | Ausgezeichnet | Gut |
| **FM²** | Komplett | Optimal (kontinuierlich) | $O(N\log N)$ | Mittel | Mittel |
| **MILP** | Komplett | Globales Optimales | NP-hart | Schlecht | Mittel ($N\leq50$) |
| **ORCA** | Wahrscheinlichkeitsvollständig | Lokales Optimum | $O(N^2)$ | Ausgezeichnet | Ausgezeichnet |
| **MARL+Attn** | Vollständige Wahrscheinlichkeit | Ungefähr | Schweres Training, schnelle Schlussfolgerung | Gut | Ausgezeichnet |

**Auswahlvorschläge:**

- **Kleiner Maßstab, hohe Sicherheitsanforderungen** ($N\leq 20$) → MILP global optimal
- **Mittlere Skala, echtzeitsensitiv** (20 $ < N \leq 100$) → A\* / RRT\* + ORCA-Konfliktlösung
- **Großer Maßstab, hohe Dichte** ($N > 100$) → MARL + Aufmerksamkeitsmechanismus (Inferenzverzögerung $< 10\,\text{ms}$)

---

## 10. Zusammenfassung und Ausblick

Die Planung von UAV-Routen in städtischen Gebieten in geringer Höhe, insbesondere bei hoher Dichte, in CBD-Szenarien ist ein multidisziplinäres systemtechnisches Problem. Dieser Artikel sortiert die komplette Methodenkette von der **Einzelmaschinen-Pfadplanung** (A\*, RRT\*, APF, FM²) über die **Mehrmaschinen-Konfliktlösung** (CD&R, ORCA, MILP) bis hin zu **Lernmethoden** (MARL, PPO, Achtung) und gibt den genauen mathematischen Ausdruck jedes Kernglieds an.

**Drei große ungelöste Herausforderungen:**

1. **Online-Neuplanung in Echtzeit**: Wenn eine plötzliche Flugverbotszone oder ein Drohnenausfall auftritt, muss das System die Neuplanung aller betroffenen Flugbahnen innerhalb von 200 ms abschließen. Derzeit bleibt MILP weit hinter dieser Anforderung zurück und MARL ist der vielversprechendste Kandidat.2. **Digitale Zwillinge und Wahrnehmungsfusion**: Präzise dreidimensionale Stadtkarten in Echtzeit (einschließlich dynamischer Gebäudekonstruktion, temporärer Einfriedungen und meteorologischer Informationen) sind die Grundlage für die Qualität der Routenplanung. Es wird erwartet, dass die digitale Zwillingstechnologie eine Synchronisierung des Luftraumstatus auf Zentimeter- und Subsekundenebene erreichen wird.

3. **Technische Umsetzung des Regulierungsrahmens**: Die Vorschriften der Civil Aviation Administration of China (CAAC) für das Tiefflugmanagement, der europäische EASA U-Space und die amerikanischen FAA UTM CONOPS haben alle klare Anforderungen an die Konfliktlösungszeit, das Format der Flugplaneinreichung, Notlandeverfahren usw., und das Algorithmusdesign muss eng mit den regulatorischen Grenzen verknüpft werden.

> Aus mathematischer Sicht ist die Routenplanung für städtische Tiefflugrouten ein nicht-konvexes, nicht-lineares, gemischt-ganzzahliges, in Echtzeit eingeschränktes Optimierungsproblem mit mehreren Agenten. Kein einzelnes Framework kann es „mit einem Klick lösen“ – in der Ingenieurspraxis handelt es sich oft um eine mehrstufige Hybridarchitektur: Kartenplanung wird auf der strategischen Ebene verwendet, ORCA wird auf der taktischen Ebene verwendet und APF wird auf der Notfallebene verwendet, die zusammen ein robustes Flugverkehrsmanagementsystem bilden.

---

**Hauptreferenzen:**1. Karaman, S. & Frazzoli, E. (2011). *Abtastbasierte Algorithmen für optimale Bewegungsplanung.* International Journal of Robotics Research, 30(7), 846–894.
2. Van den Berg, J., Guy, S. J., Lin, M. & Manocha, D. (2011). *Reziproke N-Körper-Kollisionsvermeidung.* Robotics Research, 3–19.
3. Zeng, Y., Xu, J. und Zhang, R. (2019). *Energieminimierung für die drahtlose Kommunikation mit Drehflügel-UAV.* IEEE Transactions on Wireless Communications, 18(4), 2329–2345.
4. Mueller, M. W., Hehn, M. & D'Andrea, R. (2015). *Ein rechnerisch effizientes Bewegungsprimitiv für die Trajektorienerzeugung von Quadrocoptern.* IEEE Transactions on Robotics, 31(6), 1294–1310.
5. Brittain, M. & Wei, P. (2019). *Autonomer Fluglotse: Ein tiefgreifender Multiagenten-Lernansatz.* arXiv:1905.01303.
6. Bertram, J. & Wei, P. (2020). *Verteilte rechnerische Anleitung für urbane Luftmobilität mit hoher Dichte.* AIAA Aviation Forum.
7. Valavanis, K. P. & VachtsEvanos, G. J. (Hrsg.). (2015). *Handbuch unbemannter Luftfahrzeuge.* Springer.
8. Augugliaro, F., Schoellig, A. P. & D'Andrea, R. (2012). *Erzeugung kollisionsfreier Flugbahnen für eine Quadrocopter-Flotte.* IEEE/RSJ IROS, 3977–3982.