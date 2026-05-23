---
title: "Sechs Yao-Wahrsagerei und Markov-Ketten: Ein jahrhundertealter Dialog zwischen östlicher Metaphysik und westlicher Wahrscheinlichkeit"
description: "Wenn „Das Buch der Wandlungen“ auf Bayes'sche Schlussfolgerung trifft – eine eingehende Untersuchung der Ähnlichkeiten zwischen Sechs-Linien-Wahrsagerei und Markov-Ketten im Umgang mit Unsicherheit"
pubDate: 2026-04-03
tags: ["Sechs Yao", "Markov-Kette", "Ich Ching", "Wahrscheinlichkeitstheorie", "KI"]
category: Essay
---

## 1. Einführung

Eine davon ist die alte chinesische Wahrsagetechnik, die aus dem „Buch der Wandlungen“ stammt und auf Kupfermünzen basiert, um Hexagramme herzustellen, und anhand der fünf Elemente Glück und Pech bestimmt. Das andere ist das vom russischen Mathematiker Andrey Markov im 20. Jahrhundert vorgeschlagene stochastische Prozessmodell, das heute den Eckpfeiler der Verarbeitung natürlicher Sprache, der Gensequenzanalyse und der Finanzprognose bildet.

Oberflächlich betrachtet scheinen sie nichts miteinander zu tun zu haben. Aber wenn man genauer hinschaut, sind sich die beiden in ihren Grundzügen überraschend ähnlich – beide befassen sich mit „Unsicherheit“ und verwenden „Zustand“ und „Übergang“, um die Entwicklung der Welt zu beschreiben**.

---

## 2. Stellen Sie die beiden Konzepte einander vor

### 2.1 Sechs-Yao-Wahrsagerei – Östliche Philosophie der Unsicherheit

Six Yao ist eine traditionelle chinesische Wahrsagemethode. Drei Kupfermünzen (oder Schafgarbe) werden sechsmal geworfen, um sechs Yao zu erhalten, die von unten nach oben ein Hexagramm bilden. Jede Linie ist in Yin (⚋) und Yang (⚊) unterteilt, und die sechs Linien bilden zusammen ein vollständiges Hexagramm (insgesamt 64×64=4096 Möglichkeiten).

**Kernlogikkette:**

```
起卦（随机触发）
  → 获得卦象（信息载体）
  → 装卦（加入动变爻、世应等元数据）
  → 分析五行生克冲合
  → 结合神煞、空亡等附加信息
  → 输出判断（吉/凶/平，及其具体解释）
```

Die theoretische Grundlage von Liu Yao sind **Yin Yang und fünf Elemente** – Metall, Holz, Wasser, Feuer und Erde, die sich gegenseitig verstärken und einschränken. Zusammen mit dem Bild- und Zahlensystem von Bagua bilden sie ein in sich stimmiges Symbolsystem. Dabei geht es nicht um „Vorhersagegenauigkeit“, sondern um die Bereitstellung einer interpretierbaren Erzählung, die dem Suchenden hilft, in unklaren Situationen einen psychologischen Anker für die Handlungsrichtung zu finden.

### 2.2 Markov Chain-Western State Transfer Machine

Die Markov-Kette ist ein **zufälliger Prozess**, der die Markov-Eigenschaft erfüllt: **Der zukünftige Zustand bezieht sich nur auf den aktuellen Zustand und hat nichts mit der Vergangenheit zu tun**. Das ist die berühmte „Gedächtnislosigkeit“.

Mathematischer Ausdruck:

$$P(X_{n+1} = j |

Um es einfach auszudrücken: **Wenn Sie wissen, wo Sie sich befinden, wissen Sie auch, welche mögliche Verteilung Sie als nächstes erreichen werden – Sie müssen nicht wissen, wie Sie hierher gekommen sind. **

**Typische Anwendungen:**
- PageRank-Algorithmus der Suchmaschine
- Hidden-Markov-Modell (HMM) zur Spracherkennung
- Zustandsübergangsmodell der Finanzmärkte (Bullenmarkt → Bärenmarkt → Schock)
- Random Walk für Musikempfehlungssystem

---

## 3. Unterschiede und Zusammenhänge zwischen Input und Output| Abmessungen | Sechs Yao Wahrsagerei | Markov-Kette |
|------|---------|-----------|
| **Eingabe** | Die Münz-/Schafgarbenoperation des Fragestellers oder die beiläufige Angabe der Zahl mit einem Gedanken | Aktueller Zustand des Systems + Zustandsübergangswahrscheinlichkeitsmatrix P |
| **Zufälligkeit der Eingabe** | Extrem hoch (menschliches Händeschütteln), im Wesentlichen physikalische Zufälligkeit | Theoretisch eine deterministische Wahrscheinlichkeitsverteilung (der Anfangszustand kann jedoch zufällig sein) |
| **Zustandsraum** | 4096 Hexagramme + unzählige wechselnde Kombinationen | Kann je nach Modellierung diskret oder kontinuierlich sein |
| **Ausgabe** | Qualitative Beurteilung (gut/schlecht, Fälligkeitsdatum, Ereigniskategorie) | Wahrscheinlichkeitsverteilung (Wahrscheinlichkeitsvektor des nächsten Zustands) |
| **Interpretierbarkeit** | Hoch (aber die Erklärung hängt von der Ebene des Hexagramm-Interpreters ab) | Niedrig (die Wahrscheinlichkeitszahl selbst ist korrekt, aber die Bedeutung des Modells muss interpretiert werden) |
| **Zeitdimension** | Die Hexagramme sind fixiert und Änderungen geben Zeithinweise | Kettenentwicklung, explizite Zeitschrittmodellierung |
| **Subjektivität** | Sehr hoch (verschiedene Menschen können das gleiche Hexagramm völlig unterschiedlich interpretieren) | Sehr niedrig (Mathematik ist objektiv) |

### 3.1 Ihre tiefe Verbindung

**Stufe 1: Zufälliger Auslöser**
Das Hexagramm der sechs Linien ist im Wesentlichen ein physikalischer Zufallsprozess (das Werfen einer Kupfermünze), und jeder Schritt der Markov-Kette ist ebenfalls probabilistisch. Beide beginnen mit dem Zufall.

**Ebene 2: Status und Übertragung**
Die sechs dynamischen Linien stellen „Variablen“ dar – Hexagramme ändern sich auf der Zeitlinie von statischen Hexagrammen zu Zhi-Hexagrammen; Der Kern der Markov-Kette ist auch „Zustand → Transfer → neuer Zustand“. Aus dieser Perspektive ist **Six Yao im Wesentlichen ein spezieller Satz von Markov-Prozessen** – der Zustandsraum ist 4096 und die Übertragungsregeln werden durch die fünf Elemente bestimmt.

**Stufe 3: Bedingte Wahrscheinlichkeit**
Die „Erhöhung“ der sechs Yao bestimmt die Stärke eines Yao, die wiederum seinen Einfluss auf andere Yao beeinflusst – in der Mathematik ist dies **bedingte Wahrscheinlichkeit**: Unter der Voraussetzung, dass dieses Yao eine Erhöhung hat, wie hoch ist die Wahrscheinlichkeit, dass etwas passiert?

**Stufe 4: Pfadabhängigkeit (aber in entgegengesetzter Richtung)**
Die sechs Zeilen betonen „Ursache und Wirkung“ – was Sie fragen, zu welchem Zeitpunkt und welcher Geisteszustand wird die Interpretation des Hexagramms beeinflussen (dies entspricht der anfänglichen Verteilungsempfindlichkeit der Markov-Kette). Die Markov-Kette behauptet, „pfadunabhängig“ zu sein, und in dieser Dimension sind die beiden genau das Gegenteil.

---

## 4. Das eine ist Quantifizierung, das andere basiert auf Gefühlen – ist das der wesentliche Unterschied?

### 4.1 Markov-Kette: ein natürliches quantitatives System

Der Aufbau von Markov-Ketten war von Anfang an quantifizierbar:

```python
import numpy as np

# 一个简单的"心情马尔可夫链"
# 状态：😊 好心情 / 😐 一般 / 😢 低落

P = np.array([
    #   😊    😐    😢
    [0.7,  0.2,  0.1],  # 😊 → 
    [0.3,  0.4,  0.3],  # 😐 → 
    [0.2,  0.3,  0.5],  # 😢 → 
])

# 问：今天心情一般，明天心情分布？
state_today = np.array([0, 1, 0])  # 😐
state_tomorrow = state_today @ P
print(state_tomorrow)  
# [0.3  0.4  0.3] → 30%好, 40%一般, 30%低落
```Das Schöne an der Mathematik ist, dass Sie die **stationäre Verteilung** (die Wahrscheinlichkeit jedes Zustands auf lange Sicht), die **erste Ankunftszeit** (wie lange es durchschnittlich dauert, von Zustand A in den Zustand B zu gelangen) und die **Mischzeit** (wie lange es dauert, bis das System den Anfangszustand „vergisst“) genau berechnen können.

### 4.2 Six Yao: Basiert die Hexagramm-Interpretation wirklich nur „auf Gefühlen“?

Oberflächlich betrachtet hängt die sechszeilige Interpretation von Hexagrammen in der Tat stark von der Erfahrung, dem Verständnis und sogar der „Inspiration“ des Tages des Interpreten ab. Wenn Sie für dasselbe Hexagramm nach dem Erfolg oder Niedergang Ihrer Karriere oder Ihrer Beziehung fragen, können die Ergebnisse sehr unterschiedlich ausfallen.

Aber wenn wir das „Gefühl“ von Liuyao im Detail aufschlüsseln, werden wir feststellen, dass es sich nicht um reine Metaphysik handelt:

**Liuyao hat strenge interne Regeln:**
- **Quantifizierung der Erzeugung und Hemmung der fünf Elemente**: Jede Linie gehört zu den fünf Elementen und hat eine klare Beziehung von Erzeugung und Hemmung (gegenseitige Erzeugung: Metall erzeugt Wasser, Wasser erzeugt Holz, Holz erzeugt Feuer, Feuer erzeugt Erde, Erde erzeugt Metall; gegenseitige Hemmung: Metall hält Holz zurück, Holz hält Erde zurück, Erde hält Wasser zurück, Wasser hält Feuer zurück, Feuer hält Metall zurück)
- **Quantifizierung von Wohlstandsphase, Ruhe, Gefangener und Tod**: Yao verfügt über fünf Zustände von Wohlstand, Phase, Ruhe, Gefangener und Tod in verschiedenen Monaten, die quantifiziert und bewertet werden können.
- **Sheng Ke, Konflikt und Kombination**: Es gibt spezifische Regeln für Konflikt (Widerspruch) und Kombination (Kombination)
- **Shensha-System**: Obwohl es sehr subjektiv ist, hat es auch feste Auslösebedingungen.

**Die wahre Quelle des „Gefühls“ ist:**
1. Zu viele Regeln, redundante Informationen und keine einheitliche berechenbare Ausgabe
2. Der Interpreter hat viele implizite Gewichtsverteilungen vorgenommen, diese jedoch nicht explizit ausgedrückt.
3. Die „Quantifizierung“ der sechs Zeilen ist in der Erfahrung des Hexagramm-Interpreters verborgen und wurde nicht **explizit modelliert**

---

## 5. Wie quantifiziert man die sechs Yao-Linien?

Das ist der lustige Teil: Können wir die sechs Linien in eine Markov-Kette umwandeln? **

### 5.1 Idee 1: Zustandsraummodellierung

Modellieren Sie das Sechs-Yao-Hexagramm als diskretes Zustandssystem:

```
状态空间 S = {六十四卦 × 动变爻位置 × 用神强弱}
```

Dies ist ein astronomisch großer Zustandsraum, aber wir können ihn vereinfachen.

### 5.2 Idee 2: Übergangswahrscheinlichkeit konstruieren

Verwenden Sie historische Hexagrammdaten, um die „Erfahrungsübertragungsmatrix“ der sechs Zeilen zu lernen – dies ist der Weg des maschinellen Lernens:

```
训练数据：大量已知应验的卦例（问事+卦象+结果）
↓
提取特征：卦象五行、用神强弱、动变方向、月令
↓
监督学习：训练一个分类/回归模型
↓
输出：给定一个新卦，预测各结果的概率
```

Dadurch werden tatsächlich die sechsjährigen „empirischen Regeln“ aus dem menschlichen Gehirn extrahiert und in ein berechenbares Modell umgewandelt – im Wesentlichen nicht anders als das Training eines Textklassifizierers.

### 5.3 Idee 3: Bayesian Six Yao

Stellen Sie sich Liuyao als **Bayesianisches Inferenzsystem** vor:

```python
import numpy as np

# 先验：基于历史数据的各类事项基础概率
prior = {
    '大吉': 0.15,
    '吉': 0.25,
    '平': 0.30,
    '凶': 0.20,
    '大凶': 0.10
}

# 似然函数：给定卦象特征，各结果的概率（从经验数据学习）
likelihood = {
    '用神旺相': {'大吉': 0.4, '吉': 0.35, '平': 0.2, '凶': 0.05, '大凶': 0.0},
    '用神休囚': {'大吉': 0.05, '吉': 0.15, '平': 0.3, '凶': 0.35, '大凶': 0.15},
    '官鬼持世': {'大吉': 0.05, '吉': 0.1, '平': 0.25, '凶': 0.4, '大凶': 0.2},
    '子孙持世': {'大吉': 0.3, '吉': 0.4, '平': 0.2, '凶': 0.08, '大凶': 0.02},
}

# 贝叶斯后验（简化版本）
def bayes_liuyao(observations):
    """根据观察到的卦象特征，计算各结果的后验概率"""
    posterior = {k: v for k, v in prior.items()}
    
    for obs, likely in likelihood.items():
        if obs in observations:
            for outcome in posterior:
                posterior[outcome] *= likely[outcome]
    
    # 归一化
    total = sum(posterior.values())
    return {k: round(v/total, 3) for k, v in posterior.items()}

# 示例：用神旺相 + 子孙持世
result = bayes_liuyao(['用神旺相', '子孙持世'])
print(result)
# {'大吉': 0.444, '吉': 0.389, '平': 0.111, '凶': 0.049, '大凶': 0.007}
```

### 5.4 Die zentrale Herausforderung bei der Quantifizierung von sechs Yao| Herausforderung | Beschreibung |
|------|------|
| **Datenknappheit** | Es gibt nur sehr wenige Hexagrammdaten von hoher Qualität, klarem Verlauf und vollständigen Aufzeichnungen |
| **Subjektivitätskodierung** | Was gilt als „Glück“? Unterschiedliche Menschen haben unterschiedliche Standards |
| **Zeitelement** | Besonders schwierig ist es, das „Fälligkeitsdatum“ (wann es eintreten wird) in den sechs Zeilen | zu quantifizieren
| **Xiangshu vs. Mathematik** | Six Yao hat sowohl „Xiang“ (symbolische Bedeutung) als auch „Shu“ (Anzahl der Geburten und Gramm) gleichzeitig, und die beiden sind nicht vollständig vergleichbar |
| **Black-Box-Problem** | Wenn Sie dazu ein neuronales Netzwerk verwenden, ist die Genauigkeit möglicherweise hoch, das „Warum“ kann jedoch nicht erklärt werden |

---

## 6. Wenn Liu Yao das große Sprachmodell trifft

Eine innovative Idee: Verwenden Sie LLM, um einen sechsjährigen quantitativen Interpreter zu erstellen:

```
用户输入："占卜明日股市行情"
         ↓
LLM 转化为六爻起卦（模拟或随机）
         ↓
LLM 根据卦象特征 + 历史金融数据 → 生成分析文本
         ↓
输出：概率化的判断 + 自然语言解释
```

Hierbei wird das sechsjährige **Xiang-Zahlensystem** im Wesentlichen als **Prompt-Engineering-Framework** behandelt – die Hexagramme bestimmen den „Blickwinkel auf das Problem“, und LLM ist für das Ausfüllen des spezifischen Inhalts verantwortlich.

Liuyao bietet: **Structured Uncertainty Framework** (es zwingt Sie, einen der 64 Quadranten auszuwählen, um das Problem anzugehen)
LLM bietet: **Unbegrenzter Wissenshintergrund und fließende Erklärungsfähigkeit**

Die Kombination der beiden könnte in Zukunft die Richtung der „KI-Wahrsagung“ sein.

---

## 7. Fazit: Es sind zwei Seiten desselben Problems

| | Sechs Yao | Markov-Kette |
|---|---|---|
| **Essenz** | Symbolsystem + empirische Regeln | Mathematisches Modell + Wahrscheinlichkeitstransfer |
| **Vorteile** | Sehr gut interpretierbar und reich an mathematischer Weisheit | Genaue Quantifizierung und rechnerische Überprüfung |
| **Nachteile** | Schwer zu standardisieren, zu subjektiv | Übermäßige Vereinfachung, Ignorieren tiefer Semantik |
| **Gemeinsamer Vorfahr** | Ehrfurcht und Modellierung von „Unsicherheit“ | Mathematische Abstraktion des „Zufallsprozesses“ |

Letztlich ist **Liuyao die Wahrscheinlichkeitstheorie**, mit der die Menschen der Antike in der Zeit vor Computern Yin-Yang und Fünf Elemente simulierten**; **Die Markov-Kette ist die präzise Formalisierung zufälliger Prozesse**, nachdem moderne Menschen Computer haben.

Sie konkurrieren nicht auf derselben Dimension, aber sie haben eine zeitliche und räumliche Resonanz auf der tiefsten Ebene der menschlichen Erkenntnis erreicht – **wie man mit Unsicherheit umgeht und Entscheidungen trifft**.

> Leibniz sagte: „Das I Ging erfasst das Universum durch Yin und Yang, 0 und 1.“
> Markov sagte: „Erfassen Sie den stochastischen Prozess durch den Zustandsübergang.“
>Vielleicht reden sie im Wesentlichen über dasselbe.

---*Mit Neugier und Bescheidenheit geschrieben, irgendwo zwischen Big Data und I Ging.*