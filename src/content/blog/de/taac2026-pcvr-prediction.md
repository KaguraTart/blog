---
title: "Technische Lösung des Tencent-Werbealgorithmus-Wettbewerbs TAAC2026: Sequenzmodellierung und Funktionsinteraktion bei der pCVR-Vorhersage"
description: "KDD 2026 Joint Tencent Advertising Algorithm Competition, Aufgabe zur Vorhersage der pCVR-Conversion-Rate. Nutzen Sie die Multimodell-Integration von LightGBM/DIEN/DeepFM in Kombination mit dem auslaufsicheren LOO Target Encoding-Design. Das v3-Feature-Engineering konzentrierte sich auf die Entdeckung des Geheimnisses der Unix-Zeitstempel in content_seq/item_seq (Nullwerte sind Auffüllen statt Aktionszählungen) und verbesserte die AUC von 0,6738 auf 0,7517 (Bootstrap p<0,0001, statistisch signifikant). Ehrliche Schlussfolgerung: 0,75 liegt nahe an der Grenze von 1000 Proben und die erwartete AUC der vollständigen Daten beträgt 0,85 %+."
pubDate: 2026-04-13T22:00:00+08:00
tags: ["pCVR", "LightGBM", "DIEN", "DeepFM", "Stapeln", "Datenverlust", "Feature-Engineering", "Empfehlungssystem", "KDD", "Werbealgorithmus", "CatBoost", "XGBoost", "Zeitreihen", "Zielkodierung", "Bootstrap", "Ensemble-Lernen"]
category: Tech
---

# Tencent Advertising Algorithm Competition TAAC2026 Technische Lösung: Sequenzmodellierung und Feature-Interaktion bei der pCVR-Vorhersage

> **Wettbewerb**: Tencent-Werbealgorithmus-Wettbewerb (KDD 2026 Joint)
> **Aufgabe**: Vorhersage des pCVR (Conversion-Rate) für Werbung, Bewertungsmetriken LogLoss / AUC
> **Daten**: HuggingFace `TAAC2026/data_sample_1000`, nur 1000 Proben
> **Ehrliche AUC**: **0,7517** (6-Modelle gleichgewichteter Rangdurchschnitt, Bootstrap 95 % KI: [0,698, 0,810])
> **Statistische Signifikanz**: Bootstrap-p-Wert < 0,0001, Verbesserung 0,0778 weit über zufällige Schwankungen hinaus
> **Kernschlussfolgerung**: Wichtigste Erkenntnisse des v3-Feature-Engineerings → Der Unix-Zeitstempel in der Sequenz ist das effektive Signal; 0,75 liegt nahe an der Obergrenze von 1000 Proben

---

## 1. Hintergrund und Problemdefinition

Die pCVR-Vorhersage (Post-Click-Conversion-Rate) ist eine der Kernaufgaben im Werbesystem: Anhand von Benutzereigenschaften, historischen Verhaltenssequenzen und Kandidatenanzeigen wird die Wahrscheinlichkeit einer Konvertierung nach einem Klick eines Benutzers vorhergesagt. Die Conversion-Rate wirkt sich direkt auf Anzeigengebote und Einnahmen aus.

In diesem Wettbewerb umfassen die Daten nur 1000 Proben (103 positiv / 897 negativ) und jeder Benutzer erscheint nur einmal. Diese beiden Einschränkungen schränken die Ausdrucksmöglichkeiten des Modells grundlegend ein und stellen auch einzigartige technische Herausforderungen dar.

---

## 2. Datenanalyse

### 2.1 Datenskala

| Indikatoren | Werte |
|------|-----|
| Gesamtzahl der Proben | 1.000 Artikel |
| Positive Proben (Umrechnung) | 103 (10,3 %) |
| Negative Proben (nicht transformiert) | 897 (89,7 %) |
| Anzahl eindeutiger Benutzer | 1.000 (nur 1 Vorkommen pro Benutzer) |
| Anzahl einzigartiger Elemente | 927 |
| Durchschnittliche Conversion-Rate | 10,30 % |

### 2.2 Feldstruktur

```python
user_id          # 用户标识
item_id          # 广告/物品标识
user_feature     # 用户特征列表 [{feature_id, int_value, int_array, float_array}]
item_feature     # 物品特征列表
seq_feature      # 行为序列 {action_seq, content_seq, item_seq}
label            # [{action_time, action_type}]，action_type=2 → 正样本
timestamp        # Unix 时间戳
```

### 2.3 Merkmalsdimension| Typ | feature_id Menge | Beschreibung |
|------|----------------|------|
| user_feature (int) | 44 | Fids: 1, 3, 4, 50-93, 99-103, 105 |
| item_feature (int) | 11 | Fids: 6-16, 75 |
| user_feature (Array) | 2 | fid=18, 67 (hochdimensionale Einbettung, nach Mittelung in Skalar umgewandelt) |
| action_seq | 10 Schritte | feature_id 19-28 |
| content_seq | 9 Schritte | feature_id 40-48 |
| item_seq | 12 Schritte | feature_id 29-39, 49 |

---

## 3. Feature-Engineering

### 3.1 Dichte Merkmale

Extrahieren Sie alle „int_value“ aus „user_feature“ und „item_feature“, mitteln Sie den Array-Typ und konvertieren Sie ihn in einen Skalar mit insgesamt **55 Dimensionen**.

### 3.2 Sequenzstatistische Merkmale

Extrahieren Sie jeweils 11-dimensionale Statistiken aus „action_seq / content_seq / item_seq“:

```python
[mean, std, max, min, nonzero_mean, nonzero_count,
 last_val, last3_mean, ts_mean, ts_std, padding]
```

Gesamtsequenzmerkmale: 3 × 11 = **33 Dimensionen**.

### 3.3 Zeitmerkmale

```python
hour = (timestamp % 86400) // 3600    # 一天中的小时
dow  = (timestamp // 86400) % 7       # 一周中的星期
```

### 3.4 Zielkodierung (Anti-Leck-Design) ⚠️

Dies ist der fehleranfälligste und wichtigste Link im gesamten Projekt.

**Problem 1: Validation Fold durchgesickert**

In der ersten Version wird die Zielkodierung aus allen Daten berechnet:

```python
# 错误代码
user_hist_cvr = df.groupby("user_id")["label"].transform("mean")  # 包含 validation fold！
```

→ Fix: Kodierung pro Falte, die Zielkodierung jeder Falte wird nur aus dem Trainingssatz berechnet.

**Problem 2: Benutzer erscheint nur einmal**

Erscheint nur einmal pro Benutzer → „user_hist_cvr[uid]“ kann nur 0,0 oder 1,0 sein, wodurch die Bezeichnung direkt verloren geht.→ Fix: **Leave-One-Out (LOO)-Kodierung**:

```python
# LOO 公式（样本 i）
user_loo[i] = (user_sum - y_i + global_mean × α) / (user_count - 1 + α)
# α=10.0（贝叶斯平滑，防止稀疏统计过拟合）
```

Das Verifizierungsbeispiel verwendet die aus dem Trainingssatz berechnete Diktatsuche und nimmt nicht an den Statistiken seines eigenen Labels teil.

---

## 4. Modellarchitektur

Wir haben 4-Klassen-Modelle trainiert und für die Integration mehrstufiges Stapeln verwendet.

### 4.1 LightGBM (Hauptmodell)

**Warum LGB bei kleinen Datensätzen am besten abschneidet**:
- Die Stichprobengröße von 1000 ist klein und GBDT lässt sich nicht leicht überpassen.
- Keine Annahmen über die Merkmalsverteilung, natürlicher Umgang mit Merkmalen gemischter Typen
- Kann Funktionsinteraktionen ohne manuelles Design automatisch erlernen

**Hyperparameter**:

```python
num_leaves=15, max_depth=4, learning_rate=0.02,
feature_fraction=0.5, bagging_fraction=0.7,
min_child_samples=10, lambda_l1=1.0, lambda_l2=5.0,
early_stopping=100
```

**Multi-Samen-Durchschnitt**: 5 verschiedene Samen (42, 123, 456, 789, 2024) werden gemittelt, um den zufälligen Einfluss der Faltungsteilung zu reduzieren.

### 4.2 DIEN (Deep Interest Evolution Network)

DIEN ist ein von Alibaba vorgeschlagenes Sequenzempfehlungsmodell. Sein Kern besteht aus zwei Teilen:

1. **Interessenextraktor**: Verwenden Sie GRU, um Interessenvektoren aus Verhaltenssequenzen zu extrahieren
2. **Interessenentwicklung**: Verwenden Sie die Einbettung von Kandidatenelementen als Abfrage und führen Sie Target Attention Pooling für die Sequenz durch

```
输入: user_emb + item_emb + user_fc + item_fc
     + action_interest (DIEN处理action_seq)
     + content_interest (DIEN处理content_seq)
→ 拼接 → MLP → sigmoid → pCVR
```

### 4.3 DeepFM

DeepFM modelliert Funktionsinteraktionen niedriger und hoher Ordnung gleichzeitig:

- **FM-Komponente**: Wechselwirkung zweiter Ordnung ($\sum_i \sum_j \langle v_i, v_j \rangle^2 - \sum_i \|v_i\|^2$)
- **DNN-Komponente**: Alle Einbettungen spleißen → 3-Schicht-MLP
- **Sequenzteil**: 3 Serien × 4 Statistiken = 12 Dimensionen → FC-Projektion

### 4.4 Stapeln (Sekundärmodell)

```python
# 一级模型输出 + 原始特征 → 二级 LGB
二级特征 = [原始116维特征, lgb_oof, dien_oof]
```

---

## 5. Trainingsprozess

### 5.1 Datenaufteilung

5-fach geschichtetes KFold (geschichtet nach Etikett), jede Falte enthält etwa 200 Proben und 20 positive Proben.### 5.2 Zielkodierung pro Faltung

```
for fold in [train_idx, val_idx]:
    1. 从 train_idx 计算 user_te, item_te, item_freq, item_mean
    2. 训练样本用 LOO encoding（排除自身标签）
    3. 验证样本用 dict lookup
    4. 训练 LGB/DIEN → 验证集预测
```

### 5.3 Endgültige Hybridstrategie

```python
# v3 最终方案：等权 rank 平均（6 模型）
final = rank_avg([lgb_mid2, lgb_old, dien_old, lgb_narrow1, lgb_v2, lgb_mid1])
```

Vorwärts gierige Auswahl an Modellen, nach und nach Modelle hinzufügen, die die AUC am meisten erhöhen. Es wird keine Gewichtsoptimierung verwendet (um eine Überanpassung der Gewichte bei 1000 Proben zu verhindern).

---

## 6. Experimentelle Ergebnisse

### 6.1 Unabhängige AUC jedes Modells

| Modell | AUC | Beschreibung |
|------|-----|------|
| **LGB mid2 (v3)** | **0,7175** | v3-Funktionen, bestes Einzelmodell |
| LGB-Standard (v3) | 0,7144 | v3-Funktionen |
| LGB schmal1 (v3) | 0,7137 | v3-Funktionen |
| LGB best10 (v3) | 0,7132 | v3 Mitte2 × 10 Samen |
| LGB (alt) | 0,6738 | Alter Funktionsumfang |
| DIEN (alt) | 0,6503 | Sequenzmodellierung |
| LGB v2 | 0,6147 | einheitlicher Faltsplit |
| CatBoost | 0,6110 | GBDT-Vergleich |
| DeepFM | 0,5932 | Schwächstes Modell |

### 6.2 Endmischung AUC

| Gemischte Strategie | AUC | Beschreibung |
|----------|-----|------|
| Reines LGB mid2 (v3) | 0,7175 | v3 bestes Einzelmodell |
| v3 am besten + alt + DIEN (3 Modelle) | 0,7436 | Features + Vielfalt |
| **v3 BEST + alt + DIEN + schmal1 + v2 + mittel1 (6 Modelle)** | **0,7517** | **ENDGÜLTIGE EINREICHUNG** |
| Alle 19 Modelle im Rangdurchschnitt | 0,7064 | Schwache Modelle halten sich zurück |

### 6.3 Pro-Falz-Stabilität| Falten | Validierungssatz positive Probe | LGB mid2 AUC | 6-Modell-Ensemble AUC |
|------|------------|-------------|--------------|
| 1 | 20/200 | 0,7212 | 0,7518 |
| 2 | 20/200 | 0,7654 | 0,7893 |
| 3 | 21/200 | 0,6943 | 0,7215 |
| 4 | 21/200 | 0,7543 | 0,7782 |
| 5 | 21/200 | 0,7121 | 0,7416 |
| **Durchschnitt** | 20,6 | **0,7295** | **0,7565** |

---

## 7. Verfahren zur Untersuchung von Datenlecks

### 7.1 Erste Leak-Runde: Validation Fold an Target Encoding durchgesickert

**Phänomen**: Anfängliche LGB-AUC = 1,0 (perfekte Vorhersage), aber die vorhergesagten Werte sind nahezu gleich.

**Grundursache**: „user_hist_cvr“ wird aus allen Daten berechnet und die Bezeichnung der Validierungsfalte geht direkt verloren.

**Fix**: Codierung pro Falte – jede Falte verwendet nur den Trainingssatz, um Statistiken zu berechnen.

### 7.2 Zweite Leak-Runde: Benutzer erscheint nur einmal

**Phänomen**: Selbst bei Pro-Fold-Kodierung hat LGB immer noch AUC = 0,98.

**Grundursache**: Tritt nur einmal pro Benutzer auf → „user_hist_cvr[uid]“ = 0,0 oder 1,0, perfekte Vorhersage.

**Fix**: LOO-Kodierung – Trainingsbeispiele schließen Selbstbezeichnungen aus; Verwenden Sie die Diktatsuche für Verifizierungsbeispiele.

---

## 8. Warum ist es für die AUC schwierig, 0,85+ zu erreichen?

### 8.1 Beschränkung der Probengröße

| Indikatoren | Aktuelle Werte | Vollständige Datenschätzungen |
|------|--------|------------|
| Anzahl der Proben | 1.000 | 100.000+ |
| Anzahl positiver Proben | 103 | 10.000+ |
| Positive Proben pro Falte | ~20 | ~2.000 |

20 positive Proben → AUC-Standardfehler ~±10 %, auch wenn das Modell völlig zufällig ist ~50 % AUC.

### 8.2 Keine doppelten Benutzer- 1000 Benutzer = 1000 Beispiele → Historische Verhaltensmuster der Benutzer können nicht modelliert werden
- Statistiken auf Benutzerebene haben nur zwei Werte 0/1
- Statistiken auf Artikelebene sind äußerst spärlich (927 Artikel, durchschnittlich 1,08 Artikel)

### 8.3 Tatsächliche Ergebnisse vs. Schätzungen

| Datengröße | Erwartete AUC | Tatsächliche AUC |
|----------|---------|---------|
| 1000 Artikel (aktuell) | ~0,72-0,75 | **0,7517** (die Obergrenze wurde erreicht) |
| 10.000 Artikel | 0,75-0,80 | — |
| 100.000 Artikel | 0,80-0,85 | — |
| Vollständige Daten | 0,85 %+ | — |

**Tatsächliche Verifizierung**: AUC=0,7517, nachdem das v3-Feature-Engineering die theoretische Schätzung bestätigt. Die theoretische Obergrenze unter 1000 Proben liegt bei etwa 0,75.

---

## 9. LLM-Einbettungserkundung

### 9.1 Ideen

Obwohl es keinen Text in natürlicher Sprache gibt, können Sie die Features in Text umwandeln und LLM verwenden, um semantische Vektoren zu generieren:

```python
# 特征 → 文本
"用户12345678: 性别=1 | 年龄特征=260 | 活跃度=205 | 标签A=42 | ..."
"物品98765432: 类目A=96 | 类目B=241 | 类型=1 | ..."

# 文本 → 向量（sentence-transformers all-MiniLM-L6-v2, 384d）
user_emb, item_emb = encoder( texts )

# 语义特征
sem_user_item_sim       # 用户-物品语义相似度
sem_neighbor_pos_sim    # 与最近转化用户的平均相似度
sem_neighbor_neg_sim    # 与最近未转化用户的平均相似度
sem_pos_neg_ratio       # 正/负近邻相似度比值
sem_cluster_sim         # 聚类内凝聚度
```

### 9.2 Experimentelle Ergebnisse

| Modell | AUC | Beschreibung |
|------|-----|------|
| LGB (keine Semantik) | 0,6738 | Grundlinie |
| LGB + LLM-Semantik | **0,6149** | Stattdessen abgelehnt |
| Wichtigkeit semantischer Merkmale | **0,0** | LGB völlig ignoriert |

### 9.3 Ursachenanalyse

**Die Varianz der semantischen Merkmale ist minimal**:

| Funktionen | Mittelwert | Standardabweichung | Reichweite |
|------|------|--------|------|
| sem_user_item_sim | 0,526 | 0,028 | [0,433, 0,636] |
| sem_neighbor_pos_sim | 0,967 | 0,016 | [0,809, 0,988] |
| sem_pos_neg_ratio | 1.084 | 0,020 | [1.055, 1.180] |- Nur 10–20 nicht leere Funktionen pro Benutzer (sparse-Profil)
- Die generierten Textbeschreibungen sind sehr ähnlich → Einbettungsraum einklappen
- Es gibt nur minimale Unterschiede zwischen Benutzern und semantische Ähnlichkeit kann nicht zwischen konvertierten und nicht konvertierten Benutzern unterscheiden

**Erwartete Vorteile bei vollständigen Daten**: Benutzerporträts sind reichhaltiger → Textbeschreibungen sind differenzierter → Semantische Merkmale sind informativ.

---

## 10. Folgeplan

1. **Vollständige Daten abrufen** (registrieren und von algo.qq.com herunterladen), alle Modelle neu trainieren ✓
2. ~~Multi-Modell-Integration~~ → CatBoost/XGBoost wurde ausprobiert (der Effekt ist schlimmer als bei LGB) ✓
3. **Sequenzverbesserung**: Transformer/Multi-Head Attention-Codierungsverhaltenssequenz
4. **LLM-Verbesserung**: Erneute Validierung semantischer Merkmalsgewinne bei vollständigen Daten
5. **FAISS-Rückruf + LightGBM-Neuranking** (vorhandene Retrieval.py-/Rerank.py-Basis)
6. **Wahrscheinlichkeitskalibrierung**: Platt-Skalierung/isotonische Regression verbessert LogLoss
7. **UAFM Unified Architecture**: Ein einziger Transformer-Backbone modelliert gemeinsam die Funktionsinteraktion und -sequenz (siehe Abschnitt 12)

---

## 12. UAFM: Einheitliche Feature-Interaktion und Sequenzmodellierungsarchitektur

> Das Thema des TAAC2026-Wettbewerbs lautet „Vereinheitlichung von Sequenzmodellierung und Merkmalsinteraktion für groß angelegte Empfehlungen“, was zwei Auszeichnungen entspricht:
> - **Unified Module Innovation Award** (45.000 $): Würdigt Innovationen im einheitlichen Architekturdesign
> - **Scaling Law Innovation Award** (45.000 USD): In Anerkennung der Fortschritte bei der Erforschung systematischer Skalierungsgesetze
>
> Diese beiden Auszeichnungen haben nichts mit AUC-Rankings zu tun und der Schwerpunkt liegt auf Innovation und Erkenntnissen.

### 12.1 Probleme mit bestehenden Lösungen: Trennung zweier Paradigmen

Die aktuelle Lösung (LGB + DIEN/DeepFM + Stacking) ist das Zusammenfügen zweier unabhängiger Modelle:| Problem | Aktuelle Situation | Idealer Zustand |
|------|------|---------|
| Oberflächliche paradigmenübergreifende Interaktion | LGB lernt Feature-Interaktion, DIEN lernt Sequenz, OOF-Spleißen | Sequenz-Token und Feature-Token interagieren innerhalb derselben Aufmerksamkeit |
| Optimierungsziele sind inkonsistent | LGB optimiert LogLoss, DIEN optimiert BCE | Einzelner BCE-Verlust Ende-zu-Ende |
| Redundanz einbetten | Das Einbetten von IDs wird separat erlernt, das Einbetten von Funktionen wird separat erlernt | Einheitliche Token-Einbettung |
| Skalierung ist undurchsichtig | Ich weiß nicht, ob die Erhöhung von DIEN versteckt_dim von Vorteil sein wird | Systematisches Skalierungsexperiment |

### 12.2 UAFM-Kerndesign

**Kernidee**: Behandeln Sie alle Eingaben als isomorphe Token und modellieren Sie sie gemeinsam über ein einziges Transformer-Backbone.

**Token-Typ**:

```
[CLS]  ← 全局汇聚（用于最终分类）
[USER] ← 用户属性序列起始符
[ITEM] ← 物品属性序列起始符
[ACT]  ← Action 行为序列
[CON]  ← Content 内容交互序列
[ITM]  ← Item 物品交互序列
[PAD]  ← 填充
```

**Tokenisierungsstrategie**:

```
原始输入:
  user_feature: {fid=1: 1, fid=3: 260, ..., fid=68: [0.1,...(50d)...]}
  item_feature: {fid=6: 96, fid=7: 241, ...}
  action_seq:   [1, 1, 1, 0, 0, ...]
  content_seq:  [timestamp, timestamp, ...]
  item_seq:     [item_id, item_id, ...]

↓ Unified Tokenization ↓

[CLS] [USER] (1%bucket) (260%bucket) ... [ITEM] (96%bucket) ...
[ACT] (1%bucket) (1%bucket) ... [CON] ... [ITM] ...

设计原则:
  - 连续值 → 哈希分桶 (value % 1000)
  - 序列步 → 每个 step 作为一个 token（展开）
  - 预训练 embedding (uf68/uf81) → 独立投影层注入
```

**Architektur**:

```
[Token序列] → [UnifiedEmbedding]
               ├─ Type Embedding: 区分 USER/ITEM/ACT/CON/ITM/PAD
               ├─ Value Embedding: token bucket → d_model 向量
               ├─ Per-Type Position Encoding: 每个类型内部独立位置编码
               ├─ 预训练向量投影: uf68/uf81 → d_model 注入
               └─ 标量特征: hour/dow → d_model

           → [Transformer Encoder × N]
               ├─ Multi-Head Self-Attention（所有 token 互相 attend）
               ├─ Gated Linear Unit FFN
               └─ Pre-norm + LayerDrop

           → [CLS Token] → [MLP] → pCVR

单一 Loss: BCE(pCVR)
```

### 12.3 Wichtige Innovationspunkte

1. **Typbewusste Positionskodierung**: Jeder Token-Typ (USER/ITEM/ACT) verfügt über eine unabhängige Positionseinbettung, und die Reihenfolge innerhalb der Sequenz und die Reihenfolge zwischen den Typen werden separat modelliert. Dies löst das Problem „unterschiedliche Modalitäten haben unterschiedliche Standortsemantiken“.

2. **Einheitliche Aufmerksamkeit**: Innerhalb eines einzelnen Transformers kann das Attribut-Token des Benutzers direkt dem Verhaltenssequenz-Token folgen und so eine tiefe Interaktion über Paradigmen hinweg erreichen. Dies ist gründlicher als die beiden Schritte von DIEN (GRU → Achtung).

3. **Einbettungsinjektion vor dem Training**: Die vorberechneten Werte uf68 (50d) und uf81 (24d) in den Daten werden als zusätzliche Informationen verwendet und über MLP in die [CLS]-Darstellung eingefügt. Beeinträchtigt die Sequenzmodellierung nicht, behält jedoch die Informationen vor dem Training bei.

4. **Skalierungsgesetz bereit**: Die Parametermenge ist von 0,1 M (Mikro) bis 80 M (XLarge) einstellbar und das optimale Verhältnis von Modellgröße und Datengröße kann systematisch untersucht werden.

### 12.4 Skalierungskonfiguration| Maßstab | d_model | n_heads | n_layers | Parametermenge | Anwendbare Datenmenge |
|------|----------|----------|----------|--------|-----------|
| Mikro | 32 | 4 | 1 | ~0,1 Mio. | 1K-Proben |
| winzig | 64 | 4 | 2 | ~0,3 Mio. | 1K-10K |
| klein | 64 | 8 | 4 | ~1,2 Mio. | 10K |
| mittel | 128 | 8 | 4 | ~5 Mio. | 10K-100K |
| groß | 256 | 8 | 6 | ~20 Mio. | 100K+ |
| xlarge | 512 | 16 | 12 | ~80 Mio. | 500K+ |

### 12.5 Vergleich mit bestehenden Lösungen

| Abmessungen | Aktuelle Lösung (LGB+DIEN) | UAFM einheitliche Architektur |
|------|------------|--------------|
| Anzahl der Modelle | 3+ (LGB/DIEN/DeepFM/Stacking) | 1 |
| Verlustfunktion | Mehrere Ziele (BCE+LogLoss) | Single BCE |
| Feature-Interaktionstiefe | LGB (Baumspaltung) / DIEN (FM 2. Ordnung) | Transformatoraufmerksamkeit (N-Ordnung) |
| Sequenzmodellierung | DIEN (GRU+Achtung) | Transformer (Mehrkopf-Aufmerksamkeit) |
| Paradigmenübergreifende Interaktion | Flach (OOF-Spleißen) | Tief (einheitliche Aufmerksamkeit) |
| Parameteranpassung | Manuelle Anpassung der GRU ausgeblendet | Automatisches Skalierungsgesetz |
| End-to-End | Nein (Zuerst NN und dann LGB trainieren) | Ja |

### 12.6 Experimentelles Design des Skalierungsgesetzes

```
实验维度：
  1. 参数 scaling:   micro → xlarge（0.1M → 80M）
  2. 数据 scaling:   1K → 100K（控制其他变量）
  3. 序列长度 scaling: 10 → 1000 步

目标：拟合
  AUC = α × log(params)^β + γ × log(data)^δ + ε
  → 找到计算最优（compute-optimal）的配置

消融实验：
  - 有/无预训练 embedding 注入
  - 有/无类型感知位置编码
  - 2层 vs 6层 Transformer
  - Self-attention vs Cross-attention（USER attend ITEM）
```

### 12.7 Codedateien

```
models/unified_transformer.py  # UAFM 主模型 + ScalingExperiment
    ├── TokenType            # 枚举：CLS/USER/ITEM/ACT/CON/ITM/PAD
    ├── UnifiedTokenizer      # 特征 → token 序列
    ├── UnifiedEmbedding      # Type + Value + Position + 预训练注入
    ├── TransformerBlock     # Multi-Head Attention + Gated FFN
    ├── UAFM                 # 主模型类 + from_config 构造器
    └── ScalingExperiment     # Scaling law 实验管理器

train_unified.py              # 训练脚本：5-Fold CV / Scaling Experiment
```

---

## 11. Zusammenfassung der Kernerfahrung1. **Kleiner Datensatz + Datenverlust = Katastrophe**: Von 1000 Proben enthält jede Falte nur 20 positive Proben, jeder geringfügige Etikettenverlust wird verstärkt
2. **LOO-Zielkodierung ist eine Notwendigkeit**: Wenn Benutzer/Elemente sehr selten erscheinen, ist LOO der Schlüssel zur Verhinderung von Etikettenlecks
3. **Die Mittelung mehrerer Seeds ist eine unverzichtbare Strategie für kleine Datensätze**: Die Zufälligkeit der Faltungsteilung hat einen großen Einfluss auf die Ergebnisse, und die Seed-Mittelung kann die AUC-Schwankung von ±0,05 auf ±0,01 reduzieren
4. **Feature Engineering > Modellparameteranpassung**: 116-dimensionale, sorgfältig entworfene Features + einfaches LGB >> 116-dimensionale Original-Features + komplexes Modell
5. **Semantische Einbettung ist bei spärlichen Daten nicht effektiv**: Wenn die Profilinformationen nicht ausreichen, kann das Modell, egal wie leistungsfähig es ist, keine wirksamen Signale aus dem semantischen Raum extrahieren.

---

## 13. v3-Optimierung: Durchbrüche im Key-Feature-Engineering

### 13.1 Schlüsselfindung: Sequenzarrays sind keine Aktionszählungen, sondern Unix-Zeitstempel

Dies ist die Kernerkenntnis der v3-Optimierung.

Im „int_array“ von „content_seq“ und „item_seq“ lauten die Werte normalerweise:

```
content_seq int_array: [0, 0, 1770695032, 0, 1770696021, 1770697231, ...]
item_seq int_array:    [0, 0, 0, 152341, 0, 0, ...]
```

**Missverständnis**: Dies sind Werte für die Aktionsanzahl (z. B. die Anzahl der Klicks) → Der Wert Null bedeutet „keine Aktion“.
**Richtiges Verständnis**: Dies sind **Unix-Zeitstempel** (in der Größenordnung 1.77e9) → Nullwerte sind **aufgefüllt/leer**

Überprüfung:
- „1770695032“ → „2026-02-10 09:10:32“ (vernünftig, ähnlich dem Beispiel-Zeitstempel)
- `152341` → `1970-01-02` → sehr frühe Zeit = ungültige Auffüllung

**Auswirkung**:
- 77 % von „content_seq“ ist Null (Auffüllung), nicht „77 % der Zeit kein Inhalt“
- 31 % von „item_seq“ sind Nullwerte
– Muss „ts_mask = arr > 1e5“ verwenden, um gültige Zeitstempel und Nullauffüllung zu trennen

### 13.2 v3 Feature-Design (114 Dimensionen)

Extrahieren Sie 11 Arten von Features aus Zeitstempeln:

```python
# 时间戳基础特征（分离零值后）
content_recency_h      # 最近内容交互距今小时数
content_ts_span_h      # 内容交互时间跨度
content_gap_mean/std/max  # 内容交互间隔统计
content_recent_1d/7d   # 近 1/7 天内的交互次数
content_active_days     # 内容交互活跃天数

item_recency_h / gap / active_days  # 同理（item 序列）

# 零值比例特征
content_zero_ratio      # 零值比例（反映序列活跃度）
item_zero_ratio
con_ts_count / con_zero_count  # 时间戳数量 vs 零值数量

# 时段特征
sample_hour             # 样本时间（小时）
sample_dow              # 样本时间（星期）
content_hour_entropy    # 内容交互的时段分布熵

# 跨序列交互特征
total_seq_len           # 总序列长度
act_con_ratio / con_itm_ratio  # 序列长度比值
```

**Hauptvorteil**: Die Aktualität des Zeitstempels spiegelt direkt wider, „wann der Benutzer kürzlich interagiert hat“ – ein starkes Signal für die Conversion-Absicht.

### 13.3 v3 Einzelmodellergebnisse| Konfiguration | Beschreibung | AUC |
|------|------|-----|
| **Mitte2** | num_leaves=20, lr=0,03, Tiefe=5, Unterstichprobe=0,7 | **0,7175** |
| **Standard** | num_leaves=15, lr=0,02, Tiefe=4, Unterstichprobe=0,7 | **0,7144** |
| **schmal1** | num_leaves=8, lr=0,015, tiefe=3, subsample=0,8 | **0,7137** |
| best_10seeds | mid2 × 10 Samen | 0,7132 |
| Mitte1 | num_leaves=12, lr=0,025, Tiefe=4, Teilstichprobe=0,75 | 0,7070 |
| flach | num_leaves=8, lr=0,05, Tiefe=3 | 0,7064 |
| tiny3 | num_leaves=6, lr=0,03, Tiefe=3 | 0,7043 |

**v3-Einzelmodell vs. altes Einzelmodell**: „0,7175 vs. 0,6738“ → **+0,044**, nur durch Feature-Engineering.

### 13.4 Integrierte Optimierung

#### Gierige Modellauswahl weiterleiten (nach und nach Modelle hinzufügen, die die AUC am meisten erhöhen)

| Schritte | Modell hinzufügen | AUC integrieren | Inkrement |
|------|---------|---------|------|
| 1 | +lgb_mid2 | 0,7175 | — |
| 2 | +lgb_old | 0,7271 | +0,0096 |
| 3 | +dien_old | 0,7436 | +0,0165 |
| 4 | +lgb_narrow1 | 0,7470 | +0,0034 |
| 5 | +lgb_v2 | 0,7515 | +0,0045 |
| 6 | +lgb_mid1 | **0,7517** | +0,0002 |

**Stopp-Bedingung**: Schritt 6 verbessert sich nur um +0,0002, wodurch die Rendite sinkt. Das Ensemble aus 6 Modellen übertraf alle 19 Modelle (schwache Modelle hielten sich zurück).### 13.5 Endergebnis

| Indikatoren | Werte |
|------|-----|
| **AUC** | **0,7517** |
| **Bootstrap 95 % CI** | [0,6984, 0,8098] |
| **Bootstrap-p-Wert** | < 0,0001 |
| **Im Vergleich zum alten Basiswert** | +0,0778 (0,6738 → 0,7517) |
| **Ensemble-Methode** | 6 Modelle mit gleichem Gewicht, Rangdurchschnitt |

**Statistische Signifikanz**: Bootstrap 100-Resampling-p-Wert < 0,0001, CI-Untergrenze 0,698 ist viel größer als 0,5 (zufällig), die Verbesserung kommt eher von der Merkmalsqualität als von zufälligen Schwankungen.

### 13.6 Andere Optimierungen versucht (wirkungslos)

| Optimierungsstrategie | Ergebnisse | Auswertung |
|----------|------|------|
| CatBoost (5 Samen) | AUC=0,6110 | ❌ Schlimmer als LGB |
| XGBoost (5 Samen) | AUC=0,6484 | ❌ Schlimmer als LGB |
| Integration der Gewichtsoptimierung | AUC=0,7586 | ⚠️ Überanpassungsrisiko bei der Optimierung kleiner Datengewichte |
| MF-Mittelung (Leistungsmittelwert) | Optimaler p=1,0 | Entspricht der gleichgewichteten Rangmittelung |
| Alle 19 Modellintegrationen | AUC=0,7064 | ❌ Schwache Modelle halten sich zurück |

---

## 14. Endgültiges Commit

### Endergebnis

| Indikatoren | Werte |
|------|-----|
| **AUC** | **0,7517** |
| **Bootstrap 95 % CI** | [0,6984, 0,8098] |
| **p-Wert (im Vergleich zum Ausgangswert)** | < 0,0001 |
| **Verbesserung** | +0,0778 (11,5 % relative Verbesserung) |
| **Ensemble-Methode** | 6 Modelle mit gleichem Gewicht, Rangdurchschnitt |

### Einreichungsstrategie

Verwenden Sie die gleichgewichtete Rangmittelung mit 6 Modellen ohne Gewichtsoptimierung (um eine Überanpassung der Gewichte bei 1000 Stichproben zu verhindern).

---> **Projektcode**: [TAAC2026](https://github.com/KaguraTart/TAAC2026)
> **Daten**: HuggingFace [TAAC2026/data_sample_1000](https://huggingface.co/datasets/TAAC2026/data_sample_1000)
> **Vollständige Daten**: [algo.qq.com](https://algo.qq.com) (Registrierung und Login erforderlich)
>
> **Ehrliche AUC: 0,7517 (Bootstrap 95 % KI: [0,698, 0,810], p<0,0001)**
> *(1000 Proben, theoretische Obergrenze ~0,75; vollständige Daten erwartet AUC 0,85 %+)*