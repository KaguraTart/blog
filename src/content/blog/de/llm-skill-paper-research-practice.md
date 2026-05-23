---
title: "Von der Illusion zum praktischen akademischen Forschungsworkflow: Ich habe mit OpenClaw Skills ein Papierverfolgungssystem erstellt"
description: "Zeichnen Sie auf, wie ich zwei OpenClaw-Fähigkeiten, Papierrecherche und Papierverifizierer, entworfen habe, um einen Arbeitsablauf für die akademische Dokumentenrecherche aufzubauen, bei dem „echt und überprüfbar“ im Vordergrund steht. Grundprinzipien: Erstellen Sie keine falsche Literatur, manuelle Suche + werkzeuggestützte Sortierung und arbeiten Sie mit dem Zotero-Management zusammen, um einen vollständigen geschlossenen Kreislauf vom Abruf bis zur Überprüfung zu bilden."
tags: ["OpenClaw", "Fähigkeit", "akademische Forschung", "LLM", "Zotero", "Dokumentenmanagement", "LAPPEN", "Wissensmanagement"]
pubDate: 2026-04-15
---

# Von der Illusion zum praktischen akademischen Forschungsworkflow: Ich habe OpenClaw Skills verwendet, um ein Papierverfolgungssystem zu erstellen

## 0. Herkunft: Eine Lektion aus dem Beinahe-Überschreiten der Grenze

Im März 2026 bat ich AI, meine Arbeit für mich zu schreiben. Der Prozess verlief reibungslos – das Papier ist klar strukturiert, die Zahlen sind schön und die Daten sehen sehr vernünftig aus.

Dann entdeckte ich: **Diese Daten wurden von KI generiert, nicht aus echten Experimenten. **

92 % Erfolgsquote, Skalierbarkeitsanalyse von 500 Drohnen – es scheint tadellos, aber es ist alles ein „vernünftiger Abzug“. Wenn Sie es so einreichen, handelt es sich um akademischen Betrug.

Diese Erfahrung hat mich über eine Sache völlig nachdenken lassen: **Die größte Gefahr im LLM besteht nicht darin, die Frage nicht richtig beantworten zu können, sondern selbstbewusst zu antworten, aber die Antwort falsch zu bekommen. Im akademischen Umfeld kann dieses Selbstvertrauen fatal sein. **

Deshalb habe ich meinen Arbeitsablauf bei der Literaturrecherche neu gestaltet. Es gibt nur ein Grundprinzip:

> **Generieren Sie keine nicht überprüfbaren Inhalte. Alle Dokumente müssen manuell durchsucht werden und tatsächlich vorhanden sein; Alle Daten müssen gut dokumentiert sein. **

Dies ist die ursprüngliche Designabsicht der beiden Skills „Paper-Research“ und „Paper-Verifier“.

## 1. Der Kern des Problems: Warum ist die direkte Literatursuche der KI unzuverlässig?

Google Scholar verfügt über strenge Anti-Crawler-Mechanismen. Automatisierte Vorgänge von Selenium/Playwright sind anfällig für IP-Blockierung, das Auslösen von Bestätigungscodes und instabile Ergebnisse. Noch wichtiger: **Die von der KI durchsuchten Arbeiten existieren möglicherweise überhaupt nicht** – LLM ist gut in halluzinatorischen Antworten, die „vernünftig klingen“. In Literaturrezensionen ist dies das gefährlichste Minenfeld.

Häufige Falschliteratur-Routinen:
- Legen Sie den Titel und den Autor der Arbeit fest
- Herstellung nicht existierender Zeitschriftenkonferenzen
- Übertriebene Zitierzahlen oder Impact-Faktoren
- Nennen Sie gewöhnliche Zeitschriften die Top-Zeitschriften in der Region

Deshalb habe ich mich für eine Methode entschieden, die „dämlich“ aussieht, aber absolut zuverlässig ist: **Mensch-Maschine-Kollaboration, menschenzentriert**.

## 2. Gesamtdesign: Vierstufiger Closed-Loop-Workflow

```
手动搜索文献
     ↓
交互式收集元数据
     ↓
LLM 辅助生成综述
     ↓
Zotero 统一管理
     ↓
paper-verifier 交叉验证（可选）
```

**Kernlogik**: Menschen sind dafür verantwortlich, „echte Dinge zu finden“, und LLM ist dafür verantwortlich, „sie effizient zu organisieren“. LLM generiert niemals Inhalte, sondern organisiert und präsentiert nur echte, von Menschen eingegebene Informationen.

## 3. Papierrecherche-Fähigkeit: Vollständiger Literaturrecherche-Workflow

### 3.1 Schnelle Initialisierung

```bash
# 创建研究工作区
./scripts/setup_manual_search.sh ./my_research "UAV path planning"
```

Dadurch werden drei Dateien im angegebenen Verzeichnis generiert:

```
my_research/
├── 搜索指南.md         # 告诉你去哪搜、怎么搜
├── 文献记录模板.json   # 论文元数据存储格式
└── 收集文献.sh         # 交互式收集脚本
```

### 3.2 Suchanleitung: Manuelle Suchanleitung mit Strategie„Search Guide.md“ ist nicht nur „Google Scholar durchsuchen“ – es enthält:

**Empfohlene Datenbank**:
- Google Scholar (umfassend, am vollständigsten)
- IEEE Xplore (Ingenieurbehörde)
- Web of Science (Top Journal Index)
- ACM Digital Library (Computer)

**Suchstrategie**: 
```bash
# 示例：2023-2025 年 UAV 路径规划相关一区论文
"UAV path planning" AND "low altitude" AND year:2023..2025
site:ieeexplore.ieee.org "urban air mobility"
```

**Filterkriterien**:
- Sortieren Sie nach der Anzahl der Zitate und geben Sie den häufig zitierten Artikeln Vorrang
- Überprüfen Sie, ob sich die Zeitschrift im Distrikt 1 der Chinesischen Akademie der Wissenschaften befindet (Abfrage bei fenqubiao.com).
- DOI oder zugänglichen Link aufzeichnen

**Referenzliste für Bereich 1 der Chinesischen Akademie der Wissenschaften** (in Skill integriert):

| Geben Sie | ein Repräsentative Zeitschrift |
|------|---------|
| IEEE-Transaktionen | TRO, TITTEN, TAE, TCST |
| Roboter-Top-Themen | Automatica, JFR, RAS |
| Luftfahrtgipfel | ICRA, IROS, AIAA SciTech |

### 3.3 Interaktive Literatursammlung

Es gibt zwei Möglichkeiten, und interaktives Scripting wird empfohlen:

```bash
cd my_research
./收集文献.sh
```

Das Skript fordert Sie nach und nach zur Eingabe von Informationen zu jedem Artikel auf:

```
===== 文献收集 =====
论文标题: Multi-Agent Path Planning for UAV Swarms
作者: Zhang S, Li M, Wang W
年份: 2024
期刊/会议: IEEE Transactions on Robotics
DOI: 10.1109/TRO.2024.3391285
引用数: 45
关键词: UAV, path planning, multi-agent, reinforcement learning
一句话总结: 用集中式训练+分布式执行的框架解决
            无人机集群路径冲突问题...
继续添加下一篇? (y/n):
```

Wenn es zu langsam ist, können Sie die „documentation record template.json“ auch direkt bearbeiten:

```json
[
  {
    "title": "Multi-Agent Path Planning for UAV Swarms",
    "authors": ["Zhang S", "Li M", "Wang W"],
    "year": 2024,
    "venue": "IEEE Transactions on Robotics",
    "doi": "10.1109/TRO.2024.3391285",
    "citations": 45,
    "keywords": ["UAV", "path planning", "multi-agent"],
    "summary": "集中式训练+分布式执行框架解决无人机集群路径冲突"
  }
]
```

### 3.4 Markdown aus JSON Literature Review generieren

Nachdem die Sammlung abgeschlossen ist, erstellen Sie mit einem Klick eine strukturierte Bewertung:

```bash
python3 scripts/paper_collection.py \
    --input 文献记录模板.json \
    --output-md 文献综述.md
```

Das Ausgabeformat ist ungefähr wie folgt:

```markdown
## 1. 研究背景

### 1.1 无人机集群路径规划

无人机集群路径规划是城市低空空域管理的核心问题...
Zhang et al. (2024) 提出的集中式训练+分布式执行框架...

### 1.2 多智能体强化学习

MARL 是解决分布式协同决策的主流方法...

## 2. 方法论分类

### 2.1 基于优化的方法
...

### 2.2 基于学习的方法
...

## 3. 关键文献汇总

| 论文 | 年份 | venue | 方法 | 贡献 |
|------|------|-------|------|------|
| Zhang et al. | 2024 | TRO | MARL | 提出 MGAT-AC 架构 |
...
```

Die Rolle von LLM spiegelt sich hier perfekt wider – es geht nicht darum, Papiere für Sie zu finden, sondern die tatsächlich gesammelten Informationen manuell in einem strukturierten Rezensionstext zu organisieren**.

### 3.5 Zotero-Synchronisation: Aufbau einer persönlichen Literaturdatenbank

Gesammelte Literatur kann mit einem Klick in Zotero importiert werden:

```bash
export ZOTERO_LIBRARY_ID="你的图书馆ID"
export ZOTERO_API_KEY="你的API Key"

python3 scripts/zotero_manager.py \
    --library-id $ZOTERO_LIBRARY_ID \
    --api-key $ZOTERO_API_KEY \
    batch-add --file 收集的文献.json
```

Auf diese Weise werden alle manuell gesammelten echten Dokumente mit Zotero synchronisiert, um eine wiederverwendbare persönliche Dokumentenbibliothek zu bilden. Jedes Mal, wenn Sie neue Recherchen durchführen, können Sie die vorhandenen Bibliotheken schrittweise erweitern.

## 4. Fähigkeit des Papierprüfers: Kreuzvalidierung der AuthentizitätNach dem Sammeln der Literatur muss diese noch überprüft werden. Dies ist die zweite Fähigkeit – **Tool zur Überprüfung der Authentizität von Abschlussarbeiten**.

### 4.1 Warum ist eine Verifizierung erforderlich?

Auch bei manuellen Suchen können Fehler auftreten:
- DOI falsch ausgefüllt
- Der Name des Autors ist falsch geschrieben
- Der Zeitschriftenname wird als Abkürzung statt als vollständiger Name geschrieben
- Die Anzahl der Zitate wurde falsch erfasst (dutzende Male)
- Ich dachte, es wäre Zone 1, aber es stellte sich heraus, dass es nur CCF-B war.

Bevor Sie die Überprüfung abschließen, führen Sie daher mit dem Papierverifizierer eine Gegenprüfung durch.

### 4.2 DOI + Metadatenüberprüfung

```bash
python3 scripts/verify_papers.py \
    --input papers.json \
    --output verification_report.md
```

Format „papers.json“:

```json
{
  "title": "Multi-Agent Path Planning for UAV Swarms",
  "authors": "Zhang S, Li M",
  "year": 2024,
  "venue": "IEEE Transactions on Robotics",
  "doi": "10.1109/TRO.2024.3391285",
  "citations": 45
}
```

Das Skript wird über zwei APIs überprüft:

**Crossref API** – Erhalten Sie offizielle Metadaten:
- Stimmt der Titel genau überein?
- Ist die Autorenliste korrekt?
- Sind die Jahre konsistent?
- Wie lautet der vollständige Name der Zeitschrift?

**Semantic Scholar API** – Sekundäre Verifizierung:
- Ob das Papier tatsächlich existiert
- Wie hoch ist die tatsächliche Anzahl der Zitate?
- Besorgen Sie sich die Zusammenfassung der Arbeit, um den Inhalt zu überprüfen

### 4.3 Format des Verifizierungsberichts

```
## 核查报告

### ✓ Zhang et al. (2024) - TRO
- DOI: 10.1109/TRO.2024.3391285 → 有效
- 期刊: IEEE Transactions on Robotics → 中科院一区 ✓
- 引用数: 声称45 → 实际52（Semantic Scholar）
  ⚠️ 引用数有出入，差7次
- 作者: Zhang S, Li M, Wang W → 核对通过 ✓

### ✗ Li et al. (2023) - ICRA
- DOI: 10.1109/ICRA.2023.1001234 → 有效
- ⚠️ 论文标题不匹配：实际为 "Single-Agent ..."
  → 请核实是否填错了论文
```

Eventuelle manuelle Eingabefehler können so zeitnah erkannt werden.

## 5. Dateistruktur von zwei Skills

```
~/.openclaw/workspace/skills/
├── paper-research/          # 文献研究工作流
│   ├── SKILL.md            # 使用说明
│   ├── references/
│   │   └── zotero_setup.md # Zotero 配置指南
│   └── scripts/
│       ├── setup_manual_search.sh    # 工作区初始化
│       ├── paper_collection.py       # 文献收集+综述生成
│       ├── zotero_manager.py         # Zotero API 同步
│       ├── search_scholar.py         # Scholar 搜索（需API）
│       ├── search_serpapi.py         # SerpAPI 搜索
│       └── generate_summary.py       # 综述文本生成
│
└── paper-verifier/         # 真实性核查
    ├── SKILL.md
    └── scripts/
        └── verify_papers.py # 核心核查脚本
```

## 6. Verknüpfung mit der RAG-Wissensdatenbank

Die gesammelten echten Dokumente können nicht nur eine Rezension generieren, sondern auch in die **LLMRAG-Wissensdatenbank** für anschließende Fragen und Antworten und Schreibhilfe aufgenommen werden.

Der gesamte Verknüpfungsprozess:

```
手动搜索 → 收集到 JSON → 生成 Markdown 综述
    ↓
导入 Zotero（长期管理）
    ↓
添加到 LLMRA G知识库（向量检索）
    ↓
后续论文写作时，RAG 问答检索真实文献
```

Auf diese Weise ist Literaturrecherche keine einmalige Aufgabe, sondern ein Wissensschatz, der gesammelt, abgerufen und wiederverwendet werden kann.

## 7. Zusammenfassung der Designphilosophie

Das Design dieses Workflows beantwortet eine grundlegende Frage: **Wo liegen die Grenzen von LLM in der akademischen Forschung? **

| LLM ist gut in | LLM ist nicht gut darin (die Leute müssen kommen) |
|---------|--------|
| Informationen zusammenstellen und organisieren | Stellen Sie fest, ob das Dokument tatsächlich existiert |
| Strukturierten Text generieren | Überprüfen Sie die Genauigkeit des DOI/der Zitierzählung |
| Finden Sie Wissenslücken und Zusammenhänge | Durchsuchen Sie maßgebliche Datenbanken |
| Polieren und Umschreiben | Entscheiden, welche Arbeiten in die Rezension aufgenommen werden sollten |**Fazit**: Alle Links, die eine „Authentizitätsgarantie“ erfordern, werden von Menschen gehandhabt; LLM ist nur für die Sortierung und Generierung durch Menschen auf der Grundlage realer Eingaben verantwortlich.

Dies steht auch im Einklang mit der Lektion, die ich aus meinem vorherigen Vorfall gelernt habe: **Lassen Sie nicht zu, dass LLM etwas produziert, das nicht auf die Quelle zurückgeführt werden kann**. Alle Daten und jedes Dokument müssen nachvollziehbar sein.

## 8. Zukünftige Expansionsrichtungen

- [ ] Greifen Sie auf SerpAPI zu, um eine halbautomatische Google Scholar-Suche zu implementieren (manuelle Vorgänge reduzieren)
- [ ] Zwei-Wege-Synchronisierung mit der Zotero-API, um automatisch Metadaten gelesener Artikel abzurufen
- [ ] Erstellen Sie eine domänenspezifische Wissensdatenbank für Zeitschriften/Konferenzen mit integrierten Zitierformatvorlagen
- [ ] Verifizierte Dokumente automatisch in das Obsidian-Notizformat konvertieren und in das Double-Chain-Notiz-Ökosystem integrieren

---

*Skill-Pfade: „~/.openclaw/workspace/skills/paper-research/“ und „paper-verifier/“*

*Autor: Kagura Tart | 15.04.2026 | Geschrieben für mein zukünftiges Ich und alle Forscher, die eine Literaturrecherche durchführen müssen*