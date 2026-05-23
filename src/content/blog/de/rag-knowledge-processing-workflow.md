---
title: "RAG-Wissensdatenbank-Wissensverarbeitungsworkflow: eine Komplettlösung von der PDF-Analyse bis zur intelligenten Klassifizierung"
description: "Ausführliche Diskussion der drei Hauptaspekte der Wissensextraktion, -verarbeitung und -klassifizierung beim Aufbau der RAG-Wissensdatenbank, Analyse der anwendbaren Szenarien und Einschränkungen von LLM, Bewertung der Leistungsgrenzen externer Tools wie Claude Code CLI / Gemini CLI und Vorschlag einer Hybridarchitektur, die eine dedizierte Parsing-Bibliothek und einen Agenten-Workflow kombiniert"
pubDate: 2026-04-07T12:13:15+08:00
tags: ["LAPPEN", "Wissensbasis", "Wissensextraktion", "PDF-Analyse", "Claude Code", "Arbeitsablauf", "Agent", "LamaIndex", "LangChain"]
category: Tech
---

# RAG-Wissensdatenbank-Wissensverarbeitungsworkflow: eine Komplettlösung von der PDF-Analyse bis zur intelligenten Klassifizierung

> Beim Aufbau einer RAG-Wissensdatenbank ist das am meisten unterschätzte Glied nicht der Abrufalgorithmus, sondern die Verarbeitungsarbeit, bevor das Wissen in das System gelangt. Vom Hochladen einer PDF-Datei bis hin zur Umwandlung in ein qualitativ hochwertiges Vektorfragment sind mehr als zehn Schritte des Parsens, Bereinigens, der Strukturextraktion, der Entitätserkennung, der Klassifizierung, der Deduplizierung und der Qualitätsbewertung erforderlich. LLM ist möglich, aber möglicherweise nicht die beste Wahl. Dieser Artikel sortiert systematisch jeden Link in diesem Workflow, bewertet die Leistungsgrenzen von LLM und verschiedenen Spezialtools und stellt eine umsetzbare Hybridarchitektur und spezifische Implementierungsschritte bereit.

---

## 1. Globale Perspektive des Wissensverarbeitungsworkflows

### 1.1 Vollständiger Lebenszyklus der Wissensverarbeitung

Die Wissensverarbeitung der RAG-Wissensbasis ist in fünf Phasen unterteilt:

```
┌──────────────────────────────────────────────────────────────┐
│  阶段一：知识获取                                           │
│  PDF / Word / Markdown / 数据库 / 网页 / API                 │
└─────────────────────────────┬──────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  阶段二：知识解析                                           │
│  文字提取 · 表格解析 · 图表描述 · 层级结构识别              │
└─────────────────────────────┬──────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  阶段三：知识处理                                           │
│  清洗规范化 · 分块策略 · 实体抽取 · 关系抽取                │
└─────────────────────────────┬──────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  阶段四：知识分类                                          │
│  主题分类 · 层级分类 · 领域标签 · 质量评分                  │
└─────────────────────────────┬──────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  阶段五：知识索引                                          │
│  向量化 · 知识图谱 · 元数据索引 · 混合检索                   │
└──────────────────────────────────────────────────────────────┘
```

**Kernfrage**: Welche Links in diesen fünf Phasen sollten LLM verwenden und welche sollten spezielle Tools verwenden? Wo ist die Grenze zwischen den beiden?

### 1.2 Vergleich der Fähigkeiten von LLM mit Spezialwerkzeugen| Links | LLM-Anwendbarkeit | Spezialwerkzeuge | Empfohlene Lösungen |
|------|-----------|---------|---------|
| **Textextraktion** | ⚠️ Langsam/teuer/instabil | ✅ pdfminer / PyMuPDF / pdfplumber | **Spezialwerkzeuge** |
| **Tabellenanalyse** | ✅ Tabellensemantik verstehen | ✅ Camelot / Tabula / pdfplumber | **LLM + dedizierte doppelte Verifizierung** |
| **Diagrammbeschreibung** | ✅ Visuelles Verständnis | ⚠️ Erfordert multimodales Modell | **Multimodales LLM** |
| **Hierarchische Strukturerkennung** | ✅ Titelbeziehungen verstehen | ⚠️ Schwach | **LLM** |
| **Entitätsextraktion (NER)** | ✅ Wenige Proben/null Proben | ✅ spaCy / BERT-NER | **LLM (allgemein) / speziell (vertikales Feld)** |
| **Beziehungsextraktion** | ✅ Stark | ❌ Spezialwerkzeuge werden grundsätzlich nicht unterstützt | **LLM** |
| **Wissensklassifizierung** | ✅ Flexibel | ✅ Traditioneller ML-Klassifikator | **LLM (Low Data) / ML (Viele Daten) ** |
| **Qualitätsbewertung** | ✅ LLM-Selbsteinschätzung | ❌ Nicht anwendbar | **LLM** |
| **Duplikaterkennung** | ✅ Semantische Deduplizierung | ⚠️ Es kann nur eine wörtliche Deduplizierung durchgeführt werden | **LLM + MinHash** |

**Schlussfolgerung**: LLM eignet sich gut für **Verstehens-, Argumentations- und Generierungsaufgaben**, und Spezialwerkzeuge eignen sich gut für **Extraktion, Stabilität und schnelle** Aufgaben. Die beiden ergänzen sich, nicht entweder/oder.

---

## 2. Phase 1 und 2: Wissensanalyse – Warum LLM nicht ausreicht

### 2.1 Strukturelle Komplexität von PDF

PDF ist kein „strukturiertes Dokument“. Die unterste Ebene besteht aus einer Reihe von Zeichenanweisungen (Schriftart, Position, Farbe). Dies erschwert das Parsen von PDF-Dateien grundsätzlich:

- **Textrichtung**: vertikaler Text, gemischter Text (Chinesisch vertikal + Englisch horizontal)
- **Tabellenstruktur**: ungerahmte Tabellen, zeilen- und spaltenübergreifende Zellen, verschachtelte Tabellen
- **In Bilder eingebetteter Text**: Text in Scans und Screenshots kann nicht direkt extrahiert werden
- **Formeln**: LaTeX-Formeln, mathematische Symbole, chemische Formeln
- **Kopf- und Fußzeile**: Wiederholen Sie dies auf jeder Seite und müssen gefiltert werden
- **Fußnoten und Endnoten**: Anführungszeichen werden mit dem Haupttext verwechseltLLM kann PDFs nicht direkt verarbeiten, Text muss zuerst extrahiert werden. Aber reine Textextraktion ≠ hochwertige Inhaltswiederherstellung.

### 2.2 Ökologie dedizierter Analysetools

```python
# ============================================================
# 多层次 PDF 解析：完整解析管线
# ============================================================

class PDFParsePipeline:
    """
    PDF 解析完整管线：
    Layer 1: 文字提取（PyMuPDF + pdfplumber）
    Layer 2: 表格提取（双重策略：规则 + LLM 校验）
    Layer 3: 层级结构（LLM 重建）
    Layer 4: 图片 + 公式（多模态 LLM 处理）
    """
    
    def __init__(self, llm=None, ocr_enabled=True):
        self.llm = llm
        self.ocr_enabled = ocr_enabled
        self.results = []
    
    # ---- Layer 1: 文字提取 ----
    def extract_text_layer(self, pdf_path):
        """三层文字提取策略"""
        import pymupdf
        import pdfplumber
        
        pages_text = {}
        
        # 方法1: PyMuPDF（速度快，适合纯文字 PDF）
        doc = pymupdf.open(pdf_path)
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            pages_text[page_num] = {
                'text': text,
                'blocks': page.get_text("blocks"),
                'source': 'pymupdf'
            }
        
        # 方法2: pdfplumber（保留更多布局信息）
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                chars = page.chars  # 每个字符的详细信息
                lines = page.lines  # 表格线（若有）
                tables = page.extract_tables()
                
                # 合并两套提取结果，取更长的
                if len(chars) > len(pages_text[page_num]['text']):
                    words = page.extract_words()
                    combined_text = ' '.join([w['text'] for w in words])
                    pages_text[page_num]['text'] = combined_text
                    pages_text[page_num]['source'] = 'pdfplumber'
                
                pages_text[page_num]['tables'] = tables
                pages_text[page_num]['lines'] = lines
        
        return pages_text
    
    # ---- Layer 2: 表格提取（规则 + LLM 双重策略）----
    def extract_tables(self, page_tables, raw_text):
        """
        表格提取：先用 pdfplumber 规则提取，再用 LLM 校验
        关键问题：pdfplumber 对无线框表格效果差
        """
        verified_tables = []
        
        for i, table in enumerate(page_tables):
            if table is None:
                continue
            
            # 规则提取（pdfplumber）
            df_rule = self._table_to_dataframe(table)
            
            # LLM 校验（判断表格是否有效）
            if self.llm and len(df_rule) > 0:
                is_valid, corrected_df = self._llm_validate_table(
                    df_rule, raw_text
                )
                if is_valid:
                    verified_tables.append(corrected_df)
                else:
                    # 规则提取失败，尝试用 LLM 直接理解表格区域
                    llm_table = self._llm_extract_table_from_image(
                        raw_text, i
                    )
                    if llm_table is not None:
                        verified_tables.append(llm_table)
            else:
                verified_tables.append(df_rule)
        
        return verified_tables
    
    def _llm_validate_table(self, df, context_text):
        """用 LLM 校验表格提取结果"""
        if df is None or df.empty:
            return False, None
        
        table_md = df.to_markdown()
        
        prompt = f"""
请判断以下表格提取结果是否正确。参考上下文文本为：
---
{context_text[:500]}
---

表格内容：
{table_md}

请判断：
1. 表格是否有意义（不是随机碎片）
2. 表格结构和内容是否与上下文一致
3. 如果有问题，请直接修复

输出 JSON：
{{"is_valid": true/false, "corrected_table": "markdown表格或null"}}
"""
        
        response = self.llm.invoke(prompt)
        result = json.loads(response.content)
        return result['is_valid'], result.get('corrected_table')
    
    # ---- Layer 3: 层级结构重建 ----
    def rebuild_hierarchy(self, pages_text):
        """
        用 LLM 重建文档层级结构（标题 → 章节 → 段落）
        这步 LLM 非常擅长，但也可以用规则（字体大小 + 关键词）
        """
        if not self.llm:
            return self._rule_based_hierarchy(pages_text)
        
        all_text = '\n\n'.join([
            f"[Page {p}] {data['text']}"
            for p, data in sorted(pages_text.items())
        ])
        
        prompt = f"""
请分析以下文档，识别其层级结构。输出一个结构化的大纲：

文档内容：
---
{all_text[:8000]}
---

请按以下格式输出（JSON）：
{{
  "title": "文档标题",
  "sections": [
    {{"level": 1, "title": "一级标题", "page": 页码, "content_summary": "本节内容摘要"}},
    {{"level": 2, "title": "二级标题", "page": 页码, "content_summary": "..."}}
  ]
}}
"""
        response = self.llm.invoke(prompt)
        return json.loads(response.content)
    
    def _rule_based_hierarchy(self, pages_text):
        """无 LLM 时的规则方法（基于字体大小和关键词）"""
        sections = []
        import re
        
        for page_num, data in sorted(pages_text.items()):
            text = data['text']
            blocks = data.get('blocks', [])
            
            for block in blocks:
                if block.get('type') != 0:  # 只处理文字块
                    continue
                
                block_text = block.get('text', '').strip()
                font_size = block.get('size', 12)
                
                # 字体大小 > 16 视为标题
                if font_size > 16 and len(block_text) < 100:
                    level = 1 if font_size > 20 else 2
                    sections.append({
                        'level': level,
                        'title': block_text,
                        'page': page_num
                    })
        
        return {'sections': sections}
```

---

## 3. Phase 3: Wissensverarbeitung – LLMs Stärke, erfordert aber Strategie

### 3.1 Entitätsextraktion und Beziehungsextraktion

Entitätsextraktion (NER) und Beziehungsextraktion sind der Kern der Wissensverarbeitung. LLM zeichnet sich in diesem Bereich aus:

```python
# ============================================================
# LLM 驱动的实体 + 关系抽取
# ============================================================

class EntityRelationExtractor:
    """
    基于 LLM 的实体和关系抽取
    相比传统 NER 的优势：
    - 零样本 / 少样本，无需标注数据
    - 支持复杂关系（嵌套、多跳）
    - 可自定义 Schema
    """
    
    def __init__(self, llm, schema):
        self.llm = llm
        self.schema = schema  # 定义抽取的实体类型和关系类型
    
    def extract(self, text, schema_type='default'):
        """
        一次性抽取实体和关系
        schema_type: 领域适配的 Schema
        """
        schemas = {
            'default': {
                'entities': ['人物', '组织', '地点', '时间', '事件'],
                'relations': ['属于', '位于', '创建', '相关于']
            },
            'technical': {
                'entities': ['技术名称', '方法', '工具', '指标', '版本'],
                'relations': ['基于', '使用', '优于', '依赖']
            },
            'legal': {
                'entities': ['当事人', '法条', '案件', '法院'],
                'relations': ['起诉', '判决', '依据', '涉及']
            }
        }
        
        s = schemas.get(schema_type, schemas['default'])
        
        prompt = f"""
你是一个专业的实体和关系抽取系统。请从以下文本中抽取实体和关系。

实体类型：{', '.join(s['entities'])}
关系类型：{', '.join(s['relations'])}

文本：
---
{text}
---

输出 JSON 格式：
{{
  "entities": [
    {{"name": "实体名", "type": "实体类型", "description": "简要描述"}}
  ],
  "relations": [
    {{"source": "实体A", "target": "实体B", "relation": "关系类型", "confidence": 0.0-1.0}}
  ]
}}

注意：
1. 只抽取文本中明确提到的实体
2. 关系必须是明确的，包含两个端点
3. 所有字段均需填写，不可为空
"""
        
        response = self.llm.invoke(prompt)
        try:
            result = json.loads(response.content)
            return result['entities'], result['relations']
        except:
            return [], []
    
    def batch_extract(self, chunks, schema_type='default', batch_size=10):
        """
        批量抽取（对 chunks 分批调用 LLM）
        关键优化：不是每个 chunk 都调用，优先对高信息量 chunk 抽取
        """
        results = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            
            # 构建批量 prompt
            combined_text = '\n---\n'.join([
                f"[Chunk {i+j}] {chunk['text']}"
                for j, chunk in enumerate(batch)
            ])
            
            prompt = f"""
请从以下多个文本片段中批量抽取实体和关系。

实体类型：{schemas[schema_type]['entities']}
关系类型：{schemas[schema_type]['relations']}

内容：
---
{combined_text[:10000]}
---

输出 JSON：
{{
  "extractions": [
    {{
      "chunk_id": 0,
      "entities": [...],
      "relations": [...]
    }},
    ...
  ]
}}
"""
            
            response = self.llm.invoke(prompt)
            try:
                data = json.loads(response.content)
                results.extend(data.get('extractions', []))
            except:
                results.extend([{'chunk_id': j, 'entities': [], 'relations': []} 
                               for j in range(len(batch))])
        
        return results
```

### 3.2 Einschränkungen des LLM für die Wissensverarbeitung

LLM weist in der Phase der Wissensverarbeitung klare Einschränkungen auf, die die Entwicklung von Strategien zur Bewältigung folgender Themen erfordern:

| Einschränkungen | Leistung | Bewältigungsstrategien |
|--------|------|---------|
| **Langsam** | Einzelner API-Aufruf 1–5 Sekunden | Asynchrone Parallelität + Stapelverarbeitung |
| **Hohe Kosten** | Die Kosten für den Aufruf der API für eine große Anzahl von Blöcken explodieren | Priorisieren Sie Aufrufe von Schlüsselblöcken und verwenden Sie Regeln für allgemeine Blöcke |
| **Instabile Ausgabe** | Die Ergebnisse des mehrmaligen Aufrufs desselben Textes können unterschiedlich sein | Wenige Eingabeaufforderung + Ausgabeschema-Einschränkungen |
| **Illusion** | Möglicherweise fiktive Entitäten, die nicht existieren | Konfidenzfilterung + Originaltext-Rückprüfung |
| **Kontextfenster** | Abschneiden extrem langer Dokumente | Schiebefenster-Segmentierungsverarbeitung |
| **Domänenanpassung** | Allgemeines LLM ist bei der Erkennung vertikaler Domänenentitäten schwach | LoRA-Feinabstimmung oder RAG-Verbesserung |

---

## 4. Stufe 4: Wissensklassifizierung – LLM vs. spezialisiertes Modell

### 4.1 Auswahl der Klassifizierungsstrategie

```python
# ============================================================
# 知识分类：三层策略（从快到准）
# ============================================================

class KnowledgeClassifier:
    """
    三层分类策略：
    Layer 1: 规则匹配（最快，关键词命中）
    Layer 2: Embedding 相似度（中等速度，无需训练）
    Layer 3: LLM 推理（最准，但最慢）
    """
    
    def __init__(self, llm, embed_model, taxonomy):
        self.llm = llm
        self.embed_model = embed_model
        self.taxonomy = taxonomy  # taxonomy: {'技术': [...], '产品': [...], ...}
        
        # Layer 2: 预计算 taxonomy 的 embedding
        self.taxonomy_embeddings = {}
        for category, keywords in taxonomy.items():
            self.taxonomy_embeddings[category] = embed_model.encode(keywords)
    
    def classify(self, chunk, strategy='cascade'):
        """
        级联分类策略：从快到准逐层尝试
        """
        if strategy == 'rule':
            return self._rule_classify(chunk)
        elif strategy == 'embedding':
            return self._embedding_classify(chunk)
        elif strategy == 'llm':
            return self._llm_classify(chunk)
        else:
            # 默认：级联
            return self._cascade_classify(chunk)
    
    def _cascade_classify(self, chunk):
        """级联：先用规则，规则不行用 Embedding，Embedding 不行用 LLM"""
        # Layer 1: 规则
        rule_result = self._rule_classify(chunk)
        if rule_result['confidence'] > 0.9:
            return rule_result
        
        # Layer 2: Embedding 相似度
        emb_result = self._embedding_classify(chunk)
        if emb_result['confidence'] > 0.7:
            return emb_result
        
        # Layer 3: LLM（最慢但最准）
        return self._llm_classify(chunk)
    
    def _rule_classify(self, chunk):
        """Layer 1: 关键词规则（毫秒级）"""
        text_lower = chunk['text'].lower()
        scores = {}
        
        for category, keywords in self.taxonomy.items():
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            if score > 0:
                scores[category] = score / len(keywords)
        
        if scores:
            best_cat = max(scores, key=scores.get)
            return {
                'category': best_cat,
                'confidence': scores[best_cat],
                'method': 'rule'
            }
        
        return {'category': 'unknown', 'confidence': 0.0, 'method': 'rule'}
    
    def _embedding_classify(self, chunk):
        """Layer 2: Embedding 相似度（快速，无需 LLM 调用）"""
        chunk_emb = self.embed_model.encode([chunk['text']])[0]
        
        best_cat = None
        best_score = -1
        
        for category, cat_embs in self.taxonomy_embeddings.items():
            similarities = np.dot(chunk_emb, cat_embs.T)
            max_sim = similarities.max()
            
            if max_sim > best_score:
                best_score = max_sim
                best_cat = category
        
        return {
            'category': best_cat or 'unknown',
            'confidence': float(best_score) if best_score > 0 else 0.0,
            'method': 'embedding'
        }
    
    def _llm_classify(self, chunk):
        """Layer 3: LLM 推理（最准确）"""
        prompt = f"""
请将以下文本分类到最合适的类别。

可选类别：{', '.join(self.taxonomy.keys())}

文本内容：
---
{chunk['text'][:2000]}
---

请输出 JSON：
{{"category": "类别名", "confidence": 0.0-1.0, "reasoning": "分类理由"}}
"""
        
        response = self.llm.invoke(prompt)
        result = json.loads(response.content)
        result['method'] = 'llm'
        return result
```

---

## 5. Eingehende Bewertung externer Agententools

### 5.1 Claude Code CLI – ein unerwartetes Tool zum Aufbau von Wissensdatenbanken

Claude Code CLI („claude-code“) ist nicht nur ein Tool zur Codegenerierung, seine **Read-Browse-Analyze-Funktion** kann in Wissensdatenbank-Workflows verwendet werden:

```bash
# 安装 Claude Code CLI
npm install -g @anthropic-ai/claude-code

# 认证
claude-code auth
```

**Verwendung von Claude Code in der Wissensverarbeitung**:

```python
# ============================================================
# Claude Code CLI 作为知识处理 Agent
# ============================================================

class ClaudeCodeAgent:
    """
    将 Claude Code CLI 作为通用知识处理 Agent
    优势：
    - 原生多步推理（Think 模式）
    - 内置文件读写 + 搜索能力
    - 可执行 Python / Shell 脚本处理数据
    """
    
    def __init__(self, model='sonnet'):
        self.model = model
    
    def process_pdf(self, pdf_path, task='extract_knowledge'):
        """
        使用 Claude Code 执行 PDF 知识提取任务
        """
        task_prompts = {
            'extract_knowledge': f"""
请读取文件 {pdf_path}，完成以下知识提取任务：

1. 识别文档类型（论文/报告/手册/合同等）
2. 提取所有实体（人名/组织/技术术语/关键概念）
3. 识别文档结构（章节标题、层级关系）
4. 提取关键信息（主要观点、方法、数据、结论）
5. 生成 200 字以内的摘要

输出格式：结构化 JSON
""",
            'quality_review': """
请审查这段文本的质量：

1. 是否存在事实错误或矛盾？
2. 是否有重要信息缺失？
3. 是否包含过时或不确定的信息？
4. 给出 0-10 的质量评分

输出格式：JSON
""",
            'table_understand': """
请理解以下表格的含义，并描述：
1. 表格的主题是什么？
2. 各列/行各代表什么？
3. 最关键的发现是什么？

输出格式：结构化描述
"""
        }
        
        task_prompt = task_prompts.get(task, task_prompts['extract_knowledge'])
        
        # 通过 CLI 执行
        cmd = f'''
claude-code --model {self.model} -p "{task_prompt}" --output-format json
'''
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        try:
            return json.loads(result.stdout)
        except:
            return {'raw': result.stdout, 'error': 'parse failed'}
    
    def batch_process(self, file_list, task='extract_knowledge'):
        """
        批量处理多个文件（并行 Claude Code 调用）
        """
        from concurrent.futures import ThreadPoolExecutor
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self.process_pdf, f, task): f
                for f in file_list
            }
            results = {}
            for future in as_completed(futures):
                f = futures[future]
                results[f] = future.result()
        
        return results
    
    def interactive_analysis(self, text, question):
        """
        交互式分析：针对一段文本提问
        """
        prompt = f"""
请分析以下文本并回答问题：

文本：
---
{text[:8000]}
---

问题：{question}

请给出详细、准确的回答。如果文本信息不足以回答，请明确说明。
"""
        
        cmd = f'claude-code -p "{prompt}" --model {self.model}'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout
```

### 5.2 Gemini CLI – Das Potenzial der multimodalen PDF-VerarbeitungGoogles **Gemini CLI** (über „gemini“ SDK oder AI Studio) unterstützt Multimodalität und bietet einzigartige Vorteile bei der Verarbeitung von PDFs mit Diagrammen:

```python
class GeminiMultimodalProcessor:
    """
    Gemini CLI 多模态处理 PDF
    核心能力：
    - 直接读取 PDF（含图片页）
    - 图表理解 + 描述生成
    - 跨页理解（表格跨页问题）
    """
    
    def __init__(self, api_key=None):
        import google.generativeai as genai
        genai.configure(api_key=api_key or os.environ.get('GOOGLE_API_KEY'))
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    
    def process_pdf_page(self, page_image_bytes):
        """
        直接将 PDF 页面作为图片输入 Gemini
        适合：扫描件、含图表页面、多列复杂排版
        """
        import PIL.Image
        import io
        
        img = PIL.Image.open(io.BytesIO(page_image_bytes))
        
        prompt = """
请分析这个 PDF 页面，完成以下任务：

1. 文字识别：提取所有文字内容，保持原有结构
2. 表格理解：描述表格内容，提取关键数据
3. 图表描述：描述图表类型（柱状图/折线图/流程图等）和关键信息
4. 结构识别：判断页面类型（正文/目录/参考文献/封面等）

输出 JSON 格式：
{
  "page_type": "页面类型",
  "text": "提取的文字",
  "tables": [{"description": "表格描述", "data": [[...]]}],
  "figures": [{"type": "类型", "description": "描述", "key_findings": ["关键发现"]}],
  "confidence": 0.0-1.0
}
"""
        
        response = self.model.generate_content([img, prompt])
        return json.loads(response.text)
    
    def extract_figure_descriptions(self, pdf_path):
        """
        从 PDF 中提取所有图片页，用 Gemini 生成描述
        用于：生成图片的替代文本（Alt-text），增强检索
        """
        import pymupdf
        
        doc = pymupdf.open(pdf_path)
        figure_descriptions = []
        
        for page_num, page in enumerate(doc):
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                pix = pymupdf.Pixmap(doc, xref)
                
                if pix.n > 4:  # CMYK 或其他，转换为 RGB
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                
                img_bytes = pix.tobytes("png")
                
                # Gemini 描述
                desc = self.process_pdf_page(img_bytes)
                figure_descriptions.append({
                    'page': page_num,
                    'image_index': img_index,
                    'description': desc['figures'][0]['description'] if desc.get('figures') else ''
                })
        
        return figure_descriptions
```

### 5.3 Mehrwert von Sonderwerkzeugen

| Werkzeuge | Anwendbare Szenarien | Machbarkeit einer LLM-Alternative | Gründe für die Empfehlung |
|------|---------|--------------|---------|
| **PyMuPDF** | Schnelle Text-/Bildextraktion | ❌ LLM kann die Extraktion nicht ersetzen | Schnell und stabil |
| **pdfplumber** | Tischauszug (Drahtgestell) | ⚠️ LLM kann bei der Verifizierung helfen | Die Regelextraktion ist zuverlässiger |
| **Camelot** | Komplexe Tabellen (keine Frames) | ⚠️ Der LLM-Verifizierungseffekt ist besser | Spezieller Algorithmus ist genauer |
| **Pytesserakt** | OCR-Texterkennung | ⚠️ Multimodales LLM austauschbar | Erfordert Trainingsdaten |
| **MarkItDown** | Dokument in Markdown konvertieren | ⚠️ LLM kann die Struktur rekonstruieren | Die Konvertierung mit einem Klick ist die einfachste |
| **Unstrukturiert.io** | Unternehmensdokumente (gemischtes Format) | ⚠️ LLM für die Nachbearbeitung | Die beste Bibliothek zum Parsen von Unternehmensdokumenten |
| **Nougat** | LaTeX-Formel + akademisches PDF | ✅ Multimodale LLM-Alternative | Nur akademisches PDF |
| **MinerU** (Alibaba) | Komplexes chinesisches PDF | ✅ Unter Berücksichtigung von Geschwindigkeit und Genauigkeit | Die erste Wahl für chinesische Dokumente |

---

## 6. Optimaler Workflow: Hybrid-Architekturdesign

### 6.1 Grundlegende Designprinzipien

**Grundsatz 1: LLM sollte nur für das eingesetzt werden, was es am besten kann**
- Entitäts-/Beziehungsextraktion → LLM
- Hierarchierekonstruktion → LLM
- Qualitätsbewertung → LLM
- Diagrammbeschreibung → Multimodales LLM

**Prinzip 2: Stabile und schnelle Aufgaben Spezialwerkzeugen überlassen**
- Textextraktion → PyMuPDF/pdfplumber
- OCR → Pytesseract/EasyOCR
- Chunking → Regeln + Überlappende Schiebefenster

**Prinzip 3: Stapelverarbeitung statt Einzelaufrufe**
- Die LLM-Anrufkosten sind hoch und werden in Stapeln verpackt (10–50 Blöcke pro Anfrage).**Prinzip 4: Asynchrone Parallelität**
- LLM-Aufrufe sind asynchron und können von speziellen Tools synchron und schnell verarbeitet werden

### 6.2 Vollständiger Hybrid-Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    混合知识处理工作流                            │
│                                                                 │
│  [1. 文档摄入]                                                  │
│     ↓                                                          │
│  [2. 路由判断]                                                  │
│     ├── 扫描件/图片PDF  ──→  OCR (Pytesseract)                │
│     ├── 普通文字PDF  ──→  PyMuPDF 文字提取                      │
│     ├── 企业文档(PDF/Word) ──→  Unstructured.io                │
│     └── 学术PDF  ──→  Nougat + PyMuPDF                         │
│     ↓                                                          │
│  [3. 表格双重处理]                                              │
│     pdfplumber规则提取 ──→ LLM校验 ──→ 修正/保留               │
│     ↓                                                          │
│  [4. 图表多模态处理]                                            │
│     Gemini多模态页 ──→ 图表描述生成 ──→  Alt-text存入元数据    │
│     ↓                                                          │
│  [5. 层级结构重建]                                              │
│     字体大小规则 + LLM语义 ──→ 章节树 ──→ 指导分块               │
│     ↓                                                          │
│  [6. 智能分块]                                                 │
│     按章节/段落分块 + 重叠 + Chunk质量评估                       │
│     ↓                                                          │
│  [7. 知识抽取（LLM批量）]                                       │
│     批量实体+关系抽取 ──→ 置信度过滤 ──→ 知识图谱                │
│     ↓                                                          │
│  [8. 分类（级联）]                                              │
│     规则 → Embedding → LLM（逐层升级）                         │
│     ↓                                                          │
│  [9. 去重检测]                                                  │
│     MinHash 字面去重 + Embedding 语义去重                       │
│     ↓                                                          │
│  [10. 质量评分 + 索引]                                          │
│      LLM自评质量 + 向量化 + 知识图谱 + 元数据索引                │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 Vollständiger Implementierungscode

```python
# ============================================================
# 完整知识处理工作流（混合架构）
# ============================================================

import asyncio
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
from pathlib import Path

@dataclass
class ProcessedChunk:
    """处理后的知识片段"""
    chunk_id: str
    content: str
    source: str
    page_number: int
    chunk_type: str           # 'text' / 'table' / 'figure' / 'heading'
    entities: List[Dict]
    relations: List[Dict]
    categories: List[str]
    quality_score: float
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None


class RAGKnowledgePipeline:
    """
    RAG 知识库处理完整工作流
    设计理念：LLM做理解，专用工具做提取，规则做路由
    """
    
    def __init__(
        self,
        llm,                      # LLM 实例（Claude / GPT-4 / 本地模型）
        embed_model,               # Embedding 模型
        vector_store,              # 向量数据库
        graph_db=None,             # 知识图谱（可选）
        config=None
    ):
        self.llm = llm
        self.embed_model = embed_model
        self.vector_store = vector_store
        self.graph_db = graph_db
        self.config = config or {}
        
        # 初始化专用解析器
        self.pdf_parser = PDFParsePipeline(llm=llm)
        self.extractor = EntityRelationExtractor(llm, schema=None)
        self.classifier = KnowledgeClassifier(llm, embed_model, self._default_taxonomy())
    
    def _default_taxonomy(self):
        return {
            '技术文档': ['算法', '系统', '架构', '模块', '接口', '协议'],
            '产品规格': ['型号', '参数', '规格', '尺寸', '性能', '指标'],
            '政策法规': ['规定', '要求', '标准', '法规', '条例', '政策'],
            '研究报告': ['分析', '调研', '数据', '趋势', '市场', '预测'],
            '操作手册': ['步骤', '指南', '说明', '操作', '维护', '使用'],
        }
    
    async def process_document(self, file_path: str) -> List[ProcessedChunk]:
        """
        异步处理单个文档（主要接口）
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        # Step 1: 路由 + 提取
        if suffix == '.pdf':
            raw_data = await self._process_pdf(file_path)
        elif suffix in ['.docx', '.doc']:
            raw_data = await self._process_docx(file_path)
        elif suffix == '.md':
            raw_data = self._process_markdown(file_path)
        else:
            raw_data = self._process_text(file_path)
        
        # Step 2: 分块
        chunks = self._chunk_text(raw_data)
        
        # Step 3: 批量实体 + 关系抽取（LLM）
        extractions = await self._batch_extract(chunks)
        
        # Step 4: 分类
        categorized = await self._batch_classify(chunks)
        
        # Step 5: 去重
        deduped = self._deduplicate(chunks)
        
        # Step 6: 质量评分
        scored = await self._batch_quality_score(deduped)
        
        # Step 7: 索引
        await self._index_chunks(scored, extractions)
        
        return scored
    
    async def _process_pdf(self, path):
        """PDF 处理"""
        pages_text = self.pdf_parser.extract_text_layer(str(path))
        
        # 表格处理
        for page_num, data in pages_text.items():
            tables = self.pdf_parser.extract_tables(
                data.get('tables', []), 
                data['text']
            )
            data['verified_tables'] = tables
        
        # 层级重建
        hierarchy = self.pdf_parser.rebuild_hierarchy(pages_text)
        
        return {
            'type': 'pdf',
            'pages': pages_text,
            'hierarchy': hierarchy,
            'full_text': '\n\n'.join([
                d['text'] for d in pages_text.values()
            ])
        }
    
    async def _process_docx(self, path):
        """Word 文档处理（Unstructured.io）"""
        from unstructured.partition.auto import partition
        
        elements = partition(filename=str(path))
        
        text_blocks = []
        tables = []
        for el in elements:
            if el.category == 'UnstructuredText':
                text_blocks.append(el.text)
            elif el.category == 'Table':
                tables.append(el.text)
        
        return {
            'type': 'docx',
            'elements': elements,
            'text_blocks': text_blocks,
            'tables': tables,
            'full_text': '\n\n'.join(text_blocks)
        }
    
    def _process_markdown(self, path):
        """Markdown 处理（最简单）"""
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            'type': 'md',
            'full_text': content,
            'sections': self._split_markdown_sections(content)
        }
    
    def _chunk_text(self, raw_data, chunk_size=500, overlap=50):
        """
        智能分块策略：
        - 优先按语义单元（段落/章节）分块
        - 超长段落内部分块
        - 保留重叠确保上下文连续
        """
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        
        chunks = []
        
        if raw_data['type'] == 'md':
            sections = raw_data.get('sections', [])
            for sec in sections:
                text = sec['content']
                tokens = enc.encode(text)
                
                for start in range(0, len(tokens), chunk_size - overlap):
                    chunk_tokens = tokens[start:start + chunk_size]
                    chunk_text = enc.decode(chunk_tokens)
                    
                    if len(chunk_text.strip()) < 50:  # 太短的 chunk 跳过
                        continue
                    
                    chunks.append({
                        'text': chunk_text,
                        'section': sec.get('title', ''),
                        'chunk_type': 'text'
                    })
        else:
            # 通用分块
            text = raw_data['full_text']
            tokens = enc.encode(text)
            
            for start in range(0, len(tokens), chunk_size - overlap):
                chunk_tokens = tokens[start:start + chunk_size]
                chunk_text = enc.decode(chunk_tokens)
                
                if len(chunk_text.strip()) < 50:
                    continue
                
                chunks.append({
                    'text': chunk_text,
                    'section': '',
                    'chunk_type': 'text'
                })
        
        # 分配 chunk_id
        for i, chunk in enumerate(chunks):
            chunk['chunk_id'] = hashlib.md5(
                (chunk['text'][:100] + str(i)).encode()
            ).hexdigest()[:12]
        
        return chunks
    
    async def _batch_extract(self, chunks):
        """批量实体+关系抽取（LLM并发）"""
        batch_size = 10
        all_results = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            
            # 构造批量 prompt
            combined = '\n\n'.join([
                f"[{j}] {c['text'][:500]}"
                for j, c in enumerate(batch)
            ])
            
            prompt = f"""
从以下文本片段中批量抽取实体和关系。

类型：['人物', '组织', '地点', '技术术语', '时间', '事件']
关系：['属于', '创建', '使用', '相关于', '优于']

文本：
---
{combined[:8000]}
---

输出 JSON：
{{
  "extractions": [
    {{"chunk_id": 0, "entities": [], "relations": []}},
    ...
  ]
}}
"""
            
            response = await self.llm.agenerate(prompt)  # 异步调用
            try:
                data = json.loads(response)
                all_results.extend(data.get('extractions', []))
            except:
                all_results.extend([{'entities': [], 'relations': []} 
                                   for _ in batch])
        
        return all_results
    
    async def _batch_classify(self, chunks):
        """批量分类（级联策略）"""
        tasks = []
        for chunk in chunks:
            cat = self.classifier.classify(chunk, strategy='cascade')
            chunk['categories'] = [cat['category']]
            chunk['category_confidence'] = cat['confidence']
            chunk['category_method'] = cat['method']
            tasks.append(chunk)
        
        return tasks
    
    def _deduplicate(self, chunks):
        """两层去重：MinHash + Embedding"""
        from datasketch import MinHash, MinHashLSH
        
        # Layer 1: MinHash 字面去重
        lsh = MinHashLSH(threshold=0.8)
        seen = {}
        
        for chunk in chunks:
            m = MinHash()
            for token in chunk['text'][:500].split():
                m.update(token.encode('utf8'))
            
            matches = lsh.query(m)
            if matches:
                # 字面相似，保留较长的
                existing_id = matches[0]
                if len(chunk['text']) > len(seen[existing_id]['text']):
                    seen[existing_id] = chunk
            else:
                lsh.insert(chunk['chunk_id'], m)
                seen[chunk['chunk_id']] = chunk
        
        # Layer 2: Embedding 语义去重
        remaining = list(seen.values())
        if len(remaining) < 2:
            return remaining
        
        texts = [c['text'] for c in remaining]
        embs = self.embed_model.encode(texts)
        
        keep = []
        seen_embs = []
        
        for chunk, emb in zip(remaining, embs):
            is_dup = False
            for seen_emb in seen_embs:
                sim = np.dot(emb, seen_emb)  # 已归一化
                if sim > 0.92:  # 语义重复阈值
                    is_dup = True
                    break
            
            if not is_dup:
                keep.append(chunk)
                seen_embs.append(emb)
        
        return keep
    
    async def _batch_quality_score(self, chunks):
        """LLM 自评质量"""
        for chunk in chunks:
            if len(chunk['text']) < 100:
                chunk['quality_score'] = 0.5
                continue
            
            prompt = f"""
请评估以下文本片段的质量，给出 0-10 分的质量评分。

质量标准：
- 信息完整性（是否包含完整信息）
- 语义清晰度（是否有意义、连贯）
- 独立可读性（脱离上下文是否可理解）
- 信息价值（是否包含实质性内容）

文本：
---
{chunk['text'][:1000]}
---

输出 JSON：{{"quality_score": 0.0-10.0}}
"""
            
            try:
                response = await self.llm.agenerate(prompt)
                result = json.loads(response)
                chunk['quality_score'] = result.get('quality_score', 5.0) / 10.0
            except:
                chunk['quality_score'] = 0.5
        
        return chunks
    
    async def _index_chunks(self, chunks, extractions):
        """索引到向量数据库 + 知识图谱"""
        for chunk, ext in zip(chunks, extractions):
            # 向量
            emb = self.embed_model.encode([chunk['text']])[0]
            
            self.vector_store.insert(
                chunk_id=chunk['chunk_id'],
                content=chunk['content'],
                embedding=emb,
                metadata={
                    'categories': chunk.get('categories', []),
                    'quality_score': chunk.get('quality_score', 0.5),
                    'chunk_type': chunk.get('chunk_type', 'text'),
                    'entities': ext.get('entities', []),
                    'relations': ext.get('relations', []),
                }
            )
            
            # 知识图谱（Neo4j）
            if self.graph_db:
                for entity in ext.get('entities', []):
                    self.graph_db.create_entity(
                        name=entity['name'],
                        type=entity['type'],
                        chunk_id=chunk['chunk_id']
                    )
                for rel in ext.get('relations', []):
                    self.graph_db.create_relation(
                        from_entity=rel['source'],
                        to_entity=rel['target'],
                        relation=rel['relation']
                    )
    
    def process_corpus(self, file_dir: str, max_workers=4):
        """
        并行处理整个文档目录
        """
        files = list(Path(file_dir).glob('**/*'))
        files = [f for f in files if f.suffix.lower() 
                in ['.pdf', '.docx', '.md', '.txt']]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(asyncio.run, self.process_document(str(f))): f
                for f in files
            }
            
            all_chunks = []
            for future in as_completed(futures):
                f = futures[future]
                try:
                    chunks = future.result()
                    all_chunks.extend(chunks)
                    print(f"✅ {f.name}: {len(chunks)} chunks")
                except Exception as e:
                    print(f"❌ {f.name}: {e}")
        
        return all_chunks
```

---

## 7. Spezifische Umsetzungsschritte

### Schritt 1: Umgebungsvorbereitung (Tag 1)

```bash
# 核心依赖
pip install pymupdf pdfplumber tiktoken datasketch sentence-transformers
pip install pymupdf  # PDF 文字提取
pip install pdfplumber  # PDF 表格提取
pip install unstructured  # 企业文档解析
pip install anthropic openai  # LLM API

# OCR（扫描件处理）
pip install pytesseract EasyOCR

# 知识图谱（可选）
pip install neo4j python-nemo

# Claude Code CLI（可选，用于复杂分析任务）
npm install -g @anthropic-ai/claude-code
```

### Schritt 2: Schnelle Baseline (Tag 1-2)

Gehen Sie zuerst den einfachsten Prozess durch und streben Sie nicht nach Perfektion:

```python
# 最简工作流（不含 LLM 实体抽取）
def quick_pipeline(pdf_path):
    # 1. 文字提取（专用工具）
    doc = pymupdf.open(pdf_path)
    text = '\n'.join([p.get_text() for p in doc])
    
    # 2. 规则分块
    chunks = recursive_chunk(text, chunk_size=500)
    
    # 3. Embedding 索引
    embeddings = embed_model.encode([c['text'] for c in chunks])
    vector_store.insert(chunks, embeddings)
    
    return chunks
```

### Schritt 3: Bewerten Sie die Basisqualität (Tag 2)

- Erstellen Sie einen Testsatz: 50–100 Abfragen, kennzeichnen Sie die erwarteten Antworten manuell
- Bewertungsindikatoren: RAGAS (Faithfulness/Answer Relevance/Context Precision)
- Lokalisieren Sie den Engpass: Handelt es sich um ein Parsing-Problem, ein Chunking-Problem oder ein Abrufproblem?

### Schritt 4: Gezielte Optimierung (Tag 3-7)

Priorisieren Sie basierend auf den Bewertungsergebnissen die Lösung der größten Engpässe:

| Engpass | Optimierungsplan |
|------|---------|
| Tabelleninformationen gehen verloren | PDFplumber-Regelextraktion + LLM-Überprüfung hinzufügen |
| Chunking Breaking Semantik | Verbesserte Chunking-Strategie (nach Kapitel/mehr Überlappung) |
| Geringe Sucherinnerung | Wissensgraph + BM25-Hybridsuche hinzufügen |
| Die Entitätserkennung ist ungenau | Verwenden Sie LLM für die Entitätsextraktion + KG-Index |
| Klassifizierungsfehler | Domänentaxonomie + LLM-Klassifizierung hinzufügen |

### Schritt 5: Integration der LLM-Wissensextraktion (Woche 2)

```python
# 集成实体+关系抽取
async def step5_entity_extraction(chunks):
    extractor = EntityRelationExtractor(llm=claude_llm, schema=domain_schema)
    
    # 批量抽取（异步并发）
    results = await extractor.batch_extract(chunks, batch_size=20)
    
    # 置信度过滤
    filtered = [r for r in results if r['confidence'] > 0.7]
    
    # 存入知识图谱
    await kg_graph.bulk_insert(filtered)
    
    return filtered
```

### Schritt 6: Kontinuierliche Optimierung und Überwachung (Woche 3+)

- **Periodische Bewertung**: Jede Woche wird eine zufällige Abfrage durchgeführt, um die RAG-Qualität zu bewerten
- **Feedback-Schwungrad**: Sammeln Sie Benutzerfeedback zu Antworten und fügen Sie automatisch Trainingsdaten hinzu
- **Wissensaktualisierung**: inkrementelle Verarbeitung neuer Dokumente ohne vollständige Rekonstruktion

---

## 8. Entscheidungsbaum für die Werkzeugauswahl

```
输入文档类型？
├── 扫描件/图片 PDF
│   ├── 优先：EasyOCR / Pytesseract（速度优先）
│   └── 高质量：Gemini 多模态处理（精度优先）
│
├── 普通文字 PDF
│   ├── 学术论文 → Nougat（公式）+ PyMuPDF（文字）
│   ├── 企业报告 → Unstructured.io（混合内容）
│   └── 中文文档 → MinerU（中文优化）
│
├── Word / Excel / PPT
│   └── Unstructured.io（统一处理所有 Office 格式）
│
└── Markdown / HTML
    └── 直接解析（无需特殊工具）

处理目标？
├── 表格 → pdfplumber（规则）+ LLM 校验
├── 图表 → Gemini 多模态（描述生成）
├── 公式 → Nougat / Mathpix（LaTeX 转换）
└── 纯文字 → PyMuPDF / pdfplumber 即可

是否需要实体关系？
├── 是
│   ├── 通用领域 → LLM Few-shot NER
│   └── 垂直领域 → 专用 NER 模型（+ LLM 校验）
└── 否
    └── 跳过，直接 Embedding 索引
```

---

## 9. Zusammenfassung: Best Practices für die Werkzeugkombination

Der Aufbau einer hochwertigen RAG-Wissensdatenbank kann nicht mit einem einzigen Tool gelöst werden. Die beste Vorgehensweise ist die **dreistufige Tool-Zusammenarbeit**:| Hierarchie | Werkzeuge | Verantwortlichkeiten |
|------|------|------|
| **Extraktionsebene** | PyMuPDF/pdfplumber/Unstructured/Nougat | Schnelle und stabile Extraktion des Originalinhalts |
| **Verstehensebene** | Claude / GPT-4 / Zwillinge | Semantik verstehen, Entitäten extrahieren, Strukturen rekonstruieren und Qualität bewerten |
| **Indexschicht** | Milvus / Qdrant / Neo4j / PostgreSQL | Effiziente Speicherung von Vektoren, Grafiken und Metadaten |

**Schlüsselverständnis**: LLM ist eine Verständnis-Engine, kein Extraktionstool. Es ist eine Verschwendung, LLM zum Extrahieren von Text zu verwenden. Der richtige Weg besteht darin, spezielle Tools zu verwenden, um es zu extrahieren und LLM verständlich zu machen.

Der Wert von Claude Code CLI liegt nicht darin, den gesamten Prozess zu ersetzen, sondern in der Handhabung komplexer Analyseaufgaben, die eine mehrstufige Argumentation erfordern (z. B. dokumentenübergreifende vergleichende Argumentation, komplexes Tabellenverständnis, Multi-Chart-Korrelationsanalyse).

---

**Referenzen:**1. Han, K., et al. (2026). *Von symbolischen zu natürlichsprachlichen Beziehungen: Überdenken der Konstruktion von Wissensgraphen im Zeitalter großer Sprachmodelle.* arXiv.
2. Li, L., et al. (2025). *Aufbau eines Wissensgraphen für Unternehmensfusionen und -übernahmen: Cross-Domain-Value-Mining-Methode von Large Language Model (LLM) und Graph Neural Network.* IEEE Access.
3. Longo, C. F., et al. (2024). *HTC-GEN: Ein generativer LLM-basierter Ansatz zur Bewältigung von Datenknappheit bei der hierarchischen Textklassifizierung.* DATA-Konferenz.
4. Santos, J., et al. (2025). *Feinabstimmung transformatorbasierter LLMs in der hierarchischen Textklassifizierung.* Datenwissenschaft.
5. Dong, J., et al. (2025). *Wissensextraktion und -ausrichtung für die Minenbelüftung: Ein Wissensgraph-Konstruktionsrahmen basierend auf großen Sprachmodellen.* Technische Anwendungen künstlicher Intelligenz.