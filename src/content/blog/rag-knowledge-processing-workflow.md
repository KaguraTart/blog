---
title: "RAG 知识库知识处理工作流：从 PDF 解析到智能分类的完整方案"
description: "深入探讨 RAG 知识库构建中的知识提取、处理、分类三大环节，分析 LLM 的适用场景与局限性，评估 Claude Code CLI / Gemini CLI 等外部工具的能力边界，并提出一套结合专用解析库与 Agent 工作流的混合架构"
pubDate: 2026-04-07T12:13:15+08:00
tags: ["RAG", "知识库", "知识抽取", "PDF解析", "Claude Code", "工作流", "Agent", "LlamaIndex", "LangChain"]
category: Tech
---

# RAG 知识库知识处理工作流：从 PDF 解析到智能分类的完整方案

> 构建 RAG 知识库时，最常被低估的环节不是检索算法，而是**知识进入系统之前的处理工作**。一份 PDF 从上传到变成高质量向量片段，涉及解析、清洗、结构提取、实体识别、分类、去重、质量评估十余个步骤。LLM 能做，但未必是最优选择。本文系统梳理这个工作流中的每个环节，评估 LLM 和各类专用工具的能力边界，并给出**可落地的混合架构**与**具体执行步骤**。

---

## 1. 知识处理工作流的全局视角

### 1.1 知识处理的完整生命周期

RAG 知识库的知识处理分为五个阶段：

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

**核心问题**：这五个阶段中，哪些环节应该用 LLM，哪些应该用专用工具？两者的边界在哪里？

### 1.2 LLM vs 专用工具的能力对比

| 环节 | LLM 适用性 | 专用工具 | 推荐方案 |
|------|-----------|---------|---------|
| **文字提取** | ⚠️ 慢/贵/不稳定 | ✅ pdfminer / PyMuPDF / pdfplumber | **专用工具** |
| **表格解析** | ✅ 理解表格语义 | ✅ Camelot / Tabula / pdfplumber | **LLM + 专用双重校验** |
| **图表描述** | ✅ 视觉理解 | ⚠️ 需要多模态模型 | **多模态 LLM** |
| **层级结构识别** | ✅ 理解标题关系 | ⚠️ 弱 | **LLM** |
| **实体抽取（NER）** | ✅ 少样本/零样本 | ✅ spaCy / BERT-NER | **LLM（通用）/ 专用（垂直领域）** |
| **关系抽取** | ✅ 强 | ❌ 专用工具基本不支持 | **LLM** |
| **知识分类** | ✅ 灵活 | ✅ 传统 ML 分类器 | **LLM（少量数据）/ ML（大量数据）** |
| **质量评估** | ✅ LLM 自评 | ❌ 不适用 | **LLM** |
| **去重检测** | ✅ 语义去重 | ⚠️ 只能做字面去重 | **LLM + MinHash** |

**结论**：LLM 擅长**理解、推理、生成**类任务，专用工具擅长**提取、稳定、快速**类任务。两者互补，不是非此即彼。

---

## 2. 阶段一 & 二：知识解析 — 为什么 LLM 不够用

### 2.1 PDF 的结构复杂性

PDF 并不是"结构化文档"，它的底层是**一堆绘制指令**（字体、位置、颜色）。这导致 PDF 解析天然困难：

- **文字方向**：竖排文字、混排文字（中文竖排 + 英文横排）
- **表格结构**：无线框表格、跨行跨列单元格、嵌套表格
- **图片内嵌文字**：扫描件、截图中的文字无法直接提取
- **公式**：LaTeX 公式、数学符号、化学式
- **页眉页脚**：每页重复，需要过滤
- **脚注尾注**：引用标注与正文混淆

LLM 无法直接处理 PDF，必须先提取文字。但纯文字提取≠高质量内容还原。

### 2.2 专用解析工具生态

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

## 3. 阶段三：知识处理 — LLM 的强项，但需要策略

### 3.1 实体抽取与关系抽取

实体抽取（NER）和关系抽取是知识处理的核心。LLM 在这个环节表现出色：

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

### 3.2 LLM 做知识处理的局限性

LLM 在知识处理阶段有明确的局限，需要设计策略应对：

| 局限性 | 表现 | 应对策略 |
|--------|------|---------|
| **速度慢** | 单次 API 调用 1-5 秒 | 异步并发 + 批处理 |
| **成本高** | 大量 chunks 调用 API 成本爆炸 | 优先对关键 chunks 调用，通用 chunks 用规则 |
| **输出不稳定** | 同一文本多次调用结果可能不同 | Few-shot prompt + 输出 Schema 约束 |
| **幻觉** | 可能虚构不存在的实体 | 置信度过滤 + 原文回溯校验 |
| **上下文窗口** | 超长文档截断 | 滑动窗口分段处理 |
| **领域适配** | 通用 LLM 对垂直领域实体识别弱 | LoRA 微调或 RAG 增强 |

---

## 4. 阶段四：知识分类 — LLM vs 专用模型

### 4.1 分类策略选择

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

## 5. 外部 Agent 工具的深度评估

### 5.1 Claude Code CLI — 知识库构建的意外利器

Claude Code CLI（`claude-code`）并不只是代码生成工具，它的 **Read-Browse-Analyze 能力**可以用于知识库工作流：

```bash
# 安装 Claude Code CLI
npm install -g @anthropic-ai/claude-code

# 认证
claude-code auth
```

**Claude Code 在知识处理中的用法**：

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

### 5.2 Gemini CLI — 多模态 PDF 处理的潜力

Google 的 **Gemini CLI**（通过 `gemini` SDK 或 AI Studio）支持多模态，在处理含图表的 PDF 时有独特优势：

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

### 5.3 专用工具的补充价值

| 工具 | 适用场景 | LLM 替代可行性 | 推荐理由 |
|------|---------|--------------|---------|
| **PyMuPDF** | 快速文字/图片提取 | ❌ LLM 无法替代提取 | 速度快，稳定 |
| **pdfplumber** | 表格提取（有线框） | ⚠️ LLM 可辅助校验 | 规则提取更可靠 |
| **Camelot** | 复杂表格（无线框） | ⚠️ LLM 校验效果更好 | 专用算法更准 |
| **Pytesseract** | OCR 文字识别 | ⚠️ 多模态 LLM 可替代 | 需要训练数据 |
| **MarkItDown** | 文档转 Markdown | ⚠️ LLM 可重建结构 | 一键转换最简单 |
| **Unstructured.io** | 企业文档（混合格式） | ⚠️ LLM 做后处理 | 最好的企业文档解析库 |
| **Nougat** | LaTeX 公式 + 学术 PDF | ✅ 多模态 LLM 可替代 | 学术 PDF 专用 |
| **MinerU**（阿里） | 复杂中文 PDF | ✅ 兼顾速度与准确性 | 中文文档首选 |

---

## 6. 最优工作流：混合架构设计

### 6.1 核心设计原则

**原则一：LLM 只用于它最擅长的环节**
- 实体/关系抽取 → LLM
- 层级结构重建 → LLM
- 质量评估 → LLM
- 图表描述 → 多模态 LLM

**原则二：稳定、快速的任务交给专用工具**
- 文字提取 → PyMuPDF / pdfplumber
- OCR → Pytesseract / EasyOCR
- 分块 → 规则 + 重叠滑动窗口

**原则三：批处理代替逐条调用**
- LLM 调用成本高，批量打包（每个请求 10-50 个 chunks）

**原则四：异步并发**
- LLM 调用异步化，专用工具同步快速处理

### 6.2 完整混合工作流

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

### 6.3 完整实现代码

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

## 7. 具体实施步骤

### 步骤一：环境准备（Day 1）

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

### 步骤二：快速基线（Day 1-2）

先跑通最简单的流程，不追求完美：

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

### 步骤三：评估基线质量（Day 2）

- 构建测试集：50-100 个 Query，手动标注期望答案
- 评估指标：RAGAS（Faithfulness / Answer Relevance / Context Precision）
- 定位瓶颈：是解析问题、分块问题、还是检索问题？

### 步骤四：针对性优化（Day 3-7）

根据评估结果，优先解决最大瓶颈：

| 瓶颈 | 优化方案 |
|------|---------|
| 表格信息丢失 | 加入 pdfplumber 规则提取 + LLM 校验 |
| 分块打断语义 | 改进分块策略（按章节/重叠更多） |
| 检索召回低 | 加入知识图谱 + BM25 混合检索 |
| 实体识别不准 | 用 LLM 做实体抽取 + KG 索引 |
| 分类错误 | 加入领域 Taxonomy + LLM 分类 |

### 步骤五：LLM 知识抽取集成（Week 2）

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

### 步骤六：持续优化与监控（Week 3+）

- **定期评估**：每周随机采样 Query，评估 RAG 质量
- **反馈飞轮**：收集用户对答案的反馈，自动加入训练数据
- **知识更新**：增量处理新文档，不全量重建

---

## 8. 工具选型决策树

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

## 9. 总结：工具组合的最佳实践

构建高质量 RAG 知识库不是单一工具能解决的。最佳实践是**三层工具协同**：

| 层级 | 工具 | 职责 |
|------|------|------|
| **提取层** | PyMuPDF / pdfplumber / Unstructured / Nougat | 快速稳定地提取原始内容 |
| **理解层** | Claude / GPT-4 / Gemini | 理解语义、抽取实体、重建结构、评估质量 |
| **索引层** | Milvus / Qdrant / Neo4j / PostgreSQL | 高效存储向量、图谱和元数据 |

**关键认知**：LLM 是理解引擎，不是提取工具。用 LLM 提取文字是浪费，用专用工具提取后让 LLM 理解才是正道。

Claude Code CLI 的价值不在于替代整个流程，而在于处理那些**需要多步推理的复杂分析任务**（如跨文档对比推理、复杂表格理解、多图表关联分析）。

---

**参考文献：**

1. Han, K., et al. (2026). *From Symbolic to Natural-Language Relations: Rethinking Knowledge Graph Construction in the Era of Large Language Models.* arXiv.
2. Li, L., et al. (2025). *Construction of Knowledge Graph for Enterprise Mergers and Acquisitions: Cross-Domain Value Mining Method of Large Language Model (LLM) and Graph Neural Network.* IEEE Access.
3. Longo, C. F., et al. (2024). *HTC-GEN: A Generative LLM-Based Approach to Handle Data Scarcity in Hierarchical Text Classification.* DATA Conference.
4. Santos, J., et al. (2025). *Fine-Tuning Transformer-Based LLMs in Hierarchical Text Classification.* Data Science.
5. Dong, J., et al. (2025). *Knowledge Extraction and Alignment for Mine Ventilation: A Knowledge Graph Construction Framework Based on Large Language Models.* Engineering Applications of Artificial Intelligence.
