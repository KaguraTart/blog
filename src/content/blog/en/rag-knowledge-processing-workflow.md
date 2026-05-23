---
title: "RAG knowledge base knowledge processing workflow: a complete solution from PDF parsing to intelligent classification"
description: "In-depth discussion of the three major aspects of knowledge extraction, processing, and classification in the construction of RAG knowledge base, analyzing the applicable scenarios and limitations of LLM, evaluating the capability boundaries of external tools such as Claude Code CLI / Gemini CLI, and proposing a hybrid architecture that combines a dedicated parsing library and Agent workflow"
pubDate: 2026-04-07T12:13:15+08:00
tags: ["RAG", "knowledge base", "knowledge extraction", "PDF analysis", "Claude Code", "Workflow", "Agent", "LlamaIndex", "LangChain"]
category: Tech
---

# RAG knowledge base knowledge processing workflow: a complete solution from PDF parsing to intelligent classification

> When building a RAG knowledge base, the most underestimated link is not the retrieval algorithm, but the processing work before the knowledge enters the system. From uploading a PDF to becoming a high-quality vector fragment, it involves more than ten steps of parsing, cleaning, structure extraction, entity recognition, classification, deduplication, and quality assessment. LLM can be done, but it may not be the best choice. This article systematically sorts out each link in this workflow, evaluates the capability boundaries of LLM and various special tools, and provides an implementable hybrid architecture and specific implementation steps.

---

## 1. Global perspective of knowledge processing workflow

### 1.1 Complete life cycle of knowledge processing

The knowledge processing of RAG knowledge base is divided into five stages:

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

**Core question**: Among these five stages, which links should use LLM and which should use special tools? Where is the boundary between the two?

### 1.2 Comparison of capabilities of LLM vs special tools| Links | LLM applicability | Special tools | Recommended solutions |
|------|-----------|---------|---------|
| **Text Extraction** | ⚠️ Slow/Expensive/Unstable | ✅ pdfminer / PyMuPDF / pdfplumber | **Specialized Tools** |
| **Table parsing** | ✅ Understand table semantics | ✅ Camelot / Tabula / pdfplumber | **LLM + dedicated double verification** |
| **Chart description** | ✅ Visual understanding | ⚠️ Requires multimodal model | **Multimodal LLM** |
| **Hierarchical Structure Recognition** | ✅ Understand title relationships | ⚠️ Weak | **LLM** |
| **Entity Extraction (NER)** | ✅ Few samples/zero samples | ✅ spaCy / BERT-NER | **LLM (general) / special (vertical field)** |
| **Relationship Extraction** | ✅ Strong | ❌ Special tools are basically not supported | **LLM** |
| **Knowledge Classification** | ✅ Flexible | ✅ Traditional ML Classifier | **LLM (Low Data) / ML (Lots of Data) ** |
| **Quality Assessment** | ✅ LLM Self-Assessment | ❌ Not applicable | **LLM** |
| **Duplication detection** | ✅ Semantic deduplication | ⚠️ Only literal deduplication can be done | **LLM + MinHash** |

**Conclusion**: LLM is good at **understanding, reasoning, and generation** tasks, and special tools are good at **extraction, stability, and fast** tasks. The two are complementary, not either/or.

---

## 2. Phase 1 & 2: Knowledge Analysis - Why LLM is Not Enough

### 2.1 Structural complexity of PDF

PDF is not a "structured document". Its bottom layer is a bunch of drawing instructions (font, position, color). This makes PDF parsing inherently difficult:

- **Text direction**: vertical text, mixed text (Chinese vertical + English horizontal)
- **Table structure**: unframed tables, cross-row and cross-column cells, nested tables
- **Text embedded in pictures**: Text in scans and screenshots cannot be directly extracted
- **Formulas**: LaTeX formulas, mathematical symbols, chemical formulas
- **Header and Footer**: Repeat on each page and need to be filtered
- **Footnotes and Endnotes**: Quotation marks are confused with the main textLLM cannot process PDFs directly, text must be extracted first. But pure text extraction ≠ high-quality content restoration.

### 2.2 Ecology of dedicated analysis tools

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

## 3. Phase 3: Knowledge processing - LLM’s strength, but requires strategy

### 3.1 Entity extraction and relationship extraction

Entity extraction (NER) and relationship extraction are the core of knowledge processing. LLM excels in this area:

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

### 3.2 Limitations of LLM for knowledge processing

LLM has clear limitations in the knowledge processing stage, which require designing strategies to deal with:

| Limitations | Performance | Coping Strategies |
|--------|------|---------|
| **Slow** | Single API call 1-5 seconds | Asynchronous concurrency + batch processing |
| **High cost** | The cost of calling the API for a large number of chunks explodes | Prioritize calls to key chunks and use rules for general chunks |
| **Unstable output** | The results of calling the same text multiple times may be different | Few-shot prompt + output Schema constraints |
| **Illusion** | Possibly fictitious entities that do not exist | Confidence filtering + original text back-checking |
| **Context Window** | Ultra-long document truncation | Sliding window segmentation processing |
| **Domain Adaptation** | General LLM is weak in vertical domain entity recognition | LoRA fine-tuning or RAG enhancement |

---

## 4. Stage 4: Knowledge Classification - LLM vs Specialized Model

### 4.1 Classification strategy selection

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

## 5. In-depth evaluation of external agent tools

### 5.1 Claude Code CLI — an unexpected tool for building knowledge bases

Claude Code CLI (`claude-code`) is not just a code generation tool, its **Read-Browse-Analyze capability** can be used in knowledge base workflows:

```bash
# 安装 Claude Code CLI
npm install -g @anthropic-ai/claude-code

# 认证
claude-code auth
```

**Usage of Claude Code in knowledge processing**:

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

### 5.2 Gemini CLI — The potential of multimodal PDF processingGoogle's **Gemini CLI** (via `gemini` SDK or AI Studio) supports multi-modality and has unique advantages when processing PDFs with charts:

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

### 5.3 Supplementary value of special tools

| Tools | Applicable scenarios | LLM alternative feasibility | Reasons for recommendation |
|------|---------|--------------|---------|
| **PyMuPDF** | Fast text/image extraction | ❌ LLM cannot replace extraction | Fast and stable |
| **pdfplumber** | Table extraction (wired frame) | ⚠️ LLM can assist verification | Rule extraction is more reliable |
| **Camelot** | Complex tables (no frames) | ⚠️ LLM verification effect is better | Special algorithm is more accurate |
| **Pytesseract** | OCR text recognition | ⚠️ Multi-modal LLM replaceable | Requires training data |
| **MarkItDown** | Convert document to Markdown | ⚠️ LLM can reconstruct the structure | One-click conversion is the easiest |
| **Unstructured.io** | Enterprise documents (mixed format) | ⚠️ LLM for post-processing | The best enterprise document parsing library |
| **Nougat** | LaTeX formula + academic PDF | ✅ Multimodal LLM alternative | Academic PDF only |
| **MinerU** (Alibaba) | Complex Chinese PDF | ✅ Taking into account both speed and accuracy | The first choice for Chinese documents |

---

## 6. Optimal workflow: hybrid architecture design

### 6.1 Core Design Principles

**Principle 1: LLM should only be used for what it is best at**
- Entity/relationship extraction → LLM
- Hierarchy reconstruction → LLM
- Quality assessment → LLM
- Chart Description → Multimodal LLM

**Principle 2: Leave stable and fast tasks to special tools**
- Text extraction → PyMuPDF/pdfplumber
- OCR → Pytesseract/EasyOCR
- Chunking → Rules + Overlapping sliding windows

**Principle 3: Batch processing instead of one-by-one calls**
- LLM call cost is high and packaged in batches (10-50 chunks per request)**Principle 4: Asynchronous Concurrency**
- LLM calls are asynchronous, and special tools can process them synchronously and quickly

### 6.2 Complete hybrid workflow

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

### 6.3 Complete implementation code

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

## 7. Specific implementation steps

### Step 1: Environment preparation (Day 1)

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

### Step 2: Quick Baseline (Day 1-2)

Go through the simplest process first and don’t pursue perfection:

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

### Step 3: Assess baseline quality (Day 2)

- Build a test set: 50-100 Query, manually label expected answers
- Evaluation indicators: RAGAS (Faithfulness/Answer Relevance/Context Precision)
- Locate the bottleneck: Is it a parsing problem, a chunking problem, or a retrieval problem?

### Step 4: Targeted optimization (Day 3-7)

Based on the evaluation results, prioritize solving the biggest bottlenecks:

| Bottleneck | Optimization plan |
|------|---------|
| Table information is lost | Add pdfplumber rule extraction + LLM verification |
| Chunking Breaking Semantics | Improved chunking strategy (by chapter/more overlap) |
| Low search recall | Add knowledge graph + BM25 hybrid search |
| Entity recognition is inaccurate | Use LLM for entity extraction + KG index |
| Classification Error | Add Domain Taxonomy + LLM Classification |

### Step 5: LLM knowledge extraction integration (Week 2)

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

### Step 6: Continuous optimization and monitoring (Week 3+)

- **Periodic evaluation**: Randomly sample Query every week to evaluate RAG quality
- **Feedback Flywheel**: Collect user feedback on answers and automatically add training data
- **Knowledge update**: incremental processing of new documents without full reconstruction

---

## 8. Tool selection decision tree

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

## 9. Summary: Best Practices for Tool Combination

Building a high-quality RAG knowledge base cannot be solved by a single tool. The best practice is **three-tier tool collaboration**:| Hierarchy | Tools | Responsibilities |
|------|------|------|
| **Extraction Layer** | PyMuPDF/pdfplumber/Unstructured/Nougat | Fast and stable extraction of original content |
| **Understanding layer** | Claude / GPT-4 / Gemini | Understand semantics, extract entities, reconstruct structures, and evaluate quality |
| **Index Layer** | Milvus / Qdrant / Neo4j / PostgreSQL | Efficient storage of vectors, graphs and metadata |

**Key understanding**: LLM is an understanding engine, not an extraction tool. It is a waste to use LLM to extract text. The right way is to use special tools to extract and let LLM understand it.

The value of Claude Code CLI does not lie in replacing the entire process, but in handling complex analysis tasks that require multi-step reasoning (such as cross-document comparative reasoning, complex table understanding, multi-chart correlation analysis).

---

**References:**1. Han, K., et al. (2026). *From Symbolic to Natural-Language Relations: Rethinking Knowledge Graph Construction in the Era of Large Language Models.* arXiv.
2. Li, L., et al. (2025). *Construction of Knowledge Graph for Enterprise Mergers and Acquisitions: Cross-Domain Value Mining Method of Large Language Model (LLM) and Graph Neural Network.* IEEE Access.
3. Longo, C. F., et al. (2024). *HTC-GEN: A Generative LLM-Based Approach to Handle Data Scarcity in Hierarchical Text Classification.* DATA Conference.
4. Santos, J., et al. (2025). *Fine-Tuning Transformer-Based LLMs in Hierarchical Text Classification.* Data Science.
5. Dong, J., et al. (2025). *Knowledge Extraction and Alignment for Mine Ventilation: A Knowledge Graph Construction Framework Based on Large Language Models.* Engineering Applications of Artificial Intelligence.