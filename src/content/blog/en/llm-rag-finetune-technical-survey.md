---
title: "Panoramic survey of LLM RAG knowledge base and fine-tuning training technology"
description: "An in-depth analysis of the RAG knowledge base full-process technology stack (retrieval/Embedding/vector database/reordering) and the complete guide to LLM fine-tuning (LoRA/QLoRA/full fine-tuning/SFT/RLHF), from architecture design to project implementation, with comparison of mainstream frameworks and selection suggestions."
pubDate: 2026-04-07T11:43:17+08:00
tags: ["large language model", "RAG", "knowledge base", "vector database", "fine-tuning", "LoRA", "QLoRA", "SFT", "RLHF", "Embedding", "LLM"]
category: Tech
sourceHash: "2860624c046e8bbdb4c91f6836f0348410230fbc"
---

# LLM RAG Panoramic survey on knowledge base and fine-tuning training technology

> Two core routes for building enterprise-level large model applications: **RAG (Retrieval Enhanced Generation)** and **Fine-tuning**. The former allows the model to "read thousands of books", and the latter allows the model to "learn new skills". This article completely connects the two routes: from RAG's knowledge base construction, retrieval strategy, answer generation, to fine-tuned pre-training/instruction fine-tuning/SFT/RLHF entire process, with the latest framework comparison, engineering pitfall notes and selection suggestions. Whether you want to do private knowledge question and answer, vertical field adaptation, or let the model learn a specific output format, this survey can give you a complete technical map.

---

## 1. RAG Overview: Why do you need retrieval enhancement?

### 1.1 Knowledge Dilemma of LLM

Large language models have three inherent limitations:

| Restriction type | Performance | Typical cases |
|---------|------|---------|
| **Knowledge Cutoff** | Training data has a cutoff date | GPT-4 Turbo knowledge cutoff 2024-06 |
| **Hallucination** | Serious nonsense about uncertain knowledge | Fictional legal terms and product parameters |
| **Unable to access private data** | Internal corporate documents and databases are not available to the public | Financial reports, customer service knowledge base, code base |
| **Lack of long-tail knowledge** | Sparse knowledge in unpopular fields | Specific industry terminology, self-developed technologies |
| **Difficulty updating knowledge** | New knowledge requires retraining | Today’s price, real-time inventory |

**RAG's core value**: "Arm" LLM by retrieving external knowledge without modifying the model weights - not only solves the problem of knowledge timeliness, avoids illusions, but also naturally supports private data.

### 1.2 Three generations of RAG architecture evolution

**First Generation: Naive RAG (2020–2023)**

```
用户问题 → 向量化 → Top-K 检索 → 拼接上下文 → LLM 生成
```

The process is simple and straightforward, but the problems are also obvious: poor retrieval quality, insufficient use of context windows, and generated content that is out of touch with retrieval results.

**Second Generation: Advanced RAG (2023–2024)**

```
用户问题 → Query 改写/扩展 → 向量化 → 检索 → 重排序 → LLM 生成
         ↑                                   ↑
      HyDE 假设文档                 Cross-Encoder 重排序
```Improvement points: Rewrite the Query before retrieval (HyDE, Query Expansion), and reorder the results after retrieval (Cross-Encoder / BM25 + Vector Hybrid), significantly improving the retrieval recall rate and precision.

**Third Generation: Modular RAG (2024–)**

```
用户问题 → 路由 → 工具调用 → 检索 → 后处理 → LLM 生成
                      ↑
              知识图谱 / Web 搜索 / 计算器 / API
```

Modular architecture: Search becomes a pluggable tool, and the router decides when to search, what to search, and what tool to use. Representative work: NeME, Self-RAG, Corrective-RAG (CRAG).

---

## 2. The whole process of building RAG knowledge base

### 2.1 Document parsing and text extraction

The first step in a knowledge base is to turn raw documents into clean text. The difficulty of processing different formats varies greatly:

```python
from pathlib import Path

class DocumentProcessor:
    """多格式文档解析器"""
    
    SUPPORTED_FORMATS = {
        '.pdf': 'parse_pdf',
        '.docx': 'parse_docx',
        '.txt': 'parse_txt',
        '.md': 'parse_markdown',
        '.html': 'parse_html',
        '.csv': 'parse_csv',
        '.xlsx': 'parse_excel',
        '.pptx': 'parse_pptx',
    }
    
    def parse_pdf(self, file_path):
        """PDF 解析：文字 PDF vs 扫描 PDF"""
        import pymupdf  # fitz
        
        doc = pymupdf.open(file_path)
        full_text = []
        
        for page_num, page in enumerate(doc):
            # 方法1: 直接提取文字
            text = page.get_text()
            
            if text.strip():
                full_text.append({
                    'page': page_num + 1,
                    'text': text,
                    'bbox': None
                })
            else:
                # 方法2: OCR（扫描件或图片PDF）
                # 使用 pymupdf 的 pixmap + OCR
                pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
                ocr_text = self._ocr_image(pix.tobytes("png"))
                full_text.append({
                    'page': page_num + 1,
                    'text': ocr_text,
                    'is_ocr': True
                })
        
        return full_text
    
    def parse_docx(self, file_path):
        """Word 文档解析（保留层级结构）"""
        from docx import Document
        
        doc = Document(file_path)
        sections = []
        current_section = {'title': '', 'content': []}
        
        for para in doc.paragraphs:
            if para.style.name.startswith('Heading'):
                # 保存上一个段落
                if current_section['content']:
                    sections.append(current_section)
                current_section = {
                    'title': para.text,
                    'content': []
                }
            else:
                if para.text.strip():
                    current_section['content'].append(para.text)
        
        # 最后一个段落
        if current_section['content']:
            sections.append(current_section)
        
        return sections
    
    def _ocr_image(self, image_bytes):
        """OCR 识别（支持中文）"""
        import pytesseract
        import PIL.Image
        import io
        
        img = PIL.Image.open(io.BytesIO(image_bytes))
        # 指定中文识别
        return pytesseract.image_to_string(img, lang='chi_sim+eng')
```

**Special handling of enterprise documents**:

```python
def extract_table_as_markdown(table_element):
    """将 HTML/Word 表格提取为 Markdown"""
    rows = []
    for row in table_element.find_all('tr'):
        cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
        rows.append('| ' + ' | '.join(cells) + ' |')
    
    if not rows:
        return ''
    
    # 添加分隔行
    sep = '| ' + ' | '.join(['---'] * len(rows[0].split('|'))) + ' |'
    rows.insert(1, sep)
    return '\n'.join(rows)
```

### 2.2 Text chunking strategy (Chunking)

Chunking is the most critical and most overlooked aspect of RAG. The chunking strategy directly affects the retrieval quality and generation effect.

**Core Principles**:
- **Semantic Completeness**: Try to let each chunk express a complete semantic unit
- **Length Control**: Limited by the LLM context window and the Token upper limit of the Embedding model
- **Overlap Design**: Keep overlap between adjacent chunks to avoid loss of boundary information

#### Strategy 1: Fixed-length chunking (the simplest)

```python
def fixed_chunk(text, chunk_size=500, overlap=50):
    """
    固定 token 数分块（重叠设计）
    chunk_size: 每个 chunk 的最大 token 数
    overlap: 相邻 chunk 重叠的 token 数
    """
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 同款编码器
    
    tokens = enc.encode(text)
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        
        chunks.append({
            'text': chunk_text,
            'start_token': start,
            'end_token': end,
        })
        
        start = end - overlap  # 滑动窗口，重叠 overlap 个 token
    
    return chunks
```

#### Strategy 2: Recursive character chunking (maintaining semantic boundaries)

```python
def recursive_chunk(text, separators=['\n\n', '\n', '. ', ' ', ''], max_length=500):
    """
    递归分块：优先在大分隔符处切分，不够则用小分隔符
    效果：尽量保持段落/句子完整性
    """
    def split_by_separator(text, sep):
        parts = text.split(sep)
        return [(part, sep) for part in parts if part.strip()]
    
    def recurse(parts, sep_level=0):
        if sep_level >= len(separators):
            return parts
        
        sep = separators[sep_level]
        
        merged = []
        current = ''
        current_sep = ''
        
        for part, orig_sep in parts:
            test = current + current_sep + part
            
            if len(test) <= max_length:
                current = test
                current_sep = sep
            else:
                if current:
                    merged.append((current, current_sep))
                current = part
                current_sep = ''
        
        if current:
            merged.append((current, current_sep))
        
        return recurse(merged, sep_level + 1)
    
    result = split_by_separator(text, separators[0])
    return [text for text, _ in recurse(result)]
```

#### Strategy 3: Semantic chunking (based on LLM/Embedding)

```python
def semantic_chunk_by_embedding(text, embed_model, max_length=500, threshold=0.7):
    """
    基于句子级别 Embedding 相似度的语义分块
    原理：相邻句子 embedding 相似度突变 = 话题转换点
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    # 按句子切分
    sentences = text.split('。')
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= 2:
        return [{'text': '。'.join(sentences), 'sentences': sentences}]
    
    # 获取每个句子的 embedding
    embeddings = embed_model.encode(sentences)
    
    # 计算相邻句子的余弦相似度
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0, 0]
        similarities.append(sim)
    
    # 相似度低于阈值的位置 = 分块边界
    boundaries = [0]
    for i, sim in enumerate(similarities):
        if sim < threshold:
            boundaries.append(i + 1)
    boundaries.append(len(sentences))
    
    # 构建 chunks
    chunks = []
    for i in range(len(boundaries) - 1):
        chunk_text = '。'.join(sentences[boundaries[i]:boundaries[i+1]])
        chunks.append({
            'text': chunk_text,
            'sentences': sentences[boundaries[i]:boundaries[i+1]],
            'boundary_scores': similarities[boundaries[i]:boundaries[i+1]-1]
        })
    
    return chunks
```

#### Strategy 4: Domain Adaptive Blocking

```python
def domain_aware_chunk(file_path, domain='code'):
    """
    领域自适应的分块策略
    """
    if domain == 'code':
        # 代码：按函数/类级别分块
        return code_chunk_by_ast(file_path)
    elif domain == 'legal':
        # 法律文书：按条款/章节分块
        return legal_chunk_by_article(file_path)
    elif domain == 'medical':
        # 医学文献：按段落/小节分块（避免切断疾病描述）
        return medical_chunk_by_section(file_path)
    elif domain == 'qa':
        # 问答对：一个问题+答案 = 一个 chunk
        return qa_chunk_by_pair(file_path)
```

**Block size reference table:**| Scenario | Recommended Chunk Size | Description |
|------|--------------|------|
| General document | 500–1000 tokens | Balance semantic completeness and retrieval accuracy |
| Codebase | 200–500 tokens | By function/class level, preserving calling context |
| Paper/Report | 1000–2000 tokens | Long paragraphs require a large window to understand |
| Short Q&A | 100–200 tokens | Exact matching to avoid irrelevant context interference |
| Legal terms | 500–800 tokens | Single term is the smallest unit |
| Multi-modal (PDF) | Separate chunks for tables/pictures | Markdown for tables and descriptions for pictures |

### 2.3 Metadata and knowledge graph

**The value of Metadata**: Attaching descriptive information to each chunk greatly improves retrieval accuracy and filtering capabilities.

```python
@dataclass
class ChunkMetadata:
    """Chunk 元数据"""
    source: str                    # 文档名称/URL
    source_type: str               # pdf/docx/html/slide
    page_number: int               # 页码
    section_title: str             # 所属章节标题
    heading_path: List[str]        # 标题层级路径
    author: Optional[str]          # 作者
    created_at: datetime           # 文档创建时间
    last_modified: datetime        # 文档修改时间
    document_id: str              # 文档唯一ID
    chunk_index: int              # Chunk 在文档中的序号
    word_count: int               # 字数
    language: str                 # 语言
    tags: List[str]               # 自动抽取的关键词标签
    legal_clause_id: Optional[str] # 法律条款编号（如有）
    table_caption: Optional[str]   # 表格标题（如有是表格 chunk）
    is_ocr: bool                  # 是否来自 OCR

class EnrichedChunk:
    def __init__(self, content: str, metadata: ChunkMetadata, embedding: np.ndarray):
        self.content = content
        self.metadata = metadata
        self.embedding = embedding
    
    def to_dict(self):
        return {
            'id': f"{self.metadata.document_id}_{self.metadata.chunk_index}",
            'content': self.content,
            'metadata': asdict(self.metadata),
            # VectorDB 存储时通常把 embedding 单独存
        }
```

**Knowledge Graph Enhanced RAG**:

Extract entities and relationships from the text, build a knowledge graph, and retrieve the graph and vectors at the same time:

```python
class GraphRAGProcessor:
    """知识图谱 + 矢量检索的双路 RAG"""
    
    def __init__(self, llm, vector_store, graph_db):
        self.llm = llm
        self.vector_store = vector_store
        self.graph_db = graph_db  # Neo4j / NebulaGraph
    
    def extract_entities_and_relations(self, text):
        """用 LLM 抽取实体和关系（few-shot prompting）"""
        prompt = """
从以下文本中抽取实体和关系，以 JSON 格式输出：

文本：{text}

输出格式：
{{
  "entities": [
    {{"name": "实体名", "type": "实体类型", "description": "描述"}}
  ],
  "relations": [
    {{"source": "实体A", "target": "实体B", "relation": "关系类型", "description": "关系描述"}}
  ]
}}
"""
        response = self.llm.invoke(prompt.format(text=text))
        return json.loads(response.content)
    
    def index_document(self, text, doc_id):
        """同时索引到向量库和图数据库"""
        # 1. 抽取
        kg_data = self.extract_entities_and_relations(text)
        
        # 2. 存入向量库
        chunks = recursive_chunk(text)
        self.vector_store.add_texts(chunks, metadata={'doc_id': doc_id})
        
        # 3. 存入图数据库
        for entity in kg_data['entities']:
            self.graph_db.create_node(
                label='Entity',
                properties=entity
            )
        for rel in kg_data['relations']:
            self.graph_db.create_relationship(
                start_node=rel['source'],
                end_node=rel['target'],
                type=rel['relation'],
                properties={'description': rel['description']}
            )
    
    def query(self, question):
        """混合检索：向量 + 图谱"""
        # 1. 向量检索
        vector_results = self.vector_store.similarity_search(question, k=5)
        
        # 2. 图谱检索（通过实体匹配）
        graph_results = self.graph_db.query_cypher(f"""
            MATCH (e:Entity)
            WHERE e.name CONTAINS '{question}' OR e.description CONTAINS '{question}'
            RETURN e, [(e)-[r]-(related) | {{node: related, relation: type(r)}}]
        """)
        
        # 3. 融合排序（RRF: Reciprocal Rank Fusion）
        fused_results = self.rrf_fusion(vector_results, graph_results, k=60)
        
        return fused_results
    
    def rrf_fusion(self, results_a, results_b, k=60):
        """RRF 融合算法"""
        scores = {}
        
        for rank, result in enumerate(results_a):
            doc_id = result['id']
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        
        for rank, result in enumerate(results_b):
            doc_id = result['id']
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        
        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [self.vector_store.get(id=sid) for sid, _ in sorted_ids]
```

---

## 3. Embedding model and vector database

### 3.1 Embedding model selection

The Embedding model is the "perception layer" of RAG - its quality directly determines the recall and precision of retrieval.

**Comparison of mainstream Embedding models in 2024–2025:**| Model | Dimensions | Context | MTEB Accuracy | Advantage Scenarios | GitHub ⭐ |
|------|------|--------|---------|---------|---------|
| **text-embedding-3-large** (OpenAI) | 3072 | 8191 | ~66% | Universal/English | - |
| **text-embedding-3-small** (OpenAI) | 1536 | 8191 | ~62% | Cost sensitive | - |
| **e5-mistral-7b-instruct** (Microsoft) | 4096 | 4096 | ~66% | Multiple languages/instructions | 10k+ |
| **bge-large-zh-v1.5** (BAAI) | 1024 | 512 | ~64% | Mainly Chinese | 20k+ |
| **bge-m3** (BAAI) | 1024 | 8192 | ~65% | Multilingual/Hybrid Search | 8k+ |
| **GTE-Qwen2-7B-instruct** (Alibaba) | 1024 | 8192 | ~67% | Chinese/English | 5k+ |
| **NV-Embed-v2** (NVIDIA) | 4096 | 128K | ~69% | Long context | - |
| **Cohere-embed-v3** | 1024 | 512 | ~65% | English/Multi-language | - |
| **GritLM-7B** (mix of embedding+LLM) | 4096 | 8K | ~67% | embedding+ generates unity | 3k+ |

**Chinese scene recommendation**: `bge-large-zh-v1.5` or `GTE-Qwen2-7B-instruct`

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingModel:
    """统一的 Embedding 模型封装"""
    
    def __init__(self, model_name='BAAI/bge-large-zh-v1.5', device='cuda'):
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Loaded {model_name}, embedding dim: {self.dimension}")
    
    def encode(self, texts, batch_size=32, normalize=True):
        """批量编码"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=normalize,  # L2 归一化后点积=Cosine
            convert_to_numpy=True
        )
        
        return embeddings
    
    def encode_queries(self, queries):
        """专门为查询优化的编码（加查询指令前缀）"""
        prefixed = [f"为这个句子生成表示以用于检索相关文章：{q}" for q in queries]
        return self.encode(prefixed)
```

**Embedding model fine-tuning** (optional, greatly improves the effect in specific areas):

```python
from sentence_transformers import SentenceTransformerTrainer, losses
from sentence_transformers.datasets import SentenceLabelDataset

def fine-tune_embedding(model_name, train_data, output_dir, n_epochs=3):
    """
    使用对比学习微调 Embedding 模型
    train_data: List[(query, positive_chunk, negative_chunks)]
    """
    model = SentenceTransformer(model_name)
    
    # 对比损失：正例距离拉近，负例距离推远
    train_loss = losses.TripletLoss(model)
    
    # 构造训练集
    train_dataset = SentenceLabelDataset(int_data=train_data)
    
    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=train_dataset,
        loss=train_loss,
        optimizer_class=torch.optim.AdamW,
        optimizer_params={'lr': 2e-5},
    )
    
    trainer.train(epochs=n_epochs)
    model.save(output_dir)
    
    return model
```

### 3.2 Selection and use of vector database

The vector database is responsible for storing Embeddings and performing approximate nearest neighbor (ANN) searches.

**Comparison of mainstream vector databases:**| Database | Algorithms | Index types | Filtering support | Deployment | Scale | Latency | Special capabilities |
|--------|------|---------|---------|------|------|------|----------|
| **Milvus** | HNSW / IVF / DiskANN | Hybrid | ✅ Native | K8s / Docker | Billion level | Microsecond | Strong metadata filtering |
| **Qdrant** | HNSW / DiskANN | Hybrid | ✅ Native | Standalone/K8s | Billion level | Microsecond | Score rescoring |
| **Weaviate** | HNSW | Hybrid | ✅ Native | K8s / Cloud | Billions | Microseconds | GraphQL interface |
| **Chroma** | HNSW (approximate) | Approximate | ✅ Python | Standalone | Millions | Milliseconds | Easiest to use |
| **Pinecone** | — (Cloud) | — | ✅ Managed | Fully Managed | Billions | Microseconds | Fully Managed |
| **FAISS** | HNSW / IVF | Exact + Approximate | ⚠️ Requires additional processing | Embedded | Millions/single machine | Microseconds | GPU accelerated |
| **Elasticsearch** (8.0+) | HNSW | Hybrid | ✅ Native | K8s / Cloud | Billion level | Milliseconds | Full text + vector hybrid |
| **pgvector** (PostgreSQL) | HNSW / IVFFlat | Hybrid | ✅ SQL | K8s / Docker | Billions | Milliseconds | SQL Union Query |

```python
# ============================================================
# Milvus 使用示例（推荐生产环境）
# ============================================================
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

class MilvusVectorStore:
    def __init__(self, host='localhost', port='19530', collection_name='rag_knowledge_base'):
        connections.connect(host=host, port=port)
        self.collection_name = collection_name
        self.embedding_dim = 1024
    
    def create_collection(self, if_exists='drop'):
        """创建 Collection（HNSW 索引）"""
        if utility.has_collection(self.collection_name):
            if if_exists == 'drop':
                utility.drop_collection(self.collection_name)
            else:
                return
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="page", dtype=DataType.INT16),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256),
        ]
        
        schema = CollectionSchema(fields=fields, description="RAG Knowledge Base")
        collection = Collection(name=self.collection_name, schema=schema)
        
        # 创建 HNSW 索引（检索精度高，速度快）
        index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 256}
        }
        collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        
        # 创建标量索引（支持高效过滤）
        collection.create_index(field_name="source", index_params={"index_type": "STL_SORT"})
        
        collection.load()
        print(f"Collection '{self.collection_name}' ready")
        return collection
    
    def insert(self, chunks, embeddings, metadatas):
        """批量插入"""
        collection = Collection(self.collection_name)
        entities = [
            [c['content'] for c in chunks],
            embeddings.tolist(),
            [m.get('source', '') for m in metadatas],
            [m.get('page', 0) for m in metadatas],
            [m.get('doc_id', '') for m in metadatas],
        ]
        collection.insert(entities)
        collection.flush()
        print(f"Inserted {len(chunks)} chunks")
    
    def search(self, query_embedding, k=5, filter_expr=None):
        """
        混合检索：向量相似度 + 元数据过滤
        """
        collection = Collection(self.collection_name)
        collection.load()
        
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 128}  # HNSW 搜索参数，越大越精确越慢
        }
        
        results = collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=k,
            expr=filter_expr,  # e.g., "source == '产品手册' and page > 3"
            output_fields=["content", "source", "page", "doc_id"]
        )
        
        return [
            {
                'id': hit.id,
                'content': hit.entity.get('content'),
                'source': hit.entity.get('source'),
                'page': hit.entity.get('page'),
                'score': hit.distance
            }
            for hit in results[0]
        ]
```

```python
# ============================================================
# Qdrant 使用示例（轻量，推荐中小规模）
# ============================================================
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue

class QdrantVectorStore:
    def __init__(self, url='http://localhost:6333', collection_name='rag_kb'):
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
    
    def create_collection(self, vector_size=1024):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            ),
            # 启用 payload 索引（支持过滤）
            optimizers_config={
                "indexing_threshold": 0,
            }
        )
    
    def upsert(self, chunk_ids, embeddings, payloads):
        """upsert = insert or update"""
        points = [
            {
                "id": chunk_id,
                "vector": embedding.tolist(),
                "payload": {
                    "content": payload['content'],
                    "source": payload.get('source', ''),
                    "page": payload.get('page', 0),
                    "doc_id": payload.get('doc_id', ''),
                    "metadata": payload.get('metadata', {})
                }
            }
            for chunk_id, embedding, payload in zip(chunk_ids, embeddings, payloads)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
    
    def search(self, query_embedding, k=5, filter_source=None):
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=k,
            query_filter=(
                Filter(
                    should=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=filter_source)
                        )
                    ]
                ) if filter_source else None
            ),
            with_payload=True,
            with_vectors=False,
            score_threshold=0.5  # 只返回相似度 > 0.5 的结果
        )
        
        return [
            {
                'id': r.id,
                'content': r.payload['content'],
                'source': r.payload.get('source'),
                'score': r.score
            }
            for r in results
        ]
```

### 3.3 Hybrid retrieval strategy

Limitations of single vector retrieval: queries with different synonyms but similar semantics may fail to retrieve.

**Hybrid Search = Dense (vector) + Sparse (BM25) retrieval fusion**

```python
class HybridRetriever:
    """混合检索：向量 + BM25 + RRF 融合"""
    
    def __init__(self, vector_store, bm25_store, embed_model):
        self.vector_store = vector_store
        self.bm25_store = bm25_store  # 使用 rank_bm25 库
        self.embed_model = embed_model
    
    def retrieve(self, query, k=5, vector_weight=0.7):
        """
        Hybrid Retrieval + RRF 融合
        """
        # 1. 向量检索
        query_embedding = self.embed_model.encode_queries([query])[0]
        vector_results = self.vector_store.search(query_embedding, k=k*2)
        
        # 2. BM25 检索
        bm25_results = self.bm25_store.search(query, k=k*2)
        
        # 3. RRF 融合
        fused = self._rrf_fuse(
            results_a=vector_results,
            results_b=bm25_results,
            weight_a=vector_weight,
            weight_b=1 - vector_weight,
            k=60
        )
        
        return fused[:k]
    
    def _rrf_fuse(self, results_a, results_b, weight_a, weight_b, k=60):
        """加权的 RRF 融合"""
        scores = {}
        
        for rank, r in enumerate(results_a):
            scores[r['id']] = scores.get(r['id'], 0) + weight_a * 1 / (k + rank + 1)
        
        for rank, r in enumerate(results_b):
            scores[r['id']] = scores.get(r['id'], 0) + weight_b * 1 / (k + rank + 1)
        
        # 合并内容
        content_map = {}
        for r in results_a + results_b:
            content_map[r['id']] = r.get('content', '')
        
        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {'id': sid, 'score': score, 'content': content_map.get(sid, '')}
            for sid, score in sorted_ids
        ]


# ============================================================
# BM25 存储（rank_bm25）
# ============================================================
import rank_bm25

class BM25Store:
    def __init__(self):
        self.tokenized_corpus = []
        self.corpus = []
        self.model = None
    
    def build(self, chunks):
        self.corpus = chunks
        self.tokenized_corpus = [self._tokenize(c) for c in chunks]
        self.model = rank_bm25.BM25Okapi(self.tokenized_corpus)
    
    def _tokenize(self, text):
        """中英文分词（使用 jieba）"""
        import jieba
        return list(jieba.cut(text))
    
    def search(self, query, k=5):
        tokenized_query = self._tokenize(query)
        scores = self.model.get_scores(tokenized_query)
        
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        return [
            {'id': idx, 'content': self.corpus[idx], 'score': scores[idx]}
            for idx in top_indices
        ]
```

---

## 4. Query understanding and retrieval optimization

### 4.1 Query rewriting technologyThere are often lexical differences between user questions and representations in the knowledge base. Query rewriting is the core module of Advanced RAG.

**HyDE (Hypothetical Document Embeddings)**:

```python
class HyDEQueryRewrite:
    """HyDE: 让 LLM 先生成假设性答案，再用答案检索"""
    
    def __init__(self, llm, embed_model, vector_store):
        self.llm = llm
        self.embed_model = embed_model
        self.vector_store = vector_store
    
    def rewrite(self, query):
        """生成假设性文档，用它来检索"""
        prompt = f"""
你是一个知识库文档生成器。请根据用户问题，生成一段假设性的文档内容，
这段文档应当准确回答用户的问题。

用户问题：{query}

请生成一段详细、专业的文档内容（100-200字）：
"""
        hypothetical_doc = self.llm.invoke(prompt)
        
        # 用假设性文档检索
        hypothetical_embedding = self.embed_model.encode([hypothetical_doc.content])[0]
        results = self.vector_store.search(hypothetical_embedding, k=5)
        
        return results, hypothetical_doc.content
```

**Multi-Query Retrieval (multi-query expansion)**:

```python
def multi_query_rewrite(query, llm, n_queries=3):
    """从不同角度改写问题，扩展检索面"""
    prompt = f"""
请从不同的角度为这个问题生成 {n_queries} 个不同的表述方式。
每个表述应该使用不同的词汇或问法，但表达相同的核心问题。

问题：{query}

输出 JSON 数组格式：
["表述1", "表述2", "表述3"]
"""
    response = llm.invoke(prompt)
    queries = json.loads(response.content)
    
    # 原始查询 + 改写查询，全部检索
    all_queries = [query] + queries
    
    return all_queries
```

**Step-Back Prompting**:

For questions that require abstract reasoning, first extract high-level concepts and then retrieve:

```python
def step_back_rewrite(query, llm):
    """Step-Back: 提取高层概念后检索"""
    prompt = f"""
问题：{query}

请从这个问题中提取核心概念和原则。
输出格式：先给出核心概念（一句话），再给出这个概念下的具体问题。

示例：
问题：特斯拉为什么在中国降价？
核心概念：跨国企业在不同市场的定价策略
具体问题：特斯拉在中国市场的定价历史和竞争策略
"""
    step_back = llm.invoke(prompt)
    
    # 同时检索原始查询和 step-back 查询
    return step_back.content
```

### 4.2 Re-ranking

The retrieved Top-K candidates are further ranked through a re-ranking model, bringing the most relevant results to the top.

```python
# ============================================================
# Cross-Encoder 重排序
# ============================================================
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    """
    Cross-Encoder: (query, document) → 相关性分数
    比 Bi-Encoder 更精确，但更慢（用于重排序，不用于初检）
    """
    
    def __init__(self, model_name='BAAI/bge-reranker-large'):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query, documents, top_k=5):
        """
        对检索结果重排序
        
        documents: List[Dict] - 包含 'content' 字段的文档列表
        """
        pairs = [(query, doc['content']) for doc in documents]
        scores = self.model.predict(pairs)
        
        # 按分数排序
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        return [
            {**documents[i], 'rerank_score': float(scores[i])}
            for i in ranked_indices[:top_k]
        ]
```

**Full Advanced RAG Process**:

```python
class AdvancedRAGPipeline:
    """完整的 Advanced RAG 管线"""
    
    def __init__(self, llm, embed_model, vector_store, reranker):
        self.llm = llm
        self.embed_model = embed_model
        self.vector_store = vector_store
        self.reranker = reranker
    
    def query(self, question, mode='hyde'):
        # Step 1: Query 改写
        if mode == 'hyde':
            hyde = HyDEQueryRewrite(self.llm, self.embed_model, self.vector_store)
            docs, hyde_doc = hyde.rewrite(question)
            rewrite_note = f"[假设文档: {hyde_doc[:100]}...]"
        elif mode == 'multi_query':
            queries = multi_query_rewrite(question, self.llm)
            rewrite_note = f"[多查询: {', '.join(queries[:2])}]"
        else:
            docs = []
            rewrite_note = ''
        
        # Step 2: 向量检索（使用所有改写查询）
        all_docs = []
        for q in (queries if mode == 'multi_query' else [question]):
            emb = self.embed_model.encode_queries([q])[0]
            results = self.vector_store.search(emb, k=5)
            all_docs.extend(results)
        
        # Step 3: 去重（相同 doc_id）
        seen_ids = set()
        unique_docs = []
        for doc in all_docs:
            if doc.get('doc_id') not in seen_ids:
                seen_ids.add(doc.get('doc_id'))
                unique_docs.append(doc)
        
        # Step 4: Cross-Encoder 重排序
        reranked = self.reranker.rerank(question, unique_docs, top_k=5)
        
        # Step 5: 生成答案
        context = '\n\n'.join([f"[来源 {i+1}] {d['content']}" 
                               for i, d in enumerate(reranked)])
        
        answer = self.llm.invoke(f"""
根据以下参考文档回答问题。如果文档中没有足够信息，请明确说明。

参考文档：
{context}

问题：{question}

要求：
1. 引用来源编号标注答案依据
2. 如果信息不足，不要编造
3. 回答简洁准确
""")
        
        return {
            'answer': answer.content,
            'sources': reranked,
            'rewrite_note': rewrite_note
        }
```

---

## 5. LLM fine-tuning: full process technical guide

### 5.1 When should fine-tuning be done?

The choice of RAG vs Fine-tuning is the most common question in engineering decisions:

| Scenario | Recommended solution | Reason |
|------|---------|------|
| Knowledge base Q&A (frequently updated knowledge) | **RAG** | Fine-tuning cannot keep up with the pace of knowledge updates |
| Need to reference external documents | **RAG** | Fine-tuning model cannot access external documents |
| Learning new output formats/styles | **Fine-tuning** | Format and tone need to be internalized into weights |
| Understanding terminology in vertical fields | **Fine-tuning** | A large number of domain-specific concepts need to be internalized |
| Reduce latency/inference cost | **Fine-tuning** | Fine-tuned models can use small models |
| Fix specific error patterns | **Fine-tuning** | Recurring errors require fundamental fixes |
| Requires multiple turns of conversation style | **Fine-tuning** | Conversation style and personality need to be internalized |

**Best practice (most scenarios)**: **RAG + Fine-tuning combined**.
- RAG is responsible for knowledge accuracy and timeliness
- Fine-tuning is responsible for the optimization of style, format and reasoning mode### 5.2 Panorama of fine-tuning methods

```
LLM Fine-tuning 方法
├── Full Fine-tuning（全量微调）
│   ├── 因果语言建模（CLM）
│   ├── 指令微调（SFT）
│   └── RLHF（奖励模型 + PPO/DPO）
│
├── PEFT（参数高效微调）
│   ├── 添加式（Additive）
│   │   ├── Adapter
│   │   └── Prefix Tuning / Prompt Tuning
│   │
│   ├── 重参数化（Reparameterized）
│   │   ├── LoRA / QLoRA
│   │   ├── DoRA（方向分解）
│   │   └── LoftQ
│   │
│   └── 混合式（Hybrid）
│       ├── AdaLoRA
│       ├── QAdaLoRA
│       └── Scaled-LoRA
```

### 5.3 Full Fine-tuning vs PEFT

| Dimensions | Full Fine-tuning | LoRA / QLoRA |
|------|----------------|-------------|
| Video memory requirements | 70B model approximately 140GB (FP16) | 70B model approximately 35GB (QLoRA 4bit) |
| Training time | Dozens of hours/day | Several hours |
| Storage cost | Complete set of weights per task | Only Adapter weights stored per task |
| Catastrophic oblivion | Severe | Mild (only a few parameters are updated) |
| Upper limit of effect | Higher (more learnable parameters) | Slightly lower but the gap is narrowing |
| Hardware requirements | A100 80G × multiple cards | A single card A100 can run 70B |

**Conclusion**: After 2024 **QLoRA** has become the de facto standard - it brings the fine-tuning of the 70B model down to the level of video memory accessible by a single card.

### 5.4 QLoRA in-depth analysis

Combination of **QLoRA = Quantization + LoRA**:

1. **4-bit NormalFloat (NF4) quantization**: compress the pre-training weights to 4-bit with minimal accuracy loss
2. **Double Quant**: Requantize the quantization constant itself to further save video memory
3. **Paged Optimizer**: Automatically swap pages between CPU and GPU when handling gradient update bursts

```python
# ============================================================
# QLoRA 微调完整实现（使用 transformers + peft）
# ============================================================
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, get_peft_model, prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset
from trl import SFTTrainer  # Supervised Fine-tuning Trainer

# ============================================================
# Step 1: 4-bit 量化加载模型
# ============================================================
def load_model_quantized(model_name, load_in_4bit=True):
    """QLoRA: 4-bit 量化加载"""
    
    # NF4 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type="nf4",        # NormalFloat4，比 standard 4bit 更优
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,    # 双重量化
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # QLoRA 需要将模型转为千进制训练模式
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer


# ============================================================
# Step 2: LoRA 配置
# ============================================================
def get_lora_config(target_modules=None):
    """LoRA 配置详解"""
    
    # target_modules: 指定要应用 LoRA 的线性层
    # 不同模型架构的注意力层名称不同：
    # LLaMA: q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj
    # Qwen: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    # ChatGLM: query_key_value, dense, dense_h_to_4h, dense_4h_to_h
    
    if target_modules is None:
        # 自动从模型结构推断
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    lora_config = LoraConfig(
        r=16,                           # LoRA 秩，r=8~64 常用，越大越强但显存也高
        lora_alpha=32,                  # 缩放因子，通常设为 r 的 2 倍
        target_modules=target_modules,
        lora_dropout=0.05,             # dropout 防止过拟合
        bias="none",                   # 不训练 bias（"all" 会慢且易过拟合）
        task_type=TaskType.CAUSAL_LM,
        
        # 高级参数
        modules_to_save=None,          # 指定额外需要全量更新的模块（如输出层）
        inference_mode=False,
        
        # DoRA（Directional LoRA）— LoRA 的改进版
        use_dora=True,                 # 分解为 magnitude + direction，效果更好
    )
    
    return lora_config


# ============================================================
# Step 3: 数据准备（指令微调格式）
# ============================================================
def prepare_instruction_data(dataset_path, tokenizer, max_length=2048):
    """
    将数据转换为指令微调格式
    格式: <|user|>prompt<|assistant|>response<|eos|>
    """
    
    def format_example(example):
        # chat template 格式
        messages = [
            {"role": "system", "content": example.get("system", "你是一个有帮助的助手。")},
            {"role": "user", "content": example["instruction"] + 
                (f"\n\n输入: {example['input']}" if example.get('input') else "")},
            {"role": "assistant", "content": example["output"]}
        ]
        
        # 用 tokenizer 的 chat_template 格式化
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": text}
    
    # 加载数据集
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    # 格式化
    dataset = dataset.map(
        format_example,
        remove_columns=dataset.column_names,
        desc="Formatting"
    )
    
    # Tokenize
    def tokenize(example):
        result = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None
        )
        result["labels"] = result["input_ids"].copy()
        return result
    
    dataset = dataset.map(tokenize, remove_columns=["text"], desc="Tokenizing")
    
    return dataset


# ============================================================
# Step 4: 训练配置
# ============================================================
def get_training_args(output_dir="./outputs", per_device_train_batch_size=4,
                      gradient_accumulation_steps=4, learning_rate=2e-4,
                      num_train_epochs=3, warmup_ratio=0.03):
    """
    QLoRA 训练关键配置：
    - bf16: 使用 bfloat16 精度
    - gradient checkpointing: 节省显存
    - optim: paged_adamw_32bit（paged 版本处理突发梯度）
    - weight decay: 0.001 防止过拟合
    """
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # 总实际 batch size = 4 × 4 = 16
        
        learning_rate=learning_rate,
        weight_decay=0.001,
        
        num_train_epochs=num_train_epochs,
        
        # 学习率调度
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        
        # 精度与显存
        bf16=True,                        # BFloat16，比 FP16 更稳定
        fp16=False,
        gradient_checkpointing=True,      # 用时间换显存
        gradient_checkpointing_kwargs={"use_reentrant": False},
        
        # 优化器
        optim="paged_adamw_32bit",         # Paged 版本，避免显存峰值
        
        # 日志与保存
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        
        # 其他
        dataloader_num_workers=4,
        remove_unused_columns=False,
        group_by_length=True,             # 相似长度样本放一起，减少 padding
        max_grad_norm=0.3,                # 梯度裁剪，防止梯度爆炸
        
        # 早停（可选）
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        report_to="tensorboard",
    )
    
    return training_args


# ============================================================
# Step 5: 完整训练流程
# ============================================================
def train_qloRA(
    model_name="Qwen/Qwen2-7B-Instruct",
    train_data_path="./data/train.jsonl",
    output_dir="./outputs/qwen2-7b-sft",
    r=16,
    target_modules=None,
):
    """完整的 QLoRA 训练流程"""
    
    print(f"Loading model: {model_name}")
    model, tokenizer = load_model_quantized(model_name)
    
    print("Applying LoRA config...")
    lora_config = get_lora_config(target_modules)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # 打印示例: "trainable params: 83M || all params: 6.7B || 1.24%"
    
    print("Preparing data...")
    train_dataset = prepare_instruction_data(train_data_path, tokenizer)
    # 划分训练/验证集
    split_ds = train_dataset.train_test_split(test_size=0.1, seed=42)
    
    print("Starting training...")
    training_args = get_training_args(output_dir=output_dir)
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=split_ds['train'],
        eval_dataset=split_ds['test'],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        tokenizer=tokenizer,
        max_seq_length=2048,
    )
    
    trainer.train()
    
    # 保存最终模型
    trainer.save_model(f"{output_dir}/final")
    trainer.save_state()
    
    # 合并 LoRA 权重到基础模型（可选，用于推理）
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(f"{output_dir}/merged")
    
    print(f"Training complete! Model saved to {output_dir}")
    
    return model


# ============================================================
# Step 6: 推理使用
# ============================================================
def inference_with_peft(base_model_path, adapter_path, prompt):
    """加载 LoRA 适配器进行推理"""
    from peft import PeftModel
    import transformers
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # 加载 LoRA 适配器
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    output = pipeline(
        text,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    
    return output[0]['generated_text']
```

### 5.5 SFT vs RLHF: How to choose| Method | Data requirements | Training complexity | Effect | Applicable scenarios |
|------|---------|-----------|------|---------|
| **SFT (supervised fine-tuning)** | 1K–10K high-quality question and answer pairs | Low (simple gradient descent) | Basic capability alignment | Vertical domain adaptation, format learning |
| **DPO (Direct Preference Optimization)** | 5K–50K preference pairs | Medium (no Reward model required) | Better aligned with human preferences than SFT | Security improvements, answer quality improvements |
| **PPO-RLHF** | Reward model + preference data | High (requires Reward + PPO) | Strongest but unstable training | Scenarios that require the strongest alignment |
| **KTO (Kahneman-Taversky Optimization)** | Single preference annotation | Medium | More stable than DPO | When annotation cost is limited |

**DPO code example** (much simpler than PPO):

```python
from trl import DPOTrainer
from transformers import AutoModelForCausalLM

def train_dpo(base_model_path, train_data_path, output_dir):
    """
    DPO 训练：不需要 Reward 模型，直接用偏好对优化
    核心思想：正例得分↑，负例得分↓
    
    损失函数：
    L = -log σ( β * (log π_θ(y+) - log π_θ(y-)) - β * (log π_ref(y+) - log π_ref(y-)) )
    """
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
    )
    
    ref_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
    )
    
    # DPO 数据格式：List[{prompt, chosen, rejected}]
    # chosen = 用户喜欢的回答，rejected = 不喜欢的回答
    dpo_dataset = load_dataset("json", data_files=train_data_path)['train']
    
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        beta=0.1,                   # KL 散度系数，0.1~0.3 常用
        train_dataset=dpo_dataset,
        args=TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=1e-5,       # DPO 学习率比 SFT 低
            num_train_epochs=3,
            bf16=True,
            logging_steps=10,
        ),
        tokenizer=tokenizer,
    )
    
    dpo_trainer.train()
```

---

## 6. RAG + Fine-tuning joint optimization

### 6.1 Data Flywheel: Learning from RAG Mistakes

The best architecture is not to choose one from the other, but to let RAG and Fine-tuning enhance each other:

```
RAG 推理 → 低质量回答 → 人工标注/反馈 → 新训练数据 → Fine-tuning → 更好的基座模型 → 更好的 RAG
```

```python
class RAGFineTuningPipeline:
    """RAG + Fine-tuning 联合优化飞轮"""
    
    def __init__(self, rag_pipeline, llm_for_sft, embed_model):
        self.rag = rag_pipeline
        self.llm = llm_for_sft
        self.embed_model = embed_model
        self.feedback_store = []
    
    def collect_and_improve(self, question, rag_answer, user_feedback):
        """
        用户对 RAG 回答的反馈 → 自动收集到训练数据
        """
        if user_feedback == 'thumbs_up':
            return  # 好答案，不用管
        
        if user_feedback == 'thumbs_down':
            # 用户不喜欢 RAG 的回答，收集偏好数据
            # 同时生成一个更好的回答（可以用更慢/更贵的模型）
            better_answer = self.llm.invoke(f"""
用户问：{question}
RAG 给出的回答：{rag_answer}
请给出一个更好的、更准确的回答：
""")
            
            self.feedback_store.append({
                'prompt': question,
                'chosen': better_answer,
                'rejected': rag_answer,
                'feedback_type': 'preference',
                'timestamp': datetime.now()
            })
            
            # 保存为 DPO 训练数据
            self.save_as_dpo_data()
    
    def retrain_periodically(self, batch_size=100):
        """定期用收集的数据微调模型"""
        if len(self.feedback_store) >= batch_size:
            # 过滤高质量反馈（用户明确标注的）
            high_quality = [d for d in self.feedback_store 
                           if d.get('user_verified', False)]
            
            if len(high_quality) >= batch_size:
                print(f"Retraining with {len(high_quality)} samples...")
                train_qloRA(
                    model_name=self.base_model,
                    train_data_path=high_quality,
                    output_dir=f"./checkpoints/{datetime.now().date()}"
                )
                self.feedback_store.clear()
```

### 6.2 Embedding fine-tuning data is automatically generated

Use LLM to automatically generate Hard Negative samples to fine-tune Embedding:

```python
class HardNegativeGenerator:
    """自动生成困难负例，提升 Embedding 模型区分能力"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def generate_triplets(self, positive_chunks):
        """
        为每个正例 chunk 生成困难负例
        
        困难负例 = 语义相关但不是正确答案的 chunk
        （完全无关的 chunk 模型很容易区分，没训练价值）
        """
        triplets = []
        
        for chunk in positive_chunks:
            prompt = f"""
给定的正确文档：
---
{chunk['content']}
---

请生成 3 个"容易混淆但错误"的文档，这些文档：
1. 与正确文档主题相关
2. 包含相似的关键词或表述
3. 但在关键细节上是错误的或不完全正确的

以 JSON 格式输出：
{{
  "negatives": ["错误文档1", "错误文档2", "错误文档3"]
}}
"""
            response = self.llm.invoke(prompt)
            negatives = json.loads(response.content)['negatives']
            
            triplets.append({
                'query': chunk.get('question', chunk['content']),
                'positive': chunk['content'],
                'negatives': negatives
            })
        
        return triplets
```

---

## 7. Production environment deployment and optimization

### 7.1 RAG production architecture

```
用户请求
    ↓
[API Gateway]
    ↓
[Query 预处理]
    ├── 拼写检查 / 同义词替换
    ├── 意图分类（闲聊 / 知识问答 / 任务执行）
    └── 路由（路由到对应知识库）
    ↓
[检索引擎] × N
    ├── 向量数据库（Milvus / Qdrant）
    ├── BM25 倒排索引
    └── 知识图谱（可选）
    ↓
[重排序层]（Cross-Encoder）
    ↓
[LLM 生成]（本地模型 / API）
    ├── Context 组装
    ├── Prompt Template 注入
    └── 生成参数调优
    ↓
[输出校验]（可选）
    ├── 幻觉检测（LLM 自评）
    ├── 引用来源验证
    └── 安全过滤
    ↓
返回用户
```

### 7.2 Fine-tuning model inference optimization

```python
# vLLM: 高吞吐量 LLM 推理框架（支持 LoRA 适配器）
from vllm import LLM, SamplingParams

class VLLMInference:
    """vLLM 推理服务（支持 QLoRA 适配器）"""
    
    def __init__(self, base_model_path, adapter_path=None, tensor_parallel_size=1):
        self.llm = LLM(
            model=base_model_path,
            tokenizer=base_model_path,
            tensor_parallel_size=tensor_parallel_size,  # 多卡并行
            max_model_len=8192,
            gpu_memory_utilization=0.9,
            enforce_eager=False,  # 使用 CUDA graph，加速显著
            
            # LoRA 适配器支持
            enable_lora=True if adapter_path else False,
            lora_modules=["q_proj", "v_proj"],
            lora_weights=adapter_path if adapter_path else None,
        )
    
    def batch_generate(self, prompts, max_tokens=512, temperature=0.7):
        """批量推理（vLLM 支持连续批处理，吞吐率提升 10x+）"""
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_tokens,
            stop=["<|user|>", "<|eos|>"],
        )
        
        outputs = self.llm.generate(prompts, sampling_params)
        
        return [output.outputs[0].text for output in outputs]
```

### 7.3 Evaluation system

**RAG Assessment Metrics** (RAGAS Framework):| Indicators | Meaning | Assessment methods |
|------|------|---------|
| **Faithfulness** | Whether the answer is faithful to the retrieved context | LLM scoring |
| **Answer Relevance** | The relevance of the answer to the question | LLM Rating |
| **Context Precision** | The precision of retrieving context | Relevance weighted sorting |
| **Context Recall** | Whether the context covers the information required for the answer | LLM scoring |
| **Answer Correctness** | Factual correctness of the answer | Comparison with marked answers |

**Fine-tuned evaluation metrics**:

| Indicator | Description |
|------|------|
| Perplexity (PPL) | Language model perplexity, the lower the better |
| Rouge-L | Rouge-L similarity to the reference answer |
| Task accuracy | Accuracy of a specific task (question and answer/classification) |
| Human Evaluation | Win Rate Comparison (A/B Testing) |
| Safety Score | Hazardous output ratio |

---

## 8. Recommended mainstream frameworks and tools| Tasks | Recommended Tools | Description |
|------|---------|------|
| **RAG Framework** | LangChain / LangGraph, LlamaIndex | Quickly build RAG pipeline |
| **Embedding** | sentence-transformers, BAAI/bge | Open source Embedding model |
| **Vector Database** | Milvus, Qdrant | Production grade options |
| **Fine-tuning framework** | LLaMA-Factory, Axolotl, SWIFT | The most complete domestic LLaMA-Factory |
| **RLHF/DPO** | TRL (HuggingFace), DPO-Mirror | HuggingFace Official |
| **Inference Service** | vLLM, SGLang, Text Generation Inference | High-throughput inference |
| **MLOps** | Weights & Biases, MLflow | Experiment Tracking |
| **Evaluation** | RAGAS, BIG-bench, MT-Bench | Multi-dimensional evaluation |

**LLaMA-Factory usage example** (the strongest domestic fine-tuning framework):

```bash
# 一键启动 QLoRA 微调
llamafactory-cli train \
    --stage sft \
    --model_name_or_path Qwen/Qwen2-7B-Instruct \
    --template qwen2 \
    --dataset data/custom_sft.json \
    --cutoff_len 2048 \
    --lora_target qproj,vproj,kproj,proj,o_proj,gate_proj,up_proj,down_proj \
    --quantization_bit 4 \
    --bnb_4bit_compute_dtype bfloat16 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --module_saving_dir ./outputs/lora_qwen2 \
    --output_dir ./outputs/qwen2-7b-finetuned \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --num_train_epochs 3 \
    --bf16 true \
    --prompt_template qwen2
```

---

## 9. Summary: Engineering Decision Guide for RAG and Fine-tuning

**RAG applicable conditions**:
- Knowledge needs to be updated frequently (prices, products, news)
- Need to cite the original text of external documents
- The knowledge base is large but the retrieval frequency is relatively low
- Don’t want to bear the computational cost of fine-tuning

**Applicable conditions for Fine-tuning**:
-Vertical domain knowledge is relatively stable
- Requires a specific output format or tone
- Inference latency and cost are key constraints
- Have sufficient annotated data

**Final recommended architecture**:

```
用户问题
    ↓
[RAG 知识检索] ──→ 提供最新/私有知识上下文
    ↓
[微调模型生成] ──→ 使用微调后的模型（更懂领域语言和格式）
    ↓
[双重校验] ──────→ 用 RAG 检索结果验证生成内容的准确性
    ↓
返回用户
```

This architecture combines the knowledge timeliness of RAG with the quality optimization of Fine-tuning, and is the optimal solution for current engineering practice.

---

**References:**1. Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG).* Conference on Neural Information Processing Systems (NeurIPS).
2. Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* International Conference on Learning Representations (ICLR).
3. Dettmers, T., et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* Conference on Neural Information Processing Systems (NeurIPS).
4. Ouyang, L., et al. (2022). *Training language models to follow instructions with human feedback (InstructGPT).* Conference on Neural Information Processing Systems (NeurIPS).
5. Rafailov, R., et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model (DPO).* Conference on Neural Information Processing Systems (NeurIPS).
6. Lowe, R., et al. (2017). *Multi-agent actor-critic for mixed cooperative-competitive environments (MADDPG).* Conference on Neural Information Processing Systems (NeurIPS).
7. Rashid, T., et al. (2018). *QMIX: Monotonic value function factorisation for deep multi-agent reinforcement learning.* International Conference on Machine Learning (ICML).
8. Veličković, P., et al. (2018). *Graph attention networks.* International Conference on Learning Representations (ICLR).
9. Shah, S., et al. (2018). *AirSim: High-Fidelity Visual and Physical Simulation for Autonomous Vehicles.* Field and Service Robotics (FSR), Springer.
10. Guu, K., et al. (2020). *REALM: Retrieval-Augmented Language Model Pre-Training.* arXiv:2002.08909. *(Preprint, corresponding to ICML 2020 poster)*
11. Borgeaud, S., et al. (2022). *Improving Language Models by Retrieving from Trillions of Tokens.* International Conference on Machine Learning (ICML).
12. Izacard, G., et al. (2022). *Atlas: Few-Shot Learning with Retrieval Augmented Language Models.* Journal of Machine Learning Research (JMLR).
13. Jiang, Z., et al. (2023). *Active Retrieval Augmented Generation.* Conference on Empirical Methods in Natural Language Processing (EMNLP).
14. Asai, A., et al. (2023). *Sakret: Tool-Augmented Language Models for Grounded Reasoning.* Annual Meeting of the Association for Computational Linguistics (ACL).
15. Fan, T., et al. (2020). *Distributed Multi-Robot Collision Avoidance via Deep Reinforcement Learning for Navigation in Complex Scenarios.* The International Journal of Robotics Research (IJRR).
16. Rashid, T., et al. (2018). *QMIX: Monotonic value function factorisation for deep multi-agent reinforcement learning.* International Conference on Machine Learning (ICML).
17. Veličković, P., et al. (2018). *Graph attention networks.* International Conference on Learning Representations (ICLR).