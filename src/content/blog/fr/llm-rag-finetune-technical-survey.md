---
title: "Enquête panoramique sur la base de connaissances LLM RAG et mise au point de la technologie de formation"
description: "Une analyse approfondie de la pile technologique complète de la base de connaissances RAG (récupération/intégration/base de données vectorielles/réorganisation) et le guide complet de réglage fin du LLM (LoRA/QLoRA/full fine-tuning/SFT/RLHF), de la conception de l'architecture à la mise en œuvre du projet, avec comparaison des frameworks traditionnels et suggestions de sélection."
pubDate: 2026-04-07T11:43:17+08:00
tags: ["grand modèle de langage", "CHIFFON", "base de connaissances", "base de données de vecteurs", "réglage fin", "LoRA", "QLoRA", "SFT", "RLHF", "Intégration", "LLM"]
category: Tech
sourceHash: "2860624c046e8bbdb4c91f6836f0348410230fbc"
---

# LLM RAG Enquête panoramique sur la base de connaissances et la mise au point des technologies de formation

> Deux voies principales pour créer des applications de grands modèles au niveau de l'entreprise : **RAG (Retrieval Enhanced Generation)** et **Fine-tuning**. Le premier permet au modèle de « lire des milliers de livres », et le second permet au modèle « d'acquérir de nouvelles compétences ». Cet article relie complètement les deux voies : depuis la construction de la base de connaissances de RAG, la stratégie de récupération, la génération de réponses, jusqu'à l'ensemble du processus de pré-formation/mise au point des instructions/SFT/RLHF, avec la dernière comparaison de cadres, les notes sur les pièges techniques et les suggestions de sélection. Que vous souhaitiez effectuer des questions et réponses de connaissances privées, une adaptation de champ vertical ou laisser le modèle apprendre un format de sortie spécifique, cette enquête peut vous donner une carte technique complète.

---

## 1. Présentation de RAG : Pourquoi avez-vous besoin d'une amélioration de la récupération ?

### 1.1 Dilemme de connaissances du LLM

Les grands modèles de langage présentent trois limitations inhérentes :

| Type de restriction | Performances | Cas typiques |
|---------|------|---------|
| **Seuil de connaissances** | Les données d'entraînement ont une date limite | Date limite des connaissances GPT-4 Turbo 2024-06 |
| **Hallucination** | De sérieuses absurdités sur des connaissances incertaines | Termes juridiques fictifs et paramètres du produit |
| **Impossible d'accéder aux données privées** | Les documents et bases de données internes de l'entreprise ne sont pas accessibles au public | Rapports financiers, base de connaissances du service client, base de code |
| **Manque de connaissances à longue traîne** | Connaissances éparses dans des domaines impopulaires | Terminologie spécifique à l'industrie, technologies auto-développées |
| **Difficulté à mettre à jour les connaissances** | Les nouvelles connaissances nécessitent un recyclage | Prix ​​du jour, inventaire en temps réel |

**Valeur fondamentale de RAG** : "Armer" LLM en récupérant des connaissances externes sans modifier les poids du modèle - non seulement résout le problème de l'actualité des connaissances, évite les illusions, mais prend également naturellement en charge les données privées.

### 1.2 Trois générations d'évolution de l'architecture RAG

**Première génération : Naive RAG (2020-2023)**

```
用户问题 → 向量化 → Top-K 检索 → 拼接上下文 → LLM 生成
```

Le processus est simple et direct, mais les problèmes sont également évidents : mauvaise qualité de récupération, utilisation insuffisante des fenêtres contextuelles et contenu généré qui n'est pas en phase avec les résultats de la récupération.

**Deuxième génération : RAG avancé (2023-2024)**

```
用户问题 → Query 改写/扩展 → 向量化 → 检索 → 重排序 → LLM 生成
         ↑                                   ↑
      HyDE 假设文档                 Cross-Encoder 重排序
```Points d'amélioration : Réécrivez la requête avant la récupération (HyDE, Query Expansion), et réorganisez les résultats après la récupération (Cross-Encoder / BM25 + Vector Hybrid), améliorant considérablement le taux de rappel et la précision de la récupération.

**Troisième génération : RAG modulaire (2024–)**

```
用户问题 → 路由 → 工具调用 → 检索 → 后处理 → LLM 生成
                      ↑
              知识图谱 / Web 搜索 / 计算器 / API
```

Architecture modulaire : la recherche devient un outil enfichable et le routeur décide quand effectuer la recherche, quoi rechercher et quel outil utiliser. Travaux représentatifs : NeME, Self-RAG, Corrective-RAG (CRAG).

---

## 2. L'ensemble du processus de création de la base de connaissances RAG

### 2.1 Analyse de documents et extraction de texte

La première étape d’une base de connaissances consiste à transformer des documents bruts en texte clair. La difficulté de traiter différents formats est très variable :

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

**Traitement particulier des documents d'entreprise** :

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

### 2.2 Stratégie de segmentation du texte (Chunking)

Le chunking est l’aspect le plus critique et le plus négligé de RAG. La stratégie de segmentation affecte directement la qualité de la récupération et l'effet de génération.

**Principes fondamentaux** :
- **Complétude sémantique** : essayez de laisser chaque morceau exprimer une unité sémantique complète
- **Contrôle de longueur** : Limité par la fenêtre contextuelle LLM et la limite supérieure du jeton du modèle d'intégration
- **Conception de chevauchement** : conservez le chevauchement entre les morceaux adjacents pour éviter la perte d'informations sur les limites.

#### Stratégie 1 : chunking de longueur fixe (le plus simple)

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

#### Stratégie 2 : couplage récursif des caractères (en maintenant les limites sémantiques)

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

#### Stratégie 3 : Semantic chunking (basé sur LLM/Embedding)

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

#### Stratégie 4 : blocage adaptatif de domaine

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

**Tableau de référence de taille de bloc :**| Scénario | Taille de morceau recommandée | Descriptif |
|------|--------------|------|
| Document général | 500 à 1 000 jetons | Équilibrer l'exhaustivité sémantique et l'exactitude de la récupération |
| Base de code | 200 à 500 jetons | Par niveau de fonction/classe, en préservant le contexte d'appel |
| Document/Rapport | 1 000 à 2 000 jetons | Les longs paragraphes nécessitent une grande fenêtre pour être compris |
| Courtes questions et réponses | 100 à 200 jetons | Correspondance exacte pour éviter les interférences contextuelles non pertinentes |
| Mentions légales | 500 à 800 jetons | Un terme unique est la plus petite unité |
| Multimodal (PDF) | Morceaux séparés pour les tableaux/images | Markdown pour les tableaux et descriptions pour les images |

### 2.3 Métadonnées et graphique de connaissances

**La valeur des métadonnées** : joindre des informations descriptives à chaque bloc améliore considérablement la précision de la récupération et les capacités de filtrage.

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

**RAG amélioré du graphique de connaissances** :

Extrayez les entités et les relations du texte, créez un graphe de connaissances et récupérez le graphe et les vecteurs en même temps :

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

## 3. Modèle d'intégration et base de données vectorielles

### 3.1 Sélection du modèle d'intégration

Le modèle Embedding est la « couche de perception » de RAG – sa qualité détermine directement le rappel et la précision de la récupération.

**Comparaison des modèles d'intégration traditionnels en 2024-2025 :**| Modèle | Dimensions | Contexte | Précision MTEB | Scénarios d'avantages | GitHub ⭐ |
|------|------|--------|---------|---------|---------|
| **text-embedding-3-large** (OpenAI) | 3072 | 8191 | ~66% | Universel/Anglais | - |
| **text-embedding-3-small** (OpenAI) | 1536 | 8191 | ~62% | Sensible aux coûts | - |
| **e5-mistral-7b-instruct** (Microsoft) | 4096 | 4096 | ~66% | Plusieurs langues/instructions | 10 000+ |
| **bge-large-zh-v1.5** (BAAI) | 1024 | 512 | ~64% | Principalement chinois | 20 000+ |
| **bge-m3** (BAAI) | 1024 | 8192 | ~65% | Recherche multilingue/hybride | 8k+ |
| **GTE-Qwen2-7B-instruct** (Alibaba) | 1024 | 8192 | ~67% | Chinois/Anglais | 5k+ |
| **NV-Embed-v2** (NVIDIA) | 4096 | 128 Ko | ~69% | Contexte long | - |
| **Cohere-embed-v3** | 1024 | 512 | ~65% | Anglais/Multilingue | - |
| **GritLM-7B** (mélange d'intégration + LLM) | 4096 | 8K | ~67% | embedding+ génère l'unité | 3k+ |

**Recommandation de scène chinoise** : `bge-large-zh-v1.5` ou `GTE-Qwen2-7B-instruct`

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

**Ajustement précis du modèle d'intégration** (facultatif, améliore considérablement l'effet dans des zones spécifiques) :

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

### 3.2 Sélection et utilisation de la base de données vectorielles

La base de données vectorielles est chargée de stocker les intégrations et d'effectuer des recherches du voisin le plus proche (ANN).

**Comparaison des bases de données vectorielles traditionnelles :**| Base de données | Algorithmes | Types d'index | Prise en charge du filtrage | Déploiement | Échelle | Latence | Capacités spéciales |
|--------|------|---------|---------|------|------|------|----------|
| **Milvus** | HNSW / FIV / DiskANN | Hybride | ✅Autochtone | K8 / Docker | Niveau milliard | Microseconde | Filtrage fort des métadonnées |
| **Qdrant** | HNSW / DiskANN | Hybride | ✅Autochtone | Autonome/K8 | Niveau milliard | Microseconde | Reprise des scores |
| **Tisser** | HNSW | Hybride | ✅Autochtone | K8 / Nuage | Des milliards | Microsecondes | Interface GraphQL |
| **Chrome** | HNSW (approximatif) | approximatif | ✅ Python | Autonome | Millions | Millisecondes | Le plus simple à utiliser |
| **Pomme de pin** | — (Nuage) | — | ✅ Géré | Entièrement géré | Des milliards | Microsecondes | Entièrement géré |
| **FAISS** | HNSW / FIV | Exact + approximatif | ⚠️ Nécessite un traitement supplémentaire | Intégré | Millions/machine unique | Microsecondes | GPU accéléré |
| **Elasticsearch** (8.0+) | HNSW | Hybride | ✅Autochtone | K8 / Nuage | Niveau milliard | Millisecondes | Texte intégral + vecteur hybride |
| **pgvecteur** (PostgreSQL) | HNSW / IVFFlat | Hybride | ✅SQL | K8 / Docker | Des milliards | Millisecondes | Requête d'union SQL |

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

### 3.3 Stratégie de récupération hybride

Limites de la récupération d'un seul vecteur : les requêtes avec des synonymes différents mais une sémantique similaire peuvent échouer.

**Recherche hybride = fusion de récupération dense (vecteur) + clairsemée (BM25)**

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

## 4. Compréhension des requêtes et optimisation de la récupération

### 4.1 Technologie de réécriture de requêtesIl existe souvent des différences lexicales entre les questions des utilisateurs et les représentations dans la base de connaissances. La réécriture de requêtes est le module principal d'Advanced RAG.

**HyDE (incorporations de documents hypothétiques)** :

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

**Récupération multi-requêtes (extension multi-requêtes)** :

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

**Invite de recul** :

Pour les questions qui nécessitent un raisonnement abstrait, extrayez d’abord les concepts de haut niveau, puis récupérez :

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

### 4.2 Reclassement

Les candidats Top-K récupérés sont ensuite classés via un modèle de reclassement, plaçant les résultats les plus pertinents au sommet.

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

**Processus RAG avancé complet** :

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

## 5. Mise au point du LLM : guide technique complet du processus

### 5.1 Quand faut-il procéder au réglage fin ?

Le choix entre RAG et Fine-tuning est la question la plus courante dans les décisions d’ingénierie :

| Scénario | Solution recommandée | Raison |
|------|---------|------|
| Base de connaissances Q&A (connaissances fréquemment mises à jour) | **CHIFFON** | La mise au point ne peut pas suivre le rythme des mises à jour des connaissances |
| Besoin de référencer des documents externes | **CHIFFON** | Le modèle de réglage fin ne peut pas accéder aux documents externes |
| Apprendre de nouveaux formats/styles de sortie | **Réglage fin** | Le format et le ton doivent être internalisés dans les pondérations |
| Comprendre la terminologie dans les domaines verticaux | **Réglage fin** | Un grand nombre de concepts spécifiques à un domaine doivent être internalisés |
| Réduire le coût de latence/d'inférence | **Réglage fin** | Les modèles affinés peuvent utiliser de petits modèles |
| Corriger des modèles d'erreur spécifiques | **Réglage fin** | Les erreurs récurrentes nécessitent des correctifs fondamentaux |
| Nécessite plusieurs tours de style de conversation | **Réglage fin** | Le style de conversation et la personnalité doivent être intériorisés |

**Meilleure pratique (la plupart des scénarios)** : **RAG + réglage fin combinés**.
- RAG est responsable de l'exactitude et de l'actualité des connaissances
- Le réglage fin est responsable de l'optimisation du style, du format et du mode de raisonnement### 5.2 Panorama des méthodes de réglage fin

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

### 5.3 Réglage fin complet par rapport au PEFT

| Dimensions | Mise au point complète | LoRA/QLoRA |
|------|------|-------------|
| Exigences en matière de mémoire vidéo | Modèle 70B environ 140 Go (FP16) | Modèle 70B environ 35 Go (QLoRA 4 bits) |
| Temps de formation | Des dizaines d'heures/jour | Plusieurs heures |
| Coût de stockage | Ensemble complet de poids par tâche | Seuls les poids d'adaptateur stockés par tâche |
| Oubli catastrophique | Sévère | Léger (seuls quelques paramètres sont mis à jour) |
| Limite supérieure d'effet | Supérieur (paramètres plus apprenables) | Légère baisse mais l'écart se réduit |
| Exigences matérielles | A100 80G × plusieurs cartes | Une seule carte A100 peut exécuter 70B |

**Conclusion** : Après 2024, **QLoRA** est devenu le standard de facto : il ramène le réglage fin du modèle 70B au niveau de mémoire vidéo accessible par une seule carte.

### 5.4 Analyse approfondie de QLoRA

Combinaison de **QLoRA = Quantification + LoRA** :

1. **Quantisation NormalFloat (NF4) 4 bits** : compressez les poids de pré-entraînement à 4 bits avec une perte de précision minimale
2. **Double Quant** : requantifiez la constante de quantification elle-même pour économiser davantage la mémoire vidéo
3. **Paged Optimizer** : échangez automatiquement les pages entre le CPU et le GPU lors de la gestion des rafales de mise à jour en dégradé

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

### 5.5 SFT vs RLHF : Comment choisir| Méthode | Exigences en matière de données | Complexité de la formation | Effet | Scénarios applicables |
|------|---------|-----------|------|---------|
| **SFT (mise au point supervisée)** | 1 000 à 10 000 paires de questions et réponses de haute qualité | Faible (simple descente de pente) | Alignement des capacités de base | Adaptation de domaine vertical, apprentissage de format |
| **DPO (Optimisation des préférences directes)** | Paires de préférences 5K-50K | Moyen (aucun modèle de récompense requis) | Mieux aligné sur les préférences humaines que SFT | Améliorations de la sécurité, améliorations de la qualité des réponses |
| **PPO-RLHF** | Modèle de récompense + données de préférence | Élevé (nécessite une récompense + PPO) | Formation la plus forte mais instable | Scénarios qui nécessitent l'alignement le plus fort |
| **KTO (optimisation Kahneman-Taversky)** | Annotation de préférence unique | Moyen | Plus stable que DPO | Quand le coût des annotations est limité |

**Exemple de code DPO** (beaucoup plus simple que PPO) :

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

## 6. RAG + Affiner l'optimisation des articulations

### 6.1 Volant de données : apprendre des erreurs RAG

La meilleure architecture n’est pas de choisir l’une parmi l’autre, mais de laisser RAG et Fine-tuning s’améliorer mutuellement :

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

### 6.2 L'intégration des données de réglage fin est automatiquement générée

Utilisez LLM pour générer automatiquement des échantillons Hard Negative afin d’affiner l’intégration :

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

## 7. Déploiement et optimisation de l'environnement de production

### 7.1 Architecture de production RAG

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

### 7.2 Affiner l'optimisation de l'inférence du modèle

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

### 7.3 Système d'évaluation

**Mesures d'évaluation RAG** (cadre RAGAS) :| Indicateurs | Signification | Méthodes d'évaluation |
|------|------|--------------|
| **Fidélité** | Si la réponse est fidèle au contexte récupéré | Notation LLM |
| **Pertinence de la réponse** | La pertinence de la réponse à la question | Classement LLM |
| **Précision du contexte** | La précision de la récupération du contexte | Tri pondéré par pertinence |
| **Rappel de contexte** | Si le contexte couvre les informations requises pour la réponse | Notation LLM |
| ** Exactitude de la réponse ** | Exactitude factuelle de la réponse | Comparaison avec les réponses marquées |

**Mesures d'évaluation affinées** :

| Indicateur | Descriptif |
|------|------|
| Perplexité (PPL) | Perplexité du modèle linguistique, plus il est faible, mieux c'est |
| Rouge-L | Similitude Rouge-L avec la réponse de référence |
| Précision des tâches | Précision d'une tâche spécifique (question et réponse/classification) |
| Évaluation humaine | Comparaison des taux de victoire (tests A/B) |
| Score de sécurité | Taux de production dangereuse |

---

## 8. Cadres et outils grand public recommandés| Tâches | Outils recommandés | Descriptif |
|------|---------|------|
| **Cadre RAG** ​​| LangChain / LangGraph, LlamaIndex | Créez rapidement un pipeline RAG |
| **Intégration** | transformateurs de phrases, BAAI/bge | Modèle d'intégration open source |
| **Base de données vectorielles** | Milvus, Qdrant | Options de qualité de production |
| **Cadre de réglage fin** | LLaMA-Factory, Axolotl, SWIFT | L'usine LLaMA domestique la plus complète |
| **RLHF/DPO** | TRL (HuggingFace), DPO-Miroir | HuggingFace Officiel |
| **Service d'inférence** | vLLM, SGLang, inférence de génération de texte | Inférence à haut débit |
| **MLOps** | Poids et biais, MLflow | Suivi des expériences |
| **Évaluation** | RAGAS, BIG-banc, MT-Banc | Évaluation multidimensionnelle |

**Exemple d'utilisation de LLaMA-Factory** (le cadre de réglage fin national le plus puissant) :

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

## 9. Résumé : Guide de décision d'ingénierie pour RAG et réglage fin

**Conditions applicables du RAG** :
- Les connaissances doivent être mises à jour fréquemment (prix, produits, actualités)
- Nécessité de citer le texte original de documents externes
- La base de connaissances est vaste mais la fréquence de récupération est relativement faible
- Je ne veux pas supporter le coût de calcul du réglage fin

**Conditions applicables pour le réglage fin** :
-La connaissance du domaine vertical est relativement stable
- Nécessite un format de sortie ou une tonalité spécifique
- La latence et le coût d'inférence sont des contraintes clés
- Avoir suffisamment de données annotées

**Architecture finale recommandée** :

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

Cette architecture combine l'actualité des connaissances de RAG avec l'optimisation de la qualité du réglage fin et constitue la solution optimale pour les pratiques d'ingénierie actuelles.

---

**Références :**1. Lewis, P. et coll. (2020). *Génération augmentée par récupération pour les tâches PNL à forte intensité de connaissances (RAG).* Conférence sur les systèmes de traitement de l'information neuronale (NeurIPS).
2. Hu, E.J. et al. (2021). *LoRA : Adaptation de bas rang de grands modèles de langage.* Conférence internationale sur les représentations d'apprentissage (ICLR).
3. Dettmers, T. et coll. (2023). *QLoRA : Efficient Finetuning of Quantized LLMs.* Conférence sur les systèmes de traitement de l'information neuronale (NeurIPS).
4. Ouyang, L., et al. (2022). *Formation de modèles de langage pour suivre les instructions avec retour humain (InstructGPT).* Conférence sur les systèmes de traitement de l'information neuronale (NeurIPS).
5. Rafailov, R., et al. (2023). *Optimisation directe des préférences : votre modèle linguistique est secrètement un modèle de récompense (DPO).* Conférence sur les systèmes de traitement de l'information neuronale (NeurIPS).
6. Lowe, R. et coll. (2017). *Acteur-critique multi-agents pour les environnements mixtes coopératifs-compétitifs (MADDPG).* Conférence sur Neural Insystèmes de traitement de formation (NeurIPS).
7. Rashid, T. et coll. (2018). *QMIX : Factorisation de fonctions de valeur monotones pour un apprentissage par renforcement multi-agents approfondi.* Conférence internationale sur l'apprentissage automatique (ICML).
8. Veličković, P., et al. (2018). *Réseaux d'attention graphique.* Conférence internationale sur les représentations d'apprentissage (ICLR).
9. Shah, S., et coll. (2018). *AirSim : simulation visuelle et physique haute fidélité pour les véhicules autonomes.* Field and Service Robotics (FSR), Springer.
10. Guu, K. et al. (2020). *REALM : pré-formation sur le modèle de langage augmenté par récupération.* arXiv : 2002.08909. *(Préimpression, correspondant à l'affiche ICML 2020)*
11. Borgeaud, S., et al. (2022). *Améliorer les modèles linguistiques en récupérant des milliards de jetons.* Conférence internationale sur l'apprentissage automatique (ICML).
12. Izacard, G., et al. (2022). *Atlas : Apprentissage en quelques étapes avec récupération de modèles de langage augmentés.* Journal of Machine Learning Research (JMLR).
13. Jiang, Z. et coll. (2023). *Active Retrieval Augmented Generation.* Conférence sur les méthodes empiriques de traitement du langage naturel (EMNLP).
14. Asai, A. et coll. (2023). *Sakret : Modèles linguistiques augmentés par des outils pour un raisonnement fondé.* Réunion annuelle de l'Association for Computational Linguistics (ACL).
15. Fan, T. et coll. (2020). *Évitement distribué des collisions multi-robots via un apprentissage par renforcement profond pour la navigation dans des scénarios complexes.* The International Journal of Robotics Research (IJRR).
16. Rashid, T. et coll. (2018). *QMIX : Factorisation de fonctions de valeur monotones pour un apprentissage par renforcement multi-agents approfondi.* Conférence internationale sur l'apprentissage automatique (ICML).
17. Veličković, P., et al. (2018). *Réseaux d'attention graphique.* Conférence internationale sur les représentations d'apprentissage (ICLR).