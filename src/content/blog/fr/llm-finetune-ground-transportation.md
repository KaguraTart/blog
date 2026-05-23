---
title: "Pratique de mise au point LLM : créer un modèle professionnel à grande échelle pour le transport terrestre"
description: "De la construction des données à la mise au point du déploiement, nous vous apprendrons étape par étape comment utiliser LoRA/QLoRA pour affiner le LLM open source afin de créer un modèle expert dans le domaine des transports."
pubDate: 2026-04-25
tags: ['LLM', '微调', '地面交通', 'LoRA', 'QLoRA']
category: Tech
---

##Présentation

Le système de transport terrestre implique de nombreux aspects tels que le contrôle des signaux, la planification des itinéraires, la simulation du trafic, la prévision des accidents, etc. Les méthodes traditionnelles basées sur des règles ou d'apprentissage superficiel présentent des goulots d'étranglement évidents dans les capacités de généralisation et la compréhension du contexte. La mise au point d'un grand modèle linguistique (LLM) en tant qu'expert dans le domaine des transports permet :

- **Questions et réponses interactives sur le trafic en langage naturel** : utilisez le langage naturel pour interroger les conditions de circulation et les causes des embouteillages
- **Génération de description de simulation de scène de trafic** : Générez des fichiers de configuration de simulation en fonction de la situation du trafic
- **Raisonnement sur la stratégie de contrôle du signal** : donnez des suggestions de synchronisation du signal basées sur des données de séries chronologiques
- **Analyse automatique et synthèse des rapports d'accidents** : Extrayez les informations clés d'un grand nombre de dossiers d'accidents

Cet article utilise la série open source LLaMA/Qwen comme modèle de base pour présenter le processus complet de construction de zéro des ensembles de données de trafic jusqu'au déploiement de réglage fin de LoRA.

## 1. Construction des données : ensemble de données sur le domaine du transport

### 1.1 Source de données

| Type de données | Source | Descriptif |
|--------------|------|------|
| Nouvelles/rapports sur le trafic | API de trafic AutoNavi/Baidu, données publiques de la police de la circulation à divers endroits | Faible difficulté d'annotation, grand volume |
| Description des accidents de la route | Rapport d'accident, 122 données de désensibilisation d'enregistrement d'alarme | Annotation professionnelle requise |
| Plan de synchronisation des signaux | Données du service de police de la circulation ou de la plateforme de simulation | Extraction structurée requise |
| Description de la scène de simulation de trafic | Exportation depuis un logiciel de simulation tel que SUMO/Paramics | Haute valeur, professionnalisme fort |
| Articles universitaires sur les transports (résumé) | Journaux de conférences sur le domaine des transports (TRR, IEEE ITS) | Pré-formation continue possible |

### 1.2 Nettoyage et annotation des données

```python
# 示例：使用 LLM 做初步数据清洗与标注
import json

def clean_and_label_traffic_data(raw_text: str, label_model):
    """利用 LLM 对原始交通文本进行实体和意图标注"""
    prompt = f"""你是一位交通工程专家。请对以下文本进行标注，返回 JSON 格式：
    {{
        "entities": ["地点", "事件类型", "拥堵程度"...],
        "intent": "查询/控制/分析/报告",
        "summary": "一句话概括",
        "traffic_terms": ["信号灯", "交叉口", "拥堵"... ]
    }}

    文本：{raw_text}
    """
    response = label_model.chat(prompt)
    return json.loads(response)
```

### 1.3 Format de données de réglage des instructions

Il est recommandé d'utiliser le format `alpaca` ou `sharegpt` :

```json
{
  "instruction": "北京晚高峰东三环严重拥堵，有什么疏导建议？",
  "input": "当前路况：东三环双向车速 < 15km/h，持续时间 > 40min",
  "output": "根据当前态势，建议以下疏导方案：1) 将国贸桥至长虹桥段信号配时调整为..." 
}
```

### 1.4 Correspondance des données

Un ensemble de données de réglage fin efficace pour le domaine du trafic contient généralement :

- **Conservation générale des capacités (20 %)** : conservez certaines données générales d'assurance qualité, de résumé et de traduction pour éviter tout oubli catastrophique.
- **Injection de connaissances sur le domaine (50%)** : questions-réponses sur les connaissances professionnelles en matière de trafic, contrôle des signaux, planification des chemins
- **Données orientées tâches (30%)** : génération de configurations de simulation, analyse de rapports, raisonnement d'incidents de circulation

## 2. Sélection des méthodes de réglage fin

### 2.1 Réglage fin complet des paramètres vs réglage fin efficace des paramètres| Méthode | Montant du paramètre | Exigences en matière de mémoire vidéo | Effet | Scénarios applicables |
|------|--------|---------|------|---------|
| Réglage fin des paramètres complets (Full FT) | 100% | A100 80G × Multicarte | Meilleur | Puissance de calcul suffisante |
| LoRA | 0,1% ~ 1% | Une seule carte 24G est réalisable | Près de la référence complète | Choix grand public |
| QLoRA | 0,1 % ~ 1 % (4 bits) | Carte unique 16G réalisable | Légèrement inférieur à LoRA | Ressources limitées |
| Réglage de l'adaptateur | 1%~5% | Carte unique 24G | Moyen | De nombreux changements de tâches |

**Recommandation** : privilégiez l'utilisation de **QLoRA**, qui permet de compléter le réglage fin du modèle 7B avec 16 Go de mémoire vidéo.

### 2.2 Principes fondamentaux de LoRA

LoRA ajoute un adaptateur de bas rang à côté de la matrice Q/K/V/O de la couche d'attention :

```python
# LoRA 核心公式
# W_orig: 原始预训练权重 (d × d)
# W_new = W_orig + ΔW = W_orig + BA
# B ∈ R^(d×r), A ∈ R^(r×d), r << d (低秩)
# 训练时只更新 B 和 A，冻结 W_orig

import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, original_layer: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        d_out, d_in = original_layer.weight.shape
        self.original = original_layer
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        self.scaling = alpha / rank
        
    def forward(self, x):
        return self.original(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
```

### 2.3 Conception de l'adaptateur de champ de circulation

Pour les scénarios de trafic, vous pouvez vous concentrer sur l’ajout de LoRA aux couches suivantes :

```
decoder_block.attn.q_proj      ← 语义理解（query 理解交通场景描述）
decoder_block.attn.k_proj      ← 关键实体（key 关联交通要素）
decoder_block.attn.v_proj      ← 知识记忆（value 存储交通规则）
decoder_block.mlp.gate_proj    ← 领域知识凝练
```

## 3. Affiner le code réel

### 3.1 Préparation de l'environnement

```bash
pip install transformers peft accelerate bitsandbytes trl
# 或使用 AxT >= 0.1 的统一命令：
pip install autotorch
```

### 3.2 Script de réglage fin de QLoRA

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 加载基座模型
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B-Instruct",  # 或 LLaMA-3-8B
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model = prepare_model_for_kbit_training(model)

# LoRA 配置 — 针对交通领域重点调参
lora_config = LoraConfig(
    r=16,                      # 秩，越大效果越好但显存更高
    lora_alpha=32,             # 缩放因子
    target_modules=[            # 重点微调的模块
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 可训练参数: ~0.41% (仅 8.9M 参数)

# 训练配置
from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir="./checkpoints/traffic-llm",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,    # 有效 batch = 4×4=16
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=False,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
)
```

### 3.3 Extension du vocabulaire spécial dans le domaine des transports

S'il existe un grand nombre de termes professionnels dans le domaine des transports (tels que « zone d'entrelacement », « phase du signal », « ceinture d'ondes vertes »), il est recommandé d'élargir d'abord le vocabulaire puis d'affiner :

```python
from transformers import Tokenizer
from sentencepiece import sentencepiece_model_pb2

# 1. 收集交通术语
traffic_terms = [
    "信号相位", "绿波带", "交织区", "排队长度",
    "饱和流率", "延误时间", "路口渠化", "可变车道",
    "感应控制", "自适应配时", "区域协调控制"
]

# 2. 使用 sentencepiece 增量训练扩展词表
# 3. 用扩展词表重新 tokenize 数据集并微调
```

## 4. Compétences de formation et pièges

### 4.1 Protection contre l'oubli catastrophique

Un oubli catastrophique (capacité générale d'oubli du modèle) est susceptible de se produire lors du réglage fin complet des paramètres. Solution :

1. **Mélanger les données communes (20 %)** : conservez toujours certaines données d'assurance qualité communes
2. **Utilisez la planification « gamma »** : reportez-vous à la décongélation progressive de SwissArmyKnife
3. **Définissez un taux d'apprentissage plus petit** : généralement 1/10 à 1/100 de la pré-formation

### 4.2 Problème d'illusion dans le domaine des transports

Le domaine des transports a des exigences élevées en matière de factualité, et LLM a tendance à dire des bêtises sérieusement. Contre-mesures :

```python
# RAG 增强：结合交通知识库检索
def traffic_rag_query(query: str, knowledge_base, llm):
    docs = knowledge_base.similarity_search(query, k=3)
    context = "\n".join([d.content for d in docs])
    
    prompt = f"""基于以下交通知识回答问题。如果信息不足，据实说明不知道。
    
    知识：{context}
    问题：{query}"""
    return llm.chat(prompt)
```

### 4.3 Extension multimodaleSi vous avez besoin de traiter des images de trafic (photos de route, captures d'écran de simulation), vous pouvez accéder aux séries **LLaVA** ou **Qwen-VL** pour un réglage fin multimodal :

```python
# 以 LLaVA 为例：图像 + 交通文本联合微调
from llava.model.builder import load_pretrained_model
model, tokenizer, image_processor = load_pretrained_model("liuhaotian/llava-v1.6-7b")
```

## 5. Évaluation et déploiement

### 5.1 Indicateurs d'évaluation

| Tâche | Paramètres d'évaluation | Descriptif |
|------|---------|------|
| Questions et réponses sur les transports | BLEU-4 / ROUGE-L | Comparez avec les réponses d'experts |
| Recommandations de synchronisation du signal | Fonction de notation personnalisée (taux de réduction des délais) | Nécessite une vérification de l'environnement de simulation |
| Résumé du rapport d'incident | BERTScore / G-Eval | Évaluation du niveau sémantique |
| Test de connaissance du domaine | Référentiel de transport (auto-construit) | Questions à choix multiples/fausses/fausses |

### 5.2 Exemples de questions d'évaluation typiques

```json
{
  "question": "在某交叉口，晚高峰期间左转流量很大（280 pcu/h），对向直行流量为 600 pcu/h，
              该交叉口采用两相位控制，配时为 G40-Y3-R37。建议如何优化？",
  "expected": "建议增加左转专用相位或将配时调整为 G55-Y3-R32，以提高左转通行能力..."
}
```

### 5.3 Déploiement

```python
# 使用 vLLM 部署推理服务
from vllm import LLM, SamplingParams

llm = LLM(
    model="./checkpoints/traffic-llm/checkpoint-300",
    tensor_parallel_size=1,        # 单卡部署
    gpu_memory_utilization=0.90,
)
params = SamplingParams(temperature=0.3, max_tokens=512)

output = llm.generate(
    "晚高峰北京西直门桥拥堵，分析原因并给出疏导建议",
    params
)
print(output[0].outputs[0].text)
```

## 6. Direction avancée

- **Collaboration multi-agents** : divisez le contrôle des signaux, la planification des chemins et la planification de la simulation en plusieurs collaborations d'agents
- **Fusion de données de séries chronologiques** : combiné avec l'entrée de données de trafic GeoJSON/CSV pour obtenir un raisonnement hybride texte + données
- **Affinement continu** : utiliser régulièrement de nouvelles données pour effectuer SFT afin de maintenir l'actualité du modèle

---

*Si vous avez des questions ou des suggestions, n'hésitez pas à laisser un message sur le blog ! *