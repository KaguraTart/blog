---
title: "LLM-Feinabstimmungspraxis: Erstellung eines professionellen Großmodells für den Bodentransport"
description: "Von der Datenkonstruktion bis zur Feinabstimmung der Bereitstellung zeigen wir Ihnen Schritt für Schritt, wie Sie mit LoRA/QLoRA Open-Source-LLM optimieren und ein Expertenmodell im Transportbereich erstellen."
pubDate: 2026-04-25
tags: ['LLM', '微调', '地面交通', 'LoRA', 'QLoRA']
category: Tech
sourceHash: "e3866339bf46e05c8aa1b79ce5655fb7c4adcd60"
---

## Einführung

Das Bodentransportsystem umfasst viele Aspekte wie Signalsteuerung, Wegplanung, Verkehrssimulation, Unfallvorhersage usw. Herkömmliche regelbasierte oder flache Lernmethoden weisen offensichtliche Engpässe bei den Generalisierungsfähigkeiten und dem Kontextverständnis auf. Die Feinabstimmung eines großen Sprachmodells (LLM) als Transportdomänenexperte ermöglicht Folgendes:

- **Interaktive Fragen und Antworten zum Verkehr in natürlicher Sprache**: Verwenden Sie natürliche Sprache, um Verkehrsbedingungen und Stauursachen abzufragen
- **Generierung einer Verkehrsszenensimulationsbeschreibung**: Generieren Sie Simulationskonfigurationsdateien basierend auf der Verkehrssituation
- **Begründung der Signalsteuerungsstrategie**: Geben Sie Signal-Timing-Vorschläge basierend auf Zeitreihendaten
- **Automatische Analyse und Zusammenfassung von Unfallberichten**: Extrahieren Sie wichtige Informationen aus einer großen Anzahl von Unfallaufzeichnungen

In diesem Artikel wird die Open-Source-Serie LLaMA/Qwen als Basismodell verwendet, um den gesamten Prozess von der Grundkonstruktion von Verkehrsdatensätzen bis zur LoRA-Feinabstimmung der Bereitstellung vorzustellen.

## 1. Datenkonstruktion: Transportfelddatensatz

### 1.1 Datenquelle

| Datentyp | Quelle | Beschreibung |
|---------|------|------|
| Verkehrsnachrichten/-berichte | AutoNavi/Baidu-Verkehrs-API, öffentliche Daten der Verkehrspolizei an verschiedenen Orten | Geringe Annotationsschwierigkeit, großes Volumen |
| Beschreibung eines Verkehrsunfalls | Unfallbericht, 122 Alarmaufzeichnungs-Desensibilisierungsdaten | Professionelle Anmerkung erforderlich |
| Signalzeitplan | Daten der Verkehrspolizeibehörde oder Simulationsplattform | Strukturierte Extraktion erforderlich |
| Beschreibung der Verkehrssimulationsszene | Export aus Simulationssoftware wie SUMO/Paramics | Hoher Wert, starke Professionalität |
| Wissenschaftliche Arbeiten zum Thema Transport (Zusammenfassung) | Konferenzzeitschriften zum Thema Transport (TRR, IEEE ITS) | Kontinuierliche Vorschulung möglich |

### 1.2 Datenbereinigung und Annotation

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

### 1.3 Befehls-Tuning-Datenformat

Es wird empfohlen, das „Alpaca“-Format oder das „Sharegpt“-Format zu verwenden:

```json
{
  "instruction": "北京晚高峰东三环严重拥堵，有什么疏导建议？",
  "input": "当前路况：东三环双向车速 < 15km/h，持续时间 > 40min",
  "output": "根据当前态势，建议以下疏导方案：1) 将国贸桥至长虹桥段信号配时调整为..." 
}
```

### 1.4 Datenabgleich

Ein effektiver Feinabstimmungsdatensatz für die Verkehrsdomäne enthält normalerweise:

- **Allgemeine Fähigkeitserhaltung (20 %)**: Behalten Sie einige allgemeine QA-, Zusammenfassungs- und Übersetzungsdaten bei, um katastrophales Vergessen zu verhindern
- **Injektion von Domänenwissen (50 %)**: Fragen und Antworten zum Fachwissen im Verkehrsbereich, Signalsteuerung, Pfadplanung
- **Aufgabenorientierte Daten (30 %)**: Erstellung von Simulationskonfigurationen, Berichtsanalyse, Begründung von Verkehrsstörungen

## 2. Auswahl der Feinabstimmungsmethoden

### 2.1 Vollständige Parameter-Feinabstimmung vs. Parameter-effiziente Feinabstimmung| Methode | Parametermenge | Anforderungen an den Videospeicher | Wirkung | Anwendbare Szenarien |
|------|--------|---------|------|---------|
| Vollständige Parameter-Feinabstimmung (Full FT) | 100 % | A100 80G × Multi-Karte | Am besten | Ausreichende Rechenleistung |
| LoRA | 0,1 % ~ 1 % | Eine 24G-Karte ist machbar | Nahezu vollständige Referenz | Mainstream-Wahl |
| QLoRA | 0,1 % ~ 1 % (4-Bit) | Einzelne Karte 16G machbar | Etwas niedriger als LoRA | Ressourcen begrenzt |
| Adapter-Tuning | 1 % ~ 5 % | Einzelkarte 24G | Mittel | Viele Aufgabenwechsel |

**Empfehlung**: Priorisieren Sie die Verwendung von **QLoRA**, das die Feinabstimmung des 7B-Modells mit 16G Videospeicher abschließen kann.

### 2.2 LoRA-Kernprinzipien

LoRA fügt neben der Q/K/V/O-Matrix der Aufmerksamkeitsschicht einen Low-Rank-Adapter hinzu:

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

### 2.3 Gestaltung des Verkehrsfeldadapters

Bei Verkehrsszenarien können Sie sich auf das Hinzufügen von LoRA zu den folgenden Ebenen konzentrieren:

```
decoder_block.attn.q_proj      ← 语义理解（query 理解交通场景描述）
decoder_block.attn.k_proj      ← 关键实体（key 关联交通要素）
decoder_block.attn.v_proj      ← 知识记忆（value 存储交通规则）
decoder_block.mlp.gate_proj    ← 领域知识凝练
```

## 3. Feinabstimmung des eigentlichen Codes

### 3.1 Umgebungsvorbereitung

```bash
pip install transformers peft accelerate bitsandbytes trl
# 或使用 AxT >= 0.1 的统一命令：
pip install autotorch
```

### 3.2 QLoRA-Feinabstimmungsskript

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

### 3.3 Spezielle Wortschatzerweiterung im Transportbereich

Wenn es im Transportbereich viele Fachbegriffe gibt (z. B. „Verflechtungsbereich“, „Signalphase“, „grüner Wellengürtel“), empfiehlt es sich, zunächst den Wortschatz zu erweitern und anschließend zu verfeinern:

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

## 4. Trainingsfähigkeiten und Fallstricke

### 4.1 Katastrophaler Vergessensschutz

Während der vollständigen Feinabstimmung der Parameter kann es zu katastrophalem Vergessen (allgemeine Fähigkeit des Modellvergessens) kommen. Lösung:

1. **Gemeinsame Daten einmischen (20 %)**: Behalten Sie immer einige gemeinsame QS-Daten bei
2. **Verwenden Sie die „Gamma“-Planung**: Beachten Sie das progressive Auftauen von SwissArmyKnife
3. **Stellen Sie eine kleinere Lernrate ein**: normalerweise 1/10 ~ 1/100 der Vorschulung

### 4.2 Illusionsproblem im Transportbereich

Im Transportbereich werden hohe Anforderungen an die Sachlichkeit gestellt, und LLM neigt dazu, ernsthaft Unsinn zu reden. Gegenmaßnahmen:

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

### 4.3 Multimodale ErweiterungWenn Sie Verkehrsbilder (Straßenfotos, Simulations-Screenshots) verarbeiten müssen, können Sie zur multimodalen Feinabstimmung auf die Serien **LLaVA** oder **Qwen-VL** zurückgreifen:

```python
# 以 LLaVA 为例：图像 + 交通文本联合微调
from llava.model.builder import load_pretrained_model
model, tokenizer, image_processor = load_pretrained_model("liuhaotian/llava-v1.6-7b")
```

## 5. Bewertung und Bereitstellung

### 5.1 Bewertungsindikatoren

| Aufgabe | Bewertungsmetriken | Beschreibung |
|------|---------|------|
| Fragen und Antworten zum Thema Transport | BLEU-4 / ROUGE-L | Mit Expertenantworten vergleichen |
| Signal-Timing-Empfehlungen | Benutzerdefinierte Bewertungsfunktion (Verzögerungsreduzierungsrate) | Erfordert eine Überprüfung der Simulationsumgebung |
| Zusammenfassung des Vorfallberichts | BERTScore / G-Eval | Semantische Ebenenbewertung |
| Domänenwissenstest | Transport-Benchmark (selbst gebaut) | Multiple-Choice/falsche/falsche Fragen |

### 5.2 Beispiele typischer Bewertungsfragen

```json
{
  "question": "在某交叉口，晚高峰期间左转流量很大（280 pcu/h），对向直行流量为 600 pcu/h，
              该交叉口采用两相位控制，配时为 G40-Y3-R37。建议如何优化？",
  "expected": "建议增加左转专用相位或将配时调整为 G55-Y3-R32，以提高左转通行能力..."
}
```

### 5.3 Bereitstellung

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

## 6. Erweiterte Richtung

- **Multi-Agenten-Zusammenarbeit**: Teilen Sie Signalsteuerung, Pfadplanung und Simulationsplanung in mehrere Agenten-Zusammenarbeit auf
- **Zeitreihendatenfusion**: Kombiniert mit GeoJSON/CSV-Verkehrsdateneingabe, um eine hybride Argumentation von Text + Daten zu erreichen
- **Kontinuierliche Feinabstimmung**: Verwenden Sie regelmäßig neue Daten für die SFT, um die Aktualität des Modells aufrechtzuerhalten

---

*Wenn Sie Fragen oder Anregungen haben, hinterlassen Sie bitte eine Nachricht im Blog! *