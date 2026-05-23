---
title: "LLM fine-tuning practice: creating a professional large-scale model for ground transportation"
description: "From data construction to fine-tuning deployment, we will teach you step by step how to use LoRA/QLoRA to fine-tune open source LLM to build an expert model in the transportation field."
pubDate: 2026-04-25
tags: ['LLM', '微调', '地面交通', 'LoRA', 'QLoRA']
category: Tech
---

## Introduction

The ground transportation system involves many aspects such as signal control, path planning, traffic simulation, accident prediction, etc. Traditional rule-based or shallow learning methods have obvious bottlenecks in generalization capabilities and context understanding. Fine-tuning a large language model (LLM) as a transportation domain expert enables:

- **Natural language interactive traffic Q&A**: Use natural language to query traffic conditions and congestion causes
- **Traffic scene simulation description generation**: Generate simulation configuration files based on traffic situation
- **Signal control strategy reasoning**: Give signal timing suggestions based on time series data
- **Automatic analysis and summary of accident reports**: Extract key information from a large number of accident records

This article uses the open source LLaMA/Qwen series as the base model to introduce the complete process from scratch construction of traffic data sets to LoRA fine-tuning deployment.

## 1. Data construction: transportation field data set

### 1.1 Data source

| Data type | Source | Description |
|---------|------|------|
| Traffic news/reports | AutoNavi/Baidu traffic API, public data of traffic police in various places | Low annotation difficulty, large volume |
| Traffic accident description | Accident report, 122 alarm record desensitization data | Professional annotation required |
| Signal timing plan | Traffic police department or simulation platform data | Structured extraction required |
| Traffic simulation scene description | Export from simulation software such as SUMO/Paramics | High value, strong professionalism |
| Transportation academic papers (abstract) | Transportation field conference journals (TRR, IEEE ITS) | Continuous pre-training possible |

### 1.2 Data cleaning and annotation

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

### 1.3 Instruction-Tuning data format

It is recommended to use `alpaca` format or `sharegpt` format:

```json
{
  "instruction": "北京晚高峰东三环严重拥堵，有什么疏导建议？",
  "input": "当前路况：东三环双向车速 < 15km/h，持续时间 > 40min",
  "output": "根据当前态势，建议以下疏导方案：1) 将国贸桥至长虹桥段信号配时调整为..." 
}
```

### 1.4 Data matching

An effective fine-tuning dataset for traffic domain usually contains:

- **General capability retention (20%)**: retain some general QA, summary, and translation data to prevent catastrophic forgetting
- **Domain knowledge injection (50%)**: Traffic professional knowledge Q&A, signal control, path planning
- **Task-oriented data (30%)**: simulation configuration generation, report analysis, traffic incident reasoning

## 2. Selection of fine-tuning methods

### 2.1 Full parameter fine-tuning vs parameter-efficient fine-tuning| Method | Parameter amount | Video memory requirements | Effect | Applicable scenarios |
|------|--------|---------|------|---------|
| Full parameter fine-tuning (Full FT) | 100% | A100 80G × Multi-card | Best | Sufficient computing power |
| LoRA | 0.1%~1% | Single card 24G is feasible | Close to full reference | Mainstream choice |
| QLoRA | 0.1%~1% (4-bit) | Single card 16G feasible | Slightly lower than LoRA | Resource limited |
| Adapter-tuning | 1%~5% | Single card 24G | Medium | Many task switching |

**Recommendation**: Prioritize the use of **QLoRA**, which can complete the fine-tuning of the 7B model with 16G of video memory.

### 2.2 LoRA core principles

LoRA adds a low-rank adapter next to the Q/K/V/O matrix of the attention layer:

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

### 2.3 Traffic field adapter design

For traffic scenarios, you can focus on adding LoRA to the following layers:

```
decoder_block.attn.q_proj      ← 语义理解（query 理解交通场景描述）
decoder_block.attn.k_proj      ← 关键实体（key 关联交通要素）
decoder_block.attn.v_proj      ← 知识记忆（value 存储交通规则）
decoder_block.mlp.gate_proj    ← 领域知识凝练
```

## 3. Fine-tuning the actual code

### 3.1 Environment preparation

```bash
pip install transformers peft accelerate bitsandbytes trl
# 或使用 AxT >= 0.1 的统一命令：
pip install autotorch
```

### 3.2 QLoRA fine-tuning script

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

### 3.3 Special vocabulary expansion in the transportation field

If there are a large number of professional terms in the transportation field (such as "interweaving area", "signal phase", "green wave belt"), it is recommended to expand the vocabulary first and then fine-tune:

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

## 4. Training skills and pitfalls

### 4.1 Catastrophic forgetting protection

Catastrophic forgetting (general ability of model forgetting) is prone to occur during full parameter fine-tuning. Solution:

1. **Mix in common data (20%)**: Always keep some common QA data
2. **Use `gamma` scheduling**: Refer to SwissArmyKnife’s progressive thawing
3. **Set a smaller learning rate**: usually 1/10 ~ 1/100 of pre-training

### 4.2 Illusion problem in transportation field

The transportation field has high requirements for factuality, and LLM is prone to talking nonsense seriously. Countermeasures:

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

### 4.3 Multimodal extensionIf you need to process traffic pictures (road photos, simulation screenshots), you can access the **LLaVA** or **Qwen-VL** series for multi-modal fine-tuning:

```python
# 以 LLaVA 为例：图像 + 交通文本联合微调
from llava.model.builder import load_pretrained_model
model, tokenizer, image_processor = load_pretrained_model("liuhaotian/llava-v1.6-7b")
```

## 5. Assessment and Deployment

### 5.1 Evaluation indicators

| Task | Evaluation Metrics | Description |
|------|---------|------|
| Transportation Q&A | BLEU-4 / ROUGE-L | Compare with expert answers |
| Signal timing recommendations | Custom scoring function (delay reduction rate) | Requires simulation environment verification |
| Incident report summary | BERTScore / G-Eval | Semantic level evaluation |
| Domain knowledge test | Transportation benchmark (self-built) | Multiple choice/false/false questions |

### 5.2 Examples of typical evaluation questions

```json
{
  "question": "在某交叉口，晚高峰期间左转流量很大（280 pcu/h），对向直行流量为 600 pcu/h，
              该交叉口采用两相位控制，配时为 G40-Y3-R37。建议如何优化？",
  "expected": "建议增加左转专用相位或将配时调整为 G55-Y3-R32，以提高左转通行能力..."
}
```

### 5.3 Deployment

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

## 6. Advanced direction

- **Multi-agent collaboration**: Divide signal control, path planning, and simulation scheduling into multiple Agent collaborations
- **Time series data fusion**: Combined with GeoJSON/CSV traffic data input to achieve hybrid reasoning of text + data
- **Continuous fine-tuning**: Regularly use new data to do SFT to maintain the timeliness of the model

---

*If you have any questions or suggestions, please leave a message on the blog! *