---
title: "LLM 微調整の実践: 地上輸送用の本格的な大規模モデルの作成"
description: "データ構築から展開の微調整まで、LoRA/QLoRA を使用してオープンソース LLM を微調整し、交通分野のエキスパート モデルを構築する方法を段階的に説明します。"
pubDate: 2026-04-25
tags: ['LLM', '微调', '地面交通', 'LoRA', 'QLoRA']
category: Tech
sourceHash: "e3866339bf46e05c8aa1b79ce5655fb7c4adcd60"
---

## はじめに

地上交通システムには、信号制御、経路計画、交通シミュレーション、事故予測などの多くの側面が含まれます。従来のルールベースまたは浅い学習方法には、一般化機能とコンテキストの理解において明らかなボトルネックがあります。輸送ドメインの専門家として大規模言語モデル (LLM) を微調整すると、次のことが可能になります。

- **自然言語による対話型交通 Q&A**: 自然言語を使用して交通状況と渋滞の原因をクエリします
- **交通シーンのシミュレーション記述の生成**: 交通状況に基づいてシミュレーション構成ファイルを生成します。
- **信号制御戦略の推論**: 時系列データに基づいて信号タイミングの提案を提供します。
- **事故報告書の自動分析と要約**: 多数の事故記録から重要な情報を抽出します。

この記事では、オープンソースの LLaMA/Qwen シリーズをベース モデルとして使用し、トラフィック データ セットのスクラッチ構築から LoRA の微調整展開までの完全なプロセスを紹介します。

## 1. データ構築：交通分野のデータセット

### 1.1 データソース

|データ型 |出典 |説明 |
|-------|------|------|
|交通ニュース/レポート | AutoNavi/Baidu交通API、各地の交通警察の公開データ |アノテーションの難易度は低く、ボリュームは大きい |
|交通事故の説明 |事故レポート、122 アラーム記録の減感作データ |専門的な注釈が必要 |
|信号タイミング計画 |交通警察署またはシミュレーション プラットフォームのデータ |構造化された抽出が必要 |
|交通シミュレーションシーンの説明 | SUMO/Paramics などのシミュレーション ソフトウェアからエクスポート |高価値、強力なプロ意識 |
|交通に関する学術論文（抄録） |交通分野の会議誌（TRR、IEEE ITS） |継続的な事前トレーニングが可能 |

### 1.2 データのクリーニングと注釈

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

### 1.3 命令チューニングのデータ形式

「alpaca」形式または「sharegpt」形式を使用することをお勧めします。

```json
{
  "instruction": "北京晚高峰东三环严重拥堵，有什么疏导建议？",
  "input": "当前路况：东三环双向车速 < 15km/h，持续时间 > 40min",
  "output": "根据当前态势，建议以下疏导方案：1) 将国贸桥至长虹桥段信号配时调整为..." 
}
```

### 1.4 データマッチング

トラフィック ドメインの効果的な微調整データセットには通常、次のものが含まれます。

- **一般的な機能保持 (20%)**: 致命的な忘れを防ぐために、一部の一般的な QA、要約、および翻訳データを保持します。
- **ドメイン知識の注入 (50%)**: 交通専門知識 Q&A、信号制御、経路計画
- **タスク指向データ (30%)**: シミュレーション構成の生成、レポート分析、交通事故の推論

## 2. 微調整方法の選択

### 2.1 完全なパラメータ微調整とパラメータ効率の良い微調整|方法 |パラメータ量 |ビデオメモリ要件 |効果 |該当するシナリオ |
|------|--------|-----------|------|---------|
|フルパラメータ微調整（フルFT） | 100% | A100 80G × マルチカード |ベスト |十分なコンピューティング能力 |
|ロラ | 0.1%~1% |シングルカード24Gも実現可能 |ほぼ完全なリファレンス |主流の選択 |
| QLoRA | 0.1%~1% (4ビット) |シングルカード16G実現可能 | LoRA よりわずかに低い |リソースが限られている |
|アダプターチューニング | 1%~5% |シングルカード 24G |中 |多くのタスク切り替え |

**推奨事項**: 16G のビデオ メモリを備えた 7B モデルの微調整を完了できる **QLoRA** の使用を優先します。

### 2.2 LoRA の基本原則

LoRA は、アテンション層の Q/K/V/O マトリックスの隣に低ランクのアダプターを追加します。

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

### 2.3 トラフィックフィールドアダプターの設計

トラフィック シナリオの場合は、次のレイヤーに LoRA を追加することに重点を置くことができます。

```
decoder_block.attn.q_proj      ← 语义理解（query 理解交通场景描述）
decoder_block.attn.k_proj      ← 关键实体（key 关联交通要素）
decoder_block.attn.v_proj      ← 知识记忆（value 存储交通规则）
decoder_block.mlp.gate_proj    ← 领域知识凝练
```

## 3. 実際のコードを微調整する

### 3.1 環境の準備

```bash
pip install transformers peft accelerate bitsandbytes trl
# 或使用 AxT >= 0.1 的统一命令：
pip install autotorch
```

### 3.2 QLoRA 微調整スクリプト

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

### 3.3 交通分野における特別な語彙の拡張

交通分野に多数の専門用語 (「織り交ぜエリア」、「信号位相」、「グリーン ウェーブ ベルト」など) がある場合は、まず語彙を増やしてから微調整することをお勧めします。

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

## 4. トレーニングスキルと落とし穴

### 4.1 壊滅的な忘れ防止

壊滅的な忘却 (モデルの忘却の一般的な能力) は、パラメーターの完全な微調整中に発生する傾向があります。解決策:

1. **共通データを混ぜる (20%)**: 常にいくつかの共通 QA データを保持します。
2. **「ガンマ」スケジューリングを使用**: SwissArmyKnife のプログレッシブ解凍を参照してください。
3. **学習率を低く設定**: 通常は事前トレーニングの 1/10 ～ 1/100

### 4.2 輸送分野における錯覚問題

輸送分野では事実の要求が高く、LLM はナンセンスな話を真剣に話す傾向があります。対策:

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

### 4.3 マルチモーダル拡張交通写真 (道路写真、シミュレーション スクリーンショット) を処理する必要がある場合は、**LLaVA** または **Qwen-VL** シリーズにアクセスしてマルチモーダル微調整を行うことができます。

```python
# 以 LLaVA 为例：图像 + 交通文本联合微调
from llava.model.builder import load_pretrained_model
model, tokenizer, image_processor = load_pretrained_model("liuhaotian/llava-v1.6-7b")
```

## 5. 評価と導入

### 5.1 評価指標

|タスク |評価指標 |説明 |
|------|------|------|
|交通に関するQ&A |ブルー-4 / ルージュ-L |専門家の回答と比較する |
|信号タイミングの推奨事項 |カスタムスコアリング機能（遅延短縮率）｜シミュレーション環境の検証が必要 |
|インシデントレポートの概要 | BERTScore / G 評価 |意味レベルの評価 |
|ドメイン知識テスト |輸送ベンチマーク (自作) |多肢選択/誤/誤問題 |

### 5.2 典型的な評価質問の例

```json
{
  "question": "在某交叉口，晚高峰期间左转流量很大（280 pcu/h），对向直行流量为 600 pcu/h，
              该交叉口采用两相位控制，配时为 G40-Y3-R37。建议如何优化？",
  "expected": "建议增加左转专用相位或将配时调整为 G55-Y3-R32，以提高左转通行能力..."
}
```

### 5.3 導入

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

## 6. 高度な方向性

- **マルチエージェント コラボレーション**: 信号制御、経路計画、シミュレーション スケジューリングを複数のエージェント コラボレーションに分割します。
- **時系列データの融合**: GeoJSON/CSV トラフィック データ入力と組み合わせて、テキスト + データのハイブリッド推論を実現します。
- **継続的な微調整**: モデルの適時性を維持するために、新しいデータを定期的に使用して SFT を実行します。

---

※ご質問やご提案がございましたら、ブログにメッセージを残してください！ *