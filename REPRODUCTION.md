# MedicalGPT 项目复现教程

本文档提供详细的 MedicalGPT 项目复现步骤，帮助您从零开始完成环境搭建、数据准备、模型训练和推理部署的全流程。

## 📋 目录

- [1. 环境准备](#1-环境准备)
- [2. 数据准备](#2-数据准备)
- [3. 模型准备](#3-模型准备)
- [4. 训练流程](#4-训练流程)
  - [4.1 阶段一：增量预训练 (PT)](#41-阶段一增量预训练-pt)
  - [4.2 阶段二：有监督微调 (SFT)](#42-阶段二有监督微调-sft)
  - [4.3 阶段三：奖励建模 (RM)](#43-阶段三奖励建模-rm)
  - [4.4 阶段四：强化学习训练](#44-阶段四强化学习训练)
  - [4.5 阶段三替代方案：DPO训练](#45-阶段三替代方案dpo训练)
  - [4.6 阶段三替代方案：ORPO训练](#46-阶段三替代方案orpo训练)
- [5. 模型推理](#5-模型推理)
- [6. 模型部署](#6-模型部署)
- [7. 常见问题](#7-常见问题)
- [8. 资源需求](#8-资源需求)

---

## 1. 环境准备

### 1.1 系统要求

- **操作系统**: Linux (推荐 Ubuntu 20.04+)
- **Python**: 3.8 或更高版本
- **CUDA**: 11.8 或更高版本（GPU训练必需）
- **显存**: 根据模型大小和训练方法，至少需要 6GB（QLoRA 4bit 训练 7B 模型）

### 1.2 克隆项目

```bash
# 克隆仓库
git clone https://github.com/shibing624/MedicalGPT.git
cd MedicalGPT
```

### 1.3 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
# venv\Scripts\activate  # Windows

# 安装 PyTorch（根据您的 CUDA 版本选择）
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装项目依赖
pip install -r requirements.txt --upgrade
```

### 1.4 验证安装

```bash
# 验证关键库是否正确安装
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import transformers; print(f'Transformers版本: {transformers.__version__}')"
python -c "import peft; print(f'PEFT版本: {peft.__version__}')"
python -c "import trl; print(f'TRL版本: {trl.__version__}')"
```

---

## 2. 数据准备

### 2.1 数据目录结构

创建数据目录并准备数据文件：

```bash
mkdir -p data/pretrain   # 预训练数据
mkdir -p data/finetune  # 微调数据
mkdir -p data/reward    # 奖励模型数据（偏好数据）
```

### 2.2 预训练数据格式

预训练数据应为纯文本文件，每行一个文档。示例：

```
这是第一个医疗文档的内容...
这是第二个医疗文档的内容...
```

### 2.3 微调数据格式

微调数据支持多种格式，推荐使用 JSONL 格式（ShareGPT 格式）：

**单轮对话格式** (`alpaca` 模板):
```json
{"instruction": "小孩发烧怎么办", "input": "", "output": "发烧是身体对感染的反应..."}
{"instruction": "如何预防感冒", "input": "", "output": "预防感冒的方法包括..."}
```

**多轮对话格式** (`sharegpt`/`vicuna` 模板):
```json
{"conversations": [{"from": "human", "value": "你好"}, {"from": "gpt", "value": "你好！有什么可以帮助你的吗？"}]}
{"conversations": [{"from": "human", "value": "感冒了怎么办"}, {"from": "gpt", "value": "感冒时应该..."}]}
```

### 2.4 奖励模型数据格式（偏好数据）

DPO/RM 训练需要偏好数据，格式如下：

```json
{"prompt": "小孩发烧怎么办", "chosen": "正确的回答...", "rejected": "不正确的回答..."}
{"prompt": "如何预防感冒", "chosen": "好的回答...", "rejected": "不好的回答..."}
```

### 2.5 数据集下载

#### 医疗数据集
- 240万条中文医疗数据集: [shibing624/medical](https://huggingface.co/datasets/shibing624/medical)
- 22万条中文医疗对话数据集: [shibing624/huatuo_medical_qa_sharegpt](https://huggingface.co/datasets/shibing624/huatuo_medical_qa_sharegpt)

#### 通用数据集
- 10万条多语言ShareGPT GPT4多轮对话: [shibing624/sharegpt_gpt4](https://huggingface.co/datasets/shibing624/sharegpt_gpt4)
- 2万条中英文偏好数据集: [shibing624/DPO-En-Zh-20k-Preference](https://huggingface.co/datasets/shibing624/DPO-En-Zh-20k-Preference)

使用 Hugging Face datasets 下载：

```python
from datasets import load_dataset

# 下载医疗数据集
dataset = load_dataset("shibing624/medical")
dataset.save_to_disk("./data/medical")

# 下载偏好数据集
pref_dataset = load_dataset("shibing624/DPO-En-Zh-20k-Preference")
pref_dataset.save_to_disk("./data/reward")
```

---

## 3. 模型准备

### 3.1 选择基础模型

根据硬件资源选择合适的模型：

| 模型系列 | 推荐模型 | 显存需求 (QLoRA 4bit) | 说明 |
|---------|---------|---------------------|------|
| Qwen2.5 | Qwen2.5-0.5B/1.5B/7B | 6GB/8GB/16GB | 推荐，中文支持好 |
| Qwen2 | Qwen2-7B | 16GB | 性能优秀 |
| LLaMA3 | Llama-3-8B | 20GB | 开源社区广泛使用 |
| LLaMA2 | Llama-2-7B-chat | 16GB | 经典选择 |

### 3.2 下载模型

使用 Hugging Face Hub 下载模型：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # 示例，根据需求选择
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 或使用命令行
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir ./models/Qwen2.5-0.5B-Instruct
```

### 3.3 模型路径配置

在训练脚本中，将模型路径设置为：

```bash
--model_name_or_path Qwen/Qwen2.5-0.5B-Instruct  # 使用 Hugging Face Hub 名称
# 或
--model_name_or_path ./models/Qwen2.5-0.5B-Instruct  # 使用本地路径
```

---

## 4. 训练流程

MedicalGPT 提供完整的训练流程，可以根据需求选择不同的阶段组合。

### 4.1 阶段一：增量预训练 (PT)

**目的**: 在领域文档上继续预训练，注入领域知识（可选，但推荐）

**数据**: `./data/pretrain` 目录下的纯文本文件

**训练命令**:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 pretraining.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --train_file_dir ./data/pretrain \
    --validation_file_dir ./data/pretrain \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --use_peft True \
    --seed 42 \
    --max_train_samples 10000 \
    --max_eval_samples 10 \
    --num_train_epochs 0.5 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 50 \
    --eval_strategy steps \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 13 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 10 \
    --block_size 512 \
    --group_by_length True \
    --output_dir outputs-pt-qwen-v1 \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --bf16 \
    --device_map auto \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing True \
    --cache_dir ./cache
```

**关键参数说明**:
- `--model_name_or_path`: 基础模型路径
- `--train_file_dir`: 训练数据目录
- `--use_peft True`: 使用 LoRA 微调（节省显存）
- `--lora_rank 8`: LoRA 秩大小
- `--output_dir`: 模型输出目录

**输出**: `outputs-pt-qwen-v1` 目录下的 LoRA 权重

**显存需求**: 
- QLoRA 4bit: 约 6-8GB (7B 模型)
- LoRA 16bit: 约 16GB (7B 模型)

---

### 4.2 阶段二：有监督微调 (SFT)

**目的**: 在指令数据上微调，对齐指令意图（必需）

**数据**: `./data/finetune` 目录下的 JSONL 格式文件

**训练命令**:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 supervised_finetuning.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --train_file_dir ./data/finetune \
    --validation_file_dir ./data/finetune \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --template_name qwen \
    --use_peft True \
    --max_train_samples 1000 \
    --max_eval_samples 10 \
    --model_max_length 4096 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --weight_decay 0.05 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 50 \
    --eval_strategy steps \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 13 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 4 \
    --output_dir outputs-sft-qwen-v1 \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --bf16 \
    --device_map auto \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing True \
    --cache_dir ./cache \
    --flash_attn True
```

**关键参数说明**:
- `--template_name`: 对话模板名称（如 `qwen`, `vicuna`, `alpaca`）
- `--model_max_length`: 最大序列长度
- `--flash_attn`: 是否使用 Flash Attention（加速训练）

**如果使用 PT 阶段的输出**:

```bash
# 需要先合并 PT 阶段的 LoRA 权重
python merge_peft_adapter.py \
    --base_model_name_or_path Qwen/Qwen2.5-0.5B \
    --peft_model_path outputs-pt-qwen-v1/checkpoint-500 \
    --output_dir merged-pt-qwen-v1

# 然后在 SFT 时使用合并后的模型
--model_name_or_path merged-pt-qwen-v1
```

**输出**: `outputs-sft-qwen-v1` 目录下的 LoRA 权重

---

### 4.3 阶段三：奖励建模 (RM)

**目的**: 训练奖励模型，建模人类偏好（RLHF 流程必需）

**数据**: `./data/reward` 目录下的偏好数据（chosen/rejected 格式）

**训练命令**:

```bash
# 注意：reward model 训练暂不支持 torchrun 多卡训练
CUDA_VISIBLE_DEVICES=0,1 python reward_modeling.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --train_file_dir ./data/reward \
    --validation_file_dir ./data/reward \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --use_peft True \
    --seed 42 \
    --max_train_samples 1000 \
    --max_eval_samples 10 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --weight_decay 0.001 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 50 \
    --eval_strategy steps \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 3 \
    --max_source_length 1024 \
    --max_target_length 256 \
    --output_dir outputs-rm-qwen-v1 \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --bf16 \
    --torch_dtype bfloat16 \
    --device_map auto \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --remove_unused_columns False \
    --gradient_checkpointing True
```

**输出**: `outputs-rm-qwen-v1` 目录下的奖励模型权重

---

### 4.4 阶段四：强化学习训练 (PPO)

**目的**: 使用奖励模型优化生成策略（RLHF 流程的最后一步）

**训练命令**:

```bash
CUDA_VISIBLE_DEVICES=0,1 python ppo_training.py \
    --sft_model_path outputs-sft-qwen-v1/checkpoint-500 \
    --reward_model_path outputs-rm-qwen-v1/checkpoint-500 \
    --template_name qwen \
    --torch_dtype bfloat16 \
    --train_file_dir ./data/finetune \
    --validation_file_dir ./data/finetune \
    --max_source_length 1024 \
    --response_length 1000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing True \
    --do_train \
    --total_episodes 30000 \
    --output_dir outputs-ppo-qwen-v1 \
    --missing_eos_penalty 1.0 \
    --eval_strategy steps \
    --eval_steps 100 \
    --num_train_epochs 3 \
    --report_to tensorboard
```

**关键参数说明**:
- `--sft_model_path`: SFT 阶段训练好的模型路径
- `--reward_model_path`: RM 阶段训练好的奖励模型路径
- `--total_episodes`: PPO 训练的总回合数

**输出**: `outputs-ppo-qwen-v1` 目录下的最终模型

---

### 4.5 阶段三替代方案：DPO训练

**目的**: 直接偏好优化，无需奖励模型（推荐，更简单高效）

**数据**: 同 RM 阶段，需要偏好数据

**训练命令**:

```bash
CUDA_VISIBLE_DEVICES=0,1 python dpo_training.py \
    --model_name_or_path outputs-sft-qwen-v1/checkpoint-500 \
    --template_name qwen \
    --train_file_dir ./data/reward \
    --validation_file_dir ./data/reward \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --use_peft True \
    --max_train_samples 1000 \
    --max_eval_samples 10 \
    --max_steps 100 \
    --eval_steps 20 \
    --save_steps 50 \
    --max_source_length 1024 \
    --max_target_length 512 \
    --output_dir outputs-dpo-qwen-v1 \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --bf16 True \
    --fp16 False \
    --device_map auto \
    --report_to tensorboard \
    --remove_unused_columns False \
    --gradient_checkpointing True \
    --cache_dir ./cache
```

**关键参数说明**:
- `--model_name_or_path`: 使用 SFT 阶段训练好的模型
- DPO 无需单独的奖励模型，直接优化偏好

**输出**: `outputs-dpo-qwen-v1` 目录下的 DPO 模型

**优势**: 
- 比 RLHF 流程更简单，无需训练奖励模型
- 训练更稳定，效果通常更好
- 计算资源需求更少

---

### 4.6 阶段三替代方案：ORPO训练

**目的**: 比值比偏好优化，不需要参考模型（最新方法）

**训练命令**:

```bash
# 参考 run_orpo.sh
CUDA_VISIBLE_DEVICES=0,1 python orpo_training.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --template_name qwen \
    --train_file_dir ./data/reward \
    --validation_file_dir ./data/reward \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --use_peft True \
    --max_train_samples 1000 \
    --max_eval_samples 10 \
    --max_steps 100 \
    --eval_steps 20 \
    --save_steps 50 \
    --max_source_length 1024 \
    --max_target_length 512 \
    --output_dir outputs-orpo-qwen-v1 \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --bf16 True \
    --device_map auto \
    --report_to tensorboard \
    --gradient_checkpointing True \
    --cache_dir ./cache
```

**优势**:
- 不需要参考模型（ref_model）
- 可以同时进行 SFT 和对齐训练
- 缓解灾难性遗忘问题

---

## 5. 模型推理

### 5.1 基本推理

使用训练好的模型进行推理：

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --base_model Qwen/Qwen2.5-0.5B-Instruct \
    --lora_model outputs-sft-qwen-v1/checkpoint-500 \
    --interactive
```

**参数说明**:
- `--base_model`: 基础模型路径
- `--lora_model`: LoRA 权重路径（如果已合并，可不指定）
- `--interactive`: 交互式对话模式
- `--template_name`: 对话模板名称

### 5.2 批量推理

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --base_model Qwen/Qwen2.5-0.5B-Instruct \
    --lora_model outputs-sft-qwen-v1/checkpoint-500 \
    --data_file input.jsonl \
    --output_file output.jsonl \
    --template_name qwen
```

### 5.3 合并 LoRA 权重（可选）

如果希望部署合并后的模型：

```bash
python merge_peft_adapter.py \
    --base_model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --peft_model_path outputs-sft-qwen-v1/checkpoint-500 \
    --output_dir merged-sft-qwen-v1
```

合并后可直接使用合并后的模型路径，无需指定 `--lora_model`。

---

## 6. 模型部署

### 6.1 Gradio Web 界面

启动 Web 界面进行交互式对话：

```bash
CUDA_VISIBLE_DEVICES=0 python gradio_demo.py \
    --base_model Qwen/Qwen2.5-0.5B-Instruct \
    --lora_model outputs-sft-qwen-v1/checkpoint-500 \
    --template_name qwen
```

访问 `http://localhost:7860` 即可使用 Web 界面。

### 6.2 FastAPI 服务

启动 API 服务：

```bash
CUDA_VISIBLE_DEVICES=0 python fastapi_server_demo.py \
    --base_model Qwen/Qwen2.5-0.5B-Instruct \
    --lora_model outputs-sft-qwen-v1/checkpoint-500 \
    --template_name qwen
```

### 6.3 vLLM 部署（生产环境推荐）

使用 vLLM 进行高性能部署：

```bash
sh vllm_deployment.sh
```

或手动配置：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model merged-sft-qwen-v1 \
    --tensor-parallel-size 1 \
    --port 8000
```

---

## 7. 常见问题

### 7.1 显存不足 (OOM)

**问题**: `CUDA out of memory`

**解决方案**:
1. 减小 `--per_device_train_batch_size`
2. 增大 `--gradient_accumulation_steps`（保持有效 batch size 不变）
3. 使用 QLoRA 4bit 量化：
   ```bash
   pip install bitsandbytes
   # 在脚本中添加量化配置
   ```
4. 启用 `--gradient_checkpointing`（已默认启用）
5. 减小 `--model_max_length` 或 `--block_size`

### 7.2 训练速度慢

**解决方案**:
1. 启用 Flash Attention: `--flash_attn True`
2. 使用 bfloat16: `--bf16 --torch_dtype bfloat16`
3. 增加 `--preprocessing_num_workers`
4. 使用多卡训练（torchrun）
5. 检查数据加载是否成为瓶颈

### 7.3 数据格式错误

**问题**: `KeyError` 或数据加载失败

**解决方案**:
1. 检查数据格式是否符合模板要求
2. 验证 JSONL 文件格式正确性：
   ```bash
   python validate_jsonl.py your_data.jsonl
   ```
3. 确认 `--template_name` 与数据格式匹配

### 7.4 模型生成质量差

**解决方案**:
1. 增加训练数据量和质量
2. 调整学习率（通常 SFT 使用 1e-5 到 5e-5）
3. 增加训练轮数（注意过拟合）
4. 使用 DPO/ORPO 进行偏好对齐
5. 检查数据预处理是否正确

### 7.5 多卡训练失败

**问题**: DDP 训练报错

**解决方案**:
1. 确保使用 `torchrun` 而不是 `python`
2. 检查 `CUDA_VISIBLE_DEVICES` 设置
3. 增加 `--ddp_timeout` 值
4. 某些脚本（如 reward_modeling.py）不支持多卡，使用单卡

---

## 8. 资源需求

### 8.1 显存需求参考表

| 训练方法 | 精度 | 7B模型 | 13B模型 | 70B模型 |
|---------|------|--------|---------|---------|
| 全参数 | AMP | 120GB | 240GB | 1200GB |
| 全参数 | 16bit | 60GB | 120GB | 600GB |
| LoRA | 16bit | 16GB | 32GB | 160GB |
| QLoRA | 8bit | 10GB | 20GB | 80GB |
| QLoRA | 4bit | 6GB | 12GB | 48GB |

### 8.2 训练时间估算

以 Qwen2.5-7B 模型，1000 条数据为例：

- **PT 阶段**: 约 1-2 小时（单卡 A100）
- **SFT 阶段**: 约 30 分钟 - 1 小时
- **DPO 阶段**: 约 20-30 分钟
- **PPO 阶段**: 约 1-2 小时

实际时间取决于数据量、模型大小和硬件配置。

---

## 9. 训练流程示例

### 完整流程：PT → SFT → DPO

```bash
# 1. 增量预训练
sh run_pt.sh

# 2. 合并 PT 权重（可选）
python merge_peft_adapter.py \
    --base_model_name_or_path Qwen/Qwen2.5-0.5B \
    --peft_model_path outputs-pt-qwen-v1/checkpoint-500 \
    --output_dir merged-pt-qwen-v1

# 3. 有监督微调
# 修改 run_sft.sh 中的 model_name_or_path 为 merged-pt-qwen-v1
sh run_sft.sh

# 4. DPO 训练
sh run_dpo.sh

# 5. 合并最终模型
python merge_peft_adapter.py \
    --base_model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --peft_model_path outputs-dpo-qwen-v1/checkpoint-100 \
    --output_dir merged-final-qwen-v1

# 6. 推理测试
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --base_model merged-final-qwen-v1 \
    --interactive
```

### 简化流程：SFT → DPO（推荐快速上手）

```bash
# 1. 有监督微调
sh run_sft.sh

# 2. DPO 训练
sh run_dpo.sh

# 3. 推理测试
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --base_model Qwen/Qwen2.5-0.5B-Instruct \
    --lora_model outputs-dpo-qwen-v1/checkpoint-100 \
    --interactive
```

---

## 10. 进阶技巧

### 10.1 使用 TensorBoard 监控训练

```bash
# 启动 TensorBoard
tensorboard --logdir outputs-sft-qwen-v1/runs

# 访问 http://localhost:6006 查看训练曲线
```

### 10.2 扩充领域词表

如果有特殊领域的词汇，可以扩充词表：

```bash
python build_domain_tokenizer.py \
    --base_tokenizer_path Qwen/Qwen2.5-0.5B-Instruct \
    --domain_file_path ./data/vocab/medical_vocab.txt \
    --output_dir ./tokenizer-extended
```

### 10.3 数据转换

如果需要转换数据格式：

```bash
python convert_dataset.py \
    --input_file your_data.json \
    --output_file converted_data.jsonl \
    --template_name qwen
```

---

## 11. 参考文献

- [Direct Preference Optimization 论文](https://arxiv.org/pdf/2305.18290.pdf)
- [ORPO 论文](https://arxiv.org/abs/2403.07691)
- [RLHF 训练流程](https://karpathy.ai/stateofgpt.pdf)
- [项目 Wiki](https://github.com/shibing624/MedicalGPT/wiki)

---

## 12. 获取帮助

- **Issues**: [GitHub Issues](https://github.com/shibing624/MedicalGPT/issues)
- **Wiki**: [项目 Wiki](https://github.com/shibing624/MedicalGPT/wiki)
- **邮件**: xuming624@qq.com

---

## 附录：快速命令参考

```bash
# 环境安装
pip install -r requirements.txt --upgrade

# 训练
sh run_sft.sh      # 有监督微调
sh run_dpo.sh      # DPO 训练
sh run_orpo.sh     # ORPO 训练
sh run_pt.sh       # 增量预训练
sh run_rm.sh       # 奖励建模
sh run_ppo.sh      # PPO 强化学习

# 推理
python inference.py --base_model MODEL --lora_model LORA --interactive
python gradio_demo.py --base_model MODEL --lora_model LORA

# 工具
python merge_peft_adapter.py  # 合并 LoRA 权重
python validate_jsonl.py      # 验证数据格式
python convert_dataset.py     # 转换数据格式
```

---

**祝您训练顺利！如有问题，欢迎提交 Issue 或查阅项目 Wiki。**

