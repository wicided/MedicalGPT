# MedicalGPT 学习指南

> 本文档提供从入门到精通的完整学习路径，帮助你系统掌握 MedicalGPT 项目

## 📋 目录

- [一、项目概述](#一项目概述)
- [二、学习路径](#二学习路径)
- [三、环境准备](#三环境准备)
- [四、理论学习](#四理论学习)
- [五、实践入门](#五实践入门)
- [六、核心功能](#六核心功能)
- [七、代码阅读](#七代码阅读)
- [八、常见问题](#八常见问题)
- [九、进阶学习](#九进阶学习)

---

## 一、项目概述

### 1.1 项目定位

MedicalGPT 是一个医疗大语言模型训练框架，实现了完整的 ChatGPT 训练流程。项目以医疗领域为例，展示了如何训练领域专用的大语言模型。

### 1.2 核心功能
项目实现了三个主要训练阶段：
- **PT（增量预训练）**：在海量领域文档数据上二次预训练GPT模型
- **SFT（监督微调）**：构造指令微调数据集，对齐指令意图并注入领域知识
- **RLHF/DPO（对齐训练）**：
  - RLHF：奖励模型建模 + 强化学习训练
  - DPO：直接偏好优化，无需奖励模型
  - ORPO：单步优化方法
  - GRPO：纯强化学习方法

### 1.3 技术栈

- **深度学习框架**：PyTorch, Transformers
- **训练优化**：LoRA, QLoRA, DeepSpeed
- **注意力优化**：FlashAttention-2, LongLoRA
- **量化技术**：4-bit/8-bit 量化训练

---

## 二、学习路径

### 2.1 学习阶段

**阶段一：基础了解（1-3天）** - 阅读 README.md，了解项目结构，理解训练流程

**阶段二：环境搭建（1天）** - 安装依赖，配置 GPU 环境，运行第一个示例

**阶段三：理论准备（3-5天）** - 理解 GPT 训练原理，学习 RLHF/DPO 方法，了解 LoRA 微调技术

**阶段四：实践入门（5-7天）** - 运行 Colab Notebook，本地 SFT 微调实验，模型推理测试

**阶段五：深入理解（1-2周）** - 阅读核心代码，理解训练参数，自定义数据集训练

**阶段六：进阶应用（持续）** - 实现 DPO/ORPO 训练，部署 API 服务，优化训练效率

### 2.2 时间估算

- **快速上手**：1-2天（能运行基础示例）
- **掌握基本使用**：1-2周（能完成 SFT 训练）
- **深入理解**：1-2个月（能自定义训练流程）

---

## 三、环境准备

### 3.1 硬件要求

| 训练方法 | 精度 | 7B模型 | 13B模型 |
|---------|------|--------|---------|
| QLoRA | 4-bit | 6GB | 12GB |
| QLoRA | 8-bit | 10GB | 20GB |
| LoRA | 16-bit | 16GB | 32GB |
| 全参数 | 16-bit | 60GB | 120GB |

**建议**：入门使用 RTX 3090/4090（24GB）+ QLoRA；进阶使用 A100（40GB/80GB）+ LoRA

### 3.2 软件环境

```bash
# 1. 克隆项目
git clone https://github.com/shibing624/MedicalGPT
cd MedicalGPT

# 2. 创建虚拟环境
conda create -n medicalgpt python=3.10
conda activate medicalgpt

# 3. 安装依赖
pip install -r requirements.txt --upgrade

# 4. 可选：安装 FlashAttention-2（RTX 3090/4090/A100）
pip install flash-attn --no-build-isolation
```

## 四、理论学习

### 4.1 必读资源

1. **GPT 训练流程**
   - 论文：[State of GPT (Andrej Karpathy)](https://karpathy.ai/stateofgpt.pdf)
   - 视频：[State of GPT 讲解](https://build.microsoft.com/en-US/sessions/db3f4859-cd30-4445-a0cd-553c3304f8e2)
   - 核心：理解预训练 → 监督微调 → 对齐训练的三个阶段

2. **DPO 方法**
   - 论文：[Direct Preference Optimization](https://arxiv.org/pdf/2305.18290.pdf)
   - 核心：直接优化语言模型实现偏好对齐，无需奖励模型

3. **LoRA 微调**
   - 论文：[LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
   - 核心：低秩适应，用少量参数实现高效微调

4. **QLoRA**
   - 论文：[QLoRA: Efficient Finetuning](https://arxiv.org/abs/2305.14314)
   - 核心：量化 + LoRA，进一步降低显存需求

### 4.2 核心概念

**PT（增量预训练）**：让模型适应领域数据分布，数据为原始文档，训练方式为自回归语言建模。

**SFT（监督微调）**：让模型学会遵循指令，数据为指令-回答对，训练方式为监督学习。

**RLHF（人类反馈强化学习）**：让模型生成更符合人类偏好的回答。步骤：①训练奖励模型（RM）；②强化学习（RL）用奖励模型指导优化。

**DPO（直接偏好优化）**：与 RLHF 目标相同，但不需要训练奖励模型，训练更稳定。数据为偏好对（chosen vs rejected）。

---

## 五、实践入门

### 5.1 第一步：体验推理

在训练前，先体验模型推理效果：

```bash
# 使用预训练模型进行推理
python inference.py \
    --base_model shibing624/vicuna-baichuan-13b-chat \
    --interactive

# 或使用 Gradio 界面
python gradio_demo.py \
    --base_model shibing624/vicuna-baichuan-13b-chat
```

**学习目标**：理解模型输入输出格式，体验不同模型效果，了解推理参数作用。

### 5.2 第二步：运行 Colab Notebook

最快上手的实践方式：

1. **DPO Pipeline**：`run_training_dpo_pipeline.ipynb`（约15分钟，完整 PT→SFT→DPO 流程）
2. **RLHF Pipeline**：`run_training_ppo_pipeline.ipynb`（约20分钟，完整 PT→SFT→RLHF 流程）

**学习要点**：观察每个阶段的训练过程，理解数据格式变化，对比不同方法的训练效果。

### 5.3 第三步：本地 SFT 训练

准备一个小规模数据集进行实验：

```bash
# 1. 准备数据集（JSON/JSONL 格式）
# data/finetune/train.jsonl
{"instruction": "问题", "input": "", "output": "回答"}

# 2. 运行 SFT 训练
CUDA_VISIBLE_DEVICES=0 python supervised_finetuning.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --train_file_dir ./data/finetune \
    --output_dir outputs-sft \
    --use_peft True \
    --lora_rank 8 \
    --lora_alpha 16 \
    --max_train_samples 1000 \
    --num_train_epochs 1
```

**学习目标**：理解训练参数含义，观察训练过程日志，学会调试训练问题。

---

## 六、核心功能

### 6.1 SFT（监督微调）

**数据格式**：支持 ShareGPT 格式（推荐）和 Alpaca 格式

```json
// ShareGPT 格式
{"conversations": [{"from": "human", "value": "你好"}, {"from": "gpt", "value": "你好！"}]}

// Alpaca 格式
{"instruction": "写一首诗", "input": "", "output": "诗的内容..."}
```

**关键参数**：
- `--model_name_or_path`: 基础模型路径
- `--use_peft`: 是否使用 LoRA（True/False）
- `--lora_rank`: LoRA 秩（默认8）
- `--lora_alpha`: LoRA alpha（默认16，通常为 rank 的2倍）
- `--learning_rate`: 学习率（LoRA: 1e-4~5e-4，全参: 1e-5~5e-5）

**训练技巧**：
- 显存优化：使用 `--load_in_4bit True`、`--gradient_checkpointing True`
- 训练稳定性：使用 `--bf16`/`--fp16`、设置合适的 `--warmup_ratio`（0.05-0.1）

### 6.2 DPO（直接偏好优化）

**数据格式**：
```json
{"prompt": "问题", "chosen": "好的回答", "rejected": "差的回答"}
```

**训练流程**：先进行 SFT 训练，再使用 SFT 模型进行 DPO 训练。

**关键参数**：`--beta`（DPO 温度参数，默认0.1）、`--reference_model`（参考模型路径）

### 6.3 PT（增量预训练）

**适用场景**：模型词汇表需要扩展、需要更好理解领域术语、有大量领域相关原始文本。

**数据格式**：纯文本格式，每行一个文档或段落。

**训练要点**：需要较长训练时间，建议在百万级数据上进行，可结合词表扩展（`--modules_to_save embed_tokens,lm_head`）。

---

## 七、代码阅读

### 7.1 核心文件

```
MedicalGPT/
├── supervised_finetuning.py    # SFT 训练核心代码（重点）
├── dpo_training.py             # DPO 训练代码
├── inference.py                # 推理代码（重点）
├── template.py                 # 对话模板定义（重要）
└── run_*.sh                    # 训练脚本示例
```

### 7.2 阅读顺序

**第一优先级**：`inference.py`（模型加载、对话模板、流式生成）→ `template.py`（对话格式、prompt 构造）→ `supervised_finetuning.py`（参数定义、数据处理、训练循环）

**第二优先级**：`dpo_training.py`（DPO 损失函数）→ `pretraining.py`（预训练数据处理）

### 7.3 阅读技巧

从入口开始（`run_sft.sh`）→ 关注参数定义（`@dataclass`）→ 跟踪数据流（加载→预处理→模型输入→损失计算）→ 理解模板系统（`template.py`）

---

## 八、常见问题

**显存不足（OOM）**：使用 QLoRA（`--load_in_4bit True --qlora True`）、减小批次大小增大梯度累积、使用梯度检查点（`--gradient_checkpointing True`）、使用 DeepSpeed ZeRO（`--deepspeed zero2.json`）

**训练不收敛**：检查学习率（建议从 2e-5 开始）、确保数据质量、调整 warmup_ratio（0.05-0.1）、使用较小的模型先验证流程

**数据格式错误**：使用 `validate_jsonl.py` 验证、检查 JSON/JSONL 格式、确保字段名称一致（conversations/instruction/output等）、参考 `docs/datasets.md`

**模型生成效果差**：可能原因包括数据质量或数量不足、训练轮数不够、学习率设置不当、模型本身能力限制

**多卡训练问题**：使用 `torchrun` 而非 `python`、确保 `nproc_per_node` 与 GPU 数量一致、使用 DeepSpeed 进行多卡训练、检查 CUDA_VISIBLE_DEVICES 设置

---

## 九、进阶学习

### 9.1 优化训练效率

**加速训练**：使用 FlashAttention-2（`--flash_attn True`）、混合精度训练（`--bf16`/`--fp16`）、梯度累积优化。

**显存优化**：QLoRA 量化（`--load_in_4bit True --qlora True`）、DeepSpeed ZeRO（`--deepspeed zero2.json`）。

### 9.2 模型部署与进阶应用

**API 服务**：`python fastapi_server_demo.py`（FastAPI）或 `python openai_api.py`（OpenAI 兼容接口）

**高性能推理**：`bash vllm_deployment.sh`（vLLM 部署）

**模型量化**：使用 `eval_quantize.py` 和 `model_quant.py`

**RAG（检索增强生成）**：使用 `chatpdf.py` 实现基于知识库的问答

### 9.3 学习资源

**官方文档**：README.md、docs/training_params.md、docs/datasets.md、docs/FAQ.md

**外部资源**：Hugging Face 文档、相关论文（State of GPT, DPO, LoRA, QLoRA）、GitHub Issues、微信群

---

## 十、学习检查清单

**基础阶段**：理解 GPT 训练流程、运行推理代码、完成 Colab Notebook、本地完成 SFT 训练

**进阶阶段**：理解 LoRA/QLoRA 原理、能自定义数据集训练、理解 DPO 训练流程、能调试训练问题

**高级阶段**：能优化训练效率、能部署模型服务、能实现 RAG 应用、能解决复杂训练问题

---

## 结语

学习 MedicalGPT 是一个循序渐进的过程。建议：**先动手再深入**（先跑通流程，再理解原理）、**从小规模开始**（用小数据集和小模型验证）、**多实践多思考**（遇到问题多尝试，多查阅资料）、**参与社区交流**（在 GitHub Issues 或微信群中提问和分享）。

**记住**：大模型训练需要时间和耐心，不要急于求成。每个阶段都扎实掌握，最终你会成为大模型训练的专家！

---

**祝你学习愉快！如有问题，欢迎在 GitHub Issues 中提问。** 🚀
