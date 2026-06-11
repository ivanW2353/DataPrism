# DataPrism：面向 LLM 微调的梯度驱动数据选择框架

DataPrism（数据棱镜）将海量微调数据分解为不同维度的影响力信号——识别对模型有帮助的数据（proponent）、有害的数据（opponent）和冗余数据（neutral）——从而精准筛选出真正有价值的训练子集，在保持模型性能的同时将数据量压缩至 20% 以下。

名称隐喻：棱镜将一束白光分解为不同波长的光谱。类似地，DataPrism 将海量微调数据分解为多维度的影响力信号，最终汇聚为高质量训练集。

---

## 架构

```
原始数据池 (10万+)
       │
       ▼
┌─────────────────────────────────┐
│         DataPrism 棱镜框架        │
│                                  │
│  ① 离线质控（TracIn 自影响）       │  → 去噪、去重、去冗余
│  ② 在线采样（梯度上界选样）        │  → 聚焦难例、加速收敛
│  ③ 迭代对齐（验证集驱动重选）      │  → 持续匹配下游目标
│                                  │
└─────────────────────────────────┘
       │
       ▼
高质量子集 (<20% 原始数据)
```

---

## 三个阶段

### Phase 1：LoRA 空间 TracInCP 离线数据质控

**目标**：训练前剔除噪声、错误标注、冗余样本。

将 TracInCP（Pruthi et al., NeurIPS 2020）迁移到 LoRA 参数空间：

1. 对预训练 LLM 加 LoRA 适配器，用全量数据跑 1-3 epoch 初步 SFT
2. 每 N 步保存 checkpoint（~10-20 个，仅保存 adapter 权重）
3. 对每个训练样本计算 **self-influence**：

```
SelfInfluence(z) = Σᵢ ηᵢ · ∇ℓ(z, θᵢ) · ∇ℓ(z, θᵢ)
```

- **自影响异常高** → 可能是错误标注（模型被迫"硬记"）
- **影响向量高度相似** → 冗余样本，聚类后只保留代表

**关键优化**：所有梯度操作限制在 LoRA 参数空间（~168M 维），而非全参数（~8B 维），4,000× 计算量缩减。仅存储标量影响分数，不存完整梯度向量。

### Phase 2：在线重要性采样

**目标**：训练过程中动态聚焦难例，加速收敛。

将重要性采样方法（Katharopoulos & Fleuret, ICML 2018）适配到 LLM：

1. 每步从数据池预采样 4× batch_size 候选集
2. 仅做前向传播，计算各候选的 token 级交叉熵损失
3. 以 softmax(loss / τ) 为概率采样目标 batch
4. 对选中样本按 1/prob 降权，保证梯度无偏估计
5. EMA 跟踪方差缩减率 η；当 η < 阈值时自动切回均匀采样

**关键优势**：loss 是前向传播的副产品，零额外架构开销。温度 τ 从高到低退火，从探索逐步转向利用。

### Phase 3：验证集驱动的迭代数据重选

**目标**：在多轮训练中持续对齐特定下游能力。

将 FLDebugger 分层影响力思想（Li et al., ICDE 2021）适配到 LLM 微调：

1. 构建多维验证集：V = {V_reasoning, V_safety, V_chat, V_factual}
2. 每个 epoch 结束后计算 TracInVS 评分：

```
Score(z, Vⱼ) = Σᵢ ∇ℓ(z, θᵢ) · ∇ℓ(Vⱼ, θᵢ)
```

3. 按目标权重聚合：`TotalScore(z) = Σ λⱼ · Score(z, Vⱼ)`
4. 下一轮调高 proponents 采样权重，压低 opponents

**关键创新**：可以为同一批数据生成 **影响力光谱**——对推理有帮助但可能损害安全性的样本将被显式标记，给使用者透明的取舍权。

---

## 安装

```bash
# 安装依赖
pip install -e ".[dev]"

# 需要的主要依赖
# torch>=2.1, transformers>=4.45, peft>=0.13, datasets>=3.0,
# accelerate>=1.0, numpy, scipy, scikit-learn, pyyaml, wandb, h5py
```

---

## 快速开始

### 仅运行 Phase 1（TracInCP 离线筛选）

```bash
python scripts/run_phase1_tracin.py \
  --config default phase1_tracin data/alpaca \
  --device cuda
```

### 完整管线（Phase 1 → Phase 2 → Phase 3）

```bash
python scripts/run_pipeline.py \
  --config default phase1_tracin phase2_importance phase3_multieval \
  --device cuda
```

### 运行所有基线

```bash
python scripts/run_baselines.py \
  --config default data/alpaca \
  --device cuda
```

### 评估已训练模型

```bash
python scripts/run_evaluation.py \
  --model_path outputs/phase2_training/final_model \
  --benchmarks mmlu gsm8k --device cuda
```

---

## 配置系统

所有超参数通过 YAML 配置文件管理，支持多层合并和 CLI 覆盖。

### 配置优先级（后覆盖前）

```
configs/default.yaml → configs/model/xxx.yaml → configs/data/xxx.yaml → CLI --override
```

### 切换模型和数据集

```bash
# Llama-3.1-8B + Alpaca（默认）
python scripts/run_pipeline.py --config default model/llama3_8b data/alpaca --device cuda

# Qwen2.5-7B + WizardLM
python scripts/run_pipeline.py --config default model/qwen2.5_7b data/wizardlm --device cuda

# Llama + OpenOrca
python scripts/run_pipeline.py --config default model/llama3_8b data/openorca --device cuda
```

### CLI 参数覆盖

```bash
python scripts/run_pipeline.py \
  --config default phase1_tracin \
  --override phase1.outlier_percentile=99.0 \
            phase2.initial_tau=0.5 \
            training.learning_rate=1e-4 \
            lora.r=32 \
            data.num_samples=null \
  --device cuda
```

### 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lora.r` | 64 | LoRA rank |
| `lora.alpha` | 128 | LoRA alpha |
| `training.learning_rate` | 2e-4 | 学习率 |
| `training.num_epochs` | 3 | 训练轮数 |
| `training.per_device_train_batch_size` | 4 | 每卡 batch size |
| `training.max_seq_length` | 2048 | 最大序列长度 |
| `phase1.outlier_percentile` | 95.0 | 自影响异常阈值（百分位） |
| `phase1.redundancy_similarity_threshold` | 0.85 | 冗余去重余弦相似度阈值 |
| `phase2.initial_tau` | 1.0 | 重要性采样温度初值 |
| `phase2.candidate_multiplier` | 4 | 候选集倍数（×batch_size） |
| `phase2.variance_reduction_threshold` | 0.05 | 方差缩减下限（低于此值切回均匀） |
| `phase3.lambda_weights` | 0.30, 0.25, 0.25, 0.20 | 验证维度权重（推理/安全/对话/事实） |
| `phase3.max_rounds` | 5 | 最大迭代重选轮数 |

### 配置文件结构

```
configs/
├── default.yaml                  # 全局默认配置
├── model/
│   ├── llama3_8b.yaml           # Llama-3.1-8B-Instruct
│   └── qwen2.5_7b.yaml          # Qwen2.5-7B-Instruct
├── lora/
│   └── default.yaml             # LoRA (r=64, α=128)
├── data/
│   ├── alpaca.yaml              # Alpaca-52K
│   ├── wizardlm.yaml            # WizardLM-70K
│   └── openorca.yaml            # OpenOrca
├── phase1_tracin.yaml           # Phase 1 配置
├── phase2_importance.yaml       # Phase 2 配置
├── phase3_multieval.yaml        # Phase 3 配置
└── baseline/
    ├── uniform.yaml             # 随机采样
    ├── full_data.yaml           # 全量数据
    ├── less.yaml                # LESS (ICML 2024)
    ├── rho_loss.yaml            # RHO-LOSS (NeurIPS 2022)
    └── dsir.yaml                # DSIR (NeurIPS 2023)
```

---

## 模型与数据

### 基座模型

| 模型 | 参数量 | HuggingFace 路径 |
|------|--------|-----------------|
| Llama-3.1-8B-Instruct | 8B | `meta-llama/Llama-3.1-8B-Instruct` |
| Qwen2.5-7B-Instruct | 7B | `Qwen/Qwen2.5-7B-Instruct` |

微调方式：LoRA (r=64, α=128)，目标模块为 q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj。可训练参数 ~168M，占总量 2.05%。

### 训练数据

| 数据集 | 规模 | 格式 | 用途 |
|--------|------|------|------|
| Alpaca-52K | 52K | instruction-output | 默认训练集 |
| WizardLM-70K | 143K | 多轮对话 | 复杂指令 |
| OpenOrca | 980K | 多种格式 | 多样化问答 |

### 评估基准

| 基准 | 类型 | 指标 |
|------|------|------|
| MMLU | 57 学科知识 | 5-shot 准确率 |
| GSM8K | 数学推理 | 8-shot CoT |
| HumanEval | 代码生成 | pass@1 |
| MT-Bench | 多轮对话 | GPT-4 judge |
| AlpacaEval | 指令遵循 | LLM-judge 胜率 |

---

## 基线方法

| 方法 | 类型 | 说明 | 论文 |
|------|------|------|------|
| Uniform Random | 下界 | 随机均匀采样 | — |
| Full Data | 上界 | 全量数据训练 | — |
| LESS | 表征相似度 | LoRA checkpoint hidden state 相似度选样 | Xia et al., ICML 2024 |
| RHO-LOSS | 在线选样 | holdout-loss 驱动的样本优先级排序 | Mindermann et al., NeurIPS 2022 |
| DSIR | 分布匹配 | n-gram 分布重要性重采样 | Xie et al., NeurIPS 2023 |
| **DataPrism** | **梯度驱动** | **三段式 TracIn + 重要性采样 + 多目标迭代** | **Ours** |

---

## 项目结构

```
dataprism/
├── config/           # YAML → dataclass 配置系统
├── core/             # DataSelector 基类、注册器、类型定义
├── data/             # 数据加载、流式池、验证集构建
├── models/           # 模型加载、LoRA 管理、梯度捕获
├── influence/        # TracInCP、梯度采集、检查点、影响存储
├── sampling/         # 温度调度、方差追踪、重要性采样
├── training/         # Trainer 子类、数据收集器、回调、训练循环
├── selection/        # 8 种选择器（TracIn/重要性/多目标 + 4基线 + 均匀）
├── evaluation/       # MMLU/GSM8K/HumanEval/MT-Bench/AlpacaEval
├── pipeline/         # Phase 1/2/3 + Full 管线编排
└── utils/            # 日志（含输出捕获）、种子、聚类工具

scripts/              # 入口脚本
tests/                # 单元测试（40 个，CPU 无 GPU 可运行）
configs/              # 15 个 YAML 配置文件
outputs/              # 运行时输出（checkpoints/influences/results/logs）
```

---

## 日志

运行时自动生成两个日志文件：

```
outputs/logs/
├── phase1_20260611_093103.log       # 结构化日志（Python logging）
└── phase1_20260611_093103_full.log  # 完整终端输出（含 tqdm 进度条、训练指标）
```

---

## 实验流程

```bash
# 1. Phase 1：离线筛选噪声和冗余数据
python scripts/run_phase1_tracin.py \
  --config default phase1_tracin data/alpaca --device cuda

# 2. Phase 2：用筛选后数据 + 重要性采样训练
python scripts/run_phase2_sampling.py \
  --config default phase2_importance data/alpaca --device cuda

# 3. Phase 3：多目标验证集驱动迭代重选
python scripts/run_phase3_reselect.py \
  --config default phase3_multieval data/alpaca --device cuda

# 4. 评估
python scripts/run_evaluation.py \
  --model_path outputs/phase2_training/final_model \
  --benchmarks mmlu gsm8k --device cuda

# 5. 运行基线对比
python scripts/run_baselines.py \
  --config default data/alpaca --device cuda
```

---

## 引用

### 核心参考

- **TracIn**: Garima Pruthi, Frederick Liu, Satyen Kale, Mukund Sundararajan. *"Estimating Training Data Influence by Tracing Gradient Descent."* NeurIPS 2020.
- **Importance Sampling**: Angelos Katharopoulos & François Fleuret. *"Not All Samples Are Created Equal: Deep Learning with Importance Sampling."* ICML 2018.
- **FLDebugger**: Anran Li, Lan Zhang, Junhao Wang, et al. *"Efficient Federated-Learning Model Debugging."* ICDE 2021.

### 基线参考

- **LESS**: Mengzhou Xia, et al. *"LESS: Selecting Influential Data for Targeted Instruction Tuning."* ICML 2024.
- **RHO-LOSS**: Sören Mindermann, et al. *"Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt."* NeurIPS 2022.
- **DSIR**: Sang Michael Xie, et al. *"Data Selection for Language Models via Importance Resampling."* NeurIPS 2023.

### BibTeX

```bibtex
@software{dataprism2024,
  title  = {DataPrism: A Gradient-Driven Data Selection Framework for LLM Fine-Tuning},
  year   = {2024},
  url    = {https://github.com/ivanW2353/DataPrism},
}
```
