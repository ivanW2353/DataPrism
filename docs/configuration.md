# 配置系统

## 概述

DataPrism 使用 YAML + dataclass 分层配置系统。所有超参数通过类型化的 dataclass 定义，YAML 文件提供默认值，CLI `--override` 进行实验级覆盖。

**核心类**: `config/dataclass.py` — `DataPrismConfig` 及 8 个子配置类
**加载器**: `config/loader.py` — YAML 合并 + CLI 覆盖 + 类型转换

## 配置层级

```
configs/default.yaml          ← 全局默认 (最低优先级)
  │
  ├─ configs/model/xxx.yaml   ← 模型特定
  ├─ configs/lora/xxx.yaml    ← LoRA 特定
  ├─ configs/data/xxx.yaml    ← 数据集特定
  ├─ configs/phase1_xxx.yaml  ← Phase 1 特定
  ├─ configs/phase2_xxx.yaml  ← Phase 2 特定
  ├─ configs/phase3_xxx.yaml  ← Phase 3 特定
  │
  └─ CLI --override key=value ← 命令行覆盖 (最高优先级)
```

合并规则：**deep merge，后者覆盖前者**。

## 配置文件列表

```
configs/
├── default.yaml                  # 全局默认
├── model/
│   ├── llama3_8b.yaml           # Llama-3.1-8B-Instruct
│   └── qwen2.5_7b.yaml          # Qwen2.5-7B-Instruct
├── lora/
│   └── default.yaml             # LoRA (r=64, α=128)
├── data/
│   ├── alpaca.yaml              # Alpaca-52K
│   ├── wizardlm.yaml            # WizardLM-70K
│   └── openorca.yaml            # OpenOrca
├── phase1_tracin.yaml           # Phase 1 参数
├── phase2_importance.yaml       # Phase 2 参数
├── phase3_multieval.yaml        # Phase 3 + 评估参数
└── baseline/
    ├── uniform.yaml             # 随机采样
    ├── full_data.yaml           # 全量数据
    ├── less.yaml                # LESS
    ├── rho_loss.yaml            # RHO-LOSS
    └── dsir.yaml                # DSIR
```

## 使用方式

### 加载配置

```python
from dataprism.config import load_config

# 加载多个配置文件（后覆盖前）
config = load_config([
    "default",              # configs/default.yaml
    "model/llama3_8b",      # configs/model/llama3_8b.yaml
    "phase1_tracin",        # configs/phase1_tracin.yaml
])

# 带 CLI 覆盖
config = load_config(
    config_paths=["default", "phase1_tracin"],
    overrides={"phase1.num_epochs": "3", "training.learning_rate": "1e-4"}
)
```

### CLI 使用

```bash
# 通过 --config 指定多个配置文件
python scripts/run_pipeline.py \
  --config default model/llama3_8b data/alpaca phase1_tracin

# 通过 --override 覆盖任意参数
python scripts/run_pipeline.py \
  --config default phase1_tracin \
  --override phase1.outlier_percentile=99.0 \
            phase2.initial_tau=0.5 \
            lora.r=32 \
            training.num_epochs=5
```

### 清理旧日志

训练输出保存到 `outputs/`，需要定期清理：

```bash
rm -rf outputs/checkpoints outputs/logs
```

## 完整参数参考

### ModelConfig（模型）

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `model.name` | str | `meta-llama/Meta-Llama-3-8B` | 模型名或本地路径 |
| `model.torch_dtype` | str | `bfloat16` | float16/bfloat16/float32 |
| `model.use_flash_attention_2` | bool | true | FlashAttention2 |
| `model.trust_remote_code` | bool | false | 允许远程代码 |
| `model.device_map` | str | `auto` | 设备分配策略 |

### LoRAConfig（LoRA 适配器）

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `lora.r` | int | 64 | LoRA rank |
| `lora.alpha` | int | 128 | LoRA alpha (scaling factor) |
| `lora.dropout` | float | 0.1 | LoRA dropout |
| `lora.target_modules` | list | [q_proj,k_proj,v_proj,o_proj] | 注意力投影层 |
| `lora.target_modules_extra` | list | [gate_proj,up_proj,down_proj] | MLP 投影层 |
| `lora.bias` | str | `none` | bias 处理方式 |

### TrainingConfig（训练）

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `training.num_epochs` | int | 3 | 训练轮数 |
| `training.per_device_train_batch_size` | int | 4 | 每卡 batch size |
| `training.per_device_eval_batch_size` | int | 8 | 评估 batch size |
| `training.gradient_accumulation_steps` | int | 4 | 梯度累积 |
| `training.learning_rate` | float | 2e-4 | 学习率 |
| `training.weight_decay` | float | 0.01 | 权重衰减 |
| `training.warmup_ratio` | float | 0.1 | warmup 比例 |
| `training.lr_scheduler_type` | str | `cosine` | 学习率调度器 |
| `training.max_seq_length` | int | 2048 | 最大序列长度 |
| `training.max_grad_norm` | float | 1.0 | 梯度裁剪 |

### DataConfig（数据集）

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `data.name` | str | `tatsu-lab/alpaca` | HF 数据集名 |
| `data.local_path` | str | null | 本地路径 (优先) |
| `data.num_samples` | int | null | 采样数 (null=全部) |
| `data.prompt_template` | str | `alpaca` | 格式模板 |
| `data.response_only_loss` | bool | true | 仅对 response 算 loss |

### Phase1TracInConfig（离线质控）

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `phase1.enabled` | bool | true | 启用开关 |
| `phase1.num_epochs` | int | 2 | SFT 训练轮数 |
| `phase1.checkpoint_every_n_steps` | int | 50 | 检查点间隔 |
| `phase1.max_checkpoints` | int | 20 | 最大检查点数 |
| `phase1.self_influence_method` | str | `dot_product` | 自影响计算方式 |
| `phase1.normalize_gradients` | bool | true | L2 归一化 |
| `phase1.outlier_percentile` | float | 95.0 | 异常阈值百分位 |
| `phase1.redundancy_method` | str | `kmeans` | 聚类方法 |
| `phase1.redundancy_similarity_threshold` | float | 0.85 | 冗余相似度阈值 |
| `phase1.max_samples_after_redundancy` | int | 50000 | 冗余去除后上限 |

### Phase2ImportanceConfig（在线采样）

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `phase2.enabled` | bool | true | 启用开关 |
| `phase2.candidate_multiplier` | int | 4 | 候选倍数 |
| `phase2.initial_tau` | float | 1.0 | 温度初值 |
| `phase2.tau_min` | float | 0.1 | 温度终值 |
| `phase2.tau_annealing` | str | `linear` | 退火策略 |
| `phase2.tau_schedule_length` | int | 5000 | 退火步数 |
| `phase2.variance_ema_alpha` | float | 0.95 | 方差追踪 EMA |
| `phase2.variance_reduction_threshold` | float | 0.05 | 方差缩减下限 |
| `phase2.recheck_interval` | int | 100 | 重新检查间隔 |
| `phase2.importance_weight_clip` | float | 10.0 | 权重截断 |

### Phase3MultiEvalConfig（多目标重选）

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `phase3.enabled` | bool | true | 启用开关 |
| `phase3.selection_fraction` | float | 0.5 | 每轮保留比例 |
| `phase3.max_rounds` | int | 5 | 最大迭代轮数 |
| `phase3.proponent_weight_boost` | float | 2.0 | Proponent 权重提升 |
| `phase3.opponent_weight_decay` | float | 0.5 | Opponent 权重衰减 |
| `phase3.neutral_label_threshold` | float | 0.05 | 中性阈值 |

### EvaluationConfig（评估）

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `evaluation.benchmarks` | list | [mmlu,gsm8k,...] | 评估基准列表 |
| `evaluation.mmlu_num_fewshot` | int | 5 | MMLU few-shot |
| `evaluation.gsm8k_num_fewshot` | int | 8 | GSM8K few-shot |
| `evaluation.gsm8k_use_cot` | bool | true | GSM8K chain-of-thought |

## 添加新配置

1. 在 `config/dataclass.py` 中添加对应的 dataclass 字段
2. 在 `configs/` 下创建对应的 YAML 文件
3. 在 `config/loader.py` 的 `sub_config_map` 中注册类型映射

## 配置验证

```python
config = DataPrismConfig()   # 使用所有默认值
config._validate()           # 自动触发 (__post_init__)

# 验证规则：
# - Phase 3 lambda_weights 之和 ≈ 1.0
# - Phase 2 tau_annealing ∈ {linear, cosine, constant}
```
