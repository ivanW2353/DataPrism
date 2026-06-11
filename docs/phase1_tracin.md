# Phase 1：LoRA 空间 TracInCP 离线数据质控

## 概述

Phase 1 在训练前对原始数据池进行离线筛选，剔除噪声/错误标注样本和冗余样本，输出一个高质量核心子集。

**核心思想**：样本对自身 loss 的梯度贡献越大（self-influence 越高），模型越需要"硬记"它，越可能是错误标注。

## 运行命令

```bash
python scripts/run_phase1_tracin.py \
  --config default phase1_tracin data/alpaca \
  --override phase1.num_epochs=2 \
            phase1.outlier_percentile=95.0 \
  --device cuda
```

## 执行流程

```
scripts/run_phase1_tracin.py                    # 入口脚本
│
├─ 1. 配置加载
│   load_config(config_paths, overrides)        # config/loader.py
│   └─ YAML 多层合并 → CLI 覆盖 → DataPrismConfig 实例
│
├─ 2. 模型加载
│   load_model_and_tokenizer(config.model)       # models/model_registry.py
│   └─ AutoModelForCausalLM.from_pretrained(local_path)
│       Llama-3.1-8B-Instruct (bf16) → 8B 参数, ~16GB 显存
│
├─ 3. 数据加载
│   load_and_prepare_dataset(config.data)        # data/dataset.py
│   ├─ load_from_disk(local_path)                # 从本地加载
│   ├─ 格式归一化 (alpaca → prompt+response)
│   ├─ Tokenization (max_seq_length)
│   └─ labels 构造 (prompt 部分 mask 为 -100，只算 response loss)
│
├─ 4. Phase1Pipeline.run()                      # pipeline/phase1_pipeline.py
│   │
│   ├─ 4a. apply_lora(model, config.lora)       # models/lora_manager.py
│   │   LoraConfig(r=64, α=128, dropout=0.1)
│   │   目标层: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
│   │   可训参数: 168M / 8B = 2.05%
│   │
│   ├─ 4b. Initial SFT 训练（生成检查点）
│   │   │
│   │   ├─ CheckpointManager(checkpoint_dir, max_checkpoints=20)
│   │   │   influence/checkpoint_manager.py
│   │   │   管理 LoRA 检查点生命周期：保存、加载、滑动窗口剪枝
│   │   │
│   │   ├─ DataPrismTrainer(...)                 # training/trainer.py
│   │   │   HF Trainer 子类，支持：
│   │   │   - 自定义 data_collator
│   │   │   - LoRA checkpoint 回调保存
│   │   │
│   │   └─ 训练循环:
│   │       for step in range(num_steps):
│   │           batch = next(dataloader)
│   │           loss = model(batch).loss
│   │           loss.backward()                  ← 仅 LoRA 参数需要梯度
│   │           optimizer.step()
│   │           if step % checkpoint_every_n_steps == 0:
│   │               model.save_pretrained(f"checkpoint-{step}")
│   │       └─ 只保存 adapter_model.safetensors (~640MB/ckpt)
│   │
│   ├─ 4c. 初始化 TracInCP 引擎
│   │   GradientCollector(model)                 # models/gradient_hooks.py
│   │   └─ 识别所有 "lora_" 参数 → 168M 维梯度空间
│   │   TracInCP(model, checkpoint_manager, collector)
│   │
│   └─ 4d. TracInSelector.select()              # selection/tracin_selector.py
│       │
│       ├── Step 1: 计算 Self-Influence ────────────────
│       │   tracin_cp.compute_self_influence()   # influence/tracin_cp.py
│       │   │
│       │   │   SelfInfluence(z) = Σᵢ ηᵢ · ∇ℓ(z;θᵢ) · ∇ℓ(z;θᵢ)
│       │   │
│       │   │   算法:
│       │   │   ① 保存当前 LoRA 权重到 CPU (仅 lora_ 参数)
│       │   │   ② for each checkpoint:
│       │   │        checkpoint_manager.load(step)  ← 注入该 ckpt 的 LoRA 权重
│       │   │        for each sample:
│       │   │            input → model → loss → loss.backward()
│       │   │            grad = GradientCollector.get_flattened_gradients()
│       │   │            self_infl = grad · grad     ← 168M 维内积
│       │   │            score[样本] += η × self_infl
│       │   │   ③ load_state_dict(original_lora)  ← 恢复原始权重
│       │   │
│       │   │   输出: scores[n_samples]  每样本一个标量分数
│       │   │   存储: InfluenceStore → outputs/influences/phase1_self_influence.npz
│       │   │
│       │   │   关键优化:
│       │   │   - 所有操作限于 LoRA 空间 (168M维 vs 8B维, 4000× 缩减)
│       │   │   - 只存储标量分数, 不存完整梯度向量
│       │   │   - 逐样本计算避免 batch 梯度平均
│       │   │
│       ├── Step 2: 异常检测 ──────────────────────────
│       │   threshold = np.percentile(scores, outlier_percentile)
│       │   scores ≥ threshold → OUTLIER (疑似错误标注/格式损坏)
│       │
│       ├── Step 3: 冗余去除 ──────────────────────────
│       │   cluster_by_similarity(scores[~outlier])  # utils/cluster_utils.py
│       │   │
│       │   │   对非异常样本按 self-influence 分数聚类 (KMeans)
│       │   │   每类选距离中心最近的 → REPRESENTATIVE
│       │   │   其余同类的 → REDUNDANT
│       │
│       └── Step 4: 输出筛选结果 ──────────────────────
│           filtered = dataset[非OUTLIER且非REDUNDANT]
│           filtered.columns += [influence_score, influence_label, original_index]
│
└─ 5. 输出结果
    500 → ~450 samples (90% 保留)
    Clean/Rep: 450 | Outlier: 25 | Redundant: 25
```

## 数据流

```
原始数据 (500条)
  │
  ├─ SFT 训练: batch=1, grad_accum=2, 1 epoch
  │   250 步, ~2.5 分钟 (RTX 4080)
  │   输出: 10 个 LoRA checkpoint (各 ~640MB)
  │
  ├─ TracInCP 计算: 10 ckpt × 500 样本
  │   每样本: forward + backward → 168M 维梯度内积
  │   输出: 500 个 self-influence 标量
  │
  └─ 筛选: score ≥ 95%分位 → 异常
           相似分数聚类 → 冗余
           输出: ~450 条 Clean/Rep 数据
```

## 关键配置

| 参数 | 默认 | 说明 |
|------|------|------|
| `phase1.num_epochs` | 2 | SFT 轮数（生成检查点用） |
| `phase1.checkpoint_every_n_steps` | 50 | 检查点保存间隔 |
| `phase1.max_checkpoints` | 20 | 最大检查点数（滑动窗口） |
| `phase1.self_influence_method` | `dot_product` | 自影响计算方式 |
| `phase1.normalize_gradients` | true | L2 归一化梯度向量 |
| `phase1.outlier_percentile` | 95.0 | 异常阈值百分位 |
| `phase1.redundancy_method` | `kmeans` | 聚类算法 |
| `phase1.redundancy_similarity_threshold` | 0.85 | 冗余余弦相似度阈值 |

## 输出文件

```
outputs/
├── checkpoints/phase1/
│   ├── checkpoint-25/adapter_model.safetensors    # LoRA 权重快照
│   ├── checkpoint-50/adapter_model.safetensors
│   └── ...
├── influences/
│   └── phase1_self_influence.npz                  # 自影响分数
└── logs/
    ├── phase1_*.log                               # 结构化日志
    └── phase1_*_full.log                          # 完整终端输出
```
