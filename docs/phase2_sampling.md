# Phase 2：在线重要性采样训练

## 概述

Phase 2 在训练过程中动态选择"难例"——每一步从数据池中预采样候选集，仅做前向传播计算 loss，以 softmax(loss/τ) 为概率采样目标 batch，聚焦高信息量样本加速收敛。

**核心思想**：梯度大的样本对参数更新贡献大，而 loss 是梯度范数的上界（cross-entropy 下两者成正比）。因此选 loss 高的样本等同选高梯度的样本。

## 运行命令

```bash
python scripts/run_phase2_sampling.py \
  --config default phase2_importance data/alpaca \
  --override phase2.initial_tau=1.0 \
            phase2.candidate_multiplier=4 \
            training.num_epochs=3 \
  --device cuda
```

## 执行流程

```
scripts/run_phase2_sampling.py
│
├─ 1-3. 配置/模型/数据加载 (同 Phase 1)
│
├─ 4. Phase2Pipeline.run()                       # pipeline/phase2_pipeline.py
│   │
│   ├─ 4a. StreamingDataPool(dataset)            # data/streaming_pool.py
│   │   基于 HuggingFace Dataset 的零拷贝索引访问
│   │   sample_candidates(batch_size, multiplier)
│   │     → 返回 multiplier × batch_size 个随机索引
│   │
│   ├─ 4b. TemperatureScheduler(...)             # sampling/temperature_scheduler.py
│   │   管理 softmax 温度 τ 的退火策略
│   │   策略: linear / cosine / constant
│   │   从 initial_tau 退火到 tau_min
│   │
│   ├─ 4c. VarianceTracker(...)                  # sampling/variance_tracker.py
│   │   监测重要性采样的方差缩减率 η
│   │   η = 1 - Var_importance / Var_uniform
│   │   当 η < threshold → 退回到均匀采样 (避免无用功)
│   │
│   ├─ 4d. ImportanceSampler(...)                # sampling/importance_sampler.py
│   │   核心采样逻辑
│   │
│   └─ 4e. DataPrismTrainer(importance_sampler)  # training/trainer.py
│       │
│       │   覆盖 get_train_dataloader()
│       │     → ImportanceDataLoader
│       │       __iter__ 每次调用 sampler.sample_batch()
│       │
│       │   覆盖 compute_loss()
│       │     加权 loss = mean(loss * importance_weight)
│       │
│       └── 训练循环 ────────────────────────────────
│           for step in range(num_steps):
│               │
│               ├── Step 1: 检查是否该用均匀采样
│               │   if variance_tracker.should_use_uniform:
│               │       return uniform_sample(batch_size)
│               │       └─ 方差缩减不显著，不做无用功
│               │
│               ├── Step 2: 预采样候选集
│               │   candidates = pool.sample_candidates(batch_size, multiplier=4)
│               │   └─ 从 N 条候选池中随机抽 4×B 条
│               │
│               ├── Step 3: 前向计算 loss（无梯度）
│               │   with torch.no_grad():
│               │       for each candidate:
│               │           loss[i] = model(input_i).loss
│               │   └─ 纯前向，零额外开销
│               │
│               ├── Step 4: 计算采样概率
│               │   τ = tau_scheduler.get_tau(step)
│               │   probs = softmax(losses / τ)     ← 温度退火
│               │   └─ 高 τ → 接近均匀 (早期探索)
│               │   └─ 低 τ → 聚焦高 loss (后期利用)
│               │
│               ├── Step 5: 采样目标 batch
│               │   selected = np.random.choice(candidates, p=probs)
│               │
│               ├── Step 6: 计算重要性权重
│               │   importance_weights = 1 / (probs[selected] × n_candidates)
│               │   clip(importance_weights, max=10.0)  ← 防止极端权重
│               │   └─ 保证梯度估计无偏
│               │
│               ├── Step 7: 前向 + 加权反向
│               │   loss = model(selected_batch).loss
│               │   weighted_loss = mean(loss * importance_weights)
│               │   weighted_loss.backward()
│               │   optimizer.step()
│               │
│               └── Step 8: 更新方差追踪器
│                   variance_tracker.update_from_losses(
│                       selected_losses, sampling_probs
│                   )
│                   └─ EMA 估算 η
│                      如果 η < threshold → 切回均匀
│
└─ 5. 输出: trained PeftModel
          采样统计: tau 历史, loss 分布, 方差缩减率
```

## 温度退火策略

```
τ(step)
  │
  │ 高τ=1.0 ┐
  │          \      (linear)
  │           \      (cosine)
  │            \
  │ 低τ=0.1    └────────────────→ step
  │
  0               schedule_length
```

- **高 τ** (早期) → 近乎均匀采样，广泛探索数据分布
- **低 τ** (后期) → 聚焦高 loss 样本，针对性提升难例

## 方差缩减监测

```
η = (Var_uniform - Var_importance) / Var_uniform

η > 0.05  → 重要性采样有效，继续
η < 0.05  → 方差缩减不显著，切回均匀采样
每 recheck_interval 步重新评估
```

重要性采样并非总是有效——当 loss 分布均匀时（大量样本难易程度接近），重要性采样反而引入额外方差。方差追踪器自动检测这种退化情况。

## 数据流

```
数据池 (N 条 tokenized 数据)
  │
  │  每步训练:
  │  ① 随机预采样 4×B 候选
  │  ② 前向计算 loss (no grad)
  │  ③ softmax(loss/τ) → 采样概率
  │  ④ 采样 B 条 → 加权训练
  │
  ▼
Trained Model (LoRA)
+ 采样日志: [tau, loss_mean, variance_reduction, ...]
```

## 关键配置

| 参数 | 默认 | 说明 |
|------|------|------|
| `phase2.candidate_multiplier` | 4 | 候选倍数 (×batch_size) |
| `phase2.initial_tau` | 1.0 | 温度初值 |
| `phase2.tau_min` | 0.1 | 温度终值 |
| `phase2.tau_annealing` | `linear` | 退火策略 |
| `phase2.tau_schedule_length` | 5000 | 退火步数 |
| `phase2.variance_ema_alpha` | 0.95 | 方差追踪 EMA 系数 |
| `phase2.variance_reduction_threshold` | 0.05 | 方差缩减下限 |
| `phase2.recheck_interval` | 100 | 重新检查间隔 |
| `phase2.importance_weight_clip` | 10.0 | 权重截断上限 |

## 预期效果

- 相同 wall-clock 时间，训练 loss 下降提速 30-50%
- 下游任务指标提升 5-10%
- 额外开销：仅候选集前向（无梯度），接近零成本
