# 完整管线：DataPrism 三段式数据选择

## 概述

DataPrism 的三段式管线串联 Phase 1、Phase 2、Phase 3，覆盖数据选择的全生命周期：训练前离线筛选 → 训练中在线采样 → 训练后迭代重选。

```
原始数据池 (10万+)
       │
       ▼
┌─────────────────────────────────────────────┐
│           DataPrism 完整管线                  │
│                                              │
│  Phase 1  ──→  Phase 2  ──→  Phase 3        │
│  离线质控       在线采样       迭代重选        │
│                                              │
│  去噪去重       聚焦难例       多目标对齐      │
│  数据量↓70%    收敛加速30%    维度光谱透明     │
│                                              │
└─────────────────────────────────────────────┘
       │
       ▼
高质量子集 (<20% 原始数据) + 影响力分析报告
```

## 运行命令

```bash
# 完整三段式管线
python scripts/run_pipeline.py \
  --config default phase1_tracin phase2_importance phase3_multieval \
  --device cuda

# 仅 Phase 1 + Phase 2（跳过迭代重选）
python scripts/run_pipeline.py \
  --config default phase1_tracin phase2_importance \
  --override phase3.enabled=false \
  --device cuda

# 仅 Phase 2（直接重要性采样训练）
python scripts/run_pipeline.py \
  --config default phase2_importance \
  --override phase1.enabled=false phase3.enabled=false \
  --device cuda
```

## 执行流程

```
scripts/run_pipeline.py
│
├─ Phase 0: 加载
│   seed_everything(config.seed)
│   load_model_and_tokenizer(config.model)
│   load_and_prepare_dataset(config.data)
│   └─ dataset_sizes["raw"] = N
│
├─ Phase 1: 离线质控 (if enabled)
│   Phase1Pipeline.run(model, tokenizer, raw_dataset)
│   │
│   │   apply_lora → SFT训练 → 保存checkpoints
│   │   → TracInCP 自影响计算
│   │   → 异常检测 + 冗余聚类
│   │   → 筛选子集
│   │
│   └─ 输出: peft_model, filtered_dataset (~50% 原始量)
│           checkpoint_manager (传递给 Phase 3 复用)
│
├─ Phase 2: 在线采样 (if enabled)
│   Phase2Pipeline.run(peft_model, tokenizer, dataset)
│   │
│   │   StreamingDataPool → ImportanceSampler
│   │   → TemperatureScheduler + VarianceTracker
│   │   → 重要性加权训练
│   │
│   └─ 输出: trained_model
│           采样统计: tau, loss分布, 方差缩减率
│
├─ Phase 3: 迭代重选 (if enabled)
│   Phase3Pipeline.run(model, tokenizer, dataset, checkpoint_manager)
│   │
│   │   ValidationSetBuilder → 4维验证集
│   │   → TracInVS 多目标影响力
│   │   → 样本分类 (proponent/opponent/neutral)
│   │   → 迭代加权训练 (max_rounds 轮)
│   │
│   └─ 输出: final_model + 影响力光谱报告
│
└─ 结果汇总
    {
      "dataset_sizes": {"raw": 52000, "after_phase1": 26000, "final": 26000},
      "phase_stats": {
        "phase1": {"checkpoints_saved": 20, "samples_retained": 26000},
        "phase2": {"variance_reduction": 0.15, "tau_final": 0.3},
        "phase3": {"n_proponent": 120, "n_opponent": 80, "n_neutral": 300},
      },
      "final_model": peft_model
    }
```

## Phase 间数据流

```
Phase 0:    原始数据集 (52K)
                │
Phase 1:        ▼
            TracInCP 筛选
            移除离群 (~5%) + 冗余 (~45%)
                │
                ▼
            筛选后数据集 (~26K)    ← 附带 influence_score, influence_label
                │
Phase 2:        ▼
            重要性采样训练
            每步动态选样
            聚焦难例
                │
                ▼
            LoRA 模型 (已训练)     ← 附带采样统计日志
                │
Phase 3:        ▼
            多目标迭代重选
            验证集驱动权重更新
            N 轮迭代
                │
                ▼
            最终模型 + 影响力光谱    ← 附带可解释数据影响报告
```

## 各 Phase 可独立运行

三个 Phase 均可通过配置开关独立启用/禁用：

```yaml
# configs/default.yaml 中的开关
phase1:
  enabled: true    # 改为 false 跳过

phase2:
  enabled: true    # 改为 false 跳过

phase3:
  enabled: true    # 改为 false 跳过
```

也可以命令行覆盖：

```bash
--override phase1.enabled=false phase2.enabled=true phase3.enabled=false
```

## 运行时输出

```
outputs/
├── checkpoints/
│   ├── phase1/              # LoRA 检查点 (Phase 1 & 3 共用)
│   └── phase3/              # 迭代轮次检查点
├── influences/
│   ├── phase1_self_influence.npz    # 自影响分数
│   └── phase3_multi.npz             # 多维度 TracInVS
├── logs/
│   ├── *_full.log            # 完整终端输出
│   └── *.log                 # 结构化日志
├── results/
│   └── eval_results.json     # 评估结果
├── phase1_sft/               # Phase 1 SFT 模型
├── phase2_training/          # Phase 2 训练模型
└── phase3_round{1..N}/       # Phase 3 各轮模型
```
