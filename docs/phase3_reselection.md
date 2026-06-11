# Phase 3：验证集驱动的多目标迭代数据重选

## 概述

Phase 3 在训练过程中持续对齐特定的下游能力维度。通过构建多维验证集（推理、安全、对话、事实），计算每个训练样本对各维度的 TracInVS 影响力评分，实现可解释的多目标数据选择。

**核心思想**：同一批数据对不同能力有不同影响——有些提升推理但损害安全，有些提高对话但降低事实准确性。Phase 3 将这种冲突显式化，让使用者基于权重做透明取舍。

## 运行命令

```bash
python scripts/run_phase3_reselect.py \
  --config default phase3_multieval data/alpaca \
  --device cuda
```

## 执行流程

```
scripts/run_phase3_reselect.py
│
├─ 1-3. 配置/模型/数据加载 (同 Phase 1)
│
├─ 4. Phase3Pipeline.run()                        # pipeline/phase3_pipeline.py
│   │
│   ├─ 4a. ValidationSetBuilder(config, tokenizer) # data/validation_sets.py
│   │   构建多维验证集:
│   │   │
│   │   ├─ V_reasoning  ← GSM8K (test, 500 条)    数学推理
│   │   ├─ V_safety     ← PKU-SafeRLHF (500 条)    安全性
│   │   ├─ V_chat       ← MT-Bench (500 条)        对话质量
│   │   └─ V_factual    ← HotpotQA (500 条)        事实准确性
│   │
│   ├─ 4b. 初始化 TracInCP + GradientCollector
│   │   重用 Phase 1 的 CheckpointManager
│   │
│   ├─ 4c. MultiObjectiveSelector.select()         # selection/multi_obj_selector.py
│   │   │
│   │   ├── 多维度影响力计算 ──────────────────────────
│   │   │   tracin_cp.compute_multi_objective_influence()
│   │   │   │
│   │   │   │   对每个验证维度 j:
│   │   │   │     $$\text{Score}(z, V_j) = \sum_{i=1}^{K} \eta_i \cdot \nabla_\theta \ell(z; \theta_i) \cdot \nabla_\theta \ell(V_j; \theta_i)$$
│   │   │   │     │
│   │   │   │     │  $\nabla\ell(z;\theta_i)$   ← 训练样本 z 在检查点 i 的 LoRA 梯度
│   │   │   │     │  $\nabla\ell(V_j;\theta_i)$ ← 验证集 Vⱼ 在检查点 i 的平均 LoRA 梯度
│   │   │   │     │  dot product              ← 两者方向一致 → 正影响; 相反 → 负影响
│   │   │   │
│   │   │   │   聚合:
│   │   │   │     $$\text{TotalScore}(z) = \sum_j \lambda_j \cdot \text{Score}(z, V_j)$$
│   │   │   │     │
│   │   │   │     │  λ_reasoning = 0.30  (推理最重要)
│   │   │   │     │  λ_safety    = 0.25
│   │   │   │     │  λ_chat      = 0.25
│   │   │   │     │  λ_factual   = 0.20
│   │   │   │
│   │   │   │   输出: per-dimension scores + total_score
│   │   │   │   存储: InfluenceStore → outputs/influences/phase3_multi.npz
│   │   │
│   │   ├── 样本分类 ────────────────────────────────
│   │   │   threshold = neutral_label_threshold (0.05)
│   │   │   │
│   │   │   │  TotalScore > +0.05  → PROPONENT  (助推器)
│   │   │   │  TotalScore < -0.05  → OPPONENT   (抑制剂)
│   │   │   │  |TotalScore| ≤ 0.05 → NEUTRAL    (无关)
│   │   │
│   │   └── 初始权重分配 ────────────────────────────
│   │       proponents:  weight = 1.0 + score
│   │       opponents:   weight = max(0.01, 1.0 + score)
│   │       neutrals:    weight = 1.0
│   │       └─ 归一化使均值 = 1.0
│   │
│   └─ 4d. 迭代训练循环
│       │
│       │   for round in 1..max_rounds:
│       │     │
│       │     ├── 加权采样训练 ────────────────────────
│       │     │   DataPrismTrainer(data_weights=weights)
│       │     │   └─ WeightedRandomSampler(weights)
│       │     │      proponents 采样概率 ↑
│       │     │      opponents 采样概率 ↓
│       │     │   训练 1 epoch (lr 每轮衰减 50%)
│       │     │
│       │     ├── 保存 checkpoint (for TracInVS)
│       │     │
│       │     └── 更新权重 (除最后一轮)
│       │         selector.update_weights(model, round)
│       │         │
│       │         │   for each sample:
│       │         │     if PROPONENT:  weight *= 2.0   ↑ boost
│       │         │     if OPPONENT:   weight *= 0.5   ↓ decay
│       │         │     if NEUTRAL:    weight unchanged
│       │         │   normalize → sum to 1
│       │
│       └─ 最终报告:
│           n_proponent: 120  (24%)
│           n_opponent:   80  (16%)
│           n_neutral:   300  (60%)
│
└─ 5. 输出: trained PeftModel + 影响力光谱报告
```

## 影响力光谱

```
          推理(+0.15) 安全(+0.08) 对话(-0.12) 事实(+0.03)     Total
          ─────────────────────────────────────────────────    ─────
样本 A:    ████████░░  ████░░░░░░  ██████░░░░  █░░░░░░░░░    +0.035  (NEUTRAL)
样本 B:    ██████████  █████████░  █████████░  ████████░░    +0.250  (PROPONENT)
样本 C:    ████░░░░░░  ░░░░░░░░░░  ██░░░░░░░░  ██████████    -0.060  (OPPONENT)

样本 A: 综合中性，但提升推理稍损害对话 → 保留但降权
样本 B: 全面提升，强正影响               → 重点保留，升权
样本 C: 正面维度弱，负影响大              → 降权或移除
```

每轮迭代后 proponents 被不断上采样，opponents 被逐渐边缘化，模型持续向目标维度对齐。

## 数据流

```
训练数据 (N 条)
  │
  │  ① 构建 4 维验证集
  │  ② TracInVS 计算: N × 4 个影响力分数
  │  ③ 聚合 + 分类 → proponents/opponents/neutrals
  │  ④ 初始权重分配
  │
  │  ⑤ 迭代训练 (max_rounds 轮):
  │     加权采样训练 1 epoch
  │     更新权重 (boost proponents, decay opponents)
  │
  ▼
Trained Model + 影响力光谱报告
```

## 关键配置

| 参数 | 默认 | 说明 |
|------|------|------|
| `phase3.validation_sets` | 4 维度 | 验证集来源和采样数 |
| `phase3.lambda_weights` | 0.30,0.25,0.25,0.20 | 各维度重要度权重 |
| `phase3.selection_fraction` | 0.5 | 每轮保留比例 |
| `phase3.max_rounds` | 5 | 最大迭代轮数 |
| `phase3.proponent_weight_boost` | 2.0 | Proponent 权重提升倍数 |
| `phase3.opponent_weight_decay` | 0.5 | Opponent 权重衰减倍数 |
| `phase3.neutral_label_threshold` | 0.05 | 中性标签阈值 |

## 与 FLDebugger 的关系

Phase 3 借鉴 FLDebugger (Li et al., ICDE 2021) 的分层影响力分析思想：
- FLDebugger 在联邦学习中定位问题客户端/数据
- DataPrism Phase 3 在 LLM 微调中定位各能力维度的有利/有害数据

两者都使用"多层影响力度量 → 分类标记 → 定向调整"的范式，DataPrism 将其迁移到 LoRA 梯度空间并扩展到多目标优化场景。
