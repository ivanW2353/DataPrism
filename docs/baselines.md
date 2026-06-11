# 基线方法

## 概述

DataPrism 实现了 5 个基线方法用于对比实验，覆盖了数据选择的主要范式：随机、全量、表征相似度、在线损失、分布匹配。

## 运行命令

```bash
# 运行所有基线
python scripts/run_baselines.py \
  --config default data/alpaca \
  --device cuda

# 运行指定基线
python scripts/run_baselines.py \
  --config default data/alpaca \
  --baseline less \
  --device cuda
```

## 基线列表

| 方法 | 类型 | 论文 | 选择时机 |
|------|------|------|----------|
| Uniform Random | 下界 | — | 离线 |
| Full Data | 上界 | — | — |
| LESS | 表征相似度 | ICML 2024 | 离线 |
| RHO-LOSS | 在线损失 | NeurIPS 2022 | 在线 |
| DSIR | 分布匹配 | NeurIPS 2023 | 离线 |

---

## 1. Uniform Random（随机均匀采样）

**文件**: `selection/uniform_selector.py`

**原理**: 从数据池中随机抽取固定比例的子集，不做任何智能筛选。作为实验下界。

```
原始 N → random.choice(N, k=N×fraction) → k 条
```

**配置**:
```yaml
selection:
  method: uniform
  fraction: 0.2    # 保留比例
```

---

## 2. Full Data（全量数据训练）

**原理**: 使用全部可用数据训练，不做任何筛选。作为实验上界。

**配置**:
```yaml
selection:
  method: full
```

---

## 3. LESS（表征相似度选样）

**文件**: `selection/less_selector.py`

**论文**: Xia et al., "LESS: Selecting Influential Data for Targeted Instruction Tuning", ICML 2024.

**原理**: 使用 LoRA checkpoint ensemble 的最后一层 hidden state 作为每个样本的表征，选择与验证集表征最相似的训练样本。

```
流程:
  ① 对小规模 warmup 集训练几个 LoRA checkpoint
  ② 提取每样本的 last hidden state (mean-pool over sequence)
  ③ 计算训练样本与验证集的余弦相似度矩阵
  ④ 按平均相似度排序 → 选 top-k
```

**关键差异 vs DataPrism**:
- LESS 用 **表征相似度**（hidden state 距离）
- DataPrism 用 **梯度内积**（TracInCP，保留了梯度方向信息）

LESS 丢失了"这个样本对模型参数更新的方向影响"这一关键信息。两个语义相似的问题可能有完全不同的梯度方向。

**配置**:
```yaml
selection:
  method: less
  fraction: 0.2
  less:
    similarity_metric: cosine
    representation_layer: -1    # 最后一层
```

---

## 4. RHO-LOSS（在线 holdout-loss 选样）

**文件**: `selection/rho_loss_selector.py`

**论文**: Mindermann et al., "Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt", NeurIPS 2022.

**原理**: 维护一个小型 holdout 集，估计每个训练样本对 holdout loss 的减少量（可学习 + 值得学习 + 尚未学会），优先选择能最大减少 holdout loss 的样本。

```
流程:
  ① 从数据中切分 holdout 集 (5%)
  ② 计算 holdout 集 loss 基线
  ③ 对每个候选样本:
      用影响函数近似估算: 训练该样本后 holdout loss 的预期变化
      选取减少 holdout loss 最多的样本
```

**关键差异 vs DataPrism**:
- RHO-LOSS 用 **单一 holdout loss** 作为优化目标
- DataPrism Phase 3 用 **多维度验证集 + 梯度内积**，同时优化多个能力维度

**配置**:
```yaml
selection:
  method: rho_loss
  fraction: 0.2
  rho_loss:
    holdout_fraction: 0.05
    temperature: 1.0
```

---

## 5. DSIR（分布匹配重要性重采样）

**文件**: `selection/dsir_selector.py`

**论文**: Xie et al., "Data Selection for Language Models via Importance Resampling", NeurIPS 2023.

**原理**: 基于 n-gram 特征分布，计算原始数据池与目标高质量数据之间的重要性权重，重采样原始数据使分布向目标靠近。

```
流程:
  ① 构建原始数据的 n-gram 词频分布 P_source
  ② 构建目标数据 (如 curated 高质量集) 的 n-gram 分布 P_target
  ③ 每样本重要性权重 = Π P_target(ngram) / P_source(ngram)
  ④ 按权重采样 → 选 top-k
```

**关键差异 vs DataPrism**:
- DSIR 基于 **浅层文本特征**（n-gram 分布），无需模型
- DataPrism 基于 **深层梯度信号**，能捕捉任务相关性而非仅仅词法相似性

DSIR 不需要 GPU，计算极快，但对语义层面的数据质量不敏感。

**配置**:
```yaml
selection:
  method: dsir
  fraction: 0.2
  dsir:
    feature_type: ngram
    ngram_size: 3
    num_features: 10000
    importance_smoothing: 0.01
```

---

## 基线 vs DataPrism 对比

| 维度 | Uniform | Full | LESS | RHO-LOSS | DSIR | DataPrism |
|------|---------|------|------|----------|------|-----------|
| 信号类型 | 随机 | — | Hidden State | Loss | n-gram | **梯度内积** |
| 选择时机 | 离线 | — | 离线 | 在线 | 离线 | **离线+在线+迭代** |
| 需模型 | 否 | 否 | 是 | 是 | 否 | 是 |
| 多目标 | 否 | 否 | 否 | 否 | 否 | **是** |
| 可解释性 | 无 | 无 | 低 | 中 | 低 | **高(影响力光谱)** |
| 计算成本 | O(1) | O(1) | O(N×d) | O(N×M) | O(N×V) | O(N×K×P) |
| 去噪 | 否 | 否 | 间接 | 否 | 否 | **是** |
| 去重 | 否 | 否 | 否 | 否 | 否 | **是** |
