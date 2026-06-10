# DataPrism：面向 LLM 微调的梯度驱动数据选择框架

---

## 名称释义

**DataPrism（数据棱镜）** — 棱镜能将一束白光分解为不同波长的彩色光谱。类似地，我们的框架将海量微调数据"分解"为不同维度的影响力信号（对模型有帮助的 proponent、有害的 opponent、冗余的 neutral），从而精准筛选出真正有价值的数据子集。

视觉隐喻：一束光（原始数据池）穿过三角棱镜（我们的框架），分解为红/绿/蓝三色光谱（proponent / neutral / opponent），最终汇聚为高质量训练集。

---

## 动机

### 核心问题

大语言模型（LLM）微调面临一个根本矛盾：**数据越多越好**与**高质量标注越来越贵**之间的张力。实际中：

- SFT 数据动辄数万到数十万条，其中大量样本对最终模型能力贡献甚微
- 错误标注、格式损坏、事实性幻觉数据会直接损害微调效果
- 不同下游任务（推理、安全、对话）对"好数据"的定义不同，无法用单一标准判决

### 现有方法的三条脉络

| 脉络 | 代表工作 | 核心思想 | 局限 |
|---|---|---|---|
| 重要性采样 | Katharopoulos et al., ICML 2018 | 梯度范数上界 → 在线偏重高信息量样本 | 集中式小模型，无 LLM 适配 |
| 影响力追溯 | TracIn, Pruthi et al., NeurIPS 2020 | checkpoint 梯度内积 → 量化单样本对预测的影响 | 离线分析，计算成本高 |
| 联邦调试 | FLDebugger, Li et al., ICDE 2021 | 分层影响力分析 → 定位 FL 系统中的问题数据 | 面向分类任务，非 LLM |

### 本项目定位

**DataPrism** 将上述三条脉络统一迁移到 LLM 微调场景，构建一套完整的数据选择管线：

```
  原始数据池 (10万+)
       │
       ▼
  ┌─────────────────────────────┐
  │  DataPrism 棱镜框架           │
  │                              │
  │  ① 离线质控（TracIn 自影响）    │  → 去噪、去重、去冗余
  │  ② 在线采样（梯度上界选样）     │  → 聚焦难例、加速收敛
  │  ③ 迭代对齐（验证集驱动重选）   │  → 持续匹配下游目标
  │                              │
  └─────────────────────────────┘
       │
       ▼
  高质量子集 (<20% 原始数据)
```

---

## 技术路线

### Phase 1：离线数据质量筛选（→ TracIn 迁移）

**目标**：训练前剔除噪声、错误标注、冗余样本。

**方法**：将 TracInCP 迁移到 LoRA 参数空间。

- 对预训练 LLM 加 LoRA 适配器，用全量数据跑 1–3 epoch 初步 SFT
- 每 N 步保存 checkpoint（约 10–20 个）
- 对每个训练样本计算 **self-influence**（样本对自身 loss 的影响）

$$ \text{SelfInfluence}(z) = \sum_{i=1}^{K} \eta_i \; \nabla_\theta^{\text{LoRA}} \ell(z, \theta_i) \cdot \nabla_\theta^{\text{LoRA}} \ell(z, \theta_i) $$

- 自影响异常高 → 可能是错误标注（模型必须"硬记"它）
- 影响向量高度相似 → 冗余样本，聚类后只保留代表

**预期产出**：从 10 万数据中筛选出 2–5 万高质量核心集。

---

### Phase 2：在线重要性采样（→ Katharopoulos 迁移）

**目标**：训练过程中动态聚焦难例，加速收敛。

**方法**：利用 LLM 的 token-level cross-entropy loss 作为重要性评分。

- 每个训练步，从数据池中预先采样 4× batch_size 的候选集
- 仅做前向传播，计算每个候选样本的平均 token loss
- 以 softmax(loss / τ) 为概率分布采样目标 batch
- 用 EMA 跟踪方差缩减率 η；当 η < 阈值时自动切回均匀采样（避免无用功）

**关键优势**：loss 是前向传播的副产品，零额外计算架构改动的开销。

**预期产出**：相同 wall-clock 时间下，训练 loss 下降提速 30%–50%，下游任务指标提升 5%–10%。

---

### Phase 3：验证集驱动的迭代数据重选（→ FLDebugger 分层思想迁移）

**目标**：在多轮训练中持续对齐特定下游能力（推理、安全、对话质量）。

**方法**：构建多目标 TracIn 评分体系。

- 构建多维验证集：`V = {V_reasoning, V_safety, V_chat, V_factual}`
- 每个 epoch 结束后，对每个训练样本计算其对各验证集的 TracInCP score

$$ \text{Score}(z, V_j) = \sum_{i=1}^{K} \nabla \ell(z, \theta_i) \cdot \nabla \ell(V_j, \theta_i) $$

- 根据目标权重 λ_j 聚合：`TotalScore(z) = Σ λ_j · Score(z, V_j)`
- 下一轮训练调高 proponents 的采样权重，压低 opponents

**预期产出**：可解释的数据影响报告 + 面向具体能力的定向数据优化。

---

## 创新点

1. **首次将 TracIn 系列方法系统迁移到 LLM 微调场景**。现有工作（LESS 等）仅使用最后一层表征相似度，丢失了梯度方向信息。我们在 LoRA 空间下保留完整的梯度内积结构。

2. **"离线→在线→迭代"三段式管线**。单一方法各有盲区：TracIn 离线好但在线成本高，重要性采样在线好但缺乏全局视角。三段互补，覆盖数据选择的全生命周期。

3. **多维验证集驱动的可解释数据选择**。不同于现有方法只能优化单一 eval metric，DataPrism 可以为同一批训练数据生成"影响力光谱"——对推理有帮助但可能损害安全性的样本将被显式标记，给使用者透明的取舍权。

4. **LoRA 原生的计算效率**。所有梯度操作限制在 LoRA 参数空间（~几百万维），避免全参数梯度（~70 亿维）的存储和计算灾难。

---

## 实验计划

### 基线方法

| 方法 | 类型 |
|---|---|
| 随机采样 (Uniform) | 下界 |
| Full data training | 上界 |
| LESS (Xia et al., 2024) | 表征相似度选样 |
| RHO-LOSS (Mindermann et al., 2022) | 在线 loss 选样 |
| DSIR (Xie et al., 2023) | 分布匹配选样 |
| **DataPrism (ours)** | 梯度驱动三段式 |

### 模型与数据

- **基座模型**：Llama-3-8B, Qwen2.5-7B（验证跨模型泛化）
- **微调方式**：LoRA (r=64, α=128)
- **训练数据**：Alpaca-52K, WizardLM-70K, OpenOrca（混合多源）
- **评估基准**：MMLU, GSM8K, HumanEval, MT-Bench, AlpacaEval

### 核心实验

1. **消融实验**：Phase 1/2/3 各自的独立贡献
2. **数据压缩率实验**：保留 {5%, 10%, 20%, 50%} 数据的性能曲线
3. **跨任务泛化**：选出的数据子集在不同下游任务上的表现
4. **计算开销分析**：各 Phase 的额外 GPU 小时与 wall-clock 时间
5. **联邦扩展**（optional）：在 FL 设定下验证 DataPrism 的隐私兼容性

---

## 时间规划

| 阶段 | 时间 | 内容 |
|---|---|---|
| **文献调研与原型** | 第 1–2 月 | 复现 LESS、TracIn（LoRA 版本），搭建基线 |
| **Phase 1 实现与验证** | 第 3–4 月 | LoRA 空间 TracInCP 离线筛选，消融实验 |
| **Phase 2 实现与验证** | 第 4–5 月 | 在线重要性采样，计算效率优化 |
| **Phase 3 实现与验证** | 第 5–6 月 | 多目标 TracIn，迭代重选管线 |
| **系统整合与大规模实验** | 第 7–8 月 | 三段管线串联，跨模型跨任务实验 |
| **论文撰写** | 第 8–10 月 | 撰写、修改、补充实验 |

---

## 相关工作

| 工作 | 会议 | 与本项目关联 |
|---|---|---|
| TracIn (Pruthi et al.) | NeurIPS 2020 | 核心影响力计算方法 |
| Importance Sampling (Katharopoulos et al.) | ICML 2018 | 在线梯度上界选样 |
| FLDebugger (Li et al.) | ICDE 2021 | 分层影响力分析，联邦兼容性（本组工作） |
| LESS (Xia et al.) | ICML 2024 | 最直接的 LoRA 选样基线 |
| RHO-LOSS (Mindermann et al.) | NeurIPS 2022 | 在线 holdout-loss 选样 |
| DoReMi (Xie et al.) | NeurIPS 2024 | Domain 级权重优化 |
| DSIR (Xie et al.) | NeurIPS 2023 | 基于分布匹配的重要性重采样 |
| D4 (Yu et al.) | ACL 2024 | 数据多样性驱动的 SFT 数据选择 |

---

## 备注

- 项目名称 **DataPrism** 为暂定名，可根据投稿方向调整副标题
- 若侧重联邦场景，可考虑加副标题如 "DataPrism-FL: Federated Data Selection for On-Device LLM Fine-Tuning"
- 项目代码仓库计划开源
