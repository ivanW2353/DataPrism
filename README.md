# DataPrism：面向 LLM 微调的梯度驱动数据选择框架

DataPrism（数据棱镜）将海量微调数据分解为不同维度的影响力信号——识别对模型有帮助的数据（proponent）、有害的数据（opponent）和冗余数据（neutral）——从而精准筛选出真正有价值的训练子集。

## 架构

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

## 安装

```bash
pip install -e ".[dev]"
```

## 快速开始

```bash
# 仅运行 Phase 1：TracInCP 离线数据筛选
python scripts/run_phase1_tracin.py --config configs/phase1_tracin.yaml

# 完整管线：串联所有三个 Phase
python scripts/run_pipeline.py --config configs/default.yaml
```

## 配置

所有超参数通过 `configs/` 目录下的 YAML 文件配置。使用 CLI 覆盖进行快速实验：

```bash
python scripts/run_pipeline.py --override phase1.num_epochs=3 phase2.initial_tau=0.5
```

## 基线方法

| 方法 | 配置文件 | 类型 |
|------|---------|------|
| 随机采样 (Uniform) | `configs/baseline/uniform.yaml` | 下界 |
| 全量数据 (Full Data) | `configs/baseline/full_data.yaml` | 上界 |
| LESS | `configs/baseline/less.yaml` | 表征相似度选样 |
| RHO-LOSS | `configs/baseline/rho_loss.yaml` | 在线 holdout-loss 选样 |
| DSIR | `configs/baseline/dsir.yaml` | 分布匹配重要性重采样 |
| **DataPrism (ours)** | — | 梯度驱动三段式 |

## 引用

- TracIn: Pruthi et al., NeurIPS 2020
- Importance Sampling: Katharopoulos & Fleuret, ICML 2018
- FLDebugger: Li et al., ICDE 2021
- LESS: Xia et al., ICML 2024
