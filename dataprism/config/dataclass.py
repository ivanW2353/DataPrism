"""
Structured configuration dataclasses for DataPrism.

All hyperparameters flow through typed dataclasses — no dynamic dict access.
Configs are validated at construction time via Python's type system.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Pretrained model configuration."""

    name: str = "meta-llama/Meta-Llama-3-8B"
    trust_remote_code: bool = False
    torch_dtype: str = "bfloat16"  # float16, bfloat16, float32
    use_flash_attention_2: bool = True
    device_map: str = "auto"


@dataclass
class LoRAConfig:
    """LoRA adapter hyperparameters."""

    r: int = 64
    alpha: int = 128
    dropout: float = 0.1
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    # Additional modules to target (e.g., gate_proj for MLP)
    target_modules_extra: list[str] = field(default_factory=list)


@dataclass
class TrainingConfig:
    """General training hyperparameters."""

    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 2048
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    max_grad_norm: float = 1.0
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False


@dataclass
class DataConfig:
    """Dataset configuration."""

    name: str = "tatsu-lab/alpaca"
    local_path: Optional[str] = None  # Override: path to local dataset on disk
    split: str = "train"
    num_samples: Optional[int] = None  # None = use all
    validation_split: str = "validation"
    max_eval_samples: int = 500
    # Data preprocessing
    prompt_template: str = "alpaca"  # alpaca, sharegpt, chatml
    response_only_loss: bool = True  # Only compute loss on assistant response
    shuffle_seed: int = 42


@dataclass
class Phase1TracInConfig:
    """Phase 1: LoRA-space TracInCP offline data quality screening."""

    enabled: bool = True

    # Initial SFT training for checkpoint generation
    num_epochs: int = 2
    checkpoint_every_n_steps: int = 50
    max_checkpoints: int = 20
    sft_num_samples: Optional[int] = None  # Samples for SFT (None=all)
    tracin_num_samples: Optional[int] = None  # Samples for TracInCP (None=all, uses full dataset)
    grad_batch_size: int = 8  # Batch size for per-sample gradient computation
    force_retrain: bool = False  # If True, redo SFT even if checkpoints exist

    # Self-influence computation
    self_influence_method: str = "dot_product"  # dot_product, cosine
    normalize_gradients: bool = True  # L2-normalize grads before dot product

    # Outlier detection
    outlier_percentile: float = 95.0  # Flag samples above this percentile
    min_samples_after_filter: int = 5000

    # Redundancy removal
    redundancy_method: str = "kmeans"  # kmeans, agglomerative, none
    redundancy_similarity_threshold: float = 0.85
    redundancy_clusters: Optional[int] = None  # Auto-determined if None
    max_samples_after_redundancy: int = 50000
    target_fraction: float = 0.2  # Fraction of original data to retain (0.2 = 20%)

    # Storage
    checkpoint_dir: str = "outputs/checkpoints/phase1"
    influence_store_path: str = "outputs/influences/phase1_self_influence.h5"


@dataclass
class Phase2ImportanceConfig:
    """Phase 2: Online importance sampling based on gradient norm upper bounds."""

    enabled: bool = True

    # Candidate sampling
    candidate_multiplier: int = 4  # Sample 4 * batch_size candidates each step

    # Temperature schedule
    initial_tau: float = 1.0
    tau_annealing: str = "linear"  # linear, cosine, constant
    tau_min: float = 0.1
    tau_schedule_length: int = 5000  # Steps over which to anneal

    # Variance reduction tracking
    variance_ema_alpha: float = 0.95
    variance_reduction_threshold: float = 0.05
    recheck_interval: int = 100  # Steps between variance re-checks

    # Importance weighting
    importance_weight_clip: Optional[float] = 10.0  # Clip to avoid extreme weights


@dataclass
class Phase3MultiEvalConfig:
    """Phase 3: Validation-set-driven iterative data reselection."""

    enabled: bool = True

    # Multi-dimensional validation sets
    validation_sets: dict[str, dict] = field(
        default_factory=lambda: {
            "reasoning": {"source": "gsm8k", "split": "test", "num_samples": 500},
            "safety": {"source": "PKU-Alignment/PKU-SafeRLHF", "split": "test", "num_samples": 500},
            "chat": {"source": "lmsys/mt_bench", "split": "train", "num_samples": 500},
            "factual": {"source": "hotpot_qa", "split": "validation", "num_samples": 500},
        }
    )

    # Objective weights (sum to 1)
    lambda_weights: dict[str, float] = field(
        default_factory=lambda: {
            "reasoning": 0.30,
            "safety": 0.25,
            "chat": 0.25,
            "factual": 0.20,
        }
    )

    # Iterative reselection
    selection_fraction: float = 0.5  # Fraction to keep each round
    max_rounds: int = 5
    min_rounds: int = 2

    # Sampling weight update
    proponent_weight_boost: float = 2.0
    opponent_weight_decay: float = 0.5
    neutral_label_threshold: float = 0.05  # |score| < threshold → neutral

    # TracInVS computation
    tracin_vs_num_checkpoints: Optional[int] = None  # Use same as Phase 1 if None
    tracin_vs_checkpoint_dir: str = "outputs/checkpoints/phase3"


@dataclass
class EvaluationConfig:
    """Evaluation benchmark configuration."""

    benchmarks: list[str] = field(
        default_factory=lambda: ["mmlu", "gsm8k", "humaneval", "mt_bench", "alpaca_eval"]
    )
    mmlu_num_fewshot: int = 5
    gsm8k_num_fewshot: int = 8
    gsm8k_use_cot: bool = True
    humaneval_num_samples: int = 1  # pass@1
    mt_bench_num_turns: int = 2
    save_results: bool = True
    results_dir: str = "outputs/results"


@dataclass
class DataPrismConfig:
    """Top-level configuration aggregating all sub-configs.

    Usage:
        config = load_config(["configs/default.yaml", "configs/model/llama3_8b.yaml"])
        print(config.phase1.outlier_percentile)  # typed access
    """

    # Project metadata
    seed: int = 42
    device: str = "auto"  # auto, cuda, cpu
    project_name: str = "dataprism"
    experiment_name: str = "default"
    output_dir: str = "./outputs"

    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    phase1: Phase1TracInConfig = field(default_factory=Phase1TracInConfig)
    phase2: Phase2ImportanceConfig = field(default_factory=Phase2ImportanceConfig)
    phase3: Phase3MultiEvalConfig = field(default_factory=Phase3MultiEvalConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self):
        """Validate configuration consistency."""
        self._validate()

    def _validate(self):
        """Cross-validate inter-dependent config fields."""
        # Validate Phase 3 lambda weights sum to ~1
        if self.phase3.enabled:
            total = sum(self.phase3.lambda_weights.values())
            if abs(total - 1.0) > 0.01:
                raise ValueError(
                    f"Phase 3 lambda weights must sum to 1.0, got {total}"
                )

        # Validate tau schedule
        if self.phase2.enabled:
            valid = {"linear", "cosine", "constant"}
            if self.phase2.tau_annealing not in valid:
                raise ValueError(
                    f"tau_annealing must be one of {valid}, got {self.phase2.tau_annealing}"
                )
