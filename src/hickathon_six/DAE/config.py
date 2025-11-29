"""Configuration dataclasses for the DAE training pipeline."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class WandbConfig:
    """Weights & Biases logging configuration."""

    project: str = "ml_project_embedding"
    entity: Optional[str] = None
    group_prefix: str = "dae"
    run_prefix: str = "run"
    upload_artifacts: bool = True


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing and feature selection."""

    columns: List[str] = field(default_factory=list)
    winsorize_columns: List[str] = field(default_factory=list)
    log_columns: List[str] = field(default_factory=list)
    winsor_quantile: float = 0.999
    enforce_non_negative: bool = True
    max_missing_per_row: int | None = None


@dataclass
class ArchitectureConfig:
    """Configurable encoder/decoder architecture sizes."""

    encoder_layers: List[int] = field(default_factory=lambda: [256, 128, 64])
    decoder_layers: List[int] = field(default_factory=lambda: [128, 256])


@dataclass
class DAETrainingConfig:
    """Training hyperparameters for the denoising autoencoder."""

    seed: int = 42
    n_splits: int = 5
    batch_size: int = 1024
    epochs: int = 160
    final_epochs: int = 260
    lr: float = 1e-3
    weight_decay: float = 1e-5
    noise_alpha: float = 0.1
    dropout: float = 0.1
    grad_clip_norm: float = 1.0
    patience: int = 15
    min_epochs: int = 30
    min_delta: float = 1e-3
    std_clip_quantile: float = 0.90
    std_clip_max: float = 2.0
    std_clip_floor: float = 1e-3
    device: str = "cuda"


@dataclass
class RegressionTrainingConfig:
    """Training hyperparameters for the supervised finetuning."""

    batch_size: int = 1024
    lr_head_init: float = 1e-3
    epochs_phase_1: int = 5
    lr_encoder: float = 3e-5
    lr_head_finetune: float = 3e-4
    epochs_phase_2: int = 20
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 5
    early_stop_min_delta: float = 1e-3
    holdout_ratio: float = 0.1
    final_epochs: int = 20


@dataclass
class PipelineConfig:
    """Top-level configuration for the full pipeline."""

    data_dir: str = "data"
    seed: int = 42
    run_kfold: bool = True
    train_full_only: bool = False
    generate_embeddings_only: bool = False
    target_column: str = "MathScore"
    dae_group_name: str = "dae-pretraining"
    finetune_group_name: str = "finetune-regression"
    dae_run_name: str = "dae-pretrain"
    regression_run_name: str = "regression-finetune"
    embedding_prefix: str = "embedding"
    wandb: WandbConfig = field(default_factory=WandbConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    dae: DAETrainingConfig = field(default_factory=DAETrainingConfig)
    regression: RegressionTrainingConfig = field(default_factory=RegressionTrainingConfig)
    encoder_artifact_name: str = "dae-encoder-full"
    finetuned_encoder_artifact_name: str = "finetuned-encoder-full"
    fold_encoder_artifact_prefix: str = "dae-encoder-fold"
    load_fold_encoders_from_artifacts: bool = False
    use_pretrained_full_encoder: bool = False
    use_gpu_if_available: bool = True
    fold_indices: Optional[List[int]] = None

    def resolve_device(self) -> str:
        """Return the preferred device string based on availability."""

        if self.dae.device in {"cuda", "auto"} and self.use_gpu_if_available:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.dae.device
