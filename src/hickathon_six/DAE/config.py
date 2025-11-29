from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class Config:
    """Global configuration for DAE training and fine-tuning.

    All parameters are plain dataclass fields (no pydantic) and can be overridden
    at runtime. This configuration controls data loading, preprocessing, model
    architecture, optimization, logging, and outputs.
    """

    # Data
    project_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir: str = field(default_factory=lambda: os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "data"))
    x_train_file: str = "X_train.csv"
    y_train_file: str = "y_train.csv"
    x_test_file: str = "X_test.csv"

    # Preprocessing controls
    use_columns: Optional[List[str]] = None  # if None, use all from X_train
    winsor_columns: List[str] = field(default_factory=list)  # columns to winsorise (upper cap by default)
    winsor_q: float = 0.999  # upper quantile for winsorisation (original impl used ~0.999)
    log_columns: List[str] = field(default_factory=list)  # columns to log1p (defaults to timing columns)
    timing_nonneg: bool = True  # clamp timing columns to >= 0 before log
    winsor_upper_only: bool = True  # apply only an upper cap for winsorisation (like original)

    # Data split
    val_size: float = 0.2
    shuffle: bool = True
    random_state: int = 42

    # Model architecture
    latent_dim: int = 64
    encoder_layers: List[int] = field(default_factory=lambda: [256, 128])
    decoder_layers: List[int] = field(default_factory=lambda: [128, 256])
    dropout: float = 0.1
    noise_alpha: float = 0.1

    # Optimization - DAE
    dae_batch_size: int = 256
    dae_epochs: int = 100
    dae_patience: int = 10
    dae_min_epochs: int = 10
    dae_min_delta: float = 1e-4
    dae_lr: float = 1e-3
    dae_weight_decay: float = 1e-5

    # Optimization - Finetune (regression head)
    ft_batch_size: int = 256
    # One-phase settings (used in phase 1 for head-only, or if ft_two_phase=False)
    ft_epochs: int = 200
    ft_patience: int = 20  # Early stopping
    ft_min_epochs: int = 5
    ft_min_delta: float = 1e-4
    ft_lr: float = 1e-3
    ft_weight_decay: float = 1e-6

    # Two-phase fine-tuning (Phase 1: head-only; Phase 2: unfreeze encoder except first layer)
    ft_two_phase: bool = True
    ft_epochs_phase1: int = 5
    ft_epochs_phase2: int = 20
    ft_lr_head_phase1: float = 1e-3
    ft_lr_head_phase2: float = 3e-4
    ft_lr_encoder_phase2: float = 3e-5
    ft_weight_decay_phase2: float = 1e-4
    ft_grad_clip_norm: float = 1.0
    ft_unfreeze_except_first: bool = True

    # Logging (wandb)
    wandb_project: str = "h6-dae"
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=lambda: ["dae", "finetune"])
    wandb_mode: Optional[str] = None  # e.g. "offline" to disable network
    run_prefix: str = "dae"
    group_prefix: str = "default"

    # General
    seed: int = 42
    device: Optional[str] = None  # 'cuda', 'cpu' or None to auto-select

    # Outputs
    output_dir: str = field(default_factory=lambda: os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "outputs"))

    # Special run modes
    full_train_only: bool = False  # if True, no train/val split; train only on full dataset
    embeddings_only: bool = False  # if True, skip training and only export embeddings using a provided model artifact/path
    embeddings_model_artifact: Optional[str] = None  # W&B artifact name to download encoder weights from
    embeddings_model_path: Optional[str] = None  # Local path to encoder weights (.pth)

    def resolve_paths(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
