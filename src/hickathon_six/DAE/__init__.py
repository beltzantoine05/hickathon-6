"""Denosing AutoEncoder (DAE) training package.

This package provides:
- Configuration via a dataclass (`config.Config`).
- Data loading from the project `data/` directory with index preserved.
- Preprocessing with configurable column selection, winsorisation and log transforms.
- PyTorch models for a configurable DAE and a regression head for fine-tuning.
- Training loops with standardized Weights & Biases (wandb) logging.
- Utility to export encoder embeddings for X_test as a CSV with the original index.

No external artifact storage is used (no MinIO). Only wandb logging is supported.
"""

from hickathon_six.DAE.config import Config  # re-export for convenience

__all__ = ["Config"]
