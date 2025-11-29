from __future__ import annotations

import glob
import os
from typing import Optional, Tuple
from dataclasses import replace

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

from hickathon_six.DAE.config import Config
from hickathon_six.DAE.models import DAE
from hickathon_six.DAE.preprocess import Preprocessor
from hickathon_six.DAE.training import train_dae, train_dae_full, export_encoder_embeddings
from hickathon_six.DAE.finetune import train_regressor, train_regressor_full
from hickathon_six.DAE.logging_utils import wandb_run


def embeddings_only_export(cfg: Config, X_train_df: pd.DataFrame, X_test_df: pd.DataFrame) -> str:
    """Embeddings-only flow using a provided encoder artifact or local path.

    - Fit Preprocessor on full X_train_df
    - Load encoder weights from cfg.embeddings_model_path or cfg.embeddings_model_artifact
    - Export embeddings for X_test_df to outputs/dae_embeddings_from_artifact.csv

    Returns path to the generated CSV.
    """
    prep = Preprocessor(
        cfg.winsor_columns,
        cfg.log_columns,
        q=cfg.winsor_q,
        winsor_upper_only=cfg.winsor_upper_only,
        log_nonneg_clip=cfg.timing_nonneg,
    ).fit(X_train_df)

    n_features = len(prep.state.columns) if prep.state else X_train_df.shape[1]
    dae = DAE(
        n_features=n_features,
        encoder_hidden=list(cfg.encoder_layers),
        decoder_hidden=list(cfg.decoder_layers),
        latent_dim=cfg.latent_dim,
        alpha=cfg.noise_alpha,
        dropout=cfg.dropout,
    )

    weight_path: Optional[str] = cfg.embeddings_model_path
    if weight_path is None and cfg.embeddings_model_artifact:
        name = f"{cfg.run_prefix}-embeddings-only"
        group = f"{cfg.group_prefix}-emb"
        with wandb_run(cfg, name=name, group=group, job_type="embeddings-only") as run:
            art = run.use_artifact(cfg.embeddings_model_artifact)
            dl_dir = art.download()
            cand = glob.glob(os.path.join(dl_dir, "**", "*.pth"), recursive=True)
            if not cand:
                raise FileNotFoundError("No .pth file found in the downloaded artifact.")
            weight_path = cand[0]

    if not weight_path or not os.path.exists(weight_path):
        raise FileNotFoundError("Please provide cfg.embeddings_model_path or cfg.embeddings_model_artifact with a valid .pth file.")

    device = torch.device(cfg.device) if cfg.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(weight_path, map_location=device)
    dae.encoder.load_state_dict(state)

    out_csv = os.path.join(cfg.output_dir, "dae_embeddings_from_artifact.csv")
    export_encoder_embeddings(cfg, dae, prep, X_test_df, out_csv)
    return out_csv


def full_train_only_flow(
    cfg: Config,
    X_train_df: pd.DataFrame,
    y_train: Optional[pd.Series],
    X_test_df: pd.DataFrame,
) -> Tuple[str, str]:
    """Full-train-only flow (no validation split/early stopping).

    - Train DAE on full X_train_df
    - Export X_test embeddings (pre)
    - Train regressor on full (if y provided)
    - Export X_test embeddings (post)

    Returns tuple of (pre_csv_path, post_csv_path).
    """
    dae, prep, std_vec_t = train_dae(cfg, X_train_df, None)
    pre_csv = os.path.join(cfg.output_dir, "dae_embeddings_pre.csv")
    export_encoder_embeddings(cfg, dae, prep, X_test_df, pre_csv)

    if y_train is not None:
        _ = train_regressor_full(cfg, dae, prep, X_train_df, y_train)

    post_csv = os.path.join(cfg.output_dir, "dae_embeddings_post.csv")
    export_encoder_embeddings(cfg, dae, prep, X_test_df, post_csv)
    return pre_csv, post_csv


def kfold_flow(
    cfg: Config,
    X_train_df: pd.DataFrame,
    y_train: Optional[pd.Series],
) -> None:
    """Run K-Fold training for DAE and finetuning (if y provided) before full training.

    Mirrors the original implementation idea: for each fold, fit preprocessing on the
    training fold, train DAE with ES on that fold split, and if targets are available,
    run the two-phase finetune with ES on the same split. No artifacts are saved; this
    is for monitoring/validation and W&B logging only.
    """
    n_splits = max(2, int(cfg.n_splits))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)
    indices = np.arange(len(X_train_df))

    for fold, (tr_idx, va_idx) in enumerate(kf.split(indices)):
        X_tr = X_train_df.iloc[tr_idx]
        X_val = X_train_df.iloc[va_idx]
        if y_train is not None:
            y_tr = y_train.iloc[tr_idx]
            y_val = y_train.iloc[va_idx]
        else:
            y_tr = None
            y_val = None

        # Adjust group to include fold tag
        cfg_fold = replace(cfg, group_prefix=f"{cfg.group_prefix}-fold{fold}")

        # Train DAE with ES on this fold
        dae, prep, std_vec_t = train_dae(cfg_fold, X_tr, X_val)

        # Optional: Finetune on this fold if y provided
        if y_tr is not None and y_val is not None:
            _ = train_regressor(cfg_fold, dae, prep, X_tr, y_tr, X_val, y_val)
        # No full-train inside the fold loop
