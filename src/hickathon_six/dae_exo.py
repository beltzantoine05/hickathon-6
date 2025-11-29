from __future__ import annotations

import os
from typing import Optional, List

import torch
import glob

from hickathon_six.DAE.config import Config
from hickathon_six.DAE.data import load_data, train_val_split
from hickathon_six.DAE.training import train_dae, train_dae_full, export_encoder_embeddings
from hickathon_six.DAE.finetune import train_regressor, train_regressor_full
from hickathon_six.DAE.pipeline import embeddings_only_export, full_train_only_flow, kfold_flow


def _columns_from_todo() -> List[str]:
    # Exact list ported from DAE_todo/src/train_regressor.py COLUMNS_TO_LOAD
    return [
        'reading_q1_average_score', 'reading_q2_average_score', 'reading_q3_average_score',
        'reading_q4_average_score', 'reading_q5_average_score', 'reading_q6_average_score',
        'reading_q7_average_score', 'reading_q8_average_score', 'reading_q9_average_score',
        'reading_q10_average_score', 'reading_q11_average_score', 'reading_q12_average_score',
        'reading_q13_average_score', 'reading_q14_average_score', 'reading_q15_average_score',

        'science_q1_average_score', 'science_q2_average_score', 'science_q3_average_score',
        'science_q4_average_score', 'science_q5_average_score', 'science_q6_average_score',
        'science_q7_average_score', 'science_q8_average_score', 'science_q9_average_score',
        'science_q10_average_score', 'science_q11_average_score', 'science_q12_average_score',
        'science_q13_average_score', 'science_q14_average_score', 'science_q15_average_score',
        'science_q16_average_score', 'science_q17_average_score', 'science_q18_average_score',
        'science_q19_average_score',

        'reading_q1_total_timing', 'reading_q2_total_timing', 'reading_q3_total_timing',
        'reading_q4_total_timing', 'reading_q5_total_timing', 'reading_q6_total_timing',
        'reading_q7_total_timing', 'reading_q8_total_timing', 'reading_q9_total_timing',
        'reading_q10_total_timing', 'reading_q11_total_timing', 'reading_q12_total_timing',
        'reading_q13_total_timing', 'reading_q14_total_timing', 'reading_q15_total_timing',

        'science_q1_total_timing', 'science_q2_total_timing', 'science_q3_total_timing',
        'science_q4_total_timing', 'science_q5_total_timing', 'science_q6_total_timing',
        'science_q7_total_timing', 'science_q8_total_timing', 'science_q9_total_timing',
        'science_q10_total_timing', 'science_q11_total_timing', 'science_q12_total_timing',
        'science_q13_total_timing', 'science_q14_total_timing', 'science_q15_total_timing',
        'science_q16_total_timing', 'science_q17_total_timing', 'science_q18_total_timing',
        'science_q19_total_timing',
    ]


def run() -> None:
    """Run the full DAE + finetuning pipeline using cfg.

    Steps:
    - Load CSVs from cfg.data_dir with index from first column.
    - Split train/val.
    - Train DAE with early stopping on train/val.
    - Train DAE on full train set for fixed epochs.
    - Export X_test encoder embeddings (pre-finetune) to CSV.
    - Train regression head with early stopping on train/val.
    - Train regression head on the full train set.
    - Export X_test encoder embeddings again (post-finetune) to CSV.
    """
    cfg = Config(
        wandb_project="h6",
        wandb_tags=["dae", "finetune"],
        run_prefix="exo",
        group_prefix="default"
    )
    cfg.resolve_paths()

    # Hardcode explicit column behavior from original implementation
    columns = _columns_from_todo()
    timing_cols = [c for c in columns if c.endswith("_total_timing")]

    # Populate cfg columns only if user didn't set them
    if cfg.use_columns is None:
        cfg.use_columns = columns
    if not cfg.winsor_columns:
        cfg.winsor_columns = timing_cols
    if not cfg.log_columns:
        cfg.log_columns = timing_cols
    # Ensure preprocessing flags align with original behavior
    if cfg.winsor_q is None:
        cfg.winsor_q = 0.999
    cfg.winsor_upper_only = True if cfg.winsor_upper_only is None else cfg.winsor_upper_only
    cfg.timing_nonneg = True if cfg.timing_nonneg is None else cfg.timing_nonneg

    # Load data
    X_train_df, y_train, X_test_df = load_data(cfg)

    # Embeddings-only mode using a provided artifact or local path (delegates to DAE submodule)
    if cfg.embeddings_only:
        _ = embeddings_only_export(cfg, X_train_df, X_test_df)
        return

    # Full-train-only mode: no validation split, train on all data for fixed epochs
    if cfg.full_train_only:
        _ = full_train_only_flow(cfg, X_train_df, y_train, X_test_df)
        return

    # K-Fold CV (like original): run folds for monitoring before full training
    kfold_flow(cfg, X_train_df, y_train)

    # Default flow after folds: single split with ES, then full-train counterparts
    X_tr, y_tr, X_val, y_val = train_val_split(X_train_df, y_train, cfg)
    dae, prep, std_vec_t = train_dae(cfg, X_tr, X_val)
    dae_full = train_dae_full(cfg, dae, prep, std_vec_t, X_train_df)

    pre_csv = os.path.join(cfg.output_dir, "dae_embeddings_pre.csv")
    export_encoder_embeddings(cfg, dae_full, prep, X_test_df, pre_csv)

    if y_tr is not None and y_val is not None:
        head = train_regressor(cfg, dae_full, prep, X_tr, y_tr, X_val, y_val)
        _ = train_regressor_full(cfg, dae_full, prep, X_train_df, y_train, head)
    elif y_train is not None:
        _ = train_regressor_full(cfg, dae_full, prep, X_train_df, y_train)

    post_csv = os.path.join(cfg.output_dir, "dae_embeddings_post.csv")
    export_encoder_embeddings(cfg, dae_full, prep, X_test_df, post_csv)


if __name__ == "__main__":
    run()
