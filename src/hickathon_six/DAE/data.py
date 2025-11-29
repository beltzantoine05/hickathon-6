from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from hickathon_six.DAE.config import Config


def load_data(cfg: Config) -> tuple[pd.DataFrame, Optional[pd.Series], pd.DataFrame]:
    """Load X_train, y_train, X_test from cfg.data_dir with first column as index.

    Returns
    -------
    X_train_df : pd.DataFrame
        Training features (original scale), index preserved from CSV first column.
    y_train : Optional[pd.Series]
        Target values if file exists; else None.
    X_test_df : pd.DataFrame
        Test features (original scale), index preserved.
    """
    x_train_path = os.path.join(cfg.data_dir, cfg.x_train_file)
    y_train_path = os.path.join(cfg.data_dir, cfg.y_train_file)
    x_test_path = os.path.join(cfg.data_dir, cfg.x_test_file)

    X_train_df = pd.read_csv(x_train_path, index_col=0)
    X_test_df = pd.read_csv(x_test_path, index_col=0)
    y_train = pd.read_csv(y_train_path, index_col=0).iloc[:, 0] if os.path.exists(y_train_path) else None

    if cfg.use_columns is not None:
        cols = [c for c in cfg.use_columns if c in X_train_df.columns]
        X_train_df = X_train_df[cols]
        X_test_df = X_test_df[cols]

    return X_train_df, y_train, X_test_df


def train_val_split(
    X: pd.DataFrame,
    y: Optional[pd.Series],
    cfg: Config,
) -> tuple[pd.DataFrame, Optional[pd.Series], pd.DataFrame, Optional[pd.Series]]:
    """Split X (and y if provided) into train/val according to cfg.

    If y is None, a random split of rows is applied to X only.
    """
    if y is None:
        X_tr, X_val = train_test_split(
            X, test_size=cfg.val_size, shuffle=cfg.shuffle, random_state=cfg.random_state
        )
        return X_tr, None, X_val, None
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=cfg.val_size, shuffle=cfg.shuffle, random_state=cfg.random_state
    )
    return X_tr, y_tr, X_val, y_val
