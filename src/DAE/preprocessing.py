"""Preprocessing utilities for the DAE pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .config import PreprocessingConfig


@dataclass
class PreprocessorState:
    """Container for fitted preprocessing objects."""

    caps: Dict[int, float]
    imputer: SimpleImputer
    scaler: StandardScaler


class Preprocessor:
    """Apply selection, winsorisation, log transform, imputation and scaling."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.state: PreprocessorState | None = None
        self._col_to_idx = {col: i for i, col in enumerate(self.config.columns)}

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        X = self._select(df)
        X_w = self._winsorize(X, fit=True)
        X_l = self._log_transform(X_w)
        imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
        imputer.fit(X_l)
        scaler = StandardScaler()
        scaler.fit(imputer.transform(X_l))
        self.state = PreprocessorState(caps=self._caps.copy(), imputer=imputer, scaler=scaler)
        return self

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        self.fit(df)
        return self.transform(df)

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if self.state is None:
            raise RuntimeError("Preprocessor must be fitted before calling transform.")
        X = self._select(df)
        X_w = self._winsorize(X, fit=False)
        X_l = self._log_transform(X_w)
        mask = (~np.isnan(X_l)).astype(np.float32)
        X_imp = self.state.imputer.transform(X_l)
        X_scaled = self.state.scaler.transform(X_imp).astype(np.float32)
        return X_scaled, mask

    def _select(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.config.columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in dataframe: {missing}")
        return df[self.config.columns].to_numpy(dtype=np.float32)

    def _winsorize(self, X: np.ndarray, fit: bool) -> np.ndarray:
        winsorized = X.copy()
        if fit:
            self._caps = {}
        if not self.config.winsorize_columns:
            return winsorized
        for col in self.config.winsorize_columns:
            idx = self._col_to_idx[col]
            col_data = winsorized[:, idx]
            if self.config.enforce_non_negative:
                col_data = np.where(np.isnan(col_data), col_data, np.maximum(col_data, 0.0))
            if fit:
                obs = col_data[~np.isnan(col_data)]
                obs = obs[np.isfinite(obs)]
                cap = float(np.quantile(obs, self.config.winsor_quantile)) if obs.size else np.nan
                self._caps[idx] = cap
            cap_value = self._caps.get(idx, np.nan)
            if not np.isnan(cap_value):
                col_data = np.where(np.isnan(col_data), col_data, np.minimum(col_data, cap_value))
            winsorized[:, idx] = col_data
        return winsorized

    def _log_transform(self, X: np.ndarray) -> np.ndarray:
        if not self.config.log_columns:
            return X
        transformed = X.copy()
        for col in self.config.log_columns:
            idx = self._col_to_idx[col]
            col_data = transformed[:, idx]
            transformed[:, idx] = np.log1p(col_data)
        return transformed

    @property
    def caps(self) -> Dict[int, float]:
        return {} if self.state is None else self.state.caps

    @property
    def imputer(self) -> SimpleImputer:
        if self.state is None:
            raise RuntimeError("Preprocessor has not been fitted yet.")
        return self.state.imputer

    @property
    def scaler(self) -> StandardScaler:
        if self.state is None:
            raise RuntimeError("Preprocessor has not been fitted yet.")
        return self.state.scaler
