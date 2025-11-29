from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


@dataclass
class PreprocessState:
    """Holds fitted state for preprocessing.

    This is kept in-memory only (not saved to disk), per requirements.
    """

    columns: list[str]
    winsor_columns: list[str]
    log_columns: list[str]
    winsor_bounds: dict[str, tuple[float, float]]
    imputer: SimpleImputer
    scaler: StandardScaler


class Preprocessor:
    """Configurable preprocessing pipeline.

    Steps (applied in order):
    - optional log1p on `log_columns`
    - optional winsorisation on `winsor_columns` at (q, 1-q)
    - simple imputation (mean)
    - standard scaling (StandardScaler)
    """

    def __init__(
        self,
        winsor_columns: Iterable[str] | None,
        log_columns: Iterable[str] | None,
        q: float = 0.01,
        winsor_upper_only: bool = False,
        log_nonneg_clip: bool = True,
    ):
        self.winsor_columns = list(winsor_columns or [])
        self.log_columns = list(log_columns or [])
        self.q = float(q)
        self.winsor_upper_only = bool(winsor_upper_only)
        self.log_nonneg_clip = bool(log_nonneg_clip)
        self.state: Optional[PreprocessState] = None

    def _apply_log(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.log_columns if c in X.columns]
        if cols:
            X = X.copy()
            if self.log_nonneg_clip:
                X[cols] = np.log1p(X[cols].clip(lower=0))
            else:
                X[cols] = np.log1p(X[cols])
        return X

    def _fit_winsor(self, X: pd.DataFrame) -> dict[str, tuple[float, float]]:
        bounds: dict[str, tuple[float, float]] = {}
        cols = [c for c in self.winsor_columns if c in X.columns]
        for c in cols:
            s = X[c].dropna()
            if len(s) == 0:
                bounds[c] = (np.nan, np.nan)
            else:
                if self.winsor_upper_only:
                    lo = np.nan
                    hi = float(np.quantile(s, self.q))  # interpret q as upper quantile
                else:
                    lo = float(np.quantile(s, self.q))
                    hi = float(np.quantile(s, 1 - self.q))
                bounds[c] = (lo, hi)
        return bounds

    def _apply_winsor(self, X: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
        cols = [c for c in self.winsor_columns if c in X.columns]
        if not cols:
            return X
        X = X.copy()
        for c in cols:
            lo, hi = bounds.get(c, (np.nan, np.nan))
            if not np.isnan(lo):
                X[c] = X[c].clip(lower=lo)
            if not np.isnan(hi):
                X[c] = X[c].clip(upper=hi)
        return X

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        # Determine order of columns to keep consistent
        cols = list(X.columns)
        # Apply deterministic transforms on a copy to compute stats
        X_t = self._apply_log(X)
        winsor_bounds = self._fit_winsor(X_t)
        X_t = self._apply_winsor(X_t, winsor_bounds)

        imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
        X_imp = imputer.fit_transform(X_t)

        scaler = StandardScaler()
        scaler.fit(X_imp)

        self.state = PreprocessState(
            columns=cols,
            winsor_columns=[c for c in self.winsor_columns if c in cols],
            log_columns=[c for c in self.log_columns if c in cols],
            winsor_bounds=winsor_bounds,
            imputer=imputer,
            scaler=scaler,
        )
        return self

    def transform(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Transform X into scaled numpy array and return mask of observed values before imputation.

        Returns
        -------
        X_scaled : np.ndarray of shape (n_samples, n_features)
        mask : np.ndarray of same shape, 1 for non-NaN in original X after log/winsor, 0 otherwise
        """
        if self.state is None:
            raise RuntimeError("Preprocessor must be fitted before transform().")
        X = X.copy()[self.state.columns]
        X = self._apply_log(X)
        X = self._apply_winsor(X, self.state.winsor_bounds)

        mask = (~X.isna()).astype(float).to_numpy(dtype=np.float32)
        X_imp = self.state.imputer.transform(X)
        X_scaled = self.state.scaler.transform(X_imp).astype(np.float32)
        return X_scaled, mask
