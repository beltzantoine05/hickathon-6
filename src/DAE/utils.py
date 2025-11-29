"""Helper utilities for training."""
from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed: int = 42) -> None:
    """Seed python, numpy and torch RNGs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def masked_mse_loss(recons: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    squared_diff = (recons - target) ** 2
    masked_diff = squared_diff * mask
    return masked_diff.sum() / (mask.sum() + 1e-8)


def make_std_vector_from_observed(
    X_scaled: np.ndarray,
    mask: np.ndarray,
    quantile: float,
    clip_max: float,
    clip_floor: float,
) -> Tuple[np.ndarray, float, int]:
    std_vals = np.zeros((X_scaled.shape[1],), dtype=np.float32)
    for j in range(X_scaled.shape[1]):
        obs = X_scaled[mask[:, j] == 1.0, j]
        if obs.size >= 2:
            std_vals[j] = float(np.std(obs))
    std_pos = std_vals[std_vals > 1e-8]
    if std_pos.size == 0:
        cap = 0.0
    else:
        cap = float(np.quantile(std_pos, quantile))
        cap = min(cap, float(clip_max))
        cap = max(cap, float(clip_floor))
    std_vals = np.clip(std_vals, 0.0, cap).astype(np.float32)
    nonzero = int((std_vals > 0).sum())
    std_concat = np.concatenate([std_vals, np.zeros_like(std_vals)], axis=0).astype(np.float32)
    return std_concat, cap, nonzero


def build_loader(X_scaled: np.ndarray, mask: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    X_concat = np.hstack([X_scaled, mask]).astype(np.float32)
    ds = TensorDataset(
        torch.from_numpy(X_concat),
        torch.from_numpy(X_scaled.astype(np.float32)),
        torch.from_numpy(mask.astype(np.float32)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def build_regression_loader(X_concat: np.ndarray, targets: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X_concat), torch.from_numpy(targets.astype(np.float32)))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    preds = []
    ys = []
    for xb, yb in loader:
        xb = xb.to(device)
        pred = model(xb).detach().cpu().numpy()
        preds.append(pred)
        ys.append(yb.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(ys, axis=0)


@torch.no_grad()
def eval_regression(model, loader, criterion, device) -> dict:
    preds, ys = predict_all(model, loader, device)
    mse = float(np.mean((ys - preds) ** 2))
    r2 = r2_score_np(ys, preds)
    return {"mse": mse, "r2": r2}
