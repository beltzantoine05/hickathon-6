from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from hickathon_six.DAE.config import Config
from hickathon_six.DAE.models import DAE
from hickathon_six.DAE.preprocess import Preprocessor
from hickathon_six.DAE.logging_utils import wandb_run, run_name, log_metrics


def masked_mse_loss(recons: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (recons - target) * mask
    denom = torch.clamp(mask.sum(), min=1.0)
    return (diff.pow(2).sum() / denom)


def compute_std_vector_from_observed(X_scaled: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Compute per-feature std using only observed entries (mask==1).

    Returns
    -------
    stds : np.ndarray
        Feature-wise std vector (zeros/NaNs replaced by 1.0 to avoid degenerate noise scale).
    std_clip_cap : int
        Number of features whose std was clipped/replaced (i.e., non-finite or zero).
    std_nonzero : int
        Number of features with strictly positive observed std before clipping.
    """
    n_features = X_scaled.shape[1]
    stds = np.zeros(n_features, dtype=np.float32)
    clipped = 0
    nonzero = 0
    for j in range(n_features):
        obs = X_scaled[mask[:, j] > 0.5, j]
        s = np.std(obs) if obs.size > 1 else 0.0
        if np.isfinite(s) and s > 0:
            nonzero += 1
            stds[j] = float(s)
        else:
            clipped += 1
            stds[j] = 1.0
    return stds, clipped, nonzero


def _build_loader(X_scaled: np.ndarray, mask: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    values = torch.from_numpy(X_scaled.astype(np.float32))
    mask_t = torch.from_numpy(mask.astype(np.float32))
    concat = torch.cat([values, mask_t], dim=1)
    ds = TensorDataset(concat, values, mask_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _device(cfg: Config) -> torch.device:
    if cfg.device is not None:
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_dae(
    cfg: Config,
    X_train_df: pd.DataFrame,
    X_val_df: Optional[pd.DataFrame],
) -> tuple[DAE, Preprocessor, torch.Tensor]:
    """Train DAE with early stopping on validation set (if provided).

    Returns the trained DAE model, fitted Preprocessor, and std_vector tensor used for noise.
    """
    cfg.resolve_paths()
    device = _device(cfg)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Fit preprocessing on training only
    prep = Preprocessor(
        cfg.winsor_columns,
        cfg.log_columns,
        q=cfg.winsor_q,
        winsor_upper_only=cfg.winsor_upper_only,
        log_nonneg_clip=cfg.timing_nonneg,
    ).fit(X_train_df)
    Xtr_scaled, Mtr = prep.transform(X_train_df)
    if X_val_df is not None:
        Xval_scaled, Mval = prep.transform(X_val_df)
    else:
        Xval_scaled, Mval = None, None

    std_vec, std_clip_cap, std_nonzero = compute_std_vector_from_observed(Xtr_scaled, Mtr)
    std_vec_t = torch.from_numpy(std_vec).to(device)

    n_features = Xtr_scaled.shape[1]
    model = DAE(
        n_features=n_features,
        encoder_hidden=list(cfg.encoder_layers),
        decoder_hidden=list(cfg.decoder_layers),
        latent_dim=cfg.latent_dim,
        alpha=cfg.noise_alpha,
        dropout=cfg.dropout,
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=cfg.dae_lr, weight_decay=cfg.dae_weight_decay)

    train_loader = _build_loader(Xtr_scaled, Mtr, cfg.dae_batch_size, shuffle=True)
    val_loader = _build_loader(Xval_scaled, Mval, cfg.dae_batch_size, shuffle=False) if Xval_scaled is not None else None

    best_val = float("inf")
    best_state: Optional[dict] = None
    patience_counter = 0

    group = f"{cfg.group_prefix}-dae"
    name = run_name(cfg.run_prefix, "dae")
    with wandb_run(cfg, name=name, group=group, job_type="dae-train") as run:
        for epoch in range(cfg.dae_epochs):
            model.train()
            total = 0.0
            count = 0
            for batch_in, batch_target, batch_mask in train_loader:
                batch_in = batch_in.to(device)
                batch_target = batch_target.to(device)
                batch_mask = batch_mask.to(device)
                opt.zero_grad()
                out = model(batch_in, std_vector=std_vec_t, apply_noise=True)
                loss = masked_mse_loss(out, batch_target, batch_mask)
                loss.backward()
                opt.step()
                bs = batch_in.size(0)
                total += loss.item() * bs
                count += bs
            train_loss_noisy = total / max(1, count)

            metrics = {"epoch": epoch, "dae/train_noisy": train_loss_noisy, "dae/std_clip_cap": float(std_clip_cap), "dae/std_nonzero": float(std_nonzero)}

            if val_loader is not None:
                model.eval()
                vtotal_noisy = 0.0
                vtotal_clean = 0.0
                vcount = 0
                with torch.no_grad():
                    for batch_in, batch_target, batch_mask in val_loader:
                        batch_in = batch_in.to(device)
                        batch_target = batch_target.to(device)
                        batch_mask = batch_mask.to(device)
                        # Evaluate both with noise and without noise (clean)
                        out_noisy = model(batch_in, std_vector=std_vec_t, apply_noise=True)
                        out_clean = model(batch_in, std_vector=std_vec_t, apply_noise=False)
                        vloss_noisy = masked_mse_loss(out_noisy, batch_target, batch_mask)
                        vloss_clean = masked_mse_loss(out_clean, batch_target, batch_mask)
                        bs = batch_in.size(0)
                        vtotal_noisy += vloss_noisy.item() * bs
                        vtotal_clean += vloss_clean.item() * bs
                        vcount += bs
                val_noisy = vtotal_noisy / max(1, vcount)
                val_clean = vtotal_clean / max(1, vcount)
                metrics["dae/val_noisy"] = val_noisy
                metrics["dae/val_clean"] = val_clean

                # Console log every epoch
                print(f"[DAE][ES] Epoch {epoch+1:03d} | train_noisy {train_loss_noisy:.6f} | val_noisy {val_noisy:.6f} | val_clean {val_clean:.6f}")

                # Early stopping
                improved = (best_val - val_clean) > cfg.dae_min_delta
                if improved:
                    best_val = val_clean
                    patience_counter = 0
                    best_state = model.state_dict()
                else:
                    if epoch + 1 >= cfg.dae_min_epochs:
                        patience_counter += 1
                        if patience_counter >= cfg.dae_patience:
                            log_metrics(run, metrics, step=epoch)
                            break

            log_metrics(run, metrics, step=epoch)
            if val_loader is None:
                # Console log when no val
                print(f"[DAE][ES] Epoch {epoch+1:03d} | train_noisy {train_loss_noisy:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, prep, std_vec_t


def train_dae_full(cfg: Config, model: DAE, prep: Preprocessor, std_vec_t: torch.Tensor, X_full_df: pd.DataFrame) -> DAE:
    """Train DAE on full training data for fixed epochs (no early stopping)."""
    device = _device(cfg)
    X_scaled, M = prep.transform(X_full_df)
    loader = _build_loader(X_scaled, M, cfg.dae_batch_size, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=cfg.dae_lr, weight_decay=cfg.dae_weight_decay)
    name = run_name(cfg.run_prefix, "dae-full")
    group = f"{cfg.group_prefix}-dae"
    with wandb_run(cfg, name=name, group=group, job_type="dae-full") as run:
        for epoch in range(cfg.dae_epochs):
            model.train()
            total = 0.0
            count = 0
            for batch_in, batch_target, batch_mask in loader:
                batch_in = batch_in.to(device)
                batch_target = batch_target.to(device)
                batch_mask = batch_mask.to(device)
                opt.zero_grad()
                out = model(batch_in, std_vector=std_vec_t, apply_noise=True)
                loss = masked_mse_loss(out, batch_target, batch_mask)
                loss.backward()
                opt.step()
                bs = batch_in.size(0)
                total += loss.item() * bs
                count += bs
            train_loss_noisy = total / max(1, count)
            log_metrics(run, {"epoch": epoch, "dae/train_noisy": train_loss_noisy, "dae/std_clip_cap": 0.0, "dae/std_nonzero": float(X_scaled.shape[1])}, step=epoch)
            # Console log every epoch
            print(f"[DAE][FULL] Epoch {epoch+1:03d} | train_noisy {train_loss_noisy:.6f}")
    return model


def export_encoder_embeddings(
    cfg: Config,
    model: DAE,
    prep: Preprocessor,
    X_df: pd.DataFrame,
    csv_path: str,
) -> None:
    """Export encoder embeddings for X_df to csv_path, preserving index order."""
    device = _device(cfg)
    X_scaled, M = prep.transform(X_df)
    values = torch.from_numpy(X_scaled.astype(np.float32)).to(device)
    mask = torch.from_numpy(M.astype(np.float32)).to(device)
    model.eval()
    with torch.no_grad():
        feats = model.encode(values, mask)  # [n, latent]
    feats_np = feats.detach().cpu().numpy()
    out_df = pd.DataFrame(feats_np, index=X_df.index)
    out_df.to_csv(csv_path, index=True)
