from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from hickathon_six.DAE.config import Config
from hickathon_six.DAE.models import DAE
from hickathon_six.DAE.preprocess import Preprocessor
from hickathon_six.DAE.training import _device
from hickathon_six.DAE.logging_utils import wandb_run, run_name, log_metrics


def _build_reg_loader(X_scaled: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    x_t = torch.from_numpy(X_scaled.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.float32)).view(-1, 1)
    ds = TensorDataset(x_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _build_masked_loader(X_scaled: np.ndarray, M: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    values = torch.from_numpy(X_scaled.astype(np.float32))
    mask = torch.from_numpy(M.astype(np.float32))
    x_in = torch.cat([values, mask], dim=1)
    y_t = torch.from_numpy(y.astype(np.float32)).view(-1, 1)
    ds = TensorDataset(x_in, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_regressor(
    cfg: Config,
    dae: DAE,
    prep: Preprocessor,
    X_train_df: pd.DataFrame,
    y_train: pd.Series,
    X_val_df: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
) -> nn.Module:
    """Train a regression head on top of the DAE encoder with early stopping.

    Returns trained SupervisedRegressor.
    """
    device = _device(cfg)
    # Prepare features: pass scaled inputs through encoder
    Xtr_scaled, Mtr = prep.transform(X_train_df)
    Xtr = torch.from_numpy(Xtr_scaled.astype(np.float32)).to(device)
    Mtr_t = torch.from_numpy(Mtr.astype(np.float32)).to(device)
    dae = dae.to(device).eval()
    with torch.no_grad():
        Htr = dae.encode(Xtr, Mtr_t).detach().cpu().numpy()

    if X_val_df is not None and y_val is not None:
        Xval_scaled, Mval = prep.transform(X_val_df)
        Xval = torch.from_numpy(Xval_scaled.astype(np.float32)).to(device)
        Mval_t = torch.from_numpy(Mval.astype(np.float32)).to(device)
        with torch.no_grad():
            Hval = dae.encode(Xval, Mval_t).detach().cpu().numpy()
    else:
        Hval = None

    # Build head on latent features
    latent_dim = Htr.shape[1]
    head = nn.Linear(latent_dim, 1).to(device)
    criterion = nn.MSELoss()

    # Phase 1: head-only training on precomputed latent features
    def run_phase1() -> nn.Module:
        opt = optim.Adam(head.parameters(), lr=cfg.ft_lr_head_phase1 if cfg.ft_two_phase else cfg.ft_lr, weight_decay=cfg.ft_weight_decay)
        train_loader = _build_reg_loader(Htr, y_train.to_numpy(), cfg.ft_batch_size, shuffle=True)
        val_loader = _build_reg_loader(Hval, y_val.to_numpy(), cfg.ft_batch_size, shuffle=False) if Hval is not None else None

        best_val = float("inf")
        best_state: Optional[dict] = None
        patience_counter = 0

        group = f"{cfg.group_prefix}-finetune"
        run_label = "finetune2" if cfg.ft_two_phase else "finetune"
        name = run_name(cfg.run_prefix, run_label)
        with wandb_run(cfg, name=name, group=group, job_type="finetune") as run:
            epochs = cfg.ft_epochs_phase1 if cfg.ft_two_phase else cfg.ft_epochs
            for epoch in range(epochs):
                head.train()
                total = 0.0
                count = 0
                for bx, by in train_loader:
                    bx = bx.to(device)
                    by = by.to(device)
                    opt.zero_grad()
                    pred = head(bx)
                    loss = criterion(pred, by)
                    loss.backward()
                    opt.step()
                    bs = bx.size(0)
                    total += loss.item() * bs
                    count += bs
                train_loss = total / max(1, count)
                metrics = {"epoch": epoch, ("finetune/phase1/train_mse" if cfg.ft_two_phase else "finetune/train_mse"): train_loss}

                if val_loader is not None:
                    head.eval()
                    vtotal = 0.0
                    vcount = 0
                    with torch.no_grad():
                        for bx, by in val_loader:
                            bx = bx.to(device)
                            by = by.to(device)
                            pred = head(bx)
                            loss = criterion(pred, by)
                            bs = bx.size(0)
                            vtotal += loss.item() * bs
                            vcount += bs
                    val_loss = vtotal / max(1, vcount)
                    key = "finetune/phase1/val_mse" if cfg.ft_two_phase else "finetune/val_mse"
                    metrics[key] = val_loss

                    # Console log every epoch
                    print(f"[FT][P1][ES] Epoch {epoch+1:03d} | train {train_loss:.6f} | val {val_loss:.6f}")

                    improved = (best_val - val_loss) > cfg.ft_min_delta
                    if improved:
                        best_val = val_loss
                        patience_counter = 0
                        best_state = head.state_dict()
                    else:
                        if epoch + 1 >= cfg.ft_min_epochs:
                            patience_counter += 1
                            if patience_counter >= cfg.ft_patience:
                                log_metrics(run, metrics, step=epoch)
                                break

                log_metrics(run, metrics, step=epoch)
                if val_loader is None:
                    print(f"[FT][P1][ES] Epoch {epoch+1:03d} | train {train_loss:.6f}")

        if best_state is not None:
            head.load_state_dict(best_state)
        return head

    head = run_phase1()

    # If no two-phase requested, return the head trained on latent features
    if not cfg.ft_two_phase:
        return head

    # Phase 2: unfreeze encoder (except first layer) and train jointly using concatenated inputs
    # Build loaders on masked inputs
    Xtr_scaled, Mtr = prep.transform(X_train_df)
    train_loader2 = _build_masked_loader(Xtr_scaled, Mtr, y_train.to_numpy(), cfg.ft_batch_size, shuffle=True)
    if X_val_df is not None and y_val is not None:
        Xval_scaled, Mval = prep.transform(X_val_df)
        val_loader2 = _build_masked_loader(Xval_scaled, Mval, y_val.to_numpy(), cfg.ft_batch_size, shuffle=False)
    else:
        val_loader2 = None

    # Freeze first layer of encoder if requested
    encoder: nn.Module = dae.encoder
    # Identify first Linear layer in encoder.net
    first_linear: Optional[nn.Linear] = None
    if hasattr(encoder, "net") and isinstance(encoder.net, nn.Sequential):
        for mod in encoder.net:
            if isinstance(mod, nn.Linear):
                first_linear = mod
                break
    if cfg.ft_unfreeze_except_first and first_linear is not None:
        for p in first_linear.parameters():
            p.requires_grad = False

    # Ensure remaining encoder params are trainable
    for mod in encoder.parameters():
        pass  # no-op to keep style

    # Optimizer with two param groups
    enc_params = [p for p in encoder.parameters() if p.requires_grad]
    head_params = list(head.parameters())
    opt2 = optim.Adam([
        {"params": enc_params, "lr": cfg.ft_lr_encoder_phase2, "weight_decay": cfg.ft_weight_decay_phase2},
        {"params": head_params, "lr": cfg.ft_lr_head_phase2, "weight_decay": cfg.ft_weight_decay_phase2},
    ])

    best_val2 = float("inf")
    best_state2: Optional[dict] = None
    patience_counter2 = 0

    group = f"{cfg.group_prefix}-finetune"
    name = run_name(cfg.run_prefix, "finetune2-phase2")
    with wandb_run(cfg, name=name, group=group, job_type="finetune-phase2") as run:
        for epoch in range(cfg.ft_epochs_phase2):
            dae.train()
            head.train()
            total = 0.0
            count = 0
            for bx, by in train_loader2:
                bx = bx.to(device)
                by = by.to(device)
                opt2.zero_grad()
                z = encoder(bx)
                pred = head(z)
                loss = criterion(pred, by)
                loss.backward()
                # grad clip
                if cfg.ft_grad_clip_norm is not None and cfg.ft_grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), cfg.ft_grad_clip_norm)
                opt2.step()
                bs = bx.size(0)
                total += loss.item() * bs
                count += bs
            train_loss = total / max(1, count)
            metrics = {"epoch": epoch, "finetune/phase2/train_mse": train_loss}

            if val_loader2 is not None:
                dae.eval()
                head.eval()
                vtotal = 0.0
                vcount = 0
                with torch.no_grad():
                    for bx, by in val_loader2:
                        bx = bx.to(device)
                        by = by.to(device)
                        z = encoder(bx)
                        pred = head(z)
                        loss = criterion(pred, by)
                        bs = bx.size(0)
                        vtotal += loss.item() * bs
                        vcount += bs
                val_loss = vtotal / max(1, vcount)
                metrics["finetune/phase2/val_mse"] = val_loss

                # Console log every epoch
                print(f"[FT][P2][ES] Epoch {epoch+1:03d} | train {train_loss:.6f} | val {val_loss:.6f}")

                improved = (best_val2 - val_loss) > cfg.ft_min_delta
                if improved:
                    best_val2 = val_loss
                    patience_counter2 = 0
                    # save both encoder and head
                    best_state2 = {
                        "encoder": encoder.state_dict(),
                        "head": head.state_dict(),
                    }
                else:
                    if epoch + 1 >= cfg.ft_min_epochs:
                        patience_counter2 += 1
                        if patience_counter2 >= cfg.ft_patience:
                            log_metrics(run, metrics, step=epoch)
                            break

            log_metrics(run, metrics, step=epoch)
            if val_loader2 is None:
                print(f"[FT][P2][ES] Epoch {epoch+1:03d} | train {train_loss:.6f}")

    if best_state2 is not None:
        encoder.load_state_dict(best_state2["encoder"])  # type: ignore[arg-type]
        head.load_state_dict(best_state2["head"])  # type: ignore[arg-type]

    # Return the trained head; encoder is part of dae
    return head


def train_regressor_full(
    cfg: Config,
    dae: DAE,
    prep: Preprocessor,
    X_full_df: pd.DataFrame,
    y_full: pd.Series,
    head: Optional[nn.Module] = None,
) -> nn.Module:
    """Train the regression head on the whole training set (no early stopping)."""
    device = _device(cfg)
    X_scaled, M = prep.transform(X_full_df)
    X_t = torch.from_numpy(X_scaled.astype(np.float32)).to(device)
    M_t = torch.from_numpy(M.astype(np.float32)).to(device)
    dae = dae.to(device).eval()
    # Compute latent features for phase 1
    with torch.no_grad():
        H = dae.encode(X_t, M_t).detach().cpu().numpy()

    # Init head if needed
    if head is None:
        head = nn.Linear(H.shape[1], 1).to(device)
    criterion = nn.MSELoss()

    # Phase 1 (head-only)
    opt1 = optim.Adam(head.parameters(), lr=cfg.ft_lr_head_phase1 if cfg.ft_two_phase else cfg.ft_lr, weight_decay=cfg.ft_weight_decay)
    loader1 = _build_reg_loader(H, y_full.to_numpy(), cfg.ft_batch_size, shuffle=True)
    name = run_name(cfg.run_prefix, "finetune-full-phase1" if cfg.ft_two_phase else "finetune-full")
    group = f"{cfg.group_prefix}-finetune"
    with wandb_run(cfg, name=name, group=group, job_type="finetune-full-phase1" if cfg.ft_two_phase else "finetune-full") as run:
        epochs1 = cfg.ft_epochs_phase1 if cfg.ft_two_phase else cfg.ft_epochs
        for epoch in range(epochs1):
            head.train()
            total = 0.0
            count = 0
            for bx, by in loader1:
                bx = bx.to(device)
                by = by.to(device)
                opt1.zero_grad()
                pred = head(bx)
                loss = criterion(pred, by)
                loss.backward()
                opt1.step()
                bs = bx.size(0)
                total += loss.item() * bs
                count += bs
            train_loss = total / max(1, count)
            key = "finetune/phase1/train_mse" if cfg.ft_two_phase else "finetune/train_mse"
            log_metrics(run, {"epoch": epoch, key: train_loss}, step=epoch)
            print(f"[FT][P1][FULL] Epoch {epoch+1:03d} | train {train_loss:.6f}")

    if not cfg.ft_two_phase:
        return head

    # Phase 2 (encoder unfreezed except first layer)
    # Prepare masked input loader on full data
    X_scaled, M = prep.transform(X_full_df)
    loader2 = _build_masked_loader(X_scaled, M, y_full.to_numpy(), cfg.ft_batch_size, shuffle=True)

    encoder: nn.Module = dae.encoder
    # Freeze first linear layer
    first_linear: Optional[nn.Linear] = None
    if hasattr(encoder, "net") and isinstance(encoder.net, nn.Sequential):
        for mod in encoder.net:
            if isinstance(mod, nn.Linear):
                first_linear = mod
                break
    if cfg.ft_unfreeze_except_first and first_linear is not None:
        for p in first_linear.parameters():
            p.requires_grad = False

    enc_params = [p for p in encoder.parameters() if p.requires_grad]
    head_params = list(head.parameters())
    opt2 = optim.Adam([
        {"params": enc_params, "lr": cfg.ft_lr_encoder_phase2, "weight_decay": cfg.ft_weight_decay_phase2},
        {"params": head_params, "lr": cfg.ft_lr_head_phase2, "weight_decay": cfg.ft_weight_decay_phase2},
    ])

    name = run_name(cfg.run_prefix, "finetune-full-phase2")
    with wandb_run(cfg, name=name, group=group, job_type="finetune-full-phase2") as run:
        for epoch in range(cfg.ft_epochs_phase2):
            dae.train(); head.train()
            total = 0.0
            count = 0
            for bx, by in loader2:
                bx = bx.to(device)
                by = by.to(device)
                opt2.zero_grad()
                z = encoder(bx)
                pred = head(z)
                loss = criterion(pred, by)
                loss.backward()
                if cfg.ft_grad_clip_norm is not None and cfg.ft_grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), cfg.ft_grad_clip_norm)
                opt2.step()
                bs = bx.size(0)
                total += loss.item() * bs
                count += bs
            train_loss = total / max(1, count)
            log_metrics(run, {"epoch": epoch, "finetune/phase2/train_mse": train_loss}, step=epoch)
            print(f"[FT][P2][FULL] Epoch {epoch+1:03d} | train {train_loss:.6f}")

    return head
