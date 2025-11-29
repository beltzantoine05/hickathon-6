"""Pipeline orchestration for DAE pretraining and finetuning."""
from __future__ import annotations

import copy
import os
from dataclasses import replace
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.model_selection import KFold, train_test_split

from .config import PipelineConfig
from .models import DAE, FlexibleEncoder, MaskedSupervisedRegressor
from .preprocessing import Preprocessor
from .utils import (
    build_loader,
    build_regression_loader,
    eval_regression,
    make_std_vector_from_observed,
    masked_mse_loss,
    seed_everything,
)


class PipelineRunner:
    """Train the denoising autoencoder and finetuning head end-to-end."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = torch.device(config.resolve_device())
        seed_everything(config.seed)

    def load_features(self, filename: str) -> pd.DataFrame:
        print(f"Loading {filename}...")
        path = os.path.join(self.config.data_dir, filename)
        return pd.read_csv(path, index_col=0)

    def load_targets(self) -> pd.Series:
        path = os.path.join(self.config.data_dir, "y_train.csv")
        return pd.read_csv(path, index_col=0)[self.config.target_column]

    def run(self) -> None:
        X_train_df = self.load_features("X_train.csv")
        X_test_df = self.load_features("X_test.csv")
        y_train = self.load_targets()

        X_train_df, y_train = self._filter_rows_with_missing(X_train_df, y_train)

        if self.config.generate_embeddings_only:
            self.generate_embeddings(X_train_df, X_test_df)
            return

        if not self.config.train_full_only and self.config.run_kfold:
            self.train_dae_kfold(X_train_df)
            self.train_regressor_kfold(X_train_df, y_train)

        encoder_state, preprocessor = self.train_dae_full(X_train_df)
        finetuned_state, finetune_preproc = self.train_regressor_full(X_train_df, y_train, encoder_state)
        self.generate_embeddings(X_train_df, X_test_df, encoder_state, finetuned_state, preprocessor)

    def _filter_rows_with_missing(
        self, X_df: pd.DataFrame, y: pd.Series | None = None
    ) -> Tuple[pd.DataFrame, pd.Series | None]:
        threshold = self.config.preprocessing.max_missing_per_row
        columns = self.config.preprocessing.columns
        if threshold is None or not columns:
            return X_df, y
        missing_counts = X_df[columns].isna().sum(axis=1)
        keep_mask = missing_counts < threshold
        removed = int((~keep_mask).sum())
        if removed > 0:
            print(f"[Preprocess] Dropped {removed} rows with >= {threshold} missing values")
        X_filtered = X_df.loc[keep_mask]
        y_filtered = y.loc[keep_mask] if y is not None else None
        return X_filtered, y_filtered

    def train_dae_kfold(self, X_df: pd.DataFrame) -> List[FlexibleEncoder]:
        cfg = self.config
        encoders: List[FlexibleEncoder] = []
        kfold = KFold(n_splits=cfg.dae.n_splits, shuffle=True, random_state=cfg.seed)
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_df)):
            run = self._start_run(
                group=cfg.dae_group_name,
                name=f"{cfg.wandb.run_prefix}-dae-fold-{fold}",
                job_type="dae-pretrain",
                config=cfg,
            )
            preprocessor = Preprocessor(cfg.preprocessing)
            X_train_scaled, mask_train = preprocessor.fit_transform(X_df.iloc[train_idx])
            X_val_scaled, mask_val = preprocessor.transform(X_df.iloc[val_idx])
            encoder_state = self._train_single_dae(
                run,
                X_train_scaled,
                mask_train,
                X_val_scaled,
                mask_val,
                cfg,
            )
            encoder = self._build_encoder(cfg)
            encoder.load_state_dict(encoder_state)
            encoders.append(encoder)
            run.finish()
        return encoders

    def _train_single_dae(
        self,
        run: wandb.wandb_sdk.wandb_run.Run,
        X_train_scaled: np.ndarray,
        mask_train: np.ndarray,
        X_val_scaled: np.ndarray,
        mask_val: np.ndarray,
        cfg: PipelineConfig,
    ) -> dict:
        dae_cfg = cfg.dae
        std_concat, cap, nonzero = make_std_vector_from_observed(
            X_train_scaled,
            mask_train,
            quantile=dae_cfg.std_clip_quantile,
            clip_max=dae_cfg.std_clip_max,
            clip_floor=dae_cfg.std_clip_floor,
        )
        std_vector_tensor = torch.from_numpy(std_concat).to(self.device)
        train_loader = build_loader(X_train_scaled, mask_train, dae_cfg.batch_size, shuffle=True)
        val_loader = build_loader(X_val_scaled, mask_val, dae_cfg.batch_size, shuffle=False)

        model = DAE(
            input_dim=2 * X_train_scaled.shape[1],
            output_dim=X_train_scaled.shape[1],
            encoder_hidden=cfg.architecture.encoder_layers,
            decoder_hidden=cfg.architecture.decoder_layers,
            alpha=dae_cfg.noise_alpha,
            dropout=dae_cfg.dropout,
        ).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=dae_cfg.lr, weight_decay=dae_cfg.weight_decay)

        best_val_clean = float("inf")
        best_encoder = None
        patience_counter = 0

        for epoch in range(dae_cfg.epochs):
            train_loss_online = self._train_dae_epoch(model, train_loader, optimizer, std_vector_tensor, dae_cfg.grad_clip_norm)
            val_noisy, val_clean = self._eval_dae(model, val_loader, std_vector_tensor)
            print(
                f"[DAE][Epoch {epoch+1:03d}] train_noisy={train_loss_online:.4f} "
                f"val_noisy={val_noisy:.4f} val_clean={val_clean:.4f}"
            )
            run.log(
                {
                    "dae/epoch": epoch,
                    "dae/train_loss": train_loss_online,
                    "dae/val_loss_noisy": val_noisy,
                    "dae/val_loss_clean": val_clean,
                    "dae/std_clip_cap": cap,
                    "dae/std_nonzero": nonzero,
                }
            )
            improved = (best_val_clean - val_clean) > dae_cfg.min_delta
            if improved:
                best_val_clean = val_clean
                patience_counter = 0
                best_encoder = copy.deepcopy(model.encoder.state_dict())
            else:
                if epoch + 1 >= dae_cfg.min_epochs:
                    patience_counter += 1
                    if patience_counter >= dae_cfg.patience:
                        break
        return best_encoder if best_encoder is not None else model.encoder.state_dict()

    def _train_dae_epoch(self, model, loader, optimizer, std_vector_tensor, clip_norm: float) -> float:
        model.train()
        running_loss = 0.0
        for batch_input, batch_target, batch_mask in loader:
            batch_input = batch_input.to(self.device)
            batch_target = batch_target.to(self.device)
            batch_mask = batch_mask.to(self.device)
            optimizer.zero_grad(set_to_none=True)
            recons = model(batch_input, std_vector=std_vector_tensor, apply_noise=True)
            loss = masked_mse_loss(recons, batch_target, batch_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
            running_loss += loss.item() * batch_input.size(0)
        return running_loss / len(loader.dataset)

    @torch.no_grad()
    def _eval_dae(self, model, loader, std_vector_tensor) -> Tuple[float, float]:
        model.eval()
        loss_noisy = 0.0
        loss_clean = 0.0
        for batch_input, batch_target, batch_mask in loader:
            batch_input = batch_input.to(self.device)
            batch_target = batch_target.to(self.device)
            batch_mask = batch_mask.to(self.device)
            recons_noisy = model(batch_input, std_vector=std_vector_tensor, apply_noise=True)
            recons_clean = model(batch_input, std_vector=None, apply_noise=False)
            ln = masked_mse_loss(recons_noisy, batch_target, batch_mask)
            lc = masked_mse_loss(recons_clean, batch_target, batch_mask)
            bs = batch_input.size(0)
            loss_noisy += ln.item() * bs
            loss_clean += lc.item() * bs
        n = len(loader.dataset)
        return loss_noisy / n, loss_clean / n

    def _build_encoder(self, cfg: PipelineConfig) -> FlexibleEncoder:
        return FlexibleEncoder(
            input_dim=2 * len(cfg.preprocessing.columns),
            hidden_sizes=cfg.architecture.encoder_layers,
            dropout=cfg.dae.dropout,
        )

    def _start_run(self, group: str, name: str, job_type: str, config: PipelineConfig):
        wandb_config = {
            "dae": vars(config.dae),
            "regression": vars(config.regression),
            "preprocessing": vars(config.preprocessing),
            "architecture": vars(config.architecture),
        }
        return wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            group=group,
            name=name,
            job_type=job_type,
            config=wandb_config,
        )

    def train_dae_full(self, X_df: pd.DataFrame) -> Tuple[dict, Preprocessor]:
        cfg = self.config
        run = self._start_run(cfg.dae_group_name, f"{cfg.wandb.run_prefix}-dae-full", "dae-full", cfg)
        preprocessor = Preprocessor(cfg.preprocessing)
        X_scaled, mask = preprocessor.fit_transform(X_df)
        dae_cfg = replace(cfg.dae, epochs=cfg.dae.final_epochs)
        encoder_state = self._train_single_dae(run, X_scaled, mask, X_scaled, mask, replace(cfg, dae=dae_cfg))
        if cfg.wandb.upload_artifacts:
            encoder_path = "dae_encoder_full.pth"
            torch.save(encoder_state, encoder_path)
            artifact = wandb.Artifact(cfg.encoder_artifact_name, type="model")
            artifact.add_file(encoder_path)
            run.log_artifact(artifact, aliases=["latest"])
            os.remove(encoder_path)
        run.finish()
        return encoder_state, preprocessor

    def train_regressor_kfold(self, X_df: pd.DataFrame, y: pd.Series) -> None:
        cfg = self.config
        kfold = KFold(n_splits=cfg.dae.n_splits, shuffle=True, random_state=cfg.seed)
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_df)):
            run = self._start_run(
                group=cfg.finetune_group_name,
                name=f"{cfg.wandb.run_prefix}-reg-fold-{fold}",
                job_type="finetune",
                config=cfg,
            )
            preprocessor = Preprocessor(cfg.preprocessing)
            X_train_scaled, mask_train = preprocessor.fit_transform(X_df.iloc[train_idx])
            X_val_scaled, mask_val = preprocessor.transform(X_df.iloc[val_idx])
            encoder_state, _ = self.train_dae_full(X_df.iloc[train_idx])
            encoder = self._build_encoder(cfg).to(self.device)
            encoder.load_state_dict(encoder_state)
            model = MaskedSupervisedRegressor(encoder, latent_dim=cfg.architecture.encoder_layers[-1]).to(self.device)
            self._train_regressor(run, model, preprocessor, X_train_scaled, mask_train, y.iloc[train_idx].to_numpy(), X_val_scaled, mask_val, y.iloc[val_idx].to_numpy())
            run.finish()

    def _train_regressor(
        self,
        run,
        model: MaskedSupervisedRegressor,
        preprocessor: Preprocessor,
        X_train_scaled: np.ndarray,
        mask_train: np.ndarray,
        y_train: np.ndarray,
        X_val_scaled: np.ndarray,
        mask_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        reg_cfg = self.config.regression
        train_concat = np.hstack([X_train_scaled, mask_train]).astype(np.float32)
        val_concat = np.hstack([X_val_scaled, mask_val]).astype(np.float32)
        train_loader = build_regression_loader(train_concat, y_train, reg_cfg.batch_size, shuffle=True)
        val_loader = build_regression_loader(val_concat, y_val, reg_cfg.batch_size, shuffle=False)
        criterion = nn.MSELoss()
        # phase 1 freeze encoder
        for p in model.encoder.parameters():
            p.requires_grad = False
        opt1 = optim.Adam(model.head.parameters(), lr=reg_cfg.lr_head_init)
        for epoch in range(reg_cfg.epochs_phase_1):
            self._train_reg_epoch(model, train_loader, criterion, opt1, reg_cfg.grad_clip_norm)
            train_metrics = eval_regression(model, train_loader, criterion, self.device)
            val_metrics = eval_regression(model, val_loader, criterion, self.device)
            print(
                f"[Finetune P1][Epoch {epoch+1:03d}] train_mse={train_metrics['mse']:.4f} "
                f"val_mse={val_metrics['mse']:.4f} train_r2={train_metrics['r2']:.4f} "
                f"val_r2={val_metrics['r2']:.4f}"
            )
            run.log(
                {
                    "finetune/phase": 1,
                    "finetune/epoch": epoch,
                    "finetune/train_r2": train_metrics["r2"],
                    "finetune/val_r2": val_metrics["r2"],
                    "finetune/train_mse": train_metrics["mse"],
                    "finetune/val_mse": val_metrics["mse"],
                }
            )
        # phase 2 unfreeze except first layer
        for p in model.encoder.parameters():
            p.requires_grad = True
        first_linear = next(m for m in model.encoder.net if isinstance(m, nn.Linear))
        for p in first_linear.parameters():
            p.requires_grad = False
        opt2 = optim.AdamW(
            [
                {"params": [p for n, p in model.encoder.named_parameters() if p.requires_grad], "lr": reg_cfg.lr_encoder},
                {"params": model.head.parameters(), "lr": reg_cfg.lr_head_finetune},
            ],
            weight_decay=reg_cfg.weight_decay,
        )
        best_state = None
        best_val = float("-inf")
        patience_counter = 0
        for epoch in range(reg_cfg.epochs_phase_2):
            self._train_reg_epoch(model, train_loader, criterion, opt2, reg_cfg.grad_clip_norm)
            train_metrics = eval_regression(model, train_loader, criterion, self.device)
            val_metrics = eval_regression(model, val_loader, criterion, self.device)
            print(
                f"[Finetune P2][Epoch {epoch+1:03d}] train_mse={train_metrics['mse']:.4f} "
                f"val_mse={val_metrics['mse']:.4f} train_r2={train_metrics['r2']:.4f} "
                f"val_r2={val_metrics['r2']:.4f}"
            )
            run.log(
                {
                    "finetune/phase": 2,
                    "finetune/epoch": reg_cfg.epochs_phase_1 + epoch,
                    "finetune/train_r2": train_metrics["r2"],
                    "finetune/val_r2": val_metrics["r2"],
                    "finetune/train_mse": train_metrics["mse"],
                    "finetune/val_mse": val_metrics["mse"],
                }
            )
            improved = (val_metrics["r2"] - best_val) > reg_cfg.early_stop_min_delta
            if improved:
                best_val = val_metrics["r2"]
                patience_counter = 0
                best_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= reg_cfg.early_stop_patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)

    def _train_reg_epoch(self, model, loader, criterion, optimizer, clip_norm) -> float:
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
            running += loss.item() * xb.size(0)
        return running / len(loader.dataset)

    def train_regressor_full(
        self,
        X_df: pd.DataFrame,
        y: pd.Series,
        encoder_state: dict,
    ) -> Tuple[dict, Preprocessor]:
        cfg = self.config
        run = self._start_run(cfg.finetune_group_name, f"{cfg.wandb.run_prefix}-reg-full", "finetune-full", cfg)
        preprocessor = Preprocessor(cfg.preprocessing)
        X_scaled, mask = preprocessor.fit_transform(X_df)
        train_concat = np.hstack([X_scaled, mask]).astype(np.float32)
        train_x, val_x, train_y, val_y = train_test_split(train_concat, y.to_numpy(), test_size=cfg.regression.holdout_ratio, random_state=cfg.seed)
        model = MaskedSupervisedRegressor(self._build_encoder(cfg).to(self.device), latent_dim=cfg.architecture.encoder_layers[-1]).to(self.device)
        model.encoder.load_state_dict(encoder_state)
        self._train_regressor(
            run,
            model,
            preprocessor,
            train_x[:, : X_scaled.shape[1]],
            train_x[:, X_scaled.shape[1] :],
            train_y,
            val_x[:, : X_scaled.shape[1]],
            val_x[:, X_scaled.shape[1] :],
            val_y,
        )
        # final fit on all data for best epoch count
        full_loader = build_regression_loader(train_concat, y.to_numpy(), cfg.regression.batch_size, shuffle=True)
        criterion = nn.MSELoss()
        for p in model.encoder.parameters():
            p.requires_grad = True
        first_linear = next(m for m in model.encoder.net if isinstance(m, nn.Linear))
        for p in first_linear.parameters():
            p.requires_grad = False
        opt_final = optim.AdamW(
            [
                {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": cfg.regression.lr_encoder},
                {"params": model.head.parameters(), "lr": cfg.regression.lr_head_finetune},
            ],
            weight_decay=cfg.regression.weight_decay,
        )
        for _ in range(cfg.regression.final_epochs):
            self._train_reg_epoch(model, full_loader, criterion, opt_final, cfg.regression.grad_clip_norm)
        if cfg.wandb.upload_artifacts:
            finetuned_path = "finetuned_encoder_full.pth"
            torch.save(model.encoder.state_dict(), finetuned_path)
            artifact = wandb.Artifact(cfg.finetuned_encoder_artifact_name, type="model")
            artifact.add_file(finetuned_path)
            run.log_artifact(artifact, aliases=["latest"])
            os.remove(finetuned_path)
        run.finish()
        return model.encoder.state_dict(), preprocessor

    def generate_embeddings(
        self,
        X_train_df: pd.DataFrame,
        X_test_df: pd.DataFrame,
        encoder_state: dict | None = None,
        finetuned_state: dict | None = None,
        preprocessor: Preprocessor | None = None,
    ) -> None:
        cfg = self.config
        if encoder_state is None:
            encoder_state = self._download_encoder(cfg.encoder_artifact_name)
        if finetuned_state is None:
            finetuned_state = self._download_encoder(cfg.finetuned_encoder_artifact_name)
        if preprocessor is None:
            preprocessor = Preprocessor(cfg.preprocessing).fit(X_train_df)
        X_test_scaled, mask_test = preprocessor.transform(X_test_df)
        concat_test = np.hstack([X_test_scaled, mask_test]).astype(np.float32)
        encoder = self._build_encoder(cfg).to(self.device)
        encoder.load_state_dict(encoder_state)
        encoder.eval()
        with torch.no_grad():
            embeddings = encoder(torch.from_numpy(concat_test).to(self.device)).cpu().numpy()
        emb_df = pd.DataFrame(embeddings, index=X_test_df.index)
        emb_path = os.path.join(cfg.data_dir, f"{cfg.embedding_prefix}_dae_full.csv")
        emb_df.to_csv(emb_path)

        # finetuned embeddings
        finetuned_encoder = self._build_encoder(cfg).to(self.device)
        finetuned_encoder.load_state_dict(finetuned_state)
        finetuned_encoder.eval()
        with torch.no_grad():
            ft_embeddings = finetuned_encoder(torch.from_numpy(concat_test).to(self.device)).cpu().numpy()
        ft_path = os.path.join(cfg.data_dir, f"{cfg.embedding_prefix}_finetuned_full.csv")
        pd.DataFrame(ft_embeddings, index=X_test_df.index).to_csv(ft_path)

    def _download_encoder(self, artifact_name: str) -> dict:
        cfg = self.config
        api = wandb.Api()
        entity = cfg.wandb.entity
        if entity is None:
            raise ValueError("wandb.entity must be set to download artifacts")
        artifact = api.artifact(f"{entity}/{cfg.wandb.project}/{artifact_name}:latest", type="model")
        download_dir = artifact.download()
        pth_files = [f for f in os.listdir(download_dir) if f.endswith(".pth")]
        if not pth_files:
            raise FileNotFoundError("No .pth file in downloaded artifact")
        path = os.path.join(download_dir, pth_files[0])
        return torch.load(path, map_location=self.device)
