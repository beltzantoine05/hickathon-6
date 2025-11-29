import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import wandb

from infrastructure import ArtifactManager
from models import Encoder

COLUMNS_TO_LOAD = [
    'reading_q1_average_score', 'reading_q2_average_score', 'reading_q3_average_score',
    'reading_q4_average_score', 'reading_q5_average_score', 'reading_q6_average_score',
    'reading_q7_average_score', 'reading_q8_average_score', 'reading_q9_average_score',
    'reading_q10_average_score', 'reading_q11_average_score', 'reading_q12_average_score',
    'reading_q13_average_score', 'reading_q14_average_score', 'reading_q15_average_score',

    'math_q1_average_score', 'math_q2_average_score', 'math_q3_average_score',
    'math_q4_average_score', 'math_q5_average_score', 'math_q6_average_score',
    'math_q7_average_score', 'math_q8_average_score', 'math_q9_average_score',
    'math_q10_average_score', 'math_q11_average_score', 'math_q12_average_score',
    'math_q13_average_score', 'math_q14_average_score', 'math_q15_average_score',
    'math_q16_average_score', 'math_q17_average_score', 'math_q18_average_score',
    'math_q19_average_score', 'math_q20_average_score', 'math_q21_average_score',

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

    'math_q1_total_timing', 'math_q2_total_timing', 'math_q3_total_timing',
    'math_q4_total_timing', 'math_q5_total_timing', 'math_q6_total_timing',
    'math_q7_total_timing', 'math_q8_total_timing', 'math_q9_total_timing',
    'math_q10_total_timing', 'math_q11_total_timing', 'math_q12_total_timing',
    'math_q13_total_timing', 'math_q14_total_timing', 'math_q15_total_timing',
    'math_q16_total_timing', 'math_q17_total_timing', 'math_q18_total_timing',
    'math_q19_total_timing', 'math_q20_total_timing', 'math_q21_total_timing',

    'science_q1_total_timing', 'science_q2_total_timing', 'science_q3_total_timing',
    'science_q4_total_timing', 'science_q5_total_timing', 'science_q6_total_timing',
    'science_q7_total_timing', 'science_q8_total_timing', 'science_q9_total_timing',
    'science_q10_total_timing', 'science_q11_total_timing', 'science_q12_total_timing',
    'science_q13_total_timing', 'science_q14_total_timing', 'science_q15_total_timing',
    'science_q16_total_timing', 'science_q17_total_timing', 'science_q18_total_timing',
    'science_q19_total_timing'
]

N_FEATS = 110

CONFIG = {
    "project_name": "ml_project_embedding",
    "group_name": "Supervised_Finetuning_Masked",

    "seed": 42,
    "n_splits": 5,
    "batch_size": 1024,

    "input_dim": 220,
    "dropout": 0.1,

    "lr_head_init": 1e-3,
    "epochs_phase_1": 5,

    "lr_encoder": 3e-5,
    "lr_head_finetune": 3e-4,
    "epochs_phase_2": 20,
    "weight_decay": 1e-4,

    "grad_clip_norm": 1.0,

    "timing_winsor_q": 0.999,
    "timing_nonneg": True,

    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


class MaskedSupervisedRegressor(nn.Module):
    def __init__(self, encoder: Encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = x[:, :N_FEATS]
        mask = x[:, N_FEATS:]
        x_in = torch.cat([values * mask, mask], dim=1)
        z = self.encoder(x_in)
        return self.head(z).squeeze(-1)


def timing_indices() -> list[int]:
    return [i for i, c in enumerate(COLUMNS_TO_LOAD) if c.endswith("_total_timing")]


def fit_timing_caps(X_raw: np.ndarray, idx: list[int], q: float) -> dict[int, float]:
    caps: dict[int, float] = {}
    for j in idx:
        col = X_raw[:, j]
        obs = col[~np.isnan(col)]
        obs = obs[np.isfinite(obs)]
        caps[j] = float(np.quantile(obs, q)) if obs.size else np.nan
    return caps


def apply_timing_transform(X_raw: np.ndarray, idx: list[int], caps: dict[int, float]) -> np.ndarray:
    X = X_raw.copy()
    for j in idx:
        col = X[:, j]
        if CONFIG["timing_nonneg"]:
            col = np.where(np.isnan(col), col, np.maximum(col, 0.0))
        cap = caps.get(j, np.nan)
        if not np.isnan(cap):
            col = np.where(np.isnan(col), col, np.minimum(col, cap))
        X[:, j] = np.log1p(col).astype(np.float32)
    return X


def load_data_and_targets():
    x_path = os.path.join("datas", "X_train.csv")
    y_path = os.path.join("datas", "y_train.csv")
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError("Fichiers introuvables dans datas/")

    df_x = pd.read_csv(x_path, index_col=0)
    df_y = pd.read_csv(y_path, index_col=0)

    df = df_x.join(df_y[["MathScore"]], how="inner")

    missing_cols = [c for c in COLUMNS_TO_LOAD if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans X : {missing_cols}")

    X = df[COLUMNS_TO_LOAD].to_numpy(dtype=np.float32)
    y = df["MathScore"].to_numpy(dtype=np.float32)

    n_missing = np.isnan(X).sum(axis=1)
    keep = n_missing < 100
    return X[keep], y[keep]


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    running = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG["grad_clip_norm"])
        optimizer.step()

        running += loss.item() * xb.size(0)
    return running / len(loader.dataset)


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


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


@torch.no_grad()
def eval_metrics(model, loader, criterion, device) -> dict:
    preds, ys = predict_all(model, loader, device)
    mse = float(np.mean((ys - preds) ** 2))
    r2 = r2_score_np(ys, preds)
    # mse_torch juste pour cohérence si tu veux
    return {"mse": mse, "r2": r2}


def main():
    mgr = ArtifactManager()
    X_raw_all, y_all = load_data_and_targets()
    device = torch.device(CONFIG["device"])

    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    kfold = KFold(n_splits=CONFIG["n_splits"], shuffle=True, random_state=CONFIG["seed"])

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_raw_all)):
        run = wandb.init(
            project=CONFIG["project_name"],
            group=CONFIG["group_name"],
            name=f"regressor_masked_fold_{fold}",
            job_type="finetuning",
            config=CONFIG,
        )

        X_train_raw = X_raw_all[train_idx]
        X_val_raw = X_raw_all[val_idx]
        y_train = y_all[train_idx]
        y_val = y_all[val_idx]

        # Preprocess fit sur train
        tidx = timing_indices()
        caps = fit_timing_caps(X_train_raw, tidx, q=float(CONFIG["timing_winsor_q"]))
        X_train_t = apply_timing_transform(X_train_raw, tidx, caps)
        X_val_t = apply_timing_transform(X_val_raw, tidx, caps)

        imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
        X_train_imp = imputer.fit_transform(X_train_t)
        X_val_imp = imputer.transform(X_val_t)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp).astype(np.float32)
        X_val_scaled = scaler.transform(X_val_imp).astype(np.float32)

        mask_train = (~np.isnan(X_train_t)).astype(np.float32)
        mask_val = (~np.isnan(X_val_t)).astype(np.float32)

        X_train = np.hstack([X_train_scaled, mask_train]).astype(np.float32)
        X_val = np.hstack([X_val_scaled, mask_val]).astype(np.float32)

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
            batch_size=CONFIG["batch_size"],
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
            batch_size=CONFIG["batch_size"],
            shuffle=False,
        )

        # Encoder pré-entrainé
        encoder_path = f"dl_encoder_fold_{fold}.pth"
        mgr.download_artifact(f"dae-encoder-fold-{fold}:latest", encoder_path, run)

        encoder = Encoder(input_dim=CONFIG["input_dim"], dropout=CONFIG["dropout"]).to(device)
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))

        model = MaskedSupervisedRegressor(encoder).to(device)
        criterion = nn.MSELoss()

        # Phase 1
        for p in model.encoder.parameters():
            p.requires_grad = False
        opt1 = optim.Adam(model.head.parameters(), lr=CONFIG["lr_head_init"])

        for epoch in range(int(CONFIG["epochs_phase_1"])):
            train_one_epoch(model, train_loader, criterion, opt1, device)

            train_metrics = eval_metrics(model, train_loader, criterion, device)
            val_metrics = eval_metrics(model, val_loader, criterion, device)

            run.log({
                "phase": 1,
                "epoch": epoch,
                "train_r2": train_metrics["r2"],
                "val_r2": val_metrics["r2"],
                "train_mse": train_metrics["mse"],
                "val_mse": val_metrics["mse"],
            })

            print(
                f"P1 Epoch {epoch+1} | Train R2 {train_metrics['r2']:.4f} | "
                f"Val R2 {val_metrics['r2']:.4f}"
            )

        # Phase 2
        for p in model.encoder.parameters():
            p.requires_grad = True
        for p in model.encoder.net[0].parameters():
            p.requires_grad = False

        opt2 = optim.AdamW(
            [
                {"params": model.encoder.net[3].parameters(), "lr": CONFIG["lr_encoder"]},
                {"params": model.encoder.net[6].parameters(), "lr": CONFIG["lr_encoder"]},
                {"params": model.head.parameters(), "lr": CONFIG["lr_head_finetune"]},
            ],
            weight_decay=CONFIG["weight_decay"],
        )

        best_val_r2 = -1e18
        best_path = f"temp_best_reg_{fold}.pth"

        for epoch in range(int(CONFIG["epochs_phase_2"])):
            train_one_epoch(model, train_loader, criterion, opt2, device)

            train_metrics = eval_metrics(model, train_loader, criterion, device)
            val_metrics = eval_metrics(model, val_loader, criterion, device)

            run.log({
                "phase": 2,
                "epoch": epoch,
                "train_r2": train_metrics["r2"],
                "val_r2": val_metrics["r2"],
                "train_mse": train_metrics["mse"],
                "val_mse": val_metrics["mse"],
            })

            print(
                f"P2 Epoch {epoch+1} | Train R2 {train_metrics['r2']:.4f} | "
                f"Val R2 {val_metrics['r2']:.4f}"
            )

            if val_metrics["r2"] > best_val_r2:
                best_val_r2 = val_metrics["r2"]
                torch.save(model.state_dict(), best_path)

        model.load_state_dict(torch.load(best_path, map_location=device))

        reg_path = f"regressor_masked_fold_{fold}.pth"
        emb_path = f"embedding_masked_fold_{fold}.pth"
        prep_path = f"preprocess_fold_{fold}.pkl"

        torch.save(model.state_dict(), reg_path)
        torch.save(model.encoder.state_dict(), emb_path)
        joblib.dump({"caps": caps, "imputer": imputer, "scaler": scaler}, prep_path)

        mgr.log_model(reg_path, f"supervised-regressor-masked-fold-{fold}", run)
        mgr.log_model(emb_path, f"final-embedding-masked-fold-{fold}", run, aliases=["production"])
        mgr.log_model(prep_path, f"preprocess-masked-fold-{fold}", run, metadata={"type": "preprocess"})

        run.finish()

        for f in [encoder_path, best_path, reg_path, emb_path, prep_path]:
            if os.path.exists(f):
                os.remove(f)


if __name__ == "__main__":
    main()
