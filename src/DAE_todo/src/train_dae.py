import os
import copy
import joblib
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import wandb

from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from infrastructure import ArtifactManager
from models import DAE

COLUMNS_TO_LOAD = [
    # Reading average scores
    'reading_q1_average_score', 'reading_q2_average_score', 'reading_q3_average_score',
    'reading_q4_average_score', 'reading_q5_average_score', 'reading_q6_average_score',
    'reading_q7_average_score', 'reading_q8_average_score', 'reading_q9_average_score',
    'reading_q10_average_score', 'reading_q11_average_score', 'reading_q12_average_score',
    'reading_q13_average_score', 'reading_q14_average_score', 'reading_q15_average_score',
    # Math average scores
    'math_q1_average_score', 'math_q2_average_score', 'math_q3_average_score',
    'math_q4_average_score', 'math_q5_average_score', 'math_q6_average_score',
    'math_q7_average_score', 'math_q8_average_score', 'math_q9_average_score',
    'math_q10_average_score', 'math_q11_average_score', 'math_q12_average_score',
    'math_q13_average_score', 'math_q14_average_score', 'math_q15_average_score',
    'math_q16_average_score', 'math_q17_average_score', 'math_q18_average_score',
    'math_q19_average_score', 'math_q20_average_score', 'math_q21_average_score',
    # Science average scores
    'science_q1_average_score', 'science_q2_average_score', 'science_q3_average_score',
    'science_q4_average_score', 'science_q5_average_score', 'science_q6_average_score',
    'science_q7_average_score', 'science_q8_average_score', 'science_q9_average_score',
    'science_q10_average_score', 'science_q11_average_score', 'science_q12_average_score',
    'science_q13_average_score', 'science_q14_average_score', 'science_q15_average_score',
    'science_q16_average_score', 'science_q17_average_score', 'science_q18_average_score',
    'science_q19_average_score',
    # Reading total timing
    'reading_q1_total_timing', 'reading_q2_total_timing', 'reading_q3_total_timing',
    'reading_q4_total_timing', 'reading_q5_total_timing', 'reading_q6_total_timing',
    'reading_q7_total_timing', 'reading_q8_total_timing', 'reading_q9_total_timing',
    'reading_q10_total_timing', 'reading_q11_total_timing', 'reading_q12_total_timing',
    'reading_q13_total_timing', 'reading_q14_total_timing', 'reading_q15_total_timing',
    # Math total timing
    'math_q1_total_timing', 'math_q2_total_timing', 'math_q3_total_timing',
    'math_q4_total_timing', 'math_q5_total_timing', 'math_q6_total_timing',
    'math_q7_total_timing', 'math_q8_total_timing', 'math_q9_total_timing',
    'math_q10_total_timing', 'math_q11_total_timing', 'math_q12_total_timing',
    'math_q13_total_timing', 'math_q14_total_timing', 'math_q15_total_timing',
    'math_q16_total_timing', 'math_q17_total_timing', 'math_q18_total_timing',
    'math_q19_total_timing', 'math_q20_total_timing', 'math_q21_total_timing',
    # Science total timing
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
    "group_name": "DAE_Pretraining_Masked",
    "seed": 42,
    "n_splits": 5,

    "batch_size": 1024,
    "epochs": 160,
    "lr": 1e-3,
    "weight_decay": 1e-5,

    "noise_alpha": 0.1,
    "dropout": 0.1,
    "grad_clip_norm": 1.0,

    # early stop folds
    "patience": 15,
    "min_epochs": 30,
    "min_delta": 1e-3,

    # std clipping (pour bruit)
    "std_clip_quantile": 0.90,
    "std_clip_max": 2.0,
    "std_clip_floor": 1e-3,

    # timing preprocessing
    "timing_winsor_q": 0.999,
    "timing_nonneg": True,

    # final fit full data
    "final_epochs": 260,

    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def load_data_with_nans() -> np.ndarray:
    file_path = os.path.join("data", "X_train.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    print(f"[Data] Chargement de {file_path}...")
    df = pd.read_csv(file_path)

    missing_cols = [c for c in COLUMNS_TO_LOAD if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes : {missing_cols}")

    df = df[COLUMNS_TO_LOAD]
    X = df.to_numpy(dtype=np.float32)

    initial_len = len(X)
    n_missing = np.isnan(X).sum(axis=1)
    mask_keep = n_missing < 100
    X_clean = X[mask_keep]

    removed_count = initial_len - len(X_clean)
    print(f"[Data] Colonnes gardées : {X_clean.shape[1]}")
    print(f"[Data Cleaning] Lignes supprimées (>= 100 NaN) : {removed_count}")
    print(f"[Data] Final Shape : {X_clean.shape}")
    return X_clean


def masked_mse_loss(recons: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    squared_diff = (recons - target) ** 2
    masked_diff = squared_diff * mask
    return masked_diff.sum() / (mask.sum() + 1e-8)


def timing_indices() -> list[int]:
    return [i for i, c in enumerate(COLUMNS_TO_LOAD) if c.endswith("_total_timing")]


def fit_timing_caps(X_raw: np.ndarray, idx: list[int], q: float) -> dict[int, float]:
    caps: dict[int, float] = {}
    for j in idx:
        col = X_raw[:, j]
        obs = col[~np.isnan(col)]
        obs = obs[np.isfinite(obs)]
        if obs.size == 0:
            caps[j] = np.nan
        else:
            caps[j] = float(np.quantile(obs, q))
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

        col = np.log1p(col)
        X[:, j] = col.astype(np.float32)
    return X


def make_std_vector_from_observed(X_scaled: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float, int]:
    std_vals = np.zeros((X_scaled.shape[1],), dtype=np.float32)

    for j in range(X_scaled.shape[1]):
        obs = X_scaled[mask[:, j] == 1.0, j]
        if obs.size >= 2:
            std_vals[j] = float(np.std(obs))

    std_pos = std_vals[std_vals > 1e-8]
    if std_pos.size == 0:
        cap = 0.0
    else:
        cap = float(np.quantile(std_pos, CONFIG["std_clip_quantile"]))
        cap = min(cap, float(CONFIG["std_clip_max"]))
        cap = max(cap, float(CONFIG["std_clip_floor"]))

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


def train_one_epoch(model: DAE, loader: DataLoader, optimizer, std_vector: torch.Tensor, device) -> float:
    model.train()
    running_loss = 0.0

    for batch_input, batch_target, batch_mask in loader:
        batch_input = batch_input.to(device)
        batch_target = batch_target.to(device)
        batch_mask = batch_mask.to(device)

        optimizer.zero_grad(set_to_none=True)

        recons = model(batch_input, std_vector=std_vector, apply_noise=True)
        loss = masked_mse_loss(recons, batch_target, batch_mask)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG["grad_clip_norm"])
        optimizer.step()

        running_loss += loss.item() * batch_input.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: DAE, loader: DataLoader, device, std_vector: torch.Tensor) -> tuple[float, float]:
    model.eval()
    loss_noisy = 0.0
    loss_clean = 0.0

    for batch_input, batch_target, batch_mask in loader:
        batch_input = batch_input.to(device)
        batch_target = batch_target.to(device)
        batch_mask = batch_mask.to(device)

        recons_noisy = model(batch_input, std_vector=std_vector, apply_noise=True)
        recons_clean = model(batch_input, std_vector=None, apply_noise=False)

        ln = masked_mse_loss(recons_noisy, batch_target, batch_mask)
        lc = masked_mse_loss(recons_clean, batch_target, batch_mask)

        bs = batch_input.size(0)
        loss_noisy += ln.item() * bs
        loss_clean += lc.item() * bs

    n = len(loader.dataset)
    return loss_noisy / n, loss_clean / n


def run_fold(
    mgr: ArtifactManager,
    device: torch.device,
    run_name: str,
    group: str,
    fold_tag: str,
    X_train_raw: np.ndarray,
    X_val_raw: np.ndarray,
    epochs: int,
    patience: int,
    min_epochs: int,
    min_delta: float,
) -> None:
    run = wandb.init(
        project=CONFIG["project_name"],
        group=group,
        name=run_name,
        config=CONFIG,
    )

    tidx = timing_indices()
    caps = fit_timing_caps(X_train_raw, tidx, q=float(CONFIG["timing_winsor_q"]))
    X_train_t = apply_timing_transform(X_train_raw, tidx, caps)
    X_val_t = apply_timing_transform(X_val_raw, tidx, caps)

    mask_train = (~np.isnan(X_train_t)).astype(np.float32)
    mask_val = (~np.isnan(X_val_t)).astype(np.float32)

    imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
    X_train_imp = imputer.fit_transform(X_train_t)
    X_val_imp = imputer.transform(X_val_t)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp).astype(np.float32)
    X_val_scaled = scaler.transform(X_val_imp).astype(np.float32)

    std_concat, cap, nonzero = make_std_vector_from_observed(X_train_scaled, mask_train)
    std_vector_tensor = torch.from_numpy(std_concat).to(device)

    train_loader = build_loader(X_train_scaled, mask_train, CONFIG["batch_size"], shuffle=True)
    val_loader = build_loader(X_val_scaled, mask_val, CONFIG["batch_size"], shuffle=False)

    model = DAE(
        input_dim=2 * N_FEATS,
        output_dim=N_FEATS,
        alpha=CONFIG["noise_alpha"],
        dropout=CONFIG["dropout"],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])

    best_val_clean = float("inf")
    best_encoder = None
    patience_counter = 0

    for epoch in range(epochs):
        train_loss_online = train_one_epoch(model, train_loader, optimizer, std_vector_tensor, device)
        val_noisy, val_clean = evaluate(model, val_loader, device, std_vector_tensor)

        run.log({
            "epoch": epoch,
            "train_loss_online": train_loss_online,
            "val_loss_noisy": val_noisy,
            "val_loss_clean": val_clean,
            "std_clip_cap": cap,
            "std_nonzero": nonzero,
        })

        print(
            f"{run_name} | Epoch {epoch+1:03d} | "
            f"Train(noisy) {train_loss_online:.5f} | Val(noisy) {val_noisy:.5f} | Val(clean) {val_clean:.5f}"
        )

        improved = (best_val_clean - val_clean) > min_delta
        if improved:
            best_val_clean = val_clean
            patience_counter = 0
            best_encoder = copy.deepcopy(model.encoder.state_dict())
        else:
            if epoch + 1 >= min_epochs:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    imputer_path = f"imputer_{fold_tag}.pkl"
    scaler_path = f"scaler_{fold_tag}.pkl"
    encoder_path = f"dae_encoder_{fold_tag}.pth"
    caps_path = f"timing_caps_{fold_tag}.pkl"

    joblib.dump(imputer, imputer_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(caps, caps_path)
    torch.save(best_encoder if best_encoder is not None else model.encoder.state_dict(), encoder_path)

    mgr.log_model(imputer_path, f"imputer-{fold_tag}", run, metadata={"type": "imputer"})
    mgr.log_model(scaler_path, f"scaler-{fold_tag}", run, metadata={"type": "scaler"})
    mgr.log_model(caps_path, f"timing-caps-{fold_tag}", run, metadata={"type": "timing_caps"})
    mgr.log_model(encoder_path, f"dae-encoder-{fold_tag}", run, aliases=["latest"])

    run.finish()

    for f in [imputer_path, scaler_path, caps_path, encoder_path]:
        if os.path.exists(f):
            os.remove(f)


def run_full_fit(
    mgr: ArtifactManager,
    device: torch.device,
    X_full_raw: np.ndarray,
) -> None:
    run = wandb.init(
        project=CONFIG["project_name"],
        group=CONFIG["group_name"] + "_FULL",
        name="dae_masked_full_fit",
        config=CONFIG,
    )

    tidx = timing_indices()
    caps = fit_timing_caps(X_full_raw, tidx, q=float(CONFIG["timing_winsor_q"]))
    X_full_t = apply_timing_transform(X_full_raw, tidx, caps)

    mask_full = (~np.isnan(X_full_t)).astype(np.float32)

    imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
    X_full_imp = imputer.fit_transform(X_full_t)

    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_full_imp).astype(np.float32)

    std_concat, cap, nonzero = make_std_vector_from_observed(X_full_scaled, mask_full)
    std_vector_tensor = torch.from_numpy(std_concat).to(device)

    full_loader = build_loader(X_full_scaled, mask_full, CONFIG["batch_size"], shuffle=True)

    model = DAE(
        input_dim=2 * N_FEATS,
        output_dim=N_FEATS,
        alpha=CONFIG["noise_alpha"],
        dropout=CONFIG["dropout"],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])

    for epoch in range(int(CONFIG["final_epochs"])):
        train_loss_online = train_one_epoch(model, full_loader, optimizer, std_vector_tensor, device)

        run.log({
            "epoch": epoch,
            "train_loss_online": train_loss_online,
            "std_clip_cap": cap,
            "std_nonzero": nonzero,
        })

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"dae_masked_full_fit | Epoch {epoch+1:03d} | Train(noisy) {train_loss_online:.5f}")

    imputer_path = "imputer_full.pkl"
    scaler_path = "scaler_full.pkl"
    encoder_path = "dae_encoder_full.pth"
    caps_path = "timing_caps_full.pkl"

    joblib.dump(imputer, imputer_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(caps, caps_path)
    torch.save(model.encoder.state_dict(), encoder_path)

    mgr.log_model(imputer_path, "imputer-full", run, metadata={"type": "imputer"})
    mgr.log_model(scaler_path, "scaler-full", run, metadata={"type": "scaler"})
    mgr.log_model(caps_path, "timing-caps-full", run, metadata={"type": "timing_caps"})
    mgr.log_model(encoder_path, "dae-encoder-full", run, aliases=["latest"])

    run.finish()

    for f in [imputer_path, scaler_path, caps_path, encoder_path]:
        if os.path.exists(f):
            os.remove(f)


def main():
    mgr = ArtifactManager()
    data_raw = load_data_with_nans()
    device = torch.device(CONFIG["device"])

    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    kfold = KFold(n_splits=CONFIG["n_splits"], shuffle=True, random_state=CONFIG["seed"])

    for fold, (train_idx, val_idx) in enumerate(kfold.split(data_raw)):
        X_train_raw = data_raw[train_idx]
        X_val_raw = data_raw[val_idx]

        run_fold(
            mgr=mgr,
            device=device,
            run_name=f"dae_masked_fold_{fold}",
            group=CONFIG["group_name"],
            fold_tag=f"fold_{fold}",
            X_train_raw=X_train_raw,
            X_val_raw=X_val_raw,
            epochs=int(CONFIG["epochs"]),
            patience=int(CONFIG["patience"]),
            min_epochs=int(CONFIG["min_epochs"]),
            min_delta=float(CONFIG["min_delta"]),
        )

    run_full_fit(mgr=mgr, device=device, X_full_raw=data_raw)


if __name__ == "__main__":
    main()
