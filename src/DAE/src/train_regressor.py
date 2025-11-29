import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
import wandb
import joblib
import os
import pandas as pd

# Imports locaux
from infrastructure import ArtifactManager
from models import Encoder, SupervisedRegressor

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

# Configuration du Fine-tuning
CONFIG = {
    "project_name": "ml_project_embedding",
    "group_name": "Supervised_Finetuning_Masked",
    "seed": 42,
    "n_splits": 5,
    "batch_size": 1024,

    # Architecture
    "input_dim": 220,  # 110 features + 110 masques
    "dropout": 0.1,

    # Phase 1: Linear Probe
    "lr_head_init": 1e-3,
    "epochs_phase_1": 5,

    # Phase 2: Fine-tuning
    "lr_encoder": 3e-5,
    "lr_head_finetune": 3e-4,
    "epochs_phase_2": 20,
    "weight_decay": 1e-4,

    # Timing preprocessing (doit être identique à train_dae.py)
    "timing_winsor_q": 0.999,
    "timing_nonneg": True,

    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


def load_data_and_targets():
    """
    Charge X et Y, filtre les colonnes, et applique la condition < 100 NaNs.
    """
    x_path = os.path.join("datas", "X_train.csv")
    y_path = os.path.join("datas", "y_train.csv")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError("Fichiers introuvables dans ../datas/")

    print("[Data] Chargement des CSV...")

    # 1. Chargement avec index
    df_x = pd.read_csv(x_path, index_col=0)
    df_y = pd.read_csv(y_path, index_col=0)

    print(f"[Data] X shape avant join: {df_x.shape}, Y shape avant join: {df_y.shape}")

    # 2. Inner join sur l'index pour garantir l'alignement
    df_merged = df_x.join(df_y[["MathScore"]], how="inner")
    
    print(f"[Data] Shape après join: {df_merged.shape}")
    print(f"[Data] Lignes perdues lors du join: X={len(df_x) - len(df_merged)}, Y={len(df_y) - len(df_merged)}")

    # 3. Vérifier les colonnes features
    missing_cols = [c for c in COLUMNS_TO_LOAD if c not in df_merged.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans X : {missing_cols}")

    # 4. Séparer X et Y depuis le DataFrame joint
    X = df_merged[COLUMNS_TO_LOAD].to_numpy(dtype=np.float32)
    Y = df_merged["MathScore"].to_numpy(dtype=np.float32)

    # 5. Filtre Lignes (Condition < 100 NaNs sur X)
    initial_len = len(X)
    n_missing = np.isnan(X).sum(axis=1)
    mask_keep = n_missing < 100

    X_clean = X[mask_keep]
    Y_clean = Y[mask_keep]

    print(f"[Data] Colonnes sélectionnées : {X_clean.shape[1]}")
    print(f"[Data Cleaning] Lignes supprimées (>= 100 NaN) : {initial_len - len(X_clean)}")
    print(f"[Data] X: {X_clean.shape}, Y: {Y_clean.shape}")

    return X_clean, Y_clean


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        preds = model(batch_x).squeeze()
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_x.size(0)
    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds = model(batch_x).squeeze()
            loss = criterion(preds, batch_y)
            running_loss += loss.item() * batch_x.size(0)
    return running_loss / len(loader.dataset)


def timing_indices() -> list[int]:
    """Retourne les indices des colonnes de timing."""
    return [i for i, c in enumerate(COLUMNS_TO_LOAD) if c.endswith("_total_timing")]


def fit_timing_caps(X_raw: np.ndarray, idx: list[int], q: float) -> dict[int, float]:
    """Calcule les plafonds (caps) pour les colonnes de timing."""
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
    """Applique la transformation aux colonnes de timing (caps + log1p)."""
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


def preprocess_data(X_raw: np.ndarray, imputer, scaler, caps: dict[int, float]) -> np.ndarray:
    """
    Applique le preprocessing complet (identique à train_dae.py) :
    1. Transformation des timings (caps + log1p)
    2. Calcul du masque
    3. Imputation
    4. Scaling
    5. Concaténation [Values, Mask] -> 220 dims
    """
    # 1. Transformation des timings
    tidx = timing_indices()
    X_transformed = apply_timing_transform(X_raw, tidx, caps)

    # 2. Masque (1 = Present, 0 = Absent)
    mask = (~np.isnan(X_transformed)).astype(np.float32)

    # 3. Imputation
    X_imputed = imputer.transform(X_transformed)

    # 4. Scaling
    X_scaled = scaler.transform(X_imputed).astype(np.float32)

    # 5. Concaténation [Values, Mask] -> Dim 220
    X_final = np.hstack([X_scaled, mask]).astype(np.float32)

    return X_final


def main():
    mgr = ArtifactManager()
    X_raw_all, Y_all = load_data_and_targets()
    device = torch.device(CONFIG["device"])

    kfold = KFold(n_splits=CONFIG["n_splits"], shuffle=True, random_state=CONFIG["seed"])

    print(f"Starting Supervised Training (Masked Input) on {CONFIG['device']}...")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_raw_all)):
        print(f"\n=== Fold {fold} ===")

        run = wandb.init(
            project=CONFIG["project_name"],
            group=CONFIG["group_name"],
            name=f"regressor_masked_fold_{fold}",
            job_type="finetuning",
            config=CONFIG
        )

        # -----------------------------------------------------------
        # 1. TÉLÉCHARGEMENT DE L'ENCODEUR PRÉ-ENTRAÎNÉ (v0 = espace latent 64)
        # -----------------------------------------------------------
        encoder_path = f"dl_encoder_fold_{fold}.pth"
        mgr.download_artifact(f"dae-encoder-fold_{fold}:64", encoder_path, run)

        # -----------------------------------------------------------
        # 2. PREPROCESSING (identique à train_dae.py)
        # -----------------------------------------------------------
        X_train_raw = X_raw_all[train_idx]
        X_val_raw = X_raw_all[val_idx]

        # A. Fit timing caps sur le train
        tidx = timing_indices()
        caps = fit_timing_caps(X_train_raw, tidx, q=float(CONFIG["timing_winsor_q"]))
        
        # B. Appliquer la transformation timing
        X_train_t = apply_timing_transform(X_train_raw, tidx, caps)
        X_val_t = apply_timing_transform(X_val_raw, tidx, caps)

        # C. Fit imputer sur train
        imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
        X_train_imp = imputer.fit_transform(X_train_t)
        X_val_imp = imputer.transform(X_val_t)

        # D. Fit scaler sur train
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp).astype(np.float32)
        X_val_scaled = scaler.transform(X_val_imp).astype(np.float32)

        # E. Créer les masques
        mask_train = (~np.isnan(X_train_t)).astype(np.float32)
        mask_val = (~np.isnan(X_val_t)).astype(np.float32)

        # F. Concaténer [Values, Mask]
        X_train = np.hstack([X_train_scaled, mask_train]).astype(np.float32)
        X_val = np.hstack([X_val_scaled, mask_val]).astype(np.float32)

        Y_train, Y_val = Y_all[train_idx], Y_all[val_idx]

        # Dataloaders
        train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float())
        val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).float())

        train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False)

        # -----------------------------------------------------------
        # 3. INITIALISATION DU MODÈLE
        # -----------------------------------------------------------
        encoder = Encoder(input_dim=CONFIG["input_dim"], dropout=CONFIG["dropout"])
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))

        model = SupervisedRegressor(encoder).to(device)
        criterion = nn.MSELoss()

        # -----------------------------------------------------------
        # 4. PHASE 1 : LINEAR PROBE (Freeze Encoder)
        # -----------------------------------------------------------
        print("--> Phase 1: Training Head Only")
        for param in model.encoder.parameters():
            param.requires_grad = False

        optimizer_p1 = optim.Adam(model.head.parameters(), lr=CONFIG["lr_head_init"])

        for epoch in range(CONFIG["epochs_phase_1"]):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer_p1, device)
            val_loss = validate(model, val_loader, criterion, device)
            run.log({"phase": 1, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            print(f"P1 Epoch {epoch + 1} | Loss: {train_loss:.4f} | Val: {val_loss:.4f} ")

        # -----------------------------------------------------------
        # 5. PHASE 2 : FINE-TUNING (Unfreeze progressif)
        # -----------------------------------------------------------
        print("--> Phase 2: Fine-tuning")

        # Unfreeze toutes les couches de l'encodeur
        for param in model.encoder.parameters():
            param.requires_grad = True

        # Optimizer avec learning rates différenciés
        # Note: l'encoder est un Sequential, donc on accède aux couches via net[index]
        # Couches: net[0]=Linear(220,256), net[3]=Linear(256,128), net[6]=Linear(128,64)
        
        # On peut freeze la première couche qui a appris la fusion masque+valeurs
        for param in model.encoder.net[0].parameters():
            param.requires_grad = False

        optimizer_p2 = optim.AdamW([
            {'params': model.encoder.net[3].parameters(), 'lr': CONFIG["lr_encoder"]},  # Layer 256->128
            {'params': model.encoder.net[6].parameters(), 'lr': CONFIG["lr_encoder"]},  # Layer 128->64
            {'params': model.head.parameters(), 'lr': CONFIG["lr_head_finetune"]}
        ], weight_decay=CONFIG["weight_decay"])

        best_val_loss = float('inf')

        for epoch in range(CONFIG["epochs_phase_2"]):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer_p2, device)
            val_loss = validate(model, val_loader, criterion, device)

            run.log({"phase": 2, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            print(f"P2 Epoch {epoch + 1} | Loss: {train_loss:.4f} | Val: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"temp_best_reg_{fold}.pth")

        # -----------------------------------------------------------
        # 6. SAUVEGARDE FINALE
        # -----------------------------------------------------------
        model.load_state_dict(torch.load(f"temp_best_reg_{fold}.pth"))

        # Save Full Regressor
        reg_path = f"regressor_masked_fold_{fold}.pth"
        torch.save(model.state_dict(), reg_path)
        mgr.log_model(reg_path, f"supervised-regressor-masked-fold-{fold}", run)

        # Save Embedding Model (Encoder 220 inputs)
        emb_path = f"embedding_masked_fold_{fold}.pth"
        torch.save(model.encoder.state_dict(), emb_path)
        mgr.log_model(emb_path, f"final-embedding-masked-fold-{fold}", run, aliases=["production"])

        run.finish()

        # Cleanup
        for f in [encoder_path, f"temp_best_reg_{fold}.pth", reg_path, emb_path]:
            if os.path.exists(f): 
                os.remove(f)


if __name__ == "__main__":
    main()