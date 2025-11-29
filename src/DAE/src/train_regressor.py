import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import numpy as np
import wandb
import joblib
import os
import pandas as pd

# Imports locaux
from src.infrastructure import ArtifactManager
from src.models import Encoder, SupervisedRegressor

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

# Configuration du Fine-tuning
CONFIG = {
    "project_name": "ml_project_embedding",
    "group_name": "Supervised_Finetuning_Masked",
    "seed": 42,
    "n_splits": 5,
    "batch_size": 1024,
    
    # Architecture
    "input_dim": 220, # 110 features + 110 masques
    
    # Phase 1: Linear Probe
    "lr_head_init": 1e-3,
    "epochs_phase_1": 5,
    
    # Phase 2: Fine-tuning
    "lr_encoder": 3e-5,
    "lr_head_finetune": 3e-4,
    "epochs_phase_2": 20,
    "weight_decay": 1e-4,
    
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def load_data_and_targets():
    """
    Charge X et Y, filtre les colonnes, et applique la condition < 100 NaNs.
    """
    x_path = os.path.join("..", "datas", "X_train.csv")
    y_path = os.path.join("..", "datas", "y_train.csv")
    
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError("Fichiers introuvables dans ../datas/")

    print("[Data] Chargement des CSV...")
    
    # 1. Chargement X
    df_x = pd.read_csv(x_path)
    
    # Filtre Colonnes
    missing_cols = [c for c in COLUMNS_TO_LOAD if c not in df_x.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans X : {missing_cols}")
        
    df_x = df_x[COLUMNS_TO_LOAD]
    X = df_x.to_numpy(dtype=np.float32)
    
    # 2. Chargement Y
    df_y = pd.read_csv(y_path)
    Y = df_y.to_numpy(dtype=np.float32)
    if Y.ndim > 1: Y = Y.squeeze()
    
    # Check alignement initial
    assert len(X) == len(Y), "X et Y n'ont pas la même longueur !"
    
    # 3. Filtre Lignes (Condition < 100 NaNs)
    initial_len = len(X)
    
    # Compte des NaNs par ligne sur X
    n_missing = np.isnan(X).sum(axis=1)
    
    # Masque booléen
    mask_keep = n_missing < 100
    
    # Application du masque sur X ET Y
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

def preprocess_data(X_raw, imputer, scaler):
    """
    Applique la transformation rigoureuse 220 dims :
    1. Calcul Masque
    2. Imputation (Transform)
    3. Scaling (Transform)
    4. Concaténation
    """
    # 1. Masque (1 = Present, 0 = Absent)
    mask = (~np.isnan(X_raw)).astype(np.float32)
    
    # 2. Imputation (Utilise l'imputer chargé)
    X_imputed = imputer.transform(X_raw)
    
    # 3. Scaling (Utilise le scaler chargé)
    X_scaled = scaler.transform(X_imputed)
    
    # 4. Concaténation [Values, Mask] -> Dim 220
    X_final = np.hstack([X_scaled, mask])
    
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
        # 1. TÉLÉCHARGEMENT DES ARTEFACTS (Imputer + Scaler + Encoder)
        # -----------------------------------------------------------
        
        # A. Imputer
        imputer_path = f"dl_imputer_fold_{fold}.pkl"
        mgr.download_artifact(f"imputer-fold-{fold}", imputer_path, run)
        imputer = joblib.load(imputer_path)
        
        # B. Scaler
        scaler_path = f"dl_scaler_fold_{fold}.pkl"
        mgr.download_artifact(f"scaler-fold-{fold}", scaler_path, run)
        scaler = joblib.load(scaler_path)
        
        # C. Encoder Weights
        encoder_path = f"dl_encoder_fold_{fold}.pth"
        mgr.download_artifact(f"dae-masked-encoder-fold-{fold}", encoder_path, run)
        
        # -----------------------------------------------------------
        # 2. PREPROCESSING (Miroir du DAE)
        # -----------------------------------------------------------
        X_train = preprocess_data(X_raw_all[train_idx], imputer, scaler)
        X_val = preprocess_data(X_raw_all[val_idx], imputer, scaler)
        
        Y_train, Y_val = Y_all[train_idx], Y_all[val_idx]
        
        # Dataloaders
        train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float())
        val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).float())
        
        train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False)
        
        # -----------------------------------------------------------
        # 3. INITIALISATION DU MODÈLE
        # -----------------------------------------------------------
        # Note: Input dim est maintenant 220
        encoder = Encoder(input_dim=CONFIG["input_dim"]) 
        encoder.load_state_dict(torch.load(encoder_path))
        
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
            print(f"P1 Epoch {epoch+1} | Val: {val_loss:.4f}")

        # -----------------------------------------------------------
        # 5. PHASE 2 : FINE-TUNING (Partial Unfreeze)
        # -----------------------------------------------------------
        print("--> Phase 2: Fine-tuning")
        
        # Unfreeze total
        for param in model.encoder.parameters():
            param.requires_grad = True
            
        # Freeze PREMIÈRE COUCHE (Celle qui map 220 -> 256)
        # C'est important car elle a appris à fusionner Masque et Valeur
        for param in model.encoder.layer_1.parameters():
            param.requires_grad = False
            
        optimizer_p2 = optim.AdamW([
            {'params': model.encoder.layer_2.parameters(), 'lr': CONFIG["lr_encoder"]},
            {'params': model.encoder.layer_3.parameters(), 'lr': CONFIG["lr_encoder"]},
            {'params': model.head.parameters(), 'lr': CONFIG["lr_head_finetune"]}
        ], weight_decay=CONFIG["weight_decay"])
        
        best_val_loss = float('inf')
        
        for epoch in range(CONFIG["epochs_phase_2"]):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer_p2, device)
            val_loss = validate(model, val_loader, criterion, device)
            
            run.log({"phase": 2, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            print(f"P2 Epoch {epoch+1} | Val: {val_loss:.4f}")
            
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
        for f in [imputer_path, scaler_path, encoder_path, f"temp_best_reg_{fold}.pth", reg_path, emb_path]:
            if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    main()