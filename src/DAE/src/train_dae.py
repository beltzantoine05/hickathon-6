import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer # <--- Nouveau
import numpy as np
import wandb
import copy
import os
import joblib
import pandas as pd

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

CONFIG = {
    "project_name": "ml_project_embedding",
    "group_name": "DAE_Pretraining_Masked",
    "seed": 42,
    "n_splits": 5,
    "batch_size": 1024,
    "epochs": 50,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "noise_alpha": 0.1,
    "patience": 5,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def load_data_with_nans():
    """
    1. Charge X_train.csv
    2. Garde uniquement COLUMNS_TO_LOAD
    3. Supprime les lignes ayant >= 100 valeurs manquantes (sur les 110 colonnes)
    """
    file_path = os.path.join("datas", "X_train.csv")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    print(f"[Data] Chargement de {file_path}...")
    
    # Lecture
    df = pd.read_csv(file_path)
    
    # --- A. FILTRE COLONNES ---
    # On s'assure qu'on ne travaille que sur les features désirées
    missing_cols = [c for c in COLUMNS_TO_LOAD if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes : {missing_cols}")
    
    df = df[COLUMNS_TO_LOAD]
    
    # Conversion NumPy (N, 110)
    X = df.to_numpy(dtype=np.float32)
    initial_len = len(X)
    
    # --- B. FILTRE LIGNES (Condition < 100 NaNs) ---
    # On compte le nombre de NaN par ligne (axis=1)
    n_missing = np.isnan(X).sum(axis=1)
    
    # Condition : Avoir strictement moins de 100 valeurs manquantes
    mask_keep = n_missing < 100
    
    X_clean = X[mask_keep]
    
    removed_count = initial_len - len(X_clean)
    print(f"[Data] Colonnes gardées : {X_clean.shape[1]}")
    print(f"[Data Cleaning] Lignes supprimées (>= 100 NaN) : {removed_count}")
    print(f"[Data] Final Shape : {X_clean.shape}")
    
    return X_clean

def masked_mse_loss(input_recons, target, mask):
    """
    Calcule la MSE uniquement sur les valeurs présentes (Mask = 1).
    input_recons: [Batch, 110]
    target:       [Batch, 110]
    mask:         [Batch, 110] (1 si valeur présente, 0 si manquante)
    """
    squared_diff = (input_recons - target) ** 2
    # On annule l'erreur là où la donnée manquait à l'origine
    masked_diff = squared_diff * mask
    # Somme des erreurs / Nombre de valeurs présentes (epsilon pour éviter div/0)
    loss = masked_diff.sum() / (mask.sum() + 1e-8)
    return loss

def train_one_epoch(model, loader, optimizer, std_vector, device):
    model.train()
    running_loss = 0.0
    
    # Le loader retourne: Inputs (220), Targets (110), Masque (110)
    for batch_input, batch_target, batch_mask in loader:
        batch_input = batch_input.to(device)   # [Batch, 220]
        batch_target = batch_target.to(device) # [Batch, 110] (Valeurs scalées imputées)
        batch_mask = batch_mask.to(device)     # [Batch, 110]
        
        optimizer.zero_grad()
        
        # Forward : L'input contient [Values | Mask]
        reconstruction = model(batch_input, std_vector) # Output [Batch, 110]
        
        # Loss custom : On compare la reconstruction à la target, filtré par le masque
        loss = masked_mse_loss(reconstruction, batch_target, batch_mask)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_input.size(0)
        
    return running_loss / len(loader.dataset)

def validate(model, loader, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch_input, batch_target, batch_mask in loader:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            batch_mask = batch_mask.to(device)
            
            # Validation sans bruit
            reconstruction = model(batch_input, std_vector=None)
            
            loss = masked_mse_loss(reconstruction, batch_target, batch_mask)
            running_loss += loss.item() * batch_input.size(0)
            
    return running_loss / len(loader.dataset)

def main():
    mgr = ArtifactManager()
    data_raw = load_data_with_nans() # Données brutes avec np.nan
    device = torch.device(CONFIG["device"])
    
    kfold = KFold(n_splits=CONFIG["n_splits"], shuffle=True, random_state=CONFIG["seed"])
    
    print(f"Starting Mask-Aware DAE Training...")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(data_raw)):
        print(f"\n=== Fold {fold} ===")
        
        run = wandb.init(
            project=CONFIG["project_name"],
            group=CONFIG["group_name"],
            name=f"dae_masked_fold_{fold}",
            config=CONFIG
        )
        
        # 1. SPLIT
        X_train_raw = data_raw[train_idx]
        X_val_raw = data_raw[val_idx]
        
        # 2. CALCUL DU MASQUE DE PRÉSENCE (1=Present, 0=Absent)
        # On le fait avant imputation bien sûr
        mask_train = (~np.isnan(X_train_raw)).astype(np.float32)
        mask_val = (~np.isnan(X_val_raw)).astype(np.float32)
        
        # 3. IMPUTATION (Mean)
        # Fit sur Train uniquement
        imputer = SimpleImputer(strategy='mean')
        X_train_imputed = imputer.fit_transform(X_train_raw)
        X_val_imputed = imputer.transform(X_val_raw)
        
        # Sauvegarde Imputer
        imputer_path = f"imputer_fold_{fold}.pkl"
        joblib.dump(imputer, imputer_path)
        mgr.log_model(imputer_path, f"imputer-fold-{fold}", run, metadata={"type": "imputer"})
        
        # 4. SCALING (Robust)
        # Fit sur Train Imputé
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_val_scaled = scaler.transform(X_val_imputed)
        
        # Sauvegarde Scaler
        scaler_path = f"scaler_fold_{fold}.pkl"
        joblib.dump(scaler, scaler_path)
        mgr.log_model(scaler_path, f"scaler-fold-{fold}", run, metadata={"type": "scaler"})
        
        # 5. MERGE (Création de l'input 220 colonnes)
        # Input = [Scaled Values, Mask]
        X_train_concat = np.hstack([X_train_scaled, mask_train]) # (N, 220)
        X_val_concat = np.hstack([X_val_scaled, mask_val])       # (N, 220)
        
        # 6. STD VECTOR (Sur les 220 colonnes !)
        train_std = np.std(X_train_concat, axis=0)
        train_std[train_std == 0] = 1.0
        std_vector_tensor = torch.from_numpy(train_std).float().to(device)
        
        # 7. DATALOADERS
        # On passe 3 Tensors : Input(220), Target(110), Mask(110)
        # Note: Target = X_train_scaled (les valeurs imputées servent de target pour la forme, 
        # mais la loss les ignorera grâce au masque)
        train_ds = TensorDataset(
            torch.from_numpy(X_train_concat).float(), 
            torch.from_numpy(X_train_scaled).float(),
            torch.from_numpy(mask_train).float()
        )
        val_ds = TensorDataset(
            torch.from_numpy(X_val_concat).float(), 
            torch.from_numpy(X_val_scaled).float(),
            torch.from_numpy(mask_val).float()
        )
        
        train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False)
        
        # 8. MODEL INIT (220 inputs -> 110 outputs)
        model = DAE(input_dim=220, output_dim=110, alpha=CONFIG["noise_alpha"]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
        
        # 9. TRAINING
        best_val_loss = float('inf')
        patience_counter = 0
        best_encoder = None
        
        for epoch in range(CONFIG["epochs"]):
            # On passe optimiser et std_vector à la fonction custom
            train_loss = train_one_epoch(model, train_loader, optimizer, std_vector_tensor, device)
            val_loss = validate(model, val_loader, device)
            
            run.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            print(f"Epoch {epoch+1:02d} | Val Loss (Masked): {val_loss:.5f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_encoder = copy.deepcopy(model.encoder.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= CONFIG["patience"]:
                    break
        
        # 10. SAVE
        encoder_path = f"dae_encoder_fold_{fold}.pth"
        torch.save(best_encoder, encoder_path)
        
        mgr.log_model(encoder_path, f"dae-masked-encoder-fold-{fold}", run, aliases=["latest"])
        
        run.finish()
        # Cleanup local
        for f in [imputer_path, scaler_path, encoder_path]:
            if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    main()