import random
import os
import numpy as np
import torch

def seed_everything(seed: int = 42):
    """
    Fixe toutes les graines aléatoires pour garantir la reproductibilité.
    Indispensable pour que les runs W&B soient comparables.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Pour le multi-GPU
    
    # Force le déterminisme des algorithmes de convolution (peut ralentir légèrement)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"[Utils] Global seed set to {seed}")

def count_parameters(model):
    """Retourne le nombre de paramètres entraînables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)