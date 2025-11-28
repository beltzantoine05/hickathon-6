import torch
import torch.nn as nn

class NoiseInjection(nn.Module):
    """
    S'applique maintenant sur l'entrée concaténée (220 dim).
    Le std_vector fourni devra donc faire 220 de long.
    """
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, x, std_vector):
        if self.training and self.alpha > 0:
            noise = torch.randn_like(x)
            return x + self.alpha * std_vector * noise
        return x

class Encoder(nn.Module):
    """
    Input Dim passe par défaut à 220 (110 features + 110 masques)
    """
    def __init__(self, input_dim=220): # <--- CHANGE ICI
        super().__init__()
        
        self.layer_1 = nn.Linear(input_dim, 256)
        self.layer_2 = nn.Linear(256, 128)
        self.layer_3 = nn.Linear(128, 64)
        
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.layer_1(x))
        x = self.act(self.layer_2(x))
        x = self.act(self.layer_3(x))
        return x

class Decoder(nn.Module):
    """
    Output Dim reste à 110 (On reconstruit seulement les features)
    """
    def __init__(self, output_dim=110): # <--- RESTE 110
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, output_dim) # Reconstruction des 110 features originales
        )

    def forward(self, x):
        return self.net(x)

class DAE(nn.Module):
    def __init__(self, input_dim=220, output_dim=110, alpha=0.1):
        super().__init__()
        self.noise_layer = NoiseInjection(alpha)
        self.encoder = Encoder(input_dim)
        self.decoder = Decoder(output_dim)

    def forward(self, x, std_vector=None):
        # x est de dimension 220 ici (Values + Mask)
        if std_vector is not None:
            x_noisy = self.noise_layer(x, std_vector)
        else:
            x_noisy = x
            
        latent = self.encoder(x_noisy)
        reconstruction = self.decoder(latent) # Sortie 110
        return reconstruction

# La classe SupervisedRegressor reste inchangée, elle prendra juste l'encoder modifié
class SupervisedRegressor(nn.Module):
    def __init__(self, encoder: Encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        features = self.encoder(x)
        return self.head(features)