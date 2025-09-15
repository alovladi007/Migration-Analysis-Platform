"""Temporal GNN model for sequence-based quantile prediction."""
import math
import json
from typing import List, Tuple, Dict

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = object
    HAS_TORCH = False

def pinball_loss(y_pred, y_true, tau: float):
    """Pinball loss for quantile regression."""
    if not HAS_TORCH:
        return 0.0
    diff = y_true - y_pred
    return torch.mean(torch.maximum(tau * diff, (tau - 1) * diff))

class GRUQuantiles(nn.Module if HAS_TORCH else object):
    """GRU-based quantile prediction model."""
    
    def __init__(self, input_dim: int, hidden: int = 64, horizons: int = 3, 
                 quantiles: Tuple[float] = (0.1, 0.5, 0.9)):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not installed. Install requirements-ml.txt")
        
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.quantiles = quantiles
        self.horizons = horizons
        
        # Separate head for each quantile
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, horizons)
            ) for _ in quantiles
        ])
    
    def forward(self, x):
        """Forward pass returning quantile predictions."""
        # x: (B, T, F)
        _, h = self.gru(x)  # h: (1, B, H)
        h = h.squeeze(0)    # (B, H)
        
        # Get predictions for each quantile
        outs = [head(h) for head in self.heads]  # List of (B, horizons)
        return outs

def train_gru_quantiles(seqs, targets, input_dim: int, horizons: int = 3, 
                       epochs: int = 5, lr: float = 1e-3, 
                       quantiles: Tuple[float] = (0.1, 0.5, 0.9), 
                       device: str = "cpu"):
    """Train GRU quantile model with pinball loss."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch not installed")
    
    model = GRUQuantiles(input_dim, horizons=horizons, quantiles=quantiles).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    qnames = {0.1: "p10", 0.5: "p50", 0.9: "p90"}
    taus = list(quantiles)
    
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        
        outs = model(seqs.to(device))  # List of (B, H)
        loss = 0.0
        
        for out, tau in zip(outs, taus):
            if qnames[tau] in targets:
                y_true = targets[qnames[tau]].to(device)
                loss = loss + pinball_loss(out.flatten(), y_true.flatten(), tau)
        
        loss.backward()
        opt.step()
    
    return model
