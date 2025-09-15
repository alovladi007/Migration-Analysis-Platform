#!/usr/bin/env python3
"""Train temporal GNN model for quantile sequence prediction."""
import argparse
import pathlib
import numpy as np
import pandas as pd
import json
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not installed. Install requirements-ml.txt to use TGNN.")

from api.models.temporal_gnn import train_gru_quantiles

ROOT = pathlib.Path(__file__).resolve().parents[1]

def build_sequences(df: pd.DataFrame, T: int = 6, horizons: int = 3):
    """Build temporal sequences for training."""
    df = df.copy()
    df["t"] = pd.PeriodIndex(df["period"], freq="M").to_timestamp(how="end")
    df = df.sort_values(["origin_id", "dest_id", "t"])
    
    # Feature columns
    feats_cols = [
        "pop_o", "chirps_spi3_o", "era5_tmax_anom_o", "access_score_o",
        "pop_d", "chirps_spi3_d", "era5_tmax_anom_d", "access_score_d"
    ]
    
    # Add optional features
    for c in ["lag1_flow", "trigger", "acled_intensity_o", "acled_intensity_d"]:
        if c in df.columns:
            feats_cols.append(c)
        elif c in ["lag1_flow", "trigger"]:
            df[c] = 0.0
            feats_cols.append(c)
    
    # Build sequences
    Xs = []
    Ys = {"p10": [], "p50": [], "p90": []}
    
    for (o, d), g in df.groupby(["origin_id", "dest_id"]):
        g = g.reset_index(drop=True)
        y = np.log1p(g["flow"].values)
        
        # Normalize features
        X = g[feats_cols].fillna(0).values.astype(np.float32)
        
        # Create sliding windows
        for i in range(len(g) - T - horizons + 1):
            Xs.append(X[i:i+T])
            
            # Target values for each quantile
            target_values = y[i+T:i+T+horizons]
            for q in ["p10", "p50", "p90"]:
                # For training, we use the actual values as targets for all quantiles
                # The model learns to predict different quantiles through the loss function
                Ys[q].append(target_values)
    
    if not Xs:
        raise RuntimeError(f"Not enough data to build sequences (need at least {T+horizons} periods)")
    
    # Convert to tensors
    Xs = torch.tensor(np.stack(Xs), dtype=torch.float32)
    targets = {k: torch.tensor(np.stack(v), dtype=torch.float32) for k, v in Ys.items()}
    input_dim = Xs.shape[-1]
    
    return Xs, targets, input_dim

def main():
    """Main training function."""
    if not HAS_TORCH:
        raise SystemExit("PyTorch required. Install with: pip install -r requirements-ml.txt")
    
    ap = argparse.ArgumentParser(description="Train Temporal GNN")
    ap.add_argument("--region", required=True, help="Region identifier")
    ap.add_argument("--window", type=int, default=6, help="Sequence window size")
    ap.add_argument("--horizons", type=int, default=3, help="Prediction horizons")
    ap.add_argument("--epochs", type=int, default=8, help="Training epochs")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    
    args = ap.parse_args()
    
    # Load data
    panel_path = ROOT / "data" / args.region / "model_panel.csv"
    if not panel_path.exists():
        raise SystemExit(f"Panel not found: {panel_path}")
    
    print(f"Loading data from {panel_path}")
    df = pd.read_csv(panel_path)
    
    # Add trigger features
    from api.models.hawkes import add_hawkes_trigger
    df = add_hawkes_trigger(df)
    
    # Build sequences
    print(f"Building sequences (window={args.window}, horizons={args.horizons})...")
    Xs, targets, input_dim = build_sequences(df, T=args.window, horizons=args.horizons)
    print(f"Created {len(Xs)} sequences with {input_dim} features")
    
    # Train model
    print(f"Training GRU quantile model for {args.epochs} epochs...")
    model = train_gru_quantiles(
        Xs, targets, input_dim,
        horizons=args.horizons,
        epochs=args.epochs,
        lr=args.lr
    )
    
    # Save model
    outdir = ROOT / "models" / args.region
    outdir.mkdir(parents=True, exist_ok=True)
    
    model_path = outdir / "tgnn_quantiles.pt"
    torch.save(model.state_dict(), model_path)
    
    meta_path = outdir / "tgnn_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "window": args.window,
            "horizons": args.horizons,
            "input_dim": input_dim,
            "epochs": args.epochs
        }, f)
    
    print(f"✓ Saved TGNN model to {model_path}")
    print(f"✓ Saved metadata to {meta_path}")

if __name__ == "__main__":
    main()
