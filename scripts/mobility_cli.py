#!/usr/bin/env python3
"""Command-line interface for mobility intelligence operations."""
import argparse
import pathlib
import sys
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

# Add parent to path for imports
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from api.models.gravity import GravityModel
from api.models.hawkes import add_hawkes_trigger

ROOT = pathlib.Path(__file__).resolve().parents[1]

def build_panel(region: str):
    """Build model panel for a region."""
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "etl" / "build_region_panel.py"), region], check=True)

def train(region: str, trigger: bool = True):
    """Train gravity model with optional trigger uplift."""
    panel_path = ROOT / "data" / region / "model_panel.csv"
    if not panel_path.exists():
        raise SystemExit(f"Panel not found for region '{region}'. Run: python etl/build_region_panel.py {region}")
    
    print(f"Loading panel from {panel_path}")
    df = pd.read_csv(panel_path)
    
    if trigger:
        print("Adding Hawkes trigger features...")
        df = add_hawkes_trigger(df)
    
    print("Training gravity model...")
    g = GravityModel.train(df)
    yhat_g = g.predict(df)
    
    # Save models
    (ROOT / "models" / region).mkdir(parents=True, exist_ok=True)
    gravity_path = ROOT / "models" / region / "baseline_gravity.joblib"
    dump(g.pipe, gravity_path)
    print(f"Saved gravity model to {gravity_path}")
    
    # Train uplift if trigger features exist
    if trigger and "trigger" in df.columns and "lag1_flow" in df.columns:
        print("Training trigger uplift model...")
        resid = np.log1p(df["flow"].values) - np.log1p(yhat_g + 1e-6)
        X = df[["trigger", "lag1_flow"]].values
        lm = LinearRegression().fit(X, resid)
        
        uplift_path = ROOT / "models" / region / "trigger_uplift.joblib"
        dump(lm, uplift_path)
        print(f"Saved uplift model to {uplift_path}")
    else:
        print("Skipped uplift training (no trigger features)")

def train_quantiles(region: str, trigger: bool = True):
    """Train quantile regression models."""
    panel_path = ROOT / "data" / region / "model_panel.csv"
    if not panel_path.exists():
        raise SystemExit(f"Panel not found for region '{region}'")
    
    print(f"Loading panel from {panel_path}")
    df = pd.read_csv(panel_path)
    
    if trigger:
        print("Adding Hawkes trigger features...")
        df = add_hawkes_trigger(df)
    
    # Prepare features
    from api.models.gravity import FEATURES as GFEATS
    feats = list(GFEATS)
    for extra in ["trigger", "lag1_flow"]:
        if extra in df.columns:
            feats.append(extra)
    
    available_feats = [f for f in feats if f in df.columns]
    
    # Prepare data
    from api.models.gravity import GravityModel
    df_prep = GravityModel.prepare(df)
    X = df_prep[available_feats].values
    y = np.log1p(df["flow"].values)
    
    # Train quantile models
    (ROOT / "models" / region).mkdir(parents=True, exist_ok=True)
    
    for alpha, name in [(0.1, "p10"), (0.5, "p50"), (0.9, "p90")]:
        print(f"Training {name} (alpha={alpha})...")
        gbr = GradientBoostingRegressor(
            loss="quantile",
            alpha=alpha,
            n_estimators=300,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        gbr.fit(X, y)
        
        model_path = ROOT / "models" / region / f"quantile_{name}.joblib"
        dump(gbr, model_path)
        print(f"Saved {name} model to {model_path}")
    
    print(f"✓ Trained quantile models (P10/P50/P90) for {region}")

def main():
    """Main CLI entry point."""
    ap = argparse.ArgumentParser(description="Mobility Intelligence CLI")
    ap.add_argument("command", choices=["build_panel", "train", "train_quantiles"],
                   help="Command to execute")
    ap.add_argument("--region", required=True, help="Region identifier")
    ap.add_argument("--no-trigger", action="store_true", help="Skip trigger features")
    
    args = ap.parse_args()
    
    if args.command == "build_panel":
        build_panel(args.region)
    elif args.command == "train":
        train(args.region, trigger=(not args.no_trigger))
    elif args.command == "train_quantiles":
        train_quantiles(args.region, trigger=(not args.no_trigger))

if __name__ == "__main__":
    main()
