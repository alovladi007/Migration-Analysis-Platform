"""FastAPI application for MAP - Migration Analysis Platform."""
from fastapi import FastAPI, Query, Depends, Response, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import pathlib
import os
import json
import io
import csv
from joblib import load

from api.auth import require_api_key, apply_aggregation_floor
from api.scenarios import DEFAULT_PRIORS, apply_scenario
from api.models.gravity import GravityModel
from api.models.hawkes import add_hawkes_trigger

app = FastAPI(
    title="MAP - Migration Analysis Platform", 
    version="0.1.1",
    description="Migration flow prediction with uncertainty quantification"
)
ROOT = pathlib.Path(__file__).resolve().parents[1]
AGG_FLOOR = float(os.environ.get("AGGREGATION_FLOOR", "5"))

class ForecastRequest(BaseModel):
    region: str = "toy"
    period_start: str = "2020-12"
    periods: int = 1
    scenario: str = "baseline"

class ScenarioRequest(ForecastRequest):
    months: int = 3
    priors: Optional[Dict] = None

def get_region_models(region: str):
    """Load gravity and uplift models for a region."""
    base = ROOT / "models" / region
    if not base.exists():
        base = ROOT / "models" / "toy"
    
    gpath = base / "baseline_gravity.joblib"
    upath = base / "trigger_uplift.joblib"
    
    gpipe = load(str(gpath)) if gpath.exists() else None
    uplift = load(str(upath)) if upath.exists() else None
    
    return gpipe, uplift

def get_region_quantile_models(region: str):
    """Load quantile models for a region."""
    base = ROOT / "models" / region
    if not base.exists():
        base = ROOT / "models" / "toy"
    
    paths = {
        "p10": base / "quantile_p10.joblib",
        "p50": base / "quantile_p50.joblib",
        "p90": base / "quantile_p90.joblib"
    }
    
    models = {}
    for k, p in paths.items():
        models[k] = load(str(p)) if p.exists() else None
    
    return models

def load_region_panel(region: str):
    """Load the model panel for a region."""
    data_path = ROOT / "data" / region / "model_panel.csv"
    if not data_path.exists():
        data_path = ROOT / "data" / "toy" / "model_panel.csv"
    
    if not data_path.exists():
        raise HTTPException(status_code=404, detail=f"No data found for region: {region}")
    
    return pd.read_csv(data_path)

@app.get("/")
def root():
    """API root endpoint."""
    return {"message": "MAP - Migration Analysis Platform", "version": "0.1.1"}

@app.post("/forecast_map_region")
def forecast_map_region(req: ForecastRequest, ok: bool = Depends(require_api_key)):
    """Return per-admin predictions for map visualization."""
    df_hist = load_region_panel(req.region)
    df_aug = add_hawkes_trigger(df_hist)
    lastp = df_aug["period"].max()
    base = df_aug[df_aug["period"] == lastp].copy()
    
    gpipe, uplift = get_region_models(req.region)
    if gpipe is None:
        raise HTTPException(status_code=404, detail=f"No model found for region: {req.region}")
    
    g = GravityModel(pipe=gpipe)
    yhat_g = g.predict(base)
    
    # Apply uplift if available
    yhat = yhat_g
    if uplift is not None and "trigger" in base.columns and "lag1_flow" in base.columns:
        X = base[["trigger", "lag1_flow"]].values
        resid_hat = uplift.predict(X)
        yhat = np.expm1(np.log1p(yhat_g) + resid_hat)
    
    base["yhat"] = yhat
    per_origin = base.groupby("origin_id", as_index=False)["yhat"].sum()
    
    preds = [{"admin_id": r.origin_id, "yhat": float(r.yhat)} for r in per_origin.itertuples()]
    preds = apply_aggregation_floor(preds, AGG_FLOOR)
    
    return {"region": req.region, "period": lastp, "predictions": preds}

@app.post("/forecast_quantiles_region")
def forecast_quantiles_region(req: ForecastRequest, ok: bool = Depends(require_api_key)):
    """Return per-admin quantile predictions."""
    df_hist = load_region_panel(req.region)
    df_aug = add_hawkes_trigger(df_hist)
    lastp = df_aug["period"].max()
    base = df_aug[df_aug["period"] == lastp].copy()
    
    # Build features
    from api.models.gravity import FEATURES as GFEATS
    feats = list(GFEATS)
    for extra in ["trigger", "lag1_flow"]:
        if extra in base.columns:
            feats.append(extra)
    
    available_feats = [f for f in feats if f in base.columns]
    X = base[available_feats].values
    
    # Load quantile models
    qmods = get_region_quantile_models(req.region)
    
    # Predict quantiles
    def predict_q(model, default):
        if model is None:
            return default
        yq = model.predict(X)
        return np.expm1(yq)
    
    # Fallback to gravity if no quantile models
    gpipe, _ = get_region_models(req.region)
    if gpipe is None:
        raise HTTPException(status_code=404, detail=f"No models found for region: {req.region}")
    
    g = GravityModel(pipe=gpipe)
    y50_default = g.predict(base)
    
    y10 = predict_q(qmods.get("p10"), np.maximum(0, y50_default * 0.7))
    y50 = predict_q(qmods.get("p50"), y50_default)
    y90 = predict_q(qmods.get("p90"), y50_default * 1.3)
    
    base["p10"] = y10
    base["p50"] = y50
    base["p90"] = y90
    
    agg = base.groupby("origin_id", as_index=False)[["p10", "p50", "p90"]].sum()
    preds = [
        {"admin_id": r.origin_id, "p10": float(r.p10), "p50": float(r.p50), "p90": float(r.p90)}
        for r in agg.itertuples()
    ]
    preds = apply_aggregation_floor(preds, AGG_FLOOR)
    
    return {"region": req.region, "period": lastp, "predictions": preds}

@app.post("/forecast_scenarios_region")
def forecast_scenarios_region(req: ScenarioRequest, ok: bool = Depends(require_api_key)):
    """Run scenario projections with quantile bands."""
    df_hist = load_region_panel(req.region)
    df_aug = add_hawkes_trigger(df_hist)
    lastp = df_aug["period"].max()
    last_slice = df_aug[df_aug["period"] == lastp].copy()
    
    # Apply scenario
    prior = req.priors if req.priors else DEFAULT_PRIORS.get(req.scenario, DEFAULT_PRIORS["baseline"])
    future = apply_scenario(last_slice, prior, months=req.months)
    
    # Build features
    from api.models.gravity import FEATURES as GFEATS
    feats = list(GFEATS)
    for extra in ["trigger", "lag1_flow"]:
        if extra in future.columns:
            feats.append(extra)
    
    available_feats = [f for f in feats if f in future.columns]
    X = future[available_feats].values
    
    # Get models
    qmods = get_region_quantile_models(req.region)
    gpipe, uplift = get_region_models(req.region)
    
    if gpipe is None:
        raise HTTPException(status_code=404, detail=f"No models found for region: {req.region}")
    
    # Predict with gravity baseline
    g = GravityModel(pipe=gpipe)
    y50_default = g.predict(future)
    
    if uplift is not None and "trigger" in future.columns and "lag1_flow" in future.columns:
        Xup = future[["trigger", "lag1_flow"]].values
        resid_hat = uplift.predict(Xup)
        y50_default = np.expm1(np.log1p(y50_default) + resid_hat)
    
    # Get quantiles
    def predict_q(model, default):
        if model is None:
            return default
        yq = model.predict(X)
        return np.expm1(yq)
    
    y10 = predict_q(qmods.get("p10"), np.maximum(0, y50_default * 0.7))
    y50 = predict_q(qmods.get("p50"), y50_default)
    y90 = predict_q(qmods.get("p90"), y50_default * 1.3)
    
    future["p10"] = y10
    future["p50"] = y50
    future["p90"] = y90
    
    # Aggregate per horizon
    future["horizon"] = future.groupby(["origin_id"]).cumcount() + 1
    res = []
    
    for (h, p), g in future.groupby(["horizon", "period"]):
        agg = g.groupby("origin_id")[["p10", "p50", "p90"]].sum().reset_index()
        preds = [
            {"admin_id": r.origin_id, "p10": float(r.p10), "p50": float(r.p50), "p90": float(r.p90)}
            for r in agg.itertuples()
        ]
        preds = apply_aggregation_floor(preds, AGG_FLOOR)
        res.append({"period": p, "horizon": int(h), "predictions": preds})
    
    return {
        "region": req.region,
        "base_period": lastp,
        "scenario": req.scenario,
        "priors": prior,
        "results": res
    }

@app.post("/download_scenario_csv")
def download_scenario_csv(req: ScenarioRequest, ok: bool = Depends(require_api_key)):
    """Download scenario results as CSV."""
    data = forecast_scenarios_region(req, ok=True)
    
    rows = []
    for item in data["results"]:
        period = item["period"]
        horizon = item["horizon"]
        for r in item["predictions"]:
            rows.append([
                period, horizon, r["admin_id"],
                r.get("p10", 0.0), r.get("p50", 0.0), r.get("p90", 0.0)
            ])
    
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["period", "horizon", "admin_id", "p10", "p50", "p90"])
    w.writerows(rows)
    
    return Response(content=buf.getvalue(), media_type="text/csv",
                   headers={"Content-Disposition": f"attachment; filename=scenario_{req.scenario}_{req.region}.csv"})
