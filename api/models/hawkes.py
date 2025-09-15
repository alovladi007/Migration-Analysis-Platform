"""Hawkes-style trigger model for shock propagation."""
import numpy as np
import pandas as pd

def exp_kernel_decay(series: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Compute exponentially weighted sum for self-excitation memory."""
    vals = []
    prev = 0.0
    for x in series:
        prev = alpha * prev + x
        vals.append(prev)
    return np.array(vals)

def add_hawkes_trigger(panel: pd.DataFrame) -> pd.DataFrame:
    """Add trigger and lag features to panel."""
    df = panel.copy()
    
    # Normalize period to timestamp for sorting
    df["period"] = pd.PeriodIndex(df["period"], freq="M").to_timestamp(how="end")
    df = df.sort_values(["origin_id", "dest_id", "period"])
    
    # Create trigger from access_score (or acled_intensity if available)
    trigger_col = "acled_intensity_o" if "acled_intensity_o" in df.columns else "access_score_o"
    
    # Compute trigger per origin
    origin_period = df.groupby(["origin_id", "period"], as_index=False)[trigger_col].mean()
    parts = []
    for origin, grp in origin_period.groupby("origin_id"):
        trig = exp_kernel_decay(grp[trigger_col].values, alpha=0.6)
        g = grp.copy()
        g["trigger"] = trig
        parts.append(g)
    
    if parts:
        op = pd.concat(parts, ignore_index=True)[["origin_id", "period", "trigger"]]
        df = df.merge(op, on=["origin_id", "period"], how="left")
    else:
        df["trigger"] = 0.0
    
    # Add lagged flow
    df["lag1_flow"] = df.groupby(["origin_id", "dest_id"])["flow"].shift(1).fillna(0.0)
    
    return df
