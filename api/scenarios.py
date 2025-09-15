"""Scenario engine for future projections."""
import numpy as np
import pandas as pd

DEFAULT_PRIORS = {
    "baseline": {"chirps_spi3": 0.0, "era5_tmax_anom": 0.0, "acled_intensity": 0.0},
    "drought": {"chirps_spi3": -1.0, "era5_tmax_anom": 0.2, "acled_intensity": 0.0},
    "heatwave": {"chirps_spi3": 0.0, "era5_tmax_anom": 0.8, "acled_intensity": 0.0},
    "conflict_spike": {"chirps_spi3": 0.0, "era5_tmax_anom": 0.0, "acled_intensity": 2.0}
}

def apply_scenario(last_slice: pd.DataFrame, prior: dict, months: int = 3):
    """Generate future panels by applying scenario priors."""
    base = last_slice.copy()
    base["t"] = pd.PeriodIndex(base["period"], freq="M").to_timestamp(how="end")
    start = base["t"].max()
    
    frames = []
    for k in range(1, months + 1):
        step = base.copy()
        step["t"] = start + pd.offsets.MonthEnd(k)
        step["period"] = step["t"].dt.to_period("M").astype(str)
        
        # Apply scenario deltas
        for name, delta in prior.items():
            if name == "chirps_spi3":
                if "chirps_spi3_o" in step.columns:
                    step["chirps_spi3_o"] = step["chirps_spi3_o"] + delta
                if "chirps_spi3_d" in step.columns:
                    step["chirps_spi3_d"] = step["chirps_spi3_d"] + delta * 0.2
            
            if name == "era5_tmax_anom":
                if "era5_tmax_anom_o" in step.columns:
                    step["era5_tmax_anom_o"] = step["era5_tmax_anom_o"] + delta
                if "era5_tmax_anom_d" in step.columns:
                    step["era5_tmax_anom_d"] = step["era5_tmax_anom_d"] + delta * 0.2
            
            if name == "acled_intensity":
                if "acled_intensity_o" in step.columns:
                    step["acled_intensity_o"] = step["acled_intensity_o"] + delta
                if "acled_intensity_d" in step.columns:
                    step["acled_intensity_d"] = step["acled_intensity_d"] + delta * 0.1
        
        # Momentum: carry forward lag1_flow
        if "lag1_flow" in step.columns:
            step["lag1_flow"] = step["flow"]
        
        frames.append(step.drop(columns=["t"]))
    
    return pd.concat(frames, ignore_index=True)
