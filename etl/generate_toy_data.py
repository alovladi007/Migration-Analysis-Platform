#!/usr/bin/env python3
"""Generate synthetic O-D flows and features for testing."""

import numpy as np
import pandas as pd
import pathlib

BASE = pathlib.Path(__file__).resolve().parents[1] / "data" / "toy"
BASE.mkdir(parents=True, exist_ok=True)

# Configuration
nodes = list("ABCDE")  # 5 toy admin units
periods = pd.period_range("2020-01", "2020-12", freq="M").astype(str)
rng = np.random.default_rng(42)

# Generate node features
pop = {n: rng.integers(50_000, 500_000) for n in nodes}
rows_feat = []
for p in periods:
    for n in nodes:
        spi3 = rng.normal(0, 1)
        tmax = rng.normal(0, 1) 
        intensity = max(0, rng.normal(0.4 if n in ["C","D"] else 0.1, 0.3))
        access = rng.uniform(0.2, 0.9)
        rows_feat.append({
            "period": p, 
            "admin_id": n, 
            "pop": pop[n],
            "chirps_spi3": spi3, 
            "era5_tmax_anom": tmax,
            "acled_intensity": intensity,
            "access_score": access
        })

nf = pd.DataFrame(rows_feat)
nf.to_csv(BASE / "node_features.csv", index=False)

# Generate O-D flows using gravity-like model
coords = {n: (i, i) for i, n in enumerate(nodes)}

def dist(i, j):
    (x1, y1), (x2, y2) = coords[i], coords[j]
    return abs(x1-x2) + abs(y1-y2) + 1

rows_flow = []
for p in periods:
    for o in nodes:
        for d in nodes:
            if o == d:
                continue
            fo = nf[(nf["period"]==p) & (nf["admin_id"]==o)].iloc[0]
            fd = nf[(nf["period"]==p) & (nf["admin_id"]==d)].iloc[0]
            
            # Gravity components
            mass = np.log1p(fo["pop"]) + 0.3*np.log1p(fd["pop"])
            push = 0.8*fo["chirps_spi3"] + 1.3*fo["acled_intensity"]
            pull = 0.4*fd["access_score"]
            friction = 1.2*np.log(dist(o, d))
            
            mean_flow = np.exp(-friction + 0.00002*mass + push + pull)
            flow = np.maximum(0, rng.poisson(lam=max(0.1, mean_flow)) * rng.uniform(0.8, 1.2))
            
            rows_flow.append({
                "period": p,
                "origin_id": o,
                "dest_id": d,
                "flow": float(flow),
                "flow_type": "toy",
                "source": "toygen"
            })

pd.DataFrame(rows_flow).to_csv(BASE / "od_flows.csv", index=False)

# Build panel
od = pd.read_csv(BASE / "od_flows.csv")
panel = od.merge(nf.add_suffix("_o"), left_on=["period","origin_id"], right_on=["period_o","admin_id_o"], how="left") \
          .merge(nf.add_suffix("_d"), left_on=["period","dest_id"], right_on=["period_d","admin_id_d"], how="left")

keep = ["period", "origin_id", "dest_id", "flow", "flow_type", "source",
        "pop_o", "chirps_spi3_o", "era5_tmax_anom_o", "acled_intensity_o", "access_score_o",
        "pop_d", "chirps_spi3_d", "era5_tmax_anom_d", "acled_intensity_d", "access_score_d"]
panel = panel[keep]
panel.to_csv(BASE / "model_panel.csv", index=False)

print(f"✓ Generated toy data in {BASE}")
print(f"  - node_features.csv: {len(nf)} rows")
print(f"  - od_flows.csv: {len(rows_flow)} rows")
print(f"  - model_panel.csv: {len(panel)} rows")
