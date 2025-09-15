#!/usr/bin/env python3
"""Compute climate features for Horn of Africa."""
import pathlib
import pandas as pd
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from etl.providers.chirps_spi import compute_spi3, compute_era5_tmax_anom

ROOT = pathlib.Path(__file__).resolve().parents[2]
raw_dir = ROOT / "etl" / "horn_of_africa" / "raw"

admin_geojson = raw_dir / "admin1.geojson"
chirps_nc = raw_dir / "CHIRPS_monthly.nc"
era5_nc = raw_dir / "ERA5_tmax_monthly.nc"

print("[CLIMATE:horn_of_africa] Computing CHIRPS SPI-3...")
spi = compute_spi3(str(chirps_nc), str(admin_geojson))

print("[CLIMATE:horn_of_africa] Computing ERA5 Tmax anomalies...")
tmax = compute_era5_tmax_anom(str(era5_nc), str(admin_geojson))

# Merge into node_features.csv
nf_path = raw_dir / "node_features.csv"
if not nf_path.exists():
    raise SystemExit("node_features.csv not found; run fetch_all.py first")

nf = pd.read_csv(nf_path)

# Merge climate features
df = nf.merge(spi, on=["period", "admin_id"], how="left") \
       .merge(tmax, on=["period", "admin_id"], how="left")

# Handle duplicate columns
if "chirps_spi3_x" in df.columns:
    df["chirps_spi3"] = df["chirps_spi3_y"].fillna(df["chirps_spi3_x"])
    df.drop(columns=[c for c in df.columns if c.startswith("chirps_spi3_")], inplace=True)

if "era5_tmax_anom_x" in df.columns:
    df["era5_tmax_anom"] = df["era5_tmax_anom_y"].fillna(df["era5_tmax_anom_x"])
    df.drop(columns=[c for c in df.columns if c.startswith("era5_tmax_anom_")], inplace=True)

df.to_csv(nf_path, index=False)
print(f"✓ Updated {nf_path} with climate features")
