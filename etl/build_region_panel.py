#!/usr/bin/env python3
"""Build model panel for a region by joining flows and features."""
import sys
import pathlib
import pandas as pd
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
region = sys.argv[1] if len(sys.argv) > 1 else "toy"

raw_dir = ROOT / "etl" / region / "raw"
out_dir = ROOT / "data" / region
out_dir.mkdir(parents=True, exist_ok=True)

flows_path = raw_dir / "od_flows.csv"
features_path = raw_dir / "node_features.csv"

# Check if raw files exist
if not flows_path.exists() or not features_path.exists():
    print(f"[{region}] Raw files not found, falling back to toy panel")
    toy = ROOT / "data" / "toy" / "model_panel.csv"
    if toy.exists():
        pd.read_csv(toy).to_csv(out_dir / "model_panel.csv", index=False)
        print(f"✓ Copied toy panel to {out_dir / 'model_panel.csv'}")
    else:
        print(f"✗ No toy panel found. Run: python etl/generate_toy_data.py")
    sys.exit(0)

print(f"Building panel for region: {region}")

# Load data
od = pd.read_csv(flows_path)
nf = pd.read_csv(features_path)

print(f"  - Flows: {len(od)} rows")
print(f"  - Features: {len(nf)} rows")

# Join with suffixes
panel = od.merge(
    nf.add_suffix("_o"),
    left_on=["period", "origin_id"],
    right_on=["period_o", "admin_id_o"],
    how="left"
).merge(
    nf.add_suffix("_d"),
    left_on=["period", "dest_id"],
    right_on=["period_d", "admin_id_d"],
    how="left"
)

# Select columns
keep_cols = [
    "period", "origin_id", "dest_id", "flow", "flow_type", "source",
    "pop_o", "chirps_spi3_o", "era5_tmax_anom_o", "access_score_o",
    "pop_d", "chirps_spi3_d", "era5_tmax_anom_d", "access_score_d"
]

# Add optional columns if they exist
for col in ["acled_intensity_o", "acled_intensity_d"]:
    if col in panel.columns:
        keep_cols.append(col)

panel = panel[keep_cols]

# Save panel
panel.to_csv(out_dir / "model_panel.csv", index=False)
print(f"✓ Wrote panel to {out_dir / 'model_panel.csv'} ({len(panel)} rows)")

# Copy admin GeoJSON to web if it exists
admin_geo_path = raw_dir / "admin1.geojson"
web_geo_out = ROOT / "web" / "map" / f"{region}_admin1.geojson"

if admin_geo_path.exists():
    (ROOT / "web" / "map").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(admin_geo_path, web_geo_out)
    print(f"✓ Copied admin polygons to {web_geo_out}")
