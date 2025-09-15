#!/usr/bin/env python3
"""Fetch all data for Sahel region."""
import pathlib
import pandas as pd
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from etl.providers.unhcr import fetch_refugee_stocks, to_od_monthly
from etl.providers.acled import fetch_acled_events, monthly_intensity
from etl.providers.geoboundaries import fetch_admin1_for_countries
from etl.providers.worldpop import admin_population_stub

ROOT = pathlib.Path(__file__).resolve().parents[2]
raw_dir = ROOT / "etl" / "sahel" / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)

iso3 = ["MLI", "NER", "BFA", "TCD", "MRT", "NGA"]

print("[ETL:sahel] Fetching admin1 boundaries...")
try:
    gdf = fetch_admin1_for_countries([c[:3] for c in iso3])
    if len(gdf) > 0:
        gdf.to_file(raw_dir / "admin1.geojson", driver="GeoJSON")
        print(f"  ✓ Saved {len(gdf)} admin units")
except Exception as e:
    print(f"  ✗ GeoBoundaries fetch failed: {e}")

print("[ETL:sahel] Fetching UNHCR refugee stocks...")
frames = []
for o in iso3:
    for d in iso3:
        if o == d:
            continue
        try:
            df = fetch_refugee_stocks(origin_iso3=o, asylum_iso3=d, year_from=2018, year_to=2024)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"  ✗ UNHCR fetch failed for {o}->{d}: {e}")

if frames:
    unhcr = pd.concat(frames, ignore_index=True)
    od = to_od_monthly(unhcr, period_col="year")
else:
    od = pd.DataFrame(columns=["period", "origin_id", "dest_id", "flow", "flow_type", "source"])

od.to_csv(raw_dir / "od_flows.csv", index=False)
print(f"  ✓ Saved {len(od)} O-D flows")

print("[ETL:sahel] Fetching ACLED events...")
ac = fetch_acled_events([c[:3] for c in iso3], date_from="2018-01-01")

# Spatial join with admin polygons if available
try:
    import geopandas as gpd
    gdf_admin = gpd.read_file(raw_dir / "admin1.geojson")
    
    if not ac.empty and "longitude" in ac.columns and "latitude" in ac.columns:
        pts = gpd.GeoDataFrame(
            ac.copy(),
            geometry=gpd.points_from_xy(
                pd.to_numeric(ac["longitude"], errors="coerce"),
                pd.to_numeric(ac["latitude"], errors="coerce")
            ),
            crs="EPSG:4326"
        )
        
        # Remove invalid geometries
        pts = pts[pts.geometry.is_valid]
        
        joined = gpd.sjoin(pts, gdf_admin[["admin_id", "geometry"]], predicate="intersects", how="left")
        joined["period"] = pd.to_datetime(joined["event_date"]).dt.to_period("M").astype(str)
        ac_month = joined.groupby(["period", "admin_id"], as_index=False).size()
        ac_month = ac_month.rename(columns={"size": "acled_intensity"})
    else:
        ac_month = monthly_intensity(ac)
except Exception as e:
    print(f"  ✗ Admin spatial join failed: {e}")
    ac_month = monthly_intensity(ac)

ac_month.to_csv(raw_dir / "acled_monthly.csv", index=False)
print(f"  ✓ Saved {len(ac_month)} monthly intensity records")

print("[ETL:sahel] Building node features...")
try:
    import geopandas as gpd
    gdf = gpd.read_file(raw_dir / "admin1.geojson")
    pop = admin_population_stub(gdf)
    
    periods = pd.period_range("2018-01", "2024-12", freq="M").astype(str)
    rows = []
    
    for p in periods:
        for admin_id in pop["admin_id"].astype(str):
            rows.append({
                "period": p,
                "admin_id": admin_id,
                "pop": float(pop[pop["admin_id"] == admin_id]["pop"].iloc[0]),
                "chirps_spi3": 0.0,
                "era5_tmax_anom": 0.0,
                "acled_intensity": 0.0,
                "access_score": 0.5
            })
    
    nf = pd.DataFrame(rows)
except Exception:
    nf = pd.DataFrame(columns=["period", "admin_id", "pop", "chirps_spi3", "era5_tmax_anom", "acled_intensity", "access_score"])

# Merge ACLED intensity
acp = raw_dir / "acled_monthly.csv"
if acp.exists():
    acm = pd.read_csv(acp)
    nf = nf.merge(acm, on=["period", "admin_id"], how="left")
    
    # Handle duplicate columns
    if "acled_intensity_y" in nf.columns:
        nf["acled_intensity"] = nf["acled_intensity_y"].fillna(nf["acled_intensity_x"])
        nf = nf.drop(columns=[c for c in nf.columns if c.endswith("_x") or c.endswith("_y")])
    
    nf["acled_intensity"] = nf["acled_intensity"].fillna(0.0)

nf.to_csv(raw_dir / "node_features.csv", index=False)
print(f"  ✓ Saved {len(nf)} node features")
print("✓ ETL complete for sahel")
