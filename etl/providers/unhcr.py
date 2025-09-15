"""UNHCR refugee data fetcher."""
import requests
import pandas as pd
from .common import get_env

BASE = get_env("UNHCR_BASE", "https://api.unhcr.org/population/v1")

def fetch_refugee_stocks(origin_iso3=None, asylum_iso3=None, year_from=2018, year_to=2024):
    """Fetch refugee stock data from UNHCR API."""
    params = {
        "population_group": "REF",
        "yearFrom": year_from,
        "yearTo": year_to
    }
    if origin_iso3:
        params["coo"] = origin_iso3  # country of origin
    if asylum_iso3:
        params["coa"] = asylum_iso3  # country of asylum
    
    try:
        url = f"{BASE}/population"
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        js = r.json()
        df = pd.json_normalize(js.get("data", js)).rename(columns=str.lower)
        return df
    except Exception as e:
        print(f"UNHCR fetch error: {e}")
        return pd.DataFrame()

def to_od_monthly(df: pd.DataFrame, period_col="year"):
    """Convert yearly stocks to monthly O-D flows."""
    if df.empty:
        return pd.DataFrame(columns=["period","origin_id","dest_id","flow","flow_type","source"])
    
    out = []
    for _, r in df.iterrows():
        try:
            y = int(r.get(period_col, 0) or 0)
            origin = r.get("coo", r.get("origin"))
            dest = r.get("coa", r.get("asylum"))
            val = float(r.get("value", r.get("refugees", 0)) or 0)
            
            if not origin or not dest or not y:
                continue
                
            for m in range(1, 13):
                out.append({
                    "period": f"{y:04d}-{m:02d}",
                    "origin_id": str(origin),
                    "dest_id": str(dest),
                    "flow": val / 12.0,
                    "flow_type": "refugee_stock",
                    "source": "UNHCR"
                })
        except Exception:
            continue
    
    return pd.DataFrame(out)
