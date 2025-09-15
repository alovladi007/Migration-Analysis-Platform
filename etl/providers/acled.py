"""ACLED conflict events fetcher."""
import requests
import pandas as pd
from .common import get_env

def fetch_acled_events(iso3_list, date_from="2018-01-01", date_to="2025-12-31", page_size=5000):
    """Fetch ACLED conflict events for given countries."""
    email = get_env("ACLED_EMAIL", required=False)
    key = get_env("ACLED_KEY", required=False)
    base = "https://api.acleddata.com/acled/read"
    
    frames = []
    for iso3 in iso3_list:
        page = 1
        while True:
            params = {
                "country_iso": iso3,
                "event_date": f"{date_from}|{date_to}",
                "limit": page_size,
                "page": page
            }
            if email:
                params["email"] = email
            if key:
                params["key"] = key
            
            try:
                r = requests.get(base, params=params, timeout=60)
                r.raise_for_status()
                js = r.json()
                data = js.get("data", [])
                if not data:
                    break
                frames.append(pd.DataFrame(data))
                if len(data) < page_size:
                    break
                page += 1
            except Exception as e:
                print(f"ACLED fetch error for {iso3}: {e}")
                break
    
    if frames:
        df = pd.concat(frames, ignore_index=True)
        if "event_date" in df.columns:
            df["event_date"] = pd.to_datetime(df["event_date"])
        return df
    return pd.DataFrame()

def monthly_intensity(df: pd.DataFrame, admin_map=None):
    """Aggregate events to monthly intensity per admin."""
    if df.empty:
        return pd.DataFrame(columns=["period","admin_id","acled_intensity"])
    
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["period"] = df["event_date"].dt.to_period("M").astype(str)
    
    key = "country_iso" if "country_iso" in df.columns else "country"
    df[key] = df[key].fillna("UNK")
    
    agg = df.groupby(["period", key], as_index=False).size()
    agg = agg.rename(columns={"size": "acled_intensity", key: "admin_id"})
    return agg
