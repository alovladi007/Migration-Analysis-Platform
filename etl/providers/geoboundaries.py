"""GeoBoundaries admin polygon fetcher."""
import requests
import pandas as pd

try:
    import geopandas as gpd
    HAS_GEO = True
except ImportError:
    HAS_GEO = False

def fetch_admin1_for_countries(iso3_list):
    """Fetch Admin-1 boundaries for given countries."""
    if not HAS_GEO:
        print("Warning: geopandas not installed. Install requirements-geo.txt for geo support.")
        return pd.DataFrame(columns=["admin_id","name","geometry"])
    
    frames = []
    for iso in iso3_list:
        try:
            url = f"https://www.geoboundaries.org/api/current/gbOpen/{iso}/ADM1"
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            js = r.json()
            
            if isinstance(js, list) and js:
                link = js[0]["gjDownloadURL"]
                gdf = gpd.read_file(link)
                
                # Find ID and name columns
                idcol = next((c for c in gdf.columns if c.lower().endswith("id")), "shapeID")
                namecol = next((c for c in gdf.columns if "name" in c.lower()), "shapeName")
                
                sub = gdf[[idcol, namecol, "geometry"]].rename(columns={
                    idcol: "admin_id",
                    namecol: "name"
                })
                sub["admin_id"] = sub["admin_id"].astype(str)
                frames.append(sub)
        except Exception as e:
            print(f"GeoBoundaries fetch error for {iso}: {e}")
    
    if frames and HAS_GEO:
        out = pd.concat(frames, ignore_index=True)
        out.crs = "EPSG:4326"
        return out
    
    if HAS_GEO:
        return gpd.GeoDataFrame(columns=["admin_id","name","geometry"], geometry="geometry", crs="EPSG:4326")
    return pd.DataFrame(columns=["admin_id","name","geometry"])
