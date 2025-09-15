"""WorldPop population data utilities."""
import pandas as pd
import numpy as np

def admin_population_stub(admin_gdf):
    """Generate population estimates per admin (stub implementation)."""
    if admin_gdf is None or len(admin_gdf) == 0:
        return pd.DataFrame(columns=["admin_id", "pop"])
    
    # Deterministic population based on admin_id hash
    pops = []
    for admin_id in admin_gdf["admin_id"].astype(str):
        seed = int.from_bytes(admin_id.encode("utf-8"), "little") % 1000
        base_pop = 100000 + seed * 1000
        pops.append({"admin_id": admin_id, "pop": base_pop})
    
    return pd.DataFrame(pops)
