"""Authentication and security utilities."""
import os
from fastapi import Header, HTTPException

def require_api_key(x_api_key: str | None = Header(default=None)):
    """Require API key if configured."""
    expected = os.environ.get("API_KEY")
    if expected:
        if x_api_key != expected:
            raise HTTPException(status_code=401, detail="Unauthorized")
    return True

def apply_aggregation_floor(records, floor: float):
    """Apply aggregation floor to predictions for privacy."""
    out = []
    for r in records:
        r = dict(r)
        for k in ("yhat", "p10", "p50", "p90"):
            if k in r and r[k] is not None:
                r[k] = 0.0 if r[k] < floor else r[k]
        out.append(r)
    return out
