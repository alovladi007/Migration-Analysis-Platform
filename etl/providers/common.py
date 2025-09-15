"""Common utilities for ETL providers."""
import os
from dotenv import load_dotenv

load_dotenv()

def get_env(name: str, default=None, required=False):
    """Get environment variable with optional requirement check."""
    v = os.environ.get(name, default)
    if required and (v is None or v == ""):
        raise RuntimeError(f"Missing required env var: {name}")
    return v
