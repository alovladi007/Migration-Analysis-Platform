"""Gravity model for baseline flow prediction."""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump, load
from typing import List

FEATURES = [
    "log_pop_o", "log_pop_d", "chirps_spi3_o", "era5_tmax_anom_o", "access_score_o",
    "chirps_spi3_d", "era5_tmax_anom_d", "access_score_d", "log_distance"
]

def _toy_distance(o: str, d: str) -> float:
    """Calculate toy distance between admin units."""
    # For toy data or when no real coordinates available
    order = {c: i for i, c in enumerate(list("ABCDE"))}
    if o in order and d in order:
        return abs(order[o] - order[d]) + 1
    # Default distance for unknown admins
    return 2.0

@dataclass
class GravityModel:
    """Gravity-based flow prediction model."""
    pipe: Pipeline
    
    @staticmethod
    def prepare(df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for the model."""
        df = df.copy()
        
        # Log-transform populations
        df["log_pop_o"] = np.log1p(df["pop_o"])
        df["log_pop_d"] = np.log1p(df["pop_d"])
        
        # Calculate distance (use toy distance for simplicity)
        df["log_distance"] = df.apply(
            lambda r: np.log(_toy_distance(r["origin_id"], r["dest_id"])), 
            axis=1
        )
        
        # Target in log space
        df["y"] = np.log1p(df["flow"])
        
        # Fill missing values
        for col in FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)
        
        return df
    
    @classmethod
    def train(cls, df: pd.DataFrame):
        """Train the gravity model."""
        dfp = cls.prepare(df)
        
        # Filter to features that exist
        available_features = [f for f in FEATURES if f in dfp.columns]
        
        X = dfp[available_features].values
        y = dfp["y"].values
        
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lm", LinearRegression())
        ])
        pipe.fit(X, y)
        
        return cls(pipe=pipe)
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict flows in original scale."""
        dfp = self.prepare(df)
        
        # Use same features as training
        available_features = [f for f in FEATURES if f in dfp.columns]
        X = dfp[available_features].values
        
        yhat = self.pipe.predict(X)
        return np.expm1(yhat)  # Back to original scale
    
    def save(self, path: str):
        """Save model to disk."""
        dump(self.pipe, path)
    
    @classmethod
    def load(cls, path: str):
        """Load model from disk."""
        return cls(pipe=load(path))
