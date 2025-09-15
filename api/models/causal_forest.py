"""Causal inference for intervention impact assessment."""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

# Try to import causal inference libraries
try:
    from econml.dml import CausalForestDML
    from econml.dml import DML
    from econml.cate_interpreter import SingleTreeCateInterpreter
    HAS_ECONML = True
except ImportError:
    HAS_ECONML = False
    print("Warning: econml not installed. Install with: pip install econml")

try:
    import sklearn
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)

class CausalImpactEstimator:
    """Causal inference model for estimating intervention effects on migration."""
    
    def __init__(self, method: str = 'causal_forest', **kwargs):
        """
        Initialize causal impact estimator.
        
        Args:
            method: 'causal_forest' or 'dml'
            **kwargs: Model parameters
        """
        self.method = method
        self.model = None
        self.fitted = False
        
        if not HAS_ECONML:
            logger.warning("econml not available, using sklearn fallback")
            self._init_sklearn_fallback()
        else:
            self._init_econml_models(**kwargs)
    
    def _init_econml_models(self, **kwargs):
        """Initialize econml-based models."""
        if self.method == 'causal_forest':
            self.model = CausalForestDML(
                n_estimators=kwargs.get('n_estimators', 100),
                min_samples_leaf=kwargs.get('min_samples_leaf', 10),
                max_depth=kwargs.get('max_depth', None),
                random_state=kwargs.get('random_state', 42)
            )
        elif self.method == 'dml':
            self.model = DML(
                model_y=RandomForestRegressor(n_estimators=100),
                model_t=RandomForestRegressor(n_estimators=100),
                model_final=LinearRegression(),
                random_state=kwargs.get('random_state', 42)
            )
    
    def _init_sklearn_fallback(self):
        """Initialize sklearn-based fallback models."""
        if not HAS_SKLEARN:
            raise RuntimeError("Neither econml nor sklearn available")
        
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        logger.info("Using sklearn RandomForest as causal inference fallback")
    
    def estimate_intervention_effect(self, 
                                   df: pd.DataFrame, 
                                   treatment_col: str, 
                                   outcome_col: str,
                                   feature_cols: List[str] = None,
                                   confounders: List[str] = None) -> Dict:
        """
        Estimate causal effect of interventions on migration flows.
        
        Args:
            df: DataFrame with treatment, outcome, and feature columns
            treatment_col: Name of binary treatment variable
            outcome_col: Name of outcome variable (migration flow)
            feature_cols: List of feature column names
            confounders: List of confounder column names
            
        Returns:
            Dictionary with causal effect estimates
        """
        if feature_cols is None:
            # Default feature columns
            feature_cols = ['pop', 'climate', 'conflict', 'distance', 'access']
        
        # Filter available columns
        available_features = [col for col in feature_cols if col in df.columns]
        
        if not available_features:
            raise ValueError("No feature columns found in DataFrame")
        
        # Prepare data
        X = df[available_features].fillna(0).values
        T = df[treatment_col].values
        Y = df[outcome_col].values
        
        # Handle missing values
        valid_idx = ~(np.isnan(X).any(axis=1) | np.isnan(T) | np.isnan(Y))
        X, T, Y = X[valid_idx], T[valid_idx], Y[valid_idx]
        
        if len(X) == 0:
            raise ValueError("No valid data points after cleaning")
        
        logger.info(f"Training causal model with {len(X)} samples, {X.shape[1]} features")
        
        # Fit model
        if HAS_ECONML and self.method in ['causal_forest', 'dml']:
            self.model.fit(Y, T, X=X)
            self.fitted = True
            
            # Get treatment effects
            effects = self.model.effect(X)
            
            # Calculate confidence intervals
            try:
                effect_intervals = self.model.effect_interval(X, alpha=0.05)
                ci_lower = effect_intervals[0]
                ci_upper = effect_intervals[1]
            except:
                ci_lower = effects - 1.96 * np.std(effects)
                ci_upper = effects + 1.96 * np.std(effects)
            
            results = {
                'average_treatment_effect': np.mean(effects),
                'treatment_effects': effects,
                'heterogeneity': np.std(effects),
                'confidence_intervals': {
                    'lower': ci_lower,
                    'upper': ci_upper
                },
                'method': self.method,
                'n_samples': len(X),
                'n_features': X.shape[1]
            }
            
            # Feature importance for causal forest
            if self.method == 'causal_forest':
                try:
                    feature_importance = self.model.feature_importances_
                    results['feature_importance'] = dict(zip(available_features, feature_importance))
                except:
                    pass
            
        else:
            # Fallback to simple comparison
            treated = Y[T == 1]
            control = Y[T == 0]
            
            if len(treated) > 0 and len(control) > 0:
                ate = np.mean(treated) - np.mean(control)
                se = np.sqrt(np.var(treated) / len(treated) + np.var(control) / len(control))
                
                results = {
                    'average_treatment_effect': ate,
                    'standard_error': se,
                    'confidence_intervals': {
                        'lower': ate - 1.96 * se,
                        'upper': ate + 1.96 * se
                    },
                    'method': 'simple_comparison',
                    'n_treated': len(treated),
                    'n_control': len(control)
                }
            else:
                raise ValueError("No variation in treatment assignment")
        
        return results
    
    def estimate_heterogeneous_effects(self, df: pd.DataFrame, **kwargs) -> Dict:
        """Estimate heterogeneous treatment effects across subgroups."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        if not HAS_ECONML:
            logger.warning("Heterogeneous effects require econml")
            return {}
        
        # Define subgroups
        subgroups = {
            'high_conflict': df['conflict'] > df['conflict'].median(),
            'low_conflict': df['conflict'] <= df['conflict'].median(),
            'drought_affected': df.get('climate', 0) < -1.0,
            'high_population': df['pop'] > df['pop'].median()
        }
        
        results = {}
        for subgroup_name, mask in subgroups.items():
            if mask.sum() > 10:  # Minimum sample size
                subgroup_df = df[mask]
                try:
                    subgroup_results = self.estimate_intervention_effect(
                        subgroup_df, **kwargs
                    )
                    results[subgroup_name] = subgroup_results
                except Exception as e:
                    logger.warning(f"Failed to estimate effects for {subgroup_name}: {e}")
        
        return results
    
    def interpret_effects(self, feature_names: List[str] = None) -> Dict:
        """Interpret treatment effects using tree-based methods."""
        if not self.fitted or not HAS_ECONML:
            logger.warning("Effect interpretation requires fitted econml model")
            return {}
        
        try:
            # Create interpreter
            interpreter = SingleTreeCateInterpreter(
                include_model_uncertainty=True,
                max_depth=3,
                min_samples_leaf=10
            )
            
            # Fit interpreter
            interpreter.interpret(self.model, X=None, feature_names=feature_names)
            
            return {
                'tree_rules': interpreter.export(),
                'feature_importance': interpreter.feature_importances_
            }
            
        except Exception as e:
            logger.error(f"Failed to interpret effects: {e}")
            return {}
    
    def predict_counterfactual(self, 
                             df: pd.DataFrame,
                             treatment_col: str,
                             outcome_col: str,
                             feature_cols: List[str] = None,
                             treatment_value: float = 1.0) -> np.ndarray:
        """Predict counterfactual outcomes under different treatment scenarios."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        if feature_cols is None:
            feature_cols = ['pop', 'climate', 'conflict', 'distance', 'access']
        
        available_features = [col for col in feature_cols if col in df.columns]
        X = df[available_features].fillna(0).values
        
        if HAS_ECONML:
            # Use econml's predict method
            counterfactual = self.model.effect(X, T0=0, T1=treatment_value)
            return counterfactual
        else:
            # Simple prediction
            return self.model.predict(X)
    
    def validate_assumptions(self, df: pd.DataFrame, **kwargs) -> Dict:
        """Validate causal inference assumptions."""
        validation_results = {}
        
        # Check for overlap (common support)
        treatment_col = kwargs.get('treatment_col', 'treatment')
        if treatment_col in df.columns:
            treated = df[df[treatment_col] == 1]
            control = df[df[treatment_col] == 0]
            
            validation_results['overlap'] = {
                'n_treated': len(treated),
                'n_control': len(control),
                'balance_ratio': len(treated) / len(control) if len(control) > 0 else float('inf')
            }
        
        # Check for random assignment (if applicable)
        # This would require domain knowledge about the treatment assignment
        
        return validation_results

# Example usage and testing functions
def create_synthetic_intervention_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create synthetic data for testing causal inference."""
    np.random.seed(42)
    
    # Generate features
    pop = np.random.lognormal(10, 1, n_samples)
    climate = np.random.normal(0, 1, n_samples)
    conflict = np.random.exponential(1, n_samples)
    distance = np.random.uniform(1, 10, n_samples)
    access = np.random.uniform(0, 1, n_samples)
    
    # Generate treatment (e.g., border policy change)
    # Treatment is influenced by some features (violating random assignment)
    treatment_prob = 1 / (1 + np.exp(-(climate + conflict * 0.5 - access)))
    treatment = np.random.binomial(1, treatment_prob)
    
    # Generate outcome (migration flow) with treatment effect
    base_flow = (np.log(pop) * 0.5 + 
                climate * 0.3 + 
                conflict * 0.4 - 
                distance * 0.2 + 
                access * 0.3)
    
    # True treatment effect is heterogeneous
    treatment_effect = 0.5 + climate * 0.2 + conflict * 0.1
    outcome = base_flow + treatment * treatment_effect + np.random.normal(0, 0.1, n_samples)
    
    return pd.DataFrame({
        'pop': pop,
        'climate': climate,
        'conflict': conflict,
        'distance': distance,
        'access': access,
        'treatment': treatment,
        'migration_flow': outcome
    })

if __name__ == "__main__":
    # Test causal inference
    print("Testing causal inference models...")
    
    # Create synthetic data
    df = create_synthetic_intervention_data(1000)
    print(f"Created synthetic dataset with {len(df)} samples")
    
    # Test causal forest
    if HAS_ECONML:
        estimator = CausalImpactEstimator(method='causal_forest')
        results = estimator.estimate_intervention_effect(
            df, 
            treatment_col='treatment',
            outcome_col='migration_flow'
        )
        print(f"Causal Forest ATE: {results['average_treatment_effect']:.3f}")
        print(f"Heterogeneity: {results['heterogeneity']:.3f}")
    
    # Test fallback
    estimator_fallback = CausalImpactEstimator(method='simple')
    results_fallback = estimator_fallback.estimate_intervention_effect(
        df,
        treatment_col='treatment', 
        outcome_col='migration_flow'
    )
    print(f"Simple comparison ATE: {results_fallback['average_treatment_effect']:.3f}")
