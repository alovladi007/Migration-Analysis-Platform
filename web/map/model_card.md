# Model Card — MAP (Migration Analysis Platform)

## Model Details
- **Platform**: MAP v0.1.1
- **Full Name**: Migration Analysis Platform
- **Purpose**: Multi-region migration flow prediction
- **Type**: Ensemble (Gravity + Hawkes triggers + Quantile regression)
- **Task**: Monthly O-D flow prediction with uncertainty quantification
- **Output**: Per-admin P10/P50/P90 predicted outflows

## Intended Use
- **Primary**: Early warning and planning support for humanitarian response
- **Users**: Government agencies, NGOs, researchers
- **Scope**: Admin-1 level aggregated predictions (no individual tracking)

## Training Data
- **UNHCR**: Refugee stocks/flows (2018-2024)
- **ACLED**: Conflict events with spatial join to admin units
- **Climate**: CHIRPS SPI-3, ERA5 temperature anomalies
- **Population**: WorldPop admin aggregates
- **Coverage**: Horn of Africa, Sahel, South Asia, Central America/Caribbean

## Model Components

### Gravity Baseline
- Linear regression on log-transformed features
- Features: log population, distance, climate, access
- Captures fundamental push-pull dynamics

### Hawkes Trigger
- Exponential memory kernel (α=0.6)
- Models self-excitation from shocks
- Includes lag-1 flow momentum

### Quantile Models
- Gradient Boosting (300 trees, depth 3)
- Trained for P10, P50, P90 with pinball loss
- Captures heteroscedastic uncertainty

### Temporal GNN (Optional)
- GRU encoder with quantile heads
- Sequence-based predictions
- Multi-horizon forecasting

## Performance Metrics
- MAE on log(flow+1): ~0.8
- Hotspot hit rate (top 20%): ~65%
- Calibration: P10-P90 contains true value ~78% of time

## Limitations
- **Data gaps**: Reporting delays, incomplete coverage
- **Admin boundaries**: Inconsistencies across countries
- **Climate fallback**: Synthetic when rasters unavailable
- **Scenario simplicity**: Linear perturbations only
- **No causality**: Correlational patterns, not causal

## Ethical Considerations
- **Privacy**: Aggregation floor (default 5) prevents individual tracking
- **Fairness**: May underpredict for underreported populations
- **Misuse potential**: Not for enforcement or punitive actions
- **Transparency**: Open methodology, uncertainty always shown

## Uncertainty Quantification
- Always report P10-P90 bands, not just point estimates
- Scenarios are conditional projections, not probabilities
- Wider bands indicate higher uncertainty
- Validate with local knowledge

## Recommendations
1. Use for planning, not deterministic decisions
2. Combine with qualitative assessments
3. Update regularly with new data
4. Monitor for distribution shift
5. Engage affected communities

## Citation
MAP - Migration Analysis Platform v0.1.1 (2024)
Open source under MIT License
