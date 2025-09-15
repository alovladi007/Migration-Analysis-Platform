# MAP - Migration Analysis Platform

End-to-end platform for analyzing and forecasting human & animal migration patterns driven by climate, disasters, and conflict.

## Features

### Core Analytics
- Historical pattern analysis from UNHCR, ACLED, CHIRPS, ERA5 data
- Multi-region support (Horn of Africa, Sahel, South Asia, Central America/Caribbean)
- Gravity baseline + Hawkes-style triggers + Quantile forecasts (P10/P50/P90)
- Scenario projections (drought, heatwave, conflict)
- Interactive choropleth maps with uncertainty visualization
- CSV export and API access control

### Advanced Capabilities
- **Real-time Streaming**: Kafka-based event processing with anomaly detection
- **Causal Inference**: Treatment effect estimation for policy interventions
- **LSTM Sequences**: Deep learning for long-term temporal predictions
- **Data Quality**: Automated validation with Great Expectations
- **Network Analysis**: Migration corridor identification and cascade simulation
- **Satellite Monitoring**: Google Earth Engine integration for displacement detection
- **Economic Indicators**: FRED API integration for inflation, food prices, currency rates
- **Social Media Sentiment**: Twitter analysis for early warning signals
- **Agent-Based Modeling**: Individual decision-making simulation with Mesa
- **Automated Reporting**: AI-generated insights with PDF/HTML export

## Quick Start

### 1. Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run with Toy Data
```bash
# Generate sample data
python etl/generate_toy_data.py

# Build panel
python etl/build_region_panel.py toy

# Train models
python scripts/mobility_cli.py train --region toy
python scripts/mobility_cli.py train_quantiles --region toy

# Start API
uvicorn api.main:app --reload --port 8000

# Open web/map/index.html in browser
```

### 3. Real Region Pipeline (Horn of Africa)
Install geo dependencies
```bash
pip install -r requirements-geo.txt
```

Configure credentials
```bash
cp .env.example .env
# Edit .env with your ACLED_EMAIL and ACLED_KEY
```

Fetch data
```bash
python etl/horn_of_africa/fetch_all.py
python etl/horn_of_africa/compute_climate.py
```

Build panel and train
```bash
python etl/build_region_panel.py horn_of_africa
python scripts/mobility_cli.py train --region horn_of_africa
python scripts/mobility_cli.py train_quantiles --region horn_of_africa
```

Serve and view
```bash
uvicorn api.main:app --reload --port 8000
# Open web/map/index.html → select "Horn of Africa"
```

## API Endpoints

Forecast current period
```bash
curl -X POST http://localhost:8000/forecast_quantiles_region \
  -H "content-type: application/json" \
  -d '{"region":"horn_of_africa","period_start":"2020-12","periods":1}'
```

Run scenarios
```bash
curl -X POST http://localhost:8000/forecast_scenarios_region \
  -H "content-type: application/json" \
  -d '{"region":"horn_of_africa","scenario":"drought","months":3}'
```

Download CSV
```bash
curl -X POST http://localhost:8000/download_scenario_csv \
  -H "content-type: application/json" \
  -d '{"region":"sahel","scenario":"heatwave","months":3}' -o scenario.csv
```

## Access Control
Set API key requirement:
```bash
export API_KEY="your-secret"
```

Set aggregation floor (default 5):
```bash
export AGGREGATION_FLOOR=10
```

## Multi-Region Support
Available regions:
- toy (demo data)
- horn_of_africa (ET, SO, DJ, ER, KE, SS)
- sahel (ML, NE, BF, TD, MR, NG)
- south_asia (IN, PK, BD, NP, LK, AF)
- central_america_caribbean (GT, SV, HN, NI, CR, PA, HT, DO, CU, JM, PR)

For each region:
```bash
make fetch-<region>
make climate-<region>
make panel-<region>
make train-<region>
```

## Advanced Models

### Machine Learning
```bash
# Core ML models
pip install -r requirements-ml.txt
python scripts/train_tgnn.py --region horn_of_africa --window 6 --horizons 3

# Advanced features
pip install -r requirements-advanced.txt
```

### Real-time Streaming
```bash
# Kafka streaming setup
pip install -r requirements-streaming.txt
python etl/streaming/kafka_ingestion.py
```

### Causal Inference
```python
from api.models.causal_forest import CausalImpactEstimator
estimator = CausalImpactEstimator(method='causal_forest')
results = estimator.estimate_intervention_effect(df, 'treatment', 'outcome')
```

### Network Analysis
```python
from api.models.migration_network import MigrationNetwork
network = MigrationNetwork(df)
cascade = network.predict_cascade_effects('shock_node', 0.5)
```

### Satellite Monitoring
```python
from etl.providers.satellite_monitor import DisplacementMonitor
monitor = DisplacementMonitor()
detections = monitor.detect_settlement_changes(bbox, start_date, end_date)
```

### Agent-Based Modeling
```python
from api.models.agent_simulation import run_migration_simulation
results = run_migration_simulation(n_agents=1000, n_steps=100, climate_scenario='drought')
```

### Automated Reporting
```python
from api.reports.generator import generate_weekly_migration_report
report = generate_weekly_migration_report('region', predictions, output_format='pdf')
```

## Climate Data
Place monthly NetCDF files in etl/<region>/raw/:
- CHIRPS_monthly.nc (precipitation)
- ERA5_tmax_monthly.nc (temperature)

Without rasters, the system uses deterministic synthetic series as fallback.

## Ethics & Safety
- Aggregation floors prevent individual-level predictions
- Model card available in UI explains limitations
- Role-based access via API keys
- No PII exposure; admin-level aggregation only

## Troubleshooting
- If ACLED fails: Check API credentials in .env
- If maps don't render: Ensure admin1.geojson was copied to web/map/
- For CUDA/GPU support: Install PyTorch with appropriate CUDA version
