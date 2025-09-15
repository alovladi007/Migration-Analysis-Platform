# MAP - Migration Analysis Platform
# Makefile for data processing, training, and deployment

.PHONY: help toy fetch-all train-all serve clean

help:
	@echo "MAP - Migration Analysis Platform"
	@echo "================================"
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  toy                 Generate toy data and train"
	@echo "  fetch-<region>      Fetch data for region"
	@echo "  climate-<region>    Compute climate features"
	@echo "  panel-<region>      Build model panel"
	@echo "  train-<region>      Train models"
	@echo "  serve              Start API server"
	@echo "  clean              Remove generated files"

toy:
	python etl/generate_toy_data.py
	python etl/build_region_panel.py toy
	python scripts/mobility_cli.py train --region toy

fetch-horn:
	python etl/horn_of_africa/fetch_all.py

climate-horn:
	python etl/horn_of_africa/compute_climate.py

panel-horn:
	python etl/build_region_panel.py horn_of_africa

train-horn:
	python scripts/mobility_cli.py train --region horn_of_africa
	python scripts/mobility_cli.py train_quantiles --region horn_of_africa

fetch-sahel:
	python etl/sahel/fetch_all.py

climate-sahel:
	python etl/sahel/compute_climate.py

panel-sahel:
	python etl/build_region_panel.py sahel

train-sahel:
	python scripts/mobility_cli.py train --region sahel
	python scripts/mobility_cli.py train_quantiles --region sahel

fetch-southasia:
	python etl/south_asia/fetch_all.py

climate-southasia:
	python etl/south_asia/compute_climate.py

panel-southasia:
	python etl/build_region_panel.py south_asia

train-southasia:
	python scripts/mobility_cli.py train --region south_asia

fetch-cac:
	python etl/central_america_caribbean/fetch_all.py

climate-cac:
	python etl/central_america_caribbean/compute_climate.py

panel-cac:
	python etl/build_region_panel.py central_america_caribbean

train-cac:
	python scripts/mobility_cli.py train --region central_america_caribbean

serve:
	uvicorn api.main:app --reload --port 8000

clean:
	rm -rf data/*/
	rm -rf models/*/
	rm -rf __pycache__ */__pycache__ */*/__pycache__
