# RaeLM — Reliable Agentic Extraction with Language Models

RaeLM is a medical document understanding platform that integrates the olmOCR-2-7B-MLX model for reliable OCR, schema mapping, and confidence-scored outputs with optional human review.

## Overview
- End-to-end pipeline covering ingestion, preprocessing, layout detection, inference, validation, and structured export.
- Human-in-the-loop tooling to triage low-confidence predictions and capture corrections.
- Docker Compose stack providing API services, background workers, monitoring, and annotation tools.

## Getting Started
1. Install dependencies: `make setup`
2. Start local infrastructure: `make start-infra`
3. Launch the demo workflow: `./scripts/run_demo.sh`

Service endpoints:
- Demo UI: http://localhost:8501
- API Docs: http://localhost:8000/docs
- MLflow: http://localhost:5000
- Grafana: http://localhost:3001

## Architecture Highlights
- FastAPI orchestrates preprocessing, Celery tasks, and response delivery.
- olmOCR-2-7B-MLX integration in `inference/olmocr_adapter.py` supports self-consistency sampling and confidence estimation.
- Validation and schema mapping modules apply medical domain rules prior to export.

## Development
- Run tests: `make test`
- Lint code: `make lint`
- Generate synthetic samples: `python scripts/synthetic_data_generator.py --count 100 --output datasets/raw/synthetic_v1`

## Key Directories
- `ml_platform/` model registry, datasets, calibration utilities
- `inference/` model adapters, prompt assets, routing logic
- `preprocessing/` image enhancement and layout detection components
- `api/` FastAPI interfaces and task coordination
- `ui/` Streamlit-based review and monitoring surfaces
- `docker/` Docker Compose stack definitions and configs

