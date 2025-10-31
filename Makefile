# RaeLM Makefile - Development workflows

.PHONY: help setup start stop restart test lint format clean install dev-install

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Initial setup (install deps, init DVC, start services)
	@echo "=== Setting up RaeLM development environment ==="
	pip install -e ".[dev]"
	pre-commit install || true
	dvc init || echo "DVC already initialized"
	@echo "=== Starting Docker services ==="
	docker-compose -f docker/docker-compose.yml up -d
	@echo "=== Waiting for services to be ready (30s) ==="
	sleep 30
	@echo "=== Services ready! ==="
	@echo "  - MLflow: http://localhost:5000"
	@echo "  - Label Studio: http://localhost:8080"
	@echo "  - Grafana: http://localhost:3000"
	@echo "  - Prometheus: http://localhost:9090"

install: ## Install package in editable mode
	pip install -e .

dev-install: ## Install package with dev dependencies
	pip install -e ".[dev]"

start: ## Start all Docker services
	docker-compose -f docker/docker-compose.yml up -d
	@echo "Services starting... Check status with 'make status'"

stop: ## Stop all Docker services
	docker-compose -f docker/docker-compose.yml down

restart: ## Restart all Docker services
	docker-compose -f docker/docker-compose.yml restart

status: ## Check Docker services status
	docker-compose -f docker/docker-compose.yml ps

logs: ## Show Docker services logs
	docker-compose -f docker/docker-compose.yml logs -f

test: ## Run test suite
	pytest tests/ -v --cov=ml_platform --cov=inference --cov=preprocessing --cov=validation

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	pytest tests/integration/ -v

test-regression: ## Run regression tests only
	pytest tests/regression/ -v

lint: ## Run linters
	ruff check .
	mypy ml_platform/ inference/ preprocessing/ validation/ || true

format: ## Format code
	black .
	ruff check --fix .

clean: ## Clean temporary files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov .ruff_cache
	rm -rf build dist *.egg-info

clean-all: clean stop ## Clean everything including Docker volumes
	docker-compose -f docker/docker-compose.yml down -v
	@echo "All clean! Run 'make setup' to start fresh."

db-shell: ## Open PostgreSQL shell
	docker exec -it raelm_postgres psql -U raelm -d raelm

redis-cli: ## Open Redis CLI
	docker exec -it raelm_redis redis-cli

# DVC commands
dvc-pull: ## Pull data from DVC remote
	dvc pull

dvc-push: ## Push data to DVC remote
	dvc push

dvc-status: ## Check DVC status
	dvc status

# Development workflow
dev: ## Start development environment
	@echo "Starting services..."
	make start
	@echo "Waiting for services..."
	sleep 10
	@echo "Development environment ready!"

# Example data generation
synthetic: ## Generate synthetic training data
	python scripts/synthetic_data_generator.py --count 100 --output datasets/raw/synthetic_v1

# Model operations
register-olmocr: ## Register olmOCR baseline model in MLflow
	python scripts/register_baseline_model.py || echo "Model registration script not yet implemented"

benchmark: ## Benchmark olmOCR baseline
	python scripts/benchmark_olmocr.py --config configs/benchmark_baseline.yaml || echo "Benchmark script not yet implemented"

# UI
streamlit: ## Start Streamlit UI
	streamlit run ui/app.py --server.port 8501

# API
api: ## Start FastAPI server
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Celery
celery-worker: ## Start Celery worker
	celery -A inference.tasks worker --loglevel=info

celery-flower: ## Start Celery Flower (monitoring)
	celery -A inference.tasks flower --port=5555

# Documentation
docs: ## Generate documentation
	@echo "Documentation generation not yet implemented"

# Health checks
health: ## Check all services health
	@echo "=== Checking service health ==="
	@curl -sf http://localhost:5000/health > /dev/null && echo "✓ MLflow OK" || echo "✗ MLflow DOWN"
	@curl -sf http://localhost:8080/health > /dev/null && echo "✓ Label Studio OK" || echo "✗ Label Studio DOWN"
	@curl -sf http://localhost:3000/api/health > /dev/null && echo "✓ Grafana OK" || echo "✗ Grafana DOWN"
	@docker exec raelm_postgres pg_isready -U raelm > /dev/null && echo "✓ Postgres OK" || echo "✗ Postgres DOWN"
	@docker exec raelm_redis redis-cli ping > /dev/null && echo "✓ Redis OK" || echo "✗ Redis DOWN"

