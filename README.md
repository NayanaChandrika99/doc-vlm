# RaeLM — Reliable Agentic Extraction with Language Models

**End-to-end medical document OCR and extraction platform with human-in-the-loop review, confidence scoring, and MLOps monitoring**

---

## Why This Project?

Medical document processing demands **100% accuracy** on structured data extraction (patient names, dates, medications, diagnoses), but OCR errors and document variability make full automation impossible. Manual review of every document is too slow.

The solution: **confidence-based human review** where:
- **High-confidence predictions** (≥95%) auto-approve and flow directly to downstream systems
- **Low-confidence predictions** (<95%) route to human annotators for correction
- **Model improves continuously** from human feedback via active learning
- **Full observability** with MLflow tracking, Prometheus metrics, and Grafana dashboards

This platform bridges the gap between unreliable full automation and costly manual processing.

---

## Visual Flow

```
┌──────────────────────────┐
│  Document Ingestion      │
│  (PDF, Image, Fax)       │
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────────────┐
│   Preprocessing Pipeline         │
│   ┌────────────────────────────┐ │
│   │ Image Enhancement          │ │
│   │ • Deskew, denoise          │ │
│   │ • Contrast normalization   │ │
│   └────────────────────────────┘ │
│   ┌────────────────────────────┐ │
│   │ Layout Detection           │ │
│   │ • Form fields              │ │
│   │ • Tables, checkboxes       │ │
│   └────────────────────────────┘ │
└───────────┬──────────────────────┘
            │
            ▼
┌──────────────────────────────────┐
│   Inference Engine               │
│   ┌────────────────────────────┐ │
│   │ olmOCR-2-7B-MLX           │ │
│   │ (Quantized VLM)            │ │
│   └────────────────────────────┘ │
│   ┌────────────────────────────┐ │
│   │ Self-Consistency Sampling  │ │
│   │ • Multiple predictions     │ │
│   │ • Agreement scoring        │ │
│   └────────────────────────────┘ │
│   ┌────────────────────────────┐ │
│   │ Confidence Estimation      │ │
│   │ • Per-field calibration    │ │
│   └────────────────────────────┘ │
└───────────┬──────────────────────┘
            │
            ▼
┌──────────────────────────────────┐
│   Schema Mapping & Validation    │
│   • Medical domain rules         │
│   • Field type checking          │
│   • Cross-field validation       │
└───────────┬──────────────────────┘
            │
            ▼
      ┌─────┴─────┐
      │ Conf ≥ τ? │
      └─────┬─────┘
            │
    ┌───────┴────────┐
    │                │
    ▼ YES (≥95%)     ▼ NO (<95%)
┌─────────┐    ┌──────────────────┐
│ Auto-   │    │ Human Review     │
│ Approve │    │ Queue            │
│         │    │ ┌──────────────┐ │
│ Export  │    │ │ Streamlit UI │ │
│ to      │    │ │ Annotation   │ │
│ EHR/DB  │    │ └──────────────┘ │
└─────────┘    │ Corrections      │
               │ Fed Back         │
               └─────┬────────────┘
                     │
                     ▼
               ┌──────────────────┐
               │ Active Learning  │
               │ • Dataset update │
               │ • Model retrain  │
               └──────────────────┘
                     │
                     ▼
               ┌──────────────────┐
               │ Observability    │
               │ • MLflow tracking│
               │ • Prometheus     │
               │ • Grafana        │
               └──────────────────┘
```

---

## Tech Stack

### Core ML/AI
- **Vision-Language Model**: olmOCR-2-7B-MLX (quantized for Apple Silicon)
- **Framework**: PyTorch with MLX acceleration
- **Confidence Estimation**: Self-consistency sampling + calibration
- **Active Learning**: Dataset manager with annotation queue

### Backend Services
- **API Framework**: FastAPI (async endpoints)
- **Task Queue**: Celery with Redis broker
- **Database**: PostgreSQL (metadata, annotations, audit logs)
- **Model Registry**: MLflow (experiment tracking, model versioning)

### Observability
- **Metrics**: Prometheus (scraping /metrics endpoints)
- **Dashboards**: Grafana (pre-configured visualizations)
- **Logging**: Structured logs with request tracing

### Infrastructure
- **Containerization**: Docker + Docker Compose
- **Orchestration**: Make-based workflows
- **UI**: Streamlit (annotation interface, monitoring dashboard)

---

## Features & ML Components

### Core Capabilities
✓ **End-to-End Pipeline** — Ingestion → Preprocessing → Inference → Validation → Export  
✓ **Human-in-the-Loop** — Confidence-based routing to annotation queue  
✓ **Self-Consistency Sampling** — Multiple predictions with agreement scoring  
✓ **Calibrated Confidence** — Per-field confidence estimation with calibration  
✓ **Schema Mapping** — Medical domain-specific validation and type checking  
✓ **Active Learning** — Continuous model improvement from human corrections  
✓ **Multi-Model Support** — Model router with fallback strategies  
✓ **Full Observability** — MLflow, Prometheus, Grafana integration  

### ML Platform Components
- **Model Registry** (`ml_platform/model_registry.py`): Version control for models
- **Dataset Manager** (`ml_platform/dataset_manager.py`): Unified data access
- **Calibration** (`ml_platform/calibration.py`): Confidence score calibration
- **Annotation Queue** (`ml_platform/annotation_queue.py`): Human review orchestration
- **Active Learning** (`ml_platform/active_learning.py`): Sample selection strategies

### Inference Components
- **olmOCR Adapter** (`inference/olmocr_adapter.py`): Model loading and inference
- **Confidence Scorer** (`inference/confidence_scorer.py`): Multi-method confidence estimation
- **Prompt Selector** (`inference/prompt_selector.py`): Dynamic prompt management
- **Self-Consistency** (`inference/self_consistency.py`): Ensemble prediction

## Key Directories

```
tennr-realm/
├── api/
│   ├── main.py              # FastAPI application entry
│   ├── model_router.py      # Multi-model routing logic
│   └── routes/
│       ├── extraction.py    # Document extraction endpoints
│       ├── annotation.py    # Annotation queue APIs
│       └── health.py        # Health checks
├── inference/
│   ├── olmocr_adapter.py    # Model loading and inference
│   ├── confidence_scorer.py # Confidence estimation
│   ├── prompt_selector.py   # Dynamic prompt selection
│   └── self_consistency.py  # Ensemble prediction
├── ml_platform/
│   ├── model_registry.py    # MLflow model management
│   ├── dataset_manager.py   # Dataset versioning
│   ├── calibration.py       # Confidence calibration
│   ├── annotation_queue.py  # Human review orchestration
│   └── active_learning.py   # Sample selection strategies
├── preprocessing/
│   ├── image_enhancer.py    # Image quality improvement
│   └── layout_detector.py   # Form structure detection
├── validation/
│   ├── schema_validator.py  # Field type checking
│   ├── medical_rules.py     # Domain-specific validation
│   └── cross_field.py       # Consistency checks
├── ui/
│   ├── annotation_app.py    # Streamlit annotation interface
│   ├── monitoring.py        # Real-time metrics dashboard
│   └── components/          # Reusable UI components
├── docker/
│   ├── docker-compose.yml   # Service orchestration
│   ├── grafana/             # Dashboard configs
│   └── prometheus/          # Scraping configs
├── configs/
│   ├── models/              # Model configurations
│   ├── prompts/             # Prompt templates
│   └── training/            # Training hyperparameters
├── scripts/
│   ├── run_demo.sh          # End-to-end demo
│   ├── synthetic_data_generator.py
│   └── verify_milestone1.py
└── tests/
    ├── test_inference.py
    ├── test_validation.py
    ├── test_active_learning.py
    └── test_integration.py
```

---

## Documentation

- **[docs/API_CONTRACT_FIX.md](docs/API_CONTRACT_FIX.md)** — API schema validation fixes
- **[docs/IMPLEMENTATION_COMPLETE.md](docs/IMPLEMENTATION_COMPLETE.md)** — Implementation checklist
- **[docs/MILESTONE1_SUMMARY.md](docs/MILESTONE1_SUMMARY.md)** — Phase 1 completion summary
- **[docs/PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md)** — High-level project overview
- **[docs/SECURITY_FIX_SUMMARY.md](docs/SECURITY_FIX_SUMMARY.md)** — Security vulnerability patches

---

## License & Credits

RaeLM is a production-oriented medical document processing platform.  
Built with olmOCR-2-7B-MLX, FastAPI, MLflow, and Streamlit.  
Designed for HIPAA-compliant healthcare workflows with human-in-the-loop quality assurance.

**Key Technologies**:
- olmOCR-2-7B-MLX by richardyoung (Hugging Face)
- FastAPI by Sebastián Ramírez
- MLflow by Databricks
- Grafana Labs observability stack
