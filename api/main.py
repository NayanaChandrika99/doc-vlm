"""
FastAPI Service - Production inference API

Endpoints (per EXECPLAN):
- POST /extract - Upload file for async processing
- GET /status/{job_id} - Poll processing state
- GET /results/{job_id} - Retrieve extraction results
- POST /review/correct - Submit corrections (HITL feedback)
- GET /review-queue - Fetch samples needing review
- GET /health - Service health check
- GET /metrics - Prometheus metrics
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import logging
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response
import time
import uuid
from pathlib import Path
import shutil
import re

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RaeLM Document Understanding API",
    description="Production-grade medical form extraction service",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'raelm_requests_total',
    'Total requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'raelm_request_latency_seconds',
    'Request latency',
    ['method', 'endpoint']
)

EXTRACTION_COUNT = Counter(
    'raelm_extractions_total',
    'Total extractions',
    ['region_type']
)


# Security utilities
def _sanitize_filename(filename: str) -> str:
    """
    Sanitize uploaded filename to prevent path traversal and other attacks.
    
    Security measures:
    1. Normalize path separators (handle both Unix and Windows)
    2. Extract only the basename (no directory components)
    3. Remove/replace dangerous characters
    4. Limit length to prevent DoS
    5. Ensure non-empty result
    
    Args:
        filename: User-supplied filename
    
    Returns:
        Safe filename suitable for filesystem storage
    
    Examples:
        >>> _sanitize_filename("../../etc/passwd")
        'passwd'
        >>> _sanitize_filename("../../../bad.txt")
        'bad.txt'
        >>> _sanitize_filename("C:\\Windows\\System32\\cmd.exe")
        'cmd.exe'
        >>> _sanitize_filename("normal_file.pdf")
        'normal_file.pdf'
        >>> _sanitize_filename("file with spaces.pdf")
        'file_with_spaces.pdf'
    """
    if not filename:
        return "unnamed_file"
    
    # 1. Normalize path separators (handle Windows backslashes on Unix and vice versa)
    # Replace backslashes with forward slashes for consistent handling
    filename = filename.replace('\\', '/')
    
    # 2. Get only the basename (strip any directory components)
    # This handles "../../../etc/passwd" → "passwd"
    # and "C:/Windows/System32/cmd.exe" → "cmd.exe"
    filename = Path(filename).name
    
    if not filename:
        return "unnamed_file"
    
    # 3. Remove or replace dangerous characters
    # Keep: alphanumeric, dots, underscores, hyphens
    # Replace spaces with underscores
    # Remove everything else
    filename = filename.replace(" ", "_")
    filename = re.sub(r'[^a-zA-Z0-9._-]', '', filename)
    
    # 4. Prevent dotfiles exploits, but preserve extension if present
    # If stripping dots would leave only an extension (e.g., ".pdf" → "pdf"),
    # keep the dot. Otherwise strip leading dots.
    if filename.startswith('.'):
        without_dots = filename.lstrip('.')
        # If no dots remain after stripping, it's likely just an extension
        # Preserve the dot if it looks like an extension (short, no dots)
        if not '.' in without_dots and len(without_dots) <= 5 and without_dots:
            # This is just an extension (e.g., ".pdf"), keep the dot
            pass  # Keep filename as is
        else:
            # This is a dotfile (like .bashrc), strip it
            filename = without_dots
    
    # 5. Limit length (filesystem limit is typically 255, use 200 for safety)
    max_length = 200
    if len(filename) > max_length:
        # Preserve extension if present
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        if ext:
            # Keep extension, truncate name
            name = name[:max_length - len(ext) - 1]
            filename = f"{name}.{ext}"
        else:
            filename = filename[:max_length]
    
    # 6. Final check: ensure non-empty
    if not filename or filename == '.':
        return "unnamed_file"
    
    return filename


# Response models
class ExtractResponse(BaseModel):
    """Response for file upload."""
    job_id: str
    status: str  # queued
    message: str
    eta_seconds: Optional[int] = None


class StatusResponse(BaseModel):
    """Response for status polling."""
    job_id: str
    status: str  # queued, processing, completed, failed
    progress: Optional[float] = None  # 0.0 - 1.0
    message: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ResultsResponse(BaseModel):
    """Response for extraction results."""
    job_id: str
    status: str
    results: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None
    processing_time_ms: Optional[float] = None
    error: Optional[str] = None
    regions: Optional[List[Dict]] = None


class CorrectionRequest(BaseModel):
    """Request for human correction."""
    job_id: str
    region_id: str
    corrected_value: Any
    original_value: Any
    comment: Optional[str] = None


class ReviewQueueItem(BaseModel):
    """Item in review queue."""
    job_id: str
    region_id: str
    region_type: str
    predicted_value: Any
    confidence: float
    priority: float
    reason: str
    image_url: Optional[str] = None


# Endpoints
@app.post("/extract", response_model=ExtractResponse)
async def extract_document(
    file: UploadFile = File(...),
    priority: int = Form(5),
    model_variant: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None
):
    """
    Upload document file for extraction.
    
    Example:
        curl -X POST http://localhost:8000/extract \
             -F 'file=@test.pdf' \
             -F 'priority=5'
    
    Workflow:
    1. Save uploaded file
    2. Generate job_id
    3. Submit to Celery queue
    4. Return job_id for polling
    """
    start_time = time.time()
    
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # SECURITY: Sanitize filename to prevent path traversal
        safe_filename = _sanitize_filename(file.filename)
        
        # Save uploaded file
        upload_dir = Path("datasets/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Use sanitized filename
        file_path = upload_dir / f"{job_id}_{safe_filename}"
        
        # Verify the resolved path is still within upload_dir (defense in depth)
        if not file_path.resolve().is_relative_to(upload_dir.resolve()):
            raise HTTPException(
                status_code=400,
                detail="Invalid filename: path traversal detected"
            )
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Received file upload: {file.filename} → {safe_filename} → job_id={job_id}")
        
        # Placeholder: Submit to Celery
        # from inference.tasks import process_document
        # task = process_document.delay(job_id, str(file_path), priority, model_variant)
        
        REQUEST_COUNT.labels(
            method="POST",
            endpoint="/extract",
            status="success"
        ).inc()
        
        return ExtractResponse(
            job_id=job_id,
            status="queued",
            message=f"Document {file.filename} queued for processing",
            eta_seconds=30
        )
        
    except Exception as e:
        logger.error(f"Failed to process upload: {e}")
        REQUEST_COUNT.labels(
            method="POST",
            endpoint="/extract",
            status="error"
        ).inc()
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        REQUEST_LATENCY.labels(
            method="POST",
            endpoint="/extract"
        ).observe(time.time() - start_time)


@app.get("/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str):
    """
    Poll processing status.
    
    Example:
        curl http://localhost:8000/status/abc-123
    
    Returns:
    - status: queued, processing, completed, failed
    - progress: 0.0 - 1.0 (if processing)
    - message: Current processing step
    """
    try:
        logger.debug(f"Status poll for job {job_id}")
        
        # Placeholder: Query Celery result backend or database
        # In production:
        # from inference.tasks import process_document
        # result = process_document.AsyncResult(job_id)
        # status = result.state  # PENDING, STARTED, SUCCESS, FAILURE
        
        REQUEST_COUNT.labels(
            method="GET",
            endpoint="/status",
            status="success"
        ).inc()
        
        # Mock response
        return StatusResponse(
            job_id=job_id,
            status="completed",
            progress=1.0,
            message="Processing complete"
        )
        
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        REQUEST_COUNT.labels(
            method="GET",
            endpoint="/status",
            status="error"
        ).inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results/{job_id}", response_model=ResultsResponse)
async def get_results(job_id: str):
    """
    Retrieve extraction results.
    
    Example:
        curl http://localhost:8000/results/abc-123
    
    Returns:
    - results: Structured JSON extraction
    - confidence: Overall confidence score
    - regions: Per-region details with confidence
    """
    try:
        logger.info(f"Retrieving results for job {job_id}")
        
        # Placeholder: Query database for completed job
        # In production, fetch from Postgres or Redis cache
        
        REQUEST_COUNT.labels(
            method="GET",
            endpoint="/results",
            status="success"
        ).inc()
        
        # Mock response
        return ResultsResponse(
            job_id=job_id,
            status="completed",
            results={
                "patient_id": "12345",
                "date": "10/30/2025",
                "symptoms": [
                    {"label": "Fever", "checked": True, "confidence": 0.95},
                    {"label": "Cough", "checked": True, "confidence": 0.88}
                ]
            },
            confidence=0.92,
            processing_time_ms=2350.5,
            regions=[
                {
                    "region_id": "reg_001",
                    "type": "checkbox",
                    "bbox": [100, 200, 50, 50],
                    "confidence": 0.95,
                    "difficulty": 0.3
                }
            ]
        )
        
    except Exception as e:
        logger.error(f"Failed to retrieve results: {e}")
        REQUEST_COUNT.labels(
            method="GET",
            endpoint="/results",
            status="error"
        ).inc()
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")


@app.post("/review/correct")
async def submit_correction(correction: CorrectionRequest):
    """
    Submit human correction for model output.
    
    Example:
        curl -X POST http://localhost:8000/review/correct \
             -H 'Content-Type: application/json' \
             -d '{"job_id": "abc-123", "region_id": "reg_001", 
                  "corrected_value": {"checked": true}, 
                  "original_value": {"checked": false}}'
    
    Used for:
    - Active learning (queue for future model improvement)
    - Confidence calibration (update historical accuracy)
    - Prompt performance tracking
    """
    try:
        logger.info(f"Received correction for job {correction.job_id}, region {correction.region_id}")
        
        # Placeholder: Store correction in database
        # from ml_platform.annotation_queue import AnnotationQueue
        # queue = AnnotationQueue(db)
        # queue.record_correction(correction.job_id, correction.region_id, 
        #                        correction.corrected_value, correction.original_value)
        
        # Update confidence scorer historical accuracy
        # from inference.confidence_scorer import HeuristicConfidenceScorer
        # scorer = HeuristicConfidenceScorer(db)
        # was_correct = (correction.original_value == correction.corrected_value)
        # scorer.update_historical_accuracy(region_type, prompt_id, was_correct)
        
        REQUEST_COUNT.labels(
            method="POST",
            endpoint="/review/correct",
            status="success"
        ).inc()
        
        return {
            "status": "success",
            "message": "Correction recorded for active learning"
        }
        
    except Exception as e:
        logger.error(f"Failed to record correction: {e}")
        REQUEST_COUNT.labels(
            method="POST",
            endpoint="/review/correct",
            status="error"
        ).inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/review-queue", response_model=List[ReviewQueueItem])
async def get_review_queue(
    limit: int = 10,
    min_priority: float = 0.0
):
    """
    Fetch samples needing human review, sorted by priority.
    
    Example:
        curl http://localhost:8000/review-queue?limit=10
    
    Returns:
    - List of samples with low confidence or validation errors
    - Sorted by priority (high to low)
    - Includes predicted value and reason for review
    """
    try:
        logger.debug(f"Fetching review queue (limit={limit})")
        
        # Placeholder: Query annotation_queue from database
        # from ml_platform.annotation_queue import AnnotationQueue
        # queue = AnnotationQueue(db)
        # items = queue.get_next_batch(limit, min_priority)
        
        REQUEST_COUNT.labels(
            method="GET",
            endpoint="/review-queue",
            status="success"
        ).inc()
        
        # Mock response
        return [
            ReviewQueueItem(
                job_id="job_001",
                region_id="reg_001",
                region_type="checkbox",
                predicted_value={"label": "Fever", "checked": True},
                confidence=0.72,
                priority=0.9,
                reason="low_confidence",
                image_url="/images/job_001_reg_001.png"
            ),
            ReviewQueueItem(
                job_id="job_002",
                region_id="reg_003",
                region_type="table",
                predicted_value={"date": "99/99/9999"},
                confidence=0.85,
                priority=0.8,
                reason="validation_error",
                image_url="/images/job_002_reg_003.png"
            )
        ]
        
    except Exception as e:
        logger.error(f"Failed to fetch review queue: {e}")
        REQUEST_COUNT.labels(
            method="GET",
            endpoint="/review-queue",
            status="error"
        ).inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Service health check."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": time.time()
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

