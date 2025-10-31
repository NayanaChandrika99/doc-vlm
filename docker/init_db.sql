-- RaeLM Database Schema Initialization
-- Creates tables for documents, annotations, jobs, datasets, and prompt statistics

-- Documents table: tracks uploaded documents and processing status
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    uploaded_at TIMESTAMP DEFAULT NOW(),
    processing_status TEXT DEFAULT 'queued',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(processing_status);
CREATE INDEX IF NOT EXISTS idx_documents_uploaded ON documents(uploaded_at DESC);

-- Datasets table: metadata for versioned datasets
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    version TEXT NOT NULL,
    manifest JSONB NOT NULL,
    total_samples INTEGER,
    statistics JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_datasets_version ON datasets(version);

-- Annotations table: human annotations and corrections
CREATE TABLE IF NOT EXISTS annotations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sample_id UUID NOT NULL,
    annotator_id TEXT NOT NULL,
    annotation JSONB NOT NULL,
    time_spent_seconds INTEGER,
    redundancy_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_annotations_sample ON annotations(sample_id);
CREATE INDEX IF NOT EXISTS idx_annotations_annotator ON annotations(annotator_id);
CREATE INDEX IF NOT EXISTS idx_annotations_created ON annotations(created_at DESC);

-- Annotation queue: samples awaiting human review
CREATE TABLE IF NOT EXISTS annotation_queue (
    sample_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    region_image_path TEXT,
    region_type TEXT,
    priority_score FLOAT NOT NULL,
    status TEXT DEFAULT 'pending',
    assigned_to TEXT,
    assigned_at TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_queue_priority ON annotation_queue(priority_score DESC, created_at ASC) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS idx_queue_status ON annotation_queue(status);
CREATE INDEX IF NOT EXISTS idx_queue_assignee ON annotation_queue(assigned_to);

-- Model versions table: tracks registered models
CREATE TABLE IF NOT EXISTS model_versions (
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    mlflow_run_id TEXT,
    model_path TEXT,
    metadata JSONB DEFAULT '{}',
    registered_at TIMESTAMP DEFAULT NOW(),
    stage TEXT DEFAULT 'None',
    PRIMARY KEY (name, version)
);

CREATE INDEX IF NOT EXISTS idx_model_stage ON model_versions(name, stage);

-- Prompt statistics: track performance per prompt template
CREATE TABLE IF NOT EXISTS prompt_stats (
    region_type TEXT NOT NULL,
    prompt_id TEXT NOT NULL,
    correct_count INTEGER DEFAULT 0,
    total_count INTEGER DEFAULT 0,
    avg_latency_ms FLOAT,
    avg_confidence FLOAT,
    last_updated TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (region_type, prompt_id)
);

-- Inference results: store extraction outputs
CREATE TABLE IF NOT EXISTS inference_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    region_id TEXT NOT NULL,
    region_type TEXT,
    prediction JSONB NOT NULL,
    confidence_score FLOAT,
    self_consistency_agreement FLOAT,
    validation_pass_rate FLOAT,
    model_variant TEXT,
    prompt_id TEXT,
    difficulty_score FLOAT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_results_document ON inference_results(document_id);
CREATE INDEX IF NOT EXISTS idx_results_confidence ON inference_results(confidence_score);
CREATE INDEX IF NOT EXISTS idx_results_created ON inference_results(created_at DESC);

-- Job tracking: Celery job states
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    status TEXT DEFAULT 'queued',
    progress JSONB DEFAULT '{}',
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_document ON jobs(document_id);

-- Drift monitoring: track distribution statistics
CREATE TABLE IF NOT EXISTS drift_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    snapshot_date DATE NOT NULL,
    statistics JSONB NOT NULL,
    psi_scores JSONB DEFAULT '{}',
    drift_detected BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_drift_date ON drift_snapshots(snapshot_date DESC);

-- Calibration history: track temperature parameter updates
CREATE TABLE IF NOT EXISTS calibration_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    temperature FLOAT NOT NULL,
    ece FLOAT,
    mce FLOAT,
    num_samples INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_calibration_created ON calibration_history(created_at DESC);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to documents table
DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;
CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions (adjust as needed for production)
-- These are commented out for local dev; uncomment and customize for production
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO raelm;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO raelm;

