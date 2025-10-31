"""
Unit tests for Model Registry

Tests model registration, loading, promotion, and comparison.
"""
import pytest
from pathlib import Path
import tempfile
import json


def test_model_registry_init():
    """Test ModelRegistry initialization"""
    from ml_platform.model_registry import ModelRegistry
    
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ModelRegistry(
            mlflow_uri="http://localhost:5000",
            artifact_root=Path(tmpdir)
        )
        
        assert registry.mlflow_uri == "http://localhost:5000"
        assert registry.artifact_root.exists()


def test_compute_file_hash():
    """Test file hash computation"""
    from ml_platform.model_registry import ModelRegistry
    
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ModelRegistry(
            mlflow_uri="http://localhost:5000",
            artifact_root=Path(tmpdir)
        )
        
        # Create test file
        test_file = Path(tmpdir) / "test.txt"
        test_file.write_text("test content")
        
        # Compute hash
        hash1 = registry._compute_file_hash(test_file)
        hash2 = registry._compute_file_hash(test_file)
        
        # Same file should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 produces 64 hex characters


# Note: Full integration tests for register_model, load_model, etc.
# require running MLflow server and will be in integration tests

