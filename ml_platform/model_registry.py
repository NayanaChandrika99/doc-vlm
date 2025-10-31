"""
Model Registry - MLflow-backed model versioning and lineage tracking

Manages model lifecycles: registration, loading, promotion, comparison.
Tracks model metadata, performance metrics, and parent-child relationships.
"""
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import mlflow
from mlflow.tracking import MlflowClient
import json
from datetime import datetime
import hashlib
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Centralized model versioning with lineage tracking.
    
    Wraps MLflow Model Registry to provide:
    - Model registration with metadata and lineage
    - Version loading and promotion
    - Model comparison across versions
    - Integrity verification via hashing
    """
    
    def __init__(self, mlflow_uri: str, artifact_root: Path):
        """
        Initialize model registry.
        
        Args:
            mlflow_uri: MLflow tracking server URI (e.g., http://localhost:5000)
            artifact_root: Local path for artifacts before upload
        """
        self.mlflow_uri = mlflow_uri
        self.artifact_root = Path(artifact_root)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        
        self.client = MlflowClient(mlflow_uri)
        mlflow.set_tracking_uri(mlflow_uri)
        
        logger.info(f"ModelRegistry initialized: {mlflow_uri}")
    
    def register_model(
        self,
        name: str,
        version: str,
        model_path: Path,
        metadata: Dict,
        parent_version: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Register model with full lineage and metadata.
        
        Args:
            name: Model name (e.g., "olmocr_baseline", "checkbox_detector")
            version: Semantic version (e.g., "v1.0.0")
            model_path: Path to model weights
            metadata: Dict containing:
                - architecture: Model architecture name
                - training_data_version: Dataset version used (if applicable)
                - hyperparameters: Training hyperparameters (if applicable)
                - evaluation_metrics: Performance metrics
                - hardware_used: Training hardware specs
                - quantization: Quantization method (e.g., "4-bit")
            parent_version: Parent model version for lineage tracking
            tags: Additional tags for filtering
        
        Returns:
            model_uri: MLflow model URI
        """
        experiment_name = f"{name}_registration"
        experiment = mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(experiment_id=experiment.experiment_id):
            # Log basic info
            mlflow.log_param("model_name", name)
            mlflow.log_param("version", version)
            mlflow.log_param("registered_at", datetime.utcnow().isoformat())
            
            # Log metadata
            if "hyperparameters" in metadata:
                for key, value in metadata["hyperparameters"].items():
                    mlflow.log_param(f"hp_{key}", value)
            
            if "evaluation_metrics" in metadata:
                for key, value in metadata["evaluation_metrics"].items():
                    mlflow.log_metric(key, value)
            
            # Log full metadata as artifact
            metadata_path = self.artifact_root / f"{name}_{version}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            mlflow.log_artifact(str(metadata_path))
            
            # Log model weights if not too large
            if model_path.exists():
                model_size_mb = model_path.stat().st_size / (1024 * 1024)
                if model_size_mb < 100:
                    mlflow.log_artifact(str(model_path))
                else:
                    # For large models, log path reference
                    mlflow.log_param("model_path", str(model_path))
                    logger.info(f"Model too large ({model_size_mb:.1f}MB), logged path only")
                
                # Compute model hash for integrity
                model_hash = self._compute_file_hash(model_path)
                mlflow.log_param("model_hash", model_hash)
            
            # Track lineage
            if parent_version:
                mlflow.set_tag("parent_version", parent_version)
                mlflow.set_tag("lineage_type", "fine-tuned")
            else:
                mlflow.set_tag("lineage_type", "baseline")
            
            # Custom tags
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)
            
            # Register model
            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/{name}"
            
            try:
                registered_model = mlflow.register_model(model_uri, name)
                
                # Add version alias
                self.client.set_registered_model_alias(
                    name=name,
                    alias=version,
                    version=registered_model.version
                )
                
                logger.info(f"Registered {name}:{version} (MLflow version {registered_model.version})")
                
            except Exception as e:
                logger.error(f"Failed to register model: {e}")
                raise
        
        return model_uri
    
    def load_model(
        self,
        name: str,
        version: str = "latest",
        stage: Optional[str] = None
    ) -> Tuple[Path, Dict]:
        """
        Load model with all metadata.
        
        Args:
            name: Model name
            version: Version string or "latest"
            stage: MLflow stage (None, "Staging", "Production")
        
        Returns:
            (model_path, metadata): Path to downloaded model and metadata dict
        """
        try:
            if stage:
                model_versions = self.client.get_latest_versions(name, stages=[stage])
                if not model_versions:
                    raise ValueError(f"No model found in stage {stage}")
                model_version = model_versions[0]
            else:
                if version == "latest":
                    model_versions = self.client.search_model_versions(f"name='{name}'")
                    if not model_versions:
                        raise ValueError(f"No versions found for model {name}")
                    model_version = max(model_versions, key=lambda v: int(v.version))
                else:
                    model_version = self.client.get_model_version_by_alias(name, version)
            
            # Download artifacts
            artifact_uri = model_version.source
            local_path = mlflow.artifacts.download_artifacts(artifact_uri)
            
            # Load metadata
            metadata_file = Path(local_path) / f"{name}_{version}_metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
            else:
                metadata = {}
                logger.warning(f"Metadata file not found for {name}:{version}")
            
            logger.info(f"Loaded {name}:{version} from {local_path}")
            return Path(local_path), metadata
            
        except Exception as e:
            logger.error(f"Failed to load model {name}:{version}: {e}")
            raise
    
    def promote_model(self, name: str, version: str, stage: str):
        """
        Promote model to different stage (Staging, Production).
        
        Args:
            name: Model name
            version: Version to promote
            stage: Target stage ("Staging" or "Production")
        """
        try:
            model_version = self.client.get_model_version_by_alias(name, version)
            self.client.transition_model_version_stage(
                name=name,
                version=model_version.version,
                stage=stage
            )
            logger.info(f"Promoted {name}:{version} to {stage}")
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            raise
    
    def compare_models(
        self,
        model_a: Tuple[str, str],
        model_b: Tuple[str, str],
        metrics: List[str]
    ) -> Dict:
        """
        Compare two model versions across specified metrics.
        
        Args:
            model_a: (name, version) tuple
            model_b: (name, version) tuple
            metrics: List of metric names to compare
        
        Returns:
            Comparison dict with deltas
        """
        try:
            _, metadata_a = self.load_model(*model_a)
            _, metadata_b = self.load_model(*model_b)
            
            comparison = {
                "model_a": f"{model_a[0]}:{model_a[1]}",
                "model_b": f"{model_b[0]}:{model_b[1]}",
                "metrics": {}
            }
            
            for metric in metrics:
                value_a = metadata_a.get("evaluation_metrics", {}).get(metric)
                value_b = metadata_b.get("evaluation_metrics", {}).get(metric)
                
                if value_a is not None and value_b is not None:
                    delta = value_b - value_a
                    percent_change = (delta / value_a) * 100 if value_a != 0 else 0
                    
                    comparison["metrics"][metric] = {
                        "model_a": value_a,
                        "model_b": value_b,
                        "delta": delta,
                        "percent_change": percent_change
                    }
            
            logger.info(f"Compared {model_a} vs {model_b}")
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare models: {e}")
            raise
    
    def _compute_file_hash(self, path: Path) -> str:
        """Compute SHA256 hash of file for integrity checking."""
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

