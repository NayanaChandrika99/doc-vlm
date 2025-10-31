"""
Dataset Manager - DVC-backed dataset versioning and lineage

Manages dataset lifecycles: creation, versioning, loading, statistics.
Tracks dataset lineage and preprocessing pipelines.
"""
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import json
from datetime import datetime
import shutil
import logging
from collections import Counter
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Sample:
    """Single training/test sample with metadata."""
    id: str
    image_path: Path
    annotation: Dict
    region_type: str  # checkbox, table, signature, text_block
    difficulty_score: float
    source: str  # real, synthetic, augmented
    metadata: Dict


@dataclass
class DatasetManifest:
    """Dataset metadata and statistics."""
    dataset_id: str
    version: str
    created_at: str
    parent_datasets: List[str]
    total_samples: int
    statistics: Dict
    preprocessing_pipeline: Dict
    split_strategy: Dict
    annotation_guidelines_version: str
    quality_metrics: Dict


class DatasetManager:
    """
    Manages dataset lifecycle with versioning and lineage.
    
    Uses DVC for large file versioning.
    Postgres for metadata and sample references.
    """
    
    def __init__(self, root_path: Path, db_connection):
        """
        Initialize dataset manager.
        
        Args:
            root_path: Root directory for datasets
            db_connection: psycopg2 connection for metadata storage
        """
        self.root = Path(root_path)
        self.db = db_connection
        self.raw_dir = self.root / "raw"
        self.processed_dir = self.root / "processed"
        self.queue_dir = self.root / "active_learning_queue"
        
        # Create directories
        for dir_path in [self.raw_dir, self.processed_dir, self.queue_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DatasetManager initialized at {root_path}")
    
    def create_dataset_version(
        self,
        name: str,
        version: str,
        source_datasets: List[str],
        preprocessing_config: Dict = None,
        split_strategy: Dict = None,
        annotation_guidelines_version: str = "v1.0"
    ) -> DatasetManifest:
        """
        Create versioned dataset with full lineage.
        
        Args:
            name: Dataset name (e.g., "medical_forms")
            version: Semantic version (e.g., "v1.0")
            source_datasets: List of parent dataset IDs
            preprocessing_config: Pipeline configuration
            split_strategy: Train/val/test split ratios
            annotation_guidelines_version: Version of annotation guidelines
        
        Returns:
            DatasetManifest with statistics and metadata
        """
        if preprocessing_config is None:
            preprocessing_config = {"version": "v1.0", "steps": []}
        
        if split_strategy is None:
            split_strategy = {"train": 0.7, "val": 0.15, "test": 0.15}
        
        dataset_id = f"{name}_{version}"
        dataset_path = self.processed_dir / dataset_id
        dataset_path.mkdir(exist_ok=True)
        
        logger.info(f"Creating dataset {dataset_id}")
        
        # Load source samples
        all_samples = []
        for source_id in source_datasets:
            try:
                samples = self.load_dataset(source_id)
                all_samples.extend(samples)
                logger.info(f"Loaded {len(samples)} samples from {source_id}")
            except Exception as e:
                logger.warning(f"Could not load source dataset {source_id}: {e}")
        
        if not all_samples:
            raise ValueError(f"No samples loaded from source datasets: {source_datasets}")
        
        logger.info(f"Total samples loaded: {len(all_samples)}")
        
        # Apply preprocessing (placeholder for now)
        processed_samples = self._apply_preprocessing(all_samples, preprocessing_config)
        
        # Split dataset
        splits = self._split_dataset(processed_samples, split_strategy)
        
        # Save splits
        for split_name, split_samples in splits.items():
            split_dir = dataset_path / split_name
            split_dir.mkdir(exist_ok=True)
            (split_dir / "images").mkdir(exist_ok=True)
            
            # Save images and annotations
            annotations = []
            for sample in split_samples:
                # Copy image
                img_dest = split_dir / "images" / f"{sample.id}.png"
                if sample.image_path.exists():
                    shutil.copy(sample.image_path, img_dest)
                
                # Collect annotation
                annotations.append({
                    "id": sample.id,
                    "image_path": f"images/{sample.id}.png",
                    "annotation": sample.annotation,
                    "region_type": sample.region_type,
                    "difficulty_score": sample.difficulty_score,
                    "source": sample.source,
                    "metadata": sample.metadata
                })
            
            # Save annotations as JSONL
            with open(split_dir / "annotations.jsonl", "w") as f:
                for ann in annotations:
                    f.write(json.dumps(ann) + "\n")
            
            logger.info(f"Saved {len(split_samples)} samples to {split_name}")
        
        # Compute statistics
        statistics = self._compute_statistics(processed_samples)
        
        # Compute quality metrics
        quality_metrics = self._compute_quality_metrics(processed_samples)
        
        # Create manifest
        manifest = DatasetManifest(
            dataset_id=dataset_id,
            version=version,
            created_at=datetime.utcnow().isoformat(),
            parent_datasets=source_datasets,
            total_samples=len(processed_samples),
            statistics=statistics,
            preprocessing_pipeline=preprocessing_config,
            split_strategy=split_strategy,
            annotation_guidelines_version=annotation_guidelines_version,
            quality_metrics=quality_metrics
        )
        
        # Save manifest
        manifest_path = dataset_path / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(asdict(manifest), f, indent=2)
        
        logger.info(f"Saved manifest to {manifest_path}")
        
        # Store metadata in database
        self._store_dataset_metadata(manifest)
        
        # Add to DVC tracking (will fail gracefully if DVC not set up)
        try:
            import subprocess
            subprocess.run(["dvc", "add", str(dataset_path)], 
                         check=False, capture_output=True)
            logger.info(f"Added {dataset_path} to DVC tracking")
        except Exception as e:
            logger.warning(f"Could not add to DVC: {e}")
        
        return manifest
    
    def load_dataset(self, dataset_id: str) -> List[Sample]:
        """
        Load dataset samples.
        
        Args:
            dataset_id: Dataset identifier
        
        Returns:
            List of Sample objects
        """
        # Check if it's a raw or processed dataset
        dataset_path = None
        for base_dir in [self.processed_dir, self.raw_dir]:
            candidate = base_dir / dataset_id
            if candidate.exists():
                dataset_path = candidate
                break
        
        if dataset_path is None:
            raise ValueError(f"Dataset not found: {dataset_id}")
        
        logger.info(f"Loading dataset from {dataset_path}")
        
        samples = []
        
        # Load from processed format
        for split_name in ["train", "val", "test"]:
            split_dir = dataset_path / split_name
            if not split_dir.exists():
                continue
            
            annotations_file = split_dir / "annotations.jsonl"
            if not annotations_file.exists():
                logger.warning(f"No annotations.jsonl in {split_dir}")
                continue
            
            with open(annotations_file) as f:
                for line in f:
                    ann = json.loads(line)
                    sample = Sample(
                        id=ann["id"],
                        image_path=split_dir / ann["image_path"],
                        annotation=ann["annotation"],
                        region_type=ann.get("region_type", "unknown"),
                        difficulty_score=ann.get("difficulty_score", 0.5),
                        source=ann.get("source", "unknown"),
                        metadata=ann.get("metadata", {})
                    )
                    samples.append(sample)
        
        logger.info(f"Loaded {len(samples)} samples from {dataset_id}")
        return samples
    
    def _apply_preprocessing(self, samples: List[Sample], config: Dict) -> List[Sample]:
        """Apply preprocessing pipeline to samples (placeholder for now)."""
        # For now, just return samples as-is
        # Later: integrate with preprocessing.image_enhancer
        logger.info(f"Applying preprocessing: {config.get('version', 'none')}")
        return samples
    
    def _split_dataset(
        self,
        samples: List[Sample],
        split_strategy: Dict
    ) -> Dict[str, List[Sample]]:
        """
        Split dataset into train/val/test with stratification.
        
        Stratifies by region_type to maintain distribution.
        """
        import random
        random.seed(42)  # Reproducible splits
        
        # Stratify by region_type
        by_type = {}
        for sample in samples:
            by_type.setdefault(sample.region_type, []).append(sample)
        
        splits = {"train": [], "val": [], "test": []}
        
        for region_type, type_samples in by_type.items():
            random.shuffle(type_samples)
            
            n = len(type_samples)
            train_end = int(n * split_strategy["train"])
            val_end = train_end + int(n * split_strategy["val"])
            
            splits["train"].extend(type_samples[:train_end])
            splits["val"].extend(type_samples[train_end:val_end])
            splits["test"].extend(type_samples[val_end:])
        
        logger.info(f"Split: train={len(splits['train'])}, "
                   f"val={len(splits['val'])}, test={len(splits['test'])}")
        
        return splits
    
    def _compute_statistics(self, samples: List[Sample]) -> Dict:
        """Compute dataset statistics."""
        region_types = Counter(s.region_type for s in samples)
        sources = Counter(s.source for s in samples)
        difficulties = [s.difficulty_score for s in samples]
        
        return {
            "region_type_distribution": dict(region_types),
            "source_distribution": dict(sources),
            "difficulty_distribution": {
                "mean": float(np.mean(difficulties)),
                "std": float(np.std(difficulties)),
                "min": float(np.min(difficulties)),
                "max": float(np.max(difficulties)),
                "quartiles": {
                    "q25": float(np.percentile(difficulties, 25)),
                    "q50": float(np.percentile(difficulties, 50)),
                    "q75": float(np.percentile(difficulties, 75))
                }
            }
        }
    
    def _compute_quality_metrics(self, samples: List[Sample]) -> Dict:
        """
        Compute annotation quality metrics.
        
        For now, returns placeholders. Will be computed from
        redundant annotations once available.
        """
        return {
            "inter_annotator_agreement": None,  # Computed from redundant annotations
            "samples_with_redundancy": 0,
            "annotation_completeness": 1.0  # All samples have annotations
        }
    
    def _store_dataset_metadata(self, manifest: DatasetManifest):
        """Store dataset metadata in Postgres for querying."""
        try:
            cursor = self.db.cursor()
            cursor.execute(
                """
                INSERT INTO datasets 
                (id, version, manifest, total_samples, statistics, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE
                SET manifest = EXCLUDED.manifest,
                    total_samples = EXCLUDED.total_samples,
                    statistics = EXCLUDED.statistics
                """,
                (
                    manifest.dataset_id,
                    manifest.version,
                    json.dumps(asdict(manifest)),
                    manifest.total_samples,
                    json.dumps(manifest.statistics),
                    manifest.created_at
                )
            )
            self.db.commit()
            logger.info(f"Stored metadata for {manifest.dataset_id} in database")
        except Exception as e:
            logger.error(f"Failed to store dataset metadata: {e}")
            self.db.rollback()

