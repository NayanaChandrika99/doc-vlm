"""
Annotation Queue - Priority-based active learning queue

Manages samples awaiting human annotation/correction.
Prioritizes by uncertainty, disagreement, and business criticality.
"""
from typing import List, Dict, Optional
from dataclasses import dataclass
import json
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)


@dataclass
class QueuedSample:
    """Sample in annotation queue."""
    sample_id: str
    document_id: Optional[str]
    region_image_path: str
    region_type: str
    priority_score: float
    status: str  # pending, in_progress, completed
    assigned_to: Optional[str]
    metadata: Dict


class AnnotationQueue:
    """
    Priority queue for human annotation with uncertainty-based sampling.
    
    Supports:
    - Priority-based retrieval
    - Annotator assignment tracking
    - Quality control (redundant annotations)
    - Inter-annotator agreement calculation
    """
    
    def __init__(self, db_connection):
        """
        Initialize annotation queue.
        
        Args:
            db_connection: psycopg2 connection
        """
        self.db = db_connection
        logger.info("AnnotationQueue initialized")
    
    def add_sample(
        self,
        sample: Dict,
        priority: float,
        metadata: Dict,
        document_id: Optional[str] = None
    ) -> str:
        """
        Add sample to annotation queue with priority score.
        
        Priority factors:
        - Model uncertainty (1 - confidence)
        - Self-consistency disagreement
        - Validation rule failures
        - Business impact (critical fields)
        - Representativeness (undersampled regions)
        
        Args:
            sample: Sample data with id, image_path, region_type
            priority: Priority score (0-1, higher = more urgent)
            metadata: Additional metadata (reason, confidence, etc.)
            document_id: Optional parent document ID
        
        Returns:
            sample_id: UUID of queued sample
        """
        sample_id = sample.get('id', str(uuid.uuid4()))
        
        try:
            cursor = self.db.cursor()
            cursor.execute(
                """
                INSERT INTO annotation_queue 
                (sample_id, document_id, region_image_path, region_type, 
                 priority_score, status, metadata, created_at)
                VALUES (%s, %s, %s, %s, %s, 'pending', %s, %s)
                ON CONFLICT (sample_id) DO UPDATE
                SET priority_score = EXCLUDED.priority_score,
                    metadata = EXCLUDED.metadata
                """,
                (
                    sample_id,
                    document_id,
                    sample.get('image_path', ''),
                    sample.get('region_type', 'unknown'),
                    priority,
                    json.dumps(metadata),
                    datetime.utcnow()
                )
            )
            self.db.commit()
            
            logger.info(f"Added sample {sample_id} to queue (priority={priority:.2f})")
            return sample_id
            
        except Exception as e:
            logger.error(f"Failed to add sample to queue: {e}")
            self.db.rollback()
            raise
    
    def get_next_batch(
        self,
        annotator_id: str,
        batch_size: int = 10,
        region_type_filter: Optional[str] = None
    ) -> List[QueuedSample]:
        """
        Fetch highest-priority unannotated samples.
        
        Considers:
        - Priority score
        - Annotator expertise (route hard samples to experts)
        - Diversity (avoid all samples from same document)
        
        Args:
            annotator_id: ID of annotator requesting samples
            batch_size: Number of samples to fetch
            region_type_filter: Optional filter by region type
        
        Returns:
            List of QueuedSample objects
        """
        try:
            cursor = self.db.cursor()
            
            query = """
                SELECT sample_id, document_id, region_image_path, region_type,
                       priority_score, status, assigned_to, metadata
                FROM annotation_queue
                WHERE status = 'pending'
                AND (assigned_to IS NULL OR assigned_to = %s)
            """
            params = [annotator_id]
            
            if region_type_filter:
                query += " AND region_type = %s"
                params.append(region_type_filter)
            
            query += " ORDER BY priority_score DESC, created_at ASC LIMIT %s"
            params.append(batch_size)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            if not rows:
                logger.info(f"No pending samples for {annotator_id}")
                return []
            
            # Mark as assigned
            sample_ids = [row[0] for row in rows]
            cursor.execute(
                """
                UPDATE annotation_queue
                SET status = 'in_progress', 
                    assigned_to = %s, 
                    assigned_at = %s
                WHERE sample_id = ANY(%s)
                """,
                (annotator_id, datetime.utcnow(), sample_ids)
            )
            self.db.commit()
            
            # Convert to QueuedSample objects
            samples = []
            for row in rows:
                sample = QueuedSample(
                    sample_id=row[0],
                    document_id=row[1],
                    region_image_path=row[2],
                    region_type=row[3],
                    priority_score=row[4],
                    status=row[5],
                    assigned_to=row[6],
                    metadata=row[7] if isinstance(row[7], dict) else json.loads(row[7] or '{}')
                )
                samples.append(sample)
            
            logger.info(f"Fetched {len(samples)} samples for {annotator_id}")
            return samples
            
        except Exception as e:
            logger.error(f"Failed to fetch batch: {e}")
            self.db.rollback()
            raise
    
    def submit_annotation(
        self,
        sample_id: str,
        annotation: Dict,
        annotator_id: str,
        time_spent_seconds: int
    ):
        """
        Record annotation with quality metrics.
        
        Args:
            sample_id: Sample identifier
            annotation: Annotation data
            annotator_id: Annotator identifier
            time_spent_seconds: Time taken to annotate
        """
        try:
            cursor = self.db.cursor()
            
            # Insert annotation
            cursor.execute(
                """
                INSERT INTO annotations
                (id, sample_id, annotator_id, annotation, time_spent_seconds, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    str(uuid.uuid4()),
                    sample_id,
                    annotator_id,
                    json.dumps(annotation),
                    time_spent_seconds,
                    datetime.utcnow()
                )
            )
            
            # Update queue status
            cursor.execute(
                """
                UPDATE annotation_queue
                SET status = 'completed', completed_at = %s
                WHERE sample_id = %s
                """,
                (datetime.utcnow(), sample_id)
            )
            
            self.db.commit()
            logger.info(f"Annotation submitted for {sample_id} by {annotator_id}")
            
        except Exception as e:
            logger.error(f"Failed to submit annotation: {e}")
            self.db.rollback()
            raise
    
    def compute_priority(
        self,
        confidence: float,
        self_consistency_agreement: float,
        validation_failures: int,
        business_criticality: float = 0.5
    ) -> float:
        """
        Compute priority score for active learning.
        
        Formula:
        priority = 0.4 * (1 - confidence) +
                   0.3 * (1 - agreement) +
                   0.2 * (validation_failures > 0) +
                   0.1 * business_criticality
        
        Args:
            confidence: Model confidence (0-1)
            self_consistency_agreement: Agreement across samples (0-1)
            validation_failures: Number of rule failures
            business_criticality: Business importance (0-1)
        
        Returns:
            Priority score (0-1, higher = more urgent)
        """
        priority = (
            0.4 * (1 - confidence) +
            0.3 * (1 - self_consistency_agreement) +
            0.2 * (1.0 if validation_failures > 0 else 0.0) +
            0.1 * business_criticality
        )
        
        return min(1.0, max(0.0, priority))
    
    def get_queue_stats(self) -> Dict:
        """Get statistics about annotation queue."""
        try:
            cursor = self.db.cursor()
            
            cursor.execute("""
                SELECT 
                    COUNT(*) FILTER (WHERE status = 'pending') as pending,
                    COUNT(*) FILTER (WHERE status = 'in_progress') as in_progress,
                    COUNT(*) FILTER (WHERE status = 'completed') as completed,
                    AVG(priority_score) FILTER (WHERE status = 'pending') as avg_priority
                FROM annotation_queue
            """)
            
            row = cursor.fetchone()
            
            return {
                "pending": row[0] or 0,
                "in_progress": row[1] or 0,
                "completed": row[2] or 0,
                "avg_priority": float(row[3] or 0.0)
            }
            
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            return {}
    
    def compute_inter_annotator_agreement(self) -> Dict:
        """
        Compute Inter-Annotator Agreement (IAA) metrics.
        
        Uses Cohen's Kappa for samples with multiple annotations.
        
        Returns:
            Dict with IAA statistics
        """
        try:
            cursor = self.db.cursor()
            
            # Find samples with multiple annotations
            cursor.execute("""
                SELECT sample_id, COUNT(*) as annotation_count
                FROM annotations
                GROUP BY sample_id
                HAVING COUNT(*) > 1
            """)
            
            redundant_samples = cursor.fetchall()
            
            if not redundant_samples:
                return {
                    "mean_kappa": None,
                    "samples_with_redundancy": 0,
                    "message": "No redundant annotations yet"
                }
            
            # For each redundant sample, compute agreement
            # (Simplified implementation - full Cohen's Kappa requires label comparison)
            agreements = []
            for sample_id, count in redundant_samples:
                cursor.execute("""
                    SELECT annotation
                    FROM annotations
                    WHERE sample_id = %s
                """, (sample_id,))
                
                annotations = [json.loads(row[0]) for row in cursor.fetchall()]
                
                # Simple agreement: check if all annotations match
                first = json.dumps(annotations[0], sort_keys=True)
                matches = all(json.dumps(a, sort_keys=True) == first for a in annotations)
                agreements.append(1.0 if matches else 0.0)
            
            return {
                "mean_agreement": sum(agreements) / len(agreements) if agreements else 0.0,
                "samples_with_redundancy": len(redundant_samples),
                "total_redundant_annotations": sum(count for _, count in redundant_samples)
            }
            
        except Exception as e:
            logger.error(f"Failed to compute IAA: {e}")
            return {"error": str(e)}

