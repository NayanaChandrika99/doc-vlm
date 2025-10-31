"""
Confidence Scorer - Heuristic confidence estimation

Combines multiple signals to produce calibrated confidence scores:
- Self-consistency agreement
- Validation rule pass rate
- Region difficulty
- Historical accuracy
"""
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class HeuristicConfidenceScorer:
    """
    Compute confidence scores using heuristic formula.
    
    Formula:
        confidence = 0.40 * agreement +
                     0.30 * validation_pass_rate +
                     0.20 * (1 - difficulty) +
                     0.10 * historical_accuracy
    
    Clamped to [0.5, 0.98] to avoid overconfidence.
    """
    
    def __init__(self, db_connection=None):
        """
        Initialize confidence scorer.
        
        Args:
            db_connection: Optional database connection for historical stats
        """
        self.db = db_connection
        self._cache = {}  # Cache for historical accuracy
    
    def compute_confidence(
        self,
        prediction: Dict,
        metadata: Dict
    ) -> float:
        """
        Compute confidence score for prediction.
        
        Args:
            prediction: Model output
            metadata: Dict containing:
                - self_consistency_agreement: Agreement ratio (0-1)
                - validation_pass_rate: Proportion of rules passed (0-1)
                - difficulty_score: Region difficulty (0-1)
                - region_type: Type of region
                - prompt_id: Prompt template used
        
        Returns:
            Confidence score (0.5-0.98)
        """
        # Extract components
        agreement = metadata.get('self_consistency_agreement', 1.0)
        validation_pass = metadata.get('validation_pass_rate', 1.0)
        difficulty = metadata.get('difficulty_score', 0.5)
        
        # Get historical accuracy
        region_type = metadata.get('region_type', 'unknown')
        prompt_id = metadata.get('prompt_id', 'default')
        historical_acc = self.get_historical_accuracy(region_type, prompt_id)
        
        # Weighted combination
        confidence = (
            0.40 * agreement +
            0.30 * validation_pass +
            0.20 * (1 - difficulty) +
            0.10 * historical_acc
        )
        
        # Clamp to safe range
        confidence = max(0.5, min(0.98, confidence))
        
        logger.debug(
            f"Confidence: {confidence:.3f} "
            f"(agreement={agreement:.2f}, "
            f"validation={validation_pass:.2f}, "
            f"difficulty={difficulty:.2f}, "
            f"historical={historical_acc:.2f})"
        )
        
        return confidence
    
    def get_historical_accuracy(
        self,
        region_type: str,
        prompt_id: str
    ) -> float:
        """
        Get running accuracy for (region_type, prompt_id) combination.
        
        Starts at 0.85, updates as corrections come in.
        Uses exponential moving average (alpha=0.1).
        
        Args:
            region_type: Type of region
            prompt_id: Prompt template ID
        
        Returns:
            Historical accuracy (0-1)
        """
        key = f"{region_type}:{prompt_id}"
        
        # Check cache first
        if key in self._cache:
            return self._cache[key]
        
        # Query database if available
        if self.db:
            try:
                cursor = self.db.cursor()
                cursor.execute(
                    """
                    SELECT correct_count, total_count
                    FROM prompt_stats
                    WHERE region_type = %s AND prompt_id = %s
                    """,
                    (region_type, prompt_id)
                )
                row = cursor.fetchone()
                
                if row and row[1] > 0:
                    accuracy = row[0] / row[1]
                    self._cache[key] = accuracy
                    return accuracy
                    
            except Exception as e:
                logger.warning(f"Failed to fetch historical accuracy: {e}")
        
        # Default: assume 85% accuracy
        return 0.85
    
    def update_historical_accuracy(
        self,
        region_type: str,
        prompt_id: str,
        was_correct: bool,
        alpha: float = 0.1
    ):
        """
        Update historical accuracy with new observation.
        
        Uses exponential moving average for smooth updates.
        
        Args:
            region_type: Type of region
            prompt_id: Prompt template ID
            was_correct: Whether prediction was correct
            alpha: Smoothing factor for EMA
        """
        if not self.db:
            return
        
        try:
            cursor = self.db.cursor()
            
            # Increment counters
            cursor.execute(
                """
                INSERT INTO prompt_stats (region_type, prompt_id, correct_count, total_count)
                VALUES (%s, %s, %s, 1)
                ON CONFLICT (region_type, prompt_id) 
                DO UPDATE SET
                    correct_count = prompt_stats.correct_count + %s,
                    total_count = prompt_stats.total_count + 1,
                    last_updated = NOW()
                """,
                (region_type, prompt_id, 1 if was_correct else 0, 1 if was_correct else 0)
            )
            
            self.db.commit()
            
            # Invalidate cache
            key = f"{region_type}:{prompt_id}"
            if key in self._cache:
                del self._cache[key]
            
            logger.debug(f"Updated historical accuracy for {key}")
            
        except Exception as e:
            logger.error(f"Failed to update historical accuracy: {e}")
            self.db.rollback()

