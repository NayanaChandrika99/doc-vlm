"""
Active Learning - Intelligent sample selection for human annotation

Strategies:
- Uncertainty sampling: Low confidence predictions
- Diversity sampling: Representative samples across distribution
- Hard negative mining: Known difficult cases
"""
from typing import List, Dict, Tuple
import numpy as np
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)


class ActiveLearningSelector:
    """
    Select samples for human annotation to maximize model improvement.
    
    Prioritizes:
    1. Low confidence predictions
    2. High-value corrections (common patterns)
    3. Diverse samples (coverage)
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize active learning selector.
        
        Args:
            config: Configuration dict {
                'uncertainty_weight': 0.5,
                'diversity_weight': 0.3,
                'value_weight': 0.2,
                'batch_size': 10
            }
        """
        self.config = config or self._default_config()
    
    def select_batch(
        self,
        predictions: List[Dict],
        batch_size: int = None
    ) -> List[str]:
        """
        Select batch of samples for annotation.
        
        Args:
            predictions: List of predictions with metadata:
                {
                    'document_id': str,
                    'region_id': str,
                    'confidence': float,
                    'difficulty': float,
                    'region_type': str,
                    'embedding': np.ndarray (optional)
                }
            batch_size: Number of samples to select
        
        Returns:
            List of (document_id, region_id) tuples
        """
        if not predictions:
            return []
        
        batch_size = batch_size or self.config['batch_size']
        
        # Compute scores for each prediction
        scores = self._compute_scores(predictions)
        
        # Select top k
        top_indices = np.argsort(scores)[-batch_size:]
        
        selected = [
            (predictions[i]['document_id'], predictions[i]['region_id'])
            for i in top_indices
        ]
        
        logger.info(f"Selected {len(selected)} samples for annotation")
        return selected
    
    def _compute_scores(self, predictions: List[Dict]) -> np.ndarray:
        """
        Compute composite score for each prediction.
        
        Score = uncertainty_score * w1 + diversity_score * w2 + value_score * w3
        """
        n = len(predictions)
        
        # Uncertainty score (inverse confidence)
        uncertainty_scores = np.array([
            1 - p.get('confidence', 0.5) for p in predictions
        ])
        
        # Diversity score (distance to already-annotated samples)
        diversity_scores = self._compute_diversity_scores(predictions)
        
        # Value score (common patterns, high difficulty)
        value_scores = self._compute_value_scores(predictions)
        
        # Weighted combination
        w1 = self.config['uncertainty_weight']
        w2 = self.config['diversity_weight']
        w3 = self.config['value_weight']
        
        total_scores = (
            w1 * uncertainty_scores +
            w2 * diversity_scores +
            w3 * value_scores
        )
        
        return total_scores
    
    def _compute_diversity_scores(self, predictions: List[Dict]) -> np.ndarray:
        """
        Compute diversity scores using clustering.
        
        Prefers samples far from existing annotated data.
        """
        n = len(predictions)
        
        # Extract embeddings if available
        embeddings = []
        for p in predictions:
            if 'embedding' in p:
                embeddings.append(p['embedding'])
            else:
                # Use simple feature vector: [confidence, difficulty, region_type_hash]
                region_type_hash = hash(p.get('region_type', '')) % 100 / 100
                embeddings.append([
                    p.get('confidence', 0.5),
                    p.get('difficulty', 0.5),
                    region_type_hash
                ])
        
        embeddings = np.array(embeddings)
        
        if len(embeddings) < 2:
            return np.ones(n)
        
        # Cluster into groups
        k = min(5, len(embeddings))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        # Score by distance to cluster center
        distances = []
        for i, emb in enumerate(embeddings):
            center = kmeans.cluster_centers_[labels[i]]
            dist = np.linalg.norm(emb - center)
            distances.append(dist)
        
        # Normalize to [0, 1]
        distances = np.array(distances)
        if distances.max() > 0:
            distances = distances / distances.max()
        
        return distances
    
    def _compute_value_scores(self, predictions: List[Dict]) -> np.ndarray:
        """
        Compute value scores based on difficulty and frequency.
        
        High-value samples are:
        - High difficulty (more informative)
        - Common region types (more impact)
        """
        n = len(predictions)
        
        # Difficulty contribution
        difficulty_scores = np.array([
            p.get('difficulty', 0.5) for p in predictions
        ])
        
        # Frequency contribution (simplified - count by region type)
        region_types = [p.get('region_type', 'unknown') for p in predictions]
        type_counts = {}
        for rt in region_types:
            type_counts[rt] = type_counts.get(rt, 0) + 1
        
        frequency_scores = np.array([
            type_counts[rt] / n for rt in region_types
        ])
        
        # Combine
        value_scores = 0.6 * difficulty_scores + 0.4 * frequency_scores
        
        return value_scores
    
    def _default_config(self) -> Dict:
        """Default active learning configuration."""
        return {
            'uncertainty_weight': 0.5,
            'diversity_weight': 0.3,
            'value_weight': 0.2,
            'batch_size': 10
        }

