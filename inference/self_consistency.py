"""
Self-Consistency Engine - Multi-sample inference with majority voting

Improves reliability by running multiple inferences and selecting consensus.
Confidence based on agreement ratio across samples.
"""
from typing import Dict, Tuple, List
from collections import Counter
import json
import logging
import asyncio

logger = logging.getLogger(__name__)


async def run_self_consistency(
    image,
    prompt: str,
    adapter,
    k: int = 3
) -> Tuple[Dict, float]:
    """
    Run k inferences with same prompt, compute consensus.
    
    Algorithm:
    1. Run olmOCR k times with different seeds
    2. Compare outputs
    3. Use majority voting for final answer
    4. Compute agreement score as confidence indicator
    
    Args:
        image: PIL Image
        prompt: Inference prompt
        adapter: OlmOCRAdapter instance
        k: Number of samples (1, 3, or 5)
    
    Returns:
        (majority_prediction, agreement_score)
    """
    if k == 1:
        # Fast path: single inference
        result = adapter.infer(image, prompt, seed=42)
        return result['output'], 1.0
    
    # Run k parallel inferences
    logger.debug(f"Running self-consistency with k={k}")
    
    tasks = []
    for seed in range(k):
        # In production, these should run in parallel
        tasks.append(_run_single_inference(image, prompt, adapter, seed))
    
    predictions = await asyncio.gather(*tasks)
    
    # Majority voting
    prediction, agreement = _majority_vote(predictions)
    
    logger.debug(f"Self-consistency: {k} samples, agreement={agreement:.2f}")
    
    return prediction, agreement


async def _run_single_inference(image, prompt: str, adapter, seed: int) -> Dict:
    """Run single inference asynchronously."""
    # For now, run synchronously (would be async in production)
    result = adapter.infer(image, prompt, seed=seed)
    return result['output']


def _majority_vote(predictions: List[Dict]) -> Tuple[Dict, float]:
    """
    Compute majority vote across predictions.
    
    Args:
        predictions: List of prediction dicts
    
    Returns:
        (majority_prediction, agreement_ratio)
    """
    k = len(predictions)
    
    # Serialize predictions for comparison
    serialized = [json.dumps(p, sort_keys=True) for p in predictions]
    
    # Count occurrences
    counter = Counter(serialized)
    most_common, count = counter.most_common(1)[0]
    
    # Agreement ratio
    agreement = count / k
    
    # Deserialize majority prediction
    majority_prediction = json.loads(most_common)
    
    return majority_prediction, agreement


def compute_confidence_from_agreement(
    agreement_ratio: float,
    difficulty: float,
    k: int
) -> float:
    """
    Compute confidence score based on self-consistency agreement.
    
    Formula:
        confidence = agreement_ratio * (1 - difficulty * 0.2)
        
        If unanimous (agreement=1.0), add bonus: +0.05
    
    Args:
        agreement_ratio: Proportion of samples that agree (0-1)
        difficulty: Region difficulty score (0-1)
        k: Number of samples
    
    Returns:
        Confidence score (0.5-0.98)
    """
    # Base confidence from agreement
    base_conf = agreement_ratio
    
    # Difficulty penalty (harder regions less reliable)
    difficulty_penalty = difficulty * 0.2
    
    confidence = base_conf * (1 - difficulty_penalty)
    
    # Unanimous bonus
    if agreement_ratio == 1.0:
        confidence = min(confidence + 0.05, 0.98)
    
    # Clamp to reasonable range
    confidence = max(0.5, min(0.98, confidence))
    
    return confidence

