"""
Confidence Calibration - Temperature Scaling

Calibrates model confidence scores to match empirical accuracy.
Uses temperature scaling (Platt scaling variant) for post-hoc calibration.
"""
from typing import List, Tuple
import numpy as np
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


class TemperatureScaling:
    """
    Temperature scaling for confidence calibration.
    
    Learns a single temperature parameter T that scales logits:
        calibrated_confidence = sigmoid(logit / T)
    
    Optimizes T to minimize calibration error on validation set.
    """
    
    def __init__(self):
        """Initialize temperature scaling calibrator."""
        self.temperature = 1.0
        self.is_fitted = False
    
    def fit(
        self,
        confidences: List[float],
        correctness: List[bool]
    ):
        """
        Fit temperature parameter on validation data.
        
        Args:
            confidences: Model confidence scores (0-1)
            correctness: Ground truth correctness (True/False)
        """
        if len(confidences) != len(correctness):
            raise ValueError("confidences and correctness must have same length")
        
        if len(confidences) < 10:
            logger.warning("Very few calibration samples, results may be unreliable")
        
        # Convert to numpy
        confidences = np.array(confidences)
        correctness = np.array(correctness, dtype=float)
        
        # Convert confidences to logits
        # logit = log(p / (1 - p))
        epsilon = 1e-7
        confidences = np.clip(confidences, epsilon, 1 - epsilon)
        logits = np.log(confidences / (1 - confidences))
        
        # Optimize temperature to minimize NLL
        def negative_log_likelihood(T):
            scaled_logits = logits / T
            scaled_probs = 1 / (1 + np.exp(-scaled_logits))
            
            # Binary cross-entropy
            loss = -np.mean(
                correctness * np.log(scaled_probs + epsilon) +
                (1 - correctness) * np.log(1 - scaled_probs + epsilon)
            )
            return loss
        
        # Optimize
        result = minimize(
            negative_log_likelihood,
            x0=1.0,
            bounds=[(0.1, 10.0)],
            method='L-BFGS-B'
        )
        
        self.temperature = result.x[0]
        self.is_fitted = True
        
        logger.info(f"Temperature scaling fitted: T={self.temperature:.3f}")
    
    def calibrate(self, confidence: float) -> float:
        """
        Apply temperature scaling to calibrate confidence.
        
        Args:
            confidence: Uncalibrated confidence (0-1)
        
        Returns:
            Calibrated confidence (0-1)
        """
        if not self.is_fitted:
            logger.warning("Temperature scaling not fitted, returning original confidence")
            return confidence
        
        # Convert to logit
        epsilon = 1e-7
        confidence = np.clip(confidence, epsilon, 1 - epsilon)
        logit = np.log(confidence / (1 - confidence))
        
        # Scale by temperature
        scaled_logit = logit / self.temperature
        
        # Convert back to probability
        calibrated = 1 / (1 + np.exp(-scaled_logit))
        
        return float(calibrated)
    
    def calibrate_batch(self, confidences: List[float]) -> List[float]:
        """Calibrate a batch of confidences."""
        return [self.calibrate(c) for c in confidences]


def compute_expected_calibration_error(
    confidences: List[float],
    correctness: List[bool],
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Measures how well confidence scores match empirical accuracy.
    
    Algorithm:
    1. Bin predictions by confidence
    2. For each bin, compute |avg_confidence - accuracy|
    3. Weighted average across bins
    
    Args:
        confidences: Model confidence scores
        correctness: Ground truth correctness
        n_bins: Number of bins (default: 10)
    
    Returns:
        ECE score (0-1, lower is better)
    """
    confidences = np.array(confidences)
    correctness = np.array(correctness, dtype=float)
    
    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    
    ece = 0.0
    total_samples = len(confidences)
    
    for i in range(n_bins):
        # Find samples in this bin
        in_bin = (confidences >= bins[i]) & (confidences < bins[i + 1])
        
        if i == n_bins - 1:  # Include right edge in last bin
            in_bin = (confidences >= bins[i]) & (confidences <= bins[i + 1])
        
        bin_count = in_bin.sum()
        
        if bin_count == 0:
            continue
        
        # Compute bin accuracy and confidence
        bin_accuracy = correctness[in_bin].mean()
        bin_confidence = confidences[in_bin].mean()
        
        # Add weighted contribution
        ece += (bin_count / total_samples) * abs(bin_confidence - bin_accuracy)
    
    return ece


class CalibrationMonitor:
    """
    Monitor and report calibration metrics over time.
    
    Tracks ECE and updates temperature scaling periodically.
    """
    
    def __init__(self, update_frequency: int = 100):
        """
        Initialize calibration monitor.
        
        Args:
            update_frequency: Refit temperature every N samples
        """
        self.update_frequency = update_frequency
        self.calibrator = TemperatureScaling()
        
        self.confidences_buffer = []
        self.correctness_buffer = []
        self.sample_count = 0
    
    def add_sample(self, confidence: float, is_correct: bool):
        """Add new sample to calibration buffer."""
        self.confidences_buffer.append(confidence)
        self.correctness_buffer.append(is_correct)
        self.sample_count += 1
        
        # Refit if buffer is full
        if len(self.confidences_buffer) >= self.update_frequency:
            self._refit()
    
    def _refit(self):
        """Refit temperature scaling on buffered samples."""
        logger.info(f"Refitting calibration on {len(self.confidences_buffer)} samples")
        
        self.calibrator.fit(self.confidences_buffer, self.correctness_buffer)
        
        # Compute ECE
        ece = compute_expected_calibration_error(
            self.confidences_buffer,
            self.correctness_buffer
        )
        
        logger.info(f"ECE: {ece:.4f}, Temperature: {self.calibrator.temperature:.3f}")
        
        # Clear buffer (keep last 20% for overlap)
        keep_count = int(0.2 * len(self.confidences_buffer))
        self.confidences_buffer = self.confidences_buffer[-keep_count:]
        self.correctness_buffer = self.correctness_buffer[-keep_count:]
    
    def calibrate(self, confidence: float) -> float:
        """Calibrate confidence using current temperature."""
        return self.calibrator.calibrate(confidence)

