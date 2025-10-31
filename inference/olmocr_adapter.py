"""
olmOCR Adapter - Wrapper for olmOCR-2-7B-MLX inference

Handles model loading, image preprocessing, inference, and output parsing.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import json
import re
import time
import logging
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# Try to import MLX VLM, fallback to mock if not available
try:
    from mlx_vlm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    logger.warning("mlx-vlm not installed. Install with: pip install mlx-vlm")


class OlmOCRAdapter:
    """
    Adapter for olmOCR-2-7B-MLX model.
    
    Provides:
    - Model loading and lifecycle management
    - Image preprocessing for model input
    - Inference with temperature sampling
    - Structured JSON output parsing
    """
    
    def __init__(self, model_path: Optional[Path] = None, model_name: Optional[str] = None):
        """
        Initialize olmOCR adapter.
        
        Args:
            model_path: Path to local olmOCR model weights directory (optional)
            model_name: HuggingFace model identifier (e.g., "richardyoung/olmOCR-2-7B-1025-MLX-4bit")
                        If not provided, defaults to "richardyoung/olmOCR-2-7B-1025-MLX-4bit"
        """
        self.model_path = Path(model_path) if model_path else None
        self.model_name = model_name or "richardyoung/olmOCR-2-7B-1025-MLX-4bit"
        self.model = None
        self.processor = None
        
        logger.info(f"OlmOCRAdapter initialized (model will load on first inference)")
    
    def _load_model(self):
        """Lazy load model on first inference."""
        if self.model is not None:
            return
        
        if not MLX_AVAILABLE:
            logger.error(
                "mlx-vlm is not installed. Please install it with: pip install mlx-vlm\n"
                "Falling back to mock mode."
            )
            self.model = "mock_model"
            self.processor = "mock_processor"
            return
        
        try:
            logger.info(f"Loading olmOCR model: {self.model_name}")
            
            # Determine model path: use local path if provided, otherwise use model_name
            if self.model_path and self.model_path.exists():
                load_path = str(self.model_path)
                logger.info(f"Loading from local path: {load_path}")
            else:
                load_path = self.model_name
                logger.info(f"Loading from HuggingFace: {load_path}")
            
            # Load model and processor using mlx_vlm
            self.model, self.processor = load(load_path)
            
            logger.info(f"Successfully loaded olmOCR model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load olmOCR model: {e}")
            logger.warning("Falling back to mock mode")
            self.model = "mock_model"
            self.processor = "mock_processor"
    
    def infer(
        self,
        image: Image.Image,
        prompt: str,
        seed: int = 42,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> Dict[str, Any]:
        """
        Run inference on image with prompt.
        
        Args:
            image: PIL Image
            prompt: Text prompt for extraction
            seed: Random seed for reproducibility (used for numpy random state)
            temperature: Sampling temperature (0.0 = deterministic, higher = more random)
            max_tokens: Maximum tokens to generate
        
        Returns:
            Dict with 'output', 'raw_text', 'latency_ms', 'model', 'seed'
        """
        self._load_model()
        
        start_time = time.time()
        
        try:
            # Check if model is actually loaded (not mock)
            if self.model == "mock_model" or not MLX_AVAILABLE:
                logger.warning("Using mock inference - model not available")
                output_text = self._run_mock_inference(prompt)
            else:
                # Ensure image is RGB
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Set random seed for reproducibility
                if seed is not None:
                    np.random.seed(seed)
            
                # Run actual MLX inference
                logger.debug(f"Running inference with prompt: {prompt[:100]}...")
                output_text = generate(
                    self.model,
                    self.processor,
                    image,
                    prompt,
                    max_tokens=max_tokens,
                    temp=temperature
                )
                logger.debug(f"Generated output: {output_text[:200]}...")
            
            # Parse output
            parsed_output = self._parse_output(output_text)
            
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                "output": parsed_output,
                "raw_text": output_text,
                "latency_ms": latency_ms,
                "model": self.model_name,
                "seed": seed
            }
            
        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            raise
    
    def _run_mock_inference(self, prompt: str) -> str:
        """
        Mock inference for testing when model is not available.
        
        Returns mock output based on prompt content.
        """
        logger.warning("Using mock inference output")
        
        # Mock output based on prompt
        if "checkbox" in prompt.lower():
            return json.dumps([
                {"label": "Fever", "checked": True},
                {"label": "Chills", "checked": False},
                {"label": "Cough", "checked": True}
            ])
        elif "table" in prompt.lower():
            return json.dumps({
                "rows": [
                    {"date": "01/15/2024", "symptom": "Fever", "severity": "High"},
                    {"date": "01/16/2024", "symptom": "Cough", "severity": "Medium"}
                ]
            })
        elif "signature" in prompt.lower():
            return json.dumps({
                "detected": True,
                "confidence": 0.85
            })
        else:
            return json.dumps({
                "text": "Sample extracted text",
                "confidence": 0.85
            })
    
    def _parse_output(self, output_text: str) -> Dict:
        """
        Parse model output to structured JSON.
        
        Handles:
        - Clean JSON extraction
        - Malformed JSON recovery
        - Fallback to regex extraction
        """
        # Try direct JSON parse
        try:
            return json.loads(output_text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from text
        json_match = re.search(r'\{.*\}|\[.*\]', output_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Fallback: return raw text
        logger.warning("Could not parse as JSON, returning raw text")
        return {"raw_text": output_text, "parsed": False}

