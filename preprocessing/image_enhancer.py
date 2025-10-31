"""
Image Enhancement - Deterministic preprocessing pipeline

Deskew, denoise, normalize DPI, enhance contrast for medical forms.
All transformations are deterministic and versioned.
"""
from pathlib import Path
from typing import Tuple, Dict
import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class ImageEnhancer:
    """
    Preprocessing pipeline for document images.
    
    Steps:
    1. Deskew: Detect and correct rotation
    2. Denoise: Bilateral filter (preserves edges)
    3. Normalize DPI: Resize to 300 DPI standard
    4. Enhance contrast: CLAHE
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize enhancer with configuration.
        
        Args:
            config: Processing parameters {
                'target_dpi': 300,
                'deskew_threshold': 0.5,
                'denoise_strength': 75,
                'contrast_clip_limit': 2.0
            }
        """
        self.config = config or self._default_config()
        logger.info(f"ImageEnhancer initialized with config: {self.config}")
    
    def process(self, image: Image.Image) -> Image.Image:
        """
        Apply full preprocessing pipeline.
        
        Args:
            image: Input PIL Image
        
        Returns:
            Enhanced PIL Image
        """
        # Convert to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Apply pipeline
        img_cv = self.deskew(img_cv)
        img_cv = self.denoise(img_cv)
        img_cv = self.normalize_dpi(img_cv, image.size)
        img_cv = self.enhance_contrast(img_cv)
        
        # Convert back to PIL
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)
    
    def deskew(self, img: np.ndarray) -> np.ndarray:
        """
        Detect and correct skew using Hough line detection.
        
        Args:
            img: OpenCV image (BGR)
        
        Returns:
            Deskewed image
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        
        if lines is None:
            logger.debug("No lines detected for deskew")
            return img
        
        # Compute median angle
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta) - 90
            # Filter near-horizontal lines
            if -45 < angle < 45:
                angles.append(angle)
        
        if not angles:
            return img
        
        median_angle = np.median(angles)
        
        # Only correct if angle exceeds threshold
        if abs(median_angle) > self.config['deskew_threshold']:
            # Rotate image
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), 
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)
            
            logger.debug(f"Deskewed by {median_angle:.2f} degrees")
        
        return img
    
    def denoise(self, img: np.ndarray) -> np.ndarray:
        """
        Apply bilateral filter for noise reduction while preserving edges.
        
        Args:
            img: OpenCV image (BGR)
        
        Returns:
            Denoised image
        """
        strength = self.config['denoise_strength']
        denoised = cv2.bilateralFilter(
            img, 
            d=9, 
            sigmaColor=strength, 
            sigmaSpace=strength
        )
        
        logger.debug("Applied bilateral denoise")
        return denoised
    
    def normalize_dpi(
        self, 
        img: np.ndarray, 
        original_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Normalize to target DPI (usually 300).
        
        Assumes 8.5" x 11" page size.
        
        Args:
            img: OpenCV image (BGR)
            original_size: Original (width, height)
        
        Returns:
            Resized image
        """
        target_dpi = self.config['target_dpi']
        
        # Target dimensions for 8.5" x 11" at target DPI
        target_width = int(8.5 * target_dpi)
        target_height = int(11 * target_dpi)
        
        h, w = img.shape[:2]
        
        # Only resize if significantly different
        if abs(w - target_width) > 100 or abs(h - target_height) > 100:
            img = cv2.resize(
                img, 
                (target_width, target_height),
                interpolation=cv2.INTER_CUBIC
            )
            logger.debug(f"Resized from ({w}, {h}) to ({target_width}, {target_height})")
        
        return img
    
    def enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            img: OpenCV image (BGR)
        
        Returns:
            Contrast-enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(
            clipLimit=self.config['contrast_clip_limit'],
            tileGridSize=(8, 8)
        )
        l = clahe.apply(l)
        
        # Merge and convert back
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        logger.debug("Applied CLAHE contrast enhancement")
        return enhanced
    
    def _default_config(self) -> Dict:
        """Default preprocessing configuration."""
        return {
            'target_dpi': 300,
            'deskew_threshold': 0.5,  # degrees
            'denoise_strength': 75,
            'contrast_clip_limit': 2.0
        }

