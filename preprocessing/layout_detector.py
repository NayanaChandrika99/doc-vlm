"""
CV Layout Detection - OpenCV + PyMuPDF based region detection

NO deep learning models. Fast, deterministic, offline.
Detects: checkboxes, tables, text blocks, signatures.
"""
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class Region:
    """Detected document region."""
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    region_type: str  # checkbox, table, text_block, signature, complex
    confidence: float  # Detection confidence (0-1)
    difficulty_score: float  # Processing difficulty (0-1)
    priority: int  # Processing priority (0=highest)
    metadata: Dict


class CVLayoutDetector:
    """
    Fast CV-based layout detection.
    
    Handles 70-80% of cases. Falls back to olmOCR for complex layouts.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize layout detector.
        
        Args:
            config: Detection parameters
        """
        self.config = config or self._default_config()
        
        # Checkbox detection parameters
        self.checkbox_min_area = 100
        self.checkbox_max_area = 400
        self.checkbox_aspect_ratio_range = (0.8, 1.2)  # Nearly square
        
        # Table detection parameters
        self.min_line_length = 50
        self.max_line_gap = 10
        
        logger.info("CVLayoutDetector initialized")
    
    def detect_regions(
        self,
        image_path: Path,
        pdf_path: Optional[Path] = None
    ) -> List[Region]:
        """
        Detect all regions in document page.
        
        Args:
            image_path: Path to page image
            pdf_path: Optional path to source PDF for text extraction
        
        Returns:
            List of detected regions sorted by priority
        """
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return []
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        regions = []
        
        # 1. Detect text blocks (PyMuPDF - requires PDF)
        if pdf_path and pdf_path.exists():
            try:
                text_regions = self._detect_text_blocks_pymupdf(pdf_path)
                regions.extend(text_regions)
            except Exception as e:
                logger.warning(f"PyMuPDF text detection failed: {e}")
        
        # 2. Detect checkboxes (OpenCV contours)
        checkbox_regions = self._detect_checkboxes(gray, img)
        regions.extend(checkbox_regions)
        
        # 3. Detect tables (Hough line detection)
        table_regions = self._detect_tables(gray, img)
        regions.extend(table_regions)
        
        # 4. Detect signatures (connected components)
        signature_regions = self._detect_signatures(gray, img)
        regions.extend(signature_regions)
        
        # 5. Check for complex/undetected areas
        complex_regions = self._detect_complex_areas(gray, regions)
        regions.extend(complex_regions)
        
        # Remove overlapping regions (keep higher confidence)
        regions = self._remove_overlaps(regions)
        
        # Score difficulty and priority
        regions = self._score_regions(regions, img)
        
        # Sort by priority
        regions.sort(key=lambda r: (r.priority, -r.confidence))
        
        logger.info(f"Detected {len(regions)} regions")
        return regions
    
    def _detect_text_blocks_pymupdf(self, pdf_path: Path) -> List[Region]:
        """Extract text blocks using PyMuPDF (fast, accurate)."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning("PyMuPDF not available, skipping text block detection")
            return []
        
        regions = []
        
        try:
            doc = fitz.open(str(pdf_path))
            for page_num, page in enumerate(doc):
                blocks = page.get_text("dict")["blocks"]
                
                for block in blocks:
                    if block.get("type") == 0:  # Text block
                        bbox = block["bbox"]
                        x, y, x2, y2 = bbox
                        w, h = x2 - x, y2 - y
                        
                        # Filter small blocks
                        if w * h < 500:
                            continue
                        
                        regions.append(Region(
                            bbox=(int(x), int(y), int(w), int(h)),
                            region_type="text_block",
                            confidence=0.95,  # PyMuPDF is reliable
                            difficulty_score=0.2,  # Text is easy
                            priority=2,  # Lower priority than checkboxes
                            metadata={"page": page_num, "method": "pymupdf"}
                        ))
            
            doc.close()
        except Exception as e:
            logger.error(f"PyMuPDF processing failed: {e}")
        
        return regions
    
    def _detect_checkboxes(self, gray: np.ndarray, img: np.ndarray) -> List[Region]:
        """Detect checkbox candidates using contour analysis."""
        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        regions = []
        
        for contour in contours:
            # Compute bounding box
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Filter by size
            if not (self.checkbox_min_area < area < self.checkbox_max_area):
                continue
            
            # Filter by aspect ratio (checkboxes are square-ish)
            aspect_ratio = w / h if h > 0 else 0
            if not (self.checkbox_aspect_ratio_range[0] < aspect_ratio < 
                    self.checkbox_aspect_ratio_range[1]):
                continue
            
            # Check for square-like shape
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            
            if len(approx) == 4:
                confidence = 0.8
            else:
                confidence = 0.5
            
            # Check if checkbox is filled/checked
            roi = gray[y:y+h, x:x+w]
            mean_intensity = roi.mean() if roi.size > 0 else 255
            checked = mean_intensity < 200
            
            regions.append(Region(
                bbox=(x, y, w, h),
                region_type="checkbox",
                confidence=confidence,
                difficulty_score=0.3,  # Checkboxes moderate difficulty
                priority=0,  # Highest priority
                metadata={
                    "checked": checked,
                    "intensity": float(mean_intensity),
                    "method": "contour"
                }
            ))
        
        return regions
    
    def _detect_tables(self, gray: np.ndarray, img: np.ndarray) -> List[Region]:
        """Detect tables using Hough line detection."""
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=100,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        if lines is None:
            return []
        
        # Separate horizontal and vertical lines
        h_lines = []
        v_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if angle < 10 or angle > 170:  # Horizontal
                h_lines.append((x1, y1, x2, y2))
            elif 80 < angle < 100:  # Vertical
                v_lines.append((x1, y1, x2, y2))
        
        # If we have grid-like structure, it's likely a table
        if len(h_lines) >= 2 and len(v_lines) >= 2:
            # Compute bounding box
            all_x = [x for line in h_lines + v_lines for x in [line[0], line[2]]]
            all_y = [y for line in h_lines + v_lines for y in [line[1], line[3]]]
            
            x, y = min(all_x), min(all_y)
            x2, y2 = max(all_x), max(all_y)
            w, h = x2 - x, y2 - y
            
            return [Region(
                bbox=(x, y, w, h),
                region_type="table",
                confidence=0.7,
                difficulty_score=0.5,  # Tables are harder
                priority=1,
                metadata={
                    "h_lines": len(h_lines),
                    "v_lines": len(v_lines),
                    "method": "hough"
                }
            )]
        
        return []
    
    def _detect_signatures(self, gray: np.ndarray, img: np.ndarray) -> List[Region]:
        """Detect signature regions (handwriting, irregular shapes)."""
        # Binary threshold
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        
        regions = []
        img_height = img.shape[0]
        
        for i in range(1, num_labels):  # Skip background
            x, y, w, h, area = stats[i]
            
            # Signatures are typically medium-sized
            if not (500 < area < 10000):
                continue
            
            # Signatures have irregular aspect ratios
            aspect_ratio = w / h if h > 0 else 0
            if 0.3 < aspect_ratio < 0.7 or aspect_ratio > 2:
                continue
            
            # Check if in bottom third (common signature location)
            if y > img_height * 0.66:
                confidence = 0.6
            else:
                confidence = 0.4
            
            regions.append(Region(
                bbox=(x, y, w, h),
                region_type="signature",
                confidence=confidence,
                difficulty_score=0.7,  # Signatures are hard
                priority=3,  # Lower priority
                metadata={"area": int(area), "method": "connected_components"}
            ))
        
        return regions
    
    def _detect_complex_areas(
        self,
        gray: np.ndarray,
        detected_regions: List[Region]
    ) -> List[Region]:
        """Identify areas not covered by specific detectors."""
        h, w = gray.shape
        
        # Create mask of detected regions
        mask = np.zeros((h, w), dtype=np.uint8)
        for region in detected_regions:
            x, y, rw, rh = region.bbox
            mask[y:y+rh, x:x+rw] = 255
        
        # Find undetected areas with content
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        undetected = cv2.bitwise_and(binary, cv2.bitwise_not(mask))
        
        # Find contours in undetected areas
        contours, _ = cv2.findContours(
            undetected,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        regions = []
        for contour in contours:
            x, y, rw, rh = cv2.boundingRect(contour)
            area = rw * rh
            
            # Only care about substantial undetected areas
            if area > 5000:
                regions.append(Region(
                    bbox=(x, y, rw, rh),
                    region_type="complex",
                    confidence=0.0,  # Unknown, needs olmOCR
                    difficulty_score=0.8,  # Assume hard
                    priority=0,  # High priority to resolve
                    metadata={"reason": "undetected_content", "area": int(area)}
                ))
        
        return regions
    
    def _remove_overlaps(self, regions: List[Region]) -> List[Region]:
        """Remove overlapping regions, keeping higher confidence ones."""
        regions = sorted(regions, key=lambda r: -r.confidence)
        kept = []
        
        for region in regions:
            overlap = False
            for kept_region in kept:
                if self._iou(region.bbox, kept_region.bbox) > 0.3:
                    overlap = True
                    break
            
            if not overlap:
                kept.append(region)
        
        return kept
    
    def _iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Compute Intersection over Union of two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Compute intersection
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Compute union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def _score_regions(self, regions: List[Region], img: np.ndarray) -> List[Region]:
        """Compute difficulty scores and priorities for regions."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        for region in regions:
            x, y, w, h = region.bbox
            
            # Ensure valid ROI
            x, y = max(0, x), max(0, y)
            w = min(w, gray.shape[1] - x)
            h = min(h, gray.shape[0] - y)
            
            if w <= 0 or h <= 0:
                continue
            
            roi = gray[y:y+h, x:x+w]
            
            # Visual complexity (edge density)
            edges = cv2.Canny(roi, 50, 150)
            edge_density = edges.sum() / (w * h * 255) if (w * h) > 0 else 0
            
            # Contrast
            contrast = roi.std() / 255 if roi.size > 0 else 0
            
            # Sharpness (Laplacian variance)
            laplacian = cv2.Laplacian(roi, cv2.CV_64F)
            sharpness = laplacian.var() / 1000
            
            # Combine factors
            difficulty = (
                0.3 * edge_density +
                0.3 * (1 - contrast) +  # Low contrast = harder
                0.2 * (1 - min(sharpness, 1.0)) +  # Blur = harder
                0.2 * self._type_difficulty(region.region_type)
            )
            
            region.difficulty_score = float(np.clip(difficulty, 0, 1))
        
        return regions
    
    def _type_difficulty(self, region_type: str) -> float:
        """Base difficulty by region type."""
        return {
            "text_block": 0.2,
            "checkbox": 0.3,
            "table": 0.5,
            "signature": 0.7,
            "complex": 0.8
        }.get(region_type, 0.5)
    
    def _default_config(self) -> Dict:
        return {
            "checkbox_detection": True,
            "table_detection": True,
            "signature_detection": True,
            "use_pymupdf": True
        }

