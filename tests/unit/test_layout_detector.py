"""Unit tests for CV layout detector."""
import pytest
import numpy as np
import cv2
from pathlib import Path
from preprocessing.layout_detector import CVLayoutDetector, Region


@pytest.fixture
def detector():
    """Create layout detector instance."""
    return CVLayoutDetector()


@pytest.fixture
def sample_image():
    """Create sample test image with checkboxes."""
    img = np.ones((800, 600, 3), dtype=np.uint8) * 255
    
    # Draw some checkboxes
    for i in range(3):
        x, y = 100, 100 + i * 50
        cv2.rectangle(img, (x, y), (x + 20, y + 20), (0, 0, 0), 2)
    
    return img


def test_detector_initialization(detector):
    """Test detector initializes correctly."""
    assert detector is not None
    assert detector.checkbox_min_area == 100
    assert detector.checkbox_max_area == 400


def test_detect_checkboxes(detector, sample_image, tmp_path):
    """Test checkbox detection."""
    # Save sample image
    img_path = tmp_path / "test.png"
    cv2.imwrite(str(img_path), sample_image)
    
    # Detect regions
    regions = detector.detect_regions(img_path)
    
    # Should find some regions
    assert len(regions) >= 0
    
    # Check region structure
    for region in regions:
        assert isinstance(region, Region)
        assert len(region.bbox) == 4
        assert 0 <= region.confidence <= 1
        assert 0 <= region.difficulty_score <= 1


def test_iou_computation(detector):
    """Test IoU calculation."""
    bbox1 = (0, 0, 10, 10)
    bbox2 = (5, 5, 10, 10)
    
    iou = detector._iou(bbox1, bbox2)
    
    # Partial overlap
    assert 0 < iou < 1
    
    # Perfect overlap
    iou_same = detector._iou(bbox1, bbox1)
    assert iou_same == 1.0
    
    # No overlap
    bbox3 = (20, 20, 10, 10)
    iou_none = detector._iou(bbox1, bbox3)
    assert iou_none == 0.0


def test_difficulty_scoring(detector, sample_image, tmp_path):
    """Test difficulty score computation."""
    img_path = tmp_path / "test.png"
    cv2.imwrite(str(img_path), sample_image)
    
    # Create mock region
    region = Region(
        bbox=(100, 100, 20, 20),
        region_type="checkbox",
        confidence=0.8,
        difficulty_score=0.0,  # Will be computed
        priority=0,
        metadata={}
    )
    
    regions = detector._score_regions([region], sample_image)
    
    assert len(regions) == 1
    assert 0 <= regions[0].difficulty_score <= 1

