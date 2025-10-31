#!/usr/bin/env python3
"""
Synthetic Medical Form Generator

Generates synthetic medical forms with checkboxes, tables, and text fields.
Supports difficulty levels: easy (clean), medium (noisy), hard (poor quality).
"""
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import json
from dataclasses import dataclass, asdict
import random
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib import colors
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np


@dataclass
class CheckboxAnnotation:
    x: int
    y: int
    width: int
    height: int
    label: str
    checked: bool
    confidence: float = 1.0  # Ground truth confidence


class SyntheticFormGenerator:
    """Generate synthetic medical forms with controlled characteristics."""
    
    # Medical form field templates
    SYMPTOM_OPTIONS = [
        "Fever", "Chills", "Cough", "Shortness of Breath",
        "Fatigue", "Muscle Aches", "Headache", "Loss of Taste/Smell",
        "Sore Throat", "Congestion", "Nausea", "Diarrhea"
    ]
    
    DIAGNOSIS_OPTIONS = [
        "Hypertension", "Diabetes", "Asthma", "COPD",
        "Heart Disease", "Kidney Disease", "Cancer", "None"
    ]
    
    def __init__(self, output_dir: Path):
        """
        Initialize generator.
        
        Args:
            output_dir: Directory to save generated forms
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_batch(
        self,
        count: int,
        difficulty_distribution: Dict[str, float] = None
    ) -> List[Tuple[Path, Dict]]:
        """
        Generate batch of synthetic forms with annotations.
        
        Args:
            count: Number of forms to generate
            difficulty_distribution: {"easy": 0.6, "medium": 0.3, "hard": 0.1}
        
        Returns:
            List of (image_path, annotation_dict) tuples
        """
        if difficulty_distribution is None:
            difficulty_distribution = {"easy": 0.6, "medium": 0.3, "hard": 0.1}
        
        results = []
        difficulties = list(difficulty_distribution.keys())
        weights = list(difficulty_distribution.values())
        
        for i in range(count):
            difficulty = random.choices(difficulties, weights=weights)[0]
            
            form_id = f"synthetic_{i:05d}"
            img_path, annotation = self.generate_form(form_id, difficulty)
            results.append((img_path, annotation))
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{count} forms...")
        
        print(f"✓ Generated {count} synthetic forms in {self.output_dir}")
        return results
    
    def generate_form(
        self,
        form_id: str,
        difficulty: str = "medium"
    ) -> Tuple[Path, Dict]:
        """
        Generate single form with specified difficulty.
        
        Difficulty factors:
        - easy: Clean scan, clear checkboxes, good contrast
        - medium: Some noise, moderate density
        - hard: Poor scan, dense layout, low contrast, artifacts
        
        Args:
            form_id: Unique form identifier
            difficulty: "easy", "medium", or "hard"
        
        Returns:
            (image_path, annotation_dict)
        """
        # Create PDF
        pdf_path = self.output_dir / f"{form_id}.pdf"
        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        width, height = letter
        
        # Form header
        c.setFont("Helvetica-Bold", 16)
        c.drawString(inch, height - inch, "Medical Intake Form")
        
        # Patient info section
        y_pos = height - 2*inch
        c.setFont("Helvetica", 12)
        c.drawString(inch, y_pos, f"Patient ID: {random.randint(10000, 99999)}")
        y_pos -= 0.3*inch
        c.drawString(inch, y_pos, f"Date: {datetime.now().strftime('%m/%d/%Y')}")
        y_pos -= 0.8*inch
        
        # Symptom checklist
        c.setFont("Helvetica-Bold", 14)
        c.drawString(inch, y_pos, "Current Symptoms (check all that apply):")
        y_pos -= 0.5*inch
        
        # Generate checkboxes
        checkboxes = []
        symptom_selection = random.sample(
            self.SYMPTOM_OPTIONS,
            k=random.randint(0, len(self.SYMPTOM_OPTIONS))
        )
        
        c.setFont("Helvetica", 10)
        checkbox_size = 12
        
        for symptom in self.SYMPTOM_OPTIONS:
            checked = symptom in symptom_selection
            
            # Draw checkbox
            c.rect(inch, y_pos - checkbox_size, checkbox_size, checkbox_size)
            
            if checked:
                # Draw checkmark
                c.line(
                    inch + 2, y_pos - checkbox_size + 2,
                    inch + 5, y_pos - checkbox_size + 10
                )
                c.line(
                    inch + 5, y_pos - checkbox_size + 10,
                    inch + 10, y_pos - checkbox_size + 2
                )
            
            # Draw label
            c.drawString(inch + checkbox_size + 10, y_pos - checkbox_size + 2, symptom)
            
            # Record annotation (convert to image coordinates later)
            checkboxes.append(CheckboxAnnotation(
                x=int(inch * 72),
                y=int(y_pos - checkbox_size),
                width=checkbox_size,
                height=checkbox_size,
                label=symptom,
                checked=checked
            ))
            
            y_pos -= 0.4*inch
        
        c.save()
        
        # Convert PDF to image with difficulty-appropriate artifacts
        img_path = self._pdf_to_image(pdf_path, difficulty)
        
        # Create annotation
        annotation = {
            "form_id": form_id,
            "checkboxes": [asdict(cb) for cb in checkboxes],
            "difficulty": difficulty,
            "region_type": "checkbox_group",
            "source": "synthetic",
            "metadata": {
                "symptom_count": len(symptom_selection),
                "generated_at": datetime.utcnow().isoformat()
            }
        }
        
        # Save annotation
        ann_path = self.output_dir / f"{form_id}.json"
        with open(ann_path, "w") as f:
            json.dump(annotation, f, indent=2)
        
        return img_path, annotation
    
    def _pdf_to_image(self, pdf_path: Path, difficulty: str) -> Path:
        """
        Convert PDF to image with difficulty-appropriate artifacts.
        
        Args:
            pdf_path: Path to PDF file
            difficulty: "easy", "medium", or "hard"
        
        Returns:
            Path to generated PNG image
        """
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(str(pdf_path), dpi=300)
            img = images[0]
        except ImportError:
            # Fallback: create blank image if pdf2image not available
            print("Warning: pdf2image not available, using blank placeholder")
            img = Image.new('RGB', (2550, 3300), color='white')
        
        # Apply difficulty-appropriate transformations
        if difficulty == "medium":
            # Add slight noise
            img = self._add_noise(img, amount=0.05)
            # Slight blur
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        elif difficulty == "hard":
            # Heavy noise
            img = self._add_noise(img, amount=0.15)
            # More blur
            img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
            # Reduce contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(0.7)
            # Add skew
            angle = random.uniform(-2, 2)
            img = img.rotate(angle, fillcolor="white", expand=False)
        
        # Save image
        img_path = pdf_path.with_suffix(".png")
        img.save(img_path)
        
        return img_path
    
    def _add_noise(self, img: Image.Image, amount: float = 0.1) -> Image.Image:
        """Add Gaussian noise to image."""
        arr = np.array(img)
        noise = np.random.normal(0, amount * 255, arr.shape).astype(np.int16)
        noisy = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic medical forms"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of forms to generate (default: 100)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/raw/synthetic_v1"),
        help="Output directory (default: datasets/raw/synthetic_v1)"
    )
    parser.add_argument(
        "--easy",
        type=float,
        default=0.6,
        help="Proportion of easy forms (default: 0.6)"
    )
    parser.add_argument(
        "--medium",
        type=float,
        default=0.3,
        help="Proportion of medium forms (default: 0.3)"
    )
    parser.add_argument(
        "--hard",
        type=float,
        default=0.1,
        help="Proportion of hard forms (default: 0.1)"
    )
    
    args = parser.parse_args()
    
    # Normalize distribution
    total = args.easy + args.medium + args.hard
    distribution = {
        "easy": args.easy / total,
        "medium": args.medium / total,
        "hard": args.hard / total
    }
    
    print(f"Generating {args.count} forms with distribution: {distribution}")
    
    generator = SyntheticFormGenerator(args.output)
    results = generator.generate_batch(
        count=args.count,
        difficulty_distribution=distribution
    )
    
    print(f"\n✓ Complete! Generated {len(results)} forms at {args.output}")
    print(f"  - Easy: {int(args.count * distribution['easy'])}")
    print(f"  - Medium: {int(args.count * distribution['medium'])}")
    print(f"  - Hard: {int(args.count * distribution['hard'])}")


if __name__ == "__main__":
    main()

