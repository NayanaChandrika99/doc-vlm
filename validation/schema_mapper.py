"""
Schema Mapper - Convert raw extractions to typed, structured schemas

Maps olmOCR outputs to target medical record schema.
Handles type conversion, field mapping, and defaults.
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Checkbox:
    """Structured checkbox representation."""
    label: str
    checked: bool
    confidence: float = 1.0


@dataclass
class MedicalIntakeForm:
    """Target schema for medical intake forms."""
    patient_id: Optional[str] = None
    date: Optional[str] = None
    symptoms: List[Checkbox] = field(default_factory=list)
    diagnoses: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    vital_signs: Dict[str, float] = field(default_factory=dict)
    signature_present: bool = False
    
    # Metadata
    confidence: float = 0.0
    extracted_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    version: str = "1.0"


class SchemaMapper:
    """
    Map raw extractions to structured schemas.
    
    Responsibilities:
    - Field mapping (raw key → schema field)
    - Type conversion (str → int/float/bool/date)
    - Default value handling
    - Schema validation
    """
    
    def __init__(self, schema_definition: Dict = None):
        """
        Initialize schema mapper.
        
        Args:
            schema_definition: Optional custom schema definition
        """
        self.schema = schema_definition or self._default_schema()
    
    def map_to_schema(
        self,
        raw_extraction: Dict,
        schema_name: str = "medical_intake"
    ) -> Dict[str, Any]:
        """
        Map raw extraction to target schema.
        
        Args:
            raw_extraction: Raw olmOCR output
            schema_name: Target schema name
        
        Returns:
            Structured, typed data matching schema
        """
        if schema_name == "medical_intake":
            return self._map_medical_intake(raw_extraction)
        else:
            raise ValueError(f"Unknown schema: {schema_name}")
    
    def _map_medical_intake(self, raw: Dict) -> Dict:
        """Map to MedicalIntakeForm schema."""
        # Create form instance
        form = MedicalIntakeForm()
        
        # Direct field mappings
        form.patient_id = self._extract_field(raw, ['patient_id', 'patientId', 'id'])
        form.date = self._extract_field(raw, ['date', 'visitDate', 'intake_date'])
        
        # Checkbox mapping
        if 'checkboxes' in raw:
            form.symptoms = [
                Checkbox(
                    label=cb.get('label', ''),
                    checked=cb.get('checked', False),
                    confidence=cb.get('confidence', 1.0)
                )
                for cb in raw['checkboxes']
            ]
        
        # List field mappings
        if 'diagnoses' in raw:
            form.diagnoses = self._to_list(raw['diagnoses'])
        
        if 'medications' in raw:
            form.medications = self._to_list(raw['medications'])
        
        # Dict field mapping
        if 'vital_signs' in raw:
            form.vital_signs = self._map_vital_signs(raw['vital_signs'])
        
        # Boolean field
        form.signature_present = raw.get('signature_present', False)
        
        # Metadata
        form.confidence = raw.get('confidence', 0.0)
        
        return asdict(form)
    
    def _extract_field(
        self,
        data: Dict,
        possible_keys: List[str]
    ) -> Optional[str]:
        """Extract field trying multiple possible keys."""
        for key in possible_keys:
            if key in data and data[key]:
                return str(data[key])
        return None
    
    def _to_list(self, value: Any) -> List[str]:
        """Convert value to list of strings."""
        if isinstance(value, list):
            return [str(v) for v in value]
        elif isinstance(value, str):
            # Split by comma
            return [s.strip() for s in value.split(',') if s.strip()]
        else:
            return [str(value)]
    
    def _map_vital_signs(self, data: Any) -> Dict[str, float]:
        """Map vital signs to typed dict."""
        if not isinstance(data, dict):
            return {}
        
        vital_signs = {}
        
        # Standard vital sign keys
        mappings = {
            'temperature': ['temp', 'temperature', 'body_temp'],
            'blood_pressure_systolic': ['bp_sys', 'systolic', 'bp_systolic'],
            'blood_pressure_diastolic': ['bp_dia', 'diastolic', 'bp_diastolic'],
            'heart_rate': ['hr', 'heart_rate', 'pulse'],
            'respiratory_rate': ['rr', 'resp_rate', 'respiratory_rate'],
            'oxygen_saturation': ['o2', 'spo2', 'oxygen_sat']
        }
        
        for standard_key, possible_keys in mappings.items():
            for key in possible_keys:
                if key in data:
                    try:
                        vital_signs[standard_key] = float(data[key])
                        break
                    except (ValueError, TypeError):
                        logger.warning(f"Failed to convert {key}={data[key]} to float")
        
        return vital_signs
    
    def _default_schema(self) -> Dict:
        """Default schema definition."""
        return {
            "medical_intake": {
                "fields": [
                    {"name": "patient_id", "type": "string", "required": True},
                    {"name": "date", "type": "date", "required": True},
                    {"name": "symptoms", "type": "checkbox_list"},
                    {"name": "diagnoses", "type": "string_list"},
                    {"name": "medications", "type": "string_list"},
                    {"name": "vital_signs", "type": "dict"},
                    {"name": "signature_present", "type": "boolean"}
                ]
            }
        }

