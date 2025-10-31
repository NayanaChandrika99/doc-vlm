"""
Validation Rule Engine - Domain-specific validation rules

Detects errors, inconsistencies, and improbable values.
Rules are configurable, versioned, and composable.
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from datetime import datetime
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of applying validation rule."""
    rule_id: str
    passed: bool
    message: Optional[str] = None
    severity: str = "error"  # error, warning, info
    field: Optional[str] = None


class ValidationRule(ABC):
    """Base class for validation rules."""
    
    def __init__(self, rule_id: str, severity: str = "error"):
        self.rule_id = rule_id
        self.severity = severity
    
    @abstractmethod
    def validate(self, data: Dict) -> ValidationResult:
        """Apply validation rule to data."""
        pass


class DateFormatRule(ValidationRule):
    """Validate date format (MM/DD/YYYY)."""
    
    def validate(self, data: Dict) -> ValidationResult:
        date_str = data.get('date', '')
        
        # Check format
        pattern = r'^\d{2}/\d{2}/\d{4}$'
        if not re.match(pattern, date_str):
            return ValidationResult(
                rule_id=self.rule_id,
                passed=False,
                message=f"Invalid date format: {date_str} (expected MM/DD/YYYY)",
                severity=self.severity,
                field='date'
            )
        
        # Check validity
        try:
            datetime.strptime(date_str, '%m/%d/%Y')
        except ValueError:
            return ValidationResult(
                rule_id=self.rule_id,
                passed=False,
                message=f"Invalid date value: {date_str}",
                severity=self.severity,
                field='date'
            )
        
        return ValidationResult(
            rule_id=self.rule_id,
            passed=True,
            severity=self.severity
        )


class CheckboxConsistencyRule(ValidationRule):
    """Validate checkbox logical consistency."""
    
    def __init__(self, rule_id: str = "checkbox_consistency"):
        super().__init__(rule_id)
        
        # Mutually exclusive groups
        self.exclusive_groups = [
            ["Yes", "No"],
            ["Male", "Female"]
        ]
    
    def validate(self, data: Dict) -> ValidationResult:
        checkboxes = data.get('checkboxes', [])
        
        # Check for mutually exclusive selections
        for group in self.exclusive_groups:
            checked_in_group = [
                cb for cb in checkboxes
                if cb.get('label') in group and cb.get('checked')
            ]
            
            if len(checked_in_group) > 1:
                labels = [cb['label'] for cb in checked_in_group]
                return ValidationResult(
                    rule_id=self.rule_id,
                    passed=False,
                    message=f"Mutually exclusive checkboxes selected: {labels}",
                    severity="error",
                    field='checkboxes'
                )
        
        return ValidationResult(
            rule_id=self.rule_id,
            passed=True,
            severity=self.severity
        )


class RequiredFieldRule(ValidationRule):
    """Validate required fields are present and non-empty."""
    
    def __init__(
        self,
        required_fields: List[str],
        rule_id: str = "required_fields"
    ):
        super().__init__(rule_id)
        self.required_fields = required_fields
    
    def validate(self, data: Dict) -> ValidationResult:
        missing = []
        
        for field in self.required_fields:
            if field not in data or not data[field]:
                missing.append(field)
        
        if missing:
            return ValidationResult(
                rule_id=self.rule_id,
                passed=False,
                message=f"Missing required fields: {missing}",
                severity="error",
                field=missing[0] if missing else None
            )
        
        return ValidationResult(
            rule_id=self.rule_id,
            passed=True,
            severity=self.severity
        )


class RangeValidationRule(ValidationRule):
    """Validate numeric values are in expected range."""
    
    def __init__(
        self,
        field: str,
        min_val: float,
        max_val: float,
        rule_id: str = None
    ):
        rule_id = rule_id or f"range_{field}"
        super().__init__(rule_id)
        self.field = field
        self.min_val = min_val
        self.max_val = max_val
    
    def validate(self, data: Dict) -> ValidationResult:
        if self.field not in data:
            # Field not present, skip validation
            return ValidationResult(
                rule_id=self.rule_id,
                passed=True,
                severity=self.severity
            )
        
        value = data[self.field]
        
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            return ValidationResult(
                rule_id=self.rule_id,
                passed=False,
                message=f"Non-numeric value for {self.field}: {value}",
                severity="error",
                field=self.field
            )
        
        if not (self.min_val <= numeric_value <= self.max_val):
            return ValidationResult(
                rule_id=self.rule_id,
                passed=False,
                message=f"Value {numeric_value} out of range [{self.min_val}, {self.max_val}]",
                severity="warning",
                field=self.field
            )
        
        return ValidationResult(
            rule_id=self.rule_id,
            passed=True,
            severity=self.severity
        )


class ValidationEngine:
    """
    Orchestrate validation rule application.
    
    Supports rule composition, priority, and fail-fast/fail-slow modes.
    """
    
    def __init__(self, rules: List[ValidationRule] = None):
        """
        Initialize validation engine.
        
        Args:
            rules: List of validation rules to apply
        """
        self.rules = rules or self._default_rules()
    
    def validate(
        self,
        data: Dict,
        fail_fast: bool = False
    ) -> Dict[str, Any]:
        """
        Apply all validation rules to data.
        
        Args:
            data: Data to validate
            fail_fast: Stop at first failure
        
        Returns:
            Dict with 'valid', 'errors', 'warnings', 'pass_rate'
        """
        results = []
        
        for rule in self.rules:
            try:
                result = rule.validate(data)
                results.append(result)
                
                if fail_fast and not result.passed:
                    break
                    
            except Exception as e:
                logger.error(f"Rule {rule.rule_id} failed: {e}")
                results.append(ValidationResult(
                    rule_id=rule.rule_id,
                    passed=False,
                    message=f"Rule execution error: {e}",
                    severity="error"
                ))
        
        # Separate by severity
        errors = [r for r in results if not r.passed and r.severity == "error"]
        warnings = [r for r in results if not r.passed and r.severity == "warning"]
        
        # Compute pass rate
        total = len(results)
        passed = len([r for r in results if r.passed])
        pass_rate = passed / total if total > 0 else 1.0
        
        return {
            "valid": len(errors) == 0,
            "pass_rate": pass_rate,
            "errors": [
                {"rule": e.rule_id, "message": e.message, "field": e.field}
                for e in errors
            ],
            "warnings": [
                {"rule": w.rule_id, "message": w.message, "field": w.field}
                for w in warnings
            ],
            "total_rules": total,
            "passed_rules": passed
        }
    
    def _default_rules(self) -> List[ValidationRule]:
        """Default validation rules for medical forms."""
        return [
            RequiredFieldRule(['date', 'patient_id']),
            DateFormatRule('date_format_check'),
            CheckboxConsistencyRule()
        ]

