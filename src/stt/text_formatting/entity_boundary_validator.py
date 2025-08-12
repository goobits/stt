#!/usr/bin/env python3
"""
Entity Boundary Validation System

PHASE 23: Entity Boundary Validation Infrastructure

This module provides comprehensive boundary validation to prevent entity overlap
conflicts and improve system robustness. It focuses on VALIDATION/ROBUSTNESS
without changing entity detection or conversion functionality.

Key Features:
- Entity boundary overlap detection and validation  
- Conflict resolution through validation warnings
- Boundary integrity checking throughout processing pipeline
- Logging for boundary issues without breaking functionality
- Prevention of Phase 12-style boundary regressions
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

from .common import Entity, EntityType
from .utils import overlaps_with_entity, is_inside_entity
from ..core.config import setup_logging

logger = setup_logging(__name__)


class BoundaryConflictType(Enum):
    """Types of boundary conflicts that can be detected."""
    OVERLAP = "overlap"
    CONTAINMENT = "containment"
    INVALID_BOUNDS = "invalid_bounds"
    BOUNDARY_MISMATCH = "boundary_mismatch"
    ZERO_LENGTH = "zero_length"


@dataclass
class BoundaryConflict:
    """Represents a boundary conflict between entities."""
    type: BoundaryConflictType
    entity1: Entity
    entity2: Optional[Entity] = None
    message: str = ""
    severity: str = "warning"  # warning, error, critical
    
    def __post_init__(self):
        if not self.message:
            if self.entity2:
                self.message = f"{self.type.value}: {self.entity1.type} at {self.entity1.start}-{self.entity1.end} conflicts with {self.entity2.type} at {self.entity2.start}-{self.entity2.end}"
            else:
                self.message = f"{self.type.value}: {self.entity1.type} at {self.entity1.start}-{self.entity1.end}"


@dataclass
class ValidationResult:
    """Result of entity boundary validation."""
    is_valid: bool
    conflicts: List[BoundaryConflict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validated_entities: List[Entity] = field(default_factory=list)
    
    def add_conflict(self, conflict: BoundaryConflict):
        """Add a boundary conflict to the result."""
        self.conflicts.append(conflict)
        if conflict.severity in ["error", "critical"]:
            self.is_valid = False
    
    def add_warning(self, message: str):
        """Add a validation warning."""
        self.warnings.append(message)


class EntityBoundaryValidator:
    """
    PHASE 23: Comprehensive entity boundary validation system.
    
    This validator provides boundary validation infrastructure to prevent
    entity overlap conflicts and improve robustness without changing
    entity detection or conversion functionality.
    """
    
    def __init__(self, strict_mode: bool = False, language: str = "en"):
        """
        Initialize the boundary validator.
        
        Args:
            strict_mode: If True, treat warnings as errors
            language: Language code for resources
        """
        self.strict_mode = strict_mode
        self.language = language
        self.logger = setup_logging(__name__)
        
        # Statistics for monitoring
        self._validation_stats = {
            "validations_performed": 0,
            "conflicts_detected": 0,
            "entities_rejected": 0,
            "overlap_conflicts": 0,
            "containment_conflicts": 0,
            "boundary_errors": 0
        }
    
    def validate_entity_boundaries(self, entities: List[Entity], 
                                 text: str, 
                                 step: str = "unknown") -> ValidationResult:
        """
        Validate entity boundaries for a list of entities.
        
        Args:
            entities: List of entities to validate
            text: Source text
            step: Processing step for logging context
            
        Returns:
            ValidationResult with validation outcome and any conflicts
        """
        self._validation_stats["validations_performed"] += 1
        
        result = ValidationResult(is_valid=True)
        
        try:
            # Validate individual entity boundaries
            valid_entities = []
            for entity in entities:
                entity_result = self._validate_single_entity(entity, text)
                
                if entity_result.is_valid:
                    valid_entities.append(entity)
                else:
                    result.conflicts.extend(entity_result.conflicts)
                    result.warnings.extend(entity_result.warnings)
                    if self.strict_mode:
                        result.is_valid = False
                    self._validation_stats["entities_rejected"] += 1
            
            # Validate for overlaps and containment conflicts
            if valid_entities:
                overlap_result = self._validate_entity_overlaps(valid_entities, text)
                result.conflicts.extend(overlap_result.conflicts)
                result.warnings.extend(overlap_result.warnings)
                
                if not overlap_result.is_valid and self.strict_mode:
                    result.is_valid = False
            
            result.validated_entities = valid_entities
            
            # Log validation results
            if result.conflicts:
                self._validation_stats["conflicts_detected"] += len(result.conflicts)
                self.logger.debug(f"BOUNDARY_VALIDATION [{step}]: Found {len(result.conflicts)} conflicts in {len(entities)} entities")
                
                for conflict in result.conflicts:
                    self.logger.debug(f"  - {conflict.message}")
            
        except Exception as e:
            result.is_valid = False
            result.add_warning(f"Boundary validation failed in {step}: {e}")
            self.logger.error(f"Entity boundary validation error in {step}: {e}")
        
        return result
    
    def validate_entity_list_consistency(self, entities: List[Entity], 
                                       text: str) -> ValidationResult:
        """
        Validate consistency across the entire entity list.
        
        Args:
            entities: List of entities to validate
            text: Source text
            
        Returns:
            ValidationResult with consistency validation outcome
        """
        result = ValidationResult(is_valid=True)
        
        try:
            # Sort entities by position for systematic validation
            sorted_entities = sorted(entities, key=lambda e: (e.start, e.end))
            
            # Check for duplicate boundaries with different types
            boundary_map = {}
            for entity in sorted_entities:
                boundary_key = (entity.start, entity.end)
                if boundary_key in boundary_map:
                    existing = boundary_map[boundary_key]
                    if existing.type != entity.type:
                        conflict = BoundaryConflict(
                            type=BoundaryConflictType.BOUNDARY_MISMATCH,
                            entity1=existing,
                            entity2=entity,
                            message=f"Duplicate boundaries with different types: {existing.type} vs {entity.type} at {entity.start}-{entity.end}",
                            severity="warning"
                        )
                        result.add_conflict(conflict)
                else:
                    boundary_map[boundary_key] = entity
            
            # Validate entity sequence for overlaps
            for i in range(len(sorted_entities) - 1):
                current = sorted_entities[i]
                next_entity = sorted_entities[i + 1]
                
                if current.end > next_entity.start:
                    conflict = BoundaryConflict(
                        type=BoundaryConflictType.OVERLAP,
                        entity1=current,
                        entity2=next_entity,
                        message=f"Entity sequence overlap: {current.type}('{current.text}') at {current.start}-{current.end} overlaps with {next_entity.type}('{next_entity.text}') at {next_entity.start}-{next_entity.end}",
                        severity="error"
                    )
                    result.add_conflict(conflict)
                    self._validation_stats["overlap_conflicts"] += 1
            
            result.validated_entities = sorted_entities
            
        except Exception as e:
            result.is_valid = False
            result.add_warning(f"Entity list consistency validation failed: {e}")
            self.logger.error(f"Entity list consistency validation error: {e}")
        
        return result
    
    def validate_boundary_integrity_after_conversion(self, 
                                                   original_entities: List[Entity],
                                                   updated_entities: List[Entity], 
                                                   original_text: str,
                                                   converted_text: str) -> ValidationResult:
        """
        Validate boundary integrity after entity conversion.
        
        This method helps prevent Phase 12-style boundary regressions by
        validating that conversions maintain proper entity boundaries.
        
        Args:
            original_entities: Entities before conversion
            updated_entities: Entities after conversion
            original_text: Text before conversion
            converted_text: Text after conversion
            
        Returns:
            ValidationResult with conversion validation outcome
        """
        result = ValidationResult(is_valid=True)
        
        try:
            # Validate that updated entities have correct boundaries in converted text
            for entity in updated_entities:
                if not self._validate_entity_in_text(entity, converted_text):
                    conflict = BoundaryConflict(
                        type=BoundaryConflictType.BOUNDARY_MISMATCH,
                        entity1=entity,
                        message=f"Post-conversion boundary mismatch: {entity.type} at {entity.start}-{entity.end} doesn't match text '{converted_text[entity.start:entity.end] if entity.end <= len(converted_text) else 'OUT_OF_BOUNDS'}'",
                        severity="error"
                    )
                    result.add_conflict(conflict)
            
            # Check for boundary drift (entities moving unexpectedly)
            drift_warnings = self._detect_boundary_drift(original_entities, updated_entities, original_text, converted_text)
            result.warnings.extend(drift_warnings)
            
            result.validated_entities = updated_entities
            
        except Exception as e:
            result.is_valid = False
            result.add_warning(f"Post-conversion boundary validation failed: {e}")
            self.logger.error(f"Post-conversion boundary validation error: {e}")
        
        return result
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics for monitoring."""
        return {
            **self._validation_stats,
            "strict_mode": self.strict_mode,
            "language": self.language
        }
    
    def reset_statistics(self):
        """Reset validation statistics."""
        for key in self._validation_stats:
            self._validation_stats[key] = 0
    
    # Private validation methods
    
    def _validate_single_entity(self, entity: Entity, text: str) -> ValidationResult:
        """Validate a single entity's boundaries."""
        result = ValidationResult(is_valid=True)
        
        # Check for basic boundary validity
        if entity.start < 0:
            conflict = BoundaryConflict(
                type=BoundaryConflictType.INVALID_BOUNDS,
                entity1=entity,
                message=f"Negative start position: {entity.start}",
                severity="error"
            )
            result.add_conflict(conflict)
            self._validation_stats["boundary_errors"] += 1
        
        if entity.end > len(text):
            conflict = BoundaryConflict(
                type=BoundaryConflictType.INVALID_BOUNDS,
                entity1=entity,
                message=f"End position {entity.end} exceeds text length {len(text)}",
                severity="error"
            )
            result.add_conflict(conflict)
            self._validation_stats["boundary_errors"] += 1
        
        if entity.start >= entity.end:
            conflict = BoundaryConflict(
                type=BoundaryConflictType.ZERO_LENGTH,
                entity1=entity,
                message=f"Invalid entity span: start {entity.start} >= end {entity.end}",
                severity="error"
            )
            result.add_conflict(conflict)
            self._validation_stats["boundary_errors"] += 1
        
        # Validate entity text matches position
        if (entity.start >= 0 and entity.end <= len(text) and 
            entity.start < entity.end):
            
            expected_text = text[entity.start:entity.end]
            if entity.text != expected_text:
                conflict = BoundaryConflict(
                    type=BoundaryConflictType.BOUNDARY_MISMATCH,
                    entity1=entity,
                    message=f"Entity text mismatch: expected '{expected_text}' but entity has '{entity.text}' at {entity.start}-{entity.end}",
                    severity="warning"
                )
                result.add_conflict(conflict)
        
        return result
    
    def _validate_entity_overlaps(self, entities: List[Entity], text: str) -> ValidationResult:
        """Validate entities for overlaps and containment."""
        result = ValidationResult(is_valid=True)
        
        # Sort entities for efficient overlap checking
        sorted_entities = sorted(entities, key=lambda e: e.start)
        
        for i, entity1 in enumerate(sorted_entities):
            for entity2 in sorted_entities[i + 1:]:
                # Check if entities overlap
                if not (entity1.end <= entity2.start or entity2.end <= entity1.start):
                    # Determine conflict type
                    if (entity1.start <= entity2.start and entity1.end >= entity2.end):
                        # entity1 contains entity2
                        conflict_type = BoundaryConflictType.CONTAINMENT
                        self._validation_stats["containment_conflicts"] += 1
                    elif (entity2.start <= entity1.start and entity2.end >= entity1.end):
                        # entity2 contains entity1
                        conflict_type = BoundaryConflictType.CONTAINMENT
                        self._validation_stats["containment_conflicts"] += 1
                    else:
                        # Partial overlap
                        conflict_type = BoundaryConflictType.OVERLAP
                        self._validation_stats["overlap_conflicts"] += 1
                    
                    conflict = BoundaryConflict(
                        type=conflict_type,
                        entity1=entity1,
                        entity2=entity2,
                        severity="warning" if conflict_type == BoundaryConflictType.CONTAINMENT else "error"
                    )
                    result.add_conflict(conflict)
        
        return result
    
    def _validate_entity_in_text(self, entity: Entity, text: str) -> bool:
        """Check if entity boundaries are valid in the given text."""
        if (entity.start < 0 or entity.end > len(text) or 
            entity.start >= entity.end):
            return False
        
        actual_text = text[entity.start:entity.end]
        return actual_text == entity.text
    
    def _detect_boundary_drift(self, original_entities: List[Entity],
                              updated_entities: List[Entity],
                              original_text: str,
                              converted_text: str) -> List[str]:
        """Detect unexpected boundary drift after conversion."""
        warnings = []
        
        try:
            # Create maps by entity type and original position for tracking
            original_map = {}
            for entity in original_entities:
                key = (entity.type, entity.start, entity.end)
                original_map[key] = entity
            
            updated_map = {}
            for entity in updated_entities:
                # Try to find corresponding original entity
                for orig_key, orig_entity in original_map.items():
                    if (orig_entity.type == entity.type and 
                        abs(orig_entity.start - entity.start) <= 5):  # Allow small drift
                        updated_map[orig_key] = entity
                        break
            
            # Check for significant boundary drift
            for orig_key, orig_entity in original_map.items():
                if orig_key in updated_map:
                    updated_entity = updated_map[orig_key]
                    start_drift = abs(updated_entity.start - orig_entity.start)
                    end_drift = abs(updated_entity.end - orig_entity.end)
                    
                    if start_drift > 10 or end_drift > 10:
                        warnings.append(
                            f"Significant boundary drift detected for {orig_entity.type}: "
                            f"start {orig_entity.start} -> {updated_entity.start} (drift: {start_drift}), "
                            f"end {orig_entity.end} -> {updated_entity.end} (drift: {end_drift})"
                        )
        
        except Exception as e:
            warnings.append(f"Boundary drift detection failed: {e}")
        
        return warnings


class BoundaryValidationManager:
    """
    Manager for integrating boundary validation into the processing pipeline.
    
    This class provides easy integration points for adding boundary validation
    throughout the text formatting pipeline without disrupting existing functionality.
    """
    
    def __init__(self, strict_mode: bool = False, language: str = "en"):
        """Initialize the boundary validation manager."""
        self.validator = EntityBoundaryValidator(strict_mode, language)
        self.enabled = True
        self.validation_points = set()
        
    def enable_validation(self):
        """Enable boundary validation."""
        self.enabled = True
    
    def disable_validation(self):
        """Disable boundary validation."""
        self.enabled = False
    
    def validate_at_step(self, step: str, entities: List[Entity], 
                        text: str) -> Tuple[List[Entity], List[str]]:
        """
        Validate entities at a specific pipeline step.
        
        Args:
            step: Pipeline step name
            entities: Entities to validate
            text: Current text
            
        Returns:
            Tuple of (validated_entities, warnings)
        """
        if not self.enabled:
            return entities, []
        
        self.validation_points.add(step)
        result = self.validator.validate_entity_boundaries(entities, text, step)
        
        # Always return entities - don't break functionality
        # Just log warnings for monitoring
        if result.warnings:
            logger.debug(f"BOUNDARY_VALIDATION [{step}]: {len(result.warnings)} warnings generated")
        
        return entities, result.warnings
    
    def validate_conversion_result(self, original_entities: List[Entity],
                                 updated_entities: List[Entity],
                                 original_text: str,
                                 converted_text: str) -> List[str]:
        """
        Validate the result of entity conversion.
        
        Args:
            original_entities: Entities before conversion
            updated_entities: Entities after conversion  
            original_text: Text before conversion
            converted_text: Text after conversion
            
        Returns:
            List of validation warnings
        """
        if not self.enabled:
            return []
        
        result = self.validator.validate_boundary_integrity_after_conversion(
            original_entities, updated_entities, original_text, converted_text
        )
        
        return result.warnings
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation activity."""
        return {
            "enabled": self.enabled,
            "validation_points": list(self.validation_points),
            "statistics": self.validator.get_validation_statistics()
        }


# Factory functions for easy integration

def create_boundary_validator(strict_mode: bool = False, 
                            language: str = "en") -> EntityBoundaryValidator:
    """Create an entity boundary validator."""
    return EntityBoundaryValidator(strict_mode, language)


def create_validation_manager(strict_mode: bool = False, 
                            language: str = "en") -> BoundaryValidationManager:
    """Create a boundary validation manager for pipeline integration."""
    return BoundaryValidationManager(strict_mode, language)