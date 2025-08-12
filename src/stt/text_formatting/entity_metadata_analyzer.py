#!/usr/bin/env python3
"""
PHASE 6: Entity Metadata Analyzer

This module provides enhanced entity metadata analysis capabilities for tracking
entity relationships, conflicts, and processing state throughout the pipeline.

Key Features:
1. Entity relationship detection (overlapping, containing, adjacent, conflicting)
2. Processing state tracking and validation
3. Conflict resolution based on enriched metadata
4. Recovery mechanisms for corrupted entities
5. Comprehensive debugging and forensic capabilities
"""

from __future__ import annotations

import time
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

from stt.core.config import setup_logging
from stt.text_formatting.common import Entity, EntityType
from stt.text_formatting.universal_priority_manager import get_priority_manager

logger = setup_logging(__name__)


@dataclass
class EntityRelationshipAnalysis:
    """Comprehensive analysis of entity relationships and conflicts."""
    
    entity_id: str
    entity: Entity
    overlapping_entities: List[Tuple[str, Entity, str]]  # (id, entity, relationship_type)
    contained_entities: List[Tuple[str, Entity]]         # Entities contained within this one
    containing_entity: Optional[Tuple[str, Entity]]      # Entity that contains this one
    adjacent_entities: List[Tuple[str, Entity, float]]   # (id, entity, distance)
    conflicting_entities: List[Tuple[str, Entity, str]]  # (id, entity, conflict_type)
    priority_score: float
    resolution_recommendation: str
    conflict_severity: str  # low, medium, high, critical


class EntityMetadataAnalyzer:
    """
    PHASE 6: Enhanced entity metadata analyzer for relationship tracking and conflict resolution.
    
    This analyzer builds upon the existing pipeline state management to provide
    comprehensive entity lifecycle tracking, relationship analysis, and debugging capabilities.
    """
    
    def __init__(self, language: str = "en"):
        self.language = language
        self.priority_manager = get_priority_manager(language)
        
        # Entity registry for cross-entity analysis
        self.entity_registry: Dict[str, Entity] = {}
        self.relationship_cache: Dict[str, EntityRelationshipAnalysis] = {}
        
        # Performance tracking
        self.analysis_start_time = 0
        self.analysis_metrics = {
            'entities_analyzed': 0,
            'relationships_found': 0,
            'conflicts_detected': 0,
            'recoveries_performed': 0
        }
    
    def generate_entity_id(self, entity: Entity) -> str:
        """Generate a unique, stable ID for an entity."""
        return f"{entity.type.value}_{entity.start}_{entity.end}_{hash(entity.text) % 10000}"
    
    def register_entities(self, entities: List[Entity], text: str) -> Dict[str, str]:
        """
        Register entities for metadata analysis and return ID mapping.
        
        Args:
            entities: List of entities to register
            text: Original text containing the entities
            
        Returns:
            Dictionary mapping entity object id() to entity_id for lookup
        """
        start_time = time.perf_counter()
        entity_mapping = {}
        
        for entity in entities:
            entity_id = self.generate_entity_id(entity)
            
            # Set detection metadata
            entity.set_detection_info(
                method='pipeline_detection',
                confidence=1.0,
                context=self._extract_context(entity, text)
            )
            
            # Prepare recovery data immediately
            entity.prepare_recovery_data()
            
            # Update processing state
            entity.update_processing_state('registered', 'metadata_analyzer', 'Entity registered for analysis')
            
            # Store in registry
            self.entity_registry[entity_id] = entity
            entity_mapping[id(entity)] = entity_id
            
            logger.debug(f"PHASE_6: Registered entity {entity_id}: {entity.type.name}('{entity.text}')")
        
        # Analyze relationships between all registered entities
        self._analyze_entity_relationships(text)
        
        elapsed = time.perf_counter() - start_time
        self.analysis_metrics['entities_analyzed'] = len(entities)
        logger.info(f"PHASE_6: Registered {len(entities)} entities and analyzed relationships in {elapsed:.4f}s")
        
        return entity_mapping
    
    def _extract_context(self, entity: Entity, text: str, window: int = 30) -> str:
        """Extract surrounding context for an entity."""
        start = max(0, entity.start - window)
        end = min(len(text), entity.end + window)
        context = text[start:end]
        
        # Mark the entity within the context
        entity_start_in_context = entity.start - start
        entity_end_in_context = entity.end - start
        
        context_with_markers = (
            context[:entity_start_in_context] + 
            "<<<" + context[entity_start_in_context:entity_end_in_context] + ">>>" +
            context[entity_end_in_context:]
        )
        
        return context_with_markers
    
    def _analyze_entity_relationships(self, text: str):
        """Analyze relationships between all registered entities."""
        entities_list = list(self.entity_registry.items())
        relationships_found = 0
        
        for i, (entity_id1, entity1) in enumerate(entities_list):
            analysis = EntityRelationshipAnalysis(
                entity_id=entity_id1,
                entity=entity1,
                overlapping_entities=[],
                contained_entities=[],
                containing_entity=None,
                adjacent_entities=[],
                conflicting_entities=[],
                priority_score=self.priority_manager.get_entity_priority(entity1.type),
                resolution_recommendation="",
                conflict_severity="low"
            )
            
            # Compare with all other entities
            for j, (entity_id2, entity2) in enumerate(entities_list):
                if i == j:
                    continue
                
                relationship_type = self._determine_relationship(entity1, entity2)
                
                if relationship_type == "overlapping":
                    analysis.overlapping_entities.append((entity_id2, entity2, relationship_type))
                    entity1.add_relationship('overlapping_entities', entity_id2)
                    relationships_found += 1
                    
                elif relationship_type == "contains":
                    analysis.contained_entities.append((entity_id2, entity2))
                    entity1.add_relationship('contained_entities', entity_id2)
                    entity2.add_relationship('containing_entity', entity_id1)
                    relationships_found += 1
                    
                elif relationship_type == "contained_by":
                    analysis.containing_entity = (entity_id2, entity2)
                    entity1.add_relationship('containing_entity', entity_id2)
                    relationships_found += 1
                    
                elif relationship_type == "adjacent":
                    distance = self._calculate_distance(entity1, entity2)
                    if distance < 20:  # Adjacent if within 20 characters
                        analysis.adjacent_entities.append((entity_id2, entity2, distance))
                        entity1.add_relationship('adjacent_entities', entity_id2)
                        relationships_found += 1
            
            # Detect conflicts
            conflicts = self._detect_conflicts(entity1, analysis)
            analysis.conflicting_entities = conflicts
            
            for conflict_id, conflict_entity, conflict_type in conflicts:
                entity1.add_relationship('conflicting_entities', conflict_id)
            
            # Calculate conflict severity and resolution recommendation
            analysis.conflict_severity = self._assess_conflict_severity(analysis)
            analysis.resolution_recommendation = self._recommend_resolution(analysis)
            
            # Store analysis
            self.relationship_cache[entity_id1] = analysis
            
            # Update entity metadata with relationship info
            entity1.metadata['relationships']['priority_score'] = analysis.priority_score
        
        self.analysis_metrics['relationships_found'] = relationships_found
        logger.info(f"PHASE_6: Found {relationships_found} entity relationships")
    
    def _determine_relationship(self, entity1: Entity, entity2: Entity) -> str:
        """Determine the relationship between two entities."""
        # Check for exact overlap
        if entity1.start == entity2.start and entity1.end == entity2.end:
            return "identical"
        
        # Check for containment
        if entity1.start <= entity2.start and entity1.end >= entity2.end:
            return "contains"
        
        if entity2.start <= entity1.start and entity2.end >= entity1.end:
            return "contained_by"
        
        # Check for partial overlap
        if not (entity1.end <= entity2.start or entity2.end <= entity1.start):
            return "overlapping"
        
        # Check for adjacency (no overlap but close)
        distance = self._calculate_distance(entity1, entity2)
        if distance < 20:
            return "adjacent"
        
        return "separate"
    
    def _calculate_distance(self, entity1: Entity, entity2: Entity) -> float:
        """Calculate the distance between two entities."""
        if entity1.end <= entity2.start:
            return entity2.start - entity1.end
        elif entity2.end <= entity1.start:
            return entity1.start - entity2.end
        else:
            # Overlapping entities have distance 0
            return 0.0
    
    def _detect_conflicts(self, entity: Entity, analysis: EntityRelationshipAnalysis) -> List[Tuple[str, Entity, str]]:
        """Detect conflicts for a specific entity based on its relationships."""
        conflicts = []
        
        # Overlapping entities with different types are potential conflicts
        for other_id, other_entity, _ in analysis.overlapping_entities:
            if entity.type != other_entity.type:
                # Check priority to determine if this is a real conflict
                entity_priority = self.priority_manager.get_entity_priority(entity.type)
                other_priority = self.priority_manager.get_entity_priority(other_entity.type)
                
                if entity_priority != other_priority:
                    conflict_type = "priority_overlap"
                    conflicts.append((other_id, other_entity, conflict_type))
                    
                    # Log the conflict in entity metadata
                    entity.add_validation_error(
                        f"Priority conflict with {other_entity.type.name} entity: {other_entity.text}",
                        "warning"
                    )
        
        # Check for type-specific conflicts
        conflicts.extend(self._detect_type_specific_conflicts(entity, analysis))
        
        return conflicts
    
    def _detect_type_specific_conflicts(self, entity: Entity, analysis: EntityRelationshipAnalysis) -> List[Tuple[str, Entity, str]]:
        """Detect type-specific conflicts based on entity semantics."""
        conflicts = []
        
        # URL vs Filename conflicts
        if entity.type == EntityType.URL:
            for other_id, other_entity, _ in analysis.overlapping_entities:
                if other_entity.type == EntityType.FILENAME:
                    conflicts.append((other_id, other_entity, "url_filename_conflict"))
        
        # Math vs Assignment conflicts
        elif entity.type == EntityType.ASSIGNMENT:
            for other_id, other_entity, _ in analysis.overlapping_entities:
                if other_entity.type == EntityType.MATH:
                    conflicts.append((other_id, other_entity, "assignment_math_conflict"))
        
        # Email vs URL conflicts
        elif entity.type == EntityType.EMAIL:
            for other_id, other_entity, _ in analysis.overlapping_entities:
                if other_entity.type == EntityType.URL:
                    conflicts.append((other_id, other_entity, "email_url_conflict"))
        
        return conflicts
    
    def _assess_conflict_severity(self, analysis: EntityRelationshipAnalysis) -> str:
        """Assess the severity of conflicts for an entity."""
        num_conflicts = len(analysis.conflicting_entities)
        
        if num_conflicts == 0:
            return "none"
        elif num_conflicts == 1:
            return "low"
        elif num_conflicts <= 3:
            return "medium"
        else:
            return "high"
    
    def _recommend_resolution(self, analysis: EntityRelationshipAnalysis) -> str:
        """Recommend a resolution strategy for entity conflicts."""
        if analysis.conflict_severity == "none":
            return "no_action_needed"
        
        entity = analysis.entity
        
        # Priority-based resolution
        if analysis.conflicting_entities:
            highest_conflict_priority = max(
                self.priority_manager.get_entity_priority(conflict_entity.type)
                for _, conflict_entity, _ in analysis.conflicting_entities
            )
            
            if analysis.priority_score > highest_conflict_priority:
                return "keep_entity_remove_conflicts"
            elif analysis.priority_score < highest_conflict_priority:
                return "remove_entity_keep_conflicts" 
            else:
                return "length_based_resolution"
        
        return "manual_review_required"
    
    def resolve_conflicts(self, text: str) -> Tuple[List[Entity], List[str]]:
        """
        Resolve entity conflicts based on metadata analysis.
        
        Returns:
            Tuple of (resolved_entities, resolution_log)
        """
        resolution_log = []
        resolved_entities = []
        entities_to_remove = set()
        
        # Process entities by priority (highest first)
        sorted_analyses = sorted(
            self.relationship_cache.values(),
            key=lambda a: a.priority_score,
            reverse=True
        )
        
        for analysis in sorted_analyses:
            entity_id = analysis.entity_id
            entity = analysis.entity
            
            # Skip if already marked for removal
            if entity_id in entities_to_remove:
                resolution_log.append(f"SKIPPED: {entity.type.name}('{entity.text}') - already removed by higher priority entity")
                continue
            
            # Apply resolution recommendation
            if analysis.resolution_recommendation == "keep_entity_remove_conflicts":
                resolved_entities.append(entity)
                entity.update_processing_state('resolved', 'conflict_resolver', 'Kept as highest priority')
                
                # Mark conflicting entities for removal
                for conflict_id, conflict_entity, conflict_type in analysis.conflicting_entities:
                    entities_to_remove.add(conflict_id)
                    resolution_log.append(
                        f"RESOLVED: Kept {entity.type.name}('{entity.text}') over {conflict_entity.type.name}('{conflict_entity.text}') due to priority"
                    )
                    
            elif analysis.resolution_recommendation == "remove_entity_keep_conflicts":
                entities_to_remove.add(entity_id)
                entity.update_processing_state('removed', 'conflict_resolver', 'Removed due to lower priority')
                resolution_log.append(f"REMOVED: {entity.type.name}('{entity.text}') - lower priority than conflicts")
                
            elif analysis.resolution_recommendation == "length_based_resolution":
                # Use length as tiebreaker for same priority
                should_keep = True
                for conflict_id, conflict_entity, _ in analysis.conflicting_entities:
                    if len(conflict_entity.text) > len(entity.text):
                        should_keep = False
                        break
                
                if should_keep:
                    resolved_entities.append(entity)
                    entity.update_processing_state('resolved', 'conflict_resolver', 'Kept based on length tiebreaker')
                    for conflict_id, _, _ in analysis.conflicting_entities:
                        entities_to_remove.add(conflict_id)
                else:
                    entities_to_remove.add(entity_id)
                    entity.update_processing_state('removed', 'conflict_resolver', 'Removed based on length tiebreaker')
                    
            else:
                # No conflicts or manual review needed - keep entity
                resolved_entities.append(entity)
                entity.update_processing_state('resolved', 'conflict_resolver', 'No conflicts detected')
        
        self.analysis_metrics['conflicts_detected'] = len([a for a in sorted_analyses if a.conflicting_entities])
        
        logger.info(f"PHASE_6: Resolved conflicts for {len(self.relationship_cache)} entities")
        logger.info(f"PHASE_6: Final count: {len(resolved_entities)} entities kept, {len(entities_to_remove)} entities removed")
        
        return resolved_entities, resolution_log
    
    def validate_entities(self, entities: List[Entity], text: str) -> List[Entity]:
        """
        Validate entities and attempt recovery for corrupted ones.
        
        Args:
            entities: List of entities to validate
            text: Current text content
            
        Returns:
            List of validated entities (with recovery attempts if needed)
        """
        validated_entities = []
        
        for entity in entities:
            # Check if entity boundaries are still valid
            if entity.start >= 0 and entity.end <= len(text) and entity.start < entity.end:
                # Check if text still matches
                actual_text = text[entity.start:entity.end]
                
                if actual_text == entity.text:
                    # Entity is valid
                    entity.mark_validation_status('valid', 'Entity boundaries and text match')
                    validated_entities.append(entity)
                else:
                    # Text doesn't match - attempt recovery
                    logger.warning(f"PHASE_6: Entity text mismatch - expected '{entity.text}', found '{actual_text}'")
                    
                    if entity.attempt_recovery(text):
                        validated_entities.append(entity)
                        self.analysis_metrics['recoveries_performed'] += 1
                        logger.info(f"PHASE_6: Successfully recovered entity: {entity.type.name}('{entity.text}')")
                    else:
                        entity.mark_validation_status('invalid', 'Recovery failed - text mismatch')
                        logger.warning(f"PHASE_6: Failed to recover entity: {entity.type.name}('{entity.text}')")
            else:
                # Invalid boundaries - attempt recovery
                logger.warning(f"PHASE_6: Entity has invalid boundaries: {entity.start}-{entity.end}")
                
                if entity.attempt_recovery(text):
                    validated_entities.append(entity)
                    self.analysis_metrics['recoveries_performed'] += 1
                    logger.info(f"PHASE_6: Successfully recovered entity: {entity.type.name}('{entity.text}')")
                else:
                    entity.mark_validation_status('invalid', 'Recovery failed - invalid boundaries')
                    logger.warning(f"PHASE_6: Failed to recover entity: {entity.type.name}('{entity.text}')")
        
        logger.info(f"PHASE_6: Validated {len(validated_entities)}/{len(entities)} entities")
        return validated_entities
    
    def get_debug_report(self) -> str:
        """Generate a comprehensive debugging report."""
        report_lines = [
            "=== PHASE 6: Entity Metadata Analysis Debug Report ===",
            "",
            f"Analysis Metrics:",
            f"  Entities Analyzed: {self.analysis_metrics['entities_analyzed']}",
            f"  Relationships Found: {self.analysis_metrics['relationships_found']}",
            f"  Conflicts Detected: {self.analysis_metrics['conflicts_detected']}",
            f"  Recoveries Performed: {self.analysis_metrics['recoveries_performed']}",
            "",
            "Entity Details:",
        ]
        
        for entity_id, analysis in self.relationship_cache.items():
            entity = analysis.entity
            report_lines.extend([
                f"",
                f"Entity: {entity.type.name}('{entity.text}') [{entity.start}:{entity.end}]",
                f"  ID: {entity_id}",
                f"  Priority Score: {analysis.priority_score}",
                f"  Conflict Severity: {analysis.conflict_severity}",
                f"  Resolution: {analysis.resolution_recommendation}",
                f"  Overlapping: {len(analysis.overlapping_entities)}",
                f"  Conflicting: {len(analysis.conflicting_entities)}",
                f"  Processing Stage: {entity.metadata['processing_state']['lifecycle_stage']}",
                f"  Validation Status: {entity.metadata['processing_state']['validation_status']}",
            ])
            
            # Show conflicts if any
            if analysis.conflicting_entities:
                report_lines.append("  Conflicts:")
                for conflict_id, conflict_entity, conflict_type in analysis.conflicting_entities:
                    report_lines.append(f"    - {conflict_entity.type.name}('{conflict_entity.text}') [{conflict_type}]")
        
        return "\n".join(report_lines)
    
    def get_analysis_for_entity(self, entity_id: str) -> Optional[EntityRelationshipAnalysis]:
        """Get relationship analysis for a specific entity."""
        return self.relationship_cache.get(entity_id)
    
    def clear_cache(self):
        """Clear the analysis cache and reset metrics."""
        self.entity_registry.clear()
        self.relationship_cache.clear()
        self.analysis_metrics = {
            'entities_analyzed': 0,
            'relationships_found': 0,
            'conflicts_detected': 0,
            'recoveries_performed': 0
        }


def create_entity_metadata_analyzer(language: str = "en") -> EntityMetadataAnalyzer:
    """Factory function to create an entity metadata analyzer."""
    return EntityMetadataAnalyzer(language=language)