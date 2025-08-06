"""
Utility functions for working with the simplified entity hierarchy.

This module provides helper functions that demonstrate how the entity
category system simplifies entity type management.
"""

from typing import List, Dict, Set, Tuple
from collections import defaultdict

from stt.text_formatting.common import Entity, EntityType
from stt.text_formatting.entity_categories import (
    EntityCategory, 
    get_entity_category,
    get_entity_hierarchy,
    is_same_category
)


def group_entities_by_category(entities: List[Entity]) -> Dict[EntityCategory, List[Entity]]:
    """
    Group entities by their categories.
    
    This simplifies processing entities by type without dealing with 70+ types.
    
    Args:
        entities: List of entities to group
        
    Returns:
        Dictionary mapping categories to entity lists
    """
    grouped = defaultdict(list)
    for entity in entities:
        category = get_entity_category(entity.type)
        grouped[category].append(entity)
    return dict(grouped)


def filter_entities_by_category(entities: List[Entity], 
                              categories: Set[EntityCategory]) -> List[Entity]:
    """
    Filter entities to only include those in specified categories.
    
    Args:
        entities: List of entities to filter
        categories: Set of categories to include
        
    Returns:
        Filtered list of entities
    """
    return [e for e in entities if get_entity_category(e.type) in categories]


def get_entity_priority(entity_type: EntityType) -> int:
    """
    Get processing priority for an entity type based on its category.
    
    This provides a simplified priority system based on categories
    rather than individual entity types.
    
    Args:
        entity_type: The entity type
        
    Returns:
        Priority value (higher = process first)
    """
    category_priority = {
        EntityCategory.CODE: 100,      # Code entities often need protection
        EntityCategory.WEB: 90,        # URLs/emails are usually preserved
        EntityCategory.FINANCIAL: 80,  # Money formatting is important
        EntityCategory.NUMERIC: 70,    # Numbers have complex rules
        EntityCategory.TEMPORAL: 60,   # Time/date formatting
        EntityCategory.MEASUREMENT: 50, # Units and quantities
        EntityCategory.OPERATOR: 40,   # Math/code operators
        EntityCategory.TEXT: 30,       # General text entities
    }
    
    category = get_entity_category(entity_type)
    return category_priority.get(category, 0)


def merge_adjacent_entities(entities: List[Entity]) -> List[Entity]:
    """
    Merge adjacent entities of the same category.
    
    This is useful for combining related entities that should be
    processed together.
    
    Args:
        entities: List of entities sorted by position
        
    Returns:
        List with adjacent same-category entities merged
    """
    if not entities:
        return []
        
    merged = []
    current = entities[0]
    
    for next_entity in entities[1:]:
        # Check if entities are adjacent and same category
        if (current.end == next_entity.start and 
            is_same_category(current.type, next_entity.type)):
            # Merge entities
            current = Entity(
                start=current.start,
                end=next_entity.end,
                text=current.text + next_entity.text,
                type=current.type,  # Keep first entity's specific type
                metadata={
                    "merged": True,
                    "original_types": [current.type, next_entity.type],
                    "original_metadata": [current.metadata, next_entity.metadata]
                }
            )
        else:
            merged.append(current)
            current = next_entity
            
    merged.append(current)
    return merged


def get_category_converter_map() -> Dict[EntityCategory, str]:
    """
    Get mapping of categories to converter class names.
    
    This simplifies routing entities to appropriate converters.
    
    Returns:
        Dictionary mapping categories to converter names
    """
    return {
        EntityCategory.NUMERIC: "NumericConverter",
        EntityCategory.WEB: "WebConverter", 
        EntityCategory.CODE: "CodeConverter",
        EntityCategory.TEXT: "TextConverter",
        EntityCategory.TEMPORAL: "TemporalConverter",
        EntityCategory.FINANCIAL: "FinancialConverter",
        EntityCategory.MEASUREMENT: "MeasurementConverter",
        EntityCategory.OPERATOR: "OperatorConverter",
    }


def analyze_entity_distribution(entities: List[Entity]) -> Dict[str, any]:
    """
    Analyze the distribution of entities by category and type.
    
    This provides insights into entity patterns in text.
    
    Args:
        entities: List of entities to analyze
        
    Returns:
        Dictionary with analysis results
    """
    hierarchy = get_entity_hierarchy()
    
    # Group by category
    by_category = group_entities_by_category(entities)
    
    # Count by specific type
    type_counts = defaultdict(int)
    for entity in entities:
        type_counts[entity.type] += 1
        
    # Calculate statistics
    analysis = {
        "total_entities": len(entities),
        "categories": {
            cat.name: {
                "count": len(entities_list),
                "percentage": len(entities_list) / len(entities) * 100 if entities else 0
            }
            for cat, entities_list in by_category.items()
        },
        "top_types": sorted(
            [(t.name, count) for t, count in type_counts.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10],
        "category_diversity": len(by_category),
        "type_diversity": len(type_counts),
    }
    
    return analysis


def suggest_entity_type(text: str, context: str = "") -> Tuple[EntityType, float]:
    """
    Suggest the most likely entity type for given text.
    
    This demonstrates how categories can simplify entity type selection
    for AI agents.
    
    Args:
        text: The text to classify
        context: Optional surrounding context
        
    Returns:
        Tuple of (suggested EntityType, confidence score)
    """
    text_lower = text.lower()
    
    # Simple heuristics based on patterns
    if "@" in text:
        return EntityType.EMAIL, 0.9
    elif text.startswith("http") or ".com" in text or ".org" in text:
        return EntityType.URL, 0.9
    elif text.startswith("/") and len(text) > 1:
        return EntityType.SLASH_COMMAND, 0.8
    elif text.startswith("--") or text.startswith("-"):
        return EntityType.COMMAND_FLAG, 0.8
    elif text.endswith((".py", ".js", ".java", ".cpp")):
        return EntityType.FILENAME, 0.9
    elif text.isdigit():
        return EntityType.CARDINAL, 0.8
    elif text.endswith(("st", "nd", "rd", "th")) and text[:-2].isdigit():
        return EntityType.ORDINAL, 0.9
    elif "$" in text or "€" in text or "£" in text:
        return EntityType.CURRENCY, 0.9
    elif "%" in text:
        return EntityType.PERCENT, 0.8
    elif any(unit in text_lower for unit in ["kg", "km", "mb", "gb"]):
        return EntityType.QUANTITY, 0.7
    else:
        return EntityType.TEXT, 0.3


def validate_entity_consistency(entities: List[Entity]) -> List[str]:
    """
    Validate entity list for consistency issues.
    
    Uses category relationships to identify potential problems.
    
    Args:
        entities: List of entities to validate
        
    Returns:
        List of validation warnings
    """
    warnings = []
    hierarchy = get_entity_hierarchy()
    
    # Check for overlapping entities
    for i, entity1 in enumerate(entities):
        for entity2 in entities[i+1:]:
            if entity1.start < entity2.end and entity2.start < entity1.end:
                # Entities overlap
                if not is_same_category(entity1.type, entity2.type):
                    warnings.append(
                        f"Overlapping entities of different categories: "
                        f"{entity1.type.name} ({get_entity_category(entity1.type).name}) "
                        f"and {entity2.type.name} ({get_entity_category(entity2.type).name}) "
                        f"at positions {entity1.start}-{entity1.end} and "
                        f"{entity2.start}-{entity2.end}"
                    )
                    
    # Check for unusual patterns
    category_sequence = [get_entity_category(e.type) for e in entities]
    
    # Detect repeated patterns that might indicate issues
    for i in range(len(category_sequence) - 2):
        if (category_sequence[i] == category_sequence[i+2] and 
            category_sequence[i] != category_sequence[i+1]):
            warnings.append(
                f"Unusual pattern: {category_sequence[i].name} -> "
                f"{category_sequence[i+1].name} -> {category_sequence[i].name} "
                f"at positions {i}-{i+2}"
            )
            
    return warnings