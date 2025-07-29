#!/usr/bin/env python3
"""Shared utility functions for text formatting modules."""
from __future__ import annotations

from .common import Entity


def is_inside_entity(start: int, end: int, entities: list[Entity]) -> bool:
    """
    Check if a span is inside any existing entity.

    Args:
        start: Start position of the span to check
        end: End position of the span to check
        entities: List of existing entities to check against

    Returns:
        True if the span is inside any existing entity, False otherwise

    """
    return any(start >= entity.start and end <= entity.end for entity in entities)


def overlaps_with_entity(start: int, end: int, entities: list[Entity]) -> bool:
    """
    Check if a span overlaps with any existing entity.

    Args:
        start: Start position of the span to check
        end: End position of the span to check
        entities: List of existing entities to check against

    Returns:
        True if the span overlaps with any existing entity, False otherwise

    """
    return any(not (end <= entity.start or start >= entity.end) for entity in entities)
