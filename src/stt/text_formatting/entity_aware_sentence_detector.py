#!/usr/bin/env python3
"""
Entity-aware sentence boundary detection for improved capitalization.

This module provides enhanced sentence boundary detection that considers entity context
to determine where sentence start capitalization should occur.
"""

import re
from typing import List, Optional
from stt.core.config import setup_logging
from stt.text_formatting.common import Entity, EntityType

logger = setup_logging(__name__)


class EntityAwareSentenceDetector:
    """Detects sentence boundaries while considering entity context for capitalization."""
    
    def __init__(self):
        # Entity types that should NOT prevent sentence start capitalization
        self.non_blocking_entity_types = {
            EntityType.CARDINAL,         # Numbers
            EntityType.ORDINAL,          # Ordinal numbers  
            EntityType.CURRENCY,         # Currency amounts
            EntityType.MONEY,            # Money amounts
            EntityType.PERCENT,          # Percentages
            EntityType.DATA_SIZE,        # File sizes
            EntityType.PORT_NUMBER,      # Port numbers
            EntityType.FREQUENCY,        # Frequencies
            EntityType.MATH,             # Math expressions (in context)
            EntityType.MATH_EXPRESSION,  # Math expressions
            # Note: FILENAME entities need special handling
        }
        
        # Entity types that SHOULD prevent sentence start capitalization
        self.blocking_entity_types = {
            EntityType.CLI_COMMAND,           # "git commit" shouldn't become "Git commit"
            EntityType.COMMAND_FLAG,          # "--verbose" shouldn't become "--Verbose" 
            EntityType.PROGRAMMING_KEYWORD,   # "function" at start might be technical
            EntityType.VERSION,               # "v1.2.3" shouldn't become "V1.2.3"
            EntityType.SLASH_COMMAND,         # "/deploy" shouldn't become "/Deploy"
            EntityType.UNDERSCORE_DELIMITER,  # "__init__" shouldn't become "__Init__"
        }
    
    def should_capitalize_at_position(self, position: int, text: str, entities: List[Entity]) -> bool:
        """
        Determine if a character position should be capitalized considering entity context.
        
        This is the core logic for entity-aware sentence capitalization.
        
        Args:
            position: Character position to check for capitalization
            text: Full text context
            entities: List of entities in the text
            
        Returns:
            True if the position should be capitalized, False otherwise
        """
        if not text or position >= len(text):
            return False
            
        char = text[position]
        if not char.isalpha():
            return False
        
        # Find if this position is inside any entity
        containing_entity = self._find_containing_entity(position, entities)
        
        if containing_entity:
            return self._should_capitalize_within_entity(
                position, containing_entity, text, entities
            )
        
        # Position is not inside any entity - check for other constraints
        if self._follows_abbreviation(position, text):
            logger.debug(f"Position {position} follows abbreviation - not capitalizing")
            return False
        
        # Check if this is a valid sentence boundary
        if position == 0 or self._is_after_sentence_ending(position, text):
            logger.debug(f"Position {position} is valid sentence start - capitalizing")
            return True
        
        return False
    
    def _find_containing_entity(self, position: int, entities: List[Entity]) -> Optional[Entity]:
        """Find the entity that contains the given position."""
        for entity in entities:
            if entity.start <= position < entity.end:
                return entity
        return None
    
    def _should_capitalize_within_entity(
        self, 
        position: int, 
        entity: Entity, 
        text: str, 
        entities: List[Entity]
    ) -> bool:
        """
        Determine if a position within an entity should be capitalized.
        
        This handles the complex logic of entity-aware capitalization.
        """
        entity_type = entity.type
        
        # For blocking entity types, never capitalize
        if entity_type in self.blocking_entity_types:
            logger.debug(f"Position {position} in blocking entity {entity_type} - not capitalizing")
            return False
        
        # For non-blocking entity types, always allow capitalization
        if entity_type in self.non_blocking_entity_types:
            logger.debug(f"Position {position} in non-blocking entity {entity_type} - allowing capitalization")
            return True
        
        # Special handling for FILENAME entities
        if entity_type == EntityType.FILENAME:
            return self._should_capitalize_filename_context(position, entity, text)
        
        # For other entity types, use default logic
        # This handles cases not explicitly categorized
        if position == 0 or self._is_after_sentence_ending(position, text):
            logger.debug(f"Position {position} in entity {entity_type} at sentence boundary - allowing capitalization")
            return True
        
        return False
    
    def _should_capitalize_filename_context(self, position: int, entity: Entity, text: str) -> bool:
        """
        Special logic for filename entity capitalization.
        
        Filenames themselves shouldn't be capitalized, but surrounding context should be.
        """
        # If we're at the very start of the filename, don't capitalize the filename
        if position == entity.start:
            # Check if this is at sentence start
            if position == 0 or self._is_after_sentence_ending(position, text):
                # This is a sentence start, but the entity itself shouldn't be capitalized
                logger.debug(f"Filename '{entity.text}' at sentence start position {position} - not capitalizing filename itself")
                return False
        
        # If we're inside the filename but not at the start, allow capitalization
        # This handles cases where capitalization rules might apply within filenames
        logger.debug(f"Position {position} within filename entity - allowing capitalization")
        return True
    
    def _is_after_sentence_ending(self, position: int, text: str) -> bool:
        """Check if position comes after sentence-ending punctuation."""
        if position == 0:
            return True
        
        # Look backwards for sentence-ending punctuation
        for i in range(position - 1, -1, -1):
            char = text[i]
            if char in '.!?':
                # Found sentence ending - check if there's only whitespace between
                between_text = text[i + 1:position]
                if between_text.strip() == '':
                    return True
                break
            elif not char.isspace():
                # Found non-whitespace before any sentence ending
                break
        
        return False
    
    def _follows_abbreviation(self, position: int, text: str) -> bool:
        """Check if position follows a common abbreviation."""
        if position < 5:  # Not enough space for abbreviation
            return False
        
        # Look back for common abbreviations
        preceding_text = text[max(0, position - 10):position]
        abbrev_patterns = [
            r'\bi\.e\.\s*$',
            r'\be\.g\.\s*$', 
            r'\betc\.\s*$',
            r'\bvs\.\s*$',
            r'\bmr\.\s*$',
            r'\bmrs\.\s*$',
            r'\bdr\.\s*$',
        ]
        
        for pattern in abbrev_patterns:
            if re.search(pattern, preceding_text, re.IGNORECASE):
                return True
        
        return False
    
    def detect_sentence_start_positions(self, text: str, entities: List[Entity]) -> List[int]:
        """
        Detect all positions in text that should be capitalized as sentence starts.
        
        Args:
            text: Text to analyze
            entities: List of entities in the text
            
        Returns:
            List of character positions that should be capitalized
        """
        positions = []
        
        if not text:
            return positions
        
        # Check position 0 (start of text)
        if self.should_capitalize_at_position(0, text, entities):
            positions.append(0)
        
        # Find positions after sentence-ending punctuation
        for match in re.finditer(r'([.!?]\s+)([a-zA-Z])', text):
            letter_position = match.start(2)
            if self.should_capitalize_at_position(letter_position, text, entities):
                positions.append(letter_position)
        
        logger.debug(f"Detected sentence start positions: {positions}")
        return positions