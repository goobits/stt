#!/usr/bin/env python3
"""Capitalization rules and patterns for the SmartCapitalizer."""
from __future__ import annotations

from stt.text_formatting.common import EntityType


class CapitalizationRules:
    """Manages capitalization rules, patterns, and constants."""

    def __init__(self, resources: dict, language: str = "en"):
        """Initialize capitalization rules with language-specific resources.
        
        Args:
            resources: Language-specific resource dictionary from constants
            language: Language code (e.g., "en", "es")
        """
        self.resources = resources
        self.language = language
        
        # Entity types that must have their casing preserved under all circumstances
        self.STRICTLY_PROTECTED_TYPES = {
            EntityType.URL,
            EntityType.SPOKEN_URL,
            EntityType.SPOKEN_PROTOCOL_URL,
            EntityType.EMAIL,
            EntityType.SPOKEN_EMAIL,
            EntityType.FILENAME,
            EntityType.INCREMENT_OPERATOR,
            EntityType.DECREMENT_OPERATOR,
            EntityType.SLASH_COMMAND,
            EntityType.COMMAND_FLAG,
            EntityType.SIMPLE_UNDERSCORE_VARIABLE,
            EntityType.UNDERSCORE_DELIMITER,
            EntityType.VARIABLE,  # Single-letter variables must preserve their case (e.g., 'i' in code contexts)
            EntityType.PORT_NUMBER,  # URLs with ports should stay lowercase (e.g., localhost:8080)
            # Note: VERSION removed - version numbers at sentence start should be capitalized
            EntityType.ASSIGNMENT,
            EntityType.COMPARISON,
            EntityType.MATH_EXPRESSION,  # Math expressions should preserve their formatting
            EntityType.MATH_CONSTANT,  # Mathematical constants like π, ∞ should preserve their exact form
            EntityType.ABBREVIATION,  # Latin abbreviations like i.e., e.g. should stay lowercase
            # Note: CLI_COMMAND removed - they should be capitalized at sentence start
        }

        # Version patterns that indicate technical content
        self.version_patterns = {"version", "v.", "v", "build", "release"}

        # Abbreviation patterns and their corrections (only applies mid-sentence)
        self.abbreviation_fixes = {
            # Only fix capitalized abbreviations in mid-sentence contexts
            # At sentence start, they should remain capitalized
            "Vs.": "vs.",  # vs. should not be capitalized mid-sentence
            "Cf.": "cf.",  # cf. should not be capitalized mid-sentence
            "Ie.": "i.e.",  # Fix wrong abbreviation form
            "Eg.": "e.g.",  # Fix wrong abbreviation form  
            "Ex.": "e.g.",  # Convert Ex. to e.g.
        }

        # Load uppercase abbreviations from resources
        self.uppercase_abbreviations = self.resources.get("technical", {}).get("uppercase_abbreviations", {})

        # Load common abbreviations from resources
        self.common_abbreviations = tuple(self.resources.get("technical", {}).get("common_abbreviations", []))

    def is_strictly_protected_type(self, entity_type: EntityType) -> bool:
        """Check if an entity type is strictly protected from capitalization.
        
        Args:
            entity_type: The entity type to check
            
        Returns:
            True if the entity type should never be modified for capitalization
        """
        return entity_type in self.STRICTLY_PROTECTED_TYPES

    def should_protect_entity_from_spacy_capitalization(self, entity_type: EntityType) -> bool:
        """Check if an entity should be protected from SpaCy proper noun capitalization.
        
        Args:
            entity_type: The entity type to check
            
        Returns:
            True if the entity should be protected from SpaCy capitalization
        """
        protected_types = {
            EntityType.URL,
            EntityType.SPOKEN_URL,
            EntityType.SPOKEN_PROTOCOL_URL,
            EntityType.EMAIL,
            EntityType.SPOKEN_EMAIL,
            EntityType.FILENAME,
            EntityType.ASSIGNMENT,
            EntityType.INCREMENT_OPERATOR,
            EntityType.DECREMENT_OPERATOR,
            EntityType.COMMAND_FLAG,
            EntityType.PORT_NUMBER,
            EntityType.ABBREVIATION,  # Protect abbreviations from SpaCy proper noun capitalization
            EntityType.VARIABLE,  # Protect single-letter variables from SpaCy capitalization
        }
        return entity_type in protected_types

    def should_protect_entity_from_uppercase_conversion(self, entity_type: EntityType) -> bool:
        """Check if an entity should be protected from uppercase abbreviation conversion.
        
        Args:
            entity_type: The entity type to check
            
        Returns:
            True if the entity should be protected from uppercase conversion
        """
        protected_types = {
            EntityType.URL,
            EntityType.SPOKEN_URL,
            EntityType.SPOKEN_PROTOCOL_URL,
            EntityType.EMAIL,
            EntityType.SPOKEN_EMAIL,
            EntityType.FILENAME,
            EntityType.ASSIGNMENT,
            EntityType.INCREMENT_OPERATOR,
            EntityType.DECREMENT_OPERATOR,
            EntityType.COMMAND_FLAG,
            EntityType.PORT_NUMBER,
            EntityType.ABBREVIATION,  # Protect abbreviations from uppercase conversion
            EntityType.VARIABLE,  # Protect single-letter variables from uppercase conversion
        }
        return entity_type in protected_types

    def get_technical_verbs(self) -> list[str]:
        """Get list of technical verbs that should not be capitalized as proper nouns.
        
        Returns:
            List of technical verb strings
        """
        return self.resources.get("technical", {}).get("verbs", [])

    def get_common_abbreviations_pattern(self) -> str:
        """Get regex pattern for common abbreviations.
        
        Returns:
            Regex pattern string for matching common abbreviations
        """
        common_abbreviations = self.resources.get("technical", {}).get("common_abbreviations", [])
        return "|".join(abbrev.replace(".", "\\.") for abbrev in common_abbreviations)

    def get_capitalization_context_rules(self) -> dict:
        """Get language-specific capitalization context rules.
        
        Returns:
            Dictionary with capitalization context rules
        """
        return self.resources.get("capitalization_context", {})

    def should_preserve_case_after_entity(self, entity_type: EntityType, following_word: str) -> bool:
        """Check if case should be preserved for a word following a specific entity type.
        
        Args:
            entity_type: The entity type that precedes the word
            following_word: The word that follows the entity
            
        Returns:
            True if the word's case should be preserved (not auto-capitalized)
        """
        # Get capitalization context rules
        context_rules = self.get_capitalization_context_rules()
        entity_rules = context_rules.get("entity_capitalization_rules", {})
        
        # Check if we have specific rules for this entity type
        entity_type_name = entity_type.name if hasattr(entity_type, 'name') else str(entity_type)
        entity_rule = entity_rules.get(entity_type_name, {})
        
        # If capitalize_after is False, preserve case
        if not entity_rule.get("capitalize_after", True):
            return True
        
        # For Spanish language, check additional context
        if self.language == "es":
            return self._should_preserve_case_spanish(entity_type, following_word, context_rules)
        
        return False

    def _should_preserve_case_spanish(self, entity_type: EntityType, following_word: str, context_rules: dict) -> bool:
        """Spanish-specific case preservation logic.
        
        Args:
            entity_type: The entity type
            following_word: The word following the entity
            context_rules: Capitalization context rules
            
        Returns:
            True if case should be preserved
        """
        # Get Spanish-specific word lists
        continuation_words = set(context_rules.get("continuation_words", []))
        lowercase_after_entities = set(context_rules.get("spanish_lowercase_after_entities", []))
        
        # Check if word is in continuation or lowercase lists
        word_lower = following_word.lower()
        if word_lower in continuation_words or word_lower in lowercase_after_entities:
            return True
        
        # Check if global setting says not to capitalize after technical entities
        if not context_rules.get("capitalize_after_technical_entity", True):
            technical_entity_types = {
                EntityType.SLASH_COMMAND,
                EntityType.COMMAND_FLAG, 
                EntityType.SIMPLE_UNDERSCORE_VARIABLE,
                EntityType.UNDERSCORE_DELIMITER
            }
            if entity_type in technical_entity_types:
                return True
        
        return False

    def get_spanish_continuation_words(self) -> set:
        """Get Spanish words that should not be capitalized in continuation contexts.
        
        Returns:
            Set of Spanish continuation words
        """
        context_rules = self.get_capitalization_context_rules()
        return set(context_rules.get("continuation_words", []))

    def apply_entity_type_spacing_rules(self, text: str, entities: list, language: str) -> str:
        """Apply entity type-specific spacing rules.
        
        Args:
            text: Text to process
            entities: List of entities with position information
            language: Language code (e.g., "en", "es")
            
        Returns:
            Text with spacing rules applied
        """
        if not entities:
            return text
        
        # Get entity-specific rules
        context_rules = self.get_capitalization_context_rules()
        entity_rules = context_rules.get("entity_capitalization_rules", {})
        
        # Sort entities by position (reverse order to maintain positions during modification)
        sorted_entities = sorted(entities, key=lambda e: getattr(e, 'start', 0), reverse=True)
        
        for entity in sorted_entities:
            entity_type_name = getattr(entity.type, 'name', str(entity.type)) if hasattr(entity, 'type') else None
            if not entity_type_name:
                continue
                
            entity_rule = entity_rules.get(entity_type_name, {})
            spacing_rule = entity_rule.get("spacing_before", {})
            
            if spacing_rule.get("require_space"):
                text = self._apply_spacing_rule(text, entity, spacing_rule, language)
        
        return text

    def _apply_spacing_rule(self, text: str, entity, spacing_rule: dict, language: str) -> str:
        """Apply spacing rule for a specific entity.
        
        Args:
            text: Text to process
            entity: Entity object
            spacing_rule: Spacing rule configuration
            language: Language code
            
        Returns:
            Text with spacing rule applied
        """
        entity_start = getattr(entity, 'start', 0)
        entity_end = getattr(entity, 'end', 0)
        entity_text = getattr(entity, 'text', '')
        
        if entity_start <= 0 or entity_end > len(text):
            return text
        
        # Check preposition requirements
        prepositions = spacing_rule.get("prepositions", [])
        if prepositions and entity_start > 0:
            # Look for prepositions before this entity
            preceding_text = text[:entity_start].strip()
            words_before = preceding_text.lower().split()
            
            # Check if the last word(s) before the entity form a preposition
            for prep in prepositions:
                prep_words = prep.lower().split()
                if len(words_before) >= len(prep_words):
                    # Check if the last N words match this preposition
                    last_words = words_before[-len(prep_words):]
                    if last_words == prep_words:
                        # Ensure there's a space between preposition and entity
                        pre_entity_char = text[entity_start - 1] if entity_start > 0 else ''
                        if pre_entity_char != ' ':
                            text = text[:entity_start] + ' ' + text[entity_start:]
                            # Update entity positions for any subsequent processing
                            if hasattr(entity, 'start'):
                                entity.start += 1
                            if hasattr(entity, 'end'):
                                entity.end += 1
                        break
        
        return text

    def should_prevent_sentence_capitalization(self, entity_type: EntityType, entity_text: str) -> bool:
        """Check if an entity should prevent sentence-start capitalization.
        
        Args:
            entity_type: The entity type to check
            entity_text: The entity text
            
        Returns:
            True if this entity should prevent sentence-start capitalization
        """
        context_rules = self.get_capitalization_context_rules()
        entity_rules = context_rules.get("entity_capitalization_rules", {})
        
        entity_type_name = entity_type.name if hasattr(entity_type, 'name') else str(entity_type)
        entity_rule = entity_rules.get(entity_type_name, {})
        
        capitalization_rule = entity_rule.get("capitalization", {})
        return capitalization_rule.get("prevent_sentence_capitalization", False)

    def should_preserve_entity_case(self, entity_type: EntityType, context: str = "") -> bool:
        """Check if an entity's case should be preserved.
        
        Args:
            entity_type: The entity type to check
            context: Surrounding text context
            
        Returns:
            True if the entity case should be preserved
        """
        context_rules = self.get_capitalization_context_rules()
        entity_rules = context_rules.get("entity_capitalization_rules", {})
        
        entity_type_name = entity_type.name if hasattr(entity_type, 'name') else str(entity_type)
        entity_rule = entity_rules.get(entity_type_name, {})
        
        capitalization_rule = entity_rule.get("capitalization", {})
        return capitalization_rule.get("preserve_case", False)