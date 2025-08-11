#!/usr/bin/env python3
"""
Theory 18: Intelligent Word-After-Entity Classification

This module provides semantic analysis of words that appear after entities,
enabling context-aware word classification for improved text formatting.

Key Features:
1. Conservative word classification based on semantic context
2. Entity-aware word analysis (words following comparison operators, assignments)
3. Spanish-specific semantic rules for technical contexts
4. Integration with existing conversational flow processing

Core Problems Addressed:
- "comprobar si valor mayor que cero" → "cero" should be "0" in comparison context  
- "resultado igual a más b" → "más" should be "+" not "_más_" in assignment context
- Context-sensitive word classification for better semantic understanding
"""

import logging
import re
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

from .common import Entity, EntityType
from .constants import get_resources

logger = logging.getLogger(__name__)


class WordContext(Enum):
    """Classification contexts for word analysis."""
    UNKNOWN = "unknown"
    COMPARISON_NUMERIC = "comparison_numeric"      # Numbers in comparison contexts
    ASSIGNMENT_OPERATOR = "assignment_operator"   # Operators in assignment contexts
    TECHNICAL_NUMERIC = "technical_numeric"       # Numbers in technical contexts
    MATHEMATICAL = "mathematical"                 # Mathematical expressions


@dataclass
class WordClassification:
    """Result of word classification analysis."""
    word: str
    context: WordContext
    suggested_replacement: Optional[str] = None
    confidence: float = 0.0
    semantic_context: Optional[str] = None


class IntelligentWordClassifier:
    """
    Analyzes words in context to provide semantic classification suggestions.
    
    This classifier focuses on high-confidence, conservative improvements to
    word classification based on the semantic context provided by surrounding
    entities and patterns.
    """
    
    def __init__(self, language: str = "es"):
        self.language = language
        if language != "es":
            self.active = False
            return
            
        self.active = True
        self.resources = get_resources(language)
        self._init_classification_patterns()
        
    def _init_classification_patterns(self):
        """Initialize word classification patterns."""
        # Get conversational entities from resources
        conversational_entities = self.resources.get("conversational_patterns", {}).get("conversational_entities", {})
        
        # High-confidence number conversions in comparison contexts
        self.comparison_numbers = conversational_entities.get("numbers_in_context", {})
        
        # High-confidence operator conversions in assignment contexts
        self.assignment_operators = conversational_entities.get("operators_in_context", {})
        
        # Comparison context patterns
        self.comparison_patterns = [
            r'\b(?:mayor que|menor que|igual a|diferente de)\s+(\w+)',
            r'\bsi\s+\w+\s+(?:mayor que|menor que|igual|diferente)\s+(\w+)',
            r'\bcomprobar si\s+\w+\s+(?:mayor que|menor que|igual a)\s+(\w+)',
            r'\bverificar que\s+\w+\s+(?:mayor que|menor que|igual)\s+(\w+)'
        ]
        
        # Assignment context patterns  
        self.assignment_patterns = [
            r'\b(\w+)\s+igual\s+(?:a\s+)?\w+\s+(\w+)\s+\w+',  # "resultado igual a más b"
            r'\b(\w+)\s+igual\s+(?:a\s+)?(\w+)(?:\s+\w+)*',   # "x igual más y"
            r'\b(\w+)\s+igual\s+\w+\s+(\w+)\s+\w+',          # "total igual precio por cantidad"
        ]
        
        # Technical context patterns
        self.technical_patterns = [
            r'\bíndice\s+(\w+)',          # "índice cero" 
            r'\barray de tamaño\s+(\w+)', # "array de tamaño diez"
            r'\bversión\s+(\w+)',         # "versión dos"
            r'\bpuerto\s+(\w+)',          # "puerto tres"
        ]
        
    def classify_words_in_context(self, text: str, entities: List[Entity]) -> List[WordClassification]:
        """
        Classify words based on their semantic context within the text.
        
        Args:
            text: The input text to analyze
            entities: Detected entities that provide context
            
        Returns:
            List of word classifications with high-confidence suggestions
        """
        if not self.active:
            return []
            
        classifications = []
        text_lower = text.lower()
        
        logger.debug(f"THEORY_18: Analyzing text for word classification: '{text}'")
        
        # 1. Classify numbers in comparison contexts
        comparison_classifications = self._classify_comparison_numbers(text_lower)
        classifications.extend(comparison_classifications)
        
        # 2. Classify operators in assignment contexts
        assignment_classifications = self._classify_assignment_operators(text_lower)
        classifications.extend(assignment_classifications)
        
        # 3. Classify numbers in technical contexts
        technical_classifications = self._classify_technical_numbers(text_lower)
        classifications.extend(technical_classifications)
        
        # Log results
        if classifications:
            logger.info(f"THEORY_18: Found {len(classifications)} word classifications")
            for classification in classifications:
                logger.debug(f"THEORY_18: {classification.word} → {classification.suggested_replacement} (context: {classification.context.value}, confidence: {classification.confidence})")
        
        return classifications
    
    def _classify_comparison_numbers(self, text: str) -> List[WordClassification]:
        """Classify numbers that appear in comparison contexts."""
        classifications = []
        
        for pattern in self.comparison_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                number_word = match.group(1).lower()
                
                # Only classify if we have high confidence
                if number_word in self.comparison_numbers:
                    replacement = self.comparison_numbers[number_word]
                    
                    classification = WordClassification(
                        word=number_word,
                        context=WordContext.COMPARISON_NUMERIC,
                        suggested_replacement=replacement,
                        confidence=0.9,  # High confidence for comparison contexts
                        semantic_context=match.group(0)
                    )
                    classifications.append(classification)
                    
                    logger.debug(f"THEORY_18: Comparison number detected: '{number_word}' → '{replacement}' in '{match.group(0)}'")
        
        return classifications
    
    def _classify_assignment_operators(self, text: str) -> List[WordClassification]:
        """Classify operators that appear in assignment contexts."""
        classifications = []
        
        for pattern in self.assignment_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Look for operators in the matched groups
                full_match = match.group(0)
                
                # Check for different operators in assignment context
                assignment_operators = {
                    "más": "+",
                    "por": "×", 
                    "menos": "-",
                    "dividido por": "÷",
                    "dividido entre": "÷"
                }
                
                for operator_word, operator_symbol in assignment_operators.items():
                    if operator_word in full_match and "igual" in full_match:
                        # Create a specific pattern for this operator
                        operator_pattern = rf'\w+\s+igual\s+(?:a\s+)?\w+\s+{re.escape(operator_word)}\s+\w+'
                        if re.search(operator_pattern, full_match, re.IGNORECASE):
                            classification = WordClassification(
                                word=operator_word,
                                context=WordContext.ASSIGNMENT_OPERATOR,
                                suggested_replacement=operator_symbol,
                                confidence=0.85,  # High confidence for clear assignment contexts
                                semantic_context=full_match
                            )
                            classifications.append(classification)
                            
                            logger.debug(f"THEORY_18: Assignment operator detected: '{operator_word}' → '{operator_symbol}' in '{full_match}'")
        
        return classifications
    
    def _classify_technical_numbers(self, text: str) -> List[WordClassification]:
        """Classify numbers that appear in technical contexts."""
        classifications = []
        
        for pattern in self.technical_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                number_word = match.group(1).lower()
                
                # Only classify if we have high confidence
                if number_word in self.comparison_numbers:  # Reuse number mappings
                    replacement = self.comparison_numbers[number_word]
                    
                    classification = WordClassification(
                        word=number_word,
                        context=WordContext.TECHNICAL_NUMERIC,
                        suggested_replacement=replacement,
                        confidence=0.8,  # Good confidence for technical contexts
                        semantic_context=match.group(0)
                    )
                    classifications.append(classification)
                    
                    logger.debug(f"THEORY_18: Technical number detected: '{number_word}' → '{replacement}' in '{match.group(0)}'")
        
        return classifications
    
    def apply_word_classifications(self, text: str, classifications: List[WordClassification]) -> str:
        """
        Apply word classifications to text with conservative replacement strategy.
        
        Args:
            text: Original text
            classifications: Word classifications to apply
            
        Returns:
            Text with high-confidence word replacements applied
        """
        if not classifications:
            return text
            
        processed_text = text
        changes_applied = 0
        
        # Sort classifications by position in text to avoid replacement conflicts
        # Process from end to beginning to maintain positions
        position_sorted = []
        for classification in classifications:
            # Find position of word in text
            word_pattern = r'\b' + re.escape(classification.word) + r'\b'
            for match in re.finditer(word_pattern, processed_text, re.IGNORECASE):
                position_sorted.append((match.start(), match.end(), classification))
        
        # Sort by position (descending to process from end to beginning)
        position_sorted.sort(key=lambda x: x[0], reverse=True)
        
        # Apply replacements only if confidence is high enough
        for start_pos, end_pos, classification in position_sorted:
            if classification.confidence >= 0.8:  # Conservative threshold
                
                # Double-check the replacement makes sense in context
                if self._validate_replacement_context(processed_text, start_pos, end_pos, classification):
                    # Apply the replacement
                    processed_text = (
                        processed_text[:start_pos] + 
                        classification.suggested_replacement + 
                        processed_text[end_pos:]
                    )
                    changes_applied += 1
                    
                    logger.debug(f"THEORY_18: Applied replacement: '{classification.word}' → '{classification.suggested_replacement}' (confidence: {classification.confidence})")
        
        if changes_applied > 0:
            logger.info(f"THEORY_18: Applied {changes_applied} word classifications")
            logger.debug(f"THEORY_18: Result: '{processed_text}'")
        
        return processed_text
    
    def _validate_replacement_context(self, text: str, start_pos: int, end_pos: int, classification: WordClassification) -> bool:
        """
        Validate that a word replacement makes sense in the specific context.
        
        Conservative validation to ensure we don't make incorrect replacements.
        """
        # Get context around the word
        context_start = max(0, start_pos - 20)
        context_end = min(len(text), end_pos + 20)
        context = text[context_start:context_end].lower()
        
        word = text[start_pos:end_pos].lower()
        
        # Validate based on classification context
        if classification.context == WordContext.COMPARISON_NUMERIC:
            # Must be in a comparison context
            comparison_indicators = ["mayor que", "menor que", "igual a", "diferente", "comprobar si"]
            return any(indicator in context for indicator in comparison_indicators)
            
        elif classification.context == WordContext.ASSIGNMENT_OPERATOR:
            # Must be in an assignment context with "igual"
            return "igual" in context and word == "más"
            
        elif classification.context == WordContext.TECHNICAL_NUMERIC:
            # Must be in a technical context
            technical_indicators = ["índice", "array", "versión", "puerto", "tamaño"]
            return any(indicator in context for indicator in technical_indicators)
        
        return False
    
    def process_text_with_word_classification(self, text: str, entities: List[Entity]) -> Tuple[str, int]:
        """
        Main entry point: Process text with intelligent word classification.
        
        Args:
            text: Input text to process
            entities: Detected entities for context
            
        Returns:
            Tuple of (processed_text, changes_applied)
        """
        if not self.active:
            return text, 0
            
        logger.info(f"THEORY_18: Processing text with word classification: '{text}'")
        
        # 1. Analyze and classify words
        classifications = self.classify_words_in_context(text, entities)
        
        if not classifications:
            logger.debug("THEORY_18: No word classifications found")
            return text, 0
        
        # 2. Apply classifications
        processed_text = self.apply_word_classifications(text, classifications)
        
        # 3. Count changes
        changes_applied = len([c for c in classifications if c.confidence >= 0.8])
        
        return processed_text, changes_applied