#!/usr/bin/env python3
"""
Universal Code Symbol Mapping Registry

This module provides a smart, extensible system for code entity processing across ALL languages.
It centralizes code symbol mappings and provides cross-language pattern recognition capabilities.

Key Features:
- Universal code symbol mapping for any language
- Cross-language code pattern recognition
- Context-aware code processing
- Extensible resource architecture
- Smart detection that scales to future languages

This is the foundation for Theory 6: Generalized Cross-Language Code Entity Processing.
"""
from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from stt.core.config import setup_logging
from stt.text_formatting.constants import get_resources

logger = setup_logging(__name__)


class CodeSymbolType(Enum):
    """Types of code symbols that can be mapped universally."""
    SLASH = "/"
    UNDERSCORE = "_" 
    DASH = "-"
    DOT = "."
    COLON = ":"
    AT_SIGN = "@"
    AMPERSAND = "&"
    EQUALS = "="
    PLUS = "+"
    MINUS = "-"
    QUESTION_MARK = "?"
    
    # Operators
    INCREMENT = "++"
    DECREMENT = "--"
    EQUALS_EQUALS = "=="
    
    # Mathematical operators
    TIMES = "×"
    DIVIDED_BY = "÷"
    SQUARED = "²"
    CUBED = "³"
    POWER = "^"


@dataclass
class CodeSymbolMapping:
    """Represents a mapping from spoken words to code symbols."""
    spoken_phrases: List[str]  # List of spoken variations
    symbol: str               # The actual code symbol
    symbol_type: CodeSymbolType
    language: str
    context_hints: List[str] = None  # Context words that indicate this is code


@dataclass
class UniversalCodePattern:
    """A universal code pattern that works across languages."""
    pattern_name: str
    symbol_types: List[CodeSymbolType]  # What symbols this pattern uses
    pattern_builder: callable          # Function to build regex pattern
    description: str


class UniversalCodeMapper:
    """
    Universal Code Symbol Mapping Registry
    
    Provides smart, extensible system for code entity processing across ALL languages.
    This centralizes all code symbol mappings and provides universal pattern building.
    """
    
    def __init__(self):
        """Initialize the universal code mapper."""
        self._symbol_mappings: Dict[str, Dict[CodeSymbolType, List[CodeSymbolMapping]]] = {}
        self._compiled_patterns: Dict[str, Dict[str, re.Pattern]] = {}
        self._context_cache: Dict[str, Set[str]] = {}
        self._load_all_language_mappings()
    
    def _load_all_language_mappings(self) -> None:
        """Load code symbol mappings for all available languages."""
        # Currently supported languages - easily extensible
        languages = ["en", "es"]  # Can add "fr", "de", "it", "pt", etc.
        
        for language in languages:
            logger.debug(f"Loading code mappings for language: {language}")
            self._load_language_mappings(language)
    
    def _load_language_mappings(self, language: str) -> None:
        """Load code symbol mappings for a specific language."""
        if language not in self._symbol_mappings:
            self._symbol_mappings[language] = {}
        
        resources = get_resources(language)
        
        # Load mappings from different resource sections
        self._load_code_keywords(language, resources)
        self._load_url_keywords(language, resources)
        self._load_operators(language, resources)
        self._load_mathematical_operations(language, resources)
    
    def _load_code_keywords(self, language: str, resources: Dict[str, Any]) -> None:
        """Load code keywords from resources."""
        code_keywords = resources.get("spoken_keywords", {}).get("code", {})
        
        for spoken_phrase, symbol in code_keywords.items():
            symbol_type = self._determine_symbol_type(symbol)
            if symbol_type:
                mapping = CodeSymbolMapping(
                    spoken_phrases=[spoken_phrase],
                    symbol=symbol,
                    symbol_type=symbol_type,
                    language=language
                )
                self._add_mapping(language, symbol_type, mapping)
    
    def _load_url_keywords(self, language: str, resources: Dict[str, Any]) -> None:
        """Load URL keywords that are also used in code contexts."""
        url_keywords = resources.get("spoken_keywords", {}).get("url", {})
        
        for spoken_phrase, symbol in url_keywords.items():
            symbol_type = self._determine_symbol_type(symbol)
            if symbol_type:
                mapping = CodeSymbolMapping(
                    spoken_phrases=[spoken_phrase],
                    symbol=symbol,
                    symbol_type=symbol_type,
                    language=language,
                    context_hints=["url", "web", "domain", "email", "http", "https", "www"]
                )
                self._add_mapping(language, symbol_type, mapping)
    
    def _load_operators(self, language: str, resources: Dict[str, Any]) -> None:
        """Load operators from resources."""
        operators = resources.get("spoken_keywords", {}).get("operators", {})
        
        for spoken_phrase, symbol in operators.items():
            symbol_type = self._determine_symbol_type(symbol)
            if symbol_type:
                mapping = CodeSymbolMapping(
                    spoken_phrases=[spoken_phrase],
                    symbol=symbol,
                    symbol_type=symbol_type,
                    language=language,
                    context_hints=["code", "programming", "variable", "function", "assignment"]
                )
                self._add_mapping(language, symbol_type, mapping)
    
    def _load_mathematical_operations(self, language: str, resources: Dict[str, Any]) -> None:
        """Load mathematical operations that are also used in code."""
        math_ops = resources.get("spoken_keywords", {}).get("mathematical", {}).get("operations", {})
        
        for spoken_phrase, symbol in math_ops.items():
            symbol_type = self._determine_symbol_type(symbol)
            if symbol_type:
                mapping = CodeSymbolMapping(
                    spoken_phrases=[spoken_phrase],
                    symbol=symbol,
                    symbol_type=symbol_type,
                    language=language,
                    context_hints=["math", "calculation", "expression", "formula"]
                )
                self._add_mapping(language, symbol_type, mapping)
    
    def _determine_symbol_type(self, symbol: str) -> Optional[CodeSymbolType]:
        """Determine the CodeSymbolType for a given symbol."""
        type_mapping = {
            "/": CodeSymbolType.SLASH,
            "_": CodeSymbolType.UNDERSCORE,
            "-": CodeSymbolType.DASH,
            ".": CodeSymbolType.DOT,
            ":": CodeSymbolType.COLON,
            "@": CodeSymbolType.AT_SIGN,
            "&": CodeSymbolType.AMPERSAND,
            "=": CodeSymbolType.EQUALS,
            "+": CodeSymbolType.PLUS,
            "?": CodeSymbolType.QUESTION_MARK,
            "++": CodeSymbolType.INCREMENT,
            "--": CodeSymbolType.DECREMENT,
            "==": CodeSymbolType.EQUALS_EQUALS,
            "×": CodeSymbolType.TIMES,
            "÷": CodeSymbolType.DIVIDED_BY,
            "²": CodeSymbolType.SQUARED,
            "³": CodeSymbolType.CUBED,
            "^": CodeSymbolType.POWER,
        }
        return type_mapping.get(symbol)
    
    def _add_mapping(self, language: str, symbol_type: CodeSymbolType, mapping: CodeSymbolMapping) -> None:
        """Add a code symbol mapping to the registry."""
        if symbol_type not in self._symbol_mappings[language]:
            self._symbol_mappings[language][symbol_type] = []
        
        self._symbol_mappings[language][symbol_type].append(mapping)
    
    def get_spoken_phrases_for_symbol(self, language: str, symbol_type: CodeSymbolType) -> List[str]:
        """Get all spoken phrases that map to a specific symbol type in a language."""
        mappings = self._symbol_mappings.get(language, {}).get(symbol_type, [])
        phrases = []
        for mapping in mappings:
            phrases.extend(mapping.spoken_phrases)
        return sorted(phrases, key=len, reverse=True)  # Longest first for regex
    
    def get_symbol_for_type(self, symbol_type: CodeSymbolType) -> str:
        """Get the actual symbol for a symbol type."""
        return symbol_type.value
    
    def build_universal_pattern(self, language: str, symbol_types: List[CodeSymbolType], 
                               pattern_template: str) -> re.Pattern[str]:
        """
        Build a universal regex pattern for the specified symbol types and language.
        
        Args:
            language: Language code (e.g., "en", "es")
            symbol_types: List of symbol types to include in pattern
            pattern_template: Template string with placeholders like {SLASH_KEYWORDS}
            
        Returns:
            Compiled regex pattern
        """
        # Build keyword mappings for template substitution
        substitutions = {}
        
        for symbol_type in symbol_types:
            phrases = self.get_spoken_phrases_for_symbol(language, symbol_type)
            if phrases:
                escaped_phrases = [re.escape(phrase) for phrase in phrases]
                keyword_pattern = f"(?:{'|'.join(escaped_phrases)})"
                substitutions[f"{symbol_type.name}_KEYWORDS"] = keyword_pattern
            else:
                logger.warning(f"No phrases found for {symbol_type} in language {language}")
                substitutions[f"{symbol_type.name}_KEYWORDS"] = "(?:__NO_MATCH__)"
        
        # Substitute into template
        try:
            pattern_string = pattern_template.format(**substitutions)
            return re.compile(pattern_string, re.VERBOSE | re.IGNORECASE)
        except KeyError as e:
            logger.error(f"Template substitution failed: missing key {e}")
            raise ValueError(f"Invalid pattern template - missing placeholder: {e}")
    
    def detect_code_context(self, text: str, language: str) -> Dict[str, float]:
        """
        Analyze text to detect code context indicators.
        
        Returns:
            Dictionary mapping context types to confidence scores (0.0-1.0)
        """
        text_lower = text.lower()
        contexts = {
            "programming": 0.0,
            "web_url": 0.0,
            "file_path": 0.0,
            "command_line": 0.0,
            "mathematical": 0.0
        }
        
        # Programming context indicators
        prog_indicators = [
            "variable", "función", "function", "method", "método", "class", "clase",
            "assignment", "asignación", "código", "code", "programming", "programación"
        ]
        for indicator in prog_indicators:
            if indicator in text_lower:
                contexts["programming"] += 0.2
        
        # Web/URL context indicators
        web_indicators = ["http", "www", "com", "org", "net", "url", "website", "domain"]
        for indicator in web_indicators:
            if indicator in text_lower:
                contexts["web_url"] += 0.3
        
        # File path context indicators
        file_indicators = ["archivo", "file", "path", "directorio", "directory", "folder", "carpeta"]
        for indicator in file_indicators:
            if indicator in text_lower:
                contexts["file_path"] += 0.2
        
        # Command line context indicators
        cmd_indicators = ["command", "comando", "execute", "ejecutar", "run", "correr", "terminal"]
        for indicator in cmd_indicators:
            if indicator in text_lower:
                contexts["command_line"] += 0.3
        
        # Mathematical context indicators  
        math_indicators = ["calculate", "calcular", "math", "mathematics", "matemáticas", "equation", "ecuación"]
        for indicator in math_indicators:
            if indicator in text_lower:
                contexts["mathematical"] += 0.2
        
        # Cap at 1.0
        return {k: min(v, 1.0) for k, v in contexts.items()}
    
    def is_code_entity_likely(self, text: str, language: str, symbol_types: List[CodeSymbolType]) -> bool:
        """
        Determine if detected symbol usage in text is likely a code entity vs natural language.
        
        This implements context-aware code processing that distinguishes code from natural language.
        """
        contexts = self.detect_code_context(text, language)
        
        # If any strong code context indicator is present, it's likely code
        strong_contexts = ["programming", "web_url", "command_line"]
        for context in strong_contexts:
            if contexts[context] >= 0.3:
                return True
        
        # Special handling for different symbol types
        if CodeSymbolType.SLASH in symbol_types:
            # Slash is likely code if not in math context or URL context
            if contexts["mathematical"] < 0.2 or contexts["web_url"] >= 0.3:
                return True
        
        if CodeSymbolType.UNDERSCORE in symbol_types:
            # Underscore is almost always code in programming contexts
            if contexts["programming"] >= 0.2:
                return True
        
        # Default to false for ambiguous cases
        return False
    
    def get_supported_languages(self) -> List[str]:
        """Get list of all supported languages."""
        return list(self._symbol_mappings.keys())
    
    def get_available_symbol_types(self, language: str) -> List[CodeSymbolType]:
        """Get all available symbol types for a language."""
        return list(self._symbol_mappings.get(language, {}).keys())
    
    def debug_mappings(self, language: str) -> Dict[str, Any]:
        """Get debug information about loaded mappings for a language."""
        if language not in self._symbol_mappings:
            return {"error": f"No mappings for language {language}"}
        
        debug_info = {}
        for symbol_type, mappings in self._symbol_mappings[language].items():
            debug_info[symbol_type.name] = {
                "symbol": symbol_type.value,
                "phrases": []
            }
            for mapping in mappings:
                debug_info[symbol_type.name]["phrases"].extend(mapping.spoken_phrases)
        
        return debug_info


# Global instance - singleton pattern for efficiency
_universal_mapper: Optional[UniversalCodeMapper] = None


def get_universal_code_mapper() -> UniversalCodeMapper:
    """Get the global universal code mapper instance."""
    global _universal_mapper
    if _universal_mapper is None:
        _universal_mapper = UniversalCodeMapper()
        logger.info("Universal Code Mapper initialized")
    return _universal_mapper


# Convenience functions for common operations
def get_code_pattern_for_language(language: str, pattern_name: str, 
                                symbol_types: List[CodeSymbolType],
                                pattern_template: str) -> re.Pattern[str]:
    """Build a code pattern for a specific language - convenience function."""
    mapper = get_universal_code_mapper()
    return mapper.build_universal_pattern(language, symbol_types, pattern_template)


def detect_language_code_entities(text: str, language: str) -> Dict[str, Any]:
    """Detect and analyze code entities in text for a specific language."""
    mapper = get_universal_code_mapper()
    
    results = {
        "language": language,
        "context": mapper.detect_code_context(text, language),
        "available_symbols": mapper.get_available_symbol_types(language),
        "text_analysis": text
    }
    
    return results