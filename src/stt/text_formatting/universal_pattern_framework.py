#!/usr/bin/env python3
"""
Universal Pattern Framework for Language-Agnostic Text Formatting

This module provides a comprehensive framework for creating language-agnostic
text formatting patterns that can be extended for any language through
inheritance and composition.

Key Components:
- Universal base pattern classes
- Language-specific pattern factories
- Pattern inheritance and composition system
- Multi-language validation framework
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Pattern, Protocol, Type, Union, Optional
from enum import Enum

from stt.core.config import setup_logging
from .constants import get_resources, get_nested_resource

logger = setup_logging(__name__)


# ==============================================================================
# PATTERN FRAMEWORK TYPES AND PROTOCOLS
# ==============================================================================

class PatternType(Enum):
    """Universal pattern types that work across all languages."""
    MATHEMATICAL = "mathematical"
    NUMERICAL = "numerical"
    TEMPORAL = "temporal"
    CURRENCY = "currency"
    WEB = "web"
    CODE = "code"
    LINGUISTIC = "linguistic"
    TECHNICAL = "technical"


class PatternPriority(Enum):
    """Pattern priority levels for conflict resolution."""
    CRITICAL = 500  # Must be processed first (e.g., idioms)
    HIGH = 400      # Important patterns (e.g., math expressions)
    MEDIUM = 300    # Standard patterns (e.g., URLs, emails)
    LOW = 200       # Basic patterns (e.g., numbers)
    FALLBACK = 100  # Last resort patterns


@dataclass
class PatternMetadata:
    """Metadata for pattern definitions."""
    name: str
    pattern_type: PatternType
    priority: PatternPriority
    languages: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    conflicts_with: List[str] = field(default_factory=list)
    description: str = ""


class LanguageResourceProvider(Protocol):
    """Protocol for language resource providers."""
    
    def get_keywords(self, category: str, subcategory: str = None) -> Dict[str, str]:
        """Get keyword mappings for a category."""
        ...
    
    def get_word_lists(self, category: str) -> List[str]:
        """Get word lists for a category."""
        ...
    
    def get_patterns(self, category: str) -> List[str]:
        """Get regex pattern strings for a category."""
        ...


# ==============================================================================
# UNIVERSAL PATTERN BASE CLASSES
# ==============================================================================

class UniversalPattern(ABC):
    """
    Abstract base class for all universal patterns.
    
    This class provides the foundation for language-agnostic pattern definitions
    that can be specialized for specific languages through inheritance.
    """
    
    def __init__(self, language: str = "en", metadata: Optional[PatternMetadata] = None):
        self.language = language
        self.metadata = metadata or self._default_metadata()
        self._resource_provider = self._create_resource_provider()
        self._compiled_pattern: Optional[Pattern[str]] = None
        self._pattern_cache: Dict[str, Pattern[str]] = {}
    
    @abstractmethod
    def _default_metadata(self) -> PatternMetadata:
        """Return default metadata for this pattern type."""
        pass
    
    @abstractmethod
    def _build_pattern_components(self) -> Dict[str, str]:
        """Build language-specific pattern components."""
        pass
    
    @abstractmethod
    def _compile_pattern(self, components: Dict[str, str]) -> Pattern[str]:
        """Compile the final regex pattern from components."""
        pass
    
    def _create_resource_provider(self) -> LanguageResourceProvider:
        """Create a resource provider for this pattern's language."""
        return StandardResourceProvider(self.language)
    
    def get_pattern(self) -> Pattern[str]:
        """Get the compiled regex pattern, building it if necessary."""
        if self._compiled_pattern is None:
            components = self._build_pattern_components()
            self._compiled_pattern = self._compile_pattern(components)
        return self._compiled_pattern
    
    def validate_against_language(self, test_cases: List[str]) -> Dict[str, bool]:
        """Validate the pattern against language-specific test cases."""
        pattern = self.get_pattern()
        results = {}
        
        for test_case in test_cases:
            try:
                match = pattern.search(test_case)
                results[test_case] = match is not None
            except Exception as e:
                logger.warning(f"Pattern validation error for '{test_case}': {e}")
                results[test_case] = False
        
        return results
    
    def get_pattern_info(self) -> Dict[str, Any]:
        """Get comprehensive information about this pattern."""
        return {
            "language": self.language,
            "metadata": self.metadata,
            "pattern": self.get_pattern().pattern,
            "flags": self.get_pattern().flags,
            "type": self.__class__.__name__
        }


class MathematicalPattern(UniversalPattern):
    """Universal mathematical expression pattern."""
    
    def _default_metadata(self) -> PatternMetadata:
        return PatternMetadata(
            name="mathematical_expression",
            pattern_type=PatternType.MATHEMATICAL,
            priority=PatternPriority.HIGH,
            description="Universal mathematical expression pattern"
        )
    
    def _build_pattern_components(self) -> Dict[str, str]:
        """Build mathematical pattern components using language resources."""
        try:
            # Get mathematical operations from resources
            operations = self._resource_provider.get_keywords("spoken_keywords", "mathematical")
            if "operations" in operations:
                operations = operations["operations"]
            
            # Build operator patterns
            basic_ops = []
            power_ops = []
            comparison_ops = []
            
            for spoken, symbol in operations.items():
                escaped = re.escape(spoken)
                if any(word in spoken.lower() for word in ["squared", "cubed", "power"]):
                    power_ops.append(escaped)
                elif any(word in spoken.lower() for word in ["equals", "equal"]):
                    comparison_ops.append(escaped)
                else:
                    basic_ops.append(escaped)
            
            # Get number words for this language
            number_words = self._get_number_words()
            
            return {
                "basic_operators": "|".join(basic_ops) if basic_ops else "plus|minus|times",
                "power_operators": "|".join(power_ops) if power_ops else "squared|cubed",
                "comparison_operators": "|".join(comparison_ops) if comparison_ops else "equals",
                "number_words": "|".join(number_words),
                "variables": r"[a-zA-Z][\w]*",
                "digits": r"\d+"
            }
        except Exception as e:
            logger.warning(f"Failed to build math components for {self.language}: {e}")
            return self._fallback_components()
    
    def _get_number_words(self) -> List[str]:
        """Get number words for this language."""
        try:
            number_data = self._resource_provider.get_keywords("number_words", "ones")
            return list(number_data.keys()) if number_data else []
        except:
            return ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    
    def _fallback_components(self) -> Dict[str, str]:
        """Fallback components for when resources are unavailable."""
        return {
            "basic_operators": "plus|minus|times|divided by",
            "power_operators": "squared|cubed",
            "comparison_operators": "equals",
            "number_words": "zero|one|two|three|four|five|six|seven|eight|nine",
            "variables": r"[a-zA-Z][\w]*",
            "digits": r"\d+"
        }
    
    def _compile_pattern(self, components: Dict[str, str]) -> Pattern[str]:
        """Compile mathematical expression pattern."""
        pattern = rf"""
        \b                                      # Word boundary
        (?:                                     # Mathematical expression
            (?:{components['number_words']}|{components['digits']}|{components['variables']})  # First operand
            \s+                                 # Space
            (?:{components['basic_operators']}) # Operator
            \s+                                 # Space
            (?:{components['number_words']}|{components['digits']}|{components['variables']})  # Second operand
            (?:\s+(?:{components['power_operators']}))?  # Optional power
            |                                   # OR
            (?:{components['number_words']}|{components['digits']}|{components['variables']})  # Variable
            \s+                                 # Space
            (?:{components['comparison_operators']})     # Equals
            \s+                                 # Space
            (?:{components['number_words']}|{components['digits']}|{components['variables']})  # Value
        )
        \b                                      # Word boundary
        """
        
        return re.compile(pattern, re.VERBOSE | re.IGNORECASE)


class CurrencyPattern(UniversalPattern):
    """Universal currency pattern."""
    
    def _default_metadata(self) -> PatternMetadata:
        return PatternMetadata(
            name="currency_expression",
            pattern_type=PatternType.CURRENCY,
            priority=PatternPriority.MEDIUM,
            description="Universal currency pattern"
        )
    
    def _build_pattern_components(self) -> Dict[str, str]:
        """Build currency pattern components."""
        try:
            # Get currency units from resources
            currency_units = self._resource_provider.get_word_lists("currency")
            if not currency_units:
                currency_map = self._resource_provider.get_keywords("units", "currency_map")
                currency_units = list(currency_map.keys()) if currency_map else []
            
            # Get number words
            number_words = self._get_number_words()
            
            return {
                "currency_units": "|".join(re.escape(unit) for unit in currency_units),
                "number_words": "|".join(number_words),
                "digits": r"\d+(?:\.\d+)?",
                "currency_symbols": r"[$£€¥₹₩₱₽₣₺₪]"
            }
        except Exception as e:
            logger.warning(f"Failed to build currency components for {self.language}: {e}")
            return self._fallback_currency_components()
    
    def _get_number_words(self) -> List[str]:
        """Get number words for currency amounts."""
        try:
            number_data = self._resource_provider.get_keywords("number_words", "ones")
            return list(number_data.keys()) if number_data else []
        except:
            return ["zero", "one", "two", "three", "four", "five"]
    
    def _fallback_currency_components(self) -> Dict[str, str]:
        """Fallback currency components."""
        return {
            "currency_units": "dollar|dollars|pound|pounds|euro|euros",
            "number_words": "zero|one|two|three|four|five|six|seven|eight|nine",
            "digits": r"\d+(?:\.\d+)?",
            "currency_symbols": r"[$£€¥₹₩₱₽₣₺₪]"
        }
    
    def _compile_pattern(self, components: Dict[str, str]) -> Pattern[str]:
        """Compile currency pattern."""
        pattern = rf"""
        \b                                      # Word boundary
        (?:                                     # Currency amount
            (?:{components['number_words']}|{components['digits']})  # Amount
            \s+                                 # Space
            (?:{components['currency_units']})  # Currency unit
            |                                   # OR
            {components['currency_symbols']}    # Currency symbol
            \s*                                 # Optional space
            (?:{components['digits']})          # Amount
        )
        \b                                      # Word boundary
        """
        
        return re.compile(pattern, re.VERBOSE | re.IGNORECASE)


class WebPattern(UniversalPattern):
    """Universal web URL/email pattern."""
    
    def _default_metadata(self) -> PatternMetadata:
        return PatternMetadata(
            name="web_url_email",
            pattern_type=PatternType.WEB,
            priority=PatternPriority.MEDIUM,
            description="Universal web URL and email pattern"
        )
    
    def _build_pattern_components(self) -> Dict[str, str]:
        """Build web pattern components."""
        try:
            # Get URL keywords from resources
            url_keywords = self._resource_provider.get_keywords("spoken_keywords", "url")
            
            # Build URL component mappings
            url_components = {}
            for spoken, symbol in url_keywords.items():
                if symbol == ".":
                    url_components["dot"] = spoken
                elif symbol == "@":
                    url_components["at"] = spoken
                elif symbol == "/":
                    url_components["slash"] = spoken
                elif symbol == ":":
                    url_components["colon"] = spoken
            
            # Get top-level domains
            tlds = self._resource_provider.get_word_lists("top_level_domains")
            
            return {
                "dot_word": url_components.get("dot", "dot"),
                "at_word": url_components.get("at", "at"),
                "slash_word": url_components.get("slash", "slash"),
                "colon_word": url_components.get("colon", "colon"),
                "tlds": "|".join(tlds) if tlds else "com|org|net",
                "domain_chars": r"[a-zA-Z0-9\-]+",
                "protocol_words": "http|https|ftp"
            }
        except Exception as e:
            logger.warning(f"Failed to build web components for {self.language}: {e}")
            return self._fallback_web_components()
    
    def _fallback_web_components(self) -> Dict[str, str]:
        """Fallback web components."""
        return {
            "dot_word": "dot",
            "at_word": "at",
            "slash_word": "slash", 
            "colon_word": "colon",
            "tlds": "com|org|net|edu|gov",
            "domain_chars": r"[a-zA-Z0-9\-]+",
            "protocol_words": "http|https|ftp"
        }
    
    def _compile_pattern(self, components: Dict[str, str]) -> Pattern[str]:
        """Compile web pattern."""
        pattern = rf"""
        \b                                      # Word boundary
        (?:                                     # Web entity
            # Email pattern
            {components['domain_chars']}        # Username
            \s+{components['at_word']}\s+       # " at "
            {components['domain_chars']}        # Domain
            \s+{components['dot_word']}\s+      # " dot "
            (?:{components['tlds']})            # TLD
            |                                   # OR
            # URL pattern
            (?:{components['protocol_words']}   # Protocol
            \s+{components['colon_word']}\s+    # " colon "
            {components['slash_word']}\s+       # " slash "
            {components['slash_word']}\s+)?     # " slash "
            {components['domain_chars']}        # Domain
            \s+{components['dot_word']}\s+      # " dot "
            (?:{components['tlds']})            # TLD
        )
        \b                                      # Word boundary
        """
        
        return re.compile(pattern, re.VERBOSE | re.IGNORECASE)


# ==============================================================================
# RESOURCE PROVIDER IMPLEMENTATIONS
# ==============================================================================

class StandardResourceProvider:
    """Standard implementation of LanguageResourceProvider using the existing resource system."""
    
    def __init__(self, language: str):
        self.language = language
        self._resources = get_resources(language)
    
    def get_keywords(self, category: str, subcategory: str = None) -> Dict[str, str]:
        """Get keyword mappings for a category."""
        try:
            if subcategory:
                return get_nested_resource(self.language, category, subcategory)
            else:
                return get_nested_resource(self.language, category)
        except (KeyError, ValueError):
            return {}
    
    def get_word_lists(self, category: str) -> List[str]:
        """Get word lists for a category."""
        try:
            data = get_nested_resource(self.language, category)
            if isinstance(data, dict):
                return list(data.get("units", data.keys()))
            elif isinstance(data, list):
                return data
            else:
                return []
        except (KeyError, ValueError):
            return []
    
    def get_patterns(self, category: str) -> List[str]:
        """Get regex pattern strings for a category."""
        try:
            return get_nested_resource(self.language, category, "patterns")
        except (KeyError, ValueError):
            return []


# ==============================================================================
# PATTERN FACTORY AND REGISTRY
# ==============================================================================

class UniversalPatternFactory:
    """Factory for creating language-specific pattern instances."""
    
    _pattern_registry: Dict[str, Type[UniversalPattern]] = {
        "mathematical": MathematicalPattern,
        "currency": CurrencyPattern,
        "web": WebPattern,
    }
    
    @classmethod
    def register_pattern(cls, name: str, pattern_class: Type[UniversalPattern]):
        """Register a new pattern type."""
        cls._pattern_registry[name] = pattern_class
    
    @classmethod
    def create_pattern(cls, pattern_name: str, language: str = "en") -> UniversalPattern:
        """Create a pattern instance for a specific language."""
        if pattern_name not in cls._pattern_registry:
            raise ValueError(f"Unknown pattern type: {pattern_name}")
        
        pattern_class = cls._pattern_registry[pattern_name]
        return pattern_class(language=language)
    
    @classmethod
    def create_all_patterns(cls, language: str = "en") -> Dict[str, UniversalPattern]:
        """Create all registered patterns for a language."""
        patterns = {}
        for name in cls._pattern_registry:
            try:
                patterns[name] = cls.create_pattern(name, language)
            except Exception as e:
                logger.warning(f"Failed to create pattern {name} for {language}: {e}")
        return patterns
    
    @classmethod
    def get_available_patterns(cls) -> List[str]:
        """Get list of available pattern types."""
        return list(cls._pattern_registry.keys())


# ==============================================================================
# PATTERN VALIDATION FRAMEWORK
# ==============================================================================

@dataclass
class ValidationResult:
    """Result of pattern validation."""
    pattern_name: str
    language: str
    passed_tests: int
    total_tests: int
    success_rate: float
    failed_cases: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class CrossLanguageValidator:
    """Validator for testing patterns across multiple languages."""
    
    def __init__(self):
        self.test_cases = self._load_test_cases()
    
    def _load_test_cases(self) -> Dict[str, Dict[str, List[str]]]:
        """Load test cases for different pattern types and languages."""
        return {
            "mathematical": {
                "en": [
                    "x plus y equals z",
                    "two times three",
                    "a squared plus b squared",
                    "five divided by two"
                ],
                "es": [
                    "x más y igual z",
                    "dos por tres", 
                    "a al cuadrado más b al cuadrado",
                    "cinco dividido por dos"
                ]
            },
            "currency": {
                "en": [
                    "five dollars",
                    "ten pounds",
                    "$20",
                    "€15"
                ],
                "es": [
                    "cinco dólares",
                    "diez euros",
                    "$20",
                    "€15"
                ]
            },
            "web": {
                "en": [
                    "user at example dot com",
                    "http colon slash slash example dot org",
                    "visit github dot com"
                ],
                "es": [
                    "usuario arroba ejemplo punto com",
                    "http dos puntos barra barra ejemplo punto org",
                    "visita github punto com"
                ]
            }
        }
    
    def validate_pattern(self, pattern: UniversalPattern) -> ValidationResult:
        """Validate a single pattern against test cases."""
        pattern_type = pattern.metadata.name.split('_')[0]  # Extract base type
        language = pattern.language
        
        test_cases = self.test_cases.get(pattern_type, {}).get(language, [])
        if not test_cases:
            return ValidationResult(
                pattern_name=pattern_type,
                language=language,
                passed_tests=0,
                total_tests=0,
                success_rate=0.0,
                errors=["No test cases available"]
            )
        
        results = pattern.validate_against_language(test_cases)
        passed = sum(1 for success in results.values() if success)
        total = len(results)
        failed_cases = [case for case, success in results.items() if not success]
        
        return ValidationResult(
            pattern_name=pattern_type,
            language=language,
            passed_tests=passed,
            total_tests=total,
            success_rate=passed / total if total > 0 else 0.0,
            failed_cases=failed_cases
        )
    
    def validate_all_patterns(self, language: str = "en") -> Dict[str, ValidationResult]:
        """Validate all patterns for a language."""
        patterns = UniversalPatternFactory.create_all_patterns(language)
        results = {}
        
        for pattern_name, pattern in patterns.items():
            try:
                results[pattern_name] = self.validate_pattern(pattern)
            except Exception as e:
                results[pattern_name] = ValidationResult(
                    pattern_name=pattern_name,
                    language=language,
                    passed_tests=0,
                    total_tests=0,
                    success_rate=0.0,
                    errors=[str(e)]
                )
        
        return results
    
    def compare_languages(self, languages: List[str]) -> Dict[str, Dict[str, ValidationResult]]:
        """Compare pattern performance across multiple languages."""
        results = {}
        for language in languages:
            results[language] = self.validate_all_patterns(language)
        return results


# ==============================================================================
# PATTERN COMPOSITION AND INHERITANCE
# ==============================================================================

class CompositePattern(UniversalPattern):
    """Pattern that combines multiple sub-patterns."""
    
    def __init__(self, language: str = "en", sub_patterns: List[UniversalPattern] = None):
        super().__init__(language)
        self.sub_patterns = sub_patterns or []
    
    def add_pattern(self, pattern: UniversalPattern):
        """Add a sub-pattern to this composite."""
        self.sub_patterns.append(pattern)
    
    def _default_metadata(self) -> PatternMetadata:
        return PatternMetadata(
            name="composite_pattern",
            pattern_type=PatternType.TECHNICAL,
            priority=PatternPriority.MEDIUM,
            description="Composite pattern combining multiple sub-patterns"
        )
    
    def _build_pattern_components(self) -> Dict[str, str]:
        """Build components from all sub-patterns."""
        components = {}
        for i, pattern in enumerate(self.sub_patterns):
            pattern_components = pattern._build_pattern_components()
            for key, value in pattern_components.items():
                components[f"{pattern.__class__.__name__.lower()}_{key}"] = value
        return components
    
    def _compile_pattern(self, components: Dict[str, str]) -> Pattern[str]:
        """Compile a pattern that matches any sub-pattern."""
        sub_pattern_strings = []
        for pattern in self.sub_patterns:
            sub_pattern_strings.append(f"({pattern.get_pattern().pattern})")
        
        combined_pattern = "|".join(sub_pattern_strings)
        return re.compile(combined_pattern, re.VERBOSE | re.IGNORECASE)


class LanguageSpecificPattern(UniversalPattern):
    """Base class for language-specific pattern extensions."""
    
    def __init__(self, base_pattern: UniversalPattern, language_extensions: Dict[str, Any] = None):
        super().__init__(base_pattern.language)
        self.base_pattern = base_pattern
        self.language_extensions = language_extensions or {}
    
    def _default_metadata(self) -> PatternMetadata:
        base_metadata = self.base_pattern.metadata
        return PatternMetadata(
            name=f"{base_metadata.name}_extended",
            pattern_type=base_metadata.pattern_type,
            priority=base_metadata.priority,
            description=f"Extended {base_metadata.description}"
        )
    
    def _build_pattern_components(self) -> Dict[str, str]:
        """Build components by extending the base pattern."""
        base_components = self.base_pattern._build_pattern_components()
        
        # Apply language-specific extensions
        for key, extension in self.language_extensions.items():
            if key in base_components:
                # Extend existing component
                base_components[key] = f"{base_components[key]}|{extension}"
            else:
                # Add new component
                base_components[key] = extension
        
        return base_components
    
    def _compile_pattern(self, components: Dict[str, str]) -> Pattern[str]:
        """Use the base pattern's compilation logic."""
        return self.base_pattern._compile_pattern(components)


# ==============================================================================
# FRAMEWORK UTILITIES
# ==============================================================================

def create_language_framework(language: str = "en") -> Dict[str, Any]:
    """Create a complete language framework instance."""
    factory = UniversalPatternFactory()
    validator = CrossLanguageValidator()
    
    # Create all patterns for the language
    patterns = factory.create_all_patterns(language)
    
    # Validate patterns
    validation_results = validator.validate_all_patterns(language)
    
    return {
        "language": language,
        "patterns": patterns,
        "validation_results": validation_results,
        "factory": factory,
        "validator": validator
    }


def demonstrate_extensibility(new_language_code: str, sample_resources: Dict[str, Any]) -> Dict[str, Any]:
    """Demonstrate how to extend the framework for a new language."""
    
    # Create a temporary resource provider for the new language
    class TempResourceProvider:
        def __init__(self, resources):
            self.resources = resources
        
        def get_keywords(self, category: str, subcategory: str = None) -> Dict[str, str]:
            try:
                if subcategory:
                    return self.resources[category][subcategory]
                else:
                    return self.resources[category]
            except KeyError:
                return {}
        
        def get_word_lists(self, category: str) -> List[str]:
            try:
                data = self.resources[category]
                if isinstance(data, dict):
                    return list(data.get("units", data.keys()))
                elif isinstance(data, list):
                    return data
                else:
                    return []
            except KeyError:
                return []
        
        def get_patterns(self, category: str) -> List[str]:
            return self.resources.get(category, {}).get("patterns", [])
    
    # Create patterns with temporary resources
    temp_provider = TempResourceProvider(sample_resources)
    
    # Create a mathematical pattern for the new language
    math_pattern = MathematicalPattern(language=new_language_code)
    math_pattern._resource_provider = temp_provider
    
    # Test the pattern
    test_cases = sample_resources.get("test_cases", [])
    if test_cases:
        validation_result = math_pattern.validate_against_language(test_cases)
    else:
        validation_result = {}
    
    return {
        "language": new_language_code,
        "pattern": math_pattern,
        "pattern_info": math_pattern.get_pattern_info(),
        "validation_result": validation_result,
        "extensibility_demo": True
    }