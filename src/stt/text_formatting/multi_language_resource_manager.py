#!/usr/bin/env python3
"""
Multi-Language Resource Management System

This module provides advanced resource management capabilities for
the universal pattern framework, including resource inheritance,
composition, and dynamic loading.
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union
from pathlib import Path

from stt.core.config import setup_logging
from .constants import get_resources, _RESOURCE_PATH

logger = setup_logging(__name__)


# ==============================================================================
# RESOURCE MANAGEMENT TYPES
# ==============================================================================

@dataclass
class LanguageInfo:
    """Information about a supported language."""
    code: str
    name: str
    family: str  # Language family (e.g., "Romance", "Germanic")
    fallback_languages: List[str] = field(default_factory=list)
    supported_patterns: Set[str] = field(default_factory=set)
    resource_completeness: float = 0.0  # 0.0 to 1.0


@dataclass
class ResourceInheritance:
    """Defines how resources should inherit from other languages."""
    base_language: str
    inherited_categories: List[str] = field(default_factory=list)
    overridden_categories: List[str] = field(default_factory=list)
    merged_categories: List[str] = field(default_factory=list)


# ==============================================================================
# ABSTRACT RESOURCE INTERFACES
# ==============================================================================

class ResourceLoader(ABC):
    """Abstract interface for loading language resources."""
    
    @abstractmethod
    def load_language(self, language_code: str) -> Dict[str, Any]:
        """Load resources for a specific language."""
        pass
    
    @abstractmethod
    def get_available_languages(self) -> List[str]:
        """Get list of available language codes."""
        pass
    
    @abstractmethod
    def validate_resource_structure(self, language_code: str) -> Dict[str, Any]:
        """Validate the structure of language resources."""
        pass


class ResourceMerger(ABC):
    """Abstract interface for merging resources from multiple sources."""
    
    @abstractmethod
    def merge_resources(self, primary: Dict[str, Any], secondary: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two resource dictionaries."""
        pass
    
    @abstractmethod
    def resolve_conflicts(self, conflicts: List[str], resources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve conflicts when merging resources."""
        pass


# ==============================================================================
# CONCRETE RESOURCE IMPLEMENTATIONS
# ==============================================================================

class StandardResourceLoader(ResourceLoader):
    """Standard implementation using the existing JSON resource system."""
    
    def __init__(self, resource_path: str = None):
        self.resource_path = resource_path or _RESOURCE_PATH
        self._language_cache: Dict[str, Dict[str, Any]] = {}
        self._language_info_cache: Dict[str, LanguageInfo] = {}
        self._available_languages_cache: Optional[List[str]] = None
    
    def load_language(self, language_code: str) -> Dict[str, Any]:
        """Load resources for a specific language with caching."""
        if language_code in self._language_cache:
            return self._language_cache[language_code]
        
        try:
            resources = get_resources(language_code)
            self._language_cache[language_code] = resources
            return resources
        except Exception as e:
            logger.error(f"Failed to load resources for language {language_code}: {e}")
            return {}
    
    def get_available_languages(self) -> List[str]:
        """Get list of available language codes by scanning resource files."""
        if self._available_languages_cache is not None:
            return self._available_languages_cache
            
        if not os.path.exists(self.resource_path):
            self._available_languages_cache = ["en"]  # Fallback to English
            return self._available_languages_cache
        
        languages = []
        for file in os.listdir(self.resource_path):
            if file.endswith('.json'):
                lang_code = file[:-5]  # Remove .json extension
                languages.append(lang_code)
        
        self._available_languages_cache = sorted(languages)
        return self._available_languages_cache
    
    def validate_resource_structure(self, language_code: str) -> Dict[str, Any]:
        """Validate the structure of language resources."""
        resources = self.load_language(language_code)
        
        # Define expected top-level categories
        expected_categories = [
            "spoken_keywords", "abbreviations", "context_words", "currency",
            "number_words", "temporal", "units", "entity_priorities"
        ]
        
        validation_result = {
            "language": language_code,
            "valid": True,
            "missing_categories": [],
            "extra_categories": [],
            "category_completeness": {},
            "overall_completeness": 0.0
        }
        
        # Check for missing categories
        for category in expected_categories:
            if category not in resources:
                validation_result["missing_categories"].append(category)
                validation_result["valid"] = False
            else:
                # Check completeness of each category
                completeness = self._calculate_category_completeness(resources[category], category)
                validation_result["category_completeness"][category] = completeness
        
        # Check for extra categories
        for category in resources:
            if category not in expected_categories:
                validation_result["extra_categories"].append(category)
        
        # Calculate overall completeness
        if validation_result["category_completeness"]:
            validation_result["overall_completeness"] = sum(
                validation_result["category_completeness"].values()
            ) / len(validation_result["category_completeness"])
        
        return validation_result
    
    def _calculate_category_completeness(self, category_data: Dict[str, Any], category_name: str) -> float:
        """Calculate completeness score for a resource category."""
        if not isinstance(category_data, dict):
            return 1.0 if category_data else 0.0
        
        # Define expected subcategories for each category
        expected_subcategories = {
            "spoken_keywords": ["url", "code", "operators", "mathematical", "letters"],
            "number_words": ["ones", "tens", "scales", "digit_words"],
            "temporal": ["month_names", "relative_days", "date_keywords"],
            "units": ["currency_map", "time_units", "length_units", "weight_units"],
            "context_words": ["filename_actions", "email_actions", "programming_keywords"]
        }
        
        expected = expected_subcategories.get(category_name, [])
        if not expected:
            return 1.0  # Categories without defined structure are considered complete
        
        present = sum(1 for subcat in expected if subcat in category_data)
        return present / len(expected) if expected else 1.0
    
    def get_language_info(self, language_code: str) -> LanguageInfo:
        """Get detailed information about a language."""
        if language_code in self._language_info_cache:
            return self._language_info_cache[language_code]
        
        # Define language families and fallback hierarchies
        language_families = {
            "en": ("English", "Germanic", []),
            "es": ("Spanish", "Romance", ["en"]),
            "fr": ("French", "Romance", ["en"]),
            "de": ("German", "Germanic", ["en"]),
            "it": ("Italian", "Romance", ["es", "en"]),
            "pt": ("Portuguese", "Romance", ["es", "en"])
        }
        
        name, family, fallbacks = language_families.get(language_code, (language_code.title(), "Unknown", ["en"]))
        
        # Create basic language info without full validation to defer resource loading
        language_info = LanguageInfo(
            code=language_code,
            name=name,
            family=family,
            fallback_languages=fallbacks,
            supported_patterns=set(),  # Populate lazily when needed
            resource_completeness=0.0  # Calculate lazily when needed
        )
        
        self._language_info_cache[language_code] = language_info
        return language_info


class InheritanceResourceMerger(ResourceMerger):
    """Resource merger that supports inheritance and composition."""
    
    def merge_resources(self, primary: Dict[str, Any], secondary: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two resource dictionaries with inheritance rules."""
        merged = secondary.copy()  # Start with secondary (fallback) resources
        
        # Recursively merge primary resources
        for key, value in primary.items():
            if key in merged:
                if isinstance(value, dict) and isinstance(merged[key], dict):
                    merged[key] = self._merge_dictionaries(value, merged[key])
                else:
                    merged[key] = value  # Primary takes precedence
            else:
                merged[key] = value
        
        return merged
    
    def _merge_dictionaries(self, primary_dict: Dict[str, Any], secondary_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge two dictionaries."""
        result = secondary_dict.copy()
        
        for key, value in primary_dict.items():
            if key in result:
                if isinstance(value, dict) and isinstance(result[key], dict):
                    result[key] = self._merge_dictionaries(value, result[key])
                elif isinstance(value, list) and isinstance(result[key], list):
                    # Merge lists by extending (avoiding duplicates)
                    combined = result[key] + [item for item in value if item not in result[key]]
                    result[key] = combined
                else:
                    result[key] = value  # Primary takes precedence
            else:
                result[key] = value
        
        return result
    
    def resolve_conflicts(self, conflicts: List[str], resources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve conflicts when merging multiple resource dictionaries."""
        if not resources:
            return {}
        
        # Start with the first resource set
        merged = resources[0].copy()
        
        # Merge each subsequent resource set
        for resource_set in resources[1:]:
            merged = self.merge_resources(resource_set, merged)
        
        # Log conflicts for debugging
        for conflict in conflicts:
            logger.debug(f"Resolved conflict for key: {conflict}")
        
        return merged


# ==============================================================================
# MULTI-LANGUAGE RESOURCE MANAGER
# ==============================================================================

class MultiLanguageResourceManager:
    """
    Comprehensive resource manager for multi-language support.
    
    This class provides advanced features including:
    - Language inheritance and fallback chains
    - Resource composition and merging
    - Dynamic resource loading and validation
    - Pattern-specific resource optimization
    """
    
    def __init__(self, resource_loader: ResourceLoader = None, resource_merger: ResourceMerger = None):
        self.loader = resource_loader or StandardResourceLoader()
        self.merger = resource_merger or InheritanceResourceMerger()
        self._inheritance_rules: Dict[str, ResourceInheritance] = {}
        self._cached_merged_resources: Dict[str, Dict[str, Any]] = {}
        self._language_registry: Dict[str, LanguageInfo] = {}
        self._languages_discovered: bool = False
        
        # Defer language discovery until actually needed
    
    def _ensure_languages_discovered(self):
        """Ensure languages have been discovered (lazy loading)."""
        if not self._languages_discovered:
            self._discover_languages()
            self._languages_discovered = True
    
    def _discover_languages(self):
        """Discover and register available languages."""
        available_languages = self.loader.get_available_languages()
        
        for lang_code in available_languages:
            try:
                lang_info = self.loader.get_language_info(lang_code)
                self.register_language(lang_info)
            except Exception as e:
                logger.warning(f"Failed to register language {lang_code}: {e}")
    
    def register_language(self, language_info: LanguageInfo):
        """Register a language with the resource manager."""
        self._language_registry[language_info.code] = language_info
        self._languages_discovered = True  # Mark as discovered if manually registering
        logger.debug(f"Registered language: {language_info.name} ({language_info.code})")
    
    def set_inheritance_rule(self, language_code: str, inheritance: ResourceInheritance):
        """Set inheritance rules for a language."""
        self._inheritance_rules[language_code] = inheritance
        # Clear cached resources for this language
        if language_code in self._cached_merged_resources:
            del self._cached_merged_resources[language_code]
    
    def get_resources(self, language_code: str, use_inheritance: bool = True) -> Dict[str, Any]:
        """
        Get resources for a language with optional inheritance.
        
        Args:
            language_code: Target language code
            use_inheritance: Whether to apply inheritance rules
            
        Returns:
            Merged resource dictionary
        """
        self._ensure_languages_discovered()
        
        cache_key = f"{language_code}::{use_inheritance}"
        
        if cache_key in self._cached_merged_resources:
            return self._cached_merged_resources[cache_key]
        
        if not use_inheritance:
            resources = self.loader.load_language(language_code)
            self._cached_merged_resources[cache_key] = resources
            return resources
        
        # Apply inheritance rules
        resources = self._build_inherited_resources(language_code)
        self._cached_merged_resources[cache_key] = resources
        return resources
    
    def _build_inherited_resources(self, language_code: str) -> Dict[str, Any]:
        """Build resources with inheritance applied."""
        self._ensure_languages_discovered()
        
        # Start with the base language resources
        primary_resources = self.loader.load_language(language_code)
        
        # Get language info for fallback chain
        if language_code in self._language_registry:
            fallback_languages = self._language_registry[language_code].fallback_languages
        else:
            fallback_languages = ["en"] if language_code != "en" else []
        
        # Build inheritance chain
        resource_chain = [primary_resources]
        
        for fallback_lang in fallback_languages:
            try:
                fallback_resources = self.loader.load_language(fallback_lang)
                resource_chain.append(fallback_resources)
            except Exception as e:
                logger.warning(f"Failed to load fallback language {fallback_lang}: {e}")
        
        # Merge resources using inheritance rules
        if len(resource_chain) == 1:
            return resource_chain[0]
        
        merged_resources = resource_chain[-1]  # Start with most fallback language
        
        for resources in reversed(resource_chain[:-1]):
            merged_resources = self.merger.merge_resources(resources, merged_resources)
        
        return merged_resources
    
    def validate_language_support(self, language_code: str) -> Dict[str, Any]:
        """Validate comprehensive language support."""
        validation = self.loader.validate_resource_structure(language_code)
        
        # Add pattern-specific validation
        resources = self.get_resources(language_code)
        pattern_support = self._validate_pattern_support(resources)
        
        validation["pattern_support"] = pattern_support
        validation["inheritance_chain"] = self._get_inheritance_chain(language_code)
        
        return validation
    
    def _validate_pattern_support(self, resources: Dict[str, Any]) -> Dict[str, bool]:
        """Validate support for specific pattern types."""
        pattern_support = {}
        
        # Mathematical patterns
        pattern_support["mathematical"] = (
            "spoken_keywords" in resources and 
            "mathematical" in resources.get("spoken_keywords", {})
        )
        
        # Currency patterns
        pattern_support["currency"] = (
            "currency" in resources or 
            ("units" in resources and "currency_map" in resources.get("units", {}))
        )
        
        # Web patterns
        pattern_support["web"] = (
            "spoken_keywords" in resources and 
            "url" in resources.get("spoken_keywords", {})
        )
        
        # Code patterns
        pattern_support["code"] = (
            "spoken_keywords" in resources and 
            "code" in resources.get("spoken_keywords", {})
        )
        
        # Temporal patterns
        pattern_support["temporal"] = (
            "temporal" in resources and 
            len(resources.get("temporal", {})) > 0
        )
        
        return pattern_support
    
    def _get_inheritance_chain(self, language_code: str) -> List[str]:
        """Get the inheritance chain for a language."""
        self._ensure_languages_discovered()
        
        if language_code in self._language_registry:
            return [language_code] + self._language_registry[language_code].fallback_languages
        else:
            return [language_code, "en"] if language_code != "en" else [language_code]
    
    def get_supported_languages(self) -> List[LanguageInfo]:
        """Get list of all supported languages."""
        self._ensure_languages_discovered()
        return list(self._language_registry.values())
    
    def get_language_recommendations(self, target_patterns: List[str]) -> List[str]:
        """Recommend languages based on pattern support requirements."""
        self._ensure_languages_discovered()
        
        recommendations = []
        
        for lang_code, lang_info in self._language_registry.items():
            # Check if language supports the required patterns
            if all(pattern in lang_info.supported_patterns for pattern in target_patterns):
                recommendations.append(lang_code)
        
        # Sort by resource completeness
        recommendations.sort(
            key=lambda lang: self._language_registry[lang].resource_completeness,
            reverse=True
        )
        
        return recommendations
    
    def export_language_resources(self, language_code: str, output_path: str, include_inherited: bool = True):
        """Export language resources to a file."""
        resources = self.get_resources(language_code, use_inheritance=include_inherited)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(resources, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported resources for {language_code} to {output_path}")
    
    def create_new_language_template(self, language_code: str, base_language: str = "en") -> Dict[str, Any]:
        """Create a template for a new language based on an existing one."""
        base_resources = self.get_resources(base_language)
        
        # Create a template with placeholder translations
        template = {}
        
        for category, content in base_resources.items():
            if isinstance(content, dict):
                template[category] = self._create_template_dict(content, language_code)
            elif isinstance(content, list):
                template[category] = content.copy()  # Copy lists as-is
            else:
                template[category] = content  # Copy other types as-is
        
        return template
    
    def _create_template_dict(self, source_dict: Dict[str, Any], language_code: str) -> Dict[str, Any]:
        """Create a template dictionary with placeholder translations."""
        template = {}
        
        for key, value in source_dict.items():
            if isinstance(value, dict):
                template[key] = self._create_template_dict(value, language_code)
            elif isinstance(value, str) and key in ["dot", "at", "slash", "colon"]:
                # Keep symbolic mappings as-is
                template[key] = value
            elif isinstance(value, str):
                # Create placeholder for translation
                template[key] = f"[TRANSLATE_{language_code.upper()}]{value}"
            else:
                template[key] = value
        
        return template
    
    def get_resource_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about resource usage."""
        self._ensure_languages_discovered()
        
        stats = {
            "total_languages": len(self._language_registry),
            "average_completeness": 0.0,
            "pattern_coverage": {},
            "language_families": {},
            "inheritance_usage": len(self._inheritance_rules)
        }
        
        # Calculate average completeness
        if self._language_registry:
            completeness_sum = sum(
                lang.resource_completeness for lang in self._language_registry.values()
            )
            stats["average_completeness"] = completeness_sum / len(self._language_registry)
        
        # Calculate pattern coverage
        all_patterns = set()
        for lang_info in self._language_registry.values():
            all_patterns.update(lang_info.supported_patterns)
        
        for pattern in all_patterns:
            supporting_languages = [
                lang.code for lang in self._language_registry.values()
                if pattern in lang.supported_patterns
            ]
            stats["pattern_coverage"][pattern] = {
                "supporting_languages": supporting_languages,
                "coverage_percentage": len(supporting_languages) / len(self._language_registry) * 100
            }
        
        # Group by language families
        for lang_info in self._language_registry.values():
            family = lang_info.family
            if family not in stats["language_families"]:
                stats["language_families"][family] = []
            stats["language_families"][family].append(lang_info.code)
        
        return stats


# ==============================================================================
# GLOBAL RESOURCE MANAGER INSTANCE
# ==============================================================================

# Global instance for easy access
_global_resource_manager: Optional[MultiLanguageResourceManager] = None


def get_resource_manager() -> MultiLanguageResourceManager:
    """Get the global resource manager instance."""
    global _global_resource_manager
    if _global_resource_manager is None:
        _global_resource_manager = MultiLanguageResourceManager()
    return _global_resource_manager


def reset_resource_manager():
    """Reset the global resource manager (useful for testing)."""
    global _global_resource_manager
    _global_resource_manager = None