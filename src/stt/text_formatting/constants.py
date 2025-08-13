#!/usr/bin/env python3
"""Shared constants for text formatting modules."""
from __future__ import annotations

# Standard library imports
import json
import os
import threading
from functools import lru_cache
from typing import Any, Optional

# ==============================================================================
# I18N RESOURCE LOADER
# ==============================================================================

_RESOURCES: dict[str, dict[str, Any]] = {}  # Cache for loaded languages
_RESOURCE_PATH = os.path.join(os.path.dirname(__file__), "resources")
_LOCK = threading.Lock()  # For thread-safe lazy loading

# Enhanced caching for nested resource access
_NESTED_CACHE: dict[str, Any] = {}  # Cache for nested resource paths
_NESTED_LOCK = threading.Lock()  # For thread-safe nested access
_ACCESS_STATS: dict[str, int] = {}  # Track resource access frequency

# Cache to suppress repeated warnings for missing language files
_WARNED_LANGUAGES: set[str] = set()


def get_resources(language: str = "en") -> dict[str, Any]:
    """
    Loads and caches language-specific resources from a JSON file.
    This is the single point of entry for all language-dependent constants.

    Args:
        language: Language code (e.g., 'en', 'es', 'fr')

    Returns:
        dict: Loaded language resources

    Raises:
        ValueError: If default language resource not found

    """
    # Return from cache if already loaded
    if language in _RESOURCES:
        return _RESOURCES[language]

    # Thread-safe block to load a new language
    with _LOCK:
        # Double-check if another thread loaded it while we were waiting
        if language in _RESOURCES:
            return _RESOURCES[language]

        try:
            filepath = os.path.join(_RESOURCE_PATH, f"{language}.json")
            with open(filepath, encoding="utf-8") as f:
                resources: dict[str, Any] = json.load(f)
                _RESOURCES[language] = resources
            return resources
        except FileNotFoundError:
            # Fallback to English if the requested language is not found
            if language != "en":
                # Only warn once per language to avoid spam
                if language not in _WARNED_LANGUAGES:
                    _WARNED_LANGUAGES.add(language)
                    print(f"Warning: Language resource '{language}.json' not found. Falling back to 'en'.")
                return get_resources("en")
            raise ValueError("Default language resource 'en.json' not found.") from None
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {language}.json") from e


def get_nested_resource(language: str, *keys: str) -> Any:
    """
    Optimized nested resource access with caching.
    
    Args:
        language: Language code (e.g., 'en', 'es')
        *keys: Sequence of nested keys to access (e.g., "spoken_keywords", "code")
    
    Returns:
        The nested resource value
    
    Example:
        # Instead of: resources["spoken_keywords"]["code"]
        # Use: get_nested_resource("en", "spoken_keywords", "code")
    """
    # Create cache key for this specific nested path
    cache_key = f"{language}::{':'.join(keys)}"
    
    # Check cache first
    with _NESTED_LOCK:
        if cache_key in _NESTED_CACHE:
            _ACCESS_STATS[cache_key] = _ACCESS_STATS.get(cache_key, 0) + 1
            return _NESTED_CACHE[cache_key]
    
    # Get base resources (this is cached by get_resources)
    resources = get_resources(language)
    
    # Navigate the nested structure
    result = resources
    try:
        for key in keys:
            result = result[key]
    except (KeyError, TypeError) as e:
        raise KeyError(f"Nested resource path not found: {cache_key}") from e
    
    # Cache the result
    with _NESTED_LOCK:
        _NESTED_CACHE[cache_key] = result
        _ACCESS_STATS[cache_key] = _ACCESS_STATS.get(cache_key, 0) + 1
    
    return result


@lru_cache(maxsize=128)
def get_cached_resource_path(language: str, path: str) -> Any:
    """
    LRU cached access for specific resource paths.
    
    Args:
        language: Language code
        path: Dot-separated path (e.g., "spoken_keywords.code")
    
    Returns:
        The resource at the specified path
    """
    keys = path.split('.')
    return get_nested_resource(language, *keys)


def preload_common_resources(language: str = "en") -> None:
    """
    Preload commonly accessed resources for improved performance.
    
    Args:
        language: Language code to preload resources for
    """
    common_paths = [
        ("spoken_keywords", "code"),
        ("spoken_keywords", "url"),
        ("spoken_keywords", "operators"),
        ("spoken_keywords", "letters"),
        ("number_words", "ones"),
        ("number_words", "teens"),
        ("number_words", "tens"),
        ("currency",),
        ("data_units",),
        ("abbreviations",),
        ("context_words",),
        ("temporal",),
        ("units", "length"),
        ("units", "weight"),
        ("units", "volume"),
        ("technical", "programming"),
    ]
    
    for path in common_paths:
        try:
            get_nested_resource(language, *path)
        except (KeyError, ValueError):
            # Skip paths that don't exist in this language
            continue


def get_resource_access_stats() -> dict[str, int]:
    """
    Get statistics on resource access frequency.
    
    Returns:
        Dictionary mapping resource paths to access counts
    """
    with _NESTED_LOCK:
        return _ACCESS_STATS.copy()


def clear_resource_caches() -> None:
    """
    Clear all resource caches. Useful for testing or memory management.
    """
    with _LOCK:
        _RESOURCES.clear()
    
    with _NESTED_LOCK:
        _NESTED_CACHE.clear()
        _ACCESS_STATS.clear()
    
    # Clear the LRU cache as well
    get_cached_resource_path.cache_clear()


def get_resource_cache_info() -> dict[str, Any]:
    """
    Get information about resource cache usage.
    
    Returns:
        Dictionary with cache statistics and information
    """
    lru_info = get_cached_resource_path.cache_info()
    
    with _NESTED_LOCK:
        nested_cache_size = len(_NESTED_CACHE)
        total_access_count = sum(_ACCESS_STATS.values())
        top_accessed = sorted(_ACCESS_STATS.items(), key=lambda x: x[1], reverse=True)[:10]
    
    with _LOCK:
        languages_loaded = list(_RESOURCES.keys())
    
    return {
        "languages_loaded": languages_loaded,
        "nested_cache_size": nested_cache_size,
        "total_access_count": total_access_count,
        "lru_cache_hits": lru_info.hits,
        "lru_cache_misses": lru_info.misses,
        "lru_cache_size": lru_info.currsize,
        "lru_cache_maxsize": lru_info.maxsize,
        "top_accessed_paths": top_accessed,
    }


# ==============================================================================
# REGIONAL MEASUREMENT PREFERENCES
# ==============================================================================

def get_regional_defaults(language: str) -> dict[str, str]:
    """
    Get regional measurement defaults for language code.
    
    Args:
        language: Language code (e.g., 'en-US', 'en-GB', 'en-CA')
    
    Returns:
        Dictionary with regional defaults for temperature, length, weight
    
    Example:
        >>> get_regional_defaults("en-US")
        {"temperature": "fahrenheit", "length": "imperial", "weight": "imperial"}
    """
    try:
        # Check if language-specific resource file exists
        filepath = os.path.join(_RESOURCE_PATH, f"{language}.json")
        if os.path.exists(filepath):
            resources = get_resources(language)
            regional_defaults = resources.get("regional_defaults", {})
            if regional_defaults:
                return regional_defaults
        
        # Smart fallback for unsupported English variants
        if language.startswith("en-") and language not in ["en-US"]:
            # Most English variants outside US use metric system
            return {
                "temperature": "celsius",
                "length": "metric", 
                "weight": "metric"
            }
        
        # For non-English languages, use metric as global standard
        if not language.startswith("en"):
            return {
                "temperature": "celsius",
                "length": "metric",
                "weight": "metric" 
            }
        
        # Fallback to actual resource loading for supported languages
        resources = get_resources(language)
        return resources.get("regional_defaults", {})
        
    except Exception:
        # If anything fails, provide sensible metric defaults
        return {
            "temperature": "celsius",
            "length": "metric",
            "weight": "metric"
        }


def get_measurement_preference(language: str, measurement_type: str, user_override: Optional[str] = None) -> str:
    """
    Get measurement preference with fallback chain.
    
    Precedence: User override → Regional default → System fallback (metric)
    
    Args:
        language: Language code (e.g., 'en-US', 'en-GB')
        measurement_type: Type of measurement ('temperature', 'length', 'weight')
        user_override: Optional user preference override
    
    Returns:
        Measurement unit preference string
    
    Example:
        >>> get_measurement_preference("en-US", "temperature", "celsius")
        "celsius"  # User override takes precedence
        
        >>> get_measurement_preference("en-US", "temperature")
        "fahrenheit"  # Regional default for US
        
        >>> get_measurement_preference("en-GB", "temperature")
        "celsius"  # Regional default for UK
    """
    # Validate user override first
    valid_units = {
        "temperature": ["celsius", "fahrenheit"],
        "length": ["metric", "imperial"], 
        "weight": ["metric", "imperial"]
    }
    
    if user_override and user_override in valid_units.get(measurement_type, []):
        return user_override
        
    # Fall back to regional default
    regional_defaults = get_regional_defaults(language)
    return regional_defaults.get(measurement_type, "metric")  # System fallback
