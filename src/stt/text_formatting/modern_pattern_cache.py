#!/usr/bin/env python3
"""
Modern Pattern Compilation Cache - Theory 10 Implementation

This module provides a sophisticated caching system for compiled regex patterns
with performance optimization and consistency improvements. It consolidates
pattern compilation across the entire text formatting system.

Features:
- Thread-safe LRU cache with automatic eviction
- Pattern compilation optimization  
- Cache warming for common patterns
- Performance monitoring and statistics
- Language-aware pattern caching
- Cross-module pattern deduplication

Theory 10 Goals:
1. Eliminate duplicate pattern compilation across modules
2. Provide consistent pattern behavior across the system
3. Improve performance with intelligent caching
4. Enable cross-language pattern optimization
"""
from __future__ import annotations

import functools
import hashlib
import re
import threading
from typing import Any, Callable, Pattern, TypeVar, Dict, List, Optional, Tuple
from collections import OrderedDict
from dataclasses import dataclass
from contextlib import contextmanager
import time

from ..core.config import setup_logging

logger = setup_logging(__name__)

# Type variable for decorated functions
F = TypeVar('F', bound=Callable[..., Pattern[str]])

@dataclass
class PatternCacheStats:
    """Statistics for pattern cache performance monitoring."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    compilation_time: float = 0.0
    cache_size: int = 0
    languages_cached: set = None
    
    def __post_init__(self):
        if self.languages_cached is None:
            self.languages_cached = set()
    
    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def miss_ratio(self) -> float:
        total = self.hits + self.misses
        return self.misses / total if total > 0 else 0.0
    
    @property
    def avg_compilation_time(self) -> float:
        return self.compilation_time / self.misses if self.misses > 0 else 0.0


class ModernPatternCache:
    """Modern pattern cache with advanced features for Theory 10."""
    
    def __init__(self, max_size: int = 512):
        self.max_size = max_size
        self._cache: OrderedDict[str, Pattern[str]] = OrderedDict()
        self._stats = PatternCacheStats()
        self._lock = threading.RLock()
        self._pattern_builders: Dict[str, Callable] = {}
        self._common_patterns: Dict[str, Pattern[str]] = {}
        
        # Pattern category tracking for intelligent eviction
        self._pattern_categories: Dict[str, str] = {}
        self._category_priorities = {
            'core': 100,      # Core system patterns - highest priority
            'common': 80,     # Commonly used patterns
            'language': 60,   # Language-specific patterns
            'specialized': 40, # Specialized use case patterns
            'temp': 20        # Temporary/experimental patterns
        }
    
    def cached_pattern(self, category: str = 'common', language_aware: bool = True) -> Callable[[F], F]:
        """
        Advanced pattern caching decorator with categorization and language awareness.
        
        Args:
            category: Pattern category for intelligent eviction ('core', 'common', 'language', 'specialized', 'temp')
            language_aware: Whether this pattern varies by language
            
        Returns:
            Decorator function for pattern builders
        """
        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Pattern[str]:
                return self._get_or_create_pattern(
                    func, args, kwargs, category, language_aware
                )
            
            # Register the pattern builder for cache management
            self._pattern_builders[func.__name__] = func
            return wrapper  # type: ignore
        
        return decorator
    
    def _get_or_create_pattern(
        self, 
        func: Callable,
        args: tuple,
        kwargs: dict,
        category: str,
        language_aware: bool
    ) -> Pattern[str]:
        """Get pattern from cache or create and cache it."""
        cache_key = self._generate_cache_key(func.__name__, args, kwargs, language_aware)
        
        with self._lock:
            # Check cache first
            if cache_key in self._cache:
                self._stats.hits += 1
                # Move to end (mark as recently used)
                self._cache.move_to_end(cache_key)
                return self._cache[cache_key]
            
            # Cache miss - need to compile pattern
            start_time = time.time()
        
        # Compile pattern outside of lock to minimize lock time
        try:
            pattern = func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Pattern compilation failed for {func.__name__}: {e}")
            raise
        
        compilation_time = time.time() - start_time
        
        with self._lock:
            # Double-check that another thread didn't add this pattern
            if cache_key not in self._cache:
                # Evict patterns if needed before adding new one
                self._intelligent_eviction(category)
                
                # Cache the new pattern
                self._cache[cache_key] = pattern
                self._pattern_categories[cache_key] = category
                self._stats.misses += 1
                self._stats.compilation_time += compilation_time
                self._stats.cache_size = len(self._cache)
                
                # Track language if language-aware
                if language_aware and args:
                    # Assume first argument or 'language' kwarg contains language
                    language = kwargs.get('language') or (args[0] if args else 'en')
                    self._stats.languages_cached.add(language)
                
                logger.debug(f"Cached pattern {func.__name__} ({category}) - cache size: {len(self._cache)}")
            else:
                # Another thread added it, use theirs
                self._stats.hits += 1
                self._cache.move_to_end(cache_key)
                pattern = self._cache[cache_key]
        
        return pattern
    
    def _generate_cache_key(
        self, 
        func_name: str, 
        args: tuple, 
        kwargs: dict,
        language_aware: bool
    ) -> str:
        """Generate cache key with improved collision handling."""
        key_parts = [func_name]
        
        # Add arguments
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            else:
                # For complex objects, use their string representation
                key_parts.append(str(type(arg).__name__))
        
        # Add sorted kwargs
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        
        # Create key string
        key_string = "|".join(key_parts)
        
        # For long keys, use hash to prevent memory issues
        if len(key_string) > 128:
            key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:20]
            return f"{func_name}_{key_hash}"
        
        return key_string
    
    def _intelligent_eviction(self, new_category: str) -> None:
        """Intelligent cache eviction based on pattern categories and priorities."""
        if len(self._cache) < self.max_size:
            return
        
        # Sort patterns by priority (lowest priority first for eviction)
        patterns_by_priority = []
        for cache_key in self._cache:
            category = self._pattern_categories.get(cache_key, 'temp')
            priority = self._category_priorities.get(category, 0)
            patterns_by_priority.append((priority, cache_key))
        
        patterns_by_priority.sort(key=lambda x: x[0])
        
        # Evict lowest priority patterns first
        new_priority = self._category_priorities.get(new_category, 0)
        eviction_count = 0
        
        for priority, cache_key in patterns_by_priority:
            if len(self._cache) >= self.max_size:
                # Only evict if new pattern has higher priority than existing
                if priority < new_priority:
                    del self._cache[cache_key]
                    if cache_key in self._pattern_categories:
                        del self._pattern_categories[cache_key]
                    eviction_count += 1
                    self._stats.evictions += 1
                else:
                    # If we can't evict higher priority patterns, just evict LRU
                    lru_key = next(iter(self._cache))
                    del self._cache[lru_key]
                    if lru_key in self._pattern_categories:
                        del self._pattern_categories[lru_key]
                    eviction_count += 1
                    self._stats.evictions += 1
                    break
            else:
                break
        
        if eviction_count > 0:
            logger.debug(f"Evicted {eviction_count} patterns for new {new_category} pattern")
    
    def warm_common_patterns(self, language: str = "en") -> None:
        """Pre-warm cache with commonly used patterns."""
        logger.info(f"Warming pattern cache for language: {language}")
        
        # Core regex patterns that are used frequently
        core_patterns = [
            ("url_basic", r"https?://[^\s]+", re.IGNORECASE),
            ("email_basic", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", re.IGNORECASE),
            ("number_word", r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten)\b", re.IGNORECASE),
            ("decimal_number", r"\b\d+\.?\d*\b", 0),
            ("whitespace_normalize", r"\s+", 0),
            ("punctuation_cleanup", r"[.!?]{2,}", 0),
        ]
        
        with self._lock:
            for name, pattern_str, flags in core_patterns:
                cache_key = f"warm_{name}_{language}"
                if cache_key not in self._cache:
                    try:
                        pattern = re.compile(pattern_str, flags)
                        self._cache[cache_key] = pattern
                        self._pattern_categories[cache_key] = 'core'
                        logger.debug(f"Warmed core pattern: {name}")
                    except re.error as e:
                        logger.warning(f"Failed to warm pattern {name}: {e}")
            
            self._stats.cache_size = len(self._cache)
            self._stats.languages_cached.add(language)
    
    def get_stats(self) -> PatternCacheStats:
        """Get current cache statistics."""
        with self._lock:
            stats_copy = PatternCacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                compilation_time=self._stats.compilation_time,
                cache_size=self._stats.cache_size,
                languages_cached=self._stats.languages_cached.copy()
            )
        return stats_copy
    
    def clear_cache(self, category: Optional[str] = None) -> int:
        """Clear cache patterns, optionally by category."""
        with self._lock:
            if category is None:
                # Clear all patterns
                cleared = len(self._cache)
                self._cache.clear()
                self._pattern_categories.clear()
                self._stats = PatternCacheStats()
                return cleared
            else:
                # Clear patterns in specific category
                keys_to_remove = [
                    key for key, cat in self._pattern_categories.items()
                    if cat == category
                ]
                for key in keys_to_remove:
                    del self._cache[key]
                    del self._pattern_categories[key]
                
                self._stats.cache_size = len(self._cache)
                return len(keys_to_remove)
    
    def optimize_patterns(self) -> Dict[str, int]:
        """Analyze and optimize cached patterns for better performance."""
        with self._lock:
            optimization_results = {
                'duplicate_patterns': 0,
                'oversized_patterns': 0,
                'unused_patterns': 0
            }
            
            # Find duplicate patterns (same compiled regex)
            pattern_map: Dict[str, List[str]] = {}
            for key, pattern in self._cache.items():
                pattern_str = pattern.pattern
                if pattern_str not in pattern_map:
                    pattern_map[pattern_str] = []
                pattern_map[pattern_str].append(key)
            
            # Log duplicate patterns
            for pattern_str, keys in pattern_map.items():
                if len(keys) > 1:
                    optimization_results['duplicate_patterns'] += len(keys) - 1
                    logger.debug(f"Found {len(keys)} duplicate patterns: {pattern_str}")
            
            # Check for oversized patterns (potential performance issues)
            for key, pattern in self._cache.items():
                if len(pattern.pattern) > 1000:
                    optimization_results['oversized_patterns'] += 1
                    logger.debug(f"Oversized pattern detected: {key} ({len(pattern.pattern)} chars)")
            
            return optimization_results


# Global instance
_global_cache = ModernPatternCache(max_size=512)

# Convenience functions for backward compatibility
def cached_pattern(category: str = 'common', language_aware: bool = True) -> Callable[[F], F]:
    """Global cached pattern decorator."""
    return _global_cache.cached_pattern(category, language_aware)

def get_cache_stats() -> PatternCacheStats:
    """Get global cache statistics."""
    return _global_cache.get_stats()

def clear_cache(category: Optional[str] = None) -> int:
    """Clear global cache."""
    return _global_cache.clear_cache(category)

def warm_common_patterns(language: str = "en") -> None:
    """Warm global cache with common patterns."""
    return _global_cache.warm_common_patterns(language)

def optimize_patterns() -> Dict[str, int]:
    """Optimize global cache patterns."""
    return _global_cache.optimize_patterns()

@contextmanager
def pattern_compilation_timing():
    """Context manager for timing pattern compilation."""
    start_time = time.time()
    try:
        yield
    finally:
        compilation_time = time.time() - start_time
        logger.debug(f"Pattern compilation took {compilation_time:.4f} seconds")