#!/usr/bin/env python3
"""
Pattern Cache Performance Monitor - Theory 10 Support

This module provides monitoring and analysis tools for the pattern cache system
to help identify optimization opportunities and ensure optimal performance.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional
from contextlib import contextmanager
from dataclasses import dataclass

from .modern_pattern_cache import get_cache_stats, optimize_patterns, clear_cache
from ..core.config import setup_logging

logger = setup_logging(__name__)

@dataclass
class PatternPerformanceReport:
    """Performance report for pattern cache usage."""
    hit_ratio: float
    miss_ratio: float
    total_requests: int
    cache_size: int
    evictions: int
    avg_compilation_time: float
    optimization_opportunities: Dict[str, int]
    languages_supported: List[str]
    recommendations: List[str]

class PatternCacheMonitor:
    """Monitor and analyzer for pattern cache performance."""
    
    def __init__(self):
        self.monitoring_enabled = False
        self.performance_history: List[Dict] = []
        self.baseline_stats = None
    
    def enable_monitoring(self) -> None:
        """Enable performance monitoring."""
        self.monitoring_enabled = True
        self.baseline_stats = get_cache_stats()
        logger.info("Pattern cache monitoring enabled")
    
    def disable_monitoring(self) -> None:
        """Disable performance monitoring."""
        self.monitoring_enabled = False
        logger.info("Pattern cache monitoring disabled")
    
    @contextmanager
    def monitor_operation(self, operation_name: str):
        """Context manager to monitor cache operations."""
        if not self.monitoring_enabled:
            yield
            return
        
        start_stats = get_cache_stats()
        start_time = time.time()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_stats = get_cache_stats()
            
            # Record performance data
            operation_data = {
                'operation': operation_name,
                'duration': end_time - start_time,
                'hits_delta': end_stats.hits - start_stats.hits,
                'misses_delta': end_stats.misses - start_stats.misses,
                'cache_size_delta': end_stats.cache_size - start_stats.cache_size,
                'timestamp': time.time()
            }
            
            self.performance_history.append(operation_data)
            
            # Log significant events
            if operation_data['misses_delta'] > 5:
                logger.debug(f"High cache miss rate in {operation_name}: {operation_data['misses_delta']} misses")
    
    def generate_performance_report(self) -> PatternPerformanceReport:
        """Generate comprehensive performance report."""
        stats = get_cache_stats()
        optimization_data = optimize_patterns()
        
        # Generate recommendations based on stats
        recommendations = []
        
        if stats.hit_ratio < 0.8:
            recommendations.append("Consider increasing cache size - hit ratio is below 80%")
        
        if optimization_data['duplicate_patterns'] > 0:
            recommendations.append(f"Found {optimization_data['duplicate_patterns']} duplicate patterns - consolidate for better memory usage")
        
        if optimization_data['oversized_patterns'] > 0:
            recommendations.append(f"Found {optimization_data['oversized_patterns']} oversized patterns - consider optimization")
        
        if stats.evictions > stats.cache_size:
            recommendations.append("High eviction rate detected - consider increasing cache size")
        
        if stats.avg_compilation_time > 0.001:  # 1ms
            recommendations.append("High pattern compilation times detected - consider pattern optimization")
        
        return PatternPerformanceReport(
            hit_ratio=stats.hit_ratio,
            miss_ratio=stats.miss_ratio,
            total_requests=stats.hits + stats.misses,
            cache_size=stats.cache_size,
            evictions=stats.evictions,
            avg_compilation_time=stats.avg_compilation_time,
            optimization_opportunities=optimization_data,
            languages_supported=list(stats.languages_cached),
            recommendations=recommendations
        )
    
    def log_cache_performance(self) -> None:
        """Log current cache performance statistics."""
        report = self.generate_performance_report()
        
        logger.info("Pattern Cache Performance Report:")
        logger.info(f"  Hit Ratio: {report.hit_ratio:.2%}")
        logger.info(f"  Cache Size: {report.cache_size} patterns")
        logger.info(f"  Languages: {', '.join(report.languages_supported)}")
        logger.info(f"  Evictions: {report.evictions}")
        
        if report.avg_compilation_time > 0:
            logger.info(f"  Avg Compilation Time: {report.avg_compilation_time:.4f}s")
        
        if report.optimization_opportunities['duplicate_patterns'] > 0:
            logger.warning(f"  Duplicate Patterns: {report.optimization_opportunities['duplicate_patterns']}")
        
        for recommendation in report.recommendations:
            logger.info(f"  Recommendation: {recommendation}")
    
    def reset_monitoring(self) -> None:
        """Reset monitoring history and baseline."""
        self.performance_history.clear()
        self.baseline_stats = get_cache_stats()
        logger.debug("Pattern cache monitoring reset")

# Global monitor instance
_global_monitor = PatternCacheMonitor()

def enable_cache_monitoring() -> None:
    """Enable global cache monitoring."""
    _global_monitor.enable_monitoring()

def disable_cache_monitoring() -> None:
    """Disable global cache monitoring."""
    _global_monitor.disable_monitoring()

def get_performance_report() -> PatternPerformanceReport:
    """Get current performance report."""
    return _global_monitor.generate_performance_report()

def log_cache_performance() -> None:
    """Log current cache performance."""
    _global_monitor.log_cache_performance()

@contextmanager
def monitor_cache_operation(operation_name: str):
    """Monitor a cache operation."""
    with _global_monitor.monitor_operation(operation_name):
        yield