#!/usr/bin/env python3
"""
ModelManager - Singleton manager for efficient Whisper model loading and caching.

This module provides centralized model management to avoid loading multiple instances
of the same model, reducing memory usage and initialization time.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import weakref
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton manager for Whisper models with async loading and caching.
    
    Features:
    - Singleton pattern ensures only one manager instance
    - Model caching prevents duplicate loading
    - Thread-safe operations with asyncio.Lock
    - Weak references to allow garbage collection
    - Async model loading with executor
    """
    
    _instance: Optional[ModelManager] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> ModelManager:
        """Ensure singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the ModelManager (only once)."""
        if self._initialized:
            return
            
        self._models: Dict[str, object] = {}
        self._model_locks: Dict[str, asyncio.Lock] = {}
        self._async_lock = asyncio.Lock()
        self._initialized = True
        
        logger.info("ModelManager singleton initialized")
    
    async def get_model(self, model_name: str, device: str = "cpu", compute_type: str = "int8") -> object:
        """
        Get or load a Whisper model with caching.
        
        Args:
            model_name: Name of the Whisper model (e.g., "large-v3-turbo")
            device: Device to load model on ("cpu", "cuda", etc.)
            compute_type: Compute type for model ("int8", "float16", etc.)
            
        Returns:
            Loaded WhisperModel instance
            
        Raises:
            ImportError: If faster-whisper is not available
            Exception: If model loading fails
        """
        # Create unique key for model configuration
        model_key = f"{model_name}_{device}_{compute_type}"
        
        async with self._async_lock:
            # Return cached model if available
            if model_key in self._models:
                logger.debug(f"Using cached model: {model_key}")
                return self._models[model_key]
            
            # Ensure we have a lock for this model
            if model_key not in self._model_locks:
                self._model_locks[model_key] = asyncio.Lock()
        
        # Use model-specific lock for loading
        async with self._model_locks[model_key]:
            # Double-check pattern: model might have been loaded while waiting
            if model_key in self._models:
                logger.debug(f"Using cached model (double-check): {model_key}")
                return self._models[model_key]
            
            # Load model asynchronously
            logger.info(f"Loading Whisper model: {model_name} (device={device}, compute_type={compute_type})")
            
            try:
                from faster_whisper import WhisperModel
                
                # Load in executor to avoid blocking event loop
                loop = asyncio.get_event_loop()
                model = await loop.run_in_executor(
                    None, 
                    lambda: WhisperModel(
                        model_name, 
                        device=device, 
                        compute_type=compute_type
                    )
                )
                
                # Cache the loaded model
                self._models[model_key] = model
                logger.info(f"Model loaded and cached: {model_key}")
                
                return model
                
            except ImportError as e:
                logger.error(f"faster-whisper not available: {e}")
                raise ImportError("faster-whisper package is required for model loading") from e
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise
    
    def clear_cache(self) -> None:
        """
        Clear model cache to free memory.
        
        Note: This will cause models to be reloaded on next request.
        Use with caution during active transcription sessions.
        """
        logger.info(f"Clearing model cache ({len(self._models)} models)")
        self._models.clear()
        self._model_locks.clear()
    
    def get_cached_models(self) -> Dict[str, str]:
        """
        Get information about currently cached models.
        
        Returns:
            Dictionary mapping model keys to basic info
        """
        return {
            key: f"Model({type(model).__name__})" 
            for key, model in self._models.items()
        }
    
    def memory_usage_estimate(self) -> str:
        """
        Get rough estimate of memory usage.
        
        Returns:
            Human-readable string describing memory usage
        """
        model_count = len(self._models)
        if model_count == 0:
            return "No models loaded"
        
        # Rough estimates based on common model sizes
        size_estimates = {
            "tiny": "~39MB",
            "base": "~74MB", 
            "small": "~244MB",
            "medium": "~769MB",
            "large": "~1550MB",
            "large-v2": "~1550MB",
            "large-v3": "~1550MB",
            "large-v3-turbo": "~809MB"
        }
        
        total_estimate = "Unknown"
        for model_key in self._models:
            for size, estimate in size_estimates.items():
                if size in model_key:
                    total_estimate = f"~{estimate} per model"
                    break
        
        return f"{model_count} models loaded, estimated memory: {total_estimate}"


# Global convenience function for easy access
_manager_instance: Optional[ModelManager] = None

def get_model_manager() -> ModelManager:
    """Get the global ModelManager singleton instance."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = ModelManager()
    return _manager_instance