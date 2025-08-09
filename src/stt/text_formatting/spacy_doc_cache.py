"""
Centralized SpaCy document processing with smart caching.

This module provides a centralized way to manage SpaCy document creation
and caching to eliminate redundant nlp(text) calls throughout the text
formatting pipeline.
"""

import logging
from typing import Optional, Dict, Any
from functools import lru_cache

logger = logging.getLogger(__name__)


class SpacyDocumentProcessor:
    """Centralized SpaCy document processing with smart caching and thread safety"""
    
    def __init__(self, nlp_model: Optional[Any] = None, max_cache_size: int = 10):
        """
        Initialize the document processor.
        
        Args:
            nlp_model: SpaCy NLP model instance
            max_cache_size: Maximum number of documents to cache
        """
        self.nlp = nlp_model
        self.max_cache_size = max_cache_size
        self._doc_cache: Dict[int, Any] = {}
        self._cache_order = []  # For LRU eviction
        
    def get_or_create_doc(self, text: str, force_create: bool = False) -> Optional[Any]:
        """
        Get cached doc or create new one with LRU eviction.
        
        Args:
            text: Text to process
            force_create: If True, always create a new doc (for modified text)
            
        Returns:
            SpaCy doc object or None if processing fails
        """
        if not text or not self.nlp:
            return None
            
        # For empty or very short text, don't cache
        if len(text.strip()) < 3:
            try:
                return self.nlp(text)
            except (AttributeError, ValueError, IndexError) as e:
                logger.warning(f"SpaCy processing failed for short text: {e}")
                return None
        
        text_hash = hash(text) if text else 0
        
        # If forcing new creation (e.g., text was modified), skip cache
        if force_create:
            try:
                doc = self.nlp(text)
                # Update cache with new doc
                self._update_cache(text_hash, doc)
                return doc
            except (AttributeError, ValueError, IndexError) as e:
                logger.warning(f"SpaCy processing failed: {e}")
                return None
        
        # Check cache first
        if text_hash in self._doc_cache:
            self._update_access_order(text_hash)
            return self._doc_cache[text_hash]
        
        # Create new document
        try:
            doc = self.nlp(text)
        except (AttributeError, ValueError, IndexError) as e:
            logger.warning(f"SpaCy processing failed: {e}")
            return None
            
        # Cache with LRU eviction
        self._update_cache(text_hash, doc)
        return doc
    
    def _update_cache(self, text_hash: int, doc: Any) -> None:
        """Update cache with LRU eviction."""
        # Remove from cache if already exists (for re-insertion at end)
        if text_hash in self._doc_cache:
            self._cache_order.remove(text_hash)
        
        # Evict oldest if at max capacity
        while len(self._doc_cache) >= self.max_cache_size and self._cache_order:
            oldest_hash = self._cache_order.pop(0)
            self._doc_cache.pop(oldest_hash, None)
        
        # Add new entry
        self._doc_cache[text_hash] = doc
        self._cache_order.append(text_hash)
    
    def _update_access_order(self, text_hash: int) -> None:
        """Update access order for LRU."""
        if text_hash in self._cache_order:
            self._cache_order.remove(text_hash)
            self._cache_order.append(text_hash)
    
    def clear_cache(self) -> None:
        """Clear document cache to free memory."""
        self._doc_cache.clear()
        self._cache_order.clear()
        
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics for monitoring."""
        return {
            'cache_size': len(self._doc_cache),
            'max_cache_size': self.max_cache_size,
            'cache_utilization': len(self._doc_cache) / self.max_cache_size if self.max_cache_size > 0 else 0
        }


# Global instance for use throughout the text formatting system
_global_doc_processor: Optional[SpacyDocumentProcessor] = None


def initialize_global_doc_processor(nlp_model: Optional[Any] = None, max_cache_size: int = 10) -> None:
    """Initialize the global document processor."""
    global _global_doc_processor
    _global_doc_processor = SpacyDocumentProcessor(nlp_model, max_cache_size)


def get_global_doc_processor() -> Optional[SpacyDocumentProcessor]:
    """Get the global document processor instance."""
    return _global_doc_processor


@lru_cache(maxsize=128)
def get_cached_spacy_doc(text: str, nlp_model_id: int) -> Optional[Any]:
    """
    LRU cached function for SpaCy document creation.
    
    This provides an additional caching layer using functools.lru_cache
    for frequently accessed short texts.
    
    Args:
        text: Text to process
        nlp_model_id: ID of the nlp model (for cache key uniqueness)
        
    Returns:
        SpaCy doc object or None
    """
    processor = get_global_doc_processor()
    if processor and processor.nlp:
        return processor.get_or_create_doc(text)
    return None