#!/usr/bin/env python3
"""
Token Bucket Rate Limiter - Advanced rate limiting with burst protection.

This module provides a robust token bucket algorithm for rate limiting that:
- Allows controlled bursts while maintaining long-term rate limits
- Provides configurable rates and burst capacities
- Includes automatic cleanup of inactive clients
- Thread-safe operations for concurrent access
"""

from __future__ import annotations

import threading
import time
from typing import Dict, NamedTuple


class TokenBucket(NamedTuple):
    """Represents a token bucket for a specific client."""
    tokens: float
    last_update: float


class TokenBucketRateLimiter:
    """
    Advanced token bucket rate limiter with burst protection.
    
    The token bucket algorithm allows for controlled bursts while maintaining
    an average rate limit over time. This is more flexible than simple
    time-window rate limiting.
    
    Features:
    - Configurable rate (tokens per second) and burst capacity
    - Automatic token replenishment over time
    - Thread-safe operations with fine-grained locking
    - Automatic cleanup of inactive client buckets
    - Memory-efficient storage
    """
    
    def __init__(self, rate: float = 10.0, capacity: float = 20.0, cleanup_interval: float = 3600.0):
        """
        Initialize the token bucket rate limiter.
        
        Args:
            rate: Number of tokens replenished per second (requests/second)
            capacity: Maximum number of tokens in bucket (burst capacity)
            cleanup_interval: How often to clean up inactive buckets (seconds)
        """
        self.rate = rate
        self.capacity = capacity
        self.cleanup_interval = cleanup_interval
        
        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = threading.RLock()  # Reentrant lock for nested operations
        self._last_cleanup = time.time()
        
    def allow_request(self, client_id: str, tokens_required: float = 1.0) -> bool:
        """
        Check if a request is allowed under the rate limit.
        
        Args:
            client_id: Unique identifier for the client
            tokens_required: Number of tokens required for this request
            
        Returns:
            True if request is allowed, False if rate limited
        """
        now = time.time()
        
        with self._lock:
            # Perform cleanup if needed
            self._cleanup_if_needed(now)
            
            # Get or create bucket for client
            if client_id not in self._buckets:
                self._buckets[client_id] = TokenBucket(
                    tokens=self.capacity,
                    last_update=now
                )
            
            bucket = self._buckets[client_id]
            
            # Calculate tokens to add based on elapsed time
            elapsed = now - bucket.last_update
            tokens_to_add = elapsed * self.rate
            new_tokens = min(self.capacity, bucket.tokens + tokens_to_add)
            
            # Check if we have enough tokens
            if new_tokens >= tokens_required:
                # Allow request and update bucket
                self._buckets[client_id] = TokenBucket(
                    tokens=new_tokens - tokens_required,
                    last_update=now
                )
                return True
            else:
                # Deny request but update last_update time
                self._buckets[client_id] = TokenBucket(
                    tokens=new_tokens,
                    last_update=now
                )
                return False
    
    def get_client_status(self, client_id: str) -> Dict[str, float]:
        """
        Get rate limiting status for a specific client.
        
        Args:
            client_id: Unique identifier for the client
            
        Returns:
            Dictionary with current tokens, capacity, and estimated wait time
        """
        now = time.time()
        
        with self._lock:
            if client_id not in self._buckets:
                return {
                    "tokens": self.capacity,
                    "capacity": self.capacity,
                    "wait_time_seconds": 0.0
                }
            
            bucket = self._buckets[client_id]
            elapsed = now - bucket.last_update
            current_tokens = min(self.capacity, bucket.tokens + elapsed * self.rate)
            
            # Calculate wait time until next token is available
            wait_time = 0.0
            if current_tokens < 1.0:
                tokens_needed = 1.0 - current_tokens
                wait_time = tokens_needed / self.rate
            
            return {
                "tokens": current_tokens,
                "capacity": self.capacity,
                "wait_time_seconds": wait_time
            }
    
    def reset_client(self, client_id: str) -> None:
        """
        Reset rate limiting for a specific client (fill bucket).
        
        Args:
            client_id: Unique identifier for the client
        """
        with self._lock:
            self._buckets[client_id] = TokenBucket(
                tokens=self.capacity,
                last_update=time.time()
            )
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get overall rate limiter statistics.
        
        Returns:
            Dictionary with rate limiter configuration and current state
        """
        with self._lock:
            total_clients = len(self._buckets)
            active_clients = sum(
                1 for bucket in self._buckets.values()
                if time.time() - bucket.last_update < 300  # Active in last 5 minutes
            )
            
            return {
                "rate_per_second": self.rate,
                "burst_capacity": self.capacity,
                "total_clients": total_clients,
                "active_clients": active_clients,
                "cleanup_interval": self.cleanup_interval
            }
    
    def cleanup_inactive_clients(self, max_age: float = None) -> int:
        """
        Clean up buckets for clients that haven't made requests recently.
        
        Args:
            max_age: Maximum age in seconds (defaults to cleanup_interval)
            
        Returns:
            Number of client buckets removed
        """
        if max_age is None:
            max_age = self.cleanup_interval
            
        now = time.time()
        cutoff_time = now - max_age
        
        with self._lock:
            clients_to_remove = [
                client_id for client_id, bucket in self._buckets.items()
                if bucket.last_update < cutoff_time
            ]
            
            for client_id in clients_to_remove:
                del self._buckets[client_id]
            
            self._last_cleanup = now
            return len(clients_to_remove)
    
    def _cleanup_if_needed(self, now: float) -> None:
        """Internal method to perform cleanup if interval has passed."""
        if now - self._last_cleanup > self.cleanup_interval:
            removed = self.cleanup_inactive_clients()
            if removed > 0:
                # Note: We can't use logger here as it might cause circular imports
                # The server will log rate limiter activities
                pass