#!/usr/bin/env python3
"""
Standardized logging implementation for STT with structured output and context management.

Features:
- Environment-driven configuration (LOG_LEVEL, LOG_OUTPUT)
- Structured JSON logging for production, readable format for development
- Context management with request/session IDs
- Cloud-native defaults (stdout/stderr) with file fallback
- Automatic environment detection
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import threading
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Context variables for automatic context propagation
_request_context: ContextVar[Dict[str, Any]] = ContextVar('request_context', default={})


class StructuredFormatter(logging.Formatter):
    """
    Formatter that outputs JSON for production or readable format for development.
    """
    
    def __init__(self, use_json: bool = False):
        self.use_json = use_json
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        # Get context from contextvars
        context = _request_context.get({})
        
        # Merge context from record if available
        if hasattr(record, 'context') and record.context:
            context = {**context, **record.context}
        
        if self.use_json:
            return self._format_json(record, context)
        else:
            return self._format_readable(record, context)
    
    def _format_json(self, record: logging.LogRecord, context: Dict[str, Any]) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': self.formatTime(record, datefmt='%Y-%m-%dT%H:%M:%S.%fZ'),
            'level': record.levelname,
            'module': record.name,
            'message': record.getMessage(),
        }
        
        # Add context if present
        if context:
            log_entry['context'] = context
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add any extra fields
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created', 'msecs',
                          'relativeCreated', 'thread', 'threadName', 'processName',
                          'process', 'message', 'exc_info', 'exc_text', 'stack_info',
                          'context'):
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)
    
    def _format_readable(self, record: logging.LogRecord, context: Dict[str, Any]) -> str:
        """Format log record in human-readable format."""
        timestamp = self.formatTime(record, datefmt='%Y-%m-%d %H:%M:%S')
        
        # Build context string
        context_str = ""
        if context:
            context_parts = [f"{k}={v}" for k, v in context.items()]
            context_str = f" | {' '.join(context_parts)}"
        
        base_msg = f"{timestamp} | {record.levelname:5} | {record.name} | {record.getMessage()}{context_str}"
        
        # Add exception info if present
        if record.exc_info:
            base_msg += f"\n{self.formatException(record.exc_info)}"
        
        return base_msg


class ContextLogger(logging.LoggerAdapter):
    """
    Logger adapter that automatically includes context in log messages.
    """
    
    def __init__(self, logger: logging.Logger, extra: Optional[Dict[str, Any]] = None):
        super().__init__(logger, extra or {})
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Process log call to include context."""
        # Get current context
        context = _request_context.get({})
        
        # Merge with adapter's extra context
        if self.extra:
            context = {**context, **self.extra}
        
        # Merge with call-specific context
        if 'extra' in kwargs and kwargs['extra']:
            call_context = kwargs['extra'].pop('context', {})
            context = {**context, **call_context}
        
        # Add context to record
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        kwargs['extra']['context'] = context
        
        return msg, kwargs


def get_log_level() -> str:
    """Get log level from environment or default to INFO."""
    return os.environ.get('LOG_LEVEL', 'INFO').upper()


def get_log_output() -> str:
    """Get log output mode from environment or default to console."""
    return os.environ.get('LOG_OUTPUT', 'console').lower()


def is_production_env() -> bool:
    """Detect if running in production environment."""
    env_indicators = [
        os.environ.get('NODE_ENV') == 'production',
        os.environ.get('ENVIRONMENT') == 'production',
        os.environ.get('ENV') == 'production',
        os.environ.get('STT_ENV') == 'production',
        # Docker/Kubernetes indicators
        os.path.exists('/.dockerenv'),
        os.environ.get('KUBERNETES_SERVICE_HOST') is not None,
    ]
    return any(env_indicators)


def setup_structured_logging(
    name: str,
    log_level: Optional[str] = None,
    log_output: Optional[str] = None,
    force_json: Optional[bool] = None,
    context: Optional[Dict[str, Any]] = None,
) -> ContextLogger:
    """
    Setup standardized structured logging.
    
    Args:
        name: Logger name (typically __name__)
        log_level: Log level override (DEBUG, INFO, WARN, ERROR)
        log_output: Output mode override (console, file, both)
        force_json: Force JSON output regardless of environment detection
        context: Default context to include in all log messages
    
    Returns:
        ContextLogger instance with structured logging configured
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return ContextLogger(logger, context)
    
    # Get configuration
    level = log_level or get_log_level()
    output = log_output or get_log_output()
    use_json = force_json if force_json is not None else is_production_env()
    
    logger.setLevel(getattr(logging, level))
    
    # Create formatter
    formatter = StructuredFormatter(use_json=use_json)
    
    # Setup handlers based on output mode
    if output in ('console', 'both'):
        _setup_console_handlers(logger, formatter)
    
    if output in ('file', 'both'):
        _setup_file_handler(logger, formatter, name)
    
    # Prevent propagation to avoid duplicate messages
    logger.propagate = False
    
    return ContextLogger(logger, context)


def _setup_console_handlers(logger: logging.Logger, formatter: StructuredFormatter) -> None:
    """Setup console handlers with proper stream routing."""
    # INFO and DEBUG to stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(lambda record: record.levelno < logging.WARNING)
    logger.addHandler(stdout_handler)
    
    # WARN and ERROR to stderr
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    stderr_handler.setLevel(logging.WARNING)
    logger.addHandler(stderr_handler)


def _setup_file_handler(logger: logging.Logger, formatter: StructuredFormatter, name: str) -> None:
    """Setup rotating file handler."""
    try:
        # Import config to get project directory
        from .config import get_config
        config = get_config()
        logs_dir = Path(config.project_dir) / "logs"
    except ImportError:
        # Fallback if config not available
        logs_dir = Path.cwd() / "logs"
    
    logs_dir.mkdir(exist_ok=True)
    
    # Generate log filename from module name
    module_basename = name.split('.')[-1] if '.' in name else name
    log_file = logs_dir / f"{module_basename}.log"
    
    # Use rotating file handler (5MB max, 3 backups)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def set_context(**kwargs: Any) -> None:
    """
    Set logging context for the current execution context.
    
    Args:
        **kwargs: Context key-value pairs (e.g., request_id="123", user_id="456")
    """
    current_context = _request_context.get({})
    updated_context = {**current_context, **kwargs}
    _request_context.set(updated_context)


def clear_context() -> None:
    """Clear all logging context for the current execution context."""
    _request_context.set({})


def get_context() -> Dict[str, Any]:
    """Get current logging context."""
    return _request_context.get({}).copy()


def with_context(**kwargs: Any):
    """
    Decorator to add context to all log messages within a function.
    
    Usage:
        @with_context(request_id="123", operation="transcribe")
        def process_audio():
            logger.info("Processing started")  # Will include context
    """
    def decorator(func):
        def wrapper(*args, **func_kwargs):
            # Save current context
            old_context = get_context()
            
            # Set new context
            set_context(**{**old_context, **kwargs})
            
            try:
                return func(*args, **func_kwargs)
            finally:
                # Restore old context
                _request_context.set(old_context)
        
        return wrapper
    return decorator


class LogContext:
    """
    Context manager for temporary logging context.
    
    Usage:
        with LogContext(request_id="123", user_id="456"):
            logger.info("Processing request")  # Will include context
    """
    
    def __init__(self, **kwargs: Any):
        self.new_context = kwargs
        self.old_context: Dict[str, Any] = {}
    
    def __enter__(self):
        self.old_context = get_context()
        set_context(**{**self.old_context, **self.new_context})
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        _request_context.set(self.old_context)


# Backward compatibility with existing setup_logging function
def get_logger(
    name: str,
    log_level: Optional[str] = None,
    include_console: Optional[bool] = None,
    include_file: Optional[bool] = None,
    context: Optional[Dict[str, Any]] = None,
) -> ContextLogger:
    """
    Get a structured logger with backward compatibility.
    
    This function provides compatibility with the existing get_logger interface
    while adding structured logging capabilities.
    """
    # Convert old parameters to new format
    if include_console is False and include_file is True:
        output = 'file'
    elif include_console is True and include_file is False:
        output = 'console'
    elif include_console is True and include_file is True:
        output = 'both'
    else:
        output = None  # Use default
    
    return setup_structured_logging(
        name=name,
        log_level=log_level,
        log_output=output,
        context=context
    )


# For modules that want simple usage
def get_simple_logger(name: str) -> ContextLogger:
    """Get a logger with default structured logging settings."""
    return setup_structured_logging(name)


# Example usage and testing
if __name__ == "__main__":
    # Test different configurations
    logger = setup_structured_logging(__name__, force_json=False)
    
    # Basic logging
    logger.info("Application starting")
    logger.debug("Debug information")
    logger.warning("Warning message")
    logger.error("Error occurred")
    
    # Context-aware logging
    set_context(request_id="req-123", user_id="user-456")
    logger.info("Processing user request")
    
    # Context manager
    with LogContext(operation="transcribe", model="whisper-large"):
        logger.info("Starting transcription")
        logger.debug("Model loaded successfully")
    
    # Clear context
    clear_context()
    logger.info("Context cleared")
    
    # Test JSON output
    json_logger = setup_structured_logging("test.json", force_json=True)
    set_context(session_id="sess-789")
    json_logger.info("JSON formatted message")