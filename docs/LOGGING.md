# Structured Logging Implementation

## Overview

The STT project now includes a comprehensive structured logging system that provides:

- **Environment-driven configuration** (`LOG_LEVEL`, `LOG_OUTPUT`)
- **Structured JSON output** for production, readable format for development
- **Context management** with automatic request/session ID propagation
- **Cloud-native defaults** (stdout/stderr) with file fallback
- **Backward compatibility** with existing logging code

## Quick Start

### Basic Usage (New Code)

```python
from stt.core.logging import setup_structured_logging, set_context

# Get a structured logger
logger = setup_structured_logging(__name__)

# Basic logging
logger.info("Application starting")
logger.error("Something went wrong")

# With context
set_context(request_id="req-123", user_id="user-456")
logger.info("Processing user request")  # Context automatically included
```

### Backward Compatibility (Existing Code)

```python
from stt.core.config import setup_logging

# Existing code continues to work
logger = setup_logging(__name__)
logger.info("This still works and gets structured logging benefits")
```

## Environment Configuration

### Environment Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `LOG_LEVEL` | DEBUG, INFO, WARN, ERROR | INFO | Minimum log level |
| `LOG_OUTPUT` | console, file, both | console | Output destination |
| `STT_ENV` | production, development | auto-detect | Environment mode |

### Output Modes

**console (default):**
- INFO/DEBUG → stdout
- WARN/ERROR → stderr
- Perfect for containers and cloud deployments

**file:**
- All logs to rotating files in `logs/` directory
- 5MB max size, 3 backup files per module

**both:**
- Logs to both console and file

## Log Formats

### Development (Human-Readable)

```
2025-08-02 18:09:08 | INFO  | module.name | Message text | key1=value1 key2=value2
```

### Production (JSON)

```json
{
  "timestamp": "2025-08-02T18:09:08.123456Z",
  "level": "INFO",
  "module": "module.name",
  "message": "Message text",
  "context": {
    "request_id": "req-123",
    "user_id": "user-456"
  }
}
```

## Context Management

### Manual Context

```python
from stt.core.logging import set_context, clear_context

# Set context for current execution
set_context(request_id="req-123", session_id="sess-456")
logger.info("Processing request")  # Context included automatically

# Clear when done
clear_context()
```

### Context Manager

```python
from stt.core.logging import LogContext

with LogContext(operation="transcribe", model="whisper-large"):
    logger.info("Starting transcription")  # Context included
    
    with LogContext(chunk_id="chunk-001"):  # Nested context
        logger.debug("Processing chunk")    # Both contexts included
# Context automatically cleared
```

### Decorator

```python
from stt.core.logging import with_context

@with_context(component="audio_processor", version="1.2.3")
def process_audio():
    logger.info("Processing audio")  # Context included automatically
```

## Advanced Features

### Request Tracking

```python
# In web handlers
with LogContext(
    request_id=str(uuid.uuid4()),
    client_ip=request.remote_addr,
    user_id=get_current_user_id()
):
    logger.info("Handling request")
    # All nested calls inherit this context
```

### Error Handling

```python
try:
    risky_operation()
except Exception as e:
    logger.exception(
        "Operation failed", 
        extra={"event": "operation_error", "operation_id": "op-123"}
    )
```

### Custom Logger Configuration

```python
# Force JSON output regardless of environment
logger = setup_structured_logging(
    __name__,
    force_json=True,
    log_level="DEBUG",
    context={"service": "transcription"}
)

# File-only logging (for pipeline modules)
logger = setup_structured_logging(
    __name__,
    log_output="file",
    context={"component": "text_formatter"}
)
```

## Migration Guide

### Step 1: Update Imports (Optional)

**Old:**
```python
from stt.core.config import setup_logging
logger = setup_logging(__name__)
```

**New:**
```python
from stt.core.logging import setup_structured_logging
logger = setup_structured_logging(__name__)
```

### Step 2: Add Context (Recommended)

```python
# Add context to key operations
with LogContext(request_id=request_id, operation="transcribe"):
    result = transcribe_audio(audio_data)
    logger.info("Transcription completed")
```

### Step 3: Enhance Error Logging

**Old:**
```python
logger.error(f"Failed to process {file_id}: {e}")
```

**New:**
```python
logger.error(
    "Failed to process file", 
    extra={
        "event": "file_processing_error",
        "file_id": file_id,
        "error": str(e)
    }
)
```

## Best Practices

### Context Keys

Use consistent context keys across the application:

- `request_id` - Unique identifier for requests
- `session_id` - User session identifier
- `user_id` - User identifier
- `client_id` - Client connection identifier
- `operation` - Current operation (transcribe, format, etc.)
- `component` - System component (audio, text_formatting, etc.)
- `stage` - Processing stage (validation, processing, response)

### Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General operational events
- **WARN**: Unexpected but recoverable conditions
- **ERROR**: Error conditions that should be investigated

### Message Structure

**Good:**
```python
logger.info("Transcription completed", extra={
    "event": "transcription_complete",
    "duration_ms": 1234,
    "model": "whisper-large"
})
```

**Avoid:**
```python
logger.info(f"Transcription of {file} completed in {duration}ms using {model}")
```

## Testing and Development

### Disable Logging in Tests

```python
# In test configuration
LOG_LEVEL=CRITICAL python -m pytest
```

### Debug Logging

```python
# Enable debug logging
LOG_LEVEL=DEBUG LOG_OUTPUT=console python your_script.py
```

### Production Simulation

```python
# Test JSON output locally
STT_ENV=production python your_script.py
```

## File Organization

### Log Files

- **Module logs**: `logs/{module_name}.log`
- **Rotating files**: 5MB max, 3 backups
- **Permissions**: 644 (readable by monitoring tools)

### Example Files

- `logs/transcription.log` - WebSocket server logs
- `logs/text_formatting.log` - Text processing logs
- `logs/audio_capture.log` - Audio input logs

## Monitoring Integration

### Log Aggregation

The JSON format is designed for log aggregation tools:

- **ELK Stack**: Direct JSON ingestion
- **Fluentd**: Structured field extraction
- **CloudWatch**: JSON field indexing
- **Datadog**: Automatic field parsing

### Alerting Queries

```javascript
// Error rate by component
level:"ERROR" | stats count by context.component

// Request tracing
context.request_id:"req-123" | sort timestamp

// User activity
context.user_id:"user-456" AND level:"INFO"
```

## Performance Considerations

- **Context propagation**: Uses Python's `contextvars` for minimal overhead
- **JSON serialization**: Only in production mode
- **File rotation**: Automatic cleanup prevents disk space issues
- **Handler reuse**: Loggers are cached to prevent duplicate handlers

## Examples

See `examples/structured_logging_demo.py` for comprehensive usage examples.

## Troubleshooting

### Common Issues

**No log output:**
- Check `LOG_LEVEL` environment variable
- Verify `LOG_OUTPUT` setting
- Ensure `logs/` directory permissions

**Duplicate messages:**
- Logger handlers may be duplicated
- Use the provided functions instead of direct `logging.getLogger()`

**Context not appearing:**
- Ensure context is set before logging
- Check that you're using `setup_structured_logging()`

**JSON format issues:**
- Verify `STT_ENV=production` or `force_json=True`
- Check for custom formatters overriding the structured formatter