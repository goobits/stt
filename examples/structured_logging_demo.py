#!/usr/bin/env python3
"""
Demonstration of the new structured logging system for STT.

This example shows how to use the new logging features including:
- Environment configuration
- Structured JSON output
- Context management
- Request/session tracking
"""
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stt.core.logging import (
    setup_structured_logging,
    set_context,
    clear_context,
    LogContext,
    with_context,
)


def demo_basic_logging():
    """Demonstrate basic structured logging."""
    print("=== Basic Structured Logging ===")
    
    # Get logger with default settings
    logger = setup_structured_logging(__name__)
    
    logger.info("Application starting")
    logger.debug("Debug information")
    logger.warning("Warning message")
    logger.error("Error occurred")
    print()


def demo_environment_configuration():
    """Demonstrate environment-based configuration."""
    print("=== Environment Configuration ===")
    
    # Set environment variables
    os.environ['LOG_LEVEL'] = 'DEBUG'
    os.environ['LOG_OUTPUT'] = 'console'
    
    logger = setup_structured_logging("demo.env")
    logger.debug("This debug message will show because LOG_LEVEL=DEBUG")
    logger.info("Environment configured logging")
    print()


def demo_json_output():
    """Demonstrate JSON structured output."""
    print("=== JSON Output (Production Mode) ===")
    
    # Force JSON output
    logger = setup_structured_logging("demo.json", force_json=True)
    
    logger.info("JSON formatted message")
    logger.warning("JSON warning with details")
    logger.error("JSON error message")
    print()


def demo_context_management():
    """Demonstrate context management."""
    print("=== Context Management ===")
    
    logger = setup_structured_logging("demo.context")
    
    # Manual context setting
    set_context(request_id="req-123", user_id="user-456")
    logger.info("Processing user request")
    logger.debug("User context included")
    
    # Clear context
    clear_context()
    logger.info("Context cleared")
    print()


def demo_context_manager():
    """Demonstrate context manager usage."""
    print("=== Context Manager ===")
    
    logger = setup_structured_logging("demo.manager")
    
    with LogContext(operation="transcribe", model="whisper-large"):
        logger.info("Starting transcription")
        logger.debug("Model loaded successfully")
        
        # Nested context
        with LogContext(session_id="sess-789", chunk_id="chunk-001"):
            logger.info("Processing audio chunk")
            logger.debug("Chunk size: 1024 bytes")
    
    logger.info("Context automatically cleared")
    print()


@with_context(operation="audio_processing", component="encoder")
def demo_decorator():
    """Demonstrate decorator-based context."""
    logger = setup_structured_logging("demo.decorator")
    logger.info("Processing audio with automatic context")
    logger.debug("Encoding parameters set")


def demo_web_request_simulation():
    """Simulate web request logging with full context."""
    print("=== Web Request Simulation ===")
    
    logger = setup_structured_logging("demo.web")
    
    # Simulate incoming request
    with LogContext(
        request_id="req-uuid-12345",
        session_id="sess-abcdef",
        user_id="user-789",
        client_ip="192.168.1.100"
    ):
        logger.info("Incoming transcription request")
        
        # Simulate processing stages
        with LogContext(stage="validation"):
            logger.debug("Validating audio format")
            logger.info("Audio validation passed")
        
        with LogContext(stage="transcription", model="whisper-large-v3"):
            logger.info("Starting transcription")
            logger.debug("Model loaded successfully")
            logger.info("Transcription completed")
        
        with LogContext(stage="response"):
            logger.info("Sending response to client")
            logger.debug("Response size: 2.1KB")
    
    print()


def demo_error_handling():
    """Demonstrate error logging with context."""
    print("=== Error Handling with Context ===")
    
    logger = setup_structured_logging("demo.errors")
    
    set_context(operation="file_processing", file_id="audio-123.wav")
    
    try:
        # Simulate an error
        raise ValueError("Invalid audio format")
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
    
    clear_context()
    print()


def demo_different_outputs():
    """Demonstrate different output modes."""
    print("=== Different Output Modes ===")
    
    # Console only
    console_logger = setup_structured_logging("demo.console", log_output="console")
    console_logger.info("Console-only logging")
    
    # File only
    file_logger = setup_structured_logging("demo.file", log_output="file")
    file_logger.info("File-only logging (check logs/demo.log)")
    
    # Both console and file
    both_logger = setup_structured_logging("demo.both", log_output="both")
    both_logger.info("Both console and file logging")
    
    print()


if __name__ == "__main__":
    print("STT Structured Logging Demonstration")
    print("=" * 50)
    
    # Run all demos
    demo_basic_logging()
    demo_environment_configuration()
    demo_json_output()
    demo_context_management()
    demo_context_manager()
    
    print("=== Decorator Demo ===")
    demo_decorator()
    print()
    
    demo_web_request_simulation()
    demo_error_handling()
    demo_different_outputs()
    
    print("Demo completed! Check the logs/ directory for file outputs.")
    print("\nTo test production JSON output, run:")
    print("STT_ENV=production python examples/structured_logging_demo.py")
    print("\nTo test different log levels:")
    print("LOG_LEVEL=DEBUG python examples/structured_logging_demo.py")