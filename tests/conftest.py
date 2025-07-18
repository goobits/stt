import pytest
import os
import sys

# Add workspace directory to Python path for proper imports
workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

@pytest.fixture
def preloaded_formatter():
    """Provide preloaded text formatter for tests."""
    from src.text_formatting.formatter import format_transcription
    return format_transcription