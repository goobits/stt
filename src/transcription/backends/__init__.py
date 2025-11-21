"""
Transcription backends package.
"""
import logging
from typing import Optional, Type
from .base import TranscriptionBackend
from .faster_whisper_backend import FasterWhisperBackend

logger = logging.getLogger(__name__)

# Conditionally import Parakeet backend
try:
    from .parakeet_backend import ParakeetBackend
    PARAKEET_AVAILABLE = True
except ImportError:
    PARAKEET_AVAILABLE = False


def get_backend_class(backend_name: str) -> Type[TranscriptionBackend]:
    """
    Factory function to get the backend class based on name.

    Args:
        backend_name: 'faster_whisper' or 'parakeet'

    Returns:
        The backend class.

    Raises:
        ValueError: If backend is unknown or unavailable.
    """
    if backend_name == "faster_whisper":
        return FasterWhisperBackend

    if backend_name == "parakeet":
        if not PARAKEET_AVAILABLE:
            raise ValueError("Parakeet backend requested but 'parakeet-mlx' is not installed or available.")
        return ParakeetBackend

    raise ValueError(f"Unknown backend: {backend_name}")
