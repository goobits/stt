import logging
import os
import time
from typing import Tuple, Optional, Dict

from .base import TranscriptionBackend
from ...core.config import get_config

logger = logging.getLogger(__name__)
config = get_config()

# Enforce dependencies at module level so import fails if missing
try:
    import mlx.core
    import parakeet_mlx
except ImportError:
    raise ImportError("parakeet-mlx or mlx is not installed")

class ParakeetBackend(TranscriptionBackend):
    """Backend implementation using parakeet-mlx for Apple Silicon."""

    def __init__(self):
        self.model_name = config.get("parakeet.model", "mlx-community/parakeet-tdt-0.6b-v3")
        self.model = None
        self.processor = None

    async def load(self):
        """Load Parakeet model."""
        try:
            from parakeet_mlx import from_pretrained

            logger.info(f"Loading Parakeet model: {self.model_name}")
            # parakeet loading might be blocking, so we should consider running it in executor if it takes time
            # However, MLX lazy loading might make it fast.
            self.model = from_pretrained(self.model_name)
            logger.info(f"Parakeet model {self.model_name} loaded successfully")

        except Exception as e:
            logger.exception(f"Failed to load Parakeet model: {e}")
            raise

    def transcribe(self, audio_path: str, language: str = "en") -> Tuple[str, dict]:
        """Transcribe audio using Parakeet."""
        if self.model is None:
            raise RuntimeError("Parakeet Model not loaded")

        start_time = time.time()

        try:
            result = self.model.transcribe(audio_path)
            text = result.text.strip()

            # Calculate duration (approximate if not available)
            duration = time.time() - start_time

            # Attempt to get accurate duration from result if available
            audio_duration = 0.0
            if hasattr(result, 'sentences') and result.sentences:
                audio_duration = result.sentences[-1].end

            # Use processing time if we couldn't get duration from audio
            if audio_duration == 0.0:
                audio_duration = duration

            return text, {
                "duration": audio_duration,
                "language": "en", # Parakeet is primarily English AFAIK
                "backend": "parakeet"
            }

        except Exception as e:
            logger.error(f"Parakeet transcription failed: {e}")
            raise

    @property
    def is_ready(self) -> bool:
        return self.model is not None
