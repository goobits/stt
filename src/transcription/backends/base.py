from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional

class TranscriptionBackend(ABC):
    """Abstract base class for transcription backends."""

    @abstractmethod
    async def load(self):
        """Load the model asynchronously."""
        pass

    @abstractmethod
    def transcribe(self, audio_path: str, language: str = "en") -> Tuple[str, dict]:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to the audio file.
            language: Language code (e.g., "en").

        Returns:
            A tuple containing:
            - The transcribed text.
            - A dictionary with metadata (e.g., duration, language).
        """
        pass

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if the backend is ready/loaded."""
        pass
