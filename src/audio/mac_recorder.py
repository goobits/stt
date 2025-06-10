"""Mac audio recording functionality using sox/ffmpeg.
"""

import os
import subprocess
import time
import wave
import logging
from ..config import (
    AUDIO_SAMPLE_RATE, AUDIO_CHANNELS, RECORD_STOP_DELAY, TEMP_DIR
)

logger = logging.getLogger(__name__)

class MacAudioRecorder:
    """Handles audio recording on Mac using sox or ffmpeg."""

    def __init__(self):
        self.recording_process = None
        self.output_file = None
        self.cancel_file = None
        self.recording_tool = self._detect_recording_tool()

    def _detect_recording_tool(self):
        """Detect available audio recording tool on Mac."""
        # Check for sox first (preferred)
        try:
            subprocess.run(["sox", "--version"],
                         capture_output=True, check=True)
            logger.info("Using sox for Mac audio recording")
            return "sox"
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # Check for ffmpeg
        try:
            subprocess.run(["ffmpeg", "-version"],
                         capture_output=True, check=True)
            logger.info("Using ffmpeg for Mac audio recording")
            return "ffmpeg"
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # Fallback to system audio (requires additional setup)
        logger.warning("Neither sox nor ffmpeg found - audio recording may fail")
        return None

    def start_recording(self, filename_prefix="audio_recording"):
        """Start audio recording to a temporary file."""
        if not self.recording_tool:
            logger.error("No audio recording tool available")
            return None, None

        # Generate unique filenames
        timestamp = str(int(time.time() * 1000))
        self.output_file = os.path.join(TEMP_DIR, f"{filename_prefix}_{timestamp}.wav")
        self.cancel_file = os.path.join(TEMP_DIR, f"cancel_{timestamp}.txt")

        # Clean up any stale cancel files
        self._cleanup_cancel_files()

        try:
            if self.recording_tool == "sox":
                # Use sox for recording
                self.recording_process = subprocess.Popen([
                    "sox",
                    "-d",  # Default input device
                    "-r", str(AUDIO_SAMPLE_RATE),
                    "-c", str(AUDIO_CHANNELS),
                    "-b", "16",  # 16-bit depth
                    self.output_file
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            elif self.recording_tool == "ffmpeg":
                # Use ffmpeg for recording
                self.recording_process = subprocess.Popen([
                    "ffmpeg",
                    "-f", "avfoundation",  # Mac audio framework
                    "-i", ":0",  # Default audio input
                    "-ar", str(AUDIO_SAMPLE_RATE),
                    "-ac", str(AUDIO_CHANNELS),
                    "-acodec", "pcm_s16le",
                    "-y",  # Overwrite output file
                    self.output_file
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            logger.info(f"Started Mac recording to: {self.output_file}")
            return self.output_file, self.cancel_file

        except Exception as e:
            logger.error(f"Failed to start Mac recording: {e}")
            return None, None

    def stop_recording(self):
        """Stop the current recording and return the audio file path."""
        if not self.recording_process:
            return None

        try:
            # Terminate the recording process
            self.recording_process.terminate()
            self.recording_process.wait(timeout=5)

            # Brief delay to ensure file is written
            time.sleep(RECORD_STOP_DELAY)

            # Verify the file exists and has content
            if os.path.exists(self.output_file) and os.path.getsize(self.output_file) > 0:
                logger.info(f"Mac recording stopped: {self.output_file}")
                return self.output_file
            logger.warning("Mac recording file is empty or doesn't exist")
            return None

        except Exception as e:
            logger.error(f"Failed to stop Mac recording: {e}")
            return None
        finally:
            self.recording_process = None

    def cancel_recording(self):
        """Cancel the current recording."""
        if self.cancel_file:
            try:
                with open(self.cancel_file, "w") as f:
                    f.write("cancelled")
                logger.info("Mac recording cancelled")
            except Exception as e:
                logger.error(f"Failed to create cancel file: {e}")

        if self.recording_process:
            try:
                self.recording_process.terminate()
                self.recording_process.wait(timeout=5)
            except Exception as e:
                logger.error(f"Failed to terminate Mac recording process: {e}")
            finally:
                self.recording_process = None

        # Clean up output file
        if self.output_file and os.path.exists(self.output_file):
            try:
                os.remove(self.output_file)
            except Exception as e:
                logger.error(f"Failed to remove Mac recording file: {e}")

    def get_audio_duration(self, audio_file):
        """Get the duration of an audio file in seconds."""
        try:
            with wave.open(audio_file, "rb") as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                duration = frames / float(sample_rate)
                return duration
        except Exception as e:
            logger.error(f"Failed to get Mac audio duration: {e}")
            return None

    def _cleanup_cancel_files(self):
        """Clean up old cancel files to prevent conflicts."""
        try:
            cancel_files = [f for f in os.listdir(TEMP_DIR) if f.startswith("cancel_")]
            for cancel_file in cancel_files:
                os.remove(os.path.join(TEMP_DIR, cancel_file))
        except Exception as e:
            logger.warning(f"Failed to cleanup cancel files: {e}")

    def is_recording(self):
        """Check if currently recording."""
        return self.recording_process is not None and self.recording_process.poll() is None

    def get_recording_tool_info(self):
        """Get information about the detected recording tool."""
        return {
            "tool": self.recording_tool,
            "available": self.recording_tool is not None,
            "sox_available": self._tool_available("sox"),
            "ffmpeg_available": self._tool_available("ffmpeg")
        }

    def _tool_available(self, tool_name):
        """Check if a specific tool is available."""
        try:
            subprocess.run([tool_name, "--version"],
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
