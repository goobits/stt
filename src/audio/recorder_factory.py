"""Cross-platform audio recorder factory.
"""

import platform
import logging

logger = logging.getLogger(__name__)

def get_audio_recorder():
    """Get the appropriate audio recorder for the current platform."""
    system = platform.system()
    
    if system == "Linux":
        from .linux_recorder import AudioRecorder
        logger.debug("Using Linux audio recorder (arecord)")
        return AudioRecorder()
    elif system == "Darwin":
        from .mac_recorder import MacAudioRecorder
        logger.debug("Using Mac audio recorder (sox/ffmpeg)")
        return MacAudioRecorder()
    else:
        raise NotImplementedError(f"Audio recording not supported on {system}")

def get_recorder_info():
    """Get information about the current platform's audio recording capabilities."""
    system = platform.system()
    
    if system == "Linux":
        return {
            'platform': 'Linux',
            'recorder_type': 'arecord',
            'description': 'ALSA audio recording utility',
            'requirements': ['arecord (part of alsa-utils package)']
        }
    elif system == "Darwin":
        from .mac_recorder import MacAudioRecorder
        recorder = MacAudioRecorder()
        tool_info = recorder.get_recording_tool_info()
        
        return {
            'platform': 'Mac',
            'recorder_type': tool_info['tool'],
            'description': f"Mac audio recording using {tool_info['tool']}",
            'requirements': ['sox (brew install sox) or ffmpeg (brew install ffmpeg)'],
            'tool_status': tool_info
        }
    else:
        return {
            'platform': system,
            'recorder_type': None,
            'description': 'Platform not supported',
            'requirements': []
        }