#!/usr/bin/env python3
"""
GOOBITS STT Operation Modes

This module contains different operation modes for the STT engine:
- conversation: Continuous VAD-based listening
- tap_to_talk: Hotkey toggle recording
- hold_to_talk: Push-to-talk recording
"""

# Import modes conditionally to avoid dependency issues
try:
    from .conversation import ConversationMode
except ImportError:
    ConversationMode = None

try:
    from .tap_to_talk import TapToTalkMode
except ImportError:
    TapToTalkMode = None

try:
    from .hold_to_talk import HoldToTalkMode
except ImportError:
    HoldToTalkMode = None

__all__ = ["ConversationMode", "TapToTalkMode", "HoldToTalkMode"]