#!/usr/bin/env python3
"""
GOOBITS STT Operation Modes

This module contains different operation modes for the STT engine:
- conversation: Continuous VAD-based listening
- tap_to_talk: Hotkey toggle recording
- hold_to_talk: Push-to-talk recording
"""
from __future__ import annotations

from typing import Optional, Type, TYPE_CHECKING
import contextlib

if TYPE_CHECKING:
    from .conversation import ConversationMode as ConversationModeType
    from .tap_to_talk import TapToTalkMode as TapToTalkModeType
    from .hold_to_talk import HoldToTalkMode as HoldToTalkModeType

# Import modes conditionally to avoid dependency issues
ConversationMode: type | None = None
with contextlib.suppress(ImportError):
    from .conversation import ConversationMode

TapToTalkMode: type | None = None
with contextlib.suppress(ImportError):
    from .tap_to_talk import TapToTalkMode

HoldToTalkMode: type | None = None
with contextlib.suppress(ImportError):
    from .hold_to_talk import HoldToTalkMode

__all__ = ["ConversationMode", "HoldToTalkMode", "TapToTalkMode", "ConversationModeType", "TapToTalkModeType", "HoldToTalkModeType"]
