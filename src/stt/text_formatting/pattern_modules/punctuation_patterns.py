#!/usr/bin/env python3
"""
Punctuation normalization and profanity filtering patterns for text formatting.

This module contains patterns and functions for normalizing punctuation,
handling repeated punctuation marks, and filtering profanity.
"""
from __future__ import annotations

import re

from ..pattern_cache import cached_pattern


# ==============================================================================
# PUNCTUATION NORMALIZATION
# ==============================================================================

# Pattern builder functions
@cached_pattern  
def build_repeated_commas_pattern() -> re.Pattern[str]:
    """Build repeated commas, semicolons, colons pattern."""
    return re.compile(r"([,;:])\1+")


@cached_pattern
def build_repeated_dots_pattern() -> re.Pattern[str]:
    """Build pattern for repeated dots."""
    return re.compile(r"\.\.+")


@cached_pattern
def build_repeated_question_marks_pattern() -> re.Pattern[str]:
    """Build pattern for repeated question marks."""
    return re.compile(r"\?\?+")


@cached_pattern
def build_repeated_exclamation_marks_pattern() -> re.Pattern[str]:
    """Build pattern for repeated exclamation marks."""
    return re.compile(r"!!+")


# Normalize repeated punctuation
def get_repeated_punctuation_patterns() -> list[tuple[re.Pattern[str], str]]:
    """Get the repeated punctuation patterns with their replacements."""
    return [
        (build_repeated_commas_pattern(), r"\1"),  # Repeated commas, semicolons, colons
        (build_repeated_dots_pattern(), "."),  # Multiple dots to single dot
        (build_repeated_question_marks_pattern(), "?"),  # Multiple question marks
        (build_repeated_exclamation_marks_pattern(), "!"),  # Multiple exclamation marks
    ]


# For backward compatibility
REPEATED_PUNCTUATION_PATTERNS = get_repeated_punctuation_patterns()


def get_repeated_dots_pattern() -> re.Pattern[str]:
    """Get the repeated dots pattern."""
    return REPEATED_DOTS_PATTERN


def get_repeated_question_marks_pattern() -> re.Pattern[str]:
    """Get the repeated question marks pattern."""
    return REPEATED_QUESTION_MARKS_PATTERN


def get_repeated_exclamation_marks_pattern() -> re.Pattern[str]:
    """Get the repeated exclamation marks pattern."""
    return REPEATED_EXCLAMATION_MARKS_PATTERN


# Pre-compiled patterns for performance
REPEATED_DOTS_PATTERN = build_repeated_dots_pattern()
REPEATED_QUESTION_MARKS_PATTERN = build_repeated_question_marks_pattern()
REPEATED_EXCLAMATION_MARKS_PATTERN = build_repeated_exclamation_marks_pattern()


# ==============================================================================
# PROFANITY FILTERING
# ==============================================================================

@cached_pattern
def create_profanity_pattern(profanity_words: list[str]) -> re.Pattern[str]:
    """
    Create a pattern to filter profanity words.

    Only matches lowercase profanity to avoid filtering proper nouns
    and sentence beginnings (e.g., "Hell, Michigan" vs "go to hell").
    """
    escaped_words = [re.escape(word) for word in profanity_words]
    # Match only when the word starts with lowercase letter
    pattern_string = r"\b(?:" + "|".join(f"[{word[0].lower()}]{re.escape(word[1:])}" for word in escaped_words) + r")\b"
    return re.compile(pattern_string)