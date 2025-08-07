#!/usr/bin/env python3
"""
Basic numeric patterns for text formatting.

This module contains basic number constants, ordinal patterns, fraction patterns,
and numeric range patterns used throughout the text formatting system.
"""
from __future__ import annotations

import re
from typing import Pattern

from ..common import NumberParser


# ==============================================================================
# NUMERIC CONSTANTS
# ==============================================================================

# Number words for speech recognition
NUMBER_WORDS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
    "million",
    "billion",
    "trillion",
]


# ==============================================================================
# ORDINAL AND FRACTION PATTERN BUILDERS
# ==============================================================================


def build_ordinal_pattern(language: str = "en") -> re.Pattern[str]:
    """Build the spoken ordinal pattern for the specified language."""
    # For now, use English patterns. Could be extended for other languages.
    return re.compile(
        r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
        r"eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|"
        r"eighteenth|nineteenth|twentieth|twenty[-\s]?first|twenty[-\s]?second|"
        r"twenty[-\s]?third|twenty[-\s]?fourth|twenty[-\s]?fifth|twenty[-\s]?sixth|"
        r"twenty[-\s]?seventh|twenty[-\s]?eighth|twenty[-\s]?ninth|thirtieth|"
        r"thirty[-\s]?first|thirty[-\s]?second|thirty[-\s]?third|thirty[-\s]?fourth|"
        r"thirty[-\s]?fifth|thirty[-\s]?sixth|thirty[-\s]?seventh|thirty[-\s]?eighth|"
        r"thirty[-\s]?ninth|fortieth|forty[-\s]?first|forty[-\s]?second|forty[-\s]?third|"
        r"forty[-\s]?fourth|forty[-\s]?fifth|forty[-\s]?sixth|forty[-\s]?seventh|"
        r"forty[-\s]?eighth|forty[-\s]?ninth|fiftieth|fifty[-\s]?first|fifty[-\s]?second|"
        r"fifty[-\s]?third|fifty[-\s]?fourth|fifty[-\s]?fifth|fifty[-\s]?sixth|"
        r"fifty[-\s]?seventh|fifty[-\s]?eighth|fifty[-\s]?ninth|sixtieth|sixty[-\s]?first|"
        r"sixty[-\s]?second|sixty[-\s]?third|sixty[-\s]?fourth|sixty[-\s]?fifth|"
        r"sixty[-\s]?sixth|sixty[-\s]?seventh|sixty[-\s]?eighth|sixty[-\s]?ninth|"
        r"seventieth|seventy[-\s]?first|seventy[-\s]?second|seventy[-\s]?third|"
        r"seventy[-\s]?fourth|seventy[-\s]?fifth|seventy[-\s]?sixth|seventy[-\s]?seventh|"
        r"seventy[-\s]?eighth|seventy[-\s]?ninth|eightieth|eighty[-\s]?first|"
        r"eighty[-\s]?second|eighty[-\s]?third|eighty[-\s]?fourth|eighty[-\s]?fifth|"
        r"eighty[-\s]?sixth|eighty[-\s]?seventh|eighty[-\s]?eighth|eighty[-\s]?ninth|"
        r"ninetieth|ninety[-\s]?first|ninety[-\s]?second|ninety[-\s]?third|"
        r"ninety[-\s]?fourth|ninety[-\s]?fifth|ninety[-\s]?sixth|ninety[-\s]?seventh|"
        r"ninety[-\s]?eighth|ninety[-\s]?ninth|hundredth|thousandth)\b",
        re.IGNORECASE,
    )


def build_fraction_pattern(language: str = "en") -> re.Pattern[str]:
    """Build the spoken fraction pattern for the specified language."""
    return re.compile(
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+"
        r"(half|halves|third|thirds|quarter|quarters|fourth|fourths|fifth|fifths|"
        r"sixth|sixths|seventh|sevenths|eighth|eighths|ninth|ninths|tenth|tenths)\b",
        re.IGNORECASE,
    )


def build_compound_fraction_pattern(language: str = "en") -> re.Pattern[str]:
    """Build the compound fraction pattern for mixed numbers like 'one and one half'."""
    return re.compile(
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
        r"and\s+"
        r"(one|a|two|three|four|five|six|seven|eight|nine|ten)\s+"
        r"(half|halves|third|thirds|quarter|quarters|fourth|fourths|fifth|fifths|"
        r"sixth|sixths|seventh|sevenths|eighth|eighths|ninth|ninths|tenth|tenths)\b",
        re.IGNORECASE,
    )


def build_numeric_range_pattern(language: str = "en") -> re.Pattern[str]:
    """Build the numeric range pattern for ranges like 'one to ten'."""
    # Get the number words from a single source of truth
    _number_parser_instance = NumberParser(language)
    _number_words_pattern = "(?:" + "|".join(_number_parser_instance.all_number_words) + ")"

    # Define a reusable pattern for a sequence of one or more number words
    number_word_sequence = f"{_number_words_pattern}(?:\\s+{_number_words_pattern})*"

    # Build the range pattern from components - much more readable and maintainable
    return re.compile(
        rf"""
        \b                      # Word boundary
        (                       # Capture group 1: Start of range
            {number_word_sequence}
        )
        \s+to\s+                # The word "to"
        (                       # Capture group 2: End of range
            {number_word_sequence}
        )
        \b                      # Word boundary
        """,
        re.IGNORECASE | re.VERBOSE,
    )


# ==============================================================================
# GETTER FUNCTIONS
# ==============================================================================


def get_ordinal_pattern(language: str = "en") -> re.Pattern[str]:
    """Get the ordinal pattern for the specified language."""
    return build_ordinal_pattern(language)


def get_fraction_pattern(language: str = "en") -> re.Pattern[str]:
    """Get the fraction pattern for the specified language."""
    return build_fraction_pattern(language)


def get_compound_fraction_pattern(language: str = "en") -> re.Pattern[str]:
    """Get the compound fraction pattern for the specified language."""
    return build_compound_fraction_pattern(language)


def get_numeric_range_pattern(language: str = "en") -> re.Pattern[str]:
    """Get the numeric range pattern for the specified language."""
    return build_numeric_range_pattern(language)


def get_number_words() -> list[str]:
    """Get the list of number words."""
    return NUMBER_WORDS.copy()