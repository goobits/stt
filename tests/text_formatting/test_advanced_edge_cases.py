#!/usr/bin/env python3
"""
Test advanced edge cases, entity conflicts, and untested functionality
in the text formatting system.
"""

import pytest
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from stt_hotkeys.text_formatting.formatter import format_transcription

class TestUntestedEntities:
    """Test entities that are implemented but not yet covered by tests."""

    def test_fractions_and_ranges(self):
        """Test fraction and numeric range formatting."""
        test_cases = {
            "one half": "½",
            "two thirds": "⅔",
            "three quarters": "¾",
            "ten to twenty": "10-20",
            "one hundred to one fifty": "100-150",
        }
        for input_text, expected in test_cases.items():
            assert format_transcription(input_text) == expected

    def test_relative_time_and_ordinals(self):
        """Test relative time and ordinal number formatting."""
        test_cases = {
            "it is quarter past three": "It is 3:15.",
            "the meeting is at half past nine": "The meeting is at 9:30.",
            "he came in first place": "He came in 1st place.",
            "this is the twenty third time": "This is the 23rd time.",
        }
        for input_text, expected in test_cases.items():
            assert format_transcription(input_text) == expected

    def test_specialized_numeric_entities(self):
        """Test specialized numeric entities like temperature and metric units."""
        test_cases = {
            "it is twenty degrees celsius outside": "It is 20°C outside.",
            "the oven is at three hundred fifty fahrenheit": "The oven is at 350°F",
            "the distance is five kilometers": "The distance is 5 km.",
            "add ten kilograms of flour": "Add 10 kg of flour.",
        }
        for input_text, expected in test_cases.items():
            assert format_transcription(input_text) == expected

    def test_fun_entities_emoji_and_music(self):
        """Test spoken emoji and music notation."""
        test_cases = {
            "that makes me a happy smiley face": "That makes me a happy 🙂.",
            "launch the rocket emoji": "Launch the 🚀.",
            "play a C sharp": "Play a C♯.",
            "the note is B flat": "The note is B♭.",
        }
        for input_text, expected in test_cases.items():
            assert format_transcription(input_text) == expected


class TestComplexLogic:
    """Test complex conditional logic, like filename casing."""

    def test_filename_casing_rules(self):
        """Test that different file extensions trigger correct casing rules."""
        test_cases = {
            # PascalCase for .tsx, .java, .cs
            "my new component dot tsx": "MyNewComponent.tsx",
            "create a user service dot java": "Create a UserService.java.",
            # kebab-case for .css, .scss
            "edit my stylesheet dot css": "edit my-stylesheet.css.",
            # lower_snake (default) for .py
            "open my script dot py": "open my_script.py.",
            # UPPER_SNAKE for .md
            "check the readme dot md file": "Check the README.md file",
        }
        for input_text, expected in test_cases.items():
            # Some get punctuation, some don't. Strip for consistency.
            assert format_transcription(input_text).strip('. ') == expected.strip('. ')


class TestEntityConflictResolution:
    """Test how the system resolves overlapping or conflicting entities."""

    def test_filename_vs_url_conflict(self):
        """
        Test that FILENAME (high priority) wins over SPOKEN_URL (lower priority).
        The input can be interpreted as both a filename and a URL.
        """
        input_text = "go to main dot py on example dot com"
        # Expected: "main.py" is a FILENAME entity, the rest is just text.
        # Incorrect: The whole phrase is treated as a URL.
        expected = "Go to main.py on example.com."
        assert format_transcription(input_text) == expected

    def test_cardinal_vs_datasize_conflict(self):
        """Test that DATA_SIZE (specific) wins over CARDINAL (generic)."""
        input_text = "the file is five megabytes"
        # Expected: "five megabytes" is a single DATA_SIZE entity -> "5MB"
        # Incorrect: "five" is a CARDINAL -> "5", and "megabytes" is just text.
        expected = "The file is 5MB."
        assert format_transcription(input_text) == expected


class TestBoundaryAndNegativeCases:
    """Test the formatter's behavior with unusual or invalid inputs."""

    def test_input_with_only_filler_words(self):
        """Test that input containing only filler words results in an empty string."""
        assert format_transcription("um uh hmm") == ""
        assert format_transcription("uhh") == ""

    def test_malformed_spoken_numbers(self):
        """Test that grammatically incorrect numbers are not converted."""
        # "twenty ten" is not a valid number.
        input_text = "the value is twenty ten"
        expected = "The value is twenty ten."
        assert format_transcription(input_text) == expected

    def test_unknown_file_extension(self):
        """Test that spoken filenames with unknown extensions are still formatted."""
        input_text = "open my custom file dot custom"
        # Should still format the dot and snake_case the name.
        expected = "open my_custom_file.custom"
        assert format_transcription(input_text) == expected