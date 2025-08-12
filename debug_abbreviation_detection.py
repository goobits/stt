#!/usr/bin/env python3
"""Debug script to test abbreviation detection directly"""

import sys
import os
import re

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from stt.text_formatting.pattern_modules.letter_patterns import ABBREVIATION_PATTERN

def debug_abbreviation_detection():
    test_input = "i.e. the code must be clean"
    print(f"Input: '{test_input}'")
    print(f"Pattern: {ABBREVIATION_PATTERN.pattern}")
    
    matches = list(ABBREVIATION_PATTERN.finditer(test_input))
    print(f"Found {len(matches)} matches:")
    
    for i, match in enumerate(matches):
        print(f"  {i+1}. Match at [{match.start()}:{match.end()}] = '{match.group(0)}' (group 1: '{match.group(1)}')")

if __name__ == "__main__":
    debug_abbreviation_detection()