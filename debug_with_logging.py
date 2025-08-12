#!/usr/bin/env python3
"""Debug script with logging enabled"""

import sys
import os
import logging

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

from stt.text_formatting.formatter import format_transcription

def debug_abbreviation_processing():
    test_input = "i.e. the code must be clean"
    print(f"Input: '{test_input}'")
    
    result = format_transcription(test_input)
    print(f"Output: '{result}'")
    print(f"Expected: 'I.e. the code must be clean'")
    
    if result == "I.e. the code must be clean":
        print("✓ Test PASSED!")
    else:
        print("✗ Test FAILED!")

if __name__ == "__main__":
    debug_abbreviation_processing()