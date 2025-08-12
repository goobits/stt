#!/usr/bin/env python3
"""Debug script to understand punctuation model behavior"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from stt.text_formatting.nlp_provider import get_punctuator

def debug_punctuation_model():
    test_input = "i.e. the code must be clean"
    print(f"Input: '{test_input}'")
    
    punctuator = get_punctuator()
    if punctuator:
        result = punctuator.restore_punctuation(test_input)
        print(f"Punctuation model output: '{result}'")
    else:
        print("No punctuation model available")

if __name__ == "__main__":
    debug_punctuation_model()