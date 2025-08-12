#!/usr/bin/env python3
"""Debug script to understand entity detection"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from stt.text_formatting.formatter_components.pipeline.step2_detection import detect_all_entities

def debug_entity_detection():
    test_input = "i.e. the code must be clean"
    print(f"Input: '{test_input}'")
    
    # Detect entities
    entities = detect_all_entities(test_input, language="en")
    
    print(f"Found {len(entities)} entities:")
    for i, entity in enumerate(entities):
        entity_text = test_input[entity.start:entity.end]
        print(f"  {i+1}. {entity.type} at [{entity.start}:{entity.end}] = '{entity_text}' (expected: '{entity.text}')")

if __name__ == "__main__":
    debug_entity_detection()