#!/usr/bin/env python3
"""
Performance profiling script for text formatting engine.
This will identify actual hotspots for targeted optimization.
"""
import cProfile
import pstats
import io
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Test samples for profiling
TEST_TEXTS = [
    "i have five plus years of experience",
    "i think x equals five at example dot com but the API says otherwise", 
    "edit change log dot md",
    "the error is in main dot js on line five",
    "function opens the door dot js",
    "the file utils dot py is ready",
    "the derivative of x squared",
    "a is an element of b",
    "from nine to five",
    "between one and two hours",
    "three thirty and twenty seconds PM",
    "version two point one point zero",
    "go to http colon slash slash example dot com",
    "send email to test at domain dot org",
    "run npm install dash dash save",
    "the price is twenty five dollars and fifty cents",
    "set temperature to twenty three degrees celsius",
    "download five gigabytes of data",
    "wait for three point five seconds",
    "multiply x by negative two",
]

def profile_formatter():
    """Profile the text formatter with realistic test cases."""
    from stt.text_formatting.formatter import TextFormatter
    
    # Create formatter instance
    formatter = TextFormatter(language="en")
    
    def run_formatting_tests():
        """Run formatting tests for profiling."""
        for text in TEST_TEXTS:
            # Run each test multiple times to get meaningful data
            for _ in range(10):
                result = formatter.format_transcription(text)
    
    # Create profiler
    pr = cProfile.Profile()
    
    print("Starting performance profiling...")
    
    # Profile the formatting
    pr.enable()
    run_formatting_tests()
    pr.disable()
    
    # Capture stats
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    
    # Write detailed stats to file
    with open('/workspace/profiling_results.txt', 'w') as f:
        f.write(s.getvalue())
    
    # Print top hotspots to console
    print("\n=== TOP 20 PERFORMANCE HOTSPOTS ===")
    s.seek(0)
    lines = s.getvalue().split('\n')
    
    # Find the start of the function list
    start_idx = 0
    for i, line in enumerate(lines):
        if 'ncalls' in line and 'tottime' in line:
            start_idx = i + 1
            break
    
    # Print header and top 20 functions
    if start_idx > 0:
        print(lines[start_idx - 1])  # Header
        for i in range(start_idx, min(start_idx + 20, len(lines))):
            if lines[i].strip():
                print(lines[i])
    
    print(f"\nDetailed profiling results saved to: /workspace/profiling_results.txt")
    print(f"Tested {len(TEST_TEXTS)} different text samples, 10 iterations each")

if __name__ == "__main__":
    profile_formatter()