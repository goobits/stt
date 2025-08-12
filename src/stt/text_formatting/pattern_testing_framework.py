#!/usr/bin/env python3
"""
Pattern Testing Framework for Text Formatting

This framework provides systematic validation and testing of regex patterns
and text conversion patterns used in the text formatting system.

Features:
- Pattern isolation testing
- Context pattern validation
- Pattern conflict detection
- Performance measurement
- Automated test case generation
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Pattern, Tuple, Any
from pathlib import Path
import json

from stt.core.config import setup_logging
from stt.text_formatting.common import Entity, EntityType
from stt.text_formatting.constants import get_resources
from stt.text_formatting.formatter import format_transcription

logger = setup_logging(__name__)


@dataclass
class PatternTestCase:
    """Represents a single pattern test case."""
    input_text: str
    expected_output: str
    pattern_name: str
    context: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PatternTestResult:
    """Results of pattern testing."""
    test_case: PatternTestCase
    actual_output: str
    passed: bool
    execution_time_ms: float
    error_message: Optional[str] = None
    matched_entities: Optional[List[Entity]] = None
    formatter_output: Optional[str] = None  # Full formatter output for integration testing


@dataclass
class PatternDefinition:
    """Defines a pattern for testing."""
    name: str
    pattern: Pattern[str]
    entity_type: EntityType
    test_cases: List[PatternTestCase]
    description: str
    category: str = "general"


class PatternTestRegistry:
    """Registry for pattern test cases and definitions."""
    
    def __init__(self, language: str = "en"):
        self.language = language
        self.patterns: Dict[str, PatternDefinition] = {}
        self.test_cases: Dict[str, List[PatternTestCase]] = {}
        self._initialize_default_patterns()
    
    def _initialize_default_patterns(self):
        """Initialize default patterns for testing."""
        # Load time range patterns from resources
        resources = get_resources(self.language)
        temporal_resources = resources.get("temporal", {})
        
        # Time range pattern test cases - comprehensive set
        time_range_cases = [
            PatternTestCase(
                "from nine to five",
                "From 9 to 5",
                "time_range",
                context="standalone_time_range",
                metadata={"priority": "high", "test_type": "regression"}
            ),
            PatternTestCase(
                "meeting from two to three on friday",
                "Meeting from 2 to 3 on Friday",
                "time_range",
                context="time_range_in_sentence",
                metadata={"priority": "high", "test_type": "regression"}
            ),
            PatternTestCase(
                "from ten AM to two PM",
                "From 10 AM to 2 PM",
                "time_range",
                context="time_range_with_ampm",
                metadata={"priority": "medium", "test_type": "enhancement"}
            ),
            PatternTestCase(
                "nine to five",
                "9-5",
                "time_range_compact",
                context="compact_time_range",
                metadata={"priority": "high", "test_type": "regression"}
            ),
            PatternTestCase(
                "nine to five shift",
                "9-5 shift",
                "time_range_compact",
                context="time_range_with_job_context",
                metadata={"priority": "medium", "test_type": "enhancement"}
            ),
            # Additional edge cases for comprehensive testing
            PatternTestCase(
                "from eight to six",
                "From 8 to 6",
                "time_range",
                context="reverse_time_range",
                metadata={"priority": "medium", "test_type": "edge_case"}
            ),
            PatternTestCase(
                "between one and three",
                "Between 1 and 3",
                "time_range_between",
                context="between_time_range",
                metadata={"priority": "medium", "test_type": "alternative_pattern"}
            ),
            PatternTestCase(
                "working hours are nine to five",
                "Working hours are 9-5",
                "time_range_context",
                context="time_range_with_context",
                metadata={"priority": "medium", "test_type": "contextual"}
            )
        ]
        
        # Numerical entity test cases for conflict detection
        numeric_conflict_cases = [
            PatternTestCase(
                "from two to five dollars",
                "From $2 to $5",  # Should be financial, not time
                "financial_range",
                context="financial_range_not_time",
                metadata={"priority": "high", "test_type": "conflict_detection"}
            ),
            PatternTestCase(
                "divide from nine to five",
                "Divide from 9 to 5",  # Mathematical context, not time
                "math_range",
                context="mathematical_not_time",
                metadata={"priority": "medium", "test_type": "conflict_detection"}
            )
        ]
        
        self.test_cases["time_range"] = time_range_cases
        self.test_cases["conflict_detection"] = numeric_conflict_cases
    
    def register_pattern(self, pattern_def: PatternDefinition):
        """Register a pattern definition."""
        self.patterns[pattern_def.name] = pattern_def
        logger.info(f"Registered pattern '{pattern_def.name}' with {len(pattern_def.test_cases)} test cases")
    
    def register_test_case(self, category: str, test_case: PatternTestCase):
        """Register a test case for a pattern category."""
        if category not in self.test_cases:
            self.test_cases[category] = []
        self.test_cases[category].append(test_case)
    
    def get_test_cases(self, category: str) -> List[PatternTestCase]:
        """Get test cases for a specific category."""
        return self.test_cases.get(category, [])
    
    def get_all_test_cases(self) -> List[PatternTestCase]:
        """Get all registered test cases."""
        all_cases = []
        for cases in self.test_cases.values():
            all_cases.extend(cases)
        return all_cases


class PatternValidator:
    """Validates patterns in isolation and detects conflicts."""
    
    def __init__(self, language: str = "en"):
        self.language = language
        self.registry = PatternTestRegistry(language)
        self.formatter_cache = {}  # Cache formatter instances for performance
    
    def validate_pattern_syntax(self, pattern: Pattern[str]) -> Tuple[bool, Optional[str]]:
        """Validate that a regex pattern compiles correctly."""
        try:
            # Test the pattern with various inputs
            test_inputs = ["test string", "123 456", "from nine to five", ""]
            for test_input in test_inputs:
                pattern.search(test_input)
            return True, None
        except Exception as e:
            return False, str(e)
    
    def test_pattern_isolation(self, pattern: Pattern[str], test_cases: List[PatternTestCase]) -> List[PatternTestResult]:
        """Test a pattern in isolation with provided test cases."""
        results = []
        
        for test_case in test_cases:
            start_time = time.perf_counter()
            
            try:
                # Test if the pattern matches the input
                match = pattern.search(test_case.input_text)
                execution_time = (time.perf_counter() - start_time) * 1000
                
                if match:
                    # For isolation testing, we just check if the pattern matches
                    # The actual conversion logic is tested separately
                    result = PatternTestResult(
                        test_case=test_case,
                        actual_output=match.group(),
                        passed=True,  # Pattern matched
                        execution_time_ms=execution_time,
                    )
                else:
                    result = PatternTestResult(
                        test_case=test_case,
                        actual_output="",
                        passed=False,
                        execution_time_ms=execution_time,
                        error_message="Pattern did not match input"
                    )
                
                results.append(result)
                
            except Exception as e:
                execution_time = (time.perf_counter() - start_time) * 1000
                results.append(PatternTestResult(
                    test_case=test_case,
                    actual_output="",
                    passed=False,
                    execution_time_ms=execution_time,
                    error_message=str(e)
                ))
        
        return results
    
    def test_full_formatter_integration(self, test_cases: List[PatternTestCase]) -> List[PatternTestResult]:
        """Test patterns using the full text formatter pipeline."""
        results = []
        
        for test_case in test_cases:
            start_time = time.perf_counter()
            
            try:
                # Use the actual formatter to process the text
                formatter_output = format_transcription(test_case.input_text)
                execution_time = (time.perf_counter() - start_time) * 1000
                
                # Compare with expected output
                passed = formatter_output.strip() == test_case.expected_output.strip()
                
                result = PatternTestResult(
                    test_case=test_case,
                    actual_output=formatter_output,
                    passed=passed,
                    execution_time_ms=execution_time,
                    formatter_output=formatter_output,
                    error_message=None if passed else f"Expected '{test_case.expected_output}', got '{formatter_output}'"
                )
                
                results.append(result)
                
            except Exception as e:
                execution_time = (time.perf_counter() - start_time) * 1000
                results.append(PatternTestResult(
                    test_case=test_case,
                    actual_output="",
                    passed=False,
                    execution_time_ms=execution_time,
                    error_message=f"Formatter error: {str(e)}",
                    formatter_output=None
                ))
        
        return results
    
    def analyze_pattern_coverage(self, test_results: List[PatternTestResult]) -> Dict[str, Any]:
        """Analyze pattern coverage and effectiveness."""
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results if r.passed)
        failed_tests = total_tests - passed_tests
        
        # Categorize by test type
        by_priority = {"high": [], "medium": [], "low": []}
        by_test_type = {}
        by_context = {}
        
        for result in test_results:
            # Priority analysis
            priority = result.test_case.metadata.get("priority", "medium") if result.test_case.metadata else "medium"
            by_priority[priority].append(result)
            
            # Test type analysis
            test_type = result.test_case.metadata.get("test_type", "unknown") if result.test_case.metadata else "unknown"
            if test_type not in by_test_type:
                by_test_type[test_type] = []
            by_test_type[test_type].append(result)
            
            # Context analysis
            context = result.test_case.context or "no_context"
            if context not in by_context:
                by_context[context] = []
            by_context[context].append(result)
        
        # Calculate metrics
        priority_metrics = {}
        for priority, results in by_priority.items():
            if results:
                priority_metrics[priority] = {
                    "total": len(results),
                    "passed": sum(1 for r in results if r.passed),
                    "success_rate": sum(1 for r in results if r.passed) / len(results) * 100
                }
        
        test_type_metrics = {}
        for test_type, results in by_test_type.items():
            if results:
                test_type_metrics[test_type] = {
                    "total": len(results),
                    "passed": sum(1 for r in results if r.passed),
                    "success_rate": sum(1 for r in results if r.passed) / len(results) * 100
                }
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "overall_success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "priority_breakdown": priority_metrics,
            "test_type_breakdown": test_type_metrics,
            "context_breakdown": {context: len(results) for context, results in by_context.items()}
        }
    
    def detect_pattern_conflicts(self, patterns: List[Tuple[str, Pattern[str]]]) -> Dict[str, List[str]]:
        """Detect overlapping patterns that might cause conflicts."""
        conflicts = {}
        
        # Enhanced test strings covering more edge cases
        test_strings = [
            # Time range patterns
            "from nine to five",
            "meeting from two to three",
            "nine to five shift",
            "between ten and twelve",
            "two to three hours",
            "from january to march",
            # Potential conflict patterns
            "from two to five dollars",  # Financial vs time
            "divide from nine to five",  # Mathematical vs time
            "nine to five employees",    # Job context vs time
            "from page nine to five",   # Document reference vs time
            "numbers from one to ten",  # Numerical sequence vs time
            "chapters two to five",     # Reference vs time
            "versions nine to five",    # Technical vs time
        ]
        
        for test_string in test_strings:
            matching_patterns = []
            for name, pattern in patterns:
                try:
                    if pattern.search(test_string):
                        matching_patterns.append(name)
                except Exception as e:
                    logger.warning(f"Pattern '{name}' failed on '{test_string}': {e}")
            
            if len(matching_patterns) > 1:
                conflicts[test_string] = matching_patterns
                
        return conflicts
    
    def generate_pattern_performance_report(self, results: List[PatternTestResult]) -> Dict[str, Any]:
        """Generate detailed performance analysis of patterns."""
        if not results:
            return {"error": "No results to analyze"}
            
        # Timing analysis
        execution_times = [r.execution_time_ms for r in results]
        avg_time = sum(execution_times) / len(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)
        
        # Performance by context
        context_performance = {}
        for result in results:
            context = result.test_case.context or "no_context"
            if context not in context_performance:
                context_performance[context] = []
            context_performance[context].append(result.execution_time_ms)
        
        context_stats = {}
        for context, times in context_performance.items():
            context_stats[context] = {
                "avg_time_ms": sum(times) / len(times),
                "min_time_ms": min(times),
                "max_time_ms": max(times),
                "test_count": len(times)
            }
        
        # Identify slow patterns
        slow_threshold = avg_time * 2  # Patterns taking more than 2x average
        slow_patterns = []
        for result in results:
            if result.execution_time_ms > slow_threshold:
                slow_patterns.append({
                    "input": result.test_case.input_text,
                    "time_ms": result.execution_time_ms,
                    "context": result.test_case.context
                })
        
        return {
            "timing_stats": {
                "avg_time_ms": avg_time,
                "min_time_ms": min_time,
                "max_time_ms": max_time,
                "total_tests": len(results)
            },
            "context_performance": context_stats,
            "slow_patterns": slow_patterns,
            "performance_threshold_ms": slow_threshold
        }


class PatternPerformanceTester:
    """Tests pattern performance and optimization."""
    
    def __init__(self):
        self.benchmark_inputs = [
            "short text",
            "medium length text with some numbers like nine to five",
            "much longer text with multiple potential pattern matches from nine to five and two to three and between ten and twelve AM to two PM with various other patterns that might match",
        ]
    
    def benchmark_pattern(self, pattern: Pattern[str], iterations: int = 1000) -> Dict[str, float]:
        """Benchmark pattern performance."""
        results = {}
        
        for test_input in self.benchmark_inputs:
            times = []
            for _ in range(iterations):
                start_time = time.perf_counter()
                pattern.search(test_input)
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms
            
            results[f"input_length_{len(test_input)}"] = {
                "avg_time_ms": sum(times) / len(times),
                "min_time_ms": min(times),
                "max_time_ms": max(times)
            }
        
        return results


class PatternTestFramework:
    """Main framework for pattern testing and validation."""
    
    def __init__(self, language: str = "en"):
        self.language = language
        self.registry = PatternTestRegistry(language)
        self.validator = PatternValidator(language)
        self.performance_tester = PatternPerformanceTester()
        self.results_cache = {}
        self.integration_mode = True  # Enable full formatter integration by default
    
    def add_time_range_patterns(self):
        """Add time range patterns for testing."""
        # Define common time range patterns
        from_to_pattern = re.compile(
            r"\bfrom\s+(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+to\s+(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
            re.IGNORECASE
        )
        
        compact_range_pattern = re.compile(
            r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+to\s+(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)(?=\s+(shift|schedule|hours|work|job|daily))",
            re.IGNORECASE
        )
        
        # Register patterns
        time_range_def = PatternDefinition(
            name="from_to_time_range",
            pattern=from_to_pattern,
            entity_type=EntityType.TIME_CONTEXT,
            test_cases=self.registry.get_test_cases("time_range"),
            description="Matches 'from X to Y' time range patterns",
            category="temporal"
        )
        
        compact_range_def = PatternDefinition(
            name="compact_time_range",
            pattern=compact_range_pattern,
            entity_type=EntityType.TIME_CONTEXT,
            test_cases=self.registry.get_test_cases("time_range"),
            description="Matches compact 'X to Y' time ranges in job contexts",
            category="temporal"
        )
        
        self.registry.register_pattern(time_range_def)
        self.registry.register_pattern(compact_range_def)
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete pattern validation suite."""
        logger.info("Starting comprehensive pattern validation")
        
        # Add our test patterns
        self.add_time_range_patterns()
        
        results = {
            "timestamp": time.time(),
            "language": self.language,
            "pattern_results": {},
            "conflict_analysis": {},
            "performance_benchmarks": {},
            "summary": {}
        }
        
        # Test each registered pattern
        for pattern_name, pattern_def in self.registry.patterns.items():
            logger.info(f"Testing pattern: {pattern_name}")
            
            # Syntax validation
            is_valid, error = self.validator.validate_pattern_syntax(pattern_def.pattern)
            
            # Isolation testing
            isolation_results = self.validator.test_pattern_isolation(
                pattern_def.pattern,
                pattern_def.test_cases
            )
            
            # Performance testing
            performance_results = self.performance_tester.benchmark_pattern(pattern_def.pattern)
            
            results["pattern_results"][pattern_name] = {
                "valid_syntax": is_valid,
                "syntax_error": error,
                "isolation_tests": [
                    {
                        "input": r.test_case.input_text,
                        "expected": r.test_case.expected_output,
                        "actual": r.actual_output,
                        "passed": r.passed,
                        "execution_time_ms": r.execution_time_ms,
                        "error": r.error_message,
                        "context": r.test_case.context
                    }
                    for r in isolation_results
                ],
                "performance": performance_results
            }
        
        # Conflict detection
        all_patterns = [(name, pdef.pattern) for name, pdef in self.registry.patterns.items()]
        conflicts = self.validator.detect_pattern_conflicts(all_patterns)
        results["conflict_analysis"] = conflicts
        
        # Generate summary
        total_tests = sum(len(pr["isolation_tests"]) for pr in results["pattern_results"].values())
        passed_tests = sum(
            sum(1 for test in pr["isolation_tests"] if test["passed"])
            for pr in results["pattern_results"].values()
        )
        
        results["summary"] = {
            "total_patterns": len(self.registry.patterns),
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "patterns_with_conflicts": len(conflicts),
        }
        
        logger.info(f"Pattern validation complete. Success rate: {results['summary']['success_rate']:.1f}%")
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save test results to a file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_file}")
    
    def generate_pattern_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable report from test results."""
        report = []
        report.append("=" * 60)
        report.append("PATTERN TESTING FRAMEWORK REPORT")
        report.append("=" * 60)
        
        summary = results["summary"]
        report.append(f"\nSUMMARY:")
        report.append(f"  Total patterns tested: {summary['total_patterns']}")
        report.append(f"  Total test cases: {summary['total_tests']}")
        report.append(f"  Passed: {summary['passed_tests']}")
        report.append(f"  Failed: {summary['failed_tests']}")
        report.append(f"  Success rate: {summary['success_rate']:.1f}%")
        report.append(f"  Patterns with conflicts: {summary['patterns_with_conflicts']}")
        
        # Failed tests detail
        report.append(f"\nFAILED TESTS:")
        for pattern_name, pattern_results in results["pattern_results"].items():
            failed_tests = [test for test in pattern_results["isolation_tests"] if not test["passed"]]
            if failed_tests:
                report.append(f"\n  Pattern: {pattern_name}")
                for test in failed_tests:
                    report.append(f"    Input: '{test['input']}'")
                    report.append(f"    Expected: '{test['expected']}'")
                    report.append(f"    Actual: '{test['actual']}'")
                    report.append(f"    Context: {test['context']}")
                    if test['error']:
                        report.append(f"    Error: {test['error']}")
                    report.append("")
        
        # Pattern conflicts
        if results["conflict_analysis"]:
            report.append(f"\nPATTERN CONFLICTS:")
            for test_input, conflicting_patterns in results["conflict_analysis"].items():
                report.append(f"  Input: '{test_input}'")
                report.append(f"  Conflicting patterns: {', '.join(conflicting_patterns)}")
                report.append("")
        
        return "\n".join(report)


# Test utility functions
def create_test_framework() -> PatternTestFramework:
    """Create and initialize the pattern test framework."""
    return PatternTestFramework()


def run_time_range_pattern_tests() -> Dict[str, Any]:
    """Run tests specifically for time range patterns."""
    framework = create_test_framework()
    results = framework.run_full_validation()
    return results


if __name__ == "__main__":
    # Run the framework
    framework = create_test_framework()
    results = framework.run_full_validation()
    
    # Generate and print report
    report = framework.generate_pattern_report(results)
    print(report)
    
    # Save results
    framework.save_results(results, "/tmp/pattern_test_results.json")