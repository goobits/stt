#!/usr/bin/env python3
"""
Pattern Integration Tester

This module extends the pattern testing framework to provide systematic
validation of patterns through the full text formatting pipeline. It ensures
patterns work correctly in the complete system, not just in isolation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path

from stt.core.config import setup_logging
from .pattern_testing_framework import (
    PatternTestFramework, PatternTestCase, PatternTestResult, create_test_framework
)
from .formatter import format_transcription

logger = setup_logging(__name__)


@dataclass
class IntegrationTestResult:
    """Result from full integration testing."""
    test_case: PatternTestCase
    formatter_output: str
    expected_output: str
    passed: bool
    execution_time_ms: float
    error_message: Optional[str] = None
    partial_match: bool = False
    confidence_score: float = 0.0


@dataclass
class PatternIntegrationReport:
    """Comprehensive report for pattern integration testing."""
    pattern_category: str
    test_count: int
    passed_count: int
    failed_count: int
    partial_count: int
    success_rate: float
    avg_execution_time_ms: float
    detailed_results: List[IntegrationTestResult]
    recommendations: List[str]
    timestamp: float


class PatternIntegrationTester:
    """Tests patterns through the complete text formatting pipeline."""
    
    def __init__(self, language: str = "en"):
        self.language = language
        self.framework = create_test_framework()
        self.results_cache = {}
        
    def test_pattern_category(self, category: str) -> PatternIntegrationReport:
        """Test all patterns in a specific category through full pipeline."""
        logger.info(f"Starting integration test for category: {category}")
        
        test_cases = self.framework.registry.get_test_cases(category)
        if not test_cases:
            logger.warning(f"No test cases found for category: {category}")
            return self._empty_report(category)
        
        start_time = time.time()
        results = []
        
        # Run each test case through the full formatter
        for test_case in test_cases:
            result = self._run_integration_test(test_case)
            results.append(result)
            
            # Log progress for high priority cases
            if (test_case.metadata and 
                test_case.metadata.get("priority") == "high" and 
                not result.passed):
                logger.warning(f"High priority test failed: {test_case.input_text}")
        
        # Generate report
        report = self._generate_integration_report(
            category, results, time.time() - start_time
        )
        
        logger.info(
            f"Integration test complete for {category}: "
            f"{report.success_rate:.1f}% success rate ({report.passed_count}/{report.test_count})"
        )
        
        return report
    
    def test_time_range_patterns(self) -> PatternIntegrationReport:
        """Specialized test for time range patterns."""
        return self.test_pattern_category("time_range")
    
    def test_all_categories(self) -> Dict[str, PatternIntegrationReport]:
        """Test all pattern categories available."""
        all_categories = list(self.framework.registry.test_cases.keys())
        reports = {}
        
        for category in all_categories:
            reports[category] = self.test_pattern_category(category)
        
        return reports
    
    def generate_comprehensive_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report for all patterns."""
        logger.info("Generating comprehensive validation report...")
        
        # Test all categories
        category_reports = self.test_all_categories()
        
        # Calculate overall metrics
        total_tests = sum(report.test_count for report in category_reports.values())
        total_passed = sum(report.passed_count for report in category_reports.values())
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        # Find critical failures (high priority failed tests)
        critical_failures = []
        for report in category_reports.values():
            for result in report.detailed_results:
                if (not result.passed and 
                    result.test_case.metadata and 
                    result.test_case.metadata.get("priority") == "high"):
                    critical_failures.append({
                        "category": report.pattern_category,
                        "input": result.test_case.input_text,
                        "expected": result.expected_output,
                        "actual": result.formatter_output,
                        "error": result.error_message
                    })
        
        # Analyze performance patterns
        performance_issues = []
        for report in category_reports.values():
            if report.avg_execution_time_ms > 50.0:  # 50ms threshold
                performance_issues.append({
                    "category": report.pattern_category,
                    "avg_time_ms": report.avg_execution_time_ms
                })
        
        # Generate recommendations
        recommendations = self._generate_system_recommendations(
            category_reports, critical_failures, performance_issues
        )
        
        comprehensive_report = {
            "summary": {
                "timestamp": time.time(),
                "total_categories": len(category_reports),
                "total_tests": total_tests,
                "total_passed": total_passed,
                "total_failed": total_tests - total_passed,
                "overall_success_rate": overall_success_rate,
                "critical_failures_count": len(critical_failures),
                "performance_issues_count": len(performance_issues)
            },
            "category_breakdown": {
                category: {
                    "success_rate": report.success_rate,
                    "test_count": report.test_count,
                    "passed_count": report.passed_count,
                    "avg_time_ms": report.avg_execution_time_ms
                }
                for category, report in category_reports.items()
            },
            "critical_failures": critical_failures,
            "performance_issues": performance_issues,
            "recommendations": recommendations,
            "detailed_reports": {
                category: asdict(report) for category, report in category_reports.items()
            }
        }
        
        # Special focus on target case
        target_case_status = self._analyze_target_case(category_reports)
        if target_case_status:
            comprehensive_report["target_case_analysis"] = target_case_status
        
        return comprehensive_report
    
    def _run_integration_test(self, test_case: PatternTestCase) -> IntegrationTestResult:
        """Run a single test case through the full formatter pipeline."""
        start_time = time.perf_counter()
        
        try:
            # Use the complete text formatting pipeline
            formatter_output = format_transcription(test_case.input_text)
            execution_time = (time.perf_counter() - start_time) * 1000
            
            # Check exact match
            passed = formatter_output.strip() == test_case.expected_output.strip()
            
            # Check partial match if exact match failed
            partial_match = False
            confidence_score = 0.0
            
            if not passed:
                # Calculate similarity for partial match detection
                confidence_score = self._calculate_similarity(
                    formatter_output, test_case.expected_output
                )
                partial_match = confidence_score > 0.7  # 70% similarity threshold
            else:
                confidence_score = 1.0
            
            return IntegrationTestResult(
                test_case=test_case,
                formatter_output=formatter_output,
                expected_output=test_case.expected_output,
                passed=passed,
                execution_time_ms=execution_time,
                partial_match=partial_match,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Integration test failed for '{test_case.input_text}': {e}")
            
            return IntegrationTestResult(
                test_case=test_case,
                formatter_output="",
                expected_output=test_case.expected_output,
                passed=False,
                execution_time_ms=execution_time,
                error_message=str(e),
                confidence_score=0.0
            )
    
    def _generate_integration_report(
        self, 
        category: str, 
        results: List[IntegrationTestResult], 
        total_time: float
    ) -> PatternIntegrationReport:
        """Generate integration report from test results."""
        passed_count = sum(1 for r in results if r.passed)
        failed_count = len(results) - passed_count
        partial_count = sum(1 for r in results if r.partial_match and not r.passed)
        
        success_rate = (passed_count / len(results) * 100) if results else 0
        avg_execution_time = sum(r.execution_time_ms for r in results) / len(results) if results else 0
        
        # Generate recommendations
        recommendations = []
        
        # Identify high-priority failures
        high_priority_failures = [
            r for r in results 
            if not r.passed and r.test_case.metadata and r.test_case.metadata.get("priority") == "high"
        ]
        if high_priority_failures:
            recommendations.append(
                f"Fix {len(high_priority_failures)} high-priority pattern failures"
            )
        
        # Performance recommendations
        slow_tests = [r for r in results if r.execution_time_ms > 100]
        if slow_tests:
            recommendations.append(
                f"Optimize {len(slow_tests)} slow pattern tests (>100ms)"
            )
        
        # Partial match recommendations
        if partial_count > 0:
            recommendations.append(
                f"Review {partial_count} tests with partial matches - may need pattern refinement"
            )
        
        return PatternIntegrationReport(
            pattern_category=category,
            test_count=len(results),
            passed_count=passed_count,
            failed_count=failed_count,
            partial_count=partial_count,
            success_rate=success_rate,
            avg_execution_time_ms=avg_execution_time,
            detailed_results=results,
            recommendations=recommendations,
            timestamp=time.time()
        )
    
    def _empty_report(self, category: str) -> PatternIntegrationReport:
        """Create empty report for categories with no test cases."""
        return PatternIntegrationReport(
            pattern_category=category,
            test_count=0,
            passed_count=0,
            failed_count=0,
            partial_count=0,
            success_rate=0.0,
            avg_execution_time_ms=0.0,
            detailed_results=[],
            recommendations=["No test cases available for this category"],
            timestamp=time.time()
        )
    
    def _calculate_similarity(self, actual: str, expected: str) -> float:
        """Calculate similarity score between actual and expected output."""
        # Simple character-based similarity
        if not actual and not expected:
            return 1.0
        if not actual or not expected:
            return 0.0
        
        # Normalize strings
        actual_norm = actual.lower().strip()
        expected_norm = expected.lower().strip()
        
        if actual_norm == expected_norm:
            return 1.0
        
        # Calculate Levenshtein distance-based similarity
        import difflib
        similarity = difflib.SequenceMatcher(None, actual_norm, expected_norm).ratio()
        return similarity
    
    def _generate_system_recommendations(
        self, 
        category_reports: Dict[str, PatternIntegrationReport],
        critical_failures: List[Dict[str, Any]],
        performance_issues: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate system-wide recommendations."""
        recommendations = []
        
        # Critical failure analysis
        if critical_failures:
            recommendations.append(
                f"CRITICAL: {len(critical_failures)} high-priority patterns are failing. "
                "These should be addressed immediately."
            )
            
            # Group by category
            failure_by_category = {}
            for failure in critical_failures:
                category = failure["category"]
                if category not in failure_by_category:
                    failure_by_category[category] = 0
                failure_by_category[category] += 1
            
            for category, count in failure_by_category.items():
                recommendations.append(
                    f"Focus on {category} patterns: {count} critical failures"
                )
        
        # Performance analysis
        if performance_issues:
            recommendations.append(
                f"Performance concerns in {len(performance_issues)} pattern categories. "
                "Consider pattern optimization."
            )
        
        # Success rate analysis
        low_success_categories = [
            (name, report.success_rate) 
            for name, report in category_reports.items() 
            if report.success_rate < 80.0 and report.test_count > 0
        ]
        if low_success_categories:
            recommendations.append(
                f"Pattern categories with <80% success rate: "
                f"{', '.join(f'{name} ({rate:.1f}%)' for name, rate in low_success_categories)}"
            )
        
        return recommendations
    
    def _analyze_target_case(self, category_reports: Dict[str, PatternIntegrationReport]) -> Optional[Dict[str, Any]]:
        """Analyze the specific target case: 'from nine to five' → 'From 9 to 5'."""
        target_input = "from nine to five"
        target_expected = "From 9 to 5"
        
        # Search for the target case in all reports
        for category, report in category_reports.items():
            for result in report.detailed_results:
                if result.test_case.input_text.lower() == target_input.lower():
                    return {
                        "found": True,
                        "category": category,
                        "input": result.test_case.input_text,
                        "expected": target_expected,
                        "actual": result.formatter_output,
                        "passed": result.passed,
                        "confidence": result.confidence_score,
                        "execution_time_ms": result.execution_time_ms,
                        "error": result.error_message,
                        "status": "PASS" if result.passed else "FAIL"
                    }
        
        return {
            "found": False,
            "status": "NOT_FOUND",
            "message": "Target case 'from nine to five' not found in test suite"
        }
    
    def save_report(self, report: Dict[str, Any], output_path: str):
        """Save comprehensive report to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Integration test report saved to {output_file}")
    
    def print_summary_report(self, report: Dict[str, Any]):
        """Print a human-readable summary of the validation report."""
        print("=" * 70)
        print("PATTERN INTEGRATION VALIDATION REPORT")
        print("=" * 70)
        
        summary = report["summary"]
        print(f"\nOVERALL SUMMARY:")
        print(f"  Total test categories: {summary['total_categories']}")
        print(f"  Total tests run: {summary['total_tests']}")
        print(f"  Tests passed: {summary['total_passed']}")
        print(f"  Tests failed: {summary['total_failed']}")
        print(f"  Success rate: {summary['overall_success_rate']:.1f}%")
        print(f"  Critical failures: {summary['critical_failures_count']}")
        print(f"  Performance issues: {summary['performance_issues_count']}")
        
        # Target case analysis
        if "target_case_analysis" in report:
            target = report["target_case_analysis"]
            print(f"\nTARGET CASE ANALYSIS:")
            print(f"  Case: '{target.get('input', 'from nine to five')}' → '{target.get('expected', 'From 9 to 5')}'")
            print(f"  Status: {target.get('status', 'UNKNOWN')}")
            if target.get('found'):
                print(f"  Actual output: '{target.get('actual', 'N/A')}'")
                print(f"  Confidence: {target.get('confidence', 0):.2f}")
            
        # Category breakdown
        print(f"\nCATEGORY BREAKDOWN:")
        category_breakdown = report["category_breakdown"]
        for category, metrics in category_breakdown.items():
            print(f"  {category}:")
            print(f"    Success rate: {metrics['success_rate']:.1f}%")
            print(f"    Tests: {metrics['passed_count']}/{metrics['test_count']}")
            print(f"    Avg time: {metrics['avg_time_ms']:.1f}ms")
        
        # Critical failures
        if report["critical_failures"]:
            print(f"\nCRITICAL FAILURES:")
            for i, failure in enumerate(report["critical_failures"][:5], 1):  # Show first 5
                print(f"  {i}. [{failure['category']}] '{failure['input']}'")
                print(f"     Expected: '{failure['expected']}'")
                print(f"     Actual: '{failure['actual']}'")
        
        # Recommendations
        if report["recommendations"]:
            print(f"\nRECOMMENDATIONS:")
            for i, rec in enumerate(report["recommendations"], 1):
                print(f"  {i}. {rec}")
        
        print("=" * 70)


def create_integration_tester(language: str = "en") -> PatternIntegrationTester:
    """Create and initialize pattern integration tester."""
    return PatternIntegrationTester(language)


def run_target_case_validation() -> Dict[str, Any]:
    """Run validation specifically for the target case."""
    tester = create_integration_tester()
    
    # Run time range pattern tests
    time_range_report = tester.test_time_range_patterns()
    
    # Generate comprehensive report
    full_report = tester.generate_comprehensive_validation_report()
    
    return full_report


if __name__ == "__main__":
    # Run the integration tester
    tester = create_integration_tester()
    report = tester.generate_comprehensive_validation_report()
    
    # Print summary
    tester.print_summary_report(report)
    
    # Save detailed report
    tester.save_report(report, "/tmp/pattern_integration_report.json")