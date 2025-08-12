#!/usr/bin/env python3
"""
Pattern Validation Framework for Cross-Language Testing

This module provides comprehensive validation and testing capabilities
for the universal pattern framework, ensuring patterns work consistently
across different languages.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Callable
from enum import Enum

from stt.core.config import setup_logging
from .universal_pattern_framework import (
    UniversalPattern, UniversalPatternFactory, PatternType, PatternPriority
)
from .multi_language_resource_manager import get_resource_manager

logger = setup_logging(__name__)


# ==============================================================================
# VALIDATION FRAMEWORK TYPES
# ==============================================================================

class ValidationLevel(Enum):
    """Validation strictness levels."""
    BASIC = "basic"         # Check if patterns compile and basic functionality
    STANDARD = "standard"   # Include test case validation
    COMPREHENSIVE = "comprehensive"  # Full linguistic and edge case testing
    STRESS = "stress"       # Performance and robustness testing


class TestCaseType(Enum):
    """Types of test cases for pattern validation."""
    POSITIVE = "positive"   # Should match
    NEGATIVE = "negative"   # Should not match
    EDGE_CASE = "edge_case" # Boundary conditions
    PERFORMANCE = "performance"  # Speed/efficiency tests
    LINGUISTIC = "linguistic"   # Language-specific correctness


@dataclass
class TestCase:
    """Individual test case for pattern validation."""
    input_text: str
    expected_match: bool
    test_type: TestCaseType
    description: str = ""
    expected_groups: Optional[List[str]] = None
    language_specific: bool = False
    performance_threshold_ms: Optional[float] = None


@dataclass
class ValidationResult:
    """Result of pattern validation."""
    pattern_name: str
    language: str
    test_level: ValidationLevel
    total_tests: int
    passed_tests: int
    failed_tests: int
    success_rate: float
    execution_time_ms: float
    failed_cases: List[TestCase] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class CrossLanguageReport:
    """Comprehensive report comparing patterns across languages."""
    pattern_type: str
    languages_tested: List[str]
    validation_results: Dict[str, ValidationResult] = field(default_factory=dict)
    consistency_score: float = 0.0
    common_failures: List[str] = field(default_factory=list)
    language_specific_issues: Dict[str, List[str]] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


# ==============================================================================
# TEST CASE GENERATORS
# ==============================================================================

class TestCaseGenerator:
    """Generates test cases for different pattern types and languages."""
    
    def __init__(self):
        self.resource_manager = get_resource_manager()
    
    def generate_mathematical_test_cases(self, language: str) -> List[TestCase]:
        """Generate test cases for mathematical patterns."""
        resources = self.resource_manager.get_resources(language)
        
        # Get language-specific mathematical terms
        math_ops = self._get_math_operations(resources, language)
        number_words = self._get_number_words(resources, language)
        
        test_cases = []
        
        # Basic mathematical expressions
        if math_ops and number_words:
            plus_op = math_ops.get("plus", "plus")
            minus_op = math_ops.get("minus", "minus")
            equals_op = math_ops.get("equals", "equals")
            
            test_cases.extend([
                TestCase(
                    f"x {plus_op} y {equals_op} z",
                    True,
                    TestCaseType.POSITIVE,
                    "Basic addition equation"
                ),
                TestCase(
                    f"a {minus_op} b",
                    True,
                    TestCaseType.POSITIVE,
                    "Simple subtraction"
                ),
                TestCase(
                    f"{number_words[0]} {plus_op} {number_words[1]}",
                    True,
                    TestCaseType.POSITIVE,
                    "Number words with operation"
                ),
                TestCase(
                    "random text without math",
                    False,
                    TestCaseType.NEGATIVE,
                    "Non-mathematical text"
                ),
                TestCase(
                    f"incomplete {plus_op}",
                    False,
                    TestCaseType.EDGE_CASE,
                    "Incomplete expression"
                )
            ])
        
        return test_cases
    
    def generate_currency_test_cases(self, language: str) -> List[TestCase]:
        """Generate test cases for currency patterns."""
        resources = self.resource_manager.get_resources(language)
        
        # Get currency information
        currency_units = self._get_currency_units(resources, language)
        number_words = self._get_number_words(resources, language)
        
        test_cases = []
        
        if currency_units and number_words:
            dollar_unit = self._find_currency_unit(currency_units, ["dollar", "dólar"])
            euro_unit = self._find_currency_unit(currency_units, ["euro"])
            
            if dollar_unit:
                test_cases.extend([
                    TestCase(
                        f"{number_words[0]} {dollar_unit}",
                        True,
                        TestCaseType.POSITIVE,
                        "Number word with currency"
                    ),
                    TestCase(
                        f"5 {dollar_unit}",
                        True,
                        TestCaseType.POSITIVE,
                        "Digit with currency"
                    )
                ])
            
            # Symbol-based currency
            test_cases.extend([
                TestCase("$10", True, TestCaseType.POSITIVE, "Dollar symbol"),
                TestCase("€20", True, TestCaseType.POSITIVE, "Euro symbol"),
                TestCase("invalid currency", False, TestCaseType.NEGATIVE, "Non-currency text"),
                TestCase("$", False, TestCaseType.EDGE_CASE, "Symbol without amount")
            ])
        
        return test_cases
    
    def generate_web_test_cases(self, language: str) -> List[TestCase]:
        """Generate test cases for web patterns."""
        resources = self.resource_manager.get_resources(language)
        
        # Get web-related keywords
        url_keywords = self._get_url_keywords(resources, language)
        
        test_cases = []
        
        if url_keywords:
            dot_word = url_keywords.get("dot", "dot")
            at_word = url_keywords.get("at", "at")
            
            test_cases.extend([
                TestCase(
                    f"user {at_word} example {dot_word} com",
                    True,
                    TestCaseType.POSITIVE,
                    "Email with spoken keywords"
                ),
                TestCase(
                    f"visit github {dot_word} com",
                    True,
                    TestCaseType.POSITIVE,
                    "URL with spoken keywords"
                ),
                TestCase(
                    "regular text",
                    False,
                    TestCaseType.NEGATIVE,
                    "Non-web text"
                ),
                TestCase(
                    f"incomplete {at_word}",
                    False,
                    TestCaseType.EDGE_CASE,
                    "Incomplete web address"
                )
            ])
        
        return test_cases
    
    def _get_math_operations(self, resources: Dict[str, Any], language: str) -> Dict[str, str]:
        """Extract mathematical operations from resources."""
        try:
            spoken_keywords = resources.get("spoken_keywords", {})
            mathematical = spoken_keywords.get("mathematical", {})
            return mathematical.get("operations", {})
        except:
            return {}
    
    def _get_number_words(self, resources: Dict[str, Any], language: str) -> List[str]:
        """Extract number words from resources."""
        try:
            number_words = resources.get("number_words", {})
            ones = number_words.get("ones", {})
            return list(ones.keys())[:5]  # Get first 5 number words
        except:
            return []
    
    def _get_currency_units(self, resources: Dict[str, Any], language: str) -> List[str]:
        """Extract currency units from resources."""
        try:
            # Try units.currency_map first
            units = resources.get("units", {})
            currency_map = units.get("currency_map", {})
            if currency_map:
                return list(currency_map.keys())
            
            # Fallback to currency.units
            currency = resources.get("currency", {})
            return currency.get("units", [])
        except:
            return []
    
    def _get_url_keywords(self, resources: Dict[str, Any], language: str) -> Dict[str, str]:
        """Extract URL keywords from resources."""
        try:
            spoken_keywords = resources.get("spoken_keywords", {})
            return spoken_keywords.get("url", {})
        except:
            return {}
    
    def _find_currency_unit(self, currency_units: List[str], search_terms: List[str]) -> Optional[str]:
        """Find a currency unit that matches search terms."""
        for unit in currency_units:
            for term in search_terms:
                if term.lower() in unit.lower():
                    return unit
        return None


# ==============================================================================
# PATTERN VALIDATORS
# ==============================================================================

class PatternValidator:
    """Validates individual patterns against test cases."""
    
    def __init__(self):
        self.test_generator = TestCaseGenerator()
    
    def validate_pattern(
        self, 
        pattern: UniversalPattern, 
        test_level: ValidationLevel = ValidationLevel.STANDARD,
        custom_test_cases: Optional[List[TestCase]] = None
    ) -> ValidationResult:
        """Validate a single pattern."""
        start_time = time.time()
        
        # Generate or use provided test cases
        if custom_test_cases:
            test_cases = custom_test_cases
        else:
            test_cases = self._generate_test_cases_for_pattern(pattern, test_level)
        
        # Run validation
        validation_result = self._run_validation_tests(pattern, test_cases, test_level)
        
        # Calculate execution time
        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        validation_result.execution_time_ms = execution_time
        
        return validation_result
    
    def _generate_test_cases_for_pattern(
        self, 
        pattern: UniversalPattern, 
        test_level: ValidationLevel
    ) -> List[TestCase]:
        """Generate appropriate test cases for a pattern."""
        pattern_type = pattern.metadata.pattern_type
        language = pattern.language
        
        if pattern_type == PatternType.MATHEMATICAL:
            test_cases = self.test_generator.generate_mathematical_test_cases(language)
        elif pattern_type == PatternType.CURRENCY:
            test_cases = self.test_generator.generate_currency_test_cases(language)
        elif pattern_type == PatternType.WEB:
            test_cases = self.test_generator.generate_web_test_cases(language)
        else:
            # Generic test cases
            test_cases = [
                TestCase(
                    "generic test input",
                    False,
                    TestCaseType.POSITIVE,
                    "Generic test case"
                )
            ]
        
        # Filter test cases based on validation level
        if test_level == ValidationLevel.BASIC:
            test_cases = [tc for tc in test_cases if tc.test_type == TestCaseType.POSITIVE][:3]
        elif test_level == ValidationLevel.STANDARD:
            test_cases = [tc for tc in test_cases if tc.test_type in [TestCaseType.POSITIVE, TestCaseType.NEGATIVE]]
        # COMPREHENSIVE and STRESS include all test cases
        
        return test_cases
    
    def _run_validation_tests(
        self, 
        pattern: UniversalPattern, 
        test_cases: List[TestCase], 
        test_level: ValidationLevel
    ) -> ValidationResult:
        """Run the actual validation tests."""
        result = ValidationResult(
            pattern_name=pattern.metadata.name,
            language=pattern.language,
            test_level=test_level,
            total_tests=len(test_cases),
            passed_tests=0,
            failed_tests=0,
            success_rate=0.0,
            execution_time_ms=0.0
        )
        
        try:
            compiled_pattern = pattern.get_pattern()
        except Exception as e:
            result.errors.append(f"Failed to compile pattern: {e}")
            result.failed_tests = len(test_cases)
            return result
        
        # Run each test case
        performance_times = []
        
        for test_case in test_cases:
            try:
                # Measure performance for each test
                test_start = time.time()
                match = compiled_pattern.search(test_case.input_text)
                test_time = (time.time() - test_start) * 1000
                performance_times.append(test_time)
                
                # Check if result matches expectation
                if (match is not None) == test_case.expected_match:
                    result.passed_tests += 1
                    
                    # Check performance threshold if specified
                    if (test_case.performance_threshold_ms and 
                        test_time > test_case.performance_threshold_ms):
                        result.warnings.append(
                            f"Test '{test_case.description}' exceeded performance threshold: "
                            f"{test_time:.2f}ms > {test_case.performance_threshold_ms}ms"
                        )
                else:
                    result.failed_tests += 1
                    result.failed_cases.append(test_case)
                    
                    logger.debug(
                        f"Test failed: '{test_case.input_text}' - "
                        f"Expected match: {test_case.expected_match}, Got match: {match is not None}"
                    )
                
            except Exception as e:
                result.failed_tests += 1
                result.failed_cases.append(test_case)
                result.errors.append(f"Test '{test_case.description}' raised exception: {e}")
        
        # Calculate metrics
        result.success_rate = result.passed_tests / result.total_tests if result.total_tests > 0 else 0.0
        
        if performance_times:
            result.performance_metrics = {
                "avg_time_ms": sum(performance_times) / len(performance_times),
                "max_time_ms": max(performance_times),
                "min_time_ms": min(performance_times),
                "total_time_ms": sum(performance_times)
            }
        
        return result


# ==============================================================================
# CROSS-LANGUAGE VALIDATOR
# ==============================================================================

class CrossLanguageValidator:
    """Validates patterns across multiple languages for consistency."""
    
    def __init__(self):
        self.pattern_validator = PatternValidator()
        self.resource_manager = get_resource_manager()
    
    def validate_pattern_across_languages(
        self, 
        pattern_name: str, 
        languages: List[str],
        test_level: ValidationLevel = ValidationLevel.STANDARD
    ) -> CrossLanguageReport:
        """Validate a pattern type across multiple languages."""
        report = CrossLanguageReport(
            pattern_type=pattern_name,
            languages_tested=languages
        )
        
        validation_results = {}
        success_rates = []
        
        # Validate pattern for each language
        for language in languages:
            try:
                # Create pattern for this language
                pattern = UniversalPatternFactory.create_pattern(pattern_name, language)
                
                # Validate the pattern
                validation_result = self.pattern_validator.validate_pattern(pattern, test_level)
                validation_results[language] = validation_result
                success_rates.append(validation_result.success_rate)
                
            except Exception as e:
                # Create error result for failed language
                error_result = ValidationResult(
                    pattern_name=pattern_name,
                    language=language,
                    test_level=test_level,
                    total_tests=0,
                    passed_tests=0,
                    failed_tests=1,
                    success_rate=0.0,
                    execution_time_ms=0.0,
                    errors=[f"Failed to create/validate pattern: {e}"]
                )
                validation_results[language] = error_result
                success_rates.append(0.0)
        
        report.validation_results = validation_results
        
        # Calculate consistency score
        if success_rates:
            # Consistency is measured by how close success rates are to each other
            avg_success = sum(success_rates) / len(success_rates)
            variance = sum((rate - avg_success) ** 2 for rate in success_rates) / len(success_rates)
            # Convert variance to consistency score (0 to 1, where 1 is perfectly consistent)
            report.consistency_score = max(0.0, 1.0 - variance)
        
        # Analyze common failures and language-specific issues
        report.common_failures = self._find_common_failures(validation_results)
        report.language_specific_issues = self._find_language_specific_issues(validation_results)
        report.recommendations = self._generate_recommendations(report)
        
        return report
    
    def validate_all_patterns_for_language(
        self, 
        language: str,
        test_level: ValidationLevel = ValidationLevel.STANDARD
    ) -> Dict[str, ValidationResult]:
        """Validate all available patterns for a specific language."""
        available_patterns = UniversalPatternFactory.get_available_patterns()
        results = {}
        
        for pattern_name in available_patterns:
            try:
                pattern = UniversalPatternFactory.create_pattern(pattern_name, language)
                results[pattern_name] = self.pattern_validator.validate_pattern(pattern, test_level)
            except Exception as e:
                logger.error(f"Failed to validate pattern {pattern_name} for {language}: {e}")
                results[pattern_name] = ValidationResult(
                    pattern_name=pattern_name,
                    language=language,
                    test_level=test_level,
                    total_tests=0,
                    passed_tests=0,
                    failed_tests=1,
                    success_rate=0.0,
                    execution_time_ms=0.0,
                    errors=[str(e)]
                )
        
        return results
    
    def generate_comprehensive_report(
        self, 
        languages: List[str],
        test_level: ValidationLevel = ValidationLevel.COMPREHENSIVE
    ) -> Dict[str, Any]:
        """Generate a comprehensive validation report for multiple languages."""
        available_patterns = UniversalPatternFactory.get_available_patterns()
        
        report = {
            "summary": {
                "languages_tested": languages,
                "patterns_tested": available_patterns,
                "test_level": test_level.value,
                "overall_success_rate": 0.0,
                "language_rankings": {},
                "pattern_rankings": {}
            },
            "pattern_reports": {},
            "language_summaries": {},
            "recommendations": []
        }
        
        all_success_rates = []
        language_scores = {lang: [] for lang in languages}
        pattern_scores = {pattern: [] for pattern in available_patterns}
        
        # Validate each pattern across all languages
        for pattern_name in available_patterns:
            pattern_report = self.validate_pattern_across_languages(
                pattern_name, languages, test_level
            )
            report["pattern_reports"][pattern_name] = pattern_report
            
            # Collect scores for rankings
            for language, validation_result in pattern_report.validation_results.items():
                success_rate = validation_result.success_rate
                all_success_rates.append(success_rate)
                language_scores[language].append(success_rate)
                pattern_scores[pattern_name].append(success_rate)
        
        # Calculate overall success rate
        if all_success_rates:
            report["summary"]["overall_success_rate"] = sum(all_success_rates) / len(all_success_rates)
        
        # Calculate language rankings
        for language, scores in language_scores.items():
            avg_score = sum(scores) / len(scores) if scores else 0.0
            report["summary"]["language_rankings"][language] = avg_score
        
        # Calculate pattern rankings
        for pattern, scores in pattern_scores.items():
            avg_score = sum(scores) / len(scores) if scores else 0.0
            report["summary"]["pattern_rankings"][pattern] = avg_score
        
        # Generate language summaries
        for language in languages:
            language_results = self.validate_all_patterns_for_language(language, test_level)
            report["language_summaries"][language] = self._summarize_language_results(language_results)
        
        # Generate overall recommendations
        report["recommendations"] = self._generate_overall_recommendations(report)
        
        return report
    
    def _find_common_failures(self, validation_results: Dict[str, ValidationResult]) -> List[str]:
        """Find test cases that failed across multiple languages."""
        failure_counts = {}
        
        for result in validation_results.values():
            for failed_case in result.failed_cases:
                test_desc = failed_case.description
                failure_counts[test_desc] = failure_counts.get(test_desc, 0) + 1
        
        # Return failures that occurred in more than half the languages
        threshold = len(validation_results) // 2 + 1
        return [desc for desc, count in failure_counts.items() if count >= threshold]
    
    def _find_language_specific_issues(self, validation_results: Dict[str, ValidationResult]) -> Dict[str, List[str]]:
        """Find issues specific to individual languages."""
        language_issues = {}
        
        for language, result in validation_results.items():
            issues = []
            
            # Add errors
            issues.extend(result.errors)
            
            # Add warnings
            issues.extend(result.warnings)
            
            # Add performance issues
            if result.performance_metrics:
                avg_time = result.performance_metrics.get("avg_time_ms", 0)
                if avg_time > 10.0:  # Threshold for slow performance
                    issues.append(f"Slow performance: {avg_time:.2f}ms average")
            
            # Add unique failed cases (not in common failures)
            for failed_case in result.failed_cases:
                issues.append(f"Failed: {failed_case.description}")
            
            if issues:
                language_issues[language] = issues
        
        return language_issues
    
    def _generate_recommendations(self, report: CrossLanguageReport) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Low consistency score
        if report.consistency_score < 0.7:
            recommendations.append(
                f"Pattern consistency is low ({report.consistency_score:.2f}). "
                "Review resource definitions across languages."
            )
        
        # Common failures
        if report.common_failures:
            recommendations.append(
                f"Address common failures: {', '.join(report.common_failures)}"
            )
        
        # Language-specific issues
        for language, issues in report.language_specific_issues.items():
            if len(issues) > 2:
                recommendations.append(
                    f"Review {language} language support - multiple issues detected"
                )
        
        return recommendations
    
    def _summarize_language_results(self, language_results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """Summarize validation results for a single language."""
        total_tests = sum(result.total_tests for result in language_results.values())
        total_passed = sum(result.passed_tests for result in language_results.values())
        total_failed = sum(result.failed_tests for result in language_results.values())
        
        avg_success_rate = sum(result.success_rate for result in language_results.values()) / len(language_results)
        
        return {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "average_success_rate": avg_success_rate,
            "pattern_count": len(language_results),
            "perfect_patterns": [
                name for name, result in language_results.items() 
                if result.success_rate == 1.0
            ],
            "problematic_patterns": [
                name for name, result in language_results.items() 
                if result.success_rate < 0.5
            ]
        }
    
    def _generate_overall_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate overall recommendations for the entire framework."""
        recommendations = []
        
        overall_success = report["summary"]["overall_success_rate"]
        
        if overall_success < 0.8:
            recommendations.append(
                f"Overall success rate is {overall_success:.1%}. "
                "Consider improving resource quality and pattern definitions."
            )
        
        # Identify worst-performing languages
        language_rankings = report["summary"]["language_rankings"]
        worst_languages = sorted(language_rankings.items(), key=lambda x: x[1])[:2]
        
        for lang, score in worst_languages:
            if score < 0.6:
                recommendations.append(
                    f"Language '{lang}' has low success rate ({score:.1%}). "
                    "Review resource completeness and accuracy."
                )
        
        # Identify worst-performing patterns
        pattern_rankings = report["summary"]["pattern_rankings"]
        worst_patterns = sorted(pattern_rankings.items(), key=lambda x: x[1])[:2]
        
        for pattern, score in worst_patterns:
            if score < 0.6:
                recommendations.append(
                    f"Pattern '{pattern}' has low success rate ({score:.1%}). "
                    "Review pattern logic and test cases."
                )
        
        return recommendations


# ==============================================================================
# TESTING UTILITIES
# ==============================================================================

def quick_validation_test(language: str = "en") -> Dict[str, Any]:
    """Run a quick validation test for demonstration purposes."""
    validator = CrossLanguageValidator()
    
    # Test mathematical pattern
    try:
        pattern = UniversalPatternFactory.create_pattern("mathematical", language)
        result = validator.pattern_validator.validate_pattern(pattern, ValidationLevel.BASIC)
        
        return {
            "language": language,
            "pattern": "mathematical",
            "success": result.success_rate > 0.5,
            "success_rate": result.success_rate,
            "test_count": result.total_tests,
            "errors": result.errors
        }
    except Exception as e:
        return {
            "language": language,
            "pattern": "mathematical",
            "success": False,
            "error": str(e)
        }


def compare_languages_test(languages: List[str] = ["en", "es"]) -> Dict[str, Any]:
    """Compare pattern performance across languages."""
    validator = CrossLanguageValidator()
    
    try:
        # Test mathematical pattern across languages
        report = validator.validate_pattern_across_languages(
            "mathematical", languages, ValidationLevel.STANDARD
        )
        
        return {
            "languages": languages,
            "consistency_score": report.consistency_score,
            "success_rates": {
                lang: result.success_rate 
                for lang, result in report.validation_results.items()
            },
            "common_failures": report.common_failures,
            "recommendations": report.recommendations
        }
    except Exception as e:
        return {
            "languages": languages,
            "error": str(e)
        }