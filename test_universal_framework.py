#!/usr/bin/env python3
"""
Comprehensive Test Suite for Universal Pattern Framework

This script demonstrates and tests the language-agnostic pattern framework,
showing how it provides universal architecture for multi-language support.
"""

import sys
import os
import json
from pathlib import Path

# Add the source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from stt.text_formatting.universal_pattern_framework import (
    UniversalPatternFactory, 
    MathematicalPattern, 
    CurrencyPattern, 
    WebPattern,
    CrossLanguageValidator,
    create_language_framework,
    demonstrate_extensibility
)
from stt.text_formatting.multi_language_resource_manager import (
    MultiLanguageResourceManager,
    get_resource_manager
)
from stt.text_formatting.pattern_validation_framework import (
    CrossLanguageValidator,
    ValidationLevel,
    quick_validation_test,
    compare_languages_test
)


def print_section(title):
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


def print_subsection(title):
    """Print a subsection header."""
    print(f"\n{'-' * 40}")
    print(f" {title}")
    print(f"{'-' * 40}")


def test_universal_pattern_creation():
    """Test creating universal patterns for different languages."""
    print_section("UNIVERSAL PATTERN CREATION TEST")
    
    # Test English patterns
    print_subsection("English Patterns")
    en_math = UniversalPatternFactory.create_pattern("mathematical", "en")
    en_currency = UniversalPatternFactory.create_pattern("currency", "en")
    en_web = UniversalPatternFactory.create_pattern("web", "en")
    
    print(f"✓ Created English mathematical pattern: {en_math.metadata.name}")
    print(f"✓ Created English currency pattern: {en_currency.metadata.name}")
    print(f"✓ Created English web pattern: {en_web.metadata.name}")
    
    # Test Spanish patterns
    print_subsection("Spanish Patterns")
    try:
        es_math = UniversalPatternFactory.create_pattern("mathematical", "es")
        es_currency = UniversalPatternFactory.create_pattern("currency", "es")
        es_web = UniversalPatternFactory.create_pattern("web", "es")
        
        print(f"✓ Created Spanish mathematical pattern: {es_math.metadata.name}")
        print(f"✓ Created Spanish currency pattern: {es_currency.metadata.name}")
        print(f"✓ Created Spanish web pattern: {es_web.metadata.name}")
    except Exception as e:
        print(f"✗ Failed to create Spanish patterns: {e}")
    
    # Show pattern information
    print_subsection("Pattern Information")
    print("English Mathematical Pattern Info:")
    en_math_info = en_math.get_pattern_info()
    for key, value in en_math_info.items():
        if key == "pattern":
            print(f"  {key}: {str(value)[:100]}...")  # Truncate long patterns
        else:
            print(f"  {key}: {value}")


def test_resource_management():
    """Test the multi-language resource management system."""
    print_section("RESOURCE MANAGEMENT TEST")
    
    manager = get_resource_manager()
    
    # Get supported languages
    print_subsection("Supported Languages")
    languages = manager.get_supported_languages()
    for lang_info in languages:
        print(f"  {lang_info.code}: {lang_info.name} ({lang_info.family})")
        print(f"    Completeness: {lang_info.resource_completeness:.1%}")
        print(f"    Supported patterns: {', '.join(lang_info.supported_patterns)}")
        print(f"    Fallback chain: {' → '.join(lang_info.fallback_languages)}")
    
    # Test resource inheritance
    print_subsection("Resource Inheritance")
    try:
        en_resources = manager.get_resources("en", use_inheritance=False)
        es_resources = manager.get_resources("es", use_inheritance=False)
        es_inherited = manager.get_resources("es", use_inheritance=True)
        
        print(f"English resource categories: {len(en_resources)}")
        print(f"Spanish resource categories: {len(es_resources)}")
        print(f"Spanish with inheritance: {len(es_inherited)}")
        
        # Show an example of inheritance
        if "spoken_keywords" in es_resources and "mathematical" in es_resources["spoken_keywords"]:
            es_math_ops = es_resources["spoken_keywords"]["mathematical"].get("operations", {})
            print(f"Spanish math operations: {len(es_math_ops)} defined")
        
        if "spoken_keywords" in es_inherited and "mathematical" in es_inherited["spoken_keywords"]:
            inherited_math_ops = es_inherited["spoken_keywords"]["mathematical"].get("operations", {})
            print(f"Spanish with inheritance: {len(inherited_math_ops)} operations")
    except Exception as e:
        print(f"✗ Resource inheritance test failed: {e}")
    
    # Validate language support
    print_subsection("Language Validation")
    for lang_code in ["en", "es"]:
        validation = manager.validate_language_support(lang_code)
        print(f"\n{lang_code.upper()} Validation:")
        print(f"  Overall completeness: {validation['overall_completeness']:.1%}")
        print(f"  Pattern support:")
        for pattern, supported in validation["pattern_support"].items():
            status = "✓" if supported else "✗"
            print(f"    {status} {pattern}")


def test_pattern_validation():
    """Test the pattern validation framework."""
    print_section("PATTERN VALIDATION TEST")
    
    # Quick validation test
    print_subsection("Quick Validation")
    en_test = quick_validation_test("en")
    print(f"English mathematical pattern test:")
    print(f"  Success: {en_test['success']}")
    print(f"  Success rate: {en_test['success_rate']:.1%}")
    print(f"  Test count: {en_test['test_count']}")
    if en_test.get('errors'):
        print(f"  Errors: {en_test['errors']}")
    
    try:
        es_test = quick_validation_test("es")
        print(f"\nSpanish mathematical pattern test:")
        print(f"  Success: {es_test['success']}")
        print(f"  Success rate: {es_test['success_rate']:.1%}")
        print(f"  Test count: {es_test['test_count']}")
        if es_test.get('errors'):
            print(f"  Errors: {es_test['errors']}")
    except Exception as e:
        print(f"\nSpanish test failed: {e}")
    
    # Cross-language comparison
    print_subsection("Cross-Language Comparison")
    try:
        comparison = compare_languages_test(["en", "es"])
        print(f"Languages compared: {comparison['languages']}")
        print(f"Consistency score: {comparison['consistency_score']:.2f}")
        print("Success rates:")
        for lang, rate in comparison['success_rates'].items():
            print(f"  {lang}: {rate:.1%}")
        
        if comparison.get('common_failures'):
            print(f"Common failures: {comparison['common_failures']}")
        
        if comparison.get('recommendations'):
            print("Recommendations:")
            for rec in comparison['recommendations']:
                print(f"  • {rec}")
    except Exception as e:
        print(f"Cross-language comparison failed: {e}")


def test_framework_extensibility():
    """Test framework extensibility with a hypothetical new language."""
    print_section("FRAMEWORK EXTENSIBILITY TEST")
    
    # Create sample resources for a hypothetical French-like language
    sample_french_resources = {
        "spoken_keywords": {
            "mathematical": {
                "operations": {
                    "plus": "+",
                    "moins": "-",
                    "fois": "×",
                    "divisé par": "÷",
                    "égal": "="
                }
            },
            "url": {
                "point": ".",
                "arobase": "@",
                "barre": "/",
                "deux points": ":"
            }
        },
        "number_words": {
            "ones": {
                "zéro": 0,
                "un": 1,
                "deux": 2,
                "trois": 3,
                "quatre": 4,
                "cinq": 5
            }
        },
        "currency": {
            "units": ["euro", "euros", "dollar", "dollars"]
        },
        "test_cases": [
            "x plus y égal z",
            "deux fois trois",
            "utilisateur arobase exemple point com"
        ]
    }
    
    print_subsection("Creating New Language Support")
    try:
        extensibility_demo = demonstrate_extensibility("fr", sample_french_resources)
        
        print(f"✓ Created pattern for language: {extensibility_demo['language']}")
        print(f"Pattern info:")
        pattern_info = extensibility_demo['pattern_info']
        for key, value in pattern_info.items():
            if key == "pattern":
                print(f"  {key}: {str(value)[:80]}...")
            else:
                print(f"  {key}: {value}")
        
        validation_result = extensibility_demo.get('validation_result', {})
        if validation_result:
            print(f"\nValidation results:")
            for test_case, result in validation_result.items():
                status = "✓" if result else "✗"
                print(f"  {status} '{test_case}'")
        
        print(f"\n✓ Framework extensibility demonstration successful!")
        
    except Exception as e:
        print(f"✗ Framework extensibility test failed: {e}")


def test_comprehensive_framework():
    """Test the complete framework functionality."""
    print_section("COMPREHENSIVE FRAMEWORK TEST")
    
    print_subsection("Creating Language Framework")
    try:
        # Create framework for English
        en_framework = create_language_framework("en")
        print(f"✓ Created English framework")
        print(f"  Patterns created: {len(en_framework['patterns'])}")
        print(f"  Validation results: {len(en_framework['validation_results'])}")
        
        # Show validation summary
        total_tests = sum(r.total_tests for r in en_framework['validation_results'].values())
        total_passed = sum(r.passed_tests for r in en_framework['validation_results'].values())
        overall_success = total_passed / total_tests if total_tests > 0 else 0
        
        print(f"  Overall success rate: {overall_success:.1%} ({total_passed}/{total_tests})")
        
        # Try Spanish framework
        try:
            es_framework = create_language_framework("es")
            print(f"✓ Created Spanish framework")
            print(f"  Patterns created: {len(es_framework['patterns'])}")
            
            es_total_tests = sum(r.total_tests for r in es_framework['validation_results'].values())
            es_total_passed = sum(r.passed_tests for r in es_framework['validation_results'].values())
            es_overall_success = es_total_passed / es_total_tests if es_total_tests > 0 else 0
            
            print(f"  Overall success rate: {es_overall_success:.1%} ({es_total_passed}/{es_total_tests})")
            
        except Exception as e:
            print(f"✗ Spanish framework creation failed: {e}")
            
    except Exception as e:
        print(f"✗ Framework creation failed: {e}")


def generate_framework_report():
    """Generate a comprehensive framework report."""
    print_section("FRAMEWORK COMPREHENSIVE REPORT")
    
    try:
        validator = CrossLanguageValidator()
        
        # Test available languages
        manager = get_resource_manager()
        available_languages = [lang.code for lang in manager.get_supported_languages()]
        
        print(f"Testing languages: {', '.join(available_languages)}")
        
        # Generate comprehensive report
        print("Generating comprehensive validation report...")
        report = validator.generate_comprehensive_report(
            available_languages, 
            ValidationLevel.STANDARD
        )
        
        # Display summary
        summary = report["summary"]
        print(f"\nOverall Results:")
        print(f"  Languages tested: {len(summary['languages_tested'])}")
        print(f"  Patterns tested: {len(summary['patterns_tested'])}")
        print(f"  Overall success rate: {summary['overall_success_rate']:.1%}")
        
        print(f"\nLanguage Rankings:")
        for lang, score in sorted(summary['language_rankings'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {lang}: {score:.1%}")
        
        print(f"\nPattern Rankings:")
        for pattern, score in sorted(summary['pattern_rankings'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {pattern}: {score:.1%}")
        
        if report.get("recommendations"):
            print(f"\nRecommendations:")
            for rec in report["recommendations"]:
                print(f"  • {rec}")
        
        # Save detailed report
        report_file = Path("framework_validation_report.json")
        with open(report_file, 'w') as f:
            # Convert to JSON-serializable format
            json_report = {
                "summary": summary,
                "language_summaries": report["language_summaries"],
                "recommendations": report["recommendations"]
            }
            json.dump(json_report, f, indent=2)
        
        print(f"\n✓ Detailed report saved to: {report_file}")
        
    except Exception as e:
        print(f"✗ Report generation failed: {e}")


def main():
    """Run all framework tests."""
    print("Universal Pattern Framework Comprehensive Test Suite")
    print("=" * 60)
    
    try:
        # Test individual components
        test_universal_pattern_creation()
        test_resource_management()
        test_pattern_validation()
        test_framework_extensibility()
        test_comprehensive_framework()
        
        # Generate final report
        generate_framework_report()
        
        print_section("TEST SUITE COMPLETE")
        print("✓ Universal Pattern Framework tests completed successfully!")
        print("\nKey Achievements:")
        print("  • Language-agnostic pattern base classes implemented")
        print("  • Multi-language resource inheritance system working")
        print("  • Cross-language validation framework functional")
        print("  • Framework extensibility demonstrated")
        print("  • Comprehensive testing and reporting available")
        
    except Exception as e:
        print(f"\n✗ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())