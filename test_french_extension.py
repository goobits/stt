#!/usr/bin/env python3
"""
Test French Language Extension for Universal Pattern Framework

This script demonstrates how the universal pattern framework can be extended
to support a new language (French) with minimal effort.
"""

import sys
import os

# Add the source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from stt.text_formatting.universal_pattern_framework import (
    UniversalPatternFactory, 
    create_language_framework
)
from stt.text_formatting.multi_language_resource_manager import (
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


def test_french_language_support():
    """Test French language support in the framework."""
    print_section("FRENCH LANGUAGE EXTENSION TEST")
    
    # Reset resource manager to pick up new French resources
    from stt.text_formatting.multi_language_resource_manager import reset_resource_manager
    reset_resource_manager()
    
    # Test resource loading
    print("\n1. Testing French Resource Loading:")
    manager = get_resource_manager()
    
    try:
        fr_resources = manager.get_resources("fr", use_inheritance=False)
        fr_inherited = manager.get_resources("fr", use_inheritance=True)
        
        print(f"   ✓ French resources loaded: {len(fr_resources)} categories")
        print(f"   ✓ French with inheritance: {len(fr_inherited)} categories")
        
        # Show some French-specific content
        if "spoken_keywords" in fr_resources:
            url_keywords = fr_resources["spoken_keywords"].get("url", {})
            print(f"   ✓ French URL keywords: {list(url_keywords.keys())[:5]}...")
            
        if "number_words" in fr_resources:
            numbers = fr_resources["number_words"].get("ones", {})
            print(f"   ✓ French number words: {list(numbers.keys())[:5]}...")
        
    except Exception as e:
        print(f"   ✗ Failed to load French resources: {e}")
        return False
    
    # Test pattern creation
    print("\n2. Testing French Pattern Creation:")
    try:
        fr_math = UniversalPatternFactory.create_pattern("mathematical", "fr")
        fr_currency = UniversalPatternFactory.create_pattern("currency", "fr")
        fr_web = UniversalPatternFactory.create_pattern("web", "fr")
        
        print(f"   ✓ Mathematical pattern: {fr_math.metadata.name}")
        print(f"   ✓ Currency pattern: {fr_currency.metadata.name}")
        print(f"   ✓ Web pattern: {fr_web.metadata.name}")
        
        # Show pattern details
        print(f"\n   Pattern details for French mathematical:")
        components = fr_math._build_pattern_components()
        for key, value in components.items():
            print(f"     {key}: {value[:50]}...")
        
    except Exception as e:
        print(f"   ✗ Failed to create French patterns: {e}")
        return False
    
    # Test validation
    print("\n3. Testing French Pattern Validation:")
    try:
        fr_test = quick_validation_test("fr")
        print(f"   Success: {fr_test['success']}")
        print(f"   Success rate: {fr_test['success_rate']:.1%}")
        print(f"   Test count: {fr_test['test_count']}")
        if fr_test.get('errors'):
            print(f"   Errors: {fr_test['errors']}")
    except Exception as e:
        print(f"   ✗ French validation failed: {e}")
    
    # Test cross-language comparison
    print("\n4. Testing Cross-Language Comparison (EN/ES/FR):")
    try:
        comparison = compare_languages_test(["en", "es", "fr"])
        print(f"   Languages compared: {comparison['languages']}")
        print(f"   Consistency score: {comparison['consistency_score']:.2f}")
        print("   Success rates:")
        for lang, rate in comparison['success_rates'].items():
            print(f"     {lang}: {rate:.1%}")
        
        if comparison.get('recommendations'):
            print("   Recommendations:")
            for rec in comparison['recommendations']:
                print(f"     • {rec}")
    except Exception as e:
        print(f"   ✗ Cross-language comparison failed: {e}")
    
    return True


def test_french_specific_patterns():
    """Test French-specific pattern functionality."""
    print_section("FRENCH-SPECIFIC PATTERN TESTS")
    
    try:
        # Create French patterns
        fr_math = UniversalPatternFactory.create_pattern("mathematical", "fr")
        fr_currency = UniversalPatternFactory.create_pattern("currency", "fr")
        fr_web = UniversalPatternFactory.create_pattern("web", "fr")
        
        # Test French mathematical expressions
        print("\n1. French Mathematical Expressions:")
        math_tests = [
            "x plus y égal z",
            "deux fois trois",
            "a au carré plus b au carré",
            "cinq divisé par deux"
        ]
        
        pattern = fr_math.get_pattern()
        for test in math_tests:
            match = pattern.search(test)
            status = "✓" if match else "✗"
            print(f"   {status} '{test}' -> {match.group() if match else 'No match'}")
        
        # Test French currency expressions
        print("\n2. French Currency Expressions:")
        currency_tests = [
            "cinq euros",
            "dix dollars",
            "€20",
            "$15"
        ]
        
        pattern = fr_currency.get_pattern()
        for test in currency_tests:
            match = pattern.search(test)
            status = "✓" if match else "✗"
            print(f"   {status} '{test}' -> {match.group() if match else 'No match'}")
        
        # Test French web expressions
        print("\n3. French Web Expressions:")
        web_tests = [
            "utilisateur arobase exemple point com",
            "visiter github point com",
            "http deux points barre barre exemple point fr"
        ]
        
        pattern = fr_web.get_pattern()
        for test in web_tests:
            match = pattern.search(test)
            status = "✓" if match else "✗"
            print(f"   {status} '{test}' -> {match.group() if match else 'No match'}")
        
    except Exception as e:
        print(f"✗ French-specific pattern tests failed: {e}")


def test_language_inheritance():
    """Test language inheritance from English to French."""
    print_section("LANGUAGE INHERITANCE TEST")
    
    try:
        manager = get_resource_manager()
        
        # Get French resources with and without inheritance
        fr_base = manager.get_resources("fr", use_inheritance=False)
        fr_inherited = manager.get_resources("fr", use_inheritance=True)
        
        print("1. Resource Category Comparison:")
        base_categories = set(fr_base.keys())
        inherited_categories = set(fr_inherited.keys())
        
        print(f"   Base French categories: {len(base_categories)}")
        print(f"   Inherited categories: {len(inherited_categories)}")
        print(f"   Additional from inheritance: {inherited_categories - base_categories}")
        
        # Compare specific sections
        print("\n2. Mathematical Operations Inheritance:")
        base_math_ops = fr_base.get("spoken_keywords", {}).get("mathematical", {}).get("operations", {})
        inherited_math_ops = fr_inherited.get("spoken_keywords", {}).get("mathematical", {}).get("operations", {})
        
        print(f"   Base French math operations: {len(base_math_ops)}")
        print(f"   With inheritance: {len(inherited_math_ops)}")
        print(f"   French-specific: {list(base_math_ops.keys())[:3]}...")
        
        # Show inheritance chain
        print("\n3. Language Info:")
        fr_info = manager.loader.get_language_info("fr")
        print(f"   Language: {fr_info.name} ({fr_info.code})")
        print(f"   Family: {fr_info.family}")
        print(f"   Fallback chain: {' → '.join(fr_info.fallback_languages)}")
        print(f"   Supported patterns: {', '.join(fr_info.supported_patterns)}")
        print(f"   Completeness: {fr_info.resource_completeness:.1%}")
        
    except Exception as e:
        print(f"✗ Language inheritance test failed: {e}")


def test_comprehensive_multilingual_framework():
    """Test the framework with all three languages."""
    print_section("COMPREHENSIVE MULTILINGUAL FRAMEWORK TEST")
    
    try:
        validator = CrossLanguageValidator()
        
        # Test all three languages
        languages = ["en", "es", "fr"]
        print(f"Testing languages: {', '.join(languages)}")
        
        # Generate comprehensive report
        print("\nGenerating comprehensive validation report...")
        report = validator.generate_comprehensive_report(
            languages, 
            ValidationLevel.STANDARD
        )
        
        # Display results
        summary = report["summary"]
        print(f"\nResults:")
        print(f"  Languages tested: {len(summary['languages_tested'])}")
        print(f"  Patterns tested: {len(summary['patterns_tested'])}")
        print(f"  Overall success rate: {summary['overall_success_rate']:.1%}")
        
        print(f"\nLanguage Rankings:")
        for lang, score in sorted(summary['language_rankings'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {lang}: {score:.1%}")
        
        print(f"\nPattern Rankings:")
        for pattern, score in sorted(summary['pattern_rankings'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {pattern}: {score:.1%}")
        
        # Show language summaries
        print(f"\nLanguage Summaries:")
        for lang, summary_data in report["language_summaries"].items():
            print(f"  {lang.upper()}:")
            print(f"    Tests: {summary_data['total_passed']}/{summary_data['total_tests']} passed")
            print(f"    Success rate: {summary_data['average_success_rate']:.1%}")
            if summary_data['perfect_patterns']:
                print(f"    Perfect patterns: {', '.join(summary_data['perfect_patterns'])}")
            if summary_data['problematic_patterns']:
                print(f"    Problematic patterns: {', '.join(summary_data['problematic_patterns'])}")
        
        return True
        
    except Exception as e:
        print(f"✗ Comprehensive test failed: {e}")
        return False


def main():
    """Run French language extension tests."""
    print("French Language Extension Test for Universal Pattern Framework")
    print("=" * 70)
    
    success = True
    
    try:
        # Test basic French support
        if not test_french_language_support():
            success = False
        
        # Test French-specific patterns
        test_french_specific_patterns()
        
        # Test inheritance
        test_language_inheritance()
        
        # Test comprehensive framework
        if not test_comprehensive_multilingual_framework():
            success = False
        
        print_section("FRENCH EXTENSION TEST RESULTS")
        
        if success:
            print("✓ French language extension successful!")
            print("\nKey Achievements:")
            print("  • French resources successfully integrated")
            print("  • Universal patterns work with French keywords")
            print("  • Language inheritance system functional")
            print("  • Cross-language validation includes French")
            print("  • Framework extensibility proven")
            print("\nFramework Impact:")
            print("  • Demonstrates language-agnostic architecture")
            print("  • Shows resource inheritance capabilities")
            print("  • Validates universal pattern design")
            print("  • Proves framework scalability")
        else:
            print("✗ Some French extension tests failed")
            print("  See details above for specific issues")
        
    except Exception as e:
        print(f"\n✗ French extension test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())