#!/usr/bin/env python3
"""
Time Range Pattern Conflict Detector

This module provides specialized conflict detection for time range patterns,
ensuring they don't interfere with other numeric patterns like financial,
mathematical, or measurement patterns.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from stt.core.config import setup_logging
from .common import Entity, EntityType
from .constants import get_resources

logger = setup_logging(__name__)


class ConflictSeverity(Enum):
    """Severity levels for pattern conflicts."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ConflictReport:
    """Report of a detected pattern conflict."""
    test_input: str
    conflicting_patterns: List[str]
    expected_pattern: str
    severity: ConflictSeverity
    context_clues: List[str]
    recommendation: str


class TimeRangeConflictDetector:
    """Detects conflicts between time range patterns and other numeric patterns."""
    
    def __init__(self, language: str = "en"):
        self.language = language
        self.resources = get_resources(language)
        
        # Define time range patterns
        self.time_range_patterns = self._create_time_range_patterns()
        
        # Define conflicting patterns that might interfere
        self.conflicting_patterns = self._create_conflicting_patterns()
        
        # Define test cases for conflict detection
        self.conflict_test_cases = self._create_conflict_test_cases()
    
    def _create_time_range_patterns(self) -> Dict[str, re.Pattern]:
        """Create time range patterns to test."""
        patterns = {}
        
        # Basic "from X to Y" pattern
        patterns["from_to_time"] = re.compile(
            r"\bfrom\s+(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+to\s+(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
            re.IGNORECASE
        )
        
        # Compact "X to Y" pattern (context-dependent)
        patterns["compact_time"] = re.compile(
            r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+to\s+(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
            re.IGNORECASE
        )
        
        # "Between X and Y" pattern
        patterns["between_time"] = re.compile(
            r"\bbetween\s+(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+and\s+(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
            re.IGNORECASE
        )
        
        return patterns
    
    def _create_conflicting_patterns(self) -> Dict[str, re.Pattern]:
        """Create patterns that might conflict with time ranges."""
        patterns = {}
        
        # Financial range patterns
        patterns["financial_from_to"] = re.compile(
            r"\bfrom\s+(?:(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+)?(?:dollar|euro|pound|cent)s?\s+to\s+(?:(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+)?(?:dollar|euro|pound|cent)s?\b",
            re.IGNORECASE
        )
        
        # Mathematical range patterns
        patterns["math_range"] = re.compile(
            r"\b(?:divide|calculate|compute|sum|add|subtract|multiply)\s+(?:from\s+)?(one|two|three|four|five|six|seven|eight|nine|ten)\s+to\s+(one|two|three|four|five|six|seven|eight|nine|ten)\b",
            re.IGNORECASE
        )
        
        # Measurement range patterns
        patterns["measurement_range"] = re.compile(
            r"\bfrom\s+(one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:inch|foot|meter|mile|gram|pound|liter)\s+to\s+(one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:inch|foot|meter|mile|gram|pound|liter)\b",
            re.IGNORECASE
        )
        
        # Page/chapter reference patterns
        patterns["reference_range"] = re.compile(
            r"\b(?:from\s+)?(?:page|chapter|section|line)\s+(one|two|three|four|five|six|seven|eight|nine|ten)\s+to\s+(?:(?:page|chapter|section|line)\s+)?(one|two|three|four|five|six|seven|eight|nine|ten)\b",
            re.IGNORECASE
        )
        
        # Version/sequence patterns
        patterns["version_range"] = re.compile(
            r"\b(?:version|release|build|iteration)\s+(one|two|three|four|five|six|seven|eight|nine|ten)\s+to\s+(one|two|three|four|five|six|seven|eight|nine|ten)\b",
            re.IGNORECASE
        )
        
        return patterns
    
    def _create_conflict_test_cases(self) -> List[Tuple[str, str, ConflictSeverity]]:
        """Create test cases for conflict detection."""
        return [
            # High severity - clear financial context
            ("from two to five dollars", "financial", ConflictSeverity.HIGH),
            ("between ten and twenty euros", "financial", ConflictSeverity.HIGH),
            ("from five dollars to ten dollars", "financial", ConflictSeverity.CRITICAL),
            
            # High severity - clear mathematical context
            ("divide from nine to five", "mathematical", ConflictSeverity.HIGH),
            ("calculate from two to eight", "mathematical", ConflictSeverity.HIGH),
            ("add from one to ten", "mathematical", ConflictSeverity.MEDIUM),
            
            # Medium severity - reference context
            ("from page nine to five", "reference", ConflictSeverity.MEDIUM),
            ("chapters two to five", "reference", ConflictSeverity.MEDIUM),
            ("from line one to ten", "reference", ConflictSeverity.LOW),
            
            # Medium severity - measurement context
            ("from five inches to ten inches", "measurement", ConflictSeverity.HIGH),
            ("between two meters and five meters", "measurement", ConflictSeverity.MEDIUM),
            
            # Medium severity - version/technical context
            ("version nine to five", "technical", ConflictSeverity.MEDIUM),
            ("from build two to three", "technical", ConflictSeverity.LOW),
            
            # Low severity - ambiguous but likely time
            ("working from nine to five", "time", ConflictSeverity.LOW),
            ("schedule from two to four", "time", ConflictSeverity.LOW),
            ("meeting nine to five", "time", ConflictSeverity.LOW),
            
            # Test cases that should clearly be time ranges
            ("from nine to five shift", "time", ConflictSeverity.LOW),
            ("business hours nine to five", "time", ConflictSeverity.LOW),
            ("office from eight to six", "time", ConflictSeverity.LOW),
        ]
    
    def detect_conflicts(self) -> List[ConflictReport]:
        """Detect all pattern conflicts for time ranges."""
        logger.info("Starting time range pattern conflict detection")
        
        conflicts = []
        
        for test_input, expected_context, severity in self.conflict_test_cases:
            conflict_report = self._analyze_single_input(test_input, expected_context, severity)
            if conflict_report:
                conflicts.append(conflict_report)
        
        logger.info(f"Found {len(conflicts)} pattern conflicts")
        return conflicts
    
    def _analyze_single_input(
        self, 
        test_input: str, 
        expected_context: str, 
        severity: ConflictSeverity
    ) -> Optional[ConflictReport]:
        """Analyze a single input for pattern conflicts."""
        matching_patterns = []
        
        # Check time range patterns
        for name, pattern in self.time_range_patterns.items():
            if pattern.search(test_input):
                matching_patterns.append(f"time_range_{name}")
        
        # Check conflicting patterns
        for name, pattern in self.conflicting_patterns.items():
            if pattern.search(test_input):
                matching_patterns.append(f"conflict_{name}")
        
        # Only report if there are multiple matches or unexpected matches
        if len(matching_patterns) > 1:
            context_clues = self._extract_context_clues(test_input)
            recommendation = self._generate_recommendation(
                test_input, matching_patterns, expected_context
            )
            
            return ConflictReport(
                test_input=test_input,
                conflicting_patterns=matching_patterns,
                expected_pattern=expected_context,
                severity=severity,
                context_clues=context_clues,
                recommendation=recommendation
            )
        
        # Check for missed time patterns (should match but doesn't)
        elif expected_context == "time" and not any("time_range" in p for p in matching_patterns):
            return ConflictReport(
                test_input=test_input,
                conflicting_patterns=matching_patterns,
                expected_pattern="time",
                severity=ConflictSeverity.MEDIUM,
                context_clues=self._extract_context_clues(test_input),
                recommendation="Time range pattern not detected - may need pattern enhancement"
            )
        
        return None
    
    def _extract_context_clues(self, text: str) -> List[str]:
        """Extract context clues that indicate the intended pattern type."""
        clues = []
        text_lower = text.lower()
        
        # Financial clues
        if any(word in text_lower for word in ["dollar", "euro", "pound", "cent", "money", "cost", "price"]):
            clues.append("financial_context")
        
        # Time clues
        if any(word in text_lower for word in ["shift", "hours", "work", "business", "office", "schedule", "am", "pm"]):
            clues.append("time_context")
        
        # Mathematical clues
        if any(word in text_lower for word in ["divide", "calculate", "compute", "add", "subtract", "multiply", "sum"]):
            clues.append("mathematical_context")
        
        # Reference clues
        if any(word in text_lower for word in ["page", "chapter", "section", "line", "paragraph"]):
            clues.append("reference_context")
        
        # Measurement clues
        if any(word in text_lower for word in ["inch", "foot", "meter", "mile", "gram", "pound", "liter", "gallon"]):
            clues.append("measurement_context")
        
        # Technical clues
        if any(word in text_lower for word in ["version", "release", "build", "iteration"]):
            clues.append("technical_context")
        
        return clues
    
    def _generate_recommendation(
        self, 
        test_input: str, 
        matching_patterns: List[str], 
        expected_context: str
    ) -> str:
        """Generate a recommendation for resolving the conflict."""
        recommendations = []
        
        # If multiple patterns match, suggest priority ordering
        if len(matching_patterns) > 1:
            recommendations.append(
                f"Multiple patterns match ({', '.join(matching_patterns)}). "
                f"Implement priority ordering to prefer {expected_context} context."
            )
        
        # Context-specific recommendations
        context_clues = self._extract_context_clues(test_input)
        
        if "financial_context" in context_clues and expected_context != "financial":
            recommendations.append(
                "Strong financial context detected. Financial patterns should take precedence."
            )
        
        if "time_context" in context_clues and expected_context == "time":
            recommendations.append(
                "Time context clues present. Ensure time range patterns have sufficient priority."
            )
        
        if "mathematical_context" in context_clues:
            recommendations.append(
                "Mathematical context detected. Math patterns should override time patterns."
            )
        
        # Pattern improvement suggestions
        if not any("time_range" in p for p in matching_patterns) and expected_context == "time":
            recommendations.append(
                "Consider expanding time range patterns to cover this case."
            )
        
        if len(recommendations) == 0:
            recommendations.append("Review pattern priority and context detection logic.")
        
        return " ".join(recommendations)
    
    def generate_conflict_summary(self, conflicts: List[ConflictReport]) -> Dict[str, any]:
        """Generate a summary of all detected conflicts."""
        if not conflicts:
            return {
                "total_conflicts": 0,
                "severity_breakdown": {},
                "pattern_breakdown": {},
                "recommendations": ["No conflicts detected - patterns appear to be working correctly."]
            }
        
        # Severity breakdown
        severity_counts = {}
        for conflict in conflicts:
            severity = conflict.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Pattern breakdown
        pattern_conflicts = {}
        for conflict in conflicts:
            for pattern in conflict.conflicting_patterns:
                pattern_conflicts[pattern] = pattern_conflicts.get(pattern, 0) + 1
        
        # Most problematic inputs
        critical_conflicts = [c for c in conflicts if c.severity == ConflictSeverity.CRITICAL]
        high_conflicts = [c for c in conflicts if c.severity == ConflictSeverity.HIGH]
        
        # Generate overall recommendations
        overall_recommendations = []
        
        if critical_conflicts:
            overall_recommendations.append(
                f"CRITICAL: {len(critical_conflicts)} critical conflicts need immediate attention"
            )
        
        if high_conflicts:
            overall_recommendations.append(
                f"HIGH: {len(high_conflicts)} high-severity conflicts should be resolved"
            )
        
        # Most common conflicting patterns
        if pattern_conflicts:
            most_common = max(pattern_conflicts.items(), key=lambda x: x[1])
            overall_recommendations.append(
                f"Most problematic pattern: {most_common[0]} ({most_common[1]} conflicts)"
            )
        
        return {
            "total_conflicts": len(conflicts),
            "severity_breakdown": severity_counts,
            "pattern_breakdown": pattern_conflicts,
            "critical_cases": [c.test_input for c in critical_conflicts],
            "high_severity_cases": [c.test_input for c in high_conflicts],
            "recommendations": overall_recommendations,
            "detailed_conflicts": [
                {
                    "input": c.test_input,
                    "expected": c.expected_pattern,
                    "patterns": c.conflicting_patterns,
                    "severity": c.severity.value,
                    "context_clues": c.context_clues,
                    "recommendation": c.recommendation
                }
                for c in conflicts
            ]
        }
    
    def print_conflict_report(self, conflicts: List[ConflictReport]):
        """Print a human-readable conflict report."""
        print("=" * 70)
        print("TIME RANGE PATTERN CONFLICT DETECTION REPORT")
        print("=" * 70)
        
        if not conflicts:
            print("✅ No conflicts detected! Patterns appear to be working correctly.")
            return
        
        summary = self.generate_conflict_summary(conflicts)
        
        print(f"\nSUMMARY:")
        print(f"  Total conflicts: {summary['total_conflicts']}")
        print(f"  Severity breakdown: {summary['severity_breakdown']}")
        
        # Show critical and high severity cases
        if summary['critical_cases']:
            print(f"\n🚨 CRITICAL CONFLICTS:")
            for case in summary['critical_cases']:
                print(f"    '{case}'")
        
        if summary['high_severity_cases']:
            print(f"\n⚠️  HIGH SEVERITY CONFLICTS:")
            for case in summary['high_severity_cases']:
                print(f"    '{case}'")
        
        # Show detailed conflicts
        print(f"\nDETAILED CONFLICT ANALYSIS:")
        for i, conflict in enumerate(conflicts[:10], 1):  # Show first 10
            severity_icon = {
                ConflictSeverity.CRITICAL: "🚨",
                ConflictSeverity.HIGH: "⚠️",
                ConflictSeverity.MEDIUM: "⚡",
                ConflictSeverity.LOW: "ℹ️"
            }.get(conflict.severity, "?")
            
            print(f"\n  {i}. {severity_icon} '{conflict.test_input}'")
            print(f"     Expected: {conflict.expected_pattern}")
            print(f"     Matches: {', '.join(conflict.conflicting_patterns)}")
            if conflict.context_clues:
                print(f"     Context: {', '.join(conflict.context_clues)}")
            print(f"     Fix: {conflict.recommendation}")
        
        # Show recommendations
        print(f"\nRECOMMENDATIONS:")
        for i, rec in enumerate(summary['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("=" * 70)


def run_time_range_conflict_detection(language: str = "en") -> Dict[str, any]:
    """Run time range conflict detection and return results."""
    detector = TimeRangeConflictDetector(language)
    conflicts = detector.detect_conflicts()
    summary = detector.generate_conflict_summary(conflicts)
    
    return {
        "language": language,
        "conflicts": conflicts,
        "summary": summary
    }


if __name__ == "__main__":
    # Run conflict detection
    detector = TimeRangeConflictDetector()
    conflicts = detector.detect_conflicts()
    
    # Print report
    detector.print_conflict_report(conflicts)
    
    # Save detailed results
    import json
    from pathlib import Path
    
    summary = detector.generate_conflict_summary(conflicts)
    output_file = Path("/tmp/time_range_conflict_report.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nDetailed report saved to: {output_file}")