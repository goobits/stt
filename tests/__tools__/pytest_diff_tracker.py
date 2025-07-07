# tests/pytest_diff_tracker.py

import pytest
import json
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
import re
from collections import defaultdict

# --- Constants ---
# Use the same history file as the old script to ensure compatibility.
HISTORY_DIR = Path(".test_artifacts")
HISTORY_FILE = HISTORY_DIR / "test_history.json"

# --- Pytest Hooks ---

def pytest_addoption(parser):
    """Add options to pytest command line."""
    group = parser.getgroup("test-diff-tracker")
    group.addoption(
        "--track-diff",
        action="store_true",
        default=False,
        help="Run tests and show diff vs last run."
    )
    group.addoption(
        "--history",
        action="store_true",
        default=False,
        help="Show recent test run history."
    )
    group.addoption(
        "--diff",
        dest="diff_range",
        nargs="*",
        metavar="INDICES",
        default=None,
        help="Compare runs. Examples: '--diff -1' (last vs current), '--diff -5 -1' (5th last vs last), '--diff 0 5' (first vs 5th)"
    )

def pytest_configure(config):
    """Called after command line options are parsed."""
    # If any of our options are used, register the plugin
    if config.getoption("--track-diff") or config.getoption("--history") or config.getoption("diff_range"):
        plugin = DiffTracker(config)
        config.pluginmanager.register(plugin, "difftracker")

def pytest_collection_modifyitems(config, items):
    """Modify collection to skip tests for read-only operations."""
    # If we're only showing history or diff, don't run any tests
    if config.getoption("--history") or config.getoption("diff_range"):
        # Clear all collected items to prevent test execution
        items.clear()

# --- Plugin Class ---

class DiffTracker:
    def __init__(self, config):
        self.config = config
        self.current_run_results = {}
        # Ensure the .test_artifacts directory exists before we do anything.
        HISTORY_DIR.mkdir(exist_ok=True)

    def _load_history(self) -> Dict:
        """Load test history from file."""
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        return {"runs": [], "test_metadata": {}}

    def _save_history(self, history: Dict):
        """Save test history to file."""
        # Keep only last 50 runs to avoid file bloat
        if len(history["runs"]) > 50:
            history["runs"] = history["runs"][-50:]
        
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)

    def _simplify_nodeid(self, nodeid: str) -> str:
        """Convert pytest nodeid to simplified test name matching old format."""
        # Example: 'tests/text_formatting/test_capitalization.py::TestCapitalization::test_sentence_case'
        # Should become: 'TestCapitalization::test_sentence_case'
        
        parts = nodeid.split('::')
        if len(parts) >= 2:
            # Get the class and method parts
            if len(parts) >= 3:
                # Class::method format
                return f"{parts[-2]}::{parts[-1]}"
            else:
                # Just method format
                return parts[-1]
        else:
            # Fallback: just return the nodeid as is
            return nodeid

    def _parse_failure_details(self, report) -> Dict:
        """Extract failure details from test report."""
        failure_details = {}
        
        if hasattr(report, 'longrepr') and report.longrepr:
            repr_str = str(report.longrepr)
            
            # Look for assertion errors with expected vs actual
            if 'AssertionError' in repr_str:
                lines = repr_str.split('\n')
                expected = None
                actual = None
                
                for line in lines:
                    if 'E     - ' in line:
                        expected = line.split('E     - ', 1)[1].strip()
                    elif 'E     + ' in line:
                        actual = line.split('E     + ', 1)[1].strip()
                        if expected and actual:
                            failure_details = {
                                "expected": expected,
                                "actual": actual
                            }
                            break
        
        return failure_details

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_logreport(self, report):
        """Capture result of each test."""
        # This hook is called for setup, call, and teardown.
        # We only care about the 'call' phase for pass/fail status.
        if report.when == "call":
            # Get the simplified test name
            short_name = self._simplify_nodeid(report.nodeid)
            
            # Determine status
            if report.passed:
                status = "PASSED"
            elif report.failed:
                status = "FAILED"
            elif report.skipped:
                status = "SKIPPED"
            else:
                status = "UNKNOWN"
            
            # Store the result
            result = {
                "status": status,
                "full_path": report.nodeid
            }
            
            # Add failure details if test failed
            if status == "FAILED":
                failure_details = self._parse_failure_details(report)
                if failure_details:
                    result["failure_details"] = failure_details
            
            self.current_run_results[short_name] = result

        yield # Allows other plugins to process the report

    def _create_run_record(self, test_path: str = "tests/") -> Dict:
        """Create a run record compatible with the old format."""
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "test_path": test_path,
            "summary": {
                "total": len(self.current_run_results),
                "passed": sum(1 for r in self.current_run_results.values() if r["status"] == "PASSED"),
                "failed": sum(1 for r in self.current_run_results.values() if r["status"] == "FAILED"),
                "skipped": sum(1 for r in self.current_run_results.values() if r["status"] == "SKIPPED"),
            },
            "tests": self.current_run_results,
            "return_code": 0 if all(r["status"] != "FAILED" for r in self.current_run_results.values()) else 1
        }

    def _get_diff(self, history: Dict, run1_idx: int = -2, run2_idx: int = -1) -> Dict:
        """Generate diff between two test runs."""
        if len(history["runs"]) < 2:
            return {"error": "Not enough runs to compare"}
        
        run1 = history["runs"][run1_idx]
        run2 = history["runs"][run2_idx]
        
        tests1 = run1["tests"]
        tests2 = run2["tests"]
        
        all_tests = set(tests1.keys()) | set(tests2.keys())
        
        diff = {
            "run1_timestamp": run1["timestamp"],
            "run2_timestamp": run2["timestamp"],
            "newly_passing": [],
            "newly_failing": [],
            "still_failing": [],
            "still_passing": [],
            "new_tests": [],
            "removed_tests": [],
            "summary": {
                "total_change": len(tests2) - len(tests1),
                "passed_change": run2["summary"]["passed"] - run1["summary"]["passed"],
                "failed_change": run2["summary"]["failed"] - run1["summary"]["failed"],
            }
        }
        
        for test in all_tests:
            if test not in tests1:
                diff["new_tests"].append(test)
            elif test not in tests2:
                diff["removed_tests"].append(test)
            else:
                status1 = tests1[test]["status"]
                status2 = tests2[test]["status"]
                
                if status1 == "FAILED" and status2 == "PASSED":
                    diff["newly_passing"].append(test)
                elif status1 == "PASSED" and status2 == "FAILED":
                    diff["newly_failing"].append({
                        "name": test,
                        "details": tests2[test].get("failure_details", {})
                    })
                elif status1 == "FAILED" and status2 == "FAILED":
                    diff["still_failing"].append({
                        "name": test,
                        "details": tests2[test].get("failure_details", {})
                    })
                elif status1 == "PASSED" and status2 == "PASSED":
                    diff["still_passing"].append(test)
        
        return diff

    def _print_diff(self, diff: Dict, run_summary: Dict, session=None):
        """Print a human-readable diff matching the proposal format."""
        if "error" in diff:
            self._write_line(f"\n❌ {diff['error']}", session)
            return
        
        # Changes summary
        newly_failing_count = len(diff["newly_failing"])
        newly_passing_count = len(diff["newly_passing"])
        
        self._write_line(f"\n📊 CHANGES SINCE LAST RUN: {newly_failing_count} newly failing, {newly_passing_count} newly passing", session)
        
        # Newly failing tests
        if diff["newly_failing"]:
            self._write_line("\nNEWLY_FAILING:", session)
            for test in diff["newly_failing"]:
                self._write_line(f"- {test['name']}", session)
                if test['details']:
                    self._write_line(f"  Expected: \"{test['details'].get('expected', 'N/A')}\"", session)
                    self._write_line(f"  Actual: \"{test['details'].get('actual', 'N/A')}\"", session)
        
        # Newly passing tests
        if diff["newly_passing"]:
            self._write_line("\nNEWLY_PASSING:", session)
            for test in diff["newly_passing"]:
                self._write_line(f"+ {test}", session)
        
        # Still failing tests (show count and a few examples)
        if diff["still_failing"]:
            still_failing_count = len(diff["still_failing"])
            self._write_line(f"\nSTILL_FAILING:", session)
            for test in diff["still_failing"][:3]:  # Show first 3
                self._write_line(f"- {test['name']} (continuing)", session)
            if still_failing_count > 3:
                self._write_line(f"  ... and {still_failing_count - 3} more", session)

    def _print_history(self, history: Dict, limit: int = 10):
        """Print test run history in the proposal format."""
        if not history["runs"]:
            print("No test runs recorded yet.")
            return
        
        print("\n📜 TEST RUN HISTORY")
        print("="*60)
        
        runs_to_show = history["runs"][-limit:]
        for i, run in enumerate(runs_to_show):
            # Calculate index from end
            actual_idx = len(history["runs"]) - limit + i
            if actual_idx < 0:
                actual_idx = i
            
            timestamp = run["timestamp"][:16].replace('T', ' ')  # Format: 2025-01-07 10:15
            passed = run["summary"]["passed"]
            total = run["summary"]["total"]
            
            status = "(current)" if i == len(runs_to_show) - 1 else ""
            print(f"[{actual_idx}] {timestamp}: {passed}/{total} passing {status}")

    def _write_line(self, text: str, session=None):
        """Write line to terminal, using session if available."""
        # Force output to stdout for debugging
        import sys
        print(text, file=sys.stdout, flush=True)

    def _parse_diff_range(self, diff_range: list) -> tuple:
        """Parse diff range list into (from_idx, to_idx).
        
        Examples:
          ['-1'] -> compare last run vs current (which would be new)
          ['-5', '-1'] -> compare 5th last vs last 
          ['0', '5'] -> compare first vs 5th
          ['-3'] -> compare 3rd last vs current
        """
        if len(diff_range) == 1:
            # Single index (compare with current/latest)
            from_idx = int(diff_range[0])
            to_idx = -1  # Latest run
            return from_idx, to_idx
        elif len(diff_range) == 2:
            # Two indices
            from_idx = int(diff_range[0])
            to_idx = int(diff_range[1])
            return from_idx, to_idx
        else:
            raise ValueError(f"Invalid diff range: {diff_range}. Expected 1 or 2 indices.")

    def pytest_sessionfinish(self, session):
        """Called after the entire test session finishes."""
        # Only run if one of our flags is active
        if not (self.config.getoption("--track-diff") or self.config.getoption("--history") or self.config.getoption("diff_range")):
            return
        
        # Load existing history
        history = self._load_history()
        
        if self.config.getoption("--track-diff"):
            # Create new run record
            run_record = self._create_run_record()
            
            # Add to history
            history["runs"].append(run_record)
            
            # Save updated history
            self._save_history(history)
            
            # Calculate and print diff if there's a previous run
            if len(history["runs"]) >= 2:
                diff = self._get_diff(history)
                self._print_diff(diff, run_record["summary"], session)
            else:
                self._write_line("\n📊 First run recorded. No diff to show.", session)
        
        elif self.config.getoption("--history"):
            self._print_history(history)
        
        elif self.config.getoption("diff_range"):
            diff_range = self.config.getoption("diff_range")
            try:
                from_idx, to_idx = self._parse_diff_range(diff_range)
                # Convert to negative indices for easier handling
                if from_idx >= 0:
                    from_idx = from_idx - len(history["runs"])
                if to_idx >= 0:
                    to_idx = to_idx - len(history["runs"])
                
                diff = self._get_diff(history, from_idx, to_idx)
                if "error" not in diff:
                    # For diff-only mode, create a summary from the target run
                    target_run = history["runs"][to_idx]
                    self._print_diff(diff, target_run["summary"], session)
                else:
                    self._write_line(f"\n❌ {diff['error']}", session)
            except (ValueError, IndexError) as e:
                self._write_line(f"\n❌ Error parsing diff range '{diff_range}': {e}", session)