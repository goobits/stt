#!/usr/bin/env python3
"""
Code-related regular expression patterns for text formatting.

This module contains all programming, command, file, and technical patterns used
throughout the text formatting system, organized logically and using verbose
formatting for readability and maintainability.

All patterns use re.VERBOSE flag where beneficial and include detailed comments
explaining each component.
"""
from __future__ import annotations

import re
from typing import Pattern

from ..constants import get_resources, get_nested_resource
from ..pattern_cache import cached_pattern
from ..modern_pattern_cache import cached_pattern as modern_cached_pattern
from ..universal_code_mapper import get_universal_code_mapper, CodeSymbolType


# ==============================================================================
# FILE EXTENSIONS CONSTANTS
# ==============================================================================

# File extensions organized by category
FILE_EXTENSIONS = {
    "code": [
        "py",
        "js",
        "ts",
        "tsx",
        "jsx",
        "cpp",
        "c",
        "h",
        "hpp",
        "java",
        "cs",
        "go",
        "rs",
        "rb",
        "php",
        "sh",
        "bash",
        "zsh",
        "fish",
        "bat",
        "cmd",
        "ps1",
        "swift",
        "kt",
        "scala",
        "r",
        "m",
        "lua",
        "pl",
        "asm",
    ],
    "data": [
        "json",
        "jsonl",
        "xml",
        "yaml",
        "yml",
        "toml",
        "ini",
        "cfg",
        "conf",
        "csv",
        "tsv",
        "sql",
        "db",
        "sqlite",
        "custom",
    ],
    "document": ["md", "txt", "rst", "tex", "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "odt", "ods", "odp"],
    "web": ["html", "htm", "css", "scss", "sass", "less"],
    "media": [
        "png",
        "jpg",
        "jpeg",
        "gif",
        "svg",
        "ico",
        "bmp",
        "webp",
        "mp3",
        "mp4",
        "avi",
        "mov",
        "mkv",
        "webm",
        "wav",
        "flac",
        "ogg",
    ],
    "archive": ["zip", "tar", "gz", "bz2", "xz", "rar", "7z", "deb", "rpm", "dmg", "pkg", "exe", "msi", "app"],
}

# Flatten all extensions for use in patterns
ALL_FILE_EXTENSIONS = sum(FILE_EXTENSIONS.values(), [])


# ==============================================================================
# CODE PATTERN BUILDERS
# ==============================================================================


@modern_cached_pattern(category='core', language_aware=True)
def build_slash_command_pattern(language: str = "en") -> re.Pattern[str]:
    """Builds the slash command pattern using Universal Code Mapper for cross-language support."""
    # Use Universal Code Mapper for smart cross-language pattern building
    mapper = get_universal_code_mapper()
    
    pattern_template = r"""
        \b                                  # Word boundary
        {SLASH_KEYWORDS}\s+                 # Slash keyword followed by space
        ([a-zA-Z][a-zA-Z0-9_-]*)           # Command name (starts with letter, can contain letters, numbers, underscore, hyphen)
        \b                                  # Word boundary
    """
    
    return mapper.build_universal_pattern(
        language=language,
        symbol_types=[CodeSymbolType.SLASH],
        pattern_template=pattern_template
    )


@modern_cached_pattern(category='core', language_aware=True)
def build_underscore_delimiter_pattern(language: str = "en") -> re.Pattern[str]:
    """Builds the underscore delimiter pattern using Universal Code Mapper for cross-language support."""
    # Use Universal Code Mapper for smart cross-language pattern building
    mapper = get_universal_code_mapper()
    
    pattern_template = r"""
        \b                                      # Word boundary
        ((?:{UNDERSCORE_KEYWORDS}\s+)+)        # One or more underscore keywords followed by space (captured)
        ([a-zA-Z][a-zA-Z0-9_-]*)               # Content (starts with letter, can contain letters, numbers, underscore, hyphen)
        ((?:\s+{UNDERSCORE_KEYWORDS})+)        # One or more space followed by underscore keywords (captured)
        (?=\s|$)                               # Must be followed by space or end of string
    """
    
    return mapper.build_universal_pattern(
        language=language,
        symbol_types=[CodeSymbolType.UNDERSCORE],
        pattern_template=pattern_template
    )


@modern_cached_pattern(category='core', language_aware=True)
def build_simple_underscore_pattern(language: str = "en") -> re.Pattern[str]:
    """Builds the simple underscore pattern using Universal Code Mapper for cross-language support."""
    # Use Universal Code Mapper for smart cross-language pattern building
    mapper = get_universal_code_mapper()
    
    pattern_template = r"""
        \b                                      # Word boundary
        ([\w][\w0-9_-]*)                       # First word (starts with letter, supports Unicode)
        \s+{UNDERSCORE_KEYWORDS}\s+             # Space, underscore keyword, space
        ([\w][\w0-9_-]*)                       # Second word (starts with letter, supports Unicode)
        \b                                      # Word boundary
    """
    
    return mapper.build_universal_pattern(
        language=language,
        symbol_types=[CodeSymbolType.UNDERSCORE],
        pattern_template=pattern_template
    )


@modern_cached_pattern(category='common', language_aware=True)
def build_long_flag_pattern(language: str = "en") -> re.Pattern[str]:
    """Builds the long flag pattern using Universal Code Mapper for cross-language support."""
    # Use Universal Code Mapper for smart cross-language pattern building
    mapper = get_universal_code_mapper()
    
    # Define known compound flags that should be matched as two words
    compound_flag_names = [
        "save dev", "dry run", "no cache", "cache dir", "output dir", 
        "config file", "help text", "version info", "no deps", "dev deps"
    ]
    
    # Create patterns for compound flags (must come first to have priority)
    compound_patterns = []
    for compound in compound_flag_names:
        # Match the compound with flexible spacing, no individual capture groups
        escaped = re.escape(compound).replace(r'\ ', r'\s+')
        compound_patterns.append(escaped)
    
    # Create the final pattern that captures the flag name (compound or single)
    if compound_patterns:
        # Use non-capturing groups for alternation, with one overall capture group
        compound_part = f"(?:{'|'.join(compound_patterns)})"
        single_word_part = r"[a-zA-Z][a-zA-Z0-9_-]*"
        # Create a single capture group that encompasses both options
        flag_name_pattern = f"({compound_part}|{single_word_part})"
    else:
        flag_name_pattern = r"([a-zA-Z][a-zA-Z0-9_-]*)"
    
    pattern_template = f"\\b{{DASH_KEYWORDS}}\\s+{{DASH_KEYWORDS}}\\s+{flag_name_pattern}\\b"
    
    return mapper.build_universal_pattern(
        language=language,
        symbol_types=[CodeSymbolType.DASH],
        pattern_template=pattern_template
    )


@modern_cached_pattern(category='common', language_aware=True)
def build_short_flag_pattern(language: str = "en") -> re.Pattern[str]:
    """Builds the short flag pattern using Universal Code Mapper for cross-language support."""
    # Use Universal Code Mapper for smart cross-language pattern building
    mapper = get_universal_code_mapper()
    
    # For Spanish, we need to handle "guión bajo" (underscore) vs "guión" (dash)
    if language == "es":
        # Get dash keywords from mapper, excluding ones that conflict with underscore
        dash_phrases = mapper.get_spoken_phrases_for_symbol(language, CodeSymbolType.DASH)
        # Remove "guión" and replace with "guión" + negative lookahead
        safe_dash_phrases = []
        for phrase in dash_phrases:
            if phrase == "guión":
                safe_dash_phrases.append("(?:guión)(?!\\s+bajo)")
            else:
                safe_dash_phrases.append(re.escape(phrase))
        # Deduplicate
        safe_dash_phrases = list(set(safe_dash_phrases))
        dash_pattern = f"(?:{'|'.join(safe_dash_phrases)})"
        pattern_template = f"\\b{dash_pattern}\\s+([a-zA-Z0-9-]+)\\b"
        return re.compile(pattern_template, re.IGNORECASE)
    else:
        pattern_template = r"\b{DASH_KEYWORDS}\s+([a-zA-Z0-9-]+)\b"
    
    return mapper.build_universal_pattern(
        language=language,
        symbol_types=[CodeSymbolType.DASH],
        pattern_template=pattern_template
    )


@modern_cached_pattern(category='common', language_aware=True)
def build_assignment_pattern(language: str = "en") -> re.Pattern[str]:
    """Builds the assignment pattern using Universal Code Mapper for cross-language support."""
    # Use Universal Code Mapper for smart cross-language pattern building
    mapper = get_universal_code_mapper()
    
    pattern_template = r"""
        \b                                      # Word boundary
        (?:(let|const|var)\s+)?                 # Optional variable declaration keyword (capture group 1)
        ([a-zA-Z_]\w*)                          # Variable name (capture group 2)
        \s+{EQUALS_KEYWORDS}\s+                 # Space, equals keyword, space
        (                                       # Value (capture group 3) - now captures chained assignments
            (?!.*\b(?:times|plus|minus|divided\s+by|over|squared?|cubed?)\b)  # Not a math expression
            (?:(?!\s+(?:and|or|but|if|when|then|while|unless)\s+).)+?        # Stop at conjunctions
        )
        (?=\s*$|\s*[.!?]|\s+(?:and|or|but|if|when|then|while|unless)\s+|--|\+\+)  # Lookahead: end of string, punctuation, conjunctions, or operators
    """
    
    return mapper.build_universal_pattern(
        language=language,
        symbol_types=[CodeSymbolType.EQUALS],
        pattern_template=pattern_template
    )


# ==============================================================================
# GETTER FUNCTIONS
# ==============================================================================


def get_slash_command_pattern(language: str = "en") -> re.Pattern[str]:
    """Get the slash command pattern for the specified language."""
    return build_slash_command_pattern(language)


def get_underscore_delimiter_pattern(language: str = "en") -> re.Pattern[str]:
    """Get the underscore delimiter pattern for the specified language."""
    return build_underscore_delimiter_pattern(language)


def get_simple_underscore_pattern(language: str = "en") -> re.Pattern[str]:
    """Get the simple underscore pattern for the specified language."""
    return build_simple_underscore_pattern(language)


def get_long_flag_pattern(language: str = "en") -> re.Pattern[str]:
    """Get the long flag pattern for the specified language."""
    return build_long_flag_pattern(language)


def get_short_flag_pattern(language: str = "en") -> re.Pattern[str]:
    """Get the short flag pattern for the specified language."""
    return build_short_flag_pattern(language)


def get_assignment_pattern(language: str = "en") -> re.Pattern[str]:
    """Get the assignment pattern for the specified language."""
    return build_assignment_pattern(language)


# ==============================================================================
# DEFAULT PATTERNS (BACKWARD COMPATIBILITY)
# ==============================================================================

# Default English patterns for backward compatibility
SLASH_COMMAND_PATTERN = build_slash_command_pattern("en")
UNDERSCORE_DELIMITER_PATTERN = build_underscore_delimiter_pattern("en")
SIMPLE_UNDERSCORE_PATTERN = build_simple_underscore_pattern("en")
LONG_FLAG_PATTERN = build_long_flag_pattern("en")
SHORT_FLAG_PATTERN = build_short_flag_pattern("en")
ASSIGNMENT_PATTERN = build_assignment_pattern("en")


# ==============================================================================
# FILENAME PATTERNS
# ==============================================================================

# File extension detection pattern
FILE_EXTENSION_DETECTION_PATTERN = re.compile(
    r"""
    \.                                  # Literal dot
    (?:                                 # Non-capturing group for file extensions
        """
    + "|".join(ALL_FILE_EXTENSIONS)
    + r"""
    )
    \b                                  # Word boundary
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Filename detection with extension
FILENAME_WITH_EXTENSION_PATTERN = re.compile(
    r"""
    \b                                  # Word boundary
    (?:                                 # Filename part
        \w+                             # First word/component
        (?:[_\-]\w+)*                   # Additional components connected by _ or - (NOT space)
    )                                   # Required filename
    \.                                  # Literal dot
    ("""
    + "|".join(ALL_FILE_EXTENSIONS)
    + r""")  # File extension (grouped)
    \b                                  # Word boundary
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Spoken filename patterns
SPOKEN_DOT_FILENAME_PATTERN = re.compile(r"\s+dot\s+(" + "|".join(ALL_FILE_EXTENSIONS) + r")\b", re.IGNORECASE)

# Comprehensive spoken filename pattern that captures the full filename
# Matches patterns like "my script dot py", "config loader dot json", etc.
# Uses capture groups to separate filename and extension
@cached_pattern
def build_full_spoken_filename_pattern() -> re.Pattern[str]:
    """Build full spoken filename pattern."""
    return re.compile(
        rf"""
        \b                                          # Word boundary
        ([a-z]\w*(?:\s+[a-z]\w*)*)                 # Capture filename part (one or more words)
        \s+dot\s+                                   # " dot "
        ({"|".join(ALL_FILE_EXTENSIONS)})           # Capture file extension
        \b                                          # Word boundary
        """,
        re.VERBOSE | re.IGNORECASE,
    )

FULL_SPOKEN_FILENAME_PATTERN = build_full_spoken_filename_pattern()

# Java package pattern: "com dot example dot package"
@cached_pattern
def build_java_package_pattern() -> re.Pattern[str]:
    """Build Java package pattern."""
    return re.compile(r"\b([a-zA-Z]\w*(?:\s+dot\s+[a-zA-Z]\w*){2,})\b", re.IGNORECASE)

JAVA_PACKAGE_PATTERN = build_java_package_pattern()

# Main spoken filename pattern (simple anchor)
SPOKEN_FILENAME_PATTERN = SPOKEN_DOT_FILENAME_PATTERN


# ==============================================================================
# COMPARISON AND PROGRAMMING PATTERNS
# ==============================================================================

# Mixed case technical terms (pre-compiled)
@cached_pattern
def build_mixed_case_tech_pattern() -> re.Pattern[str]:
    """Build mixed case technical terms pattern."""
    return re.compile(
        r"\b(?:JavaScript|TypeScript|GitHub|GitLab|BitBucket|DevOps|GraphQL|MongoDB|"
        r"PostgreSQL|MySQL|NoSQL|WebSocket|OAuth|iOS|macOS|iPadOS|tvOS|watchOS|"
        r"iPhone|iPad|macBook|iMac|AirPods|WiFi|Bluetooth|HTTP|HTTPS|API|JSON|XML|"
        r"HTML|CSS|SQL|PDF|URL|UUID|CSV|TSV|ZIP|RAM|CPU|GPU|SSD|USB|HDMI|"
        r"YouTube|LinkedIn|Facebook|Twitter|Instagram|TikTok|WhatsApp|Zoom|Slack|"
        r"Visual\s+Studio|IntelliJ|PyCharm|WebStorm|Eclipse|NetBeans|Xcode)\b"
    )

MIXED_CASE_TECH_PATTERN = build_mixed_case_tech_pattern()

# Technical sequence pattern
@cached_pattern
def build_tech_sequence_pattern() -> re.Pattern[str]:
    """Build technical sequence pattern."""
    return re.compile(r"\b(?:[A-Z]{2,}(?:\s+[A-Z]{2,})+)\b")

TECH_SEQUENCE_PATTERN = build_tech_sequence_pattern()

# Math expression pattern for variable assignments
@cached_pattern
def build_math_expression_pattern() -> re.Pattern[str]:
    """Build math expression pattern."""
    return re.compile(r"\b[a-zA-Z_]\w*\s*=\s*[\w\d]+(?:\s*[+\-*/×÷]\s*[\w\d]+)*\b")

MATH_EXPRESSION_PATTERN = build_math_expression_pattern()

# Version number patterns
@cached_pattern
def build_version_pattern() -> re.Pattern[str]:
    """Build version number pattern."""
    return re.compile(
        r"\bversion\s+(?:(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
        r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)\s*)+",
        re.IGNORECASE,
    )

VERSION_PATTERN = build_version_pattern()


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================


def get_file_extensions_by_category(category: str) -> list[str]:
    """Get file extensions for a specific category."""
    return FILE_EXTENSIONS.get(category, [])


def get_all_file_extensions() -> list[str]:
    """Get all file extensions as a flat list."""
    return ALL_FILE_EXTENSIONS.copy()


def create_alternation_pattern(items: list[str], word_boundaries: bool = True) -> str:
    """Create a regex alternation pattern from a list of items."""
    escaped_items = [re.escape(item) for item in items]
    pattern = "|".join(escaped_items)
    if word_boundaries:
        pattern = rf"\b(?:{pattern})\b"
    return pattern


def get_compiled_code_pattern(pattern_name: str) -> Pattern | None:
    """Get a pre-compiled code pattern by name."""
    pattern_map = {
        "slash_command": SLASH_COMMAND_PATTERN,
        "underscore_delimiter": UNDERSCORE_DELIMITER_PATTERN,
        "simple_underscore": SIMPLE_UNDERSCORE_PATTERN,
        "long_flag": LONG_FLAG_PATTERN,
        "short_flag": SHORT_FLAG_PATTERN,
        "assignment": ASSIGNMENT_PATTERN,
        "file_extension_detection": FILE_EXTENSION_DETECTION_PATTERN,
        "filename_with_extension": FILENAME_WITH_EXTENSION_PATTERN,
        "spoken_dot_filename": SPOKEN_DOT_FILENAME_PATTERN,
        "full_spoken_filename": FULL_SPOKEN_FILENAME_PATTERN,
        "java_package": JAVA_PACKAGE_PATTERN,
        "spoken_filename": SPOKEN_FILENAME_PATTERN,
        "mixed_case_tech": MIXED_CASE_TECH_PATTERN,
        "tech_sequence": TECH_SEQUENCE_PATTERN,
        "math_expression": MATH_EXPRESSION_PATTERN,
        "version": VERSION_PATTERN,
    }
    return pattern_map.get(pattern_name)


# Getter functions for the cached patterns
def get_file_extension_detection_pattern() -> re.Pattern[str]:
    """Get the file extension detection pattern."""
    return FILE_EXTENSION_DETECTION_PATTERN


def get_filename_with_extension_pattern() -> re.Pattern[str]:
    """Get the filename with extension pattern."""
    return FILENAME_WITH_EXTENSION_PATTERN


def get_spoken_dot_filename_pattern() -> re.Pattern[str]:
    """Get the spoken dot filename pattern."""
    return SPOKEN_DOT_FILENAME_PATTERN


def get_full_spoken_filename_pattern() -> re.Pattern[str]:
    """Get the full spoken filename pattern."""
    return build_full_spoken_filename_pattern()


def get_java_package_pattern() -> re.Pattern[str]:
    """Get the Java package pattern."""
    return build_java_package_pattern()


def get_mixed_case_tech_pattern() -> re.Pattern[str]:
    """Get the mixed case tech pattern."""
    return build_mixed_case_tech_pattern()


def get_tech_sequence_pattern() -> re.Pattern[str]:
    """Get the technical sequence pattern."""
    return build_tech_sequence_pattern()


def get_math_expression_pattern() -> re.Pattern[str]:
    """Get the math expression pattern."""
    return build_math_expression_pattern()


def get_version_pattern() -> re.Pattern[str]:
    """Get the version pattern."""
    return build_version_pattern()