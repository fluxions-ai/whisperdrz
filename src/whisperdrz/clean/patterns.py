"""Consolidated regex patterns for WhisperDRZ transcription processing."""

import re

# =============================================================================
# BASE COMPONENTS
# =============================================================================

_DIGIT = r"\d+"
_FLOAT = r"\d+\.\d+"
_TIMESTAMP = rf"<\|{_FLOAT}\|>"  # <|1.23|>

# Base pattern exports (for files that need raw patterns)
_DIGIT_PATTERN = _DIGIT
_FLOAT_PATTERN = _FLOAT
_TIMING_CORE = _TIMESTAMP

# =============================================================================
# TIMESTAMPS  -  <|0.00|>word<|0.50|>
# =============================================================================

# <|1.23|> -> captures "1.23"
TIMINGS_FORMAT = re.compile(rf"<\|({_FLOAT})\|>")

# =============================================================================
# SPEAKERS  -  [0] Hello [1] Hi
# =============================================================================

# [0], [1] -> captures digit
BRACKET_DIGIT_MATCH = re.compile(rf"\[({_DIGIT})\]")

# [0] or [c] at line start -> captures speaker id
SPEAKER_LINE_START = re.compile(r"^\[(\d+|c)\]\s*")

# [0] or [c] not at line start
SPEAKER_TAGS_GENERAL = re.compile(rf"(?<=[^\n])\[(?:{_DIGIT}|c)\]\s*")

# =============================================================================
# METADATA TAGS  -  <emotion:happy> <accent:british>
# =============================================================================

# <emotion:happy>, <accent:british>, <lang:en>
METADATA = re.compile(r"<[^|][^>]*>")

# =============================================================================
# ALIGNMENT TIMING  -  <a.5|1.23|>word<a.5|1.50|>
# =============================================================================

# <a.5|1.23|> -> captures score, time
SCORED_TIMING_PATTERN = re.compile(rf"<a\.({_DIGIT})\|({_FLOAT})\|>")

# =============================================================================
# OVERLAP PATTERNS  -  [1 word word 1]
# =============================================================================

# Content that allows nested [tag] patterns: matches non-brackets OR complete [lowercase] tags
_CONTENT_WITH_TAGS = r"(?:[^\[\]]|\[[a-z/]+\])+"

# [1 ... 1] any content
INTERRUPTIONS = re.compile(rf"\[({_DIGIT})\s{_CONTENT_WITH_TAGS}\s\1\]")

# =============================================================================
# WER CLEANING PATTERNS
# =============================================================================

BRACKETS_GENERAL = re.compile(r"\[.*?\]")
ENDOFTEXT_PATTERN = re.compile(r"endoftext", re.IGNORECASE)
