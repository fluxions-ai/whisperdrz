"""Consolidated regex patterns for WhisperDRZ transcription processing."""

import re

# =============================================================================
# BASE COMPONENTS
# =============================================================================

_DIGIT = r"\d+"
_FLOAT = r"\d+\.\d+"
_TIMESTAMP = rf"<\|{_FLOAT}\|>"  # <|1.23|>
_SPEAKER = rf"\[{_DIGIT}\]"  # [0], [1], [2]
_SPEAKER_OR_C = rf"(?:{_DIGIT}|c)"  # 0, 1, c

# =============================================================================
# TIMESTAMPS  -  <|0.00|>word<|0.50|>
# =============================================================================

# <|1.23|> -> captures "1.23"
TIMINGS_FORMAT = re.compile(rf"<\|({_FLOAT})\|>")

# <|0.00|>word<|0.50|> -> captures start, word, end
TIMESTAMP_WORD = re.compile(rf"<\|({_FLOAT})\|>([^<]*)<\|({_FLOAT})\|>")

# Validates exact format: <|1.23|>
TIMESTAMP_VALID = re.compile(r"^<\|\d+\.\d{2}\|>$")

# =============================================================================
# SPEAKERS  -  [0] Hello [1] Hi
# =============================================================================

# [0], [1] -> captures digit
SPEAKER = re.compile(rf"\[({_DIGIT})\]")

# [0], [c] at line start -> captures speaker
SPEAKER_LINE_START = re.compile(rf"^\[({_SPEAKER_OR_C})\]\s*")

# ^[0]  or ^[c]  with trailing content
SPEAKER_FULL = re.compile(r"^(\[(?:\d+|c)\]\s*)")

# ^[0]  specifically
SPEAKER_ZERO = re.compile(r"^\[0\]\s*")

# =============================================================================
# NON-SPEECH TAGS  -  [laugh] [breath] [/whisper]
# =============================================================================

# [laugh], [breath] - lowercase only
NONSPEECH = re.compile(r"\[[a-z]+\]")

# [Laugh], [BREATH] - any case
NONSPEECH_ANY = re.compile(r"\[[a-zA-Z]+\]")

# [/laugh], [/whisper]
CLOSING_TAG = re.compile(r"\[/[a-zA-Z]+\]")

# [laughing], [whisper], [shouting], [voicebreaking] - paired tags
NONSPEECH_PAIRED = re.compile(r"\[(laughing|whisper|shouting|voicebreaking)\]\s*")

# [laugh], [breath] but NOT paired tags
NONSPEECH_SINGLE = re.compile(
    r"\[(?!/?(?:laughing|whisper|shouting|voicebreaking)\])([a-zA-Z]{2,})\]\s*"
)

# Droppable nonspeech (excludes inaudible, pause, crowd sounds)
# Also removes surrounding timestamps when present: <|t|>[tag]<|t|> -> ""
NONSPEECH_DROPPABLE = re.compile(
    rf"(?:{_TIMESTAMP})?\[(?!/?(?:laughing|whisper|shouting|voicebreaking|inaudible|pause|chatter|applause|cheering|crowd\w+)\])([a-zA-Z]{{2,}})\](?:{_TIMESTAMP})?\s*"
)

# \n[0] or [0] with trailing space
SPEAKER_TAG_LINE = re.compile(r"\n?\[(\d+|c)\]\s*")

# =============================================================================
# METADATA TAGS  -  <emotion:happy> <accent:british>
# =============================================================================

# <emotion:happy>, <accent:british>, <lang:en>
METADATA = re.compile(r"<[^|][^>]*>")

# <emotion:X> but not <accent:X>
EMOTION = re.compile(r"<(?!\|)\w+(?<!accent)>\s*")

# <Xaccent> e.g. <britishaccent>
ACCENT = re.compile(r"<\w+accent>\s*")

# =============================================================================
# PAUSES  -  <|1.00|>[pause]<|2.00|>
# =============================================================================

# <|1.00|>[pause]<|2.00|> -> captures start, end times
PAUSE_TIMED = re.compile(rf"<\|({_FLOAT})\|>\[pause\]<\|({_FLOAT})\|>")

# [pause] not preceded by timestamp
PAUSE_UNTIMED = re.compile(r"(?<!\|>)\[pause\]")

# =============================================================================
# INTERRUPTIONS  -  [1 yeah 1] or (1 mmhmm 1)
# =============================================================================

# [1 yeah 1] -> captures speaker, content
INTERRUPTION = re.compile(rf"\[({_DIGIT})\s+(.+?)\s+\1\]")

# (1 mmhmm 1) or (c applause c)
INTERRUPTION_PAREN = re.compile(rf"\(({_SPEAKER_OR_C})\s+(.+?)\s+\1\)")

# =============================================================================
# MUSIC  -  [music] ♪ ♫
# =============================================================================

# [music] at start with speaker: [music] [0]
MUSIC_SPEAKER = re.compile(r"^(\[music\]\s*\[(?:\d+|c)\]\s*)")

# ♪ or ♫
MUSIC_SYMBOL = re.compile(r"[\u266a\u266b]")

# =============================================================================
# WHITESPACE
# =============================================================================

# Multiple spaces -> one space
MULTI_SPACE = re.compile(r" +")

# Spaces/tabs
SPACE_TAB = re.compile(r"[ \t]+")

# =============================================================================
# CLEANING HELPERS
# =============================================================================

# <aq...> alignment quality tags
AQ_TAG = re.compile(r"<aq[^>]*>")

# (cut), (env), (mouthnoise) with optional timestamps
CUT_TAG = re.compile(rf"({_TIMESTAMP})?\(cut\)({_TIMESTAMP})?")
ENV_TAG = re.compile(rf"({_TIMESTAMP})?\(env\)({_TIMESTAMP})?")
MOUTHNOISE_TAG = re.compile(rf"({_TIMESTAMP})?\(mouthnoise\)({_TIMESTAMP})?")

# " ]" not followed by " ["
SPACE_BRACKET = re.compile(r" \](?! *\[)")

# ]]
DOUBLE_BRACKET = re.compile(r"\]\]")

# =============================================================================
# ALIGNMENT TIMING  -  <a.5|1.23|>word<a.5|1.50|>
# =============================================================================

_ATIMING = rf"<a\.{_DIGIT}\|{_FLOAT}\|>"  # <a.5|1.23|>
_ATIMING_SIMPLE = rf"<a\|{_FLOAT}\|>"  # <a|1.23|> (no score)

# <a.5|1.23|> -> captures score, time
SCORED_TIMING_PATTERN = re.compile(rf"<a\.({_DIGIT})\|({_FLOAT})\|>")

# <|1.00|><a.5|1.23|> -> captures the alignment timing
BEFORE_A_TIMING = re.compile(rf"{_TIMESTAMP}({_ATIMING})")

# <a.5|1.23|><|1.50|> -> captures the alignment timing
AFTER_A_TIMING = re.compile(rf"({_ATIMING}){_TIMESTAMP}")

# <a|0.22|>word<a|0.42|> -> captures full match (word with alignment timings)
# Only matches words (alphanumeric + common punctuation), not tags like [breath]
ATIMING_PAIR = re.compile(rf"({_ATIMING_SIMPLE}[\w.,!?'\"()-]+{_ATIMING_SIMPLE})")

# =============================================================================
# DECODING/INFERENCE PATTERNS
# =============================================================================

# ^[0 or ^[1 (incomplete speaker at string end)
BRACKET_DIGIT_START = re.compile(rf"^\[{_DIGIT}$")

# " [0" at string end
SPACE_BRACKET_DIGIT = re.compile(rf" \[{_DIGIT}$")

# [laugh (incomplete tag at string end)
BRACKET_TAG_END = re.compile(r"\[[a-z]+$")

# [0  (start of interruption)
INTERRUPTION_START = re.compile(rf"\[{_DIGIT}\s")

#  0] (end of interruption)
INTERRUPTION_END = re.compile(rf"\s{_DIGIT}\]")

# Just digits
DIGIT_MATCH = re.compile(_DIGIT)

# Optional space + digits
SPACE_DIGIT_MATCH = re.compile(rf"\s?{_DIGIT}")

# [0, [1 prefix
BRACKET_DIGIT_PREFIX = re.compile(rf"\[{_DIGIT}")

# [0 ... incomplete at end
INTERRUPTION_UNMATCHED = re.compile(rf"\[{_DIGIT}+\s.+$")

# =============================================================================
# TIMED INTERRUPTIONS  -  <|1.00|>[0]<|1.50|>
# =============================================================================

#  <|t|>[0]?<|t|>
TIMED_INT = re.compile(rf"\s+{_TIMESTAMP}{_SPEAKER}?{_TIMESTAMP}")

#  <|t|>[0]<|t|>
TIMED_INT2 = re.compile(rf"\s+{_TIMESTAMP}\[?{_DIGIT}\]{_TIMESTAMP}")

# =============================================================================
# OVERLAP PATTERNS  -  [1 word word 1]
# =============================================================================

# Content that allows nested [tag] patterns: matches non-brackets OR complete [lowercase] tags
_CONTENT_WITH_TAGS = r"(?:[^\[\]]|\[[a-z/]+\])+"

# [1 word 1] or [1 word] - handles nested [breath], [laugh], [/whisper] etc.
OVERLAP_WITH_TIMINGS_REGEX = re.compile(
    rf"\[({_DIGIT})\s+({_CONTENT_WITH_TAGS})(?:\s+\1)?\]"
)

# (1 word 1) or (c word c)
OVERLAP_WITH_TIMINGS_REGEX_PAREN = re.compile(
    rf"\(({_SPEAKER_OR_C})\s+({_CONTENT_WITH_TAGS})(?:\s+\1)?\)"
)

# Strict: requires matching close [1 word 1]
INTERRUPTION_STRICT_REGEX = re.compile(rf"\[({_DIGIT})\s+({_CONTENT_WITH_TAGS})\s+\1\]")

# Strict paren: (1 word 1)
INTERRUPTION_STRICT_REGEX_PAREN = re.compile(
    rf"\(({_SPEAKER_OR_C})\s+({_CONTENT_WITH_TAGS})\s+\1\)"
)

# [1 ... 1] any content
INTERRUPTIONS = re.compile(rf"\[({_DIGIT})\s{_CONTENT_WITH_TAGS}\s\1\]")

# =============================================================================
# MUSIC TIMING PATTERNS
# =============================================================================

# <|t|>[music]
MUSIC_TAG_PATTERN = re.compile(rf"{_TIMESTAMP}\[music\]\s")

# [stopmusic]<|t|>
MUSIC_END_TAG_PATTERN = re.compile(rf"\s\[stopmusic\]{_TIMESTAMP}")

# =============================================================================
# ERROR DETECTION PATTERNS
# =============================================================================

# [laugh (unclosed tag before space/timestamp)
UNCLOSED_TAGS = re.compile(r"\[[a-z]+(?!\])(?=[\s<])")

# <|t|>[tag]<|t|> repeated 5+ times (hallucination)
HALLUCINTION = re.compile(
    r"(?:<\|\d+\.\d+\|>\[([a-z]+)\]<\|\d+\.\d+\|>(\s?\d+?\s?)){5,}"
)

# Lines with only nonspeech tags
LINES_WITH_ONLY_NONSPEECH = re.compile(
    rf"(?:^|\n)(?:\[{_DIGIT}\](?:\s+\[[a-z]+\])*\s*)(?:\n|$)"
)

# =============================================================================
# DATASET FORMAT PATTERNS
# =============================================================================

# " [0 " -> " (0 "
BRACKET_PARENTHESES_PATTERN = re.compile(rf" \[((?:{_DIGIT}|c)) ")

# " 0] " -> " 0) "
PARENTHESES_BRACKET_PATTERN = re.compile(rf" ((?:{_DIGIT}|c))\] ")

#  <|t|> <|t|> (timestamp with whitespace around)
TIMESTAMP_REGEX_SPACE = re.compile(rf"\s+{_TIMESTAMP}\s+")

# Exact validation: <|1.23|>
TIMESTAMP_VALIDATION_PATTERN = re.compile(r"^<\|\d+\.\d{2}\|>$")

# (0 or (c extract
SPEAKER_ID_EXTRACT_PAREN = re.compile(rf"\(({_SPEAKER_OR_C})")

# [tag] lowercase
NS = re.compile(r"\[[a-z]+]")

# =============================================================================
# LEGACY ALIASES (backwards compatibility)
# =============================================================================

TIMESTAMP_REGEX = TIMINGS_FORMAT
TIMING = TIMINGS_FORMAT
SPEAKER_FORMAT = SPEAKER
SPEAKER_ID_EXTRACT = re.compile(rf"\[({_SPEAKER_OR_C})")
BRACKET_DIGIT_MATCH = SPEAKER
WHITESPACE_NORMALIZE = MULTI_SPACE
NON_SPEECH_TAG = NONSPEECH
NON_SPEECH_TAG_ANY_CASE = NONSPEECH_ANY
TIMESTAMPED_PAUSE = PAUSE_TIMED
UNTIMESTAMPED_PAUSE = PAUSE_UNTIMED
SPEAKER_TAG_FULL = SPEAKER_FULL
MUSIC_MARKER = MUSIC_SYMBOL
INTERRUPTION_START_PATTERN = INTERRUPTION_START
INTERRUPTION_END_PATTERN = INTERRUPTION_END
INTMARKS = INTERRUPTIONS
METADATA_TAG = METADATA
EMOTION_TAG = EMOTION
ACCENT_TAG = ACCENT
NON_SPEECH_TAG_CAPTURE = re.compile(r"\[([a-zA-Z]+)\]")  # [laugh] -> "laugh"

# Speaker positioning
SPEAKER_AT_LINE_END = re.compile(rf"\[(?:{_DIGIT}|c)\]\s*$")  # [0] at end
SPEAKER_NOT_AT_LINE_START = re.compile(rf"(?<!\n)\[(?:{_DIGIT}|c)\]\s*$")
SPEAKER_TAGS_GENERAL = re.compile(rf"(?<=[^\n])\[(?:{_DIGIT}|c)\]\s*")

# Laughter patterns
LAUGHTER_HA_PATTERN = re.compile(r"\b(?:h[ha]*)+h?\b", re.IGNORECASE)  # haha, hahaha
REPEATING_HA = re.compile(r"\b(he){2,}\b", re.IGNORECASE)  # hehe

# Music tag patterns
MUSIC_NEWLINE_PATTERN = re.compile(rf"\[music\](.*?)\[({_DIGIT})\]")
MUSIC_SPAN_PATTERN = re.compile(r"\[music\]\s?(?:<\|\d+\.\d+\|>\s?){2}")
MUSIC_TAG_ORDER_PATTERN = re.compile(rf"{_TIMESTAMP}\s*\[music\](.*?)\[({_DIGIT})\]")
MUSIC_START_TAG = re.compile(rf"{_TIMESTAMP}\[music\]")  # <|t|>[music]
MUSIC_END_TAG = re.compile(rf"\[stopmusic\]{_TIMESTAMP}")  # [stopmusic]<|t|>
TIMINGS_FORMAT_NO_MUSIC = re.compile(rf"(?<!\[stopmusic\]){_TIMESTAMP}(?!\[music\])")

# Other patterns
ENDOFTEXT_PATTERN = re.compile(r"endoftext", re.IGNORECASE)

# Base pattern exports (for files that need raw patterns)
_DIGIT_PATTERN = _DIGIT
_FLOAT_PATTERN = _FLOAT
_TIMING_CORE = _TIMESTAMP
_WHITESPACE = r"\s+"

# Aliases
SMARKS = SPEAKER
QUANTIZE_TIMESTAMP_PATTERN = TIMINGS_FORMAT

# =============================================================================
# TAG CONSTANTS
# =============================================================================

BREATH = "[breath]"
HESITATION = "[hesitate]"
TUT = "[tut]"
PUNC = [" ", ".", ",", "?", "!", "...", "\n", "-"]

# =============================================================================
# CONSTANTS
# =============================================================================

ADDITIONAL_DIACRITICS = {
    "œ": "oe",
    "Œ": "OE",
    "ø": "o",
    "Ø": "O",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
    "ẞ": "SS",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "th",
    "ł": "l",
    "Ł": "L",
}

# =============================================================================
# WER CLEANING PATTERNS
# =============================================================================

BRACKETS_GENERAL = re.compile(r"\[.*?\]")
ANGLE_BRACKETS = re.compile(r"«.*?»")
DOUBLE_ANGLE_BRACKETS = re.compile(r"《.*?》")
INCOMPLETE_WORDS = re.compile(r"\b\w*-\s")
STAMMERING = re.compile(r"\b(\w+)(?:-\1)+\b")
