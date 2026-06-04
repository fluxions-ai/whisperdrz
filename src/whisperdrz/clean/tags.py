"""
Tag handling functions for WhisperDRZ transcription processing.

This module handles:
- Non-speech tags like [breath], [laugh], [music], etc.
- Interruption markers like [1 word word 1]
- Tag spacing and whitespace normalization
- Music tag ordering and newline handling
- Laughter pattern normalization
"""

import re

from .patterns import (
    _DIGIT_PATTERN,
)

# Pre-compiled patterns for split_interruptions_to_speaker_turns (performance optimization)
_SPEAKER_AT_START_PATTERN = re.compile(rf"^\[({_DIGIT_PATTERN})\]")
_INTERRUPTION_PATTERN = re.compile(rf"\[({_DIGIT_PATTERN})\s")
_MULTI_SPACE_PATTERN = re.compile(r"  +")

# Pre-compiled patterns for strip_tag_whitespace
_BRACKET_CONTENT_PAT = re.compile(r"\[([^\]]+)\]")
_INTERRUPTION_CHECK_PAT = re.compile(r"^(\d+)\s.+\s\1$")
_DIGIT_SPACE_START_PAT = re.compile(r"^\d+\s")
_INNER_WHITESPACE_PAT = re.compile(r"\s+")

# Tags that have opening/closing pairs like [singing]...[/singing]
# These should stay as brackets, not be converted to parentheses
PAIRED_TAGS = frozenset(
    {
        "accent",
        "crowdcheering",
        "dc",
        "effect",
        "laughing",
        "mispeak",
        "music",
        "reverb",
        "screaming",
        "shout",
        "shouting",
        "sing",
        "singing",
        "voicebreaking",
        "whisper",
    }
)

# Pre-compiled patterns for remove_orphan_paired_tags (performance optimization)
# Compile once at module load instead of per-line
_OPENING_PAIRED_TAG_PATTERN = re.compile(
    r"\[(" + "|".join(PAIRED_TAGS) + r")\]", re.IGNORECASE
)
_CLOSING_PAIRED_TAG_PATTERNS = {
    tag: re.compile(
        r"\[(?:stop" + tag + r"|" + tag + r"stop|/" + tag + r")\]",
        re.IGNORECASE,
    )
    for tag in PAIRED_TAGS
}


def remove_orphan_paired_tags(text: str) -> str:
    """
    Remove orphan opening/closing paired tags that don't have a matching pair on the same line.

    Paired tag formats:
        Opening: [singing], [music], [laughing], etc.
        Closing: [stopsing], [/singing], [singstop], etc.

    Examples:
        "word [stopsing] more" -> "word more" (no [sing] before)
        "[singing] text" -> "text" (no [stopsing] or [/singing] after)
        "[singing] text [stopsing]" -> "[singing] text [stopsing]" (valid pair)

    Args:
        text: Text with paired tags

    Returns:
        Text with orphan paired tags removed
    """
    lines = text.split("\n")
    result_lines = []

    for line in lines:
        # Find all opening tags (uses pre-compiled pattern)
        opening_matches = list(_OPENING_PAIRED_TAG_PATTERN.finditer(line))

        # For each opening tag, check if there's a matching closing tag after it
        orphan_spans = []
        used_closings = set()

        for m in opening_matches:
            tag_name = m.group(1).lower()
            closing_pat = _CLOSING_PAIRED_TAG_PATTERNS.get(tag_name)
            if closing_pat:
                # Find closing tags after this opening
                found_closing = False
                for cm in closing_pat.finditer(line):
                    if cm.start() > m.end() and cm.start() not in used_closings:
                        found_closing = True
                        used_closings.add(cm.start())
                        break
                if not found_closing:
                    orphan_spans.append((m.start(), m.end()))

        # Find all closing tags and check for matching opening before them
        for tag_name, closing_pat in _CLOSING_PAIRED_TAG_PATTERNS.items():
            for cm in closing_pat.finditer(line):
                if cm.start() in used_closings:
                    continue
                # Check if there's an opening tag before this closing
                opening_before = False
                for m in opening_matches:
                    if m.group(1).lower() == tag_name and m.end() <= cm.start():
                        # Check if this opening isn't already marked as orphan
                        if (m.start(), m.end()) not in orphan_spans:
                            opening_before = True
                            break
                if not opening_before:
                    orphan_spans.append((cm.start(), cm.end()))

        # Remove orphan spans from end to start to preserve positions
        orphan_spans.sort(reverse=True)
        for start, end in orphan_spans:
            line = line[:start] + line[end:]

        # Clean up extra whitespace from removals
        line = _MULTI_SPACE_PATTERN.sub(" ", line)
        line = line.strip()

        result_lines.append(line)

    return "\n".join(result_lines)


def consolidate_adjacent_tags(text: str) -> str:
    """
    Merge consecutive identical non-speech tags into a single tag.

    Handles both timestamped and non-timestamped tags:
        <|5.14|>[breath]<|5.24|> <|5.26|>[breath]<|5.30|> -> <|5.14|>[breath]<|5.30|>
        [breath] [breath] [breath] -> [breath]

    Args:
        text: Text with tags (timestamped or not)

    Returns:
        Text with adjacent identical tags consolidated
    """
    # First pass: merge timestamped tags
    # Pattern to match: <|t1|>[tag]<|t2|>
    timed_tag_pattern = re.compile(r"<\|(\d+\.?\d*)\|>\[([^\]]+)\]<\|(\d+\.?\d*)\|>")

    def merge_timed_pass(txt):
        """Single pass to merge adjacent timestamped tags. Returns (new_text, changed)."""
        matches = list(timed_tag_pattern.finditer(txt))
        if len(matches) < 2:
            return txt, False

        result = []
        last_end = 0
        i = 0
        changed = False

        while i < len(matches):
            m1 = matches[i]

            if i + 1 < len(matches):
                m2 = matches[i + 1]
                between = txt[m1.end() : m2.start()]

                tag1, tag2 = m1.group(2), m2.group(2)

                if tag1 == tag2 and between.strip() == "":
                    t1_start = m1.group(1)
                    t2_end = m2.group(3)
                    result.append(txt[last_end : m1.start()])
                    result.append(f"<|{t1_start}|>[{tag1}]<|{t2_end}|>")
                    last_end = m2.end()
                    i += 2
                    changed = True
                    continue

            result.append(txt[last_end : m1.end()])
            last_end = m1.end()
            i += 1

        result.append(txt[last_end:])
        return "".join(result), changed

    # Keep merging timestamped tags until no more changes
    while True:
        text, changed = merge_timed_pass(text)
        if not changed:
            break

    # Second pass: merge non-timestamped adjacent identical tags
    # Pattern: [tag] followed by spaces (not newlines) and same [tag]
    # Excludes speaker tags [0], [1], [c] and interruption markers [1 ... 1]
    def merge_untimed_pass(txt):
        """Merge adjacent non-timestamped identical tags on the same line."""
        # Match [tag] followed by spaces/tabs (no newlines) and same [tag]
        # Only matches alphabetic tags (not speaker IDs like [0], [1], [c])
        untimed_pattern = re.compile(
            r"\[([a-zA-Z][a-zA-Z]*)\]"  # [tag] where tag starts with letter
            r"[ \t]+"  # spaces/tabs only (no newlines)
            r"\[\1\]"  # same [tag]
        )

        while True:
            new_txt = untimed_pattern.sub(r"[\1]", txt)
            if new_txt == txt:
                break
            txt = new_txt

        return txt

    # Apply untimed merging
    return merge_untimed_pass(text)
