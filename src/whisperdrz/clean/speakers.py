"""
Speaker tag handling functions for WhisperDRZ transcription processing.

This module handles:
- Fixing and normalizing speaker IDs
- Removing trailing speaker tags
- Removing speaker tags not at line starts
"""

import re

from .patterns import (
    SPEAKER_TAGS_GENERAL,
)


def fix_speaker_ids(text):
    """Ensure each ID is incrementing by one as they occur."""
    speakers_in_order = []
    for m in re.finditer(r"\[(\d+)\]", text):
        s = m.group(1)
        if s not in speakers_in_order:
            speakers_in_order.append(s)

    if not speakers_in_order:
        return text

    mapping = {s: str(i) for i, s in enumerate(speakers_in_order)}

    def _replace(m):
        return f"[{mapping[m.group(1)]}]"

    return re.sub(r"\[(\d+)\]", _replace, text)


def remove_speaker_tags_not_after_newline(text):
    """Remove ALL speaker tags [0], [1], [2], etc. that don't follow a newline"""
    return SPEAKER_TAGS_GENERAL.sub("", text)


_SPEAKER_PREFIX = re.compile(r"^\[(\d+|c)\]\s*")


def merge_consecutive_speakers(text: str) -> str:
    """Merge consecutive lines from the same speaker, skipping blank lines."""
    chunks = text.split("[CHOP]")
    result = []
    for chunk in chunks:
        lines = chunk.split("\n")
        merged = []
        for line in lines:
            if not line.strip():
                continue
            m = _SPEAKER_PREFIX.match(line)
            if m and merged:
                prev_m = _SPEAKER_PREFIX.match(merged[-1])
                if prev_m and prev_m.group(1) == m.group(1):
                    merged[-1] = merged[-1].rstrip() + " " + line[m.end() :]
                    continue
            merged.append(line)
        result.append("\n".join(merged))
    return "[CHOP]".join(result)
