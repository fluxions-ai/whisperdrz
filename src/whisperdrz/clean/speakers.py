"""
Speaker tag handling functions for WhisperDRZ transcription processing.

This module handles:
- Fixing and normalizing speaker IDs
- Removing trailing speaker tags
- Removing speaker tags not at line starts
"""

import re

from .patterns import (
    SPEAKER_LINE_START,
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
            m = SPEAKER_LINE_START.match(line)
            if m and merged:
                prev_m = SPEAKER_LINE_START.match(merged[-1])
                if prev_m and prev_m.group(1) == m.group(1):
                    merged[-1] = merged[-1].rstrip() + " " + line[m.end() :]
                    continue
            merged.append(line)
        result.append("\n".join(merged))
    return "[CHOP]".join(result)


_SPEAKER_PREFIX = re.compile(r"^\[(\d+|c)\]\s*")
_TIMESTAMP = re.compile(r"<\|(\d+(?:\.\d+)?)\|>")
_ANY_TAG = re.compile(r"\[[^\]]*\]|<[^>|]*>")
_LINE_SPLIT = re.compile(r"(\n|\[CHOP\])")


def _line_content(line: str):
    """A speaker line's time span (first to last timestamp, None if untimed), its words
    (lowercased, punctuation stripped) and whether it carries any [tag] or <tag>."""
    m = _SPEAKER_PREFIX.match(line)
    body = line[m.end() :] if m else line
    ts = [float(t) for t in _TIMESTAMP.findall(body)]
    untimed = _TIMESTAMP.sub(" ", body)
    has_tag = bool(_ANY_TAG.search(untimed))
    words = [re.sub(r"[^\w']", "", w.lower()) for w in _ANY_TAG.sub(" ", untimed).split()]
    return ((min(ts), max(ts)) if ts else None), [w for w in words if w], has_tag


def _drop_parts(parts: list[str], drop: set[int]) -> str:
    """Join re.split(_LINE_SPLIT) parts without the dropped lines (and a newline beside each)."""
    parts = list(parts)
    for k in drop:
        parts[k] = ""
        if k + 1 < len(parts) and parts[k + 1] == "\n":
            parts[k + 1] = ""
        elif k > 0 and parts[k - 1] == "\n":
            parts[k - 1] = ""
    return "".join(parts)


def drop_contentless_lines(text: str) -> str:
    """Drop speaker lines that say nothing: a bare `[3]`, or `[0] <|t|> <|t|>` (timestamps only).
    Lines with a tag (`[0] [laugh]`, `[c] [music] <|t|> <|t|> [/music]`) stay."""
    parts = _LINE_SPLIT.split(text)
    drop = set()
    for k in range(0, len(parts), 2):
        line = parts[k].strip()
        if _SPEAKER_PREFIX.match(line):
            _, words, has_tag = _line_content(line)
            if not words and not has_tag:
                drop.add(k)
    return _drop_parts(parts, drop) if drop else text


def drop_overlapping_copy_lines(text: str, min_words: int = 3, min_overlap: float = 0.5) -> str:
    """Drop a speaker line the decoder wrote twice: one window's text, where a line of
    `min_words`+ words says words another line also says (one contains the other), over the
    same stretch of time (their spans overlap by at least `min_overlap` of the shorter).
    Of the pair the shorter goes, or the later when they are the same length.

    The decoder sometimes writes one utterance on several lines under different speaker tags
    with nearly the same timestamps ("I'm not sure if I can know." x13 on an ICSI meeting);
    the speaker remap then puts the copies in one cluster and the merge glues them into a
    doubled line. Real repeats are said one after the other (a second person echoing, a
    self-repair, "that's great. that's great."), so they don't overlap in time and stay.
    On 17 meetings (docs/inference_artifacts.md) it removed 68 lines; speaker-blind WER
    fell ~0.5 points (ICSI 6 cpWER points), no recording worse by more than 0.1.
    """
    parts = _LINE_SPLIT.split(text)
    info = {}
    for k in range(0, len(parts), 2):
        line = parts[k].strip()
        if _SPEAKER_PREFIX.match(line):
            span, words, _ = _line_content(line)
            if span and len(words) >= min_words:
                info[k] = (span, words)
    keys = sorted(info)
    drop: set[int] = set()
    for x, i in enumerate(keys):
        if i in drop:
            continue
        sa, wa = info[i]
        for j in keys[:x]:
            if j not in drop and _is_copy(sa, wa, *info[j], min_overlap):
                drop.add(i if len(wa) <= len(info[j][1]) else j)
                break
    return _drop_parts(parts, drop) if drop else text


def _is_copy(sa, wa, sb, wb, min_overlap: float) -> bool:
    """Two lines (span, words) say the same words (one contains the other) over the same time."""
    overlap = min(sa[1], sb[1]) - max(sa[0], sb[0])
    if overlap < min_overlap * max(0.05, min(sa[1] - sa[0], sb[1] - sb[0])):
        return False
    short, long_ = (wa, wb) if len(wa) <= len(wb) else (wb, wa)
    n = len(short)
    return any(long_[p : p + n] == short for p in range(len(long_) - n + 1))

