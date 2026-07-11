"""
Timing manipulation and validation functions for WhisperDRZ transcription processing.

This module handles:
- Parsing and formatting timing markers (<|time|>)
- Fixing malformed timing markers
- Quantizing and scaling timestamps
- Removing and managing timestamp pairs
- Detecting and handling cutoff tags
- Reconciling decoder and alignment timings
"""

import re

from .patterns import (
    _DIGIT_PATTERN,
    _FLOAT_PATTERN,
    _TIMING_CORE,
    SCORED_TIMING_PATTERN,
    TIMINGS_FORMAT,
)


def parse_timing(t: str) -> float:
    """Parse timing string to float."""
    return float(t.replace("<|", "").replace("|>", ""))


def cut_after_last_timed_word(input_string: str) -> str:
    """Cut text after the last timed word."""
    matches = list(
        re.finditer(rf"{_TIMING_CORE}([^\s<\[]+){_TIMING_CORE}", input_string)
    )
    if matches:
        return input_string[: matches[-1].end()]
    return input_string


def reconcile_decoder_and_alignment_timings(
    text: str, confidence_threshold: int = 92
) -> str:
    """
    Reconcile decoder and alignment timings in the pattern: <|d|><a.score|a|>word<a.score|a|><|d|>

    Rule:
    - If score >= threshold: Remove decoder timings, convert alignment to standard format
    - If score < threshold: Remove alignment timings, keep decoder timings
    - Keep all other timings unchanged

    Args:
        text: Text with mixed timings like "[0] <|0.08|><a.96|0.14|>word<a.96|0.20|><|0.20|>"
        confidence_threshold: Minimum alignment score to prefer over decoder timing

    Returns:
        Text with reconciled timings like "[0] <|0.14|>word<|0.20|>"
    """
    pattern = rf"(<\|{_FLOAT_PATTERN}\|>)(<a\.{_DIGIT_PATTERN}\|{_FLOAT_PATTERN}\|>)(.*?)(<a\.{_DIGIT_PATTERN}\|{_FLOAT_PATTERN}\|>)(<\|{_FLOAT_PATTERN}\|>)"

    def replace_func(match):
        decoder_start = match.group(1)
        align_start = match.group(2)
        word_content = match.group(3)
        align_end = match.group(4)
        decoder_end = match.group(5)

        score_match = re.search(rf"<a\.({_DIGIT_PATTERN})\|", align_start)
        if not score_match:
            return match.group(0)

        score = int(score_match.group(1))

        if score >= confidence_threshold:
            time_start = re.search(
                rf"<a\.{_DIGIT_PATTERN}\|({_FLOAT_PATTERN})\|>", align_start
            ).group(1)
            time_end = re.search(
                rf"<a\.{_DIGIT_PATTERN}\|({_FLOAT_PATTERN})\|>", align_end
            ).group(1)
            return f"<|{time_start}|>{word_content}<|{time_end}|>"
        return f"{decoder_start}{word_content}{decoder_end}"

    result = re.sub(pattern, replace_func, text)

    return SCORED_TIMING_PATTERN.sub(lambda m: f"<|{float(m.group(2)):.2f}|>", result)


_EXTRA_TS_PAT1 = re.compile(rf"{_TIMING_CORE}+({_TIMING_CORE}\w+)")
_EXTRA_TS_PAT2 = re.compile(rf"(\w+{_TIMING_CORE})-?{_TIMING_CORE}+")


def round_to(x: float, step: float):
    return round(x / step) * step


def quantize_timestamp(match, step: float, offset: float = 0.0):
    time = float(match.group(1))
    quantized = round_to(time, step)
    offset = round_to(offset, step)
    quantized = quantized + offset
    if offset == 0:
        quantized = min(quantized, 30.0)
    return f"<|{quantized:.2f}|>"


def quantize_timestamps(text: str, offset: float = 0.0, step: float = 0.02):
    def repl(match):
        return quantize_timestamp(match, step, offset=offset)

    return TIMINGS_FORMAT.sub(repl, text)


def offset_timestamps(text: str, offset: float) -> str:
    """Shift all timestamps by a fixed offset (for concatenating audio segments).

    Args:
        text: Text containing timestamps in <|time|> format
        offset: Seconds to add to all timestamps

    Returns:
        Text with shifted timestamps
    """

    def repl(match):
        time = float(match.group(1))
        new_time = max(0.0, time + offset)  # Never go negative
        new_time = round_to(new_time, 0.02)
        return f"<|{new_time:.2f}|>"

    return TIMINGS_FORMAT.sub(repl, text)


def remove_invalid_timing_pairs(text: str, min_duration: float = 0.02) -> str:
    """
    Remove timing pairs that are invalid:
    - Duration less than min_duration seconds (t2 - t1 < min_duration)
    - Second timestamp not greater than first (t2 <= t1)

    Only operates on actual timing pairs (timestamps wrapping content with <=2 spaces).
    Stray/unpaired timestamps are left untouched as they influence segment durations.

    Args:
        text: Text containing timestamp pairs in <|time|> format
        min_duration: Minimum duration in seconds for a valid pair (default 0.02)

    Returns:
        Text with invalid timing pairs removed

    Examples:
        "<|1.00|>word<|1.02|>" -> "word" (duration 0.02 <= 0.02)
        "<|1.50|>word<|1.40|>" -> "word" (t2 < t1)
        "<|1.00|>word<|1.10|>" -> "<|1.00|>word<|1.10|>" (valid, duration 0.10)
        "<|1.00|>\n[0] <|2.00|>word<|2.02|>" -> "<|1.00|>\n[0] word" (stray timing preserved)
    """
    matches = list(TIMINGS_FORMAT.finditer(text))

    if len(matches) < 2:
        return text

    # Find actual timing pairs: consecutive timestamps with <=2 spaces between
    # (matching the logic in timings_format_correct)
    pairs_to_remove = []
    used_indices = set()

    for i in range(len(matches) - 1):
        if i in used_indices:
            continue

        t1_match = matches[i]
        t2_match = matches[i + 1]

        # Check if this is a valid pair (content between has <=2 spaces and no newlines)
        between = text[t1_match.end() : t2_match.start()]
        if between.count(" ") > 2 or "\n" in between:
            # Not a pair, t1 is a stray timing (newline = line-ending timestamp)
            continue

        # This is a timing pair
        used_indices.add(i)
        used_indices.add(i + 1)

        t1 = float(t1_match.group(1))
        t2 = float(t2_match.group(1))
        duration = t2 - t1

        # Mark for removal if duration is too short or negative/zero
        if duration <= min_duration:
            pairs_to_remove.append((t1_match, t2_match))

    # Remove pairs in reverse order to preserve positions
    for t1_match, t2_match in reversed(pairs_to_remove):
        # Remove second timestamp first (higher position)
        text = text[: t2_match.start()] + text[t2_match.end() :]
        # Then remove first timestamp
        text = text[: t1_match.start()] + text[t1_match.end() :]

    return text


def fix_nonmonotonic_timestamps(text: str) -> str:
    """Fix backwards timestamps within each line.

    Within a single line, timestamps must be monotonically non-decreasing.
    If a timestamp goes backwards, clamp it to the previous value + 0.02s.

    Examples:
        "<|26.68|>I'm<|26.78|> <|26.08|>doing.<|27.08|>"
        -> "<|26.68|>I'm<|26.78|> <|26.80|>doing.<|27.08|>"
    """
    result_lines = []
    for line in text.split("\n"):
        matches = list(TIMINGS_FORMAT.finditer(line))
        if len(matches) < 2:
            result_lines.append(line)
            continue

        last_t = float(matches[0].group(1))
        replacements = []
        for m in matches[1:]:
            t = float(m.group(1))
            if t < last_t:
                new_t = last_t + 0.02
                replacements.append((m.start(), m.end(), f"<|{new_t:.2f}|>"))
                last_t = new_t
            else:
                last_t = t

        if replacements:
            for start, end, repl in reversed(replacements):
                line = line[:start] + repl + line[end:]

        result_lines.append(line)
    return "\n".join(result_lines)


def remove_short_nonspeech(text: str, min_duration_s: float = 0.02) -> str:
    """Remove non-speech tags with duration <= min_duration_s.

    Matches tags like [breath], [laugh] etc. with surrounding timings.

    Example:
        "<|1.00|>[breath]<|1.01|>" -> "" (10ms, removed)
        "<|1.00|>[breath]<|1.10|>" -> kept (100ms)
    """

    def replace_short_tag(match):
        t1 = float(match.group(1))
        t2 = float(match.group(3))
        if t2 - t1 <= min_duration_s + 0.001:  # Small epsilon for float precision
            return ""
        return match.group(0)

    return re.sub(
        r"<\|([\d.]+)\|>\s*(\[[a-z]+\])\s*<\|([\d.]+)\|>", replace_short_tag, text
    )


def remove_short_pauses(text: str, min_pause_ms: int = 800) -> str:
    """Remove [pause] tags that represent pauses shorter than min_pause_ms.

    Matches [pause] tags with optional surrounding timings like:
        <|1.00|>[pause]<|1.50|> - 500ms pause, removed if < 800ms

    Example:
        "<|1.00|>[pause]<|1.50|>" with min_pause_ms=800 -> "<|1.50|>" (500ms removed, keep end timing)
        "<|1.00|>[pause]<|2.00|>" with min_pause_ms=800 -> "<|1.00|>[pause]<|2.00|>" (1000ms kept)
    """
    min_pause_s = min_pause_ms / 1000.0

    def replace_short_pause(match):
        t1 = float(match.group(1))
        t2 = float(match.group(2))
        gap = t2 - t1
        if gap < min_pause_s:
            # Keep ending timestamp so next word is still timed
            return f"<|{t2:.2f}|>"
        return match.group(0)

    return re.sub(
        r"<\|([\d.]+)\|>\s*\[pause\]\s*<\|([\d.]+)\|>", replace_short_pause, text
    )


_SPEAKER_LINE_PATTERN = re.compile(r"^\[(\d+|c)\]", re.MULTILINE)
_LINE_STARTS_WITH_TIMESTAMP = re.compile(r"^\[(?:\d+|c)\]\s*<\|\d+\.\d+\|>")
_LINE_ENDS_WITH_TIMESTAMP = re.compile(r"<\|\d+\.\d+\|>\s*$")

