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
    TIMESTAMP_WORD,
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


def _line_timing_units(line: str) -> list[tuple[float, float, list[tuple[int, int]]]]:
    """Group a line's timestamps into droppable units.

    A unit is a `<|s|>word<|e|>` pair (both timestamps drop together, keeping
    the word) or a lone timestamp. Returns (t_first, t_last, spans) per unit,
    where spans are the character ranges of just the timestamp tokens.
    """
    units: list[tuple[float, float, list[tuple[int, int]]]] = []
    covered: list[tuple[int, int]] = []
    for m in TIMESTAMP_WORD.finditer(line):
        t0, t1 = float(m.group(1)), float(m.group(3))
        ts0_end = m.start() + len(m.group(1)) + 4  # <| ... |>
        ts1_start = m.end() - (len(m.group(3)) + 4)
        units.append((t0, t1, [(m.start(), ts0_end), (ts1_start, m.end())]))
        covered.append((m.start(), m.end()))
    for m in TIMINGS_FORMAT.finditer(line):
        if any(s <= m.start() < e for s, e in covered):
            continue
        t = float(m.group(1))
        units.append((t, t, [(m.start(), m.end())]))
    units.sort(key=lambda u: u[2][0][0])
    return units


def fix_nonmonotonic_timestamps(text: str, max_time: float | None = None) -> str:
    """Make each line's timestamps monotonic by DROPPING outlier timing units.

    The old behaviour clamped a backwards timestamp to the previous value
    + 0.02s — so a single hallucinated far-forward timestamp poisoned every
    later (correct, lower) timestamp on the line, flattening whole lines into
    0.02s staircases. Instead, keep the longest non-decreasing chain of timing
    units and strip the rest; their words simply become untimed, which is
    valid WhisperDRZ format.

    max_time: when given (the audio length of the window), units starting
    beyond it are dropped outright — they can't be real no matter how
    internally consistent they look.

    Example (the backwards unit is dropped, its neighbours keep their times):
        "<|26.68|>I'm<|26.78|> <|26.08|>doing.<|27.08|> <|27.08|>ok<|27.30|>"
        -> "<|26.68|>I'm<|26.78|> doing. <|27.08|>ok<|27.30|>"
    """
    result_lines = []
    for line in text.split("\n"):
        units = _line_timing_units(line)
        if not units:
            result_lines.append(line)
            continue

        candidates = []  # indices eligible for the monotonic chain
        forced_drop = set()
        for i, (t0, t1, _spans) in enumerate(units):
            if (
                t1 < t0 - 0.001  # internally backwards pair
                or (max_time is not None and t0 > max_time + 0.02)
                # 0.5s slack: a word straddling the chunk boundary may
                # legitimately end slightly past the audio; a far-out end is bogus.
                or (max_time is not None and t1 > max_time + 0.5)
            ):
                forced_drop.add(i)
            else:
                candidates.append(i)

        # Longest non-decreasing chain over candidate units (O(n^2), n small).
        # Tie-break on smaller ending time so impossible late tails lose.
        best_len = [0] * len(units)
        best_prev = [-1] * len(units)
        for ci, i in enumerate(candidates):
            best_len[i] = 1
            for j in candidates[:ci]:
                if units[i][0] >= units[j][1] - 0.001 and (
                    best_len[j] + 1 > best_len[i]
                    or (
                        best_len[j] + 1 == best_len[i]
                        and best_prev[i] != -1
                        and units[j][1] < units[best_prev[i]][1]
                    )
                ):
                    best_len[i] = best_len[j] + 1
                    best_prev[i] = j
        keep: set[int] = set()
        if candidates:
            end = max(
                candidates,
                key=lambda i: (best_len[i], -units[i][1]),
            )
            while end != -1:
                keep.add(end)
                end = best_prev[end]

        drops = [i for i in range(len(units)) if i not in keep]
        if not drops:
            result_lines.append(line)
            continue
        spans_to_remove = sorted(
            (span for i in drops for span in units[i][2]), reverse=True
        )
        for s, e in spans_to_remove:
            line = line[:s] + line[e:]
        line = re.sub(r" {2,}", " ", line).rstrip()
        result_lines.append(line)
    return "\n".join(result_lines)


def remove_short_nonspeech(text: str, min_duration_s: float = 0.02) -> str:
    """Remove non-speech tags with duration <= min_duration_s.

    Only a tag wrapped in its OWN pair (`<|t1|>[tag]<|t2|>`, no spaces) is
    removed, together with both timestamps. Allowing whitespace let the pattern
    straddle an untimed tag between two words and delete one word's closing and
    the next word's opening timestamp, welding them: `Hi<|0.30|> [breath]
    <|0.32|>there` -> `Hithere`.

    Example:
        "a<|0.9|> <|1.00|>[breath]<|1.01|> <|1.1|>b" -> "a<|0.9|> <|1.1|>b" (10ms, removed)
        "<|1.00|>[breath]<|1.10|>" -> kept (100ms)
    """

    def replace_short_tag(match):
        t1 = float(match.group(2))
        t2 = float(match.group(4))
        if t2 - t1 <= min_duration_s + 0.001:  # Small epsilon for float precision
            return ""
        return match.group(0)

    return re.sub(
        r"( ?)<\|([\d.]+)\|>(\[[a-z]+\])<\|([\d.]+)\|>", replace_short_tag, text
    )


def remove_short_pauses(text: str, min_pause_ms: int = 800) -> str:
    """Remove [pause] tags that represent pauses shorter than min_pause_ms.

    The whole `<|t1|>[pause]<|t2|>` pair goes, both timestamps with it. The old
    version kept `<|t2|>` "so the next word is still timed", but the next word
    already carries its own opening timestamp: the kept one was a lone marker
    that shifted every positional timing pair after it on the line.

    Example:
        "a<|1.00|> <|1.00|>[pause]<|1.50|> <|1.50|>b<|1.8|>" with min_pause_ms=800
            -> "a<|1.00|> <|1.50|>b<|1.8|>" (500ms removed)
        "<|1.00|>[pause]<|2.00|>" with min_pause_ms=800 -> "<|1.00|>[pause]<|2.00|>" (1000ms kept)
    """
    min_pause_s = min_pause_ms / 1000.0

    def replace_short_pause(match):
        t1 = float(match.group(2))
        t2 = float(match.group(3))
        if t2 - t1 < min_pause_s:
            return ""
        return match.group(0)

    return re.sub(
        r"( ?)<\|([\d.]+)\|>\[pause\]<\|([\d.]+)\|>", replace_short_pause, text
    )


def remove_dangling_timestamps(text: str) -> str:
    """Drop a timestamp that opens a pair at the very end of a line.

    A window whose audio runs out right after an opening timestamp leaves
    `word<|t1|> <|t2|>` on its last line. Once same-speaker lines are merged that
    marker sits mid-line and every positional pair after it is misread
    (`<|t2|> <|t3|>` becomes an empty pair). Only a marker with whitespace before
    it and nothing but `[CHOP]` after it is dropped; a closing marker glued to its
    word or tag is left alone.

    Example:
        "[0] <|0.00|>Plan<|0.44|> <|0.46|>[CHOP]" -> "[0] <|0.00|>Plan<|0.44|>[CHOP]"
    """
    lines = []
    for line in text.split("\n"):
        matches = list(TIMINGS_FORMAT.finditer(line))
        if len(matches) % 2:
            last = matches[-1]
            head, tail = line[: last.start()], line[last.end() :]
            if not tail.replace("[CHOP]", "").strip() and head[-1:].isspace():
                line = head.rstrip() + tail.strip()
        lines.append(line)
    return "\n".join(lines)


_SPEAKER_LINE_PATTERN = re.compile(r"^\[(\d+|c)\]", re.MULTILINE)
_LINE_STARTS_WITH_TIMESTAMP = re.compile(r"^\[(?:\d+|c)\]\s*<\|\d+\.\d+\|>")
_LINE_ENDS_WITH_TIMESTAMP = re.compile(r"<\|\d+\.\d+\|>\s*$")

