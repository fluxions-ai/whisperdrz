"""Evaluation metrics for WhisperDRZ transcripts: WER, WDER and cpWER.

- WER (word error rate): Levenshtein word distance after normalization.
- WDER (word diarization error rate): of the words that align between
  reference and hypothesis, the fraction assigned to the wrong speaker
  (under the best speaker permutation). Insertions/deletions are WER errors
  and don't count toward WDER.
- cpWER (concatenated minimum-permutation WER): concatenate each speaker's
  words into one stream, pick the speaker permutation that minimises the total
  word errors, then report a single WER over the concatenated streams. Unlike
  WDER it *does* charge insertions/deletions, so it jointly scores transcription
  and diarization. Computed with meeteval.

All operate on WhisperDRZ-format text (speaker tags ``[N]``, ``<|t|>`` timestamps,
``[tag]`` non-speech events). Requires the ``eval`` extra (jiwer,
whisper-normalizer, meeteval): ``pip install -e ".[eval]"``.
"""

import re
from itertools import permutations

from whisperdrz.clean.patterns import (
    BRACKETS_GENERAL,
    ENDOFTEXT_PATTERN,
    INTERRUPTIONS,
    METADATA,
    TIMINGS_FORMAT,
)

_ANGLE_TAGS = re.compile(r"<[^|][^>]*>")  # <accent:..>, <emotion:..> (not <|t|>)
_EMPHASIS = re.compile(r"\*(\w+)\*")
_UNK = re.compile(r"<unk>", re.IGNORECASE)
_ELLIPSIS = re.compile(r"\.{2,}")
_SINGLE_LETTER_DEDUP = re.compile(r"\b(\w)(?:\s+\1)+\b")
_SPEAKER_LINE = re.compile(r"^\[(\d+|c)\]\s*")
_WORD_OK = re.compile(r"^[\w'-]+$")

_normalizer = None


def _normalize(text: str) -> str:
    global _normalizer
    if _normalizer is None:
        from whisper_normalizer.english import EnglishTextNormalizer

        _normalizer = EnglishTextNormalizer()
    return _normalizer(text)


def _strip_tags(text: str) -> str:
    """Remove angle-bracket metadata tags and *emphasis* markers."""
    return _EMPHASIS.sub(r"\1", _ANGLE_TAGS.sub("", text))


def clean_for_wer(text: str) -> str:
    """Strip WhisperDRZ-specific syntax then run whisper-normalizer for a canonical
    WER comparison form (lowercased, numbers spelled out, punctuation removed)."""
    if not text:
        return ""
    text = _strip_tags(str(text).strip())
    text = ENDOFTEXT_PATTERN.sub("", text)
    text = TIMINGS_FORMAT.sub("", text)
    text = INTERRUPTIONS.sub("", text)
    text = BRACKETS_GENERAL.sub("", text)
    text = METADATA.sub("", text)
    text = _UNK.sub("", text)
    text = _ELLIPSIS.sub(" ", text)
    text = _normalize(text)
    return _SINGLE_LETTER_DEDUP.sub(r"\1", text)


def calculate_wer(ref_text: str, hyp_text: str) -> dict:
    """WER between reference and hypothesis transcripts.

    Returns wer plus raw insertion/deletion/substitution counts and the
    reference word count (so callers can compute a corpus micro-average).
    """
    import jiwer

    ref = clean_for_wer(ref_text)
    hyp = clean_for_wer(hyp_text)
    if not ref:
        return {
            "wer": 0.0,
            "insertions": 0,
            "deletions": 0,
            "substitutions": 0,
            "reference_words": 0,
        }
    out = jiwer.process_words(ref, hyp)
    return {
        "wer": out.wer,
        "insertions": out.insertions,
        "deletions": out.deletions,
        "substitutions": out.substitutions,
        "reference_words": len(ref.split()),
    }


def words_with_speakers(text: str) -> list[tuple[str, str]]:
    """Parse ``[N] words`` lines into ``[(word, speaker), ...]`` (lowercased,
    timestamps/tags stripped). Used by WDER."""
    out: list[tuple[str, str]] = []
    speaker = "0"
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = _SPEAKER_LINE.match(line)
        if m:
            speaker = m.group(1)
            line = line[m.end() :]
        line = _strip_tags(TIMINGS_FORMAT.sub("", line))
        # drop [tag] / (tag) non-speech events
        line = re.sub(r"\[[a-zA-Z/]+\]|\([a-zA-Z]+\)", " ", line)
        for raw in line.split():
            word = raw.strip(".,!?;:\"()[]{}").lower()
            if word and _WORD_OK.match(word):
                out.append((word, speaker))
    return out


def speaker_streams(text: str) -> dict[str, str]:
    """Group a transcript into one whisper-normalized word stream per speaker.

    Each ``[N]`` line's words are appended to that speaker's stream (in order),
    then the whole stream is run through :func:`clean_for_wer`. This is the input
    shape cpWER needs: ``{speaker_label: "normalized words ..."}``.
    """
    streams: dict[str, list[str]] = {}
    speaker = "0"
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = _SPEAKER_LINE.match(line)
        if m:
            speaker = m.group(1)
            line = line[m.end() :]
        streams.setdefault(speaker, []).append(line)
    return {spk: clean_for_wer(" ".join(parts)) for spk, parts in streams.items()}


def calculate_cpwer(ref_text: str, hyp_text: str) -> dict:
    """Concatenated minimum-permutation WER (cpWER) via meeteval.

    Concatenates each speaker's words, finds the speaker assignment minimising
    total errors, and returns a single WER over the concatenated streams plus
    the raw error counts (for a corpus micro-average) and the chosen assignment.
    """
    from meeteval.wer import cp_word_error_rate

    ref = {k: v for k, v in speaker_streams(ref_text).items() if v}
    hyp = {k: v for k, v in speaker_streams(hyp_text).items() if v}
    if not ref:
        return {
            "cpwer": 0.0,
            "errors": 0,
            "insertions": 0,
            "deletions": 0,
            "substitutions": 0,
            "reference_words": 0,
            "assignment": (),
        }
    if not hyp:
        hyp = {"0": ""}

    er = cp_word_error_rate(ref, hyp)
    return {
        "cpwer": er.error_rate if er.error_rate is not None else 0.0,
        "errors": er.errors,
        "insertions": er.insertions,
        "deletions": er.deletions,
        "substitutions": er.substitutions,
        "reference_words": er.length,
        "assignment": er.assignment,
    }


def calculate_wder(ref_text: str, hyp_text: str) -> dict:
    """Word diarization error rate.

    Aligns reference and hypothesis words (jiwer), then over the aligned
    (equal/substituted) word pairs counts how many have a mismatched speaker
    under the best speaker permutation. ``wder = 1 - correct / aligned``.
    """
    import jiwer

    ref_words = words_with_speakers(ref_text)
    hyp_words = words_with_speakers(hyp_text)
    if not ref_words:
        return {"wder": 0.0, "correct": 0, "aligned": 0, "speaker_map": {}}
    if not hyp_words:
        return {"wder": 1.0, "correct": 0, "aligned": 0, "speaker_map": {}}

    out = jiwer.process_words(
        " ".join(w for w, _ in ref_words), " ".join(w for w, _ in hyp_words)
    )
    pairs: list[tuple[str, str]] = []
    for chunk in out.alignments[0]:
        if chunk.type in ("equal", "substitute"):
            for r, h in zip(
                range(chunk.ref_start_idx, chunk.ref_end_idx),
                range(chunk.hyp_start_idx, chunk.hyp_end_idx),
                strict=False,
            ):
                pairs.append((ref_words[r][1], hyp_words[h][1]))

    if not pairs:
        return {"wder": 1.0, "correct": 0, "aligned": 0, "speaker_map": {}}

    ref_spk = list({r for r, _ in pairs})
    hyp_spk = list({h for _, h in pairs})
    # Cap permutation search to avoid factorial blowup on many speakers.
    MAX = 5
    ref_spk, hyp_spk = ref_spk[:MAX], hyp_spk[:MAX]
    n = max(len(ref_spk), len(hyp_spk))
    ref_pad = ref_spk + [f"_r{i}" for i in range(n - len(ref_spk))]
    hyp_pad = hyp_spk + [f"_h{i}" for i in range(n - len(hyp_spk))]

    best_correct, best_map = 0, {}
    for perm in permutations(hyp_pad):
        mapping = dict(zip(perm, ref_pad, strict=False))
        correct = sum(1 for r, h in pairs if mapping.get(h) == r)
        if correct > best_correct:
            best_correct = correct
            best_map = {k: v for k, v in mapping.items() if not k.startswith("_")}

    return {
        "wder": 1.0 - best_correct / len(pairs),
        "correct": best_correct,
        "aligned": len(pairs),
        "speaker_map": best_map,
    }


def has_speakers(text: str) -> bool:
    """True if the text has more than one distinct ``[N]`` speaker tag."""
    return len({s for _, s in words_with_speakers(text)}) > 1
