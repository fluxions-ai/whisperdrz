"""
Inference Utilities for WhisperDRZ ASR

This module provides shared utilities for inference:
- Tokenizer-derived token lists (blocked tokens, timing tokens)
- InferenceState dataclass for tracking decoding state
- finalize_transcription() - final text processing

FOR BATCHED INFERENCE, USE: src/whisperdrz/transcriber.py::Transcriber
"""

import os
import re
from dataclasses import dataclass, field

import torch
from torch import Tensor

from whisperdrz.audio import SAMPLE_RATE
from whisperdrz.clean import (
    BRACKET_DIGIT_MATCH,
    TIMINGS_FORMAT,
    quantize_timestamps,
)

# Debug logging controlled by environment variable
# Set AKRO_DEBUG_INFERENCE=1 to enable detailed inference logging
DEBUG_INFERENCE = os.getenv("AKRO_DEBUG_INFERENCE", "0") == "1"


# Tokenizer-derived state, populated by init_for_model(). Empty until a model is
# loaded -- Transcriber.__init__ calls init_for_model() before any consumer runs.
tok = None
VOCAB = {}
REVOCAB = {}
TIMINGS = []
_initialized_vocab_size = None


def init_for_model(model):
    """Initialize tokenizer and token maps for a specific model.

    This ensures the tokenizer matches the model's vocab size (e.g., 51865 for medium,
    51866 for large-v3). Must be called before inference with a new model type.
    """
    global tok, VOCAB, REVOCAB, TIMINGS, _initialized_vocab_size

    vocab_size = model.dims.n_vocab
    if _initialized_vocab_size == vocab_size:
        return  # Already initialized for this vocab size

    tok = model.tokenizer
    VOCAB = {i: tok.decode([i]) for i in range(tok.encoding.n_vocab)}
    REVOCAB = {v: k for k, v in VOCAB.items()}
    TIMINGS = [k for k, v in VOCAB.items() if re.match(TIMINGS_FORMAT, v)]
    _initialized_vocab_size = vocab_size
    _dont_predict_for_lang_cache.clear()  # lang lists derive from dont_predict

    # Rebuild derived token lists
    _init_token_lists()


def tokens_with_c(t: str):
    # Search VOCAB directly instead of REVOCAB to catch all tokens
    # (REVOCAB only has one entry per decoded string, so duplicates are lost)
    return [index for index, token in VOCAB.items() if t in token]


# Token lists - populated by _init_token_lists() (via init_for_model)
initial_tokens = []
dont_predict = []
angle_tokens_blocked = []  # <> tokens blocked for tags - can be unblocked if allow_tags=True
asterisk_tokens_unblockable = []  # * and " *" - can be unblocked if allow_asterisks=True


def _init_token_lists():
    """Initialize all derived token lists based on current tokenizer.

    Must be called after tok/VOCAB/REVOCAB are initialized.
    """
    global initial_tokens, dont_predict
    global angle_tokens_blocked, asterisk_tokens_unblockable

    from whisperdrz.vocab_constraints import (
        get_always_blocked_tokens,
        get_blocked_tokens_for_language,
    )

    # Default blocked tokens (used before language detection)
    # Uses English (Latin-only) as the default - most restrictive
    never_predict = list(
        set(get_blocked_tokens_for_language("en")) | set(get_always_blocked_tokens())
    )

    # Tokens that start with space+punctuation (e.g., " !", " ?", " .", etc.)
    space_punctuation_tokens = [
        idx
        for token, idx in REVOCAB.items()
        if token
        and len(token) >= 2
        and token[0] == " "
        and token[1] in ["!", "?", ".", ",", "-"]
    ]

    # Double punctuation tokens to prevent (excluding ellipsis "...")
    double_punctuation = [
        idx
        for token, idx in REVOCAB.items()
        if any(double in token for double in ["!!", "??", "..", ",,", "--", "''"])
        and "..." not in token
    ]

    # Add special tokens to never_predict
    never_predict += [
        k
        for k, t in VOCAB.items()
        if "<|" in t and "endoftext|>" not in t and k not in TIMINGS
    ]

    initial_tokens = [tok.sot]

    # Angle bracket tokens that can be unblocked for <tag> prediction
    # e.g. <happy>, <excited>, <americanaccent>, <britishaccent>
    # Only include simple < and > tokens, not special tokens like <|en|> or timestamps
    # Exclude >< tokens since consecutive tags like <happy><excited> are valid
    angle_tokens_blocked = [
        *[
            t
            for t in tokens_with_c("<")
            if t not in TIMINGS and t != tok.eot and "><" not in tok.decode([t])
        ],
        *[
            t
            for t in tokens_with_c(">")
            if t not in TIMINGS and t != tok.eot and "><" not in tok.decode([t])
        ],
    ]

    # Asterisk tokens - can be unblocked with allow_asterisks=True
    # Only allow "*" (token 9) and " *" (token 1853), block all others like "**", "***", etc.
    asterisk_allowed = {tok.encode("*")[0], tok.encode(" *")[0]}  # {9, 1853}
    asterisk_tokens_blocked = [
        t for t in tokens_with_c("*") if t not in asterisk_allowed
    ]
    asterisk_tokens_unblockable = list(asterisk_allowed)  # These can be unblocked

    dont_predict = [
        *never_predict,
        *initial_tokens,
        *double_punctuation,
        *space_punctuation_tokens,
        tok.no_timestamps,
        *asterisk_tokens_blocked,  # Disable complex asterisks permanently (**,***,etc)
        *asterisk_tokens_unblockable,  # Disable simple asterisks (can be unblocked with allow_asterisks)
        *tokens_with_c("("),  # Disable parentheses globally
        *tokens_with_c(")"),
        *tokens_with_c(" ]"),  # Disable space-bracket token
        *angle_tokens_blocked,  # Disable < > by default (can be unblocked with allow_tags)
        *tokens_with_c("【"),  # Disable Chinese/Japanese corner brackets
        *tokens_with_c("】"),
        *tokens_with_c("♪"),  # Disable musical note tokens
        *tokens_with_c("一"),  # Disable Chinese numeral one (hallucination artifact)
        *tokens_with_c("]["),  # Disable ][ token (invalid bracket sequence)
        27506,  # Disable "Kivol" token (hallucination artifact)
        *tokens_with_c("[...]"),  # Disable ellipsis bracket tokens
        *tokens_with_c("[?"),  # Disable question bracket tokens
        *tokens_with_c("–"),  # Disable en-dash (use regular hyphen only)
        *tokens_with_c("—"),  # Disable em-dash (use regular hyphen only)
        *tokens_with_c("""),  # Disable smart quotes (use straight quotes only)
        *tokens_with_c("""),
        *tokens_with_c("\u2018"),  # Disable smart apostrophes (left ')
        *tokens_with_c("\u2019"),  # Disable smart apostrophes (right ')
        *tokens_with_c("´"),  # Disable acute accent (use straight apostrophe)
    ]



_dont_predict_for_lang_cache: dict[str, tuple[int, ...]] = {}


def get_dont_predict_for_lang(lang: str | None) -> tuple[int, ...]:
    """Return dont_predict with lang-specific script block swapped from English default.

    `dont_predict` is initialized using English's script blocks (non-Latin tokens blocked).
    For other languages we need to remove those English-specific blocks and add the
    target language's blocks (e.g., for "ru" that blocks non-Cyrillic tokens, which
    includes Latin tokens that are OK for "en").

    Always-blocked tokens (from `get_always_blocked_tokens`) are language-independent
    so they remain in both sets — no swap needed.

    Falls back to English `dont_predict` when lang is None/"auto"/unknown.
    """
    if not lang or lang == "auto":
        return tuple(dont_predict)
    if lang in _dont_predict_for_lang_cache:
        return _dont_predict_for_lang_cache[lang]

    from whisperdrz.vocab_constraints import get_blocked_tokens_for_language

    try:
        en_script = set(get_blocked_tokens_for_language("en"))
        tgt_script = set(get_blocked_tokens_for_language(lang))
    except Exception:
        result = tuple(dont_predict)
        _dont_predict_for_lang_cache[lang] = result
        return result

    dp = set(dont_predict)
    dp -= en_script
    dp |= tgt_script
    result = tuple(sorted(dp))
    _dont_predict_for_lang_cache[lang] = result
    return result


@dataclass(eq=False)
class InferenceState:
    id: int = None
    language: str = "auto"
    fails: int = 0
    audio: Tensor | None = None
    index: int = -1
    offset: float = 0
    status: str = "ready"
    last_offset: float = -1
    sofar: str = ""
    cur: str = ""
    initial_tokens: Tensor = field(default_factory=lambda: torch.tensor(initial_tokens))
    last_timing: float | None = None
    speaker_last_timings: dict = field(
        default_factory=dict
    )  # Per-speaker timing floors
    cutearly = False
    keepspeaker = False
    tokens: Tensor = field(default_factory=lambda: torch.tensor(initial_tokens))
    last_token_id: int = -1
    token_count: int = 0
    recent_tokens: list[int] = field(default_factory=list)  # Last 10 token IDs (CPU)
    meta: dict = field(default_factory=dict)
    needs_alignment: bool = False
    alignment_result: tuple[str, float, bool] | None = None
    forced_timing_before_newline: bool = False
    force_timing_every_word: bool = False
    force_timestamp_next: bool = False  # Force timestamp on next iteration
    predict_next: str | None = None
    cached_len: int = 0  # Track how many tokens have been cached in KV
    stop_reason: str | None = None  # Why decoding stopped
    speaker_hidden_states: list = field(
        default_factory=list
    )  # [(speaker_id, hidden_state), ...]
    _last_hidden_state: Tensor | None = None  # Buffer for previous step's hidden state
    _window_max_timing: float = (
        0.0  # Max timing seen in current window (not reset on newline)
    )

    def __eq__(self, other):
        """Compare states by ID and index to avoid tensor comparison issues."""
        if not isinstance(other, InferenceState):
            return False
        return self.id == other.id and self.index == other.index

    def __hash__(self):
        """Make InferenceState hashable based on ID and index."""
        return hash((self.id, self.index))

    def reset(self):
        self.last_timing = None
        self.offset = 0
        self.cur = ""
        self.sofar = ""
        # NOTE: `language` is intentionally NOT reset — it is slot-level config
        # (forced via Transcriber(lang=...) or detected once) that must survive
        # per-chunk resets; decoding maps "auto" to the English token mask.
        self.tokens = self.initial_tokens.clone()
        self.last_token_id = -1
        self.token_count = 0
        self.recent_tokens = []
        self.fails = 0
        self.cutearly = False
        self.needs_alignment = False
        self.alignment_result = None
        self.forced_timing_before_newline = False
        self.predict_next = None
        self.stop_reason = None
        self.speaker_hidden_states = []
        self._last_hidden_state = None
        self._window_max_timing = 0.0
        if "prediction" in self.meta:
            del self.meta["prediction"]

    @property
    def duration(self):
        if self.audio is None:
            return self.meta.get("duration")

        if isinstance(self.audio, str):
            from whisperdrz.audio import get_audio_duration

            dur = get_audio_duration(self.audio)
            return dur if dur is not None else 0.0
        return self.audio.shape[-1] / SAMPLE_RATE


def finalize_transcription(
    state: InferenceState, cur: str, apply_offset: bool = True
) -> str:
    """Common finalization logic for transcription segments.

    When `apply_offset=False`, leaves timestamps in window-relative time —
    used so DTW alignment can operate in the same coordinate system as its
    xa input. Caller must apply `state.offset` afterwards (e.g. via
    `offset_timestamps`).
    """
    offset = state.offset if apply_offset else 0.0

    # Debug: Show offset being applied
    if DEBUG_INFERENCE and offset > 0:
        first_timing = TIMINGS_FORMAT.search(cur)
        if first_timing:
            print(
                f"[OFFSET] Applying offset {offset:.2f}s to timings (first timing before: {first_timing.group(1)})"
            )

    cur = quantize_timestamps(cur, offset=offset)

    if DEBUG_INFERENCE and offset > 0:
        first_timing_after = TIMINGS_FORMAT.search(cur)
        if first_timing_after:
            print(f"[OFFSET] First timing after offset: {first_timing_after.group(1)}")

    # Fix speaker tags that appear without newlines
    # Replace [] with \n[0]
    cur = cur.replace("[]", "\n[0]")

    # Add newline before speaker tags that don't have one
    # Pattern: finds speaker tags like [0], [1], etc. that aren't preceded by newline or start of string
    cur = re.sub(r"(?<!\n)(?<=.)(\[\d\])", r"\n\1", cur)

    # Also handle speaker tags at the very start of the chunk
    # (The above pattern requires at least one preceding character due to (?<=.))
    # Adding \n ensures speaker tags survive the later remove_speaker_tags_not_after_newline cleanup
    if re.match(r"^\[\d\]", cur):
        cur = "\n" + cur

    if not state.keepspeaker:
        cur += "\n"
    if state.cutearly:
        # Remove speaker id at front when continuing same speaker (only check first 10 chars)
        prefix = cur[:10]
        matches = list(BRACKET_DIGIT_MATCH.finditer(prefix))
        if len(matches):
            cur = cur[matches[0].end() :].lstrip()
    return cur
