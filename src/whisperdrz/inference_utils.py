"""
Inference Utilities for WhisperDRZ ASR

This module provides shared utilities for inference:
- Token manipulation (enable/disable tokens, timing tokens)
- InferenceState dataclass for tracking decoding state
- decode() function - legacy single-sequence decoding (used by whisperid)
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
    parse_timing,
    quantize_timestamps,
)
from whisperdrz.tokenizer import get_tokenizer

# Debug logging controlled by environment variable
# Set AKRO_DEBUG_INFERENCE=1 to enable detailed inference logging
DEBUG_INFERENCE = os.getenv("AKRO_DEBUG_INFERENCE", "0") == "1"


# Token manipulation utilities needed for inference
# These are initialized with defaults and can be reinitialized via init_for_model()
tok = None
VOCAB = {}
REVOCAB = {}
contains_token_map = {}
TIMINGS = []
TIMINGS_VALUES = {}
_initialized_vocab_size = None


def init_for_model(model):
    """Initialize tokenizer and token maps for a specific model.

    This ensures the tokenizer matches the model's vocab size (e.g., 51865 for medium,
    51866 for large-v3). Must be called before inference with a new model type.
    """
    global \
        tok, \
        VOCAB, \
        REVOCAB, \
        contains_token_map, \
        TIMINGS, \
        TIMINGS_VALUES, \
        _initialized_vocab_size
    global _language_blocked_tokens_cache

    vocab_size = model.dims.n_vocab
    if _initialized_vocab_size == vocab_size:
        return  # Already initialized for this vocab size

    tok = model.tokenizer
    VOCAB = {i: tok.decode([i]) for i in range(tok.encoding.n_vocab)}
    REVOCAB = {v: k for k, v in VOCAB.items()}
    contains_token_map = {}  # Clear cache
    _language_blocked_tokens_cache = {}  # Clear language cache
    TIMINGS = [k for k, v in VOCAB.items() if re.match(TIMINGS_FORMAT, v)]
    TIMINGS_VALUES = {parse_timing(VOCAB[t]): t for t in TIMINGS}
    _initialized_vocab_size = vocab_size

    # Reinitialize derived token lists
    _init_token_lists()


def _init_default_tokenizer():
    """Initialize with default tokenizer (large-v3 settings)."""
    global tok, VOCAB, REVOCAB, TIMINGS, TIMINGS_VALUES, _initialized_vocab_size
    tok = get_tokenizer()
    VOCAB = {i: tok.decode([i]) for i in range(tok.encoding.n_vocab)}
    REVOCAB = {v: k for k, v in VOCAB.items()}
    TIMINGS = [k for k, v in VOCAB.items() if re.match(TIMINGS_FORMAT, v)]
    TIMINGS_VALUES = {parse_timing(VOCAB[t]): t for t in TIMINGS}
    _initialized_vocab_size = tok.encoding.n_vocab
    # Note: _init_token_lists() is called at end of module after it's defined


# Initialize base tokenizer data structures (tok, VOCAB, REVOCAB, TIMINGS)
# Token lists are initialized after _init_token_lists is defined
_init_default_tokenizer()


def tokens_with_c(t: str):
    if t not in contains_token_map:
        # Search VOCAB directly instead of REVOCAB to catch all tokens
        # (REVOCAB only has one entry per decoded string, so duplicates are lost)
        ret = [index for index, token in VOCAB.items() if t in token]
        contains_token_map[t] = ret
    return contains_token_map[t]


def disable_timings(logits):
    logits[..., TIMINGS] = -torch.inf
    return logits


def disable_token(t: str | int, logits):
    id = tokens_with_c(t) if isinstance(t, str) else [t]

    # Prevent out of index errors
    id = [x for x in id if x < logits.shape[-1]]
    logits[..., id] = -torch.inf
    return logits


TIMINGS_VALUES = {parse_timing(VOCAB[t]): t for t in TIMINGS}


def timings_in_range(s: float, e: float | None = None):
    if e is not None:
        return [TIMINGS_VALUES[k] for k in TIMINGS_VALUES if k >= s and k <= e]

    return [TIMINGS_VALUES[k] for k in TIMINGS_VALUES if k >= s]


def enable_timings_only(logits, start=0, end=None):
    timings = TIMINGS if start == 0 and end is None else timings_in_range(start, end)
    mask = torch.full_like(logits, -torch.inf)

    # Convert timings list to tensor for proper indexing
    # Filter out any indices that are out of bounds
    valid_timings = [t for t in timings if t < logits.shape[-1]]
    if valid_timings:
        timings_tensor = torch.tensor(
            valid_timings, dtype=torch.long, device=logits.device
        )
        # Use advanced indexing with .index_copy_ for batch safety
        # This ensures the operation works regardless of batch dimensions
        mask.index_copy_(
            dim=-1,
            index=timings_tensor,
            source=logits.index_select(dim=-1, index=timings_tensor),
        )

    return mask


def disable_single(c: str, logits):
    id = [REVOCAB[c]]
    logits[..., id] = -torch.inf
    return logits


def disable_space_starting_tokens(logits):
    """Disable all tokens that start with a space."""
    logits[..., space_starting_tokens] = -torch.inf
    return logits


def enable_only_digits(logits):
    """Enable only tokens that start with a digit (0-9)"""
    digit_tokens = [idx for idx, token in VOCAB.items() if token and token[0].isdigit()]
    mask = torch.full_like(logits, -torch.inf)
    mask[..., digit_tokens] = logits[..., digit_tokens]
    return mask


def enable_only_space_tokens(logits):
    """Enable only tokens that start with a space"""
    space_tokens = [
        idx for idx, token in VOCAB.items() if token and token.startswith(" ")
    ]
    mask = torch.full_like(logits, -torch.inf)
    mask[..., space_tokens] = logits[..., space_tokens]
    return mask


def cat_tokens(tokens: Tensor, newtoken: Tensor, dim: int = -1) -> Tensor:
    """Concatenate tokens ensuring they are on the same device.

    Args:
        tokens: The existing token tensor
        newtoken: The new token tensor to concatenate
        dim: The dimension along which to concatenate (default: -1)

    Returns:
        Concatenated tensor on the same device as tokens
    """
    if newtoken.device != tokens.device:
        newtoken = newtoken.to(tokens.device)
    return torch.cat([tokens, newtoken], dim=dim)


def enable_best_timing(logits, state_cur: str) -> Tensor:
    """Enable only appropriate timing tokens based on current state.

    This method analyzes the current line to determine:
    - Whether we're in a pause (no upper bound on timing)
    - Whether we're in an interruption (use timing after opening marker)
    - What the last relevant timing is (to constrain next timing)
    - How many words have been decoded since the last timing (to allow more time)

    Args:
        logits: Logits tensor to modify
        state_cur: Current decoded text (state.cur)

    Returns:
        Modified logits with only appropriate timings enabled
    """
    # Get current line
    line = state_cur[state_cur.rindex("\n") + 1 :] if "\n" in state_cur else state_cur

    # Check if we're in a pause context (can have arbitrary length, no upper bound)
    inpause = "pause]" in line[-6:] if len(line) >= 6 else False

    # Check if we're in a laugh context (allow up to 10 seconds)
    inlaugh = "laugh]" in line[-6:] if len(line) >= 6 else False

    # Find the last relevant timing based on context
    last_relevant_timing = None
    text_after_timing = ""

    # Not in interruption: use the last timing in the line
    all_timings = list(TIMINGS_FORMAT.finditer(line))
    if all_timings:
        last_relevant_timing = parse_timing(all_timings[-1].group(0))
        text_after_timing = line[all_timings[-1].end() :]

    # Apply timing constraints based on context
    if last_relevant_timing is None:
        # No timing found, allow all timings
        return enable_timings_only(logits)
    if inpause:
        # In pause: t2 must be >= t1 (but can be arbitrarily larger)
        return enable_timings_only(logits, start=last_relevant_timing)
    if inlaugh:
        # In laugh: t2 must be >= t1, but at most 10 seconds longer
        return enable_timings_only(
            logits, start=last_relevant_timing, end=last_relevant_timing + 10.0
        )

    # Normal case: constrain based on word count since last timing
    # Remove special markers and count actual words
    cleaned_text = re.sub(r"[\[\]\*\(\)]", "", text_after_timing)
    words = [w for w in cleaned_text.split() if w.strip()]
    word_count = len(words)

    # Add 1 second per word plus 1 second base leeway
    leeway = word_count * 1.0 + 1.0

    return enable_timings_only(
        logits, start=last_relevant_timing, end=last_relevant_timing + leeway
    )


def block_all_punctuation(logits):
    """Block ALL punctuation tokens: !, ?, ., ,, -, '

    This uses multiple blocking strategies:
    1. Block the pre-computed punctuation_after_word_only list
    2. Explicitly disable each punctuation character

    IMPORTANT: Does NOT block timing tokens (e.g., <|0.00|>) even though they contain '.'

    Returns:
        Modified logits with all punctuation blocked
    """
    # Create sets for faster lookup
    timing_set = set(TIMINGS)

    # Block via the pre-computed list (but preserve timing tokens)
    punct_indices = [
        idx
        for idx in punctuation_after_word_only
        if idx not in timing_set and idx < logits.shape[-1]
    ]
    if punct_indices:
        punct_tensor = torch.tensor(
            punct_indices, dtype=torch.long, device=logits.device
        )
        logits[..., punct_tensor] = -torch.inf

    # Explicitly block each punctuation character
    # NOTE: "." is handled separately below to avoid blocking timing tokens
    for punct_char in ["!", "?", ",", "-", "'"]:
        logits = disable_token(punct_char, logits)

    # For ".", only block tokens that contain "." but are NOT timing tokens
    # (timing tokens like <|0.00|> contain "." but must not be blocked)
    dot_indices = [
        idx
        for idx in tokens_with_c(".")
        if idx not in timing_set and idx < logits.shape[-1]
    ]
    if dot_indices:
        dot_tensor = torch.tensor(dot_indices, dtype=torch.long, device=logits.device)
        logits[..., dot_tensor] = -torch.inf

    return logits


# Use language-aware vocabulary constraints
# This is populated once inference starts and we know the language
_language_blocked_tokens_cache = {}


def get_language_blocked_tokens(language_code: str) -> list[int]:
    """Get blocked tokens for a specific language (with caching).

    Args:
        language_code: Language code (e.g., "en", "es", "zh") or "auto"

    Returns:
        List of token IDs to block for this language
    """
    if language_code not in _language_blocked_tokens_cache:
        from whisperdrz.vocab_constraints import (
            get_always_blocked_tokens,
            get_blocked_tokens_for_language,
        )

        # Get language-specific blocks
        lang_blocked = list(get_blocked_tokens_for_language(language_code))

        # Always add the always-blocked tokens
        always_blocked = list(get_always_blocked_tokens())

        # Combine and cache
        _language_blocked_tokens_cache[language_code] = list(
            set(lang_blocked + always_blocked)
        )

    return _language_blocked_tokens_cache[language_code]


# Token lists - initialized by _init_token_lists()
never_predict = []
punctuation_after_word_only = []
space_punctuation_tokens = []
double_punctuation = []
alphanumeric_no_space = []
space_starting_tokens = []
alphanumeric_with_space = []
first_tokens = []
initial_tokens = []
dont_predict = []
angle_tokens_blocked = []  # <> tokens blocked for tags - can be unblocked if allow_tags=True
asterisk_tokens_blocked = []  # * tokens blocked permanently (**, ***, etc)
asterisk_tokens_unblockable = []  # * and " *" - can be unblocked if allow_asterisks=True
transcribe_en = []


def _init_token_lists():
    """Initialize all derived token lists based on current tokenizer.

    Must be called after tok/VOCAB/REVOCAB are initialized.
    """
    global never_predict, punctuation_after_word_only, space_punctuation_tokens
    global double_punctuation, alphanumeric_no_space, space_starting_tokens
    global alphanumeric_with_space, first_tokens, initial_tokens, dont_predict
    global \
        angle_tokens_blocked, \
        asterisk_tokens_blocked, \
        asterisk_tokens_unblockable, \
        transcribe_en

    # Default blocked tokens (used before language detection)
    # Uses English (Latin-only) as the default - most restrictive
    never_predict = get_language_blocked_tokens("en")

    # Punctuation tokens that should only appear after words (not after space, ], or >)
    punctuation_after_word_only = [
        idx
        for token, idx in REVOCAB.items()
        if token
        and any(token.lstrip().startswith(p) for p in ["!", "?", ".", ",", "-"])
    ]

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

    # Alphanumeric tokens (without leading space)
    alphanumeric_no_space = [
        idx
        for idx, token in VOCAB.items()
        if token and len(token) > 0 and token[0].isalnum()
    ]

    # All tokens starting with a space (used to block after opening timing)
    space_starting_tokens = [
        idx for idx, token in VOCAB.items() if token and token.startswith(" ")
    ]

    # Alphanumeric tokens (with leading space) - used for normal word continuation
    alphanumeric_with_space = [
        idx
        for idx, token in VOCAB.items()
        if token
        and len(token) > 0
        and token[0] == " "
        and len(token) > 1
        and token[1].isalnum()
    ]

    # Add special tokens to never_predict
    never_predict += [
        k
        for k, t in VOCAB.items()
        if "<|" in t and "endoftext|>" not in t and k not in TIMINGS
    ]

    first_tokens = [tok.sot]
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

    transcribe_en = [
        tok.sot,
        tok.lang_token,
        tok.transcribe,
    ]


# Initialize token lists on module load
_init_token_lists()


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


# Non-speech marker words that cannot repeat within a line
NON_SPEECH_WORDS = [
    "breath",
    "hesitate",
    "hesitation",
    "laugh",
    "pause",
    "sigh",
    "tut",
    "gasp",
    "cough",
    "music",
    "applause",
    "laughter",
    "inaudible",
    "crosstalk",
]


def block_repeated_non_speech(logits, line: str):
    """Block non-speech marker words that already appeared in current line.

    Non-speech markers like [breath], [hesitate], etc. should not repeat
    within a single line, even with timings between them.
    """
    for word in NON_SPEECH_WORDS:
        if f"[{word}]" in line:
            logits = disable_token(word, logits)
    return logits


@dataclass(eq=False)
class InferenceState:
    id: int = None
    langauge: str = "auto"
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
        self.language = "auto"
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


# =============================================================================
# LEGACY DECODE FUNCTION - Commented out, kept for reference
# Used by whisperid.py and whisperid_eval.py - to be migrated to new batched approach
# =============================================================================
#
# def _handle_language_selection(
#     state: InferenceState,
#     logits: Tensor,
#     device,
#     temperature: float,
#     top_k: int | None,
#     top_p: float | None,
# ) -> bool:
#     """Handle language token selection (first token). Returns True if handled."""
#     if len(state.tokens) != 1:
#         return False
#
#     state.forced_timing_before_newline = False
#     lang_tokens = list(tok.all_language_tokens)
#     mask = torch.full_like(logits, -torch.inf)
#     mask[..., lang_tokens] = logits[..., lang_tokens]
#     next_tokens = sample_logits(mask, temperature, top_k, top_p)
#     tokens = torch.cat([state.tokens.to(device), next_tokens], dim=-1)
#     state.langauge = tok.decode(next_tokens)
#     state.langauge = state.langauge.replace("<|", "").replace("|>", "")
#     state.tokens = tokens
#     return True
#
#
# def _handle_transcribe_token(state: InferenceState, device) -> bool:
#     """Handle transcribe token addition (second token). Returns True if handled."""
#     if len(state.tokens) != 2:
#         return False
#
#     transcribe_token = torch.tensor([tok.transcribe], device=device)
#     state.tokens = cat_tokens(state.tokens, transcribe_token)
#     return True
#
#
# def decode(
#     logits: Tensor,
#     state: InferenceState,
#     temperature: float = 0.0,
#     top_k: int | None = None,
#     top_p: float | None = None,
#     max_duration_offset: float = 0.5,
# ):
#     device = logits.device
#     logits = logits[:, -1]
#     original_logits = logits.clone()
#
#     # Early exit if already at EOT or not started
#     if state.tokens[-1] == tok.eot or state.status != "started":
#         state.status = "process"
#         eot_token = torch.tensor([tok.eot], device=device)
#         state.tokens = cat_tokens(state.tokens, eot_token)
#         return
#
#     # Handle language selection (first token)
#     if _handle_language_selection(state, logits, device, temperature, top_k, top_p):
#         return
#
#     # Handle transcribe token (second token)
#     if _handle_transcribe_token(state, device):
#         return
#
#     # Handle deferred prediction (e.g., newline after forced timing)
#     if state.predict_next is not None:
#         newtext = state.predict_next
#         state.predict_next = None
#         newtoken = tok.encode_tensor(newtext)
#         tokens = cat_tokens(state.tokens, newtoken)
#         state.cur += newtext
#         state.tokens = tokens
#
#         # Check if this was an EOT
#         is_eot = newtoken.item() == tok.eot if newtoken.numel() == 1 else False
#         if is_eot:
#             state.status = "process"
#         return
#
#     # Prevent special tokens and language-incompatible characters
#     # Use detected language from state if available, otherwise use default (English)
#     language_code = getattr(state, "langauge", "en")
#     language_blocked = get_language_blocked_tokens(language_code)
#     combined_blocked = list(set(dont_predict + language_blocked))
#     logits[:, combined_blocked] = -torch.inf
#
#     # Extract current state
#     cur = state.cur
#     tokens = state.tokens
#     offset = state.offset
#
#     # Get current line and timings
#     line = cur[cur.rindex("\n") + 1 :] if "\n" in cur else cur
#
#     # Parse timings from the current line
#     all_timing_matches = list(TIMINGS_FORMAT.finditer(line))
#     if len(all_timing_matches) >= 2:
#         # We have at least one complete pair - take the last two
#         last_timings = [
#             parse_timing(all_timing_matches[-2].group(0)),
#             parse_timing(all_timing_matches[-1].group(0)),
#         ]
#     elif len(all_timing_matches) == 1:
#         # Only opening timing so far
#         last_timings = [parse_timing(all_timing_matches[0].group(0)), None]
#     else:
#         last_timings = [None, None]
#
#     newtext: str | None = None
#     newtoken = None
#
#     # First stage of guarded decoding (prevent tokens from being predicted that are invalid)
#
#     # Beginning of new line we always predict speaker
#     if len(line) < 5:
#         if len(line) == 0:
#             newtext = "["
#         elif len(line) == 1:
#             logits = enable_only_digits(logits)
#             newtoken = sample_logits(logits, temperature, top_k, top_p)
#             newtext = tok.decode(newtoken)
#         elif len(line) == 2:
#             newtext = "]"
#         elif len(line) == 3:
#             newtext = " "
#         elif len(line) == 4:
#             logits = enable_best_timing(logits, state.cur)
#             newtoken = sample_logits(logits, temperature, top_k, top_p)
#             newtext = tok.decode(newtoken)
#
#         newtoken = tok.encode_tensor(newtext)
#         tokens = cat_tokens(tokens, newtoken)
#         state.cur += newtext
#         state.tokens = tokens
#         return
#
#     # Ok now we need clever rules
#     # Block non-speech markers that already appeared in this line
#     logits = block_repeated_non_speech(logits, line)
#
#     last_token = tokens[-1]
#     last_token_str = tok.decode(last_token)
#     last_was_timing = TIMINGS_FORMAT.match(last_token_str) is not None
#
#     even_timings = len(TIMINGS_FORMAT.findall(line)) % 2 == 0
#     last_was_space = line[-1] == " " if len(line) else False
#
#     # Check the last character to determine what tokens are valid
#     last_char = line[-1] if len(line) else ""
#     last_char_is_punct = last_char in ".,!?-'"
#     last_char_is_punct_or_alphanum = last_char_is_punct or last_char.isalnum()
#     time_remaining = (
#         state.duration - state.offset - state.last_timing
#         if state.last_timing is not None
#         else None
#     )
#     near_end = time_remaining is not None and time_remaining < 3
#     approaching_end = time_remaining is not None and time_remaining < 5
#
#     # Count words since last timing (for approaching_end logic)
#     words_since_timing = 0
#     if approaching_end:
#         all_timing_matches_check = list(TIMINGS_FORMAT.finditer(line))
#         if all_timing_matches_check:
#             text_after_timing = line[all_timing_matches_check[-1].end() :]
#             cleaned = re.sub(r"[\[\]\*\(\)]", "", text_after_timing)
#             words = [w for w in cleaned.split() if w.strip()]
#             words_since_timing = len(words)
#
#     if last_was_space and near_end:
#         # We need to force timings if we're within 3 seconds of the end of the audio
#         logits = enable_best_timing(logits, state.cur)
#     elif last_was_space and approaching_end and words_since_timing >= 2:
#         # Within 5 seconds of end: force timing every 2-3 words
#         logits = enable_best_timing(logits, state.cur)
#     elif last_was_space:
#         logits = block_all_punctuation(logits)
#     elif last_was_timing:
#         # After a timing token, disable subsequent timings, interruptions, and punctuation
#         logits = disable_timings(logits)
#         logits = disable_single("(", logits)
#         logits = block_all_punctuation(logits)
#
#         if even_timings:
#             logits = enable_only_space_tokens(logits)
#         else:
#             # After opening timing, expect word content (no leading space)
#             logits = disable_space_starting_tokens(logits)
#     elif last_token_str == "[":
#         logits = disable_single("]", logits)
#     elif last_char_is_punct_or_alphanum:
#         logits = disable_single("[", logits)
#     elif last_token_str == "]":
#         if even_timings:
#             logits = enable_only_space_tokens(logits)
#         else:
#             logits = disable_single("]", logits)
#             logits = disable_token("[", logits)
#
#     if last_char_is_punct or last_token_str == "]":
#         logits = block_all_punctuation(logits)
#
#     in_open_bracket = line.rfind("[") > line.rfind("]")
#     # For angle brackets, exclude timing tokens <|...|> using regex
#     # <(?!\|) matches < not followed by |, (?<!\|)> matches > not preceded by |
#     tag_opens = list(re.finditer(r"<(?!\|)", line))
#     tag_closes = list(re.finditer(r"(?<!\|)>", line))
#     last_tag_open = tag_opens[-1].start() if tag_opens else -1
#     last_tag_close = tag_closes[-1].start() if tag_closes else -1
#     in_open_angle = last_tag_open > last_tag_close
#     in_open_paren = line.rfind("(") > line.rfind(")")
#
#     if in_open_bracket:
#         logits = disable_token(" ", logits)
#         logits = block_all_punctuation(logits)
#         logits = disable_single("[", logits)
#         logits = disable_timings(logits)
#     else:
#         logits = disable_single("]", logits)
#
#     if in_open_angle:
#         logits = disable_token(" ", logits)
#         logits = block_all_punctuation(logits)
#         logits = disable_single("<", logits)
#         logits = disable_timings(logits)
#     else:
#         logits = disable_single(">", logits)
#
#     if in_open_paren:
#         logits = disable_token(" ", logits)
#         logits = block_all_punctuation(logits)
#         logits = disable_single("(", logits)
#         logits = disable_timings(logits)
#     else:
#         logits = disable_single(")", logits)
#
#     if even_timings and not last_was_space:
#         logits = disable_timings(logits)
#
#     # SAMPLING!
#     if newtext is None:
#         newtoken = sample_logits(logits, temperature, top_k, top_p)
#         newtext = tok.decode(newtoken)
#
#     # Check if the next token is EOT
#     is_eot = newtoken.item() == tok.eot if newtoken.numel() == 1 else False
#
#     # Force a timing at the end of a segment (before newline or EOT)
#     # Only do this if the last non-space token wasn't a timing
#     # (handles case like `<|29.96|> ` where last token is space but timing already present)
#     last_non_space_was_timing = last_was_timing or (
#         last_was_space and line.rstrip().endswith("|>")
#     )
#     if (newtext == "\n" or is_eot) and not last_non_space_was_timing:
#         # Always predict a closing timing before newline/EOT to mark segment end
#         # If this has happened, we need to make sure the state skips everything and predicts what was coming:
#         state.predict_next = newtext
#         logits = enable_best_timing(original_logits, state.cur)
#         newtoken = sample_logits(logits, temperature, top_k, top_p)
#         newtext = tok.decode(newtoken)
#
#     # Force a timing before ( (for interruptions)
#     if newtext == " (" or (newtext == "(" and not last_was_timing):
#         # Force timing before interruption marker
#         state.predict_next = newtext
#         logits = enable_best_timing(original_logits, state.cur)
#         newtoken = sample_logits(logits, temperature, top_k, top_p)
#         newtext = tok.decode(newtoken)
#
#     # Post decoding we need to correct problems
#     # If we have unclosed ] we should make sure to close them.
#
#     # Check if sampled token starts with space or is a timing token
#     next_is_space = newtext and len(newtext) > 0 and newtext[0] == " "
#     next_is_timing = TIMINGS_FORMAT.match(newtext) is not None if newtext else False
#
#     # Force just a space when we'd otherwise emit " word" near end of segment
#     # This triggers timing on next decode (via last_was_space and near_end/approaching_end checks)
#     force_space_for_timing = (
#         next_is_space
#         and newtext != " "
#         and (near_end or (approaching_end and words_since_timing >= 2))
#     )
#     if force_space_for_timing:
#         newtext = " "
#         newtoken = tok.encode_tensor(newtext)
#
#     # Handle unclosed markers when predicting space
#     elif next_is_space:
#         unclosed_bracket = line.count("[") != line.count("]")
#         # Count < and > that are NOT part of timing tokens <|...|>
#         unclosed_angle = len(re.findall(r"<(?!\|)", line)) != len(
#             re.findall(r"(?<!\|)>", line)
#         )
#         unclosed_paren = line.count("(") != line.count(")")
#
#         if last_token_str != "]" and unclosed_bracket:
#             newtext = "]"
#             newtoken = tok.encode_tensor(newtext)
#         elif last_token_str != ">" and unclosed_angle:
#             newtext = ">"
#             newtoken = tok.encode_tensor(newtext)
#         elif last_token_str != ")" and unclosed_paren:
#             newtext = ")"
#             newtoken = tok.encode_tensor(newtext)
#         elif not even_timings:
#             logits = enable_best_timing(logits, state.cur)
#             newtoken = sample_logits(logits, temperature, top_k, top_p)
#             newtext = tok.decode(newtoken)
#
#     # Check if the newly decoded token is a timing token
#     # Re-sample to ensure the timing is within valid range
#     if next_is_timing:
#         logits = enable_best_timing(original_logits, state.cur)
#         newtoken = sample_logits(logits, temperature, top_k, top_p)
#         newtext = tok.decode(newtoken)
#
#     tokens = cat_tokens(tokens, newtoken)
#     cur += newtext
#
#     # Re-parse timings from the updated line
#     line = cur[cur.rindex("\n") + 1 :] if "\n" in cur else cur
#     all_timing_matches = list(TIMINGS_FORMAT.finditer(line))
#     if len(all_timing_matches) >= 2:
#         last_timings = [
#             parse_timing(all_timing_matches[-2].group(0)),
#             parse_timing(all_timing_matches[-1].group(0)),
#         ]
#     elif len(all_timing_matches) == 1:
#         last_timings = [parse_timing(all_timing_matches[0].group(0)), None]
#     else:
#         last_timings = [None, None]
#
#     # Check if predicted timing exceeds segment bounds (offset + timing >= duration)
#     line = cur[cur.rindex("\n") + 1 :] if "\n" in cur else cur
#     even_timings = len(TIMINGS_FORMAT.findall(line)) % 2 == 0
#     if last_timings[1] is not None and even_timings:
#         duration = min(state.duration - state.offset, 30)
#         if last_timings[1] >= duration - max_duration_offset:
#             state.status = "process"
#
#     if "[cutoff]" in line:
#         state.status = "process"
#
#     # Update state
#     state.cur = cur
#     state.last_timing = last_timings[1]
#     state.tokens = tokens
#     state.offset = offset
#
# =============================================================================
# END LEGACY DECODE
# =============================================================================


def finalize_transcription(
    state: InferenceState, cur: str, apply_offset: bool = True
) -> str:
    """Common finalization logic for transcription segments.

    When `apply_offset=False`, leaves timestamps in window-relative time —
    used so DTW alignment can operate in the same coordinate system as its
    xa input. Caller must apply `state.offset` afterwards (e.g. via
    `offset_timestamps`).
    """
    from whisperdrz.clean import TIMINGS_FORMAT

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
