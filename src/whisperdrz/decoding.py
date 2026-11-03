"""
GPU-optimized decoding for WhisperDRZ ASR

Batched decoding with grouped constraint application:
- Phase 1: Classify states by constraint pattern
- Phase 2: Build batch_mask [BS, V] with grouped operations
- Phase 3: Single batched argmax
- Phase 4: Timing corrections for states that need them

Key optimizations:
- Pre-computed GPU tensors for constraint masks
- O(1) timing range lookup via direct index math
- vocab_list[id] for token->string (no tok.decode())
"""

import re
import unicodedata
from dataclasses import dataclass

import torch
from torch import Tensor

from whisperdrz.clean import TIMINGS_FORMAT, parse_timing


def is_word_char(c: str) -> bool:
    """Word-character test that handles combining marks.

    str.isalnum() returns False for Unicode combining marks (category M*), which
    appear as the final char of some Thai/Hindi/Tamil/Bengali tokens. Treat those
    as word characters too — a Hindi word ending in a vowel mark is still a word.
    """
    if not c:
        return False
    if c.isalnum():
        return True
    return unicodedata.category(c).startswith("M")


def text_has_word(s: str) -> bool:
    """Like is_word_char but for any char in a string."""
    return any(is_word_char(c) for c in s)


# Pre-compiled regexes for hot path
_BRACKETS_CLEAN = re.compile(r"[\[\]\*\(\)]")
_TAG_OPEN = re.compile(r"<(?!\|)")
_TAG_CLOSE = re.compile(r"(?<!\|)>")
_TIMED_SPAN = re.compile(r"<\|\d+\.\d+\|>([^<]*)<\|\d+\.\d+\|>")

VALID_BRACKET_TAGS = {
    # Body sounds
    "breath",
    "cough",
    "sniff",
    "sigh",
    "gasp",
    "clearthroat",
    "swallow",
    "sneeze",
    "wheeze",
    "exhale",
    "noseexhale",
    "sharpinhale",
    # Mouth sounds
    "tut",
    "click",
    "kiss",
    "kissing",
    "kissteeth",
    "chew",
    "chewing",
    "clap",
    "clapping",
    "snap",
    "slurp",
    "mouthnoise",
    # Laugh variants
    "laugh",
    "chuckle",
    "giggle",
    "giggling",
    "scoff",
    "laughter",
    # Speech
    "hesitate",
    "mumble",
    "inaudible",
    "babble",
    "slurring",
    "mispeak",
    "lisp",
    # Crowd
    "applause",
    "cheering",
    "crowdnoise",
    "crowdlaugh",
    "crowdlaughing",
    "crowdcheering",
    "crowdchatter",
    "crowdclapping",
    "crowdcheer",
    "crowdcough",
    "crowdgasp",
    "jeering",
    # Actions
    "pause",
    "panting",
    "crying",
    "screaming",
    "scream",
    "barking",
    "grunt",
    "whimper",
    "whistle",
    "squeal",
    "yelp",
    "cooing",
    "imitating",
    "struggling",
    # Paired opening
    "laughing",
    "singing",
    "whisper",
    "shouting",
    "music",
    "voicebreaking",
    "shout",
    "sing",
    # Paired closing (/ prefix)
    "/laughing",
    "/singing",
    "/whisper",
    "/shouting",
    "/music",
    "/voicebreaking",
    "/shout",
    "/sing",
    "/screaming",
    "/mispeak",
    "/slurring",
    "/clapping",
    "/crowdcheering",
    # Stop variants
    "stoplaughing",
    "stopmusic",
    "stopshouting",
    "stopsinging",
    "stopmispeak",
    "stopwhisper",
    "stopscreaming",
    "stopvoicebreaking",
    # Special
    "c",
}

_BRACKET_MAX_LEN = 20
_ASTERISK_MAX_LEN = 30  # max chars between opening * and the forced closing *


def asterisk_state(line: str) -> tuple[bool, str | None]:
    """Classify the line's asterisk state for decoder constraint application.

    Returns (in_asterisk, force_close_reason). `in_asterisk` is True when the
    line has an odd number of `*` (i.e. there's an unclosed opening).
    `force_close_reason` is set when the body since the last `*` has gone
    invalid and the caller should emit a closing `*` immediately:
      - "space_in_partial": a space landed inside `*...` (shouldn't happen)
      - "newline_in_partial": same for newline
      - "too_long": body exceeded _ASTERISK_MAX_LEN

    Otherwise returns (in_asterisk, None) and the caller proceeds with normal
    constraint masking.
    """
    if line.count("*") % 2 == 0:
        return False, None
    partial = line[line.rfind("*") + 1 :]
    if " " in partial:
        return True, "space_in_partial"
    if "\n" in partial:
        return True, "newline_in_partial"
    if len(partial) > _ASTERISK_MAX_LEN:
        return True, "too_long"
    return True, None

VALID_BRACKET_PREFIXES = set()
for _tag in VALID_BRACKET_TAGS:
    for _i in range(1, len(_tag) + 1):
        VALID_BRACKET_PREFIXES.add(_tag[:_i])


def _has_timed_word(line: str) -> bool:
    """Check if line has any alphanumeric word (not just tags) that has been timed.

    Returns True if there's a <|t|>word<|t|> span where 'word' contains alphanumeric
    content outside of [brackets]. Tags like [pause] don't count as words.
    """
    for match in _TIMED_SPAN.finditer(line):
        content = match.group(1)
        # Remove bracket content (tags like [pause], [breath])
        without_brackets = re.sub(r"\[[^\]]*\]", "", content)
        # Check for word content (handles combining marks in Hindi/Thai/Tamil/etc.)
        if text_has_word(without_brackets):
            return True
    return False


def _has_timed_content(line: str) -> bool:
    """Check if line has any timed content (words OR tags).

    Returns True if there's a <|t|>content<|t|> span with any non-empty content.
    Used to determine if newline should be allowed (line has valid content).
    """
    for match in _TIMED_SPAN.finditer(line):
        content = match.group(1).strip()
        if content:
            return True
    return False


def _get_line(cur: str) -> str:
    """Extract the current line from transcription cursor."""
    return cur[cur.rindex("\n") + 1 :] if "\n" in cur else cur


def _line_has_content(cur: str) -> bool:
    """Check if current line has word or tag content (not just timestamps)."""
    line = _get_line(cur)
    after_speaker = line.split("] ", 1)[-1] if "] " in line else line
    stripped = TIMINGS_FORMAT.sub("", after_speaker)
    return text_has_word(stripped) or "[" in stripped


@dataclass
class VirtualMask:
    """GPU-optimized mask using bool tensor instead of list + cat + unique.

    Key optimizations:
    - Uses pre-allocated bool mask (no cat/unique overhead)
    - Direct bool tensor operations instead of index gathering
    - Reuses tensors from DecoderState cache
    - Can be reset and reused to avoid allocation
    """

    blocked_mask: Tensor | None = None  # bool mask [vocab_size]
    enabled_only: Tensor | None = None  # index tensor
    timing_start: float | None = None
    timing_end: float | None = None
    _device: torch.device | None = None
    _vocab_size: int = 0

    def reset(self):
        """Reset mask for reuse - clears all constraints."""
        if self.blocked_mask is not None:
            self.blocked_mask.zero_()
        self.enabled_only = None
        self.timing_start = None
        self.timing_end = None


    def apply(self, logits: Tensor, ds: "DecoderState") -> Tensor:
        """Apply all constraints in single pass. Returns modified tensor."""
        vocab_size = logits.shape[-1]
        NEGINF = float("-inf")

        if self.enabled_only is not None and self.enabled_only.numel() > 0:
            neg_inf = ds.get_gpu_tensor("neg_inf_mask")
            result = neg_inf[:vocab_size].expand_as(logits).clone()
            valid_indices = self.enabled_only[self.enabled_only < vocab_size]
            if valid_indices.numel() > 0:
                result[..., valid_indices] = logits[..., valid_indices]
            # Apply blocked_mask on top
            if self.blocked_mask is not None:
                result[..., self.blocked_mask[:vocab_size]] = NEGINF
            return result

        if self.timing_start is not None:
            timing_tokens = ds.get_timing_tokens_in_range_gpu(
                self.timing_start, self.timing_end
            )
            neg_inf = ds.get_gpu_tensor("neg_inf_mask")
            result = neg_inf[:vocab_size].expand_as(logits).clone()
            valid_indices = timing_tokens[timing_tokens < vocab_size]
            if valid_indices.numel() > 0:
                result[..., valid_indices] = logits[..., valid_indices]
            # Apply blocked_mask on top
            if self.blocked_mask is not None:
                result[..., self.blocked_mask[:vocab_size]] = NEGINF
            return result

        if self.blocked_mask is not None:
            logits[..., self.blocked_mask[:vocab_size]] = NEGINF

        return logits


class DecoderState:
    """Decoder vocabulary and precomputed token sets with GPU tensors.

    O(1) lookups for:
    - Token ID -> string (vocab_list index)
    - String -> token ID (revocab dict)
    - Timing range -> token IDs (direct index math, 0.02s intervals)
    - Char -> tokens containing char (pre-computed dict)
    """

    def __init__(self, tokenizer, device: str | torch.device = "cuda"):
        self.tok = tokenizer
        self.device = torch.device(device) if isinstance(device, str) else device
        self.vocab_size = tokenizer.encoding.n_vocab

        self.vocab_list = [tokenizer.decode([i]) for i in range(self.vocab_size)]
        self.vocab = dict(enumerate(self.vocab_list))
        self.revocab = {v: k for k, v in self.vocab.items()}

        self.timing_step = 0.02
        self.max_timing_idx = 1500

        self.timings = [k for k, v in self.vocab.items() if re.match(TIMINGS_FORMAT, v)]
        self.timings_set = set(self.timings)
        self.timings_values = {parse_timing(self.vocab[t]): t for t in self.timings}

        self.space_starting = [
            idx for idx, token in self.vocab.items() if token and token.startswith(" ")
        ]
        self.digit_starting = [
            idx for idx, token in self.vocab.items() if token and token[0].isdigit()
        ]
        self.space_tokens_only = [
            idx
            for idx, token in self.vocab.items()
            if token and token.startswith(" ") and len(token) > 1 and token[1].isalnum()
        ]

        self.punctuation = [
            idx
            for token, idx in self.revocab.items()
            if token
            and any(token.lstrip().startswith(p) for p in ["!", "?", ".", ",", "-"])
            and idx not in self.timings_set
        ]

        self._contains_cache: dict[str, list[int]] = {}

        self._punct_block_sets: list[list[int]] = []
        for p in ["!", "?", ",", "-", ".", "…", "'"]:
            self._punct_block_sets.append(
                [t for t in self.tokens_containing(p) if t not in self.timings_set]
            )

        self._tokens_with_space = self.tokens_containing(" ")
        self._tokens_with_bracket = self.tokens_containing("[")
        self._tokens_with_angle = self.tokens_containing("<")
        self._tokens_with_paren = self.tokens_containing("(")

        self._all_punct_blocked: set[int] = set(self.punctuation)
        for punct_set in self._punct_block_sets:
            self._all_punct_blocked.update(punct_set)

        self._gpu_tensors_initialized = False
        self._device_tensors: dict[str, dict] = {}
        self._cpu_masks: dict[str, Tensor] = {}

        # Persistent per-slot base_blocked buffer. Shape [BS, V], updated in-place when
        # a slot's `state.language` changes (set at slot init and at language-prediction
        # step — both infrequent). GPU-resident so the per-step clone is a fast D2D copy
        # and all downstream mask ops in decode_batch stay on GPU (no H2D per step).
        self._slot_base_buf_gpu: Tensor | None = None
        self._slot_langs_cached: list[str | None] = []
        # Persistent per-step scratch buffers — reused across decode_batch calls so
        # we're not reallocating [BS, V] bool + fp16 tensors every step.
        self._batch_mask_gpu: Tensor | None = None
        self._additive_mask_gpu: Tensor | None = None
        self._additive_dtype: torch.dtype | None = None

        # Precompute candidate tokens for bracket tag validation
        # These are tokens that could plausibly appear inside [...] tags
        _bad_chars = set(" \t\n[]()<>|.,!?-'\"*")
        self._bracket_candidates: list[tuple[int, str]] = []
        for token_id in range(self.vocab_size):
            token_text = self.vocab_list[token_id]
            if not token_text:
                continue
            if any(c in _bad_chars for c in token_text):
                continue
            if any(c.isdigit() for c in token_text):
                continue
            self._bracket_candidates.append((token_id, token_text))
        self._bracket_allowed_cache: dict[str, list[int]] = {}

        # Pre-cached single-char token IDs as Python ints (avoids tensor creation in hot path)
        self._single_char_ids: dict[str, int] = {}
        for char in ["[", "]", "(", ")", "<", ">", " ", "0", "\n", "*"]:
            self._single_char_ids[char] = self.tok.encode(char)[0]


    def init_gpu_tensors(self, device: torch.device | None = None):
        if device is not None:
            self.device = device

        if self._gpu_tensors_initialized and str(self.device) in self._device_tensors:
            return

        d = self.device
        key = str(d)
        self._device_tensors[key] = {}
        cache = self._device_tensors[key]

        cache["timings"] = torch.tensor(self.timings, device=d, dtype=torch.long)
        cache["space_starting"] = torch.tensor(
            self.space_starting, device=d, dtype=torch.long
        )
        cache["digit_starting"] = torch.tensor(
            self.digit_starting, device=d, dtype=torch.long
        )
        cache["punctuation"] = torch.tensor(
            self.punctuation, device=d, dtype=torch.long
        )

        sorted_timing_items = sorted(self.timings_values.items())
        cache["timing_values"] = torch.tensor(
            [v for v, _ in sorted_timing_items], device=d, dtype=torch.float32
        )
        cache["timing_token_ids"] = torch.tensor(
            [t for _, t in sorted_timing_items], device=d, dtype=torch.long
        )

        cache["char_to_tokens"] = {}
        for char in [
            "[",
            "]",
            "(",
            ")",
            "<",
            ">",
            " ",
            ".",
            ",",
            "!",
            "?",
            "-",
            "'",
            "*",
        ]:
            tokens = self.tokens_containing(char)
            cache["char_to_tokens"][char] = torch.tensor(
                tokens, device=d, dtype=torch.long
            )

        all_punct = list(self._all_punct_blocked)
        cache["all_punct"] = torch.tensor(all_punct, device=d, dtype=torch.long)
        cache["neg_inf_mask"] = torch.full((self.vocab_size,), float("-inf"), device=d)

        # Pre-computed bool mask: True = blocked (dont_predict tokens)
        from whisperdrz.inference_utils import (
            angle_tokens_blocked,
            asterisk_tokens_unblockable,
            dont_predict,
        )

        def _blocked_from_indices(indices) -> Tensor:
            mask = torch.zeros(self.vocab_size, device=d, dtype=torch.bool)
            if indices:
                t = torch.tensor(list(indices), device=d, dtype=torch.long)
                t = t[t < self.vocab_size]
                mask[t] = True
            return mask

        base_blocked = _blocked_from_indices(dont_predict)

        # Angle tokens mask - can be unblocked when allow_tags=True
        angle_blocked_mask = torch.zeros(self.vocab_size, device=d, dtype=torch.bool)
        if angle_tokens_blocked:
            angle_indices = torch.tensor(
                angle_tokens_blocked, device=d, dtype=torch.long
            )
            angle_indices = angle_indices[angle_indices < self.vocab_size]
            angle_blocked_mask[angle_indices] = True
        cache["angle_blocked_mask"] = angle_blocked_mask

        # Asterisk tokens mask - can be unblocked when allow_asterisks=True
        asterisk_blocked_mask = torch.zeros(self.vocab_size, device=d, dtype=torch.bool)
        if asterisk_tokens_unblockable:
            asterisk_indices = torch.tensor(
                asterisk_tokens_unblockable, device=d, dtype=torch.long
            )
            asterisk_indices = asterisk_indices[asterisk_indices < self.vocab_size]
            asterisk_blocked_mask[asterisk_indices] = True
        cache["asterisk_blocked_mask"] = asterisk_blocked_mask

        # Language-independent "extras" that are always part of base_blocked regardless
        # of predicted language. Kept separately so we can re-use for per-language masks.
        # - space+punct tokens (e.g. " ," " !") — punct always follows words directly
        # - '][' token (7131) — never valid, timing must go between tags
        extra_blocked = torch.zeros(self.vocab_size, device=d, dtype=torch.bool)
        space_punct_tokens = [
            idx
            for token, idx in self.revocab.items()
            if token and len(token) >= 2 and token[0] == " " and token[1] in "?!.,-'\"…"
        ]
        for t in space_punct_tokens:
            extra_blocked[t] = True
        BRACKET_CLOSE_OPEN = 7131
        if self.vocab_size > BRACKET_CLOSE_OPEN:
            extra_blocked[BRACKET_CLOSE_OPEN] = True

        base_blocked = base_blocked | extra_blocked
        cache["base_blocked"] = base_blocked
        cache["base_extra_blocked"] = extra_blocked

        # Pre-computed combined constraint masks (bool: True = blocked)
        # These avoid repeated cat/unique operations at decode time
        V = self.vocab_size

        # Helper to create bool mask from index tensor
        def to_bool_mask(indices: Tensor) -> Tensor:
            mask = torch.zeros(V, device=d, dtype=torch.bool)
            valid = indices[indices < V]
            if valid.numel() > 0:
                mask[valid] = True
            return mask

        # Single token masks
        cache["block_open_bracket"] = to_bool_mask(
            torch.tensor([self.tok.encode("[")[0]], device=d)
        )
        cache["block_close_bracket"] = to_bool_mask(
            torch.tensor([self.tok.encode("]")[0]], device=d)
        )
        cache["block_open_paren"] = to_bool_mask(
            torch.tensor([self.tok.encode("(")[0]], device=d)
        )
        cache["block_close_paren"] = to_bool_mask(
            torch.tensor([self.tok.encode(")")[0]], device=d)
        )
        cache["block_less_than"] = to_bool_mask(
            torch.tensor([self.tok.encode("<")[0]], device=d)
        )
        cache["block_greater_than"] = to_bool_mask(
            torch.tensor([self.tok.encode(">")[0]], device=d)
        )
        # >< tokens for consecutive tags like <happy><excited>
        angle_consecutive = [
            t for t in range(self.vocab_size) if "><" in self.tok.decode([t])
        ]
        cache["block_angle_consecutive"] = to_bool_mask(
            torch.tensor(angle_consecutive, device=d)
            if angle_consecutive
            else torch.tensor([], device=d, dtype=torch.long)
        )
        # Single asterisk token for blocking when not after space/timing
        cache["block_asterisk"] = to_bool_mask(
            torch.tensor([self.tok.encode("*")[0]], device=d)
        )
        # Space+asterisk token - needs special handling to allow after words/timings
        cache["space_asterisk_token"] = to_bool_mask(
            torch.tensor([self.tok.encode(" *")[0]], device=d)
        )

        # Token-containing masks
        cache["block_contains_space"] = to_bool_mask(cache["char_to_tokens"][" "])
        cache["block_contains_close_bracket"] = to_bool_mask(
            cache["char_to_tokens"]["]"]
        )
        cache["block_contains_open_bracket"] = to_bool_mask(
            cache["char_to_tokens"]["["]
        )

        # Quote-containing tokens (block after a quote to prevent "" loops)
        quote_tokens = [
            t for t in range(self.vocab_size) if '"' in self.tok.decode([t])
        ]
        cache["block_quote"] = to_bool_mask(
            torch.tensor(quote_tokens, device=d)
            if quote_tokens
            else torch.tensor([], device=d, dtype=torch.long)
        )

        # Category masks
        cache["block_timings"] = to_bool_mask(cache["timings"])
        cache["block_space_starting"] = to_bool_mask(cache["space_starting"])
        cache["block_digit_starting"] = to_bool_mask(cache["digit_starting"])
        cache["block_punct"] = to_bool_mask(cache["all_punct"])
        cache["block_newlines"] = to_bool_mask(
            torch.tensor([self.tok.encode("\n")[0]], device=d)
        )

        # Combined masks for common constraint patterns
        # in_bracket: block spaces, punct, [, timings, digit_starting
        in_bracket = (
            cache["block_contains_space"]
            | cache["block_punct"]
            | cache["block_open_bracket"]
            | cache["block_timings"]
            | cache["block_digit_starting"]
        )
        cache["in_bracket_blocked"] = in_bracket

        # in_angle: block spaces, punct, <, timings, digit_starting
        in_angle = (
            cache["block_contains_space"]
            | cache["block_punct"]
            | cache["block_less_than"]
            | cache["block_timings"]
            | cache["block_digit_starting"]
        )
        cache["in_angle_blocked"] = in_angle

        # in_paren: block spaces, punct, (, timings
        in_paren = (
            cache["block_contains_space"]
            | cache["block_punct"]
            | cache["block_open_paren"]
            | cache["block_timings"]
        )
        cache["in_paren_blocked"] = in_paren

        # after_timing: block timings, punct, (
        after_timing = (
            cache["block_timings"] | cache["block_punct"] | cache["block_open_paren"]
        )
        cache["after_timing_blocked"] = after_timing

        # Language tokens (for step 1)
        lang_tokens = torch.tensor(
            list(self.tok.all_language_tokens), device=d, dtype=torch.long
        )
        cache["lang_tokens"] = lang_tokens
        lang_enabled = torch.zeros(V, device=d, dtype=torch.bool)
        lang_enabled[lang_tokens] = True
        cache["lang_enabled"] = lang_enabled

        # Space-starting enabled (for after timing with even count)
        # Also allow newline token for speaker changes
        space_enabled = torch.zeros(V, device=d, dtype=torch.bool)
        space_enabled[cache["space_starting"]] = True
        newline_token = self.tok.encode("\n")[0]
        if newline_token < V:
            space_enabled[newline_token] = True
        cache["space_starting_enabled"] = space_enabled

        # Pre-cached single token tensors (avoids repeated encode calls)
        cache["token_tensors"] = {}
        for char in ["[", "]", "(", ")", "<", ">", " ", "0", "\n", "*"]:
            cache["token_tensors"][char] = torch.tensor(
                self.tok.encode(char), device=d, dtype=torch.long
            )

        # Pre-allocated VirtualMask pool for batch decoding (avoids per-step allocation)
        max_batch = 128  # Max batch size we support
        self._mask_pool = []
        for _ in range(max_batch):
            mask = VirtualMask()
            mask.blocked_mask = torch.zeros(V, device=d, dtype=torch.bool)
            mask._device = d
            mask._vocab_size = V
            self._mask_pool.append(mask)

        # Precompute allowed token tensors for every valid bracket prefix
        close_bracket = self.tok.encode("]")[0]
        bracket_allowed: dict[str, Tensor] = {}
        empty = torch.tensor([], device=d, dtype=torch.long)
        for prefix in [*list(VALID_BRACKET_PREFIXES), ""]:
            allowed = []
            for token_id, token_text in self._bracket_candidates:
                if (prefix + token_text) in VALID_BRACKET_PREFIXES:
                    allowed.append(token_id)
            if prefix in VALID_BRACKET_TAGS:
                allowed.append(close_bracket)
            bracket_allowed[prefix] = (
                torch.tensor(allowed, device=d, dtype=torch.long) if allowed else empty
            )
        cache["bracket_allowed"] = bracket_allowed

        # Speaker ID mask (digits 0-9 + 'c' for crowd) - pre-computed
        speaker_id_mask = torch.zeros(V, device=d, dtype=torch.bool)
        for ch in "0123456789c":
            t = self.tok.encode(ch)[0]
            if t < V:
                speaker_id_mask[t] = True
        cache["speaker_id_mask"] = speaker_id_mask

        # CPU mask cache: bool tensors on CPU for batch_mask construction
        # Eliminates all GPU index_put_ kernel launches during decode
        cpu = {}
        mask_names = [
            "base_blocked",
            "base_extra_blocked",
            "after_timing_blocked",
            "block_less_than",
            "block_angle_consecutive",
            "block_asterisk",
            "space_asterisk_token",
            "block_punct",
            "block_space_starting",
            "block_timings",
            "block_contains_close_bracket",
            "block_open_bracket",
            "in_angle_blocked",
            "block_greater_than",
            "block_close_paren",
            "in_paren_blocked",
            "lang_enabled",
            "space_starting_enabled",
            "speaker_id_mask",
            "angle_blocked_mask",
            "asterisk_blocked_mask",
            "block_quote",
        ]
        for name in mask_names:
            t = cache[name]
            cpu[name] = t[:V].cpu() if t.shape[0] > V else t.cpu()
        # timing_token_ids as CPU list for fast slicing
        cpu["timing_token_ids"] = cache["timing_token_ids"].cpu().tolist()
        # bracket_allowed as CPU Python lists
        cpu["bracket_allowed"] = {}
        for prefix, t in bracket_allowed.items():
            cpu["bracket_allowed"][prefix] = t.cpu().tolist() if t.numel() > 0 else []
        # Per-language base_blocked cache: populated lazily by get_base_blocked_cpu.
        # English (the default) is precomputed so the common path hits the cache.
        cpu["base_blocked_by_lang"] = {"en": cpu["base_blocked"].clone()}
        self._cpu_masks = cpu

        self._gpu_tensors_initialized = True

    def get_gpu_tensor(self, name: str) -> Tensor:
        key = str(self.device)
        if key not in self._device_tensors:
            self.init_gpu_tensors()
        return self._device_tensors[key][name]


    def get_base_blocked_cpu(self, lang: str | None) -> Tensor:
        """Return the CPU base_blocked bool mask for a language (cached).

        Falls back to the English default when lang is None, "auto", or unknown.
        """
        if not self._gpu_tensors_initialized:
            self.init_gpu_tensors()
        cache = self._cpu_masks["base_blocked_by_lang"]
        if not lang or lang == "auto":
            return cache["en"]
        if lang in cache:
            return cache[lang]

        from whisperdrz.inference_utils import get_dont_predict_for_lang

        V = self.vocab_size
        mask = torch.zeros(V, dtype=torch.bool)
        dp = get_dont_predict_for_lang(lang)
        if dp:
            idx = torch.tensor(dp, dtype=torch.long)
            idx = idx[idx < V]
            mask[idx] = True
        mask |= self._cpu_masks["base_extra_blocked"]
        cache[lang] = mask
        return mask

    def update_slot_base_buffer(self, states) -> Tensor:
        """Return persistent [BS, V] per-slot base_blocked buffer on GPU, refreshed in-place.

        Only rows whose `state.language` differs from the cached value get updated
        (rare: once at slot init, once at language prediction). Callers should
        `.clone()` before mutating since the buffer persists across steps.
        """
        if not self._gpu_tensors_initialized:
            self.init_gpu_tensors()
        BS = len(states)
        V = self.vocab_size
        buf = self._slot_base_buf_gpu
        if buf is None or buf.shape[0] != BS:
            buf = torch.empty((BS, V), dtype=torch.bool, device=self.device)
            self._slot_base_buf_gpu = buf
            self._slot_langs_cached = [None] * BS
            # Force refill of all rows
            for i, s in enumerate(states):
                lang = getattr(s, "language", None)
                buf[i].copy_(self.get_base_blocked_cpu(lang).to(self.device))
                self._slot_langs_cached[i] = lang
            return buf

        cached = self._slot_langs_cached
        for i, s in enumerate(states):
            lang = getattr(s, "language", None)
            if lang != cached[i]:
                buf[i].copy_(self.get_base_blocked_cpu(lang).to(self.device))
                cached[i] = lang
        return buf

    def get_batch_mask_scratch(self, BS: int) -> Tensor:
        """Return a persistent [BS, V] bool scratch tensor on GPU. Caller fills it."""
        buf = self._batch_mask_gpu
        if buf is None or buf.shape[0] != BS:
            buf = torch.empty(
                (BS, self.vocab_size), dtype=torch.bool, device=self.device
            )
            self._batch_mask_gpu = buf
        return buf

    def get_additive_scratch(self, BS: int, dtype: torch.dtype) -> Tensor:
        """Return a persistent [BS, V] scratch tensor on GPU for the additive mask."""
        buf = self._additive_mask_gpu
        if buf is None or buf.shape[0] != BS or self._additive_dtype != dtype:
            buf = torch.empty((BS, self.vocab_size), dtype=dtype, device=self.device)
            self._additive_mask_gpu = buf
            self._additive_dtype = dtype
        return buf

    def get_cached_timing_set(self) -> set[int]:
        """Get cached timing token IDs as a Python set (no GPU sync)."""
        return self.timings_set


    def get_timing_tokens_in_range_gpu(
        self, start: float, end: float | None = None
    ) -> Tensor:
        key = str(self.device)
        if key not in self._device_tensors:
            self.init_gpu_tensors()

        timing_ids = self._device_tensors[key]["timing_token_ids"]
        start_idx = max(0, round(start / self.timing_step))
        if end is None:
            return timing_ids[start_idx:]
        end_idx = min(self.max_timing_idx + 1, round(end / self.timing_step) + 1)
        return timing_ids[start_idx:end_idx]


    def tokens_containing(self, t: str) -> list[int]:
        if t not in self._contains_cache:
            self._contains_cache[t] = [
                idx for idx, token in self.vocab.items() if t in token
            ]
        return self._contains_cache[t]

    def timings_in_range(self, start: float, end: float | None = None) -> list[int]:
        if end is not None:
            return [
                self.timings_values[k] for k in self.timings_values if start <= k <= end
            ]
        return [self.timings_values[k] for k in self.timings_values if k >= start]

    def token_id(self, char: str) -> int:
        """Get token ID for a single character. Returns -1 if not found."""
        return self.revocab.get(char, -1)


def sample_logits(
    logits: Tensor,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> Tensor:
    if temperature == 0.0:
        return logits.argmax(dim=-1)

    # In fp32: fp16 masked logits (-65504, x1.2 by the repetition penalty) / temperature overflow to
    # -inf, and a row with nothing left (an idle slot, a fully masked state) softmaxes to NaN, which
    # multinomial rejects with a device-side assert for the whole batch. Such rows take argmax.
    greedy = logits.argmax(dim=-1)
    all_masked = logits.max(dim=-1).values <= -60000
    logits = logits.float() / temperature

    if top_k is not None:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, -torch.inf)

    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, -torch.inf)

    probs = torch.softmax(logits, dim=-1)
    bad = ~torch.isfinite(probs).all(dim=-1) | all_masked
    probs = torch.where(bad.unsqueeze(-1), torch.ones_like(probs), probs)
    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return torch.where(bad, greedy, sampled)


_TOKEN_BYTES_CACHE: dict[int, bytes] = {}


def _token_bytes(token_id: int) -> bytes:
    """Raw bytes of a vocab token (for byte-fragment reassembly)."""
    b = _TOKEN_BYTES_CACHE.get(token_id)
    if b is None:
        from whisperdrz.tokenizer import get_tokenizer

        try:
            b = get_tokenizer().encoding.decode_single_token_bytes(token_id)
        except Exception:
            b = b""
        _TOKEN_BYTES_CACHE[token_id] = b
    return b


def update_state_after_decode(
    state, newtext: str, newtoken_id: int, skip_cur: bool = False
):
    """Update InferenceState after decoding a token.

    Tracks token_count and last_token_id as Python ints (no GPU tensors).
    Automatically skips cur update for special tokens (language, transcribe, etc.)

    Byte-fragment tokens (partial UTF-8, per-token decode contains '�';
    reachable for languages whose mask unblocks them — see
    get_dont_predict_for_lang) are buffered as raw bytes and appended to
    `cur` only once they assemble a valid character, so `cur` never carries
    replacement chars for chars like word-initial ё.
    """
    state.last_token_id = newtoken_id
    state.token_count += 1
    rt = state.recent_tokens
    rt.append(newtoken_id)
    if len(rt) > 10:
        del rt[:-10]

    # Skip cur update for special tokens (language tokens like <|en|>, transcribe token)
    # These are handled specially and shouldn't appear in transcription output
    is_special = (
        newtext.startswith("<|") and newtext.endswith("|>") and "." not in newtext
    ) or newtext == ""

    if not skip_cur and not is_special:
        if "�" in newtext:
            pend = state._pending_bytes + _token_bytes(newtoken_id)
            try:
                state.cur += pend.decode("utf-8")
                state._pending_bytes = b""
            except UnicodeDecodeError as e:
                if e.reason == "unexpected end of data" and len(pend) < 8:
                    state._pending_bytes = pend  # still mid-character
                else:
                    state._pending_bytes = b""  # invalid sequence — drop
        else:
            if state._pending_bytes:
                state._pending_bytes = b""  # interrupted fragment — drop partial
            state.cur += newtext

    cur = state.cur
    line = _get_line(cur)

    is_new_timing = (
        newtext.startswith("<|") and newtext.endswith("|>") and "." in newtext
    )
    is_newline = "\n" in newtext

    if is_newline:
        # Stop if line that just ended had timestamps but no words/tags
        if state._timing_count > 0 and not _line_has_content(
            state.cur[: state.cur.rindex("\n")]
        ):
            state.status = "process"
            state.stop_reason = "empty_line"
        state._timing_count = 0
        state._line_cache = line
        state.last_timing = None
    elif is_new_timing:
        try:
            new_timing_val = float(newtext[2:-2])
            state._timing_count += 1
            state.last_timing = new_timing_val
            state._window_max_timing = max(state._window_max_timing, new_timing_val)
            state._line_cache = line
        except (ValueError, IndexError):
            state._line_cache = ""
    else:
        state._line_cache = line

    even_timings = state._timing_count % 2 == 0
    if state.last_timing is not None and even_timings:
        # Timestamps are relative to current window (0-based after each continuation).
        # Compare against remaining audio in this window, capped at 30s.
        remaining = (state.duration - state.offset) if state.duration else 30
        max_time = min(remaining, 30)
        # Stop if timing exceeds segment duration (check on even timings to complete pairs)
        if state.last_timing >= max_time - 0.1:
            state.status = "process"
            state.stop_reason = (
                f"timing_exceeded:{state.last_timing:.2f}>={max_time - 0.1:.2f}"
            )

    # Window-level audio exhaustion check (catches post-newline hallucination).
    # After newline, last_timing resets to None and _timing_count to 0, so the
    # per-line timing_exceeded check above won't fire. _window_max_timing persists
    # across lines within the same window. Only for final (padded) windows.
    if state._window_max_timing > 0 and state.duration and state._timing_count == 0:
        window_audio = state.duration - state.offset
        if window_audio < 30 and state._window_max_timing >= window_audio - 1.0:
            state.status = "process"
            state.stop_reason = f"audio_exhausted:{state._window_max_timing:.2f}>={window_audio - 1.0:.2f}"

    if "[cutoff]" in state.cur:
        time_remaining = None
        if state.last_timing is not None and state.duration:
            time_remaining = state.duration - state.offset - state.last_timing
        if time_remaining is not None and time_remaining > 2.0:
            state.cur = state.cur.replace("[cutoff]", "")
        else:
            state.status = "process"
            state.stop_reason = "cutoff_tag"


# =============================================================================
# BATCHED DECODE
# =============================================================================


def decode_batch(
    logits: Tensor,
    states: list,
    ds: DecoderState,
    temperature: float = 0.0,
    last_token_ids: list[int] | None = None,
    token_counts: list[int] | None = None,
    allow_tags: bool = False,
    allow_asterisks: bool = False,
    speaker_tags: bool = True,
    all_tokens_gpu: Tensor | None = None,
    slot_indices: list[int] | None = None,
) -> list[tuple[str, int]]:
    """
    GPU-optimized batched decode with grouped constraint application.

    Key optimization: Instead of building constraints per-state (O(BS) kernel launches),
    we classify states by constraint pattern and apply in groups (O(num_patterns) launches).

    Args:
        logits: [BS, V] or [BS, 1, V] logits tensor
        states: List of InferenceState objects
        ds: DecoderState with tokenizer and GPU tensors
        temperature: Sampling temperature (0 = greedy)
        last_token_ids: Optional list of last token IDs per state (Python ints, no GPU sync)
        token_counts: Optional list of token counts per state (Python ints, no GPU sync)
        all_tokens_gpu: Optional GPU tensor [BS_total, max_seq_len] for repetition detection
        slot_indices: Optional list of slot indices into all_tokens_gpu

    Phase 1: Classify states, collect indices by constraint pattern
    Phase 2: Build batch_mask [BS, V] with grouped operations
    Phase 3: Single batched argmax
    Phase 4: Timing corrections for states that need them
    """
    device = logits.device
    BS = len(states)
    V = logits.shape[-1]

    if logits.dim() == 3:
        logits = logits[:, -1, :]

    # Build Python lists if not provided (fallback for compatibility)
    if last_token_ids is None:
        last_token_ids = [s.tokens[-1].item() for s in states]
    if token_counts is None:
        token_counts = [len(s.tokens) for s in states]

    # Cache timing token set ONCE per batch (was being recomputed per-state per-token!)
    timing_set = ds.get_cached_timing_set()

    # All comparisons use Python lists - no GPU→CPU copies needed
    eot_id = ds.tok.eot
    eot_mask = [t == eot_id for t in last_token_ids]
    status_not_started = [s.status != "started" for s in states]

    # Pre-compute last_token_str for all states (avoids per-state vocab lookup)
    last_token_strs = [ds.vocab_list[t] for t in last_token_ids]

    # Results array - returns (text, token_id_int) pairs
    results: list[tuple[str, int] | None] = [None] * BS

    # ==========================================================================
    # PHASE 1: Classify states and handle special cases
    # ==========================================================================

    # Index lists for grouped constraint application
    need_sampling: list[int] = []
    contexts: list[dict] = [{}] * BS

    # Constraint groups (indices into states)
    lang_indices: list[int] = []
    line_start_digit_indices: list[int] = []  # len(line) == 1
    line_start_timing_indices: list[int] = []  # len(line) == 4
    normal_indices: list[int] = []

    # Constraint pattern groups for normal decoding
    in_bracket_indices: list[int] = []
    in_angle_indices: list[int] = []
    in_paren_indices: list[int] = []
    not_in_bracket_indices: list[int] = []
    not_in_angle_indices: list[int] = []
    not_in_paren_indices: list[int] = []
    last_was_timing_indices: list[int] = []
    last_was_timing_even_indices: list[
        int
    ] = []  # after timing, even count -> enable space only
    last_was_timing_even_first_word_indices: list[
        int
    ] = []  # after timing, even count, no content yet -> only bare space
    last_was_timing_even_timed_tag_indices: list[
        int
    ] = []  # after timing, even count, has timed tag but no word -> allow space/newline
    last_was_timing_odd_indices: list[
        int
    ] = []  # after timing, odd count -> block space
    last_was_space_indices: list[int] = []
    even_timings_not_after_timing_indices: list[int] = []
    force_timing_indices: list[int] = []  # near end, force timing
    force_timestamp_indices: list[int] = []  # too many words without timing
    last_char_punct_or_bracket_indices: list[int] = []
    block_open_bracket_indices: list[int] = []  # last char is punct or alphanum
    block_open_angle_indices: list[
        int
    ] = []  # last char not space OR >=2 angle tags on line
    block_angle_consecutive_indices: list[int] = []  # >=1 angle tag on line -> block ><
    in_asterisk_indices: list[int] = []  # inside *...*
    not_in_asterisk_indices: list[int] = []  # not inside *...*
    block_open_asterisk_indices: list[int] = []  # block * unless after space/timing
    allow_space_asterisk_indices: list[int] = []  # allow " *" after word or timing
    block_star_after_star_indices: list[int] = []  # last char is *: forbid * (no ** or close-then-open)
    block_space_in_asterisk_indices: list[int] = []  # inside *...*: forbid space tokens
    block_timing_in_asterisk_indices: list[int] = []  # inside *...*: forbid timing tokens
    close_bracket_even_timing_indices: list[
        int
    ] = []  # last_token_str == "]" and even_timings
    space_even_timing_indices: list[
        int
    ] = []  # last was space, even timing, has timed word -> force timing before [tag]
    block_punct_nonalnum_indices: list[
        int
    ] = []  # last char not alphanumeric -> block punct
    block_quote_indices: list[int] = []  # last token had " -> block more quotes

    # Timing correction data for phase 4
    forced_close_indices: set[int] = set()

    MAX_TOKENS = 448
    for i, state in enumerate(states):
        # Early exit: not started or max tokens reached
        if status_not_started[i] or token_counts[i] >= MAX_TOKENS:
            state.status = "process"
            if status_not_started[i]:
                state.stop_reason = "not_started"
            else:
                state.stop_reason = f"max_tokens:{token_counts[i]}>={MAX_TOKENS}"
            results[i] = ("<|endoftext|>", eot_id)
            continue

        # EOT check
        if eot_mask[i]:
            state.status = "process"
            state.stop_reason = "eot_predicted"
            results[i] = ("<|endoftext|>", eot_id)
            continue

        # Repetition detection: any pattern repeated 5+ times (ignoring timing tokens)
        # Only check every 10 tokens to reduce overhead (was major GPU util bottleneck)
        MIN_REPEATS = 5
        token_count = token_counts[i]
        if token_count >= MIN_REPEATS * 5 and token_count % 10 == 0:
            # Read from GPU buffer if available (one small copy every 10 steps)
            N = MIN_REPEATS * 20
            if all_tokens_gpu is not None and slot_indices is not None:
                start = max(0, token_count - N)
                recent = (
                    all_tokens_gpu[slot_indices[i], start:token_count].cpu().tolist()
                )
            else:
                recent = state.tokens[-N:].tolist()
            recent_no_timing = [t for t in recent if t not in timing_set]
            if len(recent_no_timing) >= MIN_REPEATS:
                for pattern_len in range(1, len(recent_no_timing) // MIN_REPEATS + 1):
                    check_len = MIN_REPEATS * pattern_len
                    if check_len > len(recent_no_timing):
                        break
                    tokens = recent_no_timing[-check_len:]
                    pattern = tokens[:pattern_len]
                    is_repeating = all(
                        tokens[j * pattern_len : (j + 1) * pattern_len] == pattern
                        for j in range(MIN_REPEATS)
                    )
                    if is_repeating:
                        repeated_text = ds.tok.decode(pattern)
                        state.status = "process"
                        state.stop_reason = (
                            f"repetition:{repeated_text!r}x{MIN_REPEATS}"
                        )
                        results[i] = ("<|endoftext|>", eot_id)
                        break
            if state.status == "process":
                continue

        # Language selection (first token) - use pre-computed token counts
        if token_counts[i] == 1:
            state.forced_timing_before_newline = False
            lang_indices.append(i)
            need_sampling.append(i)
            contexts[i] = {"phase": "language"}
            continue

        # Transcribe token (second token)
        if token_counts[i] == 2:
            results[i] = ("", ds.tok.transcribe)
            continue

        # Forced timestamp (must fire before predict_next so timing precedes deferred bracket)
        if state.force_timestamp_next:
            state.force_timestamp_next = False
            force_timestamp_indices.append(i)
            need_sampling.append(i)
            # Provide context for timing token selection
            cur = state.cur
            line = _get_line(cur)
            timing_matches = list(TIMINGS_FORMAT.finditer(line))
            last_timing = float(timing_matches[-1].group(1)) if timing_matches else None
            max_timing = (state.duration - state.offset) if state.duration else None
            contexts[i] = {
                "phase": "force_timestamp",
                "cur": cur,
                "last_timing": last_timing,
                "max_timing": max_timing,
            }
            continue

        # Deferred prediction
        if state.predict_next is not None:
            text = state.predict_next
            state.predict_next = None
            if text == "\n" and not _line_has_content(state.cur):
                state.status = "process"
                state.stop_reason = "empty_line_eot"
                results[i] = ("<|endoftext|>", eot_id)
            elif text == "<|endoftext|>":
                state.status = "process"
                state.stop_reason = "eot_deferred"
                results[i] = (text, eot_id)
            else:
                results[i] = (text, ds.tok.encode(text)[0])
            continue

        # Get state info
        cur = state.cur
        line = _get_line(cur)

        # Line start handling (first 5 chars for speaker tag)
        if speaker_tags and len(line) < 5:
            if len(line) == 0:
                results[i] = ("[", ds._single_char_ids["["])
            elif len(line) == 1:
                # First line of transcription -> force speaker 0
                if cur.strip() == "[":
                    results[i] = ("0", ds._single_char_ids["0"])
                else:
                    line_start_digit_indices.append(i)
                    need_sampling.append(i)
                    contexts[i] = {"phase": "line_start_digit"}
            elif len(line) == 2:
                results[i] = ("]", ds._single_char_ids["]"])
            elif len(line) == 3:
                results[i] = (" ", ds._single_char_ids[" "])
            elif len(line) == 4:
                line_start_timing_indices.append(i)
                need_sampling.append(i)
                # Calculate timing range for this state
                max_timing = (state.duration - state.offset) if state.duration else None
                # Only enforce monotonic timestamps if same speaker as previous line
                # Different speakers can overlap (speak at same time)
                prev_last_timing = None
                current_speaker = (
                    line[1] if len(line) > 1 else None
                )  # e.g. "0" from "[0] "
                prev_lines = cur.rstrip().split("\n")
                if len(prev_lines) >= 2:
                    prev_line = prev_lines[-2]
                    # Check if previous line has same speaker
                    if prev_line.startswith(f"[{current_speaker}]"):
                        all_timings = list(TIMINGS_FORMAT.finditer(cur))
                        prev_last_timing = (
                            parse_timing(all_timings[-1].group(0))
                            if all_timings
                            else None
                        )
                contexts[i] = {
                    "phase": "line_start_timing",
                    "cur": cur,
                    "max_timing": max_timing,
                    "last_timing": prev_last_timing,
                }
            continue

        # No speaker tags: at line start, go straight to timing
        if not speaker_tags and len(line) == 0:
            line_start_timing_indices.append(i)
            need_sampling.append(i)
            max_timing = (state.duration - state.offset) if state.duration else None
            # Allow timing to go back up to 5s for speaker overlap
            prev_last_timing = None
            all_timings = list(TIMINGS_FORMAT.finditer(cur))
            if all_timings:
                last_t = parse_timing(all_timings[-1].group(0))
                prev_last_timing = max(0, last_t - 5.0)
            contexts[i] = {
                "phase": "line_start_timing",
                "cur": cur,
                "max_timing": max_timing,
                "last_timing": prev_last_timing,
            }
            continue

        # Normal decoding - classify by constraint pattern
        normal_indices.append(i)
        need_sampling.append(i)

        # Cache timing info
        if line != getattr(state, "_line_cache", ""):
            matches = list(TIMINGS_FORMAT.finditer(line))
            state._timing_count = len(matches)
            state.last_timing = parse_timing(matches[-1].group(0)) if matches else None
            state._line_cache = line

        timing_count = state._timing_count
        even_timings = timing_count % 2 == 0
        last_timing = state.last_timing

        last_token_str = last_token_strs[i]  # Use pre-computed value
        last_was_timing = last_token_str.startswith("<|") and last_token_str.endswith(
            "|>"
        )
        last_was_space = line[-1] == " " if line else False
        last_char = line[-1] if line else ""

        # Time remaining calculations
        duration = state.duration
        offset = state.offset
        time_remaining = (
            (duration - offset - last_timing)
            if (duration and last_timing is not None)
            else None
        )
        near_end = time_remaining is not None and time_remaining < 3
        approaching_end = time_remaining is not None and time_remaining < 5

        # Words since timing (always calculate for sparse mode)
        words_since_timing = 0
        if timing_count > 0:
            last_timing_end = line.rfind("|>")
            if last_timing_end != -1:
                text_after = line[last_timing_end + 2 :]
                cleaned = _BRACKETS_CLEAN.sub("", text_after)
                words_since_timing = len([w for w in cleaned.split() if w.strip()])

        # Store context for phase 4 corrections
        max_timing = (duration - offset) if duration else None
        ctx = {
            "phase": "normal",
            "line": line,
            "cur": cur,
            "last_timing": last_timing,
            "last_token_str": last_token_str,
            "even_timings": even_timings,
            "last_was_timing": last_was_timing,
            "last_was_space": last_was_space,
            "words_since_timing": words_since_timing,
            "max_timing": max_timing,
            "duration": duration,
            "offset": offset,
        }
        contexts[i] = ctx

        # Bracket states
        in_bracket = line.rfind("[") > line.rfind("]")
        tag_opens = list(_TAG_OPEN.finditer(line))
        tag_closes = list(_TAG_CLOSE.finditer(line))
        in_angle = (tag_opens[-1].start() if tag_opens else -1) > (
            tag_closes[-1].start() if tag_closes else -1
        )
        in_paren = line.rfind("(") > line.rfind(")")
        # Asterisk state: odd count means we're inside `*...`
        in_asterisk, asterisk_force_close_reason = asterisk_state(line)

        # Force timing near end OR before first word (if only tags timed so far).
        # force_timing_every_word opens EVERY word/tag with a timestamp; the
        # close-before-space/newline corrections then finish each pair, so the
        # output is <|t|>word<|t|> <|t|>word2<|t|> with nothing left untimed.
        force_timing = last_was_space and (
            near_end
            or (approaching_end and words_since_timing >= 2)
            or (even_timings and not _has_timed_word(line))
            or (even_timings and state.force_timing_every_word)
        )

        # Classify into constraint groups
        if force_timing:
            force_timing_indices.append(i)
        elif last_was_timing:
            last_was_timing_indices.append(i)
            if even_timings:
                # After even timing (word/tag ended), check if first word needs timing
                if not _has_timed_word(line):
                    # No word timed yet
                    if _has_timed_content(line):
                        # Has timed tag (e.g. [laugh]) but no word -> allow space/newline
                        last_was_timing_even_timed_tag_indices.append(i)
                    else:
                        # No content at all -> only allow bare space
                        last_was_timing_even_first_word_indices.append(i)
                else:
                    last_was_timing_even_indices.append(i)
            else:
                last_was_timing_odd_indices.append(i)
        elif last_was_space:
            last_was_space_indices.append(i)
            # After space with even timing and timed word -> force timing before [tag]
            # Ensures pattern: word<|t|> <|t|>[tag] (timing before inline tags)
            if even_timings and _has_timed_word(line):
                space_even_timing_indices.append(i)

        if in_bracket:
            bracket_pos = line.rfind("[")
            partial = line[bracket_pos + 1 :]
            if (partial and partial not in VALID_BRACKET_PREFIXES) or len(
                partial
            ) > _BRACKET_MAX_LEN:
                results[i] = ("]", ds._single_char_ids["]"])
                forced_close_indices.add(i)
                continue
            in_bracket_indices.append(i)
        else:
            not_in_bracket_indices.append(i)

        if in_angle:
            in_angle_indices.append(i)
        else:
            not_in_angle_indices.append(i)

        if in_paren:
            in_paren_indices.append(i)
        else:
            not_in_paren_indices.append(i)

        if in_asterisk:
            # Force-close (mirror of bracket force-close): if the body since
            # the last `*` has gone invalid (contains a space/newline, or has
            # grown too long), emit the closing `*` directly so the open span
            # gets sealed off instead of bleeding into the next phrase.
            if asterisk_force_close_reason is not None:
                results[i] = ("*", ds._single_char_ids["*"])
                forced_close_indices.add(i)
                continue
            in_asterisk_indices.append(i)
            block_space_in_asterisk_indices.append(i)
            block_timing_in_asterisk_indices.append(i)
        else:
            not_in_asterisk_indices.append(i)

        # Block `*` whenever the previous char was `*`. Covers both `**`
        # (back-to-back open) and the close-then-open case `*word**word*`.
        if last_char == "*":
            block_star_after_star_indices.append(i)

        if even_timings and not last_was_space and not force_timing:
            even_timings_not_after_timing_indices.append(i)

        if last_char in ".,!?-'" or last_token_str == "]":
            last_char_punct_or_bracket_indices.append(i)

        if is_word_char(last_char) or last_char in ".,!?-'":
            block_open_bracket_indices.append(i)

        # Block < unless last char is space AND < 2 angle tags on line
        # This enforces " <" pattern and limits to 2 tags per line
        if last_char != " " or len(tag_opens) >= 2:
            block_open_angle_indices.append(i)

        # Block >< unless we have exactly 1 unclosed tag (allows <tag1><tag2> pattern)
        # in_angle means we're inside an unclosed <...
        if not (in_angle and len(tag_opens) == 1):
            block_angle_consecutive_indices.append(i)

        # Block * unless after space or timing (like brackets but no timing required)
        if not (last_was_space or last_was_timing):
            block_open_asterisk_indices.append(i)

        # Allow " *" after word or timing (for patterns like "word *emphasis*")
        if is_word_char(last_char) or last_was_timing:
            allow_space_asterisk_indices.append(i)

        # Block punct tokens unless last char is word-like (e.g. "word." ok, "]." not ok).
        # Word-like covers Unicode combining marks too (Hindi/Thai vowel marks).
        if not is_word_char(last_char):
            block_punct_nonalnum_indices.append(i)

        # Block quotes after a quote token (prevents "" loops)
        if '"' in last_token_str:
            block_quote_indices.append(i)

        # After ] -> force timing prediction before next content
        # This prevents [pause] [pause] loops by requiring <|t|> after each tag
        if last_token_str == "]":
            close_bracket_even_timing_indices.append(i)

    # ==========================================================================
    # PHASE 1.5: Compute derived groups
    # ==========================================================================

    block_eot_indices = []
    block_newline_indices = []
    NEWLINE_TOKEN = ds._single_char_ids["\n"]
    for i in normal_indices:
        ctx = contexts[i]
        has_content = ctx.get("last_timing") is not None
        if not has_content:
            tr = (
                ctx.get("duration", 0) - ctx.get("offset", 0)
                if ctx.get("duration")
                else None
            )
            if tr is not None and tr > 2.0:
                block_eot_indices.append(i)
        line = ctx.get("line", "")
        after_speaker = (
            line.split("] ", 1)[-1] if (speaker_tags and "] " in line) else line
        )
        without_timestamps = TIMINGS_FORMAT.sub("", after_speaker)
        has_line_content = (
            text_has_word(without_timestamps) or "[" in without_timestamps
        )
        if not has_line_content or not ctx.get("even_timings", True):
            block_newline_indices.append(i)

    BRACKET_OPEN = 58
    BRACKET_CLOSE = 60
    SPACE_BRACKET = 542
    CUT_TOKEN = 6672
    block_cut_indices = [
        i for i, t in enumerate(last_token_ids) if t in (BRACKET_OPEN, SPACE_BRACKET)
    ]
    block_bracket_after_close = [
        i for i, t in enumerate(last_token_ids) if t == BRACKET_CLOSE
    ]

    if allow_asterisks:
        block_open_asterisk_set = set(block_open_asterisk_indices)
        block_asterisk_combined = [
            i for i in not_in_asterisk_indices if i in block_open_asterisk_set
        ]
    else:
        block_asterisk_combined = []

    special_timing_indices = (
        force_timing_indices
        + force_timestamp_indices
        + line_start_timing_indices
        + close_bracket_even_timing_indices
    )
    special_timing_set = set(special_timing_indices)

    # ==========================================================================
    # PHASE 2: Build batch_mask on CPU, transfer once to GPU
    # ==========================================================================

    if not need_sampling:
        return results

    # All mask construction on GPU — predefined mask tensors live in _device_tensors
    # so rule application is a bitwise op on existing GPU memory (no H2D per step).
    cm = ds.get_gpu_tensor
    # Per-slot base_blocked picked by each state's current language, so non-Latin
    # scripts aren't blocked when transcribing e.g. ru/zh/ar/etc. Persistent buffer
    # is updated in-place only when a slot's language changes (rare).
    slot_base = ds.update_slot_base_buffer(states)
    # Reuse a persistent scratch buffer instead of allocating per step.
    batch_mask = ds.get_batch_mask_scratch(BS)
    batch_mask.copy_(slot_base)

    # Unblock angle bracket tokens when allow_tags=True
    if allow_tags:
        batch_mask &= ~cm("angle_blocked_mask")
        if block_open_angle_indices:
            batch_mask[block_open_angle_indices] |= cm("block_less_than")
        if block_angle_consecutive_indices:
            batch_mask[block_angle_consecutive_indices] |= cm("block_angle_consecutive")

    # Unblock asterisk tokens when allow_asterisks=True
    if allow_asterisks:
        batch_mask &= ~cm("asterisk_blocked_mask")
        if block_asterisk_combined:
            batch_mask[block_asterisk_combined] |= cm("block_asterisk")
        # Block `*` whenever previous char was `*` (no `**`, no `*x**y*`).
        if block_star_after_star_indices:
            batch_mask[block_star_after_star_indices] |= cm("asterisk_blocked_mask")
        # Inside `*...*`: block space-containing tokens (close before space).
        if block_space_in_asterisk_indices:
            batch_mask[block_space_in_asterisk_indices] |= cm("block_contains_space")
        # Inside `*...*`: block timing tokens (no `<|t|>` inside `*word*`).
        if block_timing_in_asterisk_indices:
            batch_mask[block_timing_in_asterisk_indices] |= cm("block_timings")

    # Single-token blocking
    if block_eot_indices:
        batch_mask[block_eot_indices, ds.tok.eot] = True
    if block_newline_indices:
        batch_mask[block_newline_indices, NEWLINE_TOKEN] = True
    if block_cut_indices:
        batch_mask[block_cut_indices, CUT_TOKEN] = True
    BRACKET_CLOSE_OPEN = 7131
    batch_mask[:, BRACKET_CLOSE_OPEN] = True
    if block_bracket_after_close:
        batch_mask[block_bracket_after_close, BRACKET_OPEN] = True
        batch_mask[block_bracket_after_close, SPACE_BRACKET] = True

    # Language selection: block all, enable only lang tokens
    if lang_indices:
        batch_mask[lang_indices] = True
        batch_mask[lang_indices] &= ~cm("lang_enabled")

    # Line start digit: block all, enable only speaker IDs (0-9, c)
    if line_start_digit_indices:
        batch_mask[line_start_digit_indices] = True
        batch_mask[line_start_digit_indices] &= ~cm("speaker_id_mask")

    # Last was timing -> block timings, punct, (
    if last_was_timing_indices:
        batch_mask[last_was_timing_indices] |= cm("after_timing_blocked")

    # Last was timing + even -> enable only space+word tokens
    BARE_SPACE = 220
    if last_was_timing_even_indices:
        batch_mask[last_was_timing_even_indices] = True
        batch_mask[last_was_timing_even_indices] &= ~cm("space_starting_enabled")
        # Re-apply per-slot base_blocked (lang-aware) so non-Latin tokens stay correctly
        # (un)blocked for this subset of slots. Reads from persistent buffer, no copy.
        batch_mask[last_was_timing_even_indices] |= slot_base[
            last_was_timing_even_indices
        ]
        batch_mask[last_was_timing_even_indices, 542] = True  # Block ' ['

    # Last was timing + even + no content yet -> only allow bare space
    if last_was_timing_even_first_word_indices:
        batch_mask[last_was_timing_even_first_word_indices] = True
        batch_mask[last_was_timing_even_first_word_indices, BARE_SPACE] = False

    # Last was timing + even + has timed tag but no word -> allow space and newline
    if last_was_timing_even_timed_tag_indices:
        batch_mask[last_was_timing_even_timed_tag_indices] = True
        batch_mask[last_was_timing_even_timed_tag_indices, BARE_SPACE] = False
        batch_mask[last_was_timing_even_timed_tag_indices, NEWLINE_TOKEN] = False

    # Last was timing + odd -> block space starting
    if last_was_timing_odd_indices:
        batch_mask[last_was_timing_odd_indices] |= cm("block_space_starting")

    # Last was space -> block punct and block space (no consecutive spaces)
    if last_was_space_indices:
        batch_mask[last_was_space_indices] |= cm("block_punct")
        batch_mask[last_was_space_indices] |= cm("block_space_starting")

    # Block punct tokens unless last char is alphanumeric
    if block_punct_nonalnum_indices:
        batch_mask[block_punct_nonalnum_indices] |= cm("block_punct")

    # Block quote tokens after a quote (prevents "" degeneration)
    if block_quote_indices:
        batch_mask[block_quote_indices] |= cm("block_quote")

    # In bracket -> only allow tokens that form valid tag prefixes (per-state)
    if in_bracket_indices:
        close_bracket_id = ds._single_char_ids["]"]
        bracket_allowed_cpu = ds._cpu_masks["bracket_allowed"]
        for i in in_bracket_indices:
            line = contexts[i].get("line", "")
            bracket_pos = line.rfind("[")
            partial = line[bracket_pos + 1 :]
            allowed = bracket_allowed_cpu.get(partial, [])
            allow_mask = torch.zeros(V, dtype=torch.bool, device=device)
            valid = [t for t in allowed if t < V]
            # Fall back to bare close bracket when no valid continuations exist.
            # Checking `valid` on CPU avoids a GPU sync that `allow_mask.any()` triggers.
            if valid:
                allow_mask[valid] = True
            else:
                allow_mask[close_bracket_id] = True
            batch_mask[i] |= ~allow_mask

    # Not in bracket -> block ]
    if not_in_bracket_indices:
        batch_mask[not_in_bracket_indices] |= cm("block_contains_close_bracket")

    # In angle -> block spaces, punct, <, timings, digits, *, [
    if in_angle_indices:
        batch_mask[in_angle_indices] |= cm("in_angle_blocked")
        batch_mask[in_angle_indices] |= cm("block_asterisk")
        batch_mask[in_angle_indices] |= cm("space_asterisk_token")
        batch_mask[in_angle_indices] |= cm("block_open_bracket")

    # Not in angle -> block >
    if not_in_angle_indices:
        batch_mask[not_in_angle_indices] |= cm("block_greater_than")

    # In asterisk -> block [ and <
    if in_asterisk_indices:
        batch_mask[in_asterisk_indices] |= cm("block_open_bracket")
        batch_mask[in_asterisk_indices] |= cm("block_less_than")

    # In paren -> block spaces, punct, (, timings
    if in_paren_indices:
        batch_mask[in_paren_indices] |= cm("in_paren_blocked")

    # Not in paren -> block )
    if not_in_paren_indices:
        batch_mask[not_in_paren_indices] |= cm("block_close_paren")

    # Even timings (not after timing) -> block timings
    if even_timings_not_after_timing_indices:
        batch_mask[even_timings_not_after_timing_indices] |= cm("block_timings")

    # Last char punct or ] -> block punct
    if last_char_punct_or_bracket_indices:
        batch_mask[last_char_punct_or_bracket_indices] |= cm("block_punct")

    # Last char is punct or alphanum -> block [
    if block_open_bracket_indices:
        batch_mask[block_open_bracket_indices] |= cm("block_open_bracket")

    # Block timing tokens <= last_timing per state (prevents backwards timestamps)
    timing_token_ids_cpu = ds._cpu_masks["timing_token_ids"]
    for i in normal_indices:
        ctx = contexts[i]
        last_timing = ctx.get("last_timing")
        if last_timing is not None and last_timing > 0:
            end_block_idx = round(last_timing / ds.timing_step) + 1
            if end_block_idx > 0:
                batch_mask[i, timing_token_ids_cpu[:end_block_idx]] = True

    # Unblock " *" after word/timing when allow_asterisks=True (must come last)
    if allow_asterisks and allow_space_asterisk_indices:
        batch_mask[allow_space_asterisk_indices] &= ~cm("space_asterisk_token")

    # Special timing states: block everything, then unblock valid timing range on CPU
    # This replaces per-state _sample_masked calls with batched CPU mask construction
    timing_token_ids_list = ds._cpu_masks["timing_token_ids"]
    if special_timing_indices:
        batch_mask[special_timing_indices] = True
        for i in special_timing_indices:
            ctx = contexts[i]
            state = states[i]
            cur = ctx.get("cur", "") or state.cur
            line = _get_line(cur)
            max_timing = ctx.get("max_timing") or (
                (state.duration - state.offset) if state.duration else None
            )
            min_timing_floor = ctx.get("last_timing")
            s_idx, e_idx = _get_timing_range_cpu(
                cur, line, max_timing, ds, min_timing=min_timing_floor
            )
            if s_idx < e_idx:
                batch_mask[i, timing_token_ids_list[s_idx:e_idx]] = False

    # batch_mask already on GPU — no transfer needed (all ops above stay on device)

    # ==========================================================================
    # PHASE 3: Sample
    # ==========================================================================

    # Convert to sets for O(1) lookup in loops
    space_even_timing_set = set(space_even_timing_indices)

    # Convert bool mask to additive and apply using persistent scratch buffer.
    # -65504 (max finite for fp16) instead of -inf avoids NaN issues.
    additive_mask = ds.get_additive_scratch(BS, logits.dtype)
    additive_mask.zero_()
    additive_mask.masked_fill_(batch_mask, -65504)
    masked_logits = logits + additive_mask

    # Apply repetition penalty to prevent hallucination loops (e.g. "Our.Our.Our...")
    # Batched: build penalty mask on CPU, one transfer + two vectorized where ops
    REP_PENALTY = 1.2
    penalty_states = [
        i
        for i in need_sampling
        if i not in special_timing_set
        and i not in forced_close_indices
        and token_counts[i] > 0
    ]
    if penalty_states:
        # Build penalty mask on CPU, one transfer + two vectorized where ops
        penalty_rows = []
        penalty_cols = []
        for i in penalty_states:
            for t in set(states[i].recent_tokens):
                if 0 <= t < V:
                    penalty_rows.append(i)
                    penalty_cols.append(t)
        if penalty_rows:
            penalty_mask = torch.zeros(BS, V, dtype=torch.bool)
            penalty_mask[penalty_rows, penalty_cols] = True
            penalty_mask_gpu = penalty_mask.to(device)
            penalized = torch.where(
                masked_logits > 0,
                masked_logits / REP_PENALTY,
                masked_logits * REP_PENALTY,
            )
            masked_logits = torch.where(penalty_mask_gpu, penalized, masked_logits)

    # Sample all at once
    all_tokens = sample_logits(masked_logits, temperature)

    # Store results for all sampling states (including special timing, now in batch mask)
    # Use .cpu().numpy() for one batched memcpy, then return Python ints
    all_tokens_np = all_tokens.cpu().numpy()
    for i in need_sampling:
        if i not in forced_close_indices:
            tok_id = int(all_tokens_np[i])
            text = ds.vocab_list[tok_id]
            results[i] = (text, tok_id)

    # Update language state
    for i in lang_indices:
        if results[i] is not None:
            text = results[i][0]
            states[i].language = text.replace("<|", "").replace("|>", "")

    # ==========================================================================
    # PHASE 4: Corrections (timing before newline/EOT, unclosed brackets, etc.)
    # Masks built on CPU, deferred resampling batched at end
    # ==========================================================================

    resample_indices: list[int] = []
    resample_masks: list[Tensor] = []

    for i in normal_indices:
        if results[i] is None or i in forced_close_indices:
            continue

        newtext, _ = results[i]
        ctx = contexts[i]
        state = states[i]

        last_token_str = ctx.get("last_token_str", "")
        last_was_timing = ctx.get("last_was_timing", False)
        last_was_space = ctx.get("last_was_space", False)
        line = ctx.get("line", "")
        max_timing = ctx.get("max_timing")

        last_non_space_was_timing = last_was_timing or (
            last_was_space and line.rstrip().endswith("|>")
        )

        is_eot = newtext == "<|endoftext|>"

        # Reject EOT only before model has produced content
        # Once content exists (has_timing), allow EOT so model can stop when speech ends
        if is_eot:
            has_timing = ctx.get("last_timing") is not None
            raw_remaining = ctx.get("duration", 0) - ctx.get("offset", 0)
            # Only block EOT if no content yet and significant audio remains
            should_block_eot = not has_timing and raw_remaining > 2.0
            if should_block_eot:
                blocked_cpu = slot_base[i].clone()
                if allow_tags:
                    blocked_cpu &= ~cm("angle_blocked_mask")
                    last_char_resample = line[-1:] if line else ""
                    angle_tag_count = len(_TAG_OPEN.findall(line))
                    if last_char_resample != " " or angle_tag_count >= 2:
                        blocked_cpu |= cm("block_less_than")
                    tag_closes_count = len(_TAG_CLOSE.findall(line))
                    in_angle_resample = angle_tag_count > tag_closes_count
                    if not (in_angle_resample and angle_tag_count == 1):
                        blocked_cpu |= cm("block_angle_consecutive")
                blocked_cpu[ds.tok.eot] = True
                last_char_resample = line[-1:] if line else ""
                if not is_word_char(last_char_resample):
                    blocked_cpu |= cm("block_punct")
                results[i] = _sample_masked(logits[i], blocked_cpu, temperature, ds)
                newtext = results[i][0]
                is_eot = False

        next_is_space = newtext and newtext[0] == " "
        next_is_timing = bool(TIMINGS_FORMAT.match(newtext)) if newtext else False

        correction_needed = False
        force_timing_sample = False

        # Force timing before newline/EOT - but close brackets first
        if (newtext == "\n" or is_eot) and not last_non_space_was_timing:
            unclosed_bracket, unclosed_angle, unclosed_paren, unclosed_asterisk = (
                _get_unclosed(line)
            )

            if last_token_str != "]" and unclosed_bracket:
                # Close bracket first - don't set predict_next, let model try newline again
                # which will then trigger timing-before-newline on next step
                results[i] = ("]", ds._single_char_ids["]"])
                correction_needed = True
            elif last_token_str != ">" and unclosed_angle:
                # Close angle first
                results[i] = (">", ds._single_char_ids[">"])
                correction_needed = True
            elif last_token_str != ")" and unclosed_paren:
                # Close paren first
                results[i] = (")", ds._single_char_ids[")"])
                correction_needed = True
            elif last_token_str != "*" and unclosed_asterisk and allow_asterisks:
                # Close asterisk first
                results[i] = ("*", ds._single_char_ids["*"])
                correction_needed = True
            else:
                # No unclosed brackets, force timing before newline
                state.predict_next = newtext
                force_timing_sample = True
                correction_needed = True

        # Force timing before (
        if not correction_needed and (
            newtext == " (" or (newtext == "(" and not last_was_timing)
        ):
            state.predict_next = newtext
            force_timing_sample = True
            correction_needed = True

        # Force timing before [ tag (ensures tags have timing context)
        # Skip if last token was timing - close_bracket logic already forced one
        if (
            not correction_needed
            and not last_was_timing
            and (
                newtext == "["
                or (newtext.startswith(" [") and not newtext.startswith(" []"))
            )
        ):
            if newtext.startswith(" ["):
                # Emit space first so timing doesn't jam against punctuation
                results[i] = (" ", ds._single_char_ids[" "])
                state.predict_next = newtext[1:]
                state.force_timestamp_next = True
            else:
                state.predict_next = newtext
                force_timing_sample = True
            correction_needed = True

        # Handle unclosed brackets when predicting space
        if not correction_needed and next_is_space:
            unclosed_bracket, unclosed_angle, unclosed_paren, unclosed_asterisk = (
                _get_unclosed(line)
            )
            even_timings = ctx.get("even_timings", True)

            if last_token_str != "]" and unclosed_bracket:
                results[i] = ("]", ds._single_char_ids["]"])
                correction_needed = True
            elif last_token_str != ">" and unclosed_angle:
                results[i] = (">", ds._single_char_ids[">"])
                correction_needed = True
            elif last_token_str != ")" and unclosed_paren:
                results[i] = (")", ds._single_char_ids[")"])
                correction_needed = True
            elif last_token_str != "*" and unclosed_asterisk and allow_asterisks:
                results[i] = ("*", ds._single_char_ids["*"])
                correction_needed = True
            elif not even_timings:
                # Must close open timestamp before space (both dense and sparse)
                force_timing_sample = True
                correction_needed = True

        # Force space near end
        if not correction_needed and next_is_space and newtext != " ":
            _, near_end, approaching_end = _get_timing_status(ctx)
            words_since = ctx.get("words_since_timing", 0)
            if near_end or (approaching_end and words_since >= 2):
                results[i] = (" ", ds._single_char_ids[" "])
                correction_needed = True

        # Force timestamp after too many words (prevent untethered generation/repetition)
        # When model predicts a space-starting token after 5+ untimed words:
        # 1. Output just the space now
        # 2. Force timestamp on next iteration
        # force_timing_every_word does this for EVERY space-prefixed token: a
        # " versus" token would otherwise start the word with no opening
        # timestamp. Splitting (rather than masking " versus") keeps the model's
        # choice of a word over a newline; it re-predicts the word after <|t|>.
        MAX_WORDS_WITHOUT_TIMING = 5
        if not correction_needed and next_is_space:
            even_timings = ctx.get("even_timings", True)
            words_since = ctx.get("words_since_timing", 0)
            if state.force_timing_every_word:
                split = newtext != " "
            else:
                split = words_since >= MAX_WORDS_WITHOUT_TIMING
            if split and even_timings:
                results[i] = (" ", ds._single_char_ids[" "])
                state.force_timestamp_next = True
                correction_needed = True

        # Sparse mode: reject unwanted opening timestamps and re-sample without timestamps
        # This preserves the model's natural space vs newline distribution
        # Skip if this state was in space_even_timing_indices (we forced timing before [tag])
        if (
            not correction_needed
            and not state.force_timing_every_word
            and i not in space_even_timing_set
        ):
            even_timings = ctx.get("even_timings", True)
            last_was_space = ctx.get("last_was_space", False)
            words_since = ctx.get("words_since_timing", 0)

            # Check if model predicted an opening timestamp (after space with even timings)
            is_opening_timestamp = (
                newtext.startswith("<|")
                and newtext.endswith("|>")
                and even_timings
                and last_was_space
            )

            if is_opening_timestamp:
                # Check if this is a wanted timestamp position
                time_remaining, near_end, approaching_end = _get_timing_status(ctx)
                is_periodic = words_since >= 3
                is_forced = near_end or (approaching_end and words_since >= 2)

                # If not wanted, reject and re-sample without timestamps
                if not is_periodic and not is_forced:
                    # Build mask on CPU, defer to batched resample
                    blocked_cpu = (slot_base[i] | cm("block_timings")).clone()
                    if allow_tags:
                        blocked_cpu &= ~cm("angle_blocked_mask")
                        last_char_resample = line[-1:] if line else ""
                        angle_tag_count = len(_TAG_OPEN.findall(line))
                        if last_char_resample != " " or angle_tag_count >= 2:
                            blocked_cpu |= cm("block_less_than")
                        tag_closes_count = len(_TAG_CLOSE.findall(line))
                        in_angle_resample = angle_tag_count > tag_closes_count
                        if not (in_angle_resample and angle_tag_count == 1):
                            blocked_cpu |= cm("block_angle_consecutive")
                    if last_was_space:
                        blocked_cpu |= cm("block_space_starting")
                    if time_remaining is not None and time_remaining > 2.0:
                        blocked_cpu[ds.tok.eot] = True
                    last_char_resample = line[-1:] if line else ""
                    if not is_word_char(last_char_resample):
                        blocked_cpu |= cm("block_punct")
                    resample_indices.append(i)
                    resample_masks.append(blocked_cpu)
                    correction_needed = True

        # Check for backwards timing within the current line
        # Use ctx["last_timing"] which is parsed from the line (always accurate)
        if not correction_needed and next_is_timing:
            last_timing_in_line = ctx.get("last_timing")
            if (
                last_timing_in_line is not None
                and newtext.startswith("<|")
                and newtext.endswith("|>")
            ):
                predicted_timing = parse_timing(newtext)
                if (
                    predicted_timing is not None
                    and predicted_timing < last_timing_in_line
                ):
                    force_timing_sample = True
                    correction_needed = True

        # Re-sample timing if needed (but NOT if last token was already a timing)
        if not correction_needed and next_is_timing and not last_was_timing:
            force_timing_sample = True
            correction_needed = True

        # Sample timing token with per-line timing floor
        if force_timing_sample:
            cur = ctx.get("cur", "")
            min_timing_floor = ctx.get("last_timing")
            s_idx, e_idx = _get_timing_range_cpu(
                cur, line, max_timing, ds, min_timing=min_timing_floor
            )
            blocked_mask = torch.ones(V, dtype=torch.bool, device=device)
            timing_ids = ds._cpu_masks["timing_token_ids"]
            if s_idx < e_idx:
                blocked_mask[timing_ids[s_idx:e_idx]] = False
            resample_indices.append(i)
            resample_masks.append(blocked_mask)

    # Batch resample all Phase 4 corrections (force_timing + sparse_rejection)
    if resample_indices:
        stacked = torch.stack(resample_masks)
        idx_t = torch.tensor(resample_indices, dtype=torch.long, device=device)
        correction_logits = logits.index_select(0, idx_t)
        additive = stacked.to(correction_logits.dtype) * -65504
        corrected = correction_logits + additive
        corrected_tokens = sample_logits(corrected, temperature)
        corrected_np = corrected_tokens.cpu().numpy()
        for j, i in enumerate(resample_indices):
            tok_id = int(corrected_np[j])
            results[i] = (ds.vocab_list[tok_id], tok_id)

    return results


def _sample_masked(
    logits_row: Tensor,
    blocked_mask: Tensor,
    temperature: float,
    ds: DecoderState,
) -> tuple[str, int]:
    """Sample from logits with blocked mask applied. Returns (text, token_id_int)."""
    NEGINF = float("-inf")
    row_logits = logits_row.clone()
    row_logits[blocked_mask] = NEGINF

    if temperature == 0.0:
        tok = row_logits.argmax()
    else:
        tok = sample_logits(row_logits.unsqueeze(0), temperature).squeeze()

    tok_id = tok.item()
    text = ds.vocab_list[tok_id]
    return (text, tok_id)


def _get_unclosed(line: str) -> tuple[bool, bool, bool, bool]:
    """Check for unclosed brackets, angles, parens, asterisks in line."""
    unclosed_bracket = line.count("[") != line.count("]")
    unclosed_angle = len(_TAG_OPEN.findall(line)) != len(_TAG_CLOSE.findall(line))
    unclosed_paren = line.count("(") != line.count(")")
    unclosed_asterisk = line.count("*") % 2 == 1  # Odd count = unclosed
    return unclosed_bracket, unclosed_angle, unclosed_paren, unclosed_asterisk


def _get_timing_status(ctx: dict) -> tuple[float, bool, bool]:
    """Calculate time_remaining, near_end, approaching_end from context."""
    time_remaining = (
        ctx.get("duration", 0) - ctx.get("offset", 0) - (ctx.get("last_timing") or 0)
    )
    has_timing = ctx.get("last_timing") is not None
    near_end = time_remaining < 3 if has_timing else False
    approaching_end = time_remaining < 5 if has_timing else False
    return time_remaining, near_end, approaching_end


def _get_timing_range_cpu(
    cur: str,
    line: str,
    max_timing: float | None,
    ds: DecoderState,
    min_timing: float | None = None,
) -> tuple[int, int]:
    """Get (start_idx, end_idx) into timing_token_ids for valid timing range.

    Returns indices suitable for slicing timing_token_ids_cpu[start_idx:end_idx].
    """
    all_timing_matches = list(TIMINGS_FORMAT.finditer(line))
    if all_timing_matches:
        last_t = parse_timing(all_timing_matches[-1].group(0))
        text_after = line[all_timing_matches[-1].end() :]
    else:
        last_t = None
        text_after = ""

    if min_timing is not None:
        if last_t is None or last_t < min_timing:
            last_t = min_timing

    step = ds.timing_step
    max_idx = ds.max_timing_idx + 1

    def _range(start: float, end: float | None) -> tuple[int, int]:
        s = max(0, round(start / step))
        if end is None:
            return (s, max_idx)
        if max_timing is not None:
            end = min(end, max_timing)
        e = min(max_idx, round(end / step) + 1)
        return (s, e)

    if last_t is None:
        all_cur_timings = list(TIMINGS_FORMAT.finditer(cur))
        if all_cur_timings:
            prev_t = parse_timing(all_cur_timings[-1].group(0))
            start = max(0, prev_t - 5.0)
            cap = prev_t + 2.0
            if max_timing is not None:
                cap = min(cap, max_timing)
            return _range(start, cap)
        cap = 1.0
        if max_timing is not None:
            cap = min(cap, max_timing)
        return _range(0, cap)

    min_next = last_t + step

    inpause = "pause]" in line[-6:] if len(line) >= 6 else False
    inlaugh = "laugh]" in line[-6:] if len(line) >= 6 else False

    if inpause or inlaugh:
        cap = last_t + 10.0
        if max_timing is not None:
            cap = min(cap, max_timing)
        return _range(min_next, cap)

    cleaned = _BRACKETS_CLEAN.sub("", text_after)
    words = len([w for w in cleaned.split() if w.strip()])
    leeway = words * 1.0 + 1.0
    cap = last_t + leeway
    if max_timing is not None:
        cap = min(cap, max_timing)
    return _range(min_next, cap)

