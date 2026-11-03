"""
Language-aware vocabulary constraints for ASR inference.

This module provides functions to determine which tokens should be blocked
during inference based on the detected language. This prevents hallucinated
characters from non-compatible scripts (e.g., blocking CJK characters when
transcribing English, Spanish, French, etc.).

Usage:
    from whisperdrz.vocab_constraints import get_blocked_tokens_for_language

    # Get tokens to block for English
    blocked = get_blocked_tokens_for_language("en")
    logits[:, blocked] = -torch.inf
"""

import unicodedata
from functools import lru_cache

# Import will be done lazily to avoid circular dependencies
_VOCAB = None
_tok = None


def _get_vocab():
    """Lazy load vocabulary to avoid circular import."""
    global _VOCAB, _tok
    if _VOCAB is None:
        from whisperdrz.tokenizer import get_tokenizer

        _tok = get_tokenizer()
        _VOCAB = {i: _tok.decode([i]) for i in range(_tok.encoding.n_vocab)}
    return _VOCAB


# Define script compatibility for language groups
# Each language group specifies which scripts are ALLOWED
LANGUAGE_SCRIPT_COMPATIBILITY = {
    # Latin script languages (European + Latin America + more)
    "latin_only": {
        "languages": [
            "en",
            "es",
            "fr",
            "it",
            "pt",
            "de",
            "nl",
            "sv",
            "no",
            "da",
            "fi",
            "pl",
            "cs",
            "sk",
            "ro",
            "hu",
            "hr",
            "sl",
            "sq",
            "et",
            "lv",
            "lt",
            "tr",
            "id",
            "ms",
            "tl",
            "vi",
            "sw",
            "so",
            "af",
            "eu",
            "ca",
            "gl",
            "is",
            "ga",
            "cy",
            "mt",
            "lb",
        ],
        "allowed_scripts": ["LATIN"],
        "allow_ascii_only": False,  # Allow Latin with diacritics
    },
    # Cyrillic script languages (Slavic + Central Asian).
    # sr/bs are bi-scriptal — Cyrillic is primary/official for Serbian; Bosnian
    # is written in both. Listed here so both scripts are allowed at inference.
    "cyrillic": {
        "languages": [
            "ru",
            "uk",
            "bg",
            "sr",
            "bs",
            "mk",
            "be",
            "kk",
            "ky",
            "mn",
            "uz",
            "tg",
            "tt",
            "ba",
        ],
        "allowed_scripts": [
            "LATIN",
            "CYRILLIC",
        ],  # Some Latin letters appear in loanwords
        "allow_ascii_only": False,
    },
    # Greek
    "greek": {
        "languages": ["el"],
        "allowed_scripts": ["LATIN", "GREEK"],
        "allow_ascii_only": False,
    },
    # Arabic script languages
    "arabic": {
        "languages": ["ar", "fa", "ur", "ps", "sd"],
        "allowed_scripts": ["LATIN", "ARABIC"],  # Latin for loanwords
        "allow_ascii_only": False,
    },
    # Hebrew script languages (Hebrew + Yiddish)
    "hebrew": {
        "languages": ["he", "yi"],
        "allowed_scripts": ["LATIN", "HEBREW"],
        "allow_ascii_only": False,
    },
    # CJK (Chinese, Japanese, Korean)
    "cjk": {
        "languages": ["zh", "ja", "ko", "yue"],
        "allowed_scripts": ["LATIN", "CJK", "HIRAGANA", "KATAKANA", "HANGUL"],
        "allow_ascii_only": False,
    },
    # Indian languages (Devanagari, Bengali, Tamil, etc.). Urdu is NOT here —
    # it's Arabic-script and lives in the "arabic" group.
    "indic": {
        "languages": [
            "hi",
            "bn",
            "ta",
            "te",
            "mr",
            "gu",
            "kn",
            "ml",
            "pa",
            "or",
            "as",
            "ne",
            "si",
            "sa",
        ],
        "allowed_scripts": [
            "LATIN",
            "DEVANAGARI",
            "BENGALI",
            "TAMIL",
            "TELUGU",
            "GUJARATI",
            "GURMUKHI",
            "KANNADA",
            "MALAYALAM",
            "ORIYA",
            "SINHALA",
        ],
        "allow_ascii_only": False,
    },
    # Thai
    "thai": {
        "languages": ["th"],
        "allowed_scripts": ["LATIN", "THAI"],
        "allow_ascii_only": False,
    },
    # Armenian
    "armenian": {
        "languages": ["hy"],
        "allowed_scripts": ["LATIN", "ARMENIAN"],
        "allow_ascii_only": False,
    },
    # Georgian
    "georgian": {
        "languages": ["ka"],
        "allowed_scripts": ["LATIN", "GEORGIAN"],
        "allow_ascii_only": False,
    },
    # Ethiopic (Amharic, Tigrinya)
    "ethiopic": {
        "languages": ["am", "ti"],
        "allowed_scripts": ["LATIN", "ETHIOPIC"],
        "allow_ascii_only": False,
    },
}


def _get_language_group(language_code: str) -> dict | None:
    """Get the script compatibility group for a language code.

    Args:
        language_code: ISO 639-1 or 639-3 language code (e.g., "en", "es", "zh")

    Returns:
        Dictionary with allowed_scripts and allow_ascii_only, or None if unknown
    """
    for group in LANGUAGE_SCRIPT_COMPATIBILITY.values():
        if language_code in group["languages"]:
            return {
                "allowed_scripts": group["allowed_scripts"],
                "allow_ascii_only": group["allow_ascii_only"],
            }

    # Default: allow Latin script for unknown languages
    return {
        "allowed_scripts": ["LATIN"],
        "allow_ascii_only": False,
    }


@lru_cache(maxsize=128)
def _get_token_script(token: str) -> str | None:
    """Determine the primary script of a token.

    Args:
        token: Decoded token string

    Returns:
        Script name (e.g., "LATIN", "CYRILLIC", "CJK") or None if ASCII/whitespace
    """
    if not token:
        return None

    for char in token:
        # Skip whitespace and ASCII punctuation
        if char in " \n\t" or (ord(char) < 128 and not char.isalpha()):
            continue

        # ASCII letters are Latin
        if ord(char) < 128 and char.isalpha():
            return "LATIN"

        # Check Unicode script
        try:
            name = unicodedata.name(char, "")
            if not name:
                continue

            # Extract script from character name
            # E.g., "LATIN SMALL LETTER A" -> "LATIN"
            #       "CYRILLIC SMALL LETTER A" -> "CYRILLIC"
            #       "CJK UNIFIED IDEOGRAPH-4E00" -> "CJK"

            script_name = name.split()[0]

            # Special cases
            if "HIRAGANA" in name:
                return "HIRAGANA"
            if "KATAKANA" in name:
                return "KATAKANA"
            if "HANGUL" in name:
                return "HANGUL"
            if "IDEOGRAPHIC" in name:
                # IDEOGRAPHIC = CJK characters (e.g., IDEOGRAPHIC FULL STOP = 。)
                return "CJK"
            if script_name in [
                "LATIN",
                "CYRILLIC",
                "ARABIC",
                "HEBREW",
                "GREEK",
                "DEVANAGARI",
                "BENGALI",
                "TAMIL",
                "TELUGU",
                "GUJARATI",
                "GURMUKHI",
                "KANNADA",
                "MALAYALAM",
                "ORIYA",
                "SINHALA",
                "THAI",
                "LAO",
                "MYANMAR",
                "ARMENIAN",
                "GEORGIAN",
                "ETHIOPIC",
                "CJK",
            ]:
                return script_name
        except:
            pass

    return None


@lru_cache(maxsize=128)
def get_blocked_tokens_for_language(language_code: str) -> tuple[int, ...]:
    """Get list of token IDs that should be blocked for a given language.

    This function determines which tokens contain characters from scripts that are
    incompatible with the specified language, and returns their IDs for blocking
    during inference.

    Args:
        language_code: ISO 639-1 or 639-3 language code (e.g., "en", "es", "zh")
                      Use "auto" or None to allow all scripts

    Returns:
        Tuple of token IDs to block during inference

    Examples:
        >>> # Block non-Latin tokens for English
        >>> blocked = get_blocked_tokens_for_language("en")
        >>> logits[:, blocked] = -torch.inf

        >>> # Block non-Cyrillic tokens for Russian
        >>> blocked = get_blocked_tokens_for_language("ru")
        >>> logits[:, blocked] = -torch.inf
    """
    # Special case: auto-detect or None means don't block anything
    if language_code in [None, "auto", ""]:
        return ()

    # Get allowed scripts for this language
    group = _get_language_group(language_code.lower())
    if group is None:
        return ()

    allowed_scripts = set(group["allowed_scripts"])

    # Get vocabulary
    vocab = _get_vocab()

    # Collect tokens to block
    blocked_tokens = []

    for token_id, token_text in vocab.items():
        # Skip special tokens (start with <|)
        if token_text.startswith("<|"):
            continue

        # Get the script of this token
        script = _get_token_script(token_text)

        # If we couldn't determine script (ASCII punctuation/whitespace), allow it
        if script is None:
            continue

        # Block if script is not in allowed list
        if script not in allowed_scripts:
            blocked_tokens.append(token_id)

    return tuple(blocked_tokens)


# UTF-8 byte ranges of each script's codepoint block, as (first_char, last_char)
# encoded bytes. Used to unblock byte-fragment tokens per language group (see
# get_fragment_tokens_to_unblock). Only scripts whose characters commonly lack
# complete vocab tokens need an entry. A trailing partial character is allowed
# iff it is a byte-prefix of some character in an allowed script's block —
# UTF-8 byte order tracks codepoint order, so comparing the partial against the
# truncated endpoints suffices. (A lead-byte-only check would be too coarse for
# 3-byte scripts: 0xE0 starts Devanagari but also Thai, Lao, and Tibetan.)
_SCRIPT_BYTE_RANGES = {
    "CYRILLIC": ("\u0400".encode(), "\u052f".encode()),  # Cyrillic + supplement
    "HEBREW": ("\u0590".encode(), "\u05ff".encode()),
    "ARABIC": ("\u0600".encode(), "\u06ff".encode()),
    "DEVANAGARI": ("\u0900".encode(), "\u097f".encode()),
}


def _partial_in_ranges(seq: bytes, ranges: list[tuple[bytes, bytes]]) -> bool:
    """True if `seq` (an incomplete UTF-8 character) is a byte-prefix of some
    character inside one of the (lo, hi) codepoint-block byte ranges."""
    return any(lo[: len(seq)] <= seq <= hi[: len(seq)] for lo, hi in ranges)


def _fragment_bytes_allowed(
    core: bytes, ranges: list[tuple[bytes, bytes]], allowed_scripts: set[str]
) -> bool:
    """Walk a token's raw bytes (known NOT to be fully-valid UTF-8) and decide
    whether everything it can contribute is compatible with the allowed
    scripts: complete chars must be allowed-script/ASCII, and a trailing
    partial character must be a byte-prefix of an allowed script's block.
    Leading continuation bytes are fine — they finish a character started by
    the PREVIOUS token, which was itself script-checked."""
    i = 0
    n = len(core)
    while i < n and 0x80 <= core[i] <= 0xBF:
        i += 1
    while i < n:
        b = core[i]
        if b < 0x80:
            i += 1
            continue
        if 0xC0 <= b <= 0xDF:
            need = 1
        elif 0xE0 <= b <= 0xEF:
            need = 2
        elif 0xF0 <= b <= 0xF7:
            need = 3
        else:
            return False
        seq = core[i : i + need + 1]
        if len(seq) < need + 1:
            # trailing partial character: must be a prefix of an allowed block
            return _partial_in_ranges(seq, ranges)
        try:
            ch = seq.decode("utf-8")
        except UnicodeDecodeError:
            return False
        script = _get_token_script(ch)
        if script is not None and script not in allowed_scripts:
            return False
        i += need + 1
    return True


@lru_cache(maxsize=32)
def get_fragment_tokens_to_unblock(language_code: str) -> tuple[int, ...]:
    """Byte-fragment tokens (partial UTF-8; they decode to '�' standalone)
    that a language legitimately needs to emit characters lacking a complete
    vocab token — e.g. word-initial ё/Ё in Russian.

    get_always_blocked_tokens blocks every '�'-containing token, which bans
    those characters outright. This returns the SAFE subset to re-allow for a
    language: fragments whose bytes can only produce allowed-script
    characters. Pure continuation-byte tokens are included — they cannot
    START a character, and every character's start bytes are still gated, so
    an out-of-script character can never be assembled. Latin-script languages
    return () (all their characters have complete tokens), so their masks are
    unchanged.
    """
    if not language_code:
        return ()
    group = _get_language_group(language_code.lower())
    if group is None:
        return ()
    allowed_scripts = set(group["allowed_scripts"])
    ranges = [
        _SCRIPT_BYTE_RANGES[s] for s in allowed_scripts if s in _SCRIPT_BYTE_RANGES
    ]
    if not ranges:
        return ()

    _get_vocab()  # ensure _tok is initialized
    enc = _tok.encoding
    out = []
    for token_id in range(enc.n_vocab):
        try:
            bs = enc.decode_single_token_bytes(token_id)
        except Exception:
            continue
        core = bs.lstrip(b" ")
        if not core:
            continue
        try:
            core.decode("utf-8")
            continue  # fully valid UTF-8 — not a fragment
        except UnicodeDecodeError:
            pass
        if _fragment_bytes_allowed(core, ranges, allowed_scripts):
            out.append(token_id)
    return tuple(out)


@lru_cache(maxsize=32)
def get_always_blocked_tokens() -> tuple[int, ...]:
    """Get tokens that should always be blocked regardless of language.

    This includes:
    - Replacement character (�)
    - Certain symbols that shouldn't appear in transcriptions
    - Malformed tokens

    Returns:
        Tuple of token IDs to always block
    """
    from whisperdrz.tokenizer import get_tokenizer

    tok = get_tokenizer()
    vocab = _get_vocab()

    blocked = []

    # Always block specific problematic characters
    always_block_chars = [
        "�",  # Replacement character
        "/",  # Forward slash
        "{",
        "}",  # Curly braces
        "+",
        "=",  # Math symbols
        "^",
        "~",
        "_",  # Other symbols
        "#",  # Hash
        "»",
        "«",  # Special quotes
        ";",  # Semicolon
        ":",  # Colon (prevents spurious colons in transcriptions)
        "《",
        "》",  # CJK brackets
        "「",
        "」",
        "『",
        "』",  # Japanese brackets
        "؟",  # Arabic question mark
    ]

    for char in always_block_chars:
        # Find all tokens containing this character
        for token_id, token_text in vocab.items():
            if char in token_text:
                blocked.append(token_id)

    # Block standalone < > | to prevent timing token corruption
    try:
        blocked.append(tok.encoding.encode_single_token("<"))
        blocked.append(tok.encoding.encode_single_token(">"))
        blocked.append(tok.encoding.encode_single_token("|"))
    except:
        pass

    return tuple(set(blocked))  # Remove duplicates


def print_blocked_tokens_sample(language_code: str, max_samples: int = 20):
    """Debug function to print sample of blocked tokens for a language.

    Args:
        language_code: Language to check (e.g., "en", "ru", "zh")
        max_samples: Maximum number of samples to print
    """
    vocab = _get_vocab()
    blocked = get_blocked_tokens_for_language(language_code)

    print(f"=== Blocked tokens for language '{language_code}' ===")
    print(f"Total blocked: {len(blocked)}")
    print(f"\nFirst {max_samples} blocked tokens:")

    for i, token_id in enumerate(blocked[:max_samples]):
        token_text = vocab[token_id]
        script = _get_token_script(token_text)
        print(f"  {token_id:5d}: {token_text!r:20s} (script: {script})")


if __name__ == "__main__":
    # Test the module
    print("Testing vocab_constraints module\n")

    # Test English (should block non-Latin)
    print_blocked_tokens_sample("en", 10)
    print()

    # Test Russian (should block non-Cyrillic/Latin)
    print_blocked_tokens_sample("ru", 10)
    print()

    # Test Chinese (should block non-CJK/Latin)
    print_blocked_tokens_sample("zh", 10)
    print()

    # Always blocked
    always_blocked = get_always_blocked_tokens()
    print(f"Always blocked tokens: {len(always_blocked)}")
