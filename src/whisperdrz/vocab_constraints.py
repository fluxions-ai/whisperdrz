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
        "languages": ["ar", "fa", "ur", "ps"],
        "allowed_scripts": ["LATIN", "ARABIC"],  # Latin for loanwords
        "allow_ascii_only": False,
    },
    # Hebrew
    "hebrew": {
        "languages": ["he"],
        "allowed_scripts": ["LATIN", "HEBREW"],
        "allow_ascii_only": False,
    },
    # CJK (Chinese, Japanese, Korean)
    "cjk": {
        "languages": ["zh", "ja", "ko", "yue"],
        "allowed_scripts": ["LATIN", "CJK", "HIRAGANA", "KATAKANA", "HANGUL"],
        "allow_ascii_only": False,
    },
    # Indian languages (Devanagari, Bengali, Tamil, etc.)
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
            "ur",
            "ne",
            "si",
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
