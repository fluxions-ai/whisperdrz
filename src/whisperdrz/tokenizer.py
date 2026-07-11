import base64
import os
from dataclasses import dataclass, field
from functools import cache, cached_property

import tiktoken
from torch import Tensor

LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
}

# language code lookup by name, with a few language aliases
TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
    "mandarin": "zh",
}


@dataclass
class Tokenizer:
    """A thin wrapper around `tiktoken` providing quick access to special tokens"""

    encoding: tiktoken.Encoding
    num_languages: int
    language: str | None = None
    task: str | None = None
    sot_sequence: tuple[int] = ()
    special_tokens: dict[str, int] = field(default_factory=dict)
    lang_token: int = None

    def __post_init__(self):
        for special in self.encoding.special_tokens_set:
            special_token = self.encoding.encode_single_token(special)
            self.special_tokens[special] = special_token

        sot: int = self.special_tokens["<|startoftranscript|>"]
        translate: int = self.special_tokens["<|translate|>"]
        transcribe: int = self.special_tokens["<|transcribe|>"]

        langs = tuple(LANGUAGES.keys())[: self.num_languages]

        sot_sequence = [sot]

        if self.language is not None:
            self.lang_token = sot + 1 + langs.index(self.language)
            sot_sequence.append(self.lang_token)
        if self.task is not None:
            task_token: int = transcribe if self.task == "transcribe" else translate
            sot_sequence.append(task_token)

        self.sot_sequence = tuple(sot_sequence)

    def encode(self, text, **kwargs):
        return self.encoding.encode(text, allowed_special="all", **kwargs)

    def decode(self, token_ids: list[int] | Tensor, **kwargs) -> str:
        # NOTE(james) filter out timestamp tokens when deciding how to present to the end user etc
        # token_ids = [t for t in token_ids if t < self.timestamp_begin]
        if not isinstance(token_ids, list):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return self.encoding.decode(token_ids, **kwargs)

    @cached_property
    def eot(self) -> int:
        return self.encoding.eot_token

    @cached_property
    def transcribe(self) -> int:
        return self.special_tokens["<|transcribe|>"]

    @cached_property
    def translate(self) -> int:
        return self.special_tokens["<|translate|>"]

    @cached_property
    def sot(self) -> int:
        return self.special_tokens["<|startoftranscript|>"]


    @cached_property
    def no_timestamps(self) -> int:
        return self.special_tokens["<|notimestamps|>"]


    @cached_property
    def speech(self) -> int:
        return self.special_tokens["<|speech|>"]

    @cached_property
    def emotion(self) -> int:
        return self.special_tokens["<|emotion|>"]

    @cached_property
    def nonspeech(self) -> int:
        return self.special_tokens["<|nonspeech|>"]

    @cached_property
    def accent(self) -> int:
        return self.special_tokens["<|accent|>"]

    @cached_property
    def timestamp_begin(self) -> int:
        return self.special_tokens["<|0.00|>"]


    def to_language_token(self, language):
        if token := self.special_tokens.get(f"<|{language}|>", None):
            return token

        raise KeyError(f"Language {language} not found in tokenizer.")

    @cached_property
    def all_language_tokens(self) -> tuple[int]:
        result = []
        for token, token_id in self.special_tokens.items():
            if token.strip("<|>") in LANGUAGES:
                result.append(token_id)
        return tuple(result)[: self.num_languages]


    def expand_timestamps(
        self, text: str, step_range: tuple[float, float] = (0.2, 0.6)
    ) -> str:
        """Replace <t> placeholders with sequential random timestamps."""
        import random

        parts = text.split("<t>")
        if len(parts) == 1:
            return text

        result = [parts[0]]
        t = 0.0
        for part in parts[1:]:
            t_rounded = round(t / 0.02) * 0.02
            result.append(f"<|{t_rounded:.2f}|>")
            result.append(part)
            t += random.uniform(*step_range)

        return "".join(result)

    def print_token_map(self, text: str):
        """Print token ID to token mapping for the given text."""
        text = self.expand_timestamps(text)
        tokens = self.encode(text)
        print(f"Text: {text!r}")
        print(f"Tokens: {len(tokens)}")
        for token_id in tokens:
            decoded = self.decode([token_id])
            print(f"  {token_id:6d} -> {decoded!r}")


@cache
def get_encoding(
    name: str = "gpt2",
    num_languages: int = 99,
    extra_tokens: tuple[str, ...] = (),
):
    vocab_path = os.path.join(os.path.dirname(__file__), "assets", f"{name}.tiktoken")
    ranks = {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in open(vocab_path) if line)
    }
    n_vocab = len(ranks)
    special_tokens = {}

    specials = [
        "<|endoftext|>",
        "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in list(LANGUAGES.keys())[:num_languages]],
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
        *extra_tokens,
    ]

    for token in specials:
        special_tokens[token] = n_vocab
        n_vocab += 1

    return tiktoken.Encoding(
        name=os.path.basename(vocab_path),
        explicit_n_vocab=n_vocab,
        pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        mergeable_ranks=ranks,
        special_tokens=special_tokens,
    )


@cache
def get_tokenizer(
    multilingual: bool = True,
    num_languages: int = 100,
    language: str | None = "en",
    task: str | None = "transcribe",
    extra_tokens: tuple[str, ...] = (),
) -> Tokenizer:
    if language is not None:
        language = language.lower()
        if language not in LANGUAGES:
            if language in TO_LANGUAGE_CODE:
                language = TO_LANGUAGE_CODE[language]
            else:
                raise ValueError(f"Unsupported language: {language}")

    if multilingual:
        encoding_name = "multilingual"
        language = language or "en"
        task = task or "transcribe"
    else:
        encoding_name = "gpt2"
        language = None
        task = None

    encoding = get_encoding(
        name=encoding_name,
        num_languages=num_languages,
        extra_tokens=extra_tokens,
    )

    return Tokenizer(
        encoding=encoding, num_languages=num_languages, language=language, task=task
    )


if __name__ == "__main__":
    import sys

    tok = get_tokenizer()

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        tok.print_token_map(text)
    else:
        print("Interactive tokenizer (Ctrl+C to exit)")
        while True:
            try:
                text = input("> ")
                if text.strip():
                    tok.print_token_map(text)
            except (KeyboardInterrupt, EOFError):
                break
