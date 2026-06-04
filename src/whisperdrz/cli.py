import argparse
import json
import sys

from whisperdrz import load_model
from whisperdrz.audio import SAMPLE_RATE, load_audio


def _load_mono_16k(path: str):
    audio, sr = load_audio(path, sample_rate=SAMPLE_RATE)
    if audio.dim() > 1:
        audio = audio.mean(dim=0)
    return audio


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="whisperdrz", description="Transcribe audio with WhisperDRZ."
    )
    parser.add_argument("audio", nargs="+", help="Audio file(s) to transcribe")
    parser.add_argument(
        "--model",
        default="whisperdrz-large-v3.safetensors",
        help="Checkpoint path, HF repo filename, or HF repo id",
    )
    parser.add_argument(
        "--lang", default="en", help="Language code (e.g. en). Use 'auto' to detect"
    )
    parser.add_argument(
        "--output_format",
        choices=["txt", "json"],
        default="txt",
        help="Output format",
    )
    args = parser.parse_args(argv)

    lang = None if args.lang == "auto" else args.lang
    transcriber = load_model(args.model, lang=lang)

    for path in args.audio:
        audio = _load_mono_16k(path)
        result = transcriber.transcribe(audio)
        if args.output_format == "json":
            json.dump(
                {"text": result.text, "segments": result.segments},
                sys.stdout,
                indent=2,
                ensure_ascii=False,
            )
            sys.stdout.write("\n")
        else:
            print(result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
