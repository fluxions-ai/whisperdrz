"""Run WhisperDRZ over a manifest and report WER and WDER.

Manifest is JSON Lines, one object per utterance:

    {"audio": "path/to/a.wav", "text": "[0] hello there [1] hi"}
    {"audio": "path/to/b.wav", "text": "the quick brown fox"}

`text` is the reference transcript. Speaker tags `[N]` are optional; WDER is
only computed over utterances whose reference has more than one speaker.

Usage:
    pip install -e ".[eval]"
    python scripts/eval.py manifest.jsonl --model whisperdrz-large-v3.safetensors
"""

import argparse
import json

from whisperdrz import load_model
from whisperdrz.audio import SAMPLE_RATE, load_audio
from whisperdrz.evals import calculate_wder, calculate_wer, has_speakers


def _read_manifest(path: str) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate WhisperDRZ WER/WDER.")
    parser.add_argument("manifest", help="JSONL manifest of {audio, text}")
    parser.add_argument("--model", default="whisperdrz-large-v3.safetensors")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--limit", type=int, default=None, help="Eval first N items")
    parser.add_argument("--save", default=None, help="Write per-utterance JSON here")
    args = parser.parse_args(argv)

    items = _read_manifest(args.manifest)
    if args.limit:
        items = items[: args.limit]

    transcriber = load_model(args.model, lang=args.lang)

    total_ref_words = total_wer_errors = 0
    total_aligned = total_wder_wrong = 0
    wer_list: list[float] = []
    wder_list: list[float] = []
    results = []

    for i, item in enumerate(items):
        audio, _ = load_audio(item["audio"], sample_rate=SAMPLE_RATE)
        audio = audio.mean(0) if audio.dim() > 1 else audio
        hyp = transcriber.transcribe(audio).text
        ref = item["text"]

        w = calculate_wer(ref, hyp)
        total_ref_words += w["reference_words"]
        total_wer_errors += w["insertions"] + w["deletions"] + w["substitutions"]
        wer_list.append(w["wer"])

        row = {"audio": item["audio"], "wer": w["wer"], "hypothesis": hyp}
        if has_speakers(ref):
            d = calculate_wder(ref, hyp)
            total_aligned += d["aligned"]
            total_wder_wrong += d["aligned"] - d["correct"]
            wder_list.append(d["wder"])
            row["wder"] = d["wder"]
        results.append(row)

        if (i + 1) % 50 == 0:
            micro = total_wer_errors / max(total_ref_words, 1) * 100
            print(f"  {i + 1}/{len(items)}  WER={micro:.2f}%")

    micro_wer = total_wer_errors / max(total_ref_words, 1)
    macro_wer = sum(wer_list) / max(len(wer_list), 1)
    print("\n==== Results ====")
    print(f"utterances:     {len(items)}")
    print(f"WER  (micro):   {micro_wer * 100:.2f}%")
    print(f"WER  (macro):   {macro_wer * 100:.2f}%")
    if wder_list:
        micro_wder = total_wder_wrong / max(total_aligned, 1)
        macro_wder = sum(wder_list) / len(wder_list)
        print(f"WDER (micro):   {micro_wder * 100:.2f}%  ({len(wder_list)} multi-speaker)")
        print(f"WDER (macro):   {macro_wder * 100:.2f}%")
    else:
        print("WDER:           n/a (no multi-speaker references)")

    if args.save:
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nPer-utterance results written to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
