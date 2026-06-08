# WhisperDRZ - Adding Diarization to Whisper

WhisperDRZ is a speaker-aware automatic speech recognition model. It transcribes audio
into text with word-level timestamps, per-line speaker tags, and non-speech event
tags. It is a Whisper-style encoder-decoder model and handles long audio by
chunking and stitching internally.

This repository is inference only. Weights are distributed on the Hugging Face
Hub at [`fluxions/whisperdrz`](https://huggingface.co/fluxions/whisperdrz).

📖 **Full write-up** — how it works, the timing-token and diarization approach,
honest results (WER, DER, timing, events, multilingual), and what didn't work:
**[blog.md](blog.md)**.

## Install

```bash
pip install whisperdrz
# optional extras:
pip install "whisperdrz[flash]"   # Flash Attention — much faster decoding on CUDA
pip install "whisperdrz[demo]"    # the gradio demo
pip install "whisperdrz[eval]"    # WER/WDER metrics (jiwer, whisper-normalizer)
```

Or from a clone of this repository (editable install):

```bash
pip install -e .            # add ".[flash]", ".[demo]", or ".[eval]" for extras
```

Requires Python 3.12+. Runs on a CUDA GPU or on CPU. Flash Attention is
optional: when it (and a CUDA GPU) is available it is used with CUDA graphs for
fast decoding; otherwise a pure-torch attention path is used automatically.

## Command line

```bash
whisperdrz audio.wav                                       # defaults: --model whisperdrz-large-v3.safetensors, --lang en
whisperdrz audio.wav --output_format json > out.json
whisperdrz audio.wav --model my-checkpoint.pt --lang auto  # override the defaults
```

`--model` defaults to `whisperdrz-large-v3.safetensors` and accepts a local checkpoint,
a filename hosted in the weights repo, or a Hugging Face repo id. The weights
download automatically on first use. `--lang` defaults to `en`; use `auto` to detect.

## Python

```python
import whisperdrz
from whisperdrz.audio import load_audio, SAMPLE_RATE

transcriber = whisperdrz.load_model("whisperdrz-large-v3.safetensors", lang="en")

audio, _ = load_audio("audio.wav", sample_rate=SAMPLE_RATE)
result = transcriber.transcribe(audio.mean(0))  # mono, 16 kHz

print(result.text)        # speaker-tagged text with timestamps
print(result.segments)    # list of {speaker, start, end, text}
```

## Output format

Each line begins with a speaker tag. Timed words and tags are wrapped in a
start/end timestamp pair; not every word is timed, but the first and last word
of each line always are:

```
[0] <|0.00|>Hello<|0.45|> there <|0.80|>world.<|1.10|>
[1] <|1.20|>Hi<|1.40|> <|1.45|>[laugh]<|1.60|> <|1.70|>there.<|1.95|>
```

- `[0]`, `[1]`, ... are speaker IDs; `[c]` marks crowd/ambient.
- `<|t|>` are timestamps in seconds (two decimals), always in a pair wrapping a word or tag.
- The first and last word of every line are always timed; middle words may be bare.
- `[laugh]`, `[breath]`, and similar are non-speech event tags.

`transcribe()` returns a `TranscribeResults` with the raw `text` plus parsed
`segments`, each a dict of `speaker`, `start`, `end`, and `text`.

## Evaluation

WhisperDRZ reports two metrics:

- **WER** (word error rate) — Levenshtein word distance after normalization
  (whisper-normalizer), measuring transcription accuracy.
- **WDER** (word diarization error rate) — of the words that align between
  reference and hypothesis, the fraction assigned to the wrong speaker under
  the best speaker permutation. It isolates diarization quality from WER
  (insertions/deletions don't count toward WDER).

Measured on the released checkpoint:

| Benchmark | Metric | Score |
|---|---|---|
| ESB (English ASR, 1000 utts) | WER | 9.6% macro / 5.9% micro |
| VoxConverse dev (216 files, overlap-heavy) | DER | 40.7% (63.4% on overlapping speech) |
| Internal conversational (26 clips) | WER / WDER | 11.1% / 33% |

WhisperDRZ is an ASR-first model: transcription is strong, but diarization trails
purpose-built systems (which reach ~10–25% DER on VoxConverse) and overlap is its
weak spot. See [the write-up](blog.md) for full results (timing, non-speech
events, multilingual) and analysis.

Reproduce WER/WDER on your own data with the metrics in `whisperdrz.evals` and the runner:

```bash
pip install -e ".[eval]"
# manifest.jsonl: one {"audio": "a.wav", "text": "[0] ref ... [1] ..."} per line
python scripts/eval.py manifest.jsonl --model whisperdrz-large-v3.safetensors
```

`text` is the reference transcript; `[N]` speaker tags are optional and WDER is
only computed for multi-speaker references.

## Demo

```bash
python demo/app.py --model whisperdrz-large-v3.safetensors
```

## License

MIT. See [LICENSE](LICENSE).
