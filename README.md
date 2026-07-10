# WhisperDRZ - Adding Diarization to Whisper

WhisperDRZ is a speaker-aware automatic speech recognition model. It transcribes audio
into text with word-level timestamps, per-line speaker tags, and non-speech event
tags. It is a Whisper-style encoder-decoder model and handles long audio by
chunking and stitching internally.

This repository is inference only. Weights are distributed on the Hugging Face
Hub at [`fluxions/whisperdrz`](https://huggingface.co/fluxions/whisperdrz).

📖 **Full write-up** — how it works, the timing-token and diarization approach,
honest results (WER, DER, timing, events, multilingual), and what didn't work:
**[writeup.md](writeup.md)**.

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

WhisperDRZ reports transcription, diarization, and joint metrics:

- **WER** (word error rate) — Levenshtein word distance after normalization
  (whisper-normalizer), measuring transcription accuracy, speaker-agnostic.
- **WDER** (word diarization error rate) — of the words that align between
  reference and hypothesis, the fraction assigned to the wrong speaker under
  the best speaker permutation. It isolates diarization quality from WER
  (insertions/deletions don't count toward WDER).
- **cpWER** (concatenated minimum-permutation WER) — concatenate each speaker's
  words into one stream, pick the speaker permutation that minimises total word
  errors, then report a single WER over the concatenated streams. Unlike WDER it
  *does* charge insertions/deletions, so it jointly scores transcription and
  diarization in one number (computed with meeteval).
- **tcpWER** (time-constrained cpWER) — cpWER with the extra constraint that a
  hypothesis word only matches a reference word if their timestamps fall within
  a collar, so it also rewards accurate word timing (meeteval).
- **DER** (diarization error rate) — time-based: missed speech + false alarm +
  speaker confusion, over total reference speech, at a 0.25s collar. The standard
  diarization metric; speaker-count- and word-agnostic.

Measured on the released checkpoint:

| Benchmark | WER | DER (miss / FA / conf) | cpWER / tcpWER |
|---|---|---|---|
| ESB (English ASR, 1000 utts) | 9.6% macro / 5.9% micro | — | — |
| Internal conversational (26 clips) | 11.1% | — (WDER 33%) | 46% cpWER |
| VoxConverse dev (216, overlap-heavy) | — | **26.3%** (3.1 / 15.8 / 7.5) | — |
| CALLHOME eng (140, 2-spk telephone) | — | **38.6%** (5.9 / 14.0 / 18.7) | — |
| AMI test (16 meetings, Mix-Headset) | 22.8% | **50.6%** (10.6 / 28.9 / 11.1) | 72% / 84% |

DER is at the standard 0.25s collar, scoring overlapping speech. AMI cpWER/tcpWER
and the 22.8% WER use **oracle speaker count** (the model is told how many
speakers each meeting has); its natural clustering over-fragments long meetings
(up to ~35 speakers where there are 4), which makes the permutation-based
cp-metrics intractable. Interestingly, oracle-count *raises* AMI DER to 64.9%
(confusion 26.6%): the natural over-fragmentation was quietly flattering DER,
whose optimal mapping absorbs excess clusters as false alarm while keeping
confusion low.

**WhisperDRZ is an ASR-first model.** Transcription is strong across the board
(AMI Mix-Headset WER 22.8%), but diarization trails purpose-built systems
(~10–25% DER) — confusion dominates, and the ~49-point gap between AMI WER (23%)
and cpWER (72%) is almost entirely speaker attribution: the words are right, but
*who said what* is often wrong, especially on long, overlapping multi-party
audio. See [the write-up](writeup.md) for full analysis (timing, non-speech
events, multilingual).

**A note on false alarm — WhisperDRZ has no VAD.** It transcribes over silence
and music, so false alarm is a large part of the DER above (28.9% of the 50.6% on
AMI). Gating the hypothesis to detected speech *before* scoring — a component the
shipped model doesn't include — recovers much of it. The DER above is the honest
out-of-the-box number; the columns below show what a non-speech filter would buy:

| Benchmark | DER (ungated) | + silero VAD | + oracle VAD |
|---|---|---|---|
| VoxConverse dev | 26.3% | 24.0% | 20.9% |
| CALLHOME eng | 38.6% | 34.2% | 28.6% |
| AMI test | 50.6% | 40.6% | 31.1% |

silero VAD is a real off-the-shelf detector; oracle VAD (reference speech regions)
is the ceiling. What a VAD *cannot* fix is speaker confusion and over-hypothesized
overlap — the genuine diarization error that remains after detection is perfect.

> Note: an earlier revision reported VoxConverse dev DER as 40.7%. The 26.3%
> above is a re-measurement on the released checkpoint with the current scoring
> pipeline (pyannote.metrics, 0.25s collar, overlap scored); the discrepancy is
> still being reconciled.

Reproduce WER/WDER/cpWER on your own data with the metrics in `whisperdrz.evals`
and the runner:

```bash
pip install -e ".[eval]"
# manifest.jsonl: one {"audio": "a.wav", "text": "[0] ref ... [1] ..."} per line
python scripts/eval.py manifest.jsonl --model whisperdrz-large-v3.safetensors
```

`text` is the reference transcript; `[N]` speaker tags are optional and WDER /
cpWER are only computed for multi-speaker references. DER and tcpWER require
time-aligned references (RTTM / word timings) and the meeteval + pyannote.metrics
tooling used for the benchmarks above.

## Demo

```bash
python demo/app.py --model whisperdrz-large-v3.safetensors
```

## License

MIT. See [LICENSE](LICENSE).
