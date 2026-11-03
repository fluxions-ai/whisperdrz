# WhisperDRZ - Adding Diarization to Whisper

WhisperDRZ is a speaker-aware automatic speech recognition model. It transcribes audio
into text with word-level timestamps, per-line speaker tags, and non-speech event
tags. It is a Whisper-style encoder-decoder model and handles long audio by
chunking and stitching internally.

This repository is inference only. Weights are distributed on the Hugging Face
Hub at [`fluxions/whisperdrz`](https://huggingface.co/fluxions/whisperdrz).

🎙️ **Try it in the browser** — no install needed:
**[fluxions.ai/transcribe](https://fluxions.ai/transcribe)**.

📖 **Full write-up** — how it works, the timing-token and diarization approach,
honest results (WER, DER, timing, events, multilingual), and what didn't work:
**[writeup.md](writeup.md)**.

## Install

```bash
pip install -e .            # add ".[demo]" for the gradio demo, ".[eval]" for the metrics
```

Requires Python 3.12+ and FFmpeg (for audio decoding: `apt install ffmpeg` /
`brew install ffmpeg`). Runs on a CUDA GPU or on CPU.

Flash Attention is optional: when `flash-attn` is installed (and a CUDA GPU is
available) it is used with CUDA graphs for fast decoding; otherwise a pure-torch
attention path is used automatically. flash-attn needs torch present at build
time, so install it separately: `pip install flash-attn --no-build-isolation`.

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

Measured on the released checkpoint with this package's defaults (natural
speaker clustering, no speaker count given), 2026-11:

| Benchmark | WER | DER (miss / FA / conf) | cpWER / tcpWER |
|---|---|---|---|
| ESB (English ASR, 1000 utts) | 9.6% macro / 5.9% micro | — | — |
| Internal conversational (23 clips) | 10.3% | — (WDER 26%) | 20% cpWER |
| VoxConverse dev (216, overlap-heavy) | — | **11.4%** (3.1 / 4.6 / 3.7) | — |
| CALLHOME eng (140, 2-spk telephone) | — | **21.5%** (6.1 / 8.9 / 6.5) | — |
| AMI test (16 meetings, Mix-Headset) | 22.8% | **34.3%** (13.1 / 17.9 / 3.3) | 31.6% / 37.1% |

DER is at the standard 0.25s collar, scoring overlapping speech (AMI inside its
UEM). cpWER/tcpWER (meeteval, 5s collar for tcpWER) and WDER are micro-averaged;
the AMI WER is speaker-agnostic (all words in time order). Long recordings (60+
speaker lines) are clustered tighter and small clusters folded into their nearest
neighbour, so a 4-person AMI meeting comes out with 5 speakers (median, max 8),
and the permutation-based cp-metrics are computable without telling the model the
speaker count.

**WhisperDRZ is an ASR-first model.** Transcription is strong across the board
(AMI Mix-Headset WER 22.8%). Diarization is in the range published for
purpose-built systems on VoxConverse and CALLHOME (~10–25% DER, though those are
mostly scored without a collar, so not like-for-like), and behind them on long
multi-party meetings: on AMI, attribution adds ~9 points on top of the
words (WER 23% vs cpWER 32%), and overlapped speech is mostly missed (56% DER
inside overlap). See [the write-up](writeup.md) for full analysis (timing,
non-speech events, multilingual).

**A note on false alarm — WhisperDRZ has no VAD.** It transcribes over silence
and music, so false alarm is the largest part of the DER above (17.9% of the
34.3% on AMI). Gating the hypothesis to detected speech *before* scoring — a
component the shipped model doesn't include — recovers much of it. The DER above
is the honest out-of-the-box number; the columns below show what a non-speech
filter would buy:

| Benchmark | DER (ungated) | + silero VAD | + oracle VAD |
|---|---|---|---|
| VoxConverse dev | 11.4% | 10.3% | 7.1% |
| CALLHOME eng | 21.5% | 19.0% | 12.9% |
| AMI test | 34.3% | 27.9% | 16.7% |

silero VAD is a real off-the-shelf detector; oracle VAD (reference speech regions)
is the ceiling. What a VAD *cannot* fix is speaker confusion and over-hypothesized
overlap — the genuine diarization error that remains after detection is perfect.

> Note: these numbers replace earlier ones (VoxConverse 26.3%, CALLHOME 38.6%,
> AMI 50.6% DER, AMI cpWER 72% with oracle speaker count). Same weights; the
> difference is inference fixes made in 2026-11: a CUDA-graph replay race that
> corrupted decoding on files over ~60 s (and made output change run to run),
> speaker clustering by file length, and dropping lines the decoder wrote twice
> over the same time. The old code, scored with the same pipeline, gives
> VoxConverse 26.0%, CALLHOME 42.1%, AMI 49.8% DER. An earlier VoxConverse figure
> of 40.7% came from a different scoring setup.

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

## Citation

```
@software{whisperdrz_2026,
  author = {Coultas Blum, Harry},
  month = {07},
  title = {{WhisperDRZ}},
  url = {https://github.com/fluxions-ai/whisperdrz},
  version = {0.1.0},
  year = {2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
