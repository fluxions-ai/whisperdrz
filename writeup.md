# WhisperDRZ: teaching Whisper to hear who said what

Most ASR systems answer one question: *what words were said?* But that's not what listening is. When you hear a conversation you also know *who* spoke, when they overlapped, when someone took a breath, hesitated, or laughed, or stammered on some words. Those signals are most of what makes speech feel human, and they're exactly what transcription throws away.

WhisperDRZ is a fine-tune of [Whisper-large-v3](https://github.com/openai/whisper) that tries to keep them. It transcribes, diarizes (labels who said what), predicts word-level timings that tolerate overlapping speech, and emits non-speech events like `[breath]` and `[hesitate]`, s-s-stutters, people cutting themselves off "hey would you-" etc. 

The longer-term goal was never transcription for its own sake — it was to build the listening half of a speech system with enough understanding of *what is actually happening* in audio to eventually generate it with the same realism.

This is a write-up of what worked, what didn't, and what I'd do with more budget. The bulk of this work was done in 2025; the released checkpoint was finalized in early 2026.

## Results at a glance

Every number below is measured on the **publicly released checkpoint**, so it's what you'd actually get. The honest one-line summary: transcription is strong, timing is decent, diarization is usable but well behind specialist systems, and events are showing some signal, but need more data to improve further.

| Capability | Metric (dataset) | WhisperDRZ | Reference |
|---|---|---|---|
| Transcription, clean English | WER ([ESB](https://huggingface.co/datasets/esb/datasets), 1000) | **9.6%** macro / 5.9% micro | whisper-large-v3: 7.4% |
| Transcription, conversational | WER (internal, 26) | 11.1% | — |
| Number formatting | format-exact / digit-recall (267) | 23.6% / 34.8% | whisper-large-v3: 32.2% / 59.9% |
| Diarization | DER ([VoxConverse](https://github.com/joonson/voxconverse) dev) | 40.7% (overlap **63.4%**) | specialist systems ~10–25% |
| Speaker attribution | WDER (internal) | 33% | no-linking baseline: ~42% |
| Word timing | MAE (full pipeline) | median ~90 ms (mean ~1.5 s) | — |
| Non-speech events | NSER (internal) | ~59% | — |
| Multilingual | WER/CER ([FLEURS](https://huggingface.co/datasets/google/fleurs)) | <7% on major langs; degrades low-resource | — |

The rest of this post is how each of those works, and where the numbers come from.

## Why fine-tune Whisper instead of training from scratch

The expensive way to get a model that understands speech events and speakers is to collect a massive labelled dataset and train from scratch. I didn't have that budget.

The bet behind WhisperDRZ is that Whisper has *already seen* most of what we want, it just doesn't surface it. Whisper-large-v3 was trained on a huge, messy web-scale corpus that almost certainly included TV and film caption tracks. Caption tracks are full of `[cheering]`, `[laughter]`, speaker turn markers, and the kind of overlapping, disfluent speech you never find in clean read-speech datasets. So the capabilities are latent in the weights; fine-tuning is mostly about *unlocking* them with a small amount of correctly-formatted data, not teaching them from zero.

That bet largely paid off. WhisperDRZ learns to emit non-speech tags from a tiny fraction of the data you'd expect, because it isn't learning what a laugh sounds like — it's learning that we want it written down.

One sharp limit on this: Whisper runs at 16 kHz. High-frequency non-speech events (tuts, sharp laughs, lip sounds) lose a lot of their detail at that sample rate. Low-frequency events like `[breath]` come through best; I suspect properly capturing the rest needs 24 kHz audio, which is future work.

How well does it actually do? Honestly, this is the weakest of WhisperDRZ's capabilities to put a number on. On our internal eval the non-speech-event error rate sits around **59%** — i.e. roughly half the events are missed or spurious. So treat event detection as "present and promising" rather than reliable: the model *will* surface breaths, laughs and hesitations, but you can't yet trust it to catch them all or avoid false ones. (A proper per-class precision/recall would need a dedicated event-annotated test set, which we don't have — the 59% is an aggregate over all tag types.)

## The format: speakers as audio channels

Each speaker gets a numeric label, `[0]`, `[1]`, and each speaker's words go on their own line:

```
[0] <|2.10|>hey<|2.32|> did you <|2.95|>[breath]<|3.10|> finish <|3.30|>the<|3.50|>
[1] <|3.05|>yeah<|3.28|> I sent <|3.60|>it<|3.85|>
```

Think of each line as an *audio channel*. Because the channels are separate, two speakers' words can overlap in time without the format breaking — which matters, because overlap is the norm, not the exception. (Watch almost any unscripted TV and you'll notice people talk over each other constantly; clean turn-taking is the artificial case.)

## Timing tokens: the core idea

Whisper's native timestamps are notoriously rough, and they fall apart on overlapping speech. Fixing timing was the first real problem.

The approach comes from [nanodrz](https://github.com/mogwai/nanodrz): instead of inferring timings after the fact, predict them *inline* as tokens. A line looks like `<|2.10|>word<|2.74|>` — the model emits a timestamp token, the word, and a closing timestamp.

Two things made this work well:

**Encode the time value, not the token position.** A naive timestamp token is just another vocabulary entry — the model has no idea that `<|2.10|>` and `<|2.12|>` are *almost the same moment*. We fix this by adding a sinusoidal embedding of the actual time value to each timing token, the same trick that gives transformers a sense of position, but applied to the time axis. This way the timestamps know their relative position to one another. This single change made a large difference to timing quality.

**Predict sparse timings, fill the rest with DTW.** Emitting a timestamp for every word doubles the decode length, and the decoder is the inference bottleneck. So WhisperDRZ predicts only the timings that matter most — the first and last word of each line, plus a few scattered anchors so the model always knows where it is in the audio — and then uses dynamic time warping (DTW) over the cross-attention weights to fill in the rest. We found timing tokens substantially outperform pure DTW alignment, especially on overlapping speech, even when DTW is helped along with strong attention-alignment losses. Combining the two gives the best of both: token-level accuracy at the anchors, cheap interpolation everywhere else.

How good are the timings? It depends how you measure. On a standalone DTW-alignment benchmark, word start times land at a median absolute error of ~35 ms — but that's the best case. Through the **full production pipeline on real conversational audio**, the median word-timing error is about **90 ms** (start and end), which is still solid — but the *mean* is ~1.5 s, because a long tail of words ends up badly mistimed. So: most words are placed within ~100 ms, but timing is not uniformly reliable, and the occasional word can be seconds off.

## Inference guardrails: a "virtual mask"

Training on real, messy caption data has a downside: run inference naively and the model occasionally falls into a bad context and hallucinates catastrophically — invalid timings, malformed speaker lines, runaway brackets.

Rather than fix this purely with more data (the correct but expensive answer), I added a *VirtualMask* at inference time. After each step it inspects the partial prediction and masks out tokens that would be illegal given what's already been generated. A few examples of the rules it enforces:

- Timings within a line must be **monotonic** (the same speaker can't go backwards in time; different speakers are allowed a few seconds of overlap).
- A timing token can only appear in **valid positions**, and lines must be properly closed with an end timing.
- A timing can't jump **too far ahead or behind** what's already happened.
- Brackets, tags and parentheses must be **properly nested**, with forced closes if they aren't.

The full rule set is in the code. Borrowing the React analogy, the mask works like a virtual DOM: we compute the legal-token mask for the whole batch in one pass and apply it, so the guardrails cost almost nothing at inference time.

How much does it actually help? I ablated it — same released model, mask on vs off:

- On **messy conversational audio** (the in-domain case), it helps modestly: WER **11.1% with the mask, 12.3% without**, a handful of clips a few points worse when it's removed. No catastrophic blow-ups in that set, though — on the released model the mask is a small win, not the dramatic save I'd expected.
- On **clean English read speech** it's essentially neutral — very slightly *worse*, in fact (5.4% vs 5.2%). The model doesn't hallucinate on easy audio, so the constraints have nothing to fix.
- On **other languages** it's actively harmful: the rules are shaped around English and WhisperDRZ's format, and forcing them onto a language the model wasn't trained to format derails decoding badly (more on that in the multilingual section).

So it's an honest hack — a modest in-distribution win, neutral on easy audio, and a liability out-of-distribution. The cleaner fix is still more data in the right format.

## Speaker identity: an honest negative result

Diarizing a 30-second chunk is one thing. Keeping speaker `[0]` *the same person* across an hour-long recording is much harder, and it's where I spent the most effort for the least payoff.

The plan was elegant: add a **speaker embedding head** that decodes a speaker vector straight from the `[N]` speaker-label token, train it with a contrastive (cosine-similarity) loss so the same person maps to nearby vectors within and across chunks, then link speakers by comparing embeddings. The head trained to **93%+ top-1 speaker accuracy** on held-out eval. It looked done.

It wasn't. On real multi-speaker audio the integrated head was **no better than not linking speakers at all** — its cross-chunk word diarization error rate matched the no-op baseline. The embeddings simply didn't generalize from training conditions to noisy, overlapping real audio. (Part of the problem was a train/inference mismatch in *where* the embedding was read from, but even accounting for that, the head underperformed.)

So WhisperDRZ falls back to an external speaker-embedding model (ECAPA2) with per-line clustering to link speakers across chunks. It helps measurably — on the internal eval it brings word-diarization error rate down to about **33%, from ~42% with no cross-chunk linking** — but two things are worth being honest about: a third of words still land on the wrong speaker, and the whole approach leans on an unprincipled clustering step rather than something the model itself learned. The diarization quality you saw on VoxConverse above is *carried* by this external linker, not by WhisperDRZ's own speaker representation.

Beyond the accuracy, clustering is an inherently unstable, offline approach to speaker ID, and it makes true streaming hard (though not impossible). The right answer is in-context speaker identification — more on that below.

The honest summary: the integrated speaker head is a clean idea that didn't survive contact with real data, and I'm reporting it as a negative result rather than burying it.

Perhaps this was a data problem in that our data wasn't clean enough, or perhaps the cross-entropy loss fought too hard with the various contrastive losses we experimented with.

## Does adding all this hurt transcription?

The natural worry: piling diarization, timings and event tags onto Whisper should wreck its word error rate. It does cost something, but less than you'd fear.

| Model | Avg WER (ESB) | Notes |
|---|---|---|
| whisper-large-v3 (baseline) | 7.44% | [Open ASR Leaderboard](https://github.com/huggingface/open_asr_leaderboard) |
| **WhisperDRZ large-v3 (released)** | **9.61%** | full ESB, 1000 samples (micro 5.88%) |

So the full speaker-aware, event-aware, timing-aware model costs about **2 points of WER** over the base model (9.6% vs 7.4% macro) — a modest price for everything it adds, and on clean [LibriSpeech](https://www.openslr.org/12) the gap is much smaller. Some of getting that number down was *decoding* discipline rather than training — on an earlier, shakier checkpoint, tightening the inference guardrails and dropping a few pathological token classes took WER from ~18% to ~7%. (On the released model the guardrails matter much less — see the ablation above.)

### A note on these numbers

Every headline number in this post — this ESB WER, the VoxConverse DER below, and the end-to-end timing figures — is measured on the **publicly released weights**, i.e. exactly the checkpoint you can download. Where two numbers exist for the same thing (e.g. best-case DTW timing vs. full-pipeline timing), both are given so the gap is visible rather than hidden.

It's also cheap to run: WhisperDRZ reaches ~92x real-time on a single GPU, which works out to roughly $0.01 per hour of audio on a commodity cloud GPU — about 36x cheaper than calling a hosted Whisper API, with diarization included rather than billed separately.

## Numbers: still spoken, not written

A specific weakness worth naming: WhisperDRZ doesn't reliably render spoken numbers in their *written* form. Say a phone number out loud and you want `07700 900123`, not "oh seven seven oh oh…"; a price should come back as `£91.09`, a date as `2025-02-15`.

On a dedicated 267-clip eval across 11 categories (phone, money, date, time, version, fraction, measurement, score, ordinal, IP, code), the released checkpoint hits **23.6% format-exact / 34.8% digit-recall** — *behind* whisper-large-v3's **32.2% / 59.9%**. The dominant failure isn't wrong digits, it's leaving the number as words entirely: **37% of clips emit no digits at all.** The one place it clearly wins is ISO dates (`2025-02-15`), a format the base model never produces.

The digits *are* present in the training targets, so this reads as a data-proportion / undertraining issue rather than a wrong-objective one — more correctly-formatted number data, or longer training on it, should close the gap. For now, don't rely on WhisperDRZ for structured numeric transcription. (Standard WER hides this entirely: Whisper's text normalizer collapses `747` to "seven hundred forty seven", so this eval scores against un-normalized ground truth on both format-exact and digit-recall.)

<details>
<summary>Full per-category breakdown (267 clips)</summary>

| category | n | WhisperDRZ format | large-v3 format | WhisperDRZ digit-recall | large-v3 digit-recall |
|---|---|---|---|---|---|
| phone | 25 | 28% | 32% | 28% | 76% |
| money | 25 | 8% | 16% | 40% | 80% |
| date | 25 | 12% | 0% | 12% | 0% |
| time | 25 | 4% | 0% | 32% | 44% |
| version | 25 | 44% | 36% | 72% | 100% |
| fraction | 17 | 0% | 0% | 12% | 0% |
| measurement | 25 | 20% | 20% | 28% | 44% |
| score | 25 | 28% | 44% | 28% | 56% |
| ordinal | 25 | 36% | 76% | 36% | 76% |
| ip | 25 | 20% | 72% | 24% | 84% |
| code | 25 | 52% | 48% | 64% | 80% |
| **overall** | **267** | **23.6%** | **32.2%** | **34.8%** | **59.9%** |

*format-exact* = the expected written string appears in the output (ignoring spacing/case); *digit-recall* = the expected digit sequence appears, any format. WhisperDRZ = released checkpoint.

</details>

## How good is the diarization, really?

Transcription is the easy part; "who said what" is the harder claim, so I measured it on a public benchmark — **VoxConverse**, in-the-wild conversational video (interviews, debates, panel shows) with heavy overlap. The metric is **diarization error rate (DER)**: missed speech + false alarm + speaker confusion, over total reference speech, at the standard 0.25s collar. This runs the full production pipeline — chunked decoding, DTW word timings, and [ECAPA2](https://huggingface.co/Jenthe/ECAPA2) cross-chunk speaker linking.

| VoxConverse dev (216 files) | DER |
|---|---|
| **Overall** | **40.7%** |
| Median per-file | 38.2% |
| **Overlap regions only** | **63.4%** |

The honest read:

- **~41% DER overall.** Specialist diarizers ([pyannote](https://github.com/pyannote/pyannote-audio), the [VoxSRC](https://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/workshop.html) challenge systems) land roughly in the **10–25%** range on this data. So WhisperDRZ — an ASR-first model with diarization folded into its output format — is *usable* but **not competitive with purpose-built diarization systems**.
- **Overlap is the weak spot (63% DER).** This is the most important correction to the optimistic framing earlier in this post. WhisperDRZ can *represent* overlapping speech in its channel format, but actually getting overlap *right* on real, crowded audio is much harder than the format makes it look. Overlap is an open problem here, not a solved feature.
- **Cross-chunk speaker linking matters enormously.** Without the ECAPA2 step, speaker labels scramble across 30-second chunks and DER roughly doubles. The embedding-based linking is doing a lot of the work — and, as the speaker section above admits, it's the least principled part of the system.
- **No VAD = false alarm.** WhisperDRZ has no voice-activity gate, so it transcribes over music and silence; on VoxConverse clips that are mostly non-speech, that inflates false alarm (a few clips even exceed 100% DER, where hypothesis speech dwarfs the few seconds of real reference speech). A simple non-speech filter would recover a meaningful slice of this.

So: WhisperDRZ tells you who said what well enough to be useful on clean, few-speaker audio, but trails dedicated diarizers on crowded in-the-wild conversation — and overlap specifically remains unsolved. (Baseline caveat: I haven't yet run pyannote on this exact 216-file set; the 10–25% reference is from published numbers, so treat the comparison as indicative.)

## What about other languages?

Whisper-large-v3 speaks around 100 languages, and WhisperDRZ was fine-tuned almost entirely on English. The obvious fear is that we bought English diarization at the cost of everything else. To check, I ran a **forced-language** WER eval on FLEURS across **all 83 languages it covers**, on the **released** WhisperDRZ checkpoint. (Chinese, Japanese, Thai, Cantonese, Khmer, Lao and Burmese use character error rate rather than word error rate, since they aren't space-segmented.)

A note on what this measures. "Forced-language" means the model is *told* the language up front — the ground-truth language token is fed into the decoder prefix — so this isolates raw transcription ability from language *identification* (whether the model can guess the language is a separate question, and one where fine-tuning did cost us). It's also the friendly case in two other ways: FLEURS is clean, single-speaker *read* speech, and the eval scores only the words — none of WhisperDRZ's diarization, timings or event tags are exercised here. So this answers one narrow question: *given the right language and easy audio, how much did English fine-tuning damage WhisperDRZ's transcription elsewhere?*

The good news: **the major world languages survive fine-tuning well.** Every one of these stays under ~10%:

| Language | Metric | Error |
|---|---|---|
| Spanish | WER | 2.5% |
| Italian | WER | 2.9% |
| Catalan | WER | 4.7% |
| Portuguese | WER | 4.9% |
| Russian | WER | 5.2% |
| English | WER | 5.5% |
| German | WER | 5.7% |
| Japanese | CER | 5.7% |
| Polish | WER | 6.1% |
| Indonesian | WER | 6.4% |
| Dutch | WER | 6.4% |
| Ukrainian | WER | 7.2% |
| Turkish | WER | 7.3% |
| Malay | WER | 7.7% |
| Finnish | WER | 9.2% |
| Swedish | WER | 9.4% |
| Vietnamese | WER | 10.0% |

Past that, quality falls off progressively. Where I have a controlled comparison against stock large-v3 (a 19-language subset run through the identical harness), the only **clear double-digit regression is Chinese** (CER roughly 7% → 19%); French slips a few points (6% → 11%); most other high-resource languages stay within about a point of the base model. A long tail of lower-resource languages — Welsh (34%), Icelandic (40%), Nepali (49%), and most Indic, several African and Central-Asian languages — degrades sharply, and the bottom of the list is effectively unsupported. (Korean looks poor at ~68%, but it's scored with WER where it really wants CER, so treat that as a scoring artifact rather than a true collapse.)

A big caveat on the worst numbers, though: **much of the broken tail is a decoding artifact, not the model being incapable.** This eval runs plain greedy decoding with *no* guardrails at all — and the languages scoring *above* 100% error (Amharic at 284%, Burmese, Mongolian, Sindhi…) are simply the decoder falling into a repetition loop and emitting tokens until it hits the limit. That's not "the model can't transcribe Amharic"; it's "with zero decoding discipline it runs away." Even language-agnostic rules — a repetition penalty, no-repeat-ngram, a cap on consecutive tokens — would collapse most of that tail, before any per-language work.

And per-language work is the deeper point. WhisperDRZ's real inference path applies a [VirtualMask](#inference-guardrails-a-virtual-mask) whose rules — where timing tokens may appear, how brackets and speaker digits nest — were shaped around English and WhisperDRZ's training format. I tested what happens when you point those rules at another language, and the result is emphatic: running these broken-tail languages through the *full production pipeline* (guardrails on) made them **dramatically worse**, not better — Korean went from 68% to **466%** WER, Amharic from 284% to 420%, Serbian from 88% to 359%. Forcing WhisperDRZ's English timing-and-format token patterns onto a language it was never trained to format derails decoding into runaway insertion. So the guardrails actively *hurt* non-English, and the plain-greedy numbers in the table above are the *better* view of what the weights actually know. So the honest read of this table isn't a clean capability ranking — it's a lower bound that conflates the weights' actual multilingual knowledge with the absence (and, in production, the English-centricity) of decoding rules. Getting trustworthy multilingual numbers would mean building per-language decoding constraints and re-measuring. One more reason it's a lower bound: this is clean read speech with the language handed to the model; real conversational, diarized audio is harder still.

<details>
<summary>All 83 languages (WhisperDRZ v1, forced-language, sorted)</summary>

| Language | Metric | Error |
|---|---|---|
| Spanish (es) | WER | 2.5% |
| Italian (it) | WER | 2.9% |
| Catalan (ca) | WER | 4.7% |
| Portuguese (pt) | WER | 4.9% |
| Russian (ru) | WER | 5.2% |
| English (en) | WER | 5.5% |
| German (de) | WER | 5.7% |
| Japanese (ja) | CER | 5.7% |
| Polish (pl) | WER | 6.1% |
| Indonesian (id) | WER | 6.4% |
| Dutch (nl) | WER | 6.4% |
| Ukrainian (uk) | WER | 7.2% |
| Turkish (tr) | WER | 7.3% |
| Malay (ms) | WER | 7.7% |
| Finnish (fi) | WER | 9.2% |
| Swedish (sv) | WER | 9.4% |
| Vietnamese (vi) | WER | 10.0% |
| Norwegian (no) | WER | 10.2% |
| Romanian (ro) | WER | 10.3% |
| Czech (cs) | WER | 10.4% |
| Croatian (hr) | WER | 10.7% |
| French (fr) | WER | 10.9% |
| Slovak (sk) | WER | 11.0% |
| Galician (gl) | WER | 12.6% |
| Bulgarian (bg) | WER | 12.6% |
| Tagalog (tl) | WER | 13.7% |
| Arabic (ar) | WER | 13.9% |
| Greek (el) | WER | 14.0% |
| Bosnian (bs) | WER | 14.5% |
| Danish (da) | WER | 14.9% |
| Thai (th) | CER | 16.0% |
| Cantonese (yue) | CER | 16.9% |
| Hindi (hi) | WER | 18.3% |
| Hungarian (hu) | WER | 18.3% |
| Estonian (et) | WER | 18.6% |
| Macedonian (mk) | WER | 19.0% |
| Chinese (zh) | CER | 19.3% |
| Azerbaijani (az) | WER | 21.4% |
| Latvian (lv) | WER | 21.9% |
| Slovenian (sl) | WER | 21.9% |
| Tamil (ta) | WER | 22.3% |
| Persian (fa) | WER | 22.5% |
| Urdu (ur) | WER | 23.5% |
| Hebrew (he) | WER | 25.7% |
| Lithuanian (lt) | WER | 29.2% |
| Welsh (cy) | WER | 34.4% |
| Afrikaans (af) | WER | 35.2% |
| Kazakh (kk) | WER | 37.9% |
| Icelandic (is) | WER | 39.6% |
| Swahili (sw) | WER | 40.2% |
| Maori (mi) | WER | 44.1% |
| Belarusian (be) | WER | 46.1% |
| Armenian (hy) | WER | 48.9% |
| Nepali (ne) | WER | 49.3% |
| Telugu (te) | WER | 51.9% |
| Bengali (bn) | WER | 56.2% |
| Marathi (mr) | WER | 59.1% |
| Gujarati (gu) | WER | 60.5% |
| Punjabi (pa) | WER | 63.2% |
| Korean (ko) | WER | 67.9% |
| Occitan (oc) | WER | 70.9% |
| Lingala (ln) | WER | 72.4% |
| Maltese (mt) | WER | 72.7% |
| Javanese (jw) | WER | 77.4% |
| Luxembourgish (lb) | WER | 86.2% |
| Serbian (sr) | WER | 87.7% |
| Uzbek (uz) | WER | 92.5% |
| Georgian (ka) | WER | 93.2% |
| Pashto (ps) | WER | 94.7% |
| Yoruba (yo) | WER | 97.5% |
| Hausa (ha) | WER | 99.4% |
| Khmer (km) | CER | 100.0% |
| Kannada (kn) | WER | 100.0% |
| Assamese (as) | WER | 100.0% |
| Malayalam (ml) | WER | 100.0% |
| Somali (so) | WER | 101.0% |
| Lao (lo) | CER | 101.2% |
| Shona (sn) | WER | 102.6% |
| Mongolian (mn) | WER | 107.2% |
| Sindhi (sd) | WER | 108.8% |
| Tajik (tg) | WER | 109.7% |
| Burmese (my) | CER | 110.8% |
| Amharic (am) | WER | 284.2% |

</details>

So the honest claim is narrower than "WhisperDRZ retains all of Whisper's languages." It's: **WhisperDRZ keeps strong transcription across the major world languages, regresses clearly on Chinese, and degrades across most lower-resource languages** — the inference masking would need per-language tuning, and several languages would want dedicated data.

## What I'd do next

**Move to audio codecs.** It was obvious from the start that the right substrate for this is a discrete audio codec, not 16 kHz mel features at 50 Hz — and ideally a **low-frame-rate** one, **sub-12.5 Hz** (where Kyutai's [Mimi](https://github.com/kyutai-labs/moshi) sits), so each token covers more time, sequences stay short, context stretches to hours at low latency, and event detection gets higher fidelity. Training a codec and a model on top of it was simply out of budget and bandwidth for one person, and would have meant sourcing a huge from-scratch dataset. But it's the clear path forward — and the substrate that makes everything below, streaming especially, tractable.

**In-context speaker ID.** With a longer context you could do what I showed work in [nanodrz](https://github.com/mogwai/nanodrz): prompt each new chunk with a few labelled chunks from earlier in the same recording, so the model carries speaker identity forward *in context*. That throws away the whole flawed embedding-and-clustering pipeline and makes streaming natural. The budget is realistic on a low-Hz codec: getting to ~12 minutes of context (only ~9k tokens at 12.5 Hz, fewer below that) is enough to carry a decent number of speakers forward from the previous inference chunk — they arrive **pre-identified**, so each new chunk just has to *match* against labelled exemplars rather than re-cluster from scratch. (Recent systems like [VibeVoice-ASR](https://github.com/microsoft/VibeVoice) point in a similar direction.)

**Towards true streaming.** Everything here is chunked and offline — 30s windows, then external clustering stitched on top. On the low-Hz codec above, a genuinely streaming WhisperDRZ just needs a way to emit text *as the audio streams in*, via one of two routes:

- **A delayed text channel** — [Kyutai's delayed-streams modeling](https://github.com/kyutai-labs/delayed-streams-modeling): model audio and text as parallel, time-aligned streams with the text delayed a fixed offset behind the audio (Kyutai run ~500 ms for their 1B, 2.5 s for the 2.6B). Text simply falls out of the logits as audio arrives — no wrapper logic — and you get word-level timestamps and a semantic "is the speaker done" VAD for free.
- **Reverse the vui chunk format** — [vui](https://github.com/fluxions-ai/vui) (our TTS model) interleaves each short chunk as `[text → audio codes]`; flip it to `[audio codes → text]` and the exact same architecture becomes a chunked streaming ASR. That also unifies the listening and speaking halves into one format, which was the point all along.

**A pure-diarization task.** Everything in this post couples diarization to transcription — the model has to *read* the words to assign speakers, and the cross-chunk linking then leans on an external embedder. One thing I'd like to try is training WhisperDRZ to do diarization *on its own*, with no transcription at all — a target that's just speaker lines and speech-region timestamps:

```
[0] <|t|><|speech|><|t|>
[1] <|t|><|speech|><|t|>
...
```

Each line says "speaker N is talking from t to t," nothing more. Two things make this appealing. First, it lets the model learn diarization as a first-class task rather than a side effect of transcription, which might finally give it a speaker representation good enough to replace the clustering hack. Second, because it never decodes words, it could double as a very fast standalone speaker-ID model — answer "are these two segments the same person?" directly, without paying for transcription. That's exactly the capability the failed embedding head was reaching for, approached as a decoding task instead of an auxiliary head.

**Scale matters for this task.** I tried the same fine-tuning recipe on smaller Whisper bases and the results were poor. Doing diarization *and* transcription well seems to need 1B+ parameters — the smaller models just don't have the headroom to track speakers and words at once. In fact it may be the case that > 1B is needed to do all these things.

**Emotions.** We started work on emotions and there are some preliminary interesting results. More data is needed to see whether this works; it will be part of the next version.

## Closing

WhisperDRZ started from a simple premise: telling a model more about *what is actually happening* in audio — who's speaking, when, and the small human noises in between — makes it a better listener, not a worse transcriber. The evidence mostly supports that. The transcription cost of adding speakers, timings and events is small; the timing approach is genuinely good; and even the parts that failed (the speaker head) failed in instructive ways.

The model weights are released for inference. The code, including the full guardrail rule set, is in this repository.
