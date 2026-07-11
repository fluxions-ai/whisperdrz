import os
import re
from collections import defaultdict
from dataclasses import dataclass

import torch

from whisperdrz.audio import LogMelSpectrogram
from whisperdrz.speaker_clustering import (
    cluster_speakers,
    normalize_speaker_ids,
)
from whisperdrz.utils import flash_kernel

DEBUG = os.getenv("AKRO_DEBUG_INFERENCE", "0") == "1"

# Batch size buckets for CUDA graphs (decode step only)
CUDA_GRAPH_BUCKETS = [1, 2, 4, 8, 16, 32, 42, 52, 54]


def _get_bucket(batch_size: int, buckets: list[int] | None = None) -> int:
    """Get smallest bucket that fits batch_size."""
    bucket_list = buckets if buckets is not None else CUDA_GRAPH_BUCKETS
    for b in bucket_list:
        if batch_size <= b:
            return b
    # Return batch_size itself if larger than all buckets (triggers fallback)
    return batch_size


class BucketedCUDAGraphs:
    """CUDA graphs for different batch size buckets to speed up decode loop.

    CUDA graphs capture a sequence of GPU operations and replay them with minimal
    CPU overhead. For autoregressive decoding where we call forward_flash hundreds
    of times, this eliminates per-step kernel launch overhead.

    Supports both modes:
    - return_hidden_states=False: faster, returns only logits
    - return_hidden_states=True: returns (logits, hidden_states) for speaker embeddings
    """

    def __init__(
        self,
        model,
        self_kv: list,
        cross_kv: list,
        return_hidden_states: bool = False,
    ):
        self.model = model
        self.self_kv = self_kv
        self.cross_kv = cross_kv
        self.device = model.device
        self.dtype = model.dtype
        self.return_hidden_states = return_hidden_states

        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.static_tokens: dict[int, torch.Tensor] = {}
        self.static_positions: dict[int, torch.Tensor] = {}
        self.static_slot_indices: dict[int, torch.Tensor] = {}
        self.static_output: dict[int, torch.Tensor] = {}
        self.static_hidden: dict[int, torch.Tensor] = {}  # For hidden states
        self.buckets: list[int] = []

        self.hits = defaultdict(int)
        self.misses = 0

    def warmup_and_capture(self, bucket: int):
        """Warmup and capture CUDA graph for a specific bucket size."""
        self.static_tokens[bucket] = torch.zeros(
            bucket, 1, dtype=torch.long, device=self.device
        )
        self.static_positions[bucket] = torch.zeros(
            bucket, dtype=torch.long, device=self.device
        )
        self.static_slot_indices[bucket] = torch.arange(
            bucket, dtype=torch.long, device=self.device
        )
        self.static_output[bucket] = torch.zeros(
            bucket, 1, self.model.dims.n_vocab, dtype=self.dtype, device=self.device
        )
        if self.return_hidden_states:
            self.static_hidden[bucket] = torch.zeros(
                bucket,
                1,
                self.model.dims.n_text_state,
                dtype=self.dtype,
                device=self.device,
            )

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _ = self.model.decoder.forward_flash(
                    self.static_tokens[bucket],
                    self.self_kv,
                    self.cross_kv,
                    self.static_positions[bucket],
                    slot_indices=self.static_slot_indices[bucket],
                    return_hidden_states=self.return_hidden_states,
                )
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            result = self.model.decoder.forward_flash(
                self.static_tokens[bucket],
                self.self_kv,
                self.cross_kv,
                self.static_positions[bucket],
                slot_indices=self.static_slot_indices[bucket],
                return_hidden_states=self.return_hidden_states,
            )
            if self.return_hidden_states:
                self.static_output[bucket], self.static_hidden[bucket] = result
            else:
                self.static_output[bucket] = result
        self.graphs[bucket] = g

    def capture_all(self, max_bucket: int | None = None):
        """Capture graphs for all bucket sizes up to max_bucket."""
        self.buckets = [
            b for b in CUDA_GRAPH_BUCKETS if max_bucket is None or b <= max_bucket
        ]
        # Ensure max_bucket itself is included for exact batch size match
        if max_bucket is not None and max_bucket not in self.buckets:
            self.buckets.append(max_bucket)
            self.buckets.sort()
        if DEBUG:
            print(f"Capturing CUDA graphs for buckets: {self.buckets}")
        for bucket in self.buckets:
            self.warmup_and_capture(bucket)
        if DEBUG:
            print(f"Captured {len(self.buckets)} CUDA graphs")

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        slot_indices: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass using appropriate CUDA graph.

        Returns:
            If return_hidden_states=False: logits tensor
            If return_hidden_states=True: (logits, hidden_states) tuple
        """
        B = tokens.shape[0]
        bucket = _get_bucket(B, self.buckets)

        if bucket not in self.graphs:
            self.misses += 1
            return self.model.decoder.forward_flash(
                tokens,
                self.self_kv,
                self.cross_kv,
                positions,
                slot_indices=slot_indices,
                return_hidden_states=self.return_hidden_states,
            )

        self.hits[bucket] += 1

        self.static_tokens[bucket][:B].copy_(tokens)
        self.static_positions[bucket][:B].copy_(positions)
        self.static_slot_indices[bucket][:B].copy_(slot_indices)

        self.graphs[bucket].replay()

        if self.return_hidden_states:
            return self.static_output[bucket][:B], self.static_hidden[bucket][:B]
        return self.static_output[bucket][:B]


from whisperdrz.clean import (
    SPEAKER_LINE_START,
    TIMINGS_FORMAT,
    consolidate_adjacent_tags,
    fix_nonmonotonic_timestamps,
    fix_speaker_ids,
    merge_consecutive_speakers,
    reconcile_decoder_and_alignment_timings,
    remove_invalid_timing_pairs,
    remove_orphan_paired_tags,
    remove_short_nonspeech,
    remove_short_pauses,
    remove_speaker_tags_not_after_newline,
)
from whisperdrz.decoding import DecoderState, decode_batch, update_state_after_decode



def _strip_to_last_closing(cur: str) -> str:
    """Strip trailing content after the last closing timestamp.

    When continuing to the next window, content past the last closing timestamp
    is re-covered by the next window. Stripping it prevents duplicate timestamps
    at window stitch points.

    A closing timestamp is identified by even parity (timestamps alternate
    open/close on each line). If the last line ends with an orphaned opening,
    we strip back to the previous closing.
    """
    lines = cur.split("\n")
    for i in range(len(lines) - 1, -1, -1):
        ts = list(TIMINGS_FORMAT.finditer(lines[i]))
        if not ts:
            continue
        if len(ts) % 2 == 0:
            # Even count: last timestamp is a closing, strip after it
            lines[i] = lines[i][: ts[-1].end()]
            return "\n".join(lines[: i + 1])
        if len(ts) > 1:
            # Odd count: last is orphaned opening, strip to previous (closing)
            lines[i] = lines[i][: ts[-2].end()]
            return "\n".join(lines[: i + 1])
        # Single orphaned opening on this line — drop it, check previous line
    return cur


@dataclass
class TranscribeResults:
    text: str
    raw_chunks: list[dict] | None = None
    segments: list[dict] | None = None


def parse_to_segments(text: str) -> list[dict]:
    """Parse transcription text into segments with speaker, timing, and text.

    Each line becomes a segment with:
        - speaker: int | str (speaker ID from [N] tag, or "c" for crowd)
        - start: float (first timestamp)
        - end: float (last timestamp)
        - text: str (words without timestamps)
    """
    segments = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("[CHOP]"):
            continue

        speaker: int | str = 0
        speaker_match = SPEAKER_LINE_START.match(line)
        if speaker_match:
            spk = speaker_match.group(1)
            speaker = spk if spk == "c" else int(spk)
            line = line[speaker_match.end() :]

        timings = list(TIMINGS_FORMAT.finditer(line))
        if len(timings) < 2:
            continue

        start = float(timings[0].group(1))
        end = float(timings[-1].group(1))
        if end <= start:
            continue

        words = TIMINGS_FORMAT.sub("", line).strip()
        words = re.sub(r"\s+", " ", words)

        segments.append(
            {
                "speaker": speaker,
                "start": round(start, 2),
                "end": round(end, 2),
                "text": words,
            }
        )

    return segments


def extract_line_audio_segments(
    audio: torch.Tensor,
    text: str,
    min_duration: float = 0.3,
    max_duration: float = 10.0,
) -> list[tuple[int, int, torch.Tensor]]:
    """Extract one audio tensor per speaker line for speaker embedding.

    For lines longer than max_duration, picks a random window of 3-max_duration
    seconds. Speaker embedding models only need a few seconds of audio.

    Returns:
        List of (line_index, speaker_id, audio_tensor) for lines with enough
        audio. Lines with [c] crowd tag or < min_duration are skipped.
    """
    import random

    segments = parse_to_segments(text)
    results = []
    min_window = 3.0
    for line_idx, seg in enumerate(segments):
        if seg["speaker"] == "c":
            continue
        s_sample = int(max(0, seg["start"]) * 16000)
        e_sample = int(seg["end"] * 16000)
        if e_sample > len(audio):
            e_sample = len(audio)
        duration_samples = e_sample - s_sample
        if duration_samples < int(min_duration * 16000):
            continue
        max_samples = int(max_duration * 16000)
        if duration_samples > max_samples:
            # Pick a random window between min_window and max_duration
            window = int(random.uniform(min_window, max_duration) * 16000)
            window = min(window, duration_samples)
            offset = random.randint(0, duration_samples - window)
            s_sample += offset
            e_sample = s_sample + window
        results.append((line_idx, seg["speaker"], audio[s_sample:e_sample]))
    return results


def remap_speakers_per_line(
    text: str,
    audio: torch.Tensor,
    speaker_model,
    threshold: float = 1.1,
    min_duration: float = 1.0,
) -> str:
    """Remap speaker IDs by clustering per-line embeddings from scratch.

    Does NOT trust the model's speaker IDs. Instead:
    1. Parse all lines, extract audio per line
    2. Embed each embeddable line (>= min_duration) with ECAPA2
    3. Cluster embeddings with AHC (ignoring model-predicted IDs)
    4. Short unembedded lines inherit speaker from nearest embedded neighbor,
       preferring neighbors whose model-predicted ID matches the short line's

    Args:
        threshold: Euclidean distance threshold for AHC clustering.
            For L2-normalized ECAPA2 embeddings, ~1.0 is a good starting point.
        min_duration: Minimum line duration in seconds to attempt embedding.
    """
    segments = parse_to_segments(text)
    if not segments:
        return text

    device = next(speaker_model.parameters()).device

    line_segments = extract_line_audio_segments(audio, text, min_duration=min_duration)
    if not line_segments:
        return text

    # Embed each line — pass 0 as local_id since we don't trust model IDs
    embedded: list[tuple[int, int, torch.Tensor]] = []
    for line_idx, _spk, line_audio in line_segments:
        emb = _embed_audio_segment(line_audio, speaker_model, device)
        embedded.append((line_idx, 0, emb))

    if len(embedded) < 2:
        return text

    # Cluster from scratch
    cluster_map = cluster_speakers(embedded, threshold=threshold)

    # Build line_idx -> cluster_id for embedded lines
    embedded_cluster: dict[int, int] = {}
    for line_idx, _, _ in embedded:
        embedded_cluster[line_idx] = cluster_map[(line_idx, 0)]

    # Build lookup of embedded line indices and their start times for neighbor search
    embedded_set = set(embedded_cluster.keys())
    embedded_info = []
    for line_idx in sorted(embedded_set):
        embedded_info.append((line_idx, segments[line_idx]["start"]))

    # Model-predicted speaker per embedded line (for matching short lines)
    model_spk_for_line: dict[int, int | str] = {}
    for line_idx, spk, _ in line_segments:
        model_spk_for_line[line_idx] = spk
    # Also record model speaker for all segments (including unembedded)
    for line_idx, seg in enumerate(segments):
        if seg["speaker"] != "c" and line_idx not in model_spk_for_line:
            model_spk_for_line[line_idx] = seg["speaker"]

    # For unembedded speakers with no embedded match, allocate new cluster IDs
    max_cluster = max(embedded_cluster.values()) if embedded_cluster else -1
    unembedded_spk_to_cluster: dict[int | str, int] = {}

    # Assign all lines
    all_line_speakers: dict[int, int] = {}
    for line_idx, seg in enumerate(segments):
        if seg["speaker"] == "c":
            continue
        if line_idx in embedded_cluster:
            all_line_speakers[line_idx] = embedded_cluster[line_idx]
        elif embedded_info:
            # Short/unembedded line: find nearest embedded neighbor
            line_start = seg["start"]
            model_spk = seg["speaker"]

            # First try: find an embedded line whose model-predicted speaker matches
            best_match_dist = float("inf")
            best_match_id = None
            for ei, ei_start in embedded_info:
                if model_spk_for_line.get(ei) == model_spk:
                    dist = abs(line_start - ei_start)
                    if dist < best_match_dist:
                        best_match_dist = dist
                        best_match_id = ei

            if best_match_id is not None:
                all_line_speakers[line_idx] = embedded_cluster[best_match_id]
            else:
                # No embedded line shares this model speaker — this speaker only
                # has short lines (interjections). Give them a distinct cluster
                # so they aren't merged into surrounding speakers.
                if model_spk not in unembedded_spk_to_cluster:
                    max_cluster += 1
                    unembedded_spk_to_cluster[model_spk] = max_cluster
                all_line_speakers[line_idx] = unembedded_spk_to_cluster[model_spk]

    # Remap [N] tags in text line by line
    lines = text.split("\n")
    result_lines = []
    seg_idx = 0
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("[CHOP]"):
            result_lines.append(line)
            continue
        m = SPEAKER_LINE_START.match(stripped)
        if not m:
            result_lines.append(line)
            continue
        spk_str = m.group(1)
        if spk_str == "c":
            result_lines.append(line)
            timings = list(TIMINGS_FORMAT.finditer(stripped))
            if (
                len(timings) >= 2
                and seg_idx < len(segments)
                and segments[seg_idx]["speaker"] == "c"
            ):
                seg_idx += 1
            continue
        timings = list(TIMINGS_FORMAT.finditer(stripped))
        if len(timings) >= 2 and seg_idx < len(segments):
            new_spk = all_line_speakers.get(seg_idx, int(spk_str))
            rest = stripped[m.end() :]
            new_line = f"[{new_spk}] {rest}" if rest else f"[{new_spk}]"
            result_lines.append(new_line)
            seg_idx += 1
        else:
            result_lines.append(line)

    return "\n".join(result_lines)


def _embed_audio_segment(
    audio_tensor: torch.Tensor, model, device: torch.device
) -> torch.Tensor:
    """Embed a single audio segment with the speaker model.

    Supports:
    - ECAPA2 JIT models (input: raw audio (1, T) at 16kHz, float32 only)
    - whisperdrz speakerid models (input: mel spectrogram)
    """
    import torch.nn.functional as F

    is_jit = isinstance(model, torch.jit.ScriptModule)

    if is_jit:
        # ECAPA2: raw audio (1, T), must be float32 (half produces garbage)
        inp = audio_tensor.unsqueeze(0).to(device).float()
        with torch.inference_mode():
            emb = model(inp)
        return F.normalize(emb.squeeze(0).cpu().float(), dim=-1)
    # whisperdrz speakerid: mel spectrogram input
    model.to_mel.to(device)
    mel = model.to_mel(audio_tensor.unsqueeze(0).to(device))
    n_frames = mel.shape[-1]
    if n_frames > 3000:
        mel = mel[:, :, :3000]
    elif n_frames < 3000:
        mel = F.pad(mel, (0, 3000 - n_frames))
    with torch.inference_mode():
        emb = model(mel)
    return F.normalize(emb.squeeze(0).cpu().float(), dim=-1)


class Transcriber:
    """
    Batched ASR transcriber with continuous batching and optional word-level alignment.

    Architecture:
        - Encoder-decoder transformer (Whisper-style) with Flash Attention KV caches
        - Continuous batching: slots are refilled as they complete, maximizing GPU util
        - Long audio: auto-chunked into max_audio_length windows, stitched with [CHOP] markers

    Usage modes:
        1. Sync:  transcriber.transcribe(audio) -> str
        2. Batch: transcriber(chunk_dict) -> list[dict] | None  (for whisperdrz.pipe pipelines)
        3. Pipe:  use as worker with load() method called after process spawn

    Flow:
        pending_chunks -> _init_slot() -> _prefill_slots() (encoder + first token)
        -> decode loop -> status="process"/"finished" -> _batch_align_slots()
        -> cleaned output with speaker tags, timestamps, non-speech events

    Slot statuses:
        - "ready":    empty, available for new chunk
        - "started":  initialized and decoding
        - "process":  hit max_tokens or timing boundary, needs alignment + may continue
        - "finished": done decoding, needs final alignment

    Output format (see CLAUDE.md for full spec):
        [0] <|0.00|>Hello<|0.45|> <|0.50|>world.<|0.90|>
        [1] <|1.00|>Hi<|1.20|> [laugh] <|1.40|>there.<|1.70|>
    """

    def __init__(
        self,
        model_path,
        BS=52,
        max_audio_length=30,
        max_tokens=448,
        lang=None,
        model=None,
        align_override_score=40,
        dtw_align: bool = True,
        temperature=0.0,
        top_k=None,
        top_p=None,
        force_timing_every_word=False,
        skip_non_english=False,
        precomputed_mels=False,
        allow_tags=False,
        allow_asterisks=False,
        speaker_tags: bool | None = None,
        speaker_remap: bool = True,
        speaker_remap_model: str = "ecapa2",
        speaker_remap_threshold: float = 1.1,
        n_warm: int = 0,
    ):
        """
        Args:
            model_path: Local path to a checkpoint, a filename in the HF weights repo, or an HF repo id
            BS: Batch size - max concurrent slots. Higher = more throughput, more VRAM
            max_audio_length: Window size in seconds (30s = Whisper default)
            max_tokens: Max decoder tokens per window before forcing alignment
            lang: Force language (e.g. "en"). None = auto-detect from first token
            model: Pre-loaded WhisperDRZ model instance (skips loading from model_path)
            align_override_score: Threshold to prefer DTW timing over decoder timing
            temperature: Sampling temperature. 0 = greedy
            force_timing_every_word: Emit timestamp for every word (vs decoder's natural timing)
            skip_non_english: Immediately return empty for non-English detected audio
            precomputed_mels: Expect chunk["mel"] instead of chunk["audio"]
            allow_tags: Allow <tag> tokens (e.g. <emotion:happy>, <accent:british>)
            allow_asterisks: Allow * tokens (e.g. *emphasis*)
        """
        from whisperdrz.inference_utils import InferenceState, init_for_model
        from whisperdrz.model import WhisperDRZ

        self.model_path = model_path
        # BS = number of slots actively decoding (CUDA graph bucket size).
        # n_warm = extra slots holding prefilled KV in the background so prefill can
        # overlap with the decode loop on a side CUDA stream. Total slot capacity
        # is BS + n_warm. Default n_warm=0 (dormant) — only pays off on GPUs with
        # enough VRAM to keep ~BS extra slots prefilled (H100/A100 with large-v3).
        # For measurable benefit set n_warm close to BS.
        self.BS = BS
        self.n_warm = n_warm
        self.n_slots = BS + n_warm
        self.max_audio_length = max_audio_length
        self.max_tokens = max_tokens
        self.align_override_score = align_override_score
        self.dtw_align = dtw_align
        self._dtw_head_mask = None  # lazy: built when model is on device
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.force_timing_every_word = force_timing_every_word
        self.skip_non_english = skip_non_english
        self.precomputed_mels = precomputed_mels
        self.allow_tags = allow_tags
        self.allow_asterisks = allow_asterisks
        self.speaker_remap = speaker_remap
        self.speaker_remap_model_name = speaker_remap_model
        self.speaker_remap_threshold = speaker_remap_threshold
        self.speaker_remap_model = None
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = WhisperDRZ.from_pretrained_inf(model_path)
        else:
            raise ValueError("Either model_path or model must be provided")
        self.detect_language = lang is None
        self.lang = lang

        if speaker_tags is None:
            speaker_tags = True
        self.speaker_tags = speaker_tags

        # Initialize inference_utils tokenizer to match model's vocab size
        init_for_model(self.model)

        self.init_tokens = self._build_init_tokens(lang)
        self.init_tensor = torch.tensor(self.init_tokens)
        self.init_len = len(self.init_tokens)

        self.states = [
            InferenceState(
                index=i,
                tokens=self.init_tensor.clone(),
                initial_tokens=self.init_tensor.clone(),
                language=lang,
                force_timing_every_word=force_timing_every_word,
            )
            for i in range(self.n_slots)
        ]
        self.pending_chunks = []
        self.batch_start_time = None
        self.loaded = False
        self.prefill_stream = None  # set in load_model
        # slot_idx -> (event, slot_idx) for slots whose prefill is in flight on
        # prefill_stream. Main stream waits on the event before using the KV.
        self.prefill_events: dict[int, torch.cuda.Event] = {}

    def _build_init_tokens(self, lang: str | None) -> list[int]:
        tok = self.model.tokenizer
        if lang is None:
            # Language detection: just sot, model predicts language
            return [tok.sot]
        lang_token = tok.to_language_token(lang)
        if self.speaker_tags:
            return [
                tok.sot,
                lang_token,
                tok.transcribe,
                tok.encode("[")[0],
                tok.encode("0")[0],
                tok.encode("]")[0],
                tok.encode(" ")[0],
            ]
        return [tok.sot, lang_token, tok.transcribe]

    def load_model(self, device=None):
        """Load the model for inference (called automatically on first use).

        Picks CUDA when available, otherwise CPU. CUDA graphs and the flash
        attention kernel are only used on CUDA with flash-attn installed; on
        other devices a pure-torch SDPA path is used (see whisperdrz.model).
        """
        if self.loaded:
            return

        from whisperdrz.model import HAS_FLASH_ATTN

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_cuda = device != "cpu" and torch.cuda.is_available()

        if self.use_cuda:
            if not str(self.model.device).startswith("cuda"):
                self.model = self.model.cuda()
        else:
            # CPU SDPA does not support fp16; run in fp32.
            self.model = self.model.cpu().float()

        self.device = self.model.device
        self.dtype = self.model.dtype
        # CPU mel computation (per-chunk, matching training normalization)
        self.tomel = LogMelSpectrogram(n_mels=self.model.dims.n_mels)

        # Flash attention KV caches — size for n_slots (decode BS + warm pool).
        # CUDA graphs are captured with these KV tensor references; slot indices
        # ≥ BS are valid because KV has room for them.
        self.max_seq_len = self.max_tokens
        self.model.decoder.cache_embedding_weight(self.dtype)
        self.self_kv, self.cross_kv = self.model.create_flash_kv(
            self.n_slots, max_tokens=self.max_seq_len
        )

        # GPU-optimized decoder state
        self.ds = DecoderState(self.model.tokenizer, device=self.device)
        self.ds.init_gpu_tensors()

        # Pre-allocated tensors for continuous decoding — sized for n_slots
        self.all_tokens = torch.zeros(
            self.n_slots, self.max_seq_len, dtype=torch.long, device=self.device
        )
        self.seq_lens = torch.zeros(self.n_slots, dtype=torch.long, device=self.device)
        self.batch_idx = torch.arange(self.n_slots, device=self.device)

        # Cache for encoder outputs per slot
        self.xa_cache = torch.zeros(
            self.n_slots,
            1500,
            self.model.dims.n_audio_state,
            dtype=self.dtype,
            device=self.device,
        )

        # Pre-cached slot tensors to avoid per-step allocations
        self.slot_tensor_cache = [
            torch.tensor([i], device=self.device, dtype=torch.long)
            for i in range(self.n_slots)
        ]

        # Pre-allocated index tensor for batch operations
        self._slot_indices_buffer = torch.arange(
            self.n_slots, device=self.device, dtype=torch.long
        )

        # Dedicated stream for encoder + prefill — overlaps with decoder default stream.
        self.prefill_stream = torch.cuda.Stream(self.device) if self.use_cuda else None

        self.seconds_processed = 0.0
        self._max_tokens_reached = 0

        # Initialize CUDA graphs for decode loop
        self.cuda_graphs = BucketedCUDAGraphs(
            self.model,
            self.self_kv,
            self.cross_kv,
            return_hidden_states=False,
        )
        # CUDA graphs only help (and only work) on CUDA with flash attention.
        # Otherwise the decode loop falls back to a direct forward_flash call.
        if self.use_cuda and HAS_FLASH_ATTN:
            self.cuda_graphs.capture_all(max_bucket=self.BS)

        # Load speaker model for cross-chunk speaker matching
        if self.speaker_remap:
            try:
                if self.speaker_remap_model_name == "ecapa2":
                    from huggingface_hub import hf_hub_download

                    try:
                        model_file = hf_hub_download(
                            repo_id="Jenthe/ECAPA2", filename="ecapa2.pt"
                        )
                    except Exception:
                        model_file = hf_hub_download(
                            repo_id="Jenthe/ECAPA2",
                            filename="ecapa2.pt",
                            local_files_only=True,
                        )
                    self.speaker_remap_model = torch.jit.load(
                        model_file, map_location=self.device
                    )
                    self.speaker_remap_model.eval()
                    if DEBUG:
                        print("Loaded ECAPA2 for speaker remapping")
                else:
                    print(
                        f"WARNING: Unknown speaker_remap_model: {self.speaker_remap_model_name}"
                    )
            except Exception as e:
                print(f"WARNING: Failed to load speaker remap model: {e}")
                self.speaker_remap_model = None

        self.loaded = True


    def transcribe(self, audio: "torch.Tensor | np.ndarray") -> TranscribeResults:
        """
        Transcribe audio synchronously, handling chunking internally.

        If model has speaker_head, speaker embeddings are clustered and
        speaker IDs in the output are remapped to consistent global IDs.

        Args:
            audio: Audio tensor or numpy array at 16kHz, shape (samples,) or (1, samples)

        Returns:
            TranscribeResults with text, speaker_embeddings, and raw_chunks
        """
        import numpy as np

        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        audio = audio.squeeze()

        chunk_len = int(self.max_audio_length * 16000)

        # Split into chunks
        chunks = []
        for i, start in enumerate(range(0, len(audio), chunk_len)):
            end = min(start + chunk_len, len(audio))
            chunks.append(
                {
                    "audio": audio[start:end],
                    "id": f"chunk_{i}",
                    "duration": (end - start) / 16000,
                    "offset": start / 16000,
                }
            )

        # Process all chunks
        all_results = []
        self.pending_chunks.extend(chunks)
        while self.pending_chunks or any(s.status != "ready" for s in self.states):
            results = self._run_batched_inference()
            if results:
                all_results.extend(results)

        # Sort by chunk ID to ensure correct order (continuous batching returns out-of-order)
        all_results.sort(key=lambda r: int(r.get("id", "chunk_0").split("_")[1]))

        # Combine all chunk predictions (with timestamp offsets already applied)
        combined = "\n".join(
            r.get("prediction", "").replace("[CHOP]", "\n").rstrip()
            for r in all_results
        )

        # Per-line speaker remapping using external embeddings
        if (
            self.speaker_remap_model is not None
            and self.speaker_remap_model_name == "ecapa2"
        ):
            combined = remap_speakers_per_line(
                combined,
                audio,
                self.speaker_remap_model,
                threshold=self.speaker_remap_threshold,
            )

        merged = merge_consecutive_speakers(combined)
        merged = fix_nonmonotonic_timestamps(merged)
        text = normalize_speaker_ids(merged)
        return TranscribeResults(
            text=text,
            raw_chunks=all_results,
            segments=parse_to_segments(text),
        )

    def __call__(self, chunk):
        """
        Process chunk(s) through the transcriber (for pipeline usage).

        Args:
            chunk: Single dict or list of dicts with:
                - id: unique identifier (returned in results)
                - audio: torch.Tensor (samples,) at 16kHz, OR
                - mel: pre-computed mel spectrogram if precomputed_mels=True
                - duration: audio length in seconds
                - offset: (optional) timestamp offset to add to all output timings
                - ...any extra keys are passed through to results

        Returns:
            list[dict] with 'id', 'prediction', 'transcription', 'stop_reason', etc.
            None if batch not ready yet (buffering for efficiency)

        Note: Call flush() after final chunk to drain remaining buffered items.
        """
        # Drain buffer if getting too large (backpressure)
        max_pending = self.BS * 2
        results = []
        while len(self.pending_chunks) >= max_pending:
            batch_results = self._run_batched_inference()
            if batch_results:
                results.extend(batch_results)

        # Handle both single chunk and list of chunks
        # Filter out "end" signals - they're not real chunks
        if isinstance(chunk, list):
            self.pending_chunks.extend(c for c in chunk if isinstance(c, dict))
        elif isinstance(chunk, dict):
            self.pending_chunks.append(chunk)

        batch_results = self._try_process_batch()
        if batch_results:
            results.extend(batch_results)
        return results if results else None

    def _try_process_batch(self):
        """
        Try to process pending chunks with continuous batching.
        Slot initialization is handled by _run_batched_inference.

        Returns:
            List of results or None if conditions not met
        """
        import time

        has_pending = len(self.pending_chunks) > 0
        ready_count = sum(1 for s in self.states if s.status == "ready")

        # Track when we first had work to do
        if has_pending and self.batch_start_time is None:
            self.batch_start_time = time.time()

        # Process when we have enough pending to fill available slots
        if has_pending and len(self.pending_chunks) >= ready_count:
            self.batch_start_time = None
            return self._run_batched_inference()

        # Process when we have pending work and have waited 1 second
        if has_pending:
            elapsed = time.time() - self.batch_start_time
            if elapsed >= 1.0:
                self.batch_start_time = None
                return self._run_batched_inference()

        return None

    def _init_slot(self, slot_idx: int, chunk: dict):
        """Initialize a slot with a new chunk for processing."""
        s = self.states[slot_idx]
        s.reset()
        s._continuation_count = 0  # Reset continuation counter for new chunk
        s.id = chunk["id"]
        audio = chunk.get("audio")
        if audio is not None and not isinstance(audio, torch.Tensor):
            import numpy as np

            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio)
        s.audio = audio
        s.mel = chunk.get("mel")  # Pre-computed mel spectrogram (optional)

        # Compute mel once for full audio if not provided (avoids recompute on continuations)
        if s.mel is None and s.audio is not None:
            with torch.no_grad():
                # tomel expects CPU, so move audio to CPU first, compute, then move result to GPU
                audio_cpu = s.audio.cpu() if s.audio.is_cuda else s.audio
                s.mel = (
                    self.tomel(audio_cpu.unsqueeze(0)).squeeze(0).to(self.device)
                )  # (n_mels, frames)

        s.status = "started"
        s._timing_count = 0
        s._line_cache = ""
        s.last_timing = None
        s._window_max_timing = 0.0
        s.forced_timing_before_newline = False
        s.predict_next = None

        # Check if this chunk covers the rest of the audio
        remaining_audio = s.duration - s.offset if s.duration else 0
        s.is_final_chunk = remaining_audio <= self.max_audio_length

        # Store metadata
        for k in chunk:
            if k not in ["id", "audio", "mel"]:
                s.meta[k] = chunk[k]

        # Per-chunk lang override: build init_tokens + set state.language.
        # Falls back to the Transcriber's global lang when chunk doesn't specify one.
        chunk_lang = chunk.get("lang")
        if chunk_lang is not None:
            slot_init_tokens = self._build_init_tokens(chunk_lang)
            slot_init_tensor = torch.tensor(slot_init_tokens, device=self.device)
            slot_init_len = len(slot_init_tokens)
            s.language = chunk_lang
            effective_lang = chunk_lang
        else:
            slot_init_tokens = self.init_tokens
            slot_init_tensor = self.init_tensor.to(self.device)
            slot_init_len = self.init_len
            effective_lang = self.lang

        # Initialize token tensors
        self.all_tokens[slot_idx, :slot_init_len] = slot_init_tensor
        self.all_tokens[slot_idx, slot_init_len:] = 0
        self.seq_lens[slot_idx] = slot_init_len

        # CPU-side tracking (avoids GPU→CPU copies in decode loop)
        s.token_count = slot_init_len
        s.last_token_id = slot_init_tokens[-1]

        s._line_count = 0

        if effective_lang is not None:
            if self.speaker_tags:
                s.cur = "[0] "
            else:
                s.cur = ""

    def _kick_off_async_prefill(self, slots: list[int]):
        """Start prefill for slots on prefill_stream, record event, mark warming.

        Caller must wait on the event (via `_promote_warm_slots`) before these
        slots' KV is used by the main-stream decoder.
        """
        if not slots:
            return
        for s in slots:
            self.states[s].status = "warming"
        with torch.cuda.stream(self.prefill_stream):
            self._prefill_slots(slots)
        event = torch.cuda.Event()
        event.record(self.prefill_stream)
        for s in slots:
            self.states[
                s
            ].status = "warming"  # _prefill_slots doesn't set, but be explicit
            self.prefill_events[s] = event

    def _promote_warm_slots(self) -> list[int]:
        """Poll warming slots; promote any whose prefill event has completed.

        Returns list of promoted slot indices. Main stream syncs on the event
        (once per event) so subsequent CUDA graph replays see the prefilled KV.
        """
        if not self.prefill_events:
            return []
        promoted = []
        synced_events: set[int] = set()
        for slot_idx, event in list(self.prefill_events.items()):
            if event.query():
                if id(event) not in synced_events:
                    torch.cuda.current_stream().wait_event(event)
                    synced_events.add(id(event))
                self.states[slot_idx].status = "started"
                del self.prefill_events[slot_idx]
                promoted.append(slot_idx)
        return promoted

    def _fill_warm_slots(self):
        """Assign pending_chunks to empty slots and start async prefill.

        Keeps up to n_warm slots in 'warming' state. Called throughout the
        decode loop to keep the pipeline primed. No-op when n_warm=0.
        """
        if self.n_warm <= 0 or not self.pending_chunks:
            return
        # Count current warming; fill until we hit n_warm.
        current_warming = sum(1 for s in self.states if s.status == "warming")
        room = self.n_warm - current_warming
        if room <= 0:
            return
        new_slots = []
        for slot_idx in range(self.n_slots):
            if room <= 0 or not self.pending_chunks:
                break
            if self.states[slot_idx].status != "ready":
                continue
            chunk = self.pending_chunks.pop(0)
            self._init_slot(slot_idx, chunk)
            new_slots.append(slot_idx)
            room -= 1
        if new_slots:
            self._kick_off_async_prefill(new_slots)

    def _prefill_slots(self, slots: list[int]):
        """Run encoder and prefill for slots. Returns first logits."""
        if not slots:
            return

        slot_tensor = torch.tensor(slots, device=self.device, dtype=torch.long)
        target_samples = int(self.max_audio_length * 16000)
        target_frames = 3000

        # Reset KV for these slots
        self.model.reset_flash_kv(self.self_kv, self.cross_kv, slot_tensor)

        # Check if we have pre-computed mels
        has_mels = all(self.states[slot_idx].mel is not None for slot_idx in slots)

        if has_mels:
            # Use pre-computed mels directly, sliced by offset
            mel_batch = []
            for slot_idx in slots:
                s = self.states[slot_idx]
                s.last_offset = s.offset
                mel = s.mel.to(self.device)
                # Convert offset (seconds) to mel frames (100 fps = 16000/160 hop)
                offset_frames = int(s.offset * 100)
                mel_slice = mel[:, offset_frames : offset_frames + target_frames]
                # Pad mel to target frames if needed
                if mel_slice.shape[-1] < target_frames:
                    mel_slice = torch.nn.functional.pad(
                        mel_slice, (0, target_frames - mel_slice.shape[-1]), value=-1.5
                    )
                mel_batch.append(mel_slice[:, :target_frames])
            xa = torch.stack(mel_batch, dim=0).to(self.dtype)
        else:
            # Prepare audio batch and compute mels on CPU (per-chunk normalization)
            padded_batch = []
            for slot_idx in slots:
                s = self.states[slot_idx]
                s.last_offset = s.offset

                start_sample = int(s.offset * 16000)
                end_sample = start_sample + target_samples
                audio_chunk = s.audio[start_sample:end_sample]  # Keep on CPU for mel

                if len(audio_chunk) < target_samples:
                    audio_chunk = torch.nn.functional.pad(
                        audio_chunk, (0, target_samples - len(audio_chunk))
                    )
                padded_batch.append(audio_chunk[:target_samples])

            xa = torch.stack(padded_batch, dim=0)
            xa = self.tomel(xa).to(self.device, self.dtype)  # Compute on CPU, then move

        with flash_kernel():
            xa = self.model.encoder(xa)
        self.xa_cache[slot_tensor] = xa

        # Prefill tokens (read directly from GPU buffer, no CPU→GPU copy)
        slot_states = [self.states[i] for i in slots]
        tokens_batch = self.all_tokens[slot_tensor, : self.init_len]
        logits = self.model.prefill_flash(
            tokens_batch,
            xa,
            self.self_kv,
            self.cross_kv,
            slot_tensor,
            return_hidden_states=False,
        )
        first_logits = logits[:, -1, :]

        # Decode first token (pass Python lists, no GPU→CPU copies)
        last_token_ids = [s.last_token_id for s in slot_states]
        token_counts_list = [s.token_count for s in slot_states]
        results = decode_batch(
            first_logits,
            slot_states,
            self.ds,
            temperature=self.temperature,
            last_token_ids=last_token_ids,
            token_counts=token_counts_list,
            allow_tags=self.allow_tags,
            allow_asterisks=self.allow_asterisks,
            speaker_tags=self.speaker_tags,
            all_tokens_gpu=self.all_tokens,
            slot_indices=slots,
        )

        _write_slots = []
        _write_positions = []
        _write_tokens = []
        for s, (newtext, newtoken_id) in zip(slot_states, results, strict=False):
            update_state_after_decode(s, newtext, newtoken_id)
            new_len = s.token_count
            if new_len <= self.max_seq_len:
                _write_slots.append(s.index)
                _write_positions.append(new_len - 1)
                _write_tokens.append(newtoken_id)
        if _write_slots:
            _slots_t = torch.tensor(_write_slots, dtype=torch.long, device=self.device)
            _pos_t = torch.tensor(
                _write_positions, dtype=torch.long, device=self.device
            )
            _vals_t = torch.tensor(_write_tokens, dtype=torch.long, device=self.device)
            self.all_tokens[_slots_t, _pos_t] = _vals_t
            self.seq_lens[_slots_t] = _pos_t + 1

    def _prepare_slot_for_alignment(self, s) -> dict | None:
        """Prepare a finished slot for batched alignment. Returns alignment item or None."""
        # Only return empty for non-English when skip_non_english is explicitly set.
        # By default we let alignment + post-processing run on the model's prediction
        # in whatever language it detected.
        if (
            self.skip_non_english
            and self.detect_language
            and hasattr(s, "language")
            and s.language
            and s.token_count >= 2
        ):
            if s.language != "en":
                ret = s.meta.copy()
                ret["id"] = s.id
                ret["prediction"] = ""
                ret["language"] = s.language
                ret["stop_reason"] = f"non_english:{s.language}"
                s.status = "ready"
                return {"skip": True, "result": ret, "state": s}

        if s.status == "process":
            last_timing = s.last_timing or 0
            stop_reason = s.stop_reason or ""

            # If EOT predicted with no timing, don't try to continue - model has nothing more to say
            if (
                "eot_predicted" in stop_reason
                and s.last_timing is None
                or "repetition" in stop_reason
                and ("[pause" in stop_reason or "[breath" in stop_reason)
            ):
                more = False
            # Other repetition (speech loops) - skip ahead 1s and try to continue
            elif "repetition" in stop_reason:
                last_timing = min(
                    last_timing + 1.0,
                    s.duration - s.offset if s.duration else last_timing + 1.0,
                )
                more = s.duration - (s.offset + last_timing) > 0.5
            else:
                # Continue if there's >0.5s of audio remaining after last timing
                more = s.duration - (s.offset + last_timing) > 0.5

            return {
                "state": s,
                "cur": s.cur,
                "clip_to_last_timing": more,
                "more": more,
                "phase": "process",
                "last_timing_override": last_timing
                if "repetition" in stop_reason
                else None,
            }

        if s.status == "finished":
            return {
                "state": s,
                "cur": s.cur,
                "clip_to_last_timing": False,
                "more": False,
                "phase": "finished",
            }

        return None

    def _dtw_fill_cur(self, s, cur: str) -> str:
        """Fill in `<|t|>word<|t|>` for words the decoder didn't natively time,
        via DTW on cross-attention. Reuses the encoder output already in
        `self.xa_cache[s.index]` — no encoder re-run.
        """
        if not self.dtw_align or not cur.strip():
            return cur
        from whisperdrz.audio import FRAMES_PER_SECOND, N_FRAMES
        from whisperdrz.dtw_align import fill_word_timings, make_head_mask

        if self._dtw_head_mask is None:
            self._dtw_head_mask = make_head_mask(self.model).to(self.device)

        # Each finalisation corresponds to one decode window of length
        # max_audio_length. The window's audio actually present is duration -
        # last_offset (could be < max_audio_length on the trailing window).
        if s.duration is not None:
            win_dur = min(self.max_audio_length, max(0.0, s.duration - s.last_offset))
        else:
            win_dur = float(self.max_audio_length)
        num_frames = min(int(win_dur * FRAMES_PER_SECOND), N_FRAMES)
        if num_frames < 50:  # ~0.5 s — too short to align reliably
            return cur

        xa = self.xa_cache[s.index]
        try:
            return fill_word_timings(
                self.model,
                self.model.tokenizer,
                xa,
                cur,
                head_mask=self._dtw_head_mask,
                num_frames=num_frames,
            )
        except Exception as e:
            if DEBUG:
                print(f"dtw_fill failed for slot {s.index}: {type(e).__name__}: {e}")
            return cur

    def _finalize_slot_after_alignment(
        self,
        s,
        out: str,
        last_timing: float,
        keepspeaker: bool,
        align_scores,
        more: bool,
        phase: str,
        pre_augmented_cur: str | None = None,
    ) -> dict | None:
        """Finalize a slot after alignment. Returns result dict or None if continuing."""
        from whisperdrz.inference_utils import finalize_transcription

        n_tokens = s.token_count
        if align_scores is not None:
            s.meta["alignment_scores"] = align_scores

        if phase == "process":
            if pre_augmented_cur is not None:
                cur = pre_augmented_cur
            else:
                # DTW must see window-relative timestamps so its window-relative
                # xa output matches. Apply s.offset after DTW fills.
                cur = finalize_transcription(s, out, apply_offset=False)
                cur = self._dtw_fill_cur(s, cur)
                if s.offset > 0:
                    from whisperdrz.clean import offset_timestamps

                    cur = offset_timestamps(cur, s.offset)
            # Ensure minimum progress to prevent infinite loops (at least 1s advance)
            min_advance = 1.0
            actual_advance = max(last_timing, min_advance) if more else last_timing
            s.offset += actual_advance
            # Strip trailing content after last closing timestamp when continuing
            # Next window re-covers this audio, preventing duplicate timestamps at stitch
            if more:
                cur = _strip_to_last_closing(cur)
            # Ensure speaker tags always start on a new line
            content_so_far = s.sofar.replace("[CHOP]", "").rstrip()
            if not content_so_far or content_so_far.endswith("\n"):
                sep = ""
            elif cur.startswith("["):
                sep = "\n"
            else:
                sep = " "
            s.sofar += sep + cur + "[CHOP]"
            s.tokens = s.initial_tokens.clone()
            s.token_count = self.init_len
            s.last_token_id = self.init_tokens[-1]
            # Sync cur with init tokens
            if self.lang is not None and self.speaker_tags:
                s.cur = "[0] "
            else:
                s.cur = ""
            # Reset cutearly to False - at continuation we need the fresh [0] speaker tag
            # (cutearly strips speaker tag at start, which is wrong for new windows)
            s.cutearly = False
            s.last_timing = None
            s._window_max_timing = 0.0

            # Track continuation attempts to prevent infinite loops
            if not hasattr(s, "_continuation_count"):
                s._continuation_count = 0
            s._continuation_count += 1
            max_continuations = 20  # Max continuations per original audio chunk

            if more and s._continuation_count < max_continuations:
                s.status = "started"
                # Reset token tensors for continuation - critical for EOT blocking
                slot_idx = s.index
                self.all_tokens[slot_idx, : self.init_len] = self.init_tensor.to(
                    self.device
                )
                self.all_tokens[slot_idx, self.init_len :] = 0
                self.seq_lens[slot_idx] = self.init_len
                remaining_audio = s.duration - s.offset if s.duration else 0
                s.is_final_chunk = remaining_audio <= self.max_audio_length
                return None  # Continue processing
            if more:
                # Hit max continuations - force finish
                s.stop_reason = f"max_continuations:{s._continuation_count}"
            s.status = "finished"
            if not s.stop_reason:
                s.stop_reason = "audio_exhausted"
            # Fall through to finished processing

        if s.status == "finished":
            if phase == "finished":
                if pre_augmented_cur is not None:
                    cur = pre_augmented_cur
                else:
                    cur = finalize_transcription(s, out, apply_offset=False)
                    cur = self._dtw_fill_cur(s, cur)
                    if s.offset > 0:
                        from whisperdrz.clean import offset_timestamps

                        cur = offset_timestamps(cur, s.offset)
                s.sofar += cur + "[CHOP]"

            # Clean up transcription
            cleaned = s.sofar
            cleaned = reconcile_decoder_and_alignment_timings(
                cleaned, confidence_threshold=self.align_override_score
            )
            cleaned = consolidate_adjacent_tags(cleaned)
            cleaned = remove_invalid_timing_pairs(cleaned)
            cleaned = remove_speaker_tags_not_after_newline(cleaned)
            cleaned = merge_consecutive_speakers(cleaned)
            cleaned = fix_nonmonotonic_timestamps(cleaned)
            cleaned = re.sub(r'\s*\["\s*(?=\n|$)', "", cleaned)
            cleaned = re.sub(r"\s+[01]\s*(?=\n|$)", "", cleaned)
            cleaned = re.sub(r"(?<=\n)]|^]", "", cleaned)
            cleaned = re.sub(r"\[(?=\n|$)", "", cleaned)
            cleaned = re.sub(r"(?<![a-zA-Z])!(?![a-zA-Z])", "", cleaned)
            cleaned = re.sub(
                r"\bCouncill(?:<\|[\d.]+\|>)?", "", cleaned
            )  # Remove "Councill" hallucination
            cleaned = remove_orphan_paired_tags(cleaned)
            cleaned = re.sub(
                r"^\[(?:\d+|c)\]\s*(?:<\|[\d.]+\|>\s*)*(?:\[[a-zA-Z]+\]\s*(?:<\|[\d.]+\|>\s*)*)+\s*$",
                "",
                cleaned,
                flags=re.MULTILINE,
            )
            cleaned = re.sub(r"\n{2,}", "\n", cleaned)
            cleaned = remove_short_nonspeech(cleaned, min_duration_s=0.02)
            cleaned = remove_short_pauses(cleaned, min_pause_ms=800)
            # Remove orphaned timestamps: [0] <|t|> <|t2|> -> [0] <|t2|>
            cleaned = re.sub(r"(\[\d+\] )<\|[\d.]+\|> (<\|[\d.]+\|>)", r"\1\2", cleaned)

            # Normalize speaker IDs so first speaker is [0], second is [1], etc.
            # Store reverse mapping so speaker clustering can map cleaned IDs back
            speaker_order = []
            for m in re.finditer(r"\[(\d+)\]", cleaned):
                sid = int(m.group(1))
                if sid not in speaker_order:
                    speaker_order.append(sid)
            cleaned_to_original = dict(enumerate(speaker_order))
            cleaned = fix_speaker_ids(cleaned)

            # Apply chunk offset to timestamps if provided
            chunk_offset = s.meta.get("offset", 0)
            if chunk_offset > 0:
                from whisperdrz.clean import offset_timestamps

                cleaned = offset_timestamps(cleaned, chunk_offset)

            s.meta["prediction"] = cleaned
            s.meta["raw_cur"] = s.sofar  # Raw output before cleaning
            s.meta["language"] = getattr(s, "language", "unknown")
            if not s.stop_reason:
                s.stop_reason = "completed"
            s.meta["stop_reason"] = s.stop_reason
            if n_tokens > self._max_tokens_reached:
                self._max_tokens_reached = n_tokens
            ret = s.meta.copy()
            ret["audio"] = s.audio
            ret["id"] = s.id
            ret["transcription"] = cleaned
            ret["stop_reason"] = s.stop_reason
            ret["speaker_id_map"] = cleaned_to_original
            s.status = "ready"
            return ret

        return None

    def _batch_align_slots(self, pending_align: list[dict]) -> list[dict | None]:
        """Run batched alignment on collected slots."""
        from whisperdrz.inference_utils import TIMINGS_FORMAT

        if not pending_align:
            return []

        # Separate skipped items from items needing alignment
        results = [None] * len(pending_align)
        to_align = []
        to_align_indices = []

        for i, item in enumerate(pending_align):
            if item.get("skip"):
                results[i] = item["result"]
            else:
                to_align.append(item)
                to_align_indices.append(i)

        if not to_align:
            return results

        from whisperdrz.clean.timings import cut_after_last_timed_word

        # Pre-compute per-slot (post-cut) `out` and `last_timing`. Anything
        # that depends on the slot's raw decoder output happens here, so
        # the downstream batched DTW operates on the right text.
        per_slot_out: list[str] = []
        per_slot_last_timing: list[float] = []
        for item in to_align:
            s = item["state"]
            out = item["cur"]
            if item.get("clip_to_last_timing"):
                out = cut_after_last_timed_word(out)
                timing_matches = list(TIMINGS_FORMAT.finditer(out))
                if timing_matches:
                    last_timing = float(timing_matches[-1].group(1))
                else:
                    last_timing = (
                        item.get("last_timing_override") or s.last_timing or 0
                    )
            else:
                last_timing = item.get("last_timing_override") or s.last_timing or 0
            per_slot_out.append(out)
            per_slot_last_timing.append(last_timing)

        # Batched DTW: ONE padded decoder forward across all finalising slots.
        pre_augs: list[str | None] = [None] * len(to_align)
        if self.dtw_align:
            from whisperdrz.audio import FRAMES_PER_SECOND, N_FRAMES
            from whisperdrz.dtw_align import fill_word_timings_batch, make_head_mask
            from whisperdrz.inference_utils import finalize_transcription

            if self._dtw_head_mask is None:
                self._dtw_head_mask = make_head_mask(self.model).to(self.device)

            slot_data = []
            for item, out in zip(to_align, per_slot_out, strict=False):
                s = item["state"]
                # window-relative pre-DTW (apply s.offset after fill)
                raw_cur = finalize_transcription(s, out, apply_offset=False)
                if s.duration is not None:
                    win_dur = min(
                        self.max_audio_length,
                        max(0.0, s.duration - s.last_offset),
                    )
                else:
                    win_dur = float(self.max_audio_length)
                num_frames = min(int(win_dur * FRAMES_PER_SECOND), N_FRAMES)
                slot_data.append((self.xa_cache[s.index], raw_cur, num_frames))

            try:
                pre_augs = list(
                    fill_word_timings_batch(
                        self.model,
                        self.model.tokenizer,
                        slot_data,
                        head_mask=self._dtw_head_mask,
                    )
                )
                from whisperdrz.clean import offset_timestamps

                for i, item in enumerate(to_align):
                    s = item["state"]
                    if pre_augs[i] is not None and s.offset > 0:
                        pre_augs[i] = offset_timestamps(pre_augs[i], s.offset)
            except Exception as e:
                if DEBUG:
                    print(
                        f"batched dtw failed, falling back per-slot: "
                        f"{type(e).__name__}: {e}"
                    )
                pre_augs = [None] * len(to_align)

        for idx, item in enumerate(to_align):
            s = item["state"]
            orig_idx = to_align_indices[idx]
            out = per_slot_out[idx]
            last_timing = per_slot_last_timing[idx]
            keepspeaker = s.keepspeaker if hasattr(s, "keepspeaker") else True
            result = self._finalize_slot_after_alignment(
                s,
                out,
                last_timing,
                keepspeaker,
                None,
                item["more"],
                item["phase"],
                pre_augmented_cur=pre_augs[idx],
            )
            results[orig_idx] = result
        return results

    @torch.inference_mode()
    def _run_batched_inference(self):
        """Run batched inference with true continuous batching - refills slots as they complete."""

        try:
            self.load_model()
            if DEBUG:
                slot_statuses = [s.status for s in self.states]
                print(
                    f"[CB] Starting inference, pending={len(self.pending_chunks)}, slot_statuses={slot_statuses}"
                )

            # Track slots in Python to avoid GPU syncs
            empty_slots = {i for i, s in enumerate(self.states) if s.status == "ready"}
            # Include "process" status - slots may enter this function already at "process"
            # from a previous call that set status but didn't complete alignment in same call
            active_slots = [
                i
                for i, s in enumerate(self.states)
                if s.status in ("started", "process")
            ]
            completed_results = []

            # Fill initial empty slots from pending
            new_slots = []
            for slot_idx in list(empty_slots):
                if not self.pending_chunks:
                    break
                chunk = self.pending_chunks.pop(0)
                self._init_slot(slot_idx, chunk)
                new_slots.append(slot_idx)
                empty_slots.discard(slot_idx)
                active_slots.append(slot_idx)

            if new_slots:
                self._prefill_slots(new_slots)
                if DEBUG:
                    total_dur = sum(
                        self.states[i].meta.get("duration", 0) for i in new_slots
                    )
                    print(
                        f"[Transcriber] Loading batch: {len(new_slots)} chunks, {total_dur:.1f}s audio, {len(self.pending_chunks)} pending"
                    )
            # Start warm-slot prefill in background so refills later are instant.
            self._fill_warm_slots()
            if DEBUG:
                print(f"[CB] Filled {len(new_slots)} slots, active={len(active_slots)}")

            if not active_slots:
                return None

            # Main decode loop with continuous batching
            for step in range(self.max_seq_len * 2):  # Allow extra steps for refills
                if DEBUG and step % 50 == 0:
                    print(
                        f"[CB] Step {step}, active={len(active_slots)}, pending={len(self.pending_chunks)}"
                    )

                # Promote any warm slots whose prefill finished so we have more
                # capacity immediately, before considering sync refill.
                promoted = self._promote_warm_slots()
                if promoted:
                    active_slots.extend(promoted)

                # Get currently active slots (include "process" since we handle them later)
                active_slots = [
                    i
                    for i in active_slots
                    if self.states[i].status in ("started", "process")
                ]

                # Keep warm pool topped up while decode runs (prefill_stream in bg).
                self._fill_warm_slots()

                # Fill empty slots from pending at start of each iteration.
                # First promote any more warm that became ready, then sync-refill if
                # still nothing active and pending chunks remain (fallback).
                if not active_slots and self.pending_chunks and not self.prefill_events:
                    if DEBUG:
                        print("[CB] Refilling empty slots from pending (sync)")
                    new_slots = []
                    empty_now = [
                        i
                        for i in range(self.n_slots)
                        if self.states[i].status == "ready"
                    ]
                    for slot_idx in empty_now:
                        if not self.pending_chunks:
                            break
                        chunk = self.pending_chunks.pop(0)
                        self._init_slot(slot_idx, chunk)
                        new_slots.append(slot_idx)
                        active_slots.append(slot_idx)
                    if new_slots:
                        self._prefill_slots(new_slots)
                        if DEBUG:
                            print(f"[CB] Refilled {len(new_slots)} slots")
                    self._fill_warm_slots()

                # If still no active but we have warming slots, let another iteration
                # run so promotion can pick them up once events complete.
                if (
                    not active_slots
                    and not self.pending_chunks
                    and not self.prefill_events
                ):
                    if DEBUG:
                        print(f"[CB] Done at step {step}")
                    break

                # Safety check: max tokens and excessive generation rate
                for slot_idx in active_slots:
                    s = self.states[slot_idx]
                    if s.status != "started":
                        continue
                    token_count = s.token_count
                    if token_count > self.max_seq_len - 2:
                        s.status = "process"
                        s.stop_reason = f"safety_max_tokens:{token_count}"
                        continue

                    # Early stop if generating way more tokens than audio duration warrants
                    # Normal speech: ~8-12 tokens/second. >20 tokens/sec = likely hallucinating
                    if s.duration and token_count > 20 * s.duration:
                        s.status = "process"
                        s.stop_reason = (
                            f"excessive_tokens:{token_count}>{20*s.duration:.0f}"
                        )

                # Get slots that need decoding (started status only)
                decode_slots = [
                    i for i in active_slots if self.states[i].status == "started"
                ]

                if decode_slots:
                    # Note: torch.tensor() from list is unavoidable here since decode_slots
                    # is dynamically filtered. The real optimization is in hidden states below.
                    active_tensor = torch.tensor(
                        decode_slots, device=self.device, dtype=torch.long
                    )
                    active_states = [self.states[i] for i in decode_slots]

                    # Get positions and last tokens
                    active_seq_lens = self.seq_lens[active_tensor]
                    active_positions = self.self_kv[0].seq_lens[active_tensor]
                    active_tokens = self.all_tokens[
                        active_tensor, active_seq_lens - 1
                    ].unsqueeze(1)

                    # Decode step - always use CUDA graphs (faster than direct forward)
                    result = self.cuda_graphs.forward(
                        active_tokens, active_positions, active_tensor
                    )
                    active_logits = result[:, 0, :]

                    # Sample next tokens (pass Python lists, no GPU→CPU copies)
                    last_token_ids = [s.last_token_id for s in active_states]
                    token_counts_list = [s.token_count for s in active_states]
                    decode_results = decode_batch(
                        active_logits,
                        active_states,
                        self.ds,
                        temperature=self.temperature,
                        last_token_ids=last_token_ids,
                        token_counts=token_counts_list,
                        allow_tags=self.allow_tags,
                        allow_asterisks=self.allow_asterisks,
                        speaker_tags=self.speaker_tags,
                        all_tokens_gpu=self.all_tokens,
                        slot_indices=decode_slots,
                    )

                    # Collect token writes for batched GPU update
                    _write_slots = []
                    _write_positions = []
                    _write_tokens = []
                    for i, (s, (newtext, newtoken_id)) in enumerate(
                        zip(active_states, decode_results, strict=False)
                    ):
                        update_state_after_decode(s, newtext, newtoken_id)
                        new_len = s.token_count
                        if new_len <= self.max_seq_len:
                            _write_slots.append(s.index)
                            _write_positions.append(new_len - 1)
                            _write_tokens.append(newtoken_id)

                        # Early exit for non-English if skip_non_english=True
                        if (
                            self.skip_non_english
                            and self.detect_language
                            and hasattr(s, "language")
                            and s.language
                            and s.language != "en"
                            and s.status not in ("finished", "ready")
                        ):
                            s.status = "finished"
                            s.stop_reason = f"skip_non_english:{s.language}"
                            s.sofar = ""
                            s.cur = ""

                    # Batch write all tokens at once (one GPU op vs N per-state writes)
                    if _write_slots:
                        _slots_t = torch.tensor(
                            _write_slots, dtype=torch.long, device=self.device
                        )
                        _pos_t = torch.tensor(
                            _write_positions, dtype=torch.long, device=self.device
                        )
                        _vals_t = torch.tensor(
                            _write_tokens, dtype=torch.long, device=self.device
                        )
                        self.all_tokens[_slots_t, _pos_t] = _vals_t
                        self.seq_lens[_slots_t] = _pos_t + 1

                # Collect finished slots for batched alignment
                pending_align = []
                finished_slot_indices = []
                for slot_idx in list(active_slots):
                    s = self.states[slot_idx]
                    if s.status in ("process", "finished"):
                        align_item = self._prepare_slot_for_alignment(s)
                        if align_item:
                            pending_align.append(align_item)
                            finished_slot_indices.append(slot_idx)

                # Batch align all finished slots
                if pending_align:
                    if DEBUG:
                        print(f"[CB] Batch aligning {len(pending_align)} slots")
                    align_results = self._batch_align_slots(pending_align)

                    # Process alignment results
                    slots_needing_continue = []
                    for slot_idx, result in zip(
                        finished_slot_indices, align_results, strict=False
                    ):
                        s = self.states[slot_idx]
                        if result:
                            completed_results.append(result)
                            if DEBUG:
                                print(
                                    f"[CB] Slot {slot_idx} completed, total={len(completed_results)}"
                                )

                        if s.status == "ready":
                            # Slot is done - reset KV and mark as empty
                            self.model.reset_flash_kv(
                                self.self_kv,
                                self.cross_kv,
                                self.slot_tensor_cache[slot_idx],
                            )
                            active_slots.remove(slot_idx)
                            empty_slots.add(slot_idx)

                        elif s.status == "started":
                            # Slot continues with more audio - needs re-prefill
                            slots_needing_continue.append(slot_idx)

                    # Batch refill: refill immediately when slots are empty to maximize GPU util
                    min_refill_batch = 1
                    should_refill = self.pending_chunks and (
                        len(empty_slots) >= min_refill_batch or not active_slots
                    )
                    if should_refill:
                        refill_count = min(len(empty_slots), len(self.pending_chunks))
                        refill_slots = list(empty_slots)[:refill_count]
                        for slot_idx in refill_slots:
                            chunk = self.pending_chunks.pop(0)
                            self._init_slot(slot_idx, chunk)
                            empty_slots.discard(slot_idx)
                            active_slots.append(slot_idx)
                        slots_needing_continue.extend(refill_slots)
                        if DEBUG:
                            total_dur = sum(
                                self.states[i].meta.get("duration", 0)
                                for i in refill_slots
                            )
                            print(
                                f"[Transcriber] Refilling: {len(refill_slots)} chunks, {total_dur:.1f}s audio, {len(self.pending_chunks)} pending"
                            )

                    # Prefill slots that need to continue
                    if slots_needing_continue:
                        self._prefill_slots(slots_needing_continue)

            if DEBUG and completed_results:
                total_dur = sum(r.get("duration", 0) for r in completed_results)
                print(
                    f"[Transcriber] Completed: {len(completed_results)} chunks, {total_dur:.1f}s audio"
                )
            return completed_results if completed_results else None

        except Exception as e:
            import traceback

            print(f"TRANSCRIBER ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            raise e

    def flush(self):
        """
        Drain all pending and in-progress chunks, yielding results.

        Must be called after the final chunk to ensure all buffered audio is processed.
        Blocks until all slots are empty.

        Yields:
            dict: Result dicts with 'id', 'prediction', 'transcription', 'stop_reason'
        """
        while self.pending_chunks or any(s.status != "ready" for s in self.states):
            results = self._run_batched_inference()
            if results:
                yield from results
