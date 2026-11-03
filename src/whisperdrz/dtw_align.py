"""Whisper-style DTW alignment on WhisperDRZ for word-level timestamp fill-in.

This module replaces the Transcriber's wav2vec2-based forced-alignment path.
The decoder emits sparse <|t|>word<|t|> timestamps by design (sparse-timestamp
augmentation during training), so ~55% of predicted words are untimed. DTW on
cross-attention captures their timings without loading a second model.

Entry point: `fill_word_timings(model, tokenizer, xa, pred_text, head_mask)`.

The algorithm runs a parallel forward over the predicted tokens with hooks on
each decoder block's cross_attn.query / cross_attn.key linears, then DTWs the
selected alignment heads to map text positions to audio frames. For texts that
exceed the decoder positional-embedding cap (448), we split into chunks at
speaker-line boundaries (renumbered per chunk to match training distribution)
and anchor-stitch chunks 2+ to the nearest decoder-emitted timestamp.

Char-level retokenization at DTW time (no retraining) measured at 60-87 ms
median start/end error on a 150-sample benchmark — about 5x the decoder's own
~30 ms-emissions but covering every word.
"""

import base64
from concurrent.futures import ThreadPoolExecutor
import gzip
import os
import re

import numba
import numpy as np
import torch

# Shared CPU pool for the per-chunk DTW kernels at finalize time. Sized to the
# benchmarked sweet-spot of 8 — beyond that, L3 contention dominates parallelism
# gains. Held module-global so we don't pay ThreadPoolExecutor's spawn cost on
# every fill_word_timings_batch call.
_DTW_POOL = ThreadPoolExecutor(
    max_workers=min(8, max(1, (os.cpu_count() or 8))),
    thread_name_prefix="dtw",
)

from whisperdrz.audio import (
    TOKENS_PER_SECOND,
    LogMelSpectrogram,
)
from whisperdrz.clean import TIMINGS_FORMAT

# Whisper large-v3 alignment heads (32 layers x 20 heads, 10 selected).
# WhisperDRZ is fine-tuned from large-v3 so the heads transfer.
LARGE_V3_ALIGNMENT_HEADS = b"ABzY8gWO1E0{>%R7(9S+Kn!D~%ngiGaR?*L!iJG9p-nab0JQ=-{D1-g00"


def make_head_mask(model) -> torch.Tensor:
    """Decode the alignment-head bitmask onto the model's layer/head shape."""
    arr = np.frombuffer(
        gzip.decompress(base64.b85decode(LARGE_V3_ALIGNMENT_HEADS)), dtype=bool
    ).copy()
    return torch.from_numpy(arr).reshape(
        model.dims.n_text_layer, model.dims.n_text_head
    )


# ---------- DTW kernel (CPU, numba) ----------


def _median_filter_sort(x: torch.Tensor, width: int) -> torch.Tensor:
    """Generic sort-based median filter (fallback when triton unavailable)."""
    pad = width // 2
    if x.shape[-1] <= pad:
        return x
    orig_shape = x.shape
    flat = x.reshape(-1, 1, orig_shape[-1])
    flat = torch.nn.functional.pad(flat, (pad, pad), mode="reflect")
    return (
        flat.unfold(-1, width, 1)
        .sort(dim=-1)
        .values[..., width // 2]
        .reshape(orig_shape)
    )


# Triton kernel: ported from whisper/triton_ops.py — bubble-sort network per
# row (one program per row of the unfolded view), much cheaper than the
# generic sort on width=7.
_MEDIAN_KERNEL_CACHE: dict[int, object] = {}


def _build_median_kernel(filter_width: int):
    import triton  # local import — keep optional dep contained
    import triton.language as tl

    @triton.jit
    def _kernel(y, x, x_stride, y_stride, BLOCK_SIZE: tl.constexpr):
        row_idx = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < y_stride
        x_ptr = x + row_idx * x_stride  # noqa: F841
        y_ptr = y + row_idx * y_stride
        LOAD_ALL_ROWS_HERE
        BUBBLESORT_HERE
        tl.store(y_ptr + offsets, MIDDLE_ROW_HERE, mask=mask)

    kernel = triton.JITFunction(_kernel.fn)
    src = kernel.src.replace(
        "    LOAD_ALL_ROWS_HERE",
        "\n".join(
            f"    row{i} = tl.load(x_ptr + offsets + {i}, mask=mask)"
            for i in range(filter_width)
        ),
    )
    bubble = []
    for i in range(filter_width // 2 + 1):
        passes = []
        for j in range(filter_width - i - 1):
            passes.append(
                "\n".join(
                    [
                        f"    smaller = tl.where(row{j} < row{j + 1}, row{j}, row{j + 1})",
                        f"    larger = tl.where(row{j} > row{j + 1}, row{j}, row{j + 1})",
                        f"    row{j} = smaller",
                        f"    row{j + 1} = larger",
                    ]
                )
            )
        bubble.append("\n\n".join(passes))
    src = src.replace("    BUBBLESORT_HERE", "\n\n".join(bubble))
    src = src.replace("MIDDLE_ROW_HERE", f"row{filter_width // 2}")
    if hasattr(kernel, "_unsafe_update_src"):
        kernel._unsafe_update_src(src)
        kernel.hash = None
    else:
        kernel.src = src
    return kernel


def _median_filter_triton(x: torch.Tensor, width: int) -> torch.Tensor:
    pad = width // 2
    if x.shape[-1] <= pad:
        return x
    orig_shape = x.shape
    T = orig_shape[-1]
    flat = x.reshape(-1, T)  # [N, T]
    # F.pad mode="reflect" needs a leading channel dim; squeeze it off.
    padded = (
        torch.nn.functional.pad(flat.unsqueeze(0), (pad, pad), mode="reflect")
        .squeeze(0)
        .contiguous()
    )  # [N, T + 2*pad]
    if width not in _MEDIAN_KERNEL_CACHE:
        _MEDIAN_KERNEL_CACHE[width] = _build_median_kernel(width)
    kernel = _MEDIAN_KERNEL_CACHE[width]
    N = padded.shape[0]
    y = torch.empty(N, T, dtype=x.dtype, device=x.device)
    block = 1 << max(0, (T - 1).bit_length())
    kernel[(N,)](y, padded, padded.stride(0), y.stride(0), BLOCK_SIZE=block)
    return y.reshape(orig_shape)


def median_filter(x: torch.Tensor, width: int) -> torch.Tensor:
    """Median filter along the last dim. Uses a triton bubble-sort-network on
    CUDA when available (~0.1 ms for width=7 vs ~1.8 ms for the generic
    sort-based fallback); falls back to sort-based on CPU or on triton failure."""
    if x.is_cuda:
        try:
            return _median_filter_triton(x, width)
        except Exception:
            pass
    return _median_filter_sort(x, width)


@numba.njit(cache=True, parallel=False, nogil=True)
def _dtw_numba(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward DP + backtrace, all inside one nogil numba kernel so threading
    actually scales. The serial python backtrack used to hold the GIL on every
    chunk; folding it into the compiled kernel lets a thread pool run B DTWs
    truly in parallel (numba releases the GIL when `nogil=True`).
    Returns `(text_idx, time_idx)` as 1-D int64 arrays of length `path_len`."""
    N, M = x.shape
    cost = np.full((N + 1, M + 1), np.inf, dtype=np.float32)
    trace = np.empty((N + 1, M + 1), dtype=np.int8)
    for k in range(M + 1):
        trace[0, k] = 2
    for k in range(N + 1):
        trace[k, 0] = 1
    trace[0, 0] = -1
    cost[0, 0] = 0.0
    for j in range(1, M + 1):
        for i in range(1, N + 1):
            c0 = cost[i - 1, j - 1]
            c1 = cost[i - 1, j]
            c2 = cost[i, j - 1]
            if c0 <= c1 and c0 <= c2:
                c, t = c0, 0
            elif c1 <= c2:
                c, t = c1, 1
            else:
                c, t = c2, 2
            cost[i, j] = x[i - 1, j - 1] + c
            trace[i, j] = t
    # Backtrace. Max path length = N + M (worst case all 1- or 2-moves).
    text_buf = np.empty(N + M, dtype=np.int64)
    time_buf = np.empty(N + M, dtype=np.int64)
    i = N
    j = M
    k = 0
    while i > 0 or j > 0:
        text_buf[k] = i - 1
        time_buf[k] = j - 1
        t = trace[i, j]
        if t == 0:
            i -= 1
            j -= 1
        elif t == 1:
            i -= 1
        else:
            j -= 1
        k += 1
    # Reverse into output arrays (np.flip on a slice isn't nogil-numba-friendly).
    text_idx = np.empty(k, dtype=np.int64)
    time_idx = np.empty(k, dtype=np.int64)
    for n in range(k):
        text_idx[n] = text_buf[k - 1 - n]
        time_idx[n] = time_buf[k - 1 - n]
    return text_idx, time_idx


def dtw_cpu(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns `(text_idx, time_idx)` — same shape contract as before
    (`(2, path_len)` callers unpack via `text_idx, time_idx = dtw_cpu(...)`)."""
    return _dtw_numba(x)


# ---------- Tokenization (char-level with explicit spaces) ----------


# Per-tokenizer cache of char → token-id tuples. tokens_chars calls
# `tokenizer.encode(ch)` once per character; without caching that's hundreds
# of CoreBPE dispatches per DTW call and dominates the alignment CPU time
# (~28 ms / call for a typical 200-char chunk).
_CHAR_TOK_CACHE: dict[int, dict[str, tuple[int, ...]]] = {}


def _char_tokens(tokenizer, ch: str) -> tuple[int, ...]:
    cache = _CHAR_TOK_CACHE.get(id(tokenizer))
    if cache is None:
        cache = {}
        _CHAR_TOK_CACHE[id(tokenizer)] = cache
    hit = cache.get(ch)
    if hit is not None:
        return hit
    eot = tokenizer.eot
    encoded = tuple(t for t in tokenizer.encode(ch) if t < eot)
    cache[ch] = encoded
    return encoded


def tokens_chars(
    plain_text: str, tokenizer
) -> tuple[list[int], list[str], list[list[int]]]:
    """Char-level tokenization with an explicit space token between words.

    Whisper's default BPE folds each word's preceding silence into its
    leading-space token, so DTW can't separate them. Char-level gives DTW one
    cross-attention row per character → cleaner alignment without retraining.

    The eot lives in its OWN dummy word group at the end (rather than merged
    onto the last real word's group), so DTW doesn't silently drop the last
    word's timing.
    """
    words = plain_text.split()
    if not words:
        return [], [], []
    space_toks = _char_tokens(tokenizer, " ")
    ids: list[int] = []
    word_token_groups: list[list[int]] = []
    for wi, w in enumerate(words):
        start = len(ids)
        if wi > 0:
            ids.extend(space_toks)
        for ch in w:
            ids.extend(_char_tokens(tokenizer, ch))
        word_token_groups.append(ids[start:])
    words = [*words, ""]
    word_token_groups = [*word_token_groups, [tokenizer.eot]]
    return ids, words, word_token_groups


# ---------- Predicted-text parsing ----------

_TOKEN_RE = re.compile(r"<\|\d+\.\d+\|>|[^\s<]+")
_STRIP_BRACKET_TAG = re.compile(r"\[/?[a-zA-Z]+\]")
_STRIP_META = re.compile(r"<(?!\|)[^>]*>")
_STRIP_CHOP = re.compile(r"\[CHOP\]")
_SPK_LINE_FIND = re.compile(r"\[(\d+|c)\]\s*")


def _extract_line_words_and_tag_anchors(
    line_text: str,
) -> tuple[
    list[tuple[str, float | None, float | None]],
    list[tuple[float, float, int | None, int | None]],
]:
    """Walk a single line's text and pull out words AND timed `[tag]` anchors.

    Returns:
      words: per-word (surface, decoder_start_or_None, decoder_end_or_None)
      tag_anchors: per timed `<|s|>[tag]<|e|>` triple,
                   (audio_s, audio_e, prev_word_idx_in_line_or_None,
                    next_word_idx_in_line_or_None).
                   prev/next are line-local word indices (0-based into `words`).
    """
    # Strip metadata `<accent:..>`/`<emotion:..>` and CHOP, but KEEP bracket
    # tags so we can detect timed ones. Speaker prefix has already been
    # removed by parse_predicted_lines before this is called.
    text = _STRIP_META.sub(" ", _STRIP_CHOP.sub(" ", line_text))
    items = [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]
    words: list[tuple[str, float | None, float | None]] = []
    tag_anchors: list[tuple[float, float, int | None, int | None]] = []
    for i, (tok, ts, te) in enumerate(items):
        if TIMINGS_FORMAT.fullmatch(tok):
            continue
        # Check left and right adjacent timestamps (no whitespace gap).
        left_t = right_t = None
        if i > 0:
            prev_tok, _, prev_end = items[i - 1]
            m = TIMINGS_FORMAT.fullmatch(prev_tok)
            if m is not None and prev_end == ts:
                left_t = float(m.group(1))
        if i + 1 < len(items):
            next_tok, next_start, _ = items[i + 1]
            m = TIMINGS_FORMAT.fullmatch(next_tok)
            if m is not None and next_start == te:
                right_t = float(m.group(1))

        if tok.startswith("[") and tok.endswith("]") and len(tok) > 2:
            # `[tag]`. Timed tag → anchor; un-timed → skip (paired tags etc).
            if left_t is not None and right_t is not None:
                prev_wi = len(words) - 1 if words else None
                next_wi = len(words)  # next word that will be appended
                tag_anchors.append((left_t, right_t, prev_wi, next_wi))
            continue
        # Plain word.
        words.append((tok, left_t, right_t))

    # Convert tag-anchor "next_wi == final length" into None (no next word).
    fixed = []
    for a_s, a_e, p, n in tag_anchors:
        if n is not None and n >= len(words):
            n = None
        if p is None and n is None:
            continue  # nothing to constrain
        fixed.append((a_s, a_e, p, n))
    return words, fixed


def parse_predicted_lines(
    text: str,
) -> tuple[
    list[tuple[int | str, list[tuple[int, str, float | None, float | None]]]],
    list[tuple[str, float | None, float | None]],
    list[tuple[float, float, int | None, int | None]],
]:
    """Split WhisperDRZ prediction into per-speaker lines AND keep:
      - a flat per-word index that ties each line word back to its global
        pred_words position; and
      - a flat list of timed-tag audio anchors with global word indices.

    Each tag anchor is `(audio_s, audio_e, prev_global_word_idx,
    next_global_word_idx)` where prev/next refer to indices in
    `pred_words_flat`. Either may be `None` at a line boundary.
    """
    text = _STRIP_CHOP.sub(" ", text)
    matches = list(_SPK_LINE_FIND.finditer(text))
    lines: list[
        tuple[int | str, list[tuple[int, str, float | None, float | None]]]
    ] = []
    pred_words_flat: list[tuple[str, float | None, float | None]] = []
    tag_anchors_global: list[tuple[float, float, int | None, int | None]] = []

    def consume(sub: str, spk: int | str) -> None:
        ws, tags = _extract_line_words_and_tag_anchors(sub)
        if not ws and not tags:
            return
        line_global_indices: list[int] = []
        line_ws = []
        for w, s, e in ws:
            gi = len(pred_words_flat)
            line_global_indices.append(gi)
            line_ws.append((gi, w, s, e))
            pred_words_flat.append((w, s, e))
        if line_ws:
            lines.append((spk, line_ws))
        for a_s, a_e, p_li, n_li in tags:
            p_gi = (
                line_global_indices[p_li]
                if p_li is not None and 0 <= p_li < len(line_global_indices)
                else None
            )
            n_gi = (
                line_global_indices[n_li]
                if n_li is not None and 0 <= n_li < len(line_global_indices)
                else None
            )
            if p_gi is None and n_gi is None:
                continue
            tag_anchors_global.append((a_s, a_e, p_gi, n_gi))

    if not matches or matches[0].start() > 0:
        end = matches[0].start() if matches else len(text)
        consume(text[:end], 0)

    for i, m in enumerate(matches):
        spk_str = m.group(1)
        try:
            spk: int | str = int(spk_str)
        except ValueError:
            spk = spk_str
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        consume(text[m.end() : end], spk)

    return lines, pred_words_flat, tag_anchors_global


# ---------- Speaker-aware chunking ----------


def _render_chunk(
    chunk_lines: list[
        tuple[int | str, list[tuple[int, str, float | None, float | None]]]
    ],
) -> tuple[str, list[tuple[int | None, str]]]:
    """Render a chunk with renumbered speakers (model-facing). Returns
    `(text, tagged)` where `tagged[i] = (orig_pred_idx_or_None, surface)` —
    `None` is for speaker-tag tokens, real word indices for the rest."""
    speaker_map: dict[int | str, int] = {}
    text_parts: list[str] = []
    tagged: list[tuple[int | None, str]] = []
    for spk, line_ws in chunk_lines:
        if spk not in speaker_map:
            speaker_map[spk] = len(speaker_map)
        ns = speaker_map[spk]
        prefix = f"[{ns}]"
        text_parts.append(prefix)
        tagged.append((None, prefix))
        for orig_idx, w, _, _ in line_ws:
            text_parts.append(w)
            tagged.append((orig_idx, w))
    return " ".join(text_parts), tagged


def chunk_speaker_lines(
    lines: list[tuple[int | str, list[tuple[int, str, float | None, float | None]]]],
    tokenizer,
    max_tokens: int,
):
    """Pack speaker lines into chunks ≤ max_tokens (chars-mode token count).
    Prefers line boundaries; only splits mid-line when a single line alone
    exceeds the budget."""

    def fits(cl):
        text, _ = _render_chunk(cl)
        ids, _, _ = tokens_chars(text, tokenizer)
        return len(ids) <= max_tokens

    chunks: list[list] = []
    cur: list = []
    queue = list(lines)
    while queue:
        spk, ws = queue[0]
        if fits([*cur, (spk, ws)]):
            cur.append((spk, ws))
            queue.pop(0)
            continue
        if cur:
            chunks.append(cur)
            cur = []
            continue
        lo, hi = 1, len(ws)
        best = 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if fits([(spk, ws[:mid])]):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        chunks.append([(spk, ws[:best])])
        queue[0] = (spk, ws[best:])
        if not queue[0][1]:
            queue.pop(0)
    if cur:
        chunks.append(cur)
    return chunks


# ---------- Cross-attention capture & per-chunk DTW ----------


class CrossAttnCapture:
    """Hooks cross_attn.query / cross_attn.key Linears per decoder block,
    captures their pre-attention projections. Active only inside the `with`.

    When `layers` is provided, only those layer indices get hooked. The whisper
    large-v3 alignment heads select ~6 of 32 layers, so by default we hook the
    union of layers in head_mask — skipping the other ~26 layers cuts the
    per-call Python hook overhead by ~80%."""

    def __init__(
        self,
        model,
        layers: set[int] | None = None,
    ):
        self.model = model
        self.n_layers = model.dims.n_text_layer
        self.n_heads = model.dims.n_text_head
        self.head_dim = model.dims.n_text_state // self.n_heads
        self.q_out: list[torch.Tensor | None] = [None] * self.n_layers
        self.k_out: list[torch.Tensor | None] = [None] * self.n_layers
        self._layers = layers
        self._hooks: list = []

    def __enter__(self):
        for i, block in enumerate(self.model.decoder.blocks):
            if block.cross_attn is None:
                continue
            if self._layers is not None and i not in self._layers:
                continue

            def make_q_hook(idx):
                def h(_m, _ins, out):
                    self.q_out[idx] = out

                return h

            def make_k_hook(idx):
                def h(_m, _ins, out):
                    self.k_out[idx] = out

                return h

            self._hooks.append(
                block.cross_attn.query.register_forward_hook(make_q_hook(i))
            )
            self._hooks.append(
                block.cross_attn.key.register_forward_hook(make_k_hook(i))
            )
        return self

    def __exit__(self, *a):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def compute_attention(self, head_mask: torch.Tensor) -> torch.Tensor:
        """Recompute QK^T (no softmax) for the selected alignment heads.
        Returns [n_selected_heads, T_text, T_audio] when B=1, else
        [B, n_selected_heads, T_text, T_audio].

        Gathers all selected (layer, head) Qs and Ks into stacked tensors, then
        runs ONE batched matmul instead of the old per-head Python loop. Same
        math, ~10x fewer kernel launches on a typical 10-head selection."""
        scale = self.head_dim**-0.25  # whisper-style scale split between Q and K
        sel = head_mask.nonzero()  # [n_selected, 2] of (layer, head)
        if sel.shape[0] == 0:
            raise ValueError("head_mask selects no heads")

        qs: list[torch.Tensor] = []
        ks: list[torch.Tensor] = []
        for layer, head in sel.tolist():
            q = self.q_out[layer]
            k = self.k_out[layer]
            assert q is not None and k is not None
            B, T_text, _ = q.shape
            _, T_audio, _ = k.shape
            qh = q.view(B, T_text, self.n_heads, self.head_dim)[:, :, head]
            kh = k.view(B, T_audio, self.n_heads, self.head_dim)[:, :, head]
            qs.append(qh)
            ks.append(kh)

        # [n_selected, B, T_text, head_dim] / [n_selected, B, T_audio, head_dim]
        Q = torch.stack(qs, dim=0).float() * scale
        K = torch.stack(ks, dim=0).float() * scale
        # One matmul: [n_selected, B, T_text, T_audio]
        weights = Q @ K.transpose(-1, -2)
        # Reorder to the expected [B, n_selected, T_text, T_audio]
        weights = weights.transpose(0, 1).contiguous()
        return weights[0] if weights.shape[0] == 1 else weights


def _apply_anchor_bands(
    neg_matrix: np.ndarray,
    external_words: list[str],
    external_word_tokens: list[list[int]],
    audio_cap: int,
    word_anchors: list[tuple[int, float, float]] | None,
    tag_anchors: list[tuple[float, float, int | None, int | None]] | None,
    slack_frames: int,
    col_offset: int = 0,
) -> None:
    """In-place anchor masking on neg_matrix. Walked out of _dtw_from_attention
    so the batched DTW path can reuse the exact same band logic on CPU.

    col_offset: audio position of neg_matrix's column 0 when the matrix was
    sliced to a sub-range of the window; anchor times (window-absolute seconds)
    are translated into the sliced coordinate frame.
    """
    n_rows = neg_matrix.shape[0]
    word_tokens_lengths = [len(t) for t in external_word_tokens[:-1]]
    word_boundaries_anchor = np.cumsum([1, *word_tokens_lengths])
    bands: list[tuple[int, int, int | None, int | None]] = []
    if word_anchors:
        seen_max_audio = -1.0
        for wi, a_s, a_e in word_anchors:
            if wi >= len(external_words) - 1:
                continue
            if a_e <= seen_max_audio:
                continue
            a_lo = max(
                0, int(round(a_s * TOKENS_PER_SECOND)) - slack_frames - col_offset
            )
            a_hi = min(
                audio_cap,
                int(round(a_e * TOKENS_PER_SECOND)) + slack_frames - col_offset,
            )
            if a_lo >= a_hi:
                continue
            r_lo = int(word_boundaries_anchor[wi])
            r_hi = min(int(word_boundaries_anchor[wi + 1]), n_rows)
            if r_lo >= r_hi:
                continue
            bands.append((r_lo, r_hi, a_lo, a_hi))
            seen_max_audio = a_e
    if tag_anchors:
        for a_s, a_e, p_local, n_local in tag_anchors:
            a_lo = max(
                0, int(round(a_s * TOKENS_PER_SECOND)) - slack_frames - col_offset
            )
            a_hi = min(
                audio_cap,
                int(round(a_e * TOKENS_PER_SECOND)) + slack_frames - col_offset,
            )
            if a_hi <= 0 or a_lo >= audio_cap:
                continue
            if p_local is not None and p_local < len(external_words) - 1:
                last_row_p = min(
                    int(word_boundaries_anchor[p_local + 1]) - 1, n_rows - 1
                )
                if last_row_p >= 0:
                    bands.append((last_row_p, last_row_p + 1, None, a_hi))
            if n_local is not None and n_local < len(external_words) - 1:
                first_row_n = min(int(word_boundaries_anchor[n_local]), n_rows - 1)
                if first_row_n >= 0:
                    bands.append((first_row_n, first_row_n + 1, a_lo, None))
    for r_lo, r_hi, a_lo, a_hi in bands:
        if a_lo is not None and a_lo > 0:
            neg_matrix[r_lo:r_hi, :a_lo] = np.inf
        if a_hi is not None and a_hi < neg_matrix.shape[1]:
            neg_matrix[r_lo:r_hi, a_hi:] = np.inf


def _dtw_from_neg_matrix(
    neg_matrix: np.ndarray,
    external_words: list[str],
    external_word_tokens: list[list[int]],
) -> list[tuple[str, float, float]]:
    """Run DTW on a pre-prepped (and anchor-masked) neg-cost matrix and emit
    per-word (start, end) in seconds."""
    if neg_matrix.shape[0] == 0 or neg_matrix.shape[1] == 0:
        return []
    text_idx, time_idx = dtw_cpu(neg_matrix)
    words = list(external_words)
    word_tokens = list(external_word_tokens)
    if len(word_tokens) <= 1:
        return []
    word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))
    jumps = np.pad(np.diff(text_idx), (1, 0), constant_values=1).astype(bool)
    jump_times = time_idx[jumps] / TOKENS_PER_SECOND
    if word_boundaries[-1] >= len(jump_times):
        return []
    start_times = jump_times[word_boundaries[:-1]]
    end_times = jump_times[word_boundaries[1:]]
    return list(zip(words, start_times, end_times, strict=False))


# Per-device arange template cache. _dtw_prep_batch builds two `torch.arange`s
# every call to construct audio/text masks; caching them removes a handful of
# kernel launches per finalize.
_ARANGE_CACHE: dict[tuple[int, torch.device], torch.Tensor] = {}


def _arange_cached(n: int, device: torch.device) -> torch.Tensor:
    key = (n, device)
    t = _ARANGE_CACHE.get(key)
    if t is None:
        t = torch.arange(n, device=device)
        _ARANGE_CACHE[key] = t
    return t


# Persistent pinned host buffer for the D→H transfer at the end of
# _dtw_prep_batch. PyTorch's default .cpu() uses pageable memory (~2 GB/s on
# this host); a pre-pinned destination via .copy_(non_blocking=True) hits
# ~25 GB/s, dropping the dominant prep cost roughly 10x.
#
# ONE grow-only flat buffer per dtype, viewed to the requested shape per call.
# The old dict keyed by exact (shape, dtype) leaked unboundedly under the
# continuous-batching server: B/T_text_pad/audio_cap vary almost every finalize,
# so it nearly never hit and cudaHostAlloc'd a fresh 2-32MB pinned block per
# call — the "RssShmem/pinned-host" leak (~0.5 GB/min per GPU worker; /dev/zero
# MAP_SHARED maps growing ~70/min). Pinned blocks are cached by torch's host
# allocator and never returned to the OS, so the cache MUST be bounded here.
_PINNED_HOST_BUFFERS: dict[torch.dtype, torch.Tensor] = {}


def _pinned_host_buffer(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    numel = 1
    for d in shape:
        numel *= d
    buf = _PINNED_HOST_BUFFERS.get(dtype)
    if buf is None or buf.numel() < numel:
        buf = torch.empty(numel, dtype=dtype, pin_memory=True)
        _PINNED_HOST_BUFFERS[dtype] = buf
    return buf[:numel].view(shape)


def _dtw_prep_batch(
    full_weights: torch.Tensor,
    actual_lens: list[int],
    audio_caps: list[int],
    medfilt_width: int = 7,
) -> np.ndarray:
    """Batched GPU preprocessing of the cross-attention tensor.

    Replaces the per-chunk `softmax + std/mean normalise + median + mean-over-heads
    + .cpu()` loop in `_dtw_from_attention` with one set of GPU ops over the full
    `[B, H, T_text_pad, T_audio]` tensor, followed by a SINGLE device→host transfer.

    `audio_caps[bi]` caps the softmax to each chunk's real audio length: cols
    beyond the cap are masked to −inf so softmax ignores them. `actual_lens[bi]`
    limits the std/mean reduction to each chunk's valid text rows; padding rows
    (eot-filled) would otherwise pull the per-(head, audio_pos) stats sideways.

    Returns a CPU `np.ndarray` of shape `[B, T_text_pad, max_audio_cap]`. The
    caller slices per chunk via `arr[bi, sot_len : actual_lens[bi] - 1, :audio_caps[bi]]`.
    """
    B, H, T_text_pad, T_audio = full_weights.shape
    device = full_weights.device

    actual_lens_t = torch.tensor(actual_lens, dtype=torch.long, device=device)
    audio_caps_t = torch.tensor(audio_caps, dtype=torch.long, device=device)
    max_audio_cap = int(audio_caps_t.max().item())

    arange_a = _arange_cached(T_audio, device)
    audio_mask = arange_a[None, :] < audio_caps_t[:, None]  # [B, T_audio]
    w = full_weights.masked_fill(~audio_mask[:, None, None, :], float("-inf"))
    w = torch.softmax(w, dim=-1)  # masked cols become 0

    arange_t = _arange_cached(T_text_pad, device)
    text_mask = arange_t[None, :] < actual_lens_t[:, None]  # [B, T_text_pad]
    text_mask_f = text_mask[:, None, :, None].to(w.dtype)
    n = actual_lens_t.view(B, 1, 1, 1).to(w.dtype)

    mean = (w * text_mask_f).sum(dim=-2, keepdim=True) / n
    var = (((w - mean) ** 2) * text_mask_f).sum(dim=-2, keepdim=True) / n
    std = var.clamp_min(1e-12).sqrt()
    w = (w - mean) / std

    if medfilt_width > 1:
        w = median_filter(w, medfilt_width)

    matrix = w.mean(dim=1)  # [B, T_text_pad, T_audio]
    matrix = matrix[:, :, :max_audio_cap].contiguous()
    host = _pinned_host_buffer(tuple(matrix.shape), matrix.dtype)
    host.copy_(matrix, non_blocking=True)
    torch.cuda.synchronize()
    # numpy() shares memory with host; subsequent calls reuse the same buffer,
    # so the previous call's numpy array becomes stale. That's fine — the
    # caller consumes matrices_cpu before the next prep call.
    return host.numpy()


def _dtw_from_attention(
    attn_weights: torch.Tensor,
    tokenizer,
    external_words: list[str],
    external_word_tokens: list[list[int]],
    num_frames: int,
    medfilt_width: int = 7,
    word_anchors: list[tuple[int, float, float]] | None = None,
    tag_anchors: list[tuple[float, float, int | None, int | None]] | None = None,
    slack_frames: int = 7,
    audio_lo: int = 0,
    audio_hi: int | None = None,
) -> list[tuple[str, float, float]]:
    """Convert raw cross-attention QK^T (selected heads, no softmax) into per-word
    timings via softmax → norm → median-filter → cost mask → DTW.

    attn_weights: [n_heads, T_text_full, T_audio] for ONE chunk. T_text_full
    includes sot + no_timestamps + text_tokens + eot rows.

    audio_lo/audio_hi: audio-position sub-range this chunk's words can occupy
    (from surrounding decoder anchors). DTW runs on the sliced matrix so a
    chunk covering the window's tail can't be corner-to-corner mapped onto the
    whole window. Returned times are window-absolute (audio_lo added back).
    """
    audio_cap = num_frames // 2
    weights = attn_weights[:, :, :audio_cap]
    weights = torch.softmax(weights, dim=-1)
    std, mean = torch.std_mean(weights, dim=-2, keepdim=True, unbiased=False)
    weights = (weights - mean) / std
    if medfilt_width > 1:
        weights = median_filter(weights, medfilt_width)

    matrix = weights.mean(dim=0)
    matrix = matrix[len(tokenizer.sot_sequence) : -1]
    neg_matrix = -matrix.cpu().numpy()

    lo = max(0, min(audio_lo, audio_cap))
    hi = audio_cap if audio_hi is None else max(lo, min(audio_hi, audio_cap))
    if hi - lo < 2:
        return []
    if lo > 0 or hi < audio_cap:
        neg_matrix = np.ascontiguousarray(neg_matrix[:, lo:hi])

    if word_anchors or tag_anchors:
        _apply_anchor_bands(
            neg_matrix,
            external_words,
            external_word_tokens,
            hi - lo,
            word_anchors,
            tag_anchors,
            slack_frames,
            col_offset=lo,
        )

    result = _dtw_from_neg_matrix(neg_matrix, external_words, external_word_tokens)
    if lo > 0 and result:
        off = lo / TOKENS_PER_SECOND
        result = [(w, s + off, e + off) for w, s, e in result]
    return result


def _find_alignment_chunk(
    model,
    tokenizer,
    xa: torch.Tensor,
    text_tokens: list[int],
    external_words: list[str],
    external_word_tokens: list[list[int]],
    head_mask: torch.Tensor,
    num_frames: int,
    medfilt_width: int = 7,
    word_anchors: list[tuple[int, float, float]] | None = None,
    tag_anchors: list[tuple[float, float, int | None, int | None]] | None = None,
    slack_frames: int = 7,
    audio_lo: int = 0,
    audio_hi: int | None = None,
) -> list[tuple[str, float, float]]:
    """One decoder parallel forward + DTW for ONE chunk. Returns (word, start,
    end) per word group in `external_words` (drops the trailing eot group)."""
    if not text_tokens:
        return []
    full_tokens = torch.tensor(
        [
            *tokenizer.sot_sequence,
            tokenizer.no_timestamps,
            *text_tokens,
            tokenizer.eot,
        ],
        device=xa.device,
    )
    alignment_layers = {int(l) for l in head_mask.nonzero()[:, 0].tolist()}
    with CrossAttnCapture(model, layers=alignment_layers) as cap:
        with torch.inference_mode():
            # Decoder-only call — skips encoder by reusing the cached xa.
            model.decoder(full_tokens.unsqueeze(0), xa.unsqueeze(0))
        weights = cap.compute_attention(head_mask.to(xa.device))
    return _dtw_from_attention(
        weights,
        tokenizer,
        external_words,
        external_word_tokens,
        num_frames,
        medfilt_width=medfilt_width,
        word_anchors=word_anchors,
        tag_anchors=tag_anchors,
        slack_frames=slack_frames,
        audio_lo=audio_lo,
        audio_hi=audio_hi,
    )


# ---------- Public entry points ----------


def _chunk_audio_bounds(
    recs: list[dict | None],
    pred_words_flat: list[tuple[str, float | None, float | None]],
    win_dur: float,
    overlap_slack: float = 5.0,
) -> list[tuple[float, float] | None]:
    """Per prepared chunk: the (lo, hi) window-audio sub-range its words can
    plausibly occupy, derived from the nearest decoder-emitted timestamps
    OUTSIDE the chunk. Replaces the old post-hoc anchor/prev-end shift: the
    DTW search itself is restricted, so fills are in-range by construction.

    overlap_slack mirrors the decoder's own line-start rule (timings may go
    back up to 5 s for overlapping speakers), so legitimate overlaps aren't
    squeezed. A chunk's own anchors always widen the range to contain them.
    """
    bounds: list[tuple[float, float] | None] = []
    for rec in recs:
        if rec is None:
            bounds.append(None)
            continue
        gids = [oi for oi, _ in rec["tagged"] if oi is not None]
        if not gids:
            bounds.append((0.0, win_dur))
            continue
        g0, g1 = min(gids), max(gids)
        lo = 0.0
        for gi in range(g0 - 1, -1, -1):
            _w, dec_s, dec_e = pred_words_flat[gi]
            t = dec_e if dec_e is not None else dec_s
            if t is not None:
                lo = max(0.0, t - overlap_slack)
                break
        hi = win_dur
        for gi in range(g1 + 1, len(pred_words_flat)):
            _w, dec_s, dec_e = pred_words_flat[gi]
            t = dec_s if dec_s is not None else dec_e
            if t is not None:
                hi = min(win_dur, t + overlap_slack)
                break
        # The chunk's own decoder anchors are ground truth — contain them.
        for _i, a_s, a_e in rec["word_anchors"]:
            lo = min(lo, max(0.0, a_s - 0.5))
            hi = max(hi, min(win_dur, a_e + 0.5))
        for a_s, a_e, _p, _n in rec["tag_anchors"]:
            lo = min(lo, max(0.0, a_s - 0.5))
            hi = max(hi, min(win_dur, a_e + 0.5))
        if hi - lo < 0.2:
            lo, hi = 0.0, win_dur
        bounds.append((lo, hi))
    return bounds


def _drop_bad_fills(
    lines: list[tuple[int | str, list[tuple[int, str, float | None, float | None]]]],
    pred_words_flat: list[tuple[str, float | None, float | None]],
    out_timings: list[tuple[float, float] | None],
    win_dur: float,
    tag_anchors: list[tuple[float, float, int | None, int | None]] | None = None,
    tol: float = 0.02,
) -> None:
    """In-place safety net over DTW fills: a fill that runs past the window's
    real audio, or breaks monotonicity against its own line's decoder
    timestamps (word AND timed-tag) / accepted fills, is dropped (the word
    stays untimed — valid WhisperDRZ format) instead of being emitted wrong.
    Backsteps within tol are clamped rather than dropped."""
    # Timed tags ([breath], [pause], ...) sit BETWEEN words but carry decoder
    # timestamps; fills must be monotonic against them too.
    tag_after: dict[int, float] = {}  # word gi is preceded by a tag ending here
    tag_before: dict[int, float] = {}  # word gi is followed by a tag starting here
    for a_s, a_e, p_gi, n_gi in tag_anchors or []:
        if n_gi is not None:
            tag_after[n_gi] = max(tag_after.get(n_gi, 0.0), a_e)
        if p_gi is not None:
            tag_before[p_gi] = min(tag_before.get(p_gi, float("inf")), a_s)

    for _spk, line_ws in lines:
        # Forward pass: fills must not go backwards or past the audio end.
        last_t: float | None = None
        for orig_idx, _w, dec_s, dec_e in line_ws:
            ta = tag_after.get(orig_idx)
            if ta is not None:
                last_t = ta if last_t is None else max(last_t, ta)
            if dec_s is not None or dec_e is not None:
                t = dec_e if dec_e is not None else dec_s
                last_t = t if last_t is None else max(last_t, t)
                continue
            ft = out_timings[orig_idx]
            if ft is None:
                continue
            s, e = ft
            tb = tag_before.get(orig_idx)
            if (
                s < -0.01
                or e > win_dur + 0.25
                or (last_t is not None and s < last_t - tol)
                or (tb is not None and e > tb + tol)
            ):
                out_timings[orig_idx] = None
                continue
            if last_t is not None and s < last_t:
                s = last_t  # clamp sub-tol backstep to keep the grid monotonic
                e = max(e, s)
                out_timings[orig_idx] = (s, e)
            last_t = max(last_t, e) if last_t is not None else e
        # Backward pass: fills must not overrun the next decoder timestamp.
        next_t: float | None = None
        for orig_idx, _w, dec_s, dec_e in reversed(line_ws):
            tb = tag_before.get(orig_idx)
            if tb is not None:
                next_t = tb if next_t is None else min(next_t, tb)
            if dec_s is not None or dec_e is not None:
                t = dec_s if dec_s is not None else dec_e
                next_t = t if next_t is None else min(next_t, t)
                continue
            ft = out_timings[orig_idx]
            if ft is None:
                continue
            s, e = ft
            if next_t is not None and e > next_t + tol:
                out_timings[orig_idx] = None
                continue
            if next_t is not None and e > next_t:
                e = next_t  # clamp sub-tol overrun
                s = min(s, e)
                out_timings[orig_idx] = (s, e)
            next_t = s if next_t is None else min(next_t, s)


def fill_word_timings(
    model,
    tokenizer,
    xa: torch.Tensor,
    pred_text: str,
    *,
    head_mask: torch.Tensor | None = None,
    num_frames: int | None = None,
    max_tokens: int = 440,
) -> str:
    """Augment WhisperDRZ-format pred_text with <|t|>word<|t|> for words missing them.

    Args:
        model: WhisperDRZ model (must be `.eval()`, on the same device as xa).
        tokenizer: whisperdrz Tokenizer.
        xa: cached encoder output, [n_audio_ctx, n_audio_state]. The encoder
            doesn't run again here.
        pred_text: decoder's output text (sparse-timestamped, with [N] tags).
        head_mask: per-layer/head alignment mask. Built from large-v3 blob if
            omitted.
        num_frames: actual audio frames (caps DTW search to the un-padded audio
            region). Defaults to xa.shape[0] = 1500 (full 30 s window).
        max_tokens: chunk budget below the decoder positional-embedding cap.

    Returns: pred_text with one <|t|>word<|t|> per real word. Decoder-emitted
    timings are kept where present; DTW fills the rest. Each chunk's DTW is
    restricted to the audio sub-range implied by surrounding decoder
    timestamps (see _chunk_audio_bounds), and fills that would still land
    out-of-range or non-monotonic are dropped (_drop_bad_fills).
    """
    if head_mask is None:
        head_mask = make_head_mask(model)
    if num_frames is None:
        num_frames = xa.shape[0]

    lines, pred_words_flat, tag_anchors_global = parse_predicted_lines(pred_text)
    if not pred_words_flat:
        return pred_text

    chunks = chunk_speaker_lines(lines, tokenizer, max_tokens)
    recs = [
        _prepare_chunk(cl, tokenizer, pred_words_flat, tag_anchors_global, max_tokens)
        for cl in chunks
    ]
    win_dur = (num_frames // 2) / TOKENS_PER_SECOND
    bounds = _chunk_audio_bounds(recs, pred_words_flat, win_dur)
    out_timings: list[tuple[float, float] | None] = [None] * len(pred_words_flat)

    for rec, bnd in zip(recs, bounds, strict=False):
        if rec is None or bnd is None:
            continue
        lo, hi = bnd
        dtw = _find_alignment_chunk(
            model,
            tokenizer,
            xa,
            rec["ids"],
            rec["ext_words"],
            rec["ext_word_tokens"],
            head_mask,
            num_frames,
            word_anchors=rec["word_anchors"],
            tag_anchors=rec["tag_anchors"],
            audio_lo=int(lo * TOKENS_PER_SECOND),
            audio_hi=round(hi * TOKENS_PER_SECOND),
        )
        if not dtw:
            continue

        # DTW returns len(ext_words) - 1 entries (eot dummy dropped); aligned 1:1 with `tagged`.
        tagged = rec["tagged"]
        for i, (orig_idx, _surf) in enumerate(tagged):
            if orig_idx is None or i >= len(dtw):
                continue
            _w, ds, de = dtw[i]
            out_timings[orig_idx] = (float(ds), float(de))

    _drop_bad_fills(
        lines, pred_words_flat, out_timings, win_dur, tag_anchors=tag_anchors_global
    )
    return rebuild_with_timings(pred_text, pred_words_flat, out_timings)


# Comprehensive tokenizer for in-place rewriting. Order matters: bracket tags
# and speaker tags must be tried before the bare-word class so they aren't
# absorbed into it.
_PARSE_TOKEN = re.compile(
    r"\[/?[a-zA-Z]+\]"  # paired tags / non-speech markers
    r"|\[(?:\d+|c)\]"  # speaker tag
    r"|<\|\d+\.\d+\|>"  # timestamp
    r"|<[^>]+>"  # metadata tag
    r"|[^\s<\[]+"  # word (anything not whitespace, '<', or '[')
    r"|\s+"  # whitespace
)
_TIMESTAMP_FULL = re.compile(r"<\|\d+\.\d+\|>")
_SPEAKER_FULL = re.compile(r"\[(?:\d+|c)\]")
_BRACKET_TAG_FULL = re.compile(r"\[/?[a-zA-Z]+\]")


def rebuild_with_timings(
    pred_text: str,
    pred_words_flat: list[tuple[str, float | None, float | None]],
    dtw_timings: list[tuple[float, float] | None],
) -> str:
    """Inject `<|s|>word<|e|>` around words that are currently untimed.

    Walks the ORIGINAL text token by token so all non-word content
    (speaker tags, non-speech markers like `[breath]`, paired tags, metadata
    `<accent:..>`) and all decoder-emitted timings are preserved verbatim.

    Order in `pred_words_flat` and `dtw_timings` corresponds to the order
    real words appear in `pred_text`.

    Injection is gated on line-monotonicity against EVERY timestamp token in
    the original text — including stray/orphan timestamps that carry no word
    (e.g. a bare line-start `<|t|>`), which the word/tag-level guard can't
    see. A fill that would go backwards vs the last timestamp on the line is
    skipped (sub-grid backsteps are clamped).
    """
    out: list[str] = []
    word_idx = 0
    last_ts: float | None = None  # last timestamp value emitted on this line
    for m in _PARSE_TOKEN.finditer(pred_text):
        s = m.group(0)
        if "\n" in s:
            last_ts = None
        if (
            _TIMESTAMP_FULL.fullmatch(s)
            or _SPEAKER_FULL.fullmatch(s)
            or _BRACKET_TAG_FULL.fullmatch(s)
        ):
            if s.startswith("<|"):
                t = float(s[2:-2])
                last_ts = t if last_ts is None else max(last_ts, t)
            out.append(s)
        elif s.startswith("<") and s.endswith(">"):
            out.append(s)  # metadata tag (<accent:..>, <emotion:..>, etc)
        elif s.isspace():
            out.append(s)
        else:
            # bare word
            if word_idx >= len(pred_words_flat):
                out.append(s)
                continue
            _w, dec_s, dec_e = pred_words_flat[word_idx]
            if dec_s is None and dec_e is None and dtw_timings[word_idx] is not None:
                ts, te = dtw_timings[word_idx]
                # Quantise to the 0.02 s WhisperDRZ grid.
                ts = max(0.0, round(ts * 50) / 50)
                te = max(ts, round(te * 50) / 50)
                if last_ts is not None and ts < last_ts:
                    if ts < last_ts - 0.02:
                        out.append(s)  # backwards vs the line — leave untimed
                        word_idx += 1
                        continue
                    ts = last_ts
                    te = max(te, ts)
                out.append(f"<|{ts:.2f}|>{s}<|{te:.2f}|>")
                last_ts = te
            else:
                out.append(s)
            word_idx += 1
    return "".join(out)


# ---------- Batched entry point (multi-slot finalisation) ----------


def _prepare_chunk(
    chunk_lines,
    tokenizer,
    pred_words_flat,
    tag_anchors_global,
    max_tokens,
):
    """Compute everything a single chunk needs for DTW that doesn't require
    GPU. Returns None if the chunk can't be processed (out of budget, etc).

    tag_anchors_global: list of (audio_s, audio_e, prev_gi_or_None, next_gi_or_None)
    from `parse_predicted_lines`. We translate global word indices to local
    tagged-list indices for tags whose adjacent words both land in this chunk.
    """
    rendered, tagged = _render_chunk(chunk_lines)
    ids, ext_words, ext_word_tokens = tokens_chars(rendered, tokenizer)
    if len(ids) < 3 or len(ids) > max_tokens:
        return None
    if len(tagged) != len(ext_words) - 1:
        return None
    # word-position anchors and a reverse global→local index map.
    word_anchors: list[tuple[int, float, float]] = []
    gi_to_local: dict[int, int] = {}
    for i, (orig_idx, _surf) in enumerate(tagged):
        if orig_idx is None:
            continue
        gi_to_local[orig_idx] = i
        _w, dec_s, dec_e = pred_words_flat[orig_idx]
        if dec_s is not None and dec_e is not None:
            word_anchors.append((i, dec_s, dec_e))
    # Tag anchors that survive into this chunk: keep ones whose declared
    # neighbour(s) are within this chunk. We don't require both — a tag at the
    # line boundary still contributes one side's constraint.
    tag_anchors: list[tuple[float, float, int | None, int | None]] = []
    if tag_anchors_global:
        for a_s, a_e, p_gi, n_gi in tag_anchors_global:
            p_local = gi_to_local.get(p_gi) if p_gi is not None else None
            n_local = gi_to_local.get(n_gi) if n_gi is not None else None
            if p_local is None and n_local is None:
                continue
            tag_anchors.append((a_s, a_e, p_local, n_local))
    return {
        "ids": ids,
        "ext_words": ext_words,
        "ext_word_tokens": ext_word_tokens,
        "tagged": tagged,
        "word_anchors": word_anchors,
        "tag_anchors": tag_anchors,
    }


def _build_full_tokens_batch(
    tokenizer, ids_list: list[list[int]], device, dtype=torch.long
) -> tuple[torch.Tensor, list[int]]:
    """Right-pad each chunk's (sot+no_ts+ids+eot) sequence to the max length;
    pad value is eot. Returns (batched tensor, list of actual lengths)."""
    sot_seq = list(tokenizer.sot_sequence)
    actual_lens = [len(sot_seq) + 1 + len(ids) + 1 for ids in ids_list]
    max_len = max(actual_lens)
    B = len(ids_list)
    out = torch.full((B, max_len), tokenizer.eot, dtype=dtype, device=device)
    for bi, ids in enumerate(ids_list):
        seq = [*sot_seq, tokenizer.no_timestamps, *list(ids), tokenizer.eot]
        out[bi, : len(seq)] = torch.tensor(seq, dtype=dtype, device=device)
    return out, actual_lens


def fill_word_timings_batch(
    model,
    tokenizer,
    slot_data: list[tuple[torch.Tensor, str, int]],
    *,
    head_mask: torch.Tensor | None = None,
    max_tokens: int = 440,
) -> list[str]:
    """Batched analog of fill_word_timings: ONE padded decoder forward across
    every (slot × chunk) finalising in this step, then per-chunk DTW on CPU.

    slot_data: list of (xa, pred_text, num_frames), one per finalising slot.
    Returns parallel list of augmented pred_texts.

    Why batched: at typical Transcriber slot counts (3-5 slots in a finalise
    batch), serial decoder-forward dominates. Packing them into one call cuts
    the GPU portion roughly Bx — the cross-attention K computation gets shared
    encoder-output work amortised across batch rows even though each row has
    its own xa.
    """
    n_slots = len(slot_data)
    if n_slots == 0:
        return []

    device = slot_data[0][0].device
    if head_mask is None:
        head_mask = make_head_mask(model)
    head_mask = head_mask.to(device)

    # Per-slot prep: parse, chunk, prepare each chunk.
    per_slot_lines: list = [None] * n_slots
    per_slot_flat: list = [None] * n_slots
    per_slot_tag_anchors: list = [None] * n_slots
    per_slot_chunk_recs: list[list] = [[] for _ in range(n_slots)]
    for si, (xa, pred_text, num_frames) in enumerate(slot_data):
        lines, flat, tag_anchors = parse_predicted_lines(pred_text)
        per_slot_lines[si] = lines
        per_slot_flat[si] = flat
        per_slot_tag_anchors[si] = tag_anchors
        if not flat:
            continue
        chunks = chunk_speaker_lines(lines, tokenizer, max_tokens)
        for chunk_lines in chunks:
            per_slot_chunk_recs[si].append(
                _prepare_chunk(chunk_lines, tokenizer, flat, tag_anchors, max_tokens)
            )

    # Per-slot audio bounds for each chunk (window-position units), so a
    # chunk's DTW searches only the sub-range implied by surrounding decoder
    # timestamps rather than being corner-to-corner mapped onto the full window.
    per_slot_bounds: list[list[tuple[float, float] | None]] = []
    for si, recs in enumerate(per_slot_chunk_recs):
        win_dur = (slot_data[si][2] // 2) / TOKENS_PER_SECOND
        per_slot_bounds.append(
            _chunk_audio_bounds(recs, per_slot_flat[si], win_dur) if recs else []
        )

    # Collect every non-None chunk into a flat list for batching.
    pooled: list[tuple[int, int, dict]] = []  # (slot_idx, chunk_idx, rec)
    ids_list, xa_list, num_frames_list, bounds_pos = [], [], [], []
    for si, recs in enumerate(per_slot_chunk_recs):
        for ci, rec in enumerate(recs):
            if rec is None:
                continue
            bnd = per_slot_bounds[si][ci]
            if bnd is None:
                continue
            pooled.append((si, ci, rec))
            ids_list.append(rec["ids"])
            xa_list.append(slot_data[si][0])
            num_frames_list.append(slot_data[si][2])
            bounds_pos.append(
                (
                    int(bnd[0] * TOKENS_PER_SECOND),
                    round(bnd[1] * TOKENS_PER_SECOND),
                )
            )

    # Default outputs: unchanged pred_text where there was nothing to do.
    outs: list[str] = [data[1] for data in slot_data]

    if not pooled:
        return outs

    # Single padded decoder forward, hooks fire once for the whole batch.
    full_tokens, actual_lens = _build_full_tokens_batch(tokenizer, ids_list, device)
    xa_batch = torch.stack(xa_list, dim=0)
    alignment_layers = {int(l) for l in head_mask.nonzero()[:, 0].tolist()}
    with CrossAttnCapture(model, layers=alignment_layers) as cap:
        with torch.inference_mode():
            model.decoder(full_tokens, xa_batch)
        full_weights = cap.compute_attention(head_mask)  # [B, H, T_text_pad, T_audio]
    # `compute_attention` returns [H, ...] if B==1; lift back to [1, H, ...].
    if full_weights.dim() == 3:
        full_weights = full_weights.unsqueeze(0)

    # Batched GPU prep: softmax + masked normalise + median + mean-over-heads on
    # the FULL [B, H, T_text_pad, T_audio] tensor, then ONE .cpu() transfer.
    # Replaces the per-chunk Python loop that forced n_pooled GPU↔CPU syncs.
    sot_len = len(tokenizer.sot_sequence)
    T_audio_full = full_weights.shape[-1]
    audio_caps_list = [
        max(0, min(int(nf) // 2, T_audio_full)) for nf in num_frames_list
    ]
    matrices_cpu = _dtw_prep_batch(full_weights, actual_lens, audio_caps_list)
    # Drop full_weights early — frees a multi-GB GPU tensor before the CPU phase.
    del full_weights

    # Parallel per-chunk DTW on CPU. Each pooled chunk is independent here;
    # numba's DTW kernel releases the GIL so a thread pool scales linearly with
    # cores. Anchor stitching (which has intra-slot ordering) runs sequentially
    # after this phase.
    def _dtw_one_chunk(bi: int) -> list[tuple[str, float, float]]:
        rec = pooled[bi][2]
        actual_len = actual_lens[bi]
        audio_cap = audio_caps_list[bi]
        lo, hi = bounds_pos[bi]
        lo = max(0, min(lo, audio_cap))
        hi = max(lo, min(hi, audio_cap))
        if hi - lo < 2:
            return []
        matrix = matrices_cpu[bi, sot_len : actual_len - 1, lo:hi]
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            return []
        neg_matrix = np.ascontiguousarray(-matrix, dtype=np.float32)
        if rec.get("word_anchors") or rec.get("tag_anchors"):
            _apply_anchor_bands(
                neg_matrix,
                rec["ext_words"],
                rec["ext_word_tokens"],
                hi - lo,
                rec.get("word_anchors"),
                rec.get("tag_anchors"),
                slack_frames=7,
                col_offset=lo,
            )
        result = _dtw_from_neg_matrix(
            neg_matrix, rec["ext_words"], rec["ext_word_tokens"]
        )
        if lo > 0 and result:
            off = lo / TOKENS_PER_SECOND
            result = [(w, s + off, e + off) for w, s, e in result]
        return result

    if len(pooled) <= 1:
        dtw_results = [_dtw_one_chunk(bi) for bi in range(len(pooled))]
    else:
        dtw_results = list(_DTW_POOL.map(_dtw_one_chunk, range(len(pooled))))

    # Per-slot assembly. No stitch shift: each chunk's DTW ran on its own
    # bounded audio sub-range, so times are already window-absolute. (The old
    # anchor/prev-end shift chained errors across chunks and pushed fills up
    # to a full window-length past the real audio.)
    per_slot_out_timings: list[list[tuple[float, float] | None] | None] = [
        None
    ] * n_slots

    for bi, (si, _ci, rec) in enumerate(pooled):
        dtw = dtw_results[bi]
        if not dtw:
            continue
        flat = per_slot_flat[si]
        if per_slot_out_timings[si] is None:
            per_slot_out_timings[si] = [None] * len(flat)
        tagged = rec["tagged"]
        for i, (orig_idx, _surf) in enumerate(tagged):
            if orig_idx is None or i >= len(dtw):
                continue
            _w, ds, de = dtw[i]
            per_slot_out_timings[si][orig_idx] = (float(ds), float(de))

    for si in range(n_slots):
        if per_slot_out_timings[si] is None:
            continue
        win_dur = (slot_data[si][2] // 2) / TOKENS_PER_SECOND
        _drop_bad_fills(
            per_slot_lines[si],
            per_slot_flat[si],
            per_slot_out_timings[si],
            win_dur,
            tag_anchors=per_slot_tag_anchors[si],
        )
        outs[si] = rebuild_with_timings(
            slot_data[si][1], per_slot_flat[si], per_slot_out_timings[si]
        )
    return outs


# ---------- Standalone helper (no Transcriber needed) ----------


_MEL_SINGLETON: dict[tuple[int, str], LogMelSpectrogram] = {}

