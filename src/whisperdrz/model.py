from __future__ import annotations

import base64
import gzip
import os
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, autocast

from whisperdrz.config import Config, ModelDimensions
from whisperdrz.tokenizer import get_tokenizer
from whisperdrz.utils import load_what_you_can

DEFAULT_REPO = "fluxions/whisperdrz"


def _sdpa_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    cache_seqlens=None,
    cache_batch_idx=None,
    softmax_scale=None,
    causal=False,
):
    """Pure-torch fallback for ``flash_attn_with_kvcache`` (SDPA-based).

    Mirrors the subset of the flash-attn API the decoder uses so inference runs
    on devices without flash-attn (CPU, or GPUs lacking a flash-attn build):
    - self-attention: appends ``k``/``v`` at each slot's ``cache_seqlens`` and
      attends causally over the cache up to the new tokens;
    - cross-attention (``k`` is None): attends over the full cache, read-only.

    All q/k/v and caches are ``[batch_or_slots, seq, n_heads, head_dim]``;
    ``cache_batch_idx`` maps the batch dim to cache slots.
    """
    B, sq, _, _ = q.shape
    idx = (
        torch.arange(B, device=q.device)
        if cache_batch_idx is None
        else cache_batch_idx.long()
    )

    if k is not None:
        cs = cache_seqlens.long()
        for i in range(B):
            s, p = int(idx[i]), int(cs[i])
            k_cache[s, p : p + sq] = k[i]
            v_cache[s, p : p + sq] = v[i]
        max_len = int((cs + sq).max())
        keys = k_cache[idx, :max_len]
        vals = v_cache[idx, :max_len]
        jj = torch.arange(max_len, device=q.device)
        tt = torch.arange(sq, device=q.device)
        # query t (absolute position cs+t) may attend to key positions <= cs+t
        attn_mask = (jj[None, None, :] <= (cs[:, None, None] + tt[None, :, None]))[
            :, None
        ]
    else:
        keys = k_cache[idx]
        vals = v_cache[idx]
        attn_mask = None

    out = F.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3),
        keys.permute(0, 2, 1, 3),
        vals.permute(0, 2, 1, 3),
        attn_mask=attn_mask,
        scale=softmax_scale,
    )
    return out.permute(0, 2, 1, 3)


try:
    from flash_attn import flash_attn_with_kvcache as _flash_impl

    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    _flash_impl = None


def flash_attn_with_kvcache(q, *args, **kwargs):
    """Dispatch to flash-attn on CUDA when available, else the SDPA fallback.

    flash-attn only supports CUDA tensors, so CPU inference always uses SDPA.
    """
    if HAS_FLASH_ATTN and q.is_cuda:
        return _flash_impl(q, *args, **kwargs)
    return _sdpa_attn_with_kvcache(q, *args, **kwargs)

# Sequence length buckets for compiled decode
BUCKETS = [32, 64, 128, 256, 448]


class FlashKVCache(nn.Module):
    """KV cache optimized for flash_attn_with_kvcache.

    Uses shape [batch_size, max_seq, n_heads, head_dim] which flash_attn expects.
    Tracks per-slot sequence lengths for continuous batching.
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_length: int,
        n_heads: int,
        head_dim: int,
        dtype=torch.float16,
        device=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.n_heads = n_heads
        self.head_dim = head_dim

        self.register_buffer(
            "k_cache",
            torch.zeros(
                batch_size,
                max_seq_length,
                n_heads,
                head_dim,
                dtype=dtype,
                device=device,
            ),
        )
        self.register_buffer(
            "v_cache",
            torch.zeros(
                batch_size,
                max_seq_length,
                n_heads,
                head_dim,
                dtype=dtype,
                device=device,
            ),
        )
        self.register_buffer(
            "seq_lens",
            torch.zeros(batch_size, dtype=torch.int32, device=device),
        )
        self.done = False
        self.cross_attn_len = 0


    def prefill(
        self,
        k: Tensor,
        v: Tensor,
        start_pos: int = 0,
        slot_indices: Tensor | None = None,
    ):
        """Prefill cache. k,v: [B, seq, heads, head_dim]"""
        seq_len = k.shape[1]
        end_pos = start_pos + seq_len
        if slot_indices is not None:
            self.k_cache[slot_indices, start_pos:end_pos] = k
            self.v_cache[slot_indices, start_pos:end_pos] = v
            self.seq_lens[slot_indices] = end_pos
        else:
            B = k.shape[0]
            self.k_cache[:B, start_pos:end_pos] = k
            self.v_cache[:B, start_pos:end_pos] = v
            self.seq_lens[:B] = end_pos

    def set_cross_attention(
        self, k: Tensor, v: Tensor, slot_indices: Tensor | None = None
    ):
        """Set cross-attention cache. k,v: [B, seq, heads, head_dim]"""
        seq_len = k.shape[1]
        if slot_indices is not None:
            self.k_cache[slot_indices, :seq_len] = k
            self.v_cache[slot_indices, :seq_len] = v
        else:
            B = k.shape[0]
            self.k_cache[:B, :seq_len] = k
            self.v_cache[:B, :seq_len] = v
        self.cross_attn_len = seq_len
        self.done = True

    def reset(self, slot_indices: Tensor | None = None):
        """Reset cache for new batch."""
        if slot_indices is not None:
            self.seq_lens[slot_indices] = 0
        else:
            self.seq_lens.zero_()
        self.done = False
        self.cross_attn_len = 0


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            with autocast("cuda", enabled=True, dtype=torch.float32):
                return super().forward(x).type_as(x)
        return super().forward(x)


# Use standard PyTorch layers - autocast handles dtype conversions
Linear = nn.Linear
Conv1d = nn.Conv1d


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


def timing_sinusoids(
    n_timestamps: int = 1501, channels: int = 1280, max_time: float = 30.0
):
    """Returns sinusoidal embeddings for timing tokens based on actual time values.

    Unlike positional sinusoids which encode sequence position (0, 1, 2, ...),
    these encode actual time values in seconds (0.0, 0.02, 0.04, ..., 30.0).
    This gives the model explicit temporal semantics - nearby timestamps
    will have similar embeddings.

    Args:
        n_timestamps: Number of timing tokens (default 1501 for 0.00 to 30.00 at 0.02s)
        channels: Embedding dimension
        max_time: Maximum time value in seconds

    Returns:
        Tensor of shape [n_timestamps, channels]
    """
    assert channels % 2 == 0
    time_values = torch.linspace(0, max_time, n_timestamps)
    log_timescale_increment = np.log(10000) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = time_values[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_state: int,
        n_head: int,
        causal: bool = False,
        flash: bool = True,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.causal = causal
        self.flash = flash

        self.n_head = n_head
        self.n_state = n_state
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: Tensor, xa: Tensor | None = None, mask: Tensor | None = None):
        """Full (uncached) attention, used by the encoder.

        xa is the cross-attention input (encoder output); None for self-attention.
        """
        q = self.query(x)
        k = self.key(x if xa is None else xa)
        v = self.value(x if xa is None else xa)
        return self.qkv_attention(q, k, v, mask)


    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None
    ):
        _, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25

        q = rearrange(q, "B T (h d) -> B h T d", h=self.n_head)
        k = rearrange(k, "B T (h d) -> B h T d", h=self.n_head)
        v = rearrange(v, "B T (h d) -> B h T d", h=self.n_head)

        is_causal = self.causal if self.training else self.causal and n_ctx > 1

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=is_causal,
            attn_mask=mask,
            scale=scale * scale,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = rearrange(out, "B h T d -> B T (h d)")
        return self.dropout(self.out(out))


    def forward_flash(
        self,
        x: Tensor,
        kv_cache: FlashKVCache,
        is_cross_attn: bool = False,
        slot_indices: Tensor | None = None,
    ) -> Tensor:
        """Forward using flash_attn_with_kvcache (no masks, uses cache_seqlens).

        Args:
            x: [B, seq, n_state] input (seq=1 for decode)
            kv_cache: FlashKVCache with [BS, max_seq, heads, head_dim] layout
            is_cross_attn: True for cross-attention (read-only from cache)
            slot_indices: Optional [B] tensor mapping input batch to cache slots

        Returns:
            [B, seq, n_state] attention output
        """
        B, seq_len, _ = x.shape
        head_dim = self.n_state // self.n_head
        scale = head_dim**-0.5

        q = self.query(x).view(B, seq_len, self.n_head, head_dim)

        # Get cache_seqlens for the slots we're accessing
        cache_seqlens = (
            kv_cache.seq_lens
            if slot_indices is None
            else kv_cache.seq_lens[slot_indices]
        )
        # cache_batch_idx must be int32 for flash_attn
        cache_batch_idx = (
            slot_indices.to(torch.int32) if slot_indices is not None else None
        )

        if is_cross_attn:
            out = flash_attn_with_kvcache(
                q,
                kv_cache.k_cache,
                kv_cache.v_cache,
                cache_batch_idx=cache_batch_idx,
                softmax_scale=scale,
                causal=False,
            )
        else:
            k_new = self.key(x).view(B, seq_len, self.n_head, head_dim)
            v_new = self.value(x).view(B, seq_len, self.n_head, head_dim)

            out = flash_attn_with_kvcache(
                q,
                kv_cache.k_cache,
                kv_cache.v_cache,
                k=k_new,
                v=v_new,
                cache_seqlens=cache_seqlens,
                cache_batch_idx=cache_batch_idx,
                softmax_scale=scale,
                causal=self.causal,
            )

        out = out.view(B, seq_len, self.n_state)
        return self.dropout(self.out(out))


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        n_state: int,
        n_head: int,
        cross_attention: bool = False,
        causal: bool = False,
        flash: bool = True,
        dropout_p: float = 0.0,
    ):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head, causal=causal, flash=flash)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head, causal=False, flash=flash)
            if cross_attention
            else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp),
            nn.GELU(),
            Linear(n_mlp, n_state),
            nn.Dropout(p=dropout_p),
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(self, x: Tensor, xa: Tensor = None, mask: Tensor = None):
        """Full (uncached) forward, used by the encoder."""
        x = x + self.attn(self.attn_ln(x), xa=None, mask=mask)
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa)
        return x + self.mlp(self.mlp_ln(x))


    def forward_flash(
        self,
        x: Tensor,
        self_kv: FlashKVCache,
        cross_kv: FlashKVCache,
        slot_indices: Tensor | None = None,
    ):
        """Forward using flash_attn_with_kvcache (no masks needed).

        Args:
            x: [B, seq, n_state] input
            self_kv: FlashKVCache for self-attention
            cross_kv: FlashKVCache for cross-attention
            slot_indices: Optional batch indices for specific slots

        Note: Caller must update seq_lens after all layers have processed.
        """
        x = x + self.attn.forward_flash(
            self.attn_ln(x),
            self_kv,
            is_cross_attn=False,
            slot_indices=slot_indices,
        )
        if self.cross_attn:
            x = x + self.cross_attn.forward_flash(
                self.cross_attn_ln(x),
                cross_kv,
                is_cross_attn=True,
                slot_indices=slot_indices,
            )
        return x + self.mlp(self.mlp_ln(x))


class AudioEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        flash: bool = True,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    n_state, n_head, causal=False, flash=flash, dropout_p=dropout_p
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_post = LayerNorm(n_state)
    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        x = x + self.positional_embedding[: x.shape[1]].to(x.dtype)

        for block in self.blocks:
            x = block(x)

        return self.ln_post(x)


class TextDecoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        flash: bool = True,
        dropout_p: float = 0.0,
        timestamp_begin: int | None = None,
        n_timestamps: int = 1501,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        # Pre-compute position indices buffer to avoid repeated torch.arange calls
        self.register_buffer(
            "position_ids",
            torch.arange(n_ctx, dtype=torch.long).unsqueeze(0),
            persistent=False,
        )

        # Timing embeddings: sinusoidal encoding based on actual time values
        # When timestamp_begin is set, timing tokens get additional temporal embedding
        self.timestamp_begin = timestamp_begin
        self.n_timestamps = n_timestamps
        if timestamp_begin is not None:
            self.register_buffer(
                "timing_embedding",
                timing_sinusoids(n_timestamps, n_state),
                persistent=False,
            )
        else:
            self.timing_embedding = None

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    n_state,
                    n_head,
                    causal=True,
                    cross_attention=True,
                    flash=flash,
                    dropout_p=dropout_p,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)
        self._embedding_weight_T: Tensor | None = None
    def cache_embedding_weight(self, dtype=torch.float16):
        """Cache transposed embedding weight to avoid repeated dtype conversion."""
        self._embedding_weight_T = self.token_embedding.weight.to(dtype).T.contiguous()

    def _get_timing_embedding(self, token_ids: Tensor) -> Tensor:
        """Timing embeddings for timestamp tokens; zeros for everything else."""
        if self.timing_embedding is None or self.timestamp_begin is None:
            return 0
        timestamp_end = self.timestamp_begin + self.n_timestamps
        is_timing = (token_ids >= self.timestamp_begin) & (token_ids < timestamp_end)
        timing_indices = (token_ids - self.timestamp_begin).clamp(
            0, self.n_timestamps - 1
        )
        timing_emb = self.timing_embedding[timing_indices]
        return timing_emb * is_timing.unsqueeze(-1).to(timing_emb.dtype)

    def forward(
        self, x: Tensor, xa: Tensor, pos: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Full (uncached) parallel forward over all tokens.

        Used for DTW word alignment, which runs the decoder over the predicted
        tokens with cross-attention hooks. Self-attention is causal (applied
        inside qkv_attention via is_causal).
        """
        if pos is None:
            pos = self.position_ids[:, : x.shape[1]].expand(x.shape[0], -1)
        token_ids = x
        x = self.token_embedding(x) + self.positional_embedding[pos]
        x = x + self._get_timing_embedding(token_ids)
        for block in self.blocks:
            x = block(x, xa)
        x = self.ln(x)
        if self._embedding_weight_T is not None:
            logits = x @ self._embedding_weight_T
        else:
            logits = x @ self.token_embedding.weight.to(x.dtype).T
        return logits, x

    def forward_flash(
        self,
        tokens: Tensor,
        self_kv: list[FlashKVCache],
        cross_kv: list[FlashKVCache],
        positions: Tensor,
        slot_indices: Tensor | None = None,
        return_hidden_states: bool = False,
    ):
        """Forward using flash_attn_with_kvcache (no masks needed).

        Args:
            tokens: [B, seq] token ids
            self_kv: List of FlashKVCache for self-attention (one per layer)
            cross_kv: List of FlashKVCache for cross-attention (one per layer)
            positions: [B, seq] or [B] position indices
            slot_indices: Optional batch indices for specific slots
            return_hidden_states: If True, also return hidden states for speaker embedding

        Returns:
            logits: [B, seq, vocab_size]
            hidden_states: [B, seq, n_text_state] (only if return_hidden_states=True)
        """
        B, seq_len = tokens.shape
        if positions.dim() == 1:
            positions = positions.unsqueeze(1).expand(B, seq_len)

        x = self.token_embedding(tokens) + self.positional_embedding[positions]

        for block, s_kv, c_kv in zip(self.blocks, self_kv, cross_kv, strict=True):
            x = block.forward_flash(x, s_kv, c_kv, slot_indices=slot_indices)

        # Update seq_lens in all layers after processing
        for s_kv in self_kv:
            if slot_indices is not None:
                s_kv.seq_lens[slot_indices] += seq_len
            else:
                s_kv.seq_lens[:B] += seq_len

        x = self.ln(x)
        if self._embedding_weight_T is not None:
            logits = x @ self._embedding_weight_T
        else:
            logits = x @ self.token_embedding.weight.to(x.dtype).T

        if return_hidden_states:
            return logits, x
        return logits


class SpeakerEmbeddingHead(nn.Module):
    """Projects decoder hidden states to speaker embedding space.

    Extracts embeddings at [N] digit positions for contrastive learning.
    Uses LayerNorm (DDP-safe) between layers, L2-normalized output.
    """

    def __init__(self, hidden_dim: int, embedding_dim: int = 512, num_layers: int = 3):
        super().__init__()
        layers = []
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                ]
            )
        layers.append(nn.Linear(hidden_dim, embedding_dim))
        self.head = nn.Sequential(*layers)

    def forward(self, hidden_states: Tensor) -> Tensor:
        embeddings = self.head(hidden_states)
        return F.normalize(embeddings, p=2, dim=-1)


class WhisperDRZ(nn.Module):
    "Akroates, Greek for listen"

    def __init__(
        self,
        dims: ModelDimensions,
        dropout_p: float = 0.0,
        flash: bool = True,
        config: Config = None,
    ):
        super().__init__()
        self.dims = dims
        self.config = config
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
            flash=flash,
            dropout_p=dropout_p,
        )
        timestamp_begin = None
        if config is not None and config.timing_embedding:
            tokenizer = get_tokenizer(multilingual=True)
            timestamp_begin = tokenizer.timestamp_begin
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
            flash=flash,
            dropout_p=dropout_p,
            timestamp_begin=timestamp_begin,
        )

        # use the last half among the decoder layers for time alignment by default;
        # to use a specific set of heads, see `set_alignment_heads()` below.
        all_heads = torch.zeros(
            self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        # self.register_buffer("alignment_heads", all_heads, persistent=False)
        # self.set_alignment_heads(_ALIGNMENT_HEADS[config.base])

        # Pre-computed cu_seq_cross for varlen attention (lazily sized)
        self._cu_cross_buf: Tensor | None = None

        # Speaker embedding head (optional)
        self.speaker_head = None
        if config is not None and config.speaker_embedding_enabled:
            self.speaker_head = SpeakerEmbeddingHead(
                hidden_dim=self.dims.n_text_state,
                embedding_dim=config.speaker_embedding_dim,
                num_layers=config.speaker_head_layers,
            )

    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        )
        self.register_buffer("alignment_heads", mask, persistent=False)


    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def is_multilingual(self):
        return self.dims.n_vocab >= 51865

    @property
    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)

    @property
    def tokenizer(self):
        return get_tokenizer(
            self.is_multilingual, num_languages=self.num_languages, language="en"
        )

    @staticmethod
    def from_pretrained(
        path: str,
        dtype: torch.dtype | None = None,
    ) -> WhisperDRZ:
        """Load an WhisperDRZ checkpoint.

        Accepts:
        - a local ``.safetensors`` or ``.pt`` file path
        - a filename hosted in the default Hugging Face repo (``fluxions/whisperdrz``)
        - a Hugging Face repo id (e.g. ``"fluxions/whisperdrz"``), which loads
          ``model.safetensors`` from that repo
        """
        if not os.path.exists(path):
            from huggingface_hub import hf_hub_download

            if path.endswith((".pt", ".safetensors")):
                path = hf_hub_download(repo_id=DEFAULT_REPO, filename=path)
            else:
                path = hf_hub_download(repo_id=path, filename="model.safetensors")

        if path.endswith(".safetensors"):
            import json

            from safetensors import safe_open
            from safetensors.torch import load_file as safetensors_load

            state = safetensors_load(path, device="cpu")
            with safe_open(path, framework="pt") as f:
                metadata = f.metadata() or {}
            config = Config.from_dict(json.loads(metadata.get("config", "{}")))
        else:
            ckpt = torch.load(path, weights_only=False, map_location="cpu")
            config = Config.from_dict(ckpt["config"])
            state = ckpt["model"]

        dims = ModelDimensions.from_name(config.base)
        dims.n_text_ctx = state["decoder.positional_embedding"].shape[0]
        # Size vocab from the checkpoint's token embedding (may include extra
        # conditioning tokens beyond the base vocab).
        dims.n_vocab = state["decoder.token_embedding.weight"].shape[0]
        model = WhisperDRZ(dims, config=config)

        if dtype:
            state = {
                k: v.to(dtype) if v.is_floating_point() else v for k, v in state.items()
            }
            model = model.to(dtype)
        load_what_you_can(state, model)
        return model

    @staticmethod
    def from_pretrained_inf(ckpt: str) -> WhisperDRZ:
        return WhisperDRZ.from_pretrained(ckpt, dtype=torch.float16).eval()


    def create_flash_kv(
        self, batch_size: int, max_tokens: int | None = None
    ) -> tuple[list[FlashKVCache], list[FlashKVCache]]:
        """Create FlashKVCache lists for flash_attn_with_kvcache.

        Returns:
            self_kv: List of FlashKVCache for self-attention (one per layer)
            cross_kv: List of FlashKVCache for cross-attention (one per layer)
        """
        n_head = self.dims.n_text_head
        head_dim = self.dims.n_text_state // n_head
        text_ctx = max_tokens or self.dims.n_text_ctx

        self_kv = []
        cross_kv = []

        for _ in self.decoder.blocks:
            self_kv.append(
                FlashKVCache(
                    batch_size,
                    text_ctx,
                    n_head,
                    head_dim,
                    dtype=self.dtype,
                    device=self.device,
                )
            )
            cross_kv.append(
                FlashKVCache(
                    batch_size,
                    self.dims.n_audio_ctx,
                    n_head,
                    head_dim,
                    dtype=self.dtype,
                    device=self.device,
                )
            )

        return self_kv, cross_kv

    def reset_flash_kv(
        self,
        self_kv: list[FlashKVCache],
        cross_kv: list[FlashKVCache],
        slot_indices: Tensor | None = None,
    ):
        """Reset FlashKVCache for new batch or specific slots."""
        for s_kv, c_kv in zip(self_kv, cross_kv, strict=True):
            s_kv.reset(slot_indices)
            c_kv.reset(slot_indices)

    def prefill_flash(
        self,
        tokens: Tensor,
        xa: Tensor,
        self_kv: list[FlashKVCache],
        cross_kv: list[FlashKVCache],
        slot_indices: Tensor | None = None,
        return_hidden_states: bool = False,
    ):
        """Prefill using flash attention path.

        Args:
            tokens: [B, seq] initial tokens
            xa: [B, audio_ctx, audio_state] encoder output
            self_kv: Self-attention KV caches
            cross_kv: Cross-attention KV caches
            slot_indices: Optional batch indices for specific slots
            return_hidden_states: If True, also return hidden states

        Returns:
            logits: [B, seq, vocab_size]
            hidden_states: [B, seq, n_text_state] (only if return_hidden_states=True)
        """
        B, seq_len = tokens.shape
        n_head = self.dims.n_text_head
        head_dim = self.dims.n_text_state // n_head

        # Compute K, V for cross-attention from encoder output
        for layer_idx, block in enumerate(self.decoder.blocks):
            k_cross = block.cross_attn.key(xa).view(B, -1, n_head, head_dim)
            v_cross = block.cross_attn.value(xa).view(B, -1, n_head, head_dim)
            cross_kv[layer_idx].set_cross_attention(k_cross, v_cross, slot_indices)

        # Set initial sequence lengths
        for s_kv in self_kv:
            if slot_indices is not None:
                s_kv.seq_lens[slot_indices] = 0
            else:
                s_kv.seq_lens[:B] = 0

        positions = (
            torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(B, -1)
        )
        return self.decoder.forward_flash(
            tokens, self_kv, cross_kv, positions, slot_indices, return_hidden_states
        )

