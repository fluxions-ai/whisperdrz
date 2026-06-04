from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
from torch import Tensor


def exact_div(x, y):
    assert x % y == 0
    return x // y


# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
SR = SAMPLE_RATE
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token


def get_audio_duration(file_path: str | Path) -> float | None:
    """Get the duration of an audio file in seconds.

    Uses torchcodec for efficient metadata reading without decoding the entire file.

    Args:
        file_path: Path to the audio file

    Returns:
        Duration in seconds, or None if the file cannot be read
    """
    try:
        from torchcodec._core import get_container_metadata_from_header

        metadata = get_container_metadata_from_header(file_path)
        # Try to get audio stream duration
        if metadata.best_audio_stream_index is not None:
            audio_stream = metadata.streams[metadata.best_audio_stream_index]
            return audio_stream.duration_seconds_from_header
        return None
    except Exception:
        return None


def load_audio(
    path: str | Path | bytes, sample_rate: int | None = None
) -> tuple[torch.Tensor, int]:
    """Load audio file using TorchCodec.

    Args:
        path: Path to audio file, or bytes/BytesIO with audio data
        sample_rate: Target sample rate (resamples on-the-fly if provided)

    Returns:
        Tuple of (audio_tensor, sample_rate) where audio_tensor is shape [channels, samples]
    """
    from io import BytesIO

    from torchcodec.decoders import AudioDecoder

    # Handle BytesIO by reading bytes
    if isinstance(path, BytesIO):
        path = path.read()

    # AudioDecoder accepts str (path) or bytes
    if isinstance(path, bytes):
        decoder = AudioDecoder(path, sample_rate=sample_rate)
    else:
        decoder = AudioDecoder(str(path), sample_rate=sample_rate)
    samples = decoder.get_all_samples()
    # TorchCodec returns [num_channels, num_samples]
    return samples.data, samples.sample_rate


class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        n_mels: int = 128,
        sample_rate: int = 16_000,
        n_fft: int = 400,
        hop_length: int = 160,
        norm: str = "slaney",
        mel_scale: str = "slaney",
    ):
        super().__init__()

        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            n_stft=self.n_fft // 2 + 1,
            norm=norm,
            mel_scale=mel_scale,
        )

        self.register_buffer("window", torch.hann_window(self.n_fft))

    def forward(self, waveform, target_frames: int | None = None):
        # Handle different input types: single tensor, batched tensor, or list of tensors
        if isinstance(waveform, list):
            # List of audio tensors (potentially different lengths)
            mel_results = []
            for audio in waveform:
                mel = self._process_single_audio(audio, target_frames)
                mel_results.append(mel)
            return torch.stack(mel_results)
        # Single tensor or batched tensor
        is_batched = waveform.dim() > 1
        if not is_batched:
            return self._process_single_audio(waveform, target_frames)
        # Batched tensor - process each audio in the batch
        mel_results = []
        for i in range(waveform.shape[0]):
            mel = self._process_single_audio(waveform[i], target_frames)
            mel_results.append(mel)
        return torch.stack(mel_results)

    def _process_single_audio(self, audio: Tensor, target_frames: int | None = None):
        """Process a single audio tensor with optional target_frames padding."""
        mel_spec = self._compute_mel_spec(audio)
        if target_frames is None:
            return mel_spec
        current_frames = mel_spec.shape[-1]
        if current_frames < target_frames:
            frames_to_pad = target_frames - current_frames
            return torch.nn.functional.pad(
                mel_spec, (0, frames_to_pad), mode="constant", value=-1.5
            )
        return mel_spec[..., :target_frames]

    def _compute_mel_spec(self, waveform: Tensor):
        """Helper method to compute mel spectrogram from waveform."""
        stft = torch.stft(
            waveform,
            self.n_fft,
            self.hop_length,
            window=self.window,
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs() ** 2

        mel_spec = self.mel_scale(magnitudes)

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        return (log_spec + 4.0) / 4.0
