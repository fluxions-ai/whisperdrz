from whisperdrz.model import WhisperDRZ
from whisperdrz.transcriber import Transcriber

__version__ = "0.1.0"

__all__ = ["WhisperDRZ", "Transcriber", "load_model", "__version__"]


def load_model(name_or_path: str, lang: str | None = "en", **kwargs) -> Transcriber:
    """Load a WhisperDRZ model for transcription.

    Args:
        name_or_path: A local checkpoint path (``.safetensors`` or ``.pt``), a
            filename hosted in the default Hugging Face repo, or a Hugging Face
            repo id.
        lang: Force a language (e.g. ``"en"``). Pass ``None`` to auto-detect.
        **kwargs: Forwarded to :class:`~whisperdrz.transcriber.Transcriber`.

    Returns:
        A :class:`~whisperdrz.transcriber.Transcriber` ready to ``transcribe(audio)``.
    """
    return Transcriber(model_path=name_or_path, lang=lang, **kwargs)
