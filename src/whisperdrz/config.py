from dataclasses import dataclass, fields


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int

    @classmethod
    def from_name(cls, name: str) -> "ModelDimensions":
        base_name = name.replace(".en", "")

        dims_by_name = {
            "large-v3": cls(
                n_mels=128,
                n_audio_ctx=1500,
                n_audio_state=1280,
                n_audio_head=20,
                n_audio_layer=32,
                n_vocab=51866,
                n_text_ctx=448,
                n_text_state=1280,
                n_text_head=20,
                n_text_layer=32,
            ),
        }

        if base_name not in dims_by_name:
            raise ValueError(
                f"Unknown model name: {name}. Supported: {list(dims_by_name.keys())}"
            )

        return dims_by_name[base_name]


@dataclass
class Config:
    """Inference-relevant subset of the training config stored in checkpoints.

    Unknown keys in checkpoint config dicts are ignored.
    """

    base: str = "large-v3"

    # Architecture flags
    timing_embedding: bool = False
    speaker_embedding_enabled: bool = False
    speaker_embedding_dim: int = 256
    speaker_head_layers: int = 2

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

