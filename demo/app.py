import argparse

import gradio as gr
import torch

from whisperdrz import load_model
from whisperdrz.audio import SAMPLE_RATE
from whisperdrz.clean import TIMINGS_FORMAT

_transcriber = None


def _format(result) -> str:
    lines = []
    for seg in result.segments:
        text = TIMINGS_FORMAT.sub("", seg["text"]).strip()
        lines.append(f"[{seg['speaker']}] ({seg['start']:.2f}s - {seg['end']:.2f}s): {text}")
    return "\n".join(lines)


def transcribe(audio_path):
    if audio_path is None:
        return "No audio provided.", ""
    from torchcodec.decoders import AudioDecoder

    samples = AudioDecoder(audio_path, sample_rate=SAMPLE_RATE, num_channels=1)
    audio = samples.get_all_samples().data.squeeze(0)
    result = _transcriber.transcribe(audio)
    return _format(result), result.text


def build() -> gr.Blocks:
    with gr.Blocks(title="WhisperDRZ") as demo:
        gr.Markdown("# WhisperDRZ transcription")
        audio_in = gr.Audio(type="filepath", sources=["upload", "microphone"])
        btn = gr.Button("Transcribe", variant="primary")
        out = gr.Textbox(label="Transcription", lines=15)
        raw = gr.Textbox(label="Raw (with timestamps)", lines=10)
        btn.click(fn=transcribe, inputs=[audio_in], outputs=[out, raw])
    return demo


def main():
    global _transcriber
    parser = argparse.ArgumentParser(description="WhisperDRZ gradio demo")
    parser.add_argument(
        "--model", default="whisperdrz-large-v3.safetensors", help="Checkpoint or HF repo id"
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    _transcriber = load_model(args.model, lang="en")
    with torch.inference_mode():
        build().launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
