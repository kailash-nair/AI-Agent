import os
from typing import List

import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Expose translation utility here for backward compatibility.
# Some older code expects translate_malayalam_to_english in this module.


SAMPLE_RATE = 16000

# ASR model/processor
processor_asr = Wav2Vec2Processor.from_pretrained(
    "gvs/wav2vec2-large-xlsr-malayalam"
)
model_asr = Wav2Vec2ForCTC.from_pretrained(
    "gvs/wav2vec2-large-xlsr-malayalam"
)
model_asr.eval()


def load_audio(audio_path: str) -> torch.Tensor:
    """Load an audio file as a mono waveform tensor at SAMPLE_RATE."""
    waveform, sr = torchaudio.load(audio_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(
            waveform
        )
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform.squeeze()


def transcribe_malayalam(audio_path: str) -> str:
    """Transcribe Malayalam speech using the Wav2Vec2 model."""
    waveform = load_audio(audio_path)
    inputs = processor_asr(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        logits = model_asr(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
    return processor_asr.batch_decode(predicted_ids)[0]


def split_audio(
    audio_path: str, output_dir: str, chunk_duration: int = 30
) -> List[str]:
    """Split ``audio_path`` into chunks of ``chunk_duration`` seconds.

    Each chunk is saved as a separate WAV file in ``output_dir`` and the
    list of file paths is returned.
    """

    waveform = load_audio(audio_path)
    os.makedirs(output_dir, exist_ok=True)
    total_samples = waveform.shape[-1]
    chunk_samples = SAMPLE_RATE * chunk_duration
    chunk_paths: List[str] = []
    for i in range(0, total_samples, chunk_samples):
        chunk = waveform[i : i + chunk_samples]
        idx = i // chunk_samples
        chunk_path = os.path.join(output_dir, f"chunk_{idx}.wav")
        torchaudio.save(chunk_path, chunk.unsqueeze(0), SAMPLE_RATE)
        chunk_paths.append(chunk_path)
    return chunk_paths
