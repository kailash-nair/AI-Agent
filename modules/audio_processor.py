import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

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
