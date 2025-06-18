from modules.audio_extractor import extract_audio_from_video
from modules.translator import translate_malayalam_to_english
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
import soundfile as sf
import torch

# ─── CONFIG ───
VIDEO_PATH = r"C:\\Users\\HP\\Desktop\\Meeting_2.mp4"
AUDIO_PATH = "audio.wav"

# ─── LOAD ASR MODEL ───
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
asr_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")


def transcribe_malayalam(audio_path: str) -> str:
    """Transcribe Malayalam speech to text using Whisper without warnings."""
    speech, sr = sf.read(audio_path)
    inputs = processor(speech, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        generated_ids = asr_model.generate(
            input_features=inputs.input_features,
        )
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def main():
    print("🎬 Extracting audio...")
    extract_audio_from_video(VIDEO_PATH, AUDIO_PATH)

    print("🧠 Transcribing Malayalam...")
    ml_text = transcribe_malayalam(AUDIO_PATH)
    print("\n📝 Malayalam transcript:\n", ml_text)

    print("🌍 Translating to English...")
    en_text = translate_malayalam_to_english(ml_text)
    print("\n📝 English translation:\n", en_text)


if __name__ == "__main__":
    main()
