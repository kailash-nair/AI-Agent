from modules.audio_extractor import extract_audio_from_video
from modules.translator import translate_malayalam_to_english
from transformers import pipeline

# ─── CONFIG ───
VIDEO_PATH = r"C:\\Users\\HP\\Desktop\\Meeting_2.mp4"
AUDIO_PATH = "audio.wav"

# ─── LOAD ASR PIPELINE ───
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny",
    device=-1,
    chunk_length_s=30,
)


def transcribe_malayalam(audio_path: str) -> str:
    """Transcribe Malayalam speech to text using Whisper."""
    result = asr_pipeline(audio_path)
    return result["text"]


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
