from modules.audio_extractor import extract_audio_from_video
from modules.audio_processor import transcribe_malayalam
from modules.translator import translate_malayalam_to_english

# ─── CONFIG ───
VIDEO_PATH = r"C:\\Users\\HP\\Desktop\\Meeting_2.mp4"
AUDIO_PATH = "audio.wav"




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
