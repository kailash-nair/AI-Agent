from modules.audio_extractor import extract_audio_from_video
from modules.audio_processor import transcribe_malayalam, split_audio
from modules.translator import translate_malayalam_to_english

# ─── CONFIG ───
VIDEO_PATH = r"C:\\Users\\HP\\Desktop\\Meeting_2.mp4"
AUDIO_PATH = "audio.wav"
CHUNK_DIR = "chunks"




def main():
    print("🎬 Extracting audio...")
    extract_audio_from_video(VIDEO_PATH, AUDIO_PATH)

    print("🔪 Splitting audio into chunks...")
    chunk_paths = split_audio(AUDIO_PATH, CHUNK_DIR, chunk_duration=30)

    for idx, chunk in enumerate(chunk_paths, start=1):
        print(f"\n🧠 Transcribing Malayalam (chunk {idx})...")
        ml_text = transcribe_malayalam(chunk)
        print("📝 Malayalam transcript:\n", ml_text)

        print("🌍 Translating to English...")
        en_text = translate_malayalam_to_english(ml_text)
        print("📝 English translation:\n", en_text)


if __name__ == "__main__":
    main()
