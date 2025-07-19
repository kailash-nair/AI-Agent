import os
from modules.audio_extractor import extract_audio_from_video
from modules.audio_processor import transcribe_malayalam, split_audio
from modules.translator import translate_malayalam_to_english
from modules.transcript_joiner import join_transcripts

# ─── CONFIG ───
VIDEO_PATH = r"C:\\Users\\HP\\Desktop\\Meeting_2.mp4"
AUDIO_PATH = "audio.wav"
CHUNK_DIR = "chunks"
TRANSCRIPT_DIR = "transcripts"




def main():
    print("🎬 Extracting audio...")
    extract_audio_from_video(VIDEO_PATH, AUDIO_PATH)

    print("🔪 Splitting audio into chunks...")
    chunk_paths = split_audio(AUDIO_PATH, CHUNK_DIR, chunk_duration=30)
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

    for idx, chunk in enumerate(chunk_paths, start=1):
        print(f"\n🧠 Transcribing Malayalam (chunk {idx})...")
        ml_text = transcribe_malayalam(chunk)
        print("📝 Malayalam transcript:\n", ml_text)

        print("🌍 Translating to English...")
        en_text = translate_malayalam_to_english(ml_text)
        print("📝 English translation:\n", en_text)

        # Save transcript to a text file for each chunk
        transcript_path = os.path.join(TRANSCRIPT_DIR, f"chunk_{idx}.txt")
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(ml_text)

    # After processing all chunks, join the individual transcripts
    final_transcript_path = os.path.join(TRANSCRIPT_DIR, "full_transcript.txt")
    join_transcripts(TRANSCRIPT_DIR, final_transcript_path)
    print(f"\n📄 Combined transcript saved to {final_transcript_path}")



if __name__ == "__main__":
    main()
