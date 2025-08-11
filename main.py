import os
from modules.audio_extractor import extract_audio_from_video
from modules.audio_processor import transcribe_malayalam, split_audio
from modules.translator import translate_malayalam_to_english
from modules.transcript_joiner import join_transcripts
from modules.text_polisher import polish_business_english
from modules.meeting_summarizer import summarize_meeting

# ─── CONFIG ───
VIDEO_PATH = r"C:\\Users\\HP\\Desktop\\Meeting_2.mp4"
AUDIO_PATH = "audio.wav"
CHUNK_DIR = "chunks"
TRANSCRIPT_DIR = "transcripts"
EN_TRANSCRIPT_DIR = "transcripts_en"

# Metadata for the meeting summary
MEETING_DATE = "2024-01-01"
MEETING_PURPOSE = "Operations coordination meeting"
ATTENDEES = [
    "Installation",
    "Production",
    "Design",
    "Estimation",
    "Quality",
    "Logistics",
    "Stores",
]
SUMMARY_PATH = "meeting_summary.txt"




def main():
    print("🎬 Extracting audio...")
    extract_audio_from_video(VIDEO_PATH, AUDIO_PATH)

    print("🔪 Splitting audio into chunks...")
    chunk_paths = split_audio(AUDIO_PATH, CHUNK_DIR, chunk_duration=30)
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    os.makedirs(EN_TRANSCRIPT_DIR, exist_ok=True)

    for idx, chunk in enumerate(chunk_paths, start=1):
        print(f"\n🧠 Transcribing Malayalam (chunk {idx})...")
        ml_text = transcribe_malayalam(chunk)
        print("📝 Malayalam transcript:\n", ml_text)

        print("🌍 Translating to English...")
        en_text = translate_malayalam_to_english(ml_text)
        polished_text = polish_business_english(en_text)
        print("📝 Polished English translation:\n", polished_text)

        # Save transcripts to text files for each chunk
        transcript_path = os.path.join(TRANSCRIPT_DIR, f"chunk_{idx}.txt")
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(ml_text)

        en_transcript_path = os.path.join(EN_TRANSCRIPT_DIR, f"chunk_{idx}.txt")
        with open(en_transcript_path, "w", encoding="utf-8") as f:
            f.write(polished_text)

    # After processing all chunks, join the individual transcripts
    final_transcript_path = os.path.join(TRANSCRIPT_DIR, "full_transcript.txt")
    join_transcripts(TRANSCRIPT_DIR, final_transcript_path)

    final_en_transcript_path = os.path.join(
        EN_TRANSCRIPT_DIR, "full_transcript_polished.txt"
    )
    join_transcripts(EN_TRANSCRIPT_DIR, final_en_transcript_path)

    print(f"\n📄 Combined Malayalam transcript saved to {final_transcript_path}")
    print(
        f"📄 Combined polished English transcript saved to {final_en_transcript_path}"
    )

    # Generate structured meeting summary from the polished transcript
    with open(final_en_transcript_path, "r", encoding="utf-8") as f:
        full_transcript_en = f.read()

    summary = summarize_meeting(
        full_transcript_en,
        purpose=MEETING_PURPOSE,
        date=MEETING_DATE,
        attendees=ATTENDEES,
    )

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"📄 Structured meeting summary saved to {SUMMARY_PATH}")



if __name__ == "__main__":
    main()
