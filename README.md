# AI-Agent

This project extracts audio from a video, transcribes Malayalam speech using a Wav2Vec2 ASR model, and translates the transcript to English using `indictrans2`. The English text is then polished into formal business language by removing filler words and standardizing corporate terminology.

Dependencies:

- `transformers`
- `torchaudio`
- `ffmpeg` (for audio extraction)
- `torch` with CPU support

Run `python main.py` after installing the dependencies. Malayalam transcripts for each audio chunk are saved as text files in the `transcripts` directory. Polished English translations are saved to `transcripts_en`. After all chunks are processed, the individual transcripts are automatically joined into `transcripts/full_transcript.txt` and `transcripts_en/full_transcript_polished.txt`.
