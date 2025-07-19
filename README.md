# AI-Agent

This project extracts audio from a video, transcribes Malayalam speech using a Wav2Vec2 ASR model, and translates the transcript to English using `indictrans2`.

Dependencies:

- `transformers`
- `torchaudio`
- `ffmpeg` (for audio extraction)
- `torch` with CPU support

Run `python main.py` after installing the dependencies. Transcripts for each
audio chunk will be saved as text files in the `transcripts` directory. After
all chunks are processed, the individual transcripts are automatically joined
into `transcripts/full_transcript.txt`.
