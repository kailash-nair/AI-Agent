import ffmpeg


def extract_audio_from_video(
    video_path: str, output_audio_path: str, sample_rate: int = 16000
):
    """Extract audio from a video file using ffmpeg."""
    print(f" Extracting audio from {video_path} using ffmpeg...")
    try:
        (
            ffmpeg.input(video_path)
            .output(output_audio_path, ar=sample_rate, ac=1, format="wav")
            .overwrite_output()
            .run(quiet=True)
        )
        print(f" Audio saved to {output_audio_path}")
    except ffmpeg.Error as e:
        print(" FFmpeg error:", e)
