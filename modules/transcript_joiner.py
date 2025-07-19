"""Utility for joining multiple transcript files into a single file."""

import glob
import os
from typing import List


def join_transcripts(transcript_dir: str, output_file: str) -> str:
    """Join ``*.txt`` files from ``transcript_dir`` and write to ``output_file``.

    The files are joined in lexicographic order to match the order
    of processed audio chunks. The combined transcript string is
    returned.
    """

    pattern = os.path.join(transcript_dir, "*.txt")
    file_paths: List[str] = sorted(glob.glob(pattern))
    combined_lines: List[str] = []
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            combined_lines.append(f.read().strip())
    combined_text = "\n".join(combined_lines)
    with open(output_file, "w", encoding="utf-8") as out:
        out.write(combined_text)
    return combined_text
