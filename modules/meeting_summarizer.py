"""Utilities for generating structured meeting summaries."""

from __future__ import annotations

from typing import List

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Use a lightweight instruction-tuned model for summarization.
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")


def summarize_meeting(transcript: str, purpose: str, date: str, attendees: List[str]) -> str:
    """Return a structured meeting summary for ``transcript``.

    The summary includes an introduction paragraph with ``purpose``, ``date``
    and ``attendees`` followed by issue-wise bullet points highlighting
    discussion points, decisions and action items.
    """

    intro = f"Meeting Purpose: {purpose}\nDate: {date}\nAttendees: {', '.join(attendees)}"
    prompt = (
        "You are an assistant that summarises meeting transcripts into structured reports.\n"
        f"{intro}\n"
        "Provide a structured summary. For each issue raised include:\n"
        "- discussion highlights\n- decisions made\n- action items with owners and deadlines.\n"
        "Use headings for each issue and bullet lists. Emphasise action items in **bold**.\n"
        f"Transcript:\n{transcript}"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    outputs = model.generate(**inputs, max_length=1024)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
