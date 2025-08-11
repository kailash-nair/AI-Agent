"""Utilities for polishing transcripts into business English."""

from __future__ import annotations

import re
from typing import Dict

FILLER_WORDS = {"um", "ah"}

TERMINOLOGY_MAP: Dict[str, str] = {
    r"\bsetting up\b": "deployment",
    r"\bset up\b": "deploy",
}


def polish_business_english(text: str) -> str:
    """Return a polished business English version of ``text``.

    The function removes simple filler words, standardizes corporate
    terminology, collapses whitespace, and applies basic sentence
    capitalization and punctuation.
    """

    result = text

    # Remove filler words
    for filler in FILLER_WORDS:
        result = re.sub(rf"\b{filler}\b", "", result, flags=re.IGNORECASE)

    # Standardize terminology
    for pattern, replacement in TERMINOLOGY_MAP.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # Collapse whitespace
    result = re.sub(r"\s+", " ", result).strip()

    # Basic sentence handling
    sentences = re.split(r"(?<=[.!?]) +", result)
    polished_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if sentence[-1] not in ".!?":
            sentence += "."
        polished_sentences.append(sentence[0].upper() + sentence[1:])

    return " ".join(polished_sentences)
