"""Utility for translating Malayalam to English using IndicTrans2."""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained(
    "ai4bharat/indictrans2-indic-en-1B", trust_remote_code=True
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "ai4bharat/indictrans2-indic-en-1B", trust_remote_code=True
)


def translate_malayalam_to_english(text_ml: str) -> str:
    """Translate Malayalam text to English using IndicTrans2.

    The tokenizer for IndicTrans2 expects the input string to contain the
    source language code, target language code, and the text separated by
    spaces. We prepend the codes ``mal_Mlym`` and ``eng_Latn`` before the
    actual text to form a valid input.
    """

    text_with_lang = f"mal_Mlym eng_Latn {text_ml}"
    inputs = tokenizer(text_with_lang, return_tensors="pt", truncation=True, max_length=512)
    output_ids = model.generate(**inputs, max_length=512)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
