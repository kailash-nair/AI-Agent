"""Utility for translating Malayalam to English using IndicTrans2."""

# Models from :class:`transformers` are heavy to load and require several
# optional dependencies. Attempting to import and initialize them when this
# module is imported can therefore raise errors (e.g. ``ImportError`` if
# ``transformers`` is missing) or trigger large downloads.  To make the
# ``translate_malayalam_to_english`` function importable in lightweight
# environments, the model is loaded lazily when first needed.

tokenizer = None
model = None


def _load_model() -> None:
    """Load the IndicTrans2 model and tokenizer if not already loaded."""
    global tokenizer, model
    if tokenizer is None or model is None:
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        except ImportError as exc:
            raise ImportError(
                "The 'transformers' package is required for translation."
            ) from exc

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

    _load_model()

    text_with_lang = f"mal_Mlym eng_Latn {text_ml}"
    inputs = tokenizer(text_with_lang, return_tensors="pt", truncation=True, max_length=512)
    output_ids = model.generate(**inputs, max_length=512)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
