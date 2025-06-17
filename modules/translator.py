from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained(
    "ai4bharat/indictrans2-indic-en-1B", trust_remote_code=True
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "ai4bharat/indictrans2-indic-en-1B", trust_remote_code=True
)


def translate_malayalam_to_english(text_ml: str) -> str:
    inputs = tokenizer(text_ml, return_tensors="pt", truncation=True, max_length=512)
    output_ids = model.generate(**inputs, max_length=512)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
